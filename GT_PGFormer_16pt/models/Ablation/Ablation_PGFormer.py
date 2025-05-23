import math
import logging
from functools import partial,reduce
from collections import OrderedDict
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from models.modulated_gcn_conv import ModulatedGraphConv
from models.modulated_gcn import _GraphConv,_ResGraphConv
from models.ChebConv import _ResChebGC,ChebConv

import matplotlib.pyplot as plt
import seaborn as sns

class _GraphChebConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphChebConv, self).__init__()

        self.adj = adj
        self.gconv = ChebConv(input_dim, output_dim, K=2)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x, self.adj)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LAM_Gconv(nn.Module):

    def __init__(self, in_features, out_features, activation=nn.ReLU(inplace=True)):
        super(LAM_Gconv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = activation

    def laplacian(self, A_hat):
        D_hat = (torch.sum(A_hat, 0) + 1e-5) ** (-0.5)
        L = D_hat * A_hat * D_hat
        return L

    def laplacian_batch(self, A_hat):
        batch, N = A_hat.shape[:2]
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, X, A):
        batch = X.size(0)
        A_hat = A.unsqueeze(0).repeat(batch, 1, 1)
        X = self.fc(torch.bmm(self.laplacian_batch(A_hat), X))
        if self.activation is not None:
            X = self.activation(X)
        return X

#replace MLP
class GraphNet(nn.Module):

    def __init__(self, in_features=2, out_features=None, n_pts=16):
        super(GraphNet, self).__init__()
        out_features = out_features or in_features

        self.A_hat = nn.Parameter(torch.eye(n_pts).float(), requires_grad=True)
        self.gconv1 = LAM_Gconv(in_features, in_features * 2)
        self.gconv2 = LAM_Gconv(in_features * 2, out_features, activation=None)

    def forward(self, X):
        X_0 = self.gconv1(X, self.A_hat)
        X_1 = self.gconv2(X_0, self.A_hat)
        return X_1

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PoolingAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.local_pooling_group_1 = [2, 3, 5, 6, 1, 4, 0, 7, 8, 9, 14, 15, 11, 12, 10, 13]
        self.local_pooling_group_2 = [0, 1, 2, 3, 5, 6, 4, 7]
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        print("q shape",q.shape)

        pools = []
        pools.append(x.clone().transpose(1, 2))
        pool = self.pool(x[:, self.local_pooling_group_1].transpose(1, 2))
        print("p1 shape",pool.shape)
        pools.append(pool)
        pool = self.pool(x[:, self.local_pooling_group_2].transpose(1, 2))
        print("p2 shape", pool.shape)
        pools.append(pool)

        pools = torch.cat(pools, dim=2)
        print("p3 shape", pools.shape)
        pools = self.norm(pools.permute(0, 2, 1))

        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        print("kv shape", k.shape)
        print('k tran',k.transpose(-2, -1).shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        print('attn shape',attn.shape)
        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        print('x shape', x.shape)
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)
        print('x shape', x.shape)

        x = self.proj(x)

        return x

class SpatialPoolingAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.local_pooling_group_1 = [2, 3, 5, 6, 1, 4, 0, 7, 8, 9, 14, 15, 11, 12, 10, 13]
        self.local_pooling_group_2 = [0, 1, 2, 3, 5, 6, 4, 7]
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm = nn.LayerNorm(dim)

        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, x):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        pools = []
        pools.append(x.clone().transpose(1, 2))
        pool = self.pool(x[:, self.local_pooling_group_1].transpose(1, 2))
        pools.append(pool)
        pool = self.pool(x[:, self.local_pooling_group_2].transpose(1, 2))
        pools.append(pool)

        pools = torch.cat(pools, dim=2)
        pools = self.norm(pools.permute(0, 2, 1))

        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)

        x = self.proj(x)

        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PyramaidBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PoolingAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PyramaidGBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PoolingAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = GraphNet(in_features=dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PyramaidGBSlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PoolingAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = GraphNet(in_features=dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

#Transformer改成block
class former_4(nn.Module):
    def __init__(self, adjMatrix, hid_dim=64, coords_dim=(2, 3), num_layers=2, p_dropout=None, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        super(former_4, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        _gconv_input = [_GraphConv(adjMatrix, coords_dim[0], hid_dim, p_dropout=p_dropout)]
        _gconv_layers = []
        _gconv_output = [ModulatedGraphConv(hid_dim, coords_dim[1], adjMatrix)]

        for i in range(num_layers):
            _gconv_layers.append(_ResGraphConv(adjMatrix, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
            _gconv_layers.append(Block(
                    dim=hid_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer))

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output =nn.Sequential(*_gconv_output)


    def forward(self, x):
        out = self.gconv_input(x)
        out = self.gconv_layers(out)
        out = self.gconv_output(out)
        return out

class former_5(nn.Module):
    def __init__(self, adjMatrix, hid_dim=64, coords_dim=(2, 3), num_layers=2, p_dropout=None, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        super(former_5, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        _gconv_input = [_GraphConv(adjMatrix, coords_dim[0], hid_dim, p_dropout=p_dropout)]
        _gconv_layers = []
        _gconv_output = [ModulatedGraphConv(hid_dim, coords_dim[1], adjMatrix)]

        for i in range(num_layers):
            _gconv_layers.append(_ResGraphConv(adjMatrix, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
            _gconv_layers.append(PyramaidGBlock(
                    dim=hid_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer))

        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output =nn.Sequential(*_gconv_output)


    def forward(self, x):
        out = self.gconv_input(x)
        out = self.gconv_layers(out)
        out = self.gconv_output(out)
        return out

class semGPFormer(nn.Module):
    def __init__(self, adjMatrix, hid_dim=64, coords_dim=(2, 3), num_layers=2, p_dropout=None, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        super(former_6, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        _gconv_layers = []

        for i in range(num_layers):
            _gconv_layers.append(_ResChebGC(adj=adjMatrix, input_dim=hid_dim, output_dim=hid_dim,hid_dim=hid_dim, p_dropout=p_dropout))
            _gconv_layers.append(PyramaidGBlock(
                    dim=hid_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer))

        self.gconv_input = _GraphChebConv(adj=adjMatrix,input_dim=coords_dim[0], output_dim=hid_dim, p_dropout=p_dropout)
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output = _GraphChebConv(adj=adjMatrix,input_dim=hid_dim, output_dim=coords_dim[1], p_dropout=p_dropout)


    def forward(self, x):
        out = self.gconv_input(x)
        out = self.gconv_layers(out)
        out = self.gconv_output(x)
        return out