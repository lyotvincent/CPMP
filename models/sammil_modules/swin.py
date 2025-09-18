import copy
import torch
import numpy as np
from torch import nn
from einops import repeat
import torchvision.models as models
from models.sammil_modules.emb_position import *
from models.sammil_modules.transformer import *
from models.sammil_modules.mlp import *
from models.sammil_modules.datten import *
import torch.nn.functional as F
from models.sammil_modules.translayer import *
from models.sammil_modules.nystrom_attention import NystromAttention

import math

import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import sys

from models.sammil_modules.utils import cosine_scheduler


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # ref from meituan
            # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def initialize_weights_noic(module):
    for m in module.modules():
        # if isinstance(m, nn.Conv2d):
        #     # ref from huggingface
        #         m.weight.data.normal_(0, 0.02)
        #     #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     if m.bias is not None:
        #         m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class SwinEncoder(nn.Module):
    '''
    learnable mask token
    '''

    def __init__(self, mlp_dim=512, pos_pos=0, pos='none', peg_k=7, attn='ntrans', swin_window_num=8, drop_out=0.1,
                 n_layers=2, n_heads=8, multi_scale=False, drop_path=0, pool='attn', da_act='tanh', reduce_ratio=0,
                 ffn=False, ffn_act='gelu', mlp_ratio=4., da_gated=False, da_bias=False, da_dropout=False, trans_dim=64,
                 n_cycle=1, trans_conv=True, rpe=False, window_size=0, min_win_num=0, min_win_ratio=0, qkv_bias=True,
                 shift_size=False, peg_bias=True, peg_1d=False, init=False, l2_n_heads=8, moe_fl_enable=False,
                 **kwargs):
        super(SwinEncoder, self).__init__()

        # 不需要降维
        if reduce_ratio == 0:
            pass
        # 根据多尺度自动降维，维度前后不变
        elif reduce_ratio == -1:
            reduce_ratio = n_layers - 1
        # 指定降维维度 (2的n次)
        else:
            pass

        self.final_dim = mlp_dim // (2 ** reduce_ratio) if reduce_ratio > 0 else mlp_dim
        if multi_scale:
            self.final_dim = self.final_dim * (2 ** (n_layers - 1))

        self.pool = pool
        if pool == 'cls_token':
            self.cls_token = nn.Parameter(torch.randn(1, 1, mlp_dim))
            nn.init.normal_(self.cls_token, std=1e-6)
        elif pool == 'attn':
            self.pool_fn = DAttention(self.final_dim, da_act, gated=da_gated, bias=da_bias, dropout=da_dropout)

        self.norm = nn.LayerNorm(self.final_dim)

        # l2_shortcut = kwargs.pop('l2_out_shortcut')
        # self.all_shortcut = kwargs.pop('all_shortcut')
        # l2_sc_ratio = kwargs.pop('l2_sc_ratio')
        # l2_conv = kwargs.pop('l2_conv_k')
        l2_shortcut = True
        self.all_shortcut = kwargs.pop('all_shortcut')
        l2_sc_ratio = 0
        l2_conv = 0
        if l2_conv == 0:
            l2_conv_k = 0
            l2_conv = False
        else:
            l2_conv_k = l2_conv
            l2_conv = True

        # if attn == 'trans':
        self.layer1 = TransLayer1(dim=mlp_dim, head=n_heads, drop_out=drop_out, drop_path=drop_path,
                                  need_down=multi_scale, need_reduce=reduce_ratio != 0, down_ratio=2 ** reduce_ratio,
                                  ffn=ffn, ffn_act=ffn_act, mlp_ratio=mlp_ratio, trans_dim=trans_dim, n_cycle=n_cycle,
                                  attn=attn, n_window=swin_window_num, trans_conv=trans_conv, rpe=rpe,
                                  window_size=window_size, min_win_num=min_win_num, min_win_ratio=min_win_ratio,
                                  qkv_bias=qkv_bias, shift_size=shift_size, moe_enable=moe_fl_enable, fl=True,
                                  shortcut=True, **kwargs)
        # self.layer2 = TransLayer1(dim=mlp_dim,head=n_heads,drop_out=drop_out,drop_path=drop_path)
        if reduce_ratio > 0:
            mlp_dim = mlp_dim // (2 ** reduce_ratio)

        if multi_scale:
            mlp_dim = mlp_dim * 2

        if n_layers >= 2:
            self.layers = []
            for i in range(n_layers - 2):
                self.layers += [TransLayer1(dim=mlp_dim, head=n_heads, drop_out=drop_out, drop_path=drop_path,
                                            need_down=multi_scale, ffn=ffn, ffn_act=ffn_act, mlp_ratio=mlp_ratio,
                                            trans_dim=trans_dim, n_cycle=n_cycle, attn=attn, n_window=swin_window_num,
                                            trans_conv=trans_conv, rpe=rpe, window_size=window_size,
                                            min_win_num=min_win_num, min_win_ratio=min_win_ratio, qkv_bias=qkv_bias)]
                if multi_scale:
                    mlp_dim = mlp_dim * 2
            # self.layers += [TransLayer1(dim=mlp_dim,head=n_heads,drop_out=drop_out,drop_path=drop_path,ffn=ffn,ffn_act=ffn_act,mlp_ratio=mlp_ratio,trans_dim=trans_dim,n_cycle=n_cycle,attn=attn,n_window=swin_window_num,trans_conv=trans_conv,rpe=rpe,window_size=window_size,min_win_num=min_win_num,min_win_ratio=min_win_ratio,qkv_bias=qkv_bias,shift_size=shift_size,**kwargs)]
            kwargs.pop('conv_k')
            self.layers += [
                TransLayer1(dim=mlp_dim, head=l2_n_heads, drop_out=drop_out, drop_path=drop_path, need_down=multi_scale,
                            need_reduce=reduce_ratio != 0, down_ratio=2 ** reduce_ratio, ffn=ffn, ffn_act=ffn_act,
                            mlp_ratio=mlp_ratio, trans_dim=trans_dim, n_cycle=n_cycle, attn=attn,
                            n_window=swin_window_num, rpe=rpe, window_size=window_size, min_win_num=min_win_num,
                            min_win_ratio=min_win_ratio, qkv_bias=qkv_bias, shift_size=shift_size, moe_enable=True,
                            trans_conv=l2_conv, conv_k=l2_conv_k, shortcut=l2_shortcut, sc_ratio=l2_sc_ratio, **kwargs)]
            self.layers = nn.Sequential(*self.layers)
        else:
            self.layers = nn.Identity()

        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=mlp_dim, k=peg_k, bias=peg_bias, conv_1d=peg_1d)
        elif pos == 'sincos':
            self.pos_embedding = SINCOS(embed_dim=mlp_dim)
        elif pos == 'peg':
            self.pos_embedding = PEG(mlp_dim, k=peg_k, bias=peg_bias, conv_1d=peg_1d)
        else:
            self.pos_embedding = nn.Identity()

        self.pos_pos = pos_pos

        if init:
            self.apply(initialize_weights)

    def forward(self, x, no_pool=False, return_trans_attn=False, return_attn=False, no_norm=False):
        shape_len = 3
        # for N,C
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            shape_len = 2
        # for B,C,H,W
        if len(x.shape) == 4:
            x = x.reshape(x.size(0), x.size(1), -1)
            x = x.transpose(1, 2)
            shape_len = 4
        batch, num_patches, C = x.shape  # 直接是特征
        x_shortcut = x
        patch_idx = 0
        if self.pos_pos == -2:
            x = self.pos_embedding(x)

        # cls_token
        if self.pool == 'cls_token':
            cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=batch)
            x = torch.cat((cls_tokens, x), dim=1)
            patch_idx = 1

        if self.pos_pos == -1:
            x[:, patch_idx:, :] = self.pos_embedding(x[:, patch_idx:, :])

        # translayer1
        if return_trans_attn:
            x, trans_attn = self.layer1(x, True)
        else:
            x = self.layer1(x)
            trans_attn = None
        # print(x)
        # 加位置编码
        if self.pos_pos == 0:
            x[:, patch_idx:, :] = self.pos_embedding(x[:, patch_idx:, :])

        # translayer2
        for i, layer in enumerate(self.layers.children()):
            if return_trans_attn:
                x, trans_attn = layer(x, return_trans_attn)
            else:
                x = layer(x)
                trans_attn = None

        if self.all_shortcut:
            x = x + x_shortcut

        # ---->cls_token
        x = self.norm(x)

        if no_pool or self.pool == 'none':
            if shape_len == 2:
                x = x.squeeze(0)
            elif shape_len == 4:
                x = x.transpose(1, 2)
                x = x.reshape(batch, C, int(num_patches ** 0.5), int(num_patches ** 0.5))
            return x

        if self.pool == 'cls_token':
            logits = x[:, 0, :]
        elif self.pool == 'avg':
            logits = x.mean(dim=1)
        elif self.pool == 'attn':
            if return_attn:
                logits, a, _ = self.pool_fn(x, return_attn=True, no_norm=no_norm)
            else:
                logits, _, _ = self.pool_fn(x)

        else:
            logits = x

        if shape_len == 2:
            logits = logits.squeeze(0)
        elif shape_len == 4:
            logits = logits.transpose(1, 2)
            logits = logits.reshape(batch, C, int(num_patches ** 0.5), int(num_patches ** 0.5))

        if return_attn:
            return logits, a, trans_attn

        else:
            return logits, trans_attn


class Swin(nn.Module):
    def __init__(self, input_dim=1024, mlp_dim=512, act='relu', n_classes=2, dropout=0.25, pos_pos=0, n_robust=0,
                 pos='ppeg', peg_k=7, attn='trans', pool='attn', swin_window_num=8, n_layers=2, n_heads=8,
                 multi_scale=False, drop_path=0., da_act='relu', trans_dropout=0.1, reduce_ratio=0, ffn=False,
                 ffn_act='gelu', ic=False, mlp_ratio=4., da_gated=False, da_bias=False, da_dropout=False, trans_dim=64,
                 n_cycle=1, trans_conv=False, rpe=False, window_size=0, min_win_num=0, min_win_ratio=0, qkv_bias=True,
                 shift_size=False, **kwargs):
        super(Swin, self).__init__()

        self.patch_to_emb = [nn.Linear(input_dim, 512)]

        if act.lower() == 'relu':
            self.patch_to_emb += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.patch_to_emb += [nn.GELU()]

        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)

        self.swin_window_num = swin_window_num

        self.online_encoder = SwinEncoder(mlp_dim=mlp_dim, pos_pos=pos_pos, pos=pos, peg_k=peg_k, attn=attn,
                                          swin_window_num=swin_window_num, n_layers=n_layers, n_heads=n_heads,
                                          multi_scale=multi_scale, drop_path=drop_path, pool=pool, da_act=da_act,
                                          drop_out=trans_dropout, reduce_ratio=reduce_ratio, ffn=ffn, ffn_act=ffn_act,
                                          mlp_ratio=mlp_ratio, da_gated=da_gated, da_bias=da_bias,
                                          da_dropout=da_dropout, trans_dim=trans_dim, n_cycle=n_cycle,
                                          trans_conv=trans_conv, rpe=rpe, window_size=window_size,
                                          min_win_num=min_win_num, min_win_ratio=min_win_ratio, qkv_bias=qkv_bias,
                                          shift_size=shift_size, **kwargs)

        self.predictor = nn.Linear(self.online_encoder.final_dim, n_classes)

        if ic:
            self.apply(initialize_weights)
        else:
            self.apply(initialize_weights_noic)

        if n_robust > 0:
            # 改变init
            rng = torch.random.get_rng_state()
            torch.random.manual_seed(n_robust)
            if ic:
                self.apply(initialize_weights)
            else:
                self.apply(initialize_weights_noic)
            torch.random.set_rng_state(rng)
            # 改变后续的随机状态
            [torch.rand((1024, 512)) for i in range(n_robust)]

    def crop(self, x):
        B, L, C = x.shape
        H, W = int(np.floor(np.sqrt(L))), int(np.floor(np.sqrt(L)))
        _n = H % self.swin_window_num
        H, W = H - _n, W - _n
        crop_length = L - H * W
        if crop_length > 0:
            if self.training:
                _indices = list(range(L))
                np.random.shuffle(_indices)
                _indices = _indices[:crop_length]
            else:
                _indices = np.linspace(0, L - 1, crop_length, dtype=int)
            _mask = torch.ones((B, L, C), device=x.device)
            _mask.scatter_(1, torch.tensor(_indices, device=x.device).unsqueeze(0).unsqueeze(-1).repeat((B, 1, C)), 0)
            _mask = _mask == 1
            x = x[_mask].view(1, -1, C)
        return x

    def forward(self, x, return_attn=False, no_norm=False, return_trans_attn=False):
        x = self.patch_to_emb(x)  # n*512
        x = self.dp(x)

        ps = x.size(1)

        # forward online network
        if return_attn:
            x, a, t_a = self.online_encoder(x, return_attn=True, no_norm=no_norm, return_trans_attn=return_trans_attn)
        else:
            x, t_a = self.online_encoder(x, return_trans_attn=return_trans_attn)

        # prediction
        logits = self.predictor(x)

        if return_attn:
            if return_trans_attn:
                return logits, a, t_a
            else:
                return logits, a
        else:
            if return_trans_attn:
                return logits, t_a
            else:
                return logits




# --------------------------------------------------------
# Fused kernel for window process for SwinTransformer
# Copyright (c) 2022 Nvidia
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")


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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, head_dim=None, window_size=None, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0., conv=True, conv_k=15, conv_2d=False, conv_bias=True, conv_type='attn'):

        super().__init__()
        self.dim = dim
        self.window_size = [window_size, window_size] if window_size is not None else None  # Wh, Ww
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        if window_size is not None:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                            num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, head_dim * num_heads * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.conv_2d = conv_2d
        self.conv_type = conv_type
        if conv:
            kernel_size = conv_k
            padding = kernel_size // 2

            if conv_2d:
                if conv_type == 'attn':
                    self.pe = nn.Conv2d(num_heads, num_heads, kernel_size, padding=padding, groups=num_heads,
                                        bias=conv_bias)
                elif conv_type == 'newattn':
                    # self.pe = nn.Conv2d(head_dim * num_heads, head_dim * num_heads, kernel_size, padding = padding, groups = head_dim * num_heads, bias = conv_bias)
                    self.pe = nn.Conv2d(head_dim * num_heads, num_heads, kernel_size, padding=padding, groups=num_heads,
                                        bias=conv_bias)
                    # self.pe_mlp = nn.Sequential(nn.Linear(head_dim * num_heads, num_heads, bias=False))
                else:
                    self.pe = nn.Conv2d(head_dim * num_heads, head_dim * num_heads, kernel_size, padding=padding,
                                        groups=head_dim * num_heads, bias=conv_bias)
            else:
                if conv_type == 'attn':
                    self.pe = nn.Conv2d(num_heads, num_heads, (kernel_size, 1), padding=(padding, 0), groups=num_heads,
                                        bias=conv_bias)
                    # self.pe = nn.Conv2d(num_heads, num_heads, (1,kernel_size), padding = (0,padding), groups = num_heads, bias = conv_bias)
                elif conv_type == 'newattn':
                    self.pe = nn.Conv2d(head_dim * num_heads, num_heads, (kernel_size, 1), padding=(padding, 0),
                                        groups=num_heads, bias=conv_bias)
                    # self.pe_mlp = nn.Sequential(nn.Linear(head_dim * num_heads, num_heads, bias=False))
                else:
                    self.pe = nn.Conv2d(head_dim * num_heads, head_dim * num_heads, (kernel_size, 1),
                                        padding=(padding, 0), groups=head_dim * num_heads, bias=conv_bias)
                # self.pe = nn.Conv2d(num_heads, num_heads, (1, kernel_size), padding = (0, padding), groups = num_heads, bias = conv_bias)
        else:
            self.pe = None

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, return_attn=False, no_pe=False):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        L = int(np.ceil(np.sqrt(N)))
        # x = self.pe(x)

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.pe is not None and not no_pe:
            if self.conv_type == 'attn':
                # print(attn.size())
                # B,H,N,N ->B,H,N,N-0.5,N-0.5
                # if self.conv_2d:
                #     pe = self.pe(attn.permute(0,2,1,3).reshape(-1,self.num_heads,int(np.ceil(np.sqrt(N))),int(np.ceil(np.sqrt(N)))))
                #     attn = attn+pe.reshape(B_,N,self.num_heads,N).transpose(1,2)
                # else:
                # print(attn[0,0])
                pe = self.pe(attn)
                attn = attn + pe
            # todo:newattn
            elif self.conv_type == 'newattn':
                # attn: B, H, N, N      x: B, N, C
                if self.conv_2d:
                    pe = self.pe(x.transpose(-1, -2).reshape(-1, C, L, L)).reshape(-1, self.num_heads, N)  # B, H, N
                else:
                    pe = self.pe(x.transpose(-1, -2).unsqueeze(-1)).reshape(-1, self.num_heads, N)
                # pe = self.pe_mlp(pe.transpose(-1,-2)).transpose(-1,-2) # B, H, N
                pe = (pe.unsqueeze(-1) @ pe.unsqueeze(-1).transpose(-2, -1))
                attn = attn + pe

        if self.window_size is not None:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        if self.pe is not None and self.conv_type == 'value_bf':
            # B,H,N,C -> B,HC,N-0.5,N-0.5 
            pe = self.pe(v.permute(0, 3, 1, 2).reshape(B_, C, int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))))
            # pe = torch.einsum('ahbd->abhd',pe).flatten(-2,-1)
            v = v + pe.reshape(B_, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        # print(v.size())

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.num_heads * self.head_dim)

        if self.pe is not None and self.conv_type == 'value_af':
            # print(v.size())
            pe = self.pe(v.permute(0, 3, 1, 2).reshape(B_, C, int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))))
            # print(pe.size())
            # print(v.size())
            x = x + pe.reshape(B_, self.num_heads * self.head_dim, N).transpose(-1, -2)

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn:
            return x, attn
        else:
            return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinAttntion(nn.Module):
    def __init__(self, dim, input_resolution=None, head_dim=None, num_heads=8, window_size=0, shift_size=False,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., window_num=8, conv=False, rpe=False,
                 min_win_num=0, min_win_ratio=0., glob_pe='none', win_attn='native', moe_enable=False, fl=False,
                 moe_k=1, wandb=None, l1_shortcut=True, no_weight_to_all=False, minmax_weight=True, moe_mask_diag=False,
                 mask_diag=False, moe_mlp=False, moe_norm=False, moe_mlp_act='tanh', **kawrgs):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size if window_size > 0 else None
        self.shift_size = shift_size
        self.window_num = window_num
        self.min_win_num = min_win_num
        self.min_win_ratio = min_win_ratio
        self.rpe = rpe

        if glob_pe == 'sincos':
            self.pe = SINCOS()
        # elif glob_pe == 'rpe':
        #     self.pe = RPE()
        elif glob_pe == 'ape':
            self.pe = APE()
        else:
            self.pe = nn.Identity()

        if self.window_size is not None:
            self.window_num = None
        self.fused_window_process = False

        if win_attn == 'native':
            self.attn = WindowAttention(
                dim, head_dim=head_dim, num_heads=num_heads, window_size=self.window_size if rpe else None,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, conv=conv, **kawrgs)
        elif win_attn == 'ntrans':
            self.attn = NystromAttention(
                dim=dim,
                dim_head=head_dim,
                heads=num_heads,
                dropout=drop
            )
        self.wandb = wandb
        self.fl = fl
        self.l1_shortcut = l1_shortcut
        self.no_weight_to_all = no_weight_to_all
        self.minmax_weight = minmax_weight
        self.moe_mask_diag = moe_mask_diag
        self.mask_diag = mask_diag
        self.moe_mlp = moe_mlp
        if moe_enable:
            self.norm = nn.LayerNorm(self.dim) if moe_norm else nn.Identity()
            if moe_mlp:
                self.phi = [nn.Linear(self.dim, self.dim // 4, bias=False)]
                if moe_mlp_act == 'tanh':
                    self.phi += [nn.Tanh()]
                elif moe_mlp_act == 'relu':
                    self.phi += [nn.ReLU()]
                elif moe_mlp_act == 'gelu':
                    self.phi += [nn.GELU()]
                self.phi += [nn.Linear(self.dim // 4, moe_k, bias=False)]
                self.phi = nn.Sequential(*self.phi)
            else:
                self.phi = nn.Parameter(
                    torch.empty(
                        (self.dim, moe_k),
                    )
                )
                nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))
        else:
            self.phi = None

        # if self.shift_size > 0:
        #     # calculate attention mask for SW-MSA
        #     H, W = self.input_resolution
        #     img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        #     h_slices = (slice(0, -self.window_size),
        #                 slice(-self.window_size, -self.shift_size),
        #                 slice(-self.shift_size, None))
        #     w_slices = (slice(0, -self.window_size),
        #                 slice(-self.window_size, -self.shift_size),
        #                 slice(-self.shift_size, None))
        #     cnt = 0
        #     for h in h_slices:
        #         for w in w_slices:
        #             img_mask[:, h, w, :] = cnt
        #             cnt += 1

        #     mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        #     mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        #     attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        #     attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        # else:
        self.attn_mask = None

    def padding_new(self, x):
        B, L, C = x.shape
        # padding to square
        H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
        _n = -H % 2
        H, W = H + _n, W + _n
        _add_length = H * W - L
        if _add_length > 0:
            x = torch.cat([x, torch.zeros((B, _add_length, C), device=x.device)], dim=1)
            mask = torch.cat(
                [torch.zeros((B, L, 1), device=x.device), torch.ones((B, _add_length, 1), device=x.device)], dim=1)
        else:
            mask = torch.zeros((B, L, 1))
        x = x.view(B, H, W, C)
        mask = mask.view(B, H, W, 1)
        # padding to window
        _n = -H % self.window_num
        H, W = H + _n, W + _n
        window_size = int(H // self.window_num)
        window_num = self.window_num
        window_pad = (0, 0, (_n // 2) + (_n % 2), _n // 2, (_n // 2) + (_n % 2), _n // 2)
        x = torch.nn.functional.pad(x, window_pad, "constant", 0).view(B, -1, C)

        mask = torch.nn.functional.pad(mask, window_pad, "constant", 1).view(B, -1, 1)
        mask = window_partition(mask.view(B, H, W, 1), window_size)  # nW, window_size, window_size, 1
        mask = mask.view(-1, window_size * window_size)
        mask = mask.unsqueeze(1) + mask.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, float(-100.0)).masked_fill(mask == 0, float(0.0))

        return x, H, W, _add_length, window_pad, window_num, window_size, mask

    def standar_input(self, x):
        B, L, C = x.shape
        H, W = int(np.floor(np.sqrt(L))), int(np.floor(np.sqrt(L)))
        _n = H % self.window_num
        H, W = H - _n, W - _n
        window_size = int(H // self.window_num)
        window_num = self.window_num
        crop_length = L - H * W
        if crop_length > 0:
            # if self.training:
            #     _indices = list(range(L))
            #     np.random.shuffle(_indices)
            #     _indices = _indices[:crop_length]
            # else:
            #     _indices = np.linspace(0, L-1, crop_length, dtype=int)
            x_surplus = x[:, -crop_length:, :]
            x = x[:, :-crop_length, :]
        return x, x_surplus, H, W, crop_length, window_num, window_size

    def padding(self, x):
        B, L, C = x.shape
        if self.window_size is not None:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            # print(L)
            # print(H)
            _n = -H % self.window_size
            H, W = H + _n, W + _n
            window_num = int(H // self.window_size)
            window_size = self.window_size
        else:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.window_num
            H, W = H + _n, W + _n
            window_size = int(H // self.window_num)
            window_num = self.window_num

        add_length = H * W - L
        # self.wandb.log({
        #     'add_length_ratio': add_length / L,
        #     'add_length': add_length,
        #     'L': L
        # },commit=False)
        # print(add_length)
        # 如果要补的太多，就放弃window attention
        if (add_length > L / (self.min_win_ratio + 1e-8) or L < self.min_win_num) and not self.rpe:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H + _n, W + _n
            add_length = H * W - L
            window_size = H
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B, add_length, C), device=x.device)], dim=1)
            # mask = torch.cat([torch.zeros((B,L,1),device=x.device),torch.ones((B,add_length,1),device=x.device)],dim=1)
            # mask = window_partition(mask.view(B,H,W,1),window_size) # nW, window_size, window_size, 1
            # mask = mask.view(-1, window_size * window_size)
            # attn_mask = mask.unsqueeze(1) + mask.unsqueeze(2)
            # attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            # mask = mask.masked_fill(mask != 0, float(-100.0)).masked_fill(mask == 0, float(0.0)).unsqueeze(-1)
        # else:
        #     mask = torch.zeros((B,L,1),device=x.device)

        return x, H, W, add_length, window_num, window_size

    def forward(self, x, return_attn=False):
        B, L, C = x.shape
        shift_size = 0

        # padding
        x, H, W, add_length, window_num, window_size = self.padding(x)
        # x,x_surplus,H,W,partition_length,window_num,window_size = self.standar_input(x)
        # x,H,W,crop_length,window_num,window_size = self.crop(x)
        # x,H,W,add_length,window_pad,window_num,window_size,mask = self.padding_new(x)

        # assert L == H * W, "input feature has wrong size"

        # shuffle   self.shift_size and  
        # if self.training and self.shift_size and torch.rand(1) > 0.5:
        #     x,g_idx = patch_shuffle(x,10,return_g_idx=True)
        # else:
        #     g_idx = None

        x = x.view(B, H, W, C)

        # if self.shift_size:
        # shift_size = self.window_size // 2
        # else:
        #     shift_size = 0

        # cyclic shift
        if shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, window_size)  # nW*B, window_size, window_size, C

        # rand = torch.rand(x_windows.size(0), window_size*window_size,device=x_windows.device)
        # batch_rand_perm = rand.argsort(dim=1)
        # x_windows = torch.gather(x_windows.view(-1, window_size * window_size, C),dim=1,index=batch_rand_perm.unsqueeze(-1).repeat(1, 1, C)).view(-1,window_size,window_size,C)

        x_windows = self.pe(x_windows)

        x_windows = x_windows.view(-1, window_size * window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        if self.phi is None:
            attn_mask = torch.diag(
                torch.tensor([float(-100.0) for i in range(x_windows.size(-2))], device=x_windows.device)).unsqueeze(
                0).repeat((x_windows.size(0), 1, 1)) if self.mask_diag else None
            if return_attn:
                attn_windows, _attn = self.attn(x_windows, attn_mask, return_attn)  # nW*B, window_size*window_size, C
                dispatch_weights, combine_weights, dispatch_weights_1 = None, None, None
            else:
                attn_windows = self.attn(x_windows, attn_mask)  # nW*B, window_size*window_size, C
            # x_surplus = self.attn(x_surplus)
        # soft moe
        elif self.phi is not None and self.fl:
            attn_mask_moe = torch.diag(
                torch.tensor([float(-100.0) for i in range(x_windows.size(0))], device=x_windows.device)).unsqueeze(
                0) if self.moe_mask_diag else None
            attn_mask = torch.diag(
                torch.tensor([float(-100.0) for i in range(x_windows.size(-2))], device=x_windows.device)).unsqueeze(
                0).repeat((x_windows.size(0), 1, 1)) if self.mask_diag else None
            x_windows = self.attn(x_windows, attn_mask)  # nW*B, window_size*window_size, C
            if self.moe_mlp:
                logits = self.phi(x_windows).transpose(1, 2)  # W*B, sW, window_size*window_size
            else:
                logits = torch.einsum("w p c, c n -> w p n", x_windows, self.phi).transpose(1,
                                                                                            2)  # nW*B, sW, window_size*window_size
            sW = logits.size(1)
            dispatch_weights = logits.softmax(dim=-1)
            combine_weights = logits.softmax(dim=1)
            if self.minmax_weight:
                logits_min, _ = logits.min(dim=-1)
                logits_max, _ = logits.max(dim=-1)
                dispatch_weights_1 = (logits - logits_min.unsqueeze(-1)) / (
                            logits_max.unsqueeze(-1) - logits_min.unsqueeze(-1) + 1e-8)
            else:
                dispatch_weights_1 = dispatch_weights
            if self.wandb is not None:
                self.wandb.log({
                    "dispatch_max": dispatch_weights.max(),
                    "dispatch_min": dispatch_weights.min(),
                    "combine_max": combine_weights.max(),
                    "combine_min": combine_weights.min(),
                }, commit=False)
            attn_windows = torch.einsum("w p c, w n p -> w n p c", x_windows, dispatch_weights).sum(dim=-2).transpose(0,
                                                                                                                      1)  # sW, nW, C
            if return_attn:
                attn_windows, _attn = self.attn(attn_windows, attn_mask_moe, return_attn, no_pe=True)  # sW, nW, C
                attn_windows = attn_windows.transpose(0, 1)  # nW, sW, C
            else:
                attn_windows = self.attn(attn_windows, attn_mask_moe, no_pe=True).transpose(0, 1)  # nW, sW, C
            if self.no_weight_to_all:
                attn_windows = attn_windows.unsqueeze(-2).repeat(
                    (1, 1, window_size * window_size, 1))  # nW, sW, window_size*window_size, C
            else:
                attn_windows = torch.einsum("w n c, w n p -> w n p c", attn_windows,
                                            dispatch_weights_1)  # nW, sW, window_size*window_size, C
            # if sW > 1:
            attn_windows = torch.einsum("w n p c, w n p -> w n p c", attn_windows, combine_weights).sum(
                dim=1)  # nW, window_size*window_size, C
            # else:
            #     attn_windows = attn_windows.squeeze(1)
            if self.l1_shortcut:
                attn_windows = x_windows + attn_windows

        else:
            # old: 先对特征进行mean-pooling，然后再算logits
            # x_windows_avg = torch.nn.functional.adaptive_avg_pool1d(x_windows.transpose(-1,-2),1) # nW*B, C, 1
            # x_windows_avg = x_windows_avg.view(-1,B,C).permute(1,0,2)  # B, nW, C
            # logits = torch.einsum("b w c, c n -> b w n", x_windows_avg, self.phi) # B, nW, sW
            # 先算logits，再用maxpooling 或者 meanpooling
            # logits = torch.einsum("b w c, c n -> b w n", x_windows, self.phi) # nW*B, window_size*window_size, sW
            # #logits = self.phi(x_windows) # B, nW, sW
            # logits,_ = logits.max(dim=-2)
            # logits = logits.view(-1,B,logits.size(-1)).transpose(0,1) # B, nW, sW
            # # DA
            # #logits = self.attention(x_windows_avg) # B, nW, sW
            # dispatch_weights = logits.softmax(dim=1)
            # combine_weights = logits.softmax(dim=-1)
            # if self.wandb is not None:
            #     self.wandb.log({
            #             "dispatch_max":dispatch_weights.max(),
            #             "dispatch_min":dispatch_weights.min(),
            #             "combine_max":combine_weights.max(),
            #             "combine_min":combine_weights.min(),
            #         },commit=False)
            # x_windows_avg = torch.einsum("b w p c, b w n -> b n p c", x_windows.view(-1,B,window_size * window_size, C).transpose(0,1),dispatch_weights) 
            # x_windows_avg = self.attn(x_windows_avg.permute(1,0,2,3).view(-1,window_size*window_size,C)) # sW*B, window_size*window_size, C
            # attn_windows = torch.einsum("b w p c, b w n -> b w n p c",x_windows.view(-1,B,window_size*window_size,C).transpose(0,1),1-dispatch_weights) 
            # attn_windows = torch.einsum("b w n p c, b w n -> b w p c",attn_windows,combine_weights)
            # x_windows_avg = torch.einsum("b n p c, b w n -> b w n p c", x_windows_avg.view(-1,B,window_size*window_size,C).transpose(0,1), dispatch_weights)
            # x_windows_avg = torch.einsum("b w n p c, b w n -> b w p c",x_windows_avg,combine_weights)
            # attn_windows = (attn_windows + x_windows_avg).permute(1,0,2,3).view(-1, window_size * window_size, C)
            attn_mask = torch.diag(
                torch.tensor([float(-100.0) for i in range(x_windows.size(0))], device=x_windows.device)).unsqueeze(
                0) if self.moe_mask_diag else None
            if self.moe_mlp:
                logits = self.phi(x_windows).transpose(1, 2)  # W*B, sW, window_size*window_size
            else:
                logits = torch.einsum("w p c, c n -> w p n", x_windows, self.phi).transpose(1,
                                                                                            2)  # nW*B, sW, window_size*window_size
            # sW = logits.size(1)
            dispatch_weights = logits.softmax(dim=-1)
            combine_weights = logits.softmax(dim=1)
            if self.minmax_weight:
                logits_min, _ = logits.min(dim=-1)
                logits_max, _ = logits.max(dim=-1)
                dispatch_weights_1 = (logits - logits_min.unsqueeze(-1)) / (
                            logits_max.unsqueeze(-1) - logits_min.unsqueeze(-1) + 1e-8)
            else:
                dispatch_weights_1 = dispatch_weights
            if self.wandb is not None:
                self.wandb.log({
                    "dispatch_max": dispatch_weights.max(),
                    "patch_num": L,
                    'add_num': add_length,
                    "combine_max": combine_weights.max(),
                    "combine_min": combine_weights.min(),
                }, commit=False)
            attn_windows = torch.einsum("w p c, w n p -> w n p c", x_windows, dispatch_weights).sum(dim=-2).transpose(0,
                                                                                                                      1)  # sW, nW, C
            # attn_windows = torch.nn.functional.adaptive_avg_pool1d(x_windows.transpose(-1,-2),1).squeeze(-1).unsqueeze(0)
            if return_attn:
                attn_windows, _attn = self.attn(self.norm(attn_windows), attn_mask, return_attn)  # sW, nW, C
                attn_windows = attn_windows.transpose(0, 1)  # nW, sW, C
            else:
                attn_windows = self.attn(self.norm(attn_windows), attn_mask).transpose(0, 1)  # nW, sW, C
            if self.no_weight_to_all:
                attn_windows = attn_windows.unsqueeze(-2).repeat(
                    (1, 1, window_size * window_size, 1))  # nW, sW, window_size*window_size, C
            else:
                attn_windows = torch.einsum("w n c, w n p -> w n p c", attn_windows,
                                            dispatch_weights_1)  # nW, sW, window_size*window_size, C
            attn_windows = torch.einsum("w n p c, w n p -> w n p c", attn_windows, combine_weights).sum(
                dim=1)  # nW, window_size*window_size, C
            # attn_windows = attn_windows.unsqueeze(0)
            # if self.l2_shortcut:
            #     attn_windows = x_windows + attn_windows
            # attn_windows = x_windows + attn_windows

        # merge windows
        attn_windows = attn_windows.view(-1, window_size, window_size, C)

        # pe
        # attn_windows = self.pe(attn_windows)

        # reverse cyclic shift
        if shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)

        # and self.shift_size and 
        # if self.training and self.shift_size and g_idx is not None:
        #     _,g_idx = g_idx.sort()
        #     x = patch_shuffle(x,10,g_idx=g_idx)

        # unpad
        # if sum(window_pad) != 0:
        #      x = (x.view(B, H, W, C)[:,window_pad[-2]:-window_pad[-1],window_pad[-4]:-window_pad[-3],:])
        #      x = x.reshape((B,x.size(1) * x.size(1),C))

        if add_length > 0:
            x = x[:, :-add_length]
        # if partition_length > 0:
        #     x = torch.cat([x,x_surplus],dim=1)
        if return_attn:
            return x, (_attn, dispatch_weights, combine_weights, dispatch_weights_1)
        else:
            return x


class SwinAttntion_(nn.Module):
    def __init__(self, dim, input_resolution=None, head_dim=None, num_heads=8, window_size=None, shift_size=False,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., window_num=8, conv=False):
        super().__init__()

        self.attn = SwinAttntion_(dim=dim, head_dim=head_dim, num_heads=num_heads, drop=drop, attn_drop=attn_drop,
                                  window_num=window_num, conv=conv)
        self.swattn = SwinAttntion_(dim=dim, head_dim=head_dim, num_heads=num_heads, drop=drop, attn_drop=attn_drop,
                                    window_num=window_num, conv=conv, shift_size=True)

    def forward(self, x, return_attn=False):
        shortcut = x
        return self.swattn(shortcut + self.attn(x))


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # H, W = self.input_resolution
        B, L, C = x.shape
        H, W = int(L ** 0.5), int(L ** 0.5)
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops






import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
from .emb_position import PEG
import math

# --------------------------------------------------------
# Fused kernel for window process for SwinTransformer
# Copyright (c) 2022 Nvidia
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")


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


def window_partition(x, window_size):
    """
    Args:
        x: (B, L, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, L, C = x.shape
    x = x.view(B, L // window_size, window_size, C)
    windows = x.transpose(1,2).contiguous().view(-1, window_size, C)
    return windows


def window_reverse(windows, window_size, L):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (L / window_size))
    x = windows.view(B, L // window_size,  window_size, -1)
    x = x.transpose(1,2).contiguous().view(B, L, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, head_dim=None, window_size=None, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,conv=False,conv_k=3,conv_2d=False,conv_bias=False,conv_type='attn',**kawrgs):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        if window_size is not None:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, head_dim * num_heads * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.conv_2d = conv_2d
        self.conv_type = conv_type
        if conv:
            kernel_size = conv_k
            padding = kernel_size // 2
    
            if conv_2d:
                if conv_type == 'attn':
                    self.pe = nn.Conv2d(num_heads, num_heads, kernel_size, padding = padding, groups = num_heads, bias = conv_bias)
                else:
                    self.pe = nn.Conv2d(head_dim * num_heads, head_dim * num_heads, kernel_size, padding = padding, groups = head_dim * num_heads, bias = conv_bias)
            else:
                self.pe = nn.Conv2d(num_heads, num_heads, (kernel_size, 1), padding = (padding, 0), groups = num_heads, bias = conv_bias)
        else:
            self.pe = None

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        if self.pe is not None:
            if self.conv_2d:
                pe = self.pe(attn.permute(0,2,1,3).reshape(-1,self.num_heads,int(np.ceil(np.sqrt(N))),int(np.ceil(np.sqrt(N)))))
                attn = attn+pe.reshape(B_,N,self.num_heads,N).transpose(1,2)
            else:
                pe = self.pe(attn)
                attn = attn+pe

        if self.window_size is not None:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        if self.pe is not None:
            #print(v.size())
            pe = self.pe(v)
            #pe = torch.einsum('ahbd->abhd',pe).flatten(-2,-1)
            v = v + pe

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.num_heads*self.head_dim)
        
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinAttntion1D(nn.Module):
    def __init__(self, dim, input_resolution=None, head_dim=None,num_heads=8, window_size=None, shift_size=0,qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., window_num=8,moe_enable=False,moe_mlp=False,moe_k=3,moe_mask_diag=False,**kawrgs):
        super().__init__()
 
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.window_num = window_num

        self.attn = WindowAttention(
            dim, head_dim=head_dim,num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,**kawrgs)
        
        self.moe_mask_diag = moe_mask_diag
        self.moe_mlp = moe_mlp
        if moe_enable:
            self.norm = nn.Identity()
            if moe_mlp:
                self.phi = [nn.Linear(self.dim, self.dim // 4,bias=False)]
                self.phi += [nn.Tanh()]

                self.phi += [nn.Linear(self.dim // 4, moe_k,bias=False)]
                self.phi = nn.Sequential(*self.phi)
            else:
                self.phi = nn.Parameter(
                    torch.empty(
                        (self.dim, moe_k),
                    )
                )
                nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))
        else:
            self.phi = None

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            self.attn_mask = None

    def forward(self,x,return_attn = False):
        B, L, C = x.shape

        # padding
    
        _n = -L % self.window_num
        _L = L+_n
        self.window_size = int(_L // self.window_num)

        add_length = _L - L

        # 如果要补的太多，就放弃window attention
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B,add_length,C),device=x.device)],dim = 1)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        if self.phi is None:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_mask = torch.diag(torch.tensor([float(-100.0) for i in range(x_windows.size(0))],device=x_windows.device)).unsqueeze(0) if self.moe_mask_diag else None
            if self.moe_mlp:
                logits = self.phi(x_windows).transpose(1,2) # W*B, sW, window_size*window_size
            else:
                logits = torch.einsum("w p c, c n -> w p n", x_windows, self.phi).transpose(1,2) # nW*B, sW, window_size*window_size
            # sW = logits.size(1)
            dispatch_weights = logits.softmax(dim=-1)
            combine_weights = logits.softmax(dim=1)

            logits_min,_ = logits.min(dim=-1)
            logits_max,_ = logits.max(dim=-1)
            dispatch_weights_1 = (logits - logits_min.unsqueeze(-1)) / (logits_max.unsqueeze(-1) - logits_min.unsqueeze(-1) + 1e-8)
            attn_windows =  torch.einsum("w p c, w n p -> w n p c", x_windows,dispatch_weights).sum(dim=-2).transpose(0,1) # sW, nW, C
            #attn_windows = torch.nn.functional.adaptive_avg_pool1d(x_windows.transpose(-1,-2),1).squeeze(-1).unsqueeze(0)
            if return_attn:
                attn_windows,_attn = self.attn(self.norm(attn_windows),attn_mask,return_attn)  # sW, nW, C
                attn_windows = attn_windows.transpose(0,1) # nW, sW, C
            else:
                attn_windows = self.attn(self.norm(attn_windows),attn_mask).transpose(0,1)  # nW, sW, C
            attn_windows = torch.einsum("w n c, w n p -> w n p c", attn_windows, dispatch_weights_1) # nW, sW, window_size*window_size, C
            attn_windows = torch.einsum("w n p c, w n p -> w n p c", attn_windows, combine_weights).sum(dim=1) # nW, 

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, _L)  # B H' W' C
            x = shifted_x
        x = x.view(B, _L, C)
        
        if add_length >0:
            x = x[:,:-add_length]
        return x