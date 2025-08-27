# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch.nn as nn
import torch

from .modulate import ModLN
from .plucker_multiheadattn import PluckerMultiheadAttention


class BasicBlock(nn.Module):
    """
    Transformer block that is in its simplest form.
    Designed for PF-LRM architecture.
    """
    # Block contains a self-attention layer and an MLP
    def __init__(self, inner_dim: int, num_heads: int, eps: float,
                 attn_drop: float = 0., attn_bias: bool = False,
                 mlp_ratio: float = 4., mlp_drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(inner_dim, eps=eps)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm2 = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )

    def forward(self, x):
        # x: [N, L, D]
        before_sa = self.norm1(x)
        x = x + self.self_attn(before_sa, before_sa, before_sa, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ConditionBlock(nn.Module):
    """
    Transformer block that takes in a cross-attention condition.
    Designed for SparseLRM architecture.
    """
    # Block contains a cross-attention layer, a self-attention layer, and an MLP
    def __init__(self, inner_dim: int, cond_dim: int, num_heads: int, eps: float,
                 attn_drop: float = 0., attn_bias: bool = False,
                 mlp_ratio: float = 4., mlp_drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(inner_dim, eps=eps)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm2 = nn.LayerNorm(inner_dim, eps=eps)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm3 = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )

    def forward(self, x, cond):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond]
        x = x + self.cross_attn(self.norm1(x), cond, cond, need_weights=False)[0]
        before_sa = self.norm2(x)
        x = x + self.self_attn(before_sa, before_sa, before_sa, need_weights=False)[0]
        x = x + self.mlp(self.norm3(x))
        return x


class ConditionExplicitattnBlock(nn.Module):
    """
    Transformer block that takes in a cross-attention condition.
    Designed for applying biased attention based on the distance matrix which
    you calculate between the plucker coordinates of the images used in the dataset
    and the plucker coordinates that define the 3D triplanes. These are referred
    to as the Base_Triplanes.
    """
    # Block contains a cross-attention layer, a self-attention layer, and an MLP
    def __init__(self, inner_dim: int, cond_dim: int, num_heads: int, eps: float,
                 attn_drop: float = 0., attn_bias: bool = False,
                 mlp_ratio: float = 4., mlp_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        self.gamma_ca = nn.Parameter(torch.tensor(1.0))
        self.gamma_sa = nn.Parameter(torch.tensor(1.0))
        self.norm1 = nn.LayerNorm(inner_dim, eps=eps)
        self.cross_attn = PluckerMultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm2 = nn.LayerNorm(inner_dim, eps=eps)
        self.self_attn = PluckerMultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm3 = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )

    def forward(self, x, cond, sa_distance_matrix, ca_distance_matrix):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond]
        scaled_ca_distance_matrix = ca_distance_matrix * (self.gamma_ca**2) * -1.0
        # Reshape to match the shape required by scaled_dot_product_attention
        # Repeat for each head
        scaled_ca_distance_matrix = torch.repeat_interleave(scaled_ca_distance_matrix, repeats=self.num_heads, dim=0)
        x = x + self.cross_attn(self.norm1(x), cond, cond, attn_mask=scaled_ca_distance_matrix, need_weights=False)[0]
        before_sa = self.norm2(x)
        scaled_sa_distance_matrix = sa_distance_matrix * (self.gamma_sa**2) * -1.0
        scaled_sa_distance_matrix = torch.repeat_interleave(scaled_sa_distance_matrix, repeats=self.num_heads, dim=0)
        x = x + self.self_attn(before_sa, before_sa, before_sa, attn_mask=scaled_sa_distance_matrix, need_weights=False)[0]
        x = x + self.mlp(self.norm3(x))
        return x


class ConditionModulationBlock(nn.Module):
    """
    Transformer block that takes in a cross-attention condition and another modulation vector applied to sub-blocks.
    Designed for raw LRM architecture.
    """
    # Block contains a cross-attention layer, a self-attention layer, and an MLP
    def __init__(self, inner_dim: int, cond_dim: int, mod_dim: int, num_heads: int, eps: float,
                 attn_drop: float = 0., attn_bias: bool = False,
                 mlp_ratio: float = 4., mlp_drop: float = 0.):
        super().__init__()
        self.norm1 = ModLN(inner_dim, mod_dim, eps)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm2 = ModLN(inner_dim, mod_dim, eps)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm3 = ModLN(inner_dim, mod_dim, eps)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )

    def forward(self, x, cond, mod):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond]
        # mod: [N, D_mod]
        x = x + self.cross_attn(self.norm1(x, mod), cond, cond, need_weights=False)[0]
        before_sa = self.norm2(x, mod)
        x = x + self.self_attn(before_sa, before_sa, before_sa, need_weights=False)[0]
        x = x + self.mlp(self.norm3(x, mod))
        return x
