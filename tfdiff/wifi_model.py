import math
from math import sqrt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import complex.complex_module as cm


def init_weight_norm(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_weight_zero(module):
    if isinstance(module, nn.Linear):
        nn.init.constant_(module.weight, 0)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_weight_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


@torch.jit.script
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_step, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(
            max_step, embed_dim), persistent=False)
        self.projection = nn.Sequential(
            cm.ComplexLinear(embed_dim, hidden_dim, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, hidden_dim, bias=True),
        )
        self.hidden_dim = hidden_dim
        self.apply(init_weight_norm)

    def forward(self, t):
        if t.dtype in [torch.int32, torch.int64]:
            x = self.embedding[t]
        else:
            x = self._lerp_embedding(t)
        return self.projection(x)

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_step, embed_dim):
        steps = torch.arange(max_step).unsqueeze(1)  # [T, 1]
        dims = torch.arange(embed_dim).unsqueeze(0)  # [1, E]
        table = steps * torch.exp(-math.log(max_step)
                                  * dims / embed_dim)  # [T, E]
        table = torch.view_as_real(torch.exp(1j * table))
        return table


# TODO: Replace MLP with nn.Embedding
class MLPConditionEmbedding(nn.Module):
    def __init__(self, cond_dim, hidden_dim=256):
        super().__init__()
        self.projection = nn.Sequential(
            cm.ComplexLinear(cond_dim, hidden_dim, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, hidden_dim*4, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim*4, hidden_dim, bias=True),
        )
        self.apply(init_weight_norm)

    def forward(self, c):
        return self.projection(c)


class PositionEmbedding(nn.Module):
    def __init__(self, max_len, input_dim, hidden_dim):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(
            max_len, hidden_dim), persistent=False)
        self.projection = cm.ComplexLinear(input_dim, hidden_dim)
        self.apply(init_weight_xavier)

    def forward(self, x): 
        x = self.projection(x)
        return cm.complex_mul(x, self.embedding.to(x.device))

    def _build_embedding(self, max_len, hidden_dim):
        steps = torch.arange(max_len).unsqueeze(1)  # [P,1]
        dims = torch.arange(hidden_dim).unsqueeze(0)          # [1,E]
        table = steps * torch.exp(-math.log(max_len)
                                  * dims / hidden_dim)     # [P,E]
        table = torch.view_as_real(torch.exp(1j * table))
        return table


class DiA(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.attn = cm.ComplexMultiHeadAttention(
            hidden_dim, hidden_dim, num_heads, dropout, bias=True, **block_kwargs)
        self.norm2 = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            cm.ComplexLinear(hidden_dim, mlp_hidden_dim, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(mlp_hidden_dim, hidden_dim, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, 6*hidden_dim, bias=True)
        )
        self.apply(init_weight_xavier)
        self.adaLN_modulation.apply(init_weight_zero)

    def forward(self, x, c):
        """
        Embedding diffusion step t with adaptive layer-norm.
        Embedding condition c with cross-attention.
        - Input:\\
          x, [B, N, H, 2], \\ 
          t, [B, H, 2], \\
          c, [B, N, H, 2], \\
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c).chunk(6, dim=1)
        mod_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + \
            gate_msa.unsqueeze(
                1) * self.attn(mod_x, mod_x, mod_x)
        x = x + \
            gate_mlp.unsqueeze(
                1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.linear = cm.ComplexLinear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, 2*hidden_dim, bias=True)
        )
        self.apply(init_weight_zero)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


# class SpatialDiffusion(nn.Module):
#     """
#     Process each sample of a sequence.
#     Take CSI diffusion as an example.
#     - Input:\\
#       x, [B, S, A, 2], \\
#       t, [B], \\
#       c, [B, C, 2], \\
#     - Output:
#       n, [B, S*A, 2]
#     """

#     def __init__(self, params):
#         super().__init__()
#         self.learn_tfdiff = params.learn_tfdiff
#         self.num_block = params.num_spatial_block
#         self.input_dim = params.extra_dim[-1]  # A
#         self.input_len = params.extra_dim[-2]  # S
#         self.output_dim = self.input_dim * self.input_len  # S*A
#         self.hidden_dim = params.spatial_hidden_dim  # D
#         self.num_heads = params.num_heads  # H
#         self.max_step = params.max_step  # T
#         self.embed_dim = params.embed_dim  # E
#         self.cond_dim = params.cond_dim[-1]  # C
#         self.dropout = params.dropout
#         self.task_id = params.task_id
#         self.mlp_ratio = params.mlp_ratio
#         self.p_embed = PositionEmbedding(
#             self.input_len, self.input_dim, self.hidden_dim)
#         self.t_embed = DiffusionEmbedding(
#             self.max_step, self.embed_dim, self.hidden_dim)
#         self.c_embed = MLPConditionEmbedding(self.cond_dim, self.hidden_dim)
#         # A series of concatenated DiA blocks.
#         self.blocks = nn.ModuleList([
#             DiA(self.hidden_dim, self.num_heads, self.dropout, self.mlp_ratio) for _ in range(self.num_block)
#         ])
#         self.adaMLP = nn.Sequential(
#             # Flatten [B, S, A, 2] to [B, S*A, 2]
#             nn.Flatten(start_dim=1, end_dim=-2),
#             cm.ComplexLinear(self.input_len*self.hidden_dim, self.output_dim),
#             cm.ComplexSiLU(),
#             cm.ComplexLinear(self.output_dim, self.output_dim),
#         )
#         self.adaMLP.apply(init_weight_xavier)

#     def forward(self, x, t, c):
#         self.p_embed = self.p_embed.to(x.device)
#         self.t_embed = self.t_embed.to(x.device)
#         self.c_embed = self.c_embed.to(x.device)
#         x = self.p_embed(x)
#         t = self.t_embed(t)
#         c = self.c_embed(c)
#         for block in self.blocks:
#             block = block.to(x.device)
#             x = block(x, t, c)
#         self.adaMLP = self.adaMLP.to(x.device)
#         x = self.adaMLP(x)
#         return x


# class TimeFrequencyDiffusion(nn.Module):
#     """
#     Process the whole sequence.
#     Take CSI diffusion as an example.
#     - Input:\\
#       x, [B, N, S*A, 2], \\
#       t, [B], \\
#       c, [B, N, C, 2], \\
#     - Output:
#       n, [B, N, S*A, 2]
#     """

#     def __init__(self, params):
#         super().__init__()
#         self.learn_tfdiff = params.learn_tfdiff
#         self.num_block = params.num_tf_block
#         self.batch_size = params.batch_size
#         self.input_dim = np.prod(params.extra_dim)  # S*A
#         self.input_len = params.sample_rate  # N
#         self.output_dim = self.input_dim  # S*A
#         self.hidden_dim = params.tf_hidden_dim  # D
#         self.num_heads = params.num_heads  # H
#         self.max_step = params.max_step  # T
#         self.embed_dim = params.embed_dim  # E
#         self.cond_dim = np.prod(params.cond_dim)  # C
#         self.dropout = params.dropout
#         self.task_id = params.task_id
#         self.mlp_ratio = params.mlp_ratio
#         self.p_embed = PositionEmbedding(
#             self.input_len, self.input_dim, self.hidden_dim)
#         self.t_embed = DiffusionEmbedding(
#             self.max_step, self.embed_dim, self.hidden_dim)
#         self.c_embed = MLPConditionEmbedding(self.cond_dim, self.hidden_dim)
#         self.blocks = nn.ModuleList([
#             DiA(self.hidden_dim, self.num_heads, self.dropout, self.mlp_ratio) for _ in range(self.num_block)
#         ])
#         self.final_layer = FinalLayer(
#             self.hidden_dim, self.output_dim)

#     def forward(self, x, t, c):
#         x = self.p_embed(x)
#         t = self.t_embed(t)
#         if self.task_id == 0:
#             c = c.reshape([self.batch_size, self.input_len, -1, 2])
#         if self.task_id == 1:
#             c = c.reshape([self.batch_size, self.input_len, -1, 2])
#         c = self.c_embed(c)
#         for block in self.blocks:
#             x = block(x, t, c)
#         x = self.final_layer(x, t)
#         return x


# class tfdiff_WiFi(nn.Module):
#     """
#     Signal Modulation and Augmentation via Generative Diffusion Model.
#     Take CSI diffusion as an example.
#     - Input:\\
#       x, [B, N, S, A, 2], \\
#       t, [B], \\
#       c, [B, N, C, 2], \\
#     - Output:
#       n, [B, N, S, A, 2]
#     """

#     def __init__(self, params):
#         super().__init__()
#         self.params = params
#         self.task_id = params.task_id
#         self.sample_rate = params.sample_rate
#         self.extra_dim = params.extra_dim
#         self.cond_dim = params.cond_dim
#         self.batch_size = params.batch_size
#         self.spatial_dim = np.prod(self.extra_dim)
#         # N parallel SpatialDiffusion blocks.
#         self.spatial_block = SpatialDiffusion(self.params)
#         self.tf_block = TimeFrequencyDiffusion(self.params)

#     def forward(self, x, t, c):
#         x_s = x.reshape([-1]+self.extra_dim+[2])  # [B*N, S, A, 2]
#         x_s = self.spatial_block(x_s, t.repeat(self.sample_rate), c.repeat(self.sample_rate))
#         x = x_s.reshape([-1, self.sample_rate]+[self.spatial_dim, 2])  # [B, N, S*A, 2]
#         x = self.tf_block(x, t, c)  # [B, N, S*A, 2]
#         x = x.reshape([-1, self.sample_rate]+self.extra_dim+[2])  # [B, N, S, A, 2]
#         return x
    

class tfdiff_WiFi(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.learn_tfdiff = params.learn_tfdiff
        self.input_dim = params.input_dim
        self.output_dim = self.input_dim
        self.hidden_dim = params.hidden_dim
        self.num_heads = params.num_heads
        self.dropout = params.dropout
        self.task_id = params.task_id
        self.mlp_ratio = params.mlp_ratio
        self.p_embed = PositionEmbedding(
            params.sample_rate, params.input_dim, params.hidden_dim)
        self.t_embed = DiffusionEmbedding(
            params.max_step, params.embed_dim, params.hidden_dim)
    
        self.c_embed = MLPConditionEmbedding(params.cond_dim, params.hidden_dim)
        self.blocks = nn.ModuleList([
            DiA(self.hidden_dim, self.num_heads, self.dropout, self.mlp_ratio) for _ in range(params.num_block)
        ])
        self.final_layer = FinalLayer(self.hidden_dim, self.output_dim)

    def forward(self, x, t, c):
        x = self.p_embed(x)
        t = self.t_embed(t)
        c = self.c_embed(c)
        c = c + t
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return x