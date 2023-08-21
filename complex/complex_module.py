import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import math


def apply_complex(F_r, F_i, X):
    X_r, X_i = [x.squeeze(dim=-1) for x in torch.split(X, 1, dim=-1)]
    return torch.stack((F_r(X_r) - F_i(X_i), F_r(X_i) + F_i(X_r)), dim=-1)

def apply_complex_sep(F_r, F_i, X):
    X_r, X_i = [x.squeeze(dim=-1) for x in torch.split(X, 1, dim=-1)]
    return torch.stack((F_r(X_r), F_i(X_i)), dim=-1)

@torch.jit.script
def complex_mul(X, Y):
    X_r, X_i = [x.squeeze(dim=-1) for x in torch.split(X, 1, dim=-1)]
    Y_r, Y_i = [y.squeeze(dim=-1) for y in torch.split(Y, 1, dim=-1)]
    Z_r = torch.mul(X_r, Y_r) - torch.mul(X_i, Y_i)
    Z_i = torch.mul(X_r, Y_i) + torch.mul(X_i, Y_r)
    return torch.stack((Z_r, Z_i), dim=-1)

@torch.jit.script
def complex_bmm(X, Y):
    X_r, X_i = [x.squeeze(dim=-1) for x in torch.split(X, 1, dim=-1)]
    Y_r, Y_i = [y.squeeze(dim=-1) for y in torch.split(Y, 1, dim=-1)]
    Z_r = torch.bmm(X_r, Y_r) - torch.bmm(X_i, Y_i)
    Z_i = torch.bmm(X_r, Y_i) + torch.bmm(X_i, Y_r)
    return torch.stack((Z_r, Z_i), dim=-1)

@torch.jit.script
def complex_softmax(X):
    X_r, X_i = [x.squeeze(dim=-1) for x in torch.split(X, 1, dim=-1)]
    return torch.stack((F.softmax(X_r, dim=-1), F.softmax(X_i, dim=-1)), dim=-1)

@torch.jit.script
def transpose_qkv(x, num_heads: int):
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1, 2)
    x = x.transpose(1, 2)
    return x.reshape(-1, x.shape[2], x.shape[3], 2)

@torch.jit.script
def transpose_output(x, num_heads: int):
    x = x.reshape(-1, num_heads, x.shape[1], x.shape[2], 2)
    x = x.transpose(1, 2)
    return x.reshape(x.shape[0], x.shape[1], -1, 2)


class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, X):
        device = X.device
        dtype = X.dtype
        mask = torch.ones(*X.shape[-3:], device=device, dtype=dtype)
        mask = F.dropout1d(mask, p=0.5, training=self.training)
        return torch.mul(X, mask)


class ComplexGELU(nn.Module):
    def __init__(self, approximate='none'):
        super().__init__()
        self.gelu_r = nn.GELU(approximate)
        self.gelu_i = nn.GELU(approximate)
    
    def forward(self, X):
        return apply_complex_sep(self.gelu_r, self.gelu_i, X)


class ComplexSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu_r = nn.SiLU()
        self.silu_i = nn.SiLU()

    def forward(self, X):
        return apply_complex_sep(self.silu_r, self.silu_i, X)


class ComplexReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_r = nn.ReLU()
        self.relu_i = nn.ReLU()

    def forward(self, X):
        return apply_complex_sep(self.relu_r, self.relu_i, X)


class ComplexAvgPool3d(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.avg_pool_r = nn.AvgPool3d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.avg_pool_i = nn.AvgPool3d(
            kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, X):
        return apply_complex_sep(self.avg_pool_r, self.avg_pool_i, X)


class ComplexFlatten(nn.Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.flt_r = nn.Flatten(start_dim=start_dim, end_dim=end_dim)
        self.flt_i = nn.Flatten(start_dim=start_dim, end_dim=end_dim)

    def forward(self, X):
        return apply_complex_sep(self.flt_r, self.flt_i, X)


class NaiveComplexBatchNorm3d(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(NaiveComplexBatchNorm3d, self).__init__()
        self.bn_r = nn.BatchNorm3d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.bn_i = nn.BatchNorm3d(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, X):
        return apply_complex_sep(self.bn_r, self.bn_i, X)


class NaiveComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(NaiveComplexLayerNorm, self).__init__()
        self.ln_r = nn.LayerNorm(normalized_shape, eps, elementwise_affine)
        self.ln_i = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, X):
        return apply_complex_sep(self.ln_r, self.ln_i, X)


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.l_r = nn.Linear(in_features, out_features, bias=bias, dtype=torch.float32)
        self.l_i = nn.Linear(in_features, out_features, bias=bias, dtype=torch.float32)

    def forward(self, X):
        return apply_complex(self.l_r, self.l_i, X)


class ComplexMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=ComplexGELU, bias=True, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ComplexLinear(in_features, hidden_features, bias)
        self.act = act_layer()
        self.drop1 = ComplexDropout(dropout)
        self.fc2 = ComplexLinear(hidden_features, out_features, bias)
        self.drop2 = ComplexDropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class ComplexConv3d(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size, padding, stride=1):
        super().__init__()
        self.conv_r = nn.Conv3d(
            input_channels,
            num_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dtype=torch.float32,
        )
        self.conv_i = nn.Conv3d(
            input_channels,
            num_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dtype=torch.float32,
        )

    def forward(self, X):
        return apply_complex(self.conv_r, self.conv_i, X)


class ComplexResidual3d(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size, padding, stride=1):
        super().__init__()
        self.conv1 = ComplexConv3d(
            input_channels,
            num_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.conv2 = ComplexConv3d(
            num_channels,
            num_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.conv3 = ComplexConv3d(
            input_channels, num_channels, kernel_size=1, padding=0, stride=stride
        )
        self.bn1 = NaiveComplexBatchNorm3d(num_channels)
        self.bn2 = NaiveComplexBatchNorm3d(num_channels)
        self.relu1 = ComplexReLU()
        self.relu2 = ComplexReLU()

    def forward(self, X):
        Y = self.relu1(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y)) + self.conv3(X)
        return self.relu2(Y)


# [32 32 10 10 3] -> [32 10 32*10*3]
class ComplexSegment(nn.Module):
    def __init__(self, input_channels, seg_channels, seg_size):
        super().__init__()
        self.seg_conv = ComplexResidual3d(
            input_channels,
            seg_channels,
            kernel_size=seg_size,
            padding=(0, 0, 0),
            stride=seg_size,
        )
        self.flt = ComplexFlatten(start_dim=2, end_dim=-1)

    def forward(self, X):
        Y = self.seg_conv(X)
        Y = Y.transpose(1, 2)
        Y = self.flt(Y)
        return Y


class Complex2Real(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, X):
        X = self.linear1(X)
        X = self.linear2(F.relu(X))
        return X.squeeze(dim=-1)


class ComplexDotProductAttention(nn.Module):
    """
    Query shape: [batch_size, query_num, query_key_dim]
    Key shape: [batch_size, key_value_num, query_key_dim]
    Value shape: [batch_size, key_value_num, value_dim]
    """
    def __init__(self, dropout, **kwargs):
        super(ComplexDotProductAttention, self).__init__(**kwargs)
        self.dropout = ComplexDropout(dropout)

    def forward(self, queries, keys, values):
        query_key_dim = queries.shape[-2]
        self.attention_weights = complex_softmax(
            complex_bmm(queries, keys.transpose(1, 2)) / math.sqrt(query_key_dim)
        )
        Y = complex_bmm(self.dropout(self.attention_weights), values)
        return Y


class ComplexMultiHeadAttention(nn.Module):
    def __init__(
        self,
        query_size,
        num_hiddens,
        num_heads,
        dropout,
        key_size=None,
        value_size=None,
        bias=False,
        **kwargs
    ):
        super(ComplexMultiHeadAttention, self).__init__(**kwargs)
        key_size = key_size or query_size
        value_size = value_size or query_size
        self.num_heads = num_heads
        self.attention = ComplexDotProductAttention(dropout=dropout)
        self.w_q = ComplexLinear(query_size, num_hiddens, bias=bias)
        self.w_k = ComplexLinear(key_size, num_hiddens, bias=bias)
        self.w_v = ComplexLinear(value_size, num_hiddens, bias=bias)
        self.w_o = ComplexLinear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values):
        queries = transpose_qkv(self.w_q(queries), self.num_heads)
        keys = transpose_qkv(self.w_k(keys), self.num_heads)
        values = transpose_qkv(self.w_v(values), self.num_heads)
        output = self.attention(queries, keys, values)
        output_concat = transpose_output(output, self.num_heads)
        Y = self.w_o(output_concat)
        return Y


class ComplexPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout, max_len=10000):
        super(ComplexPositionalEncoding, self).__init__()
        self.dropout = ComplexDropout(dropout)
        pcode = torch.zeros((1, max_len, hidden_dim, 2), dtype=torch.float32)
        pos = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, hidden_dim, dtype=torch.float32) / hidden_dim
        )
        pcode[:, :, :, 0] = torch.cos(pos)
        pcode[:, :, :, 1] = torch.sin(pos)
        self.register_buffer("pcode", pcode, persistent=False)

    def forward(self, X):
        X = complex_mul(X, self.pcode[:, : X.shape[1], :, :].to(X.device))
        Y = self.dropout(X)
        return Y


class PositionWiseFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.linear1 = ComplexLinear(input_dim, hidden_dim)
        self.relu = ComplexReLU()
        self.linear2 = ComplexLinear(hidden_dim, output_dim)

    def forward(self, X):
        Y = self.linear2(self.relu(self.linear1(X)))
        return Y


class ComplexAddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(ComplexAddNorm, self).__init__(**kwargs)
        self.dropout = ComplexDropout(dropout)
        self.ln = NaiveComplexLayerNorm(normalized_shape)

    def forward(self, X, Y):
        Y = self.ln(self.dropout(Y) + X)
        return Y


class ComplexEncoderBlock(nn.Module):
    def __init__(
        self,
        key_dim,
        query_dim,
        value_dim,
        hidden_dim,
        norm_shape,
        ffn_input_dim,
        ffn_hidden_dim,
        num_heads,
        dropout,
        use_bias=False,
        **kwargs
    ):
        super(ComplexEncoderBlock, self).__init__(**kwargs)
        self.attention = ComplexMultiHeadAttention(
            key_dim, query_dim, value_dim, hidden_dim, num_heads, dropout, use_bias
        )
        self.addnorm1 = ComplexAddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_input_dim, ffn_hidden_dim, ffn_hidden_dim)
        self.addnorm2 = ComplexAddNorm(norm_shape, dropout)

    def forward(self, X):
        Y = self.attention(X, X, X)
        Z = self.addnorm1(X, Y)
        return self.addnorm2(Z, self.ffn(Y))


class ComplexTransformerEncoder(nn.Module):
    def __init__(
        self,
        key_dim,
        query_dim,
        value_dim,
        hidden_dim,
        norm_shape,
        ffn_input_dim,
        ffn_hidden_dim,
        num_heads,
        num_layers,
        dropout,
        use_bias=False,
        **kwargs
    ):
        super(ComplexTransformerEncoder, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.pos_encoding = ComplexPositionalEncoding(hidden_dim, dropout)
        self.blks = nn.Sequential()
        for n in range(num_layers):
            self.blks.add_module(
                "Block" + str(n),
                ComplexEncoderBlock(
                    key_dim,
                    query_dim,
                    value_dim,
                    hidden_dim,
                    norm_shape,
                    ffn_input_dim,
                    ffn_hidden_dim,
                    num_heads,
                    dropout,
                    use_bias,
                ),
            )

    def forward(self, X, *args):
        X = self.pos_encoding(X * math.sqrt(self.hidden_dim))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X
