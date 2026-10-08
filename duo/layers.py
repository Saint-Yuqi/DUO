"""Transformer pieces shared by the from-scratch Duo streams.

Same shapes and initialisation as va_bimanual_b/models/dit_flow.py (R-series, 2026-09) and duo_jepa/models/layers.py
(2026-10-08), so numbers produced with those packages are a fair baseline for anything built here.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t, dim, max_period=10000.0):
    """t (B,) float -> (B, dim). Same convention as va_bimanual_b/models/common.py."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """Multi-head attention, self (ctx=None) or cross (ctx given). qk_norm = RMSNorm on q and k per head.
    rope=True applies 1-D rotary embeddings over the token index of q and k (see duo.rope); for cross-attention between
    two action streams this makes step t of one arm 'see' step t of the other arm at relative offset 0."""

    def __init__(self, dim, heads, dropout=0.0, qk_norm=True, rope=False):
        super().__init__()
        assert dim % heads == 0, (dim, heads)
        self.h, self.dh = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = dropout
        self.qn = RMSNorm(self.dh) if qk_norm else nn.Identity()
        self.kn = RMSNorm(self.dh) if qk_norm else nn.Identity()
        self.rope = bool(rope)

    def forward(self, x, ctx=None):
        from .rope import rotate   # local import: rope.py imports Attention for RoPEAttention
        ctx = x if ctx is None else ctx
        B, N, D = x.shape
        q = self.qn(self.q(x).view(B, N, self.h, self.dh)).transpose(1, 2)
        k, v = self.kv(ctx).view(B, ctx.shape[1], 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        k = self.kn(k)
        if self.rope:
            q, k = rotate(q), rotate(k)
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.out(o.transpose(1, 2).reshape(B, N, D))


class DiTBlock(nn.Module):
    """adaLN-zero block: self-attention over the stream's own tokens -> cross-attention into the observation tokens ->
    MLP. `c` (B, dim) is the pooled conditioning (time + pooled observation + own proprio)."""

    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0, qk_norm=True, rope=False):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads, dropout, qk_norm, rope)
        self.n2 = nn.LayerNorm(dim, eps=1e-6)
        self.cross = Attention(dim, heads, dropout, qk_norm, rope=False)
        self.n3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hid), nn.GELU(approximate="tanh"), nn.Dropout(dropout), nn.Linear(hid, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, obs, c):
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.n1(x), s1, sc1))
        x = x + self.cross(self.n2(x), obs)
        x = x + g2.unsqueeze(1) * self.mlp(modulate(self.n3(x), s2, sc2))
        return x
