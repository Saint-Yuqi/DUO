"""1-D rotary position embedding over the chunk index.

Port of /tmp/robotwin_rope/action_rope.py (quic5000, 2026-10-05): used by the FLOWER 'custom DuoDiT + RoPE' run
(66.50 / 19.56) and, through `enable_rope`, by the cross-arm modules of the OpenWAM native-dual head. Both streams use
positions 0..H-1, so the t-th token of the left stream and the t-th token of the right stream share a position.
"""
from functools import lru_cache

import torch
import torch.nn.functional as F

from .layers import Attention


@lru_cache(maxsize=64)
def _freqs(length, head_dim, device_str):
    if head_dim % 2:
        raise ValueError("RoPE requires an even attention head dimension")
    device = torch.device(device_str)
    inverse = 10000.0 ** (-torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    angles = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inverse)
    return torch.polar(torch.ones_like(angles), angles)[None, None]       # (1, 1, L, dh/2) complex


def rotate(x):
    """x (B, heads, L, dh) -> rotated by its own token index."""
    freqs = _freqs(x.shape[-2], x.shape[-1], str(x.device))
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(pairs * freqs).flatten(-2).to(x.dtype)


class RoPEAttention(Attention):
    """Attention whose q and k are rotated by token index (same as Attention(rope=True); kept for `enable_rope`)."""

    def forward(self, x, ctx=None):
        ctx = x if ctx is None else ctx
        B, N, D = x.shape
        q = self.qn(self.q(x).view(B, N, self.h, self.dh)).transpose(1, 2)
        k, v = self.kv(ctx).view(B, ctx.shape[1], 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        k = self.kn(k)
        o = F.scaled_dot_product_attention(rotate(q), rotate(k), v, dropout_p=self.dropout if self.training else 0.0)
        return self.out(o.transpose(1, 2).reshape(B, N, D))


def enable_rope(attention):
    """In-place: turn an existing duo.layers.Attention into a RoPE attention (no new parameters)."""
    if not isinstance(attention, Attention):
        raise TypeError(f"Expected duo.layers.Attention, got {type(attention)}")
    attention.rope = True
    return attention
