"""[duo.adapters — verbatim copy, do not edit here]
source : /tmp/robotwin_rope/action_rope.py
copied : 2026-10-08 from quic5000 (as it ran; snapshot date 2026-10-05)
ledger : flower_customduodit_rope_clean2500; openwam_native_dual_clean2500 (CrossArm RoPE)
note   : original RoPE helper; the library version is duo/rope.py
Edit the canonical copy in the training workspace, then re-copy; the ledger row points at this file + the source path.
"""
"""RoPE for action-to-action attention; visual attention stays unchanged."""
from functools import lru_cache

import torch
import torch.nn.functional as F

from models.dit_flow import Attention


@lru_cache(maxsize=32)
def _freqs(length, head_dim, device):
    if head_dim % 2:
        raise ValueError("RoPE requires an even attention head dimension")
    inverse = 10000.0 ** (-torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    angles = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inverse)
    return torch.polar(torch.ones_like(angles), angles)[None, None]


def rotate(x):
    freqs = _freqs(x.shape[-2], x.shape[-1], x.device)
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(pairs * freqs).flatten(-2).to(x.dtype)


class RoPEAttention(Attention):
    def forward(self, x, ctx=None):
        ctx = x if ctx is None else ctx
        batch, length, dim = x.shape
        q = self.qn(self.q(x).view(batch, length, self.h, self.dh)).transpose(1, 2)
        k, v = self.kv(ctx).view(batch, ctx.shape[1], 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        k = self.kn(k)
        out = F.scaled_dot_product_attention(
            rotate(q), rotate(k), v, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out(out.transpose(1, 2).reshape(batch, length, dim))


def enable_rope(attention):
    if type(attention) is not Attention:
        raise TypeError(f"Expected VA Attention, got {type(attention)}")
    attention.__class__ = RoPEAttention


if __name__ == "__main__":
    torch.manual_seed(42)
    attention = Attention(64, 4, qk_norm=True)
    enable_rope(attention)
    x = torch.randn(2, 7, 64, requires_grad=True)
    y = attention(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)
    a = torch.cat((q, k), dim=2)
    b = torch.cat((torch.zeros_like(q), q, k), dim=2)
    score_a = (rotate(a)[:, :, 0] * rotate(a)[:, :, 1]).sum()
    score_b = (rotate(b)[:, :, 1] * rotate(b)[:, :, 2]).sum()
    assert torch.allclose(score_a, score_b, atol=1e-5)
    print("RoPE attention shape, gradient, and relative-position checks passed")
