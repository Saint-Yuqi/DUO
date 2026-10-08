"""Cross-arm communication: the one thing every Duo head has in common.

CrossArm   : own tokens attend the other stream's tokens; the residual is scaled by tanh(gain), gain = 0 at init, so the
             head starts as N fully decoupled per-arm experts and training opens the channel where the data needs it.
             (va_bimanual_b/models/duo_dit.py 2026-09-24; FLOWER heads 2026-10-03; OpenWAM native dual 2026-10-06.)
CrossArmSet: one CrossArm per (layer, direction); `apply` updates all streams simultaneously from their pre-update
             values and multiplies the residual by a per-sample gate g in [0, 1] (rule / learned / always 1).
block_cross_stream_attention: for MoT-style hosts (OpenWAM / Fast-WAM) whose action tokens share one joint attention —
             masks the left<->right blocks so the only inter-arm path is the CrossArm modules.
"""
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .layers import Attention


class CrossArm(nn.Module):
    def __init__(self, dim, heads, qk_norm=True, rope=False):
        super().__init__()
        self.nq = nn.LayerNorm(dim, eps=1e-6)
        self.nk = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads, 0.0, qk_norm, rope)
        self.gain = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        return torch.tanh(self.gain).to(x.dtype) * self.attn(self.nq(x), self.nk(y))

    def strength(self):
        return float(torch.tanh(self.gain.detach()).abs())


class CrossArmSet(nn.Module):
    """Modules keyed f"{src}{dst}@{layer}" for every ordered pair of streams at every listed layer (1-based block index).
    With 2 streams this is the usual L<-R and R<-L pair, separate weights per direction."""

    def __init__(self, dim, heads, layers: Sequence[int], n_streams=2, qk_norm=True, rope=False):
        super().__init__()
        self.layers = tuple(sorted({int(x) for x in layers}))
        self.n = int(n_streams)
        self.mods = nn.ModuleDict()
        for li in self.layers:
            for dst in range(self.n):
                for src in range(self.n):
                    if src != dst:
                        self.mods[self.key(src, dst, li)] = CrossArm(dim, heads, qk_norm, rope)

    @staticmethod
    def key(src, dst, layer):
        return f"{src}to{dst}@{layer}"

    def has_layer(self, layer):
        return layer in self.layers

    def apply(self, layer: int, hs: List[torch.Tensor], gate: Optional[torch.Tensor] = None):
        """hs: per-stream token tensors (B, T_i, dim). gate: None (=1) or (B,) in [0, 1]. Simultaneous update."""
        if layer not in self.layers or self.n == 1:
            return hs
        g = None if gate is None else gate.view(-1, 1, 1).to(hs[0].dtype)
        out = []
        for dst in range(self.n):
            d = hs[dst]
            for src in range(self.n):
                if src == dst:
                    continue
                delta = self.mods[self.key(src, dst, layer)](hs[dst], hs[src])
                d = d + (delta if g is None else g * delta)
            out.append(d)
        return out

    def strengths(self) -> Dict[str, float]:
        return {k: m.strength() for k, m in self.mods.items()}

    def mean_strength(self):
        s = self.strengths()
        return sum(s.values()) / max(1, len(s))


def block_cross_stream_attention(mask: torch.Tensor, start: int, n_action_tokens: int, n_streams=2, allowed=True):
    """mask: boolean (S, S) or (B, S, S) attention mask where True = may attend (OpenWAM / Fast-WAM convention).
    The action tokens occupy [start, start + n_action_tokens) as n_streams equal consecutive blocks; this sets the
    off-diagonal stream-to-stream blocks to False so arms only communicate through CrossArm. Returns the mask."""
    assert n_action_tokens % n_streams == 0, (n_action_tokens, n_streams)
    half = n_action_tokens // n_streams
    for a in range(n_streams):
        for b in range(n_streams):
            if a != b:
                ra = slice(start + a * half, start + (a + 1) * half)
                rb = slice(start + b * half, start + (b + 1) * half)
                mask[..., ra, rb] = not allowed if isinstance(allowed, bool) and allowed is False else False
    return mask


def gate_from_batch(batch, B, device, mode="open", key="contact_gate", drop=0.0, training=False):
    """mode: 'open' -> ones; 'closed' -> zeros; 'rule' -> batch[key] (B,) float, or (batch['contact'] >= 2) as in the
    R-series caches, zeros when absent. `drop` (training only) randomly closes open gates (cross_drop of duo_dit.py)."""
    if mode == "open":
        g = torch.ones(B, device=device)
    elif mode == "closed":
        g = torch.zeros(B, device=device)
    elif mode == "rule":
        g = batch.get(key)
        if g is None:
            c = batch.get("contact")
            g = (c.view(-1) >= 2).float() if c is not None else torch.zeros(B, device=device)
        g = g.to(device).float().view(B)
    else:
        raise ValueError(mode)
    if training and drop > 0:
        g = g * (torch.rand(B, device=device) >= drop).float()
    return g
