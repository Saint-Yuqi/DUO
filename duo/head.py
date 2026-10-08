"""DuoFlowHead — the complete from-scratch reference head.

Layout (R3 / FLOWER-custom layout, 2026-09-24 → 2026-10-05): one ArmStream per partition stream = action in-projection,
learned chunk positions (or RoPE), time MLP, L adaLN-zero DiTBlocks that read that stream's own observation tokens,
final adaLN + out-projection to the stream's action dims. CrossArmSet couples the streams after the listed blocks,
multiplied by a per-sample gate. Flow matching (rectified flow, x_t = (1-t) x0 + t x1, target x1 - x0), Euler sampler.

The host backbone only has to supply, per stream, `obs` (B, N_i, dim) observation tokens and `pooled` (B, dim); see
duo/adapters/flower_custom_dit.py for how FLOWER's Florence-2 tokens were fed to exactly this head (65.38 / 18.84).
Partition.joint() gives the single-stream reference; Partition.mixed() the arm-grouping ablation of the paper.
"""
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from .coupling import CrossArmSet
from .layers import DiTBlock, modulate, sinusoidal_embedding
from .partition import Partition


class ArmStream(nn.Module):
    def __init__(self, act_dim, dim, layers, heads, chunk, mlp_ratio=4.0, dropout=0.0, qk_norm=True, rope=False, learned_pos=True):
        super().__init__()
        self.act_dim, self.chunk = int(act_dim), int(chunk)
        self.act_in = nn.Linear(act_dim, dim)
        self.act_pos = nn.Parameter(torch.randn(1, chunk, dim) * 0.02) if learned_pos else None
        self.t_mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.obs_norm = nn.LayerNorm(dim, eps=1e-6)
        self.obs_pool = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, mlp_ratio, dropout, qk_norm, rope) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.act_out = nn.Linear(dim, act_dim)
        nn.init.zeros_(self.final_ada[1].weight); nn.init.zeros_(self.final_ada[1].bias)
        nn.init.zeros_(self.act_out.weight); nn.init.zeros_(self.act_out.bias)

    def condition(self, obs, pooled_extra, t):
        """obs (B, N, dim) raw observation tokens; pooled_extra (B, dim) or None (e.g. own-proprio embedding)."""
        obs = self.obs_norm(obs)
        pooled = self.obs_pool(obs.mean(1))
        if pooled_extra is not None:
            pooled = pooled + pooled_extra.to(pooled.dtype)
        c = pooled + self.t_mlp(sinusoidal_embedding(t * 1000.0, 256).to(obs.dtype))
        return obs, c

    def embed(self, x):
        h = self.act_in(x)
        return h + self.act_pos.to(h.dtype) if self.act_pos is not None else h

    def finish(self, h, c):
        shift, scale = self.final_ada(c).chunk(2, dim=-1)
        return self.act_out(modulate(self.final_norm(h), shift, scale))


class DuoFlowHead(nn.Module):
    """ctx for forward/loss/sample: list (per stream) of dicts {"obs": (B, N_i, dim), "pooled": (B, dim) or None}, plus an
    optional gate (B,) in [0, 1] (None = always coupled)."""

    def __init__(self, partition: Partition = None, dim=512, layers=8, heads=8, chunk=50, cross_layers=(2, 4, 6, 8),
                 mlp_ratio=4.0, dropout=0.0, qk_norm=True, rope=False, share_streams=False, sample_steps=10):
        super().__init__()
        self.partition = partition or Partition.arm_semantic(7)
        self.chunk, self.dim, self.sample_steps = int(chunk), int(dim), int(sample_steps)
        self.cross_layers = tuple(i for i in cross_layers if 1 <= i <= layers)
        self.share_streams = bool(share_streams)
        if self.share_streams:
            dims = set(self.partition.stream_dims)
            assert len(dims) == 1, "shared stream weights need equal stream widths"
            one = ArmStream(dims.pop(), dim, layers, heads, chunk, mlp_ratio, dropout, qk_norm, rope, learned_pos=not rope)
            self.streams = nn.ModuleList([one] * self.partition.n_streams)
            self.stream_embed = nn.Parameter(torch.randn(self.partition.n_streams, dim) * 0.02)
        else:
            self.streams = nn.ModuleList([ArmStream(d, dim, layers, heads, chunk, mlp_ratio, dropout, qk_norm, rope, learned_pos=not rope)
                                          for d in self.partition.stream_dims])
            self.stream_embed = None
        self.cross = CrossArmSet(dim, heads, self.cross_layers, self.partition.n_streams, qk_norm, rope) if self.partition.n_streams > 1 else None
        self.n_layers = layers

    # ---- denoiser
    def velocity(self, x_t, t, ctx: List[dict], gate: Optional[torch.Tensor] = None):
        parts = self.partition.split(x_t)
        hs, obs, cs = [], [], []
        for i, (s, p) in enumerate(zip(self.streams, parts)):
            o, c = s.condition(ctx[i]["obs"], ctx[i].get("pooled"), t)
            h = s.embed(p.to(o.dtype))
            if self.stream_embed is not None:
                h = h + self.stream_embed[i].to(h.dtype)
            hs.append(h); obs.append(o); cs.append(c)
        for li in range(1, self.n_layers + 1):
            hs = [s.blocks[li - 1](h, o, c) for s, h, o, c in zip(self.streams, hs, obs, cs)]
            if self.cross is not None and self.cross.has_layer(li):
                hs = self.cross.apply(li, hs, gate)
        outs = [s.finish(h, c) for s, h, c in zip(self.streams, hs, cs)]
        return self.partition.merge(outs)

    def loss(self, actions, ctx, gate=None, mask=None, t=None):
        """Rectified-flow loss. actions (B, H, D) normalised; mask (B, H) or (B, H, D) weights (None = all ones)."""
        x1 = actions.float()
        x0 = torch.randn_like(x1)
        B = x1.shape[0]
        if t is None:
            t = torch.rand(B, device=x1.device).clamp(1e-3, 1 - 1e-3)
        x_t = (1 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * x1
        v = self.velocity(x_t, t, ctx, gate).float()
        err = (v - (x1 - x0)).square()
        if mask is None:
            return err.mean()
        w = mask.to(err.dtype)
        if w.dim() == 2:
            w = w.unsqueeze(-1).expand_as(err)
        return (err * w).sum() / w.sum().clamp_min(1.0)

    @torch.no_grad()
    def sample(self, ctx, gate=None, steps=None, generator=None):
        steps = steps or self.sample_steps
        B, dev = ctx[0]["obs"].shape[0], ctx[0]["obs"].device
        x = torch.randn(B, self.chunk, self.partition.total_dim, device=dev, generator=generator)
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((B,), k * dt, device=dev)
            x = x + dt * self.velocity(x, t, ctx, gate).float()
        return x

    def coupling_strength(self):
        return 0.0 if self.cross is None else self.cross.mean_strength()
