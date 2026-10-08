"""[duo.adapters — verbatim copy, do not edit here]
source : /mnt/cpfs/yangyq/code/va_bimanual_b/models/duo_dit.py
copied : 2026-10-08 from quic5000 (as it ran; snapshot date 2026-09-25)
ledger : R3, R3b, R3c (zzm500 protocol); P16_DUO_* real-robot runs
note   : v1 (09-24) + v2 (09-25: head_contact aux head, loss_step_w critical-point weights, DUO_EVAL_GATE); imports va_bimanual_b (dit_flow.DiTFlowPolicy, common, tricks). Same file as /mnt/workspace/zzm/duo_dit/models/duo_dit.py (diff: identical).
Edit the canonical copy in the training workspace, then re-copy; the ledger row points at this file + the source path.
"""
"""Duo DiT: two per-arm DiT flow policies (head + own wrist + own state each - exactly the B9 pair) whose action-token
streams exchange features through zero-initialised cross-arm attention blocks that are OPEN only when the contact-graph
gate says the two kinematic chains are coupled (S / A frames).
    gate = 0  ->  two independent per-arm samplers over their own observations (B9, bit for bit at init)
    gate = 1  ->  joint denoising: after the listed DiT blocks each stream attends the other's action(+future) tokens
The other arm's raw joints never enter either stream: coupling flows only through the action-token features.
Gate source: batch["contact_gate"] (deploy: contact_rules.gate_online on the measured state) or the offline label
batch["contact"] (contact_v1.npy of the cache, mode >= 2) during training / validation; zeros when neither exists.
Inputs (train_va.py --model duo --static_cam <head> --gripper_cam <left wrist> --third_cam <right wrist> --arm both):
rgb_static = head (shared), rgb_gripper = left wrist, rgb_third = right wrist, proprio 14(+2 flags), actions 14."""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import VAPolicyBase, weighted_mse, sinusoidal_embedding
from .dit_flow import DiTFlowPolicy, Attention, modulate
from .tricks import loss_weights

ARM = 7


class CrossArm(nn.Module):
    """Own tokens attend the other stream's tokens; tanh(gain) with gain = 0 makes the block an identity at init."""

    def __init__(self, dim, heads, qk_norm=False):
        super().__init__()
        self.nq = nn.LayerNorm(dim, eps=1e-6)
        self.nk = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads, 0.0, qk_norm)
        self.gain = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        return torch.tanh(self.gain).to(x.dtype) * self.attn(self.nq(x), self.nk(y))


class DuoDiTPolicy(VAPolicyBase):
    n_obs_steps = 1
    stochastic = True

    def __init__(self, action_dim=14, state_dim=14, chunk=50, img_size=224, cross_layers="2,4,6,8", cross_drop=0.0,
                 head_contact=False, head_w=0.2, **kw):
        super().__init__(action_dim, state_dim, chunk, img_size)
        self.head_contact, self.head_w = bool(head_contact), float(head_w)
        self.eval_gate = os.environ.get("DUO_EVAL_GATE", "or")   # evaluation gate: head OR rule | head | rule
        self.last_pred = None
        assert action_dim == 2 * ARM, action_dim
        self.extra = int(state_dim) - 2 * ARM            # 0 or 2 presence flags appended by apply_tricks
        assert self.extra in (0, 2), state_dim
        sub = dict(kw)
        # each stream gets its own 7 joints + BOTH presence flags: identical input layout to a per-arm model trained with
        # the tricks (7 + 2), so R0's left / right checkpoints load into L / R one to one (--init_from)
        sub.update(action_dim=ARM, state_dim=ARM + self.extra, chunk=chunk, img_size=img_size, n_cams=2, aux_flip_weight=0.0)
        self.L = DiTFlowPolicy(**sub)
        self.R = DiTFlowPolicy(**sub)
        self.n_cams = 3
        self.future_grid = self.L.future_grid
        self.future_layer = getattr(self.L, "future_layer", 0)
        self.future_weight = getattr(self.L, "future_weight", 0.0)
        self.sample_steps = self.L.sample_steps
        self.extra_ckpt = dict(self.L.extra_ckpt)
        dim, heads = self.L.act_in.out_features, self.L.blocks[0].attn.h
        qk = not isinstance(self.L.blocks[0].attn.qn, nn.Identity)
        self.cross_layers = sorted({int(x) for x in str(cross_layers).split(",") if str(x).strip()})
        assert all(1 <= i <= len(self.L.blocks) for i in self.cross_layers), self.cross_layers
        self.xL = nn.ModuleDict({str(i): CrossArm(dim, heads, qk) for i in self.cross_layers})
        self.xR = nn.ModuleDict({str(i): CrossArm(dim, heads, qk) for i in self.cross_layers})
        self.cross_drop = float(cross_drop)
        if self.head_contact:   # future-contact head on the two streams' pooled conditioning vectors (same width as the DiT cond)
            cdim = self.L.t_mlp[-1].out_features if isinstance(self.L.t_mlp, nn.Sequential) else dim
            self.contact_head = nn.Sequential(nn.Linear(2 * cdim, 512), nn.GELU(), nn.Linear(512, chunk * 5 + 1))
            nn.init.zeros_(self.contact_head[2].weight); nn.init.zeros_(self.contact_head[2].bias)

    def backbone_modules(self):
        return self.L.backbone_modules() + self.R.backbone_modules()

    # ---- observations
    def _split_state(self, prop):
        if self.extra:
            flags = prop[..., 2 * ARM:2 * ARM + self.extra]
            return torch.cat([prop[..., :ARM], flags], -1), torch.cat([prop[..., ARM:2 * ARM], flags], -1)
        return prop[..., :ARM], prop[..., ARM:2 * ARM]

    def gate_of(self, batch, B, device):
        g = batch.get("contact_gate")
        if g is None:
            c = batch.get("contact")
            g = (c.view(-1) >= 2).float() if c is not None else torch.zeros(B, device=device)
        g = g.to(device).float().view(B)
        if self.training and self.cross_drop > 0:
            g = g * (torch.rand(B, device=device) >= self.cross_drop).float()
        return g

    def encode_observations(self, batch):
        pL, pR = self._split_state(batch["proprio"])
        bL = {"rgb_static": batch["rgb_static"], "rgb_gripper": batch["rgb_gripper"], "proprio": pL}
        bR = {"rgb_static": batch["rgb_static"], "rgb_gripper": batch["rgb_third"], "proprio": pR}
        if "ground_feat" in batch:
            bL["ground_feat"] = batch["ground_feat"]; bR["ground_feat"] = batch["ground_feat"]
        cL, cR = self.L.encode_observations(bL), self.R.encode_observations(bR)
        cL.pop("aux_logit", None); cR.pop("aux_logit", None)
        B = cL["pooled"].shape[0]
        gate = self.gate_of(batch, B, cL["pooled"].device)
        logits = None
        if self.head_contact:
            logits = self.contact_head(torch.cat([cL["pooled"], cR["pooled"]], -1).float())
            if not self.training and self.eval_gate != "rule":
                p = torch.sigmoid(logits[:, -1]); hg = (p > 0.5).float()
                self.last_pred = {"rule_gate": gate.detach().cpu(), "head_gate": hg.detach().cpu(), "p_couple": p.detach().cpu(),
                                  "modes": logits[:, :-1].view(B, self.chunk, 5).argmax(-1).detach().cpu()}
                gate = hg if self.eval_gate == "head" else torch.maximum(gate, hg)
                self.last_pred["gate"] = gate.detach().cpu()
        return {"L": cL, "R": cR, "gate": gate, "logits": logits}

    # ---- denoiser
    @staticmethod
    def _stream_in(P, x, t, cond):
        c = P.t_mlp(sinusoidal_embedding(t * 1000.0, 256).to(x.dtype)) + cond["pooled"]
        h = P.act_in(x) + P.act_pos
        if P.future_grid > 0:
            h = torch.cat([h, P.fut_embed.expand(h.shape[0], -1, -1).to(h.dtype)], dim=1)
        return h, c

    def velocity(self, x_t, t, cond, return_future=False):
        H = x_t.shape[1]
        hL, cL = self._stream_in(self.L, x_t[..., :ARM], t, cond["L"])
        hR, cR = self._stream_in(self.R, x_t[..., ARM:], t, cond["R"])
        g = cond["gate"].view(-1, 1, 1).to(hL.dtype)
        futL = futR = None
        for i, (bL, bR) in enumerate(zip(self.L.blocks, self.R.blocks)):
            hL = bL(hL, cond["L"]["obs"], cL)
            hR = bR(hR, cond["R"]["obs"], cR)
            k = str(i + 1)
            if k in self.xL:
                dL, dR = self.xL[k](hL, hR), self.xR[k](hR, hL)
                hL = hL + g * dL
                hR = hR + g * dR
            if self.future_grid > 0 and i + 1 == self.future_layer:
                futL, futR = hL[:, H:], hR[:, H:]
        outs = []
        for P, h, c in ((self.L, hL, cL), (self.R, hR, cR)):
            s, sc = P.final_ada(c).chunk(2, dim=-1)
            outs.append(P.act_out(modulate(P.final_norm(h[:, :H]), s, sc)))
        v = torch.cat(outs, dim=-1)
        return (v, (futL, futR)) if return_future else v

    def compute_loss(self, batch):
        cond = self.encode_observations(batch)
        actions = batch["actions"]
        w = loss_weights(batch)
        if "loss_step_w" in batch:   # critical-point weighting: transitions of the contact graph inside the chunk
            w = w * batch["loss_step_w"].to(w.dtype).unsqueeze(-1)
        B = actions.shape[0]
        x1 = actions.float(); x0 = torch.randn_like(x1)
        t = self.L._sample_t(B, x1.device)
        x_t = (1 - t.view(B, 1, 1)) * x0 + t.view(B, 1, 1) * x1
        v, (futL, futR) = self.velocity(x_t, t, cond, return_future=True)
        flow = weighted_mse(v, x1 - x0, w)
        loss = flow
        logs = {"flow": float(flow.detach()), "gate": float(cond["gate"].mean())}
        if cond.get("logits") is not None and "contact_future" in batch:
            H = actions.shape[1]; logits = cond["logits"]
            fm = batch["contact_future"].long().view(B, H)
            sw = batch["loss_step_w"].float().view(B, H) if "loss_step_w" in batch else torch.ones(B, H, device=logits.device)
            sw = sw * batch["action_mask"].float().view(B, H)
            ce = (F.cross_entropy(logits[:, :-1].reshape(B * H, 5), fm.reshape(-1), reduction="none").view(B, H) * sw).sum() / sw.sum().clamp_min(1.0)
            ck = batch["contact_gate"].float().view(B)
            bce = F.binary_cross_entropy_with_logits(logits[:, -1], ck)
            loss = loss + self.head_w * (ce + bce)
            with torch.no_grad():
                pred = logits[:, :-1].view(B, H, 5).argmax(-1)
                logs["aux_ce"] = float(ce); logs["aux_bce"] = float(bce)
                logs["head_acc"] = float((((pred == fm).float() * batch["action_mask"].float()).sum() / batch["action_mask"].float().sum().clamp_min(1.0)))
                logs["head_gate_acc"] = float(((torch.sigmoid(logits[:, -1]) > 0.5).float() == ck).float().mean())
        if self.future_grid > 0 and "fut_target" in batch and futL is not None:
            tgt = batch["fut_target"].float(); n = self.future_grid ** 2
            tL = torch.cat([tgt[:, :n], tgt[:, n:2 * n]], 1)          # head + left wrist targets
            tR = torch.cat([tgt[:, :n], tgt[:, 2 * n:3 * n]], 1)      # head + right wrist targets
            fl = 0.5 * (F.mse_loss(self.L.fut_head(futL).float(), tL) + F.mse_loss(self.R.fut_head(futR).float(), tR))
            loss = loss + self.future_weight * fl
            logs["fut"] = float(fl.detach())
        return loss, logs

    # ---- sampling
    @torch.no_grad()
    def sample(self, cond, steps=None, generator=None):
        steps = steps or self.sample_steps
        B, dev = cond["gate"].shape[0], cond["gate"].device
        x = torch.randn(B, self.chunk, self.action_dim, device=dev, generator=generator)
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((B,), k * dt, device=dev)
            x = x + dt * self.velocity(x, t, cond).float()
        return x

    @classmethod
    def _rep_cond(cls, cond, k):
        if k <= 1:
            return cond
        rep = lambda d: {kk: cls._repeat(v, k) for kk, v in d.items()}  # noqa: E731
        return {"L": rep(cond["L"]), "R": rep(cond["R"]), "gate": cls._repeat(cond["gate"], k), "logits": None}

    @torch.no_grad()
    def predict_chunk(self, batch, n_samples: int = 1, generator=None, steps=None):
        was = self.training
        self.eval()
        cond = self.encode_observations(batch)
        B = cond["gate"].shape[0]
        out = self.sample(self._rep_cond(cond, n_samples), steps, generator)
        self.train(was)
        return self._average_samples(out.float(), B, n_samples)
