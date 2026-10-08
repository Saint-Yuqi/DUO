"""[duo.adapters — verbatim copy, do not edit here]
source : /mnt/workspace/yangyq/flower_customduodit_robotwin/custom_dit.py
copied : 2026-10-08 from quic5000 (as it ran; snapshot date 2026-10-05)
ledger : flower_customduodit_clean2500 (65.38 / 18.84 / 42.11)
note   : va_bimanual_b DiTBlock streams (8 x 512) from scratch + CrossArm(qk_norm) at 2,4,6,8, gate from batch (open when absent); the Florence-2 encoder is run once per wrist (head + own wrist + prompt + text). duo.head.DuoFlowHead is the library form of ArmHead + this velocity loop.
Edit the canonical copy in the training workspace, then re-copy; the ledger row points at this file + the source path.
"""
"""Florence-2 observation encoder with the independently implemented DuoDiT action head."""
import sys
from contextlib import nullcontext

import torch
from torch import nn

sys.path.insert(0, "/mnt/workspace/yangyq/code/flower_bimanual_v3")
from model import FlowerBimanual  # noqa: E402
sys.path.insert(0, "/mnt/workspace/yangyq/code/va_bimanual_b")
from models.common import sinusoidal_embedding  # noqa: E402
from models.dit_flow import DiTBlock, modulate  # noqa: E402
from models.duo_dit import CrossArm  # noqa: E402


class ArmHead(nn.Module):
    """The action-token path of va_bimanual_b's DiTFlowPolicy, with FLOWER tokens as observations."""

    def __init__(self, chunk=50, dim=512, layers=8, heads=8):
        super().__init__()
        self.obs_norm = nn.LayerNorm(dim, eps=1e-6)
        self.obs_pool = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.proprio_mlp = nn.Sequential(nn.Linear(7, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.act_in = nn.Linear(7, dim)
        self.act_pos = nn.Parameter(torch.randn(1, chunk, dim) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([DiTBlock(dim, heads, qk_norm=True) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.act_out = nn.Linear(dim, 7)
        nn.init.zeros_(self.final_ada[1].weight)
        nn.init.zeros_(self.final_ada[1].bias)
        nn.init.zeros_(self.act_out.weight)
        nn.init.zeros_(self.act_out.bias)

    def condition(self, obs, state, t):
        obs = self.obs_norm(obs)
        pooled = self.obs_pool(obs.mean(1)) + self.proprio_mlp(state)
        timing = self.t_mlp(sinusoidal_embedding(t * 1000, 256).to(obs.dtype))
        return obs, pooled + timing

    def finish(self, hidden, condition):
        shift, scale = self.final_ada(condition).chunk(2, dim=-1)
        return self.act_out(modulate(self.final_norm(hidden), shift, scale))


class FlowerDuoDiT(FlowerBimanual):
    def __init__(self, vlm_path, chunk=50, freeze_vlm=False, dit_dim=512,
                 n_layers=8, n_heads=8, sample_steps=10, pretrained_head=None):
        if pretrained_head:
            raise ValueError("FLOWER action-head weights cannot initialize the custom DuoDiT blocks")
        super().__init__(vlm_path=vlm_path, action_dim=14, state_dim=14, chunk=chunk,
                         freeze_vlm=freeze_vlm, use_proprio=False, token_dropout=0.1,
                         dit_dim=dit_dim, n_layers=1, n_heads=n_heads,
                         num_sampling_steps=sample_steps, control_hz=30.0,
                         robot_name="Aloha AgileX", action_space_desc="Joint Position")
        self.pretrain_report = None
        self.dit = nn.ModuleList()  # discard FLOWER's action blocks; only Florence-2 remains
        for bank in (self.action_encoders, self.action_decoders, getattr(self, "proprio_encoders", None),
                     self.adaln, self.t_embedder, self.frequency_embedder, self.action_space_embedder):
            if bank is not None:
                for parameter in bank.parameters():
                    parameter.requires_grad = False
        self.left = ArmHead(chunk, dit_dim, n_layers, n_heads)
        self.right = ArmHead(chunk, dit_dim, n_layers, n_heads)
        layers = (2, 4, 6, 8)
        self.left_cross = nn.ModuleDict({str(i): CrossArm(dit_dim, n_heads, qk_norm=True)
                                         for i in layers if i <= n_layers})
        self.right_cross = nn.ModuleDict({str(i): CrossArm(dit_dim, n_heads, qk_norm=True)
                                          for i in layers if i <= n_layers})

    def encode_observations(self, batch):
        device = self.device
        head = self._prep_images(batch["rgb_static"].to(device, non_blocking=True))
        left = self._prep_images(batch["rgb_gripper"].to(device, non_blocking=True))
        right = self._prep_images(batch["rgb_third"].to(device, non_blocking=True))
        prompts = [self.format_instruction(text) for text in batch["lang_text"]]
        with torch.no_grad() if self.freeze_vlm else nullcontext():
            images = [self.vlm._encode_image(image) for image in (head, left, right)]
            text = self._get_text_embeddings(prompts, device).to(images[0].dtype)
            prompt = self.prompt_embeds.expand(len(head), -1, -1).to(device=device, dtype=text.dtype)
            obs = []
            for wrist in (images[1], images[2]):
                merged = torch.cat((images[0], wrist, prompt, text), dim=1)
                attention = torch.ones(merged.shape[:2], device=device)
                encoded = self.vlm.get_encoder()(inputs_embeds=merged, attention_mask=attention).last_hidden_state
                obs.append(self.vlm_token_dropout(self.cond_linear(self.cond_norm(encoded))))
        state = batch["proprio"].to(device, non_blocking=True).float()
        assert state.shape == (len(head), 14)
        gate = batch.get("contact_gate")
        # Open cross attention until RoboTwin contact labels are supplied.
        gate = torch.ones(len(head), device=device) if gate is None else gate.to(device).float().view(-1)
        return {"left": obs[0], "right": obs[1], "state": state, "gate": gate}

    def velocity(self, actions, t, cond):
        state = cond["state"].to(actions.dtype)
        obs_l, c_l = self.left.condition(cond["left"], state[:, :7], t)
        obs_r, c_r = self.right.condition(cond["right"], state[:, 7:], t)
        h_l = self.left.act_in(actions[:, :, :7]) + self.left.act_pos
        h_r = self.right.act_in(actions[:, :, 7:]) + self.right.act_pos
        gate = cond["gate"].view(-1, 1, 1).to(h_l.dtype)
        for i, (block_l, block_r) in enumerate(zip(self.left.blocks, self.right.blocks), 1):
            h_l = block_l(h_l, obs_l, c_l)
            h_r = block_r(h_r, obs_r, c_r)
            key = str(i)
            if key in self.left_cross:
                delta_l = self.left_cross[key](h_l, h_r)
                delta_r = self.right_cross[key](h_r, h_l)
                h_l, h_r = h_l + gate * delta_l, h_r + gate * delta_r
        return torch.cat((self.left.finish(h_l, c_l), self.right.finish(h_r, c_r)), dim=-1)

    def forward(self, batch):
        cond = self.encode_observations(batch)
        actions = batch["actions"].to(cond["state"].device).float()
        noise = torch.randn_like(actions)
        t = torch.rand(len(actions), device=actions.device).clamp(1e-3, 1 - 1e-3)
        noisy = (1 - t[:, None, None]) * noise + t[:, None, None] * actions
        error = (self.velocity(noisy, t, cond).float() - (actions - noise)).square()
        mask = batch["action_mask"].to(error.device, error.dtype).unsqueeze(-1)
        return (error * mask).sum() / (mask.sum() * 14).clamp_min(1)

    @torch.no_grad()
    def sample(self, cond, steps=None):
        steps = steps or self.num_sampling_steps
        x = torch.randn(cond["state"].shape[0], self.chunk, 14, device=cond["state"].device)
        for i in range(steps):
            t = torch.full((len(x),), i / steps, device=x.device)
            x = x + self.velocity(x, t, cond).float() / steps
        return x.clamp(-1, 1)
