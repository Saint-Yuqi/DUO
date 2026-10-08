"""[duo.adapters — verbatim copy, do not edit here]
source : /mnt/workspace/yangyq/flower_duodit_robotwin/flower_duodit.py
copied : 2026-10-08 from quic5000 (as it ran; snapshot date 2026-10-05)
ledger : flower_duodit_clean2500 (68.30 / 26.62 / 47.46)
note   : pretrained FLOWER FlowBlock stream (12 x 1024) deep-copied for the right arm; CrossArm = nn.MultiheadAttention (no qk-norm, no RoPE) at blocks 2,4,6,8; NO gate (always coupled); imports flower_bimanual_v3.model.FlowerBimanual
Edit the canonical copy in the training workspace, then re-copy; the ledger row points at this file + the source path.
"""
"""Florence-2 FLOWER backbone with two 7-D DiT streams and cross-arm attention."""
import copy
from contextlib import nullcontext
import sys

import torch
from torch import nn

sys.path.insert(0, "/mnt/workspace/yangyq/code/flower_bimanual_v3")
from model import FlowerBimanual  # noqa: E402
from flower.models.networks.transformers import stateless_norm  # noqa: E402


class CrossArm(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(self, query, context):
        delta = self.attn(self.query_norm(query), self.context_norm(context),
                          self.context_norm(context), need_weights=False)[0]
        return torch.tanh(self.gain) * delta


class FlowerDuoDiT(FlowerBimanual):
    def __init__(self, vlm_path, chunk=50, freeze_vlm=False, dit_dim=1024,
                 n_layers=12, n_heads=16, sample_steps=4, pretrained_head=None):
        super().__init__(vlm_path=vlm_path, action_dim=14, state_dim=14, chunk=chunk,
                         freeze_vlm=freeze_vlm, proprio_dropout=0.0, token_dropout=0.1,
                         dit_dim=dit_dim, n_layers=n_layers, n_heads=n_heads,
                         num_sampling_steps=sample_steps, control_hz=30.0,
                         robot_name="Aloha AgileX", action_space_desc="Joint Position")
        self.pretrain_report = self.load_pretrained_head(pretrained_head) if pretrained_head else None
        self.right_dit = copy.deepcopy(self.dit)
        self.left_adaln = copy.deepcopy(self.adaln["bimanual_nav"])
        self.right_adaln = copy.deepcopy(self.adaln["bimanual_nav"])
        self.left_action_in = nn.Sequential(nn.Linear(7, dit_dim), nn.GELU(), nn.Linear(dit_dim, dit_dim))
        self.right_action_in = copy.deepcopy(self.left_action_in)
        self.left_action_out = nn.Linear(dit_dim, 7)
        self.right_action_out = nn.Linear(dit_dim, 7)
        self.left_proprio = nn.Sequential(nn.Linear(7, dit_dim), nn.GELU(), nn.Linear(dit_dim, dit_dim))
        self.right_proprio = copy.deepcopy(self.left_proprio)
        layers = (2, 4, 6, 8)
        self.left_cross = nn.ModuleDict({str(i): CrossArm(dit_dim, n_heads) for i in layers if i <= n_layers})
        self.right_cross = nn.ModuleDict({str(i): CrossArm(dit_dim, n_heads) for i in layers if i <= n_layers})
        # The parent's joint-action modules are replaced by the two 7-D heads.
        for bank in (self.action_encoders, self.action_decoders, self.proprio_encoders, self.adaln):
            for parameter in bank.parameters():
                parameter.requires_grad = False

    def encode_observations(self, batch):
        device = self.device
        dtype = next(self.dit.parameters()).dtype
        head = self._prep_images(batch["rgb_static"].to(device, non_blocking=True))
        left = self._prep_images(batch["rgb_gripper"].to(device, non_blocking=True))
        right = self._prep_images(batch["rgb_third"].to(device, non_blocking=True))
        prompts = [self.format_instruction(text) for text in batch["lang_text"]]
        with torch.no_grad() if self.freeze_vlm else nullcontext():
            images = [self.vlm._encode_image(image.to(dtype)) for image in (head, left, right)]
            text = self._get_text_embeddings(prompts, device).to(images[0].dtype)
            prompt = self.prompt_embeds.expand(len(head), -1, -1).to(device=device, dtype=text.dtype)
            features = []
            for wrist in (images[1], images[2]):
                merged = torch.cat((images[0], wrist, prompt, text), dim=1)
                attention = torch.ones(merged.shape[:2], device=device)
                encoded = self.vlm.get_encoder()(inputs_embeds=merged, attention_mask=attention).last_hidden_state
                features.append(self.cond_linear(self.cond_norm(encoded)))
        state = batch["proprio"].to(device, non_blocking=True).float()
        assert state.shape == (len(head), 14)
        frequency = self.frequency_embedder(torch.full((len(head),), 30.0, device=device))
        return {"left": self.vlm_token_dropout(features[0]),
                "right": self.vlm_token_dropout(features[1]),
                "state": state, "frequency": frequency}

    def velocity(self, x, t, cond):
        dtype = next(self.dit.parameters()).dtype
        x = x.to(dtype)
        state = cond["state"].to(dtype)
        frequency = cond["frequency"].to(dtype)
        timing = stateless_norm(self.t_embedder(t.to(dtype))) + stateless_norm(frequency)
        left_condition = timing + stateless_norm(self.left_proprio(state[:, :7]))
        right_condition = timing + stateless_norm(self.right_proprio(state[:, 7:]))
        h_left = self.left_action_in(x[:, :, :7])
        h_right = self.right_action_in(x[:, :, 7:])
        left_adaln = self.left_adaln(left_condition)
        right_adaln = self.right_adaln(right_condition)
        for i, (left_block, right_block) in enumerate(zip(self.dit, self.right_dit), 1):
            h_left = left_block(h_left, left_condition, context=cond["left"],
                                is_causal=True, global_adaln=left_adaln)
            h_right = right_block(h_right, right_condition, context=cond["right"],
                                  is_causal=True, global_adaln=right_adaln)
            key = str(i)
            if key in self.left_cross:
                delta_left = self.left_cross[key](h_left, h_right)
                delta_right = self.right_cross[key](h_right, h_left)
                h_left, h_right = h_left + delta_left, h_right + delta_right
        return torch.cat((self.left_action_out(h_left), self.right_action_out(h_right)), dim=-1)

    def flow_loss(self, cond, actions, mask=None):
        actions = actions.to(device=self.device, dtype=next(self.dit.parameters()).dtype)
        batch_size = actions.shape[0]
        t = (torch.rand(1, device=actions.device) +
             torch.arange(batch_size, device=actions.device) / batch_size) % (1 - 1e-5)
        noise = torch.randn_like(actions)
        noisy = (1 - t[:, None, None]) * actions + t[:, None, None] * noise
        error = (self.velocity(noisy, t, cond).float() - (noise - actions).float()).square()
        if mask is None:
            return error.mean()
        weight = mask.to(device=error.device, dtype=error.dtype).unsqueeze(-1)
        return (error * weight).sum() / (weight.sum() * 14).clamp_min(1)

    def forward(self, batch):
        return self.flow_loss(self.encode_observations(batch), batch["actions"], batch["action_mask"])

    @torch.no_grad()
    def sample(self, cond, steps=None):
        steps = steps or self.num_sampling_steps
        batch_size = cond["state"].shape[0]
        x = torch.randn(batch_size, self.chunk, 14, device=cond["state"].device,
                        dtype=next(self.dit.parameters()).dtype)
        for i in range(steps, 0, -1):
            t = torch.full((batch_size,), i / steps, device=x.device, dtype=x.dtype)
            x = x - self.velocity(x, t, cond) / steps
        return x.clamp(-1, 1)
