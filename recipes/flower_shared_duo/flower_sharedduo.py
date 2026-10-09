"""FLOWER + Duo, SHARED-weight variant (the π0.5-Duo recipe transplanted onto FLOWER).

One pretrained FLOWER action expert (12 x 1024 FlowBlocks, author checkpoint 360000_model_weights.pt) processes BOTH arms:

    suffix tokens  [ s_L, a_L(0..H-1), s_R, a_R(0..H-1) ]          (2 * (1 + H) = 102 tokens for H = 50)
    rope position  [ 0,   1 .. H,      0,   1 .. H     ]          both arms share the time axis

  * a_L / a_R : per-arm action-in MLPs = the pretrained 16-d bimanual encoder with fc1 restricted to that arm's 7 columns
                (fc2 copied); per-arm action-out = the pretrained 16-d decoder restricted to that arm's 7 rows.
  * s_L / s_R : own 7-d state + presence flag -> one token (new Linear, zero-initialised, as in π0.5-Duo). The proprio no
                longer enters the adaLN conditioning, so each arm's raw state enters only through its own token.
  * stream embedding (2 x 1024, N(0, 0.02)) added to every token of a stream.
  * self-attention mask: a token at time p may attend tokens with time <= p (FLOWER's pretrained causal rule, now over
    both arms). Gate open (default, = the official π0.5-Duo run) -> the two streams see each other; gate closed -> each
    stream sees only itself (two per-arm experts with shared weights).
  * observations: ONE Florence-2 pass over [head, left wrist, right wrist, prompt, text] shared by both streams (π0.5-Duo
    also shares the VLM prefix). FlowerDuoDiT ran Florence twice (head + own wrist per stream).
  * no CrossArm modules, no tanh gains: nothing to "open"; parameter count ≈ the single-stream FLOWER head (~438 M).
Training / sampling conventions are identical to flower_duodit.py (noisy = (1-t) a + t eps, target eps - a, Euler from t=1).
"""
import copy
import sys
import types
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, "/mnt/workspace/yangyq/code/flower_bimanual_v3")
from model import FlowerBimanual  # noqa: E402
from flower.models.networks.transformers import stateless_norm, apply_rotary_pos_emb  # noqa: E402

ARM = 7


def _rope_self_attn_forward(self, x, custom_attn_mask=None, is_causal=False):
    """FlowerAttention.forward with explicit RoPE position ids (self._duo_pos, shape (T,)); otherwise identical."""
    B, T, C = x.size()
    qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q = self.q_norm(q)
    k = self.k_norm(k)
    q, k = apply_rotary_pos_emb(q, k, self.cos, self.sin, position_ids=self._duo_pos)
    mask = None if custom_attn_mask is None else custom_attn_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None if mask is None else ~mask,
                                         dropout_p=self.attn_dropout.p if self.training else 0.0, scale=self.scale)
    out = out.transpose(1, 2).reshape(B, T, C)
    return self.resid_dropout(self.proj(out))


class FlowerSharedDuo(FlowerBimanual):
    def __init__(self, vlm_path, chunk=50, freeze_vlm=False, dit_dim=1024, n_layers=12, n_heads=16,
                 sample_steps=4, pretrained_head=None, gate="open"):
        assert gate in ("open", "closed"), gate
        super().__init__(vlm_path=vlm_path, action_dim=14, state_dim=14, chunk=chunk,
                         freeze_vlm=freeze_vlm, proprio_dropout=0.0, token_dropout=0.1,
                         dit_dim=dit_dim, n_layers=n_layers, n_heads=n_heads,
                         num_sampling_steps=sample_steps, control_hz=30.0,
                         robot_name="Aloha AgileX", action_space_desc="Joint Position")
        self.gate_mode = gate
        self.pretrain_report = self.load_pretrained_head(pretrained_head) if pretrained_head else None
        assert self.dit[0].self_attn.cos.shape[0] >= chunk + 1, "RoPE table too short for 1 + chunk positions"
        enc, dec = self.action_encoders["bimanual_nav"], self.action_decoders["bimanual_nav"]
        self.action_in_L, self.action_in_R = copy.deepcopy(enc), copy.deepcopy(enc)
        self.action_out_L = nn.Linear(dit_dim, ARM)
        self.action_out_R = nn.Linear(dit_dim, ARM)
        self.state_in_L = nn.Linear(ARM + 1, dit_dim)
        self.state_in_R = nn.Linear(ARM + 1, dit_dim)
        self.stream_embed = nn.Parameter(torch.zeros(2, dit_dim))
        nn.init.normal_(self.stream_embed, std=0.02)
        self.init_streams_from_joint()
        for blk in self.dit:
            blk.self_attn.forward = types.MethodType(_rope_self_attn_forward, blk.self_attn)
            blk.self_attn._duo_pos = None
        # the joint 16-d encoder / decoder / proprio MLP / per-type adaLN banks are replaced -> frozen (kept so that the
        # checkpoint layout stays FLOWER's); the bimanual adaLN is still used and trainable.
        for bank in (self.action_encoders, self.action_decoders, self.proprio_encoders):
            for p in bank.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def init_streams_from_joint(self):
        enc, dec = self.action_encoders["bimanual_nav"], self.action_decoders["bimanual_nav"]
        for mod, cols in ((self.action_in_L, slice(0, ARM)), (self.action_in_R, slice(ARM, 2 * ARM))):
            fc1 = nn.Linear(ARM, enc.fc1.out_features).to(enc.fc1.weight.dtype)
            fc1.weight.copy_(enc.fc1.weight[:, cols]); fc1.bias.copy_(enc.fc1.bias)
            mod.fc1 = fc1                       # fc2 is the pretrained one (deepcopy)
        self.action_out_L.weight.copy_(dec.weight[:ARM]); self.action_out_L.bias.copy_(dec.bias[:ARM])
        self.action_out_R.weight.copy_(dec.weight[ARM:2 * ARM]); self.action_out_R.bias.copy_(dec.bias[ARM:2 * ARM])
        for m in (self.state_in_L, self.state_in_R):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def encode_observations(self, batch):
        device = self.device
        dtype = next(self.dit.parameters()).dtype
        cams = [self._prep_images(batch[k].to(device, non_blocking=True)) for k in ("rgb_static", "rgb_gripper", "rgb_third")]
        prompts = [self.format_instruction(text) for text in batch["lang_text"]]
        with torch.no_grad() if self.freeze_vlm else nullcontext():
            images = [self.vlm._encode_image(c.to(dtype)) for c in cams]
            text = self._get_text_embeddings(prompts, device).to(images[0].dtype)
            prompt = self.prompt_embeds.expand(len(cams[0]), -1, -1).to(device=device, dtype=text.dtype)
            merged = torch.cat((*images, prompt, text), dim=1)
            attention = torch.ones(merged.shape[:2], device=device)
            encoded = self.vlm.get_encoder()(inputs_embeds=merged, attention_mask=attention).last_hidden_state
            context = self.cond_linear(self.cond_norm(encoded))
        state = batch["proprio"].to(device, non_blocking=True).float()
        assert state.shape == (len(cams[0]), 14)
        B = state.shape[0]
        g = batch.get("duo_gate")
        if g is None:
            g = torch.full((B,), 1.0 if self.gate_mode == "open" else 0.0, device=device)
        frequency = self.frequency_embedder(torch.full((B,), 30.0, device=device))
        return {"context": self.vlm_token_dropout(context), "state": state, "frequency": frequency,
                "gate": g.to(device).float().view(B)}

    def attention_mask(self, gate, H, device):
        """(B, L, L) bool, True = blocked (FlowerAttention convention). Time-causal over both arms; cross-stream blocks
        follow the per-sample gate."""
        S = H + 1
        pos = torch.arange(S, device=device)
        later = pos[None, :] > pos[:, None]                    # key time > query time -> blocked
        B = gate.shape[0]
        m = torch.ones(B, 2 * S, 2 * S, dtype=torch.bool, device=device)
        m[:, :S, :S] = later; m[:, S:, S:] = later
        closed = (gate <= 0.5).view(B, 1, 1)
        cross = later.unsqueeze(0) | closed
        m[:, :S, S:] = cross; m[:, S:, :S] = cross
        return m

    def velocity(self, x, t, cond):
        dtype = next(self.dit.parameters()).dtype
        B, H, _ = x.shape
        x = x.to(dtype)
        state = cond["state"].to(dtype)
        one = torch.ones(B, 1, device=x.device, dtype=dtype)
        timing = stateless_norm(self.t_embedder(t.to(dtype))) + stateless_norm(cond["frequency"].to(dtype))
        global_adaln = self.adaln["bimanual_nav"](timing)
        e0, e1 = self.stream_embed[0].to(dtype), self.stream_embed[1].to(dtype)
        sL = self.state_in_L(torch.cat([state[:, :ARM], one], -1))[:, None]
        sR = self.state_in_R(torch.cat([state[:, ARM:], one], -1))[:, None]
        h = torch.cat([sL + e0, self.action_in_L(x[..., :ARM]) + e0, sR + e1, self.action_in_R(x[..., ARM:]) + e1], 1)
        pos = torch.arange(H + 1, device=x.device).repeat(2)
        for blk in self.dit:
            blk.self_attn._duo_pos = pos
        mask = self.attention_mask(cond["gate"], H, x.device)
        for blk in self.dit:
            h = blk(h, timing, context=cond["context"].to(dtype), custom_attn_mask=mask, global_adaln=global_adaln)
        return torch.cat((self.action_out_L(h[:, 1:H + 1]), self.action_out_R(h[:, H + 2:])), dim=-1)

    def flow_loss(self, cond, actions, mask=None):
        actions = actions.to(device=self.device, dtype=next(self.dit.parameters()).dtype)
        B = actions.shape[0]
        t = (torch.rand(1, device=actions.device) + torch.arange(B, device=actions.device) / B) % (1 - 1e-5)
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
        B = cond["state"].shape[0]
        x = torch.randn(B, self.chunk, 14, device=cond["state"].device, dtype=next(self.dit.parameters()).dtype)
        for i in range(steps, 0, -1):
            t = torch.full((B,), i / steps, device=x.device, dtype=x.dtype)
            x = x - self.velocity(x, t, cond) / steps
        return x.clamp(-1, 1)
