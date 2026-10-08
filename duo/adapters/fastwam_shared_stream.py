"""[duo.adapters — class body extracted from the patcher, do not edit here]
source : /mnt/cpfs/yangyq/code/duo_fastwam/apply_duo_patch.py  (the string appended to src/fastwam/models/wan22/action_dit.py)
copied : 2026-10-08 (snapshot 2026-09-27)
ledger : duo_fastwam_stage1_open_frozenvideo_fix (10-task subset; trained from the official RoboTwin weights that saw randomized scenes -> not leaderboard protocol)
note   : the SHARED-weights Duo variant: the bimanual chunk becomes 2T tokens through the same ActionDiT blocks, RoPE positions
         0..T-1 for both streams, a learned arm embedding, per-arm in/out layers sliced from the 14-d ones. `duo_cross=True`
         keeps the MoT attention open between the two streams (NO CrossArm modules in this variant); False masks them off.
         Subclass of fastwam.models.wan22.action_dit.ActionDiT — only importable inside the Fast-WAM tree.
"""
import torch
from torch import nn
try:
    from fastwam.models.wan22.action_dit import ActionDiT
except Exception:   # pragma: no cover — documentation import outside the Fast-WAM tree
    ActionDiT = object


class DuoActionDiT(ActionDiT):
    """Duo: the bimanual chunk [B, T, 14] becomes two per-arm token streams (left 7-d, right 7-d), 2T tokens processed by
    the SAME ActionDiT blocks (all weights shared). Both streams reuse RoPE positions 0..T-1, so matching steps of the
    two arms are time-aligned; a learned arm embedding tells the streams apart. Whether the streams attend to each other
    is set in FastWAM._build_mot_attention_mask (`duo_cross`, default True = always).
    The original 14-d `action_encoder` / `head` are kept so every non-Duo checkpoint loads unchanged; they seed the
    per-arm layers (init_arms_from_joint) and otherwise only enter the output as 0 * sum(...) (zero, but a gradient
    exists, so DeepSpeed/DDP never see unused parameters)."""

    ARM = 7
    duo = True
    ACTION_BACKBONE_SKIP_PREFIXES = ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES + (
        "action_encoder_L.", "action_encoder_R.", "head_L.", "head_R.", "arm_embed")

    def __init__(self, *args, duo_cross: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        if self.action_dim != 2 * self.ARM:
            raise ValueError(f"DuoActionDiT needs a 14-d bimanual action, got action_dim={self.action_dim}")
        H = self.hidden_dim
        self.duo_cross = bool(duo_cross)
        self.action_encoder_L = nn.Linear(self.ARM, H)
        self.action_encoder_R = nn.Linear(self.ARM, H)
        self.head_L = nn.Linear(H, self.ARM)
        self.head_R = nn.Linear(H, self.ARM)
        self.arm_embed = nn.Parameter(torch.randn(2, H) * 0.02)
        self.init_arms_from_joint()

    @torch.no_grad()
    def init_arms_from_joint(self):
        """Per-arm layers = the 14-d layers restricted to that arm's 7 dims (encoder columns, head rows)."""
        W, b = self.action_encoder.weight, self.action_encoder.bias
        self.action_encoder_L.weight.copy_(W[:, : self.ARM]); self.action_encoder_L.bias.copy_(b)
        self.action_encoder_R.weight.copy_(W[:, self.ARM:]); self.action_encoder_R.bias.copy_(b)
        Wo, bo = self.head.weight, self.head.bias
        self.head_L.weight.copy_(Wo[: self.ARM]); self.head_L.bias.copy_(bo[: self.ARM])
        self.head_R.weight.copy_(Wo[self.ARM:]); self.head_R.bias.copy_(bo[self.ARM:])

    def num_action_tokens(self, horizon: int) -> int:
        return 2 * int(horizon)

    def pre_dit(self, action_tokens, timestep, context, context_mask=None):
        pre = super().pre_dit(action_tokens=action_tokens, timestep=timestep, context=context, context_mask=context_mask)
        T = int(action_tokens.shape[1])
        emb = self.arm_embed.to(dtype=action_tokens.dtype, device=action_tokens.device)
        tL = self.action_encoder_L(action_tokens[..., : self.ARM]) + emb[0]
        tR = self.action_encoder_R(action_tokens[..., self.ARM:]) + emb[1]
        pre["tokens"] = torch.cat([tL, tR], dim=1)                                   # [B, 2T, H]
        pre["freqs"] = torch.cat([pre["freqs"], pre["freqs"]], dim=0)               # both streams at positions 0..T-1
        pre["context_mask"] = pre["context_mask"][:, :1].expand(-1, 2 * T, -1)
        pre["meta"] = dict(pre["meta"], seq_len=2 * T, horizon=T)
        return pre

    def post_dit(self, tokens, pre_state):
        T = int(pre_state["meta"]["horizon"])
        out = torch.cat([self.head_L(tokens[:, :T]), self.head_R(tokens[:, T:])], dim=-1)   # [B, T, 14]
        keep = (self.action_encoder.weight.sum() + self.action_encoder.bias.sum() + self.head.weight.sum() + self.head.bias.sum())
        return out + 0.0 * keep.to(out.dtype)
