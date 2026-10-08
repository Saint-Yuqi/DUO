"""[duo.adapters — verbatim copy, do not edit here]
source : /mnt/cpfs/yangyq/code/duo_fastwam/apply_duo_patch.py
copied : 2026-10-08 from quic5000 (as it ran; snapshot date 2026-09-27)
ledger : duo_fastwam_stage1_open_frozenvideo_fix (10-task subset, not leaderboard protocol)
note   : idempotent patcher that adds DuoActionDiT to Fast-WAM's action_dit.py and the duo hooks to fastwam.py; the class body is also in fastwam_shared_stream.py
Edit the canonical copy in the training workspace, then re-copy; the ledger row points at this file + the source path.
"""
#!/usr/bin/env python3
"""Turn our copy of Fast-WAM (/mnt/cpfs/yangyq/code/duo_fastwam) into Duo-FastWAM. Idempotent; asserts every anchor."""
import pathlib, shutil
ROOT = pathlib.Path("/mnt/cpfs/yangyq/code/duo_fastwam")
AD = ROOT / "src/fastwam/models/wan22/action_dit.py"
FW = ROOT / "src/fastwam/models/wan22/fastwam.py"
for p in (AD, FW):
    bak = p.with_suffix(".py.orig")
    if not bak.exists():
        shutil.copy2(p, bak)

s = AD.read_text()
if "class DuoActionDiT" not in s:
    a = "    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:\n        return self.head(tokens)\n"
    assert s.count(a) == 1
    s = s.replace(a, a + "\n    def num_action_tokens(self, horizon: int) -> int:\n        return int(horizon)\n")
    s += '''

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
'''
    AD.write_text(s)
    print("action_dit.py: DuoActionDiT added")
else:
    print("action_dit.py: already patched")

f = FW.read_text()
if "DuoActionDiT" not in f:
    rep = [
        ("from .action_dit import ActionDiT\n", "from .action_dit import ActionDiT, DuoActionDiT\n"),
        ("""        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,""",
         """        _action_cfg = dict(action_dit_config or {})
        _duo = bool(_action_cfg.pop("duo", False))
        if not _duo:
            _action_cfg.pop("duo_cross", None)
        action_expert = (DuoActionDiT if _duo else ActionDiT).from_pretrained(
            action_dit_config=_action_cfg,"""),
        ("""        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
""", """        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        ae = self.action_expert
        if getattr(ae, "duo", False) and not getattr(ae, "duo_cross", True):   # Duo, arms never see each other
            half = action_seq_len // 2
            a0 = video_seq_len
            mask[a0:a0 + half, a0 + half:] = False
            mask[a0 + half:, a0:a0 + half] = False
"""),
        ("            action_seq_len=latents_action.shape[1],\n",
         "            action_seq_len=self.action_expert.num_action_tokens(latents_action.shape[1]),\n"),
        ("""        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
""", """        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
            if getattr(self.action_expert, "duo", False) and not any("action_encoder_L" in k for k in payload["mot"]):
                self.action_expert.init_arms_from_joint()
                logger.info("Duo: non-Duo checkpoint %s -> per-arm action layers initialised from the 14-d ones", path)
"""),
    ]
    for a, b in rep:
        assert f.count(a) == 1, a[:70]
        f = f.replace(a, b)
    FW.write_text(f)
    print("fastwam.py: Duo hooks added")
else:
    print("fastwam.py: already patched")

cfg = ROOT / "configs/model/fastwam.yaml"; duo_cfg = ROOT / "configs/model/fastwam_duo.yaml"
c = cfg.read_text()
anchor = "action_dit_config:\n"
assert c.count(anchor) == 1
duo_cfg.write_text(c.replace(anchor, anchor + "  duo: true\n  duo_cross: true   # the two arm streams always attend to each other (false = never)\n"))
print("configs/model/fastwam_duo.yaml written")
