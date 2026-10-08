"""[duo.adapters — verbatim copy, do not edit here]
source : /tmp/robotwin_native_dual/native_dual_dit.py
copied : 2026-10-08 from quic5000 (as it ran; snapshot date 2026-10-06)
ledger : openwam_native_dual_clean2500 (evaluating since 2026-10-08 12:52)
note   : two FULL pretrained OpenWAM ActionDiT copies (left / right), each a 10-d EEF stream in the official 80-d space (slots 0-9 / 34-43); MoT attention mask blocks left<->right; CrossArm(qk_norm)+RoPE after every even layer (15 per direction over 30 layers); warm start slices action_encoder columns / action_decoder rows. Installed into the OpenWAM tree by legacy/openwam_install_native_dual.py. /tmp on quic5000 is NOT persistent.
Edit the canonical copy in the training workspace, then re-copy; the ledger row points at this file + the source path.
"""
"""Two pretrained OpenWAM action experts with sparse cross-arm attention."""

from copy import copy
from dataclasses import dataclass
import sys

import torch
from torch import nn

from openwam.model.action_backbone.base import ActionDiTBackbone
from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.architectures.base import ActionState
from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver

sys.path.insert(0, '/mnt/workspace/zzm/duo_dit')
from models.duo_dit import CrossArm  # noqa: E402
from action_rope import enable_rope  # noqa: E402


@dataclass
class DualPayload:
    x_action: torch.Tensor
    left: ActionState
    right: ActionState


class NativeDualActionDiT(ActionDiTBackbone):
    """Native MoT blocks with two 10D EEF streams in the official 80D space."""

    def __init__(self, *, action_dim, dim, ffn_dim, num_heads, num_layers,
                 video_dim, bridge_layers, attn_head_dim, text_dim=4096,
                 shift_action=5.0, max_action_len=50, **_):
        super().__init__()
        if action_dim != 80:
            raise ValueError(f"Expected the official 80D action contract, got {action_dim}")
        self._shift_action = shift_action
        self._num_heads = num_heads
        self._head_dim = attn_head_dim
        self._num_layers = num_layers
        self._bridge_layers = tuple(bridge_layers)
        args = dict(action_dim=10, dim=dim, ffn_dim=ffn_dim, num_heads=num_heads,
                    num_layers=num_layers, video_dim=video_dim, bridge_layers=bridge_layers,
                    variant="joint_self_attn", attn_head_dim=attn_head_dim,
                    text_dim=text_dim, shift_action=shift_action,
                    max_action_len=max_action_len)
        self.left = ActionDiT(**args)
        self.right = ActionDiT(**args)
        self.left_cross = nn.ModuleDict()
        self.right_cross = nn.ModuleDict()
        for layer in range(1, num_layers + 1):
            if layer % 2 == 0:
                self.left_cross[str(layer)] = CrossArm(dim, 8, qk_norm=True)
                self.right_cross[str(layer)] = CrossArm(dim, 8, qk_norm=True)
        for cross in (*self.left_cross.values(), *self.right_cross.values()):
            enable_rope(cross.attn)

    @property
    def num_heads(self):
        return self._num_heads

    @property
    def head_dim(self):
        return self._head_dim

    @property
    def num_layers(self):
        return self._num_layers

    def forward(self, *args, **kwargs):
        raise RuntimeError("NativeDualActionDiT runs through the MoT driver")

    def prepare_state(self, noisy_actions, timestep, **kwargs):
        left = self.left.prepare_state(noisy_actions[..., :10], timestep, **kwargs)
        right = self.right.prepare_state(noisy_actions[..., 34:44], timestep, **kwargs)
        x = torch.cat((left.payload.x_action, right.payload.x_action), dim=1)
        return ActionState(noisy_actions, timestep, DualPayload(x, left, right))

    def _arm_states(self, astate):
        payload = astate.payload
        xl, xr = payload.x_action.chunk(2, dim=1)
        states = []
        for template, x in ((payload.left, xl), (payload.right, xr)):
            state = copy(template)
            state.payload = copy(template.payload)
            state.payload.x_action = x
            states.append(state)
        return states

    def pre_attn_at_layer(self, layer_id, astate):
        left, right = self._arm_states(astate)
        ql, kl, vl, pl = self.left.pre_attn_at_layer(layer_id, left)
        qr, kr, vr, pr = self.right.pre_attn_at_layer(layer_id, right)
        return (torch.cat((ql, qr), 1), torch.cat((kl, kr), 1),
                torch.cat((vl, vr), 1), (left, right, pl, pr))

    def post_attn_at_layer(self, layer_id, astate, attn_out, post_state):
        left, right, pl, pr = post_state
        out_l, out_r = attn_out.chunk(2, dim=1)
        xl = self.left.post_attn_at_layer(layer_id, left, out_l, pl).payload.x_action
        xr = self.right.post_attn_at_layer(layer_id, right, out_r, pr).payload.x_action
        key = str(layer_id + 1)
        if key in self.left_cross:
            dl = self.left_cross[key](xl, xr)
            dr = self.right_cross[key](xr, xl)
            xl, xr = xl + dl, xr + dr
        astate.payload.x_action = torch.cat((xl, xr), 1)
        return astate

    def extract_prediction(self, astate):
        xl, xr = astate.payload.x_action.chunk(2, dim=1)
        prediction = torch.zeros_like(astate.action_latents)
        prediction[..., :10] = self.left.action_decoder(xl)
        prediction[..., 34:44] = self.right.action_decoder(xr)
        return prediction


def native_dual_warm_start(source, target):
    """Copy native layers and select each arm's pretrained EEF input/output slots."""
    compatible = {}
    for key, value in target.items():
        source_key = key
        for arm, start in (("left", 0), ("right", 34)):
            prefix = f"action_backbone.{arm}."
            if key.startswith(prefix):
                source_key = key.replace(prefix, "action_backbone.", 1)
                break
        else:
            start = None
        if source_key not in source:
            if key.startswith(("action_backbone.left_cross.", "action_backbone.right_cross.")):
                continue
            raise ValueError(f"Pretrained tensor missing: {source_key}")
        weight = source[source_key]
        if start is not None:
            if source_key == "action_backbone.action_encoder.weight":
                weight = weight[:, start:start + 10]
            elif source_key in ("action_backbone.action_decoder.weight", "action_backbone.action_decoder.bias"):
                weight = weight[start:start + 10]
        if weight.shape != value.shape:
            raise ValueError(f"Pretrained shape mismatch: {key}: {weight.shape} vs {value.shape}")
        compatible[key] = weight
    return compatible


class NativeDualMoTDriver(DualSystemMoTDriver):
    def _build_attention_mask(self, s_video, s_action, video_tokens_per_frame, *, device):
        if s_action % 2:
            raise ValueError(f"Two action streams require even token count, got {s_action}")
        mask = super()._build_attention_mask(s_video, s_action, video_tokens_per_frame, device=device)
        half = s_action // 2
        left = slice(s_video, s_video + half)
        right = slice(s_video + half, s_video + s_action)
        mask[left, right] = False
        mask[right, left] = False
        return mask
