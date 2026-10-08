"""duo — the Duo (arm-factorised, gated two-stream) action head as a backbone-agnostic library.

The pieces every host backbone needs:
  partition   which action dimensions form which stream (arm-semantic / mixed / OpenWAM 80-d EEF slots / joint)
  layers      RMSNorm, Attention (qk-norm, optional RoPE), adaLN-zero DiTBlock
  coupling    CrossArm (tanh-gain, zero-initialised) and CrossArmSet (per-layer, per-direction, gated), MoT mask helper
  init        warm start of two streams from one pretrained single-stream expert (encoder columns / decoder rows)
  head        DuoFlowHead: the complete from-scratch reference head (flow matching, N streams, Euler sampler)

`duo.adapters` keeps the exact code that produced every ledger row (one file per host backbone, provenance in the header).
"""
from .partition import Partition
from .layers import RMSNorm, Attention, DiTBlock, modulate, sinusoidal_embedding
from .rope import RoPEAttention, enable_rope, rotate
from .coupling import CrossArm, CrossArmSet, block_cross_stream_attention, gate_from_batch
from .init import warm_start_from_single_stream, slice_linear_in, slice_linear_out
from .head import ArmStream, DuoFlowHead

__version__ = "0.1.0"
__all__ = ["Partition", "RMSNorm", "Attention", "DiTBlock", "modulate", "sinusoidal_embedding", "RoPEAttention", "enable_rope",
           "rotate", "CrossArm", "CrossArmSet", "block_cross_stream_attention", "gate_from_batch", "warm_start_from_single_stream",
           "slice_linear_in", "slice_linear_out", "ArmStream", "DuoFlowHead"]
