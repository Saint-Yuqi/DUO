"""Warm start of a Duo head from a pretrained single-stream expert.

The three places this was done and what they share (all in duo/adapters, verbatim):
  * FLOWER FlowBlock Duo (flower_flowblock.py)     : right stream = deepcopy(left stream); new per-arm in/out projections
  * Fast-WAM DuoActionDiT (fastwam_shared_stream.py): shared blocks; per-arm encoder = 14-d encoder columns of that arm,
                                                      per-arm head = 14-d head rows of that arm
  * OpenWAM native dual (openwam_native_dual.py)   : two full copies of the pretrained ActionDiT; action_encoder
                                                      columns / action_decoder rows sliced per arm (0..9 and 34..43)
The linear input layer is linear in its input, so  W[a_L; a_R] + b = W[:, L] a_L + W[:, R] a_R + b : slicing gives each
arm the embedding the pretrained expert already learned for its own dimensions. Cross-arm modules are always new and
start at zero gain, so the warm-started Duo head equals N decoupled copies of the pretrained expert at step 0.
"""
from typing import Dict, Iterable, Optional, Sequence

import torch

from .partition import Partition


def slice_linear_in(weight: torch.Tensor, bias: Optional[torch.Tensor], idx: Sequence[int]):
    """Input projection (out, in_total) -> (out, len(idx)); bias unchanged."""
    return weight[:, list(idx)].clone(), (None if bias is None else bias.clone())


def slice_linear_out(weight: torch.Tensor, bias: Optional[torch.Tensor], idx: Sequence[int]):
    """Output projection (out_total, in) -> (len(idx), in); bias (out_total,) -> (len(idx),)."""
    return weight[list(idx)].clone(), (None if bias is None else bias[list(idx)].clone())


def warm_start_from_single_stream(target: Dict[str, torch.Tensor], source: Dict[str, torch.Tensor], partition: Partition, *,
                                  stream_prefixes: Sequence[str], source_prefix: str = "", in_proj: str = "act_in",
                                  out_proj: str = "act_out", new_prefixes: Iterable[str] = ("cross",), strict=True):
    """Build a state dict for a multi-stream target from a single-stream source.

    target          : target.state_dict() (shapes are taken from it)
    source          : the pretrained single-stream state dict
    stream_prefixes : e.g. ("streams.0.", "streams.1.") — every target key under stream i maps to the same key under
                      `source_prefix` in the source; `in_proj`/`out_proj` (relative names) are sliced by partition.streams[i]
    new_prefixes    : target key prefixes that have no source (CrossArm modules, arm embeddings, …) and are left as they are
    Returns the merged dict (target values for new keys). Raises on shape mismatch when strict."""
    out = dict(target)
    for i, sp in enumerate(stream_prefixes):
        idx = partition.streams[i]
        for k, v in target.items():
            if not k.startswith(sp):
                continue
            rel = k[len(sp):]
            if any(rel.startswith(p) or k.startswith(p) for p in new_prefixes):
                continue
            sk = source_prefix + rel
            if sk not in source:
                if strict:
                    raise KeyError(f"source has no tensor for {k} (looked for {sk})")
                continue
            w = source[sk]
            if rel.startswith(in_proj + ".weight"):
                w, _ = slice_linear_in(w, None, idx)
            elif rel.startswith(out_proj + ".weight"):
                w, _ = slice_linear_out(w, None, idx)
            elif rel.startswith(out_proj + ".bias"):
                w = w[list(idx)].clone()
            if w.shape != v.shape:
                if strict:
                    raise ValueError(f"{k}: source {tuple(w.shape)} vs target {tuple(v.shape)}")
                continue
            out[k] = w.to(v.dtype)
    return out
