"""Which action dimensions belong to which stream.

Arm-semantic (default): [q_L^1..6, g_L] | [q_R^1..6, g_R]  — the Duo head.
Mixed (ablation control, Duo-WAM paper Sec. 'arm-semantic versus mixed'): [q_L^1..3, q_R^1..3, g_L] | [q_L^4..6, q_R^4..6, g_R].
OpenWAM 80-d unified EEF space: left = slots 0..9, right = slots 34..43 (xyz, rot6d, gripper per arm).
Joint: one stream over all dims (the single-stream reference).
"""
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch


@dataclass(frozen=True)
class Partition:
    name: str
    total_dim: int
    streams: Tuple[Tuple[int, ...], ...]      # per stream, the indices into the canonical action vector

    def __post_init__(self):
        flat = [i for s in self.streams for i in s]
        assert len(set(flat)) == len(flat), "a dimension is assigned to two streams"
        assert all(0 <= i < self.total_dim for i in flat), (self.total_dim, flat)

    @property
    def n_streams(self):
        return len(self.streams)

    @property
    def stream_dims(self):
        return tuple(len(s) for s in self.streams)

    @property
    def covered(self):
        return sorted(i for s in self.streams for i in s)

    def split(self, x):
        """x (..., total_dim) -> list of (..., len(stream_i))."""
        assert x.shape[-1] == self.total_dim, (x.shape, self.total_dim)
        return [x[..., list(s)] for s in self.streams]

    def merge(self, parts, fill=0.0):
        """Inverse of split; dimensions no stream owns are filled with `fill` (e.g. the unused slots of the 80-d space)."""
        assert len(parts) == self.n_streams
        out = parts[0].new_full((*parts[0].shape[:-1], self.total_dim), fill)
        for s, p in zip(self.streams, parts):
            out[..., list(s)] = p.to(out.dtype)
        return out

    # ---- factories
    @staticmethod
    def arm_semantic(arm_dim=7):
        return Partition("arm_semantic", 2 * arm_dim, (tuple(range(arm_dim)), tuple(range(arm_dim, 2 * arm_dim))))

    @staticmethod
    def mixed(arm_dim=7):
        """Paper control: first branch = first half of each arm's joints + g_L, second = remaining joints + g_R."""
        j = arm_dim - 1                     # joints per arm (gripper is the last dim of each arm)
        half = j // 2
        L, R = list(range(arm_dim)), list(range(arm_dim, 2 * arm_dim))
        s1 = tuple(L[:half] + R[:half] + [L[-1]])
        s2 = tuple(L[half:j] + R[half:j] + [R[-1]])
        return Partition("mixed", 2 * arm_dim, (s1, s2))

    @staticmethod
    def eef_openwam80():
        return Partition("eef_openwam80", 80, (tuple(range(0, 10)), tuple(range(34, 44))))

    @staticmethod
    def joint(dim=14):
        return Partition("joint", dim, (tuple(range(dim)),))

    @staticmethod
    def by_name(name, arm_dim=7):
        return {"arm_semantic": lambda: Partition.arm_semantic(arm_dim), "mixed": lambda: Partition.mixed(arm_dim),
                "eef_openwam80": Partition.eef_openwam80, "joint": lambda: Partition.joint(2 * arm_dim)}[name]()
