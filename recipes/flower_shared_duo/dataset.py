"""Use the existing FLOWER frame cache with RoboTwin's three cameras."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/mnt/workspace/yangyq/code/flower_bimanual_v3")
from data import FrameCacheDataset, Normalizer  # noqa: E402


class RoboTwinDataset(FrameCacheDataset):
    def __init__(self, cache, chunk=50, episodes=None):
        super().__init__(cache, "cam_high", "cam_left_wrist", chunk=chunk,
                         episodes=episodes, splice_prob=0.0)
        self.idx["cam_right_wrist"] = np.load(Path(cache) / "cam_right_wrist.idx.npy")

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["rgb_third"] = self._image("cam_right_wrist", item["index"])
        return item
