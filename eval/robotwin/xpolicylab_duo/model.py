"""XPolicyLab adapter: RoboTwin's official evaluator -> our Duo policy server (any backbone).

UNTESTED SKELETON (2026-10-08). Our finished numbers were produced with the legacy adapter (eval/robotwin/legacy_adapter,
zzm's patched RoboTwin runtime on the cluster). This file is the portable route for AutoDL / a fresh RoboTwin 2.0 checkout:
copy eval/robotwin/xpolicylab_duo to <RoboTwin>/XPolicyLab/policy/Duo, start eval/robotwin/policy_server.py on a GPU, then
    bash scripts/eval_policy.sh --task_name <task> --task_config demo_clean --policy_name Duo --port <p> --instruction_type unseen --seed 0 --test_num 100
Observation keys follow XPolicyLab.model_template / policy/Pi_05/model.py: obs["vision"][cam]["color"] (decoded uint8),
joint state via XPolicyLab.utils.process_data.pack_robot_state. Check both against the installed XPolicyLab version first.
"""
import os
from multiprocessing.connection import Client

import numpy as np

from XPolicyLab.model_template import ModelTemplate

CAMS = ("head_camera", "left_camera", "right_camera")


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.cfg = model_cfg
        self.port = int(model_cfg.get("duo_server_port", os.environ.get("DUO_SERVER_PORT", 19700)))
        self.host = model_cfg.get("duo_server_host", "127.0.0.1")
        self.conn = Client((self.host, self.port), authkey=b"robotwin-baseline")
        self.obs = None
        self.episode = 0
        self.instruction = model_cfg.get("instruction", "")

    def reset(self):
        self.conn.send({"op": "reset", "seed": 42 + self.episode}); assert self.conn.recv()["ok"]
        self.episode += 1

    def update_obs(self, obs):
        self.obs = obs
        if obs.get("instruction"):
            self.instruction = obs["instruction"]

    def update_obs_batch(self, obs_list):
        self.update_obs(obs_list[0])

    def _state14(self, obs):
        try:
            from XPolicyLab.utils.process_data import pack_robot_state
            return np.asarray(pack_robot_state(obs, self.cfg.get("action_type", "joint")), dtype=np.float32)
        except Exception:
            j = obs["joint_action"]
            return np.concatenate([np.asarray(j["left_arm"]), [j["left_gripper"]], np.asarray(j["right_arm"]), [j["right_gripper"]]]).astype(np.float32)

    def get_action(self):
        assert self.obs is not None, "update_obs first"
        v = self.obs["vision"]
        req = {"op": "infer", "state": self._state14(self.obs), "prompt": self.instruction,
               "cam_high": np.asarray(v[CAMS[0]]["color"]), "cam_left_wrist": np.asarray(v[CAMS[1]]["color"]), "cam_right_wrist": np.asarray(v[CAMS[2]]["color"])}
        self.conn.send(req)
        actions = self.conn.recv()["actions"]          # (50, 14) joints: left 6 + gripper, right 6 + gripper
        return [self._to_action_dict(a) for a in actions]

    def get_action_batch(self, env_idx_list=None):
        return [self.get_action()]

    @staticmethod
    def _to_action_dict(a):
        return {"left_arm_joint_state": a[:6], "left_ee_joint_state": a[6:7], "right_arm_joint_state": a[7:13], "right_ee_joint_state": a[13:14]}
