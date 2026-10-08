"""Generic Duo policy server for the RoboTwin evaluation client (the wire protocol all our runs used).

Protocol (zzm's baseline_policy.py / eval_task.py, and eval/robotwin/xpolicylab_duo/model.py):
    Listener(("127.0.0.1", port), authkey=b"robotwin-baseline"); prints "READY <port>" when the model is loaded.
    request {"op": "reset", "seed": int}                      -> {"ok": True}
    request {"op": "infer", "state": (14,) float32 joints+grippers (left 7, right 7), "prompt": str,
             "cam_high"/"cam_left_wrist"/"cam_right_wrist": uint8 HxWx3}   -> {"actions": (50, 14) float32}
The client executes the whole 50-step chunk open-loop, then asks again.

Plug a policy in with --policy <module:callable>; the callable gets the checkpoint path and returns an object with
    reset(seed) and act(state14: np.ndarray, prompt: str, cam_high, cam_left_wrist, cam_right_wrist) -> np.ndarray (50, 14)
Examples live next to the heads: duo/adapters/pi05_lerobot/policy_server_duo.py and eval/robotwin/legacy_adapter/policy_server_flower.py.
"""
import argparse
import importlib
from multiprocessing.connection import Listener

import numpy as np


def load_policy(spec, checkpoint):
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn or "load")(checkpoint)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="module:callable returning an object with reset(seed) / act(...)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    policy = load_policy(args.policy, args.checkpoint)
    with Listener((args.host, args.port), authkey=b"robotwin-baseline") as listener:
        print(f"READY {args.port}", flush=True)
        while True:
            with listener.accept() as conn:
                while True:
                    try:
                        req = conn.recv()
                    except (EOFError, ConnectionResetError, BrokenPipeError):
                        break
                    if req.get("op") == "reset":
                        policy.reset(int(req.get("seed", 0)))
                        conn.send({"ok": True})
                        continue
                    state = np.asarray(req["state"], dtype=np.float32)
                    assert state.shape == (14,) and np.isfinite(state).all(), state
                    actions = np.asarray(policy.act(state, req["prompt"], req["cam_high"], req["cam_left_wrist"], req["cam_right_wrist"]), dtype=np.float32)
                    assert actions.shape == (50, 14) and np.isfinite(actions).all(), actions.shape
                    conn.send({"actions": actions})


if __name__ == "__main__":
    main()
