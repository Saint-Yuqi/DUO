"""Serve FLOWER + shared-weight Duo actions to RoboTwin's official evaluator (copy of flower_duodit_robotwin/policy_server.py,
model class swapped, --device added)."""
import argparse
from multiprocessing.connection import Listener

import numpy as np
from PIL import Image
import torch

from flower_sharedduo import FlowerSharedDuo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = FlowerSharedDuo(ckpt["config"]["vlm"], gate=ckpt["config"].get("gate", "open")).to(args.device).eval()
    model.load_state_dict(ckpt["model"])
    state_stats, action_stats = ckpt["norm"]["state"], ckpt["norm"]["action"]
    state_lo = np.asarray(state_stats["lo"], dtype=np.float32)
    state_rng = np.asarray(state_stats["rng"], dtype=np.float32)
    state_dead = np.asarray(state_stats["dead"])
    action_lo = np.asarray(action_stats["lo"], dtype=np.float32)
    action_rng = np.asarray(action_stats["rng"], dtype=np.float32)
    action_dead = np.asarray(action_stats["dead"])

    def image(raw):
        arr = np.asarray(raw, dtype=np.uint8)
        assert arr.ndim == 3 and arr.shape[2] == 3
        resized = np.asarray(Image.fromarray(arr).resize((224, 224), Image.BILINEAR, reducing_gap=2.0))
        return torch.from_numpy(resized.copy()).permute(2, 0, 1).unsqueeze(0)

    with Listener(("127.0.0.1", args.port), authkey=b"robotwin-baseline") as listener:
        print(f"READY {args.port}", flush=True)
        while True:
            with listener.accept() as conn:
                while True:
                    try:
                        req = conn.recv()
                        if req["op"] == "reset":
                            torch.manual_seed(req["seed"])
                            conn.send({"ok": True})
                            continue
                        state = np.asarray(req["state"], dtype=np.float32)
                        assert state.shape == (14,) and np.isfinite(state).all()
                        normalized = np.where(state_dead, 0, 2 * (state - state_lo) / state_rng - 1)
                        batch = {"rgb_static": image(req["cam_high"]),
                                 "rgb_gripper": image(req["cam_left_wrist"]),
                                 "rgb_third": image(req["cam_right_wrist"]),
                                 "proprio": torch.from_numpy(np.clip(normalized, -1, 1).astype(np.float32))[None],
                                 "lang_text": [req["prompt"]]}
                        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                            pred = model.predict(batch)[0].float().cpu().numpy()
                        actions = np.where(action_dead, action_lo, (pred + 1) * action_rng / 2 + action_lo)
                        assert actions.shape == (50, 14) and np.isfinite(actions).all()
                        conn.send({"actions": actions.astype(np.float32)})
                    except (EOFError, ConnectionResetError, BrokenPipeError):
                        break


if __name__ == "__main__":
    main()
