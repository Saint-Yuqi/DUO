"""[duo.adapters — verbatim copy, do not edit here]
source : /mnt/cpfs/yangyq/pi05_duo/policy_server_duo.py
copied : 2026-10-08 from quic5000 (as it ran; snapshot date 2026-09-28)
ledger : pi05 duo runs
note   : zzm-protocol policy server (multiprocessing.connection, authkey robotwin-baseline) for DuoPI05Policy
Edit the canonical copy in the training workspace, then re-copy; the ledger row points at this file + the source path.
"""
"""RoboTwin policy server for the Duo pi0.5 variants - zzm's wire protocol (Listener 127.0.0.1:port, authkey
b'robotwin-baseline'; reset / infer requests; 50x14 absolute actions back). Differences to his server: the Duo policy
class + DuoProcessor in the preprocessor (prompt without state, own-state tokens), and the contact gate computed per
query by the causal kinematic rule (contact_rules.gate_online) from the measured state and the previous query's state."""
import argparse
import os
import sys
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT)]
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402
from strict_policy import load_policy  # noqa: E402
from comparison import ensure_duo_step  # noqa: E402
from contact_rules import AlohaFK, gate_online  # noqa: E402

TOKENIZER = "/mnt/workspace/zzm/pi05_robotwin_mask_comparison/ablations/state_mask_20260917/tokenizer"
URDF = os.environ.get("DUO_URDF", "/mnt/cpfs/yangyq/rt/assets/aloha_agilex.urdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    variant = os.environ.get("COMPARISON_VARIANT")
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
    cfg.device = "cuda"; cfg.compile_model = False; cfg.gradient_checkpointing = False
    policy = load_policy(args.checkpoint, cfg).eval()
    pre_over = {"device_processor": {"device": "cuda"}, "tokenizer_processor": {"tokenizer_name": TOKENIZER}}
    post_over = {"device_processor": {"device": "cpu"}}
    if os.environ.get("DUO_STATS_JSON"):   # smoke with the base weights: the base preprocessor carries no dataset statistics
        import json
        stats = json.load(open(os.environ["DUO_STATS_JSON"]))
        stats = {k: {n: np.array(v, dtype=np.float32) for n, v in d.items()} for k, d in stats.items() if k in ("observation.state", "action")}
        pre_over["normalizer_processor"] = {"stats": stats, "features": {**cfg.input_features, **cfg.output_features}, "norm_map": cfg.normalization_mapping}
        post_over["unnormalizer_processor"] = {"stats": stats, "features": cfg.output_features, "norm_map": cfg.normalization_mapping}
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(args.checkpoint), preprocessor_overrides=pre_over, postprocessor_overrides=post_over)
    step = ensure_duo_step(pre, variant, training=False)
    fk = AlohaFK(URDF)
    print(f"DUO SERVER variant={variant} step={step.variant} urdf={URDF}", flush=True)
    stats = {"n": 0, "gate": 0}
    with Listener(("127.0.0.1", args.port), authkey=b"robotwin-baseline") as listener:
        print(f"READY {args.port}", flush=True)
        while True:
            with listener.accept() as conn:
                prev_state = None
                while True:
                    try:
                        request = conn.recv()
                        if request["op"] == "reset":
                            policy.reset(); torch.manual_seed(request["seed"]); prev_state = None
                            conn.send({"ok": True}); continue
                        state = np.asarray(request["state"], dtype=np.float32)
                        assert state.shape == (14,) and np.isfinite(state).all()
                        gate, mode = gate_online(fk, state, prev_state)
                        prev_state = state.copy()
                        if os.environ.get("DUO_FORCE_GATE") in ("0", "1"):   # evaluation-only ablation: rule gate forced off / on
                            gate, mode = float(os.environ["DUO_FORCE_GATE"]), "forced"
                        step.override_gate = torch.tensor([gate], dtype=torch.float32)
                        batch = {"observation.state": torch.from_numpy(state).unsqueeze(0), "task": [request["prompt"]]}
                        for cam in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
                            img = np.asarray(request[cam], dtype=np.uint8)
                            assert img.shape in [(240, 320, 3), (480, 640, 3)], img.shape
                            batch["observation.images." + cam] = torch.from_numpy(img.copy()).permute(2, 0, 1).float().div_(255).unsqueeze(0)
                        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                            actions = post(policy.predict_action_chunk(pre(batch)))[0].float().cpu().numpy()
                        assert actions.shape == (50, 14) and np.isfinite(actions).all()
                        n_exec = int(os.environ.get("DUO_EXEC_STEPS", "50"))      # closed-loop horizon: the client executes what it receives
                        conn.send({"actions": actions[:n_exec]})
                        stats["n"] += 1; stats["gate"] += int(gate > 0)
                        lp = getattr(policy.model, "last_pred", None)
                        if lp is not None:
                            stats["head"] = stats.get("head", 0) + int(float(lp["head_gate"][0]) > 0); stats["final"] = stats.get("final", 0) + int(float(lp["gate"][0]) > 0)
                        if stats["n"] % 100 == 1:
                            extra = "" if lp is None else f" head_gate={float(lp['head_gate'][0]):.0f} p={float(lp['p_couple'][0]):.2f} head_rate={stats['head'] / stats['n']:.3f} final_rate={stats['final'] / stats['n']:.3f} modes={lp['modes'][0][:12].tolist()}"
                            print(f"infer #{stats['n']}: rule gate={gate:.0f} ({mode}) rule_rate={stats['gate'] / stats['n']:.3f}{extra} prompt={request['prompt']!r}", flush=True)
                    except (EOFError, ConnectionResetError, BrokenPipeError):
                        break


if __name__ == "__main__":
    main()
