# eval/robotwin — RoboTwin 2.0 evaluation

## The protocol we report (official Co-train leaderboard protocol)
* train: 50 tasks × 50 `demo_clean` episodes (2,500); eval: 50 tasks × {demo_clean, demo_randomized} × 100 episodes, seed 0,
  unseen instructions, expert-filtered episodes (`tasks50.txt`). Overall = mean(clean, random).
* Anything trained on more than the 2,500 clean demos (e.g. the official Fast-WAM weights, 27,500 episodes) is **not** this
  protocol — see `ledger/protocols.yaml`.

## How every finished number was produced (cluster, 2026-10)
1. A **policy server** per GPU speaks the wire protocol in `policy_server.py` (`Listener(('127.0.0.1', port), authkey=b'robotwin-baseline')`,
   `{"op":"infer", state(14,), prompt, cam_high/cam_left_wrist/cam_right_wrist uint8}` → `{"actions": (50,14)}`; the client executes the whole chunk).
   Backbone-specific servers: `legacy_adapter/policy_server_flower.py`, `../../duo/adapters/pi05_lerobot/policy_server_duo.py`.
2. `legacy_adapter/run_eval_fast.py` starts 6 simulator workers per GPU; each runs `legacy_adapter/eval_task.py --task T --setting S --episodes 100`
   inside zzm's patched RoboTwin runtime (`/mnt/workspace/zzm/idea/pi05_robotwin_behavior_probe/RoboTwin`), which calls `baseline_policy.py`.
   Results land in `eval_results/<task>/<setting>/result.json` (resumable; `episode_budget.py` keeps the first 100 valid episodes).
3. `summarize.py <run_root> --run-id <id>` → the `results:` block for `ledger/runs.yaml` + `ledger/per_task/<id>.json`.
   OpenWAM runs use the OpenWAM evaluator instead (`legacy_adapter/eval_summary_openwam.py` shows the result layout; `summarize.py` reads it too).

## Portable route (AutoDL / fresh checkout) — `xpolicylab_duo/`
RoboTwin's official `XPolicyLab` expects a policy package with `model.py` (ModelTemplate) + `deploy.py`. `xpolicylab_duo/model.py`
bridges to our policy server so every backbone is evaluated through one path. **Untested**: verify the observation keys against the
installed XPolicyLab before trusting a number from it, and run one task with `--test_num 3` first.

```bash
# on the eval machine
git clone https://github.com/RoboTwin-Platform/RoboTwin && cd RoboTwin && bash scripts/_install.sh && bash scripts/_download_assets.sh   # or pull assets from OSS, see scripts/pull_data.sh
cp -r <DUO>/eval/robotwin/xpolicylab_duo XPolicyLab/policy/Duo
# GPU 0: policy server;  GPU 0 too: simulator client
python <DUO>/eval/robotwin/policy_server.py --policy my_backbone.serve:load --checkpoint ckpt.pt --port 19700 &
bash scripts/eval_policy.sh --task_name adjust_bottle --task_config demo_clean --policy_name Duo --port 6000 --instruction_type unseen --seed 0 --test_num 100
```
