# DUO — arm-factorised, gated two-stream action head for bimanual policies

**Duo** keeps one action-token stream per arm (6 joints + gripper, or a 10-D end-effector pose), lets each stream read its own
observation, and couples the two streams only through zero-initialised, tanh-gated cross-arm attention after every second block.
At step 0 a Duo head is exactly two decoupled per-arm experts; training opens the channel where the data needs coordination, and a
per-sample gate (rule, learned, or always-open) can switch it. The head is backbone-agnostic: it has been plugged into a DINOv2 DiT,
FLOWER, π0.5, OpenWAM and Fast-WAM (see `duo/adapters/`).

Mirrors: `origin` https://github.com/Saint-Yuqi/DUO · `codeup` https://codeup.aliyun.com/6a3ce6c6a6fcee143fa25a90/DUO
(push to both with `scripts/push_remotes.sh`). **Experiment ledger:** [`docs/index.html`](docs/index.html) (built from `ledger/`).

## Layout
| path | what |
|---|---|
| `duo/` | the library: `Partition` (arm-semantic / mixed / OpenWAM-EEF / joint), `CrossArm` + `CrossArmSet` (tanh zero-init gain, per-sample gate), `block_cross_stream_attention` for MoT hosts, RoPE, `warm_start_from_single_stream`, and `DuoFlowHead` — the complete from-scratch reference head (flow matching, Euler sampler). Pure torch; `tests/test_duo_cpu.py` runs on CPU. |
| `duo/adapters/` | **verbatim** copies of the head as it ran inside each backbone (provenance header: source path, snapshot date, ledger runs). |
| `eval/robotwin/` | the RoboTwin 2.0 evaluation route: wire protocol + generic policy server, the legacy runner that produced every number, the result summariser, an (untested) XPolicyLab bridge for fresh machines. |
| `ledger/` | **single source of truth** for every run, head version, external number and paper table (YAML) + `build.py` → `docs/` + `paper/tables/`. |
| `paper/tables/` | generated LaTeX tables in the shape of the Overleaf files (Duo-WAM, CVPR 2026 template; mirror github.com/pikonguwu/Duo-WAM). |
| `scripts/` | `setup_env.sh`, `setup_autodl.sh`, `pull_data.sh` (OSS), `push_remotes.sh`, `sync_paper.sh`. |

## Quick start
```bash
git clone git@github.com:Saint-Yuqi/DUO.git && cd DUO
bash scripts/setup_env.sh            # venv + pip install -e . + CPU tests
python ledger/build.py               # rebuild docs/index.html, docs/ledger.json, paper/tables/*.tex from ledger/*.yaml
```
```python
from duo import Partition, DuoFlowHead
head = DuoFlowHead(Partition.arm_semantic(7), dim=512, layers=8, heads=8, chunk=50, cross_layers=(2, 4, 6, 8))
ctx = [{"obs": tokens_left, "pooled": proprio_left}, {"obs": tokens_right, "pooled": proprio_right}]   # (B, N, 512), (B, 512)
loss = head.loss(actions, ctx, gate=None, mask=action_mask)      # actions (B, 50, 14) normalised
chunk = head.sample(ctx)                                          # (B, 50, 14)
```
`Partition.mixed(7)` / `Partition.joint(14)` give the paper's arm-grouping ablation and the single-stream reference with the same code.

## Results so far (official RoboTwin 2.0 protocol: clean2500 train, 50 tasks × clean/random × 100)
| backbone + Duo | clean | random | overall | ledger id |
|---|---|---|---|---|
| π0.5 + Duo (gate open) | 80.36 | 46.36 | 63.36 | `pi05_duo_open_clean2500` |
| FLOWER + Duo (pretrained FlowBlock streams) | 68.30 | 26.62 | 47.46 | `flower_duodit_clean2500` |
| FLOWER + custom Duo + RoPE (scratch) | 66.50 | 19.56 | 43.03 | `flower_customduodit_rope_clean2500` |
| FLOWER + custom Duo (scratch) | 65.38 | 18.84 | 42.11 | `flower_customduodit_clean2500` |
| OpenWAM + native dual Duo (two pretrained experts, EEF) | — | — | — | `openwam_native_dual_clean2500` (evaluating) |

Leaderboard references (snapshot 2026-09-24): π0.5 70.70 / 46.00 / 58.35, OpenWAM 89.40 / 48.70 / 69.05, Fast-WAM 77.80 / 1.90 / 39.85.
Everything else (zzm 500-episode runs, the Fast-WAM 10-task check) is internal and lives in the ledger under its own protocol.

## Adding a run / a backbone
See `duo/adapters/README.md` (backbone pattern) and `ledger/README.md` (recording). Team workflow in Chinese: `README_zh.md`.
