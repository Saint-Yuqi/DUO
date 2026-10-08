# duo/adapters — the exact code behind every ledger row

| file | host backbone | head version (ledger/variants.yaml) | runs | environment it needs |
|---|---|---|---|---|
| `dino_dit_va_b.py` | DINOv2-B + 8×512 DiT-flow (`va_bimanual_b`) | `dit_duo_v1`, `dit_duo_v2` | R3 / R3b / R3c (zzm500), real-robot P16_DUO_* | cluster `flower` env (`/mnt/cpfs/yangyq/envs/flower`), `va_bimanual_b` on PYTHONPATH |
| `flower_flowblock.py` | FLOWER VLA, pretrained FlowBlock head | `flower_duo_flowblock` | flower_duodit_clean2500 **68.30 / 26.62** | `flower` env + `flower_bimanual_v3` |
| `flower_custom_dit.py` | FLOWER encoder + scratch DiTBlocks | `flower_duo_custom` | flower_customduodit_clean2500 65.38 / 18.84 | `flower` env + `flower_bimanual_v3` + `va_bimanual_b` |
| `flower_custom_dit_rope.py` (+ `legacy/action_rope.py`) | same, RoPE | `flower_duo_custom_rope` | flower_customduodit_rope_clean2500 66.50 / 19.56 | same |
| `pi05_lerobot/comparison.py` (+ `contact_rules.py`, `policy_server_duo.py`) | π0.5 (lerobot 0.6.1 `PI05Policy`) | `pi05_duo_v1`, `_v1_open`, `_v2`, `_v3` | pi05_duo_open_clean2500 **80.36 / 46.36**; all zzm500 π0.5-Duo runs | public `pi05_runtime` conda env (`/mnt/cpfs/miniconda3/envs/pi05_runtime`, py3.12) |
| `openwam_native_dual.py` (+ `legacy/openwam_install_native_dual.py`) | OpenWAM-Alpha (Wan2.2 5B video DiT + ActionDiT) | `openwam_native_dual` | openwam_native_dual_clean2500 (evaluating) | wsh `fastwam-py310` env + OpenWAM source clone; the installer patches the clone |
| `fastwam_shared_stream.py` (+ `legacy/fastwam_apply_duo_patch.py`) | Fast-WAM | `fastwam_duo_shared` | duo_fastwam_stage1 (10-task check) | `duo_fastwam` copy of wsh's Fast-WAM tree, DeepSpeed |

Rules: these files are **copies** (header says source path + snapshot date). Change the source in its training workspace, re-copy,
and bump the variant/run in the ledger. The backbone-agnostic pieces they all share (CrossArm with tanh zero-init gain, per-arm
partition, warm-start slicing, MoT mask, RoPE) are reimplemented and tested in `duo/`; a new backbone should use `duo/` and add a
thin adapter here (observation tokens in → `DuoFlowHead` or a `CrossArmSet` around the host's own blocks).

## Adding a new backbone (the pattern the five above follow)
1. Decide the stream type: (a) the host has a per-token DiT/flow block list → run two copies (or one shared copy + stream embedding)
   and insert `CrossArmSet.apply` after the listed blocks (FLOWER, DINO-DiT, OpenWAM); (b) the host has one joint attention over all
   tokens (MoT) → double the action tokens, mask the inter-stream blocks with `block_cross_stream_attention`, and either add
   `CrossArm` modules (OpenWAM native dual) or leave the attention open (Fast-WAM `duo_cross=true`).
2. Partition: `Partition.arm_semantic(7)` for 14-D joints, `Partition.eef_openwam80()` for OpenWAM's 80-D EEF space,
   `Partition.mixed(7)` / `Partition.joint(14)` for the paper's ablations.
3. Warm start from the host's pretrained single-stream expert with `warm_start_from_single_stream` (encoder columns / decoder rows).
4. Serve it through `eval/robotwin/policy_server.py` (`--policy your_module:load`) and evaluate; record the run in `ledger/runs.yaml`.
