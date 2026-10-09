# DUO — arm-factorised, gated two-stream action head for bimanual policies

**Duo** keeps one action-token stream per arm (6 joints + gripper, or a 10-D end-effector pose), lets each stream read its own
observation, and couples the two streams only through zero-initialised, tanh-gated cross-arm attention after every second block.
At step 0 a Duo head is exactly two decoupled per-arm experts; training opens the channel where the data needs coordination, and a
per-sample gate (rule, learned, or always-open) can switch it. The head is backbone-agnostic: it has been plugged into a DINOv2 DiT,
FLOWER, π0.5, OpenWAM and Fast-WAM (see `duo/adapters/`).

Mirrors: `origin` https://github.com/Saint-Yuqi/DUO · `codeup` https://codeup.aliyun.com/6a3ce6c6a6fcee143fa25a90/DUO
(push to both with `scripts/push_remotes.sh`). **Experiment ledger:** [`docs/index.html`](docs/index.html) (built from `ledger/`), published at https://claude.ai/artifact/LYUcUNwvGWM3XQeWZgQHSR.

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

## Architecture

### What every Duo head shares
* **One action-token stream per arm.** The bimanual chunk `A ∈ R^{H×14}` is split by `Partition` into `a^L, a^R ∈ R^{H×7}`
  (6 joints + gripper; OpenWAM uses 10-D EEF slots 0–9 / 34–43). Each stream has its own input projection, output projection
  and, in most variants, its own copy of the transformer blocks; both streams share the time axis (learned chunk positions
  or RoPE over 0..H−1, so step *t* of the left arm lines up with step *t* of the right arm).
* **Coupling only through cross-arm attention.** After every second block, each stream attends the other stream's tokens:
  `h^L ← h^L + g · tanh(γ^L_ℓ) · CA(LN(h^L), LN(h^R))` and symmetrically for the right stream, computed simultaneously from the
  pre-update values, separate weights per direction. The gain `γ` starts at 0, so at initialisation a Duo head is exactly two
  decoupled per-arm experts; `g ∈ [0,1]` is a per-sample gate (always 1 in the deployed recipes; contact rule / learned head in the
  gated variants). In MoT hosts (OpenWAM, Fast-WAM) the joint attention between the two streams is masked off so this is the only inter-arm path.
* **The other arm's raw state never enters a stream.** In the DiT variants each stream sees the shared head camera + its own wrist
  camera + its own 7-D joints (+ presence flags); in the π0.5 variant the VLM prefix is shared and only the suffix is split.
* **Warm start by slicing.** When the host has a pretrained single-stream expert, its input projection columns / output projection
  rows are sliced per arm and its blocks are copied (`duo.init.warm_start_from_single_stream`); cross-arm modules are new.
* **Flow matching.** Rectified flow on the normalised chunk, one noise level per sample shared by both arms, Euler sampling
  (10 steps in the DiT variants, the host's schedule otherwise).

### Variant matrix
| variant (ledger id) | host / observation per stream | stream blocks | cross-arm attention | gate | init | auxiliary objective | action |
|---|---|---|---|---|---|---|---|
| DINO-DiT Duo (`dit_duo_v1/v2`; parcel sorting, R3) | DINOv2-B tokens: head + own wrist, own 7-D state (+2 flags) | 2 × (8 × 512 adaLN-zero DiT) | own `Attention`, qk-norm, blocks 2,4,6,8 | rule (v1) / rule∨head (v2) / **1.0 deployed** | scratch, or from two per-arm checkpoints | **FLARE future tokens** (below); v2 adds a future-contact head | chunk_delta joints, H=50 |
| FLOWER FlowBlock Duo (`flower_duo_flowblock`) | Florence-2 encoder run once per wrist (head + own wrist + prompt + text); own 7-D state | 2 × (12 × 1024 FlowBlock, pretrained) | `nn.MultiheadAttention`, no qk-norm, 2,4,6,8 | none (always 1) | pretrained FLOWER head, right = deepcopy | none | abs joints, H=50 |
| FLOWER custom Duo (`flower_duo_custom`, `+RoPE`) | same Florence-2 tokens | 2 × (8 × 512 DiTBlock, scratch) | own `Attention`, qk-norm (+RoPE) | from batch, 1 when absent | scratch | none | abs joints |
| π0.5-Duo (`pi05_duo_v1/_open/_v2/_v3`) | shared PaliGemma prefix; suffix `[s_L, 50 L tokens, s_R, 50 R tokens]` | **one** shared pretrained expert, stream embeddings | block mask inside the suffix self-attention (no extra module) | rule / anticipatory+head / **1 (official run)** | sliced 32-d projections | v2: future-contact head (modes + coupled-in-25) | abs joints |
| OpenWAM native dual (`openwam_native_dual`) | Wan2.2 video DiT tokens (MoT), 384×320 canvas | 2 × full pretrained ActionDiT (30 × 1024) | va_b `CrossArm` + RoPE, every even layer | none | sliced encoder/decoder, blocks copied | host video DiT trained (λ_video = 1) | 10-D EEF per arm, H=50 |
| Fast-WAM shared (`fastwam_duo_shared`) | Fast-WAM MoT (video expert frozen in stage 1) | one shared ActionDiT over 2T tokens + arm embedding | none: joint attention left open (`duo_cross=true`) | none | sliced 14-d layers | none | abs joints |
| duo_jepa (`duo_jepa_v1`, YYQ_Quic_EX) | frozen DINOv2-B + adapter, R3 layout | 2 × (8 × 512) | `GatedCross`, 2,4,6,8 (+ a second set for JEPA tokens) | learned from shared latent | scratch | FLARE future tokens + per-arm / shared JEPA | chunk joints |

So the FLARE-style future tokens are **not** in every Duo: they exist only in the DINO-DiT family (R-series, the deployed
parcel-sorting model, duo_jepa). FLOWER-Duo, π0.5-Duo and Fast-WAM-Duo have no future objective; OpenWAM native dual gets its
future signal from the host's own video objective instead.

### FLARE-style future tokens (DINO-DiT family)
An implicit world-model loss: predict the frozen DINO features of the frame at the end of the chunk, no pixels generated.
1. **Future frame.** For each sample the three cameras at `t + 50` (1.67 s at 30 Hz, clamped to the episode end), **un-augmented**.
2. **Targets.** A separate frozen DINOv2-B in the trainer (not in the checkpoint) encodes the three frames; patch tokens (16×16 at 224²)
   are average-pooled to a 4×4 grid and LayerNorm-ed per token → 16 × 768-d targets per camera, 48 in total.
3. **Query tokens.** Each stream appends 32 learned tokens (2 cameras × 16, init std 0.02) after its 50 action tokens; they run
   through the same DiT blocks (self-attention with the noisy action tokens, cross-attention to the stream's observation tokens,
   adaLN from time + pooled observation) and are read out after block 4 of 8.
4. **Loss.** `LayerNorm + Linear(512→768)` per stream; left predicts head + left-wrist targets, right predicts head + right-wrist;
   `L = L_flow + 0.1 · ½(MSE_L + MSE_R)`.
5. **Inference.** The 32 tokens still run (cheap), the head is not evaluated, no image is decoded, the target encoder is not needed.
Because the query tokens see the noisy plan `x_t`, they learn "what the scene looks like after this plan", tying actions to
consequences. The 4×4 grid keeps only coarse layout (where the parcel is, flipped or not, where the hands are). Each stream predicts
only head + own wrist, consistent with the arm split (the 2026-10-08 duo_jepa probe found the future-prediction variance
decomposes per arm with ≈0 cross-arm synergy). No real-robot ablation isolates this term; it has been on since 2026-09-16 (A-group).

### Deployed parcel-sorting recipe (DINO-DiT Duo, 2026-10-09, `DIT_Yuqi/configs/duo_box.yaml`)
Three 224² cameras (head, left wrist, right wrist); 14-D joints + 2 presence flags, proprio dropout 0.3; DINOv2-B fine-tuned at
0.1× lr, 2×2 pooled tokens; two 8×512 streams, cross-arm at 2,4,6,8, gate fixed to 1; chunk_delta actions, H=50; FLARE tokens
4×4 @ t+50, weight 0.1; lr 4e-4, warmup 2000, global batch 512, 17.5k steps (~3 passes over 2.97 M box frames); deploy the last-step
EMA. Served by `dit_mit` on bi-piperx-1 at 30 Hz with RTC prefix guidance (chunk 35 / exec 15), cosine cross-fade and zero-phase
smoothing; ~100 ms per two-arm inference on a 4090. Training repo: `~/QuicData/DIT_Yuqi` (byte-equivalent to `va_bimanual_b`).

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
