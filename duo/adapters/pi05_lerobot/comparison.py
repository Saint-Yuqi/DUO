"""[duo.adapters — verbatim copy, do not edit here]
source : /mnt/cpfs/yangyq/pi05_duo/comparison.py
copied : 2026-10-08 from quic5000 (as it ran; snapshot date 2026-09-28)
ledger : pi05_duo_open_clean2500 (80.36 / 46.36 / 63.36, official); duo / duo_g3 / duo_v2* / duo_v3 (zzm500 protocol)
note   : DuoPI05Pytorch: two-stream gated suffix inside lerobot 0.6.1 PI05 — [state_L, 50 left tokens, state_R, 50 right tokens], per-stream in/out projections sliced from the pretrained 32-d ones, SHARED expert weights, learned stream embeddings, block mask opened by the gate; v2 adds anticipatory gate + critical-point weights + future-contact head; v3 contact tokens. The official run used variant duo with DUO_TRAIN_GATE=1 (gate always open).
Edit the canonical copy in the training workspace, then re-copy; the ledger row points at this file + the source path.
"""
"""Duo components inside the pi0.5 action expert, on zzm's fixed RoboTwin protocol (500 episodes, 1,200 updates,
50 tasks x clean/randomized x 10 episodes). Drop-in for his `comparison.py` (same names: VARIANTS, MaskedPI05Policy,
configure_preprocessor) so train.py / strict_policy.py / run_train.sh / parallel_eval.py are reused unchanged.

v1 (duo, duo_g3) - what changes inside the policy (nothing in the VLM prefix except that the state text is gone):
  * two action streams in the suffix: [state_L, 50 left action tokens, state_R, 50 right action tokens]; per-stream
    action in/out projections initialised from the pretrained 32-d projections (columns 0:7 / 7:14), learned stream
    embeddings, the pretrained expert weights shared by both streams;
  * the other arm's raw joints never enter: the prompt has no "State:" any more, each stream gets ONLY its own 7-d state
    (+ a presence flag) as one token; the VLM prefix (images + task) is shared;
  * cross-stream attention is gated by the contact-graph gate c (S / A frames -> 1, else 0): with c = 0 the two streams
    are two independent per-arm experts over the same prefix (= the B9 structure), with c = 1 they attend each other
    (= the original joint expert). v1 training gate = kinematic rule on the raw state/chunk; evaluation gate = the
    causal rule contact_rules.gate_online computed by the policy server from the measured state.
  duo_g3 : + zzm's g3 intervention on the own-state tokens (15% hide left, 15% hide right, 30% hide both).

v2 (duo_v2 and its ablations) - the two papers' lessons (critical point = the contact-graph transition; contact as a
FUTURE prediction target, not an input):
  * labels: simulator ground truth (contact_labels_v2.py: which gripper touches which object, replayed in the seeded
    scene), looked up per training sample by (episode_index, timestamp) of the lerobot batch;
  * anticipatory gate: training gate = 1 if the two arms are coupled anywhere in the next GATE_K steps of the chunk
    (the approach BEFORE contact is where coordination is decided);
  * critical-point weighting: per-step flow-matching weights 1 + CP_LAMBDA * exp(-|t - t_c| / CP_TAU), t_c = nearest
    coupling transition inside the chunk, normalised to mean 1 (no weighting when the chunk has no transition);
  * future-contact head: from the pooled VLM prefix + own states, predict the contact mode of every step of the chunk
    (5 classes, same step weights) and "coupled within GATE_K steps"; at evaluation the gate = head prediction OR the
    causal rule (DUO_EVAL_GATE = or | head | rule).
  duo_v2        : gate + weights + head
  duo_v2_nohead : gate + weights, no head (evaluation gate = causal rule as in v1)
  duo_v2_nocp   : gate + head, uniform step weights
  duo_v2_l1     : duo_v2 mechanisms, but trained on the v1 kinematic-rule labels (label-quality ablation)
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch, create_sinusoidal_pos_embedding, make_att_2d_masks, clone_past_key_values
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from lerobot.processor.normalize_processor import NormalizerProcessorStep

from contact_rules import AlohaFK, gate_train

ROOT = Path(__file__).resolve().parent
VARIANTS = ("duo", "duo_g3", "duo_v2", "duo_v2_nohead", "duo_v2_nocp", "duo_v2_l1", "duo_v2_appr", "duo_v2_ovs", "duo_v2_gate", "duo_v2_sg", "duo_v3")
STATE_CFG = {"duo": (0.0, 0.0), "duo_g3": (0.3, 0.3), "duo_v2": (0.0, 0.0), "duo_v2_nohead": (0.0, 0.0), "duo_v2_nocp": (0.0, 0.0), "duo_v2_l1": (0.0, 0.0),
             "duo_v2_appr": (0.0, 0.0), "duo_v2_ovs": (0.0, 0.0), "duo_v2_gate": (0.0, 0.0), "duo_v2_sg": (0.0, 0.0), "duo_v3": (0.0, 0.0)}
V2 = {"duo_v2": dict(head=True, cp=True), "duo_v2_nohead": dict(head=False, cp=True), "duo_v2_nocp": dict(head=True, cp=False),
      "duo_v2_l1": dict(head=True, cp=True),   # duo_v2_l1: the v2 mechanisms trained on the v1 kinematic labels (DUO_LABELS)
      "duo_v2_appr": dict(head=True, cp=True),  # + DUO_CP_PRE approach weighting (set by the launcher)
      "duo_v2_ovs": dict(head=True, cp=True),   # + DUO_SAMPLE_W oversampling of frames with the anticipatory gate open (train.py)
      "duo_v2_gate": dict(head=False, cp=False),  # anticipatory gate from the v2 labels only (minimal change over duo)
      "duo_v2_sg": dict(head=True, cp=True, sg=True),  # head on stop-gradient prefix features: the aux loss cannot touch the VLM trunk
      "duo_v3": dict(head=True, cp=True, ctok=True)}     # ME-Dex style: C contact tokens inside the action expert suffix predict the future
                                                       # contact modes; both arm streams always read them (a narrow consequence channel)
URDF = os.environ.get("DUO_URDF", "/mnt/cpfs/yangyq/rt/assets/aloha_agilex.urdf")
LABELS = os.environ.get("DUO_LABELS", "/mnt/cpfs/yangyq/rt/assets/contact_v2_lookup_robotwin500.npz")
GATE_K = int(os.environ.get("DUO_GATE_HORIZON", "25"))
CP_LAMBDA = float(os.environ.get("DUO_CP_LAMBDA", "4.0")); CP_TAU = float(os.environ.get("DUO_CP_TAU", "5.0"))
AUX_W = float(os.environ.get("DUO_AUX_W", "0.2"))
EVAL_GATE = os.environ.get("DUO_EVAL_GATE", "or")
FPS = 30
DUO_PREFIXES = ("action_in_L", "action_in_R", "action_out_L", "action_out_R", "state_in_L", "state_in_R", "stream_embed", "contact_head", "contact_queries", "contact_seg_head", "contact_k_head")
ARM = 7
NMODE = 5
C_TOK = int(os.environ.get("DUO_CTOK", "10"))     # v3: number of contact tokens (each predicts H / C_TOK future steps)


class ContactLabels:
    """Per-episode label sequences from the lookup npz (keys: episode_index = ORIGINAL lerobot id, frame_index, mode,
    coupled).  Falls back to mode >= 2 when the file has no `coupled` column (v1 lookup)."""

    def __init__(self, path):
        z = np.load(path)
        ep, fr, mode = z["episode_index"].astype(np.int64), z["frame_index"].astype(np.int64), z["mode"].astype(np.int64)
        cp = z["coupled"].astype(bool) if "coupled" in z.files else (mode >= 2)
        if os.environ.get("DUO_GATE_SYNC") == "1" and "graspL" in z.files:      # synchronised-parallel regime also opens the gate
            cp = cp | (z["graspL"].astype(bool) & z["graspR"].astype(bool))
        self.seq = {}
        for e in np.unique(ep):
            m = ep == e; o = np.argsort(fr[m])
            self.seq[int(e)] = (mode[m][o], cp[m][o])
        self.path = path

    def future(self, e, f, H):
        mode, cp = self.seq[int(e)]
        n = len(mode); f = min(max(int(f), 0), n - 1)
        idx = np.minimum(np.arange(f, f + H), n - 1)
        return mode[idx], cp[idx], bool(cp[f - 1] if f > 0 else cp[f])


CP_PRE = int(os.environ.get("DUO_CP_PRE", "0"))      # >0: approach weighting - full weight on the PRE steps before a transition


def step_weights(fc, prev, lam, tau, pre=None):
    """1 + lam * exp(-d / tau), d = distance to the nearest coupling transition in the chunk; mean 1; ones if none.
    With pre > 0 (DUO_CP_PRE) the PRE steps leading up to a transition get the full 1 + lam (the approach, where the
    probe finds the error), and the exponential decay applies after it."""
    H = len(fc)
    tc = np.where(np.concatenate([[fc[0] != prev], fc[1:] != fc[:-1]]))[0]
    if len(tc) == 0 or lam <= 0:
        return np.ones(H, np.float32)
    pre = CP_PRE if pre is None else pre
    k = np.arange(H)[:, None]; rel = k - tc[None, :]                      # steps relative to each transition
    if pre > 0:
        d = np.where(rel <= 0, np.maximum(0, -rel - pre), rel).min(1)     # 0 inside [tc - pre, tc], grows outside
    else:
        d = np.abs(rel).min(1)
    w = 1.0 + lam * np.exp(-d / tau)
    return (w / w.mean()).astype(np.float32)


# ----------------------------------------------------------------------------------------------- processor step

@ProcessorStepRegistry.register(name="robotwin_pi05_duo")
class DuoProcessor(ProcessorStep):
    """Replaces Pi05PrepareStateTokenizerProcessorStep: prompt WITHOUT the state; own-arm states, presence flags and the
    contact gate go to the complementary data (-> policy batch as duo_state / duo_flags / duo_gate, and for v2
    duo_future_mode / duo_step_w / duo_couple_k / duo_valid)."""

    def __init__(self, variant="duo", seed=42):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant, self.seed = variant, seed
        self.xarm, self.pdrop = STATE_CFG[variant]
        self.v2 = V2.get(variant)
        self._training = False
        self._generators = {}
        self._fk = None
        self._labels = None
        self._stats = None          # {"observation.state": (q01, q99), "action": (q01, q99)} from the normalizer step
        self.override_gate = None   # evaluation: the policy server sets the causal gate here before calling the pipeline

    def get_config(self):
        return {"variant": self.variant, "seed": self.seed}

    def transform_features(self, features):
        return features

    def _gen(self, device):
        key = str(device)
        if key not in self._generators:
            self._generators[key] = torch.Generator(device=device).manual_seed(self.seed + 100003 * int(os.environ.get("RANK", "0")))
        return self._generators[key]

    def _unnorm(self, key, y):
        q01, q99 = self._stats[key]
        q01 = torch.as_tensor(q01, dtype=torch.float32, device=y.device); q99 = torch.as_tensor(q99, dtype=torch.float32, device=y.device)
        return (y.float() + 1.0) * (q99 - q01) / 2.0 + q01

    def _train_gate(self, state, actions):
        """v1: gate from the kinematic rule on the RAW state and the RAW action chunk (contact_rules.gate_train)."""
        if self._fk is None:
            self._fk = AlohaFK(URDF)
        if self._stats is None or actions is None:
            raise ValueError("DuoProcessor needs the normalizer stats (ensure_duo_step) and the action chunk to compute the training gate")
        raw_s = self._unnorm(OBS_STATE, state).cpu().numpy(); raw_a = self._unnorm(ACTION, actions).cpu().numpy()
        gate, _ = gate_train(self._fk, raw_s, raw_a)
        return torch.from_numpy(gate).to(state.device)

    def _v2_targets(self, complementary, B, H, device):
        """v2: anticipatory gate, critical-point step weights and the future contact modes from the simulator labels."""
        if self._labels is None:
            self._labels = ContactLabels(LABELS)
        ep = complementary.get("episode_index"); ts = complementary.get("timestamp")
        if ep is None or ts is None:
            raise ValueError("v2 training needs episode_index and timestamp in the batch (lerobot _COMPLEMENTARY_KEYS)")
        ep = torch.as_tensor(ep).view(-1).tolist(); fr = torch.round(torch.as_tensor(ts).view(-1).float() * FPS).long().tolist()
        gate = np.zeros(B, np.float32); w = np.ones((B, H), np.float32); fm = np.zeros((B, H), np.int64)
        for i in range(B):
            m, c, prev = self._labels.future(ep[i], fr[i], H)
            fm[i] = m; gate[i] = float(c[:GATE_K].any())
            if self.v2["cp"]:
                w[i] = step_weights(c, prev, CP_LAMBDA, CP_TAU)
        return (torch.from_numpy(gate).to(device), torch.from_numpy(w).to(device), torch.from_numpy(fm).to(device))

    def __call__(self, transition):
        result = transition.copy()
        obs = dict(result[TransitionKey.OBSERVATION])
        complementary = dict(result.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        state = obs[OBS_STATE]
        if state.ndim != 2 or state.shape[-1] != 14:
            raise ValueError(f"Expected bimanual state [B,14], got {state.shape}")
        B, device = state.shape[0], state.device
        tasks = complementary.get("task")
        if tasks is None:
            raise ValueError("No task found in complementary data")
        prompts = []
        for task in tasks:
            cleaned = str(task).strip().replace("_", " ").replace("\n", " ")
            prompts.append(f"Task: {cleaned};\nAction: ")
        complementary["task"] = prompts
        flags = torch.ones(B, 2, dtype=state.dtype, device=device)
        st = state.clone()
        if self._training and (self.xarm > 0 or self.pdrop > 0):
            u = torch.rand(B, device=device, generator=self._gen(device))
            hide_l = u < self.xarm / 2
            hide_r = (u >= self.xarm / 2) & (u < self.xarm)
            both = (u >= self.xarm) & (u < self.xarm + self.pdrop)
            flags[hide_l | both, 0] = 0; flags[hide_r | both, 1] = 0
            st[:, :ARM] = st[:, :ARM] * flags[:, :1]; st[:, ARM:] = st[:, ARM:] * flags[:, 1:]
        if os.environ.get("DUO_NO_STATE") == "1":   # attribution: no proprioception anywhere (the Duo prompt already has none)
            st = torch.zeros_like(st); flags = torch.zeros_like(flags)
        complementary["duo_state"] = st
        complementary["duo_flags"] = flags
        actions = result.get(TransitionKey.ACTION)
        H = actions.shape[1] if actions is not None else 50
        step_w = None
        if self._training:
            if self.v2 is not None:
                gate, step_w, fm = self._v2_targets(complementary, B, H, device)
                complementary["duo_gate"] = gate; complementary["duo_step_w"] = step_w; complementary["duo_future_mode"] = fm
                complementary["duo_couple_k"] = gate.clone()
            else:
                complementary["duo_gate"] = self._train_gate(state, actions)
            if os.environ.get("DUO_TRAIN_GATE") in ("0", "1"):   # attribution: training gate forced off / on for every sample
                complementary["duo_gate"] = torch.full((B,), float(os.environ["DUO_TRAIN_GATE"]), device=device)
        else:
            g = self.override_gate if self.override_gate is not None else complementary.get("duo_gate")
            complementary["duo_gate"] = torch.zeros(B, device=device) if g is None else torch.as_tensor(g, dtype=torch.float32, device=device).view(-1).expand(B).clone()
        if actions is not None:
            if actions.ndim != 3 or actions.shape[1:] != (50, 14):
                raise ValueError(f"Expected bimanual action [B,50,14], got {actions.shape}")
            pad = complementary.get("action_is_pad")
            if self._training and pad is None:
                raise ValueError("Training requires action_is_pad")
            valid = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device) if pad is None else ~pad.bool()
            complementary["duo_valid"] = valid
            lw = valid.to(actions.dtype)
            if step_w is not None:
                lw = lw * step_w.to(actions.dtype)
            complementary["loss_w"] = lw.unsqueeze(-1).expand_as(actions)
        result[TransitionKey.OBSERVATION] = obs
        result[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return result


def ensure_duo_step(preprocessor, variant, seed=42, training=False):
    """Put the DuoProcessor where the state tokenizer step was (training and evaluation pipelines)."""
    steps = preprocessor.steps
    norm = [s for s in steps if isinstance(s, NormalizerProcessorStep)]
    if len(norm) != 1 or not norm[0].stats:
        raise ValueError(f"Expected one normalizer step with stats, found {len(norm)}")
    stats = {k: (norm[0].stats[k]["q01"], norm[0].stats[k]["q99"]) for k in (OBS_STATE, ACTION) if k in norm[0].stats}
    step = next((s for s in steps if isinstance(s, DuoProcessor)), None)
    if step is None:
        idx = [i for i, s in enumerate(steps) if isinstance(s, Pi05PrepareStateTokenizerProcessorStep)]
        if len(idx) != 1:
            raise ValueError(f"Expected one pi05 state tokenizer step, found {len(idx)}")
        step = DuoProcessor(variant, seed)
        steps[idx[0]] = step
    step._training = training
    step._stats = stats
    return step


def configure_preprocessor(preprocessor, seed):
    variant = os.environ.get("COMPARISON_VARIANT")
    if variant not in VARIANTS:
        raise ValueError(variant)
    step = ensure_duo_step(preprocessor, variant, seed, training=True)
    if step.v2 is not None:
        print(f"DUO {variant}: v2 {step.v2}; labels={LABELS}; gate = coupled within {GATE_K} steps; cp weights lambda={CP_LAMBDA} tau={CP_TAU} pre={CP_PRE}; aux_w={AUX_W}; sample_w={os.environ.get('DUO_SAMPLE_W', '0')}; prompt without state", flush=True)
    else:
        print(f"DUO {variant}: xarm={step.xarm} pdrop={step.pdrop}; prompt without state; gate = contact_rules.gate_train on the raw state/chunk; full real-action loss; padding masked", flush=True)


# ----------------------------------------------------------------------------------------------- model

class DuoPI05Pytorch(PI05Pytorch):
    """PI05Pytorch + the two-stream gated suffix (+ the v2 future-contact head). Instances are created by re-classing a
    loaded PI05Pytorch."""

    def init_duo(self, v2=None):
        self.v2 = v2
        W = self.action_in_proj.out_features
        ref = self.action_in_proj.weight
        kw = dict(device=ref.device, dtype=ref.dtype)
        self.action_in_L = nn.Linear(ARM, W, **kw); self.action_in_R = nn.Linear(ARM, W, **kw)
        self.action_out_L = nn.Linear(W, ARM, **kw); self.action_out_R = nn.Linear(W, ARM, **kw)
        self.state_in_L = nn.Linear(ARM + 1, W, **kw); self.state_in_R = nn.Linear(ARM + 1, W, **kw)
        self.ctok = bool(v2 is not None and v2.get("ctok"))
        self.stream_embed = nn.Parameter(torch.zeros(3 if self.ctok else 2, W, **kw))
        nn.init.normal_(self.stream_embed, std=0.02)
        for m in (self.state_in_L, self.state_in_R):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        if self.ctok:
            H = self.config.chunk_size; assert H % C_TOK == 0, (H, C_TOK)
            self.contact_queries = nn.Parameter(torch.zeros(C_TOK, W, **kw)); nn.init.normal_(self.contact_queries, std=0.02)
            self.contact_seg_head = nn.Linear(W, (H // C_TOK) * NMODE, **kw); self.contact_k_head = nn.Linear(W, 1, **kw)
            for m in (self.contact_seg_head, self.contact_k_head):
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        elif v2 is not None and v2["head"]:
            Wv = self.paligemma_with_expert.paligemma.config.text_config.hidden_size
            self.contact_head = nn.Sequential(nn.Linear(Wv + 2 * ARM + 2, 512, **kw), nn.GELU(), nn.Linear(512, self.config.chunk_size * NMODE + 1, **kw))
            nn.init.zeros_(self.contact_head[2].weight); nn.init.zeros_(self.contact_head[2].bias)
        self.last_pred = None
        self.init_duo_from_base()

    @torch.no_grad()
    def init_duo_from_base(self):
        """Per-stream projections = the pretrained 32-d projections restricted to that arm's 7 dims."""
        Wi, bi = self.action_in_proj.weight, self.action_in_proj.bias
        Wo, bo = self.action_out_proj.weight, self.action_out_proj.bias
        self.action_in_L.weight.copy_(Wi[:, :ARM]); self.action_in_L.bias.copy_(bi)
        self.action_in_R.weight.copy_(Wi[:, ARM:2 * ARM]); self.action_in_R.bias.copy_(bi)
        self.action_out_L.weight.copy_(Wo[:ARM]); self.action_out_L.bias.copy_(bo[:ARM])
        self.action_out_R.weight.copy_(Wo[ARM:2 * ARM]); self.action_out_R.bias.copy_(bo[ARM:2 * ARM])

    def has_head(self):
        return self.v2 is not None and self.v2["head"] and (hasattr(self, "contact_head") or self.ctok)

    def contact_logits_ctok(self, cout):
        """v3: contact token j predicts the modes of steps [j*seg, (j+1)*seg); token 0 also predicts coupled-within-K."""
        B = cout.shape[0]; dt = self.contact_seg_head.weight.dtype
        seg = self.contact_seg_head(cout.to(dt)).float().view(B, -1)                 # (B, H*5) in step order
        k = self.contact_k_head(cout[:, 0].to(dt)).float()                           # (B, 1)
        return torch.cat([seg, k], -1)

    def contact_logits(self, prefix_out, prefix_pad, state14, flags):
        """Future-contact head on the masked-mean VLM prefix output + own states + presence flags -> (B, H*5 + 1)."""
        m = prefix_pad.to(prefix_out.dtype)[..., None]
        if self.v2 is not None and self.v2.get("sg"):
            prefix_out = prefix_out.detach()
        pooled = (prefix_out * m).sum(1) / m.sum(1).clamp_min(1.0)
        x = torch.cat([pooled.float(), state14.float(), flags.float()], -1).to(self.contact_head[0].weight.dtype)
        return self.contact_head(x).float()

    # ---- suffix
    def embed_suffix_duo(self, x_t, timestep, state14, flags):
        W = self.action_in_proj.out_features
        time_emb = create_sinusoidal_pos_embedding(timestep, W, min_period=self.config.min_period, max_period=self.config.max_period, device=timestep.device)
        time_emb = time_emb.type(dtype=timestep.dtype)
        dt = self.action_in_L.weight.dtype
        aL = self.action_in_L(x_t[..., :ARM].to(dt)); aR = self.action_in_R(x_t[..., ARM:2 * ARM].to(dt))
        sL = self.state_in_L(torch.cat([state14[:, :ARM], flags[:, :1]], -1).to(dt))[:, None]
        sR = self.state_in_R(torch.cat([state14[:, ARM:2 * ARM], flags[:, 1:]], -1).to(dt))[:, None]
        e0, e1 = self.stream_embed[0].view(1, 1, W), self.stream_embed[1].view(1, 1, W)
        embs = torch.cat([sL + e0, aL + e0, sR + e1, aR + e1], dim=1)               # (B, 2*(1+H), W)
        if self.ctok:                                                                 # + C contact tokens (learned queries)
            e2 = self.stream_embed[2].view(1, 1, W)
            embs = torch.cat([embs, (self.contact_queries.to(embs.dtype) + e2).expand(embs.shape[0], -1, -1)], dim=1)

        def time_mlp_func(t):
            x = self.time_mlp_in(t); x = F.silu(x); x = self.time_mlp_out(x); return F.silu(x)
        adarms_cond = self._apply_checkpoint(time_mlp_func, time_emb)
        pad = torch.ones(embs.shape[0], embs.shape[1], dtype=torch.bool, device=embs.device)
        return embs, pad, adarms_cond

    def suffix_block_mask(self, gate, H):
        S = H + 1; B = gate.shape[0]; C = C_TOK if self.ctok else 0
        m = torch.zeros(B, 2 * S + C, 2 * S + C, dtype=torch.bool, device=gate.device)
        m[:, :S, :S] = True; m[:, S:2 * S, S:2 * S] = True
        g = (gate > 0.5).view(B, 1, 1)
        m[:, :S, S:2 * S] = g.expand(B, S, S); m[:, S:2 * S, :S] = g.expand(B, S, S)
        if C:
            m[:, 2 * S:, :] = True          # contact tokens read the whole suffix (both streams)
            m[:, :2 * S, 2 * S:] = True     # both streams always read the contact tokens (narrow, always-on channel)
        return m

    def split_out(self, suffix_out):
        H = self.config.chunk_size
        out = suffix_out.to(torch.float32)
        vL = self.action_out_L(out[:, 1:1 + H].to(self.action_out_L.weight.dtype)).float()
        vR = self.action_out_R(out[:, 2 + H:2 + 2 * H].to(self.action_out_R.weight.dtype)).float()
        return torch.cat([vL, vR], dim=-1)                                            # (B, H, 14)

    # ---- training
    def forward_duo(self, images, img_masks, tokens, masks, actions, noise, time, state14, flags, gate):
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, tokens, masks)
        suffix_embs, suffix_pad, adarms_cond = self.embed_suffix_duo(x_t, time, state14, flags)
        if self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16); prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        B, P = prefix_pad.shape; S2 = suffix_pad.shape[1]
        pad = torch.cat([prefix_pad, suffix_pad], dim=1)
        att = torch.zeros(B, P + S2, P + S2, dtype=torch.bool, device=pad.device)
        att[:, :P, :P] = make_att_2d_masks(prefix_pad, prefix_att)
        att[:, P:, :P] = prefix_pad[:, None, :].expand(B, S2, P)
        att[:, P:, P:] = self.suffix_block_mask(gate, self.config.chunk_size)
        position_ids = torch.cumsum(pad, dim=1) - 1
        att4 = self._prepare_attention_masks_4d(att)

        def forward_func(prefix_embs, suffix_embs, att4, position_ids, adarms_cond):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(attention_mask=att4, position_ids=position_ids, past_key_values=None,
                                                                             inputs_embeds=[prefix_embs, suffix_embs], use_cache=False, adarms_cond=[None, adarms_cond])
            return prefix_out, suffix_out
        prefix_out, suffix_out = self._apply_checkpoint(forward_func, prefix_embs, suffix_embs, att4, position_ids, adarms_cond)
        v_t = self.split_out(suffix_out[:, -S2:])
        if self.ctok:
            logits = self.contact_logits_ctok(suffix_out[:, -C_TOK:])
        else:
            logits = self.contact_logits(prefix_out[:, :P], prefix_pad, state14, flags) if self.has_head() else None
        return F.mse_loss(u_t[..., :2 * ARM], v_t, reduction="none"), logits         # (B, H, 14), (B, H*5+1)

    # ---- inference
    @torch.no_grad()
    def sample_actions_duo(self, images, img_masks, tokens, masks, state14, flags, gate, noise=None, num_steps=None):
        num_steps = num_steps or self.config.num_inference_steps
        B, device = tokens.shape[0], tokens.device
        if noise is None:
            noise = self.sample_noise((B, self.config.chunk_size, self.config.max_action_dim), device)
        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att4 = self._prepare_attention_masks_4d(make_att_2d_masks(prefix_pad, prefix_att))
        prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(attention_mask=prefix_att4, position_ids=prefix_pos, past_key_values=None,
                                                                              inputs_embeds=[prefix_embs, None], use_cache=True)
        rule_gate = gate.clone()
        if self.ctok:
            t0 = torch.ones(B, dtype=torch.float32, device=device)
            _, cout = self.denoise_step_duo(prefix_pad, past_key_values, noise, t0, state14, flags, gate, return_ctok=True)
            logits = self.contact_logits_ctok(cout)
        if self.has_head():
            if not self.ctok:
                logits = self.contact_logits(prefix_out, prefix_pad, state14, flags)
            p_couple = torch.sigmoid(logits[:, -1])
            head_gate = (p_couple > 0.5).float()
            modes = logits[:, :-1].view(B, self.config.chunk_size, NMODE).argmax(-1)
            if EVAL_GATE == "head":
                gate = head_gate
            elif EVAL_GATE == "or":
                gate = torch.maximum(gate, head_gate)
            self.last_pred = {"rule_gate": rule_gate.detach().cpu(), "head_gate": head_gate.detach().cpu(), "p_couple": p_couple.detach().cpu(),
                              "gate": gate.detach().cpu(), "modes": modes.detach().cpu()}
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            t = torch.tensor(1.0 + step * dt, dtype=torch.float32, device=device).expand(B)
            v14 = self.denoise_step_duo(prefix_pad, past_key_values, x_t, t, state14, flags, gate)
            v_t = torch.zeros_like(x_t); v_t[..., :2 * ARM] = v14
            x_t = x_t + dt * v_t
        return x_t

    def denoise_step_duo(self, prefix_pad, past_key_values, x_t, timestep, state14, flags, gate, return_ctok=False):
        suffix_embs, suffix_pad, adarms_cond = self.embed_suffix_duo(x_t, timestep, state14, flags)
        B, P = prefix_pad.shape; S2 = suffix_pad.shape[1]
        full = torch.cat([prefix_pad[:, None, :].expand(B, S2, P), self.suffix_block_mask(gate, self.config.chunk_size)], dim=2)
        position_ids = torch.sum(prefix_pad, dim=-1)[:, None] + torch.cumsum(suffix_pad, dim=1) - 1
        full4 = self._prepare_attention_masks_4d(full)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        pkv = clone_past_key_values(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(attention_mask=full4, position_ids=position_ids, past_key_values=pkv,
                                                               inputs_embeds=[None, suffix_embs], use_cache=False, adarms_cond=[None, adarms_cond])
        out = outputs_embeds[1][:, -S2:]
        if return_ctok:
            return self.split_out(out), out[:, -C_TOK:]
        return self.split_out(out)


class DuoPI05Policy(PI05Policy):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        variant = os.environ.get("COMPARISON_VARIANT", "duo")
        self.model.__class__ = DuoPI05Pytorch
        self.model.init_duo(V2.get(variant))

    def duo_inputs(self, batch, B, device):
        st = batch.get("duo_state")
        if st is None:
            raise KeyError("duo_state missing: the DuoProcessor step must run before the policy")
        flags = batch.get("duo_flags", torch.ones(B, 2, device=device))
        gate = batch.get("duo_gate", torch.zeros(B, device=device))
        return st.to(device), flags.to(device), torch.as_tensor(gate, dtype=torch.float32, device=device).view(B)

    def forward(self, batch, reduction="mean"):
        images, image_masks = self._preprocess_images(batch)
        actions = self.prepare_action(batch)
        B, device = actions.shape[0], actions.device
        noise = self.model.sample_noise(actions.shape, device)
        time = self.model.sample_time(B, device)
        st, flags, gate = self.duo_inputs(batch, B, device)
        losses, logits = self.model.forward_duo(images, image_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
                                                actions, noise, time, st, flags, gate)
        losses = losses[:, :, :self.config.output_features[ACTION].shape[0]]
        weights = batch["loss_w"].to(losses)
        if weights.shape != losses.shape:
            raise ValueError(f"Loss/mask shape mismatch: {losses.shape}, {weights.shape}")
        weighted = losses * weights
        per_sample = weighted.mean(dim=(1, 2))
        if reduction not in ("mean", "none"):
            raise ValueError(reduction)
        loss = per_sample if reduction == "none" else per_sample.mean()
        metrics = {"loss": per_sample.mean().item(), "loss_per_dim": weighted.mean(dim=(0, 1)).detach().cpu().tolist(),
                   "valid_fraction": (weights > 0).float().mean().item(), "valid_element_mse": (weighted.sum() / (weights > 0).float().sum().clamp_min(1)).item(),
                   "gate_fraction": gate.mean().item(), "flags_mean": flags.float().mean().item()}
        if logits is not None and "duo_future_mode" in batch:
            H = self.config.chunk_size
            fm = batch["duo_future_mode"].to(device).long().view(B, H)
            valid = batch.get("duo_valid", torch.ones(B, H, dtype=torch.bool, device=device)).to(device).float().view(B, H)
            sw = batch.get("duo_step_w", torch.ones(B, H, device=device)).to(device).float().view(B, H) * valid
            ce = F.cross_entropy(logits[:, :-1].reshape(B * H, NMODE), fm.reshape(-1), reduction="none").view(B, H)
            ce = (ce * sw).sum() / sw.sum().clamp_min(1.0)
            ck = batch["duo_couple_k"].to(device).float().view(B)
            bce = F.binary_cross_entropy_with_logits(logits[:, -1], ck)
            aux = ce + bce
            loss = loss + AUX_W * aux if reduction == "mean" else loss + AUX_W * aux
            with torch.no_grad():
                pred = logits[:, :-1].view(B, H, NMODE).argmax(-1)
                acc = ((pred == fm).float() * valid).sum() / valid.sum().clamp_min(1.0)
                cp_t = (fm >= 2).float(); cp_p = (pred >= 2).float()
                coupled_recall = ((cp_p * cp_t * valid).sum() / (cp_t * valid).sum().clamp_min(1.0))
                gate_acc = ((torch.sigmoid(logits[:, -1]) > 0.5).float() == ck).float().mean()
            metrics.update({"aux_ce": ce.item(), "aux_bce": bce.item(), "head_acc": acc.item(), "head_coupled_recall": coupled_recall.item(),
                            "head_gate_acc": gate_acc.item(), "couple_k_fraction": ck.mean().item(), "step_w_max": sw.max().item()})
        return loss, metrics

    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        self.eval()
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        B, device = tokens.shape[0], tokens.device
        st, flags, gate = self.duo_inputs(batch, B, device)
        actions = self.model.sample_actions_duo(images, img_masks, tokens, masks, st, flags, gate)
        return actions[:, :, :self.config.output_features[ACTION].shape[0]]


MaskedPI05Policy = DuoPI05Policy   # name expected by strict_policy.py / train.py of the reference pipeline
