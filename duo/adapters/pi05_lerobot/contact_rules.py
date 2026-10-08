"""[duo.adapters — verbatim copy, do not edit here]
source : /mnt/cpfs/yangyq/pi05_duo/contact_rules.py
copied : 2026-10-08 from quic5000 (as it ran; snapshot date 2026-09-28)
ledger : pi05 duo (v1 gate), DiT R3 (rule gate)
note   : FK-based contact-graph rules: S = both grippers closed & TCP distance < 0.30 m; A = one closed & moved < 3 cm in 15 steps while the other is within 0.30 m; gate_online for the policy server
Edit the canonical copy in the training workspace, then re-copy; the ledger row points at this file + the source path.
"""
"""Contact-graph rules for the RoboTwin aloha-agilex embodiment, shared by the offline labeler (contact_labels.py), the
training-time gate (comparison.py) and the online policy servers. Attachment is judged by kinematic consequences,
never by the gripper command alone:
  S  (rigid co-grasp): both grippers closed, TCPs within NEAR (offline labels also require a stationary relative pose)
  A  (anchor + actor): one gripper closed and its TCP stationary (past query online / next 15 action steps in training),
                       the other TCP working within NEAR of it
  gate = 1 for S and A, 0 otherwise (U / loose)."""
import xml.etree.ElementTree as ET

import numpy as np

NEAR = 0.30            # m between the two TCPs
TCP_X = 0.12           # RoboTwin gripper_bias: TCP 12 cm along link6 x
ANCHOR_MOVE = 0.03     # m: a closed hand that moves less than this over the horizon counts as an anchor
ANCHOR_STEPS = 15      # training: anchor stationarity judged over the next 15 action steps (0.5 s)


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]); Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]); Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _axis_rot(axis, q):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    c, s = np.cos(q)[:, None, None], np.sin(q)[:, None, None]
    return np.eye(3)[None] + s * K[None] + (1 - c) * (K @ K)[None]


class AlohaFK:
    """Forward kinematics of both arms from the embodiment URDF (footprint frame). state14 = [L6, Lgrip, R6, Rgrip]."""

    def __init__(self, urdf):
        root = ET.parse(urdf).getroot()
        joints = {j.get("name"): j for j in root.findall("joint")}
        self.chains = {}
        for prefix in ("fl", "fr"):
            seq = [f"{prefix}_base_joint"] + [f"{prefix}_joint{i}" for i in range(1, 7)]
            ch = []
            for n in seq:
                j = joints[n]; o = j.find("origin"); a = j.find("axis")
                xyz = np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()])
                R = _rpy(*[float(v) for v in (o.get("rpy") if o is not None and o.get("rpy") else "0 0 0").split()])
                ch.append((xyz, R, None if j.get("type") == "fixed" else np.array([float(v) for v in a.get("xyz").split()])))
            self.chains[prefix] = ch

    def _fk(self, ch, q):
        q = np.atleast_2d(np.asarray(q, float)); N = q.shape[0]
        p = np.zeros((N, 3)); R = np.tile(np.eye(3), (N, 1, 1)); k = 0
        for xyz, R0, axis in ch:
            p = p + np.einsum("nij,j->ni", R, xyz); R = R @ R0[None]
            if axis is not None:
                R = R @ _axis_rot(axis, q[:, k]); k += 1
        p = p + np.einsum("nij,j->ni", R, np.array([TCP_X, 0, 0]))
        return p, R

    def tcp(self, state14):
        """(N,14) -> pL (N,3), RL (N,3,3), pR, RR"""
        s = np.atleast_2d(np.asarray(state14, float))
        pL, RL = self._fk(self.chains["fl"], s[:, 0:6]); pR, RR = self._fk(self.chains["fr"], s[:, 7:13])
        return pL, RL, pR, RR


def gate_online(fk: AlohaFK, state14, prev_state14=None):
    """Causal gate from the current (and previous) measured state. Both closed & near -> 1 (tentative S: a false S only
    makes two free hands move consistently, a false U with two hands on one object is what breaks things). One closed,
    stationary since the previous query, other within NEAR -> 1 (A). Else 0."""
    s = np.asarray(state14, float)
    pL, _, pR, _ = fk.tcp(s); pL, pR = pL[0], pR[0]
    closedL, closedR = s[6] < 0.5, s[13] < 0.5
    d = float(np.linalg.norm(pL - pR))
    if closedL and closedR:
        return (1.0, "S") if d < NEAR else (0.0, "U")
    if closedL != closedR and d < NEAR:
        if prev_state14 is None:
            return 0.0, "L"
        qL, _, qR, _ = fk.tcp(np.asarray(prev_state14, float))
        moved = float(np.linalg.norm((pL - qL[0]) if closedL else (pR - qR[0])))
        if moved < ANCHOR_MOVE:
            return 1.0, "A_L" if closedL else "A_R"
    return 0.0, "U" if d >= NEAR else "L"


def gate_train(fk: AlohaFK, state14, actions):
    """Same rule family for a training batch, with the anchor stationarity judged on the action chunk ahead.
    state14 (B,14) raw, actions (B,H,14) raw absolute targets -> gate (B,) float32, modes list."""
    s = np.asarray(state14, float); a = np.asarray(actions, float)
    B = s.shape[0]
    pL, _, pR, _ = fk.tcp(s)
    k = min(ANCHOR_STEPS, a.shape[1]) - 1
    fL, _, fR, _ = fk.tcp(a[:, k])
    closedL, closedR = s[:, 6] < 0.5, s[:, 13] < 0.5
    d = np.linalg.norm(pL - pR, axis=1)
    movedL, movedR = np.linalg.norm(fL - pL, axis=1), np.linalg.norm(fR - pR, axis=1)
    S = closedL & closedR & (d < NEAR)
    AL = closedL & ~closedR & (d < NEAR) & (movedL < ANCHOR_MOVE)
    AR = closedR & ~closedL & (d < NEAR) & (movedR < ANCHOR_MOVE)
    gate = (S | AL | AR).astype(np.float32)
    modes = ["S" if S[i] else "A_L" if AL[i] else "A_R" if AR[i] else ("U" if d[i] >= NEAR else "L") for i in range(B)]
    return gate, modes
