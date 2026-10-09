"""CPU checks for FlowerSharedDuo with the real Florence-2-base, the author's pretrained head and the real clean2500 cache.
Run on the cluster:  /mnt/cpfs/yangyq/envs/flower/bin/python smoke_cpu.py   (~5 min, no GPU)"""
import copy, json, os, subprocess, sys, time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dataset import RoboTwinDataset            # noqa: E402
from flower_sharedduo import FlowerSharedDuo   # noqa: E402

VLM = "/mnt/workspace/models/Florence-2-base"
HEAD = "/mnt/workspace/models/flower_vla_pret/360000_model_weights.pt"
CACHE = "/mnt/workspace/yangyq/data/flower_robotwin_clean2500"
torch.manual_seed(0); torch.set_num_threads(16)
ok = lambda name: print(f"PASS {name}", flush=True)

m = FlowerSharedDuo(VLM, pretrained_head=HEAD).eval()
rep = m.pretrain_report
assert rep["dit_loaded"] == "240/264", rep["dit_loaded"]
assert [x[0] for x in rep["shape_mismatch"]] == ["cond_linear.weight", "cond_norm.weight"], rep["shape_mismatch"]
pc = m.count_params(); print("PARAMS", {k: f"{v/1e6:.1f}M" for k, v in pc.items()})
assert 430e6 < pc["total"] < 450e6, pc
ok("pretrained head loads exactly like FlowerDuoDiT (240/264 DiT tensors; only cond_linear/cond_norm mismatch); ~438M params")

enc = m.action_encoders["bimanual_nav"]
x = torch.randn(3, 50, 14)
full = enc.fc1(F.pad(x, (0, 2)))
split = m.action_in_L.fc1(x[..., :7]) + m.action_in_R.fc1(x[..., 7:]) - enc.fc1.bias
assert torch.allclose(full, split, atol=1e-5)
assert torch.equal(m.action_in_L.fc2.weight, enc.fc2.weight) and m.action_in_L.fc2.weight.data_ptr() != enc.fc2.weight.data_ptr()
dec = m.action_decoders["bimanual_nav"]
assert torch.equal(m.action_out_L.weight, dec.weight[:7]) and torch.equal(m.action_out_R.bias, dec.bias[7:14])
assert float(m.state_in_L.weight.abs().sum()) == 0.0
ok("per-arm in/out projections are exact slices of the pretrained 16-d bimanual encoder/decoder; state tokens start at zero")

ds = RoboTwinDataset(CACHE, chunk=50)
items = [ds[i] for i in (1000, 200000)]
batch = {k: (torch.stack([torch.as_tensor(it[k]) for it in items]) if torch.is_tensor(items[0][k]) or isinstance(items[0][k], np.ndarray) else [it[k] for it in items]) for k in items[0]}
print("batch keys", {k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__) for k, v in batch.items()})
with torch.no_grad():
    cond = m.encode_observations(batch)
print("context", tuple(cond["context"].shape))
assert cond["context"].shape[1] > 150, cond["context"].shape  # 3 x 50 image tokens + prompt + text
ok("one Florence pass over three cameras + prompt + text")

def vel(c, x, t):
    with torch.no_grad():
        return m.velocity(x, t, c)
B = 2; xt = torch.randn(B, 50, 14); t = torch.full((B,), 0.6)
for gate, expect_dep in (("closed", False), ("open", True)):
    c = dict(cond); c["gate"] = torch.full((B,), 1.0 if gate == "open" else 0.0)
    v0 = vel(c, xt, t)
    x2 = xt.clone(); x2[..., 7:] = torch.randn_like(x2[..., 7:])        # change the right arm's noisy actions
    c2 = dict(c); s2 = c["state"].clone(); s2[:, 7:] += 0.5; c2["state"] = s2   # and its state
    v1 = vel(c2, x2, t)
    dep = not torch.allclose(v0[..., :7], v1[..., :7], atol=1e-5)
    assert dep == expect_dep, (gate, dep, (v0[..., :7] - v1[..., :7]).abs().max())
    assert not torch.allclose(v0[..., 7:], v1[..., 7:], atol=1e-5)
ok("gate closed: left velocity independent of right actions/state; gate open: dependent")

c = dict(cond); c["gate"] = torch.ones(B)
v0 = vel(c, xt, t); x3 = xt.clone(); x3[:, 30:] = torch.randn_like(x3[:, 30:])
v1 = vel(c, x3, t)
assert torch.allclose(v0[:, :30], v1[:, :30], atol=1e-5) and not torch.allclose(v0[:, 30:], v1[:, 30:], atol=1e-5)
ok("time-causal attention over both arms (steps < 30 unaffected by changes at steps >= 30), RoPE positions 0..H per arm")

m.train()
loss = m(batch); loss.backward()
assert torch.isfinite(loss)
nograd = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
bad = [n for n, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
assert not bad, bad
need = ["state_in_L.weight", "action_in_R.fc1.weight", "action_out_L.weight", "stream_embed", "dit.11.self_attn.qkv.weight", "vlm.vision_tower"]
have = {n for n, p in m.named_parameters() if p.grad is not None}
for k in need:
    assert any(h.startswith(k) for h in have), k
print("loss", float(loss), "| trainable params without grad:", sorted({n.split('.')[0] for n in nograd}))
ok("loss finite, gradients finite, new modules + DiT + VLM all receive gradients")
m.eval()
with torch.no_grad():
    a = m.predict(batch)
assert a.shape == (2, 50, 14) and torch.isfinite(a).all() and a.abs().max() <= 1
ok("predict -> (B, 50, 14) in [-1, 1]")

ck = Path("/tmp/flower_sharedduo_smoke/ckpt.pt"); ck.parent.mkdir(exist_ok=True)
torch.save({"model": m.state_dict(), "config": {"vlm": VLM, "gate": "open"},
            "norm": {"state": ds.norm_state.to_dict(), "action": ds.norm_action.to_dict()}}, ck)
port = 19987
log = open(ck.parent / "server.log", "w")
srv = subprocess.Popen([sys.executable, str(HERE / "policy_server.py"), "--checkpoint", str(ck), "--port", str(port), "--device", "cpu"], stdout=log, stderr=subprocess.STDOUT)
try:
    for _ in range(600):
        if f"READY {port}" in (ck.parent / "server.log").read_text(): break
        assert srv.poll() is None, (ck.parent / "server.log").read_text()[-2000:]
        time.sleep(1)
    conn = Client(("127.0.0.1", port), authkey=b"robotwin-baseline")
    conn.send({"op": "reset", "seed": 42}); assert conn.recv()["ok"]
    img = lambda k: np.asarray(batch[k][0].permute(1, 2, 0).numpy() if batch[k].dtype == torch.uint8 else (batch[k][0].permute(1, 2, 0).numpy() * 255), dtype=np.uint8)
    state_raw = np.zeros(14, np.float32) + 0.1
    conn.send({"op": "infer", "state": state_raw, "prompt": "pick up the bottle", "cam_high": np.zeros((240, 320, 3), np.uint8) + 100,
               "cam_left_wrist": img("rgb_gripper"), "cam_right_wrist": img("rgb_third")})
    acts = conn.recv()["actions"]
    assert acts.shape == (50, 14) and np.isfinite(acts).all()
    conn.close()
    ok("policy_server.py loads the checkpoint and answers the RoboTwin wire protocol with (50, 14) actions")
finally:
    srv.terminate()
print("ALL PASS")
