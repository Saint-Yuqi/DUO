"""CPU tests for the duo library. Run: python -m pytest tests/ -q   (or python tests/test_duo_cpu.py)"""
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from duo import (Partition, DuoFlowHead, CrossArm, CrossArmSet, Attention, enable_rope, rotate,
                 block_cross_stream_attention, warm_start_from_single_stream, gate_from_batch)

torch.manual_seed(0)
B, H, D = 3, 10, 32


def ctx_for(part, n_obs=5, dim=D):
    return [{"obs": torch.randn(B, n_obs + i, dim), "pooled": torch.randn(B, dim)} for i in range(part.n_streams)]


def test_partition_roundtrip():
    for p in (Partition.arm_semantic(7), Partition.mixed(7), Partition.eef_openwam80(), Partition.joint(14)):
        x = torch.randn(B, H, p.total_dim)
        y = p.merge(p.split(x))
        idx = p.covered
        assert torch.equal(x[..., idx], y[..., idx]), p.name
        assert sum(p.stream_dims) == len(idx)
    m = Partition.mixed(7)
    assert m.streams == ((0, 1, 2, 7, 8, 9, 6), (3, 4, 5, 10, 11, 12, 13))
    e = Partition.eef_openwam80()
    assert e.streams[1][0] == 34 and e.streams[1][-1] == 43


def test_zero_init_identity_means_decoupled():
    """At init the cross-arm gain is 0, so the Duo head equals two independent streams: changing the other arm's
    observation must not change this arm's velocity."""
    part = Partition.arm_semantic(7)
    head = DuoFlowHead(part, dim=D, layers=4, heads=4, chunk=H, cross_layers=(2, 4)).eval()
    for s in head.streams:   # the output layers are zero-initialised (adaLN-zero); randomise them so outputs depend on inputs
        torch.nn.init.normal_(s.act_out.weight, std=0.1); torch.nn.init.normal_(s.final_ada[1].weight, std=0.1)
    ctx = ctx_for(part)
    x = torch.randn(B, H, 14); t = torch.rand(B)
    v = head.velocity(x, t, ctx)
    ctx2 = copy.deepcopy(ctx); ctx2[1]["obs"] = torch.randn_like(ctx2[1]["obs"])
    v2 = head.velocity(x, t, ctx2)
    assert torch.allclose(v[..., :7], v2[..., :7], atol=1e-6)
    assert not torch.allclose(v[..., 7:], v2[..., 7:])
    # open the gain -> the left arm now depends on the right observation
    for m in head.cross.mods.values():
        m.gain.data.fill_(1.0)
    v3 = head.velocity(x, t, ctx2)
    assert not torch.allclose(v[..., :7], v3[..., :7])
    # gate = 0 closes it again regardless of the gain
    v4 = head.velocity(x, t, ctx2, gate=torch.zeros(B))
    assert torch.allclose(v[..., :7], v4[..., :7], atol=1e-6)


def test_loss_and_sample_shapes_all_partitions():
    for part in (Partition.arm_semantic(7), Partition.mixed(7), Partition.joint(14)):
        head = DuoFlowHead(part, dim=D, layers=2, heads=4, chunk=H)
        ctx = ctx_for(part)
        loss = head.loss(torch.randn(B, H, 14), ctx, mask=torch.ones(B, H))
        assert torch.isfinite(loss)
        loss.backward()
        s = head.sample(ctx, steps=2)
        assert s.shape == (B, H, 14)


def test_shared_streams_with_stream_embedding():
    part = Partition.arm_semantic(7)
    head = DuoFlowHead(part, dim=D, layers=2, heads=4, chunk=H, share_streams=True, rope=True)
    assert head.streams[0] is head.streams[1]
    v = head.velocity(torch.randn(B, H, 14), torch.rand(B), ctx_for(part))
    assert v.shape == (B, H, 14)


def test_rope_relative_position():
    q = torch.randn(1, 1, 1, 16); k = torch.randn(1, 1, 1, 16)
    a = torch.cat((q, k), dim=2)
    b = torch.cat((torch.zeros_like(q), q, k), dim=2)
    sa = (rotate(a)[:, :, 0] * rotate(a)[:, :, 1]).sum()
    sb = (rotate(b)[:, :, 1] * rotate(b)[:, :, 2]).sum()
    assert torch.allclose(sa, sb, atol=1e-5)
    att = Attention(D, 4, qk_norm=True)
    y0 = att(torch.randn(B, H, D))
    enable_rope(att)
    assert att.rope and att(torch.randn(B, H, D)).shape == y0.shape


def test_cross_arm_set_simultaneous_update():
    xs = CrossArmSet(D, 4, layers=(1,), n_streams=2)
    for m in xs.mods.values():
        m.gain.data.fill_(0.5)
    hL, hR = torch.randn(B, H, D), torch.randn(B, H, D)
    oL, oR = xs.apply(1, [hL, hR])
    # L must have been updated from the ORIGINAL R (not the updated one)
    ref = hL + xs.mods[xs.key(1, 0, 1)](hL, hR)
    assert torch.allclose(oL, ref, atol=1e-6)
    assert len(xs.strengths()) == 2 and abs(xs.mean_strength() - float(torch.tanh(torch.tensor(0.5)))) < 1e-6


def test_mot_mask_blocks_streams():
    S_v, S_a = 6, 8
    mask = torch.ones(S_v + S_a, S_v + S_a, dtype=torch.bool)
    block_cross_stream_attention(mask, S_v, S_a, 2)
    assert mask[:S_v].all() and mask[S_v:S_v + 4, S_v:S_v + 4].all()
    assert not mask[S_v:S_v + 4, S_v + 4:].any() and not mask[S_v + 4:, S_v:S_v + 4].any()
    assert mask[S_v + 4:, :S_v].all()


def test_warm_start_slicing_equivalence():
    """A single 14-d input projection equals the sum of its two sliced per-arm projections (+ one bias)."""
    part = Partition.arm_semantic(7)
    single = DuoFlowHead(Partition.joint(14), dim=D, layers=2, heads=4, chunk=H)
    duo = DuoFlowHead(part, dim=D, layers=2, heads=4, chunk=H, cross_layers=(2,))
    sd = warm_start_from_single_stream(duo.state_dict(), single.state_dict(), part,
                                       stream_prefixes=("streams.0.", "streams.1."), source_prefix="streams.0.",
                                       new_prefixes=("cross.",))
    duo.load_state_dict(sd)
    x = torch.randn(B, H, 14)
    joint = single.streams[0].act_in(x) - single.streams[0].act_in.bias
    split = duo.streams[0].act_in(x[..., :7]) + duo.streams[1].act_in(x[..., 7:]) - 2 * single.streams[0].act_in.bias
    assert torch.allclose(joint, split, atol=1e-5)
    assert torch.equal(duo.streams[1].act_out.weight, single.streams[0].act_out.weight[7:])
    assert duo.cross.mean_strength() == 0.0


def test_gate_from_batch():
    assert gate_from_batch({}, 4, "cpu", "open").sum() == 4
    g = gate_from_batch({"contact": torch.tensor([0, 1, 2, 4])}, 4, "cpu", "rule")
    assert g.tolist() == [0.0, 0.0, 1.0, 1.0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
