"""Train FLOWER + shared-weight Duo (flower_sharedduo.FlowerSharedDuo) on the 2,500 clean RoboTwin demonstrations.
Identical to /mnt/workspace/yangyq/flower_duodit_robotwin/train.py (the 68.30 / 26.62 FlowBlock-Duo run) except: the model
class, --gate, --device (cpu for smoke tests) and the --smoke-* limits."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
import numpy as np

from dataset import RoboTwinDataset
from flower_sharedduo import FlowerSharedDuo


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--vlm", default="/mnt/workspace/models/Florence-2-base")
    p.add_argument("--pretrained-head", default="/mnt/workspace/models/flower_vla_pret/360000_model_weights.pt")
    p.add_argument("--batch", type=int, default=2, help="per GPU microbatch")
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=60000)
    p.add_argument("--min-steps", type=int, default=15000)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr-head", type=float, default=1e-4)
    p.add_argument("--lr-vlm", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze-vlm", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--gate", choices=("open", "closed"), default="open", help="open = the official pi0.5-Duo setting")
    p.add_argument("--device", default="cuda", help="cpu only for smoke tests")
    p.add_argument("--smoke-frames", type=int, default=0, help=">0: train on the first N frames only (smoke test)")
    p.add_argument("--smoke-val", type=int, default=0, help=">0: validate on N anchors only (smoke test)")
    return p.parse_args()


def device_batch(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def val_loss(model, loader, device):
    model.eval()
    losses = []
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []), torch.no_grad():
        torch.manual_seed(12345)
        for batch in loader:
            batch = device_batch(batch, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                losses.append(float(model(batch).float()))
    model.train()
    return sum(losses) / len(losses)


def main():
    cfg = args()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl" if cfg.device == "cuda" else "gloo")
    if cfg.device == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    torch.set_num_threads(4)
    torch.manual_seed(cfg.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    dataset = RoboTwinDataset(cfg.cache, chunk=50)
    assert len(dataset.ep_list) == 2500 and len(set(dataset.ep_list)) == 2500
    assert all(550 * t + j in dataset.episodes for t in range(50) for j in range(50))
    train_set = Subset(dataset, list(range(cfg.smoke_frames))) if cfg.smoke_frames > 0 else dataset
    sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    train_loader = DataLoader(train_set, batch_size=cfg.batch, sampler=sampler, num_workers=cfg.workers,
                              pin_memory=True, persistent_workers=cfg.workers > 0, drop_last=True)
    anchors = []
    for task in range(50):
        episode = dataset.episodes[550 * task + 49]
        middle = (episode["from"] + episode["to"]) // 2
        anchors.append(int(np.searchsorted(dataset.frames, middle)))
    anchors = anchors[:cfg.smoke_val] if cfg.smoke_val > 0 else anchors
    validation = DataLoader(Subset(dataset, anchors), batch_size=cfg.batch,
                            shuffle=False, num_workers=2, pin_memory=True)
    model = FlowerSharedDuo(cfg.vlm, freeze_vlm=cfg.freeze_vlm, gate=cfg.gate,
                            pretrained_head=cfg.pretrained_head if not cfg.resume else None)
    if rank == 0:
        print("PRETRAIN", model.pretrain_report, flush=True)
        print("FRAMES", len(dataset), "EPISODES", len(dataset.ep_list), flush=True)
        print("PARAMS", model.count_params(), flush=True)
        (out / "config.json").write_text(json.dumps(vars(cfg), indent=2))
    model.to(device)
    parameters = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("vlm.")],
         "lr": cfg.lr_vlm},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("vlm.")],
         "lr": cfg.lr_head},
    ]
    optimizer = torch.optim.AdamW(parameters, weight_decay=0.01)
    start, best, bad = 0, float("inf"), 0
    if cfg.resume:
        ckpt = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start, best, bad = ckpt["step"], ckpt["best"], ckpt["bad"]
    wrapped = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=True) if world > 1 else model
    epoch = 0
    sampler.set_epoch(epoch)
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    for step in range(start + 1, cfg.max_steps + 1):
        total_loss = 0.0
        for micro in range(cfg.accum):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(train_loader)
                batch = next(iterator)
            batch = device_batch(batch, device)
            sync = wrapped.no_sync() if world > 1 and micro < cfg.accum - 1 else nullcontext()
            with sync, torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = wrapped(batch) / cfg.accum
            loss.backward()
            total_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        multiplier = min(1.0, step / 500) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / cfg.max_steps)))
        for group, base in zip(optimizer.param_groups, (cfg.lr_vlm, cfg.lr_head)):
            group["lr"] = base * multiplier
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 50 == 0 and rank == 0:
            row = {"step": step, "train_loss": total_loss, "epoch": epoch,
                   "minutes": round((time.time() - started) / 60, 2)}
            with (out / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(row, flush=True)
        if step % cfg.val_every == 0 or step == cfg.max_steps:
            if world > 1:
                dist.barrier()
            if rank == 0:
                value = val_loss(model, validation, device)
                improved = value < best * 0.998
                best, bad = (value, 0) if improved else (best, bad + 1)
                record = {"step": step, "val_loss": value, "best": best, "bad": bad}
                with (out / "val.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print("VAL", record, flush=True)
                payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                           "step": step, "config": vars(cfg),
                           "norm": {"state": dataset.norm_state.to_dict(),
                                    "action": dataset.norm_action.to_dict()}}
                if improved:
                    torch.save(payload, out / "best.pt.tmp")
                    (out / "best.pt.tmp").replace(out / "best.pt")
                payload.update(optimizer=optimizer.state_dict(), best=best, bad=bad)
                torch.save(payload, out / "last.pt.tmp")
                (out / "last.pt.tmp").replace(out / "last.pt")
                stop = step >= cfg.min_steps and bad >= cfg.patience
                (out / "stop.flag").write_text("1" if stop else "0")
            if world > 1:
                dist.barrier()
            if (out / "stop.flag").read_text().strip() == "1":
                if rank == 0:
                    print("CONVERGED", step, flush=True)
                break
    if rank == 0:
        (out / "DONE").write_text(str(step))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
