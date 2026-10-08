"""Turn a finished RoboTwin evaluation into the ledger's result block (and check it is complete).

Supported layouts (auto-detected):
  A. ours (run_eval_fast.py):        <root>/eval_results/<task>/<setting>/result.json  with episodes / successes
  B. zzm runs (pi05_duo/runs/<run>):  <root>/summary.json  with evaluation.<setting>.per_task
  C. OpenWAM / XPolicyLab:            <root>/**/eval_result/<task>/**/<setting>/**/_result.txt  (last line = rate 0..1)
Prints YAML you can paste under `results:` of a run in ledger/runs.yaml, plus per-task rates to ledger/per_task/<run>.json.
Usage: python eval/robotwin/summarize.py <root> --run-id <id> [--episodes 100] [--out ledger/per_task]
"""
import argparse
import glob
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = [t for t in (HERE / "tasks50.txt").read_text().split() if t]
SETTINGS = ("demo_clean", "demo_randomized")


def layout_a(root):
    out = {}
    for s in SETTINGS:
        out[s] = {}
        for t in TASKS:
            f = root / "eval_results" / t / s / "result.json"
            if f.exists():
                d = json.loads(f.read_text())
                out[s][t] = (d["successes"], d["episodes"])
    return out if any(out[s] for s in SETTINGS) else None


def layout_b(root):
    f = root / "summary.json"
    if not f.exists():
        return None
    ev = json.loads(f.read_text())["evaluation"]
    return {s: {r["task"]: (r["successes"], r["episodes"]) for r in ev.get(s, {}).get("per_task", [])} for s in SETTINGS}


def layout_c(root, episodes):
    out = {s: {} for s in SETTINGS}
    for f in glob.glob(str(root / "**" / "eval_result" / "*" / "**" / "_result.txt"), recursive=True):
        parts = f.split(os.sep)
        task = parts[parts.index("eval_result") + 1]
        setting = next((s for s in SETTINGS if s in parts), None)
        if setting is None or task not in TASKS:
            continue
        rate = float(open(f).read().strip().splitlines()[-1])
        prev = out[setting].get(task)
        if prev is None or os.path.getmtime(f) > prev[2]:
            out[setting][task] = (round(rate * episodes), episodes, os.path.getmtime(f))
    out = {s: {t: v[:2] for t, v in d.items()} for s, d in out.items()}
    return out if any(out[s] for s in SETTINGS) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--run-id", required=True); ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default=str(HERE.parents[1] / "ledger" / "per_task"))
    a = ap.parse_args()
    root = Path(a.root)
    res = layout_a(root) or layout_b(root) or layout_c(root, a.episodes)
    if res is None:
        raise SystemExit(f"no results found under {root}")
    block, per_task, complete = {}, {}, True
    for s in SETTINGS:
        d = res[s]
        succ = sum(v[0] for v in d.values()); eps = sum(v[1] for v in d.values())
        missing = [t for t in TASKS if t not in d or d[t][1] < a.episodes]
        complete &= not missing
        key = "clean" if s == "demo_clean" else "random"
        block[key] = round(100 * succ / eps, 2) if eps else None
        block[key + "_n"] = f"{succ}/{eps}"
        if missing:
            block[key + "_missing"] = missing
        per_task[s] = {t: round(100 * v[0] / v[1], 1) for t, v in d.items()}
    if block.get("clean") is not None and block.get("random") is not None:
        block["overall"] = round((block["clean"] + block["random"]) / 2, 2)
    block["complete"] = complete
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"{a.run_id}.json").write_text(json.dumps(per_task, indent=1))
    print("results:")
    for k, v in block.items():
        print(f"  {k}: {json.dumps(v) if isinstance(v, list) else v}")
    print(f"# per-task rates -> {Path(a.out) / (a.run_id + '.json')}")


if __name__ == "__main__":
    main()
