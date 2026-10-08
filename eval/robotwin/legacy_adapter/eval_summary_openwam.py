# [verbatim copy 2026-10-08] source: /tmp/robotwin_native_dual/eval_summary.py (quic5000) — how the OpenWAM official eval is summarised
"""Summarize the complete official OpenWAM RoboTwin evaluation."""
import json
from pathlib import Path

root = Path('/tmp/robotwin_native_dual')
tasks = [task for gpu in range(4, 8)
         for task in (root / f'eval_tasks_gpu{gpu}.txt').read_text().splitlines()]
assert len(tasks) == len(set(tasks)) == 50
results = {}
for mode in ('demo_clean', 'demo_randomized'):
    values = {}
    for task in tasks:
        files = list(root.glob(
            f'eval_runtime_gpu*/eval_result/{task}/openwam2robotwin_interface/'
            f'{mode}/openwam_native_dual/*/_result.txt'))
        if not files:
            raise FileNotFoundError(f'{task} {mode} has no official result')
        rate = float(max(files, key=lambda path: path.stat().st_mtime).read_text().splitlines()[-1])
        if not 0 <= rate <= 1:
            raise ValueError(f'{task} {mode}: {rate}')
        values[task] = round(rate * 100)
    successes = sum(values.values())
    results[mode] = {'successes': successes, 'episodes': 5000,
                     'success_rate': successes / 5000, 'tasks': values}
output = root / 'eval_summary.json'
output.write_text(json.dumps(results, indent=2) + '\n')
print(output)
print(f"OpenWAM native dual ActionDiT Easy {results['demo_clean']['success_rate']*100:.2f}%, "
      f"Hard {results['demo_randomized']['success_rate']*100:.2f}%, "
      f"Average {(results['demo_clean']['success_rate']+results['demo_randomized']['success_rate'])*50:.2f}%")
