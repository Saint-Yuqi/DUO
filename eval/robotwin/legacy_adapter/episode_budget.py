# [verbatim copy 2026-10-08] source: /mnt/workspace/yangyq/flower_duodit_robotwin/eval_adapter/episode_budget.py (zzm's RoboTwin runtime: /mnt/workspace/zzm/idea/pi05_robotwin_behavior_probe/RoboTwin)
"""Keep the first N valid episodes and resume incomplete prefixes."""
import json
from pathlib import Path


def read_records(path):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    assert all(a['seed'] < b['seed'] for a,b in zip(rows,rows[1:])), 'Duplicate or unordered episode seeds'
    return rows


def resume_state(rows, target):
    assert target > 0
    selected = rows[:target]
    return dict(episodes=len(selected), successes=sum(bool(r['success']) for r in selected),
                next_seed=selected[-1]['seed']+1 if selected else None)


def prefix_result(rows, target, task, setting):
    assert len(rows) >= target
    assert all(r['task'] == task for r in rows)
    state = resume_state(rows,target)
    return dict(task=task,setting=setting,**state,success_rate=state['successes']/target,
                preflight=False,source_episodes=len(rows),selection=f'first_{target}_valid_episodes')
