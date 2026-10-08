# [verbatim copy 2026-10-08] source: /mnt/workspace/yangyq/flower_duodit_robotwin/eval_adapter/eval_task.py (zzm's RoboTwin runtime: /mnt/workspace/zzm/idea/pi05_robotwin_behavior_probe/RoboTwin)
"""Run the upstream expert-filtered, unseen-language, budgeted evaluation protocol."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ROBOTWIN=Path('/mnt/workspace/zzm/idea/pi05_robotwin_behavior_probe/RoboTwin')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',required=True)
    parser.add_argument('--setting',choices=['demo_clean','demo_randomized'],required=True)
    parser.add_argument('--port',type=int,default=19504)
    parser.add_argument('--episodes',type=int,default=10)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ['BASELINE_EPISODE_LOG']=str(args.output/'episodes.jsonl')
    os.environ['BASELINE_EVAL_EPISODES']=str(args.episodes)
    os.chdir(ROOT/'runtime')
    sys.path.extend([str(ROBOTWIN),str(ROBOTWIN/'script'),str(ROBOTWIN/'description/utils')])
    import eval_official
    original=eval_official.eval_policy
    def recorded(*a,**kw):
        start=time.time()
        next_seed,success=original(*a,**kw)
        result=dict(task=args.task,setting=args.setting,episodes=args.episodes,successes=int(success),success_rate=success/args.episodes,next_seed=int(next_seed),elapsed_seconds=time.time()-start,preflight=os.environ.get('BASELINE_PREFLIGHT')=='1')
        (args.output/'result.json').write_text(json.dumps(result,indent=2))
        return next_seed,success
    eval_official.eval_policy=recorded
    eval_official.main(dict(task_name=args.task,task_config=args.setting,ckpt_setting='flower_duodit_clean2500',policy_name='baseline_policy',instruction_type='unseen',seed=0,port=args.port))


if __name__=='__main__':
    main()
    # SAPIEN's native shutdown can segfault after a fully saved result.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
