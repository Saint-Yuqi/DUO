# [verbatim copy 2026-10-08] source: /mnt/workspace/yangyq/flower_duodit_robotwin/run_eval.py
"""Run RoboTwin's expert-filtered Easy/Hard evaluation after training finishes."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time

from lerobot.envs.robotwin import ROBOTWIN_TASKS

ROOT = Path('/mnt/workspace/yangyq/flower_duodit_robotwin')
ADAPTER = ROOT / 'eval_adapter'
TRAIN_PY = '/mnt/cpfs/yangyq/envs/flower/bin/python'
SIM_PY = '/mnt/workspace/wsh/envs/fastwam-py310/bin/python'
CHECKPOINT = ROOT / 'training/best.pt'
SETTINGS = ('demo_clean', 'demo_randomized')
GPUS = (4, 5, 6, 7)


def train_finished():
    path = ROOT / 'train.log'
    if not path.exists():
        return False
    with path.open('rb') as f:
        f.seek(max(0, path.stat().st_size - 8192))
        return b'End of training' in f.read()


def eval_gpu(gpu, tasks):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['HF_ENDPOINT'] = 'https://hf-mirror.com'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['__EGL_VENDOR_LIBRARY_FILENAMES'] = '/mnt/workspace/wsh/envs/fastwam-py310/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json'
    env['PYTHONPATH'] = '/mnt/workspace/yangyq/pi05_duo/vendor:' + str(ADAPTER)
    port = 19504 + gpu
    server_log = (ROOT / f'server_gpu{gpu}.log').open('w')
    server = subprocess.Popen([TRAIN_PY, str(ADAPTER / 'policy_server.py'), '--checkpoint', str(CHECKPOINT), '--port', str(port)], env=env, stdout=server_log, stderr=subprocess.STDOUT)
    try:
        for _ in range(180):
            if server.poll() is not None:
                raise RuntimeError(f'GPU {gpu} policy server exited; see server_gpu{gpu}.log')
            if f'READY {port}' in (ROOT / f'server_gpu{gpu}.log').read_text(errors='replace'):
                break
            time.sleep(1)
        else:
            raise TimeoutError(f'GPU {gpu} policy server did not start')
        for task in tasks:
            for setting in SETTINGS:
                output = ROOT / 'eval_results' / task / setting
                output.mkdir(parents=True, exist_ok=True)
                result = output / 'result.json'
                if result.exists() and json.loads(result.read_text()).get('episodes') == 100:
                    continue
                cmd = [SIM_PY, str(ADAPTER / 'eval_task.py'), '--task', task, '--setting', setting, '--port', str(port), '--episodes', '100', '--output', str(output)]
                for attempt in range(3):
                    with (output / 'eval.log').open('a') as log:
                        rc = subprocess.run(cmd, env=env, cwd=ADAPTER, stdout=log, stderr=subprocess.STDOUT).returncode
                    if result.exists() and json.loads(result.read_text()).get('episodes') == 100:
                        print(f'GPU {gpu}: {task} {setting}: {json.loads(result.read_text())["success_rate"]:.2%}', flush=True)
                        break
                    print(f'GPU {gpu}: {task} {setting}: attempt {attempt + 1} failed (exit {rc})', flush=True)
                else:
                    raise RuntimeError(f'{task} {setting} failed 3 times; see {output / "eval.log"}')
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        server_log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    tasks = list(ROBOTWIN_TASKS)
    assert len(tasks) == 50 and len(set(tasks)) == 50
    assignments = {gpu: tasks[i::4] for i, gpu in enumerate(GPUS)}
    assert sorted(t for group in assignments.values() for t in group) == sorted(tasks)
    if args.check:
        print('50 tasks, 100 Easy/Hard task-settings, 100 episodes each, GPUs 4–7')
        return
    pid = int((ROOT / 'train.pid').read_text())
    while not train_finished():
        if subprocess.run(['ps', '-p', str(pid)], stdout=subprocess.DEVNULL).returncode:
            raise RuntimeError('Training exited before End of training; see train.log')
        time.sleep(60)
    if not (CHECKPOINT / 'model.safetensors').exists():
        raise FileNotFoundError(CHECKPOINT / 'model.safetensors')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(eval_gpu, gpu, assignments[gpu]) for gpu in GPUS]
        for future in futures:
            future.result()
    rows = [json.loads(p.read_text()) for p in (ROOT / 'eval_results').glob('*/*/result.json')]
    assert len(rows) == 100 and all(x['episodes'] == 100 for x in rows)
    summary = {setting: {'successes': sum(x['successes'] for x in rows if x['setting'] == setting), 'episodes': 5000} for setting in SETTINGS}
    for value in summary.values():
        value['success_rate'] = value['successes'] / value['episodes']
    (ROOT / 'eval_summary.json').write_text(json.dumps(summary, indent=2))
    (ROOT / 'finished').touch()
    print(summary, flush=True)


if __name__ == '__main__':
    main()
