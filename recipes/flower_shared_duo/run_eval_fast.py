"""Resume the same RoboTwin protocol with 6 simulator workers per GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from queue import Empty, Queue
import subprocess
import time

from lerobot.envs.robotwin import ROBOTWIN_TASKS
from run_eval import ROOT, ADAPTER, TRAIN_PY, SIM_PY, CHECKPOINT, SETTINGS, GPUS

WORKERS_PER_GPU = int(os.environ.get('WORKERS_PER_GPU', '6'))


def complete(task, setting):
    path = ROOT / 'eval_results' / task / setting / 'result.json'
    try:
        return json.loads(path.read_text()).get('episodes') == 100
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def active_jobs():
    jobs = set()
    for line in subprocess.check_output(['ps', '-eo', 'args'], text=True).splitlines():
        fields = line.split()
        if 'eval_task.py' not in line or '--task' not in fields or '--setting' not in fields:
            continue
        jobs.add((fields[fields.index('--task') + 1], fields[fields.index('--setting') + 1]))
    return jobs


def worker(index, queue):
    gpu = GPUS[index % len(GPUS)]
    port = 19600 + index
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), HF_ENDPOINT='https://hf-mirror.com',
               TOKENIZERS_PARALLELISM='false',
               __EGL_VENDOR_LIBRARY_FILENAMES='/mnt/workspace/wsh/envs/fastwam-py310/lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json',
               PYTHONPATH='/mnt/workspace/yangyq/pi05_duo/vendor:' + str(ADAPTER))
    log_path = ROOT / f'fast_server_{index}.log'
    with log_path.open('w') as server_log:
        server = subprocess.Popen([TRAIN_PY, str(ROOT / 'policy_server.py'), '--checkpoint', str(CHECKPOINT), '--port', str(port)],
                                  env=env, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            for _ in range(900):
                if server.poll() is not None:
                    raise RuntimeError(f'policy server {index} exited; see {log_path}')
                if f'READY {port}' in log_path.read_text(errors='replace'):
                    break
                time.sleep(1)
            else:
                raise TimeoutError(f'policy server {index} did not start')
            while True:
                try:
                    task, setting = queue.get_nowait()
                except Empty:
                    return
                try:
                    if complete(task, setting):
                        continue
                    out = ROOT / 'eval_results' / task / setting
                    out.mkdir(parents=True, exist_ok=True)
                    cmd = [SIM_PY, str(ADAPTER / 'eval_task.py'), '--task', task, '--setting', setting,
                           '--port', str(port), '--episodes', '100', '--output', str(out)]
                    for attempt in range(3):
                        with (out / 'eval.log').open('a') as log:
                            rc = subprocess.run(cmd, env=env, cwd=ADAPTER, stdout=log, stderr=subprocess.STDOUT).returncode
                        if complete(task, setting):
                            print(f'GPU {gpu} worker {index}: {task} {setting} done', flush=True)
                            break
                        print(f'GPU {gpu} worker {index}: {task} {setting} attempt {attempt + 1} exit {rc}', flush=True)
                    else:
                        raise RuntimeError(f'{task} {setting} failed 3 times; see {out / "eval.log"}')
                finally:
                    queue.task_done()
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    assert len(ROBOTWIN_TASKS) == 50 and len(set(ROBOTWIN_TASKS)) == 50
    all_jobs = [(task, setting) for task in ROBOTWIN_TASKS for setting in SETTINGS]
    assert len(all_jobs) == 100 and len(set(all_jobs)) == 100
    if args.check:
        print(f'100 task-settings, 100 episodes each, {WORKERS_PER_GPU * len(GPUS)} workers on GPUs {GPUS}; checkpoint {CHECKPOINT}')
        return
    assert CHECKPOINT.exists()
    active = active_jobs()
    pending = [job for job in all_jobs if not complete(*job)]
    queue = Queue()
    for job in [x for x in pending if x not in active] + [x for x in pending if x in active]:
        queue.put(job)
    print(f'Pending {len(pending)}; active legacy jobs last: {sorted(active)}', flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS_PER_GPU * len(GPUS)) as pool:
        futures = [pool.submit(worker, index, queue) for index in range(WORKERS_PER_GPU * len(GPUS))]
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
