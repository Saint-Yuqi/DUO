# [verbatim copy 2026-10-08] source: /mnt/workspace/yangyq/flower_duodit_robotwin/eval_adapter/baseline_policy.py (zzm's RoboTwin runtime: /mnt/workspace/zzm/idea/pi05_robotwin_behavior_probe/RoboTwin)
"""RoboTwin's standard get_model/eval/reset_model interface, using a local policy process."""
import os
from multiprocessing.connection import Client
import numpy as np


def get_model(args):
    return {'connection':None if os.environ.get('BASELINE_PREFLIGHT')=='1' else Client(('127.0.0.1',int(args['port'])),authkey=b'robotwin-baseline'), 'episode':0}


def reset_model(model):
    if model['connection']:
        model['connection'].send({'op':'reset','seed':42+model['episode']})
        assert model['connection'].recv()['ok']
    model['episode']+=1


def eval(env,model,observation):
    cameras=observation['observation']
    state=np.asarray(observation['joint_action']['vector'],dtype=np.float32)
    assert state.shape==(14,)
    if model['connection']:
        model['connection'].send(dict(op='infer',state=state,prompt=env.get_instruction(),cam_high=cameras['head_camera']['rgb'],cam_left_wrist=cameras['left_camera']['rgb'],cam_right_wrist=cameras['right_camera']['rgb']))
        actions=model['connection'].recv()['actions']
    else:
        actions=np.repeat(state[None],50,axis=0)
    for action in actions:
        env.take_action(action)
        if env.eval_success or env.take_action_cnt>=env.step_lim: break
