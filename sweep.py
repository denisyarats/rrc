import numpy as np
import numpy.random as npr

import time
import os
import argparse

from subprocess import Popen, DEVNULL

os.environ['MKL_THREADING_LAYER'] = 'GNU'


class Overrides(object):
    def __init__(self):
        self.kvs = dict()

    def add(self, key, values):
        processed_values = []
        for v in values:
            if type(v) == str:
                processed_values.append(v)
            else:
                processed_values.append(str(v))
        value = ','.join(processed_values)
        assert key not in self.kvs
        self.kvs[key] = value

    def cmd(self):
        cmd = []
        for k, v in self.kvs.items():
            cmd.append(f'{k}={v}')
        return cmd





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()

    overrides = Overrides()
    overrides.add('hydra/launcher', ['submitit'])
    overrides.add(key='experiment', values=[args.experiment])
    overrides.add(key='log_save_tb', values=['false'])
    overrides.add(key='video_fps', values=[10])

    overrides.add(key='env', values=['task1'])
    overrides.add(key='num_train_steps', values=[100000])
    overrides.add(key='eval_frequency', values=[10000])
    overrides.add(key='num_eval_episodes', values=[10])
    overrides.add(key='save_frequency', values=[50000])

    overrides.add(key='action_type', values=['torque'])
    overrides.add(key='episode_length', values=[250])
    overrides.add(key='train_initializer', values=['fixed_goal'])
    overrides.add(key='eval_initializer', values=['fixed_goal'])
    overrides.add(key='lr', values=[1e-4])
    overrides.add(key='batch_size', values=[128])
    overrides.add(key='actor_stddev', values=[0.2])
    overrides.add(key='nstep', values=[5])
    # seeds
    overrides.add(key='seed', values=list(range(1, 5)))

    cmd = ['python', 'train.py', '-m']
    cmd += overrides.cmd()

    if args.dry:
        print(cmd)
    else:
        env = os.environ.copy()
        p = Popen(cmd, env=env)
        p.communicate()


if __name__ == '__main__':
    main()
