#!/usr/bin/env python3

import copy
import math
import os
import pickle as pkl
import sys
import shutil
import time

import numpy as np

from rrc import envs
import hydra
import torch
import utils
from rrc.logger import Logger
from rrc.video import VideoRecorder

from rrc.envs.tasks import ReachObject, ReachAndPush, RRC
from rrc.sacx.multi_replay_buffer import MultiReplayBuffer

torch.backends.cudnn.benchmark = True


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        self.model_dir = utils.make_dir(self.work_dir, 'model')
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        #------------------------------
        self.n_tasks = cfg.n_tasks

        self.loggers = []
        for i in range(self.n_tasks):
            dir = self.work_dir + f'/task_{i}'
            os.mkdir(dir)
            self.loggers.append(Logger(dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             action_repeat=cfg.action_repeat,
                             agent=cfg.agent.name))

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)


        initializer = envs.make_initializer(cfg.difficulty, cfg.fixed_env)
        task_list = []
        task_list.append(ReachObject(initializer))
        task_list.append(ReachAndPush(initializer))
        task_list.append(RRC(initializer))

        self.env = envs.make_multi(cfg.env, task_list, cfg.action_type,
                                cfg.action_repeat, cfg.episode_length, cfg.seed)
        self.eval_env = envs.make_multi(cfg.env, task_list, cfg.action_type,
                                cfg.action_repeat, cfg.episode_length, cfg.seed + 1)
#-------------------------------

        obs_space = self.env.observation_space
        action_space = self.env.action_space

        cfg.agent.params.obs_shape = obs_space.shape
        cfg.agent.params.action_shape = action_space.shape
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.buffer = MultiReplayBuffer(obs_space.shape,
                                        action_space.shape,
                                        cfg.replay_buffer_capacity,
                                        self.device, cfg.random_nstep,
                                        self.n_tasks)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            fps=cfg.video_fps // cfg.action_repeat)

        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        average_episode_length = 0
        for task_id in range(self.n_tasks):
            for episode in range(self.cfg.num_eval_episodes):
                obs = self.eval_env.reset(vis_id=task_id)
                self.video_recorder.init(enabled=(episode == 0))
                done = False
                episode_reward = 0
                episode_step = 0
                while not done:
                    with utils.eval_mode(self.agent):
                        action = self.agent.act(obs, task_id, sample=False)
                    obs, reward, done, info = self.eval_env.step(action)
                    self.video_recorder.record()
                    episode_reward += reward[task_id]
                    episode_step += 1

                average_episode_reward += episode_reward
                average_episode_length += episode_step
                self.video_recorder.save(f'{self.step}_{task_id}.mp4')
            average_episode_reward /= self.cfg.num_eval_episodes
            average_episode_length /= self.cfg.num_eval_episodes
            self.loggers[task_id].log(f'eval/episode_reward', average_episode_reward,
                            self.step)
            self.loggers[task_id].log(f'eval/episode_length', average_episode_length,
                            self.step)
            self.loggers[task_id].dump(self.step, ty='eval')

    def run(self):
        episode, episode_step, done = 0, 0, True
        episode_reward = np.zeros(self.n_tasks)
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            # query scheduler to select task
            if episode_step % self.cfg.switch_task_frequency == 0:
                task_id = self.agent.scheduler.choose_task()

            if done:
                if self.step > 0:
                    fps = episode_step / (time.time() - start_time)
                    start_time = time.time()

                    for i in range(self.n_tasks):
                        self.loggers[i].log('train/fps', fps, self.step)
                        self.loggers[i].dump(
                            self.step,
                            save=(self.step > self.cfg.num_seed_steps),
                            ty='train')

                for i in range(self.n_tasks):
                    self.loggers[i].log('train/episode_reward', episode_reward[i],
                            self.step)

                task_id = self.agent.scheduler.choose_task()
                self.env.set_task(task_id)

                obs = self.env.reset()
                done = False
                episode_reward = np.zeros(self.n_tasks)
                episode_step = 0
                episode += 1

                for i in range(self.n_tasks):
                    self.loggers[i].log('train/episode', episode, self.step)

            # evaluate agent periodically
            if self.step % (self.cfg.eval_frequency //
                            self.cfg.action_repeat) == 0:
                for i in range(self.n_tasks):
                    self.loggers[i].log('eval/episode', episode, self.step)
                #self.evaluate()

            if self.step % (self.cfg.save_frequency //
                            self.cfg.action_repeat) == 0:
                self.agent.save(self.model_dir, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action_space = self.env.action_space
                action = np.random.uniform(action_space.low.min(),
                                           action_space.high.max(),
                                           action_space.shape)
                log_prob = 0.0
            else:
                with utils.eval_mode(self.agent):
                    action, log_prob = self.agent.act(obs, task_id,
                                                    sample=True, log_prob=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.buffer, self.loggers,
                                      self.step)

            next_obs, reward, done, info = self.env.step(action)
            episode_reward += reward

            self.buffer.add(obs, action, reward, next_obs, done, log_prob)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='config.yaml')
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
