#!/usr/bin/env python3

import copy
import math
import os
import pickle as pkl
import sys
import shutil
import time

import numpy as np

import envs
import hydra
import torch
import utils
from logger import Logger
from replay_buffer import ReplayBuffer
from video import VideoRecorder
from collections import defaultdict

from rrc_simulation.tasks import move_cube

torch.backends.cudnn.benchmark = True


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        self.model_dir = utils.make_dir(self.work_dir, 'model')
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             action_repeat=cfg.action_repeat,
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # make initializers
        self.train_initializer = envs.make_initializer(cfg.train_initializer,
                                                       cfg.difficulty,
                                                       cfg.curriculum_init_p,
                                                       cfg.curriculum_max_step)
        self.eval_initializer = envs.make_initializer(cfg.eval_initializer,
                                                      cfg.difficulty,
                                                      cfg.curriculum_init_p,
                                                      cfg.curriculum_max_step)
        self.true_eval_initializer = envs.make_initializer(
            'random', cfg.difficulty)

        # make envs
        self.env = envs.make(cfg.env, cfg.action_type, cfg.action_repeat,
                             cfg.episode_length, self.train_initializer,
                             cfg.seed, cfg.use_curriculum, cfg.start_shape,
                             cfg.goal_shape, cfg.curriculum_buffer_capacity,
                             cfg.R_min, cfg.R_max, cfg.new_goal_freq,
                             cfg.target_task_freq, cfg.n_random_actions)
        self.eval_env = envs.make(cfg.env, cfg.action_type, cfg.action_repeat,
                                  cfg.episode_length, self.eval_initializer,
                                  cfg.seed + 1, False)
        # always eval on the true target task
        self.true_eval_env = envs.make('cube', cfg.action_type,
                                       cfg.action_repeat,
                                       move_cube.episode_length,
                                       self.true_eval_initializer,
                                       cfg.seed + 1)

        obs_space = self.env.observation_space
        action_space = self.env.action_space

        cfg.agent.params.obs_shape = obs_space.shape
        cfg.agent.params.action_shape = action_space.shape
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(obs_space.shape, action_space.shape,
                                          cfg.replay_buffer_capacity,
                                          self.device, cfg.random_nstep)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            fps=cfg.video_fps // cfg.action_repeat)

        self.step = 0

    def evaluate(self, env, tag='eval'):
        assert tag in ['eval', 'true_eval']
        average_episode_reward = 0
        average_episode_length = 0
        average_reward_infos = defaultdict(float)
        denominator = 1 if tag == 'true_eval' else self.cfg.episode_length
        for episode in range(self.cfg.num_eval_episodes):
            obs = env.reset()
            self.video_recorder.init(enabled=(episode == 0 and tag == 'eval'))
            done = False
            episode_reward = 0
            episode_step = 0
            reward_infos = defaultdict(float)
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = env.step(action)
                for k, v in info.items():
                    if k.startswith('reward_'):
                        reward_infos[k] += v
                self.video_recorder.record()
                episode_reward += reward
                episode_step += 1

            average_episode_reward += episode_reward / denominator
            average_episode_length += episode_step
            for k, v in reward_infos.items():
                average_reward_infos[k] += v / denominator
            self.video_recorder.save(f'{self.step}.mp4')

        average_episode_reward /= self.cfg.num_eval_episodes
        average_episode_length /= self.cfg.num_eval_episodes
        self.logger.log(tag + '/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log(tag + '/episode_length', average_episode_length,
                        self.step)
        for k, v in average_reward_infos.items():
            self.logger.log(tag + f'/{k}', v / self.cfg.num_eval_episodes,
                            self.step)
        self.logger.dump(self.step, ty=tag)

    def run(self):
        episode, episode_reward, episode_step, done = 0, 0, 0, True
        reward_infos = defaultdict(float)
        start_time = time.time()
        assert self.cfg.eval_frequency % self.cfg.episode_length == 0, 'to prevent envs collision'
        while self.step < self.cfg.num_train_steps:
            # evaluate agent periodically
            if self.step % (self.cfg.eval_frequency //
                            self.cfg.action_repeat) == 0:
                self.logger.log('eval/episode', episode, self.step)
                self.evaluate(self.eval_env, tag='eval')
                if self.cfg.run_true_eval:
                    self.logger.log('true_eval/episode', episode, self.step)
                    self.evaluate(self.true_eval_env, tag='true_eval')

            if self.step % (self.cfg.save_frequency //
                            self.cfg.action_repeat) == 0:
                self.agent.save(self.model_dir, self.step)

            if done:
                if self.step > 0:
                    fps = episode_step / (time.time() - start_time)
                    self.logger.log('train/fps', fps, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step,
                        save=(self.step > self.cfg.num_seed_steps),
                        ty='train')

                self.logger.log('train/episode_reward',
                                episode_reward / self.cfg.episode_length,
                                self.step)
                for k, v in reward_infos.items():
                    self.logger.log(f'train/{k}', v / self.cfg.episode_length,
                                    self.step)

                self.train_initializer.update(self.step)
                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                reward_infos = defaultdict(float)

                self.logger.log('train/episode', episode, self.step)
                self.train_initializer.log(self.logger, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action_space = self.env.action_space
                action = np.random.uniform(action_space.low.min(),
                                           action_space.high.max(),
                                           action_space.shape)
                log_prob = -np.log(
                    np.prod(action_space.high - action_space.low))
            else:
                with utils.eval_mode(self.agent):
                    action, log_prob = self.agent.act(obs,
                                                      sample=True,
                                                      log_prob=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger,
                                      self.step)

            next_obs, reward, done, info = self.env.step(action)
            episode_reward += reward
            for k, v in info.items():
                if k.startswith('reward_'):
                    reward_infos[k] += v

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   log_prob)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='config.yaml')
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
