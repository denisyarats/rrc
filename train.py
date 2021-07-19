#!/usr/bin/env python3

import copy
import math
import os
import pickle as pkl
import sys
import shutil
import time

import numpy as np

import hydra
import torch
import utils
from logger import Logger
from replay_buffer import ReplayBuffer
from video import VideoRecorder, TrainVideoRecorder
from collections import defaultdict

#from rrc_simulation.tasks import move_cube

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
                             agent=cfg.agent.name)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        if cfg.use_old_simulator:
            import old_envs as envs
        else:
            import envs as envs

        # make initializers
        self.initializer = envs.make_initializer(cfg.train_initializer,
                                                 cfg.difficulty,
                                                 cfg.curriculum_init_p,
                                                 cfg.curriculum_max_step)
        
        
        # make envs
        self.env = envs.make(cfg.env,
                             cfg.action_type,
                             cfg.action_repeat,
                             cfg.episode_length,
                             cfg.num_corners,
                             self.initializer,
                             cfg.seed,
                             randomize=cfg.randomize,
                             obj_pos_noise_std=cfg.obj_pos_noise_std,
                             time_step=cfg.time_step,
                             time_step_range=cfg.time_step_range,
                             cube_mass=cfg.cube_mass,
                             cube_mass_range=cfg.cube_mass_range,
                             gravity=cfg.gravity,
                             gravity_range=cfg.gravity_range,
                             restitution=cfg.restitution,
                             restitution_range=cfg.restitution_range,
                             max_velocity=cfg.max_velocity,
                             max_velocity_range=cfg.max_velocity_range,
                             lateral_friction=cfg.lateral_friction,
                             lateral_friction_range=cfg.lateral_friction_range,
                             camera_rate_fps=cfg.camera_rate_fps,
                             camera_rate_fps_range=cfg.camera_rate_fps_range,
                             random_robot_position=cfg.random_robot_position,
                             delta_pos=cfg.delta_pos,
                             delta=cfg.delta)

        obs_space = self.env.observation_space
        action_space = self.env.action_space

        cfg.agent.params.obs_shape = obs_space.shape
        cfg.agent.params.obs_slices = self.env.obs_slices
        cfg.agent.params.action_shape = action_space.shape
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)
        if cfg.use_pretrained:
            self.agent.load(cfg.pretrained_model_dir, 1)

        if cfg.use_teacher:
            cfg.agent.params.excluded_obses = cfg.teacher_excluded_obses
            self.teacher = hydra.utils.instantiate(cfg.agent)
            self.teacher.load(cfg.teacher_model_dir, cfg.teacher_model_step)

        self.replay_buffer = ReplayBuffer(obs_space.shape, action_space.shape,
                                          cfg.replay_buffer_capacity,
                                          self.device, cfg.random_nstep)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None, fps=cfg.video_fps)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None, fps=cfg.video_fps)

        self.step = 0

    def evaluate(self, agent):
        average_episode_reward = 0
        average_episode_length = 0
        denominator = self.cfg.episode_length
        for episode in range(self.cfg.num_eval_episodes):
            visualize = (episode == 0)
            obs = self.env.reset(visualize_goal=visualize)
            self.video_recorder.init(enabled=visualize)
            done = False
            episode_reward = 0
            episode_step = 0
            reward_infos = defaultdict(float)
            while not done:
                with utils.eval_mode(agent):
                    action = agent.act(obs, sample=False)
                obs, reward, done, info = self.env.step(action)
                self.video_recorder.record()
                episode_reward += reward
                episode_step += 1

            average_episode_reward += episode_reward / denominator
            average_episode_length += episode_step
            self.video_recorder.save(f'{self.step}.mp4')

        average_episode_reward /= self.cfg.num_eval_episodes
        average_episode_length /= self.cfg.num_eval_episodes
        self.logger.log(f'eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/episode_length', average_episode_length,
                        self.step)
        self.logger.dump(self.step, ty='eval')

    def run(self):
        episode, episode_step, done = 0, 0, True
        episode_reward = 0
        start_time = time.time()

        while self.step <= self.cfg.num_train_steps:
            # evaluate agent periodically

            if self.step % (self.cfg.save_frequency) == 0:
                self.agent.save(self.model_dir, self.step)

            if done:
                if self.step > 0:
                    fps = episode_step / (time.time() - start_time)
                    self.logger.log('train/fps', fps, self.step)
                    start_time = time.time()

                    self.train_video_recorder.save(f'{self.step}.mp4')

                    self.logger.log(f'train/episode_reward',
                                    episode_reward / self.cfg.episode_length,
                                    self.step)

                    self.logger.dump(
                        self.step,
                        save=(self.step > self.cfg.num_seed_steps),
                        ty='train')

                if self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate(self.agent)

                ratio = max(self.cfg.teacher_max_step - self.step,
                            0) / self.cfg.teacher_max_step
                teacher_p = self.cfg.teacher_init_p * ratio
                teacher_steps = int(np.random.rand() * teacher_p *
                                    self.cfg.episode_length)
                self.logger.log('train/teacher_p', teacher_p, self.step)
                self.logger.log('train/teacher_steps', teacher_steps,
                                self.step)

                self.initializer.update(self.step)
                obs = self.env.reset()
                self.train_video_recorder.init(enabled=True)
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if not self.cfg.use_pretrained and self.step < self.cfg.num_seed_steps:
                action_space = self.env.action_space
                #import ipdb; ipdb.set_trace()
                action = np.random.uniform(action_space.low.min(),
                                           action_space.high.max(),
                                           action_space.shape)
            else:
                use_teacher = self.cfg.use_teacher and episode_step < teacher_steps
                agent = self.teacher if use_teacher else self.agent
                with utils.eval_mode(agent):
                    action = agent.act(obs, sample=not use_teacher)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger,
                                      self.step)

            next_obs, reward, done, info = self.env.step(action)
            self.train_video_recorder.record()
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='config.yaml')
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
