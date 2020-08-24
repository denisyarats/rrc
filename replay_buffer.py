import numpy as np

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from dm_control.utils import rewards as dm_rewards
from rrc_simulation.tasks import move_cube


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device, random_nstep,
                 obs_specs, episode_length, her_k):
        self.capacity = capacity
        self.device = device
        self.random_nstep = random_nstep
        self.obs_specs = obs_specs
        self.episode_length = episode_length
        self.future_p = 1.0 - (1.0 / (1.0 + her_k))

        self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.dones[self.idx], float(done))

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size, discount, n):
        assert n <= self.idx or self.full
        last_idx = (self.capacity if self.full else self.idx) - (n - 1)
        idxs = np.random.randint(0, last_idx, size=batch_size)
        ns = np.random.randint(1, n + 1, size=(batch_size, 1))

        #print(f'idx: {idxs.sum()}')

        assert idxs.max() + n <= len(self)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        rewards = np.zeros((idxs.shape[0], 1), dtype=np.float32)
        #import ipdb; ipdb.set_trace()
        dones = np.zeros((idxs.shape[0], 1), dtype=np.float32)
        discounts = np.ones((idxs.shape[0], 1), dtype=np.float32)

        for i in range(n):
            rewards += discounts * (1 - dones) * self.rewards[idxs + i]
            # if done keep the next observation same, otherwise overwrite it
            next_obses = dones * next_obses + (
                1 - dones) * self.next_obses[idxs + i]
            discounts *= dones + (1 - dones) * discount
            random_dones = (i >= ns).astype(np.float32)
            if self.random_nstep:
                dones = np.maximum(dones, random_dones)
            dones = np.maximum(dones, self.dones[idxs + i])

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        discounts = torch.as_tensor(discounts, device=self.device)

        return obses, actions, rewards, next_obses, discounts

    def her_sample(self, batch_size, discount, n):
        assert n == 1
        assert n <= self.idx or self.full

        last_idx = (self.capacity if self.full else self.idx)
        idxs = np.random.randint(0, last_idx, size=batch_size)
        # find corresponding episode endings
        ends = (idxs + self.episode_length -
                1) // self.episode_length * self.episode_length
        lens = ends - idxs
        future_offsets = (np.random.uniform(size=batch_size) * lens).astype(
            np.int)
        her_idxs = np.where(
            np.random.uniform(size=batch_size) < self.future_p)[0]
        future_idxs = (idxs + future_offsets)[her_idxs]

        def replace(source, target, key):
            left, right = self.obs_specs[key]
            source[her_idxs, left:right] = target[:, left:right]

        obses = self.obses[idxs].copy()
        future_obses = self.obses[future_idxs].copy()
        replace(obses, future_obses, 'achieved_goal_position')
        replace(obses, future_obses, 'achieved_goal_orientation')

        next_obses = self.next_obses[idxs].copy()
        future_next_obses = self.next_obses[future_idxs].copy()
        replace(next_obses, future_next_obses, 'achieved_goal_position')
        replace(next_obses, future_next_obses, 'achieved_goal_orientation')

        def get_slice(x, key):
            left, right = self.obs_specs[key]
            return x[:, left:right]

        object_pos = get_slice(next_obses, 'achieved_goal_position')
        target_pos = get_slice(next_obses, 'desired_goal_position')
        object_to_target = np.linalg.norm(object_pos - target_pos, axis=1)
        cube_radius = move_cube._cube_3d_radius
        arena_radius = move_cube._ARENA_RADIUS

        rewards = np.zeros((idxs.shape[0], 1), dtype=np.float32)
        for i in range(batch_size):
            rewards[i] = dm_rewards.tolerance(object_to_target[i],
                                              bounds=(0, 0.2 * cube_radius),
                                              margin=arena_radius,
                                              sigmoid='long_tail')

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(rewards, device=self.device)
        discounts = np.ones(rewards.shape, dtype=np.float32) * discount
        discounts = torch.as_tensor(discounts, device=self.device)

        return obses, actions, rewards, next_obses, discounts
