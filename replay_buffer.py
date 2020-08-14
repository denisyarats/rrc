import numpy as np

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, reward_shape, capacity, device,
                 random_nstep):
        self.capacity = capacity
        self.device = device
        self.random_nstep = random_nstep

        self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, *reward_shape), dtype=np.float32)
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
        assert n == 1
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        discounts = np.ones(rewards.shape, dtype=np.float32) * discount
        discounts = torch.as_tensor(discounts, device=self.device)

        return obses, actions, rewards, next_obses, discounts

    def sample_n(self, batch_size, discount, n):
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
