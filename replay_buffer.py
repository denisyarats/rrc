import numpy as np

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device,
                 random_nstep):
        self.capacity = capacity
        self.device = device
        self.random_nstep = random_nstep

        self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)

        self.log_probs = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, log_prob=None):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.dones[self.idx], float(done))

        if log_prob is not None:
            np.copyto(self.log_probs[self.idx], log_prob)

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

    def sample_full_n(self, batch_size, discount, n, log_prob=True):
        """returns tensors of size (batch_size, n, dim)"""
        assert discount <= 1.0
        assert n >= 1
        assert n <= self.idx or self.full
        last_idx = (self.capacity if self.full else self.idx) - (n - 1)
        idxs = np.random.randint(0, last_idx, size=batch_size)
        assert idxs.max() + n <= len(self)

        obses, actions, rewards, next_obses, discounts, log_probs = [],[],[],[],[],[]
        not_done = 1.0 - self.dones[idxs]
        for i in range(n):
            obses.append(torch.as_tensor(self.obses[idxs + i],
                                device=self.device).float().unsqueeze(1))
            next_obses.append(torch.as_tensor(self.next_obses[idxs + i],
                                device=self.device).float().unsqueeze(1))
            actions.append(torch.as_tensor(self.actions[idxs + i],
                                device=self.device).unsqueeze(1))
            rewards.append(torch.as_tensor(self.rewards[idxs + i],
                                device=self.device).unsqueeze(1))
            not_done *= (1.0 - self.dones[idxs+i])
            discounts.append(torch.as_tensor(discount * not_done,
                                device=self.device).unsqueeze(1))
            if log_prob:
                log_probs.append(torch.as_tensor(self.log_probs[idxs + i],
                                    device=self.device).unsqueeze(1))

        obses = torch.cat(obses, dim=1)
        actions = torch.cat(actions, dim=1)
        rewards = torch.cat(rewards, dim=1)
        next_obses = torch.cat(next_obses, dim=1)
        discounts = torch.cat(discounts, dim=1)

        if not log_prob:
            return obses, actions, rewards, next_obses, discounts
        else:
            log_probs = torch.cat(log_probs, dim=1)
            return obses, actions, rewards, next_obses, discounts, log_probs
