import numpy as np

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from rrc.replay_buffer import ReplayBuffer


class MultiReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device,
                 random_nstep, n_tasks):

        self.n_tasks = n_tasks

        self.replay_buffer_list = []
        for i in range(n_tasks):
            self.replay_buffer_list.append(
                        ReplayBuffer(obs_shape, action_shape,
                                        capacity, device, random_nstep)
            )

    def __len__(self):
        buff = self.replay_buffer_list[0]
        return buff.capacity if buff.full else buff.idx

    def add(self, obs, action, reward, next_obs, done, log_prob=None):
        for i in range(self.n_tasks):
            self.replay_buffer_list[i].add(obs[i], action, reward[i],
                                                next_obs[i], done, log_prob)

    def sample(self, batch_size, discount, n, task_id, log_prob=False):
        return self.replay_buffer_list[task_id].sample(batch_size, discount, n, log_prob)

    def sample_n(self, batch_size, discount, n, task_id):
        return self.replay_buffer_list[task_id].sample_n(batch_size, discount, n)

    def sample_full_n(self, batch_size, discount, n, task_id, log_prob=True):
        return self.replay_buffer_list[task_id].sample_full_n(batch_size, discount,
                                                    n, log_prob)
