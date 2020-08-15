"""
Outline:
- need n tasks (reward functions) on same env
    - need to write up more tasks
- fill replay buffer with r_1,..., r_n
    - need to write a new replay buffer
- learn n policies with shared parameters
    - need to write a new network architecture
- during rollouts: act with 2 policies according to "scheduler"
    - act method needs to know the timestep
- during learning: update all policies from buffer off-policy
    - simple for loop to make more updates

Questions:
- do we need off-policy Q-learning alg like retrace?
- how do we make scheduler?
- how do we choose tasks?
"""

"""
TODO:
- multitask logging
"""


"""
This is a modification of DDPG to use multiple tasks along with a scheduler
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
import hydra
import kornia


class Actor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, obs_shape, action_shape, hidden_dim, hidden_depth,
                 stddev, parameterization, use_ln, head_init_coef,
                 n_tasks):
        super().__init__()
        self.n_tasks = n_tasks

        assert parameterization in ['clipped', 'squashed']
        self.stddev = stddev
        self.dist_type = utils.SquashedNormal if parameterization == 'squashed' else utils.ClippedNormal

        self.trunk = utils.sacx_mlp(n_tasks,
                               obs_shape[0],
                               hidden_dim,
                               action_shape[0],
                               hidden_depth,
                               use_ln=use_ln)

        self.outputs = dict()
        self.apply(utils.weight_init)

        self.trunk[-1].weight.data *= head_init_coef

    def forward(self, obs):
        mu = self.trunk(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * self.stddev

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        mu = torch.chunk(mu, self.n_tasks, dim=-1)
        std = torch.chunk(std, self.n_tasks, dim=-1)

        dists = []
        for i in range(self.n_tasks):
            dists.append(self.dist_type(mu[i], std[i]))
        return dists

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)




class Critic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_shape, action_shape, hidden_dim, hidden_depth,
                 use_ln,
                 n_tasks):
        super().__init__()
        self.n_tasks = n_tasks

        self.Q1 = utils.sacx_mlp(n_tasks,
                            obs_shape[0] + action_shape[0],
                            hidden_dim,
                            1,
                            hidden_depth,
                            use_ln=use_ln)
        self.Q2 = utils.sacx_mlp(n_tasks,
                            obs_shape[0] + action_shape[0],
                            hidden_dim,
                            1,
                            hidden_depth,
                            use_ln=use_ln)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        q1 = torch.chunk(q1, self.n_tasks, dim=-1)
        q2 = torch.chunk(q2, self.n_tasks, dim=-1)

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)


class Scheduler(object):
    """Uniform random scheduler for now."""
    def __init__(self, n_tasks):
        self.n_tasks = n_tasks

    def choose_task(self):
        return np.random.choice(self.n_tasks)


class SACXAgent(object):
    """Data regularized Q: actor-critic method for learning from pixels."""
    def __init__(self, obs_shape, action_shape, action_range, device,
                 critic_cfg, actor_cfg, discount, lr,
                 actor_update_frequency, critic_tau,
                 critic_target_update_frequency, batch_size, nstep,
                 use_ln, head_init_coef,
                 n_tasks):
        self.n_tasks = n_tasks
        self.scheduler = Scheduler(n_tasks)

        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.nstep = nstep

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, task_id, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)[task_id]
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)[task_id]
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, discount, logger,
                      step, task_id):
        with torch.no_grad():
            dist = self.actor(next_obs)[task_id]
            next_action = dist.rsample()
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_Q1 = target_Q1[task_id]
            target_Q2 = target_Q2[task_id]
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        logger.log('train_critic/target_q1', target_Q1.mean(), step)
        logger.log('train_critic/target_q2', target_Q2.mean(), step)
        logger.log('train_critic/q', target_Q.mean(), step)
        logger.log('train_critic/v', target_V.mean(), step)

        Q1, Q2 = self.critic(obs, action)
        Q1 = Q1[task_id]
        Q2 = Q2[task_id]
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        logger.log('train_critic/q1', Q1.mean(), step)
        logger.log('train_critic/q2', Q2.mean(), step)
        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor(self, obs, logger, step, task_id):
        dist = self.actor(obs)[task_id]
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q1 = Q1[task_id]
        Q2 = Q2[task_id]
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)


    def update(self, multi_replay_buffer, loggers, step):
        for task_id in range(self.n_tasks):
            obs, action, reward, next_obs, discount = \
                multi_replay_buffer.sample(self.batch_size, self.discount,
                                                self.nstep, task_id)

            loggers[task_id].log('train/batch_reward', reward.mean(), step)

            self.update_critic(obs, action, reward, next_obs, discount, loggers[task_id],
                                step, task_id)

            if step % self.actor_update_frequency == 0:
                self.update_actor(obs, loggers[task_id], step, task_id)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                         self.critic_tau)




    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
