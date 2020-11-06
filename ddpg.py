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
                 stddev, parameterization, use_ln):
        super().__init__()

        assert parameterization in ['clipped', 'squashed']
        self.stddev = stddev
        self.dist_type = utils.SquashedNormal if parameterization == 'squashed' else utils.ClippedNormal

        self.trunk = utils.mlp(obs_shape[0],
                               hidden_dim,
                               action_shape[0],
                               hidden_depth,
                               use_ln=use_ln)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs):
        mu = self.trunk(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * self.stddev

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = self.dist_type(mu, std)
        return dist

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_actor/{k}_hist', v, step)

        for i, m in enumerate(self.trunk):
            if type(m) == nn.Linear:
                logger.log_param(f'train_actor/fc{i}', m, step)


class Critic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_shape, action_shape, hidden_dim, hidden_depth,
                 use_ln):
        super().__init__()

        self.Q1 = utils.mlp(obs_shape[0] + action_shape[0],
                            hidden_dim,
                            1,
                            hidden_depth,
                            use_ln=use_ln)
        self.Q2 = utils.mlp(obs_shape[0] + action_shape[0],
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


class RewardMixer(nn.Module):
    """Reward integrator"""
    def __init__(self, reward_shape, init_value, final_value, start, period):
        super().__init__()
        self.reward_shape = reward_shape
        self.init_values = np.ones(reward_shape) * init_value
        self.final_values = np.ones(reward_shape) * final_value
        self.starts = np.arange(reward_shape[0]) * start
        self.periods = np.ones(reward_shape) * period
        self.alphas = self.init_values.copy()
        #import ipdb; ipdb.set_trace()
        self.outputs = dict()

    def update(self, step):
        #import ipdb; ipdb.set_trace()
        ps = ((step - self.starts) / self.periods).clip(0.0, 1.0)
        dfs = self.final_values - self.init_values
        self.alphas = self.init_values + dfs * ps

    def forward(self, rewards):
        #import ipdb; ipdb.set_trace()
        #import ipdb; ipdb.set_trace()
        w = np.random.dirichlet(self.alphas, size=rewards.shape[0])
        for i in range(w.shape[1]):
            self.outputs[f'alpha_{i}'] = self.alphas[i]
            self.outputs[f'weight_{i}'] = w[:, i].mean()
        w = torch.tensor(w, device=rewards.device).float()
        reward = (rewards * w).sum(axis=1, keepdim=True)
        return reward

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log(f'train_mixer/{k}', v, step)


class FixedRewardMixer(nn.Module):
    """Reward integrator"""
    def __init__(self, reward_shape, alpha_0, alpha_1, alpha_2):
        super().__init__()
        self.reward_shape = reward_shape
        self.alphas = np.array([alpha_0, alpha_1, alpha_1, alpha_2])
        #import ipdb; ipdb.set_trace()
        self.outputs = dict()

    def forward(self, rewards):
        #import ipdb; ipdb.set_trace()
        w = np.random.dirichlet(self.alphas, size=rewards.shape[0])
        for i in range(w.shape[1]):
            self.outputs[f'alpha_{i}'] = self.alphas[i]
            self.outputs[f'weight_{i}'] = w[:, i].mean()
        w = torch.tensor(w, device=rewards.device).float()
        reward = (rewards * w).sum(axis=1, keepdim=True)
        return reward

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log(f'train_mixer/{k}', v, step)


class ConstantRewardMixer(nn.Module):
    """Reward integrator"""
    def __init__(self, reward_shape, w0, w1, w2, w3):
        super().__init__()
        self.reward_shape = reward_shape
        self.weights = np.array([w0, w1, w2, w3])[:reward_shape[0]]
        self.weights /= self.weights.sum()
        #import ipdb; ipdb.set_trace()
        self.outputs = dict()

    def update(self, step):
        pass

    def forward(self, rewards):
        #import ipdb; ipdb.set_trace()
        #import ipdb; ipdb.set_trace()
        w = torch.tensor(self.weights, device=rewards.device).float()
        #import ipdb; ipdb.set_trace()
        w = w.unsqueeze(0)
        for i in range(w.shape[1]):
            self.outputs[f'weight_{i}'] = w[:, i].mean()
        reward = (rewards * w).sum(axis=1, keepdim=True)
        return reward

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log(f'train_mixer/{k}', v, step)


class DDPGAgent(object):
    """Data regularized Q: actor-critic method for learning from pixels."""
    def __init__(self, obs_shape, obs_slices, action_shape, action_range,
                 device, critic_cfg, actor_cfg, discount, lr,
                 actor_update_frequency, critic_tau,
                 critic_target_update_frequency, batch_size, nstep, use_ln,
                 excluded_obses):
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.nstep = nstep

        excluded_obses = [] if excluded_obses is None else excluded_obses.split(
            ':')
        valid_obs_idxs = []
        for key, left, right in obs_slices:
            if key not in excluded_obses:
                for i in range(left, right):
                    valid_obs_idxs.append(i)
        self.valid_obs_idxs = np.array(sorted(valid_obs_idxs))

        actor_cfg.params.obs_shape = self.valid_obs_idxs.shape
        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        critic_cfg.params.obs_shape = self.valid_obs_idxs.shape
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

    def preprocess_obs(self, obs):
        return obs[:, self.valid_obs_idxs]

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        obs = self.preprocess_obs(obs)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, discount, logger,
                      step):
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        logger.log('train_critic/target_q1', target_Q1.mean(), step)
        logger.log('train_critic/target_q2', target_Q2.mean(), step)
        logger.log('train_critic/q', target_Q.mean(), step)
        logger.log('train_critic/v', target_V.mean(), step)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        logger.log('train_critic/q1', Q1.mean(), step)
        logger.log('train_critic/q2', Q2.mean(), step)
        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor(self, obs, logger, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, discount = \
          replay_buffer.sample(self.batch_size, self.discount, self.nstep)

        obs = self.preprocess_obs(obs)
        next_obs = self.preprocess_obs(next_obs)

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, discount, logger,
                           step)

        if step % self.actor_update_frequency == 0:
            self.update_actor(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

    def save(self, model_dir, step):
        torch.save(self.actor.state_dict(),
                   '%s/actor_%s.pt' % (model_dir, step))
        torch.save(self.critic.state_dict(),
                   '%s/critic_%s.pt' % (model_dir, step))

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step)))
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step)))
