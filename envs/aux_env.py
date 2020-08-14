"""
An env that contains several tasks operating on the same underlying CubeEnv
"""

import enum

import numpy as np
import gym
import pybullet

from rrc_simulation import TriFingerPlatform
from rrc_simulation import visual_objects
from rrc_simulation.tasks import move_cube

from dm_control.utils import rewards

from envs import ActionType
from initializers import RandomInitializer, FixedInitializer
from tasks import Reach


"""TODO: fix task abstraction to separate from env"""


class AuxEnv(gym.GoalEnv):
    """Gym environment for a list of auxiliary tasks."""
    def __init__(
        self,
        tasks,
    ):
        self.env=tasks[0]
        self.tasks = tasks
        self.n_tasks = len(tasks)


    def step(self, action):
        """Take one step in env. Return obs and reward from all tasks"""
        observation, reward, is_done, info = env.step(action)
        outputs = []
        for t in self.tasks:
            new_obs = self._create_observation(observation, t)
            new_reward = t.compute_reward(new_obs, info)
            outputs.append((new_obs, new_reward, is_done, info))

        return outputs

    def reset(self):
        obs = self.env.reset()
        observations = []
        for t in self.tasks:
            # need to reset tast in case the goal is randomized
            _ = t.reset()
            observations.append(self._create_observation(obs, t))

        return observations

    def _create_observation(observation, task):
        observation['desired_goal'] = task.goal
        return observation


    def seed(self, seed=None):
        """Sets the seed for this env’s random number generator.

        .. note::

           Spaces need to be seeded separately.  E.g. if you want to sample
           actions directly from the action space using
           ``env.action_space.sample()`` you can set a seed there using
           ``env.action_space.seed()``.

        Returns:
            List of seeds used by this environment.  This environment only uses
            a single seed, so the list contains only one element.
        """
        for t in self.tasks:
            t.seed(seed)
        return [seed]
