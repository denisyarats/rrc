import numpy as np
import gym

from rrc_simulation.gym_wrapper.envs import cube_env
from rrc_simulation.tasks import move_cube

from reach_env import ReachEnv


def flat_space(space, value=None):
    if type(space) == gym.spaces.Box:
        yield (space, value)
    else:
        assert type(space) == gym.spaces.Dict
        for key in space.spaces:
            for x in flat_space(space[key], None if value is None else value[key]):
                yield x


class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        low = [
            space.low.flatten()
            for space, _ in flat_space(self.env.observation_space)
        ]

        high = [
            space.high.flatten()
            for space, _ in flat_space(self.env.observation_space)
        ]

        self.observation_space = gym.spaces.Box(low=np.concatenate(low, axis=0),
                                                high=np.concatenate(high, axis=0))

    def observation(self, obs):
        observation = [x.flatten() for _, x in flat_space(self.env.observation_space, obs)]

        observation = np.concatenate(observation, axis=0)
        return observation


class ActionScalingWrapper(gym.ActionWrapper):
    def __init__(self, env, low, high):
        super().__init__(env)
        self.low = low
        self.high = high

        self.action_space = gym.spaces.Box(low=low,
                                           high=high,
                                           shape=self.env.action_space.shape,
                                           dtype=self.env.action_space.dtype)

    def action(self, action):
        scale = self.high - self.low
        action = (action - self.low) / scale
        true_scale = self.env.action_space.high - self.env.action_space.low
        action = action * true_scale + self.env.action_space.low
        return action.astype(self.env.action_space.dtype)

"""
def FlattenActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        pass
"""

def make_initializer(difficulty, fixed):
    if fixed:
        init = move_cube.sample_goal(-1)
        goal = move_cube.sample_goal(difficulty)
        initializer = cube_env.FixedInitializer(difficulty, init, goal)
    else:
        initializer = cube_env.RandomInitializer(difficulty=difficulty)
    return initializer


def make(env_name, action_type, action_repeat, initializer, seed):
    assert action_type in ['position', 'torque', 'both']

    if action_type == 'position':
        action_type = cube_env.ActionType.POSITION
    elif action_type == 'torque':
        action_type == cube_env.ActionType.TORQUE
    else:
        action_type == cube_env.ActionType.TORQUE_AND_POSITION

    # env = gym.make(
    #     f'rrc_simulation.gym_wrapper:{env_name}-v1',
    #     initializer=initializer,
    #     action_type=action_type,
    #     frameskip=action_repeat,
    #     visualization=False,
    # )
    env = ReachEnv(
        initializer=initializer,
        action_type=action_type,
        frameskip=action_repeat,
        visualization=False,
    )
    env.seed(seed)

    env = FlattenObservationWrapper(env)
    env = ActionScalingWrapper(env, low=-1.0, high=+1.0)

    action_space = env.action_space
    assert np.all(action_space.low >= -1.0)
    assert np.all(action_space.high <= +1.0)

    return env
