import numpy as np
import gym

from envs import initializers, wrappers
from rrc_simulation.gym_wrapper.envs import cube_env

ActionType = cube_env.ActionType


def make_initializer(name, init_p, max_step, difficulty):
    if name == 'fixed':
        return initializers.FixedInitializer(difficulty)
    elif name == 'fixed_goal':
        return initializers.FixedGoalInitializer(difficulty)
    elif name == 'goal_curriculum':
        return initializers.GoalCurriculumInitializer(init_p, max_step,
                                                      difficulty)
    elif name == 'random':
        return initializers.RandomInitializer(difficulty)
    else:
        assert False, f'wrong initializer: {name}'


def make(env_name, action_type, action_repeat, episode_length, initializer,
         seed):
    assert action_type in ['position', 'torque', 'both']

    if action_type == 'position':
        action_type = ActionType.POSITION
    elif action_type == 'torque':
        action_type = ActionType.TORQUE
    else:
        action_type = ActionType.TORQUE_AND_POSITION

    env = gym.make(f'{env_name}-v1',
                   initializer=initializer,
                   action_type=action_type,
                   frameskip=action_repeat,
                   visualization=False,
                   episode_length=episode_length)

    env.seed(seed)

    env = wrappers.FlattenObservationWrapper(env)
    env = wrappers.ActionScalingWrapper(env, low=-1.0, high=+1.0)

    action_space = env.action_space
    assert np.all(action_space.low >= -1.0)
    assert np.all(action_space.high <= +1.0)

    return env


from gym.envs.registration import register

register(
    id="reach-v1",
    entry_point="envs.reach_env:ReachEnv",
)

register(
    id="task1-v1",
    entry_point="envs.task_one_env:TaskOneEnv",
)
