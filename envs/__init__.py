import numpy as np
import gym

from envs import initializers, wrappers
from trifinger_simulation.gym_wrapper.envs import cube_env

ActionType = cube_env.ActionType


def make_initializer(name, difficulty, init_p=None, max_step=None):
    if name == 'fixed':
        return initializers.FixedInitializer(difficulty)
    elif name == 'fixed_goal':
        return initializers.FixedGoalInitializer(difficulty)
    elif name == 'goal_curriculum':
        return initializers.GoalCurriculumInitializer(init_p, max_step,
                                                      difficulty)
    elif name == 'random':
        return initializers.RandomInitializer(difficulty)
    elif name == 'custom_random':
        return initializers.CustomRandomInitializer(difficulty)
    else:
        assert False, f'wrong initializer: {name}'


def make(env_name,
         action_type,
         action_repeat,
         episode_length,
         num_corners,
         control_margin,
         initializer,
         seed,
         randomize=True,
         obj_position_noise_std=0.1):
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
                   episode_length=episode_length,
                   num_corners=num_corners,
                   control_margin=control_margin,
                   enable_visual_objects=False)

    env.seed(seed)

    if randomize:
        env = wrappers.RandomizedTimeStepWrapper(env)
        env = wrappers.RandomizedObjectPositionWrapper(env,
                                                       obj_position_noise_std)

    env = wrappers.QuaternionToCornersWrapper(env, num_corners)
    env = wrappers.FlattenObservationWrapper(env)
    env = wrappers.ActionScalingWrapper(env, low=-1.0, high=+1.0)

    action_space = env.action_space
    assert np.all(action_space.low >= -1.0)
    assert np.all(action_space.high <= +1.0)

    return env


from gym.envs.registration import register

register(
    id="task1-v1",
    entry_point="envs.task_one_env:TaskOneEnv",
)

register(
    id="task2-v1",
    entry_point="envs.task_two_env:TaskTwoEnv",
)

register(
    id="task3-v1",
    entry_point="envs.task_three_env:TaskThreeEnv",
)

register(
    id="task4-v1",
    entry_point="envs.task_four_env:TaskFourEnv",
)
