import numpy as np
import gym

from envs import initializers, wrappers
from trifinger_simulation_v2.gym_wrapper.envs import cube_env

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


def make(env_name, action_type, action_repeat, episode_length, num_corners,
         action_scale, initializer, seed, randomize, obj_pos_noise_std,
         time_step_low, time_step_high, cube_mass_low, cube_mass_high,
         gravity_low, gravity_high, restitution_low, restitution_high,
         max_velocity_low, max_velocity_high, lateral_friction_low,
         lateral_friction_high, camera_rate_fps_low, camera_rate_fps_high):
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
                   enable_visual_objects=False)

    env.seed(seed)

    if randomize:
        env = wrappers.DomainRandomizationWrapper(
            env,
            obj_pos_noise_std=obj_pos_noise_std,
            time_step_low=time_step_low,
            time_step_high=time_step_high,
            cube_mass_low=cube_mass_low,
            cube_mass_high=cube_mass_high,
            gravity_low=gravity_low,
            gravity_high=gravity_high,
            restitution_low=restitution_low,
            restitution_high=restitution_high,
            max_velocity_low=max_velocity_low,
            max_velocity_high=max_velocity_high,
            lateral_friction_low=lateral_friction_low,
            lateral_friction_high=lateral_friction_high)

    env = wrappers.QuaternionToCornersWrapper(env, num_corners)
    env = wrappers.FlattenObservationWrapper(env)
    env = wrappers.ActionScalingWrapper(env,
                                        alpha=action_scale,
                                        low=-1.0,
                                        high=+1.0)

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
