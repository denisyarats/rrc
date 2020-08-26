import numpy as np
import gym

from envs import initializers, wrappers, curriculum
from rrc_simulation.gym_wrapper.envs import cube_env

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
    else:
        assert False, f'wrong initializer: {name}'


def make(env_name,
         action_type,
         action_repeat,
         episode_length,
         initializer,
         seed,
         use_curriculum=False,
         start_shape=None,
         goal_shape=None,
         buffer_capacity=None,
         R_min=None,
         R_max=None,
         new_goal_freq=None,
         target_task_freq=None,
         n_random_actions=None):
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

    if use_curriculum:
        if env_name == 'reach':
            env = curriculum.ReachCurriculum(env,
                                             start_shape=start_shape,
                                             goal_shape=goal_shape,
                                             buffer_capacity=buffer_capacity,
                                             R_min=R_min,
                                             R_max=R_max,
                                             new_goal_freq=new_goal_freq,
                                             target_task_freq=target_task_freq,
                                             n_random_actions=n_random_actions)
        else:
            env = curriculum.CubeCurriculum(env,
                                            start_shape=start_shape,
                                            goal_shape=goal_shape,
                                            buffer_capacity=buffer_capacity,
                                            R_min=R_min,
                                            R_max=R_max,
                                            new_goal_freq=new_goal_freq,
                                            target_task_freq=target_task_freq,
                                            n_random_actions=n_random_actions,
                                            difficulty=initializer.difficulty)

    env.seed(seed)

    #excluded = []
    # for task 1, 2, and 3 we don't need orientation
    #if initializer.difficulty in [1, 2, 3]:
    #    excluded += ['achieved_goal_orientation']
    #    excluded += ['desired_goal_orientation']
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
    id="cube-v1",
    entry_point="envs.cube_env:CubeEnv",
)

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