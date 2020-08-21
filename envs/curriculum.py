from rrc.envs.cube_env import CubeEnv
from rrc_simulation.tasks import move_cube

import numpy as np

from dm_control.utils import rewards

from rrc_simulation import TriFingerPlatform
from rrc_simulation import camera, visual_objects
from rrc_simulation.tasks import move_cube

import pybullet

from rrc.envs.reach_env import ReachEnv
from rrc.envs import ActionType

#from matplotlib import pyplot as plt

"""
Curriculum strategy:

1. init buffer of (start, goal) pairs with start=goal pairs
Each iteration:
    1. sample some (start,goal) episodes from buffer, and some new start=goal pairs
    2. add random actions to move the start position
    3. add (start,goal) pairs to buffer if R_min < return < R_max

"""


class Curriculum(ReachEnv):
    """Curriculum over goals and initial states."""

    def __init__(self,
                initializer=None,
                action_type=ActionType.TORQUE,
                frameskip=1,
                visualization=True,
                episode_length=move_cube.episode_length,
                buffer_capacity=1000, R_min=0.2, R_max=0.9,
                new_goal_freq=3, target_task_freq=10,
                n_random_actions=50,
                eval = False):
        super().__init__(initializer, action_type, frameskip, visualization, episode_length)

        self.new_goal_freq = new_goal_freq
        self.target_task_freq = target_task_freq
        self.n_random_actions = n_random_actions

        # start state in joint angles
        self.start_shape = (9,)
        # goal state in xyz of end effector
        self.goal_shape = (3,3)

        self.R_min = R_min * episode_length
        self.R_max = R_max * episode_length

        self.curriculum_buffer = CurriculumBuffer(self.start_shape,
                                                self.goal_shape,
                                                buffer_capacity,
                                                self.R_min,
                                                self.R_max
                                                )

        self.eval = eval
        return

    def reset(self):
        if self.eval:
            return super().reset()

        # store to buffer
        if len(self.curriculum_buffer) > 0:
            self.curriculum_buffer.add(self.start, self.goal, self.ep_reward)
        self.ep_reward = 0

        # reset start and goal
        if len(self.curriculum_buffer) % self.new_goal_freq == 0:
            # generate equal start and goal
            del self.platform
            self.platform = TriFingerPlatform()
            self.goal = self._generate_pose_goal()
            self.start = self._set_robot_xyz(self.goal)

            # this is a bit of a hack
            if len(self.curriculum_buffer) == 0:
                self.curriculum_buffer.add(self.start, self.goal, self.R_min + 1)
        elif len(self.curriculum_buffer) % self.target_task_freq == 0:
            return super().reset()
        else:
            # sample goal from buffer and add a few random actions
            start, goal = self.curriculum_buffer.sample()
            self.goal = goal
            # reset simulation
            del self.platform
            self.platform = TriFingerPlatform(initial_robot_position=start,
                                    initial_object_pose=move_cube.Pose(np.array([0,0,-1])))
            self.step_count = 0
            for i in range(self.n_random_actions):
                super().step(self.action_space.sample())
            t = self.platform.simfinger._t
            self.start = self.platform.get_robot_observation(t+1).position

        # reset simulation
        del self.platform
        self.platform = TriFingerPlatform(
            visualization=self.visualization,
            initial_robot_position=self.start,
            initial_object_pose=move_cube.Pose(np.array([0,0,-1]))
        )

        for i in range(3):
            visual_objects.CubeMarker(
                    width=0.025,
                    position=self.goal[i],
                    orientation=[0,0,0,1],
                )

        self.info = {}
        self.step_count = 0
        return self._create_observation(0)

    def step(self, action):
        if self.eval:
            return super().step(action)

        observation, reward, is_done, info = super().step(action)
        self.ep_reward += reward
        return observation, reward, is_done, info


class CurriculumBuffer(object):
    def __init__(self, start_shape, goal_shape, capacity,
                R_min, R_max):
        self.capacity = capacity

        self.R_min = R_min
        self.R_max = R_max

        self.starts = np.empty((capacity, *start_shape), dtype=np.float32)
        self.goals = np.empty((capacity, *goal_shape), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, start, goal, ep_reward):
        if self.R_min < ep_reward and ep_reward < self.R_max:
            np.copyto(self.starts[self.idx], start)
            np.copyto(self.goals[self.idx], goal)

            self.idx = (self.idx + 1) % self.capacity
            self.full = self.full or self.idx == 0
        return

    def sample(self, batch_size=1):
        idx = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)
        start = self.starts[idx]
        goal = self.goals[idx]

        return np.squeeze(start), np.squeeze(goal)
