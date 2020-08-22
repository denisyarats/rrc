import numpy as np
import ast

import gym
from dm_control.utils import rewards

from rrc_simulation import TriFingerPlatform
from rrc_simulation import camera, visual_objects
from rrc_simulation.tasks import move_cube

import pybullet

from rrc_simulation.gym_wrapper.envs.cube_env import ActionType

"""
Reverse curriculum strategy:

1. init buffer of (start, goal) pairs with start=goal pairs
Each iteration:
    1. sample some (start,goal) episodes from buffer, some new start=goal pairs,
        and some (start,goal) pairs from the target task
    2. if (start,goal) is from buffer add random actions to move the start position
    3. add (start,goal) pairs to buffer if R_min < return < R_max

based off of Florensa et al. https://arxiv.org/abs/1707.05300
"""


class Curriculum(gym.GoalEnv):
    """Curriculum over goals and initial states."""
    def __init__(self,
                env, start_shape, goal_shape,
                buffer_capacity=1000, R_min=0.2, R_max=0.9,
                new_goal_freq=3, target_task_freq=10,
                n_random_actions=50):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

        self.new_goal_freq = new_goal_freq
        self.target_task_freq = target_task_freq
        self.n_random_actions = n_random_actions
        self.buffer_capacity = buffer_capacity

        self.start_shape = ast.literal_eval(start_shape)
        self.goal_shape = ast.literal_eval(goal_shape)

        self.R_min = R_min * self.env.episode_length
        self.R_max = R_max * self.env.episode_length

        self.curriculum_buffer = CurriculumBuffer(self.start_shape,
                                                self.goal_shape,
                                                self.buffer_capacity,
                                                self.R_min,
                                                self.R_max)
        return

    def sample_new_start_and_goal(self):
        """
        Returns a new start and goal.
        This pair should be *very easy* and will be slowly made harder
        by the automati curriculum.
        Goal should match the goal of self.env
        """
        raise NotImplementedError

    def sample_target_start_and_goal(self):
        raise NotImplementedError

    def reset_simulator(self, start, goal):
        raise NotImplementedError

    def reset(self):
        # store to buffer
        if len(self.curriculum_buffer) > 0:
            self.curriculum_buffer.add(self.start, self.goal, self.ep_reward)
        self.ep_reward = 0

        # reset start and goal
        # somtimes sample new (start, goal) pairs
        if len(self.curriculum_buffer) % self.new_goal_freq == 0:
            # generate new start and goal
            del self.env.platform
            self.env.platform = TriFingerPlatform()
            start, goal = self.sample_new_start_and_goal()
            self.start = start
            self.goal = goal

            # always add the first (start,goal) pair to the buffer
            if len(self.curriculum_buffer) == 0:
                self.curriculum_buffer.add(self.start, self.goal, self.R_min + 1)
        # sometimes sample (start, goal) from the target task
        elif len(self.curriculum_buffer) % self.target_task_freq == 0:
            start, goal = self.sample_target_start_and_goal()
            self.start = start
            self.goal = goal
        # usually sample goal from buffer and add a few random actions
        else:
            start, goal = self.curriculum_buffer.sample()
            self.goal = goal
            self.reset_simulator(start, goal)

            self.env.step_count = 0
            for i in range(self.n_random_actions):
                self.env.step(self.action_space.sample())
            t = self.env.platform.simfinger._t
            self.start = self.env.platform.get_robot_observation(t+1).position

        # reset simulation
        self.reset_simulator(self.start, self.goal)

        self.env.info = {}
        self.env.step_count = 0
        return self.env._create_observation(0)

    def step(self, action):
        observation, reward, is_done, info = self.env.step(action)
        self.ep_reward += reward
        return observation, reward, is_done, info

    def seed(self, seed=None):
        return self.env.seed(seed)

    def _get_robot_xyz(self):
        robot_id = self.env.platform.simfinger.finger_id
        return np.array([pybullet.getLinkState(robot_id, i)[0] for
                        i in self.env.platform.simfinger.pybullet_tip_link_indices])

    def _set_robot_xyz(self, target_locs):
        """
        Sets the xyz position of the end effectors.
        Assumes that the target locations are valid and do not cause collisions.
        """
        robot_id = self.env.platform.simfinger.finger_id
        tip_ids = self.env.platform.simfinger.pybullet_tip_link_indices

        pos = np.zeros((3,3))
        for i, tip_id in enumerate(tip_ids):
            pos[i] = pybullet.calculateInverseKinematics(
                                robot_id, tip_id, target_locs[i],
                                maxNumIterations=1000)[3*i:3*i+3]
        joint_angles = pos.flatten()
        self.env.platform.simfinger.reset_finger_positions_and_velocities(joint_angles)
        return joint_angles


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



class ReachCurriculum(Curriculum):
    def __init__(self,
                env, start_shape=(9,), goal_shape=(3,3),
                buffer_capacity=1000, R_min=0.2, R_max=0.9,
                new_goal_freq=3, target_task_freq=10,
                n_random_actions=50):
        super().__init__(env, start_shape, goal_shape,
                        buffer_capacity, R_min, R_max,
                        new_goal_freq, target_task_freq,
                        n_random_actions)
        return

    def sample_new_start_and_goal(self):
        goal = self._generate_pose_goal()
        start = self._set_robot_xyz(goal)
        return start, goal

    def sample_target_start_and_goal(self):
        goal = self._generate_pose_goal()
        start = TriFingerPlatform.spaces.robot_position.default
        return start, goal

    def reset_simulator(self, start, goal):
        del self.env.platform
        self.env.platform = TriFingerPlatform(
            visualization=self.env.visualization,
            initial_robot_position=start,
            initial_object_pose=move_cube.Pose(np.array([0,0,-10]))
        )
        self.env.goal = goal

        for i in range(3):
            visual_objects.CubeMarker(
                    width=0.025,
                    position=goal[i],
                    orientation=[0,0,0,1],
                )
        return

    def _random_xyz(self):
        # sample uniform position in circle (https://stackoverflow.com/a/50746409)
        radius = move_cube._max_cube_com_distance_to_center * np.sqrt(np.random.random())
        theta = np.random.uniform(0, 2 * np.pi)

        # x,y-position of the cube
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        z = np.random.uniform() * 0.1

        return np.array([x, y, z])

    def _generate_pose_goal(self):
        """
        Generates a feasible goal pose.
        Only call this AFTER the robot initial position is chosen
        """
        robot_id = self.env.platform.simfinger.finger_id
        init_tip_locs = self._get_robot_xyz()

        goals = []
        dists = []
        for i in range(3):
            g = self._random_xyz()
            goals.append(g)
            d = []
            for j in range(3):
                d.append(np.linalg.norm(g - init_tip_locs[j]))
            dists.append(np.array(d))
        dists = np.array(dists)

        ordered_goals = np.zeros((3,3))
        for i in range(3):
            idx = np.argmin(dists)
            goal_id = idx // 3
            finger_id = idx % 3

            dists[goal_id,:] = np.inf
            dists[:, finger_id] = np.inf

            ordered_goals[finger_id] = goals[goal_id]

        return ordered_goals


class CubeCurriculum(Curriculum):
    def __init__(self,
                env, difficulty,
                start_shape=(9+7,), goal_shape=(3+4,),
                buffer_capacity=1000, R_min=0.2, R_max=0.9,
                new_goal_freq=3, target_task_freq=10,
                n_random_actions=50):
        super().__init__(env, start_shape, goal_shape,
                        buffer_capacity, R_min, R_max,
                        new_goal_freq, target_task_freq,
                        n_random_actions)
        self.difficulty = difficulty
        return

    def sample_new_start_and_goal(self):
        # set finger tips near the corners of the cube, and cube near target
        goal_dict = move_cube.sample_goal(difficulty=self.difficulty)
        goal = np.concatenate([goal_dict.position, goal_dict.orientation])

        cube_init_pos = np.array([goal[0], goal[1],
                                move_cube._CUBE_WIDTH / 2.])
        cube_init_or = np.array([0,0,0,1])
        tip_init = self._choose_tip_init(move_cube.Pose(position=cube_init_pos,
                                        orientation=cube_init_or))
        start = np.concatenate([tip_init, cube_init_pos, cube_init_or])
        return start, goal

    def sample_target_start_and_goal(self):
        # standard task init
        robot_init = TriFingerPlatform.spaces.robot_position.default
        cube_init = move_cube.sample_goal(difficulty=-1)
        start = np.concatenate([robot_init,
                                cube_init.position, cube_init.orientation])

        goal_dict = move_cube.sample_goal(difficulty=self.difficulty)
        goal = np.concatenate([goal_dict.position, goal_dict.orientation])

        return start, goal

    def reset_simulator(self, start, goal):
        del self.env.platform
        self.env.platform = TriFingerPlatform(
            visualization=self.env.visualization,
            initial_robot_position=start[:9],
            initial_object_pose=move_cube.Pose(position=start[9:12],
                                            orientation=start[12:])
        )

        self.env.goal = {
            "position": goal[:3],
            "orientation": goal[3:],
        }

        if self.env.visualization:
            self.goal_marker = visual_objects.CubeMarker(
                width=0.065,
                position=self.env.goal.position,
                orientation=self.env.goal.orientation,
            )
        return

    def _choose_tip_init(self, cube_pose):
        robot_id = self.env.platform.simfinger.finger_id
        init_tip_locs = self._get_robot_xyz()
        cube_corners = move_cube.get_cube_corner_positions(cube_pose)
        epsilon=0.1

        dists = []
        for i in range(8):
            d = []
            for j in range(3):
                d.append(np.linalg.norm(cube_corners[i] - init_tip_locs[j]))
            dists.append(np.array(d))
        dists = np.array(dists)

        ordered_goals = np.zeros((3,3))
        for i in range(3):
            idx = np.argmin(dists.flatten())
            corner_id = idx // 3
            finger_id = idx % 3
            ordered_goals[finger_id] = cube_corners[corner_id]
            dists[corner_id,:] = np.inf
            dists[:, finger_id] = np.inf

        ordered_goals[:,:-1] *= 1 + epsilon
        ordered_goals[:,-1] /= 2

        return ordered_goals
