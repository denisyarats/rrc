"""Gym environment for the Real Robot Challenge Phase 1 (Simulation)."""
import enum

import numpy as np
import gym
import pybullet

from numpy import random

from rrc_simulation import TriFingerPlatform
from rrc_simulation import visual_objects
from rrc_simulation.tasks import move_cube

from dm_control.utils import rewards

from envs import ActionType


class ReachEnv(gym.GoalEnv):
    """Gym environment for reaching with simulated TriFingerPro."""
    def __init__(
        self,
        initializer=None,
        action_type=ActionType.TORQUE,
        frameskip=1,
        visualization=True,
        episode_length=move_cube.episode_length,
    ):
        """Initialize.

        Args:
            initializer: Initializer class for providing initial cube pose and
                goal pose.  See :class:`RandomInitializer` and
                :class:`FixedInitializer`.
            action_type (ActionType): Specify which type of actions to use.
                See :class:`ActionType` for details.
            frameskip (int):  Number of actual control steps to be performed in
                one call of step().
            visualization (bool): If true, the pyBullet GUI is run for
                visualization.
        """
        # Basic initialization
        # ====================

        self.action_type = action_type
        self.visualization = visualization
        self.episode_length = episode_length

        # TODO: The name "frameskip" makes sense for an atari environment but
        # not really for our scenario.  The name is also misleading as
        # "frameskip = 1" suggests that one frame is skipped while it actually
        # means "do one step per step" (i.e. no skip).
        if frameskip < 1:
            raise ValueError("frameskip cannot be less than 1.")
        self.frameskip = frameskip

        # will be initialized in reset()
        self.platform = None

        # Create the action and observation spaces
        # ========================================

        spaces = TriFingerPlatform.spaces

        robot_xyz_space = gym.spaces.Box(low=np.array([[-0.3, -0.3, 0]] * 3,
                                                      dtype=np.float32),
                                         high=np.array([[0.3, 0.3, 0.3]] * 3,
                                                       dtype=np.float32))

        if self.action_type == ActionType.TORQUE:
            self.action_space = spaces.robot_torque.gym
        elif self.action_type == ActionType.POSITION:
            self.action_space = spaces.robot_position.gym
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict({
                "torque":
                spaces.robot_torque.gym,
                "position":
                spaces.robot_position.gym,
            })
        else:
            raise ValueError("Invalid action_type")

        self.observation_space = gym.spaces.Dict({
            "observation":
            gym.spaces.Dict({
                "position": spaces.robot_position.gym,
                "velocity": spaces.robot_velocity.gym,
                "torque": spaces.robot_torque.gym,
            }),
            "desired_goal":
            robot_xyz_space,
            "achieved_goal":
            robot_xyz_space,
        })

    def compute_reward(self, observation, info):
        """Compute the reward for the given achieved and desired goal.
        """
        reward = 0.
        for i in range(3):
            dist = np.linalg.norm(observation['desired_goal'][i] -
                                  observation['achieved_goal'][i])
            reward += rewards.tolerance(dist,
                                        bounds=(0., 0.01),
                                        margin=0.05,
                                        value_at_margin=0.1,
                                        sigmoid='long_tail')
        return reward / 3.

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling
        ``reset()`` to reset this environment's state.

        Args:
            action: An action provided by the agent (depends on the selected
                :class:`ActionType`).

        Returns:
            tuple:

            - observation (dict): agent's observation of the current
              environment.
            - reward (float) : amount of reward returned after previous action.
            - done (bool): whether the episode has ended, in which case further
              step() calls will return undefined results.
            - info (dict): info dictionary containing the difficulty level of
              the goal.
        """
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(action):
            raise ValueError(
                "Given action is not contained in the action space.")

        num_steps = self.frameskip

        # ensure episode length is not exceeded due to frameskip
        step_count_after = self.step_count + num_steps
        if step_count_after > self.episode_length:
            excess = step_count_after - self.episode_length
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        for _ in range(num_steps):
            self.step_count += 1
            if self.step_count > self.episode_length:
                raise RuntimeError("Exceeded number of steps for one episode.")

            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            t = self.platform.append_desired_action(robot_action)

            # Use observations of step t + 1 to follow what would be expected
            # in a typical gym environment.  Note that on the real robot, this
            # will not be possible
            observation = self._create_observation(t + 1)

            reward += self.compute_reward(observation, self.info)

        is_done = self.step_count == self.episode_length

        return observation, reward, is_done, self.info

    def reset(self):
        # reset simulation
        del self.platform

        # initialize simulation
        initial_robot_position = TriFingerPlatform.spaces.robot_position.default

        self.platform = TriFingerPlatform(
            visualization=self.visualization,
            initial_robot_position=initial_robot_position,
            initial_object_pose=move_cube.Pose(np.array([0, 0, -10])))

        self.goal = self._generate_pose_goal()

        # visualize the goals
        for i in range(3):
            visual_objects.CubeMarker(
                width=0.025,
                position=self.goal[i],
                orientation=[0, 0, 0, 1],
            )

        self.info = {}

        self.step_count = 0

        return self._create_observation(0)

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
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        move_cube.random = self.np_random
        return [seed]

    def _create_observation(self, t):
        robot_observation = self.platform.get_robot_observation(t)
        robot_xyz = self._get_robot_xyz()

        observation = {
            "observation": {
                "position": robot_observation.position,
                "velocity": robot_observation.velocity,
                "torque": robot_observation.torque,
            },
            "desired_goal": self.goal,
            "achieved_goal": robot_xyz,
        }

        return observation

    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type
        if self.action_type == ActionType.TORQUE:
            robot_action = self.platform.Action(torque=gym_action)
        elif self.action_type == ActionType.POSITION:
            robot_action = self.platform.Action(position=gym_action)
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            robot_action = self.platform.Action(
                torque=gym_action["torque"], position=gym_action["position"])
        else:
            raise ValueError("Invalid action_type")

        return robot_action

    def _random_xyz(self):
        # sample uniform position in circle (https://stackoverflow.com/a/50746409)
        radius = move_cube._max_cube_com_distance_to_center * np.sqrt(
            random.random())
        theta = random.uniform(0, 2 * np.pi)

        # x,y-position of the cube
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        z = random.uniform() * 0.1

        return np.array([x, y, z])

    def _get_robot_xyz(self):
        robot_id = self.platform.simfinger.finger_id
        return np.array([
            pybullet.getLinkState(robot_id, i)[0]
            for i in self.platform.simfinger.pybullet_tip_link_indices
        ])

    def _generate_pose_goal(self):
        """
        Generates a feasible goal pose.
        Only call this AFTER the robot initial position is chosen
        """
        robot_id = self.platform.simfinger.finger_id
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

        ordered_goals = np.zeros((3, 3))
        for i in range(3):
            idx = np.argmin(dists)
            goal_id = idx // 3
            finger_id = idx % 3

            dists[goal_id, :] = np.inf
            dists[:, finger_id] = np.inf

            ordered_goals[finger_id] = goals[goal_id]

        return ordered_goals

    def _set_robot_xyz(self, target_locs):
        """
        Sets the xyz position of the end effectors.
        Assumes that the target locations are valid and do not cause collisions.
        """
        robot_id = self.platform.simfinger.finger_id
        tip_ids = self.platform.simfinger.pybullet_tip_link_indices

        pos = np.zeros((3, 3))
        for i, tip_id in enumerate(tip_ids):
            pos[i] = pybullet.calculateInverseKinematics(
                robot_id, tip_id, target_locs[i],
                maxNumIterations=1000)[3 * i:3 * i + 3]
        joint_angles = pos.flatten()
        self.platform.simfinger.reset_finger_positions_and_velocities(
            joint_angles)
        return joint_angles
