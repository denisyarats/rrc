"""Gym environment for the Real Robot Challenge Phase 1 (Simulation)."""
import enum

import numpy as np
import gym
import pybullet
from copy import deepcopy

import trifinger_simulation
import trifinger_simulation.visual_objects
from trifinger_simulation import trifingerpro_limits
from trifinger_simulation.tasks import move_cube
from scipy.spatial.transform import Rotation

from dm_control.utils import rewards as dmr

from envs import ActionType
from envs.visual_objects import OrientationMarker, CubeMarker


class TaskOneEnv(gym.GoalEnv):
    """Gym environment for moving cubes with simulated TriFingerPro."""
    def __init__(
        self,
        initializer,
        action_type=ActionType.TORQUE,
        frameskip=1,
        episode_length=move_cube.episode_length,
        num_corners=0,
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

        assert initializer.difficulty == 1
        self.initializer = initializer
        self.action_type = action_type
        self.episode_length = episode_length * frameskip
        assert self.episode_length <= move_cube.episode_length
        self.num_corners = num_corners

        self.info = {"difficulty": self.initializer.difficulty}

        # TODO: The name "frameskip" makes sense for an atari environment but
        # not really for our scenario.  The name is also misleading as
        # "frameskip = 1" suggests that one frame is skipped while it actually
        # means "do one step per step" (i.e. no skip).
        if frameskip < 1:
            raise ValueError("frameskip cannot be less than 1.")
        self.frameskip = frameskip

        # will be initialized in reset()
        self.platform = None

        self.time_step_s = 0.004

        # Create the action and observation spaces
        # ========================================

        # Create the action and observation spaces
        # ========================================

        robot_torque_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_torque.low,
            high=trifingerpro_limits.robot_torque.high,
        )
        robot_position_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_position.low,
            high=trifingerpro_limits.robot_position.high,
        )
        robot_velocity_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_velocity.low,
            high=trifingerpro_limits.robot_velocity.high,
        )

        self.object_state_space = gym.spaces.Dict({
            "position":
            gym.spaces.Box(
                low=trifingerpro_limits.object_position.low,
                high=trifingerpro_limits.object_position.high,
            ),
            "orientation":
            gym.spaces.Box(
                low=trifingerpro_limits.object_orientation.low,
                high=trifingerpro_limits.object_orientation.high,
            ),
        })

        if self.action_type == ActionType.TORQUE:
            self.action_space = robot_torque_space
            self._initial_action = trifingerpro_limits.robot_torque.default
        elif self.action_type == ActionType.POSITION:
            self.action_space = robot_position_space
            self._initial_action = trifingerpro_limits.robot_position.default
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict({
                "torque":
                robot_torque_space,
                "position":
                robot_position_space,
            })
            self._initial_action = {
                "torque": trifingerpro_limits.robot_torque.default,
                "position": trifingerpro_limits.robot_position.default,
            }
        else:
            raise ValueError("Invalid action_type")

        self.observation_space = gym.spaces.Dict({
            "observation":
            gym.spaces.Dict({
                "position": robot_position_space,
                "velocity": robot_velocity_space,
                "torque": robot_torque_space,
                #"tip_force": gym.spaces.Box(low=np.zeros(3, dtype=np.float32), high=np.ones(3, dtype=np.float32))
            }),
            "action":
            deepcopy(self.action_space),
            "desired_goal":
            deepcopy(self.object_state_space),
            "achieved_goal":
            deepcopy(self.object_state_space),
        })

        self.reward_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))

    def compute_reward(self, observation, info):
        cube_radius = move_cube._cube_3d_radius
        arena_radius = move_cube._ARENA_RADIUS
        min_height = move_cube._min_height
        max_height = move_cube._max_height

        robot_id = self.platform.simfinger.finger_id
        finger_ids = self.platform.simfinger.pybullet_tip_link_indices

        # compute reward to see if the object reached the target
        object_pos = observation['achieved_goal']['position']
        target_pos = observation['desired_goal']['position']
        object_to_target = np.linalg.norm(object_pos[:2] - target_pos[:2])
        in_place = dmr.tolerance(object_to_target,
                                 bounds=(0, 0.001 * cube_radius),
                                 margin=cube_radius,
                                 sigmoid='long_tail')

        return np.array([in_place])

        velocity = np.linalg.norm(observation['observation']['velocity'])
        control = dmr.tolerance(velocity,
                                margin=1,
                                value_at_margin=0,
                                sigmoid='quadratic')
        small_control = (control + 4) / 5

        grasp = 0
        for finger_id in finger_ids:
            finger_pos = pybullet.getLinkState(robot_id, finger_id)[0]
            finger_to_object = np.linalg.norm(finger_pos - object_pos)
            grasp = max(
                grasp,
                dmr.tolerance(finger_to_object,
                              bounds=(0, 0.5 * cube_radius),
                              margin=arena_radius,
                              sigmoid='long_tail'))

        grasp = (3 * grasp + 3) / 6

        reward = in_place * small_control * grasp
        return np.array([reward])

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
            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            t = self.platform.append_desired_action(robot_action)

            observation = self._create_observation(t, action)

            reward += self.compute_reward(observation, self.info)

            self.step_count = t
            # make sure to not exceed the episode length
            if self.step_count >= self.episode_length:
                break

        is_done = self.step_count == self.episode_length

        self.goal_marker.set_state(observation['desired_goal']['position'],
                                   observation['desired_goal']['orientation'])
        self.goal_orientation_marker.set_state(
            observation['desired_goal']['position'],
            observation['desired_goal']['orientation'])
        self.object_orientation_marker.set_state(
            observation['achieved_goal']['position'],
            observation['achieved_goal']['orientation'])

        return observation, reward, is_done, self.info

    def reset(self, **kwargs):
        # By changing the `_reset_*` method below you can switch between using
        # the platform frontend, which is needed for the submission system, and
        # the direct simulation, which may be more convenient if you want to
        # pre-train locally in simulation.
        self._reset_direct_simulation(**kwargs)

        self.step_count = 0

        # need to already do one step to get initial observation
        # TODO disable frameskip here?
        observation, _, _, _ = self.step(self._initial_action)

        return observation

    def _reset_direct_simulation(self, **kwargs):
        """Reset direct simulation.

        With this the env can be used without backend.
        """

        # reset simulation
        del self.platform

        self.initializer.reset()

        # initialize simulation
        initial_object_pose = self.initializer.get_initial_state()
        goal_object_pose = self.initializer.get_goal()

        self.goal = {
            "position": goal_object_pose.position,
            "orientation": goal_object_pose.orientation,
        }

        # verify that the given goal pose is contained in the cube state space
        if not self.object_state_space.contains(self.goal):
            raise ValueError("Invalid goal pose.")

        time_step_s = kwargs.get('time_step_s', self.time_step_s)
        self.platform = trifinger_simulation.TriFingerPlatform(
            visualization=False,
            initial_object_pose=initial_object_pose,
            time_step_s=time_step_s)

        self.goal_marker = CubeMarker(
            width=0.065,
            position=goal_object_pose.position,
            orientation=goal_object_pose.orientation,
            physicsClientId=self.platform.simfinger._pybullet_client_id,
        )

        self.object_orientation_marker = OrientationMarker(
            length=0.5 * move_cube._CUBE_WIDTH,
            radius=0.01,
            position=initial_object_pose.position,
            orientation=initial_object_pose.orientation,
            physicsClientId=self.platform.simfinger._pybullet_client_id,
        )

        self.goal_orientation_marker = OrientationMarker(
            length=0.5 * move_cube._CUBE_WIDTH,
            radius=0.01,
            position=goal_object_pose.position,
            orientation=goal_object_pose.orientation,
            physicsClientId=self.platform.simfinger._pybullet_client_id,
        )

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

    def _create_observation(self, t, action):
        robot_observation = self.platform.get_robot_observation(t)
        camera_observation = self.platform.get_camera_observation(t)

        observation = {
            "observation": {
                "position": robot_observation.position,
                "velocity": robot_observation.velocity,
                "torque": robot_observation.torque,
                #"tip_force": robot_observation.tip_force,
            },
            "action": action,
            "desired_goal": self.goal,
            "achieved_goal": {
                "position": camera_observation.object_pose.position,
                "orientation": camera_observation.object_pose.orientation,
            },
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
