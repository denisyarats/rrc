"""Gym environment for the Real Robot Challenge Phase 1 (Simulation)."""
import enum

import numpy as np
import gym
import pybullet

from rrc_simulation import TriFingerPlatform
from rrc_simulation import visual_objects
from rrc_simulation.tasks import move_cube

from dm_control.utils import rewards as dmr
from scipy.spatial.transform import Rotation

from old_envs import ActionType


class TaskOneEnv(gym.GoalEnv):
    """Gym environment for moving cubes with simulated TriFingerPro."""
    def __init__(
        self,
        initializer,
        action_type=ActionType.TORQUE,
        frameskip=1,
        visualization=False,
        episode_length=move_cube.episode_length,
        num_corners=8,
        enable_visual_objects=False):
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
        self.visualization = visualization
        self.episode_length = episode_length
        self.num_corners = num_corners

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

        object_state_space = gym.spaces.Dict({
            "position":
            spaces.object_position.gym,
            "orientation":
            spaces.object_orientation.gym,
        })

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
            object_state_space,
            "achieved_goal":
            object_state_space,
        })

        self.reward_space = gym.spaces.Box(low=0.0,
                                           high=1.0,
                                           shape=(2 + num_corners,))
        

    def compute_reward(self, observation, info):
        """Compute the reward for the given achieved and desired goal.

        Args:
            achieved_goal (dict): Current pose of the object.
            desired_goal (dict): Goal pose of the object.
            info (dict): An info dictionary containing a field "difficulty"
                which specifies the difficulty level.

        Returns:
            float: The reward that corresponds to the provided achieved goal
            w.r.t. to the desired goal. Note that the following should always
            hold true::

                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(
                    ob['achieved_goal'],
                    ob['desired_goal'],
                    info,
                )
        """
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

        above_ground = dmr.tolerance(
            object_pos[2],
            bounds=(target_pos[2] - 0.001 * min_height,
                    target_pos[2] + 0.001 * min_height),
            margin=0.999 * min_height,
            sigmoid='long_tail')

        # compute reward to see that each fingert is close to the cube
        #grasp = 0
        #hand_away = 0
        #for finger_id in finger_ids:
        #    finger_pos = pybullet.getLinkState(robot_id, finger_id)[0]
        #    finger_to_object = np.linalg.norm(finger_pos - object_pos)
        #    grasp += dmr.tolerance(finger_to_object,
        #                           bounds=(0, 0.5 * cube_radius),
        #                           margin=arena_radius,
        #                           sigmoid='long_tail')

        #grasp /= len(finger_ids)
        #hand_away /= len(finger_ids)

        actual_pose = move_cube.Pose.from_dict(observation['achieved_goal'])
        goal_pose = move_cube.Pose.from_dict(observation['desired_goal'])

        actual_corners = move_cube.get_cube_corner_positions(actual_pose)
        goal_corners = move_cube.get_cube_corner_positions(goal_pose)

        orientation_errors = np.linalg.norm(goal_corners - actual_corners,
                                            axis=1)[:self.num_corners]

        rewards = [above_ground, in_place]
        for e in orientation_errors:
            r = dmr.tolerance(e,
                              bounds=(0, 0.001 * cube_radius),
                              margin=arena_radius,
                              sigmoid='long_tail')
            rewards.append(r)

        return np.mean(rewards)
        return np.array(rewards)

    def compute_reward3(self, observation, info):
        actual_pose = move_cube.Pose.from_dict(observation['achieved_goal'])
        goal_pose = move_cube.Pose.from_dict(observation['desired_goal'])

        range_xy_dist = move_cube._ARENA_RADIUS * 2
        range_z_dist = move_cube._max_height

        xy_dist = np.linalg.norm(goal_pose.position[:2] -
                                 actual_pose.position[:2])
        z_dist = abs(goal_pose.position[2] - actual_pose.position[2])

        # weight xy- and z-parts by their expected range
        position_error = (xy_dist / range_xy_dist + z_dist / range_z_dist) / 2

        position_reward = dmr.tolerance(position_error,
                                        bounds=(0, 0.001),
                                        margin=0.999,
                                        sigmoid='long_tail')

        goal_rot = Rotation.from_quat(goal_pose.orientation)
        actual_rot = Rotation.from_quat(actual_pose.orientation)
        error_rot = goal_rot.inv() * actual_rot
        orientation_error = error_rot.magnitude() / np.pi

        orientation_reward = dmr.tolerance(orientation_error,
                                           bounds=(0, 0.001),
                                           margin=0.999,
                                           sigmoid='long_tail')

        return np.array([position_reward, orientation_reward])

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

    def reset(self, **kwargs):
        # reset simulation
        del self.platform

        # initialize simulation
        #initial_robot_position = np.random.uniform(
        #    TriFingerPlatform.spaces.robot_position.low,
        #    TriFingerPlatform.spaces.robot_position.high)
        initial_robot_position = (
            TriFingerPlatform.spaces.robot_position.default)
        self.initializer.reset()
        initial_object_pose = self.initializer.get_initial_state()
        goal_object_pose = self.initializer.get_goal()

        self.platform = TriFingerPlatform(
            visualization=self.visualization,
            initial_robot_position=initial_robot_position,
            initial_object_pose=initial_object_pose,
        )

        self.goal = {
            "position": goal_object_pose.position,
            "orientation": goal_object_pose.orientation,
        }

        # visualize the goal
        self.goal_marker = visual_objects.CubeMarker(
            width=0.065,
            position=goal_object_pose.position,
            orientation=goal_object_pose.orientation,
            physicsClientId=self.platform.simfinger._pybullet_client_id,
        )

        self.info = {"difficulty": self.initializer.difficulty}

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
        object_observation = self.platform.get_object_pose(t)

        observation = {
            "observation": {
                "position": robot_observation.position,
                "velocity": robot_observation.velocity,
                "torque": robot_observation.torque,
            },
            "desired_goal": self.goal,
            "achieved_goal": {
                "position": object_observation.position,
                "orientation": object_observation.orientation,
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
