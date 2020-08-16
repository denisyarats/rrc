"""
An env that contains several tasks operating on the same underlying CubeEnv
"""

import enum

import numpy as np
import gym
import pybullet

from rrc_simulation import TriFingerPlatform
from rrc_simulation import visual_objects
from rrc_simulation.tasks import move_cube

from dm_control.utils import rewards

from rrc.envs import ActionType

from rrc.envs.cube_env import CubeEnv


class MultiTaskEnv(gym.GoalEnv):
    """Multiple tasks on a single robot platform"""
    def __init__(
        self,
        task_list,
        action_type=ActionType.POSITION,
        frameskip=1,
        visualization=False,
        episode_length=move_cube.episode_length,
    ):
        """Initialize.

        Args:
            Task: Task class for reward and init
            action_type (ActionType): Specify which type of actions to use.
                See :class:`ActionType` for details.
            frameskip (int):  Number of actual control steps to be performed in
                one call of step().
            visualization (bool): If true, the pyBullet GUI is run for
                visualization.
        """
        # Basic initialization
        # ====================

        self.task_list = task_list
        self.n_tasks = len(task_list)
        self.task_id = 0

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

    def set_task(self, task_id):
        self.task_id = task_id

    def compute_reward(self, observations):
        reward = []
        for i,t in enumerate(self.task_list):
            reward.append(t.compute_reward(observations[i], self.platform))
        return np.array(reward)

    def reset(self, vis_id = None):
        # reset simulation
        del self.platform

        # initialize simulation
        initial_robot_position = TriFingerPlatform.spaces.robot_position.default
        initial_object_pose = self.task_list[self.task_id].get_initial_state()

        self.platform = TriFingerPlatform(
            visualization=self.visualization,
            initial_robot_position=initial_robot_position,
            initial_object_pose=initial_object_pose,
        )

        self.goals = []
        for t in self.task_list:
            goal_object_pose = t.get_goal()

            goal = {"position": goal_object_pose.position,
                    "orientation": goal_object_pose.orientation,
                    }
            self.goals.append(goal)

            if vis_id is not None:
                if vis_id == t:
                    goal_marker = visual_objects.CubeMarker(
                        width=0.065,
                        position=goal_object_pose.position,
                        orientation=goal_object_pose.orientation,
                    )

        self.step_count = 0

        return self._create_observations(0)

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

        reward = np.zeros(self.n_tasks)
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
            observations = self._create_observations(t + 1)

            reward += self.compute_reward(observations)

        is_done = self.step_count == self.episode_length

        return observations, reward, is_done, None

    def _create_observations(self, t):
        robot_observation = self.platform.get_robot_observation(t)
        object_observation = self.platform.get_object_pose(t)

        observations = []
        for i,t in enumerate(self.task_list):
            obs = {
                "observation": {
                    "position": robot_observation.position,
                    "velocity": robot_observation.velocity,
                    "torque": robot_observation.torque,
                },
                "desired_goal": self.goals[i],
                "achieved_goal": {
                    "position": object_observation.position,
                    "orientation": object_observation.orientation,
                },
            }
            observations.append(obs)

        return observations

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
