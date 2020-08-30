"""Gym environment for the Real Robot Challenge Phase 1 (Simulation)."""
import enum

import numpy as np
import gym

from rrc_simulation import TriFingerPlatform
from rrc_simulation import visual_objects
from rrc_simulation.tasks import move_cube

from rrc_simulation.gym_wrapper.envs import cube_env
ActionType = cube_env.ActionType

class RandomInitializer:
    """Initializer that samples random initial states and goals."""

    def __init__(self, difficulty):
        """Initialize.
        Args:
            difficulty (int):  Difficulty level for sampling goals.
        """
        self.difficulty = difficulty

    def get_initial_state(self):
        """Get a random initial object pose (always on the ground)."""
        return move_cube.sample_goal(difficulty=-1)

    def get_goal(self):
        """Get a random goal depending on the difficulty."""
        return move_cube.sample_goal(difficulty=self.difficulty)


class FixedInitializer:
    """Initializer that uses fixed values for initial pose and goal."""

    def __init__(self, difficulty, initial_state, goal):
        """Initialize.
        Args:
            difficulty (int):  Difficulty level of the goal.  This is still
                needed even for a fixed goal, as it is also used for computing
                the reward (the cost function is different for the different
                levels).
            initial_state (move_cube.Pose):  Initial pose of the object.
            goal (move_cube.Pose):  Goal pose of the object.
        Raises:
            Exception:  If initial_state or goal are not valid.  See
            :meth:`move_cube.validate_goal` for more information.
        """
        move_cube.validate_goal(initial_state)
        move_cube.validate_goal(goal)
        self.difficulty = difficulty
        self.initial_state = initial_state
        self.goal = goal

    def get_initial_state(self):
        """Get the initial state that was set in the constructor."""
        return self.initial_state

    def get_goal(self):
        """Get the goal that was set in the constructor."""
        return self.goal


class CubeEnv(gym.GoalEnv):
    """Gym environment for moving cubes with simulated TriFingerPro."""

    def __init__(
        self,
        initializer,
        action_type=ActionType.POSITION,
        frameskip=1,
        visualization=False,
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

        self.initializer = initializer
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

        object_state_space = gym.spaces.Dict(
            {
                "position": spaces.object_position.gym,
                "orientation": spaces.object_orientation.gym,
            }
        )

        xyz_space = gym.spaces.Box(
                    low=np.array([-0.3,-0.3,0]*3, dtype=np.float32),
                    high=np.array([0.3]*9, dtype=np.float32))

        if self.action_type == ActionType.TORQUE:
            self.action_space = spaces.robot_torque.gym
        elif self.action_type == ActionType.POSITION:
            #self.action_space = spaces.robot_position.gym
            self.action_space = xyz_space
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict(
                {
                    "torque": spaces.robot_torque.gym,
                    "position": spaces.robot_position.gym,
                }
            )
        else:
            raise ValueError("Invalid action_type")

        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Dict(
                    {
                        "position": spaces.robot_position.gym,
                        "velocity": spaces.robot_velocity.gym,
                        "torque": spaces.robot_torque.gym,
                    }
                ),
                "desired_goal": object_state_space,
                "achieved_goal": object_state_space,
            }
        )

    def compute_reward(self, achieved_goal, desired_goal, info):
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
        return -move_cube.evaluate_state(
            move_cube.Pose.from_dict(desired_goal),
            move_cube.Pose.from_dict(achieved_goal),
            info["difficulty"],
        )

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
                "Given action is not contained in the action space."
            )

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

            reward += self.compute_reward(
                observation["achieved_goal"],
                observation["desired_goal"],
                self.info,
            )

        is_done = self.step_count == self.episode_length

        return observation, reward, is_done, self.info

    def reset(self):
        # reset simulation
        del self.platform

        # initialize simulation
        initial_robot_position = TriFingerPlatform.spaces.robot_position.default
        initial_object_pose=self.initializer.get_initial_state()
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
        if self.visualization:
            self.goal_marker = visual_objects.CubeMarker(
                width=0.065,
                position=goal_object_pose.position,
                orientation=goal_object_pose.orientation,
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
            # solve ik
            robot_id = self.platform.simfinger.finger_id
            tip_ids = self.platform.simfinger.pybullet_tip_link_indices
            pos = np.zeros(9)
            for i, tip_id in enumerate(tip_ids):
                pos[3*i:3*i+3] = pybullet.calculateInverseKinematics(
                                        robot_id, tip_id, gym_action[3*i:3*i+3],
                                        maxNumIterations=1000)[3*i:3*i+3]
            gym_action = pos

            robot_action = self.platform.Action(position=gym_action)
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            robot_action = self.platform.Action(
                torque=gym_action["torque"], position=gym_action["position"]
            )
        else:
            raise ValueError("Invalid action_type")

        return robot_action
