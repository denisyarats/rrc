import enum

import numpy as np
import gym

from rrc_simulation.tasks import move_cube


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
    def __init__(self, difficulty, initial_state=None, goal=None):
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
        x = 0.9 * move_cube._max_cube_com_distance_to_center * np.cos(np.pi)
        y = 0.9 * move_cube._max_cube_com_distance_to_center * np.sin(np.pi)
        z = move_cube._CUBE_WIDTH / 2 + 1e-6
        default_init_state = move_cube.Pose(np.array([x, y, z]))

        x = 0.9 * move_cube._max_cube_com_distance_to_center * np.cos(0.0)
        y = 0.9 * move_cube._max_cube_com_distance_to_center * np.sin(0.0)
        z = move_cube._CUBE_WIDTH / 2 + 1e-6
        default_goal = move_cube.Pose(np.array([x, y, z]))

        initial_state = initial_state or default_init_state
        goal = goal or default_goal

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
