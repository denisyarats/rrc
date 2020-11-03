import enum

import numpy as np
import gym

from trifinger_simulation_v2.tasks import move_cube
from scipy.spatial.transform import Rotation


class RandomInitializer:
    """Initializer that samples random initial states and goals."""
    def __init__(self, difficulty):
        """Initialize.

        Args:
            difficulty (int):  Difficulty level for sampling goals.
        """
        self.difficulty = difficulty

    def reset(self):
        pass

    def update(self, step):
        pass

    def log(self, logger, step):
        pass

    def get_initial_state(self):
        """Get a random initial object pose (always on the ground)."""
        return move_cube.sample_goal(difficulty=-1)

    def get_goal(self):
        """Get a random goal depending on the difficulty."""
        return move_cube.sample_goal(difficulty=self.difficulty)


class CustomRandomInitializer:
    """Initializer that samples random initial states and goals."""
    def __init__(self, difficulty):
        """Initialize.

        Args:
            difficulty (int):  Difficulty level for sampling goals.
        """
        self.difficulty = difficulty

    def reset(self):
        pass

    def update(self, step):
        pass

    def log(self, logger, step):
        pass

    def get_initial_state(self):
        """Get a random initial object pose (always on the ground)."""
        radius = 0.3 * move_cube._max_cube_com_distance_to_center * np.sqrt(
            move_cube.random.random())
        theta = move_cube.random.uniform(0, 2 * np.pi)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = move_cube._CUBE_WIDTH / 2

        yaw = move_cube.random.uniform(0, 2 * np.pi)
        orientation = Rotation.from_euler("z", yaw).as_quat()

        pose = move_cube.Pose()
        pose.position = np.array((x, y, z))
        pose.orientation = orientation

        return pose

    def get_goal(self):
        if self.difficulty == 1:
            x = 0.0
            y = 0.1
            z = move_cube._min_height
        else:
            radius = move_cube._max_cube_com_distance_to_center * np.sqrt(
                move_cube.random.random())
            theta = move_cube.random.uniform(0, 2 * np.pi)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = move_cube.random.uniform(move_cube._min_height,
                                         move_cube._max_height)

        orientation = np.array([0, 0, 0, 1])
        #yaw = move_cube.random.uniform(0, 2 * np.pi)
        #orientation = Rotation.from_euler("z", yaw).as_quat()

        pose = move_cube.Pose()
        pose.position = np.array((x, y, z))
        pose.orientation = orientation
        return pose


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

    def reset(self):
        pass

    def update(self, step):
        pass

    def log(self, logger, step):
        pass

    def get_initial_state(self):
        """Get the initial state that was set in the constructor."""
        return self.initial_state

    def get_goal(self):
        """Get the goal that was set in the constructor."""
        return self.goal


class CurriculumInitializer:
    """Initializer that samples random initial states and goals."""
    def __init__(self, init_p, max_step, difficulty):
        """Initialize.

        Args:
            difficulty (int):  Difficulty level for sampling goals.
        """
        assert init_p > 0.0 and init_p <= 1.0
        assert max_step > 0
        assert difficulty == 1
        self.difficulty = difficulty
        self.init_p = init_p
        self.max_step = max_step
        self.p = init_p
        self.diameter = 2 * move_cube._max_cube_com_distance_to_center

    def update(self, step):
        self.p = min(1.0, step / self.max_step)

    def reset(self):
        self.goal = move_cube.sample_goal(difficulty=self.difficulty)
        move_cube.validate_goal(self.goal)

        delta = 1.0 - self.init_p
        max_distance = self.diameter * (self.init_p + delta * self.p)

        while True:
            self.initial_state = move_cube.sample_goal(difficulty=-1)
            dist = np.linalg.norm(self.goal.position -
                                  self.initial_state.position)
            if dist < max_distance:
                break

        self.distance = dist
        move_cube.validate_goal(self.initial_state)

    def get_initial_state(self):
        """Get a random initial object pose (always on the ground)."""
        return self.initial_state

    def get_goal(self):
        """Get a random goal depending on the difficulty."""
        return self.goal


class FixedGoalInitializer:
    """Initializer that samples random initial states and goals."""
    def __init__(self, difficulty):
        """Initialize.

        Args:
            difficulty (int):  Difficulty level for sampling goals.
        """
        assert difficulty == 1
        self.difficulty = difficulty

        self.goal = move_cube.Pose(
            np.array([0.0, 0.0, move_cube._CUBE_WIDTH / 2 + 1e-6]))
        move_cube.validate_goal(self.goal)

    def update(self, step):
        pass

    def reset(self):
        self.initial_state = move_cube.sample_goal(difficulty=-1)
        move_cube.validate_goal(self.initial_state)

    def log(self, logger, step):
        pass

    def get_initial_state(self):
        """Get a random initial object pose (always on the ground)."""
        return self.initial_state

    def get_goal(self):
        """Get a random goal depending on the difficulty."""
        return self.goal


class GoalCurriculumInitializer:
    """Initializer that samples random initial states and goals."""
    def __init__(self, init_p, max_step, difficulty):
        """Initialize.

        Args:
            difficulty (int):  Difficulty level for sampling goals.
        """
        assert init_p >= 0.0 and init_p <= 1.0
        assert max_step > 0
        assert difficulty == 1
        self.difficulty = difficulty
        self.init_p = init_p
        self.max_step = max_step
        self.p = init_p
        self.max_radius = move_cube._max_cube_com_distance_to_center * np.sqrt(
            np.random.rand()) - 1e-6

    def update(self, step):
        self.p = min(1.0, step / self.max_step)

    def reset(self):
        theta = np.random.rand() * 2 * np.pi

        delta = 1.0 - self.init_p
        max_radius = self.max_radius * (self.init_p + delta * self.p)
        self.radius = max_radius * np.random.rand()

        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        z = move_cube._CUBE_WIDTH / 2 + 1e-6
        self.goal = move_cube.Pose(np.array([x, y, z]))
        move_cube.validate_goal(self.goal)

        self.initial_state = move_cube.sample_goal(difficulty=-1)
        move_cube.validate_goal(self.initial_state)

    def log(self, logger, step):
        logger.log('train/curriculum_p', self.p, step)
        logger.log('train/curriculum_radius', self.radius, step)

    def get_initial_state(self):
        """Get a random initial object pose (always on the ground)."""
        return self.initial_state

    def get_goal(self):
        """Get a random goal depending on the difficulty."""
        return self.goal
