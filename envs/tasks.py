from rrc.envs.cube_env import CubeEnv
from rrc_simulation.tasks import move_cube

import numpy as np

from dm_control.utils import rewards

from rrc.envs import ActionType
from rrc.envs import initializers

import pybullet


class Task:
    """a Task specifies a reward and initializer in a CubeEnv."""
    def __init__(self, initializer):
        self.initializer = initializer
        self.difficulty = initializer.difficulty

    def compute_reward(self, observation, platform):
        raise NotImplementedError

    def get_initial_state(self):
        return self.initializer.get_initial_state()

    def get_goal(self):
        return self.initializer.get_goal()


class ReachAndPush(Task):
    """Task to reach and push."""
    def __init__(self, initializer):
        super().__init__(initializer)


    def compute_reward(self, observation, platform):
        radius = move_cube._cube_3d_radius
        robot_id = platform.simfinger.finger_id
        finger_ids = platform.simfinger.pybullet_tip_link_indices

        # compute reward to see if the object reached the target
        object_pos = move_cube.Pose.from_dict(observation['achieved_goal']).position
        target_pos = move_cube.Pose.from_dict(observation['desired_goal']).position
        object_to_target = np.linalg.norm(object_pos - target_pos)
        in_place = rewards.tolerance(object_to_target,
                                     bounds=(0, 0.1 * radius),
                                     margin=radius,
                                     sigmoid='long_tail')


        # compute reward to see that each fingert is close to the cube
        grasp = 0
        hand_away = 0
        for finger_id in finger_ids:
            finger_pos = pybullet.getLinkState(robot_id, finger_id)[0]
            finger_to_object = np.linalg.norm(finger_pos - object_pos)
            grasp += rewards.tolerance(finger_to_object,
                                       bounds=(0, radius),
                                       margin=radius,
                                       sigmoid='long_tail')

            finger_to_target = np.linalg.norm(finger_pos - target_pos)
            hand_away += rewards.tolerance(finger_to_target,
                                           bounds=(3 * radius, np.inf),
                                           margin=4 * radius,
                                           sigmoid='long_tail')

        #import ipdb; ipdb.set_trace()
        grasp /= len(finger_ids)
        hand_away /= len(finger_ids)

        grasp_or_hand_away = grasp * (1 - in_place) + hand_away * in_place
        in_place_weight = 10.0

        return (grasp_or_hand_away +
                in_place_weight * in_place) / (1.0 + in_place_weight)


class ReachObject(Task):
    """Dense reaching task"""
    def __init__(self, initializer):
        super().__init__(initializer)

    def compute_reward(self, observation, platform):
        radius = move_cube._cube_3d_radius
        robot_id = platform.simfinger.finger_id
        finger_ids = platform.simfinger.pybullet_tip_link_indices
        object_pos = move_cube.Pose.from_dict(observation['achieved_goal']).position

        reward = 0
        for finger_id in finger_ids:
            finger_pos = pybullet.getLinkState(robot_id, finger_id)[0]
            finger_to_object = np.linalg.norm(finger_pos - object_pos)
            reward += rewards.tolerance(finger_to_object,
                                       bounds=(0, radius),
                                       margin=radius,
                                       value_at_margin=0.2,
                                       sigmoid='long_tail')

        return reward / 3.


class RRC(Task):
    """the task from the RRC challenge, with exponentiated reward"""
    def __init__(self, initializer):
        super().__init__(initializer)

    def compute_reward(self, observation, platform):
        radius = move_cube._cube_3d_radius
        object_pos = move_cube.Pose.from_dict(observation['achieved_goal']).position
        target_pos = move_cube.Pose.from_dict(observation['desired_goal']).position
        object_to_target = np.linalg.norm(object_pos - target_pos)
        reward = rewards.tolerance(object_to_target,
                                     bounds=(0, 0.1 * radius),
                                     margin=radius,
                                     value_at_margin=0.2,
                                     sigmoid='long_tail')
        return reward
        # reward = -move_cube.evaluate_state(
        #             move_cube.Pose.from_dict(observation['desired_goal']),
        #             move_cube.Pose.from_dict(observation['achieved_goal']),
        #             self.difficulty,
        #             )
        # return np.exp(reward)
