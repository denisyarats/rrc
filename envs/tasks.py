from cube_env import CubeEnv
from rrc_simulation.tasks import move_cube

import numpy as np

from dm_control.utils import rewards

from envs import ActionType


class Task(CubeEnv):
    """a Task specifies a reward and initializer in a CubeEnv."""

    def __init__(self,
                initializer,
                action_type=ActionType.POSITION,
                frameskip=1,
                visualization=False,
                episode_length=move_cube.episode_length,
            ):
        super(Task, self).__init__(initializer, action_type, frameskip,
                                    visualization, episode_length)

    def compute_reward(self, observation, info):

        raise NotImplementedError


class ReachAndPush(Task):
    """Task to reach and push."""
    def compute_reward(self, observation, info):
        radius = move_cube._cube_3d_radius
        robot_id = self.platform.simfinger.finger_id
        finger_ids = self.platform.simfinger.pybullet_tip_link_indices

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

        finger_pos = pybullet.getLinkState(robot_id, finger_tip_id)[0]
        target_pos = move_cube.Pose.from_dict(observation['desired_goal'])

        dist = np.linalg.norm(finger_pos - target_pos.position)
        radius = move_cube._CUBE_WIDTH

        return rewards.tolerance(dist, bounds=(0, radius), margin=radius)


class ReachObject(Task):
    """Dense reaching task"""
    def compute_reward(self, observation, info):
        radius = move_cube._cube_3d_radius
        robot_id = self.platform.simfinger.finger_id
        finger_ids = self.platform.simfinger.pybullet_tip_link_indices
        object_pos = move_cube.Pose.from_dict(observation['achieved_goal']).position

        dist = 0
        for finger_id in finger_ids:
            finger_pos = pybullet.getLinkState(robot_id, finger_id)[0]
            dist += np.linalg.norm(finger_pos - object_pos)

        return np.exp(-dist)


class RRC(Task):
    """the task from the RRC challenge, with exponentiated reward"""
    def compute_reward(self, observation, info):
        reward = -move_cube.evaluate_state(
                    move_cube.Pose.from_dict(desired_goal),
                    move_cube.Pose.from_dict(achieved_goal),
                    info["difficulty"],
                    )
        return np.exp(reward)
