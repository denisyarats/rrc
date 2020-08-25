#!/usr/bin/env python3
"""Example evaluation script to evaluate a policy.

This is an example evaluation script for evaluating a "RandomPolicy".  Use this
as a base for your own script to evaluate your policy.  All you need to do is
to replace the `RandomPolicy` and potentially the Gym environment with your own
ones (see the TODOs in the code below).

This script will be executed in an automated procedure.  For this to work, make
sure you do not change the overall structure of the script!

This script expects the following arguments in the given order:
 - Difficulty level (needed for reward computation)
 - initial pose of the cube (as JSON string)
 - goal pose of the cube (as JSON string)
 - file to which the action log is written

It is then expected to initialize the environment with the given initial pose
and execute exactly one episode with the policy that is to be evaluated.

When finished, the action log, which is created by the TriFingerPlatform class,
is written to the specified file.  This log file is crucial as it is used to
evaluate the actual performance of the policy.
"""
import sys
import os

import gym

from rrc_simulation.gym_wrapper.envs import cube_env
from rrc_simulation.tasks import move_cube

sys.path.append('../..')
from ddpg import Actor
import torch
import utils
from envs import wrappers
from video import VideoRecorder


class Policy:
    def __init__(self, obs_shape, action_shape, action_range, snapshot_path,
                 device):
        self.actor = Actor(obs_shape,
                           action_shape,
                           hidden_dim=1024,
                           hidden_depth=2,
                           stddev=0.2,
                           parameterization='clipped',
                           use_ln=True,
                           head_init_coef=1.0).to(device)
        self.actor.load_state_dict(torch.load(snapshot_path))
        self.action_range = action_range
        self.device = device

    def act(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])


def make_policy(env):
    device = torch.device('cpu')
    snapshot_path = f'../../pretrained_agents/task_{env.initializer.difficulty}/actor.pt'

    action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max())
    ]

    policy = Policy(obs_shape=env.observation_space.shape,
                    action_shape=env.action_space.shape,
                    action_range=action_range,
                    snapshot_path=snapshot_path,
                    device=device)

    return policy


def make_env(difficulty, initial_pose, goal_pose):
    initializer = cube_env.FixedInitializer(difficulty, initial_pose,
                                            goal_pose)

    env = gym.make(
        "rrc_simulation.gym_wrapper:real_robot_challenge_phase_1-v1",
        initializer=initializer,
        action_type=cube_env.ActionType.POSITION,
        visualization=False,
    )

    excluded = []
    if difficulty in [1, 2, 3]:
        excluded.append('achieved_goal_orientation')
        excluded.append('desired_goal_orientation')

    #env = wrappers.CubeMarkerWrapper(env)
    env = wrappers.FlattenObservationWrapper(env, excluded=excluded)
    env = wrappers.ActionScalingWrapper(env, low=-1.0, high=+1.0)

    return env


def main():
    try:
        difficulty = int(sys.argv[1])
        initial_pose_json = sys.argv[2]
        goal_pose_json = sys.argv[3]
        output_file = sys.argv[4]
    except IndexError:
        print("Incorrect number of arguments.")
        print("Usage:\n"
              "\tevaluate_policy.py <difficulty_level> <initial_pose>"
              " <goal_pose> <output_file>")
        sys.exit(1)


    # the poses are passes as JSON strings, so they need to be converted first
    initial_pose = move_cube.Pose.from_json(initial_pose_json)
    goal_pose = move_cube.Pose.from_json(goal_pose_json)

    # create a FixedInitializer with the given values
    initializer = cube_env.FixedInitializer(difficulty, initial_pose,
                                            goal_pose)

    # TODO: Replace with your environment if you used a custom one.
    env = make_env(difficulty, initial_pose, goal_pose)

    # TODO: Replace this with your model
    # Note: You may also use a different policy for each difficulty level (difficulty)
    policy = make_policy(env)
    
    output_dir = os.path.dirname(output_file)
    video_recorder = VideoRecorder(output_dir, fps=50)

    # Execute one episode.  Make sure that the number of simulation steps
    # matches with the episode length of the task.  When using the default Gym
    # environment, this is the case when looping until is_done == True.  Make
    # sure to adjust this in case your custom environment behaves differently!
    done = False
    observation = env.reset()
    video_recorder.init(enabled=True)
    accumulated_reward = 0
    while not done:
        action = policy.act(observation)
        observation, reward, done, info = env.step(action)
        video_recorder.record()
        accumulated_reward += reward

    print("Accumulated reward: {}".format(accumulated_reward))
    video_file = os.path.basename(output_file).replace('json', 'mp4').replace('action_log', 'video')
    video_recorder.save(video_file)

    # store the log for evaluation
    env.platform.store_action_log(output_file)


if __name__ == "__main__":
    main()
