import argparse
import pybullet
import imageio
import cv2

import robot_fingers

import os
import sys
import json

from trifinger_cameras.utils import convert_image
from trifinger_simulation_v2 import camera

import torch
import numpy as np
import gym
from copy import deepcopy

from sim2real import collect_sim_data, collect_open_loop_data, dotdict, make_env


def set_robot(env, pos):
    robot_id = env.platform.simfinger.finger_id
    joint_ids = env.platform.simfinger.pybullet_joint_indices
    for i,j in enumerate(joint_ids):
        pybullet.resetJointState(robot_id, j, pos[i])
    return

def render_trajectory(data, difficulty, goal, camera_id, 
                        caption_text=None, frameskip=1):
        
    frames = []
    
    cameras = camera.TriFingerCameras() # image_size=(256, 256)
    
    init = dotdict(data[0]['achieved_goal'])
    init.position[2] = max(0.037, init.position[2])
    env = make_env(difficulty, init, goal, obs_wrappers=False)
    env.reset()
    
    robot_id = env.platform.simfinger.finger_id
    finger_ids = env.platform.simfinger.pybullet_tip_link_indices
    joint_ids = env.platform.simfinger.pybullet_joint_indices
    block_id = env.platform.cube.block
    
    for t in range(len(data)):
        if t % frameskip == 0:
            pos = data[t]['observation']['position']
            for i,j in enumerate(joint_ids):
                pybullet.resetJointState(robot_id, j, pos[i])
                
            block_pos = data[t]['achieved_goal']['position']
            block_or = data[t]['achieved_goal']['orientation']
            pybullet.resetBasePositionAndOrientation(block_id, block_pos, block_or)
                
            img = np.float32(cameras.get_images()[camera_id])
            if caption_text is not None:
                caption(img, caption_text)
            frames.append(img)
    
    return frames


def caption(img, text):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,200)
    fontScale              = 1
    fontColor              = (0,0,0)
    lineType               = 2

    cv2.putText(img, text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--camera", default=0, type=int)
    parser.add_argument("--steps", default=1000, type=int)
    parser.add_argument("--frameskip", default=10, type=int)
    parser.add_argument("--fps", default=1, type=int)
    parser.add_argument("--n_policies", default=1, type=int)
    #parser.add_argument("--visualization", default=False)
    args = parser.parse_args()

    goal_path = os.path.join(args.data_path, 'goal.json')
    with open(goal_path) as f:
        goal_dict = dotdict(json.load(f))
    difficulty = goal_dict.difficulty
    goal = dotdict(goal_dict.goal)
    goal.position = np.array(goal.position)
    goal.orientation = np.array(goal.orientation)

    #data = torch.load(args.data_path + '/data.pt')

    print('loading data')
    robot_data_file = os.path.join(args.data_path, 'robot_data.dat')
    camera_data_file = os.path.join(args.data_path, 'camera_data.dat')
    log = robot_fingers.TriFingerPlatformLog(robot_data_file, camera_data_file)

    data = []
    camera_frames = []

    for t in range(log.get_first_timeindex(), log.get_first_timeindex() + args.steps):
        robot_observation = log.get_robot_observation(t)
        camera_observation = log.get_camera_observation(t)
        desired_action = log.get_desired_action(t)
        applied_action = log.get_desired_action(t)

        observation = {
            'observation': {
                'position': robot_observation.position,
                'velocity': robot_observation.velocity,
                'torque': robot_observation.torque,
            },
            'desired_action': desired_action.torque,
            'applied_accion': applied_action.torque,
            'achieved_goal': {
                'position': camera_observation.object_pose.position,
                'orientation': camera_observation.object_pose.orientation,
            },
        }
        data.append(observation)
        if t % args.frameskip == 0:
            camera_frames.append(convert_image(
                            camera_observation.cameras[args.camera].image))
    
    data_save_path = os.path.join(args.data_path, 'data.pt')
    torch.save(data, data_save_path)
    data = torch.load(data_save_path)
    
    print('making replay')
    replay_frames = render_trajectory(data, difficulty, goal, args.camera, 
                                        'replay', args.frameskip)

    print('making open loop')
    open_loop_data = collect_open_loop_data(data, difficulty, goal)
    open_data_save_path = os.path.join(args.data_path, 'open_loop_data.pt')
    torch.save(open_loop_data, open_data_save_path)
    open_frames = render_trajectory(open_loop_data, difficulty, goal, args.camera, 
                                        'open loop', args.frameskip)

    print('making policy')
    policy_data = collect_sim_data(args.data_path, data, difficulty, goal, args.n_policies)
    policy_data_save_path = os.path.join(args.data_path, 'policy_data.pt')
    torch.save(policy_data, policy_data_save_path)
    policy_frames = render_trajectory(policy_data, difficulty, goal, args.camera, 
                                        'policy', args.frameskip)

    stacked_frames = []
    for t in range(args.steps // args.frameskip):
        top = np.concatenate([camera_frames[t], replay_frames[t]])
        bottom = np.concatenate([open_frames[t], policy_frames[t]])
        frame = np.concatenate([top, bottom], axis=1)
        stacked_frames.append(frame)
    print('rendering')
    imageio.mimwrite(args.data_path + '/sim.mp4', stacked_frames, fps=args.fps)

    



