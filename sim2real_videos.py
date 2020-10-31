import argparse
import pybullet
import imageio
import cv2

import robot_fingers

import os
import sys
import json

from trifinger_cameras.utils import convert_image
from trifinger_simulation import camera
from trifinger_simulation.gym_wrapper.envs import cube_env
from trifinger_simulation.tasks import move_cube

import torch
import torch.nn as nn
import numpy as np
import gym
from copy import deepcopy


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
    init.position[2] = max(0.036, init.position[2])
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

def collect_sim_data(policy_path, data, difficulty, goal, n_policies):
    
    steps = len(data)
    init = dotdict(data[0]['achieved_goal'])
    init.position[2] = max(0.036, init.position[2])

    pol_env = make_env(difficulty, init, goal)
    env = make_env(difficulty, init, goal, obs_wrappers=False)
    
    policy = make_policy(pol_env, policy_path, n_policies)
    
    pol_obs = pol_env.reset()
    obs = env.reset()
    
    set_robot(pol_env, data[0]['observation']['position'])
    set_robot(env, data[0]['observation']['position'])
    
    sim_data = []
    
    for t in range(steps):
        action = policy.act(pol_obs)
        
        pol_obs, _, _, _ = pol_env.step(action)
        obs, r, done, info = env.step(action)
        
        obs.update({'desired_action': action})
        sim_data.append(obs)
        
    return sim_data

def collect_open_loop_data(data, difficulty, goal):

    steps = len(data)
    init = dotdict(data[0]['achieved_goal'])
    init.position[2] = max(0.036, init.position[2])

    env = make_env(difficulty, init, goal, obs_wrappers=False, act_wrappers=False)
    
    obs = env.reset()
    
    set_robot(env, data[0]['observation']['position'])
    
    open_loop_data = []
    
    for t in range(steps):
        action = data[t]['desired_action']
        
        obs, r, done, info = env.step(action)
        
        obs.update({'desired_action': action})
        open_loop_data.append(obs)
        
    return open_loop_data

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__




def mlp(input_dim,
        hidden_dim,
        output_dim,
        hidden_depth,
        output_mod=None,
        use_ln=False):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim)]
        if use_ln:
            mods += [nn.LayerNorm(hidden_dim), nn.Tanh()]
        else:
            mods += [nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def flat_space(space, value=None, keys=[]):
    if type(space) == gym.spaces.Box:
        yield (space, value, '_'.join(keys))
    else:
        assert type(space) == gym.spaces.Dict
        for key in space.spaces:
            for x in flat_space(space[key],
                                None if value is None else value[key],
                                keys + [key]):
                yield x
                


class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        low = [
            space.low.flatten()
            for space, _, key in flat_space(self.env.observation_space)
        ]

        high = [
            space.high.flatten()
            for space, _, key in flat_space(self.env.observation_space)
        ]

        self.observation_space = gym.spaces.Box(low=np.concatenate(low,
                                                                   axis=0),
                                                high=np.concatenate(high,
                                                                    axis=0))
        self.obs_slices = []
        offset = 0
        for space, _, key in flat_space(self.env.observation_space):
            n = space.shape[0]
            self.obs_slices.append([key, offset, offset + n])
            offset += n

    def observation(self, obs):

        observation = [
            x.flatten()
            for _, x, key in flat_space(self.env.observation_space, obs)
        ]

        observation = np.concatenate(observation, axis=0)
        return observation



class ActionScalingWrapper(gym.ActionWrapper):
    def __init__(self, env, low, high):
        super().__init__(env)
        self.low = low
        self.high = high

        self.action_space = gym.spaces.Box(low=low,
                                           high=high,
                                           shape=self.env.action_space.shape,
                                           dtype=self.env.action_space.dtype)

    def action(self, action):
        scale = self.high - self.low
        action = (action - self.low) / scale
        true_scale = self.env.action_space.high - self.env.action_space.low
        action = action * true_scale + self.env.action_space.low
        return action.astype(self.env.action_space.dtype)
    
    
class QuaternionToCornersWrapper(gym.ObservationWrapper):
    def __init__(self, env, n):
        super().__init__(env)

        self.n = n
        self.observation_space = deepcopy(self.env.observation_space)

        self.observation_space.spaces['achieved_goal'].spaces[
            'orientation'] = gym.spaces.Box(low=-1e9,
                                            high=1e9,
                                            shape=(3 * n,),
                                            dtype=np.float32)
        self.observation_space.spaces['desired_goal'].spaces[
            'orientation'] = gym.spaces.Box(low=-1e9,
                                            high=1e9,
                                            shape=(3 * n,),
                                            dtype=np.float32)

    def observation(self, obs):
        achieved_pose = move_cube.Pose.from_dict(obs['achieved_goal'])
        desired_pose = move_cube.Pose.from_dict(obs['desired_goal'])

        achieved_corners = move_cube.get_cube_corner_positions(
            achieved_pose)[:self.n, :]
        desired_corners = move_cube.get_cube_corner_positions(
            desired_pose)[:self.n, :]

        obs = deepcopy(obs)

        obs['achieved_goal']['orientation'] = achieved_corners.flatten()
        obs['desired_goal']['orientation'] = desired_corners.flatten()

        return obs
    
    
class Net(nn.Module):
    def __init__(self, obs_dim, action_dim):
        
        super().__init__()
        
        self.trunk = mlp(input_dim=obs_dim,
                       hidden_dim=1024,
                       output_dim=action_dim,
                       hidden_depth=2,
                       use_ln=True)
        
    def forward(self, obs):
        mu = self.trunk(obs)
        mu = torch.tanh(mu)
        return mu

    
class Policy:
    def __init__(self, obs_shape, obs_slices, action_shape, action_range, snapshots, excluded_obses):
        self.action_range = action_range
        self.device = torch.device('cpu')
        
        excluded_obses = [] if excluded_obses is None else excluded_obses.split(
            ':')
        valid_obs_idxs = []
        for key, left, right in obs_slices:
            if key not in excluded_obses:
                for i in range(left, right):
                    valid_obs_idxs.append(i)
        self.valid_obs_idxs = np.array(sorted(valid_obs_idxs))
        
        self.nets = []
        for snapshot in snapshots:
            net = Net(self.valid_obs_idxs.shape[0], action_shape[0])
            net.load_state_dict(torch.load(snapshot, map_location=self.device))
            self.nets.append(net)
        
    def preprocess_obs(self, obs):
        return obs[:, self.valid_obs_idxs]

    def act(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            obs = self.preprocess_obs(obs)
            actions = []
            for net in self.nets:
                action = net(obs)
                action = action.clamp(*self.action_range)
                assert action.ndim == 2 and action.shape[0] == 1
                actions.append(action.cpu().detach().numpy())
            actions = np.concatenate(actions, axis=0)
            action = actions.mean(axis=0)
            return action


def make_policy(env, path, n_policies):
    #snapshots = [path]
    snapshots = [path + f'/policy_{i}.pt' for i in range(n_policies)]
    print(f'workingdir: {os.getcwd()}')
    # for subdir, dirs, files in os.walk('.'):
    #     for file in files:
    #         print(file)
    action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max())
    ]
    
    excluded_obses = 'action' #:desired_goal_orientation:achieved_goal_orientation'

    policy = Policy(obs_shape=env.observation_space.shape,
                    obs_slices=env.obs_slices,
                    action_shape=env.action_space.shape,
                    action_range=action_range,
                    snapshots=snapshots,
                    excluded_obses=excluded_obses)

    return policy


def make_env(difficulty, init, goal, obs_wrappers=True, act_wrappers=True):
    
    initializer = cube_env.FixedInitializer(difficulty, init, goal)
    env = cube_env.CubeEnv(
        initializer, cube_env.ActionType.POSITION, frameskip=1
    )

    #env = wrappers.CubeMarkerWrapper(env)
    if obs_wrappers:
        env = QuaternionToCornersWrapper(env, 1)
        env = FlattenObservationWrapper(env)
    if act_wrappers:
        env = ActionScalingWrapper(env, low=-1.0, high=+1.0)

    return env


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
    parser.add_argument("--fps", default=10, type=int)
    parser.add_argument("--n_policies", default=1, type=int)
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
            'desired_action': desired_action.position,
            'applied_accion': applied_action.position,
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
    open_frames = render_trajectory(open_loop_data, difficulty, goal, args.camera, 
                                        'open loop', args.frameskip)

    print('making policy')
    sim_data = collect_sim_data(args.data_path, data, difficulty, goal, args.n_policies)
    sim_frames = render_trajectory(sim_data, difficulty, goal, args.camera, 
                                        'policy', args.frameskip)

    stacked_frames = []
    for t in range(args.steps // args.frameskip):
        top = np.concatenate([camera_frames[t], replay_frames[t]])
        bottom = np.concatenate([open_frames[t], sim_frames[t]])
        frame = np.concatenate([top, bottom], axis=1)
        stacked_frames.append(frame)
    print('rendering')
    imageio.mimwrite(args.data_path + '/sim.mp4', stacked_frames, fps=args.fps)

    



