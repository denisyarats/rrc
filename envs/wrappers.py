import numpy as np
import gym
from copy import deepcopy
from scipy.spatial.transform import Rotation as R

from trifinger_simulation_v2 import visual_objects
from trifinger_simulation_v2.tasks import move_cube


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


class QuaternionToMatrixWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = deepcopy(self.env.observation_space)

        self.observation_space.spaces['achieved_goal'].spaces[
            'orientation'] = gym.spaces.Box(low=-1e9,
                                            high=1e9,
                                            shape=(9,),
                                            dtype=np.float32)
        self.observation_space.spaces['desired_goal'].spaces[
            'orientation'] = gym.spaces.Box(low=-1e9,
                                            high=1e9,
                                            shape=(9,),
                                            dtype=np.float32)

    def observation(self, obs):
        def quat_to_mat(q):
            return R.from_quat(q).as_matrix().flatten().copy()

        obs = deepcopy(obs)
        obs['achieved_goal']['orientation'] = quat_to_mat(
            obs['achieved_goal']['orientation'])
        obs['desired_goal']['orientation'] = quat_to_mat(
            obs['desired_goal']['orientation'])

        return obs


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


class QuaternionToEulerWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = deepcopy(self.env.observation_space)

        self.observation_space.spaces['achieved_goal'].spaces[
            'orientation'] = gym.spaces.Box(low=-1.0,
                                            high=1.0,
                                            shape=(6,),
                                            dtype=np.float32)
        self.observation_space.spaces['desired_goal'].spaces[
            'orientation'] = gym.spaces.Box(low=-1.0,
                                            high=1.0,
                                            shape=(6,),
                                            dtype=np.float32)

    def observation(self, obs):
        def quat_to_euler(q):
            e = R.from_quat(q).as_euler('xyz')
            fs = []
            for i in range(e.shape[0]):
                fs += [np.sin(e[i]), np.cos(e[i])]
            return np.array(fs)

        obs = deepcopy(obs)
        obs['achieved_goal']['orientation'] = quat_to_euler(
            obs['achieved_goal']['orientation'])
        obs['desired_goal']['orientation'] = quat_to_euler(
            obs['desired_goal']['orientation'])

        return obs


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
    def __init__(self, env, alpha, low, high):
        super().__init__(env)
        self.alpha = alpha
        self.low = low
        self.high = high

        self.action_space = gym.spaces.Box(low=low,
                                           high=high,
                                           shape=self.env.action_space.shape,
                                           dtype=self.env.action_space.dtype)

    def action(self, action):
        scale = self.high - self.low
        # decrease magnitude
        action *= self.alpha
        action = (action - self.low) / scale
        true_scale = self.env.action_space.high - self.env.action_space.low
        action = action * true_scale + self.env.action_space.low
        return action.astype(self.env.action_space.dtype)


class CubeMarkerWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        self.goal_marker = visual_objects.CubeMarker(
            width=0.065,
            position=self.env.goal['position'],
            orientation=self.env.goal['orientation'])

        return obs


class RandomizedTimeStepWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
    ):
        super().__init__(env)

    def reset(self, **kwargs):
        low = 1.0 / 1000.0
        high = 1.0 / 60.0

        time_step_s = np.random.uniform(low, high)

        obs = self.env.reset(time_step_s=time_step_s, **kwargs)

        return obs


class DomainRandomizationWrapper(gym.Wrapper):
    def __init__(self, env, obj_pos_noise_std, time_step_low, time_step_high,
                 cube_mass_low, cube_mass_high, gravity_low, gravity_high,
                 restitution_low, restitution_high, max_velocity_low,
                 max_velocity_high, lateral_friction_low,
                 lateral_friction_high, camera_rate_fps_low, camera_rate_fps_high):

        super().__init__(env)

        self.obj_pos_noise_std = obj_pos_noise_std
        self.time_step_low = time_step_low
        self.time_step_high = time_step_high
        self.cube_mass_low = cube_mass_low
        self.cube_mass_high = cube_mass_high
        self.gravity_low = gravity_low
        self.gravity_high = gravity_high
        self.restitution_low = restitution_low
        self.restitution_high = restitution_high
        self.max_velocity_low = max_velocity_low
        self.max_velocity_high = max_velocity_high
        self.lateral_friction_low = lateral_friction_low
        self.lateral_friction_high = lateral_friction_high
        self.camera_rate_fps_low = camera_rate_fps_low
        self.camera_rate_fps_high = camera_rate_fps_high

    def reset(self, **kwargs):
        time_step = np.random.uniform(self.time_step_low, self.time_step_high)
        cube_mass = np.random.uniform(self.cube_mass_low, self.cube_mass_high)
        gravity = np.random.uniform(self.gravity_low, self.gravity_high)
        restitution = np.random.uniform(self.restitution_low,
                                        self.restitution_high)
        max_velocity = np.random.uniform(self.max_velocity_low,
                                         self.max_velocity_high)
        lateral_friction = np.random.uniform(self.lateral_friction_low,
                                             self.lateral_friction_high)
        camera_rate_fps = np.random.uniform(self.camera_rate_fps_low, self.camera_rate_fps_high)

        obs = self.env.reset(time_step_s=time_step,
                             cube_mass=cube_mass,
                             gravity=gravity,
                             restitution=restitution,
                             max_velocity=max_velocity,
                             lateral_friction=lateral_friction,
                             camera_rate_fps=camera_rate_fps,
                             **kwargs)
        
        return self.observation(obs)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, done, info

    def observation(self, obs):

        pos_noise = np.random.normal(
            0.0,
            self.obj_pos_noise_std,
            size=obs['achieved_goal']['position'].shape)
        or_noise = np.random.normal(
            0.0,
            self.obj_pos_noise_std,
            size=obs['achieved_goal']['orientation'].shape)

        obs['achieved_goal']['position'] += pos_noise
        obs['achieved_goal']['orientation'] += or_noise

        return obs


class RandomizedObjectPositionWrapper(gym.ObservationWrapper):
    def __init__(self, env, std=0.01):
        super().__init__(env)
        self.std = std

    def observation(self, obs):

        pos_noise = np.random.normal(
            0.0, self.std, size=obs['achieved_goal']['position'].shape)
        or_noise = np.random.normal(
            0.0, self.std, size=obs['achieved_goal']['orientation'].shape)

        obs['achieved_goal']['position'] += pos_noise
        obs['achieved_goal']['orientation'] += or_noise

        return obs
