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
