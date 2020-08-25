import numpy as np
import gym

from rrc_simulation import visual_objects


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
    def __init__(self, env, excluded=[]):
        super().__init__(env)

        self.excluded = excluded

        low = [
            space.low.flatten()
            for space, _, key in flat_space(self.env.observation_space)
            if key not in excluded
        ]

        high = [
            space.high.flatten()
            for space, _, key in flat_space(self.env.observation_space)
            if key not in excluded
        ]

        self.observation_space = gym.spaces.Box(low=np.concatenate(low,
                                                                   axis=0),
                                                high=np.concatenate(high,
                                                                    axis=0))

    def observation(self, obs):
        observation = [
            x.flatten()
            for _, x, key in flat_space(self.env.observation_space, obs)
            if key not in self.excluded
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


"""
def FlattenActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        pass
"""
