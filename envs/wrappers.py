import numpy as np
import gym


def flat_space(space, value=None):
    if type(space) == gym.spaces.Box:
        yield (space, value)
    else:
        assert type(space) == gym.spaces.Dict
        for key in space.spaces:
            for x in flat_space(space[key],
                                None if value is None else value[key]):
                yield x


class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        low = [
            space.low.flatten()
            for space, _ in flat_space(self.env.observation_space)
        ]

        high = [
            space.high.flatten()
            for space, _ in flat_space(self.env.observation_space)
        ]

        self.observation_space = gym.spaces.Box(low=np.concatenate(low,
                                                                   axis=0),
                                                high=np.concatenate(high,
                                                                    axis=0))

    def _transform(self, obs):
        obs = [
            x.flatten() for _, x in flat_space(self.env.observation_space, obs)
        ]
        return np.concatenate(obs, axis=0)

    def observation(self, obs):
        return self._transform(obs)

    def relabel_transition(self, idx, k):
        transitions = []
        for t in self.env.relabel_transition(idx, k):
            t['obs'] = self._transform(t['obs'])
            t['next_obs'] = self._transform(t['next_obs'])
            transitions.append(t)

        return transitions


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


"""
def FlattenActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        pass
"""
