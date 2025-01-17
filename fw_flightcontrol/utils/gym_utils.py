import gymnasium as gym
import numpy as np
from math import pi

class MyNormalizeObservation(gym.wrappers.NormalizeObservation):
    """
        Custom observation normalization wrapper for gym environments.
        Allows to get and set the observation normalization parameters so we can use the same
        ones for training and evaluation of the agent.
    """
    def __init__(self, env: gym.Env, eval: bool = False, epsilon: float = 1e-8):
        super().__init__(env, epsilon)
        self.eval = eval

    def get_obs_rms(self):
        return self.obs_rms

    def set_obs_rms(self, mean, var):
        self.obs_rms.mean = mean
        self.obs_rms.var = var

    def normalize(self, obs):
        if not self.eval: # if training, update the obs rms statistics
            self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)


class NormalizeObservationEnvMinMax(gym.ObservationWrapper):
    """
        Custom observation normalization wrapper for gym environments.
        Normalizes the observation space to be in the range [-1, 1]. With known min and max values from the observation space."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.low_bounds = np.array([300, -300, -136, -136, -pi, -2*pi, -pi, -pi, 0, -1, 0])
        self.high_bounds = np.array([900, 300, 136, 136, pi, 2*pi, pi, pi, 260, 1, 1])

    def observation(self, obs):
        # norm_obs = 2 * (obs - self.observation_space.low) / (self.observation_space.high - self.observation_space.low) - 1
        norm_obs = 2 * (obs - self.low_bounds) / (self.high_bounds - self.low_bounds) - 1
        return norm_obs