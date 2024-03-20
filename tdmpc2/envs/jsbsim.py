import gymnasium as gym
import jsbgym

def make_env(cfg):
    """
        Make an env for TD-MPC2 experiments in the JSBSim simulator.
    """
    env = gym.make('SimpleAC-v0', config_file = '../../../jsbsim_cfg.yaml', telemetry_file = 'telemetry.csv',
                   render_mode = 'none')
    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env