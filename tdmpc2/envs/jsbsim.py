import gymnasium as gym
import jsbgym

def make_env(cfg):
    """
        Make an env for TD-MPC2 experiments in the JSBSim simulator.
    """
    env = gym.make('ACNoVaIntegErr-v0', config_file = '../../../jsbsim_cfg.yaml', telemetry_file = 'telemetry.csv',
                   render_mode = 'none')
    # env = gym.make('SimpleAC_OMAC-v0', config_file = '../../../jsbsim_cfg.yaml', telemetry_file = 'telemetry.csv',
    #                render_mode = 'none')
    env = gym.wrappers.RecordEpisodeStatistics(env)

    print("observation space: ", env.observation_space.shape)
    print("action space: ", env.action_space.shape)

    return env