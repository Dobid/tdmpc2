import random
import torch
import numpy as np
import hydra
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from fw_flightcontrol.agents import ppo
from fw_flightcontrol.utils.train_utils import make_env
from fw_jsbgym.trim.trim_point import TrimPoint



@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig):
    np.set_printoptions(precision=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"**** Using Device: {device} ****")

    # seeding
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    total_steps = 2000

    # shorter cfg aliases
    cfg_ppo = cfg.rl.PPO
    cfg_sim = cfg.env.jsbsim
    cfg_task = cfg.env.task

    # env setup
    env = make_env(cfg_ppo.env_id, cfg.env, cfg_sim.render_mode,
                       'telemetry/telemetry.csv', eval=True)()

    # loading the agent
    trim = TrimPoint('x8')
    trim_action = np.array([trim.aileron, trim.elevator, trim.throttle])

    obs, _ = env.reset(options=cfg_sim.eval_sim_options)
    # obs = torch.Tensor(obs).unsqueeze_(0).to(device)

    ep_obss = [obs]
    ep_rewards = [0]
    step = 0
    target = np.array([0, 330, 600])
    while step < total_steps:
        env.set_target_state(target[0], target[1], target[2])
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(trim_action)
        # obs = torch.Tensor(obs).unsqueeze_(0).to(device)
        ep_obss.append(obs)
        ep_rewards.append(reward)

        done = np.logical_or(terminated, truncated)
        if done:
            if info['out_of_bounds']:
                print("Out of bounds")
                break

            print(f"Episode reward: {info['episode']['r']}")
            print(f"******* {step}/{total_steps} *******")
            break
            obs, last_info = env.reset()
        step += 1

    ep_obss = np.array(ep_obss)
    ep_rewards = np.array(ep_rewards)
    errs_x = ep_obss[:, 3]
    errs_y = ep_obss[:, 4]
    errs_z = ep_obss[:, 5]
    dist_to_target = np.sqrt(errs_x**2 + errs_y**2 + errs_z**2)
    tsteps = np.linspace(0, ep_obss.shape[0], ep_obss.shape[0])
    fig, ax = plt.subplots(2, 3)

      # plot roll and pitch in 2D plot
    ax[0,0].plot(tsteps, ep_obss[:, 6])
    ax[0,0].set_xlabel("Roll")

    ax[0,1].plot(tsteps, ep_obss[:, 7])
    ax[0,1].set_xlabel("Pitch")

    # plot the positions in a 3D plot with the line being a gradient of colors between red and purple, red being the first timestep and purple the last
    ax[0,2].remove()
    ax[0,2] = fig.add_subplot(1,3,3, projection='3d')
    ax[0,2].plot([target[0]], [target[1]], [target[2]], 'ro')
    ax[0,2].set_xlabel("X")
    ax[0,2].set_ylabel("Y")
    ax[0,2].set_zlabel("Z")
    ax[0,2].set_xlim(-400, 400)
    ax[0,2].set_ylim(-100, 400)
    ax[0,2].set_zlim(400, 800)

 
    for i in range(ep_obss.shape[0] - 1):
        ax[0,2].plot(ep_obss[i:i+2, 0], ep_obss[i:i+2, 1], ep_obss[i:i+2, 2], c=plt.cm.plasma(i/ep_obss.shape[0]))

    ax[1,0].plot(tsteps, dist_to_target)
    ax[1,0].set_xlabel("Distance to target")

    ax[1,1].plot(tsteps, ep_rewards)
    ax[1,1].set_xlabel("Rewards")

    print(f"Last position: {ep_obss[-1, :3]}")

    plt.show()



if __name__ == '__main__':
    main()

# if __name__ == '__main__':
#     env_id = "ACBohnNoVa-v0" 
#     config = ""
#     if env_id == "ACBohn-v0":
#         config = "config/ppo_caps.yaml"
#     elif env_id == "ACBohnNoVa-v0":
#         config = "config/ppo_caps_no_va.yaml"

#     env = gym.make(env_id, config, render_mode="plot", telemetry_file="telemetry/sandbox.csv")
#     env = gym.wrappers.RecordEpisodeStatistics(env)

#     obs, _ = env.reset()
#     trim_point = TrimPoint('x8')
#     throttle_action = []
#     throttle_cmd = []
#     action = [0.0, 0.0, 0.0]

#     for step in range(1000):
#         if step == 300:
#             action = [trim_point.aileron, trim_point.elevator, trim_point.throttle]
#         throttle_action.append(action[2])
#         obs, reward, trunc, term, info = env.step(action)
#         throttle_cmd.append(env.sim["fcs/throttle-cmd-norm"])

#         if term or trunc:
#             print("Episode done")
#             break

#     throttle_action = np.array(throttle_action)
#     throttle_cmd = np.array(throttle_cmd)

#     # plot throttle action and throttle cmd on top of each other
#     plt.plot(throttle_action, label="throttle action")
#     plt.plot(throttle_cmd, label="throttle cmd")
#     plt.legend()
#     plt.show()
