import random
import torch
import numpy as np
import hydra
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from fw_flightcontrol.agents import ppo
from fw_flightcontrol.utils.train_utils import make_env
from fw_jsbgym.trim.trim_point import TrimPoint



@hydra.main(version_base=None, config_path="../../config", config_name="default")
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
    env = make_env('AltitudeTracking-v0', cfg.env, cfg_sim.render_mode,
                       'telemetry/telemetry.csv', eval=True)()

    # loading the agent
    train_dict = torch.load(cfg.model_path, map_location=device)
    ppo_agent = ppo.Agent_PPO(env, cfg).to(device)
    ppo_agent.load_state_dict(train_dict['agent'])
    ppo_agent.eval()

    # loading the agent
    trim = TrimPoint('x8')
    trim_action = np.array([trim.aileron, trim.elevator, trim.throttle])

    obs, _ = env.reset(options=cfg_sim.eval_sim_options)
    ep_obss = [obs]
    obs = torch.Tensor(obs).unsqueeze_(0).to(device)

    ep_rewards = [0]
    rolls = [env.unwrapped.sim['attitude/roll-rad']]
    step = 0
    target = np.array([630])
    while step < total_steps:
        env.set_target_state(target)
        # action = env.action_space.sample()
        action = ppo_agent.get_action_and_value(obs)[1].squeeze_(0).detach().cpu().numpy()
        # obs, reward, terminated, truncated, info = env.step([trim_action[1], 1.0])
        # obs, reward, terminated, truncated, info = env.step([trim_action[1], trim_action[2]])
        obs, reward, terminated, truncated, info = env.step(action)
        ep_obss.append(obs)
        ep_rewards.append(reward)
        obs = torch.Tensor(obs).unsqueeze_(0).to(device)
        rolls.append(env.unwrapped.sim['attitude/roll-rad'])

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
    errs_z = ep_obss[:, 1]
    dist_to_target = np.sqrt(errs_z**2)
    tsteps = np.linspace(0, ep_obss.shape[0], ep_obss.shape[0])
    fig, ax = plt.subplots(2, 3)

    ax[0,0].plot(tsteps, np.rad2deg(ep_obss[:, 3]))
    ax[0,0].set_title("Pitch")

    ax[0,1].plot(tsteps, ep_obss[:, 0])
    ax[0,1].set_title("Altitude")

    ax[0,2].plot(tsteps, ep_obss[:, 8])
    ax[0,2].set_title("Airspeed")

    # plot the positions in a 3D plot with the line being a gradient of colors between red and purple, red being the first timestep and purple the last
    # ax[0,2].remove()
    # ax[0,2] = fig.add_subplot(1,3,3, projection='3d')
    # ax[0,2].plot([target[0]], [target[1]], [target[2]], 'ro')
    # ax[0,2].set_title("X")
    # ax[0,2].set_ylabel("Y")
    # ax[0,2].set_zlabel("Z")
    # ax[0,2].set_xlim(-400, 400)
    # ax[0,2].set_ylim(-100, 400)
    # ax[0,2].set_zlim(400, 800)

 
    # for i in range(ep_obss.shape[0] - 1):
    #     ax[0,2].plot(ep_obss[i:i+2, 0], ep_obss[i:i+2, 1], ep_obss[i:i+2, 2], c=plt.cm.plasma(i/ep_obss.shape[0]))

    ax[1,0].plot(tsteps, dist_to_target)
    ax[1,0].set_title("Distance to target")

    # ax[1,1].plot(tsteps, ep_rewards)
    # ax[1,1].set_title("Rewards")

    ax[1,1].plot(tsteps, np.rad2deg(rolls))
    ax[1,1].set_title("Roll")

    ax[1,2].plot(tsteps, ep_obss[:, 9], label="elevator")
    ax[1,2].plot(tsteps, ep_obss[:, 10], label="throttle")
    ax[1,2].set_title("Commands")
    ax[1,2].legend()

    # print(f"Last position: {ep_obss[-1, :3]}")

    plt.show()



if __name__ == '__main__':
    main()
