import random
import torch
import numpy as np
import os
import csv
import hydra

from omegaconf import DictConfig
from common.parser import parse_cfg
from jsbgym.trim.trim_point import TrimPoint
from envs import make_env
from tdmpc2 import TDMPC2


@hydra.main(version_base=None, config_path="config", config_name="config")
def eval(cfg: DictConfig):
    np.set_printoptions(precision=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    print(f"**** Using Device: {device} ****")

    cfg.rl = parse_cfg(cfg.rl)
    os.chdir(hydra.utils.get_original_cwd())

    # shorter cfg aliases
    cfg_rl = cfg.rl
    cfg_sim = cfg.env.jsbsim

    # env setup
    env = make_env(cfg)

    # Load agent
    agent = TDMPC2(cfg.rl)
    assert os.path.exists(cfg.rl.checkpoint), f"Checkpoint {cfg.rl.checkpoint} not found! Must be a valid filepath."
    agent.load(cfg.rl.checkpoint)

    # load the reference sequence and initialize the evaluation arrays
    simple_ref_data = np.load(f'eval/refs/{cfg_rl.ref_file}.npy')

    # set default target values
    # roll_ref: float = np.deg2rad(58)
    # pitch_ref: float = np.deg2rad(28)

    # if no render mode, run the simulation for the whole reference sequence given by the .npy file
    if cfg_sim.render_mode == "none":
        total_steps = 50_000
    else: # otherwise, run the simulation for 8000 steps
        total_steps = 4000

    if cfg_sim.eval_sim_options.atmosphere.severity == "all":
        severity_range = ["off", "light", "moderate", "severe"]
    else:
        severity_range = [cfg_sim.eval_sim_options.atmosphere.severity]

    all_mse = []
    all_rmse = []
    all_fcs_fluct = []

    if not os.path.exists("eval/outputs"):
        os.makedirs("eval/outputs")

    eval_res_csv = f"eval/outputs/{cfg_rl.res_file}.csv"
    eval_fieldnames = ["severity", "roll_mse", "pitch_mse", "roll_rmse", 
                        "pitch_rmse", "roll_fcs_fluct", "pitch_fcs_fluct"]

    with open(eval_res_csv, "w") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=eval_fieldnames)
        csv_writer.writeheader()

    for i, severity in enumerate(severity_range):
        cfg_sim.eval_sim_options.atmosphere.severity = severity
        e_obs = []
        eps_fcs_fluct = []
        print(f"********** TDMPC2 METRICS {severity} **********")
        obs, _ = env.reset(options=cfg_sim.eval_sim_options)
        ep_cnt = 0 # episode counter
        ep_step = 0 # step counter within an episode
        step, t = 0, 0
        refs = simple_ref_data[ep_cnt]
        roll_ref, pitch_ref = refs[0], refs[1]
        while step < total_steps:
            env.set_target_state(roll_ref, pitch_ref)
            # action = agent.get_action_and_value(obs)[1].squeeze_(0).detach().cpu().numpy()
            action = agent.act(obs, t0=t==0, eval_mode=True)
            obs, reward, done, info = env.step(action)
            e_obs.append(info["non_norm_obs"])
            t += 1

            if done:
                t = 0
                if info['out_of_bounds']:
                    print("Out of bounds")
                    e_obs[len(e_obs)-ep_step:] = [] # delete the last observations if the ep is oob
                    step -= ep_step # set the step counter back to the last episode
                    ep_step = 0 # reset the episode step counter
                else:
                    ep_step = 0 # reset the episode step counter
                    ep_cnt += 1 # increment the episode counter
                print(f"Episode reward: {info['episode']['r']}")
                print(f"******* {step}/{total_steps} *******")
                # break
                obs, last_info = env.reset()
                ep_fcs_pos_hist = np.array(last_info["fcs_pos_hist"]) # get fcs pos history of the finished episode
                eps_fcs_fluct.append(np.mean(np.abs(np.diff(ep_fcs_pos_hist, axis=0)), axis=0)) # get fcs fluctuation of the episode and append it to the list of all fcs fluctuations
                if ep_cnt < len(simple_ref_data):
                    refs = simple_ref_data[ep_cnt]
                roll_ref, pitch_ref = refs[0], refs[1]
            ep_step += 1
            step += 1

        all_fcs_fluct.append(np.mean(np.array(eps_fcs_fluct), axis=0))
        e_obs = np.array(e_obs)
        print(f"e_obs shape: {e_obs.shape}")
        print(f"eps_fcs_fluct shape: {np.array(eps_fcs_fluct).shape}")
        roll_mse = np.mean(np.square(e_obs[:, 6]))
        pitch_mse = np.mean(np.square(e_obs[:, 7]))
        all_mse.append([roll_mse, pitch_mse])
        roll_rmse = np.sqrt(roll_mse)
        pitch_rmse = np.sqrt(pitch_mse)
        all_rmse.append([roll_rmse, pitch_rmse])

    for mse, rmse, fcs_fluct, severity in zip(all_mse, all_rmse, all_fcs_fluct, severity_range):
        print("\nSeverity: ", severity)
        print(f"  Roll MSE: {mse[0]:.4f}\n  Pitch MSE: {mse[1]:.4f}")
        print(f"  Roll RMSE: {rmse[0]:.4f}\n  Pitch RMSE: {rmse[1]:.4f}")
        print(f"  Roll fluctuation: {fcs_fluct[0]:.4f}\n  Pitch fluctuation: {fcs_fluct[1]:.4f}")
        with open(eval_res_csv, "a") as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=eval_fieldnames)
            csv_writer.writerow({"severity": severity, "roll_mse": mse[0], "pitch_mse": mse[1], 
                                "roll_rmse": rmse[0], "pitch_rmse": rmse[1], 
                                "roll_fcs_fluct": fcs_fluct[0], "pitch_fcs_fluct": fcs_fluct[1]})

    env.close()


if __name__ == '__main__':
    eval()