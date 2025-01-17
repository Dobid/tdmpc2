# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from fw_flightcontrol.utils import train_utils
from fw_flightcontrol.agents.sac_norm_2 import SACAgent



@hydra.main(version_base=None, config_path="../config", config_name="default")
def train(cfg: DictConfig):
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
            poetry run pip install "stable_baselines3==2.0.0a1"
            """
        )

    if OmegaConf.is_missing(cfg.rl.SAC, "seed"):
        cfg.rl.SAC.seed = random.randint(0, 9999)
        print(f"Seed not provided, using random seed: {cfg.rl.SAC.seed}")
    else:
        print(f"Seed provided, using seed from config: {cfg.rl.SAC.seed}")

    # shorter cfg aliases
    cfg_sac = cfg.rl.SAC
    cfg_sim = cfg.env.jsbsim
    cfg_mdp = cfg.env.task.mdp

    np.set_printoptions(suppress=True)

    run_name = f"sac_{cfg_sac.exp_name}_{cfg_sac.seed}"
    if cfg_sac.track:
        import wandb

        wandb.init(
            project=cfg_sac.wandb_project_name,
            entity=cfg_sac.wandb_entity,
            sync_tensorboard=True,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            tags=["SAC"]
        )
        wandb.define_metric("global_step")
        wandb.define_metric("charts/*", step_metric="global_step")
        wandb.define_metric("losses/*", step_metric="global_step")
        wandb.define_metric("eval/*", step_metric="global_step")

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(cfg_sac).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg_sac.seed)
    np.random.seed(cfg_sac.seed)
    torch.manual_seed(cfg_sac.seed)
    torch.backends.cudnn.deterministic = cfg_sac.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cfg_sac.cuda else "cpu")
    print(f"**** Using Device: {device} ****")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [train_utils.make_env(cfg_sac.env_id, cfg.env, cfg_sim.render_mode, None, eval=False, gamma=cfg_sac.gamma)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    print("Single Env Observation Space Shape = ", envs.single_observation_space.shape)
    unwr_envs = envs.envs[0].unwrapped

    sac_agent = SACAgent(envs, cfg_sac)

    # Automatic entropy tuning
    if cfg_sac.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=cfg_sac.q_lr)
    else:
        alpha = cfg_sac.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        int(cfg_sac.buffer_size),
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    # initial roll and pitch references
    targets = train_utils.sample_targets(True, cfg_sac.env_id, cfg, cfg_sac)
    global_step = 0
    prev_gl_step = 0

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(options=cfg_sim.train_sim_options)
    for global_step in range(int(cfg_sac.total_timesteps)):
        if cfg_sac.track:
            wandb.log({"global_step": global_step})

        # run periodic evaluation
        prev_div, _ = divmod(prev_gl_step, cfg_sac.eval_freq)
        curr_div, _ = divmod(global_step, cfg_sac.eval_freq)
        if cfg_sac.periodic_eval and (prev_div != curr_div or global_step == 0):
            eval_dict = train_utils.periodic_eval(cfg_sac.env_id, cfg_mdp, cfg_sim, envs.envs[0], actor, device)
            for k, v in eval_dict.items():
                writer.add_scalar("eval/" + k, v, global_step)
        prev_gl_step = global_step

        # ALGO LOGIC: put action logic here
        unwr_envs.set_target_state(targets)
        if global_step < cfg_sac.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = sac_agent.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        done = np.logical_or(terminations, truncations)
        if done:
            targets = train_utils.sample_targets(True, cfg_sac.env_id, cfg, cfg_sac)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']} \n" + \
                      f"episode_end={info['episode_end']}, out_of_bounds={info['out_of_bounds']}\n********")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > cfg_sac.learning_starts:
            data = rb.sample(cfg_sac.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = sac_agent.get_action(data.next_observations)
                qfs_next_target = sac_agent.get_targetq_value(data.next_observations, next_state_actions)
                qf1_next_target = qfs_next_target[0]
                qf2_next_target = qfs_next_target[1]
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * cfg_sac.gamma * (min_qf_next_target).view(-1)

            qfs_a_values = sac_agent.get_q_value(data.observations, data.actions)
            qf1_a_values = qfs_a_values[0].view(-1)
            qf2_a_values = qfs_a_values[1].view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            sac_agent.q_optimizer.zero_grad()
            qf_loss.backward()
            sac_agent.q_optimizer.step()

            if global_step % cfg_sac.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    cfg_sac.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    # SAC policy loss
                    pi, log_pi, _ = sac_agent.actor.get_action(data.observations)
                    qfs_pi = sac_agent.get_q_value(data.observations, pi)
                    qf1_pi = qfs_pi[0]
                    qf2_pi = qfs_pi[1]
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)

                    # CAPS loss ts
                    act_mean = sac_agent.get_action(data.observations)[2]
                    next_act_mean = sac_agent.get_action(data.next_observations)[2]
                    ts_loss = F.mse_loss(act_mean, next_act_mean)

                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean() + cfg_sac.ts_coef * ts_loss

                    sac_agent.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    sac_agent.actor_optimizer.step()

                    if cfg_sac.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = sac_agent.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % cfg_sac.target_network_frequency == 0:
                for param, target_param in zip(sac_agent.qf1.parameters(), sac_agent.qf1.parameters()):
                    target_param.data.copy_(cfg_sac.tau * param.data + (1 - cfg_sac.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(cfg_sac.tau * param.data + (1 - cfg_sac.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if cfg_sac.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if cfg_sac.final_traj_plot:
        train_utils.final_traj_plot(envs.envs[0], cfg_sac.env_id, cfg_sim, actor, device, run_name)

    train_utils.save_model_SAC(run_name, actor, qf1, qf2, cfg_sac.seed)
    envs.close()
    writer.close()


if __name__ == "__main__":
    train()

