from time import time

import numpy as np
import torch
import pandas as pd
from tensordict.tensordict import TensorDict

from trainer.base import Trainer
from fw_flightcontrol.utils import train_utils


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		# reference sequence [roll, pitch] for each episode [easy, medium, hard]
		self.ref_seq: np.ndarray = np.array([
												[	# roll			,pitch
													[np.deg2rad(25), np.deg2rad(15)], # easy
													[np.deg2rad(-25), np.deg2rad(-15)],
													[np.deg2rad(25), np.deg2rad(-15)],
													[np.deg2rad(-25), np.deg2rad(15)]
												],
												[
													[np.deg2rad(40), np.deg2rad(22)], # medium
													[np.deg2rad(-40), np.deg2rad(-22)],
													[np.deg2rad(40), np.deg2rad(-22)],
													[np.deg2rad(-40), np.deg2rad(22)]
												],
												[
													[np.deg2rad(55), np.deg2rad(28)], # hard
													[np.deg2rad(-55), np.deg2rad(-28)],
													[np.deg2rad(55), np.deg2rad(-28)],
													[np.deg2rad(-55), np.deg2rad(28)]
												]
											])
		


	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def to_td(self, obs, action=None, reward=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		td = TensorDict(dict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
		), batch_size=(1,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, True
		env_id = f"{self.cfg_all.rl.task}-v0"
		while self._step <= self.cfg.steps:

			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				print("**********")
				if self.logger._wandb:
					self.logger._wandb.log({"global_step": self._step})
				if eval_next and self.cfg.periodic_eval:
					# eval_metrics = self.eval()
					eval_metrics = train_utils.periodic_eval(env_id, self.cfg_all.env.task.mdp, self.cfg_all.env.jsbsim, 
															self.env, self.agent, self.agent.device)
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				# reset the environment with training options
				obs, info = self.env.reset(options=self.cfg_all.env.jsbsim.train_sim_options)
				self._tds = [self.to_td(obs)]
				targets = train_utils.sample_targets(True, env_id, self.cfg_all, self.cfg_all.rl)
				if 'AC' in env_id:
					print(f"Env done, new targets : "\
						f"roll = {np.rad2deg(targets[0]):.3f}, "\
						f"pitch = {np.rad2deg(targets[1]):.3f}")
				elif 'Waypoint' in env_id:
					print(f"Env done, new targets : "\
							f"x = {targets[0]:.3f}, "\
							f"y = {targets[1]:.3f}, "\
							f"z = {targets[2]:.3f}")

			# Set roll and pitch references
			# self.env.unwrapped.set_target_state(roll_ref, pitch_ref)
			self.env.set_target_state(targets)

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			obs, reward, term, trunc, info = self.env.step(action)
			done = np.logical_or(term, trunc)

			self._tds.append(self.to_td(obs, action, reward))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)

			self._step += 1

		# Final plot of a trajectory into wandb
		if self.cfg.final_traj_plot:
			train_utils.final_traj_plot(self.env, env_id, self.cfg_all.env.jsbsim, 
										self.agent, self.agent.device, self.logger.exp_name)

		self.logger.finish(self.agent)
