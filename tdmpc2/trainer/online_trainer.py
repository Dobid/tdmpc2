from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		# reference sequence [roll, pitch] medium/easy
		self.ref_seq: np.ndarray = np.array([
												[np.deg2rad(25), np.deg2rad(15)], # easy
												[np.deg2rad(-25), np.deg2rad(-15)], 
												[np.deg2rad(25), np.deg2rad(-15)],
												[np.deg2rad(-25), np.deg2rad(15)],
												[np.deg2rad(40), np.deg2rad(22)], # medium
												[np.deg2rad(-40), np.deg2rad(-22)],
												[np.deg2rad(40), np.deg2rad(-22)],
												[np.deg2rad(-40), np.deg2rad(22)],
											])

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes = [], []
		self.cfg.eval_episodes = self.ref_seq.shape[0] # set the number of episodes to the number of reference sequences
		init_reset = True # flag to indicate if it is the reset() call is for initialization
		ep_obs, ep_fcs_fluct = [], [] # dicts storing the observations and fluctuation of the flight controls for an episode
		for i in range(self.cfg.eval_episodes):
			obs, info = self.env.reset()
			if not init_reset:
				ep_fcs_pos_hist = np.array(info['fcs_pos_hist'])
				ep_fcs_fluct.append(np.mean(np.abs(np.diff(ep_fcs_pos_hist, axis=0)), axis=0)) # compute the fcs fluctuation of the episode being reset and append to the list
			obs, info, done, ep_reward, t = obs, info, False, 0, 0
			init_reset = False
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				# Set roll and pitch references
				self.env.set_target_state(self.ref_seq[i, 0], self.ref_seq[i, 1]) # 0: roll, 1: pitch
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_obs.append(info['non_norm_obs']) # append the non-normalized observation to the list
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if self.cfg.save_video:
				self.logger.video.save(self._step)

		all_fcs_fluct = np.mean(np.array(ep_fcs_fluct), axis=0) # compute the mean fcs fluctuation over all episodes
		ep_obs = np.array(ep_obs)
		roll_rmse = np.sqrt(np.mean(np.square(ep_obs[:, 6])))
		pitch_rmse = np.sqrt(np.mean(np.square(ep_obs[:, 7])))

		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
			roll_rmse=roll_rmse,
			pitch_rmse=pitch_rmse,
			ail_fluct=all_fcs_fluct[0],
			ele_fluct=all_fcs_fluct[1],
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
		# initial roll and pitch references
		roll_limit = np.deg2rad(60)
		pitch_limit = np.deg2rad(30)
		a = b = 0.70
		while self._step <= self.cfg.steps:

			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				print("**********")
				if eval_next:
					eval_metrics = self.eval()
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

				obs, info = self.env.reset()
				self._tds = [self.to_td(obs)]
				roll_ref = np.random.uniform(-roll_limit, roll_limit)
				pitch_ref = np.random.uniform(-pitch_limit, pitch_limit)
				print(f"Env Done, new ref : roll = {roll_ref}, pitch = {pitch_ref} sampled")

			# Set roll and pitch references
			# self.env.unwrapped.set_target_state(roll_ref, pitch_ref)
			self.env.set_target_state(roll_ref, pitch_ref)

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
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
	
		self.logger.finish(self.agent)
