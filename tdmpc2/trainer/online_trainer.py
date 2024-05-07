from time import time

import numpy as np
import torch
import pandas as pd
from tensordict.tensordict import TensorDict

from trainer.base import Trainer


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
													# [np.deg2rad(-25), np.deg2rad(-15)],
													# [np.deg2rad(25), np.deg2rad(-15)],
													# [np.deg2rad(-25), np.deg2rad(15)]
												],
												[
													[np.deg2rad(40), np.deg2rad(22)], # medium
													# [np.deg2rad(-40), np.deg2rad(-22)],
													# [np.deg2rad(40), np.deg2rad(-22)],
													# [np.deg2rad(-40), np.deg2rad(22)]
												],
												[
													[np.deg2rad(55), np.deg2rad(28)], # hard
													# [np.deg2rad(-55), np.deg2rad(-28)],
													# [np.deg2rad(55), np.deg2rad(-28)],
													# [np.deg2rad(-55), np.deg2rad(28)]
												]
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
		dif_obs = []
		dif_fcs_fluct = [] # dicts storing all obs across all episodes and fluctuation of the flight controls for all episodes
		# self.cfg.eval_episodes = self.ref_seq.shape[0] * self.ref_seq.shape[1] # set the number of episodes to the number of reference sequences (3 difficulty levels * 4 episodes per level = 12)
		init_reset = True # flag to indicate if it is the reset() call is for initialization
		i = 0
		for dif_idx, ref_dif in enumerate(self.ref_seq): # iterate over the difficulty levels
			dif_obs.append([])
			dif_fcs_fluct.append([])
			for ref_idx, ref_ep in enumerate(ref_dif): # iterate over the ref for 1 episode
				obs, info = self.env.reset()
				obs, info, done, ep_reward, t = obs, info, False, 0, 0
				if self.cfg.save_video:
					self.logger.video.init(self.env, enabled=(i==0))
				while not done:
					# Set roll and pitch references
					self.env.set_target_state(ref_ep[0], ref_ep[1]) # 0: roll, 1: pitch
					action = self.agent.act(obs, t0=t==0, eval_mode=True)
					obs, reward, done, info = self.env.step(action)
					dif_obs[dif_idx].append(info['non_norm_obs']) # append the non-normalized observation to the list
					ep_reward += reward
					t += 1
					if self.cfg.save_video:
						self.logger.video.record(self.env)

				ep_fcs_pos_hist = np.array(info['fcs_pos_hist'])
				dif_fcs_fluct[dif_idx].append(np.mean(np.abs(np.diff(ep_fcs_pos_hist, axis=0)), axis=0)) # compute the fcs fluctuation of the episode being reset and append to the list

				ep_rewards.append(ep_reward)
				ep_successes.append(info['success'])
				if self.cfg.save_video:
					self.logger.video.save(self._step)
				i += 1
		
		# computing the mean fcs fluctuation across all episodes for each difficulty level
		dif_fcs_fluct = np.array(dif_fcs_fluct)
		easy_fcs_fluct = np.mean(np.array(dif_fcs_fluct[0]), axis=0)
		medium_fcs_fluct = np.mean(np.array(dif_fcs_fluct[1]), axis=0)
		hard_fcs_fluct = np.mean(np.array(dif_fcs_fluct[2]), axis=0)

		# computing the rmse of the roll and pitch angles across all episodes for each difficulty level
		dif_obs = np.array(dif_obs)
		easy_roll_rmse = np.sqrt(np.mean(np.square(dif_obs[0, :, 6])))
		easy_pitch_rmse = np.sqrt(np.mean(np.square(dif_obs[0, :, 7])))
		medium_roll_rmse = np.sqrt(np.mean(np.square(dif_obs[1, :, 6])))
		medium_pitch_rmse = np.sqrt(np.mean(np.square(dif_obs[1, :, 7])))
		hard_roll_rmse = np.sqrt(np.mean(np.square(dif_obs[2, :, 6])))
		hard_pitch_rmse = np.sqrt(np.mean(np.square(dif_obs[2, :, 7])))

		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
			easy_roll_rmse=easy_roll_rmse,
			easy_pitch_rmse=easy_pitch_rmse,
			medium_roll_rmse=medium_roll_rmse,
			medium_pitch_rmse=medium_pitch_rmse,
			hard_roll_rmse=hard_roll_rmse,
			hard_pitch_rmse=hard_pitch_rmse,
			easy_ail_fluct=easy_fcs_fluct[0],
			easy_ele_fluct=easy_fcs_fluct[1],
			medium_ail_fluct=medium_fcs_fluct[0],
			medium_ele_fluct=medium_fcs_fluct[1],
			hard_ail_fluct=hard_fcs_fluct[0],
			hard_ele_fluct=hard_fcs_fluct[1],
		)
	

	def log_test_traj(self):
		"""
			Log the trajectory telemetry of a test episode.
			Done at the end of the training, to give an graphical idea of the agent's performance.
		"""
		telemetry_file = self.logger._log_dir / 'telemetry.csv'
		obs, info = self.env.reset(options={'render_mode': 'log'})
		obs, info, done, ep_reward, t = obs, info, False, 0, 0
		self.env.telemetry_setup(telemetry_file)
		roll_ref = np.deg2rad(30)
		pitch_ref = np.deg2rad(15)
		while not done:
			# Set roll and pitch references
			self.env.set_target_state(roll_ref, pitch_ref) # 0: roll, 1: pitch
			action = self.agent.act(obs, t0=t==0, eval_mode=True)
			obs, reward, done, info = self.env.step(action)
			ep_reward += reward
			t += 1
		print(f"End of training test trajectory episode reward: {ep_reward}")
		telemetry_df = pd.read_csv(telemetry_file)
		telemetry_table = self.logger._wandb.Table(dataframe=telemetry_df)
		self.logger._wandb.log({"telemetry": telemetry_table})


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

		# Final evaluation
		if self.cfg.final_traj:
			self.log_test_traj()

		self.logger.finish(self.agent)
