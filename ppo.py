import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam

from model import FeedForwardNN


class PPO:
    """
    A from-scratch PPO implementation (clipped-surrogate, single-env,
    rewards-to-go advantage estimation) for a discrete-action environment
    like the snake game, where env.reset() -> obs and
    env.step(action_idx) -> (next_obs, reward, done).
    """

    def __init__(self, env, obs_dim, act_dim, **hyperparameters):
        # Default hyperparameters, overridable via kwargs
        self.gamma = 0.99
        self.clip = 0.2
        self.lr = 3e-4
        self.n_updates_per_iteration = 5
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 300
        self.ent_coef = 0.01       # entropy bonus, helps avoid premature policy collapse
        self.max_grad_norm = 0.5
        self.save_freq = 25

        for k, v in hyperparameters.items():
            setattr(self, k, v)

        self.env = env
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.actor = FeedForwardNN(obs_dim, act_dim)
        self.critic = FeedForwardNN(obs_dim, 1)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

    def get_action(self, obs):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()
        logits = self.actor(batch_obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(batch_acts)
        entropy = dist.entropy()
        return V, log_probs, entropy

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        return torch.tensor(batch_rtgs, dtype=torch.float32)

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []

        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = []
            obs = self.env.reset()

            ep_t = 0
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, done = self.env.step(action)

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32)
        batch_acts = torch.tensor(batch_acts, dtype=torch.long)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32)
        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rews

    def learn(self, total_timesteps, verbose=True):
        t_so_far = 0
        i_so_far = 0
        history = []

        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_rews = self.rollout()

            t_so_far += int(np.sum(batch_lens))
            i_so_far += 1

            with torch.no_grad():
                V = self.critic(batch_obs).squeeze()
            A_k = batch_rtgs - V
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs, entropy = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean() - self.ent_coef * entropy.mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()

            avg_ep_rew = float(np.mean([np.sum(ep_rews) for ep_rews in batch_rews]))
            avg_ep_len = float(np.mean(batch_lens))
            history.append(avg_ep_rew)

            if verbose:
                print(f"Iter {i_so_far:4d} | Timesteps {t_so_far:7d} | "
                      f"AvgEpLen {avg_ep_len:6.1f} | AvgEpReward {avg_ep_rew:7.2f}")

            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), "ppo_actor.pth")
                torch.save(self.critic.state_dict(), "ppo_critic.pth")

        return history