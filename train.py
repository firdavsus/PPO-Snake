import argparse

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from game import GAME
from ppo import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--total_timesteps", type=int, default=3_000_000)
    parser.add_argument("--timesteps_per_batch", type=int, default=4800)
    parser.add_argument("--max_timesteps_per_episode", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    env = GAME(grid_size=args.grid_size)
    obs_dim = args.grid_size * args.grid_size
    act_dim = 4  # up, down, left, right

    model = PPO(
        env, obs_dim, act_dim,
        timesteps_per_batch=args.timesteps_per_batch,
        max_timesteps_per_episode=args.max_timesteps_per_episode,
        lr=args.lr,
    )

    history = model.learn(total_timesteps=args.total_timesteps)

    torch.save(model.actor.state_dict(), "ppo_actor.pth")
    torch.save(model.critic.state_dict(), "ppo_critic.pth")

    plt.figure(figsize=(8, 4))
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Average episodic reward")
    plt.title("PPO training progress on Snake")
    plt.tight_layout()
    plt.savefig("training_curve.png")
    print("Saved ppo_actor.pth, ppo_critic.pth, training_curve.png")


if __name__ == "__main__":
    main()