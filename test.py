import argparse
import time

import torch

from game import GAME
from model import FeedForwardNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--actor_path", type=str, default="ppo_actor.pth")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.15)
    args = parser.parse_args()

    obs_dim = args.grid_size * args.grid_size
    act_dim = 4

    actor = FeedForwardNN(obs_dim, act_dim)
    actor.load_state_dict(torch.load(args.actor_path, map_location="cpu"))
    actor.eval()

    env = GAME(grid_size=args.grid_size)

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.draw()
            with torch.no_grad():
                logits = actor(obs)
                action = torch.argmax(logits).item()
            obs, rew, done = env.step(action)
            total_reward += rew
            time.sleep(args.delay)
        print(f"Episode {ep + 1} finished | total reward {total_reward} | snake length {len(env.snake)}")


if __name__ == "__main__":
    main()