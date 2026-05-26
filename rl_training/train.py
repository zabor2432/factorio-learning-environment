"""
PPO training entry point for the Factorio coal-mining task.

Usage:
    uv run python -m rl_training.train
    uv run python -m rl_training.train --total-timesteps 500000 --device cpu

Environment variables:
    FACTORIO_SERVER_ADDRESS  — hostname of Factorio server (default: localhost)
    FACTORIO_SERVER_PORT     — RCON port (default: 27015)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from rl_training.config import PPOConfig
from rl_training.networks import ActorCritic
from rl_training.ppo import PPOTrainer

from fle.env import FactorioInstance
from fle.env.atomic_env.environment import CoalMiningAtomicEnv
from fle.eval.tasks.coal_mining_task import CoalMiningTask


def make_env(config: PPOConfig) -> CoalMiningAtomicEnv:
    task = CoalMiningTask(trajectory_length=config.max_episode_steps)
    address = os.getenv("FACTORIO_SERVER_ADDRESS", "localhost")
    tcp_port = int(os.getenv("FACTORIO_SERVER_PORT", "27000"))  # host port per docker-compose
    instance = FactorioInstance(
        address=address,
        tcp_port=tcp_port,
        all_technologies_researched=True,
        peaceful=True,
        inventory=task.STARTING_INVENTORY,
    )
    return CoalMiningAtomicEnv(
        instance=instance,
        task=task,
        max_steps=config.max_episode_steps,
    )


def train(config: PPOConfig) -> None:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = make_env(config)
    # Sync action_nvec from the actual env in case constants changed
    config.action_nvec = tuple(int(n) for n in env.action_space.nvec)
    model = ActorCritic(config).to(device)
    trainer = PPOTrainer(model, config, device)

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    obs, _ = env.reset()
    episode_reward = 0.0
    episode_rewards: list[float] = []
    total_steps = 0
    rollout_count = 0

    print(
        f"Starting PPO  total_timesteps={config.total_timesteps}  "
        f"n_steps={config.n_steps}  batch_size={config.batch_size}"
    )

    while total_steps < config.total_timesteps:
        # ── Collect rollout ──────────────────────────────────────────────────
        for _ in range(config.n_steps):
            obs_t = {
                "grid":     torch.from_numpy(obs["grid"]).unsqueeze(0).to(device),
                "features": torch.from_numpy(obs["features"]).unsqueeze(0).to(device),
            }
            with torch.no_grad():
                action_t, logp_t, _, value_t = model.get_action_and_value(obs_t)

            action_np = action_t.cpu().numpy()[0]
            logp_np = logp_t.cpu().item()
            value_np = value_t.cpu().item()

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            trainer.buffer.store(obs, action_np, logp_np, value_np, reward, done)
            episode_reward += reward
            total_steps += 1
            obs = next_obs

            if done:
                episode_rewards.append(episode_reward)
                print(f"  episode done  reward={episode_reward:.3f}  steps={total_steps}")
                episode_reward = 0.0
                obs, _ = env.reset()

            if trainer.buffer.is_full():
                break

        # Bootstrap value for GAE
        with torch.no_grad():
            obs_t = {
                "grid":     torch.from_numpy(obs["grid"]).unsqueeze(0).to(device),
                "features": torch.from_numpy(obs["features"]).unsqueeze(0).to(device),
            }
            _, _, _, last_value_t = model.get_action_and_value(obs_t)
            last_value = last_value_t.cpu().item()

        # ── PPO update ───────────────────────────────────────────────────────
        metrics = trainer.update(last_value)
        rollout_count += 1

        if rollout_count % config.log_interval == 0:
            recent = episode_rewards[-20:] if episode_rewards else [0.0]
            print(
                f"rollout={rollout_count:5d}  steps={total_steps:8d}  "
                f"ep_reward={np.mean(recent):8.3f}  "
                f"pg_loss={metrics['policy_loss']:.4f}  "
                f"v_loss={metrics['value_loss']:.4f}  "
                f"entropy={metrics['entropy']:.4f}"
            )

        if rollout_count % config.save_interval == 0:
            path = os.path.join(config.checkpoint_dir, f"model_{rollout_count}.pt")
            torch.save({"model_state": model.state_dict(), "config": config}, path)
            print(f"Saved checkpoint: {path}")

    env.close()
    print("Training complete.")


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description="PPO for Factorio coal mining")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()
    return PPOConfig(
        total_timesteps=args.total_timesteps,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    train(parse_args())
