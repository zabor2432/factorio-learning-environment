from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from rl_training.config import PPOConfig


class RolloutBuffer:
    """
    Fixed-size buffer for one PPO rollout.

    Observation arrays are allocated once at construction to avoid
    per-step allocation in the hot path.
    """

    def __init__(self, n_steps: int, config: PPOConfig):
        C, H, W = config.n_channels, config.grid_size, config.grid_size
        F = config.n_features
        n_dims = len(config.action_nvec)

        self.obs_grid = np.zeros((n_steps, C, H, W), dtype=np.float32)
        self.obs_features = np.zeros((n_steps, F), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_dims), dtype=np.int64)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)

        self.n_steps = n_steps
        self.ptr = 0

    def store(
        self,
        obs: dict,
        action: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        i = self.ptr
        self.obs_grid[i] = obs["grid"]
        self.obs_features[i] = obs["features"]
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.n_steps

    def reset(self) -> None:
        self.ptr = 0

    def compute_gae_returns(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and lambda-returns in one backward pass."""
        advantages = np.zeros(self.n_steps, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        return advantages, returns

    def to_tensors(self, device: torch.device) -> dict:
        return {
            "obs_grid":     torch.from_numpy(self.obs_grid).to(device),
            "obs_features": torch.from_numpy(self.obs_features).to(device),
            "actions":      torch.from_numpy(self.actions).to(device),
            "log_probs":    torch.from_numpy(self.log_probs).to(device),
            "values":       torch.from_numpy(self.values).to(device),
        }


class PPOTrainer:
    """
    Proximal Policy Optimisation with:
    - Clipped surrogate objective
    - Value function loss (MSE)
    - Entropy bonus
    - Factored log-probability for MultiDiscrete actions
    """

    def __init__(self, model: nn.Module, config: PPOConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.buffer = RolloutBuffer(config.n_steps, config)

    def update(self, last_value: float) -> dict:
        """Run PPO update on the current rollout. Returns a dict of training metrics."""
        cfg = self.config
        advantages, returns = self.buffer.compute_gae_returns(
            last_value, cfg.gamma, cfg.gae_lambda
        )

        adv_t = torch.from_numpy(advantages).to(self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.from_numpy(returns).to(self.device)

        tensors = self.buffer.to_tensors(self.device)
        n = self.buffer.n_steps
        idx = np.arange(n)

        total_pg = total_v = total_ent = total_loss = 0.0
        n_updates = 0

        for _ in range(cfg.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, cfg.batch_size):
                b = idx[start: start + cfg.batch_size]
                b_obs = {
                    "grid":     tensors["obs_grid"][b],
                    "features": tensors["obs_features"][b],
                }
                b_actions = tensors["actions"][b]
                b_old_logp = tensors["log_probs"][b]
                b_adv = adv_t[b]
                b_ret = ret_t[b]

                _, new_logp, entropy, new_value = self.model.get_action_and_value(
                    b_obs, b_actions
                )

                ratio = (new_logp - b_old_logp).exp()
                pg1 = -b_adv * ratio
                pg2 = -b_adv * ratio.clamp(1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                v_loss = 0.5 * (new_value - b_ret).pow(2).mean()
                ent_loss = entropy.mean()

                loss = pg_loss + cfg.value_coef * v_loss - cfg.entropy_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                total_pg += pg_loss.item()
                total_v += v_loss.item()
                total_ent += ent_loss.item()
                total_loss += loss.item()
                n_updates += 1

        self.buffer.reset()

        return {
            "policy_loss": total_pg / n_updates,
            "value_loss": total_v / n_updates,
            "entropy": total_ent / n_updates,
            "total_loss": total_loss / n_updates,
        }
