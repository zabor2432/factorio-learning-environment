from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl_training.config import PPOConfig


class CNNEncoder(nn.Module):
    """
    3-layer CNN for a (C, 32, 32) spatial grid observation.

    Spatial progression: 32 → 16 → 8 (stride-2 convolutions).
    Output: flat embedding of size `embedding_dim`.
    """

    def __init__(self, in_channels: int, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # 16×16
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),   # 8×8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, embedding_dim),
            nn.ReLU(),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureEncoder(nn.Module):
    """Single-layer MLP encoder for the flat feature vector."""

    def __init__(self, in_features: int, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """
    Actor-Critic network for a MultiDiscrete action space.

    Observation:
        grid     — (B, N_CHANNELS, GRID_SIZE, GRID_SIZE)
        features — (B, N_FEATURES)

    Action: MultiDiscrete with `len(action_nvec)` independent categorical dims.
    One actor head per dimension; log_prob is the sum across dimensions
    (factored-action factorisation).
    """

    def __init__(self, config: PPOConfig):
        super().__init__()
        self.cnn = CNNEncoder(config.n_channels, config.cnn_embedding_dim)
        self.feat_enc = FeatureEncoder(config.n_features, config.feature_embedding_dim)

        joint_dim = config.cnn_embedding_dim + config.feature_embedding_dim
        self.trunk = nn.Sequential(
            nn.Linear(joint_dim, config.trunk_dim),
            nn.ReLU(),
        )

        self.actor_heads = nn.ModuleList([
            nn.Linear(config.trunk_dim, n) for n in config.action_nvec
        ])
        self.critic = nn.Linear(config.trunk_dim, 1)

    def _trunk(self, obs_grid: torch.Tensor, obs_features: torch.Tensor) -> torch.Tensor:
        cnn_emb = self.cnn(obs_grid)
        feat_emb = self.feat_enc(obs_features)
        return self.trunk(torch.cat([cnn_emb, feat_emb], dim=-1))

    def forward(
        self, obs_grid: torch.Tensor, obs_features: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        trunk = self._trunk(obs_grid, obs_features)
        logits = [head(trunk) for head in self.actor_heads]
        value = self.critic(trunk).squeeze(-1)
        return logits, value

    def get_action_and_value(
        self,
        obs: dict,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample or evaluate an action.

        Args:
            obs:    dict with keys "grid" and "features"
            action: (B, n_dims) tensor for evaluation; None for sampling

        Returns:
            action   — (B, n_dims)
            log_prob — (B,)  sum of per-dim log-probs
            entropy  — (B,)  sum of per-dim entropies
            value    — (B,)
        """
        logits_list, value = self(obs["grid"], obs["features"])
        dists = [Categorical(logits=lg) for lg in logits_list]

        if action is None:
            sampled = torch.stack([d.sample() for d in dists], dim=-1)
        else:
            sampled = action

        log_prob = sum(dists[i].log_prob(sampled[:, i]) for i in range(len(dists)))
        entropy = sum(d.entropy() for d in dists)
        return sampled, log_prob, entropy, value
