from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PPOConfig:
    # Environment
    grid_size: int = 32
    n_channels: int = 4
    n_features: int = 4
    action_nvec: Tuple[int, ...] = (3, 32, 32, 4)  # MultiDiscrete: [NOOP/DRILL/POLE, row, col, dir]

    # Training schedule
    total_timesteps: int = 1_000_000
    n_steps: int = 512       # rollout length before each PPO update
    n_epochs: int = 4        # PPO update epochs per rollout
    batch_size: int = 64     # minibatch size

    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4

    # Network architecture
    cnn_embedding_dim: int = 256
    feature_embedding_dim: int = 64
    trunk_dim: int = 256

    # Logging / checkpointing
    log_interval: int = 1        # log every N rollouts
    save_interval: int = 100     # save checkpoint every N rollouts
    checkpoint_dir: str = "checkpoints"

    # Episode length
    max_episode_steps: int = 256

    # Runtime
    seed: int = 0
    device: str = "cuda"
