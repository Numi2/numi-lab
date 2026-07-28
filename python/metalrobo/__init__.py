"""MetalRobo's native Python and MLX reinforcement-learning interface."""

from .env import FrankaEnv
from .native import (
    MetalRoboError,
    NativeRuntime,
    RuntimeStats,
    library_version,
)
from .ppo import ActorCritic, PPOConfig, PPOTrainer

__all__ = [
    "ActorCritic",
    "FrankaEnv",
    "MetalRoboError",
    "NativeRuntime",
    "PPOConfig",
    "PPOTrainer",
    "RuntimeStats",
    "library_version",
]

__version__ = "0.3.0"
