"""MetalRobo's native Python and MLX reinforcement-learning interface."""

from .env import FrankaEnv
from .native import (
    MetalRoboError,
    NativeRuntime,
    RuntimeStats,
    library_version,
)
from .mlx_world import (
    ContactEvidence,
    MLXCompiledWorld,
    MetalWorldCapacityProfile,
    SceneBodyState,
    SolverCache,
    StepOutput,
    WorldState,
    compile_world,
    initial_state,
    step,
)
from .mlx_ppo import (
    MLXPPOTrainer,
    MLXRolloutBatch,
    MLXRolloutCollector,
    MLXRolloutState,
)
from .ppo import ActorCritic, PPOConfig, PPOTrainer

__all__ = [
    "ActorCritic",
    "ContactEvidence",
    "FrankaEnv",
    "MetalRoboError",
    "MLXCompiledWorld",
    "MetalWorldCapacityProfile",
    "MLXPPOTrainer",
    "MLXRolloutBatch",
    "MLXRolloutCollector",
    "MLXRolloutState",
    "NativeRuntime",
    "PPOConfig",
    "PPOTrainer",
    "RuntimeStats",
    "SceneBodyState",
    "SolverCache",
    "StepOutput",
    "WorldState",
    "compile_world",
    "initial_state",
    "library_version",
    "step",
]

__version__ = "0.4.0"
