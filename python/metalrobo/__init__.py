"""MetalRobo's native Python and MLX reinforcement-learning interface."""

from .env import FrankaEnv
from .episode import compile_episode_manifest
from .hybrid_renderer import (
    GAUSSIAN_ASSET_LOCAL,
    GAUSSIAN_BODY_LOCAL,
    GAUSSIAN_WORLD,
    HYBRID_GAUSSIAN_DTYPE,
    HybridObservationDeviceBuffers,
    HybridObservationRenderer,
    HybridObservationSnapshot,
    HybridRendererLayout,
    make_asset_gaussians,
)
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
    initial_state_from_world_family,
    step,
)
from .worlds import FrankaPickPlaceWorldFamily, PackedWorldFamily
from .mlx_ppo import (
    MLXPPOTrainer,
    MLXRolloutBatch,
    MLXRolloutCollector,
    MLXRolloutState,
)
from .mlx_locomotion import (
    MLXG1PPOTrainer,
    MLXG1RolloutCollector,
)
from .mlx_surgical import (
    MLXPSMNeedlePPOTrainer,
    MLXPSMNeedleRolloutCollector,
    psm_physical_position_targets,
)
from .ppo import ActorCritic, PPOConfig, PPOTrainer

__all__ = [
    "ActorCritic",
    "ContactEvidence",
    "FrankaEnv",
    "FrankaPickPlaceWorldFamily",
    "GAUSSIAN_ASSET_LOCAL",
    "GAUSSIAN_BODY_LOCAL",
    "GAUSSIAN_WORLD",
    "HYBRID_GAUSSIAN_DTYPE",
    "HybridObservationDeviceBuffers",
    "HybridObservationRenderer",
    "HybridObservationSnapshot",
    "HybridRendererLayout",
    "MetalRoboError",
    "MLXCompiledWorld",
    "MLXG1PPOTrainer",
    "MLXG1RolloutCollector",
    "MLXPSMNeedlePPOTrainer",
    "MLXPSMNeedleRolloutCollector",
    "MetalWorldCapacityProfile",
    "MLXPPOTrainer",
    "MLXRolloutBatch",
    "MLXRolloutCollector",
    "MLXRolloutState",
    "NativeRuntime",
    "PPOConfig",
    "PPOTrainer",
    "PackedWorldFamily",
    "RuntimeStats",
    "SceneBodyState",
    "SolverCache",
    "StepOutput",
    "WorldState",
    "compile_world",
    "compile_episode_manifest",
    "initial_state",
    "initial_state_from_world_family",
    "library_version",
    "make_asset_gaussians",
    "psm_physical_position_targets",
    "step",
]

__version__ = "0.4.0"
