"""Compatibility surface for the production MLX ARDY hyper-policy compiler."""

from .hyperpolicy.mlx_production_model import (
    ARDYHyperNetwork, ARDYHyperPolicyConfiguration, ARDYMotionConditionedPolicy,
    ARDYMotionEncoder, HyperNetworkOutput, MotionAdapterDecoder,
    PhaseVaryingLowRankActor,
)
from .hyperpolicy.mlx_training import (
    HyperPolicyLossWeights, HyperPolicyMetaLearner, HyperPolicySupervisionBatch,
)
from .hyperpolicy.mlx_compiler import (
    ARDYHyperPolicyCompiler, GeneratedAdapterDistribution,
)
from .hyperpolicy.mlx_specialist import (
    MLXSpecialistAdapterLearner, SpecialistAdapterBatch,
    SpecialistAdapterProgram,
)
from .hyperpolicy.mlx_bridge import (
    export_hyper_base, initialize_hyper_base_from_policy_pack,
)

__all__ = [
    "ARDYHyperNetwork", "ARDYHyperPolicyCompiler",
    "ARDYHyperPolicyConfiguration", "ARDYMotionConditionedPolicy",
    "ARDYMotionEncoder", "GeneratedAdapterDistribution",
    "HyperNetworkOutput", "HyperPolicyLossWeights",
    "HyperPolicyMetaLearner", "HyperPolicySupervisionBatch",
    "MLXSpecialistAdapterLearner", "MotionAdapterDecoder",
    "PhaseVaryingLowRankActor", "SpecialistAdapterBatch",
    "SpecialistAdapterProgram",
    "export_hyper_base", "initialize_hyper_base_from_policy_pack",
]
