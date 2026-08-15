#pragma once

#include "metalrobo/engine_types.h"

#define MR_ROD_GPU_ABI_VERSION 10u
#define MR_ROD_GPU_MAX_NODES 128u
#define MR_ROD_GPU_MAX_ATTACHMENTS 8u
#define MR_ROD_GPU_MAX_SELF_FRICTION_CONTACTS 64u
#define MR_ROD_GPU_SELF_FRICTION_ITERATIONS 8u
#define MR_ROD_GPU_MAX_SELF_CONTACT_PAIRS \
    (((MR_ROD_GPU_MAX_NODES - 2u) * \
      (MR_ROD_GPU_MAX_NODES - 3u)) / 2u)
#define MR_ROD_GPU_SELF_CONTACT_PAIR_WORDS \
    ((MR_ROD_GPU_MAX_SELF_CONTACT_PAIRS + 31u) / 32u)
#define MR_ROD_GPU_TOOL_WITNESSES_PER_PAIR 4u
#define MR_ROD_ACTIVE_GLOBAL_METADATA_WORDS 4u
#define MR_ROD_ACTIVE_PER_ROD_METADATA_WORDS 5u
#define MR_ROD_GPU_INVALID_BODY 0xffffffffu
#define MR_ROD_GPU_TWIST_CORRECTION_INDEX_BASE 128u

enum {
    MR_ROD_GPU_SUCCESS = 0u,
    MR_ROD_GPU_INVALID_DISPATCH = 1u,
    MR_ROD_GPU_DEGENERATE_GEOMETRY = 2u,
    MR_ROD_GPU_NONFINITE_RESULT = 3u,
    MR_ROD_GPU_DID_NOT_CONVERGE = 4u,
    MR_ROD_GPU_SELF_FRICTION_CAPACITY_OVERFLOW = 5u,
};

enum {
    MR_ROD_GPU_FLAG_SELF_COLLISION = 1u << 0u,
    MR_ROD_GPU_FLAG_TOOL_COLLISION = 1u << 1u,
    MR_ROD_GPU_FLAG_ENABLE_CCD = 1u << 2u,
    MR_ROD_GPU_FLAG_TOOL_WARM_START = 1u << 3u,
};

// Immutable procedural capsule generated for one rod edge. The world compiler
// appends these after rigid colliders and cooks stable edge/tool templates;
// projection replaces the capsule endpoints from candidate rod nodes at each
// temporal or event-time relinearization.
typedef struct MR_ALIGN16 MRRodColliderGPU {
    mr_u32 rodIndex;
    mr_u32 edgeIndex;
    mr_u32 nodeA;
    mr_u32 nodeB;

    mr_u32 materialIndex;
    mr_u32 collisionGroup;
    mr_u32 collisionMask;
    mr_u32 topologyGeneration;

    // radius, contact offset, rest offset, conservative bounding radius.
    mr_float4 radiusAndOffsets;
    // flags, typed dynamic-node index, exclusion count, reserved.
    mr_uint4 flagsAndExclusions;
} MRRodColliderGPU;

typedef struct MR_ALIGN16 MRRodToolPairGPU {
    mr_u32 rodCollider;
    mr_u32 rigidCollider;
    mr_u32 pairClass;
    mr_u32 flags;
} MRRodToolPairGPU;

enum MRRodToolPairFlags : mr_u32 {
    MR_ROD_TOOL_PAIR_VALID = 1u << 0u,
    MR_ROD_TOOL_PAIR_ENABLE_CCD = 1u << 1u,
};

enum MRRodToolWitnessFlags : mr_u32 {
    MR_ROD_TOOL_WITNESS_VALID = 1u << 0u,
    MR_ROD_TOOL_WITNESS_WARM_STARTED = 1u << 1u,
    MR_ROD_TOOL_WITNESS_NEW_IMPACT = 1u << 2u,
    MR_ROD_TOOL_WITNESS_MESH = 1u << 3u,
    MR_ROD_TOOL_WITNESS_HARD_CONVEX = 1u << 4u,
};

// Pair-owned thread/tool witness and impulse cache. One immutable pair owns
// four canonical feature-ordered slots in every environment. The rod anchor
// is edge-local (barycentric coordinate plus transported radial direction);
// the tool anchor remains body local so both sides survive rigid motion.
typedef struct MR_ALIGN16 MRRodToolWitnessGPU {
    // environment, stable pair, rod edge, rigid collider.
    mr_uint4 identity;
    // rod feature, rigid feature, rigid body, flags.
    mr_uint4 featuresAndFlags;
    // rod material, rigid/triangle material, rod topology generation,
    // reserved. The material override is resolved during narrowphase.
    mr_uint4 materialAndGeneration;

    // xyz rod surface point in world, w edge barycentric coordinate.
    mr_float4 rodPointAndWeight;
    // xyz rigid surface point in world, w signed surface separation.
    mr_float4 toolPointAndSeparation;
    // xyz normal from rod toward rigid tool, w pre-solve normal velocity.
    mr_float4 normalAndPreSolveVelocity;
    // xyz first tangent, w rod twist Jacobian along that tangent.
    mr_float4 tangentUAndTwistJacobian;
    // xyz rod radial direction in the transported contact frame, w rod
    // twist Jacobian along tangent-v (tangent-v = normal x tangent-u).
    mr_float4 radialAndTwistJacobianV;
    // normal, tangent-u, tangent-v, torsional impulse.
    mr_float4 impulses;
} MRRodToolWitnessGPU;

typedef struct MR_ALIGN16 MRRodNodeStateGPU {
    mr_float4 position;
    mr_float4 velocity;
} MRRodNodeStateGPU;

typedef struct MR_ALIGN16 MRRodEdgeStateGPU {
    // x material-frame twist, y twist rate, z/w reserved.
    mr_float4 twistAndRate;
} MRRodEdgeStateGPU;

// Reusable implicit free-motion factor owned by one connected rod component.
// Numerical blocks live in stream-local private arenas; this record is the
// stable descriptor and diagnostic sidecar consumed by temporal and quality.
typedef struct MR_ALIGN16 MRRodFactorCacheGPU {
    mr_u32 environment;
    mr_u32 rodIndex;
    mr_u32 velocityOffset;
    mr_u32 velocityCount;

    mr_u32 firstBlock;
    mr_u32 blockCount;
    mr_u32 blockWidth;
    mr_u32 generation;

    mr_u32 code;
    mr_u32 failingElement;
    mr_u32 projectedCurvatureCount;
    mr_u32 flags;

    // Minimum pivot, maximum pivot, relative factor residual, and maximum
    // projected negative curvature.
    mr_float4 diagnostics;
} MRRodFactorCacheGPU;

enum MRRodFactorCacheFlags : mr_u32 {
    MR_ROD_FACTOR_CACHE_VALID = 1u << 0u,
    MR_ROD_FACTOR_CACHE_TRANSLATION_BAND = 1u << 1u,
    MR_ROD_FACTOR_CACHE_TWIST_BAND = 1u << 2u,
    MR_ROD_FACTOR_CACHE_PROJECTED_CURVATURE = 1u << 3u,
    MR_ROD_FACTOR_CACHE_SELECTED_INVERSE = 1u << 4u,
};

// Translation uses a scalar lower Cholesky band with half-width eight. This
// exactly covers the three-node bend stencil under xyz-interleaved node
// ordering. Twist is a scalar bidiagonal Cholesky factor.
#define MR_ROD_FACTOR_TRANSLATION_BAND_WIDTH 9u
#define MR_ROD_FACTOR_TRANSLATION_FLOATS_PER_NODE 27u
#define MR_ROD_FACTOR_TWIST_FLOATS_PER_EDGE 2u

typedef struct MR_ALIGN16 MRRodGPUDispatch {
    mr_u32 abiVersion;
    mr_u32 environmentCount;
    mr_u32 nodeCount;
    mr_u32 edgeCount;

    mr_u32 attachmentCount;
    mr_u32 solverIterations;
    mr_u32 stateNodeStride;
    mr_u32 stateEdgeStride;

    mr_u32 rigidBodyCount;
    mr_u32 stateBodyStride;
    mr_u32 flags;
    mr_u32 toolShapeCount;

    mr_u32 toolPairCount;
    mr_u32 toolContactStride;
    mr_u32 toolContactIterations;
    mr_u32 rodMaterialIndex;

    // Flattened heterogeneous-world bases. Standalone rod programs use zero
    // bases and toolPairWorldStride == toolPairCount. Persistent worlds bind
    // global rod/node/edge/pair arenas without pointer rebasing or host-side
    // staging between rods.
    mr_u32 rodNodeBase;
    mr_u32 rodEdgeBase;
    mr_u32 toolPairBase;
    mr_u32 toolPairWorldStride;

    // xyz gravity, w timestep.
    mr_float4 gravityAndTimestep;
    // linear damping, twist damping, derivative step, tolerance.
    mr_float4 dampingDerivativeTolerance;
    // radius, margin, compliance, Coulomb self-contact friction.
    mr_float4 selfCollision;
    // rod contact offset, rest offset, normal compliance, damping.
    mr_float4 toolContact;
    // restitution, threshold, friction scale, maximum depenetration speed.
    mr_float4 toolResponse;
} MRRodGPUDispatch;

typedef struct MR_ALIGN16 MRRodGPUAttachment {
    // xyz target, w compliance.
    mr_float4 targetAndCompliance;
    // xyz velocity, w must be zero.
    mr_float4 velocity;
    mr_u32 nodeIndex;
    mr_u32 reserved0;
    mr_u32 reserved1;
    mr_u32 reserved2;
} MRRodGPUAttachment;

typedef struct MR_ALIGN16 MRRodGPURigidBinding {
    // xyz local anchor, w must be zero.
    mr_float4 localAnchor;
    mr_u32 bodyIndex;
    mr_u32 reserved0;
    mr_u32 reserved1;
    mr_u32 reserved2;
} MRRodGPURigidBinding;

typedef struct MR_ALIGN16 MRRodGPUAttachmentReaction {
    // xyz impulse exerted by the rod on the target, w final position error.
    mr_float4 impulseAndError;
    // xyz step-average force exerted by the rod on the target.
    mr_float4 averageForce;
    mr_u32 nodeIndex;
    mr_u32 attachmentIndex;
    mr_u32 bodyIndex;
    mr_u32 reserved;
} MRRodGPUAttachmentReaction;

typedef struct MR_ALIGN16 MRRodGPUStatus {
    mr_u32 code;
    mr_u32 environment;
    mr_u32 iterations;
    // For nonconvergence, node indices are [0, 127] and twist-edge indices
    // are MR_ROD_GPU_TWIST_CORRECTION_INDEX_BASE + edge. Other failures use
    // their owning element when available or MR_ROD_GPU_INVALID_BODY.
    mr_u32 failingIndex;

    // Maximum constraint error, maximum centerline/radius-scaled surface
    // correction in metres, maximum self penetration, projected
    // self-contact count.
    mr_float4 diagnostics;
} MRRodGPUStatus;

#ifdef __cplusplus
static_assert(sizeof(MRRodGPUDispatch) == 160);
static_assert(alignof(MRRodGPUDispatch) == 16);
static_assert(sizeof(MRRodGPUAttachment) == 48);
static_assert(alignof(MRRodGPUAttachment) == 16);
static_assert(sizeof(MRRodGPURigidBinding) == 32);
static_assert(alignof(MRRodGPURigidBinding) == 16);
static_assert(sizeof(MRRodGPUAttachmentReaction) == 48);
static_assert(alignof(MRRodGPUAttachmentReaction) == 16);
static_assert(sizeof(MRRodGPUStatus) == 32);
static_assert(alignof(MRRodGPUStatus) == 16);
static_assert(sizeof(MRRodColliderGPU) == 64);
static_assert(alignof(MRRodColliderGPU) == 16);
static_assert(sizeof(MRRodToolPairGPU) == 16);
static_assert(sizeof(MRRodToolWitnessGPU) == 144);
static_assert(alignof(MRRodToolWitnessGPU) == 16);
static_assert(sizeof(MRRodNodeStateGPU) == 32);
static_assert(sizeof(MRRodEdgeStateGPU) == 16);
static_assert(sizeof(MRRodFactorCacheGPU) == 64);
static_assert(alignof(MRRodFactorCacheGPU) == 16);
#endif
