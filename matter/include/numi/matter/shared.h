#pragma once

// Versioned, pointer-free ABI shared by C++, Objective-C++, and Metal.
// Every offset and stride is measured in elements. Runtime storage is fixed
// after cooking; kernels never allocate, append through unordered atomics, or
// require a CPU-visible count to continue a step.

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#define NM_ALIGN16 alignas(16)
typedef uint nm_u32;
typedef int nm_i32;
typedef ulong nm_u64;
typedef float4 nm_float4;
typedef uint4 nm_uint4;
#else
#include <cstdint>
#define NM_ALIGN16 alignas(16)
typedef std::uint32_t nm_u32;
typedef std::int32_t nm_i32;
typedef std::uint64_t nm_u64;
typedef struct NM_ALIGN16 nm_float4 {
    float x;
    float y;
    float z;
    float w;
} nm_float4;
typedef struct NM_ALIGN16 nm_uint4 {
    nm_u32 x;
    nm_u32 y;
    nm_u32 z;
    nm_u32 w;
} nm_uint4;
#endif

#define NM_MATTER_ABI_VERSION 1u
#define NM_INVALID_INDEX 0xffffffffu
#define NM_SIMD_WIDTH 32u
#define NM_MAX_EXPRESSION_STACK 96u
#define NM_MAX_MATERIAL_PARAMETERS 32u
#define NM_MPM_STENCIL_WIDTH 27u
#define NM_TET_NODE_COUNT 4u
#define NM_EVENT_CLASS_COUNT 8u
#define NM_MAX_RATE_EXPONENT 8u
#define NM_MAX_PCG_ITERATIONS 1024u

#define NM_ENUM(name) enum name : nm_u32

NM_ENUM(NMRepresentationKind) {
    NM_REPRESENTATION_RIGID = 0u,
    NM_REPRESENTATION_MPM = 1u,
    NM_REPRESENTATION_FEM = 2u,
    NM_REPRESENTATION_ROD = 3u,
    NM_REPRESENTATION_SURFACE = 4u,
};

NM_ENUM(NMConstitutiveModelKind) {
    NM_CONSTITUTIVE_BYTECODE = 0u,
    NM_CONSTITUTIVE_NEO_HOOKEAN = 1u,
    NM_CONSTITUTIVE_COROTATED = 2u,
    NM_CONSTITUTIVE_HENCKY = 3u,
    NM_CONSTITUTIVE_DRUCKER_PRAGER = 4u,
    NM_CONSTITUTIVE_VON_MISES = 5u,
    NM_CONSTITUTIVE_NEWTONIAN = 6u,
    NM_CONSTITUTIVE_VISCO_HYPERELASTIC = 7u,
};

NM_ENUM(NMExpressionOpcode) {
    NM_EXPR_CONSTANT = 0u,
    NM_EXPR_PARAMETER = 1u,
    NM_EXPR_STATE = 2u,
    NM_EXPR_F = 3u,
    NM_EXPR_DF = 4u,
    NM_EXPR_RATE = 5u,
    NM_EXPR_DRATE = 6u,
    NM_EXPR_ADD = 7u,
    NM_EXPR_SUBTRACT = 8u,
    NM_EXPR_MULTIPLY = 9u,
    NM_EXPR_DIVIDE = 10u,
    NM_EXPR_NEGATE = 11u,
    NM_EXPR_LOG = 12u,
    NM_EXPR_EXP = 13u,
    NM_EXPR_SQRT = 14u,
    NM_EXPR_ABS = 15u,
    NM_EXPR_MIN = 16u,
    NM_EXPR_MAX = 17u,
    NM_EXPR_POW_INTEGER = 18u,
    NM_EXPR_CLAMP = 19u,
};

NM_ENUM(NMMatterStatusCode) {
    NM_STATUS_SUCCESS = 0u,
    NM_STATUS_INVALID_DISPATCH = 1u,
    NM_STATUS_CAPACITY_OVERFLOW = 2u,
    NM_STATUS_NONFINITE_INPUT = 3u,
    NM_STATUS_NONFINITE_RESULT = 4u,
    NM_STATUS_INVALID_DEFORMATION = 5u,
    NM_STATUS_CONTACT_FAILURE = 6u,
    NM_STATUS_LINEAR_SOLVER_FAILURE = 7u,
    NM_STATUS_EVENT_CAPACITY = 8u,
    NM_STATUS_UNSUPPORTED = 9u,
};

NM_ENUM(NMMatterFlags) {
    NM_MATTER_DETERMINISTIC = 1u << 0u,
    NM_MATTER_ENABLE_CONTACT = 1u << 1u,
    NM_MATTER_ENABLE_ADAPTIVE = 1u << 2u,
    NM_MATTER_ENABLE_IDENTIFICATION = 1u << 3u,
    NM_MATTER_CAPTURE_DIAGNOSTICS = 1u << 4u,
};

NM_ENUM(NMObjectFlags) {
    NM_OBJECT_ACTIVE = 1u << 0u,
    NM_OBJECT_DYNAMIC = 1u << 1u,
    NM_OBJECT_KINEMATIC = 1u << 2u,
    NM_OBJECT_TWO_WAY_COUPLED = 1u << 3u,
    NM_OBJECT_ADAPTIVE = 1u << 4u,
    NM_OBJECT_IDENTIFIABLE = 1u << 5u,
};

NM_ENUM(NMMaterialFlags) {
    NM_MATERIAL_HAS_DISSIPATION = 1u << 0u,
    NM_MATERIAL_FAST_CONSTITUTIVE = 1u << 1u,
};

NM_ENUM(NMRigidProxyFlags) {
    NM_RIGID_PROXY_ARTICULATED = 1u << 0u,
    NM_RIGID_PROXY_DYNAMIC = 1u << 1u,
    NM_RIGID_PROXY_KINEMATIC = 1u << 2u,
};

NM_ENUM(NMRigidShapeKind) {
    NM_RIGID_SHAPE_PLANE = 0u,
    NM_RIGID_SHAPE_SPHERE = 1u,
    NM_RIGID_SHAPE_CAPSULE = 2u,
    NM_RIGID_SHAPE_BOX = 3u,
    NM_RIGID_SHAPE_SDF = 4u,
};

NM_ENUM(NMContactFlags) {
    NM_CONTACT_VALID = 1u << 0u,
    NM_CONTACT_STICKING = 1u << 1u,
    NM_CONTACT_NEW = 1u << 2u,
    NM_CONTACT_ADHESIVE = 1u << 3u,
};

NM_ENUM(NMAdaptiveMode) {
    NM_ADAPTIVE_CONTINUUM = 0u,
    NM_ADAPTIVE_RIGID = 1u,
    NM_ADAPTIVE_PROMOTING = 2u,
    NM_ADAPTIVE_DEMOTING = 3u,
};

NM_ENUM(NMAdaptiveTriggerFlags) {
    NM_ADAPTIVE_TRIGGER_VALID = 1u << 0u,
    NM_ADAPTIVE_TRIGGER_FORCE_CONTINUUM = 1u << 1u,
    NM_ADAPTIVE_TRIGGER_FORCE_RIGID = 1u << 2u,
};

NM_ENUM(NMEventClass) {
    NM_EVENT_CONTACT_ONSET = 0u,
    NM_EVENT_CONTACT_RELEASE = 1u,
    NM_EVENT_SLIP_ONSET = 2u,
    NM_EVENT_YIELD_ONSET = 3u,
    NM_EVENT_DAMAGE_ONSET = 4u,
    NM_EVENT_INVERSION_RISK = 5u,
    NM_EVENT_SOLVER_RESIDUAL = 6u,
    NM_EVENT_RATE_CHANGE = 7u,
};

NM_ENUM(NMIdentificationFlags) {
    NM_IDENTIFICATION_LOG_SPACE = 1u << 0u,
    NM_IDENTIFICATION_ENABLED = 1u << 1u,
};

typedef struct NM_ALIGN16 NMMatterDispatchGPU {
    nm_u32 abiVersion;
    nm_u32 flags;
    nm_u32 environmentCount;
    nm_u32 objectCount;

    nm_u32 materialCount;
    nm_u32 parameterCount;
    nm_u32 particleCount;
    nm_u32 gridNodeCount;

    nm_u32 femNodeCount;
    nm_u32 tetCount;
    nm_u32 rigidProxyCount;
    nm_u32 contactPairCount;

    nm_u32 maximumRateExponent;
    nm_u32 femPCGIterations;
    nm_u32 identificationCandidateCount;
    nm_u32 eventStride;

    // xyz gravity, w frame/control timestep.
    nm_float4 gravityAndTimestep;
    // contact slop, max depenetration speed, determinant floor, finite limit.
    nm_float4 numericalLimits;
} NMMatterDispatchGPU;

typedef struct NM_ALIGN16 NMMicrostepGPU {
    nm_u32 controlStep;
    nm_u32 microtick;
    nm_u32 microtickCount;
    nm_u32 pcgIteration;

    nm_u32 seedLo;
    nm_u32 seedHi;
    nm_u32 runIdentification;
    nm_u32 runAdaptiveTransfer;

    // x dt, y 1/dt, z frame-relative time, w reserved.
    nm_float4 time;
} NMMicrostepGPU;

typedef struct NM_ALIGN16 NMBridgeDispatchGPU {
    nm_u32 environmentCount;
    nm_u32 rigidProxyCount;
    nm_u32 currentBodyCount;
    nm_u32 currentBodyStride;

    nm_u32 articulatedBodyCount;
    nm_u32 articulatedStride;
    nm_u32 sceneBodyCount;
    nm_u32 sceneStride;

    nm_u32 reactionStride;
    nm_u32 flags;
    nm_u32 reserved0;
    nm_u32 reserved1;

    // x inverse dt (impulse->force), y dt, zw reserved.
    nm_float4 time;
} NMBridgeDispatchGPU;

typedef struct NM_ALIGN16 NMExpressionInstructionGPU {
    nm_u32 opcode;
    nm_u32 index;
    nm_i32 integer;
    nm_u32 reserved;
    // x scalar literal; yzw reserved.
    nm_float4 literal;
} NMExpressionInstructionGPU;

typedef struct NM_ALIGN16 NMScalarProgramGPU {
    nm_u32 firstInstruction;
    nm_u32 instructionCount;
    nm_u32 maximumStack;
    nm_u32 flags;
} NMScalarProgramGPU;

typedef struct NM_ALIGN16 NMParameterRangeGPU {
    // current/default, lower, upper, proposal sigma.
    nm_float4 valueAndBounds;
    // dimension L/M/T/temperature encoded as signed bytes cast to uint32.
    nm_uint4 dimension;
    // global parameter index, material-local index, flags, reserved.
    nm_uint4 identity;
} NMParameterRangeGPU;

typedef struct NM_ALIGN16 NMMaterialGPU {
    nm_u32 modelKind;
    nm_u32 flags;
    nm_u32 firstParameter;
    nm_u32 parameterCount;

    nm_u32 firstState;
    nm_u32 stateCount;
    nm_u32 validityProgram;
    nm_u32 reserved0;

    // Nine scalar programs in row-major order for P=dPsi/dF.
    nm_uint4 stressPrograms0;
    nm_uint4 stressPrograms1;
    nm_uint4 stressProgram8AndPadding;
    // Nine scalar programs for dP(F)[dF].
    nm_uint4 tangentPrograms0;
    nm_uint4 tangentPrograms1;
    nm_uint4 tangentProgram8AndPadding;
    // Dissipative stress dPhi/dD and its directional derivative.
    nm_uint4 dissipativeStressPrograms0;
    nm_uint4 dissipativeStressPrograms1;
    nm_uint4 dissipativeStressProgram8AndPadding;
    nm_uint4 dissipativeTangentPrograms0;
    nm_uint4 dissipativeTangentPrograms1;
    nm_uint4 dissipativeTangentProgram8AndPadding;

    // static friction, dynamic friction, restitution, adhesion stress.
    nm_float4 interfaceResponse;
    // minimum J, maximum J, maximum stress, maximum energy density.
    nm_float4 validityLimits;
    // Fast-path parameter indices: mu, lambda, viscosity, density.
    nm_uint4 fastParameters;
} NMMaterialGPU;

typedef struct NM_ALIGN16 NMContinuumObjectGPU {
    nm_u32 representation;
    nm_u32 materialIndex;
    nm_u32 flags;
    nm_u32 rigidBinding;

    nm_u32 firstParticle;
    nm_u32 particleCount;
    nm_u32 firstGridNode;
    nm_u32 gridNodeCount;

    nm_u32 firstFEMNode;
    nm_u32 femNodeCount;
    nm_u32 firstTet;
    nm_u32 tetCount;

    // characteristic length, deformation tolerance, promote strain, demote strain.
    nm_float4 adaptation;
    // reference center of mass xyz, reference total mass.
    nm_float4 referenceCenterAndMass;
    // reference inverse inertia rows about COM.
    nm_float4 inverseInertiaRow0;
    nm_float4 inverseInertiaRow1;
    nm_float4 inverseInertiaRow2;
} NMContinuumObjectGPU;

typedef struct NM_ALIGN16 NMParticleStateGPU {
    // world position xyz, mass.
    nm_float4 positionAndMass;
    // world velocity xyz, reference volume.
    nm_float4 velocityAndVolume;
    // Deformation gradient F rows, row-major xyz; w reserved.
    nm_float4 F0;
    nm_float4 F1;
    nm_float4 F2;
    // APIC affine velocity matrix rows.
    nm_float4 C0;
    nm_float4 C1;
    nm_float4 C2;
    // Reference position xyz, object index.
    nm_float4 referenceAndObject;
} NMParticleStateGPU;

typedef struct NM_ALIGN16 NMGridNodeStateGPU {
    // Reference/world node position xyz, mass.
    nm_float4 positionAndMass;
    // Velocity xyz, accumulated temperature/unused.
    nm_float4 velocityAndAux;
    // Force xyz, reserved.
    nm_float4 force;
} NMGridNodeStateGPU;

typedef struct NM_ALIGN16 NMMPMStencilGPU {
    nm_u32 particleIndex;
    nm_u32 nodeIndex;
    nm_u32 objectIndex;
    nm_u32 localSlot;
    // x shape weight, yzw reference gradient.
    nm_float4 weightAndGradient;
} NMMPMStencilGPU;

typedef struct NM_ALIGN16 NMFEMNodeStateGPU {
    // world position xyz, mass.
    nm_float4 positionAndMass;
    // velocity xyz, inverse mass.
    nm_float4 velocityAndInverseMass;
    // reference position xyz, object index.
    nm_float4 referenceAndObject;
} NMFEMNodeStateGPU;

typedef struct NM_ALIGN16 NMTetrahedronGPU {
    nm_uint4 nodes;
    // inverse rest matrix rows.
    nm_float4 inverseRest0;
    nm_float4 inverseRest1;
    nm_float4 inverseRest2;
    // x rest volume, y object index, z material index, w reserved.
    nm_float4 volumeAndIdentity;
} NMTetrahedronGPU;

typedef struct NM_ALIGN16 NMFEMElementVectorGPU {
    nm_float4 node0;
    nm_float4 node1;
    nm_float4 node2;
    nm_float4 node3;
} NMFEMElementVectorGPU;

typedef struct NM_ALIGN16 NMIncidenceRangeGPU {
    nm_u32 first;
    nm_u32 count;
    nm_u32 owner;
    nm_u32 flags;
} NMIncidenceRangeGPU;

typedef struct NM_ALIGN16 NMPCGScalarGPU {
    // rr, pAp, alpha, beta.
    nm_float4 values;
    // initial rr, relative residual, iteration, flags.
    nm_float4 diagnostics;
} NMPCGScalarGPU;

typedef struct NM_ALIGN16 NMRigidProxyGPU {
    nm_u32 shapeKind;
    nm_u32 bodyIndex;
    nm_u32 materialIndex;
    nm_u32 flags;

    // World center or plane normal xyz; plane offset/radius in w.
    nm_float4 centerAndRadius;
    // World orientation xyzw.
    nm_float4 orientation;
    // Box half extents or capsule local half-axis xyz; contact offset in w.
    nm_float4 extentAndOffset;
    // Linear velocity xyz, inverse mass.
    nm_float4 linearVelocityAndInverseMass;
    // Angular velocity xyz, reserved.
    nm_float4 angularVelocity;
    // World inverse inertia rows.
    nm_float4 inverseInertiaRow0;
    nm_float4 inverseInertiaRow1;
    nm_float4 inverseInertiaRow2;
} NMRigidProxyGPU;

typedef struct NM_ALIGN16 NMContactPairGPU {
    nm_u32 objectIndex;
    nm_u32 continuumIndex;
    nm_u32 rigidProxyIndex;
    nm_u32 representation;
} NMContactPairGPU;

typedef struct NM_ALIGN16 NMContactSampleGPU {
    // continuum index, rigid proxy, object, flags.
    nm_uint4 identity;
    // contact point xyz, signed separation.
    nm_float4 pointAndSeparation;
    // normal from rigid toward continuum xyz, pre-solve normal velocity.
    nm_float4 normalAndVelocity;
    // impulse on continuum xyz, normal impulse magnitude.
    nm_float4 impulse;
    // tangential relative velocity xyz, slip speed.
    nm_float4 tangentAndSlip;
} NMContactSampleGPU;

typedef struct NM_ALIGN16 NMRigidReactionGPU {
    // Equal/opposite impulse on rigid body xyz, count in w.
    nm_float4 impulseAndCount;
    // Angular impulse about body COM xyz, reserved.
    nm_float4 angularImpulse;
} NMRigidReactionGPU;

typedef struct NM_ALIGN16 NMAdaptiveStateGPU {
    nm_u32 mode;
    nm_u32 previousMode;
    nm_u32 stableFrames;
    nm_u32 transitionCount;

    // measured strain, residual, max stress, minimum J.
    nm_float4 measures;
    // continuum COM xyz, total mass.
    nm_float4 centerAndMass;
    // linear velocity xyz, deformation sample weight.
    nm_float4 linearVelocityAndWeight;
    // angular velocity xyz, reserved.
    nm_float4 angularVelocity;
    // fitted orientation xyzw.
    nm_float4 orientation;
} NMAdaptiveStateGPU;

typedef struct NM_ALIGN16 NMAdaptiveTriggerGPU {
    nm_u32 flags;
    nm_u32 objectIndex;
    nm_u32 reserved0;
    nm_u32 reserved1;
    // contact impulse, closing speed, equivalent stress, damage.
    nm_float4 signals;
} NMAdaptiveTriggerGPU;

typedef struct NM_ALIGN16 NMSchedulerStateGPU {
    nm_u32 rateExponent;
    nm_u32 previousRateExponent;
    nm_u32 highActivityFrames;
    nm_u32 lowActivityFrames;

    nm_u32 previousContactCount;
    nm_u32 eventCursor;
    nm_u32 activeThisMicrotick;
    nm_u32 flags;

    // max slip, max stress, min J, solver residual.
    nm_float4 signals;
} NMSchedulerStateGPU;

typedef struct NM_ALIGN16 NMEventTokenGPU {
    nm_u32 eventClass;
    nm_u32 objectIndex;
    nm_u32 controlStep;
    nm_u32 microtick;
    // severity, delta time, primary value, secondary value.
    nm_float4 values;
} NMEventTokenGPU;

typedef struct NM_ALIGN16 NMIdentificationDistributionGPU {
    nm_u32 parameterIndex;
    nm_u32 flags;
    nm_u32 generation;
    nm_u32 reserved;
    // mean, log standard deviation, lower, upper.
    nm_float4 distribution;
    // learning rate, sigma floor, sigma ceiling, baseline momentum.
    nm_float4 optimizer;
} NMIdentificationDistributionGPU;

typedef struct NM_ALIGN16 NMIdentificationCandidateGPU {
    nm_u32 parameterIndex;
    nm_u32 candidateIndex;
    nm_u32 generation;
    nm_u32 flags;
    // candidate value, normalized perturbation, loss, weight.
    nm_float4 values;
} NMIdentificationCandidateGPU;

typedef struct NM_ALIGN16 NMIdentificationPassGPU {
    nm_u32 candidateCount;
    nm_u32 distributionCount;
    nm_u32 generation;
    nm_u32 flags;
    nm_u32 seedLo;
    nm_u32 seedHi;
    nm_u32 environmentParameterStride;
    nm_u32 reserved;
} NMIdentificationPassGPU;

typedef struct NM_ALIGN16 NMMatterStatusGPU {
    nm_u32 code;
    nm_u32 environment;
    nm_u32 objectIndex;
    nm_u32 failingIndex;

    nm_u32 completedMicrosteps;
    nm_u32 contactCount;
    nm_u32 eventCount;
    nm_u32 pcgIterations;

    // max stress, min J, PCG residual, max penetration.
    nm_float4 diagnostics;
} NMMatterStatusGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(nm_float4) == 16);
static_assert(sizeof(nm_uint4) == 16);
static_assert(sizeof(NMMatterDispatchGPU) == 96);
static_assert(sizeof(NMMicrostepGPU) == 48);
static_assert(sizeof(NMBridgeDispatchGPU) == 64);
static_assert(sizeof(NMExpressionInstructionGPU) == 32);
static_assert(sizeof(NMScalarProgramGPU) == 16);
static_assert(sizeof(NMParameterRangeGPU) == 48);
static_assert(sizeof(NMContinuumObjectGPU) == 128);
static_assert(sizeof(NMParticleStateGPU) == 144);
static_assert(sizeof(NMGridNodeStateGPU) == 48);
static_assert(sizeof(NMMPMStencilGPU) == 32);
static_assert(sizeof(NMFEMNodeStateGPU) == 48);
static_assert(sizeof(NMTetrahedronGPU) == 80);
static_assert(sizeof(NMFEMElementVectorGPU) == 64);
static_assert(sizeof(NMIncidenceRangeGPU) == 16);
static_assert(sizeof(NMPCGScalarGPU) == 32);
static_assert(sizeof(NMRigidProxyGPU) == 144);
static_assert(sizeof(NMContactPairGPU) == 16);
static_assert(sizeof(NMContactSampleGPU) == 80);
static_assert(sizeof(NMRigidReactionGPU) == 32);
static_assert(sizeof(NMAdaptiveStateGPU) == 96);
static_assert(sizeof(NMAdaptiveTriggerGPU) == 32);
static_assert(sizeof(NMSchedulerStateGPU) == 48);
static_assert(sizeof(NMEventTokenGPU) == 32);
static_assert(sizeof(NMIdentificationDistributionGPU) == 48);
static_assert(sizeof(NMIdentificationCandidateGPU) == 32);
static_assert(sizeof(NMIdentificationPassGPU) == 32);
static_assert(sizeof(NMMatterStatusGPU) == 48);
#endif
