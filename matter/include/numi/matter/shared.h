#pragma once

// Pointer-free ABI shared by C++, Objective-C++, and Metal.
// Every offset and stride is measured in elements, never bytes.

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#define NM_ALIGN16 alignas(16)
typedef uint nm_u32;
typedef int nm_i32;
typedef ulong nm_u64;
typedef float4 nm_float4;
typedef uint4 nm_uint4;
typedef int4 nm_int4;
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
typedef struct NM_ALIGN16 nm_int4 {
    nm_i32 x;
    nm_i32 y;
    nm_i32 z;
    nm_i32 w;
} nm_int4;
#endif

#define NM_MATTER_ABI_VERSION 19u
#define NM_INVALID_INDEX 0xffffffffu
#define NM_EXPRESSION_STACK_CAPACITY 96u
#define NM_MPM_STENCIL_WIDTH 27u
#define NM_EVENT_CLASS_COUNT 8u
#define NM_MAX_RATE_EXPONENT 8u
#define NM_MPM_BLOCK_EDGE 8u
#define NM_MPM_BLOCK_NODE_COUNT 512u
#define NM_MPM_MAX_PARTICLES_PER_BLOCK 256u
#define NM_MAX_MATERIAL_STATE 16u
#define NM_MIXED_NEWTON_ITERATIONS 7u
#define NM_MIXED_FGMRES_RESTART 16u
#define NM_MIXED_FGMRES_ITERATIONS 48u
#define NM_MIXED_LINE_SEARCH_STEPS 8u
#define NM_MIXED_MUTATION_RESTARTS 4u
#define NM_LEARNED_MAX_LAYERS 8u
#define NM_LEARNED_MAX_WIDTH 16u
#define NM_LEARNED_MAX_INVARIANTS 8u
// Coupled-candidate storage is part of Matter's compiled ABI. Keep these
// capacities synchronized with the borrowed MetalWorld articulated operator;
// the bridge asserts equality at compile time.
#define NM_MATTER_MAX_ARTICULATED_DOFS 40u
#define NM_MATTER_MAX_ARTICULATED_Q 41u

enum NMRepresentationKind : nm_u32 {
    NM_REPRESENTATION_RIGID = 0u,
    NM_REPRESENTATION_MPM = 1u,
    NM_REPRESENTATION_FEM = 2u,
};

enum NMConstitutiveKind : nm_u32 {
    NM_CONSTITUTIVE_BYTECODE = 0u,
    NM_CONSTITUTIVE_NEO_HOOKEAN = 1u,
    NM_CONSTITUTIVE_COROTATED = 2u,
    NM_CONSTITUTIVE_DRUCKER_PRAGER = 3u,
    NM_CONSTITUTIVE_VON_MISES = 4u,
    NM_CONSTITUTIVE_VISCO_HYPERELASTIC = 5u,
    NM_CONSTITUTIVE_POLYCONVEX_ICNN = 6u,
};

enum NMExpressionOpcode : nm_u32 {
    NM_EXPR_CONSTANT = 0u,
    NM_EXPR_PARAMETER = 1u,
    NM_EXPR_F = 2u,
    NM_EXPR_DF = 3u,
    NM_EXPR_STATE = 4u,
    NM_EXPR_ADD = 5u,
    NM_EXPR_SUBTRACT = 6u,
    NM_EXPR_MULTIPLY = 7u,
    NM_EXPR_DIVIDE = 8u,
    NM_EXPR_NEGATE = 9u,
    NM_EXPR_LOG = 10u,
    NM_EXPR_EXP = 11u,
    NM_EXPR_SQRT = 12u,
    NM_EXPR_ABS = 13u,
    NM_EXPR_MIN = 14u,
    NM_EXPR_MAX = 15u,
    NM_EXPR_POW_INTEGER = 16u,
    NM_EXPR_CLAMP = 17u,
    NM_EXPR_DT = 18u,
    NM_EXPR_TEMPERATURE = 19u,
    NM_EXPR_RATE = 20u,
    NM_EXPR_NEXT_STATE = 21u,
};

enum NMStatusCode : nm_u32 {
    NM_STATUS_SUCCESS = 0u,
    NM_STATUS_INVALID_DISPATCH = 1u,
    NM_STATUS_CAPACITY_OVERFLOW = 2u,
    NM_STATUS_NONFINITE_INPUT = 3u,
    NM_STATUS_NONFINITE_RESULT = 4u,
    NM_STATUS_INVALID_DEFORMATION = 5u,
    NM_STATUS_CONTACT_FAILURE = 6u,
    NM_STATUS_LINEAR_SOLVER_FAILURE = 7u,
    NM_STATUS_UNSUPPORTED = 8u,
    NM_STATUS_RIGID_WORLD_FAILURE = 9u,
    NM_STATUS_NONLINEAR_SOLVER_FAILURE = 10u,
    NM_STATUS_MULTIPHYSICS_FAILURE = 11u,
    NM_STATUS_TOPOLOGY_FAILURE = 12u,
    NM_STATUS_LEARNED_MATERIAL_FAILURE = 13u,
    NM_STATUS_LOCAL_MATERIAL_FAILURE = 14u,
    NM_STATUS_TOPOLOGY_GROWTH_REQUIRED = 15u,
};

enum NMMaterialFlags : nm_u32 {
    NM_MATERIAL_HAS_STATE = 1u << 0u,
    NM_MATERIAL_HAS_DISSIPATION = 1u << 1u,
    NM_MATERIAL_HAS_IMPLICIT_STATE = 1u << 2u,
};

enum NMMaterialProjectionKind : nm_u32 {
    NM_MATERIAL_PROJECTION_GENERIC = 0u,
    NM_MATERIAL_PROJECTION_VON_MISES = 1u,
    NM_MATERIAL_PROJECTION_DRUCKER_PRAGER = 2u,
};

enum NMMaterialStateTransfer : nm_u32 {
    NM_MATERIAL_TRANSFER_AVERAGE = 0u,
    NM_MATERIAL_TRANSFER_MAXIMUM = 1u,
    NM_MATERIAL_TRANSFER_SUM = 2u,
};

enum NMMatterFlags : nm_u32 {
    NM_MATTER_DETERMINISTIC = 1u << 0u,
    NM_MATTER_CONTACT = 1u << 1u,
    NM_MATTER_ADAPTIVE = 1u << 2u,
    NM_MATTER_IDENTIFICATION = 1u << 3u,
    NM_MATTER_MIXED_FEM = 1u << 4u,
    NM_MATTER_MULTIPHYSICS = 1u << 5u,
    NM_MATTER_MUTATION = 1u << 6u,
    NM_MATTER_LEARNED_MATERIAL = 1u << 7u,
    NM_MATTER_IPC = 1u << 8u,
};

enum NMObjectFlags : nm_u32 {
    NM_OBJECT_ACTIVE = 1u << 0u,
    NM_OBJECT_TWO_WAY_COUPLED = 1u << 1u,
    NM_OBJECT_ADAPTIVE = 1u << 2u,
    NM_OBJECT_IDENTIFIABLE = 1u << 3u,
    NM_OBJECT_MIXED_FEM = 1u << 4u,
    NM_OBJECT_MULTIPHYSICS = 1u << 5u,
    NM_OBJECT_MUTABLE_TOPOLOGY = 1u << 6u,
};

enum NMFieldBoundaryFlags : nm_u32 {
    NM_FIELD_DIRICHLET_TEMPERATURE = 1u << 0u,
    NM_FIELD_DIRICHLET_PORE_PRESSURE = 1u << 1u,
    NM_FIELD_DIRICHLET_ELECTRIC_POTENTIAL = 1u << 2u,
    NM_FIELD_DIRICHLET_ACTIVATION = 1u << 3u,
    NM_FIELD_NEUMANN_TEMPERATURE = 1u << 4u,
    NM_FIELD_NEUMANN_PORE_PRESSURE = 1u << 5u,
    NM_FIELD_NEUMANN_ELECTRIC_CURRENT = 1u << 6u,
};

enum NMMutationKind : nm_u32 {
    NM_MUTATION_COHESIVE_SEPARATION = 0u,
    NM_MUTATION_PLANE_EROSION = 1u,
    NM_MUTATION_CYLINDER_PUNCTURE = 2u,
    NM_MUTATION_DEACTIVATE_TETRAHEDRON = 3u,
    NM_MUTATION_EDGE_SPLIT = 4u,
    NM_MUTATION_EDGE_COLLAPSE = 5u,
    NM_MUTATION_FACE_FLIP_23 = 6u,
    NM_MUTATION_FACE_FLIP_32 = 7u,
    NM_MUTATION_VERTEX_SMOOTH = 8u,
};

typedef struct NM_ALIGN16 NMTopologyGrowthRequestGPU {
    // Required marker, allocation generation, first requesting object, reason.
    nm_uint4 identity;
    // Requested global node, tetrahedron, cohesive-face, channel capacities.
    nm_uint4 topology;
    // Requested incidence, mutation, rigid-contact, deformable-contact arenas.
    nm_uint4 work;
} NMTopologyGrowthRequestGPU;

enum NMMutationFlags : nm_u32 {
    NM_MUTATION_ACTIVE = 1u << 0u,
    NM_MUTATION_PHYSICS_TRIGGERED = 1u << 1u,
};

enum NMTopologyFlags : nm_u32 {
    NM_TOPOLOGY_ACTIVE = 1u << 0u,
    NM_TOPOLOGY_SURFACE = 1u << 1u,
    NM_TOPOLOGY_COHESIVE = 1u << 2u,
    NM_TOPOLOGY_ERODED = 1u << 3u,
    NM_TOPOLOGY_SEPARATED = 1u << 4u,
};

enum NMLearnedActivationKind : nm_u32 {
    NM_LEARNED_SOFTPLUS = 0u,
};

enum NMRigidShapeKind : nm_u32 {
    NM_RIGID_PLANE = 0u,
    NM_RIGID_SPHERE = 1u,
    NM_RIGID_CAPSULE = 2u,
    NM_RIGID_BOX = 3u,
};

enum NMRigidBindingFlags : nm_u32 {
    NM_RIGID_ARTICULATED = 1u << 0u,
    NM_RIGID_DYNAMIC = 1u << 1u,
};

enum NMResetFlags : nm_u32 {
    NM_RESET_ENABLED = 1u << 0u,
    NM_RESET_PARAMETERS = 1u << 1u,
};

enum NMContactFlags : nm_u32 {
    NM_CONTACT_VALID = 1u << 0u,
    NM_CONTACT_STICKING = 1u << 1u,
    NM_CONTACT_NEW = 1u << 2u,
};

enum NMEventClass : nm_u32 {
    NM_EVENT_CONTACT_ONSET = 0u,
    NM_EVENT_CONTACT_RELEASE = 1u,
    NM_EVENT_SLIP_ONSET = 2u,
    NM_EVENT_YIELD_ONSET = 3u,
    NM_EVENT_DAMAGE_ONSET = 4u,
    NM_EVENT_INVERSION_RISK = 5u,
    NM_EVENT_SOLVER_RESIDUAL = 6u,
    NM_EVENT_RATE_CHANGE = 7u,
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
    nm_u32 tetrahedronCount;
    nm_u32 rigidProxyCount;
    nm_u32 contactPairCount;

    nm_u32 maximumRateExponent;
    nm_u32 femPCGIterations;
    nm_u32 identificationCandidateCount;
    nm_u32 eventStride;

    nm_u32 mpmGridCount;
    nm_u32 mpmBlockCount;
    nm_u32 mpmBlockLookupCount;
    nm_u32 maximumParticlesPerBlock;

    // Fixed scalar stride used by every particle/tetrahedron material-state
    // record and total authored state-initializer count.
    nm_u32 materialStateStride;
    nm_u32 stateInitialCount;
    nm_u32 mixedMaterialCount;

    nm_u32 fieldBoundaryCount;
    nm_u32 cohesiveFaceCount;
    nm_u32 mutationCommandCount;
    nm_u32 learnedMaterialCount;

    nm_u32 learnedLayerCount;
    nm_u32 learnedWeightCount;
    nm_u32 topologyNodeCapacity;
    nm_u32 topologyTetrahedronCapacity;

    nm_u32 punctureChannelCount;
    nm_u32 femCapacityCount;
    // Per-environment capacity of the compact active-MPM Krylov arena. The
    // physical grid and node-to-active map retain gridNodeCount stride.
    nm_u32 mpmActiveNodeCapacity;
    nm_u32 reservedMixed1;

    // Cooked source-face count (package/source audit), dynamic
    // deformable-contact capacity, conservative articulated/free-body v
    // capacity, and conservative articulated q capacity. Runtime surface
    // primitives are derived from all four faces of each active tetrahedron.
    nm_u32 surfaceFaceCount;
    nm_u32 deformableContactCapacity;
    nm_u32 rigidGeneralizedCapacity;
    nm_u32 rigidQCapacity;

    // xyz gravity, w frame timestep.
    nm_float4 gravityAndTimestep;
    // contact slop, maximum depenetration speed, determinant floor, finite limit.
    nm_float4 numericalLimits;
} NMMatterDispatchGPU;

typedef struct NM_ALIGN16 NMMixedSolverGPU {
    // Newton, FGMRES restart, total FGMRES, determinant backtracking.
    nm_uint4 nonlinearIterations;
    // Velocity PCG, pressure Schur PCG, field PCG, mutation restarts.
    nm_uint4 blockIterations;
    // relative residual, relative correction, volume, pressure.
    nm_float4 nonlinearTolerances;
    // natural map, cone, complementarity, energy.
    nm_float4 contactTolerances;
    // diagonal floor, initial LM shift, maximum LM shift, curvature tolerance.
    nm_float4 regularization;
    // Armijo coefficient, minimum absolute temperature, activation epsilon,
    // pressure-projection stabilization.
    nm_float4 globalization;
} NMMixedSolverGPU;

typedef struct NM_ALIGN16 NMFEMCapacityGPU {
    // node capacity, tetrahedron capacity, cohesive-face capacity, channel capacity.
    nm_uint4 topology;
    // mutation-command capacity, incidence capacity, contact capacity,
    // puncture impulse threshold encoded as IEEE-754 float bits.
    nm_uint4 work;
} NMFEMCapacityGPU;

typedef struct NM_ALIGN16 NMMixedMaterialGPU {
    // bulk modulus, thermal expansion, Biot coefficient, reference temperature.
    nm_float4 mechanics;
    // heat capacity, conductivity, heat source, electrical Joule fraction.
    nm_float4 thermal;
    // pore storage, permeability/viscosity, pore source, reserved.
    nm_float4 porous;
    // electrical conductivity, activation diffusivity, activation on/off rate.
    nm_float4 electrical;
    // reference fibre xyz, maximum active Cauchy tension.
    nm_float4 fibre;
    // activation threshold, activation slope, cohesive strength, fracture energy.
    nm_float4 coupling;
} NMMixedMaterialGPU;

typedef struct NM_ALIGN16 NMFEMFieldStateGPU {
    // mechanical pressure, temperature, pore pressure, electric potential.
    nm_float4 primary;
    // activation, previous log(J), accumulated Joule heat, flags-as-float.
    nm_float4 secondary;
} NMFEMFieldStateGPU;

typedef struct NM_ALIGN16 NMFieldBoundaryGPU {
    // global FEM node, object, flags, stable identifier.
    nm_uint4 identity;
    // Dirichlet temperature, pore pressure, electric potential, activation.
    nm_float4 value;
    // thermal flux, pore flux, electric current, reserved.
    nm_float4 flux;
} NMFieldBoundaryGPU;

typedef struct NM_ALIGN16 NMFEMTopologyNodeGPU {
    // source node, object, topology generation, flags.
    nm_uint4 identity;
} NMFEMTopologyNodeGPU;

typedef struct NM_ALIGN16 NMFEMSurfaceFaceGPU {
    // First tetrahedron, second tetrahedron or invalid, object, flags.
    nm_uint4 adjacency;
    // Opposite corner in first/second tetrahedron, stable face id, reserved.
    nm_uint4 sides;
} NMFEMSurfaceFaceGPU;

enum NMContinuumSurfaceKind : nm_u32 {
    NM_CONTINUUM_SURFACE_TRIANGLE = 0u,
    NM_CONTINUUM_SURFACE_POINT = 1u,
};

typedef struct NM_ALIGN16 NMContinuumSurfacePrimitiveGPU {
    // Up to three unified continuum nodes and owning object. MPM nodes use
    // [0, gridNodeCount); FEM nodes use gridNodeCount + local FEM node.
    nm_uint4 nodesAndObject;
    // Stable source, NMContinuumSurfaceKind, topology generation, flags.
    nm_uint4 identity;
    // Swept AABB minimum xyz and current measure (one for a point).
    nm_float4 boundsMinimum;
    // Swept AABB maximum xyz and maximum vertex travel.
    nm_float4 boundsMaximum;
} NMContinuumSurfacePrimitiveGPU;

typedef struct NM_ALIGN16 NMDeformableContactCandidateGPU {
    // First/second environment-local surface primitive, stable candidate id,
    // flags.
    nm_uint4 identity;
} NMDeformableContactCandidateGPU;

enum NMDeformableContactKind : nm_u32 {
    NM_DEFORMABLE_CONTACT_VERTEX_TRIANGLE = 0u,
    NM_DEFORMABLE_CONTACT_EDGE_EDGE = 1u,
    NM_DEFORMABLE_CONTACT_POINT_POINT = 2u,
};

typedef struct NM_ALIGN16 NMDeformableContactGPU {
    // Four unified continuum nodes carrying the contact Jacobian. Unused
    // entries are NM_INVALID_INDEX.
    nm_uint4 nodes;
    // Positive-side object, negative-side object, feature kind, flags.
    nm_uint4 identity;
    // Signed nodal weights; sum is zero for translational invariance.
    nm_float4 weights;
    // Per-node timestep divided by the positive-side reference timestep.
    // These scales make cross-rate IPC the gradient/Hessian of one action.
    nm_float4 timeScales;
    // Contact normal xyz and normalized time of impact in [0, 1].
    nm_float4 normalAndTOI;
    // Contact point xyz and unsigned feature separation at impact.
    nm_float4 pointAndSeparation;
    // Relative velocity xyz and lumped normal inverse response.
    nm_float4 velocityAndResponse;
    // Source primitive pair, stable candidate id, topology generation.
    nm_uint4 source;
    // Primal barrier impulse, two lagged-friction coordinates, friction.
    nm_float4 impulseAndFriction;
    // IPC normal impulse, velocity-space PSD curvature, thickness, barrier k.
    nm_float4 barrier;
    // Rows of the PSD-projected spatial barrier/friction Hessian. Signed
    // feature weights map this 3x3 metric into the complete nodal block.
    nm_float4 barrierHessianRow0;
    nm_float4 barrierHessianRow1;
    nm_float4 barrierHessianRow2;
    // Analytic gradient of the IPC feature mollifier with respect to each
    // nodal position. Point-point rows are zero.
    nm_float4 mollifierGradientRow0;
    nm_float4 mollifierGradientRow1;
    nm_float4 mollifierGradientRow2;
    nm_float4 mollifierGradientRow3;
    // mollifier value, dt * unmollified barrier energy,
    // dt^2 * PSD Gauss-Newton curvature, unmollified barrier stiffness.
    nm_float4 mollifier;
} NMDeformableContactGPU;

// Compact transaction-owned deformable-contact history. The
// geometric contact record is rebuilt from current surface primitives on each
// Newton candidate; only the stable source, transported frame, and lagged
// primal-potential history cross iteration/microstep boundaries.
typedef struct NM_ALIGN16 NMDeformableContactHistoryGPU {
    // Source primitive pair, stable candidate id, topology generation.
    nm_uint4 source;
    // Previous contact normal xyz and primal barrier impulse.
    nm_float4 normalAndBarrier;
    // Lagged tangent-potential coordinates, effective friction, valid marker.
    nm_float4 laggedTangentAndFriction;
} NMDeformableContactHistoryGPU;

typedef struct NM_ALIGN16 NMFEMTopologyStateGPU {
    // active nodes, active tetrahedra, active cohesive faces, active channels.
    nm_uint4 counts;
    // accepted arena, candidate arena, checkpoint arena, generation.
    nm_uint4 roles;
    // conserved mass, removed mass, removed energy, cohesive energy.
    nm_float4 accounting;
} NMFEMTopologyStateGPU;

typedef struct NM_ALIGN16 NMCohesiveFaceGPU {
    // three nodes and first tetrahedron.
    nm_uint4 nodesAndFirst;
    // second tetrahedron, object, stable identifier, flags.
    nm_uint4 adjacency;
    // damage, maximum opening, traction, fracture work.
    nm_float4 state;
    // rest normal xyz and rest area.
    nm_float4 geometry;
    // Separated-side node copies; w is reserved.
    nm_uint4 separatedNodes;
} NMCohesiveFaceGPU;

typedef struct NM_ALIGN16 NMMutationCommandGPU {
    // kind, object, stable identifier, flags.
    nm_uint4 identity;
    // Control step, deterministic target, priority, reserved. Remeshing
    // targets are object-local tetrahedra except vertex smoothing, whose
    // target is an object-local node.
    nm_uint4 schedule;
    // plane normal or cylinder axis, w plane offset/radius.
    nm_float4 geometry0;
    // cylinder origin, w finite half length.
    nm_float4 geometry1;
} NMMutationCommandGPU;

typedef struct NM_ALIGN16 NMPunctureChannelGPU {
    // object, stable identifier, topology generation, flags.
    nm_uint4 identity;
    // origin xyz and radius.
    nm_float4 originAndRadius;
    // axis xyz and finite half length.
    nm_float4 axisAndHalfLength;
} NMPunctureChannelGPU;

typedef struct NM_ALIGN16 NMLearnedMaterialGPU {
    // first layer, layer count, first weight, weight count.
    nm_uint4 layout;
    // invariant count, material index, activation kind, flags.
    nm_uint4 identity;
    // softplus beta, determinant floor, growth coefficient, reserved.
    nm_float4 policy;
    // canonical content fingerprint low/high, accepted revision low/high.
    nm_uint4 fingerprint;
} NMLearnedMaterialGPU;

typedef struct NM_ALIGN16 NMLearnedLayerGPU {
    // input width, output width, first weight, first bias.
    nm_uint4 layout;
    // input skip offset, hidden recurrence offset, flags, reserved.
    nm_uint4 routing;
} NMLearnedLayerGPU;

typedef struct NM_ALIGN16 NMLearnedDifferentialGPU {
    nm_float4 deformationRow0;
    nm_float4 deformationRow1;
    nm_float4 deformationRow2;
    nm_float4 directionRow0;
    nm_float4 directionRow1;
    nm_float4 directionRow2;
    nm_float4 firstPiolaRow0;
    nm_float4 firstPiolaRow1;
    nm_float4 firstPiolaRow2;
    nm_float4 tangentRow0;
    nm_float4 tangentRow1;
    nm_float4 tangentRow2;
    nm_float4 diagnostics;
} NMLearnedDifferentialGPU;

typedef struct NM_ALIGN16 NMSolverCertificateGPU {
    // nonlinear residual, relative correction, volume residual, pressure residual.
    nm_float4 nonlinear;
    // maximum barrier impulse, minimum separation, tangential impulse,
    // barrier energy.
    nm_float4 contact;
    // thermal, pore, electric, activation residual.
    nm_float4 transport;
    // minimum determinant, minimum curvature, energy error, accepted flag.
    nm_float4 validity;
} NMSolverCertificateGPU;

enum NMMicrostepFlags : nm_u32 {
    NM_MICROSTEP_CAPTURE_EVENTS = 1u << 0u,
    NM_MICROSTEP_FGMRES_OPERATOR = 1u << 1u,
    NM_MICROSTEP_FIELD_PRECONDITIONER = 1u << 2u,
};

typedef struct NM_ALIGN16 NMMicrostepGPU {
    nm_u32 controlStep;
    nm_u32 microtick;
    nm_u32 microtickCount;
    nm_u32 pcgIteration;

    nm_u32 seedLo;
    nm_u32 seedHi;
    nm_u32 flags;
    nm_u32 reserved;

    // microtick dt, inverse dt, execution time, duration.
    nm_float4 time;
} NMMicrostepGPU;

typedef struct NM_ALIGN16 NMBridgeDispatchGPU {
    nm_u32 environmentCount;
    nm_u32 rigidProxyCount;
    nm_u32 currentBodyCount;
    nm_u32 currentBodyStride;

    nm_u32 bodyWrenchCount;
    nm_u32 sceneBodyCount;
    nm_u32 bodyWrenchStride;
    nm_u32 sceneStride;

    nm_u32 reactionStride;
    nm_u32 flags;
    // Number of body-owned reaction ranges compiled by the runtime.
    nm_u32 reactionBodyCount;
    nm_u32 reserved1;

    // inverse dt, dt, reserved, reserved.
    nm_float4 time;
} NMBridgeDispatchGPU;

typedef struct NM_ALIGN16 NMResetPassGPU {
    nm_u32 controlStep;
    nm_u32 resetMaskStepStride;
    nm_u32 stateWidth;
    nm_u32 flags;
} NMResetPassGPU;

typedef struct NM_ALIGN16 NMIdentificationPassGPU {
    nm_u32 candidateCount;
    nm_u32 distributionCount;
    nm_u32 generation;
    nm_u32 flags;

    nm_u32 seedLo;
    nm_u32 seedHi;
    nm_u32 reserved0;
    nm_u32 reserved1;
} NMIdentificationPassGPU;

typedef struct NM_ALIGN16 NMExpressionInstructionGPU {
    nm_u32 opcode;
    nm_u32 index;
    nm_i32 integer;
    nm_u32 reserved;
    nm_float4 immediate;
} NMExpressionInstructionGPU;

typedef struct NM_ALIGN16 NMScalarProgramGPU {
    nm_u32 firstInstruction;
    nm_u32 instructionCount;
    nm_u32 maximumStack;
    nm_u32 flags;
} NMScalarProgramGPU;

typedef struct NM_ALIGN16 NMMaterialGPU {
    nm_u32 constitutiveKind;
    nm_u32 parameterOffset;
    nm_u32 parameterCount;
    nm_u32 stateCount;

    nm_u32 stressProgramOffset;
    nm_u32 tangentProgramOffset;
    nm_u32 validityProgram;
    nm_u32 flags;

    // Material-local initial state, next-state update programs, dissipation
    // scalar program, and projection policy.
    nm_u32 stateInitialOffset;
    nm_u32 stateUpdateProgramOffset;
    nm_u32 dissipationProgram;
    nm_u32 projectionKind;

    // Viscous stress and rate-tangent program ranges. Invalid offsets mean
    // that the material has no rate-dependent constitutive contribution.
    nm_u32 viscousStressProgramOffset;
    nm_u32 viscousTangentProgramOffset;
    nm_u32 implicitResidualProgramOffset;
    nm_u32 implicitJacobianProgramOffset;

    // R_F dF programs, P_z programs, packed 2-bit transfer policies, and the
    // fixed local Newton budget. These programs produce the consistent
    // algorithmic tangent P_F - P_z R_z^-1 R_F.
    nm_u32 implicitDeformationProgramOffset;
    nm_u32 stressStateDerivativeProgramOffset;
    nm_u32 stateTransferMask;
    nm_u32 localNewtonIterations;

    // density, reference temperature, specialized shear modulus, lambda.
    nm_float4 bulk;
    // static friction, dynamic friction, restitution, adhesion.
    nm_float4 interfaceResponse;
    // yield/cohesion strength, isotropic hardening, DP alpha, DP cohesion.
    nm_float4 inelastic;
    // minimum J, maximum J, maximum stress, maximum energy density.
    nm_float4 validity;
} NMMaterialGPU;

typedef struct NM_ALIGN16 NMParameterRangeGPU {
    // value, lower, upper, logarithmic flag.
    nm_float4 valueAndBounds;
} NMParameterRangeGPU;

typedef struct NM_ALIGN16 NMContinuumObjectGPU {
    nm_u32 representation;
    nm_u32 materialIndex;
    nm_u32 flags;
    nm_u32 schedulerIndex;

    nm_u32 stateOffset;
    nm_u32 stateCount;
    nm_u32 elementOffset;
    nm_u32 elementCount;

    nm_u32 auxiliaryOffset;
    nm_u32 auxiliaryCount;
    nm_u32 rigidBinding;
    nm_u32 topologyGeneration;

    // base rate exponent, maximum exponent, PCG iterations, reserved.
    nm_uint4 solver;
    // characteristic length, rigid tolerance, promotion strain, demotion strain.
    nm_float4 fidelity;
} NMContinuumObjectGPU;


typedef struct NM_ALIGN16 NMMPMGridGPU {
    // xyz inclusive minimum global node coordinate; w object index.
    nm_int4 nodeMinimumAndObject;
    // xyz node dimensions; w first node in the global node arena.
    nm_uint4 nodeDimensionsAndOffset;
    // xyz inclusive minimum block coordinate; w first block.
    nm_int4 blockMinimumAndOffset;
    // xyz block dimensions; w first dense block-lookup entry.
    nm_uint4 blockDimensionsAndLookup;
    // cell width, inverse cell width, support radius in cells, reserved.
    nm_float4 metrics;
} NMMPMGridGPU;

typedef struct NM_ALIGN16 NMMPMBlockGPU {
    // low/high Morton key, grid index, object index.
    nm_uint4 identity;
    // xyz block coordinate, w local dense lookup index.
    nm_int4 coordinateAndLookup;
} NMMPMBlockGPU;

typedef struct NM_ALIGN16 NMMPMParticleKeyGPU {
    // block index, low/high cell Morton key, flags.
    nm_uint4 identity;
} NMMPMParticleKeyGPU;

typedef struct NM_ALIGN16 NMActiveMPMBlockGPU {
    nm_u32 environment;
    nm_u32 blockIndex;
    nm_u32 firstParticle;
    nm_u32 particleCount;
} NMActiveMPMBlockGPU;

typedef struct NM_ALIGN16 NMIndirectDispatchGPU {
    nm_u32 threadgroupsX;
    nm_u32 threadgroupsY;
    nm_u32 threadgroupsZ;
    nm_u32 reserved;
} NMIndirectDispatchGPU;

typedef struct NM_ALIGN16 NMParticleStateGPU {
    nm_float4 positionAndMass;
    nm_float4 velocityAndReferenceVolume;
    nm_float4 deformationRow0;
    nm_float4 deformationRow1;
    nm_float4 deformationRow2;
    nm_float4 affineRow0;
    nm_float4 affineRow1;
    nm_float4 affineRow2;
    nm_float4 referenceAndTemperature;
    // object, material, topology generation, flags.
    nm_uint4 identity;
} NMParticleStateGPU;

typedef struct NM_ALIGN16 NMGridNodeStateGPU {
    nm_float4 positionAndMass;
    nm_float4 velocityAndInverseMass;
    nm_float4 forceAndEnergy;
    nm_float4 candidateVelocity;
} NMGridNodeStateGPU;

typedef struct NM_ALIGN16 NMMPMStencilGPU {
    nm_u32 particleIndex;
    nm_u32 nodeIndex;
    nm_u32 localSlot;
    nm_u32 reserved;
    // xyz reference gradient, w weight.
    nm_float4 gradientAndWeight;
} NMMPMStencilGPU;

typedef struct NM_ALIGN16 NMIncidenceRangeGPU {
    nm_u32 first;
    nm_u32 count;
    nm_u32 objectIndex;
    nm_u32 reserved;
} NMIncidenceRangeGPU;

typedef struct NM_ALIGN16 NMFEMNodeStateGPU {
    nm_float4 positionAndMass;
    nm_float4 velocityAndInverseMass;
    nm_float4 restAndFixed;
    nm_float4 deltaVelocity;
} NMFEMNodeStateGPU;

typedef struct NM_ALIGN16 NMTetrahedronGPU {
    nm_uint4 nodes;
    // inverse rest matrix rows; row0.w is rest volume.
    nm_float4 inverseRestRow0;
    nm_float4 inverseRestRow1;
    nm_float4 inverseRestRow2;
    // material, object, topology generation, flags.
    nm_uint4 identity;
} NMTetrahedronGPU;

typedef struct NM_ALIGN16 NMFEMElementVectorGPU {
    nm_float4 node0;
    nm_float4 node1;
    nm_float4 node2;
    nm_float4 node3;
} NMFEMElementVectorGPU;

typedef struct NM_ALIGN16 NMPCGScalarGPU {
    // r.r, p.Ap, alpha, beta.
    nm_float4 reduction;
    // residual, initial residual, converged, iteration.
    nm_float4 diagnostics;
} NMPCGScalarGPU;

typedef struct NM_ALIGN16 NMFGMRESStateGPU {
    // Current residual, initial residual, convergence flag, Arnoldi columns.
    nm_float4 diagnostics;
    // First Newton residual, current Newton residual, last correction ratio,
    // nonlinear convergence flag. Private runtime state; never serialized.
    nm_float4 nonlinear;
} NMFGMRESStateGPU;

typedef struct NM_ALIGN16 NMMixedPCGScalarGPU {
    nm_float4 rz;
    nm_float4 pap;
    nm_float4 alpha;
    nm_float4 beta;
    nm_float4 residual;
    nm_float4 initialResidual;
    nm_uint4 converged;
    nm_uint4 iterations;
} NMMixedPCGScalarGPU;

typedef struct NM_ALIGN16 NMRigidProxyGPU {
    nm_u32 shapeKind;
    // Global EngineModel body index. This is the canonical index used by the
    // current-body arena and the unified external-wrench arena.
    nm_u32 bodyIndex;
    // Environment-local scene-body index used only when adaptive transfer
    // publishes a rigid state back into MetalWorld's scene-state stream.
    nm_u32 sceneBodyIndex;
    nm_u32 materialIndex;

    nm_u32 flags;
    // Adaptive continuum object that owns this rigid fallback, or invalid.
    nm_u32 adaptiveObjectIndex;
    // Stable compact free-body generalized coordinate owner. Every proxy on
    // the same dynamic non-articulated body shares one index; all other
    // proxies carry NM_INVALID_INDEX.
    nm_u32 generalizedFreeBodyIndex;
    nm_u32 reserved2;

    // body-local center or plane normal; w radius or plane offset.
    nm_float4 localCenterAndRadius;
    // capsule endpoint B or box half extent.
    nm_float4 localExtent;
    nm_float4 localOrientation;
} NMRigidProxyGPU;

typedef struct NM_ALIGN16 NMRigidStateGPU {
    nm_float4 centerAndRadius;
    nm_float4 extent;
    nm_float4 orientation;
    nm_float4 linearVelocityAndInverseMass;
    nm_float4 angularVelocity;
    nm_float4 bodyCenter;
    nm_float4 inverseInertiaRow0;
    nm_float4 inverseInertiaRow1;
    nm_float4 inverseInertiaRow2;
} NMRigidStateGPU;

typedef struct NM_ALIGN16 NMContactPairGPU {
    // Unified node index: MPM nodes first, then FEM nodes.
    nm_u32 continuumNode;
    nm_u32 rigidProxy;
    nm_u32 objectIndex;
    nm_u32 materialInterface;
} NMContactPairGPU;

typedef struct NM_ALIGN16 NMContactSampleGPU {
    // continuum node, rigid proxy, object, flags.
    nm_uint4 identity;
    nm_float4 pointAndSeparation;
    nm_float4 normalAndVelocity;
    // xyz impulse on rigid, w normal impulse.
    nm_float4 impulseAndNormal;
    // xyz angular impulse on rigid, w tangential impulse.
    nm_float4 angularImpulseAndTangent;
    // IPC normal impulse, velocity-space PSD curvature, thickness, barrier k.
    nm_float4 barrier;
    // Rows of the primal velocity-space PSD barrier/friction Hessian.
    nm_float4 barrierHessianRow0;
    nm_float4 barrierHessianRow1;
    nm_float4 barrierHessianRow2;
} NMContactSampleGPU;

typedef struct NM_ALIGN16 NMRigidReactionGPU {
    nm_float4 impulseAndCount;
    nm_float4 angularImpulse;
} NMRigidReactionGPU;

typedef struct NM_ALIGN16 NMAdaptiveStateGPU {
    nm_u32 activeRepresentation;
    nm_u32 requestedRepresentation;
    nm_u32 stableFrames;
    nm_u32 flags;

    // mass, maximum strain, RMS strain, reconstruction residual.
    nm_float4 massAndError;
    // Current world-space centre of mass; w is the measured bounding radius.
    nm_float4 centerAndRadius;
    // Immutable authored rest-frame centre used for rigid-to-continuum
    // reconstruction. Keeping it separate from the current COM prevents
    // translation from leaking into the promoted deformation.
    nm_float4 referenceCenter;
    nm_float4 linearVelocityAndAngularSpeed;
    // xyz angular velocity, w minimum determinant.
    nm_float4 angularVelocityAndMinimumJ;
    nm_float4 orientation;
    nm_float4 inverseInertiaRow0;
    nm_float4 inverseInertiaRow1;
    nm_float4 inverseInertiaRow2;
} NMAdaptiveStateGPU;

typedef struct NM_ALIGN16 NMIdentificationDistributionGPU {
    // material, local parameter, global parameter, flags.
    nm_uint4 identity;
    // mean, standard deviation, lower, upper.
    nm_float4 momentsAndBounds;
    // learning rate, temperature, minimum std, logarithmic flag.
    nm_float4 update;
} NMIdentificationDistributionGPU;

typedef struct NM_ALIGN16 NMIdentificationCandidateGPU {
    nm_u32 environment;
    nm_u32 candidate;
    nm_u32 distribution;
    nm_u32 antitheticPartner;
    // value, normalized perturbation, loss, weight.
    nm_float4 sample;
} NMIdentificationCandidateGPU;

typedef struct NM_ALIGN16 NMSchedulerStateGPU {
    nm_u32 baseExponent;
    nm_u32 activeExponent;
    nm_u32 requestedExponent;
    nm_u32 quietFrames;

    // contact speed, slip impulse, maximum strain, maximum stress.
    nm_float4 physical;
    // minimum J, solver residual, damage, continuum-enabled flag.
    nm_float4 numerical;
    // contact, slip, strain, residual thresholds.
    nm_float4 thresholds;
    // elapsed episode time, last event time, last event delta, event count.
    nm_float4 timing;
} NMSchedulerStateGPU;

typedef struct NM_ALIGN16 NMEventTokenGPU {
    nm_u32 environment;
    nm_u32 objectIndex;
    nm_u32 eventClass;
    nm_u32 flags;
    // episode time, delta since the previous object event, severity, current.
    nm_float4 payload;
    // previous value, current value, reserved, reserved.
    nm_float4 transition;
} NMEventTokenGPU;

typedef struct NM_ALIGN16 NMMatterStatusGPU {
    nm_u32 code;
    nm_u32 environment;
    nm_u32 objectIndex;
    nm_u32 failingIndex;

    nm_u32 completedMicrosteps;
    nm_u32 pcgIterations;
    nm_u32 contactCount;
    nm_u32 eventCount;

    // minimum J, maximum stress, residual, correction.
    nm_float4 diagnostics;
} NMMatterStatusGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(nm_float4) == 16);
static_assert(sizeof(nm_uint4) == 16);
static_assert(sizeof(nm_int4) == 16);
static_assert(sizeof(NMMatterDispatchGPU) % 16 == 0);
static_assert(sizeof(NMMicrostepGPU) % 16 == 0);
static_assert(sizeof(NMResetPassGPU) == 16);
static_assert(sizeof(NMBridgeDispatchGPU) % 16 == 0);
static_assert(sizeof(NMRigidProxyGPU) == 80);
static_assert(sizeof(NMMaterialGPU) % 16 == 0);
static_assert(sizeof(NMContinuumObjectGPU) % 16 == 0);
static_assert(sizeof(NMMPMGridGPU) == 80);
static_assert(sizeof(NMMPMBlockGPU) == 32);
static_assert(sizeof(NMParticleStateGPU) % 16 == 0);
static_assert(sizeof(NMFEMNodeStateGPU) % 16 == 0);
static_assert(sizeof(NMAdaptiveStateGPU) == 160);
static_assert(sizeof(NMSchedulerStateGPU) == 80);
static_assert(sizeof(NMEventTokenGPU) == 48);
#endif
