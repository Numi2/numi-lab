#pragma once

// Versioned, pointer-free ABI shared by C++, Objective-C++, and Metal.
// The old MRModelGPU path remains available while the generic engine is
// brought online. New engine code must use offsets/counts and report capacity
// failures. This is the current hard capacity of one connected throughput
// island. Scene-level pipelines partition independent islands into separate
// batches; a connected island above this limit returns an explicit overflow.

#include "metalrobo/gpu_types.h"
#include "metalrobo/constraint_ir_shared.h"

#define MR_ENGINE_ABI_VERSION 4u
#define MR_INVALID_INDEX 0xffffffffu
#define MR_MAX_CONTACTS_PER_SOLVER_BATCH 128u
#define MR_MAX_BODIES_PER_SOLVER_BATCH \
    (2u * MR_MAX_CONTACTS_PER_SOLVER_BATCH)
#define MR_BROADPHASE_SCAN_BLOCK_SIZE 256u
#define MR_MAX_BROADPHASE_SCAN_BLOCKS 256u
// Generic articulated-operator capacities are intentionally independent of
// the Franka-era MR_MAX_DOF/MR_MAX_LINKS compatibility path. They cover the
// floating-base Unitree G1 (30 bodies, 35 velocity coordinates) with room for
// larger trees while keeping one dense FP32 Cholesky factor inside Apple GPU
// threadgroup memory.
#define MR_ARTICULATED_OPERATOR_MAX_BODIES 64u
#define MR_ARTICULATED_OPERATOR_MAX_DOFS 64u
// Retained as the legacy standalone operator's recommended allocation class.
// It is not a runtime limit: checked GPU strides and caller-provided storage
// now determine the point/contact capacity.
#define MR_ARTICULATED_OPERATOR_MAX_POINTS 1024u
// Versioned FP32 backward-error gate for M * deltaV = J^T * impulse.
// A finite but inaccurate factor solve is a failure, never publishable state.
#define MR_ARTICULATED_OPERATOR_MAX_RELATIVE_RESIDUAL 0.00003f
// The O(n) articulated-body kernel uses fixed-size threadgroup scratch for
// one tree per 32-lane threadgroup. These are explicit ABI capacities rather
// than silent implementation assumptions.
#define MR_ARTICULATED_ABA_MAX_BODIES 32u
#define MR_ARTICULATED_ABA_MAX_DOFS 40u
#define MR_ARTICULATED_ABA_MAX_Q 41u
#define MR_ARTICULATED_INVERSE_MASS_MAX_RHS 3u
// Versioned first generic Metal-world graph. One submission may encode many
// control steps, each with a bounded number of ABA physics substeps, without
// a command-buffer completion or CPU-visible intermediate state.
#define MR_METAL_WORLD_ABI_VERSION 5u
#define MR_METAL_WORLD_MAX_PHYSICS_SUBSTEPS 64u
#define MR_METAL_WORLD_CONTACT_ABI_VERSION 7u
#define MR_METAL_WORLD_MANIFOLD_POINT_CAPACITY 4u
#define MR_METAL_WORLD_RAW_CONTACTS_PER_PAIR 8u
#define MR_WAVE32_CONTACTS_PER_TILE 32u
#define MR_WAVE32_ROWS_PER_TILE \
    (3u * MR_WAVE32_CONTACTS_PER_TILE)
#define MR_WAVE32_DISTRIBUTED_TILE_THRESHOLD 8u
#define MR_WORLD_QUEUE_THREADS_PER_THREADGROUP 64u
#define MR_WORLD_MAX_DYNAMIC_NODES 256u
#define MR_CONVEX_SIMPLEX_CAPACITY 4u
#define MR_GJK_MAX_ITERATIONS 32u
#define MR_EPA_MAX_ITERATIONS 48u
#define MR_EPA_VERTEX_CAPACITY 64u
#define MR_EPA_FACE_CAPACITY 128u
#define MR_MESH_BVH_BRANCHING 4u
#define MR_MESH_BVH_LEAF_BIT 0x80000000u
#define MR_MESH_BVH_LEAF_COUNT_MASK 0x000000ffu
#define MR_MESH_BVH_ESCAPE_SHIFT 8u
#define MR_MESH_BVH_ESCAPE_MASK 0x7fffff00u
#define MR_MESH_BVH_INVALID_ESCAPE 0x007fffffu
#define MR_CCD_DEFAULT_MAX_EVENTS 4u
#define MR_CCD_MAX_EVENTS 16u
#define MR_CCD_DEFAULT_ADVANCE_SOLVE_PASSES 4u
#define MR_CCD_MAX_ADVANCE_SOLVE_PASSES 8u
#define MR_CCD_DEFAULT_ZERO_TIME_REPLAYS 2u
#define MR_CCD_MAX_ZERO_TIME_REPLAYS 8u
// Authored collision coordinates, local offsets, primitive dimensions, and
// contact/rest/bounding-radius values use this direct-input domain. With
// normalized rotations, every supported finite primitive derived from these
// records remains well inside MR_MAX_COLLISION_COORDINATE. This intentional
// 10x gap prevents CPU/Metal rounding from becoming an admission decision.
#define MR_MAX_COLLISION_INPUT_COORDINATE 100000.0f
// Derived transforms and finite AABBs use this larger execution domain.
// Larger worlds use per-environment origin rebasing.
#define MR_MAX_COLLISION_COORDINATE 1000000.0f
// Metal bounds are inflated to cover FP32 transform/normalization and
// expression-order error. Broadphase false positives are permitted; inward
// bounds and false negatives are not.
#define MR_COLLISION_AABB_RELATIVE_PAD 0.00000762939453125f
#define MR_COLLISION_QUERY_RELATIVE_PAD 0.00000762939453125f
// Quaternion scale carries no physical meaning. Admission uses direct
// component bounds rather than a backend-sensitive dot-product tolerance,
// then normalizes. Every unit quaternion has a maximum component >= 0.5.
#define MR_MIN_QUATERNION_MAX_COMPONENT 0.25f
#define MR_MAX_QUATERNION_MAX_COMPONENT 1.001f
// Non-plane active geometry below one nanometre is outside the robotics
// collision contract. This also keeps Metal's flush-to-zero mode from
// changing whether a positive authored extent exists.
#define MR_MIN_COLLISION_EXTENT 0.000000001f

enum MRMotionType : mr_u32 {
    MR_MOTION_STATIC = 0u,
    MR_MOTION_KINEMATIC = 1u,
    MR_MOTION_DYNAMIC = 2u,
};

enum MRRootType : mr_u32 {
    MR_ROOT_FIXED = 0u,
    MR_ROOT_FLOATING = 1u,
};

enum MRJointTypeExt : mr_u32 {
    MR_JOINT_REVOLUTE = 0u,
    MR_JOINT_PRISMATIC = 1u,
    MR_JOINT_CONTINUOUS = 2u,
    MR_JOINT_SPHERICAL = 3u,
    MR_JOINT_PLANAR = 4u,
    MR_JOINT_FIXED = 5u,
    MR_JOINT_FREE = 6u,
};

enum MRConstraintType : mr_u32 {
    MR_CONSTRAINT_CONTACT = 0u,
    MR_CONSTRAINT_BILATERAL = 1u,
    MR_CONSTRAINT_LIMIT = 2u,
    MR_CONSTRAINT_DISTANCE = 3u,
    MR_CONSTRAINT_WELD = 4u,
    MR_CONSTRAINT_GEAR = 5u,
    MR_CONSTRAINT_TENDON = 6u,
    MR_CONSTRAINT_DRY_FRICTION = 7u,
};

enum MRCollisionPairClass : mr_u32 {
    MR_COLLISION_PAIR_UNSUPPORTED = 0u,
    MR_COLLISION_PAIR_SPHERE_SPHERE = 1u,
    MR_COLLISION_PAIR_SPHERE_PLANE = 2u,
    MR_COLLISION_PAIR_CAPSULE_PLANE = 3u,
    MR_COLLISION_PAIR_BOX_PLANE = 4u,
    MR_COLLISION_PAIR_CYLINDER_PLANE = 5u,
    MR_COLLISION_PAIR_SPHERE_CAPSULE = 6u,
    MR_COLLISION_PAIR_CAPSULE_CAPSULE = 7u,
    MR_COLLISION_PAIR_SPHERE_BOX = 8u,
    MR_COLLISION_PAIR_CAPSULE_BOX = 9u,
    MR_COLLISION_PAIR_BOX_BOX = 10u,
    MR_COLLISION_PAIR_CONVEX = 11u,
    MR_COLLISION_PAIR_MESH = 12u,
};

enum MRFrictionConeType : mr_u32 {
    MR_FRICTION_CONE_ELLIPTIC = 0u,
    MR_FRICTION_CONE_PYRAMID_4 = 1u,
    MR_FRICTION_CONE_PYRAMID_8 = 2u,
};

enum MRSolverType : mr_u32 {
    MR_SOLVER_REFERENCE_FP64 = 0u,
    MR_SOLVER_QUALITY_NEWTON = 1u,
    MR_SOLVER_THROUGHPUT_TGS = 2u,
    MR_SOLVER_THROUGHPUT_PGS = 3u,
};

enum MRFreeBodyIntegratorType : mr_u32 {
    MR_FREE_BODY_SYMPLECTIC_EULER = 0u,
    MR_FREE_BODY_IMPLICIT_MIDPOINT = 1u,
};

enum MRStepStatusCode : mr_u32 {
    MR_STEP_SUCCESS = 0u,
    MR_STEP_FIXED_BUDGET_COMPLETE = 1u,
    MR_STEP_NONFINITE_INPUT = 2u,
    MR_STEP_NONFINITE_RESULT = 3u,
    MR_STEP_PAIR_CAPACITY_OVERFLOW = 4u,
    MR_STEP_CONTACT_CAPACITY_OVERFLOW = 5u,
    MR_STEP_MANIFOLD_CAPACITY_OVERFLOW = 6u,
    MR_STEP_CONSTRAINT_CAPACITY_OVERFLOW = 7u,
    MR_STEP_ISLAND_CAPACITY_OVERFLOW = 8u,
    MR_STEP_CCD_EVENT_BUDGET_EXHAUSTED = 9u,
    MR_STEP_FACTORIZATION_FAILED = 10u,
    MR_STEP_DID_NOT_CONVERGE = 11u,
    MR_STEP_UNSUPPORTED = 12u,
    MR_STEP_CCD_CAPACITY_OVERFLOW = 13u,
    MR_STEP_STATUS_COUNT = 14u,
};

enum MRConstraintFlags : mr_u32 {
    MR_CONSTRAINT_FLAG_NEW_IMPACT = 1u << 0u,
    MR_CONSTRAINT_FLAG_WARM_STARTED = 1u << 1u,
    MR_CONSTRAINT_FLAG_DISABLED = 1u << 2u,
    // At least one endpoint is a procedural rod edge. ConstraintIR remains
    // byte-for-byte unchanged; this bit selects the typed endpoint operator.
    MR_CONSTRAINT_FLAG_ROD_ENDPOINT = 1u << 3u,
    // Runtime block was seeded from the immutable mechanism program and
    // evaluates sparse generalized-coordinate endpoints rather than a
    // spatial contact point. These blocks share ConstraintIR, islands, and
    // the quality product-cone solver, but are excluded from contact tiles.
    MR_CONSTRAINT_FLAG_GENERALIZED = 1u << 4u,
};

enum MRShapeFlags : mr_u32 {
    // Retain factual/cooked geometry in the model while excluding it from
    // simulation until its narrowphase is executable.
    MR_SHAPE_FLAG_SIMULATION_DISABLED = 1u << 0u,
    // Enables exact event CCD for pairs containing this shape. Swept AABBs
    // and speculative constraints remain available without this flag.
    MR_SHAPE_FLAG_ENABLE_CCD = 1u << 1u,
    // Triangle meshes are one-sided unless this bit is authored.
    MR_SHAPE_FLAG_MESH_TWO_SIDED = 1u << 2u,
};

enum MRRawContactFlags : mr_u32 {
    // featureAndFlags[3] lower bits carry a cooked material index when set.
    MR_RAW_CONTACT_MATERIAL_OVERRIDE = 1u << 31u,
    MR_RAW_CONTACT_MATERIAL_INDEX_MASK = 0x7fffffffu,
};

enum MRWorldWorkClass : mr_u32 {
    MR_WORLD_WORK_ANALYTIC = 0u,
    MR_WORLD_WORK_SAT_CLIP = 1u,
    MR_WORLD_WORK_PRIMITIVE_GJK = 2u,
    MR_WORLD_WORK_HULL_GJK = 3u,
    MR_WORLD_WORK_HARD_CONVEX = 4u,
    MR_WORLD_WORK_MESH = 5u,
    MR_WORLD_WORK_MANIFOLD = 6u,
    MR_WORLD_WORK_SOLVER = 7u,
    MR_WORLD_WORK_SOLVER_SPILL = 8u,
    MR_WORLD_WORK_CCD = 9u,
    MR_WORLD_WORK_SOLVER_DISTRIBUTED = 10u,
    MR_WORLD_WORK_CLASS_COUNT = 11u,
};

enum MRWorldQueueFlags : mr_u32 {
    MR_WORLD_QUEUE_PERSISTENT_WORKER = 1u << 27u,
    MR_WORLD_QUEUE_COHORT_8 = 1u << 28u,
    MR_WORLD_QUEUE_COHORT_16 = 1u << 29u,
};

enum MRWorldCCDMode : mr_u32 {
    MR_WORLD_CCD_DISABLED = 0u,
    MR_WORLD_CCD_SPECULATIVE = 1u,
    MR_WORLD_CCD_HYBRID = 2u,
};

enum MRScanFlags : mr_u32 {
    MR_SCAN_BOOLEAN_INPUT = 1u << 0u,
};

enum MRGeometryKind : mr_u32 {
    MR_GEOMETRY_NONE = 0u,
    MR_GEOMETRY_CONVEX = 1u,
    MR_GEOMETRY_TRIANGLE_MESH = 2u,
    MR_GEOMETRY_HEIGHTFIELD = 3u,
};

enum MRGeometryFlags : mr_u32 {
    MR_GEOMETRY_FLAG_CLOSED = 1u << 0u,
    MR_GEOMETRY_FLAG_CONVEX = 1u << 1u,
    MR_GEOMETRY_FLAG_TWO_SIDED = 1u << 2u,
    MR_GEOMETRY_FLAG_QUANTIZED_BVH = 1u << 3u,
};

enum MRDofFlags : mr_u32 {
    // Root coordinates are never implicitly actuated. Floating-root records
    // carry this flag alone and keep every limit/drive value exactly zero.
    MR_DOF_FLAG_ROOT = 1u << 0u,
    MR_DOF_FLAG_ACTUATED = 1u << 1u,
    MR_DOF_FLAG_POSITION_LIMIT = 1u << 2u,
    MR_DOF_FLAG_VELOCITY_LIMIT = 1u << 3u,
    MR_DOF_FLAG_EFFORT_LIMIT = 1u << 4u,
    MR_DOF_FLAG_DRIVE = 1u << 5u,
};

enum MRActuatorProfileFlags : mr_u32 {
    MR_ACTUATOR_PROFILE_ACTIVE = 1u << 0u,
    // Set only for parameters identified from measurements on the target
    // mechanism. An authored engineering prior remains executable but must
    // not be presented as calibrated.
    MR_ACTUATOR_PROFILE_CALIBRATED = 1u << 1u,
};

typedef struct MR_ALIGN16 MRWorldGPU {
    mr_u32 abiVersion;
    mr_u32 bodyCount;
    mr_u32 articulationCount;
    mr_u32 jointCount;

    mr_u32 shapeCount;
    mr_u32 materialCount;
    mr_u32 nq;
    mr_u32 nv;

    mr_u32 pairCapacity;
    mr_u32 contactCapacity;
    mr_u32 constraintCapacity;
    mr_u32 islandCapacity;

    mr_u32 solverType;
    mr_u32 frictionConeType;
    mr_u32 flags;
    mr_u32 reserved;

    // xyz = gravity in world coordinates, w = frame timestep.
    mr_float4 gravityAndTimestep;
    // solver tolerance, minimum compliance, maximum depenetration speed, slop.
    mr_float4 solverScales;
} MRWorldGPU;

typedef struct MR_ALIGN16 MRArticulationGPU {
    mr_u32 rootBody;
    mr_u32 rootType;
    mr_u32 firstBody;
    mr_u32 bodyCount;

    mr_u32 firstJoint;
    mr_u32 jointCount;
    mr_u32 qOffset;
    mr_u32 nq;

    mr_u32 vOffset;
    mr_u32 nv;
    mr_u32 flags;
    mr_u32 solverGroup;
} MRArticulationGPU;

typedef struct MR_ALIGN16 MRJointDescriptorGPU {
    mr_u32 parentBody;
    mr_u32 childBody;
    mr_u32 jointType;
    mr_u32 flags;

    mr_u32 qOffset;
    mr_u32 nq;
    mr_u32 vOffset;
    mr_u32 nv;

    // Axes are expressed in the joint frame. Multi-DOF joints use axis1/axis2.
    mr_float4 axis0;
    mr_float4 axis1;
    mr_float4 axis2;
    // Anchor coordinates are relative to each body's COM-centred state
    // origin, while orientation remains the imported body/link frame.
    mr_float4 parentAnchor;
    mr_float4 childAnchor;
    // Joint-frame orientation (x, y, z, w) in each body.
    mr_float4 parentRotation;
    mr_float4 childRotation;
} MRJointDescriptorGPU;

// One authoritative record per global generalized-velocity coordinate.
// Records are stored in global v order; vIndex is repeated deliberately so
// malformed streams fail validation instead of silently changing ownership.
// qIndex is MR_INVALID_INDEX when a velocity coordinate has no one-to-one
// scalar configuration coordinate (floating/spherical quaternion rates).
typedef struct MR_ALIGN16 MRDofPropertiesGPU {
    mr_u32 articulationIndex;
    mr_u32 jointIndex;
    mr_u32 qIndex;
    mr_u32 vIndex;

    mr_u32 localDof;
    mr_u32 flags;
    mr_u32 reserved0;
    mr_u32 reserved1;

    // lower position, upper position, maximum velocity, maximum effort.
    // An inactive limit has both its flag and corresponding value(s) zero.
    // These limits are authoritative metadata; this record does not imply
    // post-step clamping or an actuator/limit constraint implementation.
    mr_float4 limits;
    // stiffness, damping, armature inertia, dry-friction loss.
    // Armature is physical generalized inertia and is independent of whether
    // a drive is enabled. The generic dynamics operators consume armature;
    // the explicit CPU articulated-actuation evaluator consumes named-model
    // gains and dry friction. The generic Metal operator does not yet execute
    // the actuation law.
    mr_float4 drive;
} MRDofPropertiesGPU;

// Optional authored motor/transmission truth in global generalized-velocity
// order. When present, EngineModel carries exactly one record per DoF;
// inactive/root entries are zero. The cooker derives stall torque so Metal
// hot loops consume a fixed record without recomputing authored products.
typedef struct MR_ALIGN16 MRActuatorProfileGPU {
    // joint-side torque constant N*m/A, current limit A,
    // no-load speed rad/s (or m/s), efficiency [0,1].
    mr_float4 motorAndSpeed;
    // backlash play in joint units, command delay seconds,
    // cooked stall torque N*m (or N), reserved.
    mr_float4 transmissionAndEnvelope;
    // global v index, MRActuatorProfileFlags, reserved, reserved.
    mr_uint4 identity;
} MRActuatorProfileGPU;

typedef struct MR_ALIGN16 MRBodyPropertiesGPU {
    mr_u32 articulationIndex;
    mr_u32 parentBody;
    mr_u32 inboundJoint;
    mr_u32 motionType;

    // x = mass, y = inverse mass. zw reserved.
    mr_float4 massAndInverseMass;
    // xyz = center of mass in the body frame.
    mr_float4 centerOfMass;
    // Symmetric inertia about COM in the body frame.
    mr_float4 inertiaRow0;
    mr_float4 inertiaRow1;
    mr_float4 inertiaRow2;
    mr_float4 inverseInertiaRow0;
    mr_float4 inverseInertiaRow1;
    mr_float4 inverseInertiaRow2;
    // linear damping, angular damping, max linear speed, max angular speed.
    mr_float4 dampingAndSpeedLimits;
} MRBodyPropertiesGPU;

typedef struct MR_ALIGN16 MRBodyStateGPU {
    // xyz = COM position in world coordinates.
    mr_float4 position;
    // Normalizable quaternion (x, y, z, w), body-to-world. Collision
    // canonicalizes it under the shared component-domain contract.
    mr_float4 orientation;
    // xyz = world linear velocity at COM; w = inverse mass.
    mr_float4 linearVelocityAndInverseMass;
    // xyz = world angular velocity.
    mr_float4 angularVelocity;
    // Inverse inertia about COM, already rotated into world coordinates.
    mr_float4 inverseInertiaWorldRow0;
    mr_float4 inverseInertiaWorldRow1;
    mr_float4 inverseInertiaWorldRow2;
    // x = MRMotionType, y = articulation, z = link, w = sleep/other flags.
    mr_u32 flagsAndIndices[4];
} MRBodyStateGPU;

typedef struct MR_ALIGN16 MRBodyWrenchGPU {
    mr_float4 force;
    mr_float4 torque;
} MRBodyWrenchGPU;

typedef struct MR_ALIGN16 MRFreeBodyBatchGPU {
    mr_u32 bodyOffset;
    mr_u32 bodyCount;
    mr_u32 integratorType;
    mr_u32 nonlinearIterations;

    // xyz = gravity, w = timestep.
    mr_float4 gravityAndTimestep;
    // nonlinear tolerance, reserved, reserved, reserved.
    mr_float4 convergence;
} MRFreeBodyBatchGPU;

typedef struct MR_ALIGN16 MRFreeBodyStatusGPU {
    mr_u32 code;
    mr_u32 iterations;
    mr_u32 bodyIndex;
    mr_u32 reserved;

    // nonlinear residual, quaternion norm error, angular speed, reserved.
    mr_float4 diagnostics;
} MRFreeBodyStatusGPU;

enum MRArticulatedOperatorStatusCode : mr_u32 {
    MR_ARTICULATED_OPERATOR_SUCCESS = 0u,
    MR_ARTICULATED_OPERATOR_INVALID_DISPATCH = 1u,
    MR_ARTICULATED_OPERATOR_CAPACITY_OVERFLOW = 2u,
    MR_ARTICULATED_OPERATOR_INVALID_MODEL = 3u,
    MR_ARTICULATED_OPERATOR_UNSUPPORTED_TOPOLOGY = 4u,
    MR_ARTICULATED_OPERATOR_NONFINITE_INPUT = 5u,
    MR_ARTICULATED_OPERATOR_FACTORIZATION_FAILED = 6u,
    MR_ARTICULATED_OPERATOR_NONFINITE_RESULT = 7u,
    MR_ARTICULATED_OPERATOR_ACCURACY_FAILED = 8u,
};

enum MRArticulatedOperatorFlags : mr_u32 {
    // The dense mass matrix is diagnostic/reference output. Runtime impulse
    // response always factor-solves M * deltaV = J^T * impulse and never
    // forms or applies an explicit inverse.
    MR_ARTICULATED_OPERATOR_WRITE_DIAGNOSTIC_MASS = 1u << 0u,
    // Publishes the lower-triangular Cholesky factor in the mass-matrix
    // output buffer. Upper-triangular entries are zero. This is the reusable
    // device-side factor cache consumed by the contact world.
    MR_ARTICULATED_OPERATOR_WRITE_CHOLESKY_FACTOR = 1u << 1u,
    // Computes and publishes body poses without assembling/factorizing M.
    // pointCount must be zero. This is the collider-projection fast path.
    MR_ARTICULATED_OPERATOR_KINEMATICS_ONLY = 1u << 2u,
    // Adds h D + h^2 K to generalized drive inertia. The matching
    // acceleration RHS is prepared from position targets by MetalWorld.
    MR_ARTICULATED_OPERATOR_IMPLICIT_DRIVES = 1u << 3u,
    // Computes body poses, point positions, and analytic point Jacobians but
    // skips mass assembly, factorization, and impulse response. This is the
    // spatial-row frontend for multi-articulation contact graphs whose shared
    // inverse-ABA stage owns mass response.
    MR_ARTICULATED_OPERATOR_KINEMATICS_JACOBIANS_ONLY = 1u << 4u,
};

// One dispatch describes a batch of states for one immutable articulation.
// q and point records are environment-major. All stride fields are element
// counts, never bytes. q stores articulation-local generalized coordinates.
typedef struct MR_ALIGN16 MRArticulatedOperatorDispatchGPU {
    mr_u32 articulationIndex;
    mr_u32 environmentCount;
    mr_u32 pointCount;
    mr_u32 flags;

    mr_u32 qStride;
    mr_u32 pointStride;
    mr_u32 bodyPoseStride;
    mr_u32 pointWorldStride;

    mr_u32 massMatrixStride;
    mr_u32 pointJacobianStride;
    mr_u32 generalizedStride;
    mr_u32 reserved0;
} MRArticulatedOperatorDispatchGPU;

// A world impulse applied at a COM-relative body point. bodyIndex is global
// in MRWorldGPU. Query flags and reserved words must currently be zero.
typedef struct MR_ALIGN16 MRArticulatedPointImpulseGPU {
    mr_u32 bodyIndex;
    mr_u32 flags;
    mr_u32 reserved0;
    mr_u32 reserved1;

    mr_float4 localPoint;
    mr_float4 worldImpulse;
} MRArticulatedPointImpulseGPU;

typedef struct MR_ALIGN16 MRArticulatedBodyPoseGPU {
    // xyz = body COM world position.
    mr_float4 position;
    // Normalized body-to-world quaternion xyzw.
    mr_float4 orientation;
} MRArticulatedBodyPoseGPU;

typedef struct MR_ALIGN16 MRArticulatedPointWorldGPU {
    // xyz = queried point in world coordinates.
    mr_float4 position;
} MRArticulatedPointWorldGPU;

typedef struct MR_ALIGN16 MRArticulatedOperatorStatusGPU {
    mr_u32 code;
    mr_u32 environment;
    mr_u32 articulationIndex;
    mr_u32 failingIndex;

    mr_u32 bodyCount;
    mr_u32 nq;
    mr_u32 nv;
    mr_u32 pointCount;

    // minimum Cholesky pivot, maximum pivot, relative solve residual, and
    // maximum absolute mass-matrix entry.
    mr_float4 diagnostics;
} MRArticulatedOperatorStatusGPU;

enum MRABAStatusCode : mr_u32 {
    MR_ABA_SUCCESS = 0u,
    MR_ABA_INVALID_DISPATCH = 1u,
    MR_ABA_INVALID_MODEL = 2u,
    MR_ABA_NONFINITE_INPUT = 3u,
    MR_ABA_FACTORIZATION_FAILED = 4u,
    MR_ABA_NONFINITE_RESULT = 5u,
    MR_ABA_INVALID_QUATERNION = 6u,
    MR_ABA_UNSUPPORTED_TOPOLOGY = 7u,
};

enum MRABAFlags : mr_u32 {
    MR_ABA_HAS_BODY_WRENCHES = 1u << 0u,
    MR_ABA_APPLY_BODY_DAMPING = 1u << 1u,
    MR_ABA_IMPLICIT_DRIVES = 1u << 2u,
};

// One dispatch advances a compact environment-major batch for one immutable
// articulation. Every stride is measured in elements, never bytes. The
// checked public host API derives compact strides and owns all bound storage.
typedef struct MR_ALIGN16 MRABADispatchGPU {
    mr_u32 articulationIndex;
    mr_u32 environmentCount;
    mr_u32 flags;
    mr_u32 reserved0;

    mr_u32 qStride;
    mr_u32 vStride;
    mr_u32 effortStride;
    mr_u32 wrenchStride;

    mr_u32 accelerationStride;
    mr_u32 nextVStride;
    mr_u32 nextQStride;
    mr_u32 reserved1;
} MRABADispatchGPU;

// World-frame force and torque about a body's center of mass. Records are
// articulation-local within each environment-major wrench block.
typedef struct MR_ALIGN16 MRABABodyWrenchGPU {
    mr_float4 force;
    mr_float4 torque;
} MRABABodyWrenchGPU;

typedef struct MR_ALIGN16 MRABAStatusGPU {
    mr_u32 code;
    mr_u32 environment;
    mr_u32 articulationIndex;
    mr_u32 failingIndex;

    mr_u32 bodyCount;
    mr_u32 nq;
    mr_u32 nv;
    mr_u32 flags;

    // Minimum/maximum factor pivot, maximum absolute acceleration, and root
    // quaternion norm error. Only successful records guarantee finite values.
    mr_float4 diagnostics;
} MRABAStatusGPU;

// MLX-native transactional wrapper around one or more ABA microsteps. The
// custom primitive passes state and effort as MLX arrays, encodes into MLX's
// active Metal command encoder, and publishes the candidate only when the ABA
// status for every preceding microstep is valid.
typedef struct MR_ALIGN16 MRMLXWorldStepDispatchGPU {
    mr_u32 environmentCount;
    mr_u32 nq;
    mr_u32 nv;
    mr_u32 physicsSubstep;

    mr_u32 physicsSubsteps;
    mr_u32 articulationIndex;
    mr_u32 reserved0;
    mr_u32 reserved1;
} MRMLXWorldStepDispatchGPU;

typedef struct MR_ALIGN16 MRMLXWorldStepStatusGPU {
    mr_u32 code;
    mr_u32 abaCode;
    mr_u32 failingSubstep;
    mr_u32 failingIndex;

    mr_u32 successfulSubsteps;
    mr_u32 requiredCapacity;
    mr_u32 flags;
    mr_u32 reserved0;
} MRMLXWorldStepStatusGPU;

// Fixed-shape semantic adapter used by the MLX contact primitive. Physics
// state itself remains in the canonical world/contact ABI; this record only
// describes MLX array extents.
typedef struct MR_ALIGN16 MRMLXContactAdapterDispatchGPU {
    mr_u32 environmentCount;
    mr_u32 sceneBodyCount;
    mr_u32 bodyStateStride;
    mr_u32 contactCapacity;

    mr_u32 manifoldCapacity;
    mr_u32 eligiblePairCount;
    mr_u32 nq;
    mr_u32 nv;
} MRMLXContactAdapterDispatchGPU;

enum MRMetalWorldFlags : mr_u32 {
    MR_METAL_WORLD_APPLY_BODY_DAMPING = 1u << 0u,
    MR_METAL_WORLD_DETERMINISTIC = 1u << 1u,
    MR_METAL_WORLD_HAS_RESETS = 1u << 2u,
    // This first graph intentionally composes free articulated motion,
    // transactional state publication, reset, and observation capture. The
    // flag prevents it from being mistaken for the future contact graph.
    MR_METAL_WORLD_FREE_MOTION_ONLY = 1u << 3u,
    // Enables the device-resident collision/manifold/constraint/island graph.
    // Exactly one of CONTACTS and FREE_MOTION_ONLY is set.
    MR_METAL_WORLD_CONTACTS = 1u << 4u,
    // The action stream carries desired generalized positions. Root and
    // unactuated coordinates are ignored; driven scalar joints use the
    // model's stiffness/damping in an implicit acceleration solve.
    MR_METAL_WORLD_IMPLICIT_POSITION_DRIVES = 1u << 5u,
    // Native task kernels own reset, control, randomization, observation,
    // reward, and termination around the physics graph.
    MR_METAL_WORLD_NATIVE_TASK = 1u << 6u,
};

// Immutable strides and dimensions for one environment-major rollout.
// Effort and output streams are control-step major. Every stride is measured
// in elements, never bytes.
typedef struct MR_ALIGN16 MRMetalWorldDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 articulationIndex;
    mr_u32 environmentCount;
    mr_u32 controlStepCount;

    mr_u32 physicsSubsteps;
    mr_u32 flags;
    mr_u32 nq;
    mr_u32 nv;

    mr_u32 qStride;
    mr_u32 vStride;
    mr_u32 effortEnvironmentStride;
    mr_u32 observationEnvironmentStride;

    mr_u32 effortStepStride;
    mr_u32 resetMaskStepStride;
    mr_u32 observationStepStride;
    mr_u32 accelerationStepStride;
} MRMetalWorldDispatchGPU;

// Small pass record copied into the command stream for prepare, transactional
// commit, and capture kernels. MR_INVALID_INDEX denotes a non-substep pass.
typedef struct MR_ALIGN16 MRMetalWorldPassGPU {
    mr_u32 controlStep;
    mr_u32 physicsSubstep;
    mr_u32 reserved0;
    mr_u32 reserved1;
} MRMetalWorldPassGPU;

// One record is reused while a control step executes, then copied to the
// public status stream. SuccessfulSubsteps counts committed state updates.
typedef struct MR_ALIGN16 MRMetalWorldStatusGPU {
    mr_u32 code;
    mr_u32 environment;
    mr_u32 controlStep;
    mr_u32 successfulSubsteps;

    mr_u32 abaCode;
    mr_u32 failingSubstep;
    mr_u32 failingIndex;
    mr_u32 flags;

    // Minimum/maximum ABA pivot, maximum absolute acceleration, and maximum
    // root-quaternion norm error across successfully committed substeps.
    mr_float4 diagnostics;
} MRMetalWorldStatusGPU;

enum MRMetalWorldContactFlags : mr_u32 {
    MR_METAL_WORLD_CONTACT_DETERMINISTIC = 1u << 0u,
    MR_METAL_WORLD_CONTACT_WARM_START = 1u << 1u,
    MR_METAL_WORLD_CONTACT_CAPTURE_EVIDENCE = 1u << 2u,
    MR_METAL_WORLD_CONTACT_HAS_KINEMATIC_TARGETS = 1u << 3u,
    MR_METAL_WORLD_CONTACT_WAVE32 = 1u << 4u,
    MR_METAL_WORLD_CONTACT_CCD = 1u << 5u,
    MR_METAL_WORLD_CONTACT_HAS_FUTURE_KINEMATICS = 1u << 6u,
    MR_METAL_WORLD_CONTACT_QUALITY = 1u << 7u,
    MR_METAL_WORLD_CONTACT_BODY_PARAMETERS = 1u << 8u,
};

// One stable, cooker-produced pair. The pair stream is canonical collider
// order and already applies static exclusions and collision masks. pairClass
// is the narrowphase class; zero is an explicit unsupported class.
typedef struct MR_ALIGN16 MRCompiledCollisionPairGPU {
    mr_u32 colliderA;
    mr_u32 colliderB;
    mr_u32 pairClass;
    mr_u32 flags;
} MRCompiledCollisionPairGPU;

// Fixed capacities and strides for the contact graph. Every stride is in
// elements, never bytes. Dynamic counts remain GPU-resident.
typedef struct MR_ALIGN16 MRMetalWorldContactDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 environmentCount;
    mr_u32 articulationIndex;
    mr_u32 solverType;

    mr_u32 bodyCount;
    mr_u32 sceneBodyCount;
    mr_u32 shapeCount;
    mr_u32 eligiblePairCount;

    mr_u32 pairCapacity;
    mr_u32 rawContactCapacity;
    mr_u32 manifoldCapacity;
    mr_u32 constraintCapacity;

    mr_u32 rowCapacity;
    mr_u32 islandCapacity;
    mr_u32 sceneBodyStride;
    mr_u32 bodyStateStride;

    mr_u32 pairStride;
    mr_u32 rawContactStride;
    mr_u32 manifoldStride;
    mr_u32 constraintStride;

    mr_u32 rowStride;
    mr_u32 islandStride;
    mr_u32 pointQueryStride;
    mr_u32 factorStride;

    mr_u32 nv;
    mr_u32 flags;
    mr_u32 velocityIterations;
    mr_u32 finalVelocityIterations;

    mr_u32 hardConvexCapacity;
    mr_u32 meshCandidateCapacity;
    mr_u32 solverTileCapacity;
    mr_u32 spillRowCapacity;

    mr_u32 ccdCandidateCapacity;
    mr_u32 ccdEventCapacity;
    mr_u32 ccdMode;
    mr_u32 maxCCDEvents;

    mr_u32 maxConservativeAdvancementIterations;
    mr_u32 workQueueClassCount;
    mr_u32 queueStride;
    mr_u32 convexCacheStride;

    mr_u32 maxCCDAdvanceSolvePasses;
    mr_u32 maxCCDZeroTimeReplays;
    mr_u32 waveWorkerGroupCount;
    mr_u32 rodToolPairCount;

    mr_u32 articulationCount;
    mr_u32 dynamicNodeCount;
    mr_u32 islandNodeReferenceCapacity;
    mr_u32 islandConstraintReferenceCapacity;

    mr_u32 rodCount;
    mr_u32 rodNodeCount;
    mr_u32 rodEdgeCount;
    mr_u32 operatorVelocityCapacity;

    mr_u32 nq;
    mr_u32 qStride;
    mr_u32 vStride;
    mr_u32 rodContactOuterIterations;

    // Immutable mechanism prefix. Runtime contact compilation appends after
    // these blocks, retaining fixed three-row/two-endpoint slots per block so
    // contact, rod, and generalized records share one capacity/addressing
    // scheme without host-visible counts.
    mr_u32 authoredConstraintCount;
    mr_u32 authoredEndpointCount;
    mr_u32 authoredRowCount;
    mr_u32 authoredConeCount;

    // timestep, penetration slop, maximum depenetration speed, warm scale.
    mr_float4 timestepAndBias;
    // separation break, tangential break, merge distance, normal cosine.
    mr_float4 manifoldThresholds;
    // minimum advancement, TOI tolerance, speculative scale, speed envelope.
    mr_float4 ccdParameters;
    // simultaneous TOI tolerance, full-time tolerance, reserved, reserved.
    mr_float4 ccdEventParameters;
} MRMetalWorldContactDispatchGPU;

// Layout-compatible with MTLDispatchThreadgroupsIndirectArguments. The final
// word is adjacent device evidence and is not consumed by Metal's indirect
// dispatch command.
typedef struct MR_ALIGN16 MRIndirectDispatchArgumentsGPU {
    mr_u32 threadgroupsX;
    mr_u32 threadgroupsY;
    mr_u32 threadgroupsZ;
    mr_u32 activeCount;
} MRIndirectDispatchArgumentsGPU;

// One header per work class. Count and requirement are produced by stable
// scans. workerCursor is used only by the MLX persistent-worker adapter;
// standalone dispatch consumes indirect directly.
typedef struct MR_ALIGN16 MRWorkQueueHeaderGPU {
    mr_u32 count;
    mr_u32 capacity;
    mr_u32 required;
    mr_u32 workClass;

    mr_u32 firstStableKeyLow;
    mr_u32 firstStableKeyHigh;
    mr_u32 overflow;
    mr_u32 workerCursor;

    MRIndirectDispatchArgumentsGPU indirect;
    mr_u32 flags;
    mr_u32 scanLevelCount;
    mr_u32 reserved0;
    mr_u32 reserved1;
} MRWorkQueueHeaderGPU;

typedef struct MR_ALIGN16 MRScanLevelGPU {
    mr_u32 elementCount;
    mr_u32 blockCount;
    mr_u32 inputOffset;
    mr_u32 outputOffset;

    mr_u32 blockSumOffset;
    mr_u32 parentOffset;
    mr_u32 workClass;
    mr_u32 flags;
} MRScanLevelGPU;

typedef struct MR_ALIGN16 MRPairWorkGPU {
    mr_u32 environment;
    mr_u32 compiledPair;
    mr_u32 cacheSlot;
    mr_u32 workClass;

    mr_u32 stableKeyLow;
    mr_u32 stableKeyHigh;
    mr_u32 flags;
    mr_u32 reserved;
} MRPairWorkGPU;

typedef struct MR_ALIGN16 MRIslandWorkGPU {
    mr_u32 environment;
    mr_u32 islandIndex;
    mr_u32 firstConstraint;
    mr_u32 constraintCount;

    mr_u32 firstTile;
    mr_u32 tileCount;
    mr_u32 dofClass;
    mr_u32 flags;
} MRIslandWorkGPU;

// Stable SIMD32 packet. The first four island slots form four wave8 cohorts,
// the first two form two wave16 cohorts, and the first slot owns wave32 or
// spill work. Packet construction is scan ordered; worker atomics only claim
// these immutable records.
typedef struct MR_ALIGN16 MRWaveWorkPacketGPU {
    mr_uint4 islandSlots;
    mr_uint4 stableKeyLow;
    mr_uint4 stableKeyHigh;
    // x cohort width, y valid cohort count, z event generation, w phase.
    mr_uint4 metadata;
} MRWaveWorkPacketGPU;

enum MRIslandWorkFlags : mr_u32 {
    MR_ISLAND_WORK_VALID = 1u << 0u,
    MR_ISLAND_WORK_HAS_ARTICULATION = 1u << 1u,
    MR_ISLAND_WORK_SPILL = 1u << 2u,
    MR_ISLAND_WORK_DISTRIBUTED = 1u << 3u,
    MR_ISLAND_WORK_STIFF_REPLAY = 1u << 4u,
    // The island owns at least one deforming rod dynamic node. Wave32 uses
    // this bit to select the typed endpoint operator and to assign unique
    // lane ownership to nodal and twist velocity updates.
    MR_ISLAND_WORK_HAS_ROD = 1u << 5u,
};

typedef struct MR_ALIGN16 MRContactTileGPU {
    mr_u32 environment;
    mr_u32 islandIndex;
    mr_u32 firstConstraint;
    mr_u32 constraintCount;

    mr_u32 nextTile;
    mr_u32 partialOffset;
    mr_u32 flags;
    mr_u32 reserved;
} MRContactTileGPU;

typedef struct MR_ALIGN16 MRWave32PreconditionerGPU {
    // xyz are rows of the inverse coupled normal/tangent response.
    mr_float4 row0;
    mr_float4 row1;
    mr_float4 row2;
} MRWave32PreconditionerGPU;

typedef struct MR_ALIGN16 MRWave32IslandStatusGPU {
    mr_u32 code;
    mr_u32 environment;
    mr_u32 islandIndex;
    mr_u32 iterations;

    // impulse delta, normal residual, cone violation, stiffness indicator.
    mr_float4 residuals;
} MRWave32IslandStatusGPU;

typedef struct MR_ALIGN16 MRCCDPairGPU {
    mr_u32 environment;
    mr_u32 compiledPair;
    mr_u32 colliderA;
    mr_u32 colliderB;

    // x lower TOI, y upper TOI, z current distance, w closing-speed bound.
    mr_float4 intervalAndDistance;
    // xyz separating/impact normal; w current iteration.
    mr_float4 normalAndIteration;
    mr_u32 stableKeyLow;
    mr_u32 stableKeyHigh;
    mr_u32 flags;
    mr_u32 status;
} MRCCDPairGPU;

enum MRCCDPairFlags : mr_u32 {
    MR_CCD_PAIR_VALID = 1u << 0u,
    MR_CCD_PAIR_START_OVERLAP = 1u << 1u,
    MR_CCD_PAIR_HAS_IMPACT = 1u << 2u,
    MR_CCD_PAIR_SPECULATIVE_FALLBACK = 1u << 3u,
    MR_CCD_PAIR_MESH = 1u << 4u,
    MR_CCD_PAIR_ANALYTIC = 1u << 5u,
    MR_CCD_PAIR_CONSERVATIVE_ADVANCEMENT = 1u << 6u,
    MR_CCD_PAIR_UNRESOLVED = 1u << 7u,
};

enum MRCCDEventStateFlags : mr_u32 {
    MR_CCD_EVENT_ACTIVE = 1u << 0u,
    MR_CCD_EVENT_FINISHED = 1u << 1u,
    MR_CCD_EVENT_HAS_IMPACT = 1u << 2u,
    MR_CCD_EVENT_SPECULATIVE_REMAINDER = 1u << 3u,
    MR_CCD_EVENT_ZERO_TIME_REPLAY = 1u << 4u,
    MR_CCD_EVENT_FAILED = 1u << 5u,
};

enum MRCCDSegmentMode : mr_u32 {
    // Predict over all time still owned by the current event cursor.
    MR_CCD_SEGMENT_REMAINING = 0u,
    // Materialize only the duration selected by the current TOI pass.
    MR_CCD_SEGMENT_SELECTED = 1u,
    // Execute the ordinary fixed microstep. This lets kernels shared by the
    // discrete and literal-event graphs retain one ABI without reading the
    // transient event cursor outside hybrid CCD.
    MR_CCD_SEGMENT_FULL_MICROSTEP = 2u,
};

// Transient per-environment event cursor. It is initialized for every
// physical microstep and ping-ponged only inside the submission; it is not
// semantic WorldState and never crosses the MLX API.
typedef struct MR_ALIGN16 MRCCDEventStateGPU {
    mr_u32 environment;
    mr_u32 splitCount;
    mr_u32 simultaneousEventCount;
    mr_u32 zeroTimeReplayCount;

    mr_u32 flags;
    mr_u32 generation;
    mr_u32 lastStableKeyLow;
    mr_u32 lastStableKeyHigh;

    // x absolute time, y remaining time, z consumed time, w selected TOI.
    mr_float4 time;
    // x first event slot, y event count, z speculative-safe, w reserved.
    mr_uint4 cluster;
} MRCCDEventStateGPU;

// Deterministic simultaneous-impact cluster selected from the sorted CCD
// candidate prefix. Every interval component is in physical seconds relative
// to the current event state, never normalized time.
typedef struct MR_ALIGN16 MRCCDImpactClusterGPU {
    mr_u32 environment;
    mr_u32 firstEventSlot;
    mr_u32 eventCount;
    mr_u32 generation;

    mr_u32 stableKeyLow;
    mr_u32 stableKeyHigh;
    mr_u32 flags;
    mr_u32 reserved0;

    // x selected TOI, y lower bound, z upper bound, w tolerance.
    mr_float4 interval;
} MRCCDImpactClusterGPU;

// Environment-major immutable-for-one-microstep collider projection. This
// record is produced once per collider, then reused by broadphase and
// narrowphase instead of reconstructing shape frames for every eligible pair.
typedef struct MR_ALIGN16 MRProjectedColliderGPU {
    // projection status, simulation-disabled flag, reserved, reserved.
    mr_uint4 statusAndFlags;
    // xyz center, w radius.
    mr_float4 centerAndRadius;
    mr_float4 rotation;
    // xyz lower AABB, w half length.
    mr_float4 lowerAndHalfLength;
    // xyz upper AABB, w contact offset.
    mr_float4 upperAndContactOffset;
} MRProjectedColliderGPU;

// Solver/contact metadata for one active manifold point. Anchors are
// COM-relative body-local points. Query slots are deterministic:
// 2*constraint and 2*constraint+1.
typedef struct MR_ALIGN16 MRContactPointMetaGPU {
    mr_u32 colliderA;
    mr_u32 colliderB;
    mr_u32 manifoldIndex;
    mr_u32 pointIndex;

    mr_float4 localAnchorA;
    mr_float4 localAnchorB;
} MRContactPointMetaGPU;

// Pair-owned count and prefix record for deterministic manifold-to-IR
// compilation. Narrowphase/finalization writes counts into fixed compiled-pair
// slots, one SIMDgroup per environment computes stable exclusive prefixes,
// and scatter writes only when the complete environment fits every arena.
// This removes atomics from semantic ordering and keeps exact overflow
// requirements device-resident.
typedef struct MR_ALIGN16 MRManifoldIRScatterGPU {
    // candidate pairs, raw witnesses, manifolds, constraint blocks.
    mr_uint4 counts0;
    // rows, endpoints, point queries, evidence records.
    mr_uint4 counts1;
    // Exclusive prefixes corresponding to counts0.
    mr_uint4 offsets0;
    // Exclusive prefixes corresponding to counts1.
    mr_uint4 offsets1;
    // status, CCD event slot, retained points, new points.
    mr_uint4 diagnostics0;
    // hard-convex requirement, mesh candidates, fallbacks, first failure.
    mr_uint4 diagnostics1;
    // maximum penetration and reserved finite diagnostics.
    mr_float4 metrics;
    mr_uint4 reserved;
} MRManifoldIRScatterGPU;

// Canonical dynamic endpoint represented in the heterogeneous island graph.
// One articulation tree, one maximal-coordinate free body, or one connected
// rod component owns exactly one node. Static and kinematic endpoints never
// allocate nodes. Semantic q/v and rod arrays remain environment-major; the
// velocity offsets below address the immutable flattened world layout.
enum MRWorldDynamicNodeKind : mr_u32 {
    MR_WORLD_DYNAMIC_NODE_ARTICULATION = 0u,
    MR_WORLD_DYNAMIC_NODE_FREE_BODY = 1u,
    MR_WORLD_DYNAMIC_NODE_ROD = 2u,
};

enum MRWorldDynamicNodeFlags : mr_u32 {
    MR_WORLD_DYNAMIC_NODE_VALID = 1u << 0u,
    MR_WORLD_DYNAMIC_NODE_FLOATING = 1u << 1u,
    MR_WORLD_DYNAMIC_NODE_SLEEPING = 1u << 2u,
    MR_WORLD_DYNAMIC_NODE_HAS_IMPLICIT_FACTOR = 1u << 3u,
};

typedef struct MR_ALIGN16 MRWorldDynamicNodeGPU {
    mr_u32 environment;
    mr_u32 stableId;
    mr_u32 kind;
    mr_u32 ownerIndex;

    mr_u32 velocityOffset;
    mr_u32 velocityCount;
    mr_u32 configurationOffset;
    mr_u32 configurationCount;

    mr_u32 factorIndex;
    mr_u32 generation;
    mr_u32 operatorBucket;
    mr_u32 flags;

    mr_u32 reserved0;
    mr_u32 reserved1;
    mr_u32 reserved2;
    mr_u32 reserved3;
} MRWorldDynamicNodeGPU;

// Stable compact reference emitted after deterministic island root
// construction. localVelocityOffset is relative to the island's gathered
// generalized-velocity vector; dynamicNode addresses the immutable node table.
typedef struct MR_ALIGN16 MRIslandNodeRefGPU {
    mr_u32 dynamicNode;
    mr_u32 localVelocityOffset;
    mr_u32 velocityCount;
    mr_u32 factorIndex;
} MRIslandNodeRefGPU;

// ConstraintIR blocks remain canonical and are never physically reordered.
// Islands therefore compact references rather than copying semantic records.
typedef struct MR_ALIGN16 MRIslandConstraintRefGPU {
    mr_u32 blockIndex;
    mr_u32 rowOffset;
    mr_u32 rowCount;
    mr_u32 flags;
} MRIslandConstraintRefGPU;

typedef struct MR_ALIGN16 MRContactIslandGPU {
    mr_u32 environment;
    mr_u32 stableRoot;
    mr_u32 firstConstraint;
    mr_u32 constraintCount;

    mr_u32 dynamicNodeCount;
    mr_u32 flags;
    mr_u32 firstNode;
    mr_u32 firstRow;

    mr_u32 generalizedVelocityOffset;
    mr_u32 generalizedVelocityCount;
    mr_u32 articulationNodeCount;
    mr_u32 freeBodyNodeCount;

    mr_u32 rodNodeCount;
    mr_u32 operatorBucket;
    mr_u32 generation;
    mr_u32 reserved;
} MRContactIslandGPU;

typedef struct MR_ALIGN16 MRArticulationFactorCacheGPU {
    mr_u32 environment;
    mr_u32 articulationIndex;
    mr_u32 nv;
    mr_u32 generation;

    mr_u32 code;
    mr_u32 failingIndex;
    mr_u32 reserved0;
    mr_u32 reserved1;

    // Minimum pivot, maximum pivot, relative residual, maximum mass entry.
    mr_float4 diagnostics;
} MRArticulationFactorCacheGPU;

// Per-environment evidence. Counts are exact even when a capacity overflows;
// candidate physical/manifold state is then discarded transactionally.
typedef struct MR_ALIGN16 MRMetalWorldContactStatusGPU {
    mr_u32 code;
    mr_u32 environment;
    mr_u32 controlStep;
    mr_u32 physicsSubstep;

    mr_u32 requiredPairs;
    mr_u32 requiredRawContacts;
    mr_u32 requiredManifolds;
    mr_u32 requiredConstraints;

    mr_u32 requiredRows;
    mr_u32 requiredIslands;
    mr_u32 activePairs;
    mr_u32 activeContacts;

    mr_u32 retainedPoints;
    mr_u32 newPoints;
    mr_u32 islandCount;
    mr_u32 spillRows;

    mr_u32 firstFailingPair;
    mr_u32 firstFailingConstraint;
    mr_u32 solverIterations;
    mr_u32 flags;

    mr_u32 requiredHardConvexPairs;
    mr_u32 requiredMeshCandidates;
    mr_u32 requiredSolverTiles;
    mr_u32 requiredSpillRows;

    mr_u32 requiredCCDCandidates;
    mr_u32 requiredCCDEvents;
    mr_u32 hardConvexPairs;
    mr_u32 meshCandidates;

    mr_u32 solverTiles;
    mr_u32 ccdCandidates;
    mr_u32 ccdEvents;
    mr_u32 hardFallbacks;

    mr_u32 firstFailingStableKeyLow;
    mr_u32 firstFailingStableKeyHigh;
    mr_u32 unresolvedCCDCount;
    mr_u32 queueFlags;

    mr_u32 ccdAdvanceCount;
    mr_u32 clusteredCCDImpacts;
    mr_u32 zeroTimeCCDReplays;
    mr_u32 speculativeRemainderUses;

    mr_u32 workerPackets;
    mr_u32 workerEmptyPulls;
    mr_u32 workerHighWater;
    mr_u32 eventGeneration;

    mr_u32 firstFailingEventKeyLow;
    mr_u32 firstFailingEventKeyHigh;
    mr_u32 reservedEvent0;
    mr_u32 reservedEvent1;

    mr_u32 qualityNewtonIterations;
    mr_u32 qualityPCGIterations;
    mr_u32 qualityLineSearchBacktracks;
    mr_u32 qualitySolvePath;

    // impulse delta, normal residual, cone violation, factor residual.
    mr_float4 residuals;
    // manifold retention, maximum penetration, minimum pivot, maximum pivot.
    mr_float4 diagnostics;
    // consumed time, remaining time, earliest TOI, selected TOI.
    mr_float4 eventTimes;
    // optimality, cone/scalar feasibility, equality feasibility,
    // complementarity.
    mr_float4 qualityCertificates;
    // dynamics backward error, Newton decrement, objective change,
    // effective compliance/regularization.
    mr_float4 qualityDiagnostics;
} MRMetalWorldContactStatusGPU;

enum MRInverseMassStatusCode : mr_u32 {
    MR_INVERSE_MASS_SUCCESS = 0u,
    MR_INVERSE_MASS_INVALID_DISPATCH = 1u,
    MR_INVERSE_MASS_INVALID_MODEL = 2u,
    MR_INVERSE_MASS_NONFINITE_INPUT = 3u,
    MR_INVERSE_MASS_FACTORIZATION_FAILED = 4u,
    MR_INVERSE_MASS_NONFINITE_RESULT = 5u,
    MR_INVERSE_MASS_INVALID_QUATERNION = 6u,
    MR_INVERSE_MASS_UNSUPPORTED_TOPOLOGY = 7u,
};

// Applies one immutable articulation's generalized inverse mass to one to
// three right-hand sides per environment. All strides are float element
// counts. Each environment contains rhsCount vectors separated by
// rhsVectorStride/outputVectorStride.
typedef struct MR_ALIGN16 MRInverseMassDispatchGPU {
    mr_u32 articulationIndex;
    mr_u32 environmentCount;
    mr_u32 rhsCount;
    mr_u32 reserved0;

    mr_u32 qStride;
    mr_u32 rhsEnvironmentStride;
    mr_u32 rhsVectorStride;
    mr_u32 outputEnvironmentStride;

    mr_u32 outputVectorStride;
    mr_u32 reserved1;
    mr_u32 reserved2;
    mr_u32 reserved3;
} MRInverseMassDispatchGPU;

typedef struct MR_ALIGN16 MRInverseMassStatusGPU {
    mr_u32 code;
    mr_u32 environment;
    mr_u32 articulationIndex;
    mr_u32 failingIndex;

    mr_u32 bodyCount;
    mr_u32 nq;
    mr_u32 nv;
    mr_u32 rhsCount;

    // Minimum/maximum factor pivot, maximum absolute output, and maximum
    // absolute input. Only successful records guarantee finite values.
    mr_float4 diagnostics;
} MRInverseMassStatusGPU;

typedef struct MR_ALIGN16 MRMaterialGPU {
    // Static/dynamic coefficients; effective rolling/torsional lengths (m).
    mr_float4 friction;
    // restitution, restitution velocity threshold, compliance, dissipation.
    mr_float4 response;
    // contact skin width, adhesion impulse cap, reserved, reserved.
    mr_float4 geometry;
} MRMaterialGPU;

// Immutable cooker output. All offsets address the corresponding typed
// EngineModel arena; no device pointer is stored in the ABI.
typedef struct MR_ALIGN16 MRGeometryHeaderGPU {
    mr_u32 kind;
    mr_u32 flags;
    mr_u32 vertexOffset;
    mr_u32 vertexCount;

    mr_u32 indexOffset;
    mr_u32 indexCount;
    mr_u32 faceOffset;
    mr_u32 faceCount;

    mr_u32 halfEdgeOffset;
    mr_u32 halfEdgeCount;
    mr_u32 bvhOffset;
    mr_u32 bvhCount;

    mr_u32 triangleOffset;
    mr_u32 triangleCount;
    mr_u32 materialOffset;
    mr_u32 materialCount;

    mr_float4 localLower;
    mr_float4 localUpper;
} MRGeometryHeaderGPU;

typedef struct MR_ALIGN16 MRConvexFaceGPU {
    // xyz = outward unit normal, w = plane offset dot(n, x).
    mr_float4 plane;
    mr_u32 firstHalfEdge;
    mr_u32 halfEdgeCount;
    mr_u32 featureKey;
    mr_u32 reserved;
} MRConvexFaceGPU;

typedef struct MR_ALIGN16 MRConvexHalfEdgeGPU {
    mr_u32 originVertex;
    mr_u32 twinHalfEdge;
    mr_u32 nextHalfEdge;
    mr_u32 faceIndex;
} MRConvexHalfEdgeGPU;

// Four-child, stackless mesh node. Child bounds are uint16 coordinates
// packed in uint4 records and are conservatively decompressed against the
// geometry header bounds. childMeta encodes leaf counts and escape links.
typedef struct MR_ALIGN16 MRMeshBVHNodeGPU {
    mr_uint4 quantizedLower[MR_MESH_BVH_BRANCHING];
    mr_uint4 quantizedUpper[MR_MESH_BVH_BRANCHING];
    mr_uint4 childIndices;
    mr_uint4 childMeta;
} MRMeshBVHNodeGPU;

typedef struct MR_ALIGN16 MRMeshTriangleGPU {
    // xyz are vertex indices; w is the stable triangle feature ID.
    mr_uint4 verticesAndFeature;
    // xyz are adjacent triangle indices or MR_INVALID_INDEX; w is edge mask.
    mr_uint4 adjacencyAndEdges;
    // x material, y flags, z/w reserved.
    mr_uint4 materialAndFlags;
} MRMeshTriangleGPU;

// Explicit persistent support/simplex cache. It is semantic state for MLX:
// callers carry it between pure primitive invocations.
typedef struct MR_ALIGN16 MRConvexQueryCacheGPU {
    mr_float4 separatingAxisAndDistance;
    mr_uint4 supportA;
    mr_uint4 supportB;
    mr_float4 barycentricWeights;
    mr_uint4 featureAndStatus;
} MRConvexQueryCacheGPU;

typedef struct MR_ALIGN16 MRShapeGPU {
    mr_u32 bodyIndex;
    mr_u32 shapeType;
    mr_u32 materialIndex;
    mr_u32 flags;

    mr_u32 collisionGroup;
    mr_u32 collisionMask;
    mr_u32 slotGeneration;
    mr_u32 geometryOffset;

    mr_u32 geometryCount;
    mr_u32 reserved0;
    mr_u32 reserved1;
    mr_u32 reserved2;

    // Position relative to the body's COM-centred state origin.
    mr_float4 localPosition;
    // Normalizable local quaternion under the same collision contract.
    mr_float4 localRotation;
    // Sphere: x=radius. Capsule/cylinder: x=radius, y=half length.
    // Box: xyz=half extents. Mesh/convex: xyz=local scale.
    mr_float4 dimensions;
    // contact offset, rest offset, conservative bounding radius, reserved.
    mr_float4 contactRestAndBoundingRadius;
} MRShapeGPU;

typedef struct MR_ALIGN16 MRAabbGPU {
    mr_float4 lower;
    mr_float4 upper;
} MRAabbGPU;

typedef struct MR_ALIGN16 MRCandidatePairGPU {
    mr_u32 environment;
    mr_u32 colliderA;
    mr_u32 colliderB;
    mr_u32 flags;
} MRCandidatePairGPU;

// Deterministic flag/scan/emit broadphase dispatch. `logicalPairCount` is
// shapeCount * (shapeCount - 1) / 2 and `scanBlockCount` is its ceiling
// division by MR_BROADPHASE_SCAN_BLOCK_SIZE. Current block-sum scan capacity
// is explicit; larger scenes must be partitioned or use a recursive scan.
typedef struct MR_ALIGN16 MRBroadphaseDispatchGPU {
    mr_u32 shapeCount;
    mr_u32 bodyCount;
    mr_u32 logicalPairCount;
    mr_u32 scanBlockCount;

    mr_u32 pairCapacity;
    mr_u32 exclusionCount;
    mr_u32 environment;
    mr_u32 flags;
} MRBroadphaseDispatchGPU;

typedef struct MR_ALIGN16 MRBroadphaseStatusGPU {
    mr_u32 code;
    mr_u32 requiredPairs;
    mr_u32 emittedPairs;
    mr_u32 logicalPairs;
} MRBroadphaseStatusGPU;

// Transient geometric witness record. The solver consumes a separately
// reduced MRContactConstraintGPU so manifold refresh never loses surface data.
typedef struct MR_ALIGN16 MRRawContactGPU {
    // xyz = normal A->B, w = geometric separation.
    mr_float4 normalAndSeparation;
    mr_float4 pointAWorld;
    mr_float4 pointBWorld;
    // feature A, feature B, patch seed, flags.
    mr_u32 featureAndFlags[4];
} MRRawContactGPU;

typedef struct MR_ALIGN16 MRManifoldHeaderGPU {
    // environment, collider A, collider B, point count.
    mr_u32 pairAndCount[4];
    // generation A, generation B, patch id, flags.
    mr_u32 generationsAndFlags[4];
    // xyz = persistent normal, w = frames since full rebuild.
    mr_float4 normalAndAge;
    // xyz = stable tangent, w = manifold breaking metric.
    mr_float4 tangentAndMetric;
} MRManifoldHeaderGPU;

typedef struct MR_ALIGN16 MRManifoldPointGPU {
    mr_float4 localAnchorA;
    mr_float4 localAnchorB;
    // normal, tangent-u, tangent-v, rolling impulse.
    mr_float4 impulses;
    // feature A, feature B, lifetime, flags.
    mr_u32 featureAndLife[4];
} MRManifoldPointGPU;

typedef struct MR_ALIGN16 MRConstraintBlockGPU {
    mr_u32 type;
    mr_u32 dimension;
    mr_u32 flags;
    mr_u32 islandIndex;

    mr_u32 bodyA;
    mr_u32 bodyB;
    mr_u32 rowOffset;
    mr_u32 impulseOffset;

    mr_u64 pairKey;
    mr_u64 featureKey;
} MRConstraintBlockGPU;

typedef struct MR_ALIGN16 MRContactConstraintGPU {
    mr_u32 bodyA;
    mr_u32 bodyB;
    mr_u32 flags;
    mr_u32 islandIndex;

    mr_u64 pairKey;
    mr_u64 featureKey;

    // xyz = world contact point, w = signed separation (negative overlaps).
    mr_float4 pointAndSeparation;
    // xyz = unit normal from body A toward body B.
    mr_float4 normal;
    // xyz = unit tangent-u used by the impulse y component. Tangent-v is
    // cross(normal, tangent). Retaining the basis makes solved impulses
    // directly interpretable by tactile, wrench, and logging stages.
    mr_float4 tangent;
    // Static/dynamic coefficients; effective rolling/torsional lengths (m).
    mr_float4 friction;
    // restitution, threshold, compliance, maximum normal impulse (0=unbounded).
    mr_float4 response;
    // xyz = target relative surface velocity; w = pre-solve normal velocity.
    mr_float4 targetVelocityAndPreSolveNormal;
    // normal, tangent-u, tangent-v, torsional impulses.
    mr_float4 impulses;
} MRContactConstraintGPU;

typedef struct MR_ALIGN16 MRSolverBatchGPU {
    mr_u32 bodyOffset;
    mr_u32 bodyCount;
    mr_u32 contactOffset;
    mr_u32 contactCount;

    mr_u32 velocityIterations;
    mr_u32 enableWarmStart;
    mr_u32 enableEarlyExit;
    mr_u32 deterministic;

    // timestep, error reduction, penetration slop, max depenetration velocity.
    mr_float4 timestepAndBias;
    // impulse tolerance, warm-start scale, minimum inverse linear effective
    // mass, minimum inverse angular effective mass.
    mr_float4 convergence;
} MRSolverBatchGPU;

typedef struct MR_ALIGN16 MRSolverStatusGPU {
    mr_u32 code;
    mr_u32 iterations;
    mr_u32 activeContacts;
    mr_u32 islandCount;

    // Max impulse delta, normal residual, cone violation, and dimensionless
    // inverse-linear-effective-mass spread.
    mr_float4 residuals;
    // Required capacities when code reports overflow.
    mr_u32 requiredPairs;
    mr_u32 requiredContacts;
    mr_u32 requiredConstraints;
    mr_u32 requiredIslands;
} MRSolverStatusGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(MRWorldGPU) % 16 == 0);
static_assert(sizeof(MRArticulationGPU) % 16 == 0);
static_assert(sizeof(MRJointDescriptorGPU) % 16 == 0);
static_assert(sizeof(MRDofPropertiesGPU) == 64);
static_assert(alignof(MRDofPropertiesGPU) == 16);
static_assert(__builtin_offsetof(MRDofPropertiesGPU, localDof) == 16);
static_assert(__builtin_offsetof(MRDofPropertiesGPU, limits) == 32);
static_assert(__builtin_offsetof(MRDofPropertiesGPU, drive) == 48);
static_assert(sizeof(MRActuatorProfileGPU) == 48);
static_assert(alignof(MRActuatorProfileGPU) == 16);
static_assert(
    __builtin_offsetof(
        MRActuatorProfileGPU,
        transmissionAndEnvelope
    ) == 16
);
static_assert(
    __builtin_offsetof(MRActuatorProfileGPU, identity) == 32
);
static_assert(sizeof(MRBodyPropertiesGPU) % 16 == 0);
static_assert(sizeof(MRBodyStateGPU) % 16 == 0);
static_assert(sizeof(MRBodyWrenchGPU) == 32);
static_assert(sizeof(MRFreeBodyBatchGPU) % 16 == 0);
static_assert(sizeof(MRFreeBodyStatusGPU) % 16 == 0);
static_assert(sizeof(MRArticulatedOperatorDispatchGPU) == 48);
static_assert(sizeof(MRArticulatedPointImpulseGPU) == 48);
static_assert(sizeof(MRArticulatedBodyPoseGPU) == 32);
static_assert(sizeof(MRArticulatedPointWorldGPU) == 16);
static_assert(sizeof(MRArticulatedOperatorStatusGPU) == 48);
static_assert(sizeof(MRABADispatchGPU) == 48);
static_assert(sizeof(MRABABodyWrenchGPU) == 32);
static_assert(sizeof(MRABAStatusGPU) == 48);
static_assert(sizeof(MRMLXWorldStepDispatchGPU) == 32);
static_assert(sizeof(MRMLXWorldStepStatusGPU) == 32);
static_assert(sizeof(MRMLXContactAdapterDispatchGPU) == 32);
static_assert(sizeof(MRMetalWorldDispatchGPU) == 64);
static_assert(sizeof(MRMetalWorldPassGPU) == 16);
static_assert(sizeof(MRMetalWorldStatusGPU) == 48);
static_assert(sizeof(MRCompiledCollisionPairGPU) == 16);
static_assert(sizeof(MRMetalWorldContactDispatchGPU) == 304);
static_assert(sizeof(MRIndirectDispatchArgumentsGPU) == 16);
static_assert(sizeof(MRWorkQueueHeaderGPU) == 64);
static_assert(sizeof(MRScanLevelGPU) == 32);
static_assert(sizeof(MRPairWorkGPU) == 32);
static_assert(sizeof(MRIslandWorkGPU) == 32);
static_assert(sizeof(MRWaveWorkPacketGPU) == 64);
static_assert(sizeof(MRContactTileGPU) == 32);
static_assert(sizeof(MRWave32PreconditionerGPU) == 48);
static_assert(sizeof(MRWave32IslandStatusGPU) == 32);
static_assert(sizeof(MRCCDPairGPU) == 64);
static_assert(sizeof(MRCCDEventStateGPU) == 64);
static_assert(sizeof(MRCCDImpactClusterGPU) == 48);
static_assert(sizeof(MRProjectedColliderGPU) == 80);
static_assert(sizeof(MRContactPointMetaGPU) == 48);
static_assert(sizeof(MRManifoldIRScatterGPU) == 128);
static_assert(sizeof(MRWorldDynamicNodeGPU) == 64);
static_assert(sizeof(MRIslandNodeRefGPU) == 16);
static_assert(sizeof(MRIslandConstraintRefGPU) == 16);
static_assert(sizeof(MRContactIslandGPU) == 64);
static_assert(sizeof(MRArticulationFactorCacheGPU) == 48);
static_assert(sizeof(MRMetalWorldContactStatusGPU) == 288);
static_assert(sizeof(MRInverseMassDispatchGPU) == 48);
static_assert(sizeof(MRInverseMassStatusGPU) == 48);
static_assert(sizeof(MRMaterialGPU) % 16 == 0);
static_assert(sizeof(MRGeometryHeaderGPU) == 96);
static_assert(sizeof(MRConvexFaceGPU) == 32);
static_assert(sizeof(MRConvexHalfEdgeGPU) == 16);
static_assert(sizeof(MRMeshBVHNodeGPU) == 160);
static_assert(sizeof(MRMeshTriangleGPU) == 48);
static_assert(sizeof(MRConvexQueryCacheGPU) == 80);
static_assert(sizeof(MRShapeGPU) % 16 == 0);
static_assert(sizeof(MRAabbGPU) == 32);
static_assert(sizeof(MRCandidatePairGPU) == 16);
static_assert(sizeof(MRBroadphaseDispatchGPU) == 32);
static_assert(sizeof(MRBroadphaseStatusGPU) == 16);
static_assert(sizeof(MRRawContactGPU) == 64);
static_assert(sizeof(MRManifoldHeaderGPU) == 64);
static_assert(sizeof(MRManifoldPointGPU) == 64);
static_assert(sizeof(MRConstraintBlockGPU) % 16 == 0);
static_assert(sizeof(MRContactConstraintGPU) % 16 == 0);
static_assert(sizeof(MRSolverBatchGPU) % 16 == 0);
static_assert(sizeof(MRSolverStatusGPU) % 16 == 0);
#endif
