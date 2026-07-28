#pragma once

// Versioned, pointer-free ABI shared by C++, Objective-C++, and Metal.
// The old MRModelGPU path remains available while the generic engine is
// brought online. New engine code must use offsets/counts and report capacity
// failures. This is the current hard capacity of one connected throughput
// island. Scene-level pipelines partition independent islands into separate
// batches; a connected island above this limit returns an explicit overflow.

#include "metalrobo/gpu_types.h"

#define MR_ENGINE_ABI_VERSION 1u
#define MR_INVALID_INDEX 0xffffffffu
#define MR_MAX_CONTACTS_PER_SOLVER_BATCH 128u
#define MR_MAX_BODIES_PER_SOLVER_BATCH \
    (2u * MR_MAX_CONTACTS_PER_SOLVER_BATCH)
#define MR_BROADPHASE_SCAN_BLOCK_SIZE 256u
#define MR_MAX_BROADPHASE_SCAN_BLOCKS 256u
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
};

enum MRConstraintFlags : mr_u32 {
    MR_CONSTRAINT_FLAG_NEW_IMPACT = 1u << 0u,
    MR_CONSTRAINT_FLAG_WARM_STARTED = 1u << 1u,
    MR_CONSTRAINT_FLAG_DISABLED = 1u << 2u,
};

enum MRShapeFlags : mr_u32 {
    // Retain factual/cooked geometry in the model while excluding it from
    // simulation until its narrowphase is executable.
    MR_SHAPE_FLAG_SIMULATION_DISABLED = 1u << 0u,
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

typedef struct MR_ALIGN16 MRMaterialGPU {
    // Static/dynamic coefficients; effective rolling/torsional lengths (m).
    mr_float4 friction;
    // restitution, restitution velocity threshold, compliance, dissipation.
    mr_float4 response;
    // contact skin width, adhesion impulse cap, reserved, reserved.
    mr_float4 geometry;
} MRMaterialGPU;

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
static_assert(sizeof(MRBodyPropertiesGPU) % 16 == 0);
static_assert(sizeof(MRBodyStateGPU) % 16 == 0);
static_assert(sizeof(MRBodyWrenchGPU) == 32);
static_assert(sizeof(MRFreeBodyBatchGPU) % 16 == 0);
static_assert(sizeof(MRFreeBodyStatusGPU) % 16 == 0);
static_assert(sizeof(MRMaterialGPU) % 16 == 0);
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
