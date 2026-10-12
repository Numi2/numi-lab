#include <metal_stdlib>

using namespace metal;
typedef uint mr_u32;
typedef int mr_i32;
typedef ulong mr_u64;
typedef float4 mr_float4;
typedef uint4 mr_uint4;
enum MRShapeType : mr_u32 {
    MR_SHAPE_SPHERE = 0u,
    MR_SHAPE_CAPSULE = 1u,
    MR_SHAPE_BOX = 2u,
    MR_SHAPE_PLANE = 3u,
    MR_SHAPE_CYLINDER = 4u,
    MR_SHAPE_CONVEX = 5u,
    MR_SHAPE_TRIANGLE_MESH = 6u,
    MR_SHAPE_HEIGHTFIELD = 7u,
    MR_SHAPE_SDF = 8u,
};
typedef struct alignas(16) MRModelGPU {
    mr_u32 dofCount;
    mr_u32 linkCount;
    mr_u32 colliderCount;
    mr_u32 observationCount;
    mr_u32 actionCount;
    mr_u32 episodeHorizon;
    mr_u32 substeps;
    mr_u32 flags;
    mr_float4 gravityAndTimestep;
    mr_float4 groundPlane;
    mr_float4 targetLowerAndRadius;
    mr_float4 targetUpperAndBonus;
    mr_float4 rewardScales;
} MRModelGPU;
typedef struct alignas(16) MRJointGPU {
    mr_i32 parentLink;
    mr_u32 childLink;
    mr_u32 jointType;
    mr_u32 reserved;
    mr_float4 axis;
    mr_float4 parentOffset;
    mr_float4 parentRotation;
    mr_float4 limits;
    mr_float4 drive;
} MRJointGPU;
typedef struct alignas(16) MRLinkGPU {
    mr_float4 massAndCOMX;
    mr_float4 inertiaRow0;
    mr_float4 inertiaRow1;
    mr_float4 inertiaRow2;
} MRLinkGPU;
typedef struct alignas(16) MRColliderGPU {
    mr_i32 linkIndex;
    mr_u32 shapeType;
    mr_u32 collisionGroup;
    mr_u32 collisionMask;
    mr_float4 centerAndRadius;
    mr_float4 extent;
    mr_float4 material;
} MRColliderGPU;
typedef struct alignas(16) MRStepUniformsGPU {
    mr_u32 environmentCount;
    mr_u32 seedLo;
    mr_u32 seedHi;
    mr_u32 resetAll;
    mr_u32 autoReset;
    mr_u32 captureBodyPoses;
    mr_u32 reserved0;
    mr_u32 reserved1;
} MRStepUniformsGPU;
typedef struct alignas(16) MRCompensatedRootTranslationGPU {
    mr_float4 reference;
    mr_float4 displacement;
    mr_float4 correction;
} MRCompensatedRootTranslationGPU;
struct MRCompensatedScalar { float high; float low; };
inline MRCompensatedScalar mrCompensatedSum(float a, float b) {
    const float high = a + b;
    const float virtualB = high - a;
    const float low = (a - (high - virtualB)) + (b - virtualB);
    return {high, low};
}
inline float mrCompensatedRoundOdd(MRCompensatedScalar value);
inline MRCompensatedScalar mrCompensatedAdd(
    MRCompensatedScalar a, MRCompensatedScalar b) {
    const auto high = mrCompensatedSum(a.high, b.high);
    const auto low = mrCompensatedSum(a.low, b.low);
    const auto middle = mrCompensatedSum(high.low, low.high);
    const auto merged = mrCompensatedSum(high.high, middle.high);
    const auto tail0 = mrCompensatedSum(merged.low, middle.low);
    const auto tail1 = mrCompensatedSum(tail0.high, low.low);
    const float sticky = mrCompensatedRoundOdd(mrCompensatedSum(tail1.low, tail0.low));
    const float tail = mrCompensatedRoundOdd(mrCompensatedSum(tail1.high, sticky));
    return mrCompensatedSum(merged.high, tail);
}
inline float mrCompensatedFMA(float a, float b, float c) {
    return fma(a, b, c);
}
inline MRCompensatedScalar mrCompensatedProduct(float a, float b) {
    const float high = a * b;
    return {high, mrCompensatedFMA(a, b, -high)};
}
inline bool mrCompensatedFinite(float value) {
    return isfinite(value);
}
inline mr_u32 mrCompensatedBits(float value) {
    return as_type<uint>(value);
}
inline MRCompensatedScalar mrCompensatedPositionScalar(
    float reference, float displacement, float correction, float local) {
    if (displacement == 0.0f && correction == 0.0f && local == 0.0f)
        return {reference, 0.0f};
    return mrCompensatedAdd(mrCompensatedSum(reference, local),
                           {displacement, correction});
}
struct MRCompensatedPositionGPU { mr_float4 high; mr_float4 low; };
inline MRCompensatedPositionGPU mrCompensatedTranslationPosition(
    const MRCompensatedRootTranslationGPU value, mr_float4 local) {
    const auto x = mrCompensatedPositionScalar(value.reference.x, value.displacement.x,
                                              value.correction.x, local.x);
    const auto y = mrCompensatedPositionScalar(value.reference.y, value.displacement.y,
                                              value.correction.y, local.y);
    const auto z = mrCompensatedPositionScalar(value.reference.z, value.displacement.z,
                                              value.correction.z, local.z);
    return {{x.high, y.high, z.high, 0.0f}, {x.low, y.low, z.low, 0.0f}};
}
inline float mrCompensatedRoundOdd(MRCompensatedScalar value) {
    if (value.low == 0.0f || (mrCompensatedBits(value.high) & 1u) != 0u)
        return value.high;
    return nextafter(value.high, value.low > 0.0f ? INFINITY : -INFINITY);
}
inline float mrCompensatedProjectionScalar(float reference, float displacement, float correction) {
    if (displacement == 0.0f && correction == 0.0f) return reference;
    const auto first = mrCompensatedSum(reference, displacement);
    const auto second = mrCompensatedSum(first.low, correction);
    const auto third = mrCompensatedSum(first.high, second.high);
    const float tail = mrCompensatedRoundOdd(mrCompensatedSum(third.low, second.low));
    return third.high + tail;
}
inline mr_float4 mrCompensatedTranslationProjection(
    const MRCompensatedRootTranslationGPU value) {
    return {
        mrCompensatedProjectionScalar(value.reference.x, value.displacement.x, value.correction.x),
        mrCompensatedProjectionScalar(value.reference.y, value.displacement.y, value.correction.y),
        mrCompensatedProjectionScalar(value.reference.z, value.displacement.z, value.correction.z),
        0.0f};
}
inline MRCompensatedRootTranslationGPU mrCompensatedTranslationFromProjection(
    mr_float4 position) {
    position.w = 0.0f;
    return {position, {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
}
inline bool mrCompensatedCanonicalScalar(float high, float low) {
    if (!mrCompensatedFinite(high) || !mrCompensatedFinite(low)) return false;
    const auto pair = mrCompensatedSum(high, low);
    return mrCompensatedBits(pair.high) == mrCompensatedBits(high) &&
           mrCompensatedBits(pair.low) == mrCompensatedBits(low);
}
inline bool mrCompensatedTranslationValid(
    const MRCompensatedRootTranslationGPU value) {
    const auto projected = mrCompensatedTranslationProjection(value);
    return mrCompensatedFinite(value.reference.x) &&
        mrCompensatedFinite(value.reference.y) &&
        mrCompensatedFinite(value.reference.z) &&
        mrCompensatedBits(value.reference.w) == 0u &&
        mrCompensatedBits(value.displacement.w) == 0u &&
        mrCompensatedBits(value.correction.w) == 0u &&
        mrCompensatedCanonicalScalar(value.displacement.x, value.correction.x) &&
        mrCompensatedCanonicalScalar(value.displacement.y, value.correction.y) &&
        mrCompensatedCanonicalScalar(value.displacement.z, value.correction.z) &&
        mrCompensatedFinite(projected.x) && mrCompensatedFinite(projected.y) &&
        mrCompensatedFinite(projected.z);
}
inline MRCompensatedRootTranslationGPU mrCompensatedTranslationAdvance(
    const MRCompensatedRootTranslationGPU source, mr_float4 velocity, float dt) {
    MRCompensatedRootTranslationGPU result = source;
    const auto x = mrCompensatedAdd({source.displacement.x, source.correction.x},
                                    mrCompensatedProduct(dt, velocity.x));
    const auto y = mrCompensatedAdd({source.displacement.y, source.correction.y},
                                    mrCompensatedProduct(dt, velocity.y));
    const auto z = mrCompensatedAdd({source.displacement.z, source.correction.z},
                                    mrCompensatedProduct(dt, velocity.z));
    result.displacement = {x.high, y.high, z.high, 0.0f};
    result.correction = {x.low, y.low, z.low, 0.0f};
    return result;
}
inline mr_float4 mrCompensatedPositionDifference(mr_float4 ah, mr_float4 al,
                                                 mr_float4 bh, mr_float4 bl) {
    const auto x = mrCompensatedAdd(mrCompensatedSum(ah.x, -bh.x), mrCompensatedSum(al.x, -bl.x));
    const auto y = mrCompensatedAdd(mrCompensatedSum(ah.y, -bh.y), mrCompensatedSum(al.y, -bl.y));
    const auto z = mrCompensatedAdd(mrCompensatedSum(ah.z, -bh.z), mrCompensatedSum(al.z, -bl.z));
    return {x.high, y.high, z.high, 0.0f};
}
enum MRConstraintIRBlockFlags : mr_u32 {
    MR_CONSTRAINT_IR_BLOCK_NEW_IMPACT = 1u << 0u,
    MR_CONSTRAINT_IR_BLOCK_WARM_STARTED = 1u << 1u,
    MR_CONSTRAINT_IR_BLOCK_DISABLED = 1u << 2u,
    MR_CONSTRAINT_IR_BLOCK_ROD_ENDPOINT = 1u << 3u,
    MR_CONSTRAINT_IR_BLOCK_GENERALIZED = 1u << 4u,
    MR_CONSTRAINT_IR_BLOCK_ROD_ATTACHMENT = 1u << 5u,
    MR_CONSTRAINT_IR_BLOCK_ROD_TWIST_ATTACHMENT = 1u << 6u,
    MR_CONSTRAINT_IR_BLOCK_ROD_TANGENT_ATTACHMENT = 1u << 7u,
};
enum MRConstraintIRRowFlags : mr_u32 {
    MR_CONSTRAINT_IR_ROW_POSITION_STABILIZED = 1u << 0u,
    MR_CONSTRAINT_IR_ROW_UNILATERAL = 1u << 1u,
    MR_CONSTRAINT_IR_ROW_CONTACT_NORMAL = 1u << 2u,
    MR_CONSTRAINT_IR_ROW_CONTACT_TANGENT = 1u << 3u,
    MR_CONSTRAINT_IR_ROW_CONTACT_TORSION = 1u << 4u,
};
enum MRConstraintIREndpointRole : mr_u32 {
    MR_CONSTRAINT_IR_ENDPOINT_A = 0u,
    MR_CONSTRAINT_IR_ENDPOINT_B = 1u,
    MR_CONSTRAINT_IR_ENDPOINT_WORLD = 2u,
};
enum MRConstraintIRJacobianKind : mr_u32 {
    MR_CONSTRAINT_IR_JACOBIAN_WORLD_POINT = 0u,
    MR_CONSTRAINT_IR_JACOBIAN_BODY_LOCAL_POINT = 1u,
    MR_CONSTRAINT_IR_JACOBIAN_GENERALIZED = 2u,
    MR_CONSTRAINT_IR_JACOBIAN_ANGULAR = 3u,
    MR_CONSTRAINT_IR_JACOBIAN_ROD_EDGE = 4u,
    MR_CONSTRAINT_IR_JACOBIAN_ROD_NODE = 5u,
};
enum MRConstraintIREndpointFlags : mr_u32 {
    MR_CONSTRAINT_IR_ENDPOINT_ROW_MASK = 0x000000ffu,
    MR_CONSTRAINT_IR_ENDPOINT_Q_INDEX_VALID = 1u << 8u,
};
typedef struct alignas(16) MRConstraintIRStableKeyGPU {
    mr_u32 words[4];
} MRConstraintIRStableKeyGPU;
typedef struct alignas(16) MRConstraintIRBlockGPU {
    MRConstraintIRStableKeyGPU key;
    mr_u32 type ;
    mr_u32 dimension ;
    mr_u32 flags ;
    mr_u32 islandIndex ;
    mr_u32 endpointOffset ;
    mr_u32 endpointCount ;
    mr_u32 rowOffset ;
    mr_u32 impulseOffset ;
    mr_u32 coneIndex ;
    mr_u32 eventSlot ;
    mr_u32 reserved0 ;
    mr_u32 reserved1 ;
} MRConstraintIRBlockGPU;
typedef struct alignas(16) MRConstraintIREndpointGPU {
    mr_u32 objectIndex ;
    mr_u32 articulationIndex ;
    mr_u32 linkIndex ;
    mr_u32 role ;
    mr_u32 jacobianKind ;
    mr_u32 flags ;
    mr_u32 reserved0 ;
    mr_u32 reserved1 ;
    mr_float4 anchor;
    mr_float4 axis;
} MRConstraintIREndpointGPU;
enum MRConstraintIREndpointOwnerKind : mr_u32 {
    MR_CONSTRAINT_IR_OWNER_WORLD = 0u,
    MR_CONSTRAINT_IR_OWNER_ARTICULATION = 1u,
    MR_CONSTRAINT_IR_OWNER_FREE_BODY = 2u,
    MR_CONSTRAINT_IR_OWNER_ROD_EDGE = 3u,
    MR_CONSTRAINT_IR_OWNER_ROD_NODE = 4u,
};
enum MRConstraintIREndpointRuntimeFlags : mr_u32 {
    MR_CONSTRAINT_IR_RUNTIME_DYNAMIC = 1u << 0u,
    MR_CONSTRAINT_IR_RUNTIME_HAS_POINT_QUERY = 1u << 1u,
    MR_CONSTRAINT_IR_RUNTIME_HAS_TWIST = 1u << 2u,
    MR_CONSTRAINT_IR_RUNTIME_KINEMATIC = 1u << 3u,
};
typedef struct alignas(16) MRConstraintEndpointRuntimeGPU {
    mr_u32 dynamicNode ;
    mr_u32 ownerKind ;
    mr_u32 ownerIndex ;
    mr_u32 elementIndex ;
    mr_u32 queryIndex ;
    mr_u32 secondaryIndex ;
    mr_u32 twistIndex ;
    mr_u32 flags ;
    mr_float4 weights;
    mr_float4 localAnchorOrRadial;
} MRConstraintEndpointRuntimeGPU;
typedef struct alignas(16) MRConstraintIRRowGPU {
    mr_float4 direction;
    float positionError ;
    float targetVelocity ;
    float compliance ;
    float dissipation ;
    float timeConstant ;
    float dampingRatio ;
    float impulseLower ;
    float impulseUpper ;
    mr_u32 flags ;
    mr_u32 reserved0 ;
    mr_u32 reserved1 ;
    mr_u32 reserved2 ;
} MRConstraintIRRowGPU;
typedef struct alignas(16) MRConstraintIRConeGPU {
    float staticFrictionU ;
    float staticFrictionV ;
    float dynamicFrictionU ;
    float dynamicFrictionV ;
    float rollingLength ;
    float torsionalLength ;
    float restitution ;
    float restitutionThreshold ;
    float adhesionImpulse ;
    float maximumNormalImpulse ;
    float stictionTransitionVelocity ;
    float reserved ;
} MRConstraintIRConeGPU;
typedef struct alignas(16) MREvaluatedConstraintIRRowGPU {
    mr_float4 direction;
    float targetVelocity ;
    float regularization ;
    float impulseLower ;
    float impulseUpper ;
    float sourcePositionError ;
    float stabilizationVelocity ;
    float sourceTargetVelocity ;
    float relativeVelocity ;
    float preSolveVelocity ;
    float reserved0 ;
    float reserved1 ;
    float reserved2 ;
} MREvaluatedConstraintIRRowGPU;
typedef struct alignas(16) MREvaluatedConstraintIRConeGPU {
    float effectiveFrictionU ;
    float effectiveFrictionV ;
    float staticFrictionU ;
    float staticFrictionV ;
    float dynamicFrictionU ;
    float dynamicFrictionV ;
    float rollingLength ;
    float torsionalLength ;
    float restitutionVelocity ;
    float restitutionThreshold ;
    float adhesionImpulse ;
    float maximumNormalImpulse ;
} MREvaluatedConstraintIRConeGPU;
enum MRMotionType : mr_u32 {
    MR_MOTION_STATIC = 0u,
    MR_MOTION_KINEMATIC = 1u,
    MR_MOTION_DYNAMIC = 2u,
};
enum MRBodyStateFlags : mr_u32 {
    MR_BODY_STATE_PRESERVE_RESET_VELOCITY = 1u << 0u,
    MR_BODY_STATE_COLLISION_DISABLED = 1u << 1u,
    MR_BODY_STATE_INERTIA_OVERRIDE = 1u << 2u,
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
    MR_JOINT_FUNCTION_BASED = 7u,
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
    MR_SOLVER_TEMPORAL_CONE = 2u,
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
    MR_CONSTRAINT_FLAG_ROD_ENDPOINT = 1u << 3u,
    MR_CONSTRAINT_FLAG_GENERALIZED = 1u << 4u,
};
enum MRShapeFlags : mr_u32 {
    MR_SHAPE_FLAG_SIMULATION_DISABLED = 1u << 0u,
    MR_SHAPE_FLAG_ENABLE_CCD = 1u << 1u,
    MR_SHAPE_FLAG_MESH_TWO_SIDED = 1u << 2u,
};
enum MRRawContactFlags : mr_u32 {
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
    MR_DOF_FLAG_ROOT = 1u << 0u,
    MR_DOF_FLAG_ACTUATED = 1u << 1u,
    MR_DOF_FLAG_POSITION_LIMIT = 1u << 2u,
    MR_DOF_FLAG_VELOCITY_LIMIT = 1u << 3u,
    MR_DOF_FLAG_EFFORT_LIMIT = 1u << 4u,
    MR_DOF_FLAG_DRIVE = 1u << 5u,
};
enum MRActuatorProfileFlags : mr_u32 {
    MR_ACTUATOR_PROFILE_ACTIVE = 1u << 0u,
    MR_ACTUATOR_PROFILE_CALIBRATED = 1u << 1u,
};
typedef struct alignas(16) MRWorldGPU {
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
    mr_float4 gravityAndTimestep;
    mr_float4 solverScales;
} MRWorldGPU;
typedef struct alignas(16) MRArticulationGPU {
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
typedef struct alignas(16) MRJointDescriptorGPU {
    mr_u32 parentBody;
    mr_u32 childBody;
    mr_u32 jointType;
    mr_u32 flags;
    mr_u32 qOffset;
    mr_u32 nq;
    mr_u32 vOffset;
    mr_u32 nv;
    mr_float4 axis0;
    mr_float4 axis1;
    mr_float4 axis2;
    mr_float4 parentAnchor;
    mr_float4 childAnchor;
    mr_float4 parentRotation;
    mr_float4 childRotation;
} MRJointDescriptorGPU;
typedef struct alignas(16) MRDofPropertiesGPU {
    mr_u32 articulationIndex;
    mr_u32 jointIndex;
    mr_u32 qIndex;
    mr_u32 vIndex;
    mr_u32 localDof;
    mr_u32 flags;
    mr_u32 reserved0;
    mr_u32 reserved1;
    mr_float4 limits;
    mr_float4 drive;
} MRDofPropertiesGPU;
typedef struct alignas(16) MRActuatorProfileGPU {
    mr_float4 motorAndSpeed;
    mr_float4 transmissionAndEnvelope;
    mr_uint4 identity;
} MRActuatorProfileGPU;
typedef struct alignas(16) MRBodyPropertiesGPU {
    mr_u32 articulationIndex;
    mr_u32 parentBody;
    mr_u32 inboundJoint;
    mr_u32 motionType;
    mr_float4 massAndInverseMass;
    mr_float4 centerOfMass;
    mr_float4 inertiaRow0;
    mr_float4 inertiaRow1;
    mr_float4 inertiaRow2;
    mr_float4 inverseInertiaRow0;
    mr_float4 inverseInertiaRow1;
    mr_float4 inverseInertiaRow2;
    mr_float4 dampingAndSpeedLimits;
} MRBodyPropertiesGPU;
typedef struct alignas(16) MRBodyStateGPU {
    mr_float4 position;
    mr_float4 orientation;
    mr_float4 linearVelocityAndInverseMass;
    mr_float4 angularVelocity;
    mr_float4 inverseInertiaWorldRow0;
    mr_float4 inverseInertiaWorldRow1;
    mr_float4 inverseInertiaWorldRow2;
    mr_u32 flagsAndIndices[4];
} MRBodyStateGPU;
typedef struct alignas(16) MRBodyWrenchGPU {
    mr_float4 force;
    mr_float4 torque;
} MRBodyWrenchGPU;
typedef struct alignas(16) MRFreeBodyBatchGPU {
    mr_u32 bodyOffset;
    mr_u32 bodyCount;
    mr_u32 integratorType;
    mr_u32 nonlinearIterations;
    mr_float4 gravityAndTimestep;
    mr_float4 convergence;
} MRFreeBodyBatchGPU;
typedef struct alignas(16) MRFreeBodyStatusGPU {
    mr_u32 code;
    mr_u32 iterations;
    mr_u32 bodyIndex;
    mr_u32 reserved;
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
    MR_ARTICULATED_OPERATOR_WRITE_DIAGNOSTIC_MASS = 1u << 0u,
    MR_ARTICULATED_OPERATOR_WRITE_CHOLESKY_FACTOR = 1u << 1u,
    MR_ARTICULATED_OPERATOR_KINEMATICS_ONLY = 1u << 2u,
    MR_ARTICULATED_OPERATOR_IMPLICIT_DRIVES = 1u << 3u,
    MR_ARTICULATED_OPERATOR_KINEMATICS_JACOBIANS_ONLY = 1u << 4u,
    MR_ARTICULATED_OPERATOR_IGNORE_FOREIGN_POINTS = 1u << 5u,
    MR_ARTICULATED_OPERATOR_COMPENSATED_TRANSLATION = 1u << 6u,
    MR_ARTICULATED_OPERATOR_KINEMATICS_CACHE_PREPARE = 1u << 7u,
    MR_ARTICULATED_OPERATOR_KINEMATICS_CACHE_CONSUME = 1u << 8u,
    MR_ARTICULATED_OPERATOR_WRITE_BODY_SPATIAL_JACOBIANS = 1u << 9u,
    MR_ARTICULATED_OPERATOR_FUSE_CANONICAL_BODY_PROBES = 1u << 10u,
};
typedef struct alignas(16) MRArticulatedOperatorDispatchGPU {
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
enum MRArticulatedPointFlags : mr_u32 {
    MR_ARTICULATED_POINT_INACTIVE = 1u << 0u,
    MR_ARTICULATED_POINT_SPHERE_SUPPORT = 1u << 1u,
    MR_ARTICULATED_POINT_ELLIPSOID_SUPPORT = 1u << 2u,
};
typedef struct alignas(16) MRArticulatedPointImpulseGPU {
    mr_u32 bodyIndex;
    mr_u32 flags;
    mr_u32 reserved0;
    mr_u32 reserved1;
    mr_float4 localPoint;
    mr_float4 worldImpulse;
    mr_float4 supportPlaneNormalAndRadius;
    mr_float4 supportRadii;
    mr_float4 supportOrientation;
} MRArticulatedPointImpulseGPU;
typedef struct alignas(16) MRArticulatedBodyPoseGPU {
    mr_float4 position;
    mr_float4 orientation;
} MRArticulatedBodyPoseGPU;
typedef struct alignas(16) MRArticulatedPointWorldGPU {
    mr_float4 position;
} MRArticulatedPointWorldGPU;
typedef struct alignas(16) MRArticulatedOperatorStatusGPU {
    mr_u32 code;
    mr_u32 environment;
    mr_u32 articulationIndex;
    mr_u32 failingIndex;
    mr_u32 bodyCount;
    mr_u32 nq;
    mr_u32 nv;
    mr_u32 pointCount;
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
typedef struct alignas(16) MRABADispatchGPU {
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
typedef struct alignas(16) MRABABodyWrenchGPU {
    mr_float4 force;
    mr_float4 torque;
} MRABABodyWrenchGPU;
typedef struct alignas(16) MRABAStatusGPU {
    mr_u32 code;
    mr_u32 environment;
    mr_u32 articulationIndex;
    mr_u32 failingIndex;
    mr_u32 bodyCount;
    mr_u32 nq;
    mr_u32 nv;
    mr_u32 flags;
    mr_float4 diagnostics;
} MRABAStatusGPU;
typedef struct alignas(16) MRMLXWorldStepDispatchGPU {
    mr_u32 environmentCount;
    mr_u32 nq;
    mr_u32 nv;
    mr_u32 physicsSubstep;
    mr_u32 physicsSubsteps;
    mr_u32 articulationIndex;
    mr_u32 reserved0;
    mr_u32 reserved1;
} MRMLXWorldStepDispatchGPU;
typedef struct alignas(16) MRMLXWorldStepStatusGPU {
    mr_u32 code;
    mr_u32 abaCode;
    mr_u32 failingSubstep;
    mr_u32 failingIndex;
    mr_u32 successfulSubsteps;
    mr_u32 requiredCapacity;
    mr_u32 flags;
    mr_u32 reserved0;
} MRMLXWorldStepStatusGPU;
typedef struct alignas(16) MRMLXContactAdapterDispatchGPU {
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
    MR_METAL_WORLD_FREE_MOTION_ONLY = 1u << 3u,
    MR_METAL_WORLD_CONTACTS = 1u << 4u,
    MR_METAL_WORLD_IMPLICIT_POSITION_DRIVES = 1u << 5u,
    MR_METAL_WORLD_NATIVE_TASK = 1u << 6u,
    MR_METAL_WORLD_HAS_BODY_WRENCHES = 1u << 7u,
};
typedef struct alignas(16) MRMetalWorldDispatchGPU {
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
typedef struct alignas(16) MRMetalWorldPassGPU {
    mr_u32 controlStep;
    mr_u32 physicsSubstep;
    mr_u32 reserved0;
    mr_u32 reserved1;
} MRMetalWorldPassGPU;
typedef struct alignas(16) MRMetalWorldStatusGPU {
    mr_u32 code;
    mr_u32 environment;
    mr_u32 controlStep;
    mr_u32 successfulSubsteps;
    mr_u32 abaCode;
    mr_u32 failingSubstep;
    mr_u32 failingIndex;
    mr_u32 flags;
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
    MR_METAL_WORLD_CONTACT_STREAMED_RESPONSES = 1u << 9u,
    MR_METAL_WORLD_CONTACT_BODY_WRENCHES = 1u << 10u,
};
typedef struct alignas(16) MRCompiledCollisionPairGPU {
    mr_u32 colliderA;
    mr_u32 colliderB;
    mr_u32 pairClass;
    mr_u32 convexCacheSlot;
} MRCompiledCollisionPairGPU;
typedef struct alignas(16) MRMetalWorldContactDispatchGPU {
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
    mr_u32 authoredConstraintCount;
    mr_u32 authoredEndpointCount;
    mr_u32 authoredRowCount;
    mr_u32 authoredConeCount;
    mr_float4 timestepAndBias;
    mr_float4 manifoldThresholds;
    mr_float4 ccdParameters;
    mr_float4 ccdEventParameters;
} MRMetalWorldContactDispatchGPU;
typedef struct alignas(16) MRIndirectDispatchArgumentsGPU {
    mr_u32 threadgroupsX;
    mr_u32 threadgroupsY;
    mr_u32 threadgroupsZ;
    mr_u32 activeCount;
} MRIndirectDispatchArgumentsGPU;
typedef struct alignas(16) MRWorkQueueHeaderGPU {
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
typedef struct alignas(16) MRScanLevelGPU {
    mr_u32 elementCount;
    mr_u32 blockCount;
    mr_u32 inputOffset;
    mr_u32 outputOffset;
    mr_u32 blockSumOffset;
    mr_u32 parentOffset;
    mr_u32 workClass;
    mr_u32 flags;
} MRScanLevelGPU;
typedef struct alignas(16) MRPairWorkGPU {
    mr_u32 environment;
    mr_u32 compiledPair;
    mr_u32 cacheSlot;
    mr_u32 workClass;
    mr_u32 stableKeyLow;
    mr_u32 stableKeyHigh;
    mr_u32 flags;
    mr_u32 reserved;
} MRPairWorkGPU;
typedef struct alignas(16) MRIslandWorkGPU {
    mr_u32 environment;
    mr_u32 islandIndex;
    mr_u32 firstConstraint;
    mr_u32 constraintCount;
    mr_u32 firstTile;
    mr_u32 tileCount;
    mr_u32 dofClass;
    mr_u32 flags;
} MRIslandWorkGPU;
typedef struct alignas(16) MRWaveWorkPacketGPU {
    mr_uint4 islandSlots;
    mr_uint4 stableKeyLow;
    mr_uint4 stableKeyHigh;
    mr_uint4 metadata;
} MRWaveWorkPacketGPU;
enum MRIslandWorkFlags : mr_u32 {
    MR_ISLAND_WORK_VALID = 1u << 0u,
    MR_ISLAND_WORK_HAS_ARTICULATION = 1u << 1u,
    MR_ISLAND_WORK_SPILL = 1u << 2u,
    MR_ISLAND_WORK_DISTRIBUTED = 1u << 3u,
    MR_ISLAND_WORK_STIFF_REPLAY = 1u << 4u,
    MR_ISLAND_WORK_HAS_ROD = 1u << 5u,
};
typedef struct alignas(16) MRContactTileGPU {
    mr_u32 environment;
    mr_u32 islandIndex;
    mr_u32 firstConstraint;
    mr_u32 constraintCount;
    mr_u32 nextTile;
    mr_u32 partialOffset;
    mr_u32 flags;
    mr_u32 reserved;
} MRContactTileGPU;
typedef struct alignas(16) MRWave32PreconditionerGPU {
    mr_float4 row0;
    mr_float4 row1;
    mr_float4 row2;
} MRWave32PreconditionerGPU;
typedef struct alignas(16) MRWave32IslandStatusGPU {
    mr_u32 code;
    mr_u32 environment;
    mr_u32 islandIndex;
    mr_u32 iterations;
    mr_float4 residuals;
} MRWave32IslandStatusGPU;
typedef struct alignas(16) MRCCDPairGPU {
    mr_u32 environment;
    mr_u32 compiledPair;
    mr_u32 colliderA;
    mr_u32 colliderB;
    mr_float4 intervalAndDistance;
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
    MR_CCD_SEGMENT_REMAINING = 0u,
    MR_CCD_SEGMENT_SELECTED = 1u,
    MR_CCD_SEGMENT_FULL_MICROSTEP = 2u,
};
typedef struct alignas(16) MRCCDEventStateGPU {
    mr_u32 environment;
    mr_u32 splitCount;
    mr_u32 simultaneousEventCount;
    mr_u32 zeroTimeReplayCount;
    mr_u32 flags;
    mr_u32 generation;
    mr_u32 lastStableKeyLow;
    mr_u32 lastStableKeyHigh;
    mr_float4 time;
    mr_uint4 cluster;
} MRCCDEventStateGPU;
typedef struct alignas(16) MRCCDImpactClusterGPU {
    mr_u32 environment;
    mr_u32 firstEventSlot;
    mr_u32 eventCount;
    mr_u32 generation;
    mr_u32 stableKeyLow;
    mr_u32 stableKeyHigh;
    mr_u32 flags;
    mr_u32 reserved0;
    mr_float4 interval;
} MRCCDImpactClusterGPU;
typedef struct alignas(16) MRProjectedColliderGPU {
    mr_uint4 statusAndFlags;
    mr_float4 centerAndRadius;
    mr_float4 rotation;
    mr_float4 lowerAndHalfLength;
    mr_float4 upperAndContactOffset;
} MRProjectedColliderGPU;
typedef struct alignas(16) MRContactPointMetaGPU {
    mr_u32 colliderA;
    mr_u32 colliderB;
    mr_u32 manifoldIndex;
    mr_u32 pointIndex;
    mr_float4 localAnchorA;
    mr_float4 localAnchorB;
} MRContactPointMetaGPU;
typedef struct alignas(16) MRManifoldIRScatterGPU {
    mr_uint4 counts0;
    mr_uint4 counts1;
    mr_uint4 offsets0;
    mr_uint4 offsets1;
    mr_uint4 diagnostics0;
    mr_uint4 diagnostics1;
    mr_float4 metrics;
    mr_uint4 reserved;
} MRManifoldIRScatterGPU;
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
typedef struct alignas(16) MRWorldDynamicNodeGPU {
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
typedef struct alignas(16) MRIslandNodeRefGPU {
    mr_u32 dynamicNode;
    mr_u32 localVelocityOffset;
    mr_u32 velocityCount;
    mr_u32 factorIndex;
} MRIslandNodeRefGPU;
typedef struct alignas(16) MRIslandConstraintRefGPU {
    mr_u32 blockIndex;
    mr_u32 rowOffset;
    mr_u32 rowCount;
    mr_u32 flags;
} MRIslandConstraintRefGPU;
typedef struct alignas(16) MRContactIslandGPU {
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
typedef struct alignas(16) MRArticulationFactorCacheGPU {
    mr_u32 environment;
    mr_u32 articulationIndex;
    mr_u32 nv;
    mr_u32 generation;
    mr_u32 code;
    mr_u32 failingIndex;
    mr_u32 reserved0;
    mr_u32 reserved1;
    mr_float4 diagnostics;
} MRArticulationFactorCacheGPU;
typedef struct alignas(16) MRMetalWorldContactStatusGPU {
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
    mr_float4 residuals;
    mr_float4 diagnostics;
    mr_float4 eventTimes;
    mr_float4 qualityCertificates;
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
typedef struct alignas(16) MRInverseMassDispatchGPU {
    mr_u32 articulationIndex;
    mr_u32 environmentCount;
    mr_u32 rhsCount;
    mr_u32 flags;
    mr_u32 qStride;
    mr_u32 rhsEnvironmentStride;
    mr_u32 rhsVectorStride;
    mr_u32 outputEnvironmentStride;
    mr_u32 outputVectorStride;
    mr_u32 reserved1;
    mr_u32 reserved2;
    mr_u32 reserved3;
} MRInverseMassDispatchGPU;
enum MRInverseMassDispatchFlags : mr_u32 {
    MR_INVERSE_MASS_IMPLICIT_DRIVES = 1u << 0u,
};
typedef struct alignas(16) MRInverseMassStatusGPU {
    mr_u32 code;
    mr_u32 environment;
    mr_u32 articulationIndex;
    mr_u32 failingIndex;
    mr_u32 bodyCount;
    mr_u32 nq;
    mr_u32 nv;
    mr_u32 rhsCount;
    mr_float4 diagnostics;
} MRInverseMassStatusGPU;
enum MRCoupledCandidateOperation : mr_u32 {
    MR_COUPLED_CANDIDATE_KINEMATICS = 0u,
    MR_COUPLED_CANDIDATE_MASS_ACTION = 1u,
    MR_COUPLED_CANDIDATE_INVERSE_MASS = 2u,
    MR_COUPLED_CANDIDATE_PUBLISH = 3u,
};
typedef struct alignas(16) MRCoupledCandidateDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 operation;
    mr_u32 environmentCount;
    mr_u32 articulationIndex;
    mr_u32 qStride;
    mr_u32 vStride;
    mr_u32 bodyStride;
    mr_u32 statusStride;
    mr_u32 qOffset;
    mr_u32 vOffset;
    mr_u32 firstBody;
    mr_u32 bodyCount;
    mr_u32 nq;
    mr_u32 nv;
    mr_u32 pointCount;
    mr_u32 pointStride;
    mr_u32 pointJacobianStride;
    mr_u32 reserved0;
    mr_float4 timestepAndInverse;
} MRCoupledCandidateDispatchGPU;
typedef struct alignas(16) MRExternalArticulatedResponseDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 environmentCount;
    mr_u32 pointCount;
    mr_u32 responseEntryCount;
    mr_u32 nv;
    mr_u32 pointStride;
    mr_u32 generalizedVectorStride;
    mr_u32 inverseMassStatusStride;
} MRExternalArticulatedResponseDispatchGPU;
typedef struct alignas(16) MRMaterialGPU {
    mr_float4 friction;
    mr_float4 response;
    mr_float4 geometry;
} MRMaterialGPU;
typedef struct alignas(16) MRGeometryHeaderGPU {
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
typedef struct alignas(16) MRConvexFaceGPU {
    mr_float4 plane;
    mr_u32 firstHalfEdge;
    mr_u32 halfEdgeCount;
    mr_u32 featureKey;
    mr_u32 reserved;
} MRConvexFaceGPU;
typedef struct alignas(16) MRConvexHalfEdgeGPU {
    mr_u32 originVertex;
    mr_u32 twinHalfEdge;
    mr_u32 nextHalfEdge;
    mr_u32 faceIndex;
} MRConvexHalfEdgeGPU;
typedef struct alignas(16) MRMeshBVHNodeGPU {
    mr_uint4 quantizedLower[4u];
    mr_uint4 quantizedUpper[4u];
    mr_uint4 childIndices;
    mr_uint4 childMeta;
} MRMeshBVHNodeGPU;
typedef struct alignas(16) MRMeshTriangleGPU {
    mr_uint4 verticesAndFeature;
    mr_uint4 adjacencyAndEdges;
    mr_uint4 materialAndFlags;
} MRMeshTriangleGPU;
typedef struct alignas(16) MRConvexQueryCacheGPU {
    mr_float4 separatingAxisAndDistance;
    mr_uint4 supportA;
    mr_uint4 supportB;
    mr_float4 barycentricWeights;
    mr_uint4 featureAndStatus;
} MRConvexQueryCacheGPU;
typedef struct alignas(16) MRShapeGPU {
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
    mr_float4 localPosition;
    mr_float4 localRotation;
    mr_float4 dimensions;
    mr_float4 contactRestAndBoundingRadius;
} MRShapeGPU;
typedef struct alignas(16) MRAabbGPU {
    mr_float4 lower;
    mr_float4 upper;
} MRAabbGPU;
typedef struct alignas(16) MRCandidatePairGPU {
    mr_u32 environment;
    mr_u32 colliderA;
    mr_u32 colliderB;
    mr_u32 flags;
} MRCandidatePairGPU;
typedef struct alignas(16) MRBroadphaseDispatchGPU {
    mr_u32 shapeCount;
    mr_u32 bodyCount;
    mr_u32 logicalPairCount;
    mr_u32 scanBlockCount;
    mr_u32 pairCapacity;
    mr_u32 exclusionCount;
    mr_u32 environment;
    mr_u32 flags;
} MRBroadphaseDispatchGPU;
typedef struct alignas(16) MRBroadphaseStatusGPU {
    mr_u32 code;
    mr_u32 requiredPairs;
    mr_u32 emittedPairs;
    mr_u32 logicalPairs;
} MRBroadphaseStatusGPU;
typedef struct alignas(16) MRRawContactGPU {
    mr_float4 normalAndSeparation;
    mr_float4 pointAWorld;
    mr_float4 pointBWorld;
    mr_u32 featureAndFlags[4];
} MRRawContactGPU;
typedef struct alignas(16) MRManifoldHeaderGPU {
    mr_u32 pairAndCount[4];
    mr_u32 generationsAndFlags[4];
    mr_float4 normalAndAge;
    mr_float4 tangentAndMetric;
} MRManifoldHeaderGPU;
typedef struct alignas(16) MRManifoldPointGPU {
    mr_float4 localAnchorA;
    mr_float4 localAnchorB;
    mr_float4 impulses;
    mr_u32 featureAndLife[4];
} MRManifoldPointGPU;
typedef struct alignas(16) MRConstraintBlockGPU {
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
typedef struct alignas(16) MRContactConstraintGPU {
    mr_u32 bodyA;
    mr_u32 bodyB;
    mr_u32 flags;
    mr_u32 islandIndex;
    mr_u64 pairKey;
    mr_u64 featureKey;
    mr_float4 pointAndSeparation;
    mr_float4 normal;
    mr_float4 tangent;
    mr_float4 friction;
    mr_float4 response;
    mr_float4 targetVelocityAndPreSolveNormal;
    mr_float4 impulses;
} MRContactConstraintGPU;
typedef struct alignas(16) MRSolverBatchGPU {
    mr_u32 bodyOffset;
    mr_u32 bodyCount;
    mr_u32 contactOffset;
    mr_u32 contactCount;
    mr_u32 velocityIterations;
    mr_u32 enableWarmStart;
    mr_u32 enableEarlyExit;
    mr_u32 deterministic;
    mr_float4 timestepAndBias;
    mr_float4 convergence;
} MRSolverBatchGPU;
typedef struct alignas(16) MRSolverStatusGPU {
    mr_u32 code;
    mr_u32 iterations;
    mr_u32 activeContacts;
    mr_u32 islandCount;
    mr_float4 residuals;
    mr_u32 requiredPairs;
    mr_u32 requiredContacts;
    mr_u32 requiredConstraints;
    mr_u32 requiredIslands;
} MRSolverStatusGPU;
enum MRMujocoMuscleReferenceGPUStatus : mr_u32 {
    MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS = 0u,
    MR_MUJOCO_MUSCLE_REFERENCE_INVALID_PROGRAM = 1u,
    MR_MUJOCO_MUSCLE_REFERENCE_INVALID_STATE = 2u,
    MR_MUJOCO_MUSCLE_REFERENCE_INVALID_PATH = 3u,
    MR_MUJOCO_MUSCLE_REFERENCE_NONFINITE_RESULT = 4u,
};
enum MRMujocoMuscleRouteNodeType : mr_u32 {
    MR_MUJOCO_MUSCLE_ROUTE_SITE = 1u,
    MR_MUJOCO_MUSCLE_ROUTE_SPHERE = 2u,
    MR_MUJOCO_MUSCLE_ROUTE_CYLINDER = 3u,
};
typedef struct alignas(16) MRMujocoMuscleReferenceDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 muscleCount;
    mr_u32 siteCount;
    mr_u32 wrapCount;
    mr_u32 routeNodeCount;
    mr_u32 environmentCount;
    mr_u32 bodyPoseStride;
    mr_u32 articulationFirstBody;
    mr_u32 dofCount;
    mr_u32 pointJacobianStride;
    mr_u32 bodyJacobianPointOffset;
    mr_u32 bodyJacobianPointStride;
    mr_float4 timestepSecondsAndReserved;
} MRMujocoMuscleReferenceDispatchGPU;
typedef struct alignas(16) MRMujocoMuscleGPU {
    mr_uint4 route;
    mr_float4 lengthRangeAndAcceleration;
    mr_float4 controlRange;
    mr_float4 gainParameters[3];
    mr_float4 biasParameters[3];
    mr_float4 dynamicParameters[3];
    mr_float4 compliantArchitecture0;
    mr_float4 compliantArchitecture1;
} MRMujocoMuscleGPU;
typedef struct alignas(16) MRMujocoMuscleSiteGPU {
    mr_u32 bodyIndex;
    mr_u32 reserved0;
    mr_u32 reserved1;
    mr_u32 reserved2;
    mr_float4 localPoint;
} MRMujocoMuscleSiteGPU;
typedef struct alignas(16) MRMujocoMuscleWrapGPU {
    mr_u32 bodyIndex;
    mr_u32 type;
    mr_u32 reserved0;
    mr_u32 reserved1;
    mr_float4 localCenter;
    mr_float4 rotationRow0;
    mr_float4 rotationRow1;
    mr_float4 rotationRow2;
    mr_float4 radius;
} MRMujocoMuscleWrapGPU;
typedef struct alignas(16) MRMujocoMuscleRouteNodeGPU {
    mr_u32 type;
    mr_u32 targetIndex;
    mr_u32 sideSiteIndex;
    mr_u32 reserved0;
} MRMujocoMuscleRouteNodeGPU;
typedef struct alignas(16) MRMujocoMuscleStateGPU {
    mr_float4 excitationAndActivation;
} MRMujocoMuscleStateGPU;
typedef struct alignas(16) MRMujocoMuscleActivationDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 stateCount;
    mr_u32 reserved0;
    mr_u32 reserved1;
    mr_float4 timestepSecondsAndReserved;
} MRMujocoMuscleActivationDispatchGPU;
typedef struct alignas(16) MRMujocoMuscleActiveForceDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 muscleCount;
    mr_u32 environmentCount;
    mr_u32 dofCount;
} MRMujocoMuscleActiveForceDispatchGPU;
typedef struct alignas(16) MRMujocoMuscleRouteCutDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 cutCount;
    mr_u32 reserved0;
    mr_u32 reserved1;
} MRMujocoMuscleRouteCutDispatchGPU;
typedef struct alignas(16) MRMujocoMuscleRouteCutGPU {
    mr_u32 muscleIndex;
    mr_u32 routeNodeOrdinal;
    mr_u32 reserved0;
    mr_u32 reserved1;
} MRMujocoMuscleRouteCutGPU;
typedef struct alignas(16) MRMujocoMuscleResultGPU {
    mr_u32 status;
    mr_u32 environment;
    mr_u32 muscleIndex;
    mr_u32 appliedWrapCount;
    mr_float4 pathForceAndActivationDerivative;
    mr_float4 endpointLengthGradients[2];
    mr_float4 activeForceAndReserved;
    mr_float4 fiberStateTendonForceResidual;
} MRMujocoMuscleResultGPU;

using namespace metal;
inline MRCompensatedScalar mrCompensatedNegate(MRCompensatedScalar a) {
    return {-a.high,-a.low};
}
inline MRCompensatedScalar mrCompensatedMultiply(MRCompensatedScalar a, MRCompensatedScalar b) {
    return mrCompensatedAdd(mrCompensatedProduct(a.high,b.high),
        mrCompensatedAdd(mrCompensatedProduct(a.high,b.low),
            mrCompensatedAdd(mrCompensatedProduct(a.low,b.high),mrCompensatedProduct(a.low,b.low))));
}
inline MRCompensatedScalar mrCompensatedDivide(MRCompensatedScalar a, MRCompensatedScalar b) {
    const float first=a.high/b.high;
    const auto remainder=mrCompensatedAdd(a,mrCompensatedNegate(mrCompensatedMultiply(b,{first,0.0f})));
    return mrCompensatedAdd({first,0.0f},{(remainder.high+remainder.low)/b.high,0.0f});
}
inline MRCompensatedScalar mrCompensatedSqrt(MRCompensatedScalar a) {
    const float first=sqrt(a.high);
    const auto remainder=mrCompensatedAdd(a,mrCompensatedNegate(mrCompensatedProduct(first,first)));
    return mrCompensatedAdd({first,0.0f},{(remainder.high+remainder.low)/(2.0f*first),0.0f});
}
inline MRCompensatedPositionGPU mrCompensatedVector(mr_float4 value) {
    value.w=0.0f; return {value,{0.0f,0.0f,0.0f,0.0f}};
}
inline MRCompensatedPositionGPU mrCompensatedVectorPack(MRCompensatedScalar x,MRCompensatedScalar y,MRCompensatedScalar z) {
    return {{x.high,y.high,z.high,0.0f},{x.low,y.low,z.low,0.0f}};
}
inline MRCompensatedPositionGPU mrCompensatedVectorAdd(MRCompensatedPositionGPU a,MRCompensatedPositionGPU b) {
    return mrCompensatedVectorPack(mrCompensatedAdd({a.high.x,a.low.x},{b.high.x,b.low.x}),
        mrCompensatedAdd({a.high.y,a.low.y},{b.high.y,b.low.y}),
        mrCompensatedAdd({a.high.z,a.low.z},{b.high.z,b.low.z}));
}
inline MRCompensatedPositionGPU mrCompensatedVectorNegate(MRCompensatedPositionGPU a) {
    return {{-a.high.x,-a.high.y,-a.high.z,0.0f},{-a.low.x,-a.low.y,-a.low.z,0.0f}};
}
inline MRCompensatedPositionGPU mrCompensatedVectorScale(MRCompensatedPositionGPU a,MRCompensatedScalar scale) {
    return mrCompensatedVectorPack(mrCompensatedMultiply({a.high.x,a.low.x},scale),
        mrCompensatedMultiply({a.high.y,a.low.y},scale),mrCompensatedMultiply({a.high.z,a.low.z},scale));
}
inline MRCompensatedPositionGPU mrCompensatedVectorCross(MRCompensatedPositionGPU a,MRCompensatedPositionGPU b) {
    const MRCompensatedScalar ax{a.high.x,a.low.x},ay{a.high.y,a.low.y},az{a.high.z,a.low.z};
    const MRCompensatedScalar bx{b.high.x,b.low.x},by{b.high.y,b.low.y},bz{b.high.z,b.low.z};
    return mrCompensatedVectorPack(mrCompensatedAdd(mrCompensatedMultiply(ay,bz),mrCompensatedNegate(mrCompensatedMultiply(az,by))),
        mrCompensatedAdd(mrCompensatedMultiply(az,bx),mrCompensatedNegate(mrCompensatedMultiply(ax,bz))),
        mrCompensatedAdd(mrCompensatedMultiply(ax,by),mrCompensatedNegate(mrCompensatedMultiply(ay,bx))));
}
inline MRCompensatedScalar mrCompensatedVectorDot(MRCompensatedPositionGPU a,MRCompensatedPositionGPU b) {
    return mrCompensatedAdd(mrCompensatedMultiply({a.high.x,a.low.x},{b.high.x,b.low.x}),
        mrCompensatedAdd(mrCompensatedMultiply({a.high.y,a.low.y},{b.high.y,b.low.y}),
            mrCompensatedMultiply({a.high.z,a.low.z},{b.high.z,b.low.z})));
}
inline MRCompensatedPositionGPU mrCompensatedQuaternionRotatePair(mr_float4 q,MRCompensatedPositionGPU vector) {
    const auto axis=mrCompensatedVector(q);
    const auto twiceCross=mrCompensatedVectorScale(mrCompensatedVectorCross(axis,vector),{2.0f,0.0f});
    return mrCompensatedVectorAdd(vector,mrCompensatedVectorAdd(
        mrCompensatedVectorScale(twiceCross,{q.w,0.0f}),mrCompensatedVectorCross(axis,twiceCross)));
}
inline MRCompensatedPositionGPU mrCompensatedQuaternionRotate(mr_float4 q,mr_float4 vector) {
    return mrCompensatedQuaternionRotatePair(q,mrCompensatedVector(vector));
}
inline mr_float4 mrCompensatedGeometryQuaternionMultiply(mr_float4 a,mr_float4 b) {
    return {a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
        a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
        a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w,
        a.w*b.w-((a.x*b.x+a.y*b.y)+a.z*b.z)};
}
inline MRCompensatedPositionGPU mrCompensatedSupportOffset(
    mr_float4 bodyOrientation,mr_float4 localPoint,mr_float4 supportOrientation,
    mr_float4 supportRadii,mr_float4 normalAndRadius,mr_u32 kind) {
    auto result=mrCompensatedQuaternionRotate(bodyOrientation,localPoint);
    if (kind != 2u) return mrCompensatedVectorAdd(result,mrCompensatedVectorNegate(
        mrCompensatedVectorScale(mrCompensatedVector(normalAndRadius),{normalAndRadius.w,0.0f})));
    const auto shape=mrCompensatedGeometryQuaternionMultiply(bodyOrientation,supportOrientation);
    const mr_float4 conjugate{-shape.x,-shape.y,-shape.z,shape.w};
    const auto direction=mrCompensatedQuaternionRotate(conjugate,normalAndRadius);
    const auto scaled=mrCompensatedVectorPack(
        mrCompensatedMultiply({direction.high.x,direction.low.x},{supportRadii.x,0.0f}),
        mrCompensatedMultiply({direction.high.y,direction.low.y},{supportRadii.y,0.0f}),
        mrCompensatedMultiply({direction.high.z,direction.low.z},{supportRadii.z,0.0f}));
    const auto magnitude=mrCompensatedSqrt(mrCompensatedVectorDot(scaled,scaled));
    const auto surface=mrCompensatedVectorPack(
        mrCompensatedDivide(mrCompensatedMultiply({scaled.high.x,scaled.low.x},{supportRadii.x,0.0f}),magnitude),
        mrCompensatedDivide(mrCompensatedMultiply({scaled.high.y,scaled.low.y},{supportRadii.y,0.0f}),magnitude),
        mrCompensatedDivide(mrCompensatedMultiply({scaled.high.z,scaled.low.z},{supportRadii.z,0.0f}),magnitude));
    return mrCompensatedVectorAdd(result,mrCompensatedVectorNegate(mrCompensatedQuaternionRotatePair(shape,surface)));
}
inline MRCompensatedPositionGPU mrCompensatedTranslationPositionPair(
    MRCompensatedRootTranslationGPU root,MRCompensatedPositionGPU local) {
    return mrCompensatedVectorAdd(mrCompensatedTranslationPosition(root,local.high),mrCompensatedVector(local.low));
}
namespace {
using MRSourcePoint = float3;
inline MRSourcePoint mrSourceRotate(float4 q,float3 v) { const float3 t=2.0f*cross(q.xyz,v); return v+q.w*t+cross(q.xyz,t); }
inline MRSourcePoint mrSourceZeroPoint() { return float3(0.0f); }
inline float3 mrSourcePointProjection(MRSourcePoint point) { return point; }
inline MRSourcePoint mrSourceBodyPoint(float4 high, device const float4*, uint) { return high.xyz; }
inline bool mrSourcePointFinite(MRSourcePoint point) { return all(isfinite(point)); }
}
namespace {
constant float kMinimum = 1.0e-7f;
constant float kPi = 3.14159265358979323846f;
inline bool finite4(const float4 value) { return all(isfinite(value)); }
inline float3 quaternionConjugateRotate(
    const float4 quaternion,
    const float3 value
) {
    const float3 conjugate = -quaternion.xyz;
    const float3 doubledCross = 2.0f * cross(conjugate, value);
    return value + quaternion.w * doubledCross + cross(conjugate, doubledCross);
}
inline float3 quaternionRotate(
    const float4 quaternion,
    const float3 value
) {
    const float3 doubledCross = 2.0f * cross(quaternion.xyz, value);
    return value + quaternion.w * doubledCross + cross(quaternion.xyz, doubledCross);
}
inline float3 matrixApply(
    const MRMujocoMuscleWrapGPU wrap,
    const float3 value
) {
    return float3(
        dot(wrap.rotationRow0.xyz, value),
        dot(wrap.rotationRow1.xyz, value),
        dot(wrap.rotationRow2.xyz, value)
    );
}
inline float3 matrixTransposeApply(
    const MRMujocoMuscleWrapGPU wrap,
    const float3 value
) {
    return wrap.rotationRow0.xyz * value.x +
        wrap.rotationRow1.xyz * value.y +
        wrap.rotationRow2.xyz * value.z;
}
inline float cross2(const float2 first, const float2 second) {
    const float product = first.y * second.x;
    return fma(first.x, second.y, -product) -
        fma(first.y, second.x, -product);
}
inline bool segmentIntersects(
    const float2 firstA,
    const float2 secondA,
    const float2 firstB,
    const float2 secondB
) {
    const float2 directionA = secondA - firstA;
    const float2 directionB = secondB - firstB;
    const float determinant = cross2(directionA, directionB);
    if (abs(determinant) < kMinimum) {
        return false;
    }
    const float firstOnA = cross2(directionA, firstB - firstA);
    const float secondOnA = cross2(directionA, secondB - secondA);
    const float firstOnB = cross2(directionB, firstA - firstB);
    const float secondOnB = cross2(directionB, secondA - secondB);
    const bool spansA = (firstOnA >= 0.0f && secondOnA <= 0.0f) ||
        (firstOnA <= 0.0f && secondOnA >= 0.0f);
    const bool spansB = (firstOnB >= 0.0f && secondOnB <= 0.0f) ||
        (firstOnB <= 0.0f && secondOnB >= 0.0f);
    return spansA && spansB;
}
inline float circleLength(
    const float2 first,
    const float2 second,
    const uint index,
    const float radius
) {
    const float determinant = -cross2(first, second);
    float angle = atan2(abs(determinant), dot(first, second));
    if ((determinant > 0.0f && index != 0u) ||
        (determinant < 0.0f && index == 0u)) {
        angle = 2.0f * kPi - angle;
    }
    return radius * angle;
}
inline bool wrapCircle(
    const float2 first,
    const float2 second,
    const bool hasSide,
    const float2 side,
    const float radius,
    thread float4& contact,
    thread float& wrappingLength
) {
    const float squaredFirst = dot(first, first);
    const float squaredSecond = dot(second, second);
    const float squaredRadius = radius * radius;
    if (squaredFirst < squaredRadius || squaredSecond < squaredRadius ||
        radius < kMinimum) {
        return false;
    }
    const float2 difference = second - first;
    const float differenceSquared = dot(difference, difference);
    if (differenceSquared < kMinimum) {
        return false;
    }
    const float fraction = clamp(-dot(difference, first) / differenceSquared,
                                 0.0f, 1.0f);
    const float2 nearest = fraction * difference + first;
    if (dot(nearest, nearest) > squaredRadius &&
        (!hasSide || dot(side, nearest) >= 0.0f)) {
        return false;
    }
    const float rootFirst = sqrt(max(0.0f, squaredFirst - squaredRadius));
    const float rootSecond = sqrt(max(0.0f, squaredSecond - squaredRadius));
    float2 firstSolutions[2];
    float2 secondSolutions[2];
    float goodness[2];
    for (uint index = 0u; index < 2u; ++index) {
        const float sign = index == 0u ? 1.0f : -1.0f;
        firstSolutions[index] = float2(
            (first.x * squaredRadius + sign * radius * first.y * rootFirst) / squaredFirst,
            (first.y * squaredRadius - sign * radius * first.x * rootFirst) / squaredFirst
        );
        secondSolutions[index] = float2(
            (second.x * squaredRadius - sign * radius * second.y * rootSecond) / squaredSecond,
            (second.y * squaredRadius + sign * radius * second.x * rootSecond) / squaredSecond
        );
        if (hasSide) {
            goodness[index] = dot(
                normalize(firstSolutions[index] + secondSolutions[index]), side
            );
        } else {
            const float2 delta = firstSolutions[index] - secondSolutions[index];
            goodness[index] = -dot(delta, delta);
        }
        if (segmentIntersects(first, firstSolutions[index], second,
                              secondSolutions[index])) {
            goodness[index] = -10000.0f;
        }
    }
    const uint selected = goodness[0] > goodness[1] ? 0u : 1u;
    contact = float4(firstSolutions[selected], secondSolutions[selected]);
    if (segmentIntersects(first, contact.xy, second, contact.zw)) {
        return false;
    }
    wrappingLength = circleLength(contact.xy, contact.zw, selected, radius);
    return isfinite(wrappingLength);
}
inline bool wrapInside(
    const float2 first,
    const float2 second,
    const float radius,
    thread float4& contact
) {
    const float firstLength = length(first);
    const float secondLength = length(second);
    if (firstLength <= radius || secondLength <= radius ||
        radius < kMinimum || firstLength < kMinimum || secondLength < kMinimum) {
        return false;
    }
    const float2 difference = second - first;
    const float differenceSquared = dot(difference, difference);
    if (differenceSquared > kMinimum) {
        const float fraction = -dot(first, difference) / differenceSquared;
        if (fraction > 0.0f && fraction < 1.0f &&
            length(first + fraction * difference) <= radius) {
            return false;
        }
    }
    float2 midpoint = first + second;
    if (length(midpoint) < kMinimum) {
        return false;
    }
    midpoint = radius * normalize(midpoint);
    const float a = radius / firstLength;
    const float b = radius / secondLength;
    const float cosine = (firstLength * firstLength + secondLength * secondLength -
                          differenceSquared) / (2.0f * firstLength * secondLength);
    contact = float4(midpoint, midpoint);
    if (cosine < -1.0f + kMinimum) {
        return false;
    }
    if (cosine > 1.0f - kMinimum) return true;
    const float angle = acos(cosine);
    float upper = asin(1.0f - 1.0e-7f);
    float lower = 0.0f;
    float theta = upper;
    float z = sin(theta);
    float residual = asin(a * z) + asin(b * z) - 2.0f * theta + angle;
    if (residual > 0.0f) return true;
    uint iteration = 0u;
    for (; iteration < 20u && abs(residual) > 1.0e-6f; ++iteration) {
        if (residual > 0.0f) lower = theta;
        else upper = theta;
        const float cosineTheta = cos(theta);
        const float derivative =
            a * cosineTheta / max(kMinimum, sqrt(max(0.0f, 1.0f - a * a * z * z))) +
            b * cosineTheta / max(kMinimum, sqrt(max(0.0f, 1.0f - b * b * z * z))) - 2.0f;
        const float next = theta - residual / derivative;
        theta = isfinite(next) && next > lower && next < upper
            ? next : 0.5f * (lower + upper);
        z = sin(theta);
        residual = asin(a * z) + asin(b * z) - 2.0f * theta + angle;
    }
    if (abs(residual) > 1.0e-6f) return true;
    const bool firstSide = first.x * second.y - first.y * second.x > 0.0f;
    const float2 vector = normalize(firstSide ? first : second);
    const float rotation = firstSide
        ? theta - asin(a * z)
        : theta - asin(b * z);
    const float2 point = radius * float2(
        cos(rotation) * vector.x - sin(rotation) * vector.y,
        sin(rotation) * vector.x + cos(rotation) * vector.y
    );
    contact = float4(point, point);
    return true;
}
inline bool siteWorld(
    const uint environment,
    const MRMujocoMuscleReferenceDispatchGPU dispatch,
    device const MRArticulatedBodyPoseGPU* bodyPoses,
    device const float4* bodyPositionLow,
    device const MRMujocoMuscleSiteGPU* sites,
    const uint index,
    thread MRSourcePoint& world
) {
    if (index >= dispatch.siteCount) return false;
    const MRMujocoMuscleSiteGPU site = sites[index];
    if (site.bodyIndex < dispatch.articulationFirstBody ||
        site.bodyIndex - dispatch.articulationFirstBody >= dispatch.bodyPoseStride ||
        site.reserved0 != 0u || site.reserved1 != 0u || site.reserved2 != 0u ||
        !finite4(site.localPoint) || site.localPoint.w != 0.0f) return false;
    const MRArticulatedBodyPoseGPU pose = bodyPoses[
        environment * dispatch.bodyPoseStride +
        site.bodyIndex - dispatch.articulationFirstBody
    ];
    if (!finite4(pose.position) || !finite4(pose.orientation)) return false;
    world = mrSourceBodyPoint(pose.position,bodyPositionLow,
        environment*dispatch.bodyPoseStride+site.bodyIndex-dispatch.articulationFirstBody) +
        mrSourceRotate(pose.orientation, site.localPoint.xyz);
    return mrSourcePointFinite(world);
}
inline bool addPointLengthGradient(
    const uint environment,
    const MRMujocoMuscleReferenceDispatchGPU dispatch,
    device const MRArticulatedBodyPoseGPU* bodyPoses,
    device const float4* bodyPositionLow,
    device const float* pointJacobians,
    const uint bodyIndex,
    const MRSourcePoint worldPoint,
    const float3 gradient,
    device float* lengthJacobian ,
    const uint dofLane = 0u,
    const uint dofStride = 1u
) {
    if (bodyIndex < dispatch.articulationFirstBody ||
        bodyIndex - dispatch.articulationFirstBody >= dispatch.bodyPoseStride ||
        dispatch.dofCount == 0u || dispatch.pointJacobianStride == 0u ||
        dispatch.bodyJacobianPointStride != 4u ||
        dispatch.bodyJacobianPointOffset >
            dispatch.pointJacobianStride / (3u * dispatch.dofCount)) {
        return false;
    }
    const uint localBody = bodyIndex - dispatch.articulationFirstBody;
    const uint bodyPoint = dispatch.bodyJacobianPointOffset +
        localBody * dispatch.bodyJacobianPointStride;
    const uint pointCount = dispatch.pointJacobianStride /
        (3u * dispatch.dofCount);
    if (bodyPoint > pointCount ||
        dispatch.bodyJacobianPointStride > pointCount - bodyPoint) {
        return false;
    }
    const MRArticulatedBodyPoseGPU pose = bodyPoses[
        environment * dispatch.bodyPoseStride + localBody
    ];
    if (!finite4(pose.position) || !finite4(pose.orientation) ||
        !mrSourcePointFinite(worldPoint) || !all(isfinite(gradient))) {
        return false;
    }
    const uint environmentBase = environment * dispatch.pointJacobianStride;
    const uint centerBase = environmentBase +
        bodyPoint * 3u * dispatch.dofCount;
    const float3 worldAxes[3] = {
        quaternionRotate(pose.orientation, float3(1.0f, 0.0f, 0.0f)),
        quaternionRotate(pose.orientation, float3(0.0f, 1.0f, 0.0f)),
        quaternionRotate(pose.orientation, float3(0.0f, 0.0f, 1.0f)),
    };
    const float3 pointLever = worldPoint - mrSourceBodyPoint(
        pose.position, bodyPositionLow,
        environment * dispatch.bodyPoseStride + localBody);
    bool finiteJacobian = true;
    for (uint dof = dofLane; dof < dispatch.dofCount; dof += dofStride) {
        const float3 centerJacobian = float3(
            pointJacobians[centerBase + dof],
            pointJacobians[centerBase + dispatch.dofCount + dof],
            pointJacobians[centerBase + 2u * dispatch.dofCount + dof]
        );
        float3 angularJacobian = float3(0.0f);
        for (uint axis = 0u; axis < 3u; ++axis) {
            const uint axisBase = centerBase +
                (axis + 1u) * 3u * dispatch.dofCount;
            const float3 axisJacobian = float3(
                pointJacobians[axisBase + dof],
                pointJacobians[axisBase + dispatch.dofCount + dof],
                pointJacobians[axisBase + 2u * dispatch.dofCount + dof]
            );
            angularJacobian += 0.5f * cross(
                worldAxes[axis],
                axisJacobian - centerJacobian
            );
        }
        const float3 pointJacobian = centerJacobian + cross(
            angularJacobian, pointLever
        );
        if (!all(isfinite(pointJacobian))) {
            return false;
        }
        lengthJacobian[dof] += dot(gradient, pointJacobian);
    }
    return true;
}
inline bool addSegmentLengthJacobian(
    const uint environment,
    const MRMujocoMuscleReferenceDispatchGPU dispatch,
    device const MRArticulatedBodyPoseGPU* bodyPoses,
    device const float4* bodyPositionLow,
    device const float* pointJacobians,
    const uint firstBody,
    const MRSourcePoint firstWorld,
    const uint secondBody,
    const MRSourcePoint secondWorld,
    device float* lengthJacobian ,
    const uint dofLane = 0u,
    const uint dofStride = 1u
) {
    const float3 difference = secondWorld - firstWorld;
    const float distance = length(difference);
    if (!(distance > kMinimum)) return true;
    const float3 direction = difference / distance;
    return addPointLengthGradient(
               environment, dispatch, bodyPoses, bodyPositionLow, pointJacobians,
               firstBody, firstWorld, -direction, lengthJacobian
                                       , dofLane, dofStride
           ) &&
        addPointLengthGradient(
            environment, dispatch, bodyPoses, bodyPositionLow, pointJacobians,
            secondBody, secondWorld, direction, lengthJacobian
                                    , dofLane, dofStride
        );
}
inline float gainLength(const float value, const float lower, const float upper) {
    if (value < lower || value > upper) return 0.0f;
    const float lowerMid = 0.5f * (lower + 1.0f);
    const float upperMid = 0.5f * (1.0f + upper);
    if (value <= lowerMid) { const float x = (value - lower) / max(kMinimum, lowerMid - lower); return 0.5f * x * x; }
    if (value <= 1.0f) { const float x = (1.0f - value) / max(kMinimum, 1.0f - lowerMid); return 1.0f - 0.5f * x * x; }
    if (value <= upperMid) { const float x = (value - 1.0f) / max(kMinimum, upperMid - 1.0f); return 1.0f - 0.5f * x * x; }
    const float x = (upper - value) / max(kMinimum, upper - upperMid); return 0.5f * x * x;
}
inline float parameter(const mr_float4 blocks[3], const uint index) {
    return blocks[index >> 2u][index & 3u];
}
inline float forceScale(const mr_float4 parameters[3], const float accelerationScale) {
    const float direct = parameter(parameters, 2u);
    return direct < 0.0f ? parameter(parameters, 3u) / max(kMinimum, accelerationScale) : direct;
}
inline float muscleGain(
    const float pathLength,
    const float pathVelocity,
    const MRMujocoMuscleGPU muscle
) {
    const float optimum = (muscle.lengthRangeAndAcceleration.y - muscle.lengthRangeAndAcceleration.x) /
        max(kMinimum, parameter(muscle.gainParameters, 1u) - parameter(muscle.gainParameters, 0u));
    const float normalized = parameter(muscle.gainParameters, 0u) +
        (pathLength - muscle.lengthRangeAndAcceleration.x) / max(kMinimum, optimum);
    const float lengthGain = gainLength(normalized, parameter(muscle.gainParameters, 4u), parameter(muscle.gainParameters, 5u));
    const float normalizedVelocity = pathVelocity / max(
        kMinimum, optimum * parameter(muscle.gainParameters, 6u)
    );
    const float eccentricLimit = parameter(muscle.gainParameters, 8u);
    const float transition = eccentricLimit - 1.0f;
    float velocityGain = 0.0f;
    if (normalizedVelocity <= -1.0f) velocityGain = 0.0f;
    else if (normalizedVelocity <= 0.0f) velocityGain =
        (normalizedVelocity + 1.0f) * (normalizedVelocity + 1.0f);
    else if (normalizedVelocity <= transition) velocityGain =
        eccentricLimit - (transition - normalizedVelocity) *
            (transition - normalizedVelocity) / max(kMinimum, transition);
    else velocityGain = eccentricLimit;
    return -forceScale(muscle.gainParameters, muscle.lengthRangeAndAcceleration.z) *
        lengthGain * velocityGain;
}
inline float normalizedPassiveForce(
    const float normalizedLength,
    const MRMujocoMuscleGPU muscle
) {
    const float upperMid = 0.5f * (1.0f + parameter(muscle.biasParameters, 5u));
    if (normalizedLength <= 1.0f) return 0.0f;
    if (normalizedLength <= upperMid) {
        const float x = (normalizedLength - 1.0f) /
            max(kMinimum, upperMid - 1.0f);
        return parameter(muscle.biasParameters, 7u) * 0.5f * x * x;
    }
    const float x = (normalizedLength - upperMid) /
        max(kMinimum, upperMid - 1.0f);
    return parameter(muscle.biasParameters, 7u) * (0.5f + x);
}
inline float normalizedVelocityGain(
    const float fiberVelocity,
    const float optimalFiberLength,
    const MRMujocoMuscleGPU muscle
) {
    const float normalizedVelocity = fiberVelocity / max(
        kMinimum,
        optimalFiberLength * parameter(muscle.gainParameters, 6u)
    );
    const float eccentricLimit = parameter(muscle.gainParameters, 8u);
    const float transition = eccentricLimit - 1.0f;
    if (normalizedVelocity <= -1.0f) return 0.0f;
    if (normalizedVelocity <= 0.0f) return
        (normalizedVelocity + 1.0f) * (normalizedVelocity + 1.0f);
    if (normalizedVelocity <= transition) return
        eccentricLimit - (transition - normalizedVelocity) *
            (transition - normalizedVelocity) / max(kMinimum, transition);
    return eccentricLimit;
}
inline float normalizedTendonForceFromStrain(
    const float strain,
    const MRMujocoMuscleGPU muscle
) {
    if (strain <= 0.0f) return 0.0f;
    const float strainAtOne = muscle.compliantArchitecture0.z;
    const float stiffness = muscle.compliantArchitecture0.w;
    const float forceAtToe = muscle.compliantArchitecture1.x;
    const float strainAtToe = strainAtOne - (1.0f - forceAtToe) /
        max(kMinimum, stiffness);
    if (strain >= strainAtToe) {
        return forceAtToe + stiffness * (strain - strainAtToe);
    }
    const float t = clamp(strain / max(kMinimum, strainAtToe), 0.0f, 1.0f);
    return (-2.0f * t * t * t + 3.0f * t * t) * forceAtToe +
        (t * t * t - t * t) * strainAtToe * stiffness;
}
inline bool compliantArchitecture(const MRMujocoMuscleGPU muscle) {
    return muscle.compliantArchitecture0.x > kMinimum &&
        muscle.compliantArchitecture0.y > kMinimum;
}
inline float compliantFiberIncrementResidual(
    const float increment, const float acceptedFiber, const float predictedPath,
    const float timestep, const float activation, const MRMujocoMuscleGPU muscle,
    thread float& normalizedTension
) {
    const float optimum = muscle.compliantArchitecture0.x;
    const float velocity = increment / timestep;
    const float normalizedFiber = (acceptedFiber + increment) / optimum;
    const float slack = muscle.compliantArchitecture0.y;
    const float initialExtension = (predictedPath - acceptedFiber) - slack;
    normalizedTension = normalizedTendonForceFromStrain(
        (initialExtension - increment) / slack, muscle);
    const float active = activation * gainLength(normalizedFiber,
        parameter(muscle.gainParameters, 4u), parameter(muscle.gainParameters, 5u)) *
        normalizedVelocityGain(velocity, optimum, muscle);
    return normalizedTension - active - normalizedPassiveForce(normalizedFiber, muscle) -
        muscle.compliantArchitecture1.z * velocity / optimum;
}
inline bool solveCompliantFiber(
    const float pathLength,
    const float pathVelocity,
    const float timestep,
    const float activation,
    const MRMujocoMuscleStateGPU state,
    const MRMujocoMuscleGPU muscle,
    thread float& fiberLength,
    thread float& fiberVelocity,
    thread float& tendonForce,
    thread float& equilibriumResidual
) {
    const float optimalFiberLength = muscle.compliantArchitecture0.x;
    const float tendonSlackLength = muscle.compliantArchitecture0.y;
    if (!(timestep > 0.0f) || !isfinite(timestep)) return false;
    const float predictedPath = max(
        kMinimum, pathLength + max(0.0f, timestep) * pathVelocity
    );
    float acceptedFiber = state.excitationAndActivation.z;
    if (!(acceptedFiber > kMinimum)) {
        float initializationLower = min(
            0.05f * optimalFiberLength,
            0.5f * pathLength
        );
        float initializationUpper = pathLength;
        for (uint iteration = 0u; iteration < 48u; ++iteration) {
            const float candidate =
                0.5f * (initializationLower + initializationUpper);
            const float normalizedFiber = candidate / optimalFiberLength;
            const float tendon = normalizedTendonForceFromStrain(
                ((pathLength - candidate) - tendonSlackLength) / tendonSlackLength, muscle
            );
            const float active = activation * gainLength(
                normalizedFiber,
                parameter(muscle.gainParameters, 4u),
                parameter(muscle.gainParameters, 5u)
            );
            const float staticResidual = tendon - active -
                normalizedPassiveForce(normalizedFiber, muscle);
            if (staticResidual > 0.0f) initializationLower = candidate;
            else initializationUpper = candidate;
        }
        acceptedFiber =
            0.5f * (initializationLower + initializationUpper);
    }
    if (pathVelocity == 0.0f) {
        float stationaryTension = 0.0f;
        const float stationaryResidual = compliantFiberIncrementResidual(
            0.0f, acceptedFiber, predictedPath, timestep, activation, muscle,
            stationaryTension
        );
        if (isfinite(stationaryResidual) && abs(stationaryResidual) <= 1.0e-4f) {
            fiberLength = acceptedFiber;
            fiberVelocity = 0.0f;
            tendonForce = max(0.0f, stationaryTension);
            equilibriumResidual = stationaryResidual;
            return isfinite(fiberLength) && isfinite(tendonForce);
        }
    }
    float lower = min(0.05f * optimalFiberLength, 0.5f * predictedPath) - acceptedFiber;
    float upper = predictedPath - acceptedFiber;
    if (!(lower < upper)) return false;
    float normalizedTension = 0.0f;
    for (uint iteration = 0u; iteration < 48u; ++iteration) {
        const float candidate = 0.5f * (lower + upper);
        if (candidate == lower || candidate == upper) break;
        const float residual = compliantFiberIncrementResidual(candidate, acceptedFiber,
            predictedPath, timestep, activation, muscle, normalizedTension);
        if (residual > 0.0f) lower = candidate;
        else upper = candidate;
    }
    float increment = 0.5f * (lower + upper);
    equilibriumResidual = compliantFiberIncrementResidual(increment, acceptedFiber,
        predictedPath, timestep, activation, muscle, normalizedTension);
    tendonForce = max(0.0f, normalizedTension);
    fiberLength = acceptedFiber + increment;
    fiberVelocity = increment / timestep;
    return isfinite(fiberLength) && isfinite(fiberVelocity) &&
        isfinite(tendonForce) && isfinite(equilibriumResidual);
}
inline float muscleBias(const float pathLength, const MRMujocoMuscleGPU muscle) {
    const float optimum = (muscle.lengthRangeAndAcceleration.y - muscle.lengthRangeAndAcceleration.x) /
        max(kMinimum, parameter(muscle.biasParameters, 1u) - parameter(muscle.biasParameters, 0u));
    const float normalized = parameter(muscle.biasParameters, 0u) +
        (pathLength - muscle.lengthRangeAndAcceleration.x) / max(kMinimum, optimum);
    const float upperMid = 0.5f * (1.0f + parameter(muscle.biasParameters, 5u));
    const float scale = forceScale(muscle.biasParameters, muscle.lengthRangeAndAcceleration.z);
    if (normalized <= 1.0f) return 0.0f;
    if (normalized <= upperMid) { const float x = (normalized - 1.0f) / max(kMinimum, upperMid - 1.0f); return -scale * parameter(muscle.biasParameters, 7u) * 0.5f * x * x; }
    const float x = (normalized - upperMid) / max(kMinimum, upperMid - 1.0f);
    return -scale * parameter(muscle.biasParameters, 7u) * (0.5f + x);
}
inline float activationDerivative(
    const MRMujocoMuscleGPU muscle,
    const float excitation,
    const float activation
) {
    const float control = clamp(excitation, 0.0f, 1.0f);
    const float act = clamp(activation, 0.0f, 1.0f);
    const float activationTime = parameter(muscle.dynamicParameters, 0u) * (0.5f + 1.5f * act);
    const float deactivationTime = parameter(muscle.dynamicParameters, 1u) / (0.5f + 1.5f * act);
    const float excess = control - activation;
    const float smoothing = parameter(muscle.dynamicParameters, 2u);
    const float tau = smoothing < kMinimum
        ? (excess > 0.0f ? activationTime : deactivationTime)
        : deactivationTime + (activationTime - deactivationTime) /
            (1.0f + exp(-(excess / smoothing + 0.5f)));
    return excess / max(kMinimum, tau);
}
}
kernel void mr_mujoco_muscle_angular_jacobians(
    device const MRArticulatedBodyPoseGPU* bodyPoses [[buffer(0)]],
    device const float* pointJacobians [[buffer(1)]],
    constant MRMujocoMuscleReferenceDispatchGPU& dispatch [[buffer(2)]],
    device float4* angularCache [[buffer(3)]],
    uint globalIndex [[thread_position_in_grid]]
) {
    if (dispatch.dofCount == 0u || dispatch.bodyPoseStride == 0u) return;
    const ulong bodyDofs = ulong(dispatch.bodyPoseStride) * dispatch.dofCount;
    const ulong total = ulong(dispatch.environmentCount) * bodyDofs;
    if (ulong(globalIndex) >= total) return;
    const uint environment = uint(ulong(globalIndex) / bodyDofs);
    const uint localBody = uint((ulong(globalIndex) % bodyDofs) / dispatch.dofCount);
    const uint dof = globalIndex % dispatch.dofCount;
    const uint bodyPoint = dispatch.bodyJacobianPointOffset +
        localBody * dispatch.bodyJacobianPointStride;
    const uint centerBase = environment * dispatch.pointJacobianStride +
        bodyPoint * 3u * dispatch.dofCount;
    const float4 orientation = bodyPoses[
        environment * dispatch.bodyPoseStride + localBody].orientation;
    const float3 worldAxes[3] = {
        quaternionRotate(orientation, float3(1.0f, 0.0f, 0.0f)),
        quaternionRotate(orientation, float3(0.0f, 1.0f, 0.0f)),
        quaternionRotate(orientation, float3(0.0f, 0.0f, 1.0f)),
    };
    const float3 centerJacobian = float3(
        pointJacobians[centerBase + dof],
        pointJacobians[centerBase + dispatch.dofCount + dof],
        pointJacobians[centerBase + 2u * dispatch.dofCount + dof]
    );
    float3 angularJacobian = float3(0.0f);
    for (uint axis = 0u; axis < 3u; ++axis) {
        const uint axisBase = centerBase +
            (axis + 1u) * 3u * dispatch.dofCount;
        const float3 axisJacobian = float3(
            pointJacobians[axisBase + dof],
            pointJacobians[axisBase + dispatch.dofCount + dof],
            pointJacobians[axisBase + 2u * dispatch.dofCount + dof]
        );
        angularJacobian += 0.5f * cross(
            worldAxes[axis], axisJacobian - centerJacobian);
    }
    angularCache[globalIndex] = float4(angularJacobian, 0.0f);
}
kernel void mr_mujoco_muscle_reference(
    device const float* generalizedVelocities [[buffer(7)]],
    device const MRArticulatedBodyPoseGPU* bodyPoses [[buffer(8)]],
    device const float* pointJacobians [[buffer(11)]],
    device const MRMujocoMuscleReferenceDispatchGPU& dispatch [[buffer(24)]],
    device const MRMujocoMuscleGPU* muscles [[buffer(25)]],
    device const MRMujocoMuscleStateGPU* states [[buffer(26)]],
    device const MRMujocoMuscleSiteGPU* sites [[buffer(27)]],
    device const MRMujocoMuscleWrapGPU* wraps [[buffer(28)]],
    device const MRMujocoMuscleRouteNodeGPU* routes [[buffer(29)]],
    device MRMujocoMuscleResultGPU* results [[buffer(30)]],
    device float* muscleGeneralizedForces [[buffer(23)]],
    uint globalIndex [[thread_position_in_grid]]
) {
    const uint dofLane = 0u;
    const uint dofStride = 1u;
    device const float4* bodyPositionLow = nullptr;
    if (dispatch.muscleCount == 0u || globalIndex >= dispatch.environmentCount * dispatch.muscleCount) return;
    const uint environment = globalIndex / dispatch.muscleCount;
    const uint muscleIndex = globalIndex - environment * dispatch.muscleCount;
    MRMujocoMuscleResultGPU result{};
    result.status = MR_MUJOCO_MUSCLE_REFERENCE_INVALID_PROGRAM;
    result.environment = environment;
    result.muscleIndex = muscleIndex;
    if (dispatch.abiVersion != 4u ||
        dispatch.bodyPoseStride == 0u || dispatch.dofCount == 0u ||
        dispatch.pointJacobianStride == 0u ||
        dispatch.bodyJacobianPointStride != 4u ||
        !finite4(dispatch.timestepSecondsAndReserved) ||
        any(dispatch.timestepSecondsAndReserved.yzw != float3(0.0f)) ||
        muscleIndex >= dispatch.muscleCount) {
        if (dofLane == 0u) results[globalIndex] = result; return;
    }
    const uint forceBase = globalIndex * dispatch.dofCount;
    for (uint dof = dofLane; dof < dispatch.dofCount; dof += dofStride) {
        muscleGeneralizedForces[forceBase + dof] = 0.0f;
    }
    const MRMujocoMuscleGPU muscle = muscles[muscleIndex];
    const uint routeOffset = muscle.route.x;
    const uint routeCount = muscle.route.y;
    if (routeCount < 2u || routeOffset > dispatch.routeNodeCount ||
        routeCount > dispatch.routeNodeCount - routeOffset || muscle.route.z != 0u || muscle.route.w != 0u ||
        !finite4(muscle.lengthRangeAndAcceleration) || !finite4(muscle.controlRange) ||
        !finite4(muscle.compliantArchitecture0) ||
        !finite4(muscle.compliantArchitecture1) ||
        muscle.lengthRangeAndAcceleration.w != 0.0f || muscle.controlRange.z != 0.0f || muscle.controlRange.w != 0.0f) {
        if (dofLane == 0u) results[globalIndex] = result; return;
    }
    for (uint block = 0u; block < 3u; ++block) {
        if (!finite4(muscle.gainParameters[block]) || !finite4(muscle.biasParameters[block]) || !finite4(muscle.dynamicParameters[block]) ||
            (block == 2u && (any(muscle.gainParameters[block].zw != float2(0.0f)) || any(muscle.biasParameters[block].zw != float2(0.0f)) || any(muscle.dynamicParameters[block].zw != float2(0.0f))))) {
            if (dofLane == 0u) results[globalIndex] = result; return;
        }
    }
    const MRMujocoMuscleStateGPU state = states[globalIndex];
    if (!finite4(state.excitationAndActivation) || state.excitationAndActivation.z < 0.0f ||
        (!compliantArchitecture(muscle) &&
         (state.excitationAndActivation.z != 0.0f || state.excitationAndActivation.w != 0.0f))) {
        result.status = MR_MUJOCO_MUSCLE_REFERENCE_INVALID_STATE; if (dofLane == 0u) results[globalIndex] = result; return;
    }
    float totalLength = 0.0f;
    uint appliedWraps = 0u;
    uint cursor = 0u;
    bool validPath = true;
    float3 originLengthGradient = float3(0.0f);
    float3 insertionLengthGradient = float3(0.0f);
    bool originGradientObserved = false;
    bool insertionGradientObserved = false;
    while (cursor + 1u < routeCount) {
        const MRMujocoMuscleRouteNodeGPU firstNode = routes[routeOffset + cursor];
        const MRMujocoMuscleRouteNodeGPU nextNode = routes[routeOffset + cursor + 1u];
        MRSourcePoint firstWorld;
        if (firstNode.type != MR_MUJOCO_MUSCLE_ROUTE_SITE || firstNode.reserved0 != 0u ||
            !siteWorld(environment, dispatch, bodyPoses, bodyPositionLow, sites, firstNode.targetIndex, firstWorld)) { validPath = false; break; }
        if (nextNode.type == MR_MUJOCO_MUSCLE_ROUTE_SITE) {
            MRSourcePoint secondWorld;
            if (nextNode.reserved0 != 0u || !siteWorld(environment, dispatch, bodyPoses, bodyPositionLow, sites, nextNode.targetIndex, secondWorld)) { validPath = false; break; }
            const MRMujocoMuscleSiteGPU secondSite = sites[nextNode.targetIndex];
            const float3 endpointDifference = secondWorld - firstWorld;
            const float endpointDistance = length(endpointDifference);
            const float3 endpointDirection = endpointDistance > kMinimum
                ? endpointDifference / endpointDistance : float3(0.0f);
            if (cursor == 0u) {
                originLengthGradient = -endpointDirection;
                originGradientObserved = true;
            }
            if (cursor + 1u == routeCount - 1u) {
                insertionLengthGradient = endpointDirection;
                insertionGradientObserved = true;
            }
            if (!addSegmentLengthJacobian(
                    environment, dispatch, bodyPoses, bodyPositionLow, pointJacobians,
                    firstNode.targetIndex < dispatch.siteCount
                        ? sites[firstNode.targetIndex].bodyIndex
                        : 0xffffffffu,
                    firstWorld, secondSite.bodyIndex, secondWorld,
                    muscleGeneralizedForces + forceBase
                                            , dofLane, dofStride
                )) { validPath = false; break; }
            totalLength += length(secondWorld - firstWorld); cursor += 1u; continue;
        }
        if ((nextNode.type != MR_MUJOCO_MUSCLE_ROUTE_SPHERE && nextNode.type != MR_MUJOCO_MUSCLE_ROUTE_CYLINDER) ||
            nextNode.reserved0 != 0u || nextNode.targetIndex >= dispatch.wrapCount || cursor + 2u >= routeCount) { validPath = false; break; }
        const MRMujocoMuscleRouteNodeGPU lastNode = routes[routeOffset + cursor + 2u];
        MRSourcePoint lastWorld;
        if (lastNode.type != MR_MUJOCO_MUSCLE_ROUTE_SITE || lastNode.reserved0 != 0u ||
            !siteWorld(environment, dispatch, bodyPoses, bodyPositionLow, sites, lastNode.targetIndex, lastWorld)) { validPath = false; break; }
        const MRMujocoMuscleWrapGPU wrap = wraps[nextNode.targetIndex];
        if (wrap.type != nextNode.type || wrap.bodyIndex < dispatch.articulationFirstBody ||
            wrap.bodyIndex - dispatch.articulationFirstBody >= dispatch.bodyPoseStride || wrap.reserved0 != 0u || wrap.reserved1 != 0u ||
            !finite4(wrap.localCenter) || !finite4(wrap.rotationRow0) || !finite4(wrap.rotationRow1) || !finite4(wrap.rotationRow2) || !finite4(wrap.radius) ||
            wrap.localCenter.w != 0.0f || wrap.rotationRow0.w != 0.0f || wrap.rotationRow1.w != 0.0f || wrap.rotationRow2.w != 0.0f ||
            wrap.radius.y != 0.0f || wrap.radius.z != 0.0f || wrap.radius.w != 0.0f || !(wrap.radius.x > kMinimum)) { validPath = false; break; }
        const MRArticulatedBodyPoseGPU pose = bodyPoses[environment * dispatch.bodyPoseStride + wrap.bodyIndex - dispatch.articulationFirstBody];
        const MRSourcePoint center = mrSourceBodyPoint(pose.position,bodyPositionLow,
            environment*dispatch.bodyPoseStride+wrap.bodyIndex-dispatch.articulationFirstBody) + mrSourceRotate(pose.orientation, wrap.localCenter.xyz);
        const float3 localFirst = matrixTransposeApply(wrap, quaternionConjugateRotate(pose.orientation, firstWorld - center));
        const float3 localLast = matrixTransposeApply(wrap, quaternionConjugateRotate(pose.orientation, lastWorld - center));
        const float3 basis0 = nextNode.type == MR_MUJOCO_MUSCLE_ROUTE_SPHERE ? normalize(localFirst) : float3(1.0f, 0.0f, 0.0f);
        float3 normal = cross(localFirst, localLast);
        if (nextNode.type == MR_MUJOCO_MUSCLE_ROUTE_SPHERE) {
            if (length(normal) < kMinimum) {
                const uint selected = abs(basis0.y) > abs(basis0.x) && abs(basis0.y) > abs(basis0.z) ? 1u : (abs(basis0.z) > abs(basis0.x) && abs(basis0.z) > abs(basis0.y) ? 2u : 0u);
                const float3 alternate = selected == 0u ? float3(0.0f, 1.0f, 1.0f) : (selected == 1u ? float3(1.0f, 0.0f, 1.0f) : float3(1.0f, 1.0f, 0.0f));
                normal = cross(basis0, alternate);
            }
            normal = normalize(normal);
        }
        const float3 basis1 = nextNode.type == MR_MUJOCO_MUSCLE_ROUTE_SPHERE ? normalize(cross(normal, basis0)) : float3(0.0f, 1.0f, 0.0f);
        const float2 projectedFirst = float2(dot(localFirst, basis0), dot(localFirst, basis1));
        const float2 projectedLast = float2(dot(localLast, basis0), dot(localLast, basis1));
        const bool hasSideSite = nextNode.sideSiteIndex != 0xffffffffu;
        bool hasProjectedSide = hasSideSite;
        bool sideInsideWrap = false;
        float2 side{};
        float3 localSide{};
        if (hasSideSite) {
            MRSourcePoint sideWorld;
            if (!siteWorld(environment, dispatch, bodyPoses, bodyPositionLow, sites, nextNode.sideSiteIndex, sideWorld)) { validPath = false; break; }
            localSide = matrixTransposeApply(wrap, quaternionConjugateRotate(pose.orientation, sideWorld - center));
            side = float2(dot(localSide, basis0), dot(localSide, basis1));
            sideInsideWrap = length(localSide) < wrap.radius.x;
            if (length(side) < kMinimum) hasProjectedSide = false; else side = wrap.radius.x * normalize(side);
        }
        float4 contact{}; float wrappingLength = 0.0f;
        bool wrapped = false;
        if (hasSideSite && sideInsideWrap) {
            wrapped = wrapInside(projectedFirst, projectedLast, wrap.radius.x, contact);
            wrappingLength = 0.0f;
        } else {
            wrapped = wrapCircle(projectedFirst, projectedLast, hasProjectedSide, side, wrap.radius.x, contact, wrappingLength);
        }
        if (!wrapped) {
            const float3 endpointDifference = lastWorld - firstWorld;
            const float endpointDistance = length(endpointDifference);
            const float3 endpointDirection = endpointDistance > kMinimum
                ? endpointDifference / endpointDistance : float3(0.0f);
            if (cursor == 0u) {
                originLengthGradient = -endpointDirection;
                originGradientObserved = true;
            }
            if (cursor + 2u == routeCount - 1u) {
                insertionLengthGradient = endpointDirection;
                insertionGradientObserved = true;
            }
            if (!addSegmentLengthJacobian(
                    environment, dispatch, bodyPoses, bodyPositionLow, pointJacobians,
                    sites[firstNode.targetIndex].bodyIndex, firstWorld,
                    sites[lastNode.targetIndex].bodyIndex, lastWorld,
                    muscleGeneralizedForces + forceBase
                                            , dofLane, dofStride
                )) { validPath = false; break; }
            totalLength += length(lastWorld - firstWorld);
        } else {
            float3 tangentFirst = basis0 * contact.x + basis1 * contact.y;
            float3 tangentLast = basis0 * contact.z + basis1 * contact.w;
            if (nextNode.type == MR_MUJOCO_MUSCLE_ROUTE_CYLINDER) {
                const float firstLeg = length(projectedFirst - contact.xy);
                const float secondLeg = length(projectedLast - contact.zw);
                const float denominator = firstLeg + wrappingLength + secondLeg;
                if (!(denominator > kMinimum)) { validPath = false; break; }
                tangentFirst.z = localFirst.z + (localLast.z - localFirst.z) * firstLeg / denominator;
                tangentLast.z = localFirst.z + (localLast.z - localFirst.z) * (firstLeg + wrappingLength) / denominator;
                wrappingLength = sqrt(wrappingLength * wrappingLength + (tangentLast.z - tangentFirst.z) * (tangentLast.z - tangentFirst.z));
            }
            const MRSourcePoint worldTangentFirst = center + mrSourceRotate(pose.orientation, matrixApply(wrap, tangentFirst));
            const MRSourcePoint worldTangentLast = center + mrSourceRotate(pose.orientation, matrixApply(wrap, tangentLast));
            if (cursor == 0u) {
                const float distance = length(worldTangentFirst - firstWorld);
                originLengthGradient = distance > kMinimum
                    ? (firstWorld - worldTangentFirst) / distance : float3(0.0f);
                originGradientObserved = true;
            }
            if (cursor + 2u == routeCount - 1u) {
                const float distance = length(lastWorld - worldTangentLast);
                insertionLengthGradient = distance > kMinimum
                    ? (lastWorld - worldTangentLast) / distance : float3(0.0f);
                insertionGradientObserved = true;
            }
            if (!addSegmentLengthJacobian(
                    environment, dispatch, bodyPoses, bodyPositionLow, pointJacobians,
                    sites[firstNode.targetIndex].bodyIndex, firstWorld,
                    wrap.bodyIndex, worldTangentFirst,
                    muscleGeneralizedForces + forceBase
                                            , dofLane, dofStride
                ) || !addSegmentLengthJacobian(
                    environment, dispatch, bodyPoses, bodyPositionLow, pointJacobians,
                    wrap.bodyIndex, worldTangentFirst,
                    wrap.bodyIndex, worldTangentLast,
                    muscleGeneralizedForces + forceBase
                                            , dofLane, dofStride
                ) || !addSegmentLengthJacobian(
                    environment, dispatch, bodyPoses, bodyPositionLow, pointJacobians,
                    wrap.bodyIndex, worldTangentLast,
                    sites[lastNode.targetIndex].bodyIndex, lastWorld,
                    muscleGeneralizedForces + forceBase
                                            , dofLane, dofStride
                )) { validPath = false; break; }
            totalLength += length(worldTangentFirst - firstWorld) + wrappingLength + length(lastWorld - worldTangentLast);
            ++appliedWraps;
        }
        cursor += 2u;
    }
    if (!validPath || cursor != routeCount - 1u || !(totalLength > kMinimum) ||
        !isfinite(totalLength) || !originGradientObserved || !insertionGradientObserved ||
        !all(isfinite(originLengthGradient)) || !all(isfinite(insertionLengthGradient))) {
        result.status = MR_MUJOCO_MUSCLE_REFERENCE_INVALID_PATH;
        if (dofLane == 0u) results[globalIndex] = result;
        return;
    }
    float pathVelocity = 0.0f;
    const uint velocityBase = environment * dispatch.dofCount;
    if (dofLane == 0u) {
        for (uint dof = 0u; dof < dispatch.dofCount; ++dof) {
            pathVelocity += muscleGeneralizedForces[forceBase + dof] *
                generalizedVelocities[velocityBase + dof];
        }
    }
    const float derivative = activationDerivative(muscle, state.excitationAndActivation.x, state.excitationAndActivation.y);
    float force = 0.0f;
    float activeForce = 0.0f;
    float candidateFiberLength = 0.0f;
    float candidateFiberVelocity = 0.0f;
    float tendonTension = 0.0f;
    float equilibriumResidual = 0.0f;
    if (compliantArchitecture(muscle)) {
        if (!solveCompliantFiber(
                totalLength,
                pathVelocity,
                dispatch.timestepSecondsAndReserved.x,
                state.excitationAndActivation.y,
                state,
                muscle,
                candidateFiberLength,
                candidateFiberVelocity,
                tendonTension,
                equilibriumResidual
            )) {
            result.status = MR_MUJOCO_MUSCLE_REFERENCE_NONFINITE_RESULT;
            if (dofLane == 0u) results[globalIndex] = result;
            return;
        }
        const float maximumForce = forceScale(
            muscle.gainParameters, muscle.lengthRangeAndAcceleration.z
        );
        force = -maximumForce * tendonTension;
        activeForce = force;
    } else {
        force = muscleGain(totalLength, pathVelocity, muscle) *
            state.excitationAndActivation.y + muscleBias(totalLength, muscle);
        activeForce = muscleGain(totalLength, pathVelocity, muscle) *
            state.excitationAndActivation.y;
    }
    if (!isfinite(derivative) || !isfinite(force) || !isfinite(pathVelocity)) { result.status = MR_MUJOCO_MUSCLE_REFERENCE_NONFINITE_RESULT; if (dofLane == 0u) results[globalIndex] = result; return; }
    bool finiteForceRows = true;
    for (uint dof = dofLane; dof < dispatch.dofCount; dof += dofStride) {
        muscleGeneralizedForces[forceBase + dof] *= force;
        if (!isfinite(muscleGeneralizedForces[forceBase + dof])) {
            result.status = MR_MUJOCO_MUSCLE_REFERENCE_NONFINITE_RESULT;
            if (dofLane == 0u) results[globalIndex] = result;
            return;
        }
    }
    result.status = MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS;
    result.appliedWrapCount = appliedWraps;
    result.pathForceAndActivationDerivative = float4(totalLength, pathVelocity, force, derivative);
    result.endpointLengthGradients[0] = float4(originLengthGradient, 0.0f);
    result.endpointLengthGradients[1] = float4(insertionLengthGradient, 0.0f);
    result.activeForceAndReserved = float4(activeForce, 0.0f, 0.0f, 0.0f);
    result.fiberStateTendonForceResidual = float4(
        candidateFiberLength,
        candidateFiberVelocity,
        tendonTension,
        equilibriumResidual
    );
    if (dofLane == 0u) results[globalIndex] = result;
}
kernel void mr_mujoco_muscle_active_force_rows(
    device const MRMujocoMuscleGPU* muscles [[buffer(0)]],
    device const MRMujocoMuscleStateGPU* states [[buffer(1)]],
    device MRMujocoMuscleResultGPU* results [[buffer(2)]],
    device float* muscleGeneralizedForces [[buffer(3)]],
    constant MRMujocoMuscleActiveForceDispatchGPU& dispatch [[buffer(4)]],
    uint globalIndex [[thread_position_in_grid]]
) {
    if (dispatch.abiVersion != 1u ||
        dispatch.muscleCount == 0u || dispatch.environmentCount == 0u ||
        dispatch.dofCount == 0u ||
        globalIndex >= dispatch.environmentCount * dispatch.muscleCount) {
        return;
    }
    device MRMujocoMuscleResultGPU& result = results[globalIndex];
    if (result.status != MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS) return;
    const uint muscleIndex = globalIndex % dispatch.muscleCount;
    const float totalForce = result.pathForceAndActivationDerivative.z;
    const MRMujocoMuscleGPU muscle = muscles[muscleIndex];
    const float activeForce = compliantArchitecture(muscle)
        ? totalForce
        : muscleGain(
            result.pathForceAndActivationDerivative.x,
            result.pathForceAndActivationDerivative.y,
            muscle
        ) * states[globalIndex].excitationAndActivation.y;
    result.activeForceAndReserved = float4(activeForce, 0.0f, 0.0f, 0.0f);
    const uint forceBase = globalIndex * dispatch.dofCount;
    if (abs(totalForce) <= kMinimum) {
        if (abs(activeForce) > kMinimum) {
            result.status = MR_MUJOCO_MUSCLE_REFERENCE_NONFINITE_RESULT;
        } else {
            for (uint dof = 0u; dof < dispatch.dofCount; ++dof) {
                muscleGeneralizedForces[forceBase + dof] = 0.0f;
            }
        }
        return;
    }
    const float scale = activeForce / totalForce;
    if (!isfinite(scale)) {
        result.status = MR_MUJOCO_MUSCLE_REFERENCE_NONFINITE_RESULT;
        return;
    }
    for (uint dof = 0u; dof < dispatch.dofCount; ++dof) {
        muscleGeneralizedForces[forceBase + dof] *= scale;
    }
}
kernel void mr_mujoco_muscle_reduce(
    device const MRMujocoMuscleReferenceDispatchGPU& dispatch [[buffer(24)]],
    device float* muscleAndGeneralizedForces [[buffer(23)]],
    uint globalIndex [[thread_position_in_grid]]
) {
    if (dispatch.abiVersion != 4u ||
        dispatch.muscleCount == 0u || dispatch.dofCount == 0u ||
        globalIndex >= dispatch.environmentCount * dispatch.dofCount) return;
    const uint environment = globalIndex / dispatch.dofCount;
    const uint dof = globalIndex - environment * dispatch.dofCount;
    float total = 0.0f;
    const uint muscleBase = environment * dispatch.muscleCount * dispatch.dofCount;
    for (uint muscle = 0u; muscle < dispatch.muscleCount; ++muscle) {
        total += muscleAndGeneralizedForces[
            muscleBase + muscle * dispatch.dofCount + dof
        ];
    }
    const uint outputBase = dispatch.environmentCount *
        dispatch.muscleCount * dispatch.dofCount;
    muscleAndGeneralizedForces[outputBase + globalIndex] = total;
}
kernel void mr_mujoco_muscle_activation_step(
    device MRMujocoMuscleStateGPU* states [[buffer(0)]],
    device const MRMujocoMuscleResultGPU* results [[buffer(1)]],
    constant MRMujocoMuscleActivationDispatchGPU& dispatch [[buffer(2)]],
    uint globalIndex [[thread_position_in_grid]]
) {
    const bool exactFirstOrderHold =
        dispatch.abiVersion == 3u;
    if ((dispatch.abiVersion != 2u &&
         !exactFirstOrderHold) ||
        dispatch.reserved0 != 0u || dispatch.reserved1 != 0u ||
        dispatch.timestepSecondsAndReserved.y != 0.0f ||
        dispatch.timestepSecondsAndReserved.z != 0.0f ||
        dispatch.timestepSecondsAndReserved.w != 0.0f ||
        !isfinite(dispatch.timestepSecondsAndReserved.x) ||
        !(dispatch.timestepSecondsAndReserved.x > 0.0f) ||
        globalIndex >= dispatch.stateCount) {
        return;
    }
    const MRMujocoMuscleStateGPU current = states[globalIndex];
    const MRMujocoMuscleResultGPU reference = results[globalIndex];
    if (reference.status != MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS ||
        !finite4(current.excitationAndActivation) ||
        current.excitationAndActivation.z < 0.0f ||
        (current.excitationAndActivation.z == 0.0f &&
         current.excitationAndActivation.w != 0.0f) ||
        !finite4(reference.pathForceAndActivationDerivative) ||
        !finite4(reference.fiberStateTendonForceResidual)) {
        return;
    }
    const float control = clamp(current.excitationAndActivation.x, 0.0f, 1.0f);
    const float activation = clamp(current.excitationAndActivation.y, 0.0f, 1.0f);
    const float excess = control - activation;
    const float derivative = reference.pathForceAndActivationDerivative.w;
    float nextActivation = activation;
    if (exactFirstOrderHold && excess != 0.0f && derivative != 0.0f) {
        const float tau = excess / derivative;
        if (!(isfinite(tau) && tau > 0.0f)) return;
        nextActivation = control - excess * exp(
            -dispatch.timestepSecondsAndReserved.x / tau
        );
    } else if (!exactFirstOrderHold) {
        nextActivation = activation +
            dispatch.timestepSecondsAndReserved.x * derivative;
    }
    nextActivation = clamp(nextActivation, 0.0f, 1.0f);
    if (!isfinite(nextActivation)) return;
    MRMujocoMuscleStateGPU next = current;
    next.excitationAndActivation.y = nextActivation;
    if (reference.fiberStateTendonForceResidual.x > 0.0f) {
        next.excitationAndActivation.z =
            reference.fiberStateTendonForceResidual.x;
        next.excitationAndActivation.w =
            reference.fiberStateTendonForceResidual.y;
    }
    states[globalIndex] = next;
}
kernel void mr_mujoco_muscle_route_suffix_jacobian(
    device const MRArticulatedBodyPoseGPU* bodyPoses [[buffer(0)]],
    device const float* pointJacobians [[buffer(1)]],
    constant MRMujocoMuscleReferenceDispatchGPU& dispatch [[buffer(2)]],
    device const MRMujocoMuscleGPU* muscles [[buffer(3)]],
    device const MRMujocoMuscleSiteGPU* sites [[buffer(4)]],
    device const MRMujocoMuscleWrapGPU* wraps [[buffer(5)]],
    device const MRMujocoMuscleRouteNodeGPU* routes [[buffer(6)]],
    constant MRMujocoMuscleRouteCutDispatchGPU& cutDispatch [[buffer(7)]],
    device const MRMujocoMuscleRouteCutGPU* cuts [[buffer(8)]],
    device float* suffixJacobians [[buffer(9)]],
    uint globalIndex [[thread_position_in_grid]]
) {
    device const float4* bodyPositionLow = nullptr;
    if (cutDispatch.cutCount == 0u ||
        globalIndex >= dispatch.environmentCount * cutDispatch.cutCount) return;
    const uint environment = globalIndex / cutDispatch.cutCount;
    const uint cutIndex = globalIndex - environment * cutDispatch.cutCount;
    const uint outputBase = globalIndex * dispatch.dofCount;
    for (uint dof = 0u; dof < dispatch.dofCount; ++dof)
        suffixJacobians[outputBase + dof] = 0.0f;
    bool valid = dispatch.abiVersion ==
            4u &&
        cutDispatch.abiVersion == 1u &&
        cutDispatch.reserved0 == 0u && cutDispatch.reserved1 == 0u &&
        dispatch.dofCount > 0u;
    const MRMujocoMuscleRouteCutGPU cut = cuts[cutIndex];
    valid = valid && cut.reserved0 == 0u && cut.reserved1 == 0u &&
        cut.muscleIndex < dispatch.muscleCount;
    if (!valid) {
        for (uint dof = 0u; dof < dispatch.dofCount; ++dof)
            suffixJacobians[outputBase + dof] = NAN;
        return;
    }
    const MRMujocoMuscleGPU muscle = muscles[cut.muscleIndex];
    valid = muscle.route.y >= 2u &&
        muscle.route.x <= dispatch.routeNodeCount &&
        muscle.route.y <= dispatch.routeNodeCount - muscle.route.x &&
        cut.routeNodeOrdinal < muscle.route.y - 1u;
    uint cursor = cut.routeNodeOrdinal;
    while (valid && cursor + 1u < muscle.route.y) {
        const MRMujocoMuscleRouteNodeGPU firstNode =
            routes[muscle.route.x + cursor];
        const MRMujocoMuscleRouteNodeGPU nextNode =
            routes[muscle.route.x + cursor + 1u];
        MRSourcePoint firstWorld;
        if (firstNode.type != MR_MUJOCO_MUSCLE_ROUTE_SITE ||
            firstNode.reserved0 != 0u ||
            !siteWorld(environment, dispatch, bodyPoses, bodyPositionLow, sites,
                       firstNode.targetIndex, firstWorld)) {
            valid = false;
            break;
        }
        if (nextNode.type == MR_MUJOCO_MUSCLE_ROUTE_SITE) {
            MRSourcePoint secondWorld;
            if (nextNode.reserved0 != 0u ||
                !siteWorld(environment, dispatch, bodyPoses, bodyPositionLow, sites,
                           nextNode.targetIndex, secondWorld) ||
                !addSegmentLengthJacobian(
                    environment, dispatch, bodyPoses, bodyPositionLow, pointJacobians,
                    sites[firstNode.targetIndex].bodyIndex, firstWorld,
                    sites[nextNode.targetIndex].bodyIndex, secondWorld,
                    suffixJacobians + outputBase)) {
                valid = false;
                break;
            }
            ++cursor;
            continue;
        }
        if ((nextNode.type != MR_MUJOCO_MUSCLE_ROUTE_SPHERE &&
             nextNode.type != MR_MUJOCO_MUSCLE_ROUTE_CYLINDER) ||
            nextNode.reserved0 != 0u ||
            nextNode.targetIndex >= dispatch.wrapCount ||
            cursor + 2u >= muscle.route.y) {
            valid = false;
            break;
        }
        const MRMujocoMuscleRouteNodeGPU lastNode =
            routes[muscle.route.x + cursor + 2u];
        MRSourcePoint lastWorld;
        if (lastNode.type != MR_MUJOCO_MUSCLE_ROUTE_SITE ||
            lastNode.reserved0 != 0u ||
            !siteWorld(environment, dispatch, bodyPoses, bodyPositionLow, sites,
                       lastNode.targetIndex, lastWorld)) {
            valid = false;
            break;
        }
        const MRMujocoMuscleWrapGPU wrap = wraps[nextNode.targetIndex];
        if (wrap.type != nextNode.type ||
            wrap.bodyIndex < dispatch.articulationFirstBody ||
            wrap.bodyIndex - dispatch.articulationFirstBody >=
                dispatch.bodyPoseStride ||
            wrap.reserved0 != 0u || wrap.reserved1 != 0u ||
            !finite4(wrap.localCenter) || !finite4(wrap.rotationRow0) ||
            !finite4(wrap.rotationRow1) || !finite4(wrap.rotationRow2) ||
            !finite4(wrap.radius) || wrap.localCenter.w != 0.0f ||
            wrap.rotationRow0.w != 0.0f || wrap.rotationRow1.w != 0.0f ||
            wrap.rotationRow2.w != 0.0f ||
            any(wrap.radius.yzw != float3(0.0f)) ||
            !(wrap.radius.x > kMinimum)) {
            valid = false;
            break;
        }
        const MRArticulatedBodyPoseGPU pose = bodyPoses[
            environment * dispatch.bodyPoseStride +
            wrap.bodyIndex - dispatch.articulationFirstBody];
        const MRSourcePoint center = mrSourceBodyPoint(pose.position,bodyPositionLow,
            environment*dispatch.bodyPoseStride+wrap.bodyIndex-dispatch.articulationFirstBody) +
            mrSourceRotate(pose.orientation, wrap.localCenter.xyz);
        const float3 localFirst = matrixTransposeApply(
            wrap, quaternionConjugateRotate(
                pose.orientation, firstWorld - center));
        const float3 localLast = matrixTransposeApply(
            wrap, quaternionConjugateRotate(
                pose.orientation, lastWorld - center));
        const float3 basis0 = nextNode.type == MR_MUJOCO_MUSCLE_ROUTE_SPHERE
            ? normalize(localFirst) : float3(1.0f, 0.0f, 0.0f);
        float3 normal = cross(localFirst, localLast);
        if (nextNode.type == MR_MUJOCO_MUSCLE_ROUTE_SPHERE) {
            if (length(normal) < kMinimum) {
                const uint selected = abs(basis0.y) > abs(basis0.x) &&
                        abs(basis0.y) > abs(basis0.z)
                    ? 1u : (abs(basis0.z) > abs(basis0.x) &&
                            abs(basis0.z) > abs(basis0.y) ? 2u : 0u);
                const float3 alternate = selected == 0u
                    ? float3(0.0f, 1.0f, 1.0f)
                    : (selected == 1u ? float3(1.0f, 0.0f, 1.0f)
                                      : float3(1.0f, 1.0f, 0.0f));
                normal = cross(basis0, alternate);
            }
            normal = normalize(normal);
        }
        const float3 basis1 = nextNode.type == MR_MUJOCO_MUSCLE_ROUTE_SPHERE
            ? normalize(cross(normal, basis0)) : float3(0.0f, 1.0f, 0.0f);
        const float2 projectedFirst(
            dot(localFirst, basis0), dot(localFirst, basis1));
        const float2 projectedLast(
            dot(localLast, basis0), dot(localLast, basis1));
        const bool hasSideSite = nextNode.sideSiteIndex != 0xffffffffu;
        bool hasProjectedSide = hasSideSite;
        bool sideInsideWrap = false;
        float2 side{};
        float3 localSide{};
        if (hasSideSite) {
            MRSourcePoint sideWorld;
            if (!siteWorld(environment, dispatch, bodyPoses, bodyPositionLow, sites,
                           nextNode.sideSiteIndex, sideWorld)) {
                valid = false;
                break;
            }
            localSide = matrixTransposeApply(
                wrap, quaternionConjugateRotate(
                    pose.orientation, sideWorld - center));
            side = float2(dot(localSide, basis0), dot(localSide, basis1));
            sideInsideWrap = length(localSide) < wrap.radius.x;
            if (length(side) < kMinimum) hasProjectedSide = false;
            else side = wrap.radius.x * normalize(side);
        }
        float4 contact{};
        float wrappingLength = 0.0f;
        bool wrapped = hasSideSite && sideInsideWrap
            ? wrapInside(projectedFirst, projectedLast, wrap.radius.x, contact)
            : wrapCircle(projectedFirst, projectedLast, hasProjectedSide,
                         side, wrap.radius.x, contact, wrappingLength);
        if (!wrapped) {
            valid = addSegmentLengthJacobian(
                environment, dispatch, bodyPoses, bodyPositionLow, pointJacobians,
                sites[firstNode.targetIndex].bodyIndex, firstWorld,
                sites[lastNode.targetIndex].bodyIndex, lastWorld,
                suffixJacobians + outputBase);
        } else {
            float3 tangentFirst = basis0 * contact.x + basis1 * contact.y;
            float3 tangentLast = basis0 * contact.z + basis1 * contact.w;
            if (nextNode.type == MR_MUJOCO_MUSCLE_ROUTE_CYLINDER) {
                const float firstLeg = length(projectedFirst - contact.xy);
                const float secondLeg = length(projectedLast - contact.zw);
                const float denominator = firstLeg + wrappingLength + secondLeg;
                if (!(denominator > kMinimum)) {
                    valid = false;
                    break;
                }
                tangentFirst.z = localFirst.z +
                    (localLast.z - localFirst.z) * firstLeg / denominator;
                tangentLast.z = localFirst.z +
                    (localLast.z - localFirst.z) *
                        (firstLeg + wrappingLength) / denominator;
            }
            const MRSourcePoint worldTangentFirst = center + mrSourceRotate(
                pose.orientation, matrixApply(wrap, tangentFirst));
            const MRSourcePoint worldTangentLast = center + mrSourceRotate(
                pose.orientation, matrixApply(wrap, tangentLast));
            valid = addSegmentLengthJacobian(
                    environment, dispatch, bodyPoses, bodyPositionLow, pointJacobians,
                    sites[firstNode.targetIndex].bodyIndex, firstWorld,
                    wrap.bodyIndex, worldTangentFirst,
                    suffixJacobians + outputBase) &&
                addSegmentLengthJacobian(
                    environment, dispatch, bodyPoses, bodyPositionLow, pointJacobians,
                    wrap.bodyIndex, worldTangentFirst,
                    wrap.bodyIndex, worldTangentLast,
                    suffixJacobians + outputBase) &&
                addSegmentLengthJacobian(
                    environment, dispatch, bodyPoses, bodyPositionLow, pointJacobians,
                    wrap.bodyIndex, worldTangentLast,
                    sites[lastNode.targetIndex].bodyIndex, lastWorld,
                    suffixJacobians + outputBase);
        }
        cursor += 2u;
    }
    valid = valid && cursor + 1u == muscle.route.y;
    if (!valid) {
        for (uint dof = 0u; dof < dispatch.dofCount; ++dof)
            suffixJacobians[outputBase + dof] = NAN;
    }
}

using namespace metal;
typedef uint nm_u32;
typedef int nm_i32;
typedef ulong nm_u64;
typedef float4 nm_float4;
typedef uint4 nm_uint4;
typedef int4 nm_int4;
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
    NM_EXPR_EXPM1_MINUS_X = 22u,
    NM_EXPR_FIBER_EXP_LINEAR = 23u,
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
    NM_OBJECT_DISABLE_SELF_CONTACT = 1u << 7u,
    NM_OBJECT_DISABLE_DEFORMABLE_CONTACT = 1u << 8u,
    NM_OBJECT_FEM_MATERIAL_FRAME = 1u << 9u,
    NM_OBJECT_FEM_REGIONAL_MATERIAL = 1u << 10u,
    NM_OBJECT_FEM_REFERENCE_CONFIGURATION = 1u << 11u,
    NM_OBJECT_FEM_REFERENCE_DISPLACEMENT_GRADIENT = 1u << 12u,
};
enum NMTetrahedronFlags : nm_u32 {
    NM_TETRAHEDRON_REFERENCE_DISPLACEMENT_GRADIENT = 1u << 1u,
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
typedef struct alignas(16) NMTopologyGrowthRequestGPU {
    nm_uint4 identity;
    nm_uint4 topology;
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
    NM_RIGID_ARC = 4u,
};
enum NMRigidBindingFlags : nm_u32 {
    NM_RIGID_ARTICULATED = 1u << 0u,
    NM_RIGID_DYNAMIC = 1u << 1u,
    NM_RIGID_PUNCTURE_TIP = 1u << 2u,
    NM_RIGID_SUTURE_STRAND = 1u << 3u,
    NM_RIGID_PUNCTURE_DILATOR = 1u << 4u,
    NM_RIGID_PRESCRIBED_TRANSLATION = 1u << 5u,
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
typedef struct alignas(16) NMMatterDispatchGPU {
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
    nm_u32 femHumanAttachmentCount;
    nm_u32 femHumanAttachmentPointJacobianStride;
    nm_u32 identificationCandidateCount;
    nm_u32 eventStride;
    nm_u32 mpmGridCount;
    nm_u32 mpmBlockCount;
    nm_u32 mpmBlockLookupCount;
    nm_u32 maximumParticlesPerBlock;
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
    nm_u32 mpmActiveNodeCapacity;
    nm_u32 reservedMixed1;
    nm_u32 surfaceFaceCount;
    nm_u32 deformableContactCapacity;
    nm_u32 rigidGeneralizedCapacity;
    nm_u32 rigidQCapacity;
    nm_float4 gravityAndTimestep;
    nm_float4 numericalLimits;
} NMMatterDispatchGPU;
typedef struct alignas(16) NMMixedSolverGPU {
    nm_uint4 nonlinearIterations;
    nm_uint4 executionBudgets;
    nm_float4 residualTolerances;
    nm_float4 contactAcceptance;
    nm_float4 regularization;
    nm_float4 globalization;
} NMMixedSolverGPU;
typedef struct alignas(16) NMFEMCapacityGPU {
    nm_uint4 topology;
    nm_uint4 work;
} NMFEMCapacityGPU;
typedef struct alignas(16) NMMixedMaterialGPU {
    nm_float4 mechanics;
    nm_float4 thermal;
    nm_float4 porous;
    nm_float4 electrical;
    nm_float4 fibre;
    nm_float4 coupling;
} NMMixedMaterialGPU;
typedef struct alignas(16) NMVascularLayoutGPU {
    nm_uint4 counts;
    nm_uint4 ranges;
    nm_uint4 offsets;
    nm_uint4 clock;
    nm_uint4 cavities;
} NMVascularLayoutGPU;
typedef struct alignas(16) NMVascularCavityGPU {
    nm_uint4 identity;
    nm_uint4 faces;
    nm_u64 sourceIdentity[4];
    nm_u64 mechanicalIdentity[4];
} NMVascularCavityGPU;
typedef struct alignas(16) NMVascularCavityFaceGPU {
    nm_uint4 nodes;
    nm_uint4 identity;
} NMVascularCavityFaceGPU;
typedef struct alignas(16) NMVascularClockGPU {
    nm_u64 low;
    nm_u64 high;
} NMVascularClockGPU;
typedef struct alignas(16) NMVascularIdentityGPU {
    nm_u64 content[4];
    nm_u64 source[4];
    nm_u64 authored[4];
} NMVascularIdentityGPU;
typedef struct alignas(16) NMVascularUnknownGPU {
    nm_float4 initialAndScaling;
} NMVascularUnknownGPU;
typedef struct alignas(16) NMVascularSpeciesGPU {
    nm_uint4 identity;
    nm_float4 scaling;
} NMVascularSpeciesGPU;
typedef struct alignas(16) NMVascularCompartmentGPU {
    nm_uint4 identity;
    nm_float4 compliance;
    nm_float4 elastance;
    nm_u64 periodTicks;
    nm_u64 periodMultiplier;
    nm_float4 waveform;
    nm_float4 pressureParameters;
} NMVascularCompartmentGPU;
typedef struct alignas(16) NMVascularConnectionGPU {
    nm_uint4 identity;
    nm_float4 physical;
    nm_float4 directional;
} NMVascularConnectionGPU;
typedef struct alignas(16) NMVascularTissueGPU {
    nm_uint4 identity;
    nm_float4 physical;
    nm_uint4 region;
    nm_float4 spatialFirst;
    nm_float4 spatialSecond0;
    nm_float4 spatialSecond1;
} NMVascularTissueGPU;
typedef struct alignas(16) NMVascularExchangeGPU {
    nm_uint4 identity;
    nm_float4 physical;
} NMVascularExchangeGPU;
typedef struct alignas(16) NMVascularTissueBindingGPU {
    nm_uint4 identity;
    nm_float4 physical;
} NMVascularTissueBindingGPU;
typedef struct alignas(16) NMVascularTissueMomentGPU {
    nm_float4 firstMassMoment;
    nm_float4 secondMassMoment0;
    nm_float4 secondMassMoment1;
    nm_float4 linearMomentum;
} NMVascularTissueMomentGPU;
typedef struct alignas(16) NMVascularRangeGPU {
    nm_u32 first;
    nm_u32 count;
    nm_u32 reserved0;
    nm_u32 reserved1;
} NMVascularRangeGPU;
typedef struct alignas(16) NMFEMFieldStateGPU {
    nm_float4 primary;
    nm_float4 secondary;
} NMFEMFieldStateGPU;
typedef struct alignas(16) NMFieldBoundaryGPU {
    nm_uint4 identity;
    nm_float4 value;
    nm_float4 flux;
} NMFieldBoundaryGPU;
typedef struct alignas(16) NMFEMTopologyNodeGPU {
    nm_uint4 identity;
} NMFEMTopologyNodeGPU;
typedef struct alignas(16) NMFEMSurfaceFaceGPU {
    nm_uint4 adjacency;
    nm_uint4 sides;
} NMFEMSurfaceFaceGPU;
enum NMContinuumSurfaceKind : nm_u32 {
    NM_CONTINUUM_SURFACE_TRIANGLE = 0u,
    NM_CONTINUUM_SURFACE_POINT = 1u,
};
typedef struct alignas(16) NMContinuumSurfacePrimitiveGPU {
    nm_uint4 nodesAndObject;
    nm_uint4 identity;
    nm_float4 boundsMinimum;
    nm_float4 boundsMaximum;
} NMContinuumSurfacePrimitiveGPU;
typedef struct alignas(16) NMDeformableContactFailureGPU {
    nm_uint4 attempt;
    nm_uint4 primitivePair;
    NMContinuumSurfacePrimitiveGPU primitives[2];
    nm_float4 start[6];
    nm_float4 finish[6];
    nm_float4 diagnostics;
    nm_float4 parameters;
} NMDeformableContactFailureGPU;
typedef struct alignas(16) NMDeformableContactCandidateGPU {
    nm_uint4 identity;
} NMDeformableContactCandidateGPU;
enum NMDeformableContactKind : nm_u32 {
    NM_DEFORMABLE_CONTACT_VERTEX_TRIANGLE = 0u,
    NM_DEFORMABLE_CONTACT_EDGE_EDGE = 1u,
    NM_DEFORMABLE_CONTACT_POINT_POINT = 2u,
};
typedef struct alignas(16) NMDeformableContactGPU {
    nm_uint4 nodes;
    nm_uint4 identity;
    nm_float4 weights;
    nm_float4 timeScales;
    nm_float4 normalAndTOI;
    nm_float4 pointAndSeparation;
    nm_float4 velocityAndResponse;
    nm_uint4 source;
    nm_float4 impulseAndFriction;
    nm_float4 barrier;
    nm_float4 barrierHessianRow0;
    nm_float4 barrierHessianRow1;
    nm_float4 barrierHessianRow2;
    nm_float4 mollifierGradientRow0;
    nm_float4 mollifierGradientRow1;
    nm_float4 mollifierGradientRow2;
    nm_float4 mollifierGradientRow3;
    nm_float4 mollifier;
} NMDeformableContactGPU;
typedef struct alignas(16) NMDeformableContactHistoryGPU {
    nm_uint4 source;
    nm_float4 normalAndBarrier;
    nm_float4 laggedTangentAndFriction;
} NMDeformableContactHistoryGPU;
typedef struct alignas(16) NMFEMTopologyStateGPU {
    nm_uint4 counts;
    nm_uint4 roles;
    nm_float4 accounting;
} NMFEMTopologyStateGPU;
typedef struct alignas(16) NMCohesiveFaceGPU {
    nm_uint4 nodesAndFirst;
    nm_uint4 adjacency;
    nm_float4 state;
    nm_float4 geometry;
    nm_uint4 separatedNodes;
} NMCohesiveFaceGPU;
typedef struct alignas(16) NMMutationCommandGPU {
    nm_uint4 identity;
    nm_uint4 schedule;
    nm_float4 geometry0;
    nm_float4 geometry1;
} NMMutationCommandGPU;
typedef struct alignas(16) NMPunctureChannelGPU {
    nm_uint4 identity;
    nm_float4 originAndRadius;
    nm_float4 axisAndHalfLength;
} NMPunctureChannelGPU;
typedef struct alignas(16) NMLearnedMaterialGPU {
    nm_uint4 layout;
    nm_uint4 identity;
    nm_float4 policy;
    nm_uint4 fingerprint;
} NMLearnedMaterialGPU;
typedef struct alignas(16) NMLearnedLayerGPU {
    nm_uint4 layout;
    nm_uint4 routing;
} NMLearnedLayerGPU;
typedef struct alignas(16) NMLearnedDifferentialGPU {
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
typedef struct alignas(16) NMSolverCertificateGPU {
    nm_float4 nonlinear;
    nm_float4 contact;
    nm_float4 transport;
    nm_float4 validity;
} NMSolverCertificateGPU;
enum NMMicrostepFlags : nm_u32 {
    NM_MICROSTEP_CAPTURE_EVENTS = 1u << 0u,
    NM_MICROSTEP_FGMRES_OPERATOR = 1u << 1u,
    NM_MICROSTEP_FIELD_PRECONDITIONER = 1u << 2u,
};
typedef struct alignas(16) NMMicrostepGPU {
    nm_u32 controlStep;
    nm_u32 microtick;
    nm_u32 microtickCount;
    nm_u32 solverIteration;
    nm_u32 seedLo;
    nm_u32 seedHi;
    nm_u32 flags;
    nm_u32 reserved;
    nm_float4 time;
} NMMicrostepGPU;
typedef struct alignas(16) NMBridgeDispatchGPU {
    nm_u32 environmentCount;
    nm_u32 rigidProxyCount;
    nm_u32 currentBodyCount;
    nm_u32 currentBodyStride;
    nm_u32 bodyWrenchCount;
    nm_u32 sceneBodyCount;
    nm_u32 bodyWrenchStride;
    nm_u32 sceneStride;
    nm_u32 reactionStride;
    nm_u32 reactionBodyCount;
    nm_u32 rodNodeCount;
    nm_u32 rodNodeStride;
    nm_float4 time;
} NMBridgeDispatchGPU;
typedef struct alignas(16) NMResetPassGPU {
    nm_u32 controlStep;
    nm_u32 resetMaskStepStride;
    nm_u32 stateWidth;
    nm_u32 flags;
} NMResetPassGPU;
typedef struct alignas(16) NMIdentificationPassGPU {
    nm_u32 candidateCount;
    nm_u32 distributionCount;
    nm_u32 generation;
    nm_u32 flags;
    nm_u32 seedLo;
    nm_u32 seedHi;
    nm_u32 reserved0;
    nm_u32 reserved1;
} NMIdentificationPassGPU;
typedef struct alignas(16) NMExpressionInstructionGPU {
    nm_u32 opcode;
    nm_u32 index;
    nm_i32 integer;
    nm_u32 reserved;
    nm_float4 immediate;
} NMExpressionInstructionGPU;
typedef struct alignas(16) NMScalarProgramGPU {
    nm_u32 firstInstruction;
    nm_u32 instructionCount;
    nm_u32 maximumStack;
    nm_u32 flags;
} NMScalarProgramGPU;
typedef struct alignas(16) NMMaterialGPU {
    nm_u32 constitutiveKind;
    nm_u32 parameterOffset;
    nm_u32 parameterCount;
    nm_u32 stateCount;
    nm_u32 stressProgramOffset;
    nm_u32 tangentProgramOffset;
    nm_u32 validityProgram;
    nm_u32 flags;
    nm_u32 stateInitialOffset;
    nm_u32 stateUpdateProgramOffset;
    nm_u32 dissipationProgram;
    nm_u32 projectionKind;
    nm_u32 viscousStressProgramOffset;
    nm_u32 viscousTangentProgramOffset;
    nm_u32 implicitResidualProgramOffset;
    nm_u32 implicitJacobianProgramOffset;
    nm_u32 implicitDeformationProgramOffset;
    nm_u32 stressStateDerivativeProgramOffset;
    nm_u32 stateTransferMask;
    nm_u32 localNewtonIterations;
    nm_float4 bulk;
    nm_float4 interfaceResponse;
    nm_float4 inelastic;
    nm_float4 validity;
} NMMaterialGPU;
typedef struct alignas(16) NMParameterRangeGPU {
    nm_float4 valueAndBounds;
} NMParameterRangeGPU;
typedef struct alignas(16) NMContinuumObjectGPU {
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
    nm_uint4 solver;
    nm_float4 fidelity;
    nm_u64 materialFrameSourceIdentity[4];
    nm_u64 materialSourceIdentity[4];
    nm_u64 referenceSourceIdentity[4];
} NMContinuumObjectGPU;
typedef struct alignas(16) NMMPMGridGPU {
    nm_int4 nodeMinimumAndObject;
    nm_uint4 nodeDimensionsAndOffset;
    nm_int4 blockMinimumAndOffset;
    nm_uint4 blockDimensionsAndLookup;
    nm_float4 metrics;
} NMMPMGridGPU;
typedef struct alignas(16) NMMPMBlockGPU {
    nm_uint4 identity;
    nm_int4 coordinateAndLookup;
} NMMPMBlockGPU;
typedef struct alignas(16) NMMPMParticleKeyGPU {
    nm_uint4 identity;
} NMMPMParticleKeyGPU;
typedef struct alignas(16) NMActiveMPMBlockGPU {
    nm_u32 environment;
    nm_u32 blockIndex;
    nm_u32 firstParticle;
    nm_u32 particleCount;
} NMActiveMPMBlockGPU;
typedef struct alignas(16) NMIndirectDispatchGPU {
    nm_u32 threadgroupsX;
    nm_u32 threadgroupsY;
    nm_u32 threadgroupsZ;
    nm_u32 reserved;
} NMIndirectDispatchGPU;
typedef struct alignas(16) NMParticleStateGPU {
    nm_float4 positionAndMass;
    nm_float4 velocityAndReferenceVolume;
    nm_float4 deformationRow0;
    nm_float4 deformationRow1;
    nm_float4 deformationRow2;
    nm_float4 affineRow0;
    nm_float4 affineRow1;
    nm_float4 affineRow2;
    nm_float4 referenceAndTemperature;
    nm_uint4 identity;
} NMParticleStateGPU;
typedef struct alignas(16) NMGridNodeStateGPU {
    nm_float4 positionAndMass;
    nm_float4 velocityAndInverseMass;
    nm_float4 forceAndEnergy;
    nm_float4 candidateVelocity;
} NMGridNodeStateGPU;
typedef struct alignas(16) NMMPMStencilGPU {
    nm_u32 particleIndex;
    nm_u32 nodeIndex;
    nm_u32 localSlot;
    nm_u32 reserved;
    nm_float4 gradientAndWeight;
} NMMPMStencilGPU;
typedef struct alignas(16) NMIncidenceRangeGPU {
    nm_u32 first;
    nm_u32 count;
    nm_u32 objectIndex;
    nm_u32 reserved;
} NMIncidenceRangeGPU;
typedef struct alignas(16) NMFEMHumanAttachmentGPU {
    nm_uint4 identity;
    nm_float4 localPoint;
} NMFEMHumanAttachmentGPU;
typedef struct alignas(16) NMFEMNodeStateGPU {
    nm_float4 positionAndMass;
    nm_float4 velocityAndInverseMass;
    nm_float4 restAndFixed;
    nm_float4 deltaVelocity;
    nm_float4 referenceDisplacementAndMode;
} NMFEMNodeStateGPU;
typedef struct alignas(16) NMTetrahedronGPU {
    nm_uint4 nodes;
    nm_float4 inverseRestRow0;
    nm_float4 inverseRestRow1;
    nm_float4 inverseRestRow2;
    nm_uint4 identity;
    nm_float4 materialFrameRotation;
} NMTetrahedronGPU;
typedef struct alignas(16) NMFEMElementVectorGPU {
    nm_float4 node0;
    nm_float4 node1;
    nm_float4 node2;
    nm_float4 node3;
} NMFEMElementVectorGPU;
typedef struct alignas(16) NMFGMRESLayoutGPU {
    nm_u32 supportContactCount;
    nm_u32 supportBase;
    nm_u32 unknownCount;
    nm_u32 reserved;
    nm_u32 vascularUnknownCount;
    nm_u32 vascularBase;
    float vascularResidualTolerance;
    nm_u32 vascularReserved;
} NMFGMRESLayoutGPU;
typedef struct alignas(16) NMHumanSupportKKTGPU {
    nm_float4 residual;
    nm_float4 projectionRow0;
    nm_float4 projectionRow1;
    nm_float4 projectionRow2;
} NMHumanSupportKKTGPU;
typedef struct alignas(16) NMFGMRESStateGPU {
    nm_float4 diagnostics;
    nm_float4 nonlinear;
} NMFGMRESStateGPU;
typedef struct alignas(16) NMRigidProxyGPU {
    nm_u32 shapeKind;
    nm_u32 bodyIndex;
    nm_u32 sceneBodyIndex;
    nm_u32 materialIndex;
    nm_u32 flags;
    nm_u32 adaptiveObjectIndex;
    nm_u32 generalizedFreeBodyIndex;
    nm_u32 reserved2;
    nm_float4 localCenterAndRadius;
    nm_float4 localExtent;
    nm_float4 localOrientation;
} NMRigidProxyGPU;
typedef struct alignas(16) NMRigidStateGPU {
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
typedef struct alignas(16) NMContactPairGPU {
    nm_u32 continuumNode;
    nm_u32 rigidProxy;
    nm_u32 objectIndex;
    nm_u32 materialInterface;
} NMContactPairGPU;
typedef struct alignas(16) NMContactSampleGPU {
    nm_uint4 identity;
    nm_float4 pointAndSeparation;
    nm_float4 normalAndVelocity;
    nm_float4 admissionVelocityAndNormal;
    nm_float4 impulseAndNormal;
    nm_float4 angularImpulseAndTangent;
    nm_float4 barrier;
    nm_float4 barrierHessianRow0;
    nm_float4 barrierHessianRow1;
    nm_float4 barrierHessianRow2;
} NMContactSampleGPU;
typedef struct alignas(16) NMHumanSupportContactGPU {
    nm_uint4 identity;
    nm_float4 localPoint;
    nm_float4 frictionSlopAndStabilization;
    nm_float4 supportRadii;
    nm_float4 supportOrientation;
} NMHumanSupportContactGPU;
typedef struct alignas(16) NMHumanSupportPointQueryGPU {
    nm_u32 bodyIndex;
    nm_u32 flags;
    nm_u32 reserved0;
    nm_u32 reserved1;
    nm_float4 localPoint;
    nm_float4 worldImpulse;
    nm_float4 supportPlaneNormalAndRadius;
    nm_float4 supportRadii;
    nm_float4 supportOrientation;
} NMHumanSupportPointQueryGPU;
typedef struct alignas(16) NMHumanSupportConsequenceGPU {
    nm_uint4 identity;
    nm_float4 pointAndSeparation;
    nm_float4 impulseAndNormal;
    nm_float4 tangentVelocityAndImpulse;
} NMHumanSupportConsequenceGPU;
typedef struct alignas(16) NMHumanSupportDispatchGPU {
    nm_u32 contactCount;
    nm_u32 articulatedNv;
    nm_u32 articulationRootBody;
    nm_u32 bodyCount;
    nm_u32 bodyStride;
    nm_u32 reserved0;
    nm_u32 reserved1;
    nm_u32 reserved2;
    nm_float4 groundPointAndTimestep;
    nm_float4 groundNormal;
} NMHumanSupportDispatchGPU;
typedef struct alignas(16) NMRigidReactionGPU {
    nm_float4 impulseAndCount;
    nm_float4 angularImpulse;
} NMRigidReactionGPU;
typedef struct alignas(16) NMAdaptiveStateGPU {
    nm_u32 activeRepresentation;
    nm_u32 requestedRepresentation;
    nm_u32 stableFrames;
    nm_u32 flags;
    nm_float4 massAndError;
    nm_float4 centerAndRadius;
    nm_float4 referenceCenter;
    nm_float4 linearVelocityAndAngularSpeed;
    nm_float4 angularVelocityAndMinimumJ;
    nm_float4 orientation;
    nm_float4 inverseInertiaRow0;
    nm_float4 inverseInertiaRow1;
    nm_float4 inverseInertiaRow2;
} NMAdaptiveStateGPU;
typedef struct alignas(16) NMIdentificationDistributionGPU {
    nm_uint4 identity;
    nm_float4 momentsAndBounds;
    nm_float4 update;
} NMIdentificationDistributionGPU;
typedef struct alignas(16) NMIdentificationCandidateGPU {
    nm_u32 environment;
    nm_u32 candidate;
    nm_u32 distribution;
    nm_u32 antitheticPartner;
    nm_float4 sample;
} NMIdentificationCandidateGPU;
typedef struct alignas(16) NMSchedulerStateGPU {
    nm_u32 baseExponent;
    nm_u32 activeExponent;
    nm_u32 requestedExponent;
    nm_u32 quietFrames;
    nm_float4 physical;
    nm_float4 numerical;
    nm_float4 thresholds;
    nm_float4 timing;
} NMSchedulerStateGPU;
typedef struct alignas(16) NMEventTokenGPU {
    nm_u32 environment;
    nm_u32 objectIndex;
    nm_u32 eventClass;
    nm_u32 flags;
    nm_float4 payload;
    nm_float4 transition;
} NMEventTokenGPU;
typedef struct alignas(16) NMMatterStatusGPU {
    nm_u32 code;
    nm_u32 environment;
    nm_u32 objectIndex;
    nm_u32 failingIndex;
    nm_u32 completedMicrosteps;
    nm_u32 fgmresIterations;
    nm_u32 contactCount;
    nm_u32 eventCount;
    nm_float4 diagnostics;
} NMMatterStatusGPU;
typedef struct alignas(16) NMHumanRespirationParameters {
    nm_float4 lung;
    nm_float4 chest;
    nm_float4 geometry;
    nm_float4 environment;
    nm_float4 metabolism;
    nm_float4 oxygen;
    nm_float4 carbonDioxide;
    nm_uint4 tissueRows;
    nm_float4 tissueFractions;
    nm_uint4 topology;
    MRMujocoMuscleGPU muscles[2];
} NMHumanRespirationParameters;
typedef struct alignas(16) NMHumanBloodBalanceLedger {
    float continuityResidualSumM3;
    float continuityResidualCompensationM3;
    float physicalVolumeDeltaSumM3;
    float physicalVolumeDeltaCompensationM3;
} NMHumanBloodBalanceLedger;
typedef struct alignas(16) NMHumanRespirationState {
    nm_float4 motion;
    nm_float4 mechanics;
    MRMujocoMuscleStateGPU muscles[2];
    nm_float4 bloodGas[21];
    nm_float4 alveolarGas;
    nm_float4 deadSpaceGas;
    nm_float4 environmentGas;
    nm_float4 metabolicGas;
    nm_float4 gasBudget;
    nm_float4 observation;
    nm_float4 control;
    nm_float4 cardiacPressure;
    nm_float4 cardiacFlow;
    nm_float4 circulation;
    nm_float4 chamberVolumes;
    nm_uint4 cardiacStatus;
    nm_float4 breath;
    nm_uint4 status;
    NMHumanBloodBalanceLedger bloodBalance;
    nm_uint4 breathTiming;
    nm_float4 breathAccounting;
    nm_float4 transportStep;
} NMHumanRespirationState;
typedef struct alignas(16) NMHumanRespirationDispatch {
    nm_u32 environmentCount;
    nm_u32 vascularStride;
    nm_u32 reject;
    nm_u32 reserved;
} NMHumanRespirationDispatch;
typedef struct alignas(16) NMHumanRespirationBrainDispatch {
    nm_u64 sourceTimeMicroseconds;
    nm_u64 targetTimeMicroseconds;
    nm_u64 sourceRootIdentity;
    nm_u64 targetRootIdentity;
    nm_u32 sourceStep;
    nm_u32 environmentCount;
    float driveScale;
    nm_u32 reserved;
} NMHumanRespirationBrainDispatch;
extern "C" {
enum {
  NB_NUMI_RESPIRATORY_CHEMOREFLEX_ABI_VERSION = 1,
  NB_NUMI_RESPIRATORY_VALID_PA_O2 = 1u << 0,
  NB_NUMI_RESPIRATORY_VALID_PA_CO2 = 1u << 1,
  NB_NUMI_RESPIRATORY_INPUT_ACCEPTED = 1u << 0,
  NB_NUMI_RESPIRATORY_RECORD_VALID = 1u << 0,
  NB_NUMI_RESPIRATORY_INPUT_BYTE_COUNT = 48,
  NB_NUMI_RESPIRATORY_STATE_BYTE_COUNT = 32,
  NB_NUMI_RESPIRATORY_OUTPUT_BYTE_COUNT = 48,
  NB_NUMI_RESPIRATORY_PARAMETERS_BYTE_COUNT = 64,
};
typedef struct NBNumiRespiratoryChemoreflexInputV1 {
  uint64_t sourceTimestampMicroseconds;
  uint64_t sourceAcceptedRootFingerprint;
  uint64_t targetTimestampMicroseconds;
  uint64_t targetTransactionFingerprint;
  float paO2MillimetersMercury;
  float paCO2MillimetersMercury;
  uint32_t validityMask;
  uint32_t flags;
} NBNumiRespiratoryChemoreflexInputV1;
typedef struct NBNumiRespiratoryChemoreflexStateV1 {
  uint64_t timestampMicroseconds;
  uint64_t acceptedRootFingerprint;
  float centralCO2DriveLitresPerMinute;
  float peripheralDriveLitresPerMinute;
  float respiratoryPhaseCycles;
  uint32_t flags;
} NBNumiRespiratoryChemoreflexStateV1;
typedef struct NBNumiRespiratoryChemoreflexOutputV1 {
  uint64_t targetTimestampMicroseconds;
  uint64_t sourceAcceptedRootFingerprint;
  uint64_t targetTransactionFingerprint;
  float targetMinuteVentilationLitresPerMinute;
  float targetFrequencyBreathsPerMinute;
  float targetTidalVolumeLitres;
  float diaphragmExcitation;
  float intercostalExcitation;
  uint32_t flags;
} NBNumiRespiratoryChemoreflexOutputV1;
typedef struct NBNumiRespiratoryChemoreflexParametersV1 {
  float restingMinuteVentilationLitresPerMinute;
  float CO2ResponseSlopeLitresPerMinutePerMmHg;
  float centralCO2Fraction;
  float centralTimeConstantSeconds;
  float peripheralTimeConstantSeconds;
  float hypoxicReferencePaO2MillimetersMercury;
  float hypoxicScalePaO2MillimetersMercury;
  float hypoxicIncrementAtPaO2FortyLitresPerMinute;
  float restingFrequencyBreathsPerMinute;
  float restingTidalVolumeLitres;
  float diaphragmExcitationGain;
  float intercostalExcitationGain;
  float inspirationFractionOfCycle;
  float minimumMinuteVentilationLitresPerMinute;
  float maximumMinuteVentilationLitresPerMinute;
  float reserved;
} NBNumiRespiratoryChemoreflexParametersV1;
static inline NBNumiRespiratoryChemoreflexParametersV1
NBNumiRespiratoryChemoreflexReferenceParametersV1(void) {
  NBNumiRespiratoryChemoreflexParametersV1 value = {
    6.0f, 2.60f, 0.67f, 120.0f,
    5.0f, 95.0f, 24.5f, 40.0f,
    12.0f, 0.50f, 0.16f, 0.10f,
    0.40f, 2.0f, 40.0f, 0.0f
  };
  return value;
}
static inline NBNumiRespiratoryChemoreflexStateV1
NBNumiRespiratoryChemoreflexInitialStateV1(
  uint64_t timestampMicroseconds, uint64_t acceptedRootFingerprint) {
  NBNumiRespiratoryChemoreflexStateV1 value = {
    timestampMicroseconds, acceptedRootFingerprint,
    0.0f, 0.0f, 0.0f, NB_NUMI_RESPIRATORY_RECORD_VALID
  };
  return value;
}
}
static_assert(sizeof(NBNumiRespiratoryChemoreflexInputV1) == 48,
  "respiratory input ABI");
static_assert(sizeof(NBNumiRespiratoryChemoreflexStateV1) == 32,
  "respiratory state ABI");
static_assert(sizeof(NBNumiRespiratoryChemoreflexOutputV1) == 48,
  "respiratory output ABI");
static_assert(sizeof(NBNumiRespiratoryChemoreflexParametersV1) == 64,
  "respiratory parameter ABI");

using namespace metal;
static_assert(sizeof(NBNumiRespiratoryChemoreflexInputV1)
    == NB_NUMI_RESPIRATORY_INPUT_BYTE_COUNT,
    "respiratory input ABI drift");
static_assert(sizeof(NBNumiRespiratoryChemoreflexStateV1)
    == NB_NUMI_RESPIRATORY_STATE_BYTE_COUNT,
    "respiratory state ABI drift");
static_assert(sizeof(NBNumiRespiratoryChemoreflexOutputV1)
    == NB_NUMI_RESPIRATORY_OUTPUT_BYTE_COUNT,
    "respiratory output ABI drift");
static_assert(sizeof(NBNumiRespiratoryChemoreflexParametersV1)
    == NB_NUMI_RESPIRATORY_PARAMETERS_BYTE_COUNT,
    "respiratory parameter ABI drift");
inline bool nb_respiratory_parameters_valid(
    constant NBNumiRespiratoryChemoreflexParametersV1 &p) {
  return isfinite(p.restingMinuteVentilationLitresPerMinute)
      && p.restingMinuteVentilationLitresPerMinute > 0.0f
      && isfinite(p.CO2ResponseSlopeLitresPerMinutePerMmHg)
      && p.CO2ResponseSlopeLitresPerMinutePerMmHg > 0.0f
      && isfinite(p.centralCO2Fraction)
      && p.centralCO2Fraction >= 0.0f && p.centralCO2Fraction <= 1.0f
      && isfinite(p.centralTimeConstantSeconds)
      && p.centralTimeConstantSeconds > 0.0f
      && isfinite(p.peripheralTimeConstantSeconds)
      && p.peripheralTimeConstantSeconds > 0.0f
      && isfinite(p.hypoxicReferencePaO2MillimetersMercury)
      && p.hypoxicReferencePaO2MillimetersMercury > 40.0f
      && isfinite(p.hypoxicScalePaO2MillimetersMercury)
      && p.hypoxicScalePaO2MillimetersMercury > 0.0f
      && isfinite(p.hypoxicIncrementAtPaO2FortyLitresPerMinute)
      && p.hypoxicIncrementAtPaO2FortyLitresPerMinute >= 0.0f
      && isfinite(p.restingFrequencyBreathsPerMinute)
      && p.restingFrequencyBreathsPerMinute > 0.0f
      && isfinite(p.restingTidalVolumeLitres)
      && p.restingTidalVolumeLitres > 0.0f
      && isfinite(p.diaphragmExcitationGain)
      && p.diaphragmExcitationGain >= 0.0f
      && p.diaphragmExcitationGain <= 1.0f
      && isfinite(p.intercostalExcitationGain)
      && p.intercostalExcitationGain >= 0.0f
      && p.intercostalExcitationGain <= 1.0f
      && isfinite(p.inspirationFractionOfCycle)
      && p.inspirationFractionOfCycle > 0.0f
      && p.inspirationFractionOfCycle < 1.0f
      && isfinite(p.minimumMinuteVentilationLitresPerMinute)
      && p.minimumMinuteVentilationLitresPerMinute > 0.0f
      && isfinite(p.maximumMinuteVentilationLitresPerMinute)
      && p.maximumMinuteVentilationLitresPerMinute
          >= p.restingMinuteVentilationLitresPerMinute
      && p.minimumMinuteVentilationLitresPerMinute
          <= p.restingMinuteVentilationLitresPerMinute;
}
inline void nb_respiratory_reject(
    device NBNumiRespiratoryChemoreflexStateV1 &candidateState,
    device NBNumiRespiratoryChemoreflexOutputV1 &output) {
  candidateState.timestampMicroseconds = 0ul;
  candidateState.acceptedRootFingerprint = 0ul;
  candidateState.centralCO2DriveLitresPerMinute = 0.0f;
  candidateState.peripheralDriveLitresPerMinute = 0.0f;
  candidateState.respiratoryPhaseCycles = 0.0f;
  candidateState.flags = 0u;
  output.targetTimestampMicroseconds = 0ul;
  output.sourceAcceptedRootFingerprint = 0ul;
  output.targetTransactionFingerprint = 0ul;
  output.targetMinuteVentilationLitresPerMinute = NAN;
  output.targetFrequencyBreathsPerMinute = NAN;
  output.targetTidalVolumeLitres = NAN;
  output.diaphragmExcitation = NAN;
  output.intercostalExcitation = NAN;
  output.flags = 0u;
}
kernel void advance_numi_respiratory_chemoreflex_v1(
    constant NBNumiRespiratoryChemoreflexParametersV1 *parameters
        [[buffer(0)]],
    device const NBNumiRespiratoryChemoreflexInputV1 *input [[buffer(1)]],
    device const NBNumiRespiratoryChemoreflexStateV1 *committedStates
        [[buffer(2)]],
    device NBNumiRespiratoryChemoreflexStateV1 *candidateStates
        [[buffer(3)]],
    device NBNumiRespiratoryChemoreflexOutputV1 *outputs [[buffer(4)]],
    uint environment [[thread_position_in_grid]]) {
  const NBNumiRespiratoryChemoreflexInputV1 sample = input[environment];
  const NBNumiRespiratoryChemoreflexStateV1 baseline =
      committedStates[environment];
  device NBNumiRespiratoryChemoreflexStateV1 &nextState =
      candidateStates[environment];
  device NBNumiRespiratoryChemoreflexOutputV1 &output = outputs[environment];
  constant NBNumiRespiratoryChemoreflexParametersV1 &p = parameters[0];
  const uint32_t required =
      NB_NUMI_RESPIRATORY_VALID_PA_O2 | NB_NUMI_RESPIRATORY_VALID_PA_CO2;
  const ulong elapsedMicroseconds = sample.targetTimestampMicroseconds
      >= sample.sourceTimestampMicroseconds
      ? sample.targetTimestampMicroseconds - sample.sourceTimestampMicroseconds
      : 0ul;
  const bool inputValid = sample.flags & NB_NUMI_RESPIRATORY_INPUT_ACCEPTED;
  const bool pressuresValid = (sample.validityMask & required) == required
      && isfinite(sample.paO2MillimetersMercury)
      && isfinite(sample.paCO2MillimetersMercury)
      && sample.paO2MillimetersMercury > 0.0f
      && sample.paO2MillimetersMercury <= 300.0f
      && sample.paCO2MillimetersMercury > 0.0f
      && sample.paCO2MillimetersMercury <= 200.0f;
  const bool chronologyValid = sample.sourceTimestampMicroseconds
          == baseline.timestampMicroseconds
      && sample.sourceAcceptedRootFingerprint
          == baseline.acceptedRootFingerprint
      && sample.sourceAcceptedRootFingerprint != 0ul
      && sample.targetTimestampMicroseconds
          > sample.sourceTimestampMicroseconds
      && sample.targetTransactionFingerprint != 0ul
      && elapsedMicroseconds <= 250000ul;
  const bool baselineValid =
      (baseline.flags & NB_NUMI_RESPIRATORY_RECORD_VALID) != 0u
      && isfinite(baseline.centralCO2DriveLitresPerMinute)
      && isfinite(baseline.peripheralDriveLitresPerMinute)
      && isfinite(baseline.respiratoryPhaseCycles)
      && baseline.respiratoryPhaseCycles >= 0.0f
      && baseline.respiratoryPhaseCycles < 1.0f;
  if (!inputValid || !pressuresValid || !chronologyValid || !baselineValid
      || !nb_respiratory_parameters_valid(p)) {
    nb_respiratory_reject(nextState, output);
    return;
  }
  const float elapsedSeconds = float(elapsedMicroseconds) * 1.0e-6f;
  const float co2Error = sample.paCO2MillimetersMercury - 40.0f;
  const float centralTarget = clamp(
      p.CO2ResponseSlopeLitresPerMinutePerMmHg
          * p.centralCO2Fraction * co2Error,
      p.minimumMinuteVentilationLitresPerMinute
          - p.restingMinuteVentilationLitresPerMinute,
      p.maximumMinuteVentilationLitresPerMinute
          - p.restingMinuteVentilationLitresPerMinute);
  const float peripheralCo2Target =
      p.CO2ResponseSlopeLitresPerMinutePerMmHg
          * (1.0f - p.centralCO2Fraction) * co2Error;
  float hypoxicTarget = 0.0f;
  if (sample.paO2MillimetersMercury
      < p.hypoxicReferencePaO2MillimetersMercury) {
    const float hypoxiaRise = exp(
        (p.hypoxicReferencePaO2MillimetersMercury
            - sample.paO2MillimetersMercury)
            / p.hypoxicScalePaO2MillimetersMercury) - 1.0f;
    const float riseAtForty = exp(
        (p.hypoxicReferencePaO2MillimetersMercury - 40.0f)
            / p.hypoxicScalePaO2MillimetersMercury) - 1.0f;
    hypoxicTarget = p.hypoxicIncrementAtPaO2FortyLitresPerMinute
        * max(hypoxiaRise, 0.0f) / max(riseAtForty, 1.0e-6f);
  }
  const float peripheralTarget = peripheralCo2Target + hypoxicTarget;
  const float centralAlpha = 1.0f - exp(
      -elapsedSeconds / p.centralTimeConstantSeconds);
  const float peripheralAlpha = 1.0f - exp(
      -elapsedSeconds / p.peripheralTimeConstantSeconds);
  const float centralDrive = baseline.centralCO2DriveLitresPerMinute
      + centralAlpha * (centralTarget
          - baseline.centralCO2DriveLitresPerMinute);
  const float peripheralDrive = baseline.peripheralDriveLitresPerMinute
      + peripheralAlpha * (peripheralTarget
          - baseline.peripheralDriveLitresPerMinute);
  const float ventilation = clamp(
      p.restingMinuteVentilationLitresPerMinute
          + centralDrive + peripheralDrive,
      p.minimumMinuteVentilationLitresPerMinute,
      p.maximumMinuteVentilationLitresPerMinute);
  const float frequency = clamp(
      p.restingFrequencyBreathsPerMinute
          * sqrt(ventilation / p.restingMinuteVentilationLitresPerMinute),
      4.0f, 40.0f);
  const float tidalVolume = clamp(
      ventilation / frequency,
      0.15f, p.maximumMinuteVentilationLitresPerMinute / frequency);
  const float phase = fract(
      baseline.respiratoryPhaseCycles
          + elapsedSeconds * frequency / 60.0f);
  const float inspirationPhase = phase / p.inspirationFractionOfCycle;
  const float inspiratoryEnvelope = phase < p.inspirationFractionOfCycle
      ? sin(M_PI_F * clamp(inspirationPhase, 0.0f, 1.0f))
      : 0.0f;
  const float amplitudeScale = sqrt(
      tidalVolume / p.restingTidalVolumeLitres);
  nextState.timestampMicroseconds = sample.targetTimestampMicroseconds;
  nextState.acceptedRootFingerprint = sample.targetTransactionFingerprint;
  nextState.centralCO2DriveLitresPerMinute = centralDrive;
  nextState.peripheralDriveLitresPerMinute = peripheralDrive;
  nextState.respiratoryPhaseCycles = phase;
  nextState.flags = NB_NUMI_RESPIRATORY_RECORD_VALID;
  output.targetTimestampMicroseconds = sample.targetTimestampMicroseconds;
  output.sourceAcceptedRootFingerprint = sample.sourceAcceptedRootFingerprint;
  output.targetTransactionFingerprint = sample.targetTransactionFingerprint;
  output.targetMinuteVentilationLitresPerMinute = ventilation;
  output.targetFrequencyBreathsPerMinute = frequency;
  output.targetTidalVolumeLitres = tidalVolume;
  output.diaphragmExcitation = clamp(
      p.diaphragmExcitationGain * amplitudeScale * inspiratoryEnvelope,
      0.0f, 1.0f);
  output.intercostalExcitation = clamp(
      p.intercostalExcitationGain * amplitudeScale * inspiratoryEnvelope,
      0.0f, 1.0f);
  output.flags = NB_NUMI_RESPIRATORY_RECORD_VALID;
}
typedef struct alignas(16) MRNumiHumanTimedRootForceGPU {
    mr_uint4 stepWindow;
    mr_float4 forceNewtons;
} MRNumiHumanTimedRootForceGPU;
inline bool mrNumiHumanTimedRootForceConfigured(
    const MRNumiHumanTimedRootForceGPU program
) {
    return program.stepWindow.x != 0u || program.stepWindow.y != 0u ||
        program.stepWindow.z != 0u || program.stepWindow.w != 0u ||
        program.forceNewtons.x != 0.0f || program.forceNewtons.y != 0.0f ||
        program.forceNewtons.z != 0.0f || program.forceNewtons.w != 0.0f;
}
inline bool mrNumiHumanTimedRootForceValid(
    const MRNumiHumanTimedRootForceGPU program,
    const mr_u32 authoritativeStepCount
) {
    const bool finiteForce = metal::all(metal::isfinite(program.forceNewtons));
    if (!finiteForce || program.stepWindow.z != 0u ||
        program.stepWindow.w != 0u || program.forceNewtons.w != 0.0f) {
        return false;
    }
    if (program.stepWindow.y == 0u) {
        return !mrNumiHumanTimedRootForceConfigured(program);
    }
    return program.stepWindow.x < authoritativeStepCount &&
        program.stepWindow.y <= authoritativeStepCount - program.stepWindow.x;
}
inline bool mrNumiHumanTimedRootForceActive(
    const MRNumiHumanTimedRootForceGPU program,
    const mr_u32 authoritativeStepIndex
) {
    return authoritativeStepIndex >= program.stepWindow.x &&
        authoritativeStepIndex - program.stepWindow.x < program.stepWindow.y;
}
enum MRNumiHumanStandSupportContactPlaneMode {
    MR_NUMI_HUMAN_STAND_SUPPORT_PLANE_GLOBAL = 0u,
    MR_NUMI_HUMAN_STAND_SUPPORT_PLANE_PER_CONTACT = 1u,
};
enum MRNumiHumanStandStatusCode {
    MR_NUMI_HUMAN_STAND_SUCCESS = 0u,
    MR_NUMI_HUMAN_STAND_INVALID_DISPATCH = 1u,
    MR_NUMI_HUMAN_STAND_INVALID_MODEL = 2u,
    MR_NUMI_HUMAN_STAND_NONFINITE_INPUT = 3u,
    MR_NUMI_HUMAN_STAND_FACTORIZATION_FAILED = 4u,
    MR_NUMI_HUMAN_STAND_CONTACT_FAILED = 5u,
    MR_NUMI_HUMAN_STAND_NONFINITE_RESULT = 6u,
    MR_NUMI_HUMAN_STAND_TENDON_TRANSFER_FAILED = 7u,
    MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED = 8u,
    MR_NUMI_HUMAN_STAND_EXTERNAL_PHYSICS_FAILED = 9u,
};
enum MRNumiHumanStandFlags {
    MR_NUMI_HUMAN_STAND_ENABLE_CONTACT = 1u << 0u,
    MR_NUMI_HUMAN_STAND_ENABLE_ROOT_ASSISTANCE = 1u << 1u,
    MR_NUMI_HUMAN_STAND_HAS_TENDON_LOADS = 1u << 2u,
    MR_NUMI_HUMAN_STAND_HAS_JOINT_EQUALITIES = 1u << 3u,
    MR_NUMI_HUMAN_STAND_PREDICT_VELOCITY_ONLY = 1u << 4u,
    MR_NUMI_HUMAN_STAND_HAS_PASSIVE_JOINT_PROGRAM = 1u << 5u,
    MR_NUMI_HUMAN_STAND_EXPORT_SOURCE_LIMIT_IMPULSES = 1u << 6u,
    MR_NUMI_HUMAN_STAND_PREPARE_ONLY = 1u << 7u,
    MR_NUMI_HUMAN_STAND_MASS_PREREQUISITES_ONLY = 1u << 8u,
    MR_NUMI_HUMAN_STAND_MASS_READY = 1u << 9u,
    MR_NUMI_HUMAN_STAND_FACTOR_ONLY = 1u << 10u,
    MR_NUMI_HUMAN_STAND_RESPONSES_READY = 1u << 11u,
    MR_NUMI_HUMAN_STAND_FREE_ONLY = 1u << 12u,
    MR_NUMI_HUMAN_STAND_FREE_READY = 1u << 13u,
    MR_NUMI_HUMAN_STAND_CPU_FINISH_READY = 1u << 14u,
    MR_NUMI_HUMAN_STAND_PROJECTED_RAW_READY = 1u << 15u,
    MR_NUMI_HUMAN_STAND_REDUCED_PROJECTED_RESPONSES = 1u << 16u,
    MR_NUMI_HUMAN_STAND_REDUCED_SOURCE_UPPER_TRIANGLE = 1u << 17u,
    MR_NUMI_HUMAN_STAND_REDUCED_PROJECTION_READY = 1u << 18u,
    MR_NUMI_HUMAN_STAND_REDUCED_PROJECTION_USED = 1u << 19u,
    MR_NUMI_HUMAN_STAND_ANALYTIC_BODY_SPATIAL_JACOBIANS = 1u << 20u,
    MR_NUMI_HUMAN_STAND_COMPENSATED_BODY_SUM = 1u << 21u,
    MR_NUMI_HUMAN_STAND_DEFERRED_EQUALITY_DIAGNOSTICS_USED = 1u << 22u,
    MR_NUMI_HUMAN_STAND_HYBRID_FACTOR_CACHE_USED = 1u << 23u,
    MR_NUMI_HUMAN_STAND_CONTACT_WARMSTART_HIT = 1u << 24u,
    MR_NUMI_HUMAN_STAND_CONTACT_WARMSTART_USED = 1u << 25u,
    MR_NUMI_HUMAN_STAND_PGS_EARLY_EXIT_USED = 1u << 26u,
    MR_NUMI_HUMAN_STAND_REDUCED_BASE_PROJECTION_USED = 1u << 27u,
    MR_NUMI_HUMAN_STAND_REDUCED_EQUALITY_DATA_DEFERRED_USED = 1u << 28u,
    MR_NUMI_HUMAN_STAND_DEFERRED_EQUALITY_DATA = 1u << 29u,
    MR_NUMI_HUMAN_STAND_REBUILD_DEFERRED_EQUALITY_DATA = 1u << 30u,
    MR_NUMI_HUMAN_STAND_DEFERRED_EQUALITY_FALLBACK_USED = 1u << 30u,
    MR_NUMI_HUMAN_STAND_DEFERRED_RESPONSE_FALLBACK_ONLY = 1u << 31u,
};
typedef struct alignas(16) MRNumiHumanStandContactGPU {
    mr_u32 bodyIndex;
    mr_u32 pointQueryIndex;
    mr_u32 sourceGeometryIndex;
    mr_u32 reserved0;
    mr_float4 frictionSlopAndStabilization;
    mr_float4 planePoint;
    mr_float4 planeNormal;
} MRNumiHumanStandContactGPU;
typedef struct alignas(16) MRNumiHumanStandContactWarmStartGPU {
    mr_uint4 identity;
    mr_uint4 metadata;
    mr_float4 impulses[32u];
} MRNumiHumanStandContactWarmStartGPU;
typedef struct alignas(16) MRNumiHumanStandContactWarmStartSlotGPU {
    MRNumiHumanStandContactWarmStartGPU accepted;
    mr_uint4 requestedIdentity;
    mr_uint4 requestState;
} MRNumiHumanStandContactWarmStartSlotGPU;
typedef struct alignas(16) MRNumiHumanHipCapsuleTermGPU {
    mr_u32 dofIndex0;
    mr_u32 dofIndex1;
    mr_u32 reserved0;
    mr_u32 reserved1;
    float coordinate0;
    float coordinate1;
    float threshold;
    float toeQuadratic;
    float toeCubic;
    float reservedFloat0;
    float reservedFloat1;
    float reservedFloat2;
} MRNumiHumanHipCapsuleTermGPU;
typedef struct alignas(16) MRNumiHumanStandDispatchGPU {
    mr_u32 abiVersion;
    mr_u32 environmentCount;
    mr_u32 articulationIndex;
    mr_u32 stepIndex;
    mr_u32 stepCount;
    mr_u32 bodyJacobianPointOffset;
    mr_u32 supportContactCount;
    mr_u32 flags;
    mr_u32 qStride;
    mr_u32 vStride;
    mr_u32 pointWorldStride;
    mr_u32 pointJacobianStride;
    mr_u32 bodyPoseStride;
    mr_u32 generalizedForceStride;
    mr_u32 generalizedForceOffset;
    mr_u32 contactIterationCount;
    mr_u32 tendonEndpointCount;
    mr_u32 tendonEnvelopeCount;
    mr_u32 tendonTransferStride;
    mr_u32 jointEqualityCount;
    mr_float4 groundPointAndTimestep;
    mr_float4 groundNormal;
    mr_float4 targetRootPosition;
    mr_float4 targetRootOrientation;
    mr_float4 assistanceGains;
    MRNumiHumanTimedRootForceGPU timedRootForce;
    mr_u32 hipCapsuleTermCount;
    mr_u32 hipCapsuleReserved[3];
    MRNumiHumanHipCapsuleTermGPU hipCapsuleTerms[
        6u];
    mr_u32 supportContactPlaneMode;
    mr_u32 supportContactPlaneCount;
    mr_u32 supportContactPlaneReserved[2];
    mr_u32 rootMomentumDiagnosticMode;
    mr_u32 rootMomentumDiagnosticReserved[3];
} MRNumiHumanStandDispatchGPU;
typedef struct alignas(16) MRNumiHumanStandStatusGPU {
    mr_u32 code;
    mr_u32 environment;
    mr_u32 completedSteps;
    mr_u32 failingIndex;
    mr_u32 activeContactCount;
    mr_u32 maximumActiveContactCount;
    mr_u32 contactIterations;
    mr_u32 flags;
    mr_float4 contactAndAcceleration;
    mr_float4 factorAndAssistance;
    mr_u32 tendonTransferCount;
    mr_u32 tendonEnvelopeTransferCount;
    mr_u32 tendonPointTransferCount;
    mr_u32 tendonFailureCount;
    mr_float4 tendonDiagnostics;
    mr_uint4 jointEqualityCounts;
    mr_float4 jointEqualityDiagnostics;
    mr_float4 jointEqualityProjectionDiagnostics;
    mr_float4 preProjectionPreStepConstraintDiagnostics;
    mr_float4 postProjectionPreStepConstraintDiagnostics;
    mr_float4 constraintImpulseDiagnostics;
    mr_uint4 constraintImpulseOwners;
    mr_float4 constraintImpulseWorkDiagnostics;
    mr_float4 constraintImpulseAbsoluteWorkDiagnostics;
    mr_float4 velocityDiagnostics;
    mr_uint4 velocityDiagnosticOwners;
} MRNumiHumanStandStatusGPU;
enum MRVisualRepresentation : mr_u32 {
    MR_VISUAL_REPRESENTATION_NONE = 0u,
    MR_VISUAL_REPRESENTATION_TRIANGLE_MESH = 1u,
    MR_VISUAL_REPRESENTATION_GAUSSIAN_FIELD = 2u,
    MR_VISUAL_REPRESENTATION_PROCEDURAL = 3u,
};
enum MRVisualBindingKind : mr_u32 {
    MR_VISUAL_BINDING_WORLD = 0u,
    MR_VISUAL_BINDING_ASSET = 1u,
    MR_VISUAL_BINDING_RIGID_BODY = 2u,
    MR_VISUAL_BINDING_ARTICULATED_LINK = 3u,
};
enum MRVisualRendererProfileKind : mr_u32 {
    MR_VISUAL_RENDERER_SENSOR_FAST = 0u,
    MR_VISUAL_RENDERER_SENSOR_REFERENCE = 1u,
};
enum MRVisualShutterModel : mr_u32 {
    MR_VISUAL_SHUTTER_GLOBAL = 0u,
    MR_VISUAL_SHUTTER_ROLLING = 1u,
};
enum MRVisualShutterDirection : mr_u32 {
    MR_VISUAL_SHUTTER_TOP_TO_BOTTOM = 0u,
    MR_VISUAL_SHUTTER_BOTTOM_TO_TOP = 1u,
    MR_VISUAL_SHUTTER_LEFT_TO_RIGHT = 2u,
    MR_VISUAL_SHUTTER_RIGHT_TO_LEFT = 3u,
};
enum MRVisualAlphaMode : mr_u32 {
    MR_VISUAL_ALPHA_OPAQUE = 0u,
    MR_VISUAL_ALPHA_MASK = 1u,
    MR_VISUAL_ALPHA_BLEND = 2u,
};
enum MRVisualLightKind : mr_u32 {
    MR_VISUAL_LIGHT_DIRECTIONAL = 0u,
    MR_VISUAL_LIGHT_POINT = 1u,
    MR_VISUAL_LIGHT_SPOT = 2u,
    MR_VISUAL_LIGHT_RECTANGLE = 3u,
};
enum MRVisualLightUnit : mr_u32 {
    MR_VISUAL_LIGHT_UNIT_LUX = 0u,
    MR_VISUAL_LIGHT_UNIT_CANDELA = 1u,
    MR_VISUAL_LIGHT_UNIT_LUMEN = 2u,
    MR_VISUAL_LIGHT_UNIT_NIT = 3u,
};
enum MRVisualTextureFlag : mr_u32 {
    MR_VISUAL_TEXTURE_SRGB = 1u << 0u,
    MR_VISUAL_TEXTURE_CLAMP_U = 1u << 1u,
    MR_VISUAL_TEXTURE_CLAMP_V = 1u << 2u,
};
enum MRVisualInstanceFlag : mr_u32 {
    MR_VISUAL_INSTANCE_CASTS_SHADOW = 1u << 0u,
    MR_VISUAL_INSTANCE_RECEIVES_SHADOW = 1u << 1u,
    MR_VISUAL_INSTANCE_GAUSSIAN_RECEIVER_PROXY = 1u << 2u,
    MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR = 1u << 3u,
};
enum MRVisualMaterialFlag : mr_u32 {
    MR_VISUAL_MATERIAL_DOUBLE_SIDED = 1u << 0u,
    MR_VISUAL_MATERIAL_UNLIT = 1u << 1u,
};
enum MRVisualFrameSource : mr_u32 {
    MR_VISUAL_SOURCE_SIMULATION = 0u,
    MR_VISUAL_SOURCE_CAPTURE = 1u,
    MR_VISUAL_SOURCE_REPLAY = 2u,
};
enum MRVisualCoordinateFrame : mr_u32 {
    MR_VISUAL_FRAME_PIXEL = 0u,
    MR_VISUAL_FRAME_CAMERA = 1u,
    MR_VISUAL_FRAME_ROBOT_BASE = 2u,
    MR_VISUAL_FRAME_WORLD = 3u,
    MR_VISUAL_FRAME_OBJECT = 4u,
};
enum MRVisualStorageKind : mr_u32 {
    MR_VISUAL_STORAGE_HOST = 0u,
    MR_VISUAL_STORAGE_METAL_BUFFER = 1u,
    MR_VISUAL_STORAGE_MLX_ARRAY = 2u,
    MR_VISUAL_STORAGE_COREML_TENSOR = 3u,
};
enum MRVisualPixelFormat : mr_u32 {
    MR_VISUAL_FORMAT_UNKNOWN = 0u,
    MR_VISUAL_FORMAT_RGBA32_FLOAT = 1u,
    MR_VISUAL_FORMAT_R32_FLOAT = 2u,
    MR_VISUAL_FORMAT_RGBA32_UINT = 3u,
    MR_VISUAL_FORMAT_R32_UINT = 4u,
    MR_VISUAL_FORMAT_R8_UINT = 5u,
    MR_VISUAL_FORMAT_RG32_FLOAT = 6u,
};
enum MRVisualModality : mr_u32 {
    MR_VISUAL_MODALITY_RGB = 1u << 0u,
    MR_VISUAL_MODALITY_DEPTH = 1u << 1u,
    MR_VISUAL_MODALITY_DEPTH_VALIDITY = 1u << 2u,
    MR_VISUAL_MODALITY_NORMAL = 1u << 3u,
    MR_VISUAL_MODALITY_MOTION = 1u << 4u,
    MR_VISUAL_MODALITY_SEMANTIC = 1u << 5u,
    MR_VISUAL_MODALITY_INSTANCE = 1u << 6u,
    MR_VISUAL_MODALITY_LINK = 1u << 7u,
    MR_VISUAL_MODALITY_KEYPOINT = 1u << 8u,
    MR_VISUAL_MODALITY_FEATURE = 1u << 9u,
    MR_VISUAL_MODALITY_OBJECT_POSE = 1u << 10u,
};
enum MRVisualValidityFlag : mr_u32 {
    MR_VISUAL_VALIDITY_FRAME = 1u << 0u,
    MR_VISUAL_VALIDITY_DEPTH = 1u << 1u,
    MR_VISUAL_VALIDITY_GEOMETRY = 1u << 2u,
};
enum MRPerceptionCapability : mr_u32 {
    MR_PERCEPTION_CAP_DENSE_DEPTH = 1u << 0u,
    MR_PERCEPTION_CAP_SEMANTIC = 1u << 1u,
    MR_PERCEPTION_CAP_INSTANCE = 1u << 2u,
    MR_PERCEPTION_CAP_TRACKING = 1u << 3u,
    MR_PERCEPTION_CAP_OBJECT_POSE = 1u << 4u,
    MR_PERCEPTION_CAP_KEYPOINT = 1u << 5u,
    MR_PERCEPTION_CAP_DENSE_FEATURE = 1u << 6u,
    MR_PERCEPTION_CAP_EMBEDDING = 1u << 7u,
};
typedef struct alignas(16) MRVisualSensorBindingGPU {
    mr_uint4 identity;
    mr_float4 timing;
    mr_float4 rangeAndResponse;
    mr_uint4 shutter;
} MRVisualSensorBindingGPU;
typedef struct alignas(16) MRVisualVertexGPUV2 {
    mr_float4 position;
    mr_float4 normalAndTangentSign;
    mr_float4 tangent;
    mr_float4 texcoord01;
    mr_float4 color;
} MRVisualVertexGPUV2;
typedef struct alignas(16) MRVisualPrimitiveGPUV2 {
    mr_uint4 geometry;
    mr_uint4 identity;
    mr_float4 boundsMinimum;
    mr_float4 boundsMaximum;
} MRVisualPrimitiveGPUV2;
typedef struct alignas(16) MRVisualTriangleGPUV2 {
    mr_uint4 verticesAndPrimitive;
} MRVisualTriangleGPUV2;
typedef struct alignas(16) MRVisualInstanceGPUV2 {
    mr_float4 translationAndScale;
    mr_float4 orientation;
    mr_uint4 binding;
    mr_uint4 identity;
    mr_uint4 geometry;
} MRVisualInstanceGPUV2;
typedef struct alignas(16) MRVisualMaterialGPUV2 {
    mr_float4 baseColorAndOpacity;
    mr_float4 emissionAndStrength;
    mr_float4 surface;
    mr_float4 coatingAndAlphaCutoff;
    mr_uint4 textureIndices0;
    mr_uint4 textureIndices1;
    mr_uint4 flags;
    mr_uint4 reserved;
} MRVisualMaterialGPUV2;
typedef struct alignas(16) MRVisualTextureBindingGPUV2 {
    mr_uint4 resource;
    mr_float4 uvTransform0;
    mr_float4 uvTransform1;
} MRVisualTextureBindingGPUV2;
typedef struct alignas(16) MRVisualEnvironmentGPUV2 {
    mr_uint4 dimensions;
    mr_float4 parameters;
} MRVisualEnvironmentGPUV2;
typedef struct alignas(16) MRVisualLightGPUV1 {
    mr_float4 positionAndRange;
    mr_float4 directionAndSpot;
    mr_float4 colorAndIntensity;
    mr_float4 shape;
    mr_uint4 identity;
    mr_uint4 shadow;
} MRVisualLightGPUV1;
typedef struct alignas(16) MRVisualShutterProfileGPUV1 {
    mr_uint4 mode;
    mr_float4 timing;
} MRVisualShutterProfileGPUV1;
typedef struct alignas(16) MRVisualFrameMetadataGPU {
    mr_uint4 dimensions;
    mr_uint4 identity;
    mr_float4 timing;
    mr_uint4 contract;
} MRVisualFrameMetadataGPU;
typedef struct alignas(16) MRVisualKeypointGPU {
    mr_float4 positionAndVisibility;
    mr_uint4 identity;
} MRVisualKeypointGPU;
typedef struct alignas(16) MRVisualPoseGPU {
    mr_float4 position;
    mr_float4 orientation;
    mr_uint4 identity;
} MRVisualPoseGPU;
typedef struct alignas(16) MRVisualContactAnnotationGPU {
    mr_float4 positionAndImpulse;
    mr_float4 normalAndSeparation;
    mr_uint4 identity;
} MRVisualContactAnnotationGPU;
typedef struct alignas(16) MRHumanRestingVertexMap {
    mr_u32 firstInfluence, influenceCount, deformationKind, chamberIndex;
    mr_float4 deformationWeight;
    mr_float4 respiratoryBasis;
} MRHumanRestingVertexMap;
typedef struct alignas(16) MRHumanRestingInfluence {
    mr_float4 positionAndWeight;
    mr_float4 normal;
    mr_uint4 body;
} MRHumanRestingInfluence;
typedef struct alignas(16) MRHumanRestingCardiacWallVertexGPU {
    mr_float4 first;
    mr_float4 second;
    mr_float4 closure;
} MRHumanRestingCardiacWallVertexGPU;
typedef struct alignas(16) MRHumanRestingCardiacWallGPU {
    mr_float4 volumePolynomial[5];
    mr_float4 scalesAndVolume;
    mr_float4 closureBoundsAndTolerance;
    mr_uint4 chambersAndFlags;
} MRHumanRestingCardiacWallGPU;
typedef struct alignas(16) MRHumanRestingCommonFieldVertexGPU { mr_float4 displacement[7u]; } MRHumanRestingCommonFieldVertexGPU;
typedef struct alignas(16) MRHumanRestingCommonCoordinateBoxGPU { mr_float4 lower[2]; mr_float4 upper[2]; } MRHumanRestingCommonCoordinateBoxGPU;
typedef struct alignas(16) MRHumanRestingCommonFieldGPU {
    mr_float4 volumePolynomial[7u][30];
    mr_float4 sourceReferenceVolumes[2];
    mr_float4 materialTargetVolumes;
    mr_float4 trialLower[2];
    mr_float4 trialUpper[2];
    mr_float4 solver;
    mr_uint4 countsAndFlags;
} MRHumanRestingCommonFieldGPU;
typedef struct alignas(16) MRHumanRestingCommonCoordinatesGPU {
    mr_float4 first; mr_float4 second; mr_uint4 status; mr_float4 diagnostics;
} MRHumanRestingCommonCoordinatesGPU;
typedef struct alignas(16) MRHumanRestingCommonFailureGPU {
    mr_uint4 identity;
    mr_uint4 matterStatus;
    mr_float4 matterDiagnostics;
    mr_uint4 standStatus;
    mr_uint4 respirationCandidateStatus;
    mr_uint4 respirationAcceptedStatus;
    mr_float4 chamberVolumes;
    mr_float4 respirationControl;
    mr_float4 commonFirst;
    mr_float4 commonSecond;
    mr_uint4 commonStatus;
    mr_float4 commonDiagnostics;
    mr_uint4 brainInputMetadata;
    mr_uint4 brainInputRoots;
    mr_uint4 brainInputTimestampHighs;
    mr_uint4 brainAcceptedMetadata;
    mr_uint4 brainAcceptedRoots;
    mr_uint4 brainCandidateMetadata;
    mr_uint4 brainCandidateRoot;
    mr_uint4 brainOutputMetadata;
    mr_uint4 brainOutputRoots;
    mr_float4 brainOutputSetpoints;
    mr_float4 brainOutputExcitations;
    mr_float4 excitationBuffer;
} MRHumanRestingCommonFailureGPU;
static_assert(sizeof(MRHumanRestingCommonFieldVertexGPU)==112);
static_assert(sizeof(MRHumanRestingCommonCoordinateBoxGPU)==64);
static_assert(sizeof(MRHumanRestingCommonFieldGPU)==3504);
static_assert(sizeof(MRHumanRestingCommonCoordinatesGPU)==64);
static_assert(sizeof(MRHumanRestingCommonFailureGPU)==384);
typedef struct alignas(16) MRHumanRestingAnatomyGPU {
    mr_uint4 bodyAndFlags;
    mr_float4 lungAnchorAndVolume;
    mr_float4 superiorAxisAndHeight;
    mr_float4 anteriorAxis;
    mr_float4 muscleAreas;
    mr_float4 diaphragmHeight;
    mr_float4 lungBasalBlend;
    mr_float4 ribPivotAndGain[24];
    mr_float4 ribAxis[24];
    mr_float4 chamberCenterAndVolume[4];
    mr_float4 chamberVolumePolynomial[4];
} MRHumanRestingAnatomyGPU;
typedef struct alignas(16) MRHumanRestingSurfaceAuditGPU {
    mr_uint4 indicesAndOwner;
    mr_float4 reference;
} MRHumanRestingSurfaceAuditGPU;
typedef struct alignas(16) MRHumanRestingVolumeAuditGroupGPU {
    mr_uint4 surfaceAndTriangleRange;
} MRHumanRestingVolumeAuditGroupGPU;
typedef struct alignas(16) MRHumanRestingVolumeAuditRangeGPU {
    mr_uint4 groupRange;
} MRHumanRestingVolumeAuditRangeGPU;
typedef struct alignas(16) MRHumanRestingVolumeAuditPartialGPU {
    float signedVolume;
    mr_u32 invalidTriangleCount;
    mr_u32 firstInvalidTriangle;
    mr_u32 reserved;
} MRHumanRestingVolumeAuditPartialGPU;
typedef struct alignas(16) MRHumanRestingSurfaceFailureGPU {
    mr_uint4 surfaceTriangleKind;
    mr_uint4 vertexIndices;
    mr_float4 renderedPositions[3];
} MRHumanRestingSurfaceFailureGPU;
static_assert(sizeof(MRHumanRestingVolumeAuditGroupGPU)==16);
static_assert(sizeof(MRHumanRestingVolumeAuditRangeGPU)==16);
static_assert(sizeof(MRHumanRestingVolumeAuditPartialGPU)==16);
constant uchar nmHumanRestingCommonExponents[120][7] = {
    {0,0,0,0,0,0,0},
    {0,0,0,0,0,0,1},
    {0,0,0,0,0,0,2},
    {0,0,0,0,0,0,3},
    {0,0,0,0,0,1,0},
    {0,0,0,0,0,1,1},
    {0,0,0,0,0,1,2},
    {0,0,0,0,0,2,0},
    {0,0,0,0,0,2,1},
    {0,0,0,0,0,3,0},
    {0,0,0,0,1,0,0},
    {0,0,0,0,1,0,1},
    {0,0,0,0,1,0,2},
    {0,0,0,0,1,1,0},
    {0,0,0,0,1,1,1},
    {0,0,0,0,1,2,0},
    {0,0,0,0,2,0,0},
    {0,0,0,0,2,0,1},
    {0,0,0,0,2,1,0},
    {0,0,0,0,3,0,0},
    {0,0,0,1,0,0,0},
    {0,0,0,1,0,0,1},
    {0,0,0,1,0,0,2},
    {0,0,0,1,0,1,0},
    {0,0,0,1,0,1,1},
    {0,0,0,1,0,2,0},
    {0,0,0,1,1,0,0},
    {0,0,0,1,1,0,1},
    {0,0,0,1,1,1,0},
    {0,0,0,1,2,0,0},
    {0,0,0,2,0,0,0},
    {0,0,0,2,0,0,1},
    {0,0,0,2,0,1,0},
    {0,0,0,2,1,0,0},
    {0,0,0,3,0,0,0},
    {0,0,1,0,0,0,0},
    {0,0,1,0,0,0,1},
    {0,0,1,0,0,0,2},
    {0,0,1,0,0,1,0},
    {0,0,1,0,0,1,1},
    {0,0,1,0,0,2,0},
    {0,0,1,0,1,0,0},
    {0,0,1,0,1,0,1},
    {0,0,1,0,1,1,0},
    {0,0,1,0,2,0,0},
    {0,0,1,1,0,0,0},
    {0,0,1,1,0,0,1},
    {0,0,1,1,0,1,0},
    {0,0,1,1,1,0,0},
    {0,0,1,2,0,0,0},
    {0,0,2,0,0,0,0},
    {0,0,2,0,0,0,1},
    {0,0,2,0,0,1,0},
    {0,0,2,0,1,0,0},
    {0,0,2,1,0,0,0},
    {0,0,3,0,0,0,0},
    {0,1,0,0,0,0,0},
    {0,1,0,0,0,0,1},
    {0,1,0,0,0,0,2},
    {0,1,0,0,0,1,0},
    {0,1,0,0,0,1,1},
    {0,1,0,0,0,2,0},
    {0,1,0,0,1,0,0},
    {0,1,0,0,1,0,1},
    {0,1,0,0,1,1,0},
    {0,1,0,0,2,0,0},
    {0,1,0,1,0,0,0},
    {0,1,0,1,0,0,1},
    {0,1,0,1,0,1,0},
    {0,1,0,1,1,0,0},
    {0,1,0,2,0,0,0},
    {0,1,1,0,0,0,0},
    {0,1,1,0,0,0,1},
    {0,1,1,0,0,1,0},
    {0,1,1,0,1,0,0},
    {0,1,1,1,0,0,0},
    {0,1,2,0,0,0,0},
    {0,2,0,0,0,0,0},
    {0,2,0,0,0,0,1},
    {0,2,0,0,0,1,0},
    {0,2,0,0,1,0,0},
    {0,2,0,1,0,0,0},
    {0,2,1,0,0,0,0},
    {0,3,0,0,0,0,0},
    {1,0,0,0,0,0,0},
    {1,0,0,0,0,0,1},
    {1,0,0,0,0,0,2},
    {1,0,0,0,0,1,0},
    {1,0,0,0,0,1,1},
    {1,0,0,0,0,2,0},
    {1,0,0,0,1,0,0},
    {1,0,0,0,1,0,1},
    {1,0,0,0,1,1,0},
    {1,0,0,0,2,0,0},
    {1,0,0,1,0,0,0},
    {1,0,0,1,0,0,1},
    {1,0,0,1,0,1,0},
    {1,0,0,1,1,0,0},
    {1,0,0,2,0,0,0},
    {1,0,1,0,0,0,0},
    {1,0,1,0,0,0,1},
    {1,0,1,0,0,1,0},
    {1,0,1,0,1,0,0},
    {1,0,1,1,0,0,0},
    {1,0,2,0,0,0,0},
    {1,1,0,0,0,0,0},
    {1,1,0,0,0,0,1},
    {1,1,0,0,0,1,0},
    {1,1,0,0,1,0,0},
    {1,1,0,1,0,0,0},
    {1,1,1,0,0,0,0},
    {1,2,0,0,0,0,0},
    {2,0,0,0,0,0,0},
    {2,0,0,0,0,0,1},
    {2,0,0,0,0,1,0},
    {2,0,0,0,1,0,0},
    {2,0,0,1,0,0,0},
    {2,0,1,0,0,0,0},
    {2,1,0,0,0,0,0},
    {3,0,0,0,0,0,0}
};
inline float nmHumanRestingCommonCoefficient(constant MRHumanRestingCommonFieldGPU& p,uint row,uint term) {
    return p.volumePolynomial[row][term>>2][term&3u];
}
inline float nmHumanRestingCommonPower(float x,uint exponent) {
    float r=1.0f; for(uint i=0;i<exponent;++i) r*=x; return r;
}
inline float nmHumanRestingCommonMonomial(thread const float* x,constant const uchar* e) {
    float r=1.0f; for(uint i=0;i<7;++i) r*=nmHumanRestingCommonPower(x[i],e[i]); return r;
}
inline void nmHumanRestingCommonPowers(thread const float* x,thread float* powers) {
    for(uint k=0;k<7;++k) {
        powers[k*4]=1.0f;
        for(uint exponent=1;exponent<4;++exponent)
            powers[k*4+exponent]=powers[k*4+exponent-1]*x[k];
    }
}
inline bool nmHumanRestingCommonEvaluateValues(constant MRHumanRestingCommonFieldGPU& p,
    thread const float* powers,thread float* values) {
#pragma clang fp contract(off)
    for(uint r=0;r<7;++r) values[r]=0.0f;
    for(uint term=0;term<120;++term) {
        constant const uchar* e=nmHumanRestingCommonExponents[term];
        float monomial=1.0f;
        for(uint k=0;k<7;++k) monomial*=powers[k*4+e[k]];
        for(uint r=0;r<7;++r)
            values[r]+=nmHumanRestingCommonCoefficient(p,r,term)*monomial;
    }
    for(uint r=0;r<7;++r) if(!isfinite(values[r])) return false;
    return true;
}
inline bool nmHumanRestingCommonEvaluateJacobian(constant MRHumanRestingCommonFieldGPU& p,
    thread const float* powers,thread float* jacobian) {
#pragma clang fp contract(off)
    for(uint r=0;r<7;++r) for(uint c=0;c<7;++c) jacobian[r*7+c]=0.0f;
    for(uint term=0;term<120;++term) {
        constant const uchar* e=nmHumanRestingCommonExponents[term];
        float derivatives[7];
        for(uint c=0;c<7;++c) {
            const uint power=e[c];
            if(!power) { derivatives[c]=0.0f; continue; }
            float derivative=float(power);
            for(uint k=0;k<7;++k)
                derivative*=powers[k*4+e[k]-(k==c?1u:0u)];
            derivatives[c]=derivative;
        }
        for(uint r=0;r<7;++r) {
            const float coefficient=nmHumanRestingCommonCoefficient(p,r,term);
            for(uint c=0;c<7;++c) if(e[c])
                jacobian[r*7+c]+=coefficient*derivatives[c];
        }
    }
    for(uint r=0;r<7;++r)
        for(uint c=0;c<7;++c) if(!isfinite(jacobian[r*7+c])) return false;
    return true;
}
inline bool nmHumanRestingCommonEvaluate(constant MRHumanRestingCommonFieldGPU& p,
    thread const float* x,thread float* values,thread float* jacobian) {
    float powers[28];
    nmHumanRestingCommonPowers(x,powers);
    return nmHumanRestingCommonEvaluateValues(p,powers,values)&&
        nmHumanRestingCommonEvaluateJacobian(p,powers,jacobian);
}
inline float nmHumanRestingCommonResidual(thread const float* value,thread const float* target) {
#pragma clang fp contract(off)
    float result=0.0f; for(uint i=0;i<7;++i) result=max(result,abs(value[i]-target[i])); return result;
}
inline bool nmHumanRestingCommonInsideTrial(constant MRHumanRestingCommonFieldGPU& p,thread const float* x) {
    for(uint i=0;i<7;++i) {
        const float low=i<4?p.trialLower[0][i]:p.trialLower[1][i-4];
        const float high=i<4?p.trialUpper[0][i]:p.trialUpper[1][i-4];
        if(!isfinite(x[i])||x[i]<low||x[i]>high) return false;
    } return true;
}
inline bool nmHumanRestingCommonInCertifiedUnion(device const MRHumanRestingCommonCoordinateBoxGPU* boxes,
    uint count,thread const float* x,thread uint& matched) {
    for(uint b=0;b<count;++b) {
        bool inside=true;
        for(uint i=0;i<7;++i) {
            const float low=i<4?boxes[b].lower[0][i]:boxes[b].lower[1][i-4];
            const float high=i<4?boxes[b].upper[0][i]:boxes[b].upper[1][i-4];
            inside=inside&&x[i]>=low&&x[i]<=high;
        } if(inside){matched=b;return true;}
    } return false;
}
inline uint nmHumanRestingCommonSolve(constant MRHumanRestingCommonFieldGPU& p,
    device const MRHumanRestingCommonCoordinateBoxGPU* boxes,thread const float* targetVolumes,
    thread float* x,thread uint& iterations,thread uint& matched,thread float& residualOut) {
#pragma clang fp contract(off)
    if(p.countsAndFlags.y!=1u||p.countsAndFlags.x==0u||!isfinite(p.solver.x)||p.solver.x<=0.0f) return 1u;
    float target[7],values[7],jacobian[49],augmented[56],delta[7],trial[7],trialValues[7],trialPowers[28];
    for(uint i=0;i<7;++i) {
        const float reference=i<4?p.sourceReferenceVolumes[0][i]:p.sourceReferenceVolumes[1][i-4];
        if(!isfinite(targetVolumes[i])||!isfinite(reference)||reference<=0.0f) return 1u;
        target[i]=targetVolumes[i]/reference; x[i]=0.0f;
    }
    if(!nmHumanRestingCommonInsideTrial(p,x)) return 1u;
    const uint maxIterations=12u,maxBacktracks=12u; const float tolerance=p.solver.x;
    for(uint iteration=0;iteration<maxIterations;++iteration) {
        iterations=iteration;
        if(!nmHumanRestingCommonEvaluate(p,x,values,jacobian)) return 1u;
        const float residual=nmHumanRestingCommonResidual(values,target); residualOut=residual;
        if(residual<=tolerance)
            return nmHumanRestingCommonInCertifiedUnion(boxes,p.countsAndFlags.x,x,matched)?0u:5u;
        for(uint r=0;r<7;++r) {
            for(uint c=0;c<7;++c) augmented[r*8+c]=jacobian[r*7+c];
            augmented[r*8+7]=values[r]-target[r];
        }
        for(uint pivot=0;pivot<7;++pivot) {
            uint best=pivot; float magnitude=abs(augmented[pivot*8+pivot]);
            for(uint r=pivot+1;r<7;++r) { const float candidate=abs(augmented[r*8+pivot]);
                if(candidate>magnitude){magnitude=candidate;best=r;} }
            if(!isfinite(magnitude)||magnitude<=1.e-12f) return 2u;
            if(best!=pivot) for(uint c=pivot;c<8;++c) {
                const float swap=augmented[pivot*8+c]; augmented[pivot*8+c]=augmented[best*8+c]; augmented[best*8+c]=swap;
            }
            const float diagonal=augmented[pivot*8+pivot];
            for(uint c=pivot;c<8;++c) augmented[pivot*8+c]/=diagonal;
            for(uint r=0;r<7;++r) if(r!=pivot) { const float factor=augmented[r*8+pivot];
                for(uint c=pivot;c<8;++c) augmented[r*8+c]-=factor*augmented[pivot*8+c]; }
        }
        for(uint i=0;i<7;++i) delta[i]=augmented[i*8+7];
        bool accepted=false; float scale=1.0f;
        for(uint backtrack=0;backtrack<maxBacktracks;++backtrack) {
            for(uint i=0;i<7;++i) trial[i]=x[i]-scale*delta[i];
            if(nmHumanRestingCommonInsideTrial(p,trial)) {
                nmHumanRestingCommonPowers(trial,trialPowers);
                if(nmHumanRestingCommonEvaluateValues(p,trialPowers,trialValues)) {
                    const float next=nmHumanRestingCommonResidual(trialValues,target);
                    if(isfinite(next)&&next<residual&&
                       nmHumanRestingCommonEvaluateJacobian(p,trialPowers,jacobian)) {
                        for(uint i=0;i<7;++i) x[i]=trial[i];
                        accepted=true; break;
                    }
                }
            } scale*=.5f;
        } if(!accepted) return 3u;
    }
    if(nmHumanRestingCommonEvaluate(p,x,values,jacobian)) {
        residualOut=nmHumanRestingCommonResidual(values,target);
        if(residualOut<=tolerance)
            return nmHumanRestingCommonInCertifiedUnion(boxes,p.countsAndFlags.x,x,matched)?0u:5u;
    } return 4u;
}
inline bool nmHumanRestingCommonEvaluateValuesSimd(
    constant MRHumanRestingCommonFieldGPU& p,thread const float* powers,uint lane,
    thread float& rowValue) {
#pragma clang fp contract(off)
    rowValue=0.0f;
    if(lane<7u) {
        for(uint term=0;term<120;++term) {
            constant const uchar* e=nmHumanRestingCommonExponents[term];
            float monomial=1.0f;
            for(uint k=0;k<7;++k) monomial*=powers[k*4+e[k]];
            rowValue+=nmHumanRestingCommonCoefficient(p,lane,term)*monomial;
        }
    }
    const bool finiteRow=lane>=7u||isfinite(rowValue);
    return simd_all(finiteRow);
}
inline bool nmHumanRestingCommonEvaluateJacobianSimd(
    constant MRHumanRestingCommonFieldGPU& p,thread const float* powers,uint lane,
    thread float* rowJacobian) {
#pragma clang fp contract(off)
    for(uint c=0;c<7;++c) rowJacobian[c]=0.0f;
    if(lane<7u) {
        for(uint term=0;term<120;++term) {
            constant const uchar* e=nmHumanRestingCommonExponents[term];
            float derivatives[7];
            for(uint c=0;c<7;++c) {
                const uint power=e[c];
                if(!power) { derivatives[c]=0.0f; continue; }
                float derivative=float(power);
                for(uint k=0;k<7;++k)
                    derivative*=powers[k*4+e[k]-(k==c?1u:0u)];
                derivatives[c]=derivative;
            }
            const float coefficient=nmHumanRestingCommonCoefficient(p,lane,term);
            for(uint c=0;c<7;++c) if(e[c])
                rowJacobian[c]+=coefficient*derivatives[c];
        }
    }
    bool finiteRow=true;
    if(lane<7u) for(uint c=0;c<7;++c)
        if(!isfinite(rowJacobian[c])) finiteRow=false;
    return simd_all(finiteRow);
}
inline float nmHumanRestingCommonResidualSimd(float rowValue,thread const float* target) {
#pragma clang fp contract(off)
    float result=0.0f;
    for(uint row=0;row<7;++row) {
        const float value=simd_broadcast(rowValue,row);
        result=max(result,abs(value-target[row]));
    }
    return result;
}
inline uint nmHumanRestingCommonSolveSimd(
    constant MRHumanRestingCommonFieldGPU& p,
    device const MRHumanRestingCommonCoordinateBoxGPU* boxes,
    thread const float* targetVolumes,thread float* x,uint lane,
    thread uint& iterations,thread uint& matched,thread float& residualOut) {
#pragma clang fp contract(off)
    if(p.countsAndFlags.y!=1u||p.countsAndFlags.x==0u||
       !isfinite(p.solver.x)||p.solver.x<=0.0f) return 1u;
    float target[7],powers[28],trialPowers[28],rowValue=0.0f;
    float rowJacobian[7],augmented[8],trial[7],trialValue=0.0f;
    for(uint i=0;i<7;++i) {
        const float reference=i<4?p.sourceReferenceVolumes[0][i]:p.sourceReferenceVolumes[1][i-4];
        if(!isfinite(targetVolumes[i])||!isfinite(reference)||reference<=0.0f) return 1u;
        target[i]=targetVolumes[i]/reference;
        x[i]=0.0f;
    }
    if(!nmHumanRestingCommonInsideTrial(p,x)) return 1u;
    const uint maxIterations=12u,maxBacktracks=12u;
    const float tolerance=p.solver.x;
    for(uint iteration=0;iteration<maxIterations;++iteration) {
        iterations=iteration;
        if(iteration==0u) {
            nmHumanRestingCommonPowers(x,powers);
            if(!nmHumanRestingCommonEvaluateValuesSimd(p,powers,lane,rowValue)||
               !nmHumanRestingCommonEvaluateJacobianSimd(p,powers,lane,rowJacobian)) return 1u;
        }
        const float residual=nmHumanRestingCommonResidualSimd(rowValue,target);
        residualOut=residual;
        if(residual<=tolerance)
            return nmHumanRestingCommonInCertifiedUnion(
                boxes,p.countsAndFlags.x,x,matched)?0u:5u;
        for(uint c=0;c<7;++c) augmented[c]=lane<7u?rowJacobian[c]:0.0f;
        augmented[7]=lane<7u?rowValue-target[lane]:0.0f;
        for(uint pivot=0;pivot<7;++pivot) {
            uint best=pivot;
            float magnitude=abs(simd_broadcast(augmented[pivot],pivot));
            for(uint r=pivot+1;r<7;++r) {
                const float candidate=abs(simd_broadcast(augmented[pivot],r));
                if(candidate>magnitude) { magnitude=candidate; best=r; }
            }
            if(!isfinite(magnitude)||magnitude<=1.e-12f) return 2u;
            if(best!=pivot) for(uint c=pivot;c<8;++c) {
                const float pivotValue=simd_broadcast(augmented[c],pivot);
                const float bestValue=simd_broadcast(augmented[c],best);
                if(lane==pivot) augmented[c]=bestValue;
                else if(lane==best) augmented[c]=pivotValue;
            }
            const float diagonal=simd_broadcast(augmented[pivot],pivot);
            if(lane==pivot) for(uint c=pivot;c<8;++c) augmented[c]/=diagonal;
            const float factor=augmented[pivot];
            for(uint c=pivot;c<8;++c) {
                const float pivotValue=simd_broadcast(augmented[c],pivot);
                if(lane!=pivot) augmented[c]-=factor*pivotValue;
            }
        }
        bool accepted=false;
        float scale=1.0f;
        for(uint backtrack=0;backtrack<maxBacktracks;++backtrack) {
            for(uint i=0;i<7;++i) {
                const float delta=simd_broadcast(augmented[7],i);
                trial[i]=x[i]-scale*delta;
            }
            if(nmHumanRestingCommonInsideTrial(p,trial)) {
                nmHumanRestingCommonPowers(trial,trialPowers);
                if(nmHumanRestingCommonEvaluateValuesSimd(
                    p,trialPowers,lane,trialValue)) {
                    const float next=nmHumanRestingCommonResidualSimd(trialValue,target);
                    if(isfinite(next)&&next<residual&&
                       nmHumanRestingCommonEvaluateJacobianSimd(
                           p,trialPowers,lane,rowJacobian)) {
                        for(uint i=0;i<7;++i) x[i]=trial[i];
                        rowValue=trialValue;
                        accepted=true;
                        break;
                    }
                }
            }
            scale*=.5f;
        }
        if(!accepted) return 3u;
    }
    nmHumanRestingCommonPowers(x,powers);
    if(nmHumanRestingCommonEvaluateValuesSimd(p,powers,lane,rowValue)&&
       nmHumanRestingCommonEvaluateJacobianSimd(p,powers,lane,rowJacobian)) {
        residualOut=nmHumanRestingCommonResidualSimd(rowValue,target);
        if(residualOut<=tolerance)
            return nmHumanRestingCommonInCertifiedUnion(
                boxes,p.countsAndFlags.x,x,matched)?0u:5u;
    }
    return 4u;
}

typedef struct alignas(16) MRHumanRestingBedGPU {
    mr_uint4 counts;
    mr_float4 originSpacing;
} MRHumanRestingBedGPU;

constant bool kRestingContouredBed [[function_constant(47)]];
constant bool kUseRestingContouredBed =
    is_function_constant_defined(kRestingContouredBed) ? kRestingContouredBed : false;
struct NMHumanRestingBedQuery {
    float3 point;
    float3 normal;
    float signedGap;
    uint facet;
    bool valid;
};
inline NMHumanRestingBedQuery nmHumanRestingBedQuery(
    constant MRHumanRestingBedGPU& bed, device const float* heights, float3 p) {
    NMHumanRestingBedQuery result{float3(0), float3(0,0,1), NAN, 0xffffffffu, false};
    if (bed.counts.z != 1u || bed.counts.w != 1u ||
        bed.counts.x < 2u || bed.counts.y < 2u ||
        !all(isfinite(p)) || !all(isfinite(bed.originSpacing)) ||
        bed.originSpacing.z <= 0 || bed.originSpacing.w <= 0) return result;
    const uint nx = bed.counts.x, ny = bed.counts.y;
    const float2 origin=bed.originSpacing.xy, step=bed.originSpacing.zw;
    const float2 end=fma(float2(nx-1u,ny-1u),step,origin);
    if (any(p.xy<origin) || any(p.xy>end)) return result;
    const float2 uv=(p.xy-origin)/step;
    if (!all(isfinite(uv))) return result;
    uint x=min(uint(floor(uv.x)),nx-2u), y=min(uint(floor(uv.y)),ny-2u);
    if(x>0u && p.x<fma(float(x),step.x,origin.x))--x;
    else if(x+2u<nx && p.x>fma(float(x+1u),step.x,origin.x))++x;
    if(y>0u && p.y<fma(float(y),step.y,origin.y))--y;
    else if(y+2u<ny && p.y>fma(float(y+1u),step.y,origin.y))++y;
    const float2 lo=fma(float2(x,y),step,origin);
    const float2 hi=fma(float2(x+1u,y+1u),step,origin);
    const float2 extent=hi-lo, f=(p.xy-lo)/extent;
    if(any(extent<=0.0f) || !all(isfinite(f)))return result;
    const uint base=y*nx+x;
    const float h00=heights[base],h10=heights[base+1u];
    const float h01=heights[base+nx],h11=heights[base+nx+1u];
    if(!all(isfinite(float4(h00,h10,h01,h11))))return result;
    const bool upper=f.x+f.y>1.0f;
    const float dx=(upper?h11-h01:h10-h00)/extent.x;
    const float dy=(upper?h11-h10:h01-h00)/extent.y;
    result.normal=normalize(float3(-dx,-dy,1));
    result.point=upper?float3(hi,h11):float3(lo,h00);
    result.signedGap=dot(p-result.point,result.normal);
    result.facet = 2u*(y*(nx-1u)+x)+(upper?1u:0u);
    result.valid = all(isfinite(result.normal)) && isfinite(result.signedGap);
    return result;
}
constant bool kValidatedSkinInfluences [[function_constant(42)]];
constant bool kUseValidatedSkinInfluences =
    is_function_constant_defined(kValidatedSkinInfluences)
        ? kValidatedSkinInfluences : false;
constant bool kTiledSkinInfluenceLayout [[function_constant(15)]];
constant bool kUseTiledSkinInfluenceLayout =
    is_function_constant_defined(kTiledSkinInfluenceLayout)
        ? kTiledSkinInfluenceLayout : false;
constant bool kSkipNonAncestorSupportBodyDofs [[function_constant(43)]];
constant bool kUseSkipNonAncestorSupportBodyDofs =
    is_function_constant_defined(kSkipNonAncestorSupportBodyDofs)
        ? kSkipNonAncestorSupportBodyDofs : false;
typedef struct alignas(16) MRHumanRestingSupportRegionGPU {
    mr_u32 pointQueryIndex;
    mr_u32 sourceGeometryIndex;
    mr_u32 reserved0;
    mr_u32 reserved1;
} MRHumanRestingSupportRegionGPU;
typedef struct alignas(16) MRHumanRestingSupportVertexRegionGPU {
    mr_u32 regionIndex;
    mr_u32 reserved0;
    mr_u32 reserved1;
    mr_u32 reserved2;
} MRHumanRestingSupportVertexRegionGPU;
typedef struct alignas(16) MRHumanRestingSupportPositionGPU {
    mr_float4 high;
    mr_float4 low;
} MRHumanRestingSupportPositionGPU;
typedef struct alignas(16) MRHumanRestingSupportRotationZBasisGPU {
    mr_float4 high;
    mr_float4 low;
} MRHumanRestingSupportRotationZBasisGPU;
typedef struct alignas(16) MRHumanRestingSupportDispatchGPU {
    mr_uint4 counts;
    mr_uint4 strides;
    mr_uint4 identity;
} MRHumanRestingSupportDispatchGPU;
typedef struct alignas(16) MRHumanRestingSupportDebugGPU {
    mr_uint4 status;
    mr_float4 position;
    mr_float4 jacobian;
} MRHumanRestingSupportDebugGPU;
static_assert(sizeof(MRHumanRestingSupportRegionGPU) == 16);
static_assert(sizeof(MRHumanRestingSupportVertexRegionGPU) == 16);
static_assert(sizeof(MRHumanRestingSupportPositionGPU) == 32);
static_assert(sizeof(MRHumanRestingSupportRotationZBasisGPU) == 32);
static_assert(sizeof(MRHumanRestingSupportDispatchGPU) == 48);
static_assert(sizeof(MRHumanRestingSupportDebugGPU) == 48);

namespace nm_human_resting_support {
inline uint skinInfluenceStride() {
    return kUseTiledSkinInfluenceLayout ? 32u : 1u;
}
inline bool validSkinInfluenceRange(
    const MRHumanRestingVertexMap mappedVertex,
    const uint influenceBufferCount
) {
    if (mappedVertex.influenceCount == 0u ||
        mappedVertex.firstInfluence >= influenceBufferCount) {
        return false;
    }
    const uint stride = skinInfluenceStride();
    return mappedVertex.influenceCount - 1u <=
        (influenceBufferCount - 1u - mappedVertex.firstInfluence) / stride;
}
inline uint skinInfluenceIndex(
    const MRHumanRestingVertexMap mappedVertex, const uint localIndex
) {
    return mappedVertex.firstInfluence +
        localIndex * skinInfluenceStride();
}
inline uint orderedFloat(float value) {
    const uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
}
inline MRCompensatedPositionGPU skinInfluencePosition(
    const MRHumanRestingInfluence influence,
    device const MRArticulatedBodyPoseGPU* bodies,
    device const float4* bodyPositionLow,
    device const float4* normalizedOrientations,
    const MRHumanRestingSupportDispatchGPU dispatch,
    const uint environment, thread bool& valid
) {
    valid = true;
        const uint bodyIndex = influence.body.x;
        if (bodyIndex == 0xffffffffu ||
            bodyIndex < dispatch.identity.x ||
            bodyIndex - dispatch.identity.x >= dispatch.strides.x ||
            !all(isfinite(influence.positionAndWeight)) ||
            !(influence.positionAndWeight.w > 0.0f)) {
            const float nan = as_type<float>(0x7fc00000u);
            valid = false;
            return {{nan, nan, nan, 0.0f}, {nan, nan, nan, 0.0f}};
        }
        const uint localBody = bodyIndex - dispatch.identity.x;
        const uint bodyOffset = environment * dispatch.strides.x + localBody;
        const MRArticulatedBodyPoseGPU pose = bodies[bodyOffset];
        const float4 orientation = normalizedOrientations[bodyOffset];
        if (!all(isfinite(orientation))) {
            const float nan = as_type<float>(0x7fc00000u);
            valid = false;
            return {{nan, nan, nan, 0.0f}, {nan, nan, nan, 0.0f}};
        }
        const MRCompensatedPositionGPU bodyPoint =
            mrCompensatedVectorAdd(
                MRCompensatedPositionGPU{
                    pose.position,
                    bodyPositionLow[bodyOffset]
                },
                mrCompensatedQuaternionRotate(
                    orientation, influence.positionAndWeight
                )
            );
    return mrCompensatedVectorScale(
        bodyPoint, {influence.positionAndWeight.w, 0.0f});
}
inline MRCompensatedPositionGPU skinPosition(
    const MRHumanRestingVertexMap mappedVertex,
    device const MRHumanRestingInfluence* influences,
    device const MRArticulatedBodyPoseGPU* bodies,
    device const float4* bodyPositionLow,
    device const float4* normalizedOrientations,
    const MRHumanRestingSupportDispatchGPU dispatch,
    const uint environment
) {
    MRCompensatedPositionGPU result = mrCompensatedVector(float4(0.0f));
    if (!validSkinInfluenceRange(mappedVertex, dispatch.identity.y)) {
        const float nan = as_type<float>(0x7fc00000u);
        return {{nan, nan, nan, 0.0f}, {nan, nan, nan, 0.0f}};
    }
    for (uint influenceIndex = 0u;
         influenceIndex < mappedVertex.influenceCount;
         ++influenceIndex) {
        const MRHumanRestingInfluence influence =
            influences[nm_human_resting_support::skinInfluenceIndex(
                mappedVertex, influenceIndex)];
        bool valid = true;
        const auto weighted = skinInfluencePosition(influence, bodies,
            bodyPositionLow, normalizedOrientations, dispatch, environment, valid);
        if (!valid) return weighted;
        result = mrCompensatedVectorAdd(result, weighted);
    }
    return result;
}
inline MRCompensatedPositionGPU skinPositionSimd(
    const MRHumanRestingVertexMap mappedVertex,
    device const MRHumanRestingInfluence* influences,
    device const MRArticulatedBodyPoseGPU* bodies,
    device const float4* bodyPositionLow,
    device const float4* normalizedOrientations,
    const MRHumanRestingSupportDispatchGPU dispatch,
    const uint environment, const uint lane, const uint width
) {
    MRCompensatedPositionGPU result = mrCompensatedVector(float4(0.0f));
    if (!validSkinInfluenceRange(mappedVertex, dispatch.identity.y)) {
        const float nan = as_type<float>(0x7fc00000u);
        return {{nan, nan, nan, 0.0f}, {nan, nan, nan, 0.0f}};
    }
    for(uint first=0u;first<mappedVertex.influenceCount;first+=width) {
        const uint count=min(width,mappedVertex.influenceCount-first);
        MRCompensatedPositionGPU weighted = mrCompensatedVector(float4(0.0f));
        bool valid=true;
        if(lane<count) {
            const auto influence=influences[skinInfluenceIndex(mappedVertex,first+lane)];
            weighted=skinInfluencePosition(influence,bodies,bodyPositionLow,
                normalizedOrientations,dispatch,environment,valid);
        }
        if(!simd_all(valid)) {
            const float nan=as_type<float>(0x7fc00000u);
            return {{nan,nan,nan,0.0f},{nan,nan,nan,0.0f}};
        }
        for(uint sourceLane=0u;sourceLane<count;++sourceLane) {
            const MRCompensatedPositionGPU term{
                simd_broadcast(weighted.high,sourceLane),
                simd_broadcast(weighted.low,sourceLane)};
            result=mrCompensatedVectorAdd(result,term);
        }
    }
    return result;
}
inline MRCompensatedScalar skinWorldZ(
    const MRHumanRestingVertexMap mappedVertex,
    device const MRHumanRestingInfluence* influences,
    device const MRArticulatedBodyPoseGPU* bodies,
    device const float4* bodyPositionLow,
    device const float4* normalizedOrientations,
    const MRHumanRestingSupportDispatchGPU dispatch,
    const uint environment
) {
    if (!validSkinInfluenceRange(mappedVertex, dispatch.identity.y)) {
        const float nan = as_type<float>(0x7fc00000u);
        return {nan, nan};
    }
    MRCompensatedScalar result{0.0f, 0.0f};
    for (uint influenceIndex = 0u;
         influenceIndex < mappedVertex.influenceCount;
         ++influenceIndex) {
        const MRHumanRestingInfluence influence =
            influences[nm_human_resting_support::skinInfluenceIndex(
                mappedVertex, influenceIndex)];
        const uint bodyIndex = influence.body.x;
        if (bodyIndex == 0xffffffffu ||
            bodyIndex < dispatch.identity.x ||
            bodyIndex - dispatch.identity.x >= dispatch.strides.x ||
            !all(isfinite(influence.positionAndWeight)) ||
            !(influence.positionAndWeight.w > 0.0f)) {
            const float nan = as_type<float>(0x7fc00000u);
            return {nan, nan};
        }
        const uint bodyOffset = environment * dispatch.strides.x +
            bodyIndex - dispatch.identity.x;
        const MRArticulatedBodyPoseGPU pose = bodies[bodyOffset];
        const float4 bodyLow = bodyPositionLow[bodyOffset];
        const float4 orientation = normalizedOrientations[bodyOffset];
        if (!all(isfinite(pose.position)) || !all(isfinite(bodyLow)) ||
            !all(isfinite(orientation))) {
            const float nan = as_type<float>(0x7fc00000u);
            return {nan, nan};
        }
        const MRCompensatedPositionGPU rotated =
            mrCompensatedQuaternionRotate(
                orientation, influence.positionAndWeight);
        const MRCompensatedScalar bodyPointZ = mrCompensatedAdd(
            {pose.position.z, bodyLow.z},
            {rotated.high.z, rotated.low.z});
        result = mrCompensatedAdd(
            result,
            mrCompensatedMultiply(
                bodyPointZ, {influence.positionAndWeight.w, 0.0f}));
    }
    return result;
}
inline MRCompensatedScalar skinWorldZBasis(
    const MRHumanRestingVertexMap mappedVertex,
    device const MRHumanRestingInfluence* influences,
    device const MRArticulatedBodyPoseGPU* bodies,
    device const float4* bodyPositionLow,
    device const MRHumanRestingSupportRotationZBasisGPU* rotationZBasis,
    const MRHumanRestingSupportDispatchGPU dispatch,
    const uint environment
) {
    if (!validSkinInfluenceRange(mappedVertex, dispatch.identity.y)) {
        const float nan = as_type<float>(0x7fc00000u);
        return {nan, nan};
    }
    MRCompensatedScalar result{0.0f, 0.0f};
    for (uint influenceIndex = 0u;
         influenceIndex < mappedVertex.influenceCount;
         ++influenceIndex) {
        const MRHumanRestingInfluence influence =
            influences[nm_human_resting_support::skinInfluenceIndex(
                mappedVertex, influenceIndex)];
        const uint bodyIndex = influence.body.x;
        if (!kUseValidatedSkinInfluences &&
            (bodyIndex == 0xffffffffu ||
             bodyIndex < dispatch.identity.x ||
             bodyIndex - dispatch.identity.x >= dispatch.strides.x ||
             !all(isfinite(influence.positionAndWeight)) ||
             !(influence.positionAndWeight.w > 0.0f))) {
            const float nan = as_type<float>(0x7fc00000u);
            return {nan, nan};
        }
        const uint bodyOffset = environment * dispatch.strides.x +
            bodyIndex - dispatch.identity.x;
        const MRArticulatedBodyPoseGPU pose = bodies[bodyOffset];
        const float4 bodyLow = bodyPositionLow[bodyOffset];
        const MRHumanRestingSupportRotationZBasisGPU basis =
            rotationZBasis[bodyOffset];
        if (basis.high.w != 1.0f) {
            const float nan = as_type<float>(0x7fc00000u);
            return {nan, nan};
        }
        const float4 local = influence.positionAndWeight;
        const MRCompensatedScalar rotatedZ = mrCompensatedAdd(
            mrCompensatedAdd(
                mrCompensatedMultiply(
                    {local.x, 0.0f}, {basis.high.x, basis.low.x}),
                mrCompensatedMultiply(
                    {local.y, 0.0f}, {basis.high.y, basis.low.y})),
            mrCompensatedMultiply(
                {local.z, 0.0f}, {basis.high.z, basis.low.z}));
        const MRCompensatedScalar bodyPointZ = mrCompensatedAdd(
            {pose.position.z, bodyLow.z}, rotatedZ);
        result = mrCompensatedAdd(
            result,
            mrCompensatedMultiply(bodyPointZ, {local.w, 0.0f}));
    }
    return result;
}
}
kernel void nm_human_resting_support_normalize_poses(
    constant MRHumanRestingSupportDispatchGPU& dispatch [[buffer(0)]],
    device const MRArticulatedBodyPoseGPU* bodies [[buffer(1)]],
    device float4* normalizedOrientations [[buffer(2)]],
    device MRHumanRestingSupportRotationZBasisGPU* rotationZBasis [[buffer(3)]],
    device const float4* bodyPositionLow [[buffer(4)]],
    uint globalIndex [[thread_position_in_grid]]) {
    const uint bodyStride = dispatch.strides.x;
    const uint environmentCount = dispatch.counts.w;
    const uint total = bodyStride * environmentCount;
    if (globalIndex >= total) return;
    const float4 orientation = bodies[globalIndex].orientation;
    const float normSquared = dot(orientation, orientation);
    if (!all(isfinite(orientation)) || !isfinite(normSquared) ||
        !(normSquared > 1.0e-20f)) {
        const float nan = as_type<float>(0x7fc00000u);
        normalizedOrientations[globalIndex] = float4(nan);
        rotationZBasis[globalIndex].high = float4(nan);
        rotationZBasis[globalIndex].high.w = 0.0f;
        rotationZBasis[globalIndex].low = float4(nan);
        rotationZBasis[globalIndex].low.w = 0.0f;
        return;
    }
    const float4 normalized = orientation * rsqrt(normSquared);
    normalizedOrientations[globalIndex] = normalized;
    const MRCompensatedPositionGPU axisX = mrCompensatedQuaternionRotate(
        normalized, float4(1.0f, 0.0f, 0.0f, 0.0f));
    const MRCompensatedPositionGPU axisY = mrCompensatedQuaternionRotate(
        normalized, float4(0.0f, 1.0f, 0.0f, 0.0f));
    const MRCompensatedPositionGPU axisZ = mrCompensatedQuaternionRotate(
        normalized, float4(0.0f, 0.0f, 1.0f, 0.0f));
    const MRArticulatedBodyPoseGPU pose = bodies[globalIndex];
    const float4 bodyLow = bodyPositionLow[globalIndex];
    const bool finitePose = all(isfinite(pose.position)) &&
        all(isfinite(bodyLow)) &&
        isfinite(axisX.high.z) && isfinite(axisY.high.z) &&
        isfinite(axisZ.high.z) && isfinite(axisX.low.z) &&
        isfinite(axisY.low.z) && isfinite(axisZ.low.z);
    rotationZBasis[globalIndex] = {
        float4(axisX.high.z, axisY.high.z, axisZ.high.z,
               finitePose ? 1.0f : 0.0f),
        float4(axisX.low.z, axisY.low.z, axisZ.low.z, 0.0f),
    };
}
kernel void nm_human_resting_support_positions(
    constant MRHumanRestingSupportDispatchGPU& dispatch [[buffer(0)]],
    device const MRHumanRestingVertexMap* vertexMap [[buffer(1)]],
    device const MRHumanRestingInfluence* influences [[buffer(2)]],
    device const MRArticulatedBodyPoseGPU* bodies [[buffer(3)]],
    device const float4* bodyPositionLow [[buffer(4)]],
    device const uint* regionForVertex [[buffer(5)]],
    device uint* positionKeys [[buffer(6)]],
    device atomic_uint* minimumHeightKeys [[buffer(7)]],
    device atomic_uint* invalidRegionFlags [[buffer(8)]],
    device const float4* normalizedOrientations [[buffer(9)]],
    device const MRHumanRestingSupportRotationZBasisGPU* rotationZBasis [[buffer(10)]],
    constant MRHumanRestingBedGPU& bed [[buffer(11), function_constant(kUseRestingContouredBed)]],
    device const float* bedHeights [[buffer(12), function_constant(kUseRestingContouredBed)]],
    uint globalIndex [[thread_position_in_grid]]) {
    const uint vertexCount = dispatch.counts.x;
    const uint regionCount = dispatch.counts.y;
    const uint environmentCount = dispatch.counts.w;
    if (vertexCount == 0u || globalIndex >= vertexCount * environmentCount) return;
    const uint environment = globalIndex / vertexCount;
    const uint vertexIndex = globalIndex - environment * vertexCount;
    const uint region = regionForVertex[vertexIndex];
    const MRHumanRestingVertexMap mappedVertex = vertexMap[vertexIndex];
    const uint flagsBase = environment * (regionCount + 1u);
    if (region >= regionCount) {
        atomic_store_explicit(invalidRegionFlags + flagsBase + regionCount,
                              1u, memory_order_relaxed);
        return;
    }
    if (mappedVertex.deformationKind != 3u || mappedVertex.influenceCount == 0u) {
        positionKeys[globalIndex] = 0xffffffffu;
        atomic_store_explicit(invalidRegionFlags + flagsBase + region,
                              1u, memory_order_relaxed);
        return;
    }
    float orderingValue;
    if (kUseRestingContouredBed) {
        const MRCompensatedPositionGPU point =
            nm_human_resting_support::skinPosition(mappedVertex, influences,
                bodies, bodyPositionLow, normalizedOrientations, dispatch, environment);
        const auto hit = nmHumanRestingBedQuery(bed, bedHeights,
            point.high.xyz + point.low.xyz);
        orderingValue = hit.valid ? hit.signedGap : NAN;
    } else {
        const MRCompensatedScalar positionZ =
            nm_human_resting_support::skinWorldZBasis(
                mappedVertex, influences, bodies, bodyPositionLow,
                rotationZBasis, dispatch, environment);
        orderingValue = isfinite(positionZ.high) && isfinite(positionZ.low)
            ? positionZ.high + positionZ.low : NAN;
    }
    if (!isfinite(orderingValue)) {
        positionKeys[globalIndex] = 0xffffffffu;
        atomic_store_explicit(invalidRegionFlags + flagsBase + region,
                              1u, memory_order_relaxed);
        return;
    }
    const uint worldZKey = nm_human_resting_support::orderedFloat(orderingValue);
    positionKeys[globalIndex] = worldZKey;
    atomic_fetch_min_explicit(
        minimumHeightKeys + environment * regionCount + region,
        worldZKey,
        memory_order_relaxed
    );
}
kernel void nm_human_resting_support_select(
    constant MRHumanRestingSupportDispatchGPU& dispatch [[buffer(0)]],
    device const uint* regionForVertex [[buffer(1)]],
    device const uint* positionKeys [[buffer(2)]],
    device const atomic_uint* minimumHeightKeys [[buffer(3)]],
    device atomic_uint* selectedVertices [[buffer(4)]],
    uint globalIndex [[thread_position_in_grid]]) {
    const uint vertexCount = dispatch.counts.x;
    const uint regionCount = dispatch.counts.y;
    const uint environmentCount = dispatch.counts.w;
    if (vertexCount == 0u || globalIndex >= vertexCount * environmentCount) return;
    const uint environment = globalIndex / vertexCount;
    const uint vertexIndex = globalIndex - environment * vertexCount;
    const uint region = regionForVertex[vertexIndex];
    if (region >= regionCount) return;
    const uint key = positionKeys[globalIndex];
    if (key == 0xffffffffu) return;
    const uint regionIndex = environment * regionCount + region;
    if (key == atomic_load_explicit(minimumHeightKeys + regionIndex,
                                    memory_order_relaxed)) {
        atomic_fetch_min_explicit(selectedVertices + regionIndex,
                                  vertexIndex, memory_order_relaxed);
    }
}
kernel void nm_human_resting_support_publish(
    constant MRHumanRestingSupportDispatchGPU& dispatch [[buffer(0)]],
    device const MRHumanRestingVertexMap* vertexMap [[buffer(1)]],
    device const MRHumanRestingInfluence* influences [[buffer(2)]],
    device const MRHumanRestingSupportRegionGPU* regions [[buffer(3)]],
    device const atomic_uint* selectedVertices [[buffer(4)]],
    device const uint* positionKeys [[buffer(5)]],
    device MRNumiHumanStandContactGPU* contacts [[buffer(6)]],
    device MRArticulatedPointWorldGPU* pointWorld [[buffer(7)]],
    device float4* pointPositionLow [[buffer(8)]],
    device float* pointJacobians [[buffer(9)]],
    device const atomic_uint* invalidRegionFlags [[buffer(10)]],
    device const MRArticulatedBodyPoseGPU* bodies [[buffer(11)]],
    device const float4* bodyPositionLow [[buffer(12)]],
    device const float4* normalizedOrientations [[buffer(13)]],
    device const uchar* bodyDofAncestry [[buffer(14)]],
    constant uint* ancestryBodyCount [[buffer(15)]],
    constant MRHumanRestingBedGPU& bed [[buffer(16), function_constant(kUseRestingContouredBed)]],
    device const float* bedHeights [[buffer(17), function_constant(kUseRestingContouredBed)]],
    uint globalIndex [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdWidth [[threads_per_simdgroup]]) {
    const uint regionCount = dispatch.counts.y;
    const uint dofCount = dispatch.counts.z;
    const uint environmentCount = dispatch.counts.w;
    const uint regionElements = environmentCount * regionCount;
    const uint jacobianElements = regionElements * 3u * dofCount;
    const uint positionWidth = kUseSkipNonAncestorSupportBodyDofs ? simdWidth : 1u;
    const uint positionElements = regionElements * positionWidth;
    if (regionCount == 0u || dofCount == 0u ||
        globalIndex >= positionElements + jacobianElements) return;
    const bool publishPosition = globalIndex < positionElements;
    const uint element = publishPosition
        ? globalIndex / positionWidth : globalIndex - positionElements;
    const uint scalarsPerRegion = 3u * dofCount;
    const uint scalarsPerEnvironment = regionCount * scalarsPerRegion;
    const uint localElement = element % scalarsPerEnvironment;
    const uint regionElement = publishPosition ? element :
        (kUseSkipNonAncestorSupportBodyDofs
            ? (element / scalarsPerEnvironment) * regionCount +
                localElement % regionCount
            : element / scalarsPerRegion);
    const uint ownedJacobianIndex = kUseSkipNonAncestorSupportBodyDofs
        ? localElement / regionCount : element % scalarsPerRegion;
    const uint environment = regionElement / regionCount;
    const uint region = regionElement - environment * regionCount;
    const uint query = regions[region].pointQueryIndex;
    const uint vertexIndex = atomic_load_explicit(
        selectedVertices + regionElement, memory_order_relaxed);
    const uint flagsBase = environment * (regionCount + 1u);
    const bool valid =
        dispatch.identity.w == 1u &&
        dispatch.identity.z == regionCount &&
        regions[region].reserved0 == 0u && regions[region].reserved1 == 0u &&
        contacts[region].pointQueryIndex == query &&
        contacts[region].sourceGeometryIndex == regions[region].sourceGeometryIndex &&
        atomic_load_explicit(invalidRegionFlags + flagsBase + region,
                             memory_order_relaxed) == 0u &&
        atomic_load_explicit(invalidRegionFlags + flagsBase + regionCount,
                             memory_order_relaxed) == 0u &&
        query < dispatch.strides.z && vertexIndex < dispatch.counts.x;
    if (query >= dispatch.strides.z) return;
    const uint pointBase = environment * dispatch.strides.z + query;
    if (!valid) {
        const float nan = as_type<float>(0x7fc00000u);
        if (publishPosition) {
            if (kUseSkipNonAncestorSupportBodyDofs && lane != 0u) return;
            pointWorld[pointBase].position = float4(nan, nan, nan, 1.0f);
            pointPositionLow[pointBase] = float4(nan, nan, nan, 0.0f);
        } else {
            const uint jacobianIndex = ownedJacobianIndex;
            const uint axis = jacobianIndex / dofCount;
            const uint dof = jacobianIndex - axis * dofCount;
            pointJacobians[environment * dispatch.strides.w +
                (3u * query + axis) * dofCount + dof] = nan;
        }
        return;
    }
    if (publishPosition) {
        const MRCompensatedPositionGPU point = kUseSkipNonAncestorSupportBodyDofs
            ? nm_human_resting_support::skinPositionSimd(
                vertexMap[vertexIndex], influences, bodies, bodyPositionLow,
                normalizedOrientations, dispatch, environment, lane, simdWidth)
            : nm_human_resting_support::skinPosition(
                vertexMap[vertexIndex], influences, bodies, bodyPositionLow,
                normalizedOrientations, dispatch, environment);
        if (kUseSkipNonAncestorSupportBodyDofs && lane != 0u) return;
        const MRHumanRestingSupportPositionGPU position = {
            point.high, point.low
        };
        if (!all(isfinite(position.high)) || !all(isfinite(position.low))) {
            const float nan = as_type<float>(0x7fc00000u);
            pointWorld[pointBase].position = float4(nan, nan, nan, 1.0f);
            pointPositionLow[pointBase] = float4(nan, nan, nan, 0.0f);
            return;
        }
        pointWorld[pointBase].position = float4(position.high.xyz, 1.0f);
        pointPositionLow[pointBase] = float4(position.low.xyz, 0.0f);
        if (kUseRestingContouredBed) {
            const auto hit = nmHumanRestingBedQuery(bed, bedHeights,
                position.high.xyz+position.low.xyz);
            contacts[region].planePoint = float4(hit.point, 0.0f);
            contacts[region].planeNormal = hit.valid
                ? float4(hit.normal, 0.0f) : float4(NAN);
        }
        return;
    }
    const uint jacobianIndex = ownedJacobianIndex;
    const uint axis = jacobianIndex / dofCount;
    const uint dof = jacobianIndex - axis * dofCount;
    const MRHumanRestingVertexMap mappedVertex = vertexMap[vertexIndex];
    const uint jacobianEnvironmentBase = environment * dispatch.strides.w;
    float value = 0.0f;
    for (uint influenceIndex = 0u;
         influenceIndex < mappedVertex.influenceCount;
         ++influenceIndex) {
        const MRHumanRestingInfluence influence =
            influences[nm_human_resting_support::skinInfluenceIndex(
                mappedVertex, influenceIndex)];
        const uint body = influence.body.x;
        if (body == 0xffffffffu ||
            body < dispatch.identity.x ||
            body - dispatch.identity.x >= dispatch.strides.x ||
            (kUseSkipNonAncestorSupportBodyDofs &&
             body - dispatch.identity.x >= ancestryBodyCount[0])) {
            value = as_type<float>(0x7fc00000u);
            break;
        }
        if (kUseSkipNonAncestorSupportBodyDofs) {
            const uint ancestryIndex =
                (body - dispatch.identity.x) * dofCount + dof;
            if (bodyDofAncestry[ancestryIndex] == 0u) continue;
        }
        const uint probe = dispatch.strides.y +
            4u * (body - dispatch.identity.x);
        const float jacobian0 = pointJacobians[
            jacobianEnvironmentBase + (3u * probe + axis) * dofCount + dof];
        const float jacobianX = pointJacobians[
            jacobianEnvironmentBase + (3u * (probe + 1u) + axis) * dofCount + dof];
        const float jacobianY = pointJacobians[
            jacobianEnvironmentBase + (3u * (probe + 2u) + axis) * dofCount + dof];
        const float jacobianZ = pointJacobians[
            jacobianEnvironmentBase + (3u * (probe + 3u) + axis) * dofCount + dof];
        const float3 local = influence.positionAndWeight.xyz;
        const float pointJacobian = jacobian0 + local.x * (jacobianX - jacobian0) +
            local.y * (jacobianY - jacobian0) +
            local.z * (jacobianZ - jacobian0);
        value += influence.positionAndWeight.w * pointJacobian;
    }
    pointJacobians[jacobianEnvironmentBase +
        (3u * query + axis) * dofCount + dof] = value;
}
kernel void nm_human_resting_support_debug(
    constant MRHumanRestingSupportDispatchGPU& dispatch [[buffer(0)]],
    device const MRHumanRestingSupportRegionGPU* regions [[buffer(1)]],
    device const atomic_uint* selectedVertices [[buffer(2)]],
    device const atomic_uint* invalidRegionFlags [[buffer(3)]],
    device const MRArticulatedPointWorldGPU* pointWorld [[buffer(4)]],
    device const float4* pointPositionLow [[buffer(5)]],
    device const float* pointJacobians [[buffer(6)]],
    device MRHumanRestingSupportDebugGPU* output [[buffer(7)]],
    uint globalIndex [[thread_position_in_grid]]) {
    const uint regionCount = dispatch.counts.y;
    const uint dofCount = dispatch.counts.z;
    const uint environmentCount = dispatch.counts.w;
    const uint regionElements = environmentCount * regionCount;
    if (regionCount == 0u || globalIndex >= regionElements) return;
    const uint environment = globalIndex / regionCount;
    const uint region = globalIndex - environment * regionCount;
    const uint flagsBase = environment * (regionCount + 1u);
    const uint query = regions[region].pointQueryIndex;
    const uint selected = atomic_load_explicit(
        selectedVertices + globalIndex, memory_order_relaxed);
    MRHumanRestingSupportDebugGPU result{};
    result.status = uint4(
        selected,
        atomic_load_explicit(invalidRegionFlags + flagsBase + region,
                             memory_order_relaxed),
        atomic_load_explicit(invalidRegionFlags + flagsBase + regionCount,
                             memory_order_relaxed),
        0u);
    const bool valid = query < dispatch.strides.z &&
        selected < dispatch.counts.x;
    if (valid) {
        const uint pointIndex = environment * dispatch.strides.z + query;
        result.position = float4(
            pointWorld[pointIndex].position.xyz + pointPositionLow[pointIndex].xyz,
            1.0f);
    } else {
        result.status.y = 1u;
        result.position = float4(as_type<float>(0x7fc00000u));
    }
    const uint jacobianBase = environment * dispatch.strides.w +
        (3u * query * dofCount);
    float maximumAbsolute = 0.0f;
    uint nonfiniteCount = 0u;
    if (query < dispatch.strides.z) {
        for (uint row = 0u; row < 3u * dofCount; ++row) {
            const float value = pointJacobians[jacobianBase + row];
            if (!isfinite(value)) ++nonfiniteCount;
            else maximumAbsolute = max(maximumAbsolute, abs(value));
        }
    } else {
        nonfiniteCount = 3u * dofCount;
    }
    result.status.w = nonfiniteCount;
    result.jacobian = float4(maximumAbsolute, 0.0f, 0.0f, 0.0f);
    output[globalIndex] = result;
}

constant bool kGasTransportSubcycling [[function_constant(40)]];
constant bool kUseGasTransportSubcycling =
    is_function_constant_defined(kGasTransportSubcycling) ? kGasTransportSubcycling : false;
constant bool kRespiratorySubcycling [[function_constant(41)]];
constant bool kUseRespiratorySubcycling =
    is_function_constant_defined(kRespiratorySubcycling) ? kRespiratorySubcycling : false;
constant bool kParallelRespiratoryMuscles [[function_constant(44)]];
constant bool kUseParallelRespiratoryMuscles =
    is_function_constant_defined(kParallelRespiratoryMuscles)
        ? kParallelRespiratoryMuscles : false;
constant bool kParallelRespiratoryGas [[function_constant(46)]];
constant bool kUseParallelRespiratoryGas =
    is_function_constant_defined(kParallelRespiratoryGas)
        ? kParallelRespiratoryGas : false;
inline float3 restingRotate(float4 q,float3 v) {
    return v+2.0f*cross(q.xyz,cross(q.xyz,v)+q.w*v);
}
inline float4 restingRibRotation(constant MRHumanRestingAnatomyGPU& anatomy,uint rib,float excursion) {
    const float angle=excursion*anatomy.ribPivotAndGain[rib].w;
    return float4(anatomy.ribAxis[rib].xyz*sin(.5f*angle),cos(.5f*angle));
}
inline float3 restingRibPoint(constant MRHumanRestingAnatomyGPU& anatomy,uint rib,float4 rotation,float3 p) {
    const float3 pivot=anatomy.ribPivotAndGain[rib].xyz;return pivot+restingRotate(rotation,p-pivot);
}
kernel void nm_human_resting_common_coordinates(
    device const NMHumanRespirationState* respiration [[buffer(0)]],
    constant MRHumanRestingCommonFieldGPU& parameters [[buffer(1)]],
    device const MRHumanRestingCommonCoordinateBoxGPU* boxes [[buffer(2)]],
    device MRHumanRestingCommonCoordinatesGPU* output [[buffer(3)]],uint lane [[thread_position_in_grid]]) {
    if(lane)return;
    const auto state=respiration[0];
    const float targets[7]={state.chamberVolumes.x,state.chamberVolumes.y,state.chamberVolumes.z,
        state.chamberVolumes.w,parameters.materialTargetVolumes.x,parameters.materialTargetVolumes.y,
        parameters.materialTargetVolumes.z};
    float coordinates[7]={0,0,0,0,0,0,0},residual=INFINITY;
    uint iterations=0,matched=0xffffffffu;
    const uint status=nmHumanRestingCommonSolve(parameters,boxes,targets,coordinates,iterations,matched,residual);
    output[0].first=status?float4(NAN):float4(coordinates[0],coordinates[1],coordinates[2],coordinates[3]);
    output[0].second=status?float4(NAN):float4(coordinates[4],coordinates[5],coordinates[6],0);
    output[0].status=uint4(status,iterations,matched,0);
    output[0].diagnostics=float4(residual,0,0,0);
}
kernel void nm_human_resting_common_coordinates_simd(
    device const NMHumanRespirationState* respiration [[buffer(0)]],
    constant MRHumanRestingCommonFieldGPU& parameters [[buffer(1)]],
    device const MRHumanRestingCommonCoordinateBoxGPU* boxes [[buffer(2)]],
    device MRHumanRestingCommonCoordinatesGPU* output [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]]) {
    const auto state=respiration[0];
    const float targets[7]={state.chamberVolumes.x,state.chamberVolumes.y,state.chamberVolumes.z,
        state.chamberVolumes.w,parameters.materialTargetVolumes.x,parameters.materialTargetVolumes.y,
        parameters.materialTargetVolumes.z};
    float coordinates[7]={0,0,0,0,0,0,0},residual=INFINITY;
    uint iterations=0,matched=0xffffffffu;
    const uint status=nmHumanRestingCommonSolveSimd(
        parameters,boxes,targets,coordinates,lane,iterations,matched,residual);
    if(lane!=0u) return;
    output[0].first=status?float4(NAN):float4(coordinates[0],coordinates[1],coordinates[2],coordinates[3]);
    output[0].second=status?float4(NAN):float4(coordinates[4],coordinates[5],coordinates[6],0);
    output[0].status=uint4(status,iterations,matched,0);
    output[0].diagnostics=float4(residual,0,0,0);
}
kernel void nm_human_resting_common_coordinate_status_gate(
    device const MRHumanRestingCommonCoordinatesGPU* coordinates [[buffer(0)]],
    device NMMatterStatusGPU* statuses [[buffer(1)]],uint lane [[thread_position_in_grid]]) {
    if(lane==0u&&coordinates[0].status.x!=0u&&statuses[0].code==NM_STATUS_SUCCESS)
        statuses[0].code=NM_STATUS_MULTIPHYSICS_FAILURE;
}
inline float nmHumanRestingCardiacVolume(constant MRHumanRestingAnatomyGPU& anatomy,uint chamber,float q) {
    const float4 c=anatomy.chamberVolumePolynomial[chamber];return ((c.w*q+c.z)*q+c.y)*q+c.x;
}
kernel void nm_human_resting_cardiac_q(
    device const NMHumanRespirationState* respiration [[buffer(0)]],
    constant MRHumanRestingAnatomyGPU& anatomy [[buffer(1)]],device float* qOut [[buffer(2)]],
    uint chamber [[thread_position_in_grid]]) {
    if(chamber>=4)return;
    const float target=respiration[0].chamberVolumes[chamber];
    float low=-.999f,high=1.0f;
    if(!isfinite(target)||nmHumanRestingCardiacVolume(anatomy,chamber,low)>target||
       nmHumanRestingCardiacVolume(anatomy,chamber,high)<target){qOut[chamber]=NAN;return;}
    for(uint iteration=0;iteration<28;++iteration){
        const float mid=.5f*(low+high);
        if(nmHumanRestingCardiacVolume(anatomy,chamber,mid)<target)low=mid;else high=mid;
    }
    qOut[chamber]=.5f*(low+high);
}
kernel void nm_human_resting_cardiac_wall_q(
    constant MRHumanRestingCardiacWallGPU& wall [[buffer(0)]],
    device const float* cardiacQ [[buffer(1)]],device float4* result [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if(i)return;
    if(!wall.chambersAndFlags.z){result[0]=0;return;}
    const float r=cardiacQ[wall.chambersAndFlags.x]/wall.scalesAndVolume.x;
    const float l=cardiacQ[wall.chambersAndFlags.y]/wall.scalesAndVolume.y;
    const float4 p0=wall.volumePolynomial[0],p1=wall.volumePolynomial[1],p2=wall.volumePolynomial[2],
        p3=wall.volumePolynomial[3],p4=wall.volumePolynomial[4];
    const float f0=p0.x+p0.y*r+p0.z*l+p0.w*r*r+p1.x*r*l+p1.y*l*l+
        p1.z*r*r*r+p1.w*r*r*l+p2.x*r*l*l+p2.y*l*l*l;
    const float f1=p2.z+p2.w*r+p3.x*l+p3.y*r*r+p3.z*r*l+p3.w*l*l;
    const float f2=p4.x+p4.y*r+p4.z*l,f3=p4.w;
    float low=wall.closureBoundsAndTolerance.x/wall.scalesAndVolume.z;
    float high=wall.closureBoundsAndTolerance.y/wall.scalesAndVolume.z;
    const float target=wall.scalesAndVolume.w;
    const float vlo=((f3*low+f2)*low+f1)*low+f0;
    const float vhi=((f3*high+f2)*high+f1)*high+f0;
    float minimumDerivative=min((3*f3*low+2*f2)*low+f1,(3*f3*high+2*f2)*high+f1);
    if(f3!=0) {
        const float stationary=-f2/(3*f3);
        if(stationary>low&&stationary<high)
            minimumDerivative=min(minimumDerivative,(3*f3*stationary+2*f2)*stationary+f1);
    }
    if(!all(isfinite(float4(vlo,vhi,minimumDerivative,target)))||target<=0||low>=high||
       minimumDerivative<=1.e-12f||vlo>target||vhi<target) {
        result[0]=float4(NAN,NAN,target,1);return;
    }
    for(uint iteration=0;iteration<32;++iteration) {
        const float mid=.5f*(low+high);
        if(((f3*mid+f2)*mid+f1)*mid+f0<target)low=mid;else high=mid;
    }
    const float s=.5f*(low+high),volume=((f3*s+f2)*s+f1)*s+f0;
    const float error=abs(volume-target)/target;
    result[0]=float4(s*wall.scalesAndVolume.z,volume,target,
        !isfinite(error)||error>wall.closureBoundsAndTolerance.z?1.0f:0.0f);
}
kernel void nm_human_resting_skin(
    constant uint4& d [[buffer(0)]], device const MRHumanRestingVertexMap* map [[buffer(1)]],
    device const MRHumanRestingInfluence* influences [[buffer(2)]], device const MRBodyStateGPU* bodies [[buffer(3)]],
    device MRVisualVertexGPUV2* vertices [[buffer(4)]],
    device const NMHumanRespirationState* respiration [[buffer(5)]],
    constant MRHumanRestingAnatomyGPU& anatomy [[buffer(6)]],device const float* cardiacQ [[buffer(7)]],
    device const MRHumanRestingCardiacWallVertexGPU* wallMap [[buffer(8)]],
    constant MRHumanRestingCardiacWallGPU& wall [[buffer(9)]],device const float4* wallQ [[buffer(10)]],
    device const MRHumanRestingCommonFieldVertexGPU* commonMap [[buffer(11)]],
    device const MRHumanRestingCommonCoordinatesGPU* commonCoordinates [[buffer(12)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=d.x)return;
    auto m=map[i];if(!m.influenceCount)return;
    const uint influenceStride = m.deformationKind == 3u
        ? nm_human_resting_support::skinInfluenceStride() : 1u;
    const auto soleInfluence=influences[m.firstInfluence];
    const bool sourceLocalDeformation=m.deformationKind==1u||m.deformationKind==3u||
        m.deformationKind==4u||m.deformationKind==9u||m.deformationKind==11u||m.deformationKind==12u;
    const bool exactBodyLocal=sourceLocalDeformation&&m.influenceCount==1u&&
        soleInfluence.body.x==anatomy.bodyAndFlags.x&&soleInfluence.positionAndWeight.w==1.0f;
    float3 p=0,n=0,strongest=0;float strongestWeight=-1;
    for(uint j=0;j<m.influenceCount;++j) {
        auto influence=influences[m.firstInfluence+j*influenceStride];
        float w=influence.positionAndWeight.w;
        float3 ni;
        if(influence.body.x==0xffffffffu) {
            p+=w*influence.positionAndWeight.xyz;ni=influence.normal.xyz;
        }else {
            auto body=bodies[influence.body.x];
            p+=w*(body.position.xyz+restingRotate(body.orientation,influence.positionAndWeight.xyz));
            ni=restingRotate(body.orientation,influence.normal.xyz);
        }
        n+=w*ni;if(w>strongestWeight){strongestWeight=w;strongest=ni;}
    }
    if(m.deformationKind && anatomy.bodyAndFlags.y) {
        auto body=bodies[anatomy.bodyAndFlags.x];
        const float4 inverse=float4(-body.orientation.xyz,body.orientation.w);
        float3 local=exactBodyLocal?soleInfluence.positionAndWeight.xyz:
            restingRotate(inverse,p-body.position.xyz);
        float3 normal=exactBodyLocal?soleInfluence.normal.xyz:restingRotate(inverse,n);
        const auto state=respiration[0];
        if(m.deformationKind==2) {
            const float4 cavity=anatomy.chamberCenterAndVolume[m.chamberIndex];
            const float weight=m.deformationWeight.x;
            if(weight!=0.0f) {
                const float scale=1.0f+cardiacQ[m.chamberIndex]*weight;
                local=cavity.xyz+scale*(local-cavity.xyz);
                normal/=scale;
            }
        } else if(m.deformationKind==12) {
            const auto solved=commonCoordinates[0];
            if(solved.status.x!=0u) local=float3(NAN);
            else {
                const auto basis=commonMap[m.chamberIndex];
                const float coordinate[7]={solved.first.x,solved.first.y,solved.first.z,solved.first.w,
                    solved.second.x,solved.second.y,solved.second.z};
                for(uint c=0;c<7;++c) local+=basis.displacement[c].xyz*coordinate[c];
            }
        } else if(m.deformationKind==10) {
            const auto basis=wallMap[m.chamberIndex];
            local+=basis.first.xyz*cardiacQ[wall.chambersAndFlags.x]+
                basis.second.xyz*cardiacQ[wall.chambersAndFlags.y]+basis.closure.xyz*wallQ[0].x;
            if(wallQ[0].w!=0)local=float3(NAN);
        } else if(m.deformationKind==8) {
            const float q=cardiacQ[m.chamberIndex],weight=m.deformationWeight.w;
            const float scale=1.0f+q*weight;
            local+=m.deformationWeight.xyz*q*weight;
            const float radialSquared=dot(m.deformationWeight.xyz,m.deformationWeight.xyz);
            if(radialSquared>1.e-12f) {
                const float3 radial=m.deformationWeight.xyz*rsqrt(radialSquared);
                normal+=radial*dot(normal,radial)*(1.0f/scale-1.0f);
            }
        } else if(m.deformationKind==5||m.deformationKind==7) {
            const float excursion=state.motion.y/anatomy.muscleAreas.y;
            const float4 rotation=restingRibRotation(anatomy,m.chamberIndex,excursion);
            float3 mapped=restingRibPoint(anatomy,m.chamberIndex,rotation,local);
            float3 mappedNormal=restingRotate(rotation,normal);
            if(m.deformationKind==7) {
                const uint other=uint(m.deformationWeight.y);
                const float4 otherRotation=restingRibRotation(anatomy,other,excursion);
                mapped=mix(mapped,restingRibPoint(anatomy,other,otherRotation,local),m.deformationWeight.z);
                mappedNormal=mix(mappedNormal,restingRotate(otherRotation,normal),m.deformationWeight.z);
            }
            local=mapped;normal=mappedNormal;
        } else if(m.deformationKind==6) {
            local+=anatomy.anteriorAxis.xyz*(state.motion.y/anatomy.muscleAreas.y);
        } else {
            const float3 axis=anatomy.superiorAxisAndHeight.xyz;
            const float basalWeight=m.respiratoryBasis.w;
            const float3 gradient=m.respiratoryBasis.xyz;
            const float displacement=state.motion.x/anatomy.lungBasalBlend.z;
            const float afterDiaphragm=anatomy.lungAnchorAndVolume.w+state.motion.x;
            const float radial=sqrt((afterDiaphragm+state.motion.y)/afterDiaphragm);
            const float3 offset=local-anatomy.lungAnchorAndVolume.xyz;
            const float3 along=dot(offset,axis)*axis,across=offset-along;
            const float weight=m.deformationKind==11?m.deformationWeight.x:
                ((m.deformationKind==3||m.deformationKind==9)?m.deformationWeight.x:1.0f);
            const float3 mapped=local-axis*(displacement*basalWeight)+(radial-1.0f)*across;
            local=mix(local,mapped,weight);
            const float3 normalAlong=dot(normal,axis)*axis;
            const float axialJacobian=1.0f-weight*displacement*dot(gradient,axis);
            normal=normalAlong/axialJacobian+
                (normal-normalAlong+weight*displacement*dot(normal,axis)/axialJacobian*
                 (gradient-dot(gradient,axis)*axis))/mix(1.0f,radial,weight);
        }
        p=body.position.xyz+restingRotate(body.orientation,local);
        n=restingRotate(body.orientation,normal);
    }
    vertices[i].position=float4(p,1);
    const float3 unitNormal=normalize(dot(n,n)>1e-12f?n:strongest);
    vertices[i].normalAndTangentSign=float4(unitNormal,1);
    const float3 tangentAxis=abs(unitNormal.z)<.9f?float3(0,0,1):float3(0,1,0);
    vertices[i].tangent=float4(normalize(cross(tangentAxis,unitNormal)),0);
}
kernel void nm_human_resting_cardiac_wall_normals(
    constant uint& count [[buffer(0)]],device const uint4* ranges [[buffer(1)]],
    device const uint* incidentTriangleStarts [[buffer(2)]],device const uint* indices [[buffer(3)]],
    device MRVisualVertexGPUV2* vertices [[buffer(4)]],uint i [[thread_position_in_grid]]) {
    if(i>=count)return;
    const uint4 row=ranges[i];float3 normal=0;
    for(uint j=row.y;j<row.y+row.z;++j) {
        const uint at=incidentTriangleStarts[j];
        const float3 a=vertices[indices[at]].position.xyz,b=vertices[indices[at+1]].position.xyz,
            c=vertices[indices[at+2]].position.xyz;
        normal+=cross(b-a,c-a);
    }
    const float squared=dot(normal,normal);
    if(!isfinite(squared)||squared<=1.e-30f){vertices[row.x].normalAndTangentSign=float4(NAN);return;}
    normal*=rsqrt(squared);vertices[row.x].normalAndTangentSign=float4(normal,1);
    const float3 tangentAxis=abs(normal.z)<.9f?float3(0,0,1):float3(0,1,0);
    vertices[row.x].tangent=float4(normalize(cross(tangentAxis,normal)),0);
}
kernel void nm_human_resting_layers(
    constant uint4& d [[buffer(0)]], device MRVisualInstanceGPUV2* instances [[buffer(1)]],
    device const uint* layerMask [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=d.y)return;
    bool visible=d.z<7 && (layerMask[i]&(1u<<d.z));
    uint bits=MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR|MR_VISUAL_INSTANCE_CASTS_SHADOW|MR_VISUAL_INSTANCE_RECEIVES_SHADOW;
    instances[i].binding.w=visible?bits:0;
}
inline bool nm_human_exact_binary32_products_equal(float a,float b,float c,float d) {
    if(!isfinite(a)||!isfinite(b)||!isfinite(c)||!isfinite(d))return false;
    const float ab=a*b,cd=c*d;
    if(ab!=cd)return false;
    const float abError=fma(a,b,-ab),cdError=fma(c,d,-cd);
    const float minimumNormal=0x1p-126f;
    if(isfinite(abError)&&isfinite(cdError)&&abError!=0.0f&&cdError!=0.0f&&
       abs(abError)>=minimumNormal&&abs(cdError)>=minimumNormal)return abError==cdError;
    const uint aBits=as_type<uint>(a),bBits=as_type<uint>(b);
    const uint cBits=as_type<uint>(c),dBits=as_type<uint>(d);
    const uint aExponent=(aBits>>23u)&0xffu,bExponent=(bBits>>23u)&0xffu;
    const uint cExponent=(cBits>>23u)&0xffu,dExponent=(dBits>>23u)&0xffu;
    if(aExponent==0xffu||bExponent==0xffu||cExponent==0xffu||dExponent==0xffu)return false;
    ulong aSignificand=ulong(aBits&0x7fffffu),bSignificand=ulong(bBits&0x7fffffu);
    ulong cSignificand=ulong(cBits&0x7fffffu),dSignificand=ulong(dBits&0x7fffffu);
    int aPower=-149,bPower=-149,cPower=-149,dPower=-149;
    if(aExponent){aSignificand|=0x800000ul;aPower=int(aExponent)-150;}
    if(bExponent){bSignificand|=0x800000ul;bPower=int(bExponent)-150;}
    if(cExponent){cSignificand|=0x800000ul;cPower=int(cExponent)-150;}
    if(dExponent){dSignificand|=0x800000ul;dPower=int(dExponent)-150;}
    ulong abSignificand=aSignificand*bSignificand,cdSignificand=cSignificand*dSignificand;
    if(abSignificand==0ul||cdSignificand==0ul)return abSignificand==cdSignificand;
    const bool abNegative=((aBits^bBits)&0x80000000u)!=0u;
    const bool cdNegative=((cBits^dBits)&0x80000000u)!=0u;
    if(abNegative!=cdNegative)return false;
    int abPower=aPower+bPower,cdPower=cPower+dPower;
    while((abSignificand&1ul)==0ul){abSignificand>>=1u;++abPower;}
    while((cdSignificand&1ul)==0ul){cdSignificand>>=1u;++cdPower;}
    return abSignificand==cdSignificand&&abPower==cdPower;
}
inline bool nm_human_binary32_requires_point_fallback(float value) {
    const uint bits=as_type<uint>(value);
    const uint exponent=(bits>>23u)&0xffu;
    const uint fraction=bits&0x7fffffu;
    return (exponent==0u&&fraction!=0u)||(exponent>0u&&exponent<25u);
}
inline float nm_human_binary32_difference_tail(float a,float b,float difference) {
    const float bVirtual=a-difference;
    const float aVirtual=difference+bVirtual;
    const float bRound=bVirtual-b;
    const float aRound=a-aVirtual;
    return aRound+bRound;
}
inline void nm_human_add_binary32_word(thread ulong* accumulator,uint limb,ulong word) {
    while(word!=0ul&&limb<9u) {
        const ulong previous=accumulator[limb];
        accumulator[limb]=previous+word;
        word=accumulator[limb]<previous?1ul:0ul;
        ++limb;
    }
}
inline void nm_human_accumulate_binary32_product(
    thread ulong* positive,thread ulong* negative,float a,float b,bool negate) {
    const uint aBits=as_type<uint>(a),bBits=as_type<uint>(b);
    const uint aExponent=(aBits>>23u)&0xffu,bExponent=(bBits>>23u)&0xffu;
    if(aExponent==0xffu||bExponent==0xffu)return;
    ulong aSignificand=ulong(aBits&0x7fffffu),bSignificand=ulong(bBits&0x7fffffu);
    int aPower=-149,bPower=-149;
    if(aExponent){aSignificand|=0x800000ul;aPower=int(aExponent)-150;}
    if(bExponent){bSignificand|=0x800000ul;bPower=int(bExponent)-150;}
    const ulong significand=aSignificand*bSignificand;
    if(significand==0ul)return;
    const int power=aPower+bPower;
    const uint shift=uint(power+298);
    const uint limb=shift>>6u,bit=shift&63u;
    const ulong low=significand<<bit;
    const ulong high=bit?significand>>(64u-bit):0ul;
    const bool negativeProduct=(((aBits^bBits)&0x80000000u)!=0u)!=negate;
    thread ulong* accumulator=negativeProduct?negative:positive;
    nm_human_add_binary32_word(accumulator,limb,low);
    if(high)nm_human_add_binary32_word(accumulator,limb+1u,high);
}
inline bool nm_human_exact_binary32_point_area_component_is_zero(
    float a0,float b0,float c0,float a1,float b1,float c1) {
    thread ulong positive[9],negative[9];
    for(uint i=0;i<9u;++i){positive[i]=0ul;negative[i]=0ul;}
    nm_human_accumulate_binary32_product(positive,negative,b0,c1,false);
    nm_human_accumulate_binary32_product(positive,negative,b0,a1,true);
    nm_human_accumulate_binary32_product(positive,negative,a0,c1,true);
    nm_human_accumulate_binary32_product(positive,negative,b1,c0,true);
    nm_human_accumulate_binary32_product(positive,negative,b1,a0,false);
    nm_human_accumulate_binary32_product(positive,negative,a1,c0,false);
    for(uint i=0;i<9u;++i)if(positive[i]!=negative[i])return false;
    return true;
}
inline bool nm_human_exact_binary32_triangle_is_zero(float3 a,float3 b,float3 c) {
    const float3 u=b-a,v=c-a;
    const bool safeCompensatedRange=
        !nm_human_binary32_requires_point_fallback(a.x)&&!nm_human_binary32_requires_point_fallback(a.y)&&
        !nm_human_binary32_requires_point_fallback(a.z)&&!nm_human_binary32_requires_point_fallback(b.x)&&
        !nm_human_binary32_requires_point_fallback(b.y)&&!nm_human_binary32_requires_point_fallback(b.z)&&
        !nm_human_binary32_requires_point_fallback(c.x)&&!nm_human_binary32_requires_point_fallback(c.y)&&
        !nm_human_binary32_requires_point_fallback(c.z)&&all(isfinite(u))&&all(isfinite(v));
    const bool exactEdges=safeCompensatedRange&&
        nm_human_binary32_difference_tail(b.x,a.x,u.x)==0.0f&&
        nm_human_binary32_difference_tail(b.y,a.y,u.y)==0.0f&&
        nm_human_binary32_difference_tail(b.z,a.z,u.z)==0.0f&&
        nm_human_binary32_difference_tail(c.x,a.x,v.x)==0.0f&&
        nm_human_binary32_difference_tail(c.y,a.y,v.y)==0.0f&&
        nm_human_binary32_difference_tail(c.z,a.z,v.z)==0.0f;
    if(!exactEdges) {
        return nm_human_exact_binary32_point_area_component_is_zero(a.y,b.y,c.y,a.z,b.z,c.z)&&
            nm_human_exact_binary32_point_area_component_is_zero(a.z,b.z,c.z,a.x,b.x,c.x)&&
            nm_human_exact_binary32_point_area_component_is_zero(a.x,b.x,c.x,a.y,b.y,c.y);
    }
    return nm_human_exact_binary32_products_equal(u.y,v.z,u.z,v.y)&&
        nm_human_exact_binary32_products_equal(u.z,v.x,u.x,v.z)&&
        nm_human_exact_binary32_products_equal(u.x,v.y,u.y,v.x);
}
kernel void nm_human_resting_audit_volume_partials(
    constant uint4& d [[buffer(0)]],
    device const MRHumanRestingSurfaceAuditGPU* surfaces [[buffer(1)]],
    device const uint* indices [[buffer(2)]],
    device const MRVisualVertexGPUV2* vertices [[buffer(3)]],
    device const MRHumanRestingVolumeAuditGroupGPU* groups [[buffer(4)]],
    device MRHumanRestingVolumeAuditPartialGPU* partials [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    if(group.x>=d.x)return;
    const auto descriptor=groups[group.x].surfaceAndTriangleRange;
    const auto surface=surfaces[descriptor.x];
    const uint4 owner=surface.indicesAndOwner;
    const float3 origin=vertices[indices[owner.x]].position.xyz;
    threadgroup float volumes[256u];
    threadgroup uint invalid[256u];
    threadgroup uint first[256u];
    float volume=0;uint invalidCount=0;
    uint firstInvalid=0xffffffffu;
    for(uint local=lane;local<descriptor.z;local+=256u) {
        const uint triangle=descriptor.y+local;
        const uint j=owner.x+3u*triangle;
        const float3 pa=vertices[indices[j]].position.xyz;
        const float3 pb=vertices[indices[j+1u]].position.xyz;
        const float3 pc=vertices[indices[j+2u]].position.xyz;
        const float3 u=pb-pa,v=pc-pa;
        const float3 area=cross(u,v);
        const bool bad=!all(isfinite(area));
        const bool collapsed=!bad&&nm_human_exact_binary32_triangle_is_zero(pa,pb,pc);
        if(bad||collapsed) {++invalidCount;firstInvalid=min(firstInvalid,triangle);}
        const float3 a=pa-origin,b=pb-origin,c=pc-origin;
        volume+=dot(a,cross(b,c))/6.0f;
    }
    volumes[lane]=volume;invalid[lane]=invalidCount;first[lane]=firstInvalid;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=256u/2u;stride;stride>>=1u) {
        if(lane<stride) {
            volumes[lane]+=volumes[lane+stride];
            invalid[lane]+=invalid[lane+stride];
            first[lane]=min(first[lane],first[lane+stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(!lane)partials[group.x]={volumes[0],invalid[0],first[0],0u};
}
kernel void nm_human_resting_reduce_volume_audits(
    constant uint4& d [[buffer(0)]],
    device const MRHumanRestingSurfaceAuditGPU* surfaces [[buffer(1)]],
    device const uint* indices [[buffer(2)]],
    device const MRVisualVertexGPUV2* vertices [[buffer(3)]],
    device const MRHumanRestingVolumeAuditRangeGPU* groupRanges [[buffer(4)]],
    device const MRHumanRestingVolumeAuditPartialGPU* partials [[buffer(5)]],
    device const NMHumanRespirationState* respiration [[buffer(6)]],
    constant MRHumanRestingAnatomyGPU& anatomy [[buffer(7)]],
    device float4* result [[buffer(8)]],
    constant MRHumanRestingCardiacWallGPU& wall [[buffer(9)]],
    device MRHumanRestingSurfaceFailureGPU* failureResults [[buffer(10)]],
    constant MRHumanRestingCommonFieldGPU& common [[buffer(11)]],
    device const MRHumanRestingCommonCoordinatesGPU* commonCoordinates [[buffer(12)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=d.y)return;
    const auto surface=surfaces[i];const uint4 owner=surface.indicesAndOwner;
    const uint4 groupRange=groupRanges[i].groupRange;
    float volume=0,compensation=0;uint invalidTriangles=0;
    uint firstInvalid=0xffffffffu;
    for(uint g=groupRange.x;g<groupRange.x+groupRange.y;++g) {
        const auto partial=partials[g];
        const float y=partial.signedVolume-compensation;
        const float next=volume+y;compensation=(next-volume)-y;volume=next;
        invalidTriangles+=partial.invalidTriangleCount;
        firstInvalid=min(firstInvalid,partial.firstInvalidTriangle);
    }
    MRHumanRestingSurfaceFailureGPU firstFailure;
    firstFailure.surfaceTriangleKind=uint4(0xffffffffu,0,0,0);
    firstFailure.vertexIndices=uint4(0);
    firstFailure.renderedPositions[0]=float4(0);
    firstFailure.renderedPositions[1]=float4(0);
    firstFailure.renderedPositions[2]=float4(0);
    if(firstInvalid<owner.y/3u) {
        const uint j=owner.x+3u*firstInvalid;
        const uint3 ids=uint3(indices[j],indices[j+1u],indices[j+2u]);
        const float3 pa=vertices[ids.x].position.xyz;
        const float3 pb=vertices[ids.y].position.xyz;
        const float3 pc=vertices[ids.z].position.xyz;
        const float3 u=pb-pa,v=pc-pa;
        const float3 area=cross(u,v);
        const bool bad=!all(isfinite(area));
        const uint areaFailure=bad?1u:
            (nm_human_exact_binary32_triangle_is_zero(pa,pb,pc)?2u:
                0u);
        firstFailure.surfaceTriangleKind=uint4(i,firstInvalid,areaFailure,0);
        firstFailure.vertexIndices=uint4(ids,0);
        firstFailure.renderedPositions[0]=float4(pa,0);
        firstFailure.renderedPositions[1]=float4(pb,0);
        firstFailure.renderedPositions[2]=float4(pc,0);
    }
    const auto state=respiration[0];
    const float afterDiaphragm=anatomy.lungAnchorAndVolume.w+state.motion.x;
    const float commonExpected=owner.w<4u?state.chamberVolumes[owner.w]:common.materialTargetVolumes[owner.w-4u];
    const float expected=owner.z==10?wall.scalesAndVolume.w:owner.z==2?state.chamberVolumes[owner.w]:
        owner.z==12?commonExpected:(surface.reference.x+surface.reference.y*state.motion.x/anatomy.lungBasalBlend.z)*
        (1+state.motion.y/afterDiaphragm);
    const float relative=abs(abs(volume)-expected)/expected;
    const uint commonFailure=(owner.z==12u&&commonCoordinates[0].status.x!=0u)?4u:0u;
    const uint status=(!isfinite(relative)||relative>2.e-4f?1u:0u)|(invalidTriangles?2u:0u)|commonFailure;
    result[i]=float4(abs(volume),expected,relative,float(status));
    failureResults[i]=firstFailure;
}
kernel void nm_human_resting_audit_mesh_triangles(
    constant uint4& d [[buffer(0)]],device const uint* indices [[buffer(1)]],
    device const MRVisualVertexGPUV2* vertices [[buffer(2)]],
    device uint4* partials [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup uint zero[256],nonfinite[256],first[256];
    uint z=0,n=0,f=0xffffffffu;
    for(uint t=group.x*256u+lane;t<d.x;t+=d.y*256u) {
        const float3 a=vertices[indices[3u*t]].position.xyz;
        const float3 b=vertices[indices[3u*t+1u]].position.xyz;
        const float3 c=vertices[indices[3u*t+2u]].position.xyz;
        const float3 u=b-a,v=c-a;
        const float3 area=cross(u,v);
        const bool bad=!all(isfinite(area));
        const bool collapsed=!bad&&nm_human_exact_binary32_triangle_is_zero(a,b,c);
        z+=uint(collapsed);n+=uint(bad);
        if(bad||collapsed)f=min(f,t);
    }
    zero[lane]=z;nonfinite[lane]=n;first[lane]=f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=128;stride;stride>>=1) {
        if(lane<stride) {
            zero[lane]+=zero[lane+stride];nonfinite[lane]+=nonfinite[lane+stride];
            first[lane]=min(first[lane],first[lane+stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(!lane)partials[group.x]=uint4(zero[0],nonfinite[0],first[0],0);
}
kernel void nm_human_resting_reduce_mesh_audit(
    constant uint4& d [[buffer(0)]],device const uint4* partials [[buffer(1)]],
    device const uint* indices [[buffer(2)]],device const MRVisualVertexGPUV2* vertices [[buffer(3)]],
    device uint4* result [[buffer(4)]],device MRHumanRestingSurfaceFailureGPU* failure [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
    if(i)return;
    uint4 sum=uint4(0,0,0xffffffffu,d.x);
    for(uint j=0;j<d.y;++j) {sum.xy+=partials[j].xy;sum.z=min(sum.z,partials[j].z);}
    result[0]=sum;
    MRHumanRestingSurfaceFailureGPU witness;
    witness.surfaceTriangleKind=uint4(0xffffffffu,0,0,0);
    witness.vertexIndices=uint4(0);
    for(uint k=0;k<3;++k)witness.renderedPositions[k]=float4(0);
    if(sum.z<d.x) {
        const uint3 ids=uint3(indices[3u*sum.z],indices[3u*sum.z+1u],indices[3u*sum.z+2u]);
        const float3 a=vertices[ids.x].position.xyz,b=vertices[ids.y].position.xyz,c=vertices[ids.z].position.xyz;
        const float3 u=b-a,v=c-a;
        const float3 area=cross(u,v);
        const bool bad=!all(isfinite(area));
        const uint kind=bad?1u:
            (nm_human_exact_binary32_triangle_is_zero(a,b,c)?2u:
                0u);
        witness.surfaceTriangleKind=uint4(0,sum.z,kind,0);
        witness.vertexIndices=uint4(ids,0);
        witness.renderedPositions[0]=float4(a,0);witness.renderedPositions[1]=float4(b,0);
        witness.renderedPositions[2]=float4(c,0);
    }
    failure[0]=witness;
}
kernel void nm_human_resting_audit_skin(
    constant uint4& d [[buffer(0)]], device const MRHumanRestingVertexMap* map [[buffer(1)]],
    device const MRVisualVertexGPUV2* vertices [[buffer(2)]],device float4* result [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float minimum[256];threadgroup uint owner[256],below[256],invalid[256];
    float value=INFINITY;uint index=0xffffffffu,count=0,nonfinite=0;
    for(uint i=lane;i<d.x;i+=256) {
        if(!all(isfinite(vertices[i].position))||!all(isfinite(vertices[i].normalAndTangentSign)))++nonfinite;
        if(map[i].deformationKind!=3)continue;
        const float z=vertices[i].position.z;
        if(z<value){value=z;index=i;}if(z<-.001f)++count;
    }
    minimum[lane]=value;owner[lane]=index;below[lane]=count;invalid[lane]=nonfinite;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=128;stride;stride>>=1) {
        if(lane<stride) {
            if(minimum[lane+stride]<minimum[lane]){minimum[lane]=minimum[lane+stride];owner[lane]=owner[lane+stride];}
            below[lane]+=below[lane+stride];invalid[lane]+=invalid[lane+stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(lane==0)result[d.w]=float4(minimum[0],float(owner[0]),float(below[0]),float(invalid[0]));
}
inline uint nmHumanRestingSkinAuditLanePriority(uint lane) {
    uint x=lane&255u;
    x=((x&0x55u)<<1)|((x>>1)&0x55u);
    x=((x&0x33u)<<2)|((x>>2)&0x33u);
    x=((x&0x0fu)<<4)|((x>>4)&0x0fu);
    return x;
}
inline bool nmHumanRestingSkinAuditOwnerBefore(uint candidate,uint current) {
    if(candidate==0xffffffffu)return false;
    if(current==0xffffffffu)return true;
    const uint candidatePriority=nmHumanRestingSkinAuditLanePriority(candidate);
    const uint currentPriority=nmHumanRestingSkinAuditLanePriority(current);
    return candidatePriority<currentPriority||
        (candidatePriority==currentPriority&&candidate/256u<current/256u);
}
kernel void nm_human_resting_audit_skin_partials(
    constant uint4& d [[buffer(0)]],device const MRHumanRestingVertexMap* map [[buffer(1)]],
    device const MRVisualVertexGPUV2* vertices [[buffer(2)]],device uint4* partials [[buffer(3)]],
    constant MRHumanRestingBedGPU& bed [[buffer(4), function_constant(kUseRestingContouredBed)]],
    device const float* bedHeights [[buffer(5), function_constant(kUseRestingContouredBed)]],
    uint lane [[thread_index_in_threadgroup]],uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float minimum[256];threadgroup uint owner[256],below[256],invalid[256];
    float value=INFINITY;uint index=0xffffffffu,count=0,nonfinite=0;
    const uint i=group.x*256u+lane;
    if(i<d.x) {
        if(!all(isfinite(vertices[i].position))||!all(isfinite(vertices[i].normalAndTangentSign)))++nonfinite;
        if(map[i].deformationKind==3u) {
            float z=vertices[i].position.z;
            if(kUseRestingContouredBed) {
                const auto hit=nmHumanRestingBedQuery(bed,bedHeights,vertices[i].position.xyz);
                if(!hit.valid)++nonfinite;
                z=hit.signedGap;
            }
            if(z<value){value=z;index=i;}
            if(z<-.001f)++count;
        }
    }
    minimum[lane]=value;owner[lane]=index;below[lane]=count;invalid[lane]=nonfinite;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=128;stride;stride>>=1) {
        if(lane<stride) {
            const uint other=lane+stride;
            if(minimum[other]<minimum[lane]||
               (minimum[other]==minimum[lane]&&
                nmHumanRestingSkinAuditOwnerBefore(owner[other],owner[lane]))) {
                minimum[lane]=minimum[other];owner[lane]=owner[other];
            }
            below[lane]+=below[other];invalid[lane]+=invalid[other];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(lane==0)partials[group.x]=uint4(as_type<uint>(minimum[0]),owner[0],below[0],invalid[0]);
}
kernel void nm_human_resting_reduce_skin_audit(
    constant uint4& d [[buffer(0)]],device const uint4* partials [[buffer(1)]],
    device float4* result [[buffer(2)]],uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float minimum[256];threadgroup uint owner[256],below[256],invalid[256];
    float value=INFINITY;uint index=0xffffffffu,count=0,nonfinite=0;
    for(uint group=lane;group<d.x;group+=256u) {
        const uint4 partial=partials[group];const float candidate=as_type<float>(partial.x);
        if(candidate<value||
           (candidate==value&&nmHumanRestingSkinAuditOwnerBefore(partial.y,index))) {
            value=candidate;index=partial.y;
        }
        count+=partial.z;nonfinite+=partial.w;
    }
    minimum[lane]=value;owner[lane]=index;below[lane]=count;invalid[lane]=nonfinite;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=128;stride;stride>>=1) {
        if(lane<stride) {
            const uint other=lane+stride;
            if(minimum[other]<minimum[lane]||
               (minimum[other]==minimum[lane]&&
                nmHumanRestingSkinAuditOwnerBefore(owner[other],owner[lane]))) {
                minimum[lane]=minimum[other];owner[lane]=owner[other];
            }
            below[lane]+=below[other];invalid[lane]+=invalid[other];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(lane==0)result[d.y]=float4(minimum[0],float(owner[0]),float(below[0]),float(invalid[0]));
}
kernel void nm_human_resting_audit_body(
    constant uint4& d [[buffer(0)]],device const MRBodyStateGPU* bodies [[buffer(1)]],
    device float4* result [[buffer(2)]],uint i [[thread_position_in_grid]]) {
    if(i)return;
    float4 sum=0,compensation=0;
    for(uint b=0;b<d.x;++b) {
        const auto body=bodies[b];
        const float inverseMass=body.linearVelocityAndInverseMass.w;
        if(inverseMass==0.0f)continue;
        if(!isfinite(inverseMass)||inverseMass<0||!all(isfinite(body.position.xyz))) {
            result[d.y]=float4(NAN);return;
        }
        const float mass=1.0f/inverseMass;
        const float4 term=float4(body.position.xyz*mass,mass)-compensation;
        const float4 next=sum+term;compensation=(next-sum)-term;sum=next;
    }
    result[d.y]=float4(sum.xyz/sum.w,sum.w);
}
kernel void nm_human_resting_prepare_world(
    constant uint4& d [[buffer(0)]], device const MRNumiHumanStandStatusGPU* body [[buffer(1)]],
    device MRMetalWorldStatusGPU* world [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if(i)return;
    MRMetalWorldStatusGPU s{};
    s.code=body[0].code==MR_NUMI_HUMAN_STAND_SUCCESS?MR_STEP_SUCCESS:MR_STEP_DID_NOT_CONVERGE;
    world[0]=s;(void)d;
}
kernel void nm_human_resting_validate_body(
    constant uint4& d [[buffer(0)]], device const MRNumiHumanStandStatusGPU* body [[buffer(1)]],
    device MRMetalWorldStatusGPU* world [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if(i)return;
    if(body[0].code!=MR_NUMI_HUMAN_STAND_SUCCESS||body[0].completedSteps!=d.x+1)
        world[0].code=MR_STEP_DID_NOT_CONVERGE;
}
kernel void nm_human_resting_validate_matter(
    constant uint4& d [[buffer(0)]], device MRNumiHumanStandStatusGPU* body [[buffer(1)]],
    device const MRMetalWorldStatusGPU* world [[buffer(2)]],
    device const NMMatterStatusGPU* matter [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if(i)return;
    if(body[0].code==MR_NUMI_HUMAN_STAND_SUCCESS&&
       (world[0].code!=MR_STEP_SUCCESS||matter[0].code!=NM_STATUS_SUCCESS)) {
        body[0].code=MR_NUMI_HUMAN_STAND_EXTERNAL_PHYSICS_FAILED;body[0].failingIndex=d.x;
    }
}
kernel void nm_human_resting_latch_common_failure(
    constant uint4& d [[buffer(0)]],
    device const MRNumiHumanStandStatusGPU* stand [[buffer(1)]],
    device const NMMatterStatusGPU* matter [[buffer(2)]],
    device const NMHumanRespirationState* respirationCandidate [[buffer(3)]],
    device const NMHumanRespirationState* respirationAccepted [[buffer(4)]],
    device const MRHumanRestingCommonCoordinatesGPU* commonCoordinates [[buffer(5)]],
    device const NBNumiRespiratoryChemoreflexInputV1* brainInput [[buffer(6)]],
    device const NBNumiRespiratoryChemoreflexStateV1* brainAccepted [[buffer(7)]],
    device const NBNumiRespiratoryChemoreflexStateV1* brainCandidate [[buffer(8)]],
    device const NBNumiRespiratoryChemoreflexOutputV1* brainOutput [[buffer(9)]],
    device const float4* excitation [[buffer(10)]],
    device MRHumanRestingCommonFailureGPU* latched [[buffer(11)]],
    constant MRHumanRestingCommonFieldGPU& commonParameters [[buffer(12)]],
    device const MRHumanRestingCommonCoordinateBoxGPU* commonBoxes [[buffer(13)]],
    uint lane [[thread_position_in_grid]]) {
    if(lane!=0u||latched[0].identity.w!=0u)return;
    const auto s=stand[0];const auto m=matter[0];const auto r=respirationCandidate[0];
    const auto c=commonCoordinates[0];
    if(s.code==MR_NUMI_HUMAN_STAND_SUCCESS&&m.code==NM_STATUS_SUCCESS&&r.status.w==0u&&c.status.x==0u)return;
    const auto bi=brainInput[0];const auto ba=brainAccepted[0];
    const auto bc=brainCandidate[0];const auto bo=brainOutput[0];
    device MRHumanRestingCommonFailureGPU& out=latched[0];
    out.identity=uint4(d.x,m.code,s.code,1u);
    out.matterStatus=uint4(m.environment,m.objectIndex,m.failingIndex,m.completedMicrosteps);
    out.matterDiagnostics=m.diagnostics;
    out.standStatus=uint4(s.completedSteps,s.failingIndex,s.contactIterations,s.flags);
    out.respirationCandidateStatus=r.status;
    out.respirationAcceptedStatus=respirationAccepted[0].status;
    out.chamberVolumes=r.chamberVolumes;
    out.respirationControl=r.control;
    float targetVolumes[7]={r.chamberVolumes.x,r.chamberVolumes.y,r.chamberVolumes.z,
        r.chamberVolumes.w,commonParameters.materialTargetVolumes.x,
        commonParameters.materialTargetVolumes.y,commonParameters.materialTargetVolumes.z};
    float attempted[7]={0,0,0,0,0,0,0},attemptResidual=INFINITY;
    uint attemptIterations=0u,attemptBox=0xffffffffu;
    const uint attemptStatus=nmHumanRestingCommonSolve(commonParameters,commonBoxes,
        targetVolumes,attempted,attemptIterations,attemptBox,attemptResidual);
    out.commonFirst=float4(attempted[0],attempted[1],attempted[2],attempted[3]);
    out.commonSecond=float4(attempted[4],attempted[5],attempted[6],0.0f);
    out.commonStatus=uint4(attemptStatus,attemptIterations,attemptBox,0u);
    out.commonDiagnostics=float4(attemptResidual,0.0f,0.0f,0.0f);
    out.brainInputMetadata=uint4(bi.validityMask,bi.flags,uint(bi.sourceTimestampMicroseconds),uint(bi.targetTimestampMicroseconds));
    out.brainInputRoots=uint4(uint(bi.sourceAcceptedRootFingerprint),uint(bi.sourceAcceptedRootFingerprint>>32),
        uint(bi.targetTransactionFingerprint),uint(bi.targetTransactionFingerprint>>32));
    out.brainInputTimestampHighs=uint4(uint(bi.sourceTimestampMicroseconds>>32),uint(bi.targetTimestampMicroseconds>>32),0u,0u);
    out.brainAcceptedMetadata=uint4(ba.flags,uint(ba.timestampMicroseconds),uint(ba.timestampMicroseconds>>32),0u);
    out.brainAcceptedRoots=uint4(uint(ba.acceptedRootFingerprint),uint(ba.acceptedRootFingerprint>>32),0u,0u);
    out.brainCandidateMetadata=uint4(bc.flags,uint(bc.timestampMicroseconds),uint(bc.timestampMicroseconds>>32),0u);
    out.brainCandidateRoot=uint4(uint(bc.acceptedRootFingerprint),uint(bc.acceptedRootFingerprint>>32),0u,0u);
    out.brainOutputMetadata=uint4(bo.flags,uint(bo.targetTimestampMicroseconds),uint(bo.targetTimestampMicroseconds>>32),0u);
    out.brainOutputRoots=uint4(uint(bo.sourceAcceptedRootFingerprint),uint(bo.sourceAcceptedRootFingerprint>>32),
        uint(bo.targetTransactionFingerprint),uint(bo.targetTransactionFingerprint>>32));
    out.brainOutputSetpoints=float4(bo.targetMinuteVentilationLitresPerMinute,bo.targetFrequencyBreathsPerMinute,
        bo.targetTidalVolumeLitres,bo.diaphragmExcitation);
    out.brainOutputExcitations=float4(bo.intercostalExcitation,0.0f,0.0f,0.0f);
    out.excitationBuffer=excitation[0];
}
kernel void nm_human_resting_capture(
    constant uint4& d [[buffer(0)]], device const MRArticulatedBodyPoseGPU* poses [[buffer(1)]],
    device const MRBodyStateGPU* properties [[buffer(2)]], device MRBodyStateGPU* bodies [[buffer(3)]],
    device const NMHumanRespirationState* respiration [[buffer(4)]],
    device NMHumanRespirationState* frame [[buffer(5)]],
    device const MRNumiHumanStandStatusGPU* statuses [[buffer(6)]],uint i [[thread_position_in_grid]]) {
    if(i>=d.y||statuses[0].code!=MR_NUMI_HUMAN_STAND_SUCCESS||respiration[0].status.x!=d.x)return;
    auto b=properties[i];b.position=poses[i].position;
    b.orientation=normalize(poses[i].orientation);bodies[i]=b;
    if(i==0)frame[0]=respiration[0];
}
kernel void nm_human_resting_present_commit(
    constant uint4& d [[buffer(0)]], device const MRBodyStateGPU* candidateBodies [[buffer(1)]],
    device const NMHumanRespirationState* candidateRespiration [[buffer(2)]],
    device MRBodyStateGPU* bodies [[buffer(3)]], device NMHumanRespirationState* respiration [[buffer(4)]],
    device const MRNumiHumanStandStatusGPU* statuses [[buffer(5)]],
    device MRHumanRestingCommonCoordinatesGPU* frameCommonCoordinates [[buffer(6)]],
    device MRHumanRestingCommonCoordinatesGPU* commonCoordinates [[buffer(7)]],
    device const MRHumanRestingCommonCoordinatesGPU* candidateCommonCoordinates [[buffer(8)]],
    device MRHumanRestingCommonCoordinatesGPU* acceptedCommonCoordinates [[buffer(9)]],
    device const NMMatterStatusGPU* matterStatuses [[buffer(10)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=d.y||statuses[0].code!=MR_NUMI_HUMAN_STAND_SUCCESS||
       statuses[0].completedSteps!=d.x+1||candidateRespiration[0].status.x!=d.x||
       (d.z&&matterStatuses[0].code!=NM_STATUS_SUCCESS))return;
    bodies[i]=candidateBodies[i];
    if(i==0) {
        respiration[0]=candidateRespiration[0];
        if(d.z) {
            frameCommonCoordinates[0]=acceptedCommonCoordinates[0];
            commonCoordinates[0]=frameCommonCoordinates[0];
            acceptedCommonCoordinates[0]=candidateCommonCoordinates[0];
        } else {
            commonCoordinates[0]=candidateCommonCoordinates[0];
        }
    }
}
kernel void nm_human_resting_terminal_present_commit(
    constant uint4& d [[buffer(0)]], device const MRBodyStateGPU* candidateBodies [[buffer(1)]],
    device const NMHumanRespirationState* candidateRespiration [[buffer(2)]],
    device MRBodyStateGPU* bodies [[buffer(3)]], device NMHumanRespirationState* respiration [[buffer(4)]],
    device const MRNumiHumanStandStatusGPU* statuses [[buffer(5)]],
    device MRHumanRestingCommonCoordinatesGPU* frameCommonCoordinates [[buffer(6)]],
    device MRHumanRestingCommonCoordinatesGPU* commonCoordinates [[buffer(7)]],
    device const MRHumanRestingCommonCoordinatesGPU* acceptedCommonCoordinates [[buffer(8)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=d.y||statuses[0].code!=MR_NUMI_HUMAN_STAND_SUCCESS||
       statuses[0].completedSteps!=d.x||candidateRespiration[0].status.x!=d.x||
       candidateRespiration[0].status.w!=0u)return;
    bodies[i]=candidateBodies[i];
    if(i==0u) {
        respiration[0]=candidateRespiration[0];
        if(d.z) {
            frameCommonCoordinates[0]=acceptedCommonCoordinates[0];
            commonCoordinates[0]=acceptedCommonCoordinates[0];
        }
    }
}
kernel void nm_human_respiration_brain_observe(
    constant NMHumanRespirationBrainDispatch& d [[buffer(0)]],
    device const NMHumanRespirationState* accepted [[buffer(1)]],
    device NBNumiRespiratoryChemoreflexInputV1* input [[buffer(2)]],
    uint env [[thread_position_in_grid]]) {
    if(env>=d.environmentCount)return;
    const auto s=accepted[env];
    NBNumiRespiratoryChemoreflexInputV1 sample{};
    sample.sourceTimestampMicroseconds=d.sourceTimeMicroseconds;
    sample.sourceAcceptedRootFingerprint=d.sourceRootIdentity;
    sample.targetTimestampMicroseconds=d.targetTimeMicroseconds;
    sample.targetTransactionFingerprint=d.targetRootIdentity;
    sample.paO2MillimetersMercury=s.observation.x;
    sample.paCO2MillimetersMercury=s.observation.y;
    sample.validityMask=NB_NUMI_RESPIRATORY_VALID_PA_O2|NB_NUMI_RESPIRATORY_VALID_PA_CO2;
    sample.flags=s.status.x==d.sourceStep&&!s.status.w?NB_NUMI_RESPIRATORY_INPUT_ACCEPTED:0;
    input[env]=sample;
}
kernel void nm_human_respiration_brain_deliver(
    constant NMHumanRespirationBrainDispatch& d [[buffer(0)]],
    device const NBNumiRespiratoryChemoreflexOutputV1* output [[buffer(1)]],
    device float4* excitation [[buffer(2)]],
    uint env [[thread_position_in_grid]]) {
    if(env>=d.environmentCount)return;
    auto o=output[env];
    if(!(o.flags&NB_NUMI_RESPIRATORY_RECORD_VALID)||o.targetTransactionFingerprint!=d.targetRootIdentity)
        excitation[env]=float4(NAN);
    else excitation[env]=float4(clamp(d.driveScale*float2(o.diaphragmExcitation,o.intercostalExcitation),0.0f,1.0f),
        o.targetFrequencyBreathsPerMinute,o.targetMinuteVentilationLitresPerMinute);
}
kernel void nm_human_respiration_brain_resolve(
    constant NMHumanRespirationBrainDispatch& d [[buffer(0)]],
    device const NBNumiRespiratoryChemoreflexStateV1* candidate [[buffer(1)]],
    device NBNumiRespiratoryChemoreflexStateV1* accepted [[buffer(2)]],
    device const NMMatterStatusGPU* statuses [[buffer(3)]],
    device const NMHumanRespirationState* respiration [[buffer(4)]],
    uint env [[thread_position_in_grid]]) {
    if(env<d.environmentCount&&statuses[env].code==NM_STATUS_SUCCESS&&!respiration[env].status.w&&
       (candidate[env].flags&NB_NUMI_RESPIRATORY_RECORD_VALID)&&
       candidate[env].acceptedRootFingerprint==d.targetRootIdentity)
        accepted[env]=candidate[env];
}
namespace human_respiration {
inline void addGas(thread float4& amount, float2 delta) {
    float2 y = delta - amount.zw;
    float2 next = amount.xy + y;
    amount.zw = (next - amount.xy) - y;
    amount.xy = next;
}
inline float saturation(float pressure, constant NMHumanRespirationParameters& p) {
    float ratio = pow(max(pressure, 0.0f) / p.oxygen.z, p.oxygen.w);
    return ratio / (1.0f + ratio);
}
inline float oxygenContent(float pressure, constant NMHumanRespirationParameters& p) {
    return p.oxygen.x * saturation(pressure, p) + p.oxygen.y * pressure;
}
inline float oxygenPressure(float content, constant NMHumanRespirationParameters& p) {
    float low = 0.0f, high = 760.0f;
    for (uint i=0; i<24; ++i) {
        float mid = (low + high) * 0.5f;
        if (oxygenContent(mid,p) < content) low=mid; else high=mid;
    }
    return (low+high)*0.5f;
}
inline float physical(device const float4* values,
                      device const NMVascularUnknownGPU* unknowns, uint base, uint row) {
    return values[base+row].x * unknowns[row].initialAndScaling.y;
}
}
struct NMRespiratoryMuscleCandidate {
    float4 state;
    float pressure;
    uint failure;
};
inline NMRespiratoryMuscleCandidate evaluateHumanRespiratoryMuscle(
    thread const NMHumanRespirationState& n,
    constant NMHumanRespirationParameters& p, const float dt, const uint m) {
    auto muscle=p.muscles[m];
    auto state=n.muscles[m];
    float excitation=n.control[m];
    NMRespiratoryMuscleCandidate result{state.excitationAndActivation,0.0f,0u};
    if (!isfinite(excitation) || excitation<0 || excitation>1) {
        result.failure=1u;return result;
    }
    float a=state.excitationAndActivation.y;
    float derivative=activationDerivative(muscle,excitation,a);
    float tau=derivative!=0 ? (excitation-a)/derivative : 1;
    float nextA=excitation-(excitation-a)*exp(-dt/tau);
    const float area=p.geometry[m+2];
    const float path=muscle.compliantArchitecture0.x+muscle.compliantArchitecture0.y-n.motion[m]/area;
    float fibre,velocity,tension,residual;
    if (!solveCompliantFiber(path,-n.motion[m+2]/area,dt,nextA,state,muscle,
                             fibre,velocity,tension,residual) || abs(residual)>1.e-3f) {
        result.failure=2u;return result;
    }
    result.state=float4(excitation,nextA,fibre,velocity);
    result.pressure=tension*forceScale(muscle.gainParameters,muscle.lengthRangeAndAcceleration.z)/area;
    return result;
}
inline void advanceHumanRespiratoryMechanics(
    thread NMHumanRespirationState& n,
    constant NMHumanRespirationParameters& p, const float dt, const bool reject,
    const uint lane=0u, const bool cooperative=false) {
    float2 pressure=0;
    if (cooperative) {
        NMRespiratoryMuscleCandidate pair{float4(0.0f),0.0f,0u};
        if (lane<2u) pair=evaluateHumanRespiratoryMuscle(n,p,dt,lane);
        for(uint m=0u;m<2u;++m) {
            const uint failure=simd_broadcast(pair.failure,m);
            const float4 state=simd_broadcast(pair.state,m);
            const float force=simd_broadcast(pair.pressure,m);
            if(failure) {n.status.w=failure;break;}
            n.muscles[m].excitationAndActivation=state;
            pressure[m]=force;
        }
    } else {
        for(uint m=0u;m<2u;++m) {
            const auto result=evaluateHumanRespiratoryMuscle(n,p,dt,m);
            if(result.failure) {n.status.w=result.failure;break;}
            n.muscles[m].excitationAndActivation=result.state;
            pressure[m]=result.pressure;
        }
    }
    const float lungE=1.0f/p.lung.z;
    const float common=dt*(p.lung.w+dt*lungE);
    const float a=p.geometry.x+dt*p.chest.z+dt*dt/p.chest.x+common;
    const float b=common;
    const float c=p.geometry.y+dt*p.chest.w+dt*dt/p.chest.y+common;
    const float recoil=lungE*(n.motion.x+n.motion.y);
    const float2 rhs=p.geometry.xy*n.motion.zw+dt*(pressure-float2(n.motion.x/p.chest.x,n.motion.y/p.chest.y)-recoil);
    const float determinant=a*c-b*b;
    const float2 speed=float2(c*rhs.x-b*rhs.y,a*rhs.y-b*rhs.x)/determinant;
    n.motion.xy+=dt*speed;
    n.motion.zw=speed;
    const float flow=speed.x+speed.y;
    const float volume=p.lung.x+n.motion.x+n.motion.y;
    const float alveolarPressure=-p.lung.w*flow;
    n.mechanics=float4(volume,alveolarPressure,
        p.environment.x+alveolarPressure-lungE*(volume-p.lung.x),flow);
    if (!all(isfinite(n.motion)) || !all(isfinite(n.mechanics)) ||
        !(volume>p.lung.y) || volume>0.008f || determinant<=0 || reject) n.status.w=3;
}
kernel void nm_human_respiration_predict(
    constant NMHumanRespirationParameters& p [[buffer(0)]],
    constant NMHumanRespirationDispatch& d [[buffer(1)]],
    device const NMHumanRespirationState* accepted [[buffer(2)]],
    device NMHumanRespirationState* candidate [[buffer(3)]],
    device const float4* excitations [[buffer(4)]],
    uint env [[thread_position_in_grid]]) {
    if (env>=d.environmentCount) return;
    NMHumanRespirationState n=accepted[env];
    n.control=excitations[env];
    n.status.w=0;
    if(kUseRespiratorySubcycling) {
        if(!all(isfinite(n.control.xy)) || any(n.control.xy<0.0f) ||
           any(n.control.xy>1.0f)) n.status.w=1u;
        if(d.reject) n.status.w=3u;
    } else advanceHumanRespiratoryMechanics(n,p,p.environment.w,d.reject!=0u);
    candidate[env]=n;
}
kernel void nm_human_respiration_exchange(
    constant NMHumanRespirationParameters& p [[buffer(0)]],
    constant NMHumanRespirationDispatch& d [[buffer(1)]],
    device const NMHumanRespirationState* accepted [[buffer(2)]],
    device NMHumanRespirationState* candidate [[buffer(3)]],
    device const float4* vascularBefore [[buffer(4)]],
    device const float4* vascularAfter [[buffer(5)]],
    device const NMVascularUnknownGPU* unknowns [[buffer(6)]],
    device const NMVascularConnectionGPU* connections [[buffer(7)]],
    device NMMatterStatusGPU* statuses [[buffer(8)]],
    device const NMVascularCompartmentGPU* compartments [[buffer(9)]],
    device const float* elastance [[buffer(10)]],
    uint globalIndex [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdWidth [[threads_per_simdgroup]]) {
    const uint env=kUseParallelRespiratoryMuscles?globalIndex/simdWidth:globalIndex;
    const bool publish=!kUseParallelRespiratoryMuscles||lane==0u;
    if(env>=d.environmentCount) return;
    NMHumanRespirationState n=candidate[env];
    if(statuses[env].code!=NM_STATUS_SUCCESS) return;
    const NMHumanRespirationState old=accepted[env];
    const float dt=p.environment.w;
    const uint base=env*d.vascularStride;
    uint gasSubsteps=1u;
    float outgoing[21];
    for(uint row=0;row<21;++row) outgoing[row]=0.0f;
    for(uint edge=0;edge<p.topology.w;++edge) {
        const auto con=connections[edge];
        const float flow=human_respiration::physical(vascularAfter,unknowns,base,21+edge);
        const uint from=flow>=0?con.identity.y:con.identity.z;
        outgoing[from]+=abs(flow);
    }
    if(kUseGasTransportSubcycling) {
        float required=kUseRespiratorySubcycling?max(1.0f,ceil(dt/0.002f)):1.0f;
        for(uint row=0;row<21;++row) {
            const float before=human_respiration::physical(vascularBefore,unknowns,base,row);
            const float after=human_respiration::physical(vascularAfter,unknowns,base,row);
            const float minimumVolume=min(before,after);
            if(!(minimumVolume>0.0f)||!isfinite(outgoing[row])) {
                n.status.w=4u;
                if(publish) {statuses[env].code=NM_STATUS_MULTIPHYSICS_FAILURE;candidate[env]=n;}
                return;
            }
            required=max(required,ceil(dt*outgoing[row]/(0.1f*minimumVolume)));
        }
        const float airwayRequired=ceil(dt*abs(n.mechanics.w)/(0.1f*p.lung.y));
        if(!isfinite(required)||required>32.0f||
           !isfinite(airwayRequired)||airwayRequired>32.0f) {
            n.status.w=(!isfinite(required)||required>32.0f)?4u:6u;
            if(publish) {statuses[env].code=NM_STATUS_MULTIPHYSICS_FAILURE;candidate[env]=n;}
            return;
        }
        gasSubsteps=uint(max(required,airwayRequired));
    }
    const float gasDt=dt/float(gasSubsteps);
    float4 compartmentGas=float4(0.0f);
    if(kUseParallelRespiratoryGas&&lane<21u) compartmentGas=n.bloodGas[lane];
    float pulmonaryO2=0;
    float totalPulmonaryO2=0;
    float netSweptVolume=0.0f, inspiredVolume=0.0f, maxDonorFraction=0.0f;
    float maxAirwayTransport=0.0f;
    for(uint substep=0u;substep<gasSubsteps;++substep) {
        const float alpha=float(substep)/float(gasSubsteps);
        const float mechanicsStartVolume=n.mechanics.x;
        if(kUseRespiratorySubcycling) {
            if(n.status.w) break;
            advanceHumanRespiratoryMechanics(n,p,gasDt,false,lane,kUseParallelRespiratoryMuscles);
            if(n.status.w) break;
            n.breath.x=max(n.breath.x,n.mechanics.x);
            n.breath.y=min(n.breath.y,n.mechanics.x);
        }
        netSweptVolume+=gasDt*n.mechanics.w;
        inspiredVolume+=gasDt*max(0.0f,n.mechanics.w);
        float2 delta[21];
        float2 compartmentDelta=0.0f;
        if(!kUseParallelRespiratoryGas)
            for(uint row=0;row<21;++row) delta[row]=0;
        for(uint row=0;row<21;++row) {
            const float before=human_respiration::physical(vascularBefore,unknowns,base,row);
            const float donorVolume=gasSubsteps==1u?before:mix(before,
                human_respiration::physical(vascularAfter,unknowns,base,row),alpha);
            maxDonorFraction=max(maxDonorFraction,gasDt*outgoing[row]/donorVolume);
            if(kUseGasTransportSubcycling&&
               (!(donorVolume>0.0f)||gasDt*outgoing[row]>0.1f*donorVolume))
                n.status.w=4u;
        }
        if(kUseGasTransportSubcycling&&n.status.w) break;
        float2 alveolarDelta=0;
        const float lungVolume=kUseRespiratorySubcycling?mechanicsStartVolume:
            (gasSubsteps==1u?old.mechanics.x:mix(old.mechanics.x,n.mechanics.x,alpha));
        const float alveolarVolume=lungVolume-p.lung.y;
        const float2 alveolarFraction=n.alveolarGas.xy/(alveolarVolume*p.environment.z);
        const float2 deadFraction=n.deadSpaceGas.xy/(p.lung.y*p.environment.z);
        const float2 alveolarPressure=alveolarFraction*(p.environment.y/133.322387415f);
        const float2 capillaryContent=float2(human_respiration::oxygenContent(alveolarPressure.x,p),
            p.carbonDioxide.x+p.carbonDioxide.y*(alveolarPressure.y-40.0f));
        for(uint edge=0;edge<p.topology.w;++edge) {
            const auto con=connections[edge];
            const float flow=human_respiration::physical(vascularAfter,unknowns,base,21+edge);
            const uint from=flow>=0?con.identity.y:con.identity.z;
            const uint to=flow>=0?con.identity.z:con.identity.y;
            const float before=human_respiration::physical(vascularBefore,unknowns,base,from);
            const float fromVolume=gasSubsteps==1u?before:mix(before,
                human_respiration::physical(vascularAfter,unknowns,base,from),alpha);
            const float2 donorGas=kUseParallelRespiratoryGas
                ?simd_shuffle(compartmentGas.xy,from):n.bloodGas[from].xy;
            const float2 content=donorGas/fromVolume;
            const float2 transported=gasDt*abs(flow)*content;
            if(kUseParallelRespiratoryGas) {
                if(lane==from) compartmentDelta-=transported;
                if(lane==to) compartmentDelta+=transported;
            } else {delta[from]-=transported;delta[to]+=transported;}
            if(edge==p.topology.x && flow>0) {
                const float2 exchange=gasDt*flow*p.carbonDioxide.z*(capillaryContent-content);
                if(kUseParallelRespiratoryGas) {
                    if(lane==to) compartmentDelta+=exchange;
                } else delta[to]+=exchange;
                alveolarDelta-=exchange;
                if(gasSubsteps==1u) pulmonaryO2=exchange.x/dt;
                else totalPulmonaryO2+=exchange.x;
            }
            if(gasDt*abs(flow)>0.1f*fromVolume || !all(isfinite(transported))) n.status.w=4;
        }
        for(uint bed=0;bed<4;++bed) {
            const float2 demand=gasDt*p.metabolism.zw*p.tissueFractions[bed];
            if(kUseParallelRespiratoryGas) {
                if(lane==p.tissueRows[bed]) compartmentDelta+=float2(-demand.x,demand.y);
            } else delta[p.tissueRows[bed]]+=float2(-demand.x,demand.y);
        }
        human_respiration::addGas(n.metabolicGas,gasDt*p.metabolism.zw);
        const float subTransport=gasDt*n.mechanics.w*p.environment.z;
        maxAirwayTransport=max(maxAirwayTransport,abs(subTransport));
        const float2 outsideFlux=subTransport*(subTransport>=0?p.metabolism.xy:deadFraction);
        const float2 airwayFlux=subTransport*(subTransport>=0?deadFraction:alveolarFraction);
        human_respiration::addGas(n.environmentGas,outsideFlux);
        human_respiration::addGas(n.deadSpaceGas,outsideFlux-airwayFlux);
        human_respiration::addGas(n.alveolarGas,airwayFlux+alveolarDelta);
        if(kUseParallelRespiratoryGas) {
            bool invalid=false;
            if(lane<21u) {
                human_respiration::addGas(compartmentGas,compartmentDelta);
                invalid=!all(isfinite(compartmentGas))||any(compartmentGas.xy<0);
            }
            if(simd_any(invalid)) n.status.w=5;
        } else {
            for(uint row=0;row<21;++row) {
                human_respiration::addGas(n.bloodGas[row],delta[row]);
                if(!all(isfinite(n.bloodGas[row]))||any(n.bloodGas[row].xy<0)) n.status.w=5;
            }
        }
        if(!all(isfinite(n.alveolarGas))||any(n.alveolarGas.xy<0)||
           any(n.deadSpaceGas.xy<0)||
           abs(subTransport)>0.1f*p.lung.y*p.environment.z) n.status.w=6;
        if(n.status.w) break;
    }
    if(kUseParallelRespiratoryGas)
        for(uint row=0u;row<21u;++row)
            n.bloodGas[row]=simd_broadcast(compartmentGas,row);
    if(gasSubsteps>1u) pulmonaryO2=totalPulmonaryO2/dt;
    n.transportStep=float4(netSweptVolume,
        (n.mechanics.x-old.mechanics.x)-netSweptVolume,
        maxDonorFraction,float(gasSubsteps));
    const uint sensed=p.topology.y;
    const float volume=human_respiration::physical(vascularAfter,unknowns,base,sensed);
    const float2 content=n.bloodGas[sensed].xy/volume;
    const float po2=human_respiration::oxygenPressure(content.x,p);
    const float pco2=40+(content.y-p.carbonDioxide.x)/p.carbonDioxide.y;
    n.observation=float4(po2,pco2,human_respiration::saturation(po2,p),pulmonaryO2);
    const uint4 pressureRows=uint4(20,16,0,17);
    for(uint i=0;i<4;++i) {
        uint row=pressureRows[i];auto c=compartments[row];
        const float v=human_respiration::physical(vascularAfter,unknowns,base,row);
        n.cardiacPressure[i]=c.compliance.w+c.compliance.y+
            (v-c.compliance.x)*elastance[env*21+row];
    }
    float bloodVolume=0;
    for(uint row=0;row<21;++row) bloodVolume+=human_respiration::physical(vascularAfter,unknowns,base,row);
    n.circulation=float4(bloodVolume,
        human_respiration::physical(vascularAfter,unknowns,base,20),
        human_respiration::physical(vascularAfter,unknowns,base,16),
        max(n.circulation.w,abs(bloodVolume-0.00515f)));
    float stepResidual=0,stepResidualCorrection=0;
    float stepPhysicalDelta=0,stepPhysicalDeltaCorrection=0;
    for(uint node=0;node<p.topology.z;++node) {
        const float oldNormalized=vascularBefore[base+node].x;
        const float newNormalized=vascularAfter[base+node].x;
        const float scale=unknowns[node].initialAndScaling.y;
        const float volumeTerm=scale*(newNormalized-oldNormalized);
        const float oldPhysical=scale*oldNormalized;
        const float newPhysical=scale*newNormalized;
        const float physicalDelta=newPhysical-oldPhysical;
        const float residualAdjusted=volumeTerm-stepResidualCorrection;
        const float residualNext=stepResidual+residualAdjusted;
        stepResidualCorrection=(residualNext-stepResidual)-residualAdjusted;
        stepResidual=residualNext;
        const float physicalAdjusted=physicalDelta-stepPhysicalDeltaCorrection;
        const float physicalNext=stepPhysicalDelta+physicalAdjusted;
        stepPhysicalDeltaCorrection=(physicalNext-stepPhysicalDelta)-physicalAdjusted;
        stepPhysicalDelta=physicalNext;
    }
    const float residualValue=stepResidual-stepResidualCorrection;
    const float residualAdjusted=residualValue-n.bloodBalance.continuityResidualCompensationM3;
    const float residualNext=n.bloodBalance.continuityResidualSumM3+residualAdjusted;
    n.bloodBalance.continuityResidualCompensationM3=
        (residualNext-n.bloodBalance.continuityResidualSumM3)-residualAdjusted;
    n.bloodBalance.continuityResidualSumM3=residualNext;
    const float physicalValue=stepPhysicalDelta-stepPhysicalDeltaCorrection;
    const float physicalAdjusted=physicalValue-n.bloodBalance.physicalVolumeDeltaCompensationM3;
    const float physicalNext=n.bloodBalance.physicalVolumeDeltaSumM3+physicalAdjusted;
    n.bloodBalance.physicalVolumeDeltaCompensationM3=
        (physicalNext-n.bloodBalance.physicalVolumeDeltaSumM3)-physicalAdjusted;
    n.bloodBalance.physicalVolumeDeltaSumM3=physicalNext;
    n.chamberVolumes=float4(
        human_respiration::physical(vascularAfter,unknowns,base,15),
        human_respiration::physical(vascularAfter,unknowns,base,16),
        human_respiration::physical(vascularAfter,unknowns,base,19),
        human_respiration::physical(vascularAfter,unknowns,base,20));
    const float aortic=max(0.0f,human_respiration::physical(vascularAfter,unknowns,base,21));
    const float pulmonary=max(0.0f,human_respiration::physical(vascularAfter,unknowns,base,41));
    const float mitral=max(0.0f,human_respiration::physical(vascularAfter,unknowns,base,44));
    n.cardiacFlow.xy+=dt*float2(aortic,pulmonary);
    n.cardiacFlow.z+=dt*aortic;
    if(mitral>1.e-9f)n.cardiacStatus.y=1;
    if(aortic<=1.e-9f && n.cardiacStatus.z && n.cardiacStatus.y) {
        ++n.cardiacStatus.x;n.cardiacStatus.y=0;
        n.cardiacFlow.w=n.cardiacFlow.z;n.cardiacFlow.z=0;
    }
    n.cardiacStatus.z=aortic>1.e-9f;
    float4 total=0;
    human_respiration::addGas(total,n.alveolarGas.xy);
    human_respiration::addGas(total,n.deadSpaceGas.xy);
    for(uint row=0;row<21;++row) human_respiration::addGas(total,n.bloodGas[row].xy);
    const float2 balance=total.xy-n.environmentGas.xy+
        float2(n.metabolicGas.x,-n.metabolicGas.y)-n.gasBudget.xy;
    n.gasBudget.zw=max(n.gasBudget.zw,abs(balance));
    if(any(abs(balance)>5.e-5f*n.gasBudget.xy)) n.status.w=7;
    n.breath.x=max(n.breath.x,n.mechanics.x);
    n.breath.y=min(n.breath.y,n.mechanics.x);
    const float inspiredAdjusted=(kUseRespiratorySubcycling?inspiredVolume:
        dt*max(0.0f,n.mechanics.w))-n.breathAccounting.y;
    const float inspiredNext=n.breath.w+inspiredAdjusted;
    n.breathAccounting.y=(inspiredNext-n.breath.w)-inspiredAdjusted;
    n.breath.w=inspiredNext;
    constexpr float breathFlowHysteresisM3PerS=1.e-9f;
    const uint inspiration=n.mechanics.w>breathFlowHysteresisM3PerS ? 1u :
        (n.mechanics.w < -breathFlowHysteresisM3PerS ? 0u : n.status.z);
    if(inspiration&&!n.status.z && n.status.x>0) {
        n.breath.z=n.breath.x-n.breath.y;
        n.breath.xy=float2(n.mechanics.x);
        n.breathTiming.y=n.breathTiming.x;
        n.breathTiming.x=n.status.x+1u;
        n.breathTiming.z=n.breathTiming.x-n.breathTiming.y;
        n.breathAccounting.z=(n.breath.w-n.breathAccounting.x)-
            (n.breathAccounting.y-n.breathAccounting.w);
        n.breathAccounting.x=n.breath.w;
        n.breathAccounting.w=n.breathAccounting.y;
        ++n.status.y;
    }
    n.status.z=inspiration;
    ++n.status.x;
    if(!all(isfinite(n.observation))||!all(isfinite(n.alveolarGas))||
       any(n.alveolarGas.xy<0)||any(n.deadSpaceGas.xy<0)||
       maxAirwayTransport>0.1f*p.lung.y*p.environment.z) n.status.w=6;
    if(publish) {
        if(n.status.w) statuses[env].code=NM_STATUS_MULTIPHYSICS_FAILURE;
        candidate[env]=n;
    }
}
kernel void nm_human_respiration_resolve(
    constant NMHumanRespirationDispatch& d [[buffer(0)]],
    device const NMHumanRespirationState* candidate [[buffer(1)]],
    device NMHumanRespirationState* accepted [[buffer(2)]],
    device const NMMatterStatusGPU* statuses [[buffer(3)]],
    uint env [[thread_position_in_grid]]) {
    if(env<d.environmentCount && statuses[env].code==NM_STATUS_SUCCESS && !candidate[env].status.w)
        accepted[env]=candidate[env];
}
kernel void nm_human_resting_capture_vertices(
    device const MRVisualVertexGPUV2* source [[buffer(0)]],
    device MRVisualVertexGPUV2* destination [[buffer(1)]],
    constant uint& vertexCount [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if(i<vertexCount)destination[i]=source[i];
}
