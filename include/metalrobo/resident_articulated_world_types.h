#pragma once

#include "metalrobo/engine_types.h"

// Fixed SIMD working-set bounds for the compact fixed-base operator.
#define MR_RESIDENT_DOF_CAPACITY 9u
#define MR_RESIDENT_BODY_CAPACITY 16u
#define MR_RESIDENT_FREE_CAPACITY 2u
#define MR_RESIDENT_SHAPE_CAPACITY 48u
#define MR_RESIDENT_PAIR_CAPACITY 64u
#define MR_RESIDENT_CONTACT_CAPACITY 31u
#define MR_RESIDENT_VELOCITY_CAPACITY \
    (MR_RESIDENT_DOF_CAPACITY + 6u * MR_RESIDENT_FREE_CAPACITY)

typedef struct MR_ALIGN16 MRResidentJointGPU {
    // Parent body, DoF, joint type, ancestor DoF mask.
    mr_uint4 topology;
    mr_float4 axis;
    mr_float4 parentAnchor;
    mr_float4 childAnchor;
    mr_float4 parentRotation;
    mr_float4 inverseChildRotation;
} MRResidentJointGPU;

typedef struct MR_ALIGN16 MRResidentBodyGPU {
    // Articulated, free slot or MR_INVALID_INDEX, scene slot or MR_INVALID_INDEX, motion type.
    mr_uint4 topology;
    // Mass, inverse mass, linear damping, angular damping.
    mr_float4 massAndDamping;
    mr_float4 inertia[3];
    mr_float4 inverseInertia[3];
    mr_float4 initialPosition;
    mr_float4 initialRotation;
    mr_float4 initialLinear;
    mr_float4 initialAngular;
} MRResidentBodyGPU;

typedef struct MR_ALIGN16 MRResidentDofGPU {
    // Child body, subtree body mask, MR_DOF_FLAG_*, action index.
    mr_uint4 topology;
    mr_float4 drive;
    mr_float4 limits;
    mr_float4 motorAndSpeed;
    mr_float4 transmissionAndEnvelope;
    // Initial position, velocity, actuator profile active, reserved.
    mr_float4 initial;
    // Target scale, lower/upper position, filter response fraction.
    mr_float4 action;
} MRResidentDofGPU;

typedef struct MR_ALIGN16 MRResidentPairGPU {
    // Colliders A/B, analytic pair class, and dynamic node mask.
    mr_uint4 geometry;
    // Static friction, dynamic friction, restitution, restitution threshold.
    mr_float4 frictionAndRestitution;
    // Compliance, dissipation, rest separation, reserved.
    mr_float4 response;
} MRResidentPairGPU;

typedef struct MR_ALIGN16 MRResidentProgramGPU {
    // Body count, DoF count, pair count, number of free bodies.
    mr_uint4 sizes;
    // Control count, substeps, cone sweeps, reserved.
    mr_uint4 schedule;
    // xyz gravity, w substep seconds.
    mr_float4 gravityAndTimestep;
    // Slop and maximum depenetration velocity.
    mr_float4 stabilization;
    // Input action count and operational contact capacity.
    mr_uint4 controls;
    MRResidentBodyGPU bodies[MR_RESIDENT_BODY_CAPACITY];
    MRResidentJointGPU joints[MR_RESIDENT_BODY_CAPACITY];
    MRResidentDofGPU dofs[MR_RESIDENT_DOF_CAPACITY];
    MRShapeGPU shapes[MR_RESIDENT_SHAPE_CAPACITY];
    MRResidentPairGPU pairs[MR_RESIDENT_PAIR_CAPACITY];
    mr_uint4 freeBody;
} MRResidentProgramGPU;

typedef struct MR_ALIGN16 MRResidentContactSampleGPU {
    // Colliders A/B, owning body A/B.
    mr_uint4 pair;
    mr_float4 pointAndSeparation;
    // A to B normal, normal impulse over the final physics substep.
    mr_float4 normalAndImpulse;
    // World impulse on B including friction, kg m/s.
    mr_float4 impulse;
} MRResidentContactSampleGPU;

typedef struct MR_ALIGN16 MRResidentBodySampleGPU {
    mr_float4 position;
    mr_float4 rotation;
    mr_float4 linear;
    mr_float4 angular;
} MRResidentBodySampleGPU;

typedef struct MR_ALIGN16 MRResidentFrameGPU {
    // Step status, number of contacts, failing substep, accepted control.
    mr_uint4 result;
    float q[MR_RESIDENT_DOF_CAPACITY];
    float v[MR_RESIDENT_DOF_CAPACITY];
    float targets[MR_RESIDENT_DOF_CAPACITY];
    MRResidentBodySampleGPU freeBodies[MR_RESIDENT_FREE_CAPACITY];
    MRResidentContactSampleGPU contacts[MR_RESIDENT_CONTACT_CAPACITY];
} MRResidentFrameGPU;

typedef struct MR_ALIGN16 MRResidentWarmPointGPU {
    mr_uint4 key;
    mr_float4 position;
    mr_float4 impulse;
} MRResidentWarmPointGPU;

typedef struct MR_ALIGN16 MRResidentWarmStateGPU {
    mr_uint4 counts;
    MRResidentWarmPointGPU points[MR_RESIDENT_CONTACT_CAPACITY];
} MRResidentWarmStateGPU;

#define MR_RESIDENT_TASK_OBSERVATION_CAPACITY 64u
typedef struct MR_ALIGN16 MRResidentNormalImpulseGPU {
    mr_u32 bodyA;
    mr_u32 bodyB;
    float normalImpulse;
    mr_u32 reserved;
} MRResidentNormalImpulseGPU;

typedef struct MR_ALIGN16 MRResidentTaskFrameGPU {
    // Episode step after physics, last reset index plus one, reset flag, episode index.
    mr_uint4 task;
    float filtered[MR_RESIDENT_DOF_CAPACITY];
    // Step code, number of contacts, completed substeps, active pairs.
    mr_uint4 result;
    float q[MR_RESIDENT_DOF_CAPACITY];
    float v[MR_RESIDENT_DOF_CAPACITY];
    float effort[MR_RESIDENT_DOF_CAPACITY];
    float acceleration[MR_RESIDENT_DOF_CAPACITY];
    MRResidentBodySampleGPU freeBodies[MR_RESIDENT_FREE_CAPACITY];
    MRResidentNormalImpulseGPU contacts[MR_RESIDENT_CONTACT_CAPACITY];
} MRResidentTaskFrameGPU;

typedef struct MR_ALIGN16 MRResidentTaskObservationGPU {
    float actor[MR_RESIDENT_TASK_OBSERVATION_CAPACITY];
    float critic[MR_RESIDENT_TASK_OBSERVATION_CAPACITY];
} MRResidentTaskObservationGPU;
