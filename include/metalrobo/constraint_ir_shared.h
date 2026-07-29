#pragma once

// Canonical pointer-free ConstraintIR ABI shared by C++, Objective-C++, and
// Metal.  Host-only containers and algorithms live in ConstraintIR.hpp; every
// record below is fixed-width, 16-byte aligned, and safe to bind directly to a
// Metal buffer.

#include "metalrobo/gpu_types.h"

#define MR_CONSTRAINT_IR_ABI_VERSION 2u
#define MR_CONSTRAINT_IR_INVALID_INDEX 0xffffffffu
#define MR_CONSTRAINT_IR_UNBOUNDED 3.402823466e+38f

#ifndef __METAL_VERSION__
#define MR_IR_DEFAULT(value) = value
#else
#define MR_IR_DEFAULT(value)
#endif

enum MRConstraintIRBlockFlags : mr_u32 {
    MR_CONSTRAINT_IR_BLOCK_NEW_IMPACT = 1u << 0u,
    MR_CONSTRAINT_IR_BLOCK_WARM_STARTED = 1u << 1u,
    MR_CONSTRAINT_IR_BLOCK_DISABLED = 1u << 2u,
    MR_CONSTRAINT_IR_BLOCK_ROD_ENDPOINT = 1u << 3u,
    // Runtime-only normalized mechanism block. Cooked ConstraintIR v2
    // records remain byte-identical; the world seeding kernel adds this bit
    // after expanding sparse authored records into fixed GPU slots.
    MR_CONSTRAINT_IR_BLOCK_GENERALIZED = 1u << 4u,
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
    // Deforming capsule endpoint. The immutable IR endpoint retains the
    // canonical anchor/axis representation while the runtime sidecar below
    // binds the two rod nodes, interpolation weights, and optional material
    // twist coordinate for applyJ/applyJT.
    MR_CONSTRAINT_IR_JACOBIAN_ROD_EDGE = 4u,
};

// GENERALIZED endpoints are sparse Jacobian terms. objectIndex is the global
// generalized-velocity index, articulationIndex owns that coordinate, axis.x
// is the signed coefficient, and the low bits of flags select the local row
// within the block. When Q_INDEX_VALID is set, linkIndex stores the matching
// global scalar configuration index; otherwise linkIndex is invalid. This
// convention represents limits, gears/mimics, tendons, drives, and bounded
// joint friction without a dense Jacobian or another ABI record.
enum MRConstraintIREndpointFlags : mr_u32 {
    MR_CONSTRAINT_IR_ENDPOINT_ROW_MASK = 0x000000ffu,
    MR_CONSTRAINT_IR_ENDPOINT_Q_INDEX_VALID = 1u << 8u,
};

typedef struct MR_ALIGN16 MRConstraintIRStableKeyGPU {
    mr_u32 words[4];
} MRConstraintIRStableKeyGPU;

typedef struct MR_ALIGN16 MRConstraintIRBlockGPU {
    MRConstraintIRStableKeyGPU key;

    mr_u32 type MR_IR_DEFAULT(0u);
    mr_u32 dimension MR_IR_DEFAULT(0u);
    mr_u32 flags MR_IR_DEFAULT(0u);
    mr_u32 islandIndex MR_IR_DEFAULT(0u);

    mr_u32 endpointOffset MR_IR_DEFAULT(0u);
    mr_u32 endpointCount MR_IR_DEFAULT(0u);
    mr_u32 rowOffset MR_IR_DEFAULT(0u);
    mr_u32 impulseOffset MR_IR_DEFAULT(0u);

    mr_u32 coneIndex MR_IR_DEFAULT(MR_CONSTRAINT_IR_INVALID_INDEX);
    mr_u32 eventSlot MR_IR_DEFAULT(MR_CONSTRAINT_IR_INVALID_INDEX);
    mr_u32 reserved0 MR_IR_DEFAULT(0u);
    mr_u32 reserved1 MR_IR_DEFAULT(0u);
} MRConstraintIRBlockGPU;

typedef struct MR_ALIGN16 MRConstraintIREndpointGPU {
    mr_u32 objectIndex MR_IR_DEFAULT(MR_CONSTRAINT_IR_INVALID_INDEX);
    mr_u32 articulationIndex MR_IR_DEFAULT(MR_CONSTRAINT_IR_INVALID_INDEX);
    mr_u32 linkIndex MR_IR_DEFAULT(MR_CONSTRAINT_IR_INVALID_INDEX);
    mr_u32 role MR_IR_DEFAULT(MR_CONSTRAINT_IR_ENDPOINT_WORLD);

    mr_u32 jacobianKind MR_IR_DEFAULT(
        MR_CONSTRAINT_IR_JACOBIAN_WORLD_POINT
    );
    mr_u32 flags MR_IR_DEFAULT(0u);
    mr_u32 reserved0 MR_IR_DEFAULT(0u);
    mr_u32 reserved1 MR_IR_DEFAULT(0u);

    mr_float4 anchor;
    mr_float4 axis;
} MRConstraintIREndpointGPU;

enum MRConstraintIREndpointOwnerKind : mr_u32 {
    MR_CONSTRAINT_IR_OWNER_WORLD = 0u,
    MR_CONSTRAINT_IR_OWNER_ARTICULATION = 1u,
    MR_CONSTRAINT_IR_OWNER_FREE_BODY = 2u,
    MR_CONSTRAINT_IR_OWNER_ROD_EDGE = 3u,
};

enum MRConstraintIREndpointRuntimeFlags : mr_u32 {
    MR_CONSTRAINT_IR_RUNTIME_DYNAMIC = 1u << 0u,
    MR_CONSTRAINT_IR_RUNTIME_HAS_POINT_QUERY = 1u << 1u,
    MR_CONSTRAINT_IR_RUNTIME_HAS_TWIST = 1u << 2u,
    MR_CONSTRAINT_IR_RUNTIME_KINEMATIC = 1u << 3u,
};

// Dynamic binding for one immutable ConstraintIR endpoint. Keeping this
// information in a separate fixed-layout record lets collision compilation
// bind articulation queries, maximal bodies, and deforming rod edges without
// changing ConstraintIR ABI v2 or embedding backend pointers in semantic IR.
typedef struct MR_ALIGN16 MRConstraintEndpointRuntimeGPU {
    mr_u32 dynamicNode MR_IR_DEFAULT(MR_CONSTRAINT_IR_INVALID_INDEX);
    mr_u32 ownerKind MR_IR_DEFAULT(MR_CONSTRAINT_IR_OWNER_WORLD);
    mr_u32 ownerIndex MR_IR_DEFAULT(MR_CONSTRAINT_IR_INVALID_INDEX);
    mr_u32 elementIndex MR_IR_DEFAULT(MR_CONSTRAINT_IR_INVALID_INDEX);

    mr_u32 queryIndex MR_IR_DEFAULT(MR_CONSTRAINT_IR_INVALID_INDEX);
    mr_u32 secondaryIndex MR_IR_DEFAULT(MR_CONSTRAINT_IR_INVALID_INDEX);
    mr_u32 twistIndex MR_IR_DEFAULT(MR_CONSTRAINT_IR_INVALID_INDEX);
    mr_u32 flags MR_IR_DEFAULT(0u);

    // Rod edges use x/y as the two nodal interpolation weights. Other
    // endpoint kinds leave this record zero.
    mr_float4 weights;
    // Rigid endpoints store a body-local anchor. Rod endpoints store the
    // cross-section radial direction in the transported material frame.
    mr_float4 localAnchorOrRadial;
} MRConstraintEndpointRuntimeGPU;

typedef struct MR_ALIGN16 MRConstraintIRRowGPU {
    mr_float4 direction;

    float positionError MR_IR_DEFAULT(0.0f);
    float targetVelocity MR_IR_DEFAULT(0.0f);
    float compliance MR_IR_DEFAULT(0.0f);
    float dissipation MR_IR_DEFAULT(0.0f);

    float timeConstant MR_IR_DEFAULT(0.01f);
    float dampingRatio MR_IR_DEFAULT(1.0f);
    float impulseLower MR_IR_DEFAULT(-MR_CONSTRAINT_IR_UNBOUNDED);
    float impulseUpper MR_IR_DEFAULT(MR_CONSTRAINT_IR_UNBOUNDED);

    mr_u32 flags MR_IR_DEFAULT(0u);
    mr_u32 reserved0 MR_IR_DEFAULT(0u);
    mr_u32 reserved1 MR_IR_DEFAULT(0u);
    mr_u32 reserved2 MR_IR_DEFAULT(0u);
} MRConstraintIRRowGPU;

typedef struct MR_ALIGN16 MRConstraintIRConeGPU {
    float staticFrictionU MR_IR_DEFAULT(0.0f);
    float staticFrictionV MR_IR_DEFAULT(0.0f);
    float dynamicFrictionU MR_IR_DEFAULT(0.0f);
    float dynamicFrictionV MR_IR_DEFAULT(0.0f);

    float rollingLength MR_IR_DEFAULT(0.0f);
    float torsionalLength MR_IR_DEFAULT(0.0f);
    float restitution MR_IR_DEFAULT(0.0f);
    float restitutionThreshold MR_IR_DEFAULT(0.0f);

    float adhesionImpulse MR_IR_DEFAULT(0.0f);
    float maximumNormalImpulse MR_IR_DEFAULT(0.0f);
    float stictionTransitionVelocity MR_IR_DEFAULT(1.0e-3f);
    float reserved MR_IR_DEFAULT(0.0f);
} MRConstraintIRConeGPU;

typedef struct MR_ALIGN16 MREvaluatedConstraintIRRowGPU {
    mr_float4 direction;

    float targetVelocity MR_IR_DEFAULT(0.0f);
    float regularization MR_IR_DEFAULT(0.0f);
    float impulseLower MR_IR_DEFAULT(-MR_CONSTRAINT_IR_UNBOUNDED);
    float impulseUpper MR_IR_DEFAULT(MR_CONSTRAINT_IR_UNBOUNDED);

    float sourcePositionError MR_IR_DEFAULT(0.0f);
    float stabilizationVelocity MR_IR_DEFAULT(0.0f);
    float sourceTargetVelocity MR_IR_DEFAULT(0.0f);
    float relativeVelocity MR_IR_DEFAULT(0.0f);

    float preSolveVelocity MR_IR_DEFAULT(0.0f);
    float reserved0 MR_IR_DEFAULT(0.0f);
    float reserved1 MR_IR_DEFAULT(0.0f);
    float reserved2 MR_IR_DEFAULT(0.0f);
} MREvaluatedConstraintIRRowGPU;

typedef struct MR_ALIGN16 MREvaluatedConstraintIRConeGPU {
    float effectiveFrictionU MR_IR_DEFAULT(0.0f);
    float effectiveFrictionV MR_IR_DEFAULT(0.0f);
    float staticFrictionU MR_IR_DEFAULT(0.0f);
    float staticFrictionV MR_IR_DEFAULT(0.0f);

    float dynamicFrictionU MR_IR_DEFAULT(0.0f);
    float dynamicFrictionV MR_IR_DEFAULT(0.0f);
    float rollingLength MR_IR_DEFAULT(0.0f);
    float torsionalLength MR_IR_DEFAULT(0.0f);

    float restitutionVelocity MR_IR_DEFAULT(0.0f);
    float restitutionThreshold MR_IR_DEFAULT(0.0f);
    float adhesionImpulse MR_IR_DEFAULT(0.0f);
    float maximumNormalImpulse MR_IR_DEFAULT(0.0f);
} MREvaluatedConstraintIRConeGPU;

#undef MR_IR_DEFAULT

#ifndef __METAL_VERSION__
static_assert(sizeof(MRConstraintIRStableKeyGPU) == 16);
static_assert(sizeof(MRConstraintIRBlockGPU) == 64);
static_assert(sizeof(MRConstraintIREndpointGPU) == 64);
static_assert(sizeof(MRConstraintEndpointRuntimeGPU) == 64);
static_assert(sizeof(MRConstraintIRRowGPU) == 64);
static_assert(sizeof(MRConstraintIRConeGPU) == 48);
static_assert(sizeof(MREvaluatedConstraintIRRowGPU) == 64);
static_assert(sizeof(MREvaluatedConstraintIRConeGPU) == 48);
#endif
