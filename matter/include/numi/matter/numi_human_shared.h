#pragma once

#include "numi/matter/shared.h"

#define NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION 9u
#define NM_NUMI_HUMAN_ARTICULAR_CONTACT_AUDIT_MAX_STEPS 4096u

enum NMNumiHumanTendonFEMNodeLoadFlags : nm_u32 {
    NM_NUMI_HUMAN_TENDON_FEM_NODE_LOAD_ACTIVE = 1u << 0u,
};

enum NMNumiHumanTendonFEMNodeAnchorFlags : nm_u32 {
    NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE = 1u << 0u,
};

enum NMNumiHumanTendonFEMEndpointReplacementFlags : nm_u32 {
    NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_ACTIVE = 1u << 0u,
    // Replace the selected muscle's complete source generalized-force row,
    // then restore only loadEndpointIndex's body reaction. The continuum and
    // its solved anchors own every remaining route force, including forces
    // that the source model previously applied through internal wrap bodies.
    NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_FULL_MUSCLE_ROW = 1u << 1u,
    // In full-muscle-row mode, also consume anchorEndpointIndex as a signed
    // zero-resultant force couple: positive node weights load one attachment
    // and negative weights load the opposing attachment. This reconstructs a
    // massless two-segment tendon such as QAT -> patella -> PTL -> tibia while
    // the load endpoint remains the explicitly restored proximal reaction.
    NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_DISTAL_FORCE_COUPLE = 1u << 2u,
};

typedef struct NM_ALIGN16 NMNumiHumanTendonFEMLoadDispatchGPU {
    nm_u32 abiVersion;
    nm_u32 environmentCount;
    nm_u32 femNodeCount;
    nm_u32 endpointCount;

    nm_u32 transferStride;
    nm_u32 stepIndex;
    nm_u32 replacementCount;
    nm_u32 dofCount;

    nm_u32 bodyPoseStride;
    nm_u32 articulationFirstBody;
    nm_u32 pointJacobianStride;
    nm_u32 bodyJacobianPointOffset;

    nm_u32 generalizedForceStride;
    nm_u32 generalizedForceOffset;
    nm_u32 muscleCount;
    nm_u32 femContactSampleCount;
    nm_u32 articularContactSampleCount;
    nm_u32 passiveLigamentCount;
    nm_u32 reserved2;
    nm_u32 reserved3;
} NMNumiHumanTendonFEMLoadDispatchGPU;

typedef struct NM_ALIGN16 NMNumiHumanTendonFEMNodeLoadGPU {
    // Up to four terminal forces may contribute to one continuum node. An
    // unused slot has NM_INVALID_INDEX and a zero scale. This is required for
    // anatomical junctions such as the quadriceps tendon, where four source
    // muscles share one proximal traction surface without arbitrary spatial
    // partitioning.
    nm_u32 endpointIndex[4];

    // Component i is the signed fraction of endpointIndex[i]'s terminal world
    // force assigned to this node. Ordinary traction components are positive.
    // Negative components are admitted only for an explicitly declared distal
    // force couple and apply the equal-and-opposite attachment load.
    nm_float4 scale;
} NMNumiHumanTendonFEMNodeLoadGPU;

typedef struct NM_ALIGN16 NMNumiHumanTendonFEMNodeAnchorGPU {
    nm_u32 bodyIndex;
    nm_u32 flags;
    nm_u32 reserved0;
    nm_u32 reserved1;

    // Prescribed attachment in COM-relative Human body coordinates.
    nm_float4 localPoint;
} NMNumiHumanTendonFEMNodeAnchorGPU;

typedef struct NM_ALIGN16 NMNumiHumanTendonFEMEndpointReplacementGPU {
    nm_u32 loadEndpointIndex;
    nm_u32 anchorEndpointIndex;
    nm_u32 flags;
    nm_u32 reserved0;

    // x is the source-force fraction replaced by the continuum. In endpoint
    // mode this is the anchor endpoint J^T share. In full-muscle-row mode it
    // is the complete selected muscle row, with the load endpoint reaction
    // restored explicitly. yzw are reserved and must be zero.
    nm_float4 forceOwnerFraction;
} NMNumiHumanTendonFEMEndpointReplacementGPU;

// One fixed-correspondence, frictionless elastic-foundation contact sample.
// The sample is internal to one FEM world: slave and master forces are
// assembled into the same external-force arena and therefore have zero net
// force by construction. Current accepted FEM positions determine closure.
typedef struct NM_ALIGN16 NMNumiHumanFEMContactSampleGPU {
    nm_u32 slaveNode;
    nm_u32 masterNode0;
    nm_u32 masterNode1;
    nm_u32 masterNode2;

    // xyz master triangle barycentric weights; w reference signed separation.
    nm_float4 barycentricAndReferenceSeparation;
    // xyz reference master-to-slave normal; w slave tributary area [m^2].
    nm_float4 normalAndArea;
    // x effective pressure-overclosure stiffness [Pa/m]; yzw reserved zero.
    nm_float4 stiffness;
} NMNumiHumanFEMContactSampleGPU;

typedef struct NM_ALIGN16 NMNumiHumanFEMContactContributionGPU {
    nm_u32 sampleIndex;
    // 0 slave, 1 master node 0, 2 master node 1, 3 master node 2.
    nm_u32 role;
    nm_u32 reserved0;
    nm_u32 reserved1;
} NMNumiHumanFEMContactContributionGPU;

// Exact paired-triangle elastic-foundation contact between two articulated
// bodies. The slave point and all three vertices of its source master triangle
// are stored in their owning body frames. Live Metal poses supply the current
// triangle, closest point, normal, and separation. Equal/opposite wrenches are
// reduced per body before generalized-force scatter.
typedef struct NM_ALIGN16 NMNumiHumanArticularContactSampleGPU {
    nm_u32 slaveBodyIndex;
    nm_u32 masterBodyIndex;
    nm_u32 flags;
    nm_u32 reserved0;
    // xyz slave-body local point; w tributary area [m^2].
    nm_float4 slaveLocalPointAndArea;
    // xyz first master-triangle vertex; w reference signed separation [m].
    nm_float4 masterLocalTriangle0AndReferenceSeparation;
    // xyz second master-triangle vertex; w pressure/closure stiffness [Pa/m].
    nm_float4 masterLocalTriangle1AndStiffness;
    // xyz third master-triangle vertex. The cooked winding points toward the
    // slave in the reference configuration. w is the maximum layer-normal
    // strain per unit pressure [1/Pa] of the two contacting foundation layers.
    nm_float4 masterLocalTriangle2AndNormalStrainPerPressure;
    // xyz reference contact normal in the master-body frame. It orients the
    // live closest-point normal through face, edge, and vertex regions so a
    // slave crossing the surface cannot silently flip its repulsive side.
    // w is reserved and must be zero.
    nm_float4 masterLocalReferenceNormalAndReserved;
} NMNumiHumanArticularContactSampleGPU;

enum NMNumiHumanPassiveLigamentFlags : nm_u32 {
    NM_NUMI_HUMAN_PASSIVE_LIGAMENT_ACTIVE = 1u << 0u,
};

// Reduced axial owner for an authored passive ligament fibre family. The two
// local points are centroids of the exact source enthesis attachment-node
// sets, not visual mesh landmarks. The reference area is inferred from source
// tetrahedral volume divided by reference centroid separation. It owns only
// axial fibre stress; the parallel neutral FEM owns matrix and 3D shape.
typedef struct NM_ALIGN16 NMNumiHumanPassiveLigamentGPU {
    nm_u32 firstBodyIndex;
    nm_u32 secondBodyIndex;
    nm_u32 flags;
    nm_u32 reserved0;

    nm_float4 firstLocalPoint;
    nm_float4 secondLocalPoint;
    // c3 [Pa], c4 [1], c5 [Pa], lambda_max [1].
    nm_float4 material;
    // rest centroid length [m], effective area [m^2], in-situ stretch [1], 0.
    nm_float4 reference;
} NMNumiHumanPassiveLigamentGPU;

// One provisional/accepted audit per environment. Equal/opposite endpoint
// loads must close in force and moment about the world origin.
typedef struct NM_ALIGN16 NMNumiHumanPassiveLigamentAuditGPU {
    // xyz force residual [N], w sum of endpoint-force magnitudes [N].
    nm_float4 forceResidualAndL1;
    // xyz moment residual [N m], w maximum element tension [N].
    nm_float4 momentResidualAndMaximumTension;
    // x minimum effective stretch, y maximum effective stretch,
    // z active element count, w accepted marker.
    nm_float4 stretchCountAndAccepted;
} NMNumiHumanPassiveLigamentAuditGPU;

enum NMNumiHumanArticularContactSampleFlags : nm_u32 {
    NM_NUMI_HUMAN_ARTICULAR_CONTACT_ACTIVE = 1u << 0u,
    // Preserve an authored articular interface whose two regions are owned by
    // the same articulated rigid body. It remains in the immutable anatomy
    // fingerprint and diagnostics, but must not inject a rigid-body wrench:
    // the relative transform is identically zero until an internal deformable
    // owner is introduced.
    NM_NUMI_HUMAN_ARTICULAR_CONTACT_INTERNAL_SAME_BODY = 1u << 1u,
};

typedef struct NM_ALIGN16 NMNumiHumanBodyWrenchGPU {
    nm_float4 force;
    nm_float4 torque;
} NMNumiHumanBodyWrenchGPU;

// One deterministic audit record per environment. Moment residual is measured
// about the world origin, so balanced contact must cancel independently of the
// articulated body origins used by the reduced wrench representation.
typedef struct NM_ALIGN16 NMNumiHumanArticularContactAuditGPU {
    // xyz net body force; w sum of body-force magnitudes [N].
    nm_float4 forceResidualAndL1;
    // xyz net moment about world origin [N m]; w maximum pressure [Pa].
    nm_float4 momentResidualAndMaximumPressure;
    // x total normal force [N]; y closed tributary area [m^2];
    // z closed distinct-body sample count; w typed same-body sample count.
    nm_float4 normalForceAreaAndCounts;
    // x stored elastic-foundation energy [J]; y maximum layer-normal strain;
    // z maximum closure [m]. w is zero for a provisional record and one only
    // after the enclosing articulated-body transaction has been accepted.
    nm_float4 energyStrainClosureAndAccepted;
} NMNumiHumanArticularContactAuditGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(NMNumiHumanTendonFEMLoadDispatchGPU) == 80u);
static_assert(sizeof(NMNumiHumanTendonFEMNodeLoadGPU) == 32u);
static_assert(sizeof(NMNumiHumanTendonFEMNodeAnchorGPU) == 32u);
static_assert(sizeof(NMNumiHumanTendonFEMEndpointReplacementGPU) == 32u);
static_assert(sizeof(NMNumiHumanFEMContactSampleGPU) == 64u);
static_assert(sizeof(NMNumiHumanFEMContactContributionGPU) == 16u);
static_assert(sizeof(NMNumiHumanArticularContactSampleGPU) == 96u);
static_assert(sizeof(NMNumiHumanPassiveLigamentGPU) == 80u);
static_assert(sizeof(NMNumiHumanPassiveLigamentAuditGPU) == 48u);
static_assert(sizeof(NMNumiHumanBodyWrenchGPU) == 32u);
static_assert(sizeof(NMNumiHumanArticularContactAuditGPU) == 64u);
#endif
