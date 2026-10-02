#pragma once

#include "numi/matter/shared.h"

// Immutable connector rows. Proxy indices address one collision proxy per
// source body; the runtime maps dynamic proxies to their compact six-DOF block.
// Reference positions, origins and insertion points use the source frame.
typedef struct NM_ALIGN16 NMSourceCylindricalJointGPU {
    nm_uint4 indices; // proxy A, proxy B, prescribed translation, prescribed rotation
    nm_float4 referenceA;
    nm_float4 referenceB;
    nm_float4 origin;
    nm_float4 axis;
    nm_float4 forceMultiplier;
    nm_float4 momentMultiplier;
    nm_float4 parameters; // force penalty, moment penalty, translation, rotation
    nm_float4 axial; // prescribed axial force, moment, reserved, reserved
} NMSourceCylindricalJointGPU;

typedef struct NM_ALIGN16 NMSourceRigidSpringGPU {
    nm_uint4 indices; // proxy A, proxy B, reserved, reserved
    nm_float4 referenceA;
    nm_float4 referenceB;
    nm_float4 insertionA;
    nm_float4 insertionB;
    nm_float4 parameters; // stiffness, free length, reserved, reserved
} NMSourceRigidSpringGPU;

// A source FEBio rigid tie owns one cooked, otherwise statically fixed FEM
// node. The runtime promotes that node to a moving eliminated coordinate and
// maps its complete residual and tangent to the named rigid proxy.
typedef struct NM_ALIGN16 NMSourceFEMRigidTieGPU {
    nm_uint4 identity; // cooked FEM node, rigid proxy, cooked object, stable source ID
    nm_float4 localPoint; // body-COM-relative material point, w = 0
} NMSourceFEMRigidTieGPU;

// One source discrete linear spring between two cooked FEM nodes. The pinned
// MCL-to-medial-meniscus set uses 402 such edges. Coordinates are SI reference
// points and stiffness is N/m; zero free length means initial separation.
typedef struct NM_ALIGN16 NMSourceFEMSpringGPU {
    nm_uint4 identity; // node A, node B, source edge ID, reserved
    nm_float4 referenceA;
    nm_float4 referenceB;
    nm_float4 parameters; // stiffness, free length, reserved, reserved
} NMSourceFEMSpringGPU;

typedef struct NM_ALIGN16 NMSourceFEMSpringNodeGPU {
    nm_uint4 identity; // cooked node, first incidence, incidence count, reserved
} NMSourceFEMSpringNodeGPU;

// Source prestrain is an externally prescribed material parameter during one
// implicit root. The first interval ramps from initial to preload; the second
// holds that preload while prescribed flexion advances. The parameter offset
// addresses the cooked environment-local material overlay, never its defaults.
typedef struct NM_ALIGN16 NMSourcePrestrainGPU {
    nm_uint4 identity; // global parameter offset, cooked material, source curve ID, reserved
    nm_float4 stretch; // at source t=0, at t=1, reserved, reserved
} NMSourcePrestrainGPU;

#ifdef __cplusplus
static_assert(sizeof(NMSourceCylindricalJointGPU) == 144);
static_assert(sizeof(NMSourceRigidSpringGPU) == 96);
static_assert(sizeof(NMSourceFEMRigidTieGPU) == 32);
static_assert(sizeof(NMSourceFEMSpringGPU) == 64);
static_assert(sizeof(NMSourceFEMSpringNodeGPU) == 16);
static_assert(sizeof(NMSourcePrestrainGPU) == 32);
#endif
