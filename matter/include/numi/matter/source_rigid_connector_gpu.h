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

#ifdef __cplusplus
static_assert(sizeof(NMSourceCylindricalJointGPU) == 144);
static_assert(sizeof(NMSourceRigidSpringGPU) == 96);
static_assert(sizeof(NMSourceFEMRigidTieGPU) == 32);
#endif
