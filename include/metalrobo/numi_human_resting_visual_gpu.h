#pragma once
#include "metalrobo/engine_types.h"
#include "metalrobo/visual_platform_types.h"

// Presentation skinning compiled once from the existing registered anatomy
// payloads. Positive source weights are retained without truncation.
typedef struct MR_ALIGN16 MRHumanRestingVertexMap {
    mr_u32 firstInfluence, influenceCount, deformationKind, chamberIndex;
    mr_float4 deformationWeight;
} MRHumanRestingVertexMap;
typedef struct MR_ALIGN16 MRHumanRestingInfluence {
    mr_float4 positionAndWeight;
    mr_float4 normal;
    mr_uint4 body;
} MRHumanRestingInfluence;

// Compiled from the registered, closed source surfaces at asset loading.
// Volumes are m^3, positions and height are m. The functional source cavities
// are ordered RA, RV, LA, LV, matching NMHumanRespirationState.chamberVolumes.
typedef struct MR_ALIGN16 MRHumanRestingAnatomyGPU {
    mr_uint4 bodyAndFlags;
    mr_float4 lungAnchorAndVolume;
    mr_float4 superiorAxisAndHeight;
    mr_float4 anteriorAxis;
    mr_float4 muscleAreas;
    mr_float4 diaphragmHeight; // source dome inferior/superior extent
    mr_float4 ribPivotAndGain[24]; // pivot m; radians per m of anterior excursion
    mr_float4 ribAxis[24]; // source costovertebral reduced hinge axis
    mr_float4 chamberCenterAndVolume[4];
    // V(q) = x + y*q + z*q^2 + w*q^3 for the per-vertex free-wall map
    // x' = c + (1 + q*weight)*(x-c). Coefficients are compiled once from the
    // admitted closed source cavity meshes, in m^3.
    mr_float4 chamberVolumePolynomial[4];
} MRHumanRestingAnatomyGPU;

typedef struct MR_ALIGN16 MRHumanRestingSurfaceAuditGPU {
    mr_uint4 indicesAndOwner; // first index, count, kind (1 lung / 2 cavity), channel
    mr_float4 reference; // x: reference enclosed volume in m^3
} MRHumanRestingSurfaceAuditGPU;
