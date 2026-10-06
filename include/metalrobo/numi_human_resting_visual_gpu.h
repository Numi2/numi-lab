#pragma once
#include "metalrobo/engine_types.h"
#include "metalrobo/visual_platform_types.h"

// Presentation skinning compiled once from the existing registered anatomy
// payloads. Positive source weights are retained without truncation.
typedef struct MR_ALIGN16 MRHumanRestingVertexMap {
    mr_u32 firstInfluence, influenceCount, deformationKind, chamberIndex;
    mr_float4 deformationWeight;
    mr_float4 respiratoryBasis; // source-space gradient xyz and scalar basal weight w
} MRHumanRestingVertexMap;
typedef struct MR_ALIGN16 MRHumanRestingInfluence {
    mr_float4 positionAndWeight;
    mr_float4 normal;
    mr_uint4 body;
} MRHumanRestingInfluence;

// Reduced source-bound ventricular myocardium map. Displacements are stored
// per unit CVSim q for RV/LV channels and per metre for volume closure. The
// material-volume polynomial uses normalized qRV/.4, qLV/.2, closure/.01.
typedef struct MR_ALIGN16 MRHumanRestingCardiacWallVertexGPU {
    mr_float4 first;   // RV displacement per unit qRV, W is zero
    mr_float4 second;  // LV displacement per unit qLV, W is zero
    mr_float4 closure; // inferred outer-wall closure displacement per metre, W is zero
} MRHumanRestingCardiacWallVertexGPU;

typedef struct MR_ALIGN16 MRHumanRestingCardiacWallGPU {
    // 20 terms: 1,r,l,r2,rl,l2,r3,r2l,rl2,l3,c,rc,lc,r2c,rlc,l2c,c2,rc2,lc2,c3.
    mr_float4 volumePolynomial[5];
    mr_float4 scalesAndVolume; // .4, .2, .01 scales and reference material volume m^3
    mr_float4 closureBoundsAndTolerance; // closure c_min/c_max in metres, relative tolerance, reserved
    mr_uint4 chambersAndFlags; // RV index, LV index, enabled flag, wall vertex count
} MRHumanRestingCardiacWallGPU;

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
    // Basal superior-axis displacement: 1-smoothstep(start,start+span,h).
    // z is d(enclosed lobe volume)/d(displacement), computed from source faces.
    mr_float4 lungBasalBlend; // start m, span m, effective area m2, reserved
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
    mr_float4 reference; // x: source enclosed volume m^3, y: source lobe swept area m^2
} MRHumanRestingSurfaceAuditGPU;
