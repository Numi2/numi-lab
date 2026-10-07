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

#define MR_HUMAN_RESTING_COMMON_COORDINATE_COUNT 7u
#define MR_HUMAN_RESTING_COMMON_VOLUME_COUNT 7u
#define MR_HUMAN_RESTING_COMMON_CUBIC_TERM_COUNT 120u
#define MR_HUMAN_RESTING_COMMON_STATUS_OUTSIDE_CERTIFIED_DOMAIN 5u
typedef struct MR_ALIGN16 MRHumanRestingCommonFieldVertexGPU { mr_float4 displacement[MR_HUMAN_RESTING_COMMON_COORDINATE_COUNT]; } MRHumanRestingCommonFieldVertexGPU;
typedef struct MR_ALIGN16 MRHumanRestingCommonCoordinateBoxGPU { mr_float4 lower[2]; mr_float4 upper[2]; } MRHumanRestingCommonCoordinateBoxGPU;
typedef struct MR_ALIGN16 MRHumanRestingCommonFieldGPU {
    mr_float4 volumePolynomial[MR_HUMAN_RESTING_COMMON_VOLUME_COUNT][30];
    mr_float4 sourceReferenceVolumes[2];
    mr_float4 materialTargetVolumes;
    mr_float4 trialLower[2];
    mr_float4 trialUpper[2];
    mr_float4 solver;
    mr_uint4 countsAndFlags;
} MRHumanRestingCommonFieldGPU;
typedef struct MR_ALIGN16 MRHumanRestingCommonCoordinatesGPU {
    mr_float4 first; mr_float4 second; mr_uint4 status; mr_float4 diagnostics;
} MRHumanRestingCommonCoordinatesGPU;
// First failed physical-step snapshot for common-field/respiration triage.
// The device latches one record per segmented command buffer; host readback
// occurs only after the segment has completed or rolled back.
typedef struct MR_ALIGN16 MRHumanRestingCommonFailureGPU {
    mr_uint4 identity; // failed control step, Matter code, stand code, valid
    mr_uint4 matterStatus; // environment, object, failing index, microsteps
    mr_float4 matterDiagnostics;
    mr_uint4 standStatus; // completed steps, failing index, contact iterations, flags
    mr_uint4 respirationCandidateStatus;
    mr_uint4 respirationAcceptedStatus;
    mr_float4 chamberVolumes;
    mr_float4 respirationControl;
    mr_float4 commonFirst; // diagnostic replay coordinates, even when presentation result is sanitized
    mr_float4 commonSecond;
    mr_uint4 commonStatus; // solver code, iterations, matched box, reserved
    mr_float4 commonDiagnostics;
    mr_uint4 brainInputMetadata; // validity, flags, source time low, target time low
    mr_uint4 brainInputRoots; // source root low/high, target root low/high
    mr_uint4 brainInputTimestampHighs; // source/target timestamp high words
    mr_uint4 brainAcceptedMetadata; // flags, timestamp low/high, reserved
    mr_uint4 brainAcceptedRoots; // accepted root low/high, reserved
    mr_uint4 brainCandidateMetadata; // flags, timestamp low/high, reserved
    mr_uint4 brainCandidateRoot; // accepted root low/high, reserved
    mr_uint4 brainOutputMetadata; // flags, target timestamp low/high, reserved
    mr_uint4 brainOutputRoots; // source root low/high, target root low/high
    mr_float4 brainOutputSetpoints; // minute ventilation, frequency, tidal volume, diaphragm excitation
    mr_float4 brainOutputExcitations; // intercostal excitation, reserved
    mr_float4 excitationBuffer; // exact input consumed by respiratory predictor
} MRHumanRestingCommonFailureGPU;
#if defined(__cplusplus)
static_assert(sizeof(MRHumanRestingCommonFieldVertexGPU)==112);
static_assert(sizeof(MRHumanRestingCommonCoordinateBoxGPU)==64);
static_assert(sizeof(MRHumanRestingCommonFieldGPU)==3504);
static_assert(sizeof(MRHumanRestingCommonCoordinatesGPU)==64);
static_assert(sizeof(MRHumanRestingCommonFailureGPU)==384);
#endif

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
#define MR_HUMAN_RESTING_VOLUME_AUDIT_GROUP_THREADS 256u
typedef struct MR_ALIGN16 MRHumanRestingVolumeAuditGroupGPU {
    mr_uint4 surfaceAndTriangleRange; // surface row, first local triangle, triangle count, reserved
} MRHumanRestingVolumeAuditGroupGPU;
typedef struct MR_ALIGN16 MRHumanRestingVolumeAuditRangeGPU {
    mr_uint4 groupRange; // first partial group, group count, reserved, reserved
} MRHumanRestingVolumeAuditRangeGPU;
typedef struct MR_ALIGN16 MRHumanRestingVolumeAuditPartialGPU {
    float signedVolume;
    mr_u32 invalidTriangleCount;
    mr_u32 firstInvalidTriangle;
    mr_u32 reserved;
} MRHumanRestingVolumeAuditPartialGPU;
#define MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONE 0u
#define MR_HUMAN_RESTING_TRIANGLE_FAILURE_NONFINITE_AREA 1u
#define MR_HUMAN_RESTING_TRIANGLE_FAILURE_EXACT_ZERO_AREA 2u
#define MR_HUMAN_RESTING_TRIANGLE_FAILURE_NO_TRIANGLE 0xffffffffu
// One compact per-surface record written by the existing volume audit. The
// first two lanes identify the audit row and local triangle ordinal; the
// third lane distinguishes non-finite from exactly zero area. Positions are
// the three binary32 rendered vertices from that same accepted presentation.
typedef struct MR_ALIGN16 MRHumanRestingSurfaceFailureGPU {
    mr_uint4 surfaceTriangleKind;
    mr_uint4 vertexIndices;
    mr_float4 renderedPositions[3];
} MRHumanRestingSurfaceFailureGPU;
#if defined(__cplusplus)
static_assert(sizeof(MRHumanRestingVolumeAuditGroupGPU)==16);
static_assert(sizeof(MRHumanRestingVolumeAuditRangeGPU)==16);
static_assert(sizeof(MRHumanRestingVolumeAuditPartialGPU)==16);
#endif
