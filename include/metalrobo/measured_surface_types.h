#pragma once

#include "metalrobo/engine_types.h"

#define MR_MEASURED_SURFACE_ABI_VERSION 5u
#define MR_MEASURED_SURFACE_ACTION_CAPACITY 32u
#define MR_MEASURED_SURFACE_COMPONENT_CAPACITY 16u
#define MR_MEASURED_SURFACE_PHASE_CLAMP 0u
#define MR_MEASURED_SURFACE_PHASE_REFLECT 1u
#define MR_MEASURED_SURFACE_PHASE_WRAP 2u

// Immutable component ranges over one fixed-topology measured surface.
typedef struct MR_ALIGN16 MRMeasuredSurfaceComponentGPU {
    mr_u32 partIdentifier;
    mr_u32 vertexOffset;
    mr_u32 vertexCount;
    mr_u32 triangleOffset;
    mr_u32 triangleCount;
    mr_u32 reserved0;
    mr_u32 reserved1;
    mr_u32 reserved2;
} MRMeasuredSurfaceComponentGPU;

// One bounded second-order surface-control lane.
typedef struct MR_ALIGN16 MRMeasuredSurfaceActionGPU {
    // lower bound, upper bound, natural frequency Hz, damping ratio.
    mr_float4 boundsFrequencyDamping;
    // Normalized action bias followed by reserved values. Policy actions are
    // residuals around this fingerprinted robot trim.
    mr_float4 normalizedBiasReserved;
} MRMeasuredSurfaceActionGPU;

// Compiled, pointer-free description. Source arrays remain separate immutable
// buffers so all environments borrow one measured geometry payload.
typedef struct MR_ALIGN16 MRMeasuredSurfaceModelGPU {
    mr_u32 abiVersion;
    mr_u32 frameCount;
    mr_u32 vertexCount;
    mr_u32 triangleCount;

    mr_u32 componentCount;
    mr_u32 actionCount;
    mr_u32 phaseBoundaryMode;
    mr_u32 sourcePeriodic;

    // Stable measured vertex anchors for body, left wing, right wing, and
    // tail. Dynamic hinges must follow the measured source instead of being
    // reconstructed from global bounds.
    mr_uint4 componentAnchorVertexIndices;

    // sample rate Hz, air density kg/m3, normal drag, tangential drag.
    mr_float4 samplingAndAerodynamics;
    // Separation onset normal ratio, full-incidence normal retention,
    // maximum ground-effect lift increment, height scale in wing spans.
    mr_float4 aerodynamicCorrections;
    // Local measured-surface center xyz and bounding radius.
    mr_float4 centerAndRadius;
    mr_float4 boundsMinimum;
    mr_float4 boundsMaximum;
} MRMeasuredSurfaceModelGPU;

// Resolved binding into one universal MetalWorld CompiledRun.
typedef struct MR_ALIGN16 MRCompiledMeasuredSurfaceDispatchGPU {
    mr_u32 environmentCount;
    mr_u32 qStride;
    mr_u32 vStride;
    mr_u32 bodyStride;

    mr_u32 qOffset;
    mr_u32 vOffset;
    mr_u32 bodyIndex;
    mr_u32 localBodyIndex;

    mr_u32 actionCount;
    mr_u32 actionHistoryStride;
    mr_u32 filterSlot;
    mr_u32 firstAction;

    mr_u32 threadsPerThreadgroup;
    mr_u32 reserved0;
    mr_u32 reserved1;
    mr_u32 reserved2;

    // Physics substep and world-frame wind velocity.
    mr_float4 timestepAndWindX;
    mr_float4 windYZ;
} MRCompiledMeasuredSurfaceDispatchGPU;

// Persistent per-environment generalized surface state. The fixed capacity is
// ABI, not a dove assumption; actionCount selects the live prefix.
typedef struct MR_ALIGN16 MRMeasuredSurfaceStateGPU {
    mr_float4 position[MR_MEASURED_SURFACE_ACTION_CAPACITY / 4u];
    mr_float4 velocity[MR_MEASURED_SURFACE_ACTION_CAPACITY / 4u];
    // phase in source-frame coordinates, signed phase rate, accumulated
    // aerodynamic impulse norm, accepted step count.
    mr_float4 phaseRateImpulseStep;
} MRMeasuredSurfaceStateGPU;

typedef struct MR_ALIGN16 MRMeasuredSurfaceEvidenceGPU {
    // force magnitude N, torque magnitude N m, area m2, phase frames.
    mr_float4 loadsAreaPhase;
    // maximum deformation m, actuator norm, status, reserved.
    mr_float4 deformationActuationStatus;
    // Signed instantaneous world-frame force xyz and magnitude N.
    mr_float4 worldForceAndMagnitude;
    // Signed instantaneous world-frame torque xyz and magnitude N m.
    mr_float4 worldTorqueAndMagnitude;
    // Accepted signed force impulse xyz N s and elapsed integration time s.
    mr_float4 worldForceImpulseAndTime;
    // Accepted signed torque impulse xyz N m s; w is reserved.
    mr_float4 worldTorqueImpulse;
} MRMeasuredSurfaceEvidenceGPU;

// Renderer-facing vertex cache. Positions are prepared from the same
// accepted measured-surface state and exact deformation equation used by the
// aerodynamic solver, then consumed by the dynamic raster/composite passes.
typedef struct MR_ALIGN16 MRMeasuredSurfaceVisualVertexGPU {
    mr_float4 currentWorldPosition;
    mr_float4 previousWorldPosition;
} MRMeasuredSurfaceVisualVertexGPU;

typedef struct MR_ALIGN16 MRMeasuredSurfacePresentationGPU {
    // environments, body stride, source accepted-state offset, body index.
    mr_uint4 counts;
    // vertex count, triangle count, selected camera, reserved.
    mr_uint4 topology;
    // semantic id, instance id, link/body id, optional-output mask.
    mr_uint4 identity;
    // Link-origin position expressed in the COM-centred body frame.
    mr_float4 localTranslationAndScale;
    // Linear base color and opacity.
    mr_float4 baseColorAndOpacity;
    // perceptual roughness, metallic, environment intensity, reserved.
    mr_float4 material;
} MRMeasuredSurfacePresentationGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(MRMeasuredSurfaceComponentGPU) == 32);
static_assert(sizeof(MRMeasuredSurfaceActionGPU) == 32);
static_assert(sizeof(MRMeasuredSurfaceModelGPU) == 128);
static_assert(sizeof(MRCompiledMeasuredSurfaceDispatchGPU) == 96);
static_assert(sizeof(MRMeasuredSurfaceStateGPU) == 272);
static_assert(sizeof(MRMeasuredSurfaceEvidenceGPU) == 96);
static_assert(sizeof(MRMeasuredSurfaceVisualVertexGPU) == 32);
static_assert(sizeof(MRMeasuredSurfacePresentationGPU) == 96);
#endif
