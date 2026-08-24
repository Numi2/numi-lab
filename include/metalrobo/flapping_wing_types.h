#pragma once

#include "metalrobo/engine_types.h"

// One explicitly authored wing in a compact bilateral flapping-airframe
// program.  The body and generalized-coordinate indices are resolved by the
// run compiler; Metal never selects a bird by name.
typedef struct MR_ALIGN16 MRFlappingWingGPU {
    mr_u32 bodyIndex;
    mr_u32 qIndex;
    mr_u32 vIndex;
    mr_u32 reserved0;

    // xyz = neutral root-COM vector in the airframe frame, w = planform area.
    mr_float4 rootToCenterAndArea;
    // xyz = unit hinge axis in the airframe frame, w = mean chord [m].
    mr_float4 hingeAxisAndChord;
    // lift slope [1/rad], profile drag, induced drag and coefficient limit.
    mr_float4 coefficients;
    // x = rotational/unsteady stroke-lift closure, y = forward stroke-plane
    // bias, zw reserved. Keeping this separate prevents a force term from
    // silently inflating the steady CL cap used by the blade elements.
    mr_float4 unsteadyCoefficients;
} MRFlappingWingGPU;

// Tail surface associated with a flapping-wing program. A direct pitch joint
// stores its resolved q/v indices in the formerly reserved lanes; fixed tails
// use MR_INVALID_INDEX. Its external wrench is evaluated from accepted state
// and the local wing-wash closure; it never replays a supplied force trace.
typedef struct MR_ALIGN16 MRAeroTailGPU {
    mr_u32 bodyIndex;
    mr_u32 rootBodyIndex;
    mr_u32 qIndex;
    mr_u32 vIndex;

    // xyz = neutral root-COM to aerodynamic center; w = area [m^2].
    mr_float4 rootToCenterAndArea;
    // x = chord [m], y = lift slope, z = profile drag, w = pitch damping.
    mr_float4 chordAndCoefficients;
} MRAeroTailGPU;

// Root airframe drag and angular-rate damping.  The reference areas are
// expressed in the airframe forward/span/up axes, so this stays valid under
// arbitrary world orientation and wind.
typedef struct MR_ALIGN16 MRAeroFuselageGPU {
    mr_u32 bodyIndex;
    mr_u32 rootBodyIndex;
    mr_u32 reserved0;
    mr_u32 reserved1;

    // xyz = reference areas normal to forward/span/up axes [m^2],
    // w = positive quadratic drag coefficient.
    mr_float4 referenceAreasAndDrag;
    // xyz = angular-rate damping about forward/span/up [N m s], w reserved.
    mr_float4 angularDamping;
} MRAeroFuselageGPU;

// Resolved program for a bilateral articulated airframe.  The model is a
// state-responsive blade-element load primitive, not a measured CFD claim:
// it consumes live root/hinge state and writes its result to the universal
// ABA external-wrench arena.
typedef struct MR_ALIGN16 MRCompiledFlappingWingDispatchGPU {
    mr_u32 environmentCount;
    mr_u32 qStride;
    mr_u32 vStride;
    mr_u32 bodyStride;

    mr_u32 qOffset;
    mr_u32 vOffset;
    mr_u32 rootBodyIndex;
    mr_u32 reserved0;

    // xyz = world wind velocity, w = air density [kg/m^3].
    mr_float4 windVelocityAndDensity;
} MRCompiledFlappingWingDispatchGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(MRFlappingWingGPU) == 80);
static_assert(sizeof(MRAeroTailGPU) == 48);
static_assert(sizeof(MRAeroFuselageGPU) == 48);
static_assert(sizeof(MRCompiledFlappingWingDispatchGPU) == 48);
#endif
