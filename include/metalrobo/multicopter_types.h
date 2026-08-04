#pragma once

#include "metalrobo/engine_types.h"

#define MR_MULTICOPTER_MAX_ROTORS 8u

// One rotor's attachment and signed reaction-torque convention, all in the
// parent body frame. The source model owns this data; the kernel has no
// airframe-specific branches.
typedef struct MR_ALIGN16 MRMulticopterRotorGPU {
    // xyz = thrust application point from parent COM, w = reaction sign.
    mr_float4 positionAndReactionSign;
} MRMulticopterRotorGPU;

typedef struct MR_ALIGN16 MRMulticopterModelGPU {
    mr_u32 rotorCount;
    mr_u32 reserved0;
    mr_u32 reserved1;
    mr_u32 reserved2;

    // thrust coefficient, moment coefficient, rotor drag, rolling moment.
    mr_float4 coefficients;
    // rise time, fall time, maximum rotor speed, physics timestep.
    mr_float4 motorAndTimestep;
} MRMulticopterModelGPU;

// Persistent speed state, rad/s. Only the first rotorCount entries are live.
typedef struct MR_ALIGN16 MRMulticopterStateGPU {
    mr_float4 rotorSpeed01;
    mr_float4 rotorSpeed45;
} MRMulticopterStateGPU;

typedef struct MR_ALIGN16 MRMulticopterDispatchGPU {
    mr_u32 environmentCount;
    mr_u32 bodyStride;
    mr_u32 bodyOffset;
    mr_u32 reserved0;

    // World-frame wind velocity. The source motor model applies rotor drag
    // against relative airspeed, not a body-velocity damping approximation.
    mr_float4 windVelocity;
} MRMulticopterDispatchGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(MRMulticopterRotorGPU) == 16);
static_assert(sizeof(MRMulticopterModelGPU) == 48);
static_assert(sizeof(MRMulticopterStateGPU) == 32);
static_assert(sizeof(MRMulticopterDispatchGPU) == 32);
#endif
