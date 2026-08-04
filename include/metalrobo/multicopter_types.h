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

// Canonical policy action: collective thrust, roll, pitch, yaw, each clipped
// to [-1, 1].  A policy writes this device buffer directly; the mixer produces
// source-unit rotor speed targets without a host control loop.
typedef struct MR_ALIGN16 MRMulticopterActionGPU {
    mr_float4 collectiveRollPitchYaw;
} MRMulticopterActionGPU;

// Hover speed followed by rad/s action scales for collective, roll/pitch, yaw.
typedef struct MR_ALIGN16 MRMulticopterMixerGPU {
    mr_float4 hoverAndScales;
} MRMulticopterMixerGPU;

// Resolved CompiledRun binding for the universal MetalWorld state layout.
// The program reads filtered task actions and writes an ABA body wrench; it
// does not own integration, gravity, collision, contact, or episode state.
typedef struct MR_ALIGN16 MRCompiledMulticopterDispatchGPU {
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

    mr_float4 windVelocity;
} MRCompiledMulticopterDispatchGPU;

// A compact, Markov hover/position task contract. Observation lanes are
// target-relative position, world velocity, body-up direction, angular
// velocity, and normalized rotor speeds. rewardAndDone = reward, done, tilt,
// target-distance. The policy need not observe privileged reward lanes.
typedef struct MR_ALIGN16 MRMulticopterFlightTaskGPU {
    mr_float4 targetPositionAndMinimumHeight;
    mr_float4 maximumHeightTiltAndScales;
} MRMulticopterFlightTaskGPU;

typedef struct MR_ALIGN16 MRMulticopterFlightTransitionGPU {
    mr_float4 positionErrorAndHeight;
    mr_float4 linearVelocity;
    mr_float4 bodyUpAndTilt;
    mr_float4 angularVelocity;
    mr_float4 normalizedRotorSpeed;
    mr_float4 rewardAndDone;
} MRMulticopterFlightTransitionGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(MRMulticopterRotorGPU) == 16);
static_assert(sizeof(MRMulticopterModelGPU) == 48);
static_assert(sizeof(MRMulticopterStateGPU) == 32);
static_assert(sizeof(MRMulticopterDispatchGPU) == 32);
static_assert(sizeof(MRMulticopterActionGPU) == 16);
static_assert(sizeof(MRMulticopterMixerGPU) == 16);
static_assert(sizeof(MRCompiledMulticopterDispatchGPU) == 64);
static_assert(sizeof(MRMulticopterFlightTaskGPU) == 32);
static_assert(sizeof(MRMulticopterFlightTransitionGPU) == 96);
#endif
