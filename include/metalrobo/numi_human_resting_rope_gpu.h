#pragma once
#include "metalrobo/engine_types.h"
// Optional massless, inextensible torso tether. No independent time or state.
typedef struct MR_ALIGN16 MRHumanRestingRopeGPU {
    mr_uint4 indices; // contact, query, global body, articulation first body
    mr_uint4 strides; // body pose, first body probe, point world, point Jacobian
    mr_uint4 counts; // environments, bodies, DOFs, ABI=1
    mr_float4 anchorAndLength; // fixed world anchor xyz, maximum length metres
    mr_float4 bodyLocalSite; // COM-relative body-local harness point xyz, w=0
} MRHumanRestingRopeGPU;
#if !defined(__METAL_VERSION__)
static_assert(sizeof(MRHumanRestingRopeGPU) == 80);
#endif
