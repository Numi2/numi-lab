#pragma once
#include "plastic_bag_gpu.h"
typedef struct MR_ALIGN16 FrankaGripPoseGPU {
 mr_float4 centerAndHalfGap;
 mr_float4 normal;
 mr_float4 tangentX;
 mr_float4 tangentZ;
} FrankaGripPoseGPU;
typedef struct MR_ALIGN16 FrankaGripRuntimeGPU {
 FrankaGripPoseGPU current;
 FrankaGripPoseGPU previous;
} FrankaGripRuntimeGPU;
typedef struct MR_ALIGN16 FrankaGripContactGPU {
 mr_float4 normalMultipliers;
 mr_float4 tangentMultiplier;
 mr_float4 incrementalWork;
} FrankaGripContactGPU;
