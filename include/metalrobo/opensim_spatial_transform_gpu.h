#pragma once

#include "metalrobo/gpu_types.h"

// Fixed-capacity, pointer-free device program for one source OpenSim
// SpatialTransform. RajagopalLaiUhlrich2023 fits these bounds exactly (at
// most five polynomial coefficients or thirteen SimmSpline knots per axis).
#define MR_OPENSIM_SPATIAL_TRANSFORM_GPU_ABI_VERSION 1u
#define MR_OPENSIM_SPATIAL_MAX_COORDINATES 6u
#define MR_OPENSIM_SPATIAL_MAX_COEFFICIENTS 16u
#define MR_OPENSIM_SPATIAL_MAX_KNOTS 16u
#define MR_OPENSIM_SPATIAL_COEFFICIENT_BLOCKS 4u
#define MR_OPENSIM_SPATIAL_KNOT_BLOCKS 4u
#define MR_OPENSIM_SPATIAL_NO_COORDINATE 0xffffffffu

enum MROpenSimFunctionGPUKind : mr_u32 {
    MR_OPENSIM_FUNCTION_CONSTANT = 0u,
    MR_OPENSIM_FUNCTION_LINEAR = 1u,
    MR_OPENSIM_FUNCTION_POLYNOMIAL = 2u,
    MR_OPENSIM_FUNCTION_SIMM_SPLINE = 3u,
};

enum MROpenSimSpatialTransformGPUStatus : mr_u32 {
    MR_OPENSIM_SPATIAL_SUCCESS = 0u,
    MR_OPENSIM_SPATIAL_INVALID_PROGRAM = 1u,
    MR_OPENSIM_SPATIAL_NONFINITE_INPUT = 2u,
    MR_OPENSIM_SPATIAL_NONFINITE_RESULT = 3u,
};

typedef struct MR_ALIGN16 MROpenSimFunctionGPU {
    mr_u32 kind;
    mr_u32 coordinateIndex;
    mr_u32 coefficientCount;
    mr_u32 knotCount;

    // xyz is the normalized source TransformAxis direction; w is zero.
    mr_float4 axis;
    mr_float4 coefficients[MR_OPENSIM_SPATIAL_COEFFICIENT_BLOCKS];
    mr_float4 abscissae[MR_OPENSIM_SPATIAL_KNOT_BLOCKS];
    mr_float4 ordinates[MR_OPENSIM_SPATIAL_KNOT_BLOCKS];
    // Source SimmSpline coefficients are compiled once on the host.
    mr_float4 splineSlope[MR_OPENSIM_SPATIAL_KNOT_BLOCKS];
    mr_float4 splineQuadratic[MR_OPENSIM_SPATIAL_KNOT_BLOCKS];
    mr_float4 splineCubic[MR_OPENSIM_SPATIAL_KNOT_BLOCKS];
} MROpenSimFunctionGPU;

typedef struct MR_ALIGN16 MROpenSimSpatialTransformGPU {
    mr_u32 abiVersion;
    mr_u32 coordinateCount;
    mr_u32 reserved0;
    mr_u32 reserved1;
    // FunctionBased order: rotation slots 0..2, translation slots 3..5.
    MROpenSimFunctionGPU axes[6];
} MROpenSimSpatialTransformGPU;

typedef struct MR_ALIGN16 MROpenSimSpatialTransformResultGPU {
    mr_u32 status;
    mr_u32 coordinateCount;
    mr_u32 reserved0;
    mr_u32 reserved1;

    // Rows of the parent-frame rotation and translation in source axes.
    mr_float4 rotationRow0;
    mr_float4 rotationRow1;
    mr_float4 rotationRow2;
    mr_float4 translation;
    // Entries beyond coordinateCount remain zero.
    mr_float4 motionAngular[MR_OPENSIM_SPATIAL_MAX_COORDINATES];
    mr_float4 motionLinear[MR_OPENSIM_SPATIAL_MAX_COORDINATES];
    mr_float4 motionAngularDot[MR_OPENSIM_SPATIAL_MAX_COORDINATES];
    mr_float4 motionLinearDot[MR_OPENSIM_SPATIAL_MAX_COORDINATES];
} MROpenSimSpatialTransformResultGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(MROpenSimFunctionGPU) == 416u);
static_assert(alignof(MROpenSimFunctionGPU) == 16u);
static_assert(sizeof(MROpenSimSpatialTransformGPU) == 2512u);
static_assert(alignof(MROpenSimSpatialTransformGPU) == 16u);
static_assert(sizeof(MROpenSimSpatialTransformResultGPU) == 464u);
static_assert(alignof(MROpenSimSpatialTransformResultGPU) == 16u);
#endif
