#pragma once

// Pointer-free records shared by C++, Objective-C++, Metal, and MLX.
// Scenario values are kept separate from mutable world state so one sampled
// world can be traced through rendering, physics, policy evaluation, and
// eventual hardware calibration without reconstructing its parameters.

#include "metalrobo/gpu_types.h"

#define MR_R2S2R_ABI_VERSION 1u

enum MRWorldSamplingMode : mr_u32 {
    MR_WORLD_SAMPLING_COVERAGE = 0u,
    MR_WORLD_SAMPLING_CURRICULUM = 1u,
};

enum MRWorldSamplingSource : mr_u32 {
    MR_WORLD_SAMPLE_BROAD = 0u,
    MR_WORLD_SAMPLE_ALIGNMENT = 1u,
    MR_WORLD_SAMPLE_FAILURE = 2u,
    MR_WORLD_SAMPLE_UNCERTAINTY = 3u,
};

enum MRScenarioValueFlags : mr_u32 {
    MR_SCENARIO_VALUE_CATEGORICAL = 1u << 0u,
    MR_SCENARIO_VALUE_MEASURED = 1u << 1u,
    MR_SCENARIO_VALUE_MISSING = 1u << 2u,
};

enum MREpisodeSource : mr_u32 {
    MR_EPISODE_SOURCE_SIMULATION = 0u,
    MR_EPISODE_SOURCE_HARDWARE = 1u,
};

enum MREpisodeTermination : mr_u32 {
    MR_EPISODE_TERMINATION_SUCCESS = 0u,
    MR_EPISODE_TERMINATION_HORIZON = 1u,
    MR_EPISODE_TERMINATION_POLICY = 2u,
    MR_EPISODE_TERMINATION_PHYSICS = 3u,
    MR_EPISODE_TERMINATION_SAFETY = 4u,
    MR_EPISODE_TERMINATION_EXTERNAL = 5u,
};

enum MRFeedbackRegionKind : mr_u32 {
    MR_FEEDBACK_REGION_FAILURE = 0u,
    MR_FEEDBACK_REGION_UNCERTAINTY = 1u,
};

typedef struct MR_ALIGN16 MRWorldScenarioHeaderGPU {
    // scenario key low/high, episode counter low/high.
    mr_uint4 identity;
    // alignment fingerprint low/high, feedback fingerprint low/high.
    mr_uint4 provenance;
    // sampling mode, sampling source, region/particle index, ABI version.
    mr_uint4 sampling;
} MRWorldScenarioHeaderGPU;

typedef struct MR_ALIGN16 MRWorldScenarioValueGPU {
    // raw scalar value, normalized base-distribution quantile, component
    // weight, reserved.
    mr_float4 value;
    // variation ordinal, categorical value, categorical ordinal, flags.
    mr_uint4 identity;
} MRWorldScenarioValueGPU;

typedef struct MR_ALIGN16 MREpisodeOutcomeGPU {
    // scenario key low/high, episode counter low/high.
    mr_uint4 scenario;
    // source, termination, success flag, physics status.
    mr_uint4 outcome;
    // failure mask low/high, step count, ABI version.
    mr_uint4 evidence;
    // return, task margin, safety margin, duration seconds.
    mr_float4 task;
    // minimum visibility, integrated contact load, peak contact load,
    // valid-contact count.
    mr_float4 interaction;
} MREpisodeOutcomeGPU;

typedef struct MR_ALIGN16 MRWorldAdaptiveSampleUniformsGPU {
    // alignment particle count, feedback region count, variation count,
    // MRWorldSamplingMode.
    mr_uint4 counts;
    // episode-counter base low/high, alignment fingerprint low/high.
    mr_uint4 identity;
    // feedback fingerprint low/high and reserved.
    mr_uint4 provenance;
    // broad, failure, uncertainty mixture weights and alignment jitter.
    mr_float4 mixture;
    // R2S2R ABI version and reserved.
    mr_uint4 abi;
} MRWorldAdaptiveSampleUniformsGPU;

typedef struct MR_ALIGN16 MRWorldAlignmentParticleGPU {
    // normalized weight, cumulative weight, replay residual, reserved.
    mr_float4 statistics;
    // particle ordinal and reserved.
    mr_uint4 identity;
} MRWorldAlignmentParticleGPU;

typedef struct MR_ALIGN16 MRWorldFeedbackRegionGPU {
    // normalized class-local weight, cumulative class-local weight,
    // source model score, reserved.
    mr_float4 statistics;
    // MRFeedbackRegionKind, region ordinal, and reserved.
    mr_uint4 identity;
} MRWorldFeedbackRegionGPU;

typedef struct MR_ALIGN16 MRMLXWorldFamilyImportDispatchGPU {
    // environment count, nq, nv, scene-body count.
    mr_uint4 state;
    // body count, articulation count, variation count, R2S2R ABI version.
    mr_uint4 topology;
    // sampled family generation low/high and reserved.
    mr_uint4 generation;
} MRMLXWorldFamilyImportDispatchGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(MRWorldScenarioHeaderGPU) == 48u);
static_assert(sizeof(MRWorldScenarioValueGPU) == 32u);
static_assert(sizeof(MREpisodeOutcomeGPU) == 80u);
static_assert(sizeof(MRWorldAdaptiveSampleUniformsGPU) == 80u);
static_assert(sizeof(MRWorldAlignmentParticleGPU) == 32u);
static_assert(sizeof(MRWorldFeedbackRegionGPU) == 32u);
static_assert(sizeof(MRMLXWorldFamilyImportDispatchGPU) == 48u);
#endif
