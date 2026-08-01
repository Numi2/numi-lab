#pragma once

#include "metalrobo/gpu_types.h"

// Stateless counter-derived random streams shared by native task and sensor
// execution. The complete key is explicit: callers never retain or advance a
// mutable RNG object. These functions intentionally use only integer
// operations before converting the upper 24 bits to a uniform float, which
// makes replay bitwise on one Apple GPU/build and portable to host mirrors.

static inline mr_u64 mr_counter_mix64(mr_u64 value) {
    value += 0x9e3779b97f4a7c15ull;
    value =
        (value ^ (value >> 30u)) *
        0xbf58476d1ce4e5b9ull;
    value =
        (value ^ (value >> 27u)) *
        0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

static inline float mr_task_counter_uniform(
    const mr_u64 seed,
    const mr_u32 environment,
    const mr_u32 episode,
    const mr_u32 controlStep,
    const mr_u32 channel
) {
    mr_u64 key = seed;
    key ^= static_cast<mr_u64>(environment + 1u) *
        0xd2b74407b1ce6e93ull;
    key ^= static_cast<mr_u64>(episode + 1u) *
        0xca5a826395121157ull;
    key ^= static_cast<mr_u64>(controlStep + 1u) *
        0x9e3779b185ebca87ull;
    key ^= static_cast<mr_u64>(channel + 1u) *
        0x94d049bb133111ebull;
    return
        static_cast<float>(
            static_cast<mr_u32>(mr_counter_mix64(key) >> 40u)
        ) *
        (1.0f / 16777216.0f);
}

static inline float mr_sensor_counter_uniform(
    const mr_u64 seed,
    const mr_u32 environment,
    const mr_u32 episode,
    const mr_u64 sensorIdentity,
    const mr_u64 sample,
    const mr_u32 channel,
    const mr_u32 purpose
) {
    mr_u64 key = seed ^
        mr_counter_mix64(sensorIdentity ^
            0x6a09e667f3bcc909ull);
    key ^= static_cast<mr_u64>(environment + 1u) *
        0xbb67ae8584caa73bull;
    key ^= static_cast<mr_u64>(episode + 1u) *
        0x3c6ef372fe94f82bull;
    key ^= mr_counter_mix64(
        sample ^ 0xa54ff53a5f1d36f1ull
    );
    key ^= static_cast<mr_u64>(channel + 1u) *
        0x510e527fade682d1ull;
    key ^= static_cast<mr_u64>(purpose + 1u) *
        0x9b05688c2b3e6c1full;
    return
        static_cast<float>(
            static_cast<mr_u32>(mr_counter_mix64(key) >> 40u)
        ) *
        (1.0f / 16777216.0f);
}
