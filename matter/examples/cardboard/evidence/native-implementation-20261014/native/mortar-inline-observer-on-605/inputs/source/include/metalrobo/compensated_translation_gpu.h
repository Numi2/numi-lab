#pragma once

#include "metalrobo/gpu_types.h"
#ifndef __METAL_VERSION__
#include <cmath>
#include <cstring>
#endif

// Authoritative floating-root translation. q.xyz remains the canonical FP32
// projection for source-compatible consumers. The immutable reference is the
// episode origin; displacement+correction is a normalized FP32 expansion in m.
// All owners of this header compile without reassociation / fast-math.
#define MR_COMPENSATED_TRANSLATION_ABI_VERSION 1u
typedef struct MR_ALIGN16 MRCompensatedRootTranslationGPU {
    mr_float4 reference;
    mr_float4 displacement;
    mr_float4 correction;
} MRCompensatedRootTranslationGPU;

struct MRCompensatedScalar { float high; float low; };

inline MRCompensatedScalar mrCompensatedSum(float a, float b) {
    const float high = a + b;
    const float virtualB = high - a;
    const float low = (a - (high - virtualB)) + (b - virtualB);
    return {high, low};
}

inline float mrCompensatedRoundOdd(MRCompensatedScalar value);

inline MRCompensatedScalar mrCompensatedAdd(
    MRCompensatedScalar a, MRCompensatedScalar b) {
    const auto high = mrCompensatedSum(a.high, b.high);
    const auto low = mrCompensatedSum(a.low, b.low);
    const auto middle = mrCompensatedSum(high.low, low.high);
    const auto merged = mrCompensatedSum(high.high, middle.high);
    const auto tail0 = mrCompensatedSum(merged.low, middle.low);
    const auto tail1 = mrCompensatedSum(tail0.high, low.low);
    const float sticky = mrCompensatedRoundOdd(mrCompensatedSum(tail1.low, tail0.low));
    const float tail = mrCompensatedRoundOdd(mrCompensatedSum(tail1.high, sticky));
    return mrCompensatedSum(merged.high, tail);
}

inline float mrCompensatedFMA(float a, float b, float c) {
#ifdef __METAL_VERSION__
    return fma(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

inline MRCompensatedScalar mrCompensatedProduct(float a, float b) {
    const float high = a * b;
    return {high, mrCompensatedFMA(a, b, -high)};
}

inline bool mrCompensatedFinite(float value) {
#ifdef __METAL_VERSION__
    return isfinite(value);
#else
    return std::isfinite(value);
#endif
}

inline mr_u32 mrCompensatedBits(float value) {
#ifdef __METAL_VERSION__
    return as_type<uint>(value);
#else
    mr_u32 bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
#endif
}

inline MRCompensatedScalar mrCompensatedPositionScalar(
    float reference, float displacement, float correction, float local) {
    // Preserve legacy signed-zero q bits at the unadvanced episode origin.
    if (displacement == 0.0f && correction == 0.0f && local == 0.0f)
        return {reference, 0.0f};
    return mrCompensatedAdd(mrCompensatedSum(reference, local),
                           {displacement, correction});
}

struct MRCompensatedPositionGPU { mr_float4 high; mr_float4 low; };
inline MRCompensatedPositionGPU mrCompensatedTranslationPosition(
    const MRCompensatedRootTranslationGPU value, mr_float4 local) {
    const auto x = mrCompensatedPositionScalar(value.reference.x, value.displacement.x,
                                              value.correction.x, local.x);
    const auto y = mrCompensatedPositionScalar(value.reference.y, value.displacement.y,
                                              value.correction.y, local.y);
    const auto z = mrCompensatedPositionScalar(value.reference.z, value.displacement.z,
                                              value.correction.z, local.z);
    return {{x.high, y.high, z.high, 0.0f}, {x.low, y.low, z.low, 0.0f}};
}

// Round an error-free two-term expansion to odd. Retaining the sticky bit
// prevents double rounding at the final projection's exact halfway case.
inline float mrCompensatedRoundOdd(MRCompensatedScalar value) {
    if (value.low == 0.0f || (mrCompensatedBits(value.high) & 1u) != 0u)
        return value.high;
#ifdef __METAL_VERSION__
    return nextafter(value.high, value.low > 0.0f ? INFINITY : -INFINITY);
#else
    return std::nextafter(value.high, value.low > 0.0f ? INFINITY : -INFINITY);
#endif
}
inline float mrCompensatedProjectionScalar(float reference, float displacement, float correction) {
    if (displacement == 0.0f && correction == 0.0f) return reference;
    const auto first = mrCompensatedSum(reference, displacement);
    const auto second = mrCompensatedSum(first.low, correction);
    const auto third = mrCompensatedSum(first.high, second.high);
    const float tail = mrCompensatedRoundOdd(mrCompensatedSum(third.low, second.low));
    return third.high + tail;
}
inline mr_float4 mrCompensatedTranslationProjection(
    const MRCompensatedRootTranslationGPU value) {
    return {
        mrCompensatedProjectionScalar(value.reference.x, value.displacement.x, value.correction.x),
        mrCompensatedProjectionScalar(value.reference.y, value.displacement.y, value.correction.y),
        mrCompensatedProjectionScalar(value.reference.z, value.displacement.z, value.correction.z),
        0.0f};
}

inline MRCompensatedRootTranslationGPU mrCompensatedTranslationFromProjection(
    mr_float4 position) {
    position.w = 0.0f;
    return {position, {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
}

inline bool mrCompensatedCanonicalScalar(float high, float low) {
    if (!mrCompensatedFinite(high) || !mrCompensatedFinite(low)) return false;
    const auto pair = mrCompensatedSum(high, low);
    return mrCompensatedBits(pair.high) == mrCompensatedBits(high) &&
           mrCompensatedBits(pair.low) == mrCompensatedBits(low);
}

inline bool mrCompensatedTranslationValid(
    const MRCompensatedRootTranslationGPU value) {
    const auto projected = mrCompensatedTranslationProjection(value);
    return mrCompensatedFinite(value.reference.x) &&
        mrCompensatedFinite(value.reference.y) &&
        mrCompensatedFinite(value.reference.z) &&
        mrCompensatedBits(value.reference.w) == 0u &&
        mrCompensatedBits(value.displacement.w) == 0u &&
        mrCompensatedBits(value.correction.w) == 0u &&
        mrCompensatedCanonicalScalar(value.displacement.x, value.correction.x) &&
        mrCompensatedCanonicalScalar(value.displacement.y, value.correction.y) &&
        mrCompensatedCanonicalScalar(value.displacement.z, value.correction.z) &&
        mrCompensatedFinite(projected.x) && mrCompensatedFinite(projected.y) &&
        mrCompensatedFinite(projected.z);
}

inline MRCompensatedRootTranslationGPU mrCompensatedTranslationAdvance(
    const MRCompensatedRootTranslationGPU source, mr_float4 velocity, float dt) {
    MRCompensatedRootTranslationGPU result = source;
    const auto x = mrCompensatedAdd({source.displacement.x, source.correction.x},
                                    mrCompensatedProduct(dt, velocity.x));
    const auto y = mrCompensatedAdd({source.displacement.y, source.correction.y},
                                    mrCompensatedProduct(dt, velocity.y));
    const auto z = mrCompensatedAdd({source.displacement.z, source.correction.z},
                                    mrCompensatedProduct(dt, velocity.z));
    result.displacement = {x.high, y.high, z.high, 0.0f};
    result.correction = {x.low, y.low, z.low, 0.0f};
    return result;
}

inline mr_float4 mrCompensatedPositionDifference(mr_float4 ah, mr_float4 al,
                                                 mr_float4 bh, mr_float4 bl) {
    const auto x = mrCompensatedAdd(mrCompensatedSum(ah.x, -bh.x), mrCompensatedSum(al.x, -bl.x));
    const auto y = mrCompensatedAdd(mrCompensatedSum(ah.y, -bh.y), mrCompensatedSum(al.y, -bl.y));
    const auto z = mrCompensatedAdd(mrCompensatedSum(ah.z, -bh.z), mrCompensatedSum(al.z, -bl.z));
    return {x.high, y.high, z.high, 0.0f};
}

#ifndef __METAL_VERSION__
static_assert(sizeof(MRCompensatedRootTranslationGPU) == 48u);
static_assert(alignof(MRCompensatedRootTranslationGPU) == 16u);
#endif
