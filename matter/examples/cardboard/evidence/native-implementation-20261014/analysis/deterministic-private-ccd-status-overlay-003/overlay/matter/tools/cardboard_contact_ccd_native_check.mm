#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "numi/matter/shared.h"

#ifndef NUMI_MATTER_METALLIB
#error "CMake must define NUMI_MATTER_METALLIB"
#endif

static_assert(sizeof(NMMatterStatusGPU) == 48u);
static_assert(alignof(NMMatterStatusGPU) == 16u);

namespace {
using Float4 = std::array<float, 4>;
using Point = std::array<double, 3>;

constexpr std::uint32_t kVertexTriangle = 0u;
constexpr std::uint32_t kEdgeEdge = 1u;
constexpr float kThickness = 1.0e-3f;
constexpr double kRelativeSampleTolerance = 2.0e-3;
constexpr double kMinimumSampleTolerance = 1.0e-14;
constexpr std::uint32_t kSamples = 10000u;
constexpr char kCapturedRaySha256[] =
    "9b50eef2b182f38e061eff23eaa46d55b158539fa579340280b9f801a1f68bc7";
constexpr char kCapturedEEInputCorpusSha256[] =
    "e116911335b4fbf88683535cde08a97b26ca99bc1975989d67532196c2ff996f";
constexpr char kCapturedEERecordSha256[] =
    "41d9c804684188137c5576f40b4d3e099a3f11184ed02b78e84376d244ed5f78";
constexpr char kBaseline367ResultSha256[] =
    "8b9b1ef8b0a32adb3938a39254ac57698fceb7b19265bae3fa69329562b3b6cc";
constexpr char kContinuation372ResultSha256[] =
    "c8fd27646a56d117931552dbe99a3c119976ea7061d53365e24080af81bda45b";
unsigned checks = 0u;

void require(const bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

void writeJsonFiniteOrNull(std::ostream& output, const double value) {
    if (std::isfinite(value)) output << value;
    else output << "null";
}

struct Case {
    std::string name;
    std::uint32_t kind = kVertexTriangle;
    float thickness = kThickness;
    // 0/1 selects direct VT/EE production CCD; 2 selects the production
    // certified Newton sweep wrapper. The geometry stores four start and
    // four end points, exactly as nm_contact_newton_ccd_regression consumes.
    float selector = 0.0f;
    std::array<Point, 4> start{};
    std::array<Point, 4> end{};
    bool expectedHit = false;
    bool expectUncertified = false;
    bool expectDirectUncertified = false;
    bool expectShortenedCertificate = false;
    bool expectRoundoffBandHit = false;
    bool verifySampledDistance = true;
};

Case vertexTriangleMiss(const float selector = 0.0f) {
    Case c;
    c.name = selector >= 2.0f ? "helper_vt_certified_miss" : "direct_vt_certified_miss";
    c.selector = selector;
    c.start = {{{0.25, 0.25, 0.10}, {0.0, 0.0, 0.0},
                {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}};
    c.end = c.start;
    return c;
}

Case vertexTriangleEndpointClearCross(const float selector = 0.0f) {
    Case c;
    c.name = selector >= 2.0f ? "helper_vt_endpoint_clear_cross" : "direct_vt_endpoint_clear_cross";
    c.selector = selector;
    c.expectedHit = true;
    c.start = {{{0.25, 0.25, 0.10}, {0.0, 0.0, 0.0},
                {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}};
    c.end = {{{0.25, 0.25, -0.10}, {0.0, 0.0, 0.0},
              {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}};
    return c;
}

Case edgeEdgeEndpointClearCross(const float selector = 1.0f) {
    Case c;
    c.name = selector >= 2.0f ? "helper_ee_endpoint_clear_cross" : "direct_ee_endpoint_clear_cross";
    c.kind = kEdgeEdge;
    c.selector = selector;
    c.expectedHit = true;
    c.start = {{{-0.1, 0.0, 0.0}, {0.1, 0.0, 0.0},
                {0.0, -0.1, 0.1}, {0.0, 0.1, 0.1}}};
    c.end = {{{-0.1, 0.0, 0.0}, {0.1, 0.0, 0.0},
              {0.0, -0.1, -0.1}, {0.0, 0.1, -0.1}}};
    return c;
}

Case edgeEdgeNearParallelLongTangent(const float selector = 1.0f) {
    Case c;
    c.name = selector >= 2.0f ? "helper_ee_nearparallel_long_tangent_miss" :
        "direct_ee_nearparallel_long_tangent_miss";
    c.kind = kEdgeEdge;
    c.selector = selector;
    c.thickness = 1.0e-3f;
    c.start = {{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
                {0.0, 0.02, 0.0}, {1.0, 0.02, 0.0}}};
    c.end = {{{100.0, 0.0, 0.0}, {101.0, 0.0, 0.0},
              {0.0, 0.02, 0.0}, {1.0, 0.02, 0.0}}};
    return c;
}

float floatFromBits(const std::uint32_t bits) {
    float value = 0.0f;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

Point pointFromBits(const std::uint32_t x, const std::uint32_t y,
                    const std::uint32_t z) {
    return {floatFromBits(x), floatFromBits(y), floatFromBits(z)};
}

Case capturedNearBandEdgeEdgeExhaustion(const float selector) {
    // Exact FP32 input for source103 capture pair (primitive 92,127), case
    // trial358_ee_edge0_edge0. Source103 bound17 exhausted at iteration16;
    // bound33 reached a certified hit on evaluation17. The returned point is
    // inside the implementation's roundoff-expanded band, not a robust
    // nominal-thickness crossing. This is a CCD continuation regression only.
    Case c;
    c.name = selector >= 2.0f
        ? "captured_ee_roundoff_band_17_to_33_helper"
        : "captured_ee_roundoff_band_17_to_33_direct";
    c.kind = kEdgeEdge;
    c.selector = selector;
    c.thickness = floatFromBits(0x3727c5acu);
    c.expectedHit = true;
    c.expectRoundoffBandHit = true;
    c.verifySampledDistance = false;
    c.start = {{
        pointFromBits(0x3b031297u, 0x3236273cu, 0x3a831ea8u),
        pointFromBits(0x3b03126au, 0x3120b246u, 0x34dbee94u),
        pointFromBits(0xb3d85123u, 0xbb03137eu, 0x3a846bdfu),
        pointFromBits(0x3b0310a7u, 0xb3794115u, 0x3a84703bu),
    }};
    c.end = {{
        pointFromBits(0x3b0312a4u, 0x325a5b76u, 0x3a8321aeu),
        pointFromBits(0x3b031269u, 0x314b9a59u, 0x350b6ba8u),
        pointFromBits(0xb3d8283bu, 0xbb03137fu, 0x3a846d2fu),
        pointFromBits(0x3b0310a7u, 0xb379acd4u, 0x3a84712bu),
    }};
    return c;
}

float roundoffExpandedThickness(const Case& c) {
    float coordinateScale = 0.0f;
    for (const auto* points : {&c.start, &c.end}) {
        for (const Point& p : *points) {
            coordinateScale = std::max(coordinateScale,
                std::fabs(static_cast<float>(p[0])));
            coordinateScale = std::max(coordinateScale,
                std::fabs(static_cast<float>(p[1])));
            coordinateScale = std::max(coordinateScale,
                std::fabs(static_cast<float>(p[2])));
        }
    }
    const float scale = std::max(std::max(c.thickness, coordinateScale), 1.0e-12f);
    const float roundoff = 8.0f * std::numeric_limits<float>::epsilon() * scale;
    return c.thickness + roundoff;
}

void checkRoundoffBandHit(const Case& c, const Float4& result) {
    if (!c.expectRoundoffBandHit) return;
    const float effectiveThickness = roundoffExpandedThickness(c);
    require(result[0] == 1.0f && result[1] == 1.0f,
        c.name + ": exact captured ray was not certified as a hit at bound33");
    require(result[3] >= c.thickness && result[3] <= effectiveThickness,
        c.name + ": returned separation left nominal/roundoff band bounds");
    require(result[2] >= 0.95f && result[2] <= 1.0f,
        c.name + ": returned hit is outside the captured terminal-time region");
}

Case degenerateCollinearCross(const float selector = 0.0f) {
    Case c;
    c.name = selector >= 2.0f ? "helper_degenerate_collinear_vt_cross" :
        "direct_degenerate_collinear_vt_cross";
    c.selector = selector;
    c.expectedHit = true;
    c.start = {{{0.5, 0.0, 0.10}, {0.0, 0.0, 0.0},
                {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}}};
    c.end = {{{0.5, 0.0, -0.10}, {0.0, 0.0, 0.0},
              {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}}};
    return c;
}

Case helperRejectsNonfiniteRay() {
    Case c = vertexTriangleMiss(2.0f);
    c.name = "helper_nonfinite_ray_rejects_uncertified";
    c.start[0][0] = std::numeric_limits<double>::quiet_NaN();
    c.end[0][0] = std::numeric_limits<double>::quiet_NaN();
    c.expectedHit = false;
    c.expectUncertified = true;
    c.verifySampledDistance = false;
    return c;
}

Case capturedUncertifiedClearRay(const float selector) {
    // Exact FP32 endpoints from contact-regional-all-pair-gate-092 / slot129.
    // The originating fixture is explicitly invalid at rest (the duplicated
    // point lies on an exterior triangle), so this is a bounded CCD regression
    // vector only, not a geometry or self-contact qualification specimen.
    Case c;
    c.name = selector >= 2.0f
        ? "captured_vt_clear_ray_helper_shortened_certificate"
        : "captured_vt_clear_ray_direct_uncertified";
    c.selector = selector;
    // Reconstruct the production value from the captured coordinate scale;
    // the source JSON's six-significant-digit trace string is rounded.
    c.thickness = 16.0f * 1.1920928955078125e-7f * 0.002000007778406143f;
    c.expectDirectUncertified = selector < 2.0f;
    c.expectShortenedCertificate = selector >= 2.0f;
    // The independent million-sample report is retained as a measurement,
    // not used as a clearance proof for this invalid-source fixture.
    c.verifySampledDistance = false;
    c.start = {{{-9.406078346763103e-12, -0.0019999935757368803, 0.0010050003184005618},
                {-3.431717410240509e-11, -0.00200000056065619, 0.0009999999310821295},
                {1.200403166624897e-12, -0.0020000000949949026, 0.0020000000949949026},
                {0.0020000000949949026, -0.0020000000949949026, 0.0020000000949949026}}};
    c.end = {{{-8.846265386885932e-10, -0.0019999847281724215, 0.0010058307088911533},
              {5.361737276388112e-09, -0.0019999989308416843, 0.0009998204186558723},
              {2.6752107196159614e-09, -0.0019999693613499403, 0.001999743515625596},
              {0.002000007778406143, -0.001999984495341778, 0.0019999067299067974}}};
    return c;
}

std::vector<Float4> pack(const std::vector<Case>& cases) {
    std::vector<Float4> result;
    result.reserve(cases.size() * 9u);
    for (const Case& c : cases) {
        result.push_back({static_cast<float>(c.kind), c.thickness, c.selector, 0.0f});
        for (const Point& p : c.start)
            result.push_back({static_cast<float>(p[0]), static_cast<float>(p[1]),
                              static_cast<float>(p[2]), 0.0f});
        for (const Point& p : c.end)
            result.push_back({static_cast<float>(p[0]), static_cast<float>(p[1]),
                              static_cast<float>(p[2]), 0.0f});
    }
    return result;
}

std::string sha256(const std::vector<Float4>& values) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]{};
    require(CC_SHA256(reinterpret_cast<const unsigned char*>(values.data()),
        static_cast<CC_LONG>(values.size() * sizeof(Float4)), digest) != nullptr,
        "case-corpus SHA256 failed");
    static constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(2u * CC_SHA256_DIGEST_LENGTH);
    for (const unsigned char value : digest) {
        result.push_back(hex[value >> 4]);
        result.push_back(hex[value & 0xf]);
    }
    return result;
}

Point add(const Point& a, const Point& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
Point subtract(const Point& a, const Point& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
Point multiply(const double s, const Point& a) {
    return {s * a[0], s * a[1], s * a[2]};
}
double dot(const Point& a, const Point& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
double length(const Point& a) { return std::sqrt(dot(a, a)); }
Point lerp(const Point& a, const Point& b, const double t) {
    return add(a, multiply(t, subtract(b, a)));
}

Point closestPointOnTriangle(const Point& p, const Point& a,
                             const Point& b, const Point& c) {
    const Point ab = subtract(b, a), ac = subtract(c, a), ap = subtract(p, a);
    const double d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return a;
    const Point bp = subtract(p, b);
    const double d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return b;
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
        return add(a, multiply(d1 / std::max(d1 - d3, 1.0e-30), ab));
    const Point cp = subtract(p, c);
    const double d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return c;
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
        return add(a, multiply(d2 / std::max(d2 - d6, 1.0e-30), ac));
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0) {
        const double w = (d4 - d3) /
            std::max((d4 - d3) + (d5 - d6), 1.0e-30);
        return add(b, multiply(w, subtract(c, b)));
    }
    const double denominator = std::max(va + vb + vc, 1.0e-30);
    const double v = vb / denominator, w = vc / denominator;
    return add(a, add(multiply(v, ab), multiply(w, ac)));
}

double segmentSegmentDistance(const Point& p0, const Point& p1,
                              const Point& q0, const Point& q1) {
    const Point dP = subtract(p1, p0), dQ = subtract(q1, q0), r = subtract(p0, q0);
    const double a = dot(dP, dP), e = dot(dQ, dQ), f = dot(dQ, r);
    double s = 0.0, t = 0.0;
    if (a <= 1.0e-30 && e <= 1.0e-30) {
        s = t = 0.0;
    } else if (a <= 1.0e-30) {
        t = std::clamp(f / std::max(e, 1.0e-30), 0.0, 1.0);
    } else {
        const double c = dot(dP, r);
        if (e <= 1.0e-30) {
            s = std::clamp(-c / std::max(a, 1.0e-30), 0.0, 1.0);
        } else {
            const double b = dot(dP, dQ), denominator = a * e - b * b;
            s = denominator > 1.0e-30
                ? std::clamp((b * f - c * e) / denominator, 0.0, 1.0) : 0.0;
            t = (b * s + f) / e;
            if (t < 0.0) { t = 0.0; s = std::clamp(-c / a, 0.0, 1.0); }
            else if (t > 1.0) {
                t = 1.0; s = std::clamp((b - c) / a, 0.0, 1.0);
            }
        }
    }
    return length(subtract(add(p0, multiply(s, dP)), add(q0, multiply(t, dQ))));
}

double sampledMinimum(const Case& c, const double pathFraction = 1.0) {
    double minimum = std::numeric_limits<double>::infinity();
    std::array<Point, 4> start{}, end{};
    for (unsigned vertex = 0u; vertex < 4u; ++vertex) {
        for (unsigned axis = 0u; axis < 3u; ++axis) {
            start[vertex][axis] = static_cast<double>(
                static_cast<float>(c.start[vertex][axis]));
            end[vertex][axis] = static_cast<double>(
                static_cast<float>(c.end[vertex][axis]));
        }
    }
    for (std::uint32_t sample = 0u; sample <= kSamples; ++sample) {
        const double t = pathFraction * static_cast<double>(sample) / kSamples;
        std::array<Point, 4> p{};
        for (unsigned i = 0u; i < 4u; ++i) p[i] = lerp(start[i], end[i], t);
        double distance = 0.0;
        if (c.kind == kVertexTriangle) {
            distance = length(subtract(p[0], closestPointOnTriangle(p[0], p[1], p[2], p[3])));
        } else {
            distance = segmentSegmentDistance(p[0], p[1], p[2], p[3]);
        }
        minimum = std::min(minimum, distance);
    }
    return minimum;
}

id<MTLBuffer> makeBuffer(id<MTLDevice> device, const void* bytes,
                         const std::size_t byteCount) {
    id<MTLBuffer> buffer = [device newBufferWithBytes:bytes
        length:byteCount options:MTLResourceStorageModeShared];
    require(buffer != nil, "Metal contact regression buffer allocation failed");
    return buffer;
}

id<MTLComputePipelineState> makePipeline(id<MTLDevice> device,
                                         id<MTLLibrary> library,
                                         NSString* name) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    require(function != nil,
        std::string("required production contact kernel missing: ") +
            [name UTF8String]);
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    require(pipeline != nil, error != nil
        ? std::string("contact kernel pipeline creation failed: ") +
            [[error localizedDescription] UTF8String]
        : "contact kernel pipeline creation failed");
    return pipeline;
}

std::vector<Float4> run(id<MTLDevice> device, id<MTLCommandQueue> queue,
                        id<MTLComputePipelineState> pipeline,
                        const std::vector<Float4>& input,
                        const std::uint32_t count,
                        const std::uint32_t outputsPerCase) {
    const std::uint32_t inputCount = count;
    std::vector<Float4> output(std::size_t(count) * outputsPerCase, Float4{});
    id<MTLBuffer> inputBuffer = makeBuffer(device, input.data(), input.size() * sizeof(Float4));
    id<MTLBuffer> outputBuffer = makeBuffer(device, output.data(), output.size() * sizeof(Float4));
    id<MTLCommandBuffer> command = [queue commandBuffer];
    require(command != nil, "CCD command buffer creation failed");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    require(encoder != nil, "CCD command encoder creation failed");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:inputBuffer offset:0u atIndex:0u];
    [encoder setBuffer:outputBuffer offset:0u atIndex:1u];
    [encoder setBytes:&inputCount length:sizeof(inputCount) atIndex:2u];
    const NSUInteger width = std::max<NSUInteger>(1u, std::min<NSUInteger>(
        pipeline.maxTotalThreadsPerThreadgroup, 32u));
    [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
        command.error != nil
            ? std::string("CCD command failed: ") +
                [[command.error localizedDescription] UTF8String]
            : "CCD command failed");
    std::memcpy(output.data(), outputBuffer.contents, output.size() * sizeof(Float4));
    return output;
}

NMMatterStatusGPU statusRecord(
    const std::uint32_t code,
    const std::uint32_t environment,
    const std::uint32_t objectIndex,
    const std::uint32_t failingIndex,
    const std::uint32_t completedMicrosteps,
    const std::uint32_t fgmresIterations,
    const std::uint32_t contactCount,
    const std::uint32_t eventCount,
    const nm_float4 diagnostics
) {
    NMMatterStatusGPU value{};
    value.code = code;
    value.environment = environment;
    value.objectIndex = objectIndex;
    value.failingIndex = failingIndex;
    value.completedMicrosteps = completedMicrosteps;
    value.fgmresIterations = fgmresIterations;
    value.contactCount = contactCount;
    value.eventCount = eventCount;
    value.diagnostics = diagnostics;
    return value;
}

void checkStatusTuple(const NMMatterStatusGPU& value,
                      const std::uint32_t code,
                      const std::uint32_t objectIndex,
                      const std::uint32_t failingIndex,
                      const nm_float4 diagnostics,
                      const std::string& name) {
    require(value.code == code && value.objectIndex == objectIndex &&
            value.failingIndex == failingIndex &&
            value.diagnostics.x == diagnostics.x &&
            value.diagnostics.y == diagnostics.y &&
            value.diagnostics.z == diagnostics.z &&
            value.diagnostics.w == diagnostics.w,
            name + ": production reducer selected or copied the wrong status tuple");
}

bool byteIdenticalStatus(const NMMatterStatusGPU& a,
                         const NMMatterStatusGPU& b) {
    return std::memcmp(&a, &b, sizeof(NMMatterStatusGPU)) == 0;
}

void checkProductionStatusReducer(id<MTLDevice> device,
                                  id<MTLCommandQueue> queue,
                                  id<MTLComputePipelineState> pipeline) {
    NMMatterDispatchGPU dispatch{};
    dispatch.abiVersion = NM_MATTER_ABI_VERSION;
    dispatch.environmentCount = 4u;
    dispatch.deformableContactCapacity = 4u;
    const std::vector<std::uint32_t> candidateCounts{4u, 3u, 0u, 2u};
    std::vector<NMMatterStatusGPU> initialStatuses{
        statusRecord(NM_STATUS_SUCCESS, 0u, 71u, 72u, 7u, 8u, 9u, 10u,
                     nm_float4{1.0f, 2.0f, 3.0f, 4.0f}),
        statusRecord(NM_STATUS_NONFINITE_INPUT, 1u, 81u, 82u, 11u, 12u, 13u, 14u,
                     nm_float4{5.0f, 6.0f, 7.0f, 8.0f}),
        statusRecord(NM_STATUS_SUCCESS, 2u, 91u, 92u, 15u, 16u, 17u, 18u,
                     nm_float4{9.0f, 10.0f, 11.0f, 12.0f}),
        statusRecord(NM_STATUS_SUCCESS, 3u, 101u, 102u, 19u, 20u, 21u, 22u,
                     nm_float4{13.0f, 14.0f, 15.0f, 16.0f}),
    };
    std::vector<NMMatterStatusGPU> candidateStatuses(
        dispatch.environmentCount * dispatch.deformableContactCapacity);
    for (std::uint32_t environment = 0u;
         environment < dispatch.environmentCount; ++environment) {
        for (std::uint32_t slot = 0u;
             slot < dispatch.deformableContactCapacity; ++slot) {
            candidateStatuses[environment * dispatch.deformableContactCapacity + slot] =
                statusRecord(NM_STATUS_SUCCESS, environment, 200u + slot, 300u + slot,
                             31u + slot, 41u + slot, 51u + slot, 61u + slot,
                             nm_float4{float(environment), float(slot), 0.0f, 0.0f});
        }
    }
    const nm_float4 slotOneDiagnostics{24.0f, 127.0f, 1.0f, 0.0f};
    const nm_float4 slotThreeDiagnostics{92.0f, 127.0f, 0.0f, 0.0f};
    candidateStatuses[1u] = statusRecord(
        NM_STATUS_CONTACT_FAILURE, 0u, 240u, 1u, 0u, 0u, 0u, 0u,
        slotOneDiagnostics);
    candidateStatuses[3u] = statusRecord(
        NM_STATUS_CONTACT_FAILURE, 0u, 920u, 3u, 0u, 0u, 0u, 0u,
        slotThreeDiagnostics);
    // A private failure cannot replace an earlier phase failure.
    candidateStatuses[4u] = statusRecord(
        NM_STATUS_CONTACT_FAILURE, 1u, 241u, 0u, 0u, 0u, 0u, 0u,
        slotOneDiagnostics);
    // This stale failure lies beyond environment 2's empty active prefix.
    candidateStatuses[8u] = statusRecord(
        NM_STATUS_CONTACT_FAILURE, 2u, 242u, 0u, 0u, 0u, 0u, 0u,
        slotOneDiagnostics);
    // This stale failure lies beyond environment 3's two healthy candidates.
    candidateStatuses[15u] = statusRecord(
        NM_STATUS_CONTACT_FAILURE, 3u, 243u, 3u, 0u, 0u, 0u, 0u,
        slotThreeDiagnostics);

    const auto dispatchBuffer = makeBuffer(device, &dispatch, sizeof(dispatch));
    const auto countBuffer = makeBuffer(device, candidateCounts.data(),
        candidateCounts.size() * sizeof(candidateCounts.front()));
    const auto candidateBuffer = makeBuffer(device, candidateStatuses.data(),
        candidateStatuses.size() * sizeof(candidateStatuses.front()));
    const auto statusBuffer = makeBuffer(device, initialStatuses.data(),
        initialStatuses.size() * sizeof(initialStatuses.front()));
    id<MTLCommandBuffer> command = [queue commandBuffer];
    require(command != nil, "status reducer command buffer creation failed");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    require(encoder != nil, "status reducer command encoder creation failed");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
    [encoder setBuffer:countBuffer offset:0u atIndex:1u];
    [encoder setBuffer:candidateBuffer offset:0u atIndex:2u];
    [encoder setBuffer:statusBuffer offset:0u atIndex:3u];
    const NSUInteger width = std::max<NSUInteger>(1u, std::min<NSUInteger>(
        pipeline.maxTotalThreadsPerThreadgroup, 32u));
    [encoder dispatchThreads:MTLSizeMake(dispatch.environmentCount, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
        command.error != nil
            ? std::string("production status reducer command failed: ") +
                [[command.error localizedDescription] UTF8String]
            : "production status reducer command failed");

    const auto* result = static_cast<const NMMatterStatusGPU*>(statusBuffer.contents);
    checkStatusTuple(result[0], NM_STATUS_CONTACT_FAILURE, 240u, 1u,
                     slotOneDiagnostics, "environment0/minimum-slot");
    require(result[0].environment == initialStatuses[0].environment &&
            result[0].completedMicrosteps == initialStatuses[0].completedMicrosteps &&
            result[0].fgmresIterations == initialStatuses[0].fgmresIterations &&
            result[0].contactCount == initialStatuses[0].contactCount &&
            result[0].eventCount == initialStatuses[0].eventCount,
            "status reducer overwrote completed-microstep/FGMRES/contact/event counters");
    require(byteIdenticalStatus(result[1], initialStatuses[1]),
            "status reducer overwrote an earlier environment failure");
    require(byteIdenticalStatus(result[2], initialStatuses[2]),
            "empty active candidate prefix selected stale candidate status");
    require(byteIdenticalStatus(result[3], initialStatuses[3]),
            "healthy environment selected a stale failure beyond its active prefix");
}

void checkProductionNarrowphaseClearsCandidateStatuses(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    id<MTLComputePipelineState> pipeline
) {
    NMMatterDispatchGPU dispatch{};
    dispatch.abiVersion = NM_MATTER_ABI_VERSION;
    dispatch.environmentCount = 2u;
    dispatch.objectCount = 1u;
    dispatch.deformableContactCapacity = 3u;
    NMMicrostepGPU microstep{};
    const std::vector<std::uint32_t> candidateCounts{1u, 0u};
    const std::size_t total = std::size_t(dispatch.environmentCount) *
        dispatch.deformableContactCapacity;
    const std::vector<NMContinuumObjectGPU> objects(1u);
    const std::vector<NMMaterialGPU> materials(1u);
    const std::vector<NMFEMNodeStateGPU> femNodes(1u);
    const std::vector<NMSchedulerStateGPU> schedulers(1u);
    const std::vector<NMGridNodeStateGPU> gridNodes(1u);
    const std::vector<NMContinuumSurfacePrimitiveGPU> primitives(1u);
    // identity.w is zero, so each counted row exits as inactive after the
    // production kernel clears its private candidate-status slot.
    const std::vector<NMDeformableContactCandidateGPU> candidates(total);
    const std::vector<NMDeformableContactHistoryGPU> histories(total);
    std::vector<NMDeformableContactGPU> contacts(total);
    std::vector<NMMatterStatusGPU> statuses(dispatch.environmentCount);
    for (std::uint32_t environment = 0u;
         environment < dispatch.environmentCount; ++environment) {
        statuses[environment] = statusRecord(
            NM_STATUS_SUCCESS, environment, 0u, 0u, 0u, 0u, 0u, 0u,
            nm_float4{0.0f, 0.0f, 0.0f, 0.0f});
    }
    std::vector<NMMatterStatusGPU> candidateStatuses(total);
    for (std::size_t slot = 0u; slot < total; ++slot) {
        candidateStatuses[slot] = statusRecord(
            NM_STATUS_CONTACT_FAILURE, static_cast<std::uint32_t>(slot / 3u),
            500u + static_cast<std::uint32_t>(slot),
            600u + static_cast<std::uint32_t>(slot), 7u, 8u, 9u, 10u,
            nm_float4{1.0f, 2.0f, 3.0f, 4.0f});
    }
    const std::vector<NMIncidenceRangeGPU> cohesiveRanges(1u);
    const std::vector<nm_uint4> cohesiveRefs(1u);
    const std::vector<NMCohesiveFaceGPU> cohesiveFaces(1u);

    const auto objectBuffer = makeBuffer(device, objects.data(), sizeof(objects.front()));
    const auto materialBuffer = makeBuffer(device, materials.data(), sizeof(materials.front()));
    const auto femNodeBuffer = makeBuffer(device, femNodes.data(), sizeof(femNodes.front()));
    const auto schedulerBuffer = makeBuffer(device, schedulers.data(), sizeof(schedulers.front()));
    const auto gridNodeBuffer = makeBuffer(device, gridNodes.data(), sizeof(gridNodes.front()));
    const auto primitiveBuffer = makeBuffer(device, primitives.data(), sizeof(primitives.front()));
    const auto candidateInputBuffer = makeBuffer(device, candidates.data(),
        candidates.size() * sizeof(candidates.front()));
    const auto countBuffer = makeBuffer(device, candidateCounts.data(),
        candidateCounts.size() * sizeof(candidateCounts.front()));
    const auto historyBuffer = makeBuffer(device, histories.data(),
        histories.size() * sizeof(histories.front()));
    const auto contactBuffer = makeBuffer(device, contacts.data(),
        contacts.size() * sizeof(contacts.front()));
    const auto statusBuffer = makeBuffer(device, statuses.data(),
        statuses.size() * sizeof(statuses.front()));
    const auto rangeBuffer = makeBuffer(device, cohesiveRanges.data(),
        sizeof(cohesiveRanges.front()));
    const auto refsBuffer = makeBuffer(device, cohesiveRefs.data(), sizeof(cohesiveRefs.front()));
    const auto faceBuffer = makeBuffer(device, cohesiveFaces.data(), sizeof(cohesiveFaces.front()));
    const auto candidateStatusBuffer = makeBuffer(device, candidateStatuses.data(),
        candidateStatuses.size() * sizeof(candidateStatuses.front()));
    const auto dispatchBuffer = makeBuffer(device, &dispatch, sizeof(dispatch));
    const std::uint32_t deterministicSelection = 1u;

    id<MTLCommandBuffer> command = [queue commandBuffer];
    require(command != nil, "narrowphase clear command buffer creation failed");
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    require(encoder != nil, "narrowphase clear command encoder creation failed");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:dispatchBuffer offset:0u atIndex:0u];
    [encoder setBytes:&microstep length:sizeof(microstep) atIndex:1u];
    [encoder setBuffer:objectBuffer offset:0u atIndex:2u];
    [encoder setBuffer:materialBuffer offset:0u atIndex:3u];
    [encoder setBuffer:femNodeBuffer offset:0u atIndex:4u];
    [encoder setBuffer:schedulerBuffer offset:0u atIndex:5u];
    [encoder setBuffer:gridNodeBuffer offset:0u atIndex:6u];
    [encoder setBuffer:primitiveBuffer offset:0u atIndex:7u];
    [encoder setBuffer:candidateInputBuffer offset:0u atIndex:8u];
    [encoder setBuffer:countBuffer offset:0u atIndex:9u];
    [encoder setBuffer:historyBuffer offset:0u atIndex:10u];
    [encoder setBuffer:contactBuffer offset:0u atIndex:11u];
    [encoder setBuffer:statusBuffer offset:0u atIndex:12u];
    [encoder setBuffer:rangeBuffer offset:0u atIndex:13u];
    [encoder setBuffer:refsBuffer offset:0u atIndex:14u];
    [encoder setBuffer:faceBuffer offset:0u atIndex:15u];
    [encoder setBytes:&deterministicSelection length:sizeof(deterministicSelection)
               atIndex:16u];
    [encoder setBuffer:candidateStatusBuffer offset:0u atIndex:17u];
    const NSUInteger width = std::max<NSUInteger>(1u, std::min<NSUInteger>(
        pipeline.maxTotalThreadsPerThreadgroup, 32u));
    [encoder dispatchThreads:MTLSizeMake(total, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
        command.error != nil
            ? std::string("production narrowphase clear command failed: ") +
                [[command.error localizedDescription] UTF8String]
            : "production narrowphase clear command failed");

    const auto* cleared = static_cast<const NMMatterStatusGPU*>(candidateStatusBuffer.contents);
    const NMMatterStatusGPU zero{};
    for (std::size_t slot = 0u; slot < total; ++slot)
        require(byteIdenticalStatus(cleared[slot], zero),
                "production narrowphase did not clear a stale private status slot");
    const auto* outputStatuses = static_cast<const NMMatterStatusGPU*>(statusBuffer.contents);
    for (std::uint32_t environment = 0u;
         environment < dispatch.environmentCount; ++environment) {
        const NMMatterStatusGPU healthy = statusRecord(
            NM_STATUS_SUCCESS, environment, 0u, 0u, 0u, 0u, 0u, 0u,
            nm_float4{0.0f, 0.0f, 0.0f, 0.0f});
        require(byteIdenticalStatus(outputStatuses[environment], healthy),
                "inactive narrowphase status-reset specimen changed environment status");
    }
}

void checkDirect(const std::vector<Case>& cases,
                 const std::vector<Float4>& result) {
    for (std::size_t i = 0u; i < cases.size(); ++i) {
        const Case& c = cases[i];
        const Float4& r = result[i];
        if (c.expectDirectUncertified) {
            require(r[0] == 0.0f && r[1] == 0.0f,
                c.name + ": captured direct ray did not retain its uncertified miss");
        } else {
            require(r[1] == 1.0f, c.name + ": direct production CCD did not certify result");
        }
        require((r[0] == 1.0f) == c.expectedHit,
            c.name + ": production direct CCD hit/miss differs from authored case");
        require(std::isfinite(r[2]) && r[2] >= 0.0f && r[2] <= 1.0f,
            c.name + ": direct CCD TOI is outside the unit sweep");
        require(std::isfinite(r[3]), c.name + ": direct CCD separation is nonfinite");
        checkRoundoffBandHit(c, r);
        if (c.verifySampledDistance) {
            const double minimum = sampledMinimum(c);
            const double sampleTolerance = std::max(
                kMinimumSampleTolerance, c.thickness * kRelativeSampleTolerance);
            if (c.expectedHit) {
                require(minimum <= c.thickness + sampleTolerance,
                    c.name + ": independent sampled geometry did not cross the contact thickness");
            } else {
                require(minimum > c.thickness + sampleTolerance,
                    c.name + ": independent sampled geometry found an unreported hit");
            }
        }
    }
}

void checkNewton(const std::vector<Case>& cases,
                 const std::vector<Float4>& result) {
    for (std::size_t i = 0u; i < cases.size(); ++i) {
        const Case& c = cases[i];
        const Float4& r = result[2u * i];
        const Float4& sweep = result[2u * i + 1u];
        if (c.expectUncertified) {
            require(r[0] == 0.0f && r[1] == 0.0f,
                c.name + ": malformed ray was not rejected as uncertified");
            require(sweep[0] == 1.0f && sweep[1] == 0.0f,
                c.name + ": helper changed fraction/retry metadata before rejecting invalid input");
            continue;
        }
        require(r[1] == 1.0f, c.name + ": Newton helper returned an uncertified outcome");
        require((r[0] == 1.0f) == c.expectedHit,
            c.name + ": Newton helper hit/miss differs from direct production CCD");
        if (c.expectShortenedCertificate) {
            require(sweep[0] > 0.0f && sweep[0] < 1.0f &&
                    sweep[1] >= 1.0f && sweep[1] <= 16.0f,
                c.name + ": bounded retry did not certify a shorter prefix");
            require(sweep[0] == std::ldexp(1.0f, -static_cast<int>(sweep[1])),
                c.name + ": tested prefix does not match the helper retry count");
        } else {
            require(sweep[0] == 1.0f && sweep[1] == 0.0f,
                c.name + ": a certifiable full ray was shortened unexpectedly");
        }
        require(std::isfinite(r[2]) && r[2] >= 0.0f && r[2] <= 1.0f &&
                std::isfinite(r[3]), c.name + ": helper outcome contains invalid geometry");
        checkRoundoffBandHit(c, r);
        if (c.verifySampledDistance) {
            const double minimum = sampledMinimum(c, sweep[0]);
            const double sampleTolerance = std::max(
                kMinimumSampleTolerance, c.thickness * kRelativeSampleTolerance);
            require(c.expectedHit ? minimum <= c.thickness + sampleTolerance
                                  : minimum > c.thickness + sampleTolerance,
                c.name + ": independent sampled geometry disagrees with helper outcome");
        }
    }
}

} // namespace

int main() {
    @autoreleasepool {
        try {
            const std::vector<Case> directCases = {
                vertexTriangleMiss(), vertexTriangleEndpointClearCross(),
                edgeEdgeEndpointClearCross(), edgeEdgeNearParallelLongTangent(),
                capturedNearBandEdgeEdgeExhaustion(0.0f),
                degenerateCollinearCross(), capturedUncertifiedClearRay(0.0f),
            };
            std::vector<Case> helperCases = {
                vertexTriangleMiss(2.0f), vertexTriangleEndpointClearCross(2.0f),
                edgeEdgeEndpointClearCross(2.0f), edgeEdgeNearParallelLongTangent(2.0f),
                capturedNearBandEdgeEdgeExhaustion(2.0f),
                degenerateCollinearCross(2.0f),
                capturedUncertifiedClearRay(2.0f), helperRejectsNonfiniteRay(),
            };
            const std::vector<Float4> directInput = pack(directCases);
            const std::vector<Float4> helperInput = pack(helperCases);

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "native CCD check requires an Apple Metal device");
            NSError* error = nil;
            NSString* path = [NSString stringWithUTF8String:NUMI_MATTER_METALLIB];
            id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:path]
                error:&error];
            require(library != nil, error != nil
                ? std::string("cannot load Matter metallib: ") +
                    [[error localizedDescription] UTF8String]
                : "cannot load Matter metallib");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(queue != nil, "native CCD command queue unavailable");

            id<MTLComputePipelineState> directPipeline = makePipeline(device, library,
                @"numi_matter_metal::nm_contact_ccd_regression");
            id<MTLComputePipelineState> newtonPipeline = makePipeline(device, library,
                @"numi_matter_metal::nm_contact_newton_ccd_regression");
            id<MTLComputePipelineState> narrowphasePipeline = makePipeline(device, library,
                @"numi_matter_metal::nm_contact_narrowphase_deformable");
            id<MTLComputePipelineState> statusReducerPipeline = makePipeline(device, library,
                @"numi_matter_metal::nm_contact_reduce_deformable_narrowphase_failures");
            checkProductionNarrowphaseClearsCandidateStatuses(
                device, queue, narrowphasePipeline);
            checkProductionStatusReducer(device, queue, statusReducerPipeline);
            const std::vector<Float4> direct = run(device, queue, directPipeline,
                directInput, static_cast<std::uint32_t>(directCases.size()), 1u);
            const std::vector<Float4> helper = run(device, queue, newtonPipeline,
                helperInput, static_cast<std::uint32_t>(helperCases.size()), 2u);
            checkDirect(directCases, direct);
            checkNewton(helperCases, helper);

            std::cout << std::setprecision(9)
                << "{\"schema\":\"numi.matter.contact-ccd-native-regression.v2\","
                << "\"case_corpus_sha256\":\"" << sha256(directInput) << "\","
                << "\"helper_case_corpus_sha256\":\"" << sha256(helperInput) << "\","
                << "\"captured_diagnostic_ray_sha256\":\"" << kCapturedRaySha256 << "\","
                << "\"captured_diagnostic_ray_id\":\"contact-regional-all-pair-gate-092/slot129\","
                << "\"captured_diagnostic_ray_artifact\":\"full-contact-self-ccd-input-092-001.json\","
                << "\"captured_diagnostic_ray_fixture_initial_geometry_valid\":false,"
                << "\"captured_diagnostic_ray_thickness_source\":\"16*FLT_EPSILON*max_abs_endpoint_coordinate reconstructed from FP32 endpoints\","
                << "\"captured_diagnostic_ray_physical_qualification\":false,"
                << "\"captured_ee_input_corpus_sha256\":\"" << kCapturedEEInputCorpusSha256 << "\","
                << "\"captured_ee_case_record_sha256\":\"" << kCapturedEERecordSha256 << "\","
                << "\"captured_ee_iteration_comparison\":{"
                << "\"case\":\"trial358_ee_edge0_edge0\","
                << "\"baseline_source\":\"source103\","
                << "\"baseline_iteration_bound\":17,"
                << "\"baseline_terminal_iteration\":16,"
                << "\"baseline_certified\":false,"
                << "\"baseline_result_sha256\":\"" << kBaseline367ResultSha256 << "\","
                << "\"continuation_iteration_bound\":33,"
                << "\"continuation_terminal_iteration\":17,"
                << "\"continuation_certified\":true,"
                << "\"continuation_result_sha256\":\"" << kContinuation372ResultSha256 << "\","
                << "\"near_band_only_not_nominal_crossing\":true,"
                << "\"full_contact_same_build_required\":true},"
                << "\"captured_ee_ray_id\":\"source103/trial358_ee_edge0_edge0/pair92-127\","
                << "\"captured_ee_diagnostic_only\":true,"
                << "\"captured_ee_physical_qualification\":false,"
                << "\"direct_cases\":" << directCases.size() << ','
                << "\"helper_cases\":" << helperCases.size() << ','
                << "\"production_private_status_instrument\":{"
                << "\"narrowphase_kernel\":\"nm_contact_narrowphase_deformable\","
                << "\"reducer_kernel\":\"nm_contact_reduce_deformable_narrowphase_failures\","
                << "\"status_reset_environment_count\":2,"
                << "\"status_reset_capacity_per_environment\":3,"
                << "\"status_reset_records_checked\":6,"
                << "\"reducer_environment_count\":4,"
                << "\"candidate_capacity_per_environment\":4,"
                << "\"active_candidate_counts\":[4,3,0,2],"
                << "\"reducer_cases\":[\"minimum_slot_wins\",\"earlier_phase_failure_preserved\",\"empty_prefix_ignores_stale_tail\",\"healthy_prefix_ignores_stale_tail\"],"
                << "\"producer_clear_verified\":true,"
                << "\"native_device_action\":true},"
                << "\"direct_results\":[";
            for (std::size_t i = 0; i < directCases.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << "{\"name\":\"" << directCases[i].name
                    << "\",\"near_band_only\":"
                    << (directCases[i].expectRoundoffBandHit ? "true" : "false")
                    << ",\"hit\":" << (direct[i][0] > 0.5f ? "true" : "false")
                    << ",\"certified\":" << (direct[i][1] > 0.5f ? "true" : "false")
                    << ",\"toi\":";
                writeJsonFiniteOrNull(std::cout, direct[i][2]);
                std::cout << ",\"separation_m\":";
                writeJsonFiniteOrNull(std::cout, direct[i][3]);
                if (directCases[i].expectRoundoffBandHit) {
                    std::cout << ",\"nominal_thickness_m\":";
                    writeJsonFiniteOrNull(std::cout, directCases[i].thickness);
                    std::cout << ",\"roundoff_expanded_thickness_m\":";
                    writeJsonFiniteOrNull(std::cout, roundoffExpandedThickness(directCases[i]));
                    std::cout << ",\"sampled_minimum_is_clearance_proof\":false";
                }
                std::cout << ",\"sampled_minimum_m\":";
                writeJsonFiniteOrNull(std::cout, sampledMinimum(directCases[i]));
                std::cout << '}';
            }
            std::cout << "],\"helper_results\":[";
            for (std::size_t i = 0; i < helperCases.size(); ++i) {
                if (i) std::cout << ',';
                const auto& first = helper[2u * i];
                const auto& second = helper[2u * i + 1u];
                std::cout << "{\"name\":\"" << helperCases[i].name
                    << "\",\"near_band_only\":"
                    << (helperCases[i].expectRoundoffBandHit ? "true" : "false")
                    << ",\"hit\":" << (first[0] > 0.5f ? "true" : "false")
                    << ",\"certified\":" << (first[1] > 0.5f ? "true" : "false")
                    << ",\"global_toi\":";
                writeJsonFiniteOrNull(std::cout, first[2]);
                std::cout << ",\"separation_m\":";
                writeJsonFiniteOrNull(std::cout, first[3]);
                if (helperCases[i].expectRoundoffBandHit) {
                    std::cout << ",\"nominal_thickness_m\":";
                    writeJsonFiniteOrNull(std::cout, helperCases[i].thickness);
                    std::cout << ",\"roundoff_expanded_thickness_m\":";
                    writeJsonFiniteOrNull(std::cout, roundoffExpandedThickness(helperCases[i]));
                    std::cout << ",\"sampled_minimum_is_clearance_proof\":false";
                }
                std::cout << ",\"tested_fraction\":";
                writeJsonFiniteOrNull(std::cout, second[0]);
                std::cout << ",\"retries\":";
                writeJsonFiniteOrNull(std::cout, second[1]);
                if (helperCases[i].verifySampledDistance ||
                    helperCases[i].expectShortenedCertificate) {
                    std::cout << ",\"sampled_minimum_tested_m\":";
                    writeJsonFiniteOrNull(std::cout,
                        sampledMinimum(helperCases[i], second[0]));
                } else {
                    std::cout << ",\"sampled_minimum_tested_m\":null";
                }
                std::cout << '}';
            }
            std::cout << "],\"checks\":" << checks
                << ",\"physical_steps\":0,\"status\":\"pass\"}\n";
            return 0;
        } catch (const std::exception& exception) {
            std::cerr << "FAIL native production deformable CCD regression: "
                      << exception.what() << '\n';
            return 1;
        }
    }
}
