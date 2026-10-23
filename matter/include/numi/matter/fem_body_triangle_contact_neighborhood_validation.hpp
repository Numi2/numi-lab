#pragma once

#include "numi/matter/numi_human_shared.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace numi::matter::fem_body_contact_neighborhood {

inline constexpr std::uint32_t kMaximumTrianglesPerSample = 128u;
inline constexpr float kMaximumDistanceMeters = 0.05f;
inline constexpr std::uint32_t kInvalidIndex = UINT32_MAX;

inline bool finite4(nm_float4 v) noexcept {
    return std::isfinite(v.x) && std::isfinite(v.y) &&
        std::isfinite(v.z) && std::isfinite(v.w);
}
inline double dot3(nm_float4 a, nm_float4 b) noexcept {
    return double(a.x)*b.x + double(a.y)*b.y + double(a.z)*b.z;
}
inline nm_float4 cross3(nm_float4 a, nm_float4 b) noexcept {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z,
            a.x*b.y-a.y*b.x, 0.0f};
}

inline bool appendUnselectedSolverRows(
    std::span<const NMNumiHumanFEMBodyContactSampleGPU> samples,
    std::span<const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU> ranges,
    const std::vector<bool>& selectedForImplicitSpring,
    std::vector<NMNumiHumanFEMBodyContactSampleGPU>& solverSamples,
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU>& solverRanges
) {
    if (selectedForImplicitSpring.size() != samples.size() ||
        (!ranges.empty() && ranges.size() != samples.size()) ||
        !solverSamples.empty() || !solverRanges.empty()) return false;
    for (std::size_t i=0;i<samples.size();++i) {
        if (selectedForImplicitSpring[i]) continue;
        solverSamples.push_back(samples[i]);
        if (!ranges.empty()) solverRanges.push_back(ranges[i]);
    }
    return true;
}

inline bool validNeighborhoodSpans(
    std::span<const NMNumiHumanFEMBodyContactSampleGPU> samples,
    std::span<const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU> ranges,
    std::span<const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> triangles
) noexcept {
    if (ranges.empty() && triangles.empty()) return true;
    if (ranges.size() != samples.size() || triangles.empty() ||
        triangles.size() > std::numeric_limits<std::uint32_t>::max())
        return false;
    std::uint64_t expectedFirst=0u;
    bool anyNeighborhood=false;
    for (std::size_t sampleIndex=0;sampleIndex<samples.size();++sampleIndex) {
        const auto& range=ranges[sampleIndex];
        if (range.firstTriangle != expectedFirst || range.reserved0 != 0u ||
            range.triangleCount > kMaximumTrianglesPerSample)
            return false;
        if (range.triangleCount==0u) {
            if (range.maximumDistanceMeters != 0.0f) return false;
            continue;
        }
        anyNeighborhood=true;
        if (samples[sampleIndex].flags !=
                (NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
                 NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT) ||
            !std::isfinite(range.maximumDistanceMeters) ||
            range.maximumDistanceMeters <= 0.0f ||
            range.maximumDistanceMeters > kMaximumDistanceMeters)
            return false;
        if (expectedFirst+range.triangleCount > triangles.size()) return false;
        for (std::uint32_t local=0;local<range.triangleCount;++local) {
            const auto& t=triangles[static_cast<std::size_t>(expectedFirst+local)];
            if (t.bodyIndex != samples[sampleIndex].bodyIndex ||
                t.boneStableId == kInvalidIndex ||
                t.triangleStableId == kInvalidIndex || t.reserved0 != 0u ||
                !finite4(t.bodyLocalVertex0) ||
                !finite4(t.bodyLocalVertex1) ||
                !finite4(t.bodyLocalVertex2) ||
                !finite4(t.authoredOutwardNormal) ||
                t.bodyLocalVertex0.w != 0.0f ||
                t.bodyLocalVertex1.w != 0.0f ||
                t.bodyLocalVertex2.w != 0.0f ||
                t.authoredOutwardNormal.w != 0.0f)
                return false;
            const double normalLength=std::sqrt(dot3(
                t.authoredOutwardNormal,t.authoredOutwardNormal));
            const auto ab=nm_float4{t.bodyLocalVertex1.x-t.bodyLocalVertex0.x,
                t.bodyLocalVertex1.y-t.bodyLocalVertex0.y,
                t.bodyLocalVertex1.z-t.bodyLocalVertex0.z,0.0f};
            const auto ac=nm_float4{t.bodyLocalVertex2.x-t.bodyLocalVertex0.x,
                t.bodyLocalVertex2.y-t.bodyLocalVertex0.y,
                t.bodyLocalVertex2.z-t.bodyLocalVertex0.z,0.0f};
            const auto face=cross3(ab,ac);
            const double faceLength=std::sqrt(dot3(face,face));
            if (!std::isfinite(normalLength) ||
                std::abs(normalLength-1.0)>1.0e-4 ||
                !std::isfinite(faceLength) || faceLength<=1.0e-10 ||
                dot3(face,t.authoredOutwardNormal)/
                    (faceLength*normalLength)<0.999)
                return false;
            for (std::uint32_t previous=0;previous<local;++previous) {
                const auto& other=triangles[
                    static_cast<std::size_t>(expectedFirst+previous)];
                if (other.boneStableId==t.boneStableId &&
                    other.triangleStableId==t.triangleStableId)
                    return false;
            }
        }
        expectedFirst += range.triangleCount;
    }
    return anyNeighborhood && expectedFirst==triangles.size();
}

} // namespace numi::matter::fem_body_contact_neighborhood
