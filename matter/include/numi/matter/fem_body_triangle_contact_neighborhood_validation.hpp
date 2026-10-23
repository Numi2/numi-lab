#pragma once

#include "numi/matter/numi_human_shared.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace numi::matter::fem_body_contact_neighborhood {

inline constexpr std::uint32_t kMaximumTrianglesPerSample = 128u;
inline constexpr float kMaximumDistanceMeters = 0.05f;
inline constexpr std::uint32_t kInvalidIndex = UINT32_MAX;
inline constexpr std::uint32_t kRangeFlagMotionEnvelope =
    NM_NUMI_HUMAN_FEM_BODY_CONTACT_NEIGHBORHOOD_MOTION_ENVELOPE;
inline constexpr float kMotionEnvelopeRoundoffUlps = 32.0f;

inline bool finite4(nm_float4 v) noexcept {
    return std::isfinite(v.x) && std::isfinite(v.y) &&
        std::isfinite(v.z) && std::isfinite(v.w);
}
inline double dot3(nm_float4 a, nm_float4 b) noexcept {
    return double(a.x)*b.x + double(a.y)*b.y + double(a.z)*b.z;
}
inline bool zero4(nm_float4 v) noexcept {
    return v.x==0.0f && v.y==0.0f && v.z==0.0f && v.w==0.0f;
}
inline double motionEnvelopeRoundoffMeters(
    nm_float4 restPointAndRadius
) noexcept {
    const double scale=std::max(1.0,
        std::max({std::abs(double(restPointAndRadius.x)),
                  std::abs(double(restPointAndRadius.y)),
                  std::abs(double(restPointAndRadius.z))})+
                  double(restPointAndRadius.w));
    return double(kMotionEnvelopeRoundoffUlps)*
        double(std::numeric_limits<float>::epsilon())*scale;
}
inline bool validMotionEnvelope(
    const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU& range
) noexcept {
    if (range.reserved0==0u) return zero4(range.restPointAndMotionRadius);
    if (range.reserved0!=kRangeFlagMotionEnvelope ||
        !finite4(range.restPointAndMotionRadius) ||
        range.restPointAndMotionRadius.w<0.0f ||
        range.restPointAndMotionRadius.w>kMaximumDistanceMeters)
        return false;
    return double(range.restPointAndMotionRadius.w)<=
        double(range.maximumDistanceMeters);
}
inline bool withinMotionEnvelope(
    const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU& range,
    nm_float4 queryBodyLocal
) noexcept {
    if (range.reserved0!=kRangeFlagMotionEnvelope ||
        !validMotionEnvelope(range) || !finite4(queryBodyLocal)) return false;
    const double dx=double(queryBodyLocal.x)-range.restPointAndMotionRadius.x;
    const double dy=double(queryBodyLocal.y)-range.restPointAndMotionRadius.y;
    const double dz=double(queryBodyLocal.z)-range.restPointAndMotionRadius.z;
    const double radius=double(range.restPointAndMotionRadius.w)+
        motionEnvelopeRoundoffMeters(range.restPointAndMotionRadius);
    return dx*dx+dy*dy+dz*dz<=radius*radius;
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
        if (range.firstTriangle != expectedFirst ||
            range.triangleCount > kMaximumTrianglesPerSample ||
            !validMotionEnvelope(range))
            return false;
        if (range.triangleCount==0u) {
            if (range.maximumDistanceMeters != 0.0f || range.reserved0!=0u ||
                !zero4(range.restPointAndMotionRadius)) return false;
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



// Preparation-time builder for a deterministic complete candidate bank.
// The caller must pass the complete registered triangle inventory for one
// articulated body and its declared count. The sample motion radius is a
// Euclidean envelope in that same body-local frame.
enum class NeighborhoodBuildFailure : std::uint8_t {
    none,
    invalidQuery,
    missingTopology,
    inventoryCountMismatch,
    invalidTriangle,
    ambiguousIdentity,
    candidateCapacityExceeded,
    flattenedIndexOverflow,
    runtimeDistanceLimitExceeded,
};

struct NeighborhoodBuildReport {
    NeighborhoodBuildFailure failure = NeighborhoodBuildFailure::none;
    double nearestRestDistanceMeters = 0.0;
    double candidateInclusionLimitMeters = 0.0;
    double certifiedQueryDistanceMeters = 0.0;
    std::uint32_t candidateCount = 0u;

    [[nodiscard]] bool complete() const noexcept {
        return failure == NeighborhoodBuildFailure::none;
    }
};

struct NeighborhoodPoint3d {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

inline NeighborhoodPoint3d neighborhoodPoint(nm_float4 p) noexcept {
    return {double(p.x),double(p.y),double(p.z)};
}
inline NeighborhoodPoint3d neighborhoodSubtract(
    NeighborhoodPoint3d a, NeighborhoodPoint3d b
) noexcept {
    return {a.x-b.x,a.y-b.y,a.z-b.z};
}
inline double neighborhoodDot(
    NeighborhoodPoint3d a, NeighborhoodPoint3d b
) noexcept {
    return a.x*b.x+a.y*b.y+a.z*b.z;
}
inline NeighborhoodPoint3d neighborhoodAddScaled(
    NeighborhoodPoint3d a, NeighborhoodPoint3d b, double scale
) noexcept {
    return {a.x+scale*b.x,a.y+scale*b.y,a.z+scale*b.z};
}
inline double neighborhoodSegmentDistanceSquared(
    NeighborhoodPoint3d p, NeighborhoodPoint3d a, NeighborhoodPoint3d b
) noexcept {
    const auto ab=neighborhoodSubtract(b,a);
    const auto ap=neighborhoodSubtract(p,a);
    const double lengthSquared=neighborhoodDot(ab,ab);
    if (!(lengthSquared>0.0)) return neighborhoodDot(ap,ap);
    const double t=std::clamp(neighborhoodDot(ap,ab)/lengthSquared,0.0,1.0);
    const auto nearest=neighborhoodAddScaled(a,ab,t);
    const auto delta=neighborhoodSubtract(p,nearest);
    return neighborhoodDot(delta,delta);
}

// Exact closest-point distance to a finite triangle, evaluated in double
// precision from the registered Float32 coordinates.
inline double pointTriangleDistanceSquared(
    nm_float4 point,
    const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU& triangle
) noexcept {
    const auto p=neighborhoodPoint(point);
    const auto a=neighborhoodPoint(triangle.bodyLocalVertex0);
    const auto b=neighborhoodPoint(triangle.bodyLocalVertex1);
    const auto c=neighborhoodPoint(triangle.bodyLocalVertex2);
    const auto ab=neighborhoodSubtract(b,a);
    const auto ac=neighborhoodSubtract(c,a);
    const auto ap=neighborhoodSubtract(p,a);
    const double d00=neighborhoodDot(ab,ab);
    const double d01=neighborhoodDot(ab,ac);
    const double d11=neighborhoodDot(ac,ac);
    const double d20=neighborhoodDot(ap,ab);
    const double d21=neighborhoodDot(ap,ac);
    const double denominator=d00*d11-d01*d01;
    if (!std::isfinite(denominator) || denominator<=0.0)
        return std::numeric_limits<double>::infinity();
    const double u=(d11*d20-d01*d21)/denominator;
    const double v=(d00*d21-d01*d20)/denominator;
    if (u>=0.0 && v>=0.0 && u+v<=1.0) {
        const auto nearest=neighborhoodAddScaled(
            neighborhoodAddScaled(a,ab,u),ac,v);
        const auto delta=neighborhoodSubtract(p,nearest);
        return neighborhoodDot(delta,delta);
    }
    return std::min({
        neighborhoodSegmentDistanceSquared(p,a,b),
        neighborhoodSegmentDistanceSquared(p,b,c),
        neighborhoodSegmentDistanceSquared(p,c,a)});
}

// Produces all triangles that can win a closest-point query for every query
// within motionRadiusMeters of restPositionBodyLocal. Let d0 be the globally
// nearest rest-point distance. A rest-nearest triangle remains within d0+R
// throughout the radius-R ball. Any omitted triangle has rest distance greater
// than d0+2R, and therefore query distance greater than d0+R throughout that
// ball; it cannot win. The Float32 guard makes both boundaries inclusive.
// Overflow is rejected; this function never truncates a complete candidate set.
inline NeighborhoodBuildReport appendCompleteNeighborhood(
    std::uint32_t bodyIndex,
    nm_float4 restPositionBodyLocal,
    float motionRadiusMeters,
    std::size_t expectedRegisteredTriangleCount,
    std::span<const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU>
        completeRegisteredBodyTriangles,
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU>&
        flattenedTriangles,
    NMNumiHumanFEMBodyContactNeighborhoodRangeGPU& outputRange
) {
    NeighborhoodBuildReport report{};
    const auto fail=[&](NeighborhoodBuildFailure failure) {
        report.failure=failure;
        return report;
    };
    if (bodyIndex==kInvalidIndex || !finite4(restPositionBodyLocal) ||
        restPositionBodyLocal.w!=0.0f ||
        !std::isfinite(motionRadiusMeters) || motionRadiusMeters<0.0f)
        return fail(NeighborhoodBuildFailure::invalidQuery);
    if (completeRegisteredBodyTriangles.empty() ||
        expectedRegisteredTriangleCount==0u)
        return fail(NeighborhoodBuildFailure::missingTopology);
    if (completeRegisteredBodyTriangles.size()!=
            expectedRegisteredTriangleCount ||
        completeRegisteredBodyTriangles.size()>
            std::numeric_limits<std::uint32_t>::max())
        return fail(NeighborhoodBuildFailure::inventoryCountMismatch);
    if (flattenedTriangles.size()>
            std::numeric_limits<std::uint32_t>::max())
        return fail(NeighborhoodBuildFailure::flattenedIndexOverflow);

    double coordinateScale=1.0;
    const auto updateScale=[&](nm_float4 p) {
        coordinateScale=std::max({coordinateScale,std::abs(double(p.x)),
            std::abs(double(p.y)),std::abs(double(p.z))});
    };
    updateScale(restPositionBodyLocal);
    std::vector<std::pair<std::uint32_t,std::uint32_t>> identities;
    identities.reserve(completeRegisteredBodyTriangles.size());
    double nearestSquared=std::numeric_limits<double>::infinity();
    std::vector<double> restDistancesSquared;
    restDistancesSquared.reserve(completeRegisteredBodyTriangles.size());
    for (const auto& triangle:completeRegisteredBodyTriangles) {
        if (triangle.bodyIndex!=bodyIndex ||
            triangle.boneStableId==kInvalidIndex ||
            triangle.triangleStableId==kInvalidIndex ||
            triangle.reserved0!=0u || !finite4(triangle.bodyLocalVertex0) ||
            !finite4(triangle.bodyLocalVertex1) ||
            !finite4(triangle.bodyLocalVertex2) ||
            !finite4(triangle.authoredOutwardNormal) ||
            triangle.bodyLocalVertex0.w!=0.0f ||
            triangle.bodyLocalVertex1.w!=0.0f ||
            triangle.bodyLocalVertex2.w!=0.0f ||
            triangle.authoredOutwardNormal.w!=0.0f)
            return fail(NeighborhoodBuildFailure::invalidTriangle);
        const double normalLength=std::sqrt(dot3(
            triangle.authoredOutwardNormal,triangle.authoredOutwardNormal));
        const auto ab=nm_float4{
            triangle.bodyLocalVertex1.x-triangle.bodyLocalVertex0.x,
            triangle.bodyLocalVertex1.y-triangle.bodyLocalVertex0.y,
            triangle.bodyLocalVertex1.z-triangle.bodyLocalVertex0.z,0.0f};
        const auto ac=nm_float4{
            triangle.bodyLocalVertex2.x-triangle.bodyLocalVertex0.x,
            triangle.bodyLocalVertex2.y-triangle.bodyLocalVertex0.y,
            triangle.bodyLocalVertex2.z-triangle.bodyLocalVertex0.z,0.0f};
        const auto face=cross3(ab,ac);
        const double faceLength=std::sqrt(dot3(face,face));
        if (!std::isfinite(normalLength) ||
            std::abs(normalLength-1.0)>1.0e-4 ||
            !std::isfinite(faceLength) || faceLength<=1.0e-10 ||
            dot3(face,triangle.authoredOutwardNormal)/
                (faceLength*normalLength)<0.999)
            return fail(NeighborhoodBuildFailure::invalidTriangle);
        updateScale(triangle.bodyLocalVertex0);
        updateScale(triangle.bodyLocalVertex1);
        updateScale(triangle.bodyLocalVertex2);
        const double distanceSquared=pointTriangleDistanceSquared(
            restPositionBodyLocal,triangle);
        if (!std::isfinite(distanceSquared) || distanceSquared<0.0)
            return fail(NeighborhoodBuildFailure::invalidTriangle);
        nearestSquared=std::min(nearestSquared,distanceSquared);
        restDistancesSquared.push_back(distanceSquared);
        identities.emplace_back(triangle.boneStableId,
                                triangle.triangleStableId);
    }
    std::sort(identities.begin(),identities.end());
    if (std::adjacent_find(identities.begin(),identities.end())!=
            identities.end())
        return fail(NeighborhoodBuildFailure::ambiguousIdentity);

    const double nearest=std::sqrt(nearestSquared);
    const double radius=double(motionRadiusMeters);
    const double guard=32.0*double(std::numeric_limits<float>::epsilon())*
        coordinateScale;
    const double envelopeScale=std::max(1.0,
        std::max({std::abs(double(restPositionBodyLocal.x)),
                  std::abs(double(restPositionBodyLocal.y)),
                  std::abs(double(restPositionBodyLocal.z))})+radius);
    const double envelopeGuard=32.0*double(
        std::numeric_limits<float>::epsilon())*envelopeScale;
    // The query guard is part of the supported motion ball. Keep enough faces
    // for its full diameter, plus a separate bound for source-triangle rounding.
    const double inclusionLimit=nearest+2.0*(radius+envelopeGuard)+guard;
    const double certifiedQueryDistance=nearest+radius+envelopeGuard+guard;
    report.nearestRestDistanceMeters=nearest;
    report.candidateInclusionLimitMeters=inclusionLimit;
    report.certifiedQueryDistanceMeters=certifiedQueryDistance;
    const float roundedQueryDistance=std::nextafter(
        static_cast<float>(certifiedQueryDistance),
        std::numeric_limits<float>::infinity());
    if (!std::isfinite(roundedQueryDistance) ||
        roundedQueryDistance<=0.0f ||
        roundedQueryDistance>kMaximumDistanceMeters)
        return fail(NeighborhoodBuildFailure::runtimeDistanceLimitExceeded);

    std::vector<std::size_t> candidateIndices;
    candidateIndices.reserve(kMaximumTrianglesPerSample+1u);
    std::uint32_t candidateCount=0u;
    const double inclusionLimitSquared=inclusionLimit*inclusionLimit;
    for (std::size_t i=0;i<restDistancesSquared.size();++i) {
        if (restDistancesSquared[i]>inclusionLimitSquared) continue;
        if (candidateCount<kMaximumTrianglesPerSample)
            candidateIndices.push_back(i);
        ++candidateCount;
    }
    report.candidateCount=candidateCount;
    if (candidateCount==0u)
        return fail(NeighborhoodBuildFailure::missingTopology);
    if (candidateCount>kMaximumTrianglesPerSample)
        return fail(NeighborhoodBuildFailure::candidateCapacityExceeded);
    const std::uint64_t total=std::uint64_t(flattenedTriangles.size())+
        candidateCount;
    if (total>std::numeric_limits<std::uint32_t>::max())
        return fail(NeighborhoodBuildFailure::flattenedIndexOverflow);

    std::sort(candidateIndices.begin(),candidateIndices.end(),
        [&](std::size_t lhs,std::size_t rhs) {
            const auto& a=completeRegisteredBodyTriangles[lhs];
            const auto& b=completeRegisteredBodyTriangles[rhs];
            if (a.boneStableId!=b.boneStableId)
                return a.boneStableId<b.boneStableId;
            return a.triangleStableId<b.triangleStableId;
        });
    const auto first=static_cast<std::uint32_t>(flattenedTriangles.size());
    flattenedTriangles.reserve(static_cast<std::size_t>(total));
    for (const auto index:candidateIndices)
        flattenedTriangles.push_back(
            completeRegisteredBodyTriangles[index]);
    outputRange={first,candidateCount,roundedQueryDistance,
        kRangeFlagMotionEnvelope,
        {restPositionBodyLocal.x,restPositionBodyLocal.y,
         restPositionBodyLocal.z,motionRadiusMeters}};
    return report;
}

} // namespace numi::matter::fem_body_contact_neighborhood
