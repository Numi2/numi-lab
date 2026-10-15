#include "numi/matter/mortar_shared_edge_guard.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

using Point = NMMortarGuardPoint3;
using Triangle = std::array<Point, 3>;
constexpr float kOverlapTolerance = 1.0e-10f;
constexpr float kClipTolerance = 1.0e-7f;

float cross2(float ax, float ay, float bx, float by) {
    return ax * by - ay * bx;
}

float signedArea(const Triangle& t) {
    return cross2(t[1].x - t[0].x, t[1].y - t[0].y,
                  t[2].x - t[0].x, t[2].y - t[0].y);
}

Point mix(Point a, Point b, float t) {
    return {a.x + t * (b.x - a.x), a.y + t * (b.y - a.y),
            a.z + t * (b.z - a.z)};
}

std::vector<Point> clip(const std::vector<Point>& input, Point start,
                        Point end, float orientation) {
    std::vector<Point> output;
    const float ex = end.x - start.x;
    const float ey = end.y - start.y;
    for (std::size_t i = 0; i < input.size(); ++i) {
        const Point a = input[i];
        const Point b = input[(i + 1u) % input.size()];
        const float sideA = orientation * cross2(ex, ey, a.x - start.x,
                                                 a.y - start.y);
        const float sideB = orientation * cross2(ex, ey, b.x - start.x,
                                                 b.y - start.y);
        const bool aInside = sideA >= -kClipTolerance;
        const bool bInside = sideB >= -kClipTolerance;
        if (aInside) output.push_back(a);
        if (aInside != bInside) {
            const float denominator = sideA - sideB;
            if (std::abs(denominator) <= 1.0e-20f) return {};
            const float fraction = std::clamp(sideA / denominator, 0.0f, 1.0f);
            const float x = a.x + fraction * (b.x - a.x);
            const float y = a.y + fraction * (b.y - a.y);
            output.push_back({x, y, 1.0f - x - y});
        }
    }
    return output;
}

float sourceIntersectionArea(const Triangle& a, const Triangle& b) {
    const float orientationArea = signedArea(b);
    if (!std::isfinite(orientationArea) || std::abs(orientationArea) <= 1.0e-10f)
        return std::numeric_limits<float>::infinity();
    const float orientation = orientationArea > 0.0f ? 1.0f : -1.0f;
    std::vector<Point> polygon(a.begin(), a.end());
    for (unsigned int edge = 0; edge < 3u; ++edge) {
        polygon = clip(polygon, b[edge], b[(edge + 1u) % 3u], orientation);
        if (polygon.size() < 3u) return 0.0f;
    }
    float twiceArea = 0.0f;
    for (std::size_t i = 0; i < polygon.size(); ++i) {
        const Point p = polygon[i];
        const Point q = polygon[(i + 1u) % polygon.size()];
        twiceArea += cross2(p.x, p.y, q.x, q.y);
    }
    return std::isfinite(twiceArea) ? 0.5f * std::abs(twiceArea)
                                    : std::numeric_limits<float>::infinity();
}

bool productionDecisionRejects(float overlap, const Triangle& a,
                               const Triangle& b) {
    if (!std::isfinite(overlap)) return true;
    return overlap > kOverlapTolerance &&
        !nmMortarExactOppositeSharedEdge(a[0], a[1], a[2], b[0], b[1], b[2]);
}

bool expect(const char* label, bool condition) {
    std::printf("%s=%s\n", label, condition ? "PASS" : "FAIL");
    return condition;
}

}  // namespace

int main() {
    // Exact source FP32 maps from shipping audit 091, records 100158/100159.
    // The ordinary clipper reports 1.460882082e-9 area on this shared edge.
    const Point a0{0x1.440cf6p-2f, 0x1.51afcap-1f, 0x1.89374cp-6f};
    const Point a1{0x1.8p-57f, 0x1.f9db22p-1f, 0x1.89374cp-7f};
    const Point a2{0x1.440cf6p-2f, 0x1.569a16p-1f, 0x1.d7dbf4p-7f};
    const Point b0{0x1.440cf6p-2f, 0x1.51afcap-1f, 0x1.89374cp-6f};
    const Point b1{0x0p+0f, 0x1.f3b646p-1f, 0x1.89374cp-6f};
    const Point b2{0x1.8p-57f, 0x1.f9db22p-1f, 0x1.89374cp-7f};
    const Triangle sourceA{a0, a1, a2};
    const Triangle sourceB{b0, b1, b2};
    const float sourceArea = sourceIntersectionArea(sourceA, sourceB);
    bool ok = true;
    ok &= expect("source_shared_edge_proof",
        nmMortarExactOppositeSharedEdge(a0, a1, a2, b0, b1, b2));
    ok &= expect("source_clip_is_tolerance_sliver",
        std::isfinite(sourceArea) && sourceArea > kOverlapTolerance);
    ok &= expect("source_false_positive_only_is_exempted",
        !productionDecisionRejects(sourceArea, sourceA, sourceB));

    // Same exact edge and same-side third vertices is a real overlapping pair.
    const Triangle sameSide{b0, b2, a2};
    const float sameSideArea = sourceIntersectionArea(sourceA, sameSide);
    ok &= expect("same_side_not_exempted",
        !nmMortarExactOppositeSharedEdge(a0, a1, a2,
            sameSide[0], sameSide[1], sameSide[2]));
    ok &= expect("same_side_overlap_rejected",
        std::isfinite(sameSideArea) && sameSideArea > kOverlapTolerance &&
        productionDecisionRejects(sameSideArea, sourceA, sameSide));

    // A positive-area overlap with no exact paired endpoint is never exempted.
    const Point centroid{
        (a0.x + a1.x + a2.x) / 3.0f,
        (a0.y + a1.y + a2.y) / 3.0f,
        (a0.z + a1.z + a2.z) / 3.0f};
    const Triangle interiorOverlap{
        mix(a0, centroid, 0.1f), mix(a1, centroid, 0.1f),
        mix(a2, centroid, 0.1f)};
    const float interiorArea = sourceIntersectionArea(sourceA, interiorOverlap);
    ok &= expect("interior_overlap_not_exempted",
        !nmMortarExactOppositeSharedEdge(a0, a1, a2,
            interiorOverlap[0], interiorOverlap[1], interiorOverlap[2]));
    ok &= expect("interior_overlap_rejected",
        std::isfinite(interiorArea) && interiorArea > kOverlapTolerance &&
        productionDecisionRejects(interiorArea, sourceA, interiorOverlap));

    // Moving one nominally shared endpoint by one ULP disables the exemption.
    Triangle nearEdge = sourceB;
    nearEdge[2].x = std::nextafter(nearEdge[2].x,
                                   std::numeric_limits<float>::infinity());
    ok &= expect("one_ulp_endpoint_mismatch_not_exempted",
        !nmMortarExactOppositeSharedEdge(a0, a1, a2,
            nearEdge[0], nearEdge[1], nearEdge[2]));

    // A collinear third vertex has no robust strict side sign and remains
    // unclassified; no tolerance-based or degenerate exemption is allowed.
    const Point collinear{0.158227846f, 0.8237721324f, 0.018f};
    ok &= expect("collinear_third_vertex_not_exempted",
        !nmMortarExactOppositeSharedEdge(a0, a1, collinear, b0, b1, b2));

    // Three shared vertices represent duplicate triangles, not an edge-only
    // contact, and therefore cannot bypass the ordinary overlap test.
    ok &= expect("identical_triangle_not_exempted",
        !nmMortarExactOppositeSharedEdge(a0, a1, a2, a0, a1, a2));

    std::printf("source_clip_area=%.10g overlap_floor=%.10g\n",
                sourceArea, kOverlapTolerance);
    return ok ? 0 : 1;
}
