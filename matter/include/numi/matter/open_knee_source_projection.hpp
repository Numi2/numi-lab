#pragma once

// Initial contact binding for the complete pinned Open Knee case. The source
// deck specifies "sliding-elastic": FEBio 2.9 registers that name to
// FESlidingElasticInterface, which projects three triangle integration points
// along the SLAVE face normal. It does not use the nearest-master-vertex
// algorithm of "sliding-node-on-facet". These rows identify active initial
// quadrature points only; the live Newton contact law and segment updates
// must still be assembled before an equilibrium can be accepted.
#include "numi/matter/open_knee_source_contact.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace numi_matter_open_knee {

using SourcePoint = std::array<double, 3>;

inline SourcePoint subtract(SourcePoint a, SourcePoint b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
inline SourcePoint add(SourcePoint a, SourcePoint b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
inline SourcePoint scale(SourcePoint a, double s) {
    return {a[0] * s, a[1] * s, a[2] * s};
}
inline double dot(SourcePoint a, SourcePoint b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline SourcePoint cross(SourcePoint a, SourcePoint b) {
    return {a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}
inline double length(SourcePoint a) {
    return std::sqrt(dot(a, a));
}

struct SourceContactGaussProjection {
    std::uint32_t pair = 0u, pass = 0u;
    std::uint32_t slaveFace = NM_INVALID_INDEX, gaussPoint = 0u;
    std::uint32_t masterFace = NM_INVALID_INDEX;
    SourcePoint masterBarycentric{};
    double gap = 0.0;               // positive means source penetration, mm
    double integrationArea = 0.0;   // current initial face area / 3, mm^2
};

struct SourceContactProjectionSummary {
    std::vector<SourceContactGaussProjection> activeRows;
    std::array<std::uint32_t, 36> activeFacesBySurface{};
    std::uint64_t quadraturePoints = 0u;
    std::uint64_t activePoints = 0u;
    double maximumGap = 0.0;
    double searchRadius = 0.0;
};

// The frozen Geometry_custom.feb contains 248,236 node coordinates. Its
// FEBoundingBox::radius() is the largest box dimension, 180.03255462646484
// mm, not half of its diagonal. FEBio BW scales each pair's search_radius by
// this value. The source contact program is hash-bound to that geometry.
inline constexpr double sourceKneeBoundingBoxRadius = 180.03255462646484;

class SourceContactTriangleTree {
public:
    struct Triangle {
        std::uint32_t face;
        std::array<SourcePoint, 3> point;
        SourcePoint minimum{}, maximum{}, center{};
    };

    SourceContactTriangleTree(
        const SourceSlidingContactProgram& program,
        const SourceContactSurface& surface,
        const std::unordered_map<std::uint32_t, SourcePoint>& positions,
        const double barycentricTolerance
    ) {
        triangles_.reserve(surface.faceCount);
        for (std::uint32_t index = surface.firstFace;
             index < surface.firstFace + surface.faceCount; ++index) {
            const auto& face = program.faces[index];
            Triangle triangle{.face = index};
            for (std::size_t vertex = 0u; vertex < 3u; ++vertex)
                triangle.point[vertex] = positions.at(face.sourceNodes[vertex]);
            triangle.minimum = triangle.maximum = triangle.point[0];
            const double scaleLength = std::max({
                length(subtract(triangle.point[1], triangle.point[0])),
                length(subtract(triangle.point[2], triangle.point[1])),
                length(subtract(triangle.point[0], triangle.point[2]))});
            for (const auto& point : triangle.point)
                for (std::size_t axis = 0u; axis < 3u; ++axis) {
                    triangle.minimum[axis] = std::min(triangle.minimum[axis], point[axis]);
                    triangle.maximum[axis] = std::max(triangle.maximum[axis], point[axis]);
                }
            // FEBio allows barycentric coordinates up to search_tol beyond
            // the geometric triangle. The inflated BVH remains a superset of
            // those ray intersections; the exact triangle test is below.
            const double expansion = barycentricTolerance * 2.0 * scaleLength +
                                     1.0e-9;
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                triangle.minimum[axis] -= expansion;
                triangle.maximum[axis] += expansion;
                triangle.center[axis] = (triangle.minimum[axis] +
                                         triangle.maximum[axis]) * 0.5;
            }
            triangles_.push_back(triangle);
        }
        order_.resize(triangles_.size());
        std::iota(order_.begin(), order_.end(), 0u);
        if (!order_.empty()) build(0u, order_.size());
    }

    [[nodiscard]] bool activeProjection(
        const SourcePoint origin, const SourcePoint slaveNormal,
        const double radius, const double tolerance,
        std::uint32_t& masterFace, SourcePoint& masterBarycentric,
        double& positiveGap
    ) const {
        if (nodes_.empty()) return false;
        std::vector<std::uint32_t> stack{0u};
        double bestRayParameter = 0.0;
        while (!stack.empty()) {
            const std::uint32_t index = stack.back();
            stack.pop_back();
            const Node& node = nodes_[index];
            if (!segmentIntersectsBox(origin, scale(slaveNormal, -radius),
                    node.minimum, node.maximum)) continue;
            if (node.left != NM_INVALID_INDEX) {
                stack.push_back(node.left);
                stack.push_back(node.right);
                continue;
            }
            for (std::size_t item = node.first; item < node.last; ++item) {
                const auto& triangle = triangles_[order_[item]];
                double rayParameter = 0.0;
                SourcePoint barycentric{};
                if (!intersectSourceTriangle(triangle, origin, slaveNormal,
                        tolerance, rayParameter, barycentric) ||
                    !(rayParameter > -radius && rayParameter < 0.0))
                    continue;
                if (masterFace == NM_INVALID_INDEX ||
                    rayParameter < bestRayParameter ||
                    (rayParameter == bestRayParameter &&
                     triangle.face < masterFace)) {
                    bestRayParameter = rayParameter;
                    masterFace = triangle.face;
                    masterBarycentric = barycentric;
                }
            }
        }
        if (masterFace == NM_INVALID_INDEX) return false;
        positiveGap = -bestRayParameter;
        return true;
    }

private:
    struct Node {
        SourcePoint minimum{}, maximum{};
        std::uint32_t left = NM_INVALID_INDEX, right = NM_INVALID_INDEX;
        std::size_t first = 0u, last = 0u;
    };
    std::vector<Triangle> triangles_;
    std::vector<std::uint32_t> order_;
    std::vector<Node> nodes_;

    std::uint32_t build(const std::size_t first, const std::size_t last) {
        const auto index = static_cast<std::uint32_t>(nodes_.size());
        nodes_.push_back({});
        Node& row = nodes_[index];
        row.first = first;
        row.last = last;
        row.minimum = triangles_[order_[first]].minimum;
        row.maximum = triangles_[order_[first]].maximum;
        SourcePoint centerMinimum = triangles_[order_[first]].center;
        SourcePoint centerMaximum = centerMinimum;
        for (std::size_t item = first + 1u; item < last; ++item)
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                const auto& triangle = triangles_[order_[item]];
                row.minimum[axis] = std::min(row.minimum[axis], triangle.minimum[axis]);
                row.maximum[axis] = std::max(row.maximum[axis], triangle.maximum[axis]);
                centerMinimum[axis] = std::min(centerMinimum[axis], triangle.center[axis]);
                centerMaximum[axis] = std::max(centerMaximum[axis], triangle.center[axis]);
            }
        if (last - first <= 8u) return index;
        std::size_t axis = 0u;
        for (std::size_t next = 1u; next < 3u; ++next)
            if (centerMaximum[next] - centerMinimum[next] >
                centerMaximum[axis] - centerMinimum[axis]) axis = next;
        const std::size_t middle = first + (last - first) / 2u;
        std::nth_element(order_.begin() + first, order_.begin() + middle,
            order_.begin() + last, [&](const std::uint32_t a,
                                        const std::uint32_t b) {
                const double ac = triangles_[a].center[axis];
                const double bc = triangles_[b].center[axis];
                return ac < bc || (ac == bc &&
                                    triangles_[a].face < triangles_[b].face);
            });
        const auto left = build(first, middle);
        const auto right = build(middle, last);
        nodes_[index].left = left;
        nodes_[index].right = right;
        return index;
    }

    static bool segmentIntersectsBox(
        const SourcePoint origin, const SourcePoint displacement,
        const SourcePoint minimum, const SourcePoint maximum
    ) {
        double first = 0.0, last = 1.0;
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            if (std::abs(displacement[axis]) < 1.0e-15) {
                if (origin[axis] < minimum[axis] ||
                    origin[axis] > maximum[axis]) return false;
                continue;
            }
            const double inverse = 1.0 / displacement[axis];
            const double a = (minimum[axis] - origin[axis]) * inverse;
            const double b = (maximum[axis] - origin[axis]) * inverse;
            first = std::max(first, std::min(a, b));
            last = std::min(last, std::max(a, b));
            if (last < first) return false;
        }
        return true;
    }

    static bool intersectSourceTriangle(
        const Triangle& triangle, const SourcePoint origin,
        const SourcePoint ray, const double tolerance,
        double& rayParameter, SourcePoint& barycentric
    ) {
        const auto e0 = subtract(triangle.point[1], triangle.point[0]);
        const auto e1 = subtract(triangle.point[2], triangle.point[0]);
        const auto faceNormal = cross(e0, e1);
        const double faceLength = length(faceNormal);
        if (!(faceLength > 0.0)) return false;
        const auto normal = scale(faceNormal, 1.0 / faceLength);
        const double opposing = dot(ray, normal);
        if (!(opposing < 0.0)) return false;
        rayParameter = dot(normal, subtract(triangle.point[0], origin)) /
                       opposing;
        const auto point = add(origin, scale(ray, rayParameter));
        const auto relative = subtract(point, triangle.point[0]);
        const double aa = dot(e0, e0), ab = dot(e0, e1), bb = dot(e1, e1);
        const double determinant = aa * bb - ab * ab;
        if (!(determinant > 0.0)) return false;
        const double r = (bb * dot(relative, e0) -
                          ab * dot(relative, e1)) / determinant;
        const double s = (aa * dot(relative, e1) -
                          ab * dot(relative, e0)) / determinant;
        barycentric = {1.0 - r - s, r, s};
        return r >= -tolerance && s >= -tolerance &&
               r + s <= 1.0 + tolerance;
    }
};

inline bool bindSourceInitialContactProjections(
    const SourceSlidingContactProgram& program,
    const std::unordered_map<std::uint32_t, SourcePoint>& positions,
    SourceContactProjectionSummary& result, std::string& error
) {
    if (program.pairs.size() != 18u || program.surfaces.size() != 36u) {
        error = "source sliding-elastic topology is incomplete";
        return false;
    }
    SourceContactProjectionSummary candidate;
    std::vector<bool> activeFace(program.faces.size(), false);
    for (std::size_t pairIndex = 0u; pairIndex < program.pairs.size(); ++pairIndex) {
        const auto& pair = program.pairs[pairIndex];
        if (pair.parameters[0] != 0.0 || pair.parameters[4] != 1.0 ||
            pair.parameters[6] != 0.0 || pair.parameters[7] != 0.01 ||
            !(pair.parameters[8] > 0.0)) {
            error = "source sliding-elastic pair has unsupported enforcement";
            return false;
        }
        const double radius = pair.parameters[8] *
                              sourceKneeBoundingBoxRadius;
        candidate.searchRadius = std::max(candidate.searchRadius, radius);
        for (std::uint32_t pass = 0u; pass < 2u; ++pass) {
            const auto surfaceIndex = pass == 0u ? pair.slave : pair.master;
            const auto& slave = program.surfaces[surfaceIndex];
            const auto& master = program.surfaces[
                pass == 0u ? pair.master : pair.slave];
            for (std::uint32_t index = master.firstFace;
                 index < master.firstFace + master.faceCount; ++index)
                for (const auto node : program.faces[index].sourceNodes)
                    if (!positions.contains(node)) {
                        error = "source sliding-elastic master node has no position";
                        return false;
                    }
            SourceContactTriangleTree tree(program, master, positions,
                                            pair.parameters[7]);
            for (std::uint32_t faceIndex = slave.firstFace;
                 faceIndex < slave.firstFace + slave.faceCount; ++faceIndex) {
                const auto& face = program.faces[faceIndex];
                std::array<SourcePoint, 3> point{};
                for (std::size_t vertex = 0u; vertex < 3u; ++vertex) {
                    const auto found = positions.find(face.sourceNodes[vertex]);
                    if (found == positions.end()) {
                        error = "source sliding-elastic slave node has no position";
                        return false;
                    }
                    point[vertex] = found->second;
                }
                const auto crossProduct = cross(
                    subtract(point[1], point[0]),
                    subtract(point[2], point[0]));
                const double jacobian = length(crossProduct);
                if (!(jacobian > 0.0) || !std::isfinite(jacobian)) {
                    error = "source sliding-elastic face is degenerate";
                    return false;
                }
                const auto normal = scale(crossProduct, 1.0 / jacobian);
                // FE_TRI3G3: three integration points at barycentric
                // permutations of (2/3, 1/6, 1/6), each weight 1/6.
                for (std::uint32_t integration = 0u; integration < 3u;
                     ++integration) {
                    ++candidate.quadraturePoints;
                    SourcePoint origin{};
                    for (std::size_t vertex = 0u; vertex < 3u; ++vertex) {
                        const double shape = vertex == integration
                            ? 2.0 / 3.0 : 1.0 / 6.0;
                        origin = add(origin, scale(point[vertex], shape));
                    }
                    SourceContactGaussProjection row;
                    row.pair = static_cast<std::uint32_t>(pairIndex);
                    row.pass = pass;
                    row.slaveFace = faceIndex;
                    row.gaussPoint = integration;
                    row.integrationArea = jacobian / 6.0;
                    if (tree.activeProjection(origin, normal, radius,
                            pair.parameters[7], row.masterFace,
                            row.masterBarycentric, row.gap)) {
                        candidate.activeRows.push_back(row);
                        ++candidate.activePoints;
                        candidate.maximumGap = std::max(candidate.maximumGap,
                                                        row.gap);
                        if (!activeFace[faceIndex]) {
                            activeFace[faceIndex] = true;
                            ++candidate.activeFacesBySurface[surfaceIndex];
                        }
                    }
                }
            }
        }
    }
    result = std::move(candidate);
    error.clear();
    return true;
}

} // namespace numi_matter_open_knee
