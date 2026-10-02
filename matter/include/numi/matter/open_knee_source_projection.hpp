#pragma once

// Source-frame initial projection for the pinned FEBio 2.9 sliding-elastic
// contact graph. FEBio 2.9 searches from the closest master vertex through
// its incident faces; this stage preserves that ordering. It is a binding
// operation, not an accepted contact equilibrium or a substitute for the
// Newton-iteration projection update.
#include "numi/matter/open_knee_source_contact.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <unordered_map>
#include <unordered_set>
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

struct SourceContactNodeProjection {
    std::uint32_t pair = 0u, pass = 0u, slaveNode = 0u;
    std::uint32_t masterFace = NM_INVALID_INDEX;
    SourcePoint barycentric{};
    double referenceArea = 0.0;
    double gap = 0.0;
};

struct SourceContactProjectionSummary {
    std::vector<SourceContactNodeProjection> rows;
    std::uint32_t projected = 0u, penetrating = 0u;
    std::uint32_t projectedSecondRing = 0u;
    std::uint32_t unresolved = 0u, degenerate = 0u;
};

class SourceContactVertexTree {
public:
    struct Vertex { std::uint32_t id; SourcePoint position; };

    explicit SourceContactVertexTree(std::vector<Vertex> vertices)
        : vertices_(std::move(vertices)) {
        order_.resize(vertices_.size());
        std::iota(order_.begin(), order_.end(), 0u);
        root_ = build(0u, order_.size(), 0u);
    }

    std::uint32_t nearest(SourcePoint point) const {
        double bestDistance = std::numeric_limits<double>::infinity();
        std::uint32_t best = NM_INVALID_INDEX;
        nearest(root_, point, bestDistance, best);
        return best;
    }

private:
    struct Node {
        std::uint32_t vertex, axis;
        std::int32_t left = -1, right = -1;
    };
    std::vector<Vertex> vertices_;
    std::vector<std::uint32_t> order_;
    std::vector<Node> nodes_;
    std::int32_t root_ = -1;

    std::int32_t build(std::size_t first, std::size_t last,
                       std::uint32_t depth) {
        if (first == last) return -1;
        const std::uint32_t axis = depth % 3u;
        const std::size_t middle = first + (last - first) / 2u;
        std::nth_element(order_.begin() + first, order_.begin() + middle,
            order_.begin() + last, [&](std::uint32_t a, std::uint32_t b) {
                const double av = vertices_[a].position[axis];
                const double bv = vertices_[b].position[axis];
                return av < bv || (av == bv && vertices_[a].id < vertices_[b].id);
            });
        const std::int32_t index = static_cast<std::int32_t>(nodes_.size());
        nodes_.push_back({order_[middle], axis});
        const auto left = build(first, middle, depth + 1u);
        const auto right = build(middle + 1u, last, depth + 1u);
        nodes_[index].left = left;
        nodes_[index].right = right;
        return index;
    }

    void nearest(std::int32_t index, SourcePoint point,
                 double& bestDistance, std::uint32_t& best) const {
        if (index < 0) return;
        const Node& node = nodes_[index];
        const Vertex& vertex = vertices_[node.vertex];
        const SourcePoint delta = subtract(point, vertex.position);
        const double distance = dot(delta, delta);
        if (distance < bestDistance ||
            (distance == bestDistance && vertex.id < best)) {
            bestDistance = distance;
            best = vertex.id;
        }
        const double plane = delta[node.axis];
        const auto near = plane < 0.0 ? node.left : node.right;
        const auto far = plane < 0.0 ? node.right : node.left;
        nearest(near, point, bestDistance, best);
        if (plane * plane <= bestDistance)
            nearest(far, point, bestDistance, best);
    }
};

inline bool projectSourceTriangle(
    SourcePoint point, const SourceContactFace& face,
    const std::unordered_map<std::uint32_t, SourcePoint>& positions,
    double tolerance, SourcePoint& barycentric, double& gap
) {
    const auto a = positions.at(face.sourceNodes[0]);
    const auto b = positions.at(face.sourceNodes[1]);
    const auto c = positions.at(face.sourceNodes[2]);
    const auto u = subtract(b, a), v = subtract(c, a);
    const auto n = cross(u, v);
    const double n2 = dot(n, n);
    if (!(n2 > 1.0e-24) || !std::isfinite(n2)) return false;
    const auto normal = scale(n, 1.0 / std::sqrt(n2));
    const double height = dot(normal, subtract(point, a));
    const auto inPlane = subtract(subtract(point, a), scale(normal, height));
    const double uu = dot(u, u), uv = dot(u, v), vv = dot(v, v);
    const double determinant = uu * vv - uv * uv;
    if (!(determinant > 1.0e-24)) return false;
    const double r = (vv * dot(inPlane, u) - uv * dot(inPlane, v)) /
        determinant;
    const double s = (uu * dot(inPlane, v) - uv * dot(inPlane, u)) /
        determinant;
    barycentric = {1.0 - r - s, r, s};
    gap = -height;
    return r >= -tolerance && s >= -tolerance &&
        r + s <= 1.0 + tolerance;
}

inline bool bindSourceInitialContactProjections(
    const SourceSlidingContactProgram& program,
    const std::unordered_map<std::uint32_t, SourcePoint>& positions,
    SourceContactProjectionSummary& result, std::string& error
) {
    if (program.pairs.size() != 18u || program.surfaces.size() != 36u) {
        error = "source initial contact topology is incomplete";
        return false;
    }
    SourceContactProjectionSummary candidate;
    for (std::size_t pairIndex = 0u; pairIndex < program.pairs.size(); ++pairIndex) {
        const auto& pair = program.pairs[pairIndex];
        if (pair.parameters[4] != 1.0 || pair.parameters[7] != 0.01 ||
            pair.parameters[0] != 0.0) {
            error = "source initial contact has unsupported search or enforcement";
            return false;
        }
        for (std::uint32_t pass = 0u; pass < 2u; ++pass) {
            const auto& slave = program.surfaces[pass == 0u ? pair.slave : pair.master];
            const auto& master = program.surfaces[pass == 0u ? pair.master : pair.slave];
            std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> incident;
            std::vector<SourceContactVertexTree::Vertex> masterVertices;
            for (std::uint32_t index = master.firstFace;
                 index < master.firstFace + master.faceCount; ++index) {
                const auto& face = program.faces[index];
                for (const auto id : face.sourceNodes) {
                    if (!positions.contains(id)) {
                        error = "source contact master vertex lacks its reference position";
                        return false;
                    }
                    auto [found, inserted] = incident.try_emplace(id);
                    if (inserted)
                        masterVertices.push_back({id, positions.at(id)});
                    found->second.push_back(index);
                }
            }
            if (masterVertices.empty()) {
                error = "source contact master surface has no vertices";
                return false;
            }
            SourceContactVertexTree tree(std::move(masterVertices));
            std::unordered_map<std::uint32_t, double> slaveArea;
            std::vector<std::uint32_t> slaveOrder;
            for (std::uint32_t index = slave.firstFace;
                 index < slave.firstFace + slave.faceCount; ++index) {
                const auto& face = program.faces[index];
                for (const auto id : face.sourceNodes)
                    if (!positions.contains(id)) {
                        error = "source contact slave vertex lacks its reference position";
                        return false;
                    }
                const auto a = positions.at(face.sourceNodes[0]);
                const auto b = positions.at(face.sourceNodes[1]);
                const auto c = positions.at(face.sourceNodes[2]);
                const double area = 0.5 * std::sqrt(dot(
                    cross(subtract(b, a), subtract(c, a)),
                    cross(subtract(b, a), subtract(c, a))));
                if (!(area > 0.0) || !std::isfinite(area)) {
                    error = "source contact slave face has invalid area";
                    return false;
                }
                for (const auto id : face.sourceNodes) {
                    auto [entry, inserted] = slaveArea.try_emplace(id, 0.0);
                    if (inserted) slaveOrder.push_back(id);
                    entry->second += area / 3.0;
                }
            }
            for (const auto id : slaveOrder) {
                SourceContactNodeProjection row;
                row.pair = static_cast<std::uint32_t>(pairIndex);
                row.pass = pass;
                row.slaveNode = id;
                row.referenceArea = slaveArea.at(id);
                const auto nearest = tree.nearest(positions.at(id));
                const auto found = incident.find(nearest);
                if (found == incident.end()) {
                    error = "source contact closest master vertex has no face";
                    return false;
                }
                for (const auto faceIndex : found->second) {
                    if (projectSourceTriangle(positions.at(id),
                            program.faces[faceIndex], positions,
                            pair.parameters[7], row.barycentric, row.gap)) {
                        row.masterFace = faceIndex;
                        break;
                    }
                }
                // FEClosestPointProjection::Project in the FEBio 2.9 source
                // searches the faces incident to every vertex of the nearest
                // vertex's incident faces before trying edge/node cases.
                // Preserve its nested traversal order, including duplicates:
                // the first admissible face owns the projection.
                if (row.masterFace == NM_INVALID_INDEX) {
                    for (const auto firstFace : found->second) {
                        for (const auto neighbor :
                             program.faces[firstFace].sourceNodes) {
                            const auto next = incident.find(neighbor);
                            if (next == incident.end()) {
                                error = "source contact adjacent master vertex has no face";
                                return false;
                            }
                            for (const auto faceIndex : next->second) {
                                if (projectSourceTriangle(positions.at(id),
                                        program.faces[faceIndex], positions,
                                        pair.parameters[7], row.barycentric,
                                        row.gap)) {
                                    row.masterFace = faceIndex;
                                    ++candidate.projectedSecondRing;
                                    break;
                                }
                            }
                            if (row.masterFace != NM_INVALID_INDEX) break;
                        }
                        if (row.masterFace != NM_INVALID_INDEX) break;
                    }
                }
                if (row.masterFace == NM_INVALID_INDEX) ++candidate.unresolved;
                else {
                    ++candidate.projected;
                    if (row.gap > 0.0) ++candidate.penetrating;
                }
                candidate.rows.push_back(row);
            }
        }
    }
    result = std::move(candidate);
    return true;
}

} // namespace numi_matter_open_knee
