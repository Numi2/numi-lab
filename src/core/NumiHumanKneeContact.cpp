#include "metalrobo/NumiHumanKneeContact.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

using Point = std::array<double, 3u>;

Point add(const Point& a, const Point& b) {
    return {a[0u] + b[0u], a[1u] + b[1u], a[2u] + b[2u]};
}

Point subtract(const Point& a, const Point& b) {
    return {a[0u] - b[0u], a[1u] - b[1u], a[2u] - b[2u]};
}

Point scale(const Point& value, const double amount) {
    return {value[0u] * amount, value[1u] * amount, value[2u] * amount};
}

double dot(const Point& a, const Point& b) {
    return a[0u] * b[0u] + a[1u] * b[1u] + a[2u] * b[2u];
}

Point cross(const Point& a, const Point& b) {
    return {
        a[1u] * b[2u] - a[2u] * b[1u],
        a[2u] * b[0u] - a[0u] * b[2u],
        a[0u] * b[1u] - a[1u] * b[0u],
    };
}

double squaredLength(const Point& value) { return dot(value, value); }

double length(const Point& value) { return std::sqrt(squaredLength(value)); }

bool finite(const Point& value) {
    return std::all_of(value.begin(), value.end(), [](const double component) {
        return std::isfinite(component);
    });
}

Point barycentricPoint(
    const std::array<Point, 3u>& triangle,
    const std::array<double, 3u>& barycentric
) {
    return add(add(scale(triangle[0u], barycentric[0u]),
                   scale(triangle[1u], barycentric[1u])),
               scale(triangle[2u], barycentric[2u]));
}

struct ClosestPoint {
    Point point{};
    std::array<double, 3u> barycentric{};
    double squaredDistance = std::numeric_limits<double>::infinity();
};

// Ericson's region tests provide an exact closest point on one triangle and
// stable barycentric weights for equal-and-opposite master traction scatter.
ClosestPoint closestPointOnTriangle(
    const Point& point,
    const std::array<Point, 3u>& triangle
) {
    const Point ab = subtract(triangle[1u], triangle[0u]);
    const Point ac = subtract(triangle[2u], triangle[0u]);
    const Point ap = subtract(point, triangle[0u]);
    const double d1 = dot(ab, ap);
    const double d2 = dot(ac, ap);
    std::array<double, 3u> weights{};
    if (d1 <= 0.0 && d2 <= 0.0) {
        weights = {1.0, 0.0, 0.0};
    } else {
        const Point bp = subtract(point, triangle[1u]);
        const double d3 = dot(ab, bp);
        const double d4 = dot(ac, bp);
        if (d3 >= 0.0 && d4 <= d3) {
            weights = {0.0, 1.0, 0.0};
        } else {
            const double vc = d1 * d4 - d3 * d2;
            if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
                const double v = d1 / (d1 - d3);
                weights = {1.0 - v, v, 0.0};
            } else {
                const Point cp = subtract(point, triangle[2u]);
                const double d5 = dot(ab, cp);
                const double d6 = dot(ac, cp);
                if (d6 >= 0.0 && d5 <= d6) {
                    weights = {0.0, 0.0, 1.0};
                } else {
                    const double vb = d5 * d2 - d1 * d6;
                    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
                        const double w = d2 / (d2 - d6);
                        weights = {1.0 - w, 0.0, w};
                    } else {
                        const double va = d3 * d6 - d5 * d4;
                        if (va <= 0.0 && (d4 - d3) >= 0.0 &&
                            (d5 - d6) >= 0.0) {
                            const double w = (d4 - d3) /
                                ((d4 - d3) + (d5 - d6));
                            weights = {0.0, 1.0 - w, w};
                        } else {
                            const double denominator = 1.0 / (va + vb + vc);
                            const double v = vb * denominator;
                            const double w = vc * denominator;
                            weights = {1.0 - v - w, v, w};
                        }
                    }
                }
            }
        }
    }
    const Point closest = barycentricPoint(triangle, weights);
    return {
        .point = closest,
        .barycentric = weights,
        .squaredDistance = squaredLength(subtract(point, closest)),
    };
}

struct Bounds {
    Point minimum{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()};
    Point maximum{
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()};
};

void include(Bounds& bounds, const Point& point) {
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        bounds.minimum[axis] = std::min(bounds.minimum[axis], point[axis]);
        bounds.maximum[axis] = std::max(bounds.maximum[axis], point[axis]);
    }
}

double squaredDistance(const Bounds& bounds, const Point& point) {
    double result = 0.0;
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        const double delta = point[axis] < bounds.minimum[axis]
            ? bounds.minimum[axis] - point[axis]
            : point[axis] > bounds.maximum[axis]
            ? point[axis] - bounds.maximum[axis] : 0.0;
        result += delta * delta;
    }
    return result;
}

struct TriangleReference {
    std::array<std::uint32_t, 3u> nodes{};
    Bounds bounds;
    Point centroid{};
};

struct BVHNode {
    Bounds bounds;
    std::uint32_t first = 0u;
    std::uint32_t count = 0u;
    std::uint32_t left = NUMI_HUMAN_KNEE_INVALID_INDEX;
    std::uint32_t right = NUMI_HUMAN_KNEE_INVALID_INDEX;
};

class TriangleBVH {
public:
    TriangleBVH(
        const NumiHumanKneePayload& payload,
        const NumiHumanKneeSurface& surface,
        const std::span<const Point> points
    ) : points_(points) {
        triangles_.reserve(surface.faceCount);
        for (std::uint32_t local = 0u; local < surface.faceCount; ++local) {
            const auto& nodes = payload.faces[surface.firstFace + local];
            TriangleReference triangle{.nodes = nodes};
            for (const std::uint32_t node : nodes)
                include(triangle.bounds, points_[node]);
            triangle.centroid = scale(add(add(
                points_[nodes[0u]], points_[nodes[1u]]),
                points_[nodes[2u]]), 1.0 / 3.0);
            triangles_.push_back(triangle);
        }
        order_.resize(triangles_.size());
        std::iota(order_.begin(), order_.end(), 0u);
        nodes_.reserve(2u * triangles_.size());
        build(0u, static_cast<std::uint32_t>(order_.size()));
    }

    struct QueryResult {
        std::array<std::uint32_t, 3u> nodes{};
        ClosestPoint closest;
    };

    QueryResult closest(const Point& point) const {
        QueryResult result;
        if (nodes_.empty()) return result;
        query(0u, point, result);
        return result;
    }

private:
    std::uint32_t build(const std::uint32_t first, const std::uint32_t count) {
        const std::uint32_t index = static_cast<std::uint32_t>(nodes_.size());
        nodes_.push_back({.first = first, .count = count});
        for (std::uint32_t local = 0u; local < count; ++local) {
            const auto& triangle = triangles_[order_[first + local]];
            include(nodes_[index].bounds, triangle.bounds.minimum);
            include(nodes_[index].bounds, triangle.bounds.maximum);
        }
        if (count <= 8u) return index;
        Point extent = subtract(
            nodes_[index].bounds.maximum, nodes_[index].bounds.minimum);
        const std::uint32_t axis = extent[1u] > extent[0u]
            ? (extent[2u] > extent[1u] ? 2u : 1u)
            : (extent[2u] > extent[0u] ? 2u : 0u);
        const std::uint32_t middle = first + count / 2u;
        std::nth_element(
            order_.begin() + first, order_.begin() + middle,
            order_.begin() + first + count,
            [&](const std::uint32_t a, const std::uint32_t b) {
                const double av = triangles_[a].centroid[axis];
                const double bv = triangles_[b].centroid[axis];
                return av == bv ? a < b : av < bv;
            });
        nodes_[index].count = 0u;
        nodes_[index].left = build(first, middle - first);
        nodes_[index].right = build(middle, first + count - middle);
        return index;
    }

    void query(
        const std::uint32_t nodeIndex,
        const Point& point,
        QueryResult& result
    ) const {
        const BVHNode& node = nodes_[nodeIndex];
        if (squaredDistance(node.bounds, point) >
            result.closest.squaredDistance) return;
        if (node.count != 0u) {
            for (std::uint32_t local = 0u; local < node.count; ++local) {
                const auto& triangle = triangles_[order_[node.first + local]];
                const std::array<Point, 3u> positions{
                    points_[triangle.nodes[0u]],
                    points_[triangle.nodes[1u]],
                    points_[triangle.nodes[2u]],
                };
                const ClosestPoint candidate =
                    closestPointOnTriangle(point, positions);
                if (candidate.squaredDistance <
                    result.closest.squaredDistance) {
                    result.nodes = triangle.nodes;
                    result.closest = candidate;
                }
            }
            return;
        }
        const double leftDistance = squaredDistance(
            nodes_[node.left].bounds, point);
        const double rightDistance = squaredDistance(
            nodes_[node.right].bounds, point);
        const std::uint32_t near = leftDistance <= rightDistance
            ? node.left : node.right;
        const std::uint32_t far = near == node.left ? node.right : node.left;
        query(near, point, result);
        query(far, point, result);
    }

    std::span<const Point> points_;
    std::vector<TriangleReference> triangles_;
    std::vector<std::uint32_t> order_;
    std::vector<BVHNode> nodes_;
};

NumiHumanKneeContactDiagnostics fail(
    const NumiHumanKneeContactStatus status,
    std::string message,
    const std::uint32_t index = NUMI_HUMAN_KNEE_INVALID_INDEX
) {
    return {.status = status, .failingIndex = index,
            .message = std::move(message)};
}

bool isArticular(const NumiHumanKneeRegionKind kind) {
    return kind == NumiHumanKneeRegionKind::cartilage ||
        kind == NumiHumanKneeRegionKind::meniscus;
}

double foundationStiffness(
    const NumiHumanKneeContactMaterial& material
) {
    if (!(std::isfinite(material.elasticModulusPascals) &&
          material.elasticModulusPascals > 0.0 &&
          std::isfinite(material.poissonRatio) &&
          material.poissonRatio >= 0.0 && material.poissonRatio < 0.5 &&
          std::isfinite(material.thicknessMeters) &&
          material.thicknessMeters > 0.0)) return 0.0;
    return ((1.0 - material.poissonRatio) *
            material.elasticModulusPascals) /
        ((1.0 + material.poissonRatio) *
         (1.0 - 2.0 * material.poissonRatio) *
         material.thicknessMeters);
}

} // namespace

NumiHumanKneeContactDiagnostics buildNumiHumanKneeArticularContactModel(
    const NumiHumanKneePayload& payload,
    const std::span<const Point> referenceWorldNodes,
    const std::span<const NumiHumanKneeContactRegionMaterial> materials,
    NumiHumanKneeContactModel& model
) {
    model = {};
    if (payload.regions.empty() || payload.surfaces.empty() ||
        payload.surfacePairs.empty() ||
        referenceWorldNodes.size() != payload.nodes.size() ||
        std::any_of(referenceWorldNodes.begin(), referenceWorldNodes.end(),
                    [](const Point& point) { return !finite(point); }))
        return fail(NumiHumanKneeContactStatus::invalidInput,
                    "knee contact input is incomplete or non-finite");

    std::vector<NumiHumanKneeContactMaterial> byRegion(
        payload.regions.size());
    std::vector<bool> hasMaterial(payload.regions.size(), false);
    for (std::uint32_t index = 0u; index < materials.size(); ++index) {
        const auto& source = materials[index];
        if (source.regionIndex >= payload.regions.size() ||
            hasMaterial[source.regionIndex] ||
            foundationStiffness(source.material) == 0.0)
            return fail(NumiHumanKneeContactStatus::invalidMaterial,
                        "knee contact material binding is invalid", index);
        byRegion[source.regionIndex] = source.material;
        hasMaterial[source.regionIndex] = true;
    }

    model.nodeCount = static_cast<std::uint32_t>(payload.nodes.size());
    for (std::uint32_t pairIndex = 0u;
         pairIndex < payload.surfacePairs.size(); ++pairIndex) {
        const auto& sourcePair = payload.surfacePairs[pairIndex];
        const auto& masterSurface = payload.surfaces[sourcePair.masterSurface];
        const auto& slaveSurface = payload.surfaces[sourcePair.slaveSurface];
        const auto& masterRegion = payload.regions[masterSurface.regionIndex];
        const auto& slaveRegion = payload.regions[slaveSurface.regionIndex];
        if (!isArticular(masterRegion.kind) ||
            !isArticular(slaveRegion.kind)) continue;
        if (!hasMaterial[masterSurface.regionIndex] ||
            !hasMaterial[slaveSurface.regionIndex])
            return fail(NumiHumanKneeContactStatus::invalidMaterial,
                        "articular region has no material", pairIndex);

        const double masterStiffness = foundationStiffness(
            byRegion[masterSurface.regionIndex]);
        const double slaveStiffness = foundationStiffness(
            byRegion[slaveSurface.regionIndex]);
        const double effectiveStiffness = 1.0 /
            (1.0 / masterStiffness + 1.0 / slaveStiffness);
        if (!std::isfinite(effectiveStiffness) || effectiveStiffness <= 0.0)
            return fail(NumiHumanKneeContactStatus::invalidMaterial,
                        "articular series compliance is invalid", pairIndex);

        TriangleBVH master(payload, masterSurface, referenceWorldNodes);
        std::unordered_map<std::uint32_t, double> slaveAreas;
        for (std::uint32_t local = 0u;
             local < slaveSurface.faceCount; ++local) {
            const auto& face = payload.faces[slaveSurface.firstFace + local];
            const Point rawNormal = cross(
                subtract(referenceWorldNodes[face[1u]],
                         referenceWorldNodes[face[0u]]),
                subtract(referenceWorldNodes[face[2u]],
                         referenceWorldNodes[face[0u]]));
            const double area = 0.5 * length(rawNormal);
            if (!std::isfinite(area) || area <= 1.0e-14)
                return fail(NumiHumanKneeContactStatus::invalidTopology,
                            "articular contact face is degenerate", pairIndex);
            for (const std::uint32_t node : face)
                slaveAreas[node] += area / 3.0;
        }

        NumiHumanKneeContactPairModel pair{
            .name = sourcePair.name,
            .sourcePairIndex = pairIndex,
            .masterRegionIndex = masterSurface.regionIndex,
            .slaveRegionIndex = slaveSurface.regionIndex,
            .firstSample = static_cast<std::uint32_t>(model.samples.size()),
            .effectiveFoundationStiffnessPascalsPerMeter =
                effectiveStiffness,
        };
        std::vector<std::pair<std::uint32_t, double>> orderedAreas(
            slaveAreas.begin(), slaveAreas.end());
        std::sort(orderedAreas.begin(), orderedAreas.end());
        for (const auto& [slaveNode, area] : orderedAreas) {
            const Point slavePoint = referenceWorldNodes[slaveNode];
            const auto closest = master.closest(slavePoint);
            if (!std::isfinite(closest.closest.squaredDistance))
                return fail(NumiHumanKneeContactStatus::invalidTopology,
                            "articular closest-point query failed", pairIndex);
            Point separation = subtract(slavePoint, closest.closest.point);
            const double separationLength = length(separation);
            Point normal{};
            if (separationLength > 1.0e-10) {
                normal = scale(separation, 1.0 / separationLength);
            } else {
                const std::array<Point, 3u> triangle{
                    referenceWorldNodes[closest.nodes[0u]],
                    referenceWorldNodes[closest.nodes[1u]],
                    referenceWorldNodes[closest.nodes[2u]],
                };
                normal = cross(subtract(triangle[1u], triangle[0u]),
                               subtract(triangle[2u], triangle[0u]));
                const double normalLength = length(normal);
                if (normalLength <= 1.0e-12)
                    return fail(NumiHumanKneeContactStatus::invalidTopology,
                                "articular closest triangle is degenerate",
                                pairIndex);
                normal = scale(normal, 1.0 / normalLength);
            }
            model.samples.push_back({
                .slaveNode = slaveNode,
                .masterNodes = closest.nodes,
                .masterBarycentric = closest.closest.barycentric,
                .referenceNormal = normal,
                .tributaryAreaSquareMeters = area,
                // Store the same projection evaluated at runtime. Using the
                // Euclidean length here creates a roundoff-scale artificial
                // preload even when currentWorldNodes are bitwise identical
                // to the reference configuration.
                .referenceSeparationMeters = dot(separation, normal),
            });
            pair.tributaryAreaSquareMeters += area;
        }
        pair.sampleCount = static_cast<std::uint32_t>(
            model.samples.size()) - pair.firstSample;
        if (pair.sampleCount == 0u ||
            !(std::isfinite(pair.tributaryAreaSquareMeters) &&
              pair.tributaryAreaSquareMeters > 0.0))
            return fail(NumiHumanKneeContactStatus::invalidTopology,
                        "articular pair has no slave surface measure",
                        pairIndex);
        model.pairs.push_back(std::move(pair));
    }
    if (model.pairs.size() != 7u)
        return fail(NumiHumanKneeContactStatus::incompleteAnatomy,
                    "Open Knee articular pair coverage is not exactly seven");
    return {};
}

NumiHumanKneeContactDiagnostics evaluateNumiHumanKneeContact(
    const NumiHumanKneeContactModel& model,
    const std::span<const Point> currentWorldNodes,
    const double prescribedClosureMeters,
    NumiHumanKneeContactResult& result
) {
    result = {};
    if (model.nodeCount == 0u || currentWorldNodes.size() != model.nodeCount ||
        !std::isfinite(prescribedClosureMeters) ||
        prescribedClosureMeters < 0.0 ||
        std::any_of(currentWorldNodes.begin(), currentWorldNodes.end(),
                    [](const Point& point) { return !finite(point); }))
        return fail(NumiHumanKneeContactStatus::invalidInput,
                    "knee contact evaluation input is invalid");
    result.nodalForcesNewtons.resize(model.nodeCount);
    result.pairs.reserve(model.pairs.size());
    for (std::uint32_t pairIndex = 0u;
         pairIndex < model.pairs.size(); ++pairIndex) {
        const auto& pair = model.pairs[pairIndex];
        NumiHumanKneeContactPairResult pairResult{.name = pair.name};
        pairResult.minimumGapChangeMeters =
            std::numeric_limits<double>::infinity();
        for (std::uint32_t local = 0u; local < pair.sampleCount; ++local) {
            const auto& sample = model.samples[pair.firstSample + local];
            const std::array<Point, 3u> masterTriangle{
                currentWorldNodes[sample.masterNodes[0u]],
                currentWorldNodes[sample.masterNodes[1u]],
                currentWorldNodes[sample.masterNodes[2u]],
            };
            const Point masterPoint = barycentricPoint(
                masterTriangle, sample.masterBarycentric);
            const Point separation = subtract(
                currentWorldNodes[sample.slaveNode], masterPoint);
            const double signedSeparation = dot(
                separation, sample.referenceNormal);
            const double gapChange = signedSeparation -
                sample.referenceSeparationMeters;
            const double overclosure = std::max(
                0.0, prescribedClosureMeters - gapChange);
            pairResult.minimumGapChangeMeters = std::min(
                pairResult.minimumGapChangeMeters, gapChange);
            if (overclosure == 0.0) continue;
            const double pressure =
                pair.effectiveFoundationStiffnessPascalsPerMeter *
                overclosure;
            const double forceMagnitude = pressure *
                sample.tributaryAreaSquareMeters;
            if (!(std::isfinite(pressure) && pressure >= 0.0 &&
                  std::isfinite(forceMagnitude) && forceMagnitude >= 0.0))
                return fail(NumiHumanKneeContactStatus::numericalFailure,
                            "articular contact force is non-finite", pairIndex);
            const Point slaveForce = scale(
                sample.referenceNormal, forceMagnitude);
            result.nodalForcesNewtons[sample.slaveNode] = add(
                result.nodalForcesNewtons[sample.slaveNode], slaveForce);
            for (std::uint32_t corner = 0u; corner < 3u; ++corner) {
                const Point masterForce = scale(
                    slaveForce, -sample.masterBarycentric[corner]);
                result.nodalForcesNewtons[sample.masterNodes[corner]] = add(
                    result.nodalForcesNewtons[sample.masterNodes[corner]],
                    masterForce);
            }
            ++pairResult.activeSampleCount;
            pairResult.maximumPressurePascals = std::max(
                pairResult.maximumPressurePascals, pressure);
            pairResult.contactAreaSquareMeters +=
                sample.tributaryAreaSquareMeters;
            pairResult.normalForceNewtons += forceMagnitude;
            pairResult.storedEnergyJoules += 0.5 * pressure * overclosure *
                sample.tributaryAreaSquareMeters;
        }
        if (!std::isfinite(pairResult.minimumGapChangeMeters))
            return fail(NumiHumanKneeContactStatus::numericalFailure,
                        "articular pair produced no finite gap", pairIndex);
        result.forceL1Newtons += 2.0 * pairResult.normalForceNewtons;
        result.storedEnergyJoules += pairResult.storedEnergyJoules;
        result.pairs.push_back(std::move(pairResult));
    }
    for (std::uint32_t node = 0u; node < model.nodeCount; ++node) {
        const Point& force = result.nodalForcesNewtons[node];
        result.forceResidualNewtons = add(
            result.forceResidualNewtons, force);
        result.momentResidualNewtonMeters = add(
            result.momentResidualNewtonMeters,
            cross(currentWorldNodes[node], force));
    }
    if (!finite(result.forceResidualNewtons) ||
        !finite(result.momentResidualNewtonMeters) ||
        !std::isfinite(result.forceL1Newtons) ||
        !std::isfinite(result.storedEnergyJoules))
        return fail(NumiHumanKneeContactStatus::numericalFailure,
                    "articular contact result is non-finite");
    return {};
}

const char* numiHumanKneeContactStatusName(
    const NumiHumanKneeContactStatus status
) noexcept {
    switch (status) {
    case NumiHumanKneeContactStatus::success: return "success";
    case NumiHumanKneeContactStatus::invalidInput: return "invalid_input";
    case NumiHumanKneeContactStatus::incompleteAnatomy:
        return "incomplete_anatomy";
    case NumiHumanKneeContactStatus::invalidMaterial:
        return "invalid_material";
    case NumiHumanKneeContactStatus::invalidTopology:
        return "invalid_topology";
    case NumiHumanKneeContactStatus::numericalFailure:
        return "numerical_failure";
    }
    return "unknown";
}

} // namespace metalrobo
