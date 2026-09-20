#include "metalrobo/NumiHumanContinuumMap.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <utility>

namespace metalrobo {
namespace {

using Point = std::array<double, 3u>;

struct Edge {
    std::uint32_t node = 0u;
    double length = 0.0;
};

[[nodiscard]] bool finitePoint(const Point& point) noexcept {
    return std::all_of(point.begin(), point.end(), [](const double value) {
        return std::isfinite(value);
    });
}

[[nodiscard]] Point subtract(const Point& a, const Point& b) noexcept {
    return {a[0u] - b[0u], a[1u] - b[1u], a[2u] - b[2u]};
}

[[nodiscard]] Point add(const Point& a, const Point& b) noexcept {
    return {a[0u] + b[0u], a[1u] + b[1u], a[2u] + b[2u]};
}

[[nodiscard]] Point scale(const Point& point, const double value) noexcept {
    return {point[0u] * value, point[1u] * value, point[2u] * value};
}

[[nodiscard]] double dot(const Point& a, const Point& b) noexcept {
    return a[0u] * b[0u] + a[1u] * b[1u] + a[2u] * b[2u];
}

[[nodiscard]] Point cross(const Point& a, const Point& b) noexcept {
    return {
        a[1u] * b[2u] - a[2u] * b[1u],
        a[2u] * b[0u] - a[0u] * b[2u],
        a[0u] * b[1u] - a[1u] * b[0u]};
}

[[nodiscard]] double length(const Point& point) noexcept {
    return std::sqrt(dot(point, point));
}

[[nodiscard]] double signedSixVolume(
    const std::span<const Point> points,
    const std::array<std::uint32_t, 4u>& tetrahedron
) noexcept {
    const Point a = points[tetrahedron[0u]];
    const Point ab = subtract(points[tetrahedron[1u]], a);
    const Point ac = subtract(points[tetrahedron[2u]], a);
    const Point ad = subtract(points[tetrahedron[3u]], a);
    return dot(ab, cross(ac, ad));
}

[[nodiscard]] bool normalizedQuaternion(
    const std::array<double, 4u>& quaternion,
    std::array<double, 4u>& normalized
) noexcept {
    double normSquared = 0.0;
    for (const double component : quaternion) {
        if (!std::isfinite(component)) return false;
        normSquared += component * component;
    }
    if (!std::isfinite(normSquared) || normSquared <= 1.0e-20) return false;
    const double inverseNorm = 1.0 / std::sqrt(normSquared);
    for (std::uint32_t component = 0u; component < 4u; ++component)
        normalized[component] = quaternion[component] * inverseNorm;
    return true;
}

[[nodiscard]] Point rotate(
    const std::array<double, 4u>& quaternion,
    const Point& point
) noexcept {
    const Point q{quaternion[0u], quaternion[1u], quaternion[2u]};
    const Point twiceCross = scale(cross(q, point), 2.0);
    return add(point, add(
        scale(twiceCross, quaternion[3u]), cross(q, twiceCross)));
}

[[nodiscard]] Point transformReferencePoint(
    const NumiHumanContinuumBodyMap& body,
    const std::array<double, 4u>& referenceOrientation,
    const std::array<double, 4u>& targetOrientation,
    const Point& point
) noexcept {
    const std::array<double, 4u> inverseReference{
        -referenceOrientation[0u], -referenceOrientation[1u],
        -referenceOrientation[2u], referenceOrientation[3u]};
    const Point local = rotate(
        inverseReference, subtract(point, body.referencePose.position));
    return add(body.targetPose.position, rotate(targetOrientation, local));
}

[[nodiscard]] NumiHumanContinuumMapDiagnostics fail(
    const NumiHumanContinuumMapStatus status,
    const std::uint32_t failingIndex,
    std::string message
) {
    NumiHumanContinuumMapDiagnostics diagnostics;
    diagnostics.status = status;
    diagnostics.failingIndex = failingIndex;
    diagnostics.message = std::move(message);
    return diagnostics;
}

} // namespace

NumiHumanContinuumMapDiagnostics mapNumiHumanContinuumToMovingEntheses(
    const std::span<const Point> referenceWorldPoints,
    const std::span<const std::array<std::uint32_t, 4u>> tetrahedra,
    const std::span<const std::uint32_t> anchorBodyIndices,
    const std::span<const NumiHumanContinuumBodyMap> bodyMaps,
    NumiHumanContinuumMapResult& result,
    const NumiHumanContinuumMapConfig& config
) {
    result = {};
    if (referenceWorldPoints.empty() || tetrahedra.empty() ||
        anchorBodyIndices.size() != referenceWorldPoints.size() ||
        bodyMaps.empty() || bodyMaps.size() > 4u ||
        !std::isfinite(config.inverseDistanceExponent) ||
        config.inverseDistanceExponent <= 0.0 ||
        !std::isfinite(config.minimumJacobian) ||
        !std::isfinite(config.maximumJacobian) ||
        config.minimumJacobian <= 0.0 ||
        config.maximumJacobian <= config.minimumJacobian) {
        return fail(NumiHumanContinuumMapStatus::invalidInput,
                    NUMI_HUMAN_CONTINUUM_INVALID_INDEX,
                    "moving-enthesis map input is invalid");
    }
    for (std::uint32_t node = 0u; node < referenceWorldPoints.size(); ++node) {
        if (!finitePoint(referenceWorldPoints[node]))
            return fail(NumiHumanContinuumMapStatus::invalidInput, node,
                        "moving-enthesis reference point is non-finite");
    }

    std::vector<std::array<double, 4u>> referenceOrientations(bodyMaps.size());
    std::vector<std::array<double, 4u>> targetOrientations(bodyMaps.size());
    for (std::uint32_t owner = 0u; owner < bodyMaps.size(); ++owner) {
        if (bodyMaps[owner].bodyIndex == NUMI_HUMAN_CONTINUUM_INVALID_INDEX ||
            !finitePoint(bodyMaps[owner].referencePose.position) ||
            !finitePoint(bodyMaps[owner].targetPose.position) ||
            !normalizedQuaternion(bodyMaps[owner].referencePose.orientation,
                                  referenceOrientations[owner]) ||
            !normalizedQuaternion(bodyMaps[owner].targetPose.orientation,
                                  targetOrientations[owner])) {
            return fail(NumiHumanContinuumMapStatus::invalidInput, owner,
                        "moving-enthesis body pose is invalid");
        }
        for (std::uint32_t previous = 0u; previous < owner; ++previous) {
            if (bodyMaps[previous].bodyIndex == bodyMaps[owner].bodyIndex)
                return fail(NumiHumanContinuumMapStatus::invalidInput, owner,
                            "moving-enthesis body owner is duplicated");
        }
    }

    const auto ownerSlot = [&](const std::uint32_t bodyIndex) {
        for (std::uint32_t owner = 0u; owner < bodyMaps.size(); ++owner) {
            if (bodyMaps[owner].bodyIndex == bodyIndex) return owner;
        }
        return NUMI_HUMAN_CONTINUUM_INVALID_INDEX;
    };
    std::vector<std::uint32_t> anchorCounts(bodyMaps.size(), 0u);
    for (std::uint32_t node = 0u; node < anchorBodyIndices.size(); ++node) {
        const std::uint32_t body = anchorBodyIndices[node];
        if (body == NUMI_HUMAN_CONTINUUM_INVALID_INDEX) continue;
        const std::uint32_t owner = ownerSlot(body);
        if (owner == NUMI_HUMAN_CONTINUUM_INVALID_INDEX)
            return fail(NumiHumanContinuumMapStatus::invalidInput, node,
                        "moving-enthesis anchor has no body map");
        ++anchorCounts[owner];
    }
    if (std::any_of(anchorCounts.begin(), anchorCounts.end(),
                    [](const std::uint32_t count) { return count == 0u; })) {
        return fail(NumiHumanContinuumMapStatus::invalidInput,
                    NUMI_HUMAN_CONTINUUM_INVALID_INDEX,
                    "moving-enthesis body has no anchor nodes");
    }

    std::vector<std::vector<Edge>> adjacency(referenceWorldPoints.size());
    constexpr std::array<std::array<std::uint32_t, 2u>, 6u> kEdges{{
        {0u, 1u}, {0u, 2u}, {0u, 3u},
        {1u, 2u}, {1u, 3u}, {2u, 3u}}};
    for (std::uint32_t index = 0u; index < tetrahedra.size(); ++index) {
        const auto& tetrahedron = tetrahedra[index];
        for (std::uint32_t a = 0u; a < 4u; ++a) {
            if (tetrahedron[a] >= referenceWorldPoints.size())
                return fail(NumiHumanContinuumMapStatus::invalidTopology,
                            index,
                            "moving-enthesis tetrahedron escapes node storage");
            for (std::uint32_t b = a + 1u; b < 4u; ++b) {
                if (tetrahedron[a] == tetrahedron[b])
                    return fail(NumiHumanContinuumMapStatus::invalidTopology,
                                index,
                                "moving-enthesis tetrahedron repeats a node");
            }
        }
        const double referenceVolume = signedSixVolume(
            referenceWorldPoints, tetrahedron);
        if (!std::isfinite(referenceVolume) ||
            std::abs(referenceVolume) <= 1.0e-18)
            return fail(NumiHumanContinuumMapStatus::invalidTopology, index,
                        "moving-enthesis tetrahedron is degenerate");
        for (const auto edge : kEdges) {
            const std::uint32_t a = tetrahedron[edge[0u]];
            const std::uint32_t b = tetrahedron[edge[1u]];
            const double edgeLength = length(subtract(
                referenceWorldPoints[b], referenceWorldPoints[a]));
            if (!std::isfinite(edgeLength) || edgeLength <= 1.0e-12)
                return fail(NumiHumanContinuumMapStatus::invalidTopology,
                            index,
                            "moving-enthesis tetrahedron has a zero edge");
            adjacency[a].push_back({b, edgeLength});
            adjacency[b].push_back({a, edgeLength});
        }
    }

    const double infinity = std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> distances(
        bodyMaps.size(), std::vector<double>(referenceWorldPoints.size(), infinity));
    using QueueEntry = std::pair<double, std::uint32_t>;
    for (std::uint32_t owner = 0u; owner < bodyMaps.size(); ++owner) {
        std::priority_queue<QueueEntry, std::vector<QueueEntry>,
                            std::greater<QueueEntry>> queue;
        for (std::uint32_t node = 0u; node < anchorBodyIndices.size(); ++node) {
            if (anchorBodyIndices[node] != bodyMaps[owner].bodyIndex) continue;
            distances[owner][node] = 0.0;
            queue.emplace(0.0, node);
        }
        while (!queue.empty()) {
            const auto [distance, node] = queue.top();
            queue.pop();
            if (distance != distances[owner][node]) continue;
            for (const Edge& edge : adjacency[node]) {
                const double candidate = distance + edge.length;
                if (candidate >= distances[owner][edge.node]) continue;
                distances[owner][edge.node] = candidate;
                queue.emplace(candidate, edge.node);
            }
        }
        for (std::uint32_t node = 0u; node < referenceWorldPoints.size(); ++node) {
            if (!std::isfinite(distances[owner][node]))
                return fail(NumiHumanContinuumMapStatus::disconnectedTopology,
                            node,
                            "moving-enthesis node is disconnected from an owner");
        }
    }

    result.targetWorldPoints.resize(referenceWorldPoints.size());
    double maximumAnchorResidual = 0.0;
    double maximumDisplacement = 0.0;
    for (std::uint32_t node = 0u; node < referenceWorldPoints.size(); ++node) {
        const Point reference = referenceWorldPoints[node];
        const std::uint32_t anchoredOwner =
            anchorBodyIndices[node] == NUMI_HUMAN_CONTINUUM_INVALID_INDEX
            ? NUMI_HUMAN_CONTINUUM_INVALID_INDEX
            : ownerSlot(anchorBodyIndices[node]);
        Point mapped{};
        if (anchoredOwner != NUMI_HUMAN_CONTINUUM_INVALID_INDEX) {
            mapped = transformReferencePoint(
                bodyMaps[anchoredOwner], referenceOrientations[anchoredOwner],
                targetOrientations[anchoredOwner], reference);
            const Point expected = transformReferencePoint(
                bodyMaps[anchoredOwner], referenceOrientations[anchoredOwner],
                targetOrientations[anchoredOwner], reference);
            maximumAnchorResidual = std::max(
                maximumAnchorResidual, length(subtract(mapped, expected)));
        } else if (bodyMaps.size() == 1u) {
            mapped = transformReferencePoint(
                bodyMaps[0u], referenceOrientations[0u],
                targetOrientations[0u], reference);
        } else {
            double weightSum = 0.0;
            for (std::uint32_t owner = 0u; owner < bodyMaps.size(); ++owner) {
                const double distance = std::max(distances[owner][node], 1.0e-12);
                const double weight = std::pow(
                    distance, -config.inverseDistanceExponent);
                const Point candidate = transformReferencePoint(
                    bodyMaps[owner], referenceOrientations[owner],
                    targetOrientations[owner], reference);
                mapped = add(mapped, scale(candidate, weight));
                weightSum += weight;
            }
            if (!std::isfinite(weightSum) || weightSum <= 0.0)
                return fail(NumiHumanContinuumMapStatus::invalidDeformation,
                            node,
                            "moving-enthesis blend weight is invalid");
            mapped = scale(mapped, 1.0 / weightSum);
        }
        if (!finitePoint(mapped))
            return fail(NumiHumanContinuumMapStatus::invalidDeformation, node,
                        "moving-enthesis mapped point is non-finite");
        result.targetWorldPoints[node] = mapped;
        maximumDisplacement = std::max(
            maximumDisplacement, length(subtract(mapped, reference)));
    }

    double minimumJacobian = std::numeric_limits<double>::infinity();
    double maximumJacobian = -std::numeric_limits<double>::infinity();
    for (std::uint32_t index = 0u; index < tetrahedra.size(); ++index) {
        const double referenceVolume = signedSixVolume(
            referenceWorldPoints, tetrahedra[index]);
        const double mappedVolume = signedSixVolume(
            result.targetWorldPoints, tetrahedra[index]);
        const double jacobian = mappedVolume / referenceVolume;
        if (!std::isfinite(jacobian) || jacobian < config.minimumJacobian ||
            jacobian > config.maximumJacobian) {
            result = {};
            return fail(NumiHumanContinuumMapStatus::invalidDeformation,
                        index,
                        "moving-enthesis map violates the Jacobian gate");
        }
        minimumJacobian = std::min(minimumJacobian, jacobian);
        maximumJacobian = std::max(maximumJacobian, jacobian);
    }

    NumiHumanContinuumMapDiagnostics diagnostics;
    diagnostics.ownerCount = static_cast<std::uint32_t>(bodyMaps.size());
    diagnostics.anchorCount = static_cast<std::uint32_t>(std::count_if(
        anchorBodyIndices.begin(), anchorBodyIndices.end(),
        [](const std::uint32_t body) {
            return body != NUMI_HUMAN_CONTINUUM_INVALID_INDEX;
        }));
    diagnostics.maximumAnchorResidualMeters = maximumAnchorResidual;
    diagnostics.maximumDisplacementMeters = maximumDisplacement;
    diagnostics.minimumJacobian = minimumJacobian;
    diagnostics.maximumJacobian = maximumJacobian;
    diagnostics.message = "moving-enthesis continuum map succeeded";
    return diagnostics;
}

const char* numiHumanContinuumMapStatusName(
    const NumiHumanContinuumMapStatus status
) noexcept {
    switch (status) {
    case NumiHumanContinuumMapStatus::success: return "success";
    case NumiHumanContinuumMapStatus::invalidInput: return "invalid_input";
    case NumiHumanContinuumMapStatus::invalidTopology:
        return "invalid_topology";
    case NumiHumanContinuumMapStatus::disconnectedTopology:
        return "disconnected_topology";
    case NumiHumanContinuumMapStatus::invalidDeformation:
        return "invalid_deformation";
    }
    return "unknown";
}

} // namespace metalrobo
