#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace numi_human_organ_floor {

inline constexpr std::size_t kSpatialBinCount = 4u;

struct Point3 {
    float x;
    float y;
    float z;
};

struct SpatialPartition {
    unsigned axis = 0u;
    std::array<std::vector<std::uint32_t>, kSpatialBinCount> bins;
};

// Build four disjoint, nonempty spatial patches using one horizontal axis.
// A partition is rejected if any bin boundary has no strict geometric gap;
// this keeps identical/tied vertices from being split into nominally distinct
// contact patches. The returned bins cover every input vertex exactly once.
inline SpatialPartition partitionFourSpatialBins(
    std::span<const Point3> points
) {
    if (points.size() < kSpatialBinCount ||
        points.size() > std::numeric_limits<std::uint32_t>::max() ||
        points.size() > std::numeric_limits<std::size_t>::max() /
            kSpatialBinCount)
        throw std::invalid_argument(
            "four spatial bins require a bounded, representable vertex range");
    for (const auto& point : points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
            !std::isfinite(point.z))
            throw std::invalid_argument("spatial-bin input contains a nonfinite point");
    }

    struct Candidate {
        bool valid = false;
        unsigned axis = 0u;
        double extent = 0.0;
        std::array<std::vector<std::uint32_t>, kSpatialBinCount> bins;
        std::array<float, kSpatialBinCount> minimum{};
        std::array<float, kSpatialBinCount> maximum{};
    };

    const auto coordinate = [&points](std::uint32_t vertex, unsigned axis) {
        const auto& point = points[vertex];
        return axis == 0u ? point.x : point.y;
    };
    const auto build = [&](unsigned axis) {
        Candidate candidate{};
        candidate.axis = axis;
        candidate.minimum.fill(INFINITY);
        candidate.maximum.fill(-INFINITY);
        std::vector<std::uint32_t> order(points.size());
        for (std::size_t vertex = 0u; vertex < order.size(); ++vertex)
            order[vertex] = static_cast<std::uint32_t>(vertex);
        std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
            const float ca = coordinate(a, axis);
            const float cb = coordinate(b, axis);
            return ca == cb ? a < b : ca < cb;
        });
        for (std::size_t rank = 0u; rank < order.size(); ++rank) {
            const std::size_t scaledRank = rank * kSpatialBinCount;
            const auto bin = std::min(
                kSpatialBinCount - 1u, scaledRank / order.size());
            const auto vertex = order[rank];
            candidate.bins[bin].push_back(vertex);
            candidate.minimum[bin] =
                std::min(candidate.minimum[bin], coordinate(vertex, axis));
            candidate.maximum[bin] =
                std::max(candidate.maximum[bin], coordinate(vertex, axis));
        }
        candidate.extent = static_cast<double>(candidate.maximum.back()) -
            static_cast<double>(candidate.minimum.front());
        candidate.valid = true;
        for (std::size_t bin = 0u; bin < kSpatialBinCount; ++bin) {
            if (candidate.bins[bin].empty()) candidate.valid = false;
            if (bin > 0u &&
                !(candidate.maximum[bin - 1u] < candidate.minimum[bin]))
                candidate.valid = false;
        }
        return candidate;
    };

    const Candidate x = build(0u);
    const Candidate y = build(1u);
    if (!x.valid && !y.valid)
        throw std::invalid_argument(
            "vertices cannot form four nonempty, separated spatial bins");
    const Candidate& chosen = !x.valid ? y :
        (!y.valid || x.extent >= y.extent ? x : y);

    std::vector<bool> assigned(points.size(), false);
    std::size_t assignedCount = 0u;
    for (const auto& bin : chosen.bins) {
        if (bin.empty())
            throw std::invalid_argument("spatial partition contains an empty bin");
        for (const auto vertex : bin) {
            if (vertex >= assigned.size() || assigned[vertex])
                throw std::invalid_argument(
                    "spatial partition does not assign each vertex exactly once");
            assigned[vertex] = true;
            ++assignedCount;
        }
    }
    if (assignedCount != points.size() ||
        std::find(assigned.begin(), assigned.end(), false) != assigned.end())
        throw std::invalid_argument(
            "spatial partition does not cover every input vertex");

    SpatialPartition result{};
    result.axis = chosen.axis;
    result.bins = chosen.bins;
    return result;
}

} // namespace numi_human_organ_floor
