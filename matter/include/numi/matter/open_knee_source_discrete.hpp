#pragma once

// Exact source discrete connectivity. Decoding and binding are prerequisites
// for execution; neither operation applies a spring force.
#include "numi/matter/open_knee_source_graph.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace numi_matter_open_knee {

struct SourceDiscreteSetRecord {
    std::uint32_t materialId = 0u, law = 0u, curveId = 0u;
    std::uint32_t firstEdge = 0u, edgeCount = 0u;
    double stiffness = 0.0, scale = 0.0;
    std::array<std::uint8_t, 32> connectivitySHA256{};
};

struct SourceDiscreteEdgeRecord {
    std::uint32_t nodeA = 0u, nodeB = 0u;
    std::uint32_t ownerA = 0u, ownerB = 0u;
    std::array<double, 3> pointA{}, pointB{};
};

struct SourceDiscreteProgram {
    std::array<std::uint8_t, 32> deckSHA256{}, geometrySHA256{},
        geometryBinarySHA256{};
    std::vector<SourceLoadCurveRecord> curves;
    std::vector<SourceDiscreteSetRecord> sets;
    std::vector<SourceDiscreteEdgeRecord> edges;
};

inline bool decodeSourceDiscreteProgram(
    std::span<const std::uint8_t> bytes, SourceDiscreteProgram& result,
    std::string& error
) {
    constexpr std::array<std::uint8_t, 8> magic{
        'N', 'H', 'D', 'I', 'S', 'C', '1', 0u};
    constexpr std::size_t header = 120u, curveBytes = 56u,
        setBytes = 68u, edgeBytes = 64u;
    constexpr std::size_t expectedBytes = header + 9u * curveBytes +
        3u * setBytes + 406u * edgeBytes;
    if (bytes.size() != expectedBytes ||
        !std::equal(magic.begin(), magic.end(), bytes.begin()) ||
        readU32LE(bytes, 8u) != 1u || readU32LE(bytes, 12u) != 9u ||
        readU32LE(bytes, 16u) != 3u || readU32LE(bytes, 20u) != 406u) {
        error = "source discrete program header or size is invalid";
        return false;
    }
    SourceDiscreteProgram candidate;
    std::copy_n(bytes.begin() + 24u, 32u, candidate.deckSHA256.begin());
    std::copy_n(bytes.begin() + 56u, 32u, candidate.geometrySHA256.begin());
    std::copy_n(bytes.begin() + 88u, 32u,
                candidate.geometryBinarySHA256.begin());
    std::size_t offset = header;
    for (std::uint32_t id = 1u; id <= 9u; ++id) {
        SourceLoadCurveRecord curve;
        curve.id = readU32LE(bytes, offset);
        if (curve.id != id || readU32LE(bytes, offset + 4u) != 3u) {
            error = "source discrete load curve identity or point count changed";
            return false;
        }
        for (std::size_t point = 0u; point < 3u; ++point) {
            const double time = readF64LE(bytes, offset + 8u + 16u * point);
            const double value = readF64LE(bytes, offset + 16u + 16u * point);
            if (!std::isfinite(time) || !std::isfinite(value) ||
                (point != 0u && time <= curve.points.back()[0])) {
                error = "source discrete load curve contains invalid values";
                return false;
            }
            curve.points.push_back({time, value});
        }
        candidate.curves.push_back(std::move(curve));
        offset += curveBytes;
    }
    constexpr std::array<std::uint32_t, 3> counts{2u, 2u, 402u};
    for (std::size_t index = 0u; index < counts.size(); ++index) {
        SourceDiscreteSetRecord set;
        set.materialId = readU32LE(bytes, offset);
        set.law = readU32LE(bytes, offset + 4u);
        set.curveId = readU32LE(bytes, offset + 8u);
        set.firstEdge = readU32LE(bytes, offset + 12u);
        set.edgeCount = readU32LE(bytes, offset + 16u);
        set.stiffness = readF64LE(bytes, offset + 20u);
        set.scale = readF64LE(bytes, offset + 28u);
        std::copy_n(bytes.begin() + offset + 36u, 32u,
                    set.connectivitySHA256.begin());
        const std::uint32_t expectedFirst = index == 0u ? 0u :
            index == 1u ? 2u : 4u;
        if (set.materialId != index + 1u ||
            set.law != (index < 2u ? 1u : 2u) ||
            set.curveId != (index < 2u ? index + 7u : 0u) ||
            set.firstEdge != expectedFirst || set.edgeCount != counts[index] ||
            !std::isfinite(set.stiffness) || !std::isfinite(set.scale) ||
            set.stiffness != (index < 2u ? 0.0 : 1000.0) ||
            set.scale != 1.0) {
            error = "source discrete spring set has unsupported mechanics";
            return false;
        }
        candidate.sets.push_back(set);
        offset += setBytes;
    }
    for (std::uint32_t index = 0u; index < 406u; ++index) {
        SourceDiscreteEdgeRecord edge;
        edge.nodeA = readU32LE(bytes, offset);
        edge.nodeB = readU32LE(bytes, offset + 4u);
        edge.ownerA = readU32LE(bytes, offset + 8u);
        edge.ownerB = readU32LE(bytes, offset + 12u);
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            edge.pointA[axis] = readF64LE(bytes, offset + 16u + axis * 8u);
            edge.pointB[axis] = readF64LE(bytes, offset + 40u + axis * 8u);
        }
        if (edge.nodeA == edge.nodeB || edge.nodeA == 0u || edge.nodeB == 0u ||
            edge.ownerA == 0u || edge.ownerB == 0u ||
            std::ranges::any_of(edge.pointA, [](double v) { return !std::isfinite(v); }) ||
            std::ranges::any_of(edge.pointB, [](double v) { return !std::isfinite(v); })) {
            error = "source discrete spring has invalid endpoints";
            return false;
        }
        candidate.edges.push_back(edge);
        offset += edgeBytes;
    }
    if (offset != bytes.size()) {
        error = "source discrete program has trailing bytes";
        return false;
    }
    result = std::move(candidate);
    return true;
}

} // namespace numi_matter_open_knee
