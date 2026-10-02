#pragma once

// Adapter for the hash-bound Open Knee(s) ElementData var="fiber" sidecar.
// FEBio's per-element source vector is converted into the existing Matter FEM
// material frame: local material +X maps to the normalized source vector in
// the immutable reference frame. The raw sidecar remains the provenance
// record; this conversion does not alter geometry or infer an unloaded state.
#include "numi/matter/matter.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <span>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace numi_matter_open_knee {

struct ElementFiberRecord {
    std::uint32_t localElementId = 0;
    std::array<double, 3> sourceVector{};
};

struct SourceNodeRecord {
    std::uint32_t sourceId = 0u;
    std::array<double, 3> coordinates{};
};

struct SourceTetrahedronRecord {
    std::uint32_t sourceId = 0u;
    std::array<std::uint32_t, 4> sourceNodeIds{};
};

struct SourceVolumeMesh {
    // FEBio source material ID assigned to this named tet4 set.
    std::uint32_t sourceMaterialId = 0u;
    std::vector<SourceNodeRecord> nodes;
    std::vector<SourceTetrahedronRecord> tetrahedra;
};

inline std::uint32_t readU32LE(std::span<const std::uint8_t> bytes,
                               std::size_t address) {
    return std::uint32_t(bytes[address]) |
        (std::uint32_t(bytes[address + 1u]) << 8u) |
        (std::uint32_t(bytes[address + 2u]) << 16u) |
        (std::uint32_t(bytes[address + 3u]) << 24u);
}

inline double readF64LE(std::span<const std::uint8_t> bytes,
                        std::size_t address) {
    std::uint64_t bits = 0u;
    for (std::size_t byte = 0u; byte < 8u; ++byte)
        bits |= std::uint64_t(bytes[address + byte]) << (8u * byte);
    return std::bit_cast<double>(bits);
}

inline bool decodeSourceVolumeMeshes(
    std::span<const std::uint8_t> bytes,
    std::vector<SourceVolumeMesh>& result,
    std::string& error
) {
    constexpr std::size_t headerStride = 20u;
    constexpr std::size_t nodeStride = 28u;
    constexpr std::size_t tetrahedronStride = 20u;
    std::vector<SourceVolumeMesh> candidate;
    std::unordered_set<std::uint32_t> materialIds;
    std::size_t offset = 0u;
    while (offset < bytes.size()) {
        if (bytes.size() - offset < headerStride) {
            error = "source volume mesh has a truncated group header";
            return false;
        }
        if (bytes[offset] != 'N' || bytes[offset + 1u] != 'O' ||
            bytes[offset + 2u] != 'K' || bytes[offset + 3u] != 'T') {
            error = "source volume mesh has an invalid group magic";
            return false;
        }
        const std::uint32_t version = readU32LE(bytes, offset + 4u);
        const std::uint32_t materialId = readU32LE(bytes, offset + 8u);
        const std::uint32_t nodeCount = readU32LE(bytes, offset + 12u);
        const std::uint32_t tetrahedronCount = readU32LE(bytes, offset + 16u);
        if (version != 1u || materialId < 5u || materialId > 16u ||
            nodeCount == 0u || tetrahedronCount == 0u ||
            !materialIds.insert(materialId).second) {
            error = "source volume mesh has an unsupported or duplicate group identity";
            return false;
        }
        offset += headerStride;
        const std::size_t remaining = bytes.size() - offset;
        if (std::size_t(nodeCount) > remaining / nodeStride) {
            error = "source volume mesh node table exceeds sidecar bounds";
            return false;
        }
        const std::size_t nodeBytes = std::size_t(nodeCount) * nodeStride;
        if (std::size_t(tetrahedronCount) >
            (remaining - nodeBytes) / tetrahedronStride) {
            error = "source volume mesh tetrahedron table exceeds sidecar bounds";
            return false;
        }

        SourceVolumeMesh mesh;
        mesh.sourceMaterialId = materialId;
        mesh.nodes.reserve(nodeCount);
        mesh.tetrahedra.reserve(tetrahedronCount);
        std::unordered_set<std::uint32_t> nodeIds;
        for (std::uint32_t index = 0u; index < nodeCount; ++index) {
            const std::size_t address = offset + std::size_t(index) * nodeStride;
            SourceNodeRecord node{
                readU32LE(bytes, address),
                {readF64LE(bytes, address + 4u),
                 readF64LE(bytes, address + 12u),
                 readF64LE(bytes, address + 20u)},
            };
            if (node.sourceId == 0u || !nodeIds.insert(node.sourceId).second ||
                !std::isfinite(node.coordinates[0]) ||
                !std::isfinite(node.coordinates[1]) ||
                !std::isfinite(node.coordinates[2])) {
                error = "source volume mesh has invalid or duplicate nodes";
                return false;
            }
            mesh.nodes.push_back(node);
        }
        offset += nodeBytes;

        std::uint32_t previousElementId = 0u;
        for (std::uint32_t index = 0u; index < tetrahedronCount; ++index) {
            const std::size_t address = offset + std::size_t(index) * tetrahedronStride;
            SourceTetrahedronRecord tet;
            tet.sourceId = readU32LE(bytes, address);
            for (std::size_t node = 0u; node < tet.sourceNodeIds.size(); ++node)
                tet.sourceNodeIds[node] = readU32LE(bytes, address + 4u + 4u * node);
            if (tet.sourceId == 0u ||
                (index != 0u && tet.sourceId != previousElementId + 1u) ||
                std::ranges::any_of(tet.sourceNodeIds,
                    [&nodeIds](std::uint32_t nodeId) {
                        return !nodeIds.contains(nodeId);
                    })) {
                error = "source volume mesh has invalid element order or node references";
                return false;
            }
            previousElementId = tet.sourceId;
            mesh.tetrahedra.push_back(tet);
        }
        offset += std::size_t(tetrahedronCount) * tetrahedronStride;
        candidate.push_back(std::move(mesh));
    }
    if (candidate.empty()) {
        error = "source volume mesh has no tetrahedral groups";
        return false;
    }
    for (std::size_t index = 1u; index < candidate.size(); ++index) {
        if (candidate[index - 1u].sourceMaterialId >= candidate[index].sourceMaterialId) {
            error = "source volume mesh groups are not in increasing material order";
            return false;
        }
    }
    result.swap(candidate);
    error.clear();
    return true;
}

inline bool decodeElementFiberRecords(
    std::span<const std::uint8_t> bytes,
    std::size_t offsetBytes,
    std::uint32_t recordCount,
    std::vector<ElementFiberRecord>& result,
    std::string& error
) {
    constexpr std::size_t stride = 28u;
    if (recordCount == 0u) {
        error = "source element-fiber group is empty";
        return false;
    }
    if (offsetBytes > bytes.size() ||
        std::size_t(recordCount) > (bytes.size() - offsetBytes) / stride) {
        error = "source element-fiber group exceeds the sidecar bounds";
        return false;
    }

    std::vector<ElementFiberRecord> candidate;
    candidate.reserve(recordCount);
    for (std::uint32_t index = 0u; index < recordCount; ++index) {
        const std::size_t address = offsetBytes + std::size_t(index) * stride;
        const std::uint32_t localId = readU32LE(bytes, address);
        const std::array<double, 3> vector{
            readF64LE(bytes, address + 4u),
            readF64LE(bytes, address + 12u),
            readF64LE(bytes, address + 20u),
        };
        const double length = std::hypot(vector[0], vector[1], vector[2]);
        if (localId != index + 1u) {
            error = "source element-fiber local IDs are not consecutive";
            return false;
        }
        if (!std::isfinite(vector[0]) || !std::isfinite(vector[1]) ||
            !std::isfinite(vector[2]) || !std::isfinite(length) ||
            !(length > 1.0e-12)) {
            error = "source element-fiber direction is nonfinite or degenerate";
            return false;
        }
        candidate.push_back({localId, vector});
    }
    result.swap(candidate);
    error.clear();
    return true;
}

inline std::array<double, 4> frameFromSourceFiber(
    const std::array<double, 3>& sourceVector
) {
    const double length = std::hypot(
        sourceVector[0], sourceVector[1], sourceVector[2]
    );
    const std::array<double, 3> direction{
        sourceVector[0] / length,
        sourceVector[1] / length,
        sourceVector[2] / length,
    };
    const double cosine = direction[0];
    const double sine = std::hypot(direction[1], direction[2]);
    if (sine == 0.0 && cosine < 0.0)
        return {0.0, 0.0, 1.0, 0.0};

    // Stable shortest Hamilton rotation from +X to d. The alternate scalar
    // formula for cosine<0 retains tiny transverse components near -X where
    // 1+cosine would round to zero.
    const double scalar = cosine >= 0.0
        ? std::sqrt((1.0 + cosine) * 0.5)
        : sine / std::sqrt(2.0 * (1.0 - cosine));
    std::array<double, 4> q{
        0.0,
        -direction[2] / (2.0 * scalar),
        direction[1] / (2.0 * scalar),
        scalar,
    };
    const double norm = std::sqrt(
        q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
    );
    for (double& component : q) component /= norm;
    return q;
}

inline std::array<double, 3> rotateMaterialX(
    const std::array<double, 4>& q
) {
    const double x = q[0], y = q[1], z = q[2], w = q[3];
    return {
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + z * w),
        2.0 * (x * z - y * w),
    };
}

inline bool attachElementFiberFrames(
    numi::matter::ObjectSource& object,
    std::span<const std::uint8_t> bytes,
    std::size_t offsetBytes,
    std::uint32_t recordCount,
    const std::array<std::uint64_t, 4>& sourceIdentity,
    std::string& error
) {
    if (object.tetrahedra.size() != recordCount) {
        error = "source fiber count does not match the FEM tetrahedron order";
        return false;
    }
    if (!object.femMaterialFrameRotations.empty() ||
        std::ranges::any_of(object.femMaterialFrameSourceIdentity,
                            [](std::uint64_t word) { return word != 0u; }) ||
        std::ranges::none_of(sourceIdentity,
                             [](std::uint64_t word) { return word != 0u; })) {
        error = "source fiber frames require an empty frame owner and complete identity";
        return false;
    }
    std::vector<ElementFiberRecord> records;
    if (!decodeElementFiberRecords(bytes, offsetBytes, recordCount, records, error))
        return false;

    std::vector<std::array<double, 4>> frames;
    frames.reserve(records.size());
    for (const ElementFiberRecord& record : records)
        frames.push_back(frameFromSourceFiber(record.sourceVector));
    object.femMaterialFrameRotations.swap(frames);
    object.femMaterialFrameSourceIdentity = sourceIdentity;
    error.clear();
    return true;
}

inline bool attachUniformElementFiberFrames(
    numi::matter::ObjectSource& object,
    const std::array<double, 3>& sourceVector,
    const std::array<std::uint64_t, 4>& sourceIdentity,
    std::string& error
) {
    if (object.tetrahedra.empty()) {
        error = "uniform source fiber requires a nonempty FEM tetrahedron owner";
        return false;
    }
    if (!object.femMaterialFrameRotations.empty() ||
        std::ranges::any_of(object.femMaterialFrameSourceIdentity,
                            [](std::uint64_t word) { return word != 0u; }) ||
        std::ranges::none_of(sourceIdentity,
                             [](std::uint64_t word) { return word != 0u; })) {
        error = "uniform source fiber frames require an empty owner and complete identity";
        return false;
    }
    const double length = std::hypot(sourceVector[0], sourceVector[1], sourceVector[2]);
    if (!std::isfinite(sourceVector[0]) || !std::isfinite(sourceVector[1]) ||
        !std::isfinite(sourceVector[2]) || !std::isfinite(length) ||
        !(length > 1.0e-12)) {
        error = "uniform source fiber direction is nonfinite or degenerate";
        return false;
    }
    const auto frame = frameFromSourceFiber(sourceVector);
    std::vector<std::array<double, 4>> frames(object.tetrahedra.size(), frame);
    object.femMaterialFrameRotations.swap(frames);
    object.femMaterialFrameSourceIdentity = sourceIdentity;
    error.clear();
    return true;
}

} // namespace numi_matter_open_knee
