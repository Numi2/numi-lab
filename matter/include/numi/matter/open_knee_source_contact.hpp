#pragma once

// Exact source sliding-contact topology and parameters. This decoder is used
// by the assembled Open Knee case; decoding does not enforce FEBio contact.
#include "numi/matter/open_knee_fiber_field.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <ranges>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace numi_matter_open_knee {

struct SourceContactSurface {
    std::string name;
    std::uint32_t materialId = 0u, firstFace = 0u, faceCount = 0u;
    std::array<std::uint8_t, 32> sourceConnectivitySHA256{};
};
struct SourceContactPair {
    std::string name;
    std::uint32_t master = 0u, slave = 0u;
    // FEBio order: laugon, tolerance, gaptol, penalty, two_pass,
    // auto_penalty, fric_coeff, search_tol, search_radius, minaug, maxaug,
    // seg_up. These remain source values until a law implements them.
    std::array<double, 12> parameters{};
    std::array<std::uint8_t, 32> sourceXMLSHA256{};
};
struct SourceContactFace {
    std::uint32_t sourceFaceId = 0u;
    std::array<std::uint32_t, 3> sourceNodes{};
};
struct SourceRigidContactNode {
    std::uint32_t sourceNodeId = 0u, materialId = 0u;
    std::array<double, 3> sourcePosition{};
};
struct SourceSlidingContactProgram {
    std::array<std::uint8_t, 32> deckSHA256{}, geometrySHA256{};
    std::array<std::uint8_t, 32> geometryBinarySHA256{}, volumeMeshSHA256{};
    std::vector<SourceContactSurface> surfaces;
    std::vector<SourceContactPair> pairs;
    std::vector<SourceContactFace> faces;
    std::vector<SourceRigidContactNode> rigidNodes;
};

inline bool decodeSourceSlidingContactProgram(
    const std::span<const std::uint8_t> bytes,
    SourceSlidingContactProgram& result, std::string& error
) {
    constexpr std::array<std::uint8_t, 8> magic{
        'N', 'H', 'C', 'N', 'T', 'P', '1', 0u};
    constexpr std::size_t header = 156u, surfaceStride = 76u,
                          pairStride = 168u, faceStride = 16u,
                          rigidNodeStride = 32u;
    if (bytes.size() < header ||
        !std::equal(magic.begin(), magic.end(), bytes.begin()) ||
        readU32LE(bytes, 8u) != 1u ||
        readU32LE(bytes, 12u) != 18u ||
        readU32LE(bytes, 16u) != 36u ||
        readU32LE(bytes, 20u) != 345070u ||
        readU32LE(bytes, 24u) != 17676u) {
        error = "source contact header or pinned topology changed";
        return false;
    }
    const std::size_t surfaces = readU32LE(bytes, 16u);
    const std::size_t pairs = readU32LE(bytes, 12u);
    const std::size_t faces = readU32LE(bytes, 20u);
    const std::size_t rigidNodes = readU32LE(bytes, 24u);
    if (bytes.size() != header + surfaces * surfaceStride +
            pairs * pairStride + faces * faceStride +
            rigidNodes * rigidNodeStride) {
        error = "source contact program size differs from its tables";
        return false;
    }
    SourceSlidingContactProgram candidate;
    std::copy_n(bytes.begin() + 28u, 32u, candidate.deckSHA256.begin());
    std::copy_n(bytes.begin() + 60u, 32u, candidate.geometrySHA256.begin());
    std::copy_n(bytes.begin() + 92u, 32u,
                candidate.geometryBinarySHA256.begin());
    std::copy_n(bytes.begin() + 124u, 32u,
                candidate.volumeMeshSHA256.begin());
    const auto nameAt = [&](const std::size_t address, std::string& name) {
        std::size_t length = 0u;
        while (length < 32u && bytes[address + length] != 0u) ++length;
        if (length == 0u || length == 32u) return false;
        for (std::size_t i = 0u; i < length; ++i)
            if (bytes[address + i] < 0x20u || bytes[address + i] > 0x7eu)
                return false;
        for (std::size_t i = length; i < 32u; ++i)
            if (bytes[address + i] != 0u) return false;
        name.assign(reinterpret_cast<const char*>(bytes.data() + address),
                    length);
        return true;
    };
    std::size_t offset = header;
    std::unordered_set<std::string> surfaceNames;
    std::uint32_t nextFace = 0u;
    candidate.surfaces.reserve(surfaces);
    for (std::size_t i = 0u; i < surfaces; ++i, offset += surfaceStride) {
        SourceContactSurface row;
        if (!nameAt(offset, row.name)) {
            error = "source contact surface name is invalid"; return false;
        }
        row.materialId = readU32LE(bytes, offset + 32u);
        row.firstFace = readU32LE(bytes, offset + 36u);
        row.faceCount = readU32LE(bytes, offset + 40u);
        std::copy_n(bytes.begin() + offset + 44u, 32u,
                    row.sourceConnectivitySHA256.begin());
        if (!surfaceNames.insert(row.name).second ||
            row.materialId == 0u || row.faceCount == 0u ||
            row.firstFace != nextFace ||
            row.faceCount > faces - nextFace) {
            error = "source contact surface range or owner is invalid";
            return false;
        }
        nextFace += row.faceCount;
        candidate.surfaces.push_back(std::move(row));
    }
    if (nextFace != faces) {
        error = "source contact surface faces do not cover the table";
        return false;
    }
    candidate.pairs.reserve(pairs);
    std::unordered_set<std::string> pairNames;
    std::vector<bool> referenced(surfaces, false);
    for (std::size_t i = 0u; i < pairs; ++i, offset += pairStride) {
        SourceContactPair row;
        if (!nameAt(offset, row.name)) {
            error = "source contact pair name is invalid"; return false;
        }
        row.master = readU32LE(bytes, offset + 32u);
        row.slave = readU32LE(bytes, offset + 36u);
        for (std::size_t k = 0u; k < row.parameters.size(); ++k)
            row.parameters[k] = readF64LE(bytes, offset + 40u + k * 8u);
        std::copy_n(bytes.begin() + offset + 136u, 32u,
                    row.sourceXMLSHA256.begin());
        if (!pairNames.insert(row.name).second ||
            row.master >= surfaces || row.slave >= surfaces ||
            row.master == row.slave ||
            std::ranges::any_of(row.parameters,
                [](const double value) { return !std::isfinite(value); }) ||
            !(row.parameters[3] > 0.0) ||
            row.parameters[2] < 0.0 || row.parameters[6] < 0.0 ||
            row.parameters[7] < 0.0 || row.parameters[8] < 0.0 ||
            row.parameters[0] != 0.0 || row.parameters[4] != 1.0 ||
            row.parameters[5] != 1.0) {
            error = "source sliding contact pair is invalid or unsupported";
            return false;
        }
        referenced[row.master] = referenced[row.slave] = true;
        candidate.pairs.push_back(std::move(row));
    }
    if (std::ranges::any_of(referenced, [](bool value) { return !value; })) {
        error = "source contact contains an unreferenced surface";
        return false;
    }
    candidate.faces.reserve(faces);
    for (std::size_t i = 0u; i < faces; ++i, offset += faceStride) {
        SourceContactFace row;
        row.sourceFaceId = readU32LE(bytes, offset);
        for (std::size_t j = 0u; j < 3u; ++j)
            row.sourceNodes[j] = readU32LE(bytes, offset + 4u + j * 4u);
        if (row.sourceFaceId == 0u || row.sourceNodes[0] == 0u ||
            row.sourceNodes[1] == 0u || row.sourceNodes[2] == 0u ||
            row.sourceNodes[0] == row.sourceNodes[1] ||
            row.sourceNodes[0] == row.sourceNodes[2] ||
            row.sourceNodes[1] == row.sourceNodes[2]) {
            error = "source contact face is invalid";
            return false;
        }
        candidate.faces.push_back(row);
    }
    candidate.rigidNodes.reserve(rigidNodes);
    std::uint32_t previousNode = 0u;
    for (std::size_t i = 0u; i < rigidNodes; ++i, offset += rigidNodeStride) {
        SourceRigidContactNode row;
        row.sourceNodeId = readU32LE(bytes, offset);
        row.materialId = readU32LE(bytes, offset + 4u);
        for (std::size_t j = 0u; j < 3u; ++j)
            row.sourcePosition[j] = readF64LE(bytes, offset + 8u + 8u * j);
        if (row.sourceNodeId <= previousNode ||
            row.materialId == 0u ||
            std::ranges::any_of(row.sourcePosition,
                [](const double value) { return !std::isfinite(value); })) {
            error = "source rigid contact node is invalid";
            return false;
        }
        previousNode = row.sourceNodeId;
        candidate.rigidNodes.push_back(row);
    }
    result = std::move(candidate);
    error.clear();
    return true;
}

} // namespace numi_matter_open_knee
