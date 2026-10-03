#pragma once

// Human-compiled source node/body ownership. These rows preserve FEBio source
// IDs; a Matter world builder resolves them to cooked FEM nodes and rigid
// proxies before the existing coupled solver can execute the attachments.
#include "numi/matter/open_knee_fiber_field.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace numi_matter_open_knee {

struct SourceRigidTieRecord {
    std::uint32_t sourceNodeId = 0u;
    std::uint32_t sourceMaterialId = 0u;
    std::uint32_t rigidBodyMaterialId = 0u;
    std::uint32_t sourceTieSetIndex = 0u;
};

struct SourceRigidTieProgram {
    std::array<std::uint8_t, 32> deckSHA256{};
    std::array<std::uint8_t, 32> geometrySHA256{};
    std::vector<SourceRigidTieRecord> rows;
};

inline bool decodeSourceRigidTieProgram(
    const std::span<const std::uint8_t> bytes,
    SourceRigidTieProgram& result, std::string& error
) {
    constexpr std::size_t headerBytes = 88u;
    constexpr std::size_t rowBytes = 16u;
    constexpr std::array<std::uint8_t, 8> magic{
        'N', 'H', 'T', 'I', 'E', 'S', '1', 0u};
    if (bytes.size() < headerBytes ||
        !std::equal(magic.begin(), magic.end(), bytes.begin()) ||
        readU32LE(bytes, 8u) != 1u || readU32LE(bytes, 12u) != 18u ||
        readU32LE(bytes, 20u) != 0u) {
        error = "source rigid-tie program header is invalid";
        return false;
    }
    const std::uint32_t count = readU32LE(bytes, 16u);
    if (count != 29427u || bytes.size() != headerBytes +
        std::size_t(count) * rowBytes) {
        error = "source rigid-tie program row count or length changed";
        return false;
    }
    SourceRigidTieProgram candidate;
    std::copy_n(bytes.begin() + 24u, 32u, candidate.deckSHA256.begin());
    std::copy_n(bytes.begin() + 56u, 32u, candidate.geometrySHA256.begin());
    if (std::ranges::all_of(candidate.deckSHA256, [](auto byte) { return byte == 0u; }) ||
        std::ranges::all_of(candidate.geometrySHA256, [](auto byte) { return byte == 0u; })) {
        error = "source rigid-tie program has no source identity";
        return false;
    }
    candidate.rows.reserve(count);
    std::uint32_t previousNode = 0u;
    constexpr std::array<std::uint32_t, 9> bodyIds{
        1u, 2u, 3u, 4u, 17u, 18u, 19u, 20u, 21u};
    for (std::uint32_t index = 0u; index < count; ++index) {
        const std::size_t offset = headerBytes + std::size_t(index) * rowBytes;
        const SourceRigidTieRecord row{
            readU32LE(bytes, offset), readU32LE(bytes, offset + 4u),
            readU32LE(bytes, offset + 8u), readU32LE(bytes, offset + 12u)};
        if (row.sourceNodeId <= previousNode ||
            row.sourceMaterialId < 5u || row.sourceMaterialId > 16u ||
            std::ranges::find(bodyIds, row.rigidBodyMaterialId) == bodyIds.end() ||
            row.sourceTieSetIndex >= 18u) {
            error = "source rigid-tie row has invalid or duplicate ownership";
            return false;
        }
        candidate.rows.push_back(row);
        previousNode = row.sourceNodeId;
    }
    result = std::move(candidate);
    error.clear();
    return true;
}

} // namespace numi_matter_open_knee
