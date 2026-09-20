#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace metalrobo {

enum class NumiHumanCartilageStatus : std::uint32_t {
    success = 0u,
    truncatedPayload,
    invalidPayload,
    sourceMismatch,
    incompleteCoverage,
    nonpositiveVolume,
};

struct NumiHumanCostalCartilageRegion {
    std::array<char, 8u> memberId{};
    std::uint32_t side = 0u;
    std::uint32_t ribLevel = 0u;
    std::uint32_t firstNode = 0u;
    std::uint32_t nodeCount = 0u;
    std::uint32_t firstTetrahedron = 0u;
    std::uint32_t tetrahedronCount = 0u;
    std::uint32_t sternalAttachmentNodeCount = 0u;
    std::uint32_t ribAttachmentNodeCount = 0u;
    std::uint32_t sourceVertexCount = 0u;
    std::uint32_t sourceTriangleCount = 0u;
    float exactVolume = 0.0f;
    float voxelVolume = 0.0f;
    float voxelSpacing = 0.0f;
    float relativeVolumeError = 0.0f;
};

struct NumiHumanCostalCartilageNode {
    std::array<float, 3u> restPosition{};
    float compiledMassKg = 0.0f;
    std::uint32_t flags = 0u;
    std::uint32_t regionIndex = 0u;
};

struct NumiHumanCostalCartilageTetrahedron {
    std::array<std::uint32_t, 4u> node{};
    std::uint32_t regionIndex = 0u;
};

struct NumiHumanCostalCartilagePayload {
    std::uint32_t payloadAbi = 0u;
    float densityKgM3 = 0.0f;
    float attachmentDistanceMeters = 0.0f;
    std::array<std::uint8_t, 32u> sourceSha256{};
    std::vector<NumiHumanCostalCartilageRegion> regions;
    std::vector<NumiHumanCostalCartilageNode> nodes;
    std::vector<NumiHumanCostalCartilageTetrahedron> tetrahedra;
};

struct NumiHumanCartilageDiagnostics {
    NumiHumanCartilageStatus status = NumiHumanCartilageStatus::success;
    std::uint32_t failingIndex = 0xffffffffu;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == NumiHumanCartilageStatus::success;
    }
};

inline constexpr std::uint32_t
    NUMI_HUMAN_COSTAL_CARTILAGE_STERNAL_ATTACHMENT = 1u;
inline constexpr std::uint32_t
    NUMI_HUMAN_COSTAL_CARTILAGE_RIB_ATTACHMENT = 2u;

[[nodiscard]] NumiHumanCartilageDiagnostics decodeNumiHumanCostalCartilagePayload(
    std::span<const std::byte> bytes,
    std::span<const std::uint8_t> expectedSourceSha256,
    NumiHumanCostalCartilagePayload& payload
);

[[nodiscard]] const char* numiHumanCartilageStatusName(
    NumiHumanCartilageStatus status
) noexcept;

} // namespace metalrobo
