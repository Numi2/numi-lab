#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

enum class NumiHumanKneeStatus : std::uint32_t {
    success = 0u,
    truncatedPayload,
    invalidPayload,
    sourceMismatch,
    incompleteCoverage,
    invalidTopology,
};

enum class NumiHumanKneeRegionKind : std::uint32_t {
    bone = 1u,
    cartilage = 2u,
    meniscus = 3u,
    ligament = 4u,
    tendon = 5u,
};

enum class NumiHumanKneeSide : std::uint32_t {
    left = 0u,
    rightMirrored = 1u,
};

struct NumiHumanKneeRegion {
    std::string name;
    NumiHumanKneeRegionKind kind = NumiHumanKneeRegionKind::bone;
    std::uint32_t visualBodyIndex = 0u;
    std::uint32_t firstNode = 0u;
    std::uint32_t nodeCount = 0u;
    std::uint32_t firstTetrahedron = 0u;
    std::uint32_t tetrahedronCount = 0u;
    std::uint32_t firstSurface = 0u;
    std::uint32_t surfaceCount = 0u;
};

struct NumiHumanKneeSurface {
    std::string name;
    std::uint32_t regionIndex = 0u;
    std::uint32_t firstFace = 0u;
    std::uint32_t faceCount = 0u;
    bool isAllFaces = false;
};

struct NumiHumanKneeNodeSet {
    std::string name;
    std::uint32_t regionIndex = 0u;
    std::uint32_t firstMembership = 0u;
    std::uint32_t membershipCount = 0u;
    std::uint32_t anchorBodyIndex = 0xffffffffu;
};

struct NumiHumanKneeSurfacePair {
    std::string name;
    std::uint32_t masterSurface = 0u;
    std::uint32_t slaveSurface = 0u;
};

struct NumiHumanKneeNode {
    std::array<float, 3u> restWorld{};
    std::uint32_t anchorBodyIndex = 0xffffffffu;
    std::array<float, 3u> visualLocal{};
    std::array<float, 3u> anchorLocal{};
    bool rigidlyAttached = false;
};

struct NumiHumanKneePayload {
    std::uint32_t payloadAbi = 0u;
    NumiHumanKneeSide side = NumiHumanKneeSide::left;
    std::array<std::uint8_t, 32u> geometrySha256{};
    std::array<std::uint8_t, 32u> modelPropertiesSha256{};
    std::array<std::uint8_t, 32u> licenseSha256{};
    std::vector<NumiHumanKneeRegion> regions;
    std::vector<NumiHumanKneeSurface> surfaces;
    std::vector<NumiHumanKneeNodeSet> nodeSets;
    std::vector<NumiHumanKneeSurfacePair> surfacePairs;
    std::vector<NumiHumanKneeNode> nodes;
    std::vector<std::array<std::uint32_t, 4u>> tetrahedra;
    std::vector<std::array<std::uint32_t, 3u>> faces;
    std::vector<std::uint32_t> memberships;
};

struct NumiHumanKneeDiagnostics {
    NumiHumanKneeStatus status = NumiHumanKneeStatus::success;
    std::uint32_t failingIndex = 0xffffffffu;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == NumiHumanKneeStatus::success;
    }
};

inline constexpr std::uint32_t NUMI_HUMAN_KNEE_INVALID_INDEX = 0xffffffffu;
inline constexpr std::uint32_t NUMI_HUMAN_KNEE_FEMUR_BODY = 145u;
inline constexpr std::uint32_t NUMI_HUMAN_KNEE_TIBIA_BODY = 150u;
inline constexpr std::uint32_t NUMI_HUMAN_KNEE_PATELLA_BODY = 156u;
inline constexpr std::uint32_t NUMI_HUMAN_KNEE_RIGHT_FEMUR_BODY = 131u;
inline constexpr std::uint32_t NUMI_HUMAN_KNEE_RIGHT_TIBIA_BODY = 136u;
inline constexpr std::uint32_t NUMI_HUMAN_KNEE_RIGHT_PATELLA_BODY = 142u;

[[nodiscard]] NumiHumanKneeDiagnostics decodeNumiHumanKneePayload(
    std::span<const std::byte> bytes,
    NumiHumanKneePayload& payload
);

[[nodiscard]] const char* numiHumanKneeStatusName(
    NumiHumanKneeStatus status
) noexcept;

} // namespace metalrobo
