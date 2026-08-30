#include "metalrobo/NumiHumanCartilage.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace metalrobo {
namespace {

#pragma pack(push, 1)
struct HeaderDisk {
    std::array<char, 8u> magic;
    std::uint32_t payloadAbi;
    std::uint32_t regionCount;
    std::uint32_t nodeCount;
    std::uint32_t tetrahedronCount;
    std::uint32_t attachmentNodeCount;
    std::uint32_t reserved0;
    float densityKgM3;
    float attachmentDistanceMeters;
    std::array<std::uint8_t, 32u> sourceSha256;
};

struct RegionDisk {
    std::array<char, 8u> memberId;
    std::uint32_t side;
    std::uint32_t ribLevel;
    std::uint32_t firstNode;
    std::uint32_t nodeCount;
    std::uint32_t firstTetrahedron;
    std::uint32_t tetrahedronCount;
    std::uint32_t sternalAttachmentNodeCount;
    std::uint32_t ribAttachmentNodeCount;
    std::uint32_t sourceVertexCount;
    std::uint32_t sourceTriangleCount;
    float exactVolume;
    float voxelVolume;
    float voxelSpacing;
    float relativeVolumeError;
};

struct NodeDisk {
    std::array<float, 3u> restPosition;
    float compiledMassKg;
    std::uint32_t flags;
    std::uint32_t regionIndex;
    std::uint32_t reserved0;
};

struct TetrahedronDisk {
    std::array<std::uint32_t, 4u> node;
    std::uint32_t regionIndex;
};
#pragma pack(pop)

static_assert(sizeof(HeaderDisk) == 72u);
static_assert(sizeof(RegionDisk) == 64u);
static_assert(sizeof(NodeDisk) == 28u);
static_assert(sizeof(TetrahedronDisk) == 20u);

template <typename Value>
bool take(const std::span<const std::byte> bytes,
          std::size_t& offset,
          Value& value) {
    if (offset > bytes.size() || sizeof(Value) > bytes.size() - offset)
        return false;
    std::memcpy(&value, bytes.data() + offset, sizeof(Value));
    offset += sizeof(Value);
    return true;
}

double signedVolume(
    const NumiHumanCostalCartilagePayload& payload,
    const TetrahedronDisk& tetrahedron
) {
    const auto& a = payload.nodes[tetrahedron.node[0u]].restPosition;
    const auto& b = payload.nodes[tetrahedron.node[1u]].restPosition;
    const auto& c = payload.nodes[tetrahedron.node[2u]].restPosition;
    const auto& d = payload.nodes[tetrahedron.node[3u]].restPosition;
    const std::array<double, 3u> ab{
        b[0u] - a[0u], b[1u] - a[1u], b[2u] - a[2u]};
    const std::array<double, 3u> ac{
        c[0u] - a[0u], c[1u] - a[1u], c[2u] - a[2u]};
    const std::array<double, 3u> ad{
        d[0u] - a[0u], d[1u] - a[1u], d[2u] - a[2u]};
    return (
        ab[0u] * (ac[1u] * ad[2u] - ac[2u] * ad[1u]) -
        ab[1u] * (ac[0u] * ad[2u] - ac[2u] * ad[0u]) +
        ab[2u] * (ac[0u] * ad[1u] - ac[1u] * ad[0u])
    ) / 6.0;
}

NumiHumanCartilageDiagnostics fail(
    const NumiHumanCartilageStatus status,
    const std::uint32_t index = 0xffffffffu
) {
    return {.status = status, .failingIndex = index};
}

} // namespace

NumiHumanCartilageDiagnostics decodeNumiHumanCostalCartilagePayload(
    const std::span<const std::byte> bytes,
    const std::span<const std::uint8_t> expectedSourceSha256,
    NumiHumanCostalCartilagePayload& payload
) {
    payload = {};
    std::size_t offset = 0u;
    HeaderDisk header{};
    if (!take(bytes, offset, header))
        return fail(NumiHumanCartilageStatus::truncatedPayload);
    constexpr std::array<char, 8u> magic{
        'N', 'H', 'C', 'A', 'R', 'T', '1', '\0'};
    if (header.magic != magic || header.payloadAbi != 1u ||
        header.regionCount != 14u || header.nodeCount == 0u ||
        header.nodeCount > 100'000u || header.tetrahedronCount == 0u ||
        header.tetrahedronCount > 600'000u ||
        header.attachmentNodeCount == 0u ||
        header.attachmentNodeCount > header.nodeCount ||
        header.reserved0 != 0u || !std::isfinite(header.densityKgM3) ||
        header.densityKgM3 < 900.0f || header.densityKgM3 > 1300.0f ||
        !std::isfinite(header.attachmentDistanceMeters) ||
        header.attachmentDistanceMeters < 0.002f ||
        header.attachmentDistanceMeters > 0.006f)
        return fail(NumiHumanCartilageStatus::invalidPayload);
    if (!expectedSourceSha256.empty() &&
        (expectedSourceSha256.size() != header.sourceSha256.size() ||
         !std::equal(expectedSourceSha256.begin(), expectedSourceSha256.end(),
                     header.sourceSha256.begin())))
        return fail(NumiHumanCartilageStatus::sourceMismatch);

    payload.payloadAbi = header.payloadAbi;
    payload.densityKgM3 = header.densityKgM3;
    payload.attachmentDistanceMeters = header.attachmentDistanceMeters;
    payload.sourceSha256 = header.sourceSha256;
    payload.regions.reserve(header.regionCount);
    constexpr std::array<std::array<char, 8u>, 14u> expectedMembers{{
        {{'F', 'J', '3', '2', '3', '9', '\0', '\0'}},
        {{'F', 'J', '3', '2', '4', '2', '\0', '\0'}},
        {{'F', 'J', '3', '2', '4', '5', '\0', '\0'}},
        {{'F', 'J', '3', '2', '4', '8', '\0', '\0'}},
        {{'F', 'J', '3', '2', '5', '1', '\0', '\0'}},
        {{'F', 'J', '3', '2', '5', '4', '\0', '\0'}},
        {{'F', 'J', '3', '2', '5', '5', '\0', '\0'}},
        {{'F', 'J', '3', '3', '3', '3', '\0', '\0'}},
        {{'F', 'J', '3', '3', '3', '5', '\0', '\0'}},
        {{'F', 'J', '3', '3', '3', '7', '\0', '\0'}},
        {{'F', 'J', '3', '3', '3', '9', '\0', '\0'}},
        {{'F', 'J', '3', '3', '4', '1', '\0', '\0'}},
        {{'F', 'J', '3', '3', '4', '3', '\0', '\0'}},
        {{'F', 'J', '3', '3', '4', '5', '\0', '\0'}},
    }};
    std::uint32_t expectedFirstNode = 0u;
    std::uint32_t expectedFirstTetrahedron = 0u;
    for (std::uint32_t index = 0u; index < header.regionCount; ++index) {
        RegionDisk disk{};
        if (!take(bytes, offset, disk))
            return fail(NumiHumanCartilageStatus::truncatedPayload, index);
        const std::uint32_t expectedSide = index < 7u ? 0u : 1u;
        const std::uint32_t expectedLevel = index % 7u + 1u;
        if (disk.memberId != expectedMembers[index] ||
            disk.side != expectedSide || disk.ribLevel != expectedLevel ||
            disk.firstNode != expectedFirstNode || disk.nodeCount < 16u ||
            disk.nodeCount > header.nodeCount - disk.firstNode ||
            disk.firstTetrahedron != expectedFirstTetrahedron ||
            disk.tetrahedronCount < 6u ||
            disk.tetrahedronCount >
                header.tetrahedronCount - disk.firstTetrahedron ||
            disk.sternalAttachmentNodeCount < 16u ||
            disk.ribAttachmentNodeCount < 16u ||
            disk.sternalAttachmentNodeCount + disk.ribAttachmentNodeCount >
                disk.nodeCount ||
            disk.sourceVertexCount == 0u || disk.sourceTriangleCount == 0u ||
            !std::isfinite(disk.exactVolume) || disk.exactVolume <= 0.0f ||
            !std::isfinite(disk.voxelVolume) || disk.voxelVolume <= 0.0f ||
            !std::isfinite(disk.voxelSpacing) || disk.voxelSpacing < 0.001f ||
            disk.voxelSpacing > 0.004f ||
            !std::isfinite(disk.relativeVolumeError) ||
            disk.relativeVolumeError < 0.0f || disk.relativeVolumeError > 0.05f)
            return fail(NumiHumanCartilageStatus::incompleteCoverage, index);
        payload.regions.push_back({
            .memberId = disk.memberId,
            .side = disk.side,
            .ribLevel = disk.ribLevel,
            .firstNode = disk.firstNode,
            .nodeCount = disk.nodeCount,
            .firstTetrahedron = disk.firstTetrahedron,
            .tetrahedronCount = disk.tetrahedronCount,
            .sternalAttachmentNodeCount = disk.sternalAttachmentNodeCount,
            .ribAttachmentNodeCount = disk.ribAttachmentNodeCount,
            .sourceVertexCount = disk.sourceVertexCount,
            .sourceTriangleCount = disk.sourceTriangleCount,
            .exactVolume = disk.exactVolume,
            .voxelVolume = disk.voxelVolume,
            .voxelSpacing = disk.voxelSpacing,
            .relativeVolumeError = disk.relativeVolumeError,
        });
        expectedFirstNode += disk.nodeCount;
        expectedFirstTetrahedron += disk.tetrahedronCount;
    }
    if (expectedFirstNode != header.nodeCount ||
        expectedFirstTetrahedron != header.tetrahedronCount)
        return fail(NumiHumanCartilageStatus::incompleteCoverage);

    payload.nodes.reserve(header.nodeCount);
    std::uint32_t attachmentNodeCount = 0u;
    std::vector<std::array<std::uint32_t, 2u>> attachmentCounts(
        header.regionCount, {0u, 0u});
    for (std::uint32_t index = 0u; index < header.nodeCount; ++index) {
        NodeDisk disk{};
        if (!take(bytes, offset, disk))
            return fail(NumiHumanCartilageStatus::truncatedPayload, index);
        if (!std::all_of(disk.restPosition.begin(), disk.restPosition.end(),
                         [](const float value) { return std::isfinite(value); }) ||
            !std::isfinite(disk.compiledMassKg) || disk.compiledMassKg <= 0.0f ||
            (disk.flags & ~3u) != 0u || disk.flags == 3u ||
            disk.regionIndex >= header.regionCount || disk.reserved0 != 0u)
            return fail(NumiHumanCartilageStatus::invalidPayload, index);
        if (disk.flags != 0u) ++attachmentNodeCount;
        if ((disk.flags & NUMI_HUMAN_COSTAL_CARTILAGE_STERNAL_ATTACHMENT) != 0u)
            ++attachmentCounts[disk.regionIndex][0u];
        if ((disk.flags & NUMI_HUMAN_COSTAL_CARTILAGE_RIB_ATTACHMENT) != 0u)
            ++attachmentCounts[disk.regionIndex][1u];
        payload.nodes.push_back({
            .restPosition = disk.restPosition,
            .compiledMassKg = disk.compiledMassKg,
            .flags = disk.flags,
            .regionIndex = disk.regionIndex,
        });
    }
    if (attachmentNodeCount != header.attachmentNodeCount)
        return fail(NumiHumanCartilageStatus::incompleteCoverage);
    for (std::uint32_t region = 0u; region < header.regionCount; ++region) {
        if (attachmentCounts[region][0u] !=
                payload.regions[region].sternalAttachmentNodeCount ||
            attachmentCounts[region][1u] !=
                payload.regions[region].ribAttachmentNodeCount)
            return fail(NumiHumanCartilageStatus::incompleteCoverage, region);
    }

    payload.tetrahedra.reserve(header.tetrahedronCount);
    for (std::uint32_t index = 0u; index < header.tetrahedronCount; ++index) {
        TetrahedronDisk disk{};
        if (!take(bytes, offset, disk))
            return fail(NumiHumanCartilageStatus::truncatedPayload, index);
        if (disk.regionIndex >= payload.regions.size() ||
            !std::all_of(disk.node.begin(), disk.node.end(),
                [&](const std::uint32_t node) {
                    return node < payload.nodes.size() &&
                        payload.nodes[node].regionIndex == disk.regionIndex;
                }))
            return fail(NumiHumanCartilageStatus::invalidPayload, index);
        if (!(signedVolume(payload, disk) > 0.0))
            return fail(NumiHumanCartilageStatus::nonpositiveVolume, index);
        payload.tetrahedra.push_back({
            .node = disk.node,
            .regionIndex = disk.regionIndex,
        });
    }
    if (offset != bytes.size())
        return fail(NumiHumanCartilageStatus::invalidPayload);
    return {};
}

const char* numiHumanCartilageStatusName(
    const NumiHumanCartilageStatus status
) noexcept {
    switch (status) {
        case NumiHumanCartilageStatus::success: return "success";
        case NumiHumanCartilageStatus::truncatedPayload: return "truncated_payload";
        case NumiHumanCartilageStatus::invalidPayload: return "invalid_payload";
        case NumiHumanCartilageStatus::sourceMismatch: return "source_mismatch";
        case NumiHumanCartilageStatus::incompleteCoverage: return "incomplete_coverage";
        case NumiHumanCartilageStatus::nonpositiveVolume: return "nonpositive_volume";
    }
    return "unknown";
}

} // namespace metalrobo
