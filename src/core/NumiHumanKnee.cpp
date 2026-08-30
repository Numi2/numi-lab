#include "metalrobo/NumiHumanKnee.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <string_view>

namespace metalrobo {
namespace {

#pragma pack(push, 1)
struct HeaderDisk {
    std::array<char, 8u> magic;
    std::uint32_t payloadAbi;
    std::uint32_t headerBytes;
    std::uint32_t regionCount;
    std::uint32_t nodeCount;
    std::uint32_t tetrahedronCount;
    std::uint32_t surfaceCount;
    std::uint32_t faceCount;
    std::uint32_t nodeSetCount;
    std::uint32_t membershipCount;
    std::uint32_t surfacePairCount;
    std::uint32_t reserved0;
    std::uint32_t reserved1;
    std::array<std::uint8_t, 96u> sourceSha256;
};

struct RegionDisk {
    std::array<char, 16u> name;
    std::uint32_t kind;
    std::uint32_t visualBodyIndex;
    std::uint32_t firstNode;
    std::uint32_t nodeCount;
    std::uint32_t firstTetrahedron;
    std::uint32_t tetrahedronCount;
    std::uint32_t firstSurface;
    std::uint32_t surfaceCount;
};

struct SurfaceDisk {
    std::array<char, 48u> name;
    std::uint32_t regionIndex;
    std::uint32_t firstFace;
    std::uint32_t faceCount;
    std::uint32_t flags;
    std::uint32_t reserved0;
};

struct NodeSetDisk {
    std::array<char, 48u> name;
    std::uint32_t regionIndex;
    std::uint32_t firstMembership;
    std::uint32_t membershipCount;
    std::uint32_t anchorBodyIndex;
    std::uint32_t reserved0;
};

struct SurfacePairDisk {
    std::array<char, 48u> name;
    std::uint32_t masterSurface;
    std::uint32_t slaveSurface;
};

struct NodeDisk {
    std::array<float, 3u> restWorld;
    std::uint32_t anchorBodyIndex;
    std::array<float, 3u> visualLocal;
    std::uint32_t reserved0;
    std::array<float, 3u> anchorLocal;
    std::uint32_t flags;
};

struct TetrahedronDisk { std::array<std::uint32_t, 4u> node; };
struct FaceDisk { std::array<std::uint32_t, 3u> node; };
struct MembershipDisk { std::uint32_t node; };
#pragma pack(pop)

static_assert(sizeof(HeaderDisk) == 152u);
static_assert(sizeof(RegionDisk) == 48u);
static_assert(sizeof(SurfaceDisk) == 68u);
static_assert(sizeof(NodeSetDisk) == 68u);
static_assert(sizeof(SurfacePairDisk) == 56u);
static_assert(sizeof(NodeDisk) == 48u);
static_assert(sizeof(TetrahedronDisk) == 16u);
static_assert(sizeof(FaceDisk) == 12u);
static_assert(sizeof(MembershipDisk) == 4u);

constexpr std::array<std::string_view, 16u> kRegionNames{{
    "QAT", "TBC-L", "PCL", "PTC", "PTB", "ACL", "FBB", "MCL",
    "PTL", "MNS-L", "MNS-M", "LCL", "TBC-M", "TBB", "FMB", "FMC",
}};
constexpr std::array<std::uint32_t, 16u> kRegionKinds{{
    5u, 2u, 4u, 2u, 1u, 4u, 1u, 4u, 5u, 3u, 3u, 4u, 2u, 1u, 1u, 2u,
}};
constexpr std::array<std::uint32_t, 16u> kLeftVisualBodies{{
    156u, 150u, 145u, 156u, 156u, 145u, 150u, 145u,
    156u, 150u, 150u, 145u, 150u, 150u, 145u, 145u,
}};
constexpr std::array<std::uint32_t, 16u> kRightVisualBodies{{
    142u, 136u, 131u, 142u, 142u, 131u, 136u, 131u,
    142u, 136u, 136u, 131u, 136u, 136u, 131u, 131u,
}};
constexpr std::array<std::uint32_t, 16u> kNodeCounts{{
    14963u, 40669u, 3714u, 26121u, 8642u, 15792u, 3794u, 15693u,
    9280u, 10901u, 11706u, 2960u, 18060u, 20900u, 20171u, 24870u,
}};
constexpr std::array<std::uint32_t, 16u> kTetrahedronCounts{{
    69410u, 200079u, 14379u, 121105u, 0u, 72552u, 0u, 62712u,
    35616u, 44953u, 51009u, 9773u, 75627u, 0u, 0u, 87072u,
}};

constexpr std::array<std::uint8_t, 32u> kGeometrySha256{{
    0x36,0x42,0xbd,0x36,0x8b,0xbc,0x86,0x75,0x69,0xf1,0x81,0xfa,0x76,0x12,0x9f,0x74,
    0x64,0x70,0xe8,0x07,0xe3,0x97,0x75,0x85,0xd3,0x80,0x3f,0x09,0x2d,0xd1,0x12,0x62,
}};
constexpr std::array<std::uint8_t, 32u> kModelPropertiesSha256{{
    0x0a,0xc4,0x46,0xce,0x09,0x8b,0x9a,0x09,0x50,0x59,0x92,0xeb,0x4f,0x44,0x19,0xc7,
    0xb9,0x44,0xcd,0x57,0xa4,0xaf,0xbf,0x63,0x92,0xb1,0x37,0xf4,0x80,0x66,0x03,0xc1,
}};
constexpr std::array<std::uint8_t, 32u> kLicenseSha256{{
    0xd7,0x29,0x18,0x83,0x8b,0x4a,0xdf,0x30,0x97,0x9d,0x2a,0x26,0xc2,0x38,0x37,0xf0,
    0xca,0x05,0x18,0x5b,0xa7,0x99,0xa3,0xa4,0xfe,0x1f,0xe1,0xb4,0xc0,0x5b,0x20,0xb8,
}};

template <typename Value>
bool take(const std::span<const std::byte> bytes, std::size_t& offset, Value& value) {
    if (offset > bytes.size() || sizeof(Value) > bytes.size() - offset) return false;
    std::memcpy(&value, bytes.data() + offset, sizeof(Value));
    offset += sizeof(Value);
    return true;
}

template <std::size_t Size>
bool decodeName(const std::array<char, Size>& disk, std::string& result) {
    const auto terminator = std::find(disk.begin(), disk.end(), '\0');
    if (terminator == disk.begin() || terminator == disk.end() ||
        std::any_of(terminator + 1, disk.end(), [](const char value) { return value != '\0'; }))
        return false;
    result.assign(disk.begin(), terminator);
    return std::all_of(result.begin(), result.end(), [](const unsigned char value) {
        return value >= 0x20u && value <= 0x7eu;
    });
}

bool finite3(const std::array<float, 3u>& value) {
    return std::all_of(value.begin(), value.end(), [](const float component) {
        return std::isfinite(component);
    });
}

bool isKneeBody(const std::uint32_t body, const NumiHumanKneeSide side) {
    return side == NumiHumanKneeSide::left
        ? body == NUMI_HUMAN_KNEE_FEMUR_BODY ||
            body == NUMI_HUMAN_KNEE_TIBIA_BODY ||
            body == NUMI_HUMAN_KNEE_PATELLA_BODY
        : body == NUMI_HUMAN_KNEE_RIGHT_FEMUR_BODY ||
            body == NUMI_HUMAN_KNEE_RIGHT_TIBIA_BODY ||
            body == NUMI_HUMAN_KNEE_RIGHT_PATELLA_BODY;
}

NumiHumanKneeDiagnostics fail(
    const NumiHumanKneeStatus status,
    const std::uint32_t index = NUMI_HUMAN_KNEE_INVALID_INDEX
) { return {.status = status, .failingIndex = index}; }

std::uint32_t regionForNode(
    const std::span<const NumiHumanKneeRegion> regions,
    const std::uint32_t node
) {
    const auto found = std::upper_bound(
        regions.begin(), regions.end(), node,
        [](const std::uint32_t value, const NumiHumanKneeRegion& region) {
            return value < region.firstNode;
        }
    );
    if (found == regions.begin()) return NUMI_HUMAN_KNEE_INVALID_INDEX;
    const auto& region = *(found - 1);
    return node < region.firstNode + region.nodeCount
        ? static_cast<std::uint32_t>((found - 1) - regions.begin())
        : NUMI_HUMAN_KNEE_INVALID_INDEX;
}

} // namespace

NumiHumanKneeDiagnostics decodeNumiHumanKneePayload(
    const std::span<const std::byte> bytes,
    NumiHumanKneePayload& payload
) {
    payload = {};
    std::size_t offset = 0u;
    HeaderDisk header{};
    if (!take(bytes, offset, header)) return fail(NumiHumanKneeStatus::truncatedPayload);
    constexpr std::array<char, 8u> magic{{'N','H','K','N','E','E','1','\0'}};
    if (header.magic != magic || header.payloadAbi != 1u ||
        header.headerBytes != sizeof(HeaderDisk) || header.regionCount != 16u ||
        header.nodeCount != 248236u || header.tetrahedronCount != 844287u ||
        header.surfaceCount != 88u || header.faceCount != 729068u ||
        header.nodeSetCount != 42u || header.membershipCount != 43260u ||
        header.surfacePairCount != 19u || header.reserved0 > 1u ||
        header.reserved1 != 0u)
        return fail(NumiHumanKneeStatus::invalidPayload);
    if (!std::equal(kGeometrySha256.begin(), kGeometrySha256.end(), header.sourceSha256.begin()) ||
        !std::equal(kModelPropertiesSha256.begin(), kModelPropertiesSha256.end(), header.sourceSha256.begin() + 32) ||
        !std::equal(kLicenseSha256.begin(), kLicenseSha256.end(), header.sourceSha256.begin() + 64))
        return fail(NumiHumanKneeStatus::sourceMismatch);

    payload.payloadAbi = header.payloadAbi;
    payload.side = static_cast<NumiHumanKneeSide>(header.reserved0);
    const auto& visualBodies = payload.side == NumiHumanKneeSide::left
        ? kLeftVisualBodies : kRightVisualBodies;
    std::copy_n(header.sourceSha256.begin(), 32u, payload.geometrySha256.begin());
    std::copy_n(header.sourceSha256.begin() + 32, 32u, payload.modelPropertiesSha256.begin());
    std::copy_n(header.sourceSha256.begin() + 64, 32u, payload.licenseSha256.begin());
    payload.regions.reserve(header.regionCount);
    std::uint32_t nextNode = 0u, nextTet = 0u, nextSurface = 0u;
    for (std::uint32_t index = 0u; index < header.regionCount; ++index) {
        RegionDisk disk{};
        if (!take(bytes, offset, disk)) return fail(NumiHumanKneeStatus::truncatedPayload, index);
        std::string name;
        if (!decodeName(disk.name, name) || name != kRegionNames[index] ||
            disk.kind != kRegionKinds[index] || disk.visualBodyIndex != visualBodies[index] ||
            disk.firstNode != nextNode || disk.nodeCount != kNodeCounts[index] ||
            disk.firstTetrahedron != nextTet || disk.tetrahedronCount != kTetrahedronCounts[index] ||
            disk.firstSurface != nextSurface || disk.surfaceCount == 0u ||
            disk.surfaceCount > header.surfaceCount - disk.firstSurface)
            return fail(NumiHumanKneeStatus::incompleteCoverage, index);
        payload.regions.push_back({
            .name = std::move(name),
            .kind = static_cast<NumiHumanKneeRegionKind>(disk.kind),
            .visualBodyIndex = disk.visualBodyIndex,
            .firstNode = disk.firstNode, .nodeCount = disk.nodeCount,
            .firstTetrahedron = disk.firstTetrahedron,
            .tetrahedronCount = disk.tetrahedronCount,
            .firstSurface = disk.firstSurface, .surfaceCount = disk.surfaceCount,
        });
        nextNode += disk.nodeCount;
        nextTet += disk.tetrahedronCount;
        nextSurface += disk.surfaceCount;
    }
    if (nextNode != header.nodeCount || nextTet != header.tetrahedronCount ||
        nextSurface != header.surfaceCount)
        return fail(NumiHumanKneeStatus::incompleteCoverage);

    payload.surfaces.reserve(header.surfaceCount);
    std::uint32_t nextFace = 0u;
    std::array<std::uint32_t, 16u> allFaceCounts{};
    for (std::uint32_t index = 0u; index < header.surfaceCount; ++index) {
        SurfaceDisk disk{};
        if (!take(bytes, offset, disk)) return fail(NumiHumanKneeStatus::truncatedPayload, index);
        std::string name;
        if (!decodeName(disk.name, name) || disk.regionIndex >= payload.regions.size() ||
            disk.firstFace != nextFace || disk.faceCount == 0u ||
            disk.faceCount > header.faceCount - disk.firstFace ||
            (disk.flags & ~1u) != 0u || disk.reserved0 != 0u)
            return fail(NumiHumanKneeStatus::invalidTopology, index);
        const auto& region = payload.regions[disk.regionIndex];
        if (index < region.firstSurface || index >= region.firstSurface + region.surfaceCount ||
            !name.starts_with(region.name + "_"))
            return fail(NumiHumanKneeStatus::incompleteCoverage, index);
        const bool isAllFaces = (disk.flags & 1u) != 0u;
        if (isAllFaces && (!name.ends_with("_All_Faces") || ++allFaceCounts[disk.regionIndex] != 1u))
            return fail(NumiHumanKneeStatus::invalidTopology, index);
        payload.surfaces.push_back({
            .name = std::move(name), .regionIndex = disk.regionIndex,
            .firstFace = disk.firstFace, .faceCount = disk.faceCount,
            .isAllFaces = isAllFaces,
        });
        nextFace += disk.faceCount;
    }
    if (nextFace != header.faceCount ||
        std::any_of(allFaceCounts.begin(), allFaceCounts.end(), [](const auto count) { return count != 1u; }))
        return fail(NumiHumanKneeStatus::incompleteCoverage);

    payload.nodeSets.reserve(header.nodeSetCount);
    std::uint32_t nextMembership = 0u;
    for (std::uint32_t index = 0u; index < header.nodeSetCount; ++index) {
        NodeSetDisk disk{};
        if (!take(bytes, offset, disk)) return fail(NumiHumanKneeStatus::truncatedPayload, index);
        std::string name;
        if (!decodeName(disk.name, name) || disk.regionIndex >= payload.regions.size() ||
            disk.firstMembership != nextMembership || disk.membershipCount == 0u ||
            disk.membershipCount > header.membershipCount - disk.firstMembership ||
            (disk.anchorBodyIndex != NUMI_HUMAN_KNEE_INVALID_INDEX &&
             !isKneeBody(disk.anchorBodyIndex, payload.side)) ||
            disk.reserved0 != 0u || !name.starts_with(payload.regions[disk.regionIndex].name + "_"))
            return fail(NumiHumanKneeStatus::invalidTopology, index);
        payload.nodeSets.push_back({
            .name = std::move(name), .regionIndex = disk.regionIndex,
            .firstMembership = disk.firstMembership,
            .membershipCount = disk.membershipCount,
            .anchorBodyIndex = disk.anchorBodyIndex,
        });
        nextMembership += disk.membershipCount;
    }
    if (nextMembership != header.membershipCount)
        return fail(NumiHumanKneeStatus::incompleteCoverage);

    payload.surfacePairs.reserve(header.surfacePairCount);
    for (std::uint32_t index = 0u; index < header.surfacePairCount; ++index) {
        SurfacePairDisk disk{};
        if (!take(bytes, offset, disk)) return fail(NumiHumanKneeStatus::truncatedPayload, index);
        std::string name;
        if (!decodeName(disk.name, name) || disk.masterSurface >= header.surfaceCount ||
            disk.slaveSurface >= header.surfaceCount || disk.masterSurface == disk.slaveSurface)
            return fail(NumiHumanKneeStatus::invalidTopology, index);
        payload.surfacePairs.push_back({
            .name = std::move(name), .masterSurface = disk.masterSurface,
            .slaveSurface = disk.slaveSurface,
        });
    }

    payload.nodes.reserve(header.nodeCount);
    for (std::uint32_t index = 0u; index < header.nodeCount; ++index) {
        NodeDisk disk{};
        if (!take(bytes, offset, disk)) return fail(NumiHumanKneeStatus::truncatedPayload, index);
        const bool attached = (disk.flags & 1u) != 0u;
        if (!finite3(disk.restWorld) || !finite3(disk.visualLocal) || !finite3(disk.anchorLocal) ||
            disk.reserved0 != 0u || (disk.flags & ~1u) != 0u ||
            (attached != (disk.anchorBodyIndex != NUMI_HUMAN_KNEE_INVALID_INDEX)) ||
            (attached && !isKneeBody(disk.anchorBodyIndex, payload.side)))
            return fail(NumiHumanKneeStatus::invalidPayload, index);
        payload.nodes.push_back({
            .restWorld = disk.restWorld, .anchorBodyIndex = disk.anchorBodyIndex,
            .visualLocal = disk.visualLocal, .anchorLocal = disk.anchorLocal,
            .rigidlyAttached = attached,
        });
    }

    payload.tetrahedra.reserve(header.tetrahedronCount);
    for (std::uint32_t regionIndex = 0u; regionIndex < payload.regions.size(); ++regionIndex) {
        const auto& region = payload.regions[regionIndex];
        for (std::uint32_t local = 0u; local < region.tetrahedronCount; ++local) {
            TetrahedronDisk disk{};
            const std::uint32_t index = region.firstTetrahedron + local;
            if (!take(bytes, offset, disk)) return fail(NumiHumanKneeStatus::truncatedPayload, index);
            if (std::any_of(disk.node.begin(), disk.node.end(), [&](const auto node) {
                    return regionForNode(payload.regions, node) != regionIndex;
                }))
                return fail(NumiHumanKneeStatus::invalidTopology, index);
            auto unique = disk.node;
            std::sort(unique.begin(), unique.end());
            if (std::adjacent_find(unique.begin(), unique.end()) != unique.end())
                return fail(NumiHumanKneeStatus::invalidTopology, index);
            payload.tetrahedra.push_back(disk.node);
        }
    }

    payload.faces.reserve(header.faceCount);
    for (std::uint32_t surfaceIndex = 0u; surfaceIndex < payload.surfaces.size(); ++surfaceIndex) {
        const auto& surface = payload.surfaces[surfaceIndex];
        for (std::uint32_t local = 0u; local < surface.faceCount; ++local) {
            FaceDisk disk{};
            const std::uint32_t index = surface.firstFace + local;
            if (!take(bytes, offset, disk)) return fail(NumiHumanKneeStatus::truncatedPayload, index);
            if (std::any_of(disk.node.begin(), disk.node.end(), [&](const auto node) {
                    return regionForNode(payload.regions, node) != surface.regionIndex;
                }))
                return fail(NumiHumanKneeStatus::invalidTopology, index);
            auto unique = disk.node;
            std::sort(unique.begin(), unique.end());
            if (std::adjacent_find(unique.begin(), unique.end()) != unique.end())
                return fail(NumiHumanKneeStatus::invalidTopology, index);
            payload.faces.push_back(disk.node);
        }
    }

    payload.memberships.reserve(header.membershipCount);
    std::vector<std::uint32_t> attachmentOwner(
        header.nodeCount, NUMI_HUMAN_KNEE_INVALID_INDEX
    );
    for (std::uint32_t setIndex = 0u; setIndex < payload.nodeSets.size(); ++setIndex) {
        const auto& set = payload.nodeSets[setIndex];
        for (std::uint32_t local = 0u; local < set.membershipCount; ++local) {
            MembershipDisk disk{};
            const std::uint32_t index = set.firstMembership + local;
            if (!take(bytes, offset, disk)) return fail(NumiHumanKneeStatus::truncatedPayload, index);
            if (regionForNode(payload.regions, disk.node) != set.regionIndex)
                return fail(NumiHumanKneeStatus::invalidTopology, index);
            if (set.anchorBodyIndex != NUMI_HUMAN_KNEE_INVALID_INDEX) {
                auto& owner = attachmentOwner[disk.node];
                if (owner != NUMI_HUMAN_KNEE_INVALID_INDEX &&
                    owner != set.anchorBodyIndex)
                    return fail(NumiHumanKneeStatus::invalidTopology, index);
                owner = set.anchorBodyIndex;
            }
            payload.memberships.push_back(disk.node);
        }
    }
    std::array<std::uint32_t, 3u> attachmentCounts{};
    const auto expectedBodies = payload.side == NumiHumanKneeSide::left
        ? std::array<std::uint32_t, 3u>{
            NUMI_HUMAN_KNEE_FEMUR_BODY, NUMI_HUMAN_KNEE_TIBIA_BODY,
            NUMI_HUMAN_KNEE_PATELLA_BODY}
        : std::array<std::uint32_t, 3u>{
            NUMI_HUMAN_KNEE_RIGHT_FEMUR_BODY, NUMI_HUMAN_KNEE_RIGHT_TIBIA_BODY,
            NUMI_HUMAN_KNEE_RIGHT_PATELLA_BODY};
    for (std::uint32_t node = 0u; node < payload.nodes.size(); ++node) {
        const auto expected = attachmentOwner[node];
        if (payload.nodes[node].anchorBodyIndex != expected ||
            payload.nodes[node].rigidlyAttached !=
                (expected != NUMI_HUMAN_KNEE_INVALID_INDEX))
            return fail(NumiHumanKneeStatus::invalidTopology, node);
        for (std::uint32_t body = 0u; body < expectedBodies.size(); ++body) {
            if (expected == expectedBodies[body]) ++attachmentCounts[body];
        }
    }
    if (attachmentCounts != std::array<std::uint32_t, 3u>{11350u, 12161u, 5394u})
        return fail(NumiHumanKneeStatus::incompleteCoverage);
    if (offset != bytes.size()) return fail(NumiHumanKneeStatus::invalidPayload);
    return {};
}

const char* numiHumanKneeStatusName(const NumiHumanKneeStatus status) noexcept {
    switch (status) {
        case NumiHumanKneeStatus::success: return "success";
        case NumiHumanKneeStatus::truncatedPayload: return "truncated_payload";
        case NumiHumanKneeStatus::invalidPayload: return "invalid_payload";
        case NumiHumanKneeStatus::sourceMismatch: return "source_mismatch";
        case NumiHumanKneeStatus::incompleteCoverage: return "incomplete_coverage";
        case NumiHumanKneeStatus::invalidTopology: return "invalid_topology";
    }
    return "unknown";
}

} // namespace metalrobo
