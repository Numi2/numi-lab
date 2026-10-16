#pragma once

#include "numi/matter/shared.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace numi::matter::cardboard::cohesive_tiles {

inline constexpr std::uint64_t kPrivatePageByteCap = 8ull * 1024ull * 1024ull;

struct Page {
    std::uint32_t firstFace = 0u;
    std::uint32_t faceCount = 0u;
    std::uint32_t dispatchThreads = 0u;
};

struct Layout {
    std::uint32_t environmentCount = 0u;
    std::uint32_t totalFaceCount = 0u;
    std::uint32_t pageCapacity = 0u;
    std::uint64_t pageBytes = 0u;
    std::uint64_t legacyPrivateBytes = 0u;
    std::vector<Page> pages;
};

inline bool makeLayout(const std::uint32_t environmentCount,
                       const std::uint32_t faceCount,
                       const std::uint32_t requestedPageCapacity,
                       const std::uint64_t deviceMaxBufferLength,
                       Layout& output) {
    output = {};
    if (environmentCount == 0u || environmentCount > 4u || faceCount == 0u)
        return false;
    // Runtime's canonical incidence word is faceIndex * 6 + localSlot.
    // Keep this helper fail-closed at the largest encodable face domain.
    if (faceCount > std::numeric_limits<std::uint32_t>::max() / 6u)
        return false;
    constexpr std::uint64_t forceBytes = sizeof(NMCohesiveFaceVectorGPU);
    constexpr std::uint64_t faceBytes = sizeof(NMCohesiveFaceGPU);
    static_assert(forceBytes == 96u, "cohesive force ABI size changed");
    static_assert(faceBytes == 384u, "cohesive face ABI size changed");
    if (forceBytes == 0u) return false;
    const std::uint64_t bytesPerPageFace = environmentCount * forceBytes;
    const std::uint64_t maximumPageCapacity =
        kPrivatePageByteCap / bytesPerPageFace;
    if (maximumPageCapacity == 0u ||
        (requestedPageCapacity != 0u &&
         requestedPageCapacity > maximumPageCapacity))
        return false;
    const std::uint64_t pageCapacity = std::min<std::uint64_t>(
        faceCount, requestedPageCapacity == 0u
            ? maximumPageCapacity : requestedPageCapacity);
    const std::uint64_t pageElements = environmentCount * pageCapacity;
    if (pageCapacity == 0u || pageCapacity > UINT32_MAX ||
        pageElements > UINT32_MAX ||
        pageElements > std::numeric_limits<std::size_t>::max())
        return false;
    const std::uint64_t pageBytes = pageElements * forceBytes;
    if (pageBytes > kPrivatePageByteCap ||
        pageBytes > deviceMaxBufferLength)
        return false;
    const std::uint64_t pageCount = 1u +
        (static_cast<std::uint64_t>(faceCount) - 1u) / pageCapacity;
    if (pageCount > UINT32_MAX || pageCount >
            std::numeric_limits<std::size_t>::max())
        return false;
    if (static_cast<std::uint64_t>(environmentCount) * faceCount > UINT32_MAX)
        return false;
    if (static_cast<std::uint64_t>(environmentCount) * faceCount >
        std::numeric_limits<std::uint64_t>::max() /
            (faceBytes + forceBytes))
        return false;

    output.environmentCount = environmentCount;
    output.totalFaceCount = faceCount;
    output.pageCapacity = static_cast<std::uint32_t>(pageCapacity);
    output.pageBytes = pageBytes;
    output.legacyPrivateBytes =
        static_cast<std::uint64_t>(environmentCount) * faceCount *
            (faceBytes + forceBytes);
    output.pages.reserve(static_cast<std::size_t>(pageCount));
    std::uint64_t cursor = 0u;
    while (cursor < faceCount) {
        const std::uint64_t count = std::min<std::uint64_t>(
            pageCapacity, static_cast<std::uint64_t>(faceCount) - cursor);
        const std::uint64_t threads = environmentCount * count;
        if (threads > UINT32_MAX) {
            output = {};
            return false;
        }
        output.pages.push_back({static_cast<std::uint32_t>(cursor),
            static_cast<std::uint32_t>(count),
            static_cast<std::uint32_t>(threads)});
        cursor += count;
    }
    return cursor == faceCount && output.pages.size() == pageCount;
}

using Force = std::array<float, 3>;
using FaceForces = std::array<Force, 6>;
using IncidenceEntry = std::array<std::uint32_t, 2>; // face index, local slot

inline std::vector<Force> legacyGather(
        const std::vector<FaceForces>& faces,
        const std::vector<std::vector<IncidenceEntry>>& nodeIncidence) {
    std::vector<Force> result(nodeIncidence.size());
    for (std::size_t node = 0u; node < nodeIncidence.size(); ++node) {
        for (const IncidenceEntry entry : nodeIncidence[node]) {
            if (entry[0] >= faces.size() || entry[1] >= 6u)
                throw std::runtime_error("invalid cohesive incidence entry");
            for (unsigned axis = 0u; axis < 3u; ++axis)
                result[node][axis] += faces[entry[0]][entry[1]][axis];
        }
    }
    return result;
}

inline std::vector<Force> tiledGather(
        const std::vector<FaceForces>& faces,
        const std::vector<std::vector<IncidenceEntry>>& nodeIncidence,
        const Layout& layout) {
    if (faces.size() != layout.totalFaceCount ||
        layout.pages.empty() || layout.totalFaceCount == 0u)
        throw std::runtime_error("cohesive page layout does not cover the force array");
    std::vector<Force> result(nodeIncidence.size());
    // Runtime builds each node's face/slot list by walking faces and slots in
    // ascending order. The page outer loop is equivalent to the legacy loop
    // only under that invariant; reject a noncanonical reference stream
    // instead of silently claiming ordered FP32 parity for arbitrary input.
    for (const auto& entries : nodeIncidence) {
        std::uint32_t previous = 0u;
        bool first = true;
        for (const IncidenceEntry entry : entries) {
            if (entry[0] >= faces.size() || entry[1] >= 6u)
                throw std::runtime_error("invalid cohesive incidence entry");
            const std::uint64_t encoded =
                static_cast<std::uint64_t>(entry[0]) * 6u + entry[1];
            if (!first && encoded < previous)
                throw std::runtime_error(
                    "tiled cohesive gather requires canonical face/slot order");
            previous = static_cast<std::uint32_t>(encoded);
            first = false;
        }
    }
    for (const Page page : layout.pages) {
        const std::uint64_t pageEnd =
            static_cast<std::uint64_t>(page.firstFace) + page.faceCount;
        if (page.faceCount == 0u || pageEnd > layout.totalFaceCount)
            throw std::runtime_error("cohesive page exceeds its authored face range");
        for (std::size_t node = 0u; node < nodeIncidence.size(); ++node) {
            for (const IncidenceEntry entry : nodeIncidence[node]) {
                if (entry[0] >= faces.size() || entry[1] >= 6u)
                    throw std::runtime_error("invalid cohesive incidence entry");
                if (entry[0] < page.firstFace || entry[0] >= pageEnd) continue;
                for (unsigned axis = 0u; axis < 3u; ++axis)
                    result[node][axis] += faces[entry[0]][entry[1]][axis];
            }
        }
    }
    return result;
}

inline bool bitIdentical(const std::vector<Force>& left,
                         const std::vector<Force>& right) noexcept {
    return left.size() == right.size() &&
        (left.empty() || std::memcmp(left.data(), right.data(),
            left.size() * sizeof(Force)) == 0);
}

} // namespace numi::matter::cardboard::cohesive_tiles
