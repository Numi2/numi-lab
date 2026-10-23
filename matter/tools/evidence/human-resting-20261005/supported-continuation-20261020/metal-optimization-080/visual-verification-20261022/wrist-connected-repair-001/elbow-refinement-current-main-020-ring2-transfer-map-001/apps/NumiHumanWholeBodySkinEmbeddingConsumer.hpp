#pragma once

// Initialization/presentation adapter for coarse rows whose ancestry is either
// an exact source vertex or a source-triangle barycentric embedding.
// This retains source ancestry and does not enable a candidate or update physical state.
#include "NumiHumanSkinFEMIntegration.wholebody.hpp"
#include "NumiHumanAcceptedFEMSkinTransfer.hpp"

#include <array>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace metalrobo::human::coarse_tissue {

struct WholeBodySkinSourceConsumerRows {
    using Transfer = numi::human::AcceptedFEMSkinTransfer;
    std::vector<std::uint32_t> directSourceIDs;
    std::vector<Transfer::Face> sourceFaces;
    std::vector<Transfer::Point> sourceBarycentric;
    std::vector<std::array<double, 86u>> fullBodyWeights86;
    std::vector<std::uint32_t> dominantBody;
    std::vector<std::uint32_t> requiredSourceIDs;
};

// Build transfer ancestry without converting a projected coarse row into a
// fabricated source vertex. The optional C source-face/barycentric arrays are
// expanded to every coarse row because the current transfer API consumes them
// as a pair over the complete coarse vertex domain.
template<class LoadedSkinT, class PositionFn>
inline bool makeWholeBodySkinSourceConsumerRows(
        const LoadedSkinT& skin,
        const RegisteredSurface& region,
        PositionFn&& positionAt,
        std::span<const std::uint8_t> detailUsed,
        WholeBodySkinSourceConsumerRows& output,
        std::string& error) {
    using Transfer = numi::human::AcceptedFEMSkinTransfer;
    output = {};
    error.clear();
    constexpr std::size_t kBindings = 86u;
    const auto fail = [&](const char* message) {
        error = message;
        output = {};
        return false;
    };
    const std::size_t n = region.worldPoints.size();
    const std::size_t detailCount = skin.vertices.size();
    if (n == 0u || skin.indices.empty() || skin.indices.size() % 3u != 0u ||
        skin.bindings.size() != kBindings ||
        skin.fullWeights.size() != detailCount * kBindings ||
        region.sourceVertexIndices.size() != n ||
        region.donorBodyIndices.size() != n ||
        detailUsed.size() != detailCount)
        return fail("coarse source consumer requires exact geometry, donor rows, and full 86-body weights");
    const bool hasEmbedding = !region.sourceEmbeddingFaceIndices.empty() ||
                              !region.sourceEmbeddingBarycentric.empty();
    if (hasEmbedding &&
        (region.sourceEmbeddingFaceIndices.size() != n ||
         region.sourceEmbeddingBarycentric.size() != n))
        return fail("coarse source embedding does not cover every candidate row");
    if (!region.sourceEmbeddingResidualWorldM.empty() &&
        (!hasEmbedding || region.sourceEmbeddingResidualWorldM.size() != n))
        return fail("coarse source residual does not cover its embedding rows");
    // The whole-body candidate uses mixed ancestry: a direct row carries an
    // exact ID and a sentinel face, while a projected row carries face+bary.
    // Validate that each row has a usable identity before expanding mixed
    // ancestry to the transfer API's complete paired arrays.
    for (std::size_t row = 0u; row < n; ++row) {
        const auto direct = region.sourceVertexIndices[row];
        const auto face = hasEmbedding
            ? region.sourceEmbeddingFaceIndices[row] : NM_INVALID_INDEX;
        if (face == NM_INVALID_INDEX) {
            if (direct == NM_INVALID_INDEX)
                return fail("coarse source row has neither a direct ID nor an embedding");
        } else {
            if (face >= skin.indices.size() / 3u)
                return fail("coarse source embedding face is outside detailed topology");
            if (!validateBarycentricCoordinates(region.sourceEmbeddingBarycentric[row], error))
                return false;
        }
    }

    // Deterministically choose the first incident source triangle for direct
    // identity rows, with its exact one-hot corner. This lets the legacy C
    // transfer consume one paired face/barycentric array for mixed rows.
    std::vector<Transfer::Face> firstIncident(detailCount);
    std::vector<std::uint8_t> hasIncident(detailCount, 0u);
    std::vector<std::uint8_t> firstCorner(detailCount, 0u);
    std::vector<std::uint32_t> firstFaceIndex(detailCount, NM_INVALID_INDEX);
    for (std::size_t face = 0u; face < skin.indices.size() / 3u; ++face) {
        const Transfer::Face f{
            skin.indices[3u * face], skin.indices[3u * face + 1u],
            skin.indices[3u * face + 2u]};
        for (std::uint8_t k = 0u; k < 3u; ++k) {
            const std::uint32_t source = f[k];
            if (source >= detailCount)
                return fail("detailed source topology contains an out-of-range vertex");
            if (!hasIncident[source]) {
                firstIncident[source] = f;
                firstCorner[source] = k;
                firstFaceIndex[source] = static_cast<std::uint32_t>(face);
                hasIncident[source] = 1u;
            }
        }
    }

    WholeBodySkinSourceConsumerRows next;
    next.directSourceIDs = region.sourceVertexIndices;
    next.sourceFaces.resize(n);
    next.sourceBarycentric.resize(n);
    next.fullBodyWeights86.resize(n);
    next.dominantBody.resize(n);
    std::vector<double> interpolated(kBindings);
    for (std::size_t row = 0u; row < n; ++row) {
        const std::uint32_t direct = region.sourceVertexIndices[row];
        const bool hasRowEmbedding = hasEmbedding &&
            region.sourceEmbeddingFaceIndices[row] != NM_INVALID_INDEX;
        if (direct == NM_INVALID_INDEX && !hasRowEmbedding)
            return fail("projected coarse row has no exact source-face embedding");
        if (hasRowEmbedding) {
            const std::uint32_t sourceFace = region.sourceEmbeddingFaceIndices[row];
            const auto faceOffset = std::size_t(sourceFace) * 3u;
            if (faceOffset + 2u >= skin.indices.size())
                return fail("coarse source embedding face is outside detailed topology");
            next.sourceFaces[row] = {
                skin.indices[faceOffset], skin.indices[faceOffset + 1u],
                skin.indices[faceOffset + 2u]};
            next.sourceBarycentric[row] = region.sourceEmbeddingBarycentric[row];
            if (direct != NM_INVALID_INDEX) {
                bool exact = false;
                for (std::size_t k = 0u; k < 3u; ++k)
                    exact = exact ||
                        (next.sourceFaces[row][k] == direct &&
                         std::abs(next.sourceBarycentric[row][k] - 1.0) <= 1.0e-8);
                if (!exact)
                    return fail("direct coarse source ID disagrees with its one-hot embedding");
            }
        } else {
            if (direct >= detailCount || !hasIncident[direct])
                return fail("direct coarse source ID has no incident detailed source triangle");
            next.sourceFaces[row] = firstIncident[direct];
            next.sourceBarycentric[row] = {0.0, 0.0, 0.0};
            next.sourceBarycentric[row][firstCorner[direct]] = 1.0;
        }

        V3 sampledSourcePoint{};
        std::uint32_t dominantBody = NM_INVALID_INDEX;
        double dominantWeight = 0.0;
        std::fill(interpolated.begin(), interpolated.end(), 0.0);
        const std::uint32_t embeddingFace = hasRowEmbedding
            ? region.sourceEmbeddingFaceIndices[row]
            : firstFaceIndex[region.sourceVertexIndices[row]];
        if (embeddingFace == NM_INVALID_INDEX ||
            !evaluateRegisteredSkinSourceEmbedding(
                skin, embeddingFace, next.sourceBarycentric[row], positionAt,
                sampledSourcePoint, std::span<double>(interpolated),
                dominantBody, dominantWeight, error))
            return false;

        if (dominantBody == NM_INVALID_INDEX ||
            region.donorBodyIndices[row] != dominantBody)
            return fail("coarse donor owner differs from dominant interpolated full-body binding");
        std::copy(interpolated.begin(), interpolated.end(),
                  next.fullBodyWeights86[row].begin());
        next.dominantBody[row] = dominantBody;
        for (std::size_t k = 0u; k < 3u; ++k) {
            const std::uint32_t source = next.sourceFaces[row][k];
            if (source >= detailCount || !detailUsed[source])
                return fail("coarse source embedding references an inactive or invalid detail vertex");
            next.requiredSourceIDs.push_back(source);
        }
        if (direct != NM_INVALID_INDEX) {
            if (direct >= detailCount || !detailUsed[direct])
                return fail("direct coarse source ID is inactive or invalid");
            next.requiredSourceIDs.push_back(direct);
        }
        (void)sampledSourcePoint;
        (void)dominantWeight;
    }
    std::sort(next.requiredSourceIDs.begin(), next.requiredSourceIDs.end());
    next.requiredSourceIDs.erase(
        std::unique(next.requiredSourceIDs.begin(), next.requiredSourceIDs.end()),
        next.requiredSourceIDs.end());
    output = std::move(next);
    error.clear();
    return true;
}

} // namespace metalrobo::human::coarse_tissue
