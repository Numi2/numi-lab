#pragma once

#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace numi_human_passive_coordinate_remap {

struct SourceQvPair {
    std::uint32_t qIndex;
    std::uint32_t vIndex;
};

inline std::uint32_t mapSourceIndex(
    std::span<const std::uint32_t> sourceToRuntime,
    const std::uint32_t sourceIndex,
    const std::uint32_t runtimeCount,
    const std::string_view label
) {
    if (sourceIndex == std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument(std::string(label) + " uses an invalid source index");
    const std::uint32_t runtimeIndex = sourceToRuntime.empty()
        ? sourceIndex
        : (sourceIndex < sourceToRuntime.size()
            ? sourceToRuntime[sourceIndex]
            : throw std::out_of_range(std::string(label) + " escapes the admitted source map"));
    if (runtimeIndex == std::numeric_limits<std::uint32_t>::max() ||
        runtimeIndex >= runtimeCount)
        throw std::out_of_range(std::string(label) + " maps outside the augmented model");
    return runtimeIndex;
}

inline void validateMap(
    const std::span<const std::uint32_t> sourceToRuntime,
    const std::uint32_t runtimeCount,
    const std::string_view label
) {
    std::vector<std::uint8_t> seen(runtimeCount, 0u);
    for (const std::uint32_t runtimeIndex : sourceToRuntime) {
        if (runtimeIndex == std::numeric_limits<std::uint32_t>::max() ||
            runtimeIndex >= runtimeCount)
            throw std::out_of_range(std::string(label) + " contains an out-of-range mapped index");
        if (seen[runtimeIndex] != 0u)
            throw std::invalid_argument(std::string(label) + " is not injective");
        seen[runtimeIndex] = 1u;
    }
}

template<class Coupling>
void remapSourceCouplings(
    std::vector<Coupling>& couplings,
    const std::span<const std::uint32_t> sourceVToRuntime,
    const std::uint32_t runtimeNv
) {
    validateMap(sourceVToRuntime, runtimeNv, "source V map");
    for (auto& coupling : couplings) {
        coupling.targetDofIndex = mapSourceIndex(
            sourceVToRuntime, coupling.targetDofIndex, runtimeNv,
            "passive coupling target");
        coupling.sourceDofIndex = mapSourceIndex(
            sourceVToRuntime, coupling.sourceDofIndex, runtimeNv,
            "passive coupling source");
    }
}

template<class Dof>
void validateMappedSourceQvPairs(
    const std::span<const Dof> runtimeDofs,
    const std::uint32_t articulationQOffset,
    const std::uint32_t articulationNq,
    const std::uint32_t articulationVOffset,
    const std::uint32_t articulationNv,
    const std::span<const std::uint32_t> sourceQToRuntime,
    const std::span<const std::uint32_t> sourceVToRuntime,
    const std::span<const SourceQvPair> sourcePairs
) {
    validateMap(sourceQToRuntime, articulationQOffset + articulationNq,
        "source q map");
    validateMap(sourceVToRuntime, articulationVOffset + articulationNv,
        "source V map");
    for (const SourceQvPair pair : sourcePairs) {
        const auto q = mapSourceIndex(sourceQToRuntime, pair.qIndex,
            articulationQOffset + articulationNq, "passive source q binding");
        const auto v = mapSourceIndex(sourceVToRuntime, pair.vIndex,
            articulationVOffset + articulationNv, "passive source V binding");
        if (q < articulationQOffset || q >= articulationQOffset + articulationNq ||
            v < articulationVOffset || v >= articulationVOffset + articulationNv ||
            v >= runtimeDofs.size() || runtimeDofs[v].qIndex != q)
            throw std::invalid_argument(
                "mapped passive source q/V pair disagrees with augmented DoF ownership");
    }
}

} // namespace numi_human_passive_coordinate_remap
