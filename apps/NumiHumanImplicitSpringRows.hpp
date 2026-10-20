#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <system_error>
#include <vector>

namespace numi::human::implicit_spring_rows {

inline bool parseRequestedCount(
    const char* setting,
    const std::uint32_t maximum,
    std::uint32_t& count
) noexcept {
    count = 0u;
    if (setting == nullptr) return true;
    const char* end = setting;
    while (*end != '\0') ++end;
    const auto parsed = std::from_chars(setting, end, count);
    return parsed.ec == std::errc{} && parsed.ptr == end &&
        count > 0u && count <= maximum;
}

template <class SampleRange>
bool selectExactFlagPrefix(
    const SampleRange& samples,
    const std::uint32_t requestedCount,
    const std::uint32_t activeFlag,
    const std::uint32_t elasticAttachmentFlag,
    std::vector<std::uint8_t>& selected
) {
    selected.assign(samples.size(), 0u);
    const std::uint32_t exactFlags = activeFlag | elasticAttachmentFlag;
    std::uint32_t selectedCount = 0u;
    for (std::size_t row = 0u; row < samples.size(); ++row) {
        if (samples[row].flags != exactFlags ||
            selectedCount == requestedCount) {
            continue;
        }
        selected[row] = 1u;
        ++selectedCount;
    }
    return selectedCount == requestedCount;
}

enum class InertialBoundStatus : std::uint8_t {
    valid,
    badSelection,
    badSkinRow,
    badNode,
    nonfiniteRatio,
    explicitRowExceedsBound,
};

struct InertialBoundResult {
    std::size_t implicitSkinRows = 0u;
    std::size_t explicitSkinRows = 0u;
    double maximumExplicitRatio = 0.0;
};

template <class SampleRange, class NodeRange>
InertialBoundStatus checkExplicitSkinRows(
    const SampleRange& allSamples,
    const std::size_t firstSkinRow,
    const std::vector<std::uint8_t>& selectedByAdapter,
    const NodeRange& nodes,
    const double timestepSeconds,
    const double maximumRatio,
    const std::uint32_t activeFlag,
    const std::uint32_t elasticAttachmentFlag,
    InertialBoundResult& result
) {
    result = {};
    if (firstSkinRow > allSamples.size() ||
        selectedByAdapter.size() != allSamples.size() ||
        !(std::isfinite(timestepSeconds) && timestepSeconds > 0.0) ||
        !(std::isfinite(maximumRatio) && maximumRatio > 0.0)) {
        return InertialBoundStatus::badSelection;
    }
    const std::uint32_t exactFlags = activeFlag | elasticAttachmentFlag;
    for (std::size_t row = firstSkinRow; row < allSamples.size(); ++row) {
        const auto& sample = allSamples[row];
        if (sample.flags != exactFlags) return InertialBoundStatus::badSkinRow;
        if (sample.slaveNode >= nodes.size()) return InertialBoundStatus::badNode;
        const auto& node = nodes[sample.slaveNode];
        const double mass = node.positionAndMass.w;
        if (node.restAndFixed.w != 0.0f || !(std::isfinite(mass) && mass > 0.0))
            return InertialBoundStatus::badNode;
        const double stiffness =
            static_cast<double>(sample.stiffnessAndNormalStrainPerPressure.x) *
            static_cast<double>(sample.bodyLocalPointAndArea.w);
        const double ratio = stiffness * timestepSeconds * timestepSeconds / mass;
        if (!std::isfinite(ratio)) return InertialBoundStatus::nonfiniteRatio;
        if (selectedByAdapter[row] != 0u) {
            ++result.implicitSkinRows;
            continue;
        }
        ++result.explicitSkinRows;
        result.maximumExplicitRatio =
            std::max(result.maximumExplicitRatio, ratio);
        if (!(ratio < maximumRatio))
            return InertialBoundStatus::explicitRowExceedsBound;
    }
    return InertialBoundStatus::valid;
}

inline const char* statusMessage(const InertialBoundStatus status) noexcept {
    switch (status) {
        case InertialBoundStatus::valid:
            return "valid";
        case InertialBoundStatus::badSelection:
            return "implicit-spring prefix selection does not match the source rows";
        case InertialBoundStatus::badSkinRow:
            return "skin attachment row is not an active elastic attachment";
        case InertialBoundStatus::badNode:
            return "skin elastic attachment has an invalid, fixed, or massless FEM node";
        case InertialBoundStatus::nonfiniteRatio:
            return "skin elastic attachment inertial ratio is nonfinite";
        case InertialBoundStatus::explicitRowExceedsBound:
            return "explicit skin elastic attachment exceeds its inertial bound";
    }
    return "unknown skin attachment guard status";
}

} // namespace numi::human::implicit_spring_rows
