#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <optional>
#include <string_view>

namespace numiHumanAcceptedQAuditWindow {

struct Window {
    bool enabled = false;
    bool bounded = false;
    std::uint32_t firstAcceptedStep = 0u;
    std::uint32_t lastAcceptedStep = 0u;

    [[nodiscard]] constexpr bool contains(const std::uint32_t acceptedStep) const noexcept {
        return enabled && acceptedStep >= firstAcceptedStep &&
            acceptedStep <= lastAcceptedStep;
    }

    [[nodiscard]] constexpr std::uint32_t rowCount() const noexcept {
        return enabled ? lastAcceptedStep - firstAcceptedStep + 1u : 0u;
    }

    // completedSteps is zero-based; accepted samples are one-based. Bounded
    // windows split normal submissions at both edges and use one-step
    // submissions only for the requested Q/Jacobian rows.
    [[nodiscard]] constexpr std::uint32_t segmentSteps(
        const std::uint32_t completedSteps,
        const std::uint32_t remainingSteps,
        const std::uint32_t normalCap) const noexcept {
        if (remainingSteps == 0u || normalCap == 0u) return 0u;
        const std::uint32_t normal = std::min(remainingSteps, normalCap);
        if (!enabled) return normal;
        if (!bounded) return 1u;
        if (completedSteps == UINT32_MAX) return 0u;
        const std::uint32_t nextAcceptedStep = completedSteps + 1u;
        if (nextAcceptedStep < firstAcceptedStep)
            return std::min(normal, firstAcceptedStep - nextAcceptedStep);
        if (nextAcceptedStep <= lastAcceptedStep) return 1u;
        // Resume at the next normal-grid endpoint after an unaligned last
        // sample; then ordinary cap-sized segments retain their original phase.
        const std::uint32_t remainder = completedSteps % normalCap;
        const std::uint32_t toNextNormalEndpoint =
            remainder == 0u ? normalCap : normalCap - remainder;
        return std::min(normal, toNextNormalEndpoint);
    }
};


[[nodiscard]] constexpr bool shouldPresentAcceptedPose(
    const bool comMomentumAudit,
    const std::uint32_t auditSegmentSteps,
    const std::uint32_t presentationCadenceSteps,
    const bool boundedAuditWindow,
    const bool normalObserverSample,
    const std::uint32_t acceptedStep,
    const std::uint32_t finalAcceptedStep) noexcept {
    if (!normalObserverSample) return false;
    if (presentationCadenceSteps == 0u) return true;
    return !comMomentumAudit ||
        (!boundedAuditWindow && auditSegmentSteps >= presentationCadenceSteps) ||
        acceptedStep % presentationCadenceSteps == 0u ||
        acceptedStep == finalAcceptedStep;
}

enum class ParseError : std::uint8_t {
    none,
    boundsWithoutAudit,
    incompleteBounds,
    emptyHorizon,
    invalidInteger,
    outOfRange,
    reversedRange,
};

struct ParseResult {
    Window window{};
    ParseError error = ParseError::none;

    [[nodiscard]] constexpr bool ok() const noexcept {
        return error == ParseError::none;
    }
};

[[nodiscard]] inline std::optional<std::uint32_t> parseStep(
    const std::string_view text) noexcept {
    if (text.empty()) return std::nullopt;
    std::uint32_t value = 0u;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
        return std::nullopt;
    return value;
}

[[nodiscard]] inline ParseResult parse(
    const bool auditEnabled,
    const std::optional<std::string_view> firstSetting,
    const std::optional<std::string_view> lastSetting,
    const std::uint32_t acceptedHorizon) noexcept {
    const bool anyBound = firstSetting.has_value() || lastSetting.has_value();
    if (!auditEnabled) {
        if (anyBound) return {{}, ParseError::boundsWithoutAudit};
        return {{}, ParseError::none};
    }
    if (acceptedHorizon == 0u) return {{}, ParseError::emptyHorizon};
    if (firstSetting.has_value() != lastSetting.has_value())
        return {{}, ParseError::incompleteBounds};
    if (!anyBound)
        return {{true, false, 1u, acceptedHorizon}, ParseError::none};

    const auto first = parseStep(*firstSetting);
    const auto last = parseStep(*lastSetting);
    if (!first || !last) return {{}, ParseError::invalidInteger};
    if (*first == 0u || *last == 0u ||
        *first > acceptedHorizon || *last > acceptedHorizon)
        return {{}, ParseError::outOfRange};
    if (*first > *last) return {{}, ParseError::reversedRange};
    return {{true, true, *first, *last}, ParseError::none};
}

}  // namespace numiHumanAcceptedQAuditWindow
