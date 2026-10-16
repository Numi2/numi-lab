#pragma once

#include <array>
#include <charconv>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string_view>
#include <system_error>

namespace numi::matter::regional_consumer_capture {

inline constexpr std::uint64_t kAggregatePayloadCapBytes =
    64ull * 1024ull * 1024ull;

struct Selector {
    std::uint32_t control = 0u;
    std::uint32_t physics = 0u;
    std::uint32_t newton = 0u;
    std::uint32_t restart = 0u;
    std::uint32_t column = 0u;
    std::uint32_t environment = 0u;
};

[[nodiscard]] inline bool singleTickScope(
    const std::uint32_t maximumRateExponent,
    const std::uint32_t microtick,
    const std::uint32_t microtickCount) noexcept {
    return maximumRateExponent == 0u && microtick == 0u &&
        microtickCount == 1u;
}

[[nodiscard]] inline bool claimOnce(
    bool& attempted,
    const bool selectorMatches) noexcept {
    if (!selectorMatches || attempted) return false;
    attempted = true;
    return true;
}

enum class SelectorError {
    none,
    empty,
    wrong_field_count,
    non_decimal,
    overflow,
    restart_zero,
};

struct SelectorParse {
    Selector value{};
    SelectorError error = SelectorError::none;
    explicit operator bool() const noexcept {
        return error == SelectorError::none;
    }
};

[[nodiscard]] inline SelectorParse parseSelector(
    const std::string_view text) noexcept {
    if (text.empty()) return {{}, SelectorError::empty};
    std::array<std::uint32_t, 6> fields{};
    std::size_t field = 0u;
    std::size_t start = 0u;
    while (start <= text.size()) {
        const std::size_t separator = text.find(':', start);
        const std::size_t end = separator == std::string_view::npos
            ? text.size() : separator;
        if (field >= fields.size())
            return {{}, SelectorError::wrong_field_count};
        if (start == end) return {{}, SelectorError::non_decimal};
        for (std::size_t i = start; i < end; ++i) {
            if (text[i] < '0' || text[i] > '9')
                return {{}, SelectorError::non_decimal};
        }
        std::uint32_t value = 0u;
        const char* first = text.data() + start;
        const char* last = text.data() + end;
        const auto result = std::from_chars(first, last, value, 10);
        if (result.ec == std::errc::result_out_of_range)
            return {{}, SelectorError::overflow};
        if (result.ec != std::errc{} || result.ptr != last)
            return {{}, SelectorError::non_decimal};
        fields[field++] = value;
        if (separator == std::string_view::npos) break;
        start = separator + 1u;
    }
    if (field != fields.size())
        return {{}, SelectorError::wrong_field_count};
    if (fields[3] == 0u) return {{}, SelectorError::restart_zero};
    return {{fields[0], fields[1], fields[2], fields[3], fields[4], fields[5]},
            SelectorError::none};
}

[[nodiscard]] inline bool matches(
    const Selector& selector,
    const std::uint32_t control,
    const std::uint32_t physics,
    const std::uint32_t newton,
    const std::uint32_t restart,
    const std::uint32_t column,
    const std::uint32_t environment) noexcept {
    return selector.control == control && selector.physics == physics &&
        selector.newton == newton && selector.restart == restart &&
        selector.column == column && selector.environment == environment;
}

[[nodiscard]] inline std::array<std::uint32_t, 4> expectedGeneratorTag(
    const Selector& selector) noexcept {
    return {selector.control, selector.physics, selector.newton,
            selector.restart - 1u};
}

[[nodiscard]] inline bool checkedArrayBytes(
    const std::initializer_list<std::uint64_t> shape,
    const std::uint64_t elementBytes,
    const std::uint64_t cap,
    std::uint64_t& out) noexcept {
    if (shape.size() == 0u || elementBytes == 0u) return false;
    std::uint64_t count = 1u;
    for (const std::uint64_t extent : shape) {
        if (extent == 0u ||
            count > std::numeric_limits<std::uint64_t>::max() / extent)
            return false;
        count *= extent;
    }
    if (count > std::numeric_limits<std::uint64_t>::max() / elementBytes)
        return false;
    out = count * elementBytes;
    return out <= cap;
}

[[nodiscard]] inline bool checkedAppendBytes(
    const std::uint64_t current,
    const std::uint64_t append,
    const std::uint64_t cap,
    std::uint64_t& out) noexcept {
    if (current > cap || append > cap - current) return false;
    out = current + append;
    return true;
}

} // namespace numi::matter::regional_consumer_capture
