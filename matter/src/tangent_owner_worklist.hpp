#pragma once
#include <cstdint>
#include <limits>
#include <initializer_list>
#include <span>
#include <vector>

namespace numi::matter::detail {

// Source tetrahedron IDs only. Physics/cache rows retain their original layout.
struct TangentOwnerWorklist {
    std::vector<std::uint32_t> tetrahedra;
    std::uint32_t wholeOwnerCount = 0u;
    std::uint32_t columnOwnerCount = 0u;
    std::uint32_t tasksPerEnvironment = 0u;
};

inline bool buildTangentOwnerWorklist(
    std::span<const std::uint8_t> wholeOwners, std::uint32_t environments,
    TangentOwnerWorklist& result) {
    result = {};
    if (environments == 0u ||
        wholeOwners.size() > std::numeric_limits<std::uint32_t>::max())
        return false;
    std::uint64_t whole = 0u;
    for (const auto owner : wholeOwners) {
        if (owner > 1u) return false;
        whole += owner;
    }
    const std::uint64_t columns = wholeOwners.size() - whole;
    const std::uint64_t tasks = whole + 9u * columns;
    if (tasks > std::numeric_limits<std::uint32_t>::max() ||
        (tasks != 0u && environments >
            std::numeric_limits<std::uint32_t>::max() / tasks))
        return false;
    result.wholeOwnerCount = static_cast<std::uint32_t>(whole);
    result.columnOwnerCount = static_cast<std::uint32_t>(columns);
    result.tasksPerEnvironment = static_cast<std::uint32_t>(tasks);
    result.tetrahedra.reserve(wholeOwners.size());
    for (const std::uint8_t selected : {std::uint8_t{1u}, std::uint8_t{0u}})
        for (std::uint32_t index = 0u; index < wholeOwners.size(); ++index)
            if (wholeOwners[index] == selected)
                result.tetrahedra.push_back(index);
    return true;
}
} // namespace numi::matter::detail
