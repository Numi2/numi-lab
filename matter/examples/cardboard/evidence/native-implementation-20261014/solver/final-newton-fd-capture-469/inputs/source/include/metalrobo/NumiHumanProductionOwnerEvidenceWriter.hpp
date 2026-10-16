#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace metalrobo {

// Publishes one immutable production-owner evidence record. Missing output
// directories are created component-by-component and each new directory entry
// is synced before descent. The regular temporary file is fully flushed on
// Darwin before a no-replace link publishes the final name; the final directory
// is synced after the temporary name is removed. Symbolic and parent-directory
// components fail closed, and the verified final directory descriptor is held
// through publication.
[[nodiscard]] bool publishNumiHumanProductionOwnerEvidenceNoReplace(
    const std::filesystem::path& target,
    std::string_view contents,
    std::string& error) noexcept;

} // namespace metalrobo
