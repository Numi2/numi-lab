#pragma once

#include "numi/matter/ir.hpp"

namespace numi::matter {

[[nodiscard]] inline ParseResult parseMatterLanguage(
    const std::string_view source
) {
    return parseMatter(source);
}

[[nodiscard]] inline ParseResult parseMatterLanguageFile(
    const std::filesystem::path& path
) {
    return parseMatterFile(path);
}

[[nodiscard]] inline std::uint64_t materialFingerprint(
    const MaterialProgram& material
) noexcept {
    return material.fingerprint;
}

} // namespace numi::matter
