#pragma once

#include "numi/matter/ir.hpp"

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace numi::matter {

struct ParseResult {
    MaterialProgram material;
    std::vector<Diagnostic> diagnostics;
    [[nodiscard]] bool succeeded() const noexcept;
};

[[nodiscard]] ParseResult parseMatterLanguage(std::string_view source);
[[nodiscard]] ParseResult parseMatterLanguageFile(
    const std::filesystem::path& path
);
[[nodiscard]] std::string dimensionName(Dimension dimension);
[[nodiscard]] std::uint64_t materialFingerprint(
    const MaterialProgram& material
) noexcept;

} // namespace numi::matter
