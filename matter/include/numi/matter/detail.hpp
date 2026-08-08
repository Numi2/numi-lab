#pragma once

#include "numi/matter/matter.hpp"

namespace numi::matter::detail {

struct ConstitutiveCompileResult {
    ConstitutiveProgram program;
    std::vector<Diagnostic> diagnostics;

    [[nodiscard]] bool succeeded() const noexcept;
};

[[nodiscard]] ConstitutiveCompileResult compileConstitutive(
    const MaterialProgram& material,
    std::uint32_t maximumStack
);

[[nodiscard]] std::uint64_t hashBytes(
    const void* data,
    std::size_t size,
    std::uint64_t seed = 1469598103934665603ull
) noexcept;

[[nodiscard]] std::uint64_t hashString(
    std::string_view value,
    std::uint64_t seed = 1469598103934665603ull
) noexcept;

} // namespace numi::matter::detail
