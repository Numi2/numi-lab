#pragma once

#include "numi/matter/ir.hpp"

#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace numi::matter {

struct CompileOptions {
    bool emitSpecializedMetal = true;
    bool deterministic = true;
    bool preferImplicitFEM = true;
    double cfl = 0.35;
    std::uint32_t maximumRateExponent = NM_MAX_RATE_EXPONENT;
    std::uint32_t maximumExpressionStack = NM_MAX_EXPRESSION_STACK;
};

struct CompileResult {
    CompiledMatterWorld world;
    std::string generatedMetal;
    std::vector<Diagnostic> diagnostics;
    [[nodiscard]] bool succeeded() const noexcept;
};

[[nodiscard]] CompileResult compileMatterWorld(
    const WorldSource& source,
    const CompileOptions& options = {}
);
[[nodiscard]] bool writeMatterPackage(
    const CompileResult& compiled,
    const std::filesystem::path& path,
    std::string* error = nullptr
);
[[nodiscard]] bool readMatterPackage(
    const std::filesystem::path& path,
    CompiledMatterWorld& world,
    std::string* generatedMetal = nullptr,
    std::string* error = nullptr
);
[[nodiscard]] std::string emitSpecializedConstitutiveMetal(
    std::span<const ConstitutiveProgram> programs
);

} // namespace numi::matter
