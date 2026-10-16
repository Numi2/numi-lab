#pragma once

#include "numi/matter/ir.hpp"

namespace numi::matter {

[[nodiscard]] inline CompileResult compileMatterWorld(
    const WorldSource& source,
    const CompileOptions& options = {}
) {
    return compileWorld(source, options);
}

[[nodiscard]] inline bool writeMatterPackage(
    const CompileResult& compiled,
    const std::filesystem::path& path,
    std::string* error = nullptr
) {
    return writePackage(compiled, path, error);
}

[[nodiscard]] inline bool readMatterPackage(
    const std::filesystem::path& path,
    CompiledMatterWorld& world,
    std::string* generatedMetal = nullptr,
    std::string* error = nullptr
) {
    return readPackage(path, world, generatedMetal, error);
}

[[nodiscard]] inline std::string emitSpecializedConstitutiveMetal(
    const std::span<const ConstitutiveProgram> programs
) {
    return emitSpecializedMetal(programs);
}

} // namespace numi::matter
