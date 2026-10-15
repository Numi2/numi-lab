#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace numi::cardboard::paper_materials {

struct SelectedPaths {
    std::filesystem::path liner;
    std::filesystem::path medium;
};

inline SelectedPaths selectPaths(
    const std::filesystem::path& materialDirectory,
    const std::string_view preset,
    const bool retainedMatrixOverlay,
    const std::filesystem::path& explicitLiner = {},
    const std::filesystem::path& explicitMedium = {}) {
    SelectedPaths paths;
    if (preset == "literature2009") {
        const char* suffix = retainedMatrixOverlay ? "hill_ideal" : "hill_bilinear_fit";
        paths.liner = materialDirectory /
            (std::string("hajali2009_liner_") + suffix + ".nmatter");
        paths.medium = materialDirectory /
            (std::string("hajali2009_medium_") + suffix + ".nmatter");
    } else {
        paths.liner = materialDirectory / "nagasawa2013_liner_orthotropic.nmatter";
        paths.medium = materialDirectory / "nagasawa2013_medium_orthotropic.nmatter";
    }
    if (!explicitLiner.empty()) paths.liner = explicitLiner;
    if (!explicitMedium.empty()) paths.medium = explicitMedium;
    return paths;
}

inline bool isBilinearFitIdentity(const std::string_view materialName) {
    return materialName == "hajali2009_liner_hill_bilinear_fit" ||
           materialName == "hajali2009_medium_hill_bilinear_fit";
}

} // namespace numi::cardboard::paper_materials
