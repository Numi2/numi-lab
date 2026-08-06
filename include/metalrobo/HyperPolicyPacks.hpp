#pragma once

#include "metalrobo/HyperPolicyProgram.hpp"
#include "metalrobo/LearningPacks.hpp"

#include <filesystem>

namespace metalrobo {

inline constexpr std::uint32_t kHyperPolicyPackFormatVersion = 1u;

[[nodiscard]] LearningPackResult writeHyperPolicyPack(
    const HyperPolicyPack& pack,
    const std::filesystem::path& path
);

[[nodiscard]] LearningPackResult readHyperPolicyPack(
    const std::filesystem::path& path,
    HyperPolicyPack& output
);

} // namespace metalrobo
