#pragma once

#include "numi/matter/matter.hpp"

#include <limits>

namespace numi::matter::detail {

[[nodiscard]] constexpr std::uint64_t
femHumanAttachmentPointJacobianScalarCount(
    const std::uint32_t attachmentCount,
    const std::uint32_t rigidGeneralizedCapacity
) noexcept {
    return static_cast<std::uint64_t>(attachmentCount) * 3u *
        rigidGeneralizedCapacity;
}

[[nodiscard]] constexpr bool femHumanAttachmentPointJacobianStrideFits(
    const std::uint32_t attachmentCount,
    const std::uint32_t rigidGeneralizedCapacity
) noexcept {
    return femHumanAttachmentPointJacobianScalarCount(
        attachmentCount,
        rigidGeneralizedCapacity
    ) <= std::numeric_limits<std::uint32_t>::max();
}

[[nodiscard]] constexpr std::uint32_t
femHumanAttachmentPointJacobianStride(
    const std::uint32_t attachmentCount,
    const std::uint32_t rigidGeneralizedCapacity
) noexcept {
    return static_cast<std::uint32_t>(
        femHumanAttachmentPointJacobianScalarCount(
            attachmentCount,
            rigidGeneralizedCapacity
        )
    );
}

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
