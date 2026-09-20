#pragma once

#include "numi/matter/matter.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
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

// Metal may flush FP32 denormals before arithmetic classification. Support
// friction therefore admits only either signed zero or a finite,
// nonnegative normal value; callers must reject rather than silently change
// an authored frictional law to frictionless.
[[nodiscard]] inline bool humanSupportFrictionAdmissible(
    const float friction
) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(friction);
    const std::uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude == 0u) return true;
    return (bits & 0x80000000u) == 0u &&
        magnitude >= 0x00800000u && magnitude < 0x7f800000u;
}

// NHCNT histories are consumed by FP32 Metal kernels. Evaluate their physical
// predicates in FP64 so hostile finite operands cannot overflow the admission
// arithmetic, then reject values whose squared tangent length or Coulomb
// radius cannot be represented by the downstream FP32 owner.
[[nodiscard]] inline bool humanSupportHistoryAdmissible(
    const nm_float4 history,
    const float friction,
    const nm_float4 groundNormal
) noexcept {
    if (!std::isfinite(history.x) || !std::isfinite(history.y) ||
        !std::isfinite(history.z) || !std::isfinite(history.w) ||
        history.w < 0.0f || !humanSupportFrictionAdmissible(friction)) {
        return false;
    }
    const double tangentProjection =
        static_cast<double>(history.x) * groundNormal.x +
        static_cast<double>(history.y) * groundNormal.y +
        static_cast<double>(history.z) * groundNormal.z;
    const double tangentSquared =
        static_cast<double>(history.x) * history.x +
        static_cast<double>(history.y) * history.y +
        static_cast<double>(history.z) * history.z;
    const double tangentMagnitude = std::sqrt(tangentSquared);
    const double coneRadius =
        static_cast<double>(friction) * history.w;
    const double scale = std::max(
        {1.0, tangentMagnitude, static_cast<double>(history.w)});
    const double tolerance = 8.0 *
        std::numeric_limits<float>::epsilon() * scale;
    return std::isfinite(tangentProjection) &&
        std::isfinite(tangentSquared) && std::isfinite(tangentMagnitude) &&
        std::isfinite(coneRadius) &&
        tangentSquared <= std::numeric_limits<float>::max() &&
        coneRadius <= std::numeric_limits<float>::max() &&
        std::abs(tangentProjection) <= tolerance &&
        tangentMagnitude <= coneRadius + tolerance;
}

// Prepared support histories extend a runtime identity only when a payload is
// actually present. Keeping this rule in the shared implementation lets a
// regression prove that the legacy empty-history fingerprint is unchanged
// without pinning a metallib-dependent whole-program fingerprint.
template <typename Mixer>
[[nodiscard]] std::uint64_t mixPreparedSupportHistoryFingerprint(
    const std::uint64_t legacyFingerprint,
    const std::span<const nm_float4> histories,
    const Mixer& mix
) noexcept {
    if (histories.empty()) {
        return legacyFingerprint;
    }
    return mix(legacyFingerprint, hashBytes(
        histories.data(), histories.size_bytes()));
}

} // namespace numi::matter::detail
