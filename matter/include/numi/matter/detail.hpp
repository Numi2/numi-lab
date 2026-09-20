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

// Device-program identity includes host command-encoding policy as well as the
// loaded metallib. Bump the revision whenever dispatch ordering, barriers, or
// pre-apply validation changes without an ABI or metallib-format change.
inline constexpr std::uint64_t kRuntimeExecutionPolicyDomain =
    0x4e4d52554e504f4cull; // "NMRUNPOL"
inline constexpr std::uint64_t kRuntimeExecutionPolicyRevision = 1u;

template <typename Mixer>
[[nodiscard]] std::uint64_t mixRuntimeExecutionPolicyFingerprint(
    const std::uint64_t fingerprint,
    const Mixer& mix
) noexcept {
    return mix(
        mix(fingerprint, kRuntimeExecutionPolicyDomain),
        kRuntimeExecutionPolicyRevision);
}

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

// NHCNT, direct-runtime, and restored histories are consumed by the FP32 Metal
// owner. Mirror its terminal stored-history predicate here: admission must not
// accept a tolerance-band value that deterministic GPU certification rejects.
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
    const float tangentProjection = std::fma(history.x, groundNormal.x,
        std::fma(history.y, groundNormal.y,
            history.z * groundNormal.z));
    const float tangentSquared = std::fma(history.x, history.x,
        std::fma(history.y, history.y, history.z * history.z));
    const float tangentMagnitude = std::sqrt(tangentSquared);
    const float coneRadius = friction * history.w;
    const float coneMargin = std::fma(
        friction, history.w, -tangentMagnitude);
    const float scale = std::max({1.0f, tangentMagnitude, history.w});
    const float projectionTolerance = 8.0f *
        std::numeric_limits<float>::epsilon() * scale;
    const bool zeroTangent = history.x == 0.0f &&
        history.y == 0.0f && history.z == 0.0f;
    return std::isfinite(tangentProjection) &&
        std::isfinite(tangentSquared) && std::isfinite(tangentMagnitude) &&
        std::isfinite(coneRadius) && std::isfinite(coneMargin) &&
        coneMargin >= 0.0f &&
        std::abs(tangentProjection) <= projectionTolerance &&
        ((history.w != 0.0f && friction != 0.0f) || zeroTangent);
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
