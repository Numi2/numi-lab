#include "metalrobo/NumiHumanLoadedKneeNumanX.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace metalrobo {
namespace {

constexpr std::uint64_t kFNVOffset = 14695981039346656037ull;
constexpr std::uint64_t kFNVPrime = 1099511628211ull;

std::uint64_t appendByte(
    const std::uint64_t hash,
    const std::uint8_t value
) noexcept {
    return (hash ^ value) * kFNVPrime;
}

std::uint64_t appendU64(
    std::uint64_t hash,
    const std::uint64_t value
) noexcept {
    for (std::uint32_t shift = 0u; shift < 64u; shift += 8u) {
        hash = appendByte(hash, static_cast<std::uint8_t>(value >> shift));
    }
    return hash;
}

std::uint64_t excitationFingerprint(
    const std::span<const float> values,
    const std::uint64_t domain
) noexcept {
    std::uint64_t hash = appendU64(kFNVOffset, domain);
    hash = appendU64(hash, values.size());
    for (const float value : values) {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
        for (std::uint32_t shift = 0u; shift < 32u; shift += 8u) {
            hash = appendByte(
                hash, static_cast<std::uint8_t>(bits >> shift));
        }
    }
    return hash == 0u ? kFNVOffset : hash;
}

bool qatIndex(const std::uint32_t index) noexcept {
    for (const std::uint32_t qat :
         kNumiHumanLoadedKneeQATMuscleIndicesV1) {
        if (index == qat) return true;
    }
    return false;
}

std::uint64_t candidateWord(
    const std::uint64_t domain,
    const NumiHumanLoadedKneeNumanXCommandAdmissionV1& command,
    const std::uint64_t baseManifestFingerprint,
    const std::uint64_t sourceComplianceFingerprint,
    const std::uint64_t attemptIdentifier
) noexcept {
    std::uint64_t hash = appendU64(kFNVOffset, domain);
    hash = appendU64(hash, command.baselineFingerprint);
    hash = appendU64(hash, command.commandFingerprint);
    hash = appendU64(
        hash, std::bit_cast<std::uint32_t>(command.increment));
    hash = appendU64(hash, baseManifestFingerprint);
    hash = appendU64(hash, sourceComplianceFingerprint);
    hash = appendU64(hash, attemptIdentifier);
    return hash == 0u ? kFNVOffset : hash;
}

} // namespace

bool admitNumiHumanLoadedKneeNumanXCommandV1(
    const std::span<const float> baseline,
    const std::span<const float> command,
    const float increment,
    NumiHumanLoadedKneeNumanXCommandAdmissionV1& output,
    std::string& error
) noexcept {
    output = {};
    error.clear();
    if (baseline.size() != kNumiHumanLoadedKneeMuscleCountV1 ||
        command.size() != kNumiHumanLoadedKneeMuscleCountV1) {
        error = "loaded-knee NumanX command requires exactly 416 muscles";
        return false;
    }
    if (!std::isfinite(increment) || !(increment > 0.0f) ||
        increment > 1.0f) {
        error = "loaded-knee NumanX QAT increment must be finite in (0, 1]";
        return false;
    }
    for (std::uint32_t index = 0u; index < baseline.size(); ++index) {
        const float base = baseline[index];
        const float driven = command[index];
        if (!std::isfinite(base) || !std::isfinite(driven) ||
            base < 0.0f || base > 1.0f || driven < 0.0f || driven > 1.0f) {
            error = "loaded-knee NumanX excitation escaped finite [0, 1]";
            return false;
        }
        if (qatIndex(index)) {
            const float expected = base + increment;
            // A positive scalar can round away at the stored FP32 baseline.
            // Require every driven QAT muscle to actually change; domain-
            // separated fingerprints alone cannot establish that contrast.
            if (!(expected > base) || !(expected <= 1.0f) ||
                std::bit_cast<std::uint32_t>(driven) !=
                    std::bit_cast<std::uint32_t>(expected)) {
                error = "loaded-knee NumanX QAT command is not the exact "
                    "baseline plus increment";
                return false;
            }
        } else if (std::bit_cast<std::uint32_t>(driven) !=
                   std::bit_cast<std::uint32_t>(base)) {
            error = "loaded-knee NumanX command changed a non-QAT muscle";
            return false;
        }
    }
    output.baselineFingerprint = excitationFingerprint(
        baseline, 0x4e584c4b42415345ull); // NXLKBASE
    output.commandFingerprint = excitationFingerprint(
        command, 0x4e584c4b434d4421ull); // NXLKCMD!
    output.increment = increment;
    return true;
}

NumiHumanLoadedKneeNumanXCandidateAuthorityV1
makeNumiHumanLoadedKneeNumanXCandidateAuthorityV1(
    const NumiHumanLoadedKneeNumanXCommandAdmissionV1& command,
    const std::uint64_t baseManifestFingerprint,
    const std::uint64_t sourceComplianceFingerprint,
    const std::uint64_t attemptIdentifier
) noexcept {
    NumiHumanLoadedKneeNumanXCandidateAuthorityV1 result;
    result.parameterVersionFingerprint = candidateWord(
        0x4e584c4b5041524dull, command, baseManifestFingerprint,
        sourceComplianceFingerprint, attemptIdentifier);
    result.motorProfileFingerprint = candidateWord(
        0x4e584c4b4d4f5452ull, command, baseManifestFingerprint,
        sourceComplianceFingerprint, attemptIdentifier);
    result.speciesTemplateFingerprint = candidateWord(
        0x4e584c4b53504543ull, command, baseManifestFingerprint,
        sourceComplianceFingerprint, attemptIdentifier);
    result.compiledSpeciesTemplateFingerprint = candidateWord(
        0x4e584c4b434f4d50ull, command, baseManifestFingerprint,
        sourceComplianceFingerprint, attemptIdentifier);
    result.brainProgramFingerprint = candidateWord(
        0x4e584c4b42524149ull, command, baseManifestFingerprint,
        sourceComplianceFingerprint, attemptIdentifier);
    result.fastProgramFingerprint = candidateWord(
        0x4e584c4b46415354ull, command, baseManifestFingerprint,
        sourceComplianceFingerprint, attemptIdentifier);
    result.brainShadowStateFingerprint = candidateWord(
        0x4e584c4b53484457ull, command, baseManifestFingerprint,
        sourceComplianceFingerprint, attemptIdentifier);
    result.fastGateFingerprint = candidateWord(
        0x4e584c4b46474154ull, command, baseManifestFingerprint,
        sourceComplianceFingerprint, attemptIdentifier);
    result.jointCommitFingerprint = candidateWord(
        0x4e584c4b4a434d54ull, command, baseManifestFingerprint,
        sourceComplianceFingerprint, attemptIdentifier);
    return result;
}

bool admitNumiHumanLoadedKneeMomentumClosureV1(
    const std::array<double, 3u>& attachmentReactionNewtons,
    const std::array<double, 3u>& externalForceNewtons,
    const std::array<double, 3u>& gravityForceNewtons,
    const std::array<double, 3u>& momentumRateNewtons,
    const double reactionL1Newtons,
    const double externalL1Newtons,
    const double gravityL1Newtons,
    const double momentumRateL1Newtons,
    const double relativeResidualTolerance,
    NumiHumanLoadedKneeMomentumClosureV1& output,
    std::string& error
) noexcept {
    output = {};
    error.clear();
    const auto finiteVector = [](const std::array<double, 3u>& value) {
        return std::all_of(value.begin(), value.end(), [](const double scalar) {
            return std::isfinite(scalar);
        });
    };
    if (!finiteVector(attachmentReactionNewtons) ||
        !finiteVector(externalForceNewtons) ||
        !finiteVector(gravityForceNewtons) ||
        !finiteVector(momentumRateNewtons) ||
        !std::isfinite(reactionL1Newtons) || reactionL1Newtons < 0.0 ||
        !std::isfinite(externalL1Newtons) || externalL1Newtons < 0.0 ||
        !std::isfinite(gravityL1Newtons) || gravityL1Newtons < 0.0 ||
        !std::isfinite(momentumRateL1Newtons) ||
            momentumRateL1Newtons < 0.0 ||
        !std::isfinite(relativeResidualTolerance) ||
            !(relativeResidualTolerance > 0.0) ||
            relativeResidualTolerance > 1.0e-3) {
        error = "loaded-knee momentum closure input is invalid";
        return false;
    }
    NumiHumanLoadedKneeMomentumClosureV1 candidate;
    candidate.attachmentReactionNewtons = attachmentReactionNewtons;
    candidate.externalForceNewtons = externalForceNewtons;
    candidate.gravityForceNewtons = gravityForceNewtons;
    candidate.momentumRateNewtons = momentumRateNewtons;
    double squaredResidual = 0.0;
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
        candidate.residualNewtons[axis] =
            attachmentReactionNewtons[axis] -
            externalForceNewtons[axis] - gravityForceNewtons[axis] +
            momentumRateNewtons[axis];
        squaredResidual += candidate.residualNewtons[axis] *
            candidate.residualNewtons[axis];
    }
    candidate.residualNormNewtons = std::sqrt(squaredResidual);
    const auto vectorNorm = [](const std::array<double, 3u>& value) {
        return std::sqrt(
            value[0u] * value[0u] + value[1u] * value[1u] +
            value[2u] * value[2u]);
    };
    candidate.forceScaleNewtons = std::max(
        1.0,
        vectorNorm(attachmentReactionNewtons) +
            vectorNorm(externalForceNewtons) +
            vectorNorm(gravityForceNewtons) +
            vectorNorm(momentumRateNewtons));
    candidate.nodewiseScaleNewtons = reactionL1Newtons +
        externalL1Newtons + gravityL1Newtons + momentumRateL1Newtons;
    // The physical gate remains 1e-4 of the aggregate resultants.  Keep the
    // distinct FP32 accumulation allowance small and explicit instead of
    // multiplying the potentially cancellation-heavy nodewise L1 by 1e-4.
    candidate.roundoffAllowanceNewtons =
        32.0 * std::numeric_limits<float>::epsilon() *
        candidate.nodewiseScaleNewtons;
    candidate.physicalToleranceNewtons =
        relativeResidualTolerance * candidate.forceScaleNewtons;
    candidate.toleranceNewtons = std::max(
        candidate.physicalToleranceNewtons,
        candidate.roundoffAllowanceNewtons);
    if (!std::isfinite(candidate.residualNormNewtons) ||
        !std::isfinite(candidate.forceScaleNewtons) ||
        !std::isfinite(candidate.nodewiseScaleNewtons) ||
        !std::isfinite(candidate.physicalToleranceNewtons) ||
        !std::isfinite(candidate.roundoffAllowanceNewtons) ||
        !std::isfinite(candidate.toleranceNewtons) ||
        candidate.residualNormNewtons > candidate.toleranceNewtons) {
        error = "loaded-knee transient FEM momentum balance exceeded the "
            "independent aggregate closure budget";
        return false;
    }
    output = candidate;
    return true;
}

} // namespace metalrobo
