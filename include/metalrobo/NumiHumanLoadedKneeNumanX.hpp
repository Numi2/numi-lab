#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>

namespace metalrobo {

inline constexpr std::uint32_t kNumiHumanLoadedKneeMuscleCountV1 = 416u;
inline constexpr std::array<std::uint32_t, 4u>
    kNumiHumanLoadedKneeQATMuscleIndicesV1{{405u, 413u, 414u, 415u}};

// Scalar admission for the deliberately narrow loaded-left-knee experiment.
// It proves that the command differs from its passive-preload baseline only by
// one explicit positive QAT increment. It does not admit a Brain policy or
// make a stand, walk, production, biological, or clinical claim.
struct NumiHumanLoadedKneeNumanXCommandAdmissionV1 {
    std::uint64_t baselineFingerprint = 0u;
    std::uint64_t commandFingerprint = 0u;
    float increment = 0.0f;
    bool candidateOnly = true;
    bool productionAuthorized = false;
    bool standClaimAuthorized = false;
};

[[nodiscard]] bool admitNumiHumanLoadedKneeNumanXCommandV1(
    std::span<const float> baseline,
    std::span<const float> command,
    float increment,
    NumiHumanLoadedKneeNumanXCommandAdmissionV1& output,
    std::string& error
) noexcept;

// Deterministic candidate identities used only to exercise the real NumanX
// proposal/preflight/ACK/publication protocol for this physical differential
// gate. These identities are evidence-bound test authority, not a substitute
// for a production NumiBrain policy artifact.
struct NumiHumanLoadedKneeNumanXCandidateAuthorityV1 {
    std::uint64_t parameterVersionFingerprint = 0u;
    std::uint64_t motorProfileFingerprint = 0u;
    std::uint64_t speciesTemplateFingerprint = 0u;
    std::uint64_t compiledSpeciesTemplateFingerprint = 0u;
    std::uint64_t brainProgramFingerprint = 0u;
    std::uint64_t fastProgramFingerprint = 0u;
    std::uint64_t brainShadowStateFingerprint = 0u;
    std::uint64_t fastGateFingerprint = 0u;
    std::uint64_t jointCommitFingerprint = 0u;
    bool candidateOnly = true;
    bool productionAuthorized = false;
    bool standClaimAuthorized = false;
};

[[nodiscard]] NumiHumanLoadedKneeNumanXCandidateAuthorityV1
makeNumiHumanLoadedKneeNumanXCandidateAuthorityV1(
    const NumiHumanLoadedKneeNumanXCommandAdmissionV1& command,
    std::uint64_t baseManifestFingerprint,
    std::uint64_t sourceComplianceFingerprint,
    std::uint64_t attemptIdentifier
) noexcept;

// One-region linear-momentum certificate for the loaded-knee FEM path.
// Matter publishes attachmentReaction as the action on the owning bone, so
// its sign convention is
//
//   attachmentReaction = externalForce + mass * gravity - dP/dt.
//
// The admitted residual is therefore R - F - Mg + dP/dt.  This prevents a
// transient one-step solve from being judged against a static force equality.
struct NumiHumanLoadedKneeMomentumClosureV1 {
    std::array<double, 3u> attachmentReactionNewtons{};
    std::array<double, 3u> externalForceNewtons{};
    std::array<double, 3u> gravityForceNewtons{};
    std::array<double, 3u> momentumRateNewtons{};
    std::array<double, 3u> residualNewtons{};
    double residualNormNewtons = 0.0;
    // Aggregate physical scale: sum of the norms of the four resultant
    // vectors, with a 1 N floor. Nodewise cancellation cannot inflate it.
    double forceScaleNewtons = 0.0;
    double nodewiseScaleNewtons = 0.0;
    double physicalToleranceNewtons = 0.0;
    double roundoffAllowanceNewtons = 0.0;
    double toleranceNewtons = 0.0;
};

[[nodiscard]] bool admitNumiHumanLoadedKneeMomentumClosureV1(
    const std::array<double, 3u>& attachmentReactionNewtons,
    const std::array<double, 3u>& externalForceNewtons,
    const std::array<double, 3u>& gravityForceNewtons,
    const std::array<double, 3u>& momentumRateNewtons,
    double reactionL1Newtons,
    double externalL1Newtons,
    double gravityL1Newtons,
    double momentumRateL1Newtons,
    double relativeResidualTolerance,
    NumiHumanLoadedKneeMomentumClosureV1& output,
    std::string& error
) noexcept;

} // namespace metalrobo
