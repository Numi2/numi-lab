#include "metalrobo/NumiHumanLoadedKneeNumanX.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

} // namespace

int main() {
    using namespace metalrobo;
    std::vector<float> baseline(kNumiHumanLoadedKneeMuscleCountV1, 0.0f);
    std::vector<float> command = baseline;
    constexpr float increment = 0.25f;
    for (const std::uint32_t index :
         kNumiHumanLoadedKneeQATMuscleIndicesV1) {
        command[index] = baseline[index] + increment;
    }

    NumiHumanLoadedKneeNumanXCommandAdmissionV1 admission;
    std::string error;
    require(admitNumiHumanLoadedKneeNumanXCommandV1(
                baseline, command, increment, admission, error),
            "exact QAT-only increment was rejected");
    require(admission.baselineFingerprint != 0u &&
                admission.commandFingerprint != 0u &&
                admission.baselineFingerprint != admission.commandFingerprint &&
                admission.candidateOnly &&
                !admission.productionAuthorized &&
                !admission.standClaimAuthorized,
            "admitted command lost its candidate-only boundary");
    const auto authority = makeNumiHumanLoadedKneeNumanXCandidateAuthorityV1(
        admission, 0x101u, 0x202u, 1u);
    const auto replayAuthority =
        makeNumiHumanLoadedKneeNumanXCandidateAuthorityV1(
            admission, 0x101u, 0x202u, 1u);
    require(authority.brainProgramFingerprint != 0u &&
                authority.brainProgramFingerprint ==
                    replayAuthority.brainProgramFingerprint &&
                authority.jointCommitFingerprint ==
                    replayAuthority.jointCommitFingerprint &&
                authority.candidateOnly &&
                !authority.productionAuthorized &&
                !authority.standClaimAuthorized,
            "candidate authority is not deterministic and fail-closed");

    auto wrongMuscle = command;
    wrongMuscle[404u] = increment;
    require(!admitNumiHumanLoadedKneeNumanXCommandV1(
                baseline, wrongMuscle, increment, admission, error),
            "non-QAT mutation was admitted");
    auto missingQAT = command;
    missingQAT[405u] = baseline[405u];
    require(!admitNumiHumanLoadedKneeNumanXCommandV1(
                baseline, missingQAT, increment, admission, error),
            "partial QAT command was admitted");
    auto wrongIncrement = command;
    wrongIncrement[413u] += 0.125f;
    require(!admitNumiHumanLoadedKneeNumanXCommandV1(
                baseline, wrongIncrement, increment, admission, error),
            "nonuniform QAT increment was admitted");
    auto nonFinite = command;
    nonFinite[414u] = std::numeric_limits<float>::quiet_NaN();
    require(!admitNumiHumanLoadedKneeNumanXCommandV1(
                baseline, nonFinite, increment, admission, error),
            "non-finite QAT command was admitted");
    require(!admitNumiHumanLoadedKneeNumanXCommandV1(
                baseline, command, 0.0f, admission, error),
            "zero QAT increment was admitted");
    require(!admitNumiHumanLoadedKneeNumanXCommandV1(
                baseline, command,
                std::numeric_limits<float>::infinity(), admission, error),
            "non-finite QAT increment was admitted");
    std::vector<float> shortCommand(command.begin(), command.end() - 1);
    require(!admitNumiHumanLoadedKneeNumanXCommandV1(
                baseline, shortCommand, increment, admission, error),
            "wrong muscle count was admitted");

    // Matter's moving-attachment reaction is the action on the bone. A transient
    // continuum therefore closes only after dP/dt is included; the static
    // R==F comparison below is deliberately false.
    NumiHumanLoadedKneeMomentumClosureV1 momentum;
    const std::array<double, 3u> reaction{{4.0, -2.0, -8.0}};
    const std::array<double, 3u> external{{5.0, -1.0, 2.0}};
    const std::array<double, 3u> gravity{{0.0, 0.0, -10.0}};
    const std::array<double, 3u> momentumRate{{1.0, 1.0, 0.0}};
    require(admitNumiHumanLoadedKneeMomentumClosureV1(
                reaction, external, gravity, momentumRate,
                10.0, 6.0, 10.0, 2.0, 1.0e-4, momentum, error),
            "exact transient momentum closure was rejected");
    require(momentum.residualNormNewtons == 0.0 &&
                momentum.toleranceNewtons > 0.0,
            "transient momentum closure did not preserve exact zero");
    const std::array<double, 3u> omittedMomentum{{0.0, 0.0, 0.0}};
    require(!admitNumiHumanLoadedKneeMomentumClosureV1(
                reaction, external, gravity, omittedMomentum,
                10.0, 6.0, 10.0, 0.0, 1.0e-4, momentum, error),
            "static force equality was admitted for a transient FEM step");
    auto wrongReaction = reaction;
    wrongReaction[2u] += 0.1;
    require(!admitNumiHumanLoadedKneeMomentumClosureV1(
                wrongReaction, external, gravity, momentumRate,
                10.1, 6.0, 10.0, 2.0, 1.0e-4, momentum, error),
            "out-of-tolerance FEM momentum residual was admitted");
    // Large opposing nodewise forces may cancel physically. They must not
    // inflate the 1e-4 relative budget as the old nodewise-L1 scale did.
    auto cancellationHiddenError = reaction;
    cancellationHiddenError[0u] += 0.05;
    require(!admitNumiHumanLoadedKneeMomentumClosureV1(
                cancellationHiddenError, external, gravity, momentumRate,
                1'000.0, 1'000.0, 1'000.0, 1'000.0, 1.0e-4,
                momentum, error),
            "nodewise cancellation inflated the FEM closure budget");

    std::cout << "numi_human_loaded_knee_numanx_command_admission=ok\n";
    return 0;
}
