#include "metalrobo/HumanBehaviorTrial.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

using namespace metalrobo;

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void requireContains(
    const std::string& text,
    const std::string& needle,
    const char* message
) {
    require(text.find(needle) != std::string::npos, message);
}

std::size_t countOccurrences(
    const std::string& text,
    const std::string& needle
) {
    if (needle.empty()) return 0u;
    std::size_t result = 0u;
    std::size_t offset = 0u;
    while ((offset = text.find(needle, offset)) != std::string::npos) {
        ++result;
        offset += needle.size();
    }
    return result;
}

HumanBehaviorDigest digest(const std::uint8_t marker) {
    HumanBehaviorDigest result{};
    for (std::size_t index = 0u; index < result.size(); ++index) {
        result[index] = static_cast<std::uint8_t>(marker + index);
    }
    return result;
}

HumanBehaviorAcceptedState initialState() {
    HumanBehaviorAcceptedState state{};
    state.acceptedRootSHA256 = digest(0x40u);
    state.acceptedTimestampNanoseconds = 1'000'000u;
    state.physicsGeneration = 100u;
    state.brainGeneration = 200u;
    state.sensorGeneration = 300u;
    state.controllerGeneration = 400u;
    state.taskGeneration = 500u;
    state.randomGeneration = 600u;
    return state;
}

HumanBehaviorAcceptedState successor(
    const HumanBehaviorAcceptedState& before,
    const std::uint8_t rootMarker,
    const std::uint64_t stepNanoseconds
) {
    auto after = before;
    after.acceptedRootSHA256 = digest(rootMarker);
    after.acceptedTimestampNanoseconds += stepNanoseconds;
    ++after.physicsGeneration;
    ++after.brainGeneration;
    ++after.sensorGeneration;
    ++after.controllerGeneration;
    ++after.taskGeneration;
    ++after.randomGeneration;
    return after;
}

HumanBehaviorTrialDescriptor standingDescriptor() {
    HumanBehaviorTrialDescriptor descriptor{};
    descriptor.trialID = "standing-core-check";
    descriptor.task = HumanBehaviorTask::standing;
    descriptor.seed = 0x1234u;
    descriptor.resetStateSHA256 = digest(0x01u);
    descriptor.protocolSHA256 = digest(0x02u);
    descriptor.stackSHA256 = digest(0x03u);
    descriptor.metricProgramSHA256 = digest(0x04u);
    descriptor.acceptedRootProofSchemaSHA256 = digest(0x05u);
    descriptor.taskLoweringReceiptSHA256 = digest(0x06u);
    descriptor.device = "Apple M4 Pro";
    descriptor.osBuild = "25A000";
    descriptor.executionID = "core-check-execution";
    descriptor.stepNanoseconds = 12'500u;
    descriptor.expectedAcceptedSteps = 3u;
    descriptor.initialAcceptedState = initialState();
    descriptor.initialPostureValid = true;
    descriptor.initialSettled = true;
    return descriptor;
}

HumanBehaviorAttemptAudit completeAudit(
    const std::uint32_t violationMask = 0u,
    const std::uint32_t forbiddenContactCount = 0u
) {
    HumanBehaviorAttemptAudit result{};
    result.coveredMask = kHumanBehaviorCompleteAuditCoverageMask;
    result.violationMask = violationMask;
    result.forbiddenContactCovered = true;
    result.forbiddenContactCount = forbiddenContactCount;
    return result;
}

HumanBehaviorAttempt rejectedAttempt(
    const std::uint64_t attemptIndex,
    const HumanBehaviorAcceptedState& state,
    const std::uint32_t violationMask = 0u
) {
    HumanBehaviorAttempt attempt{};
    attempt.attemptIndex = attemptIndex;
    attempt.transactionFingerprint = 10'000u + attemptIndex;
    attempt.disposition = HumanBehaviorAttemptDisposition::rejected;
    attempt.before = state;
    attempt.after = state;
    attempt.audit = completeAudit(violationMask);
    return attempt;
}

HumanBehaviorAttempt acceptedAttempt(
    const std::uint64_t attemptIndex,
    const HumanBehaviorAcceptedState& before,
    const HumanBehaviorAcceptedState& after,
    const bool postureValid,
    const bool settled,
    const std::optional<double> forwardSpeed = std::nullopt,
    const std::uint32_t forbiddenContactCount = 0u,
    const std::uint32_t violationMask = 0u
) {
    HumanBehaviorAttempt attempt{};
    attempt.attemptIndex = attemptIndex;
    attempt.transactionFingerprint = 20'000u + attemptIndex;
    attempt.disposition = HumanBehaviorAttemptDisposition::accepted;
    attempt.before = before;
    attempt.after = after;
    attempt.audit = completeAudit(violationMask, forbiddenContactCount);
    attempt.acceptedMetrics = HumanBehaviorAcceptedMetrics{
        postureValid,
        settled,
        forwardSpeed,
    };
    return attempt;
}

HumanBehaviorTerminalCompletion terminalCompletion(
    const std::uint64_t attempts,
    const HumanBehaviorAcceptedState& finalState
) {
    HumanBehaviorTerminalCompletion terminal{};
    terminal.quiescent = true;
    terminal.exitCode = 0;
    terminal.completedAttemptCount = attempts;
    terminal.finalAcceptedState = finalState;
    return terminal;
}

void requireUnchanged(
    const HumanBehaviorTrialRecorder& recorder,
    const std::uint64_t acceptedSteps,
    const std::uint64_t attempts,
    const char* message
) {
    require(recorder.acceptedStepCount() == acceptedSteps &&
                recorder.completedAttemptCount() == attempts &&
                !recorder.finished(),
            message);
}

void exerciseDigestContract() {
    const auto original = digest(0x80u);
    const auto encoded = encodeHumanBehaviorDigest(original);
    require(encoded.size() == 64u, "digest encoder emitted a noncanonical size");
    HumanBehaviorDigest decoded{};
    std::string error;
    require(decodeHumanBehaviorDigest(encoded, decoded, error) &&
                decoded == original && error.empty(),
            "canonical digest did not round-trip");

    auto unchanged = digest(0x20u);
    const auto sentinel = unchanged;
    auto uppercase = encoded;
    uppercase[0] = 'A';
    require(!decodeHumanBehaviorDigest(uppercase, unchanged, error) &&
                unchanged == sentinel,
            "uppercase digest was admitted or mutated output");
    require(!decodeHumanBehaviorDigest(encoded.substr(1u), unchanged, error) &&
                unchanged == sentinel,
            "short digest was admitted or mutated output");
}

std::uint32_t exerciseDescriptorFailures() {
    std::uint32_t negativeCases = 0u;
    std::string error;
    const auto deny = [&](HumanBehaviorTrialDescriptor descriptor) {
        require(!HumanBehaviorTrialRecorder::create(descriptor, error),
                "invalid descriptor was admitted");
        require(!error.empty(), "descriptor denial omitted its reason");
        ++negativeCases;
    };

    {
        auto descriptor = standingDescriptor();
        descriptor.task = static_cast<HumanBehaviorTask>(99u);
        deny(descriptor);
    }
    {
        auto descriptor = standingDescriptor();
        descriptor.device = "Generic GPU";
        deny(descriptor);
    }
    {
        auto descriptor = standingDescriptor();
        descriptor.protocolSHA256.fill(0u);
        deny(descriptor);
    }
    {
        auto descriptor = standingDescriptor();
        descriptor.expectedAcceptedSteps = 0u;
        deny(descriptor);
    }
    {
        auto descriptor = standingDescriptor();
        descriptor.initialSettled = true;
        descriptor.initialPostureValid = false;
        deny(descriptor);
    }
    {
        auto descriptor = standingDescriptor();
        descriptor.task = HumanBehaviorTask::walking;
        deny(descriptor);
    }
    {
        auto descriptor = standingDescriptor();
        descriptor.task = HumanBehaviorTask::recovery;
        deny(descriptor);
    }
    {
        auto descriptor = standingDescriptor();
        descriptor.task = HumanBehaviorTask::recovery;
        descriptor.impulse = HumanBehaviorImpulse{"pelvis", {}, {}};
        deny(descriptor);
    }
    {
        auto descriptor = standingDescriptor();
        descriptor.initialAcceptedState.acceptedTimestampNanoseconds =
            std::numeric_limits<std::uint64_t>::max() - 10u;
        descriptor.stepNanoseconds = 8u;
        descriptor.expectedAcceptedSteps = 2u;
        deny(descriptor);
    }
    {
        auto descriptor = standingDescriptor();
        descriptor.initialAcceptedState.physicsGeneration =
            std::numeric_limits<std::uint64_t>::max() - 2u;
        deny(descriptor);
    }
    return negativeCases;
}

std::string exerciseStandingTrial(std::uint32_t& negativeCases) {
    const auto descriptor = standingDescriptor();
    std::string error;
    auto recorder = HumanBehaviorTrialRecorder::create(descriptor, error);
    require(recorder != nullptr && error.empty(), "valid standing trial denied");
    require(recorder->acceptedStepCount() == 0u &&
                recorder->completedAttemptCount() == 0u,
            "fresh recorder counters are not zero");

    const auto state0 = descriptor.initialAcceptedState;
    const auto state1 = successor(state0, 0x70u, descriptor.stepNanoseconds);
    const auto state2 = successor(state1, 0x90u, descriptor.stepNanoseconds);
    const auto state3 = successor(state2, 0xb0u, descriptor.stepNanoseconds);

    {
        auto attempt = rejectedAttempt(1u, state0);
        ++attempt.after.sensorGeneration;
        require(!recorder->recordAttempt(attempt, error),
                "rejected attempt published a sensor consequence");
        requireUnchanged(*recorder, 0u, 0u,
                         "failed no-consequence check mutated recorder");
        ++negativeCases;
    }
    require(recorder->recordAttempt(rejectedAttempt(1u, state0), error),
            "valid no-consequence rejection denied");

    {
        auto attempt = acceptedAttempt(2u, state0, state1, true, true);
        attempt.audit.coveredMask &= ~HumanBehaviorAuditUnexpectedReset;
        require(!recorder->recordAttempt(attempt, error),
                "incomplete audit coverage was admitted");
        requireUnchanged(*recorder, 0u, 1u,
                         "audit-coverage denial mutated recorder");
        ++negativeCases;
    }
    {
        auto attempt = acceptedAttempt(2u, state0, state1, true, true);
        attempt.audit.forbiddenContactCovered = false;
        require(!recorder->recordAttempt(attempt, error),
                "missing contact coverage was admitted");
        requireUnchanged(*recorder, 0u, 1u,
                         "contact-coverage denial mutated recorder");
        ++negativeCases;
    }
    {
        auto discontinuous = state1;
        ++discontinuous.acceptedTimestampNanoseconds;
        require(!recorder->recordAttempt(
                    acceptedAttempt(2u, state0, discontinuous, true, true),
                    error),
                "discontinuous exact clock was admitted");
        requireUnchanged(*recorder, 0u, 1u,
                         "clock denial mutated recorder");
        ++negativeCases;
    }
    {
        auto unchangedRoot = state1;
        unchangedRoot.acceptedRootSHA256 = state0.acceptedRootSHA256;
        require(!recorder->recordAttempt(
                    acceptedAttempt(2u, state0, unchangedRoot, true, true),
                    error),
                "unchanged accepted-root proof was admitted");
        requireUnchanged(*recorder, 0u, 1u,
                         "proof denial mutated recorder");
        ++negativeCases;
    }
    {
        auto regressed = state1;
        regressed.brainGeneration = state0.brainGeneration - 1u;
        require(!recorder->recordAttempt(
                    acceptedAttempt(2u, state0, regressed, true, true), error),
                "regressed causal generation was admitted");
        requireUnchanged(*recorder, 0u, 1u,
                         "generation denial mutated recorder");
        ++negativeCases;
    }
    {
        auto attempt = acceptedAttempt(2u, state0, state1, true, true, 0.5);
        require(!recorder->recordAttempt(attempt, error),
                "standing accepted a walking-only speed metric");
        requireUnchanged(*recorder, 0u, 1u,
                         "metric denial mutated recorder");
        ++negativeCases;
    }

    require(recorder->recordAttempt(
                acceptedAttempt(2u, state0, state1, true, true), error),
            "first accepted standing root denied");
    require(recorder->sealAcceptedSpan(error), "first span did not seal");
    require(!recorder->sealAcceptedSpan(error),
            "empty accepted span was admitted");
    ++negativeCases;

    require(recorder->recordAttempt(
                acceptedAttempt(3u, state1, state2, true, false), error),
            "second accepted standing root denied");
    require(recorder->recordAttempt(rejectedAttempt(
                4u, state2, HumanBehaviorAuditRootAssistance), error),
            "audited rejected standing attempt denied");
    require(recorder->recordAttempt(
                acceptedAttempt(5u, state2, state3, true, true), error),
            "third accepted standing root denied");
    require(recorder->sealAcceptedSpan(error), "second span did not seal");
    require(recorder->recordAttempt(rejectedAttempt(
                6u, state3, HumanBehaviorAuditDirectTorque), error),
            "trailing audited rejection denied");

    auto terminal = terminalCompletion(6u, state3);
    std::string output = "unchanged";
    {
        auto active = terminal;
        active.activeAttemptCount = 1u;
        require(!recorder->finish(active, output, error) &&
                    output == "unchanged",
                "nonquiescent finish was admitted or mutated output");
        requireUnchanged(*recorder, 3u, 6u,
                         "nonquiescent finish mutated recorder");
        ++negativeCases;
    }
    {
        auto wrong = terminal;
        wrong.completedAttemptCount = 5u;
        require(!recorder->finish(wrong, output, error) &&
                    output == "unchanged",
                "wrong terminal attempt count was admitted or mutated output");
        requireUnchanged(*recorder, 3u, 6u,
                         "terminal-count denial mutated recorder");
        ++negativeCases;
    }

    require(recorder->finish(terminal, output, error) && error.empty(),
            "valid quiescent standing completion denied");
    require(recorder->finished(), "successful finish did not close recorder");
    require(countOccurrences(output, "\n") == 4u,
            "standing trace did not contain header, two spans, and footer");
    require(countOccurrences(output, "\"kind\":\"completed\"") == 1u,
            "standing trace did not contain exactly one completion footer");
    requireContains(output, "\"schema\":\"numi.human.behavior-trial.v1\"",
                    "trace schema missing");
    requireContains(output, "\"step_ns\":12500",
                    "exact nanosecond step missing");
    requireContains(output,
        "\"first_step\":1,\"last_step\":1,\"start_ns\":0,\"end_ns\":12500,\"accepted_steps\":1",
        "first accepted span bounds are wrong");
    requireContains(output,
        "\"first_step\":2,\"last_step\":3,\"start_ns\":12500,\"end_ns\":37500,\"accepted_steps\":2",
        "second accepted span bounds are wrong");
    requireContains(output,
        "\"attempt_count\":4,\"rejected_attempt_count\":2",
        "trailing rejected attempt was not merged exactly once");
    requireContains(output, "\"root_assistance_steps\":1",
                    "root-assistance violation was lost");
    requireContains(output, "\"direct_torque_steps\":1",
                    "direct-torque violation was lost");
    requireContains(output, "\"audit_covered_attempt_count\":4",
                    "attempt audit coverage accounting is wrong");
    requireContains(output, "\"accepted_steps\":3,\"final_root_sha256\":",
                    "terminal completion record is incomplete");

    std::string repeated = "different";
    require(recorder->finish(terminal, repeated, error) && repeated == output,
            "identical finish was not byte-idempotent");
    require(!recorder->recordAttempt(rejectedAttempt(7u, state3), error),
            "completed trial accepted a new attempt");
    require(!recorder->sealAcceptedSpan(error),
            "completed trial accepted another span seal");
    auto differentTerminal = terminal;
    differentTerminal.exitCode = 1;
    repeated = "unchanged";
    require(!recorder->finish(differentTerminal, repeated, error) &&
                repeated == "unchanged",
            "different repeated finish was admitted or mutated output");
    negativeCases += 3u;
    return output;
}

void exerciseWalkingTrial(std::uint32_t& negativeCases) {
    auto descriptor = standingDescriptor();
    descriptor.trialID = "walking-core-check";
    descriptor.task = HumanBehaviorTask::walking;
    descriptor.targetSpeedMetersPerSecond = 0.5;
    descriptor.expectedAcceptedSteps = 2u;
    descriptor.initialSettled = false;
    std::string error;
    auto recorder = HumanBehaviorTrialRecorder::create(descriptor, error);
    require(recorder != nullptr, "valid walking descriptor denied");

    const auto state0 = descriptor.initialAcceptedState;
    const auto state1 = successor(state0, 0xc0u, descriptor.stepNanoseconds);
    const auto state2 = successor(state1, 0xe0u, descriptor.stepNanoseconds);
    {
        auto mismatch = acceptedAttempt(
            1u, state0, state1, true, false, 0.75);
        require(!recorder->recordAttempt(mismatch, error),
                "walking settled/posture mismatch was admitted");
        requireUnchanged(*recorder, 0u, 0u,
                         "walking metric denial mutated recorder");
        ++negativeCases;
    }
    {
        auto nonfinite = acceptedAttempt(
            1u, state0, state1, true, true,
            std::numeric_limits<double>::infinity());
        require(!recorder->recordAttempt(nonfinite, error),
                "nonfinite walking speed was admitted");
        requireUnchanged(*recorder, 0u, 0u,
                         "nonfinite speed denial mutated recorder");
        ++negativeCases;
    }
    require(recorder->recordAttempt(
                acceptedAttempt(1u, state0, state1, true, true, 0.75), error),
            "first walking root denied");
    require(recorder->recordAttempt(acceptedAttempt(
                2u, state1, state2, true, false, 0.25, 1u,
                HumanBehaviorAuditKinematicOverride), error),
            "second walking root denied");

    const auto terminal = terminalCompletion(2u, state2);
    std::string output;
    require(recorder->finish(terminal, output, error),
            "valid walking completion denied");
    requireContains(output, "\"target_speed_mps\":0.5",
                    "walking target was not frozen into header");
    requireContains(output, "\"posture_violation_steps\":1",
                    "forbidden contact did not invalidate walking posture");
    requireContains(output, "\"settled_steps\":1,\"settled_suffix_steps\":0",
                    "walking settled accounting is wrong");
    requireContains(output,
        "\"speed_squared_error_sum_m2_per_s2\":0.125,\"metric_sample_count\":2",
        "walking squared error was not computed from accepted samples");
    requireContains(output, "\"speed_error_sample_count\":2",
                    "walking speed coverage is incomplete");
    requireContains(output, "\"kinematic_override_steps\":1",
                    "accepted-attempt audit violation was erased");
}

void exerciseRecoveryTrial() {
    auto descriptor = standingDescriptor();
    descriptor.trialID = "recovery-core-check";
    descriptor.task = HumanBehaviorTask::recovery;
    descriptor.expectedAcceptedSteps = 1u;
    descriptor.initialSettled = false;
    descriptor.impulse = HumanBehaviorImpulse{
        "pelvis",
        {0.125, -0.25, 0.5},
        {12.0, 0.0, -4.0},
    };
    std::string error;
    auto recorder = HumanBehaviorTrialRecorder::create(descriptor, error);
    require(recorder != nullptr, "valid recovery descriptor denied");
    const auto state0 = descriptor.initialAcceptedState;
    const auto state1 = successor(state0, 0xf0u, descriptor.stepNanoseconds);
    require(recorder->recordAttempt(
                acceptedAttempt(1u, state0, state1, true, true), error),
            "accepted recovery root denied");
    std::string output;
    require(recorder->finish(terminalCompletion(1u, state1), output, error),
            "valid recovery completion denied");
    requireContains(output,
        "\"impulse\":{\"body_semantic_id\":\"pelvis\",\"frame\":\"world\",\"at_step\":0,\"point_m\":[0.125,-0.25,0.5],\"linear_impulse_ns\":[12,0,-4]}",
        "frozen recovery impulse is missing or inexact");
}

} // namespace

int main(const int argc, const char* const argv[]) {
    try {
        exerciseDigestContract();
        const auto descriptorNegativeCases = exerciseDescriptorFailures();
        std::uint32_t attemptNegativeCases = 0u;
        const auto standingTrace = exerciseStandingTrial(attemptNegativeCases);
        exerciseWalkingTrial(attemptNegativeCases);
        exerciseRecoveryTrial();
        if (argc == 2 && std::string(argv[1]) == "--emit") {
            std::fwrite(standingTrace.data(), 1u, standingTrace.size(), stdout);
        } else {
            require(argc == 1, "usage: human_behavior_trial_check [--emit]");
            std::printf(
                "human_behavior_trial=pass descriptor_negative=%u "
                "attempt_negative=%u accepted_roots=6 exact_ns=pass "
                "quiescent_footer=pass physical_steps=0\n",
                descriptorNegativeCases, attemptNegativeCases);
        }
        return 0;
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "human_behavior_trial=fail %s\n", exception.what());
        return 1;
    }
}
