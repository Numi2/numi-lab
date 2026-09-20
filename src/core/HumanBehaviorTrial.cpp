#include "metalrobo/HumanBehaviorTrial.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

constexpr std::array<const char*, 8> kAuditNames{{
    "root_assistance_steps",
    "direct_torque_steps",
    "kinematic_override_steps",
    "unregistered_force_steps",
    "source_constraint_omission_steps",
    "unaccepted_publications",
    "nonfinite_steps",
    "unexpected_reset_steps",
}};

[[nodiscard]] bool fail(std::string& error, const char* message) {
    error = message;
    return false;
}

[[nodiscard]] bool digestPresent(const HumanBehaviorDigest& digest) noexcept {
    return std::any_of(digest.begin(), digest.end(), [](std::uint8_t value) {
        return value != 0u;
    });
}

[[nodiscard]] bool visibleText(
    const std::string& text,
    const std::size_t maximumBytes
) noexcept {
    if (text.empty() || text.size() > maximumBytes || text.front() == ' ' ||
        text.back() == ' ') {
        return false;
    }
    return std::all_of(text.begin(), text.end(), [](const char value) {
        const auto byte = static_cast<unsigned char>(value);
        return byte >= 0x20u && byte <= 0x7eu;
    });
}

[[nodiscard]] bool finiteVector(const std::array<double, 3>& value) noexcept {
    return std::all_of(value.begin(), value.end(), [](double component) {
        return std::isfinite(component);
    });
}

[[nodiscard]] bool checkedAdd(
    const std::uint64_t left,
    const std::uint64_t right,
    std::uint64_t& output
) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) return false;
    output = left + right;
    return true;
}

[[nodiscard]] bool checkedMultiply(
    const std::uint64_t left,
    const std::uint64_t right,
    std::uint64_t& output
) noexcept {
    if (left != 0u &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    output = left * right;
    return true;
}

[[nodiscard]] const char* taskName(const HumanBehaviorTask task) noexcept {
    switch (task) {
        case HumanBehaviorTask::standing: return "standing";
        case HumanBehaviorTask::recovery: return "recovery";
        case HumanBehaviorTask::walking: return "walking";
    }
    return nullptr;
}

[[nodiscard]] bool validAcceptedState(
    const HumanBehaviorAcceptedState& state
) noexcept {
    return digestPresent(state.acceptedRootSHA256);
}

[[nodiscard]] bool validateDescriptor(
    const HumanBehaviorTrialDescriptor& descriptor,
    std::string& error
) {
    if (taskName(descriptor.task) == nullptr)
        return fail(error, "unsupported Human behavior task");
    if (!visibleText(descriptor.trialID, 256u) ||
        !visibleText(descriptor.executionID, 256u) ||
        !visibleText(descriptor.device, 256u) ||
        !visibleText(descriptor.osBuild, 256u)) {
        return fail(error, "trial descriptor text must be bounded visible ASCII");
    }
    if (!descriptor.device.starts_with("Apple "))
        return fail(error, "trial descriptor requires an Apple device identity");
    for (const auto* digest : {
             &descriptor.resetStateSHA256,
             &descriptor.protocolSHA256,
             &descriptor.stackSHA256,
             &descriptor.metricProgramSHA256,
             &descriptor.acceptedRootProofSchemaSHA256,
             &descriptor.taskLoweringReceiptSHA256,
             &descriptor.initialAcceptedState.acceptedRootSHA256}) {
        if (!digestPresent(*digest))
            return fail(error, "trial descriptor contains a zero digest");
    }
    if (descriptor.stepNanoseconds == 0u ||
        descriptor.expectedAcceptedSteps == 0u)
        return fail(error, "trial descriptor requires a positive exact horizon");
    std::uint64_t duration = 0u;
    std::uint64_t terminalTimestamp = 0u;
    if (!checkedMultiply(descriptor.stepNanoseconds,
                         descriptor.expectedAcceptedSteps, duration) ||
        !checkedAdd(descriptor.initialAcceptedState.acceptedTimestampNanoseconds,
                    duration, terminalTimestamp)) {
        return fail(error, "trial exact nanosecond horizon overflows uint64");
    }
    if (descriptor.initialAcceptedState.physicsGeneration >
        std::numeric_limits<std::uint64_t>::max() -
            descriptor.expectedAcceptedSteps) {
        return fail(error, "trial physics-generation horizon overflows uint64");
    }
    if (descriptor.initialSettled && !descriptor.initialPostureValid)
        return fail(error, "an initially settled trial must be posture-valid");

    const bool walking = descriptor.task == HumanBehaviorTask::walking;
    if (walking != descriptor.targetSpeedMetersPerSecond.has_value())
        return fail(error, "walking alone requires a frozen target speed");
    if (descriptor.targetSpeedMetersPerSecond.has_value() &&
        (!std::isfinite(*descriptor.targetSpeedMetersPerSecond) ||
         *descriptor.targetSpeedMetersPerSecond < 0.0)) {
        return fail(error, "target speed must be finite and nonnegative");
    }
    const bool recovery = descriptor.task == HumanBehaviorTask::recovery;
    if (recovery != descriptor.impulse.has_value())
        return fail(error, "recovery alone requires one frozen reset impulse");
    if (descriptor.impulse.has_value()) {
        const auto& impulse = *descriptor.impulse;
        if (!visibleText(impulse.bodySemanticID, 256u) ||
            !finiteVector(impulse.pointMeters) ||
            !finiteVector(impulse.linearImpulseNewtonSeconds) ||
            std::none_of(impulse.linearImpulseNewtonSeconds.begin(),
                         impulse.linearImpulseNewtonSeconds.end(),
                         [](double value) { return value != 0.0; })) {
            return fail(error, "recovery impulse is malformed or zero");
        }
    }
    error.clear();
    return true;
}

struct SpanAccumulator {
    HumanBehaviorAcceptedState startState{};
    std::uint64_t acceptedStepsBefore = 0u;
    std::uint64_t attemptCount = 0u;
    std::uint64_t acceptedSteps = 0u;
    std::uint64_t rejectedAttemptCount = 0u;
    std::uint64_t metricSampleCount = 0u;
    std::uint64_t auditCoveredRootCount = 0u;
    std::uint64_t auditCoveredAttemptCount = 0u;
    std::uint64_t postureViolationSteps = 0u;
    std::uint64_t settledSteps = 0u;
    std::uint64_t settledSuffixSteps = 0u;
    std::uint64_t speedErrorSampleCount = 0u;
    std::array<std::uint64_t, 8> auditViolationCounts{};
    double speedSquaredErrorSum = 0.0;
    std::vector<HumanBehaviorAttempt> attempts;
};

struct AcceptedSpan {
    std::uint64_t firstStep = 0u;
    std::uint64_t lastStep = 0u;
    std::uint64_t startNanoseconds = 0u;
    std::uint64_t endNanoseconds = 0u;
    HumanBehaviorDigest startRootSHA256{};
    HumanBehaviorDigest endRootSHA256{};
    std::uint64_t acceptedSteps = 0u;
    std::uint64_t attemptCount = 0u;
    std::uint64_t rejectedAttemptCount = 0u;
    std::uint64_t metricSampleCount = 0u;
    std::uint64_t auditCoveredRootCount = 0u;
    std::uint64_t auditCoveredAttemptCount = 0u;
    std::uint64_t postureViolationSteps = 0u;
    std::uint64_t settledSteps = 0u;
    std::uint64_t settledSuffixSteps = 0u;
    std::uint64_t speedErrorSampleCount = 0u;
    std::array<std::uint64_t, 8> auditViolationCounts{};
    double speedSquaredErrorSum = 0.0;
    std::vector<HumanBehaviorAttempt> attempts;
};

[[nodiscard]] AcceptedSpan makeSpan(
    const SpanAccumulator& source,
    const HumanBehaviorAcceptedState& endState,
    const std::uint64_t stepNanoseconds
) {
    AcceptedSpan span{};
    span.firstStep = source.acceptedStepsBefore + 1u;
    span.lastStep = source.acceptedStepsBefore + source.acceptedSteps;
    span.startNanoseconds = source.acceptedStepsBefore * stepNanoseconds;
    span.endNanoseconds = span.lastStep * stepNanoseconds;
    span.startRootSHA256 = source.startState.acceptedRootSHA256;
    span.endRootSHA256 = endState.acceptedRootSHA256;
    span.acceptedSteps = source.acceptedSteps;
    span.attemptCount = source.attemptCount;
    span.rejectedAttemptCount = source.rejectedAttemptCount;
    span.metricSampleCount = source.metricSampleCount;
    span.auditCoveredRootCount = source.auditCoveredRootCount;
    span.auditCoveredAttemptCount = source.auditCoveredAttemptCount;
    span.postureViolationSteps = source.postureViolationSteps;
    span.settledSteps = source.settledSteps;
    span.settledSuffixSteps = source.settledSuffixSteps;
    span.speedErrorSampleCount = source.speedErrorSampleCount;
    span.auditViolationCounts = source.auditViolationCounts;
    span.speedSquaredErrorSum = source.speedSquaredErrorSum;
    span.attempts = source.attempts;
    return span;
}

[[nodiscard]] bool mergeRejectedTail(
    AcceptedSpan& destination,
    const SpanAccumulator& tail,
    std::string& error
) {
    if (tail.acceptedSteps != 0u || tail.metricSampleCount != 0u ||
        tail.auditCoveredRootCount != 0u || tail.postureViolationSteps != 0u ||
        tail.settledSteps != 0u || tail.settledSuffixSteps != 0u ||
        tail.speedErrorSampleCount != 0u || tail.speedSquaredErrorSum != 0.0 ||
        tail.attemptCount != tail.rejectedAttemptCount) {
        return fail(error, "invalid rejected-only trailing span");
    }
    AcceptedSpan merged = destination;
    if (!checkedAdd(merged.attemptCount, tail.attemptCount,
                    merged.attemptCount) ||
        !checkedAdd(merged.rejectedAttemptCount, tail.rejectedAttemptCount,
                    merged.rejectedAttemptCount) ||
        !checkedAdd(merged.auditCoveredAttemptCount,
                    tail.auditCoveredAttemptCount,
                    merged.auditCoveredAttemptCount)) {
        return fail(error, "trailing attempt counters overflow uint64");
    }
    for (std::size_t index = 0u;
         index < merged.auditViolationCounts.size(); ++index) {
        if (!checkedAdd(merged.auditViolationCounts[index],
                        tail.auditViolationCounts[index],
                        merged.auditViolationCounts[index])) {
            return fail(error, "trailing audit counters overflow uint64");
        }
    }
    merged.attempts.insert(
        merged.attempts.end(), tail.attempts.begin(), tail.attempts.end());
    destination = merged;
    return true;
}

void appendQuoted(std::string& output, const std::string& value) {
    output.push_back('"');
    for (const char character : value) {
        if (character == '"' || character == '\\') output.push_back('\\');
        output.push_back(character);
    }
    output.push_back('"');
}

void appendDigest(std::string& output, const HumanBehaviorDigest& digest) {
    appendQuoted(output, encodeHumanBehaviorDigest(digest));
}

void appendUnsigned(std::string& output, const std::uint64_t value) {
    output += std::to_string(value);
}

void appendAcceptedState(
    std::string& output,
    const HumanBehaviorAcceptedState& state
) {
    output += "{\"accepted_root_sha256\":";
    appendDigest(output, state.acceptedRootSHA256);
    output += ",\"accepted_timestamp_ns\":";
    appendUnsigned(output, state.acceptedTimestampNanoseconds);
    output += ",\"physics_generation\":";
    appendUnsigned(output, state.physicsGeneration);
    output += ",\"brain_generation\":";
    appendUnsigned(output, state.brainGeneration);
    output += ",\"sensor_generation\":";
    appendUnsigned(output, state.sensorGeneration);
    output += ",\"controller_generation\":";
    appendUnsigned(output, state.controllerGeneration);
    output += ",\"task_generation\":";
    appendUnsigned(output, state.taskGeneration);
    output += ",\"random_generation\":";
    appendUnsigned(output, state.randomGeneration);
    output.push_back('}');
}

void appendDouble(std::string& output, const double value) {
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite JSON number");
    if (value == 0.0) {
        output.push_back('0');
        return;
    }
    std::array<char, 64> buffer{};
    const auto encoded = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value,
        std::chars_format::general, std::numeric_limits<double>::max_digits10);
    if (encoded.ec != std::errc{})
        throw std::runtime_error("failed to encode JSON number");
    output.append(buffer.data(), encoded.ptr);
}

void appendVector(std::string& output, const std::array<double, 3>& value) {
    output.push_back('[');
    for (std::size_t index = 0u; index < value.size(); ++index) {
        if (index != 0u) output.push_back(',');
        appendDouble(output, value[index]);
    }
    output.push_back(']');
}

void appendAttempt(
    std::string& output,
    const HumanBehaviorAttempt& attempt
) {
    output += "{\"attempt_index\":";
    appendUnsigned(output, attempt.attemptIndex);
    output += ",\"transaction_fingerprint\":";
    appendUnsigned(output, attempt.transactionFingerprint);
    output += ",\"disposition\":";
    appendQuoted(
        output,
        attempt.disposition == HumanBehaviorAttemptDisposition::accepted
            ? "accepted" : "rejected");
    output += ",\"before\":";
    appendAcceptedState(output, attempt.before);
    output += ",\"after\":";
    appendAcceptedState(output, attempt.after);
    output += ",\"audit\":{\"covered_mask\":";
    appendUnsigned(output, attempt.audit.coveredMask);
    output += ",\"violation_mask\":";
    appendUnsigned(output, attempt.audit.violationMask);
    output += ",\"forbidden_contact_covered\":";
    output += attempt.audit.forbiddenContactCovered ? "true" : "false";
    output += ",\"forbidden_contact_count\":";
    appendUnsigned(output, attempt.audit.forbiddenContactCount);
    output += "},\"metrics\":";
    if (attempt.acceptedMetrics.has_value()) {
        output += "{\"posture_valid\":";
        output += attempt.acceptedMetrics->postureValid ? "true" : "false";
        output += ",\"settled\":";
        output += attempt.acceptedMetrics->settled ? "true" : "false";
        output += ",\"forward_speed_mps\":";
        if (attempt.acceptedMetrics->forwardSpeedMetersPerSecond.has_value()) {
            appendDouble(
                output,
                *attempt.acceptedMetrics->forwardSpeedMetersPerSecond);
        } else {
            output += "null";
        }
        output.push_back('}');
    } else {
        output += "null";
    }
    output.push_back('}');
}

[[nodiscard]] std::string serializeHeader(
    const HumanBehaviorTrialDescriptor& descriptor
) {
    std::string output;
    output.reserve(1400u);
    output += "{\"schema\":\"numi.human.behavior-trial.v1\",\"trial_id\":";
    appendQuoted(output, descriptor.trialID);
    output += ",\"task\":";
    appendQuoted(output, taskName(descriptor.task));
    output += ",\"seed\":";
    appendUnsigned(output, descriptor.seed);
    output += ",\"reset_state_sha256\":";
    appendDigest(output, descriptor.resetStateSHA256);
    output += ",\"target_speed_mps\":";
    if (descriptor.targetSpeedMetersPerSecond.has_value())
        appendDouble(output, *descriptor.targetSpeedMetersPerSecond);
    else
        output += "null";
    output += ",\"impulse\":";
    if (descriptor.impulse.has_value()) {
        output += "{\"body_semantic_id\":";
        appendQuoted(output, descriptor.impulse->bodySemanticID);
        output += ",\"frame\":\"world\",\"at_step\":0,\"point_m\":";
        appendVector(output, descriptor.impulse->pointMeters);
        output += ",\"linear_impulse_ns\":";
        appendVector(output, descriptor.impulse->linearImpulseNewtonSeconds);
        output.push_back('}');
    } else {
        output += "null";
    }
    output += ",\"protocol_sha256\":";
    appendDigest(output, descriptor.protocolSHA256);
    output += ",\"stack_sha256\":";
    appendDigest(output, descriptor.stackSHA256);
    output += ",\"device\":";
    appendQuoted(output, descriptor.device);
    output += ",\"metric_contract\":\"accepted-root-task-reduction.v1\",\"execution_id\":";
    appendQuoted(output, descriptor.executionID);
    output += ",\"initial_root_sha256\":";
    appendDigest(output, descriptor.initialAcceptedState.acceptedRootSHA256);
    output += ",\"expected_accepted_steps\":";
    appendUnsigned(output, descriptor.expectedAcceptedSteps);
    output += ",\"initial_accepted_state\":";
    appendAcceptedState(output, descriptor.initialAcceptedState);
    output += ",\"initial_posture_valid\":";
    output += descriptor.initialPostureValid ? "true" : "false";
    output += ",\"initial_settled\":";
    output += descriptor.initialSettled ? "true" : "false";
    output += ",\"step_ns\":";
    appendUnsigned(output, descriptor.stepNanoseconds);
    output += ",\"os_build\":";
    appendQuoted(output, descriptor.osBuild);
    output += ",\"metric_program_sha256\":";
    appendDigest(output, descriptor.metricProgramSHA256);
    output += ",\"accepted_root_proof_schema_sha256\":";
    appendDigest(output, descriptor.acceptedRootProofSchemaSHA256);
    output += ",\"task_lowering_receipt_sha256\":";
    appendDigest(output, descriptor.taskLoweringReceiptSHA256);
    output += "}\n";
    return output;
}

[[nodiscard]] std::string serializeSpan(const AcceptedSpan& span) {
    std::string output;
    output.reserve(1000u);
    output += "{\"kind\":\"accepted_span\",\"first_step\":";
    appendUnsigned(output, span.firstStep);
    output += ",\"last_step\":";
    appendUnsigned(output, span.lastStep);
    output += ",\"start_ns\":";
    appendUnsigned(output, span.startNanoseconds);
    output += ",\"end_ns\":";
    appendUnsigned(output, span.endNanoseconds);
    output += ",\"accepted_steps\":";
    appendUnsigned(output, span.acceptedSteps);
    output += ",\"start_root_sha256\":";
    appendDigest(output, span.startRootSHA256);
    output += ",\"end_root_sha256\":";
    appendDigest(output, span.endRootSHA256);
    output += ",\"attempt_count\":";
    appendUnsigned(output, span.attemptCount);
    output += ",\"rejected_attempt_count\":";
    appendUnsigned(output, span.rejectedAttemptCount);
    output += ",\"audit\":{";
    for (std::size_t index = 0u; index < kAuditNames.size(); ++index) {
        if (index != 0u) output.push_back(',');
        appendQuoted(output, kAuditNames[index]);
        output.push_back(':');
        appendUnsigned(output, span.auditViolationCounts[index]);
    }
    output += "},\"posture_violation_steps\":";
    appendUnsigned(output, span.postureViolationSteps);
    output += ",\"settled_steps\":";
    appendUnsigned(output, span.settledSteps);
    output += ",\"settled_suffix_steps\":";
    appendUnsigned(output, span.settledSuffixSteps);
    output += ",\"speed_squared_error_sum_m2_per_s2\":";
    appendDouble(output, span.speedSquaredErrorSum);
    output += ",\"metric_sample_count\":";
    appendUnsigned(output, span.metricSampleCount);
    output += ",\"audit_covered_root_count\":";
    appendUnsigned(output, span.auditCoveredRootCount);
    output += ",\"audit_covered_attempt_count\":";
    appendUnsigned(output, span.auditCoveredAttemptCount);
    output += ",\"speed_error_sample_count\":";
    appendUnsigned(output, span.speedErrorSampleCount);
    output += ",\"attempts\":[";
    for (std::size_t index = 0u; index < span.attempts.size(); ++index) {
        if (index != 0u) output.push_back(',');
        appendAttempt(output, span.attempts[index]);
    }
    output.push_back(']');
    output += "}\n";
    return output;
}

[[nodiscard]] std::string serializeFooter(
    const std::uint64_t acceptedSteps,
    const std::uint64_t completedAttempts,
    const HumanBehaviorAcceptedState& finalState
) {
    std::string output = "{\"kind\":\"completed\",\"accepted_steps\":";
    appendUnsigned(output, acceptedSteps);
    output += ",\"completed_attempt_count\":";
    appendUnsigned(output, completedAttempts);
    output += ",\"final_root_sha256\":";
    appendDigest(output, finalState.acceptedRootSHA256);
    output += ",\"final_accepted_state\":";
    appendAcceptedState(output, finalState);
    output += ",\"exit_code\":0}\n";
    return output;
}

} // namespace

struct HumanBehaviorTrialRecorder::Impl {
    explicit Impl(const HumanBehaviorTrialDescriptor& value)
        : descriptor(value), currentState(value.initialAcceptedState) {
        pending.startState = currentState;
    }

    HumanBehaviorTrialDescriptor descriptor;
    HumanBehaviorAcceptedState currentState{};
    std::uint64_t acceptedSteps = 0u;
    std::uint64_t completedAttempts = 0u;
    SpanAccumulator pending{};
    std::vector<AcceptedSpan> spans;
    bool isFinished = false;
    std::optional<HumanBehaviorTerminalCompletion> terminal;
    std::string completedJSONLines;
};

bool decodeHumanBehaviorDigest(
    const std::string_view lowerHex,
    HumanBehaviorDigest& output,
    std::string& error
) {
    if (lowerHex.size() != 64u)
        return fail(error, "digest must contain exactly 64 lowercase hex digits");
    HumanBehaviorDigest decoded{};
    const auto nibble = [](const char value, std::uint8_t& result) {
        if (value >= '0' && value <= '9') {
            result = static_cast<std::uint8_t>(value - '0');
            return true;
        }
        if (value >= 'a' && value <= 'f') {
            result = static_cast<std::uint8_t>(10 + value - 'a');
            return true;
        }
        return false;
    };
    for (std::size_t index = 0u; index < decoded.size(); ++index) {
        std::uint8_t high = 0u;
        std::uint8_t low = 0u;
        if (!nibble(lowerHex[2u * index], high) ||
            !nibble(lowerHex[2u * index + 1u], low)) {
            return fail(error, "digest must contain exactly 64 lowercase hex digits");
        }
        decoded[index] = static_cast<std::uint8_t>((high << 4u) | low);
    }
    output = decoded;
    error.clear();
    return true;
}

std::string encodeHumanBehaviorDigest(const HumanBehaviorDigest& digest) {
    constexpr char hexadecimal[] = "0123456789abcdef";
    std::string output;
    output.resize(64u);
    for (std::size_t index = 0u; index < digest.size(); ++index) {
        output[2u * index] = hexadecimal[digest[index] >> 4u];
        output[2u * index + 1u] = hexadecimal[digest[index] & 0x0fu];
    }
    return output;
}

HumanBehaviorTrialRecorder::HumanBehaviorTrialRecorder(
    std::unique_ptr<Impl> impl
) noexcept : impl_(std::move(impl)) {}

HumanBehaviorTrialRecorder::~HumanBehaviorTrialRecorder() = default;

std::unique_ptr<HumanBehaviorTrialRecorder>
HumanBehaviorTrialRecorder::create(
    const HumanBehaviorTrialDescriptor& descriptor,
    std::string& error
) {
    try {
        if (!validateDescriptor(descriptor, error)) return nullptr;
        auto impl = std::make_unique<Impl>(descriptor);
        auto result = std::unique_ptr<HumanBehaviorTrialRecorder>(
            new HumanBehaviorTrialRecorder(std::move(impl)));
        error.clear();
        return result;
    } catch (const std::exception& exception) {
        error = exception.what();
        return nullptr;
    }
}

bool HumanBehaviorTrialRecorder::recordAttempt(
    const HumanBehaviorAttempt& attempt,
    std::string& error
) {
    try {
        auto& state = *impl_;
        if (state.isFinished)
            return fail(error, "completed trial cannot accept another attempt");
        if (state.completedAttempts ==
            std::numeric_limits<std::uint64_t>::max())
            return fail(error, "attempt counter overflows uint64");
        if (attempt.attemptIndex != state.completedAttempts + 1u ||
            attempt.transactionFingerprint == 0u)
            return fail(error, "attempt identity is missing or noncontiguous");
        if (!(attempt.before == state.currentState))
            return fail(error, "attempt does not start from the accepted root");
        if (attempt.audit.coveredMask !=
                kHumanBehaviorCompleteAuditCoverageMask ||
            (attempt.audit.violationMask & ~attempt.audit.coveredMask) != 0u)
            return fail(error, "attempt audit coverage is incomplete or inconsistent");

        SpanAccumulator next = state.pending;
        if (!checkedAdd(next.attemptCount, 1u, next.attemptCount) ||
            !checkedAdd(next.auditCoveredAttemptCount, 1u,
                        next.auditCoveredAttemptCount))
            return fail(error, "attempt aggregate overflows uint64");
        for (std::size_t bit = 0u; bit < 8u; ++bit) {
            if ((attempt.audit.violationMask & (1u << bit)) != 0u &&
                !checkedAdd(next.auditViolationCounts[bit], 1u,
                            next.auditViolationCounts[bit]))
                return fail(error, "attempt audit aggregate overflows uint64");
        }

        HumanBehaviorAcceptedState nextState = state.currentState;
        std::uint64_t nextAcceptedSteps = state.acceptedSteps;
        switch (attempt.disposition) {
            case HumanBehaviorAttemptDisposition::rejected: {
                if (attempt.acceptedMetrics.has_value() ||
                    !(attempt.after == attempt.before) ||
                    attempt.audit.forbiddenContactCount != 0u)
                    return fail(error, "rejected attempt published a consequence");
                if (!checkedAdd(next.rejectedAttemptCount, 1u,
                                next.rejectedAttemptCount))
                    return fail(error, "rejected-attempt aggregate overflows uint64");
                break;
            }
            case HumanBehaviorAttemptDisposition::accepted: {
                if (!attempt.acceptedMetrics.has_value() ||
                    !attempt.audit.forbiddenContactCovered)
                    return fail(error, "accepted attempt lacks metric or contact coverage");
                if (state.acceptedSteps >= state.descriptor.expectedAcceptedSteps)
                    return fail(error, "accepted attempt exceeds the frozen horizon");
                if (!validAcceptedState(attempt.after) ||
                    attempt.after.acceptedRootSHA256 ==
                        attempt.before.acceptedRootSHA256)
                    return fail(error, "accepted root proof is absent or unchanged");
                std::uint64_t expectedTimestamp = 0u;
                if (!checkedAdd(attempt.before.acceptedTimestampNanoseconds,
                                state.descriptor.stepNanoseconds,
                                expectedTimestamp) ||
                    attempt.after.acceptedTimestampNanoseconds !=
                        expectedTimestamp)
                    return fail(error, "accepted root exact nanosecond clock is discontinuous");
                if (attempt.before.physicsGeneration ==
                        std::numeric_limits<std::uint64_t>::max() ||
                    attempt.after.physicsGeneration !=
                        attempt.before.physicsGeneration + 1u)
                    return fail(error, "accepted physics generation is discontinuous");
                if (attempt.after.brainGeneration < attempt.before.brainGeneration ||
                    attempt.after.sensorGeneration < attempt.before.sensorGeneration ||
                    attempt.after.controllerGeneration <
                        attempt.before.controllerGeneration ||
                    attempt.after.taskGeneration < attempt.before.taskGeneration ||
                    attempt.after.randomGeneration < attempt.before.randomGeneration)
                    return fail(error, "accepted causal generation regressed");

                const auto& metrics = *attempt.acceptedMetrics;
                const bool postureValid = metrics.postureValid &&
                    attempt.audit.forbiddenContactCount == 0u;
                if (metrics.settled && !postureValid)
                    return fail(error, "settled metric contradicts posture/contact state");
                const bool walking = state.descriptor.task ==
                    HumanBehaviorTask::walking;
                if (walking != metrics.forwardSpeedMetersPerSecond.has_value())
                    return fail(error, "walking alone requires a forward-speed sample");
                if (walking && metrics.settled != postureValid)
                    return fail(error, "walking settled metric must equal valid posture/contact state");
                double squaredError = 0.0;
                if (walking) {
                    const double speed = *metrics.forwardSpeedMetersPerSecond;
                    if (!std::isfinite(speed))
                        return fail(error, "walking speed sample is nonfinite");
                    const double difference = speed -
                        *state.descriptor.targetSpeedMetersPerSecond;
                    squaredError = difference * difference;
                    if (!std::isfinite(squaredError) ||
                        !std::isfinite(next.speedSquaredErrorSum + squaredError))
                        return fail(error, "walking squared-error aggregate is nonfinite");
                }
                for (auto* counter : {
                         &next.acceptedSteps,
                         &next.metricSampleCount,
                         &next.auditCoveredRootCount}) {
                    if (!checkedAdd(*counter, 1u, *counter))
                        return fail(error, "accepted metric aggregate overflows uint64");
                }
                if (!postureValid &&
                    !checkedAdd(next.postureViolationSteps, 1u,
                                next.postureViolationSteps))
                    return fail(error, "posture aggregate overflows uint64");
                if (metrics.settled) {
                    if (!checkedAdd(next.settledSteps, 1u, next.settledSteps) ||
                        !checkedAdd(next.settledSuffixSteps, 1u,
                                    next.settledSuffixSteps))
                        return fail(error, "settled aggregate overflows uint64");
                } else {
                    next.settledSuffixSteps = 0u;
                }
                if (walking) {
                    if (!checkedAdd(next.speedErrorSampleCount, 1u,
                                    next.speedErrorSampleCount))
                        return fail(error, "speed-sample aggregate overflows uint64");
                    next.speedSquaredErrorSum += squaredError;
                }
                if (!checkedAdd(state.acceptedSteps, 1u, nextAcceptedSteps))
                    return fail(error, "accepted-root counter overflows uint64");
                nextState = attempt.after;
                break;
            }
            default:
                return fail(error, "unsupported attempt disposition");
        }

        next.attempts.push_back(attempt);

        state.pending = std::move(next);
        state.currentState = nextState;
        state.acceptedSteps = nextAcceptedSteps;
        ++state.completedAttempts;
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

bool HumanBehaviorTrialRecorder::sealAcceptedSpan(std::string& error) {
    try {
        auto& state = *impl_;
        if (state.isFinished)
            return fail(error, "completed trial cannot seal another span");
        if (state.pending.acceptedSteps == 0u)
            return fail(error, "accepted span requires at least one accepted root");
        const auto span = makeSpan(
            state.pending, state.currentState,
            state.descriptor.stepNanoseconds);
        state.spans.push_back(span);
        state.pending = {};
        state.pending.startState = state.currentState;
        state.pending.acceptedStepsBefore = state.acceptedSteps;
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

bool HumanBehaviorTrialRecorder::finish(
    const HumanBehaviorTerminalCompletion& terminal,
    std::string& jsonLines,
    std::string& error
) {
    try {
        auto& state = *impl_;
        if (state.isFinished) {
            if (!state.terminal.has_value() || *state.terminal != terminal)
                return fail(error, "completed trial received a different terminal record");
            jsonLines = state.completedJSONLines;
            error.clear();
            return true;
        }
        if (!terminal.quiescent || terminal.activeAttemptCount != 0u ||
            terminal.pendingPublicationCount != 0u)
            return fail(error, "trial finish requires a quiescent runtime");
        if (terminal.exitCode != 0)
            return fail(error, "successful completion footer requires exit code zero");
        if (terminal.completedAttemptCount != state.completedAttempts)
            return fail(error, "terminal attempt count differs from recorder");
        if (!(terminal.finalAcceptedState == state.currentState))
            return fail(error, "terminal accepted state differs from recorder");
        if (state.acceptedSteps != state.descriptor.expectedAcceptedSteps)
            return fail(error, "trial has not completed its frozen accepted horizon");

        std::uint64_t expectedDuration = 0u;
        std::uint64_t expectedTimestamp = 0u;
        if (!checkedMultiply(state.descriptor.stepNanoseconds,
                             state.acceptedSteps, expectedDuration) ||
            !checkedAdd(state.descriptor.initialAcceptedState.
                            acceptedTimestampNanoseconds,
                        expectedDuration, expectedTimestamp) ||
            state.currentState.acceptedTimestampNanoseconds !=
                expectedTimestamp)
            return fail(error, "terminal exact nanosecond clock is inconsistent");

        auto spans = state.spans;
        if (state.pending.acceptedSteps != 0u) {
            spans.push_back(makeSpan(
                state.pending, state.currentState,
                state.descriptor.stepNanoseconds));
        } else if (state.pending.attemptCount != 0u) {
            if (spans.empty())
                return fail(error, "rejected-only trial cannot complete");
            if (!mergeRejectedTail(spans.back(), state.pending, error))
                return false;
        }
        if (spans.empty())
            return fail(error, "completed trial contains no accepted span");

        std::uint64_t spanAccepted = 0u;
        std::uint64_t spanAttempts = 0u;
        HumanBehaviorDigest prior =
            state.descriptor.initialAcceptedState.acceptedRootSHA256;
        for (const auto& span : spans) {
            if (span.firstStep != spanAccepted + 1u ||
                span.lastStep != spanAccepted + span.acceptedSteps ||
                span.startNanoseconds !=
                    spanAccepted * state.descriptor.stepNanoseconds ||
                span.endNanoseconds !=
                    span.lastStep * state.descriptor.stepNanoseconds ||
                span.startRootSHA256 != prior ||
                span.attemptCount !=
                    span.acceptedSteps + span.rejectedAttemptCount ||
                span.metricSampleCount != span.acceptedSteps ||
                span.auditCoveredRootCount != span.acceptedSteps ||
                span.auditCoveredAttemptCount != span.attemptCount ||
                span.settledSteps > span.acceptedSteps ||
                span.settledSuffixSteps > span.settledSteps ||
                span.postureViolationSteps > span.acceptedSteps ||
                span.speedErrorSampleCount !=
                    (state.descriptor.task == HumanBehaviorTask::walking
                         ? span.acceptedSteps : 0u) ||
                span.attempts.size() != span.attemptCount)
                return fail(error, "accepted span accounting is inconsistent");
            if (!checkedAdd(spanAccepted, span.acceptedSteps, spanAccepted) ||
                !checkedAdd(spanAttempts, span.attemptCount, spanAttempts))
                return fail(error, "accepted span totals overflow uint64");
            prior = span.endRootSHA256;
        }
        if (spanAccepted != state.acceptedSteps ||
            spanAttempts != state.completedAttempts ||
            prior != state.currentState.acceptedRootSHA256)
            return fail(error, "accepted span chain does not close the trial");

        std::string document = serializeHeader(state.descriptor);
        for (const auto& span : spans) document += serializeSpan(span);
        document += serializeFooter(
            state.acceptedSteps,
            state.completedAttempts,
            state.currentState);
        // Complete every allocation that can fail before closing the recorder.
        // The following standard-allocator moves are noexcept, which preserves
        // the public failed-call/no-mutation guarantee even under allocation
        // failure while preparing the caller's copy.
        std::string callerOutput = document;

        state.spans = std::move(spans);
        state.pending = {};
        state.pending.startState = state.currentState;
        state.pending.acceptedStepsBefore = state.acceptedSteps;
        state.terminal = terminal;
        state.completedJSONLines = std::move(document);
        state.isFinished = true;
        jsonLines = std::move(callerOutput);
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

std::uint64_t HumanBehaviorTrialRecorder::acceptedStepCount() const noexcept {
    return impl_->acceptedSteps;
}

std::uint64_t HumanBehaviorTrialRecorder::completedAttemptCount() const noexcept {
    return impl_->completedAttempts;
}

bool HumanBehaviorTrialRecorder::finished() const noexcept {
    return impl_->isFinished;
}

} // namespace metalrobo
