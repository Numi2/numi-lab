#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace metalrobo {

// A digest supplied by the accepted-root proof owner. The recorder preserves
// and chains this SHA-256 identity; it does not reconstruct physical state or
// authenticate a producer from transaction metadata.
using HumanBehaviorDigest = std::array<std::uint8_t, 32>;

[[nodiscard]] bool decodeHumanBehaviorDigest(
    std::string_view lowerHex,
    HumanBehaviorDigest& output,
    std::string& error
);

[[nodiscard]] std::string encodeHumanBehaviorDigest(
    const HumanBehaviorDigest& digest
);

enum class HumanBehaviorTask : std::uint32_t {
    standing = 0u,
    recovery = 1u,
    walking = 2u,
};

enum HumanBehaviorAuditBit : std::uint32_t {
    HumanBehaviorAuditRootAssistance = 1u << 0u,
    HumanBehaviorAuditDirectTorque = 1u << 1u,
    HumanBehaviorAuditKinematicOverride = 1u << 2u,
    HumanBehaviorAuditUnregisteredForce = 1u << 3u,
    HumanBehaviorAuditSourceConstraintOmission = 1u << 4u,
    HumanBehaviorAuditUnacceptedPublication = 1u << 5u,
    HumanBehaviorAuditNonfinite = 1u << 6u,
    HumanBehaviorAuditUnexpectedReset = 1u << 7u,
};

inline constexpr std::uint32_t kHumanBehaviorCompleteAuditCoverageMask =
    HumanBehaviorAuditRootAssistance |
    HumanBehaviorAuditDirectTorque |
    HumanBehaviorAuditKinematicOverride |
    HumanBehaviorAuditUnregisteredForce |
    HumanBehaviorAuditSourceConstraintOmission |
    HumanBehaviorAuditUnacceptedPublication |
    HumanBehaviorAuditNonfinite |
    HumanBehaviorAuditUnexpectedReset;

struct HumanBehaviorImpulse {
    std::string bodySemanticID;
    std::array<double, 3> pointMeters{};
    std::array<double, 3> linearImpulseNewtonSeconds{};
};

// Identity of one complete accepted transaction state. acceptedRootSHA256 must
// be computed by the versioned proof owner over physical, Brain, sensor,
// controller, task, random, and checkpoint state. The generation fields make
// rejection/no-consequence checks explicit instead of trusting the digest
// alone.
struct HumanBehaviorAcceptedState {
    HumanBehaviorDigest acceptedRootSHA256{};
    std::uint64_t acceptedTimestampNanoseconds = 0u;
    std::uint64_t physicsGeneration = 0u;
    std::uint64_t brainGeneration = 0u;
    std::uint64_t sensorGeneration = 0u;
    std::uint64_t controllerGeneration = 0u;
    std::uint64_t taskGeneration = 0u;
    std::uint64_t randomGeneration = 0u;

    [[nodiscard]] bool operator==(
        const HumanBehaviorAcceptedState& other
    ) const noexcept = default;
};

struct HumanBehaviorTrialDescriptor {
    std::string trialID;
    HumanBehaviorTask task = HumanBehaviorTask::standing;
    std::uint64_t seed = 0u;
    HumanBehaviorDigest resetStateSHA256{};
    std::optional<double> targetSpeedMetersPerSecond;
    std::optional<HumanBehaviorImpulse> impulse;

    HumanBehaviorDigest protocolSHA256{};
    HumanBehaviorDigest stackSHA256{};
    HumanBehaviorDigest metricProgramSHA256{};
    HumanBehaviorDigest acceptedRootProofSchemaSHA256{};
    HumanBehaviorDigest taskLoweringReceiptSHA256{};
    std::string device;
    std::string osBuild;
    std::string executionID;

    std::uint64_t stepNanoseconds = 0u;
    std::uint64_t expectedAcceptedSteps = 0u;
    HumanBehaviorAcceptedState initialAcceptedState{};
    bool initialPostureValid = false;
    bool initialSettled = false;
};

// Coverage and violations come from independent native owners. A zero mask is
// unknown, never evidence of zero violations. Forbidden-contact coverage is
// required for accepted metric samples and is kept separate from the eight
// attempt-wide audits.
struct HumanBehaviorAttemptAudit {
    std::uint32_t coveredMask = 0u;
    std::uint32_t violationMask = 0u;
    bool forbiddenContactCovered = false;
    std::uint32_t forbiddenContactCount = 0u;
};

struct HumanBehaviorAcceptedMetrics {
    bool postureValid = false;
    bool settled = false;
    // Required only for walking. The recorder computes squared error from the
    // frozen target; callers cannot inject a pre-aggregated error sum.
    std::optional<double> forwardSpeedMetersPerSecond;
};

enum class HumanBehaviorAttemptDisposition : std::uint32_t {
    accepted = 1u,
    rejected = 2u,
};

struct HumanBehaviorAttempt {
    std::uint64_t attemptIndex = 0u;
    std::uint64_t transactionFingerprint = 0u;
    HumanBehaviorAttemptDisposition disposition =
        HumanBehaviorAttemptDisposition::rejected;
    HumanBehaviorAcceptedState before{};
    HumanBehaviorAcceptedState after{};
    HumanBehaviorAttemptAudit audit{};
    std::optional<HumanBehaviorAcceptedMetrics> acceptedMetrics;
};

struct HumanBehaviorTerminalCompletion {
    bool quiescent = false;
    std::uint32_t activeAttemptCount = 0u;
    std::uint32_t pendingPublicationCount = 0u;
    std::int32_t exitCode = -1;
    std::uint64_t completedAttemptCount = 0u;
    HumanBehaviorAcceptedState finalAcceptedState{};

    [[nodiscard]] bool operator==(
        const HumanBehaviorTerminalCompletion& other
    ) const noexcept = default;
};

// Pure recorder core. It owns no simulation, command queue, physical state, or
// producer authentication. Calls are serialized by the enclosing runtime.
// Failed calls leave the recorder unchanged.
class HumanBehaviorTrialRecorder final {
public:
    [[nodiscard]] static std::unique_ptr<HumanBehaviorTrialRecorder> create(
        const HumanBehaviorTrialDescriptor& descriptor,
        std::string& error
    );

    ~HumanBehaviorTrialRecorder();
    HumanBehaviorTrialRecorder(const HumanBehaviorTrialRecorder&) = delete;
    HumanBehaviorTrialRecorder& operator=(
        const HumanBehaviorTrialRecorder&
    ) = delete;

    [[nodiscard]] bool recordAttempt(
        const HumanBehaviorAttempt& attempt,
        std::string& error
    );

    // Materialize a bounded aggregate span at a quiescent accepted-root
    // boundary. At least one accepted root must be pending.
    [[nodiscard]] bool sealAcceptedSpan(std::string& error);

    // Produces one header, one or more accepted_span records, and exactly one
    // completed footer. Repeating the identical finish is byte-identical;
    // every other operation after completion is rejected.
    [[nodiscard]] bool finish(
        const HumanBehaviorTerminalCompletion& terminal,
        std::string& jsonLines,
        std::string& error
    );

    [[nodiscard]] std::uint64_t acceptedStepCount() const noexcept;
    [[nodiscard]] std::uint64_t completedAttemptCount() const noexcept;
    [[nodiscard]] bool finished() const noexcept;

private:
    struct Impl;
    explicit HumanBehaviorTrialRecorder(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

} // namespace metalrobo
