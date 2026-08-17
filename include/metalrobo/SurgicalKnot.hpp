#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace metalrobo {

using SurgicalKnotPoint = std::array<double, 3>;

struct SurgicalKnotInstrumentSample {
    double timeSeconds = 0.0;
    SurgicalKnotPoint workingJawCenterM{};
    SurgicalKnotPoint standingJawCenterM{};
};

// One instrument-level throw. The winding interval is samples
// [0, windingEndSample], the tail-transfer interval is
// [windingEndSample, transferEndSample], and the remaining samples are the
// opposing cinch. The certificate is deliberately geometric: it proves the
// commanded jaw centres execute the authored winding, pass through the
// finite bight gate, and separate monotonically. Thread contact and a formed
// load-bearing knot remain separate DER/live-world authorities.
struct SurgicalThrowPath {
    std::vector<SurgicalKnotInstrumentSample> samples;
    std::uint32_t windingEndSample = 0u;
    std::uint32_t transferEndSample = 0u;
    SurgicalKnotPoint windingAxis{};
    SurgicalKnotPoint transferGateCenterM{};
    SurgicalKnotPoint transferGateNormal{};
    double transferGateRadiusM = 0.0;
    double instrumentEnvelopeRadiusM = 0.0;
    double minimumInstrumentClearanceM = 0.0;
    double maximumJawCenterSpeedMps = 0.0;
    double minimumCinchSeparationGainM = 0.0;
    std::uint32_t expectedWholeTurns = 0u;
    std::int32_t expectedWindingSign = 0;
    std::int32_t expectedTransferSign = 0;
};

enum class SurgicalThrowStatus : std::uint32_t {
    success = 0u,
    invalidDimensions,
    nonfiniteSample,
    nonmonotonicTime,
    invalidFrame,
    speedLimitViolation,
    windingSingularity,
    windingMismatch,
    instrumentClearanceViolation,
    transferGateCrossingMismatch,
    transferGateClearanceViolation,
    cinchSeparationReversal,
    insufficientCinch,
};

struct SurgicalThrowDiagnostics {
    SurgicalThrowStatus status = SurgicalThrowStatus::success;
    std::uint32_t rejectedSample = 0u;
    double signedWindingTurns = 0.0;
    double windingErrorTurns = 0.0;
    double minimumInstrumentClearanceM = 0.0;
    double maximumWorkingJawSpeedMps = 0.0;
    double maximumStandingJawSpeedMps = 0.0;
    std::uint32_t transferGateCrossings = 0u;
    double transferGateClearanceM = 0.0;
    double initialCinchSeparationM = 0.0;
    double finalCinchSeparationM = 0.0;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == SurgicalThrowStatus::success;
    }
};

struct SurgeonsKnotInstrumentProtocol {
    SurgicalThrowPath firstDoubleThrow;
    SurgicalThrowPath reversingSingleThrow;
};

enum class SurgeonsKnotProtocolStatus : std::uint32_t {
    success = 0u,
    invalidFirstThrow,
    invalidReversingThrow,
    invalidThrowSequence,
};

struct SurgeonsKnotProtocolDiagnostics {
    SurgeonsKnotProtocolStatus status =
        SurgeonsKnotProtocolStatus::success;
    SurgicalThrowDiagnostics firstDoubleThrow{};
    SurgicalThrowDiagnostics reversingSingleThrow{};

    [[nodiscard]] bool succeeded() const noexcept {
        return status == SurgeonsKnotProtocolStatus::success;
    }
};

// Deterministic research-scale instrument protocol derived from the
// source-pinned Large Needle Driver envelope. This is a trajectory fixture,
// not a claim of clinical technique validation or live robot execution.
[[nodiscard]] SurgeonsKnotInstrumentProtocol
makeSurgeonsKnotInstrumentProtocol();

[[nodiscard]] SurgicalThrowDiagnostics certifySurgicalThrowPath(
    const SurgicalThrowPath& path
) noexcept;

[[nodiscard]] SurgeonsKnotProtocolDiagnostics
certifySurgeonsKnotInstrumentProtocol(
    const SurgeonsKnotInstrumentProtocol& protocol
) noexcept;

[[nodiscard]] const char* surgicalThrowStatusName(
    SurgicalThrowStatus status
) noexcept;

[[nodiscard]] const char* surgeonsKnotProtocolStatusName(
    SurgeonsKnotProtocolStatus status
) noexcept;

} // namespace metalrobo
