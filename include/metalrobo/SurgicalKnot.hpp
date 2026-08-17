#pragma once

#include <array>
#include <cstdint>
#include <span>
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

struct SurgicalKnotContactSpec {
    double threadRadiusM = 0.0;
    double contactMarginM = 0.0;
    double separationToleranceM = 0.0;
    std::uint32_t minimumMaterialEdgeSeparation = 0u;
    std::uint32_t minimumContactPairCount = 0u;
};

enum class SurgicalKnotContactStatus : std::uint32_t {
    success = 0u,
    invalidSpecification,
    invalidTopology,
    nonfiniteNode,
    degenerateEdge,
    interpenetratingContact,
    insufficientContacts,
};

// Radius-correct, material-separated DER self-contact evidence. This proves
// that a resolved centreline contains a requested number of non-neighbouring
// strand contacts without centreline interpenetration. It does not infer a
// surgical throw from proximity alone; live use combines it with the
// instrument-path certificate, jaw retention, and an opposing load test.
struct SurgicalKnotContactDiagnostics {
    SurgicalKnotContactStatus status =
        SurgicalKnotContactStatus::success;
    std::uint64_t testedPairCount = 0u;
    std::uint32_t contactPairCount = 0u;
    std::uint32_t interpenetratingPairCount = 0u;
    std::uint32_t minimumContactMaterialEdgeSeparation = 0u;
    double minimumCenterlineDistanceM = 0.0;
    double minimumContactSurfaceGapM = 0.0;
    double maximumContactSurfaceGapM = 0.0;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == SurgicalKnotContactStatus::success;
    }
};

struct SurgicalSutureMaterialPlanSpec {
    double targetFreeTailLengthM = 0.0;
    double freeTailToleranceM = 0.0;
    double minimumWorkingArcLengthM = 0.0;
    double minimumStitchArcLengthM = 0.0;
    double maximumDrawPerStrokeM = 0.0;
    std::uint32_t maximumStrokeCount = 0u;
};

enum class SurgicalSutureMaterialPlanStatus : std::uint32_t {
    success = 0u,
    invalidDimensions,
    nonfiniteInput,
    invalidTopology,
    invalidSpecification,
    reversedTractOrder,
    overPulledTail,
    insufficientWorkingArc,
    insufficientStitchArc,
    strokeCapacityExceeded,
};

struct SurgicalSutureMaterialState {
    std::uint32_t opposingTractEdge = 0u;
    std::uint32_t firstTractEdge = 0u;
    double totalRestLengthM = 0.0;
    double workingArcLengthM = 0.0;
    double stitchArcLengthM = 0.0;
    double freeTailLengthM = 0.0;
    double opposingTractMaterialCoordinateM = 0.0;
    double firstTractMaterialCoordinateM = 0.0;
    double conservationErrorM = 0.0;
};

struct SurgicalSuturePullStroke {
    std::uint32_t index = 0u;
    double drawLengthM = 0.0;
    double freeTailBeforeM = 0.0;
    double freeTailAfterM = 0.0;
};

struct SurgicalSutureMaterialPlan {
    SurgicalSutureMaterialPlanStatus status =
        SurgicalSutureMaterialPlanStatus::success;
    SurgicalSutureMaterialState current{};
    SurgicalSutureMaterialState target{};
    double requiredDrawLengthM = 0.0;
    std::vector<SurgicalSuturePullStroke> strokes;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == SurgicalSutureMaterialPlanStatus::success;
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

[[nodiscard]] SurgicalKnotContactDiagnostics
certifySurgicalKnotContacts(
    std::span<const SurgicalKnotPoint> threadNodes,
    const SurgicalKnotContactSpec& spec
) noexcept;

// Partitions source material coordinates, not stretched world-space length.
// DER node zero is the needle/swage end and the final node is the free tail.
// The return tract must therefore precede the first tract in material order.
// Pull strokes transfer rest arc from the free-tail side to the working side
// while preserving the inter-tract stitch span exactly.
[[nodiscard]] SurgicalSutureMaterialPlan planSurgicalSuturePullThrough(
    std::span<const SurgicalKnotPoint> threadRestNodes,
    std::uint32_t firstTractEdge,
    std::uint32_t opposingTractEdge,
    const SurgicalSutureMaterialPlanSpec& spec
);

[[nodiscard]] const char* surgicalThrowStatusName(
    SurgicalThrowStatus status
) noexcept;

[[nodiscard]] const char* surgeonsKnotProtocolStatusName(
    SurgeonsKnotProtocolStatus status
) noexcept;

[[nodiscard]] const char* surgicalKnotContactStatusName(
    SurgicalKnotContactStatus status
) noexcept;

[[nodiscard]] const char* surgicalSutureMaterialPlanStatusName(
    SurgicalSutureMaterialPlanStatus status
) noexcept;

} // namespace metalrobo
