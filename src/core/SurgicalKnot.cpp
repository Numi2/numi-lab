#include "metalrobo/SurgicalKnot.hpp"

#include "metalrobo/SurgicalPSM.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace metalrobo {
namespace {

using Point = SurgicalKnotPoint;

Point add(const Point& left, const Point& right) {
    return {
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
    };
}

Point subtract(const Point& left, const Point& right) {
    return {
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    };
}

Point multiply(const Point& value, const double scale) {
    return {
        value[0] * scale,
        value[1] * scale,
        value[2] * scale,
    };
}

double dot(const Point& left, const Point& right) {
    return left[0] * right[0] +
        left[1] * right[1] +
        left[2] * right[2];
}

Point cross(const Point& left, const Point& right) {
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}

double length(const Point& value) {
    return std::sqrt(dot(value, value));
}

bool finite(const Point& value) {
    return
        std::isfinite(value[0]) &&
        std::isfinite(value[1]) &&
        std::isfinite(value[2]);
}

Point normalized(const Point& value) {
    const double magnitude = length(value);
    if (!(magnitude > 0.0) || !std::isfinite(magnitude)) {
        return {};
    }
    return multiply(value, 1.0 / magnitude);
}

void appendSample(
    SurgicalThrowPath& path,
    const Point& working,
    const Point& standing,
    const double targetSpeed
) {
    double time = 0.0;
    if (!path.samples.empty()) {
        const auto& previous = path.samples.back();
        const double travel = std::max(
            length(subtract(working, previous.workingJawCenterM)),
            length(subtract(standing, previous.standingJawCenterM))
        );
        time = previous.timeSeconds + travel / targetSpeed;
    }
    path.samples.push_back({
        .timeSeconds = time,
        .workingJawCenterM = working,
        .standingJawCenterM = standing,
    });
}

SurgicalThrowPath makeThrow(
    const std::uint32_t turns,
    const std::int32_t windingSign,
    const std::int32_t transferSign,
    const double instrumentEnvelopeRadius,
    const double jawLength
) {
    constexpr Point axis{0.0, 0.0, 1.0};
    constexpr Point standing{0.0, 0.0, 0.0};
    constexpr double targetSpeed = 2.0e-2;
    constexpr double wrapRadius = 1.2e-2;
    constexpr double transferHalfTravel = 1.2e-2;
    constexpr std::uint32_t windingSamplesPerTurn = 128u;
    constexpr std::uint32_t transferSteps = 31u;
    constexpr std::uint32_t cinchSteps = 64u;

    SurgicalThrowPath path;
    path.windingAxis = axis;
    path.transferGateCenterM = {wrapRadius, 0.0, 0.0};
    path.transferGateNormal = axis;
    path.transferGateRadiusM = std::max(1.2e-2, jawLength);
    path.instrumentEnvelopeRadiusM = instrumentEnvelopeRadius;
    path.minimumInstrumentClearanceM = 2.0e-3;
    path.maximumJawCenterSpeedMps = targetSpeed;
    path.minimumCinchSeparationGainM = 2.0e-2;
    path.expectedWholeTurns = turns;
    path.expectedWindingSign = windingSign;
    path.expectedTransferSign = transferSign;

    const double windingHeight =
        transferSign < 0 ? transferHalfTravel : -transferHalfTravel;
    const std::uint32_t windingSteps =
        windingSamplesPerTurn * turns;
    for (std::uint32_t sample = 0u;
         sample <= windingSteps;
         ++sample) {
        const double angle =
            static_cast<double>(windingSign) *
            2.0 * std::numbers::pi *
            static_cast<double>(sample) /
            static_cast<double>(windingSamplesPerTurn);
        appendSample(
            path,
            {
                wrapRadius * std::cos(angle),
                wrapRadius * std::sin(angle),
                windingHeight,
            },
            standing,
            targetSpeed
        );
    }
    path.windingEndSample =
        static_cast<std::uint32_t>(path.samples.size() - 1u);

    const Point transferStart = path.samples.back().workingJawCenterM;
    const Point transferEnd{
        transferStart[0],
        transferStart[1],
        -windingHeight,
    };
    for (std::uint32_t step = 1u; step <= transferSteps; ++step) {
        const double fraction =
            static_cast<double>(step) /
            static_cast<double>(transferSteps);
        appendSample(
            path,
            add(
                transferStart,
                multiply(subtract(transferEnd, transferStart), fraction)
            ),
            standing,
            targetSpeed
        );
    }
    path.transferEndSample =
        static_cast<std::uint32_t>(path.samples.size() - 1u);

    const Point terminalWorking{
        2.5e-2,
        0.0,
        transferEnd[2],
    };
    const Point terminalStanding{-2.5e-2, 0.0, 0.0};
    for (std::uint32_t step = 1u; step <= cinchSteps; ++step) {
        const double fraction =
            static_cast<double>(step) /
            static_cast<double>(cinchSteps);
        appendSample(
            path,
            add(
                transferEnd,
                multiply(
                    subtract(terminalWorking, transferEnd),
                    fraction
                )
            ),
            multiply(terminalStanding, fraction),
            targetSpeed
        );
    }
    return path;
}

} // namespace

SurgeonsKnotInstrumentProtocol makeSurgeonsKnotInstrumentProtocol() {
    const SurgicalPSMModelMetadata& psm = surgicalPSMMetadata();
    if (!(psm.instrumentDiameter > 0.0f) ||
        !(psm.largeNeedleDriverJawLength > 0.0f)) {
        throw std::logic_error(
            "source-pinned Large Needle Driver dimensions are unavailable"
        );
    }
    const double envelopeRadius =
        0.5 * static_cast<double>(psm.instrumentDiameter);
    const double jawLength =
        static_cast<double>(psm.largeNeedleDriverJawLength);
    return {
        .firstDoubleThrow = makeThrow(
            2u,
            1,
            -1,
            envelopeRadius,
            jawLength
        ),
        .reversingSingleThrow = makeThrow(
            1u,
            -1,
            1,
            envelopeRadius,
            jawLength
        ),
    };
}

SurgicalThrowDiagnostics certifySurgicalThrowPath(
    const SurgicalThrowPath& path
) noexcept {
    SurgicalThrowDiagnostics diagnostics;
    diagnostics.minimumInstrumentClearanceM =
        std::numeric_limits<double>::infinity();
    diagnostics.transferGateClearanceM =
        std::numeric_limits<double>::infinity();
    if (path.samples.size() < 5u ||
        path.windingEndSample < 2u ||
        path.transferEndSample <= path.windingEndSample ||
        path.transferEndSample >= path.samples.size() - 1u ||
        path.expectedWholeTurns == 0u ||
        std::abs(path.expectedWindingSign) != 1 ||
        std::abs(path.expectedTransferSign) != 1 ||
        !(path.transferGateRadiusM > 0.0) ||
        !(path.instrumentEnvelopeRadiusM > 0.0) ||
        !(path.minimumInstrumentClearanceM >= 0.0) ||
        !(path.maximumJawCenterSpeedMps > 0.0) ||
        !(path.minimumCinchSeparationGainM > 0.0)) {
        diagnostics.status = SurgicalThrowStatus::invalidDimensions;
        return diagnostics;
    }

    const Point axis = normalized(path.windingAxis);
    const Point gateNormal = normalized(path.transferGateNormal);
    if (!(length(axis) > 0.0) || !(length(gateNormal) > 0.0) ||
        !finite(path.transferGateCenterM)) {
        diagnostics.status = SurgicalThrowStatus::invalidFrame;
        return diagnostics;
    }

    for (std::size_t sample = 0u; sample < path.samples.size(); ++sample) {
        const auto& current = path.samples[sample];
        if (!std::isfinite(current.timeSeconds) ||
            !finite(current.workingJawCenterM) ||
            !finite(current.standingJawCenterM)) {
            diagnostics.status = SurgicalThrowStatus::nonfiniteSample;
            diagnostics.rejectedSample =
                static_cast<std::uint32_t>(sample);
            return diagnostics;
        }
        if (sample == 0u) {
            continue;
        }
        const auto& previous = path.samples[sample - 1u];
        const double elapsed =
            current.timeSeconds - previous.timeSeconds;
        if (!(elapsed > 0.0) || !std::isfinite(elapsed)) {
            diagnostics.status = SurgicalThrowStatus::nonmonotonicTime;
            diagnostics.rejectedSample =
                static_cast<std::uint32_t>(sample);
            return diagnostics;
        }
        diagnostics.maximumWorkingJawSpeedMps = std::max(
            diagnostics.maximumWorkingJawSpeedMps,
            length(subtract(
                current.workingJawCenterM,
                previous.workingJawCenterM
            )) / elapsed
        );
        diagnostics.maximumStandingJawSpeedMps = std::max(
            diagnostics.maximumStandingJawSpeedMps,
            length(subtract(
                current.standingJawCenterM,
                previous.standingJawCenterM
            )) / elapsed
        );
    }
    if (diagnostics.maximumWorkingJawSpeedMps >
            path.maximumJawCenterSpeedMps + 1.0e-10 ||
        diagnostics.maximumStandingJawSpeedMps >
            path.maximumJawCenterSpeedMps + 1.0e-10) {
        diagnostics.status = SurgicalThrowStatus::speedLimitViolation;
        return diagnostics;
    }

    Point previousRadial{};
    double signedAngle = 0.0;
    for (std::uint32_t sample = 0u;
         sample <= path.windingEndSample;
         ++sample) {
        const Point relative = subtract(
            path.samples[sample].workingJawCenterM,
            path.samples[sample].standingJawCenterM
        );
        const Point radial = subtract(
            relative,
            multiply(axis, dot(relative, axis))
        );
        const double radius = length(radial);
        if (!(radius > 0.0) || !std::isfinite(radius)) {
            diagnostics.status = SurgicalThrowStatus::windingSingularity;
            diagnostics.rejectedSample = sample;
            return diagnostics;
        }
        diagnostics.minimumInstrumentClearanceM = std::min(
            diagnostics.minimumInstrumentClearanceM,
            radius - 2.0 * path.instrumentEnvelopeRadiusM
        );
        if (sample > 0u) {
            signedAngle += std::atan2(
                dot(axis, cross(previousRadial, radial)),
                dot(previousRadial, radial)
            );
        }
        previousRadial = radial;
    }
    diagnostics.signedWindingTurns =
        signedAngle / (2.0 * std::numbers::pi);
    const double expectedTurns =
        static_cast<double>(path.expectedWindingSign) *
        static_cast<double>(path.expectedWholeTurns);
    diagnostics.windingErrorTurns =
        std::abs(diagnostics.signedWindingTurns - expectedTurns);
    if (diagnostics.windingErrorTurns > 1.0e-3) {
        diagnostics.status = SurgicalThrowStatus::windingMismatch;
        return diagnostics;
    }
    if (diagnostics.minimumInstrumentClearanceM <
        path.minimumInstrumentClearanceM) {
        diagnostics.status =
            SurgicalThrowStatus::instrumentClearanceViolation;
        return diagnostics;
    }

    for (std::uint32_t sample = path.windingEndSample + 1u;
         sample <= path.transferEndSample;
         ++sample) {
        const Point& before =
            path.samples[sample - 1u].workingJawCenterM;
        const Point& after = path.samples[sample].workingJawCenterM;
        const double beforeSide = dot(
            subtract(before, path.transferGateCenterM),
            gateNormal
        );
        const double afterSide = dot(
            subtract(after, path.transferGateCenterM),
            gateNormal
        );
        if (!((beforeSide < 0.0 && afterSide > 0.0) ||
              (beforeSide > 0.0 && afterSide < 0.0))) {
            continue;
        }
        ++diagnostics.transferGateCrossings;
        const double fraction = beforeSide / (beforeSide - afterSide);
        const Point intersection = add(
            before,
            multiply(subtract(after, before), fraction)
        );
        const Point gateDelta = subtract(
            intersection,
            path.transferGateCenterM
        );
        const Point inPlane = subtract(
            gateDelta,
            multiply(gateNormal, dot(gateDelta, gateNormal))
        );
        diagnostics.transferGateClearanceM = std::min(
            diagnostics.transferGateClearanceM,
            path.transferGateRadiusM -
                path.instrumentEnvelopeRadiusM - length(inPlane)
        );
        const double direction = dot(
            subtract(after, before),
            gateNormal
        );
        if (direction * static_cast<double>(path.expectedTransferSign) <=
            0.0) {
            diagnostics.status =
                SurgicalThrowStatus::transferGateCrossingMismatch;
            diagnostics.rejectedSample = sample;
            return diagnostics;
        }
    }
    if (diagnostics.transferGateCrossings != 1u) {
        diagnostics.status =
            SurgicalThrowStatus::transferGateCrossingMismatch;
        return diagnostics;
    }
    if (!(diagnostics.transferGateClearanceM >= 0.0)) {
        diagnostics.status =
            SurgicalThrowStatus::transferGateClearanceViolation;
        return diagnostics;
    }

    diagnostics.initialCinchSeparationM = length(subtract(
        path.samples[path.transferEndSample].workingJawCenterM,
        path.samples[path.transferEndSample].standingJawCenterM
    ));
    double previousSeparation = diagnostics.initialCinchSeparationM;
    for (std::size_t sample = path.transferEndSample + 1u;
         sample < path.samples.size();
         ++sample) {
        const double separation = length(subtract(
            path.samples[sample].workingJawCenterM,
            path.samples[sample].standingJawCenterM
        ));
        if (separation + 1.0e-12 < previousSeparation) {
            diagnostics.status =
                SurgicalThrowStatus::cinchSeparationReversal;
            diagnostics.rejectedSample =
                static_cast<std::uint32_t>(sample);
            return diagnostics;
        }
        previousSeparation = separation;
    }
    diagnostics.finalCinchSeparationM = previousSeparation;
    if (diagnostics.finalCinchSeparationM -
            diagnostics.initialCinchSeparationM <
        path.minimumCinchSeparationGainM) {
        diagnostics.status = SurgicalThrowStatus::insufficientCinch;
        return diagnostics;
    }
    return diagnostics;
}

SurgeonsKnotProtocolDiagnostics certifySurgeonsKnotInstrumentProtocol(
    const SurgeonsKnotInstrumentProtocol& protocol
) noexcept {
    SurgeonsKnotProtocolDiagnostics diagnostics;
    diagnostics.firstDoubleThrow = certifySurgicalThrowPath(
        protocol.firstDoubleThrow
    );
    if (!diagnostics.firstDoubleThrow.succeeded()) {
        diagnostics.status =
            SurgeonsKnotProtocolStatus::invalidFirstThrow;
        return diagnostics;
    }
    diagnostics.reversingSingleThrow = certifySurgicalThrowPath(
        protocol.reversingSingleThrow
    );
    if (!diagnostics.reversingSingleThrow.succeeded()) {
        diagnostics.status =
            SurgeonsKnotProtocolStatus::invalidReversingThrow;
        return diagnostics;
    }
    const SurgicalThrowPath& first = protocol.firstDoubleThrow;
    const SurgicalThrowPath& second = protocol.reversingSingleThrow;
    if (first.expectedWholeTurns != 2u ||
        second.expectedWholeTurns != 1u ||
        first.expectedWindingSign != -second.expectedWindingSign ||
        first.expectedTransferSign != -second.expectedTransferSign) {
        diagnostics.status =
            SurgeonsKnotProtocolStatus::invalidThrowSequence;
    }
    return diagnostics;
}

const char* surgicalThrowStatusName(
    const SurgicalThrowStatus status
) noexcept {
    switch (status) {
    case SurgicalThrowStatus::success:
        return "success";
    case SurgicalThrowStatus::invalidDimensions:
        return "invalid_dimensions";
    case SurgicalThrowStatus::nonfiniteSample:
        return "nonfinite_sample";
    case SurgicalThrowStatus::nonmonotonicTime:
        return "nonmonotonic_time";
    case SurgicalThrowStatus::invalidFrame:
        return "invalid_frame";
    case SurgicalThrowStatus::speedLimitViolation:
        return "speed_limit_violation";
    case SurgicalThrowStatus::windingSingularity:
        return "winding_singularity";
    case SurgicalThrowStatus::windingMismatch:
        return "winding_mismatch";
    case SurgicalThrowStatus::instrumentClearanceViolation:
        return "instrument_clearance_violation";
    case SurgicalThrowStatus::transferGateCrossingMismatch:
        return "transfer_gate_crossing_mismatch";
    case SurgicalThrowStatus::transferGateClearanceViolation:
        return "transfer_gate_clearance_violation";
    case SurgicalThrowStatus::cinchSeparationReversal:
        return "cinch_separation_reversal";
    case SurgicalThrowStatus::insufficientCinch:
        return "insufficient_cinch";
    }
    return "unknown";
}

const char* surgeonsKnotProtocolStatusName(
    const SurgeonsKnotProtocolStatus status
) noexcept {
    switch (status) {
    case SurgeonsKnotProtocolStatus::success:
        return "success";
    case SurgeonsKnotProtocolStatus::invalidFirstThrow:
        return "invalid_first_throw";
    case SurgeonsKnotProtocolStatus::invalidReversingThrow:
        return "invalid_reversing_throw";
    case SurgeonsKnotProtocolStatus::invalidThrowSequence:
        return "invalid_throw_sequence";
    }
    return "unknown";
}

} // namespace metalrobo
