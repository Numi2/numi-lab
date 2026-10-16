#include "metalrobo/SurgicalKnot.hpp"

#include "metalrobo/SurgicalPSM.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
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

double segmentDistance(
    const Point& firstA,
    const Point& firstB,
    const Point& secondA,
    const Point& secondB
) {
    const Point first = subtract(firstB, firstA);
    const Point second = subtract(secondB, secondA);
    const Point offset = subtract(firstA, secondA);
    const double aa = dot(first, first);
    const double bb = dot(second, second);
    const double ab = dot(first, second);
    const double ao = dot(first, offset);
    const double bo = dot(second, offset);
    const double denominator = aa * bb - ab * ab;
    double firstParameter = denominator > 1.0e-14 * aa * bb
        ? std::clamp(
              (ab * bo - ao * bb) / denominator,
              0.0,
              1.0
          )
        : 0.0;
    const double secondNumerator = ab * firstParameter + bo;
    double secondParameter = 0.0;
    if (secondNumerator < 0.0) {
        firstParameter = std::clamp(-ao / aa, 0.0, 1.0);
    } else if (secondNumerator > bb) {
        secondParameter = 1.0;
        firstParameter = std::clamp(
            (ab - ao) / aa,
            0.0,
            1.0
        );
    } else {
        secondParameter = secondNumerator / bb;
    }
    const Point firstPoint = add(
        firstA,
        multiply(first, firstParameter)
    );
    const Point secondPoint = add(
        secondA,
        multiply(second, secondParameter)
    );
    return length(subtract(secondPoint, firstPoint));
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

struct ProtocolHash {
    std::uint64_t value = 14695981039346656037ull;

    void word(const std::uint64_t word) noexcept {
        for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
            value ^= (word >> (8u * byte)) & 0xffu;
            value *= 1099511628211ull;
        }
    }

    void scalar(const double scalar) noexcept {
        word(std::bit_cast<std::uint64_t>(scalar));
    }

    void point(const Point& point) noexcept {
        scalar(point[0]);
        scalar(point[1]);
        scalar(point[2]);
    }

    void path(const SurgicalThrowPath& path) noexcept {
        word(path.samples.size());
        word(path.windingEndSample);
        word(path.transferEndSample);
        point(path.windingAxis);
        point(path.transferGateCenterM);
        point(path.transferGateNormal);
        scalar(path.transferGateRadiusM);
        scalar(path.instrumentEnvelopeRadiusM);
        scalar(path.minimumInstrumentClearanceM);
        scalar(path.maximumJawCenterSpeedMps);
        scalar(path.minimumCinchSeparationGainM);
        word(path.expectedWholeTurns);
        word(static_cast<std::uint64_t>(
            static_cast<std::int64_t>(path.expectedWindingSign)
        ));
        word(static_cast<std::uint64_t>(
            static_cast<std::int64_t>(path.expectedTransferSign)
        ));
        for (const SurgicalKnotInstrumentSample& sample : path.samples) {
            scalar(sample.timeSeconds);
            point(sample.workingJawCenterM);
            point(sample.standingJawCenterM);
        }
    }
};

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
    SurgeonsKnotInstrumentProtocol result;
    result.firstDoubleThrow = makeThrow(
        2u,
        1,
        -1,
        envelopeRadius,
        jawLength
    );
    for (std::size_t throwIndex = 0u;
         throwIndex < result.squareSingleThrows.size();
         ++throwIndex) {
        const std::int32_t windingSign =
            (throwIndex & 1u) == 0u ? -1 : 1;
        result.squareSingleThrows[throwIndex] = makeThrow(
            1u,
            windingSign,
            -windingSign,
            envelopeRadius,
            jawLength
        );
    }
    return result;
}

std::uint64_t surgeonsKnotInstrumentProtocolFingerprint(
    const SurgeonsKnotInstrumentProtocol& protocol
) noexcept {
    ProtocolHash hash;
    // Domain-separate the wire identity from any other FNV-backed contract.
    hash.word(0x6e756d692e6b6e74ull); // "numi.knt"
    hash.path(protocol.firstDoubleThrow);
    hash.word(protocol.squareSingleThrows.size());
    for (const SurgicalThrowPath& path : protocol.squareSingleThrows) {
        hash.path(path);
    }
    return hash.value == 0u ? 1u : hash.value;
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
    const SurgicalThrowPath& first = protocol.firstDoubleThrow;
    if (first.expectedWholeTurns != 2u) {
        diagnostics.status =
            SurgeonsKnotProtocolStatus::invalidThrowSequence;
        return diagnostics;
    }
    std::int32_t previousWindingSign = first.expectedWindingSign;
    std::int32_t previousTransferSign = first.expectedTransferSign;
    for (std::size_t throwIndex = 0u;
         throwIndex < protocol.squareSingleThrows.size();
         ++throwIndex) {
        const SurgicalThrowPath& current =
            protocol.squareSingleThrows[throwIndex];
        diagnostics.squareSingleThrows[throwIndex] =
            certifySurgicalThrowPath(current);
        if (!diagnostics.squareSingleThrows[throwIndex].succeeded()) {
            diagnostics.status =
                SurgeonsKnotProtocolStatus::invalidSingleThrow;
            diagnostics.rejectedThrow = static_cast<std::uint32_t>(
                throwIndex + 2u
            );
            return diagnostics;
        }
        if (current.expectedWholeTurns != 1u ||
            current.expectedWindingSign != -previousWindingSign ||
            current.expectedTransferSign != -previousTransferSign) {
            diagnostics.status =
                SurgeonsKnotProtocolStatus::invalidThrowSequence;
            diagnostics.rejectedThrow = static_cast<std::uint32_t>(
                throwIndex + 2u
            );
            return diagnostics;
        }
        previousWindingSign = current.expectedWindingSign;
        previousTransferSign = current.expectedTransferSign;
    }
    return diagnostics;
}

SurgicalKnotContactDiagnostics certifySurgicalKnotContacts(
    const std::span<const SurgicalKnotPoint> threadNodes,
    const SurgicalKnotContactSpec& spec
) noexcept {
    SurgicalKnotContactDiagnostics diagnostics;
    diagnostics.minimumCenterlineDistanceM =
        std::numeric_limits<double>::infinity();
    diagnostics.minimumContactSurfaceGapM =
        std::numeric_limits<double>::infinity();
    diagnostics.maximumContactSurfaceGapM =
        -std::numeric_limits<double>::infinity();
    diagnostics.minimumContactMaterialEdgeSeparation =
        std::numeric_limits<std::uint32_t>::max();
    if (!std::isfinite(spec.threadRadiusM) ||
        !std::isfinite(spec.contactMarginM) ||
        !std::isfinite(spec.separationToleranceM) ||
        !(spec.threadRadiusM > 0.0) ||
        spec.contactMarginM < 0.0 ||
        spec.separationToleranceM < 0.0 ||
        spec.minimumMaterialEdgeSeparation < 2u) {
        diagnostics.status =
            SurgicalKnotContactStatus::invalidSpecification;
        return diagnostics;
    }
    if (threadNodes.size() < 4u ||
        spec.minimumMaterialEdgeSeparation >= threadNodes.size() - 1u) {
        diagnostics.status = SurgicalKnotContactStatus::invalidTopology;
        return diagnostics;
    }
    for (std::size_t node = 0u; node < threadNodes.size(); ++node) {
        if (!finite(threadNodes[node])) {
            diagnostics.status = SurgicalKnotContactStatus::nonfiniteNode;
            return diagnostics;
        }
        if (node > 0u &&
            !(length(subtract(
                threadNodes[node],
                threadNodes[node - 1u]
            )) > 1.0e-12)) {
            diagnostics.status = SurgicalKnotContactStatus::degenerateEdge;
            return diagnostics;
        }
    }

    const std::size_t edgeCount = threadNodes.size() - 1u;
    const double diameter = 2.0 * spec.threadRadiusM;
    const double contactDistance =
        diameter + spec.contactMarginM + spec.separationToleranceM;
    for (std::size_t firstEdge = 0u;
         firstEdge < edgeCount;
         ++firstEdge) {
        for (std::size_t secondEdge =
                 firstEdge + spec.minimumMaterialEdgeSeparation;
             secondEdge < edgeCount;
             ++secondEdge) {
            ++diagnostics.testedPairCount;
            const double distance = segmentDistance(
                threadNodes[firstEdge],
                threadNodes[firstEdge + 1u],
                threadNodes[secondEdge],
                threadNodes[secondEdge + 1u]
            );
            if (!std::isfinite(distance)) {
                diagnostics.status =
                    SurgicalKnotContactStatus::invalidTopology;
                return diagnostics;
            }
            diagnostics.minimumCenterlineDistanceM = std::min(
                diagnostics.minimumCenterlineDistanceM,
                distance
            );
            if (distance < diameter - spec.separationToleranceM) {
                ++diagnostics.interpenetratingPairCount;
            }
            if (distance > contactDistance) {
                continue;
            }
            ++diagnostics.contactPairCount;
            diagnostics.minimumContactMaterialEdgeSeparation = std::min(
                diagnostics.minimumContactMaterialEdgeSeparation,
                static_cast<std::uint32_t>(secondEdge - firstEdge)
            );
            const double surfaceGap = distance - diameter;
            diagnostics.minimumContactSurfaceGapM = std::min(
                diagnostics.minimumContactSurfaceGapM,
                surfaceGap
            );
            diagnostics.maximumContactSurfaceGapM = std::max(
                diagnostics.maximumContactSurfaceGapM,
                surfaceGap
            );
        }
    }
    if (diagnostics.interpenetratingPairCount != 0u) {
        diagnostics.status =
            SurgicalKnotContactStatus::interpenetratingContact;
    } else if (diagnostics.contactPairCount <
               spec.minimumContactPairCount) {
        diagnostics.status =
            SurgicalKnotContactStatus::insufficientContacts;
    }
    return diagnostics;
}

SurgicalSutureMaterialPlan planSurgicalSuturePullThrough(
    const std::span<const SurgicalKnotPoint> threadRestNodes,
    const std::uint32_t firstTractEdge,
    const std::uint32_t opposingTractEdge,
    const SurgicalSutureMaterialPlanSpec& spec
) {
    SurgicalSutureMaterialPlan plan;
    if (threadRestNodes.size() < 3u) {
        plan.status =
            SurgicalSutureMaterialPlanStatus::invalidDimensions;
        return plan;
    }
    if (!std::isfinite(spec.targetFreeTailLengthM) ||
        !std::isfinite(spec.freeTailToleranceM) ||
        !std::isfinite(spec.minimumWorkingArcLengthM) ||
        !std::isfinite(spec.minimumStitchArcLengthM) ||
        !std::isfinite(spec.maximumDrawPerStrokeM)) {
        plan.status = SurgicalSutureMaterialPlanStatus::nonfiniteInput;
        return plan;
    }
    if (!(spec.targetFreeTailLengthM > 0.0) ||
        spec.freeTailToleranceM < 0.0 ||
        spec.freeTailToleranceM >= spec.targetFreeTailLengthM ||
        !(spec.minimumWorkingArcLengthM > 0.0) ||
        !(spec.minimumStitchArcLengthM > 0.0) ||
        !(spec.maximumDrawPerStrokeM > 0.0) ||
        spec.maximumStrokeCount == 0u) {
        plan.status =
            SurgicalSutureMaterialPlanStatus::invalidSpecification;
        return plan;
    }

    std::vector<double> materialCoordinates(threadRestNodes.size(), 0.0);
    for (std::size_t node = 0u; node < threadRestNodes.size(); ++node) {
        if (!finite(threadRestNodes[node])) {
            plan.status =
                SurgicalSutureMaterialPlanStatus::nonfiniteInput;
            return plan;
        }
        if (node == 0u) {
            continue;
        }
        const double edgeLength = length(subtract(
            threadRestNodes[node],
            threadRestNodes[node - 1u]
        ));
        if (!(edgeLength > 0.0) || !std::isfinite(edgeLength)) {
            plan.status =
                SurgicalSutureMaterialPlanStatus::invalidTopology;
            return plan;
        }
        materialCoordinates[node] =
            materialCoordinates[node - 1u] + edgeLength;
    }
    const std::size_t edgeCount = threadRestNodes.size() - 1u;
    if (firstTractEdge >= edgeCount ||
        opposingTractEdge >= edgeCount) {
        plan.status =
            SurgicalSutureMaterialPlanStatus::invalidTopology;
        return plan;
    }
    if (opposingTractEdge >= firstTractEdge) {
        plan.status =
            SurgicalSutureMaterialPlanStatus::reversedTractOrder;
        return plan;
    }

    const auto edgeCenterCoordinate = [&materialCoordinates](
        const std::uint32_t edge
    ) {
        return 0.5 * (
            materialCoordinates[edge] +
            materialCoordinates[edge + 1u]
        );
    };
    const auto makeState = [&] (
        const double opposingCoordinate,
        const double firstCoordinate
    ) {
        SurgicalSutureMaterialState state;
        state.totalRestLengthM = materialCoordinates.back();
        state.opposingTractMaterialCoordinateM = opposingCoordinate;
        state.firstTractMaterialCoordinateM = firstCoordinate;
        state.workingArcLengthM = opposingCoordinate;
        state.stitchArcLengthM =
            firstCoordinate - opposingCoordinate;
        state.freeTailLengthM =
            state.totalRestLengthM - firstCoordinate;
        state.conservationErrorM = std::abs(
            state.workingArcLengthM + state.stitchArcLengthM +
                state.freeTailLengthM - state.totalRestLengthM
        );
        const auto materialEdge = [&materialCoordinates](
            const double coordinate
        ) {
            const auto after = std::upper_bound(
                materialCoordinates.begin(),
                materialCoordinates.end(),
                coordinate
            );
            const std::size_t node = static_cast<std::size_t>(
                std::distance(materialCoordinates.begin(), after)
            );
            return static_cast<std::uint32_t>(std::clamp<std::size_t>(
                node == 0u ? 0u : node - 1u,
                0u,
                materialCoordinates.size() - 2u
            ));
        };
        state.opposingTractEdge = materialEdge(opposingCoordinate);
        state.firstTractEdge = materialEdge(firstCoordinate);
        return state;
    };

    plan.current = makeState(
        edgeCenterCoordinate(opposingTractEdge),
        edgeCenterCoordinate(firstTractEdge)
    );
    plan.current.opposingTractEdge = opposingTractEdge;
    plan.current.firstTractEdge = firstTractEdge;
    if (plan.current.freeTailLengthM <
        spec.targetFreeTailLengthM - spec.freeTailToleranceM) {
        plan.status = SurgicalSutureMaterialPlanStatus::overPulledTail;
        return plan;
    }
    if (plan.current.stitchArcLengthM <
        spec.minimumStitchArcLengthM) {
        plan.status =
            SurgicalSutureMaterialPlanStatus::insufficientStitchArc;
        return plan;
    }

    plan.requiredDrawLengthM =
        plan.current.freeTailLengthM >
                spec.targetFreeTailLengthM + spec.freeTailToleranceM
            ? plan.current.freeTailLengthM -
                spec.targetFreeTailLengthM
            : 0.0;
    plan.target = makeState(
        plan.current.opposingTractMaterialCoordinateM +
            plan.requiredDrawLengthM,
        plan.current.firstTractMaterialCoordinateM +
            plan.requiredDrawLengthM
    );
    if (plan.target.workingArcLengthM <
        spec.minimumWorkingArcLengthM) {
        plan.status =
            SurgicalSutureMaterialPlanStatus::insufficientWorkingArc;
        return plan;
    }

    double remaining = plan.requiredDrawLengthM;
    double currentTail = plan.current.freeTailLengthM;
    while (remaining > 1.0e-12) {
        if (plan.strokes.size() >= spec.maximumStrokeCount) {
            plan.status =
                SurgicalSutureMaterialPlanStatus::strokeCapacityExceeded;
            plan.strokes.clear();
            return plan;
        }
        const double draw = std::min(
            remaining,
            spec.maximumDrawPerStrokeM
        );
        const double nextTail = currentTail - draw;
        plan.strokes.push_back({
            .index = static_cast<std::uint32_t>(plan.strokes.size()),
            .drawLengthM = draw,
            .freeTailBeforeM = currentTail,
            .freeTailAfterM = nextTail,
        });
        currentTail = nextTail;
        remaining -= draw;
    }
    if (std::abs(currentTail - plan.target.freeTailLengthM) >
            1.0e-10 ||
        plan.target.conservationErrorM > 1.0e-12) {
        plan.status =
            SurgicalSutureMaterialPlanStatus::invalidTopology;
        plan.strokes.clear();
        return plan;
    }
    plan.status = SurgicalSutureMaterialPlanStatus::success;
    return plan;
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
    case SurgeonsKnotProtocolStatus::invalidSingleThrow:
        return "invalid_single_throw";
    case SurgeonsKnotProtocolStatus::invalidThrowSequence:
        return "invalid_throw_sequence";
    }
    return "unknown";
}

const char* surgicalKnotContactStatusName(
    const SurgicalKnotContactStatus status
) noexcept {
    switch (status) {
    case SurgicalKnotContactStatus::success:
        return "success";
    case SurgicalKnotContactStatus::invalidSpecification:
        return "invalid_specification";
    case SurgicalKnotContactStatus::invalidTopology:
        return "invalid_topology";
    case SurgicalKnotContactStatus::nonfiniteNode:
        return "nonfinite_node";
    case SurgicalKnotContactStatus::degenerateEdge:
        return "degenerate_edge";
    case SurgicalKnotContactStatus::interpenetratingContact:
        return "interpenetrating_contact";
    case SurgicalKnotContactStatus::insufficientContacts:
        return "insufficient_contacts";
    }
    return "unknown";
}

const char* surgicalSutureMaterialPlanStatusName(
    const SurgicalSutureMaterialPlanStatus status
) noexcept {
    switch (status) {
    case SurgicalSutureMaterialPlanStatus::success:
        return "success";
    case SurgicalSutureMaterialPlanStatus::invalidDimensions:
        return "invalid_dimensions";
    case SurgicalSutureMaterialPlanStatus::nonfiniteInput:
        return "nonfinite_input";
    case SurgicalSutureMaterialPlanStatus::invalidTopology:
        return "invalid_topology";
    case SurgicalSutureMaterialPlanStatus::invalidSpecification:
        return "invalid_specification";
    case SurgicalSutureMaterialPlanStatus::reversedTractOrder:
        return "reversed_tract_order";
    case SurgicalSutureMaterialPlanStatus::overPulledTail:
        return "over_pulled_tail";
    case SurgicalSutureMaterialPlanStatus::insufficientWorkingArc:
        return "insufficient_working_arc";
    case SurgicalSutureMaterialPlanStatus::insufficientStitchArc:
        return "insufficient_stitch_arc";
    case SurgicalSutureMaterialPlanStatus::strokeCapacityExceeded:
        return "stroke_capacity_exceeded";
    }
    return "unknown";
}

} // namespace metalrobo
