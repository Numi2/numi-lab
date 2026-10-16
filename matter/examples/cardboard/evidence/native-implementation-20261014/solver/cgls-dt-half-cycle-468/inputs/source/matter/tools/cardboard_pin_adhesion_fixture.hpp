#pragma once

// Authoring primitives for a native Matter forward model of the Johnson–Popil
// pin-adhesion fixture. Published sample size and relative crosshead rate are
// defaults below; every pin/tool dimension is an explicit engineering input.
// The six-pin convenience geometry is only a runnable tooling prototype based
// on the six pins drawn in Fig. 3, not a reconstruction of Emerson 06-00-55.

#include "numi/matter/matter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace numi::cardboard {

enum class PinDimensionBasis : std::uint32_t {
    unspecified = 0u,
    engineeringPrototype = 1u,
    externallyAuthored = 2u,
};

enum class PinCountBasis : std::uint32_t {
    unspecified = 0u,
    articleDescribedSixPins = 1u,
    engineeringPrototype = 2u,
    externallyAuthored = 3u,
};

enum class PinArrayRegistration : std::uint32_t {
    unspecified = 0u,
    centeredInSpecimenWidth = 1u,
    explicitCenterOffset = 2u,
};

enum class PinSeriesFaceAssignment : std::uint32_t {
    unspecified = 0u,
    pins135RemainingPins246Selected = 1u,
    externallyAuthored = 2u,
};

enum class PinPlatenGroup : std::uint32_t {
    remainingBoard = 0u,
    selectedFacing = 1u,
};

struct PinAdhesionFixtureSpec {
    // Article §2.2: 50 x 160 mm C-flute coupon. The article does not assign
    // MD/CD to these two dimensions; its preceding 4 x 10 in imaging coupon is
    // explicitly 10.16 cm CD x 25.4 cm MD. Keep this mapping caller-authored.
    double specimenWidthM = 0.050;
    double specimenLengthM = 0.160;
    double relativePlatenRateMPerSecond = 0.0125 / 60.0;

    // None of the values below is dimensioned in the paper. A valid geometry
    // requires the caller to supply a provenance tag and these dimensions.
    double boardCaliperM = 0.0;
    double pinPitchAcrossWidthM = 0.0;
    double pinRadiusM = 0.0;
    double pinEndOverhangM = 0.0;
    double initialClearanceM = 0.0;
    std::uint32_t totalPinCount = 0u;
    std::uint32_t firstBodyIndex = NM_INVALID_INDEX;
    std::uint32_t contactMaterialIndex = 0u;
    PinDimensionBasis dimensionBasis = PinDimensionBasis::unspecified;
    PinCountBasis pinCountBasis = PinCountBasis::unspecified;
    PinArrayRegistration arrayRegistration = PinArrayRegistration::unspecified;
    PinSeriesFaceAssignment seriesFaceAssignment =
        PinSeriesFaceAssignment::unspecified;
    double pinArrayCenterXM = 0.0;
    bool specimenWidthLengthAxesAuthored = false;

    // Board frame: x is the 50 mm coupon width, y is its 160 mm length, z is
    // out of plane. Bars span y; centers alternate across x. The source says
    // the even/odd sets contact the selected liner and remaining board but
    // does not fully dimension/register those contacts.
    bool selectedFacingIsNegativeZ = true;
    bool splitRelativeOpeningSymmetrically = true;
};

struct PinAdhesionPin {
    numi::matter::RigidProxySource proxy;
    std::uint32_t ordinal = 0u;
    PinPlatenGroup group = PinPlatenGroup::remainingBoard;
    std::array<double, 3> initialBodyTranslationM{};
};

struct PinAdhesionFixtureGeometry {
    PinAdhesionFixtureSpec spec;
    std::vector<PinAdhesionPin> pins;
    std::uint32_t remainingBoardPinCount = 0u;
    std::uint32_t selectedFacingPinCount = 0u;
    double pinOverallLengthM = 0.0;
    double firstCenterXM = 0.0;
    double lastCenterXM = 0.0;
    bool sourceExactFixtureGeometry = false;
};

struct PinAdhesionAcceptedSample {
    std::uint64_t acceptedStep = 0u;
    double relativeOpeningM = 0.0;
    // Instrument-equivalent magnitude on one crosshead platen. Do not sum the
    // equal-and-opposite reactions from both platen groups.
    double instrumentLoadMagnitudeN = 0.0;
    bool accepted = false;
};

enum class PinAdhesionPhase : std::uint32_t {
    platenApproach = 0u,
    relativeOpening = 1u,
    platenWithdrawal = 2u,
    releasedHold = 3u,
};

struct PinAdhesionSchedulePoint {
    std::uint64_t controlStep = 0u;
    PinAdhesionPhase phase = PinAdhesionPhase::platenApproach;
    std::uint32_t phaseStep = 0u;
    double elapsedSeconds = 0.0;
    // Signed separation coordinate relative to first contact. The approach
    // leg decreases from twice the initial surface clearance to zero; the
    // opening and withdrawal legs then increase monotonically.
    double startRelativeOpeningM = 0.0;
    double endRelativeOpeningM = 0.0;
    double effectiveRelativeSpeedMPerSecond = 0.0;
};

struct PinAdhesionProtocolSchedule {
    std::vector<PinAdhesionSchedulePoint> points;
    double timestepSeconds = 0.0;
    double requestedRelativeSpeedMPerSecond = 0.0;
    double maximumRelativeOpeningM = 0.0;
    double withdrawalDistanceM = 0.0;
    double totalDurationSeconds = 0.0;
};

struct PinAdhesionTraceMetrics {
    std::uint64_t acceptedPointCount = 0u;
    double firstOpeningM = 0.0;
    double lastOpeningM = 0.0;
    double peakMeasuredLoadN = 0.0;
    std::uint64_t peakAcceptedStep = 0u;
    double integratedWorkNm = 0.0;
    double integratedWorkNmm = 0.0;
    double pinStrengthNPerMm = 0.0;
    bool sourceWindowComplete = false;
};

inline bool pinFinitePositive(const double value) {
    return std::isfinite(value) && value > 0.0;
}

inline PinAdhesionFixtureSpec pinAdhesionEngineeringPrototype(
    const std::uint32_t firstBodyIndex) {
    PinAdhesionFixtureSpec spec;
    spec.boardCaliperM = 0.005;
    spec.pinPitchAcrossWidthM = 0.008;
    spec.pinRadiusM = 0.0005;
    spec.pinEndOverhangM = 0.001;
    spec.initialClearanceM = 0.00005;
    // Johnson and Popil enumerate pins 1,3,5 on one platen and 2,4,6 on the
    // other. The count and face assignment are source-described; the bar
    // diameter, pitch, registration, caliper, clearance, and end overhang are
    // explicit engineering prototype choices (see pin_adhesion_fixture_profile.json).
    spec.totalPinCount = 6u;
    spec.firstBodyIndex = firstBodyIndex;
    spec.dimensionBasis = PinDimensionBasis::engineeringPrototype;
    spec.pinCountBasis = PinCountBasis::articleDescribedSixPins;
    spec.arrayRegistration = PinArrayRegistration::centeredInSpecimenWidth;
    spec.seriesFaceAssignment =
        PinSeriesFaceAssignment::pins135RemainingPins246Selected;
    // This maps the article's unlabeled 50 x 160 mm pin coupon onto the local
    // board frame. It is explicit and remains an engineering orientation.
    spec.specimenWidthLengthAxesAuthored = true;
    return spec;
}

inline PinAdhesionFixtureGeometry makePinAdhesionFixture(
    const PinAdhesionFixtureSpec& spec) {
    if (!pinFinitePositive(spec.specimenWidthM) ||
        !pinFinitePositive(spec.specimenLengthM) ||
        !pinFinitePositive(spec.relativePlatenRateMPerSecond) ||
        !pinFinitePositive(spec.boardCaliperM) ||
        !pinFinitePositive(spec.pinPitchAcrossWidthM) ||
        !pinFinitePositive(spec.pinRadiusM) ||
        !std::isfinite(spec.pinEndOverhangM) ||
        spec.pinEndOverhangM < 0.0 ||
        !std::isfinite(spec.initialClearanceM) ||
        spec.initialClearanceM < 0.0 ||
        spec.totalPinCount < 2u ||
        spec.firstBodyIndex == NM_INVALID_INDEX ||
        spec.dimensionBasis == PinDimensionBasis::unspecified ||
        spec.pinCountBasis == PinCountBasis::unspecified ||
        spec.arrayRegistration == PinArrayRegistration::unspecified ||
        spec.seriesFaceAssignment == PinSeriesFaceAssignment::unspecified ||
        !spec.specimenWidthLengthAxesAuthored) {
        throw std::invalid_argument(
            "pin fixture requires positive authored dimensions, width/length axes, "
            "alternating-pin count, body IDs, and a dimension provenance tag");
    }
    if (spec.pinCountBasis == PinCountBasis::articleDescribedSixPins &&
        spec.totalPinCount != 6u) {
        throw std::invalid_argument(
            "the article-described pin series names pins 1 through 6");
    }
    if (spec.arrayRegistration ==
            PinArrayRegistration::centeredInSpecimenWidth &&
        spec.pinArrayCenterXM != 0.0) {
        throw std::invalid_argument(
            "centered pin registration requires a zero center offset");
    }
    const std::uint64_t lastBody =
        static_cast<std::uint64_t>(spec.firstBodyIndex) +
        static_cast<std::uint64_t>(spec.totalPinCount) - 1u;
    if (lastBody >= NM_INVALID_INDEX) {
        throw std::invalid_argument("pin fixture body-index range overflows");
    }
    const double pinArraySpan =
        static_cast<double>(spec.totalPinCount - 1u) *
        spec.pinPitchAcrossWidthM;
    const double pinArrayCenter =
        spec.arrayRegistration == PinArrayRegistration::centeredInSpecimenWidth
            ? 0.0 : spec.pinArrayCenterXM;
    const double pinArrayHalfEnvelope =
        0.5 * pinArraySpan + spec.pinRadiusM;
    if (!std::isfinite(pinArrayCenter) ||
        pinArrayHalfEnvelope > 0.5 * spec.specimenWidthM ||
        std::abs(pinArrayCenter) + pinArrayHalfEnvelope >
            0.5 * spec.specimenWidthM) {
        throw std::invalid_argument(
            "alternating pin bodies extend beyond the authored coupon width");
    }
    const double halfCapsuleCenterline =
        0.5 * spec.specimenLengthM + spec.pinEndOverhangM - spec.pinRadiusM;
    if (!(halfCapsuleCenterline > 0.0) || !std::isfinite(halfCapsuleCenterline)) {
        throw std::invalid_argument(
            "pin bar is too short for the rounded end radii");
    }

    PinAdhesionFixtureGeometry result;
    result.spec = spec;
    result.sourceExactFixtureGeometry = false;
    result.pinOverallLengthM =
        2.0 * halfCapsuleCenterline + 2.0 * spec.pinRadiusM;
    result.firstCenterXM = pinArrayCenter - 0.5 * pinArraySpan;
    result.lastCenterXM = pinArrayCenter + 0.5 * pinArraySpan;
    result.pins.reserve(spec.totalPinCount);

    const double positiveSide = spec.selectedFacingIsNegativeZ ? 1.0 : -1.0;
    const double centerOffset = 0.5 * spec.boardCaliperM +
        spec.pinRadiusM + spec.initialClearanceM;
    for (std::uint32_t ordinal = 0u;
         ordinal < spec.totalPinCount; ++ordinal) {
        const bool firstSeries = ordinal % 2u == 0u;
        const bool firstSeriesContactsRemainingBoard =
            spec.seriesFaceAssignment ==
            PinSeriesFaceAssignment::pins135RemainingPins246Selected;
        const bool contactsRemainingBoard =
            firstSeries == firstSeriesContactsRemainingBoard;
        const PinPlatenGroup group = contactsRemainingBoard
            ? PinPlatenGroup::remainingBoard
            : PinPlatenGroup::selectedFacing;
        const double x = result.firstCenterXM +
            static_cast<double>(ordinal) * spec.pinPitchAcrossWidthM;
        const double groupSign = contactsRemainingBoard
            ? positiveSide : -positiveSide;

        PinAdhesionPin pin;
        pin.ordinal = ordinal;
        pin.group = group;
        pin.initialBodyTranslationM = {x, 0.0, groupSign * centerOffset};
        pin.proxy.shape = NM_RIGID_CAPSULE;
        pin.proxy.bodyIndex = spec.firstBodyIndex + ordinal;
        pin.proxy.materialIndex = spec.contactMaterialIndex;
        // The capsule axis follows coupon length (y); pin centers alternate
        // across coupon width (x). Radius/ends and pitch are engineering inputs.
        pin.proxy.localCenter = {0.0, -halfCapsuleCenterline, 0.0};
        pin.proxy.localExtent = {0.0, halfCapsuleCenterline, 0.0};
        pin.proxy.localOrientation = {0.0, 0.0, 0.0, 1.0};
        pin.proxy.radiusOrOffset = spec.pinRadiusM;
        pin.proxy.articulated = false;
        pin.proxy.dynamic = false;
        result.pins.push_back(pin);
        if (contactsRemainingBoard)
            ++result.remainingBoardPinCount;
        else
            ++result.selectedFacingPinCount;
    }
    return result;
}

inline std::array<double, 3> pinBodyTranslationAtOpening(
    const PinAdhesionFixtureGeometry& fixture,
    const PinAdhesionPin& pin,
    const double relativeOpeningM) {
    if (!std::isfinite(relativeOpeningM) || relativeOpeningM < 0.0) {
        throw std::invalid_argument(
            "pin-fixture separation must be finite and nonnegative");
    }
    const double split = fixture.spec.splitRelativeOpeningSymmetrically
        ? 0.5 : 1.0;
    const double awaySign = fixture.spec.selectedFacingIsNegativeZ ? 1.0 : -1.0;
    const double signedMove = pin.group == PinPlatenGroup::remainingBoard
        ? awaySign * split * relativeOpeningM
        : -awaySign * split * relativeOpeningM;
    return {
        pin.initialBodyTranslationM[0],
        pin.initialBodyTranslationM[1],
        pin.initialBodyTranslationM[2] + signedMove,
    };
}

inline std::array<double, 3> pinBodyTranslationAtSeparation(
    const PinAdhesionFixtureGeometry& fixture,
    const PinAdhesionPin& pin,
    const double relativeOpeningM) {
    if (!std::isfinite(relativeOpeningM) || relativeOpeningM < 0.0) {
        throw std::invalid_argument(
            "pin-fixture separation must be finite and nonnegative");
    }
    const double awaySign = fixture.spec.selectedFacingIsNegativeZ ? 1.0 : -1.0;
    const double groupSign = pin.group == PinPlatenGroup::remainingBoard
        ? awaySign : -awaySign;
    const double zeroGapZ = pin.initialBodyTranslationM[2] -
        groupSign * fixture.spec.initialClearanceM;
    return {
        pin.initialBodyTranslationM[0],
        pin.initialBodyTranslationM[1],
        zeroGapZ + groupSign * 0.5 * relativeOpeningM,
    };
}

inline const char* pinAdhesionPhaseName(const PinAdhesionPhase phase) {
    switch (phase) {
    case PinAdhesionPhase::platenApproach: return "platen_approach";
    case PinAdhesionPhase::relativeOpening: return "relative_opening";
    case PinAdhesionPhase::platenWithdrawal: return "platen_withdrawal";
    case PinAdhesionPhase::releasedHold: return "released_hold";
    }
    return "invalid";
}

inline PinAdhesionProtocolSchedule makePinAdhesionProtocolSchedule(
    const PinAdhesionFixtureGeometry& fixture,
    const double timestepSeconds,
    const double maximumRelativeOpeningM,
    const double withdrawalDistanceM,
    const std::uint64_t maximumStepCount = 200000u) {
    if (!(std::isfinite(timestepSeconds) && timestepSeconds > 0.0) ||
        !(std::isfinite(maximumRelativeOpeningM) &&
          maximumRelativeOpeningM > 0.0) ||
        !(std::isfinite(withdrawalDistanceM) && withdrawalDistanceM > 0.0) ||
        maximumStepCount == 0u ||
        !(std::isfinite(fixture.spec.relativePlatenRateMPerSecond) &&
          fixture.spec.relativePlatenRateMPerSecond > 0.0)) {
        throw std::invalid_argument(
            "pin protocol requires positive finite timestep, opening, withdrawal, rate, and step bound");
    }

    PinAdhesionProtocolSchedule schedule;
    schedule.timestepSeconds = timestepSeconds;
    schedule.requestedRelativeSpeedMPerSecond =
        fixture.spec.relativePlatenRateMPerSecond;
    schedule.maximumRelativeOpeningM = maximumRelativeOpeningM;
    schedule.withdrawalDistanceM = withdrawalDistanceM;
    const auto appendLeg = [&](const PinAdhesionPhase phase,
                               const double startM, const double endM) {
        const double distance = std::abs(endM - startM);
        if (distance == 0.0) return;
        const double rawCount = std::ceil(distance /
            (schedule.requestedRelativeSpeedMPerSecond * timestepSeconds));
        if (!(std::isfinite(rawCount) && rawCount >= 1.0) ||
            rawCount > static_cast<double>(maximumStepCount) ||
            schedule.points.size() + static_cast<std::size_t>(rawCount) + 1u >
                maximumStepCount) {
            throw std::length_error(
                "pin protocol exceeds its bounded accepted-step count at the authored rate and timestep");
        }
        const auto count = static_cast<std::uint32_t>(rawCount);
        const double legStartTime = schedule.totalDurationSeconds;
        const double delta = endM - startM;
        for (std::uint32_t step = 0u; step < count; ++step) {
            const double a = static_cast<double>(step) / count;
            const double b = static_cast<double>(step + 1u) / count;
            const double from = startM + a * delta;
            const double to = startM + b * delta;
            schedule.points.push_back({
                static_cast<std::uint64_t>(schedule.points.size()), phase,
                step, legStartTime + static_cast<double>(step + 1u) * timestepSeconds,
                from, to, (to - from) / timestepSeconds});
        }
        schedule.totalDurationSeconds +=
            static_cast<double>(count) * timestepSeconds;
    };

    appendLeg(PinAdhesionPhase::platenApproach,
              2.0 * fixture.spec.initialClearanceM, 0.0);
    appendLeg(PinAdhesionPhase::relativeOpening, 0.0,
              maximumRelativeOpeningM);
    appendLeg(PinAdhesionPhase::platenWithdrawal,
              maximumRelativeOpeningM,
              maximumRelativeOpeningM + withdrawalDistanceM);
    if (schedule.points.size() >= maximumStepCount)
        throw std::length_error("pin protocol leaves no step for released hold");
    schedule.points.push_back({
        static_cast<std::uint64_t>(schedule.points.size()),
        PinAdhesionPhase::releasedHold, 0u,
        schedule.totalDurationSeconds + timestepSeconds,
        maximumRelativeOpeningM + withdrawalDistanceM,
        maximumRelativeOpeningM + withdrawalDistanceM, 0.0});
    schedule.totalDurationSeconds += timestepSeconds;
    if (schedule.points.empty() ||
        schedule.points.back().phase != PinAdhesionPhase::releasedHold ||
        schedule.points.back().controlStep + 1u != schedule.points.size())
        throw std::logic_error("pin protocol schedule failed its terminal release contract");
    return schedule;
}

inline double pinRelativeOpeningAtTime(
    const PinAdhesionFixtureGeometry& fixture,
    const double elapsedSeconds) {
    if (!std::isfinite(elapsedSeconds) || elapsedSeconds < 0.0) {
        throw std::invalid_argument(
            "pin-fixture elapsed time must be finite and nonnegative");
    }
    return fixture.spec.relativePlatenRateMPerSecond * elapsedSeconds;
}

inline PinAdhesionTraceMetrics summarizePinAdhesionTrace(
    const PinAdhesionFixtureGeometry& fixture,
    const std::vector<PinAdhesionAcceptedSample>& samples,
    const bool sourceWindowComplete) {
    PinAdhesionTraceMetrics result;
    result.sourceWindowComplete = sourceWindowComplete;
    bool havePoint = false;
    double previousOpening = 0.0;
    double previousForce = 0.0;
    for (const PinAdhesionAcceptedSample& sample : samples) {
        if (!sample.accepted) continue;
        if (!std::isfinite(sample.relativeOpeningM) ||
            sample.relativeOpeningM < 0.0 ||
            !std::isfinite(sample.instrumentLoadMagnitudeN) ||
            sample.instrumentLoadMagnitudeN < 0.0) {
            throw std::invalid_argument(
                "accepted pin trace contains invalid opening or reaction");
        }
        if (havePoint && sample.relativeOpeningM < previousOpening) {
            throw std::invalid_argument(
                "pin adhesion trace must be a monotone opening history");
        }
        if (!havePoint) {
            result.firstOpeningM = sample.relativeOpeningM;
            result.peakMeasuredLoadN = sample.instrumentLoadMagnitudeN;
            result.peakAcceptedStep = sample.acceptedStep;
            havePoint = true;
        } else {
            const double deltaOpening =
                sample.relativeOpeningM - previousOpening;
            result.integratedWorkNm += 0.5 *
                (previousForce + sample.instrumentLoadMagnitudeN) * deltaOpening;
        }
        if (sample.instrumentLoadMagnitudeN > result.peakMeasuredLoadN) {
            result.peakMeasuredLoadN = sample.instrumentLoadMagnitudeN;
            result.peakAcceptedStep = sample.acceptedStep;
        }
        previousOpening = sample.relativeOpeningM;
        previousForce = sample.instrumentLoadMagnitudeN;
        result.lastOpeningM = sample.relativeOpeningM;
        ++result.acceptedPointCount;
    }
    if (result.acceptedPointCount < 2u) {
        throw std::invalid_argument(
            "pin trace summary requires at least two accepted samples");
    }
    result.integratedWorkNmm = 1000.0 * result.integratedWorkNm;
    result.pinStrengthNPerMm =
        result.peakMeasuredLoadN / (fixture.spec.specimenWidthM * 1000.0);
    return result;
}

inline double pinTraceSecantStiffnessNPerM(
    const PinAdhesionAcceptedSample& start,
    const PinAdhesionAcceptedSample& end) {
    if (!start.accepted || !end.accepted ||
        !std::isfinite(start.relativeOpeningM) ||
        !std::isfinite(end.relativeOpeningM) ||
        !std::isfinite(start.instrumentLoadMagnitudeN) ||
        !std::isfinite(end.instrumentLoadMagnitudeN) ||
        !(end.relativeOpeningM > start.relativeOpeningM)) {
        throw std::invalid_argument(
            "stiffness secant requires two accepted, ordered, distinct points");
    }
    return (end.instrumentLoadMagnitudeN - start.instrumentLoadMagnitudeN) /
        (end.relativeOpeningM - start.relativeOpeningM);
}

} // namespace numi::cardboard
