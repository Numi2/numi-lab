#include "cardboard_pin_adhesion_fixture.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace numi::cardboard;

unsigned checks = 0u;

void require(const bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

bool near(const double a, const double b, const double tolerance) {
    return std::abs(a - b) <= tolerance;
}

template <class Function>
void mustThrow(Function&& function, const std::string& message) {
    bool threw = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, message);
}

template <class Function>
void mustThrowLength(Function&& function, const std::string& message) {
    bool threw = false;
    try {
        function();
    } catch (const std::length_error&) {
        threw = true;
    }
    require(threw, message);
}

void geometryAndScheduleContract() {
    constexpr std::uint32_t firstBody = 20u;
    const PinAdhesionFixtureSpec spec =
        pinAdhesionEngineeringPrototype(firstBody);
    const PinAdhesionFixtureGeometry fixture =
        makePinAdhesionFixture(spec);

    require(fixture.pins.size() == 6u &&
                fixture.remainingBoardPinCount == 3u &&
                fixture.selectedFacingPinCount == 3u,
            "six conceptual alternating pins must create two 3-pin platen groups");
    require(!fixture.sourceExactFixtureGeometry &&
                spec.dimensionBasis == PinDimensionBasis::engineeringPrototype &&
                spec.pinCountBasis == PinCountBasis::articleDescribedSixPins &&
                spec.arrayRegistration ==
                    PinArrayRegistration::centeredInSpecimenWidth &&
                spec.seriesFaceAssignment ==
                    PinSeriesFaceAssignment::pins135RemainingPins246Selected,
            "source-described pin count/alternation and engineering geometry must have separate provenance");
    require(near(spec.specimenWidthM, 0.050, 1.0e-15) &&
                near(spec.specimenLengthM, 0.160, 1.0e-15) &&
                near(spec.relativePlatenRateMPerSecond, 0.0125 / 60.0, 1.0e-15),
            "published 50 x 160 mm coupon and 12.5 mm/min relative rate must be exact");
    require(near(fixture.firstCenterXM, -0.020, 1.0e-15) &&
                near(fixture.lastCenterXM, 0.020, 1.0e-15) &&
                near(fixture.pinOverallLengthM, 0.162, 1.0e-15) &&
                near(2.0 * spec.pinRadiusM, 0.001, 1.0e-15),
            "engineering bars must use the declared 1 mm diameter, centered 8 mm pitch, and 1 mm end overhang");

    for (std::uint32_t i = 0u; i < fixture.pins.size(); ++i) {
        const PinAdhesionPin& pin = fixture.pins[i];
        require(pin.proxy.shape == NM_RIGID_CAPSULE &&
                    pin.proxy.bodyIndex == firstBody + i &&
                    !pin.proxy.dynamic && !pin.proxy.articulated,
                "each physical pin bar must use a distinct kinematic capsule body");
        require(near(pin.proxy.localCenter[0], 0.0, 1.0e-15) &&
                    near(pin.proxy.localExtent[0], 0.0, 1.0e-15) &&
                    pin.proxy.localCenter[1] < 0.0 &&
                    pin.proxy.localExtent[1] > 0.0,
                "pin capsule axes must span the 160 mm coupon length");
        require(pin.group == (i % 2u == 0u
                    ? PinPlatenGroup::remainingBoard
                    : PinPlatenGroup::selectedFacing),
                "pin bars must alternate between the two fixture platens");
    }

    const double opening = 0.001;
    const auto remaining = pinBodyTranslationAtOpening(
        fixture, fixture.pins[0], opening);
    const auto selected = pinBodyTranslationAtOpening(
        fixture, fixture.pins[1], opening);
    require(near(remaining[2] - fixture.pins[0].initialBodyTranslationM[2],
                 0.5 * opening, 1.0e-15) &&
                near(selected[2] - fixture.pins[1].initialBodyTranslationM[2],
                     -0.5 * opening, 1.0e-15),
            "symmetric platen pose targets must create the commanded relative opening");
    require(near(pinRelativeOpeningAtTime(fixture, 60.0), 0.0125, 1.0e-15),
            "one minute at the source rate must increase relative separation by 12.5 mm");

    PinAdhesionFixtureSpec reverse = spec;
    reverse.selectedFacingIsNegativeZ = false;
    const auto reversed = makePinAdhesionFixture(reverse);
    require(reversed.pins[0].initialBodyTranslationM[2] < 0.0 &&
                reversed.pins[1].initialBodyTranslationM[2] > 0.0,
            "selected-facing side must reverse the authored fixture orientation");

    const auto offsetSpec = [&] {
        auto value = spec;
        value.arrayRegistration = PinArrayRegistration::explicitCenterOffset;
        value.pinArrayCenterXM = 0.004;
        return value;
    }();
    const auto offsetFixture = makePinAdhesionFixture(offsetSpec);
    require(near(offsetFixture.firstCenterXM, -0.016, 1.0e-15) &&
                near(offsetFixture.lastCenterXM, 0.024, 1.0e-15),
            "externally authored pin-array registration must be retained exactly");
}

void traceMetricContract() {
    const auto fixture = makePinAdhesionFixture(
        pinAdhesionEngineeringPrototype(20u));
    const std::vector<PinAdhesionAcceptedSample> trace{
        {0u, 0.000, 0.0, true},
        {1u, 0.001, 100.0, true},
        {2u, 0.0015, 900.0, false}, // rolled-back attempt must not enter metrics
        {3u, 0.002, 50.0, true},
    };
    const auto metrics = summarizePinAdhesionTrace(fixture, trace, false);
    require(metrics.acceptedPointCount == 3u &&
                metrics.peakAcceptedStep == 1u &&
                near(metrics.peakMeasuredLoadN, 100.0, 1.0e-15),
            "only accepted history may contribute to peak force");
    require(near(metrics.integratedWorkNm, 0.125, 1.0e-15) &&
                near(metrics.integratedWorkNmm, 125.0, 1.0e-12),
            "work must be trapezoidal total force times relative opening in consistent units");
    require(near(metrics.pinStrengthNPerMm, 2.0, 1.0e-15) &&
                !metrics.sourceWindowComplete,
            "strength normalization must use the 50 mm authored coupon width and preserve incompleteness");
    require(near(pinTraceSecantStiffnessNPerM(trace[0], trace[1]),
                 100000.0, 1.0e-12),
            "secant stiffness must retain SI units and an explicitly selected interval");
    mustThrow([&] {
        (void)pinTraceSecantStiffnessNPerM(trace[1], trace[0]);
    }, "reversed stiffness window must fail closed");
    mustThrow([&] {
        const std::vector<PinAdhesionAcceptedSample> unloading{
            {0u, 0.001, 10.0, true}, {1u, 0.0009, 9.0, true}};
        (void)summarizePinAdhesionTrace(fixture, unloading, true);
    }, "monotone source loading metric must reject unloading histories");
}

void nativeProtocolScheduleContract() {
    const auto fixture = makePinAdhesionFixture(
        pinAdhesionEngineeringPrototype(0u));
    const auto schedule = makePinAdhesionProtocolSchedule(
        fixture, 0.005, 0.002, 0.001);
    require(!schedule.points.empty() &&
                schedule.points.front().phase == PinAdhesionPhase::platenApproach &&
                schedule.points.back().phase == PinAdhesionPhase::releasedHold,
            "native protocol must include approach and a released terminal hold");
    const auto lastApproach = std::find_if(
        schedule.points.rbegin(), schedule.points.rend(),
        [](const auto& point) {
            return point.phase == PinAdhesionPhase::platenApproach;
        });
    require(near(schedule.points.front().startRelativeOpeningM,
                 2.0 * fixture.spec.initialClearanceM, 1.0e-15) &&
                lastApproach != schedule.points.rend() &&
                lastApproach->endRelativeOpeningM == 0.0,
            "approach must close exactly the two authored surface clearances");
    require(near(schedule.points.back().startRelativeOpeningM,
                 0.003, 1.0e-15) &&
                schedule.points.back().startRelativeOpeningM ==
                    schedule.points.back().endRelativeOpeningM &&
                schedule.points.back().effectiveRelativeSpeedMPerSecond == 0.0,
            "withdrawal must end contact travel before the stationary release stage");
    require(std::any_of(schedule.points.begin(), schedule.points.end(),
                [](const auto& point) {
                    return point.phase == PinAdhesionPhase::relativeOpening;
                }) &&
                std::any_of(schedule.points.begin(), schedule.points.end(),
                [](const auto& point) {
                    return point.phase == PinAdhesionPhase::platenWithdrawal;
                }),
            "opening and platen withdrawal must be separate recorded phases");
    for (std::size_t index = 0u; index < schedule.points.size(); ++index) {
        const auto& point = schedule.points[index];
        require(point.controlStep == index && std::isfinite(point.elapsedSeconds) &&
                    point.elapsedSeconds > 0.0 &&
                    std::abs(point.effectiveRelativeSpeedMPerSecond) <=
                        schedule.requestedRelativeSpeedMPerSecond *
                            (1.0 + 1.0e-12),
                "pin schedule steps must be contiguous, finite, and rate bounded");
        if (index > 0u) {
            require(near(schedule.points[index - 1u].endRelativeOpeningM,
                         point.startRelativeOpeningM, 1.0e-15) &&
                        near(schedule.points[index - 1u].elapsedSeconds +
                                 schedule.timestepSeconds,
                             point.elapsedSeconds, 1.0e-12),
                    "pin phase boundaries must preserve pose and time continuity");
        }
    }
    const auto initialRemaining = pinBodyTranslationAtSeparation(
        fixture, fixture.pins[0], 2.0 * fixture.spec.initialClearanceM);
    const auto contactRemaining = pinBodyTranslationAtSeparation(
        fixture, fixture.pins[0], 0.0);
    const auto contactSelected = pinBodyTranslationAtSeparation(
        fixture, fixture.pins[1], 0.0);
    require(near(initialRemaining[2],
                 fixture.pins[0].initialBodyTranslationM[2], 1.0e-15) &&
                near(contactRemaining[2], 0.003, 1.0e-15) &&
                near(contactSelected[2], -0.003, 1.0e-15),
            "pose path must approach the two board faces without interior FEM targets");
    mustThrow([&] {
        (void)makePinAdhesionProtocolSchedule(fixture, 0.0, 0.002, 0.001);
    }, "zero timestep must not produce a native pin protocol");
    mustThrow([&] {
        (void)makePinAdhesionProtocolSchedule(fixture, 0.005, 0.0, 0.001);
    }, "zero opening must not silently remove the tensile phase");
    mustThrowLength([&] {
        (void)makePinAdhesionProtocolSchedule(fixture, 1.0, 1000.0, 1.0, 8u);
    }, "oversized pin protocol must fail before native allocation or stepping");
}

void inputAdmissionContract() {
    PinAdhesionFixtureSpec missing;
    missing.firstBodyIndex = 1u;
    mustThrow([&] { (void)makePinAdhesionFixture(missing); },
              "missing pin dimension provenance and geometry must not be guessed");

    auto invalid = pinAdhesionEngineeringPrototype(1u);
    invalid.totalPinCount = 8u;
    mustThrow([&] { (void)makePinAdhesionFixture(invalid); },
              "pin row extending beyond the 50 mm coupon must be rejected");

    invalid = pinAdhesionEngineeringPrototype(NM_INVALID_INDEX - 2u);
    mustThrow([&] { (void)makePinAdhesionFixture(invalid); },
              "pin body index overflow must be rejected");

    invalid = pinAdhesionEngineeringPrototype(1u);
    invalid.specimenWidthLengthAxesAuthored = false;
    mustThrow([&] { (void)makePinAdhesionFixture(invalid); },
              "unresolved source width/length orientation must be authored explicitly");

    invalid = pinAdhesionEngineeringPrototype(1u);
    invalid.totalPinCount = 5u;
    mustThrow([&] { (void)makePinAdhesionFixture(invalid); },
              "source-described six-pin topology must not be silently changed");

    invalid = pinAdhesionEngineeringPrototype(1u);
    invalid.pinRadiusM = 0.006;
    mustThrow([&] { (void)makePinAdhesionFixture(invalid); },
              "pin body radius must fit inside the coupon, not only the pin centers");

    invalid = pinAdhesionEngineeringPrototype(1u);
    invalid.pinArrayCenterXM = 0.001;
    mustThrow([&] { (void)makePinAdhesionFixture(invalid); },
              "centered registration must reject a hidden nonzero offset");

    const auto fixture = makePinAdhesionFixture(
        pinAdhesionEngineeringPrototype(1u));
    mustThrow([&] { (void)pinRelativeOpeningAtTime(fixture, -1.0); },
              "negative elapsed time must be rejected");
    mustThrow([&] {
        (void)pinBodyTranslationAtOpening(fixture, fixture.pins[0], -0.001);
    }, "negative opening must be rejected in the monotone tensile protocol");
}

} // namespace

int main() {
    try {
        geometryAndScheduleContract();
        traceMetricContract();
        nativeProtocolScheduleContract();
        inputAdmissionContract();
        std::cout << "cardboard_pin_adhesion_fixture_check: " << checks
                  << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cardboard_pin_adhesion_fixture_check: " << error.what()
                  << "\n";
        return 1;
    }
}
