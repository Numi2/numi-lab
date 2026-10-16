#include "cardboard_box_assembly_controller.hpp"
#include "cardboard_shipping_crush.hpp"
#include "cardboard_shipping_capacity.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
using namespace numi::cardboard;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

AcceptedScoreAllowanceCoupon syntheticCoupon() {
    AcceptedScoreAllowanceCoupon coupon;
    coupon.accepted = true;
    coupon.evidenceClass = CouponEvidenceClass::syntheticUnitFixture;
    coupon.couponId = "synthetic-zero-offset-unit-only";
    coupon.boardRecipeId = "literature2009";
    coupon.sourceRunId = "synthetic-unit-test";
    coupon.measurementSha256 = std::string(64u, 'a');
    coupon.scoreCenterSpanMinusInsideM = {0.0, 0.0, 0.0};
    return coupon;
}

double sixVolume(const BoxBlankPoint& a,
                 const BoxBlankPoint& b,
                 const BoxBlankPoint& c,
                 const BoxBlankPoint& d) {
    const std::array<double, 3> ab{b.x-a.x, b.y-a.y, b.z-a.z};
    const std::array<double, 3> ac{c.x-a.x, c.y-a.y, c.z-a.z};
    const std::array<double, 3> ad{d.x-a.x, d.y-a.y, d.z-a.z};
    return ab[0]*(ac[1]*ad[2]-ac[2]*ad[1]) -
        ab[1]*(ac[0]*ad[2]-ac[2]*ad[0]) +
        ab[2]*(ac[0]*ad[1]-ac[1]*ad[0]);
}

template<class F>
void expectInvalid(F&& function, const std::string& message) {
    bool rejected = false;
    try { function(); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, message);
}

void run() {
    ShippingCapacityIdentity capacityIdentity;
    capacityIdentity.boardRecipeId = "literature2009";
    capacityIdentity.sourcePhysicsFingerprint = 11u;
    capacityIdentity.deviceProgramFingerprint = 12u;
    capacityIdentity.compiledWorldFingerprint = 13u;
    capacityIdentity.sourceBindingSha256 = std::string(64u, 'a');
    capacityIdentity.assembledMatterCheckpointSha256 = std::string(64u, 'b');
    capacityIdentity.assembledRigidSidecarSha256 = std::string(64u, 'c');
    capacityIdentity.nominalCurveSha256 = std::string(64u, 'd');
    capacityIdentity.nominalPeakSampleIndex = 7u;
    capacityIdentity.nominalAcceptedSampleCount = 281u;
    capacityIdentity.nominalTerminalStrokeFraction = 0.70;
    capacityIdentity.nominalFullStrokeCompleted = true;
    capacityIdentity.nominalRateSensitivityPassed = true;
    ShippingCapacityQualification capacityEvidence;
    capacityEvidence.schema = kShippingCapacityQualificationSchema;
    capacityEvidence.status = "qualified_for_numerical_capacity_only";
    capacityEvidence.boardRecipeId = capacityIdentity.boardRecipeId;
    capacityEvidence.sourcePhysicsFingerprint = 11u;
    capacityEvidence.deviceProgramFingerprint = 12u;
    capacityEvidence.compiledWorldFingerprint = 13u;
    capacityEvidence.sourceBindingSha256 = capacityIdentity.sourceBindingSha256;
    capacityEvidence.assembledMatterCheckpointSha256 =
        capacityIdentity.assembledMatterCheckpointSha256;
    capacityEvidence.assembledRigidSidecarSha256 =
        capacityIdentity.assembledRigidSidecarSha256;
    capacityEvidence.nominalCurveSha256 = capacityIdentity.nominalCurveSha256;
    capacityEvidence.nominalPeakSampleIndex = 7u;
    capacityEvidence.nominalAcceptedSampleCount = 281u;
    capacityEvidence.nominalTerminalStrokeFraction = 0.70;
    capacityEvidence.nominalFullStrokeCompleted = true;
    capacityEvidence.nominalRateSensitivityPassed = true;
    capacityEvidence.meshStatus = "qualified";
    capacityEvidence.meshReportFile = "mesh-study.json";
    capacityEvidence.meshReportSha256 = std::string(64u, 'e');
    capacityEvidence.meshLevelCount = 3u;
    capacityEvidence.meshAllCasesComplete = true;
    capacityEvidence.meshFinestPairWithinTolerance = true;
    capacityEvidence.meshFinestPeakDifference = 0.01;
    capacityEvidence.meshFinestCurveRms = 0.02;
    capacityEvidence.meshFinestWorkDifference = 0.01;
    capacityEvidence.meshCoarseMiddlePeakDifference = 0.02;
    capacityEvidence.meshCoarseMiddleCurveRms = 0.03;
    capacityEvidence.meshCoarseMiddleWorkDifference = 0.02;
    capacityEvidence.timeStatus = "qualified";
    capacityEvidence.timeReportFile = "time-study.json";
    capacityEvidence.timeReportSha256 = std::string(64u, 'f');
    capacityEvidence.timeLevelCount = 3u;
    capacityEvidence.timeAllCasesComplete = true;
    capacityEvidence.timeFinestPairWithinTolerance = true;
    capacityEvidence.timeFinestPeakDifference = 0.01;
    capacityEvidence.timeFinestCurveRms = 0.02;
    capacityEvidence.timeFinestWorkDifference = 0.01;
    capacityEvidence.timeCoarseMiddlePeakDifference = 0.02;
    capacityEvidence.timeCoarseMiddleCurveRms = 0.03;
    capacityEvidence.timeCoarseMiddleWorkDifference = 0.02;
    require(validateShippingCapacityQualification(
                capacityEvidence, capacityIdentity).qualified,
            "complete source-bound mesh/time qualification was rejected");
    auto injected = capacityEvidence;
    injected.nominalCurveSha256 = std::string(64u, '0');
    require(!validateShippingCapacityQualification(injected, capacityIdentity).qualified,
            "qualification admitted a different nominal force curve");
    injected = capacityEvidence;
    injected.assembledRigidSidecarSha256 = std::string(64u, '0');
    require(!validateShippingCapacityQualification(injected, capacityIdentity).qualified,
            "qualification admitted a different assembled rigid checkpoint");
    injected = capacityEvidence;
    injected.meshStatus = "descriptive_three_level_decrease_only";
    require(!validateShippingCapacityQualification(injected, capacityIdentity).qualified,
            "descriptive mesh output was treated as a convergence verdict");
    injected = capacityEvidence;
    injected.timeAllCasesComplete = false;
    require(!validateShippingCapacityQualification(injected, capacityIdentity).qualified,
            "incomplete timestep cases were treated as numerical capacity evidence");
    injected = capacityEvidence;
    injected.meshFinestCurveRms = 0.051;
    require(!validateShippingCapacityQualification(injected, capacityIdentity).qualified,
            "mesh force-curve difference above tolerance was admitted");
    injected = capacityEvidence;
    injected.meshCoarseMiddlePeakDifference = 0.005;
    require(!validateShippingCapacityQualification(injected, capacityIdentity).qualified,
            "mesh triplet whose coarse-to-middle difference is smaller than middle-to-fine was admitted");
    injected = capacityEvidence;
    injected.timeCoarseMiddleCurveRms = 0.005;
    require(!validateShippingCapacityQualification(injected, capacityIdentity).qualified,
            "time triplet with no adjacent-difference decrease was admitted");
    injected = capacityEvidence;
    injected.meshCoarseMiddleWorkDifference = injected.meshFinestWorkDifference;
    require(!validateShippingCapacityQualification(injected, capacityIdentity).qualified,
            "one decreasing metric concealed an unchanged mesh work difference");
    injected = capacityEvidence;
    injected.timeCoarseMiddlePeakDifference = injected.timeFinestPeakDifference;
    require(!validateShippingCapacityQualification(injected, capacityIdentity).qualified,
            "one decreasing metric concealed an unchanged timestep peak difference");
    injected = capacityEvidence;
    injected.nominalTerminalStrokeFraction = 0.5;
    require(!validateShippingCapacityQualification(injected, capacityIdentity).qualified,
            "peak-only compression prefix was admitted without the post-peak endpoint");

    std::ostringstream capacityProperties;
    capacityProperties
        << "schema=" << kShippingCapacityQualificationSchema << '\n'
        << "status=qualified_for_numerical_capacity_only\n"
        << "board_recipe_id=literature2009\n"
        << "source_physics_fingerprint=11\n"
        << "device_program_fingerprint=12\n"
        << "compiled_world_fingerprint=13\n"
        << "source_binding_sha256=" << std::string(64u, 'a') << '\n'
        << "assembled_matter_checkpoint_sha256=" << std::string(64u, 'b') << '\n'
        << "assembled_rigid_sidecar_sha256=" << std::string(64u, 'c') << '\n'
        << "nominal_curve_sha256=" << std::string(64u, 'd') << '\n'
        << "nominal_peak_sample_index=7\n"
        << "nominal_accepted_sample_count=281\n"
        << "nominal_terminal_stroke_fraction=0.70\n"
        << "nominal_full_stroke_completed=true\n"
        << "nominal_rate_sensitivity_passed=true\n"
        << "mesh_status=qualified\nmesh_report_file=mesh-study.json\n"
        << "mesh_report_sha256=" << std::string(64u, 'e') << '\n'
        << "mesh_level_count=3\nmesh_all_cases_complete=true\n"
        << "mesh_finest_pair_within_tolerance=true\n"
        << "mesh_finest_peak_difference=0.01\nmesh_finest_curve_rms=0.02\n"
        << "mesh_finest_work_difference=0.01\n"
        << "mesh_coarse_middle_peak_difference=0.02\n"
        << "mesh_coarse_middle_curve_rms=0.03\n"
        << "mesh_coarse_middle_work_difference=0.02\n"
        << "time_status=qualified\ntime_report_file=time-study.json\n"
        << "time_report_sha256=" << std::string(64u, 'f') << '\n'
        << "time_level_count=3\ntime_all_cases_complete=true\n"
        << "time_finest_pair_within_tolerance=true\n"
        << "time_finest_peak_difference=0.01\ntime_finest_curve_rms=0.02\n"
        << "time_finest_work_difference=0.01\n"
        << "time_coarse_middle_peak_difference=0.02\n"
        << "time_coarse_middle_curve_rms=0.03\n"
        << "time_coarse_middle_work_difference=0.02\n";
    std::istringstream validCapacityInput(capacityProperties.str());
    const auto parsedCapacity = parseShippingCapacityQualification(validCapacityInput);
    require(validateShippingCapacityQualification(parsedCapacity,
                capacityIdentity).qualified,
            "versioned capacity input parser changed the qualified evidence contract");

    std::istringstream malformedCapacity("schema=x\nschema=y\n");
    expectInvalid([&] {
        (void)parseShippingCapacityQualification(malformedCapacity);
    }, "duplicate capacity-evidence properties were accepted");

    ShippingExecutionGateEvidence gateEvidence;
    gateEvidence.nativeStepAccepted = true;
    gateEvidence.completeSnapshotAvailable = true;
    gateEvidence.solverCertificateValid = true;
    gateEvidence.checkpointSourcePhysicsFingerprint = 0x1234u;
    gateEvidence.expectedSourcePhysicsFingerprint = 0x1234u;
    gateEvidence.checkpointDeviceProgramFingerprint = 0x5678u;
    gateEvidence.expectedDeviceProgramFingerprint = 0x5678u;
    gateEvidence.acceptedStepCount = 1u;
    gateEvidence.checkpointSha256 = std::string(64u, 'a');
    gateEvidence.boardRecipeId = "synthetic-gate-fixture";
    require(admitShippingExecutionPhase(
                ShippingExecutionPhase::scoreIndent, gateEvidence).admitted,
            "execution gate rejected a complete accepted score-start checkpoint");
    require(!admitShippingExecutionPhase(
                ShippingExecutionPhase::impact, gateEvidence).admitted,
            "execution gate admitted impact without released assembly and Fpeak");
    gateEvidence.rejectedStepCount = 1u;
    require(!admitShippingExecutionPhase(
                ShippingExecutionPhase::scoreIndent, gateEvidence).admitted,
            "execution gate ignored a rejected step in checkpoint lineage");
    gateEvidence.rejectedStepCount = 0u;
    gateEvidence.scorePassAccepted = true;
    require(admitShippingExecutionPhase(
                ShippingExecutionPhase::scoreWithdrawal, gateEvidence).admitted,
            "execution gate rejected accepted score withdrawal");
    require(!admitShippingExecutionPhase(
                ShippingExecutionPhase::foldPanels, gateEvidence).admitted,
            "execution gate admitted folds before score tools were withdrawn");
    gateEvidence.scoreToolsWithdrawn = true;
    gateEvidence.foldActionsAccepted = true;
    gateEvidence.manufacturerLapActivated = true;
    gateEvidence.manufacturerLapClosed = true;
    gateEvidence.endOverlapPatchesActivated = 8u;
    gateEvidence.topAndBottomEndsClosed = true;
    require(admitShippingExecutionPhase(
                ShippingExecutionPhase::releaseTemporaryClamps, gateEvidence).admitted,
            "execution gate rejected closed lap and end-flap geometry");
    gateEvidence.temporaryClampsReleased = true;
    gateEvidence.toolingParkedOrRemoved = true;
    gateEvidence.releasedAssemblySettled = true;
    gateEvidence.acceptedCompressionPeakAvailable = true;
    gateEvidence.impactCheckpointRestored = true;
    require(admitShippingExecutionPhase(
                ShippingExecutionPhase::impact, gateEvidence).admitted,
            "execution gate rejected complete released assembly and peak lineage");
    gateEvidence.solverCertificateValid = false;
    require(!admitShippingExecutionPhase(
                ShippingExecutionPhase::impact, gateEvidence).admitted,
            "execution gate admitted a phase with an invalid native certificate");
    gateEvidence.solverCertificateValid = true;
    gateEvidence.checkpointDeviceProgramFingerprint++;
    require(!admitShippingExecutionPhase(
                ShippingExecutionPhase::impact, gateEvidence).admitted,
            "execution gate admitted a checkpoint from a different device program");

    const ShippingBoxDesignDefaults defaults;
    expectInvalid([&] {
        (void)deriveShippingBoxDesign(defaults, AcceptedScoreAllowanceCoupon{});
    }, "shipping design accepted a missing allowance coupon");

    const ShippingBoxDesign design = deriveShippingBoxDesign(defaults, syntheticCoupon());
    require(std::abs(design.blankConfig.panelLength - 0.300) < 1.0e-15 &&
            std::abs(design.blankConfig.wallHeight - 0.200) < 1.0e-15 &&
            std::abs(design.blankConfig.panelWidth - 0.200) < 1.0e-15 &&
            std::abs(design.perimeterSpanM - 1.025) < 1.0e-14 &&
            std::abs(design.blankHeightM - 0.400) < 1.0e-14,
            "coupon-derived 300 x 200 x 200 mm dieline conversion is wrong");
    require(!design.allowanceFromPhysicalCoupon,
            "synthetic zero-offset coupon became physical allowance evidence");

    const GlueMeshConfig sectionConfig = makeShippingCrossSectionConfig(design);
    const auto capacity = auditShippingCrossSectionCapacity(design, sectionConfig);
    require(capacity.compatible && capacity.hasFractionalTerminalPitch &&
            capacity.clippedPitchIntervals == 130u &&
            std::abs(capacity.requestedPitchCount - 129.74683544303798) < 1.0e-10,
            "partial terminal flute was not retained at the fixed recipe pitch");
    GlueMeshConfig legacyRejected = sectionConfig;
    legacyRejected.allowPartialTerminalPitch = false;
    require(!auditShippingCrossSectionCapacity(design, legacyRejected).compatible,
            "legacy whole-pitch mode silently admitted a fractional perimeter");

    const auto depths = shippingScoreDepthsM(ShippingScoreToolDefaults{}, 4.210e-3);
    require(std::abs(depths.front() - 0.421e-3) < 1.0e-15 &&
            std::abs(depths.back() - 1.684e-3) < 1.0e-15,
            "score stroke depths do not scale from board caliper");
    const auto scoreLeg = makeShippingScoreLegSchedule(
        2.1025e-3, 50.0e-3, 25.0e-6, 4096u);
    require(scoreLeg.stepCount == 1682u &&
            std::abs(scoreLeg.durationSeconds - 1682.0 * 25.0e-6) < 1.0e-15 &&
            scoreLeg.effectiveSpeedMPerS <= 50.0e-3 &&
            scoreLeg.effectiveSpeedMPerS > 0.0,
            "shipping score leg did not derive a bounded constant-speed step schedule");
    expectInvalid([&] {
        (void)makeShippingScoreLegSchedule(
            2.1025e-3, 1.0e-3, 25.0e-6, 4096u);
    }, "shipping score leg silently exceeded its configured accepted-step limit");
    expectInvalid([&] {
        (void)makeShippingScoreLegSchedule(
            0.0, 50.0e-3, 25.0e-6, 4096u);
    }, "shipping score leg accepted a zero-length movement");
    ShippingCompressionProtocol nominalCompression;
    require(std::abs(nominalCompression.platenSpeedMPerS - 12.5e-3 / 60.0) < 1.0e-15,
            "nominal shipping compression speed differs from 12.5 mm/min");
    const double quarterMillimetreM = 0.25e-3;
    const double quarterMillimetreDurationS =
        quarterMillimetreM / nominalCompression.platenSpeedMPerS;
    const std::array<double, 3> refinementMaximumDts{
        8.0e-3, 4.0e-3, 2.0e-3};
    std::array<ShippingConstantSpeedMoveSchedule, 3> moveSchedules;
    for (std::size_t level = 0u; level < refinementMaximumDts.size(); ++level) {
        moveSchedules[level] = makeShippingConstantSpeedMoveSchedule(
            quarterMillimetreM, nominalCompression.platenSpeedMPerS,
            refinementMaximumDts[level], 4096u);
        const auto& move = moveSchedules[level];
        double duration = 0.0;
        double distance = 0.0;
        for (const auto& substep : move.substeps) {
            duration += substep.timestepSeconds;
            distance += substep.distanceM;
            require(substep.timestepSeconds > 0.0 &&
                        substep.timestepSeconds <=
                            refinementMaximumDts[level] * (1.0 + 1.0e-12) &&
                        substep.distanceM > 0.0 &&
                        std::abs(substep.distanceM / substep.timestepSeconds -
                                 nominalCompression.platenSpeedMPerS) < 1.0e-12,
                    "shipping timestep refinement changed the commanded constant speed");
        }
        require(std::abs(duration - quarterMillimetreDurationS) < 1.0e-12 &&
                    std::abs(distance - quarterMillimetreM) < 1.0e-15 &&
                    move.substeps.back().completedFraction == 1.0 &&
                    move.maximumUsedTimestepSeconds <=
                        refinementMaximumDts[level] * (1.0 + 1.0e-12),
                "shipping timestep refinement changed the physical duration or endpoint");
    }
    require(moveSchedules[0].substeps.size() == 150u &&
                moveSchedules[1].substeps.size() == 300u &&
                moveSchedules[2].substeps.size() == 600u &&
                moveSchedules[0].maximumUsedTimestepSeconds == 8.0e-3 &&
                moveSchedules[1].maximumUsedTimestepSeconds == 4.0e-3 &&
                moveSchedules[2].maximumUsedTimestepSeconds == 2.0e-3,
            "dt/dt2/dt4 compression refinements did not change accepted substep counts and maximum dt");
    const auto variableLastStep = makeShippingConstantSpeedMoveSchedule(
        0.000125, nominalCompression.platenSpeedMPerS, 0.4, 10u);
    require(variableLastStep.substeps.size() == 2u &&
                variableLastStep.substeps.front().timestepSeconds == 0.4 &&
                variableLastStep.substeps.back().timestepSeconds < 0.4 &&
                std::abs(variableLastStep.substeps.back().distanceM /
                         variableLastStep.substeps.back().timestepSeconds -
                         nominalCompression.platenSpeedMPerS) < 1.0e-12,
            "shipping move did not shorten its final partial timestep at the authored rate");
    bool rejectedRefinementBudget = false;
    try {
        (void)makeShippingConstantSpeedMoveSchedule(
            quarterMillimetreM, nominalCompression.platenSpeedMPerS,
            25.0e-6, 100u);
    } catch (const std::length_error&) {
        rejectedRefinementBudget = true;
    }
    require(rejectedRefinementBudget,
            "shipping move ignored the configured accepted-step budget");
    const auto nominalSchedule = makeShippingCompressionSchedule(
        design.defaults.inside.heightY, nominalCompression);
    ShippingCompressionProtocol halfSpeedCompression = nominalCompression;
    halfSpeedCompression.platenSpeedMPerS = 6.25e-3 / 60.0;
    const auto halfSpeedSchedule = makeShippingCompressionSchedule(
        design.defaults.inside.heightY, halfSpeedCompression);
    const auto scheduledStroke = [](const auto& schedule) {
        double stroke = 0.0;
        for (const auto& increment : schedule) stroke += increment.displacementM;
        return stroke;
    };
    require(nominalSchedule.size() == 560u &&
            halfSpeedSchedule.size() == nominalSchedule.size() &&
            std::abs(nominalSchedule.front().timestepSeconds * 2.0 -
                     halfSpeedSchedule.front().timestepSeconds) < 1.0e-12 &&
            std::abs(scheduledStroke(nominalSchedule) - 0.140) < 1.0e-12 &&
            std::abs(scheduledStroke(halfSpeedSchedule) - 0.140) < 1.0e-12,
            "nominal and half-speed compression protocols changed increment count, stroke, or speed ratio");
    auto afterContactProtocol = nominalCompression;
    afterContactProtocol.targetStrokeFractionOfInsideHeight =
        (0.70 * design.defaults.inside.heightY - 0.25e-3) /
        design.defaults.inside.heightY;
    const auto afterContactSchedule = makeShippingCompressionSchedule(
        design.defaults.inside.heightY, afterContactProtocol);
    require(afterContactSchedule.size() == 559u &&
                std::abs(scheduledStroke(afterContactSchedule) + 0.25e-3 -
                         0.70 * design.defaults.inside.heightY) < 1.0e-12,
            "compression schedule did not count the first contact-crossing increment inside the 70 percent stroke");
    require(nominalCompression.requiredPeakDropFraction == 0.01 &&
                nominalCompression.peakForceEquivalenceToleranceFraction == 0.02 &&
                nominalCompression.workEquivalenceToleranceFraction == 0.02 &&
                nominalCompression.normalizedFullCurveToleranceFraction == 0.05,
            "peak-identification and nominal-versus-half-speed gates were conflated");
    require(shippingCompressionPeakEquivalent(100.0, 102.0, 0.02) &&
                !shippingCompressionPeakEquivalent(100.0, 102.01, 0.02) &&
                !shippingCompressionPeakEquivalent(0.0, 100.0, 0.02),
            "nominal-versus-half-speed peak-equivalence gate did not enforce two percent independently");
    ShippingFirstCompressionPeakDetector peakDetector(0.01, 2u);
    require(!peakDetector.observeAccepted(0.25e-3, 10.0, 1.0) &&
                !peakDetector.observeAccepted(0.50e-3, 20.0, 1.0) &&
                !peakDetector.observeAccepted(0.75e-3, 19.0, 1.0),
            "first compression peak detector qualified before the two-decline evidence existed");
    const auto firstPeak = peakDetector.observeAccepted(1.00e-3, 18.9, 1.0);
    require(firstPeak && firstPeak->peak.strokeM == 0.50e-3 &&
                firstPeak->peak.sampleIndex == 1u &&
                firstPeak->qualifyingDropFraction >= 0.01,
            "compression peak detector did not include the first contact sample in the accepted curve");
    ShippingFirstCompressionPeakDetector fullCurveDetector(0.01, 2u);
    const std::array<ShippingCompressionForceSample, 7> firstPeakCurve{{
        {0u, 0.0000, 0.0, 0.0}, {1u, 0.0010, 40.0, 2.0},
        {2u, 0.0700, 100.0, 4.0}, {3u, 0.0800, 98.8, 4.0},
        {4u, 0.0900, 97.5, 4.0}, {5u, 0.1300, 130.0, 4.0},
        {6u, 0.1400, 80.0, 4.0}}};
    std::optional<ShippingQualifiedCompressionPeak> qualifiedFirst;
    for (std::size_t index = 1u; index < firstPeakCurve.size(); ++index) {
        const auto& sample = firstPeakCurve[index];
        qualifiedFirst = fullCurveDetector.observeAccepted(
            sample.strokeM, sample.forceN, sample.contactSamples);
        if (qualifiedFirst) break;
    }
    require(qualifiedFirst && qualifiedFirst->peak.sampleIndex == 1u &&
                firstPeakCurve[5].forceN > qualifiedFirst->peak.forceN,
            "native streaming detector fixture must distinguish local peak from later global maximum");
    const auto fullCurvePeakIndex = shippingCompressionCurveSampleIndex(
        firstPeakCurve, qualifiedFirst->peak);
    require(fullCurvePeakIndex == std::optional<std::size_t>(2u) &&
                *fullCurvePeakIndex != qualifiedFirst->peak.sampleIndex &&
                firstPeakCurve[*fullCurvePeakIndex].forceN == 100.0,
            "native first-local-peak identity must map to its full curve row including zero origin");
    require(!shippingCompressionCurveSampleIndex(firstPeakCurve,
                ShippingCompressionForceSample{99u, 0.0700, 100.0, 3.0}),
            "native curve mapping must reject a force/contact mismatch");
    const std::array<ShippingCompressionForceSample, 4> nominalCurve{{
        {0u, 0.0, 0.0, 0.0}, {1u, 0.001, 10.0, 1.0},
        {2u, 0.002, 20.0, 1.0}, {3u, 0.003, 10.0, 1.0}}};
    const std::array<ShippingCompressionForceSample, 4> withinGateCurve{{
        {0u, 0.0, 0.0, 0.0}, {1u, 0.001, 10.1, 1.0},
        {2u, 0.002, 20.1, 1.0}, {3u, 0.003, 10.1, 1.0}}};
    const auto withinGate = compareShippingCompressionCurves(
        nominalCurve, withinGateCurve);
    require(withinGate.valid && shippingCompressionSensitivityPassed(withinGate) &&
                withinGate.relativeWorkDifference < 0.02 &&
                withinGate.maximumForceDifferenceNormalizedByNominalPeak < 0.05,
            "rate-sensitivity comparison rejected a force curve inside its declared gates");
    const std::array<ShippingCompressionForceSample, 4> invalidMissingContact{{
        {0u, 0.0, 0.0, 0.0}, {1u, 0.001, 10.0, 0.0},
        {2u, 0.002, 20.0, 1.0}, {3u, 0.003, 10.0, 1.0}}};
    expectInvalid([&] {
        (void)compareShippingCompressionCurves(nominalCurve,
                                                invalidMissingContact);
    }, "rate comparison accepted a later force sample without native platen contact");
    const std::array<ShippingCompressionForceSample, 4> outsideWorkGateCurve{{
        {0u, 0.0, 0.0, 0.0}, {1u, 0.001, 10.6, 1.0},
        {2u, 0.002, 20.6, 1.0}, {3u, 0.003, 10.6, 1.0}}};
    const auto outsideWorkGate = compareShippingCompressionCurves(
        nominalCurve, outsideWorkGateCurve);
    require(outsideWorkGate.valid && outsideWorkGate.relativeWorkDifference > 0.02 &&
                !shippingCompressionSensitivityPassed(outsideWorkGate),
            "rate-sensitivity comparison admitted force-work disagreement above two percent");
    const std::array<ShippingCompressionForceSample, 4> outsideCurveGateCurve{{
        {0u, 0.0, 0.0, 0.0}, {1u, 0.001, 11.2, 1.0},
        {2u, 0.002, 20.0, 1.0}, {3u, 0.003, 8.8, 1.0}}};
    const auto outsideCurveGate = compareShippingCompressionCurves(
        nominalCurve, outsideCurveGateCurve);
    require(outsideCurveGate.valid &&
                outsideCurveGate.relativeWorkDifference < 0.02 &&
                outsideCurveGate.maximumForceDifferenceNormalizedByNominalPeak > 0.05 &&
                !shippingCompressionSensitivityPassed(outsideCurveGate),
            "full-curve gate failed to detect a localized force-shape deviation");
    const auto closure = makeShippingClosurePlan(design);
    require(closure.patches.size() == 9u && !closure.tapeSystemIncluded &&
            !closure.buttFaceAdhesionInvented && !closure.adhesiveLawCalibrated &&
            std::abs(closure.patches.front().overlapAreaM2 - 0.005) < 1.0e-14 &&
            std::abs(closure.patches.front().adhesiveFootprintAreaM2 - 0.002) < 1.0e-14 &&
            std::abs(closure.patches[1].overlapAreaM2 - 0.010) < 1.0e-14 &&
            std::abs(closure.patches[1].adhesiveFootprintAreaM2 - 0.001) < 1.0e-14,
            "all-glued manufacturer-lap/flap overlap plan is incomplete");
    const auto assembly = makeShippingAssemblyProgram(design);
    require(assembly.size() == 15u &&
            std::all_of(assembly.begin(), assembly.end(),
                [](const ShippingAssemblyStage& stage) { return !stage.executed; }) &&
            std::isnan(assembly[assembly.size() - 3u].measuredAdhesiveSetHoldSeconds) &&
            !assembly[assembly.size() - 3u].setHoldProtocolCalibrated &&
            assembly.back().kind == ShippingAssemblyStage::Kind::settleAndCheckpoint,
            "shipping assembly schedule claims execution or omits release/settle");

    ShippingAcceptedHmaActivation activation;
    activation.patchId = "synthetic-accepted-overlap-fixture";
    activation.acceptedStep = 2u;
    activation.sourceStepAccepted = true;
    activation.acceptedContactWitnessSha256 = std::string(64u, 'f');
    activation.immutableSubstrateRestSha256 = std::string(64u, 'b');
    activation.acceptedSnapshotSha256 = std::string(64u, 'c');
    activation.measuredOverlapAreaM2 = 0.001;
    activation.maximumPairGapM = 0.020e-3;
    activation.pairedNodes.push_back({1u, 2u, {0,0,0}, {0,0,0.020e-3}});
    validateShippingHmaActivation(activation);
    auto rejectedActivation = activation;
    rejectedActivation.sourceStepAccepted = false;
    expectInvalid([&] { validateShippingHmaActivation(rejectedActivation); },
                  "HMA activation accepted an unaccepted FEM step");
    rejectedActivation = activation;
    rejectedActivation.maximumPairGapM = 0.05e-3;
    expectInvalid([&] { validateShippingHmaActivation(rejectedActivation); },
                  "HMA activation trusted a caller gap inconsistent with paired coordinates");

    AcceptedCompressionPeak peak;
    peak.completed = true;
    peak.acceptedEvidence = true;
    peak.runId = "synthetic-compression-unit-fixture";
    peak.resultSha256 = std::string(64u, 'd');
    peak.sourceAssemblyCheckpointSha256 = std::string(64u, 'e');
    peak.boardRecipeId = "literature2009";
    peak.acceptedStepCount = 1u;
    peak.peakReactionForceN = 100.0;
    ReleasedAssemblyCheckpoint checkpoint;
    checkpoint.checkpointId = "synthetic-released-assembly-fixture";
    checkpoint.checkpointSha256 = std::string(64u, 'e');
    checkpoint.sourcePhysicsFingerprint = 11u;
    checkpoint.compiledWorldFingerprint = 12u;
    checkpoint.accepted = true;
    checkpoint.completeNativeSnapshot = true;
    checkpoint.temporaryGripsReleased = true;
    checkpoint.toolingParkedOrRemoved = true;
    checkpoint.settled = true;
    checkpoint.manufacturerLapClosedByGeometry = true;
    checkpoint.bottomEndClosedByGeometry = true;
    checkpoint.topEndClosedByGeometry = true;
    checkpoint.allAssemblyActionsAccepted = true;
    checkpoint.maximumNodalSpeedMPerS = 0.0;
    checkpoint.normalizedForceImbalance = 0.0;
    checkpoint.maximumEndCornerGapM = 0.0;
    checkpoint.measuredInside = {};
    checkpoint.boardRecipeId = "literature2009";
    checkpoint.measuredInsideCenterX = 0.150;
    checkpoint.measuredInsideCenterZ = 0.100;
    checkpoint.measuredRoofContactY = 0.20421;
    checkpoint.measuredOuterRoof.bounds = {
        -0.005, 0.305, -0.004, 0.204, 0.20421, 8u, true};
    checkpoint.measuredOuterRoof.measuredFromAcceptedFixtureFreeSettledCheckpoint = true;
    checkpoint.measuredOuterRoof.checkpointSha256 = checkpoint.checkpointSha256;
    checkpoint.measuredOuterRoof.sourcePhysicsFingerprint =
        checkpoint.sourcePhysicsFingerprint;
    checkpoint.measuredOuterRoof.compiledWorldFingerprint =
        checkpoint.compiledWorldFingerprint;
    const auto cases = makeShippingDropCases(peak, checkpoint);
    require(cases.size() == 6u &&
            std::abs(cases[0].massKg - 0.5 * 100.0 / 9.80665) < 1.0e-14 &&
            cases[0].restoreCheckpointSha256 == cases[5].restoreCheckpointSha256 &&
            std::abs(cases[1].impactorCenterZ - 0.152) < 1.0e-14 &&
            std::abs(cases[0].impactorPlanLengthM - std::sqrt(1.2) * 0.310) < 1.0e-14 &&
            std::abs(cases[0].impactorPlanWidthM - std::sqrt(1.2) * 0.208) < 1.0e-14 &&
            cases[0].impactorPlanBasis.find("accepted outer-roof") !=
                std::string::npos &&
            std::abs(cases[1].impactorStartCenterY -
                (checkpoint.measuredOuterRoof.bounds.topY + 0.125)) < 1.0e-14,
            "drop cases lost their common checkpoint, Fpeak mass scaling, or offset");

    auto noRoofCheckpoint = checkpoint;
    noRoofCheckpoint.measuredOuterRoof = {};
    expectInvalid([&] { (void)makeShippingDropCases(peak, noRoofCheckpoint); },
                  "mass case admitted a checkpoint without measured outer-roof geometry");
    auto misboundRoofCheckpoint = checkpoint;
    misboundRoofCheckpoint.measuredOuterRoof.checkpointSha256 =
        std::string(64u, 'f');
    expectInvalid([&] { (void)makeShippingDropCases(peak, misboundRoofCheckpoint); },
                  "mass case admitted roof bounds measured from a different checkpoint");
    auto wrongWorldRoofCheckpoint = checkpoint;
    wrongWorldRoofCheckpoint.measuredOuterRoof.compiledWorldFingerprint += 1u;
    expectInvalid([&] { (void)makeShippingDropCases(peak, wrongWorldRoofCheckpoint); },
                  "mass case admitted roof bounds from a different compiled geometry");

    std::vector<NMFEMNodeStateGPU> roofNodes(4u);
    roofNodes[0].positionAndMass = {0.0f, 1.0f, 0.0f, 1.0f};
    roofNodes[1].positionAndMass = {1.0f, 1.0f, 0.0f, 1.0f};
    roofNodes[2].positionAndMass = {0.0f, 1.0f, 1.0f, 1.0f};
    roofNodes[3].positionAndMass = {0.0f, 0.0f, 0.0f, 1.0f};
    NMTetrahedronGPU roofTet{};
    roofTet.nodes = {0u, 1u, 2u, 3u};
    roofTet.identity = {0u, 0u, 0u, 0u};
    NMFEMSurfaceFaceGPU roofFace{};
    roofFace.adjacency = {0u, NM_INVALID_INDEX, 0u, 0u};
    roofFace.sides = {3u, NM_INVALID_INDEX, 0u, 0u};
    const std::array<NMTetrahedronGPU, 1> roofTets{roofTet};
    const std::array<NMFEMSurfaceFaceGPU, 1> roofFaces{roofFace};
    const auto syntheticRoof = measureShippingOuterRoofBounds(
        roofNodes, 4u, roofTets, roofFaces, 0.005);
    require(syntheticRoof.valid && syntheticRoof.exteriorFaceCount == 1u &&
            std::abs(syntheticRoof.minimumX) < 1.0e-7 &&
            std::abs(syntheticRoof.maximumX - 1.0) < 1.0e-7 &&
            std::abs(syntheticRoof.minimumZ) < 1.0e-7 &&
            std::abs(syntheticRoof.maximumZ - 1.0) < 1.0e-7 &&
            std::abs(syntheticRoof.topY - 1.0) < 1.0e-7,
            "accepted exterior-roof measurement changed the exact face projection");
    auto interiorOnlyFace = roofFace;
    interiorOnlyFace.adjacency.y = 0u;
    expectInvalid([&] {
        (void)measureShippingOuterRoofBounds(
            roofNodes, 4u, roofTets,
            std::span<const NMFEMSurfaceFaceGPU>(&interiorOnlyFace, 1u), 0.005);
    }, "roof measurement admitted an internal face as an exterior impact surface");
    auto verticalFace = roofFace;
    verticalFace.sides.x = 2u;
    expectInvalid([&] {
        (void)measureShippingOuterRoofBounds(
            roofNodes, 4u, roofTets,
            std::span<const NMFEMSurfaceFaceGPU>(&verticalFace, 1u), 0.005);
    }, "roof measurement admitted a vertical wall as the top surface");

    ReleasedAssemblyObservation noEvidence;
    const auto qualification = qualifyReleasedAssembly(design, noEvidence);
    require(!qualification.nativeAssemblyMechanicsQualified &&
            !qualification.simulationAssemblyComplete &&
            !qualification.materialAndInterfaceCalibrationGate &&
            !qualification.physicalBoxTestGate,
            "unmeasured shipping assembly was qualified");

    // A complete accepted simulation may use explicitly declared analogue
    // laws. Its completion verdict is separate from source-grade calibration
    // and a physical box test.
    ReleasedAssemblyObservation analogueSimulation;
    analogueSimulation.nativeAssemblyTrialCompleted = true;
    analogueSimulation.acceptedCheckpoint = true;
    analogueSimulation.allAssemblyActionsAccepted = true;
    analogueSimulation.allTemporaryGripsReleased = true;
    analogueSimulation.allToolsParkedOrRemoved = true;
    analogueSimulation.manufacturerLapActivated = true;
    analogueSimulation.manufacturerLapActivated = true;
    analogueSimulation.manufacturerLapClosedByGeometry = true;
    analogueSimulation.bottomEndClosedByGeometry = true;
    analogueSimulation.topEndClosedByGeometry = true;
    analogueSimulation.flapOverlapPatchesActivated = 8u;
    analogueSimulation.maximumEndCornerGapM = 0.5e-3;
    analogueSimulation.endShapeResidualM = 0.5e-3;
    analogueSimulation.settled = true;
    analogueSimulation.maximumNodalSpeedMPerS = 0.0;
    analogueSimulation.normalizedForceImbalance = 0.0;
    analogueSimulation.measuredInside = {0.3029, 0.1981, 0.2019};
    const auto analogueQualification = qualifyReleasedAssembly(
        design, analogueSimulation);
    require(analogueQualification.simulationAssemblyComplete &&
            !analogueQualification.materialAndInterfaceCalibrationGate &&
            !analogueQualification.nativeAssemblyMechanicsQualified &&
            !analogueQualification.physicalShippingBoxQualified,
            "simulation completion was incorrectly coupled to source-grade or physical evidence");
    analogueSimulation.measuredInside.lengthX += 1.0e-3;
    const auto beyondLengthGate = qualifyReleasedAssembly(
        design, analogueSimulation);
    require(!beyondLengthGate.clearDimensionGate &&
            !beyondLengthGate.simulationAssemblyComplete,
            "1% internal-dimension gate admitted a length error above 3 mm");
    auto unsafeCheckpoint = checkpoint;
    unsafeCheckpoint.settled = false;
    expectInvalid([&] { (void)makeShippingDropCases(peak, unsafeCheckpoint); },
                  "drop case accepted an unsettled assembly checkpoint");

    // Build the flat blank only in this CPU fixture to verify clipped section
    // closure and SI volume. No folded coordinates or persistent mesh artifact
    // are created by the test.
    const ShippingBoxGeometry geometry = buildShippingBoxBlankGeometry(design);
    require(geometry.topologyNodeCount == geometry.blank.points.size() &&
            geometry.topologyTetrahedronCount == geometry.blank.tetrahedra.size() &&
            geometry.topologyNodeCount > 0u && geometry.topologyTetrahedronCount > 0u &&
            geometry.blank.panels.size() == 4u && geometry.blank.scoreLines.size() == 12u &&
            geometry.blank.slots.size() == 6u && !geometry.foldedCoordinatesAuthored &&
            !geometry.physicalAssemblyQualified,
            "shipping blank topology or qualification state is invalid");
    std::array<double, 3> positiveVolumes{};
    for (const auto& tet : geometry.blank.tetrahedra) {
        const auto& n = tet.nodes;
        const double volume = sixVolume(geometry.blank.points[n[0]],
            geometry.blank.points[n[1]], geometry.blank.points[n[2]],
            geometry.blank.points[n[3]]) / 6.0;
        require(std::isfinite(volume) && volume > 0.0,
                "shipping blank contains nonpositive or nonfinite tetrahedron");
        positiveVolumes[static_cast<std::size_t>(tet.material)] += volume;
    }
    for (std::size_t material = 0; material < positiveVolumes.size(); ++material) {
        require(positiveVolumes[material] > 0.0 &&
                std::abs(positiveVolumes[material] -
                    geometry.blank.volumeByMaterial[material]) <=
                    1.0e-8 * positiveVolumes[material],
                "shipping tetrahedral volume does not close by material");
    }
    require(std::abs(geometry.blank.perimeterSpan - 1.025) < 1.0e-14 &&
            std::abs(geometry.blank.blankHeight - 0.400) < 1.0e-14 &&
            geometry.crossSection.glueFootprints.size() == 260u,
            "shipping geometry endpoint or finite glue patch count is wrong");
    double sectionArea = 0.0;
    for (const double area : geometry.crossSection.crossSectionAreaByMaterial) {
        require(std::isfinite(area) && area > 0.0,
                "clipped section has an empty or nonfinite material region");
        sectionArea += area;
    }
    require(sectionArea > 0.0 &&
            std::abs(geometry.crossSection.glueFootprints.front().xLeft) < 1.0e-14 &&
            std::abs(geometry.crossSection.glueFootprints.front().actualHorizontalWidth -
                0.5 * geometry.crossSection.glueFootprints.front().requestedHorizontalWidth) <
                1.0e-11,
            "terminal glue footprint was not clipped at the board perimeter");
    for (const auto& triangle : geometry.crossSection.triangles) {
        const auto& a = geometry.crossSection.points[triangle.points[0]];
        const auto& b = geometry.crossSection.points[triangle.points[1]];
        const auto& c = geometry.crossSection.points[triangle.points[2]];
        const double signedArea2 = (b.x - a.x) * (c.z - a.z) -
            (b.z - a.z) * (c.x - a.x);
        require(a.x >= -1.0e-12 && a.x <= sectionConfig.length + 1.0e-12 &&
                b.x >= -1.0e-12 && b.x <= sectionConfig.length + 1.0e-12 &&
                c.x >= -1.0e-12 && c.x <= sectionConfig.length + 1.0e-12 &&
                std::isfinite(signedArea2) && signedArea2 > 1.0e-22,
                "partial-period section contains an out-of-domain point");
    }

    std::cout << "shipping_box_cpu_protocol_and_geometry_passed"
              << " coupon_fixture=synthetic_zero_offset_not_physical"
              << " period_count=" << std::setprecision(12)
              << capacity.requestedPitchCount
              << " clipped_period_intervals=" << capacity.clippedPitchIntervals
              << " nodes=" << geometry.topologyNodeCount
              << " tetrahedra=" << geometry.topologyTetrahedronCount
              << " fgmres_basis_restart16_lower_bound_bytes="
              << geometry.femOnlyFgmresBasisBytesAtRestart16
              << " HMA_patches=" << closure.patches.size()
              << " dropped_cases=synthetic_protocol_only"
              << " native_physics_qualified=false\n";
}
} // namespace

int main() {
    try {
        run();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
