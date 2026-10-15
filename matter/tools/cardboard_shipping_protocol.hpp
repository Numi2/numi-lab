#pragma once

// Shipping-size FEFCO 0201 design inputs and checkpoint-linked test protocol.
// This layer derives score-to-score geometry from accepted allowance-coupon
// evidence or an explicitly labeled engineering assumption. Zero offsets are
// restricted to preflight fixtures. It never authors a folded mesh or implies
// native/physical qualification.

#include "cardboard_box_blank.hpp"
#include "cardboard_shipping_roof.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace numi::cardboard {

struct ShippingInsideDimensions {
    double lengthX = 0.300; // internal clear length, metres
    double heightY = 0.200; // internal clear height, metres
    double widthZ = 0.200;  // internal clear width, metres
};

enum class CouponEvidenceClass : std::uint32_t {
    unspecified = 0u,
    syntheticUnitFixture = 1u,
    physicalAcceptedCoupon = 2u,
    // Accepted native score/fold/release output. This can derive a native
    // design, but it is not physical stock validation.
    nativeAcceptedCoupon = 3u,
};

struct AcceptedScoreAllowanceCoupon {
    bool accepted = false;
    CouponEvidenceClass evidenceClass = CouponEvidenceClass::unspecified;
    std::string couponId;
    std::string boardRecipeId;
    std::string sourceRunId;
    std::string measurementSha256;
    // score-center span minus the measured inside-clear dimension, in X/Y/Z
    // order. The sign is retained; the caller may not substitute a guessed
    // positive "score allowance" convention.
    std::array<double, 3> scoreCenterSpanMinusInsideM{};
};

struct ShippingBoxDesignDefaults {
    ShippingInsideDimensions inside{};
    double manufacturerJointWidthM = 0.025;
    double slotKerfM = 0.001;
    std::uint32_t subdivisionsPerBand = 16u;
    std::string boardRecipeId = "literature2009";
};

struct ShippingBoxDesign {
    ShippingBoxDesignDefaults defaults{};
    AcceptedScoreAllowanceCoupon allowanceCoupon{};
    FEFCO0201BlankConfig blankConfig{};
    double perimeterSpanM = 0.0;
    double blankHeightM = 0.0;
    double flapDepthM = 0.0;
    bool allowanceFromPhysicalCoupon = false;
    bool allowanceFromNativeCoupon = false;
    std::string geometryState = "score-allowance-derived-dieline-only";
};

struct ShippingScoreLegSchedule {
    std::uint32_t stepCount = 0u;
    double durationSeconds = 0.0;
    double effectiveSpeedMPerS = 0.0;
};

struct ShippingConstantSpeedSubstep {
    double timestepSeconds = 0.0;
    double distanceM = 0.0;
    double completedFraction = 0.0;
};

struct ShippingConstantSpeedMoveSchedule {
    std::vector<ShippingConstantSpeedSubstep> substeps;
    double requestedDistanceM = 0.0;
    double requestedSpeedMPerS = 0.0;
    double totalDurationSeconds = 0.0;
    double maximumTimestepSeconds = 0.0;
    double minimumUsedTimestepSeconds = 0.0;
    double maximumUsedTimestepSeconds = 0.0;
};

struct ShippingConstantSpeedMoveSummary {
    std::uint32_t acceptedStepCount = 0u;
    double observationDurationSeconds = 0.0;
    double minimumControlTimestepSeconds = 0.0;
    double maximumControlTimestepSeconds = 0.0;
};

// Build a constant-rate path using maximumTimestepSeconds as an upper bound,
// not as the duration of the whole move. The final substep is shortened when
// necessary so every refinement reaches the same endpoint at the same time.
inline ShippingConstantSpeedMoveSchedule makeShippingConstantSpeedMoveSchedule(
    const double distanceM,
    const double speedMPerS,
    const double maximumTimestepSeconds,
    const std::uint32_t maximumSteps) {
    if (!(std::isfinite(distanceM) && distanceM > 0.0) ||
        !(std::isfinite(speedMPerS) && speedMPerS > 0.0) ||
        !(std::isfinite(maximumTimestepSeconds) &&
          maximumTimestepSeconds > 0.0) || maximumSteps == 0u)
        throw std::invalid_argument(
            "shipping constant-speed move needs positive finite distance, speed, timestep, and step limit");

    const double duration = distanceM / speedMPerS;
    const double ratio = duration / maximumTimestepSeconds;
    if (!(std::isfinite(duration) && duration > 0.0 &&
          std::isfinite(ratio) && ratio > 0.0))
        throw std::overflow_error(
            "shipping constant-speed move duration is not representable");

    // Avoid a spurious extra, near-zero substep when the ratio is an integer
    // up to ordinary floating-point roundoff.
    const double nearestInteger = std::round(ratio);
    const double ratioTolerance = 16.0 *
        std::numeric_limits<double>::epsilon() * std::max(1.0, ratio);
    const double stepCountValue =
        std::abs(ratio - nearestInteger) <= ratioTolerance
        ? nearestInteger : std::ceil(ratio);
    if (!(std::isfinite(stepCountValue) && stepCountValue >= 1.0 &&
          stepCountValue <= static_cast<double>(maximumSteps)))
        throw std::length_error(
            "shipping constant-speed move exceeds the configured accepted-step limit");

    ShippingConstantSpeedMoveSchedule result;
    result.requestedDistanceM = distanceM;
    result.requestedSpeedMPerS = speedMPerS;
    result.totalDurationSeconds = duration;
    result.maximumTimestepSeconds = maximumTimestepSeconds;
    const auto stepCount = static_cast<std::uint32_t>(stepCountValue);
    result.substeps.reserve(stepCount);
    double elapsed = 0.0;
    double travelled = 0.0;
    result.minimumUsedTimestepSeconds =
        std::numeric_limits<double>::infinity();
    for (std::uint32_t index = 0u; index < stepCount; ++index) {
        const bool final = index + 1u == stepCount;
        const double remainingDuration = duration - elapsed;
        const double dt = final ? remainingDuration
                                : maximumTimestepSeconds;
        const double stepDistance = final ? distanceM - travelled
                                          : speedMPerS * dt;
        if (!(std::isfinite(dt) && dt > 0.0 &&
              std::isfinite(stepDistance) && stepDistance > 0.0) ||
            dt > maximumTimestepSeconds + ratioTolerance *
                maximumTimestepSeconds)
            throw std::logic_error(
                "shipping constant-speed move generated an invalid or oversized substep");
        elapsed += dt;
        travelled += stepDistance;
        result.substeps.push_back({dt, stepDistance,
            final ? 1.0 : std::min(1.0, travelled / distanceM)});
        result.minimumUsedTimestepSeconds = std::min(
            result.minimumUsedTimestepSeconds, dt);
        result.maximumUsedTimestepSeconds = std::max(
            result.maximumUsedTimestepSeconds, dt);
    }
    const double durationTolerance = 32.0 *
        std::numeric_limits<double>::epsilon() * std::max(1.0, duration);
    const double distanceTolerance = 32.0 *
        std::numeric_limits<double>::epsilon() * std::max(1.0, distanceM);
    if (std::abs(elapsed - duration) > durationTolerance ||
        std::abs(travelled - distanceM) > distanceTolerance ||
        result.substeps.empty() ||
        result.substeps.back().completedFraction != 1.0)
        throw std::logic_error(
            "shipping constant-speed move does not preserve its endpoint and duration");
    return result;
}

inline ShippingScoreLegSchedule makeShippingScoreLegSchedule(
    const double travelM,
    const double requestedSpeedMPerS,
    const double timestepSeconds,
    const std::uint32_t maximumSteps) {
    if (!(std::isfinite(travelM) && travelM > 0.0) ||
        !(std::isfinite(requestedSpeedMPerS) && requestedSpeedMPerS > 0.0) ||
        !(std::isfinite(timestepSeconds) && timestepSeconds > 0.0) ||
        maximumSteps == 0u)
        throw std::invalid_argument("shipping score leg schedule inputs must be finite and positive");
    const double rawSteps = std::ceil(
        travelM / (requestedSpeedMPerS * timestepSeconds));
    if (!(std::isfinite(rawSteps) && rawSteps >= 1.0) ||
        rawSteps > static_cast<double>(maximumSteps))
        throw std::invalid_argument(
            "shipping score leg exceeds the configured accepted-step limit at this speed and timestep");
    ShippingScoreLegSchedule result;
    result.stepCount = static_cast<std::uint32_t>(rawSteps);
    result.durationSeconds = result.stepCount * timestepSeconds;
    result.effectiveSpeedMPerS = travelM / result.durationSeconds;
    if (!(std::isfinite(result.durationSeconds) && result.durationSeconds > 0.0 &&
          std::isfinite(result.effectiveSpeedMPerS) &&
          result.effectiveSpeedMPerS > 0.0 &&
          result.effectiveSpeedMPerS <= requestedSpeedMPerS *
              (1.0 + 8.0 * std::numeric_limits<double>::epsilon())))
        throw std::invalid_argument("shipping score leg schedule is numerically invalid");
    return result;
}

inline bool isLowerHexDigest(const std::string& digest) {
    if (digest.size() != 64u) return false;
    return std::all_of(digest.begin(), digest.end(), [](const char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

inline ShippingBoxDesign deriveShippingBoxDesign(
    const ShippingBoxDesignDefaults& defaults,
    const AcceptedScoreAllowanceCoupon& coupon) {
    const auto positiveFinite = [](const double value) {
        return std::isfinite(value) && value > 0.0;
    };
    if (!coupon.accepted)
        throw std::invalid_argument(
            "shipping-box dieline requires an accepted score-allowance coupon");
    if (coupon.evidenceClass == CouponEvidenceClass::unspecified ||
        coupon.couponId.empty() || coupon.sourceRunId.empty() ||
        !isLowerHexDigest(coupon.measurementSha256))
        throw std::invalid_argument(
            "accepted score-allowance coupon provenance is incomplete");
    if (coupon.boardRecipeId.empty() ||
        coupon.boardRecipeId != defaults.boardRecipeId)
        throw std::invalid_argument(
            "score-allowance coupon board recipe does not match shipping design");
    if (!positiveFinite(defaults.inside.lengthX) ||
        !positiveFinite(defaults.inside.heightY) ||
        !positiveFinite(defaults.inside.widthZ) ||
        !positiveFinite(defaults.manufacturerJointWidthM) ||
        !positiveFinite(defaults.slotKerfM) ||
        defaults.subdivisionsPerBand == 0u ||
        defaults.subdivisionsPerBand > 64u)
        throw std::invalid_argument("shipping-box design defaults are invalid");
    for (const double allowance : coupon.scoreCenterSpanMinusInsideM)
        if (!std::isfinite(allowance))
            throw std::invalid_argument(
                "score-allowance coupon contains a nonfinite measurement");

    ShippingBoxDesign design;
    design.defaults = defaults;
    design.allowanceCoupon = coupon;
    design.allowanceFromPhysicalCoupon =
        coupon.evidenceClass == CouponEvidenceClass::physicalAcceptedCoupon;
    design.allowanceFromNativeCoupon =
        coupon.evidenceClass == CouponEvidenceClass::nativeAcceptedCoupon;
    design.blankConfig.panelLength = defaults.inside.lengthX +
        coupon.scoreCenterSpanMinusInsideM[0];
    design.blankConfig.wallHeight = defaults.inside.heightY +
        coupon.scoreCenterSpanMinusInsideM[1];
    design.blankConfig.panelWidth = defaults.inside.widthZ +
        coupon.scoreCenterSpanMinusInsideM[2];
    design.blankConfig.manufacturerJointWidth = defaults.manufacturerJointWidthM;
    design.blankConfig.slotKerf = defaults.slotKerfM;
    design.blankConfig.subdivisionsPerBand = defaults.subdivisionsPerBand;
    if (!positiveFinite(design.blankConfig.panelLength) ||
        !positiveFinite(design.blankConfig.panelWidth) ||
        !positiveFinite(design.blankConfig.wallHeight))
        throw std::invalid_argument(
            "coupon-derived score dimensions are nonpositive");
    design.perimeterSpanM = defaults.manufacturerJointWidthM +
        2.0 * (design.blankConfig.panelLength + design.blankConfig.panelWidth);
    design.flapDepthM = 0.5 * design.blankConfig.panelWidth;
    design.blankHeightM = design.blankConfig.wallHeight +
        design.blankConfig.panelWidth;
    return design;
}

inline void validateShippingBoxDesign(const ShippingBoxDesign& design) {
    const ShippingBoxDesign expected = deriveShippingBoxDesign(
        design.defaults, design.allowanceCoupon);
    const auto same = [](const double a, const double b) {
        return std::isfinite(a) && std::isfinite(b) &&
            std::abs(a - b) <= 1.0e-12 * std::max({1.0, std::abs(a), std::abs(b)});
    };
    if (!same(design.blankConfig.panelLength, expected.blankConfig.panelLength) ||
        !same(design.blankConfig.panelWidth, expected.blankConfig.panelWidth) ||
        !same(design.blankConfig.wallHeight, expected.blankConfig.wallHeight) ||
        !same(design.blankConfig.manufacturerJointWidth,
              expected.blankConfig.manufacturerJointWidth) ||
        !same(design.blankConfig.slotKerf, expected.blankConfig.slotKerf) ||
        design.blankConfig.subdivisionsPerBand !=
            expected.blankConfig.subdivisionsPerBand ||
        !same(design.perimeterSpanM, expected.perimeterSpanM) ||
        !same(design.blankHeightM, expected.blankHeightM) ||
        !same(design.flapDepthM, expected.flapDepthM) ||
        design.allowanceFromPhysicalCoupon != expected.allowanceFromPhysicalCoupon ||
        design.allowanceFromNativeCoupon != expected.allowanceFromNativeCoupon)
        throw std::invalid_argument(
            "shipping-box design fields do not match its accepted coupon and target");
}

struct ShippingCrossSectionCapacity {
    bool compatible = false;
    double requestedPitchCount = 0.0;
    std::uint32_t clippedPitchIntervals = 0u;
    bool hasFractionalTerminalPitch = false;
    std::string reason;
};

inline ShippingCrossSectionCapacity auditShippingCrossSectionCapacity(
    const ShippingBoxDesign& design,
    const GlueMeshConfig& crossSection) {
    ShippingCrossSectionCapacity result;
    try { validateShippingBoxDesign(design); }
    catch (const std::invalid_argument& error) {
        result.reason = error.what();
        return result;
    }
    if (!(std::isfinite(crossSection.pitch) && crossSection.pitch > 0.0) ||
        !(std::isfinite(crossSection.length) && crossSection.length > 0.0)) {
        result.reason = "cross-section length and pitch must be finite and positive";
        return result;
    }
    const double lengthError = std::abs(crossSection.length - design.perimeterSpanM);
    if (lengthError > 1.0e-10 * design.perimeterSpanM) {
        result.reason = "cross-section length does not match coupon-derived blank perimeter";
        return result;
    }
    result.requestedPitchCount = crossSection.length / crossSection.pitch;
    const double rounded = std::round(result.requestedPitchCount);
    result.clippedPitchIntervals = static_cast<std::uint32_t>(
        std::ceil(result.requestedPitchCount));
    result.hasFractionalTerminalPitch =
        std::abs(result.requestedPitchCount - rounded) > 1.0e-10;
    if (result.clippedPitchIntervals > 256u) {
        result.reason = "shipping perimeter exceeds the clipped-profile 256-pitch bound";
        return result;
    }
    if (result.hasFractionalTerminalPitch &&
        !crossSection.allowPartialTerminalPitch) {
        result.reason = "fractional terminal flute requires explicit clipped-profile mode";
        return result;
    }
    if (crossSection.allowPartialTerminalPitch &&
        result.clippedPitchIntervals > 256u) {
        result.reason = "partial-pitch builder capacity exceeded";
        return result;
    }
    result.compatible = true;
    result.reason = result.hasFractionalTerminalPitch
        ? "periodic profile and bond footprints clip at the authored perimeter"
        : "whole-pitch perimeter matches the coupon-derived blank";
    return result;
}

struct ShippingCorrugatedRecipe {
    // The selected cross-source literature2009 recipe. The geometry is an
    // explicit sinusoidal fallback; the paper's arc profile is not represented.
    double pitchM = 7.9e-3;
    double caliperM = 4.210e-3;
    double linerThicknessM = 0.277e-3;
    double mediumThicknessM = 0.191e-3;
    double lowerGlueGapM = 0.100e-3;
    double lowerBondWidthM = 0.800e-3;
    double upperGlueGapM = 0.030e-3;
    double upperBondWidthM = 0.600e-3;
    std::uint32_t nxPerPitch = 4u;
    std::uint32_t thicknessSlices = 1u;
};

inline GlueMeshConfig makeShippingCrossSectionConfig(
    const ShippingBoxDesign& design,
    const ShippingCorrugatedRecipe& recipe = {}) {
    validateShippingBoxDesign(design);
    GlueMeshConfig config;
    config.length = design.perimeterSpanM;
    config.pitch = recipe.pitchM;
    config.caliper = recipe.caliperM;
    config.linerThickness = recipe.linerThicknessM;
    config.mediumThickness = recipe.mediumThicknessM;
    config.glueMinimumGap = recipe.lowerGlueGapM;
    config.bondWidth = recipe.lowerBondWidthM;
    config.upperGlueMinimumGap = recipe.upperGlueGapM;
    config.upperBondWidth = recipe.upperBondWidthM;
    config.nxPerPitch = recipe.nxPerPitch;
    config.thicknessSlices = recipe.thicknessSlices;
    config.allowPartialTerminalPitch = true;
    const auto capacity = auditShippingCrossSectionCapacity(design, config);
    if (!capacity.compatible)
        throw std::invalid_argument(capacity.reason);
    return config;
}

struct ShippingBoxGeometry {
    ShippingBoxDesign design{};
    ConformingGlueMesh crossSection{};
    FEFCO0201Blank blank{};
    std::uint64_t topologyNodeCount = 0u;
    std::uint64_t topologyTetrahedronCount = 0u;
    std::array<double, 3> tetrahedralVolumeByMaterialM3{};
    std::uint64_t nodeStateBytesAt80BytesPerNode = 0u;
    std::uint64_t femOnlyFgmresBasisBytesAtRestart16 = 0u;
    bool foldedCoordinatesAuthored = false;
    bool physicalAssemblyQualified = false;
};

inline ShippingBoxGeometry buildShippingBoxBlankGeometry(
    const ShippingBoxDesign& design,
    const ShippingCorrugatedRecipe& recipe = {}) {
    validateShippingBoxDesign(design);
    ShippingBoxGeometry result;
    result.design = design;
    const GlueMeshConfig config = makeShippingCrossSectionConfig(design, recipe);
    result.crossSection = buildConformingGlueCrossSection(config);
    result.blank = buildFEFCO0201Blank(design.blankConfig, result.crossSection);
    result.topologyNodeCount = result.blank.points.size();
    result.topologyTetrahedronCount = result.blank.tetrahedra.size();
    result.tetrahedralVolumeByMaterialM3 = result.blank.volumeByMaterial;
    // shared.h fixes NMFEMNodeStateGPU at 80 bytes. Runtime mixed FGMRES
    // reserves (restart+1)+restart nm_float4 entries for each mixed unknown;
    // a FEM-only lower bound has 2 unknowns per node and restart<=16. This is
    // not a complete resident-memory estimate (operator/contact/other fields
    // and multiple transaction snapshots are excluded).
    result.nodeStateBytesAt80BytesPerNode =
        result.topologyNodeCount * 80u;
    result.femOnlyFgmresBasisBytesAtRestart16 =
        result.topologyNodeCount * 2u * (2u * 16u + 1u) * 16u;
    return result;
}

struct ShippingScoreToolDefaults {
    double punchRadiusM = 1.0e-3;
    double channelWidthM = 6.0e-3;
    // Explicit engineering geometry default for each support rail; not a
    // source-calibrated forming-die dimension.
    double supportRailWidthM = 10.0e-3;
    double dieToChannelClearanceCalipers = 0.25;
    std::array<double, 4> strokeDepthCaliperFractions{0.1, 0.2, 0.3, 0.4};
    bool channelGeometryImplementedByCurrentDieHelper = true;
    bool creaseDamageLawQualified = false;
};

inline std::array<double, 4> shippingScoreDepthsM(
    const ShippingScoreToolDefaults& tools,
    const double boardCaliperM) {
    if (!(std::isfinite(boardCaliperM) && boardCaliperM > 0.0) ||
        !(std::isfinite(tools.punchRadiusM) && tools.punchRadiusM > 0.0) ||
        !(std::isfinite(tools.channelWidthM) && tools.channelWidthM > 0.0) ||
        !(std::isfinite(tools.dieToChannelClearanceCalipers) &&
          tools.dieToChannelClearanceCalipers > 0.0))
        throw std::invalid_argument("shipping score tool dimensions are invalid");
    std::array<double, 4> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        const double fraction = tools.strokeDepthCaliperFractions[index];
        if (!(std::isfinite(fraction) && fraction >= 0.1 && fraction <= 0.4))
            throw std::invalid_argument("score stroke fractions must be in [0.1, 0.4]");
        result[index] = fraction * boardCaliperM;
    }
    return result;
}

struct ShippingRect {
    double u0M = 0.0;
    double v0M = 0.0;
    double u1M = 0.0;
    double v1M = 0.0;
    [[nodiscard]] double areaM2() const {
        return std::max(0.0, u1M - u0M) * std::max(0.0, v1M - v0M);
    }
};

enum class ShippingBondPatchKind : std::uint32_t {
    manufacturerLap = 0u,
    flapOverlap = 1u,
};

enum class ShippingEnd : std::uint32_t { bottom = 0u, top = 1u };

struct ShippingBondPatch {
    std::string patchId;
    ShippingBondPatchKind kind = ShippingBondPatchKind::manufacturerLap;
    ShippingEnd end = ShippingEnd::bottom;
    std::uint32_t majorPanel = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t minorPanel = std::numeric_limits<std::uint32_t>::max();
    ShippingRect overlapOnFaceA{};
    ShippingRect overlapOnFaceB{};
    ShippingRect adhesiveStripOnFaceA{};
    double overlapAreaM2 = 0.0;
    double adhesiveFootprintAreaM2 = 0.0;
    std::string adhesive = "hot-melt; grade and constitutive law uncalibrated";
};

struct ShippingClosurePlan {
    std::vector<ShippingBondPatch> patches;
    double overlapAreaM2 = 0.0;
    double adhesiveFootprintAreaM2 = 0.0;
    bool tapeSystemIncluded = false;
    bool buttFaceAdhesionInvented = false;
    bool adhesiveLawCalibrated = false;
};

inline ShippingClosurePlan makeShippingClosurePlan(
    const ShippingBoxDesign& design,
    const double hmaStripWidthM = 10.0e-3) {
    validateShippingBoxDesign(design);
    if (!(std::isfinite(hmaStripWidthM) && hmaStripWidthM > 0.0 &&
          hmaStripWidthM < design.defaults.manufacturerJointWidthM))
        throw std::invalid_argument(
            "HMA strip width must be positive and narrower than the manufacturer lap");
    ShippingClosurePlan plan;
    const double height = design.blankConfig.wallHeight;
    const double jointWidth = design.defaults.manufacturerJointWidthM;
    ShippingBondPatch lap;
    lap.patchId = "manufacturer-hma-lap";
    lap.kind = ShippingBondPatchKind::manufacturerLap;
    lap.overlapOnFaceA = {0.0, 0.0, jointWidth, height};
    lap.overlapOnFaceB = {0.0, 0.0, jointWidth, height};
    lap.adhesiveStripOnFaceA = {
        0.5 * (jointWidth - hmaStripWidthM), 0.0,
        0.5 * (jointWidth + hmaStripWidthM), height};
    lap.overlapAreaM2 = lap.overlapOnFaceA.areaM2();
    lap.adhesiveFootprintAreaM2 = lap.adhesiveStripOnFaceA.areaM2();
    plan.patches.push_back(lap);

    const double flap = design.flapDepthM;
    if (hmaStripWidthM >= flap)
        throw std::invalid_argument("HMA strip width must be narrower than flap overlap");
    // Two major walls (0 and 2) meet the two minor end flaps (1 and 3) at
    // four actual 100 x 100 mm overlap patches per end. The major flaps meet
    // each other at a butt line; no adhesive is assigned to that line.
    for (const ShippingEnd end : {ShippingEnd::bottom, ShippingEnd::top}) {
        for (const std::uint32_t major : {0u, 2u}) {
            for (const std::uint32_t minor : {1u, 3u}) {
                ShippingBondPatch patch;
                patch.patchId = std::string(end == ShippingEnd::bottom
                    ? "bottom" : "top") + "-major-" +
                    std::to_string(major) + "-minor-" +
                    std::to_string(minor);
                patch.kind = ShippingBondPatchKind::flapOverlap;
                patch.end = end;
                patch.majorPanel = major;
                patch.minorPanel = minor;
                patch.overlapOnFaceA = {0.0, 0.0, flap, flap};
                patch.overlapOnFaceB = {0.0, 0.0, flap, flap};
                patch.adhesiveStripOnFaceA = {
                    0.5 * (flap - hmaStripWidthM), 0.0,
                    0.5 * (flap + hmaStripWidthM), flap};
                patch.overlapAreaM2 = patch.overlapOnFaceA.areaM2();
                patch.adhesiveFootprintAreaM2 =
                    patch.adhesiveStripOnFaceA.areaM2();
                plan.patches.push_back(std::move(patch));
            }
        }
    }
    for (const auto& patch : plan.patches) {
        plan.overlapAreaM2 += patch.overlapAreaM2;
        plan.adhesiveFootprintAreaM2 += patch.adhesiveFootprintAreaM2;
    }
    return plan;
}

struct ShippingPoint3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct ShippingPairedNodeAtActivation {
    std::uint32_t substrateNode = 0u;
    std::uint32_t closureFaceNode = 0u;
    ShippingPoint3 substrateAcceptedPosition{};
    ShippingPoint3 closureAcceptedPosition{};
};

struct ShippingAcceptedHmaActivation {
    std::string patchId;
    std::uint64_t acceptedStep = 0u;
    bool sourceStepAccepted = false;
    std::string acceptedContactWitnessSha256;
    std::string immutableSubstrateRestSha256;
    std::string acceptedSnapshotSha256;
    double measuredOverlapAreaM2 = 0.0;
    double maximumPairGapM = 0.0;
    std::vector<ShippingPairedNodeAtActivation> pairedNodes;
};

inline double shippingActivationPairGapM(
    const ShippingAcceptedHmaActivation& activation) {
    double maximumGap = 0.0;
    for (const auto& pair : activation.pairedNodes) {
        const double dx = pair.closureAcceptedPosition.x -
            pair.substrateAcceptedPosition.x;
        const double dy = pair.closureAcceptedPosition.y -
            pair.substrateAcceptedPosition.y;
        const double dz = pair.closureAcceptedPosition.z -
            pair.substrateAcceptedPosition.z;
        maximumGap = std::max(maximumGap, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    return maximumGap;
}

inline void validateShippingHmaActivation(
    const ShippingAcceptedHmaActivation& activation,
    const double maximumAllowedGapM = 0.1e-3) {
    if (activation.patchId.empty() || activation.acceptedStep == 0u ||
        !activation.sourceStepAccepted ||
        !isLowerHexDigest(activation.acceptedContactWitnessSha256) ||
        !isLowerHexDigest(activation.immutableSubstrateRestSha256) ||
        !isLowerHexDigest(activation.acceptedSnapshotSha256) ||
        !std::isfinite(activation.measuredOverlapAreaM2) ||
        !(activation.measuredOverlapAreaM2 > 0.0) ||
        !std::isfinite(activation.maximumPairGapM) ||
        activation.maximumPairGapM < 0.0 ||
        activation.maximumPairGapM > maximumAllowedGapM ||
        activation.pairedNodes.empty())
        throw std::invalid_argument(
            "HMA activation requires accepted overlapping faces and immutable reference identities");
    std::vector<std::uint32_t> substrateIds;
    std::vector<std::uint32_t> closureIds;
    substrateIds.reserve(activation.pairedNodes.size());
    closureIds.reserve(activation.pairedNodes.size());
    for (const auto& pair : activation.pairedNodes) {
        const auto finite = [](const ShippingPoint3& p) {
            return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
        };
        if (!finite(pair.substrateAcceptedPosition) ||
            !finite(pair.closureAcceptedPosition))
            throw std::invalid_argument("HMA activation has nonfinite accepted coordinates");
        substrateIds.push_back(pair.substrateNode);
        closureIds.push_back(pair.closureFaceNode);
    }
    const double measuredPairGap = shippingActivationPairGapM(activation);
    if (!std::isfinite(measuredPairGap) || measuredPairGap > maximumAllowedGapM ||
        std::abs(measuredPairGap - activation.maximumPairGapM) > 1.0e-9)
        throw std::invalid_argument(
            "HMA activation reported gap differs from accepted paired-node geometry");
    std::sort(substrateIds.begin(), substrateIds.end());
    std::sort(closureIds.begin(), closureIds.end());
    if (std::adjacent_find(substrateIds.begin(), substrateIds.end()) != substrateIds.end() ||
        std::adjacent_find(closureIds.begin(), closureIds.end()) != closureIds.end())
        throw std::invalid_argument("HMA activation node pairing is not one-to-one");
    std::vector<std::uint32_t> commonIds;
    std::set_intersection(substrateIds.begin(), substrateIds.end(),
        closureIds.begin(), closureIds.end(), std::back_inserter(commonIds));
    if (!commonIds.empty())
        throw std::invalid_argument(
            "HMA activation faces must use duplicated, disjoint node identities");
    // This record seeds only a new adhesive/interface-local reference. It
    // carries the substrate rest hash as an immutable identity and has no
    // operation capable of rewriting or resetting the bulk FEM reference.
}

struct ShippingCompressionProtocol {
    // User-selected slow protocol: 12.5 mm/min, bounded by 70% of the
    // measured inside height. These are protocol settings, not source-derived
    // material calibration or a physical qualification threshold.
    double platenSpeedMPerS = 12.5e-3 / 60.0;
    double targetStrokeFractionOfInsideHeight = 0.70;
    double incrementM = 0.25e-3;
    double gravityMPerS2 = 9.80665;
    double requiredPeakDropFraction = 0.01;
    // Separate speed-sensitivity acceptance gates after both first peaks are
    // measured from the same accepted support-ready checkpoint.
    double peakForceEquivalenceToleranceFraction = 0.02;
    double workEquivalenceToleranceFraction = 0.02;
    double normalizedFullCurveToleranceFraction = 0.05;
    std::uint32_t consecutiveDeclinesForPeak = 2u;
    bool stopAtFirstAcceptedPeakOrInstability = true;
    std::string branchPurpose =
        "nominal and half-speed branches start from the same accepted support-ready checkpoint; each first local reaction peak uses a separate one-percent decline detector, then the branches are compared by two-percent peak-force and work gates and a five-percent normalized full-curve gate; impacts restore the immutable released-assembly checkpoint";
};

enum class ShippingExecutionPhase : std::uint32_t {
    scoreIndent = 0u,
    scoreWithdrawal = 1u,
    foldPanels = 2u,
    closeManufacturerLap = 3u,
    closeEndFlaps = 4u,
    releaseTemporaryClamps = 5u,
    settleReleasedAssembly = 6u,
    quasiStaticCompression = 7u,
    gravityTransfer = 8u,
    impact = 9u,
};

// Facts are derived from completed accepted transactions and their full
// checkpoint, never from a requested controller action or a rejected
// candidate. A single evidence object may carry earlier facts forward.
struct ShippingExecutionGateEvidence {
    bool nativeStepAccepted = false;
    bool completeSnapshotAvailable = false;
    bool solverCertificateValid = false;
    std::uint64_t checkpointSourcePhysicsFingerprint = 0u;
    std::uint64_t expectedSourcePhysicsFingerprint = 0u;
    std::uint64_t checkpointDeviceProgramFingerprint = 0u;
    std::uint64_t expectedDeviceProgramFingerprint = 0u;
    std::uint64_t acceptedStepCount = 0u;
    std::uint32_t rejectedStepCount = 0u;
    std::string checkpointSha256;
    std::string boardRecipeId;
    bool scorePassAccepted = false;
    bool scoreToolsWithdrawn = false;
    bool foldActionsAccepted = false;
    bool manufacturerLapActivated = false;
    bool manufacturerLapClosed = false;
    std::uint32_t endOverlapPatchesActivated = 0u;
    bool topAndBottomEndsClosed = false;
    bool temporaryClampsReleased = false;
    bool toolingParkedOrRemoved = false;
    bool releasedAssemblySettled = false;
    bool acceptedCompressionPeakAvailable = false;
    bool finiteMassSupportTransferAccepted = false;
    bool impactCheckpointRestored = false;
};

struct ShippingExecutionGateDecision {
    bool admitted = false;
    std::string reason;
};

inline ShippingExecutionGateDecision admitShippingExecutionPhase(
    const ShippingExecutionPhase phase,
    const ShippingExecutionGateEvidence& evidence) {
    const auto reject = [](std::string reason) {
        return ShippingExecutionGateDecision{false, std::move(reason)};
    };
    const bool releasedAssemblyHistoryComplete =
        evidence.scorePassAccepted && evidence.scoreToolsWithdrawn &&
        evidence.foldActionsAccepted && evidence.manufacturerLapActivated &&
        evidence.manufacturerLapClosed &&
        evidence.endOverlapPatchesActivated == 8u &&
        evidence.topAndBottomEndsClosed && evidence.temporaryClampsReleased &&
        evidence.toolingParkedOrRemoved && evidence.releasedAssemblySettled;
    if (!evidence.nativeStepAccepted || !evidence.completeSnapshotAvailable ||
        !evidence.solverCertificateValid ||
        evidence.checkpointSourcePhysicsFingerprint == 0u ||
        evidence.checkpointDeviceProgramFingerprint == 0u ||
        evidence.checkpointSourcePhysicsFingerprint !=
            evidence.expectedSourcePhysicsFingerprint ||
        evidence.checkpointDeviceProgramFingerprint !=
            evidence.expectedDeviceProgramFingerprint ||
        evidence.acceptedStepCount == 0u || evidence.rejectedStepCount != 0u ||
        !isLowerHexDigest(evidence.checkpointSha256) ||
        evidence.boardRecipeId.empty())
        return reject("phase requires a matching, complete accepted checkpoint with no rejected steps");
    switch (phase) {
    case ShippingExecutionPhase::scoreIndent:
        return {true, "accepted checkpoint admitted for score indentation"};
    case ShippingExecutionPhase::scoreWithdrawal:
        if (evidence.scorePassAccepted)
            return {true, "accepted scored state admitted for tool withdrawal"};
        return reject("score withdrawal requires an accepted completed indentation pass");
    case ShippingExecutionPhase::foldPanels:
        if (evidence.scorePassAccepted && evidence.scoreToolsWithdrawn)
            return {true, "accepted score and withdrawal state admitted for panel folds"};
        return reject("panel folds require accepted scoring and withdrawn scoring tools");
    case ShippingExecutionPhase::closeManufacturerLap:
        if (evidence.scorePassAccepted && evidence.scoreToolsWithdrawn &&
            evidence.foldActionsAccepted)
            return {true, "accepted folded state admitted for manufacturer lap closure"};
        return reject("lap closure requires accepted score, tool withdrawal, and panel folds");
    case ShippingExecutionPhase::closeEndFlaps:
        if (evidence.manufacturerLapActivated && evidence.manufacturerLapClosed &&
            evidence.endOverlapPatchesActivated == 8u)
            return {true, "accepted lap and eight overlap activations admit end closure"};
        return reject("end closure requires an accepted lap and all eight overlap-patch activations");
    case ShippingExecutionPhase::releaseTemporaryClamps:
        if (evidence.manufacturerLapClosed && evidence.topAndBottomEndsClosed &&
            evidence.endOverlapPatchesActivated == 8u)
            return {true, "accepted closure geometry admits clamp release"};
        return reject("clamp release requires accepted lap and both end closures");
    case ShippingExecutionPhase::settleReleasedAssembly:
        if (evidence.temporaryClampsReleased && evidence.toolingParkedOrRemoved)
            return {true, "released accepted assembly admitted for settling"};
        return reject("settling requires released temporary clamps and parked or removed tools");
    case ShippingExecutionPhase::quasiStaticCompression:
        if (releasedAssemblyHistoryComplete)
            return {true, "settled released assembly admitted for a load branch"};
        return reject("load branch requires the complete released and settled assembly checkpoint");
    case ShippingExecutionPhase::gravityTransfer:
        if (releasedAssemblyHistoryComplete &&
            evidence.finiteMassSupportTransferAccepted)
            return {true, "released assembly and accepted finite-mass support seed admit gravity transfer"};
        return reject("gravity transfer requires a complete assembly and accepted finite-mass support seed");
    case ShippingExecutionPhase::impact:
        if (releasedAssemblyHistoryComplete &&
            evidence.acceptedCompressionPeakAvailable &&
            evidence.impactCheckpointRestored)
            return {true, "restored assembly and measured peak admit impact"};
        return reject("impact requires a restored released checkpoint and accepted compression peak");
    }
    return reject("unknown shipping phase");
}

struct ReleasedAssemblyCheckpoint {
    std::string checkpointId;
    std::string checkpointSha256;
    std::uint64_t sourcePhysicsFingerprint = 0u;
    std::uint64_t compiledWorldFingerprint = 0u;
    bool accepted = false;
    bool completeNativeSnapshot = false;
    bool temporaryGripsReleased = false;
    bool toolingParkedOrRemoved = false;
    bool settled = false;
    bool manufacturerLapClosedByGeometry = false;
    bool bottomEndClosedByGeometry = false;
    bool topEndClosedByGeometry = false;
    bool allAssemblyActionsAccepted = false;
    std::uint32_t rejectedStepCount = 0u;
    double maximumNodalSpeedMPerS = std::numeric_limits<double>::infinity();
    double normalizedForceImbalance = std::numeric_limits<double>::infinity();
    double maximumEndCornerGapM = std::numeric_limits<double>::infinity();
    ShippingInsideDimensions measuredInside{};
    std::string boardRecipeId;
    double measuredInsideCenterX = 0.0;
    double measuredInsideCenterZ = 0.0;
    double measuredRoofContactY = 0.0;
    ShippingOuterRoofMeasurement measuredOuterRoof{};
};

struct AcceptedCompressionPeak {
    bool completed = false;
    bool acceptedEvidence = false;
    std::string runId;
    std::string resultSha256;
    std::string sourceAssemblyCheckpointSha256;
    std::string boardRecipeId;
    std::uint64_t acceptedStepCount = 0u;
    double peakReactionForceN = 0.0;
};

struct ShippingDropCase {
    std::string caseId;
    double forcePeakMassRatio = 0.0;
    double massKg = 0.0;
    double dropHeightM = 0.100;
    double gravityMPerS2 = 9.80665;
    double impactorPlanLengthM = 0.0;
    double impactorPlanWidthM = 0.0;
    std::string impactorPlanBasis = "1.2 times accepted outer-roof x/z extents";
    double impactorHeightM = 0.050;
    double impactorCenterX = 0.0;
    double impactorCenterZ = 0.0;
    double impactorStartCenterY = 0.0;
    std::string peakRunId;
    std::string peakResultSha256;
    std::string restoreCheckpointId;
    std::string restoreCheckpointSha256;
};

inline std::vector<ShippingDropCase> makeShippingDropCases(
    const AcceptedCompressionPeak& peak,
    const ReleasedAssemblyCheckpoint& checkpoint,
    const ShippingInsideDimensions targetInside = {}) {
    if (!peak.completed || !peak.acceptedEvidence || peak.acceptedStepCount == 0u ||
        peak.runId.empty() || !isLowerHexDigest(peak.resultSha256) ||
        peak.boardRecipeId.empty() ||
        !(std::isfinite(peak.peakReactionForceN) && peak.peakReactionForceN > 0.0))
        throw std::invalid_argument(
            "gravity-mass cases require an accepted measured compression Fpeak");
    if (!checkpoint.accepted || !checkpoint.completeNativeSnapshot ||
        !checkpoint.temporaryGripsReleased || !checkpoint.toolingParkedOrRemoved ||
        !checkpoint.settled || !checkpoint.allAssemblyActionsAccepted ||
        !checkpoint.manufacturerLapClosedByGeometry ||
        !checkpoint.bottomEndClosedByGeometry ||
        !checkpoint.topEndClosedByGeometry ||
        checkpoint.rejectedStepCount != 0u ||
        !(std::isfinite(checkpoint.maximumNodalSpeedMPerS) &&
          checkpoint.maximumNodalSpeedMPerS <= 1.0e-4) ||
        !(std::isfinite(checkpoint.normalizedForceImbalance) &&
          checkpoint.normalizedForceImbalance <= 1.0e-3) ||
        !(std::isfinite(checkpoint.maximumEndCornerGapM) &&
          checkpoint.maximumEndCornerGapM <= 1.0e-3) ||
        !std::isfinite(checkpoint.measuredInside.lengthX) ||
        !std::isfinite(checkpoint.measuredInside.heightY) ||
        !std::isfinite(checkpoint.measuredInside.widthZ) ||
        checkpoint.boardRecipeId.empty() || checkpoint.checkpointId.empty() ||
        !isLowerHexDigest(checkpoint.checkpointSha256))
        throw std::invalid_argument(
            "drop cases require a complete released, settled assembly checkpoint");
    if (peak.boardRecipeId != checkpoint.boardRecipeId ||
        peak.sourceAssemblyCheckpointSha256 != checkpoint.checkpointSha256)
        throw std::invalid_argument(
            "compression Fpeak must be bound to the exact assembly checkpoint used by all drop cases");
    if (!(std::isfinite(targetInside.widthZ) && targetInside.widthZ > 0.0) ||
        !std::isfinite(checkpoint.measuredInsideCenterX) ||
        !std::isfinite(checkpoint.measuredInsideCenterZ) ||
        !std::isfinite(checkpoint.measuredRoofContactY) ||
        !std::isfinite(checkpoint.measuredInside.lengthX) ||
        !std::isfinite(checkpoint.measuredInside.heightY) ||
        !std::isfinite(checkpoint.measuredInside.widthZ) ||
        std::abs(checkpoint.measuredInside.lengthX - targetInside.lengthX) > 1.0e-3 ||
        std::abs(checkpoint.measuredInside.heightY - targetInside.heightY) > 1.0e-3 ||
        std::abs(checkpoint.measuredInside.widthZ - targetInside.widthZ) > 1.0e-3)
        throw std::invalid_argument("drop checkpoint geometry is invalid");
    const auto footprint = makeShippingImpactorFootprintPlan(
        checkpoint.measuredOuterRoof, checkpoint.checkpointSha256,
        checkpoint.sourcePhysicsFingerprint,
        checkpoint.compiledWorldFingerprint);
    constexpr double gravity = 9.80665;
    constexpr std::array<double, 3> ratios{0.5, 1.0, 1.5};
    constexpr std::array<double, 2> offsets{0.0, 0.25};
    std::vector<ShippingDropCase> cases;
    cases.reserve(ratios.size() * offsets.size());
    for (const double ratio : ratios) {
        for (const double quarterWidth : offsets) {
            ShippingDropCase item;
            item.forcePeakMassRatio = ratio;
            item.massKg = ratio * peak.peakReactionForceN / gravity;
            item.impactorPlanLengthM = footprint.fullLengthX;
            item.impactorPlanWidthM = footprint.fullWidthZ;
            item.impactorCenterX = footprint.centerX;
            item.impactorCenterZ = footprint.centerZ +
                (quarterWidth == 0.0 ? 0.0 : footprint.quarterWidthOffsetZ);
            item.impactorStartCenterY = footprint.roofY +
                item.dropHeightM + 0.5 * item.impactorHeightM;
            item.restoreCheckpointId = checkpoint.checkpointId;
            item.restoreCheckpointSha256 = checkpoint.checkpointSha256;
            item.peakRunId = peak.runId;
            item.peakResultSha256 = peak.resultSha256;
            std::ostringstream name;
            name << "drop-fpeak-" << std::fixed << std::setprecision(1) << ratio
                 << (quarterWidth == 0.0 ? "-center" : "-quarter-width-positive-z");
            item.caseId = name.str();
            cases.push_back(std::move(item));
        }
    }
    return cases;
}

struct ReleasedAssemblyObservation {
    bool nativeAssemblyTrialCompleted = false;
    bool acceptedCheckpoint = false;
    bool allAssemblyActionsAccepted = false;
    std::uint32_t rejectedStepCount = 0u;
    bool allTemporaryGripsReleased = false;
    bool allToolsParkedOrRemoved = false;
    bool manufacturerLapActivated = false;
    bool manufacturerLapClosedByGeometry = false;
    bool bottomEndClosedByGeometry = false;
    bool topEndClosedByGeometry = false;
    std::uint32_t flapOverlapPatchesActivated = 0u;
    double maximumEndCornerGapM = std::numeric_limits<double>::infinity();
    double endShapeResidualM = std::numeric_limits<double>::infinity();
    bool settled = false;
    double maximumNodalSpeedMPerS = std::numeric_limits<double>::infinity();
    double normalizedForceImbalance = std::numeric_limits<double>::infinity();
    ShippingInsideDimensions measuredInside{};
    bool physicalBoxTestAvailable = false;
    bool recipeMatchedPhysicalStock = false;
    bool creaseLawCalibratedForStock = false;
    bool corrugatorStarchLawCalibratedForStock = false;
    bool closureHmaLawCalibratedForStock = false;
};

struct ReleasedAssemblyQualification {
    bool actionAndTransactionGate = false;
    bool releaseAndSettlementGate = false;
    bool clearDimensionGate = false;
    // Completion of the declared native simulation process is independent of
    // whether the analogue/source data have been calibrated for the stock.
    bool simulationAssemblyComplete = false;
    bool materialAndInterfaceCalibrationGate = false;
    bool physicalBoxTestGate = false;
    bool nativeAssemblyMechanicsQualified = false;
    bool physicalShippingBoxQualified = false;
};

inline ReleasedAssemblyQualification qualifyReleasedAssembly(
    const ShippingBoxDesign& design,
    const ReleasedAssemblyObservation& observation,
    const double clearDimensionToleranceFraction = 0.01,
    const double closureGeometryToleranceM = 1.0e-3,
    const double settledSpeedBoundMPerS = 1.0e-4,
    const double forceImbalanceBound = 1.0e-3) {
    ReleasedAssemblyQualification result;
    const auto finite = [](const double x) { return std::isfinite(x); };
    validateShippingBoxDesign(design);
    if (!(finite(clearDimensionToleranceFraction) &&
          clearDimensionToleranceFraction >= 0.0 &&
          finite(closureGeometryToleranceM) && closureGeometryToleranceM >= 0.0 &&
          finite(settledSpeedBoundMPerS) && settledSpeedBoundMPerS >= 0.0 &&
          finite(forceImbalanceBound) && forceImbalanceBound >= 0.0))
        throw std::invalid_argument("released-assembly tolerances are invalid");
    const std::array<double, 3> dimensionTolerance{
        clearDimensionToleranceFraction * design.defaults.inside.lengthX,
        clearDimensionToleranceFraction * design.defaults.inside.heightY,
        clearDimensionToleranceFraction * design.defaults.inside.widthZ};
    result.actionAndTransactionGate = observation.acceptedCheckpoint &&
        observation.allAssemblyActionsAccepted && observation.rejectedStepCount == 0u;
    result.releaseAndSettlementGate = observation.allTemporaryGripsReleased &&
        observation.allToolsParkedOrRemoved && observation.settled &&
        finite(observation.maximumNodalSpeedMPerS) &&
        observation.maximumNodalSpeedMPerS <= settledSpeedBoundMPerS &&
        finite(observation.normalizedForceImbalance) &&
        observation.normalizedForceImbalance <= forceImbalanceBound;
    result.clearDimensionGate = finite(observation.measuredInside.lengthX) &&
        finite(observation.measuredInside.heightY) &&
        finite(observation.measuredInside.widthZ) &&
        observation.manufacturerLapClosedByGeometry &&
        observation.bottomEndClosedByGeometry &&
        observation.topEndClosedByGeometry &&
        finite(observation.maximumEndCornerGapM) &&
        observation.maximumEndCornerGapM <= closureGeometryToleranceM &&
        finite(observation.endShapeResidualM) &&
        observation.endShapeResidualM <= closureGeometryToleranceM &&
        std::abs(observation.measuredInside.lengthX - design.defaults.inside.lengthX) <=
            dimensionTolerance[0] &&
        std::abs(observation.measuredInside.heightY - design.defaults.inside.heightY) <=
            dimensionTolerance[1] &&
        std::abs(observation.measuredInside.widthZ - design.defaults.inside.widthZ) <=
            dimensionTolerance[2];
    result.simulationAssemblyComplete =
        observation.nativeAssemblyTrialCompleted &&
        result.actionAndTransactionGate && result.releaseAndSettlementGate &&
        result.clearDimensionGate && observation.manufacturerLapActivated &&
        observation.flapOverlapPatchesActivated == 8u;
    result.materialAndInterfaceCalibrationGate =
        design.allowanceFromPhysicalCoupon && observation.recipeMatchedPhysicalStock &&
        observation.creaseLawCalibratedForStock &&
        observation.corrugatorStarchLawCalibratedForStock &&
        observation.closureHmaLawCalibratedForStock &&
        observation.manufacturerLapActivated &&
        observation.flapOverlapPatchesActivated == 8u;
    result.physicalBoxTestGate = observation.physicalBoxTestAvailable;
    result.nativeAssemblyMechanicsQualified =
        result.simulationAssemblyComplete &&
        result.materialAndInterfaceCalibrationGate;
    result.physicalShippingBoxQualified =
        result.nativeAssemblyMechanicsQualified && result.physicalBoxTestGate;
    return result;
}

} // namespace numi::cardboard
