#include "cardboard_shipping_driver.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
using namespace numi::cardboard;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string couponText(const std::string& evidenceClass = "native_accepted",
                       const std::string& gates = "accepted",
                       const std::string& law = "true") {
    return "schema=numi.cardboard.score-allowance-coupon.v1\n"
        "status=accepted\n"
        "evidence_class=" + evidenceClass + "\n"
        "coupon_id=coupon-A\n"
        "board_recipe_id=literature2009\n"
        "source_run_id=run-A\n"
        "measurement_file=measurements.csv\n"
        "measurement_sha256=" + std::string(64u, 'a') + "\n"
        "score_center_span_minus_inside_x_m=0.001\n"
        "score_center_span_minus_inside_y_m=-0.002\n"
        "score_center_span_minus_inside_z_m=0.003\n"
        "score_fold_release_gates=" + gates + "\n"
        "scoring_damage_law_qualified=" + law + "\n";
}

std::string starchLawText(const std::string& sourceClass =
                               "unvalidated_engineering_assumption") {
    return "schema=numi.cardboard.starch-mixed-mode-bilinear.v2\n"
        "material_id=fixture-only\n"
        "source_class=" + sourceClass + "\n"
        "normal_penalty_pa_per_m=1e12\n"
        "shear_penalty_pa_per_m=1e12\n"
        "compression_penalty_pa_per_m=1e12\n"
        "normal_strength_pa=1e6\n"
        "shear_strength_pa=1e6\n"
        "fracture_energy_mode_i_j_per_m2=100\n"
        "fracture_energy_mode_ii_j_per_m2=200\n"
        "bk_exponent=1.5\n"
        "activation_gap_tolerance_m=1e-5\n"
        "activation_event=initial_accepted_state\n";
}

void testNativeCouponAndSelection() {
    std::istringstream input(couponText());
    const auto document = parseShippingCouponDocument(input);
    require(document.completeScoreFoldReleaseGates &&
            document.scoringDamageLawQualified &&
            document.coupon.evidenceClass == CouponEvidenceClass::nativeAcceptedCoupon,
            "accepted native score allowance lost evidence class or gates");
    ShippingBoxDesignDefaults defaults;
    defaults.boardRecipeId = "literature2009";
    const auto selected = selectShippingDesign(defaults, &document, nullptr, false);
    require(selected.qualificationAllowed &&
            selected.allowanceDisposition == "accepted-native-coupon" &&
            selected.design.allowanceFromNativeCoupon &&
            !selected.design.allowanceFromPhysicalCoupon,
            "native coupon was mislabeled as physical or not accepted");
    require(std::abs(selected.design.blankConfig.panelLength - 0.301) < 1.0e-12 &&
            std::abs(selected.design.blankConfig.wallHeight - 0.198) < 1.0e-12 &&
            std::abs(selected.design.blankConfig.panelWidth - 0.203) < 1.0e-12,
            "signed score allowance did not map to X/Y/Z blank dimensions");
}

void testPhysicalAndEngineeringDisposition() {
    std::istringstream physicalInput(couponText("physical_accepted"));
    const auto physical = parseShippingCouponDocument(physicalInput);
    const auto qualified = selectShippingDesign(
        ShippingBoxDesignDefaults{}, &physical, nullptr, false);
    require(qualified.qualificationAllowed &&
            qualified.design.allowanceFromPhysicalCoupon &&
            !qualified.design.allowanceFromNativeCoupon,
            "physical coupon evidence was mislabeled");

    const auto preflight = selectShippingDesign(
        ShippingBoxDesignDefaults{}, nullptr, nullptr, true);
    require(!preflight.qualificationAllowed &&
            preflight.allowanceDisposition == "zero-offset-lower-bound-only" &&
            preflight.design.blankConfig.panelLength == 0.300,
            "missing-coupon preflight did not remain an unqualified zero-offset lower bound");

    const std::array<double, 3> offsets{0.001, -0.002, 0.003};
    const auto assumed = selectShippingDesign(
        ShippingBoxDesignDefaults{}, nullptr, &offsets, false);
    require(!assumed.qualificationAllowed &&
            assumed.allowanceDisposition ==
                "explicit-engineering-assumption-unqualified" &&
            assumed.design.blankConfig.wallHeight == 0.198,
            "explicit allowance assumption was promoted to qualified design");
    bool missingOffsetRejected = false;
    try { (void)selectShippingDesign(ShippingBoxDesignDefaults{}, nullptr, nullptr, false); }
    catch (const std::invalid_argument&) { missingOffsetRejected = true; }
    require(missingOffsetRejected,
            "shipping execution admitted without coupon or explicit allowance");

    std::istringstream allowanceOnlyInput(couponText("native_accepted",
                                                       "accepted", "false"));
    const auto allowanceOnly = parseShippingCouponDocument(allowanceOnlyInput);
    const auto designOnly = selectShippingDesign(
        ShippingBoxDesignDefaults{}, &allowanceOnly, nullptr, false);
    require(!designOnly.qualificationAllowed &&
            designOnly.allowanceDisposition ==
                "accepted-native-coupon-allowance-only" &&
            designOnly.design.allowanceFromNativeCoupon,
            "measured allowance was discarded because score damage remains unqualified");
}

void testFailClosedCouponParser() {
    for (const std::string& invalid : {
             couponText("synthetic_fixture"),
             couponText("native_accepted", "pending"),
             couponText("native_accepted", "accepted", "maybe")}) {
        bool rejected = false;
        try {
            std::istringstream input(invalid);
            (void)parseShippingCouponDocument(input);
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "coupon parser admitted an unsupported or incomplete record");
    }
    bool duplicateRejected = false;
    try {
        std::istringstream input(couponText() + "status=accepted\n");
        (void)parseShippingCouponDocument(input);
    } catch (const std::invalid_argument&) { duplicateRejected = true; }
    require(duplicateRejected, "coupon parser admitted duplicate keys");
    bool unknownRejected = false;
    try {
        std::istringstream input(couponText() + "guessed_parameter=1\n");
        (void)parseShippingCouponDocument(input);
    } catch (const std::invalid_argument&) { unknownRejected = true; }
    require(unknownRejected, "coupon parser admitted unknown keys");

    std::istringstream lawInput(starchLawText());
    const ShippingStarchLawParameters law =
        parseShippingStarchLawParameters(lawInput);
    require(law.materialId == "fixture-only" &&
            law.sourceClass == "unvalidated_engineering_assumption" &&
            law.normalPenaltyPaPerM == 1.0e12 &&
            law.fractureEnergyModeIIJPerM2 == 200.0 &&
            law.activationEvent == "initial_accepted_state",
            "explicit starch interface parameters were not preserved");
    bool legacyControlStepRejected = false;
    try {
        std::string legacy = starchLawText();
        const auto marker = legacy.find("activation_event=initial_accepted_state\n");
        legacy.replace(marker, std::string("activation_event=initial_accepted_state\n").size(),
                       "activation_control_step=7\n");
        std::istringstream input(legacy);
        (void)parseShippingStarchLawParameters(input);
    } catch (const std::invalid_argument&) { legacyControlStepRejected = true; }
    require(legacyControlStepRejected,
            "starch parser silently accepted obsolete control-step activation");
    bool unsupportedEventRejected = false;
    try {
        std::string unsupported = starchLawText();
        const auto marker = unsupported.find("initial_accepted_state");
        unsupported.replace(marker, std::string("initial_accepted_state").size(),
                            "manufacturer_lap");
        std::istringstream input(unsupported);
        (void)parseShippingStarchLawParameters(input);
    } catch (const std::invalid_argument&) { unsupportedEventRejected = true; }
    require(unsupportedEventRejected,
            "shipping parser admitted an unsupported non-startup starch activation event");
    bool unmarkedLawRejected = false;
    try {
        std::istringstream input(starchLawText("starch_calibrated"));
        (void)parseShippingStarchLawParameters(input);
    } catch (const std::invalid_argument&) { unmarkedLawRejected = true; }
    require(unmarkedLawRejected,
            "caller-supplied cohesive law was mislabeled as calibrated starch");
}
} // namespace

int main() {
    try {
        testNativeCouponAndSelection();
        testPhysicalAndEngineeringDisposition();
        testFailClosedCouponParser();
        std::cout << "cardboard shipping driver checks passed: 24\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "cardboard shipping driver check failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
