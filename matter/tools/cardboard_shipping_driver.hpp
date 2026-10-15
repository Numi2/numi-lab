#pragma once

// Strict input selection and fail-closed status helpers for the native
// shipping-box driver. This file does not run FEM, qualify materials, or turn
// assumed offsets into coupon evidence.

#include "cardboard_shipping_protocol.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <istream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace numi::cardboard {

struct ShippingCouponDocument {
    AcceptedScoreAllowanceCoupon coupon{};
    std::string measurementFile;
    std::string evidenceFileSha256;
    bool completeScoreFoldReleaseGates = false;
    bool scoringDamageLawQualified = false;
};

struct ShippingStarchLawParameters {
    std::string materialId;
    double normalPenaltyPaPerM = 0.0;
    double shearPenaltyPaPerM = 0.0;
    double compressionPenaltyPaPerM = 0.0;
    double normalStrengthPa = 0.0;
    double shearStrengthPa = 0.0;
    double fractureEnergyModeIJPerM2 = 0.0;
    double fractureEnergyModeIIJPerM2 = 0.0;
    double bkExponent = 0.0;
    double activationGapToleranceM = 0.0;
    std::string activationEvent;
    // This source class is intentionally not a calibration designation. It
    // records that a caller supplied an unvalidated engineering feasible set.
    std::string sourceClass;
};

struct ShippingHmaLawParameters {
    std::string materialId;
    std::string sourceClass;
    std::string sourceCitation;
    double transferUncertaintyFraction = 0.0;
    double normalPenaltyPaPerM = 0.0;
    double shearPenaltyPaPerM = 0.0;
    double compressionPenaltyPaPerM = 0.0;
    double normalStrengthPa = 0.0;
    double shearStrengthPa = 0.0;
    double fractureEnergyModeIJPerM2 = 0.0;
    double fractureEnergyModeIIJPerM2 = 0.0;
    double bkExponent = 0.0;
    double activationGapToleranceM = 0.0;
};

inline bool isLowerHexDigest(const std::string& digest);

struct ShippingResumeDocument {
    std::string checkpointSha256;
    std::uint64_t sourcePhysicsFingerprint = 0u;
    std::uint64_t deviceProgramFingerprint = 0u;
    std::uint32_t snapshotControlStep = 0u;
    std::uint32_t nextNativeStepIndex = 0u;
    std::uint32_t nextAssemblyActionIndex = 0u;
    double scoreDieHeightM = 0.0;
    bool scorePassComplete = false;
    bool scoreToolsWithdrawn = false;
    bool starchInitialActivationAccepted = false;
    std::uint32_t starchActivationStep = 0u;
    bool tabAnchorActive = false;
    bool temporaryClampsReleased = false;
    bool releasedAssemblySettled = false;
    std::vector<std::string> acceptedHmaEvents;
    std::string stage;
};

inline std::uint64_t parseShippingUnsigned64(const std::string& value,
                                            const std::string& field) {
    std::size_t used = 0u;
    unsigned long long parsed = 0u;
    try { parsed = std::stoull(value, &used); }
    catch (...) { throw std::invalid_argument("shipping resume has invalid " + field); }
    if (used != value.size())
        throw std::invalid_argument("shipping resume has invalid " + field);
    return static_cast<std::uint64_t>(parsed);
}

inline std::uint32_t parseShippingUnsigned32(const std::string& value,
                                            const std::string& field) {
    const std::uint64_t parsed = parseShippingUnsigned64(value, field);
    if (parsed > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("shipping resume has out-of-range " + field);
    return static_cast<std::uint32_t>(parsed);
}

inline bool parseShippingBoolean(const std::string& value,
                                 const std::string& field) {
    if (value == "true") return true;
    if (value == "false") return false;
    throw std::invalid_argument("shipping resume has invalid " + field);
}


inline std::string trimShippingText(std::string value) {
    const auto isSpace = [](const unsigned char c) { return c == ' ' ||
        c == '\t' || c == '\r' || c == '\n'; };
    const auto first = std::find_if_not(value.begin(), value.end(), isSpace);
    const auto last = std::find_if_not(value.rbegin(), value.rend(), isSpace).base();
    return first < last ? std::string(first, last) : std::string{};
}

inline std::map<std::string, std::string> parseShippingProperties(
    std::istream& input,
    const std::set<std::string>& allowedKeys,
    const std::string_view sourceName = "shipping evidence") {
    std::map<std::string, std::string> result;
    std::string line;
    std::uint32_t lineNumber = 0u;
    while (std::getline(input, line)) {
        ++lineNumber;
        line = trimShippingText(std::move(line));
        if (line.empty() || line.front() == '#') continue;
        const std::size_t separator = line.find('=');
        if (separator == std::string::npos)
            throw std::invalid_argument(std::string(sourceName) + ":" +
                std::to_string(lineNumber) + " must use key=value syntax");
        const std::string key = trimShippingText(line.substr(0u, separator));
        const std::string value = trimShippingText(line.substr(separator + 1u));
        if (key.empty() || value.empty())
            throw std::invalid_argument(std::string(sourceName) + ":"+
                std::to_string(lineNumber) + " has an empty key or value");
        if (!allowedKeys.contains(key))
            throw std::invalid_argument(std::string(sourceName) + ":"+
                std::to_string(lineNumber) + " has unsupported key '" + key + "'");
        if (!result.emplace(key, value).second)
            throw std::invalid_argument(std::string(sourceName) + ":"+
                std::to_string(lineNumber) + " repeats key '" + key + "'");
    }
    if (!input.eof())
        throw std::invalid_argument(std::string(sourceName) + " read failed");
    return result;
}

inline const std::string& shippingProperty(
    const std::map<std::string, std::string>& values,
    const std::string& key) {
    const auto found = values.find(key);
    if (found == values.end())
        throw std::invalid_argument("shipping evidence is missing '" + key + "'");
    return found->second;
}

inline double parseShippingFiniteDouble(const std::string& value,
                                        const std::string& field) {
    std::size_t used = 0u;
    double parsed = 0.0;
    try { parsed = std::stod(value, &used); }
    catch (...) { throw std::invalid_argument("shipping evidence has invalid " + field); }
    if (used != value.size() || !std::isfinite(parsed))
        throw std::invalid_argument("shipping evidence has invalid " + field);
    return parsed;
}

inline ShippingResumeDocument parseShippingResumeDocument(
    std::istream& input, const std::string& sourceName = "shipping resume") {
    const std::set<std::string> keys{
        "schema", "checkpoint_sha256", "source_physics_fingerprint",
        "device_program_fingerprint", "snapshot_control_step",
        "next_native_step_index", "next_assembly_action_index",
        "score_die_height_m", "score_pass_complete", "score_tools_withdrawn",
        "starch_initial_activation_accepted", "starch_activation_step",
        "tab_anchor_active", "temporary_clamps_released",
        "released_assembly_settled", "accepted_hma_events", "stage"};
    const auto values = parseShippingProperties(input, keys, sourceName);
    if (shippingProperty(values, "schema") !=
            "numi.cardboard.shipping-resume.v1")
        throw std::invalid_argument("shipping resume schema is unsupported");
    ShippingResumeDocument result;
    result.checkpointSha256 = shippingProperty(values, "checkpoint_sha256");
    result.sourcePhysicsFingerprint = parseShippingUnsigned64(
        shippingProperty(values, "source_physics_fingerprint"),
        "source physics fingerprint");
    result.deviceProgramFingerprint = parseShippingUnsigned64(
        shippingProperty(values, "device_program_fingerprint"),
        "device program fingerprint");
    result.snapshotControlStep = parseShippingUnsigned32(
        shippingProperty(values, "snapshot_control_step"), "snapshot control step");
    result.nextNativeStepIndex = parseShippingUnsigned32(
        shippingProperty(values, "next_native_step_index"), "next native step index");
    result.nextAssemblyActionIndex = parseShippingUnsigned32(
        shippingProperty(values, "next_assembly_action_index"), "next assembly action index");
    result.scoreDieHeightM = parseShippingFiniteDouble(
        shippingProperty(values, "score_die_height_m"), "score die height");
    result.scorePassComplete = parseShippingBoolean(
        shippingProperty(values, "score_pass_complete"), "score pass flag");
    result.scoreToolsWithdrawn = parseShippingBoolean(
        shippingProperty(values, "score_tools_withdrawn"), "tool withdrawal flag");
    result.starchInitialActivationAccepted = parseShippingBoolean(
        shippingProperty(values, "starch_initial_activation_accepted"),
        "starch activation flag");
    result.starchActivationStep = parseShippingUnsigned32(
        shippingProperty(values, "starch_activation_step"), "starch activation step");
    result.tabAnchorActive = parseShippingBoolean(
        shippingProperty(values, "tab_anchor_active"), "tab anchor flag");
    result.temporaryClampsReleased = parseShippingBoolean(
        shippingProperty(values, "temporary_clamps_released"),
        "temporary clamp release flag");
    result.releasedAssemblySettled = parseShippingBoolean(
        shippingProperty(values, "released_assembly_settled"),
        "released assembly flag");
    result.stage = shippingProperty(values, "stage");
    const auto& events = shippingProperty(values, "accepted_hma_events");
    std::size_t start = 0u;
    while (events != "none" && start < events.size()) {
        const std::size_t end = events.find(',', start);
        const std::string event = events.substr(start,
            end == std::string::npos ? std::string::npos : end - start);
        if (event.empty())
            throw std::invalid_argument("shipping resume has an empty HMA event");
        result.acceptedHmaEvents.push_back(event);
        if (end == std::string::npos) break;
        start = end + 1u;
    }
    if (!isLowerHexDigest(result.checkpointSha256) || result.stage.empty() ||
        result.nextAssemblyActionIndex > 15u ||
        !std::isfinite(result.scoreDieHeightM) ||
        (result.starchInitialActivationAccepted &&
         result.starchActivationStep >= result.nextNativeStepIndex))
        throw std::invalid_argument("shipping resume fields are inconsistent");
    return result;
}


inline ShippingCouponDocument parseShippingCouponDocument(
    std::istream& input, const std::string& sourceName = "coupon evidence") {
    const std::set<std::string> keys{
        "schema", "status", "evidence_class", "coupon_id", "board_recipe_id",
        "source_run_id", "measurement_file", "measurement_sha256",
        "score_center_span_minus_inside_x_m",
        "score_center_span_minus_inside_y_m",
        "score_center_span_minus_inside_z_m", "score_fold_release_gates",
        "scoring_damage_law_qualified"};
    const auto values = parseShippingProperties(input, keys, sourceName);
    if (shippingProperty(values, "schema") !=
            "numi.cardboard.score-allowance-coupon.v1")
        throw std::invalid_argument("coupon evidence schema is unsupported");
    ShippingCouponDocument document;
    const auto& status = shippingProperty(values, "status");
    const auto& evidenceClass = shippingProperty(values, "evidence_class");
    if (status != "accepted")
        throw std::invalid_argument("coupon evidence status is not accepted");
    if (evidenceClass == "native_accepted")
        document.coupon.evidenceClass = CouponEvidenceClass::nativeAcceptedCoupon;
    else if (evidenceClass == "physical_accepted")
        document.coupon.evidenceClass = CouponEvidenceClass::physicalAcceptedCoupon;
    else
        throw std::invalid_argument("coupon evidence class is not accepted");
    document.coupon.accepted = true;
    document.coupon.couponId = shippingProperty(values, "coupon_id");
    document.coupon.boardRecipeId = shippingProperty(values, "board_recipe_id");
    document.coupon.sourceRunId = shippingProperty(values, "source_run_id");
    document.coupon.measurementSha256 = shippingProperty(values, "measurement_sha256");
    document.measurementFile = shippingProperty(values, "measurement_file");
    document.coupon.scoreCenterSpanMinusInsideM = {
        parseShippingFiniteDouble(shippingProperty(values,
            "score_center_span_minus_inside_x_m"), "X allowance"),
        parseShippingFiniteDouble(shippingProperty(values,
            "score_center_span_minus_inside_y_m"), "Y allowance"),
        parseShippingFiniteDouble(shippingProperty(values,
            "score_center_span_minus_inside_z_m"), "Z allowance")};
    document.completeScoreFoldReleaseGates = shippingProperty(values,
        "score_fold_release_gates") == "accepted";
    const auto& damageLaw = shippingProperty(values,
        "scoring_damage_law_qualified");
    if (damageLaw != "true" && damageLaw != "false")
        throw std::invalid_argument(
            "coupon evidence scoring_damage_law_qualified must be true or false");
    document.scoringDamageLawQualified = damageLaw == "true";
    if (!document.completeScoreFoldReleaseGates || document.measurementFile.empty() ||
        document.coupon.couponId.empty() || document.coupon.sourceRunId.empty() ||
        !isLowerHexDigest(document.coupon.measurementSha256))
        throw std::invalid_argument(
            "coupon must bind a hashed measurement file and accepted score/fold/release gates");
    return document;
}

inline ShippingStarchLawParameters parseShippingStarchLawParameters(
    std::istream& input, const std::string& sourceName = "starch law parameters") {
    const std::set<std::string> keys{
        "schema", "material_id", "source_class", "normal_penalty_pa_per_m",
        "shear_penalty_pa_per_m", "compression_penalty_pa_per_m",
        "normal_strength_pa", "shear_strength_pa", "fracture_energy_mode_i_j_per_m2",
        "fracture_energy_mode_ii_j_per_m2", "bk_exponent",
        "activation_gap_tolerance_m", "activation_event"};
    const auto values = parseShippingProperties(input, keys, sourceName);
    if (shippingProperty(values, "schema") !=
            "numi.cardboard.starch-mixed-mode-bilinear.v2")
        throw std::invalid_argument("starch interface parameter schema is unsupported");
    ShippingStarchLawParameters result;
    result.materialId = shippingProperty(values, "material_id");
    result.sourceClass = shippingProperty(values, "source_class");
    if (result.materialId.empty() ||
        result.sourceClass != "unvalidated_engineering_assumption")
        throw std::invalid_argument(
            "starch law must have an ID and explicit unvalidated engineering source class");
    result.normalPenaltyPaPerM = parseShippingFiniteDouble(
        shippingProperty(values, "normal_penalty_pa_per_m"), "normal penalty");
    result.shearPenaltyPaPerM = parseShippingFiniteDouble(
        shippingProperty(values, "shear_penalty_pa_per_m"), "shear penalty");
    result.compressionPenaltyPaPerM = parseShippingFiniteDouble(
        shippingProperty(values, "compression_penalty_pa_per_m"), "compression penalty");
    result.normalStrengthPa = parseShippingFiniteDouble(
        shippingProperty(values, "normal_strength_pa"), "normal strength");
    result.shearStrengthPa = parseShippingFiniteDouble(
        shippingProperty(values, "shear_strength_pa"), "shear strength");
    result.fractureEnergyModeIJPerM2 = parseShippingFiniteDouble(
        shippingProperty(values, "fracture_energy_mode_i_j_per_m2"), "mode-I fracture energy");
    result.fractureEnergyModeIIJPerM2 = parseShippingFiniteDouble(
        shippingProperty(values, "fracture_energy_mode_ii_j_per_m2"), "mode-II fracture energy");
    result.bkExponent = parseShippingFiniteDouble(
        shippingProperty(values, "bk_exponent"), "BK exponent");
    result.activationGapToleranceM = parseShippingFiniteDouble(
        shippingProperty(values, "activation_gap_tolerance_m"), "activation gap tolerance");
    result.activationEvent = shippingProperty(values, "activation_event");
    if (result.activationEvent != "initial_accepted_state")
        throw std::invalid_argument(
            "shipping starch currently supports only the explicit initial_accepted_state activation event");
    if (!(result.normalPenaltyPaPerM > 0.0 && result.shearPenaltyPaPerM > 0.0 &&
          result.compressionPenaltyPaPerM > 0.0 && result.normalStrengthPa > 0.0 &&
          result.shearStrengthPa > 0.0 && result.fractureEnergyModeIJPerM2 > 0.0 &&
          result.fractureEnergyModeIIJPerM2 > 0.0 && result.bkExponent >= 1.0 &&
          result.activationGapToleranceM > 0.0))
        throw std::invalid_argument(
            "starch cohesive law values must be positive and BK exponent at least one");
    return result;
}

inline ShippingHmaLawParameters parseShippingHmaLawParameters(
    std::istream& input, const std::string& sourceName = "HMA law parameters") {
    const std::set<std::string> keys{
        "schema", "material_id", "source_class", "source_citation",
        "transfer_uncertainty_fraction", "normal_penalty_pa_per_m",
        "shear_penalty_pa_per_m", "compression_penalty_pa_per_m",
        "normal_strength_pa", "shear_strength_pa",
        "fracture_energy_mode_i_j_per_m2",
        "fracture_energy_mode_ii_j_per_m2", "bk_exponent",
        "activation_gap_tolerance_m"};
    const auto values = parseShippingProperties(input, keys, sourceName);
    if (shippingProperty(values, "schema") !=
            "numi.cardboard.shipping-hma-mixed-mode-bilinear.v1")
        throw std::invalid_argument("shipping HMA interface parameter schema is unsupported");
    ShippingHmaLawParameters result;
    result.materialId = shippingProperty(values, "material_id");
    result.sourceClass = shippingProperty(values, "source_class");
    result.sourceCitation = shippingProperty(values, "source_citation");
    result.transferUncertaintyFraction = parseShippingFiniteDouble(
        shippingProperty(values, "transfer_uncertainty_fraction"),
        "HMA transfer uncertainty");
    result.normalPenaltyPaPerM = parseShippingFiniteDouble(
        shippingProperty(values, "normal_penalty_pa_per_m"), "HMA normal penalty");
    result.shearPenaltyPaPerM = parseShippingFiniteDouble(
        shippingProperty(values, "shear_penalty_pa_per_m"), "HMA shear penalty");
    result.compressionPenaltyPaPerM = parseShippingFiniteDouble(
        shippingProperty(values, "compression_penalty_pa_per_m"),
        "HMA compression penalty");
    result.normalStrengthPa = parseShippingFiniteDouble(
        shippingProperty(values, "normal_strength_pa"), "HMA normal strength");
    result.shearStrengthPa = parseShippingFiniteDouble(
        shippingProperty(values, "shear_strength_pa"), "HMA shear strength");
    result.fractureEnergyModeIJPerM2 = parseShippingFiniteDouble(
        shippingProperty(values, "fracture_energy_mode_i_j_per_m2"),
        "HMA mode-I fracture energy");
    result.fractureEnergyModeIIJPerM2 = parseShippingFiniteDouble(
        shippingProperty(values, "fracture_energy_mode_ii_j_per_m2"),
        "HMA mode-II fracture energy");
    result.bkExponent = parseShippingFiniteDouble(
        shippingProperty(values, "bk_exponent"), "HMA BK exponent");
    result.activationGapToleranceM = parseShippingFiniteDouble(
        shippingProperty(values, "activation_gap_tolerance_m"),
        "HMA activation gap tolerance");
    if (result.materialId.empty() || result.sourceCitation.empty() ||
        (result.sourceClass != "unvalidated_engineering_assumption" &&
         result.sourceClass != "bounded_published_analogue_transfer") ||
        !(result.transferUncertaintyFraction > 0.0 &&
          result.normalPenaltyPaPerM > 0.0 &&
          result.shearPenaltyPaPerM > 0.0 &&
          result.compressionPenaltyPaPerM > 0.0 &&
          result.normalStrengthPa > 0.0 &&
          result.shearStrengthPa > 0.0 &&
          result.fractureEnergyModeIJPerM2 > 0.0 &&
          result.fractureEnergyModeIIJPerM2 > 0.0 &&
          result.bkExponent >= 1.0 &&
          result.activationGapToleranceM > 0.0))
        throw std::invalid_argument(
            "HMA inputs require a cited explicit source class, nonzero transfer uncertainty, and positive constitutive values");
    return result;
}

struct ShippingDesignSelection {
    ShippingBoxDesign design{};
    std::string allowanceDisposition;
    bool qualificationAllowed = false;
};

inline ShippingDesignSelection selectShippingDesign(
    const ShippingBoxDesignDefaults& defaults,
    const ShippingCouponDocument* acceptedCoupon,
    const std::array<double, 3>* explicitEngineeringAllowanceM,
    const bool preflightOnly) {
    ShippingDesignSelection selected;
    if (acceptedCoupon != nullptr) {
        if (!acceptedCoupon->completeScoreFoldReleaseGates)
            throw std::invalid_argument("coupon gate is incomplete");
        selected.design = deriveShippingBoxDesign(defaults,
                                                  acceptedCoupon->coupon);
        selected.allowanceDisposition =
            acceptedCoupon->coupon.evidenceClass ==
                CouponEvidenceClass::nativeAcceptedCoupon
                ? "accepted-native-coupon"
                : "accepted-physical-coupon";
        if (!acceptedCoupon->scoringDamageLawQualified)
            selected.allowanceDisposition += "-allowance-only";
        selected.qualificationAllowed =
            acceptedCoupon->scoringDamageLawQualified;
        return selected;
    }

    AcceptedScoreAllowanceCoupon explicitOffset;
    explicitOffset.accepted = true;
    explicitOffset.evidenceClass = CouponEvidenceClass::syntheticUnitFixture;
    explicitOffset.couponId = preflightOnly
        ? "lower-bound-zero-offset" : "explicit-engineering-assumption";
    explicitOffset.boardRecipeId = defaults.boardRecipeId;
    explicitOffset.sourceRunId = preflightOnly
        ? "not-a-run-lower-bound-only" : "user-supplied-engineering-offset";
    explicitOffset.scoreCenterSpanMinusInsideM =
        explicitEngineeringAllowanceM == nullptr
            ? std::array<double, 3>{0.0, 0.0, 0.0}
            : *explicitEngineeringAllowanceM;
    explicitOffset.measurementSha256 = std::string(64u, '0');
    for (const double offset : explicitOffset.scoreCenterSpanMinusInsideM)
        if (!std::isfinite(offset))
            throw std::invalid_argument("engineering allowance must be finite");
    if (!preflightOnly && explicitEngineeringAllowanceM == nullptr)
        throw std::invalid_argument(
            "native shipping design requires accepted coupon evidence or explicit engineering allowance");
    selected.design = deriveShippingBoxDesign(defaults, explicitOffset);
    selected.allowanceDisposition = preflightOnly
        ? "zero-offset-lower-bound-only" : "explicit-engineering-assumption-unqualified";
    selected.qualificationAllowed = false;
    return selected;
}

} // namespace numi::cardboard
