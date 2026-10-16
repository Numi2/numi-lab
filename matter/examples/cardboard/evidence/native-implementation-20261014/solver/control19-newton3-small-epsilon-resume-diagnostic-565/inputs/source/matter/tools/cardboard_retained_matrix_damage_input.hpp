#pragma once

// Strict, source-bound input for the optional retained Haj-Ali matrix damage
// overlay. This parser supplies no constitutive defaults; callers must bind a
// complete per-region feasible-set file and keep its exact SHA-256.

#include "cardboard_retained_matrix_damage.hpp"
#include "cardboard_shipping_driver.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <istream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

namespace numi::cardboard::retained_damage {

inline constexpr std::string_view kDamageInputSchema =
    "numi.cardboard.retained-matrix-damage-feasible-set.v1";

template <std::size_t N>
inline std::array<double, N> parseDamageNumberArray(
    const std::string& text, const std::string& field) {
    std::array<double, N> values{};
    std::size_t start = 0u;
    for (std::size_t index = 0u; index < N; ++index) {
        const std::size_t end = text.find(',', start);
        if ((index + 1u < N && end == std::string::npos) ||
            (index + 1u == N && end != std::string::npos))
            throw std::invalid_argument(field + " must contain exactly " +
                                        std::to_string(N) + " comma-separated values");
        const std::size_t stop = end == std::string::npos ? text.size() : end;
        const std::string token = trimShippingText(text.substr(start, stop - start));
        values[index] = parseShippingFiniteDouble(token, field + "[" +
                                                   std::to_string(index) + "]");
        start = stop + 1u;
    }
    return values;
}

inline ExplicitSetBinding parseDamageSetBinding(
    const std::map<std::string, std::string>& values,
    const std::string& prefix) {
    ExplicitSetBinding result{
        shippingProperty(values, prefix + "_set_id"),
        shippingProperty(values, prefix + "_sha256"),
        shippingProperty(values, prefix + "_evidence_class")};
    if (result.setId.empty() || !damage::isSha256(result.sha256) ||
        result.evidenceClass.empty())
        throw std::invalid_argument(prefix + " binding is incomplete");
    return result;
}

inline ExplicitDamageSet parseExplicitDamageSet(
    std::istream& input, const std::string& parameterFileSha256,
    const std::string& sourceName = "retained-matrix damage parameters") {
    const std::set<std::string> keys{
        "schema", "set_id", "evidence_class",
        "hardening_source_id", "hardening_dataset_id",
        "hardening_dataset_sha256", "hardening_mapping",
        "plastic_hardening_pa",
        "fracture_set_id", "fracture_sha256", "fracture_evidence_class",
        "fracture_strength_pa", "fracture_energy_j_per_m2",
        "fracture_rate_time_seconds",
        "rate_set_id", "rate_sha256", "rate_evidence_class",
        "densification_modulus_pa", "densification_onset_strain",
        "densification_rate_time_seconds",
        "densification_set_id", "densification_sha256",
        "densification_evidence_class", "damage_cap", "smooth_strain_width"};
    const auto values = parseShippingProperties(input, keys, sourceName);
    if (shippingProperty(values, "schema") != kDamageInputSchema)
        throw std::invalid_argument("retained-matrix damage input schema is unsupported");
    if (!damage::isSha256(parameterFileSha256))
        throw std::invalid_argument("retained-matrix parameter file SHA-256 is invalid");

    ExplicitDamageSet result;
    result.setId = shippingProperty(values, "set_id");
    result.sha256 = parameterFileSha256;
    result.evidenceClass = shippingProperty(values, "evidence_class");
    result.hardeningBinding = {
        shippingProperty(values, "hardening_source_id"),
        shippingProperty(values, "hardening_dataset_id"),
        shippingProperty(values, "hardening_dataset_sha256"),
        shippingProperty(values, "hardening_mapping")};
    result.plasticHardeningPa = parseDamageNumberArray<6u>(
        shippingProperty(values, "plastic_hardening_pa"), "plastic_hardening_pa");
    result.fractureBinding = parseDamageSetBinding(values, "fracture");
    const auto strengths = parseDamageNumberArray<8u>(
        shippingProperty(values, "fracture_strength_pa"), "fracture_strength_pa");
    const auto fractureEnergies = parseDamageNumberArray<8u>(
        shippingProperty(values, "fracture_energy_j_per_m2"),
        "fracture_energy_j_per_m2");
    const auto rateTimes = parseDamageNumberArray<8u>(
        shippingProperty(values, "fracture_rate_time_seconds"),
        "fracture_rate_time_seconds");
    for (std::size_t mode = 0u; mode < result.fractureModes.size(); ++mode)
        result.fractureModes[mode] = {
            strengths[mode], fractureEnergies[mode], rateTimes[mode]};
    result.rateBinding = parseDamageSetBinding(values, "rate");
    result.densificationModulusPa = parseDamageNumberArray<3u>(
        shippingProperty(values, "densification_modulus_pa"),
        "densification_modulus_pa");
    result.densificationOnsetStrain = parseDamageNumberArray<3u>(
        shippingProperty(values, "densification_onset_strain"),
        "densification_onset_strain");
    result.densificationRateTimeConstantSeconds = parseDamageNumberArray<3u>(
        shippingProperty(values, "densification_rate_time_seconds"),
        "densification_rate_time_seconds");
    result.densificationBinding = parseDamageSetBinding(values, "densification");
    result.damageCap = parseShippingFiniteDouble(
        shippingProperty(values, "damage_cap"), "damage_cap");
    result.smoothStrainWidth = parseShippingFiniteDouble(
        shippingProperty(values, "smooth_strain_width"), "smooth_strain_width");

    try {
        validateExplicitSet(result);
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(sourceName + ": " + error.what());
    }
    return result;
}

} // namespace numi::cardboard::retained_damage
