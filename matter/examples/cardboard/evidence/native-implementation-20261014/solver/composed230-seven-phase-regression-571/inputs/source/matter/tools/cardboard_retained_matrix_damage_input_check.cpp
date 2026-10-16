#include "cardboard_retained_matrix_damage_input.hpp"

#include <stdexcept>
#include <sstream>
#include <string>

namespace {
void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::string fixture() {
    return
        "schema=numi.cardboard.retained-matrix-damage-feasible-set.v1\n"
        "set_id=synthetic_fixture_only\n"
        "evidence_class=synthetic_unit_test_only\n"
        "hardening_source_id=fixture_source\n"
        "hardening_dataset_id=fixture_hardening\n"
        "hardening_dataset_sha256=" + std::string(64, '1') + "\n"
        "hardening_mapping=synthetic_explicit_values\n"
        "plastic_hardening_pa=1,2,3,4,5,6\n"
        "fracture_set_id=fixture_fracture\n"
        "fracture_sha256=" + std::string(64, '2') + "\n"
        "fracture_evidence_class=synthetic_unit_test_only\n"
        "fracture_strength_pa=10,11,12,13,14,15,16,17\n"
        "fracture_energy_j_per_m2=20,21,22,23,24,25,26,27\n"
        "fracture_rate_time_seconds=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8\n"
        "rate_set_id=fixture_rate\n"
        "rate_sha256=" + std::string(64, '3') + "\n"
        "rate_evidence_class=synthetic_unit_test_only\n"
        "densification_modulus_pa=30,31,32\n"
        "densification_onset_strain=0.1,0.2,0.3\n"
        "densification_rate_time_seconds=1,2,3\n"
        "densification_set_id=fixture_densification\n"
        "densification_sha256=" + std::string(64, '4') + "\n"
        "densification_evidence_class=synthetic_unit_test_only\n"
        "damage_cap=1\n"
        "smooth_strain_width=1e-7\n";
}
}

int main() {
    using numi::cardboard::retained_damage::parseExplicitDamageSet;
    const std::string source = fixture();
    std::istringstream valid(source);
    const auto parsed = parseExplicitDamageSet(valid, std::string(64, 'a'));
    require(parsed.setId == "synthetic_fixture_only" &&
                parsed.sha256 == std::string(64, 'a') &&
                parsed.plasticHardeningPa[5] == 6.0 &&
                parsed.fractureModes[7].fractureEnergyJPerM2 == 27.0 &&
                parsed.densificationRateTimeConstantSeconds[2] == 3.0,
            "explicit damage-set parser lost file identity or values");

    bool rejected = false;
    try {
        std::istringstream missing(source.substr(0, source.find("damage_cap=")));
        (void)parseExplicitDamageSet(missing, std::string(64, 'a'));
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "explicit damage-set parser accepted missing damage-cap input");

    rejected = false;
    try {
        std::istringstream badHash(source);
        (void)parseExplicitDamageSet(badHash, "not-a-sha256");
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "explicit damage-set parser accepted an invalid file hash");

    rejected = false;
    try {
        std::string wrongArray = source;
        const auto start = wrongArray.find("plastic_hardening_pa=");
        const auto end = wrongArray.find('\n', start);
        wrongArray.replace(start, end - start,
                           "plastic_hardening_pa=1,2,3,4,5");
        std::istringstream badArray(wrongArray);
        (void)parseExplicitDamageSet(badArray, std::string(64, 'a'));
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "explicit damage-set parser accepted a short hardening array");
    return 0;
}
