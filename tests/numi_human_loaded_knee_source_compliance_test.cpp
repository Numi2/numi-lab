#include "metalrobo/NumiHumanSourceConstraints.hpp"

#include <algorithm>
#include <iostream>
#include <string>

namespace {

constexpr metalrobo::NumiHumanLoadedKneeDigest kSourceArchiveSHA256{
    0x28u, 0x0du, 0x29u, 0x7au, 0xa4u, 0x96u, 0xacu, 0xccu,
    0xf3u, 0xf1u, 0xc5u, 0x37u, 0x3au, 0x13u, 0x04u, 0xd2u,
    0x3fu, 0x95u, 0x69u, 0x36u, 0x2cu, 0x2du, 0x69u, 0x60u,
    0x91u, 0x01u, 0x28u, 0xbfu, 0xbau, 0x14u, 0x49u, 0x75u,
};

bool present(const metalrobo::NumiHumanLoadedKneeDigest &value) {
  return std::any_of(value.begin(), value.end(), [](const std::uint8_t byte) {
    return byte != 0u;
  });
}

} // namespace

int main(const int argc, const char *const argv[]) {
  if (argc != 7) {
    std::cerr
        << "usage: loaded-knee-source-compliance-test MANIFEST BINDING "
           "OWNERSHIP SOURCE_COMPLIANCE NHEQ2 NHLIM1\n";
    return 64;
  }

  metalrobo::NumiHumanLoadedKneeBindingAdmissionV1 base;
  std::string error;
  if (!metalrobo::loadNumiHumanLoadedKneeBindingV1(argv[1], argv[2], argv[3],
                                                   base, error)) {
    std::cerr << error << '\n';
    return 1;
  }

  metalrobo::NumiHumanLoadedKneeSourceComplianceAdmissionV1 admission;
  if (!metalrobo::loadNumiHumanLoadedKneeSourceComplianceV1(
          argv[4], argv[5], argv[6], base, admission, error)) {
    std::cerr << error << '\n';
    return 1;
  }
  if (admission.baseManifestSHA256 != base.authoring.manifestSHA256 ||
      admission.baseManifestFileSHA256 != base.manifestFileSHA256 ||
      !present(admission.bindingSHA256) || !present(admission.fileSHA256) ||
      !present(admission.jointEqualityFileSHA256) ||
      !present(admission.jointLimitFileSHA256) ||
      admission.sourceArchiveSHA256 != kSourceArchiveSHA256 ||
      admission.nq != 129u ||
      admission.nv != 128u || admission.jointEqualityRowCount != 51u ||
      admission.jointLimitRowCount != 122u || admission.policy != 1u ||
      admission.flags != 1u) {
    std::cerr << "source-compliance admission did not retain its exact "
                 "authoring identities\n";
    return 1;
  }

  metalrobo::NumiHumanSourceConstraintProgramV1 program;
  if (!metalrobo::loadNumiHumanLoadedKneeSourceConstraintProgramV1(
          argv[5], argv[6], base, admission, program, error)) {
    std::cerr << error << '\n';
    return 1;
  }
  numi::matter::RuntimeConfiguration configuration;
  if (!metalrobo::configureNumiHumanSourceConstraintsV1(
          program, configuration, error) ||
      configuration.humanJointEqualities.size() !=
          admission.jointEqualityRowCount ||
      configuration.humanJointLimits.size() !=
          admission.jointLimitRowCount ||
      configuration.humanEqualityDispatch.qCount != admission.nq ||
      configuration.humanEqualityDispatch.dofCount != admission.nv ||
      configuration.humanEqualityDispatch.policy != admission.policy ||
      configuration.humanEqualityDispatch.flags != admission.flags ||
      program.equalityFileSHA256 != admission.jointEqualityFileSHA256 ||
      program.limitFileSHA256 != admission.jointLimitFileSHA256 ||
      program.sourceArchiveSHA256 != admission.sourceArchiveSHA256) {
    std::cerr << "source-compliance admission did not configure its exact "
                 "Matter row programs: "
              << error << '\n';
    return 1;
  }

  auto wrongBaseLink = admission;
  wrongBaseLink.baseManifestSHA256[0u] ^= 1u;
  metalrobo::NumiHumanSourceConstraintProgramV1 rejected;
  if (metalrobo::loadNumiHumanLoadedKneeSourceConstraintProgramV1(
          argv[5], argv[6], base, wrongBaseLink, rejected, error) ||
      !rejected.jointEqualities.empty() || !rejected.jointLimits.empty()) {
    std::cerr << "source-compliance program admitted a different base "
                 "manifest\n";
    return 1;
  }

  auto missingSourceIdentity = admission;
  missingSourceIdentity.sourceArchiveSHA256 = {};
  if (metalrobo::loadNumiHumanLoadedKneeSourceConstraintProgramV1(
          argv[5], argv[6], base, missingSourceIdentity, rejected, error) ||
      !rejected.jointEqualities.empty() || !rejected.jointLimits.empty()) {
    std::cerr << "source-compliance program admitted a missing source "
                 "identity\n";
    return 1;
  }

  std::cout << "loaded_knee_source_compliance_admission=passed"
            << " prepared_state=absent"
            << " runtime_constraint_execution=false"
            << " production_physical_ownership=false\n";
  return 0;
}
