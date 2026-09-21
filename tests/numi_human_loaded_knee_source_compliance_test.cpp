#include "metalrobo/NumiHumanLoadedKneeBinding.hpp"

#include <algorithm>
#include <iostream>
#include <string>

namespace {

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
      !present(admission.jointLimitFileSHA256)) {
    std::cerr << "source-compliance admission did not retain its exact "
                 "authoring identities\n";
    return 1;
  }

  std::cout << "loaded_knee_source_compliance_admission=passed"
            << " prepared_state=absent"
            << " runtime_constraint_execution=false"
            << " production_physical_ownership=false\n";
  return 0;
}
