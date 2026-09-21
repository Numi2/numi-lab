#include "metalrobo/NumiHumanLoadedKneeBinding.hpp"

#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(const int argc, const char *const argv[]) {
  if (argc != 4 && argc != 5) {
    std::cerr << "usage: loaded-knee-binding-test MANIFEST BINDING OWNERSHIP"
                 " [NHKNEE]\n";
    return 64;
  }
  metalrobo::NumiHumanLoadedKneeBindingAdmissionV1 admission;
  std::string error;
  if (!metalrobo::loadNumiHumanLoadedKneeBindingV1(argv[1], argv[2], argv[3],
                                                   admission, error)) {
    std::cerr << error << '\n';
    return 1;
  }
  if (argc == 5) {
    std::ifstream stream(argv[4], std::ios::binary | std::ios::ate);
    if (!stream) {
      std::cerr << "could not open NHKNEE payload\n";
      return 1;
    }
    const std::streamoff end = stream.tellg();
    if (end <= 0) {
      std::cerr << "NHKNEE payload is empty\n";
      return 1;
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(end));
    stream.seekg(0, std::ios::beg);
    if (!stream.read(reinterpret_cast<char *>(bytes.data()), end)) {
      std::cerr << "could not read NHKNEE payload\n";
      return 1;
    }
    metalrobo::NumiHumanKneePayload payload;
    const auto diagnostics = metalrobo::decodeNumiHumanKneePayload(
        bytes, payload);
    if (!diagnostics.succeeded() ||
        !metalrobo::validateNumiHumanLoadedKneeAuthoringV1(
            admission.authoring, &payload, error)) {
      std::cerr << (error.empty() ? "NHKNEE payload decode failed" : error)
                << '\n';
      return 1;
    }
  }
  std::cout << "loaded_knee_binding_admission=passed"
            << " source_ownership_status="
            << admission.authoring.sourceOwnershipStatus << " coverage_leaves="
            << admission.authoring.coverageLeafSHA256s.size()
            << " production_qualified="
            << (admission.authoring.productionQualified ? "true" : "false")
            << " decoded_payload_source_order="
            << (argc == 5 ? "verified" : "not_requested")
            << '\n';
  return 0;
}
