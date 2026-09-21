#pragma once

#include "metalrobo/NumiHumanLoadedKnee.hpp"

#include <filesystem>
#include <string>

namespace metalrobo {

// Native, fail-closed admission result for the immutable Human authoring
// bundle.  The ownership manifest is a required companion: the Human binding
// intentionally does not duplicate that much larger artifact, so Lab must
// authenticate it before accepting source status or semantic coverage.
struct NumiHumanLoadedKneeBindingAdmissionV1 {
  NumiHumanLoadedKneeAuthoringV1 authoring;
  NumiHumanLoadedKneeDigest manifestFileSHA256{};
  NumiHumanLoadedKneeDigest bindingSHA256{};
  NumiHumanLoadedKneeDigest bindingFileSHA256{};
  NumiHumanLoadedKneeDigest ownershipFileSHA256{};
};

[[nodiscard]] bool loadNumiHumanLoadedKneeBindingV1(
    const std::filesystem::path &manifestPath,
    const std::filesystem::path &bindingPath,
    const std::filesystem::path &ownershipManifestPath,
    NumiHumanLoadedKneeBindingAdmissionV1 &output, std::string &error);

} // namespace metalrobo
