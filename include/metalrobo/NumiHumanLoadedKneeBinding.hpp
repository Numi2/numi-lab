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

// Additive, authoring-only admission for the exact source-compliant joint-law
// pair. This is deliberately separate from the base HumanPack receipt: Human
// owns the immutable NHEQ2/NHLIM1 bytes while Matter retains sole authority
// for runtime constraint force and accepted constraint state. Nothing in this
// result is a prepared-state, runtime, physical, or production claim.
struct NumiHumanLoadedKneeSourceComplianceAdmissionV1 {
  NumiHumanLoadedKneeDigest bindingSHA256{};
  NumiHumanLoadedKneeDigest fileSHA256{};
  NumiHumanLoadedKneeDigest baseManifestSHA256{};
  NumiHumanLoadedKneeDigest baseManifestFileSHA256{};
  NumiHumanLoadedKneeDigest jointEqualityFileSHA256{};
  NumiHumanLoadedKneeDigest jointLimitFileSHA256{};
};

[[nodiscard]] bool loadNumiHumanLoadedKneeBindingV1(
    const std::filesystem::path &manifestPath,
    const std::filesystem::path &bindingPath,
    const std::filesystem::path &ownershipManifestPath,
    NumiHumanLoadedKneeBindingAdmissionV1 &output, std::string &error);

// authenticatedBase must be the successful output of
// loadNumiHumanLoadedKneeBindingV1 for the base manifest named by the
// companion. The two program paths are opened without following symlinks and
// admitted only when their complete bytes and source-compliance headers match
// the pinned NHEQ2/NHLIM1 programs.
[[nodiscard]] bool loadNumiHumanLoadedKneeSourceComplianceV1(
    const std::filesystem::path &sourceCompliancePath,
    const std::filesystem::path &jointEqualityPath,
    const std::filesystem::path &jointLimitPath,
    const NumiHumanLoadedKneeBindingAdmissionV1 &authenticatedBase,
    NumiHumanLoadedKneeSourceComplianceAdmissionV1 &output,
    std::string &error);

} // namespace metalrobo
