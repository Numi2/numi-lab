#pragma once
#include "numi/matter/matter.hpp"

namespace numi::matter {
// Apple offline lowering of the source-bound HumanPack physiology payload.
// Validates the supported constitutive schema and preserves its exact SHA256
// plus upstream graph identities. Anatomical source acquisition and calibration
// remain HumanPack authoring responsibilities. No simulation executes here.
[[nodiscard]] bool readHumanPhysiologyNetwork(
    const std::filesystem::path& path,
    VascularNetworkSource& output,
    std::string* error = nullptr);
}
