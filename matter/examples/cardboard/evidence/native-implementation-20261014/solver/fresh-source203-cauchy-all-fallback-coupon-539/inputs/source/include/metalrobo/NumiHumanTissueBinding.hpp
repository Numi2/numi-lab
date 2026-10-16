#pragma once
#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/NumiHumanCartilage.hpp"
#include "metalrobo/NumiHumanTissueMass.hpp"
#include "numi/matter/matter.hpp"

namespace metalrobo {
struct NumiHumanCostalRegionBinding {
    std::uint32_t donorBody = MR_INVALID_INDEX;
    std::uint32_t sternalBody = MR_INVALID_INDEX;
    std::uint32_t ribBody = MR_INVALID_INDEX;
};
struct NumiHumanCostalBinding {
    std::uint32_t bodyCount = 0u;
    std::uint32_t nodeCount = 0u;
    std::uint32_t tetrahedronCount = 0u;
    std::array<double,16> atlasToWorld{};
    std::array<std::uint8_t,32> registrationSHA256{};
    std::vector<NumiHumanCostalRegionBinding> regions;
};
// The caller must hash the actual payload bytes, not trust manifest hashes.
[[nodiscard]] bool decodeNumiHumanCostalBinding(
    std::span<const std::byte> bytes,
    std::span<const std::uint8_t,32> cartilageSHA256,
    std::span<const std::uint8_t,32> rigidSHA256,
    NumiHumanCostalBinding& result, std::string& error);

struct NumiHumanCostalCompilation {
    numi::matter::CompileResult matter;
    NumiHumanTissueMassResult mass;
    EngineModel rebasedHuman;
    double maximumAttachmentRestErrorM = 0.0;
    std::string error;
    [[nodiscard]] bool succeeded() const noexcept { return error.empty(); }
};
// Cooks the final Matter attachment points in the residual COM frames and
// returns matching canonical rigid mechanics. External owning packs must
// consume the same offsets before runtime admission.
[[nodiscard]] NumiHumanCostalCompilation compileNumiHumanCostalTissue(
    const NumiHumanCostalCartilagePayload& cartilage,
    const NumiHumanCostalBinding& binding, const EngineModel& human,
    const numi::matter::MaterialProgram& material, double timestepSeconds);
} // namespace metalrobo
