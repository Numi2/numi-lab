#pragma once

#include "metalrobo/numi_human_stand_gpu.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace numi::matter::detail {

struct HumanImplicitSpringOwnerStatusProof {
    const void* gateStatusRecords = nullptr;
    const void* leasedStatusRecords = nullptr;
    std::uint64_t gateStatusGPUAddress = 0u;
    std::uint64_t leasedStatusGPUAddress = 0u;
    std::uint32_t springCount = 0u;
    std::uint32_t recordCount = 0u;
    std::uint32_t environmentCount = 0u;
    std::uint32_t recordStrideBytes = 0u;
    std::uint32_t acceptedStatusCode = 0u;
    std::uint32_t completedCountOffsetBytes = UINT32_MAX;
    std::uint32_t expectedCompletedCount = 0u;
    std::uint32_t environmentIndexOffsetBytes = UINT32_MAX;
    std::uint32_t expectedEnvironmentBase = 0u;
    std::uint32_t controlStep = 0u;
    std::uint32_t physicsSubstep = 0u;
};

[[nodiscard]] constexpr bool matchesExactHumanImplicitSpringOwnerStatus(
    const HumanImplicitSpringOwnerStatusProof& proof) noexcept {
    return proof.gateStatusRecords != nullptr &&
        proof.leasedStatusRecords != nullptr &&
        proof.gateStatusRecords == proof.leasedStatusRecords &&
        proof.gateStatusGPUAddress != 0u &&
        proof.gateStatusGPUAddress == proof.leasedStatusGPUAddress &&
        proof.springCount != 0u &&
        proof.recordCount == 1u &&
        proof.environmentCount == 1u &&
        proof.recordStrideBytes == sizeof(MRNumiHumanStandStatusGPU) &&
        proof.acceptedStatusCode == MR_NUMI_HUMAN_STAND_SUCCESS &&
        proof.completedCountOffsetBytes ==
            offsetof(MRNumiHumanStandStatusGPU, completedSteps) &&
        proof.environmentIndexOffsetBytes ==
            offsetof(MRNumiHumanStandStatusGPU, environment) &&
        proof.expectedEnvironmentBase == 0u &&
        proof.physicsSubstep == 0u &&
        proof.controlStep != std::numeric_limits<std::uint32_t>::max() &&
        proof.expectedCompletedCount == proof.controlStep + 1u;
}

} // namespace numi::matter::detail
