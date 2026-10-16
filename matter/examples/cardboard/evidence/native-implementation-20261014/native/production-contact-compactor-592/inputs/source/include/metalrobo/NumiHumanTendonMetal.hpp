#pragma once

#include "metalrobo/NumiHumanTendon.hpp"
#include "metalrobo/mujoco_muscle_gpu.h"
#include "metalrobo/numi_human_tendon_gpu.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

struct NumiHumanTendonMetalProgram {
    std::vector<MRNumiHumanTendonBindingGPU> bindings;
    std::vector<MRNumiHumanTendonEnvelopeGPU> envelopes;
};

struct NumiHumanTendonMetalInput {
    std::size_t environmentCount = 0u;
    std::size_t dofCount = 0u;
    std::size_t bodyPoseStride = 0u;
    std::uint32_t articulationFirstBody = 0u;
    std::size_t pointJacobianStride = 0u;
    std::uint32_t bodyJacobianPointOffset = MR_INVALID_INDEX;
    std::span<const MRMujocoMuscleResultGPU> muscleResults{};
    std::span<const MRArticulatedBodyPoseGPU> bodyPoses{};
    std::span<const float> pointJacobians{};
};

struct NumiHumanTendonMetalResult {
    std::vector<MRNumiHumanTendonTransferResultGPU> transfers;
    std::vector<float> generalizedCorrections;
};

enum class NumiHumanTendonMetalStatus : std::uint32_t {
    success = 0u,
    invalidProgram,
    invalidInput,
    metallibUnavailable,
    metalUnavailable,
    pipelineFailure,
    commandFailure,
    gpuFailure,
};

struct NumiHumanTendonMetalDiagnostics {
    NumiHumanTendonMetalStatus status = NumiHumanTendonMetalStatus::success;
    bool dispatched = false;
    bool published = false;
    double elapsedMilliseconds = 0.0;
    std::string deviceName;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == NumiHumanTendonMetalStatus::success;
    }
};

struct NumiHumanTendonMetalConfig {
    std::string metallibPath;
};

[[nodiscard]] NumiHumanTendonDiagnostics makeNumiHumanTendonMetalProgram(
    const NumiHumanTendonPayload& payload,
    NumiHumanTendonMetalProgram& program
);

[[nodiscard]] NumiHumanTendonMetalDiagnostics runMetalNumiHumanTendonTransfer(
    const NumiHumanTendonMetalProgram& program,
    const NumiHumanTendonMetalInput& input,
    NumiHumanTendonMetalResult& result,
    const NumiHumanTendonMetalConfig& config = {}
);

[[nodiscard]] const char* numiHumanTendonMetalStatusName(
    NumiHumanTendonMetalStatus status
) noexcept;

} // namespace metalrobo
