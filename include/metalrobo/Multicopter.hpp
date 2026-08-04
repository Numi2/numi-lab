#pragma once

#include "metalrobo/FreeBodyDynamics.hpp"
#include "metalrobo/multicopter_types.h"

#include <array>
#include <cstdint>

namespace metalrobo {

enum class MulticopterStatus : std::uint32_t {
    success = 0u,
    invalidModel,
    invalidInput,
};

struct MulticopterStepResult {
    MulticopterStatus status = MulticopterStatus::success;
    BodyWrench wrench{};

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MulticopterStatus::success;
    }
};

// FP64 oracle for the device-resident rotor transition and wrench reduction.
// Commands are rotor-speed requests in rad/s, preserving upstream controller
// semantics instead of introducing a normalized-action convention here.
[[nodiscard]] MulticopterStepResult stepMulticopter(
    const MRMulticopterModelGPU& model,
    const std::array<MRMulticopterRotorGPU, MR_MULTICOPTER_MAX_ROTORS>& rotors,
    MRMulticopterStateGPU& state,
    const std::array<float, MR_MULTICOPTER_MAX_ROTORS>& commands,
    const MRBodyStateGPU& body,
    mr_float4 windVelocity = {}
);

} // namespace metalrobo
