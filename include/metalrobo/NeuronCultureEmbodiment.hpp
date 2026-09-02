#pragma once

#include "metalrobo/NeuronCulture.hpp"
#include "metalrobo/numanx_human_io_gpu.h"

#include <cstdint>
#include <span>

namespace metalrobo {

struct NeuronCultureStimulus {
    std::uint32_t electrode = 0u;
    float current = 0.0f;
    std::uint64_t sourceFingerprint = 0u;
};

// Converts the ten transactionally accepted NHCNT support consequences into
// one virtual-MEA stimulus. It has no publication authority: callers must pass
// only the consequences belonging to the same accepted embodied root.
[[nodiscard]] bool encodeAcceptedSupportStimulus(
    const CompiledNeuronCulture& culture,
    std::span<const MRNumanXHumanSupportConsequenceGPU> consequences,
    float physicsTimestepSeconds,
    float currentPerNewton,
    NeuronCultureStimulus& output
) noexcept;

} // namespace metalrobo
