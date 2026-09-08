#pragma once

#include "metalrobo/mujoco_muscle_gpu.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

// Construction-only FP32 state in the final Human ownership frames. NHINIT1
// contains q, v and source-ordered excitation/activation/fibre length/velocity.
// Source rest coordinates and physical force laws are never overwritten.
// Identity targets the composed Human source BEFORE adding initial-state
// identity, and the exact authored Matter world. This is not an equilibrium
// certificate or permission to reset a live accepted trajectory.
struct NumiHumanInitialState {
    std::uint64_t humanSourceFingerprint = 0u;
    std::uint64_t worldFingerprint = 0u;
    std::uint64_t timestepMicroseconds = 0u;
    std::array<std::uint8_t, 32u> sourceArchiveSHA256{};
    std::vector<float> q;
    std::vector<float> v;
    std::vector<MRMujocoMuscleStateGPU> muscles;
};

// Canonical little-endian, 96-byte header; exact byte count, no extensions or
// implicit initialization. All values finite; excitation/activation in [0,1],
// fibre length strictly positive. Output is unchanged on failure.
[[nodiscard]] bool encodeNumiHumanInitialState(
    const NumiHumanInitialState& state, std::vector<std::byte>& output, std::string& error);
[[nodiscard]] bool decodeNumiHumanInitialState(
    std::span<const std::byte> bytes, std::uint32_t nq, std::uint32_t nv,
    std::uint32_t muscleCount, const std::array<std::uint8_t, 32u>& sourceArchiveSHA256,
    NumiHumanInitialState& output, std::string& error);

} // namespace metalrobo
