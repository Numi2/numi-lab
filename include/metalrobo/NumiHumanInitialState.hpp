#pragma once

#include "metalrobo/mujoco_muscle_gpu.h"
#include "metalrobo/compensated_translation_gpu.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <optional>
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
    // Zero preserves NHINIT1. If both clocks are set they must agree exactly.
    std::uint64_t timestepNanoseconds = 0u;
    // Absent state derives reference=q[0:3], displacement=correction=0 at runtime.
    std::optional<MRCompensatedRootTranslationGPU> rootTranslation;
    std::array<std::uint8_t, 32u> sourceArchiveSHA256{};
    std::vector<float> q;
    std::vector<float> v;
    std::vector<MRMujocoMuscleStateGPU> muscles;
};

// Exact effective clock; zero denotes missing, inconsistent or out-of-domain
// identity. The domain remains the legacy (0,1 second] interval.
[[nodiscard]] std::uint64_t numiHumanInitialStateTimestepNanoseconds(
    const NumiHumanInitialState& state) noexcept;

// Canonical little-endian: NHINIT1 has the unchanged 96-byte header. A nonzero
// nanosecond clock or rootTranslation selects NHINIT2 and its 160-byte header.
// V2 retains v1 offsets, uses flags@32 bit0 for translation presence, ns@96,
// three explicit float4 blocks@104, and reserved zero bytes@152..159. All other
// flags/padding are zero; absent translation has 48 zero bytes. V2 requires
// nq>=3 and a nonzero effective ns clock. The pair must be canonical and its
// shared CPU/Metal projection must bit-match q[0:3]. V1 decode retains empty
// rootTranslation and ns=0 so re-encoding is byte-identical. All values finite;
// excitation/activation in [0,1], fibre length positive. Exact length and source
// identity are required. Output is unchanged on failure.
[[nodiscard]] bool encodeNumiHumanInitialState(
    const NumiHumanInitialState& state, std::vector<std::byte>& output, std::string& error);
[[nodiscard]] bool decodeNumiHumanInitialState(
    std::span<const std::byte> bytes, std::uint32_t nq, std::uint32_t nv,
    std::uint32_t muscleCount, const std::array<std::uint8_t, 32u>& sourceArchiveSHA256,
    NumiHumanInitialState& output, std::string& error);

} // namespace metalrobo
