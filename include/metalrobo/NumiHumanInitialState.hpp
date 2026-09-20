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

// Exact identity of the immutable support-contact payload that supplied the
// prepared rows. sourceRecordCount is the on-disk NHCNT record count;
// expandedRowCount is the Matter row count after NHCNT2 capsule expansion.
struct NumiHumanSupportPayloadIdentity {
    std::array<std::uint8_t, 32u> sha256{};
    std::uint64_t byteCount = 0u;
    std::uint32_t payloadABI = 0u;
    std::uint32_t sourceRecordCount = 0u;
    std::uint32_t expandedRowCount = 0u;

    [[nodiscard]] bool operator==(
        const NumiHumanSupportPayloadIdentity&) const = default;
};

// Matter's accepted support-history representation. xyz is a world-space
// tangent impulse and w is the scalar impulse along the bound ground normal.
// Units are N*s. Geometry-aware admission (tangency and the Coulomb cone) is
// performed by the owner that has the decoded NHCNT rows and ground plane.
struct NumiHumanSupportHistoryRecord {
    float tangentImpulseWorldX = 0.0f;
    float tangentImpulseWorldY = 0.0f;
    float tangentImpulseWorldZ = 0.0f;
    float normalImpulse = 0.0f;

    [[nodiscard]] bool operator==(
        const NumiHumanSupportHistoryRecord&) const = default;
};

struct NumiHumanPreparedSupportHistory {
    NumiHumanSupportPayloadIdentity support;
    std::vector<NumiHumanSupportHistoryRecord> rows;

    [[nodiscard]] bool operator==(
        const NumiHumanPreparedSupportHistory&) const = default;
};

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
    // Presence selects NHINIT3. Row order is the bound payload's canonical
    // expanded order: NHCNT1 file order; NHCNT2 primitive order with capsule
    // endpoint A followed by endpoint B.
    std::optional<NumiHumanPreparedSupportHistory> preparedSupportHistory;
    std::array<std::uint8_t, 32u> sourceArchiveSHA256{};
    std::vector<float> q;
    std::vector<float> v;
    std::vector<MRMujocoMuscleStateGPU> muscles;
};

// Preserve an offline FP64 placement in the existing authoritative FP32 root
// expansion. The public q projection remains unchanged. Reject nonfinite,
// overflowing or underflowing inputs that cannot round-trip; output is unchanged
// on failure. This is construction only, never a live-state reset operation.
[[nodiscard]] bool makeNumiHumanInitialRootTranslation(
    const std::array<double, 3u>& position,
    MRCompensatedRootTranslationGPU& output, std::string& error);

// Exact effective clock; zero denotes missing, inconsistent or out-of-domain
// identity. The domain remains the legacy (0,1 second] interval.
[[nodiscard]] std::uint64_t numiHumanInitialStateTimestepNanoseconds(
    const NumiHumanInitialState& state) noexcept;

// Canonical little-endian: NHINIT1 has the unchanged 96-byte header. A nonzero
// nanosecond clock or rootTranslation selects NHINIT2 and its 160-byte header.
// Prepared support history selects NHINIT3 and its 224-byte header. V3 retains
// every v2 offset; flags@32 bit1 is required and identifies the trailing
// expanded-row float4 records. The exact raw NHCNT SHA-256 is @160, byte count
// @192, ABI/source/expanded counts @200/@204/@208, record bytes 16 @212,
// encoding 1 (Matter world-tangent Coulomb impulse) @216 and reserved zero
// @220. q/v/muscles begin at the selected header boundary; v3 histories follow
// them. NHINIT1/2 encoding and bytes are unchanged.
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
// Bound overload required for NHINIT3. The legacy overload deliberately
// rejects NHINIT3 so prepared impulses can never be admitted without the exact
// raw NHCNT identity. Both overloads preserve output on failure.
[[nodiscard]] bool decodeNumiHumanInitialState(
    std::span<const std::byte> bytes, std::uint32_t nq, std::uint32_t nv,
    std::uint32_t muscleCount, const std::array<std::uint8_t, 32u>& sourceArchiveSHA256,
    const NumiHumanSupportPayloadIdentity& expectedSupport,
    NumiHumanInitialState& output, std::string& error);

} // namespace metalrobo
