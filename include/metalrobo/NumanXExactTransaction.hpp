#pragma once

#include "metalrobo/mrnx_bridge_v1.h"
#include "metalrobo/numanx_human_matter_adapter_gpu.h"

#include <cstddef>
#include <cstdint>

namespace metalrobo {

// Pure identities for the exact outbound half of one NumanX physical root.
// Descriptor hashes bind both the process-local borrowed buffer identity and
// its declared GPU slice. These helpers perform no Metal work and own no
// state. They are shared by native owners and qualification so the later
// executable v3 lane cannot acquire a second hash or clock convention.

[[nodiscard]] std::uint64_t metalNumanXExactInboundAuthorityV2Fingerprint(
    const mrnx_exact_inbound_authority_v2& authority
) noexcept;

[[nodiscard]] bool metalNumanXExactInboundAuthorityV2Valid(
    const mrnx_exact_inbound_authority_v2& authority
) noexcept;

// Call after the existing v2 root/substep/candidate/output/gate validators.
// This checks that the canonical receipt was extracted from those exact
// independently validated records rather than assembled from stale identities.
[[nodiscard]] bool metalNumanXExactInboundAuthorityV2Matches(
    const mrnx_exact_inbound_authority_v2& authority,
    const mrnx_brain_joint_transaction_v2& root,
    const mrnx_brain_joint_substep_v2& substep,
    const mrnx_brain_motor_candidate_v2& candidate,
    const mrnx_brain_motor_output_header_v2& output,
    const mrnx_brain_motor_ready_gate_v2& readyGate
) noexcept;

[[nodiscard]] std::uint64_t metalNumanXExactPhysicsStateV2Fingerprint(
    const MRNumanXAcceptedStateProofGPUV2& proof
) noexcept;

[[nodiscard]] std::uint64_t metalNumanXExactAcceptedStateProofV2Fingerprint(
    const MRNumanXAcceptedStateProofGPUV2& proof
) noexcept;

[[nodiscard]] bool metalNumanXExactAcceptedStateProofV2Valid(
    const MRNumanXAcceptedStateProofGPUV2& proof
) noexcept;

[[nodiscard]] std::uint64_t
metalNumanXExactAcceptedPhysicsTokenV2Fingerprint(
    const MRNumanXAcceptedPhysicsStateTokenGPUV2& token
) noexcept;

[[nodiscard]] bool metalNumanXExactAcceptedPhysicsTokenV2Valid(
    const MRNumanXAcceptedStateProofGPUV2& proof,
    const MRNumanXAcceptedPhysicsStateTokenGPUV2& token
) noexcept;

[[nodiscard]] std::uint64_t metalNumanXExactCandidateTimingV2Fingerprint(
    const mrnx_candidate_timing_v2& timing
) noexcept;

[[nodiscard]] bool metalNumanXExactCandidateTimingV2Valid(
    const mrnx_candidate_timing_v2& timing
) noexcept;

[[nodiscard]] bool metalNumanXExactCandidateChannelV2Valid(
    const mrnx_candidate_channel_v2& channel,
    const mrnx_candidate_timing_v2& timing
) noexcept;

[[nodiscard]] std::uint64_t metalNumanXExactCandidateChannelV2Fingerprint(
    const mrnx_candidate_channel_v2& channel
) noexcept;

[[nodiscard]] std::uint64_t metalNumanXExactCandidateChannelSetV2Fingerprint(
    const mrnx_candidate_channel_v2* channels,
    std::size_t channelCount
) noexcept;

[[nodiscard]] std::uint64_t metalNumanXExactSensorPacketV2Fingerprint(
    const mrnx_exact_sensor_packet_v2& packet
) noexcept;

[[nodiscard]] bool metalNumanXExactSensorPacketV2Valid(
    const mrnx_exact_inbound_authority_v2& authority,
    const MRNumanXAcceptedStateProofGPUV2& proof,
    const MRNumanXAcceptedPhysicsStateTokenGPUV2& token,
    const mrnx_candidate_timing_v2& timing,
    const mrnx_candidate_channel_v2* channels,
    std::size_t channelCount,
    const mrnx_exact_sensor_packet_v2& packet
) noexcept;

[[nodiscard]] std::uint64_t metalNumanXExactPublicationV2Fingerprint(
    const mrnx_publication_v2& publication
) noexcept;

[[nodiscard]] bool metalNumanXExactPublicationV2Valid(
    const MRNumanXAcceptedPhysicsStateTokenGPUV2& token,
    const mrnx_publication_v2& publication
) noexcept;

// Complete pointer-free family check used before exact records are made public.
// It binds physical acceptance to sensor delivery and requires a unique,
// structurally valid set of exact-clock sensor channels.
[[nodiscard]] bool metalNumanXExactOutboundFamilyV2Valid(
    const mrnx_exact_inbound_authority_v2& authority,
    const MRNumanXAcceptedStateProofGPUV2& proof,
    const MRNumanXAcceptedPhysicsStateTokenGPUV2& token,
    const mrnx_candidate_timing_v2& timing,
    const mrnx_candidate_channel_v2* channels,
    std::size_t channelCount,
    const mrnx_exact_sensor_packet_v2& packet,
    const mrnx_publication_v2& publication
) noexcept;

} // namespace metalrobo
