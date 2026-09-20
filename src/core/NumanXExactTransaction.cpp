#include "metalrobo/NumanXExactTransaction.hpp"

#include <cstddef>
#include <limits>

namespace metalrobo {
namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

void mixU32(std::uint64_t& hash, const std::uint32_t value) noexcept {
    for (std::uint32_t index = 0u; index < 4u; ++index) {
        hash ^= (value >> (index * 8u)) & 0xffu;
        hash *= kFnvPrime;
    }
}

void mixU64(std::uint64_t& hash, const std::uint64_t value) noexcept {
    for (std::uint32_t index = 0u; index < 8u; ++index) {
        hash ^= (value >> (index * 8u)) & 0xffu;
        hash *= kFnvPrime;
    }
}

[[nodiscard]] bool checkedMultiply(
    const std::uint64_t first,
    const std::uint64_t second,
    std::uint64_t& output
) noexcept {
    if (first != 0u &&
        second > std::numeric_limits<std::uint64_t>::max() / first) {
        return false;
    }
    output = first * second;
    return true;
}

[[nodiscard]] bool checkedSum(
    const std::uint64_t first,
    const std::uint64_t count,
    std::uint64_t& output
) noexcept {
    if (first > std::numeric_limits<std::uint64_t>::max() - count) {
        return false;
    }
    output = first + count;
    return true;
}

[[nodiscard]] bool validRange(
    const mrnx_metal_range_v1& range,
    const mrnx_element_type_v1 type,
    const std::uint32_t elementBytes,
    const std::uint64_t expectedBytes
) noexcept {
    std::uint64_t offsetEnd = 0u;
    std::uint64_t addressEnd = 0u;
    return range.abi_version == MRNX_BRIDGE_ABI_V1 &&
        range.struct_size == sizeof(range) && range.metal_buffer != nullptr &&
        range.gpu_address != 0u && expectedBytes != 0u &&
        range.byte_count == expectedBytes &&
        range.element_type == type &&
        range.element_byte_count == elementBytes &&
        range.byte_offset % elementBytes == 0u &&
        range.gpu_address % elementBytes == 0u &&
        range.gpu_address >= range.byte_offset &&
        checkedSum(range.byte_offset, range.byte_count, offsetEnd) &&
        checkedSum(range.gpu_address, range.byte_count, addressEnd);
}

void mixRangeIdentity(
    std::uint64_t& hash,
    const mrnx_metal_range_v1& range
) noexcept {
    // gpu_address is the slice start: native publication additionally proves
    // it equals the borrowed buffer's base address plus byte_offset.
    mixU32(hash, range.abi_version);
    mixU32(hash, range.struct_size);
    mixU64(
        hash,
        static_cast<std::uint64_t>(
            reinterpret_cast<std::uintptr_t>(range.metal_buffer)));
    mixU64(hash, range.gpu_address);
    mixU64(hash, range.byte_offset);
    mixU64(hash, range.byte_count);
    mixU32(hash, range.element_type);
    mixU32(hash, range.element_byte_count);
}

[[nodiscard]] bool disjointRanges(
    const mrnx_metal_range_v1& first,
    const mrnx_metal_range_v1& second
) noexcept {
    std::uint64_t firstEnd = 0u;
    std::uint64_t secondEnd = 0u;
    if (first.metal_buffer == second.metal_buffer) {
        return first.gpu_address >= first.byte_offset &&
            second.gpu_address >= second.byte_offset &&
            first.gpu_address - first.byte_offset ==
                second.gpu_address - second.byte_offset &&
            checkedSum(first.byte_offset, first.byte_count, firstEnd) &&
            checkedSum(second.byte_offset, second.byte_count, secondEnd) &&
            (firstEnd <= second.byte_offset ||
             secondEnd <= first.byte_offset);
    }
    return checkedSum(first.gpu_address, first.byte_count, firstEnd) &&
        checkedSum(second.gpu_address, second.byte_count, secondEnd) &&
        (firstEnd <= second.gpu_address || secondEnd <= first.gpu_address);
}

[[nodiscard]] bool validModality(const std::uint32_t modality) noexcept {
    switch (modality) {
        case MRNX_CANDIDATE_MODALITY_VISION_V1:
        case MRNX_CANDIDATE_MODALITY_AUDITION_V1:
        case MRNX_CANDIDATE_MODALITY_TOUCH_V1:
        case MRNX_CANDIDATE_MODALITY_PROPRIOCEPTION_V1:
        case MRNX_CANDIDATE_MODALITY_VESTIBULAR_V1:
        case MRNX_CANDIDATE_MODALITY_INTEROCEPTION_V1:
        case MRNX_CANDIDATE_MODALITY_KINESTHESIA_V1:
            return true;
        default:
            return false;
    }
}

} // namespace

std::uint64_t metalNumanXExactInboundAuthorityV2Fingerprint(
    const mrnx_exact_inbound_authority_v2& authority
) noexcept {
    std::uint64_t hash = kFnvOffset;
    mixU32(hash, MRNX_FINGERPRINT_DOMAIN_EXACT_INBOUND_AUTHORITY_V2);
    mixU32(hash, authority.abi_version);
    mixU32(hash, authority.struct_size);
    mixU32(hash, authority.clock_domain);
    mixU32(hash, authority.clock_quantum_nanoseconds);
    mixU64(hash, authority.accepted_brain_timestamp_nanoseconds);
    mixU64(hash, authority.brain_generation);
    mixU64(hash, authority.transaction_fingerprint);
    mixU64(hash, authority.substep_fingerprint);
    mixU64(hash, authority.motor_candidate_fingerprint);
    mixU64(hash, authority.motor_output_fingerprint);
    mixU64(hash, authority.motor_profile_fingerprint);
    mixU64(hash, authority.motor_ready_gate_fingerprint);
    mixU64(hash, authority.brain_program_fingerprint);
    mixU64(hash, authority.fast_program_fingerprint);
    mixU64(hash, authority.decision_gate_fingerprint);
    return hash;
}

bool metalNumanXExactInboundAuthorityV2Valid(
    const mrnx_exact_inbound_authority_v2& authority
) noexcept {
    return authority.abi_version == MRNX_EXACT_INBOUND_AUTHORITY_ABI_V2 &&
        authority.struct_size == sizeof(authority) &&
        authority.clock_domain ==
            MRNX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS &&
        authority.clock_quantum_nanoseconds ==
            MRNX_EXACT_CLOCK_QUANTUM_NANOSECONDS &&
        authority.accepted_brain_timestamp_nanoseconds != 0u &&
        authority.brain_generation != 0u &&
        authority.transaction_fingerprint != 0u &&
        authority.substep_fingerprint != 0u &&
        authority.motor_candidate_fingerprint != 0u &&
        authority.motor_output_fingerprint != 0u &&
        authority.motor_profile_fingerprint != 0u &&
        authority.motor_ready_gate_fingerprint != 0u &&
        authority.brain_program_fingerprint != 0u &&
        authority.fast_program_fingerprint != 0u &&
        authority.decision_gate_fingerprint != 0u &&
        authority.inbound_authority_fingerprint != 0u &&
        authority.inbound_authority_fingerprint ==
            metalNumanXExactInboundAuthorityV2Fingerprint(authority);
}

bool metalNumanXExactInboundAuthorityV2Matches(
    const mrnx_exact_inbound_authority_v2& authority,
    const mrnx_brain_joint_transaction_v2& root,
    const mrnx_brain_joint_substep_v2& substep,
    const mrnx_brain_motor_candidate_v2& candidate,
    const mrnx_brain_motor_output_header_v2& output,
    const mrnx_brain_motor_ready_gate_v2& readyGate
) noexcept {
    return metalNumanXExactInboundAuthorityV2Valid(authority) &&
        root.format_version == MRNX_BRAIN_JOINT_TRANSACTION_VERSION_V2 &&
        root.clock_domain == MRNX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS &&
        root.clock_quantum_nanoseconds ==
            MRNX_EXACT_CLOCK_QUANTUM_NANOSECONDS &&
        root.transaction_fingerprint == authority.transaction_fingerprint &&
        substep.transaction_fingerprint == root.transaction_fingerprint &&
        substep.clock_domain == root.clock_domain &&
        substep.clock_quantum_nanoseconds == root.clock_quantum_nanoseconds &&
        substep.substep_fingerprint == authority.substep_fingerprint &&
        candidate.format_version == MRNX_BRAIN_MOTOR_CANDIDATE_VERSION_V2 &&
        (candidate.flags & MRNX_BRAIN_MOTOR_CANDIDATE_VALID_V1) != 0u &&
        candidate.transaction_fingerprint == authority.transaction_fingerprint &&
        candidate.substep_fingerprint == authority.substep_fingerprint &&
        candidate.clock_domain == root.clock_domain &&
        candidate.accepted_brain_timestamp_nanoseconds ==
            authority.accepted_brain_timestamp_nanoseconds &&
        candidate.brain_generation == authority.brain_generation &&
        candidate.candidate_fingerprint ==
            authority.motor_candidate_fingerprint &&
        candidate.motor_profile_fingerprint ==
            authority.motor_profile_fingerprint &&
        output.format_version == MRNX_BRAIN_MOTOR_OUTPUT_VERSION_V2 &&
        (output.flags & MRNX_BRAIN_MOTOR_OUTPUT_VALID_V2) != 0u &&
        output.clock_domain == root.clock_domain &&
        output.timestamp_nanoseconds ==
            authority.accepted_brain_timestamp_nanoseconds &&
        output.brain_generation == authority.brain_generation &&
        output.profile_fingerprint == authority.motor_profile_fingerprint &&
        output.output_fingerprint == authority.motor_output_fingerprint &&
        readyGate.abi_version == MRNX_BRAIN_MOTOR_READY_ABI_VERSION_V2 &&
        readyGate.struct_bytes == sizeof(readyGate) &&
        readyGate.status == MRNX_BRAIN_MOTOR_READY_GATE_SUCCESS_V2 &&
        readyGate.transaction_fingerprint ==
            authority.transaction_fingerprint &&
        readyGate.substep_fingerprint == authority.substep_fingerprint &&
        readyGate.candidate_fingerprint ==
            authority.motor_candidate_fingerprint &&
        readyGate.motor_output_fingerprint ==
            authority.motor_output_fingerprint &&
        readyGate.motor_profile_fingerprint ==
            authority.motor_profile_fingerprint &&
        readyGate.brain_generation == authority.brain_generation &&
        readyGate.accepted_brain_timestamp_nanoseconds ==
            authority.accepted_brain_timestamp_nanoseconds &&
        readyGate.brain_program_fingerprint ==
            authority.brain_program_fingerprint &&
        readyGate.fast_program_fingerprint ==
            authority.fast_program_fingerprint &&
        readyGate.decision_gate_fingerprint ==
            authority.decision_gate_fingerprint &&
        readyGate.clock_domain == root.clock_domain &&
        readyGate.clock_quantum_nanoseconds ==
            root.clock_quantum_nanoseconds &&
        readyGate.gate_fingerprint ==
            authority.motor_ready_gate_fingerprint;
}

std::uint64_t metalNumanXExactPhysicsStateV2Fingerprint(
    const MRNumanXAcceptedStateProofGPUV2& proof
) noexcept {
    std::uint64_t hash = kFnvOffset;
    mixU32(hash, MR_NUMANX_FINGERPRINT_DOMAIN_PHYSICS_STATE_V2);
    mixU32(hash, MR_NUMANX_HUMAN_MATTER_EXACT_ADAPTER_ABI_VERSION);
    mixU32(hash, proof.clockDomain);
    mixU32(hash, proof.clockQuantumNanoseconds);
    mixU64(hash, proof.humanStateFingerprint);
    mixU64(hash, proof.matterStateFingerprint);
    mixU64(hash, proof.matterSourcePhysicsFingerprint);
    mixU64(hash, proof.matterDeviceProgramFingerprint);
    mixU64(hash, proof.stateProofProgramFingerprint);
    mixU64(hash, proof.adapterProgramFingerprint);
    mixU64(hash, proof.transactionPolicyFingerprint);
    mixU64(hash, proof.motorCandidateFingerprint);
    mixU64(hash, proof.inboundAuthorityFingerprint);
    mixU64(hash, proof.transactionFingerprint);
    mixU64(hash, proof.substepFingerprint);
    mixU64(hash, proof.acceptedTimestampNanoseconds);
    mixU64(hash, proof.physicsGeneration);
    mixU32(hash, proof.environment);
    return hash;
}

std::uint64_t metalNumanXExactAcceptedStateProofV2Fingerprint(
    const MRNumanXAcceptedStateProofGPUV2& proof
) noexcept {
    std::uint64_t hash = kFnvOffset;
    mixU32(hash, MR_NUMANX_FINGERPRINT_DOMAIN_ACCEPTED_STATE_PROOF_V2);
    mixU32(hash, proof.abiVersion);
    mixU32(hash, proof.structSize);
    mixU32(hash, proof.status);
    mixU32(hash, proof.environment);
    mixU64(hash, proof.transactionFingerprint);
    mixU64(hash, proof.substepFingerprint);
    mixU64(hash, proof.acceptedTimestampNanoseconds);
    mixU64(hash, proof.physicsGeneration);
    mixU32(hash, proof.clockDomain);
    mixU32(hash, proof.clockQuantumNanoseconds);
    mixU64(hash, proof.humanStateFingerprint);
    mixU64(hash, proof.matterStateFingerprint);
    mixU64(hash, proof.physicsStateFingerprint);
    mixU64(hash, proof.matterSourcePhysicsFingerprint);
    mixU64(hash, proof.matterDeviceProgramFingerprint);
    mixU64(hash, proof.stateProofProgramFingerprint);
    mixU64(hash, proof.adapterProgramFingerprint);
    mixU64(hash, proof.transactionPolicyFingerprint);
    mixU64(hash, proof.linearizationEpoch);
    mixU64(hash, proof.slotGeneration);
    mixU64(hash, proof.motorCandidateFingerprint);
    mixU64(hash, proof.inboundAuthorityFingerprint);
    return hash;
}

bool metalNumanXExactAcceptedStateProofV2Valid(
    const MRNumanXAcceptedStateProofGPUV2& proof
) noexcept {
    return proof.abiVersion ==
            MR_NUMANX_HUMAN_MATTER_EXACT_ADAPTER_ABI_VERSION &&
        proof.structSize == sizeof(proof) &&
        proof.status == MR_NUMANX_ACCEPTED_STATE_PROOF_VALID &&
        proof.transactionFingerprint != 0u &&
        proof.substepFingerprint != 0u &&
        proof.acceptedTimestampNanoseconds != 0u &&
        proof.physicsGeneration != 0u &&
        proof.clockDomain ==
            MR_NUMANX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS &&
        proof.clockQuantumNanoseconds ==
            MR_NUMANX_EXACT_CLOCK_QUANTUM_NANOSECONDS &&
        proof.humanStateFingerprint != 0u &&
        proof.matterStateFingerprint != 0u &&
        proof.physicsStateFingerprint ==
            metalNumanXExactPhysicsStateV2Fingerprint(proof) &&
        proof.matterSourcePhysicsFingerprint != 0u &&
        proof.matterDeviceProgramFingerprint != 0u &&
        proof.stateProofProgramFingerprint != 0u &&
        proof.adapterProgramFingerprint != 0u &&
        proof.transactionPolicyFingerprint != 0u &&
        proof.linearizationEpoch != 0u && proof.slotGeneration != 0u &&
        proof.motorCandidateFingerprint != 0u &&
        proof.inboundAuthorityFingerprint != 0u &&
        proof.proofFingerprint != 0u &&
        proof.proofFingerprint ==
            metalNumanXExactAcceptedStateProofV2Fingerprint(proof);
}

std::uint64_t metalNumanXExactAcceptedPhysicsTokenV2Fingerprint(
    const MRNumanXAcceptedPhysicsStateTokenGPUV2& token
) noexcept {
    std::uint64_t hash = kFnvOffset;
    mixU32(hash, MR_NUMANX_FINGERPRINT_DOMAIN_ACCEPTED_PHYSICS_TOKEN_V2);
    mixU32(hash, MR_NUMANX_ACCEPTED_PHYSICS_TOKEN_VERSION_V2);
    mixU64(hash, token.transactionFingerprint);
    mixU64(hash, token.substepFingerprint);
    mixU64(hash, token.physicsStateFingerprint);
    mixU64(hash, token.acceptedTimestampNanoseconds);
    mixU64(hash, token.physicsGeneration);
    mixU32(hash, token.environmentIdentifier);
    mixU32(hash, token.flags);
    mixU32(hash, token.clockDomain);
    mixU32(hash, token.clockQuantumNanoseconds);
    return hash;
}

bool metalNumanXExactAcceptedPhysicsTokenV2Valid(
    const MRNumanXAcceptedStateProofGPUV2& proof,
    const MRNumanXAcceptedPhysicsStateTokenGPUV2& token
) noexcept {
    return metalNumanXExactAcceptedStateProofV2Valid(proof) &&
        token.transactionFingerprint == proof.transactionFingerprint &&
        token.substepFingerprint == proof.substepFingerprint &&
        token.physicsStateFingerprint == proof.physicsStateFingerprint &&
        token.acceptedTimestampNanoseconds ==
            proof.acceptedTimestampNanoseconds &&
        token.physicsGeneration == proof.physicsGeneration &&
        token.environmentIdentifier == proof.environment && token.flags == 0u &&
        token.clockDomain == proof.clockDomain &&
        token.clockQuantumNanoseconds == proof.clockQuantumNanoseconds &&
        token.tokenFingerprint != 0u &&
        token.tokenFingerprint ==
            metalNumanXExactAcceptedPhysicsTokenV2Fingerprint(token);
}

std::uint64_t metalNumanXExactCandidateTimingV2Fingerprint(
    const mrnx_candidate_timing_v2& timing
) noexcept {
    std::uint64_t hash = kFnvOffset;
    mixU32(hash, MRNX_FINGERPRINT_DOMAIN_EXACT_SENSOR_TIMING_V2);
    mixU32(hash, timing.abi_version);
    mixU32(hash, timing.struct_size);
    mixU64(hash, timing.capture_timestamp_nanoseconds);
    mixU64(hash, timing.delivery_timestamp_nanoseconds);
    mixU64(hash, timing.latency_nanoseconds);
    mixU64(hash, timing.sample_interval_nanoseconds);
    mixU32(hash, timing.clock_domain);
    mixU32(hash, timing.clock_quantum_nanoseconds);
    return hash;
}

bool metalNumanXExactCandidateTimingV2Valid(
    const mrnx_candidate_timing_v2& timing
) noexcept {
    return timing.abi_version == MRNX_CANDIDATE_TIMING_ABI_V2 &&
        timing.struct_size == sizeof(timing) &&
        timing.clock_domain == MRNX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS &&
        timing.clock_quantum_nanoseconds ==
            MRNX_EXACT_CLOCK_QUANTUM_NANOSECONDS &&
        timing.delivery_timestamp_nanoseconds >
            timing.capture_timestamp_nanoseconds &&
        timing.latency_nanoseconds ==
            timing.delivery_timestamp_nanoseconds -
                timing.capture_timestamp_nanoseconds &&
        timing.sample_interval_nanoseconds != 0u &&
        timing.timing_fingerprint != 0u &&
        timing.timing_fingerprint ==
            metalNumanXExactCandidateTimingV2Fingerprint(timing);
}

std::uint64_t metalNumanXExactCandidateChannelV2Fingerprint(
    const mrnx_candidate_channel_v2& channel
) noexcept {
    std::uint64_t hash = kFnvOffset;
    mixU32(hash, MRNX_FINGERPRINT_DOMAIN_EXACT_SENSOR_CHANNEL_V2);
    mixU32(hash, channel.abi_version);
    mixU32(hash, channel.struct_size);
    mixU32(hash, channel.modality);
    mixU32(hash, channel.flags);
    mixU64(hash, channel.receptor_timestamp_nanoseconds);
    mixU32(hash, channel.clock_domain);
    mixU32(hash, channel.clock_quantum_nanoseconds);
    mixU32(hash, channel.receptor_count);
    mixU32(hash, channel.feature_dimension);
    mixRangeIdentity(hash, channel.values);
    mixRangeIdentity(hash, channel.validity);
    return hash;
}

bool metalNumanXExactCandidateChannelV2Valid(
    const mrnx_candidate_channel_v2& channel,
    const mrnx_candidate_timing_v2& timing
) noexcept {
    std::uint64_t valueElements = 0u;
    std::uint64_t valueBytes = 0u;
    std::uint64_t validityBytes = 0u;
    return metalNumanXExactCandidateTimingV2Valid(timing) &&
        channel.abi_version == MRNX_CANDIDATE_CHANNEL_ABI_V2 &&
        channel.struct_size == sizeof(channel) &&
        validModality(channel.modality) &&
        channel.flags == MRNX_CANDIDATE_CHANNEL_HAS_VALIDITY_V1 &&
        channel.receptor_timestamp_nanoseconds ==
            timing.capture_timestamp_nanoseconds &&
        channel.clock_domain == timing.clock_domain &&
        channel.clock_quantum_nanoseconds ==
            timing.clock_quantum_nanoseconds &&
        channel.receptor_count != 0u && channel.feature_dimension != 0u &&
        checkedMultiply(
            channel.receptor_count, channel.feature_dimension, valueElements) &&
        checkedMultiply(valueElements, sizeof(float), valueBytes) &&
        checkedMultiply(
            channel.receptor_count, sizeof(std::uint32_t), validityBytes) &&
        validRange(
            channel.values, MRNX_ELEMENT_FLOAT32_V1, sizeof(float),
            valueBytes) &&
        validRange(
            channel.validity, MRNX_ELEMENT_UINT32_V1,
            sizeof(std::uint32_t), validityBytes) &&
        disjointRanges(channel.values, channel.validity) &&
        channel.channel_fingerprint != 0u &&
        channel.channel_fingerprint ==
            metalNumanXExactCandidateChannelV2Fingerprint(channel);
}

std::uint64_t metalNumanXExactCandidateChannelSetV2Fingerprint(
    const mrnx_candidate_channel_v2* channels,
    const std::size_t channelCount
) noexcept {
    if (channels == nullptr || channelCount == 0u ||
        channelCount > MRNX_MAX_SENSOR_CHANNELS_V2) {
        return 0u;
    }
    std::uint64_t hash = kFnvOffset;
    mixU32(hash, MRNX_FINGERPRINT_DOMAIN_EXACT_SENSOR_CHANNEL_SET_V2);
    mixU32(hash, MRNX_EXACT_SENSOR_PACKET_ABI_V2);
    mixU32(hash, static_cast<std::uint32_t>(channelCount));
    for (std::size_t index = 0u; index < channelCount; ++index) {
        mixU64(hash, channels[index].channel_fingerprint);
    }
    return hash;
}

std::uint64_t metalNumanXExactSensorPacketV2Fingerprint(
    const mrnx_exact_sensor_packet_v2& packet
) noexcept {
    std::uint64_t hash = kFnvOffset;
    mixU32(hash, MRNX_FINGERPRINT_DOMAIN_EXACT_SENSOR_PACKET_V2);
    mixU32(hash, packet.abi_version);
    mixU32(hash, packet.struct_size);
    mixU32(hash, packet.clock_domain);
    mixU32(hash, packet.clock_quantum_nanoseconds);
    mixU32(hash, packet.channel_count);
    mixU32(hash, packet.channel_capacity);
    mixU64(hash, packet.transaction_fingerprint);
    mixU64(hash, packet.substep_fingerprint);
    mixU64(hash, packet.accepted_physics_token_fingerprint);
    mixU64(hash, packet.inbound_authority_fingerprint);
    mixU64(hash, packet.human_io_program_fingerprint);
    mixU64(hash, packet.sensor_fingerprint);
    mixU64(hash, packet.transaction_instance_fingerprint);
    mixU64(hash, packet.sensor_generation);
    mixU64(hash, packet.accepted_brain_generation);
    mixU64(hash, packet.device_registry_id);
    mixU64(hash, packet.timing_fingerprint);
    mixU64(hash, packet.channel_set_fingerprint);
    return hash;
}

bool metalNumanXExactSensorPacketV2Valid(
    const mrnx_exact_inbound_authority_v2& authority,
    const MRNumanXAcceptedStateProofGPUV2& proof,
    const MRNumanXAcceptedPhysicsStateTokenGPUV2& token,
    const mrnx_candidate_timing_v2& timing,
    const mrnx_candidate_channel_v2* channels,
    const std::size_t channelCount,
    const mrnx_exact_sensor_packet_v2& packet
) noexcept {
    if (!metalNumanXExactInboundAuthorityV2Valid(authority) ||
        !metalNumanXExactAcceptedPhysicsTokenV2Valid(proof, token) ||
        !metalNumanXExactCandidateTimingV2Valid(timing) ||
        timing.delivery_timestamp_nanoseconds !=
            token.acceptedTimestampNanoseconds ||
        timing.capture_timestamp_nanoseconds !=
            authority.accepted_brain_timestamp_nanoseconds ||
        channels == nullptr || channelCount == 0u ||
        channelCount > MRNX_MAX_SENSOR_CHANNELS_V2 ||
        packet.abi_version != MRNX_EXACT_SENSOR_PACKET_ABI_V2 ||
        packet.struct_size != sizeof(packet) ||
        packet.clock_domain !=
            MRNX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS ||
        packet.clock_quantum_nanoseconds !=
            MRNX_EXACT_CLOCK_QUANTUM_NANOSECONDS ||
        packet.channel_count != channelCount ||
        packet.channel_capacity != MRNX_MAX_SENSOR_CHANNELS_V2 ||
        packet.transaction_fingerprint != proof.transactionFingerprint ||
        packet.substep_fingerprint != proof.substepFingerprint ||
        packet.accepted_physics_token_fingerprint != token.tokenFingerprint ||
        authority.transaction_fingerprint != proof.transactionFingerprint ||
        authority.substep_fingerprint != proof.substepFingerprint ||
        proof.motorCandidateFingerprint !=
            authority.motor_candidate_fingerprint ||
        proof.inboundAuthorityFingerprint !=
            authority.inbound_authority_fingerprint ||
        packet.inbound_authority_fingerprint !=
            authority.inbound_authority_fingerprint ||
        packet.human_io_program_fingerprint == 0u ||
        packet.sensor_fingerprint == 0u ||
        packet.transaction_instance_fingerprint == 0u ||
        packet.sensor_generation == 0u ||
        packet.accepted_brain_generation != authority.brain_generation ||
        packet.device_registry_id == 0u ||
        packet.timing_fingerprint != timing.timing_fingerprint) {
        return false;
    }
    std::uint32_t previousModality = 0u;
    for (std::size_t index = 0u; index < channelCount; ++index) {
        const auto& channel = channels[index];
        if (!metalNumanXExactCandidateChannelV2Valid(channel, timing) ||
            channel.modality <= previousModality) {
            return false;
        }
        for (std::size_t previous = 0u; previous < index; ++previous) {
            const auto& other = channels[previous];
            if (!disjointRanges(channel.values, other.values) ||
                !disjointRanges(channel.values, other.validity) ||
                !disjointRanges(channel.validity, other.values) ||
                !disjointRanges(channel.validity, other.validity)) {
                return false;
            }
        }
        previousModality = channel.modality;
    }
    return packet.channel_set_fingerprint != 0u &&
        packet.channel_set_fingerprint ==
            metalNumanXExactCandidateChannelSetV2Fingerprint(
                channels, channelCount) &&
        packet.candidate_publication_fingerprint != 0u &&
        packet.candidate_publication_fingerprint ==
            metalNumanXExactSensorPacketV2Fingerprint(packet);
}

std::uint64_t metalNumanXExactPublicationV2Fingerprint(
    const mrnx_publication_v2& publication
) noexcept {
    std::uint64_t hash = kFnvOffset;
    mixU32(hash, MRNX_FINGERPRINT_DOMAIN_EXACT_PUBLICATION_V2);
    mixU32(hash, publication.abi_version);
    mixU32(hash, publication.struct_size);
    mixU32(hash, publication.clock_domain);
    mixU32(hash, publication.clock_quantum_nanoseconds);
    mixU64(hash, publication.transaction_fingerprint);
    mixU64(hash, publication.accepted_physics_token_fingerprint);
    mixU64(hash, publication.candidate_publication_fingerprint);
    mixU64(hash, publication.joint_commit_fingerprint);
    mixU64(hash, publication.brain_generation);
    mixU64(hash, publication.committed_timestamp_nanoseconds);
    return hash;
}

bool metalNumanXExactPublicationV2Valid(
    const MRNumanXAcceptedPhysicsStateTokenGPUV2& token,
    const mrnx_publication_v2& publication
) noexcept {
    return token.clockDomain ==
            MR_NUMANX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS &&
        token.clockQuantumNanoseconds ==
            MR_NUMANX_EXACT_CLOCK_QUANTUM_NANOSECONDS &&
        token.transactionFingerprint != 0u &&
        token.substepFingerprint != 0u &&
        token.physicsStateFingerprint != 0u &&
        token.acceptedTimestampNanoseconds != 0u &&
        token.physicsGeneration != 0u && token.flags == 0u &&
        token.tokenFingerprint != 0u &&
        token.tokenFingerprint ==
            metalNumanXExactAcceptedPhysicsTokenV2Fingerprint(token) &&
        publication.abi_version == MRNX_PUBLICATION_ABI_V2 &&
        publication.struct_size == sizeof(publication) &&
        publication.clock_domain ==
            MRNX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS &&
        publication.clock_quantum_nanoseconds ==
            MRNX_EXACT_CLOCK_QUANTUM_NANOSECONDS &&
        publication.transaction_fingerprint ==
            token.transactionFingerprint &&
        publication.accepted_physics_token_fingerprint ==
            token.tokenFingerprint &&
        publication.accepted_physics_token_fingerprint != 0u &&
        publication.candidate_publication_fingerprint != 0u &&
        publication.joint_commit_fingerprint != 0u &&
        publication.brain_generation != 0u &&
        publication.committed_timestamp_nanoseconds ==
            token.acceptedTimestampNanoseconds &&
        publication.publication_fingerprint != 0u &&
        publication.publication_fingerprint ==
            metalNumanXExactPublicationV2Fingerprint(publication);
}

bool metalNumanXExactOutboundFamilyV2Valid(
    const mrnx_exact_inbound_authority_v2& authority,
    const MRNumanXAcceptedStateProofGPUV2& proof,
    const MRNumanXAcceptedPhysicsStateTokenGPUV2& token,
    const mrnx_candidate_timing_v2& timing,
    const mrnx_candidate_channel_v2* channels,
    const std::size_t channelCount,
    const mrnx_exact_sensor_packet_v2& packet,
    const mrnx_publication_v2& publication
) noexcept {
    if (!metalNumanXExactSensorPacketV2Valid(
            authority, proof, token, timing, channels, channelCount, packet) ||
        !metalNumanXExactPublicationV2Valid(token, publication) ||
        publication.candidate_publication_fingerprint !=
            packet.candidate_publication_fingerprint ||
        publication.brain_generation != packet.accepted_brain_generation) {
        return false;
    }
    return true;
}

} // namespace metalrobo
