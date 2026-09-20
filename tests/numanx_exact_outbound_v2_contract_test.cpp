#include "metalrobo/NumanXExactTransaction.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "numanx exact outbound-v2 contract: %s\n", message);
    std::exit(1);
}

void require(const bool condition, const char* message) {
    if (!condition) fail(message);
}

mrnx_metal_range_v1 makeRange(
    void* buffer,
    const std::uint64_t address,
    const std::uint64_t byteCount,
    const std::uint32_t elementType,
    const std::uint32_t elementBytes
) {
    mrnx_metal_range_v1 result{};
    result.abi_version = MRNX_BRIDGE_ABI_V1;
    result.struct_size = sizeof(result);
    result.metal_buffer = buffer;
    result.gpu_address = address;
    result.byte_count = byteCount;
    result.element_type = elementType;
    result.element_byte_count = elementBytes;
    return result;
}

} // namespace

int main() {
    // These values are updated only with an intentional v2 hash-domain ABI
    // change and are shared review evidence for CPU and future Metal owners.
    constexpr std::uint64_t inboundAuthorityGolden = 0x67f243f667d0323dull;
    constexpr std::uint64_t physicsStateGolden = 0x18c6c3b27fbfde9bull;
    constexpr std::uint64_t proofGolden = 0xb9311ace7f609680ull;
    constexpr std::uint64_t tokenGolden = 0x49daecb782375620ull;
    constexpr std::uint64_t timingGolden = 0x5cf5d8e731b234a0ull;
    constexpr std::uint64_t channelGolden = 0x29c4e62cfb2378e8ull;
    constexpr std::uint64_t channelSetGolden = 0xc5e815775b80108full;
    constexpr std::uint64_t packetGolden = 0xa2a401361deb9f57ull;
    constexpr std::uint64_t publicationGolden = 0x7badb8d7c8eea93eull;

    static_assert(MR_NUMANX_FINGERPRINT_DOMAIN_PHYSICS_STATE_V2 ==
                  0x4e585053u);
    static_assert(MR_NUMANX_FINGERPRINT_DOMAIN_ACCEPTED_STATE_PROOF_V2 ==
                  0x4e584150u);
    static_assert(MR_NUMANX_FINGERPRINT_DOMAIN_ACCEPTED_PHYSICS_TOKEN_V2 ==
                  0x4e584154u);
    static_assert(MRNX_FINGERPRINT_DOMAIN_EXACT_INBOUND_AUTHORITY_V2 ==
                  0x4e584941u);
    static_assert(MRNX_FINGERPRINT_DOMAIN_EXACT_SENSOR_CHANNEL_V2 ==
                  0x4e584348u);
    static_assert(MRNX_FINGERPRINT_DOMAIN_EXACT_SENSOR_CHANNEL_SET_V2 ==
                  0x4e584353u);
    static_assert(MRNX_FINGERPRINT_DOMAIN_EXACT_SENSOR_TIMING_V2 ==
                  0x4e58544du);
    static_assert(MRNX_FINGERPRINT_DOMAIN_EXACT_SENSOR_PACKET_V2 ==
                  0x4e585350u);
    static_assert(MRNX_FINGERPRINT_DOMAIN_EXACT_PUBLICATION_V2 ==
                  0x4e585050u);

    static_assert(sizeof(MRNumanXAcceptedStateProofGPU) == 128u);
    static_assert(sizeof(MRNumanXAcceptedPhysicsStateTokenGPU) == 64u);
    static_assert(sizeof(MRNumanXAcceptedStateProofGPUV2) == 160u);
    static_assert(sizeof(MRNumanXAcceptedPhysicsStateTokenGPUV2) == 64u);
    static_assert(offsetof(
        MRNumanXAcceptedStateProofGPUV2, proofFingerprint) == 152u);
    static_assert(sizeof(mrnx_candidate_channel_v1) == 128u);
    static_assert(sizeof(mrnx_candidate_timing_v1) == 40u);
    static_assert(sizeof(mrnx_publication_v1) == 24u);
    static_assert(sizeof(mrnx_aggregate_snapshot_v4) == 1944u);
    static_assert(sizeof(mrnx_candidate_channel_v2) == 144u);
    static_assert(sizeof(mrnx_candidate_timing_v2) == 56u);
    static_assert(sizeof(mrnx_exact_inbound_authority_v2) == 112u);
    static_assert(sizeof(mrnx_exact_sensor_packet_v2) == 128u);
    static_assert(sizeof(mrnx_publication_v2) == 72u);
    static_assert(sizeof(mrnx_aggregate_snapshot_v5) == 2392u);
    static_assert(offsetof(mrnx_aggregate_snapshot_v5, timing) == 248u);
    static_assert(offsetof(
        mrnx_aggregate_snapshot_v5, inbound_authority) == 304u);
    static_assert(offsetof(
        mrnx_aggregate_snapshot_v5, sensor_packet) == 416u);
    static_assert(offsetof(mrnx_aggregate_snapshot_v5, publication) == 544u);
    static_assert(offsetof(mrnx_aggregate_snapshot_v5, channels) == 616u);

    mrnx_exact_inbound_authority_v2 authority{};
    authority.abi_version = MRNX_EXACT_INBOUND_AUTHORITY_ABI_V2;
    authority.struct_size = sizeof(authority);
    authority.clock_domain = MRNX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS;
    authority.clock_quantum_nanoseconds =
        MRNX_EXACT_CLOCK_QUANTUM_NANOSECONDS;
    authority.accepted_brain_timestamp_nanoseconds = 12'500u;
    authority.brain_generation = 10u;
    authority.transaction_fingerprint = 0x986252871c867014ull;
    authority.substep_fingerprint = 0x339760742e9d5b13ull;
    authority.motor_candidate_fingerprint = 0xd545ffb84702f6ccull;
    authority.motor_output_fingerprint = 0xd2997dd67ccf83f4ull;
    authority.motor_profile_fingerprint = 0x4444u;
    authority.motor_ready_gate_fingerprint = 0x7a7c4daa7709eef0ull;
    authority.brain_program_fingerprint = 0x8888u;
    authority.fast_program_fingerprint = 0x9999u;
    authority.decision_gate_fingerprint = 0xaaaau;
    authority.inbound_authority_fingerprint =
        metalrobo::metalNumanXExactInboundAuthorityV2Fingerprint(authority);

    mrnx_brain_joint_transaction_v2 inboundRoot{};
    inboundRoot.format_version = MRNX_BRAIN_JOINT_TRANSACTION_VERSION_V2;
    inboundRoot.clock_domain = authority.clock_domain;
    inboundRoot.clock_quantum_nanoseconds =
        authority.clock_quantum_nanoseconds;
    inboundRoot.transaction_fingerprint = authority.transaction_fingerprint;

    mrnx_brain_joint_substep_v2 inboundSubstep{};
    inboundSubstep.transaction_fingerprint = authority.transaction_fingerprint;
    inboundSubstep.clock_domain = authority.clock_domain;
    inboundSubstep.clock_quantum_nanoseconds =
        authority.clock_quantum_nanoseconds;
    inboundSubstep.substep_fingerprint = authority.substep_fingerprint;

    mrnx_brain_motor_candidate_v2 inboundCandidate{};
    inboundCandidate.format_version = MRNX_BRAIN_MOTOR_CANDIDATE_VERSION_V2;
    inboundCandidate.flags = MRNX_BRAIN_MOTOR_CANDIDATE_VALID_V1;
    inboundCandidate.transaction_fingerprint =
        authority.transaction_fingerprint;
    inboundCandidate.substep_fingerprint = authority.substep_fingerprint;
    inboundCandidate.accepted_brain_timestamp_nanoseconds =
        authority.accepted_brain_timestamp_nanoseconds;
    inboundCandidate.brain_generation = authority.brain_generation;
    inboundCandidate.motor_profile_fingerprint =
        authority.motor_profile_fingerprint;
    inboundCandidate.clock_domain = authority.clock_domain;
    inboundCandidate.candidate_fingerprint =
        authority.motor_candidate_fingerprint;

    mrnx_brain_motor_output_header_v2 inboundOutput{};
    inboundOutput.format_version = MRNX_BRAIN_MOTOR_OUTPUT_VERSION_V2;
    inboundOutput.flags = MRNX_BRAIN_MOTOR_OUTPUT_VALID_V2;
    inboundOutput.timestamp_nanoseconds =
        inboundCandidate.accepted_brain_timestamp_nanoseconds;
    inboundOutput.brain_generation = inboundCandidate.brain_generation;
    inboundOutput.profile_fingerprint = authority.motor_profile_fingerprint;
    inboundOutput.clock_domain = authority.clock_domain;
    inboundOutput.output_fingerprint = authority.motor_output_fingerprint;

    mrnx_brain_motor_ready_gate_v2 inboundGate{};
    inboundGate.abi_version = MRNX_BRAIN_MOTOR_READY_ABI_VERSION_V2;
    inboundGate.struct_bytes = sizeof(inboundGate);
    inboundGate.status = MRNX_BRAIN_MOTOR_READY_GATE_SUCCESS_V2;
    inboundGate.transaction_fingerprint = authority.transaction_fingerprint;
    inboundGate.substep_fingerprint = authority.substep_fingerprint;
    inboundGate.candidate_fingerprint =
        authority.motor_candidate_fingerprint;
    inboundGate.motor_output_fingerprint =
        authority.motor_output_fingerprint;
    inboundGate.motor_profile_fingerprint =
        authority.motor_profile_fingerprint;
    inboundGate.brain_generation = inboundCandidate.brain_generation;
    inboundGate.accepted_brain_timestamp_nanoseconds =
        inboundCandidate.accepted_brain_timestamp_nanoseconds;
    inboundGate.brain_program_fingerprint =
        authority.brain_program_fingerprint;
    inboundGate.fast_program_fingerprint = authority.fast_program_fingerprint;
    inboundGate.decision_gate_fingerprint =
        authority.decision_gate_fingerprint;
    inboundGate.clock_domain = authority.clock_domain;
    inboundGate.clock_quantum_nanoseconds =
        authority.clock_quantum_nanoseconds;
    inboundGate.gate_fingerprint = authority.motor_ready_gate_fingerprint;

    MRNumanXAcceptedStateProofGPUV2 proof{};
    proof.abiVersion = MR_NUMANX_HUMAN_MATTER_EXACT_ADAPTER_ABI_VERSION;
    proof.structSize = sizeof(proof);
    proof.status = MR_NUMANX_ACCEPTED_STATE_PROOF_VALID;
    proof.environment = 7u;
    proof.transactionFingerprint = authority.transaction_fingerprint;
    proof.substepFingerprint = authority.substep_fingerprint;
    proof.acceptedTimestampNanoseconds = 25'000u;
    proof.physicsGeneration = 101u;
    proof.clockDomain = MR_NUMANX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS;
    proof.clockQuantumNanoseconds =
        MR_NUMANX_EXACT_CLOCK_QUANTUM_NANOSECONDS;
    proof.humanStateFingerprint = 0x1111111111111111ull;
    proof.matterStateFingerprint = 0x2222222222222222ull;
    proof.matterSourcePhysicsFingerprint = 0x3333333333333333ull;
    proof.matterDeviceProgramFingerprint = 0x4444444444444444ull;
    proof.stateProofProgramFingerprint = 0x5555555555555555ull;
    proof.adapterProgramFingerprint = 0x6666666666666666ull;
    proof.transactionPolicyFingerprint = 0x7777777777777777ull;
    proof.linearizationEpoch = 13u;
    proof.slotGeneration = 17u;
    proof.motorCandidateFingerprint =
        authority.motor_candidate_fingerprint;
    proof.inboundAuthorityFingerprint =
        authority.inbound_authority_fingerprint;
    proof.physicsStateFingerprint =
        metalrobo::metalNumanXExactPhysicsStateV2Fingerprint(proof);
    proof.proofFingerprint =
        metalrobo::metalNumanXExactAcceptedStateProofV2Fingerprint(proof);

    MRNumanXAcceptedPhysicsStateTokenGPUV2 token{};
    token.transactionFingerprint = proof.transactionFingerprint;
    token.substepFingerprint = proof.substepFingerprint;
    token.physicsStateFingerprint = proof.physicsStateFingerprint;
    token.acceptedTimestampNanoseconds = proof.acceptedTimestampNanoseconds;
    token.physicsGeneration = proof.physicsGeneration;
    token.environmentIdentifier = proof.environment;
    token.clockDomain = proof.clockDomain;
    token.clockQuantumNanoseconds = proof.clockQuantumNanoseconds;
    token.tokenFingerprint =
        metalrobo::metalNumanXExactAcceptedPhysicsTokenV2Fingerprint(token);

    mrnx_candidate_timing_v2 timing{};
    timing.abi_version = MRNX_CANDIDATE_TIMING_ABI_V2;
    timing.struct_size = sizeof(timing);
    timing.capture_timestamp_nanoseconds = 12'500u;
    timing.delivery_timestamp_nanoseconds = 25'000u;
    timing.latency_nanoseconds = 12'500u;
    timing.sample_interval_nanoseconds = 12'500u;
    timing.clock_domain = MRNX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS;
    timing.clock_quantum_nanoseconds = MRNX_EXACT_CLOCK_QUANTUM_NANOSECONDS;
    timing.timing_fingerprint =
        metalrobo::metalNumanXExactCandidateTimingV2Fingerprint(timing);

    auto* const valuesBuffer = reinterpret_cast<void*>(
        static_cast<std::uintptr_t>(0x50000000u));
    auto* const validityBuffer = reinterpret_cast<void*>(
        static_cast<std::uintptr_t>(0x60000000u));
    mrnx_candidate_channel_v2 channel{};
    channel.abi_version = MRNX_CANDIDATE_CHANNEL_ABI_V2;
    channel.struct_size = sizeof(channel);
    channel.modality = MRNX_CANDIDATE_MODALITY_PROPRIOCEPTION_V1;
    channel.flags = MRNX_CANDIDATE_CHANNEL_HAS_VALIDITY_V1;
    channel.receptor_timestamp_nanoseconds =
        timing.capture_timestamp_nanoseconds;
    channel.clock_domain = timing.clock_domain;
    channel.clock_quantum_nanoseconds = timing.clock_quantum_nanoseconds;
    channel.receptor_count = 3u;
    channel.feature_dimension = 10u;
    channel.values = makeRange(
        valuesBuffer, 0x1000u, 120u, MRNX_ELEMENT_FLOAT32_V1,
        sizeof(float));
    channel.validity = makeRange(
        validityBuffer, 0x2000u, 12u, MRNX_ELEMENT_UINT32_V1,
        sizeof(std::uint32_t));
    channel.channel_fingerprint =
        metalrobo::metalNumanXExactCandidateChannelV2Fingerprint(channel);

    auto* const interoceptionBuffer = reinterpret_cast<void*>(
        static_cast<std::uintptr_t>(0x70000000u));
    auto* const interoceptionValidityBuffer = reinterpret_cast<void*>(
        static_cast<std::uintptr_t>(0x80000000u));
    auto interoception = channel;
    interoception.modality = MRNX_CANDIDATE_MODALITY_INTEROCEPTION_V1;
    interoception.feature_dimension = 1u;
    interoception.values = makeRange(
        interoceptionBuffer, 0x3000u, 12u, MRNX_ELEMENT_FLOAT32_V1,
        sizeof(float));
    interoception.validity = makeRange(
        interoceptionValidityBuffer, 0x4000u, 12u,
        MRNX_ELEMENT_UINT32_V1, sizeof(std::uint32_t));
    interoception.channel_fingerprint =
        metalrobo::metalNumanXExactCandidateChannelV2Fingerprint(
            interoception);
    const std::array<mrnx_candidate_channel_v2, 2u> channels{
        channel, interoception};

    mrnx_exact_sensor_packet_v2 packet{};
    packet.abi_version = MRNX_EXACT_SENSOR_PACKET_ABI_V2;
    packet.struct_size = sizeof(packet);
    packet.clock_domain = timing.clock_domain;
    packet.clock_quantum_nanoseconds = timing.clock_quantum_nanoseconds;
    packet.channel_count = channels.size();
    packet.channel_capacity = MRNX_MAX_SENSOR_CHANNELS_V2;
    packet.transaction_fingerprint = proof.transactionFingerprint;
    packet.substep_fingerprint = proof.substepFingerprint;
    packet.accepted_physics_token_fingerprint = token.tokenFingerprint;
    packet.inbound_authority_fingerprint =
        authority.inbound_authority_fingerprint;
    packet.human_io_program_fingerprint = 0xddddddddddddddddull;
    packet.sensor_fingerprint = 0x9999999999999999ull;
    packet.transaction_instance_fingerprint = 0xaaaaaaaaaaaaaaaaull;
    packet.sensor_generation = 19u;
    packet.accepted_brain_generation = authority.brain_generation;
    packet.device_registry_id = 0xbbbbbbbbbbbbbbbbull;
    packet.timing_fingerprint = timing.timing_fingerprint;
    packet.channel_set_fingerprint =
        metalrobo::metalNumanXExactCandidateChannelSetV2Fingerprint(
            channels.data(), channels.size());
    packet.candidate_publication_fingerprint =
        metalrobo::metalNumanXExactSensorPacketV2Fingerprint(packet);

    mrnx_publication_v2 publication{};
    publication.abi_version = MRNX_PUBLICATION_ABI_V2;
    publication.struct_size = sizeof(publication);
    publication.clock_domain = token.clockDomain;
    publication.clock_quantum_nanoseconds = token.clockQuantumNanoseconds;
    publication.transaction_fingerprint = token.transactionFingerprint;
    publication.accepted_physics_token_fingerprint = token.tokenFingerprint;
    publication.candidate_publication_fingerprint =
        packet.candidate_publication_fingerprint;
    publication.joint_commit_fingerprint = 0xccccccccccccccccull;
    publication.brain_generation = packet.accepted_brain_generation;
    publication.committed_timestamp_nanoseconds =
        token.acceptedTimestampNanoseconds;
    publication.publication_fingerprint =
        metalrobo::metalNumanXExactPublicationV2Fingerprint(publication);

    std::printf(
        "exact_outbound_v2_hashes=%016llx,%016llx,%016llx,%016llx,"
        "%016llx,%016llx,%016llx,%016llx,%016llx\n",
        static_cast<unsigned long long>(
            authority.inbound_authority_fingerprint),
        static_cast<unsigned long long>(proof.physicsStateFingerprint),
        static_cast<unsigned long long>(proof.proofFingerprint),
        static_cast<unsigned long long>(token.tokenFingerprint),
        static_cast<unsigned long long>(timing.timing_fingerprint),
        static_cast<unsigned long long>(channel.channel_fingerprint),
        static_cast<unsigned long long>(packet.channel_set_fingerprint),
        static_cast<unsigned long long>(
            packet.candidate_publication_fingerprint),
        static_cast<unsigned long long>(publication.publication_fingerprint));

    require(authority.inbound_authority_fingerprint ==
                inboundAuthorityGolden,
            "v2 inbound-authority fingerprint changed");
    require(proof.physicsStateFingerprint == physicsStateGolden,
            "v2 physics-state fingerprint changed");
    require(proof.proofFingerprint == proofGolden,
            "v2 accepted-proof fingerprint changed");
    require(token.tokenFingerprint == tokenGolden,
            "v2 accepted-token fingerprint changed");
    require(timing.timing_fingerprint == timingGolden,
            "v2 timing fingerprint changed");
    require(channel.channel_fingerprint == channelGolden,
            "v2 channel fingerprint changed");
    require(packet.channel_set_fingerprint == channelSetGolden,
            "v2 channel-set fingerprint changed");
    require(packet.candidate_publication_fingerprint == packetGolden,
            "v2 sensor-packet fingerprint changed");
    require(publication.publication_fingerprint == publicationGolden,
            "v2 publication fingerprint changed");

    const std::array<std::uint64_t, 9u> hashes{
        authority.inbound_authority_fingerprint,
        proof.physicsStateFingerprint,
        proof.proofFingerprint,
        token.tokenFingerprint,
        timing.timing_fingerprint,
        channel.channel_fingerprint,
        packet.channel_set_fingerprint,
        packet.candidate_publication_fingerprint,
        publication.publication_fingerprint,
    };
    for (std::size_t first = 0u; first < hashes.size(); ++first) {
        require(hashes[first] != 0u, "v2 hash domain produced zero");
        for (std::size_t second = first + 1u; second < hashes.size();
             ++second) {
            require(hashes[first] != hashes[second],
                    "distinct v2 record domains collided");
        }
    }

    require(metalrobo::metalNumanXExactInboundAuthorityV2Valid(authority),
            "coherent exact inbound authority was rejected");
    require(metalrobo::metalNumanXExactInboundAuthorityV2Matches(
                authority, inboundRoot, inboundSubstep, inboundCandidate,
                inboundOutput, inboundGate),
            "canonical inbound authority did not match its validated records");

    auto zeroTimestampAuthority = authority;
    zeroTimestampAuthority.accepted_brain_timestamp_nanoseconds = 0u;
    zeroTimestampAuthority.inbound_authority_fingerprint =
        metalrobo::metalNumanXExactInboundAuthorityV2Fingerprint(
            zeroTimestampAuthority);
    auto zeroTimestampSubstep = inboundSubstep;
    zeroTimestampSubstep.start_timestamp_nanoseconds = 0u;
    auto zeroTimestampCandidate = inboundCandidate;
    zeroTimestampCandidate.accepted_brain_timestamp_nanoseconds = 0u;
    auto zeroTimestampOutput = inboundOutput;
    zeroTimestampOutput.timestamp_nanoseconds = 0u;
    auto zeroTimestampGate = inboundGate;
    zeroTimestampGate.accepted_brain_timestamp_nanoseconds = 0u;
    require(metalrobo::metalNumanXExactInboundAuthorityV2Valid(
                zeroTimestampAuthority),
            "exact inbound authority rejected a valid zero start timestamp");
    require(metalrobo::metalNumanXExactInboundAuthorityV2Matches(
                zeroTimestampAuthority, inboundRoot, zeroTimestampSubstep,
                zeroTimestampCandidate, zeroTimestampOutput,
                zeroTimestampGate),
            "zero-start inbound authority did not match its validated records");

    require(metalrobo::metalNumanXExactAcceptedStateProofV2Valid(proof),
            "coherent exact accepted-state proof was rejected");
    require(metalrobo::metalNumanXExactAcceptedPhysicsTokenV2Valid(
                proof, token),
            "coherent exact accepted-physics token was rejected");
    require(metalrobo::metalNumanXExactCandidateTimingV2Valid(timing),
            "coherent exact sensor timing was rejected");
    require(metalrobo::metalNumanXExactCandidateChannelV2Valid(
                channel, timing),
            "coherent exact sensor channel was rejected");
    require(metalrobo::metalNumanXExactCandidateChannelV2Valid(
                interoception, timing),
            "coherent exact interoception channel was rejected");
    require(metalrobo::metalNumanXExactSensorPacketV2Valid(
                authority, proof, token, timing, channels.data(),
                channels.size(), packet),
            "coherent exact sensor packet was rejected");
    require(metalrobo::metalNumanXExactPublicationV2Valid(
                token, publication),
            "coherent exact publication was rejected");
    require(metalrobo::metalNumanXExactOutboundFamilyV2Valid(
                authority, proof, token, timing, channels.data(),
                channels.size(), packet, publication),
            "coherent exact outbound family was rejected");

    auto mixedProof = proof;
    mixedProof.clockQuantumNanoseconds = 1'000u;
    mixedProof.physicsStateFingerprint =
        metalrobo::metalNumanXExactPhysicsStateV2Fingerprint(mixedProof);
    mixedProof.proofFingerprint =
        metalrobo::metalNumanXExactAcceptedStateProofV2Fingerprint(mixedProof);
    require(!metalrobo::metalNumanXExactAcceptedStateProofV2Valid(mixedProof),
            "exact proof admitted a legacy microsecond quantum");

    auto mixedAuthority = authority;
    mixedAuthority.clock_domain = 0u;
    mixedAuthority.clock_quantum_nanoseconds = 1'000u;
    mixedAuthority.inbound_authority_fingerprint =
        metalrobo::metalNumanXExactInboundAuthorityV2Fingerprint(
            mixedAuthority);
    require(!metalrobo::metalNumanXExactInboundAuthorityV2Valid(
                mixedAuthority),
            "exact inbound authority admitted legacy-family metadata");

    auto staleCandidateAuthority = authority;
    ++staleCandidateAuthority.motor_candidate_fingerprint;
    staleCandidateAuthority.inbound_authority_fingerprint =
        metalrobo::metalNumanXExactInboundAuthorityV2Fingerprint(
            staleCandidateAuthority);
    require(metalrobo::metalNumanXExactInboundAuthorityV2Valid(
                staleCandidateAuthority),
            "self-consistent stale-candidate fixture was malformed");
    require(!metalrobo::metalNumanXExactInboundAuthorityV2Matches(
                staleCandidateAuthority, inboundRoot, inboundSubstep,
                inboundCandidate, inboundOutput, inboundGate),
            "inbound receipt matched a stale motor candidate");
    require(!metalrobo::metalNumanXExactSensorPacketV2Valid(
                staleCandidateAuthority, proof, token, timing,
                channels.data(), channels.size(), packet),
            "sensor packet admitted a stale motor candidate");

    auto staleOutputAuthority = authority;
    ++staleOutputAuthority.motor_output_fingerprint;
    staleOutputAuthority.inbound_authority_fingerprint =
        metalrobo::metalNumanXExactInboundAuthorityV2Fingerprint(
            staleOutputAuthority);
    require(!metalrobo::metalNumanXExactInboundAuthorityV2Matches(
                staleOutputAuthority, inboundRoot, inboundSubstep,
                inboundCandidate, inboundOutput, inboundGate),
            "inbound receipt matched a stale motor output");
    require(!metalrobo::metalNumanXExactSensorPacketV2Valid(
                staleOutputAuthority, proof, token, timing,
                channels.data(), channels.size(), packet),
            "sensor packet admitted a stale motor output");

    auto staleProfileAuthority = authority;
    ++staleProfileAuthority.motor_profile_fingerprint;
    staleProfileAuthority.inbound_authority_fingerprint =
        metalrobo::metalNumanXExactInboundAuthorityV2Fingerprint(
            staleProfileAuthority);
    require(!metalrobo::metalNumanXExactInboundAuthorityV2Matches(
                staleProfileAuthority, inboundRoot, inboundSubstep,
                inboundCandidate, inboundOutput, inboundGate),
            "inbound receipt matched a stale motor profile");
    require(!metalrobo::metalNumanXExactSensorPacketV2Valid(
                staleProfileAuthority, proof, token, timing,
                channels.data(), channels.size(), packet),
            "sensor packet admitted a stale motor profile");

    auto staleGateAuthority = authority;
    ++staleGateAuthority.motor_ready_gate_fingerprint;
    ++staleGateAuthority.brain_program_fingerprint;
    staleGateAuthority.inbound_authority_fingerprint =
        metalrobo::metalNumanXExactInboundAuthorityV2Fingerprint(
            staleGateAuthority);
    require(!metalrobo::metalNumanXExactInboundAuthorityV2Matches(
                staleGateAuthority, inboundRoot, inboundSubstep,
                inboundCandidate, inboundOutput, inboundGate),
            "inbound receipt matched a stale gate or Brain program");
    require(!metalrobo::metalNumanXExactSensorPacketV2Valid(
                staleGateAuthority, proof, token, timing, channels.data(),
                channels.size(), packet),
            "sensor packet admitted a stale ready gate or Brain program");

    auto staleTemporalAuthority = authority;
    ++staleTemporalAuthority.accepted_brain_timestamp_nanoseconds;
    ++staleTemporalAuthority.brain_generation;
    staleTemporalAuthority.inbound_authority_fingerprint =
        metalrobo::metalNumanXExactInboundAuthorityV2Fingerprint(
            staleTemporalAuthority);
    require(!metalrobo::metalNumanXExactInboundAuthorityV2Matches(
                staleTemporalAuthority, inboundRoot, inboundSubstep,
                inboundCandidate, inboundOutput, inboundGate),
            "inbound receipt matched stale Brain time/generation");
    require(!metalrobo::metalNumanXExactSensorPacketV2Valid(
                staleTemporalAuthority, proof, token, timing,
                channels.data(), channels.size(), packet),
            "sensor packet admitted stale Brain time/generation");

    auto relabeledPacket = packet;
    ++relabeledPacket.accepted_brain_generation;
    relabeledPacket.candidate_publication_fingerprint =
        metalrobo::metalNumanXExactSensorPacketV2Fingerprint(
            relabeledPacket);
    require(!metalrobo::metalNumanXExactSensorPacketV2Valid(
                authority, proof, token, timing, channels.data(),
                channels.size(), relabeledPacket),
            "sensor packet relabeled the accepted Brain generation");

    auto relabeledTiming = timing;
    --relabeledTiming.capture_timestamp_nanoseconds;
    ++relabeledTiming.latency_nanoseconds;
    relabeledTiming.timing_fingerprint =
        metalrobo::metalNumanXExactCandidateTimingV2Fingerprint(
            relabeledTiming);
    auto relabeledChannels = channels;
    for (auto& relabeledChannel : relabeledChannels) {
        relabeledChannel.receptor_timestamp_nanoseconds =
            relabeledTiming.capture_timestamp_nanoseconds;
        relabeledChannel.channel_fingerprint =
            metalrobo::metalNumanXExactCandidateChannelV2Fingerprint(
                relabeledChannel);
    }
    auto relabeledTimingPacket = packet;
    relabeledTimingPacket.timing_fingerprint =
        relabeledTiming.timing_fingerprint;
    relabeledTimingPacket.channel_set_fingerprint =
        metalrobo::metalNumanXExactCandidateChannelSetV2Fingerprint(
            relabeledChannels.data(), relabeledChannels.size());
    relabeledTimingPacket.candidate_publication_fingerprint =
        metalrobo::metalNumanXExactSensorPacketV2Fingerprint(
            relabeledTimingPacket);
    require(!metalrobo::metalNumanXExactSensorPacketV2Valid(
                authority, proof, token, relabeledTiming,
                relabeledChannels.data(), relabeledChannels.size(),
                relabeledTimingPacket),
            "sensor packet relabeled the accepted Brain capture timestamp");

    auto mixedToken = token;
    mixedToken.clockDomain = 0u;
    mixedToken.tokenFingerprint =
        metalrobo::metalNumanXExactAcceptedPhysicsTokenV2Fingerprint(
            mixedToken);
    require(!metalrobo::metalNumanXExactAcceptedPhysicsTokenV2Valid(
                proof, mixedToken),
            "exact token admitted a legacy clock domain");

    auto mixedTiming = timing;
    mixedTiming.abi_version = MRNX_BRIDGE_ABI_V1;
    mixedTiming.clock_domain = 0u;
    mixedTiming.clock_quantum_nanoseconds = 1'000u;
    mixedTiming.timing_fingerprint =
        metalrobo::metalNumanXExactCandidateTimingV2Fingerprint(mixedTiming);
    require(!metalrobo::metalNumanXExactCandidateTimingV2Valid(mixedTiming),
            "exact timing admitted legacy-family metadata");

    auto mixedChannel = channel;
    mixedChannel.abi_version = MRNX_BRIDGE_ABI_V1;
    mixedChannel.channel_fingerprint =
        metalrobo::metalNumanXExactCandidateChannelV2Fingerprint(
            mixedChannel);
    require(!metalrobo::metalNumanXExactCandidateChannelV2Valid(
                mixedChannel, timing),
            "exact channel admitted the legacy channel ABI");

    auto mixedPacket = packet;
    mixedPacket.abi_version = MRNX_BRIDGE_ABI_V1;
    mixedPacket.clock_domain = 0u;
    mixedPacket.clock_quantum_nanoseconds = 1'000u;
    mixedPacket.candidate_publication_fingerprint =
        metalrobo::metalNumanXExactSensorPacketV2Fingerprint(mixedPacket);
    require(!metalrobo::metalNumanXExactSensorPacketV2Valid(
                authority, proof, token, timing, channels.data(),
                channels.size(), mixedPacket),
            "exact sensor packet admitted legacy-family metadata");

    auto mixedPublication = publication;
    mixedPublication.abi_version = MRNX_BRIDGE_ABI_V1;
    mixedPublication.clock_domain = 0u;
    mixedPublication.clock_quantum_nanoseconds = 1'000u;
    mixedPublication.publication_fingerprint =
        metalrobo::metalNumanXExactPublicationV2Fingerprint(
            mixedPublication);
    require(!metalrobo::metalNumanXExactPublicationV2Valid(
                token, mixedPublication),
            "exact publication admitted legacy-family metadata");

    auto underflowTiming = timing;
    underflowTiming.capture_timestamp_nanoseconds =
        std::numeric_limits<std::uint64_t>::max() - 4u;
    underflowTiming.delivery_timestamp_nanoseconds = 3u;
    underflowTiming.latency_nanoseconds = 8u;
    underflowTiming.timing_fingerprint =
        metalrobo::metalNumanXExactCandidateTimingV2Fingerprint(
            underflowTiming);
    require(!metalrobo::metalNumanXExactCandidateTimingV2Valid(
                underflowTiming),
            "exact timing admitted wrapping delivery arithmetic");

    auto overflowChannel = channel;
    overflowChannel.receptor_count =
        std::numeric_limits<std::uint32_t>::max();
    overflowChannel.feature_dimension =
        std::numeric_limits<std::uint32_t>::max();
    overflowChannel.channel_fingerprint =
        metalrobo::metalNumanXExactCandidateChannelV2Fingerprint(
            overflowChannel);
    require(!metalrobo::metalNumanXExactCandidateChannelV2Valid(
                overflowChannel, timing),
            "exact channel admitted overflowing tensor bytes");

    auto addressOverflowChannel = channel;
    addressOverflowChannel.values.gpu_address =
        std::numeric_limits<std::uint64_t>::max() - 64u;
    addressOverflowChannel.channel_fingerprint =
        metalrobo::metalNumanXExactCandidateChannelV2Fingerprint(
            addressOverflowChannel);
    require(!metalrobo::metalNumanXExactCandidateChannelV2Valid(
                addressOverflowChannel, timing),
            "exact channel admitted an overflowing GPU range");

    auto offsetOverflowChannel = channel;
    offsetOverflowChannel.values.byte_offset =
        std::numeric_limits<std::uint64_t>::max() - 64u;
    offsetOverflowChannel.channel_fingerprint =
        metalrobo::metalNumanXExactCandidateChannelV2Fingerprint(
            offsetOverflowChannel);
    require(!metalrobo::metalNumanXExactCandidateChannelV2Valid(
                offsetOverflowChannel, timing),
            "exact channel admitted an overflowing buffer-relative range");

    auto overlappingChannel = channel;
    overlappingChannel.validity.gpu_address = 0x1040u;
    overlappingChannel.channel_fingerprint =
        metalrobo::metalNumanXExactCandidateChannelV2Fingerprint(
            overlappingChannel);
    require(!metalrobo::metalNumanXExactCandidateChannelV2Valid(
                overlappingChannel, timing),
            "exact channel admitted overlapping independent GPU ranges");

    auto slicedChannel = channel;
    slicedChannel.validity.metal_buffer = slicedChannel.values.metal_buffer;
    slicedChannel.validity.byte_offset = slicedChannel.values.byte_count;
    slicedChannel.validity.gpu_address =
        slicedChannel.values.gpu_address + slicedChannel.values.byte_count;
    slicedChannel.channel_fingerprint =
        metalrobo::metalNumanXExactCandidateChannelV2Fingerprint(
            slicedChannel);
    require(metalrobo::metalNumanXExactCandidateChannelV2Valid(
                slicedChannel, timing),
            "exact channel rejected disjoint slices of one Metal buffer");

    auto overlappingSlice = slicedChannel;
    overlappingSlice.validity.byte_offset =
        overlappingSlice.values.byte_count - sizeof(std::uint32_t);
    overlappingSlice.validity.gpu_address =
        overlappingSlice.values.gpu_address +
        overlappingSlice.validity.byte_offset;
    overlappingSlice.channel_fingerprint =
        metalrobo::metalNumanXExactCandidateChannelV2Fingerprint(
            overlappingSlice);
    require(!metalrobo::metalNumanXExactCandidateChannelV2Valid(
                overlappingSlice, timing),
            "exact channel admitted overlapping slices of one Metal buffer");

    auto mismatchedSliceBase = slicedChannel;
    mismatchedSliceBase.validity.gpu_address += sizeof(std::uint32_t);
    mismatchedSliceBase.channel_fingerprint =
        metalrobo::metalNumanXExactCandidateChannelV2Fingerprint(
            mismatchedSliceBase);
    require(!metalrobo::metalNumanXExactCandidateChannelV2Valid(
                mismatchedSliceBase, timing),
            "exact channel admitted inconsistent same-buffer slice bases");

    auto changedChannels = channels;
    changedChannels[0].values.gpu_address += sizeof(float);
    changedChannels[0].channel_fingerprint =
        metalrobo::metalNumanXExactCandidateChannelV2Fingerprint(
            changedChannels[0]);
    require(!metalrobo::metalNumanXExactSensorPacketV2Valid(
                authority, proof, token, timing, changedChannels.data(),
                changedChannels.size(), packet),
            "sensor packet did not bind its channel descriptor identities");

    auto overlappingChannels = channels;
    overlappingChannels[1].values.gpu_address =
        overlappingChannels[0].values.gpu_address + sizeof(float);
    overlappingChannels[1].channel_fingerprint =
        metalrobo::metalNumanXExactCandidateChannelV2Fingerprint(
            overlappingChannels[1]);
    auto overlappingPacket = packet;
    overlappingPacket.channel_set_fingerprint =
        metalrobo::metalNumanXExactCandidateChannelSetV2Fingerprint(
            overlappingChannels.data(), overlappingChannels.size());
    overlappingPacket.candidate_publication_fingerprint =
        metalrobo::metalNumanXExactSensorPacketV2Fingerprint(
            overlappingPacket);
    require(!metalrobo::metalNumanXExactSensorPacketV2Valid(
                authority, proof, token, timing, overlappingChannels.data(),
                overlappingChannels.size(), overlappingPacket),
            "sensor packet admitted storage overlap between modalities");

    auto wrongTimestampPublication = publication;
    ++wrongTimestampPublication.committed_timestamp_nanoseconds;
    wrongTimestampPublication.publication_fingerprint =
        metalrobo::metalNumanXExactPublicationV2Fingerprint(
            wrongTimestampPublication);
    require(!metalrobo::metalNumanXExactPublicationV2Valid(
                token, wrongTimestampPublication),
            "publication admitted a timestamp outside its accepted token");

    auto flaggedToken = token;
    flaggedToken.flags = 1u;
    flaggedToken.tokenFingerprint =
        metalrobo::metalNumanXExactAcceptedPhysicsTokenV2Fingerprint(
            flaggedToken);
    auto flaggedPublication = publication;
    flaggedPublication.accepted_physics_token_fingerprint =
        flaggedToken.tokenFingerprint;
    flaggedPublication.publication_fingerprint =
        metalrobo::metalNumanXExactPublicationV2Fingerprint(
            flaggedPublication);
    require(!metalrobo::metalNumanXExactPublicationV2Valid(
                flaggedToken, flaggedPublication),
            "publication admitted a noncanonical accepted-token flag");

    auto lateTiming = timing;
    ++lateTiming.delivery_timestamp_nanoseconds;
    ++lateTiming.latency_nanoseconds;
    lateTiming.timing_fingerprint =
        metalrobo::metalNumanXExactCandidateTimingV2Fingerprint(lateTiming);
    auto latePacket = packet;
    latePacket.timing_fingerprint = lateTiming.timing_fingerprint;
    latePacket.candidate_publication_fingerprint =
        metalrobo::metalNumanXExactSensorPacketV2Fingerprint(latePacket);
    require(!metalrobo::metalNumanXExactOutboundFamilyV2Valid(
                authority, proof, token, lateTiming, channels.data(),
                channels.size(), latePacket, publication),
            "outbound family admitted sensor delivery after acceptance");

    const std::array<mrnx_candidate_channel_v2, 2u> reversedChannels{
        interoception, channel};
    auto reversedPacket = packet;
    reversedPacket.channel_set_fingerprint =
        metalrobo::metalNumanXExactCandidateChannelSetV2Fingerprint(
            reversedChannels.data(), reversedChannels.size());
    reversedPacket.candidate_publication_fingerprint =
        metalrobo::metalNumanXExactSensorPacketV2Fingerprint(reversedPacket);
    require(!metalrobo::metalNumanXExactSensorPacketV2Valid(
                authority, proof, token, timing, reversedChannels.data(),
                reversedChannels.size(), reversedPacket),
            "sensor packet admitted non-canonical modality order");

    const std::array<mrnx_candidate_channel_v2, 2u> duplicateChannels{
        channel, channel};
    auto duplicatePacket = packet;
    duplicatePacket.channel_set_fingerprint =
        metalrobo::metalNumanXExactCandidateChannelSetV2Fingerprint(
            duplicateChannels.data(), duplicateChannels.size());
    duplicatePacket.candidate_publication_fingerprint =
        metalrobo::metalNumanXExactSensorPacketV2Fingerprint(duplicatePacket);
    require(!metalrobo::metalNumanXExactSensorPacketV2Valid(
                authority, proof, token, timing, duplicateChannels.data(),
                duplicateChannels.size(), duplicatePacket),
            "outbound family admitted a duplicate modality");

    std::puts("numanx_exact_outbound_v2_contract=pass cpu_only=true");
    return 0;
}
