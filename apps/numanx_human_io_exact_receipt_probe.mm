#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalNumanXHumanIO.hpp"
#include "metalrobo/NumanXExactTransaction.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] bool allZero(const void* bytes, const std::size_t count) {
    const auto* values = static_cast<const std::uint8_t*>(bytes);
    for (std::size_t index = 0u; index < count; ++index) {
        if (values[index] != 0u) return false;
    }
    return true;
}

template <typename To, typename From>
[[nodiscard]] To bitwiseMirror(const From& source) {
    static_assert(sizeof(To) == sizeof(From));
    To result{};
    std::memcpy(&result, &source, sizeof(result));
    return result;
}

} // namespace

int main(const int argc, const char* const argv[]) {
    @autoreleasepool {
        try {
            require(argc == 2, "usage: probe <MetalRobo.metallib>");
            static_assert(sizeof(MRNumanXExactInboundAuthorityGPUV2) ==
                          sizeof(mrnx_exact_inbound_authority_v2));
            static_assert(offsetof(
                MRNumanXExactInboundAuthorityGPUV2,
                acceptedBrainTimestampNanoseconds) ==
                offsetof(
                    mrnx_exact_inbound_authority_v2,
                    accepted_brain_timestamp_nanoseconds));
            static_assert(offsetof(
                MRNumanXExactInboundAuthorityGPUV2,
                transactionFingerprint) ==
                offsetof(
                    mrnx_exact_inbound_authority_v2,
                    transaction_fingerprint));
            static_assert(offsetof(
                MRNumanXExactInboundAuthorityGPUV2,
                motorCandidateFingerprint) ==
                offsetof(
                    mrnx_exact_inbound_authority_v2,
                    motor_candidate_fingerprint));
            static_assert(offsetof(
                MRNumanXExactInboundAuthorityGPUV2,
                motorReadyGateFingerprint) ==
                offsetof(
                    mrnx_exact_inbound_authority_v2,
                    motor_ready_gate_fingerprint));
            static_assert(offsetof(
                MRNumanXExactInboundAuthorityGPUV2,
                brainProgramFingerprint) ==
                offsetof(
                    mrnx_exact_inbound_authority_v2,
                    brain_program_fingerprint));
            static_assert(offsetof(
                MRNumanXExactInboundAuthorityGPUV2,
                decisionGateFingerprint) ==
                offsetof(
                    mrnx_exact_inbound_authority_v2,
                    decision_gate_fingerprint));
            static_assert(offsetof(
                MRNumanXExactInboundAuthorityGPUV2,
                inboundAuthorityFingerprint) ==
                offsetof(
                    mrnx_exact_inbound_authority_v2,
                    inbound_authority_fingerprint));
            static_assert(sizeof(MRNumanXBrainJointTransactionTokenV2) ==
                          sizeof(mrnx_brain_joint_transaction_v2));
            static_assert(sizeof(MRNumanXBrainJointSubstepTokenV2) ==
                          sizeof(mrnx_brain_joint_substep_v2));
            static_assert(sizeof(MRNumanXBrainMotorCandidateV2) ==
                          sizeof(mrnx_brain_motor_candidate_v2));
            static_assert(sizeof(MRNumanXBrainMotorOutputHeaderGPUV2) ==
                          sizeof(mrnx_brain_motor_output_header_v2));
            static_assert(sizeof(MRNumanXBrainMotorReadyGateGPUV2) ==
                          sizeof(mrnx_brain_motor_ready_gate_v2));

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "Metal device is unavailable");
            NSError* error = nil;
            NSString* path = [NSString stringWithUTF8String:argv[1]];
            id<MTLLibrary> library = [device
                newLibraryWithURL:[NSURL fileURLWithPath:path]
                error:&error];
            require(library != nil, "MetalRobo metallib could not be loaded");
            id<MTLFunction> function = [library
                newFunctionWithName:@"numanx_human_validate_motor_output_v2"];
            require(function != nil, "exact HumanIO validator is missing");
            id<MTLComputePipelineState> pipeline = [device
                newComputePipelineStateWithFunction:function
                error:&error];
            require(pipeline != nil, "exact HumanIO pipeline creation failed");
            id<MTLCommandQueue> queue = [device newCommandQueue];
            require(queue != nil, "Metal command queue is unavailable");

            id<MTLBuffer> headerBuffer = [device
                newBufferWithLength:sizeof(MRNumanXBrainMotorOutputHeaderGPUV2)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> excitationBuffer = [device
                newBufferWithLength:sizeof(float)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> autonomicBuffer = [device
                newBufferWithLength:MR_NUMANX_BRAIN_AUTONOMIC_COMMAND_BYTE_COUNT
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> activeSensingBuffer = [device
                newBufferWithLength:
                    MR_NUMANX_BRAIN_ACTIVE_SENSING_COMMAND_BYTE_COUNT
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> gateBuffer = [device
                newBufferWithLength:sizeof(MRNumanXBrainMotorReadyGateGPUV2)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> validationBuffer = [device
                newBufferWithLength:sizeof(std::uint32_t)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> receiptBuffer = [device
                newBufferWithLength:sizeof(MRNumanXExactInboundAuthorityGPUV2)
                options:MTLResourceStorageModePrivate];
            id<MTLBuffer> readbackBuffer = [device
                newBufferWithLength:sizeof(MRNumanXExactInboundAuthorityGPUV2)
                options:MTLResourceStorageModeShared];
            std::array<std::uint8_t,
                sizeof(MRNumanXExactInboundAuthorityGPUV2)> poisonBytes{};
            poisonBytes.fill(0xa5u);
            id<MTLBuffer> poisonBuffer = [device
                newBufferWithBytes:poisonBytes.data()
                length:poisonBytes.size()
                options:MTLResourceStorageModeShared];
            require(headerBuffer != nil && excitationBuffer != nil &&
                        autonomicBuffer != nil &&
                        activeSensingBuffer != nil && gateBuffer != nil &&
                        validationBuffer != nil && receiptBuffer != nil &&
                        readbackBuffer != nil && poisonBuffer != nil,
                    "exact HumanIO probe buffer allocation failed");
            require(receiptBuffer.storageMode == MTLStorageModePrivate,
                    "authority receipt is not device-private");
            require(receiptBuffer.gpuAddress != 0u,
                    "authority receipt has no GPU address");

            auto* excitation = static_cast<float*>(excitationBuffer.contents);
            excitation[0] = 0.5f;

            MRNumanXBrainJointTransactionTokenV2 root{};
            root.formatVersion = MR_NUMANX_BRAIN_JOINT_TRANSACTION_VERSION_V2;
            root.episodeIdentifier = 1u;
            root.controlStepIdentifier = 1u;
            root.parameterVersionFingerprint = 0x101u;
            root.committedTimestampNanoseconds = 12'500u;
            root.targetTimestampNanoseconds = 25'000u;
            root.shadowGeneration = 1u;
            root.clockDomain =
                MR_NUMANX_BRAIN_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS;
            root.clockQuantumNanoseconds =
                MR_NUMANX_BRAIN_EXACT_CLOCK_QUANTUM_NANOSECONDS;
            root.transactionFingerprint =
                metalrobo::metalNumanXBrainJointTransactionV2Fingerprint(root);

            MRNumanXBrainJointSubstepTokenV2 substep{};
            substep.transactionFingerprint = root.transactionFingerprint;
            substep.startTimestampNanoseconds =
                root.committedTimestampNanoseconds;
            substep.durationNanoseconds = 12'500u;
            substep.candidateTimestampNanoseconds =
                root.targetTimestampNanoseconds;
            substep.shadowGeneration = root.shadowGeneration;
            substep.randomCounterGeneration = root.randomCounterGeneration;
            substep.clockDomain = root.clockDomain;
            substep.clockQuantumNanoseconds = root.clockQuantumNanoseconds;
            substep.substepFingerprint =
                metalrobo::metalNumanXBrainJointSubstepV2Fingerprint(substep);

            MRNumanXBrainMotorCandidateV2 candidate{};
            candidate.formatVersion =
                MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VERSION_V2;
            candidate.flags = MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VALID |
                MR_NUMANX_BRAIN_MOTOR_CANDIDATE_DECISION_SHADOW;
            candidate.transactionFingerprint = root.transactionFingerprint;
            candidate.substepFingerprint = substep.substepFingerprint;
            candidate.acceptedBrainTimestampNanoseconds =
                substep.startTimestampNanoseconds;
            candidate.brainGeneration = root.shadowGeneration;
            candidate.motorProfileFingerprint = 0x202u;
            candidate.motorOutputHeaderGPUAddress = headerBuffer.gpuAddress;
            candidate.muscleExcitationGPUAddress =
                excitationBuffer.gpuAddress;
            candidate.randomCounterGeneration =
                root.randomCounterGeneration;
            candidate.motorOutputHeaderByteCount =
                sizeof(MRNumanXBrainMotorOutputHeaderGPUV2);
            candidate.muscleExcitationByteCount = sizeof(float);
            candidate.muscleCount = 1u;
            candidate.autonomicCommandGPUAddress =
                autonomicBuffer.gpuAddress;
            candidate.autonomicCommandByteCount =
                MR_NUMANX_BRAIN_AUTONOMIC_COMMAND_BYTE_COUNT;
            candidate.autonomicCommandCount = 1u;
            candidate.activeSensingCommandGPUAddress =
                activeSensingBuffer.gpuAddress;
            candidate.activeSensingCommandByteCount =
                MR_NUMANX_BRAIN_ACTIVE_SENSING_COMMAND_BYTE_COUNT;
            candidate.activeSensingCommandCount = 1u;
            candidate.actuatorCommandKind =
                MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION;
            candidate.clockDomain = root.clockDomain;
            candidate.speciesTemplateFingerprint = 0x303u;
            candidate.compiledSpeciesTemplateFingerprint = 0x404u;
            candidate.candidateFingerprint =
                metalrobo::metalNumanXBrainMotorCandidateV2Fingerprint(
                    candidate);

            MRNumanXBrainMotorOutputHeaderGPUV2 header{};
            header.formatVersion = MR_NUMANX_BRAIN_MOTOR_OUTPUT_VERSION_V2;
            header.flags = MR_NUMANX_BRAIN_MOTOR_OUTPUT_VALID;
            header.timestampNanoseconds =
                candidate.acceptedBrainTimestampNanoseconds;
            header.brainGeneration = candidate.brainGeneration;
            header.profileFingerprint = candidate.motorProfileFingerprint;
            header.protectiveCommandFingerprint = 0x505u;
            header.muscleCount = candidate.muscleCount;
            header.environmentIdentifier = candidate.environmentIdentifier;
            header.autonomicArousal = 0.25f;
            header.actuatorCommandKind = candidate.actuatorCommandKind;
            header.clockDomain = candidate.clockDomain;
            header.outputMaximum = 1.0f;
            header.outputFingerprint =
                metalrobo::metalNumanXBrainMotorOutputV2Fingerprint(
                    header, excitation, 1u);

            MRNumanXBrainMotorReadyGateGPUV2 gate{};
            gate.abiVersion = MR_NUMANX_BRAIN_MOTOR_READY_ABI_VERSION_V2;
            gate.structBytes = sizeof(gate);
            gate.status = MR_NUMANX_BRAIN_READY_GATE_SUCCESS;
            gate.environment = candidate.environmentIdentifier;
            gate.muscleCount = candidate.muscleCount;
            gate.actuatorCommandKind = candidate.actuatorCommandKind;
            gate.controlStep = root.controlStepIdentifier;
            gate.transactionFingerprint = root.transactionFingerprint;
            gate.substepFingerprint = substep.substepFingerprint;
            gate.candidateFingerprint = candidate.candidateFingerprint;
            gate.motorOutputFingerprint = header.outputFingerprint;
            gate.motorProfileFingerprint = candidate.motorProfileFingerprint;
            gate.brainGeneration = candidate.brainGeneration;
            gate.acceptedBrainTimestampNanoseconds =
                candidate.acceptedBrainTimestampNanoseconds;
            gate.randomCounterGeneration =
                candidate.randomCounterGeneration;
            gate.speciesTemplateFingerprint =
                candidate.speciesTemplateFingerprint;
            gate.compiledSpeciesTemplateFingerprint =
                candidate.compiledSpeciesTemplateFingerprint;
            gate.brainProgramFingerprint = 0x606u;
            gate.fastProgramFingerprint = 0x707u;
            gate.decisionGateFingerprint = 0x808u;
            gate.clockDomain = root.clockDomain;
            gate.clockQuantumNanoseconds = root.clockQuantumNanoseconds;
            gate.gateFingerprint =
                metalrobo::metalNumanXBrainMotorReadyGateV2Fingerprint(gate);

            metalrobo::MetalNumanXHumanIOInputV2 input{};
            input.root = root;
            input.substep = substep;
            input.candidate = candidate;
            input.motorOutputHeaderMetalBuffer = (__bridge void*)headerBuffer;
            input.motorOutputHeaderByteCount = sizeof(header);
            input.motorOutputHeaderEnvironmentStride = sizeof(header);
            input.expectedMotorOutputHeaderGPUAddress =
                headerBuffer.gpuAddress;
            input.excitationMetalBuffer = (__bridge void*)excitationBuffer;
            input.excitationByteCount = sizeof(float);
            input.excitationEnvironmentStride = 1u;
            input.expectedExcitationGPUAddress =
                excitationBuffer.gpuAddress;
            input.autonomicCommandMetalBuffer =
                (__bridge void*)autonomicBuffer;
            input.autonomicCommandByteCount = autonomicBuffer.length;
            input.expectedAutonomicCommandGPUAddress =
                autonomicBuffer.gpuAddress;
            input.activeSensingCommandMetalBuffer =
                (__bridge void*)activeSensingBuffer;
            input.activeSensingCommandByteCount = activeSensingBuffer.length;
            input.expectedActiveSensingCommandGPUAddress =
                activeSensingBuffer.gpuAddress;
            input.motorReadyGateMetalBuffer = (__bridge void*)gateBuffer;
            input.motorReadyGateByteCount = sizeof(gate);
            input.expectedMotorReadyGateGPUAddress = gateBuffer.gpuAddress;
            id<MTLSharedEvent> readyEvent = [device newSharedEvent];
            require(readyEvent != nil, "shared event allocation failed");
            input.motorReadySharedEvent = (__bridge void*)readyEvent;
            input.motorReadySharedEventValue = 1u;
            input.environmentCount = 1u;
            input.muscleCount = 1u;
            input.stepCount = 1u;
            input.timestepNanoseconds = substep.durationNanoseconds;
            input.receptorTimestampNanoseconds =
                substep.startTimestampNanoseconds;
            input.candidateSensorGeneration = 1u;

            MRNumanXHumanMotorDispatchGPUV2 dispatch{};
            require(metalrobo::metalNumanXHumanIOBuildMotorDispatchV2(
                        input, dispatch),
                    "exact HumanIO dispatch construction failed");

            const auto run = [&](const bool poisonReceipt) {
                std::memcpy(headerBuffer.contents, &header, sizeof(header));
                std::memcpy(gateBuffer.contents, &gate, sizeof(gate));
                *static_cast<std::uint32_t*>(validationBuffer.contents) =
                    0xffffffffu;
                std::memset(
                    readbackBuffer.contents,
                    0x5a,
                    sizeof(MRNumanXExactInboundAuthorityGPUV2));

                id<MTLCommandBuffer> command = [queue commandBuffer];
                require(command != nil, "Metal command buffer is unavailable");
                if (poisonReceipt) {
                    id<MTLBlitCommandEncoder> poison =
                        [command blitCommandEncoder];
                    require(poison != nil, "receipt poison blit unavailable");
                    [poison copyFromBuffer:poisonBuffer
                             sourceOffset:0u
                                 toBuffer:receiptBuffer
                        destinationOffset:0u
                                     size:poisonBytes.size()];
                    [poison endEncoding];
                }
                id<MTLComputeCommandEncoder> encoder =
                    [command computeCommandEncoder];
                require(encoder != nil, "exact validator encoder unavailable");
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:headerBuffer offset:0u atIndex:0u];
                [encoder setBuffer:excitationBuffer offset:0u atIndex:1u];
                [encoder setBuffer:validationBuffer offset:0u atIndex:2u];
                [encoder setBytes:&dispatch
                            length:sizeof(dispatch)
                           atIndex:3u];
                [encoder setBuffer:gateBuffer offset:0u atIndex:4u];
                [encoder setBuffer:receiptBuffer offset:0u atIndex:5u];
                [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
                [encoder endEncoding];
                id<MTLBlitCommandEncoder> readback =
                    [command blitCommandEncoder];
                require(readback != nil, "receipt readback blit unavailable");
                [readback copyFromBuffer:receiptBuffer
                            sourceOffset:0u
                                toBuffer:readbackBuffer
                       destinationOffset:0u
                                    size:sizeof(
                                        MRNumanXExactInboundAuthorityGPUV2)];
                [readback endEncoding];
                [command commit];
                [command waitUntilCompleted];
                require(command.status == MTLCommandBufferStatusCompleted &&
                            command.error == nil,
                        "exact validator command failed");
            };

            run(true);
            require(*static_cast<const std::uint32_t*>(
                        validationBuffer.contents) ==
                        MR_NUMANX_HUMAN_MOTOR_HEADER_VALID,
                    "valid exact motor authority was rejected");

            mrnx_exact_inbound_authority_v2 expected{};
            expected.abi_version = MRNX_EXACT_INBOUND_AUTHORITY_ABI_V2;
            expected.struct_size = sizeof(expected);
            expected.clock_domain = root.clockDomain;
            expected.clock_quantum_nanoseconds =
                root.clockQuantumNanoseconds;
            expected.accepted_brain_timestamp_nanoseconds =
                candidate.acceptedBrainTimestampNanoseconds;
            expected.brain_generation = candidate.brainGeneration;
            expected.transaction_fingerprint = root.transactionFingerprint;
            expected.substep_fingerprint = substep.substepFingerprint;
            expected.motor_candidate_fingerprint =
                candidate.candidateFingerprint;
            expected.motor_output_fingerprint = header.outputFingerprint;
            expected.motor_profile_fingerprint =
                candidate.motorProfileFingerprint;
            expected.motor_ready_gate_fingerprint = gate.gateFingerprint;
            expected.brain_program_fingerprint =
                gate.brainProgramFingerprint;
            expected.fast_program_fingerprint = gate.fastProgramFingerprint;
            expected.decision_gate_fingerprint =
                gate.decisionGateFingerprint;
            expected.inbound_authority_fingerprint =
                metalrobo::metalNumanXExactInboundAuthorityV2Fingerprint(
                    expected);
            require(std::memcmp(
                        readbackBuffer.contents,
                        &expected,
                        sizeof(expected)) == 0,
                    "GPU exact authority receipt differs from CPU bytes");
            require(metalrobo::metalNumanXExactInboundAuthorityV2Valid(
                        expected),
                    "CPU rejected the GPU-matched exact authority receipt");
            const auto bridgeRoot = bitwiseMirror<
                mrnx_brain_joint_transaction_v2>(root);
            const auto bridgeSubstep = bitwiseMirror<
                mrnx_brain_joint_substep_v2>(substep);
            const auto bridgeCandidate = bitwiseMirror<
                mrnx_brain_motor_candidate_v2>(candidate);
            const auto bridgeHeader = bitwiseMirror<
                mrnx_brain_motor_output_header_v2>(header);
            const auto bridgeGate = bitwiseMirror<
                mrnx_brain_motor_ready_gate_v2>(gate);
            require(metalrobo::metalNumanXExactInboundAuthorityV2Matches(
                        expected,
                        bridgeRoot,
                        bridgeSubstep,
                        bridgeCandidate,
                        bridgeHeader,
                        bridgeGate),
                    "GPU receipt does not match its exact source authority");

            // Mutating the immutable output header must erase the previously
            // successful receipt from the same private allocation.
            header.outputFingerprint ^= 1u;
            run(false);
            require(*static_cast<const std::uint32_t*>(
                        validationBuffer.contents) ==
                        MR_NUMANX_HUMAN_MOTOR_HEADER_FINGERPRINT,
                    "mutated motor header did not fail its fingerprint");
            require(allZero(readbackBuffer.contents, readbackBuffer.length),
                    "mutated motor header retained stale authority");
            header.outputFingerprint ^= 1u;

            // Re-establish a successful receipt before mutating the separate
            // excitation payload, proving both authenticated sources clear a
            // stale success rather than relying on fresh storage.
            run(false);
            require(*static_cast<const std::uint32_t*>(
                        validationBuffer.contents) ==
                        MR_NUMANX_HUMAN_MOTOR_HEADER_VALID,
                    "restored exact motor authority was rejected");
            // A stale valid receipt must be actively erased when the payload
            // changes without a matching immutable output fingerprint.
            excitation[0] = 0.75f;
            run(false);
            require(*static_cast<const std::uint32_t*>(
                        validationBuffer.contents) ==
                        MR_NUMANX_HUMAN_MOTOR_HEADER_FINGERPRINT,
                    "mutated motor payload did not fail its fingerprint");
            require(allZero(readbackBuffer.contents, readbackBuffer.length),
                    "mutated motor payload retained stale authority");
            excitation[0] = 0.5f;

            // A structurally authentic terminal failure gate is still a hard
            // rejection and must clear poisoned private storage.
            gate.status = MR_NUMANX_BRAIN_READY_GATE_FAILURE;
            gate.gateFingerprint =
                metalrobo::metalNumanXBrainMotorReadyGateV2Fingerprint(gate);
            run(true);
            require(*static_cast<const std::uint32_t*>(
                        validationBuffer.contents) ==
                        MR_NUMANX_HUMAN_MOTOR_HEADER_READY_GATE,
                    "terminal failure gate did not reject exact authority");
            require(allZero(readbackBuffer.contents, readbackBuffer.length),
                    "terminal failure gate emitted or retained authority");

            // Even a corrupt zero environment count clears thread-zero
            // storage before returning. The host builder rejects this shape;
            // this is shader-level stale-state hardening.
            dispatch.environmentCount = 0u;
            run(true);
            require(allZero(readbackBuffer.contents, readbackBuffer.length),
                    "zero-environment dispatch retained poisoned authority");

            std::printf(
                "PASS exact_human_io_receipt device=%s bytes=%zu "
                "fingerprint=%llu storage=private negative_cases=4\n",
                device.name.UTF8String,
                sizeof(expected),
                static_cast<unsigned long long>(
                    expected.inbound_authority_fingerprint));
            return 0;
        } catch (const std::exception& exception) {
            std::fprintf(
                stderr,
                "numanx_human_io_exact_receipt_probe: %s\n",
                exception.what());
            return 1;
        }
    }
}
