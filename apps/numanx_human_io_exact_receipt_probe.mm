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

            // Exercise the public production Context, not only the direct
            // validator seam. The private receipt and all sensor rows must be
            // authored on the same command-buffer timeline.
            gate.status = MR_NUMANX_BRAIN_READY_GATE_SUCCESS;
            gate.gateFingerprint =
                metalrobo::metalNumanXBrainMotorReadyGateV2Fingerprint(gate);
            std::memcpy(headerBuffer.contents, &header, sizeof(header));
            std::memcpy(gateBuffer.contents, &gate, sizeof(gate));
            readyEvent.signaledValue = 1u;

            metalrobo::MetalNumanXHumanIOConfig contextConfig{};
            contextConfig.metallibPath = argv[1];
            metalrobo::MetalNumanXHumanIOContext context(contextConfig);

            metalrobo::MetalNumanXHumanIOInputV2 invalidInput = input;
            ++invalidInput.expectedExcitationGPUAddress;
            metalrobo::MetalNumanXTransactionProgram untouchedProgram{};
            untouchedProgram.fingerprint = 0xfeedfaceu;
            metalrobo::MetalNumanXHumanIOExactPreparedView untouchedView{};
            untouchedView.sensor.sensorGeneration = 0xabcdefu;
            const auto invalidDiagnostics = context.prepare(
                invalidInput, untouchedProgram, untouchedView);
            require(invalidDiagnostics.status ==
                        metalrobo::MetalNumanXHumanIOStatus::invalidInput &&
                        untouchedProgram.fingerprint == 0xfeedfaceu &&
                        untouchedView.sensor.sensorGeneration == 0xabcdefu,
                    "invalid exact prepare mutated public outputs");

            metalrobo::MetalNumanXTransactionProgram contextProgram{};
            metalrobo::MetalNumanXHumanIOExactPreparedView prepared{};
            auto contextDiagnostics = context.prepare(
                input, contextProgram, prepared);
            require(contextDiagnostics.succeeded() &&
                        contextProgram.valid() && prepared.valid(),
                    "public exact HumanIO prepare failed");
            require(prepared.authority.byteCount == sizeof(expected) &&
                        prepared.authority.byteOffset == 0u &&
                        prepared.authority.gpuAddress != 0u &&
                        prepared.authority.metalBuffer != nullptr &&
                        prepared.sensor.receptorTimestampMicroseconds == 0u &&
                        prepared.sensor.deliveryTimestampMicroseconds == 0u &&
                        prepared.sensor.latencyMicroseconds == 0u &&
                        prepared.sensor.timestampQuantumNanoseconds == 1u &&
                        prepared.sensor.receptorTimestampNanoseconds ==
                            substep.startTimestampNanoseconds &&
                        prepared.sensor.deliveryTimestampNanoseconds ==
                            substep.candidateTimestampNanoseconds,
                    "public exact prepared view lost nanosecond authority");
            require(((__bridge id<MTLBuffer>)
                        prepared.authority.metalBuffer).storageMode ==
                        MTLStorageModePrivate,
                    "public exact authority range is not device-private");

            id<MTLBuffer> stateBuffer = [device
                newBufferWithLength:sizeof(MRMujocoMuscleStateGPU)
                           options:MTLResourceStorageModeShared];
            id<MTLBuffer> resultBuffer = [device
                newBufferWithLength:sizeof(MRMujocoMuscleResultGPU)
                           options:MTLResourceStorageModeShared];
            id<MTLBuffer> statusBuffer = [device
                newBufferWithLength:sizeof(MRNumiHumanStandStatusGPU)
                           options:MTLResourceStorageModeShared];
            require(stateBuffer != nil && resultBuffer != nil &&
                        statusBuffer != nil,
                    "public exact transaction buffer allocation failed");
            const auto initializePhysicalInputs = [&] {
                auto& state = *static_cast<MRMujocoMuscleStateGPU*>(
                    stateBuffer.contents);
                auto& result = *static_cast<MRMujocoMuscleResultGPU*>(
                    resultBuffer.contents);
                auto& status = *static_cast<MRNumiHumanStandStatusGPU*>(
                    statusBuffer.contents);
                state = {};
                result = {};
                status = {};
                state.excitationAndActivation = {0.0f, 0.5f, 0.1f, 0.0f};
                result.status = MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS;
                result.environment = 0u;
                result.muscleIndex = 0u;
                result.pathForceAndActivationDerivative = {
                    1.0f, 2.0f, -12.0f, 4.0f};
                result.activeForceAndReserved = {5.0f, 0.0f, 0.0f, 0.0f};
                result.fiberStateTendonForceResidual = {
                    0.1f, 0.0f, 6.0f, 0.01f};
                status.code = MR_NUMI_HUMAN_STAND_SUCCESS;
                status.environment = 0u;
                status.completedSteps = 1u;
                status.failingIndex = MR_INVALID_INDEX;
            };
            initializePhysicalInputs();

            const float physicalTimestepSeconds = static_cast<float>(
                static_cast<double>(input.timestepNanoseconds) * 1.0e-9);
            const auto makePass = [&](id<MTLCommandBuffer> command,
                                      const metalrobo::MetalNumanXTransactionPhase phase) {
                metalrobo::MetalNumanXTransactionPass pass{};
                pass.abiVersion = metalrobo::kMetalNumanXTransactionABIVersion;
                pass.structSize = sizeof(pass);
                pass.accessFlags =
                    metalrobo::MetalNumanXTransactionReadBorrowedState;
                if (phase ==
                    metalrobo::MetalNumanXTransactionPhase::beginStep) {
                    pass.accessFlags |=
                        metalrobo::MetalNumanXTransactionWriteMujocoExcitation;
                } else if (phase ==
                    metalrobo::MetalNumanXTransactionPhase::postDynamics) {
                    pass.accessFlags |=
                        metalrobo::MetalNumanXTransactionWriteStandFailure;
                }
                pass.commandBuffer = (__bridge void*)command;
                pass.mujocoStates = (__bridge void*)stateBuffer;
                pass.mujocoResults = (__bridge void*)resultBuffer;
                pass.standStatuses = (__bridge void*)statusBuffer;
                pass.phase = phase;
                pass.programFingerprint = contextProgram.fingerprint;
                pass.stepCount = 1u;
                pass.timestepSeconds = physicalTimestepSeconds;
                pass.environmentCount = 1u;
                pass.mujocoMuscleCount = 1u;
                pass.mujocoStateElementCount = 1u;
                pass.mujocoStateStride = 1u;
                pass.mujocoResultElementCount = 1u;
                pass.mujocoResultStride = 1u;
                pass.standStatusElementCount = 1u;
                pass.standStatusStride = 1u;
                return pass;
            };

            id<MTLBuffer> contextReceiptReadback = [device
                newBufferWithLength:prepared.authority.byteCount
                           options:MTLResourceStorageModeShared];
            id<MTLBuffer> proprioceptionReadback = [device
                newBufferWithLength:prepared.sensor.proprioceptionByteCount
                           options:MTLResourceStorageModeShared];
            id<MTLBuffer> validityReadback = [device
                newBufferWithLength:prepared.sensor.validityByteCount
                           options:MTLResourceStorageModeShared];
            id<MTLBuffer> interoceptionReadback = [device
                newBufferWithLength:prepared.sensor.interoceptionByteCount
                           options:MTLResourceStorageModeShared];
            id<MTLBuffer> interoceptionValidityReadback = [device
                newBufferWithLength:
                    prepared.sensor.interoceptionValidityByteCount
                           options:MTLResourceStorageModeShared];
            require(contextReceiptReadback != nil &&
                        proprioceptionReadback != nil &&
                        validityReadback != nil &&
                        interoceptionReadback != nil &&
                        interoceptionValidityReadback != nil,
                    "public exact readback allocation failed");

            id<MTLCommandBuffer> contextCommand = [queue commandBuffer];
            require(contextCommand != nil,
                    "public exact command buffer allocation failed");
            for (const auto phase : {
                     metalrobo::MetalNumanXTransactionPhase::beginStep,
                     metalrobo::MetalNumanXTransactionPhase::preDynamics,
                     metalrobo::MetalNumanXTransactionPhase::postDynamics,
                 }) {
                const auto pass = makePass(contextCommand, phase);
                require(contextProgram.encode(contextProgram.context, pass),
                        "public exact transaction phase was rejected");
            }
            metalrobo::MetalNumanXHumanIOTransactionKey contextKey{};
            metalrobo::MetalNumanXHumanIOSensorView pendingSensor{};
            contextDiagnostics = context.pendingCandidate(
                contextKey, pendingSensor);
            require(contextDiagnostics.succeeded() && contextKey.valid(),
                    "public exact pending key was unavailable after encoding");

            id<MTLBlitCommandEncoder> contextReadback =
                [contextCommand blitCommandEncoder];
            require(contextReadback != nil,
                    "public exact readback blit unavailable");
            [contextReadback
                copyFromBuffer:(__bridge id<MTLBuffer>)
                    prepared.authority.metalBuffer
                   sourceOffset:prepared.authority.byteOffset
                       toBuffer:contextReceiptReadback
              destinationOffset:0u
                           size:prepared.authority.byteCount];
            [contextReadback
                copyFromBuffer:(__bridge id<MTLBuffer>)
                    prepared.sensor.proprioceptionMetalBuffer
                   sourceOffset:0u
                       toBuffer:proprioceptionReadback
              destinationOffset:0u
                           size:prepared.sensor.proprioceptionByteCount];
            [contextReadback
                copyFromBuffer:(__bridge id<MTLBuffer>)
                    prepared.sensor.validityMetalBuffer
                   sourceOffset:0u
                       toBuffer:validityReadback
              destinationOffset:0u
                           size:prepared.sensor.validityByteCount];
            [contextReadback
                copyFromBuffer:(__bridge id<MTLBuffer>)
                    prepared.sensor.interoceptionMetalBuffer
                   sourceOffset:0u
                       toBuffer:interoceptionReadback
              destinationOffset:0u
                           size:prepared.sensor.interoceptionByteCount];
            [contextReadback
                copyFromBuffer:(__bridge id<MTLBuffer>)
                    prepared.sensor.interoceptionValidityMetalBuffer
                   sourceOffset:0u
                       toBuffer:interoceptionValidityReadback
              destinationOffset:0u
                           size:
                               prepared.sensor.interoceptionValidityByteCount];
            [contextReadback endEncoding];
            [contextCommand commit];
            [contextCommand waitUntilCompleted];
            require(contextCommand.status == MTLCommandBufferStatusCompleted &&
                        contextCommand.error == nil,
                    "public exact transaction command failed");
            require(std::memcmp(
                        contextReceiptReadback.contents,
                        &expected,
                        sizeof(expected)) == 0,
                    "public Context private receipt differs from CPU authority");
            require(static_cast<const float*>(
                        stateBuffer.contents)[0] == excitation[0],
                    "public exact admission did not write excitation");
            require(static_cast<const float*>(
                        proprioceptionReadback.contents)[0] == excitation[0],
                    "public exact sensor row did not preserve excitation");
            require(*static_cast<const std::uint32_t*>(
                        validityReadback.contents) ==
                        MR_NUMANX_HUMAN_PROPRIOCEPTION_VALIDITY_ALL &&
                        *static_cast<const std::uint32_t*>(
                            interoceptionValidityReadback.contents) ==
                        MR_NUMANX_HUMAN_INTEROCEPTION_VALIDITY_ALL,
                    "public exact sensor validity was not admitted");
            require(!allZero(
                        interoceptionReadback.contents,
                        interoceptionReadback.length),
                    "public exact interoception remained empty");
            contextDiagnostics = context.pendingCandidate(
                contextKey, pendingSensor);
            require(contextDiagnostics.succeeded() &&
                        pendingSensor.commandBufferIdentity ==
                            reinterpret_cast<std::uintptr_t>(
                                (__bridge void*)contextCommand) &&
                        pendingSensor.transactionInstanceFingerprint ==
                            contextKey.transactionInstanceFingerprint,
                    "public exact completion lost command identity");
            metalrobo::MetalNumanXHumanIOCandidatePublicationLease
                legacyPublicationLease{};
            contextDiagnostics = context.reserveCandidatePublication(
                contextKey, legacyPublicationLease);
            require(contextDiagnostics.status ==
                        metalrobo::MetalNumanXHumanIOStatus::
                            candidateUnavailable &&
                        !legacyPublicationLease.valid(),
                    "exact candidate escaped through the legacy publication API");
            require(context.reject(contextKey).succeeded(),
                    "public exact candidate rejection failed");

            // A terminal gate failure is transport-successful but must mark
            // the owning physical status failed, clear the private receipt,
            // and leave every reused sensor byte zero.
            initializePhysicalInputs();
            gate.status = MR_NUMANX_BRAIN_READY_GATE_FAILURE;
            gate.gateFingerprint =
                metalrobo::metalNumanXBrainMotorReadyGateV2Fingerprint(gate);
            std::memcpy(gateBuffer.contents, &gate, sizeof(gate));
            input.candidateSensorGeneration = 2u;
            contextProgram = {};
            prepared = {};
            contextDiagnostics = context.prepare(
                input, contextProgram, prepared);
            require(contextDiagnostics.succeeded() && prepared.valid(),
                    "terminal-gate negative prepare failed unexpectedly");
            id<MTLCommandBuffer> rejectedCommand = [queue commandBuffer];
            require(rejectedCommand != nil,
                    "terminal-gate command buffer allocation failed");
            for (const auto phase : {
                     metalrobo::MetalNumanXTransactionPhase::beginStep,
                     metalrobo::MetalNumanXTransactionPhase::preDynamics,
                     metalrobo::MetalNumanXTransactionPhase::postDynamics,
                 }) {
                const auto pass = makePass(rejectedCommand, phase);
                require(contextProgram.encode(contextProgram.context, pass),
                        "terminal-gate transaction phase was rejected by host");
            }
            metalrobo::MetalNumanXHumanIOTransactionKey rejectedKey{};
            contextDiagnostics = context.pendingCandidate(
                rejectedKey, pendingSensor);
            require(contextDiagnostics.succeeded() && rejectedKey.valid(),
                    "terminal-gate pending key was unavailable");
            id<MTLBlitCommandEncoder> rejectedReadback =
                [rejectedCommand blitCommandEncoder];
            require(rejectedReadback != nil,
                    "terminal-gate readback blit unavailable");
            [rejectedReadback
                copyFromBuffer:(__bridge id<MTLBuffer>)
                    prepared.authority.metalBuffer
                   sourceOffset:0u
                       toBuffer:contextReceiptReadback
              destinationOffset:0u
                           size:prepared.authority.byteCount];
            [rejectedReadback
                copyFromBuffer:(__bridge id<MTLBuffer>)
                    prepared.sensor.proprioceptionMetalBuffer
                   sourceOffset:0u
                       toBuffer:proprioceptionReadback
              destinationOffset:0u
                           size:prepared.sensor.proprioceptionByteCount];
            [rejectedReadback
                copyFromBuffer:(__bridge id<MTLBuffer>)
                    prepared.sensor.validityMetalBuffer
                   sourceOffset:0u
                       toBuffer:validityReadback
              destinationOffset:0u
                           size:prepared.sensor.validityByteCount];
            [rejectedReadback endEncoding];
            [rejectedCommand commit];
            [rejectedCommand waitUntilCompleted];
            require(rejectedCommand.status == MTLCommandBufferStatusCompleted &&
                        rejectedCommand.error == nil,
                    "terminal-gate command transport failed");
            require(static_cast<const MRNumiHumanStandStatusGPU*>(
                        statusBuffer.contents)->code ==
                        MR_NUMI_HUMAN_STAND_INVALID_DISPATCH,
                    "terminal-gate authority failure left stand status successful");
            require(allZero(
                        contextReceiptReadback.contents,
                        contextReceiptReadback.length) &&
                        allZero(
                            proprioceptionReadback.contents,
                            proprioceptionReadback.length) &&
                        allZero(
                            validityReadback.contents,
                            validityReadback.length),
                    "terminal-gate authority failure retained candidate bytes");
            require(context.reject(rejectedKey).succeeded(),
                    "terminal-gate candidate rejection failed");

            std::printf(
                "PASS exact_human_io_receipt device=%s bytes=%zu "
                "fingerprint=%llu storage=private negative_cases=6 "
                "context_v2=passed\n",
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
