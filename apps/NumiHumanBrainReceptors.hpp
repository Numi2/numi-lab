#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/numanx_human_io_gpu.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// Bounded receptor transport for the existing single-world standing probe.
// Native physics owns every source buffer and its acceptance. This helper
// appends sensor work and publishes only its own double-buffered observations.
namespace numi_human_brain {

struct ContactBinding {
    std::uint32_t contactIndex = 0u;
    std::uint32_t receptorIndex = 0u;
    std::uint32_t bodyIndex = 0u;
    std::uint32_t sourceGeometryIndex = 0u;
};

struct Channel {
    std::uint32_t modality = 0u;
    std::uint32_t receptorCount = 0u;
    std::uint32_t featureCount = 0u;
    id<MTLBuffer> values = nil;
    id<MTLBuffer> validity = nil;
};

struct Frame {
    // Sorted by Brain's canonical SensoryModality raw value: vision, audition,
    // touch, proprioception, vestibular, interoception, kinesthesia.
    std::array<Channel, 7u> channels{};
    // Head pose and source MyoSim/path measurements belong to the existing
    // pre-dynamics evaluation at this step start. Root position and linear
    // velocity are accepted endpoint coordinates. Touch force is impulse/dt over
    // [receptorTimestamp, deliveryTimestamp], with pre-step slip. The frame
    // is first deliverable at that interval's accepted end.
    std::uint64_t receptorTimestampMicroseconds = 0u;
    std::uint64_t deliveryTimestampMicroseconds = 0u;
    std::uint64_t acceptedGeneration = 0u;
    std::uint64_t acceptedPhysicsFingerprint = 0u;
};

class Receptors {
public:
    Receptors(id<MTLDevice> device, id<MTLLibrary> nativeLibrary,
              std::uint32_t bodyCount, std::uint32_t headBodyIndex,
              std::uint64_t modelFingerprint,
              std::span<const ContactBinding> contactBindings,
              std::span<const MRMujocoMuscleResultGPU> preparedMuscleResults,
              std::span<const MRDofPropertiesGPU> sourceDofs,
              std::span<const float> preparedQ, std::span<const float> preparedV,
              bool enableJointKinesthesia,
              std::uint32_t timestepMicroseconds,
              std::uint64_t epochMicroseconds)
        : device_(device), bodyCount_(bodyCount), headBodyIndex_(headBodyIndex),
          timestepMicroseconds_(timestepMicroseconds), epochMicroseconds_(epochMicroseconds),
          jointKinesthesiaEnabled_(enableJointKinesthesia) {
        require(device != nil && nativeLibrary != nil &&
                    nativeLibrary.device.registryID == device.registryID,
                "Human Brain receptors require the owning native Metal device/library");
        require(bodyCount > 0u && bodyCount <= MR_NUMI_HUMAN_STAND_MAX_BODIES &&
                    headBodyIndex < bodyCount && modelFingerprint != 0u &&
                    timestepMicroseconds > 0u && epochMicroseconds >= timestepMicroseconds &&
                    preparedMuscleResults.size() == 416u &&
                    (!enableJointKinesthesia ||
                     (sourceDofs.size() == 128u && preparedQ.size() == 129u &&
                      preparedV.size() == 128u)) &&
                    !contactBindings.empty() && contactBindings.size() <= 20u,
                "Human Brain receptor source binding or physical clock is invalid");
        std::array<bool, 20u> rows{};
        std::array<bool, 10u> receptors{};
        std::array<mr_uint4, 10u> identities{};
        std::vector<mr_uint4> mappings;
        mappings.reserve(contactBindings.size());
        hash(modelFingerprint);
        hash(bodyCount); hash(headBodyIndex); hash(timestepMicroseconds); hash(epochMicroseconds);
        // Binds sparse physical meanings, including invalid unimplemented modalities.
        const std::uint32_t receptorProgramVersion = enableJointKinesthesia ? 4u : 3u;
        hash(receptorProgramVersion);
        std::array<bool, 129u> ownedQ{};
        qIndexByV_.fill(MR_INVALID_INDEX);
        for (std::uint32_t vIndex = 0u; enableJointKinesthesia && vIndex < sourceDofs.size(); ++vIndex) {
            const auto& dof = sourceDofs[vIndex];
            require(dof.vIndex == vIndex && dof.reserved0 == 0u &&
                        dof.reserved1 == 0u && std::isfinite(preparedV[vIndex]),
                    "Human Brain kinesthesia source velocity is invalid");
            if (vIndex < 6u) {
                require((dof.flags & MR_DOF_FLAG_ROOT) != 0u,
                        "Human Brain kinesthesia root row is not source-authored root");
                continue;
            }
            require((dof.flags & MR_DOF_FLAG_ROOT) == 0u &&
                        dof.qIndex >= 7u && dof.qIndex < preparedQ.size() &&
                        !ownedQ[dof.qIndex] && std::isfinite(preparedQ[dof.qIndex]),
                    "Human Brain kinesthesia source coordinate is absent or duplicated");
            ownedQ[dof.qIndex] = true;
            qIndexByV_[vIndex] = dof.qIndex;
            hash(vIndex); hash(dof.qIndex);
            hash(std::bit_cast<std::uint32_t>(preparedQ[dof.qIndex]));
            hash(std::bit_cast<std::uint32_t>(preparedV[vIndex]));
        }
        if (enableJointKinesthesia) {
            for (std::uint32_t qIndex = 7u; qIndex < ownedQ.size(); ++qIndex)
                require(ownedQ[qIndex], "Human Brain kinesthesia source q coverage is incomplete");
        }
        qIndexBuffer_ = [device newBufferWithBytes:qIndexByV_.data()
            length:qIndexByV_.size() * sizeof(std::uint32_t)
            options:MTLResourceStorageModeShared];
        require(qIndexBuffer_ != nil, "Human Brain kinesthesia source map allocation failed");
        qIndexBuffer_.label = @"Human Brain immutable source kinesthesia coordinates";
        for (const ContactBinding& binding : contactBindings) {
            require(binding.contactIndex < contactBindings.size() && !rows[binding.contactIndex] &&
                        binding.receptorIndex < 10u && binding.bodyIndex < bodyCount,
                    "Human Brain contact binding is out of range or repeated");
            rows[binding.contactIndex] = true;
            const mr_uint4 identity{binding.bodyIndex, binding.sourceGeometryIndex, 0u, 0u};
            if (receptors[binding.receptorIndex]) {
                require(identities[binding.receptorIndex].x == identity.x &&
                            identities[binding.receptorIndex].y == identity.y,
                        "one touch receptor cannot alias different physical source geometries");
            }
            receptors[binding.receptorIndex] = true;
            identities[binding.receptorIndex] = identity;
            mappings.push_back({binding.contactIndex, binding.receptorIndex,
                                binding.bodyIndex, binding.sourceGeometryIndex});
            hash(binding.contactIndex); hash(binding.receptorIndex);
            hash(binding.bodyIndex); hash(binding.sourceGeometryIndex);
        }
        for (bool present : receptors) require(present, "canonical Human touch requires ten source receptors");
        contactCount_ = static_cast<std::uint32_t>(mappings.size());
        bindings_ = [device newBufferWithBytes:mappings.data()
            length:mappings.size() * sizeof(mr_uint4) options:MTLResourceStorageModeShared];
        require(bindings_ != nil, "Human Brain contact binding allocation failed");
        bindings_.label = @"Human Brain immutable source contact receptors";

        for (Slot& slot : slots_) {
            constexpr std::array<std::array<std::uint32_t, 3u>, 7u> shapes{{
                {{1u, 3072u, 8u}}, {{2u, 24u, 8u}}, {{3u, 10u, 7u}},
                {{4u, 416u, 10u}}, {{5u, 1u, 22u}}, {{8u, 416u, 6u}}, {{9u, 128u, 7u}}
            }};
            for (std::size_t index = 0u; index < shapes.size(); ++index) {
                const auto& shape = shapes[index];
                Channel& channel = slot.frame.channels[index];
                channel.modality = shape[0]; channel.receptorCount = shape[1]; channel.featureCount = shape[2];
                channel.values = zeroBuffer(std::size_t(shape[1]) * shape[2] * sizeof(float));
                channel.validity = zeroBuffer(std::size_t(shape[1]) * sizeof(std::uint32_t));
            }
            slot.environmentGate = zeroBuffer(sizeof(std::uint32_t));
            slot.frame.receptorTimestampMicroseconds = epochMicroseconds - timestepMicroseconds;
            slot.frame.deliveryTimestampMicroseconds = epochMicroseconds;
        }

        // The initial packet is the source evaluator's actual prepared
        // t=0 path measurement, delivered after one modeled sensory latency.
        // Prepared articulated q/v are the exact first native input and can
        // seed the one-step-delayed initial packet. No accepted impulse,
        // body-motion or physiology measurement exists yet.
        Channel& initialSpindles = slots_[publishedSlot_].frame.channels[3u];
        auto* values = static_cast<float*>(initialSpindles.values.contents);
        auto* validity = static_cast<std::uint32_t*>(initialSpindles.validity.contents);
        constexpr std::uint32_t pathValidity =
            (1u << MR_NUMANX_HUMAN_FEATURE_PATH_LENGTH_METRES) |
            (1u << MR_NUMANX_HUMAN_FEATURE_PATH_VELOCITY_METRES_PER_SECOND);
        for (std::size_t muscle = 0u; muscle < preparedMuscleResults.size(); ++muscle) {
            const auto& result = preparedMuscleResults[muscle];
            const auto path = result.pathForceAndActivationDerivative;
            require(result.status == MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS &&
                        result.environment == 0u && result.muscleIndex == muscle &&
                        std::isfinite(path.x) && path.x > 0.0f &&
                        std::isfinite(path.y),
                    "Human Brain initial spindle is not an admitted prepared path measurement");
            const std::size_t offset = muscle * MR_NUMANX_HUMAN_PROPRIOCEPTION_FEATURE_COUNT;
            values[offset + MR_NUMANX_HUMAN_FEATURE_PATH_LENGTH_METRES] = path.x;
            values[offset + MR_NUMANX_HUMAN_FEATURE_PATH_VELOCITY_METRES_PER_SECOND] = path.y;
            validity[muscle] = pathValidity;
        }
        if (enableJointKinesthesia) {
            Channel& initialKinesthesia = slots_[publishedSlot_].frame.channels[6u];
            auto* jointValues = static_cast<float*>(initialKinesthesia.values.contents);
            auto* jointValidity = static_cast<std::uint32_t*>(initialKinesthesia.validity.contents);
            for (std::uint32_t vIndex = 6u; vIndex < 128u; ++vIndex) {
                const std::uint32_t qIndex = qIndexByV_[vIndex];
                jointValues[vIndex * MR_NUMANX_HUMAN_KINESTHESIA_FEATURE_COUNT] = preparedQ[qIndex];
                jointValues[vIndex * MR_NUMANX_HUMAN_KINESTHESIA_FEATURE_COUNT + 1u] = preparedV[vIndex];
                jointValidity[vIndex] = 0x00000003u;
            }
        }

        NSError* error = nil;
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        id<MTLLibrary> localLibrary = [device newLibraryWithSource:shaderSource()
            options:options error:&error];
        require(localLibrary != nil, error == nil ? "Human Brain receptor shader compilation failed"
            : std::string(error.localizedDescription.UTF8String));
        gatePipeline_ = pipeline(localLibrary, @"human_brain_receptor_gate");
        bodyTouchPipeline_ = pipeline(localLibrary, @"human_brain_body_touch");
        // Preserve the existing native spindle feature definitions and units.
        spindlePipeline_ = pipeline(nativeLibrary, @"numanx_human_write_proprioception");
    }

    Receptors(const Receptors&) = delete;
    Receptors& operator=(const Receptors&) = delete;
    Receptors(Receptors&&) = delete;
    Receptors& operator=(Receptors&&) = delete;

    [[nodiscard]] std::uint64_t fingerprint() const noexcept { return fingerprint_; }
    [[nodiscard]] bool jointKinesthesiaEnabled() const noexcept { return jointKinesthesiaEnabled_; }
    [[nodiscard]] std::uint32_t sourceQIndex(std::uint32_t vIndex) const {
        require(vIndex >= 6u && vIndex < qIndexByV_.size(),
                "Human Brain kinesthesia source row is not articulated");
        return qIndexByV_[vIndex];
    }
    [[nodiscard]] const Frame& deliveredFrame(std::uint64_t committedTimestampMicroseconds) const {
        const Frame& frame = slots_[publishedSlot_].frame;
        require(!pending_ && frame.deliveryTimestampMicroseconds == committedTimestampMicroseconds,
                "Human Brain receptors do not match the exact committed delivery boundary");
        return frame;
    }

    // Available only to the owner-validated accepted-consequence pass. Merely
    // borrowing this candidate never advances the delivered sensor generation.
    [[nodiscard]] const Frame& pendingFrame(std::uint32_t completedStepCount) const {
        const Frame& frame = slots_[1u - publishedSlot_].frame;
        require(pending_ && encoded_ && frame.acceptedGeneration == completedStepCount,
                "Human Brain receptor candidate does not match the accepted native step");
        return frame;
    }

    // No queue, submission, wait, or retention of pass objects. The owner must
    // abandon the command buffer if this returns false, then abortCandidate().
    [[nodiscard]] bool encodeCandidate(const metalrobo::MetalNumanXTransactionPass& pass) noexcept {
        @autoreleasepool {
            try {
                using Phase = metalrobo::MetalNumanXTransactionPhase;
                if (pending_ || pass.abiVersion != metalrobo::kMetalNumanXTransactionABIVersion ||
                    pass.structSize != sizeof(pass) || pass.phase != Phase::postDynamics ||
                    pass.stepIndex >= pass.stepCount || pass.reserved0 != 0u ||
                    pass.accessFlags != (metalrobo::MetalNumanXTransactionReadBorrowedState |
                        metalrobo::MetalNumanXTransactionWriteStandFailure) ||
                    pass.environmentCount != 1u || pass.bodyCount != bodyCount_ ||
                    pass.dofCount != 128u || pass.qCoordinateCount != 129u ||
                    pass.qElementCount != 129u || pass.qStride != 129u ||
                    pass.mujocoMuscleCount != 416u || pass.mujocoStateElementCount != 416u ||
                    pass.mujocoResultElementCount != 416u || pass.mujocoStateStride != 416u ||
                    pass.mujocoResultStride != 416u || pass.standStatusElementCount != 1u ||
                    pass.standContactCount != contactCount_ || pass.standContactEnabled != 1u ||
                    pass.standContactImpulseSampleStepIndex != pass.stepIndex ||
                    pass.standContactImpulseOffset != 4u * pass.dofCount ||
                    pass.vElementCount != 128u || pass.vStride != 128u ||
                    pass.standVectorStride != pass.standVectorElementCount ||
                    pass.standVectorStride < pass.standContactImpulseOffset + 3u * contactCount_ ||
                    pass.pointCount > std::numeric_limits<std::uint32_t>::max() ||
                    pass.pointJacobianElementCount < pass.pointCount * 3u * pass.dofCount ||
                    pass.stepIndex != slots_[publishedSlot_].frame.acceptedGeneration ||
                    pass.programFingerprint == 0u || pass.commandBuffer == nullptr ||
                    pass.timestepSeconds != static_cast<float>(timestepMicroseconds_) * 1.0e-6f)
                    return false;
                const auto stepTime = std::uint64_t(pass.stepIndex) * timestepMicroseconds_;
                if (epochMicroseconds_ > std::numeric_limits<std::uint64_t>::max() - stepTime - timestepMicroseconds_)
                    return false;
                __unsafe_unretained id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
                if (command.device.registryID != device_.registryID ||
                    !validBuffer(pass.mujocoStates, 416u * sizeof(MRMujocoMuscleStateGPU)) ||
                    !validBuffer(pass.mujocoResults, 416u * sizeof(MRMujocoMuscleResultGPU)) ||
                    !validBuffer(pass.standStatuses, sizeof(MRNumiHumanStandStatusGPU)) ||
                    !validBuffer(pass.bodyPoses, bodyCount_ * sizeof(MRArticulatedBodyPoseGPU)) ||
                    !validBuffer(pass.standContacts, contactCount_ * sizeof(MRNumiHumanStandContactGPU)) ||
                    !validBuffer(pass.q, 129u * sizeof(float)) ||
                    !validBuffer(pass.v, 128u * sizeof(float)) ||
                    !validBuffer(pass.standVectorWorkspace, pass.standVectorElementCount * sizeof(float)) ||
                    !validBuffer(pass.pointJacobians, pass.pointJacobianElementCount * sizeof(float))) return false;

                pending_ = true;
                pendingCommand_ = reinterpret_cast<std::uintptr_t>(pass.commandBuffer);
                pendingProgram_ = pass.programFingerprint;
                pendingStep_ = pass.stepIndex;
                Slot& slot = slots_[1u - publishedSlot_];
                slot.frame.receptorTimestampMicroseconds = epochMicroseconds_ + stepTime;
                slot.frame.deliveryTimestampMicroseconds = epochMicroseconds_ + stepTime + timestepMicroseconds_;
                slot.frame.acceptedGeneration = std::uint64_t(pass.stepIndex) + 1u;
                slot.frame.acceptedPhysicsFingerprint = 0u;

                // These layouts are actual native ABI offsets, not a duplicate
                // result/status declaration in the inline Metal program.
                const mr_uint4 gateShape{416u, pass.stepIndex,
                    static_cast<std::uint32_t>(sizeof(MRMujocoMuscleResultGPU) / sizeof(std::uint32_t)),
                    static_cast<std::uint32_t>(offsetof(MRMujocoMuscleResultGPU, activeForceAndReserved) / sizeof(float))};
                const mr_uint4 gateOffsets{
                    static_cast<std::uint32_t>(offsetof(MRMujocoMuscleResultGPU, pathForceAndActivationDerivative) / sizeof(float)),
                    static_cast<std::uint32_t>(offsetof(MRMujocoMuscleResultGPU, fiberStateTendonForceResidual) / sizeof(float)),
                    static_cast<std::uint32_t>(offsetof(MRNumiHumanStandStatusGPU, completedSteps) / sizeof(std::uint32_t)), 0u};
                id<MTLComputeCommandEncoder> gate = [command computeCommandEncoder];
                if (gate == nil) return false;
                [gate setComputePipelineState:gatePipeline_];
                [gate setBuffer:(__bridge id<MTLBuffer>)pass.mujocoStates offset:0u atIndex:0u];
                [gate setBuffer:(__bridge id<MTLBuffer>)pass.mujocoResults offset:0u atIndex:1u];
                [gate setBuffer:(__bridge id<MTLBuffer>)pass.standStatuses offset:0u atIndex:2u];
                [gate setBuffer:slot.environmentGate offset:0u atIndex:3u];
                [gate setBytes:&gateShape length:sizeof(gateShape) atIndex:4u];
                [gate setBytes:&gateOffsets length:sizeof(gateOffsets) atIndex:5u];
                [gate dispatchThreads:MTLSizeMake(1u, 1u, 1u) threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
                [gate endEncoding];

                MRNumanXHumanProprioceptionDispatchGPU dispatch{};
                dispatch.abiVersion = MR_NUMANX_HUMAN_IO_ABI_VERSION;
                dispatch.environmentCount = 1u; dispatch.muscleCount = 416u;
                dispatch.featureCount = MR_NUMANX_HUMAN_PROPRIOCEPTION_FEATURE_COUNT;
                // One compact candidate frame. The physical status gate above
                // uses the true global step; tensor addressing uses local zero.
                dispatch.stepIndex = 0u; dispatch.stepCount = 1u;
                dispatch.stateStride = 416u; dispatch.resultStride = 416u;
                dispatch.proprioceptionEnvironmentStride = 416u * 10u;
                dispatch.proprioceptionStepStride = 416u * 10u;
                dispatch.validityEnvironmentStride = 416u; dispatch.validityStepStride = 416u;
                dispatch.timestepSecondsAndReserved.x = pass.timestepSeconds;
                id<MTLComputeCommandEncoder> writer = [command computeCommandEncoder];
                if (writer == nil) return false;
                [writer setComputePipelineState:spindlePipeline_];
                [writer setBuffer:(__bridge id<MTLBuffer>)pass.mujocoStates offset:0u atIndex:0u];
                [writer setBuffer:(__bridge id<MTLBuffer>)pass.mujocoResults offset:0u atIndex:1u];
                [writer setBuffer:slot.environmentGate offset:0u atIndex:2u];
                [writer setBuffer:slot.frame.channels[3u].values offset:0u atIndex:3u];
                [writer setBuffer:slot.frame.channels[3u].validity offset:0u atIndex:4u];
                [writer setBuffer:slot.frame.channels[5u].values offset:0u atIndex:5u];
                [writer setBuffer:slot.frame.channels[5u].validity offset:0u atIndex:6u];
                [writer setBytes:&dispatch length:sizeof(dispatch) atIndex:7u];
                [writer dispatchThreads:MTLSizeMake(416u, 1u, 1u) threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
                [writer memoryBarrierWithScope:MTLBarrierScopeBuffers];

                const mr_uint4 sensorShape{headBodyIndex_, contactCount_, 128u, static_cast<std::uint32_t>(pass.pointCount)};
                const mr_uint4 sensorOffsets{static_cast<std::uint32_t>(pass.standContactImpulseOffset),
                    3u * 128u, jointKinesthesiaEnabled_ ? 1u : 0u, 0u};
                mr_float4 groundAndTimestep = pass.standGroundNormal;
                groundAndTimestep.w = pass.timestepSeconds;
                [writer setComputePipelineState:bodyTouchPipeline_];
                [writer setBuffer:slot.environmentGate offset:0u atIndex:0u];
                [writer setBuffer:(__bridge id<MTLBuffer>)pass.bodyPoses offset:0u atIndex:1u];
                [writer setBuffer:(__bridge id<MTLBuffer>)pass.standContacts offset:0u atIndex:2u];
                [writer setBuffer:(__bridge id<MTLBuffer>)pass.standVectorWorkspace offset:0u atIndex:3u];
                [writer setBuffer:(__bridge id<MTLBuffer>)pass.pointJacobians offset:0u atIndex:4u];
                [writer setBuffer:bindings_ offset:0u atIndex:5u];
                [writer setBuffer:slot.frame.channels[2u].values offset:0u atIndex:6u];
                [writer setBuffer:slot.frame.channels[2u].validity offset:0u atIndex:7u];
                [writer setBuffer:slot.frame.channels[4u].values offset:0u atIndex:8u];
                [writer setBuffer:slot.frame.channels[4u].validity offset:0u atIndex:9u];
                [writer setBuffer:slot.frame.channels[5u].values offset:0u atIndex:10u];
                [writer setBuffer:slot.frame.channels[5u].validity offset:0u atIndex:11u];
                [writer setBytes:&sensorShape length:sizeof(sensorShape) atIndex:12u];
                [writer setBytes:&sensorOffsets length:sizeof(sensorOffsets) atIndex:13u];
                [writer setBytes:&groundAndTimestep length:sizeof(groundAndTimestep) atIndex:14u];
                [writer setBuffer:(__bridge id<MTLBuffer>)pass.v offset:0u atIndex:15u];
                [writer setBuffer:(__bridge id<MTLBuffer>)pass.q offset:0u atIndex:16u];
                [writer setBuffer:qIndexBuffer_ offset:0u atIndex:17u];
                [writer setBuffer:slot.frame.channels[6u].values offset:0u atIndex:18u];
                [writer setBuffer:slot.frame.channels[6u].validity offset:0u atIndex:19u];
                [writer dispatchThreads:MTLSizeMake(416u, 1u, 1u) threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
                [writer endEncoding];
                encoded_ = true;
                return true;
            } catch (...) { return false; }
        }
    }

    [[nodiscard]] bool canPublish(const metalrobo::MetalArticulatedOperatorDiagnostics& diagnostics,
                                 const metalrobo::MetalArticulatedOperatorResult& result,
                                 std::uint64_t acceptedPhysicsFingerprint) const noexcept {
        return pending_ && encoded_ && acceptedPhysicsFingerprint != 0u &&
            diagnostics.succeeded() && diagnostics.dispatched && diagnostics.published &&
            diagnostics.successfulEnvironmentCount == 1u && diagnostics.failedEnvironmentCount == 0u &&
            diagnostics.completedStandSteps == pendingStep_ + 1u &&
            diagnostics.commandBufferIdentity == pendingCommand_ &&
            diagnostics.numanXProgramFingerprint == pendingProgram_ &&
            result.standQ.size() == 129u && result.standV.size() == 128u &&
            result.standRootTranslations.size() == 1u && result.mujocoActivationStates.size() == 416u &&
            result.standStatuses.size() == 1u && result.standStatuses[0u].environment == 0u &&
            result.standStatuses[0u].code == MR_NUMI_HUMAN_STAND_SUCCESS &&
            result.standStatuses[0u].completedSteps == pendingStep_ + 1u;
    }

    [[nodiscard]] bool publishAccepted(const metalrobo::MetalArticulatedOperatorDiagnostics& diagnostics,
                                       const metalrobo::MetalArticulatedOperatorResult& result,
                                       std::uint64_t acceptedPhysicsFingerprint) noexcept {
        if (!canPublish(diagnostics, result, acceptedPhysicsFingerprint)) return false;
        publishedSlot_ = 1u - publishedSlot_;
        slots_[publishedSlot_].frame.acceptedPhysicsFingerprint = acceptedPhysicsFingerprint;
        abortCandidate();
        return true;
    }

    // Call only after abandonment before commit, or after the owning command
    // completed. Candidate bytes may be overwritten next time; accepted bytes
    // and the accepted generation are never changed here.
    void abortCandidate() noexcept {
        pending_ = false; encoded_ = false; pendingCommand_ = 0u; pendingProgram_ = 0u; pendingStep_ = 0u;
    }

private:
    struct Slot { Frame frame{}; id<MTLBuffer> environmentGate = nil; };
    id<MTLDevice> device_ = nil;
    id<MTLBuffer> bindings_ = nil;
    id<MTLBuffer> qIndexBuffer_ = nil;
    id<MTLComputePipelineState> gatePipeline_ = nil, spindlePipeline_ = nil, bodyTouchPipeline_ = nil;
    std::array<Slot, 2u> slots_{};
    std::array<std::uint32_t, 128u> qIndexByV_{};
    std::uint32_t bodyCount_ = 0u, headBodyIndex_ = 0u, contactCount_ = 0u;
    std::uint32_t timestepMicroseconds_ = 0u;
    std::uint64_t epochMicroseconds_ = 0u;
    bool jointKinesthesiaEnabled_ = false;
    std::uint64_t fingerprint_ = 14695981039346656037ull;
    std::uint32_t publishedSlot_ = 0u, pendingStep_ = 0u;
    std::uintptr_t pendingCommand_ = 0u;
    std::uint64_t pendingProgram_ = 0u;
    bool pending_ = false, encoded_ = false;

    static void require(bool condition, const std::string& message) {
        if (!condition) throw std::runtime_error(message);
    }
    template <typename T> void hash(const T& value) noexcept {
        const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
        for (std::size_t index = 0u; index < sizeof(T); ++index) {
            fingerprint_ ^= bytes[index]; fingerprint_ *= 1099511628211ull;
        }
        if (fingerprint_ == 0u) fingerprint_ = 14695981039346656037ull;
    }
    id<MTLBuffer> zeroBuffer(std::size_t bytes) {
        id<MTLBuffer> buffer = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        require(buffer != nil && buffer.contents != nullptr, "Human Brain receptor allocation failed");
        std::memset(buffer.contents, 0, bytes);
        return buffer;
    }
    bool validBuffer(void* object, std::uint64_t requiredBytes) const noexcept {
        if (object == nullptr) return false;
        __unsafe_unretained id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)object;
        return buffer.device.registryID == device_.registryID && buffer.length >= requiredBytes;
    }
    id<MTLComputePipelineState> pipeline(id<MTLLibrary> library, NSString* name) {
        NSError* error = nil;
        id<MTLFunction> function = [library newFunctionWithName:name];
        require(function != nil, "Human Brain receptor function is absent from its owning library");
        id<MTLComputePipelineState> result = [device_ newComputePipelineStateWithFunction:function error:&error];
        require(result != nil, error == nil ? "Human Brain receptor pipeline creation failed"
            : std::string(error.localizedDescription.UTF8String));
        return result;
    }
    static NSString* shaderSource() {
        return @R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void human_brain_receptor_gate(
    device const float4* states [[buffer(0)]],
    device const uint* resultWords [[buffer(1)]],
    device const uint* status [[buffer(2)]],
    device uint* gate [[buffer(3)]],
    constant uint4& shape [[buffer(4)]],
    constant uint4& offsets [[buffer(5)]]) {
    gate[0] = 0u;
    if (status[0] != 0u || status[1] != 0u || status[offsets.z] != shape.y + 1u) return;
    for (uint muscle = 0u; muscle < shape.x; ++muscle) {
        device const uint* row = resultWords + muscle * shape.z;
        device const float* values = reinterpret_cast<device const float*>(row);
        if (row[0] != 0u || row[1] != 0u || row[2] != muscle || !all(isfinite(states[muscle]))) return;
        for (uint component = 0u; component < 4u; ++component) {
            if (!isfinite(values[offsets.x + component]) ||
                !isfinite(values[shape.w + component]) ||
                !isfinite(values[offsets.y + component])) return;
        }
    }
    gate[0] = 1u;
}

kernel void human_brain_body_touch(
    device const uint* gate [[buffer(0)]],
    device const float4* bodyPoses [[buffer(1)]],
    device const uint4* contactWords [[buffer(2)]],
    device const float* vectors [[buffer(3)]],
    device const float* jacobians [[buffer(4)]],
    device const uint4* mappings [[buffer(5)]],
    device float* touch [[buffer(6)]], device uint* touchValidity [[buffer(7)]],
    device float* vestibular [[buffer(8)]], device uint* vestibularValidity [[buffer(9)]],
    device float* interoception [[buffer(10)]], device uint* interoceptionValidity [[buffer(11)]],
    constant uint4& shape [[buffer(12)]], constant uint4& offsets [[buffer(13)]],
    constant float4& normalAndTimestep [[buffer(14)]],
    device const float* acceptedVelocity [[buffer(15)]],
    device const float* acceptedPosition [[buffer(16)]],
    device const uint* qIndexByV [[buffer(17)]],
    device float* kinesthesia [[buffer(18)]],
    device uint* kinesthesiaValidity [[buffer(19)]],
    uint index [[thread_position_in_grid]]) {
    // HumanIO's interoception writer emits workload proxies. They are not
    // measured oxygen, fatigue or tissue damage in this physical path.
    if (index < 416u) {
        for (uint feature = 0u; feature < 6u; ++feature) interoception[index*6u+feature] = 0.0f;
        interoceptionValidity[index] = 0u;
    }
    if (index < 128u) {
        for (uint feature = 0u; feature < 7u; ++feature)
            kinesthesia[index * 7u + feature] = 0.0f;
        kinesthesiaValidity[index] = 0u;
        if (index >= 6u && offsets.z != 0u && gate[0] != 0u) {
            const uint qIndex = qIndexByV[index];
            if (qIndex >= 7u && qIndex < 129u) {
                const float position = acceptedPosition[qIndex];
                const float velocity = acceptedVelocity[index];
                if (isfinite(position) && isfinite(velocity)) {
                    kinesthesia[index * 7u] = position;
                    kinesthesia[index * 7u + 1u] = velocity;
                    kinesthesiaValidity[index] = 0x00000003u;
                }
            }
        }
    }
    if (index == 0u) {
        for (uint feature = 0u; feature < 22u; ++feature) vestibular[feature] = 0.0f;
        vestibularValidity[0] = 0u;
        const float4 headOrientation = bodyPoses[2u * shape.x + 1u];
        const float norm = dot(headOrientation, headOrientation);
        if (gate[0] != 0u && all(isfinite(headOrientation)) && abs(norm - 1.0f) < 1.0e-3f) {
            // Native head-to-world quaternion xyzw from pre-dynamics
            // kinematics. Root coordinates come from the native post-dynamics
            // candidate and are published only with the accepted transaction.
            for (uint component = 0u; component < 4u; ++component)
                vestibular[16u + component] = headOrientation[component];
            vestibularValidity[0] = 0x000f0000u;
        }
        const float3 rootPosition = float3(
            acceptedPosition[0u], acceptedPosition[1u], acceptedPosition[2u]);
        if (gate[0] != 0u && all(isfinite(rootPosition))) {
            for (uint component = 0u; component < 3u; ++component)
                vestibular[component] = rootPosition[component];
            vestibularValidity[0] |= 0x00000007u;
        }
        const float3 rootVelocity = float3(
            acceptedVelocity[0u], acceptedVelocity[1u], acceptedVelocity[2u]);
        if (gate[0] != 0u && all(isfinite(rootVelocity))) {
            for (uint component = 0u; component < 3u; ++component)
                vestibular[7u + component] = rootVelocity[component];
            vestibularValidity[0] |= 0x00000380u;
        }
    }
    if (index >= 10u) return;
    for (uint feature = 0u; feature < 7u; ++feature) touch[index*7u+feature] = 0.0f;
    touchValidity[index] = 0u;
    if (gate[0] == 0u || !all(isfinite(normalAndTimestep)) || normalAndTimestep.w <= 0.0f) return;
    const float3 normal = normalAndTimestep.xyz;
    if (abs(dot(normal, normal) - 1.0f) > 1.0e-3f) return;
    float normalImpulse = 0.0f, tangentImpulse = 0.0f, weightedSlip = 0.0f, unloadedSlip = 0.0f;
    uint matched = 0u;
    for (uint mappingIndex = 0u; mappingIndex < shape.y; ++mappingIndex) {
        const uint4 mapping = mappings[mappingIndex];
        if (mapping.y != index) continue;
        // Native contact identity is body, query index, source geometry, zero.
        const uint4 source = contactWords[2u * mapping.x];
        if (source.x != mapping.z || source.z != mapping.w || source.w != 0u || source.y >= shape.w) return;
        const uint base = offsets.x + 3u * mapping.x;
        const float3 impulses(vectors[base], vectors[base+1u], vectors[base+2u]);
        if (!all(isfinite(impulses)) || impulses.x < -1.0e-7f) return;
        float3 velocity = 0.0f;
        for (uint dof = 0u; dof < shape.z; ++dof) {
            const float speed = vectors[offsets.y + dof];
            velocity += float3(jacobians[(3u*source.y+0u)*shape.z+dof],
                               jacobians[(3u*source.y+1u)*shape.z+dof],
                               jacobians[(3u*source.y+2u)*shape.z+dof]) * speed;
        }
        if (!all(isfinite(velocity))) return;
        const float slip = length(velocity - dot(velocity, normal) * normal);
        const float normalLoad = max(impulses.x, 0.0f);
        if (matched == 0u) unloadedSlip = slip;
        weightedSlip += normalLoad * slip;
        normalImpulse += normalLoad;
        tangentImpulse += length(impulses.yz);
        ++matched;
    }
    if (matched == 0u) return;
    const float3 measurements(normalImpulse / normalAndTimestep.w,
        tangentImpulse / normalAndTimestep.w,
        normalImpulse > 0.0f ? weightedSlip / normalImpulse : unloadedSlip);
    if (!all(isfinite(measurements))) return;
    touch[index*7u+4u] = measurements.x;
    touch[index*7u+5u] = measurements.y;
    touch[index*7u+6u] = measurements.z;
    touchValidity[index] = 0x00000070u;
}
)METAL";
    }
};

static_assert(sizeof(ContactBinding) == 16u);
static_assert(sizeof(MRArticulatedBodyPoseGPU) == 2u * sizeof(mr_float4));
static_assert(offsetof(MRArticulatedBodyPoseGPU, orientation) == sizeof(mr_float4));
static_assert(sizeof(MRNumiHumanStandContactGPU) == 2u * sizeof(mr_uint4));
static_assert(offsetof(MRNumiHumanStandStatusGPU, code) == 0u);
static_assert(offsetof(MRNumiHumanStandStatusGPU, environment) == sizeof(std::uint32_t));

} // namespace numi_human_brain
