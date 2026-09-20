#pragma once
#include "metalrobo/HumanBehaviorProgram.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/mrnx_human_behavior_v1.h"
#include "metalrobo/numanx_human_matter_gpu.h"
#include <array>
#include <memory>
#include <span>
#include <string>
#include <vector>
namespace metalrobo {
struct HumanBehaviorTelemetrySnapshot {
    std::uint64_t programFingerprint=0;
    std::vector<MRHumanBehaviorCandidateGPU> candidate;
    std::vector<MRHumanBehaviorReleaseGPU> release;
    std::vector<MRNumanXHumanMatterJointPublicationFenceGPU> fences;
    std::vector<MRHumanBehaviorReductionGPU> reduction;
};
struct HumanBehaviorTraceBinding {
    std::array<std::uint8_t,32> metricProgramSHA256{};
    std::uint64_t modelSourceFingerprint=0;
    std::uint64_t acceptedStateProofProgramFingerprint=0;
    std::uint32_t clockDomain=0;
    std::uint32_t clockQuantumNanoseconds=0;
};
struct HumanBehaviorTraceAttemptContext {
    std::uint32_t abiVersion=MR_HUMAN_BEHAVIOR_TRACE_ABI_VERSION;
    std::uint32_t structSize=sizeof(HumanBehaviorTraceAttemptContext);
    std::uint32_t controlStep=0;
    std::uint32_t runtimeFailureStage=0;
    std::uint32_t auditCoveredMask=0;
    std::uint32_t auditViolationMask=0;
    std::uint32_t forbiddenContactCoverage=0;
    std::uint32_t forbiddenContactCount=0;
    std::uint32_t reserved0=0;
    std::uint32_t reserved1=0;
    std::uint32_t reserved2=0;
    std::uint64_t basePublicationEpoch=0;
    std::uint64_t basePhysicsGeneration=0;
    std::uint64_t baseAcceptedTimestampNanoseconds=0;
    std::uint64_t baseAcceptedTokenFingerprint=0;
    std::uint64_t basePublicationFingerprint=0;
    std::uint64_t candidateStateProofFingerprint=0;
    std::uint64_t candidateAcceptedTokenFingerprint=0;
    std::uint64_t candidatePublicationFingerprint=0;
    std::uint64_t afterPublicationEpoch=0;
    std::uint64_t afterPhysicsGeneration=0;
    std::uint64_t afterAcceptedTimestampNanoseconds=0;
    std::uint64_t afterAcceptedTokenFingerprint=0;
    std::uint64_t afterPublicationFingerprint=0;
};
struct HumanBehaviorTraceFinalContext {
    std::uint32_t abiVersion=MR_HUMAN_BEHAVIOR_TRACE_ABI_VERSION;
    std::uint32_t structSize=sizeof(HumanBehaviorTraceFinalContext);
    bool quiescent=false;
    bool terminalQuarantine=false;
    std::uint64_t publicationEpoch=0;
    std::uint64_t physicsGeneration=0;
    std::uint64_t brainGeneration=0;
    std::uint64_t sensorGeneration=0;
    std::uint64_t timestampNanoseconds=0;
    std::uint64_t acceptedTokenFingerprint=0;
    std::uint64_t publicationFingerprint=0;
};
// Owns only telemetry buffers/pipelines. Never creates a command queue, submits
// physics, computes host physical metrics, or retains a borrowed physical arena.
// The enclosing runtime serializes calls and invokes snapshot/restore/reset only
// at quiescent accepted-root boundaries together with its physical/Brain state.
class MetalHumanBehaviorTelemetry {
public:
    MetalHumanBehaviorTelemetry(void* device, const CompiledHumanBehaviorProgram& program,
        const std::string& metallibPath, std::uint32_t environmentCount,
        std::uint64_t initialPhysicsGeneration, std::uint64_t initialTimestampNanoseconds);
    ~MetalHumanBehaviorTelemetry();
    MetalHumanBehaviorTelemetry(const MetalHumanBehaviorTelemetry&)=delete;
    MetalHumanBehaviorTelemetry& operator=(const MetalHumanBehaviorTelemetry&)=delete;
    // Measure the exact accepted reset state in the existing owner
    // pre-dynamics pass, after kinematics exist and before dynamics advance.
    // This records posture/settled observations only; it does not authorize a
    // publication.
    [[nodiscard]] bool encodeInitial(const MetalNumanXHumanMatterPass& pass,
        std::string& error) noexcept;
    [[nodiscard]] bool encodeCandidate(const MetalNumanXHumanMatterPass& pass, std::uint64_t physicsGeneration, std::uint64_t acceptedTimestampNanoseconds, std::string& error) noexcept;
    [[nodiscard]] bool encodeFlush(void* commandBuffer, std::string& error) noexcept;
    // Internal terminal callback only. disposition1 means actual successful
    // joint release; disposition2 means an actually completed rejected attempt.
    // For acceptance a copied exact COMMITTED fence is mandatory.
    [[nodiscard]] bool terminal(const MRHumanBehaviorReleaseGPU& release,
        const MRNumanXHumanMatterJointPublicationFenceGPU* fence, std::string& error) noexcept;
    // Optional trace context is copied from the already-authoritative runtime
    // lifecycle. Missing/malformed trace context is recorded as an incomplete
    // sidecar and can never make a valid behavior terminal fail.
    [[nodiscard]] bool terminal(const MRHumanBehaviorReleaseGPU& release,
        const MRNumanXHumanMatterJointPublicationFenceGPU* fence,
        const HumanBehaviorTraceAttemptContext* trace,
        std::string& error) noexcept;
    [[nodiscard]] bool traceAttach(const mrnx_behavior_trace_config_v1& config,
        const HumanBehaviorTraceBinding& binding, std::string& error) noexcept;
    [[nodiscard]] bool traceDrain(mrnx_behavior_trace_chunk_v1& chunk,
        std::span<mrnx_behavior_trace_record_v1> records,
        std::string& error) noexcept;
    [[nodiscard]] bool traceFinalize(
        const mrnx_behavior_trace_terminal_request_v1& request,
        const HumanBehaviorTraceFinalContext& finalContext,
        mrnx_behavior_trace_terminal_v1& terminal,
        std::string& error) noexcept;
    [[nodiscard]] bool traceAttached() const noexcept;
    [[nodiscard]] bool traceFinalized() const noexcept;
    // Read-only owner-buffer copy used at the already-settled terminal
    // boundary. It does not flush, submit work, or advance telemetry state.
    [[nodiscard]] bool copyTerminalCandidate(
        MRHumanBehaviorCandidateGPU& candidate) const noexcept;
    [[nodiscard]] HumanBehaviorTelemetrySnapshot snapshot() const;
    [[nodiscard]] bool restore(const HumanBehaviorTelemetrySnapshot&, std::string& error) noexcept;
    void reset();
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;
    [[nodiscard]] std::uint64_t completedAttempts() const noexcept;
    [[nodiscard]] bool initialObservationEncoded() const noexcept;
private:
    struct State; std::unique_ptr<State> state_;
};
} // namespace metalrobo
