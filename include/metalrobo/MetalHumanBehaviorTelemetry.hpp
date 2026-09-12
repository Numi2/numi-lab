#pragma once
#include "metalrobo/HumanBehaviorProgram.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/numanx_human_matter_gpu.h"
#include <memory>
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
    [[nodiscard]] bool encodeCandidate(const MetalNumanXHumanMatterPass& pass, std::uint64_t physicsGeneration, std::uint64_t acceptedTimestampNanoseconds, std::string& error) noexcept;
    [[nodiscard]] bool encodeFlush(void* commandBuffer, std::string& error) noexcept;
    // Internal terminal callback only. disposition1 means actual successful
    // joint release; disposition2 means an actually completed rejected attempt.
    // For acceptance a copied exact COMMITTED fence is mandatory.
    [[nodiscard]] bool terminal(const MRHumanBehaviorReleaseGPU& release,
        const MRNumanXHumanMatterJointPublicationFenceGPU* fence, std::string& error) noexcept;
    [[nodiscard]] HumanBehaviorTelemetrySnapshot snapshot() const;
    [[nodiscard]] bool restore(const HumanBehaviorTelemetrySnapshot&, std::string& error) noexcept;
    void reset();
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;
    [[nodiscard]] std::uint64_t completedAttempts() const noexcept;
private:
    struct State; std::unique_ptr<State> state_;
};
} // namespace metalrobo
