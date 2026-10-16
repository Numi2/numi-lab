#pragma once

#include "metalrobo/HyperPolicyProgram.hpp"
#include "metalrobo/MetalWorld.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace metalrobo {

struct MetalHyperPolicyConfiguration {
    std::string metallibPath;
    std::uint32_t forwardSearchFrames = 12u;
};

enum class MetalHyperPolicyStatus : std::uint32_t {
    success = 0u,
    invalidProgram,
    incompatibleTask,
    metallibUnavailable,
    metalUnavailable,
    pipelineFailure,
    bufferFailure,
    internalFailure,
};

struct MetalHyperPolicyDiagnostics {
    MetalHyperPolicyStatus status = MetalHyperPolicyStatus::success;
    std::string message;
    std::string deviceName;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == MetalHyperPolicyStatus::success;
    }
};

// Owns the device-resident execution state for one compiled motion policy.
// ARDY and the MLX hypernetwork are absent at execution time. The callback
// consumes accepted q/v, native actor observations, and solved support loads,
// then writes the task action row before the actuator program is applied.
class MetalHyperPolicyRuntime {
public:
    MetalHyperPolicyRuntime() = default;
    ~MetalHyperPolicyRuntime();
    MetalHyperPolicyRuntime(MetalHyperPolicyRuntime&&) noexcept;
    MetalHyperPolicyRuntime& operator=(MetalHyperPolicyRuntime&&) noexcept;
    MetalHyperPolicyRuntime(const MetalHyperPolicyRuntime&) = delete;
    MetalHyperPolicyRuntime& operator=(const MetalHyperPolicyRuntime&) = delete;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] const std::string& deviceName() const noexcept;
    [[nodiscard]] MetalWorldDeviceActionProgram actionProgram() noexcept;
    // Drop only execution state. Immutable compiled tables and pipelines are
    // retained; the next submission recreates zeroed phase/action buffers.
    void reset() noexcept;

    // Called only after the owning MetalWorld submission has completed. The
    // returned arrays are step-major and are the exact phase/action values
    // used by the GPU policy, suitable for PolicyRolloutPack publication.
    [[nodiscard]] bool copyRolloutTrace(
        std::uint32_t controlStepCount,
        std::uint32_t environmentCount,
        std::vector<float>& phases,
        std::vector<float>& teacherActions,
        std::vector<float>& latents,
        std::vector<float>& logProbabilities
    );

private:
    struct State;
    std::shared_ptr<State> state_;

    static bool encodeCallback(
        void* context,
        const MetalWorldDeviceActionPass& pass
    ) noexcept;

    friend MetalHyperPolicyDiagnostics createMetalHyperPolicyRuntime(
        const CompiledHyperPolicyProgram&,
        const CompiledTaskProgram&,
        const MetalHyperPolicyConfiguration&,
        MetalHyperPolicyRuntime&
    );
};

[[nodiscard]] MetalHyperPolicyDiagnostics createMetalHyperPolicyRuntime(
    const CompiledHyperPolicyProgram& program,
    const CompiledTaskProgram& task,
    const MetalHyperPolicyConfiguration& configuration,
    MetalHyperPolicyRuntime& output
);

[[nodiscard]] const char* metalHyperPolicyStatusName(
    MetalHyperPolicyStatus status
) noexcept;

} // namespace metalrobo
