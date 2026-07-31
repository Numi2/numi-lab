#pragma once

#include "metalrobo/policy_program_types.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

class CompiledTaskProgram;

enum class PolicyActivation : std::uint32_t {
    identity = MR_POLICY_ACTIVATION_IDENTITY,
    relu = MR_POLICY_ACTIVATION_RELU,
    tanh = MR_POLICY_ACTIVATION_TANH,
    elu = MR_POLICY_ACTIVATION_ELU,
    silu = MR_POLICY_ACTIVATION_SILU,
};

struct PolicyDenseLayer {
    std::uint32_t inputCount = 0u;
    std::uint32_t outputCount = 0u;
    PolicyActivation activation = PolicyActivation::tanh;
    // Output-major [output][input].
    std::vector<float> weights;
    std::vector<float> bias;
};

// Portable learner/deployment boundary. MLX may update this artifact between
// rollouts; physics and inference consume only the compiled immutable tables.
struct PolicyPack {
    std::string id;
    std::uint64_t revision = 1u;
    // Actor normalization and actor dense network.
    std::vector<float> observationMean;
    std::vector<float> observationInverseStandardDeviation;
    std::vector<PolicyDenseLayer> layers;
    // Optional asymmetric critic. A stochastic policy requires this network.
    std::vector<float> criticObservationMean;
    std::vector<float>
        criticObservationInverseStandardDeviation;
    std::vector<PolicyDenseLayer> criticLayers;
    // Empty selects the deterministic actor mean. A full action-width vector
    // selects a diagonal Gaussian in actor coordinates.
    std::vector<float> actionLogStandardDeviation;
    std::vector<float> actionBias;
    std::vector<float> actionScale;
    float observationClip = 100.0f;
    float actionClip = std::numeric_limits<float>::max();
};

enum class PolicyCompileStatus : std::uint32_t {
    success = 0u,
    invalidTask,
    invalidPack,
    incompatibleContract,
    arithmeticOverflow,
    internalFailure,
};

struct PolicyCompileDiagnostics {
    PolicyCompileStatus status = PolicyCompileStatus::success;
    std::uint64_t fingerprint = 0u;
    std::string element;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == PolicyCompileStatus::success;
    }
};

struct PolicyProgramLayout {
    std::uint32_t actorLayerCount = 0u;
    std::uint32_t criticLayerCount = 0u;
    std::uint32_t actorObservationCount = 0u;
    std::uint32_t criticObservationCount = 0u;
    std::uint32_t actionCount = 0u;
    std::uint32_t maximumHiddenCount = 0u;
    bool stochastic = false;
};

class CompiledPolicyProgram {
public:
    CompiledPolicyProgram() = default;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;
    [[nodiscard]] std::uint64_t topologyFingerprint() const noexcept;
    [[nodiscard]] std::uint64_t taskFingerprint() const noexcept;
    [[nodiscard]] std::uint64_t revision() const noexcept;
    [[nodiscard]] const PolicyProgramLayout& layout() const noexcept;
    [[nodiscard]] const MRPolicyProgramHeaderGPU& header() const noexcept;
    [[nodiscard]] std::span<const MRPolicyDenseLayerGPU>
    actorLayers() const noexcept;
    [[nodiscard]] std::span<const MRPolicyDenseLayerGPU>
    criticLayers() const noexcept;
    [[nodiscard]] std::span<const std::byte> arena() const noexcept;

private:
    struct Storage;
    std::shared_ptr<const Storage> storage_;

    friend PolicyCompileDiagnostics compilePolicyProgram(
        const PolicyPack&,
        const CompiledTaskProgram&,
        CompiledPolicyProgram&
    );
};

[[nodiscard]] PolicyCompileDiagnostics compilePolicyProgram(
    const PolicyPack& pack,
    const CompiledTaskProgram& task,
    CompiledPolicyProgram& output
);

[[nodiscard]] const char* policyCompileStatusName(
    PolicyCompileStatus status
) noexcept;

} // namespace metalrobo
