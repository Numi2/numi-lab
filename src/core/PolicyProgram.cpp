#include "metalrobo/PolicyProgram.hpp"

#include "metalrobo/TaskProgram.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace metalrobo {

struct CompiledPolicyProgram::Storage {
    std::uint64_t fingerprint = 0u;
    std::uint64_t taskFingerprint = 0u;
    std::uint64_t revision = 0u;
    PolicyProgramLayout layout{};
    MRPolicyProgramHeaderGPU header{};
    std::vector<MRPolicyDenseLayerGPU> layers;
    std::vector<std::byte> arena;
};

namespace {

constexpr std::uint64_t kFNVOffset = 1469598103934665603ull;
constexpr std::uint64_t kFNVPrime = 1099511628211ull;

class Hash {
public:
    void bytes(const void* data, const std::size_t size) {
        const auto* values =
            static_cast<const unsigned char*>(data);
        for (std::size_t index = 0u; index < size; ++index) {
            value_ ^= values[index];
            value_ *= kFNVPrime;
        }
    }

    template <typename T>
    void scalar(const T& value) {
        bytes(&value, sizeof(value));
    }

    void string(const std::string_view value) {
        scalar<std::uint64_t>(value.size());
        bytes(value.data(), value.size());
    }

    [[nodiscard]] std::uint64_t finish() const noexcept {
        return value_ == 0u ? 1u : value_;
    }

private:
    std::uint64_t value_ = kFNVOffset;
};

PolicyCompileDiagnostics reject(
    const PolicyCompileStatus status,
    std::string element,
    std::string message
) {
    return {
        .status = status,
        .element = std::move(element),
        .message = std::move(message),
    };
}

bool supportedActivation(const PolicyActivation activation) {
    switch (activation) {
    case PolicyActivation::identity:
    case PolicyActivation::relu:
    case PolicyActivation::tanh:
    case PolicyActivation::elu:
    case PolicyActivation::silu:
        return true;
    }
    return false;
}

bool finite(const std::span<const float> values) {
    return std::all_of(
        values.begin(),
        values.end(),
        [](const float value) {
            return std::isfinite(value);
        }
    );
}

std::uint32_t appendArena(
    std::vector<std::byte>& arena,
    const void* values,
    const std::size_t bytes
) {
    constexpr std::size_t alignment = 16u;
    const std::size_t aligned =
        (arena.size() + alignment - 1u) /
        alignment * alignment;
    if (aligned >
            std::numeric_limits<std::uint32_t>::max() ||
        bytes >
            std::numeric_limits<std::uint32_t>::max() -
                aligned) {
        return MR_INVALID_INDEX;
    }
    arena.resize(aligned + bytes, std::byte{0});
    if (bytes != 0u && values != nullptr) {
        std::memcpy(arena.data() + aligned, values, bytes);
    }
    return static_cast<std::uint32_t>(aligned);
}

template <typename T>
std::uint32_t appendArena(
    std::vector<std::byte>& arena,
    const std::span<const T> values
) {
    return appendArena(
        arena,
        values.data(),
        values.size_bytes()
    );
}

} // namespace

bool CompiledPolicyProgram::valid() const noexcept {
    return storage_ != nullptr &&
        storage_->fingerprint != 0u &&
        storage_->taskFingerprint != 0u &&
        storage_->revision != 0u &&
        storage_->header.policyFingerprint ==
            storage_->fingerprint &&
        storage_->header.taskFingerprint ==
            storage_->taskFingerprint &&
        storage_->header.revision == storage_->revision &&
        storage_->header.abi.x ==
            MR_POLICY_PROGRAM_ABI_VERSION &&
        storage_->layout.layerCount != 0u &&
        storage_->layout.observationCount != 0u &&
        storage_->layout.actionCount != 0u;
}

std::uint64_t CompiledPolicyProgram::fingerprint() const noexcept {
    return valid() ? storage_->fingerprint : 0u;
}

std::uint64_t CompiledPolicyProgram::taskFingerprint() const noexcept {
    return valid() ? storage_->taskFingerprint : 0u;
}

std::uint64_t CompiledPolicyProgram::revision() const noexcept {
    return valid() ? storage_->revision : 0u;
}

const PolicyProgramLayout&
CompiledPolicyProgram::layout() const noexcept {
    static const PolicyProgramLayout empty{};
    return valid() ? storage_->layout : empty;
}

const MRPolicyProgramHeaderGPU&
CompiledPolicyProgram::header() const noexcept {
    static const MRPolicyProgramHeaderGPU empty{};
    return valid() ? storage_->header : empty;
}

std::span<const MRPolicyDenseLayerGPU>
CompiledPolicyProgram::layers() const noexcept {
    return valid()
        ? std::span<const MRPolicyDenseLayerGPU>{
              storage_->layers
          }
        : std::span<const MRPolicyDenseLayerGPU>{};
}

std::span<const std::byte>
CompiledPolicyProgram::arena() const noexcept {
    return valid()
        ? std::span<const std::byte>{storage_->arena}
        : std::span<const std::byte>{};
}

PolicyCompileDiagnostics compilePolicyProgram(
    const PolicyPack& pack,
    const CompiledTaskProgram& task,
    CompiledPolicyProgram& output
) {
    if (!task.valid()) {
        return reject(
            PolicyCompileStatus::invalidTask,
            "task",
            "policy compilation requires a valid compiled task"
        );
    }
    const TaskProgramLayout& taskLayout = task.layout();
    if (pack.id.empty() || pack.revision == 0u ||
        pack.layers.empty() ||
        pack.layers.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        !std::isfinite(pack.observationClip) ||
        !(pack.observationClip > 0.0f) ||
        !std::isfinite(pack.actionClip) ||
        !(pack.actionClip > 0.0f)) {
        return reject(
            PolicyCompileStatus::invalidPack,
            "policy",
            "policy identity, revision, layers, or clipping is invalid"
        );
    }
    if ((!pack.observationMean.empty() &&
         pack.observationMean.size() !=
             taskLayout.actorObservationSize) ||
        (!pack.observationInverseStandardDeviation.empty() &&
         pack.observationInverseStandardDeviation.size() !=
             taskLayout.actorObservationSize) ||
        (!pack.actionBias.empty() &&
         pack.actionBias.size() != taskLayout.actionCount) ||
        (!pack.actionScale.empty() &&
         pack.actionScale.size() != taskLayout.actionCount) ||
        !finite(pack.observationMean) ||
        !finite(pack.observationInverseStandardDeviation) ||
        !finite(pack.actionBias) ||
        !finite(pack.actionScale)) {
        return reject(
            PolicyCompileStatus::incompatibleContract,
            "normalization",
            "normalization and action transforms must match the task contract"
        );
    }

    auto staged = std::make_shared<CompiledPolicyProgram::Storage>();
    staged->taskFingerprint = task.fingerprint();
    staged->revision = pack.revision;
    staged->layout.layerCount =
        static_cast<std::uint32_t>(pack.layers.size());
    staged->layout.observationCount =
        taskLayout.actorObservationSize;
    staged->layout.actionCount = taskLayout.actionCount;

    std::uint32_t expectedInput =
        taskLayout.actorObservationSize;
    staged->layers.resize(pack.layers.size());
    for (std::size_t index = 0u;
         index < pack.layers.size();
         ++index) {
        const PolicyDenseLayer& layer = pack.layers[index];
        const std::uint64_t weightCount =
            static_cast<std::uint64_t>(layer.inputCount) *
            layer.outputCount;
        if (layer.inputCount != expectedInput ||
            layer.outputCount == 0u ||
            weightCount != layer.weights.size() ||
            layer.bias.size() != layer.outputCount ||
            !supportedActivation(layer.activation) ||
            !finite(layer.weights) ||
            !finite(layer.bias)) {
            return reject(
                PolicyCompileStatus::incompatibleContract,
                "layers[" + std::to_string(index) + "]",
                "dense layer shape, activation, or parameters are invalid"
            );
        }
        if (index + 1u < pack.layers.size()) {
            staged->layout.maximumHiddenCount = std::max(
                staged->layout.maximumHiddenCount,
                layer.outputCount
            );
        } else if (layer.outputCount !=
                   taskLayout.actionCount) {
            return reject(
                PolicyCompileStatus::incompatibleContract,
                "layers[" + std::to_string(index) + "]",
                "final dense layer width does not match task actions"
            );
        }
        expectedInput = layer.outputCount;
    }

    std::vector<float> observationMean =
        pack.observationMean;
    if (observationMean.empty()) {
        observationMean.assign(
            taskLayout.actorObservationSize,
            0.0f
        );
    }
    std::vector<float> observationInverseStd =
        pack.observationInverseStandardDeviation;
    if (observationInverseStd.empty()) {
        observationInverseStd.assign(
            taskLayout.actorObservationSize,
            1.0f
        );
    }
    if (std::any_of(
            observationInverseStd.begin(),
            observationInverseStd.end(),
            [](const float value) {
                return !(value > 0.0f);
            }
        )) {
        return reject(
            PolicyCompileStatus::invalidPack,
            "observationInverseStandardDeviation",
            "inverse standard deviations must be finite and positive"
        );
    }
    std::vector<float> actionBias = pack.actionBias;
    if (actionBias.empty()) {
        actionBias.assign(taskLayout.actionCount, 0.0f);
    }
    std::vector<float> actionScale = pack.actionScale;
    if (actionScale.empty()) {
        actionScale.assign(taskLayout.actionCount, 1.0f);
    }

    // Prove a finite bound for every dense accumulator from the authored
    // observation clip. The hot kernel can then stay branch-free and does
    // not need to mask a malformed policy after the fact.
    constexpr double kAccumulatorLimit =
        static_cast<double>(
            std::numeric_limits<float>::max()
        ) / 8.0;
    std::vector<double> inputBounds(
        taskLayout.actorObservationSize,
        pack.observationClip
    );
    for (std::size_t layerIndex = 0u;
         layerIndex < pack.layers.size();
         ++layerIndex) {
        const PolicyDenseLayer& layer =
            pack.layers[layerIndex];
        std::vector<double> outputBounds(
            layer.outputCount,
            0.0
        );
        for (std::uint32_t outputIndex = 0u;
             outputIndex < layer.outputCount;
             ++outputIndex) {
            double bound = std::abs(
                static_cast<double>(
                    layer.bias[outputIndex]
                )
            );
            const std::size_t weightBase =
                static_cast<std::size_t>(outputIndex) *
                layer.inputCount;
            for (std::uint32_t inputIndex = 0u;
                 inputIndex < layer.inputCount;
                 ++inputIndex) {
                bound +=
                    std::abs(static_cast<double>(
                        layer.weights[
                            weightBase + inputIndex
                        ]
                    )) *
                    inputBounds[inputIndex];
                if (!std::isfinite(bound) ||
                    bound > kAccumulatorLimit) {
                    return reject(
                        PolicyCompileStatus::invalidPack,
                        "layers[" +
                            std::to_string(layerIndex) +
                            "]",
                        "dense accumulator can overflow within the authored observation bounds"
                    );
                }
            }
            switch (layer.activation) {
            case PolicyActivation::tanh:
                bound = 1.0;
                break;
            case PolicyActivation::elu:
            case PolicyActivation::silu:
                bound = std::max(bound, 1.0);
                break;
            case PolicyActivation::identity:
            case PolicyActivation::relu:
                break;
            }
            if (layerIndex + 1u == pack.layers.size()) {
                bound =
                    std::abs(static_cast<double>(
                        actionBias[outputIndex]
                    )) +
                    std::abs(static_cast<double>(
                        actionScale[outputIndex]
                    )) *
                    bound;
                if (!std::isfinite(bound) ||
                    bound > kAccumulatorLimit) {
                    return reject(
                        PolicyCompileStatus::invalidPack,
                        "actionTransform",
                        "action transform can overflow before clipping"
                    );
                }
                bound = std::min<double>(
                    bound,
                    pack.actionClip
                );
            }
            outputBounds[outputIndex] = bound;
        }
        inputBounds = std::move(outputBounds);
    }

    const std::size_t layerTableBytes =
        staged->layers.size() *
        sizeof(MRPolicyDenseLayerGPU);
    const std::uint32_t layerTableOffset = appendArena(
        staged->arena,
        nullptr,
        layerTableBytes
    );
    if (layerTableOffset == MR_INVALID_INDEX) {
        return reject(
            PolicyCompileStatus::arithmeticOverflow,
            "arena",
            "policy layer table exceeds the 32-bit byte-offset ABI"
        );
    }
    for (std::size_t index = 0u;
         index < pack.layers.size();
         ++index) {
        const PolicyDenseLayer& authored = pack.layers[index];
        const std::uint32_t weights = appendArena(
            staged->arena,
            std::span<const float>{authored.weights}
        );
        const std::uint32_t bias = appendArena(
            staged->arena,
            std::span<const float>{authored.bias}
        );
        if (weights == MR_INVALID_INDEX ||
            bias == MR_INVALID_INDEX) {
            return reject(
                PolicyCompileStatus::arithmeticOverflow,
                "arena",
                "policy weights exceed the 32-bit byte-offset ABI"
            );
        }
        std::uint32_t flags = 0u;
        if (index == 0u) {
            flags |= MR_POLICY_DENSE_NORMALIZE_INPUT;
        }
        if (index + 1u == pack.layers.size()) {
            flags |=
                MR_POLICY_DENSE_TRANSFORM_OUTPUT |
                MR_POLICY_DENSE_CLAMP_OUTPUT;
        }
        staged->layers[index] = {
            {
                authored.inputCount,
                authored.outputCount,
                static_cast<std::uint32_t>(
                    authored.activation
                ),
                flags,
            },
            {weights, bias, 0u, 0u},
        };
    }
    std::memcpy(
        staged->arena.data() + layerTableOffset,
        staged->layers.data(),
        layerTableBytes
    );
    const std::uint32_t meanOffset = appendArena(
        staged->arena,
        std::span<const float>{observationMean}
    );
    const std::uint32_t inverseStdOffset = appendArena(
        staged->arena,
        std::span<const float>{observationInverseStd}
    );
    const std::uint32_t actionBiasOffset = appendArena(
        staged->arena,
        std::span<const float>{actionBias}
    );
    const std::uint32_t actionScaleOffset = appendArena(
        staged->arena,
        std::span<const float>{actionScale}
    );
    if (meanOffset == MR_INVALID_INDEX ||
        inverseStdOffset == MR_INVALID_INDEX ||
        actionBiasOffset == MR_INVALID_INDEX ||
        actionScaleOffset == MR_INVALID_INDEX) {
        return reject(
            PolicyCompileStatus::arithmeticOverflow,
            "arena",
            "policy transforms exceed the 32-bit byte-offset ABI"
        );
    }

    staged->header.counts = {
        staged->layout.layerCount,
        staged->layout.observationCount,
        staged->layout.actionCount,
        staged->layout.maximumHiddenCount,
    };
    staged->header.offsets0 = {
        layerTableOffset,
        meanOffset,
        inverseStdOffset,
        actionBiasOffset,
    };
    staged->header.offsets1 = {
        actionScaleOffset,
        0u,
        0u,
        0u,
    };
    staged->header.limits = {
        pack.observationClip,
        pack.actionClip,
        0.0f,
        0.0f,
    };
    staged->header.taskFingerprint =
        staged->taskFingerprint;
    staged->header.revision = staged->revision;
    staged->header.abi = {
        MR_POLICY_PROGRAM_ABI_VERSION,
        0u,
        0u,
        0u,
    };

    Hash hash;
    hash.string(pack.id);
    hash.scalar(staged->header);
    hash.bytes(staged->arena.data(), staged->arena.size());
    staged->fingerprint = hash.finish();
    staged->header.policyFingerprint =
        staged->fingerprint;

    output.storage_ = std::move(staged);
    return {
        .status = PolicyCompileStatus::success,
        .fingerprint = output.fingerprint(),
    };
}

const char* policyCompileStatusName(
    const PolicyCompileStatus status
) noexcept {
    switch (status) {
    case PolicyCompileStatus::success:
        return "success";
    case PolicyCompileStatus::invalidTask:
        return "invalid_task";
    case PolicyCompileStatus::invalidPack:
        return "invalid_pack";
    case PolicyCompileStatus::incompatibleContract:
        return "incompatible_contract";
    case PolicyCompileStatus::arithmeticOverflow:
        return "arithmetic_overflow";
    case PolicyCompileStatus::internalFailure:
        return "internal_failure";
    }
    return "unknown";
}

} // namespace metalrobo
