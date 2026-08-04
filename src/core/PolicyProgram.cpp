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
    std::vector<MRPolicyDenseLayerGPU> actorLayers;
    std::vector<MRPolicyDenseLayerGPU> criticLayers;
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
        storage_->layout.actorLayerCount != 0u &&
        storage_->layout.actorObservationCount != 0u &&
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
CompiledPolicyProgram::actorLayers() const noexcept {
    return valid()
        ? std::span<const MRPolicyDenseLayerGPU>{
              storage_->actorLayers
          }
        : std::span<const MRPolicyDenseLayerGPU>{};
}

std::span<const MRPolicyDenseLayerGPU>
CompiledPolicyProgram::criticLayers() const noexcept {
    return valid()
        ? std::span<const MRPolicyDenseLayerGPU>{
              storage_->criticLayers
          }
        : std::span<const MRPolicyDenseLayerGPU>{};
}

std::span<const std::byte>
CompiledPolicyProgram::arena() const noexcept {
    return valid()
        ? std::span<const std::byte>{storage_->arena}
        : std::span<const std::byte>{};
}

void bindPolicyPack(
    PolicyPack& pack,
    const CompiledTaskProgram& task
) {
    if (!task.valid()) {
        pack.contract = {};
        return;
    }
    pack.contract = {
        .version = 1u,
        .worldFingerprint = task.worldFingerprint(),
        .taskFingerprint = task.fingerprint(),
        .observationFingerprint =
            task.observationFingerprint(),
        .actionFingerprint = task.actionFingerprint(),
    };
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
    const bool hasCritic = !pack.criticLayers.empty();
    const bool stochastic =
        !pack.actionLogStandardDeviation.empty();
    if (pack.id.empty() || pack.revision == 0u ||
        pack.layers.empty() ||
        pack.layers.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        pack.criticLayers.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        (stochastic && !hasCritic) ||
        !std::isfinite(pack.observationClip) ||
        !(pack.observationClip > 0.0f) ||
        !std::isfinite(pack.actionClip) ||
        !(pack.actionClip > 0.0f)) {
        return reject(
            PolicyCompileStatus::invalidPack,
            "policy",
            "policy identity, revision, networks, distribution, or clipping is invalid"
        );
    }
    if (!pack.contract.exact()) {
        return reject(
            PolicyCompileStatus::invalidPack,
            "contract",
            "policy contract must be a complete version-1 binding"
        );
    }
    if (pack.contract.worldFingerprint !=
             task.worldFingerprint() ||
         pack.contract.taskFingerprint != task.fingerprint() ||
         pack.contract.observationFingerprint !=
             task.observationFingerprint() ||
         pack.contract.actionFingerprint !=
             task.actionFingerprint()) {
        return reject(
            PolicyCompileStatus::incompatibleContract,
            "contract",
            "policy is bound to different world, task, observation, or action semantics"
        );
    }
    if ((!pack.observationMean.empty() &&
         pack.observationMean.size() !=
             taskLayout.actorObservationSize) ||
        (!pack.observationInverseStandardDeviation.empty() &&
         pack.observationInverseStandardDeviation.size() !=
             taskLayout.actorObservationSize) ||
        (!pack.criticObservationMean.empty() &&
         (!hasCritic ||
          pack.criticObservationMean.size() !=
              taskLayout.criticObservationSize)) ||
        (!pack.criticObservationInverseStandardDeviation.empty() &&
         (!hasCritic ||
          pack.criticObservationInverseStandardDeviation.size() !=
              taskLayout.criticObservationSize)) ||
        (stochastic &&
         pack.actionLogStandardDeviation.size() !=
             taskLayout.actionCount) ||
        (!pack.actionBias.empty() &&
         pack.actionBias.size() != taskLayout.actionCount) ||
        (!pack.actionScale.empty() &&
         pack.actionScale.size() != taskLayout.actionCount) ||
        !finite(pack.observationMean) ||
        !finite(pack.observationInverseStandardDeviation) ||
        !finite(pack.criticObservationMean) ||
        !finite(
            pack.criticObservationInverseStandardDeviation
        ) ||
        !finite(pack.actionLogStandardDeviation) ||
        !finite(pack.actionBias) ||
        !finite(pack.actionScale)) {
        return reject(
            PolicyCompileStatus::incompatibleContract,
            "normalization",
            "actor, critic, distribution, and action transforms must match the task contract"
        );
    }
    if (stochastic && std::any_of(
            pack.actionLogStandardDeviation.begin(),
            pack.actionLogStandardDeviation.end(),
            [](const float value) {
                return value < -5.0f || value > 2.0f;
            }
        )) {
        return reject(
            PolicyCompileStatus::invalidPack,
            "actionLogStandardDeviation",
            "Gaussian log standard deviations must be within [-5, 2]"
        );
    }

    auto staged = std::make_shared<CompiledPolicyProgram::Storage>();
    staged->taskFingerprint = task.fingerprint();
    staged->revision = pack.revision;
    staged->layout.actorLayerCount =
        static_cast<std::uint32_t>(pack.layers.size());
    staged->layout.criticLayerCount =
        static_cast<std::uint32_t>(
            pack.criticLayers.size()
        );
    staged->layout.actorObservationCount =
        taskLayout.actorObservationSize;
    staged->layout.criticObservationCount =
        hasCritic ? taskLayout.criticObservationSize : 0u;
    staged->layout.actionCount = taskLayout.actionCount;
    staged->layout.stochastic = stochastic;

    const auto validateNetwork = [&](
        const std::span<const PolicyDenseLayer> layers,
        const std::uint32_t inputCount,
        const std::uint32_t outputCount,
        const std::string_view name
    ) -> PolicyCompileDiagnostics {
        std::uint32_t expectedInput = inputCount;
        for (std::size_t index = 0u;
             index < layers.size();
             ++index) {
            const PolicyDenseLayer& layer = layers[index];
            const std::uint64_t weightCount =
                static_cast<std::uint64_t>(
                    layer.inputCount
                ) * layer.outputCount;
            if (layer.inputCount != expectedInput ||
                layer.outputCount == 0u ||
                weightCount != layer.weights.size() ||
                layer.bias.size() != layer.outputCount ||
                !supportedActivation(layer.activation) ||
                !finite(layer.weights) ||
                !finite(layer.bias)) {
                return reject(
                    PolicyCompileStatus::
                        incompatibleContract,
                    std::string{name} + "Layers[" +
                        std::to_string(index) + "]",
                    "dense layer shape, activation, or parameters are invalid"
                );
            }
            const bool final =
                index + 1u == layers.size();
            if (!final) {
                staged->layout.maximumHiddenCount =
                    std::max(
                        staged->layout.maximumHiddenCount,
                        layer.outputCount
                    );
            } else if (layer.outputCount != outputCount ||
                       layer.activation !=
                           PolicyActivation::identity) {
                return reject(
                    PolicyCompileStatus::
                        incompatibleContract,
                    std::string{name} + "Layers[" +
                        std::to_string(index) + "]",
                    "final dense layer must use identity activation and match the compiled contract"
                );
            }
            expectedInput = layer.outputCount;
        }
        return {};
    };
    PolicyCompileDiagnostics networkStatus =
        validateNetwork(
            pack.layers,
            taskLayout.actorObservationSize,
            taskLayout.actionCount,
            "actor"
        );
    if (!networkStatus.succeeded()) {
        return networkStatus;
    }
    if (hasCritic) {
        networkStatus = validateNetwork(
            pack.criticLayers,
            taskLayout.criticObservationSize,
            1u,
            "critic"
        );
        if (!networkStatus.succeeded()) {
            return networkStatus;
        }
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
    std::vector<float> criticObservationMean =
        pack.criticObservationMean;
    std::vector<float> criticObservationInverseStd =
        pack.criticObservationInverseStandardDeviation;
    if (hasCritic) {
        if (criticObservationMean.empty()) {
            criticObservationMean.assign(
                taskLayout.criticObservationSize,
                0.0f
            );
        }
        if (criticObservationInverseStd.empty()) {
            criticObservationInverseStd.assign(
                taskLayout.criticObservationSize,
                1.0f
            );
        }
        if (std::any_of(
                criticObservationInverseStd.begin(),
                criticObservationInverseStd.end(),
                [](const float value) {
                    return !(value > 0.0f);
                }
            )) {
            return reject(
                PolicyCompileStatus::invalidPack,
                "criticObservationInverseStandardDeviation",
                "critic inverse standard deviations must be finite and positive"
            );
        }
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
    // observation clip. The hot kernels can then stay branch-free.
    constexpr double kAccumulatorLimit =
        static_cast<double>(
            std::numeric_limits<float>::max()
        ) / 8.0;
    const auto proveNetwork = [&](
        const std::span<const PolicyDenseLayer> layers,
        const std::uint32_t inputCount,
        const std::string_view name,
        std::vector<double>* finalBounds
    ) -> PolicyCompileDiagnostics {
        std::vector<double> inputBounds(
            inputCount,
            pack.observationClip
        );
        for (std::size_t layerIndex = 0u;
             layerIndex < layers.size();
             ++layerIndex) {
            const PolicyDenseLayer& layer =
                layers[layerIndex];
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
                    static_cast<std::size_t>(
                        outputIndex
                    ) * layer.inputCount;
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
                            std::string{name} + "Layers[" +
                                std::to_string(
                                    layerIndex
                                ) + "]",
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
                outputBounds[outputIndex] = bound;
            }
            inputBounds = std::move(outputBounds);
        }
        if (finalBounds != nullptr) {
            *finalBounds = std::move(inputBounds);
        }
        return {};
    };
    std::vector<double> actorOutputBounds;
    networkStatus = proveNetwork(
        pack.layers,
        taskLayout.actorObservationSize,
        "actor",
        &actorOutputBounds
    );
    if (!networkStatus.succeeded()) {
        return networkStatus;
    }
    if (hasCritic) {
        networkStatus = proveNetwork(
            pack.criticLayers,
            taskLayout.criticObservationSize,
            "critic",
            nullptr
        );
        if (!networkStatus.succeeded()) {
            return networkStatus;
        }
    }
    for (std::uint32_t action = 0u;
         action < taskLayout.actionCount;
         ++action) {
        const double transformedBound =
            std::abs(static_cast<double>(
                actionBias[action]
            )) +
            std::abs(static_cast<double>(
                actionScale[action]
            )) * actorOutputBounds[action];
        if (!std::isfinite(transformedBound) ||
            transformedBound > kAccumulatorLimit) {
            return reject(
                PolicyCompileStatus::invalidPack,
                "actionTransform",
                "Gaussian action transform can overflow before clipping"
            );
        }
    }

    const auto appendNetwork = [&](
        const std::span<const PolicyDenseLayer> authored,
        std::vector<MRPolicyDenseLayerGPU>& compiled,
        std::uint32_t& tableOffset
    ) -> bool {
        if (authored.empty()) {
            tableOffset = 0u;
            return true;
        }
        compiled.resize(authored.size());
        const std::size_t tableBytes =
            compiled.size() *
            sizeof(MRPolicyDenseLayerGPU);
        tableOffset = appendArena(
            staged->arena,
            nullptr,
            tableBytes
        );
        if (tableOffset == MR_INVALID_INDEX) {
            return false;
        }
        for (std::size_t index = 0u;
             index < authored.size();
             ++index) {
            const PolicyDenseLayer& layer = authored[index];
            const std::uint32_t weights = appendArena(
                staged->arena,
                std::span<const float>{layer.weights}
            );
            const std::uint32_t bias = appendArena(
                staged->arena,
                std::span<const float>{layer.bias}
            );
            if (weights == MR_INVALID_INDEX ||
                bias == MR_INVALID_INDEX) {
                return false;
            }
            compiled[index] = {
                {
                    layer.inputCount,
                    layer.outputCount,
                    static_cast<std::uint32_t>(
                        layer.activation
                    ),
                    index == 0u
                        ? MR_POLICY_DENSE_NORMALIZE_INPUT
                        : 0u,
                },
                {weights, bias, 0u, 0u},
            };
        }
        std::memcpy(
            staged->arena.data() + tableOffset,
            compiled.data(),
            tableBytes
        );
        return true;
    };
    std::uint32_t actorLayerTableOffset = 0u;
    std::uint32_t criticLayerTableOffset = 0u;
    if (!appendNetwork(
            pack.layers,
            staged->actorLayers,
            actorLayerTableOffset
        ) ||
        !appendNetwork(
            pack.criticLayers,
            staged->criticLayers,
            criticLayerTableOffset
        )) {
        return reject(
            PolicyCompileStatus::arithmeticOverflow,
            "arena",
            "policy layer tables or weights exceed the 32-bit byte-offset ABI"
        );
    }
    const std::uint32_t meanOffset = appendArena(
        staged->arena,
        std::span<const float>{observationMean}
    );
    const std::uint32_t inverseStdOffset = appendArena(
        staged->arena,
        std::span<const float>{observationInverseStd}
    );
    std::uint32_t criticMeanOffset = 0u;
    std::uint32_t criticInverseStdOffset = 0u;
    if (hasCritic) {
        criticMeanOffset = appendArena(
            staged->arena,
            std::span<const float>{criticObservationMean}
        );
        criticInverseStdOffset = appendArena(
            staged->arena,
            std::span<const float>{
                criticObservationInverseStd
            }
        );
    }
    const std::uint32_t actionBiasOffset = appendArena(
        staged->arena,
        std::span<const float>{actionBias}
    );
    const std::uint32_t actionScaleOffset = appendArena(
        staged->arena,
        std::span<const float>{actionScale}
    );
    std::uint32_t actionLogStdOffset = 0u;
    if (stochastic) {
        actionLogStdOffset = appendArena(
            staged->arena,
            std::span<const float>{
                pack.actionLogStandardDeviation
            }
        );
    }
    if (meanOffset == MR_INVALID_INDEX ||
        inverseStdOffset == MR_INVALID_INDEX ||
        criticMeanOffset == MR_INVALID_INDEX ||
        criticInverseStdOffset == MR_INVALID_INDEX ||
        actionBiasOffset == MR_INVALID_INDEX ||
        actionScaleOffset == MR_INVALID_INDEX ||
        actionLogStdOffset == MR_INVALID_INDEX) {
        return reject(
            PolicyCompileStatus::arithmeticOverflow,
            "arena",
            "policy transforms exceed the 32-bit byte-offset ABI"
        );
    }

    staged->header.counts0 = {
        staged->layout.actorLayerCount,
        staged->layout.criticLayerCount,
        staged->layout.actorObservationCount,
        staged->layout.criticObservationCount,
    };
    staged->header.counts1 = {
        staged->layout.actionCount,
        staged->layout.maximumHiddenCount,
        (hasCritic
             ? MR_POLICY_PROGRAM_HAS_CRITIC
             : 0u) |
            (stochastic
                 ? MR_POLICY_PROGRAM_STOCHASTIC
                 : 0u),
        0u,
    };
    staged->header.offsets0 = {
        actorLayerTableOffset,
        criticLayerTableOffset,
        meanOffset,
        inverseStdOffset,
    };
    staged->header.offsets1 = {
        criticMeanOffset,
        criticInverseStdOffset,
        actionBiasOffset,
        actionScaleOffset,
    };
    staged->header.offsets2 = {
        actionLogStdOffset,
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
    if (pack.contract.exact()) {
        hash.scalar(pack.contract.version);
        hash.scalar(pack.contract.worldFingerprint);
        hash.scalar(pack.contract.taskFingerprint);
        hash.scalar(pack.contract.observationFingerprint);
        hash.scalar(pack.contract.actionFingerprint);
    }
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
