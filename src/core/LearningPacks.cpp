#include "metalrobo/LearningPacks.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include <unistd.h>

namespace metalrobo {
namespace {

constexpr std::array<char, 8u> kMagic{
    'M', 'R', 'L', 'E', 'A', 'R', 'N', '\0',
};
constexpr std::uint32_t kTaskKind = 1u;
constexpr std::uint32_t kPolicyKind = 2u;
constexpr std::uint32_t kPolicyRolloutKind = 3u;
constexpr std::uint64_t kFNVOffset = 14695981039346656037ull;
constexpr std::uint64_t kFNVPrime = 1099511628211ull;
constexpr std::uint64_t kMaximumPayloadBytes =
    std::numeric_limits<std::uint32_t>::max();

struct LearningPackFileHeader {
    std::array<char, 8u> magic = kMagic;
    std::uint32_t formatVersion = 0u;
    std::uint32_t kind = 0u;
    std::uint64_t payloadBytes = 0u;
    std::uint64_t contentHash = 0u;
};

static_assert(
    std::is_trivially_copyable_v<LearningPackFileHeader>
);
static_assert(sizeof(LearningPackFileHeader) == 32u);
static_assert(std::endian::native == std::endian::little);

LearningPackResult fail(
    const LearningPackStatus status,
    std::string message
) {
    return {
        .status = status,
        .message = std::move(message),
    };
}

bool countFits(const std::size_t count) noexcept {
    return count <= std::numeric_limits<std::uint32_t>::max();
}

bool stringFits(const std::string_view value) noexcept {
    return countFits(value.size());
}

LearningPackResult validateTaskArtifact(
    const TaskPack& pack
) {
    if (pack.id.empty() || !stringFits(pack.id) ||
        pack.actions.empty() ||
        pack.actorFrame.empty() ||
        pack.actorHistoryLength == 0u ||
        pack.criticHistoryLength == 0u ||
        pack.maximumEpisodeSteps == 0u ||
        pack.curriculum.levelCount == 0u ||
        pack.curriculum.evaluationWindowSteps == 0u) {
        return fail(
            LearningPackStatus::invalidPack,
            "TaskPack identity, dimensions, or episode limits are invalid"
        );
    }
    if (!countFits(pack.actions.size()) ||
        !countFits(pack.actorFrame.size()) ||
        !countFits(pack.critic.size()) ||
        !countFits(pack.contactGroups.size()) ||
        !countFits(pack.frames.size()) ||
        !countFits(pack.goals.size()) ||
        !countFits(pack.signals.size()) ||
        !countFits(pack.rewards.size()) ||
        !countFits(pack.recorders.size()) ||
        !countFits(pack.terminations.size()) ||
        !countFits(pack.randomization.size()) ||
        !countFits(pack.commands.values.size()) ||
        !countFits(pack.terrain.sampleOffsets.size()) ||
        !countFits(pack.terrain.resetTranslations.size())) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "TaskPack table count exceeds the 32-bit artifact boundary"
        );
    }
    const auto validObservation = [](const auto& value) {
        return stringFits(value.target) &&
            stringFits(value.goal) &&
            stringFits(value.reference) &&
            stringFits(value.coordinate);
    };
    if (!std::all_of(
            pack.actorFrame.begin(),
            pack.actorFrame.end(),
            validObservation
        ) ||
        !std::all_of(
            pack.critic.begin(),
            pack.critic.end(),
            validObservation
        )) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "TaskPack observation semantic exceeds the 32-bit artifact boundary"
        );
    }
    for (const TaskActionBinding& value : pack.actions) {
        if (!stringFits(value.joint)) {
            return fail(
                LearningPackStatus::capacityOverflow,
                "TaskPack action semantic exceeds the 32-bit artifact boundary"
            );
        }
    }
    for (const TaskContactGroup& value : pack.contactGroups) {
        if (!stringFits(value.id) ||
            !stringFits(value.referenceBody) ||
            !countFits(value.bodies.size()) ||
            !std::all_of(
                value.bodies.begin(),
                value.bodies.end(),
                stringFits
            )) {
            return fail(
                LearningPackStatus::capacityOverflow,
                "TaskPack contact-group table exceeds the 32-bit artifact boundary"
            );
        }
    }
    for (const TaskFrameSpec& value : pack.frames) {
        if (!stringFits(value.id) || !stringFits(value.body) ||
            !stringFits(value.site)) {
            return fail(
                LearningPackStatus::capacityOverflow,
                "TaskPack frame semantic exceeds the 32-bit artifact boundary"
            );
        }
    }
    for (const TaskGoalSpec& value : pack.goals) {
        if (!stringFits(value.id)) {
            return fail(
                LearningPackStatus::capacityOverflow,
                "TaskPack goal semantic exceeds the 32-bit artifact boundary"
            );
        }
    }
    for (const TaskSignalSpec& value : pack.signals) {
        if (!stringFits(value.id) ||
            !stringFits(value.left) ||
            !stringFits(value.right) ||
            !validObservation(value.source) ||
            !countFits(value.reductionSources.size()) ||
            !std::all_of(
                value.reductionSources.begin(),
                value.reductionSources.end(),
                validObservation
            )) {
            return fail(
                LearningPackStatus::capacityOverflow,
                "TaskPack SignalIR semantic exceeds the 32-bit artifact boundary"
            );
        }
    }
    if (!std::all_of(
            pack.rewards.begin(),
            pack.rewards.end(),
            [](const auto& value) {
                return stringFits(value.signal);
            }
        ) ||
        !std::all_of(
            pack.recorders.begin(),
            pack.recorders.end(),
            [](const auto& value) {
                return stringFits(value.id) &&
                    stringFits(value.signal);
            }
        ) ||
        !std::all_of(
            pack.terminations.begin(),
            pack.terminations.end(),
            [](const auto& value) {
                return stringFits(value.sourceGroup) &&
                    stringFits(value.signal);
            }
        ) ||
        !std::all_of(
            pack.randomization.begin(),
            pack.randomization.end(),
            [](const auto& value) {
                return stringFits(value.target);
            }
        ) ||
        !std::all_of(
            pack.commands.values.begin(),
            pack.commands.values.end(),
            [](const auto& value) {
                return stringFits(value.id);
            }
        ) ||
        !stringFits(pack.curriculum.successSignal) ||
        !stringFits(pack.terrain.body)) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "TaskPack operator semantic exceeds the 32-bit artifact boundary"
        );
    }
    return {};
}

LearningPackResult validatePolicyArtifact(
    const PolicyPack& pack
) {
    if (pack.id.empty() || !stringFits(pack.id) ||
        pack.revision == 0u ||
        pack.layers.empty() ||
        !std::isfinite(pack.observationClip) ||
        !(pack.observationClip > 0.0f) ||
        !std::isfinite(pack.actionClip) ||
        !(pack.actionClip > 0.0f)) {
        return fail(
            LearningPackStatus::invalidPack,
            "PolicyPack identity, revision, layers, or clipping is invalid"
        );
    }
    if (!countFits(pack.layers.size()) ||
        !countFits(pack.criticLayers.size()) ||
        !countFits(pack.observationMean.size()) ||
        !countFits(
            pack.observationInverseStandardDeviation.size()
        ) ||
        !countFits(pack.criticObservationMean.size()) ||
        !countFits(
            pack.criticObservationInverseStandardDeviation.size()
        ) ||
        !countFits(
            pack.actionLogStandardDeviation.size()
        ) ||
        !countFits(pack.actionBias.size()) ||
        !countFits(pack.actionScale.size())) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "PolicyPack table count exceeds the 32-bit artifact boundary"
        );
    }
    const auto validLayerTable = [](
        const std::span<const PolicyDenseLayer> layers
    ) {
        return std::all_of(
            layers.begin(),
            layers.end(),
            [](const PolicyDenseLayer& layer) {
                const std::uint64_t expectedWeights =
                    static_cast<std::uint64_t>(
                        layer.inputCount
                    ) * layer.outputCount;
                const auto finiteValues = [](
                    const std::span<const float> values
                ) {
                    return std::all_of(
                        values.begin(),
                        values.end(),
                        [](const float value) {
                            return std::isfinite(value);
                        }
                    );
                };
                return layer.inputCount != 0u &&
                    layer.outputCount != 0u &&
                    static_cast<std::uint32_t>(
                        layer.activation
                    ) <= MR_POLICY_ACTIVATION_SILU &&
                    expectedWeights == layer.weights.size() &&
                    layer.bias.size() == layer.outputCount &&
                    countFits(layer.weights.size()) &&
                    countFits(layer.bias.size()) &&
                    finiteValues(layer.weights) &&
                    finiteValues(layer.bias);
            }
        );
    };
    const auto finiteValues = [](
        const std::span<const float> values
    ) {
        return std::all_of(
            values.begin(),
            values.end(),
            [](const float value) {
                return std::isfinite(value);
            }
        );
    };
    if (!validLayerTable(pack.layers) ||
        !validLayerTable(pack.criticLayers) ||
        (!pack.actionLogStandardDeviation.empty() &&
         pack.criticLayers.empty()) ||
        !finiteValues(pack.observationMean) ||
        !finiteValues(
            pack.observationInverseStandardDeviation
        ) ||
        !finiteValues(pack.criticObservationMean) ||
        !finiteValues(
            pack.criticObservationInverseStandardDeviation
        ) ||
        !finiteValues(pack.actionLogStandardDeviation) ||
        !finiteValues(pack.actionBias) ||
        !finiteValues(pack.actionScale)) {
        return fail(
            LearningPackStatus::invalidPack,
            "PolicyPack dense-layer shape or stochastic contract is invalid"
        );
    }
    return {};
}

template <typename Pack>
LearningPackResult validatePolicyRolloutArtifact(const Pack& pack) {
    if (pack.id.empty() || !stringFits(pack.id) ||
        pack.taskFingerprint == 0u ||
        pack.policyFingerprint == 0u ||
        pack.policyRevision == 0u ||
        pack.environmentCount == 0u ||
        pack.controlStepCount == 0u ||
        pack.actorObservationCount == 0u ||
        pack.criticObservationCount == 0u ||
        pack.actionCount == 0u) {
        return fail(
            LearningPackStatus::invalidPack,
            "PolicyRolloutPack identity, fingerprints, or dimensions are invalid"
        );
    }
    const auto multiply = [](
        const std::uint64_t left,
        const std::uint64_t right,
        std::uint64_t& output
    ) {
        if (right != 0u &&
            left >
                std::numeric_limits<std::uint64_t>::max() /
                    right) {
            return false;
        }
        output = left * right;
        return true;
    };
    std::uint64_t samples = 0u;
    std::uint64_t actorElements = 0u;
    std::uint64_t criticElements = 0u;
    std::uint64_t actionElements = 0u;
    if (!multiply(
            pack.environmentCount,
            pack.controlStepCount,
            samples
        ) ||
        !multiply(
            samples,
            pack.actorObservationCount,
            actorElements
        ) ||
        !multiply(
            samples,
            pack.criticObservationCount,
            criticElements
        ) ||
        !multiply(
            samples,
            pack.actionCount,
            actionElements
        ) ||
        samples >
            std::numeric_limits<std::size_t>::max() ||
        actorElements >
            std::numeric_limits<std::size_t>::max() ||
        criticElements >
            std::numeric_limits<std::size_t>::max() ||
        actionElements >
            std::numeric_limits<std::size_t>::max() ||
        pack.actorObservations.size() != actorElements ||
        pack.criticObservations.size() != criticElements ||
        pack.latents.size() != actionElements ||
        pack.logProbabilities.size() != samples ||
        pack.values.size() != samples ||
        pack.bootstrapValues.size() !=
            pack.environmentCount ||
        pack.transitions.size() != samples) {
        return fail(
            LearningPackStatus::invalidPack,
            "PolicyRolloutPack tensors do not match its declared dimensions"
        );
    }
    const auto finiteValues = [](
        const std::span<const float> values
    ) {
        return std::all_of(
            values.begin(),
            values.end(),
            [](const float value) {
                return std::isfinite(value);
            }
        );
    };
    if (!finiteValues(pack.actorObservations) ||
        !finiteValues(pack.criticObservations) ||
        !finiteValues(pack.latents) ||
        !finiteValues(pack.logProbabilities) ||
        !finiteValues(pack.values) ||
        !finiteValues(pack.bootstrapValues) ||
        !std::all_of(
            pack.transitions.begin(),
            pack.transitions.end(),
            [&](const MRTaskTransitionGPU& transition) {
                return transition.policyRevision ==
                        pack.policyRevision &&
                    std::isfinite(
                        transition.rewardAndMetrics.x
                    ) &&
                    std::isfinite(
                        transition.rewardAndMetrics.y
                    ) &&
                    std::isfinite(
                        transition.rewardAndMetrics.z
                    ) &&
                    std::isfinite(
                        transition.rewardAndMetrics.w
                    ) &&
                    std::isfinite(
                        transition.rewardBreakdown0.x
                    ) &&
                    std::isfinite(
                        transition.rewardBreakdown0.y
                    ) &&
                    std::isfinite(
                        transition.rewardBreakdown0.z
                    ) &&
                    std::isfinite(
                        transition.rewardBreakdown0.w
                    ) &&
                    std::isfinite(
                        transition.rewardBreakdown1.x
                    ) &&
                    std::isfinite(
                        transition.rewardBreakdown1.y
                    ) &&
                    std::isfinite(
                        transition.rewardBreakdown1.z
                    ) &&
                    std::isfinite(
                        transition.rewardBreakdown1.w
                    ) &&
                    std::isfinite(
                        transition.timeoutBootstrapValue
                    ) &&
                    std::isfinite(
                        transition.episodeMetric
                    ) &&
                    transition.termination.x <= 1u &&
                    transition.termination.y <= 1u &&
                    transition.termination.z <= 1u;
            }
        )) {
        return fail(
            LearningPackStatus::invalidPack,
            "PolicyRolloutPack contains non-finite or inconsistent samples"
        );
    }
    std::uint64_t payloadBytes =
        8u + pack.id.size() +
        3u * sizeof(std::uint64_t) +
        5u * sizeof(std::uint32_t) +
        7u * sizeof(std::uint64_t);
    if (payloadBytes > kMaximumPayloadBytes) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "PolicyRolloutPack payload exceeds the 32-bit artifact boundary"
        );
    }
    const auto addTable = [&]<typename Table>(
        const Table& values
    ) {
        using Value = typename Table::value_type;
        const std::uint64_t available =
            kMaximumPayloadBytes - payloadBytes;
        if (values.size() > available / sizeof(Value)) {
            return false;
        }
        payloadBytes +=
            static_cast<std::uint64_t>(values.size()) *
            sizeof(Value);
        return true;
    };
    if (!addTable(pack.actorObservations) ||
        !addTable(pack.criticObservations) ||
        !addTable(pack.latents) ||
        !addTable(pack.logProbabilities) ||
        !addTable(pack.values) ||
        !addTable(pack.bootstrapValues) ||
        !addTable(pack.transitions)) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "PolicyRolloutPack payload exceeds the 32-bit artifact boundary"
        );
    }
    return {};
}

std::uint64_t contentHash(
    const std::span<const std::byte> values
) {
    std::uint64_t hash = kFNVOffset;
    for (const std::byte value : values) {
        hash ^= std::to_integer<std::uint8_t>(value);
        hash *= kFNVPrime;
    }
    return hash == 0u ? 1u : hash;
}

class Writer {
public:
    explicit Writer(const std::size_t reservedBytes = 0u) {
        data_.reserve(reservedBytes);
    }

    template <typename T>
    void pod(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* bytes =
            reinterpret_cast<const std::byte*>(&value);
        data_.insert(data_.end(), bytes, bytes + sizeof(T));
    }

    void string(const std::string_view value) {
        const std::uint64_t count = value.size();
        pod(count);
        const auto* bytes =
            reinterpret_cast<const std::byte*>(
                value.data()
            );
        data_.insert(data_.end(), bytes, bytes + value.size());
    }

    template <typename T>
    void vector(const std::span<const T> values) {
        static_assert(std::is_trivially_copyable_v<T>);
        const std::uint64_t count = values.size();
        pod(count);
        if (!values.empty()) {
            const auto* bytes =
                reinterpret_cast<const std::byte*>(
                    values.data()
                );
            data_.insert(
                data_.end(),
                bytes,
                bytes + values.size() * sizeof(T)
            );
        }
    }

    template <typename T>
    void vector(const std::vector<T>& values) {
        vector(std::span<const T>{values});
    }

    void strings(const std::vector<std::string>& values) {
        const std::uint64_t count = values.size();
        pod(count);
        for (const std::string& value : values) {
            string(value);
        }
    }

    [[nodiscard]] const std::vector<std::byte>& data()
        const noexcept {
        return data_;
    }

private:
    std::vector<std::byte> data_;
};

class Reader {
public:
    explicit Reader(const std::span<const std::byte> data)
        : data_(data) {}

    template <typename T>
    bool pod(T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (sizeof(T) > data_.size() - cursor_) {
            return false;
        }
        std::memcpy(
            &value,
            data_.data() + cursor_,
            sizeof(T)
        );
        cursor_ += sizeof(T);
        return true;
    }

    bool string(std::string& value) {
        std::uint64_t count = 0u;
        if (!pod(count) ||
            count > data_.size() - cursor_ ||
            count > std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }
        value.assign(
            reinterpret_cast<const char*>(
                data_.data() + cursor_
            ),
            static_cast<std::size_t>(count)
        );
        cursor_ += static_cast<std::size_t>(count);
        return true;
    }

    template <typename T>
    bool vector(std::vector<T>& values) {
        static_assert(std::is_trivially_copyable_v<T>);
        std::uint64_t count = 0u;
        if (!pod(count) ||
            count > std::numeric_limits<std::uint32_t>::max() ||
            count >
                (data_.size() - cursor_) / sizeof(T)) {
            return false;
        }
        values.resize(static_cast<std::size_t>(count));
        const std::size_t bytes =
            values.size() * sizeof(T);
        if (bytes != 0u) {
            std::memcpy(
                values.data(),
                data_.data() + cursor_,
                bytes
            );
        }
        cursor_ += bytes;
        return true;
    }

    bool strings(std::vector<std::string>& values) {
        std::uint64_t count = 0u;
        if (!pod(count) ||
            count > std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }
        values.resize(static_cast<std::size_t>(count));
        for (std::string& value : values) {
            if (!string(value)) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool finished() const noexcept {
        return cursor_ == data_.size();
    }

private:
    std::span<const std::byte> data_;
    std::size_t cursor_ = 0u;
};

template <typename Enum>
void writeEnum(Writer& writer, const Enum value) {
    writer.pod(static_cast<std::uint32_t>(value));
}

template <typename Enum>
bool readEnum(Reader& reader, Enum& value) {
    std::uint32_t raw = 0u;
    if (!reader.pod(raw)) {
        return false;
    }
    value = static_cast<Enum>(raw);
    return true;
}

void writeObservation(
    Writer& writer,
    const TaskObservationOperatorSpec& value
) {
    writeEnum(writer, value.source);
    writer.string(value.target);
    writer.string(value.goal);
    writer.string(value.reference);
    writer.string(value.coordinate);
    writer.pod(value.component);
    writer.pod(value.parameters);
    writer.pod(value.scale);
    writer.pod(value.offset);
    writer.pod(value.noiseAmplitude);
    writer.pod(value.biasLower);
    writer.pod(value.biasUpper);
    writer.pod(
        static_cast<std::uint8_t>(
            value.normalizeVector3
        )
    );
}

bool readObservation(
    Reader& reader,
    TaskObservationOperatorSpec& value
) {
    std::uint8_t normalize = 0u;
    if (!readEnum(reader, value.source) ||
        !reader.string(value.target) ||
        !reader.string(value.goal) ||
        !reader.string(value.reference) ||
        !reader.string(value.coordinate) ||
        !reader.pod(value.component) ||
        !reader.pod(value.parameters) ||
        !reader.pod(value.scale) ||
        !reader.pod(value.offset) ||
        !reader.pod(value.noiseAmplitude) ||
        !reader.pod(value.biasLower) ||
        !reader.pod(value.biasUpper) ||
        !reader.pod(normalize) ||
        normalize > 1u) {
        return false;
    }
    value.normalizeVector3 = normalize != 0u;
    return true;
}

template <typename T, typename Function>
void writeRichVector(
    Writer& writer,
    const std::vector<T>& values,
    Function&& function
) {
    writer.pod(static_cast<std::uint64_t>(values.size()));
    for (const T& value : values) {
        function(writer, value);
    }
}

template <typename T, typename Function>
bool readRichVector(
    Reader& reader,
    std::vector<T>& values,
    Function&& function
) {
    std::uint64_t count = 0u;
    if (!reader.pod(count) ||
        count > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    values.resize(static_cast<std::size_t>(count));
    for (T& value : values) {
        if (!function(reader, value)) {
            return false;
        }
    }
    return true;
}

std::vector<std::byte> serializeTask(
    const TaskPack& pack
) {
    Writer writer;
    writer.string(pack.id);
    writer.pod(pack.capacities);
    writeRichVector(
        writer,
        pack.actions,
        [](Writer& target, const TaskActionBinding& value) {
            target.string(value.joint);
            target.pod(value.scale);
        }
    );
    writeRichVector(writer, pack.actorFrame, writeObservation);
    writer.pod(pack.actorHistoryLength);
    writeRichVector(writer, pack.critic, writeObservation);
    writer.pod(pack.criticHistoryLength);
    writer.pod(static_cast<std::uint8_t>(
        pack.criticIncludesCleanHistory
    ));
    writeRichVector(
        writer,
        pack.contactGroups,
        [](Writer& target, const TaskContactGroup& value) {
            target.string(value.id);
            target.strings(value.bodies);
            target.pod(static_cast<std::uint8_t>(
                value.support
            ));
            target.pod(static_cast<std::uint8_t>(
                value.forbidden
            ));
            target.string(value.referenceBody);
            target.pod(value.localReference);
            target.pod(value.gaitPhaseOffsetRadians);
            target.pod(value.stanceFraction);
        }
    );
    writeRichVector(
        writer,
        pack.frames,
        [](Writer& target, const TaskFrameSpec& value) {
            target.string(value.id);
            target.string(value.body);
            target.string(value.site);
            target.pod(value.localPosition);
            target.pod(value.localOrientation);
        }
    );
    writeRichVector(
        writer,
        pack.goals,
        [](Writer& target, const TaskGoalSpec& value) {
            target.string(value.id);
            writeEnum(target, value.mode);
            writeEnum(target, value.playback);
            target.pod(value.position);
            target.pod(value.orientation);
            target.pod(value.targetPosition);
            target.pod(value.targetOrientation);
            target.pod(value.positionOffsetLower);
            target.pod(value.positionOffsetUpper);
            target.pod(value.rotationVectorLower);
            target.pod(value.rotationVectorUpper);
            target.pod(value.durationSeconds);
            target.pod(value.phaseSeconds);
        }
    );
    writeRichVector(
        writer,
        pack.signals,
        [](Writer& target, const TaskSignalSpec& value) {
            target.string(value.id);
            writeEnum(target, value.operation);
            writeObservation(target, value.source);
            writeRichVector(
                target,
                value.reductionSources,
                writeObservation
            );
            writeEnum(target, value.transform);
            writeEnum(target, value.reduction);
            target.string(value.left);
            target.string(value.right);
            target.pod(value.parameters);
        }
    );
    writeRichVector(
        writer,
        pack.rewards,
        [](Writer& target, const TaskRewardOperatorSpec& value) {
            target.string(value.signal);
            writeEnum(target, value.channel);
            target.pod(value.weight);
        }
    );
    writeRichVector(
        writer,
        pack.recorders,
        [](Writer& target, const TaskRecorderSpec& value) {
            target.string(value.id);
            target.string(value.signal);
        }
    );
    writeRichVector(
        writer,
        pack.terminations,
        [](Writer& target,
           const TaskTerminationOperatorSpec& value) {
            writeEnum(target, value.operation);
            target.string(value.sourceGroup);
            target.string(value.signal);
            target.pod(value.reason);
            target.pod(value.priority);
            target.pod(value.threshold);
            target.pod(value.upperThreshold);
            target.pod(value.failurePenalty);
        }
    );
    writeRichVector(
        writer,
        pack.randomization,
        [](Writer& target,
           const TaskRandomizationOperatorSpec& value) {
            writeEnum(target, value.operation);
            target.string(value.target);
            target.pod(value.component);
            target.pod(value.minimumCurriculumLevel);
            target.pod(value.parameters);
        }
    );
    writeRichVector(
        writer,
        pack.commands.values,
        [](Writer& target, const TaskCommandSpec& value) {
            target.string(value.id);
            target.pod(value.lower);
            target.pod(value.upper);
            target.pod(value.limitLower);
            target.pod(value.limitUpper);
            target.pod(value.curriculumStep);
        }
    );
    writer.pod(pack.commands.zeroProbability);
    writer.pod(pack.commands.minimumDurationSeconds);
    writer.pod(pack.commands.maximumDurationSeconds);
    writer.pod(pack.phase.periodSeconds);
    writer.pod(pack.curriculum.levelCount);
    writer.pod(pack.curriculum.evaluationWindowSteps);
    writer.string(pack.curriculum.successSignal);
    writer.pod(pack.curriculum.successThreshold);
    writer.pod(pack.curriculum.minimumEpisodeSurvivalFraction);
    writer.pod(pack.pushes.maximumVelocity);
    writer.pod(pack.pushes.minimumIntervalSeconds);
    writer.pod(pack.pushes.maximumIntervalSeconds);
    writer.string(pack.terrain.body);
    writer.vector(pack.terrain.sampleOffsets);
    writer.vector(pack.terrain.resetTranslations);
    writer.pod(pack.maximumEpisodeSteps);
    writer.pod(pack.maximumActionDelaySteps);
    writer.pod(pack.maximumObservationDelaySteps);
    writer.pod(pack.supportForceThreshold);
    return writer.data();
}

bool deserializeTask(
    const std::span<const std::byte> payload,
    TaskPack& pack
) {
    Reader reader{payload};
    std::uint8_t cleanHistory = 0u;
    if (!reader.string(pack.id) ||
        !reader.pod(pack.capacities) ||
        !readRichVector(
            reader,
            pack.actions,
            [](Reader& source, TaskActionBinding& value) {
                return source.string(value.joint) &&
                    source.pod(value.scale);
            }
        ) ||
        !readRichVector(
            reader,
            pack.actorFrame,
            readObservation
        ) ||
        !reader.pod(pack.actorHistoryLength) ||
        !readRichVector(reader, pack.critic, readObservation) ||
        !reader.pod(pack.criticHistoryLength) ||
        !reader.pod(cleanHistory) ||
        cleanHistory > 1u ||
        !readRichVector(
            reader,
            pack.contactGroups,
            [](Reader& source, TaskContactGroup& value) {
                std::uint8_t support = 0u;
                std::uint8_t forbidden = 0u;
                if (!source.string(value.id) ||
                    !source.strings(value.bodies) ||
                    !source.pod(support) ||
                    !source.pod(forbidden) ||
                    support > 1u || forbidden > 1u ||
                    !source.string(value.referenceBody) ||
                    !source.pod(value.localReference) ||
                    !source.pod(
                        value.gaitPhaseOffsetRadians
                    ) ||
                    !source.pod(value.stanceFraction)) {
                    return false;
                }
                value.support = support != 0u;
                value.forbidden = forbidden != 0u;
                return true;
            }
        ) ||
        !readRichVector(
            reader,
            pack.frames,
            [](Reader& source, TaskFrameSpec& value) {
                return source.string(value.id) &&
                    source.string(value.body) &&
                    source.string(value.site) &&
                    source.pod(value.localPosition) &&
                    source.pod(value.localOrientation);
            }
        ) ||
        !readRichVector(
            reader,
            pack.goals,
            [](Reader& source, TaskGoalSpec& value) {
                return source.string(value.id) &&
                    readEnum(source, value.mode) &&
                    readEnum(source, value.playback) &&
                    source.pod(value.position) &&
                    source.pod(value.orientation) &&
                    source.pod(value.targetPosition) &&
                    source.pod(value.targetOrientation) &&
                    source.pod(value.positionOffsetLower) &&
                    source.pod(value.positionOffsetUpper) &&
                    source.pod(value.rotationVectorLower) &&
                    source.pod(value.rotationVectorUpper) &&
                    source.pod(value.durationSeconds) &&
                    source.pod(value.phaseSeconds);
            }
        ) ||
        !readRichVector(
            reader,
            pack.signals,
            [](Reader& source, TaskSignalSpec& value) {
                return source.string(value.id) &&
                    readEnum(source, value.operation) &&
                    readObservation(source, value.source) &&
                    readRichVector(
                        source,
                        value.reductionSources,
                        readObservation
                    ) &&
                    readEnum(source, value.transform) &&
                    readEnum(source, value.reduction) &&
                    source.string(value.left) &&
                    source.string(value.right) &&
                    source.pod(value.parameters);
            }
        ) ||
        !readRichVector(
            reader,
            pack.rewards,
            [](Reader& source,
               TaskRewardOperatorSpec& value) {
                return source.string(value.signal) &&
                    readEnum(source, value.channel) &&
                    source.pod(value.weight);
            }
        ) ||
        !readRichVector(
            reader,
            pack.recorders,
            [](Reader& source, TaskRecorderSpec& value) {
                return source.string(value.id) &&
                    source.string(value.signal);
            }
        ) ||
        !readRichVector(
            reader,
            pack.terminations,
            [](Reader& source,
               TaskTerminationOperatorSpec& value) {
                return readEnum(source, value.operation) &&
                    source.string(value.sourceGroup) &&
                    source.string(value.signal) &&
                    source.pod(value.reason) &&
                    source.pod(value.priority) &&
                    source.pod(value.threshold) &&
                    source.pod(value.upperThreshold) &&
                    source.pod(value.failurePenalty);
            }
        ) ||
        !readRichVector(
            reader,
            pack.randomization,
            [](Reader& source,
               TaskRandomizationOperatorSpec& value) {
                return readEnum(source, value.operation) &&
                    source.string(value.target) &&
                    source.pod(value.component) &&
                    source.pod(value.minimumCurriculumLevel) &&
                    source.pod(value.parameters);
            }
        ) ||
        !readRichVector(
            reader,
            pack.commands.values,
            [](Reader& source, TaskCommandSpec& value) {
                return source.string(value.id) &&
                    source.pod(value.lower) &&
                    source.pod(value.upper) &&
                    source.pod(value.limitLower) &&
                    source.pod(value.limitUpper) &&
                    source.pod(value.curriculumStep);
            }
        ) ||
        !reader.pod(pack.commands.zeroProbability) ||
        !reader.pod(pack.commands.minimumDurationSeconds) ||
        !reader.pod(pack.commands.maximumDurationSeconds) ||
        !reader.pod(pack.phase.periodSeconds) ||
        !reader.pod(pack.curriculum.levelCount) ||
        !reader.pod(pack.curriculum.evaluationWindowSteps) ||
        !reader.string(pack.curriculum.successSignal) ||
        !reader.pod(pack.curriculum.successThreshold) ||
        !reader.pod(
            pack.curriculum.minimumEpisodeSurvivalFraction
        ) ||
        !reader.pod(pack.pushes.maximumVelocity) ||
        !reader.pod(pack.pushes.minimumIntervalSeconds) ||
        !reader.pod(pack.pushes.maximumIntervalSeconds) ||
        !reader.string(pack.terrain.body) ||
        !reader.vector(pack.terrain.sampleOffsets) ||
        !reader.vector(pack.terrain.resetTranslations) ||
        !reader.pod(pack.maximumEpisodeSteps) ||
        !reader.pod(pack.maximumActionDelaySteps) ||
        !reader.pod(pack.maximumObservationDelaySteps) ||
        !reader.pod(pack.supportForceThreshold) ||
        !reader.finished()) {
        return false;
    }
    pack.criticIncludesCleanHistory = cleanHistory != 0u;
    return true;
}

std::vector<std::byte> serializePolicy(
    const PolicyPack& pack
) {
    Writer writer;
    writer.string(pack.id);
    writer.pod(pack.revision);
    writer.vector(pack.observationMean);
    writer.vector(
        pack.observationInverseStandardDeviation
    );
    writeRichVector(
        writer,
        pack.layers,
        [](Writer& target, const PolicyDenseLayer& layer) {
            target.pod(layer.inputCount);
            target.pod(layer.outputCount);
            writeEnum(target, layer.activation);
            target.vector(layer.weights);
            target.vector(layer.bias);
        }
    );
    writer.vector(pack.criticObservationMean);
    writer.vector(
        pack.criticObservationInverseStandardDeviation
    );
    writeRichVector(
        writer,
        pack.criticLayers,
        [](Writer& target, const PolicyDenseLayer& layer) {
            target.pod(layer.inputCount);
            target.pod(layer.outputCount);
            writeEnum(target, layer.activation);
            target.vector(layer.weights);
            target.vector(layer.bias);
        }
    );
    writer.vector(pack.actionLogStandardDeviation);
    writer.vector(pack.actionBias);
    writer.vector(pack.actionScale);
    writer.pod(pack.observationClip);
    writer.pod(pack.actionClip);
    return writer.data();
}

bool deserializePolicy(
    const std::span<const std::byte> payload,
    PolicyPack& pack
) {
    Reader reader{payload};
    return reader.string(pack.id) &&
        reader.pod(pack.revision) &&
        reader.vector(pack.observationMean) &&
        reader.vector(
            pack.observationInverseStandardDeviation
        ) &&
        readRichVector(
            reader,
            pack.layers,
            [](Reader& source, PolicyDenseLayer& layer) {
                return source.pod(layer.inputCount) &&
                    source.pod(layer.outputCount) &&
                    readEnum(source, layer.activation) &&
                    source.vector(layer.weights) &&
                    source.vector(layer.bias);
            }
        ) &&
        reader.vector(pack.criticObservationMean) &&
        reader.vector(
            pack.criticObservationInverseStandardDeviation
        ) &&
        readRichVector(
            reader,
            pack.criticLayers,
            [](Reader& source, PolicyDenseLayer& layer) {
                return source.pod(layer.inputCount) &&
                    source.pod(layer.outputCount) &&
                    readEnum(source, layer.activation) &&
                    source.vector(layer.weights) &&
                    source.vector(layer.bias);
            }
        ) &&
        reader.vector(pack.actionLogStandardDeviation) &&
        reader.vector(pack.actionBias) &&
        reader.vector(pack.actionScale) &&
        reader.pod(pack.observationClip) &&
        reader.pod(pack.actionClip) &&
        reader.finished();
}

template <typename Pack>
std::vector<std::byte> serializePolicyRollout(const Pack& pack) {
    const std::size_t payloadBytes =
        8u + pack.id.size() +
        3u * sizeof(std::uint64_t) +
        5u * sizeof(std::uint32_t) +
        7u * sizeof(std::uint64_t) +
        pack.actorObservations.size() * sizeof(float) +
        pack.criticObservations.size() * sizeof(float) +
        pack.latents.size() * sizeof(float) +
        pack.logProbabilities.size() * sizeof(float) +
        pack.values.size() * sizeof(float) +
        pack.bootstrapValues.size() * sizeof(float) +
        pack.transitions.size() *
            sizeof(MRTaskTransitionGPU);
    Writer writer{payloadBytes};
    writer.string(pack.id);
    writer.pod(pack.taskFingerprint);
    writer.pod(pack.policyFingerprint);
    writer.pod(pack.policyRevision);
    writer.pod(pack.environmentCount);
    writer.pod(pack.controlStepCount);
    writer.pod(pack.actorObservationCount);
    writer.pod(pack.criticObservationCount);
    writer.pod(pack.actionCount);
    writer.vector(pack.actorObservations);
    writer.vector(pack.criticObservations);
    writer.vector(pack.latents);
    writer.vector(pack.logProbabilities);
    writer.vector(pack.values);
    writer.vector(pack.bootstrapValues);
    writer.vector(pack.transitions);
    return writer.data();
}

bool deserializePolicyRollout(
    const std::span<const std::byte> payload,
    PolicyRolloutPack& pack
) {
    Reader reader{payload};
    return reader.string(pack.id) &&
        reader.pod(pack.taskFingerprint) &&
        reader.pod(pack.policyFingerprint) &&
        reader.pod(pack.policyRevision) &&
        reader.pod(pack.environmentCount) &&
        reader.pod(pack.controlStepCount) &&
        reader.pod(pack.actorObservationCount) &&
        reader.pod(pack.criticObservationCount) &&
        reader.pod(pack.actionCount) &&
        reader.vector(pack.actorObservations) &&
        reader.vector(pack.criticObservations) &&
        reader.vector(pack.latents) &&
        reader.vector(pack.logProbabilities) &&
        reader.vector(pack.values) &&
        reader.vector(pack.bootstrapValues) &&
        reader.vector(pack.transitions) &&
        reader.finished();
}

bool writeAll(
    const int descriptor,
    const void* bytes,
    const std::size_t byteCount
) noexcept {
    const auto* cursor =
        static_cast<const std::byte*>(bytes);
    std::size_t remaining = byteCount;
    while (remaining != 0u) {
        const ssize_t written = ::write(
            descriptor,
            cursor,
            remaining
        );
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (written == 0) {
            return false;
        }
        cursor += written;
        remaining -= static_cast<std::size_t>(written);
    }
    return true;
}

LearningPackResult writePack(
    const std::vector<std::byte>& payload,
    const std::uint32_t kind,
    const std::uint32_t version,
    const std::filesystem::path& path
) {
    if (path.empty() ||
        payload.size() > kMaximumPayloadBytes) {
        return fail(
            payload.size() > kMaximumPayloadBytes
                ? LearningPackStatus::capacityOverflow
                : LearningPackStatus::ioFailure,
            "learning pack path is empty or payload exceeds the 32-bit artifact boundary"
        );
    }
    try {
        LearningPackFileHeader header;
        header.formatVersion = version;
        header.kind = kind;
        header.payloadBytes = payload.size();
        header.contentHash = contentHash(payload);
        std::string temporaryTemplate =
            path.string() + ".tmp.XXXXXX";
        std::vector<char> temporaryCharacters(
            temporaryTemplate.begin(),
            temporaryTemplate.end()
        );
        temporaryCharacters.push_back('\0');
        const int descriptor =
            ::mkstemp(temporaryCharacters.data());
        if (descriptor < 0) {
            return fail(
                LearningPackStatus::ioFailure,
                "could not create sibling learning-pack temporary"
            );
        }
        const std::filesystem::path temporary{
            temporaryCharacters.data()
        };
        const bool wrote =
            writeAll(descriptor, &header, sizeof(header)) &&
            (payload.empty() ||
             writeAll(
                 descriptor,
                 payload.data(),
                 payload.size()
             )) &&
            ::fsync(descriptor) == 0;
        const bool closed = ::close(descriptor) == 0;
        if (!wrote || !closed) {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return fail(
                LearningPackStatus::ioFailure,
                "could not durably write learning pack"
            );
        }
        if (::rename(
                temporary.c_str(),
                path.c_str()
            ) != 0) {
            const std::string error =
                std::generic_category().message(errno);
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return fail(
                LearningPackStatus::ioFailure,
                "could not atomically publish learning pack: " +
                    error
            );
        }
        return {
            .status = LearningPackStatus::success,
            .contentHash = header.contentHash,
        };
    } catch (const std::bad_alloc&) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "learning pack allocation failed"
        );
    } catch (const std::exception& error) {
        return fail(
            LearningPackStatus::ioFailure,
            error.what()
        );
    }
}

template <typename Pack, typename Deserialize>
LearningPackResult readPack(
    const std::filesystem::path& path,
    const std::uint32_t expectedKind,
    const std::uint32_t expectedVersion,
    Pack& output,
    Deserialize&& deserialize
) {
    try {
        std::error_code sizeError;
        const std::uintmax_t fileBytes =
            std::filesystem::file_size(path, sizeError);
        if (sizeError ||
            fileBytes < sizeof(LearningPackFileHeader) ||
            fileBytes >
                sizeof(LearningPackFileHeader) +
                    kMaximumPayloadBytes) {
            return fail(
                LearningPackStatus::ioFailure,
                "learning pack is missing or has an invalid length"
            );
        }
        std::ifstream stream(path, std::ios::binary);
        LearningPackFileHeader header;
        if (!stream.read(
                reinterpret_cast<char*>(&header),
                sizeof(header)
            ) ||
            header.magic != kMagic ||
            header.kind != expectedKind) {
            return fail(
                LearningPackStatus::corruptPayload,
                "learning pack header or kind is invalid"
            );
        }
        if (header.formatVersion != expectedVersion) {
            return fail(
                LearningPackStatus::unsupportedVersion,
                "learning pack wire version is unsupported"
            );
        }
        if (header.payloadBytes !=
            fileBytes - sizeof(header)) {
            return fail(
                LearningPackStatus::corruptPayload,
                "learning pack payload length is inconsistent"
            );
        }
        std::vector<std::byte> payload(
            static_cast<std::size_t>(header.payloadBytes)
        );
        if (!payload.empty() &&
            !stream.read(
                reinterpret_cast<char*>(payload.data()),
                static_cast<std::streamsize>(payload.size())
            )) {
            return fail(
                LearningPackStatus::ioFailure,
                "could not read learning pack payload"
            );
        }
        if (contentHash(payload) != header.contentHash) {
            return fail(
                LearningPackStatus::corruptPayload,
                "learning pack content fingerprint does not match"
            );
        }
        Pack candidate;
        if (!deserialize(payload, candidate)) {
            return fail(
                LearningPackStatus::corruptPayload,
                "learning pack payload is malformed"
            );
        }
        output = std::move(candidate);
        return {
            .status = LearningPackStatus::success,
            .contentHash = header.contentHash,
        };
    } catch (const std::bad_alloc&) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "learning pack allocation failed"
        );
    } catch (const std::exception& error) {
        return fail(
            LearningPackStatus::ioFailure,
            error.what()
        );
    }
}

} // namespace

std::uint64_t learningPackContentHash(
    const std::span<const std::byte> payload
) noexcept {
    return contentHash(payload);
}

LearningPackResult writeTaskPack(
    const TaskPack& pack,
    const std::filesystem::path& path
) {
    try {
        const LearningPackResult validation =
            validateTaskArtifact(pack);
        if (!validation.succeeded()) {
            return validation;
        }
        return writePack(
            serializeTask(pack),
            kTaskKind,
            kTaskPackFormatVersion,
            path
        );
    } catch (const std::bad_alloc&) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "TaskPack serialization allocation failed"
        );
    } catch (const std::exception& error) {
        return fail(
            LearningPackStatus::internalFailure,
            error.what()
        );
    }
}

LearningPackResult readTaskPack(
    const std::filesystem::path& path,
    TaskPack& output
) {
    return readPack(
        path,
        kTaskKind,
        kTaskPackFormatVersion,
        output,
        deserializeTask
    );
}

LearningPackResult writePolicyPack(
    const PolicyPack& pack,
    const std::filesystem::path& path
) {
    try {
        const LearningPackResult validation =
            validatePolicyArtifact(pack);
        if (!validation.succeeded()) {
            return validation;
        }
        return writePack(
            serializePolicy(pack),
            kPolicyKind,
            kPolicyPackFormatVersion,
            path
        );
    } catch (const std::bad_alloc&) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "PolicyPack serialization allocation failed"
        );
    } catch (const std::exception& error) {
        return fail(
            LearningPackStatus::internalFailure,
            error.what()
        );
    }
}

LearningPackResult readPolicyPack(
    const std::filesystem::path& path,
    PolicyPack& output
) {
    return readPack(
        path,
        kPolicyKind,
        kPolicyPackFormatVersion,
        output,
        deserializePolicy
    );
}

LearningPackResult writePolicyRolloutPack(
    const PolicyRolloutPack& pack,
    const std::filesystem::path& path
) {
    return writePolicyRolloutPack(
        PolicyRolloutPackView{
            .id = pack.id,
            .taskFingerprint = pack.taskFingerprint,
            .policyFingerprint = pack.policyFingerprint,
            .policyRevision = pack.policyRevision,
            .environmentCount = pack.environmentCount,
            .controlStepCount = pack.controlStepCount,
            .actorObservationCount =
                pack.actorObservationCount,
            .criticObservationCount =
                pack.criticObservationCount,
            .actionCount = pack.actionCount,
            .actorObservations = pack.actorObservations,
            .criticObservations = pack.criticObservations,
            .latents = pack.latents,
            .logProbabilities = pack.logProbabilities,
            .values = pack.values,
            .bootstrapValues = pack.bootstrapValues,
            .transitions = pack.transitions,
        },
        path
    );
}

LearningPackResult writePolicyRolloutPack(
    const PolicyRolloutPackView& pack,
    const std::filesystem::path& path
) {
    try {
        const LearningPackResult validation =
            validatePolicyRolloutArtifact(pack);
        if (!validation.succeeded()) {
            return validation;
        }
        return writePack(
            serializePolicyRollout(pack),
            kPolicyRolloutKind,
            kPolicyRolloutPackFormatVersion,
            path
        );
    } catch (const std::bad_alloc&) {
        return fail(
            LearningPackStatus::capacityOverflow,
            "PolicyRolloutPack serialization allocation failed"
        );
    } catch (const std::exception& error) {
        return fail(
            LearningPackStatus::internalFailure,
            error.what()
        );
    }
}

LearningPackResult readPolicyRolloutPack(
    const std::filesystem::path& path,
    PolicyRolloutPack& output
) {
    PolicyRolloutPack staged;
    LearningPackResult result = readPack(
        path,
        kPolicyRolloutKind,
        kPolicyRolloutPackFormatVersion,
        staged,
        deserializePolicyRollout
    );
    if (!result.succeeded()) {
        return result;
    }
    result = validatePolicyRolloutArtifact(staged);
    if (!result.succeeded()) {
        result.status = LearningPackStatus::corruptPayload;
        return result;
    }
    output = std::move(staged);
    return result;
}

const char* learningPackStatusName(
    const LearningPackStatus status
) noexcept {
    switch (status) {
    case LearningPackStatus::success:
        return "success";
    case LearningPackStatus::invalidPack:
        return "invalid_pack";
    case LearningPackStatus::ioFailure:
        return "io_failure";
    case LearningPackStatus::unsupportedVersion:
        return "unsupported_version";
    case LearningPackStatus::corruptPayload:
        return "corrupt_payload";
    case LearningPackStatus::capacityOverflow:
        return "capacity_overflow";
    case LearningPackStatus::internalFailure:
        return "internal_failure";
    }
    return "unknown";
}

} // namespace metalrobo
