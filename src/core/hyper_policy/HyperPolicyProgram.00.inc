#include "metalrobo/HyperPolicyProgram.hpp"

#include "metalrobo/TaskProgram.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <iterator>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace metalrobo {

struct CompiledHyperPolicyProgram::Storage {
    std::uint64_t fingerprint = 0u;
    std::uint64_t taskFingerprint = 0u;
    std::uint64_t revision = 0u;
    HyperPolicyProgramLayout layout{};
    MRHyperPolicyProgramHeaderGPU header{};
    std::vector<MRHyperPolicyLayerGPU> layers;
    std::vector<std::byte> arena;
    std::vector<float> actionLower;
    std::vector<float> actionUpper;
    std::vector<float> maximumActionRate;
};

namespace {

constexpr std::uint64_t kFNVOffset = 1469598103934665603ull;
constexpr std::uint64_t kFNVPrime = 1099511628211ull;

class Hash {
public:
    void bytes(const void* data, const std::size_t size) {
        const auto* values = static_cast<const unsigned char*>(data);
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

    template <typename T>
    void vector(const std::span<const T> values) {
        scalar<std::uint64_t>(values.size());
        bytes(values.data(), values.size_bytes());
    }

    [[nodiscard]] std::uint64_t finish() const noexcept {
        return value_ == 0u ? 1u : value_;
    }

private:
    std::uint64_t value_ = kFNVOffset;
};

HyperPolicyCompileDiagnostics reject(
    const HyperPolicyCompileStatus status,
    std::string element,
    std::string message
) {
    return {
        .status = status,
        .element = std::move(element),
        .message = std::move(message),
    };
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

bool strictlyNormalizedPhases(const std::span<const float> phases) {
    if (phases.size() < 2u || !finite(phases) ||
        std::abs(phases.front()) > 1.0e-6f ||
        std::abs(phases.back() - 1.0f) > 1.0e-6f) {
        return false;
    }
    for (std::size_t index = 1u; index < phases.size(); ++index) {
        if (!(phases[index] > phases[index - 1u])) {
            return false;
        }
    }
    return true;
}

std::uint32_t appendArena(
    std::vector<std::byte>& arena,
    const void* values,
    const std::size_t bytes
) {
    constexpr std::size_t alignment = 16u;
    const std::size_t aligned =
        (arena.size() + alignment - 1u) / alignment * alignment;
    if (aligned > std::numeric_limits<std::uint32_t>::max() ||
        bytes > std::numeric_limits<std::uint32_t>::max() - aligned) {
        return MR_HYPER_POLICY_INVALID_INDEX;
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
    return appendArena(arena, values.data(), values.size_bytes());
}

std::uint64_t fingerprintPack(
    const HyperPolicyPack& pack,
    const CompiledTaskProgram& task
) {
    Hash hash;
    hash.string(pack.id);
    hash.scalar(pack.revision);
    hash.scalar(pack.contract.version);
    hash.scalar(pack.contract.worldFingerprint);
    hash.scalar(pack.contract.taskFingerprint);
    hash.scalar(pack.contract.observationFingerprint);
    hash.scalar(pack.contract.actionFingerprint);
    hash.scalar(pack.sourceMotionFingerprint);
    hash.scalar(task.fingerprint());
    hash.scalar(pack.observationClip);
    hash.scalar(pack.actionClip);
    hash.scalar(pack.signatureCount);
    hash.scalar(pack.contactTrackCount);
    hash.scalar(pack.maximumPhaseAdvancePerStep);
    hash.scalar(pack.phaseAlignmentBlend);
    hash.scalar(pack.phaseAlignmentHuberDelta);
    hash.scalar(pack.controlTimeStep);
    hash.vector<float>(pack.observationMean);
    hash.vector<float>(pack.observationInverseStandardDeviation);
    hash.vector<float>(pack.coefficientLimits);
    hash.vector<float>(pack.actionBias);
    hash.vector<float>(pack.actionScale);
    hash.vector<float>(pack.actionLogStandardDeviation);
    hash.vector<float>(pack.knotPhases);
    hash.vector<float>(pack.coefficientKnots);
    hash.vector<float>(pack.coefficientTangents);
    hash.vector<float>(pack.authorityKnots);
    hash.vector<float>(pack.authorityTangents);
    hash.vector<float>(pack.phaseRateKnots);
    hash.vector<float>(pack.phaseRateTangents);
    hash.vector<float>(pack.referencePhases);
    hash.vector<float>(pack.referenceActions);
    hash.vector<float>(pack.referenceSignatures);
    hash.vector<float>(pack.signatureWeights);
    hash.vector<std::uint32_t>(pack.referenceContactMasks);
    hash.vector<std::uint32_t>(pack.contactGroupIndices);
    hash.vector<float>(pack.actionLower);
    hash.vector<float>(pack.actionUpper);
    hash.vector<float>(pack.maximumActionRate);
    hash.scalar<std::uint64_t>(pack.layers.size());
    for (const HyperPolicyLayer& layer : pack.layers) {
        hash.scalar(layer.inputCount);
        hash.scalar(layer.outputCount);
        hash.scalar(layer.rank);
        hash.scalar(static_cast<std::uint32_t>(layer.activation));
        hash.vector<float>(layer.weights);
        hash.vector<float>(layer.bias);
        hash.vector<float>(layer.adapterDown);
        hash.vector<float>(layer.adapterUp);
        hash.vector<float>(layer.adapterBiasBasis);
    }
    hash.scalar<std::uint64_t>(pack.events.size());
    for (const HyperPolicyEventGuard& event : pack.events) {
        hash.scalar(event.phase);
        hash.scalar(event.confidence);
        hash.scalar(event.requiredContactOnMask);
        hash.scalar(event.requiredContactOffMask);
        hash.scalar(event.minimumDwellSteps);
        hash.scalar(event.kind);
    }
    return hash.finish();
}

} // namespace

bool CompiledHyperPolicyProgram::valid() const noexcept {
    return storage_ != nullptr &&
        storage_->fingerprint != 0u &&
        storage_->taskFingerprint != 0u &&
        storage_->revision != 0u &&
        storage_->header.policyFingerprint == storage_->fingerprint &&
        storage_->header.taskFingerprint == storage_->taskFingerprint &&
        storage_->header.revision == storage_->revision &&
        storage_->header.abi.x == MR_HYPER_POLICY_PROGRAM_ABI_VERSION &&
        storage_->layout.layerCount != 0u &&
        storage_->layout.actorObservationCount != 0u &&
        storage_->layout.actionCount != 0u &&
        storage_->layout.coefficientCount != 0u &&
        storage_->layout.knotCount >= 2u &&
        storage_->layout.referenceCount >= 2u;
}

std::uint64_t CompiledHyperPolicyProgram::fingerprint() const noexcept {
    return valid() ? storage_->fingerprint : 0u;
}

std::uint64_t CompiledHyperPolicyProgram::taskFingerprint() const noexcept {
    return valid() ? storage_->taskFingerprint : 0u;
}

std::uint64_t CompiledHyperPolicyProgram::revision() const noexcept {
    return valid() ? storage_->revision : 0u;
}

const HyperPolicyProgramLayout&
CompiledHyperPolicyProgram::layout() const noexcept {
    static const HyperPolicyProgramLayout empty{};
    return valid() ? storage_->layout : empty;
}

const MRHyperPolicyProgramHeaderGPU&
CompiledHyperPolicyProgram::header() const noexcept {
    static const MRHyperPolicyProgramHeaderGPU empty{};
    return valid() ? storage_->header : empty;
}

std::span<const MRHyperPolicyLayerGPU>
CompiledHyperPolicyProgram::layers() const noexcept {
    return valid()
        ? std::span<const MRHyperPolicyLayerGPU>{storage_->layers}
        : std::span<const MRHyperPolicyLayerGPU>{};
}

std::span<const std::byte>
CompiledHyperPolicyProgram::arena() const noexcept {
    return valid()
        ? std::span<const std::byte>{storage_->arena}
        : std::span<const std::byte>{};
}

