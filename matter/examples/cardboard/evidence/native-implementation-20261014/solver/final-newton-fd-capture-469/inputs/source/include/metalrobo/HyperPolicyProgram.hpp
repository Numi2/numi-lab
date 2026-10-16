#pragma once

#include "metalrobo/PolicyProgram.hpp"
#include "metalrobo/hyper_policy_program_types.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

class CompiledTaskProgram;

struct HyperPolicyLayer {
    std::uint32_t inputCount = 0u;
    std::uint32_t outputCount = 0u;
    std::uint32_t rank = 0u;
    PolicyActivation activation = PolicyActivation::elu;

    // Base output-major [output][input].
    std::vector<float> weights;
    std::vector<float> bias;
    // Adapter-down [rank][input], adapter-up [output][rank], and generated
    // bias basis [output][rank].  The hypernetwork emits one bounded gate per
    // rank and phase knot; it never materializes a full layer weight matrix.
    std::vector<float> adapterDown;
    std::vector<float> adapterUp;
    std::vector<float> adapterBiasBasis;
};

struct HyperPolicyEventGuard {
    float phase = 0.0f;
    float confidence = 1.0f;
    std::uint32_t requiredContactOnMask = 0u;
    std::uint32_t requiredContactOffMask = 0u;
    std::uint32_t minimumDwellSteps = 1u;
    std::uint32_t kind = 0u;
};

// Complete in-memory product of the offline ARDY hypernetwork compiler.  It is
// independent of ARDY at execution time and contains only authenticated robot
// references, generated adapter programs, phase guards, and safety envelopes.
struct HyperPolicyPack {
    std::string id;
    std::uint64_t revision = 1u;
    PolicyContract contract;
    std::uint64_t sourceMotionFingerprint = 0u;

    std::vector<float> observationMean;
    std::vector<float> observationInverseStandardDeviation;
    std::vector<HyperPolicyLayer> layers;
    std::vector<float> coefficientLimits;

    std::vector<float> actionBias;
    std::vector<float> actionScale;
    std::vector<float> actionLogStandardDeviation;
    float observationClip = 100.0f;
    float actionClip = std::numeric_limits<float>::max();

    // Strictly increasing normalized phases with endpoints 0 and 1.
    std::vector<float> knotPhases;
    // Knot-major [knot][coefficient].
    std::vector<float> coefficientKnots;
    std::vector<float> coefficientTangents;
    // Knot-major [knot][action].
    std::vector<float> authorityKnots;
    std::vector<float> authorityTangents;
    std::vector<float> phaseRateKnots;
    std::vector<float> phaseRateTangents;

    // Reference-major tables consumed by phase alignment and final actions.
    std::vector<float> referencePhases;
    std::vector<float> referenceActions;
    std::uint32_t signatureCount = 0u;
    std::vector<float> referenceSignatures;
    std::vector<float> signatureWeights;
    std::uint32_t contactTrackCount = 0u;
    std::vector<std::uint32_t> referenceContactMasks;
    // Maps each generated contact track to one compiled TaskPack
    // contact group. The runtime reads only solved compact contact
    // loads from these groups; ARDY contact modes remain intent.
    std::vector<std::uint32_t> contactGroupIndices;
    std::vector<HyperPolicyEventGuard> events;

    // Independently authored physical actuator envelope.  The hypernetwork may
    // reduce authority but cannot alter these values.
    std::vector<float> actionLower;
    std::vector<float> actionUpper;
    std::vector<float> maximumActionRate;

    float maximumPhaseAdvancePerStep = 0.04f;
    float phaseAlignmentBlend = 0.35f;
    float phaseAlignmentHuberDelta = 2.0f;
    float controlTimeStep = 0.02f;
};

enum class HyperPolicyCompileStatus : std::uint32_t {
    success = 0u,
    invalidTask,
    invalidPack,
    incompatibleContract,
    arithmeticOverflow,
    internalFailure,
};

struct HyperPolicyCompileDiagnostics {
    HyperPolicyCompileStatus status = HyperPolicyCompileStatus::success;
    std::uint64_t fingerprint = 0u;
    std::string element;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == HyperPolicyCompileStatus::success;
    }
};

struct HyperPolicyProgramLayout {
    std::uint32_t layerCount = 0u;
    std::uint32_t actorObservationCount = 0u;
    std::uint32_t actionCount = 0u;
    std::uint32_t coefficientCount = 0u;
    std::uint32_t knotCount = 0u;
    std::uint32_t referenceCount = 0u;
    std::uint32_t signatureCount = 0u;
    std::uint32_t eventCount = 0u;
    std::uint32_t contactTrackCount = 0u;
    std::uint32_t maximumRank = 0u;
    bool stochastic = false;
};

class CompiledHyperPolicyProgram {
public:
    CompiledHyperPolicyProgram() = default;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;
    [[nodiscard]] std::uint64_t taskFingerprint() const noexcept;
    [[nodiscard]] std::uint64_t revision() const noexcept;
    [[nodiscard]] const HyperPolicyProgramLayout& layout() const noexcept;
    [[nodiscard]] const MRHyperPolicyProgramHeaderGPU& header() const noexcept;
    [[nodiscard]] std::span<const MRHyperPolicyLayerGPU> layers() const noexcept;
    [[nodiscard]] std::span<const std::byte> arena() const noexcept;
    [[nodiscard]] std::span<const float> actionLower() const noexcept;
    [[nodiscard]] std::span<const float> actionUpper() const noexcept;
    [[nodiscard]] std::span<const float> maximumActionRate() const noexcept;

private:
    struct Storage;
    std::shared_ptr<const Storage> storage_;

    friend HyperPolicyCompileDiagnostics compileHyperPolicyProgram(
        const HyperPolicyPack&,
        const CompiledTaskProgram&,
        CompiledHyperPolicyProgram&
    );
};

[[nodiscard]] HyperPolicyCompileDiagnostics compileHyperPolicyProgram(
    const HyperPolicyPack& pack,
    const CompiledTaskProgram& task,
    CompiledHyperPolicyProgram& output
);

[[nodiscard]] const char* hyperPolicyCompileStatusName(
    HyperPolicyCompileStatus status
) noexcept;

} // namespace metalrobo
