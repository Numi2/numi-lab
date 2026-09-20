#pragma once

#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/numi_human_extensor_hood_gpu.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

struct NumiHumanExtensorHoodMetalSource {
    std::span<const MRNumiHumanExtensorHoodRayGPU> rays{};
    std::span<const MRNumiHumanExtensorHoodNodeGPU> nodes{};
    std::span<const MRNumiHumanExtensorHoodElementGPU> elements{};
    std::span<const MRNumiHumanExtensorHoodInputGPU> inputs{};
    std::uint32_t environmentCount = 0u;
};

struct NumiHumanExtensorHoodMetalConfiguration {
    std::filesystem::path metallib;
    // Tension-only elements change active set near zero extension. Keep the
    // solve bounded because float32 residuals plateau after those switches.
    std::uint32_t maximumIterations = 128u;
    std::uint32_t maximumLineSearchSteps = 16u;
    // FP32 Metal equilibrium gate; the aggregate closure gate below remains
    // independent and scales across the ray's free nodes.
    // Absolute equilibrium tolerance for this sub-newton hand network. This
    // remains below one percent of the active tendon loads while clearing the
    // measured float32 active-set floor.
    float forceToleranceNewtons = 2.5e-3f;
    float minimumLengthMeters = 1.0e-6f;
    float diagonalRegularization = 1.0e-6f;
    float armijoFraction = 1.0e-4f;
    float foundationStiffnessNewtonsPerMeter = 250.0f;
};

struct NumiHumanExtensorHoodMetalDiagnostics {
    bool initialized = false;
    std::uint32_t encodedPassCount = 0u;
    std::uint32_t abortCount = 0u;
    std::uint32_t successfulRayCount = 0u;
    std::uint32_t acceptedRayCount = 0u;
    std::uint32_t firstFailingRay = MR_INVALID_INDEX;
    std::uint32_t firstFailureStatus = MR_NUMI_HUMAN_EXTENSOR_HOOD_SUCCESS;
    std::uint32_t firstFailureCompletedIterations = 0u;
    float maximumFreeNodeResidualNewtons = 0.0f;
    float maximumForceClosureResidualNewtons = 0.0f;
    float maximumMomentClosureResidualNewtonMeters = 0.0f;
    float maximumTensionNewtons = 0.0f;
    float maximumFreeNodeDisplacementMeters = 0.0f;
    float maximumEngineeringStrain = 0.0f;
    float maximumGeneralizedCorrection = 0.0f;
    std::vector<mr_float4> solvedNodePositions;
    std::uint64_t fingerprint = 0u;
    std::string message;
};

// Same-command-buffer, quasi-static extensor-expansion owner. It keeps each
// muscle's proximal source route, removes only the declared site-only distal
// suffix, and replaces that suffix with the solved hood and bone reactions.
class NumiHumanExtensorHoodMetalAdapter {
public:
    NumiHumanExtensorHoodMetalAdapter();
    ~NumiHumanExtensorHoodMetalAdapter();
    NumiHumanExtensorHoodMetalAdapter(
        NumiHumanExtensorHoodMetalAdapter&&) noexcept;
    NumiHumanExtensorHoodMetalAdapter& operator=(
        NumiHumanExtensorHoodMetalAdapter&&) noexcept;
    NumiHumanExtensorHoodMetalAdapter(
        const NumiHumanExtensorHoodMetalAdapter&) = delete;
    NumiHumanExtensorHoodMetalAdapter& operator=(
        const NumiHumanExtensorHoodMetalAdapter&) = delete;

    [[nodiscard]] bool initialize(
        const NumiHumanExtensorHoodMetalSource& source,
        const NumiHumanExtensorHoodMetalConfiguration& configuration
    );
    [[nodiscard]] MetalNumiHumanTendonLoadProgram program() noexcept;
    [[nodiscard]] NumiHumanExtensorHoodMetalDiagnostics diagnostics() const noexcept;

private:
    [[nodiscard]] bool encodePreDynamics(
        const MetalNumiHumanTendonLoadPass& pass);
    [[nodiscard]] bool encodePostValidation(
        const MetalNumiHumanTendonLoadPass& pass);
    void abort(void* commandBuffer) noexcept;
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace metalrobo
