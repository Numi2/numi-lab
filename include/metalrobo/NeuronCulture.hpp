#pragma once

#include "metalrobo/neuron_culture_gpu.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

inline constexpr std::uint32_t kNeuronCulturePackFormatVersion = 1u;

struct NeuronCultureGrowthPack {
    std::uint32_t width = 128u;
    std::uint32_t height = 128u;
    std::uint32_t stage = 1u;
    std::uint32_t newtonIterations = 4u;
    float timestep = 0.01f;
    float phaseMobility = 1.0f;
    float interfaceCoefficient = 0.18f;
    float tubulinDiffusion = 0.12f;
    float tubulinDecay = 0.001f;
    float tubulinSource = 0.04f;
    float growthDrive = 0.08f;
    float newtonTolerance = 1.0e-4f;
};

struct NeuronCultureNetworkPack {
    float neuralTimestepSeconds = 0.001f;
    float membraneTimeConstantSeconds = 0.020f;
    float restingPotential = -65.0f;
    float resetPotential = -68.0f;
    float thresholdPotential = -50.0f;
    float refractorySeconds = 0.002f;
    float traceTimeConstantSeconds = 0.020f;
    float depressionRecoverySeconds = 0.800f;
    float stdpPotentiation = 0.004f;
    float stdpDepression = 0.005f;
    float minimumWeight = 0.0f;
    float maximumWeight = 1.0f;
};

struct NeuronCulturePack {
    std::string id;
    std::uint32_t formatVersion = kNeuronCulturePackFormatVersion;
    std::uint64_t seed = 1u;
    std::string source;
    std::string sourceRevision;
    std::string sourceLicense;
    NeuronCultureGrowthPack growth;
    NeuronCultureNetworkPack network;
    std::vector<MRNeuronCultureNeuronGPU> neurons;
    std::vector<MRNeuronCultureSynapseGPU> synapses;
    std::vector<MRNeuronCultureElectrodeGPU> electrodes;
};

enum class NeuronCultureCompileStatus : std::uint32_t {
    success = 0u,
    invalidPack,
    invalidGrowth,
    invalidNetwork,
    invalidTopology,
    capacityExceeded,
    arithmeticOverflow,
};

struct NeuronCultureCompileDiagnostics {
    NeuronCultureCompileStatus status = NeuronCultureCompileStatus::success;
    std::string element;
    std::string message;
    [[nodiscard]] bool succeeded() const noexcept {
        return status == NeuronCultureCompileStatus::success;
    }
};

class CompiledNeuronCulture {
public:
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;
    [[nodiscard]] const MRNeuronCultureHeaderGPU& header() const noexcept;
    [[nodiscard]] const MRNeuronCultureGrowthGPU& growth() const noexcept;
    [[nodiscard]] std::span<const MRNeuronCultureNeuronGPU> neurons() const noexcept;
    [[nodiscard]] std::span<const MRNeuronCultureSynapseGPU> synapses() const noexcept;
    [[nodiscard]] std::span<const MRNeuronCultureElectrodeGPU> electrodes() const noexcept;

private:
    MRNeuronCultureHeaderGPU header_{};
    MRNeuronCultureGrowthGPU growth_{};
    std::vector<MRNeuronCultureNeuronGPU> neurons_;
    std::vector<MRNeuronCultureSynapseGPU> synapses_;
    std::vector<MRNeuronCultureElectrodeGPU> electrodes_;

    friend NeuronCultureCompileDiagnostics compileNeuronCulture(
        const NeuronCulturePack&, CompiledNeuronCulture&);
};

[[nodiscard]] NeuronCultureCompileDiagnostics compileNeuronCulture(
    const NeuronCulturePack& pack,
    CompiledNeuronCulture& output
);

[[nodiscard]] NeuronCulturePack makePotterReferenceCulture(
    std::uint32_t neuronCount = 1000u,
    std::uint32_t synapseCount = 50000u,
    std::uint64_t seed = 2056u
);

struct NeuronCultureState {
    std::vector<float> membrane;
    std::vector<float> refractory;
    std::vector<float> preTrace;
    std::vector<float> postTrace;
    std::vector<float> weights;
    std::vector<float> depression;
    std::vector<std::uint32_t> spikes;
    std::vector<std::uint32_t> spikeHistory;
    std::vector<std::uint32_t> electrodeSpikeCounts;
    std::vector<float> phase;
    std::vector<float> tubulin;
    std::uint64_t tick = 0u;
    std::uint64_t growthIteration = 0u;
};

class NeuronCultureReference {
public:
    explicit NeuronCultureReference(const CompiledNeuronCulture& culture);
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] const NeuronCultureState& accepted() const noexcept;
    [[nodiscard]] const NeuronCultureState& prepared() const noexcept;
    [[nodiscard]] bool prepareTicks(
        std::uint32_t tickCount,
        std::uint32_t stimulationElectrode,
        float stimulationCurrent
    );
    [[nodiscard]] bool prepareGrowth(std::uint32_t iterationCount);
    [[nodiscard]] bool publishPrepared() noexcept;
    void rejectPrepared() noexcept;

private:
    const CompiledNeuronCulture* culture_ = nullptr;
    NeuronCultureState accepted_;
    NeuronCultureState prepared_;
    bool hasPrepared_ = false;
};

[[nodiscard]] const char* neuronCultureCompileStatusName(
    NeuronCultureCompileStatus status) noexcept;

} // namespace metalrobo
