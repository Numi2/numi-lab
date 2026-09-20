#pragma once

#include "metalrobo/neuron_culture_gpu.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

inline constexpr std::uint32_t kNeuronCulturePackFormatVersion = 1u;
inline constexpr std::uint32_t kNeuronCultureMaximumStimulusPulses = 4096u;
inline constexpr std::uint32_t kNeuronCultureMaximumWindowTicks = 100000u;
inline constexpr float kPotterReferenceSynapticCurrentScale = 175000.0f;

enum class NeuronCultureStimulusSource : std::uint32_t {
    authored = 1u,
    contextProbe = 2u,
    patternedTraining = 3u,
    randomBackground = 4u,
    numanXSupport = 5u,
};

struct NeuronCultureStimulusPulse {
    std::uint32_t electrode = 0u;
    std::uint32_t startTick = 0u;
    std::uint32_t durationTicks = 1u;
    NeuronCultureStimulusSource source = NeuronCultureStimulusSource::authored;
    float current = 0.0f;
    std::uint64_t sourceFingerprint = 0u;
};

struct NeuronCultureWindowRequest {
    std::uint64_t cultureFingerprint = 0u;
    std::uint64_t rootFingerprint = 0u;
    std::uint32_t tickCount = 0u;
    std::uint32_t recordingStartTick = 0u;
    std::uint32_t recordingDurationTicks = 0u;
    // Experiment-only ablations freeze weights without changing culture
    // identity, topology, initial state, depression, or spike dynamics.
    bool plasticityEnabled = true;
    std::vector<NeuronCultureStimulusPulse> pulses;
};

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
    // Chao et al. Text S1 DynamicStdpSynapse parameters. These are the A+/A-
    // coefficients in the bounded normalized update, not additive dW values.
    float stdpPotentiation = 0.5f;
    float stdpDepression = 0.525f;
    float minimumWeight = 0.0f;
    float maximumWeight = 1.0f;
    // Current delivered by one unit-weight, fully recovered presynaptic spike.
    // At the canonical 1 ms timestep and maximum weight 0.1, the reference
    // value produces a 17.5 mV recovered EPSP-equivalent contribution.
    float synapticCurrentScale = kPotterReferenceSynapticCurrentScale;
    float preSpikeSuppressionTimeConstantSeconds = 0.034f;
    float postSpikeSuppressionTimeConstantSeconds = 0.075f;
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
    [[nodiscard]] const std::string& id() const noexcept;
    [[nodiscard]] const std::string& source() const noexcept;
    [[nodiscard]] const std::string& sourceRevision() const noexcept;
    [[nodiscard]] const std::string& sourceLicense() const noexcept;

private:
    MRNeuronCultureHeaderGPU header_{};
    MRNeuronCultureGrowthGPU growth_{};
    std::vector<MRNeuronCultureNeuronGPU> neurons_;
    std::vector<MRNeuronCultureSynapseGPU> synapses_;
    std::vector<MRNeuronCultureElectrodeGPU> electrodes_;
    std::string id_;
    std::string source_;
    std::string sourceRevision_;
    std::string sourceLicense_;

    friend NeuronCultureCompileDiagnostics compileNeuronCulture(
        const NeuronCulturePack&, CompiledNeuronCulture&);
    friend class NeuronCultureArtifactAccess;
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

[[nodiscard]] bool validateNeuronCultureWindow(
    const CompiledNeuronCulture& culture,
    const NeuronCultureWindowRequest& request
) noexcept;

[[nodiscard]] bool neuronCultureStimulusCurrents(
    const CompiledNeuronCulture& culture,
    const NeuronCultureWindowRequest& request,
    std::uint32_t tickOffset,
    std::span<float> electrodeCurrents
) noexcept;

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
    // Monotonic accepted publication generation. It is checkpoint authority,
    // not inferred from ticks because one accepted window may advance many.
    std::uint64_t generation = 0u;
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
    [[nodiscard]] bool prepareWindow(const NeuronCultureWindowRequest& request);
    [[nodiscard]] bool prepareGrowth(std::uint32_t iterationCount);
    [[nodiscard]] bool publishPrepared() noexcept;
    void rejectPrepared() noexcept;
    [[nodiscard]] bool restoreAccepted(const NeuronCultureState& state);

private:
    const CompiledNeuronCulture* culture_ = nullptr;
    NeuronCultureState accepted_;
    NeuronCultureState prepared_;
    bool hasPrepared_ = false;
};

[[nodiscard]] const char* neuronCultureCompileStatusName(
    NeuronCultureCompileStatus status) noexcept;

} // namespace metalrobo
