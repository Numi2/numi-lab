#include "metalrobo/NeuronCulture.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <tuple>

namespace metalrobo {
namespace {

constexpr std::uint64_t kFNVOffset = 14695981039346656037ull;
constexpr std::uint64_t kFNVPrime = 1099511628211ull;

void fnvBytes(std::uint64_t& hash, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0u; i < size; ++i) {
        hash ^= bytes[i];
        hash *= kFNVPrime;
    }
}

template <typename T>
void fnvValue(std::uint64_t& hash, const T& value) {
    fnvBytes(hash, &value, sizeof(value));
}

void fnvString(std::uint64_t& hash, const std::string& value) {
    const std::uint64_t size = value.size();
    fnvValue(hash, size);
    fnvBytes(hash, value.data(), value.size());
}

bool finite(float value) noexcept { return std::isfinite(value); }

std::uint64_t splitmix64(std::uint64_t& state) noexcept {
    state += 0x9e3779b97f4a7c15ull;
    std::uint64_t z = state;
    z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31u);
}

float uniform01(std::uint64_t& state) noexcept {
    return static_cast<float>((splitmix64(state) >> 40u) * (1.0 / 16777216.0));
}

bool checkedCells(std::uint32_t width, std::uint32_t height, std::size_t& out) {
    if (width == 0u || height == 0u) return false;
    const std::uint64_t cells = static_cast<std::uint64_t>(width) * height;
    if (cells > 16u * 1024u * 1024u || cells > std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    out = static_cast<std::size_t>(cells);
    return true;
}

float clamp01(float x) noexcept { return std::clamp(x, 0.0f, 1.0f); }

void initializeState(const CompiledNeuronCulture& culture, NeuronCultureState& state) {
    const auto& h = culture.header();
    state.membrane.assign(h.neuronCount, h.restingPotential);
    state.refractory.assign(h.neuronCount, 0.0f);
    state.preTrace.assign(h.neuronCount, 0.0f);
    state.postTrace.assign(h.neuronCount, 0.0f);
    state.weights.resize(h.synapseCount);
    state.depression.assign(h.synapseCount, 1.0f);
    state.spikes.assign(h.neuronCount, 0u);
    state.spikeHistory.assign(static_cast<std::size_t>(h.neuronCount) * 256u, 0u);
    state.electrodeSpikeCounts.assign(h.electrodeCount, 0u);
    for (std::size_t i = 0u; i < culture.synapses().size(); ++i) {
        state.weights[i] = culture.synapses()[i].initialWeight;
    }
    const std::size_t cells = static_cast<std::size_t>(h.growthWidth) * h.growthHeight;
    state.phase.assign(cells, 0.0f);
    state.tubulin.assign(cells, 0.0f);

    const float sx = static_cast<float>(h.growthWidth - 1u) / 3.0f;
    const float sy = static_cast<float>(h.growthHeight - 1u) / 3.0f;
    const float radius = std::max(2.0f, 0.018f * std::min(h.growthWidth, h.growthHeight));
    for (const auto& neuron : culture.neurons()) {
        const float cx = neuron.x * sx;
        const float cy = neuron.y * sy;
        const int minX = std::max(0, static_cast<int>(std::floor(cx - 2.0f * radius)));
        const int maxX = std::min(static_cast<int>(h.growthWidth) - 1,
                                  static_cast<int>(std::ceil(cx + 2.0f * radius)));
        const int minY = std::max(0, static_cast<int>(std::floor(cy - 2.0f * radius)));
        const int maxY = std::min(static_cast<int>(h.growthHeight) - 1,
                                  static_cast<int>(std::ceil(cy + 2.0f * radius)));
        for (int y = minY; y <= maxY; ++y) {
            for (int x = minX; x <= maxX; ++x) {
                const float dx = static_cast<float>(x) - cx;
                const float dy = static_cast<float>(y) - cy;
                const float distance = std::sqrt(dx * dx + dy * dy);
                const float phi = 0.5f * (1.0f + std::tanh((radius - distance) * 0.5f));
                const std::size_t index = static_cast<std::size_t>(y) * h.growthWidth + x;
                state.phase[index] = std::max(state.phase[index], phi);
                state.tubulin[index] = std::max(state.tubulin[index], phi);
            }
        }
    }
}

} // namespace

bool CompiledNeuronCulture::valid() const noexcept {
    return header_.abiVersion == MR_NEURON_CULTURE_ABI_VERSION &&
        header_.structBytes == sizeof(MRNeuronCultureHeaderGPU) &&
        header_.cultureFingerprint != 0u && !neurons_.empty() &&
        header_.neuronCount == neurons_.size() &&
        header_.synapseCount == synapses_.size() &&
        header_.electrodeCount == electrodes_.size();
}

std::uint64_t CompiledNeuronCulture::fingerprint() const noexcept {
    return header_.cultureFingerprint;
}

const MRNeuronCultureHeaderGPU& CompiledNeuronCulture::header() const noexcept { return header_; }
const MRNeuronCultureGrowthGPU& CompiledNeuronCulture::growth() const noexcept { return growth_; }
std::span<const MRNeuronCultureNeuronGPU> CompiledNeuronCulture::neurons() const noexcept { return neurons_; }
std::span<const MRNeuronCultureSynapseGPU> CompiledNeuronCulture::synapses() const noexcept { return synapses_; }
std::span<const MRNeuronCultureElectrodeGPU> CompiledNeuronCulture::electrodes() const noexcept { return electrodes_; }

NeuronCultureCompileDiagnostics compileNeuronCulture(
    const NeuronCulturePack& pack,
    CompiledNeuronCulture& output
) {
    NeuronCultureCompileDiagnostics d;
    if (pack.formatVersion != kNeuronCulturePackFormatVersion || pack.id.empty() ||
        pack.source.empty() || pack.sourceRevision.empty() || pack.sourceLicense.empty() ||
        pack.seed == 0u) {
        d.status = NeuronCultureCompileStatus::invalidPack;
        d.element = "pack";
        d.message = "culture identity, provenance, version, and seed must be exact";
        return d;
    }
    std::size_t cells = 0u;
    const auto& g = pack.growth;
    if (!checkedCells(g.width, g.height, cells) || g.stage < 1u || g.stage > 4u ||
        g.newtonIterations == 0u || g.newtonIterations > 32u ||
        !finite(g.timestep) || g.timestep <= 0.0f || !finite(g.phaseMobility) ||
        g.phaseMobility <= 0.0f || !finite(g.interfaceCoefficient) ||
        g.interfaceCoefficient <= 0.0f || !finite(g.tubulinDiffusion) ||
        g.tubulinDiffusion < 0.0f || !finite(g.tubulinDecay) || g.tubulinDecay < 0.0f ||
        !finite(g.tubulinSource) || g.tubulinSource < 0.0f || !finite(g.growthDrive) ||
        !finite(g.newtonTolerance) || g.newtonTolerance <= 0.0f) {
        d.status = NeuronCultureCompileStatus::invalidGrowth;
        d.element = "growth";
        d.message = "growth grid and bounded implicit parameters are invalid";
        return d;
    }
    const auto& n = pack.network;
    const bool networkValid = finite(n.neuralTimestepSeconds) && n.neuralTimestepSeconds > 0.0f &&
        finite(n.membraneTimeConstantSeconds) && n.membraneTimeConstantSeconds > 0.0f &&
        finite(n.restingPotential) && finite(n.resetPotential) && finite(n.thresholdPotential) &&
        n.resetPotential < n.thresholdPotential && n.restingPotential < n.thresholdPotential &&
        finite(n.refractorySeconds) && n.refractorySeconds >= 0.0f &&
        finite(n.traceTimeConstantSeconds) && n.traceTimeConstantSeconds > 0.0f &&
        finite(n.depressionRecoverySeconds) && n.depressionRecoverySeconds > 0.0f &&
        finite(n.stdpPotentiation) && n.stdpPotentiation >= 0.0f &&
        finite(n.stdpDepression) && n.stdpDepression >= 0.0f &&
        finite(n.minimumWeight) && finite(n.maximumWeight) && n.minimumWeight <= n.maximumWeight;
    if (!networkValid) {
        d.status = NeuronCultureCompileStatus::invalidNetwork;
        d.element = "network";
        d.message = "LIF, depression, or STDP parameters are invalid";
        return d;
    }
    if (pack.neurons.empty() || pack.neurons.size() > 1'000'000u ||
        pack.synapses.size() > 16'000'000u || pack.electrodes.empty() ||
        pack.electrodes.size() > MR_NEURON_CULTURE_MAX_ELECTRODES) {
        d.status = NeuronCultureCompileStatus::capacityExceeded;
        d.element = "topology";
        d.message = "culture exceeds the compiled neuron, synapse, or MEA capacity";
        return d;
    }
    for (std::size_t i = 0u; i < pack.neurons.size(); ++i) {
        const auto& neuron = pack.neurons[i];
        if (!finite(neuron.x) || !finite(neuron.y) || neuron.x < 0.0f || neuron.x > 3.0f ||
            neuron.y < 0.0f || neuron.y > 3.0f || !finite(neuron.biasCurrent) ||
            !finite(neuron.capacitance) || neuron.capacitance <= 0.0f ||
            neuron.excitatory > 1u) {
            d.status = NeuronCultureCompileStatus::invalidTopology;
            d.element = "neuron[" + std::to_string(i) + "]";
            d.message = "neuron geometry or dynamics are invalid";
            return d;
        }
    }
    std::vector<MRNeuronCultureSynapseGPU> synapses = pack.synapses;
    for (std::size_t i = 0u; i < synapses.size(); ++i) {
        const auto& synapse = synapses[i];
        if (synapse.presynaptic >= pack.neurons.size() ||
            synapse.postsynaptic >= pack.neurons.size() ||
            synapse.presynaptic == synapse.postsynaptic || synapse.delayTicks > 255u ||
            synapse.plastic > 1u || !finite(synapse.initialWeight) ||
            synapse.initialWeight < n.minimumWeight || synapse.initialWeight > n.maximumWeight ||
            !finite(synapse.depressionUse) || synapse.depressionUse < 0.0f ||
            synapse.depressionUse > 1.0f) {
            d.status = NeuronCultureCompileStatus::invalidTopology;
            d.element = "synapse[" + std::to_string(i) + "]";
            d.message = "synapse identity, weight, delay, or depression is invalid";
            return d;
        }
    }
    std::stable_sort(synapses.begin(), synapses.end(), [](const auto& a, const auto& b) {
        return std::tie(a.postsynaptic, a.presynaptic) < std::tie(b.postsynaptic, b.presynaptic);
    });
    std::vector<MRNeuronCultureNeuronGPU> neurons = pack.neurons;
    std::size_t cursor = 0u;
    for (std::size_t neuron = 0u; neuron < neurons.size(); ++neuron) {
        const std::size_t begin = cursor;
        while (cursor < synapses.size() && synapses[cursor].postsynaptic == neuron) ++cursor;
        if (begin > std::numeric_limits<std::uint32_t>::max() ||
            cursor - begin > std::numeric_limits<std::uint32_t>::max()) {
            d.status = NeuronCultureCompileStatus::arithmeticOverflow;
            d.element = "incoming CSR";
            d.message = "incoming synapse range exceeds the Metal index ABI";
            return d;
        }
        neurons[neuron].incomingBegin = static_cast<std::uint32_t>(begin);
        neurons[neuron].incomingCount = static_cast<std::uint32_t>(cursor - begin);
    }
    for (std::size_t i = 0u; i < pack.electrodes.size(); ++i) {
        const auto& e = pack.electrodes[i];
        if (!finite(e.x) || !finite(e.y) || e.x < 0.0f || e.x > 3.0f || e.y < 0.0f ||
            e.y > 3.0f || !finite(e.recordingRadius) || e.recordingRadius <= 0.0f ||
            !finite(e.stimulationRadius) || e.stimulationRadius <= 0.0f || e.active > 1u) {
            d.status = NeuronCultureCompileStatus::invalidTopology;
            d.element = "electrode[" + std::to_string(i) + "]";
            d.message = "MEA geometry is invalid";
            return d;
        }
    }

    MRNeuronCultureHeaderGPU header{};
    header.abiVersion = MR_NEURON_CULTURE_ABI_VERSION;
    header.structBytes = sizeof(header);
    header.neuronCount = static_cast<std::uint32_t>(neurons.size());
    header.synapseCount = static_cast<std::uint32_t>(synapses.size());
    header.electrodeCount = static_cast<std::uint32_t>(pack.electrodes.size());
    header.growthWidth = g.width;
    header.growthHeight = g.height;
    header.seed = pack.seed;
    header.neuralTimestepSeconds = n.neuralTimestepSeconds;
    header.membraneTimeConstantSeconds = n.membraneTimeConstantSeconds;
    header.restingPotential = n.restingPotential;
    header.resetPotential = n.resetPotential;
    header.thresholdPotential = n.thresholdPotential;
    header.refractorySeconds = n.refractorySeconds;
    header.traceTimeConstantSeconds = n.traceTimeConstantSeconds;
    header.depressionRecoverySeconds = n.depressionRecoverySeconds;
    header.stdpPotentiation = n.stdpPotentiation;
    header.stdpDepression = n.stdpDepression;
    header.minimumWeight = n.minimumWeight;
    header.maximumWeight = n.maximumWeight;

    MRNeuronCultureGrowthGPU growth{};
    growth.width = g.width;
    growth.height = g.height;
    growth.stage = g.stage;
    growth.timestep = g.timestep;
    growth.phaseMobility = g.phaseMobility;
    growth.interfaceCoefficient = g.interfaceCoefficient;
    growth.tubulinDiffusion = g.tubulinDiffusion;
    growth.tubulinDecay = g.tubulinDecay;
    growth.tubulinSource = g.tubulinSource;
    growth.growthDrive = g.growthDrive;
    growth.newtonTolerance = g.newtonTolerance;
    growth.newtonIterations = g.newtonIterations;

    std::uint64_t hash = kFNVOffset;
    fnvString(hash, pack.id);
    fnvValue(hash, pack.formatVersion);
    fnvValue(hash, pack.seed);
    fnvString(hash, pack.source);
    fnvString(hash, pack.sourceRevision);
    fnvString(hash, pack.sourceLicense);
    MRNeuronCultureHeaderGPU identityHeader = header;
    identityHeader.cultureFingerprint = 0u;
    fnvValue(hash, identityHeader);
    fnvValue(hash, growth);
    fnvBytes(hash, neurons.data(), neurons.size() * sizeof(neurons.front()));
    fnvBytes(hash, synapses.data(), synapses.size() * sizeof(synapses.front()));
    fnvBytes(hash, pack.electrodes.data(), pack.electrodes.size() * sizeof(pack.electrodes.front()));
    if (hash == 0u) hash = kFNVOffset;
    header.cultureFingerprint = hash;

    CompiledNeuronCulture candidate;
    candidate.header_ = header;
    candidate.growth_ = growth;
    candidate.neurons_ = std::move(neurons);
    candidate.synapses_ = std::move(synapses);
    candidate.electrodes_ = pack.electrodes;
    output = std::move(candidate);
    return d;
}

NeuronCulturePack makePotterReferenceCulture(
    std::uint32_t neuronCount,
    std::uint32_t synapseCount,
    std::uint64_t seed
) {
    NeuronCulturePack pack;
    pack.id = "potter-embodied-mea-synthetic-v1";
    pack.seed = seed;
    pack.source = "Chao-Bakkum-Potter-PLoS-Comput-Biol-2008-and-Qian-et-al-Sci-Rep-2022";
    pack.sourceRevision = "doi:10.1371/journal.pcbi.1000042+doi:10.1038/s41598-022-12073-z";
    pack.sourceLicense = "CC-BY-4.0-equation-level-reimplementation";
    pack.growth.width = neuronCount >= 1000u ? 384u : 96u;
    pack.growth.height = pack.growth.width;
    pack.neurons.resize(neuronCount);
    std::uint64_t rng = seed;
    for (std::uint32_t i = 0u; i < neuronCount; ++i) {
        auto& neuron = pack.neurons[i];
        neuron.x = 3.0f * uniform01(rng);
        neuron.y = 3.0f * uniform01(rng);
        neuron.biasCurrent = 820.0f + 260.0f * uniform01(rng);
        neuron.capacitance = 1.0f;
        neuron.excitatory = i < (7u * neuronCount) / 10u ? 1u : 0u;
    }
    pack.synapses.reserve(synapseCount);
    while (pack.synapses.size() < synapseCount && neuronCount > 1u) {
        const std::uint32_t pre = static_cast<std::uint32_t>(splitmix64(rng) % neuronCount);
        std::uint32_t post = static_cast<std::uint32_t>(splitmix64(rng) % neuronCount);
        if (pre == post) post = (post + 1u) % neuronCount;
        MRNeuronCultureSynapseGPU synapse{};
        synapse.presynaptic = pre;
        synapse.postsynaptic = post;
        synapse.delayTicks = 1u + static_cast<std::uint32_t>(splitmix64(rng) % 5u);
        synapse.plastic = pack.neurons[pre].excitatory;
        synapse.initialWeight = pack.neurons[pre].excitatory ?
            0.08f + 0.12f * uniform01(rng) : 0.15f + 0.10f * uniform01(rng);
        synapse.depressionUse = 0.04f + 0.08f * uniform01(rng);
        pack.synapses.push_back(synapse);
    }
    for (std::uint32_t row = 0u; row < 8u; ++row) {
        for (std::uint32_t column = 0u; column < 8u; ++column) {
            if ((row == 0u && column == 0u) || (row == 7u && column == 7u) ||
                (row == 0u && column == 7u) || (row == 7u && column == 0u)) {
                continue;
            }
            MRNeuronCultureElectrodeGPU e{};
            e.x = 3.0f * (static_cast<float>(column) + 0.5f) / 8.0f;
            e.y = 3.0f * (static_cast<float>(row) + 0.5f) / 8.0f;
            e.recordingRadius = 0.28f;
            e.stimulationRadius = 0.34f;
            e.active = 1u;
            pack.electrodes.push_back(e);
        }
    }
    return pack;
}

NeuronCultureReference::NeuronCultureReference(const CompiledNeuronCulture& culture)
    : culture_(culture.valid() ? &culture : nullptr) {
    if (culture_) {
        initializeState(culture, accepted_);
        prepared_ = accepted_;
    }
}

bool NeuronCultureReference::valid() const noexcept { return culture_ != nullptr; }
const NeuronCultureState& NeuronCultureReference::accepted() const noexcept { return accepted_; }
const NeuronCultureState& NeuronCultureReference::prepared() const noexcept { return prepared_; }

bool NeuronCultureReference::prepareTicks(
    std::uint32_t tickCount,
    std::uint32_t stimulationElectrode,
    float stimulationCurrent
) {
    if (!culture_ || tickCount == 0u || !finite(stimulationCurrent) ||
        (stimulationElectrode != std::numeric_limits<std::uint32_t>::max() &&
         stimulationElectrode >= culture_->electrodes().size())) return false;
    prepared_ = accepted_;
    std::vector<std::uint32_t> nextSpikes(prepared_.spikes.size(), 0u);
    std::vector<float> nextMembrane(prepared_.membrane.size());
    const auto& h = culture_->header();
    const float dt = h.neuralTimestepSeconds;
    const float traceDecay = std::exp(-dt / h.traceTimeConstantSeconds);
    const float depressionRecovery = std::min(1.0f, dt / h.depressionRecoverySeconds);
    for (std::uint32_t tick = 0u; tick < tickCount; ++tick) {
        std::fill(nextSpikes.begin(), nextSpikes.end(), 0u);
        for (std::size_t i = 0u; i < culture_->neurons().size(); ++i) {
            const auto& neuron = culture_->neurons()[i];
            float current = neuron.biasCurrent;
            for (std::uint32_t j = 0u; j < neuron.incomingCount; ++j) {
                const std::size_t edge = neuron.incomingBegin + j;
                const auto& synapse = culture_->synapses()[edge];
                const std::uint64_t delayedTick = prepared_.tick + 256u - synapse.delayTicks;
                const std::size_t historyIndex = static_cast<std::size_t>(delayedTick & 255u) *
                    h.neuronCount + synapse.presynaptic;
                if (prepared_.spikeHistory[historyIndex] != 0u) {
                    const float sign = culture_->neurons()[synapse.presynaptic].excitatory ? 1.0f : -1.0f;
                    current += sign * prepared_.weights[edge] * prepared_.depression[edge] * 45.0f;
                }
            }
            if (stimulationElectrode < culture_->electrodes().size()) {
                const auto& e = culture_->electrodes()[stimulationElectrode];
                const float dx = neuron.x - e.x;
                const float dy = neuron.y - e.y;
                if (dx * dx + dy * dy <= e.stimulationRadius * e.stimulationRadius) {
                    current += stimulationCurrent;
                }
            }
            float v = prepared_.membrane[i];
            float refractory = std::max(0.0f, prepared_.refractory[i] - dt);
            if (refractory > 0.0f) {
                v = h.resetPotential;
            } else {
                v += dt * ((h.restingPotential - v) / h.membraneTimeConstantSeconds +
                           current / neuron.capacitance);
                if (v >= h.thresholdPotential) {
                    nextSpikes[i] = 1u;
                    v = h.resetPotential;
                    refractory = h.refractorySeconds;
                }
            }
            nextMembrane[i] = v;
            prepared_.refractory[i] = refractory;
        }
        for (std::size_t i = 0u; i < prepared_.spikes.size(); ++i) {
            prepared_.preTrace[i] = prepared_.preTrace[i] * traceDecay + nextSpikes[i];
            prepared_.postTrace[i] = prepared_.postTrace[i] * traceDecay + nextSpikes[i];
        }
        for (std::size_t edge = 0u; edge < culture_->synapses().size(); ++edge) {
            const auto& synapse = culture_->synapses()[edge];
            float depression = prepared_.depression[edge] +
                (1.0f - prepared_.depression[edge]) * depressionRecovery;
            if (nextSpikes[synapse.presynaptic]) {
                depression *= (1.0f - synapse.depressionUse);
            }
            prepared_.depression[edge] = clamp01(depression);
            if (synapse.plastic) {
                float delta = 0.0f;
                if (nextSpikes[synapse.postsynaptic]) {
                    delta += h.stdpPotentiation * prepared_.preTrace[synapse.presynaptic];
                }
                if (nextSpikes[synapse.presynaptic]) {
                    delta -= h.stdpDepression * prepared_.postTrace[synapse.postsynaptic];
                }
                prepared_.weights[edge] = std::clamp(
                    prepared_.weights[edge] + delta, h.minimumWeight, h.maximumWeight);
            }
        }
        prepared_.membrane.swap(nextMembrane);
        prepared_.spikes.swap(nextSpikes);
        const std::size_t historyBase = static_cast<std::size_t>(prepared_.tick & 255u) *
            h.neuronCount;
        std::copy(prepared_.spikes.begin(), prepared_.spikes.end(),
                  prepared_.spikeHistory.begin() + historyBase);
        for (std::size_t electrode = 0u; electrode < culture_->electrodes().size(); ++electrode) {
            const auto& e = culture_->electrodes()[electrode];
            std::uint32_t count = 0u;
            for (std::size_t neuron = 0u; neuron < culture_->neurons().size(); ++neuron) {
                const float dx = culture_->neurons()[neuron].x - e.x;
                const float dy = culture_->neurons()[neuron].y - e.y;
                if (prepared_.spikes[neuron] && dx * dx + dy * dy <= e.recordingRadius * e.recordingRadius) {
                    ++count;
                }
            }
            prepared_.electrodeSpikeCounts[electrode] += count;
        }
        ++prepared_.tick;
    }
    hasPrepared_ = true;
    return true;
}

bool NeuronCultureReference::prepareGrowth(std::uint32_t iterationCount) {
    if (!culture_ || iterationCount == 0u) return false;
    prepared_ = accepted_;
    const auto& g = culture_->growth();
    const std::uint32_t width = g.width;
    const std::uint32_t height = g.height;
    std::vector<float> phaseNext(prepared_.phase.size());
    std::vector<float> tubulinNext(prepared_.tubulin.size());
    auto at = [width, height](const std::vector<float>& field, int x, int y) {
        x = std::clamp(x, 0, static_cast<int>(width) - 1);
        y = std::clamp(y, 0, static_cast<int>(height) - 1);
        return field[static_cast<std::size_t>(y) * width + x];
    };
    for (std::uint32_t iteration = 0u; iteration < iterationCount; ++iteration) {
        for (std::uint32_t y = 0u; y < height; ++y) {
            for (std::uint32_t x = 0u; x < width; ++x) {
                const std::size_t index = static_cast<std::size_t>(y) * width + x;
                const float oldPhi = prepared_.phase[index];
                const float oldTubulin = prepared_.tubulin[index];
                const float lapPhi = at(prepared_.phase, x - 1, y) +
                    at(prepared_.phase, x + 1, y) + at(prepared_.phase, x, y - 1) +
                    at(prepared_.phase, x, y + 1) - 4.0f * oldPhi;
                const float lapTubulin = at(prepared_.tubulin, x - 1, y) +
                    at(prepared_.tubulin, x + 1, y) + at(prepared_.tubulin, x, y - 1) +
                    at(prepared_.tubulin, x, y + 1) - 4.0f * oldTubulin;
                float phi = oldPhi;
                for (std::uint32_t k = 0u; k < g.newtonIterations; ++k) {
                    const float reaction = phi * (1.0f - phi) * (phi - 0.5f);
                    const float drive = g.growthDrive * oldTubulin * phi * (1.0f - phi);
                    const float residual = phi - oldPhi - g.timestep * g.phaseMobility *
                        (g.interfaceCoefficient * g.interfaceCoefficient * lapPhi + reaction + drive);
                    const float derivative = 1.0f - g.timestep * g.phaseMobility *
                        ((-3.0f * phi * phi + 3.0f * phi - 0.5f) +
                         g.growthDrive * oldTubulin * (1.0f - 2.0f * phi));
                    if (!finite(derivative) || std::abs(derivative) < 1.0e-8f) return false;
                    const float step = residual / derivative;
                    phi = clamp01(phi - step);
                    if (std::abs(step) <= g.newtonTolerance) break;
                }
                phaseNext[index] = phi;
                const float source = g.tubulinSource * phi;
                const float denominator = 1.0f + g.timestep * g.tubulinDecay;
                tubulinNext[index] = std::max(0.0f,
                    (oldTubulin + g.timestep * (g.tubulinDiffusion * lapTubulin + source)) /
                    denominator);
            }
        }
        prepared_.phase.swap(phaseNext);
        prepared_.tubulin.swap(tubulinNext);
        ++prepared_.growthIteration;
    }
    hasPrepared_ = true;
    return true;
}

bool NeuronCultureReference::publishPrepared() noexcept {
    if (!hasPrepared_) return false;
    accepted_ = prepared_;
    hasPrepared_ = false;
    return true;
}

void NeuronCultureReference::rejectPrepared() noexcept {
    prepared_ = accepted_;
    hasPrepared_ = false;
}

const char* neuronCultureCompileStatusName(NeuronCultureCompileStatus status) noexcept {
    switch (status) {
        case NeuronCultureCompileStatus::success: return "success";
        case NeuronCultureCompileStatus::invalidPack: return "invalid_pack";
        case NeuronCultureCompileStatus::invalidGrowth: return "invalid_growth";
        case NeuronCultureCompileStatus::invalidNetwork: return "invalid_network";
        case NeuronCultureCompileStatus::invalidTopology: return "invalid_topology";
        case NeuronCultureCompileStatus::capacityExceeded: return "capacity_exceeded";
        case NeuronCultureCompileStatus::arithmeticOverflow: return "arithmetic_overflow";
    }
    return "unknown";
}

} // namespace metalrobo
