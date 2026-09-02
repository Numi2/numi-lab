#import <Metal/Metal.h>

#include "metalrobo/MetalNeuronCulture.hpp"
#include "metalrobo/NeuronCulture.hpp"
#include "metalrobo/NeuronCultureEmbodiment.hpp"
#include "metalrobo/NeuronCultureArtifacts.hpp"
#include "metalrobo/NeuronCultureProtocol.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <initializer_list>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sstream>
#include <vector>
#include <unistd.h>

namespace {

struct CompletionObservation {
    std::atomic<std::uint32_t> count{0u};
    std::atomic<metalrobo::MetalNeuronCultureStatus> status{
        metalrobo::MetalNeuronCultureStatus::pending};
};

void observeCompletion(
    void* raw, const metalrobo::MetalNeuronCultureStatus status) noexcept {
    auto* observation = static_cast<CompletionObservation*>(raw);
    if (observation == nullptr) return;
    observation->status.store(status, std::memory_order_release);
    observation->count.fetch_add(1u, std::memory_order_acq_rel);
}

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

std::string requiredEnvironment(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        throw std::runtime_error(std::string("missing canonical run identity: ") + name);
    }
    return value;
}

std::string jsonEscape(std::string_view value) {
    std::ostringstream output;
    for (const unsigned char byte : value) {
        switch (byte) {
            case '\"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (byte < 0x20u) {
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<unsigned>(byte)
                           << std::dec;
                } else {
                    output << static_cast<char>(byte);
                }
        }
    }
    return output.str();
}

template <typename T>
bool exact(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
        std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

bool close(const std::vector<float>& a, const std::vector<float>& b, float tolerance) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0u; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]) ||
            std::abs(a[i] - b[i]) > tolerance) return false;
    }
    return true;
}

bool sameAccepted(const metalrobo::NeuronCultureState& a,
                  const metalrobo::NeuronCultureState& b) {
    return exact(a.membrane, b.membrane) && exact(a.refractory, b.refractory) &&
        exact(a.preTrace, b.preTrace) && exact(a.postTrace, b.postTrace) &&
        exact(a.weights, b.weights) && exact(a.depression, b.depression) &&
        exact(a.spikes, b.spikes) && exact(a.spikeHistory, b.spikeHistory) &&
        exact(a.electrodeSpikeCounts, b.electrodeSpikeCounts) &&
        exact(a.phase, b.phase) && exact(a.tubulin, b.tubulin) &&
        a.generation == b.generation && a.tick == b.tick &&
        a.growthIteration == b.growthIteration;
}

std::uint64_t totalSpikes(const metalrobo::NeuronCultureState& state) {
    return std::accumulate(state.electrodeSpikeCounts.begin(),
                           state.electrodeSpikeCounts.end(), std::uint64_t{0});
}

void emitInspect(const metalrobo::CompiledNeuronCulture& culture) {
    const auto& h = culture.header();
    std::cout << "{\"schema\":\"numi.neuron-culture.inspect.v1\""
              << ",\"abi_version\":" << h.abiVersion
              << ",\"fingerprint\":" << culture.fingerprint()
              << ",\"neurons\":" << h.neuronCount
              << ",\"synapses\":" << h.synapseCount
              << ",\"electrodes\":" << h.electrodeCount
              << ",\"growth_grid\":[" << h.growthWidth << ',' << h.growthHeight << ']'
              << ",\"synaptic_current_scale\":" << h.synapticCurrentScale
              << ",\"synthetic_only\":true"
              << ",\"growth_stages\":[1,2,3,4]"
              << ",\"maturation_stage_5\":false"
              << ",\"automatic_synapse_from_crossing\":false}\n";
}

metalrobo::CompiledNeuronCulture compileCulture(std::uint32_t neurons,
                                                 std::uint32_t synapses,
                                                 std::uint64_t seed = 2056u) {
    auto pack = metalrobo::makePotterReferenceCulture(neurons, synapses, seed);
    metalrobo::CompiledNeuronCulture culture;
    const auto diagnostics = metalrobo::compileNeuronCulture(pack, culture);
    require(diagnostics.succeeded(), diagnostics.message);
    require(culture.valid(), "compiled culture is invalid");
    return culture;
}

metalrobo::CompiledNeuronCulture compilePairTimingCulture() {
    auto pack = metalrobo::makePotterReferenceCulture(2u, 1u, 0x53544450u);
    pack.id = "potter-stdp-pair-timing-v1";
    pack.network.synapticCurrentScale = 1.0f;
    pack.neurons = {
        {.x = 0.5f, .y = 0.5f, .biasCurrent = 0.0f, .capacitance = 1.0f,
         .excitatory = 1u},
        {.x = 2.5f, .y = 2.5f, .biasCurrent = 0.0f, .capacitance = 1.0f,
         .excitatory = 1u},
    };
    pack.synapses = {{.presynaptic = 0u, .postsynaptic = 1u, .delayTicks = 1u,
                      .plastic = 1u, .initialWeight = 0.05f,
                      .depressionUse = 0.0f}};
    pack.electrodes = {
        {.x = 0.5f, .y = 0.5f, .recordingRadius = 0.1f,
         .stimulationRadius = 0.1f, .active = 1u},
        {.x = 2.5f, .y = 2.5f, .recordingRadius = 0.1f,
         .stimulationRadius = 0.1f, .active = 1u},
    };
    metalrobo::CompiledNeuronCulture culture;
    const auto diagnostics = metalrobo::compileNeuronCulture(pack, culture);
    require(diagnostics.succeeded(), diagnostics.message);
    return culture;
}

float runPairTiming(
    const metalrobo::CompiledNeuronCulture& culture,
    const std::initializer_list<std::pair<std::uint32_t, std::uint32_t>> pulses
) {
    metalrobo::NeuronCultureWindowRequest request{
        .cultureFingerprint = culture.fingerprint(),
        .rootFingerprint = 0x5354445054494d45ull,
        .tickCount = 20u,
        .recordingStartTick = 0u,
        .recordingDurationTicks = 20u,
    };
    std::uint32_t ordinal = 0u;
    for (const auto [electrode, tick] : pulses) {
        request.pulses.push_back({
            .electrode = electrode, .startTick = tick, .durationTicks = 1u,
            .source = metalrobo::NeuronCultureStimulusSource::authored,
            .current = metalrobo::kPotterReferenceStimulationCurrent,
            .sourceFingerprint = request.rootFingerprint + ++ordinal,
        });
    }
    std::stable_sort(request.pulses.begin(), request.pulses.end(),
        [](const auto& left, const auto& right) {
            return left.startTick < right.startTick;
        });
    metalrobo::NeuronCultureReference cpu(culture);
    require(cpu.prepareWindow(request) && cpu.publishPrepared(),
            "pairwise CPU STDP window failed");
    auto metal = metalrobo::MetalNeuronCultureRuntime::create(culture);
    auto ticket = metal.prepareWindow(request);
    require(ticket.valid() &&
            ticket.wait() == metalrobo::MetalNeuronCultureStatus::success &&
            metal.publishPrepared() == metalrobo::MetalNeuronCultureStatus::success,
            "pairwise Metal STDP window failed");
    const auto gpu = metal.snapshotAcceptedForTesting();
    require(gpu.weights.size() == 1u && cpu.accepted().weights.size() == 1u &&
            std::abs(gpu.weights[0] - cpu.accepted().weights[0]) <= 2.0e-6f &&
            close(gpu.preTrace, cpu.accepted().preTrace, 2.0e-6f) &&
            close(gpu.postTrace, cpu.accepted().postTrace, 2.0e-6f),
            "pairwise STDP CPU/Metal parity failed");
    return cpu.accepted().weights[0];
}

metalrobo::CompiledNeuronCulture compileEvocationCulture(
    const float weight,
    const float currentScale
) {
    auto pack = metalrobo::makePotterReferenceCulture(2u, 1u, 0x45564f4b45u);
    pack.id = "potter-max-weight-evocation-v1";
    pack.network.synapticCurrentScale = currentScale;
    pack.neurons = {
        {.x = 0.5f, .y = 0.5f, .biasCurrent = 0.0f, .capacitance = 1.0f,
         .excitatory = 1u},
        {.x = 2.5f, .y = 2.5f, .biasCurrent = 7350.0f, .capacitance = 1.0f,
         .excitatory = 1u},
    };
    pack.synapses = {{.presynaptic = 0u, .postsynaptic = 1u, .delayTicks = 1u,
                      .plastic = 0u, .initialWeight = weight,
                      .depressionUse = 0.0f}};
    pack.electrodes = {
        {.x = 0.5f, .y = 0.5f, .recordingRadius = 0.1f,
         .stimulationRadius = 0.1f, .active = 1u},
        {.x = 2.5f, .y = 2.5f, .recordingRadius = 0.1f,
         .stimulationRadius = 0.1f, .active = 1u},
    };
    metalrobo::CompiledNeuronCulture culture;
    const auto diagnostics = metalrobo::compileNeuronCulture(pack, culture);
    require(diagnostics.succeeded(), diagnostics.message);
    return culture;
}

float measureEvocationProbability(const metalrobo::CompiledNeuronCulture& culture) {
    metalrobo::NeuronCultureReference reference(culture);
    constexpr std::uint32_t trials = 256u;
    constexpr std::uint32_t ticksPerTrial = 100u;
    constexpr std::uint32_t presynapticTick = 50u;
    std::uint32_t postsynapticSpikes = 0u;
    for (std::uint32_t trial = 0u; trial < trials; ++trial) {
        const std::uint64_t startTick = reference.accepted().tick;
        metalrobo::NeuronCultureWindowRequest request{
            .cultureFingerprint = culture.fingerprint(),
            .rootFingerprint = culture.fingerprint() ^ (trial + 1u),
            .tickCount = ticksPerTrial,
            .recordingStartTick = 0u,
            .recordingDurationTicks = ticksPerTrial,
            .pulses = {{
                .electrode = 0u, .startTick = presynapticTick,
                .durationTicks = 1u,
                .source = metalrobo::NeuronCultureStimulusSource::authored,
                .current = metalrobo::kPotterReferenceStimulationCurrent,
                .sourceFingerprint = culture.fingerprint() ^ (trial + 1u),
            }},
        };
        require(reference.prepareWindow(request) && reference.publishPrepared(),
                "max-weight evocation trial failed");
        const std::size_t historyIndex = static_cast<std::size_t>(
            (startTick + presynapticTick + 1u) & 255u) * 2u + 1u;
        postsynapticSpikes += reference.accepted().spikeHistory[historyIndex] != 0u;
    }
    return static_cast<float>(postsynapticSpikes) / trials;
}

void runQualification() {
    auto culture = compileCulture(64u, 512u);
    auto invalidPack = metalrobo::makePotterReferenceCulture(16u, 64u, 9u);
    invalidPack.synapses.front().postsynaptic = 99u;
    metalrobo::CompiledNeuronCulture preserved = culture;
    const auto invalid = metalrobo::compileNeuronCulture(invalidPack, preserved);
    require(!invalid.succeeded() && preserved.fingerprint() == culture.fingerprint(),
            "invalid topology did not fail transactionally");
    auto invalidDynamics = metalrobo::makePotterReferenceCulture(16u, 64u, 10u);
    invalidDynamics.network.synapticCurrentScale = 0.0f;
    const auto invalidDynamicsResult = metalrobo::compileNeuronCulture(
        invalidDynamics, preserved);
    require(!invalidDynamicsResult.succeeded() &&
            invalidDynamicsResult.status ==
                metalrobo::NeuronCultureCompileStatus::invalidNetwork &&
            preserved.fingerprint() == culture.fingerprint(),
            "invalid synaptic-current scale did not fail transactionally");

    const auto pairCulture = compilePairTimingCulture();
    const auto& pairHeader = pairCulture.header();
    const float timing10 = std::exp(-0.010f /
        pairHeader.traceTimeConstantSeconds);
    const float expectedPotentiated = 0.05f * (1.0f +
        pairHeader.stdpPotentiation * (pairHeader.maximumWeight - 0.05f) *
        timing10);
    const float expectedDepressed = 0.05f * (1.0f -
        pairHeader.stdpDepression * (0.05f - pairHeader.minimumWeight) *
        timing10);
    const float interval5 = std::exp(-0.005f /
        pairHeader.traceTimeConstantSeconds);
    const float suppression5 = 1.0f - std::exp(-0.005f /
        pairHeader.preSpikeSuppressionTimeConstantSeconds);
    const float expectedSuppressed = 0.05f * (1.0f +
        pairHeader.stdpPotentiation * (pairHeader.maximumWeight - 0.05f) *
        interval5 * suppression5);
    const float potentiated = runPairTiming(pairCulture, {{0u, 1u}, {1u, 11u}});
    const float depressed = runPairTiming(pairCulture, {{1u, 1u}, {0u, 11u}});
    const float suppressed = runPairTiming(
        pairCulture, {{0u, 1u}, {0u, 6u}, {1u, 11u}});
    require(std::abs(potentiated - expectedPotentiated) <= 2.0e-6f &&
            std::abs(depressed - expectedDepressed) <= 2.0e-6f &&
            std::abs(suppressed - expectedSuppressed) <= 2.0e-6f &&
            potentiated > 0.05f && depressed < 0.05f &&
            suppressed > 0.05f && suppressed < potentiated,
            "source-aligned bounded/suppressed STDP equation drifted");
    constexpr std::array<float, 12u> evocationScales{
        1000.0f, 5000.0f, 10000.0f, 20000.0f, 40000.0f, 80000.0f, 100000.0f,
        150000.0f, 175000.0f, 200000.0f, 300000.0f, 500000.0f};
    std::array<float, evocationScales.size()> evocationProbabilities{};
    for (std::size_t index = 0u; index < evocationScales.size(); ++index) {
        evocationProbabilities[index] = measureEvocationProbability(
            compileEvocationCulture(0.1f, evocationScales[index]));
    }
    const float maxWeightEvocation = measureEvocationProbability(
        compileEvocationCulture(
            0.1f, metalrobo::kPotterReferenceSynapticCurrentScale));
    const float baselineEvocation = measureEvocationProbability(
        compileEvocationCulture(0.0f, metalrobo::kPotterReferenceSynapticCurrentScale));
    require(maxWeightEvocation >= 0.45f && maxWeightEvocation <= 0.55f &&
            maxWeightEvocation - baselineEvocation >= 0.44f,
            "max-weight synapse drifted from the source evocation operating point");

    metalrobo::NeuronCultureReference cpu(culture);
    require(cpu.valid(), "CPU reference is invalid");
    const auto initial = cpu.accepted();
    const float referencePulse =
        metalrobo::PotterProtocolConfig{}.stimulationCurrent;
    require(cpu.prepareTicks(64u, 0u, referencePulse), "CPU tick prepare failed");
    cpu.rejectPrepared();
    require(sameAccepted(initial, cpu.accepted()), "CPU rejected ticks changed accepted state");
    require(cpu.prepareTicks(64u, 0u, referencePulse) && cpu.publishPrepared(),
            "CPU accepted ticks failed");
    const auto cpuTick = cpu.accepted();
    require(cpuTick.tick == 64u && totalSpikes(cpuTick) > 0u,
            "virtual MEA observed no evoked activity");
    metalrobo::NeuronCultureReference unstimulated(culture);
    require(unstimulated.prepareTicks(64u, UINT32_MAX, 0.0f) &&
            unstimulated.publishPrepared() &&
            !metalrobo::sameNeuronCultureState(
                cpuTick, unstimulated.accepted()),
            "canonical electrode pulse caused no neural-state intervention");

    std::vector<MRNumanXHumanSupportConsequenceGPU> support(10u);
    for (std::uint32_t i = 0u; i < support.size(); ++i) {
        support[i].identity.x = i;
        support[i].identity.w = MR_NUMANX_HUMAN_SUPPORT_CONSEQUENCE_VERSION;
        support[i].pointAndSeparation.x = -0.18f + 0.04f * static_cast<float>(i);
        support[i].pointAndSeparation.y = i < 5u ? -0.08f : 0.08f;
        support[i].impulseAndNormal.w = 0.001f * static_cast<float>(i + 1u);
        support[i].tangentVelocityAndImpulse.w = 0.0001f;
    }
    metalrobo::NeuronCultureStimulus supportStimulus;
    require(metalrobo::encodeAcceptedSupportStimulus(
                culture, support, 0.02f, 4.0f, supportStimulus) &&
            supportStimulus.electrode < culture.header().electrodeCount &&
            supportStimulus.current > 0.0f && supportStimulus.sourceFingerprint != 0u,
            "accepted NHCNT support consequence did not map to the virtual MEA");
    support.front().identity.x = 9u;
    const auto preservedStimulus = supportStimulus;
    require(!metalrobo::encodeAcceptedSupportStimulus(
                culture, support, 0.02f, 4.0f, supportStimulus) &&
            supportStimulus.sourceFingerprint == preservedStimulus.sourceFingerprint,
            "malformed NHCNT consequence did not fail transactionally");
    support.front().identity.x = 0u;
    auto unloadedSupport = support;
    for (auto& consequence : unloadedSupport) {
        consequence.impulseAndNormal.w = 0.0f;
        consequence.tangentVelocityAndImpulse.w = 0.0f;
    }
    metalrobo::NeuronCultureStimulus unloadedStimulus;
    require(metalrobo::encodeAcceptedSupportStimulus(
                culture, unloadedSupport, 0.02f, 4.0f, unloadedStimulus) &&
            unloadedStimulus.current == 0.0f &&
            unloadedStimulus.sourceFingerprint != 0u,
            "unloaded NHCNT support did not produce canonical zero stimulation");

    metalrobo::NeuronCultureReference replay(culture);
    require(replay.prepareTicks(64u, 0u, referencePulse) && replay.publishPrepared(),
            "CPU replay failed");
    require(sameAccepted(cpuTick, replay.accepted()), "CPU replay is not bitwise deterministic");

    metalrobo::NeuronCultureWindowRequest scheduled{
        .cultureFingerprint = culture.fingerprint(),
        .rootFingerprint = 0x4e554d49u,
        .tickCount = 64u,
        .pulses = {
            {.electrode = 0u, .startTick = 0u, .durationTicks = 3u,
             .source = metalrobo::NeuronCultureStimulusSource::contextProbe,
             .current = 900.0f, .sourceFingerprint = 0x101u},
            {.electrode = 1u, .startTick = 0u, .durationTicks = 3u,
             .source = metalrobo::NeuronCultureStimulusSource::patternedTraining,
             .current = 700.0f, .sourceFingerprint = 0x102u},
            {.electrode = 2u, .startTick = 24u, .durationTicks = 2u,
             .source = metalrobo::NeuronCultureStimulusSource::randomBackground,
             .current = 600.0f, .sourceFingerprint = 0x103u},
        },
    };
    metalrobo::NeuronCultureReference scheduledCPU(culture);
    require(scheduledCPU.prepareWindow(scheduled) && scheduledCPU.publishPrepared(),
            "scheduled CPU stimulation failed");
    auto scheduledGPU = metalrobo::MetalNeuronCultureRuntime::create(culture);
    auto scheduledTicket = scheduledGPU.prepareWindow(scheduled);
    require(scheduledTicket.valid() &&
            scheduledTicket.wait() == metalrobo::MetalNeuronCultureStatus::success &&
            scheduledGPU.publishPrepared() == metalrobo::MetalNeuronCultureStatus::success,
            "scheduled Metal stimulation failed");
    const auto scheduledMetalState = scheduledGPU.snapshotAcceptedForTesting();
    require(exact(scheduledCPU.accepted().spikes, scheduledMetalState.spikes) &&
            exact(scheduledCPU.accepted().spikeHistory,
                  scheduledMetalState.spikeHistory) &&
            exact(scheduledCPU.accepted().electrodeSpikeCounts,
                  scheduledMetalState.electrodeSpikeCounts) &&
            close(scheduledCPU.accepted().membrane,
                  scheduledMetalState.membrane, 2.0e-5f) &&
            close(scheduledCPU.accepted().refractory,
                  scheduledMetalState.refractory, 2.0e-5f) &&
            close(scheduledCPU.accepted().preTrace,
                  scheduledMetalState.preTrace, 2.0e-5f) &&
            close(scheduledCPU.accepted().postTrace,
                  scheduledMetalState.postTrace, 2.0e-5f) &&
            close(scheduledCPU.accepted().weights,
                  scheduledMetalState.weights, 2.0e-5f) &&
            close(scheduledCPU.accepted().depression,
                  scheduledMetalState.depression, 2.0e-5f),
            "scheduled CPU/Metal stimulation drifted");
    auto invalidSchedule = scheduled;
    invalidSchedule.pulses.front().durationTicks = 65u;
    require(!scheduledCPU.prepareWindow(invalidSchedule) &&
            !scheduledGPU.prepareWindow(invalidSchedule).valid(),
            "out-of-window stimulus schedule was accepted");

    id<MTLDevice> supportDevice = MTLCreateSystemDefaultDevice();
    id<MTLBuffer> supportBuffer = [supportDevice newBufferWithBytes:support.data()
        length:support.size() * sizeof(support.front())
        options:MTLResourceStorageModeShared];
    auto supportGPU = metalrobo::MetalNeuronCultureRuntime::create(
        culture, (__bridge void*)supportDevice);
    const metalrobo::MetalNeuronCultureSupportRequest supportRequest{
        .cultureFingerprint = culture.fingerprint(),
        .rootFingerprint = 0x4e48434e540001ull,
        .supportConsequencesBuffer = (__bridge void*)supportBuffer,
        .supportConsequencesGPUAddress = supportBuffer.gpuAddress,
        .supportCount = 10u,
        .supportStride = 10u,
        .tickCount = 64u,
        .physicsTimestepSeconds = 0.02f,
        .currentPerNewton = 4.0f,
    };
    CompletionObservation completionObservation;
    auto rejectedSupport = supportGPU.prepareSupportWindow(supportRequest);
    require(rejectedSupport.valid() && rejectedSupport.onCompleted(
                &completionObservation, &observeCompletion) &&
            rejectedSupport.wait() == metalrobo::MetalNeuronCultureStatus::success &&
            completionObservation.count.load() == 1u &&
            completionObservation.status.load() ==
                metalrobo::MetalNeuronCultureStatus::success,
            "NHCNT culture completion callback failed");
    supportGPU.rejectPrepared();
    require(sameAccepted(initial, supportGPU.snapshotAcceptedForTesting()),
            "rejected NHCNT culture preparation changed accepted state");
    auto acceptedSupport = supportGPU.prepareSupportWindow(supportRequest);
    require(acceptedSupport.valid() &&
            acceptedSupport.wait() == metalrobo::MetalNeuronCultureStatus::success &&
            supportGPU.publishPrepared() == metalrobo::MetalNeuronCultureStatus::success &&
            supportGPU.acceptedView().generation() == 1u,
            "accepted NHCNT culture preparation failed");
    auto aliasedRequest = supportRequest;
    const auto supportAccepted = supportGPU.acceptedView();
    aliasedRequest.supportConsequencesBuffer =
        supportAccepted.buffers().front().metalBuffer;
    aliasedRequest.supportConsequencesGPUAddress =
        supportAccepted.buffers().front().gpuAddress;
    require(!supportGPU.prepareSupportWindow(aliasedRequest).valid(),
            "accepted culture storage was admitted as NHCNT authority");
    id<MTLBuffer> unloadedSupportBuffer = [supportDevice
        newBufferWithBytes:unloadedSupport.data()
        length:unloadedSupport.size() * sizeof(unloadedSupport.front())
        options:MTLResourceStorageModeShared];
    auto unloadedGPU = metalrobo::MetalNeuronCultureRuntime::create(
        culture, (__bridge void*)supportDevice);
    auto unstimulatedGPU = metalrobo::MetalNeuronCultureRuntime::create(
        culture, (__bridge void*)supportDevice);
    auto unloadedRequest = supportRequest;
    unloadedRequest.supportConsequencesBuffer = (__bridge void*)unloadedSupportBuffer;
    unloadedRequest.supportConsequencesGPUAddress = unloadedSupportBuffer.gpuAddress;
    auto unloadedTicket = unloadedGPU.prepareSupportWindow(unloadedRequest);
    auto unstimulatedTicket = unstimulatedGPU.prepareTicks(64u, UINT32_MAX, 0.0f);
    require(unloadedTicket.valid() && unstimulatedTicket.valid() &&
            unloadedTicket.wait() == metalrobo::MetalNeuronCultureStatus::success &&
            unstimulatedTicket.wait() == metalrobo::MetalNeuronCultureStatus::success &&
            unloadedGPU.publishPrepared() == metalrobo::MetalNeuronCultureStatus::success &&
            unstimulatedGPU.publishPrepared() == metalrobo::MetalNeuronCultureStatus::success &&
            sameAccepted(unloadedGPU.snapshotAcceptedForTesting(),
                         unstimulatedGPU.snapshotAcceptedForTesting()),
            "unloaded NHCNT support changed the canonical zero-stimulation state");

    metalrobo::PotterProtocolConfig protocolConfig{
        .seed = 0x2056u,
        .windowTicks = 2000u,
        .probeResponseTicks = 100u,
        .calibrationWindowsPerContext = 1u,
        .baselineWindows = 4u,
        .postSwitchWindows = 4u,
    };
    metalrobo::PotterProtocolSession protocol(culture, protocolConfig);
    metalrobo::PotterProtocolSession replayProtocol(culture, protocolConfig);
    metalrobo::PotterProtocolSession sameSeedStdpOff(
        culture, protocolConfig,
        metalrobo::PotterProtocolAblation::stdpOff);
    require(protocol.result().contextElectrodes ==
                sameSeedStdpOff.result().contextElectrodes,
            "STDP ablation changed the authored CPS mapping");
    const auto pairedAdaptiveFirst = protocol.nextWindow();
    const auto pairedStdpOffFirst = sameSeedStdpOff.nextWindow();
    require(pairedAdaptiveFirst.request.plasticityEnabled &&
            !pairedStdpOffFirst.request.plasticityEnabled,
            "STDP ablation did not preserve culture identity while freezing weights");
    require(pairedAdaptiveFirst.request.pulses.size() ==
                pairedStdpOffFirst.request.pulses.size(),
            "STDP ablation changed the authored CPS schedule");
    for (std::size_t pulse = 0u;
         pulse < pairedAdaptiveFirst.request.pulses.size(); ++pulse) {
        const auto& adaptivePulse = pairedAdaptiveFirst.request.pulses[pulse];
        const auto& offPulse = pairedStdpOffFirst.request.pulses[pulse];
        require(adaptivePulse.electrode == offPulse.electrode &&
                adaptivePulse.startTick == offPulse.startTick &&
                adaptivePulse.durationTicks == offPulse.durationTicks &&
                adaptivePulse.source == offPulse.source &&
                std::bit_cast<std::uint32_t>(adaptivePulse.current) ==
                    std::bit_cast<std::uint32_t>(offPulse.current),
                "STDP ablation changed CPS pulse authority");
    }
    std::array<std::vector<std::pair<std::uint32_t, std::uint32_t>>, 5u>
        mappingCPS;
    for (std::uint32_t mapping = 0u; mapping < mappingCPS.size(); ++mapping) {
        auto mappingConfig = protocolConfig;
        mappingConfig.sensoryMapping = mapping;
        metalrobo::PotterProtocolSession mappingSession(culture, mappingConfig);
        const auto mappingWindow = mappingSession.nextWindow();
        for (const auto& pulse : mappingWindow.request.pulses) {
            if (pulse.source ==
                    metalrobo::NeuronCultureStimulusSource::contextProbe) {
                mappingCPS[mapping].emplace_back(
                    pulse.electrode, pulse.startTick);
            }
        }
        require(mappingCPS[mapping].size() == 3u,
                "Potter CPS set is incomplete");
        require(mappingCPS[mapping][0].first != mappingCPS[mapping][1].first &&
                mappingCPS[mapping][0].first != mappingCPS[mapping][2].first &&
                mappingCPS[mapping][1].first != mappingCPS[mapping][2].first,
                "Potter CPS reused an electrode within one sequence");
        for (std::uint32_t previous = 0u; previous < mapping; ++previous) {
            require(mappingCPS[mapping] != mappingCPS[previous],
                    "Potter sensory-mapping indices selected duplicate CPS sets");
        }
    }
    auto wrongTimestepPack = metalrobo::makePotterReferenceCulture(64u, 512u, 2056u);
    wrongTimestepPack.network.neuralTimestepSeconds = 0.0005f;
    metalrobo::CompiledNeuronCulture wrongTimestepCulture;
    require(metalrobo::compileNeuronCulture(
                wrongTimestepPack, wrongTimestepCulture).succeeded() &&
            !metalrobo::PotterProtocolSession(
                wrongTimestepCulture, protocolConfig).valid(),
            "Potter millisecond schedule admitted a non-1-ms culture");
    std::vector<std::uint32_t> protocolCounts(culture.header().electrodeCount, 0u);
    std::uint32_t protocolWindows = 0u;
    std::uint64_t backgroundIntervalTicks = 0u;
    std::uint32_t backgroundIntervalCount = 0u;
    while (!protocol.complete()) {
        const auto window = protocol.nextWindow();
        const auto replayWindow = replayProtocol.nextWindow();
        require(window.index == replayWindow.index &&
                window.context == replayWindow.context &&
                window.encodedContext == replayWindow.encodedContext &&
                window.trainingContext == replayWindow.trainingContext &&
                window.trainingPattern == replayWindow.trainingPattern &&
                std::bit_cast<std::uint32_t>(window.trainingBaselineRadialDelta) ==
                    std::bit_cast<std::uint32_t>(
                        replayWindow.trainingBaselineRadialDelta) &&
                window.request.recordingDurationTicks == 100u &&
                window.request.pulses.size() == replayWindow.request.pulses.size(),
                "Potter schedule replay drifted");
        if (window.trainingPattern != 0xffffffffu) {
            const auto& preceding = protocol.lastObservation();
            require(preceding.phase == metalrobo::PotterProtocolPhase::postSwitch &&
                    window.trainingContext == (preceding.context == 0u ? 2u :
                        (preceding.context == 2u ? 0u : preceding.context)) &&
                    std::bit_cast<std::uint32_t>(
                        window.trainingBaselineRadialDelta) ==
                    std::bit_cast<std::uint32_t>(
                        preceding.distanceAfter - preceding.distanceBefore),
                    "Potter PTS did not bind the exact preceding CPS movement");
        }
        for (std::size_t pulse = 0u; pulse < window.request.pulses.size(); ++pulse) {
            const auto& left = window.request.pulses[pulse];
            const auto& right = replayWindow.request.pulses[pulse];
            require(left.electrode == right.electrode && left.startTick == right.startTick &&
                    left.durationTicks == right.durationTicks && left.source == right.source &&
                    std::bit_cast<std::uint32_t>(left.current) ==
                        std::bit_cast<std::uint32_t>(right.current) &&
                    left.sourceFingerprint == right.sourceFingerprint,
                    "Potter pulse replay drifted");
        }
        std::array<std::uint32_t, 3u> cpsTicks{};
        std::uint32_t cpsCount = 0u;
        std::uint32_t ptsCount = 0u;
        std::uint32_t backgroundCount = 0u;
        std::uint32_t previousBackgroundTick = 0u;
        for (const auto& pulse : window.request.pulses) {
            if (pulse.source == metalrobo::NeuronCultureStimulusSource::contextProbe &&
                cpsCount < cpsTicks.size()) cpsTicks[cpsCount++] = pulse.startTick;
            ptsCount += pulse.source ==
                metalrobo::NeuronCultureStimulusSource::patternedTraining;
            if (pulse.source ==
                    metalrobo::NeuronCultureStimulusSource::randomBackground) {
                if (backgroundCount != 0u) {
                    const auto interval = pulse.startTick - previousBackgroundTick;
                    require(interval >= 200u && interval <= 400u,
                            "Potter RBS timing left the source 200-400 ms range");
                    backgroundIntervalTicks += interval;
                    ++backgroundIntervalCount;
                }
                previousBackgroundTick = pulse.startTick;
                ++backgroundCount;
            }
        }
        require(cpsCount == 3u && cpsTicks[1] - cpsTicks[0] >= 200u &&
                cpsTicks[1] - cpsTicks[0] <= 400u &&
                cpsTicks[2] - cpsTicks[1] >= 200u &&
                cpsTicks[2] - cpsTicks[1] <= 400u &&
                window.request.recordingStartTick == cpsTicks[2] &&
                ((ptsCount == 0u && backgroundCount > 0u) ||
                 (ptsCount > 0u && backgroundCount == 0u)),
                "Potter CPS timing drifted");
        const std::uint32_t expectedEncodedContext = window.context == 0u ? 2u :
            (window.context == 2u ? 0u : window.context);
        require(window.phase != metalrobo::PotterProtocolPhase::postSwitch ||
                    window.encodedContext == expectedEncodedContext,
                "Potter Q1/Q3 sensory switch drifted");
        const std::uint32_t electrode = (window.encodedContext * 13u + 7u) % 60u;
        protocolCounts[electrode] += 4u + window.index;
        require(protocol.observe(protocolCounts) && replayProtocol.observe(protocolCounts),
                "Potter observation was rejected");
        ++protocolWindows;
    }
    require(protocolWindows == 12u && protocol.result().completedWindows == 12u &&
            backgroundIntervalCount >= 20u &&
            static_cast<double>(backgroundIntervalTicks) /
                    backgroundIntervalCount >= 315.0 &&
            static_cast<double>(backgroundIntervalTicks) /
                    backgroundIntervalCount <= 350.0 &&
            protocol.result().protocolFingerprint == replayProtocol.result().protocolFingerprint &&
            protocol.result().baselineInward == replayProtocol.result().baselineInward &&
            protocol.result().postSwitchInward == replayProtocol.result().postSwitchInward &&
            protocol.result().postSwitchContextInward ==
                replayProtocol.result().postSwitchContextInward &&
            protocol.result().postSwitchContextMeasured ==
                replayProtocol.result().postSwitchContextMeasured &&
            protocol.result().patternedTrainingWindows +
                    protocol.result().randomBackgroundWindows ==
                protocol.result().postSwitchMeasured,
            "Potter deterministic mapping-switch replay failed");
    metalrobo::PotterProtocolSession noPTS(
        culture, protocolConfig, metalrobo::PotterProtocolAblation::patternedTrainingOff);
    std::vector<std::uint32_t> noPTSCounts(
        culture.header().electrodeCount, 0u);
    while (!noPTS.complete()) {
        const auto window = noPTS.nextWindow();
        require(window.trainingPattern == 0xffffffffu,
                "PTS-off ablation emitted a training pattern");
        const std::uint32_t electrode =
            (window.encodedContext * 13u + 7u) % 60u;
        noPTSCounts[electrode] += 4u + window.index;
        require(noPTS.observe(noPTSCounts), "PTS-off observation failed");
    }
    std::array<metalrobo::PotterProtocolPairedTrial, 15u> promotedTrials{};
    std::array<metalrobo::PotterProtocolPairedTrial, 15u> negativeTrials{};
    constexpr std::array<std::uint64_t, 3u> qualificationSeeds{
        2056u, 2057u, 2058u};
    std::size_t trialIndex = 0u;
    for (const auto seed : qualificationSeeds) {
        for (std::uint32_t mapping = 0u; mapping < 5u; ++mapping) {
            promotedTrials[trialIndex] = {
                .networkSeed = seed,
                .sensoryMapping = mapping,
                .adaptiveSuccess = 0.72f,
                .patternedTrainingOffSuccess = 0.55f,
            };
            negativeTrials[trialIndex] = {
                .networkSeed = seed,
                .sensoryMapping = mapping,
                .adaptiveSuccess = 0.59f,
                .patternedTrainingOffSuccess = 0.55f,
            };
            ++trialIndex;
        }
    }
    const auto promoted = metalrobo::qualifyPotterProtocol(promotedTrials);
    const auto negative = metalrobo::qualifyPotterProtocol(negativeTrials);
    auto shuffledTrials = promotedTrials;
    std::rotate(shuffledTrials.begin(), shuffledTrials.begin() + 7u,
                shuffledTrials.end());
    const auto shuffled = metalrobo::qualifyPotterProtocol(shuffledTrials);
    require(promoted.promoted && promoted.meanImprovement >= 0.10f &&
            promoted.lower95 > 0.0f && !negative.promoted &&
            negative.fingerprint != 0u && shuffled.promoted &&
            std::bit_cast<std::uint32_t>(shuffled.meanImprovement) ==
                std::bit_cast<std::uint32_t>(promoted.meanImprovement),
            "Potter promotion threshold or bootstrap gate drifted");

    const auto artifactDirectory = std::filesystem::temp_directory_path() /
        ("numi-neuron-culture-" + std::to_string(getpid()));
    std::filesystem::create_directories(artifactDirectory);
    const auto culturePath = artifactDirectory / "reference.nculture";
    const auto checkpointPath = artifactDirectory / "accepted.ncstate";
    const auto corruptPath = artifactDirectory / "corrupt.ncstate";
    const auto runPath = artifactDirectory / "qualification.ncrun.json";
    const auto cultureWrite = metalrobo::writeCompiledNeuronCulture(culture, culturePath);
    metalrobo::CompiledNeuronCulture decodedCulture;
    const auto cultureRead = metalrobo::readCompiledNeuronCulture(culturePath, decodedCulture);
    require(cultureWrite.succeeded() && cultureRead.succeeded() &&
            decodedCulture.fingerprint() == culture.fingerprint() &&
            cultureWrite.sha256 == cultureRead.sha256,
            "compiled .nculture round trip failed");
    const auto corruptCulturePath = artifactDirectory / "corrupt.nculture";
    std::filesystem::copy_file(culturePath, corruptCulturePath,
                               std::filesystem::copy_options::overwrite_existing);
    {
        std::fstream corrupt(
            corruptCulturePath, std::ios::binary | std::ios::in | std::ios::out);
        corrupt.seekp(-1, std::ios::end);
        const char byte = '\x7f';
        corrupt.write(&byte, 1);
    }
    auto preservedCulture = decodedCulture;
    require(!metalrobo::readCompiledNeuronCulture(
                corruptCulturePath, preservedCulture).succeeded() &&
            preservedCulture.fingerprint() == decodedCulture.fingerprint(),
            "corrupt .nculture did not fail before mutation");
    const auto checkpointWrite = metalrobo::writeNeuronCultureCheckpoint(
        culture, scheduledCPU.accepted(), checkpointPath);
    metalrobo::NeuronCultureState decodedState;
    const auto checkpointRead = metalrobo::readNeuronCultureCheckpoint(
        culture, checkpointPath, decodedState);
    require(checkpointWrite.succeeded() && checkpointRead.succeeded() &&
            metalrobo::sameNeuronCultureState(scheduledCPU.accepted(), decodedState),
            "accepted .ncstate round trip failed");
    auto otherCulture = compileCulture(64u, 512u, 2057u);
    auto identityPreserved = decodedState;
    require(!metalrobo::readNeuronCultureCheckpoint(
                otherCulture, checkpointPath, identityPreserved).succeeded() &&
            metalrobo::sameNeuronCultureState(identityPreserved, decodedState),
            "foreign-culture checkpoint did not fail before mutation");
    auto nonfiniteState = decodedState;
    nonfiniteState.membrane.front() =
        std::numeric_limits<float>::quiet_NaN();
    require(!metalrobo::writeNeuronCultureCheckpoint(
                culture, nonfiniteState,
                artifactDirectory / "nonfinite.ncstate").succeeded(),
            "nonfinite checkpoint state was published");
    auto wrongShapeState = decodedState;
    wrongShapeState.weights.pop_back();
    require(!metalrobo::writeNeuronCultureCheckpoint(
                culture, wrongShapeState,
                artifactDirectory / "wrong-shape.ncstate").succeeded(),
            "wrong-shape checkpoint state was published");
    const auto wrongVersionPath = artifactDirectory / "wrong-version.ncstate";
    std::filesystem::copy_file(checkpointPath, wrongVersionPath,
                               std::filesystem::copy_options::overwrite_existing);
    {
        std::fstream version(
            wrongVersionPath, std::ios::binary | std::ios::in | std::ios::out);
        const std::uint32_t unsupported = 99u;
        version.seekp(8, std::ios::beg);
        version.write(reinterpret_cast<const char*>(&unsupported),
                      sizeof(unsupported));
    }
    auto versionPreserved = decodedState;
    require(!metalrobo::readNeuronCultureCheckpoint(
                culture, wrongVersionPath, versionPreserved).succeeded() &&
            metalrobo::sameNeuronCultureState(versionPreserved, decodedState),
            "wrong-version checkpoint did not fail before mutation");
    const auto truncatedPath = artifactDirectory / "truncated.ncstate";
    std::filesystem::copy_file(checkpointPath, truncatedPath,
                               std::filesystem::copy_options::overwrite_existing);
    std::filesystem::resize_file(
        truncatedPath, std::filesystem::file_size(truncatedPath) - 1u);
    auto truncatedPreserved = decodedState;
    require(!metalrobo::readNeuronCultureCheckpoint(
                culture, truncatedPath, truncatedPreserved).succeeded() &&
            metalrobo::sameNeuronCultureState(truncatedPreserved, decodedState),
            "truncated checkpoint did not fail before mutation");
    const auto acceptedStateFingerprint =
        metalrobo::fingerprintNeuronCultureState(decodedState);
    auto changedFingerprintState = decodedState;
    changedFingerprintState.spikes.front() ^= 1u;
    require(acceptedStateFingerprint != 0u &&
            acceptedStateFingerprint == metalrobo::fingerprintNeuronCultureState(
                scheduledCPU.accepted()) &&
            acceptedStateFingerprint != metalrobo::fingerprintNeuronCultureState(
                changedFingerprintState),
            "accepted-state runtime fingerprint is not payload exact");
    auto restoredGPU = metalrobo::MetalNeuronCultureRuntime::create(culture);
    require(restoredGPU.restoreAccepted(decodedState) ==
                metalrobo::MetalNeuronCultureStatus::success &&
            metalrobo::sameNeuronCultureState(
                restoredGPU.snapshotAcceptedForTesting(), decodedState),
            "Metal checkpoint restore failed");
    auto uninterruptedResume = scheduledGPU.prepareWindow(scheduled);
    auto checkpointResume = restoredGPU.prepareWindow(scheduled);
    require(uninterruptedResume.valid() && checkpointResume.valid() &&
            uninterruptedResume.wait() ==
                metalrobo::MetalNeuronCultureStatus::success &&
            checkpointResume.wait() ==
                metalrobo::MetalNeuronCultureStatus::success &&
            scheduledGPU.publishPrepared() ==
                metalrobo::MetalNeuronCultureStatus::success &&
            restoredGPU.publishPrepared() ==
                metalrobo::MetalNeuronCultureStatus::success &&
            metalrobo::sameNeuronCultureState(
                scheduledGPU.snapshotAcceptedForTesting(),
                restoredGPU.snapshotAcceptedForTesting()),
            "checkpoint-resume diverged from uninterrupted Metal execution");
    auto invalidGenerationState = decodedState;
    invalidGenerationState.generation = 0u;
    const auto beforeInvalidGeneration = restoredGPU.snapshotAcceptedForTesting();
    require(!metalrobo::writeNeuronCultureCheckpoint(
                culture, invalidGenerationState,
                artifactDirectory / "invalid-generation.ncstate").succeeded() &&
            restoredGPU.restoreAccepted(invalidGenerationState) ==
                metalrobo::MetalNeuronCultureStatus::invalidArgument &&
            metalrobo::sameNeuronCultureState(
                restoredGPU.snapshotAcceptedForTesting(), beforeInvalidGeneration),
            "checkpoint generation mismatch mutated accepted state");
    std::filesystem::copy_file(checkpointPath, corruptPath,
                               std::filesystem::copy_options::overwrite_existing);
    {
        std::fstream corrupt(corruptPath, std::ios::binary | std::ios::in | std::ios::out);
        corrupt.seekp(-1, std::ios::end);
        const char byte = '\x7f';
        corrupt.write(&byte, 1);
    }
    metalrobo::NeuronCultureState preservedState = decodedState;
    require(!metalrobo::readNeuronCultureCheckpoint(culture, corruptPath, preservedState).succeeded() &&
            metalrobo::sameNeuronCultureState(preservedState, decodedState),
            "corrupt checkpoint did not fail before mutation");
    const auto runWrite = metalrobo::writeNeuronCultureRunManifest({
        .cultureFingerprint = culture.fingerprint(),
        .startingStateFingerprint = acceptedStateFingerprint,
        .acceptedStateFingerprint = acceptedStateFingerprint,
        .cultureSHA256 = cultureWrite.sha256Hex(),
        .startingCheckpointSHA256 = checkpointWrite.sha256Hex(),
        .checkpointSHA256 = checkpointWrite.sha256Hex(),
        .metalRoboRevision = std::string(40u, 'b'),
        .numiBrainRevision = std::string(40u, 'c'),
        .numanXRevision = std::string(40u, 'd'),
        .metallibSHA256 = std::string(64u, 'a'),
        .device = scheduledGPU.deviceName(),
        .operatingSystem = "macOS",
        .sdk = "Metal4",
        .command = "metalrobo_neuron_culture_probe",
        .protocol = "potter-switch-v1",
        .checkpoints = {checkpointPath.filename().string()},
        .measurements = {{"accepted_ticks", static_cast<double>(decodedState.tick)}},
        .deterministicReplay = true,
        .limitations = {"simulation only", "no wet-lab or hardware MEA control"},
    }, runPath);
    require(runWrite.succeeded() && runWrite.payloadBytes > 0u &&
            std::filesystem::file_size(runPath) == runWrite.payloadBytes,
            "canonical .ncrun.json publication failed");
    require(metalrobo::validateNeuronCultureRunManifest(
                runPath, culture.fingerprint(), "potter-switch-v1").succeeded() &&
            !metalrobo::validateNeuronCultureRunManifest(
                runPath, culture.fingerprint() ^ 1u,
                "potter-switch-v1").succeeded(),
            "canonical .ncrun.json identity validation failed");
    std::filesystem::remove_all(artifactDirectory);

    metalrobo::NeuronCultureReference plasticOff(culture);
    const auto plasticOffInitialWeights = plasticOff.accepted().weights;
    metalrobo::NeuronCultureWindowRequest plasticOffRequest{
        .cultureFingerprint = culture.fingerprint(),
        .rootFingerprint = culture.fingerprint() ^ 0x535444504f4646ull,
        .tickCount = 256u,
        .recordingStartTick = 0u,
        .recordingDurationTicks = 256u,
        .plasticityEnabled = false,
        .pulses = {{
            .electrode = 0u,
            .startTick = 0u,
            .durationTicks = 256u,
            .source = metalrobo::NeuronCultureStimulusSource::authored,
            .current = metalrobo::kPotterReferenceStimulationCurrent,
            .sourceFingerprint = culture.fingerprint(),
        }},
    };
    require(plasticOff.prepareWindow(plasticOffRequest) &&
            plasticOff.publishPrepared() &&
            exact(plasticOffInitialWeights, plasticOff.accepted().weights),
            "same-culture STDP-off ablation changed synaptic weights");
    auto plasticOffGPU = metalrobo::MetalNeuronCultureRuntime::create(culture);
    require(plasticOffGPU.valid(),
            "same-culture Metal STDP-off runtime is unavailable");
    const auto plasticOffGPUInitial = plasticOffGPU.snapshotAcceptedForTesting();
    auto plasticOffTicket = plasticOffGPU.prepareWindow(plasticOffRequest);
    require(plasticOffTicket.valid() &&
            plasticOffTicket.wait() == metalrobo::MetalNeuronCultureStatus::success &&
            plasticOffGPU.publishPrepared() ==
                metalrobo::MetalNeuronCultureStatus::success &&
            exact(plasticOffGPUInitial.weights,
                  plasticOffGPU.snapshotAcceptedForTesting().weights),
            "same-culture Metal STDP-off ablation changed synaptic weights");

    auto invalidGrowthPack = metalrobo::makePotterReferenceCulture(16u, 64u, 11u);
    invalidGrowthPack.growth.width = 0u;
    metalrobo::CompiledNeuronCulture invalidGrowthOutput;
    require(metalrobo::compileNeuronCulture(invalidGrowthPack, invalidGrowthOutput).status ==
                metalrobo::NeuronCultureCompileStatus::invalidGrowth,
            "invalid growth capacity did not fail closed");

    const auto beforeGrowth = cpu.accepted();
    require(cpu.prepareGrowth(3u) && cpu.publishPrepared(), "CPU growth failed");
    require(cpu.accepted().growthIteration == 3u &&
            !exact(beforeGrowth.phase, cpu.accepted().phase),
            "phase/tubulin growth did not evolve");

    auto gpu = metalrobo::MetalNeuronCultureRuntime::create(culture);
    require(gpu.valid(), "Metal neuron runtime is invalid");
    auto ticket = gpu.prepareTicks(64u, 0u, referencePulse);
    require(ticket.valid() && ticket.wait() == metalrobo::MetalNeuronCultureStatus::success,
            "Metal spike prepare failed");
    const auto gpuBeforePublish = gpu.snapshotAcceptedForTesting();
    require(gpuBeforePublish.tick == 0u && exact(gpuBeforePublish.membrane, initial.membrane),
            "Metal prepared state escaped before publication");
    require(gpu.publishPrepared() == metalrobo::MetalNeuronCultureStatus::success,
            "Metal spike publication failed");
    const auto gpuTick = gpu.snapshotAcceptedForTesting();
    require(gpuTick.tick == cpuTick.tick && exact(gpuTick.spikes, cpuTick.spikes) &&
            exact(gpuTick.electrodeSpikeCounts, cpuTick.electrodeSpikeCounts) &&
            close(gpuTick.membrane, cpuTick.membrane, 2.0e-5f) &&
            close(gpuTick.weights, cpuTick.weights, 2.0e-5f) &&
            close(gpuTick.depression, cpuTick.depression, 2.0e-5f),
            "Metal/CPU spiking, STDP, depression, or MEA parity failed");

    const auto retainedAccepted = gpu.acceptedView();
    require(retainedAccepted.valid() && retainedAccepted.generation() == 1u &&
            retainedAccepted.tick() == 64u && retainedAccepted.buffers().size() == 11u &&
            retainedAccepted.completionEvent() != nullptr &&
            retainedAccepted.completionValue() == 1u,
            "retained accepted GPU view is incomplete");
    const auto retainedMembrane = retainedAccepted.buffers().front();
    std::vector<std::byte> retainedBytes(retainedMembrane.byteLength);
    auto retainedBuffer = (__bridge id<MTLBuffer>)retainedMembrane.metalBuffer;
    std::memcpy(retainedBytes.data(), retainedBuffer.contents, retainedBytes.size());

    const auto acceptedGPU = gpuTick;
    auto rejected = gpu.prepareTicks(8u, 1u, 1100.0f);
    require(rejected.valid() && rejected.wait() == metalrobo::MetalNeuronCultureStatus::success,
            "Metal rejected candidate did not execute");
    gpu.rejectPrepared();
    require(sameAccepted(acceptedGPU, gpu.snapshotAcceptedForTesting()),
            "Metal rejected candidate changed accepted neural state");
    auto newer = gpu.prepareTicks(8u, 2u, 1000.0f);
    require(newer.valid() && newer.wait() == metalrobo::MetalNeuronCultureStatus::success &&
            gpu.publishPrepared() == metalrobo::MetalNeuronCultureStatus::success,
            "second accepted generation failed");
    const auto newerAccepted = gpu.acceptedView();
    require(newerAccepted.valid() && newerAccepted.generation() == 2u &&
            newerAccepted.buffers().front().gpuAddress != retainedMembrane.gpuAddress &&
            std::memcmp(retainedBytes.data(), retainedBuffer.contents, retainedBytes.size()) == 0,
            "retained accepted GPU generation was mutated or reused");

    metalrobo::NeuronCultureReference growthReference(culture);
    require(growthReference.prepareGrowth(3u) && growthReference.publishPrepared(),
            "growth reference failed");
    auto growthGPU = metalrobo::MetalNeuronCultureRuntime::create(culture);
    auto growthTicket = growthGPU.prepareGrowth(3u);
    require(growthTicket.valid() &&
            growthTicket.wait() == metalrobo::MetalNeuronCultureStatus::success &&
            growthGPU.publishPrepared() == metalrobo::MetalNeuronCultureStatus::success,
            "Metal growth failed");
    const auto gpuGrowth = growthGPU.snapshotAcceptedForTesting();
    require(gpuGrowth.growthIteration == 3u &&
            close(gpuGrowth.phase, growthReference.accepted().phase, 3.0e-5f) &&
            close(gpuGrowth.tubulin, growthReference.accepted().tubulin, 3.0e-5f),
            "Metal/CPU phase-field or tubulin parity failed");

    const auto full = compileCulture(1000u, 50000u);
    require(full.header().electrodeCount == 60u && full.header().neuronCount == 1000u &&
            full.header().synapseCount == 50000u,
            "Potter reference preset dimensions drifted");
    std::uint64_t recordedNeuronCount = 0u;
    std::uint64_t stimulatedNeuronCount = 0u;
    for (const auto& electrode : full.electrodes()) {
        for (const auto& neuron : full.neurons()) {
            const float dx = neuron.x - electrode.x;
            const float dy = neuron.y - electrode.y;
            const float distanceSquared = dx * dx + dy * dy;
            recordedNeuronCount += distanceSquared <=
                electrode.recordingRadius * electrode.recordingRadius;
            stimulatedNeuronCount += distanceSquared <=
                electrode.stimulationRadius * electrode.stimulationRadius;
        }
    }
    const double meanRecorded = static_cast<double>(recordedNeuronCount) /
        full.electrodes().size();
    const double meanStimulated = static_cast<double>(stimulatedNeuronCount) /
        full.electrodes().size();
    std::uint32_t shortAxons = 0u;
    std::uint32_t longAxons = 0u;
    const std::uint32_t spontaneousNeurons = static_cast<std::uint32_t>(
        std::count_if(full.neurons().begin(), full.neurons().end(),
            [](const auto& neuron) { return neuron.biasCurrent == 7350.0f; }));
    const std::uint32_t subthresholdNoiseNeurons = static_cast<std::uint32_t>(
        std::count_if(full.neurons().begin(), full.neurons().end(),
            [](const auto& neuron) { return neuron.biasCurrent == 2450.0f; }));
    const bool canonicalInitialWeights = std::all_of(
        full.synapses().begin(), full.synapses().end(),
        [](const auto& synapse) { return synapse.initialWeight == 0.05f; });
    bool canonicalSynapseDynamics = true;
    for (const auto& synapse : full.synapses()) {
        const auto& pre = full.neurons()[synapse.presynaptic];
        const auto& post = full.neurons()[synapse.postsynaptic];
        const float dx = pre.x - post.x;
        const float dy = pre.y - post.y;
        const float distanceSquared = dx * dx + dy * dy;
        const auto expectedDelay = std::max(1u, static_cast<std::uint32_t>(
            std::lround(std::sqrt(distanceSquared) / 0.3f)));
        canonicalSynapseDynamics = canonicalSynapseDynamics &&
            synapse.delayTicks == expectedDelay &&
            synapse.depressionUse == 0.5f;
        shortAxons += distanceSquared < 1.0f;
        longAxons += distanceSquared > 4.0f;
    }
    require(meanRecorded >= 3.0 && meanRecorded <= 8.0 &&
            meanStimulated >= 55.0 && meanStimulated <= 90.0 &&
            shortAxons > 30000u && longAxons > 500u &&
            spontaneousNeurons == 300u && subthresholdNoiseNeurons == 700u &&
            canonicalInitialWeights &&
            canonicalSynapseDynamics &&
            full.header().minimumWeight == 0.0f &&
            full.header().maximumWeight == 0.1f,
            "Potter spatial electrode or axon distribution drifted");
    auto fullRuntime = metalrobo::MetalNeuronCultureRuntime::create(full);
    require(fullRuntime.valid() && fullRuntime.residentBytes() > 0u &&
            fullRuntime.residentBytes() < 128ull * 1024ull * 1024ull,
            "Potter reference workload exceeds the 128 MiB resident budget");
    std::cout << "{\"schema\":\"numi.neuron-culture.qualification.v1\""
              << ",\"device\":\"" << gpu.deviceName() << "\""
              << ",\"fingerprint\":" << culture.fingerprint()
              << ",\"cpu_metal_parity\":true"
              << ",\"transactional_reject\":true"
              << ",\"bitwise_cpu_replay\":true"
              << ",\"virtual_mea_spikes\":" << totalSpikes(cpuTick)
              << ",\"growth_iterations\":3"
              << ",\"potter_reference\":{\"neurons\":1000,\"synapses\":50000,\"electrodes\":60}"
              << ",\"mean_recorded_neurons_per_electrode\":" << meanRecorded
              << ",\"mean_stimulated_neurons_per_electrode\":" << meanStimulated
              << ",\"short_axons\":" << shortAxons
              << ",\"long_axons\":" << longAxons
              << ",\"spontaneous_neurons\":" << spontaneousNeurons
              << ",\"initial_weight\":0.05,\"maximum_weight\":0.1"
              << ",\"depression_use\":0.5,\"conduction_metres_per_second\":0.3"
              << ",\"max_weight_evocation_probability\":" << maxWeightEvocation
              << ",\"matched_baseline_probability\":" << baselineEvocation
              << ",\"incremental_evocation_probability\":" <<
                    (maxWeightEvocation - baselineEvocation)
              << ",\"evocation_scale_sweep\":[";
    for (std::size_t index = 0u; index < evocationScales.size(); ++index) {
        if (index != 0u) std::cout << ',';
        std::cout << "[" << evocationScales[index] << ','
                  << evocationProbabilities[index] << "]";
    }
    std::cout << ']'
              << ",\"resident_bytes\":" << fullRuntime.residentBytes()
              << ",\"wet_lab\":false}\n";
}

void runLab(std::string_view command, bool quick, const std::string& outputPath,
            metalrobo::PotterProtocolAblation ablation,
            std::uint64_t networkSeed, std::uint32_t sensoryMapping,
            const std::string& packOut, const std::string& checkpointIn,
            const std::string& checkpointOut, const std::string& embodyTarget,
            const std::string& benchmarkMode, bool diagnosticSwitch,
            bool diagnosticCalibration, bool diagnosticLearning,
            std::uint32_t windowLimit) {
    const std::uint32_t neurons = quick ? 64u : 1000u;
    const std::uint32_t synapses = quick ? 512u : 50000u;
    auto culture = compileCulture(neurons, synapses, networkSeed);
    if (command == "compile" || command == "inspect") {
        if (command == "compile" && !packOut.empty()) {
            const auto result = metalrobo::writeCompiledNeuronCulture(culture, packOut);
            require(result.succeeded(), result.message);
            std::cout << "{\"schema\":\"numi.neuron-culture.compile.v1\",\"pack\":\""
                      << packOut << "\",\"sha256\":\"" << result.sha256Hex()
                      << "\",\"fingerprint\":" << culture.fingerprint() << "}\n";
            return;
        }
        if (command == "inspect" && !checkpointIn.empty()) {
            metalrobo::NeuronCultureState state;
            const auto loaded = metalrobo::readNeuronCultureCheckpoint(
                culture, checkpointIn, state);
            require(loaded.succeeded(), "checkpoint inspection failed");
            std::uint64_t plasticCount = 0u, low = 0u, high = 0u;
            double sum = 0.0, squareSum = 0.0;
            for (std::size_t edge = 0u; edge < state.weights.size(); ++edge) {
                if (culture.synapses()[edge].plastic == 0u) continue;
                const double weight = state.weights[edge];
                ++plasticCount; low += weight < 0.01; high += weight > 0.09;
                sum += weight; squareSum += weight * weight;
            }
            const double mean = plasticCount == 0u ? 0.0 : sum / plasticCount;
            const double variance = plasticCount == 0u ? 0.0 : std::max(
                0.0, squareSum / plasticCount - mean * mean);
            std::cout << std::setprecision(9)
                      << "{\"schema\":\"numi.neuron-culture.checkpoint-inspect.v1\""
                      << ",\"culture_fingerprint\":" << culture.fingerprint()
                      << ",\"state_fingerprint\":" <<
                            metalrobo::fingerprintNeuronCultureState(state)
                      << ",\"generation\":" << state.generation
                      << ",\"tick\":" << state.tick
                      << ",\"mea_spikes\":" << totalSpikes(state)
                      << ",\"mea_spikes_per_electrode_second\":" <<
                            (state.tick == 0u ? 0.0 :
                             static_cast<double>(totalSpikes(state)) /
                                (culture.header().electrodeCount * state.tick *
                                 culture.header().neuralTimestepSeconds))
                      << ",\"mean_plastic_weight\":" << mean
                      << ",\"plastic_weight_stddev\":" << std::sqrt(variance)
                      << ",\"plastic_extreme_fraction\":" <<
                            (plasticCount == 0u ? 0.0 :
                             static_cast<double>(low + high) / plasticCount)
                      << ",\"simulation_only\":true}\n";
            return;
        }
        emitInspect(culture);
        return;
    }
    if (command == "grow") {
        auto gpu = metalrobo::MetalNeuronCultureRuntime::create(culture);
        require(gpu.valid(), "Metal runtime unavailable");
        if (!checkpointIn.empty()) {
            metalrobo::NeuronCultureState restored;
            const auto loaded = metalrobo::readNeuronCultureCheckpoint(
                culture, checkpointIn, restored);
            require(loaded.succeeded() && gpu.restoreAccepted(restored) ==
                metalrobo::MetalNeuronCultureStatus::success,
                "growth checkpoint restore failed");
        }
        const std::uint32_t iterations = quick ? 4u : 32u;
        auto ticket = gpu.prepareGrowth(iterations);
        require(ticket.valid() && ticket.wait() == metalrobo::MetalNeuronCultureStatus::success,
                "growth command failed");
        require(gpu.publishPrepared() == metalrobo::MetalNeuronCultureStatus::success,
                "growth publication failed");
        const auto state = gpu.snapshotAcceptedForTesting();
        if (!checkpointOut.empty()) {
            require(metalrobo::writeNeuronCultureCheckpoint(
                culture, state, checkpointOut).succeeded(),
                "growth checkpoint publication failed");
        }
        const double phaseMass = std::accumulate(state.phase.begin(), state.phase.end(), 0.0);
        std::cout << "{\"schema\":\"numi.neuron-culture.growth.v1\",\"fingerprint\":"
                  << culture.fingerprint() << ",\"iterations\":" << iterations
                  << ",\"phase_mass\":" << std::setprecision(12) << phaseMass
                  << ",\"published\":true}\n";
        return;
    }
    if (command == "simulate" || command == "embody") {
        if (command == "embody" && embodyTarget == "numanx") {
            throw std::runtime_error(
                "NumanX culture embodiment is not yet qualified; refusing a synthetic fallback");
        }
        auto gpu = metalrobo::MetalNeuronCultureRuntime::create(culture);
        require(gpu.valid(), "Metal runtime unavailable");
        if (!checkpointIn.empty()) {
            metalrobo::NeuronCultureState restored;
            const auto loaded = metalrobo::readNeuronCultureCheckpoint(
                culture, checkpointIn, restored);
            require(loaded.succeeded() && gpu.restoreAccepted(restored) ==
                metalrobo::MetalNeuronCultureStatus::success,
                "simulation checkpoint restore failed");
        }
        if (command == "simulate" && benchmarkMode == "potter-equilibrate-v1") {
            constexpr std::uint32_t ticksPerWindow = 5000u;
            const std::uint64_t spontaneousTicks = static_cast<std::uint64_t>(
                5.0 * 3600.0 / 0.001);
            const std::uint64_t targetTicks = static_cast<std::uint64_t>(
                7.0 * 3600.0 / 0.001);
            auto accepted = gpu.snapshotAcceptedForTesting();
            require(accepted.tick <= targetTicks,
                    "equilibration checkpoint is beyond the canonical horizon");
            std::uint64_t nextRBSTick = spontaneousTicks;
            std::uint64_t rbsOrdinal = 0u;
            const auto randomAt = [&](const std::uint64_t ordinal,
                                      const std::uint64_t domain) noexcept {
                std::uint64_t value = culture.fingerprint() ^ domain ^
                    ((ordinal + 1u) * 0x9e3779b97f4a7c15ull);
                value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
                value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
                return value ^ (value >> 31u);
            };
            const auto rbsInterval = [&](const std::uint64_t ordinal) noexcept {
                const auto a = randomAt(ordinal, 0x494e54455256414cull) % 201u;
                const auto b = randomAt(ordinal, 0x494e54455256414dull) % 201u;
                return 200u + std::max(a, b);
            };
            while (nextRBSTick < accepted.tick) {
                nextRBSTick += rbsInterval(rbsOrdinal);
                ++rbsOrdinal;
            }
            const std::uint32_t availableWindows = static_cast<std::uint32_t>(
                (targetTicks - accepted.tick + ticksPerWindow - 1u) /
                    ticksPerWindow);
            const std::uint32_t runWindows = windowLimit == 0u ? availableWindows :
                std::min(windowLimit, availableWindows);
            for (std::uint32_t window = 0u; window < runWindows; ++window) {
                accepted = gpu.snapshotAcceptedForTesting();
                const std::uint32_t ticks = static_cast<std::uint32_t>(std::min<
                    std::uint64_t>(ticksPerWindow, targetTicks - accepted.tick));
                metalrobo::NeuronCultureWindowRequest request{
                    .cultureFingerprint = culture.fingerprint(),
                    .rootFingerprint = culture.fingerprint() ^
                        (accepted.tick + 0x455155494c494252ull),
                    .tickCount = ticks,
                    .recordingStartTick = 0u,
                    .recordingDurationTicks = ticks,
                };
                const std::uint64_t endTick = accepted.tick + ticks;
                while (nextRBSTick < endTick) {
                    if (nextRBSTick >= accepted.tick) {
                        const std::uint64_t pulseRandom = randomAt(
                            rbsOrdinal, 0x454c454354524f44ull);
                        request.pulses.push_back({
                            .electrode = static_cast<std::uint32_t>(pulseRandom %
                                culture.header().electrodeCount),
                            .startTick = static_cast<std::uint32_t>(
                                nextRBSTick - accepted.tick),
                            .durationTicks = 1u,
                            .source = metalrobo::NeuronCultureStimulusSource::randomBackground,
                            .current = metalrobo::kPotterReferenceStimulationCurrent,
                            .sourceFingerprint = request.rootFingerprint ^
                                (rbsOrdinal + 1u),
                        });
                    }
                    nextRBSTick += rbsInterval(rbsOrdinal);
                    ++rbsOrdinal;
                }
                auto ticket = gpu.prepareWindow(request);
                require(ticket.valid() &&
                        ticket.wait() == metalrobo::MetalNeuronCultureStatus::success &&
                        gpu.publishPrepared() == metalrobo::MetalNeuronCultureStatus::success,
                        "Potter equilibration window failed");
                if (!checkpointOut.empty() &&
                    ((window + 1u) % 120u == 0u || window + 1u == runWindows)) {
                    require(metalrobo::writeNeuronCultureCheckpoint(
                                culture, gpu.snapshotAcceptedForTesting(),
                                checkpointOut).succeeded(),
                            "equilibration checkpoint publication failed");
                }
            }
            accepted = gpu.snapshotAcceptedForTesting();
            std::uint64_t plasticCount = 0u, low = 0u, high = 0u;
            double sum = 0.0, squareSum = 0.0;
            for (std::size_t edge = 0u; edge < accepted.weights.size(); ++edge) {
                if (culture.synapses()[edge].plastic == 0u) continue;
                const double weight = accepted.weights[edge];
                ++plasticCount; low += weight < 0.01; high += weight > 0.09;
                sum += weight; squareSum += weight * weight;
            }
            const double mean = plasticCount == 0u ? 0.0 : sum / plasticCount;
            const double variance = plasticCount == 0u ? 0.0 : std::max(
                0.0, squareSum / plasticCount - mean * mean);
            std::cout << std::setprecision(9)
                      << "{\"schema\":\"numi.neuron-culture.equilibration.v1\""
                      << ",\"culture_fingerprint\":" << culture.fingerprint()
                      << ",\"tick\":" << accepted.tick
                      << ",\"target_tick\":" << targetTicks
                      << ",\"windows_executed\":" << runWindows
                      << ",\"phase\":\"" << (accepted.tick < spontaneousTicks ?
                            "spontaneous" : "rbs") << "\""
                      << ",\"mea_spikes\":" << totalSpikes(accepted)
                      << ",\"mea_spikes_per_electrode_second\":" <<
                            (accepted.tick == 0u ? 0.0 :
                             static_cast<double>(totalSpikes(accepted)) /
                                (culture.header().electrodeCount * accepted.tick *
                                 culture.header().neuralTimestepSeconds))
                      << ",\"mean_plastic_weight\":" << mean
                      << ",\"plastic_weight_stddev\":" << std::sqrt(variance)
                      << ",\"plastic_extreme_fraction\":" <<
                            (plasticCount == 0u ? 0.0 :
                             static_cast<double>(low + high) / plasticCount)
                      << ",\"complete\":" <<
                            (accepted.tick == targetTicks ? "true" : "false")
                      << ",\"simulation_only\":true}\n";
            return;
        }
        require(benchmarkMode.empty(),
                "simulate --mode supports only potter-equilibrate-v1");
        const std::uint32_t windows = quick ? 4u : 12u;
        double x = 0.0;
        double y = 0.0;
        for (std::uint32_t window = 0u; window < windows; ++window) {
            auto ticket = gpu.prepareTicks(
                100u, window % culture.header().electrodeCount,
                metalrobo::kPotterReferenceStimulationCurrent);
            require(ticket.valid() && ticket.wait() == metalrobo::MetalNeuronCultureStatus::success,
                    "network window failed");
            require(gpu.publishPrepared() == metalrobo::MetalNeuronCultureStatus::success,
                    "network window publication failed");
            const auto state = gpu.snapshotAcceptedForTesting();
            if (command == "embody") {
                const std::size_t quarter = state.electrodeSpikeCounts.size() / 4u;
                const auto sum = [&](std::size_t begin, std::size_t end) {
                    return std::accumulate(state.electrodeSpikeCounts.begin() + begin,
                                           state.electrodeSpikeCounts.begin() + end, 0.0);
                };
                const double q0 = sum(0u, quarter);
                const double q1 = sum(quarter, 2u * quarter);
                const double q2 = sum(2u * quarter, 3u * quarter);
                const double q3 = sum(3u * quarter, state.electrodeSpikeCounts.size());
                const double norm = std::max(1.0, q0 + q1 + q2 + q3);
                x += (q1 - q3) / norm;
                y += (q0 - q2) / norm;
            }
        }
        const auto state = gpu.snapshotAcceptedForTesting();
        if (!checkpointOut.empty()) {
            require(metalrobo::writeNeuronCultureCheckpoint(
                culture, state, checkpointOut).succeeded(),
                "simulation checkpoint publication failed");
        }
        std::cout << "{\"schema\":\"numi.neuron-culture."
                  << (command == "embody" ? "embodied" : "simulation")
                  << ".v1\",\"fingerprint\":" << culture.fingerprint()
                  << ",\"ticks\":" << state.tick
                  << ",\"mea_spikes\":" << totalSpikes(state)
                  << ",\"accepted_windows\":" << windows;
        if (command == "embody") std::cout << ",\"animat_position\":[" << x << ',' << y << ']';
        std::cout << ",\"simulation_only\":true}\n";
        return;
    }
    if (command == "benchmark") {
        require(benchmarkMode == "deterministic" || benchmarkMode == "throughput",
                "benchmark requires --mode deterministic|throughput");
        if (benchmarkMode == "deterministic") {
            metalrobo::NeuronCultureReference first(culture), second(culture);
            const std::uint32_t replayTicks = quick ? 100u : 1000u;
            require(first.prepareTicks(replayTicks, 0u,
                                       metalrobo::kPotterReferenceStimulationCurrent) &&
                    first.publishPrepared() &&
                    second.prepareTicks(replayTicks, 0u,
                                        metalrobo::kPotterReferenceStimulationCurrent) &&
                    second.publishPrepared() &&
                    metalrobo::sameNeuronCultureState(first.accepted(), second.accepted()),
                    "deterministic benchmark replay failed");
            auto firstGPU = metalrobo::MetalNeuronCultureRuntime::create(culture);
            auto secondGPU = metalrobo::MetalNeuronCultureRuntime::create(culture);
            auto firstTicket = firstGPU.prepareTicks(
                replayTicks, 0u, metalrobo::kPotterReferenceStimulationCurrent);
            auto secondTicket = secondGPU.prepareTicks(
                replayTicks, 0u, metalrobo::kPotterReferenceStimulationCurrent);
            require(firstTicket.valid() && secondTicket.valid() &&
                    firstTicket.wait() == metalrobo::MetalNeuronCultureStatus::success &&
                    secondTicket.wait() == metalrobo::MetalNeuronCultureStatus::success &&
                    firstGPU.publishPrepared() == metalrobo::MetalNeuronCultureStatus::success &&
                    secondGPU.publishPrepared() == metalrobo::MetalNeuronCultureStatus::success &&
                    metalrobo::sameNeuronCultureState(
                        firstGPU.snapshotAcceptedForTesting(),
                        secondGPU.snapshotAcceptedForTesting()),
                    "same-device Metal replay failed");
            std::cout << "{\"schema\":\"numi.neuron-culture.benchmark.v1\","
                         "\"mode\":\"deterministic\",\"cpu_bitwise\":true,"
                         "\"same_device_metal_bitwise\":true,\"ticks\":"
                      << replayTicks << "}\n";
            return;
        }
        auto gpu = metalrobo::MetalNeuronCultureRuntime::create(culture);
        require(gpu.valid(), "throughput benchmark Metal runtime unavailable");
        const std::uint32_t windows = quick ? 4u : 20u;
        const std::uint32_t windowTicks = quick ? 100u : 250u;
        std::vector<double> latencies;
        latencies.reserve(windows);
        const auto start = std::chrono::steady_clock::now();
        for (std::uint32_t window = 0u; window < windows; ++window) {
            @autoreleasepool {
                const auto windowStart = std::chrono::steady_clock::now();
                auto ticket = gpu.prepareTicks(
                    windowTicks, window % 60u,
                    metalrobo::kPotterReferenceStimulationCurrent);
                require(ticket.valid() &&
                        ticket.wait() == metalrobo::MetalNeuronCultureStatus::success &&
                        gpu.publishPrepared() == metalrobo::MetalNeuronCultureStatus::success,
                        "throughput benchmark failed");
                latencies.push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - windowStart).count());
            }
        }
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        auto overheadGPU = metalrobo::MetalNeuronCultureRuntime::create(culture);
        std::vector<double> overheads;
        overheads.reserve(16u);
        for (std::uint32_t sample = 0u; sample < 16u; ++sample) {
            @autoreleasepool {
                const auto sampleStart = std::chrono::steady_clock::now();
                auto ticket = overheadGPU.prepareTicks(1u, UINT32_MAX, 0.0f);
                require(ticket.valid() &&
                        ticket.wait() == metalrobo::MetalNeuronCultureStatus::success &&
                        overheadGPU.publishPrepared() ==
                            metalrobo::MetalNeuronCultureStatus::success,
                        "command-buffer overhead benchmark failed");
                overheads.push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - sampleStart).count());
            }
        }
        const auto measuredPercentile = [](std::vector<double> values, double p) {
            std::sort(values.begin(), values.end());
            const std::size_t index = std::min(values.size() - 1u,
                static_cast<std::size_t>(std::ceil(p * values.size())) - 1u);
            return values[index];
        };
        const std::uint32_t ticks = windows * windowTicks;
        std::cout << "{\"schema\":\"numi.neuron-culture.benchmark.v1\","
                     "\"mode\":\"throughput\",\"simulated_seconds_per_wall_second\":"
                  << (ticks * culture.header().neuralTimestepSeconds / seconds)
                  << ",\"prepare_ms\":{\"p50\":"
                  << measuredPercentile(latencies, 0.50)
                  << ",\"p95\":" << measuredPercentile(latencies, 0.95)
                  << ",\"p99\":" << measuredPercentile(latencies, 0.99) << "}"
                  << ",\"command_buffer_roundtrip_ms_p50\":"
                  << measuredPercentile(overheads, 0.50)
                  << ",\"resident_bytes\":" << gpu.residentBytes()
                  << ",\"peak_resident_bytes\":" << gpu.peakResidentBytes()
                  << ",\"simulation_only\":true}\n";
        return;
    }
    if (command == "protocol") {
        auto gpu = metalrobo::MetalNeuronCultureRuntime::create(culture);
        require(gpu.valid(), "Metal runtime unavailable");
        std::string startingCheckpointSHA256;
        if (!checkpointIn.empty()) {
            metalrobo::NeuronCultureState restored;
            const auto loaded = metalrobo::readNeuronCultureCheckpoint(
                culture, checkpointIn, restored);
            require(loaded.succeeded() && gpu.restoreAccepted(restored) ==
                    metalrobo::MetalNeuronCultureStatus::success,
                    "protocol checkpoint restore failed");
            startingCheckpointSHA256 = loaded.sha256Hex();
        }
        const auto startingStateFingerprint =
            metalrobo::fingerprintNeuronCultureState(
                gpu.snapshotAcceptedForTesting());
        require(startingStateFingerprint != 0u,
                "Potter starting-state fingerprint failed");
        metalrobo::PotterProtocolConfig config;
        config.seed = networkSeed ^ 0x505453u;
        config.sensoryMapping = sensoryMapping;
        if (diagnosticCalibration) {
            config.baselineWindows = 1u;
            config.postSwitchWindows = 1u;
        } else if (diagnosticLearning) {
            config.baselineWindows = 120u;
            config.postSwitchWindows = 1200u;
        } else if (diagnosticSwitch) {
            config.baselineWindows = 120u;
            config.postSwitchWindows = 120u;
        } else if (quick) {
            config.windowTicks = 2000u;
            config.calibrationWindowsPerContext = 1u;
            config.baselineWindows = 4u;
            config.postSwitchWindows = 4u;
        }
        metalrobo::PotterProtocolSession protocol(culture, config, ablation);
        require(protocol.valid(), "Potter protocol configuration is invalid");
        std::vector<double> milliseconds;
        const auto wallStart = std::chrono::steady_clock::now();
        while (!protocol.complete()) {
            @autoreleasepool {
                const auto window = protocol.nextWindow();
                const auto start = std::chrono::steady_clock::now();
                auto ticket = gpu.prepareWindow(window.request);
                require(ticket.valid() &&
                        ticket.wait() == metalrobo::MetalNeuronCultureStatus::success &&
                        gpu.publishPrepared() == metalrobo::MetalNeuronCultureStatus::success,
                        "Potter Metal window failed");
                const auto end = std::chrono::steady_clock::now();
                milliseconds.push_back(
                    std::chrono::duration<double, std::milli>(end - start).count());
                require(protocol.observe(gpu.acceptedElectrodeCountsTelemetry()),
                        "Potter MEA observation failed");
                const auto progress = protocol.result();
                if (progress.completedWindows % 120u == 0u) {
                    std::cerr << "potter_progress windows="
                              << progress.completedWindows
                              << " trailing_success="
                              << progress.postSwitchSuccess
                              << " trailing_measured="
                              << progress.finalIntervalMeasured << '\n';
                }
            }
        }
        const auto result = protocol.result();
        const auto finalState = gpu.snapshotAcceptedForTesting();
        if (!checkpointOut.empty()) {
            require(metalrobo::writeNeuronCultureCheckpoint(
                        culture, finalState, checkpointOut).succeeded(),
                    "protocol checkpoint publication failed");
        }
        const auto acceptedStateFingerprint =
            metalrobo::fingerprintNeuronCultureState(finalState);
        require(acceptedStateFingerprint != 0u,
                "Potter accepted-state fingerprint failed");
        const double meanWeight = std::accumulate(
            finalState.weights.begin(), finalState.weights.end(), 0.0) /
            std::max<std::size_t>(1u, finalState.weights.size());
        const double meanDepression = std::accumulate(
            finalState.depression.begin(), finalState.depression.end(), 0.0) /
            std::max<std::size_t>(1u, finalState.depression.size());
        std::uint64_t plasticCount = 0u;
        std::uint64_t plasticLow = 0u;
        std::uint64_t plasticHigh = 0u;
        double plasticWeightSum = 0.0;
        double plasticWeightSquareSum = 0.0;
        for (std::size_t edge = 0u; edge < finalState.weights.size(); ++edge) {
            if (culture.synapses()[edge].plastic == 0u) continue;
            const double weight = finalState.weights[edge];
            ++plasticCount;
            plasticLow += weight < 0.01;
            plasticHigh += weight > 0.09;
            plasticWeightSum += weight;
            plasticWeightSquareSum += weight * weight;
        }
        const double meanPlasticWeight = plasticCount == 0u ? 0.0 :
            plasticWeightSum / static_cast<double>(plasticCount);
        const double plasticVariance = plasticCount == 0u ? 0.0 : std::max(
            0.0, plasticWeightSquareSum / static_cast<double>(plasticCount) -
                meanPlasticWeight * meanPlasticWeight);
        const double wallSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wallStart).count();
        std::sort(milliseconds.begin(), milliseconds.end());
        const auto percentile = [&](double p) {
            const std::size_t index = static_cast<std::size_t>(
                std::ceil(p * milliseconds.size())) - 1u;
            return milliseconds[std::min(index, milliseconds.size() - 1u)];
        };
        std::ostringstream json;
        json << std::setprecision(9)
             << "{\"schema\":\"numi.neuron-culture.potter-switch.v1\""
             << ",\"culture_fingerprint\":" << culture.fingerprint()
             << ",\"starting_state_fingerprint\":"
             << startingStateFingerprint
             << ",\"culture_abi_version\":" << culture.header().abiVersion
             << ",\"protocol_algorithm_version\":"
             << metalrobo::kPotterProtocolAlgorithmVersion
             << ",\"synaptic_current_scale\":"
             << culture.header().synapticCurrentScale
             << ",\"stimulation_current\":" << config.stimulationCurrent
             << ",\"protocol_fingerprint\":" << result.protocolFingerprint
             << ",\"windows\":" << result.completedWindows
             << ",\"ablation\":" << static_cast<std::uint32_t>(ablation)
             << ",\"network_seed\":" << networkSeed
             << ",\"sensory_mapping\":" << sensoryMapping
             << ",\"baseline_success\":" << result.baselineSuccess
             << ",\"switch_interval_success\":"
             << result.switchIntervalSuccess
             << ",\"switch_interval_measured\":"
             << result.switchIntervalMeasured
             << ",\"switch_geometry_valid\":"
             << (result.switchGeometryValid ? "true" : "false")
             << ",\"context_electrodes\":["
             << '[' << result.contextElectrodes[0][0] << ','
             << result.contextElectrodes[0][1] << ','
             << result.contextElectrodes[0][2] << "],["
             << result.contextElectrodes[1][0] << ','
             << result.contextElectrodes[1][1] << ','
             << result.contextElectrodes[1][2] << "],["
             << result.contextElectrodes[2][0] << ','
             << result.contextElectrodes[2][1] << ','
             << result.contextElectrodes[2][2] << "],["
             << result.contextElectrodes[3][0] << ','
             << result.contextElectrodes[3][1] << ','
             << result.contextElectrodes[3][2] << "]]"
             << ",\"calibration_mean_center\":["
             << '[' << result.calibrationMeanCenter[0][0] << ','
             << result.calibrationMeanCenter[0][1] << "],["
             << result.calibrationMeanCenter[1][0] << ','
             << result.calibrationMeanCenter[1][1] << "],["
             << result.calibrationMeanCenter[2][0] << ','
             << result.calibrationMeanCenter[2][1] << "],["
             << result.calibrationMeanCenter[3][0] << ','
             << result.calibrationMeanCenter[3][1] << "]]"
             << ",\"calibration_center_offset\":["
             << result.calibrationCenterOffset[0] << ','
             << result.calibrationCenterOffset[1] << ']'
             << ",\"switched_mean_action\":["
             << '[' << result.switchedMeanAction[0][0] << ','
             << result.switchedMeanAction[0][1] << "],["
             << result.switchedMeanAction[1][0] << ','
             << result.switchedMeanAction[1][1] << "],["
             << result.switchedMeanAction[2][0] << ','
             << result.switchedMeanAction[2][1] << "],["
             << result.switchedMeanAction[3][0] << ','
             << result.switchedMeanAction[3][1] << "]]"
             << ",\"post_switch_success\":" << result.postSwitchSuccess
             << ",\"post_switch_measured\":" << result.postSwitchMeasured
             << ",\"post_switch_context_success\":[";
        for (std::size_t context = 0u; context < 4u; ++context) {
            if (context != 0u) json << ',';
            json << (result.postSwitchContextMeasured[context] == 0u ? 0.0 :
                static_cast<double>(result.postSwitchContextInward[context]) /
                    result.postSwitchContextMeasured[context]);
        }
        json << "],\"post_switch_context_measured\":[";
        for (std::size_t context = 0u; context < 4u; ++context) {
            if (context != 0u) json << ',';
            json << result.postSwitchContextMeasured[context];
        }
        json << "],\"post_switch_context_inward\":[";
        for (std::size_t context = 0u; context < 4u; ++context) {
            if (context != 0u) json << ',';
            json << result.postSwitchContextInward[context];
        }
        json << ']'
             << ",\"final_interval_measured\":" << result.finalIntervalMeasured
             << ",\"patterned_training_windows\":"
             << result.patternedTrainingWindows
             << ",\"random_background_windows\":"
             << result.randomBackgroundWindows
             << ",\"pattern_reinforcements\":"
             << result.patternReinforcements
             << ",\"pattern_copy_removals\":"
             << result.patternCopyRemovals
             << ",\"training_pool_copies\":" << result.trainingPoolCopies
             << ",\"distinct_weighted_patterns\":"
             << result.distinctWeightedPatterns
             << ",\"maximum_pattern_multiplicity\":"
             << result.maximumPatternMultiplicity
             << ",\"reached_goal\":" << (result.reachedGoal ? "true" : "false")
             << ",\"prepare_ms\":{\"p50\":" << percentile(0.50)
             << ",\"p95\":" << percentile(0.95)
             << ",\"p99\":" << percentile(0.99) << "}"
             << ",\"simulated_seconds_per_wall_second\":"
             << (static_cast<double>(result.completedWindows) * config.windowTicks *
                 culture.header().neuralTimestepSeconds / wallSeconds)
             << ",\"resident_bytes\":" << gpu.residentBytes()
             << ",\"peak_resident_bytes\":" << gpu.peakResidentBytes()
             << ",\"mean_weight\":" << meanWeight
             << ",\"mean_plastic_weight\":" << meanPlasticWeight
             << ",\"plastic_weight_stddev\":" << std::sqrt(plasticVariance)
             << ",\"plastic_extreme_fraction\":" << (plasticCount == 0u ? 0.0 :
                    static_cast<double>(plasticLow + plasticHigh) /
                        static_cast<double>(plasticCount))
             << ",\"plastic_low_fraction\":" << (plasticCount == 0u ? 0.0 :
                    static_cast<double>(plasticLow) /
                        static_cast<double>(plasticCount))
             << ",\"plastic_high_fraction\":" << (plasticCount == 0u ? 0.0 :
                    static_cast<double>(plasticHigh) /
                        static_cast<double>(plasticCount))
             << ",\"mean_depression\":" << meanDepression
             << ",\"accepted_state_fingerprint\":"
             << acceptedStateFingerprint
             << ",\"simulation_only\":true}\n";
        if (!outputPath.empty()) {
            std::string companionBase = outputPath;
            constexpr std::string_view runSuffix = ".ncrun.json";
            if (companionBase.ends_with(runSuffix)) {
                companionBase.resize(companionBase.size() - runSuffix.size());
            }
            const std::filesystem::path packPath = companionBase + ".nculture";
            const std::filesystem::path checkpointPath =
                companionBase + ".ncstate";
            const auto packWrite = metalrobo::writeCompiledNeuronCulture(culture, packPath);
            require(packWrite.succeeded(), "protocol culture identity publication failed");
            const auto checkpointWrite = metalrobo::writeNeuronCultureCheckpoint(
                culture, finalState, checkpointPath);
            if (!checkpointWrite.succeeded()) {
                std::error_code error;
                std::filesystem::remove(packPath, error);
                throw std::runtime_error(
                    "protocol accepted checkpoint publication failed");
            }
            const auto removeCompanions = [&]() noexcept {
                std::error_code error;
                std::filesystem::remove(packPath, error);
                error.clear();
                std::filesystem::remove(checkpointPath, error);
            };
            try {
                const auto ablationName = [&]() -> const char* {
                    switch (ablation) {
                        case metalrobo::PotterProtocolAblation::none:
                            return "none";
                        case metalrobo::PotterProtocolAblation::adaptiveSelectionOff:
                            return "adaptive-off";
                        case metalrobo::PotterProtocolAblation::patternedTrainingOff:
                            return "pts-off";
                        case metalrobo::PotterProtocolAblation::stdpOff:
                            return "stdp-off";
                    }
                    return "invalid";
                };
                std::ostringstream manifestCommand;
                manifestCommand
                    << "numi neurons protocol --preset potter-switch-v1"
                    << " --network-seed " << networkSeed
                    << " --sensory-mapping " << sensoryMapping
                    << " --ablation " << ablationName();
                if (quick) manifestCommand << " --quick";
                if (diagnosticSwitch)
                    manifestCommand << " --diagnostic-switch";
                if (diagnosticCalibration)
                    manifestCommand << " --diagnostic-calibration";
                if (diagnosticLearning)
                    manifestCommand << " --diagnostic-learning";
                if (!checkpointIn.empty())
                    manifestCommand << " --checkpoint-in "
                                    << std::quoted(checkpointIn);
                if (!checkpointOut.empty())
                    manifestCommand << " --checkpoint-out "
                                    << std::quoted(checkpointOut);
                manifestCommand << " --output " << std::quoted(outputPath);
                const auto runWrite = metalrobo::writeNeuronCultureRunManifest({
                    .cultureFingerprint = culture.fingerprint(),
                    .startingStateFingerprint = startingStateFingerprint,
                    .acceptedStateFingerprint = acceptedStateFingerprint,
                    .cultureSHA256 = packWrite.sha256Hex(),
                    .startingCheckpointSHA256 = startingCheckpointSHA256,
                    .checkpointSHA256 = checkpointWrite.sha256Hex(),
                    .metalRoboRevision = requiredEnvironment(
                        "NUMI_NEURON_METALROBO_REVISION"),
                    .numiBrainRevision = requiredEnvironment(
                        "NUMI_NEURON_NUMIBRAIN_REVISION"),
                    .numanXRevision = requiredEnvironment(
                        "NUMI_NEURON_NUMANX_REVISION"),
                    .metallibSHA256 = requiredEnvironment(
                        "NUMI_NEURON_METALLIB_SHA256"),
                    .device = gpu.deviceName(),
                    .operatingSystem = requiredEnvironment("NUMI_NEURON_OS"),
                    .sdk = requiredEnvironment("NUMI_NEURON_SDK"),
                    .command = manifestCommand.str(),
                    .protocol = "potter-switch-v1",
                    .checkpoints = {checkpointPath.filename().string()},
                    .measurements = {
                        {"network_seed", static_cast<double>(networkSeed)},
                        {"sensory_mapping", static_cast<double>(sensoryMapping)},
                        {"ablation", static_cast<double>(ablation)},
                        {"culture_abi_version",
                            static_cast<double>(culture.header().abiVersion)},
                        {"synaptic_current_scale",
                            culture.header().synapticCurrentScale},
                        {"stimulation_current", config.stimulationCurrent},
                        {"completed_windows", static_cast<double>(result.completedWindows)},
                        {"baseline_success", result.baselineSuccess},
                        {"post_switch_success", result.postSwitchSuccess},
                        {"final_interval_measured",
                            static_cast<double>(result.finalIntervalMeasured)},
                        {"patterned_training_windows",
                            static_cast<double>(result.patternedTrainingWindows)},
                        {"random_background_windows",
                            static_cast<double>(result.randomBackgroundWindows)},
                        {"pattern_reinforcements",
                            static_cast<double>(result.patternReinforcements)},
                        {"pattern_copy_removals",
                            static_cast<double>(result.patternCopyRemovals)},
                        {"reached_goal", result.reachedGoal ? 1.0 : 0.0},
                        {"prepare_ms_p50", percentile(0.50)},
                        {"prepare_ms_p95", percentile(0.95)},
                        {"prepare_ms_p99", percentile(0.99)},
                        {"simulated_seconds_per_wall_second",
                            static_cast<double>(result.completedWindows) *
                                config.windowTicks *
                                culture.header().neuralTimestepSeconds / wallSeconds},
                        {"resident_bytes", static_cast<double>(gpu.residentBytes())},
                        {"peak_resident_bytes",
                            static_cast<double>(gpu.peakResidentBytes())},
                        {"mean_weight", meanWeight},
                        {"mean_depression", meanDepression},
                    },
                    .deterministicReplay = false,
                    .limitations = {
                        "simulation only",
                        "no wet-lab or hardware MEA control",
                        "no biological calibration or clinical claim",
                    },
                }, outputPath);
                require(runWrite.succeeded(), runWrite.message);
            } catch (...) {
                removeCompanions();
                throw;
            }
        }
        std::cout << json.str();
        return;
    }
    if (command == "qualify") {
        constexpr std::array<std::uint64_t, 3u> seeds{2056u, 4099u, 8191u};
        constexpr std::uint64_t canonicalEquilibriumTicks =
            static_cast<std::uint64_t>(7.0 * 3600.0 / 0.001);
        std::array<metalrobo::NeuronCultureState, 3u> equilibriumStates;
        std::array<std::uint64_t, 3u> equilibriumFingerprints{};
        std::array<std::string, 3u> equilibriumCheckpointSHA256{};
        if (!quick) {
            require(!checkpointIn.empty() &&
                    std::filesystem::is_directory(checkpointIn),
                    "full qualification requires --checkpoint-in DIR with three canonical equilibrium checkpoints");
        }
        for (std::size_t seedIndex = 0u; seedIndex < seeds.size(); ++seedIndex) {
            const auto seedCulture = compileCulture(neurons, synapses, seeds[seedIndex]);
            if (quick) {
                metalrobo::NeuronCultureReference initial(seedCulture);
                require(initial.valid(), "quick qualification initial state is invalid");
                equilibriumStates[seedIndex] = initial.accepted();
            } else {
                const auto checkpointPath = std::filesystem::path(checkpointIn) /
                    ("potter-equilibrium-seed-" + std::to_string(seeds[seedIndex]) +
                     ".ncstate");
                const auto loaded = metalrobo::readNeuronCultureCheckpoint(
                    seedCulture, checkpointPath, equilibriumStates[seedIndex]);
                require(loaded.succeeded() &&
                        equilibriumStates[seedIndex].tick == canonicalEquilibriumTicks,
                        "qualification equilibrium checkpoint is missing or not at the canonical 7-hour horizon");
                equilibriumCheckpointSHA256[seedIndex] = loaded.sha256Hex();
            }
            equilibriumFingerprints[seedIndex] =
                metalrobo::fingerprintNeuronCultureState(
                    equilibriumStates[seedIndex]);
            require(equilibriumFingerprints[seedIndex] != 0u,
                    "qualification equilibrium fingerprint is invalid");
        }
        std::array<metalrobo::PotterProtocolPairedTrial, 15u> trials{};
        struct ModeResults {
            std::array<float, 15u> success{};
            std::array<std::uint64_t, 15u> acceptedStateFingerprint{};
            std::array<std::array<std::uint32_t, 4u>, 15u>
                contextMeasured{};
            std::array<std::array<std::uint32_t, 4u>, 15u>
                contextInward{};
        };
        std::string qualificationDevice;
        const auto runMode = [&](const metalrobo::PotterProtocolAblation mode) {
            ModeResults results;
            constexpr std::size_t concurrentTrials = 4u;
            struct ActiveTrial {
                std::size_t resultIndex = 0u;
                std::uint64_t seed = 0u;
                std::uint32_t mapping = 0u;
                metalrobo::CompiledNeuronCulture culture;
                metalrobo::MetalNeuronCultureRuntime gpu;
                std::unique_ptr<metalrobo::PotterProtocolSession> session;
            };
            for (std::size_t batchBegin = 0u;
                 batchBegin < results.success.size();
                 batchBegin += concurrentTrials) {
                const std::size_t batchEnd = std::min(
                    results.success.size(), batchBegin + concurrentTrials);
                std::vector<std::unique_ptr<ActiveTrial>> active;
                active.reserve(batchEnd - batchBegin);
                for (std::size_t index = batchBegin; index < batchEnd; ++index) {
                    auto trial = std::make_unique<ActiveTrial>();
                    trial->resultIndex = index;
                    trial->seed = seeds[index / 5u];
                    trial->mapping = static_cast<std::uint32_t>(index % 5u);
                    trial->culture = compileCulture(
                        neurons, synapses, trial->seed);
                    trial->gpu = metalrobo::MetalNeuronCultureRuntime::create(
                        trial->culture);
                    require(trial->gpu.valid() &&
                            trial->gpu.restoreAccepted(
                                equilibriumStates[index / 5u]) ==
                                metalrobo::MetalNeuronCultureStatus::success,
                            "qualification Metal equilibrium restore failed");
                    if (qualificationDevice.empty()) {
                        qualificationDevice = trial->gpu.deviceName();
                    } else {
                        require(trial->gpu.deviceName() == qualificationDevice,
                                "qualification device changed within the matrix");
                    }
                    metalrobo::PotterProtocolConfig config;
                    config.seed = trial->seed ^ 0x505453u;
                    config.sensoryMapping = trial->mapping;
                    if (quick) {
                        config.windowTicks = 2000u;
                        config.calibrationWindowsPerContext = 1u;
                        config.baselineWindows = 4u;
                        config.postSwitchWindows = 4u;
                    }
                    trial->session =
                        std::make_unique<metalrobo::PotterProtocolSession>(
                            trial->culture, config, mode);
                    require(trial->session->valid(),
                            "qualification protocol is invalid");
                    active.push_back(std::move(trial));
                }
                while (std::any_of(active.begin(), active.end(), [](const auto& trial) {
                    return !trial->session->complete();
                })) {
                    @autoreleasepool {
                        struct PendingWindow {
                            ActiveTrial* trial = nullptr;
                            metalrobo::MetalNeuronCultureTicket ticket;
                        };
                        std::vector<PendingWindow> pending;
                        pending.reserve(active.size());
                        for (const auto& trial : active) {
                            if (trial->session->complete()) continue;
                            const auto window = trial->session->nextWindow();
                            auto ticket = trial->gpu.prepareWindow(window.request);
                            require(ticket.valid(),
                                    "qualification Metal window was not admitted");
                            pending.push_back({trial.get(), std::move(ticket)});
                        }
                        for (auto& item : pending) {
                            require(item.ticket.wait() ==
                                        metalrobo::MetalNeuronCultureStatus::success &&
                                    item.trial->gpu.publishPrepared() ==
                                        metalrobo::MetalNeuronCultureStatus::success,
                                    "qualification Metal window failed");
                            require(item.trial->session->observe(
                                        item.trial->gpu.acceptedElectrodeCountsTelemetry()),
                                    "qualification MEA observation failed");
                            const auto progress = item.trial->session->result();
                            if (progress.completedWindows % 120u == 0u) {
                                std::cerr << "qualification_progress mode="
                                          << static_cast<std::uint32_t>(mode)
                                          << " seed=" << item.trial->seed
                                          << " mapping=" << item.trial->mapping
                                          << " windows=" << progress.completedWindows
                                          << " trailing_success="
                                          << progress.postSwitchSuccess << '\n';
                            }
                        }
                    }
                }
                for (const auto& trial : active) {
                    const auto trialResult = trial->session->result();
                    results.success[trial->resultIndex] =
                        trialResult.postSwitchSuccess;
                    results.contextMeasured[trial->resultIndex] =
                        trialResult.postSwitchContextMeasured;
                    results.contextInward[trial->resultIndex] =
                        trialResult.postSwitchContextInward;
                    results.acceptedStateFingerprint[trial->resultIndex] =
                        metalrobo::fingerprintNeuronCultureState(
                            trial->gpu.snapshotAcceptedForTesting());
                    require(results.acceptedStateFingerprint[trial->resultIndex] != 0u,
                            "qualification accepted-state fingerprint failed");
                    std::cerr << "qualification_trial mode="
                              << static_cast<std::uint32_t>(mode)
                              << " seed=" << trial->seed
                              << " mapping=" << trial->mapping
                              << " success="
                              << results.success[trial->resultIndex]
                              << " state_fingerprint="
                              << results.acceptedStateFingerprint[trial->resultIndex]
                              << '\n';
                }
            }
            return results;
        };
        const auto start = std::chrono::steady_clock::now();
        const auto adaptive = runMode(metalrobo::PotterProtocolAblation::none);
        const auto off = runMode(
            metalrobo::PotterProtocolAblation::patternedTrainingOff);
        const auto adaptiveOff = runMode(
            metalrobo::PotterProtocolAblation::adaptiveSelectionOff);
        const auto stdpOff = runMode(metalrobo::PotterProtocolAblation::stdpOff);
        for (std::size_t index = 0u; index < trials.size(); ++index) {
            trials[index] = {
                .networkSeed = seeds[index / 5u],
                .sensoryMapping = static_cast<std::uint32_t>(index % 5u),
                .adaptiveSuccess = adaptive.success[index],
                .patternedTrainingOffSuccess = off.success[index],
            };
        }
        const auto qualification = metalrobo::qualifyPotterProtocol(trials);
        bool deterministicReplay = false;
        if (qualification.promoted) {
            const auto adaptiveReplay = runMode(
                metalrobo::PotterProtocolAblation::none);
            const auto offReplay = runMode(
                metalrobo::PotterProtocolAblation::patternedTrainingOff);
            const auto adaptiveOffReplay = runMode(
                metalrobo::PotterProtocolAblation::adaptiveSelectionOff);
            const auto stdpOffReplay = runMode(
                metalrobo::PotterProtocolAblation::stdpOff);
            const auto sameMode = [](const ModeResults& left,
                                     const ModeResults& right) {
                return std::memcmp(left.success.data(), right.success.data(),
                                   sizeof(left.success)) == 0 &&
                    std::memcmp(left.acceptedStateFingerprint.data(),
                                right.acceptedStateFingerprint.data(),
                                sizeof(left.acceptedStateFingerprint)) == 0 &&
                    left.contextMeasured == right.contextMeasured &&
                    left.contextInward == right.contextInward;
            };
            deterministicReplay = sameMode(adaptive, adaptiveReplay) &&
                sameMode(off, offReplay) &&
                sameMode(adaptiveOff, adaptiveOffReplay) &&
                sameMode(stdpOff, stdpOffReplay);
            require(deterministicReplay,
                    "qualification rerun drifted with fresh transaction authority");
        }
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        const auto mean = [](const auto& values) {
            return std::accumulate(values.begin(), values.end(), 0.0) /
                static_cast<double>(values.size());
        };
        const std::string metalRoboRevision = quick ? std::string{} :
            requiredEnvironment("NUMI_NEURON_METALROBO_REVISION");
        const std::string numiBrainRevision = quick ? std::string{} :
            requiredEnvironment("NUMI_NEURON_NUMIBRAIN_REVISION");
        const std::string numanXRevision = quick ? std::string{} :
            requiredEnvironment("NUMI_NEURON_NUMANX_REVISION");
        const std::string metallibSHA256 = quick ? std::string{} :
            requiredEnvironment("NUMI_NEURON_METALLIB_SHA256");
        const std::string operatingSystem = quick ? std::string{} :
            requiredEnvironment("NUMI_NEURON_OS");
        const std::string sdk = quick ? std::string{} :
            requiredEnvironment("NUMI_NEURON_SDK");
        std::ostringstream qualificationCommand;
        qualificationCommand << "numi neurons qualify";
        if (quick) {
            qualificationCommand << " --quick";
        } else {
            qualificationCommand << " --checkpoint-in " << checkpointIn;
        }
        if (!outputPath.empty())
            qualificationCommand << " --output " << outputPath;
        std::ostringstream json;
        json << std::setprecision(9)
             << "{\"schema\":\"numi.neuron-culture.potter-qualification.v1\""
             << ",\"trial_count\":" << qualification.trialCount
             << ",\"bootstrap_samples\":" << qualification.bootstrapSamples
             << ",\"qualification_fingerprint\":" << qualification.fingerprint
             << ",\"mean_improvement\":" << qualification.meanImprovement
             << ",\"lower_95\":" << qualification.lower95
             << ",\"required_improvement\":" << qualification.requiredImprovement
             << ",\"culture_abi_version\":" << MR_NEURON_CULTURE_ABI_VERSION
             << ",\"synaptic_current_scale\":"
             << metalrobo::kPotterReferenceSynapticCurrentScale
             << ",\"stimulation_current\":"
             << metalrobo::PotterProtocolConfig{}.stimulationCurrent
             << ",\"equilibrium_state_fingerprints\":["
             << equilibriumFingerprints[0] << ','
             << equilibriumFingerprints[1] << ','
             << equilibriumFingerprints[2] << ']'
             << ",\"adaptive_off_mean\":" << mean(adaptiveOff.success)
             << ",\"stdp_off_mean\":" << mean(stdpOff.success)
             << ",\"promoted\":" << (qualification.promoted ? "true" : "false")
             << ",\"wall_seconds\":" << seconds
             << ",\"quick\":" << (quick ? "true" : "false")
             << ",\"deterministic_replay\":"
             << (deterministicReplay ? "true" : "false")
             << ",\"command\":\""
             << jsonEscape(qualificationCommand.str()) << "\"";
        if (!quick) {
            json << ",\"source_revisions\":{\"numi_lab\":\""
                 << jsonEscape(metalRoboRevision)
                 << "\",\"numi_brain\":\""
                 << jsonEscape(numiBrainRevision)
                 << "\",\"numan_x\":\""
                 << jsonEscape(numanXRevision) << "\"}"
                 << ",\"runtime\":{\"device\":\""
                 << jsonEscape(qualificationDevice)
                 << "\",\"operating_system\":\""
                 << jsonEscape(operatingSystem)
                 << "\",\"sdk\":\"" << jsonEscape(sdk)
                 << "\",\"metallib_sha256\":\""
                 << jsonEscape(metallibSHA256)
                 << "\"}"
                 << ",\"equilibrium_checkpoint_sha256\":[\""
                 << equilibriumCheckpointSHA256[0] << "\",\""
                 << equilibriumCheckpointSHA256[1] << "\",\""
                 << equilibriumCheckpointSHA256[2] << "\"]";
        }
        json
             << ",\"trials\":[";
        for (std::size_t index = 0u; index < trials.size(); ++index) {
            if (index != 0u) json << ',';
            json << "{\"network_seed\":" << trials[index].networkSeed
                 << ",\"sensory_mapping\":" << trials[index].sensoryMapping
                 << ",\"adaptive\":" << trials[index].adaptiveSuccess
                 << ",\"pts_off\":"
                 << trials[index].patternedTrainingOffSuccess
                 << ",\"adaptive_off\":" << adaptiveOff.success[index]
                 << ",\"stdp_off\":" << stdpOff.success[index]
                 << ",\"adaptive_context_measured\":["
                 << adaptive.contextMeasured[index][0] << ','
                 << adaptive.contextMeasured[index][1] << ','
                 << adaptive.contextMeasured[index][2] << ','
                 << adaptive.contextMeasured[index][3] << ']'
                 << ",\"adaptive_context_inward\":["
                 << adaptive.contextInward[index][0] << ','
                 << adaptive.contextInward[index][1] << ','
                 << adaptive.contextInward[index][2] << ','
                 << adaptive.contextInward[index][3] << ']'
                 << ",\"pts_off_context_measured\":["
                 << off.contextMeasured[index][0] << ','
                 << off.contextMeasured[index][1] << ','
                 << off.contextMeasured[index][2] << ','
                 << off.contextMeasured[index][3] << ']'
                 << ",\"pts_off_context_inward\":["
                 << off.contextInward[index][0] << ','
                 << off.contextInward[index][1] << ','
                 << off.contextInward[index][2] << ','
                 << off.contextInward[index][3] << ']'
                 << ",\"adaptive_state_fingerprint\":"
                 << adaptive.acceptedStateFingerprint[index]
                 << ",\"pts_off_state_fingerprint\":"
                 << off.acceptedStateFingerprint[index]
                 << ",\"adaptive_off_state_fingerprint\":"
                 << adaptiveOff.acceptedStateFingerprint[index]
                 << ",\"stdp_off_state_fingerprint\":"
                 << stdpOff.acceptedStateFingerprint[index] << '}';
        }
        json << "],\"simulation_only\":true}\n";
        if (outputPath.empty()) {
            std::cout << json.str();
        } else {
            const auto temporary = outputPath + ".tmp." + std::to_string(getpid());
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            output << json.str();
            output.close();
            require(output.good(), "qualification output write failed");
            std::filesystem::rename(temporary, outputPath);
            std::cout << json.str();
        }
        require(qualification.promoted,
                "adaptive PTS did not pass the immutable promotion threshold");
        return;
    }
    if (command == "replay") {
        metalrobo::NeuronCultureReference a(culture);
        metalrobo::NeuronCultureReference b(culture);
        require(a.prepareTicks(200u, 0u,
                               metalrobo::kPotterReferenceStimulationCurrent) &&
                a.publishPrepared() &&
                b.prepareTicks(200u, 0u,
                               metalrobo::kPotterReferenceStimulationCurrent) &&
                b.publishPrepared() &&
                sameAccepted(a.accepted(), b.accepted()), "replay mismatch");
        std::cout << "{\"schema\":\"numi.neuron-culture.replay.v1\",\"fingerprint\":"
                  << culture.fingerprint() << ",\"bitwise\":true,\"ticks\":200}\n";
        return;
    }
    if (command == "render") {
        require(!outputPath.empty(), "render requires --output FILE.ppm");
        auto gpu = metalrobo::MetalNeuronCultureRuntime::create(culture);
        auto growth = gpu.prepareGrowth(quick ? 4u : 16u);
        require(growth.valid() && growth.wait() == metalrobo::MetalNeuronCultureStatus::success &&
                gpu.publishPrepared() == metalrobo::MetalNeuronCultureStatus::success,
                "render growth failed");
        auto neural = gpu.prepareTicks(
            quick ? 100u : 400u, 0u,
            metalrobo::kPotterReferenceStimulationCurrent);
        require(neural.valid() && neural.wait() == metalrobo::MetalNeuronCultureStatus::success &&
                gpu.publishPrepared() == metalrobo::MetalNeuronCultureStatus::success,
                "render neural simulation failed");
        const auto state = gpu.snapshotAcceptedForTesting();
        constexpr std::uint32_t imageWidth = 512u;
        constexpr std::uint32_t imageHeight = 512u;
        std::vector<unsigned char> pixels(
            static_cast<std::size_t>(imageWidth) * imageHeight * 3u, 0u);
        for (std::uint32_t y = 0u; y < imageHeight; ++y) {
            for (std::uint32_t x = 0u; x < imageWidth; ++x) {
                const std::uint32_t gx = std::min(culture.header().growthWidth - 1u,
                    x * culture.header().growthWidth / imageWidth);
                const std::uint32_t gy = std::min(culture.header().growthHeight - 1u,
                    y * culture.header().growthHeight / imageHeight);
                const float phase = state.phase[static_cast<std::size_t>(gy) *
                    culture.header().growthWidth + gx];
                const float tubulin = state.tubulin[static_cast<std::size_t>(gy) *
                    culture.header().growthWidth + gx];
                const std::size_t pixel = (static_cast<std::size_t>(y) * imageWidth + x) * 3u;
                pixels[pixel + 0u] = static_cast<unsigned char>(25.0f + 50.0f * phase);
                pixels[pixel + 1u] = static_cast<unsigned char>(20.0f + 210.0f * std::min(1.0f, tubulin));
                pixels[pixel + 2u] = static_cast<unsigned char>(35.0f + 180.0f * phase);
            }
        }
        for (std::size_t i = 0u; i < culture.electrodes().size(); ++i) {
            const auto& electrode = culture.electrodes()[i];
            const int cx = static_cast<int>(electrode.x / 3.0f * imageWidth);
            const int cy = static_cast<int>(electrode.y / 3.0f * imageHeight);
            const unsigned char activity = static_cast<unsigned char>(std::min<std::uint32_t>(
                255u, 80u + 8u * state.electrodeSpikeCounts[i]));
            for (int dy = -4; dy <= 4; ++dy) {
                for (int dx = -4; dx <= 4; ++dx) {
                    if (dx * dx + dy * dy > 16) continue;
                    const int px = cx + dx;
                    const int py = cy + dy;
                    if (px < 0 || py < 0 || px >= static_cast<int>(imageWidth) ||
                        py >= static_cast<int>(imageHeight)) continue;
                    const std::size_t pixel = (static_cast<std::size_t>(py) * imageWidth + px) * 3u;
                    pixels[pixel + 0u] = activity;
                    pixels[pixel + 1u] = 40u;
                    pixels[pixel + 2u] = 30u;
                }
            }
        }
        std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
        require(output.good(), "could not open render output");
        output << "P6\n" << imageWidth << ' ' << imageHeight << "\n255\n";
        output.write(reinterpret_cast<const char*>(pixels.data()),
                     static_cast<std::streamsize>(pixels.size()));
        require(output.good(), "could not write render output");
        std::cout << "{\"schema\":\"numi.neuron-culture.render.v1\",\"fingerprint\":"
                  << culture.fingerprint() << ",\"output\":\"" << outputPath
                  << "\",\"width\":512,\"height\":512,\"mea_spikes\":"
                  << totalSpikes(state) << "}\n";
        return;
    }
    throw std::runtime_error("unknown command");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 1) {
            runQualification();
            return 0;
        }
        const std::string_view command(argv[1]);
        bool quick = false;
        bool diagnosticSwitch = false;
        bool diagnosticCalibration = false;
        bool diagnosticLearning = false;
        std::uint32_t windowLimit = 0u;
        std::string outputPath;
        auto ablation = metalrobo::PotterProtocolAblation::none;
        std::uint64_t networkSeed = 2056u;
        std::uint32_t sensoryMapping = 0u;
        std::string packOut;
        std::string checkpointIn;
        std::string checkpointOut;
        std::string embodyTarget = "animat";
        std::string benchmarkMode;
        for (int i = 2; i < argc; ++i) {
            if (std::string_view(argv[i]) == "--quick") quick = true;
            else if (std::string_view(argv[i]) == "--diagnostic-switch")
                diagnosticSwitch = true;
            else if (std::string_view(argv[i]) == "--diagnostic-calibration")
                diagnosticCalibration = true;
            else if (std::string_view(argv[i]) == "--diagnostic-learning")
                diagnosticLearning = true;
            else if (std::string_view(argv[i]) == "--output" && i + 1 < argc) {
                outputPath = argv[++i];
            }
            else if (std::string_view(argv[i]) == "--ablation" && i + 1 < argc) {
                const std::string_view value(argv[++i]);
                if (value == "none") ablation = metalrobo::PotterProtocolAblation::none;
                else if (value == "adaptive-off")
                    ablation = metalrobo::PotterProtocolAblation::adaptiveSelectionOff;
                else if (value == "pts-off")
                    ablation = metalrobo::PotterProtocolAblation::patternedTrainingOff;
                else if (value == "stdp-off")
                    ablation = metalrobo::PotterProtocolAblation::stdpOff;
                else throw std::runtime_error("unknown protocol ablation");
            }
            else if (std::string_view(argv[i]) == "--network-seed" && i + 1 < argc) {
                networkSeed = std::stoull(argv[++i]);
                if (networkSeed == 0u) throw std::runtime_error("network seed must be nonzero");
            }
            else if (std::string_view(argv[i]) == "--sensory-mapping" && i + 1 < argc) {
                sensoryMapping = static_cast<std::uint32_t>(std::stoul(argv[++i]));
                if (sensoryMapping >= 5u) throw std::runtime_error("sensory mapping must be 0...4");
            }
            else if (std::string_view(argv[i]) == "--pack-out" && i + 1 < argc)
                packOut = argv[++i];
            else if (std::string_view(argv[i]) == "--checkpoint-in" && i + 1 < argc)
                checkpointIn = argv[++i];
            else if (std::string_view(argv[i]) == "--checkpoint-out" && i + 1 < argc)
                checkpointOut = argv[++i];
            else if (std::string_view(argv[i]) == "--target" && i + 1 < argc) {
                embodyTarget = argv[++i];
                if (embodyTarget != "animat" && embodyTarget != "numanx")
                    throw std::runtime_error("embodiment target must be animat|numanx");
            }
            else if (std::string_view(argv[i]) == "--mode" && i + 1 < argc)
                benchmarkMode = argv[++i];
            else if (std::string_view(argv[i]) == "--window-limit" && i + 1 < argc) {
                windowLimit = static_cast<std::uint32_t>(std::stoul(argv[++i]));
                if (windowLimit == 0u) throw std::runtime_error("window limit must be positive");
            }
            else if (std::string_view(argv[i]) == "--preset" && i + 1 < argc) {
                if (std::string_view(argv[++i]) != "potter-switch-v1")
                    throw std::runtime_error("unknown protocol preset");
            }
            else throw std::runtime_error("usage: metalrobo_neuron_culture_probe [compile|grow|simulate|protocol|qualify|embody|benchmark|inspect|replay|render] [--quick] [--output PATH]");
        }
        if (static_cast<unsigned>(diagnosticSwitch) +
                static_cast<unsigned>(diagnosticCalibration) +
                static_cast<unsigned>(diagnosticLearning) > 1u)
            throw std::runtime_error("choose one protocol diagnostic mode");
        runLab(command, quick, outputPath, ablation, networkSeed, sensoryMapping,
               packOut, checkpointIn, checkpointOut, embodyTarget, benchmarkMode,
               diagnosticSwitch, diagnosticCalibration, diagnosticLearning,
               windowLimit);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "neuron_culture_error=" << error.what() << '\n';
        return 1;
    }
}
