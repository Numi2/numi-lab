#include "metalrobo/MetalNeuronCulture.hpp"
#include "metalrobo/NeuronCulture.hpp"
#include "metalrobo/NeuronCultureEmbodiment.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
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
        a.tick == b.tick && a.growthIteration == b.growthIteration;
}

std::uint64_t totalSpikes(const metalrobo::NeuronCultureState& state) {
    return std::accumulate(state.electrodeSpikeCounts.begin(),
                           state.electrodeSpikeCounts.end(), std::uint64_t{0});
}

void emitInspect(const metalrobo::CompiledNeuronCulture& culture) {
    const auto& h = culture.header();
    std::cout << "{\"schema\":\"numi.neuron-culture.inspect.v1\""
              << ",\"fingerprint\":" << culture.fingerprint()
              << ",\"neurons\":" << h.neuronCount
              << ",\"synapses\":" << h.synapseCount
              << ",\"electrodes\":" << h.electrodeCount
              << ",\"growth_grid\":[" << h.growthWidth << ',' << h.growthHeight << ']'
              << ",\"synthetic_only\":true"
              << ",\"growth_stages\":[1,2,3,4]"
              << ",\"maturation_stage_5\":false"
              << ",\"automatic_synapse_from_crossing\":false}\n";
}

metalrobo::CompiledNeuronCulture compileCulture(std::uint32_t neurons,
                                                 std::uint32_t synapses) {
    auto pack = metalrobo::makePotterReferenceCulture(neurons, synapses, 2056u);
    metalrobo::CompiledNeuronCulture culture;
    const auto diagnostics = metalrobo::compileNeuronCulture(pack, culture);
    require(diagnostics.succeeded(), diagnostics.message);
    require(culture.valid(), "compiled culture is invalid");
    return culture;
}

void runQualification() {
    auto culture = compileCulture(64u, 512u);
    auto invalidPack = metalrobo::makePotterReferenceCulture(16u, 64u, 9u);
    invalidPack.synapses.front().postsynaptic = 99u;
    metalrobo::CompiledNeuronCulture preserved = culture;
    const auto invalid = metalrobo::compileNeuronCulture(invalidPack, preserved);
    require(!invalid.succeeded() && preserved.fingerprint() == culture.fingerprint(),
            "invalid topology did not fail transactionally");

    metalrobo::NeuronCultureReference cpu(culture);
    require(cpu.valid(), "CPU reference is invalid");
    const auto initial = cpu.accepted();
    require(cpu.prepareTicks(64u, 0u, 900.0f), "CPU tick prepare failed");
    cpu.rejectPrepared();
    require(sameAccepted(initial, cpu.accepted()), "CPU rejected ticks changed accepted state");
    require(cpu.prepareTicks(64u, 0u, 900.0f) && cpu.publishPrepared(),
            "CPU accepted ticks failed");
    const auto cpuTick = cpu.accepted();
    require(cpuTick.tick == 64u && totalSpikes(cpuTick) > 0u,
            "virtual MEA observed no evoked activity");

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

    metalrobo::NeuronCultureReference replay(culture);
    require(replay.prepareTicks(64u, 0u, 900.0f) && replay.publishPrepared(),
            "CPU replay failed");
    require(sameAccepted(cpuTick, replay.accepted()), "CPU replay is not bitwise deterministic");

    auto plasticOffPack = metalrobo::makePotterReferenceCulture(64u, 512u, 2056u);
    plasticOffPack.id = "potter-embodied-mea-synthetic-stdp-off-v1";
    plasticOffPack.network.stdpPotentiation = 0.0f;
    plasticOffPack.network.stdpDepression = 0.0f;
    metalrobo::CompiledNeuronCulture plasticOffCulture;
    require(metalrobo::compileNeuronCulture(plasticOffPack, plasticOffCulture).succeeded(),
            "STDP-off ablation did not compile");
    metalrobo::NeuronCultureReference plasticOff(plasticOffCulture);
    const auto plasticOffInitialWeights = plasticOff.accepted().weights;
    require(plasticOff.prepareTicks(256u, 0u, 900.0f) && plasticOff.publishPrepared() &&
            exact(plasticOffInitialWeights, plasticOff.accepted().weights),
            "STDP-off ablation changed synaptic weights");

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
    auto ticket = gpu.prepareTicks(64u, 0u, 900.0f);
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

    const auto acceptedGPU = gpuTick;
    auto rejected = gpu.prepareTicks(8u, 1u, 1100.0f);
    require(rejected.valid() && rejected.wait() == metalrobo::MetalNeuronCultureStatus::success,
            "Metal rejected candidate did not execute");
    gpu.rejectPrepared();
    require(sameAccepted(acceptedGPU, gpu.snapshotAcceptedForTesting()),
            "Metal rejected candidate changed accepted neural state");

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
    std::cout << "{\"schema\":\"numi.neuron-culture.qualification.v1\""
              << ",\"device\":\"" << gpu.deviceName() << "\""
              << ",\"fingerprint\":" << culture.fingerprint()
              << ",\"cpu_metal_parity\":true"
              << ",\"transactional_reject\":true"
              << ",\"bitwise_cpu_replay\":true"
              << ",\"virtual_mea_spikes\":" << totalSpikes(cpuTick)
              << ",\"growth_iterations\":3"
              << ",\"potter_reference\":{\"neurons\":1000,\"synapses\":50000,\"electrodes\":60}"
              << ",\"wet_lab\":false}\n";
}

void runLab(std::string_view command, bool quick, const std::string& outputPath) {
    const std::uint32_t neurons = quick ? 64u : 1000u;
    const std::uint32_t synapses = quick ? 512u : 50000u;
    auto culture = compileCulture(neurons, synapses);
    if (command == "compile" || command == "inspect") {
        emitInspect(culture);
        return;
    }
    if (command == "grow") {
        auto gpu = metalrobo::MetalNeuronCultureRuntime::create(culture);
        require(gpu.valid(), "Metal runtime unavailable");
        const std::uint32_t iterations = quick ? 4u : 32u;
        auto ticket = gpu.prepareGrowth(iterations);
        require(ticket.valid() && ticket.wait() == metalrobo::MetalNeuronCultureStatus::success,
                "growth command failed");
        require(gpu.publishPrepared() == metalrobo::MetalNeuronCultureStatus::success,
                "growth publication failed");
        const auto state = gpu.snapshotAcceptedForTesting();
        const double phaseMass = std::accumulate(state.phase.begin(), state.phase.end(), 0.0);
        std::cout << "{\"schema\":\"numi.neuron-culture.growth.v1\",\"fingerprint\":"
                  << culture.fingerprint() << ",\"iterations\":" << iterations
                  << ",\"phase_mass\":" << std::setprecision(12) << phaseMass
                  << ",\"published\":true}\n";
        return;
    }
    if (command == "simulate" || command == "embody") {
        auto gpu = metalrobo::MetalNeuronCultureRuntime::create(culture);
        require(gpu.valid(), "Metal runtime unavailable");
        const std::uint32_t windows = quick ? 4u : 12u;
        double x = 0.0;
        double y = 0.0;
        for (std::uint32_t window = 0u; window < windows; ++window) {
            auto ticket = gpu.prepareTicks(100u, window % culture.header().electrodeCount, 900.0f);
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
    if (command == "replay") {
        metalrobo::NeuronCultureReference a(culture);
        metalrobo::NeuronCultureReference b(culture);
        require(a.prepareTicks(200u, 0u, 900.0f) && a.publishPrepared() &&
                b.prepareTicks(200u, 0u, 900.0f) && b.publishPrepared() &&
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
        auto neural = gpu.prepareTicks(quick ? 100u : 400u, 0u, 900.0f);
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
        std::string outputPath;
        for (int i = 2; i < argc; ++i) {
            if (std::string_view(argv[i]) == "--quick") quick = true;
            else if (std::string_view(argv[i]) == "--output" && i + 1 < argc) {
                outputPath = argv[++i];
            }
            else throw std::runtime_error("usage: metalrobo_neuron_culture_probe [compile|grow|simulate|embody|inspect|replay] [--quick]");
        }
        runLab(command, quick, outputPath);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "neuron_culture_error=" << error.what() << '\n';
        return 1;
    }
}
