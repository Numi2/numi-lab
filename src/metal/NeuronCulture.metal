#include <metal_stdlib>
#include "metalrobo/neuron_culture_gpu.h"
#include "metalrobo/numanx_human_io_gpu.h"

using namespace metal;

inline float mr_neuron_culture_membrane_noise(
    ulong seed, uint tick, uint neuron, float amplitude
) {
    ulong value = seed ^
        ((ulong(tick) + 1ul) * 0x9e3779b97f4a7c15ul) ^
        ((ulong(neuron) + 1ul) * 0xbf58476d1ce4e5b9ul);
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ul;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebul;
    value ^= value >> 31u;
    const int difference = int(value & 0xfffful) -
        int((value >> 16u) & 0xfffful);
    return amplitude * float(difference) * (1.0f / 65535.0f);
}

inline float mr_neuron_culture_suppression(
    const float intervalTrace,
    const float traceSeconds,
    const float suppressionSeconds
) {
    return 1.0f - powr(clamp(intervalTrace, 0.0f, 1.0f),
                       traceSeconds / suppressionSeconds);
}

kernel void mr_neuron_culture_support_schedule(
    constant MRNeuronCultureHeaderGPU& header [[buffer(0)]],
    device const MRNeuronCultureElectrodeGPU* electrodes [[buffer(1)]],
    constant MRNeuronCultureSupportDispatchGPU& dispatch [[buffer(2)]],
    device const MRNumanXHumanSupportConsequenceGPU* consequences [[buffer(3)]],
    device float* currents [[buffer(4)]],
    uint lane [[thread_position_in_grid]]) {
    if (lane != 0u || dispatch.status != MR_NEURON_CULTURE_STATUS_PENDING ||
        dispatch.supportCount != 10u || dispatch.supportStride != 10u ||
        dispatch.electrodeCount != header.electrodeCount ||
        dispatch.tickCount == 0u || dispatch.physicsTimestepSeconds <= 0.0f ||
        dispatch.currentPerNewton < 0.0f || dispatch.reserved0 != 0u) return;
    float2 weighted = float2(0.0f);
    float impulse = 0.0f;
    for (uint index = 0u; index < 10u; ++index) {
        const MRNumanXHumanSupportConsequenceGPU item = consequences[index];
        if (item.identity.x != index ||
            item.identity.w != MR_NUMANX_HUMAN_SUPPORT_CONSEQUENCE_VERSION ||
            !isfinite(item.pointAndSeparation.x) ||
            !isfinite(item.pointAndSeparation.y) ||
            !isfinite(item.impulseAndNormal.w) || item.impulseAndNormal.w < 0.0f) return;
        weighted += item.pointAndSeparation.xy * item.impulseAndNormal.w;
        impulse += item.impulseAndNormal.w;
    }
    if (impulse == 0.0f) return;
    const float2 culturePoint = float2(1.5f) + 3.0f *
        clamp(weighted / impulse, float2(-0.5f), float2(0.5f));
    uint nearest = 0u;
    float distance = INFINITY;
    for (uint index = 0u; index < header.electrodeCount; ++index) {
        const float2 delta = float2(electrodes[index].x, electrodes[index].y) -
            culturePoint;
        const float candidate = dot(delta, delta);
        if (candidate < distance) { distance = candidate; nearest = index; }
    }
    currents[nearest] = clamp(
        impulse / dispatch.physicsTimestepSeconds * dispatch.currentPerNewton,
        0.0f, 5000.0f);
}

kernel void mr_neuron_culture_window(
    constant MRNeuronCultureHeaderGPU& header [[buffer(0)]],
    device const MRNeuronCultureNeuronGPU* neurons [[buffer(1)]],
    device const MRNeuronCultureSynapseGPU* synapses [[buffer(2)]],
    device const MRNeuronCultureElectrodeGPU* electrodes [[buffer(3)]],
    constant MRNeuronCultureWindowGPU& window [[buffer(4)]],
    device const float* electrodeCurrents [[buffer(5)]],
    device float* membrane [[buffer(6)]],
    device float* refractory [[buffer(7)]],
    device float* preTrace [[buffer(8)]],
    device float* postTrace [[buffer(9)]],
    device float* weights [[buffer(10)]],
    device float* depression [[buffer(11)]],
    device uint* spikeHistory [[buffer(12)]],
    device uint* spikes [[buffer(13)]],
    device uint* electrodeSpikeCounts [[buffer(14)]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (window.status != MR_NEURON_CULTURE_STATUS_PENDING) return;
    for (uint offset = 0u; offset < window.tickCount; ++offset) {
        const uint absoluteTick = window.startTick + offset;
        for (uint neuronIndex = lane; neuronIndex < header.neuronCount;
             neuronIndex += 256u) {
            if (spikes[neuronIndex] != 0u) {
                postTrace[neuronIndex] = preTrace[neuronIndex];
                preTrace[neuronIndex] = 1.0f;
            }
            preTrace[neuronIndex] *= window.traceDecay;
        }
        threadgroup_barrier(mem_flags::mem_device);
        for (uint neuronIndex = lane; neuronIndex < header.neuronCount; neuronIndex += 256u) {
            const MRNeuronCultureNeuronGPU neuron = neurons[neuronIndex];
            float current = mr_neuron_culture_membrane_noise(
                header.seed, absoluteTick, neuronIndex, neuron.biasCurrent);
            for (uint local = 0u; local < neuron.incomingCount; ++local) {
                const uint edge = neuron.incomingBegin + local;
                const MRNeuronCultureSynapseGPU synapse = synapses[edge];
                const uint delayedTick = (absoluteTick + 256u - synapse.delayTicks) & 255u;
                if (spikeHistory[delayedTick * header.neuronCount + synapse.presynaptic] != 0u) {
                    const float sign = neurons[synapse.presynaptic].excitatory != 0u ? 1.0f : -1.0f;
                    current += sign * weights[edge] * depression[edge] *
                        header.synapticCurrentScale;
                }
            }
            const uint currentBase = offset * header.electrodeCount;
            for (uint electrodeIndex = 0u; electrodeIndex < header.electrodeCount;
                 ++electrodeIndex) {
                const float stimulus = electrodeCurrents[currentBase + electrodeIndex];
                if (stimulus != 0.0f) {
                    const MRNeuronCultureElectrodeGPU electrode = electrodes[electrodeIndex];
                    const float2 delta = float2(neuron.x - electrode.x, neuron.y - electrode.y);
                    if (dot(delta, delta) <= electrode.stimulationRadius * electrode.stimulationRadius)
                        current += stimulus;
                }
            }
            const float dt = header.neuralTimestepSeconds;
            float nextRefractory = max(0.0f, refractory[neuronIndex] - dt);
            float nextMembrane = membrane[neuronIndex];
            uint fired = 0u;
            if (nextRefractory > 0.0f) {
                nextMembrane = header.resetPotential;
            } else {
                nextMembrane += dt * ((header.restingPotential - nextMembrane) /
                    header.membraneTimeConstantSeconds + current / neuron.capacitance);
                if (nextMembrane >= header.thresholdPotential) {
                    fired = 1u;
                    nextMembrane = header.resetPotential;
                    nextRefractory = header.refractorySeconds;
                }
            }
            membrane[neuronIndex] = nextMembrane;
            refractory[neuronIndex] = nextRefractory;
            spikes[neuronIndex] = fired;
        }
        threadgroup_barrier(mem_flags::mem_device);
        for (uint edge = lane; edge < header.synapseCount; edge += 256u) {
            const MRNeuronCultureSynapseGPU synapse = synapses[edge];
            float nextDepression = depression[edge] +
                (1.0f - depression[edge]) * window.depressionRecovery;
            if (spikes[synapse.presynaptic] != 0u)
                nextDepression *= (1.0f - synapse.depressionUse);
            depression[edge] = clamp(nextDepression, 0.0f, 1.0f);
            if (synapse.plastic != 0u &&
                (window.flags & MR_NEURON_CULTURE_WINDOW_DISABLE_PLASTICITY) == 0u) {
                const float weight = weights[edge];
                float normalizedChange = 0.0f;
                const float preIntervalEfficacy = mr_neuron_culture_suppression(
                    postTrace[synapse.presynaptic], header.traceTimeConstantSeconds,
                    header.preSpikeSuppressionTimeConstantSeconds);
                const float postIntervalEfficacy = mr_neuron_culture_suppression(
                    postTrace[synapse.postsynaptic], header.traceTimeConstantSeconds,
                    header.postSpikeSuppressionTimeConstantSeconds);
                if (spikes[synapse.postsynaptic] != 0u) {
                    const float currentPostEfficacy = mr_neuron_culture_suppression(
                        preTrace[synapse.postsynaptic], header.traceTimeConstantSeconds,
                        header.postSpikeSuppressionTimeConstantSeconds);
                    normalizedChange += header.stdpPotentiation *
                        (header.maximumWeight - weight) *
                        preTrace[synapse.presynaptic] * preIntervalEfficacy *
                        currentPostEfficacy;
                }
                if (spikes[synapse.presynaptic] != 0u) {
                    const float currentPreEfficacy = mr_neuron_culture_suppression(
                        preTrace[synapse.presynaptic], header.traceTimeConstantSeconds,
                        header.preSpikeSuppressionTimeConstantSeconds);
                    normalizedChange -= header.stdpDepression *
                        (weight - header.minimumWeight) *
                        preTrace[synapse.postsynaptic] * currentPreEfficacy *
                        postIntervalEfficacy;
                }
                weights[edge] = clamp(weight * (1.0f + normalizedChange),
                    header.minimumWeight, header.maximumWeight);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        if (offset >= window.recordingStartTick &&
            offset - window.recordingStartTick < window.recordingDurationTicks) {
            for (uint electrodeIndex = lane; electrodeIndex < header.electrodeCount;
                 electrodeIndex += 256u) {
                const MRNeuronCultureElectrodeGPU electrode = electrodes[electrodeIndex];
                uint count = 0u;
                for (uint neuron = 0u; neuron < header.neuronCount; ++neuron) {
                    const float2 delta = float2(neurons[neuron].x - electrode.x,
                                                neurons[neuron].y - electrode.y);
                    count += spikes[neuron] != 0u && dot(delta, delta) <=
                        electrode.recordingRadius * electrode.recordingRadius;
                }
                electrodeSpikeCounts[electrodeIndex] += count;
            }
        }
        for (uint neuronIndex = lane; neuronIndex < header.neuronCount; neuronIndex += 256u)
            spikeHistory[(absoluteTick & 255u) * header.neuronCount + neuronIndex] =
                spikes[neuronIndex];
        threadgroup_barrier(mem_flags::mem_device);
    }
}

kernel void mr_neuron_culture_tick(
    constant MRNeuronCultureHeaderGPU& header [[buffer(0)]],
    device const MRNeuronCultureNeuronGPU* neurons [[buffer(1)]],
    device const MRNeuronCultureSynapseGPU* synapses [[buffer(2)]],
    device const MRNeuronCultureElectrodeGPU* electrodes [[buffer(3)]],
    constant MRNeuronCultureTickGPU& tick [[buffer(4)]],
    device float* membrane [[buffer(5)]],
    device float* refractory [[buffer(6)]],
    device float* preTrace [[buffer(7)]],
    device float* postTrace [[buffer(8)]],
    device const float* weights [[buffer(9)]],
    device const float* depression [[buffer(10)]],
    device const uint* spikeHistory [[buffer(11)]],
    device uint* spikes [[buffer(12)]],
    constant float* electrodeCurrents [[buffer(13)]],
    uint neuronIndex [[thread_position_in_grid]]) {
    if (neuronIndex >= header.neuronCount || tick.status != MR_NEURON_CULTURE_STATUS_PENDING) return;
    const MRNeuronCultureNeuronGPU neuron = neurons[neuronIndex];
    if (spikes[neuronIndex] != 0u) {
        postTrace[neuronIndex] = preTrace[neuronIndex];
        preTrace[neuronIndex] = 1.0f;
    }
    preTrace[neuronIndex] *= tick.traceDecay;
    float current = mr_neuron_culture_membrane_noise(
        header.seed, tick.tick, neuronIndex, neuron.biasCurrent);
    for (uint local = 0u; local < neuron.incomingCount; ++local) {
        const uint edge = neuron.incomingBegin + local;
        const MRNeuronCultureSynapseGPU synapse = synapses[edge];
        const uint delayedTick = (tick.tick + 256u - synapse.delayTicks) & 255u;
        if (spikeHistory[delayedTick * header.neuronCount + synapse.presynaptic] != 0u) {
            const float sign = neurons[synapse.presynaptic].excitatory != 0u ? 1.0f : -1.0f;
            current += sign * weights[edge] * depression[edge] *
                header.synapticCurrentScale;
        }
    }
    for (uint electrodeIndex = 0u; electrodeIndex < header.electrodeCount; ++electrodeIndex) {
        const float stimulus = electrodeCurrents[electrodeIndex];
        if (stimulus != 0.0f) {
            const MRNeuronCultureElectrodeGPU electrode = electrodes[electrodeIndex];
            const float2 delta = float2(neuron.x - electrode.x, neuron.y - electrode.y);
            if (dot(delta, delta) <= electrode.stimulationRadius * electrode.stimulationRadius) {
                current += stimulus;
            }
        }
    }
    const float dt = header.neuralTimestepSeconds;
    float nextRefractory = max(0.0f, refractory[neuronIndex] - dt);
    float nextMembrane = membrane[neuronIndex];
    uint fired = 0u;
    if (nextRefractory > 0.0f) {
        nextMembrane = header.resetPotential;
    } else {
        nextMembrane += dt * ((header.restingPotential - nextMembrane) /
            header.membraneTimeConstantSeconds + current / neuron.capacitance);
        if (nextMembrane >= header.thresholdPotential) {
            fired = 1u;
            nextMembrane = header.resetPotential;
            nextRefractory = header.refractorySeconds;
        }
    }
    membrane[neuronIndex] = nextMembrane;
    refractory[neuronIndex] = nextRefractory;
    spikes[neuronIndex] = fired;
}

kernel void mr_neuron_culture_plasticity(
    constant MRNeuronCultureHeaderGPU& header [[buffer(0)]],
    device const MRNeuronCultureSynapseGPU* synapses [[buffer(1)]],
    constant MRNeuronCultureTickGPU& tick [[buffer(2)]],
    device const uint* spikes [[buffer(3)]],
    device const float* preTrace [[buffer(4)]],
    device const float* postTrace [[buffer(5)]],
    device float* weights [[buffer(6)]],
    device float* depression [[buffer(7)]],
    uint edge [[thread_position_in_grid]]) {
    if (edge >= header.synapseCount || tick.status != MR_NEURON_CULTURE_STATUS_PENDING) return;
    const MRNeuronCultureSynapseGPU synapse = synapses[edge];
    float nextDepression = depression[edge] + (1.0f - depression[edge]) * tick.depressionRecovery;
    if (spikes[synapse.presynaptic] != 0u) {
        nextDepression *= (1.0f - synapse.depressionUse);
    }
    depression[edge] = clamp(nextDepression, 0.0f, 1.0f);
    if (synapse.plastic != 0u) {
        const float weight = weights[edge];
        float normalizedChange = 0.0f;
        const float preIntervalEfficacy = mr_neuron_culture_suppression(
            postTrace[synapse.presynaptic], header.traceTimeConstantSeconds,
            header.preSpikeSuppressionTimeConstantSeconds);
        const float postIntervalEfficacy = mr_neuron_culture_suppression(
            postTrace[synapse.postsynaptic], header.traceTimeConstantSeconds,
            header.postSpikeSuppressionTimeConstantSeconds);
        if (spikes[synapse.postsynaptic] != 0u) {
            const float currentPostEfficacy = mr_neuron_culture_suppression(
                preTrace[synapse.postsynaptic], header.traceTimeConstantSeconds,
                header.postSpikeSuppressionTimeConstantSeconds);
            normalizedChange += header.stdpPotentiation *
                (header.maximumWeight - weight) * preTrace[synapse.presynaptic] *
                preIntervalEfficacy * currentPostEfficacy;
        }
        if (spikes[synapse.presynaptic] != 0u) {
            const float currentPreEfficacy = mr_neuron_culture_suppression(
                preTrace[synapse.presynaptic], header.traceTimeConstantSeconds,
                header.preSpikeSuppressionTimeConstantSeconds);
            normalizedChange -= header.stdpDepression *
                (weight - header.minimumWeight) * preTrace[synapse.postsynaptic] *
                currentPreEfficacy * postIntervalEfficacy;
        }
        weights[edge] = clamp(weight * (1.0f + normalizedChange),
            header.minimumWeight, header.maximumWeight);
    }
}

kernel void mr_neuron_culture_record(
    constant MRNeuronCultureHeaderGPU& header [[buffer(0)]],
    device const MRNeuronCultureNeuronGPU* neurons [[buffer(1)]],
    device const MRNeuronCultureElectrodeGPU* electrodes [[buffer(2)]],
    device const uint* spikes [[buffer(3)]],
    device uint* electrodeSpikeCounts [[buffer(4)]],
    constant MRNeuronCultureTickGPU& tick [[buffer(5)]],
    uint electrodeIndex [[thread_position_in_grid]]) {
    if (electrodeIndex >= header.electrodeCount || tick.recordingEnabled == 0.0f) return;
    const MRNeuronCultureElectrodeGPU electrode = electrodes[electrodeIndex];
    uint count = 0u;
    for (uint neuron = 0u; neuron < header.neuronCount; ++neuron) {
        const float2 delta = float2(neurons[neuron].x - electrode.x,
                                    neurons[neuron].y - electrode.y);
        count += spikes[neuron] != 0u &&
            dot(delta, delta) <= electrode.recordingRadius * electrode.recordingRadius;
    }
    electrodeSpikeCounts[electrodeIndex] += count;
}

kernel void mr_neuron_culture_store_history(
    constant MRNeuronCultureHeaderGPU& header [[buffer(0)]],
    constant MRNeuronCultureTickGPU& tick [[buffer(1)]],
    device const uint* spikes [[buffer(2)]],
    device uint* spikeHistory [[buffer(3)]],
    uint neuronIndex [[thread_position_in_grid]]) {
    if (neuronIndex >= header.neuronCount) return;
    spikeHistory[(tick.tick & 255u) * header.neuronCount + neuronIndex] = spikes[neuronIndex];
}

kernel void mr_neuron_culture_growth(
    constant MRNeuronCultureGrowthGPU& growth [[buffer(0)]],
    device const float* phase [[buffer(1)]],
    device const float* tubulin [[buffer(2)]],
    device float* nextPhase [[buffer(3)]],
    device float* nextTubulin [[buffer(4)]],
    uint cell [[thread_position_in_grid]]) {
    const uint count = growth.width * growth.height;
    if (cell >= count || growth.width == 0u || growth.height == 0u) return;
    const uint x = cell % growth.width;
    const uint y = cell / growth.width;
    const uint left = y * growth.width + (x == 0u ? x : x - 1u);
    const uint right = y * growth.width + min(x + 1u, growth.width - 1u);
    const uint down = (y == 0u ? y : y - 1u) * growth.width + x;
    const uint up = min(y + 1u, growth.height - 1u) * growth.width + x;
    const float oldPhi = phase[cell];
    const float oldTubulin = tubulin[cell];
    const float lapPhi = phase[left] + phase[right] + phase[down] + phase[up] - 4.0f * oldPhi;
    const float lapTubulin = tubulin[left] + tubulin[right] + tubulin[down] + tubulin[up] -
        4.0f * oldTubulin;
    float phi = oldPhi;
    for (uint iteration = 0u; iteration < growth.newtonIterations; ++iteration) {
        const float reaction = phi * (1.0f - phi) * (phi - 0.5f);
        const float drive = growth.growthDrive * oldTubulin * phi * (1.0f - phi);
        const float residual = phi - oldPhi - growth.timestep * growth.phaseMobility *
            (growth.interfaceCoefficient * growth.interfaceCoefficient * lapPhi + reaction + drive);
        const float derivative = 1.0f - growth.timestep * growth.phaseMobility *
            ((-3.0f * phi * phi + 3.0f * phi - 0.5f) +
             growth.growthDrive * oldTubulin * (1.0f - 2.0f * phi));
        if (!isfinite(derivative) || abs(derivative) < 1.0e-8f) {
            phi = oldPhi;
            break;
        }
        const float step = residual / derivative;
        phi = clamp(phi - step, 0.0f, 1.0f);
        if (abs(step) <= growth.newtonTolerance) break;
    }
    nextPhase[cell] = phi;
    nextTubulin[cell] = max(0.0f,
        (oldTubulin + growth.timestep *
            (growth.tubulinDiffusion * lapTubulin + growth.tubulinSource * phi)) /
        (1.0f + growth.timestep * growth.tubulinDecay));
}

struct MRNeuronCultureVisualParams {
    uint width;
    uint height;
    uint growthWidth;
    uint growthHeight;
    uint neuronCount;
    uint synapseCount;
    uint electrodeCount;
    uint acceptedGeneration;
    float animatX;
    float animatY;
    uint protocolPhase;
    uint transactionStatus;
    float minimumWeight;
    float maximumWeight;
    uint histogramBins;
    uint reserved0;
};

kernel void mr_neuron_culture_visual_histogram(
    constant MRNeuronCultureVisualParams& p [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* depression [[buffer(2)]],
    device atomic_uint* weightHistogram [[buffer(3)]],
    device atomic_uint* depressionHistogram [[buffer(4)]],
    uint edge [[thread_position_in_grid]]) {
    if (edge >= p.synapseCount || p.histogramBins == 0u ||
        !(p.maximumWeight > p.minimumWeight)) return;
    const float normalizedWeight = clamp(
        (weights[edge] - p.minimumWeight) /
            (p.maximumWeight - p.minimumWeight), 0.0f, 1.0f);
    const float normalizedDepression = clamp(depression[edge], 0.0f, 1.0f);
    const uint weightBin = min(p.histogramBins - 1u,
        uint(normalizedWeight * float(p.histogramBins)));
    const uint depressionBin = min(p.histogramBins - 1u,
        uint(normalizedDepression * float(p.histogramBins)));
    atomic_fetch_add_explicit(&weightHistogram[weightBin], 1u,
        memory_order_relaxed);
    atomic_fetch_add_explicit(&depressionHistogram[depressionBin], 1u,
        memory_order_relaxed);
}

kernel void mr_neuron_culture_visualize(
    constant MRNeuronCultureVisualParams& p [[buffer(0)]],
    device const MRNeuronCultureNeuronGPU* neurons [[buffer(1)]],
    device const MRNeuronCultureElectrodeGPU* electrodes [[buffer(2)]],
    device const float* phase [[buffer(3)]],
    device const float* tubulin [[buffer(4)]],
    device const uint* spikes [[buffer(5)]],
    device const uint* spikeHistory [[buffer(6)]],
    device const uint* meaCounts [[buffer(7)]],
    device const atomic_uint* weightHistogram [[buffer(8)]],
    device const atomic_uint* depressionHistogram [[buffer(9)]],
    device const int* neuronMap [[buffer(10)]],
    device const int* electrodeMap [[buffer(11)]],
    texture2d<float, access::write> output [[texture(0)]],
    uint2 pixel [[thread_position_in_grid]]) {
    if (pixel.x >= p.width || pixel.y >= p.height) return;
    const float2 uv = (float2(pixel) + 0.5f) / float2(p.width, p.height);
    float3 color = float3(0.015f, 0.02f, 0.03f);
    if (uv.x < 0.5f && uv.y < 0.5f) {
        const float2 local = uv * 2.0f;
        const uint gx = min(p.growthWidth - 1u, uint(local.x * p.growthWidth));
        const uint gy = min(p.growthHeight - 1u, uint(local.y * p.growthHeight));
        const uint cell = gy * p.growthWidth + gx;
        const float a = clamp(phase[cell], 0.0f, 1.0f);
        const float b = clamp(tubulin[cell], 0.0f, 1.0f);
        color = float3(0.08f + 0.35f * a, 0.03f + 0.85f * b, 0.12f + 0.45f * a);
    } else if (uv.x >= 0.5f && uv.y < 0.5f) {
        const uint mapX = min(255u, uint((uv.x - 0.5f) * 512.0f));
        const uint mapY = min(255u, uint(uv.y * 512.0f));
        const uint mapIndex = mapY * 256u + mapX;
        const int neuron = neuronMap[mapIndex];
        if (neuron >= 0 && uint(neuron) < p.neuronCount) {
            const uint neuronIndex = uint(neuron);
            color = spikes[neuronIndex] != 0u ? float3(1.0f, 0.9f, 0.35f) :
                (neurons[neuronIndex].excitatory != 0u ?
                    float3(0.2f, 0.65f, 0.9f) : float3(0.95f, 0.25f, 0.45f));
        }
        const int electrode = electrodeMap[mapIndex];
        if (electrode >= 0 && uint(electrode) < p.electrodeCount) {
            const uint electrodeIndex = uint(electrode);
            const float activity = min(1.0f,
                float(meaCounts[electrodeIndex] & 1023u) / 128.0f);
            color = mix(float3(0.25f, 0.08f, 0.08f),
                        float3(1.0f, 0.3f, 0.1f), activity);
        }
    } else if (uv.x < 0.5f) {
        const float2 local = float2(uv.x * 2.0f, (uv.y - 0.5f) * 2.0f);
        const uint historyTick = min(255u, uint(local.x * 256.0f));
        const uint neuron = min(p.neuronCount - 1u, uint(local.y * p.neuronCount));
        const float fired = float(spikeHistory[historyTick * p.neuronCount + neuron] != 0u);
        color = mix(float3(0.015f, 0.02f, 0.035f), float3(0.2f, 1.0f, 0.65f), fired);
    } else {
        const float2 local = float2((uv.x - 0.5f) * 2.0f, (uv.y - 0.5f) * 2.0f);
        const uint bin = min(p.histogramBins - 1u,
            uint(local.x * float(p.histogramBins)));
        const float denominator = log2(1.0f + float(max(p.synapseCount, 1u)));
        const float weightHeight = log2(1.0f + float(atomic_load_explicit(
            &weightHistogram[bin], memory_order_relaxed))) / denominator;
        const float depressionHeight = log2(1.0f + float(atomic_load_explicit(
            &depressionHistogram[bin], memory_order_relaxed))) / denominator;
        color = float3(0.015f, 0.02f, 0.035f);
        const float heightFromBottom = 1.0f - local.y;
        if (heightFromBottom <= depressionHeight)
            color = float3(0.08f, 0.65f, 0.38f);
        if (heightFromBottom <= weightHeight)
            color = mix(color, float3(0.12f, 0.68f, 1.0f), 0.72f);
        const float generationBar = min(1.0f, float(p.acceptedGeneration & 255u) / 255.0f);
        if (local.y > 0.94f) color = float3(0.1f, 0.4f + 0.6f * generationBar, 0.85f);
        const float2 animat = clamp(
            float2(p.animatX, p.animatY) / 100.0f + 0.5f,
            float2(0.02f), float2(0.98f));
        if (distance(local, animat) < 0.018f) color = float3(1.0f, 0.85f, 0.15f);
        if (local.y < 0.025f) {
            const float3 states[3] = {
                float3(0.15f, 0.85f, 0.45f),
                float3(0.95f, 0.65f, 0.12f),
                float3(0.95f, 0.18f, 0.28f),
            };
            color = states[min(p.transactionStatus, 2u)];
        } else if (local.y < 0.05f) {
            const float protocolBand = float(min(p.protocolPhase, 3u)) / 3.0f;
            color = float3(0.2f + 0.7f * protocolBand, 0.25f,
                           0.8f - 0.5f * protocolBand);
        }
    }
    if (abs(uv.x - 0.5f) < 0.0015f || abs(uv.y - 0.5f) < 0.002f)
        color = float3(0.25f, 0.32f, 0.42f);
    output.write(float4(color, 1.0f), pixel);
}
