#include <metal_stdlib>
#include "metalrobo/neuron_culture_gpu.h"

using namespace metal;

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
    uint neuronIndex [[thread_position_in_grid]]) {
    if (neuronIndex >= header.neuronCount || tick.status != MR_NEURON_CULTURE_STATUS_PENDING) return;
    const MRNeuronCultureNeuronGPU neuron = neurons[neuronIndex];
    float current = neuron.biasCurrent;
    for (uint local = 0u; local < neuron.incomingCount; ++local) {
        const uint edge = neuron.incomingBegin + local;
        const MRNeuronCultureSynapseGPU synapse = synapses[edge];
        const uint delayedTick = (tick.tick + 256u - synapse.delayTicks) & 255u;
        if (spikeHistory[delayedTick * header.neuronCount + synapse.presynaptic] != 0u) {
            const float sign = neurons[synapse.presynaptic].excitatory != 0u ? 1.0f : -1.0f;
            current += sign * weights[edge] * depression[edge] * 45.0f;
        }
    }
    if (tick.stimulationEnabled != 0u && tick.stimulationElectrode < header.electrodeCount) {
        const MRNeuronCultureElectrodeGPU electrode = electrodes[tick.stimulationElectrode];
        const float2 delta = float2(neuron.x - electrode.x, neuron.y - electrode.y);
        if (dot(delta, delta) <= electrode.stimulationRadius * electrode.stimulationRadius) {
            current += tick.stimulationCurrent;
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
    preTrace[neuronIndex] = preTrace[neuronIndex] * tick.traceDecay + float(fired);
    postTrace[neuronIndex] = postTrace[neuronIndex] * tick.traceDecay + float(fired);
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
        float delta = 0.0f;
        if (spikes[synapse.postsynaptic] != 0u) {
            delta += header.stdpPotentiation * preTrace[synapse.presynaptic];
        }
        if (spikes[synapse.presynaptic] != 0u) {
            delta -= header.stdpDepression * postTrace[synapse.postsynaptic];
        }
        weights[edge] = clamp(weights[edge] + delta, header.minimumWeight, header.maximumWeight);
    }
}

kernel void mr_neuron_culture_record(
    constant MRNeuronCultureHeaderGPU& header [[buffer(0)]],
    device const MRNeuronCultureNeuronGPU* neurons [[buffer(1)]],
    device const MRNeuronCultureElectrodeGPU* electrodes [[buffer(2)]],
    device const uint* spikes [[buffer(3)]],
    device uint* electrodeSpikeCounts [[buffer(4)]],
    uint electrodeIndex [[thread_position_in_grid]]) {
    if (electrodeIndex >= header.electrodeCount) return;
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
