#pragma once

#ifndef __METAL_VERSION__
#include <stdint.h>
#endif

#ifdef __METAL_VERSION__
#define MRNC_ALIGN(N) __attribute__((aligned(N)))
#else
#define MRNC_ALIGN(N) alignas(N)
#endif

#define MR_NEURON_CULTURE_ABI_VERSION 1u
#define MR_NEURON_CULTURE_MAX_ELECTRODES 64u
#define MR_NEURON_CULTURE_STATUS_PENDING 0u
#define MR_NEURON_CULTURE_STATUS_SUCCESS 1u
#define MR_NEURON_CULTURE_STATUS_INVALID 2u

typedef struct MRNC_ALIGN(16) MRNeuronCultureHeaderGPU {
    uint32_t abiVersion;
    uint32_t structBytes;
    uint32_t neuronCount;
    uint32_t synapseCount;
    uint32_t electrodeCount;
    uint32_t growthWidth;
    uint32_t growthHeight;
    uint32_t flags;
    uint64_t cultureFingerprint;
    uint64_t seed;
    float neuralTimestepSeconds;
    float membraneTimeConstantSeconds;
    float restingPotential;
    float resetPotential;
    float thresholdPotential;
    float refractorySeconds;
    float traceTimeConstantSeconds;
    float depressionRecoverySeconds;
    float stdpPotentiation;
    float stdpDepression;
    float minimumWeight;
    float maximumWeight;
    uint32_t reserved0;
    uint32_t reserved1;
} MRNeuronCultureHeaderGPU;

typedef struct MRNC_ALIGN(16) MRNeuronCultureNeuronGPU {
    float x;
    float y;
    float biasCurrent;
    float capacitance;
    uint32_t incomingBegin;
    uint32_t incomingCount;
    uint32_t excitatory;
    uint32_t reserved0;
} MRNeuronCultureNeuronGPU;

typedef struct MRNC_ALIGN(16) MRNeuronCultureSynapseGPU {
    uint32_t presynaptic;
    uint32_t postsynaptic;
    uint32_t delayTicks;
    uint32_t plastic;
    float initialWeight;
    float depressionUse;
    float reserved0;
    float reserved1;
} MRNeuronCultureSynapseGPU;

typedef struct MRNC_ALIGN(16) MRNeuronCultureElectrodeGPU {
    float x;
    float y;
    float recordingRadius;
    float stimulationRadius;
    uint32_t active;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} MRNeuronCultureElectrodeGPU;

typedef struct MRNC_ALIGN(16) MRNeuronCultureTickGPU {
    uint32_t tick;
    uint32_t stimulationElectrode;
    uint32_t stimulationEnabled;
    uint32_t status;
    float stimulationCurrent;
    float traceDecay;
    float depressionRecovery;
    float reserved0;
} MRNeuronCultureTickGPU;

typedef struct MRNC_ALIGN(16) MRNeuronCultureGrowthGPU {
    uint32_t width;
    uint32_t height;
    uint32_t iteration;
    uint32_t stage;
    float timestep;
    float phaseMobility;
    float interfaceCoefficient;
    float tubulinDiffusion;
    float tubulinDecay;
    float tubulinSource;
    float growthDrive;
    float newtonTolerance;
    uint32_t newtonIterations;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} MRNeuronCultureGrowthGPU;

#ifndef __METAL_VERSION__
static_assert(sizeof(MRNeuronCultureHeaderGPU) == 112u);
static_assert(sizeof(MRNeuronCultureNeuronGPU) == 32u);
static_assert(sizeof(MRNeuronCultureSynapseGPU) == 32u);
static_assert(sizeof(MRNeuronCultureElectrodeGPU) == 32u);
static_assert(sizeof(MRNeuronCultureTickGPU) == 32u);
static_assert(sizeof(MRNeuronCultureGrowthGPU) == 64u);
#endif

#undef MRNC_ALIGN
