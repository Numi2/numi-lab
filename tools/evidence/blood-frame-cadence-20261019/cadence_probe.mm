#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../source/matter/tools/human_resting_runtime.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NUMI_MATTER_METALLIB
#error NUMI_MATTER_METALLIB is required
#endif
#ifndef NUMI_HUMAN_RESPIRATION_SOURCE
#error NUMI_HUMAN_RESPIRATION_SOURCE is required
#endif

namespace {
using namespace numi::human;

struct Trace {
    Respiration* respiration = nullptr;
    std::uint32_t microtickCount = 0u;
    std::uint32_t frameBegin = 0u;
    std::uint32_t candidateReady = 0u;
    std::uint32_t microstepComplete = 0u;
    std::uint32_t frameComplete = 0u;
};

bool traceCallback(void* opaque, AcceptedStepExtensionPhase phase,
                   const AcceptedStepExtensionView& view) {
    auto& trace = *static_cast<Trace*>(opaque);
    trace.microtickCount = std::max(trace.microtickCount, view.microtickCount);
    switch (phase) {
        case AcceptedStepExtensionPhase::frameBegin: ++trace.frameBegin; break;
        case AcceptedStepExtensionPhase::candidateReady: ++trace.candidateReady; break;
        case AcceptedStepExtensionPhase::microstepComplete: ++trace.microstepComplete; break;
        case AcceptedStepExtensionPhase::frameComplete: ++trace.frameComplete; break;
        default: break;
    }
    return Respiration::callback(trace.respiration, phase, view);
}

struct Result {
    NMHumanRespirationState initial{};
    NMHumanRespirationState final{};
    NBNumiRespiratoryChemoreflexStateV1 brain{};
    std::uint64_t expectedBrainRoot = 0u;
    RuntimeStateSnapshot runtime;
    Trace trace;
    std::uint32_t exponent = 0u;
    std::uint32_t activeReceiverParticles = 0u;
    std::size_t receiverParticles = 0u;
    std::array<std::uint32_t, 3> receiverGridDimensions{};
    std::uint64_t receiverGridNodeCount = 0u;
};

Result runOne(bool bloodMpm, const char* network, const char* configuration,
              const char* material) {
    if (setenv("NUMI_HUMAN_BLOOD_MPM_ENABLED", bloodMpm ? "1" : "0", 1) != 0)
        throw std::runtime_error("could not set MPM opt-in");
    if (bloodMpm) {
        if (setenv("NUMI_HUMAN_BLOOD_MPM_MATERIAL", material, 1) != 0)
            throw std::runtime_error("could not set blood material source path");
        if (setenv("NUMI_HUMAN_BLOOD_MPM_GRID_BOUNDS_M",
                   "-0.25,-0.25,-0.25,0.25,0.25,0.25", 1) != 0)
            throw std::runtime_error("could not set bounded dormant-grid extent");
    }
    if (setenv("NUMI_HUMAN_BLEEDING_ENABLED", "0", 1) != 0)
        throw std::runtime_error("could not disable outlet for cadence control");

    RestingRun run(network, configuration, 0.00025f, true);
    RespiratoryBrain brain(*run.respiration, run.world, configuration);
    Result result;
    result.initial = *static_cast<const NMHumanRespirationState*>(run.respiration->accepted.contents);
    result.exponent = run.world.dispatch.maximumRateExponent;
    if (bloodMpm) {
        result.receiverParticles = run.world.mpm.particles.size();
        if (run.world.mpm.grids.size() != 1u)
            throw std::runtime_error("expected exactly one cooked receiver grid");
        const auto& grid = run.world.mpm.grids.front();
        result.receiverGridDimensions = {grid.nodeDimensionsAndOffset.x,
                                         grid.nodeDimensionsAndOffset.y,
                                         grid.nodeDimensionsAndOffset.z};
        result.receiverGridNodeCount =
            std::uint64_t(result.receiverGridDimensions[0]) *
            std::uint64_t(result.receiverGridDimensions[1]) *
            std::uint64_t(result.receiverGridDimensions[2]);
    }
    result.trace.respiration = run.respiration.get();

    *static_cast<MRMetalWorldStatusGPU*>(run.statuses.contents) = {};
    id<MTLCommandBuffer> commandBuffer = [run.queue commandBuffer];
    need(commandBuffer != nil, "command-buffer allocation failed");
    EncodeRequest request{};
    request.commandBuffer = (__bridge void*)commandBuffer;
    request.environmentStatuses = (__bridge void*)run.statuses;
    request.controlStep = 0u;
    request.physicsSubstep = 0u;
    request.physicsSubsteps = 1u;
    request.timestepSeconds = run.runtime.timestepSeconds();
    request.runAdaptiveTransfer = false;
    request.acceptedStepExtensionContext = &result.trace;
    request.encodeAcceptedStepExtension = &traceCallback;
    request.phase = EncodePhase::preDynamics;
    auto encoded = run.runtime.encode(request);
    need(encoded.encoded, "preDynamics encode failed: " + encoded.message);
    request.phase = EncodePhase::postCommit;
    encoded = run.runtime.encode(request);
    need(encoded.encoded, "postCommit encode failed: " + encoded.message);
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    need(commandBuffer.status == MTLCommandBufferStatusCompleted,
         "Metal command buffer did not complete");

    result.final = *static_cast<const NMHumanRespirationState*>(run.respiration->accepted.contents);
    result.brain = *static_cast<const NBNumiRespiratoryChemoreflexStateV1*>(brain.accepted.contents);
    result.expectedBrainRoot = brain.root(1u);
    result.trace.respiration = nullptr;
    result.runtime = run.runtime.snapshot();
    need(result.runtime.available, "runtime snapshot unavailable: " + result.runtime.message);
    if (bloodMpm) {
        for (const auto& particle : result.runtime.particles)
            if ((particle.identity.w & NM_OBJECT_ACTIVE) != 0u) ++result.activeReceiverParticles;
    }
    return result;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

float maxVascularDifference(const std::vector<nm_float4>& a,
                            const std::vector<nm_float4>& b) {
    require(a.size() == b.size(), "vascular snapshot shapes differ");
    float maximum = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float* av = &a[i].x;
        const float* bv = &b[i].x;
        for (unsigned c = 0; c < 4; ++c)
            maximum = std::max(maximum, std::abs(av[c] - bv[c]));
    }
    return maximum;
}

void printCase(const char* name, const Result& result) {
    const auto clock = result.runtime.vascularClock.empty()
        ? NMVascularClockGPU{} : result.runtime.vascularClock.front();
    std::cout << std::setprecision(10)
              << "case=" << name
              << " accepted_frames=" << result.final.status.x
              << " cursor_control_step=" << result.runtime.controlStep
              << " cursor_physics_substep=" << result.runtime.physicsSubstep
              << " frame_begin_calls=" << result.trace.frameBegin
              << " candidate_ready_calls=" << result.trace.candidateReady
              << " microstep_complete_calls=" << result.trace.microstepComplete
              << " frame_complete_calls=" << result.trace.frameComplete
              << " reported_microticks=" << result.trace.microtickCount
              << " rate_exponent=" << result.exponent
              << " vascular_clock_high=" << clock.high
              << " vascular_clock_low=" << clock.low
              << " brain_time_us=" << result.brain.timestampMicroseconds
              << " brain_accepted_root=" << result.brain.acceptedRootFingerprint
              << " brain_expected_root=" << result.expectedBrainRoot
              << " receiver_particles=" << result.receiverParticles
              << " active_receiver_particles=" << result.activeReceiverParticles
              << " receiver_grid_x=" << result.receiverGridDimensions[0]
              << " receiver_grid_y=" << result.receiverGridDimensions[1]
              << " receiver_grid_z=" << result.receiverGridDimensions[2]
              << " receiver_grid_nodes=" << result.receiverGridNodeCount
              << std::endl;
}
}

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            require(argc == 4, "usage: cadence_probe NETWORK CONFIG MATERIAL");
            const Result baseline = runOne(false, argv[1], argv[2], argv[3]);
            const Result subcycled = runOne(true, argv[1], argv[2], argv[3]);
            printCase("baseline", baseline);
            printCase("blood_mpm", subcycled);
            require(baseline.final.status.x == 1u && subcycled.final.status.x == 1u,
                    "respiration did not publish exactly one accepted frame");
            require(baseline.runtime.controlStep == 0u && subcycled.runtime.controlStep == 0u,
                    "Matter accepted cursor differs from one body frame");
            require(baseline.runtime.vascularClock.size() == 1u &&
                    subcycled.runtime.vascularClock.size() == 1u,
                    "expected one vascular environment");
            const auto bclock = baseline.runtime.vascularClock[0];
            const auto mclock = subcycled.runtime.vascularClock[0];
            require(bclock.low == mclock.low && bclock.high == mclock.high,
                    "vascular accepted clock changed with MPM internal rate");
            require(baseline.trace.frameBegin == 1u && baseline.trace.frameComplete == 1u &&
                    baseline.trace.microtickCount == 1u,
                    "baseline callback did not execute exactly one full frame");
            require(subcycled.exponent == NM_MAX_RATE_EXPONENT &&
                    subcycled.trace.microtickCount == (1u << NM_MAX_RATE_EXPONENT),
                    "enabled MPM world did not exercise the configured full internal rate");
            require(subcycled.trace.frameBegin == 1u && subcycled.trace.frameComplete == 1u &&
                    subcycled.trace.candidateReady == subcycled.trace.microtickCount &&
                    subcycled.trace.microstepComplete == subcycled.trace.microtickCount,
                    "internal Matter callback cadence did not match reported microticks");
            require(baseline.brain.timestampMicroseconds == 250u &&
                    subcycled.brain.timestampMicroseconds == 250u &&
                    baseline.brain.acceptedRootFingerprint == baseline.expectedBrainRoot &&
                    subcycled.brain.acceptedRootFingerprint == subcycled.expectedBrainRoot,
                    "NumiBrain did not accept the same single full-frame target");
            require(subcycled.receiverParticles == 128u &&
                    subcycled.activeReceiverParticles == 0u,
                    "outlet-disabled control changed the receiver capacity or activated a particle");
            require(baseline.runtime.vascularState.size() == subcycled.runtime.vascularState.size(),
                    "vascular state shape differs");

            const double baseMetabolicO2 = baseline.final.metabolicGas.x - baseline.initial.metabolicGas.x;
            const double mpmMetabolicO2 = subcycled.final.metabolicGas.x - subcycled.initial.metabolicGas.x;
            const double baseMetabolicCO2 = baseline.final.metabolicGas.y - baseline.initial.metabolicGas.y;
            const double mpmMetabolicCO2 = subcycled.final.metabolicGas.y - subcycled.initial.metabolicGas.y;
            const double o2Ratio = std::abs(mpmMetabolicO2) /
                std::max(std::abs(baseMetabolicO2), 1.0e-18);
            const double co2Ratio = std::abs(mpmMetabolicCO2) /
                std::max(std::abs(baseMetabolicCO2), 1.0e-18);
            require(std::isfinite(o2Ratio) && std::isfinite(co2Ratio) &&
                    o2Ratio < 8.0 && co2Ratio < 8.0,
                    "metabolic gas exchange appears multiplied by the internal microtick count");
            const double vascularMaximum = maxVascularDifference(
                baseline.runtime.vascularState, subcycled.runtime.vascularState);
            require(std::isfinite(vascularMaximum), "nonfinite vascular state comparison");

            std::cout << std::setprecision(10)
                      << "cadence_control=PASS frame_dt_s=0.00025"
                      << " baseline_microticks=" << baseline.trace.microtickCount
                      << " enabled_microticks=" << subcycled.trace.microtickCount
                      << " enabled_exponent=" << subcycled.exponent
                      << " accepted_steps=" << subcycled.final.status.x
                      << " vascular_clock_high=" << mclock.high
                      << " vascular_clock_low=" << mclock.low
                      << " vascular_state_max_abs_delta=" << vascularMaximum
                      << " metabolic_o2_frame_delta=" << mpmMetabolicO2
                      << " metabolic_o2_baseline_delta=" << baseMetabolicO2
                      << " metabolic_o2_ratio=" << o2Ratio
                      << " metabolic_co2_frame_delta=" << mpmMetabolicCO2
                      << " metabolic_co2_baseline_delta=" << baseMetabolicCO2
                      << " metabolic_co2_ratio=" << co2Ratio
                      << " blood_volume_m3=" << subcycled.final.circulation.x
                      << " gas_budget_o2_max_error_m3=" << subcycled.final.gasBudget.z
                      << " gas_budget_co2_max_error_m3=" << subcycled.final.gasBudget.w
                      << " blood_continuity_residual_m3=" << subcycled.final.bloodBalance.continuityResidualSumM3
                      << " brain_target_time_us=" << subcycled.brain.timestampMicroseconds
                      << " baseline_cursor_control_step=" << baseline.runtime.controlStep
                      << " enabled_cursor_control_step=" << subcycled.runtime.controlStep
                      << " receiver_grid_nodes=" << subcycled.receiverGridNodeCount
                      << " receiver_grid_dims=" << subcycled.receiverGridDimensions[0]
                      << "," << subcycled.receiverGridDimensions[1]
                      << "," << subcycled.receiverGridDimensions[2]
                      << " receiver_particles=" << subcycled.receiverParticles
                      << " active_receiver_particles=" << subcycled.activeReceiverParticles
                      << " comparison_scope=one_no_outlet_frame_only_not_long_horizon_qualification\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "cadence_control=FAIL reason=" << error.what() << '\n';
            return 1;
        }
    }
}
