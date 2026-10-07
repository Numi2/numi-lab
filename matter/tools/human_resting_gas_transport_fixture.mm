#include "human_resting_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

using namespace numi::human;

namespace {
constexpr unsigned kPrefixSteps = 65u;
constexpr unsigned kTransitionStep = 65u;
constexpr float kLegacyDt = 0.0021f;
constexpr float kSubcyclingDt = 0.008f;
constexpr unsigned kMaximumGasSubsteps = 32u;

void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

struct EnvironmentValue {
    std::string name;
    bool existed = false;
    std::string value;
    explicit EnvironmentValue(const char* key): name(key) {
        const char* prior = std::getenv(key);
        if (prior) { existed = true; value = prior; }
    }
    ~EnvironmentValue() {
        if (existed) setenv(name.c_str(), value.c_str(), 1);
        else unsetenv(name.c_str());
    }
};

struct VascularProbe {
    __strong id<MTLDevice> device = nil;
    __strong id<MTLBuffer> before = nil;
    __strong id<MTLBuffer> after = nil;
    __strong id<MTLBuffer> unknowns = nil;
    __strong id<MTLBuffer> connections = nil;
    __strong id<MTLBuffer> compartments = nil;
    __strong id<MTLBuffer> elastance = nil;
    unsigned targetStep = MR_INVALID_INDEX;
    bool captured = false;

    explicit VascularProbe(id<MTLDevice> d, unsigned step):
        device(d), targetStep(step) {}

    bool encode(AcceptedStepExtensionPhase phase,
                const AcceptedStepExtensionView& view) {
        if (phase != AcceptedStepExtensionPhase::candidateReady ||
            view.controlStep != targetStep) return true;
        check(!captured, "vascular probe captured a control step twice");
        check(view.vascularStateStride == 45u && view.compartmentCount == 21u &&
              view.connectionCount == 24u && view.environmentCount == 1u,
              "vascular probe saw an unexpected CVSim21 layout");
        before = [device newBufferWithLength:view.vascularStateStride *
            view.environmentCount * sizeof(nm_float4)
            options:MTLResourceStorageModeShared];
        after = [device newBufferWithLength:view.vascularStateStride *
            view.environmentCount * sizeof(nm_float4)
            options:MTLResourceStorageModeShared];
        unknowns = [device newBufferWithLength:view.vascularStateStride *
            sizeof(NMVascularUnknownGPU) options:MTLResourceStorageModeShared];
        connections = [device newBufferWithLength:view.connectionCount *
            sizeof(NMVascularConnectionGPU) options:MTLResourceStorageModeShared];
        compartments = [device newBufferWithLength:view.compartmentCount *
            sizeof(NMVascularCompartmentGPU) options:MTLResourceStorageModeShared];
        elastance = [device newBufferWithLength:view.environmentCount *
            view.compartmentCount * sizeof(float) options:MTLResourceStorageModeShared];
        check(before && after && unknowns && connections && compartments && elastance,
              "vascular probe allocation failed");

        // Snapshot borrowed runtime inputs into fixture-owned buffers in the
        // command stream; the callback never retains or reads borrowed buffers.
        id<MTLCommandBuffer> command =
            (__bridge id<MTLCommandBuffer>)view.commandBuffer;
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        check(blit != nil, "vascular probe could not encode snapshot copies");
        [blit copyFromBuffer:buffer(view.vascularAccepted) sourceOffset:0
                    toBuffer:before destinationOffset:0 size:before.length];
        [blit copyFromBuffer:buffer(view.vascularCandidate) sourceOffset:0
                    toBuffer:after destinationOffset:0 size:after.length];
        [blit copyFromBuffer:buffer(view.vascularUnknowns) sourceOffset:0
                    toBuffer:unknowns destinationOffset:0 size:unknowns.length];
        [blit copyFromBuffer:buffer(view.vascularConnections) sourceOffset:0
                    toBuffer:connections destinationOffset:0 size:connections.length];
        [blit copyFromBuffer:buffer(view.vascularCompartments) sourceOffset:0
                    toBuffer:compartments destinationOffset:0 size:compartments.length];
        [blit copyFromBuffer:buffer(view.vascularElastance) sourceOffset:0
                    toBuffer:elastance destinationOffset:0 size:elastance.length];
        [blit endEncoding];
        captured = true;
        return true;
    }
};

void verifyBoundedCap(RestingRun& run, const VascularProbe& probe);

struct Rig {
    std::unique_ptr<RestingRun> run;
    std::unique_ptr<RespiratoryBrain> brain;
};

Rig makeRig(const char* network, const char* configuration, float dt,
            bool subcycling, VascularProbe* probe = nullptr) {
    setenv("NUMI_HUMAN_RESTING_TIMESTEP_SENSITIVITY", "1", 1);
    setenv("NUMI_HUMAN_GAS_TRANSPORT_SUBCYCLING", subcycling ? "1" : "0", 1);
    Rig result;
    result.run = std::make_unique<RestingRun>(network, configuration, dt);
    result.brain = std::make_unique<RespiratoryBrain>(
        *result.run->respiration, result.run->world, configuration);
    if (probe) {
        auto previous = result.run->respiration->brain;
        result.run->respiration->brain =
            [previous, probe](AcceptedStepExtensionPhase phase,
                              const AcceptedStepExtensionView& view) {
                if (previous && !previous(phase, view)) return false;
                return probe->encode(phase, view);
            };
    }
    return result;
}

std::vector<nm_float4> copyFloat4(id<MTLBuffer> source, unsigned count) {
    check(source && source.length >= size_t(count) * sizeof(nm_float4),
          "captured vascular state buffer is too short");
    const auto* data = static_cast<const nm_float4*>(source.contents);
    return std::vector<nm_float4>(data, data + count);
}

double requiredSubsteps(const NMHumanRespirationParameters& p,
                        const NMHumanRespirationState& candidate,
                        const VascularProbe& probe) {
    check(probe.captured, "required-substep calculation lacks captured inputs");
    constexpr unsigned nv = 45u;
    const auto before = copyFloat4(probe.before, nv);
    const auto after = copyFloat4(probe.after, nv);
    const auto* unknowns =
        static_cast<const NMVascularUnknownGPU*>(probe.unknowns.contents);
    const auto* connections =
        static_cast<const NMVascularConnectionGPU*>(probe.connections.contents);
    float outgoing[21]{};
    for (unsigned edge = 0; edge < p.topology.w; ++edge) {
        const float flow =
            after[21u + edge].x * unknowns[21u + edge].initialAndScaling.y;
        const unsigned from = flow >= 0.0f
            ? connections[edge].identity.y : connections[edge].identity.z;
        check(from < 21u && std::isfinite(flow),
              "captured vascular connection has invalid donor or flow");
        outgoing[from] += std::abs(flow);
    }
    double required = 1.0;
    for (unsigned row = 0; row < 21u; ++row) {
        const float v0 = before[row].x * unknowns[row].initialAndScaling.y;
        const float v1 = after[row].x * unknowns[row].initialAndScaling.y;
        const float minimum = std::min(v0, v1);
        check(minimum > 0.0f && std::isfinite(minimum) &&
              std::isfinite(outgoing[row]),
              "captured vascular donor volume is invalid");
        required = std::max(required, std::ceil(double(p.environment.w) *
            double(outgoing[row]) / (0.1 * double(minimum))));
    }
    required = std::max(required, std::ceil(double(p.environment.w) *
        std::abs(double(candidate.mechanics.w)) / (0.1 * double(p.lung.y))));
    return required;
}

void requireSnapshotPhysicalEqual(const RuntimeStateSnapshot& a,
                                  const RuntimeStateSnapshot& b,
                                  const char* message) {
    check(a.available && b.available, "runtime snapshot unavailable");
    check(a.controlStep == b.controlStep &&
          a.vascularState.size() == b.vascularState.size() &&
          a.vascularClock.size() == b.vascularClock.size() &&
          (!a.vascularState.size() ||
           std::memcmp(a.vascularState.data(), b.vascularState.data(),
                       a.vascularState.size() * sizeof(nm_float4)) == 0) &&
          (!a.vascularClock.size() ||
           std::memcmp(a.vascularClock.data(), b.vascularClock.data(),
                       a.vascularClock.size() * sizeof(NMVascularClockGPU)) == 0),
          message);
}

void verifySingleSubstepEquivalence(const char* network,
                                   const char* configuration) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    VascularProbe legacyProbe(device, 0u);
    VascularProbe candidateProbe(device, 0u);
    auto legacy = makeRig(network, configuration, kLegacyDt, false, &legacyProbe);
    legacy.run->batch(0u, 1u);
    const auto legacyState =
        *static_cast<const NMHumanRespirationState*>(legacy.run->respiration->accepted.contents);
    const auto legacySnapshot = legacy.run->runtime.snapshot();
    const auto legacyCandidate =
        *static_cast<const NMHumanRespirationState*>(legacy.run->respiration->candidate.contents);
    check(requiredSubsteps(legacy.run->respiration->parameters,
                           legacyCandidate, legacyProbe) == 1.0,
          "legacy one-step fixture unexpectedly requires gas subcycling");

    auto candidate = makeRig(network, configuration, kLegacyDt, true, &candidateProbe);
    candidate.run->batch(0u, 1u);
    const auto candidateState =
        *static_cast<const NMHumanRespirationState*>(candidate.run->respiration->accepted.contents);
    const auto candidateSnapshot = candidate.run->runtime.snapshot();
    const auto candidateCandidate =
        *static_cast<const NMHumanRespirationState*>(candidate.run->respiration->candidate.contents);
    check(requiredSubsteps(candidate.run->respiration->parameters,
                           candidateCandidate, candidateProbe) == 1.0,
          "enabled one-step fixture did not select exactly one substep");
    check(std::memcmp(&legacyState, &candidateState, sizeof(legacyState)) == 0 &&
          std::memcmp(&legacyCandidate, &candidateCandidate, sizeof(legacyCandidate)) == 0,
          "N=1 subcycling pipeline differs from legacy respiration state bitwise");
    requireSnapshotPhysicalEqual(legacySnapshot, candidateSnapshot,
          "N=1 subcycling changed accepted vascular state or clock");
}

void verifySubcyclingAndTransactions(const char* network,
                                    const char* configuration) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    VascularProbe legacyProbe(device, kTransitionStep);
    auto legacy = makeRig(network, configuration, kSubcyclingDt, false, &legacyProbe);
    legacy.run->batch(0u, kPrefixSteps);
    const auto prefixState =
        *static_cast<const NMHumanRespirationState*>(legacy.run->respiration->accepted.contents);
    const auto prefixBrain =
        *static_cast<const NBNumiRespiratoryChemoreflexStateV1*>(legacy.brain->accepted.contents);
    const auto prefixSnapshot = legacy.run->runtime.snapshot();
    check(prefixState.status.x == kPrefixSteps,
          "legacy 8 ms run did not reach the expected subcycling transition");

    legacy.run->batch(kTransitionStep, 1u, true);
    const auto legacyFailed =
        *static_cast<const NMHumanRespirationState*>(legacy.run->respiration->candidate.contents);
    check(legacyFailed.status.w == 4u,
          "legacy full-step donor CFL did not retain its rejection at the transition");
    check(requiredSubsteps(legacy.run->respiration->parameters,
                           legacyFailed, legacyProbe) > 1.0,
          "transition fixture did not require more than one gas substep");
    const auto afterLegacyFailure = legacy.run->runtime.snapshot();
    const auto afterLegacyBody =
        *static_cast<const NMHumanRespirationState*>(legacy.run->respiration->accepted.contents);
    const auto afterLegacyBrain =
        *static_cast<const NBNumiRespiratoryChemoreflexStateV1*>(legacy.brain->accepted.contents);
    requireSnapshotPhysicalEqual(prefixSnapshot, afterLegacyFailure,
          "legacy CFL rejection advanced vascular state or clock");
    check(std::memcmp(&prefixState, &afterLegacyBody, sizeof(prefixState)) == 0 &&
          std::memcmp(&prefixBrain, &afterLegacyBrain, sizeof(prefixBrain)) == 0,
          "legacy CFL rejection advanced respiration or controller history");

    VascularProbe subcycleProbe(device, kTransitionStep);
    auto subcycled = makeRig(network, configuration, kSubcyclingDt, true, &subcycleProbe);
    subcycled.run->batch(0u, kPrefixSteps);
    subcycled.run->batch(kTransitionStep, 1u);
    const auto accepted =
        *static_cast<const NMHumanRespirationState*>(subcycled.run->respiration->accepted.contents);
    const auto acceptedCandidate =
        *static_cast<const NMHumanRespirationState*>(subcycled.run->respiration->candidate.contents);
    const double actualRequired = requiredSubsteps(
        subcycled.run->respiration->parameters, acceptedCandidate, subcycleProbe);
    check(actualRequired > 1.0 && actualRequired <= kMaximumGasSubsteps,
          "accepted transition did not use a bounded multi-substep gas candidate");
    check(accepted.status.x == kPrefixSteps + 1u && accepted.status.w == 0u,
          "subcycled transition was not accepted exactly once");
    for (const auto& gas : accepted.bloodGas)
        check(std::isfinite(gas.x) && std::isfinite(gas.y) &&
              gas.x >= 0.0f && gas.y >= 0.0f,
              "subcycled blood gas amount is negative or nonfinite");
    for (float amount : {accepted.alveolarGas.x, accepted.alveolarGas.y,
                         accepted.deadSpaceGas.x, accepted.deadSpaceGas.y})
        check(std::isfinite(amount) && amount >= 0.0f,
              "subcycled lung gas amount is negative or nonfinite");
    check(std::isfinite(accepted.gasBudget.z) &&
          std::isfinite(accepted.gasBudget.w) &&
          accepted.gasBudget.z <= 5.e-5f * accepted.gasBudget.x &&
          accepted.gasBudget.w <= 5.e-5f * accepted.gasBudget.y,
          "subcycled candidate exceeded the existing gas conservation budget");

    const auto beforeRejectedStep = subcycled.run->runtime.snapshot();
    const auto bodyBeforeRejectedStep = accepted;
    const auto brainBeforeRejectedStep =
        *static_cast<const NBNumiRespiratoryChemoreflexStateV1*>(subcycled.brain->accepted.contents);
    subcycled.run->respiration->dispatch.reject = 1u;
    subcycled.run->batch(kTransitionStep + 1u, 1u, true, kTransitionStep + 1u);
    const auto bodyAfterRejectedStep =
        *static_cast<const NMHumanRespirationState*>(subcycled.run->respiration->accepted.contents);
    const auto brainAfterRejectedStep =
        *static_cast<const NBNumiRespiratoryChemoreflexStateV1*>(subcycled.brain->accepted.contents);
    const auto afterRejectedStep = subcycled.run->runtime.snapshot();
    requireSnapshotPhysicalEqual(beforeRejectedStep, afterRejectedStep,
          "rejected subcycling transaction advanced vascular state or clock");
    check(std::memcmp(&bodyBeforeRejectedStep, &bodyAfterRejectedStep,
                      sizeof(bodyBeforeRejectedStep)) == 0 &&
          std::memcmp(&brainBeforeRejectedStep, &brainAfterRejectedStep,
                      sizeof(brainBeforeRejectedStep)) == 0,
          "rejected subcycling transaction advanced body or controller history");

    subcycled.run->respiration->dispatch.reject = 0u;
    subcycled.run->batch(kTransitionStep + 1u, 1u);
    const auto replayTarget = *static_cast<const NMHumanRespirationState*>(
        subcycled.run->respiration->accepted.contents);
    const auto replayBrainTarget = *static_cast<const NBNumiRespiratoryChemoreflexStateV1*>(
        subcycled.brain->accepted.contents);
    const auto replaySnapshot = subcycled.run->runtime.snapshot();
    const auto restored = subcycled.run->runtime.restore(beforeRejectedStep);
    check(restored.encoded, "could not restore pre-rejection runtime checkpoint");
    std::memcpy(subcycled.run->respiration->accepted.contents,
                &bodyBeforeRejectedStep, sizeof(bodyBeforeRejectedStep));
    std::memcpy(subcycled.brain->accepted.contents,
                &brainBeforeRejectedStep, sizeof(brainBeforeRejectedStep));
    subcycled.run->batch(kTransitionStep + 1u, 1u);
    const auto replayed = *static_cast<const NMHumanRespirationState*>(
        subcycled.run->respiration->accepted.contents);
    const auto replayedBrain = *static_cast<const NBNumiRespiratoryChemoreflexStateV1*>(
        subcycled.brain->accepted.contents);
    const auto replayedSnapshot = subcycled.run->runtime.snapshot();
    requireSnapshotPhysicalEqual(replaySnapshot, replayedSnapshot,
          "subcycled accepted replay changed vascular state or clock");
    check(std::memcmp(&replayTarget, &replayed, sizeof(replayTarget)) == 0 &&
          std::memcmp(&replayBrainTarget, &replayedBrain, sizeof(replayBrainTarget)) == 0,
          "subcycled accepted replay changed respiration/controller state");

    verifyBoundedCap(*subcycled.run, subcycleProbe);
}

void verifyBoundedCap(RestingRun& run, const VascularProbe& probe) {
    // Bypass only the native circuit solve to construct a finite,
    // source-layout-correct exchange input whose donor-CFL needs >32
    // microsteps. This exercises the selected GPU pipeline without changing
    // RestingRun's accepted state.
    check(probe.captured, "32-step-cap fixture lacks source vascular inputs");
    const auto* sourceBefore = static_cast<const nm_float4*>(probe.before.contents);
    const auto* sourceAfter = static_cast<const nm_float4*>(probe.after.contents);
    const auto* sourceUnknowns =
        static_cast<const NMVascularUnknownGPU*>(probe.unknowns.contents);
    const auto* sourceConnections =
        static_cast<const NMVascularConnectionGPU*>(probe.connections.contents);
    std::vector<nm_float4> before(sourceBefore, sourceBefore + 45u);
    std::vector<nm_float4> after(sourceAfter, sourceAfter + 45u);
    std::vector<NMVascularUnknownGPU> unknowns(sourceUnknowns, sourceUnknowns + 45u);
    std::vector<NMVascularConnectionGPU> connections(sourceConnections,
                                                       sourceConnections + 24u);
    for (unsigned edge = 0; edge < 24u; ++edge) after[21u + edge] = {};
    const unsigned donor = connections[0].identity.y;
    check(donor < 21u, "cap fixture connection has invalid donor");
    const float donorVolume =
        before[donor].x * unknowns[donor].initialAndScaling.y;
    const float flowScale = unknowns[21u].initialAndScaling.y;
    check(donorVolume > 0.0f && flowScale > 0.0f &&
          std::isfinite(donorVolume) && std::isfinite(flowScale),
          "cap fixture lacks positive physical donor and flow scales");
    const float syntheticFlow = 500.0f * donorVolume;
    after[21u].x = syntheticFlow / flowScale;

    const auto initial =
        *static_cast<const NMHumanRespirationState*>(run.respiration->accepted.contents);
    id<MTLBuffer> accepted = [run.device newBufferWithBytes:&initial
        length:sizeof(initial) options:MTLResourceStorageModeShared];
    id<MTLBuffer> candidate = [run.device newBufferWithBytes:&initial
        length:sizeof(initial) options:MTLResourceStorageModeShared];
    id<MTLBuffer> beforeBuffer = [run.device newBufferWithBytes:before.data()
        length:before.size() * sizeof(nm_float4) options:MTLResourceStorageModeShared];
    id<MTLBuffer> afterBuffer = [run.device newBufferWithBytes:after.data()
        length:after.size() * sizeof(nm_float4) options:MTLResourceStorageModeShared];
    id<MTLBuffer> unknownBuffer = [run.device newBufferWithBytes:unknowns.data()
        length:unknowns.size() * sizeof(NMVascularUnknownGPU) options:MTLResourceStorageModeShared];
    id<MTLBuffer> connectionBuffer = [run.device newBufferWithBytes:connections.data()
        length:connections.size() * sizeof(NMVascularConnectionGPU) options:MTLResourceStorageModeShared];
    id<MTLBuffer> statusBuffer = [run.device newBufferWithLength:sizeof(NMMatterStatusGPU)
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> compartmentBuffer = [run.device newBufferWithBytes:probe.compartments.contents
        length:probe.compartments.length options:MTLResourceStorageModeShared];
    id<MTLBuffer> elastanceBuffer = [run.device newBufferWithBytes:probe.elastance.contents
        length:probe.elastance.length options:MTLResourceStorageModeShared];
    check(accepted && candidate && beforeBuffer && afterBuffer && unknownBuffer &&
          connectionBuffer && statusBuffer && compartmentBuffer && elastanceBuffer,
          "32-step-cap fixture buffer allocation failed");
    NMMatterStatusGPU status{};
    status.code = NM_STATUS_SUCCESS;
    std::memcpy(statusBuffer.contents, &status, sizeof(status));
    const auto parameters = run.respiration->parameters;
    const NMHumanRespirationDispatch dispatch{1u, 45u, 0u, 0u};
    id<MTLCommandBuffer> command = [run.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    check(command && encoder, "32-step-cap command encoding failed");
    [encoder setComputePipelineState:run.respiration->exchange];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
    [encoder setBytes:&dispatch length:sizeof(dispatch) atIndex:1];
    [encoder setBuffer:accepted offset:0 atIndex:2];
    [encoder setBuffer:candidate offset:0 atIndex:3];
    [encoder setBuffer:beforeBuffer offset:0 atIndex:4];
    [encoder setBuffer:afterBuffer offset:0 atIndex:5];
    [encoder setBuffer:unknownBuffer offset:0 atIndex:6];
    [encoder setBuffer:connectionBuffer offset:0 atIndex:7];
    [encoder setBuffer:statusBuffer offset:0 atIndex:8];
    [encoder setBuffer:compartmentBuffer offset:0 atIndex:9];
    [encoder setBuffer:elastanceBuffer offset:0 atIndex:10];
    [encoder dispatchThreads:MTLSizeMake(1u,1u,1u)
        threadsPerThreadgroup:MTLSizeMake(1u,1u,1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    check(command.status == MTLCommandBufferStatusCompleted,
          "32-step-cap Metal fixture command failed");
    const auto actual = *static_cast<const NMHumanRespirationState*>(candidate.contents);
    const auto afterStatus = *static_cast<const NMMatterStatusGPU*>(statusBuffer.contents);
    auto expected = initial;
    expected.status.w = 4u;
    check(afterStatus.code == NM_STATUS_MULTIPHYSICS_FAILURE &&
          std::memcmp(&actual, &expected, sizeof(actual)) == 0,
          "required>32 candidate did not fail closed without partial state publication");
    const auto stillAccepted =
        *static_cast<const NMHumanRespirationState*>(accepted.contents);
    check(std::memcmp(&initial, &stillAccepted, sizeof(initial)) == 0,
          "required>32 candidate mutated its accepted respiration input");
}
} // namespace

int main(int argc, const char* argv[]) { @autoreleasepool { try {
    need(argc == 3, "usage: numi-human-gas-transport-fixture NETWORK.json RESPIRATION.json");
    EnvironmentValue gasEnv("NUMI_HUMAN_GAS_TRANSPORT_SUBCYCLING");
    EnvironmentValue sensitivityEnv("NUMI_HUMAN_RESTING_TIMESTEP_SENSITIVITY");
    verifySingleSubstepEquivalence(argv[1], argv[2]);
    verifySubcyclingAndTransactions(argv[1], argv[2]);
    std::cout << "human_gas_transport_subcycling_fixture=passed n1=bitwise "
              << "n_gt_1=positive_conservative rejection_retry=exact "
              << "cap=fail_closed_32\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "human_gas_transport_subcycling_fixture=failed reason="
              << error.what() << '\n';
    return 1;
} } }

