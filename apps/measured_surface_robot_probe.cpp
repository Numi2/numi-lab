#include "metalrobo/MeasuredSurfaceRobot.hpp"
#include "metalrobo/MetalMeasuredSurfaceMechanics.hpp"
#include "metalrobo/RunProgram.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

metalrobo::MeasuredSurfaceRobotPack loadMeasuredDove(
    const std::filesystem::path& manifestPath
) {
    return metalrobo::loadDeetjenMeasuredDoveRobotPack(manifestPath);
}

metalrobo::CompiledRun compileFlightRun(
    metalrobo::MeasuredSurfaceRobotPack surface,
    std::uint32_t environments,
    std::uint32_t steps
) {
    using namespace metalrobo;
    RunManifest manifest;
    manifest.id = "measured_surface_gpu_probe";
    manifest.robot = makeMeasuredSurfaceRobotPack(
        std::move(surface), "deetjen_f03_robot");
    manifest.scene.id = "unbounded_air";
    manifest.sensors.id = "surface_flight_state";
    manifest.task = makeMeasuredSurfaceFlightTaskPack(
        manifest.robot, manifest.sensors.observation, manifest.reality.reset);
    manifest.reality.id = "nominal_air";
    manifest.teacher.id = "no_teacher";
    manifest.profile.id = "surface_gpu_probe_profile";
    manifest.profile.environmentCount = environments;
    manifest.profile.controlSteps = steps;
    manifest.profile.physicsSubsteps = 4u;
    manifest.profile.velocityIterations = 2u;
    manifest.profile.finalVelocityIterations = 1u;
    manifest.profile.controlTimestepSeconds = 1.0f / 60.0f;
    CompiledRun compiled;
    const RunCompileDiagnostics diagnostics = compileRun(manifest, compiled);
    require(diagnostics.succeeded(),
        "CompiledRun failed [" +
        std::string(runCompileStatusName(diagnostics.status)) + "] " +
        diagnostics.element + ": " + diagnostics.message);
    require(compiled.valid() && compiled.measuredSurfaceBinding() != nullptr,
        "CompiledRun lost the measured-surface mechanics binding");
    return compiled;
}

metalrobo::CompiledRun compileDropRecoveryRun(
    metalrobo::MeasuredSurfaceRobotPack surface,
    std::uint32_t environments,
    std::uint32_t steps
) {
    using namespace metalrobo;
    surface.normalizedActionBias = measuredSurfaceRecoveryTrimActions();
    RunManifest manifest;
    manifest.id = "measured_surface_fatal_drop_probe";
    manifest.robot = makeMeasuredSurfaceRobotPack(
        std::move(surface), "deetjen_f03_robot");
    manifest.scene = makeMeasuredSurfaceDropRecoveryScenePack(manifest.robot);
    manifest.sensors.id = "surface_flight_state";
    manifest.reality.id = "fatal_drop_calibration";
    manifest.teacher.id = "no_teacher";
    manifest.task = makeMeasuredSurfaceDropRecoveryTaskPack(
        manifest.robot, manifest.sensors.observation, manifest.reality.reset);
    manifest.profile.id = "surface_fatal_drop_probe_profile";
    manifest.profile.environmentCount = environments;
    manifest.profile.controlSteps = steps;
    manifest.profile.physicsSubsteps = 4u;
    manifest.profile.velocityIterations = 4u;
    manifest.profile.finalVelocityIterations = 2u;
    manifest.profile.controlTimestepSeconds = 1.0f / 60.0f;
    manifest.profile.capacities = manifest.task.capacities;
    CompiledRun compiled;
    const RunCompileDiagnostics diagnostics = compileRun(manifest, compiled);
    require(diagnostics.succeeded(),
        "drop CompiledRun failed [" +
        std::string(runCompileStatusName(diagnostics.status)) + "] " +
        diagnostics.element + ": " + diagnostics.message);
    require(compiled.valid() && compiled.measuredSurfaceBinding() != nullptr,
        "drop CompiledRun lost the measured-surface mechanics binding");
    return compiled;
}

struct FlightExecution {
    metalrobo::MetalWorldResult result;
    metalrobo::MetalWorldDiagnostics diagnostics;
    metalrobo::MetalMeasuredSurfaceStats surfaceStats;
    metalrobo::MetalMeasuredSurfaceInspection surfaceInspection;
};

FlightExecution runFlight(
    const metalrobo::CompiledRun& run,
    bool enableSurface,
    std::vector<float> actions,
    bool inspectSurface = false,
    float initialHeightMeters = 2.0f,
    float initialVerticalSpeedMetersPerSecond = 0.0f
) {
    using namespace metalrobo;
    const std::size_t environments = run.profile().environmentCount;
    const std::size_t steps = run.profile().controlSteps;
    const std::size_t nq = run.world().nq();
    const std::size_t nv = run.world().nv();
    std::vector<float> q(environments * nq);
    std::vector<float> v(environments * nv, 0.0f);
    std::vector<std::uint32_t> resetMasks(environments * steps, 0u);
    for (std::size_t environment = 0u; environment < environments; ++environment) {
        std::copy(run.model().defaultQ.begin(), run.model().defaultQ.end(),
            q.begin() + environment * nq);
        q[environment * nq + 2u] = initialHeightMeters;
        v[environment * nv + 2u] = initialVerticalSpeedMetersPerSecond;
    }
    require(actions.size() == steps * environments *
        run.task().layout().actionCount, "action stream has the wrong shape");
    const MetalWorldBatch batch{
        .environmentCount = environments,
        .controlStepCount = steps,
        .initialQ = q,
        .initialV = v,
        .actions = actions,
        .resetMasks = resetMasks,
        .initialSceneBodies = run.defaultSceneBodies(),
    };
    MetalWorldStepConfig config{
        .timestepSeconds = run.profile().controlTimestepSeconds,
        .physicsSubsteps = run.profile().physicsSubsteps,
        .solverMode = MetalWorldSolverMode::temporalCone,
        .actuationMode = MetalWorldActuationMode::implicitPositionDrive,
        .taskProgram = run.task(),
        .taskSeed = run.profile().seed,
        .velocityIterations = run.profile().velocityIterations,
        .finalVelocityIterations = run.profile().finalVelocityIterations,
        .ccdMode = MetalWorldCCDMode::disabled,
        .applyBodyDamping = false,
        .deterministic = true,
        .warmStart = false,
    };
    std::unique_ptr<MetalMeasuredSurfaceMechanics> mechanics;
    if (enableSurface) {
        mechanics = std::make_unique<MetalMeasuredSurfaceMechanics>(
            *run.measuredSurfaceBinding());
        config.deviceMechanicsProgram = mechanics->program();
    }
    MetalWorldContext context({.maximumInFlightSubmissions = 1u});
    FlightExecution execution;
    execution.diagnostics =
        context.run(run.world(), batch, config, execution.result);
    if (mechanics) {
        execution.surfaceStats = mechanics->stats();
        if (inspectSurface) {
            execution.surfaceInspection = mechanics->inspectAccepted();
        }
    }
    return execution;
}

struct FatalDropEvidence {
    float minimumHeight = std::numeric_limits<float>::infinity();
    float finalHeight = 0.0f;
    float minimumVerticalSpeed = std::numeric_limits<float>::infinity();
    float finalVerticalSpeed = 0.0f;
    std::uint32_t activeContactSteps = 0u;
};

FatalDropEvidence fatalDropEvidence(
    const metalrobo::CompiledRun& run,
    const FlightExecution& execution
) {
    FatalDropEvidence evidence;
    const std::size_t stateStride = run.world().nq() + run.world().nv();
    for (std::size_t step = 0u; step < run.profile().controlSteps; ++step) {
        const std::size_t base = step * stateStride;
        const float height = execution.result.observations[base + 2u];
        if (height > 1.0e-3f) {
            evidence.minimumHeight = std::min(evidence.minimumHeight, height);
        }
        evidence.minimumVerticalSpeed = std::min(
            evidence.minimumVerticalSpeed,
            execution.result.observations[base + run.world().nq() + 2u]);
    }
    evidence.finalHeight = execution.result.finalQ[2u];
    evidence.finalVerticalSpeed = execution.result.finalV[2u];
    evidence.activeContactSteps = static_cast<std::uint32_t>(std::count_if(
        execution.result.contactStatuses.begin(),
        execution.result.contactStatuses.end(),
        [](const MRMetalWorldContactStatusGPU& status) {
            return status.activeContacts > 0u;
        }));
    if (!std::isfinite(evidence.minimumHeight)) {
        evidence.minimumHeight = evidence.finalHeight;
    }
    return evidence;
}

void runFatalDropCalibration(
    const std::filesystem::path& manifestPath,
    const float heightMeters,
    const float verticalSpeedMetersPerSecond,
    const std::uint32_t steps
) {
    using namespace metalrobo;
    require(heightMeters > 0.0f && verticalSpeedMetersPerSecond <= 0.0f &&
            steps > 0u,
        "fatal-drop calibration requires positive height and steps and nonpositive vertical speed");
    const CompiledRun run = compileDropRecoveryRun(
        loadMeasuredDove(manifestPath), 1u, steps);
    const std::vector<float> actions(
        static_cast<std::size_t>(steps) * run.task().layout().actionCount,
        0.0f);
    const FlightExecution noFlight = runFlight(
        run, false, actions, false, heightMeters,
        verticalSpeedMetersPerSecond);
    const FlightExecution trimFlight = runFlight(
        run, true, actions, false, heightMeters,
        verticalSpeedMetersPerSecond);
    require(noFlight.diagnostics.succeeded() &&
            trimFlight.diagnostics.succeeded(),
        "fatal-drop calibration rollout failed");
    const FatalDropEvidence disabled = fatalDropEvidence(run, noFlight);
    const FatalDropEvidence trimmed = fatalDropEvidence(run, trimFlight);
    std::cout << "MeasuredSurface fatal-drop calibration\n"
              << "  release height / vertical speed: " << heightMeters
              << " m / " << verticalSpeedMetersPerSecond << " m/s\n"
              << "  flight disabled min/final height: "
              << disabled.minimumHeight << "/" << disabled.finalHeight
              << " m\n"
              << "  flight disabled min/final vertical speed: "
              << disabled.minimumVerticalSpeed << "/"
              << disabled.finalVerticalSpeed << " m/s\n"
              << "  flight disabled active-contact steps: "
              << disabled.activeContactSteps << '\n'
              << "  trim flight min/final height: "
              << trimmed.minimumHeight << "/" << trimmed.finalHeight
              << " m\n"
              << "  trim flight min/final vertical speed: "
              << trimmed.minimumVerticalSpeed << "/"
              << trimmed.finalVerticalSpeed << " m/s\n"
              << "  trim flight active-contact steps: "
              << trimmed.activeContactSteps << '\n';
    require(disabled.activeContactSteps > 0u || disabled.minimumHeight < 0.10f,
        "flight-disabled baseline did not reach the ground");
    require(trimmed.activeContactSteps == 0u && trimmed.finalHeight > 1.0f,
        "trimmed surface did not arrest the calibrated fatal drop");
    std::cout << "  verdict: passed\n";
}

float deterministicSignedUnit(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    value ^= value >> 31u;
    const double unit = static_cast<double>(value >> 11u) *
        (1.0 / 9007199254740992.0);
    return static_cast<float>(2.0 * unit - 1.0);
}

struct AuthorityLoad {
    std::array<float, 3u> force{};
    std::array<float, 3u> torque{};
};

AuthorityLoad meanAcceptedLoad(const MRMeasuredSurfaceEvidenceGPU& evidence) {
    require(evidence.worldForceImpulseAndTime.w > 0.0f,
        "authority evidence did not accumulate integration time");
    const float inverseTime = 1.0f / evidence.worldForceImpulseAndTime.w;
    return {
        .force = {
            evidence.worldForceImpulseAndTime.x * inverseTime,
            evidence.worldForceImpulseAndTime.y * inverseTime,
            evidence.worldForceImpulseAndTime.z * inverseTime,
        },
        .torque = {
            evidence.worldTorqueImpulse.x * inverseTime,
            evidence.worldTorqueImpulse.y * inverseTime,
            evidence.worldTorqueImpulse.z * inverseTime,
        },
    };
}

void runAuthoritySweep(
    const std::filesystem::path& manifestPath,
    std::uint32_t candidateCount,
    std::uint32_t steps
) {
    using namespace metalrobo;
    require(candidateCount >= 49u && candidateCount <= 4096u,
        "authority candidate count must be in [49, 4096]");
    require(steps >= 18u && steps <= 120u,
        "authority sweep must span at least one reflected wingbeat");
    MeasuredSurfaceRobotPack pack = loadMeasuredDove(manifestPath);
    const CompiledMeasuredSurfaceRobot robot =
        compileMeasuredSurfaceRobot(pack);
    const CompiledRun run = compileFlightRun(pack, candidateCount, steps);
    const std::uint32_t actionCount = run.task().layout().actionCount;
    require(actionCount == kMeasuredSurfaceActionCount,
        "authority sweep requires the complete 24-action contract");
    std::vector<float> actions(
        static_cast<std::size_t>(steps) * candidateCount * actionCount, 0.0f);
    for (std::uint32_t step = 0u; step < steps; ++step) {
        const float wingbeatPhase = 2.0f * std::numbers::pi_v<float> *
            static_cast<float>(step) / 18.0f;
        for (std::uint32_t candidate = 1u; candidate < candidateCount;
             ++candidate) {
            const std::size_t base =
                (static_cast<std::size_t>(step) * candidateCount + candidate) *
                actionCount;
            if (candidate <= 48u) {
                const std::uint32_t action = (candidate - 1u) / 2u;
                actions[base + action] = (candidate & 1u) != 0u ? 1.0f : -1.0f;
                continue;
            }
            if (candidate < 113u) {
                // Full-cadence/full-amplitude recovery envelope plus one
                // bilateral wing mode. The four sign/symmetry combinations
                // expose collective and differential authority explicitly.
                actions[base] = 1.0f;
                actions[base + 2u] = 1.0f;
                const std::uint32_t combination = candidate - 49u;
                const std::uint32_t wingLane = combination / 8u;
                const std::uint32_t pattern = combination % 4u;
                const float left = pattern < 2u ? 1.0f : -1.0f;
                const float right = (pattern & 1u) == 0u ? left : -left;
                actions[base + 4u + wingLane] = left;
                actions[base + 12u + wingLane] = right;
                continue;
            }
            for (std::uint32_t action = 0u; action < actionCount; ++action) {
                const std::uint64_t key =
                    static_cast<std::uint64_t>(candidate) * 1315423911ull +
                    static_cast<std::uint64_t>(action) * 2654435761ull;
                const float center = 0.78f * deterministicSignedUnit(key);
                const float amplitude = 0.22f * deterministicSignedUnit(
                    key ^ 0xd1b54a32d192ed03ull);
                const float phase = std::numbers::pi_v<float> *
                    deterministicSignedUnit(key ^ 0x94d049bb133111ebull);
                actions[base + action] = std::clamp(
                    center + amplitude * std::sin(wingbeatPhase + phase),
                    -1.0f, 1.0f);
            }
        }
    }
    const FlightExecution first =
        runFlight(run, true, actions, true, 20.0f);
    const FlightExecution replay =
        runFlight(run, true, actions, true, 20.0f);
    for (const FlightExecution* execution : {&first, &replay}) {
        require(execution->diagnostics.succeeded() &&
                execution->diagnostics.failedStepCount == 0u &&
                execution->surfaceInspection.acceptedEvidence.size() ==
                    candidateCount,
            "authority sweep Metal execution failed: " +
                execution->diagnostics.message);
    }
    require(first.result.finalQ == replay.result.finalQ &&
            first.result.finalV == replay.result.finalV &&
            std::memcmp(first.surfaceInspection.acceptedEvidence.data(),
                replay.surfaceInspection.acceptedEvidence.data(),
                candidateCount * sizeof(MRMeasuredSurfaceEvidenceGPU)) == 0,
        "authority sweep was not bit-identical on replay");
    std::vector<AuthorityLoad> loads(candidateCount);
    for (std::uint32_t candidate = 0u; candidate < candidateCount; ++candidate) {
        const MRMeasuredSurfaceEvidenceGPU& evidence =
            first.surfaceInspection.acceptedEvidence[candidate];
        require(evidence.deformationActuationStatus.z == 0.0f,
            "authority candidate published invalid surface evidence");
        loads[candidate] = meanAcceptedLoad(evidence);
    }
    const AuthorityLoad neutral = loads.front();
    const float weightNewtons = pack.bodyMassKilograms * 9.81f;
    constexpr float liftReserve = 1.05f;
    constexpr float requiredAngularAcceleration = 12.0f;
    std::uint32_t liftCandidate = 0u;
    std::uint32_t trimCandidate = 0u;
    float trimCost = std::numeric_limits<float>::infinity();
    std::array<std::uint32_t, 3u> positiveCandidates{};
    std::array<std::uint32_t, 3u> negativeCandidates{};
    for (std::uint32_t candidate = 1u; candidate < candidateCount; ++candidate) {
        if (loads[candidate].force[2u] > loads[liftCandidate].force[2u]) {
            liftCandidate = candidate;
        }
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            const float delta = loads[candidate].torque[axis] - neutral.torque[axis];
            const float positive = loads[positiveCandidates[axis]].torque[axis] -
                neutral.torque[axis];
            const float negative = loads[negativeCandidates[axis]].torque[axis] -
                neutral.torque[axis];
            if (delta > positive) positiveCandidates[axis] = candidate;
            if (delta < negative) negativeCandidates[axis] = candidate;
        }
        const float verticalError =
            (loads[candidate].force[2u] - weightNewtons) / weightNewtons;
        float candidateTrimCost = verticalError * verticalError +
            0.25f * (loads[candidate].force[0u] * loads[candidate].force[0u] +
                     loads[candidate].force[1u] * loads[candidate].force[1u]) /
                std::pow(weightNewtons, 2.0f);
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            const float angularAcceleration = loads[candidate].torque[axis] /
                pack.principalInertiaKilogramMetersSquared[axis];
            candidateTrimCost +=
                std::pow(angularAcceleration / requiredAngularAcceleration, 2.0f);
        }
        if (candidateTrimCost < trimCost) {
            trimCost = candidateTrimCost;
            trimCandidate = candidate;
        }
    }
    const bool liftPass = loads[liftCandidate].force[2u] >=
        liftReserve * weightNewtons;
    std::array<bool, 3u> positivePass{};
    std::array<bool, 3u> negativePass{};
    const std::array<const char*, 3u> axisNames{"roll", "pitch", "yaw"};
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        const float positiveTorque =
            loads[positiveCandidates[axis]].torque[axis] - neutral.torque[axis];
        const float negativeTorque =
            neutral.torque[axis] - loads[negativeCandidates[axis]].torque[axis];
        const float threshold = pack.principalInertiaKilogramMetersSquared[axis] *
            requiredAngularAcceleration;
        positivePass[axis] = positiveTorque >= threshold;
        negativePass[axis] = negativeTorque >= threshold;
    }
    const bool authorityPass = liftPass &&
        std::ranges::all_of(positivePass, [](bool value) { return value; }) &&
        std::ranges::all_of(negativePass, [](bool value) { return value; });
    const double environmentStepsPerSecond =
        first.diagnostics.gpuElapsedMilliseconds > 0.0
        ? static_cast<double>(candidateCount) * steps * 1000.0 /
            first.diagnostics.gpuElapsedMilliseconds
        : 0.0;
    std::cout << "MeasuredSurface control-authority sweep "
              << (authorityPass ? "PASSED" : "FAILED") << '\n'
              << "  run fingerprint: " << run.fingerprint() << '\n'
              << "  surface fingerprint: " << robot.fingerprint << '\n'
              << "  candidates x steps x substeps: " << candidateCount
              << " x " << steps << " x 4\n"
              << "  actuator contract: 24/24 lanes exercised; neutral + signed basis + bilateral recovery modes + deterministic harmonic candidates\n"
              << "  mass / weight: " << pack.bodyMassKilograms << " kg / "
              << weightNewtons << " N\n"
              << "  lift gate: mean Fz >= " << liftReserve << " x weight\n"
              << "  best lift candidate / mean Fz / weight ratio: "
              << liftCandidate << " / " << loads[liftCandidate].force[2u]
              << " N / " << loads[liftCandidate].force[2u] / weightNewtons
              << " [" << (liftPass ? "pass" : "fail") << "]\n"
              << "  best lift mean torque xyz N m: "
              << loads[liftCandidate].torque[0u] << " "
              << loads[liftCandidate].torque[1u] << " "
              << loads[liftCandidate].torque[2u] << '\n'
              << "  neutral mean force xyz N: " << neutral.force[0u] << " "
              << neutral.force[1u] << " " << neutral.force[2u] << '\n'
              << "  best trim candidate / cost: " << trimCandidate << " / "
              << trimCost << '\n'
              << "  best trim mean force xyz N: "
              << loads[trimCandidate].force[0u] << " "
              << loads[trimCandidate].force[1u] << " "
              << loads[trimCandidate].force[2u] << '\n'
              << "  best trim mean torque xyz N m: "
              << loads[trimCandidate].torque[0u] << " "
              << loads[trimCandidate].torque[1u] << " "
              << loads[trimCandidate].torque[2u] << '\n'
              << "  attitude gate: bidirectional delta torque / inertia >= "
              << requiredAngularAcceleration << " rad/s^2\n";
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        const float positiveTorque =
            loads[positiveCandidates[axis]].torque[axis] - neutral.torque[axis];
        const float negativeTorque =
            neutral.torque[axis] - loads[negativeCandidates[axis]].torque[axis];
        const float inertia = pack.principalInertiaKilogramMetersSquared[axis];
        std::cout << "  " << axisNames[axis]
                  << " +/- candidate: " << positiveCandidates[axis] << "/"
                  << negativeCandidates[axis]
                  << ", delta torque: +" << positiveTorque << "/-"
                  << negativeTorque << " N m, angular accel: +"
                  << positiveTorque / inertia << "/-" << negativeTorque / inertia
                  << " rad/s^2 ["
                  << ((positivePass[axis] && negativePass[axis]) ? "pass" : "fail")
                  << "]\n";
    }
    std::cout << "  deterministic signed-load replay: true\n"
              << "  failed environment steps: "
              << first.diagnostics.failedStepCount << '\n'
              << "  immutable/persistent bytes: "
              << first.surfaceStats.immutableBytes << "/"
              << first.surfaceStats.persistentBytes << '\n'
              << "  checkpoint/threadgroup bytes: "
              << first.surfaceStats.controlStepCheckpointBytes << "/"
              << first.surfaceStats.threadgroupBytes << '\n'
              << "  GPU ms / environment steps per second: "
              << first.diagnostics.gpuElapsedMilliseconds << " / "
              << environmentStepsPerSecond << '\n';
    require(authorityPass,
        "24-actuator contract did not satisfy every physical authority gate");
}

float deterministicNormal(std::uint64_t key) {
    const float u1 = std::max(1.0e-7f,
        0.5f * (deterministicSignedUnit(key) + 1.0f));
    const float u2 = 0.5f * (deterministicSignedUnit(
        key ^ 0x9e3779b97f4a7c15ull) + 1.0f);
    return std::sqrt(-2.0f * std::log(u1)) *
        std::cos(2.0f * std::numbers::pi_v<float> * u2);
}

float trimObjective(
    const AuthorityLoad& load,
    const metalrobo::MeasuredSurfaceRobotPack& pack
) {
    const float weight = pack.bodyMassKilograms * 9.81f;
    const float liftShortfall = std::max(0.0f, 1.05f * weight - load.force[2u]) /
        weight;
    const float liftError = (load.force[2u] - weight) / weight;
    float cost = 20.0f * liftShortfall * liftShortfall +
        liftError * liftError +
        0.10f * (load.force[0u] * load.force[0u] +
                 load.force[1u] * load.force[1u]) / (weight * weight);
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
        const float acceleration = load.torque[axis] /
            pack.principalInertiaKilogramMetersSquared[axis];
        cost += std::pow(acceleration / 12.0f, 2.0f);
    }
    return cost;
}

void runTrimSweep(
    const std::filesystem::path& manifestPath,
    std::uint32_t candidateCount,
    std::uint32_t generations,
    std::uint32_t steps
) {
    using namespace metalrobo;
    require(candidateCount >= 128u && candidateCount <= 2048u &&
            generations >= 2u && generations <= 32u &&
            steps >= 18u && steps <= 120u,
        "trim sweep dimensions exceed the bounded optimizer contract");
    MeasuredSurfaceRobotPack pack = loadMeasuredDove(manifestPath);
    const CompiledMeasuredSurfaceRobot robot = compileMeasuredSurfaceRobot(pack);
    const CompiledRun run = compileFlightRun(pack, candidateCount, steps);
    const std::uint32_t actionCount = run.task().layout().actionCount;
    std::array<float, kMeasuredSurfaceActionCount> mean{};
    std::array<float, kMeasuredSurfaceActionCount> deviation{};
    deviation.fill(0.55f);
    mean[0u] = 0.90f;
    mean[2u] = 0.90f;
    deviation[0u] = 0.20f;
    deviation[2u] = 0.20f;
    std::array<float, kMeasuredSurfaceActionCount> bestAction{};
    AuthorityLoad bestLoad{};
    float bestCost = std::numeric_limits<float>::infinity();
    const std::uint32_t eliteCount = std::max(16u, candidateCount / 16u);
    for (std::uint32_t generation = 0u; generation < generations; ++generation) {
        std::vector<std::array<float, kMeasuredSurfaceActionCount>> candidates(
            candidateCount);
        candidates[0u] = mean;
        for (std::uint32_t candidate = 1u; candidate < candidateCount;
             ++candidate) {
            for (std::uint32_t action = 0u; action < actionCount; ++action) {
                const std::uint64_t key =
                    static_cast<std::uint64_t>(generation + 1u) *
                        0xd1b54a32d192ed03ull ^
                    static_cast<std::uint64_t>(candidate) * 1315423911ull ^
                    static_cast<std::uint64_t>(action) * 2654435761ull;
                candidates[candidate][action] = std::clamp(
                    mean[action] + deviation[action] * deterministicNormal(key),
                    -1.0f, 1.0f);
            }
        }
        std::vector<float> actions(
            static_cast<std::size_t>(steps) * candidateCount * actionCount);
        for (std::uint32_t step = 0u; step < steps; ++step) {
            for (std::uint32_t candidate = 0u; candidate < candidateCount;
                 ++candidate) {
                std::copy(candidates[candidate].begin(),
                    candidates[candidate].end(),
                    actions.begin() +
                        (static_cast<std::size_t>(step) * candidateCount +
                         candidate) * actionCount);
            }
        }
        const FlightExecution execution =
            runFlight(run, true, std::move(actions), true, 20.0f);
        require(execution.diagnostics.succeeded() &&
                execution.diagnostics.failedStepCount == 0u,
            "trim optimizer Metal execution failed: " +
                execution.diagnostics.message);
        std::vector<std::pair<float, std::uint32_t>> ranked;
        ranked.reserve(candidateCount);
        for (std::uint32_t candidate = 0u; candidate < candidateCount;
             ++candidate) {
            const AuthorityLoad load = meanAcceptedLoad(
                execution.surfaceInspection.acceptedEvidence[candidate]);
            const float cost = trimObjective(load, pack);
            ranked.emplace_back(cost, candidate);
            if (cost < bestCost) {
                bestCost = cost;
                bestAction = candidates[candidate];
                bestLoad = load;
            }
        }
        std::partial_sort(ranked.begin(), ranked.begin() + eliteCount,
            ranked.end());
        for (std::uint32_t action = 0u; action < actionCount; ++action) {
            float nextMean = 0.0f;
            for (std::uint32_t elite = 0u; elite < eliteCount; ++elite) {
                nextMean += candidates[ranked[elite].second][action];
            }
            nextMean /= static_cast<float>(eliteCount);
            float variance = 0.0f;
            for (std::uint32_t elite = 0u; elite < eliteCount; ++elite) {
                const float delta =
                    candidates[ranked[elite].second][action] - nextMean;
                variance += delta * delta;
            }
            variance /= static_cast<float>(eliteCount);
            mean[action] = 0.25f * mean[action] + 0.75f * nextMean;
            deviation[action] = std::max(0.025f,
                0.25f * deviation[action] + 0.75f * std::sqrt(variance));
        }
        std::cout << "  generation " << generation + 1u << "/"
                  << generations << " best cost " << bestCost
                  << " mean Fz " << bestLoad.force[2u] << " N\n";
    }
    const float weight = pack.bodyMassKilograms * 9.81f;
    std::cout << "MeasuredSurface trim sweep complete\n"
              << "  run fingerprint: " << run.fingerprint() << '\n'
              << "  surface fingerprint: " << robot.fingerprint << '\n'
              << "  candidates x generations x steps: " << candidateCount
              << " x " << generations << " x " << steps << '\n'
              << "  best cost: " << bestCost << '\n'
              << "  mean force xyz N: " << bestLoad.force[0u] << " "
              << bestLoad.force[1u] << " " << bestLoad.force[2u] << '\n'
              << "  lift / weight: " << bestLoad.force[2u] / weight << '\n'
              << "  mean torque xyz N m: " << bestLoad.torque[0u] << " "
              << bestLoad.torque[1u] << " " << bestLoad.torque[2u] << '\n'
              << "  normalized actions:";
    for (const float value : bestAction) std::cout << " " << value;
    std::cout << '\n';
}

bool proveStatefulSingleFlight(
    const metalrobo::CompiledRun& run,
    const std::vector<float>& actions
) {
    using namespace metalrobo;
    const std::size_t environments = run.profile().environmentCount;
    const std::size_t steps = run.profile().controlSteps;
    const std::size_t nq = run.world().nq();
    const std::size_t nv = run.world().nv();
    std::vector<float> q(environments * nq);
    std::vector<float> v(environments * nv, 0.0f);
    std::vector<std::uint32_t> resetMasks(environments * steps, 0u);
    for (std::size_t environment = 0u; environment < environments; ++environment) {
        std::copy(run.model().defaultQ.begin(), run.model().defaultQ.end(),
            q.begin() + environment * nq);
    }
    const MetalWorldBatch batch{
        .environmentCount = environments,
        .controlStepCount = steps,
        .initialQ = q,
        .initialV = v,
        .actions = actions,
        .resetMasks = resetMasks,
        .initialSceneBodies = run.defaultSceneBodies(),
    };
    MetalMeasuredSurfaceMechanics mechanics(*run.measuredSurfaceBinding());
    const MetalWorldStepConfig config{
        .timestepSeconds = run.profile().controlTimestepSeconds,
        .physicsSubsteps = run.profile().physicsSubsteps,
        .solverMode = MetalWorldSolverMode::temporalCone,
        .actuationMode = MetalWorldActuationMode::implicitPositionDrive,
        .taskProgram = run.task(),
        .deviceMechanicsProgram = mechanics.program(),
        .taskSeed = run.profile().seed,
        .velocityIterations = run.profile().velocityIterations,
        .finalVelocityIterations = run.profile().finalVelocityIterations,
        .ccdMode = MetalWorldCCDMode::disabled,
        .applyBodyDamping = false,
        .deterministic = true,
        .warmStart = false,
    };
    MetalWorldContext context({.maximumInFlightSubmissions = 2u});
    MetalWorldSubmission first;
    const MetalWorldDiagnostics submitted =
        context.submit(run.world(), batch, config, first);
    if (!submitted.succeeded() || !first.valid()) return false;
    MetalWorldSubmission conflicting;
    const MetalWorldDiagnostics rejected =
        context.submit(run.world(), batch, config, conflicting);
    MetalWorldResult result;
    const MetalWorldDiagnostics completed = first.wait(result);
    return rejected.status == MetalWorldHostStatus::contextBusy &&
        !conflicting.valid() && completed.succeeded() &&
        completed.failedStepCount == 0u;
}

} // namespace

int main(int argc, char** argv) {
    try {
        using namespace metalrobo;
        if (argc == 6 && std::string_view(argv[1]) == "--fatal-drop") {
            runFatalDropCalibration(
                argv[2], std::stof(argv[3]), std::stof(argv[4]),
                static_cast<std::uint32_t>(std::stoul(argv[5])));
            return 0;
        }
        if (argc == 6 && std::string_view(argv[1]) == "--trim") {
            runTrimSweep(argv[2],
                static_cast<std::uint32_t>(std::stoul(argv[3])),
                static_cast<std::uint32_t>(std::stoul(argv[4])),
                static_cast<std::uint32_t>(std::stoul(argv[5])));
            return 0;
        }
        if (argc == 5 && std::string_view(argv[1]) == "--authority") {
            runAuthoritySweep(argv[2],
                static_cast<std::uint32_t>(std::stoul(argv[3])),
                static_cast<std::uint32_t>(std::stoul(argv[4])));
            return 0;
        }
        if (argc != 2 && argc != 4) {
            std::cerr << "usage: metalrobo_measured_surface_robot_probe "
                         "MANIFEST [ENVIRONMENTS STEPS]\n"
                         "       metalrobo_measured_surface_robot_probe "
                         "--authority MANIFEST CANDIDATES STEPS\n";
            std::cerr << "       metalrobo_measured_surface_robot_probe "
                         "--fatal-drop MANIFEST HEIGHT DOWNWARD_SPEED STEPS\n";
            return 2;
        }
        const std::uint32_t environments = argc == 4
            ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 4u;
        const std::uint32_t steps = argc == 4
            ? static_cast<std::uint32_t>(std::stoul(argv[3])) : 24u;
        require(environments > 0u && environments <= 4096u &&
                steps >= 2u && steps <= 1024u,
            "benchmark dimensions exceed the bounded probe contract");
        MeasuredSurfaceRobotPack pack = loadMeasuredDove(argv[1]);
        const CompiledMeasuredSurfaceRobot robot =
            compileMeasuredSurfaceRobot(pack);
        require(robot.vertexComponents.size() == pack.vertexCount &&
            robot.triangleComponents.size() == pack.triangleCount &&
            robot.fingerprint != 0u,
            "compiled measured-surface tables are incomplete");
        MeasuredSurfaceRobotPack invalidBoundary = pack;
        invalidBoundary.phaseBoundary = MeasuredSurfacePhaseBoundary::wrap;
        bool rejectedBoundary = false;
        try {
            (void)compileMeasuredSurfaceRobot(invalidBoundary);
        } catch (const std::invalid_argument&) {
            rejectedBoundary = true;
        }
        require(rejectedBoundary,
            "nonperiodic measured source accepted periodic wrapping");
        MeasuredSurfaceRobotPack invalidTopology = pack;
        const std::size_t leftTriangle = static_cast<std::size_t>(
            invalidTopology.components[1u].triangleOffset) * 3u;
        invalidTopology.triangleIndices[leftTriangle] = 0u;
        bool rejectedTopology = false;
        try {
            (void)compileMeasuredSurfaceRobot(invalidTopology);
        } catch (const std::invalid_argument&) {
            rejectedTopology = true;
        }
        require(rejectedTopology,
            "cross-component measured-surface topology was accepted");
        MeasuredSurfaceRobotPack invalidBias = pack;
        invalidBias.normalizedActionBias[0u] =
            std::numeric_limits<float>::infinity();
        bool rejectedBias = false;
        try {
            (void)compileMeasuredSurfaceRobot(invalidBias);
        } catch (const std::invalid_argument&) {
            rejectedBias = true;
        }
        require(rejectedBias,
            "nonfinite measured-surface action bias was accepted");

        MeasuredSurfaceActuatorState cpuState;
        const std::array<float, kMeasuredSurfaceActionCount> zeroTargets{};
        stepMeasuredSurfaceActuators(robot, zeroTargets, 0.001f, cpuState);
        require(std::ranges::all_of(cpuState.position,
                    [](float value) { return value == 0.0f; }) &&
                std::ranges::all_of(cpuState.velocity,
                    [](float value) { return value == 0.0f; }),
            "zero-action actuator invariant failed");
        const auto checkpoint = cpuState;
        auto invalid = zeroTargets;
        invalid[0] = std::nanf("");
        bool rejected = false;
        try {
            stepMeasuredSurfaceActuators(robot, invalid, 0.001f, cpuState);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected && cpuState.position == checkpoint.position &&
            cpuState.velocity == checkpoint.velocity,
            "transactional CPU actuator rollback failed");

        const CompiledRun run = compileFlightRun(pack, environments, steps);
        const std::size_t actionCount = run.task().layout().actionCount;
        std::vector<float> actions(
            static_cast<std::size_t>(steps) * environments * actionCount, 0.0f);
        for (std::uint32_t step = 0u; step < steps; ++step) {
            for (std::uint32_t environment = 0u;
                 environment < environments; ++environment) {
                const std::size_t base =
                    (static_cast<std::size_t>(step) * environments + environment) *
                    actionCount;
                actions[base + 5u] = 0.55f;
                actions[base + 13u] = -0.55f;
                actions[base + 20u] = 0.20f;
            }
        }
        const FlightExecution gravityOnly =
            runFlight(run, false, actions);
        const FlightExecution first =
            runFlight(run, true, actions, true);
        const FlightExecution replay =
            runFlight(run, true, actions);
        const CompiledRun observationPrefixRun =
            compileFlightRun(pack, environments, steps - 1u);
        const std::size_t prefixActionElements =
            static_cast<std::size_t>(steps - 1u) * environments * actionCount;
        const FlightExecution observationPrefix = runFlight(
            observationPrefixRun, true,
            std::vector<float>(actions.begin(),
                actions.begin() + prefixActionElements), true);
        for (const FlightExecution* execution :
             {&gravityOnly, &first, &replay}) {
            require(execution->diagnostics.succeeded() &&
                    execution->diagnostics.failedStepCount == 0u &&
                    execution->diagnostics.successfulStepCount ==
                        environments * steps,
                "MetalWorld measured-surface rollout failed: " +
                    execution->diagnostics.message);
        }
        require(observationPrefix.diagnostics.succeeded() &&
                observationPrefix.diagnostics.failedStepCount == 0u &&
                observationPrefix.diagnostics.successfulStepCount ==
                    environments * (steps - 1u),
            "prefix rollout for mechanics observation evidence failed");
        require(first.result.finalQ == replay.result.finalQ &&
                first.result.finalV == replay.result.finalV,
            "device measured-surface rollout was not bit-identical on replay");
        require(first.result.finalV != gravityOnly.result.finalV,
            "device surface mechanics produced no generalized-body response");
        float maximumVelocityDelta = 0.0f;
        for (std::size_t i = 0u; i < first.result.finalV.size(); ++i) {
            maximumVelocityDelta = std::max(maximumVelocityDelta,
                std::abs(first.result.finalV[i] -
                    gravityOnly.result.finalV[i]));
        }
        require(maximumVelocityDelta > 1.0e-5f,
            "surface mechanics response was below the executable threshold");
        const MetalMeasuredSurfaceStats& firstStats = first.surfaceStats;
        require(firstStats.encodedPrepareCount ==
                    static_cast<std::uint64_t>(steps) * 4u &&
                firstStats.encodedCommitCount ==
                    static_cast<std::uint64_t>(steps) * 4u &&
                firstStats.environmentCapacity == environments &&
                firstStats.threadgroupWidth == 256u &&
                firstStats.threadgroupBytes > 0u &&
                firstStats.controlStepCheckpointBytes > 0u &&
                firstStats.immutableBytes > 0u &&
                firstStats.persistentBytes > 0u &&
                first.surfaceInspection.acceptedStates.size() == environments,
            "measured-surface device runtime did not report complete execution");
        for (const MRMeasuredSurfaceStateGPU& state :
             first.surfaceInspection.acceptedStates) {
            require(state.phaseRateImpulseStep.z > 0.0f &&
                    state.phaseRateImpulseStep.w ==
                        static_cast<float>(steps * 4u),
                "accepted surface state lost impulse or substep accounting");
        }
        const std::size_t actorSize = run.task().layout().actorObservationSize;
        require(actorSize >= 4u && first.result.actorObservations.size() ==
                static_cast<std::size_t>(steps) * environments * actorSize,
            "device mechanics telemetry is missing from actor observations");
        for (std::uint32_t environment = 0u;
             environment < environments; ++environment) {
            const std::size_t firstActor =
                static_cast<std::size_t>(environment) * actorSize +
                actorSize - 4u;
            for (std::size_t component = 0u; component < 4u; ++component) {
                require(first.result.actorObservations[
                            firstActor + component] == 0.0f,
                    "reset did not clear device mechanics observation state");
            }
        }
        for (std::uint32_t environment = 0u;
             environment < environments; ++environment) {
            const std::size_t actorBase =
                ((static_cast<std::size_t>(steps) - 1u) * environments +
                 environment) * actorSize + actorSize - 4u;
            const MRMeasuredSurfaceStateGPU& state =
                observationPrefix.surfaceInspection.acceptedStates[environment];
            const MRMeasuredSurfaceEvidenceGPU& evidence =
                observationPrefix.surfaceInspection.acceptedEvidence[environment];
            const std::array<float, 4u> expected{
                state.phaseRateImpulseStep.x /
                    static_cast<float>(pack.frameCount - 1u),
                state.phaseRateImpulseStep.y,
                0.25f * evidence.loadsAreaPhase.x,
                evidence.deformationActuationStatus.y /
                    std::sqrt(static_cast<float>(kMeasuredSurfaceActionCount)),
            };
            for (std::size_t component = 0u; component < expected.size();
                 ++component) {
                require(std::abs(first.result.actorObservations[
                            actorBase + component] - expected[component]) <
                            1.0e-5f,
                    "actor observation does not match accepted mechanics state: component " +
                    std::to_string(component) + " actual=" +
                    std::to_string(first.result.actorObservations[
                        actorBase + component]) + " expected=" +
                    std::to_string(expected[component]));
            }
        }

        bool provedLateSubstepRollback = false;
        std::uint32_t rollbackSubstep = MR_INVALID_INDEX;
        float rollbackCoefficient = 0.0f;
        for (const float coefficient :
             {1.0e3f, 1.0e4f, 1.0e5f, 1.0e6f, 1.0e8f,
              1.0e10f, 1.0e12f, 1.0e16f, 1.0e20f}) {
            MeasuredSurfaceRobotPack stressedPack = pack;
            stressedPack.normalDragCoefficient = coefficient;
            stressedPack.tangentialDragCoefficient = coefficient * 0.1f;
            const CompiledRun stressed =
                compileFlightRun(std::move(stressedPack), 1u, 1u);
            std::vector<float> stressedActions(actionCount, 0.0f);
            stressedActions[5u] = 1.0f;
            stressedActions[13u] = -1.0f;
            const FlightExecution failure =
                runFlight(stressed, true, std::move(stressedActions), true);
            if (failure.diagnostics.failedStepCount == 0u ||
                failure.result.statuses.empty() ||
                failure.result.statuses.front().failingSubstep == 0u ||
                failure.result.statuses.front().failingSubstep ==
                    MR_INVALID_INDEX) {
                continue;
            }
            const MRMeasuredSurfaceStateGPU& restored =
                failure.surfaceInspection.acceptedStates.front();
            bool zero = restored.phaseRateImpulseStep.x == 0.0f &&
                (restored.phaseRateImpulseStep.y == 0.0f ||
                 restored.phaseRateImpulseStep.y == 1.0f) &&
                restored.phaseRateImpulseStep.w == 0.0f;
            for (const mr_float4 lane : restored.position) {
                zero = zero && lane.x == 0.0f && lane.y == 0.0f &&
                    lane.z == 0.0f && lane.w == 0.0f;
            }
            for (const mr_float4 lane : restored.velocity) {
                zero = zero && lane.x == 0.0f && lane.y == 0.0f &&
                    lane.z == 0.0f && lane.w == 0.0f;
            }
            require(zero,
                "late physics failure did not restore the surface control-step checkpoint");
            rollbackSubstep =
                failure.result.statuses.front().failingSubstep;
            rollbackCoefficient = coefficient;
            provedLateSubstepRollback = true;
            break;
        }
        require(provedLateSubstepRollback,
            "probe could not produce a post-substep surface failure for rollback proof");
        require(proveStatefulSingleFlight(run, actions),
            "stateful surface runtime admitted overlapping command buffers");
        const double environmentStepsPerSecond =
            first.diagnostics.gpuElapsedMilliseconds > 0.0
            ? static_cast<double>(environments * steps) * 1000.0 /
                first.diagnostics.gpuElapsedMilliseconds
            : 0.0;

        std::cout << "MeasuredSurface device mechanics probe passed\n"
                  << "  run fingerprint: " << run.fingerprint() << '\n'
                  << "  surface fingerprint: " << robot.fingerprint << '\n'
                  << "  environments x steps x substeps: "
                  << environments << " x " << steps << " x 4\n"
                  << "  prepare/commit encodes: "
                  << firstStats.encodedPrepareCount << "/"
                  << firstStats.encodedCommitCount << '\n'
                  << "  max root-velocity delta vs gravity-only: "
                  << maximumVelocityDelta << " generalized-velocity units\n"
                  << "  deterministic device replay: true\n"
                  << "  overlapping stateful submission rejected: true\n"
                  << "  late-substep checkpoint rollback: substep "
                  << rollbackSubstep << " at C="
                  << rollbackCoefficient << '\n'
                  << "  nonperiodic wrap rejected: true\n"
                  << "  cross-component topology rejected: true\n"
                  << "  nonfinite trim bias rejected: true\n"
                  << "  nonfinite actuator rollback: true\n"
                  << "  accepted impulse/substep accounting: true\n"
                  << "  accepted mechanics actor telemetry: true\n"
                  << "  immutable/persistent bytes: "
                  << firstStats.immutableBytes << "/"
                  << firstStats.persistentBytes << '\n'
                  << "  checkpoint/threadgroup bytes: "
                  << firstStats.controlStepCheckpointBytes << "/"
                  << firstStats.threadgroupBytes << '\n'
                  << "  GPU ms / environment steps per second: "
                  << first.diagnostics.gpuElapsedMilliseconds << " / "
                  << environmentStepsPerSecond << '\n';
    } catch (const std::exception& error) {
        std::cerr << "MeasuredSurface probe failed: " << error.what() << '\n';
        return 1;
    }
}
