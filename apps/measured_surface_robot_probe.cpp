#include "metalrobo/MeasuredSurfaceRobot.hpp"
#include "metalrobo/MetalMeasuredSurfaceMechanics.hpp"
#include "metalrobo/RunProgram.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

template <typename T>
std::vector<T> readBinary(const std::filesystem::path& path) {
    static_assert(std::is_trivially_copyable_v<T>);
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    const std::streamsize bytes = stream.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamsize>(sizeof(T)) != 0) {
        throw std::runtime_error("invalid binary size for " + path.string());
    }
    std::vector<T> values(static_cast<std::size_t>(bytes) / sizeof(T));
    stream.seekg(0);
    if (!values.empty() &&
        !stream.read(reinterpret_cast<char*>(values.data()), bytes)) {
        throw std::runtime_error("cannot read " + path.string());
    }
    return values;
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

metalrobo::MeasuredSurfaceRobotPack loadMeasuredDove(
    const std::filesystem::path& manifestPath
) {
    using namespace metalrobo;
    const std::filesystem::path directory = manifestPath.parent_path();
    MeasuredSurfaceRobotPack pack;
    pack.id = "deetjen-f03-surface-robot-v1";
    pack.datasetIdentifier =
        "deetjen-ob-2018-12-11-f03-complete-surface-v1";
    pack.manifestSHA256 =
        "ad42148aa9ee72d994d668ba16f8b6572cb8b192b77539fe66d97586ed9e1a13";
    pack.positionsSHA256 =
        "690b6dd2a24d593a512d799b7fe5f3f756ca7ae3ce1cd1cdc4bb12b2531567a6";
    pack.trianglesSHA256 =
        "9d832ff22ecedc15e47c454378146a1006ae7f6974512ce222994e2f12f43d61";
    pack.frameCount = 144u;
    pack.vertexCount = 2157u;
    pack.triangleCount = 3968u;
    pack.sampleRateHertz = 1000.0f;
    // The source sequence is nonperiodic. Reflection is an explicit robot
    // mechanics boundary condition, never a claim about the measurement.
    pack.sourcePeriodic = false;
    pack.phaseBoundary = MeasuredSurfacePhaseBoundary::reflect;
    pack.actions = makeMeasuredSurfaceFlightActions();
    pack.components = {
        {MeasuredSurfaceComponent::body, 0u, 1443u, 0u, 2736u},
        {MeasuredSurfaceComponent::leftWing, 1443u, 297u, 2736u, 512u},
        {MeasuredSurfaceComponent::rightWing, 1740u, 297u, 3248u, 512u},
        {MeasuredSurfaceComponent::tail, 2037u, 120u, 3760u, 208u},
    };
    pack.frameMajorPositions = readBinary<float>(directory / "positions.f32le");
    pack.triangleIndices = readBinary<std::uint16_t>(directory / "triangles.u16le");
    pack.frameTimesSeconds.resize(pack.frameCount);
    for (std::uint32_t frame = 0u; frame < pack.frameCount; ++frame) {
        pack.frameTimesSeconds[frame] =
            static_cast<float>(frame) / pack.sampleRateHertz;
    }
    return pack;
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
    bool inspectSurface = false
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
        if (argc != 2 && argc != 4) {
            std::cerr << "usage: metalrobo_measured_surface_robot_probe "
                         "MANIFEST [ENVIRONMENTS STEPS]\n";
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
