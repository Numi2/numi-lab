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

metalrobo::MetalWorldResult runFlight(
    const metalrobo::CompiledRun& run,
    bool enableSurface,
    std::vector<float> actions,
    metalrobo::MetalMeasuredSurfaceStats* surfaceStats
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
    MetalWorldResult result;
    const MetalWorldDiagnostics diagnostics =
        context.run(run.world(), batch, config, result);
    require(diagnostics.succeeded() && diagnostics.failedStepCount == 0u &&
        diagnostics.successfulStepCount == environments * steps,
        "MetalWorld measured-surface rollout failed: " + diagnostics.message);
    if (mechanics && surfaceStats) *surfaceStats = mechanics->stats();
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        using namespace metalrobo;
        if (argc != 2) {
            std::cerr << "usage: metalrobo_measured_surface_robot_probe MANIFEST\n";
            return 2;
        }
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

        constexpr std::uint32_t environments = 4u;
        constexpr std::uint32_t steps = 24u;
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
        const MetalWorldResult gravityOnly =
            runFlight(run, false, actions, nullptr);
        MetalMeasuredSurfaceStats firstStats;
        const MetalWorldResult first =
            runFlight(run, true, actions, &firstStats);
        MetalMeasuredSurfaceStats replayStats;
        const MetalWorldResult replay =
            runFlight(run, true, actions, &replayStats);
        require(first.finalQ == replay.finalQ && first.finalV == replay.finalV,
            "device measured-surface rollout was not bit-identical on replay");
        require(first.finalV != gravityOnly.finalV,
            "device surface mechanics produced no generalized-body response");
        float maximumVelocityDelta = 0.0f;
        for (std::size_t i = 0u; i < first.finalV.size(); ++i) {
            maximumVelocityDelta = std::max(maximumVelocityDelta,
                std::abs(first.finalV[i] - gravityOnly.finalV[i]));
        }
        require(maximumVelocityDelta > 1.0e-5f,
            "surface mechanics response was below the executable threshold");
        require(firstStats.encodedPrepareCount == steps * 4u &&
                firstStats.encodedCommitCount == steps * 4u &&
                firstStats.environmentCapacity == environments &&
                firstStats.threadgroupWidth == 256u &&
                firstStats.immutableBytes > 0u &&
                firstStats.persistentBytes > 0u,
            "measured-surface device runtime did not report complete execution");

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
                  << "  nonperiodic wrap rejected: true\n"
                  << "  nonfinite actuator rollback: true\n"
                  << "  immutable/persistent bytes: "
                  << firstStats.immutableBytes << "/"
                  << firstStats.persistentBytes << '\n';
    } catch (const std::exception& error) {
        std::cerr << "MeasuredSurface probe failed: " << error.what() << '\n';
        return 1;
    }
}
