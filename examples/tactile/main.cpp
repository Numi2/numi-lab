// Consolidated tactile product flows for every authored embodiment.
#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/EmbodiedTactile.hpp"
#include "metalrobo/FrankaWorld.hpp"
#include "metalrobo/MetalTactile.hpp"
#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/TactileDebug.hpp"
#include "metalrobo/WorldPack.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

template <typename Result>
requires requires(const Result& value) {
    value.succeeded();
    value.message;
}
void require(const Result& result, const char* operation) {
    if (!result.succeeded()) {
        throw std::runtime_error(
            std::string{operation} + ": " + result.message
        );
    }
}

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

mr_float4 quaternionMultiply(
    const mr_float4 a,
    const mr_float4 b
) {
    return {
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    };
}

mr_float4 rotate(
    const mr_float4 quaternion,
    const mr_float4 value
) {
    const mr_float4 twiceCross{
        2.0f * (
            quaternion.y * value.z -
            quaternion.z * value.y
        ),
        2.0f * (
            quaternion.z * value.x -
            quaternion.x * value.z
        ),
        2.0f * (
            quaternion.x * value.y -
            quaternion.y * value.x
        ),
        0.0f,
    };
    return {
        value.x + quaternion.w * twiceCross.x +
            quaternion.y * twiceCross.z -
            quaternion.z * twiceCross.y,
        value.y + quaternion.w * twiceCross.y +
            quaternion.z * twiceCross.x -
            quaternion.x * twiceCross.z,
        value.z + quaternion.w * twiceCross.z +
            quaternion.x * twiceCross.y -
            quaternion.y * twiceCross.x,
        value.w,
    };
}

mr_float4 add(const mr_float4 a, const mr_float4 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z, 0.0f};
}

mr_float4 subtract(const mr_float4 a, const mr_float4 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z, 0.0f};
}

mr_float4 multiply(const mr_float4 value, const float scale) {
    return {
        value.x * scale,
        value.y * scale,
        value.z * scale,
        0.0f,
    };
}

float dot3(const mr_float4 a, const mr_float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float length3(const mr_float4 value) {
    return std::sqrt(dot3(value, value));
}

struct SensorWorldPose {
    mr_float4 position{};
    mr_float4 orientation{0.0f, 0.0f, 0.0f, 1.0f};
    mr_float4 outwardNormal{};
};

std::vector<metalrobo::ArticulatedBodyKinematics> kinematics(
    const metalrobo::EngineModel& model,
    const std::span<const double> q,
    const std::span<const double> v
) {
    std::vector<metalrobo::ArticulatedBodyKinematics> result(
        model.articulations[0u].bodyCount
    );
    const auto diagnostics =
        metalrobo::computeArticulatedBodyKinematics(
            model,
            0u,
            q,
            v,
            result
        );
    if (!diagnostics.succeeded()) {
        throw std::runtime_error(
            "tactile example articulated kinematics failed"
        );
    }
    return result;
}

SensorWorldPose sensorWorldPose(
    const MRTactileSensorGPU& sensor,
    const std::span<
        const metalrobo::ArticulatedBodyKinematics> bodies
) {
    const auto& body = bodies[sensor.topology.x];
    const mr_float4 bodyPosition{
        static_cast<float>(body.centerOfMassPosition[0]),
        static_cast<float>(body.centerOfMassPosition[1]),
        static_cast<float>(body.centerOfMassPosition[2]),
        0.0f,
    };
    const mr_float4 bodyOrientation{
        static_cast<float>(body.orientation[0]),
        static_cast<float>(body.orientation[1]),
        static_cast<float>(body.orientation[2]),
        static_cast<float>(body.orientation[3]),
    };
    SensorWorldPose result;
    result.position = add(
        bodyPosition,
        rotate(
            bodyOrientation,
            sensor.localPositionAndQueryEpsilon
        )
    );
    result.orientation = quaternionMultiply(
        bodyOrientation,
        sensor.localOrientation
    );
    result.outwardNormal = rotate(
        result.orientation,
        {0.0f, 0.0f, 1.0f, 0.0f}
    );
    return result;
}

std::vector<MRBodyStateGPU> globalBodyStates(
    const metalrobo::EngineModel& model,
    const std::span<const double> q,
    const std::span<const double> v,
    const std::span<const MRBodyStateGPU> scene
) {
    const auto articulated = kinematics(model, q, v);
    std::vector<MRBodyStateGPU> result(model.bodies.size());
    for (const auto& body : articulated) {
        MRBodyStateGPU state{};
        state.position = {
            static_cast<float>(body.centerOfMassPosition[0]),
            static_cast<float>(body.centerOfMassPosition[1]),
            static_cast<float>(body.centerOfMassPosition[2]),
            1.0f,
        };
        state.orientation = {
            static_cast<float>(body.orientation[0]),
            static_cast<float>(body.orientation[1]),
            static_cast<float>(body.orientation[2]),
            static_cast<float>(body.orientation[3]),
        };
        state.linearVelocityAndInverseMass = {
            static_cast<float>(body.linearVelocity[0]),
            static_cast<float>(body.linearVelocity[1]),
            static_cast<float>(body.linearVelocity[2]),
            model.bodies[body.bodyIndex].massAndInverseMass.y,
        };
        state.angularVelocity = {
            static_cast<float>(body.angularVelocity[0]),
            static_cast<float>(body.angularVelocity[1]),
            static_cast<float>(body.angularVelocity[2]),
            0.0f,
        };
        state.flagsAndIndices[0] =
            model.bodies[body.bodyIndex].motionType;
        state.flagsAndIndices[1] = 0u;
        state.flagsAndIndices[2] = body.bodyIndex;
        result[body.bodyIndex] = state;
    }
    require(
        scene.size() ==
            model.bodies.size() - articulated.size(),
        "scene state does not cover every non-articulated body"
    );
    std::copy(
        scene.begin(),
        scene.end(),
        result.begin() + articulated.size()
    );
    return result;
}

std::array<SensorWorldPose, 2u> fingertipPoses(
    const metalrobo::EngineModel& model,
    const metalrobo::CookedTactileSystem& tactile,
    const std::span<const double> q,
    const std::span<const double> v
) {
    const auto bodies = kinematics(model, q, v);
    return {
        sensorWorldPose(tactile.sensors[0u], bodies),
        sensorWorldPose(tactile.sensors[1u], bodies),
    };
}

std::vector<double> closedConfiguration(
    const metalrobo::EngineModel& model,
    const metalrobo::CookedTactileSystem& tactile,
    const float targetGap
) {
    std::vector<double> q(
        model.defaultQ.begin(),
        model.defaultQ.end()
    );
    const std::vector<double> v(model.world.nv, 0.0);
    const auto gapAt = [&](const double fingerPosition) {
        q[7u] = fingerPosition;
        q[8u] = fingerPosition;
        const auto poses = fingertipPoses(
            model,
            tactile,
            q,
            v
        );
        return length3(
            subtract(poses[1u].position, poses[0u].position)
        );
    };
    const double closedGap = gapAt(0.0);
    const double openGap = gapAt(0.04);
    require(
        closedGap < targetGap && targetGap < openGap,
        "requested tactile grasp gap is outside Franka finger travel: "
        "closed=" + std::to_string(closedGap) +
        " open=" + std::to_string(openGap)
    );
    double lower = 0.0;
    double upper = 0.04;
    for (std::uint32_t iteration = 0u;
         iteration < 48u;
         ++iteration) {
        const double middle = 0.5 * (lower + upper);
        if (gapAt(middle) < targetGap) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    q[7u] = 0.5 * (lower + upper);
    q[8u] = q[7u];
    return q;
}

std::vector<float> floatVector(
    const std::span<const double> values
) {
    std::vector<float> result(values.size());
    std::transform(
        values.begin(),
        values.end(),
        result.begin(),
        [](const double value) {
            return static_cast<float>(value);
        }
    );
    return result;
}

std::vector<double> doubleVector(
    const std::span<const float> values
) {
    std::vector<double> result(values.size());
    std::transform(
        values.begin(),
        values.end(),
        result.begin(),
        [](const float value) {
            return static_cast<double>(value);
        }
    );
    return result;
}

metalrobo::TactileObservationBatch tactileObservation(
    const metalrobo::WorldTemplate& world,
    const std::span<const MRBodyStateGPU> bodies,
    const metalrobo::TactileSolverContactBatch& contacts,
    const std::uint64_t frameIndex
) {
    metalrobo::TactileCpuFrame frame;
    frame.environmentCount = 1u;
    frame.bodies = bodies;
    frame.contacts = contacts.contacts;
    frame.contactCounts = contacts.counts;
    frame.contactCapacityPerEnvironment =
        contacts.capacityPerEnvironment;
    frame.observationTimestepSeconds = 0.02f;
    frame.contactImpulseTimestepSeconds =
        contacts.contacts.empty() ? 0.0f : 0.02f / 8.0f;
    frame.frameIndex = frameIndex;
    frame.timestampSeconds = frameIndex * 0.02;
    metalrobo::TactileObservationBatch result;
    require(
        metalrobo::observeTactileCpuReference(
            world.tactileSystem,
            world.engineModel,
            frame,
            result
        ),
        "tactile example CPU observation"
    );
    return result;
}

void verifyWorldPack(
    const metalrobo::WorldFamily& family,
    const std::optional<std::filesystem::path>& outputPath =
        std::nullopt
) {
    metalrobo::MRWorldPack pack;
    require(
        metalrobo::compileWorldPack(family, pack),
        "compile tactile world pack"
    );
    const std::filesystem::path path = outputPath.value_or(
        std::filesystem::temp_directory_path() /
        (
            "metalrobo-tactile-" +
            std::to_string(pack.contentHash) +
            ".mrworld"
        )
    );
    require(
        metalrobo::writeWorldPack(pack, path),
        "write tactile world pack"
    );
    metalrobo::MRWorldPack loaded;
    const auto read = metalrobo::readWorldPack(path, loaded);
    if (!outputPath.has_value()) {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
    require(read, "read tactile world pack");
    const auto& loadedSamples =
        loaded.family.worldTemplate.tactileSystem.samples;
    const auto& originalSamples =
        family.worldTemplate.tactileSystem.samples;
    require(
        loaded.family.worldTemplate.tactileSystem.fingerprint ==
            family.worldTemplate.tactileSystem.fingerprint &&
        loadedSamples.size() == originalSamples.size() &&
        (
            loadedSamples.empty() ||
            std::memcmp(
                loadedSamples.data(),
                originalSamples.data(),
                loadedSamples.size() *
                    sizeof(MRTactileSampleGPU)
            ) == 0
        ),
        "world-pack replay changed the tactile observation definition"
    );
}

std::vector<MRBodyStateGPU> authoredSceneStates(
    const metalrobo::EngineModel& model,
    const metalrobo::EpisodeTwin& episode,
    const metalrobo::CompiledWorld& compiled
) {
    std::vector<MRBodyStateGPU> states;
    states.reserve(compiled.sceneBodyIndices().size());
    for (const std::uint32_t bodyIndex :
         compiled.sceneBodyIndices()) {
        const auto owner = std::ranges::find_if(
            episode.assets,
            [&](const metalrobo::WorldAsset& asset) {
                return std::ranges::find(
                    asset.bodyIndices,
                    bodyIndex
                ) != asset.bodyIndices.end();
            }
        );
        require(
            owner != episode.assets.end(),
            "authored scene body is not owned by an episode asset"
        );
        MRBodyStateGPU state{};
        state.position = owner->initialPose.position;
        state.position.w = 1.0f;
        state.orientation = owner->initialPose.orientation;
        state.linearVelocityAndInverseMass.w =
            model.bodies[bodyIndex].massAndInverseMass.y;
        state.flagsAndIndices[0] =
            model.bodies[bodyIndex].motionType;
        state.flagsAndIndices[1] = MR_INVALID_INDEX;
        state.flagsAndIndices[2] = bodyIndex;
        states.push_back(state);
    }
    return states;
}

metalrobo::TactileSolverContactBatch packFinalContacts(
    const metalrobo::EngineModel& model,
    const metalrobo::MetalWorldResult& result
) {
    require(
        !result.contactStatuses.empty(),
        "tactile scenario produced no contact status"
    );
    const std::array activeContactCounts{
        result.contactStatuses.back().activeContacts,
    };
    metalrobo::TactileSolverContactFrame frame;
    frame.environmentCount = 1u;
    frame.contactCapacityPerEnvironment =
        result.layout.contactDispatch.constraintCapacity;
    frame.shapes = model.shapes;
    frame.constraints = result.contactEvidence.contacts;
    frame.metadata = result.contactEvidence.contactMetadata;
    frame.activeContactCounts = activeContactCounts;
    metalrobo::TactileSolverContactBatch contacts;
    require(
        metalrobo::packTactileSolverContacts(frame, contacts),
        "pack tactile scenario solver impulses"
    );
    return contacts;
}

int runCompoundScenario(
    const std::string_view scenario,
    const std::optional<std::filesystem::path>& debugDirectory,
    const std::optional<std::filesystem::path>& worldPackPath
) {
    metalrobo::EngineModel model;
    metalrobo::EpisodeTwin episode;
    float controlTimestep = 0.02f;
    std::uint32_t controlSteps = 8u;
    std::uint32_t physicsSubsteps = 8u;
    metalrobo::MetalWorldCCDMode ccdMode =
        metalrobo::MetalWorldCCDMode::speculative;
    if (scenario == "g1-balance") {
        model = metalrobo::makeUnitreeG1TactileEngineModel();
        episode = metalrobo::makeUnitreeG1TactileEpisodeTwin();
    } else if (scenario == "psm-needle") {
        model = metalrobo::makeDvrkPsmTactileEngineModel();
        episode = metalrobo::makeDvrkPsmTactileEpisodeTwin();
        controlTimestep = 0.005f;
        controlSteps = 24u;
        ccdMode = metalrobo::MetalWorldCCDMode::hybrid;
    } else {
        throw std::runtime_error("unknown tactile example scenario");
    }

    metalrobo::WorldTemplate worldTemplate;
    require(
        metalrobo::compileEpisodeTwin(
            episode,
            model,
            worldTemplate
        ),
        "compile authored tactile scenario"
    );
    require(
        worldTemplate.tactileSystem.sensors.size() == 2u &&
            worldTemplate.tactileSystem.samples.size() ==
                2u * 32u * 32u,
        "authored tactile scenario did not retain two 32x32 sensors"
    );
    metalrobo::WorldProgram program;
    program.id = episode.id + "_program";
    metalrobo::WorldFamily family;
    require(
        metalrobo::compileWorldFamily(
            worldTemplate,
            program,
            family
        ),
        "compile authored tactile scenario family"
    );
    verifyWorldPack(family, worldPackPath);

    metalrobo::CompiledWorld compiled;
    require(
        metalrobo::compileMetalWorld(model, 0u, compiled),
        "compile authored tactile physics"
    );
    std::vector<double> q(
        model.defaultQ.begin(),
        model.defaultQ.end()
    );
    std::vector<double> v(
        model.defaultV.begin(),
        model.defaultV.end()
    );
    std::vector<MRBodyStateGPU> scene =
        authoredSceneStates(model, episode, compiled);
    if (scenario == "g1-balance") {
        v[0u] = 0.12;
    } else {
        q[6u] = -0.045;
        q[7u] = 0.045;
        require(
            scene.size() == 1u,
            "PSM tactile scenario requires one authored needle body"
        );
    }

    std::vector<float> actions(
        static_cast<std::size_t>(controlSteps) *
            compiled.nv(),
        0.0f
    );
    for (std::uint32_t stepIndex = 0u;
         stepIndex < controlSteps;
         ++stepIndex) {
        for (std::uint32_t coordinate = 0u;
             coordinate < compiled.nv();
             ++coordinate) {
            if (scenario == "g1-balance") {
                actions[
                    static_cast<std::size_t>(stepIndex) *
                        compiled.nv() +
                    coordinate
                ] = coordinate < 6u
                    ? 0.0f
                    : static_cast<float>(q[coordinate + 1u]);
            } else {
                actions[
                    static_cast<std::size_t>(stepIndex) *
                        compiled.nv() +
                    coordinate
                ] = static_cast<float>(q[coordinate]);
            }
        }
        if (scenario == "psm-needle") {
            const std::size_t base =
                static_cast<std::size_t>(stepIndex) *
                compiled.nv();
            const float phase =
                static_cast<float>(stepIndex + 1u) /
                static_cast<float>(controlSteps);
            const float jaw =
                0.045f * std::max(0.0f, 1.0f - 2.0f * phase);
            const float lift =
                0.004f * std::max(0.0f, 2.0f * phase - 1.0f);
            actions[base + 2u] =
                static_cast<float>(q[2u]) + lift;
            actions[base + 6u] = -jaw;
            actions[base + 7u] = jaw;
        }
    }

    const std::vector<float> initialQ = floatVector(q);
    const std::vector<float> initialV = floatVector(v);
    metalrobo::MetalWorldBatch batch{
        .environmentCount = 1u,
        .controlStepCount = controlSteps,
        .initialQ = initialQ,
        .initialV = initialV,
        .efforts = actions,
        .initialSceneBodies = scene,
    };
    metalrobo::MetalWorldStepConfig physicsConfig;
    physicsConfig.timestepSeconds = controlTimestep;
    physicsConfig.physicsSubsteps = physicsSubsteps;
    physicsConfig.solverMode =
        metalrobo::MetalWorldSolverMode::temporalCone;
    physicsConfig.actuationMode =
        metalrobo::MetalWorldActuationMode::
            implicitPositionDrive;
    physicsConfig.velocityIterations = 16u;
    physicsConfig.finalVelocityIterations = 8u;
    physicsConfig.ccdMode = ccdMode;
    physicsConfig.deterministic = true;
    physicsConfig.captureContactEvidence = true;

    metalrobo::MetalWorldContext physics;
    metalrobo::MetalWorldResult result;
    const auto run = physics.run(
        compiled,
        batch,
        physicsConfig,
        result
    );
    require(run, "run authored tactile scenario");
    const auto finalBodies = globalBodyStates(
        model,
        doubleVector(result.finalQ),
        doubleVector(result.finalV),
        result.finalSceneBodies
    );
    const auto solverContacts =
        packFinalContacts(model, result);
    const auto observation = tactileObservation(
        worldTemplate,
        finalBodies,
        solverContacts,
        controlSteps
    );

    metalrobo::MetalTactileConfig tactileConfig;
    tactileConfig.contactCapacityPerEnvironment =
        solverContacts.capacityPerEnvironment;
    metalrobo::MetalTactileContext tactileGPU(tactileConfig);
    require(
        tactileGPU.compile(
            worldTemplate.tactileSystem,
            model,
            1u
        ),
        "compile scenario tactile Metal context"
    );
    metalrobo::MetalTactileHostFrame tactileFrame;
    tactileFrame.environmentCount = 1u;
    tactileFrame.bodies = finalBodies;
    tactileFrame.contacts = solverContacts.contacts;
    tactileFrame.contactCounts = solverContacts.counts;
    tactileFrame.observationTimestepSeconds = controlTimestep;
    tactileFrame.contactImpulseTimestepSeconds =
        controlTimestep /
        static_cast<float>(physicsSubsteps);
    tactileFrame.frameIndex = controlSteps;
    tactileFrame.timestampSeconds =
        controlSteps * controlTimestep;
    require(
        tactileGPU.observe(tactileFrame),
        "observe authored scenario tactile maps"
    );
    metalrobo::TactileObservationBatch metalObservation;
    require(
        tactileGPU.readback(1u, metalObservation),
        "read authored scenario tactile maps"
    );

    double maximumDepthError = 0.0;
    for (std::size_t index = 0u;
         index < observation.penetrationDepthMeters.size();
         ++index) {
        maximumDepthError = std::max(
            maximumDepthError,
            static_cast<double>(std::abs(
                observation.penetrationDepthMeters[index] -
                metalObservation.penetrationDepthMeters[index]
            ))
        );
    }
    require(
        maximumDepthError < 2.0e-6,
        "authored scenario Metal depth differs from CPU"
    );
    const std::size_t activeSensors = std::ranges::count_if(
        observation.summaries,
        [](const MRTactileSummaryGPU& summary) {
            return summary.netForceAndContactArea.w > 0.0f &&
                length3(summary.netForceAndContactArea) > 0.0f;
        }
    );
    require(
        activeSensors > 0u,
        "authored scenario produced no coherent depth-and-wrench sensor: "
        "solver_contacts=" +
            std::to_string(
                result.contactStatuses.back().activeContacts
            ) +
            " sensor0_area=" +
            std::to_string(
                observation.summaries[0u].
                    netForceAndContactArea.w
            ) +
            " sensor0_force=" +
            std::to_string(
                length3(
                    observation.summaries[0u].
                        netForceAndContactArea
                )
            ) +
            " sensor1_area=" +
            std::to_string(
                observation.summaries[1u].
                    netForceAndContactArea.w
            ) +
            " sensor1_force=" +
            std::to_string(
                length3(
                    observation.summaries[1u].
                        netForceAndContactArea
                )
            )
    );

    if (debugDirectory.has_value()) {
        for (std::uint32_t sensorIndex = 0u;
             sensorIndex <
                worldTemplate.tactileSystem.sensors.size();
             ++sensorIndex) {
            metalrobo::TactileDebugExportConfig debug;
            debug.directory = *debugDirectory;
            debug.prefix =
                std::string{scenario} + "_" +
                worldTemplate.tactileSystem.sensorIds[sensorIndex];
            debug.sensor = sensorIndex;
            require(
                metalrobo::exportTactileDebugFrame(
                    worldTemplate.tactileSystem,
                    observation,
                    debug
                ),
                "export authored tactile scenario"
            );
        }
    }

    metalrobo::MetalWorldResult replay;
    require(
        physics.run(
            compiled,
            batch,
            physicsConfig,
            replay
        ),
        "replay authored tactile scenario"
    );
    require(
        result.finalQ == replay.finalQ &&
            result.finalV == replay.finalV &&
            result.finalSceneBodies.size() ==
                replay.finalSceneBodies.size() &&
            std::equal(
                result.finalSceneBodies.begin(),
                result.finalSceneBodies.end(),
                replay.finalSceneBodies.begin(),
                [](const MRBodyStateGPU& left,
                   const MRBodyStateGPU& right) {
                    return std::memcmp(
                        &left,
                        &right,
                        sizeof(left)
                    ) == 0;
                }
            ),
        "authored tactile scenario replay diverged"
    );

    std::cout
        << "scenario=" << scenario
        << " device=\"" << run.deviceName << "\""
        << " tactile_sensors="
        << worldTemplate.tactileSystem.sensors.size()
        << " active_tactile_sensors=" << activeSensors
        << " active_solver_contacts="
        << result.contactStatuses.back().activeContacts
        << " packed_solver_contacts="
        << solverContacts.counts[0u]
        << " observation_fingerprint=0x"
        << std::hex
        << worldTemplate.tactileSystem.fingerprint
        << std::dec
        << " max_cpu_gpu_error_m=" << maximumDepthError
        << " deterministic_replay=yes"
        << " world_pack="
        << (
            worldPackPath.has_value()
            ? worldPackPath->string()
            : "validated-temporary"
        )
        << " debug_export="
        << (
            debugDirectory.has_value()
            ? debugDirectory->string()
            : "disabled"
        )
        << '\n';
    return 0;
}

} // namespace

int main(const int argc, const char* const* argv) {
    try {
        if (argc < 2) {
            throw std::runtime_error(
                "usage: metalrobo_tactile_example "
                "franka-grasp|g1-balance|psm-needle "
                "[--debug-dir PATH] [--write-world-pack PATH]"
            );
        }
        const std::string_view scenario{argv[1]};
        require(
            scenario == "franka-grasp" ||
                scenario == "g1-balance" ||
                scenario == "psm-needle",
            "unknown tactile example scenario"
        );
        std::optional<std::filesystem::path> debugDirectory;
        std::optional<std::filesystem::path> worldPackPath;
        require(
            (argc - 2) % 2 == 0,
            "tactile example options require a path value"
        );
        for (int argument = 2;
             argument < argc;
             argument += 2) {
            const std::string_view option{argv[argument]};
            if (option == "--debug-dir") {
                debugDirectory =
                    std::filesystem::path{argv[argument + 1]};
            } else if (option == "--write-world-pack") {
                worldPackPath =
                    std::filesystem::path{argv[argument + 1]};
            } else {
                throw std::runtime_error(
                    "unknown tactile example option"
                );
            }
        }
        if (scenario != "franka-grasp") {
            return runCompoundScenario(
                scenario,
                debugDirectory,
                worldPackPath
            );
        }
        const metalrobo::EngineModel model =
            metalrobo::makeFrankaTactileEngineModel();
        const metalrobo::EpisodeTwin episode =
            metalrobo::makeFrankaTactileEpisodeTwin();
        metalrobo::WorldTemplate worldTemplate;
        require(
            metalrobo::compileEpisodeTwin(
                episode,
                model,
                worldTemplate
            ),
            "compile Franka tactile episode"
        );
        require(
            worldTemplate.tactileSystem.sensors.size() == 2u &&
            worldTemplate.tactileSystem.samples.size() ==
                2u * 32u * 32u &&
            (worldTemplate.capabilities &
             MR_WORLD_CAP_TACTILE_DEPTH) != 0u,
            "normal world compilation did not retain both tactile pads"
        );
        metalrobo::WorldFamily family;
        require(
            metalrobo::compileWorldFamily(
                worldTemplate,
                metalrobo::makeFrankaPickPlaceWorldProgram(),
                family
            ),
            "compile Franka tactile family"
        );
        verifyWorldPack(family, worldPackPath);

        constexpr float desiredPenetration = 0.0002f;
        constexpr float objectHalfExtent = 0.025f;
        const float desiredGap =
            2.0f * (objectHalfExtent - desiredPenetration);
        std::vector<double> q = closedConfiguration(
            model,
            worldTemplate.tactileSystem,
            desiredGap
        );
        std::vector<double> v(model.world.nv, 0.0);
        const auto initialFingertips = fingertipPoses(
            model,
            worldTemplate.tactileSystem,
            q,
            v
        );
        require(
            dot3(
                initialFingertips[0u].outwardNormal,
                initialFingertips[1u].outwardNormal
            ) < -0.999f,
            "Franka fingertip tactile normals do not oppose"
        );
        const mr_float4 objectPosition = multiply(
            add(
                initialFingertips[0u].position,
                initialFingertips[1u].position
            ),
            0.5f
        );
        std::vector<MRBodyStateGPU> scene =
            metalrobo::makeFrankaPickPlaceSceneState();
        scene[0u].position = {
            objectPosition.x,
            objectPosition.y,
            objectPosition.z,
            1.0f,
        };
        scene[0u].orientation =
            initialFingertips[0u].orientation;
        scene[0u].linearVelocityAndInverseMass =
            {0.0f, 0.0f, 0.0f, model.bodies[11u].
                massAndInverseMass.y};
        const std::vector<MRBodyStateGPU> initialBodies =
            globalBodyStates(model, q, v, scene);
        metalrobo::TactileSolverContactBatch noContacts;
        const auto initialTactile = tactileObservation(
            worldTemplate,
            initialBodies,
            noContacts,
            0u
        );
        if (debugDirectory.has_value()) {
            std::cerr
                << "initial_tactile_debug left_area="
                << initialTactile.summaries[0u].
                    netForceAndContactArea.w
                << " right_area="
                << initialTactile.summaries[1u].
                    netForceAndContactArea.w
                << " left_max="
                << initialTactile.summaries[0u].
                    netTorqueAndMaximumDepth.w
                << " right_max="
                << initialTactile.summaries[1u].
                    netTorqueAndMaximumDepth.w
                << " gap="
                << length3(subtract(
                    initialFingertips[1u].position,
                    initialFingertips[0u].position
                ))
                << " left_normal="
                << initialFingertips[0u].outwardNormal.x << ','
                << initialFingertips[0u].outwardNormal.y << ','
                << initialFingertips[0u].outwardNormal.z
                << " right_normal="
                << initialFingertips[1u].outwardNormal.x << ','
                << initialFingertips[1u].outwardNormal.y << ','
                << initialFingertips[1u].outwardNormal.z
                << '\n';
        }

        metalrobo::CompiledWorld compiled;
        require(
            metalrobo::compileMetalWorld(
                model,
                0u,
                compiled
            ),
            "compile tactile contact world"
        );
        constexpr std::uint32_t controlSteps = 1u;
        std::vector<float> actions(
            controlSteps * compiled.nv(),
            0.0f
        );
        for (std::uint32_t step = 0u;
             step < controlSteps;
             ++step) {
            for (std::uint32_t coordinate = 0u;
                 coordinate < compiled.nv();
                 ++coordinate) {
                actions[step * compiled.nv() + coordinate] =
                    static_cast<float>(q[coordinate]);
            }
            actions[step * compiled.nv() + 7u] -= 0.0001f;
            actions[step * compiled.nv() + 8u] -= 0.0001f;
        }
        const std::vector<float> initialQ = floatVector(q);
        const std::vector<float> initialV = floatVector(v);
        metalrobo::MetalWorldBatch batch{
            .environmentCount = 1u,
            .controlStepCount = controlSteps,
            .initialQ = initialQ,
            .initialV = initialV,
            .efforts = actions,
            .initialSceneBodies = scene,
        };
        metalrobo::MetalWorldStepConfig physicsConfig;
        physicsConfig.timestepSeconds = 0.02f;
        physicsConfig.physicsSubsteps = 8u;
        physicsConfig.solverMode =
            metalrobo::MetalWorldSolverMode::temporalCone;
        physicsConfig.actuationMode =
            metalrobo::MetalWorldActuationMode::
                implicitPositionDrive;
        physicsConfig.velocityIterations = 16u;
        physicsConfig.finalVelocityIterations = 8u;
        physicsConfig.deterministic = true;
        physicsConfig.captureContactEvidence = true;
        metalrobo::MetalWorldContext physics;
        metalrobo::MetalWorldResult first;
        const auto firstRun = physics.run(
            compiled,
            batch,
            physicsConfig,
            first
        );
        require(firstRun, "run tactile grasp physics");
        require(
            !first.contactStatuses.empty() &&
            first.contactStatuses.back().activeContacts >= 2u,
            "actual contact solver did not retain bilateral finger contact"
        );

        const std::vector<double> finalQ =
            doubleVector(first.finalQ);
        const std::vector<double> finalV =
            doubleVector(first.finalV);
        const std::vector<MRBodyStateGPU> finalBodies =
            globalBodyStates(
                model,
                finalQ,
                finalV,
                first.finalSceneBodies
            );
        const auto finalFingertips = fingertipPoses(
            model,
            worldTemplate.tactileSystem,
            finalQ,
            finalV
        );
        if (debugDirectory.has_value()) {
            std::cerr
                << "pose_debug final_gap="
                << length3(subtract(
                    finalFingertips[1u].position,
                    finalFingertips[0u].position
                ))
                << " object_distance_left="
                << length3(subtract(
                    first.finalSceneBodies[0u].position,
                    finalFingertips[0u].position
                ))
                << " object_distance_right="
                << length3(subtract(
                    first.finalSceneBodies[0u].position,
                    finalFingertips[1u].position
                ))
                << " object_speed="
                << length3(
                    first.finalSceneBodies[0u].
                        linearVelocityAndInverseMass
                )
                << '\n';
        }
        const std::array activeContactCounts{
            first.contactStatuses.back().activeContacts,
        };
        metalrobo::TactileSolverContactFrame contactFrame;
        contactFrame.environmentCount = 1u;
        contactFrame.contactCapacityPerEnvironment =
            first.layout.contactDispatch.constraintCapacity;
        contactFrame.shapes = model.shapes;
        contactFrame.constraints =
            first.contactEvidence.contacts;
        contactFrame.metadata =
            first.contactEvidence.contactMetadata;
        contactFrame.activeContactCounts =
            activeContactCounts;
        metalrobo::TactileSolverContactBatch solverContacts;
        require(
            metalrobo::packTactileSolverContacts(
                contactFrame,
                solverContacts
            ),
            "pack tactile solver impulses"
        );
        if (debugDirectory.has_value()) {
            for (std::uint32_t contactIndex = 0u;
                 contactIndex < solverContacts.counts[0u];
                 ++contactIndex) {
                const auto& contact =
                    solverContacts.contacts[contactIndex];
                std::cerr
                    << "contact_debug index=" << contactIndex
                    << " shapes=" << contact.shapesAndFlags.x
                    << ',' << contact.shapesAndFlags.y
                    << " impulse="
                    << length3(contact.worldImpulseOnA)
                    << '\n';
            }
        }
        const auto tactile = tactileObservation(
            worldTemplate,
            finalBodies,
            solverContacts,
            controlSteps
        );
        if (debugDirectory.has_value() &&
            tactile.summaries.size() == 2u) {
            std::cerr
                << "tactile_debug active_contacts="
                << first.contactStatuses.back().activeContacts
                << " packed_contacts=" << solverContacts.counts[0u]
                << " left_area="
                << tactile.summaries[0u].
                    netForceAndContactArea.w
                << " right_area="
                << tactile.summaries[1u].
                    netForceAndContactArea.w
                << " left_force="
                << length3(
                    tactile.summaries[0u].
                        netForceAndContactArea
                )
                << " right_force="
                << length3(
                    tactile.summaries[1u].
                        netForceAndContactArea
                )
                << " left_max="
                << tactile.summaries[0u].
                    netTorqueAndMaximumDepth.w
                << " right_max="
                << tactile.summaries[1u].
                    netTorqueAndMaximumDepth.w
                << '\n';
        }
        require(
            tactile.summaries.size() == 2u &&
            tactile.summaries[0u].
                netForceAndContactArea.w > 0.0f &&
            tactile.summaries[1u].
                netForceAndContactArea.w > 0.0f &&
            length3(
                tactile.summaries[0u].netForceAndContactArea
            ) > 0.0f &&
            length3(
                tactile.summaries[1u].netForceAndContactArea
            ) > 0.0f,
            "tactile geometry and solver wrenches did not agree on "
            "bilateral contact"
        );

        metalrobo::MetalTactileConfig tactileConfig;
        tactileConfig.contactCapacityPerEnvironment =
            solverContacts.capacityPerEnvironment;
        metalrobo::MetalTactileContext tactileGPU(tactileConfig);
        require(
            tactileGPU.compile(
                worldTemplate.tactileSystem,
                model,
                1u
            ),
            "compile Franka tactile Metal context"
        );
        metalrobo::MetalTactileHostFrame tactileFrame;
        tactileFrame.environmentCount = 1u;
        tactileFrame.bodies = finalBodies;
        tactileFrame.contacts = solverContacts.contacts;
        tactileFrame.contactCounts = solverContacts.counts;
        tactileFrame.observationTimestepSeconds = 0.02f;
        tactileFrame.contactImpulseTimestepSeconds =
            0.02f / 8.0f;
        tactileFrame.frameIndex = controlSteps;
        tactileFrame.timestampSeconds =
            controlSteps * 0.02;
        require(
            tactileGPU.observe(tactileFrame),
            "observe Franka tactile map on Metal"
        );
        metalrobo::TactileObservationBatch tactileMetal;
        require(
            tactileGPU.readback(1u, tactileMetal),
            "read Franka tactile map"
        );
        double maximumDepthError = 0.0;
        double maximumMotionError = 0.0;
        for (std::size_t index = 0u;
             index < tactile.penetrationDepthMeters.size();
             ++index) {
            maximumDepthError = std::max(
                maximumDepthError,
                static_cast<double>(std::abs(
                    tactile.penetrationDepthMeters[index] -
                    tactileMetal.penetrationDepthMeters[index]
                ))
            );
            const mr_float4 cpuMotion =
                tactile.tangentialMotion[index].
                    displacementAndVelocity;
            const mr_float4 gpuMotion =
                tactileMetal.tangentialMotion[index].
                    displacementAndVelocity;
            maximumMotionError = std::max(
                {
                    maximumMotionError,
                    static_cast<double>(
                        std::abs(cpuMotion.x - gpuMotion.x)
                    ),
                    static_cast<double>(
                        std::abs(cpuMotion.y - gpuMotion.y)
                    ),
                    static_cast<double>(
                        std::abs(cpuMotion.z - gpuMotion.z)
                    ),
                    static_cast<double>(
                        std::abs(cpuMotion.w - gpuMotion.w)
                    ),
                }
            );
        }
        require(
            maximumDepthError < 2.0e-6 &&
            maximumMotionError < 2.0e-5,
            "Franka Metal tactile geometry differs from CPU oracle"
        );
        if (debugDirectory.has_value()) {
            for (std::uint32_t sensorIndex = 0u;
                 sensorIndex <
                    worldTemplate.tactileSystem.sensors.size();
                 ++sensorIndex) {
                metalrobo::TactileDebugExportConfig debug;
                debug.directory = *debugDirectory;
                debug.prefix =
                    worldTemplate.tactileSystem.
                        sensorIds[sensorIndex];
                debug.sensor = sensorIndex;
                require(
                    metalrobo::exportTactileDebugFrame(
                        worldTemplate.tactileSystem,
                        tactile,
                        debug
                    ),
                    "export tactile debug frame"
                );
            }
        }
        for (std::uint32_t sensorIndex = 0u;
             sensorIndex < tactile.summaries.size();
             ++sensorIndex) {
            const auto& cpuSummary =
                tactile.summaries[sensorIndex];
            const auto& gpuSummary =
                tactileMetal.summaries[sensorIndex];
            require(
                length3(subtract(
                    cpuSummary.netForceAndContactArea,
                    gpuSummary.netForceAndContactArea
                )) < 2.0e-5f &&
                length3(subtract(
                    cpuSummary.
                        centerOfPressureLocalAndForceWeight,
                    gpuSummary.
                        centerOfPressureLocalAndForceWeight
                )) < 2.0e-5f &&
                length3(subtract(
                    cpuSummary.tangentialMotionAndFriction,
                    gpuSummary.tangentialMotionAndFriction
                )) < 2.0e-5f &&
                std::abs(
                    cpuSummary.tangentialMotionAndFriction.w -
                    gpuSummary.tangentialMotionAndFriction.w
                ) < 2.0e-5f,
                "Franka Metal wrench, center of pressure, or motion differs "
                "from the CPU oracle"
            );
        }

        const float leftMean =
            tactile.summaries[0u].
                centroidLocalAndMeanDepth.w;
        const float rightMean =
            tactile.summaries[1u].
                centroidLocalAndMeanDepth.w;
        constexpr float policyTargetMeanDepth = 0.00016f;
        constexpr float policyMaximumCorrection = 0.00020f;
        const float leftCorrection = std::clamp(
            policyTargetMeanDepth - leftMean,
            0.0f,
            policyMaximumCorrection
        );
        const float rightCorrection = std::clamp(
            policyTargetMeanDepth - rightMean,
            0.0f,
            policyMaximumCorrection
        );
        std::vector<float> holdActions = first.finalQ;
        std::vector<float> feedbackActions = holdActions;
        feedbackActions[7u] -= leftCorrection;
        feedbackActions[8u] -= rightCorrection;
        metalrobo::MetalWorldBatch holdBatch{
            .environmentCount = 1u,
            .controlStepCount = 1u,
            .initialQ = first.finalQ,
            .initialV = first.finalV,
            .efforts = holdActions,
            .initialSceneBodies = first.finalSceneBodies,
        };
        metalrobo::MetalWorldBatch feedbackBatch{
            .environmentCount = 1u,
            .controlStepCount = 1u,
            .initialQ = first.finalQ,
            .initialV = first.finalV,
            .efforts = feedbackActions,
            .initialSceneBodies = first.finalSceneBodies,
        };
        metalrobo::MetalWorldStepConfig continuationConfig =
            physicsConfig;
        continuationConfig.captureContactEvidence = false;
        metalrobo::MetalWorldContext holdPhysics;
        metalrobo::MetalWorldContext feedbackPhysics;
        metalrobo::MetalWorldResult holdContinuation;
        metalrobo::MetalWorldResult feedbackContinuation;
        require(
            holdPhysics.run(
                compiled,
                holdBatch,
                continuationConfig,
                holdContinuation
            ),
            "run open-loop grasp continuation"
        );
        require(
            feedbackPhysics.run(
                compiled,
                feedbackBatch,
                continuationConfig,
                feedbackContinuation
            ),
            "run tactile-feedback grasp continuation"
        );
        const float holdSlipSpeed = std::abs(
            holdContinuation.finalSceneBodies[0u].
                linearVelocityAndInverseMass.z
        );
        const float feedbackSlipSpeed = std::abs(
            feedbackContinuation.finalSceneBodies[0u].
                linearVelocityAndInverseMass.z
        );
        const float feedbackSlipReduction = std::clamp(
            (holdSlipSpeed - feedbackSlipSpeed) /
                std::max(holdSlipSpeed, 1.0e-7f),
            -1.0f,
            1.0f
        );

        metalrobo::EngineModel noGripModel = model;
        noGripModel.name =
            "franka_fer_hand_tactile_no_grip_baseline";
        noGripModel.shapes[27u].collisionMask = 0u;
        noGripModel.shapes[31u].collisionMask = 0u;
        metalrobo::CompiledWorld noGripCompiled;
        require(
            metalrobo::compileMetalWorld(
                noGripModel,
                0u,
                noGripCompiled
            ),
            "compile no-grip evaluation baseline"
        );
        metalrobo::MetalWorldStepConfig noGripConfig =
            physicsConfig;
        noGripConfig.captureContactEvidence = false;
        metalrobo::MetalWorldContext noGripPhysics;
        metalrobo::MetalWorldResult noGrip;
        require(
            noGripPhysics.run(
                noGripCompiled,
                batch,
                noGripConfig,
                noGrip
            ),
            "run no-grip evaluation baseline"
        );

        metalrobo::MetalWorldResult replay;
        const auto replayRun = physics.run(
            compiled,
            batch,
            physicsConfig,
            replay
        );
        require(
            replayRun.succeeded() &&
            first.finalQ == replay.finalQ &&
            first.finalV == replay.finalV &&
            first.finalSceneBodies.size() ==
                replay.finalSceneBodies.size() &&
            std::equal(
                first.finalSceneBodies.begin(),
                first.finalSceneBodies.end(),
                replay.finalSceneBodies.begin(),
                [](const MRBodyStateGPU& left,
                   const MRBodyStateGPU& right) {
                    return std::memcmp(
                        &left,
                        &right,
                        sizeof(left)
                    ) == 0;
                }
            ),
            "tactile grasp deterministic replay diverged"
        );

        const float balanceError =
            std::abs(leftMean - rightMean);
        const float tactileSlipSpeed = std::abs(
            first.finalSceneBodies[0u].
                linearVelocityAndInverseMass.z
        );
        const float baselineSlipSpeed = std::abs(
            noGrip.finalSceneBodies[0u].
                linearVelocityAndInverseMass.z
        );
        const float balanceScore = std::clamp(
            1.0f -
                balanceError /
                    std::max(
                        std::max(leftMean, rightMean),
                        1.0e-7f
                    ),
            0.0f,
            1.0f
        );
        const float slipReduction = std::clamp(
            (
                baselineSlipSpeed - tactileSlipSpeed
            ) / std::max(baselineSlipSpeed, 1.0e-7f),
            -1.0f,
            1.0f
        );
        const float stabilizationReward =
            balanceScore *
            std::max(feedbackSlipReduction, 0.0f);
        if (debugDirectory.has_value()) {
            std::cerr
                << "reward_debug left_mean=" << leftMean
                << " right_mean=" << rightMean
                << " balance_score=" << balanceScore
                << " baseline_slip=" << baselineSlipSpeed
                << " tactile_slip=" << tactileSlipSpeed
                << " slip_reduction=" << slipReduction
                << " hold_continuation_slip=" << holdSlipSpeed
                << " feedback_continuation_slip="
                << feedbackSlipSpeed
                << " feedback_slip_reduction="
                << feedbackSlipReduction
                << " reward=" << stabilizationReward
                << '\n';
        }
        require(
            leftMean > 0.0f &&
            rightMean > 0.0f &&
            balanceScore > 0.85f &&
            slipReduction > 0.05f &&
            feedbackSlipReduction > 0.0f &&
            stabilizationReward > 0.0f,
            "tactile-aware grasp-stabilization evaluation failed"
        );

        std::cout
            << "scenario=franka-grasp"
            << " device=\"" << firstRun.deviceName << "\""
            << " active_solver_contacts="
            << first.contactStatuses.back().activeContacts
            << " left_mean_depth_m=" << leftMean
            << " right_mean_depth_m=" << rightMean
            << " balance_error_m=" << balanceError
            << " baseline_slip_speed_mps="
            << baselineSlipSpeed
            << " tactile_slip_speed_mps="
            << tactileSlipSpeed
            << " slip_reduction=" << slipReduction
            << " hold_continuation_slip_speed_mps="
            << holdSlipSpeed
            << " feedback_continuation_slip_speed_mps="
            << feedbackSlipSpeed
            << " feedback_slip_reduction="
            << feedbackSlipReduction
            << " stabilization_reward="
            << stabilizationReward
            << " max_cpu_gpu_error_m="
            << maximumDepthError
            << " max_cpu_gpu_motion_error="
            << maximumMotionError
            << " deterministic_replay=yes"
            << " native_depth_buffer="
            << (
                tactileGPU.nativeBuffer(
                    metalrobo::MetalTactileBuffer::
                        penetrationDepth
                ) != nullptr
                ? "yes"
                : "no"
            )
            << " native_tangential_motion_buffer="
            << (
                tactileGPU.nativeBuffer(
                    metalrobo::MetalTactileBuffer::
                        tangentialMotion
                ) != nullptr
                ? "yes"
                : "no"
            )
            << " world_pack="
            << (
                worldPackPath.has_value()
                ? worldPackPath->string()
                : "validated-temporary"
            )
            << " debug_export="
            << (
                debugDirectory.has_value()
                ? debugDirectory->string()
                : "disabled"
            )
            << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_tactile_example: "
                  << error.what() << '\n';
        return 1;
    }
}
