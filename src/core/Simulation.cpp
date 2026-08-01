#include "metalrobo/Simulation.hpp"

#include "metalrobo/G1.hpp"
#include "metalrobo/WorldPack.hpp"

#include "SemanticTransform.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace metalrobo {
namespace {

constexpr std::uint64_t kFingerprintOffset =
    1469598103934665603ull;
constexpr std::uint64_t kFingerprintPrime =
    1099511628211ull;

void appendFingerprint(
    std::uint64_t& hash,
    const std::uint64_t value
) {
    for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
        hash ^= (value >> (byte * 8u)) & 0xffu;
        hash *= kFingerprintPrime;
    }
}

MRBodyPropertiesGPU staticBody() {
    MRBodyPropertiesGPU body{};
    body.articulationIndex = MR_INVALID_INDEX;
    body.parentBody = MR_INVALID_INDEX;
    body.inboundJoint = MR_INVALID_INDEX;
    body.motionType = MR_MOTION_STATIC;
    body.dampingAndSpeedLimits =
        {0.0f, 0.0f, 1.0e6f, 1.0e6f};
    return body;
}

MRBodyStateGPU staticState(const std::uint32_t bodyIndex) {
    MRBodyStateGPU state{};
    state.position.w = 1.0f;
    state.orientation.w = 1.0f;
    state.flagsAndIndices[0] = MR_MOTION_STATIC;
    state.flagsAndIndices[1] = MR_INVALID_INDEX;
    state.flagsAndIndices[2] = bodyIndex;
    return state;
}

mr_float4 normalizedQuaternion(
    const mr_float4 authored
) {
    const double normSquared =
        static_cast<double>(authored.x) * authored.x +
        static_cast<double>(authored.y) * authored.y +
        static_cast<double>(authored.z) * authored.z +
        static_cast<double>(authored.w) * authored.w;
    if (!std::isfinite(normSquared) ||
        !(normSquared > 0.0)) {
        throw std::invalid_argument(
            "WorldPack asset orientation is invalid"
        );
    }
    const float inverseNorm =
        static_cast<float>(1.0 / std::sqrt(normSquared));
    return {
        authored.x * inverseNorm,
        authored.y * inverseNorm,
        authored.z * inverseNorm,
        authored.w * inverseNorm,
    };
}

std::array<std::array<float, 3u>, 3u> rotationMatrix(
    const mr_float4 authored
) {
    const mr_float4 normalized =
        normalizedQuaternion(authored);
    const float x = normalized.x;
    const float y = normalized.y;
    const float z = normalized.z;
    const float w = normalized.w;
    return {{
        {{
            1.0f - 2.0f * (y * y + z * z),
            2.0f * (x * y - z * w),
            2.0f * (x * z + y * w),
        }},
        {{
            2.0f * (x * y + z * w),
            1.0f - 2.0f * (x * x + z * z),
            2.0f * (y * z - x * w),
        }},
        {{
            2.0f * (x * z - y * w),
            2.0f * (y * z + x * w),
            1.0f - 2.0f * (x * x + y * y),
        }},
    }};
}

WorldPose composePose(
    const WorldPose& parent,
    const WorldPose& local
) {
    WorldPose result{};
    if (!semantic_transform::compose(
            parent.position,
            parent.orientation,
            local.position,
            local.orientation,
            result.position,
            result.orientation
        )) {
        throw std::invalid_argument(
            "WorldPack pose composition is invalid"
        );
    }
    return result;
}

void writeWorldInverseInertia(
    MRBodyStateGPU& state,
    const MRBodyPropertiesGPU& properties,
    const float massScale
) {
    if (properties.motionType != MR_MOTION_DYNAMIC) {
        return;
    }
    if (!std::isfinite(massScale) || !(massScale > 0.0f)) {
        throw std::invalid_argument(
            "WorldPack dynamic asset mass scale is invalid"
        );
    }
    const auto rotation = rotationMatrix(state.orientation);
    const std::array<std::array<float, 3u>, 3u> body{{
        std::array{
            properties.inverseInertiaRow0.x,
            properties.inverseInertiaRow0.y,
            properties.inverseInertiaRow0.z,
        },
        std::array{
            properties.inverseInertiaRow1.x,
            properties.inverseInertiaRow1.y,
            properties.inverseInertiaRow1.z,
        },
        std::array{
            properties.inverseInertiaRow2.x,
            properties.inverseInertiaRow2.y,
            properties.inverseInertiaRow2.z,
        },
    }};
    float world[3u][3u]{};
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u;
             column < 3u;
             ++column) {
            for (std::size_t left = 0u;
                 left < 3u;
                 ++left) {
                for (std::size_t right = 0u;
                     right < 3u;
                     ++right) {
                    world[row][column] +=
                        rotation[row][left] *
                        body[left][right] *
                        rotation[column][right] /
                        massScale;
                }
            }
        }
    }
    state.inverseInertiaWorldRow0 = {
        world[0u][0u],
        world[0u][1u],
        world[0u][2u],
        0.0f,
    };
    state.inverseInertiaWorldRow1 = {
        world[1u][0u],
        world[1u][1u],
        world[1u][2u],
        0.0f,
    };
    state.inverseInertiaWorldRow2 = {
        world[2u][0u],
        world[2u][1u],
        world[2u][2u],
        0.0f,
    };
}

void appendGround(
    EngineModel& model,
    std::vector<MRBodyStateGPU>& sceneBodies
) {
    const std::uint32_t bodyIndex =
        static_cast<std::uint32_t>(model.bodies.size());
    model.bodies.push_back(staticBody());
    if (!model.bodyNames.empty()) {
        model.bodyNames.emplace_back("locomotion_ground");
    }

    MRShapeGPU plane{};
    plane.bodyIndex = bodyIndex;
    plane.shapeType = MR_SHAPE_PLANE;
    plane.materialIndex = 0u;
    plane.collisionGroup = 1u;
    plane.collisionMask = ~0u;
    plane.slotGeneration = 1u;
    plane.localPosition.w = 1.0f;
    // The collision plane is authored +Y. Robotics worlds are Z-up.
    constexpr float kSqrtHalf = 0.7071067811865476f;
    plane.localRotation = {
        kSqrtHalf,
        0.0f,
        0.0f,
        kSqrtHalf,
    };
    model.shapes.push_back(plane);
    if (!model.shapeNames.empty()) {
        model.shapeNames.emplace_back(
            "locomotion_ground/collision"
        );
    }
    sceneBodies.push_back(staticState(bodyIndex));
}

float terrainHeight(const float x, const float y) {
    if (x < -2.0f) {
        return 0.0f;
    }
    if (x < 0.0f) {
        return 0.08f * (x + 2.0f);
    }
    if (x < 2.0f) {
        return 0.16f + 0.03f * std::floor(x / 0.25f);
    }
    return
        0.37f +
        0.035f *
            (std::sin(1.7f * x) - std::sin(3.4f)) *
            std::sin(1.3f * y) +
        0.015f *
            (std::sin(3.1f * x) - std::sin(6.2f)) *
            std::sin(2.3f * y);
}

void appendTerrain(
    EngineModel& model,
    std::vector<MRBodyStateGPU>& sceneBodies
) {
    constexpr std::uint32_t side = 161u;
    constexpr float spacing = 0.05f;
    constexpr float halfWidth =
        0.5f * spacing * static_cast<float>(side - 1u);

    std::vector<mr_float4> vertices;
    vertices.reserve(side * side);
    float minimumHeight =
        std::numeric_limits<float>::infinity();
    float maximumHeight =
        -std::numeric_limits<float>::infinity();
    for (std::uint32_t y = 0u; y < side; ++y) {
        for (std::uint32_t x = 0u; x < side; ++x) {
            const float px =
                -halfWidth + spacing * static_cast<float>(x);
            const float py =
                -halfWidth + spacing * static_cast<float>(y);
            const float height = terrainHeight(px, py);
            minimumHeight = std::min(minimumHeight, height);
            maximumHeight = std::max(maximumHeight, height);
            vertices.push_back({px, py, height, 1.0f});
        }
    }

    const std::uint32_t geometryIndex =
        static_cast<std::uint32_t>(
            model.geometryHeaders.size()
        );
    MRGeometryHeaderGPU geometry{};
    geometry.kind = MR_GEOMETRY_HEIGHTFIELD;
    geometry.vertexOffset =
        static_cast<std::uint32_t>(
            model.geometryVertices.size()
        );
    geometry.vertexCount =
        static_cast<std::uint32_t>(vertices.size());
    geometry.localLower = {
        -halfWidth,
        -halfWidth,
        minimumHeight,
        spacing,
    };
    geometry.localUpper = {
        halfWidth,
        halfWidth,
        maximumHeight,
        1.0f / spacing,
    };
    model.geometryHeaders.push_back(geometry);
    model.geometryVertices.insert(
        model.geometryVertices.end(),
        vertices.begin(),
        vertices.end()
    );

    const std::uint32_t bodyIndex =
        static_cast<std::uint32_t>(model.bodies.size());
    model.bodies.push_back(staticBody());
    if (!model.bodyNames.empty()) {
        model.bodyNames.emplace_back("locomotion_terrain");
    }

    MRShapeGPU heightfield{};
    heightfield.bodyIndex = bodyIndex;
    heightfield.shapeType = MR_SHAPE_HEIGHTFIELD;
    heightfield.materialIndex = 0u;
    heightfield.collisionGroup = 1u;
    heightfield.collisionMask = ~0u;
    heightfield.slotGeneration = 1u;
    heightfield.localPosition.w = 1.0f;
    heightfield.localRotation.w = 1.0f;
    heightfield.dimensions = {1.0f, 1.0f, 1.0f, 0.0f};
    heightfield.contactRestAndBoundingRadius =
        {0.002f, 0.0f, 5.8f, 0.0f};
    heightfield.geometryOffset = geometryIndex;
    heightfield.geometryCount = 1u;
    model.shapes.push_back(heightfield);
    if (!model.shapeNames.empty()) {
        model.shapeNames.emplace_back(
            "locomotion_terrain/collision"
        );
    }
    sceneBodies.push_back(staticState(bodyIndex));
}

} // namespace

bool CompiledSimulation::valid() const noexcept {
    return world.valid() && sensors.valid() && task.valid() &&
        sensors.worldFingerprint() == world.fingerprint() &&
        (!policy.valid() ||
         policy.taskFingerprint() == task.fingerprint());
}

std::uint64_t CompiledSimulation::fingerprint() const noexcept {
    if (!valid()) {
        return 0u;
    }
    std::uint64_t hash = kFingerprintOffset;
    appendFingerprint(hash, world.fingerprint());
    appendFingerprint(hash, sensors.fingerprint());
    appendFingerprint(hash, task.fingerprint());
    appendFingerprint(
        hash,
        policy.valid() ? policy.fingerprint() : 0u
    );
    return hash;
}

SimulationCompileDiagnostics compileSimulation(
    const SimulationDescription& authored,
    const std::uint32_t articulationIndex,
    CompiledSimulation& compiled
) {
    SimulationCompileDiagnostics diagnostics;
    CompiledSimulation staged;
    diagnostics.world = compileMetalWorld(
        authored.model,
        articulationIndex,
        staged.world,
        authored.task.capacities
    );
    if (!diagnostics.world.succeeded()) {
        return diagnostics;
    }
    diagnostics.sensors = compileSensorProgram(
        authored.sensors,
        authored.tactileSystem,
        staged.world,
        staged.sensors
    );
    if (!diagnostics.sensors.succeeded()) {
        return diagnostics;
    }
    diagnostics.task = compileTaskProgram(
        authored.task,
        staged.world,
        staged.sensors,
        staged.task
    );
    if (!diagnostics.task.succeeded()) {
        return diagnostics;
    }
    diagnostics.policyRequested = authored.policy.has_value();
    if (authored.policy.has_value()) {
        diagnostics.policy = compilePolicyProgram(
            *authored.policy,
            staged.task,
            staged.policy
        );
        if (!diagnostics.policy.succeeded()) {
            return diagnostics;
        }
    }
    compiled = std::move(staged);
    return diagnostics;
}

void appendBuiltinSurface(
    EngineModel& model,
    std::vector<MRBodyStateGPU>& sceneBodies,
    const BuiltinSurface surface
) {
    if (model.materials.empty()) {
        throw std::invalid_argument(
            "locomotion surface requires a model material"
        );
    }
    switch (surface) {
    case BuiltinSurface::ground:
        appendGround(model, sceneBodies);
        break;
    case BuiltinSurface::terrain:
        appendTerrain(model, sceneBodies);
        break;
    default:
        throw std::invalid_argument(
            "locomotion surface is invalid"
        );
    }
    model.world.bodyCount =
        static_cast<std::uint32_t>(model.bodies.size());
    model.world.shapeCount =
        static_cast<std::uint32_t>(model.shapes.size());
}

SimulationDescription makeWorldPackSimulation(
    const MRWorldPack& worldPack,
    TaskPack task
) {
    std::string reason;
    if (!worldPack.valid(&reason)) {
        throw std::invalid_argument(
            "invalid MRWorldPack: " + reason
        );
    }
    const WorldTemplate& authored =
        worldPack.family.worldTemplate;
    const std::uint32_t robotAssetIndex =
        authored.assetIndex(authored.task.robotAssetId);
    if (robotAssetIndex == MR_INVALID_INDEX) {
        throw std::invalid_argument(
            "MRWorldPack task robot asset is unresolved"
        );
    }
    const WorldAsset& robot =
        authored.assets[robotAssetIndex];
    if (robot.articulationIndex == MR_INVALID_INDEX ||
        robot.articulationIndex >=
            authored.engineModel.articulations.size()) {
        throw std::invalid_argument(
            "MRWorldPack task robot has no executable articulation"
        );
    }

    SimulationDescription result;
    result.model = authored.engineModel;
    result.sensors = authored.sensors;
    result.tactileSystem = authored.tactileSystem;
    result.task = std::move(task);
    result.articulationIndex = robot.articulationIndex;
    std::vector<std::uint32_t> bodyToScene(
        result.model.bodies.size(),
        MR_INVALID_INDEX
    );
    for (std::uint32_t body = 0u;
         body < result.model.bodies.size();
         ++body) {
        const MRBodyPropertiesGPU& properties =
            result.model.bodies[body];
        if (properties.articulationIndex !=
            MR_INVALID_INDEX) {
            continue;
        }
        bodyToScene[body] =
            static_cast<std::uint32_t>(
                result.sceneBodies.size()
            );
        MRBodyStateGPU state{};
        state.position.w = 1.0f;
        state.orientation.w = 1.0f;
        state.flagsAndIndices[0] = properties.motionType;
        state.flagsAndIndices[1] = MR_INVALID_INDEX;
        state.flagsAndIndices[2] = body;
        if (properties.motionType == MR_MOTION_DYNAMIC) {
            state.linearVelocityAndInverseMass.w =
                properties.massAndInverseMass.y;
            writeWorldInverseInertia(
                state,
                properties,
                1.0f
            );
        }
        result.sceneBodies.push_back(state);
    }
    for (const WorldAsset& asset : authored.assets) {
        for (const std::uint32_t body : asset.bodyIndices) {
            if (body >= bodyToScene.size()) {
                throw std::invalid_argument(
                    "MRWorldPack asset body mapping exceeds its topology"
                );
            }
            const std::uint32_t localScene =
                bodyToScene[body];
            if (localScene == MR_INVALID_INDEX) {
                continue;
            }
            MRBodyStateGPU& state =
                result.sceneBodies[localScene];
            state.position = {
                asset.initialPose.position.x,
                asset.initialPose.position.y,
                asset.initialPose.position.z,
                1.0f,
            };
            state.orientation = normalizedQuaternion(
                asset.initialPose.orientation
            );
            const MRBodyPropertiesGPU& properties =
                result.model.bodies[body];
            if (properties.motionType == MR_MOTION_DYNAMIC &&
                (!std::isfinite(asset.massScale) ||
                 !(asset.massScale > 0.0f))) {
                throw std::invalid_argument(
                    "MRWorldPack dynamic asset mass scale is invalid"
                );
            }
            state.linearVelocityAndInverseMass.w =
                properties.motionType == MR_MOTION_DYNAMIC
                ? properties.massAndInverseMass.y /
                    asset.massScale
                : 0.0f;
            writeWorldInverseInertia(
                state,
                properties,
                asset.massScale
            );
        }
    }
    for (SensorSpec& sensor : result.sensors) {
        if (sensor.parentKind !=
            MR_WORLD_SENSOR_PARENT_ASSET) {
            continue;
        }
        if (!sensor.parentSite.empty()) {
            const auto site = std::ranges::find_if(
                result.model.sites,
                [&](const EngineSite& candidate) {
                    return candidate.id == sensor.parentSite;
                }
            );
            const std::uint32_t parentIndex =
                authored.assetIndex(sensor.parentAssetId);
            if (site == result.model.sites.end() ||
                parentIndex == MR_INVALID_INDEX ||
                std::ranges::find(
                    authored.assets[parentIndex].bodyIndices,
                    site->bodyIndex
                ) == authored.assets[parentIndex]
                        .bodyIndices.end()) {
                throw std::invalid_argument(
                    "MRWorldPack sensor parent site is unresolved or outside its asset"
                );
            }
            sensor.localPose = composePose(
                {
                    site->localPosition,
                    site->localOrientation,
                },
                sensor.localPose
            );
            sensor.parentBodyIndex = site->bodyIndex;
            sensor.parentKind =
                result.model.bodies[site->bodyIndex]
                        .articulationIndex == MR_INVALID_INDEX
                ? MR_WORLD_SENSOR_PARENT_RIGID_BODY
                : MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK;
            sensor.parentSite.clear();
            continue;
        }
        // Joint/actuator SensorIR is asset-owned but not spatially attached.
        // Preserve that semantic parent so SensorCompiler can resolve its
        // target directly into generalized-coordinate indices.
        if (sensor.kind == MR_WORLD_SENSOR_JOINT_STATE ||
            sensor.kind == MR_WORLD_SENSOR_ACTUATOR_STATE) {
            continue;
        }
        const std::uint32_t parentIndex =
            authored.assetIndex(sensor.parentAssetId);
        if (parentIndex == MR_INVALID_INDEX) {
            throw std::invalid_argument(
                "MRWorldPack sensor parent asset is unresolved"
            );
        }
        const WorldAsset& parent = authored.assets[parentIndex];
        if (parent.bodyIndices.size() == 1u) {
            sensor.parentBodyIndex = parent.bodyIndices.front();
            if (sensor.parentBodyIndex >= result.model.bodies.size()) {
                throw std::invalid_argument(
                    "MRWorldPack sensor parent body exceeds its topology"
                );
            }
            sensor.parentKind =
                result.model.bodies[sensor.parentBodyIndex]
                        .articulationIndex == MR_INVALID_INDEX
                ? MR_WORLD_SENSOR_PARENT_RIGID_BODY
                : MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK;
        } else if (parent.bodyIndices.empty()) {
            sensor.localPose = composePose(
                parent.initialPose,
                sensor.localPose
            );
            sensor.parentKind = MR_WORLD_SENSOR_PARENT_WORLD;
            sensor.parentBodyIndex = MR_INVALID_INDEX;
        } else {
            throw std::invalid_argument(
                "MRWorldPack asset-relative sensor on a multi-body asset "
                "must name its parent body"
            );
        }
    }
    return result;
}

TaskPack makeUnitreeG1TaskPack(
    const BuiltinSurface surface
) {
    const G1ModelMetadata& metadata = unitreeG1Metadata();
    TaskPack task;
    task.id = "unitree_g1_locomotion";
    // The topology envelope contains every eligible self-collision pair.
    // Locomotion instead compiles an explicit operational arena. Capacity
    // overflow is transactional and reports the exact required stage count,
    // so this is a replayable task contract rather than silent truncation.
    task.capacities = {
        .candidatePairs = 128u,
        .rawContacts = 128u,
        .manifolds = 32u,
        .hardConvexPairs = 64u,
        .meshTriangleCandidates = 1024u,
        .ccdCandidates = 64u,
        .ccdEvents = 8u,
    };
    task.actorHistoryLength = 5u;
    task.criticHistoryLength = 5u;
    task.criticIncludesCleanHistory = false;
    task.maximumEpisodeSteps = 1000u;
    task.maximumActionDelaySteps = 0u;
    task.maximumObservationDelaySteps = 0u;
    task.curriculumLevelCount = 11u;
    task.baseHeightTarget = 0.78f;
    task.gaitPeriodSeconds = 0.8f;
    task.clearanceTarget = 0.10f;
    task.successTrackingThreshold = 0.8f;
    task.supportForceThreshold = 1.0f;
    // Level zero is a near-static balance task with a fixed 0.1 m/s forward
    // command. This avoids an unnecessarily brittle inverted-pendulum optimum
    // while remaining below the gait-reward threshold; lateral and yaw motion
    // stay disabled until full-episode survival has been demonstrated.
    task.commands.lower = {0.1f, 0.0f, 0.0f, 0.0f};
    task.commands.upper = {0.1f, 0.0f, 0.0f, 0.0f};
    task.commands.limitLower = {
        -0.5f, -0.3f, -0.2f, 0.0f,
    };
    task.commands.limitUpper = {
        1.0f, 0.3f, 0.2f, 0.0f,
    };
    task.commands.curriculumStep = {
        0.1f, 0.1f, 0.1f, 0.0f,
    };
    task.commands.standingProbability = 0.02f;
    task.commands.minimumEpisodeSurvivalFraction = 0.8f;
    task.commands.minimumDurationSeconds = 10.0f;
    task.commands.maximumDurationSeconds = 10.0f;
    task.pushes.maximumVelocity = 0.5f;
    task.pushes.minimumIntervalSeconds = 5.0f;
    task.pushes.maximumIntervalSeconds = 5.0f;

    task.actions.reserve(metadata.jointLimits.size());
    for (const G1JointLimit& joint : metadata.jointLimits) {
        task.actions.push_back({
            .joint = std::string{joint.name},
            .scale = 0.25f,
        });
    }

    const auto observation =
        [](const TaskObservationSource source,
           const std::string_view target,
           const std::uint32_t component,
           const float scale = 1.0f,
           const float noise = 0.0f,
           const float biasLower = 0.0f,
           const float biasUpper = 0.0f,
           const bool normalize = false) {
            return TaskObservationOperatorSpec{
                .source = source,
                .target = std::string{target},
                .component = component,
                .scale = scale,
                .noiseAmplitude = noise,
                .biasLower = biasLower,
                .biasUpper = biasUpper,
                .normalizeVector3 = normalize,
            };
        };
    for (std::uint32_t component = 0u;
         component < 3u;
         ++component) {
        task.actorFrame.push_back(observation(
            TaskObservationSource::rootAngularVelocityLocal,
            {},
            component,
            0.2f,
            0.04f
        ));
    }
    for (std::uint32_t component = 0u;
         component < 3u;
         ++component) {
        task.actorFrame.push_back(observation(
            TaskObservationSource::projectedGravity,
            {},
            component,
            1.0f,
            0.05f,
            0.0f,
            0.0f
        ));
    }
    for (std::uint32_t component = 0u;
         component < 3u;
         ++component) {
        task.actorFrame.push_back(observation(
            TaskObservationSource::command,
            {},
            component
        ));
    }
    for (const G1JointLimit& joint : metadata.jointLimits) {
        task.actorFrame.push_back(observation(
            TaskObservationSource::jointPositionError,
            joint.name,
            0u,
            1.0f,
            0.01f,
            0.0f,
            0.0f
        ));
    }
    for (const G1JointLimit& joint : metadata.jointLimits) {
        task.actorFrame.push_back(observation(
            TaskObservationSource::jointVelocity,
            joint.name,
            0u,
            0.05f,
            0.075f
        ));
    }
    for (const G1JointLimit& joint : metadata.jointLimits) {
        task.actorFrame.push_back(observation(
            TaskObservationSource::previousAction,
            joint.name,
            0u
        ));
    }

    const auto bodyNames =
        [&metadata](
            const std::initializer_list<std::uint32_t> indices
        ) {
            std::vector<std::string> names;
            names.reserve(indices.size());
            for (const std::uint32_t index : indices) {
                names.emplace_back(metadata.bodyNames[index]);
            }
            return names;
        };
    task.contactGroups.push_back({
        .id = "left_foot_contact",
        .bodies = bodyNames({6u}),
        .support = true,
        .referenceBody =
            std::string{metadata.feet[0].bodyName},
        .localReference = {
            metadata.feet[0].solePosition.x,
            metadata.feet[0].solePosition.y,
            metadata.feet[0].solePosition.z,
            0.0f,
        },
        .gaitPhaseOffsetRadians = 0.0f,
        .stanceFraction = 0.55f,
    });
    task.contactGroups.push_back({
        .id = "right_foot_contact",
        .bodies = bodyNames({12u}),
        .support = true,
        .referenceBody =
            std::string{metadata.feet[1].bodyName},
        .localReference = {
            metadata.feet[1].solePosition.x,
            metadata.feet[1].solePosition.y,
            metadata.feet[1].solePosition.z,
            0.0f,
        },
        .gaitPhaseOffsetRadians =
            3.14159265358979323846f,
        .stanceFraction = 0.55f,
    });
    std::vector<std::string> undesiredContactBodies;
    undesiredContactBodies.reserve(
        metadata.bodyNames.size() - 2u
    );
    for (std::uint32_t body = 0u;
         body < metadata.bodyNames.size();
         ++body) {
        if (body != 6u && body != 12u) {
            undesiredContactBodies.emplace_back(
                metadata.bodyNames[body]
            );
        }
    }
    task.contactGroups.push_back({
        .id = "undesired_contact",
        .bodies = std::move(undesiredContactBodies),
        .forbidden = true,
    });
    std::vector<std::string> robotBodies;
    robotBodies.reserve(metadata.bodyNames.size());
    for (const std::string_view name : metadata.bodyNames) {
        robotBodies.emplace_back(name);
    }
    task.contactGroups.push_back({
        .id = "robot",
        .bodies = std::move(robotBodies),
    });

    const auto jointNames =
        [&metadata](
            const std::initializer_list<std::uint32_t> indices
        ) {
            std::vector<std::string> names;
            names.reserve(indices.size());
            for (const std::uint32_t index : indices) {
                names.emplace_back(
                    metadata.jointLimits[index].name
                );
            }
            return names;
        };
    const std::vector<std::string> waistJoints =
        jointNames({12u, 13u, 14u});
    const std::vector<std::string> hipJoints =
        jointNames({1u, 2u, 7u, 8u});
    std::vector<std::string> armJoints;
    for (std::uint32_t joint = 15u; joint < 29u; ++joint) {
        armJoints.emplace_back(
            metadata.jointLimits[joint].name
        );
    }

    // Unitree's critic is a separate clean 99-value frame with five-frame
    // history. Keep this authored as ordinary observation operators so every
    // imported robot can define the same asymmetric actor/critic contract.
    for (std::uint32_t component = 0u;
         component < 3u;
         ++component) {
        task.critic.push_back(observation(
            TaskObservationSource::rootLinearVelocityLocal,
            {},
            component
        ));
    }
    for (std::uint32_t component = 0u;
         component < 3u;
         ++component) {
        task.critic.push_back(observation(
            TaskObservationSource::rootAngularVelocityLocal,
            {},
            component,
            0.2f
        ));
    }
    for (std::uint32_t component = 0u;
         component < 3u;
         ++component) {
        task.critic.push_back(observation(
            TaskObservationSource::projectedGravity,
            {},
            component
        ));
    }
    for (std::uint32_t component = 0u;
         component < 3u;
         ++component) {
        task.critic.push_back(observation(
            TaskObservationSource::command,
            {},
            component
        ));
    }
    for (const G1JointLimit& joint : metadata.jointLimits) {
        task.critic.push_back(observation(
            TaskObservationSource::jointPositionError,
            joint.name,
            0u
        ));
    }
    for (const G1JointLimit& joint : metadata.jointLimits) {
        task.critic.push_back(observation(
            TaskObservationSource::jointVelocity,
            joint.name,
            0u,
            0.05f
        ));
    }
    for (const G1JointLimit& joint : metadata.jointLimits) {
        task.critic.push_back(observation(
            TaskObservationSource::previousAction,
            joint.name,
            0u
        ));
    }

    task.terrain.body =
        surface == BuiltinSurface::terrain
        ? "locomotion_terrain"
        : "locomotion_ground";
    task.terrain.sampleOffsets.reserve(17u * 11u);
    for (std::uint32_t x = 0u; x < 17u; ++x) {
        for (std::uint32_t y = 0u; y < 11u; ++y) {
            task.terrain.sampleOffsets.push_back({
                -0.8f + 0.1f * static_cast<float>(x),
                -0.5f + 0.1f * static_cast<float>(y),
                0.0f,
                0.0f,
            });
        }
    }
    if (surface == BuiltinSurface::terrain) {
        task.terrain.resetTranslations = {
            {3.0f, 0.0f, 0.0f, 0.0f},
            {3.0f, 0.0f, 0.0f, 0.0f},
            {3.0f, 0.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, -0.08f, 0.0f},
            {1.0f, 0.0f, -0.08f, 0.0f},
            {-3.0f, 0.0f, -0.37f, 0.0f},
            {-3.0f, 0.0f, -0.37f, 0.0f},
            {-1.0f, 0.0f, -0.28f, 0.0f},
            {-1.0f, 0.0f, -0.28f, 0.0f},
            {-1.0f, 0.0f, -0.28f, 0.0f},
            {-1.0f, 0.0f, -0.28f, 0.0f},
        };
    } else {
        task.terrain.resetTranslations = {
            {0.0f, 0.0f, 0.0f, 0.0f},
        };
    }
    const auto sourceSignal =
        [&task](
            const std::string& id,
            const TaskObservationSource source,
            const std::string_view target,
            const std::uint32_t component
        ) {
            task.signals.push_back({
                .id = id,
                .operation = TaskSignalOperator::source,
                .source = {
                    .source = source,
                    .target = std::string{target},
                    .component = component,
                },
            });
            return id;
        };
    const auto constantSignal =
        [&task](const std::string& id, const float value) {
            task.signals.push_back({
                .id = id,
                .operation = TaskSignalOperator::constant,
                .parameters = {value, 0.0f, 0.0f, 0.0f},
            });
            return id;
        };
    const auto unarySignal =
        [&task](
            const std::string& id,
            const TaskSignalOperator operation,
            const std::string& operand
        ) {
            task.signals.push_back({
                .id = id,
                .operation = operation,
                .left = operand,
            });
            return id;
        };
    const auto binarySignal =
        [&task](
            const std::string& id,
            const TaskSignalOperator operation,
            const std::string& left,
            const std::string& right
        ) {
            task.signals.push_back({
                .id = id,
                .operation = operation,
                .left = left,
                .right = right,
            });
            return id;
        };
    const auto sumSignals =
        [&binarySignal](
            const std::string& id,
            const std::vector<std::string>& terms
        ) {
            if (terms.empty()) {
                throw std::logic_error(
                    "TaskIR reduction requires at least one term"
                );
            }
            std::string sum = terms.front();
            for (std::size_t index = 1u;
                 index < terms.size();
                 ++index) {
                sum = binarySignal(
                    id + "_" + std::to_string(index),
                    TaskSignalOperator::add,
                    sum,
                    terms[index]
                );
            }
            return sum;
        };
    const auto signalReward =
        [&task](
            const std::string& signal,
            const float weight,
            const TaskRewardChannel channel
        ) {
            task.rewards.push_back({
                .operation = TaskRewardOperator::signal,
                .signal = signal,
                .channel = channel,
                .weight = weight,
            });
        };
    const auto specializedReward =
        [&task](
            const TaskRewardOperator operation,
            const float weight,
            const TaskRewardChannel channel,
            const std::string_view group = {},
            const mr_float4 parameters = {}
        ) {
            task.rewards.push_back({
                .operation = operation,
                .sourceGroup = std::string{group},
                .channel = channel,
                .weight = weight,
                .parameters = parameters,
            });
        };

    specializedReward(
        TaskRewardOperator::linearVelocityTracking,
        1.0f,
        TaskRewardChannel::primary,
        {},
        {0.25f, 0.0f, 0.0f, 0.0f}
    );
    specializedReward(
        TaskRewardOperator::yawVelocityTracking,
        0.5f,
        TaskRewardChannel::primary,
        {},
        {0.25f, 0.0f, 0.0f, 0.0f}
    );
    signalReward(
        constantSignal("alive", 1.0f),
        0.15f,
        TaskRewardChannel::primary
    );

    const std::string rootVerticalVelocity = sourceSignal(
        "root_vertical_velocity",
        TaskObservationSource::rootLinearVelocityLocal,
        {},
        2u
    );
    const std::string rootVerticalVelocitySquared = unarySignal(
        "root_vertical_velocity_squared",
        TaskSignalOperator::square,
        rootVerticalVelocity
    );
    signalReward(
        rootVerticalVelocitySquared,
        -2.0f,
        TaskRewardChannel::stability
    );

    std::vector<std::string> rootRollPitchTerms;
    for (std::uint32_t component = 0u;
         component < 2u;
         ++component) {
        const std::string leaf = sourceSignal(
            "root_roll_pitch_velocity_" +
                std::to_string(component),
            TaskObservationSource::rootAngularVelocityLocal,
            {},
            component
        );
        rootRollPitchTerms.push_back(unarySignal(
            leaf + "_squared",
            TaskSignalOperator::square,
            leaf
        ));
    }
    signalReward(
        sumSignals(
            "root_roll_pitch_velocity_squared",
            rootRollPitchTerms
        ),
        -0.05f,
        TaskRewardChannel::stability
    );

    std::vector<std::string> gravityHorizontalTerms;
    std::array<std::string, 3u> gravitySignals;
    for (std::uint32_t component = 0u;
         component < gravitySignals.size();
         ++component) {
        gravitySignals[component] = sourceSignal(
            "projected_gravity_" + std::to_string(component),
            TaskObservationSource::projectedGravity,
            {},
            component
        );
        if (component < 2u) {
            gravityHorizontalTerms.push_back(unarySignal(
                gravitySignals[component] + "_squared",
                TaskSignalOperator::square,
                gravitySignals[component]
            ));
        }
    }
    const std::string gravityHorizontalSquared = sumSignals(
        "projected_gravity_horizontal_squared",
        gravityHorizontalTerms
    );
    signalReward(
        gravityHorizontalSquared,
        -5.0f,
        TaskRewardChannel::stability
    );

    const std::string rootHeight = sourceSignal(
        "root_height",
        TaskObservationSource::rootHeight,
        {},
        0u
    );
    const std::string heightTarget = constantSignal(
        "root_height_target",
        task.baseHeightTarget
    );
    const std::string heightError = binarySignal(
        "root_height_error",
        TaskSignalOperator::subtract,
        rootHeight,
        heightTarget
    );
    signalReward(
        unarySignal(
            "root_height_error_squared",
            TaskSignalOperator::square,
            heightError
        ),
        -10.0f,
        TaskRewardChannel::stability
    );

    std::vector<std::string> jointVelocityTerms;
    jointVelocityTerms.reserve(metadata.jointLimits.size());
    for (std::size_t joint = 0u;
         joint < metadata.jointLimits.size();
         ++joint) {
        const std::string leaf = sourceSignal(
            "joint_velocity_" + std::to_string(joint),
            TaskObservationSource::jointVelocity,
            metadata.jointLimits[joint].name,
            0u
        );
        jointVelocityTerms.push_back(unarySignal(
            leaf + "_squared",
            TaskSignalOperator::square,
            leaf
        ));
    }
    signalReward(
        sumSignals("joint_velocity_squared", jointVelocityTerms),
        -0.001f,
        TaskRewardChannel::velocity
    );

    specializedReward(
        TaskRewardOperator::jointAccelerationSquared,
        -2.5e-7f,
        TaskRewardChannel::acceleration
    );
    specializedReward(
        TaskRewardOperator::actionRateSquared,
        -0.05f,
        TaskRewardChannel::control
    );
    specializedReward(
        TaskRewardOperator::jointLimitViolationAbsolute,
        -5.0f,
        TaskRewardChannel::configuration,
        {},
        {0.9f, 0.0f, 0.0f, 0.0f}
    );
    specializedReward(
        TaskRewardOperator::mechanicalPower,
        -2.0e-5f,
        TaskRewardChannel::energy
    );

    const auto postureSignal =
        [&](
            const std::string& id,
            const std::vector<std::string>& joints
        ) {
            std::vector<std::string> terms;
            terms.reserve(joints.size());
            for (std::size_t joint = 0u;
                 joint < joints.size();
                 ++joint) {
                const std::string leaf = sourceSignal(
                    id + "_posture_" +
                        std::to_string(joint),
                    TaskObservationSource::jointPositionError,
                    joints[joint],
                    0u
                );
                terms.push_back(unarySignal(
                    leaf + "_absolute",
                    TaskSignalOperator::absolute,
                    leaf
                ));
            }
            return sumSignals(
                id + "_posture_sum",
                terms
            );
        };
    signalReward(
        postureSignal("waist", waistJoints),
        -1.0f,
        TaskRewardChannel::configuration
    );
    signalReward(
        postureSignal("hips", hipJoints),
        -1.0f,
        TaskRewardChannel::configuration
    );
    signalReward(
        postureSignal("arms", armJoints),
        -0.1f,
        TaskRewardChannel::configuration
    );

    specializedReward(
        TaskRewardOperator::gaitContactMatch,
        0.5f,
        TaskRewardChannel::primary
    );
    specializedReward(
        TaskRewardOperator::footClearance,
        1.0f,
        TaskRewardChannel::primary,
        {},
        {0.05f, 2.0f, 0.0f, 0.0f}
    );
    const std::string leftSlip = sourceSignal(
        "left_support_slip",
        TaskObservationSource::contactMetric,
        "left_foot_contact",
        1u
    );
    const std::string rightSlip = sourceSignal(
        "right_support_slip",
        TaskObservationSource::contactMetric,
        "right_foot_contact",
        1u
    );
    signalReward(
        binarySignal(
            "support_slip",
            TaskSignalOperator::add,
            leftSlip,
            rightSlip
        ),
        -0.2f,
        TaskRewardChannel::contact
    );
    signalReward(
        sourceSignal(
            "forbidden_contact",
            TaskObservationSource::contactMetric,
            "undesired_contact",
            0u
        ),
        -1.0f,
        TaskRewardChannel::contact
    );

    const std::string gravityHorizontal = unarySignal(
        "projected_gravity_horizontal",
        TaskSignalOperator::squareRoot,
        gravityHorizontalSquared
    );
    const std::string negativeOne = constantSignal(
        "negative_one",
        -1.0f
    );
    const std::string negativeGravityZ = binarySignal(
        "negative_projected_gravity_z",
        TaskSignalOperator::multiply,
        gravitySignals[2u],
        negativeOne
    );
    const std::string tiltDenominator = binarySignal(
        "tilt_denominator",
        TaskSignalOperator::maximum,
        negativeGravityZ,
        constantSignal("tilt_epsilon", 1.0e-6f)
    );
    const std::string tiltAngle = binarySignal(
        "tilt_angle",
        TaskSignalOperator::atan2,
        gravityHorizontal,
        tiltDenominator
    );
    task.terminations = {
        {
            .operation = TaskTerminationOperator::signalBelow,
            .signal = rootHeight,
            .reason = MR_TASK_TERMINATION_HEIGHT,
            .priority = 1u,
            .threshold = 0.2f,
            .failurePenalty = -2.0f,
        },
        {
            .operation = TaskTerminationOperator::signalAbove,
            .signal = tiltAngle,
            .reason = MR_TASK_TERMINATION_TILT,
            .priority = 2u,
            .threshold = 0.8f,
            .failurePenalty = -2.0f,
        },
    };

    const auto random =
        [&task](
            const TaskRandomizationOperator operation,
            const std::string_view target,
            const std::uint32_t component,
            const std::uint32_t minimumCurriculumLevel,
            const mr_float4 parameters
        ) {
            task.randomization.push_back({
                .operation = operation,
                .target = std::string{target},
                .component = component,
                .minimumCurriculumLevel =
                    minimumCurriculumLevel,
                .parameters = parameters,
            });
        };
    random(
        TaskRandomizationOperator::rootPosition,
        {},
        0u,
        0u,
        {0.5f, 0.5f, 0.0f, 0.0f}
    );
    random(
        TaskRandomizationOperator::rootYaw,
        {},
        0u,
        0u,
        {-3.14f, 3.14f, 0.0f, 0.0f}
    );
    random(
        TaskRandomizationOperator::actionVelocity,
        {},
        0u,
        2u,
        {-1.0f, 1.0f, 0.0f, 0.0f}
    );
    random(
        TaskRandomizationOperator::bodyParameter,
        "robot",
        1u,
        2u,
        {0.3f, 1.0f, 0.0f, 0.0f}
    );
    random(
        TaskRandomizationOperator::bodyPayload,
        "torso_link",
        0u,
        2u,
        {-1.0f, 3.0f, 0.0f, 0.0f}
    );
    return task;
}

SimulationDescription makeUnitreeG1Simulation(
    const BuiltinSurface surface,
    const G1ActuatorPresetId actuatorPreset
) {
    SimulationDescription world{
        .model = makeUnitreeG1EngineModel(actuatorPreset),
        .task = makeUnitreeG1TaskPack(surface),
    };
    appendBuiltinSurface(
        world.model,
        world.sceneBodies,
        surface
    );
    std::string reason;
    if (!world.model.valid(&reason)) {
        throw std::runtime_error(
            "G1 locomotion world is invalid: " + reason
        );
    }
    return world;
}

} // namespace metalrobo
