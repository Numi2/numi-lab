#include "metalrobo/LocomotionWorld.hpp"

#include "metalrobo/G1.hpp"
#include "metalrobo/WorldPack.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace metalrobo {
namespace {

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

LocomotionWorldCompileDiagnostics compileLocomotionWorld(
    const LocomotionWorld& authored,
    const std::uint32_t articulationIndex,
    CompiledLocomotionWorld& compiled
) {
    LocomotionWorldCompileDiagnostics diagnostics;
    CompiledLocomotionWorld staged;
    diagnostics.world = compileMetalWorld(
        authored.model,
        articulationIndex,
        staged.world,
        authored.task.capacities
    );
    if (!diagnostics.world.succeeded()) {
        return diagnostics;
    }
    diagnostics.task = compileTaskProgram(
        authored.task,
        staged.world,
        staged.task
    );
    if (!diagnostics.task.succeeded()) {
        return diagnostics;
    }
    compiled = std::move(staged);
    return diagnostics;
}

void appendLocomotionSurface(
    EngineModel& model,
    std::vector<MRBodyStateGPU>& sceneBodies,
    const LocomotionSurface surface
) {
    if (model.materials.empty()) {
        throw std::invalid_argument(
            "locomotion surface requires a model material"
        );
    }
    switch (surface) {
    case LocomotionSurface::ground:
        appendGround(model, sceneBodies);
        break;
    case LocomotionSurface::terrain:
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

void appendLocomotionDynamicSpheres(
    LocomotionWorld& world,
    const std::span<const LocomotionDynamicSphere> spheres
) {
    if (spheres.empty()) {
        return;
    }
    if (world.model.materials.empty()) {
        throw std::invalid_argument(
            "dynamic locomotion spheres require a model material"
        );
    }

    MRMaterialGPU ballMaterial = world.model.materials.front();
    ballMaterial.friction = {0.35f, 0.30f, 0.0f, 0.0f};
    ballMaterial.response = {0.35f, 0.25f, 0.0f, 0.0f};
    const std::uint32_t materialIndex =
        static_cast<std::uint32_t>(world.model.materials.size());
    world.model.materials.push_back(ballMaterial);

    for (std::size_t index = 0u; index < spheres.size(); ++index) {
        const LocomotionDynamicSphere& source = spheres[index];
        if (!std::isfinite(source.position.x) ||
            !std::isfinite(source.position.y) ||
            !std::isfinite(source.position.z) ||
            !std::isfinite(source.linearVelocity.x) ||
            !std::isfinite(source.linearVelocity.y) ||
            !std::isfinite(source.linearVelocity.z) ||
            !std::isfinite(source.radius) ||
            !std::isfinite(source.mass) ||
            !(source.radius > 0.0f) ||
            !(source.mass > 0.0f) ||
            source.launchStep >
                (MR_BODY_STATE_LAUNCH_STEP_MASK >>
                 MR_BODY_STATE_LAUNCH_STEP_SHIFT)) {
            throw std::invalid_argument(
                "dynamic locomotion sphere parameters must be finite and positive"
            );
        }

        const std::uint32_t bodyIndex =
            static_cast<std::uint32_t>(world.model.bodies.size());
        const float inertia =
            0.4f * source.mass * source.radius * source.radius;
        const float inverseInertia = 1.0f / inertia;
        MRBodyPropertiesGPU body{};
        body.articulationIndex = MR_INVALID_INDEX;
        body.parentBody = MR_INVALID_INDEX;
        body.inboundJoint = MR_INVALID_INDEX;
        body.motionType = MR_MOTION_DYNAMIC;
        body.massAndInverseMass = {
            source.mass, 1.0f / source.mass, 0.0f, 0.0f,
        };
        body.inertiaRow0 = {inertia, 0.0f, 0.0f, 0.0f};
        body.inertiaRow1 = {0.0f, inertia, 0.0f, 0.0f};
        body.inertiaRow2 = {0.0f, 0.0f, inertia, 0.0f};
        body.inverseInertiaRow0 = {
            inverseInertia, 0.0f, 0.0f, 0.0f,
        };
        body.inverseInertiaRow1 = {
            0.0f, inverseInertia, 0.0f, 0.0f,
        };
        body.inverseInertiaRow2 = {
            0.0f, 0.0f, inverseInertia, 0.0f,
        };
        body.dampingAndSpeedLimits = {
            0.01f, 0.01f, 100.0f, 100.0f,
        };
        world.model.bodies.push_back(body);
        if (!world.model.bodyNames.empty()) {
            world.model.bodyNames.push_back(
                "locomotion_dynamic_sphere_" +
                std::to_string(index)
            );
        }

        MRShapeGPU shape{};
        shape.bodyIndex = bodyIndex;
        shape.shapeType = MR_SHAPE_SPHERE;
        shape.materialIndex = materialIndex;
        shape.collisionGroup = 1u;
        shape.collisionMask = ~0u;
        shape.slotGeneration =
            static_cast<std::uint32_t>(index + 1u);
        shape.localPosition.w = 1.0f;
        shape.localRotation.w = 1.0f;
        shape.dimensions = {source.radius, 0.0f, 0.0f, 0.0f};
        shape.contactRestAndBoundingRadius = {
            0.002f, 0.0f, source.radius, 0.0f,
        };
        world.model.shapes.push_back(shape);
        if (!world.model.shapeNames.empty()) {
            world.model.shapeNames.push_back(
                "locomotion_dynamic_sphere_" +
                std::to_string(index) + "/collision"
            );
        }

        MRBodyStateGPU state{};
        state.position = {
            source.position.x,
            source.position.y,
            source.position.z,
            1.0f,
        };
        state.orientation.w = 1.0f;
        state.linearVelocityAndInverseMass = {
            source.linearVelocity.x,
            source.linearVelocity.y,
            source.linearVelocity.z,
            1.0f / source.mass,
        };
        state.inverseInertiaWorldRow0 = body.inverseInertiaRow0;
        state.inverseInertiaWorldRow1 = body.inverseInertiaRow1;
        state.inverseInertiaWorldRow2 = body.inverseInertiaRow2;
        state.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
        state.flagsAndIndices[1] = MR_INVALID_INDEX;
        state.flagsAndIndices[2] = bodyIndex;
        state.flagsAndIndices[3] =
            MR_BODY_STATE_PRESERVE_RESET_VELOCITY |
            (source.launchStep <<
             MR_BODY_STATE_LAUNCH_STEP_SHIFT);
        world.sceneBodies.push_back(state);
    }

    world.model.world.bodyCount =
        static_cast<std::uint32_t>(world.model.bodies.size());
    world.model.world.shapeCount =
        static_cast<std::uint32_t>(world.model.shapes.size());
    world.model.world.materialCount =
        static_cast<std::uint32_t>(world.model.materials.size());
}

LocomotionWorld makeWorldPackLocomotionWorld(
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

    LocomotionWorld result;
    result.model = authored.engineModel;
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
    return result;
}

TaskPack makeUnitreeG1LocomotionTaskPack(
    const LocomotionSurface surface
) {
    const G1ModelMetadata& metadata = unitreeG1Metadata();
    TaskPack task;
    task.id = "unitree_g1_mjlab_velocity";
    // The topology envelope contains every eligible self-collision pair.
    // Locomotion instead compiles an explicit operational arena. Capacity
    // overflow is transactional and reports the exact required stage count,
    // so this is a replayable task contract rather than silent truncation.
    task.capacities = {
        .candidatePairs = 128u,
        .rawContacts = 128u,
        .manifolds = 32u,
        .constraintBlocks = 64u,
        .constraintRows = 192u,
        .hardConvexPairs = 64u,
        .meshTriangleCandidates = 1024u,
        .ccdCandidates = 64u,
        .ccdEvents = 8u,
        .endpointRuntimeRecords = 128u,
        .articulationPointQueries = 128u,
        .qualityRows = 192u,
        .islandConstraintReferences = 64u,
    };
    task.actorHistoryLength = 1u;
    task.criticHistoryLength = 1u;
    task.criticIncludesCleanHistory = false;
    task.maximumEpisodeSteps = 1000u;
    task.maximumActionDelaySteps = 0u;
    task.maximumObservationDelaySteps = 0u;
    task.curriculumLevelCount = 11u;
    task.baseHeightTarget = 0.78f;
    task.gaitPeriodSeconds = 0.6f;
    task.clearanceTarget = 0.10f;
    task.successTrackingThreshold = 0.8f;
    task.supportForceThreshold = 1.0f;
    // Level zero is a true standing task. Translational and yaw commands are
    // introduced together only after the policy has demonstrated full-episode
    // survival, rather than asking a noisy initial policy to discover balance
    // and locomotion simultaneously.
    task.commands.lower = {0.0f, 0.0f, 0.0f, 0.0f};
    task.commands.upper = {0.0f, 0.0f, 0.0f, 0.0f};
    task.commands.limitLower = {-0.5f, -0.5f, -1.0f, 0.0f};
    task.commands.limitUpper = {
        1.0f, 0.5f, 1.0f, 0.0f,
    };
    task.commands.curriculumStep = {
        0.1f, 0.1f, 0.1f, 0.0f,
    };
    task.commands.standingProbability = 1.0f;
    task.commands.minimumEpisodeSurvivalFraction = 0.8f;
    task.commands.minimumDurationSeconds = 10.0f;
    task.commands.maximumDurationSeconds = 10.0f;
    task.pushes.maximumVelocity = 0.0f;
    task.pushes.minimumIntervalSeconds = 5.0f;
    task.pushes.maximumIntervalSeconds = 5.0f;

    constexpr std::array<float, kUnitreeG1JointCount> actionScales{{
        0.55f, 0.35f, 0.55f, 0.35f, 0.44f, 0.44f,
        0.55f, 0.35f, 0.55f, 0.35f, 0.44f, 0.44f,
        0.55f, 0.44f, 0.44f,
        0.44f, 0.44f, 0.44f, 0.44f, 0.44f, 0.07f, 0.07f,
        0.44f, 0.44f, 0.44f, 0.44f, 0.44f, 0.07f, 0.07f,
    }};
    task.actions.reserve(metadata.jointLimits.size());
    for (std::size_t index = 0u;
         index < metadata.jointLimits.size();
         ++index) {
        const G1JointLimit& joint = metadata.jointLimits[index];
        task.actions.push_back({
            .joint = std::string{joint.name},
            .scale = actionScales[index],
            .responseTimeSeconds = 0.0f,
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
            component
        ));
    }
    for (std::uint32_t component = 0u;
         component < 3u;
         ++component) {
        task.actorFrame.push_back(observation(
            TaskObservationSource::projectedGravity,
            {},
            component
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
    for (std::uint32_t component = 0u;
         component < 2u;
         ++component) {
        task.actorFrame.push_back(observation(
            TaskObservationSource::gaitPhase,
            {},
            component
        ));
    }
    for (const G1JointLimit& joint : metadata.jointLimits) {
        task.actorFrame.push_back(observation(
            TaskObservationSource::jointPositionError,
            joint.name,
            0u
        ));
    }
    for (const G1JointLimit& joint : metadata.jointLimits) {
        task.actorFrame.push_back(observation(
            TaskObservationSource::jointVelocity,
            joint.name,
            0u
        ));
    }
    for (const G1JointLimit& joint : metadata.jointLimits) {
        task.actorFrame.push_back(observation(
            TaskObservationSource::previousAction,
            joint.name,
            0u
        ));
    }
    task.critic = task.actorFrame;

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
    task.jointGroups.push_back({
        .id = "waist",
        .joints = jointNames({12u, 13u, 14u}),
    });
    task.jointGroups.push_back({
        .id = "hips",
        .joints = jointNames({1u, 2u, 7u, 8u}),
    });
    std::vector<std::string> armJoints;
    for (std::uint32_t joint = 15u; joint < 29u; ++joint) {
        armJoints.emplace_back(
            metadata.jointLimits[joint].name
        );
    }
    task.jointGroups.push_back({
        .id = "arms",
        .joints = std::move(armJoints),
    });

    task.terrain.body =
        surface == LocomotionSurface::terrain
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
    if (surface == LocomotionSurface::terrain) {
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
    const auto reward =
        [&task](
            const TaskRewardOperator operation,
            const float weight,
            const std::string_view group = {},
            const mr_float4 parameters = {}
        ) {
            task.rewards.push_back({
                .operation = operation,
                .sourceGroup = std::string{group},
                .weight = weight,
                .parameters = parameters,
            });
        };
    reward(
        TaskRewardOperator::linearVelocityTracking,
        1.0f,
        {},
        {0.25f, 0.0f, 0.0f, 0.0f}
    );
    reward(
        TaskRewardOperator::yawVelocityTracking,
        0.5f,
        {},
        {0.25f, 0.0f, 0.0f, 0.0f}
    );
    reward(TaskRewardOperator::constant, 0.15f);
    reward(
        TaskRewardOperator::rootVerticalVelocitySquared,
        -2.0f
    );
    reward(
        TaskRewardOperator::rootRollPitchVelocitySquared,
        -0.05f
    );
    reward(
        TaskRewardOperator::projectedGravityHorizontalSquared,
        -5.0f
    );
    reward(
        TaskRewardOperator::rootHeightErrorSquared,
        -10.0f
    );
    reward(
        TaskRewardOperator::jointVelocitySquared,
        -0.001f
    );
    reward(
        TaskRewardOperator::jointAccelerationSquared,
        -2.5e-7f
    );
    reward(
        TaskRewardOperator::actionRateSquared,
        -0.05f
    );
    reward(
        TaskRewardOperator::jointLimitViolationAbsolute,
        -5.0f,
        {},
        {0.9f, 0.0f, 0.0f, 0.0f}
    );
    reward(
        TaskRewardOperator::mechanicalPower,
        -2.0e-5f
    );
    reward(
        TaskRewardOperator::jointGroupPostureAbsolute,
        -1.0f,
        "waist"
    );
    reward(
        TaskRewardOperator::jointGroupPostureAbsolute,
        -1.0f,
        "hips"
    );
    reward(
        TaskRewardOperator::jointGroupPostureAbsolute,
        -0.1f,
        "arms"
    );
    reward(
        TaskRewardOperator::gaitContactMatch,
        0.5f
    );
    reward(
        TaskRewardOperator::footClearance,
        1.0f,
        {},
        {0.05f, 2.0f, 0.0f, 0.0f}
    );
    reward(TaskRewardOperator::supportSlip, -0.2f);
    reward(
        TaskRewardOperator::forbiddenContact,
        -1.0f,
        "undesired_contact"
    );

    task.terminations = {
        {
            .operation =
                TaskTerminationOperator::minimumRootHeight,
            .reason = MR_TASK_TERMINATION_HEIGHT,
            .priority = 1u,
            .threshold = 0.2f,
            .failurePenalty = -2.0f,
        },
        {
            .operation = TaskTerminationOperator::maximumTilt,
            .reason = MR_TASK_TERMINATION_TILT,
            .priority = 2u,
            .threshold = 1.22173048f,
            .failurePenalty = -2.0f,
        },
    };

    return task;
}

TaskPack makeUnitreeG1DisturbanceRecoveryTaskPack(
    const LocomotionSurface surface
) {
    TaskPack task = makeUnitreeG1LocomotionTaskPack(surface);
    task.id = "unitree_g1_disturbance_recovery";

    // This skill owns balance and recovery, not commanded locomotion. Keeping
    // the actor contract unchanged permits exact initialization from the
    // official Unitree velocity actor while the critic learns the new task.
    task.commands.lower = {};
    task.commands.upper = {};
    task.commands.limitLower = {};
    task.commands.limitUpper = {};
    task.commands.curriculumStep = {};
    task.commands.standingProbability = 1.0f;
    task.commands.minimumEpisodeSurvivalFraction = 0.8f;
    task.commands.minimumDurationSeconds = 10.0f;
    task.commands.maximumDurationSeconds = 10.0f;

    // The runtime scales the maximum impulse velocity by curriculum level.
    // Level zero therefore proves quiet standing before disturbances ramp to
    // 2.5 m/s from deterministic, replayable horizontal directions.
    task.curriculumLevelCount = 11u;
    task.pushes.maximumVelocity = 2.5f;
    task.pushes.minimumIntervalSeconds = 1.5f;
    task.pushes.maximumIntervalSeconds = 3.0f;
    task.maximumActionDelaySteps = 2u;
    task.randomization = {
        {
            .operation = TaskRandomizationOperator::rootPosition,
            .minimumCurriculumLevel = 2u,
            .parameters = {0.02f, 0.02f, 0.01f, 0.0f},
        },
        {
            .operation = TaskRandomizationOperator::actionPosition,
            .minimumCurriculumLevel = 3u,
            .parameters = {-0.03f, 0.03f, 0.0f, 0.0f},
        },
        {
            .operation = TaskRandomizationOperator::actionVelocity,
            .minimumCurriculumLevel = 4u,
            .parameters = {-0.10f, 0.10f, 0.0f, 0.0f},
        },
        {
            .operation = TaskRandomizationOperator::bodyParameter,
            .target = "robot",
            .component = 0u,
            .minimumCurriculumLevel = 5u,
            .parameters = {0.9f, 1.1f, 0.0f, 0.0f},
        },
        {
            .operation = TaskRandomizationOperator::controllerParameter,
            .component = 0u,
            .minimumCurriculumLevel = 6u,
            .parameters = {0.9f, 1.1f, 0.0f, 0.0f},
        },
        {
            .operation = TaskRandomizationOperator::controllerParameter,
            .component = 1u,
            .minimumCurriculumLevel = 7u,
            .parameters = {0.9f, 1.1f, 0.0f, 0.0f},
        },
        {
            .operation = TaskRandomizationOperator::actionDelay,
            .minimumCurriculumLevel = 8u,
            .parameters = {0.0f, 2.0f, 0.0f, 0.0f},
        },
    };
    return task;
}

TaskPack makeUnitreeG1SupineGetUpDiscoveryTaskPack(
    const LocomotionSurface surface
) {
    const G1ModelMetadata& metadata = unitreeG1Metadata();
    TaskPack task = makeUnitreeG1LocomotionTaskPack(surface);
    task.id = "unitree_g1_supine_get_up_discovery";
    task.maximumEpisodeSteps = 500u;
    task.curriculumLevelCount = 1u;
    task.commands.lower = {};
    task.commands.upper = {};
    task.commands.limitLower = {};
    task.commands.limitUpper = {};
    task.commands.curriculumStep = {};
    task.commands.standingProbability = 1.0f;
    task.pushes.maximumVelocity = 0.0f;
    task.pushes.minimumIntervalSeconds = 10.0f;
    task.pushes.maximumIntervalSeconds = 10.0f;
    task.terminations.clear();

    // HumanUP's successful Stage-I actor uses ten meaningful proprioceptive
    // frames rather than exposing command and gait-phase slots that are
    // constant in a get-up task. The critic additionally sees translational
    // state, height, and bilateral support contact.
    std::vector<TaskObservationOperatorSpec> actorFrame;
    actorFrame.reserve(92u);
    actorFrame.insert(
        actorFrame.end(),
        task.actorFrame.begin(),
        task.actorFrame.begin() + 5
    );
    actorFrame.insert(
        actorFrame.end(),
        task.actorFrame.begin() + 11,
        task.actorFrame.end()
    );
    task.actorFrame = std::move(actorFrame);
    task.critic = task.actorFrame;
    const std::vector<TaskObservationOperatorSpec> privilegedState{
        {
            .source = TaskObservationSource::rootLinearVelocityLocal,
            .component = 0u,
        },
        {
            .source = TaskObservationSource::rootLinearVelocityLocal,
            .component = 1u,
        },
        {
            .source = TaskObservationSource::rootLinearVelocityLocal,
            .component = 2u,
        },
        {
            .source = TaskObservationSource::rootHeight,
            .scale = 2.0f,
        },
        {
            .source = TaskObservationSource::contactMetric,
            .target = "left_foot_contact",
            .component = 0u,
            .scale = 0.01f,
        },
        {
            .source = TaskObservationSource::contactMetric,
            .target = "right_foot_contact",
            .component = 0u,
            .scale = 0.01f,
        },
    };
    task.critic.insert(
        task.critic.end(),
        privilegedState.begin(),
        privilegedState.end()
    );
    task.actorHistoryLength = 10u;
    task.criticHistoryLength = 10u;

    task.contactGroups.push_back({
        .id = "torso_reference",
        .bodies = {"torso_link"},
        .referenceBody = "torso_link",
        // G1 authored body positions are centers of mass. This reference is
        // the torso link-frame origin used by the source task's head/upper-
        // body height shaping.
        .localReference = {
            -0.0020313345f, -0.00033972675f,
            -0.18459715f, 0.0f,
        },
    });

    task.rewards = {
        {
            .operation = TaskRewardOperator::rootHeightNormalized,
            .weight = 5.0f,
        },
        {
            .operation = TaskRewardOperator::bodyHeightExponential,
            .sourceGroup = "torso_reference",
            .weight = 5.0f,
            .parameters = {1.0f, 0.0f, 0.0f, 0.0f},
        },
        {
            .operation = TaskRewardOperator::rootHeightProgress,
            .weight = 1.0f,
        },
        {
            .operation = TaskRewardOperator::bodyUpExponential,
            .weight = 0.25f,
        },
        {
            .operation = TaskRewardOperator::supportContactCount,
            .weight = 2.5f,
        },
        {
            .operation = TaskRewardOperator::supportHeightExponential,
            .weight = 2.5f,
            .parameters = {10.0f, 0.0f, 0.0f, 0.0f},
        },
        {
            .operation = TaskRewardOperator::standingCompletion,
            .weight = 10.0f,
            .parameters = {0.65f, 0.8f, 0.0f, 0.0f},
        },
        {
            .operation = TaskRewardOperator::jointVelocitySquared,
            .weight = -1.0e-4f,
        },
        {
            .operation = TaskRewardOperator::jointAccelerationSquared,
            .weight = -1.0e-8f,
        },
        {
            .operation = TaskRewardOperator::actionRateSquared,
            .weight = -0.01f,
        },
        {
            .operation = TaskRewardOperator::jointLimitViolationAbsolute,
            .weight = -5.0f,
            .parameters = {0.95f, 0.0f, 0.0f, 0.0f},
        },
        {
            .operation = TaskRewardOperator::mechanicalPower,
            .weight = -1.0e-6f,
        },
    };

    // Canonical supine seed adapted from HumanUP's public 23-DoF G1 pose.
    // The two three-axis wrists absent from that model retain zero/default.
    constexpr std::array<float, kUnitreeG1JointCount> supine{{
        -0.3600f, 0.2481f, 1.6115f, -0.0647f, -0.8612f, -0.1226f,
        -0.3878f, 0.3584f, 1.5328f, 0.1519f, -0.8651f, 0.2362f,
        -0.0357f, 0.0685f, -0.5200f,
        0.4665f, 0.8218f, 0.4253f, 1.2972f, 0.0f, 0.0f, 0.0f,
        0.1429f, -1.0324f, -0.4241f, 1.4075f, 0.0f, 0.0f, 0.0f,
    }};
    task.randomization = {
        {
            .operation = TaskRandomizationOperator::rootOrientation,
            .parameters = {
                -0.0098234f, 0.49986f, 0.017525f, -0.86587f,
            },
        },
        {
            .operation = TaskRandomizationOperator::rootHeight,
            .parameters = {
                0.01850436f, 0.01850436f, 0.0f, 0.0f,
            },
        },
    };
    for (std::size_t index = 0u; index < supine.size(); ++index) {
        task.randomization.push_back({
            .operation = TaskRandomizationOperator::jointPosition,
            .target = std::string{metadata.jointLimits[index].name},
            .parameters = {
                supine[index], supine[index], 0.0f, 0.0f,
            },
        });
    }
    return task;
}

TaskPack makeUnitreeG1BallDisturbanceRecoveryTaskPack(
    const LocomotionSurface surface
) {
    const G1ModelMetadata& metadata = unitreeG1Metadata();
    TaskPack task = makeUnitreeG1DisturbanceRecoveryTaskPack(surface);
    task.id = "unitree_g1_ball_disturbance_recovery";
    task.pushes.maximumVelocity = 0.0f;
    // This stage learns pre-fall stability. Promotion is based on completing
    // the episode while remaining stationary, not on recovery completion;
    // recovery events below are shaping signals only.
    task.successTrackingThreshold = 0.70f;
    task.capacities.candidatePairs = 256u;
    task.capacities.rawContacts = 256u;
    task.capacities.manifolds = 64u;
    task.capacities.constraintBlocks = 128u;
    task.capacities.constraintRows = 384u;
    task.capacities.endpointRuntimeRecords = 256u;
    task.capacities.articulationPointQueries = 256u;
    task.capacities.qualityRows = 384u;
    task.capacities.islandConstraintReferences = 128u;

    // Solver-derived contact wrench is the native touch signal for impacts.
    // G1's dense tactile atlases are plantar-only, so a separate semantic
    // whole-body group detects ball contact without inventing skin sensors.
    std::vector<std::string> impactBodies;
    impactBodies.reserve(metadata.bodyNames.size() - 2u);
    for (std::uint32_t body = 0u;
         body < metadata.bodyNames.size();
         ++body) {
        if (body != 6u && body != 12u) {
            impactBodies.emplace_back(metadata.bodyNames[body]);
        }
    }
    task.contactGroups.push_back({
        .id = "impact_contact",
        .bodies = std::move(impactBodies),
        .referenceBody = "torso_link",
    });

    // The actor consumes the compact object-track contract produced by the
    // native simulation sensor and by the deployment RGB-D perception
    // provider. Exact recovery-event labels remain critic-only below.
    for (std::uint32_t sphere = 0u; sphere < 6u; ++sphere) {
        const std::string name =
            "locomotion_dynamic_sphere_" + std::to_string(sphere);
        const TaskObservationOperatorSpec confidence{
            .source = TaskObservationSource::objectTrack,
            .target = name,
            .component = 0u,
        };
        task.actorFrame.push_back(confidence);
        task.critic.push_back(confidence);
        for (std::uint32_t component = 1u;
             component < 7u;
             ++component) {
            const bool position = component <= 3u;
            const TaskObservationOperatorSpec track{
                .source = TaskObservationSource::objectTrack,
                .target = name,
                .component = component,
                .scale = position ? 0.5f : 0.2f,
                .noiseAmplitude = position ? 0.01f : 0.02f,
            };
            task.actorFrame.push_back(track);
            task.critic.push_back(track);
        }
    }

    // A launched ball touching the robot is not a policy failure. Falling is
    // still penalized by height/tilt termination, while the event rewards
    // measure what happens after accepted physical contact.
    std::erase_if(
        task.rewards,
        [](const TaskRewardOperatorSpec& reward) {
            return reward.operation ==
                TaskRewardOperator::forbiddenContact;
        }
    );

    // The asymmetric critic additionally receives physical root state plus
    // native recovery-event state so it can value impact consequences without
    // leaking privileged event labels into deployment.
    const std::array<TaskObservationOperatorSpec, 8> recoveryCritic{
        TaskObservationOperatorSpec{
            .source = TaskObservationSource::rootLinearVelocityLocal,
            .component = 0u,
        },
        TaskObservationOperatorSpec{
            .source = TaskObservationSource::rootLinearVelocityLocal,
            .component = 1u,
        },
        TaskObservationOperatorSpec{
            .source = TaskObservationSource::rootLinearVelocityLocal,
            .component = 2u,
        },
        TaskObservationOperatorSpec{
            .source = TaskObservationSource::rootHeight,
            .scale = 2.0f,
        },
        TaskObservationOperatorSpec{
            .source = TaskObservationSource::recoveryEvent,
            .component = 0u,
        },
        TaskObservationOperatorSpec{
            .source = TaskObservationSource::recoveryEvent,
            .component = 1u,
        },
        TaskObservationOperatorSpec{
            .source = TaskObservationSource::recoveryEvent,
            .component = 2u,
        },
        TaskObservationOperatorSpec{
            .source = TaskObservationSource::recoveryEvent,
            .component = 3u,
        },
    };
    task.critic.insert(
        task.critic.end(),
        recoveryCritic.begin(),
        recoveryCritic.end()
    );
    constexpr mr_float4 recoveryDefinition{
        0.04f, 0.02f, 0.30f, 0.0f,
    };
    task.rewards.push_back({
        .operation = TaskRewardOperator::recoveryTiltProgress,
        .sourceGroup = "impact_contact",
        .weight = 3.0f,
        .parameters = recoveryDefinition,
    });
    task.rewards.push_back({
        .operation = TaskRewardOperator::recoveryCompletion,
        .sourceGroup = "impact_contact",
        .weight = 1.0f,
        .parameters = recoveryDefinition,
    });

    constexpr std::array<std::array<float, 3>, 6> positionLower{{
        {{-1.7f, -0.05f, 0.75f}},
        {{ 1.3f, -0.05f, 0.75f}},
        {{-0.05f, -1.7f, 0.75f}},
        {{-0.05f,  1.3f, 0.75f}},
        {{-1.7f,  0.20f, 0.75f}},
        {{ 1.3f, -0.30f, 0.75f}},
    }};
    constexpr std::array<std::array<float, 3>, 6> positionUpper{{
        {{-1.3f,  0.05f, 1.25f}},
        {{ 1.7f,  0.05f, 1.25f}},
        {{ 0.05f, -1.3f, 1.25f}},
        {{ 0.05f,  1.7f, 1.25f}},
        {{-1.3f,  0.30f, 1.25f}},
        {{ 1.7f, -0.20f, 1.25f}},
    }};
    constexpr std::array<std::array<float, 3>, 6> velocityLower{{
        {{ 2.0f, -0.05f, 1.5f}},
        {{-3.0f, -0.05f, 1.5f}},
        {{-0.05f,  2.0f, 1.5f}},
        {{-0.05f, -3.0f, 1.5f}},
        {{ 2.0f, -0.05f, 1.5f}},
        {{-3.0f, -0.05f, 1.5f}},
    }};
    constexpr std::array<std::array<float, 3>, 6> velocityUpper{{
        {{ 3.0f,  0.05f, 3.5f}},
        {{-2.0f,  0.05f, 3.5f}},
        {{ 0.05f,  3.0f, 3.5f}},
        {{ 0.05f, -2.0f, 3.5f}},
        {{ 3.0f,  0.05f, 3.5f}},
        {{-2.0f,  0.05f, 3.5f}},
    }};
    constexpr std::array<std::array<float, 2>, 6> launchSteps{{
        {{75.0f, 125.0f}},
        {{175.0f, 225.0f}},
        {{275.0f, 325.0f}},
        {{375.0f, 425.0f}},
        {{475.0f, 525.0f}},
        {{575.0f, 625.0f}},
    }};
    for (std::uint32_t sphere = 0u; sphere < 6u; ++sphere) {
        const std::string name =
            "locomotion_dynamic_sphere_" + std::to_string(sphere);
        for (std::uint32_t component = 0u; component < 3u; ++component) {
            task.randomization.push_back({
                .operation = TaskRandomizationOperator::sceneBodyPosition,
                .target = name,
                .component = component,
                .parameters = {
                    positionLower[sphere][component],
                    positionUpper[sphere][component], 0.0f, 0.0f,
                },
            });
            task.randomization.push_back({
                .operation = TaskRandomizationOperator::sceneBodyVelocity,
                .target = name,
                .component = component,
                .parameters = {
                    velocityLower[sphere][component],
                    velocityUpper[sphere][component], 0.0f, 0.0f,
                },
            });
        }
        const std::uint32_t impactAxis =
            sphere == 2u || sphere == 3u ? 1u : 0u;
        const float direction =
            sphere == 0u || sphere == 2u || sphere == 4u
            ? 1.0f
            : -1.0f;
        for (std::uint32_t level = 1u; level <= 3u; ++level) {
            const float speedLower = 2.0f + float(level);
            const float speedUpper = 3.0f + float(level);
            task.randomization.push_back({
                .operation =
                    TaskRandomizationOperator::sceneBodyVelocity,
                .target = name,
                .component = impactAxis,
                .minimumCurriculumLevel = level,
                .parameters = direction > 0.0f
                    ? mr_float4{
                          speedLower, speedUpper, 0.0f, 0.0f,
                      }
                    : mr_float4{
                          -speedUpper, -speedLower, 0.0f, 0.0f,
                      },
            });
        }
        task.randomization.push_back({
            .operation = TaskRandomizationOperator::sceneBodyLaunchStep,
            .target = name,
            .parameters = {
                launchSteps[sphere][0], launchSteps[sphere][1],
                0.0f, 0.0f,
            },
        });
        task.randomization.push_back({
            .operation =
                TaskRandomizationOperator::sceneBodyEventImpact,
            .target = name,
            .component = sphere,
            .minimumCurriculumLevel = 1u,
            .parameters = {
                0.05f, 0.50f, 2.0f, 0.70f,
            },
        });
    }
    return task;
}

TaskPack makeUnitreeG1BallDodgeTaskPack(
    const LocomotionSurface surface
) {
    TaskPack task =
        makeUnitreeG1BallDisturbanceRecoveryTaskPack(surface);
    task.id = "unitree_g1_ball_dodge";
    task.curriculumLevelCount = 9u;
    task.projectileOutcomeCurriculum = true;
    task.successTrackingThreshold = 0.55f;
    task.commands.minimumEpisodeSurvivalFraction = 0.35f;
    task.pushes.projectileStandingProbability = 0.20f;
    task.pushes.projectileTargetHorizontalRadius = 0.40f;
    task.pushes.projectileHorizontalSpeedLower = 2.5f;
    task.pushes.projectileHorizontalSpeedUpper = 5.5f;
    task.pushes.projectileTargetHeightLower = 0.45f;
    task.pushes.projectileTargetHeightUpper = 1.35f;

    // Deployment sees only ball-segmentation-masked depth. Preserve exact
    // object tracks in the critic for asymmetric PPO, but remove them from
    // the actor so privileged state cannot leak into the deployed policy.
    std::erase_if(
        task.actorFrame,
        [](const TaskObservationOperatorSpec& observation) {
            return observation.source ==
                    TaskObservationSource::objectTrack ||
                observation.source ==
                    TaskObservationSource::gaitPhase;
        }
    );
    // Preserve the official G1 actor's exact five-frame, 96-value
    // proprioceptive prefix. Masked depth is appended below as a direct
    // device-observation suffix with its own sparse temporal offsets.
    task.actorHistoryLength = 5u;
    for (TaskObservationOperatorSpec& observation : task.actorFrame) {
        switch (observation.source) {
        case TaskObservationSource::rootAngularVelocityLocal:
            observation.scale = 0.2f;
            break;
        case TaskObservationSource::jointVelocity:
            observation.scale = 0.05f;
            break;
        case TaskObservationSource::projectedGravity:
            observation.normalizeVector3 = true;
            break;
        default:
            break;
        }
    }
    task.visual = {
        .width = 16u,
        .height = 9u,
        .frameOffsets = {0u, 3u, 8u, 18u},
        .nearDepthMeters = 0.1f,
        .farDepthMeters = 5.0f,
        .fullDropoutProbability = 0.02f,
        .pixelDropoutProbability = 0.10f,
        .depthJitterMeters = 0.15f,
        .depthNoiseSigmaMeters = 0.03f,
        .edgeFlickerProbability = 0.15f,
        .curriculumCorruptionGain = 1.0f,
    };
    const std::uint32_t visualPixels =
        task.visual.width * task.visual.height;
    for (std::uint32_t frame = 0u;
         frame < task.visual.frameOffsets.size();
         ++frame) {
        for (std::uint32_t pixel = 0u;
             pixel < visualPixels;
             ++pixel) {
            task.actorFrame.push_back({
                .source = TaskObservationSource::maskedDepth,
                .component = frame * visualPixels + pixel,
            });
        }
    }

    // Dodge learning terminates on contact instead of asking the same actor
    // to absorb the impact. The proprioceptive prefix stays compatible with
    // the stable actor while masked depth extends its first layer.
    std::erase_if(
        task.rewards,
        [](const TaskRewardOperatorSpec& reward) {
            return reward.operation ==
                    TaskRewardOperator::recoveryTiltProgress ||
                reward.operation ==
                    TaskRewardOperator::recoveryCompletion ||
                reward.operation ==
                    TaskRewardOperator::linearVelocityTracking ||
                reward.operation ==
                    TaskRewardOperator::yawVelocityTracking;
        }
    );
    for (TaskRewardOperatorSpec& reward : task.rewards) {
        if (reward.operation == TaskRewardOperator::actionRateSquared) {
            reward.weight = -0.005f;
        }
    }
    task.rewards.push_back({
        .operation = TaskRewardOperator::projectileSafeStillness,
        .weight = 0.5f,
        .parameters = {2.0f, 0.0f, 0.0f, 0.0f},
    });
    task.rewards.push_back({
        .operation = TaskRewardOperator::projectileSafeActionRate,
        .weight = -0.05f,
    });
    for (std::uint32_t sphere = 0u; sphere < 6u; ++sphere) {
        const std::string projectile =
            "locomotion_dynamic_sphere_" + std::to_string(sphere);
        task.rewards.push_back({
            .operation = TaskRewardOperator::projectileEvasion,
            .target = projectile,
            .weight = 1.0f,
            // Distance scale, horizontal-speed scale, position blend.
            .parameters = {0.3f, 1.0f, 0.9f, 0.0f},
        });
        task.rewards.push_back({
            .operation = TaskRewardOperator::linkClearanceBarrier,
            .sourceGroup = "robot",
            .target = projectile,
            .weight = 0.27f,
            // alpha, margin beyond compiled collision envelopes, clip.
            .parameters = {1.0f, 0.05f, 2.0f, 0.0f},
        });
    }
    task.rewards.push_back({
        .operation = TaskRewardOperator::projectileMiss,
        .weight = 1.0f,
    });
    task.threat = {
        .protectedGroup = "robot",
        .activationSpeed = 0.5f,
        .horizonSeconds = 2.0f,
        .safetyMargin = 0.05f,
        .cbfAlpha = 2.0f,
        .stepOverMaximumHeight = 0.35f,
        .sidestepMaximumHeight = 0.75f,
        .leanMaximumHeight = 1.10f,
        .urgencySeconds = 0.35f,
        .desiredVelocityHorizonSeconds = 0.20f,
        .projectionEpsilon = 1.0e-5f,
    };
    task.motion = {
        .anchorBody = "torso_link",
        .trackedBodies = {
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        },
    };
    task.rewards.push_back({
        .operation = TaskRewardOperator::jointCbfCorrection,
        .weight = -0.08f,
    });
    task.rewards.push_back({
        .operation = TaskRewardOperator::jointCbfBuffer,
        .weight = -0.20f,
    });
    task.terminations.push_back({
        .operation = TaskTerminationOperator::projectileContact,
        .sourceGroup = "robot",
        .reason = MR_TASK_TERMINATION_PROJECTILE_CONTACT,
        .priority = 3u,
        .threshold = 5.0f,
        .failurePenalty = -2.0f,
    });

    // A dodge throw is timed, not recovery-gated. A permissive tilt/height
    // gate and one-control-step dwell prevent the hopping exploit caused by
    // waiting for both feet or a settled pose before every launch.
    for (TaskRandomizationOperatorSpec& random : task.randomization) {
        // The promoted visual actor was trained with the complete balance
        // domain distribution. Preserve that distribution at level zero;
        // progressive difficulty is owned by visual corruption and outcome
        // promotion, not by removing actuator/state variability the parent
        // policy relies on.
        random.minimumCurriculumLevel = 0u;
        if (random.operation ==
                TaskRandomizationOperator::sceneBodyEventImpact) {
            random.parameters = {
                3.14159265f, 0.02f, 2.0f, 0.10f,
            };
        }
    }
    return task;
}

LocomotionWorld makeUnitreeG1LocomotionWorld(
    const LocomotionSurface surface,
    const UnitreeG1Task task
) {
    LocomotionWorld world{
        .model = makeUnitreeG1EngineModel(),
        .task = task == UnitreeG1Task::disturbanceRecovery
            ? makeUnitreeG1DisturbanceRecoveryTaskPack(surface)
            : task == UnitreeG1Task::ballDodge
                ? makeUnitreeG1BallDodgeTaskPack(surface)
            : task == UnitreeG1Task::ballDisturbanceRecovery
                ? makeUnitreeG1BallDisturbanceRecoveryTaskPack(surface)
            : task == UnitreeG1Task::supineGetUpDiscovery
                ? makeUnitreeG1SupineGetUpDiscoveryTaskPack(surface)
                : makeUnitreeG1LocomotionTaskPack(surface),
    };
    appendLocomotionSurface(
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
