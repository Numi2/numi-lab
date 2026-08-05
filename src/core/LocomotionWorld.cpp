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

constexpr std::array<float, kUnitreeG1JointCount>
    kUnitreeG1ActionScales{{
        0.55f, 0.35f, 0.55f, 0.35f, 0.44f, 0.44f,
        0.55f, 0.35f, 0.55f, 0.35f, 0.44f, 0.44f,
        0.55f, 0.44f, 0.44f,
        0.44f, 0.44f, 0.44f, 0.44f, 0.44f, 0.07f, 0.07f,
        0.44f, 0.44f, 0.44f, 0.44f, 0.44f, 0.07f, 0.07f,
    }};

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
    const TaskPack& task,
    const std::span<const RobotActuatorSpec> actuators,
    const TaskObservationProgram& observations,
    const TaskResetProgram& reset,
    CompiledLocomotionWorld& compiled
) {
    LocomotionWorldCompileDiagnostics diagnostics;
    CompiledLocomotionWorld staged;
    diagnostics.world = compileMetalWorld(
        authored.model,
        articulationIndex,
        staged.world,
        task.capacities
    );
    if (!diagnostics.world.succeeded()) {
        return diagnostics;
    }
    diagnostics.task = compileTaskProgram(
        task,
        actuators,
        observations,
        reset,
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

namespace {

EngineModel makeSceneComponentBase(
    const EngineModel& reference,
    const std::string_view name
) {
    if (reference.materials.empty()) {
        throw std::invalid_argument(
            "scene component requires a reference material"
        );
    }
    EngineModel component;
    component.name = std::string{name};
    component.world = reference.world;
    component.world.articulationCount = 0u;
    component.world.bodyCount = 0u;
    component.world.jointCount = 0u;
    component.world.nq = 0u;
    component.world.nv = 0u;
    component.world.shapeCount = 0u;
    component.materials.push_back(reference.materials.front());
    component.world.materialCount = 1u;
    return component;
}

} // namespace

LocomotionSceneComponent makeLocomotionSurfaceComponent(
    const EngineModel& referenceMechanics,
    const LocomotionSurface surface
) {
    LocomotionSceneComponent component;
    component.mechanics = makeSceneComponentBase(
        referenceMechanics,
        surface == LocomotionSurface::ground
            ? "locomotion_ground"
            : "locomotion_terrain"
    );
    appendLocomotionSurface(
        component.mechanics,
        component.defaultBodyStates,
        surface
    );
    component.mechanics.bodyNames = {
        surface == LocomotionSurface::ground
            ? "locomotion_ground"
            : "locomotion_terrain"
    };
    component.mechanics.shapeNames = {
        component.mechanics.bodyNames.front() + "/collision"
    };
    std::string reason;
    if (!component.mechanics.valid(&reason)) {
        throw std::runtime_error(
            "locomotion surface component is invalid: " + reason
        );
    }
    return component;
}

LocomotionSceneComponent makeLocomotionDynamicSphereComponent(
    const EngineModel& referenceMechanics,
    const std::span<const LocomotionDynamicSphere> spheres
) {
    LocomotionWorld authored;
    authored.model = makeSceneComponentBase(
        referenceMechanics,
        "locomotion_dynamic_spheres"
    );
    appendLocomotionDynamicSpheres(authored, spheres);
    authored.model.bodyNames.clear();
    authored.model.shapeNames.clear();
    for (std::size_t index = 0u; index < spheres.size(); ++index) {
        const std::string id =
            "locomotion_dynamic_sphere_" + std::to_string(index);
        authored.model.bodyNames.push_back(id);
        authored.model.shapeNames.push_back(id + "/collision");
    }
    std::string reason;
    if (!authored.model.valid(&reason)) {
        throw std::runtime_error(
            "locomotion sphere component is invalid: " + reason
        );
    }
    return {
        .mechanics = std::move(authored.model),
        .defaultBodyStates = std::move(authored.sceneBodies),
    };
}

LocomotionWorld makeWorldPackLocomotionWorld(
    const MRWorldPack& worldPack
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
    const LocomotionSurface surface,
    TaskObservationProgram& observations,
    TaskResetProgram& reset
) {
    const G1ModelMetadata& metadata = unitreeG1Metadata();
    TaskPack task;
    task.id = "unitree_g1_mjlab_velocity";
    task.outcomes = {
        {"tracking", "ratio", TaskOutcomeSource::trackingScore,
            TaskOutcomeDirection::higherIsBetter},
        {"root_height", "m", TaskOutcomeSource::rootHeight,
            TaskOutcomeDirection::neutral},
        {"tilt", "rad", TaskOutcomeSource::tilt,
            TaskOutcomeDirection::lowerIsBetter},
        {"contact_reward", "reward", TaskOutcomeSource::contactReward,
            TaskOutcomeDirection::higherIsBetter},
    };
    // The topology envelope contains every eligible self-collision pair.
    // Locomotion instead compiles an explicit operational arena. A fresh-seed
    // ball-dodge continuation reached 145 active pairs, so retain six Wave32
    // cohorts: the next aligned boundary plus one complete cohort of reserve.
    // Capacity overflow remains transactional and reports the exact required
    // stage count, so this is a replayable contract rather than truncation.
    task.capacities = {
        .candidatePairs = 192u,
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
    observations.actorHistoryLength = 1u;
    observations.criticHistoryLength = 1u;
    observations.criticIncludesCleanHistory = false;
    task.maximumEpisodeSteps = 1000u;
    reset.maximumActionDelaySteps = 0u;
    reset.maximumObservationDelaySteps = 0u;
    task.difficultyBandCount = 11u;
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
    task.commands.difficultyStep = {
        0.1f, 0.1f, 0.1f, 0.0f,
    };
    task.commands.standingProbability = 1.0f;
    task.commands.difficultySamplingExponent = 2.0f;
    task.commands.minimumDurationSeconds = 10.0f;
    task.commands.maximumDurationSeconds = 10.0f;
    task.pushes.maximumVelocity = 0.0f;
    task.pushes.minimumIntervalSeconds = 5.0f;
    task.pushes.maximumIntervalSeconds = 5.0f;

    task.actions.reserve(metadata.jointLimits.size());
    for (std::size_t index = 0u;
         index < metadata.jointLimits.size();
         ++index) {
        const G1JointLimit& joint = metadata.jointLimits[index];
        task.actions.push_back({
            .actuator = std::string{joint.name},
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
        observations.actorFrame.push_back(observation(
            TaskObservationSource::rootAngularVelocityLocal,
            {},
            component
        ));
    }
    for (std::uint32_t component = 0u;
         component < 3u;
         ++component) {
        observations.actorFrame.push_back(observation(
            TaskObservationSource::projectedGravity,
            {},
            component
        ));
    }
    for (std::uint32_t component = 0u;
         component < 3u;
         ++component) {
        observations.actorFrame.push_back(observation(
            TaskObservationSource::command,
            {},
            component
        ));
    }
    for (std::uint32_t component = 0u;
         component < 2u;
         ++component) {
        observations.actorFrame.push_back(observation(
            TaskObservationSource::gaitPhase,
            {},
            component
        ));
    }
    for (const G1JointLimit& joint : metadata.jointLimits) {
        observations.actorFrame.push_back(observation(
            TaskObservationSource::jointPositionError,
            joint.name,
            0u
        ));
    }
    for (const G1JointLimit& joint : metadata.jointLimits) {
        observations.actorFrame.push_back(observation(
            TaskObservationSource::jointVelocity,
            joint.name,
            0u
        ));
    }
    for (const G1JointLimit& joint : metadata.jointLimits) {
        observations.actorFrame.push_back(observation(
            TaskObservationSource::previousAction,
            joint.name,
            0u
        ));
    }
    observations.critic = observations.actorFrame;

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
        .supportPatchBounds =
            metadata.feet[0].supportPatchBounds,
        .supportPatchWidth = 2u,
        .supportPatchHeight = 2u,
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
        .supportPatchBounds =
            metadata.feet[1].supportPatchBounds,
        .supportPatchWidth = 2u,
        .supportPatchHeight = 2u,
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

std::span<const float>
unitreeG1LocomotionActionScales() noexcept {
    return kUnitreeG1ActionScales;
}

TaskPack makeUnitreeG1DisturbanceRecoveryTaskPack(
    const LocomotionSurface surface,
    TaskObservationProgram& observations,
    TaskResetProgram& reset
) {
    TaskPack task = makeUnitreeG1LocomotionTaskPack(
        surface, observations, reset
    );
    task.id = "unitree_g1_disturbance_recovery";

    // This skill owns balance and recovery, not commanded locomotion. Keeping
    // the actor contract unchanged permits exact initialization from the
    // official Unitree velocity actor while the critic learns the new task.
    task.commands.lower = {};
    task.commands.upper = {};
    task.commands.limitLower = {};
    task.commands.limitUpper = {};
    task.commands.difficultyStep = {};
    task.commands.standingProbability = 1.0f;
    task.commands.difficultySamplingExponent = 2.0f;
    task.commands.minimumDurationSeconds = 10.0f;
    task.commands.maximumDurationSeconds = 10.0f;

    // The runtime scales the maximum impulse velocity by sampled difficulty.
    // Level zero therefore proves quiet standing before disturbances ramp to
    // 2.5 m/s from deterministic, replayable horizontal directions.
    task.difficultyBandCount = 11u;
    task.pushes.maximumVelocity = 2.5f;
    task.pushes.minimumIntervalSeconds = 1.5f;
    task.pushes.maximumIntervalSeconds = 3.0f;
    reset.maximumActionDelaySteps = 2u;
    reset.operators = {
        {
            .operation = TaskRandomizationOperator::rootPosition,
            .minimumDifficultyBand = 2u,
            .parameters = {0.02f, 0.02f, 0.01f, 0.0f},
        },
        {
            .operation = TaskRandomizationOperator::actionPosition,
            .minimumDifficultyBand = 3u,
            .parameters = {-0.03f, 0.03f, 0.0f, 0.0f},
        },
        {
            .operation = TaskRandomizationOperator::actionVelocity,
            .minimumDifficultyBand = 4u,
            .parameters = {-0.10f, 0.10f, 0.0f, 0.0f},
        },
        {
            .operation = TaskRandomizationOperator::bodyParameter,
            .target = "robot",
            .component = 0u,
            .minimumDifficultyBand = 5u,
            .parameters = {0.9f, 1.1f, 0.0f, 0.0f},
        },
        {
            .operation = TaskRandomizationOperator::controllerParameter,
            .component = 0u,
            .minimumDifficultyBand = 6u,
            .parameters = {0.9f, 1.1f, 0.0f, 0.0f},
        },
        {
            .operation = TaskRandomizationOperator::controllerParameter,
            .component = 1u,
            .minimumDifficultyBand = 7u,
            .parameters = {0.9f, 1.1f, 0.0f, 0.0f},
        },
        {
            .operation = TaskRandomizationOperator::actionDelay,
            .minimumDifficultyBand = 8u,
            .parameters = {0.0f, 2.0f, 0.0f, 0.0f},
        },
    };
    return task;
}

TaskPack makeUnitreeG1SupineGetUpDiscoveryTaskPack(
    const LocomotionSurface surface,
    TaskObservationProgram& observations,
    TaskResetProgram& reset
) {
    const G1ModelMetadata& metadata = unitreeG1Metadata();
    TaskPack task = makeUnitreeG1LocomotionTaskPack(
        surface, observations, reset
    );
    task.id = "unitree_g1_supine_get_up_discovery";
    task.outcomes = {
        {"root_height", "m", TaskOutcomeSource::rootHeight,
            TaskOutcomeDirection::neutral},
        {"tilt", "rad", TaskOutcomeSource::tilt,
            TaskOutcomeDirection::lowerIsBetter},
        {"contact_reward", "reward", TaskOutcomeSource::contactReward,
            TaskOutcomeDirection::higherIsBetter},
    };
    // Recovery legitimately brings knees, hands, arms, trunk, and both feet
    // into the same contact graph. The first 4,096-environment band-2 soak
    // measured a pathological but finite reset at 535 candidate pairs, 530
    // raw contacts, 511 manifolds, and 519 constraint blocks. A 672-slot
    // (21 Wave32) arena gives that measured envelope 20-31% reserve while
    // staying below the Mac mini's recommended working-set limit at 4,096
    // environments.
    task.capacities.candidatePairs = 672u;
    task.capacities.rawContacts = 672u;
    task.capacities.manifolds = 672u;
    task.capacities.constraintBlocks = 672u;
    task.capacities.constraintRows = 2016u;
    task.capacities.endpointRuntimeRecords = 1344u;
    task.capacities.articulationPointQueries = 1344u;
    task.capacities.qualityRows = 2016u;
    task.capacities.islandConstraintReferences = 672u;
    // Supine recovery exposes the convex hulls that remain separated during
    // nominal standing. Keep the convex reserve in the same measured
    // envelope as the contact graph so the broadphase cannot fail first.
    task.capacities.hardConvexPairs = 672u;
    task.capacities.meshTriangleCandidates = 4096u;
    // Five seconds covers a human-scale get-up and quiet stabilization while
    // refreshing every reset state often enough for mixed-state PPO batches.
    task.maximumEpisodeSteps = 256u;
    task.difficultyBandCount = 8u;
    // Do not let an unqualified student perturb a physically successful
    // get-up guide. The executed reference is still published in autonomous
    // normalized action coordinates for shadow-mode distillation.
    task.interactionStudentAuthority = 0.0f;
    task.commands.lower = {};
    task.commands.upper = {};
    task.commands.limitLower = {};
    task.commands.limitUpper = {};
    task.commands.difficultyStep = {};
    task.commands.standingProbability = 1.0f;
    task.commands.difficultySamplingExponent = 1.0f;
    task.pushes.maximumVelocity = 0.0f;
    task.pushes.minimumIntervalSeconds = 10.0f;
    task.pushes.maximumIntervalSeconds = 10.0f;
    task.terminations.clear();
    // A standing attempt has ceased to contain useful balance corrections
    // once the pelvis is low or the body has tipped beyond recovery. Restart
    // only the standing reset bands promptly; floor and squat bands must be
    // allowed to inhabit exactly those states while learning to rise.
    task.terminations = {
        {
            .operation =
                TaskTerminationOperator::minimumRootHeight,
            .reason = MR_TASK_TERMINATION_HEIGHT,
            .priority = 1u,
            .threshold = 0.45f,
            .failurePenalty = -2.0f,
            .minimumDifficultyBand = 4u,
        },
        {
            .operation = TaskTerminationOperator::maximumTilt,
            .reason = MR_TASK_TERMINATION_TILT,
            .priority = 2u,
            .threshold = 0.9f,
            .failurePenalty = -2.0f,
            .minimumDifficultyBand = 4u,
        },
    };

    // Get-up needs the complete mechanism range: a locomotion-sized residual
    // cannot represent the hip, knee, waist, and shoulder travel between a
    // fallen configuration and nominal stance. Keep normalized actions while
    // making every reachable joint target expressible around the stance.
    constexpr std::array<float, kUnitreeG1JointCount> nominalStance{{
        -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,
        -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.35f, 0.18f, 0.0f, 0.87f, 0.0f, 0.0f, 0.0f,
        0.35f, -0.18f, 0.0f, 0.87f, 0.0f, 0.0f, 0.0f,
    }};
    std::array<float, kUnitreeG1JointCount>
        previousActionObservationScales{};
    for (std::size_t index = 0u; index < task.actions.size(); ++index) {
        const G1JointLimit& limit = metadata.jointLimits[index];
        const float fullRangeScale = std::max(
            std::abs(limit.lowerPosition - nominalStance[index]),
            std::abs(limit.upperPosition - nominalStance[index])
        );
        previousActionObservationScales[index] =
            fullRangeScale /
            unitreeG1LocomotionActionScales()[index];
    }

    // Keep recovery's deployable proprioceptive prefix identical to the
    // standing/dodge actor: five temporal frames, plantar support sense, and
    // both 2x2 sole pressure fields. This makes balance a transferable skill
    // instead of asking get-up to rediscover it through a different input
    // layout. Camera observations remain outside this task; recovery phase is
    // appended once by the imagined-interaction compiler.
    std::vector<TaskObservationOperatorSpec> criticFrame;
    criticFrame.reserve(93u);
    criticFrame.insert(
        criticFrame.end(),
        observations.actorFrame.begin(),
        observations.actorFrame.begin() + 5
    );
    criticFrame.insert(
        criticFrame.end(),
        observations.actorFrame.begin() + 11,
        observations.actorFrame.end()
    );
    criticFrame.push_back(observations.actorFrame[5u]);

    std::erase_if(
        observations.actorFrame,
        [](const TaskObservationOperatorSpec& observation) {
            return observation.source == TaskObservationSource::gaitPhase;
        }
    );
    constexpr std::array<float, 3u> supportSenseScales{
        0.002f, 1.0f, 1.0f,
    };
    for (TaskObservationOperatorSpec& observation : observations.actorFrame) {
        if (observation.source == TaskObservationSource::command) {
            observation.source = TaskObservationSource::supportSense;
            observation.target.clear();
            observation.scale = supportSenseScales[observation.component];
            observation.offset = 0.0f;
            observation.noiseAmplitude = 0.0f;
            observation.biasLower = 0.0f;
            observation.biasUpper = 0.0f;
            observation.normalizeVector3 = false;
            continue;
        }
        if (observation.source ==
                TaskObservationSource::rootAngularVelocityLocal) {
            observation.scale = 0.2f;
        } else if (observation.source ==
                       TaskObservationSource::jointVelocity) {
            observation.scale = 0.05f;
        } else if (observation.source ==
                       TaskObservationSource::projectedGravity) {
            observation.normalizeVector3 = true;
        } else if (observation.source ==
                       TaskObservationSource::previousAction) {
            for (std::size_t joint = 0u;
                 joint < metadata.jointLimits.size();
                 ++joint) {
                if (observation.target ==
                        metadata.jointLimits[joint].name) {
                    observation.scale =
                        previousActionObservationScales[joint];
                    break;
                }
            }
        }
    }
    constexpr std::array<float, 13u> supportPatchScales{
        0.002f, 0.002f, 0.002f,
        0.02f, 0.02f, 0.02f,
        10.0f, 10.0f,
        100.0f,
        2.0e-5f, 2.0e-5f, 2.0e-5f, 2.0e-5f,
    };
    for (const std::string_view group : {
             std::string_view{"left_foot_contact"},
             std::string_view{"right_foot_contact"},
         }) {
        for (std::uint32_t component = 0u;
             component < supportPatchScales.size();
             ++component) {
            observations.actorFrame.push_back({
                .source = TaskObservationSource::supportPatch,
                .target = std::string{group},
                .component = component,
                .scale = supportPatchScales[component],
            });
        }
    }
    observations.critic = std::move(criticFrame);
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
    observations.critic.insert(
        observations.critic.end(),
        privilegedState.begin(),
        privilegedState.end()
    );
    observations.actorHistoryLength = 5u;
    observations.criticHistoryLength = 10u;

    const auto appendRecoveryContact =
        [&task](const std::string& id, const std::string& body) {
            task.contactGroups.push_back({
                .id = id,
                .bodies = {body},
                .referenceBody = body,
            });
        };
    appendRecoveryContact(
        "left_wrist_contact",
        "left_wrist_yaw_link"
    );
    appendRecoveryContact(
        "right_wrist_contact",
        "right_wrist_yaw_link"
    );
    appendRecoveryContact(
        "left_knee_contact",
        "left_knee_link"
    );
    appendRecoveryContact(
        "right_knee_contact",
        "right_knee_link"
    );
    task.contactGroups.push_back({
        .id = "recovery_assist_contact",
        .bodies = {
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
            "left_knee_link",
            "right_knee_link",
        },
        .referenceBody = "pelvis",
    });
    task.contactGroups.push_back({
        .id = "trunk_contact",
        .bodies = {"pelvis", "torso_link"},
        .referenceBody = "pelvis",
    });

    task.rewards = {
        {
            .operation = TaskRewardOperator::rootHeightProgress,
            .weight = 1.0f,
        },
        {
            // This integrates to the accepted reduction in tilt over the
            // episode. It supplies direct temporal credit for rolling and
            // levering the torso upward, including intermediate hand/knee
            // support, without prescribing a particular joint trajectory.
            .operation = TaskRewardOperator::recoveryTiltProgress,
            .sourceGroup = "trunk_contact",
            .weight = 8.0f,
            .parameters = {1.40f, 0.60f, 0.40f, 0.0f},
        },
        {
            // A sustained upright recovery is a positive event, not a
            // condition that suppresses partial height or tilt progress.
            .operation = TaskRewardOperator::recoveryCompletion,
            .sourceGroup = "trunk_contact",
            .weight = 20.0f,
            .parameters = {1.40f, 0.60f, 0.40f, 0.0f},
        },
        {
            // Reward the physically meaningful continuum from bracing on a
            // hand/knee, through trunk clearance and CoP-supported load
            // transfer, to a quiet bilateral stand. The native solver owns
            // every contact and support value; this does not prescribe a
            // kinematic animation or hide partial progress behind a gate.
            .operation = TaskRewardOperator::wholeBodyRecovery,
            .sourceGroup = "recovery_assist_contact",
            .target = "trunk_contact",
            .weight = 40.0f,
            .parameters = {0.65f, 0.80f, 0.45f, 0.50f},
        },
        {
            .operation = TaskRewardOperator::standingCompletion,
            .weight = 40.0f,
            .parameters = {0.65f, 0.8f, 0.0f, 0.0f},
        },
        {
            .operation = TaskRewardOperator::restoration,
            .weight = 40.0f,
            .parameters = {0.22f, 0.40f, 0.94f, 0.35f},
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
    reset.operators = {
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
        reset.operators.push_back({
            .operation = TaskRandomizationOperator::jointPosition,
            .target = std::string{metadata.jointLimits[index].name},
            .parameters = {
                supine[index], supine[index], 0.0f, 0.0f,
            },
        });
    }
    // Uniform reset sampling replaces a promotion ladder. Bands 0-1 remain
    // supine, 2 is low squat, 3 is high squat, and 4-7 are nominal standing.
    // This keeps every recovery state live in every update while spending
    // half the batch on the balance skill required to finish a get-up.
    constexpr std::array<float, kUnitreeG1JointCount> lowSquat{{
        -1.05f, 0.0f, 0.0f, 2.10f, -0.85f, 0.0f,
        -1.05f, 0.0f, 0.0f, 2.10f, -0.85f, 0.0f,
        0.0f, 0.0f, 0.25f,
        0.90f, 0.25f, 0.0f, 1.20f, 0.0f, 0.0f, 0.0f,
        0.90f, -0.25f, 0.0f, 1.20f, 0.0f, 0.0f, 0.0f,
    }};
    constexpr std::array<float, kUnitreeG1JointCount> highSquat{{
        -0.60f, 0.0f, 0.0f, 1.20f, -0.60f, 0.0f,
        -0.60f, 0.0f, 0.0f, 1.20f, -0.60f, 0.0f,
        0.0f, 0.0f, 0.10f,
        0.60f, 0.20f, 0.0f, 1.00f, 0.0f, 0.0f, 0.0f,
        0.60f, -0.20f, 0.0f, 1.00f, 0.0f, 0.0f, 0.0f,
    }};
    const auto appendResetPose =
        [&reset, &metadata](
            const std::uint32_t band,
            const float rootHeight,
            const std::array<float, kUnitreeG1JointCount>& pose
        ) {
            reset.operators.push_back({
                .operation = TaskRandomizationOperator::rootOrientation,
                .minimumDifficultyBand = band,
                .parameters = {0.0f, 0.0f, 0.0f, 1.0f},
            });
            reset.operators.push_back({
                .operation = TaskRandomizationOperator::rootHeight,
                .minimumDifficultyBand = band,
                .parameters = {
                    rootHeight, rootHeight, 0.0f, 0.0f,
                },
            });
            for (std::size_t index = 0u; index < pose.size(); ++index) {
                reset.operators.push_back({
                    .operation =
                        TaskRandomizationOperator::jointPosition,
                    .target = std::string{
                        metadata.jointLimits[index].name
                    },
                    .minimumDifficultyBand = band,
                    .parameters = {
                        pose[index], pose[index], 0.0f, 0.0f,
                    },
                });
            }
        };
    appendResetPose(2u, 0.30f, lowSquat);
    appendResetPose(3u, 0.52f, highSquat);
    // rootHeight writes the floating-base COM coordinate. The canonical
    // standing reset is a 0.8 m pelvis link-frame origin, while G1's pelvis
    // COM is 0.07603006 m below that frame. Using 0.78 here launched the feet
    // roughly 5.6 cm above the floor and turned every standing episode into
    // an avoidable landing impact.
    appendResetPose(4u, 0.72396994f, nominalStance);
    return task;
}

TaskPack makeUnitreeG1DevelopmentalRecoveryTaskPack(
    const LocomotionSurface surface,
    TaskObservationProgram& observations,
    TaskResetProgram& reset
) {
    // Start from the native physical recovery contract, then replace its
    // flat floor-to-stand entry with an authored developmental rung. The
    // solver, contacts, observations, rewards, and actuator ABI remain the
    // authority; only the reset curriculum is specialized here.
    TaskPack task = makeUnitreeG1SupineGetUpDiscoveryTaskPack(
        surface,
        observations,
        reset
    );
    task.id = "unitree_g1_developmental_recovery";
    task.outcomes = {
        {"root_height", "m", TaskOutcomeSource::rootHeight,
            TaskOutcomeDirection::neutral},
        {"tilt", "rad", TaskOutcomeSource::tilt,
            TaskOutcomeDirection::lowerIsBetter},
        {"height_progress", "reward",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::rootHeightProgress},
        {"tilt_progress", "reward",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::recoveryTiltProgress},
        {"recovery_completion", "reward",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::recoveryCompletion},
        {"whole_body_recovery", "reward",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::wholeBodyRecovery},
        {"standing_completion", "reward",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::standingCompletion},
        {"restoration", "reward",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::restoration},
    };

    const G1ModelMetadata& metadata = unitreeG1Metadata();
    // Keep the trunk supine but lift it into a tucked, four-point-supported
    // developmental pose. This is deliberately a contact-rich rung: the
    // learner first loads wrists/knees and changes its joint geometry, then
    // rotates and lifts the trunk through later squat regions.
    constexpr std::array<float, 4u> tuckOrientation{{
        -0.0098234f, 0.49986f, 0.017525f, -0.86587f,
    }};
    constexpr std::array<float, kUnitreeG1JointCount> tuckBrace{{
        -0.70f, 0.12f, 0.70f, 1.00f, -0.83f, 0.0f,
        -0.70f, -0.12f, 0.70f, 1.00f, -0.83f, 0.0f,
        0.0f, 0.0f, 0.15f,
        0.68f, 0.54f, 0.21f, 1.25f, 0.0f, 0.0f, 0.0f,
        0.52f, -0.64f, -0.21f, 1.30f, 0.0f, 0.0f, 0.0f,
    }};

    std::vector<TaskRandomizationOperatorSpec> tuckOperators;
    tuckOperators.reserve(2u + tuckBrace.size());
    tuckOperators.push_back({
        .operation = TaskRandomizationOperator::rootOrientation,
        .minimumDifficultyBand = 1u,
        .parameters = {
            tuckOrientation[0u], tuckOrientation[1u],
            tuckOrientation[2u], tuckOrientation[3u],
        },
    });
    tuckOperators.push_back({
        .operation = TaskRandomizationOperator::rootHeight,
        .minimumDifficultyBand = 1u,
        .parameters = {0.115f, 0.115f, 0.0f, 0.0f},
    });
    for (std::size_t index = 0u; index < tuckBrace.size(); ++index) {
        tuckOperators.push_back({
            .operation = TaskRandomizationOperator::jointPosition,
            .target = std::string{metadata.jointLimits[index].name},
            .minimumDifficultyBand = 1u,
            .parameters = {
                tuckBrace[index], tuckBrace[index], 0.0f, 0.0f,
            },
        });
    }

    // Randomization operators are applied in authored order. Insert before
    // the existing low-squat rung so band 1 overrides supine, while bands 2+
    // still select their own progressively upright poses.
    const auto firstSquat = std::ranges::find_if(
        reset.operators,
        [](const TaskRandomizationOperatorSpec& operation) {
            return operation.minimumDifficultyBand >= 2u;
        }
    );
    reset.operators.insert(
        firstSquat,
        tuckOperators.begin(),
        tuckOperators.end()
    );
    return task;
}

TaskPack makeUnitreeG1AdultLocomotionTaskPack(
    const LocomotionSurface surface,
    TaskObservationProgram& observations,
    TaskResetProgram& reset
) {
    // Preserve the learned developmental actor ABI: the adult brain receives
    // the same 610-value, five-frame proprioceptive/plantar history and the
    // same 29 normalized joint actions. Only the task meaning changes: the
    // three support-sense slots become commanded velocity slots, while the
    // critic receives both command and privileged support state.
    TaskPack task = makeUnitreeG1DevelopmentalRecoveryTaskPack(
        surface, observations, reset
    );
    task.id = "unitree_g1_adult_locomotion";
    task.outcomes = {
        {"tracking", "ratio", TaskOutcomeSource::trackingScore,
            TaskOutcomeDirection::higherIsBetter},
        {"root_height", "m", TaskOutcomeSource::rootHeight,
            TaskOutcomeDirection::neutral},
        {"tilt", "rad", TaskOutcomeSource::tilt,
            TaskOutcomeDirection::lowerIsBetter},
        {"contact_reward", "reward", TaskOutcomeSource::contactReward,
            TaskOutcomeDirection::higherIsBetter},
        {"standing_completion", "reward",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::standingCompletion},
        {"restoration", "reward",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::restoration},
    };
    task.maximumEpisodeSteps = 1000u;
    task.difficultyBandCount = 11u;
    task.baseHeightTarget = 0.78f;
    task.gaitPeriodSeconds = 0.6f;
    task.clearanceTarget = 0.10f;
    task.successTrackingThreshold = 0.8f;
    task.supportForceThreshold = 1.0f;
    task.commands.lower = {0.0f, 0.0f, 0.0f, 0.0f};
    task.commands.upper = {0.0f, 0.0f, 0.0f, 0.0f};
    task.commands.limitLower = {-0.5f, -0.5f, -1.0f, 0.0f};
    task.commands.limitUpper = {1.0f, 0.5f, 1.0f, 0.0f};
    task.commands.difficultyStep = {0.1f, 0.1f, 0.1f, 0.0f};
    // Keep a substantial quiet-command fraction while the adult learns to
    // stand under command; this decays only through authored band expansion.
    task.commands.standingProbability = 0.35f;
    task.commands.difficultySamplingExponent = 2.0f;
    task.commands.minimumDurationSeconds = 10.0f;
    task.commands.maximumDurationSeconds = 10.0f;
    // Adult locomotion is also a survivability task. Impulses are sampled by
    // the same deterministic difficulty band as commands, so stress grows
    // after the quiet standing rung instead of being an untracked hazard.
    task.pushes.maximumVelocity = 1.5f;
    task.pushes.minimumIntervalSeconds = 2.0f;
    task.pushes.maximumIntervalSeconds = 4.0f;

    std::erase_if(
        task.contactGroups,
        [](const TaskContactGroup& group) {
            return group.id == "left_wrist_contact" ||
                group.id == "right_wrist_contact" ||
                group.id == "left_knee_contact" ||
                group.id == "right_knee_contact" ||
                group.id == "recovery_assist_contact" ||
                group.id == "trunk_contact";
        }
    );

    // The developmental reset contains supine, tuck, squat, and standing
    // regions. Adult episodes begin from the solver-authored nominal standing
    // pose only; later bands add measured reset/action/controller variation.
    std::erase_if(
        reset.operators,
        [](const TaskRandomizationOperatorSpec& operation) {
            return operation.minimumDifficultyBand < 4u;
        }
    );
    for (TaskRandomizationOperatorSpec& operation : reset.operators) {
        operation.minimumDifficultyBand = 0u;
    }
    reset.maximumActionDelaySteps = 2u;
    reset.maximumObservationDelaySteps = 0u;
    reset.operators.insert(
        reset.operators.end(),
        {
            {
                .operation = TaskRandomizationOperator::rootPosition,
                .minimumDifficultyBand = 2u,
                .parameters = {0.02f, 0.02f, 0.01f, 0.0f},
            },
            {
                .operation = TaskRandomizationOperator::actionPosition,
                .minimumDifficultyBand = 3u,
                .parameters = {-0.03f, 0.03f, 0.0f, 0.0f},
            },
            {
                .operation = TaskRandomizationOperator::actionVelocity,
                .minimumDifficultyBand = 4u,
                .parameters = {-0.10f, 0.10f, 0.0f, 0.0f},
            },
            {
                .operation = TaskRandomizationOperator::bodyParameter,
                .target = "robot",
                .component = 0u,
                .minimumDifficultyBand = 5u,
                .parameters = {0.9f, 1.1f, 0.0f, 0.0f},
            },
            {
                .operation = TaskRandomizationOperator::controllerParameter,
                .component = 0u,
                .minimumDifficultyBand = 6u,
                .parameters = {0.9f, 1.1f, 0.0f, 0.0f},
            },
            {
                .operation = TaskRandomizationOperator::controllerParameter,
                .component = 1u,
                .minimumDifficultyBand = 7u,
                .parameters = {0.9f, 1.1f, 0.0f, 0.0f},
            },
            {
                .operation = TaskRandomizationOperator::actionDelay,
                .minimumDifficultyBand = 8u,
                .parameters = {0.0f, 2.0f, 0.0f, 0.0f},
            },
        }
    );

    std::uint32_t convertedCommandSlots = 0u;
    for (TaskObservationOperatorSpec& observation :
             observations.actorFrame) {
        if (observation.source != TaskObservationSource::supportSense ||
            observation.component >= 3u) {
            continue;
        }
        observation.source = TaskObservationSource::command;
        observation.target.clear();
        observation.scale = 1.0f;
        observation.offset = 0.0f;
        observation.noiseAmplitude = 0.0f;
        observation.biasLower = 0.0f;
        observation.biasUpper = 0.0f;
        observation.normalizeVector3 = false;
        ++convertedCommandSlots;
    }
    if (convertedCommandSlots != 3u) {
        throw std::logic_error(
            "adult G1 task lost its transferable command observation slots"
        );
    }
    for (std::uint32_t component = 0u; component < 3u; ++component) {
        observations.critic.push_back({
            .source = TaskObservationSource::command,
            .component = component,
        });
        observations.critic.push_back({
            .source = TaskObservationSource::supportSense,
            .component = component,
            .scale = component == 0u ? 0.002f : 1.0f,
        });
    }

    task.rewards = {
        {
            .operation = TaskRewardOperator::linearVelocityTracking,
            .weight = 1.0f,
            .parameters = {0.25f, 0.0f, 0.0f, 0.0f},
        },
        {
            .operation = TaskRewardOperator::yawVelocityTracking,
            .weight = 0.5f,
            .parameters = {0.25f, 0.0f, 0.0f, 0.0f},
        },
        {.operation = TaskRewardOperator::constant, .weight = 0.15f},
        {
            .operation = TaskRewardOperator::rootVerticalVelocitySquared,
            .weight = -2.0f,
        },
        {
            .operation = TaskRewardOperator::rootRollPitchVelocitySquared,
            .weight = -0.05f,
        },
        {
            .operation =
                TaskRewardOperator::projectedGravityHorizontalSquared,
            .weight = -5.0f,
        },
        {
            .operation = TaskRewardOperator::rootHeightErrorSquared,
            .weight = -10.0f,
        },
        {
            .operation = TaskRewardOperator::jointVelocitySquared,
            .weight = -0.001f,
        },
        {
            .operation = TaskRewardOperator::jointAccelerationSquared,
            .weight = -2.5e-7f,
        },
        {
            .operation = TaskRewardOperator::actionRateSquared,
            .weight = -0.05f,
        },
        {
            .operation = TaskRewardOperator::jointLimitViolationAbsolute,
            .weight = -5.0f,
            .parameters = {0.9f, 0.0f, 0.0f, 0.0f},
        },
        {
            .operation = TaskRewardOperator::mechanicalPower,
            .weight = -2.0e-5f,
        },
        {
            .operation = TaskRewardOperator::jointGroupPostureAbsolute,
            .sourceGroup = "waist",
            .weight = -1.0f,
        },
        {
            .operation = TaskRewardOperator::jointGroupPostureAbsolute,
            .sourceGroup = "hips",
            .weight = -1.0f,
        },
        {
            .operation = TaskRewardOperator::jointGroupPostureAbsolute,
            .sourceGroup = "arms",
            .weight = -0.1f,
        },
        {
            .operation = TaskRewardOperator::gaitContactMatch,
            .weight = 0.5f,
        },
        {
            .operation = TaskRewardOperator::footClearance,
            .weight = 1.0f,
            .parameters = {0.05f, 2.0f, 0.0f, 0.0f},
        },
        {
            .operation = TaskRewardOperator::supportSlip,
            .weight = -0.2f,
        },
        {
            .operation = TaskRewardOperator::forbiddenContact,
            .sourceGroup = "undesired_contact",
            .weight = -1.0f,
        },
        {
            .operation = TaskRewardOperator::standingCompletion,
            .weight = 20.0f,
            .parameters = {0.65f, 0.8f, 0.0f, 0.0f},
        },
        {
            .operation = TaskRewardOperator::restoration,
            .weight = 10.0f,
            .parameters = {0.22f, 0.40f, 0.94f, 0.35f},
        },
    };
    task.terminations = {
        {
            .operation = TaskTerminationOperator::minimumRootHeight,
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

TaskPack makeUnitreeG1BallDisturbanceRecoveryTaskPack(
    const LocomotionSurface surface,
    TaskObservationProgram& observations,
    TaskResetProgram& reset
) {
    const G1ModelMetadata& metadata = unitreeG1Metadata();
    TaskPack task = makeUnitreeG1DisturbanceRecoveryTaskPack(
        surface, observations, reset
    );
    task.id = "unitree_g1_ball_disturbance_recovery";
    task.pushes.maximumVelocity = 0.0f;
    // This stage learns pre-fall stability. Episode completion and recovery
    // events are both retained as physical outcomes rather than collapsed into
    // a promotion verdict.
    task.successTrackingThreshold = 0.70f;
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
        observations.actorFrame.push_back(confidence);
        observations.critic.push_back(confidence);
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
            observations.actorFrame.push_back(track);
            observations.critic.push_back(track);
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
    observations.critic.insert(
        observations.critic.end(),
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
            reset.operators.push_back({
                .operation = TaskRandomizationOperator::sceneBodyPosition,
                .target = name,
                .component = component,
                .parameters = {
                    positionLower[sphere][component],
                    positionUpper[sphere][component], 0.0f, 0.0f,
                },
            });
            reset.operators.push_back({
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
            reset.operators.push_back({
                .operation =
                    TaskRandomizationOperator::sceneBodyVelocity,
                .target = name,
                .component = impactAxis,
                .minimumDifficultyBand = level,
                .parameters = direction > 0.0f
                    ? mr_float4{
                          speedLower, speedUpper, 0.0f, 0.0f,
                      }
                    : mr_float4{
                          -speedUpper, -speedLower, 0.0f, 0.0f,
                      },
            });
        }
        reset.operators.push_back({
            .operation = TaskRandomizationOperator::sceneBodyLaunchStep,
            .target = name,
            .parameters = {
                launchSteps[sphere][0], launchSteps[sphere][1],
                0.0f, 0.0f,
            },
        });
        reset.operators.push_back({
            .operation =
                TaskRandomizationOperator::sceneBodyEventImpact,
            .target = name,
            .component = sphere,
            .minimumDifficultyBand = 1u,
            .parameters = {
                0.05f, 0.50f, 2.0f, 0.70f,
            },
        });
    }
    return task;
}

TaskPack makeUnitreeG1BallDodgeTaskPack(
    const LocomotionSurface surface,
    TaskObservationProgram& observations,
    TaskResetProgram& reset
) {
    TaskPack task =
        makeUnitreeG1BallDisturbanceRecoveryTaskPack(
            surface, observations, reset
        );
    task.id = "unitree_g1_ball_dodge";
    task.outcomes.insert(task.outcomes.end(), {
        {"projectile_clearance_reward", "reward",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::linkClearanceBarrier},
        {"projectile_evasion_reward", "reward",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::projectileEvasion},
        {"projectile_miss_reward", "reward",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::projectileMiss},
        {"projectile_safe_stillness_reward", "reward",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::projectileSafeStillness},
        {"projectile_safe_action_reward", "reward",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::projectileSafeActionRate},
        {"cbf_correction_reward", "reward",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::jointCbfCorrection},
        {"cbf_buffer_reward", "reward",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::jointCbfBuffer},
        {"projectile_predicted_clearance_reward", "reward",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::projectilePredictedClearance},
    });
    // Four levels correspond to the four authored projectile-speed bands.
    // Visual corruption rises over the same normalized range, so every level
    // has a physical and perceptual meaning instead of adding corruption-only
    // tail levels.
    task.difficultyBandCount = 4u;
    // Bias toward easier throws while keeping every authored speed band in
    // the deterministic episode distribution from the first update.
    task.commands.difficultySamplingExponent = 1.5f;
    task.pushes.projectileStandingProbability = 0.20f;
    task.pushes.projectileTargetHorizontalRadius = 0.40f;
    task.pushes.projectileHorizontalSpeedLower = 1.0f;
    task.pushes.projectileHorizontalSpeedUpper = 6.0f;
    task.pushes.projectileTargetHeightLower = 0.45f;
    task.pushes.projectileTargetHeightUpper = 1.35f;

    // This deployment task has one forward-facing torso camera. Keep every
    // launch origin inside its horizontal frustum so the actor is never
    // terminated by a threat that its authored sensor cannot observe. The
    // six lanes retain meaningful approach-angle diversity; broader azimuth
    // belongs to a later pack with corresponding side/rear sensing.
    constexpr std::array<std::array<float, 2>, 6> lateralRanges{{
        {{-0.65f, -0.45f}},
        {{-0.40f, -0.20f}},
        {{-0.15f,  0.05f}},
        {{ 0.05f,  0.25f}},
        {{ 0.25f,  0.45f}},
        {{ 0.45f,  0.65f}},
    }};
    constexpr std::array<std::array<float, 2>, 4> speedBands{{
        {{1.0f, 2.0f}},
        {{2.0f, 3.0f}},
        {{3.5f, 4.5f}},
        {{5.0f, 6.0f}},
    }};
    for (std::uint32_t sphere = 0u; sphere < lateralRanges.size(); ++sphere) {
        const std::string name =
            "locomotion_dynamic_sphere_" + std::to_string(sphere);
        const std::uint32_t impactAxis =
            sphere == 2u || sphere == 3u ? 1u : 0u;
        const float direction =
            sphere == 0u || sphere == 2u || sphere == 4u
            ? 1.0f
            : -1.0f;
        for (TaskRandomizationOperatorSpec& random : reset.operators) {
            if (random.target != name) {
                continue;
            }
            if (random.operation ==
                    TaskRandomizationOperator::sceneBodyVelocity &&
                random.component == impactAxis &&
                random.minimumDifficultyBand < speedBands.size()) {
                const auto& band = speedBands[
                    random.minimumDifficultyBand
                ];
                random.parameters = direction > 0.0f
                    ? mr_float4{band[0], band[1], 0.0f, 0.0f}
                    : mr_float4{-band[1], -band[0], 0.0f, 0.0f};
                continue;
            }
            if (random.operation !=
                    TaskRandomizationOperator::sceneBodyPosition) {
                continue;
            }
            switch (random.component) {
            case 0u:
                random.parameters = {1.30f, 1.70f, 0.0f, 0.0f};
                break;
            case 1u:
                random.parameters = {
                    lateralRanges[sphere][0],
                    lateralRanges[sphere][1],
                    0.0f,
                    0.0f,
                };
                break;
            case 2u:
                random.parameters = {1.00f, 1.35f, 0.0f, 0.0f};
                break;
            default:
                break;
            }
        }
    }

    // Deployment sees only ball-segmentation-masked depth. Preserve exact
    // object tracks in the critic for asymmetric PPO, but remove them from
    // the actor so privileged state cannot leak into the deployed policy.
    std::erase_if(
        observations.actorFrame,
        [](const TaskObservationOperatorSpec& observation) {
            return observation.source ==
                    TaskObservationSource::objectTrack ||
                observation.source ==
                    TaskObservationSource::gaitPhase;
        }
    );
    // Standing commands are identically zero in this task. Reuse those exact
    // three actor/critic slots for native plantar evidence so the existing
    // network and optimizer remain shape-compatible: total support load,
    // phase-signed bilateral load balance, and maximum support slip. The
    // generic operator reduces every authored support group and carries no
    // G1-specific Metal path.
    const auto installSupportSense = [](
        std::vector<TaskObservationOperatorSpec>& observations
    ) {
        constexpr std::array<float, 3u> scales{
            0.002f, 1.0f, 1.0f,
        };
        for (TaskObservationOperatorSpec& observation : observations) {
            if (observation.source != TaskObservationSource::command) {
                continue;
            }
            observation.source = TaskObservationSource::supportSense;
            observation.target.clear();
            observation.scale = scales[observation.component];
            observation.offset = 0.0f;
            observation.noiseAmplitude = 0.0f;
            observation.biasLower = 0.0f;
            observation.biasUpper = 0.0f;
            observation.normalizeVector3 = false;
        }
    };
    installSupportSense(observations.actorFrame);
    installSupportSense(observations.critic);
    // Preserve the official G1 actor's exact five-frame, 96-value
    // proprioceptive prefix. Masked depth is appended below as a direct
    // device-observation suffix with its own sparse temporal offsets.
    observations.actorHistoryLength = 5u;
    for (TaskObservationOperatorSpec& observation : observations.actorFrame) {
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
    // Append each authored sole's local resultant and spatial pressure field
    // to the temporal proprioceptive frame. These 13 values are generic
    // support-patch components: force xyz, torque xyz, CoP xy, occupied area,
    // and the G1 pack's 2x2 row-major pressure grid. The bounds coincide with
    // the current single-box sole support polygon; another TaskPack may choose
    // any fixed grid up to the compiled limit without changing Metal code.
    constexpr std::array<float, 13u> supportPatchScales{
        0.002f, 0.002f, 0.002f,
        0.02f, 0.02f, 0.02f,
        10.0f, 10.0f,
        100.0f,
        2.0e-5f, 2.0e-5f, 2.0e-5f, 2.0e-5f,
    };
    const auto appendSupportPatches = [&supportPatchScales](
        std::vector<TaskObservationOperatorSpec>& observations
    ) {
        for (const std::string_view group : {
                 std::string_view{"left_foot_contact"},
                 std::string_view{"right_foot_contact"},
             }) {
            for (std::uint32_t component = 0u;
                 component < supportPatchScales.size();
                 ++component) {
                observations.push_back({
                    .source = TaskObservationSource::supportPatch,
                    .target = std::string{group},
                    .component = component,
                    .scale = supportPatchScales[component],
                });
            }
        }
    };
    appendSupportPatches(observations.actorFrame);
    appendSupportPatches(observations.critic);
    observations.visual = {
        .width = 16u,
        .height = 9u,
        .frameOffsets = {0u, 3u, 8u, 18u},
        .nearDepthMeters = 0.1f,
        .farDepthMeters = 5.0f,
        // At 16x9 a physical ball may cover only one winner pixel. Expand
        // that exact segmented winner by one pixel so the deployed actor
        // receives a reliable shape-compatible signal. Harder difficulty
        // levels still scale the remaining sensor corruption.
        .fullDropoutProbability = 0.005f,
        .pixelDropoutProbability = 0.03f,
        .depthJitterMeters = 0.05f,
        .depthNoiseSigmaMeters = 0.01f,
        .edgeFlickerProbability = 1.0f,
        .difficultyCorruptionGain = 1.0f,
        .includeDerivedFeatures = true,
    };
    const std::uint32_t visualPixels =
        observations.visual.width * observations.visual.height;
    for (std::uint32_t frame = 0u;
         frame < observations.visual.frameOffsets.size();
         ++frame) {
        for (std::uint32_t pixel = 0u;
             pixel < visualPixels;
             ++pixel) {
            observations.actorFrame.push_back({
                .source = TaskObservationSource::maskedDepth,
                .component = frame * visualPixels + pixel,
            });
        }
    }
    for (std::uint32_t feature = 0u;
         feature < MR_TASK_MASKED_DEPTH_FEATURE_COUNT;
         ++feature) {
        observations.actorFrame.push_back({
            .source = TaskObservationSource::maskedDepth,
            .component =
                static_cast<std::uint32_t>(
                    observations.visual.frameOffsets.size()
                ) * visualPixels + feature,
        });
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
    task.rewards.push_back({
        .operation =
            TaskRewardOperator::projectilePredictedClearance,
        .weight = 2.0f,
        // Clearance scale for the urgency-weighted signed tanh reward.
        .parameters = {0.25f, 0.0f, 0.0f, 0.0f},
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
    for (TaskRandomizationOperatorSpec& random : reset.operators) {
        // Preserve the complete balance domain at level zero because the
        // resumed actor already depends on it. Retain only the parent's three
        // staged projectile-speed overrides; launch scheduling, base velocity,
        // and impact events must remain active from level zero.
        if (random.operation !=
                TaskRandomizationOperator::sceneBodyVelocity ||
            random.minimumDifficultyBand > 3u) {
            random.minimumDifficultyBand = 0u;
        }
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
    const LocomotionSurface surface
) {
    LocomotionWorld world;
    world.model = makeUnitreeG1EngineModel();
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
