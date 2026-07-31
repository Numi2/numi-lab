#include "metalrobo/LocomotionWorld.hpp"

#include "metalrobo/G1.hpp"

#include <algorithm>
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

TaskPack makeUnitreeG1LocomotionTaskPack(
    const LocomotionSurface surface
) {
    const G1ModelMetadata& metadata = unitreeG1Metadata();
    TaskPack task;
    task.id = "unitree_g1_locomotion";
    // The topology envelope contains every eligible self-collision pair.
    // Locomotion instead compiles an explicit operational arena. Capacity
    // overflow is transactional and reports the exact required stage count,
    // so this is a replayable task contract rather than silent truncation.
    task.capacities = {
        .candidatePairs = 256u,
        .rawContacts = 2048u,
        .manifolds = 256u,
        .constraintBlocks = 1024u,
        .constraintRows = 3072u,
        .hardConvexPairs = 256u,
        .meshTriangleCandidates = 2048u,
        .ccdCandidates = 256u,
        .ccdEvents = 8u,
        .endpointRuntimeRecords = 2048u,
        .articulationPointQueries = 2048u,
        .qualityRows = 3072u,
        .islandConstraintReferences = 1024u,
    };
    task.actorHistoryLength = 5u;
    task.maximumEpisodeSteps = 1000u;
    task.maximumActionDelaySteps = 2u;
    task.maximumObservationDelaySteps = 1u;
    task.curriculumLevelCount = 11u;
    task.baseHeightTarget = 0.78f;
    task.gaitPeriodSeconds = 0.8f;
    task.clearanceTarget = 0.10f;
    task.successTrackingThreshold = 0.8f;
    task.supportForceThreshold = 1.0f;
    task.commands.lower = {-0.5f, -0.3f, -0.2f, 0.0f};
    task.commands.upper = {1.0f, 0.3f, 0.2f, 0.0f};
    task.commands.standingProbability = 0.15f;
    task.commands.minimumDurationSeconds = 5.0f;
    task.commands.maximumDurationSeconds = 10.0f;
    task.pushes.maximumVelocity = 0.5f;
    task.pushes.minimumIntervalSeconds = 2.0f;
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
            0.04f,
            -0.004f,
            0.004f
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
            0.0f,
            true
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
            -0.005f,
            0.005f
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
        .stanceFraction = 0.5f,
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
        .stanceFraction = 0.5f,
    });
    task.contactGroups.push_back({
        .id = "lower_leg_contact",
        .bodies = bodyNames({4u, 10u}),
        .forbidden = true,
    });
    task.contactGroups.push_back({
        .id = "pelvis_contact",
        .bodies = bodyNames({0u}),
        .forbidden = true,
    });
    task.contactGroups.push_back({
        .id = "torso_contact",
        .bodies = bodyNames({15u}),
        .forbidden = true,
    });
    std::vector<std::string> arms;
    for (std::uint32_t body = 16u; body < 30u; ++body) {
        arms.emplace_back(metadata.bodyNames[body]);
    }
    task.contactGroups.push_back({
        .id = "arm_contact",
        .bodies = std::move(arms),
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

    for (std::uint32_t component = 0u;
         component < 3u;
         ++component) {
        task.critic.push_back(observation(
            TaskObservationSource::rootLinearVelocityLocal,
            {},
            component
        ));
    }
    task.critic.push_back(observation(
        TaskObservationSource::rootHeight,
        {},
        0u
    ));
    for (const std::string_view group :
         {"left_foot_contact", "right_foot_contact"}) {
        for (std::uint32_t component = 0u;
             component < 6u;
             ++component) {
            task.critic.push_back(observation(
                TaskObservationSource::contactMetric,
                group,
                component
            ));
        }
    }
    for (const std::string_view group :
         {
             "lower_leg_contact",
             "pelvis_contact",
             "torso_contact",
             "arm_contact",
         }) {
        task.critic.push_back(observation(
            TaskObservationSource::contactMetric,
            group,
            0u
        ));
    }

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
            task.critic.push_back(observation(
                TaskObservationSource::terrainHeight,
                {},
                static_cast<std::uint32_t>(
                    task.terrain.sampleOffsets.size() - 1u
                )
            ));
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
    task.critic.push_back(observation(
        TaskObservationSource::bodyParameterMean,
        {},
        0u
    ));
    task.critic.push_back(observation(
        TaskObservationSource::bodyParameter,
        "torso_link",
        0u
    ));
    for (std::uint32_t component = 1u;
         component < 4u;
         ++component) {
        task.critic.push_back(observation(
            TaskObservationSource::bodyParameterMean,
            {},
            component
        ));
    }
    for (std::uint32_t component = 0u;
         component < 4u;
         ++component) {
        task.critic.push_back(observation(
            TaskObservationSource::controllerParameter,
            {},
            component
        ));
    }

    const auto reward =
        [&task](
            const TaskRewardOperator operation,
            const float weight,
            const std::string_view group = {},
            const float parameter = 0.0f
        ) {
            task.rewards.push_back({
                .operation = operation,
                .sourceGroup = std::string{group},
                .weight = weight,
                .parameters = {parameter, 0.0f, 0.0f, 0.0f},
            });
        };
    reward(
        TaskRewardOperator::linearVelocityTracking,
        1.0f,
        {},
        0.25f
    );
    reward(
        TaskRewardOperator::yawVelocityTracking,
        0.5f,
        {},
        0.25f
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
    reward(TaskRewardOperator::tiltSquared, -5.0f);
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
        TaskRewardOperator::jointLimitViolationSquared,
        -5.0f
    );
    reward(
        TaskRewardOperator::mechanicalPower,
        -2.0e-5f
    );
    reward(
        TaskRewardOperator::jointGroupPostureSquared,
        -0.2f,
        "waist"
    );
    reward(
        TaskRewardOperator::jointGroupPostureSquared,
        -0.1f,
        "hips"
    );
    reward(
        TaskRewardOperator::jointGroupPostureSquared,
        -0.05f,
        "arms"
    );
    reward(
        TaskRewardOperator::gaitContactMatch,
        0.5f
    );
    reward(
        TaskRewardOperator::swingClearance,
        1.0f,
        {},
        0.0025f
    );
    reward(TaskRewardOperator::supportSlip, -0.1f);
    reward(TaskRewardOperator::forbiddenContact, -1.0f);

    task.terminations = {
        {
            .operation =
                TaskTerminationOperator::minimumRootHeight,
            .reason = MR_TASK_TERMINATION_HEIGHT,
            .priority = 1u,
            .threshold = 0.2f,
        },
        {
            .operation = TaskTerminationOperator::maximumTilt,
            .reason = MR_TASK_TERMINATION_TILT,
            .priority = 2u,
            .threshold = 0.8f,
        },
        {
            .operation = TaskTerminationOperator::contactGroup,
            .sourceGroup = "pelvis_contact",
            .reason = MR_TASK_TERMINATION_CONTACT,
            .priority = 3u,
            .threshold = 0.0f,
        },
        {
            .operation = TaskTerminationOperator::contactGroup,
            .sourceGroup = "torso_contact",
            .reason = MR_TASK_TERMINATION_CONTACT,
            .priority = 3u,
            .threshold = 0.0f,
        },
    };

    const auto random =
        [&task](
            const TaskRandomizationOperator operation,
            const std::string_view target,
            const std::uint32_t component,
            const mr_float4 parameters
        ) {
            task.randomization.push_back({
                .operation = operation,
                .target = std::string{target},
                .component = component,
                .parameters = parameters,
            });
        };
    random(
        TaskRandomizationOperator::rootPosition,
        {},
        0u,
        {0.015f, 0.015f, 0.0f, 0.0f}
    );
    random(
        TaskRandomizationOperator::rootYaw,
        {},
        0u,
        {-0.08f, 0.08f, 0.0f, 0.0f}
    );
    random(
        TaskRandomizationOperator::actionPosition,
        {},
        0u,
        {-0.025f, 0.025f, 0.0f, 0.0f}
    );
    random(
        TaskRandomizationOperator::velocity,
        {},
        0u,
        {-0.05f, 0.05f, 0.0f, 0.0f}
    );
    random(
        TaskRandomizationOperator::bodyParameter,
        "robot",
        0u,
        {0.9f, 1.1f, 0.0f, 0.0f}
    );
    random(
        TaskRandomizationOperator::bodyParameter,
        "robot",
        1u,
        {0.3f, 1.0f, 0.0f, 0.0f}
    );
    random(
        TaskRandomizationOperator::bodyParameter,
        "robot",
        2u,
        {0.9f, 1.1f, 0.0f, 0.0f}
    );
    random(
        TaskRandomizationOperator::bodyParameter,
        "robot",
        3u,
        {0.8f, 1.2f, 0.0f, 0.0f}
    );
    random(
        TaskRandomizationOperator::bodyPayload,
        "torso_link",
        0u,
        {-1.0f, 3.0f, 0.0f, 0.0f}
    );
    for (const std::uint32_t component : {0u, 1u, 3u}) {
        random(
            TaskRandomizationOperator::controllerParameter,
            {},
            component,
            {0.8f, 1.2f, 0.0f, 0.0f}
        );
    }
    random(
        TaskRandomizationOperator::actionDelay,
        {},
        0u,
        {0.0f, 2.0f, 0.0f, 0.0f}
    );
    random(
        TaskRandomizationOperator::observationDelay,
        {},
        0u,
        {0.0f, 1.0f, 0.0f, 0.0f}
    );

    return task;
}

LocomotionWorld makeUnitreeG1LocomotionWorld(
    const LocomotionSurface surface
) {
    LocomotionWorld world{
        .model = makeUnitreeG1EngineModel(),
        .task = makeUnitreeG1LocomotionTaskPack(surface),
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
