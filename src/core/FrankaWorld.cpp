#include "metalrobo/FrankaWorld.hpp"

#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/Franka.hpp"
#include "metalrobo/RunProgram.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

struct TactileWorldPose {
    mr_float4 position{};
    mr_float4 orientation{0.0f, 0.0f, 0.0f, 1.0f};
};

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

std::array<TactileSensorSpec, 2u> frankaTactileSensors() {
    constexpr float shellThickness = 0.003f;
    constexpr float squareRootHalf = 0.7071067811865476f;
    TactilePose pose;
    // Sensor +z maps to each finger's local -y (the inner pad face);
    // sensor +x/+y map to finger local +x/+z.
    pose.orientation = {
        squareRootHalf,
        0.0f,
        0.0f,
        squareRootHalf,
    };
    // Final pad box centre minus finger COM, then move from its inner rigid
    // face outward by the undeformed 3 mm tactile shell.
    pose.position = {
        0.0f,
        -0.018305f,
        0.0232825f,
        0.0f,
    };
    TactileSensorSpec left = makeFlatTactileSensor(
        "left_fingertip_tactile",
        9u,
        {27u},
        pose,
        32u,
        32u,
        0.0175f,
        0.0185f,
        shellThickness
    );
    left.targetShapeIndices = {32u};
    left.queryEpsilonMeters = 5.0e-7f;
    TactileSensorSpec right = makeFlatTactileSensor(
        "right_fingertip_tactile",
        10u,
        {31u},
        pose,
        32u,
        32u,
        0.0175f,
        0.0185f,
        shellThickness
    );
    right.targetShapeIndices = {32u};
    right.queryEpsilonMeters = 5.0e-7f;
    return {std::move(left), std::move(right)};
}

std::array<TactileWorldPose, 2u> tactileWorldPoses(
    const EngineModel& model,
    const std::array<TactileSensorSpec, 2u>& sensors,
    const std::span<const double> q
) {
    std::vector<ArticulatedBodyKinematics> bodies(
        model.articulations.front().bodyCount
    );
    const std::vector<double> velocity(model.world.nv, 0.0);
    const auto status = computeArticulatedBodyKinematics(
        model,
        0u,
        q,
        velocity,
        bodies
    );
    if (!status.succeeded()) {
        throw std::logic_error(
            "could not derive authored Franka tactile reset"
        );
    }
    std::array<TactileWorldPose, 2u> result{};
    for (std::size_t index = 0u; index < result.size(); ++index) {
        const TactileSensorSpec& sensor = sensors[index];
        const ArticulatedBodyKinematics& body =
            bodies.at(sensor.parentBodyIndex);
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
        const mr_float4 offset = rotate(
            bodyOrientation,
            sensor.localPose.position
        );
        result[index].position = {
            bodyPosition.x + offset.x,
            bodyPosition.y + offset.y,
            bodyPosition.z + offset.z,
            0.0f,
        };
        result[index].orientation = quaternionMultiply(
            bodyOrientation,
            sensor.localPose.orientation
        );
    }
    return result;
}

void authorFrankaTactileReset(
    EngineModel& model,
    const std::array<TactileSensorSpec, 2u>& sensors
) {
    constexpr double targetGap = 2.0 * (0.025 - 0.0002);
    std::vector<double> q(
        model.defaultQ.begin(),
        model.defaultQ.end()
    );
    const auto gapAt = [&](const double fingerPosition) {
        q[7u] = fingerPosition;
        q[8u] = fingerPosition;
        const auto poses = tactileWorldPoses(model, sensors, q);
        const double x =
            poses[1u].position.x - poses[0u].position.x;
        const double y =
            poses[1u].position.y - poses[0u].position.y;
        const double z =
            poses[1u].position.z - poses[0u].position.z;
        return std::sqrt(x * x + y * y + z * z);
    };
    const double closedGap = gapAt(0.0);
    const double openGap = gapAt(0.04);
    if (!(closedGap < targetGap && targetGap < openGap)) {
        throw std::logic_error(
            "authored Franka tactile grasp is outside finger travel"
        );
    }
    double lower = 0.0;
    double upper = 0.04;
    for (std::uint32_t iteration = 0u; iteration < 48u; ++iteration) {
        const double middle = 0.5 * (lower + upper);
        if (gapAt(middle) < targetGap) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    const float finger =
        static_cast<float>(0.5 * (lower + upper));
    model.defaultQ[7u] = finger;
    model.defaultQ[8u] = finger;
}

mr_float4 cameraToward(
    const mr_float4 position,
    const mr_float4 target
) {
    const float x = target.x - position.x;
    const float y = target.y - position.y;
    const float z = target.z - position.z;
    const float inverseLength =
        1.0f / std::sqrt(x * x + y * y + z * z);
    const mr_float4 forward{
        x * inverseLength,
        y * inverseLength,
        z * inverseLength,
        0.0f,
    };
    mr_float4 orientation{
        -forward.y,
        forward.x,
        0.0f,
        1.0f + forward.z,
    };
    if (orientation.w < 1.0e-5f) {
        orientation = {1.0f, 0.0f, 0.0f, 0.0f};
    }
    const float inverseNorm = 1.0f / std::sqrt(
        orientation.x * orientation.x +
        orientation.y * orientation.y +
        orientation.z * orientation.z +
        orientation.w * orientation.w
    );
    orientation.x *= inverseNorm;
    orientation.y *= inverseNorm;
    orientation.z *= inverseNorm;
    orientation.w *= inverseNorm;
    return orientation;
}

WorldAsset makeAsset(
    std::string id,
    const MRWorldAssetRole role,
    const MRWorldRenderRepresentation render,
    const MRWorldCollisionRepresentation collision,
    const MRWorldDynamicsRepresentation dynamics
) {
    WorldAsset result;
    result.id = std::move(id);
    result.semanticClass = result.id;
    result.role = role;
    result.render = render;
    result.collision = collision;
    result.dynamics = dynamics;
    return result;
}

VariationParameter uniformVariation(
    std::string id,
    const MRWorldVariationAxis axis,
    const MRWorldVariationTarget target,
    std::string targetId,
    const float lower,
    const float upper
) {
    VariationParameter result;
    result.id = std::move(id);
    result.axis = axis;
    result.distribution = MR_WORLD_DISTRIBUTION_UNIFORM;
    result.target = target;
    result.targetId = std::move(targetId);
    result.parameters = {lower, upper, 0.0f, 0.0f};
    return result;
}

MRBodyPropertiesGPU sceneBody(
    const std::uint32_t motionType,
    const float mass,
    const mr_float4 halfExtents
) {
    MRBodyPropertiesGPU body{};
    body.articulationIndex = MR_INVALID_INDEX;
    body.parentBody = MR_INVALID_INDEX;
    body.inboundJoint = MR_INVALID_INDEX;
    body.motionType = motionType;
    if (motionType == MR_MOTION_DYNAMIC) {
        const float x = 2.0f * halfExtents.x;
        const float y = 2.0f * halfExtents.y;
        const float z = 2.0f * halfExtents.z;
        const float ixx = mass * (y * y + z * z) / 12.0f;
        const float iyy = mass * (x * x + z * z) / 12.0f;
        const float izz = mass * (x * x + y * y) / 12.0f;
        body.massAndInverseMass =
            {mass, 1.0f / mass, 0.0f, 0.0f};
        body.inertiaRow0 = {ixx, 0.0f, 0.0f, 0.0f};
        body.inertiaRow1 = {0.0f, iyy, 0.0f, 0.0f};
        body.inertiaRow2 = {0.0f, 0.0f, izz, 0.0f};
        body.inverseInertiaRow0 =
            {1.0f / ixx, 0.0f, 0.0f, 0.0f};
        body.inverseInertiaRow1 =
            {0.0f, 1.0f / iyy, 0.0f, 0.0f};
        body.inverseInertiaRow2 =
            {0.0f, 0.0f, 1.0f / izz, 0.0f};
        body.dampingAndSpeedLimits =
            {0.02f, 0.02f, 10.0f, 50.0f};
    } else {
        body.dampingAndSpeedLimits =
            {0.0f, 0.0f, 1.0e6f, 1.0e6f};
    }
    return body;
}

MRShapeGPU sceneBox(
    const std::uint32_t body,
    const std::uint32_t material,
    const mr_float4 halfExtents,
    const std::uint32_t generation
) {
    MRShapeGPU shape{};
    shape.bodyIndex = body;
    shape.shapeType = MR_SHAPE_BOX;
    shape.materialIndex = material;
    shape.collisionGroup = 1u;
    shape.collisionMask = ~0u;
    shape.slotGeneration = generation;
    shape.localPosition.w = 1.0f;
    shape.localRotation.w = 1.0f;
    shape.dimensions = halfExtents;
    shape.contactRestAndBoundingRadius = {
        0.001f,
        0.0f,
        std::sqrt(
            halfExtents.x * halfExtents.x +
            halfExtents.y * halfExtents.y +
            halfExtents.z * halfExtents.z
        ),
        0.0f,
    };
    return shape;
}

MRBodyStateGPU sceneState(
    const std::uint32_t body,
    const std::uint32_t motionType,
    const mr_float4 position
) {
    MRBodyStateGPU state{};
    state.position = {
        position.x,
        position.y,
        position.z,
        1.0f,
    };
    state.orientation.w = 1.0f;
    state.flagsAndIndices[0] = motionType;
    state.flagsAndIndices[1] = MR_INVALID_INDEX;
    state.flagsAndIndices[2] = body;
    return state;
}

void setDynamicBoxInertia(
    MRBodyPropertiesGPU& body,
    const mr_float4 halfExtents
) {
    const float mass = body.massAndInverseMass.x;
    const float x = 2.0f * halfExtents.x;
    const float y = 2.0f * halfExtents.y;
    const float z = 2.0f * halfExtents.z;
    const float ixx = mass * (y * y + z * z) / 12.0f;
    const float iyy = mass * (x * x + z * z) / 12.0f;
    const float izz = mass * (x * x + y * y) / 12.0f;
    body.inertiaRow0 = {ixx, 0.0f, 0.0f, 0.0f};
    body.inertiaRow1 = {0.0f, iyy, 0.0f, 0.0f};
    body.inertiaRow2 = {0.0f, 0.0f, izz, 0.0f};
    body.inverseInertiaRow0 =
        {1.0f / ixx, 0.0f, 0.0f, 0.0f};
    body.inverseInertiaRow1 =
        {0.0f, 1.0f / iyy, 0.0f, 0.0f};
    body.inverseInertiaRow2 =
        {0.0f, 0.0f, 1.0f / izz, 0.0f};
}

} // namespace

EngineModel makeFrankaPickPlaceSceneEngineModel() {
    const EngineModel robot = makeFrankaPandaHandEngineModel();
    EngineModel scene;
    scene.name = "franka_pick_place_scene_v1";
    scene.world.abiVersion = MR_ENGINE_ABI_VERSION;
    scene.world.gravityAndTimestep = robot.world.gravityAndTimestep;
    scene.world.solverScales = robot.world.solverScales;
    scene.world.solverType = robot.world.solverType;
    scene.world.frictionConeType = robot.world.frictionConeType;
    scene.world.pairCapacity = 32u;
    scene.world.contactCapacity = 64u;
    scene.world.constraintCapacity = 128u;
    scene.world.islandCapacity = 4u;

    MRMaterialGPU sceneMaterial = robot.materials.front();
    sceneMaterial.friction = {0.7f, 0.6f, 0.0f, 0.0f};
    sceneMaterial.response = {0.05f, 0.15f, 0.0f, 0.0f};
    scene.materials.push_back(sceneMaterial);
    scene.bodies = {
        sceneBody(MR_MOTION_DYNAMIC, 0.10f,
            {0.025f, 0.025f, 0.025f, 0.0f}),
        sceneBody(MR_MOTION_STATIC, 0.0f, {}),
        sceneBody(MR_MOTION_STATIC, 0.0f, {}),
        sceneBody(MR_MOTION_DYNAMIC, 0.08f,
            {0.03f, 0.03f, 0.03f, 0.0f}),
    };
    scene.shapes.push_back(sceneBox(
        0u, 0u, {0.025f, 0.025f, 0.025f, 0.0f}, 1001u
    ));
    MRShapeGPU ground{};
    ground.bodyIndex = 1u;
    ground.shapeType = MR_SHAPE_PLANE;
    ground.materialIndex = 0u;
    ground.collisionGroup = 4u;
    ground.collisionMask = 1u;
    ground.slotGeneration = 1002u;
    ground.localPosition.w = 1.0f;
    constexpr float kSqrtHalf = 0.7071067811865476f;
    ground.localRotation = {kSqrtHalf, 0.0f, 0.0f, kSqrtHalf};
    scene.shapes.push_back(ground);
    MRShapeGPU target = sceneBox(
        2u, 0u, {0.06f, 0.06f, 0.004f, 0.0f}, 1003u
    );
    target.collisionGroup = 4u;
    target.collisionMask = 1u;
    scene.shapes.push_back(target);
    scene.shapes.push_back(sceneBox(
        3u, 0u, {0.03f, 0.03f, 0.03f, 0.0f}, 1004u
    ));
    scene.bodyNames = {
        "pick_object", "workspace", "target_fixture", "clutter"
    };
    scene.world.bodyCount = static_cast<std::uint32_t>(scene.bodies.size());
    scene.world.shapeCount = static_cast<std::uint32_t>(scene.shapes.size());
    scene.world.materialCount =
        static_cast<std::uint32_t>(scene.materials.size());
    std::string reason;
    if (!scene.valid(&reason)) {
        throw std::logic_error(
            "Franka pick-place scene compilation failed: " + reason
        );
    }
    return scene;
}

ScenePack makeFrankaPickPlaceScenePack() {
    const EngineModel authored = makeFrankaPickPlaceSceneEngineModel();
    const auto mechanics = [&](const std::uint32_t index) {
        EngineModel model;
        model.name = "franka_scene_" + authored.bodyNames[index];
        model.world = authored.world;
        model.world.bodyCount = 1u;
        model.world.shapeCount = 1u;
        model.world.materialCount = 1u;
        model.bodies.push_back(authored.bodies[index]);
        MRShapeGPU shape = authored.shapes[index];
        shape.bodyIndex = 0u;
        shape.materialIndex = 0u;
        model.shapes.push_back(shape);
        model.materials.push_back(authored.materials.front());
        model.bodyNames.push_back(authored.bodyNames[index]);
        std::string reason;
        if (!model.valid(&reason)) {
            throw std::logic_error(
                "Franka scene object compilation failed: " + reason
            );
        }
        return model;
    };
    const auto state = [](const std::uint32_t motion, const mr_float4 pose) {
        return std::vector<MRBodyStateGPU>{sceneState(0u, motion, pose)};
    };
    ScenePack scene;
    scene.id = "franka_pick_place_scene";
    scene.objects = {
        {
            .id = "pick_object",
            .semanticClass = "manipulated_object",
            .role = MR_WORLD_ASSET_MANIPULATED,
            .render = MR_WORLD_RENDER_MESH_PBR,
            .collision = MR_WORLD_COLLISION_PRIMITIVES,
            .dynamics = MR_WORLD_DYNAMICS_RIGID,
            .mechanics = mechanics(0u),
            .defaultBodyStates = state(
                MR_MOTION_DYNAMIC, {0.50f, 0.0f, 0.025f, 0.0f}
            ),
        },
        {
            .id = "workspace",
            .semanticClass = "support_surface",
            .role = MR_WORLD_ASSET_BACKGROUND,
            .render = MR_WORLD_RENDER_NONE,
            .collision = MR_WORLD_COLLISION_PRIMITIVES,
            .dynamics = MR_WORLD_DYNAMICS_STATIC,
            .mechanics = mechanics(1u),
            .defaultBodyStates = state(
                MR_MOTION_STATIC, {0.0f, 0.0f, 0.0f, 0.0f}
            ),
        },
        {
            .id = "target_fixture",
            .semanticClass = "placement_target",
            .role = MR_WORLD_ASSET_FIXTURE,
            .render = MR_WORLD_RENDER_MESH_PBR,
            .collision = MR_WORLD_COLLISION_PRIMITIVES,
            .dynamics = MR_WORLD_DYNAMICS_STATIC,
            .mechanics = mechanics(2u),
            .defaultBodyStates = state(
                MR_MOTION_STATIC, {0.45f, 0.25f, 0.0f, 0.0f}
            ),
        },
        {
            .id = "clutter",
            .semanticClass = "dynamic_clutter",
            .role = MR_WORLD_ASSET_CLUTTER,
            .render = MR_WORLD_RENDER_MESH_PBR,
            .collision = MR_WORLD_COLLISION_PRIMITIVES,
            .dynamics = MR_WORLD_DYNAMICS_RIGID,
            .mechanics = mechanics(3u),
            .defaultBodyStates = state(
                MR_MOTION_DYNAMIC, {0.35f, -0.25f, 0.03f, 0.0f}
            ),
        },
    };
    return scene;
}

EpisodeTwin makeFrankaPickPlaceEpisodeTwin() {
    EpisodeTwin episode;
    episode.id = "franka_pick_place_anchor_v1";

    WorldAsset background = makeAsset(
        "workspace",
        MR_WORLD_ASSET_BACKGROUND,
        MR_WORLD_RENDER_GAUSSIAN_FIELD,
        MR_WORLD_COLLISION_TRIANGLE_MESH,
        MR_WORLD_DYNAMICS_STATIC
    );
    background.bodyIndices = {12u};
    background.shapeIndices = {33u};
    background.materialIndices = {2u};
    WorldAsset robot = makeAsset(
        "franka",
        MR_WORLD_ASSET_ROBOT,
        MR_WORLD_RENDER_GAUSSIAN_FIELD,
        MR_WORLD_COLLISION_PRIMITIVES,
        MR_WORLD_DYNAMICS_ARTICULATED
    );
    robot.articulationIndex = 0u;
    for (std::uint32_t body = 0u; body < 11u; ++body) {
        robot.bodyIndices.push_back(body);
    }
    for (std::uint32_t shape = 0u; shape < 32u; ++shape) {
        robot.shapeIndices.push_back(shape);
    }
    robot.materialIndices = {0u, 1u};
    WorldAsset object = makeAsset(
        "pick_object",
        MR_WORLD_ASSET_MANIPULATED,
        MR_WORLD_RENDER_MESH_PBR,
        MR_WORLD_COLLISION_CONVEX,
        MR_WORLD_DYNAMICS_RIGID
    );
    object.bodyIndices = {11u};
    object.shapeIndices = {32u};
    object.materialIndices = {2u};
    object.initialPose.position = {0.50f, 0.0f, 0.025f, 0.0f};
    WorldAsset target = makeAsset(
        "target_fixture",
        MR_WORLD_ASSET_FIXTURE,
        MR_WORLD_RENDER_MESH_PBR,
        MR_WORLD_COLLISION_TRIANGLE_MESH,
        MR_WORLD_DYNAMICS_STATIC
    );
    target.bodyIndices = {13u};
    target.shapeIndices = {34u};
    target.materialIndices = {2u};
    target.initialPose.position = {0.45f, 0.25f, 0.0f, 0.0f};
    target.anchors.push_back({
        "place_pose",
        {
            {0.0f, 0.0f, 0.025f, 0.0f},
            {0.0f, 0.0f, 0.0f, 1.0f},
        },
        0.025f,
        0u,
    });
    WorldAsset clutter = makeAsset(
        "clutter",
        MR_WORLD_ASSET_CLUTTER,
        MR_WORLD_RENDER_MESH_PBR,
        MR_WORLD_COLLISION_CONVEX,
        MR_WORLD_DYNAMICS_RIGID
    );
    clutter.bodyIndices = {14u};
    clutter.shapeIndices = {35u};
    clutter.materialIndices = {2u};
    clutter.initialPose.position = {0.35f, -0.25f, 0.03f, 0.0f};
    episode.assets = {
        std::move(background),
        std::move(robot),
        std::move(object),
        std::move(target),
        std::move(clutter),
    };

    SensorSpec fixedCamera;
    fixedCamera.id = "fixed_rgbd";
    fixedCamera.parentAssetId = "workspace";
    fixedCamera.kind = MR_WORLD_SENSOR_RGBD;
    fixedCamera.localPose.position = {0.8f, -0.6f, 0.8f, 0.0f};
    fixedCamera.localPose.orientation = cameraToward(
        fixedCamera.localPose.position,
        {0.45f, 0.0f, 0.08f, 0.0f}
    );
    fixedCamera.width = 160u;
    fixedCamera.height = 120u;
    fixedCamera.intrinsics = {140.0f, 140.0f, 80.0f, 60.0f};

    SensorSpec wristCamera = fixedCamera;
    wristCamera.id = "wrist_rgbd";
    wristCamera.parentAssetId = "franka";
    wristCamera.parentKind =
        MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK;
    wristCamera.parentBodyIndex = 10u;
    wristCamera.localPose.position = {0.0f, 0.0f, 0.08f, 0.0f};
    wristCamera.localPose.orientation =
        {0.0f, 0.0f, 0.0f, 1.0f};
    episode.sensors = {
        std::move(fixedCamera),
        std::move(wristCamera),
    };
    episode.artifacts = {
        {
            "capture",
            EpisodeArtifactKind::capture,
            EpisodeArtifactProducer::measured,
            "",
            "capture://franka-pick-place",
            "sha256:capture",
            0.0,
            15.0,
        },
        {
            "trajectory",
            EpisodeArtifactKind::robotTrajectory,
            EpisodeArtifactProducer::measured,
            "franka",
            "trajectory://franka",
            "sha256:trajectory",
            0.0,
            15.0,
        },
    };
    episode.task = {
        "pick_place",
        "franka",
        "pick_object",
        "target_fixture",
        "place_pose",
        0.05,
        15.0,
    };
    return episode;
}

EngineModel makeFrankaPickPlaceEngineModel() {
    EngineModel model = makeFrankaPandaHandEngineModel();
    model.name = "franka_fer_hand_pick_place_world_v1";

    MRMaterialGPU sceneMaterial = model.materials.front();
    sceneMaterial.friction = {0.7f, 0.6f, 0.0f, 0.0f};
    sceneMaterial.response = {0.05f, 0.15f, 0.0f, 0.0f};
    model.materials.push_back(sceneMaterial);

    model.bodies.push_back(sceneBody(
        MR_MOTION_DYNAMIC,
        0.10f,
        {0.025f, 0.025f, 0.025f, 0.0f}
    ));
    model.bodies.push_back(sceneBody(
        MR_MOTION_STATIC,
        0.0f,
        {}
    ));
    model.bodies.push_back(sceneBody(
        MR_MOTION_STATIC,
        0.0f,
        {}
    ));
    model.bodies.push_back(sceneBody(
        MR_MOTION_DYNAMIC,
        0.08f,
        {0.03f, 0.03f, 0.03f, 0.0f}
    ));
    model.bodyNames.insert(
        model.bodyNames.end(),
        {"pick_object", "workspace", "target_fixture", "clutter"}
    );

    model.shapes.push_back(sceneBox(
        11u,
        2u,
        {0.025f, 0.025f, 0.025f, 0.0f},
        1001u
    ));
    MRShapeGPU ground{};
    ground.bodyIndex = 12u;
    ground.shapeType = MR_SHAPE_PLANE;
    ground.materialIndex = 2u;
    // Support surfaces collide with free objects but not the fixed-base
    // robot mount. Robot/object interaction remains enabled.
    ground.collisionGroup = 4u;
    ground.collisionMask = 1u;
    ground.slotGeneration = 1002u;
    ground.localPosition.w = 1.0f;
    constexpr float kSqrtHalf = 0.7071067811865476f;
    ground.localRotation = {
        kSqrtHalf,
        0.0f,
        0.0f,
        kSqrtHalf,
    };
    model.shapes.push_back(ground);
    MRShapeGPU target = sceneBox(
        13u,
        2u,
        {0.06f, 0.06f, 0.004f, 0.0f},
        1003u
    );
    target.collisionGroup = 4u;
    target.collisionMask = 1u;
    model.shapes.push_back(target);
    model.shapes.push_back(sceneBox(
        14u,
        2u,
        {0.03f, 0.03f, 0.03f, 0.0f},
        1004u
    ));

    model.world.bodyCount =
        static_cast<std::uint32_t>(model.bodies.size());
    model.world.shapeCount =
        static_cast<std::uint32_t>(model.shapes.size());
    model.world.materialCount =
        static_cast<std::uint32_t>(model.materials.size());
    model.world.pairCapacity = 256u;
    model.world.contactCapacity = 128u;
    model.world.constraintCapacity = 256u;
    model.world.islandCapacity = 8u;
    std::string reason;
    if (!model.valid(&reason)) {
        throw std::logic_error(
            "Franka pick-place model compilation failed: " + reason
        );
    }
    return model;
}

std::vector<MRBodyStateGPU> makeFrankaPickPlaceSceneState() {
    return {
        sceneState(
            11u,
            MR_MOTION_DYNAMIC,
            {0.50f, 0.0f, 0.025f, 0.0f}
        ),
        sceneState(
            12u,
            MR_MOTION_STATIC,
            {0.0f, 0.0f, 0.0f, 0.0f}
        ),
        sceneState(
            13u,
            MR_MOTION_STATIC,
            {0.45f, 0.25f, 0.0f, 0.0f}
        ),
        sceneState(
            14u,
            MR_MOTION_DYNAMIC,
            {0.35f, -0.25f, 0.03f, 0.0f}
        ),
    };
}

TaskPack makeFrankaPickPlaceTaskPack() {
    TaskPack task;
    task.id = "franka_pick_place_v1";
    task.outcomes = {
        {"grasp_reward", "reward", TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::objectGrasp},
        {"lift_reward", "reward", TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::objectLift},
        {"object_position_reward", "reward",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::objectPosition},
        {"placement_reward", "reward",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::objectPlacement},
    };
    const std::array<std::string_view, 9u> joints{{
        "panda_joint1", "panda_joint2", "panda_joint3",
        "panda_joint4", "panda_joint5", "panda_joint6",
        "panda_joint7", "panda_finger_joint1",
        "panda_finger_joint2",
    }};
    for (const std::string_view joint : joints) {
        const bool finger = joint.find("finger") != std::string_view::npos;
        task.actions.push_back({
            std::string{joint}, finger ? 0.01f : 0.25f, 0.04f
        });
        for (const TaskObservationSource source : {
                 TaskObservationSource::jointPositionError,
                 TaskObservationSource::jointVelocity,
                 TaskObservationSource::previousAction,
             }) {
            task.actorFrame.push_back({
                .source = source,
                .target = std::string{joint},
            });
            task.critic.push_back({
                .source = source,
                .target = std::string{joint},
            });
        }
    }
    for (std::uint32_t component = 0u; component < 7u; ++component) {
        TaskObservationOperatorSpec object{
            .source = TaskObservationSource::objectTrack,
            .target = "pick_object",
            .component = component,
        };
        task.actorFrame.push_back(object);
        task.critic.push_back(std::move(object));
    }
    task.actorHistoryLength = 3u;
    task.criticHistoryLength = 3u;
    task.contactGroups.push_back({
        .id = "gripper",
        .bodies = {"panda_leftfinger", "panda_rightfinger"},
        .referenceBody = "panda_hand",
    });
    task.jointGroups.push_back({
        .id = "arm",
        .joints = {
            "panda_joint1", "panda_joint2", "panda_joint3",
            "panda_joint4", "panda_joint5", "panda_joint6",
            "panda_joint7",
        },
    });
    task.rewards = {
        {TaskRewardOperator::constant, {}, {}, 0.02f, {}},
        {TaskRewardOperator::objectGrasp, "gripper", "pick_object",
            2.0f, {2.0f, 2.0f, 0.0f, 0.0f}},
        {TaskRewardOperator::objectLift, {}, "pick_object",
            3.0f, {0.03f, 0.18f, 0.0f, 0.0f}},
        {TaskRewardOperator::objectPosition, {}, "pick_object",
            1.5f, {0.45f, 0.25f, 0.025f, 0.02f}},
        {TaskRewardOperator::objectPlacement, "target_fixture", "pick_object",
            5.0f, {0.01f, 0.01f, 0.04f, 0.0f}},
        {TaskRewardOperator::actionRateSquared, {}, {}, -0.01f, {}},
        {TaskRewardOperator::jointLimitViolationSquared, {}, {}, -0.2f, {}},
        {TaskRewardOperator::mechanicalPower, {}, {}, -0.0002f, {}},
    };
    task.randomization = {
        {TaskRandomizationOperator::sceneBodyPosition, "pick_object", 0u,
            0u, {-0.08f, 0.08f, 0.0f, 0.0f}},
        {TaskRandomizationOperator::sceneBodyPosition, "pick_object", 1u,
            0u, {-0.08f, 0.08f, 0.0f, 0.0f}},
    };
    task.maximumEpisodeSteps = 750u;
    task.maximumActionDelaySteps = 2u;
    task.maximumObservationDelaySteps = 2u;
    task.difficultyBandCount = 4u;
    task.commands.minimumDurationSeconds = 15.0f;
    task.commands.maximumDurationSeconds = 15.0f;
    return task;
}

WorldProgram makeFrankaPickPlaceWorldProgram() {
    WorldProgram program;
    program.id = "systematic_pick_place_family_v1";
    program.variations = {
        uniformVariation(
            "exposure",
            MR_WORLD_VARIATION_APPEARANCE,
            MR_WORLD_TARGET_APPEARANCE_EXPOSURE,
            "default",
            -1.5f,
            1.5f
        ),
        uniformVariation(
            "light_intensity",
            MR_WORLD_VARIATION_APPEARANCE,
            MR_WORLD_TARGET_APPEARANCE_LIGHT_INTENSITY,
            "default",
            0.65f,
            1.35f
        ),
        uniformVariation(
            "object_x",
            MR_WORLD_VARIATION_OBJECT_CONFIGURATION,
            MR_WORLD_TARGET_ASSET_POSITION_X,
            "pick_object",
            -0.15f,
            0.15f
        ),
        uniformVariation(
            "object_y",
            MR_WORLD_VARIATION_OBJECT_CONFIGURATION,
            MR_WORLD_TARGET_ASSET_POSITION_Y,
            "pick_object",
            -0.15f,
            0.15f
        ),
        uniformVariation(
            "mass_inertia",
            MR_WORLD_VARIATION_PHYSICS,
            MR_WORLD_TARGET_ASSET_MASS_SCALE,
            "pick_object",
            0.65f,
            1.5f
        ),
        uniformVariation(
            "friction",
            MR_WORLD_VARIATION_PHYSICS,
            MR_WORLD_TARGET_ASSET_FRICTION_SCALE,
            "pick_object",
            0.5f,
            1.5f
        ),
        uniformVariation(
            "restitution",
            MR_WORLD_VARIATION_PHYSICS,
            MR_WORLD_TARGET_ASSET_RESTITUTION_SCALE,
            "pick_object",
            0.25f,
            1.75f
        ),
        uniformVariation(
            "body_damping",
            MR_WORLD_VARIATION_PHYSICS,
            MR_WORLD_TARGET_ASSET_DAMPING_SCALE,
            "pick_object",
            0.5f,
            1.5f
        ),
        uniformVariation(
            "robot_gain",
            MR_WORLD_VARIATION_ROBOT_STATE,
            MR_WORLD_TARGET_ROBOT_GAIN_SCALE,
            "franka",
            0.8f,
            1.2f
        ),
        uniformVariation(
            "robot_damping",
            MR_WORLD_VARIATION_ROBOT_STATE,
            MR_WORLD_TARGET_ROBOT_DAMPING_SCALE,
            "franka",
            0.8f,
            1.2f
        ),
        uniformVariation(
            "robot_latency",
            MR_WORLD_VARIATION_ROBOT_STATE,
            MR_WORLD_TARGET_ROBOT_LATENCY_SECONDS,
            "franka",
            0.0f,
            0.025f
        ),
        uniformVariation(
            "payload_compensation",
            MR_WORLD_VARIATION_ROBOT_STATE,
            MR_WORLD_TARGET_ROBOT_PAYLOAD_SCALE,
            "franka",
            0.75f,
            1.25f
        ),
        uniformVariation(
            "camera_yaw",
            MR_WORLD_VARIATION_CAMERA,
            MR_WORLD_TARGET_SENSOR_ORIENTATION_YAW,
            "fixed_rgbd",
            -0.15f,
            0.15f
        ),
        uniformVariation(
            "camera_focal_scale",
            MR_WORLD_VARIATION_CAMERA,
            MR_WORLD_TARGET_SENSOR_FOCAL_SCALE,
            "fixed_rgbd",
            0.95f,
            1.05f
        ),
        uniformVariation(
            "camera_latency",
            MR_WORLD_VARIATION_CAMERA,
            MR_WORLD_TARGET_SENSOR_LATENCY_SECONDS,
            "fixed_rgbd",
            0.0f,
            0.035f
        ),
        uniformVariation(
            "depth_noise",
            MR_WORLD_VARIATION_CAMERA,
            MR_WORLD_TARGET_SENSOR_DEPTH_NOISE,
            "fixed_rgbd",
            0.0f,
            0.008f
        ),
        uniformVariation(
            "depth_dropout",
            MR_WORLD_VARIATION_CAMERA,
            MR_WORLD_TARGET_SENSOR_DEPTH_DROPOUT,
            "fixed_rgbd",
            0.0f,
            0.08f
        ),
    };
    VariationParameter clutter;
    clutter.id = "clutter_set";
    clutter.axis = MR_WORLD_VARIATION_CLUTTER;
    clutter.distribution = MR_WORLD_DISTRIBUTION_CATEGORICAL;
    clutter.target = MR_WORLD_TARGET_CLUTTER_SET;
    clutter.targetId = "clutter";
    clutter.categoricalValues = {0u, 1u, 2u, 3u};
    program.variations.push_back(std::move(clutter));
    return program;
}

EngineModel makeFrankaTactileEngineModel() {
    EngineModel model = makeFrankaPickPlaceEngineModel();
    model.name = "franka_fer_hand_tactile_shell_world";
    const auto sensors = frankaTactileSensors();
    constexpr float shellThickness = 0.003f;
    constexpr float contactOffset = 0.0035f;
    for (const std::uint32_t shapeIndex : {27u, 31u}) {
        if (shapeIndex >= model.shapes.size() ||
            model.shapes[shapeIndex].shapeType != MR_SHAPE_BOX ||
            model.shapes[shapeIndex].materialIndex != 1u) {
            throw std::logic_error(
                "Franka tactile pad topology no longer matches its "
                "pinned shape indices"
            );
        }
        model.shapes[shapeIndex].
            contactRestAndBoundingRadius.x = contactOffset;
        model.shapes[shapeIndex].
            contactRestAndBoundingRadius.y = shellThickness;
    }
    // The legacy palm capsules are deliberately broad collision proxies and
    // overlap the usable opening near the fingertip pads. They would contact
    // this narrow coupon before the tactile surfaces and inject a physically
    // unrelated impulse, so the tactile task disables those two proxies while
    // retaining the explicit finger boxes and all arm collision geometry.
    for (const std::uint32_t palmShapeIndex : {22u, 23u}) {
        model.shapes[palmShapeIndex].collisionMask = 0u;
    }
    // The tactile task uses a narrow grasp coupon instead of the pick-place
    // cube. Its 50 mm normal span exercises both pads while the reduced
    // tangent span keeps the coupon clear of the palm collision capsules.
    constexpr mr_float4 couponHalfExtents{
        0.006f,
        0.007f,
        0.025f,
        0.0f,
    };
    model.shapes[32u].dimensions = couponHalfExtents;
    model.shapes[32u].contactRestAndBoundingRadius.z =
        std::sqrt(
            couponHalfExtents.x * couponHalfExtents.x +
            couponHalfExtents.y * couponHalfExtents.y +
            couponHalfExtents.z * couponHalfExtents.z
        );
    setDynamicBoxInertia(model.bodies[11u], couponHalfExtents);
    if (model.materials.size() <= 1u) {
        throw std::logic_error(
            "Franka tactile rubber material is missing"
        );
    }
    // Position-level normal compliance (m/N). This first native profile is a
    // stable engineering prior, not a claim of measured GelSight mechanics;
    // physical sensor calibration should replace it.
    model.materials[1u].response.z = 2.0e-6f;
    model.materials[1u].response.w = 0.02f;
    authorFrankaTactileReset(model, sensors);
    std::string reason;
    if (!model.valid(&reason)) {
        throw std::logic_error(
            "Franka tactile model compilation failed: " + reason
        );
    }
    return model;
}

EpisodeTwin makeFrankaTactileEpisodeTwin() {
    EpisodeTwin episode = makeFrankaPickPlaceEpisodeTwin();
    episode.id = "franka_tactile_grasp_stabilization";
    episode.task.id = "tactile_grasp_stabilization";
    const EngineModel model = makeFrankaTactileEngineModel();
    const auto sensors = frankaTactileSensors();
    episode.tactileSensors.assign(
        sensors.begin(),
        sensors.end()
    );
    const std::vector<double> q(
        model.defaultQ.begin(),
        model.defaultQ.end()
    );
    const auto poses = tactileWorldPoses(
        model,
        sensors,
        q
    );
    const auto manipulated = std::ranges::find_if(
        episode.assets,
        [&](const WorldAsset& asset) {
            return asset.id == episode.task.manipulatedAssetId;
        }
    );
    if (manipulated == episode.assets.end()) {
        throw std::logic_error(
            "Franka tactile task has no manipulated asset"
        );
    }
    manipulated->initialPose.position = {
        0.5f * (poses[0u].position.x + poses[1u].position.x),
        0.5f * (poses[0u].position.y + poses[1u].position.y),
        0.5f * (poses[0u].position.z + poses[1u].position.z),
        0.0f,
    };
    manipulated->initialPose.orientation = poses[0u].orientation;
    return episode;
}

} // namespace metalrobo
