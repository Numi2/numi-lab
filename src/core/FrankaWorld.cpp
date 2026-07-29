#include "metalrobo/FrankaWorld.hpp"

#include "metalrobo/Franka.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace metalrobo {
namespace {

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

} // namespace

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
    fixedCamera.width = 160u;
    fixedCamera.height = 120u;
    fixedCamera.intrinsics = {140.0f, 140.0f, 80.0f, 60.0f};

    SensorSpec wristCamera = fixedCamera;
    wristCamera.id = "wrist_rgbd";
    wristCamera.parentAssetId = "franka";
    wristCamera.localPose.position = {0.0f, 0.0f, 0.08f, 0.0f};
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

} // namespace metalrobo
