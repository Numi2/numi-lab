#include "metalrobo/EmbodiedTactile.hpp"

#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/G1.hpp"
#include "metalrobo/SurgicalAssets.hpp"
#include "metalrobo/SurgicalPSM.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

constexpr float kG1ShellThickness = 0.002f;
constexpr float kG1ContactOffset = 0.0025f;
constexpr float kPsmShellThickness = 0.00025f;
constexpr float kPsmContactOffset = 0.00030f;
constexpr std::uint32_t kAtlasWidth = 32u;
constexpr std::uint32_t kAtlasHeight = 32u;

WorldAsset asset(
    std::string id,
    const MRWorldAssetRole role,
    const MRWorldCollisionRepresentation collision,
    const MRWorldDynamicsRepresentation dynamics
) {
    WorldAsset result;
    result.id = std::move(id);
    result.semanticClass = result.id;
    result.role = role;
    result.render = MR_WORLD_RENDER_NONE;
    result.collision = collision;
    result.dynamics = dynamics;
    return result;
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

MRShapeGPU staticBox(
    const std::uint32_t bodyIndex,
    const std::uint32_t materialIndex,
    const mr_float4 halfExtents,
    const std::uint32_t generation
) {
    MRShapeGPU shape{};
    shape.bodyIndex = bodyIndex;
    shape.shapeType = MR_SHAPE_BOX;
    shape.materialIndex = materialIndex;
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

mr_float4 normalized(const mr_float4 value) {
    const float length = std::sqrt(
        value.x * value.x +
        value.y * value.y +
        value.z * value.z
    );
    if (!(length > 0.0f)) {
        throw std::logic_error("tactile atlas vector is degenerate");
    }
    return {
        value.x / length,
        value.y / length,
        value.z / length,
        0.0f,
    };
}

float dot3(const mr_float4 left, const mr_float4 right) {
    return
        left.x * right.x +
        left.y * right.y +
        left.z * right.z;
}

mr_float4 cross3(const mr_float4 left, const mr_float4 right) {
    return {
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
        0.0f,
    };
}

mr_float4 tangentFromReference(
    const mr_float4 normal,
    const mr_float4 reference
) {
    return normalized({
        reference.x - normal.x * dot3(normal, reference),
        reference.y - normal.y * dot3(normal, reference),
        reference.z - normal.z * dot3(normal, reference),
        0.0f,
    });
}

TactileSampleSpec invalidSample(
    const std::uint32_t u,
    const std::uint32_t v
) {
    TactileSampleSpec sample;
    sample.atlasU = u;
    sample.atlasV = v;
    sample.valid = false;
    return sample;
}

TactileSensorSpec makeG1SoleSensor(
    const EngineModel& model,
    const G1FootFrame& foot,
    const std::uint32_t targetShape
) {
    if (foot.soleShapeIndex >= model.shapes.size() ||
        model.shapes[foot.soleShapeIndex].bodyIndex != foot.bodyIndex ||
        model.shapes[foot.soleShapeIndex].shapeType != MR_SHAPE_BOX ||
        !std::isfinite(foot.supportPatchBounds.x) ||
        !std::isfinite(foot.supportPatchBounds.y) ||
        !std::isfinite(foot.supportPatchBounds.z) ||
        !std::isfinite(foot.supportPatchBounds.w) ||
        !(foot.supportPatchBounds.z > foot.supportPatchBounds.x) ||
        !(foot.supportPatchBounds.w > foot.supportPatchBounds.y)) {
        throw std::logic_error(
            "G1 plantar atlas requires the production box sole"
        );
    }
    TactileSensorSpec sensor;
    sensor.id = std::string{foot.name} + "_tactile";
    sensor.parentBodyIndex = foot.bodyIndex;
    sensor.backingShapeIndices = {foot.soleShapeIndex};
    sensor.localPose.position = foot.solePosition;
    sensor.localPose.orientation = foot.soleRotation;
    sensor.width = kAtlasWidth;
    sensor.height = kAtlasHeight;
    sensor.surfaceKind = MR_TACTILE_SURFACE_CUSTOM_ATLAS;
    sensor.maximumDepthMeters = kG1ShellThickness;
    sensor.maximumTangentialDisplacementMeters =
        kG1ShellThickness;
    sensor.activeDepthThresholdMeters = 1.0e-6f;
    sensor.queryEpsilonMeters = 2.5e-7f;
    sensor.targetShapeIndices = {targetShape};
    sensor.samples.reserve(kAtlasWidth * kAtlasHeight);

    const float lowerX = foot.supportPatchBounds.x;
    const float upperX = foot.supportPatchBounds.z;
    const float lowerY = foot.supportPatchBounds.y;
    const float upperY = foot.supportPatchBounds.w;
    const float cellWidth =
        (upperX - lowerX) / static_cast<float>(kAtlasWidth);
    const float cellHeight =
        (upperY - lowerY) / static_cast<float>(kAtlasHeight);
    for (std::uint32_t v = 0u; v < kAtlasHeight; ++v) {
        for (std::uint32_t u = 0u; u < kAtlasWidth; ++u) {
            sensor.samples.push_back({
                {
                    lowerX +
                        (static_cast<float>(u) + 0.5f) * cellWidth,
                    lowerY +
                        (static_cast<float>(v) + 0.5f) * cellHeight,
                    0.0f,
                    0.0f,
                },
                {0.0f, 0.0f, -1.0f, 0.0f},
                {1.0f, 0.0f, 0.0f, 0.0f},
                {0.0f, -1.0f, 0.0f, 0.0f},
                cellWidth * cellHeight,
                kG1ShellThickness,
                u,
                v,
                true,
            });
        }
    }
    return sensor;
}

TactileSensorSpec makePsmJawSensor(
    const EngineModel& model,
    const std::string_view idPrefix,
    const std::uint32_t bodyOffset,
    const std::uint32_t shapeOffset,
    const bool leftJaw,
    const std::span<const std::uint32_t> targets
) {
    const std::uint32_t bodyIndex =
        bodyOffset + (leftJaw ? 7u : 8u);
    const std::array<std::uint32_t, 3u> localBackings =
        leftJaw
        ? std::array<std::uint32_t, 3u>{14u, 15u, 18u}
        : std::array<std::uint32_t, 3u>{16u, 17u, 19u};
    std::array<std::uint32_t, 3u> backings{};
    for (std::size_t index = 0u; index < backings.size(); ++index) {
        backings[index] = shapeOffset + localBackings[index];
    }

    TactileSensorSpec sensor;
    sensor.id = std::string{idPrefix} +
        (leftJaw ? "left_jaw_tactile" : "right_jaw_tactile");
    sensor.parentBodyIndex = bodyIndex;
    sensor.backingShapeIndices.assign(
        backings.begin(),
        backings.end()
    );
    sensor.localPose.orientation.w = 1.0f;
    sensor.width = kAtlasWidth;
    sensor.height = kAtlasHeight;
    sensor.surfaceKind = MR_TACTILE_SURFACE_CUSTOM_ATLAS;
    sensor.maximumDepthMeters = kPsmShellThickness;
    sensor.maximumTangentialDisplacementMeters =
        kPsmShellThickness;
    sensor.activeDepthThresholdMeters = 2.5e-8f;
    sensor.queryEpsilonMeters = 1.0e-8f;
    sensor.targetShapeIndices.assign(targets.begin(), targets.end());
    sensor.samples.reserve(kAtlasWidth * kAtlasHeight);

    const float inward = leftJaw ? -1.0f : 1.0f;
    const MRShapeGPU& capsule = model.shapes[backings[0]];
    constexpr std::uint32_t capsuleRows = 24u;
    constexpr float halfArc =
        55.0f * std::numbers::pi_v<float> / 180.0f;
    const float angleStep =
        2.0f * halfArc / static_cast<float>(capsuleRows);
    const float axialStep =
        2.0f * capsule.dimensions.y /
        static_cast<float>(kAtlasWidth);
    for (std::uint32_t v = 0u; v < kAtlasHeight; ++v) {
        for (std::uint32_t u = 0u; u < kAtlasWidth; ++u) {
            if (v < capsuleRows) {
                const float axial =
                    -capsule.dimensions.y +
                    (static_cast<float>(u) + 0.5f) * axialStep;
                const float angle =
                    -halfArc +
                    (static_cast<float>(v) + 0.5f) * angleStep;
                const mr_float4 normal = normalized({
                    inward * std::cos(angle),
                    std::sin(angle),
                    0.0f,
                    0.0f,
                });
                const mr_float4 tangentU{
                    0.0f, 0.0f, 1.0f, 0.0f,
                };
                const mr_float4 tangentV =
                    normalized(cross3(normal, tangentU));
                const float radius =
                    capsule.dimensions.x + kPsmShellThickness;
                sensor.samples.push_back({
                    {
                        capsule.localPosition.x +
                            normal.x * radius,
                        capsule.localPosition.y +
                            normal.y * radius,
                        capsule.localPosition.z + axial,
                        0.0f,
                    },
                    normal,
                    tangentU,
                    tangentV,
                    radius * angleStep * axialStep,
                    kPsmShellThickness,
                    u,
                    v,
                    true,
                });
                continue;
            }

            constexpr std::uint32_t toothWidth = 16u;
            constexpr std::uint32_t toothHeight = 8u;
            const std::uint32_t tooth = u / toothWidth;
            const float nu =
                (
                    (static_cast<float>(u % toothWidth) + 0.5f) /
                        static_cast<float>(toothWidth) *
                        2.0f -
                    1.0f
                );
            const float nv =
                (
                    (static_cast<float>(v - capsuleRows) + 0.5f) /
                        static_cast<float>(toothHeight) *
                        2.0f -
                    1.0f
                );
            if (nu * nu + nv * nv >= 1.0f) {
                sensor.samples.push_back(invalidSample(u, v));
                continue;
            }
            const MRShapeGPU& sphere =
                model.shapes[backings[1u + tooth]];
            const float normalRadius = std::sin(halfArc);
            const float projectedU = nu * normalRadius;
            const float projectedV = nv * normalRadius;
            const mr_float4 baseNormal{
                inward, 0.0f, 0.0f, 0.0f,
            };
            const mr_float4 baseU{0.0f, 0.0f, 1.0f, 0.0f};
            const mr_float4 baseV =
                normalized(cross3(baseNormal, baseU));
            const float centerWeight = std::sqrt(
                std::max(
                    0.0f,
                    1.0f -
                        projectedU * projectedU -
                        projectedV * projectedV
                )
            );
            const mr_float4 normal = normalized({
                baseNormal.x * centerWeight +
                    baseU.x * projectedU +
                    baseV.x * projectedV,
                baseNormal.y * centerWeight +
                    baseU.y * projectedU +
                    baseV.y * projectedV,
                baseNormal.z * centerWeight +
                    baseU.z * projectedU +
                    baseV.z * projectedV,
                0.0f,
            });
            const mr_float4 tangentU =
                tangentFromReference(normal, baseU);
            const mr_float4 tangentV =
                normalized(cross3(normal, tangentU));
            const float radius =
                sphere.dimensions.x + kPsmShellThickness;
            const float stepU =
                2.0f * normalRadius /
                static_cast<float>(toothWidth);
            const float stepV =
                2.0f * normalRadius /
                static_cast<float>(toothHeight);
            sensor.samples.push_back({
                {
                    sphere.localPosition.x + normal.x * radius,
                    sphere.localPosition.y + normal.y * radius,
                    sphere.localPosition.z + normal.z * radius,
                    0.0f,
                },
                normal,
                tangentU,
                tangentV,
                radius * radius * stepU * stepV /
                    std::max(centerWeight, 1.0e-4f),
                kPsmShellThickness,
                u,
                v,
                true,
            });
        }
    }
    return sensor;
}

std::array<float, 3u> psmNeedleInitialPosition(
    const EngineModel& model,
    const CurvedSutureNeedleAsset& needle
) {
    constexpr std::array<std::uint32_t, 4u> jawShapeIndices{
        15u, 17u, 18u, 19u,
    };
    std::array<ArticulatedPointQuery, jawShapeIndices.size()>
        jawQueries{};
    for (std::size_t index = 0u;
         index < jawShapeIndices.size();
         ++index) {
        const MRShapeGPU& shape =
            model.shapes[jawShapeIndices[index]];
        jawQueries[index] = {
            .bodyIndex = shape.bodyIndex,
            .localPoint = {
                shape.localPosition.x,
                shape.localPosition.y,
                shape.localPosition.z,
            },
        };
    }
    std::vector<double> q(
        model.defaultQ.begin(),
        model.defaultQ.end()
    );
    // Place the needle at the known open pickup aperture. The example then
    // closes both independently modelled jaws from this same configuration.
    q[6u] = -0.045;
    q[7u] = 0.045;
    const std::vector<double> v(
        model.defaultV.begin(),
        model.defaultV.end()
    );
    std::array<
        ArticulatedPointKinematics,
        jawShapeIndices.size()
    > jawKinematics{};
    std::vector<double> jacobian(
        3u * jawShapeIndices.size() *
            model.articulations.front().nv
    );
    const auto status = computeArticulatedPointJacobians(
        model,
        0u,
        q,
        v,
        jawQueries,
        jawKinematics,
        jacobian
    );
    if (!status.succeeded()) {
        throw std::logic_error(
            "could not derive the authored PSM needle reset"
        );
    }
    const std::uint32_t graspShapeIndex =
        (
            needle.metadata.graspShapeBegin +
            needle.metadata.graspShapeEnd
        ) / 2u;
    const MRShapeGPU& graspShape =
        needle.rigid.shapes.at(graspShapeIndex);
    std::array<double, 3u> midpoint{};
    for (const auto& point : jawKinematics) {
        midpoint[0] += point.position[0] /
            static_cast<double>(jawKinematics.size());
        midpoint[1] += point.position[1] /
            static_cast<double>(jawKinematics.size());
        midpoint[2] += point.position[2] /
            static_cast<double>(jawKinematics.size());
    }
    return {
        static_cast<float>(midpoint[0]) -
            graspShape.localPosition.x,
        static_cast<float>(midpoint[1]) -
            graspShape.localPosition.y,
        static_cast<float>(midpoint[2]) -
            graspShape.localPosition.z,
    };
}

} // namespace

EngineModel makeUnitreeG1TactileEngineModel() {
    EngineModel model = makeUnitreeG1EngineModel();
    model.name = "unitree_g1_plantar_tactile_world";
    // Bounded position-level shell compliance is an engineering prior for
    // the virtual plantar layer. It is not a measured G1 foot material.
    model.materials.at(0u).response.z = 2.0e-6f;
    model.materials.at(0u).response.w = 0.02f;
    for (std::uint32_t shapeIndex = 0u; shapeIndex < 2u;
         ++shapeIndex) {
        model.shapes[shapeIndex].contactRestAndBoundingRadius.x =
            kG1ContactOffset;
        model.shapes[shapeIndex].contactRestAndBoundingRadius.y =
            kG1ShellThickness;
    }
    const std::uint32_t groundBody =
        static_cast<std::uint32_t>(model.bodies.size());
    model.bodies.push_back(staticBody());
    if (!model.bodyNames.empty()) {
        model.bodyNames.emplace_back("ground");
    }
    model.shapes.push_back(staticBox(
        groundBody,
        0u,
        {2.0f, 2.0f, 0.05f, 0.0f},
        900001u
    ));
    if (!model.shapeNames.empty()) {
        model.shapeNames.emplace_back("ground_collision");
    }
    model.world.bodyCount =
        static_cast<std::uint32_t>(model.bodies.size());
    model.world.shapeCount =
        static_cast<std::uint32_t>(model.shapes.size());
    std::string reason;
    if (!model.valid(&reason)) {
        throw std::logic_error(
            "G1 tactile model is invalid: " + reason
        );
    }
    return model;
}

EpisodeTwin makeUnitreeG1TactileEpisodeTwin() {
    const EngineModel model = makeUnitreeG1TactileEngineModel();
    const std::uint32_t groundBody =
        static_cast<std::uint32_t>(model.bodies.size() - 1u);
    const std::uint32_t groundShape =
        static_cast<std::uint32_t>(model.shapes.size() - 1u);
    EpisodeTwin episode;
    episode.id = "unitree_g1_plantar_tactile_balance";

    WorldAsset robot = asset(
        "g1",
        MR_WORLD_ASSET_ROBOT,
        MR_WORLD_COLLISION_CONVEX,
        MR_WORLD_DYNAMICS_ARTICULATED
    );
    robot.articulationIndex = 0u;
    for (std::uint32_t body = 0u; body < groundBody; ++body) {
        robot.bodyIndices.push_back(body);
    }
    for (std::uint32_t shape = 0u; shape < groundShape; ++shape) {
        robot.shapeIndices.push_back(shape);
    }
    robot.materialIndices = {0u};

    WorldAsset ground = asset(
        "support_surface",
        MR_WORLD_ASSET_MANIPULATED,
        MR_WORLD_COLLISION_PRIMITIVES,
        MR_WORLD_DYNAMICS_STATIC
    );
    ground.bodyIndices = {groundBody};
    ground.shapeIndices = {groundShape};
    ground.materialIndices = {0u};
    ground.initialPose.position = {0.0f, 0.0f, -0.05f, 0.0f};
    ground.anchors.push_back({
        "support_center",
        {
            {0.0f, 0.0f, 0.05f, 0.0f},
            {0.0f, 0.0f, 0.0f, 1.0f},
        },
        0.05f,
        0u,
    });
    episode.assets = {std::move(robot), std::move(ground)};

    const G1ModelMetadata& metadata = unitreeG1Metadata();
    episode.tactileSensors = {
        makeG1SoleSensor(model, metadata.feet[0], groundShape),
        makeG1SoleSensor(model, metadata.feet[1], groundShape),
    };
    episode.task = {
        "g1_tactile_balance",
        "g1",
        "support_surface",
        "support_surface",
        "support_center",
        0.02,
        24.0,
    };
    return episode;
}

EngineModel makeDvrkPsmTactileEngineModel() {
    EngineModel model = makeDvrkPsmLargeNeedleDriverEngineModel();
    model.name = "dvrk_psm_jaw_tactile_needle_world";
    // This small compliant contact layer makes the authored jaw shell
    // observable without treating it as calibrated elastomer mechanics.
    model.materials.at(0u).response.z = 1.0e-5f;
    model.materials.at(0u).response.w = 0.02f;
    for (const std::uint32_t shapeIndex :
         {14u, 15u, 16u, 17u, 18u, 19u}) {
        model.shapes[shapeIndex]
            .contactRestAndBoundingRadius.x =
                kPsmContactOffset;
        model.shapes[shapeIndex]
            .contactRestAndBoundingRadius.y =
                kPsmShellThickness;
    }

    const std::uint32_t needleBody =
        static_cast<std::uint32_t>(model.bodies.size());
    const std::uint32_t needleMaterial =
        static_cast<std::uint32_t>(model.materials.size());
    CurvedSutureNeedleAsset needle =
        makeCurvedSutureNeedleAsset({
            .bodyIndex = needleBody,
            .materialIndex = needleMaterial,
            .slotGenerationBase = 910001u,
            .collisionGroup = 1u,
            .collisionMask = ~0u,
            .motionType = MR_MOTION_DYNAMIC,
        });
    for (MRShapeGPU& shape : needle.rigid.shapes) {
        shape.flags |= MR_SHAPE_FLAG_ENABLE_CCD;
    }
    model.bodies.push_back(needle.rigid.body);
    model.materials.push_back(needle.rigid.material);
    model.shapes.insert(
        model.shapes.end(),
        needle.rigid.shapes.begin(),
        needle.rigid.shapes.end()
    );
    model.world.bodyCount =
        static_cast<std::uint32_t>(model.bodies.size());
    model.world.materialCount =
        static_cast<std::uint32_t>(model.materials.size());
    model.world.shapeCount =
        static_cast<std::uint32_t>(model.shapes.size());
    std::string reason;
    if (!model.valid(&reason)) {
        throw std::logic_error(
            "PSM tactile model is invalid: " + reason
        );
    }
    return model;
}

EpisodeTwin makeDvrkPsmTactileEpisodeTwin() {
    const EngineModel model = makeDvrkPsmTactileEngineModel();
    EpisodeTwin episode;
    episode.id = "dvrk_psm_jaw_tactile_needle_hold";

    WorldAsset robot = asset(
        "psm",
        MR_WORLD_ASSET_ROBOT,
        MR_WORLD_COLLISION_PRIMITIVES,
        MR_WORLD_DYNAMICS_ARTICULATED
    );
    robot.articulationIndex = 0u;
    for (std::uint32_t body = 0u; body < 9u; ++body) {
        robot.bodyIndices.push_back(body);
    }
    for (std::uint32_t shape = 0u; shape < 20u; ++shape) {
        robot.shapeIndices.push_back(shape);
    }
    robot.materialIndices = {0u};

    WorldAsset needle = asset(
        "needle",
        MR_WORLD_ASSET_MANIPULATED,
        MR_WORLD_COLLISION_PRIMITIVES,
        MR_WORLD_DYNAMICS_RIGID
    );
    needle.bodyIndices = {9u};
    for (std::uint32_t shape = 20u;
         shape < model.shapes.size();
         ++shape) {
        needle.shapeIndices.push_back(shape);
    }
    needle.materialIndices = {1u};
    const CurvedSutureNeedleAsset needleGeometry =
        makeCurvedSutureNeedleAsset({
            .bodyIndex = 9u,
            .materialIndex = 1u,
            .slotGenerationBase = 910001u,
            .collisionGroup = 1u,
            .collisionMask = ~0u,
            .motionType = MR_MOTION_DYNAMIC,
        });
    const std::array<float, 3u> initial =
        psmNeedleInitialPosition(model, needleGeometry);
    needle.initialPose.position = {
        initial[0],
        initial[1],
        initial[2],
        0.0f,
    };
    needle.anchors.push_back({
        "hold_pose",
        {
            {0.0f, 0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 1.0f},
        },
        0.003f,
        0u,
    });
    episode.assets = {std::move(robot), std::move(needle)};

    std::vector<std::uint32_t> targets;
    for (std::uint32_t shape = 20u;
         shape < model.shapes.size();
         ++shape) {
        targets.push_back(shape);
    }
    episode.tactileSensors = {
        makePsmJawSensor(model, "", 0u, 0u, true, targets),
        makePsmJawSensor(model, "", 0u, 0u, false, targets),
    };
    episode.task = {
        "psm_tactile_needle_hold_lift",
        "psm",
        "needle",
        "needle",
        "hold_pose",
        0.01,
        4.0,
    };
    return episode;
}

DualPsmWorld makeDualDvrkPsmTactileWorld(
    const DualPsmWorldConfig& config
) {
    DualPsmWorld dual = makeDualDvrkPsmWorld(config);
    for (MRMaterialGPU& material : dual.model.materials) {
        material.response.z = 1.0e-5f;
        material.response.w = 0.02f;
    }
    for (const std::uint32_t firstShape :
         dual.metadata.firstShapes) {
        for (const std::uint32_t localShape :
             {14u, 15u, 16u, 17u, 18u, 19u}) {
            MRShapeGPU& backing =
                dual.model.shapes.at(firstShape + localShape);
            backing.contactRestAndBoundingRadius.x =
                kPsmContactOffset;
            backing.contactRestAndBoundingRadius.y =
                kPsmShellThickness;
        }
    }
    std::string reason;
    if (!dual.model.valid(&reason)) {
        throw std::logic_error(
            "dual PSM tactile model is invalid: " + reason
        );
    }
    return dual;
}

std::vector<TactileSensorSpec>
makeDualDvrkPsmTactileSensors(
    const EngineModel& model,
    const DualPsmWorldMetadata& metadata,
    const std::span<const std::uint32_t> targetShapeIndices
) {
    if (targetShapeIndices.empty()) {
        throw std::invalid_argument(
            "dual PSM tactile sensors require explicit target shapes"
        );
    }
    std::vector<TactileSensorSpec> sensors;
    sensors.reserve(4u);
    for (std::uint32_t arm = 0u; arm < 2u; ++arm) {
        const std::string prefix =
            arm == 0u ? "left_psm_" : "right_psm_";
        sensors.push_back(makePsmJawSensor(
            model,
            prefix,
            metadata.rootBodies[arm],
            metadata.firstShapes[arm],
            true,
            targetShapeIndices
        ));
        sensors.push_back(makePsmJawSensor(
            model,
            prefix,
            metadata.rootBodies[arm],
            metadata.firstShapes[arm],
            false,
            targetShapeIndices
        ));
    }
    return sensors;
}

} // namespace metalrobo
