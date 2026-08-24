#include "metalrobo/RunProgram.hpp"
#include "metalrobo/PX4X500.hpp"

#include "metalrobo/Franka.hpp"
#include "metalrobo/G1.hpp"
#include "metalrobo/SurgicalPSM.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <ranges>
#include <span>
#include <unordered_set>
#include <utility>

namespace metalrobo {
namespace {

constexpr std::uint64_t kFNVOffset = 1469598103934665603ull;
constexpr std::uint64_t kFNVPrime = 1099511628211ull;

class Hash {
public:
    void bytes(const void* data, const std::size_t size) {
        const auto* values = static_cast<const unsigned char*>(data);
        for (std::size_t index = 0u; index < size; ++index) {
            value_ ^= values[index];
            value_ *= kFNVPrime;
        }
    }
    template <typename T> void scalar(const T& value) {
        bytes(&value, sizeof(value));
    }
    void string(const std::string_view value) {
        scalar<std::uint64_t>(value.size());
        bytes(value.data(), value.size());
    }
    [[nodiscard]] std::uint64_t finish() const noexcept {
        return value_ == 0u ? 1u : value_;
    }
private:
    std::uint64_t value_ = kFNVOffset;
};

RunCompileDiagnostics reject(
    const RunCompileStatus status,
    std::string element,
    std::string message
) {
    return {status, std::move(element), std::move(message)};
}

template <typename T>
std::vector<std::uint32_t> indices(
    const std::vector<T>& values,
    const std::uint32_t offset = 0u
) {
    std::vector<std::uint32_t> result(values.size());
    for (std::size_t index = 0u; index < values.size(); ++index) {
        result[index] = offset + static_cast<std::uint32_t>(index);
    }
    return result;
}

const RobotSemanticRole* role(
    const RobotPack& robot,
    const std::string_view id,
    const RobotSemanticKind kind
) {
    const auto found = std::ranges::find_if(
        robot.roles,
        [&](const RobotSemanticRole& candidate) {
            return candidate.id == id && candidate.kind == kind;
        }
    );
    return found == robot.roles.end() ? nullptr : &*found;
}

bool validRoles(const RobotPack& robot, std::string& reason) {
    std::unordered_set<std::string> ids;
    for (const RobotSemanticRole& semantic : robot.roles) {
        if (semantic.id.empty() || semantic.members.empty() ||
            !ids.insert(semantic.id).second) {
            reason = "robot semantic roles must have unique nonempty identities";
            return false;
        }
        const std::vector<std::string>* names = nullptr;
        switch (semantic.kind) {
        case RobotSemanticKind::body:
            names = &robot.mechanics.bodyNames;
            break;
        case RobotSemanticKind::joint:
            names = &robot.mechanics.jointNames;
            break;
        case RobotSemanticKind::dof:
            names = &robot.mechanics.dofNames;
            break;
        default:
            reason = "robot semantic role '" + semantic.id +
                "' has an unknown kind";
            return false;
        }
        std::unordered_set<std::string> members;
        for (const std::string& member : semantic.members) {
            if (!members.insert(member).second ||
                std::ranges::count(*names, member) != 1) {
                reason = "robot semantic role '" + semantic.id +
                    "' contains a duplicate or unresolved member '" +
                    member + "'";
                return false;
            }
        }
    }
    return true;
}

bool validActuators(const RobotPack& robot, std::string& reason) {
    std::unordered_set<std::string> ids;
    std::unordered_set<std::string> controlledJoints;
    for (const RobotActuatorSpec& actuator : robot.actuators) {
        if (actuator.id.empty() || actuator.target.empty() ||
            !ids.insert(actuator.id).second ||
            !std::isfinite(actuator.scale) || !(actuator.scale > 0.0f) ||
            !std::isfinite(actuator.responseTimeSeconds) ||
            actuator.responseTimeSeconds < 0.0f ||
            !std::isfinite(actuator.parameters.x) ||
            !std::isfinite(actuator.parameters.y) ||
            !std::isfinite(actuator.parameters.z) ||
            !std::isfinite(actuator.parameters.w)) {
            reason = "robot actuators require unique identities, targets, and finite positive scales";
            return false;
        }
        switch (actuator.kind) {
        case RobotActuatorKind::jointPosition:
        case RobotActuatorKind::jointVelocity:
        case RobotActuatorKind::jointEffort:
        case RobotActuatorKind::gripperPosition:
        case RobotActuatorKind::flappingPosition:
            if (std::ranges::count(
                    robot.mechanics.jointNames,
                    actuator.target
                ) != 1) {
                reason = "robot actuator '" + actuator.id +
                    "' has an unresolved joint target '" +
                    actuator.target + "'";
                return false;
            }
            if (!controlledJoints.insert(actuator.target).second) {
                reason = "robot joint '" + actuator.target +
                    "' is controlled by more than one physical actuator";
                return false;
            }
            break;
        case RobotActuatorKind::tendonPosition:
            if (actuator.terms.empty() ||
                !(actuator.parameters.x > 0.0f) ||
                actuator.parameters.y < 0.0f ||
                !(actuator.parameters.z > 0.0f)) {
                reason = "tendon actuator '" + actuator.id +
                    "' requires stiffness, damping, force limit, and sparse terms";
                return false;
            }
            {
                std::unordered_set<std::string> tendonJoints;
                for (const RobotActuatorTermSpec& term : actuator.terms) {
                    if (term.joint.empty() ||
                        !std::isfinite(term.coefficient) ||
                        term.coefficient == 0.0f ||
                        !tendonJoints.insert(term.joint).second ||
                        std::ranges::count(
                            robot.mechanics.jointNames,
                            term.joint
                        ) != 1) {
                        reason = "tendon actuator '" + actuator.id +
                            "' has a duplicate or unresolved sparse term";
                        return false;
                    }
                    if (!controlledJoints.insert(term.joint).second) {
                        reason = "robot joint '" + term.joint +
                            "' is shared by incompatible physical actuators";
                        return false;
                    }
                }
            }
            break;
        case RobotActuatorKind::rotorMixer:
        case RobotActuatorKind::bodyWrench:
            // These target robot-authored named controller groups or bodies;
            // their specialized compiler validates the concrete program.
            break;
        default:
            reason = "robot actuator '" + actuator.id +
                "' has an unknown kind";
            return false;
        }
    }
    return true;
}

std::optional<std::string> compileActuatorModes(
    EngineModel& model,
    const std::span<const RobotActuatorSpec> actuators
) {
    const auto disableImplicitDrive = [&](const std::string_view jointName)
        -> std::optional<std::string> {
        const auto joint = std::ranges::find(model.jointNames, jointName);
        if (joint == model.jointNames.end()) {
            return "actuator joint is absent from composed mechanics: " +
                std::string{jointName};
        }
        const std::uint32_t jointIndex = static_cast<std::uint32_t>(
            joint - model.jointNames.begin());
        auto dof = std::ranges::find_if(
            model.dofs,
            [jointIndex](const MRDofPropertiesGPU& candidate) {
                return candidate.jointIndex == jointIndex;
            }
        );
        if (dof == model.dofs.end() ||
            std::ranges::count_if(
                model.dofs,
                [jointIndex](const MRDofPropertiesGPU& candidate) {
                    return candidate.jointIndex == jointIndex;
                }
            ) != 1u) {
            return "non-position actuator requires one scalar DoF: " +
                std::string{jointName};
        }
        // Velocity, effort, and tendon programs write physical generalized
        // effort from live microstep state. Leaving the position-drive flag
        // enabled would add a second hidden controller and implicit inertia.
        dof->flags &= ~static_cast<std::uint32_t>(MR_DOF_FLAG_DRIVE);
        dof->drive.x = 0.0f;
        dof->drive.y = 0.0f;
        return std::nullopt;
    };
    for (const RobotActuatorSpec& actuator : actuators) {
        if (actuator.kind == RobotActuatorKind::jointVelocity ||
            actuator.kind == RobotActuatorKind::jointEffort) {
            if (const auto error = disableImplicitDrive(actuator.target)) {
                return error;
            }
        } else if (actuator.kind == RobotActuatorKind::tendonPosition) {
            for (const RobotActuatorTermSpec& term : actuator.terms) {
                if (const auto error = disableImplicitDrive(term.joint)) {
                    return error;
                }
            }
        }
    }
    return std::nullopt;
}

std::uint64_t robotPackFingerprint(const RobotPack& robot) {
    Hash hash;
    hash.scalar(engineModelFingerprint(robot.mechanics));
    hash.scalar(robot.revision);
    hash.string(robot.sourceRepository);
    hash.string(robot.sourceRevision);
    hash.scalar<std::uint64_t>(robot.actuators.size());
    for (const RobotActuatorSpec& actuator : robot.actuators) {
        hash.string(actuator.id);
        hash.scalar(actuator.kind);
        hash.string(actuator.target);
        hash.scalar(actuator.scale);
        hash.scalar(actuator.responseTimeSeconds);
        hash.scalar(actuator.parameters);
        hash.scalar(actuator.component);
        hash.scalar<std::uint64_t>(actuator.terms.size());
        for (const RobotActuatorTermSpec& term : actuator.terms) {
            hash.string(term.joint);
            hash.scalar(term.coefficient);
        }
    }
    if (robot.multicopter) {
        hash.bytes(&robot.multicopter->model, sizeof(robot.multicopter->model));
        hash.bytes(robot.multicopter->rotors.data(),
            sizeof(robot.multicopter->rotors));
        hash.bytes(&robot.multicopter->mixer, sizeof(robot.multicopter->mixer));
        hash.string(robot.multicopter->bodyRole);
        hash.scalar(robot.multicopter->windVelocity);
    }
    if (robot.flappingWings) {
        hash.bytes(robot.flappingWings->wings.data(),
            sizeof(robot.flappingWings->wings));
        hash.bytes(&robot.flappingWings->tail,
            sizeof(robot.flappingWings->tail));
        hash.bytes(&robot.flappingWings->fuselage,
            sizeof(robot.flappingWings->fuselage));
        hash.string(robot.flappingWings->bodyRole);
        for (const std::string& wingRole : robot.flappingWings->wingRoles) {
            hash.string(wingRole);
        }
        hash.string(robot.flappingWings->tailRole);
        hash.scalar(robot.flappingWings->windVelocityAndDensity);
    }
    return hash.finish();
}

std::uint64_t sensorFingerprint(
    const std::span<const SensorSpec> sensors
) {
    Hash hash;
    hash.scalar<std::uint64_t>(sensors.size());
    for (const SensorSpec& sensor : sensors) {
        hash.string(sensor.id);
        hash.string(sensor.parentAssetId);
        hash.scalar(sensor.parentKind);
        hash.scalar(sensor.parentBodyIndex);
        hash.scalar(sensor.kind);
        hash.scalar(sensor.localPose.position);
        hash.scalar(sensor.localPose.orientation);
        hash.scalar(sensor.width);
        hash.scalar(sensor.height);
        hash.scalar(sensor.intrinsics);
        hash.scalar(sensor.distortion);
        hash.scalar(sensor.focalScale);
        hash.scalar(sensor.colorNoiseSigma);
        hash.scalar(sensor.depthNoiseSigma);
        hash.scalar(sensor.depthDropout);
        hash.scalar(sensor.latencySeconds);
        hash.scalar(sensor.nominalRateHz);
        hash.scalar(sensor.exposureSeconds);
        hash.scalar(sensor.shutterReadoutSeconds);
        hash.scalar(sensor.shutterModel);
        hash.scalar(sensor.shutterDirection);
        hash.scalar(sensor.frameJitterSeconds);
        hash.scalar(sensor.minimumDepthMeters);
        hash.scalar(sensor.maximumDepthMeters);
        hash.scalar(sensor.depthQuantumMeters);
        hash.scalar(sensor.motionBlurScale);
    }
    return hash.finish();
}

bool anyCapacity(const MetalWorldCapacityProfile& capacity) {
    const MetalWorldCapacityProfile empty{};
    return std::memcmp(&capacity, &empty, sizeof(capacity)) != 0;
}

std::optional<std::string> lowerRealityProgram(
    const WorldTemplate& world,
    const RunProfile& profile,
    RealityPack const& reality,
    TaskResetProgram& executable
) {
    executable = reality.reset;
    const EngineModel& model = world.engineModel;
    const auto appendBodyParameter = [&executable, &model](
        const WorldAsset& asset,
        const std::uint32_t component,
        const mr_float4 range
    ) -> std::optional<std::string> {
        for (const std::uint32_t body : asset.bodyIndices) {
            if (body >= model.bodyNames.size()) {
                return "reality asset body range exceeds compiled mechanics";
            }
            executable.operators.push_back({
                .operation = TaskRandomizationOperator::worldBodyParameter,
                .target = model.bodyNames[body],
                .component = component,
                .parameters = range,
            });
        }
        return std::nullopt;
    };
    for (const VariationParameter& variation :
         reality.program.variations) {
        mr_float4 range{};
        switch (variation.distribution) {
        case MR_WORLD_DISTRIBUTION_CONSTANT:
            range = {variation.parameters.x, variation.parameters.x,
                0.0f, 0.0f};
            break;
        case MR_WORLD_DISTRIBUTION_UNIFORM:
            range = variation.parameters;
            break;
        default:
            return "runtime RealityProgram currently requires constant or uniform physical distributions: " +
                variation.id;
        }
        const std::uint32_t assetIndex =
            world.assetIndex(variation.targetId);
        const WorldAsset* asset = assetIndex == MR_INVALID_INDEX
            ? nullptr
            : &world.assets[assetIndex];
        switch (variation.target) {
        case MR_WORLD_TARGET_ASSET_POSITION_X:
        case MR_WORLD_TARGET_ASSET_POSITION_Y:
        case MR_WORLD_TARGET_ASSET_POSITION_Z: {
            if (asset == nullptr) {
                return "reality position target is unresolved: " +
                    variation.targetId;
            }
            const std::uint32_t component =
                variation.target - MR_WORLD_TARGET_ASSET_POSITION_X;
            for (const std::uint32_t body : asset->bodyIndices) {
                if (body >= model.bodyNames.size() ||
                    model.bodies[body].articulationIndex !=
                        MR_INVALID_INDEX) {
                    continue;
                }
                executable.operators.push_back({
                    .operation =
                        TaskRandomizationOperator::sceneBodyPosition,
                    .target = model.bodyNames[body],
                    .component = component,
                    .parameters = range,
                });
            }
            break;
        }
        case MR_WORLD_TARGET_ASSET_MASS_SCALE:
        case MR_WORLD_TARGET_ASSET_FRICTION_SCALE:
        case MR_WORLD_TARGET_ASSET_RESTITUTION_SCALE:
        case MR_WORLD_TARGET_ASSET_DAMPING_SCALE: {
            if (asset == nullptr) {
                return "reality physical target is unresolved: " +
                    variation.targetId;
            }
            const std::uint32_t component =
                variation.target - MR_WORLD_TARGET_ASSET_MASS_SCALE;
            if (const auto error = appendBodyParameter(
                    *asset, component, range
                )) {
                return error;
            }
            break;
        }
        case MR_WORLD_TARGET_ROBOT_GAIN_SCALE:
        case MR_WORLD_TARGET_ROBOT_DAMPING_SCALE:
        case MR_WORLD_TARGET_ROBOT_PAYLOAD_SCALE: {
            if (asset == nullptr ||
                asset->articulationIndex == MR_INVALID_INDEX) {
                return "reality controller target is unresolved: " +
                    variation.targetId;
            }
            const std::uint32_t component =
                variation.target == MR_WORLD_TARGET_ROBOT_GAIN_SCALE
                ? 0u
                : variation.target ==
                      MR_WORLD_TARGET_ROBOT_DAMPING_SCALE
                    ? 1u
                    : 3u;
            executable.operators.push_back({
                .operation =
                    TaskRandomizationOperator::controllerParameter,
                .component = component,
                .parameters = range,
            });
            break;
        }
        case MR_WORLD_TARGET_ROBOT_LATENCY_SECONDS: {
            if (asset == nullptr ||
                asset->articulationIndex == MR_INVALID_INDEX) {
                return "reality latency target is unresolved: " +
                    variation.targetId;
            }
            const float inverseStep =
                1.0f / profile.controlTimestepSeconds;
            const std::uint32_t lower = static_cast<std::uint32_t>(
                std::max(0.0f, std::floor(range.x * inverseStep))
            );
            const std::uint32_t upper = static_cast<std::uint32_t>(
                std::max(0.0f, std::ceil(range.y * inverseStep))
            );
            executable.maximumActionDelaySteps = std::max(
                executable.maximumActionDelaySteps,
                upper
            );
            executable.operators.push_back({
                .operation = TaskRandomizationOperator::actionDelay,
                .parameters = {
                    static_cast<float>(lower),
                    static_cast<float>(upper), 0.0f, 0.0f,
                },
            });
            break;
        }
        default:
            return "RealityProgram target has no native task-runtime execution path: " +
                variation.id;
        }
    }
    return std::nullopt;
}

MRWorldCollisionRepresentation collisionRepresentation(
    const EngineModel& model
) {
    if (std::ranges::any_of(model.shapes, [](const MRShapeGPU& shape) {
            return shape.shapeType == MR_SHAPE_TRIANGLE_MESH;
        })) {
        return MR_WORLD_COLLISION_TRIANGLE_MESH;
    }
    if (std::ranges::any_of(model.shapes, [](const MRShapeGPU& shape) {
            return shape.shapeType == MR_SHAPE_CONVEX;
        })) {
        return MR_WORLD_COLLISION_CONVEX;
    }
    return model.shapes.empty()
        ? MR_WORLD_COLLISION_NONE
        : MR_WORLD_COLLISION_PRIMITIVES;
}

RobotPack genericRobot(
    std::string id,
    EngineModel mechanics,
    std::vector<std::string> capabilities
) {
    RobotPack pack;
    pack.id = std::move(id);
    pack.mechanics = std::move(mechanics);
    pack.capabilities = std::move(capabilities);
    pack.roles.push_back({
        .id = "whole_body",
        .kind = RobotSemanticKind::body,
        .members = pack.mechanics.bodyNames,
    });
    if (!pack.mechanics.jointNames.empty()) {
        pack.roles.push_back({
            .id = "all_joints",
            .kind = RobotSemanticKind::joint,
            .members = pack.mechanics.jointNames,
        });
    }
    if (!pack.mechanics.dofNames.empty()) {
        pack.roles.push_back({
            .id = "all_dofs",
            .kind = RobotSemanticKind::dof,
            .members = pack.mechanics.dofNames,
        });
    }
    for (std::uint32_t joint = 0u;
         joint < pack.mechanics.jointNames.size();
         ++joint) {
        const auto dof = std::ranges::find_if(
            pack.mechanics.dofs,
            [joint](const MRDofPropertiesGPU& candidate) {
                return candidate.jointIndex == joint &&
                    (candidate.flags & MR_DOF_FLAG_ACTUATED) != 0u;
            }
        );
        if (dof == pack.mechanics.dofs.end() ||
            std::ranges::count_if(
                pack.mechanics.dofs,
                [joint](const MRDofPropertiesGPU& candidate) {
                    return candidate.jointIndex == joint;
                }
            ) != 1u) {
            continue;
        }
        pack.actuators.push_back({
            .id = pack.mechanics.jointNames[joint],
            .kind = RobotActuatorKind::jointPosition,
            .target = pack.mechanics.jointNames[joint],
            .scale = 0.25f,
        });
    }
    return pack;
}

void addBodyRole(
    RobotPack& pack,
    std::string id,
    std::initializer_list<std::string_view> members
) {
    std::vector<std::string> resolved;
    for (const std::string_view member : members) {
        if (std::ranges::count(pack.mechanics.bodyNames, member) == 1) {
            resolved.emplace_back(member);
        }
    }
    if (!resolved.empty()) {
        pack.roles.push_back({
            .id = std::move(id),
            .kind = RobotSemanticKind::body,
            .members = std::move(resolved),
        });
    }
}

} // namespace

std::uint64_t visualSensorProgramFingerprint(
    const VisualSensorProgram& program
) {
    Hash hash;
    hash.scalar<std::uint64_t>(program.assets.size());
    for (const VisualSensorAssetProgram& asset : program.assets) {
        hash.string(asset.contentHash);
        hash.string(asset.assetId);
        hash.scalar(asset.semanticId);
        hash.scalar(asset.instanceId);
    }
    hash.string(program.environmentContentHash);
    hash.string(program.rendererProfile);
    hash.string(program.cameraParentBody);
    hash.scalar(program.cameraPosition);
    hash.scalar(program.cameraOrientation);
    hash.scalar(program.width);
    hash.scalar(program.height);
    hash.scalar(program.minimumVisiblePixels);
    hash.scalar(program.verticalFieldOfViewDegrees);
    hash.scalar(program.nominalRateHz);
    hash.scalar(program.maximumRetainedBytes);
    hash.scalar(program.captureWidth);
    hash.scalar(program.captureHeight);
    hash.scalar(program.capturePolicyCamera);
    return hash.finish();
}

bool CompiledRun::valid() const noexcept {
    return fingerprint_ != 0u && robotFingerprint_ != 0u &&
        sensorFingerprint_ != 0u && realityFingerprint_ != 0u &&
        teacherFingerprint_ != 0u && model_.valid() && world_.valid() &&
        defaultSceneBodies_.size() == world_.sceneBodyCount() &&
        task_.valid() &&
        (!boundPolicy_.has_value() || policy_.valid());
}

std::uint64_t CompiledRun::fingerprint() const noexcept {
    return valid() ? fingerprint_ : 0u;
}
std::uint64_t CompiledRun::robotFingerprint() const noexcept {
    return valid() ? robotFingerprint_ : 0u;
}
std::uint64_t CompiledRun::sensorFingerprint() const noexcept {
    return valid() ? sensorFingerprint_ : 0u;
}
std::uint64_t CompiledRun::realityFingerprint() const noexcept {
    return valid() ? realityFingerprint_ : 0u;
}
std::uint64_t CompiledRun::teacherFingerprint() const noexcept {
    return valid() ? teacherFingerprint_ : 0u;
}
const WorldFamily& CompiledRun::worldFamily() const noexcept {
    return worldFamily_;
}
const CompiledWorld& CompiledRun::world() const noexcept { return world_; }
const EngineModel& CompiledRun::model() const noexcept { return model_; }
std::span<const MRBodyStateGPU>
CompiledRun::defaultSceneBodies() const noexcept {
    return defaultSceneBodies_;
}
const CompiledTaskProgram& CompiledRun::task() const noexcept { return task_; }
const CompiledPolicyProgram& CompiledRun::policy() const noexcept {
    return policy_;
}
const PolicyPack* CompiledRun::boundPolicy() const noexcept {
    return boundPolicy_ ? &*boundPolicy_ : nullptr;
}
const RunProfile& CompiledRun::profile() const noexcept { return profile_; }
const TeacherPack& CompiledRun::teacher() const noexcept { return teacher_; }
const VisualSensorProgram*
CompiledRun::visualSensorProgram() const noexcept {
    return visualSensorProgram_ ? &*visualSensorProgram_ : nullptr;
}
const MetalWorldMulticopterProgram*
CompiledRun::multicopterProgram() const noexcept {
    return multicopterProgram_ ? &*multicopterProgram_ : nullptr;
}
const MetalWorldFlappingWingProgram*
CompiledRun::flappingWingProgram() const noexcept {
    return flappingWingProgram_ ? &*flappingWingProgram_ : nullptr;
}

RunCompileDiagnostics compileRun(
    const RunManifest& manifest,
    CompiledRun& output
) {
    try {
        if (manifest.id.empty() || manifest.robot.id.empty() ||
            manifest.scene.id.empty() || manifest.sensors.id.empty() ||
            manifest.task.id.empty() || manifest.reality.id.empty() ||
            manifest.teacher.id.empty() ||
            manifest.profile.id.empty() ||
            manifest.profile.environmentCount == 0u ||
            manifest.profile.controlSteps == 0u ||
            manifest.profile.physicsSubsteps == 0u ||
            manifest.profile.velocityIterations == 0u ||
            manifest.profile.finalVelocityIterations == 0u ||
            (manifest.profile.maximumDifficultyBand != MR_INVALID_INDEX &&
             manifest.profile.minimumDifficultyBand >
                 manifest.profile.maximumDifficultyBand) ||
            !std::isfinite(manifest.profile.controlTimestepSeconds) ||
            !(manifest.profile.controlTimestepSeconds > 0.0f)) {
            return reject(
                RunCompileStatus::invalidManifest,
                "manifest",
                "run identity, package identities, execution counts, or timestep are invalid"
            );
        }
        if (manifest.sensors.deviceVisual) {
            const VisualSensorProgram& visual =
                *manifest.sensors.deviceVisual;
            const auto finite4 = [](const mr_float4 value) {
                return std::isfinite(value.x) &&
                    std::isfinite(value.y) &&
                    std::isfinite(value.z) &&
                    std::isfinite(value.w);
            };
            const double orientationNormSquared =
                static_cast<double>(visual.cameraOrientation.x) *
                    visual.cameraOrientation.x +
                static_cast<double>(visual.cameraOrientation.y) *
                    visual.cameraOrientation.y +
                static_cast<double>(visual.cameraOrientation.z) *
                    visual.cameraOrientation.z +
                static_cast<double>(visual.cameraOrientation.w) *
                    visual.cameraOrientation.w;
            const bool validAssets = !visual.assets.empty() &&
                std::ranges::all_of(
                    visual.assets,
                    [](const VisualSensorAssetProgram& asset) {
                        return !asset.path.empty() &&
                            !asset.assetId.empty() &&
                            !asset.contentHash.empty() &&
                            asset.semanticId != 0u &&
                            asset.semanticId != MR_INVALID_INDEX &&
                            asset.instanceId != 0u &&
                            asset.instanceId != MR_INVALID_INDEX;
                    }
                );
            if (!validAssets || visual.fingerprint == 0u ||
                visual.fingerprint !=
                    visualSensorProgramFingerprint(visual) ||
                (!visual.environmentPath.empty() !=
                 !visual.environmentContentHash.empty()) ||
                (!visual.rendererProfile.empty() &&
                 visual.rendererProfile != "sensor_fast") ||
                visual.cameraParentBody.empty() ||
                !finite4(visual.cameraPosition) ||
                !finite4(visual.cameraOrientation) ||
                std::abs(orientationNormSquared - 1.0) > 1.0e-5 ||
                visual.width == 0u || visual.height == 0u ||
                visual.minimumVisiblePixels == 0u ||
                !std::isfinite(visual.verticalFieldOfViewDegrees) ||
                visual.verticalFieldOfViewDegrees < 0.0f ||
                visual.verticalFieldOfViewDegrees >= 180.0f ||
                !std::isfinite(visual.nominalRateHz) ||
                !(visual.nominalRateHz > 0.0f) ||
                ((visual.captureWidth == 0u) !=
                 (visual.captureHeight == 0u)) ||
                (visual.capturePolicyCamera &&
                 !(visual.verticalFieldOfViewDegrees > 0.0f))) {
                return reject(
                    RunCompileStatus::invalidManifest,
                    manifest.sensors.id,
                    "SensorPack visual program is invalid"
                );
            }
        }
        if (manifest.teacher.kind == TeacherKind::none) {
            if (manifest.teacher.interactions.has_value() ||
                !manifest.teacher.interactionClip.empty()) {
                return reject(
                    RunCompileStatus::invalidManifest,
                    manifest.teacher.id,
                    "disabled TeacherPack contains executable teacher data"
                );
            }
        } else if (manifest.teacher.kind ==
                       TeacherKind::motionImagination) {
            if (!manifest.teacher.interactions.has_value() ||
                manifest.teacher.interactionClip.empty() ||
                manifest.teacher.artifactFingerprint == 0u) {
                return reject(
                    RunCompileStatus::invalidManifest,
                    manifest.teacher.id,
                    "motion-imagination TeacherPack requires a fingerprinted InteractionPack and clip"
                );
            }
        } else {
            return reject(
                RunCompileStatus::invalidManifest,
                manifest.teacher.id,
                "TeacherPack kind has no native executable program"
            );
        }
        std::string reason;
        if (manifest.robot.revision == 0u ||
            !manifest.robot.mechanics.valid(&reason) ||
            manifest.robot.primaryArticulationIndex >=
                manifest.robot.mechanics.articulations.size() ||
            !validRoles(manifest.robot, reason) ||
            !validActuators(manifest.robot, reason)) {
            return reject(
                RunCompileStatus::invalidRobot,
                manifest.robot.id,
                reason.empty() ? "robot package is invalid" : reason
            );
        }

        std::vector<EngineModelComponent> components;
        components.reserve(1u + manifest.scene.objects.size());
        components.push_back({
            &manifest.robot.mechanics,
            manifest.robot.id,
            true,
        });
        for (const SceneObject& object : manifest.scene.objects) {
            if (object.id.empty() || object.semanticClass.empty() ||
                !object.mechanics.valid(&reason)) {
                return reject(
                    RunCompileStatus::invalidManifest,
                    object.id,
                    reason.empty() ? "scene object is invalid" : reason
                );
            }
            components.push_back({&object.mechanics, object.id, true});
        }
        EngineModelComposeConfig compose = manifest.profile.physics;
        compose.name = manifest.id;
        compose.gravityAndTimestep.w =
            manifest.profile.controlTimestepSeconds /
            static_cast<float>(manifest.profile.physicsSubsteps);
        EngineModel model;
        const EngineModelComposeDiagnostics composition =
            composeEngineModels(components, model, compose);
        if (!composition.succeeded()) {
            return reject(
                RunCompileStatus::compositionFailure,
                "scene",
                composition.message
            );
        }
        if (const auto actuatorError = compileActuatorModes(
                model, manifest.robot.actuators)) {
            return reject(
                RunCompileStatus::invalidRobot,
                manifest.robot.id,
                *actuatorError
            );
        }

        std::vector<MRBodyStateGPU> defaultSceneBodies;
        std::uint32_t stateBodyOffset = 0u;
        const auto appendStates = [&defaultSceneBodies](
            const EngineModel& mechanics,
            const std::span<const MRBodyStateGPU> states,
            const std::uint32_t bodyOffset,
            const std::string_view owner
        ) -> std::optional<std::string> {
            const std::size_t expected = std::ranges::count_if(
                mechanics.bodies,
                [](const MRBodyPropertiesGPU& body) {
                    return body.articulationIndex == MR_INVALID_INDEX;
                }
            );
            if (states.size() != expected) {
                return std::string{owner} +
                    " reset-state count does not match its scene bodies";
            }
            for (const MRBodyStateGPU& source : states) {
                const std::uint32_t localBody = source.flagsAndIndices[2];
                if (localBody >= mechanics.bodies.size() ||
                    mechanics.bodies[localBody].articulationIndex !=
                        MR_INVALID_INDEX) {
                    return std::string{owner} +
                        " reset state references a non-scene body";
                }
                MRBodyStateGPU state = source;
                state.flagsAndIndices[2] = bodyOffset + localBody;
                defaultSceneBodies.push_back(state);
            }
            return std::nullopt;
        };
        if (const auto error = appendStates(
                manifest.robot.mechanics,
                manifest.robot.defaultSceneBodies,
                stateBodyOffset,
                manifest.robot.id
            )) {
            return reject(
                RunCompileStatus::invalidRobot,
                manifest.robot.id,
                *error
            );
        }
        stateBodyOffset += static_cast<std::uint32_t>(
            manifest.robot.mechanics.bodies.size()
        );
        for (const SceneObject& object : manifest.scene.objects) {
            if (const auto error = appendStates(
                    object.mechanics,
                    object.defaultBodyStates,
                    stateBodyOffset,
                    object.id
                )) {
                return reject(
                    RunCompileStatus::invalidManifest,
                    object.id,
                    *error
                );
            }
            stateBodyOffset += static_cast<std::uint32_t>(
                object.mechanics.bodies.size()
            );
        }

        EpisodeTwin episode;
        episode.id = manifest.id;
        if (!manifest.scene.authoredAssets.empty()) {
            if (!manifest.scene.objects.empty()) {
                return reject(
                    RunCompileStatus::invalidManifest,
                    manifest.scene.id,
                    "ScenePack cannot mix authored asset bindings with "
                    "composable objects"
                );
            }
            episode.assets = manifest.scene.authoredAssets;
        } else {
            WorldAsset robotAsset;
            robotAsset.id = manifest.robot.id;
            robotAsset.semanticClass = "robot";
            robotAsset.role = MR_WORLD_ASSET_ROBOT;
            robotAsset.collision = collisionRepresentation(
                manifest.robot.mechanics
            );
            robotAsset.dynamics = MR_WORLD_DYNAMICS_ARTICULATED;
            robotAsset.articulationIndex =
                manifest.robot.primaryArticulationIndex;
            robotAsset.bodyIndices = indices(manifest.robot.mechanics.bodies);
            robotAsset.shapeIndices = indices(manifest.robot.mechanics.shapes);
            robotAsset.materialIndices = indices(manifest.robot.mechanics.materials);
            episode.assets.push_back(std::move(robotAsset));

            std::uint32_t bodyOffset = static_cast<std::uint32_t>(
                manifest.robot.mechanics.bodies.size()
            );
            std::uint32_t shapeOffset = static_cast<std::uint32_t>(
                manifest.robot.mechanics.shapes.size()
            );
            std::uint32_t materialOffset = static_cast<std::uint32_t>(
                manifest.robot.mechanics.materials.size()
            );
            std::uint32_t articulationOffset = static_cast<std::uint32_t>(
                manifest.robot.mechanics.articulations.size()
            );
            for (const SceneObject& object : manifest.scene.objects) {
                WorldAsset asset;
                asset.id = object.id;
                asset.semanticClass = object.semanticClass;
                asset.role = object.role;
                asset.render = object.render;
                asset.collision = object.collision;
                asset.dynamics = object.dynamics;
                asset.articulationIndex = object.mechanics.articulations.empty()
                    ? MR_INVALID_INDEX
                    : articulationOffset;
                asset.bodyIndices = indices(object.mechanics.bodies, bodyOffset);
                asset.shapeIndices = indices(object.mechanics.shapes, shapeOffset);
                asset.materialIndices = indices(
                    object.mechanics.materials,
                    materialOffset
                );
                episode.assets.push_back(std::move(asset));
                bodyOffset += static_cast<std::uint32_t>(
                    object.mechanics.bodies.size()
                );
                shapeOffset += static_cast<std::uint32_t>(
                    object.mechanics.shapes.size()
                );
                materialOffset += static_cast<std::uint32_t>(
                    object.mechanics.materials.size()
                );
                articulationOffset += static_cast<std::uint32_t>(
                    object.mechanics.articulations.size()
                );
            }
        }

        episode.sensors = manifest.sensors.worldSensors;
        std::unordered_set<std::string> sensorIds;
        for (const SensorSpec& sensor : episode.sensors) {
            if (!sensorIds.insert(sensor.id).second) {
                return reject(RunCompileStatus::invalidManifest, sensor.id,
                    "sensor identity is duplicated");
            }
        }
        for (const MountedSensor& mounted : manifest.sensors.mounted) {
            const RobotSemanticRole* mount = role(
                manifest.robot,
                mounted.mountRole,
                RobotSemanticKind::body
            );
            if (mount == nullptr || mount->members.size() != 1u) {
                return reject(
                    RunCompileStatus::unresolvedRole,
                    mounted.mountRole,
                    "sensor mount must resolve to exactly one robot body"
                );
            }
            const auto body = std::ranges::find(
                manifest.robot.mechanics.bodyNames,
                mount->members.front()
            );
            SensorSpec sensor = mounted.sensor;
            if (sensor.id.empty() || !sensorIds.insert(sensor.id).second) {
                return reject(RunCompileStatus::invalidManifest, sensor.id,
                    "mounted sensor identity is empty or duplicated");
            }
            sensor.parentAssetId = manifest.robot.id;
            sensor.parentBodyIndex = static_cast<std::uint32_t>(
                body - manifest.robot.mechanics.bodyNames.begin()
            );
            sensor.parentKind = MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK;
            episode.sensors.push_back(std::move(sensor));
        }
        episode.task = {
            .id = manifest.task.id,
            .robotAssetId = manifest.robot.id,
            .controlPeriodSeconds = manifest.profile.controlTimestepSeconds,
            .horizonSeconds = static_cast<double>(
                manifest.profile.controlTimestepSeconds
            ) * manifest.task.maximumEpisodeSteps,
        };

        WorldTemplate worldTemplate;
        const WorldCompileResult twin = compileEpisodeTwin(
            episode,
            model,
            worldTemplate
        );
        if (!twin.succeeded()) {
            return reject(RunCompileStatus::worldFailure, "world", twin.message);
        }
        WorldProgram program = manifest.reality.program;
        if (program.id.empty()) {
            program.id = manifest.reality.id.empty()
                ? manifest.id + ".reality"
                : manifest.reality.id;
        }
        WorldFamily family;
        const WorldCompileResult familyStatus = compileWorldFamily(
            worldTemplate,
            program,
            family
        );
        if (!familyStatus.succeeded()) {
            return reject(RunCompileStatus::worldFailure, "reality",
                familyStatus.message);
        }
        if (manifest.sensors.observation.actorFrame.empty() ||
            manifest.sensors.observation.critic.empty()) {
            return reject(
                RunCompileStatus::invalidManifest,
                manifest.sensors.id,
                "SensorPack must author executable actor and critic observations"
            );
        }
        TaskResetProgram executableReset;
        if (const auto realityError = lowerRealityProgram(
                worldTemplate,
                manifest.profile,
                manifest.reality,
                executableReset
            )) {
            return reject(
                RunCompileStatus::invalidManifest,
                manifest.reality.id,
                *realityError
            );
        }

        CompiledRun staged;
        const MetalWorldCapacityProfile capacities =
            anyCapacity(manifest.profile.capacities)
            ? manifest.profile.capacities
            : manifest.task.capacities;
        const MetalWorldCompileDiagnostics worldStatus = compileMetalWorld(
            model,
            manifest.robot.primaryArticulationIndex,
            staged.world_,
            capacities
        );
        if (!worldStatus.succeeded()) {
            return reject(RunCompileStatus::worldFailure, "physics",
                worldStatus.message);
        }
        const TaskCompileDiagnostics taskStatus =
            manifest.teacher.interactions
            ? compileTaskProgram(
                  manifest.task,
                  manifest.robot.actuators,
                  manifest.sensors.observation,
                  executableReset,
                  *manifest.teacher.interactions,
                  manifest.teacher.interactionClip,
                  staged.world_,
                  staged.task_
              )
            : compileTaskProgram(
                  manifest.task,
                  manifest.robot.actuators,
                  manifest.sensors.observation,
                  executableReset,
                  staged.world_,
                  staged.task_
              );
        if (!taskStatus.succeeded()) {
            return reject(RunCompileStatus::taskFailure, taskStatus.element,
                taskStatus.message);
        }
        if (manifest.robot.multicopter) {
            const MulticopterActuatorPack& authored =
                *manifest.robot.multicopter;
            const RobotSemanticRole* bodyRole = role(
                manifest.robot,
                authored.bodyRole,
                RobotSemanticKind::body
            );
            if (bodyRole == nullptr || bodyRole->members.size() != 1u ||
                authored.model.rotorCount == 0u ||
                authored.model.rotorCount > MR_MULTICOPTER_MAX_ROTORS) {
                return reject(
                    RunCompileStatus::invalidRobot,
                    manifest.robot.id + ".multicopter",
                    "multicopter program requires one body role and a valid rotor model"
                );
            }
            // model_ is published below; resolve against the composed model
            // that the task and MetalWorld already compiled.
            const auto composedBody = std::ranges::find(
                model.bodyNames,
                bodyRole->members.front()
            );
            std::array<std::uint32_t, 4u> actionByComponent{
                MR_INVALID_INDEX, MR_INVALID_INDEX,
                MR_INVALID_INDEX, MR_INVALID_INDEX,
            };
            for (std::uint32_t action = 0u;
                 action < manifest.task.actions.size();
                 ++action) {
                const auto actuator = std::ranges::find_if(
                    manifest.robot.actuators,
                    [&](const RobotActuatorSpec& value) {
                        return value.id ==
                            manifest.task.actions[action].actuator;
                    }
                );
                if (actuator == manifest.robot.actuators.end() ||
                    actuator->kind != RobotActuatorKind::rotorMixer) {
                    continue;
                }
                if (actuator->component >= actionByComponent.size() ||
                    actionByComponent[actuator->component] !=
                        MR_INVALID_INDEX ||
                    actuator->target != bodyRole->members.front()) {
                    return reject(
                        RunCompileStatus::invalidRobot,
                        actuator->id,
                        "rotor mixer components must uniquely bind the multicopter body"
                    );
                }
                actionByComponent[actuator->component] = action;
            }
            if (composedBody == model.bodyNames.end() ||
                !std::ranges::equal(
                    actionByComponent,
                    std::array<std::uint32_t, 4u>{0u, 1u, 2u, 3u}
                )) {
                return reject(
                    RunCompileStatus::invalidRobot,
                    manifest.robot.id + ".multicopter",
                    "multicopter body or canonical four-lane task action contract is unresolved"
                );
            }
            staged.multicopterProgram_ = MetalWorldMulticopterProgram{
                .model = authored.model,
                .rotors = authored.rotors,
                .mixer = authored.mixer,
                .articulationIndex = manifest.robot.primaryArticulationIndex,
                .bodyIndex = static_cast<std::uint32_t>(
                    composedBody - model.bodyNames.begin()
                ),
                .firstAction = 0u,
                .windVelocity = authored.windVelocity,
            };
            staged.multicopterProgram_->model.motorAndTimestep.w =
                manifest.profile.controlTimestepSeconds /
                static_cast<float>(manifest.profile.physicsSubsteps);
        }
        if (manifest.robot.flappingWings) {
            if (manifest.robot.multicopter) {
                return reject(
                    RunCompileStatus::invalidRobot,
                    manifest.robot.id + ".aerodynamics",
                    "one robot cannot bind both multicopter and flapping-wing programs"
                );
            }
            const FlappingWingActuatorPack& authored =
                *manifest.robot.flappingWings;
            const RobotSemanticRole* bodyRole = role(
                manifest.robot, authored.bodyRole, RobotSemanticKind::body
            );
            if (bodyRole == nullptr || bodyRole->members.size() != 1u ||
                !(authored.windVelocityAndDensity.w > 0.0f) ||
                !std::isfinite(authored.windVelocityAndDensity.x) ||
                !std::isfinite(authored.windVelocityAndDensity.y) ||
                !std::isfinite(authored.windVelocityAndDensity.z) ||
                !std::isfinite(authored.windVelocityAndDensity.w)) {
                return reject(
                    RunCompileStatus::invalidRobot,
                    manifest.robot.id + ".flapping_wings",
                    "flapping-wing program requires one airframe body and finite positive air density"
                );
            }
            const auto root = std::ranges::find(
                model.bodyNames, bodyRole->members.front()
            );
            if (root == model.bodyNames.end()) {
                return reject(
                    RunCompileStatus::invalidRobot,
                    manifest.robot.id + ".flapping_wings",
                    "flapping-wing airframe body is unresolved after composition"
                );
            }
            MetalWorldFlappingWingProgram program{};
            program.articulationIndex = manifest.robot.primaryArticulationIndex;
            program.rootBodyIndex = static_cast<std::uint32_t>(
                root - model.bodyNames.begin()
            );
            program.windVelocityAndDensity = authored.windVelocityAndDensity;
            for (std::size_t side = 0u; side < program.wings.size(); ++side) {
                const RobotSemanticRole* wingRole = role(
                    manifest.robot, authored.wingRoles[side],
                    RobotSemanticKind::body
                );
                if (wingRole == nullptr || wingRole->members.size() != 1u) {
                    return reject(
                        RunCompileStatus::invalidRobot,
                        manifest.robot.id + ".flapping_wings",
                        "each flapping-wing side requires one resolved body role"
                    );
                }
                const auto wing = std::ranges::find(
                    model.bodyNames, wingRole->members.front()
                );
                if (wing == model.bodyNames.end()) {
                    return reject(
                        RunCompileStatus::invalidRobot,
                        manifest.robot.id + ".flapping_wings",
                        "flapping-wing body is unresolved after composition"
                    );
                }
                const std::uint32_t bodyIndex = static_cast<std::uint32_t>(
                    wing - model.bodyNames.begin()
                );
                if (bodyIndex >= model.bodies.size() ||
                    model.bodies[bodyIndex].parentBody !=
                        program.rootBodyIndex ||
                    model.bodies[bodyIndex].inboundJoint >=
                        model.joints.size()) {
                    return reject(
                        RunCompileStatus::invalidRobot,
                        manifest.robot.id + ".flapping_wings",
                        "each wing must be a direct articulated child of the airframe"
                    );
                }
                const MRJointDescriptorGPU& joint = model.joints[
                    model.bodies[bodyIndex].inboundJoint
                ];
                MRFlappingWingGPU resolved = authored.wings[side];
                if (!(resolved.rootToCenterAndArea.w > 0.0f) ||
                    !(resolved.hingeAxisAndChord.w > 0.0f) ||
                    !(resolved.coefficients.x > 0.0f) ||
                    resolved.coefficients.y < 0.0f ||
                    resolved.coefficients.z < 0.0f ||
                    !(resolved.coefficients.w > 0.0f) ||
                    !std::isfinite(resolved.unsteadyCoefficients.x) ||
                    resolved.unsteadyCoefficients.x < 0.0f) {
                    return reject(
                        RunCompileStatus::invalidRobot,
                        manifest.robot.id + ".flapping_wings",
                        "wing geometry and aerodynamic coefficients must be positive"
                    );
                }
                resolved.bodyIndex = bodyIndex;
                resolved.qIndex = joint.qOffset;
                resolved.vIndex = joint.vOffset;
                program.wings[side] = resolved;
            }
            const RobotSemanticRole* tailRole = role(
                manifest.robot, authored.tailRole, RobotSemanticKind::body
            );
            if (tailRole == nullptr || tailRole->members.size() != 1u) {
                return reject(
                    RunCompileStatus::invalidRobot,
                    manifest.robot.id + ".flapping_wings",
                    "flapping-wing program requires one resolved tail body role"
                );
            }
            const auto tail = std::ranges::find(
                model.bodyNames, tailRole->members.front()
            );
            const std::uint32_t tailBodyIndex = tail == model.bodyNames.end()
                ? MR_INVALID_INDEX
                : static_cast<std::uint32_t>(tail - model.bodyNames.begin());
            if (tailBodyIndex >= model.bodies.size() ||
                model.bodies[tailBodyIndex].parentBody !=
                    program.rootBodyIndex ||
                model.bodies[tailBodyIndex].inboundJoint >=
                    model.joints.size()) {
                return reject(
                    RunCompileStatus::invalidRobot,
                    manifest.robot.id + ".flapping_wings",
                    "tail must be a direct articulated child of the airframe"
                );
            }
            const MRJointDescriptorGPU& tailJoint = model.joints[
                model.bodies[tailBodyIndex].inboundJoint
            ];
            if (tailJoint.jointType != MR_JOINT_FIXED &&
                (tailJoint.jointType != MR_JOINT_REVOLUTE ||
                 tailJoint.nq != 1u || tailJoint.nv != 1u ||
                 std::abs(tailJoint.axis0.x) > 1.0e-5f ||
                 std::abs(std::abs(tailJoint.axis0.y) - 1.0f) > 1.0e-5f ||
                 std::abs(tailJoint.axis0.z) > 1.0e-5f)) {
                return reject(
                    RunCompileStatus::invalidRobot,
                    manifest.robot.id + ".flapping_wings",
                    "tail must be fixed or a single direct pitch joint"
                );
            }
            MRAeroTailGPU resolvedTail = authored.tail;
            if (!std::isfinite(resolvedTail.rootToCenterAndArea.x) ||
                !std::isfinite(resolvedTail.rootToCenterAndArea.y) ||
                !std::isfinite(resolvedTail.rootToCenterAndArea.z) ||
                !std::isfinite(resolvedTail.rootToCenterAndArea.w) ||
                !std::isfinite(resolvedTail.chordAndCoefficients.x) ||
                !std::isfinite(resolvedTail.chordAndCoefficients.y) ||
                !std::isfinite(resolvedTail.chordAndCoefficients.z) ||
                !std::isfinite(resolvedTail.chordAndCoefficients.w) ||
                !(resolvedTail.rootToCenterAndArea.w > 0.0f) ||
                !(resolvedTail.chordAndCoefficients.x > 0.0f) ||
                !(resolvedTail.chordAndCoefficients.y > 0.0f) ||
                resolvedTail.chordAndCoefficients.z < 0.0f ||
                resolvedTail.chordAndCoefficients.w < 0.0f) {
                return reject(
                    RunCompileStatus::invalidRobot,
                    manifest.robot.id + ".flapping_wings",
                    "tail geometry, lift, drag, and pitch damping must be finite and non-negative"
                );
            }
            resolvedTail.bodyIndex = tailBodyIndex;
            resolvedTail.rootBodyIndex = program.rootBodyIndex;
            resolvedTail.qIndex = tailJoint.jointType == MR_JOINT_REVOLUTE
                ? tailJoint.qOffset : MR_INVALID_INDEX;
            resolvedTail.vIndex = tailJoint.jointType == MR_JOINT_REVOLUTE
                ? tailJoint.vOffset : MR_INVALID_INDEX;
            program.tail = resolvedTail;
            MRAeroFuselageGPU resolvedFuselage = authored.fuselage;
            if (!std::isfinite(resolvedFuselage.referenceAreasAndDrag.x) ||
                !std::isfinite(resolvedFuselage.referenceAreasAndDrag.y) ||
                !std::isfinite(resolvedFuselage.referenceAreasAndDrag.z) ||
                !std::isfinite(resolvedFuselage.referenceAreasAndDrag.w) ||
                !std::isfinite(resolvedFuselage.angularDamping.x) ||
                !std::isfinite(resolvedFuselage.angularDamping.y) ||
                !std::isfinite(resolvedFuselage.angularDamping.z) ||
                !std::isfinite(resolvedFuselage.angularDamping.w) ||
                !(resolvedFuselage.referenceAreasAndDrag.x > 0.0f) ||
                !(resolvedFuselage.referenceAreasAndDrag.y > 0.0f) ||
                !(resolvedFuselage.referenceAreasAndDrag.z > 0.0f) ||
                resolvedFuselage.referenceAreasAndDrag.w < 0.0f ||
                resolvedFuselage.angularDamping.x < 0.0f ||
                resolvedFuselage.angularDamping.y < 0.0f ||
                resolvedFuselage.angularDamping.z < 0.0f) {
                return reject(
                    RunCompileStatus::invalidRobot,
                    manifest.robot.id + ".flapping_wings",
                    "fuselage reference areas, drag, and angular damping must be finite and non-negative"
                );
            }
            resolvedFuselage.bodyIndex = program.rootBodyIndex;
            resolvedFuselage.rootBodyIndex = program.rootBodyIndex;
            program.fuselage = resolvedFuselage;
            staged.flappingWingProgram_ = program;
        }
        if (manifest.policy) {
            const PolicyCompileDiagnostics policyStatus =
                compilePolicyProgram(
                    *manifest.policy,
                    staged.task_,
                    staged.policy_
                );
            if (!policyStatus.succeeded()) {
                return reject(RunCompileStatus::policyFailure,
                    policyStatus.element, policyStatus.message);
            }
            staged.boundPolicy_ = *manifest.policy;
        }
        staged.worldFamily_ = std::move(family);
        staged.model_ = std::move(model);
        staged.defaultSceneBodies_ = std::move(defaultSceneBodies);
        staged.profile_ = manifest.profile;
        staged.profile_.physics = compose;
        staged.teacher_ = manifest.teacher;
        staged.visualSensorProgram_ = manifest.sensors.deviceVisual;
        staged.robotFingerprint_ = robotPackFingerprint(manifest.robot);
        {
            Hash sensorHash;
            sensorHash.scalar(sensorFingerprint(episode.sensors));
            sensorHash.scalar(staged.task_.observationFingerprint());
            sensorHash.scalar(
                manifest.sensors.deviceVisual
                ? manifest.sensors.deviceVisual->fingerprint
                : 0u
            );
            staged.sensorFingerprint_ = sensorHash.finish();
        }
        {
            Hash realityHash;
            realityHash.scalar(staged.worldFamily_.program.fingerprint);
            realityHash.scalar(
                manifest.reality.sourceProgramFingerprint
            );
            realityHash.scalar<std::uint64_t>(
                staged.task_.randomizationOperators().size()
            );
            if (!staged.task_.randomizationOperators().empty()) {
                realityHash.bytes(
                    staged.task_.randomizationOperators().data(),
                    staged.task_.randomizationOperators().size_bytes()
                );
            }
            realityHash.scalar(executableReset.maximumActionDelaySteps);
            realityHash.scalar(
                executableReset.maximumObservationDelaySteps
            );
            staged.realityFingerprint_ = realityHash.finish();
        }
        {
            Hash teacherHash;
            teacherHash.string(staged.teacher_.id);
            teacherHash.scalar(staged.teacher_.kind);
            teacherHash.string(staged.teacher_.provider);
            teacherHash.string(staged.teacher_.model);
            teacherHash.scalar(staged.teacher_.artifactFingerprint);
            teacherHash.string(staged.teacher_.interactionClip);
            teacherHash.scalar(staged.task_.fingerprint());
            staged.teacherFingerprint_ = teacherHash.finish();
        }
        Hash runHash;
        runHash.string(manifest.id);
        runHash.scalar(staged.robotFingerprint_);
        runHash.scalar(staged.sensorFingerprint_);
        runHash.scalar(staged.realityFingerprint_);
        runHash.scalar(staged.teacherFingerprint_);
        runHash.scalar(staged.worldFamily_.fingerprint);
        runHash.scalar(staged.world_.fingerprint());
        runHash.scalar<std::uint64_t>(staged.defaultSceneBodies_.size());
        if (!staged.defaultSceneBodies_.empty()) {
            runHash.bytes(
                staged.defaultSceneBodies_.data(),
                staged.defaultSceneBodies_.size() *
                    sizeof(MRBodyStateGPU)
            );
        }
        runHash.scalar(staged.task_.fingerprint());
        runHash.scalar(staged.policy_.fingerprint());
        if (staged.multicopterProgram_) {
            const MetalWorldMulticopterProgram& multicopter =
                *staged.multicopterProgram_;
            runHash.bytes(&multicopter.model, sizeof(multicopter.model));
            runHash.bytes(
                multicopter.rotors.data(),
                sizeof(multicopter.rotors)
            );
            runHash.bytes(&multicopter.mixer, sizeof(multicopter.mixer));
            runHash.scalar(multicopter.articulationIndex);
            runHash.scalar(multicopter.bodyIndex);
            runHash.scalar(multicopter.firstAction);
            runHash.scalar(multicopter.windVelocity);
        }
        if (staged.flappingWingProgram_) {
            const MetalWorldFlappingWingProgram& wings =
                *staged.flappingWingProgram_;
            runHash.bytes(wings.wings.data(), sizeof(wings.wings));
            runHash.bytes(&wings.tail, sizeof(wings.tail));
            runHash.bytes(&wings.fuselage, sizeof(wings.fuselage));
            runHash.scalar(wings.articulationIndex);
            runHash.scalar(wings.rootBodyIndex);
            runHash.scalar(wings.windVelocityAndDensity);
        }
        runHash.string(staged.profile_.id);
        runHash.scalar(staged.profile_.environmentCount);
        runHash.scalar(staged.profile_.controlSteps);
        runHash.scalar(staged.profile_.physicsSubsteps);
        runHash.scalar(staged.profile_.velocityIterations);
        runHash.scalar(staged.profile_.finalVelocityIterations);
        runHash.scalar(staged.profile_.controlTimestepSeconds);
        runHash.scalar(staged.profile_.seed);
        runHash.scalar(staged.profile_.streamedArticulatedContactResponses);
        runHash.scalar(staged.profile_.minimumDifficultyBand);
        runHash.scalar(staged.profile_.maximumDifficultyBand);
        runHash.bytes(
            &staged.profile_.capacities,
            sizeof(staged.profile_.capacities)
        );
        staged.fingerprint_ = runHash.finish();
        output = std::move(staged);
        return {};
    } catch (const std::bad_alloc&) {
        return reject(RunCompileStatus::internalFailure, "allocation",
            "run compilation allocation failed");
    } catch (const std::exception& error) {
        return reject(RunCompileStatus::internalFailure, "exception", error.what());
    }
}

std::vector<std::string> builtinRobotIds() {
    return {"unitree_g1", "franka_panda", "dvrk_psm", "px4_x500",
            "birdflow_deetjen_dove_hybrid",
            "birdflow_american_crow_estimated_hybrid"};
}

ScenePack makePX4X500HoverScenePack() {
    const auto robot = builtinRobotPack("px4_x500");
    if (!robot) {
        throw std::logic_error("bundled PX4 X500 RobotPack is unavailable");
    }
    LocomotionSceneComponent ground = makeLocomotionSurfaceComponent(
        robot->mechanics,
        LocomotionSurface::ground
    );
    ScenePack scene;
    scene.id = "px4_x500_hover_scene";
    scene.objects.push_back({
        .id = "flight_ground",
        .semanticClass = "ground",
        .role = MR_WORLD_ASSET_FIXTURE,
        .render = MR_WORLD_RENDER_PROCEDURAL,
        .collision = MR_WORLD_COLLISION_PRIMITIVES,
        .dynamics = MR_WORLD_DYNAMICS_STATIC,
        .mechanics = std::move(ground.mechanics),
        .defaultBodyStates = std::move(ground.defaultBodyStates),
    });
    return scene;
}

ScenePack makeBirdFlowDoveFlightScenePack() {
    const auto robot = builtinRobotPack("birdflow_deetjen_dove_hybrid");
    if (!robot) {
        throw std::logic_error("bundled BirdFlow dove RobotPack is unavailable");
    }
    LocomotionSceneComponent ground = makeLocomotionSurfaceComponent(
        robot->mechanics,
        LocomotionSurface::ground
    );
    ScenePack scene;
    scene.id = "birdflow_deetjen_dove_hybrid_flight_scene";
    scene.objects.push_back({
        .id = "flight_ground",
        .semanticClass = "ground",
        .role = MR_WORLD_ASSET_FIXTURE,
        .render = MR_WORLD_RENDER_PROCEDURAL,
        .collision = MR_WORLD_COLLISION_PRIMITIVES,
        .dynamics = MR_WORLD_DYNAMICS_STATIC,
        .mechanics = std::move(ground.mechanics),
        .defaultBodyStates = std::move(ground.defaultBodyStates),
    });
    return scene;
}

ScenePack makeBirdFlowAmericanCrowFlightScenePack() {
    const auto robot = builtinRobotPack(
        "birdflow_american_crow_estimated_hybrid"
    );
    if (!robot) {
        throw std::logic_error(
            "bundled BirdFlow American-crow RobotPack is unavailable"
        );
    }
    LocomotionSceneComponent ground = makeLocomotionSurfaceComponent(
        robot->mechanics,
        LocomotionSurface::ground
    );
    ScenePack scene;
    scene.id = "birdflow_american_crow_estimated_hybrid_flight_scene";
    scene.objects.push_back({
        .id = "flight_ground",
        .semanticClass = "ground",
        .role = MR_WORLD_ASSET_FIXTURE,
        .render = MR_WORLD_RENDER_PROCEDURAL,
        .collision = MR_WORLD_COLLISION_PRIMITIVES,
        .dynamics = MR_WORLD_DYNAMICS_STATIC,
        .mechanics = std::move(ground.mechanics),
        .defaultBodyStates = std::move(ground.defaultBodyStates),
    });
    return scene;
}

TaskPack makePX4X500HoverTaskPack(
    TaskObservationProgram& observations,
    TaskResetProgram& reset
) {
    TaskPack task;
    task.id = "px4_x500_hover";
    task.actions = {
        {"rotor_mixer.collective"},
        {"rotor_mixer.roll"},
        {"rotor_mixer.pitch"},
        {"rotor_mixer.yaw"},
    };
    task.outcomes = {
        {"root_height", "m", TaskOutcomeSource::rootHeight,
            TaskOutcomeDirection::neutral},
        {"tilt", "rad", TaskOutcomeSource::tilt,
            TaskOutcomeDirection::lowerIsBetter},
        {"tracking", "ratio", TaskOutcomeSource::trackingScore,
            TaskOutcomeDirection::higherIsBetter},
    };
    task.capacities = {
        .candidatePairs = 32u,
        .rawContacts = 64u,
        .manifolds = 16u,
        .constraintBlocks = 32u,
        .constraintRows = 96u,
        .hardConvexPairs = 32u,
        .ccdCandidates = 16u,
        .ccdEvents = 4u,
        .endpointRuntimeRecords = 64u,
        .qualityRows = 96u,
        .islandConstraintReferences = 32u,
    };
    task.maximumEpisodeSteps = 1000u;
    task.difficultyBandCount = 1u;
    task.baseHeightTarget = 2.0f;
    task.gaitPeriodSeconds = 1.0f;
    task.successTrackingThreshold = 0.85f;
    task.supportForceThreshold = 0.0f;
    task.commands.lower = {};
    task.commands.upper = {};
    task.commands.limitLower = {};
    task.commands.limitUpper = {};
    task.commands.difficultyStep = {};
    task.commands.standingProbability = 1.0f;
    task.commands.minimumDurationSeconds = 20.0f;
    task.commands.maximumDurationSeconds = 20.0f;
    task.pushes.minimumIntervalSeconds = 20.0f;
    task.pushes.maximumIntervalSeconds = 20.0f;
    task.rewards = {
        {TaskRewardOperator::constant, {}, {}, 0.25f},
        {TaskRewardOperator::rootHeightErrorSquared, {}, {}, -1.0f},
        {TaskRewardOperator::tiltSquared, {}, {}, -0.5f},
        {TaskRewardOperator::rootVerticalVelocitySquared, {}, {}, -0.1f},
        {TaskRewardOperator::rootRollPitchVelocitySquared, {}, {}, -0.05f},
        {TaskRewardOperator::actionRateSquared, {}, {}, -0.01f},
    };
    task.terminations = {
        {TaskTerminationOperator::minimumRootHeight, {},
            MR_TASK_TERMINATION_HEIGHT, 10u, 0.15f, -1.0f},
        // Flight tasks must not let an exploratory controller accumulate
        // unbounded altitude before the next reset. This is an episode safety
        // boundary, not an aerodynamic clamp; the Metal solver still evolves
        // the same resolved wing/tail load up to the boundary.
        {TaskTerminationOperator::maximumRootHeight, {},
            MR_TASK_TERMINATION_HEIGHT, 11u, 2.50f, -1.0f},
        {TaskTerminationOperator::maximumTilt, {},
            MR_TASK_TERMINATION_TILT, 20u, 1.20f, -1.0f},
    };
    observations.actorHistoryLength = 1u;
    observations.criticHistoryLength = 1u;
    observations.criticIncludesCleanHistory = false;
    for (std::uint32_t component = 0u; component < 3u; ++component) {
        observations.actorFrame.push_back({
            .source = TaskObservationSource::rootLinearVelocityLocal,
            .component = component,
            .scale = 0.5f,
        });
        observations.actorFrame.push_back({
            .source = TaskObservationSource::rootAngularVelocityLocal,
            .component = component,
            .scale = 0.25f,
        });
        observations.actorFrame.push_back({
            .source = TaskObservationSource::projectedGravity,
            .component = component,
            .normalizeVector3 = true,
        });
    }
    observations.actorFrame.push_back({
        .source = TaskObservationSource::rootHeight,
        .scale = 0.5f,
        .offset = -1.0f,
    });
    for (const TaskActionBinding& action : task.actions) {
        observations.actorFrame.push_back({
            .source = TaskObservationSource::previousAction,
            .target = action.actuator,
        });
    }
    observations.critic = observations.actorFrame;
    reset.maximumActionDelaySteps = 0u;
    reset.maximumObservationDelaySteps = 0u;
    reset.operators = {
        {.operation = TaskRandomizationOperator::rootPosition,
            .parameters = {0.05f, 0.05f, 0.0f, 0.0f}},
        {.operation = TaskRandomizationOperator::rootYaw,
            .parameters = {-0.10f, 0.10f, 0.0f, 0.0f}},
        {.operation = TaskRandomizationOperator::rootHeight,
            .parameters = {1.95f, 2.05f, 0.0f, 0.0f}},
        {.operation = TaskRandomizationOperator::velocity,
            .parameters = {-0.02f, 0.02f, 0.0f, 0.0f}},
    };
    return task;
}

TaskPack makeBirdFlowDoveFlightTaskPack(
    TaskObservationProgram& observations,
    TaskResetProgram& reset
) {
    TaskPack task = makePX4X500HoverTaskPack(observations, reset);
    task.id = "birdflow_deetjen_dove_takeoff_flight_figure_eight";
    task.actions = {
        {"wing.left_flap"},
        {"wing.right_flap"},
        {"tail.pitch"},
        {"leg.left_hip"},
        {"leg.left_knee"},
        {"leg.left_ankle"},
        {"leg.right_hip"},
        {"leg.right_knee"},
        {"leg.right_ankle"},
        {"tail.yaw_moment"},
    };
    // PX4's reusable root-state prefix is meaningful here, but a Markov
    // flapping policy also needs the resolved hinge state.  Previous action
    // alone made the aerodynamic phase partially hidden, so the learner could
    // not distinguish a downstroke from an upstroke at the same airframe pose.
    // Rebuild the suffix against the compiled articulated-wing contract.
    observations.actorFrame.resize(10u);
    // Flight is conditioned on a forward-airframe velocity, not a world-fixed
    // hover point.  Keeping the sampled command in the actor frame makes
    // changes of speed and lateral/yaw request observable rather than hidden
    // task state.
    for (std::uint32_t component = 0u; component < 3u; ++component) {
        observations.actorFrame.push_back({
            .source = TaskObservationSource::command,
            .component = component,
        });
    }
    for (const char* joint : {
             "dove_left_wing_flap", "dove_right_wing_flap",
             "dove_tail_pitch"}) {
        observations.actorFrame.push_back({
            .source = TaskObservationSource::jointPositionError,
            .target = joint,
            .scale = 1.0f / 1.35f,
        });
        observations.actorFrame.push_back({
            .source = TaskObservationSource::jointVelocity,
            .target = joint,
            .scale = 1.0f / 25.0f,
        });
    }
    // The hinge state closes the instantaneous Markov loop; this explicit
    // clock lets a feed-forward policy modulate each phase of the robot-owned
    // wingbeat instead of treating a high-rate oscillator as hidden state.
    for (std::uint32_t component = 0u; component < 2u; ++component) {
        observations.actorFrame.push_back({
            .source = TaskObservationSource::cyclicPhase,
            .component = component,
        });
    }
    for (const char* action : {
             "wing.left_flap", "wing.right_flap", "tail.pitch"}) {
        observations.actorFrame.push_back({
            .source = TaskObservationSource::previousAction,
            .target = action,
        });
    }
    // Preserve the complete 21-value flight actor prefix, then append the
    // new embodied leg state. This lets the qualified wing controller become
    // the first two lanes of the eight-action policy without changing what
    // its existing weights mean.
    for (const char* joint : {
             "dove_left_hip_pitch", "dove_left_knee_pitch",
             "dove_left_ankle_pitch", "dove_right_hip_pitch",
             "dove_right_knee_pitch", "dove_right_ankle_pitch"}) {
        observations.actorFrame.push_back({
            .source = TaskObservationSource::jointPositionError,
            .target = joint,
            .scale = 1.0f / 1.35f,
        });
        observations.actorFrame.push_back({
            .source = TaskObservationSource::jointVelocity,
            .target = joint,
            .scale = 1.0f / 25.0f,
        });
    }
    for (const char* action : {
             "leg.left_hip", "leg.left_knee", "leg.left_ankle",
             "leg.right_hip", "leg.right_knee", "leg.right_ankle",
             "tail.yaw_moment"}) {
        observations.actorFrame.push_back({
            .source = TaskObservationSource::previousAction,
            .target = action,
        });
    }
    task.contactGroups = {
        {
            .id = "left_foot_contact",
            .bodies = {"dove_left_foot"},
            .support = true,
            .referenceBody = "dove_left_foot",
            .localReference = {},
            .gaitPhaseOffsetRadians = 0.0f,
            .stanceFraction = 0.62f,
            .supportPatchBounds = {-0.045f, -0.015f, 0.045f, 0.015f},
            .supportPatchWidth = 2u,
            .supportPatchHeight = 2u,
        },
        {
            .id = "right_foot_contact",
            .bodies = {"dove_right_foot"},
            .support = true,
            .referenceBody = "dove_right_foot",
            .localReference = {},
            .gaitPhaseOffsetRadians = 3.14159265358979323846f,
            .stanceFraction = 0.62f,
            .supportPatchBounds = {-0.045f, -0.015f, 0.045f, 0.015f},
            .supportPatchWidth = 2u,
            .supportPatchHeight = 2u,
        },
        {
            .id = "non_foot_contact",
            .bodies = {
                "dove_body", "dove_left_wing", "dove_right_wing",
                "dove_tail", "dove_left_thigh", "dove_left_shank",
                "dove_right_thigh", "dove_right_shank",
            },
            .forbidden = true,
        },
    };
    task.jointGroups = {{
        .id = "legs",
        .joints = {
            "dove_left_hip_pitch", "dove_left_knee_pitch",
            "dove_left_ankle_pitch", "dove_right_hip_pitch",
            "dove_right_knee_pitch", "dove_right_ankle_pitch",
        },
    }};
    constexpr std::array<float, 13u> footLoadScales{
        0.02f, 0.02f, 0.02f,
        0.5f, 0.5f, 0.5f,
        20.0f, 20.0f, 200.0f,
        0.001f, 0.001f, 0.001f, 0.001f,
    };
    for (const char* group : {"left_foot_contact", "right_foot_contact"}) {
        for (std::uint32_t component = 0u;
             component < footLoadScales.size(); ++component) {
            observations.actorFrame.push_back({
                .source = TaskObservationSource::supportPatch,
                .target = group,
                .component = component,
                .scale = footLoadScales[component],
            });
        }
    }
    observations.critic = observations.actorFrame;
    task.outcomes = {
        {"forward_flight_tracking", "ratio",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::linearVelocityTracking},
        {"tracking", "ratio", TaskOutcomeSource::trackingScore,
            TaskOutcomeDirection::higherIsBetter},
        {"liftoff", "ratio",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::rootHeightNormalized},
        {"ground_support", "ratio",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::supportContactCount},
        {"walking_contact", "ratio",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::gaitContactMatch},
        {"push_off", "m/s",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::rootHeightProgress},
        {"figure_eight_tracking", "ratio",
            TaskOutcomeSource::rewardContribution,
            TaskOutcomeDirection::higherIsBetter,
            TaskRewardOperator::figureEightPathTracking},
        {"root_height", "m", TaskOutcomeSource::rootHeight,
            TaskOutcomeDirection::neutral},
    };
    task.baseHeightTarget = 1.0f;
    // A 10 Hz launch/flight clock lies inside the fast-flapping range of small
    // pigeons. Leg targets use a separate staged carrier below, so they do not
    // inherit this wing frequency. The
    // former 4 Hz oscillator could not support the measured 0.0234 m^2 wing
    // area without inventing an oversized aerodynamic surface.
    task.gaitPeriodSeconds = 0.10f;
    // Four complete circuits remain available after the visible takeoff.
    // The inspector paces these steps at control time instead of racing
    // through them before its first presented frame.
    task.maximumEpisodeSteps = 5'000u;
    task.difficultyBandCount = 4u;
    task.commands.difficultySamplingExponent = 1.5f;
    task.successTrackingThreshold = 0.70f;
    // The launch qualification begins from an exactly bilateral command.
    // Path curvature is then introduced by the observable figure-eight
    // reference rather than a hidden random lateral/yaw offset at reset.
    task.commands.lower = {0.90f, 0.0f, 0.0f, 0.0f};
    task.commands.upper = {0.90f, 0.0f, 0.0f, 0.0f};
    task.commands.limitLower = task.commands.lower;
    task.commands.limitUpper = task.commands.upper;
    task.commands.difficultyStep = {};
    task.commands.standingProbability = 0.0f;
    task.commands.minimumDurationSeconds = 6.0f;
    task.commands.maximumDurationSeconds = 6.0f;
    // Takeoff is a physical transition from terrain support to sustained
    // height. Signed height progress gives direct credit to the transition,
    // normalized height retains the result, and forward tracking turns the
    // climb into flight instead of a vertical bounce.
    task.rewards = {
        {TaskRewardOperator::supportContactCount, {}, {}, 0.60f},
        {TaskRewardOperator::gaitContactMatch, {}, {}, 0.45f},
        {TaskRewardOperator::footClearance, {}, {}, 0.30f,
            {0.018f, 1.5f, 0.0f, 0.0f}},
        {TaskRewardOperator::rootHeightProgress, {}, {}, 3.0f},
        {TaskRewardOperator::rootHeightNormalized, {}, {}, 1.0f},
        {TaskRewardOperator::rootHeightErrorSquared, {}, {}, -0.35f},
        {TaskRewardOperator::linearVelocityTracking, {}, {}, 2.0f,
            {0.35f, 0.0f, 0.0f, 0.0f}},
        {TaskRewardOperator::yawVelocityTracking, {}, {}, 1.0f,
            {0.30f, 0.0f, 0.0f, 0.0f}},
        {TaskRewardOperator::figureEightPathTracking, {}, {}, 1.5f,
            // The commanded curve must remain inside the authored 0.9 m/s
            // flight envelope.  A 2 m by 1 m Gerono loop over 20 seconds
            // peaks below that bound; the former 20 m by 10 m loop demanded
            // roughly 4 m/s and drove otherwise healthy policies out of the
            // visible flight volume.
            {2.0f, 1.0f, 20.0f, 3.2f}},
        {TaskRewardOperator::uprightness, {}, {}, 0.15f},
        // The Metal task gates this penalty to the supported standing and
        // walking bands.  Flight remains free to bank under its own path and
        // attitude objectives.
        {TaskRewardOperator::tiltSquared, {}, {}, -0.50f},
        {TaskRewardOperator::forbiddenContact, "non_foot_contact", {}, -1.0f},
        {TaskRewardOperator::jointGroupPostureAbsolute, "legs", {}, -0.015f},
        {TaskRewardOperator::rootVerticalVelocitySquared, {}, {}, -0.005f},
        {TaskRewardOperator::rootRollPitchVelocitySquared, {}, {}, -0.02f},
        {TaskRewardOperator::actionRateSquared, {}, {}, -0.001f},
    };
    // A ground strike, runaway climb, or unrecoverable attitude is an
    // informative episode end.  These bounds do not clamp the aerodynamic
    // solve; they keep training and the live inspector inside the authored
    // flight contract and restart from checkpoint state after failure.
    task.terminations = {
        {TaskTerminationOperator::minimumRootHeight, {},
            MR_TASK_TERMINATION_HEIGHT, 10u, 0.025f, -1.0f},
        {TaskTerminationOperator::maximumRootHeight, {},
            MR_TASK_TERMINATION_HEIGHT, 11u, 2.50f, -1.0f},
        {TaskTerminationOperator::maximumTilt, {},
            MR_TASK_TERMINATION_TILT, 20u, 1.20f, -1.0f},
        {TaskTerminationOperator::contactGroup, "non_foot_contact",
            MR_TASK_TERMINATION_CONTACT, 30u, 0.0f, -1.0f},
    };
    for (auto& randomization : reset.operators) {
        if (randomization.operation == TaskRandomizationOperator::rootYaw) {
            // The ground curriculum begins nose-forward. Heading variation
            // belongs after supported launch; random yaw at foot contact made
            // the preview read as a sideways start and obscured whether the
            // aerodynamic forward axis agreed with the authored beak axis.
            randomization.parameters = {0.0f, 0.0f, 0.0f, 0.0f};
        }
        if (randomization.operation == TaskRandomizationOperator::rootHeight) {
            // Default leg FK places the sole at z=0 when the root is 0.163 m.
            // Keep reset uncertainty inside the contact skin so every ground
            // curriculum episode begins physically supported.
            randomization.parameters = {0.1625f, 0.1635f, 0.0f, 0.0f};
        }
    }
    task.capacities = {
        .candidatePairs = 128u,
        .rawContacts = 192u,
        .manifolds = 48u,
        .constraintBlocks = 96u,
        .constraintRows = 288u,
        .hardConvexPairs = 64u,
        .ccdCandidates = 32u,
        .ccdEvents = 8u,
        .endpointRuntimeRecords = 192u,
        .qualityRows = 288u,
        .islandConstraintReferences = 96u,
    };
    return task;
}

TaskPack makeBirdFlowAmericanCrowFlightTaskPack(
    TaskObservationProgram& observations,
    TaskResetProgram& reset
) {
    TaskPack task = makeBirdFlowDoveFlightTaskPack(observations, reset);
    const auto crowIdentifier = [](std::string value) {
        std::size_t offset = 0u;
        while ((offset = value.find("dove", offset)) != std::string::npos) {
            value.replace(offset, 4u, "crow");
            offset += 4u;
        }
        return value;
    };
    task.id = "birdflow_american_crow_standing_to_flight";
    for (TaskActionBinding& action : task.actions) {
        action.actuator = crowIdentifier(std::move(action.actuator));
    }
    for (TaskContactGroup& group : task.contactGroups) {
        for (std::string& body : group.bodies) {
            body = crowIdentifier(std::move(body));
        }
        group.referenceBody = crowIdentifier(std::move(group.referenceBody));
    }
    for (TaskJointGroup& group : task.jointGroups) {
        for (std::string& joint : group.joints) {
            joint = crowIdentifier(std::move(joint));
        }
    }
    for (TaskObservationOperatorSpec& observation : observations.actorFrame) {
        observation.target = crowIdentifier(std::move(observation.target));
    }
    for (TaskObservationOperatorSpec& observation : observations.actorCurrent) {
        observation.target = crowIdentifier(std::move(observation.target));
    }
    for (TaskObservationOperatorSpec& observation : observations.critic) {
        observation.target = crowIdentifier(std::move(observation.target));
    }
    for (TaskRewardOperatorSpec& reward : task.rewards) {
        reward.target = crowIdentifier(std::move(reward.target));
    }
    // The imported profile locks a 4.6 Hz presentation wingbeat.  The
    // trainable airframe uses that selected estimate and keeps the full
    // ground-support, push-off, lift-off, and flight task path of the dove
    // hybrid rather than starting as an already airborne proxy.
    task.gaitPeriodSeconds = 1.0f / 4.6f;
    task.commands.lower.x = 0.75f;
    task.commands.upper.x = 0.75f;
    task.commands.limitLower.x = 0.75f;
    task.commands.limitUpper.x = 0.75f;
    task.baseHeightTarget = 0.85f;
    for (TaskRandomizationOperatorSpec& randomization : reset.operators) {
        if (randomization.operation == TaskRandomizationOperator::rootHeight) {
            // The crow's estimated 57 mm tarsus and scaled leg chain keep
            // both soles inside the contact skin at this root height.
            randomization.parameters = {0.1865f, 0.1875f, 0.0f, 0.0f};
        }
    }
    return task;
}

std::optional<RobotPack> builtinRobotPack(const std::string_view id) {
    if (id == "birdflow_deetjen_dove_hybrid") {
        // The Deetjen public surface sequence does not include the complete
        // same-specimen inertial record required for measured free flight.
        // These explicitly modelled body properties therefore define a
        // trainable hybrid, while its runtime loads remain native and state
        // responsive rather than prescribed replay forces.
        EngineModel mechanics = makeFreeSphereEngineModel();
        mechanics.name = "birdflow_deetjen_dove_hybrid";
        mechanics.world.gravityAndTimestep = {
            0.0f, 0.0f, -9.81f, 1.0f / 240.0f,
        };
        mechanics.articulations[0u].rootBody = 0u;
        mechanics.articulations[0u].firstBody = 0u;
        mechanics.articulations[0u].bodyCount = 10u;
        mechanics.articulations[0u].jointCount = 9u;
        mechanics.articulations[0u].nq = 16u;
        mechanics.articulations[0u].nv = 15u;
        mechanics.world.bodyCount = 10u;
        mechanics.world.jointCount = 9u;
        mechanics.world.nq = 16u;
        mechanics.world.nv = 15u;
        mechanics.bodies.resize(10u);
        MRBodyPropertiesGPU& body = mechanics.bodies[0u];
        body.articulationIndex = 0u;
        body.parentBody = MR_INVALID_INDEX;
        body.inboundJoint = MR_INVALID_INDEX;
        body.motionType = MR_MOTION_DYNAMIC;
        // Keep the complete bird at the original 0.32 kg flight mass: the
        // newly articulated legs take 32 g from the formerly lumped torso.
        body.massAndInverseMass = {0.2778f, 1.0f / 0.2778f, 0.0f, 0.0f};
        body.inertiaRow0 = {0.0021f, 0.0f, 0.0f, 0.0f};
        body.inertiaRow1 = {0.0f, 0.0054f, 0.0f, 0.0f};
        body.inertiaRow2 = {0.0f, 0.0f, 0.0061f, 0.0f};
        body.inverseInertiaRow0 = {1.0f / 0.0021f, 0.0f, 0.0f, 0.0f};
        body.inverseInertiaRow1 = {0.0f, 1.0f / 0.0054f, 0.0f, 0.0f};
        body.inverseInertiaRow2 = {0.0f, 0.0f, 1.0f / 0.0061f, 0.0f};
        // The public Deetjen sequence does not identify a free-flight tail
        // damping derivative. This explicit airframe closure represents the
        // resolved body's rotational aerodynamic damping; without it the
        // two single-DOF wings inject pitch energy but provide no stabilizer.
        body.dampingAndSpeedLimits = {0.04f, 0.50f, 30.0f, 30.0f};
        const auto configureWing = [&](const std::uint32_t bodyIndex,
                                       const std::uint32_t jointIndex) {
            MRBodyPropertiesGPU& wing = mechanics.bodies[bodyIndex];
            wing.articulationIndex = 0u;
            wing.parentBody = 0u;
            wing.inboundJoint = jointIndex;
            wing.motionType = MR_MOTION_DYNAMIC;
            wing.massAndInverseMass = {0.0001f, 10000.0f, 0.0f, 0.0f};
            wing.inertiaRow0 = {0.00000003f, 0.0f, 0.0f, 0.0f};
            wing.inertiaRow1 = {0.0f, 0.00000010f, 0.0f, 0.0f};
            wing.inertiaRow2 = {0.0f, 0.0f, 0.00000011f, 0.0f};
            wing.inverseInertiaRow0 = {1.0f / 0.00000003f, 0.0f, 0.0f, 0.0f};
            wing.inverseInertiaRow1 = {0.0f, 1.0f / 0.00000010f, 0.0f, 0.0f};
            wing.inverseInertiaRow2 = {0.0f, 0.0f, 1.0f / 0.00000011f, 0.0f};
            wing.dampingAndSpeedLimits = {0.02f, 0.02f, 30.0f, 45.0f};
        };
        configureWing(1u, 0u);
        configureWing(2u, 1u);
        MRBodyPropertiesGPU& tail = mechanics.bodies[3u];
        tail.articulationIndex = 0u;
        tail.parentBody = 0u;
        tail.inboundJoint = 2u;
        tail.motionType = MR_MOTION_DYNAMIC;
        tail.massAndInverseMass = {0.01f, 100.0f, 0.0f, 0.0f};
        tail.inertiaRow0 = {0.00002f, 0.0f, 0.0f, 0.0f};
        tail.inertiaRow1 = {0.0f, 0.00005f, 0.0f, 0.0f};
        tail.inertiaRow2 = {0.0f, 0.0f, 0.00006f, 0.0f};
        tail.inverseInertiaRow0 = {50000.0f, 0.0f, 0.0f, 0.0f};
        tail.inverseInertiaRow1 = {0.0f, 20000.0f, 0.0f, 0.0f};
        tail.inverseInertiaRow2 = {0.0f, 0.0f, 16666.666f, 0.0f};
        tail.dampingAndSpeedLimits = {0.02f, 0.02f, 30.0f, 45.0f};
        const auto configureLegBody = [&](const std::uint32_t bodyIndex,
                                          const std::uint32_t parentBody,
                                          const std::uint32_t jointIndex,
                                          const float mass,
                                          const mr_float4 inertia) {
            MRBodyPropertiesGPU& leg = mechanics.bodies[bodyIndex];
            leg.articulationIndex = 0u;
            leg.parentBody = parentBody;
            leg.inboundJoint = jointIndex;
            leg.motionType = MR_MOTION_DYNAMIC;
            leg.massAndInverseMass = {mass, 1.0f / mass, 0.0f, 0.0f};
            leg.inertiaRow0 = {inertia.x, 0.0f, 0.0f, 0.0f};
            leg.inertiaRow1 = {0.0f, inertia.y, 0.0f, 0.0f};
            leg.inertiaRow2 = {0.0f, 0.0f, inertia.z, 0.0f};
            leg.inverseInertiaRow0 = {1.0f / inertia.x, 0.0f, 0.0f, 0.0f};
            leg.inverseInertiaRow1 = {0.0f, 1.0f / inertia.y, 0.0f, 0.0f};
            leg.inverseInertiaRow2 = {0.0f, 0.0f, 1.0f / inertia.z, 0.0f};
            leg.dampingAndSpeedLimits = {0.015f, 0.015f, 30.0f, 35.0f};
        };
        configureLegBody(4u, 0u, 3u, 0.007f,
            {0.0000030f, 0.0000030f, 0.0000010f, 0.0f});
        configureLegBody(5u, 4u, 4u, 0.005f,
            {0.0000025f, 0.0000025f, 0.0000008f, 0.0f});
        configureLegBody(6u, 5u, 5u, 0.004f,
            {0.0000010f, 0.0000025f, 0.0000025f, 0.0f});
        configureLegBody(7u, 0u, 6u, 0.007f,
            {0.0000030f, 0.0000030f, 0.0000010f, 0.0f});
        configureLegBody(8u, 7u, 7u, 0.005f,
            {0.0000025f, 0.0000025f, 0.0000008f, 0.0f});
        configureLegBody(9u, 8u, 8u, 0.004f,
            {0.0000010f, 0.0000025f, 0.0000025f, 0.0f});
        mechanics.joints.resize(9u);
        const auto configureJoint = [&](const std::uint32_t index,
                                        const float side) {
            MRJointDescriptorGPU& joint = mechanics.joints[index];
            joint.parentBody = 0u;
            joint.childBody = index + 1u;
            joint.jointType = MR_JOINT_REVOLUTE;
            joint.qOffset = 7u + index;
            joint.nq = 1u;
            joint.vOffset = 6u + index;
            joint.nv = 1u;
            // Bilateral shoulders are body-sagittal mirrors.  Mirroring the
            // span without mirroring the hinge axis makes equal commands
            // scissor the wings instead of producing one symmetric stroke.
            joint.axis0 = {side, 0.0f, 0.0f, 0.0f};
            // Source-derived shoulder frame. The former +/-0.06 m body
            // anchor and +/-0.24 m wing origin were collision-box guesses;
            // rotating the measured surface around them tore its root away
            // from the body. These mirrored anchors place the hinge through
            // the measured body-wing seam and the child origin at the
            // measured wing bounding-box centre.
            joint.parentAnchor = {
                -0.0195f, side * 0.0196f, 0.0165f, 0.0f
            };
            joint.childAnchor = {
                -0.0195f, -side * 0.0477765f, 0.0103564f, 0.0f
            };
            joint.parentRotation = {0.0f, 0.0f, 0.0f, 1.0f};
            joint.childRotation = {0.0f, 0.0f, 0.0f, 1.0f};
        };
        configureJoint(0u, 1.0f);
        configureJoint(1u, -1.0f);
        MRJointDescriptorGPU& tailJoint = mechanics.joints[2u];
        tailJoint.parentBody = 0u;
        tailJoint.childBody = 3u;
        tailJoint.jointType = MR_JOINT_REVOLUTE;
        tailJoint.qOffset = 9u;
        tailJoint.nq = 1u;
        tailJoint.vOffset = 8u;
        tailJoint.nv = 1u;
        tailJoint.axis0 = {0.0f, 1.0f, 0.0f, 0.0f};
        tailJoint.parentAnchor = {-0.0305f, 0.0f, 0.0180f, 0.0f};
        tailJoint.childAnchor = {0.0903f, 0.0095f, 0.02464f, 0.0f};
        tailJoint.parentRotation = {0.0f, 0.0f, 0.0f, 1.0f};
        tailJoint.childRotation = {0.0f, 0.0f, 0.0f, 1.0f};
        const auto configureLegJoint = [&](const std::uint32_t jointIndex,
                                           const std::uint32_t parentBody,
                                           const std::uint32_t childBody,
                                           const mr_float4 parentAnchor,
                                           const mr_float4 childAnchor) {
            MRJointDescriptorGPU& joint = mechanics.joints[jointIndex];
            joint.parentBody = parentBody;
            joint.childBody = childBody;
            joint.jointType = MR_JOINT_REVOLUTE;
            joint.qOffset = 10u + (jointIndex - 3u);
            joint.nq = 1u;
            joint.vOffset = 9u + (jointIndex - 3u);
            joint.nv = 1u;
            joint.axis0 = {0.0f, 1.0f, 0.0f, 0.0f};
            joint.parentAnchor = parentAnchor;
            joint.childAnchor = childAnchor;
            joint.parentRotation = {0.0f, 0.0f, 0.0f, 1.0f};
            joint.childRotation = {0.0f, 0.0f, 0.0f, 1.0f};
        };
        for (std::uint32_t side = 0u; side < 2u; ++side) {
            const float lateral = side == 0u ? 0.032f : -0.032f;
            const std::uint32_t jointBase = side == 0u ? 3u : 6u;
            const std::uint32_t bodyBase = side == 0u ? 4u : 7u;
            configureLegJoint(jointBase, 0u, bodyBase,
                {0.025f, lateral, -0.038f, 0.0f},
                {0.0f, 0.0f, 0.0275f, 0.0f});
            configureLegJoint(jointBase + 1u, bodyBase, bodyBase + 1u,
                {0.0f, 0.0f, -0.0275f, 0.0f},
                {0.0f, 0.0f, 0.0275f, 0.0f});
            configureLegJoint(jointBase + 2u, bodyBase + 1u, bodyBase + 2u,
                {0.0f, 0.0f, -0.0275f, 0.0f},
                {-0.025f, 0.0f, 0.008f, 0.0f});
        }
        mechanics.dofs.resize(15u);
        const auto configureDof = [&](const std::uint32_t index) {
            MRDofPropertiesGPU& dof = mechanics.dofs[6u + index];
            dof.articulationIndex = 0u;
            dof.jointIndex = index;
            dof.qIndex = 7u + index;
            dof.vIndex = 6u + index;
            dof.localDof = 0u;
            dof.flags = MR_DOF_FLAG_ACTUATED | MR_DOF_FLAG_POSITION_LIMIT |
                MR_DOF_FLAG_VELOCITY_LIMIT | MR_DOF_FLAG_EFFORT_LIMIT |
                MR_DOF_FLAG_DRIVE;
            // Ten-hertz takeoff strokes require roughly 75 rad/s at the
            // authored 1.2 rad amplitude. The former 25 rad/s ceiling
            // clipped the wings to 0.4 rad and made physical lift impossible.
            dof.limits = {-1.35f, 1.35f, 90.0f, 120.0f};
            dof.drive = {800.0f, 8.00f, 0.00003f, 0.0f};
        };
        configureDof(0u);
        configureDof(1u);
        MRDofPropertiesGPU& tailDof = mechanics.dofs[8u];
        tailDof.articulationIndex = 0u;
        tailDof.jointIndex = 2u;
        tailDof.qIndex = 9u;
        tailDof.vIndex = 8u;
        tailDof.localDof = 0u;
        tailDof.flags = MR_DOF_FLAG_ACTUATED | MR_DOF_FLAG_POSITION_LIMIT |
            MR_DOF_FLAG_VELOCITY_LIMIT | MR_DOF_FLAG_EFFORT_LIMIT |
            MR_DOF_FLAG_DRIVE;
        tailDof.limits = {-0.55f, 0.55f, 18.0f, 0.35f};
        tailDof.drive = {12.0f, 0.18f, 0.00002f, 0.0f};
        const auto configureLegDof = [&](const std::uint32_t jointIndex,
                                         const mr_float4 limits,
                                         const mr_float4 drive) {
            const std::uint32_t offset = jointIndex - 3u;
            MRDofPropertiesGPU& dof = mechanics.dofs[9u + offset];
            dof.articulationIndex = 0u;
            dof.jointIndex = jointIndex;
            dof.qIndex = 10u + offset;
            dof.vIndex = 9u + offset;
            dof.localDof = 0u;
            dof.flags = MR_DOF_FLAG_ACTUATED | MR_DOF_FLAG_POSITION_LIMIT |
                MR_DOF_FLAG_VELOCITY_LIMIT | MR_DOF_FLAG_EFFORT_LIMIT |
                MR_DOF_FLAG_DRIVE;
            dof.limits = limits;
            dof.drive = drive;
        };
        for (const std::uint32_t joint : {3u, 6u}) {
            configureLegDof(joint, {-0.85f, 0.85f, 18.0f, 0.75f},
                {55.0f, 0.45f, 0.00001f, 0.0f});
            configureLegDof(joint + 1u, {-0.10f, 1.55f, 22.0f, 0.75f},
                {65.0f, 0.50f, 0.00001f, 0.0f});
            configureLegDof(joint + 2u, {-0.90f, 0.70f, 24.0f, 0.60f},
                {60.0f, 0.45f, 0.00001f, 0.0f});
        }
        mechanics.bodyNames = {
            "dove_body", "dove_left_wing", "dove_right_wing", "dove_tail",
            "dove_left_thigh", "dove_left_shank", "dove_left_foot",
            "dove_right_thigh", "dove_right_shank", "dove_right_foot"};
        mechanics.jointNames = {
            "dove_left_wing_flap", "dove_right_wing_flap", "dove_tail_pitch",
            "dove_left_hip_pitch", "dove_left_knee_pitch",
            "dove_left_ankle_pitch", "dove_right_hip_pitch",
            "dove_right_knee_pitch", "dove_right_ankle_pitch"};
        mechanics.dofNames = {
            "root_x", "root_y", "root_z", "root_rx", "root_ry", "root_rz",
            "dove_left_wing_flap", "dove_right_wing_flap",
            "dove_tail_pitch",
            "dove_left_hip_pitch", "dove_left_knee_pitch",
            "dove_left_ankle_pitch", "dove_right_hip_pitch",
            "dove_right_knee_pitch", "dove_right_ankle_pitch",
        };
        mechanics.defaultQ = {
            0.0f, 0.0f, 0.163f, 0.0f, 0.0f, 0.0f, 1.0f,
            0.35f, 0.35f, 0.0f,
            -0.12f, 0.28f, -0.16f,
            -0.12f, 0.28f, -0.16f,
        };
        mechanics.defaultV.assign(15u, 0.0f);
        mechanics.shapes.clear();
        const auto appendBox = [&](const std::uint32_t bodyIndex,
                                   const mr_float4 position,
                                   const mr_float4 halfExtents) {
            MRShapeGPU shape{};
            shape.bodyIndex = bodyIndex;
            shape.shapeType = MR_SHAPE_BOX;
            shape.materialIndex = 0u;
            shape.collisionGroup = 1u;
            shape.collisionMask = ~0u;
            shape.slotGeneration = static_cast<std::uint32_t>(
                mechanics.shapes.size() + 1u
            );
            shape.localPosition = position;
            shape.localPosition.w = 1.0f;
            shape.localRotation = {0.0f, 0.0f, 0.0f, 1.0f};
            shape.dimensions = halfExtents;
            shape.contactRestAndBoundingRadius = {
                0.001f, 0.0f,
                std::sqrt(halfExtents.x * halfExtents.x +
                          halfExtents.y * halfExtents.y +
                          halfExtents.z * halfExtents.z),
                0.0f,
            };
            mechanics.shapes.push_back(shape);
        };
        appendBox(0u, {0.0f, 0.0f, 0.0f, 1.0f}, {0.16f, 0.05f, 0.055f, 0.0f});
        // Collision proxies cover distal lifting surfaces only. Feathered
        // visual roots overlap the torso at their anatomical seams, but
        // collision boxes must not manufacture self-penetration there.
        appendBox(1u, {0.0f, 0.040f, 0.0f, 1.0f},
            {0.0805f, 0.045f, 0.012f, 0.0f});
        appendBox(2u, {0.0f, -0.040f, 0.0f, 1.0f},
            {0.0805f, 0.045f, 0.012f, 0.0f});
        appendBox(3u, {-0.080f, 0.0f, 0.0f, 1.0f},
            {0.030f, 0.110f, 0.010f, 0.0f});
        for (const std::uint32_t bodyIndex : {4u, 5u, 7u, 8u}) {
            appendBox(bodyIndex, {0.0f, 0.0f, 0.0f, 1.0f},
                {0.011f, 0.009f, 0.0275f, 0.0f});
        }
        appendBox(6u, {0.0f, 0.0f, 0.0f, 1.0f},
            {0.045f, 0.015f, 0.008f, 0.0f});
        appendBox(9u, {0.0f, 0.0f, 0.0f, 1.0f},
            {0.045f, 0.015f, 0.008f, 0.0f});
        mechanics.world.shapeCount = static_cast<std::uint32_t>(
            mechanics.shapes.size()
        );
        mechanics.shapeNames = {
            "dove_body/collision_body", "dove_left_wing/collision",
            "dove_right_wing/collision", "dove_tail/collision",
            "dove_left_thigh/collision", "dove_left_shank/collision",
            "dove_right_thigh/collision", "dove_right_shank/collision",
            "dove_left_foot/sole", "dove_right_foot/sole",
        };
        RobotPack pack = genericRobot(
            "birdflow_deetjen_dove_hybrid", std::move(mechanics),
            {"articulated_flight", "articulated_legs", "foot_contact",
             "ground_locomotion", "load_responsive_aero", "trainable_policy"}
        );
        pack.sourceRepository = "BirdFlowMetal Deetjen surface benchmark";
        pack.sourceRevision = "deetjen-ob-2018-12-11-f03-complete-surface-v1";
        pack.license = "hybrid-modelled-properties";
        addBodyRole(pack, "airframe", {"dove_body"});
        addBodyRole(pack, "left_wing", {"dove_left_wing"});
        addBodyRole(pack, "right_wing", {"dove_right_wing"});
        addBodyRole(pack, "tail", {"dove_tail"});
        addBodyRole(pack, "left_leg",
            {"dove_left_thigh", "dove_left_shank", "dove_left_foot"});
        addBodyRole(pack, "right_leg",
            {"dove_right_thigh", "dove_right_shank", "dove_right_foot"});
        addBodyRole(pack, "left_foot", {"dove_left_foot"});
        addBodyRole(pack, "right_foot", {"dove_right_foot"});
        pack.actuators = {
            {.id = "wing.left_flap", .kind = RobotActuatorKind::flappingPosition,
             .target = "dove_left_wing_flap", .scale = 1.20f,
             .responseTimeSeconds = 0.012f,
             // Differential amplitude is the bird's steering authority.
             // Preserve positive stroke amplitude at both extremes while
             // allowing decisive banking instead of a trim-only wiggle.
             .parameters = {0.50f, 0.50f, 0.0f, 0.0f}},
            {.id = "wing.right_flap", .kind = RobotActuatorKind::flappingPosition,
             .target = "dove_right_wing_flap", .scale = 1.20f,
             .responseTimeSeconds = 0.012f,
             .parameters = {0.50f, 0.50f, 0.0f, 0.0f}},
            {.id = "tail.pitch", .kind = RobotActuatorKind::jointPosition,
             .target = "dove_tail_pitch", .scale = 0.45f,
             .responseTimeSeconds = 0.020f},
            {.id = "leg.left_hip", .kind = RobotActuatorKind::jointPosition,
             .target = "dove_left_hip_pitch", .scale = 0.70f,
             .responseTimeSeconds = 0.025f},
            {.id = "leg.left_knee", .kind = RobotActuatorKind::jointPosition,
             .target = "dove_left_knee_pitch", .scale = 0.75f,
             .responseTimeSeconds = 0.020f},
            {.id = "leg.left_ankle", .kind = RobotActuatorKind::jointPosition,
             .target = "dove_left_ankle_pitch", .scale = 0.65f,
             .responseTimeSeconds = 0.015f},
            {.id = "leg.right_hip", .kind = RobotActuatorKind::jointPosition,
             .target = "dove_right_hip_pitch", .scale = 0.70f,
             .responseTimeSeconds = 0.025f},
            {.id = "leg.right_knee", .kind = RobotActuatorKind::jointPosition,
             .target = "dove_right_knee_pitch", .scale = 0.75f,
             .responseTimeSeconds = 0.020f},
            {.id = "leg.right_ankle", .kind = RobotActuatorKind::jointPosition,
             .target = "dove_right_ankle_pitch", .scale = 0.65f,
             .responseTimeSeconds = 0.015f},
            {.id = "tail.yaw_moment", .kind = RobotActuatorKind::bodyWrench,
             .target = "dove_body", .scale = 0.020f,
             .responseTimeSeconds = 0.030f, .component = 5u},
        };
        FlappingWingActuatorPack aerodynamic{};
        aerodynamic.bodyRole = "airframe";
        aerodynamic.wingRoles = {"left_wing", "right_wing"};
        aerodynamic.tailRole = "tail";
        aerodynamic.windVelocityAndDensity = {0.0f, 0.0f, 0.0f, 1.225f};
        aerodynamic.wings[0].rootToCenterAndArea = {
            -0.010f, 0.120f, 0.0061436f, 0.0720f
        };
        aerodynamic.wings[0].hingeAxisAndChord = {1.0f, 0.0f, 0.0f, 0.15f};
        aerodynamic.wings[1].rootToCenterAndArea = {
            -0.010f, -0.120f, 0.0061436f, 0.0720f
        };
        aerodynamic.wings[1].hingeAxisAndChord = {-1.0f, 0.0f, 0.0f, 0.15f};
        aerodynamic.tail.rootToCenterAndArea = {
            -0.1208f, -0.0095f, -0.00664f, 0.0361159f
        };
        aerodynamic.tail.chordAndCoefficients = {0.17f, 2.5f, 0.08f, 3.00f};
        aerodynamic.fuselage.referenceAreasAndDrag = {
            0.028f, 0.035f, 0.032f, 2.00f
        };
        aerodynamic.fuselage.angularDamping = {0.020f, 0.022f, 0.010f, 0.0f};
        for (MRFlappingWingGPU& wing : aerodynamic.wings) {
            // Explicitly authored hybrid closure: the cap includes the
            // passive-feathering stroke term used by the device kernel.
            wing.coefficients = {1.5f, 0.04f, 0.04f, 1.2f};
            // Rotational stroke lift and the authored passive-feathering
            // closure remains load-coupled and vertical. Forward flight must
            // emerge from resolved attitude, blade lift, and drag rather than
            // an always-positive trajectory bias.
            wing.unsteadyCoefficients = {8.0f, 0.50f, 0.0f, 0.0f};
        }
        pack.flappingWings = aerodynamic;
        return pack;
    }
    if (id == "birdflow_american_crow_estimated_hybrid") {
        // BirdFlow's crow asset is explicitly an estimated hybrid: it locks
        // published morphometric ranges and visual/physics anchors, but not a
        // same-specimen crow surface, inertia, or kinematic trial.  Start
        // from the already-qualified articulated standing-to-flight topology,
        // then author a distinct crow-sized mechanics and aero contract.
        auto dove = builtinRobotPack("birdflow_deetjen_dove_hybrid");
        if (!dove) {
            throw std::logic_error("bundled BirdFlow dove RobotPack is unavailable");
        }
        RobotPack pack = std::move(*dove);
        const auto crowIdentifier = [](std::string value) {
            std::size_t offset = 0u;
            while ((offset = value.find("dove", offset)) != std::string::npos) {
                value.replace(offset, 4u, "crow");
                offset += 4u;
            }
            return value;
        };
        pack.id = "birdflow_american_crow_estimated_hybrid";
        pack.revision = 5u;
        pack.sourceRepository =
            "BirdFlowMetal American-crow estimated hybrid visual model";
        pack.sourceRevision =
            "american-crow-hybrid-visual-v1"
            "@563e600ff8da2fb7461b00228d421e05c1826d1fe025840e319e0aef4e719714";
        pack.license = "estimated-hybrid-modelled-properties";
        pack.capabilities.push_back("standing_to_flight");
        pack.capabilities.push_back("estimated_crow_model");
        pack.mechanics.name = crowIdentifier(std::move(pack.mechanics.name));
        for (std::string& name : pack.mechanics.bodyNames) {
            name = crowIdentifier(std::move(name));
        }
        for (std::string& name : pack.mechanics.jointNames) {
            name = crowIdentifier(std::move(name));
        }
        for (std::string& name : pack.mechanics.dofNames) {
            name = crowIdentifier(std::move(name));
        }
        for (std::string& name : pack.mechanics.shapeNames) {
            name = crowIdentifier(std::move(name));
        }
        for (RobotSemanticRole& role : pack.roles) {
            for (std::string& member : role.members) {
                member = crowIdentifier(std::move(member));
            }
        }
        for (RobotActuatorSpec& actuator : pack.actuators) {
            actuator.id = crowIdentifier(std::move(actuator.id));
            actuator.target = crowIdentifier(std::move(actuator.target));
        }

        constexpr float massScale = 0.45f / 0.32f;
        constexpr float lengthScale = 1.15f;
        constexpr float wingSpanScale = 0.91f / 0.48f;
        constexpr float inertiaScale = massScale * lengthScale * lengthScale;
        for (MRBodyPropertiesGPU& body : pack.mechanics.bodies) {
            body.massAndInverseMass.x *= massScale;
            body.massAndInverseMass.y = 1.0f / body.massAndInverseMass.x;
            body.inertiaRow0.x *= inertiaScale;
            body.inertiaRow1.y *= inertiaScale;
            body.inertiaRow2.z *= inertiaScale;
            body.inverseInertiaRow0.x = 1.0f / body.inertiaRow0.x;
            body.inverseInertiaRow1.y = 1.0f / body.inertiaRow1.y;
            body.inverseInertiaRow2.z = 1.0f / body.inertiaRow2.z;
        }
        for (std::size_t index = 0u;
             index < pack.mechanics.shapes.size(); ++index) {
            MRShapeGPU& shape = pack.mechanics.shapes[index];
            const float spanScale = index == 1u || index == 2u
                ? wingSpanScale
                : lengthScale;
            shape.localPosition.x *= lengthScale;
            shape.localPosition.y *= spanScale;
            shape.localPosition.z *= lengthScale;
            shape.dimensions.x *= lengthScale;
            shape.dimensions.y *= spanScale;
            shape.dimensions.z *= lengthScale;
            shape.contactRestAndBoundingRadius.z = std::sqrt(
                shape.dimensions.x * shape.dimensions.x +
                shape.dimensions.y * shape.dimensions.y +
                shape.dimensions.z * shape.dimensions.z
            );
        }
        for (std::size_t index = 0u;
             index < pack.mechanics.joints.size(); ++index) {
            MRJointDescriptorGPU& joint = pack.mechanics.joints[index];
            const float spanScale = index < 2u ? wingSpanScale : lengthScale;
            joint.parentAnchor.x *= lengthScale;
            joint.parentAnchor.y *= spanScale;
            joint.parentAnchor.z *= lengthScale;
            joint.childAnchor.x *= lengthScale;
            joint.childAnchor.y *= spanScale;
            joint.childAnchor.z *= lengthScale;
        }
        for (const std::uint32_t hipJoint : {3u, 6u}) {
            // Calibrated hybrid stance offset: the scaled dove hip anchor
            // is moved 30 mm toward the tail for the heavier crow body.
            // This remains an explicit model closure, not measured crow
            // anatomy, and is accepted only if the native support probe
            // improves over the unshifted configuration.
            pack.mechanics.joints[hipJoint].parentAnchor.x -= 0.030f;
        }
        pack.mechanics.defaultQ[2u] *= lengthScale;

        FlappingWingActuatorPack& aerodynamic = *pack.flappingWings;
        for (MRFlappingWingGPU& wing : aerodynamic.wings) {
            wing.rootToCenterAndArea.x *= lengthScale;
            wing.rootToCenterAndArea.y *= wingSpanScale;
            wing.rootToCenterAndArea.z *= lengthScale;
            wing.rootToCenterAndArea.w = 0.075f;
            wing.hingeAxisAndChord.w = 0.160f;
            // This is an explicit hybrid closure, not a crow force, power,
            // or CFD measurement.  With the old 5.0/0.0 pair, even +0.300
            // residual wing action failed to sustain a body-up liftoff. Test
            // the dove's 8.0 stroke strength without its inherited forward
            // bias: resolved attitude, blade lift, drag, and live tail trim
            // must supply all forward motion.  It remains an estimated-model
            // bracket until independently validated flight evidence exists.
            wing.unsteadyCoefficients.x = 8.0f;
            wing.unsteadyCoefficients.y = 0.0f;
        }
        aerodynamic.tail.rootToCenterAndArea.x *= lengthScale;
        aerodynamic.tail.rootToCenterAndArea.y *= lengthScale;
        aerodynamic.tail.rootToCenterAndArea.z *= lengthScale;
        aerodynamic.tail.rootToCenterAndArea.w = 0.041f;
        aerodynamic.tail.chordAndCoefficients.x = 0.174f;
        aerodynamic.fuselage.referenceAreasAndDrag.x *=
            lengthScale * lengthScale;
        aerodynamic.fuselage.referenceAreasAndDrag.y *=
            lengthScale * lengthScale;
        aerodynamic.fuselage.referenceAreasAndDrag.z *=
            lengthScale * lengthScale;
        return pack;
    }
    if (id == "px4_x500") {
        EngineModel mechanics = makeFreeSphereEngineModel();
        mechanics.name = "px4_x500";
        mechanics.world.gravityAndTimestep = {
            0.0f, 0.0f, -9.81f, 1.0f / 240.0f,
        };
        mechanics.articulations[0u].rootBody = 0u;
        mechanics.articulations[0u].firstBody = 0u;
        mechanics.bodies = {makePX4X500BodyProperties()};
        mechanics.bodies[0u].articulationIndex = 0u;
        mechanics.bodyNames = {"x500_base"};
        mechanics.world.bodyCount = 1u;
        mechanics.dofNames = {
            "root_x", "root_y", "root_z",
            "root_rx", "root_ry", "root_rz",
        };
        mechanics.defaultQ = {
            0.0f, 0.0f, 2.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
        };
        mechanics.shapes.clear();
        const auto appendBox = [&](const mr_float4 position,
                                   const mr_float4 rotation,
                                   const mr_float4 halfExtents) {
            MRShapeGPU shape{};
            shape.bodyIndex = 0u;
            shape.shapeType = MR_SHAPE_BOX;
            shape.materialIndex = 0u;
            shape.collisionGroup = 1u;
            shape.collisionMask = ~0u;
            shape.slotGeneration = static_cast<std::uint32_t>(
                mechanics.shapes.size() + 1u
            );
            shape.localPosition = position;
            shape.localPosition.w = 1.0f;
            shape.localRotation = rotation;
            shape.dimensions = halfExtents;
            shape.contactRestAndBoundingRadius = {
                0.001f, 0.0f,
                std::sqrt(
                    halfExtents.x * halfExtents.x +
                    halfExtents.y * halfExtents.y +
                    halfExtents.z * halfExtents.z
                ),
                0.0f,
            };
            mechanics.shapes.push_back(shape);
        };
        const mr_float4 identity{0.0f, 0.0f, 0.0f, 1.0f};
        appendBox({0.0f, 0.0f, 0.007f, 1.0f}, identity,
            {0.1767766953f, 0.1767766953f, 0.025f, 0.0f});
        const float halfRoll = 0.175f;
        appendBox({0.0f, -0.098f, -0.123f, 1.0f},
            {-std::sin(halfRoll), 0.0f, 0.0f, std::cos(halfRoll)},
            {0.0075f, 0.0075f, 0.105f, 0.0f});
        appendBox({0.0f, 0.098f, -0.123f, 1.0f},
            {std::sin(halfRoll), 0.0f, 0.0f, std::cos(halfRoll)},
            {0.0075f, 0.0075f, 0.105f, 0.0f});
        appendBox({0.0f, -0.132f, -0.2195f, 1.0f}, identity,
            {0.125f, 0.0075f, 0.0075f, 0.0f});
        appendBox({0.0f, 0.132f, -0.2195f, 1.0f}, identity,
            {0.125f, 0.0075f, 0.0075f, 0.0f});
        constexpr std::array<mr_float4, 4u> rotorPositions{{
            {0.174f, -0.174f, 0.06f, 1.0f},
            {-0.174f, 0.174f, 0.06f, 1.0f},
            {0.174f, 0.174f, 0.06f, 1.0f},
            {-0.174f, -0.174f, 0.06f, 1.0f},
        }};
        for (const mr_float4 position : rotorPositions) {
            appendBox(position, identity,
                {0.1396153846f, 0.0084615385f,
                 0.0004230769f, 0.0f});
        }
        mechanics.world.shapeCount = static_cast<std::uint32_t>(
            mechanics.shapes.size()
        );
        mechanics.shapeNames.clear();
        for (std::size_t index = 0u; index < mechanics.shapes.size();
             ++index) {
            mechanics.shapeNames.push_back(
                "x500_base/collision_" + std::to_string(index)
            );
        }
        RobotPack pack = genericRobot(
            "px4_x500",
            std::move(mechanics),
            {"flight", "hover"}
        );
        pack.sourceRepository =
            "https://github.com/PX4/PX4-gazebo-models.git";
        pack.sourceRevision =
            "e00d3b9cde682dbcb3bf6f30a2f2b8ef4325dae8";
        pack.license = "BSD-3-Clause";
        addBodyRole(pack, "airframe", {"x500_base"});
        const auto model = makePX4X500MulticopterModel(1.0f / 240.0f);
        const float hover = std::sqrt(
            pack.mechanics.bodies[0u].massAndInverseMass.x * 9.81f /
            (4.0f * model.coefficients.x)
        );
        pack.actuators.clear();
        constexpr std::array<std::string_view, 4u> lanes{
            "collective", "roll", "pitch", "yaw",
        };
        for (std::uint32_t component = 0u; component < lanes.size();
             ++component) {
            pack.actuators.push_back({
                .id = "rotor_mixer." + std::string{lanes[component]},
                .kind = RobotActuatorKind::rotorMixer,
                .target = "x500_base",
                .scale = 1.0f,
                .component = component,
            });
        }
        pack.multicopter = MulticopterActuatorPack{
            .model = model,
            .rotors = makePX4X500Rotors(),
            .mixer = {{hover, 120.0f, 35.0f, 12.0f}},
            .bodyRole = "airframe",
        };
        return pack;
    }
    if (id == "unitree_g1") {
        RobotPack pack = genericRobot(
            "unitree_g1",
            makeUnitreeG1EngineModel(),
            {"balance", "locomotion", "whole_body_motion", "upper_body_motion"}
        );
        const G1ModelMetadata& metadata = unitreeG1Metadata();
        pack.sourceRepository = std::string(metadata.sourceRepository);
        pack.sourceRevision = std::string(metadata.sourceCommit);
        pack.license = std::string(metadata.sourceLicense);
        const std::span<const float> scales =
            unitreeG1LocomotionActionScales();
        for (std::size_t index = 0u;
             index < pack.actuators.size() && index < scales.size();
             ++index) {
            pack.actuators[index].scale = scales[index];
        }
        addBodyRole(pack, "left_foot", {"left_ankle_roll_link"});
        addBodyRole(pack, "right_foot", {"right_ankle_roll_link"});
        addBodyRole(pack, "pelvis", {"pelvis"});
        addBodyRole(pack, "left_wrist", {"left_wrist_yaw_link"});
        addBodyRole(pack, "right_wrist", {"right_wrist_yaw_link"});
        return pack;
    }
    if (id == "franka_panda") {
        RobotPack pack = genericRobot(
            "franka_panda",
            makeFrankaPandaHandEngineModel(),
            {"manipulation", "grasping", "force_control"}
        );
        for (RobotActuatorSpec& actuator : pack.actuators) {
            actuator.scale = actuator.target.find("finger") !=
                    std::string::npos
                ? 0.01f
                : 0.25f;
            actuator.responseTimeSeconds = 0.04f;
            if (actuator.target.find("finger") != std::string::npos) {
                actuator.kind = RobotActuatorKind::gripperPosition;
            }
        }
        addBodyRole(pack, "base", {"panda_link0"});
        addBodyRole(pack, "gripper", {
            "panda_hand", "panda_leftfinger", "panda_rightfinger"
        });
        addBodyRole(pack, "left_finger", {"panda_leftfinger"});
        addBodyRole(pack, "right_finger", {"panda_rightfinger"});
        return pack;
    }
    if (id == "dvrk_psm") {
        RobotPack pack = genericRobot(
            "dvrk_psm",
            makeDvrkPsmLargeNeedleDriverEngineModel(),
            {"surgical_research", "grasping", "remote_center_motion"}
        );
        const SurgicalPSMModelMetadata& metadata = surgicalPSMMetadata();
        pack.sourceRepository = std::string(metadata.orbitRepository);
        pack.sourceRevision = std::string(metadata.orbitCommit);
        pack.license = std::string(metadata.orbitLicense);
        return pack;
    }
    return std::nullopt;
}

const char* runCompileStatusName(const RunCompileStatus status) noexcept {
    switch (status) {
    case RunCompileStatus::success: return "success";
    case RunCompileStatus::invalidManifest: return "invalid_manifest";
    case RunCompileStatus::invalidRobot: return "invalid_robot";
    case RunCompileStatus::unresolvedRole: return "unresolved_role";
    case RunCompileStatus::compositionFailure: return "composition_failure";
    case RunCompileStatus::worldFailure: return "world_failure";
    case RunCompileStatus::taskFailure: return "task_failure";
    case RunCompileStatus::policyFailure: return "policy_failure";
    case RunCompileStatus::internalFailure: return "internal_failure";
    }
    return "unknown";
}

} // namespace metalrobo
