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
        case RobotActuatorKind::millardExcitation:
            if (!actuator.terms.empty() || actuator.component != 0u ||
                actuator.parameters.x != 0.0f ||
                actuator.parameters.y != 0.0f ||
                actuator.parameters.z != 0.0f ||
                actuator.parameters.w != 0.0f ||
                actuator.scale != 1.0f) {
                reason = "Millard excitation actuator '" + actuator.id +
                    "' requires unit scale and no joint/vector parameters";
                return false;
            }
            // The source muscle target is intentionally resolved only when
            // MetalWorld has the immutable Millard source program in scope.
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
    return {"unitree_g1", "franka_panda", "dvrk_psm", "px4_x500"};
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

std::optional<RobotPack> builtinRobotPack(const std::string_view id) {
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
