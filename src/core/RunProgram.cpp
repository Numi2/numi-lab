#include "metalrobo/RunProgram.hpp"

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

bool taskOwnsSensorExecution(const TaskPack& task) {
    return !task.actorFrame.empty() || !task.actorCurrent.empty() ||
        !task.critic.empty() || task.actorHistoryLength != 1u ||
        task.criticHistoryLength != 1u ||
        !task.criticIncludesCleanHistory || task.visual.width != 0u ||
        task.visual.height != 0u || !task.visual.frameOffsets.empty();
}

bool taskOwnsRealityExecution(const TaskPack& task) {
    return !task.randomization.empty() ||
        task.maximumActionDelaySteps != 0u ||
        task.maximumObservationDelaySteps != 0u;
}

std::optional<std::string> lowerRealityProgram(
    const WorldTemplate& world,
    const RunProfile& profile,
    RealityPack const& reality,
    TaskPack& executable
) {
    executable.randomization = reality.taskState;
    executable.maximumActionDelaySteps =
        reality.maximumActionDelaySteps;
    executable.maximumObservationDelaySteps =
        reality.maximumObservationDelaySteps;
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
            executable.randomization.push_back({
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
                executable.randomization.push_back({
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
            executable.randomization.push_back({
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
            executable.randomization.push_back({
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
    pack.roles.push_back({
        .id = "all_joints",
        .kind = RobotSemanticKind::joint,
        .members = pack.mechanics.jointNames,
    });
    pack.roles.push_back({
        .id = "all_dofs",
        .kind = RobotSemanticKind::dof,
        .members = pack.mechanics.dofNames,
    });
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
        if (taskOwnsSensorExecution(manifest.task)) {
            return reject(
                RunCompileStatus::invalidManifest,
                manifest.task.id,
                "TaskPack contains observation execution; move it to SensorPack"
            );
        }
        if (taskOwnsRealityExecution(manifest.task)) {
            return reject(
                RunCompileStatus::invalidManifest,
                manifest.task.id,
                "TaskPack contains reset variation execution; move it to RealityPack"
            );
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
            !validRoles(manifest.robot, reason)) {
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
            bodyOffset += static_cast<std::uint32_t>(object.mechanics.bodies.size());
            shapeOffset += static_cast<std::uint32_t>(object.mechanics.shapes.size());
            materialOffset += static_cast<std::uint32_t>(object.mechanics.materials.size());
            articulationOffset += static_cast<std::uint32_t>(object.mechanics.articulations.size());
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

        // Compile the independently authored runtime programs into one fused
        // native transaction. This is a compile-time lowering only: Metal
        // executes the resolved sensor, reality and teacher tables directly,
        // with no runtime adapter or duplicated host logic.
        TaskPack executableTask = manifest.task;
        executableTask.actorFrame = manifest.sensors.actorFrame;
        executableTask.actorHistoryLength =
            manifest.sensors.actorHistoryLength;
        executableTask.actorCurrent = manifest.sensors.actorCurrent;
        executableTask.critic = manifest.sensors.critic;
        executableTask.criticHistoryLength =
            manifest.sensors.criticHistoryLength;
        executableTask.criticIncludesCleanHistory =
            manifest.sensors.criticIncludesCleanHistory;
        executableTask.visual = manifest.sensors.visual;
        if (executableTask.actorFrame.empty() ||
            executableTask.critic.empty()) {
            return reject(
                RunCompileStatus::invalidManifest,
                manifest.sensors.id,
                "SensorPack must author executable actor and critic observations"
            );
        }
        if (const auto realityError = lowerRealityProgram(
                worldTemplate,
                manifest.profile,
                manifest.reality,
                executableTask
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
                  executableTask,
                  *manifest.teacher.interactions,
                  manifest.teacher.interactionClip,
                  staged.world_,
                  staged.task_
              )
            : compileTaskProgram(
                  executableTask,
                  staged.world_,
                  staged.task_
              );
        if (!taskStatus.succeeded()) {
            return reject(RunCompileStatus::taskFailure, taskStatus.element,
                taskStatus.message);
        }
        if (manifest.policy) {
            PolicyPack bound = *manifest.policy;
            if (bound.contract.version == 0u) {
                bindPolicyPack(bound, staged.task_);
            }
            const PolicyCompileDiagnostics policyStatus =
                compilePolicyProgram(bound, staged.task_, staged.policy_);
            if (!policyStatus.succeeded()) {
                return reject(RunCompileStatus::policyFailure,
                    policyStatus.element, policyStatus.message);
            }
            staged.boundPolicy_ = std::move(bound);
        }
        staged.worldFamily_ = std::move(family);
        staged.model_ = std::move(model);
        staged.defaultSceneBodies_ = std::move(defaultSceneBodies);
        staged.profile_ = manifest.profile;
        staged.profile_.physics = compose;
        staged.teacher_ = manifest.teacher;
        staged.robotFingerprint_ = engineModelFingerprint(
            manifest.robot.mechanics
        );
        {
            Hash sensorHash;
            sensorHash.scalar(sensorFingerprint(episode.sensors));
            sensorHash.scalar(staged.task_.observationFingerprint());
            staged.sensorFingerprint_ = sensorHash.finish();
        }
        {
            Hash realityHash;
            realityHash.scalar(staged.worldFamily_.program.fingerprint);
            realityHash.scalar<std::uint64_t>(
                staged.task_.randomizationOperators().size()
            );
            if (!staged.task_.randomizationOperators().empty()) {
                realityHash.bytes(
                    staged.task_.randomizationOperators().data(),
                    staged.task_.randomizationOperators().size_bytes()
                );
            }
            realityHash.scalar(executableTask.maximumActionDelaySteps);
            realityHash.scalar(
                executableTask.maximumObservationDelaySteps
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
    return {"unitree_g1", "franka_panda", "dvrk_psm"};
}

std::optional<RobotPack> builtinRobotPack(const std::string_view id) {
    if (id == "unitree_g1") {
        RobotPack pack = genericRobot(
            "unitree_g1",
            makeUnitreeG1EngineModel(),
            {"balance", "locomotion", "whole_body_motion", "manipulation"}
        );
        const G1ModelMetadata& metadata = unitreeG1Metadata();
        pack.sourceRepository = std::string(metadata.sourceRepository);
        pack.sourceRevision = std::string(metadata.sourceCommit);
        pack.license = std::string(metadata.sourceLicense);
        addBodyRole(pack, "left_foot", {"left_ankle_roll_link"});
        addBodyRole(pack, "right_foot", {"right_ankle_roll_link"});
        addBodyRole(pack, "pelvis", {"pelvis"});
        addBodyRole(pack, "left_hand", {"left_wrist_yaw_link"});
        addBodyRole(pack, "right_hand", {"right_wrist_yaw_link"});
        return pack;
    }
    if (id == "franka_panda") {
        RobotPack pack = genericRobot(
            "franka_panda",
            makeFrankaPandaHandEngineModel(),
            {"manipulation", "grasping", "force_control"}
        );
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
