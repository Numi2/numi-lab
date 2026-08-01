#include "metalrobo/SensorProgram.hpp"

#include "metalrobo/MetalWorld.hpp"

#include "SemanticTransform.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace metalrobo {

struct CompiledSensorProgram::Storage {
    std::uint64_t fingerprint = 0u;
    std::uint64_t worldFingerprint = 0u;
    SensorProgramLayout layout{};
    MRSensorProgramHeaderGPU header{};
    std::vector<std::string> sensorIds;
    std::vector<MRSensorDescriptorGPU> descriptors;
    std::vector<std::uint32_t> filterBodies;
    CookedTactileSystem tactile;
};

namespace {

constexpr std::uint64_t kFNVOffset = 1469598103934665603ull;
constexpr std::uint64_t kFNVPrime = 1099511628211ull;
constexpr std::uint64_t kNanosecondsPerSecond = 1'000'000'000ull;

class Hash {
public:
    void bytes(const void* data, const std::size_t size) {
        const auto* values = static_cast<const unsigned char*>(data);
        for (std::size_t index = 0u; index < size; ++index) {
            value_ ^= values[index];
            value_ *= kFNVPrime;
        }
    }

    template <typename Type>
    void scalar(const Type& value) {
        bytes(&value, sizeof(value));
    }

    void string(const std::string_view value) {
        scalar<std::uint64_t>(value.size());
        bytes(value.data(), value.size());
    }

    template <typename Type>
    void span(const std::span<const Type> values) {
        scalar<std::uint64_t>(values.size());
        if (!values.empty()) {
            bytes(values.data(), values.size_bytes());
        }
    }

    [[nodiscard]] std::uint64_t finish() const noexcept {
        return value_ == 0u ? 1u : value_;
    }

private:
    std::uint64_t value_ = kFNVOffset;
};

SensorCompileDiagnostics reject(
    const SensorCompileStatus status,
    std::string element,
    std::string message
) {
    return {
        .status = status,
        .element = std::move(element),
        .message = std::move(message),
    };
}

bool finite(const float value) {
    return std::isfinite(value);
}

bool finite(const mr_float4 value) {
    return finite(value.x) && finite(value.y) &&
        finite(value.z) && finite(value.w);
}

bool unitQuaternion(const mr_float4 value) {
    if (!finite(value)) {
        return false;
    }
    const double squared =
        static_cast<double>(value.x) * value.x +
        static_cast<double>(value.y) * value.y +
        static_cast<double>(value.z) * value.z +
        static_cast<double>(value.w) * value.w;
    return std::abs(squared - 1.0) <= 1.0e-5;
}

bool checkedAdd(
    const std::uint32_t left,
    const std::uint32_t right,
    std::uint32_t& result
) {
    const std::uint64_t wide =
        static_cast<std::uint64_t>(left) + right;
    if (wide > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    result = static_cast<std::uint32_t>(wide);
    return true;
}

bool checkedMultiply(
    const std::uint32_t left,
    const std::uint32_t right,
    std::uint32_t& result
) {
    const std::uint64_t wide =
        static_cast<std::uint64_t>(left) * right;
    if (wide > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    result = static_cast<std::uint32_t>(wide);
    return true;
}

bool tactileEmpty(const CookedTactileSystem& tactile) {
    return tactile.fingerprint == 0u &&
        tactile.sensorIds.empty() && tactile.sensors.empty() &&
        tactile.samples.empty() &&
        tactile.backingShapeIndices.empty() &&
        tactile.shapeToSensor.empty() &&
        tactile.targetShapeIndices.empty();
}

std::uint64_t sensorRandomIdentity(const std::string_view id) {
    Hash hash;
    hash.string("MetalRobo.SensorIR.counter-key");
    hash.string(id);
    return hash.finish();
}

SensorExecutionDomain executionDomain(const MRWorldSensorKind kind) {
    switch (kind) {
    case MR_WORLD_SENSOR_RGB:
    case MR_WORLD_SENSOR_DEPTH:
    case MR_WORLD_SENSOR_RGBD:
    case MR_WORLD_SENSOR_SEGMENTATION:
        return SensorExecutionDomain::presentation;
    case MR_WORLD_SENSOR_TACTILE_DEPTH:
        return SensorExecutionDomain::tactile;
    case MR_WORLD_SENSOR_STATE:
    case MR_WORLD_SENSOR_FRAME_TWIST_WORLD:
    case MR_WORLD_SENSOR_FORCE_TORQUE:
    case MR_WORLD_SENSOR_IMU:
    case MR_WORLD_SENSOR_CONTACT_STATE:
    case MR_WORLD_SENSOR_JOINT_STATE:
    case MR_WORLD_SENSOR_ACTUATOR_STATE:
        return SensorExecutionDomain::nativeState;
    }
    return SensorExecutionDomain::nativeState;
}

bool imageSensor(const MRWorldSensorKind kind) {
    return executionDomain(kind) == SensorExecutionDomain::presentation;
}

std::uint32_t channelCount(const MRWorldSensorKind kind) {
    switch (kind) {
    case MR_WORLD_SENSOR_RGB:
        return 4u;
    case MR_WORLD_SENSOR_DEPTH:
    case MR_WORLD_SENSOR_SEGMENTATION:
    case MR_WORLD_SENSOR_TACTILE_DEPTH:
        return 1u;
    case MR_WORLD_SENSOR_RGBD:
        return 5u;
    case MR_WORLD_SENSOR_STATE:
        // Parent-frame xyz plus quaternion. Twist and acceleration are
        // separate explicit modalities rather than silently zero-filled
        // channels in this persisted perception-contract kind.
        return 7u;
    case MR_WORLD_SENSOR_CONTACT_STATE:
        return 5u;
    case MR_WORLD_SENSOR_JOINT_STATE:
        return 2u;
    case MR_WORLD_SENSOR_ACTUATOR_STATE:
        return 8u;
    case MR_WORLD_SENSOR_FORCE_TORQUE:
    case MR_WORLD_SENSOR_FRAME_TWIST_WORLD:
    case MR_WORLD_SENSOR_IMU:
        return 6u;
    }
    return 0u;
}

std::uint32_t tactileIndex(
    const CookedTactileSystem& tactile,
    const std::string_view id
) {
    const auto found = std::find(
        tactile.sensorIds.begin(),
        tactile.sensorIds.end(),
        id
    );
    return found == tactile.sensorIds.end()
        ? MR_INVALID_INDEX
        : static_cast<std::uint32_t>(
              found - tactile.sensorIds.begin()
          );
}

} // namespace

bool CompiledSensorProgram::valid() const noexcept {
    return storage_ != nullptr && storage_->fingerprint != 0u &&
        storage_->worldFingerprint != 0u &&
        storage_->header.sensorFingerprint == storage_->fingerprint &&
        storage_->header.worldFingerprint ==
            storage_->worldFingerprint &&
        storage_->header.reserved.x ==
            MR_SENSOR_PROGRAM_ABI_VERSION &&
        storage_->descriptors.size() ==
            storage_->layout.sensorCount &&
        storage_->sensorIds.size() == storage_->layout.sensorCount &&
        storage_->filterBodies.size() ==
            storage_->layout.filterBodyCount &&
        storage_->header.reserved.y ==
            storage_->layout.filterBodyCount;
}

std::uint64_t CompiledSensorProgram::fingerprint() const noexcept {
    return valid() ? storage_->fingerprint : 0u;
}

std::uint64_t CompiledSensorProgram::worldFingerprint() const noexcept {
    return valid() ? storage_->worldFingerprint : 0u;
}

const SensorProgramLayout& CompiledSensorProgram::layout() const noexcept {
    static constexpr SensorProgramLayout empty{};
    return valid() ? storage_->layout : empty;
}

const MRSensorProgramHeaderGPU&
CompiledSensorProgram::header() const noexcept {
    static constexpr MRSensorProgramHeaderGPU empty{};
    return valid() ? storage_->header : empty;
}

std::span<const std::string>
CompiledSensorProgram::sensorIds() const noexcept {
    return valid()
        ? std::span<const std::string>{storage_->sensorIds}
        : std::span<const std::string>{};
}

std::span<const MRSensorDescriptorGPU>
CompiledSensorProgram::descriptors() const noexcept {
    return valid()
        ? std::span<const MRSensorDescriptorGPU>{storage_->descriptors}
        : std::span<const MRSensorDescriptorGPU>{};
}

std::span<const std::uint32_t>
CompiledSensorProgram::filterBodies() const noexcept {
    return valid()
        ? std::span<const std::uint32_t>{storage_->filterBodies}
        : std::span<const std::uint32_t>{};
}

const CookedTactileSystem&
CompiledSensorProgram::tactileSystem() const noexcept {
    static const CookedTactileSystem empty{};
    return valid() ? storage_->tactile : empty;
}

std::uint32_t CompiledSensorProgram::sensorIndex(
    const std::string_view id
) const noexcept {
    if (!valid()) {
        return MR_INVALID_INDEX;
    }
    const auto found = std::find(
        storage_->sensorIds.begin(),
        storage_->sensorIds.end(),
        id
    );
    return found == storage_->sensorIds.end()
        ? MR_INVALID_INDEX
        : static_cast<std::uint32_t>(
              found - storage_->sensorIds.begin()
          );
}

SensorCompileDiagnostics compileSensorProgram(
    const std::span<const SensorSpec> sensors,
    const CookedTactileSystem& tactile,
    const CompiledWorld& world,
    CompiledSensorProgram& output
) {
    if (!world.valid()) {
        return reject(
            SensorCompileStatus::invalidWorld,
            "world",
            "sensor compilation requires a valid compiled world"
        );
    }
    if (!tactileEmpty(tactile)) {
        std::string reason;
        if (!tactile.valid(world.model(), &reason)) {
            return reject(
                SensorCompileStatus::invalidTactileSystem,
                "tactile",
                reason
            );
        }
    }

    auto staged = std::make_shared<CompiledSensorProgram::Storage>();
    staged->worldFingerprint = world.fingerprint();
    staged->sensorIds.reserve(sensors.size());
    staged->descriptors.reserve(sensors.size());
    staged->tactile = tactile;
    std::unordered_set<std::string> uniqueIds;
    std::unordered_set<std::uint64_t> uniqueRandomIdentities;
    std::unordered_set<std::uint32_t> usedTactile;

    for (std::uint32_t index = 0u; index < sensors.size(); ++index) {
        SensorSpec sensor = sensors[index];
        const std::string element =
            "sensors[" + std::to_string(index) + "]";
        if (sensor.id.empty() ||
            !uniqueIds.insert(sensor.id).second) {
            return reject(
                sensor.id.empty()
                    ? SensorCompileStatus::invalidSpec
                    : SensorCompileStatus::duplicateSemantic,
                element + ".id",
                sensor.id.empty()
                    ? "sensor id is empty"
                    : "sensor id is duplicated"
            );
        }
        if (sensor.kind >= MR_WORLD_SENSOR_KIND_COUNT ||
            sensor.parentKind > MR_WORLD_SENSOR_PARENT_WORLD ||
            sensor.schedulePhase >
                MR_WORLD_SENSOR_PHASE_PRESENTATION ||
            sensor.historyLength == 0u ||
            sensor.historyLength > 4096u ||
            sensor.consumerFlags == 0u ||
            (sensor.consumerFlags &
             ~(MR_WORLD_SENSOR_CONSUMER_ACTOR |
               MR_WORLD_SENSOR_CONSUMER_CRITIC |
               MR_WORLD_SENSOR_CONSUMER_TRUTH |
               MR_WORLD_SENSOR_CONSUMER_RECORDER)) != 0u ||
            !finite(sensor.localPose.position) ||
            !unitQuaternion(sensor.localPose.orientation) ||
            !finite(sensor.nominalRateHz) ||
            !(sensor.nominalRateHz > 0.0f) ||
            !finite(sensor.latencySeconds) ||
            sensor.latencySeconds < 0.0f ||
            !finite(sensor.valueNoiseSigma) ||
            sensor.valueNoiseSigma < 0.0f ||
            !finite(sensor.biasNoiseSigma) ||
            sensor.biasNoiseSigma < 0.0f ||
            !finite(sensor.dropoutProbability) ||
            sensor.dropoutProbability < 0.0f ||
            sensor.dropoutProbability > 1.0f) {
            return reject(
                SensorCompileStatus::invalidSpec,
                element,
                "sensor schedule, pose, corruption, or permissions are invalid"
            );
        }
        if (!sensor.parentSite.empty()) {
            if (sensor.parentKind !=
                    MR_WORLD_SENSOR_PARENT_ASSET ||
                sensor.parentBodyIndex != MR_INVALID_INDEX ||
                sensor.kind == MR_WORLD_SENSOR_JOINT_STATE ||
                sensor.kind == MR_WORLD_SENSOR_ACTUATOR_STATE ||
                sensor.kind == MR_WORLD_SENSOR_TACTILE_DEPTH) {
                return reject(
                    SensorCompileStatus::invalidSpec,
                    element + ".parentSite",
                    "site-relative sensor must be asset-owned, spatial, non-tactile, and have no pre-resolved body"
                );
            }
            const auto found = std::find_if(
                world.model().sites.begin(),
                world.model().sites.end(),
                [&](const EngineSite& site) {
                    return site.id == sensor.parentSite;
                }
            );
            if (found == world.model().sites.end()) {
                return reject(
                    SensorCompileStatus::unresolvedSemantic,
                    element + ".parentSite",
                    "sensor parent site is unresolved: " +
                        sensor.parentSite
                );
            }
            mr_float4 composedPosition{};
            mr_float4 composedOrientation{};
            if (!semantic_transform::compose(
                    found->localPosition,
                    found->localOrientation,
                    sensor.localPose.position,
                    sensor.localPose.orientation,
                    composedPosition,
                    composedOrientation
                )) {
                return reject(
                    SensorCompileStatus::invalidSpec,
                    element + ".parentSite",
                    "sensor parent-site transform composition is invalid"
                );
            }
            sensor.localPose = {
                composedPosition,
                composedOrientation,
            };
            sensor.parentBodyIndex = found->bodyIndex;
            sensor.parentKind =
                world.model().bodies[found->bodyIndex]
                        .articulationIndex == MR_INVALID_INDEX
                ? MR_WORLD_SENSOR_PARENT_RIGID_BODY
                : MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK;
            sensor.parentSite.clear();
        }
        if (sensor.parentKind == MR_WORLD_SENSOR_PARENT_ASSET) {
            if (sensor.kind != MR_WORLD_SENSOR_JOINT_STATE &&
                sensor.kind != MR_WORLD_SENSOR_ACTUATOR_STATE) {
                return reject(
                    SensorCompileStatus::unresolvedSemantic,
                    element + ".parent",
                    "asset-relative spatial sensor parent was not resolved by the world compiler"
                );
            }
        }
        const bool jointStateSensor =
            sensor.kind == MR_WORLD_SENSOR_JOINT_STATE;
        const bool actuatorStateSensor =
            sensor.kind == MR_WORLD_SENSOR_ACTUATOR_STATE;
        const bool targetedStateSensor =
            jointStateSensor || actuatorStateSensor;
        if (targetedStateSensor != !sensor.target.empty()) {
            return reject(
                SensorCompileStatus::invalidSpec,
                element + ".target",
                targetedStateSensor
                    ? "joint/actuator-state sensor target is empty"
                    : "only joint/actuator-state sensors may author a target"
            );
        }
        const bool bodyParent =
            sensor.parentKind == MR_WORLD_SENSOR_PARENT_RIGID_BODY ||
            sensor.parentKind ==
                MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK;
        if (bodyParent !=
                (sensor.parentBodyIndex != MR_INVALID_INDEX) ||
            (bodyParent &&
             sensor.parentBodyIndex >= world.model().bodies.size())) {
            return reject(
                SensorCompileStatus::unresolvedSemantic,
                element + ".parent",
                "sensor body parent is unresolved or outside the compiled world"
            );
        }
        if (bodyParent) {
            const bool articulated =
                world.model().bodies[sensor.parentBodyIndex]
                    .articulationIndex != MR_INVALID_INDEX;
            if (articulated !=
                (sensor.parentKind ==
                 MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK)) {
                return reject(
                    SensorCompileStatus::invalidSpec,
                    element + ".parent",
                    "sensor parent kind disagrees with compiled body topology"
                );
            }
        }
        std::uint32_t sourceIndex = MR_INVALID_INDEX;
        std::uint32_t sourceOwner = MR_INVALID_INDEX;
        std::uint32_t semanticSource = MR_INVALID_INDEX;
        if (bodyParent) {
            if (sensor.parentKind ==
                MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK) {
                sourceIndex = sensor.parentBodyIndex;
                sourceOwner = world.model()
                    .bodies[sensor.parentBodyIndex]
                    .articulationIndex;
            } else {
                const auto found = std::find(
                    world.sceneBodyIndices().begin(),
                    world.sceneBodyIndices().end(),
                    sensor.parentBodyIndex
                );
                if (found == world.sceneBodyIndices().end()) {
                    return reject(
                        SensorCompileStatus::unresolvedSemantic,
                        element + ".parent",
                        "rigid sensor parent is not in the compiled scene-state layout"
                    );
                }
                sourceIndex = static_cast<std::uint32_t>(
                    found - world.sceneBodyIndices().begin()
                );
            }
        }
        if (jointStateSensor) {
            if (sensor.parentKind !=
                    MR_WORLD_SENSOR_PARENT_ASSET ||
                sensor.parentBodyIndex != MR_INVALID_INDEX) {
                return reject(
                    SensorCompileStatus::invalidSpec,
                    element + ".parent",
                    "joint-state sensor must be asset-owned and not body-attached"
                );
            }
            const auto found = std::find(
                world.model().jointNames.begin(),
                world.model().jointNames.end(),
                sensor.target
            );
            if (found == world.model().jointNames.end()) {
                return reject(
                    SensorCompileStatus::unresolvedSemantic,
                    element + ".target",
                    "joint-state target is unresolved: " +
                        sensor.target
                );
            }
            semanticSource = static_cast<std::uint32_t>(
                found - world.model().jointNames.begin()
            );
            if (semanticSource >= world.model().joints.size()) {
                return reject(
                    SensorCompileStatus::unresolvedSemantic,
                    element + ".target",
                    "joint-state target has no compiled joint descriptor"
                );
            }
            const MRJointDescriptorGPU& joint =
                world.model().joints[semanticSource];
            if (joint.nq != 1u || joint.nv != 1u ||
                (joint.jointType != MR_JOINT_REVOLUTE &&
                 joint.jointType != MR_JOINT_PRISMATIC &&
                 joint.jointType != MR_JOINT_CONTINUOUS)) {
                return reject(
                    SensorCompileStatus::invalidSpec,
                    element + ".target",
                    "joint-state target must be a scalar revolute, prismatic, or continuous joint"
                );
            }
            sourceIndex = joint.qOffset;
            sourceOwner = joint.vOffset;
        }
        if (actuatorStateSensor) {
            if (sensor.parentKind !=
                    MR_WORLD_SENSOR_PARENT_ASSET ||
                sensor.parentBodyIndex != MR_INVALID_INDEX) {
                return reject(
                    SensorCompileStatus::invalidSpec,
                    element + ".parent",
                    "actuator-state sensor must be asset-owned and not body-attached"
                );
            }
            const auto found = std::find(
                world.model().dofNames.begin(),
                world.model().dofNames.end(),
                sensor.target
            );
            if (found == world.model().dofNames.end()) {
                return reject(
                    SensorCompileStatus::unresolvedSemantic,
                    element + ".target",
                    "actuator-state target is unresolved: " +
                        sensor.target
                );
            }
            semanticSource = static_cast<std::uint32_t>(
                found - world.model().dofNames.begin()
            );
            if (semanticSource >= world.model().dofs.size() ||
                (world.model().dofs[semanticSource].flags &
                 MR_DOF_FLAG_ACTUATED) == 0u) {
                return reject(
                    SensorCompileStatus::invalidSpec,
                    element + ".target",
                    "actuator-state target must resolve to an actuated generalized coordinate"
                );
            }
            sourceIndex = semanticSource;
            sourceOwner = world.model().dofs[semanticSource]
                .articulationIndex;
        }
        if ((sensor.kind == MR_WORLD_SENSOR_STATE ||
             sensor.kind == MR_WORLD_SENSOR_FRAME_TWIST_WORLD ||
             sensor.kind == MR_WORLD_SENSOR_FORCE_TORQUE ||
             sensor.kind == MR_WORLD_SENSOR_IMU ||
             sensor.kind == MR_WORLD_SENSOR_CONTACT_STATE ||
             sensor.kind == MR_WORLD_SENSOR_TACTILE_DEPTH) &&
            !bodyParent) {
            return reject(
                SensorCompileStatus::invalidSpec,
                element + ".parent",
                "physical sensor requires a rigid-body or articulated-link parent"
            );
        }
        if (imageSensor(sensor.kind) &&
            (sensor.width == 0u || sensor.height == 0u ||
             !finite(sensor.intrinsics) ||
             !(sensor.intrinsics.x > 0.0f) ||
             !(sensor.intrinsics.y > 0.0f))) {
            return reject(
                SensorCompileStatus::invalidSpec,
                element + ".image",
                "image sensor dimensions or intrinsics are invalid"
            );
        }
        const double latencySamplesWide = std::ceil(
            static_cast<double>(sensor.latencySeconds) *
            sensor.nominalRateHz
        );
        const double historyNeeded = latencySamplesWide + 1.0;
        if (!std::isfinite(historyNeeded) ||
            historyNeeded > sensor.historyLength ||
            latencySamplesWide >
                std::numeric_limits<std::uint32_t>::max()) {
            return reject(
                SensorCompileStatus::invalidSpec,
                element + ".historyLength",
                "sensor history does not cover its authored latency"
            );
        }

        const double periodWide =
            static_cast<double>(kNanosecondsPerSecond) /
            sensor.nominalRateHz;
        if (!std::isfinite(periodWide) || periodWide < 0.5 ||
            periodWide > static_cast<double>(
                std::numeric_limits<std::int64_t>::max()
            )) {
            return reject(
                SensorCompileStatus::invalidSpec,
                element + ".nominalRateHz",
                "sensor rate is outside the deterministic scheduler tick range"
            );
        }
        const std::uint64_t period = static_cast<std::uint64_t>(
            std::llround(periodWide)
        );

        std::uint32_t channels = channelCount(sensor.kind);
        std::uint32_t outputCount = channels;
        std::uint32_t tactileOrdinal = MR_INVALID_INDEX;
        if (imageSensor(sensor.kind)) {
            std::uint32_t pixels = 0u;
            if (!checkedMultiply(sensor.width, sensor.height, pixels) ||
                !checkedMultiply(pixels, channels, outputCount)) {
                return reject(
                    SensorCompileStatus::arithmeticOverflow,
                    element + ".output",
                    "image sensor output layout overflows 32-bit indexing"
                );
            }
        } else if (sensor.kind == MR_WORLD_SENSOR_TACTILE_DEPTH) {
            tactileOrdinal = tactileIndex(tactile, sensor.id);
            if (tactileOrdinal == MR_INVALID_INDEX ||
                !usedTactile.insert(tactileOrdinal).second) {
                return reject(
                    SensorCompileStatus::invalidTactileSystem,
                    element + ".tactile",
                    "tactile metadata does not resolve uniquely into the cooked system"
                );
            }
            const MRTactileSensorGPU& cooked =
                tactile.sensors[tactileOrdinal];
            if (cooked.topology.x != sensor.parentBodyIndex ||
                cooked.atlasAndTargets.x != sensor.width ||
                cooked.atlasAndTargets.y != sensor.height) {
                return reject(
                    SensorCompileStatus::invalidTactileSystem,
                    element + ".tactile",
                    "tactile metadata disagrees with the cooked sensor"
                );
            }
            channels = 1u;
            if (!checkedAdd(cooked.topology.w, 9u, outputCount)) {
                return reject(
                    SensorCompileStatus::arithmeticOverflow,
                    element + ".output",
                    "tactile deformation, wrench, and contact-point layout overflows"
                );
            }
        }
        std::uint32_t historyCount = 0u;
        std::uint32_t nextOutput = 0u;
        std::uint32_t nextHistory = 0u;
        if (outputCount == 0u ||
            !checkedMultiply(
                outputCount,
                sensor.historyLength,
                historyCount
            ) ||
            !checkedAdd(
                staged->layout.outputElementCount,
                outputCount,
                nextOutput
            ) ||
            !checkedAdd(
                staged->layout.historyElementCount,
                historyCount,
                nextHistory
            )) {
            return reject(
                SensorCompileStatus::arithmeticOverflow,
                element + ".output",
                "sensor output/history layout overflows 32-bit indexing"
            );
        }

        const SensorExecutionDomain domain = executionDomain(sensor.kind);
        if (sensor.kind != MR_WORLD_SENSOR_CONTACT_STATE &&
            !sensor.filterBodies.empty()) {
            return reject(
                SensorCompileStatus::invalidSpec,
                element + ".filterBodies",
                "body filters are only valid for contact-state sensors"
            );
        }
        std::vector<std::uint32_t> resolvedFilters;
        resolvedFilters.reserve(sensor.filterBodies.size());
        for (const std::string& name : sensor.filterBodies) {
            const auto found = std::find(
                world.model().bodyNames.begin(),
                world.model().bodyNames.end(),
                name
            );
            if (name.empty() ||
                found == world.model().bodyNames.end()) {
                return reject(
                    SensorCompileStatus::unresolvedSemantic,
                    element + ".filterBodies",
                    "contact-state filter body is empty or unresolved: " +
                        name
                );
            }
            resolvedFilters.push_back(
                static_cast<std::uint32_t>(
                    found - world.model().bodyNames.begin()
                )
            );
        }
        std::sort(resolvedFilters.begin(), resolvedFilters.end());
        if (std::adjacent_find(
                resolvedFilters.begin(),
                resolvedFilters.end()
            ) != resolvedFilters.end()) {
            return reject(
                SensorCompileStatus::duplicateSemantic,
                element + ".filterBodies",
                "contact-state filter body is duplicated"
            );
        }
        if (staged->filterBodies.size() >
                std::numeric_limits<std::uint32_t>::max() ||
            resolvedFilters.size() >
                std::numeric_limits<std::uint32_t>::max() ||
            resolvedFilters.size() >
                std::numeric_limits<std::uint32_t>::max() -
                    staged->filterBodies.size()) {
            return reject(
                SensorCompileStatus::arithmeticOverflow,
                element + ".filterBodies",
                "contact-state filter table exceeds 32-bit indexing"
            );
        }
        const std::uint32_t filterOffset =
            static_cast<std::uint32_t>(staged->filterBodies.size());
        const std::uint32_t filterCount =
            static_cast<std::uint32_t>(resolvedFilters.size());
        staged->filterBodies.insert(
            staged->filterBodies.end(),
            resolvedFilters.begin(),
            resolvedFilters.end()
        );
        const std::uint64_t randomIdentity =
            sensorRandomIdentity(sensor.id);
        if (!uniqueRandomIdentities.insert(randomIdentity).second) {
            return reject(
                SensorCompileStatus::duplicateSemantic,
                element + ".id",
                "sensor ids collide in the 64-bit counter-RNG identity"
            );
        }
        MRSensorDescriptorGPU descriptor{};
        descriptor.identity = {
            sensor.kind,
            sensor.parentKind,
            sensor.parentBodyIndex,
            sensor.consumerFlags,
        };
        descriptor.output = {
            staged->layout.outputElementCount,
            outputCount,
            staged->layout.historyElementCount,
            sensor.historyLength,
        };
        descriptor.schedule = {
            static_cast<std::uint32_t>(period),
            static_cast<std::uint32_t>(period >> 32u),
            sensor.schedulePhase,
            static_cast<std::uint32_t>(latencySamplesWide),
        };
        descriptor.dimensions = {
            sensor.width,
            sensor.height,
            channels,
            0u,
        };
        descriptor.source = {
            sourceIndex,
            sourceOwner,
            tactileOrdinal != MR_INVALID_INDEX
                ? tactileOrdinal
                : semanticSource,
            static_cast<std::uint32_t>(domain),
        };
        descriptor.filter = {
            filterOffset,
            filterCount,
            0u,
            0u,
        };
        descriptor.randomIdentity = {
            static_cast<std::uint32_t>(randomIdentity),
            static_cast<std::uint32_t>(randomIdentity >> 32u),
            0u,
            0u,
        };
        descriptor.localPosition = sensor.localPose.position;
        descriptor.localOrientation = sensor.localPose.orientation;
        if (tactileOrdinal != MR_INVALID_INDEX) {
            const MRTactileSensorGPU& cooked =
                tactile.sensors[tactileOrdinal];
            descriptor.localPosition = {
                cooked.localPositionAndQueryEpsilon.x,
                cooked.localPositionAndQueryEpsilon.y,
                cooked.localPositionAndQueryEpsilon.z,
                0.0f,
            };
            descriptor.localOrientation =
                cooked.localOrientation;
        } else if (bodyParent) {
            const mr_float4 centerOfMass =
                world.model()
                    .bodies[sensor.parentBodyIndex]
                    .centerOfMass;
            descriptor.localPosition = {
                sensor.localPose.position.x - centerOfMass.x,
                sensor.localPose.position.y - centerOfMass.y,
                sensor.localPose.position.z - centerOfMass.z,
                0.0f,
            };
        }
        descriptor.timing = {
            sensor.latencySeconds,
            sensor.nominalRateHz,
            sensor.exposureSeconds,
            sensor.frameJitterSeconds,
        };
        descriptor.noise = {
            sensor.valueNoiseSigma,
            sensor.biasNoiseSigma,
            sensor.dropoutProbability,
            sensor.depthQuantumMeters,
        };
        descriptor.range = {
            sensor.minimumDepthMeters,
            sensor.maximumDepthMeters,
            sensor.motionBlurScale,
            sensor.focalScale,
        };
        descriptor.intrinsics = sensor.intrinsics;
        descriptor.distortion = sensor.distortion;

        staged->sensorIds.push_back(sensor.id);
        staged->descriptors.push_back(descriptor);
        staged->layout.outputElementCount = nextOutput;
        staged->layout.historyElementCount = nextHistory;
        staged->layout.filterBodyCount =
            static_cast<std::uint32_t>(staged->filterBodies.size());
        staged->layout.maximumHistoryLength = std::max(
            staged->layout.maximumHistoryLength,
            sensor.historyLength
        );
        switch (domain) {
        case SensorExecutionDomain::nativeState:
            ++staged->layout.nativeStateSensorCount;
            break;
        case SensorExecutionDomain::presentation:
            ++staged->layout.presentationSensorCount;
            break;
        case SensorExecutionDomain::tactile:
            ++staged->layout.tactileSensorCount;
            break;
        }
    }
    if (usedTactile.size() != tactile.sensors.size()) {
        return reject(
            SensorCompileStatus::invalidTactileSystem,
            "tactile",
            "cooked tactile system contains an unbound sensor"
        );
    }
    if (sensors.size() > std::numeric_limits<std::uint32_t>::max() ||
        tactile.samples.size() >
            std::numeric_limits<std::uint32_t>::max()) {
        return reject(
            SensorCompileStatus::arithmeticOverflow,
            "sensors",
            "sensor or tactile sample count exceeds 32-bit indexing"
        );
    }
    staged->layout.sensorCount =
        static_cast<std::uint32_t>(sensors.size());
    staged->layout.tactileSampleCount =
        static_cast<std::uint32_t>(tactile.samples.size());

    Hash hash;
    hash.scalar<std::uint32_t>(MR_SENSOR_PROGRAM_ABI_VERSION);
    hash.scalar(staged->worldFingerprint);
    for (std::size_t index = 0u;
         index < staged->sensorIds.size();
         ++index) {
        hash.string(staged->sensorIds[index]);
        hash.scalar(staged->descriptors[index]);
    }
    hash.scalar(tactile.fingerprint);
    hash.span<std::uint32_t>(staged->filterBodies);
    staged->fingerprint = hash.finish();
    staged->header.sensorFingerprint = staged->fingerprint;
    staged->header.worldFingerprint = staged->worldFingerprint;
    staged->header.counts = {
        staged->layout.sensorCount,
        staged->layout.outputElementCount,
        staged->layout.historyElementCount,
        staged->layout.tactileSampleCount,
    };
    staged->header.executionCounts = {
        staged->layout.nativeStateSensorCount,
        staged->layout.presentationSensorCount,
        staged->layout.tactileSensorCount,
        staged->layout.sensorCount,
    };
    staged->header.layout = {
        staged->layout.maximumHistoryLength,
        static_cast<std::uint32_t>(sizeof(MRSensorDescriptorGPU)),
        static_cast<std::uint32_t>(sizeof(MRSensorRuntimeStateGPU)),
        static_cast<std::uint32_t>(
            sizeof(MRSensorSampleMetadataGPU)
        ),
    };
    staged->header.reserved.x = MR_SENSOR_PROGRAM_ABI_VERSION;
    staged->header.reserved.y = staged->layout.filterBodyCount;

    SensorCompileDiagnostics diagnostics;
    diagnostics.fingerprint = staged->fingerprint;
    output.storage_ = std::move(staged);
    return diagnostics;
}

const char* sensorCompileStatusName(
    const SensorCompileStatus status
) noexcept {
    switch (status) {
    case SensorCompileStatus::success:
        return "success";
    case SensorCompileStatus::invalidWorld:
        return "invalid-world";
    case SensorCompileStatus::invalidSpec:
        return "invalid-spec";
    case SensorCompileStatus::unresolvedSemantic:
        return "unresolved-semantic";
    case SensorCompileStatus::duplicateSemantic:
        return "duplicate-semantic";
    case SensorCompileStatus::invalidTactileSystem:
        return "invalid-tactile-system";
    case SensorCompileStatus::arithmeticOverflow:
        return "arithmetic-overflow";
    case SensorCompileStatus::internalFailure:
        return "internal-failure";
    }
    return "unknown";
}

} // namespace metalrobo
