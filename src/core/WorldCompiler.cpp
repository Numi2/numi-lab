#include "metalrobo/WorldCompiler.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace metalrobo {
namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;
constexpr std::uint32_t kPhiloxM0 = 0xd2511f53u;
constexpr std::uint32_t kPhiloxM1 = 0xcd9e8d57u;
constexpr std::uint32_t kPhiloxW0 = 0x9e3779b9u;
constexpr std::uint32_t kPhiloxW1 = 0xbb67ae85u;

bool fail(std::string* reason, std::string message) {
    if (reason != nullptr) {
        *reason = std::move(message);
    }
    return false;
}

WorldCompileResult compileFail(
    const WorldCompileStatus status,
    std::string message
) {
    return {status, std::move(message)};
}

bool finite(const float value) {
    return std::isfinite(value);
}

bool finite(const double value) {
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

bool imageSensor(const MRWorldSensorKind kind) {
    return kind == MR_WORLD_SENSOR_RGB ||
        kind == MR_WORLD_SENSOR_DEPTH ||
        kind == MR_WORLD_SENSOR_RGBD ||
        kind == MR_WORLD_SENSOR_SEGMENTATION;
}

bool assetTarget(const MRWorldVariationTarget target) {
    return target <= MR_WORLD_TARGET_ASSET_RENDER_ALTERNATIVE ||
        target == MR_WORLD_TARGET_CLUTTER_SET;
}

bool sensorTarget(const MRWorldVariationTarget target) {
    return target >= MR_WORLD_TARGET_SENSOR_POSITION_X &&
        target <= MR_WORLD_TARGET_SENSOR_DEPTH_DROPOUT;
}

bool appearanceTarget(const MRWorldVariationTarget target) {
    return target >= MR_WORLD_TARGET_APPEARANCE_EXPOSURE &&
        target <= MR_WORLD_TARGET_APPEARANCE_LIGHT_INTENSITY;
}

class HashBuilder {
public:
    void appendBytes(const void* data, const std::size_t size) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (std::size_t index = 0u; index < size; ++index) {
            value_ ^= bytes[index];
            value_ *= kFnvPrime;
        }
    }

    template <typename T>
    void appendScalar(const T& value) {
        appendBytes(&value, sizeof(value));
    }

    template <typename T>
    void appendSpan(const std::span<const T> values) {
        const std::uint64_t size = values.size();
        appendScalar(size);
        if (!values.empty()) {
            appendBytes(values.data(), values.size_bytes());
        }
    }

    void appendString(const std::string_view value) {
        const std::uint64_t size = value.size();
        appendScalar(size);
        appendBytes(value.data(), value.size());
    }

    [[nodiscard]] std::uint64_t finish() const noexcept {
        return value_;
    }

private:
    std::uint64_t value_ = kFnvOffset;
};

void appendPose(HashBuilder& hash, const WorldPose& pose) {
    hash.appendScalar(pose.position);
    hash.appendScalar(pose.orientation);
}

void appendAnchor(HashBuilder& hash, const SemanticAnchor& anchor) {
    hash.appendString(anchor.id);
    appendPose(hash, anchor.localPose);
    hash.appendScalar(anchor.radius);
    hash.appendScalar(anchor.flags);
}

void appendAsset(HashBuilder& hash, const WorldAsset& asset) {
    hash.appendString(asset.id);
    hash.appendString(asset.semanticClass);
    hash.appendScalar(asset.role);
    hash.appendScalar(asset.render);
    hash.appendScalar(asset.collision);
    hash.appendScalar(asset.dynamics);
    appendPose(hash, asset.initialPose);
    hash.appendScalar(asset.uniformScale);
    hash.appendScalar(asset.massScale);
    hash.appendScalar(asset.frictionScale);
    hash.appendScalar(asset.restitutionScale);
    hash.appendScalar(asset.dampingScale);
    hash.appendScalar(asset.controllerGainScale);
    hash.appendScalar(asset.controllerDampingScale);
    hash.appendScalar(asset.controllerLatencySeconds);
    hash.appendScalar(asset.payloadScale);
    hash.appendScalar(asset.renderAlternative);
    hash.appendScalar(asset.collisionAlternative);
    hash.appendScalar(asset.articulationIndex);
    hash.appendString(asset.topologyCohort);
    hash.appendSpan<std::uint32_t>(asset.bodyIndices);
    hash.appendSpan<std::uint32_t>(asset.shapeIndices);
    hash.appendSpan<std::uint32_t>(asset.materialIndices);
    const std::uint64_t anchorCount = asset.anchors.size();
    hash.appendScalar(anchorCount);
    for (const SemanticAnchor& anchor : asset.anchors) {
        appendAnchor(hash, anchor);
    }
}

void appendSensor(HashBuilder& hash, const SensorSpec& sensor) {
    hash.appendString(sensor.id);
    hash.appendString(sensor.parentAssetId);
    hash.appendScalar(sensor.kind);
    appendPose(hash, sensor.localPose);
    hash.appendScalar(sensor.width);
    hash.appendScalar(sensor.height);
    hash.appendScalar(sensor.intrinsics);
    hash.appendScalar(sensor.distortion);
    hash.appendScalar(sensor.focalScale);
    hash.appendScalar(sensor.colorNoiseSigma);
    hash.appendScalar(sensor.depthNoiseSigma);
    hash.appendScalar(sensor.depthDropout);
    hash.appendScalar(sensor.latencySeconds);
}

void appendArtifact(HashBuilder& hash, const EpisodeArtifact& artifact) {
    hash.appendString(artifact.id);
    hash.appendScalar(artifact.kind);
    hash.appendScalar(artifact.producer);
    hash.appendString(artifact.assetId);
    hash.appendString(artifact.uri);
    hash.appendString(artifact.contentHash);
    hash.appendScalar(artifact.startTimeSeconds);
    hash.appendScalar(artifact.endTimeSeconds);
}

void appendTask(HashBuilder& hash, const TaskSpec& task) {
    hash.appendString(task.id);
    hash.appendString(task.robotAssetId);
    hash.appendString(task.manipulatedAssetId);
    hash.appendString(task.targetAssetId);
    hash.appendString(task.targetAnchorId);
    hash.appendScalar(task.controlPeriodSeconds);
    hash.appendScalar(task.horizonSeconds);
}

void appendEngineModel(HashBuilder& hash, const EngineModel& model) {
    hash.appendString(model.name);
    hash.appendScalar(model.world);
    hash.appendSpan<MRArticulationGPU>(model.articulations);
    hash.appendSpan<MRJointDescriptorGPU>(model.joints);
    hash.appendSpan<MRDofPropertiesGPU>(model.dofs);
    hash.appendSpan<MRBodyPropertiesGPU>(model.bodies);
    hash.appendSpan<MRShapeGPU>(model.shapes);
    hash.appendSpan<MRMaterialGPU>(model.materials);
    hash.appendSpan<MRGeometryHeaderGPU>(model.geometryHeaders);
    hash.appendSpan<mr_float4>(model.geometryVertices);
    hash.appendSpan<std::uint32_t>(model.geometryIndices);
    hash.appendSpan<MRConvexFaceGPU>(model.convexFaces);
    hash.appendSpan<MRConvexHalfEdgeGPU>(model.convexHalfEdges);
    hash.appendSpan<MRMeshBVHNodeGPU>(model.meshBvhNodes);
    hash.appendSpan<MRMeshTriangleGPU>(model.meshTriangles);
    hash.appendSpan<CollisionPairExclusion>(
        model.collisionExclusions
    );
    hash.appendScalar(model.constraintProgram.abiVersion);
    hash.appendSpan<ConstraintIRBlock>(
        model.constraintProgram.blocks
    );
    hash.appendSpan<ConstraintIREndpoint>(
        model.constraintProgram.endpoints
    );
    hash.appendSpan<ConstraintIRRow>(
        model.constraintProgram.rows
    );
    hash.appendSpan<ConstraintIRCone>(
        model.constraintProgram.cones
    );
    hash.appendSpan<float>(
        model.constraintProgram.warmImpulses
    );
    hash.appendSpan<float>(model.defaultQ);
    hash.appendSpan<float>(model.defaultV);
}

std::uint32_t computeCapabilities(
    const std::vector<WorldAsset>& assets,
    const std::vector<SensorSpec>& sensors
) {
    std::uint32_t result = MR_WORLD_CAP_STATE;
    for (const WorldAsset& asset : assets) {
        switch (asset.render) {
        case MR_WORLD_RENDER_GAUSSIAN_FIELD:
            result |= MR_WORLD_CAP_GAUSSIAN_RENDER;
            break;
        case MR_WORLD_RENDER_MESH_PBR:
        case MR_WORLD_RENDER_PROCEDURAL:
            result |= MR_WORLD_CAP_MESH_RENDER;
            break;
        case MR_WORLD_RENDER_NEURAL_RESIDUAL:
        case MR_WORLD_RENDER_NONE:
            break;
        }
        switch (asset.dynamics) {
        case MR_WORLD_DYNAMICS_RIGID:
            result |= MR_WORLD_CAP_RIGID_DYNAMICS;
            break;
        case MR_WORLD_DYNAMICS_ARTICULATED:
            result |= MR_WORLD_CAP_ARTICULATED_DYNAMICS;
            break;
        case MR_WORLD_DYNAMICS_ROD:
            result |= MR_WORLD_CAP_ROD_DYNAMICS;
            break;
        case MR_WORLD_DYNAMICS_SHELL:
            result |= MR_WORLD_CAP_SHELL_DYNAMICS;
            break;
        case MR_WORLD_DYNAMICS_SOFT_VOLUME:
            result |= MR_WORLD_CAP_SOFT_VOLUME_DYNAMICS;
            break;
        case MR_WORLD_DYNAMICS_STATIC:
        case MR_WORLD_DYNAMICS_KINEMATIC:
            break;
        }
    }
    for (const SensorSpec& sensor : sensors) {
        switch (sensor.kind) {
        case MR_WORLD_SENSOR_RGB:
            result |= MR_WORLD_CAP_RGB;
            break;
        case MR_WORLD_SENSOR_DEPTH:
            result |= MR_WORLD_CAP_DEPTH;
            break;
        case MR_WORLD_SENSOR_RGBD:
            result |= MR_WORLD_CAP_RGB | MR_WORLD_CAP_DEPTH;
            break;
        case MR_WORLD_SENSOR_SEGMENTATION:
            result |= MR_WORLD_CAP_SEGMENTATION;
            break;
        case MR_WORLD_SENSOR_STATE:
        case MR_WORLD_SENSOR_FORCE_TORQUE:
            break;
        }
    }
    return result;
}

std::array<std::uint32_t, 4> philoxRound(
    const std::array<std::uint32_t, 4>& counter,
    const std::array<std::uint32_t, 2>& key
) {
    const std::uint64_t product0 =
        static_cast<std::uint64_t>(kPhiloxM0) * counter[0];
    const std::uint64_t product1 =
        static_cast<std::uint64_t>(kPhiloxM1) * counter[2];
    return {
        static_cast<std::uint32_t>(product1 >> 32u) ^
            counter[1] ^ key[0],
        static_cast<std::uint32_t>(product1),
        static_cast<std::uint32_t>(product0 >> 32u) ^
            counter[3] ^ key[1],
        static_cast<std::uint32_t>(product0),
    };
}

std::array<std::uint32_t, 4> philox(
    const std::uint64_t seed,
    const std::uint32_t instance,
    const std::uint32_t variation,
    const MRWorldVariationGPU& descriptor
) {
    std::array<std::uint32_t, 4> counter{
        instance,
        variation,
        descriptor.random.x,
        descriptor.random.y ^ descriptor.random.z,
    };
    std::array<std::uint32_t, 2> key{
        static_cast<std::uint32_t>(seed),
        static_cast<std::uint32_t>(seed >> 32u),
    };
    for (std::uint32_t round = 0u; round < 10u; ++round) {
        counter = philoxRound(counter, key);
        key[0] += kPhiloxW0;
        key[1] += kPhiloxW1;
    }
    return counter;
}

float uniform01(const std::uint32_t bits) {
    constexpr double scale = 1.0 / 4294967296.0;
    return static_cast<float>(
        (static_cast<double>(bits) + 0.5) * scale
    );
}

struct SampledValue {
    float scalar = 0.0f;
    std::uint32_t categorical = 0u;
};

SampledValue sampleValue(
    const MRWorldVariationGPU& descriptor,
    const std::span<const std::uint32_t> categoricalValues,
    const std::array<std::uint32_t, 4>& random
) {
    const float u0 = uniform01(random[0]);
    const float u1 = uniform01(random[1]);
    switch (descriptor.binding.y) {
    case MR_WORLD_DISTRIBUTION_CONSTANT:
        return {descriptor.parameters.x, 0u};
    case MR_WORLD_DISTRIBUTION_UNIFORM:
        return {
            descriptor.parameters.x +
                u0 * (descriptor.parameters.y - descriptor.parameters.x),
            0u,
        };
    case MR_WORLD_DISTRIBUTION_LOG_UNIFORM: {
        const float lower = std::log(descriptor.parameters.x);
        const float upper = std::log(descriptor.parameters.y);
        return {std::exp(lower + u0 * (upper - lower)), 0u};
    }
    case MR_WORLD_DISTRIBUTION_NORMAL_CLAMPED: {
        const float radius = std::sqrt(
            -2.0f * std::log(std::max(u0, 1.0e-12f))
        );
        const float normal = radius *
            std::cos(2.0f * std::numbers::pi_v<float> * u1);
        const float value =
            descriptor.parameters.x + descriptor.parameters.y * normal;
        return {
            std::clamp(
                value,
                descriptor.parameters.z,
                descriptor.parameters.w
            ),
            0u,
        };
    }
    case MR_WORLD_DISTRIBUTION_CATEGORICAL: {
        const std::size_t first = descriptor.categorical.x;
        const std::size_t count = descriptor.categorical.y;
        if (first > categoricalValues.size() ||
            count > categoricalValues.size() - first ||
            count == 0u) {
            return {};
        }
        const std::size_t selected = std::min(
            static_cast<std::size_t>(u0 * static_cast<float>(count)),
            count - 1u
        );
        return {
            static_cast<float>(categoricalValues[first + selected]),
            categoricalValues[first + selected],
        };
    }
    default:
        return {};
    }
}

mr_float4 quaternionProduct(const mr_float4 a, const mr_float4 b) {
    return {
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    };
}

mr_float4 normalized(const mr_float4 value) {
    const float norm = std::sqrt(
        value.x * value.x + value.y * value.y +
        value.z * value.z + value.w * value.w
    );
    if (!(norm > 0.0f) || !finite(norm)) {
        return {0.0f, 0.0f, 0.0f, 1.0f};
    }
    const float inverse = 1.0f / norm;
    return {
        value.x * inverse,
        value.y * inverse,
        value.z * inverse,
        value.w * inverse,
    };
}

mr_float4 axisAngle(
    const float x,
    const float y,
    const float z,
    const float angle
) {
    const float half = 0.5f * angle;
    const float sine = std::sin(half);
    return {x * sine, y * sine, z * sine, std::cos(half)};
}

void applyOrientationDelta(
    mr_float4& orientation,
    const MRWorldVariationTarget target,
    const float value
) {
    mr_float4 delta{0.0f, 0.0f, 0.0f, 1.0f};
    switch (target) {
    case MR_WORLD_TARGET_ASSET_ORIENTATION_ROLL:
    case MR_WORLD_TARGET_SENSOR_ORIENTATION_ROLL:
        delta = axisAngle(1.0f, 0.0f, 0.0f, value);
        break;
    case MR_WORLD_TARGET_ASSET_ORIENTATION_PITCH:
    case MR_WORLD_TARGET_SENSOR_ORIENTATION_PITCH:
        delta = axisAngle(0.0f, 1.0f, 0.0f, value);
        break;
    case MR_WORLD_TARGET_ASSET_ORIENTATION_YAW:
    case MR_WORLD_TARGET_SENSOR_ORIENTATION_YAW:
        delta = axisAngle(0.0f, 0.0f, 1.0f, value);
        break;
    default:
        return;
    }
    orientation = normalized(quaternionProduct(orientation, delta));
}

void applyAssetVariation(
    MRWorldAssetInstanceGPU& asset,
    const MRWorldVariationTarget target,
    const SampledValue value
) {
    switch (target) {
    case MR_WORLD_TARGET_ASSET_POSITION_X:
        asset.positionAndScale.x += value.scalar;
        break;
    case MR_WORLD_TARGET_ASSET_POSITION_Y:
        asset.positionAndScale.y += value.scalar;
        break;
    case MR_WORLD_TARGET_ASSET_POSITION_Z:
        asset.positionAndScale.z += value.scalar;
        break;
    case MR_WORLD_TARGET_ASSET_SCALE:
        asset.positionAndScale.w *= value.scalar;
        break;
    case MR_WORLD_TARGET_ASSET_ORIENTATION_ROLL:
    case MR_WORLD_TARGET_ASSET_ORIENTATION_PITCH:
    case MR_WORLD_TARGET_ASSET_ORIENTATION_YAW:
        applyOrientationDelta(asset.orientation, target, value.scalar);
        break;
    case MR_WORLD_TARGET_ASSET_MASS_SCALE:
        asset.physical.x *= value.scalar;
        break;
    case MR_WORLD_TARGET_ASSET_FRICTION_SCALE:
        asset.physical.y *= value.scalar;
        break;
    case MR_WORLD_TARGET_ASSET_RESTITUTION_SCALE:
        asset.physical.z *= value.scalar;
        break;
    case MR_WORLD_TARGET_ASSET_DAMPING_SCALE:
        asset.physical.w *= value.scalar;
        break;
    case MR_WORLD_TARGET_ROBOT_GAIN_SCALE:
        asset.controller.x *= value.scalar;
        break;
    case MR_WORLD_TARGET_ROBOT_DAMPING_SCALE:
        asset.controller.y *= value.scalar;
        break;
    case MR_WORLD_TARGET_ROBOT_LATENCY_SECONDS:
        asset.controller.z += value.scalar;
        break;
    case MR_WORLD_TARGET_ROBOT_PAYLOAD_SCALE:
        asset.controller.w *= value.scalar;
        break;
    case MR_WORLD_TARGET_ASSET_RENDER_ALTERNATIVE:
    case MR_WORLD_TARGET_CLUTTER_SET:
        asset.identity.y = value.categorical;
        break;
    default:
        break;
    }
}

void applySensorVariation(
    MRWorldSensorInstanceGPU& sensor,
    const MRWorldVariationTarget target,
    const float value
) {
    switch (target) {
    case MR_WORLD_TARGET_SENSOR_POSITION_X:
        sensor.positionAndFocalScale.x += value;
        break;
    case MR_WORLD_TARGET_SENSOR_POSITION_Y:
        sensor.positionAndFocalScale.y += value;
        break;
    case MR_WORLD_TARGET_SENSOR_POSITION_Z:
        sensor.positionAndFocalScale.z += value;
        break;
    case MR_WORLD_TARGET_SENSOR_ORIENTATION_ROLL:
    case MR_WORLD_TARGET_SENSOR_ORIENTATION_PITCH:
    case MR_WORLD_TARGET_SENSOR_ORIENTATION_YAW:
        applyOrientationDelta(sensor.orientation, target, value);
        break;
    case MR_WORLD_TARGET_SENSOR_FOCAL_SCALE:
        sensor.positionAndFocalScale.w *= value;
        break;
    case MR_WORLD_TARGET_SENSOR_LATENCY_SECONDS:
        sensor.noiseAndLatency.w += value;
        break;
    case MR_WORLD_TARGET_SENSOR_COLOR_NOISE:
        sensor.noiseAndLatency.x += value;
        break;
    case MR_WORLD_TARGET_SENSOR_DEPTH_NOISE:
        sensor.noiseAndLatency.y += value;
        break;
    case MR_WORLD_TARGET_SENSOR_DEPTH_DROPOUT:
        sensor.noiseAndLatency.z =
            std::clamp(sensor.noiseAndLatency.z + value, 0.0f, 1.0f);
        break;
    default:
        break;
    }
}

void applyAppearanceVariation(
    MRWorldAppearanceInstanceGPU& appearance,
    const MRWorldVariationTarget target,
    const float value
) {
    switch (target) {
    case MR_WORLD_TARGET_APPEARANCE_EXPOSURE:
        appearance.colorAndLight.x += value;
        break;
    case MR_WORLD_TARGET_APPEARANCE_WHITE_BALANCE:
        appearance.colorAndLight.y *= value;
        break;
    case MR_WORLD_TARGET_APPEARANCE_SATURATION:
        appearance.colorAndLight.z *= value;
        break;
    case MR_WORLD_TARGET_APPEARANCE_LIGHT_INTENSITY:
        appearance.colorAndLight.w *= value;
        break;
    default:
        break;
    }
}

bool validAsset(const WorldAsset& asset, std::string* reason) {
    if (asset.id.empty()) {
        return fail(reason, "world asset id is empty");
    }
    if (asset.role > MR_WORLD_ASSET_SENSOR_RIG ||
        asset.render > MR_WORLD_RENDER_PROCEDURAL ||
        asset.collision > MR_WORLD_COLLISION_DEFORMABLE_SURFACE ||
        asset.dynamics > MR_WORLD_DYNAMICS_SOFT_VOLUME) {
        return fail(reason, "world asset has an unknown representation");
    }
    if (!finite(asset.initialPose.position) ||
        !unitQuaternion(asset.initialPose.orientation) ||
        !finite(asset.uniformScale) || !(asset.uniformScale > 0.0f) ||
        !finite(asset.massScale) || !(asset.massScale > 0.0f) ||
        !finite(asset.frictionScale) || !(asset.frictionScale >= 0.0f) ||
        !finite(asset.restitutionScale) ||
        !(asset.restitutionScale >= 0.0f) ||
        !finite(asset.dampingScale) || !(asset.dampingScale >= 0.0f) ||
        !finite(asset.controllerGainScale) ||
        !(asset.controllerGainScale >= 0.0f) ||
        !finite(asset.controllerDampingScale) ||
        !(asset.controllerDampingScale >= 0.0f) ||
        !finite(asset.controllerLatencySeconds) ||
        !(asset.controllerLatencySeconds >= 0.0f) ||
        !finite(asset.payloadScale) || !(asset.payloadScale > 0.0f) ||
        asset.topologyCohort.empty()) {
        return fail(reason, "world asset has invalid authored parameters");
    }
    if (asset.role == MR_WORLD_ASSET_MANIPULATED &&
        asset.collision == MR_WORLD_COLLISION_NONE) {
        return fail(reason, "manipulated asset has no collision representation");
    }
    std::unordered_set<std::string> anchorIds;
    for (const SemanticAnchor& anchor : asset.anchors) {
        if (anchor.id.empty() || !anchorIds.insert(anchor.id).second ||
            !finite(anchor.localPose.position) ||
            !unitQuaternion(anchor.localPose.orientation) ||
            !finite(anchor.radius) || anchor.radius < 0.0f) {
            return fail(reason, "world asset has an invalid semantic anchor");
        }
    }
    return true;
}

bool validSensor(const SensorSpec& sensor, std::string* reason) {
    if (sensor.id.empty() || sensor.parentAssetId.empty() ||
        sensor.kind > MR_WORLD_SENSOR_FORCE_TORQUE ||
        !finite(sensor.localPose.position) ||
        !unitQuaternion(sensor.localPose.orientation) ||
        !finite(sensor.intrinsics) || !finite(sensor.distortion) ||
        !finite(sensor.focalScale) || !(sensor.focalScale > 0.0f) ||
        !finite(sensor.colorNoiseSigma) ||
        !(sensor.colorNoiseSigma >= 0.0f) ||
        !finite(sensor.depthNoiseSigma) ||
        !(sensor.depthNoiseSigma >= 0.0f) ||
        !finite(sensor.depthDropout) ||
        !(sensor.depthDropout >= 0.0f && sensor.depthDropout <= 1.0f) ||
        !finite(sensor.latencySeconds) ||
        !(sensor.latencySeconds >= 0.0f)) {
        return fail(reason, "sensor has invalid authored parameters");
    }
    if (imageSensor(sensor.kind) &&
        (sensor.width == 0u || sensor.height == 0u ||
         !(sensor.intrinsics.x > 0.0f) ||
         !(sensor.intrinsics.y > 0.0f))) {
        return fail(reason, "image sensor has invalid dimensions or intrinsics");
    }
    return true;
}

std::uint32_t findAnchor(
    const WorldAsset& asset,
    const std::string& id
) {
    for (std::uint32_t index = 0u; index < asset.anchors.size(); ++index) {
        if (asset.anchors[index].id == id) {
            return index;
        }
    }
    return MR_INVALID_INDEX;
}

bool validDistribution(
    const VariationParameter& variation,
    std::string* reason
) {
    if (variation.id.empty() ||
        variation.axis > MR_WORLD_VARIATION_CAMERA ||
        variation.distribution > MR_WORLD_DISTRIBUTION_CATEGORICAL ||
        variation.target > MR_WORLD_TARGET_CLUTTER_SET ||
        !finite(variation.parameters)) {
        return fail(reason, "variation has invalid identity or parameters");
    }
    switch (variation.distribution) {
    case MR_WORLD_DISTRIBUTION_CONSTANT:
        return true;
    case MR_WORLD_DISTRIBUTION_UNIFORM:
        if (variation.parameters.x > variation.parameters.y) {
            return fail(reason, "uniform variation has reversed bounds");
        }
        return true;
    case MR_WORLD_DISTRIBUTION_LOG_UNIFORM:
        if (!(variation.parameters.x > 0.0f) ||
            variation.parameters.x > variation.parameters.y) {
            return fail(reason, "log-uniform variation has invalid bounds");
        }
        return true;
    case MR_WORLD_DISTRIBUTION_NORMAL_CLAMPED:
        if (!(variation.parameters.y > 0.0f) ||
            variation.parameters.z > variation.parameters.w) {
            return fail(reason, "normal variation has invalid sigma or bounds");
        }
        return true;
    case MR_WORLD_DISTRIBUTION_CATEGORICAL:
        if (variation.categoricalValues.empty()) {
            return fail(reason, "categorical variation has no values");
        }
        return true;
    default:
        return fail(reason, "unknown variation distribution");
    }
}

} // namespace

std::uint32_t WorldTemplate::assetIndex(const std::string& id) const {
    for (std::uint32_t index = 0u; index < assets.size(); ++index) {
        if (assets[index].id == id) {
            return index;
        }
    }
    return MR_INVALID_INDEX;
}

std::uint32_t WorldTemplate::sensorIndex(const std::string& id) const {
    for (std::uint32_t index = 0u; index < sensors.size(); ++index) {
        if (sensors[index].id == id) {
            return index;
        }
    }
    return MR_INVALID_INDEX;
}

bool WorldTemplate::valid(std::string* reason) const {
    if (reason != nullptr) {
        reason->clear();
    }
    if (id.empty() || fingerprint == 0u ||
        topologyCohorts.empty() || appearances.empty()) {
        return fail(reason, "world template identity or cohorts are empty");
    }
    std::string engineReason;
    if (!engineModel.valid(&engineReason)) {
        return fail(reason, "world template engine model: " + engineReason);
    }
    if (assets.empty() || assetBindings.size() != assets.size()) {
        return fail(reason, "world template has no assets");
    }
    std::unordered_set<std::string> assetIds;
    std::vector<std::uint32_t> bodyOwners(
        engineModel.bodies.size(),
        MR_INVALID_INDEX
    );
    std::vector<std::uint32_t> shapeOwners(
        engineModel.shapes.size(),
        MR_INVALID_INDEX
    );
    for (std::uint32_t assetIndex = 0u;
         assetIndex < assets.size();
         ++assetIndex) {
        const WorldAsset& asset = assets[assetIndex];
        if (!validAsset(asset, reason)) {
            return false;
        }
        if (!assetIds.insert(asset.id).second) {
            return fail(reason, "world template has duplicate asset ids");
        }
        if (std::ranges::find(
                topologyCohorts,
                asset.topologyCohort
            ) == topologyCohorts.end()) {
            return fail(reason, "asset references an unknown topology cohort");
        }
        const auto cohort = std::ranges::find(
            topologyCohorts,
            asset.topologyCohort
        );
        if (asset.articulationIndex != MR_INVALID_INDEX &&
            asset.articulationIndex >=
                engineModel.articulations.size()) {
            return fail(reason, "asset references an unknown articulation");
        }
        if (asset.dynamics == MR_WORLD_DYNAMICS_ARTICULATED &&
            asset.articulationIndex == MR_INVALID_INDEX) {
            return fail(
                reason,
                "articulated asset has no engine articulation binding"
            );
        }
        if ((asset.role == MR_WORLD_ASSET_ROBOT ||
             asset.role == MR_WORLD_ASSET_MANIPULATED) &&
            asset.bodyIndices.empty()) {
            return fail(reason, "task-relevant asset has no engine body");
        }
        for (const std::uint32_t bodyIndex : asset.bodyIndices) {
            if (bodyIndex >= engineModel.bodies.size() ||
                bodyOwners[bodyIndex] != MR_INVALID_INDEX) {
                return fail(
                    reason,
                    "asset body binding is invalid or multiply owned"
                );
            }
            const MRBodyPropertiesGPU& body =
                engineModel.bodies[bodyIndex];
            if ((asset.dynamics == MR_WORLD_DYNAMICS_ARTICULATED &&
                 body.articulationIndex !=
                     asset.articulationIndex) ||
                (asset.dynamics != MR_WORLD_DYNAMICS_ARTICULATED &&
                 body.articulationIndex != MR_INVALID_INDEX) ||
                (asset.dynamics == MR_WORLD_DYNAMICS_STATIC &&
                 body.motionType != MR_MOTION_STATIC) ||
                (asset.dynamics == MR_WORLD_DYNAMICS_KINEMATIC &&
                 body.motionType != MR_MOTION_KINEMATIC) ||
                (asset.dynamics == MR_WORLD_DYNAMICS_RIGID &&
                 body.motionType != MR_MOTION_DYNAMIC)) {
                return fail(
                    reason,
                    "asset dynamics and engine body binding disagree"
                );
            }
            bodyOwners[bodyIndex] = assetIndex;
        }
        for (const std::uint32_t shapeIndex : asset.shapeIndices) {
            if (shapeIndex >= engineModel.shapes.size() ||
                shapeOwners[shapeIndex] != MR_INVALID_INDEX ||
                std::ranges::find(
                    asset.bodyIndices,
                    engineModel.shapes[shapeIndex].bodyIndex
                ) == asset.bodyIndices.end()) {
                return fail(
                    reason,
                    "asset shape binding is invalid or multiply owned"
                );
            }
            shapeOwners[shapeIndex] = assetIndex;
        }
        for (const std::uint32_t materialIndex :
             asset.materialIndices) {
            if (materialIndex >= engineModel.materials.size()) {
                return fail(reason, "asset material binding is invalid");
            }
        }

        const MRWorldAssetBindingGPU& binding =
            assetBindings[assetIndex];
        const auto validBindingRange = [this](
            const std::uint32_t offset,
            const std::uint32_t count
        ) {
            return offset <= bindingIndices.size() &&
                count <= bindingIndices.size() - offset;
        };
        if (binding.identity.x != assetIndex ||
            binding.identity.y != asset.role ||
            binding.identity.z != asset.render ||
            binding.identity.w != asset.collision ||
            binding.dynamics.x != asset.dynamics ||
            binding.dynamics.y != asset.articulationIndex ||
            binding.dynamics.z !=
                static_cast<std::uint32_t>(
                    cohort - topologyCohorts.begin()
                ) ||
            binding.dynamics.w != 0u ||
            binding.geometryRanges.y != asset.bodyIndices.size() ||
            binding.geometryRanges.w != asset.shapeIndices.size() ||
            binding.materialRangeAndAlternatives.y !=
                asset.materialIndices.size() ||
            binding.materialRangeAndAlternatives.z !=
                asset.renderAlternative ||
            binding.materialRangeAndAlternatives.w !=
                asset.collisionAlternative ||
            !validBindingRange(
                binding.geometryRanges.x,
                binding.geometryRanges.y
            ) ||
            !validBindingRange(
                binding.geometryRanges.z,
                binding.geometryRanges.w
            ) ||
            !validBindingRange(
                binding.materialRangeAndAlternatives.x,
                binding.materialRangeAndAlternatives.y
            ) ||
            !std::equal(
                asset.bodyIndices.begin(),
                asset.bodyIndices.end(),
                bindingIndices.begin() +
                    binding.geometryRanges.x
            ) ||
            !std::equal(
                asset.shapeIndices.begin(),
                asset.shapeIndices.end(),
                bindingIndices.begin() +
                    binding.geometryRanges.z
            ) ||
            !std::equal(
                asset.materialIndices.begin(),
                asset.materialIndices.end(),
                bindingIndices.begin() +
                    binding.materialRangeAndAlternatives.x
            )) {
            return fail(
                reason,
                "compiled asset binding does not match its world asset"
            );
        }
    }
    std::unordered_set<std::string> sensorIds;
    for (const SensorSpec& sensor : sensors) {
        if (!validSensor(sensor, reason)) {
            return false;
        }
        if (!sensorIds.insert(sensor.id).second) {
            return fail(reason, "world template has duplicate sensor ids");
        }
        if (assetIndex(sensor.parentAssetId) == MR_INVALID_INDEX) {
            return fail(reason, "sensor parent asset does not exist");
        }
    }
    std::unordered_set<std::string> appearanceIds;
    for (const AppearanceSpec& appearance : appearances) {
        if (appearance.id.empty() ||
            !appearanceIds.insert(appearance.id).second ||
            !finite(appearance.exposureStops) ||
            !finite(appearance.whiteBalanceScale) ||
            !(appearance.whiteBalanceScale > 0.0f) ||
            !finite(appearance.saturation) ||
            !(appearance.saturation >= 0.0f) ||
            !finite(appearance.lightIntensity) ||
            !(appearance.lightIntensity >= 0.0f) ||
            !finite(appearance.hueRadians) ||
            !finite(appearance.contrast) || !(appearance.contrast >= 0.0f) ||
            !finite(appearance.roughnessScale) ||
            !(appearance.roughnessScale >= 0.0f) ||
            !finite(appearance.metallicScale) ||
            !(appearance.metallicScale >= 0.0f)) {
            return fail(reason, "world template has invalid appearance data");
        }
    }
    if (task.id.empty() || !finite(task.controlPeriodSeconds) ||
        !(task.controlPeriodSeconds > 0.0) ||
        !finite(task.horizonSeconds) || !(task.horizonSeconds > 0.0)) {
        return fail(reason, "world template task is invalid");
    }
    const std::uint32_t robot = assetIndex(task.robotAssetId);
    const std::uint32_t manipulated = assetIndex(task.manipulatedAssetId);
    const std::uint32_t target = assetIndex(task.targetAssetId);
    if (robot == MR_INVALID_INDEX ||
        manipulated == MR_INVALID_INDEX ||
        target == MR_INVALID_INDEX) {
        return fail(reason, "task references an unknown asset");
    }
    if (assets[robot].role != MR_WORLD_ASSET_ROBOT ||
        assets[manipulated].role != MR_WORLD_ASSET_MANIPULATED ||
        findAnchor(assets[target], task.targetAnchorId) == MR_INVALID_INDEX) {
        return fail(reason, "task asset roles or target anchor are invalid");
    }
    return true;
}

bool WorldInstanceBatch::valid(std::string* reason) const {
    if (reason != nullptr) {
        reason->clear();
    }
    if (familyFingerprint == 0u && !instances.empty()) {
        return fail(reason, "world instance batch has no family fingerprint");
    }
    for (const MRWorldInstanceHeaderGPU& instance : instances) {
        if (instance.identity.w != MR_WORLD_COMPILER_ABI_VERSION ||
            instance.ranges.x > assets.size() ||
            instance.ranges.y > assets.size() - instance.ranges.x ||
            instance.ranges.z > sensors.size() ||
            instance.ranges.w > sensors.size() - instance.ranges.z ||
            instance.program.x > appearances.size() ||
            instance.program.y > appearances.size() - instance.program.x) {
            return fail(reason, "world instance ranges are invalid");
        }
    }
    for (const MRWorldAssetInstanceGPU& asset : assets) {
        if (!finite(asset.positionAndScale) ||
            !(asset.positionAndScale.w > 0.0f) ||
            !unitQuaternion(asset.orientation) ||
            !finite(asset.linearVelocity) ||
            !finite(asset.angularVelocity) ||
            !finite(asset.physical) ||
            !(asset.physical.x > 0.0f) ||
            asset.physical.y < 0.0f ||
            asset.physical.z < 0.0f ||
            asset.physical.w < 0.0f ||
            !finite(asset.controller) ||
            asset.controller.x < 0.0f ||
            asset.controller.y < 0.0f ||
            asset.controller.z < 0.0f ||
            !(asset.controller.w > 0.0f) ||
            asset.identity.w > MR_WORLD_DYNAMICS_SOFT_VOLUME) {
            return fail(reason, "sampled asset instance is invalid");
        }
    }
    for (const MRWorldSensorInstanceGPU& sensor : sensors) {
        if (!finite(sensor.positionAndFocalScale) ||
            !(sensor.positionAndFocalScale.w > 0.0f) ||
            !unitQuaternion(sensor.orientation) ||
            !finite(sensor.intrinsics) ||
            !finite(sensor.distortion) ||
            !finite(sensor.noiseAndLatency) ||
            sensor.noiseAndLatency.x < 0.0f ||
            sensor.noiseAndLatency.y < 0.0f ||
            sensor.noiseAndLatency.z < 0.0f ||
            sensor.noiseAndLatency.z > 1.0f ||
            sensor.noiseAndLatency.w < 0.0f ||
            sensor.identity.y > MR_WORLD_SENSOR_FORCE_TORQUE) {
            return fail(reason, "sampled sensor instance is invalid");
        }
    }
    for (const MRWorldAppearanceInstanceGPU& appearance : appearances) {
        if (!finite(appearance.colorAndLight) ||
            !(appearance.colorAndLight.y > 0.0f) ||
            appearance.colorAndLight.z < 0.0f ||
            appearance.colorAndLight.w < 0.0f ||
            !finite(appearance.material) ||
            appearance.material.y < 0.0f ||
            appearance.material.z < 0.0f ||
            appearance.material.w < 0.0f) {
            return fail(reason, "sampled appearance instance is invalid");
        }
    }
    return true;
}

std::uint64_t worldCompilerFingerprint(
    const EpisodeTwin& episode,
    const EngineModel& engineModel
) {
    HashBuilder hash;
    hash.appendScalar<std::uint32_t>(MR_WORLD_COMPILER_ABI_VERSION);
    hash.appendString(episode.id);
    hash.appendString(episode.coordinateConvention);
    const std::uint64_t assetCount = episode.assets.size();
    const std::uint64_t sensorCount = episode.sensors.size();
    const std::uint64_t artifactCount = episode.artifacts.size();
    hash.appendScalar(assetCount);
    for (const WorldAsset& asset : episode.assets) {
        appendAsset(hash, asset);
    }
    hash.appendScalar(sensorCount);
    for (const SensorSpec& sensor : episode.sensors) {
        appendSensor(hash, sensor);
    }
    hash.appendScalar(artifactCount);
    for (const EpisodeArtifact& artifact : episode.artifacts) {
        appendArtifact(hash, artifact);
    }
    appendTask(hash, episode.task);
    appendEngineModel(hash, engineModel);
    return hash.finish();
}

WorldCompileResult compileEpisodeTwin(
    const EpisodeTwin& episode,
    const EngineModel& engineModel,
    WorldTemplate& output
) {
    std::string engineReason;
    if (!engineModel.valid(&engineReason)) {
        return compileFail(
            WorldCompileStatus::invalidEngineModel,
            std::move(engineReason)
        );
    }
    if (episode.id.empty() || episode.coordinateConvention.empty() ||
        episode.assets.empty()) {
        return compileFail(
            WorldCompileStatus::invalidEpisodeTwin,
            "episode twin identity, coordinate convention, or assets are empty"
        );
    }
    std::unordered_set<std::string> artifactIds;
    for (const EpisodeArtifact& artifact : episode.artifacts) {
        if (artifact.id.empty() ||
            !artifactIds.insert(artifact.id).second ||
            !finite(artifact.startTimeSeconds) ||
            !finite(artifact.endTimeSeconds) ||
            artifact.endTimeSeconds < artifact.startTimeSeconds ||
            artifact.kind > EpisodeArtifactKind::replay ||
            artifact.producer > EpisodeArtifactProducer::authored) {
            return compileFail(
                WorldCompileStatus::invalidEpisodeTwin,
                "episode twin contains an invalid artifact"
            );
        }
    }

    WorldTemplate candidate;
    candidate.id = episode.id;
    candidate.engineModel = engineModel;
    candidate.assets = episode.assets;
    candidate.sensors = episode.sensors;
    candidate.artifacts = episode.artifacts;
    candidate.task = episode.task;
    candidate.appearances.push_back({});
    for (const WorldAsset& asset : candidate.assets) {
        if (std::ranges::find(
                candidate.topologyCohorts,
                asset.topologyCohort
            ) == candidate.topologyCohorts.end()) {
            candidate.topologyCohorts.push_back(asset.topologyCohort);
        }
    }
    for (std::uint32_t assetIndex = 0u;
         assetIndex < candidate.assets.size();
         ++assetIndex) {
        const WorldAsset& asset = candidate.assets[assetIndex];
        const auto cohort = std::ranges::find(
            candidate.topologyCohorts,
            asset.topologyCohort
        );
        const auto appendIndices = [&candidate](
            const std::vector<std::uint32_t>& values
        ) {
            const std::uint32_t offset =
                static_cast<std::uint32_t>(
                    candidate.bindingIndices.size()
                );
            candidate.bindingIndices.insert(
                candidate.bindingIndices.end(),
                values.begin(),
                values.end()
            );
            return std::pair{
                offset,
                static_cast<std::uint32_t>(values.size()),
            };
        };
        const auto bodies = appendIndices(asset.bodyIndices);
        const auto shapes = appendIndices(asset.shapeIndices);
        const auto materials = appendIndices(asset.materialIndices);
        candidate.assetBindings.push_back({
            {
                assetIndex,
                asset.role,
                asset.render,
                asset.collision,
            },
            {
                asset.dynamics,
                asset.articulationIndex,
                static_cast<std::uint32_t>(
                    cohort - candidate.topologyCohorts.begin()
                ),
                0u,
            },
            {bodies.first, bodies.second, shapes.first, shapes.second},
            {
                materials.first,
                materials.second,
                asset.renderAlternative,
                asset.collisionAlternative,
            },
        });
    }
    candidate.capabilities =
        computeCapabilities(candidate.assets, candidate.sensors);
    candidate.fingerprint =
        worldCompilerFingerprint(episode, engineModel);

    std::string reason;
    if (!candidate.valid(&reason)) {
        return compileFail(
            WorldCompileStatus::invalidWorldTemplate,
            std::move(reason)
        );
    }
    output = std::move(candidate);
    return {};
}

WorldCompileResult compileWorldFamily(
    const WorldTemplate& worldTemplate,
    const WorldProgram& program,
    WorldFamily& output
) {
    std::string reason;
    if (!worldTemplate.valid(&reason)) {
        return compileFail(
            WorldCompileStatus::invalidWorldTemplate,
            std::move(reason)
        );
    }
    if (program.id.empty()) {
        return compileFail(
            WorldCompileStatus::invalidWorldProgram,
            "world program id is empty"
        );
    }

    CompiledWorldProgram compiled;
    compiled.id = program.id;
    compiled.instanceFlags = program.instanceFlags;
    compiled.variations.reserve(program.variations.size());
    std::unordered_set<std::string> variationIds;
    HashBuilder hash;
    hash.appendString(program.id);
    hash.appendScalar(program.instanceFlags);
    hash.appendScalar(worldTemplate.fingerprint);

    for (std::uint32_t ordinal = 0u;
         ordinal < program.variations.size();
         ++ordinal) {
        const VariationParameter& variation = program.variations[ordinal];
        if (!variationIds.insert(variation.id).second ||
            !validDistribution(variation, &reason)) {
            return compileFail(
                WorldCompileStatus::invalidWorldProgram,
                reason.empty()
                    ? "world program has duplicate variation ids"
                    : std::move(reason)
            );
        }

        std::uint32_t targetIndex = MR_INVALID_INDEX;
        if (assetTarget(variation.target)) {
            targetIndex = worldTemplate.assetIndex(variation.targetId);
        } else if (sensorTarget(variation.target)) {
            targetIndex = worldTemplate.sensorIndex(variation.targetId);
        } else if (appearanceTarget(variation.target)) {
            if (variation.targetId.empty() &&
                worldTemplate.appearances.size() == 1u) {
                targetIndex = 0u;
            } else {
                for (std::uint32_t index = 0u;
                     index < worldTemplate.appearances.size();
                     ++index) {
                    if (worldTemplate.appearances[index].id ==
                        variation.targetId) {
                        targetIndex = index;
                        break;
                    }
                }
            }
        }
        if (targetIndex == MR_INVALID_INDEX) {
            return compileFail(
                WorldCompileStatus::invalidWorldProgram,
                "variation target does not exist: " + variation.targetId
            );
        }
        if ((variation.target == MR_WORLD_TARGET_ASSET_RENDER_ALTERNATIVE ||
             variation.target == MR_WORLD_TARGET_CLUTTER_SET) &&
            variation.distribution != MR_WORLD_DISTRIBUTION_CATEGORICAL) {
            return compileFail(
                WorldCompileStatus::invalidWorldProgram,
                "asset-alternative variation must be categorical"
            );
        }
        if (compiled.categoricalValues.size() >
                std::numeric_limits<std::uint32_t>::max() ||
            variation.categoricalValues.size() >
                std::numeric_limits<std::uint32_t>::max() -
                compiled.categoricalValues.size()) {
            return compileFail(
                WorldCompileStatus::capacityOverflow,
                "categorical value arena exceeds 32-bit GPU addressing"
            );
        }

        const std::uint32_t categoricalOffset =
            static_cast<std::uint32_t>(compiled.categoricalValues.size());
        compiled.categoricalValues.insert(
            compiled.categoricalValues.end(),
            variation.categoricalValues.begin(),
            variation.categoricalValues.end()
        );
        std::uint64_t salt = variation.salt;
        if (salt == 0u) {
            HashBuilder saltHash;
            saltHash.appendString(variation.id);
            salt = saltHash.finish();
        }
        MRWorldVariationGPU descriptor{};
        descriptor.binding = {
            variation.axis,
            variation.distribution,
            variation.target,
            targetIndex,
        };
        descriptor.parameters = variation.parameters;
        descriptor.categorical = {
            categoricalOffset,
            static_cast<std::uint32_t>(
                variation.categoricalValues.size()
            ),
            ordinal,
            0u,
        };
        descriptor.random = {
            variation.stream,
            static_cast<std::uint32_t>(salt),
            static_cast<std::uint32_t>(salt >> 32u),
            0u,
        };
        compiled.variations.push_back(descriptor);
        hash.appendString(variation.id);
        hash.appendScalar(descriptor);
        hash.appendSpan<std::uint32_t>(variation.categoricalValues);
    }
    compiled.fingerprint = hash.finish();

    WorldFamily candidate;
    candidate.worldTemplate = worldTemplate;
    candidate.program = std::move(compiled);
    HashBuilder familyHash;
    familyHash.appendScalar(candidate.worldTemplate.fingerprint);
    familyHash.appendScalar(candidate.program.fingerprint);
    candidate.fingerprint = familyHash.finish();
    output = std::move(candidate);
    return {};
}

WorldInstanceBatch WorldFamily::sample(
    const std::uint32_t instanceCount,
    const std::uint64_t seed
) const {
    WorldInstanceBatch batch;
    if (fingerprint == 0u || instanceCount == 0u) {
        return batch;
    }
    const std::size_t assetCount = worldTemplate.assets.size();
    const std::size_t sensorCount = worldTemplate.sensors.size();
    const std::size_t appearanceCount = worldTemplate.appearances.size();
    const std::size_t max = std::numeric_limits<std::uint32_t>::max();
    if (assetCount > max || sensorCount > max || appearanceCount > max ||
        instanceCount > max / std::max<std::size_t>(assetCount, 1u) ||
        instanceCount > max / std::max<std::size_t>(sensorCount, 1u) ||
        instanceCount > max / std::max<std::size_t>(appearanceCount, 1u)) {
        return batch;
    }

    batch.familyFingerprint = fingerprint;
    batch.instances.resize(instanceCount);
    batch.assets.resize(static_cast<std::size_t>(instanceCount) * assetCount);
    batch.sensors.resize(
        static_cast<std::size_t>(instanceCount) * sensorCount
    );
    batch.appearances.resize(
        static_cast<std::size_t>(instanceCount) * appearanceCount
    );

    for (std::uint32_t environment = 0u;
         environment < instanceCount;
         ++environment) {
        const std::uint32_t firstAsset =
            static_cast<std::uint32_t>(environment * assetCount);
        const std::uint32_t firstSensor =
            static_cast<std::uint32_t>(environment * sensorCount);
        const std::uint32_t firstAppearance =
            static_cast<std::uint32_t>(environment * appearanceCount);
        const std::uint64_t scenarioKey =
            fingerprint ^ seed ^
            (0x9e3779b97f4a7c15ull * (environment + 1ull));
        batch.instances[environment] = {
            {
                firstAsset,
                static_cast<std::uint32_t>(assetCount),
                firstSensor,
                static_cast<std::uint32_t>(sensorCount),
            },
            {
                firstAppearance,
                static_cast<std::uint32_t>(appearanceCount),
                0u,
                0u,
            },
            {
                static_cast<std::uint32_t>(scenarioKey),
                static_cast<std::uint32_t>(scenarioKey >> 32u),
                program.instanceFlags,
                MR_WORLD_COMPILER_ABI_VERSION,
            },
        };

        for (std::uint32_t index = 0u; index < assetCount; ++index) {
            const WorldAsset& source = worldTemplate.assets[index];
            batch.assets[firstAsset + index] = {
                {
                    source.initialPose.position.x,
                    source.initialPose.position.y,
                    source.initialPose.position.z,
                    source.uniformScale,
                },
                source.initialPose.orientation,
                {0.0f, 0.0f, 0.0f, 0.0f},
                {0.0f, 0.0f, 0.0f, 0.0f},
                {
                    source.massScale,
                    source.frictionScale,
                    source.restitutionScale,
                    source.dampingScale,
                },
                {
                    source.controllerGainScale,
                    source.controllerDampingScale,
                    source.controllerLatencySeconds,
                    source.payloadScale,
                },
                {
                    index,
                    source.renderAlternative,
                    source.collisionAlternative,
                    source.dynamics,
                },
            };
        }
        for (std::uint32_t index = 0u; index < sensorCount; ++index) {
            const SensorSpec& source = worldTemplate.sensors[index];
            batch.sensors[firstSensor + index] = {
                {
                    source.localPose.position.x,
                    source.localPose.position.y,
                    source.localPose.position.z,
                    source.focalScale,
                },
                source.localPose.orientation,
                source.intrinsics,
                source.distortion,
                {
                    source.colorNoiseSigma,
                    source.depthNoiseSigma,
                    source.depthDropout,
                    source.latencySeconds,
                },
                {
                    worldTemplate.assetIndex(source.parentAssetId),
                    source.kind,
                    source.width,
                    source.height,
                },
            };
        }
        for (std::uint32_t index = 0u; index < appearanceCount; ++index) {
            const AppearanceSpec& source = worldTemplate.appearances[index];
            batch.appearances[firstAppearance + index] = {
                {
                    source.exposureStops,
                    source.whiteBalanceScale,
                    source.saturation,
                    source.lightIntensity,
                },
                {
                    source.hueRadians,
                    source.contrast,
                    source.roughnessScale,
                    source.metallicScale,
                },
                {
                    source.alternative,
                    source.environmentMap,
                    0u,
                    0u,
                },
            };
        }

        for (std::uint32_t variationIndex = 0u;
             variationIndex < program.variations.size();
             ++variationIndex) {
            const MRWorldVariationGPU& variation =
                program.variations[variationIndex];
            const SampledValue value = sampleValue(
                variation,
                program.categoricalValues,
                philox(seed, environment, variationIndex, variation)
            );
            const std::uint32_t targetIndex = variation.binding.w;
            const auto target =
                static_cast<MRWorldVariationTarget>(variation.binding.z);
            if (assetTarget(target) && targetIndex < assetCount) {
                applyAssetVariation(
                    batch.assets[firstAsset + targetIndex],
                    target,
                    value
                );
            } else if (sensorTarget(target) &&
                       targetIndex < sensorCount) {
                applySensorVariation(
                    batch.sensors[firstSensor + targetIndex],
                    target,
                    value.scalar
                );
            } else if (appearanceTarget(target) &&
                       targetIndex < appearanceCount) {
                applyAppearanceVariation(
                    batch.appearances[firstAppearance + targetIndex],
                    target,
                    value.scalar
                );
            }
        }
    }
    return batch;
}

} // namespace metalrobo
