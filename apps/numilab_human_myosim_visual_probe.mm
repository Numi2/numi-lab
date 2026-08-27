#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>
#import <Metal/Metal.h>

#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/MetalHybridRenderer.hpp"
#include "metalrobo/VisualPlatform.hpp"
#include "metalrobo/WorldCompiler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr std::array<char, 8u> kRigidMagic{
    'N', 'H', 'R', 'I', 'G', 'I', 'D', '2',
};
constexpr std::array<char, 8u> kMuscleMagic{
    'N', 'H', 'M', 'Y', 'O', '1', '\0', '\0',
};
constexpr std::uint32_t kPayloadAbi = 1u;
constexpr std::uint32_t kBodySemantic = 51001u;
constexpr std::uint32_t kSiteSemantic = 51002u;
constexpr std::uint32_t kRouteSemantic = 51003u;
constexpr std::uint32_t kFrameWidth = 640u;
constexpr std::uint32_t kFrameHeight = 640u;

#pragma pack(push, 1)
struct RigidHeader {
    std::array<char, 8u> magic{};
    std::uint32_t payloadAbi = 0u;
    std::uint32_t engineAbi = 0u;
    std::uint32_t sourceBodyCount = 0u;
    std::uint32_t engineBodyCount = 0u;
    std::uint32_t jointCount = 0u;
    std::uint32_t nq = 0u;
    std::uint32_t nv = 0u;
    std::uint32_t rootBodyIndex = 0u;
    std::uint32_t virtualBodyCount = 0u;
    std::uint32_t reserved0 = 0u;
    std::array<std::uint8_t, 32u> sourceSha256{};
};

struct SourcePoseRecord {
    float positionX = 0.0f;
    float positionY = 0.0f;
    float positionZ = 0.0f;
    float quaternionX = 0.0f;
    float quaternionY = 0.0f;
    float quaternionZ = 0.0f;
    float quaternionW = 1.0f;
};

struct MuscleHeader {
    std::array<char, 8u> magic{};
    std::uint32_t payloadAbi = 0u;
    std::uint32_t engineBodyCount = 0u;
    std::uint32_t muscleCount = 0u;
    std::uint32_t siteCount = 0u;
    std::uint32_t wrapCount = 0u;
    std::uint32_t routeNodeCount = 0u;
    std::uint32_t sourceTendonCount = 0u;
    std::uint32_t reserved0 = 0u;
    std::uint32_t reserved1 = 0u;
    std::array<std::uint8_t, 32u> sourceSha256{};
};

struct SiteRecord {
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct WrapRecord {
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    std::uint32_t type = 0u;
    float radius = 0.0f;
    float reserved0 = 0.0f;
    float centerX = 0.0f;
    float centerY = 0.0f;
    float centerZ = 0.0f;
    float rotation[9]{};
};

struct RouteRecord {
    std::uint32_t type = 0u;
    std::uint32_t targetIndex = MR_INVALID_INDEX;
    std::uint32_t sideSiteIndex = MR_INVALID_INDEX;
    std::uint32_t reserved0 = 0u;
};

struct MuscleRecord {
    std::uint32_t sourceTendonIndex = 0u;
    std::uint32_t routeOffset = 0u;
    std::uint32_t routeCount = 0u;
    std::uint32_t reserved0 = 0u;
    float values[37]{};
};

struct LoadedMuscles {
    MuscleHeader header{};
    std::vector<SiteRecord> sites;
    std::vector<WrapRecord> wraps;
    std::vector<RouteRecord> routes;
    std::vector<MuscleRecord> muscles;
};
#pragma pack(pop)

static_assert(sizeof(RigidHeader) == 80u);
static_assert(sizeof(SourcePoseRecord) == 28u);
static_assert(sizeof(MuscleHeader) == 76u);
static_assert(sizeof(SiteRecord) == 16u);
static_assert(sizeof(WrapRecord) == 64u);
static_assert(sizeof(RouteRecord) == 16u);
static_assert(sizeof(MuscleRecord) == 164u);

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
void readObject(std::istream& input, T& value, const char* description) {
    static_assert(std::is_trivially_copyable_v<T>);
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    require(input.good(), std::string("truncated ") + description);
}

template <typename T>
std::vector<T> readVector(
    std::istream& input,
    const std::size_t count,
    const char* description
) {
    std::vector<T> result(count);
    if (!result.empty()) {
        input.read(
            reinterpret_cast<char*>(result.data()),
            static_cast<std::streamsize>(result.size() * sizeof(T))
        );
        require(input.good(), std::string("truncated ") + description);
    }
    return result;
}

struct LoadedRigid {
    RigidHeader header{};
    metalrobo::EngineModel model;
};

LoadedRigid loadRigid(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "cannot open MyoSim rigid payload " + path.string());
    LoadedRigid result;
    readObject(input, result.header, "MyoSim rigid header");
    require(result.header.magic == kRigidMagic, "rigid payload magic is not NHRIGID2");
    require(result.header.payloadAbi == kPayloadAbi, "unsupported MyoSim rigid ABI");
    require(result.header.engineAbi == MR_ENGINE_ABI_VERSION, "MyoSim rigid/Core ABI mismatch");
    require(result.header.reserved0 == 0u && result.header.rootBodyIndex == 0u,
            "MyoSim rigid header is malformed");
    require(result.header.sourceBodyCount > 0u &&
                result.header.engineBodyCount >= result.header.sourceBodyCount &&
                result.header.jointCount + 1u == result.header.engineBodyCount &&
                result.header.nq == result.header.nv + 1u,
            "MyoSim rigid dimensions are malformed");
    result.model.name = "numilab_human_myosim_native_visual";
    readObject(input, result.model.world, "MyoSim world");
    MRArticulationGPU articulation{};
    readObject(input, articulation, "MyoSim articulation");
    result.model.articulations.push_back(articulation);
    result.model.bodies = readVector<MRBodyPropertiesGPU>(
        input, result.header.engineBodyCount, "MyoSim bodies"
    );
    result.model.joints = readVector<MRJointDescriptorGPU>(
        input, result.header.jointCount, "MyoSim joints"
    );
    result.model.dofs = readVector<MRDofPropertiesGPU>(
        input, result.header.nv, "MyoSim DoFs"
    );
    result.model.defaultQ = readVector<float>(
        input, result.header.nq, "MyoSim default q"
    );
    result.model.defaultV = readVector<float>(
        input, result.header.nv, "MyoSim default v"
    );
    const auto sourceToCore = readVector<std::uint32_t>(
        input, result.header.sourceBodyCount, "MyoSim source map"
    );
    (void)sourceToCore;
    const auto sourcePoses = readVector<SourcePoseRecord>(
        input, result.header.sourceBodyCount, "MyoSim source poses"
    );
    (void)sourcePoses;
    require(input.peek() == std::char_traits<char>::eof(),
            "MyoSim rigid payload has trailing bytes");
    require(result.model.world.bodyCount == result.header.engineBodyCount &&
                articulation.rootType == MR_ROOT_FLOATING &&
                articulation.bodyCount == result.header.engineBodyCount &&
                articulation.nq == result.header.nq && articulation.nv == result.header.nv,
            "MyoSim rigid world/header disagreement");
    std::string reason;
    require(result.model.valid(&reason), "MyoSim Core model invalid: " + reason);
    return result;
}

LoadedMuscles loadMuscles(
    const std::filesystem::path& path,
    const RigidHeader& rigid
) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "cannot open MyoSim muscle payload " + path.string());
    LoadedMuscles result;
    readObject(input, result.header, "MyoSim muscle header");
    require(result.header.magic == kMuscleMagic && result.header.payloadAbi == kPayloadAbi &&
                result.header.engineBodyCount == rigid.engineBodyCount &&
                result.header.sourceSha256 == rigid.sourceSha256 &&
                result.header.reserved0 == 0u && result.header.reserved1 == 0u,
            "MyoSim muscle payload/header disagreement");
    result.sites = readVector<SiteRecord>(
        input, result.header.siteCount, "MyoSim sites"
    );
    result.wraps = readVector<WrapRecord>(input, result.header.wrapCount, "MyoSim wraps");
    result.routes = readVector<RouteRecord>(input, result.header.routeNodeCount, "MyoSim routes");
    result.muscles = readVector<MuscleRecord>(input, result.header.muscleCount, "MyoSim muscles");
    require(input.peek() == std::char_traits<char>::eof(),
            "MyoSim muscle payload has trailing bytes");
    for (const SiteRecord& site : result.sites) {
        require(site.bodyIndex < rigid.engineBodyCount,
                "MyoSim site body index is out of bounds");
    }
    for (const WrapRecord& wrap : result.wraps) {
        require(wrap.bodyIndex < rigid.engineBodyCount,
                "MyoSim wrap body index is out of bounds");
    }
    for (const RouteRecord& route : result.routes) {
        require(route.reserved0 == 0u && route.type >= 1u && route.type <= 3u,
                "MyoSim route record is malformed");
        const std::size_t targetCount = route.type == 1u
            ? result.sites.size() : result.wraps.size();
        require(route.targetIndex < targetCount,
                "MyoSim route target is out of bounds");
        require(route.sideSiteIndex == MR_INVALID_INDEX ||
                    route.sideSiteIndex < result.sites.size(),
                "MyoSim route side site is out of bounds");
    }
    for (const MuscleRecord& muscle : result.muscles) {
        require(muscle.reserved0 == 0u &&
                    muscle.routeOffset <= result.routes.size() &&
                    muscle.routeCount <= result.routes.size() - muscle.routeOffset,
                "MyoSim muscle route range is invalid");
    }
    return result;
}

mr_float4 normalizeQuaternion(const mr_float4 value) {
    const float length = std::sqrt(
        value.x * value.x + value.y * value.y +
        value.z * value.z + value.w * value.w
    );
    require(std::isfinite(length) && length > 1.0e-6f,
            "Metal articulated pose has a non-normalizable orientation");
    return {value.x / length, value.y / length, value.z / length, value.w / length};
}

std::vector<MRBodyStateGPU> visualBodyStates(
    const metalrobo::EngineModel& model,
    const std::span<const MRArticulatedBodyPoseGPU> poses
) {
    require(poses.size() == model.bodies.size(), "Metal body-pose stream has an invalid size");
    std::vector<MRBodyStateGPU> result(poses.size());
    for (std::size_t index = 0u; index < poses.size(); ++index) {
        MRBodyStateGPU& state = result[index];
        state.position = poses[index].position;
        state.orientation = normalizeQuaternion(poses[index].orientation);
        state.linearVelocityAndInverseMass.w = model.bodies[index].massAndInverseMass.y;
        state.flagsAndIndices[0] = model.bodies[index].motionType;
        state.flagsAndIndices[1] = model.bodies[index].articulationIndex;
        state.flagsAndIndices[2] = static_cast<mr_u32>(index);
        state.flagsAndIndices[3] = 0u;
    }
    return result;
}

mr_float4 rotateDirection(const mr_float4 orientation, const mr_float4 direction) {
    const mr_float4 q{orientation.x, orientation.y, orientation.z, 0.0f};
    const auto cross = [](const mr_float4 left, const mr_float4 right) {
        return mr_float4{
            left.y * right.z - left.z * right.y,
            left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x,
            0.0f,
        };
    };
    const mr_float4 first = cross(q, direction);
    const mr_float4 second = cross(q, first);
    return {
        direction.x + 2.0f * (orientation.w * first.x + second.x),
        direction.y + 2.0f * (orientation.w * first.y + second.y),
        direction.z + 2.0f * (orientation.w * first.z + second.z),
        0.0f,
    };
}

mr_float4 worldPoint(
    const std::span<const MRBodyStateGPU> bodies,
    const std::uint32_t bodyIndex,
    const mr_float4 localPoint
) {
    require(bodyIndex < bodies.size(), "MyoSim visual point body index is out of bounds");
    const MRBodyStateGPU& body = bodies[bodyIndex];
    const mr_float4 offset = rotateDirection(body.orientation, localPoint);
    return {
        body.position.x + offset.x,
        body.position.y + offset.y,
        body.position.z + offset.z,
        1.0f,
    };
}

metalrobo::WorldPose cameraToward(
    const mr_float4 position,
    const mr_float4 target
) {
    const mr_float4 forward{
        target.x - position.x, target.y - position.y, target.z - position.z, 0.0f,
    };
    const float forwardLength = std::sqrt(
        forward.x * forward.x + forward.y * forward.y + forward.z * forward.z
    );
    require(forwardLength > 1.0e-5f, "native Human camera has no target direction");
    const mr_float4 unitForward{
        forward.x / forwardLength, forward.y / forwardLength, forward.z / forwardLength, 0.0f,
    };
    mr_float4 localY{0.0f, 0.0f, -1.0f, 0.0f};
    mr_float4 localX{
        localY.y * unitForward.z - localY.z * unitForward.y,
        localY.z * unitForward.x - localY.x * unitForward.z,
        localY.x * unitForward.y - localY.y * unitForward.x,
        0.0f,
    };
    const float localXLength = std::sqrt(
        localX.x * localX.x + localX.y * localX.y + localX.z * localX.z
    );
    require(localXLength > 1.0e-5f, "native Human camera is parallel to world up");
    localX.x /= localXLength;
    localX.y /= localXLength;
    localX.z /= localXLength;
    localY = {
        unitForward.y * localX.z - unitForward.z * localX.y,
        unitForward.z * localX.x - unitForward.x * localX.z,
        unitForward.x * localX.y - unitForward.y * localX.x,
        0.0f,
    };
    const float m00 = localX.x;
    const float m01 = localY.x;
    const float m02 = unitForward.x;
    const float m10 = localX.y;
    const float m11 = localY.y;
    const float m12 = unitForward.y;
    const float m20 = localX.z;
    const float m21 = localY.z;
    const float m22 = unitForward.z;
    mr_float4 orientation{};
    const float trace = m00 + m11 + m22;
    if (trace > 0.0f) {
        const float scale = 2.0f * std::sqrt(trace + 1.0f);
        orientation = {(m21 - m12) / scale, (m02 - m20) / scale,
                       (m10 - m01) / scale, 0.25f * scale};
    } else if (m00 > m11 && m00 > m22) {
        const float scale = 2.0f * std::sqrt(1.0f + m00 - m11 - m22);
        orientation = {0.25f * scale, (m01 + m10) / scale,
                       (m02 + m20) / scale, (m21 - m12) / scale};
    } else if (m11 > m22) {
        const float scale = 2.0f * std::sqrt(1.0f + m11 - m00 - m22);
        orientation = {(m01 + m10) / scale, 0.25f * scale,
                       (m12 + m21) / scale, (m02 - m20) / scale};
    } else {
        const float scale = 2.0f * std::sqrt(1.0f + m22 - m00 - m11);
        orientation = {(m02 + m20) / scale, (m12 + m21) / scale,
                       0.25f * scale, (m10 - m01) / scale};
    }
    return {position, normalizeQuaternion(orientation)};
}

metalrobo::SensorSpec makeCamera(
    const std::string& id,
    const mr_float4 position,
    const mr_float4 target
) {
    metalrobo::SensorSpec camera;
    camera.id = id;
    camera.parentAssetId = "myosim_human";
    camera.parentKind = MR_WORLD_SENSOR_PARENT_ASSET;
    camera.kind = MR_WORLD_SENSOR_RGBD;
    camera.localPose = cameraToward(position, target);
    camera.width = kFrameWidth;
    camera.height = kFrameHeight;
    camera.intrinsics = {470.0f, 470.0f, 0.5f * kFrameWidth, 0.5f * kFrameHeight};
    camera.maximumDepthMeters = 20.0f;
    return camera;
}

std::pair<mr_float4, float> frameBounds(
    const metalrobo::EngineModel& model,
    const std::span<const MRBodyStateGPU> bodies
) {
    require(model.bodies.size() == bodies.size(), "MyoSim visual body bounds size mismatch");
    mr_float4 minimum{
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), 0.0f,
    };
    mr_float4 maximum{
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), 0.0f,
    };
    std::size_t includedBodyCount = 0u;
    for (std::size_t index = 0u; index < bodies.size(); ++index) {
        if (!(model.bodies[index].massAndInverseMass.x > 1.0e-5f)) {
            continue;
        }
        const MRBodyStateGPU& body = bodies[index];
        minimum.x = std::min(minimum.x, body.position.x);
        minimum.y = std::min(minimum.y, body.position.y);
        minimum.z = std::min(minimum.z, body.position.z);
        maximum.x = std::max(maximum.x, body.position.x);
        maximum.y = std::max(maximum.y, body.position.y);
        maximum.z = std::max(maximum.z, body.position.z);
        ++includedBodyCount;
    }
    require(includedBodyCount > 0u, "MyoSim visual body bounds have no inertial body");
    const mr_float4 center{
        0.5f * (minimum.x + maximum.x),
        0.5f * (minimum.y + maximum.y),
        0.5f * (minimum.z + maximum.z), 0.0f,
    };
    const float extent = std::max({
        maximum.x - minimum.x, maximum.y - minimum.y, maximum.z - minimum.z,
    });
    return {center, std::max(1.65f * extent, 2.5f)};
}

MRVisualMaterialGPUV2 makeMaterial(
    const mr_float4 color,
    const mr_float4 emission
) {
    MRVisualMaterialGPUV2 material{};
    material.baseColorAndOpacity = color;
    material.emissionAndStrength = emission;
    material.surface = {0.55f, 0.05f, 1.0f, 1.0f};
    material.coatingAndAlphaCutoff = {0.0f, 0.0f, 1.0f, 0.5f};
    material.textureIndices0 = {
        MR_INVALID_INDEX, MR_INVALID_INDEX, MR_INVALID_INDEX, MR_INVALID_INDEX,
    };
    material.textureIndices1 = {
        MR_INVALID_INDEX, MR_INVALID_INDEX, MR_INVALID_INDEX, MR_INVALID_INDEX,
    };
    material.flags = {
        MR_VISUAL_ALPHA_OPAQUE, MR_VISUAL_MATERIAL_DOUBLE_SIDED, 0u, 1u,
    };
    material.reserved = {
        MR_INVALID_INDEX, MR_INVALID_INDEX, MR_INVALID_INDEX, MR_INVALID_INDEX,
    };
    return material;
}

struct GeometryRange {
    std::uint32_t firstIndex = 0u;
    std::uint32_t indexCount = 0u;
    mr_float4 minimum{};
    mr_float4 maximum{};
};

GeometryRange appendEllipsoid(
    metalrobo::VisualAssetPackV2& pack,
    const std::array<float, 3u> radii
) {
    constexpr std::uint32_t kLatitude = 8u;
    constexpr std::uint32_t kLongitude = 12u;
    GeometryRange result;
    result.firstIndex = static_cast<std::uint32_t>(pack.indices.size());
    const std::uint32_t vertexBase = static_cast<std::uint32_t>(pack.vertices.size());
    for (std::uint32_t latitude = 0u; latitude <= kLatitude; ++latitude) {
        const float phi = static_cast<float>(latitude) *
            std::numbers::pi_v<float> / static_cast<float>(kLatitude);
        const float sinPhi = std::sin(phi);
        const float cosPhi = std::cos(phi);
        for (std::uint32_t longitude = 0u; longitude <= kLongitude; ++longitude) {
            const float theta = static_cast<float>(longitude) *
                2.0f * std::numbers::pi_v<float> / static_cast<float>(kLongitude);
            const float cosTheta = std::cos(theta);
            const float sinTheta = std::sin(theta);
            const mr_float4 normal{sinPhi * cosTheta, sinPhi * sinTheta, cosPhi, 1.0f};
            mr_float4 tangent{-sinTheta, cosTheta, 0.0f, 0.0f};
            if (latitude == 0u || latitude == kLatitude) {
                tangent = {1.0f, 0.0f, 0.0f, 0.0f};
            }
            pack.vertices.push_back({
                {radii[0] * normal.x, radii[1] * normal.y, radii[2] * normal.z, 1.0f},
                normal,
                tangent,
                {static_cast<float>(longitude) / static_cast<float>(kLongitude),
                 static_cast<float>(latitude) / static_cast<float>(kLatitude), 0.0f, 0.0f},
                {1.0f, 1.0f, 1.0f, 1.0f},
            });
        }
    }
    const std::uint32_t row = kLongitude + 1u;
    for (std::uint32_t latitude = 0u; latitude < kLatitude; ++latitude) {
        for (std::uint32_t longitude = 0u; longitude < kLongitude; ++longitude) {
            const std::uint32_t first = vertexBase + latitude * row + longitude;
            pack.indices.insert(pack.indices.end(), {
                first, first + row, first + 1u,
                first + 1u, first + row, first + row + 1u,
            });
        }
    }
    result.indexCount = static_cast<std::uint32_t>(pack.indices.size()) - result.firstIndex;
    result.minimum = {-radii[0], -radii[1], -radii[2], 1.0f};
    result.maximum = {radii[0], radii[1], radii[2], 1.0f};
    return result;
}

std::array<float, 3u> inertiaEllipsoid(const MRBodyPropertiesGPU& body) {
    const float mass = body.massAndInverseMass.x;
    if (!(mass > 1.0e-5f)) {
        return {0.010f, 0.010f, 0.010f};
    }
    const float ixx = std::max(body.inertiaRow0.x, 1.0e-8f);
    const float iyy = std::max(body.inertiaRow1.y, 1.0e-8f);
    const float izz = std::max(body.inertiaRow2.z, 1.0e-8f);
    const auto semiAxis = [](const float squared) {
        return std::clamp(std::sqrt(std::max(squared, 1.0e-6f)), 0.014f, 0.115f);
    };
    return {
        semiAxis(2.5f * (iyy + izz - ixx) / mass),
        semiAxis(2.5f * (ixx + izz - iyy) / mass),
        semiAxis(2.5f * (ixx + iyy - izz) / mass),
    };
}

GeometryRange appendWorldTube(
    metalrobo::VisualAssetPackV2& pack,
    const mr_float4 start,
    const mr_float4 end,
    const float radius
) {
    constexpr std::uint32_t kSides = 6u;
    const mr_float4 axisRaw{
        end.x - start.x, end.y - start.y, end.z - start.z, 0.0f,
    };
    const float axisLength = std::sqrt(
        axisRaw.x * axisRaw.x + axisRaw.y * axisRaw.y + axisRaw.z * axisRaw.z
    );
    require(axisLength > 1.0e-5f && radius > 0.0f, "MyoSim route tube is degenerate");
    const mr_float4 axis{
        axisRaw.x / axisLength, axisRaw.y / axisLength, axisRaw.z / axisLength, 0.0f,
    };
    const auto cross = [](const mr_float4 left, const mr_float4 right) {
        return mr_float4{
            left.y * right.z - left.z * right.y,
            left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x,
            0.0f,
        };
    };
    const mr_float4 reference = std::abs(axis.z) < 0.85f
        ? mr_float4{0.0f, 0.0f, 1.0f, 0.0f}
        : mr_float4{0.0f, 1.0f, 0.0f, 0.0f};
    mr_float4 normal = cross(reference, axis);
    const float normalLength = std::sqrt(
        normal.x * normal.x + normal.y * normal.y + normal.z * normal.z
    );
    require(normalLength > 1.0e-5f, "MyoSim route tube has no normal basis");
    normal.x /= normalLength;
    normal.y /= normalLength;
    normal.z /= normalLength;
    const mr_float4 binormal = cross(axis, normal);
    GeometryRange result;
    result.firstIndex = static_cast<std::uint32_t>(pack.indices.size());
    result.minimum = {
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), 1.0f,
    };
    result.maximum = {
        -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), 1.0f,
    };
    const std::uint32_t vertexBase = static_cast<std::uint32_t>(pack.vertices.size());
    for (std::uint32_t ring = 0u; ring < 2u; ++ring) {
        const mr_float4 center = ring == 0u ? start : end;
        for (std::uint32_t side = 0u; side < kSides; ++side) {
            const float angle = 2.0f * std::numbers::pi_v<float> *
                static_cast<float>(side) / static_cast<float>(kSides);
            const mr_float4 surfaceNormal{
                normal.x * std::cos(angle) + binormal.x * std::sin(angle),
                normal.y * std::cos(angle) + binormal.y * std::sin(angle),
                normal.z * std::cos(angle) + binormal.z * std::sin(angle),
                1.0f,
            };
            const mr_float4 position{
                center.x + radius * surfaceNormal.x,
                center.y + radius * surfaceNormal.y,
                center.z + radius * surfaceNormal.z,
                1.0f,
            };
            pack.vertices.push_back({
                position, surfaceNormal, axis,
                {static_cast<float>(ring), static_cast<float>(side) / static_cast<float>(kSides), 0.0f, 0.0f},
                {1.0f, 1.0f, 1.0f, 1.0f},
            });
            result.minimum.x = std::min(result.minimum.x, position.x);
            result.minimum.y = std::min(result.minimum.y, position.y);
            result.minimum.z = std::min(result.minimum.z, position.z);
            result.maximum.x = std::max(result.maximum.x, position.x);
            result.maximum.y = std::max(result.maximum.y, position.y);
            result.maximum.z = std::max(result.maximum.z, position.z);
        }
    }
    for (std::uint32_t side = 0u; side < kSides; ++side) {
        const std::uint32_t next = (side + 1u) % kSides;
        const std::uint32_t a = vertexBase + side;
        const std::uint32_t b = vertexBase + next;
        const std::uint32_t c = vertexBase + kSides + side;
        const std::uint32_t d = vertexBase + kSides + next;
        pack.indices.insert(pack.indices.end(), {a, c, b, b, c, d});
    }
    result.indexCount = static_cast<std::uint32_t>(pack.indices.size()) - result.firstIndex;
    return result;
}

metalrobo::VisualAssetPackV2 makeMarkerPack(
    const metalrobo::EngineModel& model,
    const LoadedMuscles& musclePayload,
    const std::span<const MRBodyStateGPU> bodies,
    std::uint32_t& renderedBodies,
    std::uint32_t& renderedRouteSegments
) {
    metalrobo::VisualAssetPackV2 pack;
    pack.id = "myosim_fullbody_articulated_marker_view";
    pack.sourceUri = "numi://myosim/NHRIGID2+NHMYO1/articulated-marker-view";
    pack.sourceContentHash = "runtime-body-and-site-records";
    pack.license = "Apache-2.0";
    pack.preprocessingProvenance =
        "metal_articulated_operator_pose_snapshot/native_visual_marker_pack.v1";
    pack.materials.push_back(makeMaterial(
        {0.82f, 0.86f, 0.88f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}
    ));
    pack.materials.push_back(makeMaterial(
        {0.80f, 0.04f, 0.03f, 1.0f}, {0.15f, 0.0f, 0.0f, 0.25f}
    ));
    pack.materials.push_back(makeMaterial(
        {0.68f, 0.015f, 0.01f, 1.0f}, {0.35f, 0.0f, 0.0f, 0.45f}
    ));

    const GeometryRange siteGeometry = appendEllipsoid(
        pack, {0.0075f, 0.0075f, 0.0075f}
    );
    const auto appendInstance = [&pack](
        const GeometryRange& geometry,
        const std::uint32_t material,
        const std::uint32_t semantic,
        const std::uint32_t bindingKind,
        const std::uint32_t bodyIndex,
        const mr_float4 translation,
        const std::uint32_t stableId
    ) {
        const std::uint32_t instanceIndex = static_cast<std::uint32_t>(pack.instances.size());
        MRVisualInstanceGPUV2 instance{};
        instance.translationAndScale = translation;
        instance.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
        instance.binding = {
            0u, bodyIndex, bindingKind,
            MR_VISUAL_INSTANCE_CASTS_SHADOW |
                MR_VISUAL_INSTANCE_RECEIVES_SHADOW |
                MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR,
        };
        instance.identity = {semantic, stableId, bodyIndex, stableId};
        instance.geometry = {
            static_cast<std::uint32_t>(pack.primitives.size()), 1u, material, 0u,
        };
        pack.instances.push_back(instance);
        MRVisualPrimitiveGPUV2 primitive{};
        primitive.geometry = {geometry.firstIndex, geometry.indexCount, material, instanceIndex};
        primitive.identity = {semantic, stableId, bodyIndex, stableId};
        primitive.boundsMinimum = geometry.minimum;
        primitive.boundsMaximum = geometry.maximum;
        pack.primitives.push_back(primitive);
    };

    renderedBodies = 0u;
    for (std::size_t bodyIndex = 0u; bodyIndex < model.bodies.size(); ++bodyIndex) {
        const MRBodyPropertiesGPU& body = model.bodies[bodyIndex];
        if (!(body.massAndInverseMass.x > 1.0e-5f)) {
            continue;
        }
        const GeometryRange geometry = appendEllipsoid(pack, inertiaEllipsoid(body));
        appendInstance(
            geometry, 0u, kBodySemantic, MR_VISUAL_BINDING_ARTICULATED_LINK,
            static_cast<std::uint32_t>(bodyIndex),
            {0.0f, 0.0f, 0.0f, 1.0f}, static_cast<std::uint32_t>(bodyIndex + 1u)
        );
        ++renderedBodies;
    }
    for (std::size_t siteIndex = 0u; siteIndex < musclePayload.sites.size(); ++siteIndex) {
        const SiteRecord& site = musclePayload.sites[siteIndex];
        appendInstance(
            siteGeometry, 1u, kSiteSemantic, MR_VISUAL_BINDING_ARTICULATED_LINK,
            site.bodyIndex,
            {site.x, site.y, site.z, 1.0f}, static_cast<std::uint32_t>(siteIndex + 1u)
        );
    }
    const auto routePoint = [&musclePayload, bodies](const RouteRecord& route) {
        if (route.type == 1u) {
            const SiteRecord& site = musclePayload.sites[route.targetIndex];
            return worldPoint(bodies, site.bodyIndex, {site.x, site.y, site.z, 0.0f});
        }
        const WrapRecord& wrap = musclePayload.wraps[route.targetIndex];
        return worldPoint(
            bodies, wrap.bodyIndex,
            {wrap.centerX, wrap.centerY, wrap.centerZ, 0.0f}
        );
    };
    renderedRouteSegments = 0u;
    std::uint32_t stableRouteId = 1u;
    for (const MuscleRecord& muscle : musclePayload.muscles) {
        bool hasPrevious = false;
        mr_float4 previous{};
        for (std::uint32_t routeOffset = 0u; routeOffset < muscle.routeCount; ++routeOffset) {
            const mr_float4 current = routePoint(
                musclePayload.routes[muscle.routeOffset + routeOffset]
            );
            if (hasPrevious) {
                const float dx = current.x - previous.x;
                const float dy = current.y - previous.y;
                const float dz = current.z - previous.z;
                if (dx * dx + dy * dy + dz * dz > 1.0e-10f) {
                    appendInstance(
                        appendWorldTube(pack, previous, current, 0.0032f), 2u,
                        kRouteSemantic, MR_VISUAL_BINDING_WORLD, MR_INVALID_INDEX,
                        {0.0f, 0.0f, 0.0f, 1.0f}, stableRouteId++
                    );
                    ++renderedRouteSegments;
                }
            }
            previous = current;
            hasPrevious = true;
        }
    }
    pack.contentHash = metalrobo::computeVisualAssetPackContentHash(pack);
    std::string reason;
    require(pack.valid(&reason), "native Human marker pack is invalid: " + reason);
    return pack;
}

metalrobo::WorldTemplate makeWorld(
    const metalrobo::EngineModel& model,
    const std::span<const MRBodyStateGPU> bodies,
    std::array<std::string, 3u>& cameraNames
) {
    const auto [center, distance] = frameBounds(model, bodies);
    cameraNames = {"front", "side", "rear"};
    metalrobo::EpisodeTwin episode;
    episode.id = "myosim_fullbody_articulated_marker_visualization";
    metalrobo::WorldAsset human;
    human.id = "myosim_human";
    human.semanticClass = "human_articulated_marker_view";
    human.role = MR_WORLD_ASSET_ROBOT;
    human.render = MR_WORLD_RENDER_MESH_PBR;
    human.collision = MR_WORLD_COLLISION_NONE;
    human.dynamics = MR_WORLD_DYNAMICS_ARTICULATED;
    human.articulationIndex = 0u;
    human.bodyIndices.resize(model.bodies.size());
    for (std::size_t index = 0u; index < human.bodyIndices.size(); ++index) {
        human.bodyIndices[index] = static_cast<std::uint32_t>(index);
    }
    episode.assets.push_back(std::move(human));
    episode.sensors = {
        makeCamera(cameraNames[0], {center.x, center.y - distance, center.z + 0.10f * distance, 0.0f}, center),
        makeCamera(cameraNames[1], {center.x + distance, center.y, center.z + 0.16f * distance, 0.0f}, center),
        makeCamera(cameraNames[2], {center.x, center.y + distance, center.z + 0.10f * distance, 0.0f}, center),
    };
    episode.task.id = "pose_snapshot_visualization";
    episode.task.robotAssetId = "myosim_human";
    episode.task.controlPeriodSeconds = 1.0 / 120.0;
    episode.task.horizonSeconds = 1.0;
    metalrobo::WorldTemplate world;
    const auto compiled = metalrobo::compileEpisodeTwin(episode, model, world);
    require(compiled.succeeded(), "native Human visual world compile failed: " + compiled.message);
    return world;
}

metalrobo::VisualMotionSampleBatchV1 makeMotion(
    const std::span<const MRBodyStateGPU> bodies
) {
    metalrobo::VisualMotionSampleBatchV1 motion;
    motion.environmentCount = 1u;
    motion.bodyCount = static_cast<std::uint32_t>(bodies.size());
    motion.sampleCount = 2u;
    motion.exposureOpenSeconds = 0.0;
    motion.exposureCloseSeconds = 1.0 / 120.0;
    motion.timestampsSeconds = {motion.exposureOpenSeconds, motion.exposureCloseSeconds};
    motion.bodyStates.insert(motion.bodyStates.end(), bodies.begin(), bodies.end());
    motion.bodyStates.insert(motion.bodyStates.end(), bodies.begin(), bodies.end());
    motion.scenarioIdentity = 0x4d594f53494dull;
    motion.source = MR_VISUAL_SOURCE_SIMULATION;
    return motion;
}

float linearToSrgb(const float value) {
    const float mapped = std::max(value, 0.0f) / (1.0f + std::max(value, 0.0f));
    return mapped <= 0.0031308f
        ? 12.92f * mapped
        : 1.055f * std::pow(mapped, 1.0f / 2.4f) - 0.055f;
}

bool writePng(
    const std::filesystem::path& path,
    const metalrobo::HybridObservationBatch& observations
) {
    const std::size_t pixels = static_cast<std::size_t>(observations.width) * observations.height;
    if (observations.environmentCount != 1u || observations.rgb.size() != pixels) {
        return false;
    }
    std::vector<std::uint8_t> rgba(pixels * 4u);
    for (std::size_t pixel = 0u; pixel < pixels; ++pixel) {
        const mr_float4 value = observations.rgb[pixel];
        rgba[pixel * 4u + 0u] = static_cast<std::uint8_t>(std::lround(
            255.0f * std::clamp(linearToSrgb(value.x), 0.0f, 1.0f)
        ));
        rgba[pixel * 4u + 1u] = static_cast<std::uint8_t>(std::lround(
            255.0f * std::clamp(linearToSrgb(value.y), 0.0f, 1.0f)
        ));
        rgba[pixel * 4u + 2u] = static_cast<std::uint8_t>(std::lround(
            255.0f * std::clamp(linearToSrgb(value.z), 0.0f, 1.0f)
        ));
        rgba[pixel * 4u + 3u] = 255u;
    }
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(
        kCFAllocatorDefault,
        reinterpret_cast<const UInt8*>(path.c_str()),
        static_cast<CFIndex>(path.string().size()), false
    );
    if (url == nullptr) return false;
    CGDataProviderRef provider = CGDataProviderCreateWithData(nullptr, rgba.data(), rgba.size(), nullptr);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGImageRef image = provider != nullptr && colorSpace != nullptr
        ? CGImageCreate(observations.width, observations.height, 8u, 32u,
              static_cast<std::size_t>(observations.width) * 4u, colorSpace,
              static_cast<CGBitmapInfo>(
                  static_cast<std::uint32_t>(kCGBitmapByteOrderDefault) |
                  static_cast<std::uint32_t>(kCGImageAlphaLast)
              ), provider, nullptr, false, kCGRenderingIntentDefault)
        : nullptr;
    CGImageDestinationRef destination = image != nullptr
        ? CGImageDestinationCreateWithURL(url, CFSTR("public.png"), 1u, nullptr)
        : nullptr;
    bool succeeded = false;
    if (destination != nullptr) {
        CGImageDestinationAddImage(destination, image, nullptr);
        succeeded = CGImageDestinationFinalize(destination);
    }
    if (destination != nullptr) CFRelease(destination);
    if (image != nullptr) CGImageRelease(image);
    if (colorSpace != nullptr) CGColorSpaceRelease(colorSpace);
    if (provider != nullptr) CGDataProviderRelease(provider);
    CFRelease(url);
    return succeeded;
}

std::size_t coverage(
    const metalrobo::HybridObservationBatch& observations,
    const std::uint32_t semantic
) {
    return static_cast<std::size_t>(std::count(
        observations.segmentation.begin(), observations.segmentation.end(), semantic
    ));
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (argc != 4) {
                std::cerr << "usage: " << argv[0]
                          << " <myosim-fullbody-core-reference.nhrigid>"
                          << " <myosim-fullbody-muscle-reference.nhmyo> <output-directory>\n";
                return 2;
            }
            const LoadedRigid rigid = loadRigid(argv[1]);
            const LoadedMuscles musclePayload = loadMuscles(argv[2], rigid.header);
            const metalrobo::MetalArticulatedOperatorInput input{
                .articulationIndex = 0u,
                .environmentCount = 1u,
                .pointCount = 0u,
                .q = rigid.model.defaultQ,
                .points = {},
            };
            metalrobo::MetalArticulatedOperatorConfig operatorConfig;
            operatorConfig.pointJacobiansOnly = true;
            metalrobo::MetalArticulatedOperatorResult poseResult;
            const auto poseDiagnostics = metalrobo::runMetalArticulatedOperator(
                rigid.model, input, poseResult, operatorConfig
            );
            require(poseDiagnostics.succeeded() && poseDiagnostics.dispatched &&
                        poseDiagnostics.published && poseDiagnostics.successfulEnvironmentCount == 1u,
                    "native Human Metal pose pass failed: " + poseDiagnostics.message);
            const std::vector<MRBodyStateGPU> bodies = visualBodyStates(
                rigid.model, poseResult.bodyPoses
            );
            std::array<std::string, 3u> cameraNames;
            const metalrobo::WorldTemplate world = makeWorld(rigid.model, bodies, cameraNames);
            metalrobo::WorldProgram program;
            program.id = "myosim_fullbody_articulated_marker_visual_program";
            metalrobo::WorldFamily family;
            const auto familyCompile = metalrobo::compileWorldFamily(world, program, family);
            require(familyCompile.succeeded(), "native Human visual family compile failed: " + familyCompile.message);
            metalrobo::MetalWorldFamilyContext worlds;
            const auto worldsCompile = worlds.compile(family, 1u);
            require(worldsCompile.succeeded(), "native Human visual device world compile failed: " + worldsCompile.message);
            const auto worldsSample = worlds.sample(1u, 0x4d594f53494dull);
            require(worldsSample.succeeded(), "native Human visual world sample failed: " + worldsSample.message);

            std::uint32_t renderedBodies = 0u;
            std::uint32_t renderedRouteSegments = 0u;
            const metalrobo::VisualAssetPackV2 pack = makeMarkerPack(
                rigid.model, musclePayload, bodies, renderedBodies, renderedRouteSegments
            );
            const std::filesystem::path outputDirectory{argv[3]};
            std::filesystem::create_directories(outputDirectory);
            const std::filesystem::path packPath = outputDirectory / "myosim-fullbody-articulated-markers.mrvpack";
            std::string reason;
            require(metalrobo::writeVisualAssetPack(pack, packPath, &reason),
                    "could not write native Human marker pack: " + reason);
            const std::array references{
                metalrobo::VisualAssetReferenceV3{packPath, pack.contentHash, 0u, kBodySemantic, 1u},
            };
            metalrobo::VisualSceneManifestV3 manifest;
            require(metalrobo::compileVisualSceneManifestV3(
                        world, references, metalrobo::makeNeutralStudioEnvironmentV2(),
                        metalrobo::makeIndoorAreaLightRigV1(), manifest, &reason
                    ),
                    "native Human visual scene compile failed: " + reason);
            require(metalrobo::writeVisualSceneManifestV3(
                        manifest, outputDirectory / "myosim-fullbody-articulated-markers.visual.v3.json", &reason
                    ),
                    "could not write native Human visual manifest: " + reason);

            metalrobo::MetalHybridRendererConfig rendererConfig;
            rendererConfig.width = kFrameWidth;
            rendererConfig.height = kFrameHeight;
            rendererConfig.maximumReferenceFramesInFlight = 1u;
            rendererConfig.clearColorAndDepth = {0.002f, 0.006f, 0.012f, 1.0e30f};
            metalrobo::MetalHybridRenderer renderer(rendererConfig);
            const auto rendererCompile = renderer.compile(
                std::move(manifest.renderScene),
                metalrobo::VisualRendererProfileV1::sensorReference(), 1u
            );
            require(rendererCompile.succeeded(), "native Human renderer compile failed: " + rendererCompile.message);
            metalrobo::VisualMotionSampleBatchV1 motion = makeMotion(bodies);
            for (std::size_t camera = 0u; camera < cameraNames.size(); ++camera) {
                motion.sensorIdentity = camera + 1u;
                motion.sensorSequence = static_cast<std::uint32_t>(camera + 1u);
                motion.frameIndex = camera + 1u;
                const auto render = renderer.renderFrame(worlds, motion, static_cast<std::uint32_t>(camera));
                require(render.succeeded(), "native Human render failed: " + render.message);
                metalrobo::HybridObservationBatch observation;
                const auto readback = renderer.readback(observation);
                require(readback.succeeded(), "native Human render readback failed: " + readback.message);
                const std::filesystem::path frame = outputDirectory /
                    ("myosim-fullbody-articulated-" + cameraNames[camera] + ".png");
                require(writePng(frame, observation), "could not write native Human PNG " + frame.string());
                const std::size_t bodyPixels = coverage(observation, kBodySemantic);
                const std::size_t sitePixels = coverage(observation, kSiteSemantic);
                const std::size_t routePixels = coverage(observation, kRouteSemantic);
                require(bodyPixels > 0u && sitePixels > 0u && routePixels > 0u,
                        "native Human frame has no body, muscle-site, or route coverage");
                std::cout << "view=" << cameraNames[camera]
                          << " body_pixels=" << bodyPixels
                          << " muscle_site_pixels=" << sitePixels
                          << " muscle_route_pixels=" << routePixels
                          << " frame=" << frame.string() << '\n';
            }
            std::cout << std::setprecision(12)
                      << "myosim_articulated_marker_visual=ok"
                      << " metal_pose_device=\"" << poseDiagnostics.deviceName << "\""
                      << " renderer_device=\"" << rendererCompile.deviceName << "\""
                      << " core_bodies=" << rigid.header.engineBodyCount
                      << " rendered_inertial_bodies=" << renderedBodies
                      << " muscle_sites=" << musclePayload.sites.size()
                      << " route_centerline_segments=" << renderedRouteSegments
                      << " pose_stage_elapsed_ms=" << poseDiagnostics.elapsedMilliseconds
                      << " renderer_compile_ms=" << rendererCompile.elapsedMilliseconds
                      << " boundary=metal_pose_snapshot_to_native_renderer_not_bodyparts_registration_or_live_rollout\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "myosim_articulated_marker_visual=failed error=\""
                      << error.what() << "\"\n";
            return 1;
        }
    }
}
