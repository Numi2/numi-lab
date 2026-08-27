#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>
#import <Metal/Metal.h>

#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/MetalHybridRenderer.hpp"
#include "metalrobo/MujocoMuscleReference.hpp"
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
#include <optional>
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
constexpr std::uint32_t kBoneSemantic = 51004u;
constexpr std::uint32_t kDefaultFrameDimension = 1024u;
constexpr std::array<char, 8u> kBoneMagic{
    'N', 'H', 'B', 'O', 'N', 'E', 'S', '1',
};
constexpr std::uint32_t kBonePayloadAbi = 1u;

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

struct BoneHeader {
    std::array<char, 8u> magic{};
    std::uint32_t payloadAbi = 0u;
    std::uint32_t boneCount = 0u;
    std::uint32_t vertexCount = 0u;
    std::uint32_t indexCount = 0u;
    std::uint32_t reserved0 = 0u;
    std::array<std::uint8_t, 32u> sourceSha256{};
};

struct BoneRecord {
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    std::uint32_t firstVertex = 0u;
    std::uint32_t vertexCount = 0u;
    std::uint32_t firstIndex = 0u;
    std::uint32_t indexCount = 0u;
    std::uint32_t stableId = 0u;
    float translationX = 0.0f;
    float translationY = 0.0f;
    float translationZ = 0.0f;
    float quaternionX = 0.0f;
    float quaternionY = 0.0f;
    float quaternionZ = 0.0f;
    float quaternionW = 1.0f;
    float uniformScale = 1.0f;
};

struct BoneVertex {
    float positionX = 0.0f;
    float positionY = 0.0f;
    float positionZ = 0.0f;
    float normalX = 0.0f;
    float normalY = 0.0f;
    float normalZ = 1.0f;
};

struct LoadedMuscles {
    MuscleHeader header{};
    std::vector<SiteRecord> sites;
    std::vector<WrapRecord> wraps;
    std::vector<RouteRecord> routes;
    std::vector<MuscleRecord> muscles;
    std::vector<metalrobo::MujocoMuscleSite> referenceSites;
    std::vector<metalrobo::MujocoWrapGeometry> referenceWraps;
    std::vector<metalrobo::MujocoMuscleDefinition> referenceMuscles;
};

struct LoadedBones {
    BoneHeader header{};
    std::vector<BoneRecord> records;
    std::vector<BoneVertex> vertices;
    std::vector<std::uint32_t> indices;
};
#pragma pack(pop)

static_assert(sizeof(RigidHeader) == 80u);
static_assert(sizeof(SourcePoseRecord) == 28u);
static_assert(sizeof(MuscleHeader) == 76u);
static_assert(sizeof(SiteRecord) == 16u);
static_assert(sizeof(WrapRecord) == 64u);
static_assert(sizeof(RouteRecord) == 16u);
static_assert(sizeof(MuscleRecord) == 164u);
static_assert(sizeof(BoneHeader) == 60u);
static_assert(sizeof(BoneRecord) == 56u);
static_assert(sizeof(BoneVertex) == 24u);

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

metalrobo::MujocoRouteNodeType referenceRouteType(const std::uint32_t type) {
    switch (type) {
    case 1u: return metalrobo::MujocoRouteNodeType::site;
    case 2u: return metalrobo::MujocoRouteNodeType::sphere;
    case 3u: return metalrobo::MujocoRouteNodeType::cylinder;
    default: throw std::runtime_error("MyoSim route type is invalid");
    }
}

metalrobo::MujocoRouteNodeType referenceWrapType(const std::uint32_t type) {
    return referenceRouteType(type);
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
    result.referenceSites.reserve(result.sites.size());
    for (const SiteRecord& site : result.sites) {
        result.referenceSites.push_back({site.bodyIndex, {site.x, site.y, site.z}});
    }
    result.referenceWraps.reserve(result.wraps.size());
    for (const WrapRecord& wrap : result.wraps) {
        result.referenceWraps.push_back({
            wrap.bodyIndex, referenceWrapType(wrap.type),
            {wrap.centerX, wrap.centerY, wrap.centerZ},
            {wrap.rotation[0], wrap.rotation[1], wrap.rotation[2],
             wrap.rotation[3], wrap.rotation[4], wrap.rotation[5],
             wrap.rotation[6], wrap.rotation[7], wrap.rotation[8]},
            wrap.radius,
        });
    }
    result.referenceMuscles.reserve(result.muscles.size());
    for (const MuscleRecord& muscle : result.muscles) {
        metalrobo::MujocoMuscleDefinition definition;
        definition.route.reserve(muscle.routeCount);
        for (std::uint32_t offset = 0u; offset < muscle.routeCount; ++offset) {
            const RouteRecord& route = result.routes[muscle.routeOffset + offset];
            definition.route.push_back({
                referenceRouteType(route.type), route.targetIndex, route.sideSiteIndex,
            });
        }
        definition.lengthRange = {muscle.values[0], muscle.values[1]};
        definition.accelerationScale = muscle.values[2];
        definition.controlRange = {muscle.values[3], muscle.values[4]};
        for (std::size_t parameter = 0u; parameter < 10u; ++parameter) {
            definition.gainParameters[parameter] = muscle.values[5u + parameter];
            definition.biasParameters[parameter] = muscle.values[15u + parameter];
            definition.dynamicParameters[parameter] = muscle.values[25u + parameter];
        }
        result.referenceMuscles.push_back(std::move(definition));
    }
    return result;
}

LoadedBones loadBones(
    const std::filesystem::path& path,
    const RigidHeader& rigid
) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "cannot open BodyParts3D bone payload " + path.string());
    LoadedBones result;
    readObject(input, result.header, "BodyParts3D bone header");
    require(result.header.magic == kBoneMagic &&
                result.header.payloadAbi == kBonePayloadAbi &&
                result.header.reserved0 == 0u &&
                result.header.sourceSha256 == rigid.sourceSha256 &&
                result.header.boneCount > 0u &&
                result.header.vertexCount > 0u &&
                result.header.indexCount > 0u &&
                result.header.indexCount % 3u == 0u &&
                result.header.boneCount <= 256u &&
                result.header.vertexCount <= 4'000'000u &&
                result.header.indexCount <= 24'000'000u,
            "BodyParts3D bone payload/header disagreement");
    result.records = readVector<BoneRecord>(
        input, result.header.boneCount, "BodyParts3D bone records"
    );
    result.vertices = readVector<BoneVertex>(
        input, result.header.vertexCount, "BodyParts3D bone vertices"
    );
    result.indices = readVector<std::uint32_t>(
        input, result.header.indexCount, "BodyParts3D bone indices"
    );
    require(input.peek() == std::char_traits<char>::eof(),
            "BodyParts3D bone payload has trailing bytes");
    for (const BoneVertex& vertex : result.vertices) {
        const float normalLength = std::sqrt(
            vertex.normalX * vertex.normalX +
            vertex.normalY * vertex.normalY +
            vertex.normalZ * vertex.normalZ
        );
        require(std::isfinite(vertex.positionX) && std::isfinite(vertex.positionY) &&
                    std::isfinite(vertex.positionZ) && std::isfinite(normalLength) &&
                    std::abs(normalLength - 1.0f) <= 2.0e-3f,
                "BodyParts3D bone vertex is malformed");
    }
    std::vector<bool> stableIds(result.records.size() + 1u, false);
    for (const BoneRecord& record : result.records) {
        const float orientationLength = std::sqrt(
            record.quaternionX * record.quaternionX + record.quaternionY * record.quaternionY +
            record.quaternionZ * record.quaternionZ + record.quaternionW * record.quaternionW
        );
        require(record.bodyIndex < rigid.engineBodyCount && record.vertexCount > 0u &&
                    record.indexCount > 0u && record.indexCount % 3u == 0u &&
                    record.firstVertex <= result.vertices.size() &&
                    record.vertexCount <= result.vertices.size() - record.firstVertex &&
                    record.firstIndex <= result.indices.size() &&
                    record.indexCount <= result.indices.size() - record.firstIndex &&
                    record.stableId > 0u && record.stableId < stableIds.size() &&
                    !stableIds[record.stableId] && std::isfinite(record.translationX) &&
                    std::isfinite(record.translationY) && std::isfinite(record.translationZ) &&
                    std::isfinite(record.uniformScale) && record.uniformScale > 0.0f &&
                    std::isfinite(orientationLength) &&
                    std::abs(orientationLength - 1.0f) <= 2.0e-3f,
                "BodyParts3D bone record is malformed");
        stableIds[record.stableId] = true;
        for (std::uint32_t offset = 0u; offset < record.indexCount; ++offset) {
            const std::uint32_t index = result.indices[record.firstIndex + offset];
            require(index >= record.firstVertex && index < record.firstVertex + record.vertexCount,
                    "BodyParts3D bone index escapes its source mesh");
        }
    }
    return result;
}

struct MuscleDrivenVisualState {
    std::vector<float> q;
    double maximumVelocityDelta = 0.0;
    double maximumConfigurationDelta = 0.0;
    std::uint32_t appliedWrapCount = 0u;
};

MuscleDrivenVisualState integrateMuscleDrivenVisualState(
    const metalrobo::EngineModel& model,
    const LoadedMuscles& muscles,
    const double timestepSeconds
) {
    require(std::isfinite(timestepSeconds) &&
                timestepSeconds >= 1.0e-6 && timestepSeconds <= 1.0e-3,
            "muscle-driven visual step must be between 1 us and 1 ms");
    require(model.world.nv > 0u && model.defaultQ.size() == model.world.nq &&
                model.defaultV.size() == model.world.nv &&
                muscles.referenceMuscles.size() == muscles.muscles.size(),
            "muscle-driven visual state has inconsistent MyoSim dimensions");

    const std::vector<double> initialQ(model.defaultQ.begin(), model.defaultQ.end());
    const std::vector<double> initialV(model.defaultV.begin(), model.defaultV.end());
    std::vector<double> muscleForce(model.world.nv, 0.0);
    MuscleDrivenVisualState result;
    const metalrobo::MujocoMuscleState state{.excitation = 0.5, .activation = 0.5};
    for (std::size_t index = 0u; index < muscles.referenceMuscles.size(); ++index) {
        metalrobo::MujocoMuscleResult muscleResult;
        const auto diagnostics = metalrobo::projectMujocoMuscleForce(
            model, 0u, initialQ, initialV, muscles.referenceSites,
            muscles.referenceWraps, muscles.referenceMuscles[index], state,
            muscleForce, &muscleResult
        );
        require(diagnostics.succeeded(),
                "MyoSim muscle force projection failed for muscle " +
                    std::to_string(index) + ": " +
                    metalrobo::mujocoMuscleReferenceStatusName(diagnostics.status));
        result.appliedWrapCount += muscleResult.path.appliedWrapCount;
    }
    require(std::all_of(muscleForce.begin(), muscleForce.end(), [](const double value) {
                return std::isfinite(value);
            }),
            "MyoSim muscle force projection returned a non-finite generalized force");

    metalrobo::ArticulatedDynamicsConfig dynamicsConfig;
    dynamicsConfig.timestep = timestepSeconds;
    std::vector<double> passiveQ = initialQ;
    std::vector<double> passiveV = initialV;
    std::vector<double> activeQ = initialQ;
    std::vector<double> activeV = initialV;
    const std::vector<double> zeroForce(model.world.nv, 0.0);
    const auto passiveDiagnostics = metalrobo::integrateArticulatedState(
        model, 0u, passiveQ, passiveV, zeroForce, {}, dynamicsConfig
    );
    require(passiveDiagnostics.succeeded(),
            "passive free-body visual comparison step failed");
    const auto activeDiagnostics = metalrobo::integrateArticulatedState(
        model, 0u, activeQ, activeV, muscleForce, {}, dynamicsConfig
    );
    require(activeDiagnostics.succeeded(),
            "muscle-driven free-body visual step failed");
    for (std::size_t index = 0u; index < activeV.size(); ++index) {
        result.maximumVelocityDelta = std::max(
            result.maximumVelocityDelta, std::abs(activeV[index] - passiveV[index])
        );
    }
    for (std::size_t index = 0u; index < activeQ.size(); ++index) {
        result.maximumConfigurationDelta = std::max(
            result.maximumConfigurationDelta, std::abs(activeQ[index] - passiveQ[index])
        );
    }
    require(std::isfinite(result.maximumVelocityDelta) &&
                std::isfinite(result.maximumConfigurationDelta) &&
                result.maximumVelocityDelta > 1.0e-9 &&
                result.maximumConfigurationDelta > 1.0e-12,
            "the complete MyoSim muscle force did not distinguish the visual state step");
    result.q.reserve(activeQ.size());
    for (const double coordinate : activeQ) {
        require(std::isfinite(coordinate) &&
                    coordinate >= -static_cast<double>(std::numeric_limits<float>::max()) &&
                    coordinate <= static_cast<double>(std::numeric_limits<float>::max()),
                "muscle-driven visual configuration is not representable on Metal");
        result.q.push_back(static_cast<float>(coordinate));
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

struct SourceRouteCentreline {
    std::uint32_t muscleIndex = 0u;
    struct Point {
        mr_float4 world{};
        std::uint32_t attachmentBodyIndex = MR_INVALID_INDEX;
    };
    std::vector<Point> points;
};

struct SourceRouteCentrelines {
    std::vector<SourceRouteCentreline> muscles;
    std::uint32_t appliedWrapCount = 0u;
    std::uint32_t surfaceProjectedAttachmentCount = 0u;
};

SourceRouteCentrelines resolveSourceRouteCentrelines(
    const metalrobo::EngineModel& model,
    const LoadedMuscles& musclePayload,
    const std::span<const float> poseQ,
    const std::span<const std::uint32_t> requestedMuscles
) {
    require(poseQ.size() == model.world.nq && model.defaultV.size() == model.world.nv,
            "MyoSim source-route visual pose dimensions are inconsistent");
    std::vector<double> q(poseQ.begin(), poseQ.end());
    const std::vector<double> v(model.defaultV.begin(), model.defaultV.end());
    SourceRouteCentrelines result;
    const std::size_t expectedCount = requestedMuscles.empty()
        ? musclePayload.referenceMuscles.size() : requestedMuscles.size();
    result.muscles.reserve(expectedCount);
    const metalrobo::MujocoMuscleState state{};
    const auto resolve = [&](const std::uint32_t index) {
        require(index < musclePayload.referenceMuscles.size(),
                "requested MyoSim source-route muscle index is out of bounds");
        metalrobo::MujocoMuscleResult muscleResult;
        const auto diagnostics = metalrobo::evaluateMujocoMuscle(
            model, 0u, q, v, musclePayload.referenceSites,
            musclePayload.referenceWraps, musclePayload.referenceMuscles[index],
            state, muscleResult
        );
        require(diagnostics.succeeded() && muscleResult.path.centreline.size() >= 2u,
                "MyoSim source-route resolution failed for muscle " + std::to_string(index) + ": " +
                    metalrobo::mujocoMuscleReferenceStatusName(diagnostics.status));
        std::vector<SourceRouteCentreline::Point> centreline;
        centreline.reserve(muscleResult.path.centreline.size());
        for (const metalrobo::MujocoMusclePathSample& sample : muscleResult.path.centreline) {
            require(std::all_of(sample.world.begin(), sample.world.end(), [](const double value) {
                        return std::isfinite(value) &&
                            value >= -static_cast<double>(std::numeric_limits<float>::max()) &&
                            value <= static_cast<double>(std::numeric_limits<float>::max());
                    }),
                    "MyoSim source-route sample is not representable on the renderer");
            centreline.push_back({
                {
                    static_cast<float>(sample.world[0]), static_cast<float>(sample.world[1]),
                    static_cast<float>(sample.world[2]), 1.0f,
                },
                sample.attachmentBodyIndex,
            });
        }
        result.appliedWrapCount += muscleResult.path.appliedWrapCount;
        result.muscles.push_back({index, std::move(centreline)});
    };
    if (requestedMuscles.empty()) {
        for (std::uint32_t index = 0u; index < musclePayload.referenceMuscles.size(); ++index) {
            resolve(index);
        }
    } else {
        for (const std::uint32_t index : requestedMuscles) resolve(index);
    }
    return result;
}

mr_float4 subtractPoint(const mr_float4 first, const mr_float4 second) {
    return {first.x - second.x, first.y - second.y, first.z - second.z, 0.0f};
}

mr_float4 addPoint(const mr_float4 first, const mr_float4 second) {
    return {first.x + second.x, first.y + second.y, first.z + second.z, 1.0f};
}

mr_float4 scalePoint(const mr_float4 point, const float scalar) {
    return {point.x * scalar, point.y * scalar, point.z * scalar, 0.0f};
}

float dotPoint(const mr_float4 first, const mr_float4 second) {
    return first.x * second.x + first.y * second.y + first.z * second.z;
}

mr_float4 rotatePoint(const mr_float4 quaternion, const mr_float4 point) {
    const mr_float4 axis{quaternion.x, quaternion.y, quaternion.z, 0.0f};
    const mr_float4 twiceCross{
        2.0f * (axis.y * point.z - axis.z * point.y),
        2.0f * (axis.z * point.x - axis.x * point.z),
        2.0f * (axis.x * point.y - axis.y * point.x),
        0.0f,
    };
    const mr_float4 correction{
        quaternion.w * twiceCross.x + axis.y * twiceCross.z - axis.z * twiceCross.y,
        quaternion.w * twiceCross.y + axis.z * twiceCross.x - axis.x * twiceCross.z,
        quaternion.w * twiceCross.z + axis.x * twiceCross.y - axis.y * twiceCross.x,
        0.0f,
    };
    return {point.x + correction.x, point.y + correction.y, point.z + correction.z, 0.0f};
}

mr_float4 closestPointOnTriangle(
    const mr_float4 point,
    const mr_float4 first,
    const mr_float4 second,
    const mr_float4 third
) {
    const mr_float4 firstToPoint = subtractPoint(point, first);
    const mr_float4 firstToSecond = subtractPoint(second, first);
    const mr_float4 firstToThird = subtractPoint(third, first);
    const float dotFirstSecond = dotPoint(firstToSecond, firstToPoint);
    const float dotFirstThird = dotPoint(firstToThird, firstToPoint);
    if (dotFirstSecond <= 0.0f && dotFirstThird <= 0.0f) return first;

    const mr_float4 secondToPoint = subtractPoint(point, second);
    const float dotSecondSecond = dotPoint(firstToSecond, secondToPoint);
    const float dotSecondThird = dotPoint(firstToThird, secondToPoint);
    if (dotSecondSecond >= 0.0f && dotSecondThird <= dotSecondSecond) return second;

    const float edgeFirstSecond = dotFirstSecond * dotSecondThird - dotSecondSecond * dotFirstThird;
    if (edgeFirstSecond <= 0.0f && dotFirstSecond >= 0.0f && dotSecondSecond <= 0.0f) {
        return addPoint(first, scalePoint(firstToSecond, dotFirstSecond / (dotFirstSecond - dotSecondSecond)));
    }

    const mr_float4 thirdToPoint = subtractPoint(point, third);
    const float dotThirdSecond = dotPoint(firstToSecond, thirdToPoint);
    const float dotThirdThird = dotPoint(firstToThird, thirdToPoint);
    if (dotThirdThird >= 0.0f && dotThirdSecond <= dotThirdThird) return third;

    const float edgeFirstThird = dotThirdSecond * dotFirstThird - dotFirstSecond * dotThirdThird;
    if (edgeFirstThird <= 0.0f && dotFirstThird >= 0.0f && dotThirdThird <= 0.0f) {
        return addPoint(first, scalePoint(firstToThird, dotFirstThird / (dotFirstThird - dotThirdThird)));
    }

    const float edgeSecondThird = dotSecondSecond * dotThirdThird - dotThirdSecond * dotSecondThird;
    if (edgeSecondThird <= 0.0f &&
        dotSecondThird - dotSecondSecond >= 0.0f && dotThirdSecond - dotThirdThird >= 0.0f) {
        const mr_float4 secondToThird = subtractPoint(third, second);
        const float ratio = (dotSecondThird - dotSecondSecond) /
            ((dotSecondThird - dotSecondSecond) + (dotThirdSecond - dotThirdThird));
        return addPoint(second, scalePoint(secondToThird, ratio));
    }

    const float barycentricDenominator = edgeFirstSecond + edgeFirstThird + edgeSecondThird;
    if (std::abs(barycentricDenominator) <= 1.0e-12f) return first;
    const float denominator = 1.0f / barycentricDenominator;
    const float secondWeight = edgeFirstThird * denominator;
    const float thirdWeight = edgeFirstSecond * denominator;
    return addPoint(first, addPoint(scalePoint(firstToSecond, secondWeight), scalePoint(firstToThird, thirdWeight)));
}

mr_float4 boneVertexWorld(
    const BoneRecord& bone,
    const BoneVertex& vertex,
    const MRBodyStateGPU& body
) {
    const mr_float4 boneRotation{
        bone.quaternionX, bone.quaternionY, bone.quaternionZ, bone.quaternionW,
    };
    const mr_float4 local = addPoint(
        {bone.translationX, bone.translationY, bone.translationZ, 0.0f},
        scalePoint(
            rotatePoint(boneRotation, {vertex.positionX, vertex.positionY, vertex.positionZ, 0.0f}),
            bone.uniformScale
        )
    );
    return addPoint(body.position, rotatePoint(body.orientation, local));
}

void projectSourceSiteEndpointsToBoneSurfaces(
    SourceRouteCentrelines& routes,
    const LoadedBones& bones,
    const std::span<const MRBodyStateGPU> bodies
) {
    constexpr float maximumProjectionDistance = 0.12f;
    constexpr float maximumProjectionDistanceSquared =
        maximumProjectionDistance * maximumProjectionDistance;
    require(bodies.size() > 0u, "source-route surface projection has no body poses");
    for (SourceRouteCentreline& route : routes.muscles) {
        for (SourceRouteCentreline::Point& point : route.points) {
            if (point.attachmentBodyIndex == MR_INVALID_INDEX ||
                point.attachmentBodyIndex >= bodies.size()) {
                continue;
            }
            mr_float4 closest{};
            float closestDistanceSquared = std::numeric_limits<float>::infinity();
            for (const BoneRecord& bone : bones.records) {
                if (bone.bodyIndex != point.attachmentBodyIndex) continue;
                for (std::uint32_t offset = 0u; offset < bone.indexCount; offset += 3u) {
                    const std::uint32_t first = bones.indices[bone.firstIndex + offset];
                    const std::uint32_t second = bones.indices[bone.firstIndex + offset + 1u];
                    const std::uint32_t third = bones.indices[bone.firstIndex + offset + 2u];
                    const mr_float4 candidate = closestPointOnTriangle(
                        point.world,
                        boneVertexWorld(bone, bones.vertices[first], bodies[bone.bodyIndex]),
                        boneVertexWorld(bone, bones.vertices[second], bodies[bone.bodyIndex]),
                        boneVertexWorld(bone, bones.vertices[third], bodies[bone.bodyIndex])
                    );
                    const mr_float4 difference = subtractPoint(point.world, candidate);
                    const float distanceSquared = dotPoint(difference, difference);
                    if (distanceSquared < closestDistanceSquared) {
                        closestDistanceSquared = distanceSquared;
                        closest = candidate;
                    }
                }
            }
            if (closestDistanceSquared <= maximumProjectionDistanceSquared) {
                point.world = closest;
                ++routes.surfaceProjectedAttachmentCount;
            }
        }
    }
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
    const mr_float4 target,
    const std::uint32_t dimension
) {
    metalrobo::SensorSpec camera;
    camera.id = id;
    camera.parentAssetId = "myosim_human";
    camera.parentKind = MR_WORLD_SENSOR_PARENT_ASSET;
    camera.kind = MR_WORLD_SENSOR_RGBD;
    camera.localPose = cameraToward(position, target);
    camera.width = dimension;
    camera.height = dimension;
    const float focalLength = 750.0f * static_cast<float>(dimension) /
        static_cast<float>(kDefaultFrameDimension);
    camera.intrinsics = {focalLength, focalLength, 0.5f * dimension, 0.5f * dimension};
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
    // Keep the known-valid four-camera stand-off.  Resolution is raised to
    // 1024 px above; a tighter distance caused the oblique frustum to omit
    // this incomplete whole-body registration.
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

mr_float4 boneTangent(const BoneVertex& vertex) {
    const mr_float4 normal{vertex.normalX, vertex.normalY, vertex.normalZ, 1.0f};
    const mr_float4 reference = std::abs(normal.z) < 0.9f
        ? mr_float4{0.0f, 0.0f, 1.0f, 0.0f}
        : mr_float4{0.0f, 1.0f, 0.0f, 0.0f};
    mr_float4 tangent{
        reference.y * normal.z - reference.z * normal.y,
        reference.z * normal.x - reference.x * normal.z,
        reference.x * normal.y - reference.y * normal.x,
        0.0f,
    };
    const float length = std::sqrt(
        tangent.x * tangent.x + tangent.y * tangent.y + tangent.z * tangent.z
    );
    require(length > 1.0e-6f, "BodyParts3D bone tangent is degenerate");
    tangent.x /= length;
    tangent.y /= length;
    tangent.z /= length;
    return tangent;
}

GeometryRange appendBoneGeometry(
    metalrobo::VisualAssetPackV2& pack,
    const LoadedBones& bones,
    const BoneRecord& bone
) {
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
    for (std::uint32_t offset = 0u; offset < bone.vertexCount; ++offset) {
        const BoneVertex& source = bones.vertices[bone.firstVertex + offset];
        const mr_float4 position{source.positionX, source.positionY, source.positionZ, 1.0f};
        pack.vertices.push_back({
            position,
            {source.normalX, source.normalY, source.normalZ, 1.0f},
            boneTangent(source),
            {0.0f, 0.0f, 0.0f, 0.0f},
            {1.0f, 1.0f, 1.0f, 1.0f},
        });
        result.minimum.x = std::min(result.minimum.x, position.x);
        result.minimum.y = std::min(result.minimum.y, position.y);
        result.minimum.z = std::min(result.minimum.z, position.z);
        result.maximum.x = std::max(result.maximum.x, position.x);
        result.maximum.y = std::max(result.maximum.y, position.y);
        result.maximum.z = std::max(result.maximum.z, position.z);
    }
    for (std::uint32_t offset = 0u; offset < bone.indexCount; ++offset) {
        pack.indices.push_back(
            vertexBase + bones.indices[bone.firstIndex + offset] - bone.firstVertex
        );
    }
    result.indexCount = bone.indexCount;
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
    const LoadedBones* bonePayload,
    const bool muscleDriven,
    const SourceRouteCentrelines* sourceRouteCentrelines,
    std::uint32_t& renderedBodies,
    std::uint32_t& renderedRouteSegments
) {
    metalrobo::VisualAssetPackV2 pack;
    pack.id = bonePayload != nullptr
        ? "myosim_fullbody_articulated_bodyparts_bones_view"
        : "myosim_fullbody_articulated_marker_view";
    pack.sourceUri = bonePayload != nullptr
        ? "numi://bodyparts3d/NHBONES1+NHRIGID2+NHMYO1/articulated-bone-view"
        : "numi://myosim/NHRIGID2+NHMYO1/articulated-marker-view";
    pack.sourceContentHash = bonePayload != nullptr
        ? "bodyparts3d-major-bones+runtime-body-and-site-records"
        : "runtime-body-and-site-records";
    pack.license = bonePayload != nullptr ? "CC-BY-4.0 AND Apache-2.0" : "Apache-2.0";
    pack.preprocessingProvenance =
        bonePayload != nullptr
            ? (muscleDriven
                ? "bodyparts3d_source_import/provisional_rest_registration/cpu_fp64_mujoco_muscle_projection_and_articulated_free_body_step/metal_articulated_operator_pose_snapshot/native_visual_bone_pack.v2"
                : "bodyparts3d_source_import/provisional_rest_registration/metal_articulated_operator_pose_snapshot/native_visual_bone_pack.v2")
            : (muscleDriven
                ? "cpu_fp64_mujoco_muscle_projection_and_articulated_free_body_step/metal_articulated_operator_pose_snapshot/native_visual_marker_pack.v1"
                : "metal_articulated_operator_pose_snapshot/native_visual_marker_pack.v1");
    if (sourceRouteCentrelines != nullptr) {
        pack.preprocessingProvenance +=
            "/cpu_fp64_mujoco_tangent_and_wrapped_arc_centreline_at_the_rendered_pose";
        if (sourceRouteCentrelines->surfaceProjectedAttachmentCount > 0u) {
            pack.preprocessingProvenance +=
                "/visual_only_nearest_bodyparts3d_triangle_attachment_projection";
        }
    }
    pack.materials.push_back(makeMaterial(
        {0.82f, 0.86f, 0.88f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}
    ));
    pack.materials.push_back(makeMaterial(
        {0.80f, 0.04f, 0.03f, 1.0f}, {0.15f, 0.0f, 0.0f, 0.25f}
    ));
    pack.materials.push_back(makeMaterial(
        {0.68f, 0.015f, 0.01f, 1.0f}, {0.35f, 0.0f, 0.0f, 0.45f}
    ));
    pack.materials.push_back(makeMaterial(
        {0.78f, 0.66f, 0.46f, 1.0f}, {0.02f, 0.012f, 0.004f, 0.0f}
    ));

    const auto appendInstance = [&pack](
        const GeometryRange& geometry,
        const std::uint32_t material,
        const std::uint32_t semantic,
        const std::uint32_t bindingKind,
        const std::uint32_t bodyIndex,
        const mr_float4 translation,
        const mr_float4 orientation,
        const std::uint32_t stableId
    ) {
        const std::uint32_t instanceIndex = static_cast<std::uint32_t>(pack.instances.size());
        MRVisualInstanceGPUV2 instance{};
        instance.translationAndScale = translation;
        instance.orientation = orientation;
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
    if (bonePayload != nullptr) {
        for (const BoneRecord& bone : bonePayload->records) {
            const GeometryRange geometry = appendBoneGeometry(pack, *bonePayload, bone);
            appendInstance(
                geometry, 3u, kBoneSemantic, MR_VISUAL_BINDING_ARTICULATED_LINK,
                bone.bodyIndex,
                {bone.translationX, bone.translationY, bone.translationZ, bone.uniformScale},
                {bone.quaternionX, bone.quaternionY, bone.quaternionZ, bone.quaternionW},
                bone.stableId
            );
            ++renderedBodies;
        }
    } else {
        for (std::size_t bodyIndex = 0u; bodyIndex < model.bodies.size(); ++bodyIndex) {
            const MRBodyPropertiesGPU& body = model.bodies[bodyIndex];
            if (!(body.massAndInverseMass.x > 1.0e-5f)) {
                continue;
            }
            const GeometryRange geometry = appendEllipsoid(pack, inertiaEllipsoid(body));
            appendInstance(
                geometry, 0u, kBodySemantic, MR_VISUAL_BINDING_ARTICULATED_LINK,
                static_cast<std::uint32_t>(bodyIndex),
                {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f},
                static_cast<std::uint32_t>(bodyIndex + 1u)
            );
            ++renderedBodies;
        }
    }
    renderedRouteSegments = 0u;
    std::uint32_t stableRouteId = 1u;
    if (sourceRouteCentrelines != nullptr) {
        for (const SourceRouteCentreline& route : sourceRouteCentrelines->muscles) {
            require(route.muscleIndex < musclePayload.muscles.size() && route.points.size() >= 2u,
                    "MyoSim source-route visual record is malformed");
            for (std::size_t index = 1u; index < route.points.size(); ++index) {
                const mr_float4 previous = route.points[index - 1u].world;
                const mr_float4 current = route.points[index].world;
                const float dx = current.x - previous.x;
                const float dy = current.y - previous.y;
                const float dz = current.z - previous.z;
                if (dx * dx + dy * dy + dz * dz > 1.0e-10f) {
                    appendInstance(
                        appendWorldTube(
                            pack, previous, current,
                            bonePayload != nullptr ? 0.0011f : 0.0022f
                        ), 2u,
                        kRouteSemantic, MR_VISUAL_BINDING_WORLD, MR_INVALID_INDEX,
                        {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f},
                        stableRouteId++
                    );
                    ++renderedRouteSegments;
                }
            }
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
    const std::optional<std::uint32_t> focusBodyIndex,
    const std::uint32_t dimension,
    std::array<std::string, 4u>& cameraNames
) {
    const auto [center, distance] = [&]() -> std::pair<mr_float4, float> {
        if (!focusBodyIndex.has_value()) return frameBounds(model, bodies);
        require(*focusBodyIndex < bodies.size(), "MyoSim visual focus body index is out of bounds");
        // A 0.70 m stand-off retains a complete major limb around a source
        // body while making attachment inspection legible at the renderer's
        // validated 640 px multi-camera resolution.
        const MRBodyStateGPU& focus = bodies[*focusBodyIndex];
        return {{focus.position.x, focus.position.y, focus.position.z, 0.0f}, 0.70f};
    }();
    cameraNames = {"front", "oblique", "side", "rear"};
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
        makeCamera(cameraNames[0], {center.x, center.y - distance, center.z + 0.10f * distance, 0.0f}, center, dimension),
        makeCamera(cameraNames[1], {
            center.x + 0.72f * distance, center.y - 0.72f * distance,
            center.z + 0.16f * distance, 0.0f,
        }, center, dimension),
        makeCamera(cameraNames[2], {center.x + distance, center.y, center.z + 0.16f * distance, 0.0f}, center, dimension),
        makeCamera(cameraNames[3], {center.x, center.y + distance, center.z + 0.10f * distance, 0.0f}, center, dimension),
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

double parseMuscleStepSeconds(const std::string& value) {
    std::size_t parsed = 0u;
    double result = 0.0;
    try {
        result = std::stod(value, &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("--muscle-step-seconds must be a finite decimal number");
    }
    require(parsed == value.size() && std::isfinite(result),
            "--muscle-step-seconds must be a finite decimal number");
    return result;
}

std::uint32_t parseSourceRouteIndex(const std::string& value) {
    std::size_t parsed = 0u;
    unsigned long result = 0ul;
    try {
        result = std::stoul(value, &parsed, 10);
    } catch (const std::exception&) {
        throw std::runtime_error("--source-route-index must be a non-negative integer");
    }
    require(parsed == value.size() && result <= std::numeric_limits<std::uint32_t>::max(),
            "--source-route-index must be a 32-bit non-negative integer");
    return static_cast<std::uint32_t>(result);
}

std::uint32_t parseFrameDimension(const std::string& value) {
    std::size_t parsed = 0u;
    unsigned long result = 0ul;
    try {
        result = std::stoul(value, &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("--dimension must be an integer multiple of 64 from 512 through 2048");
    }
    require(parsed == value.size() && result >= 512ul && result <= 2048ul && result % 64ul == 0ul,
            "--dimension must be an integer multiple of 64 from 512 through 2048");
    return static_cast<std::uint32_t>(result);
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            std::optional<double> muscleStepSeconds;
            bool sourceRouteCentrelines = false;
            bool surfaceProjectSourceSites = false;
            std::vector<std::uint32_t> requestedSourceRouteMuscles;
            std::optional<std::uint32_t> focusBodyIndex;
            std::uint32_t frameDimension = kDefaultFrameDimension;
            std::vector<std::string> positional;
            for (int index = 1; index < argc; ++index) {
                const std::string argument{argv[index]};
                if (argument == "--muscle-step-seconds") {
                    require(index + 1 < argc && !muscleStepSeconds.has_value(),
                            "--muscle-step-seconds requires one value and may be given only once");
                    muscleStepSeconds.emplace(parseMuscleStepSeconds(argv[++index]));
                } else if (argument == "--source-route-centrelines") {
                    require(!sourceRouteCentrelines,
                            "--source-route-centrelines may be given only once");
                    sourceRouteCentrelines = true;
                } else if (argument == "--source-route-index") {
                    require(index + 1 < argc,
                            "--source-route-index requires one muscle index");
                    sourceRouteCentrelines = true;
                    requestedSourceRouteMuscles.push_back(parseSourceRouteIndex(argv[++index]));
                } else if (argument == "--surface-project-source-sites") {
                    require(!surfaceProjectSourceSites,
                            "--surface-project-source-sites may be given only once");
                    surfaceProjectSourceSites = true;
                } else if (argument == "--focus-body-index") {
                    require(index + 1 < argc && !focusBodyIndex.has_value(),
                            "--focus-body-index requires one body index and may be given only once");
                    focusBodyIndex.emplace(parseSourceRouteIndex(argv[++index]));
                } else if (argument == "--dimension") {
                    require(index + 1 < argc && frameDimension == kDefaultFrameDimension,
                            "--dimension requires one value and may be given only once");
                    frameDimension = parseFrameDimension(argv[++index]);
                } else if (!argument.starts_with("--")) {
                    positional.push_back(argument);
                } else {
                    throw std::runtime_error("unknown visual option " + argument);
                }
            }
            if (positional.size() != 3u && positional.size() != 4u) {
                std::cerr << "usage: " << argv[0]
                          << " <myosim-fullbody-core-reference.nhrigid>"
                          << " <myosim-fullbody-muscle-reference.nhmyo>"
                          << " [bodyparts3d-myosim-major-bones.nhbones] <output-directory>"
                          << " [--muscle-step-seconds <1e-6..1e-3>]"
                          << " [--source-route-centrelines] [--source-route-index <0..415>]..."
                          << " [--surface-project-source-sites]"
                          << " [--focus-body-index <0..156>]"
                          << " [--dimension <512..2048; multiple-of-64>]\n";
                return 2;
            }
            const bool bodypartsBoneVisual = positional.size() == 4u;
            const LoadedRigid rigid = loadRigid(positional[0]);
            const LoadedMuscles musclePayload = loadMuscles(positional[1], rigid.header);
            require(!focusBodyIndex.has_value() || *focusBodyIndex < rigid.header.engineBodyCount,
                    "--focus-body-index exceeds the source body count");
            std::sort(requestedSourceRouteMuscles.begin(), requestedSourceRouteMuscles.end());
            const auto duplicate = std::adjacent_find(
                requestedSourceRouteMuscles.begin(), requestedSourceRouteMuscles.end()
            );
            require(duplicate == requestedSourceRouteMuscles.end(),
                    "--source-route-index values must be unique");
            require(std::all_of(
                        requestedSourceRouteMuscles.begin(), requestedSourceRouteMuscles.end(),
                        [&musclePayload](const std::uint32_t index) {
                            return index < musclePayload.referenceMuscles.size();
                        }
                    ),
                    "--source-route-index exceeds the source muscle count");
            std::optional<LoadedBones> bonePayload;
            if (bodypartsBoneVisual) {
                bonePayload.emplace(loadBones(positional[2], rigid.header));
            }
            require(!surfaceProjectSourceSites ||
                        (bodypartsBoneVisual && sourceRouteCentrelines),
                    "--surface-project-source-sites requires BodyParts3D bones and a source-route inspection");
            std::optional<MuscleDrivenVisualState> muscleDrivenState;
            std::span<const float> poseQ = rigid.model.defaultQ;
            if (muscleStepSeconds.has_value()) {
                muscleDrivenState.emplace(integrateMuscleDrivenVisualState(
                    rigid.model, musclePayload, *muscleStepSeconds
                ));
                poseQ = muscleDrivenState->q;
            }
            const metalrobo::MetalArticulatedOperatorInput input{
                .articulationIndex = 0u,
                .environmentCount = 1u,
                .pointCount = 0u,
                .q = poseQ,
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
            std::optional<SourceRouteCentrelines> resolvedRouteCentrelines;
            if (sourceRouteCentrelines) {
                resolvedRouteCentrelines.emplace(resolveSourceRouteCentrelines(
                    rigid.model, musclePayload, poseQ, requestedSourceRouteMuscles
                ));
                if (surfaceProjectSourceSites) {
                    projectSourceSiteEndpointsToBoneSurfaces(
                        *resolvedRouteCentrelines, *bonePayload, bodies
                    );
                }
            }
            std::array<std::string, 4u> cameraNames;
            const metalrobo::WorldTemplate world = makeWorld(
                rigid.model, bodies, focusBodyIndex, frameDimension, cameraNames
            );
            metalrobo::WorldProgram program;
            program.id = bodypartsBoneVisual
                ? "myosim_fullbody_articulated_bodyparts_bone_visual_program"
                : "myosim_fullbody_articulated_marker_visual_program";
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
                rigid.model, musclePayload,
                bonePayload.has_value() ? &*bonePayload : nullptr,
                muscleDrivenState.has_value(),
                resolvedRouteCentrelines.has_value() ? &*resolvedRouteCentrelines : nullptr,
                renderedBodies, renderedRouteSegments
            );
            const std::filesystem::path outputDirectory{positional.back()};
            std::filesystem::create_directories(outputDirectory);
            const std::string stem = std::string(bodypartsBoneVisual
                ? "myosim-fullbody-articulated-bodyparts-bones"
                : "myosim-fullbody-articulated-markers") +
                (muscleDrivenState.has_value() ? "-muscle-driven" : "") +
                (sourceRouteCentrelines ? "-source-route-centrelines" : "") +
                (surfaceProjectSourceSites ? "-surface-projected-sites" : "") +
                (focusBodyIndex.has_value()
                    ? "-focus-body-" + std::to_string(*focusBodyIndex) : "");
            const std::filesystem::path packPath = outputDirectory / (stem + ".mrvpack");
            std::string reason;
            require(metalrobo::writeVisualAssetPack(pack, packPath, &reason),
                    "could not write native Human visual pack: " + reason);
            const std::array references{
                metalrobo::VisualAssetReferenceV3{
                    packPath, pack.contentHash, 0u,
                    bodypartsBoneVisual ? kBoneSemantic : kBodySemantic, 1u,
                },
            };
            metalrobo::VisualSceneManifestV3 manifest;
            require(metalrobo::compileVisualSceneManifestV3(
                        world, references, metalrobo::makeNeutralStudioEnvironmentV2(),
                        metalrobo::makeIndoorAreaLightRigV1(), manifest, &reason
                    ),
                    "native Human visual scene compile failed: " + reason);
            require(metalrobo::writeVisualSceneManifestV3(
                        manifest, outputDirectory / (stem + ".visual.v3.json"), &reason
                    ),
                    "could not write native Human visual manifest: " + reason);

            metalrobo::VisualMotionSampleBatchV1 motion = makeMotion(bodies);
            bool completeVisualCoverage = true;
            std::string rendererDeviceName;
            double rendererCompileMilliseconds = 0.0;
            for (std::size_t camera = 0u; camera < cameraNames.size(); ++camera) {
                // Reference ray workspaces can retain a large drawable and
                // acceleration structure.  Build one isolated renderer per
                // fixed angle so 2048 px anatomy review cannot reuse a prior
                // camera's in-flight workspace.
                metalrobo::VisualSceneManifestV3 cameraManifest;
                require(metalrobo::compileVisualSceneManifestV3(
                            world, references, metalrobo::makeNeutralStudioEnvironmentV2(),
                            metalrobo::makeIndoorAreaLightRigV1(), cameraManifest, &reason
                        ),
                        "native Human per-camera visual scene compile failed: " + reason);
                metalrobo::MetalHybridRendererConfig rendererConfig;
                rendererConfig.width = frameDimension;
                rendererConfig.height = frameDimension;
                rendererConfig.maximumReferenceFramesInFlight = 1u;
                rendererConfig.clearColorAndDepth = {0.002f, 0.006f, 0.012f, 1.0e30f};
                metalrobo::MetalHybridRenderer renderer(rendererConfig);
                const auto rendererCompile = renderer.compile(
                    std::move(cameraManifest.renderScene),
                    metalrobo::VisualRendererProfileV1::sensorReference(), 1u
                );
                require(rendererCompile.succeeded(), "native Human renderer compile failed: " + rendererCompile.message);
                if (camera == 0u) {
                    rendererDeviceName = rendererCompile.deviceName;
                    rendererCompileMilliseconds = rendererCompile.elapsedMilliseconds;
                } else {
                    require(rendererCompile.deviceName == rendererDeviceName,
                            "native Human visual cameras selected different renderer devices");
                }
                motion.sensorIdentity = camera + 1u;
                motion.sensorSequence = static_cast<std::uint32_t>(camera + 1u);
                motion.frameIndex = camera + 1u;
                const auto render = renderer.renderFrame(worlds, motion, static_cast<std::uint32_t>(camera));
                require(render.succeeded(), "native Human render failed: " + render.message);
                metalrobo::HybridObservationBatch observation;
                const auto readback = renderer.readback(observation);
                require(readback.succeeded(), "native Human render readback failed: " + readback.message);
                const std::filesystem::path frame = outputDirectory /
                    (stem + "-" + cameraNames[camera] + ".png");
                require(writePng(frame, observation), "could not write native Human PNG " + frame.string());
                const std::size_t bodyPixels = coverage(observation, kBodySemantic);
                const std::size_t bonePixels = coverage(observation, kBoneSemantic);
                const std::size_t sitePixels = coverage(observation, kSiteSemantic);
                const std::size_t routePixels = coverage(observation, kRouteSemantic);
                completeVisualCoverage = completeVisualCoverage &&
                    (bodypartsBoneVisual ? bonePixels > 0u : bodyPixels > 0u) &&
                    (!sourceRouteCentrelines || routePixels > 0u);
                std::cout << "view=" << cameraNames[camera]
                          << " body_pixels=" << bodyPixels
                          << " bone_pixels=" << bonePixels
                          << " muscle_site_pixels=" << sitePixels
                          << " muscle_route_pixels=" << routePixels
                          << " frame=" << frame.string() << '\n';
            }
            require(completeVisualCoverage,
                    "one or more native Human frames have no linked-body or requested source-route coverage");
            std::cout << std::setprecision(12)
                      << (bodypartsBoneVisual
                              ? "myosim_articulated_bodyparts_bone_visual=ok"
                              : "myosim_articulated_marker_visual=ok")
                      << " metal_pose_device=\"" << poseDiagnostics.deviceName << "\""
                      << " renderer_device=\"" << rendererDeviceName << "\""
                      << " frame_dimension=" << frameDimension
                      << " core_bodies=" << rigid.header.engineBodyCount
                      << " rendered_link_visuals=" << renderedBodies
                      << " bodyparts_bones=" << (bonePayload.has_value() ? bonePayload->records.size() : 0u)
                      << " muscle_sites=" << musclePayload.sites.size()
                      << " route_centerline_segments=" << renderedRouteSegments
                      << " source_route_centrelines=" << (sourceRouteCentrelines ? "true" : "false")
                      << " source_route_muscles=" << (resolvedRouteCentrelines.has_value()
                              ? resolvedRouteCentrelines->muscles.size() : 0u)
                      << " source_route_applied_wraps=" << (resolvedRouteCentrelines.has_value()
                              ? resolvedRouteCentrelines->appliedWrapCount : 0u)
                      << " source_route_surface_projected_sites=" << (resolvedRouteCentrelines.has_value()
                              ? resolvedRouteCentrelines->surfaceProjectedAttachmentCount : 0u)
                      << " focus_body_index=" << (focusBodyIndex.has_value()
                              ? std::to_string(*focusBodyIndex) : "none")
                      << " pose_stage_elapsed_ms=" << poseDiagnostics.elapsedMilliseconds
                      << " renderer_compile_ms_first_camera=" << rendererCompileMilliseconds
                      << " pose_source=" << (muscleDrivenState.has_value()
                              ? "cpu_fp64_mujoco_416_muscle_force_to_articulated_free_body_step_then_metal_kinematic_pose"
                              : "source_default_q_to_metal_kinematic_pose")
                      << " muscle_step_seconds=" << (muscleStepSeconds.has_value()
                              ? *muscleStepSeconds : 0.0)
                      << " muscle_step_applied_wraps=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->appliedWrapCount : 0u)
                      << " muscle_step_max_velocity_delta=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->maximumVelocityDelta : 0.0)
                      << " muscle_step_max_configuration_delta=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->maximumConfigurationDelta : 0.0)
                      << " boundary=" << (muscleDrivenState.has_value()
                              ? (bodypartsBoneVisual
                                  ? "cpu_fp64_complete_416_muscle_force_and_articulated_free_body_step_to_metal_pose_snapshot_with_provisional_bodyparts_bone_registration_not_contact_or_live_rollout"
                                  : "cpu_fp64_complete_416_muscle_force_and_articulated_free_body_step_to_metal_pose_snapshot_not_contact_or_live_rollout")
                              : (bodypartsBoneVisual
                                  ? "metal_pose_snapshot_to_native_renderer_with_provisional_bodyparts_bone_registration_not_collision_or_live_rollout"
                                  : "metal_pose_snapshot_to_native_renderer_not_bodyparts_registration_or_live_rollout"))
                      << " route_geometry=" << (sourceRouteCentrelines
                              ? (surfaceProjectSourceSites
                                  ? "cpu_fp64_mujoco_tangent_and_wrapped_arc_centreline_at_same_q_with_visual_only_nearest_bodyparts3d_triangle_source_site_projection_not_a_force_path_or_tendon_surface_certificate"
                                  : "cpu_fp64_mujoco_tangent_and_wrapped_arc_centreline_at_same_q_not_an_anatomical_tendon_surface_or_bodyparts3d_surface_attachment_certificate")
                              : "hidden_until_a_source_route_centreline_inspection_is_requested")
                      << '\n';
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "myosim_articulated_visual=failed error=\""
                      << error.what() << "\"\n";
            return 1;
        }
    }
}
