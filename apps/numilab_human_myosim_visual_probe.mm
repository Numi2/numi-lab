#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>
#import <Metal/Metal.h>

#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/MetalHybridRenderer.hpp"
#include "metalrobo/MetalMultiArticulatedContact.hpp"
#include "metalrobo/MultiArticulatedContact.hpp"
#include "metalrobo/MujocoMuscleReference.hpp"
#include "metalrobo/QualityContactSolver.hpp"
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
constexpr std::array<char, 8u> kSupportContactMagic{
    'N', 'H', 'C', 'N', 'T', '1', '\0', '\0',
};
constexpr std::uint32_t kPayloadAbi = 1u;
constexpr std::uint32_t kBodySemantic = 51001u;
constexpr std::uint32_t kSiteSemantic = 51002u;
constexpr std::uint32_t kRouteSemantic = 51003u;
constexpr std::uint32_t kBoneSemantic = 51004u;
constexpr std::uint32_t kMuscleSurfaceSemantic = 51005u;
constexpr std::uint32_t kTendonSurfaceSemantic = 51006u;
constexpr std::uint32_t kDefaultFrameDimension = 1024u;
constexpr std::array<char, 8u> kBoneMagic{
    'N', 'H', 'B', 'O', 'N', 'E', 'S', '1',
};
constexpr std::uint32_t kBonePayloadAbi = 2u;
constexpr std::array<char, 8u> kSoftTissueMagic{
    'N', 'H', 'T', 'I', 'S', 'S', '2', '\0',
};
constexpr std::uint32_t kSoftTissuePayloadAbi = 3u;
constexpr std::uint32_t kSoftTissueLayerMuscle = 1u;
constexpr std::uint32_t kSoftTissueLayerTendon = 2u;

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

struct SupportContactHeader {
    std::array<char, 8u> magic{};
    std::uint32_t payloadAbi = 0u;
    std::uint32_t engineBodyCount = 0u;
    std::uint32_t contactCount = 0u;
    std::uint32_t reserved0 = 0u;
    std::array<std::uint8_t, 32u> sourceSha256{};
    float groundPointX = 0.0f;
    float groundPointY = 0.0f;
    float groundPointZ = 0.0f;
    float groundNormalX = 0.0f;
    float groundNormalY = 0.0f;
    float groundNormalZ = 1.0f;
    float groundFriction = 0.0f;
};

struct SupportContactRecord {
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    std::uint32_t sourceGeometryIndex = MR_INVALID_INDEX;
    float localPointX = 0.0f;
    float localPointY = 0.0f;
    float localPointZ = 0.0f;
    float worldWitnessX = 0.0f;
    float worldWitnessY = 0.0f;
    float worldWitnessZ = 0.0f;
    float friction = 0.0f;
    float defaultSignedPlaneDistance = 0.0f;
    float reserved0 = 0.0f;
    float reserved1 = 0.0f;
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

struct SoftTissueHeader {
    std::array<char, 8u> magic{};
    std::uint32_t payloadAbi = 0u;
    std::uint32_t tissueCount = 0u;
    std::uint32_t vertexCount = 0u;
    std::uint32_t indexCount = 0u;
    std::uint32_t reserved0 = 0u;
    std::array<std::uint8_t, 32u> sourceSha256{};
};

struct SoftTissueRecord {
    std::uint32_t primaryBodyIndex = MR_INVALID_INDEX;
    std::uint32_t secondaryBodyIndex = MR_INVALID_INDEX;
    std::uint32_t firstVertex = 0u;
    std::uint32_t vertexCount = 0u;
    std::uint32_t firstIndex = 0u;
    std::uint32_t indexCount = 0u;
    std::uint32_t stableId = 0u;
    std::uint32_t layer = 0u;
    float primaryTranslationX = 0.0f;
    float primaryTranslationY = 0.0f;
    float primaryTranslationZ = 0.0f;
    float primaryQuaternionX = 0.0f;
    float primaryQuaternionY = 0.0f;
    float primaryQuaternionZ = 0.0f;
    float primaryQuaternionW = 1.0f;
    float primaryUniformScale = 1.0f;
    float secondaryTranslationX = 0.0f;
    float secondaryTranslationY = 0.0f;
    float secondaryTranslationZ = 0.0f;
    float secondaryQuaternionX = 0.0f;
    float secondaryQuaternionY = 0.0f;
    float secondaryQuaternionZ = 0.0f;
    float secondaryQuaternionW = 1.0f;
    float secondaryUniformScale = 1.0f;
};

struct SoftTissueVertex {
    float positionX = 0.0f;
    float positionY = 0.0f;
    float positionZ = 0.0f;
    float normalX = 0.0f;
    float normalY = 0.0f;
    float normalZ = 1.0f;
    float primaryWeight = 1.0f;
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
    // Immutable, source-faithful MyoSim program for the articulated Metal
    // operator. Mutable excitation/activation state is supplied separately
    // by each bounded visual transaction.
    std::vector<MRMujocoMuscleSiteGPU> gpuSites;
    std::vector<MRMujocoMuscleWrapGPU> gpuWraps;
    std::vector<MRMujocoMuscleRouteNodeGPU> gpuRoutes;
    std::vector<MRMujocoMuscleGPU> gpuMuscles;
};

struct LoadedBones {
    BoneHeader header{};
    std::vector<BoneRecord> records;
    std::vector<BoneVertex> vertices;
    std::vector<std::uint32_t> indices;
};

struct LoadedSoftTissues {
    SoftTissueHeader header{};
    std::vector<SoftTissueRecord> records;
    std::vector<SoftTissueVertex> vertices;
    std::vector<std::uint32_t> indices;
};

struct LoadedSupportContacts {
    SupportContactHeader header{};
    std::vector<SupportContactRecord> records;
};
#pragma pack(pop)

static_assert(sizeof(RigidHeader) == 80u);
static_assert(sizeof(SourcePoseRecord) == 28u);
static_assert(sizeof(MuscleHeader) == 76u);
static_assert(sizeof(SiteRecord) == 16u);
static_assert(sizeof(WrapRecord) == 64u);
static_assert(sizeof(RouteRecord) == 16u);
static_assert(sizeof(MuscleRecord) == 164u);
static_assert(sizeof(SupportContactHeader) == 84u);
static_assert(sizeof(SupportContactRecord) == 48u);
static_assert(sizeof(BoneHeader) == 60u);
static_assert(sizeof(BoneRecord) == 56u);
static_assert(sizeof(BoneVertex) == 24u);
static_assert(sizeof(SoftTissueHeader) == 60u);
static_assert(sizeof(SoftTissueRecord) == 96u);
static_assert(sizeof(SoftTissueVertex) == 28u);

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
    result.gpuSites.reserve(result.sites.size());
    for (const SiteRecord& site : result.sites) {
        result.referenceSites.push_back({site.bodyIndex, {site.x, site.y, site.z}});
        MRMujocoMuscleSiteGPU gpuSite{};
        gpuSite.bodyIndex = site.bodyIndex;
        gpuSite.localPoint = {site.x, site.y, site.z, 0.0f};
        result.gpuSites.push_back(gpuSite);
    }
    result.referenceWraps.reserve(result.wraps.size());
    result.gpuWraps.reserve(result.wraps.size());
    for (const WrapRecord& wrap : result.wraps) {
        result.referenceWraps.push_back({
            wrap.bodyIndex, referenceWrapType(wrap.type),
            {wrap.centerX, wrap.centerY, wrap.centerZ},
            {wrap.rotation[0], wrap.rotation[1], wrap.rotation[2],
             wrap.rotation[3], wrap.rotation[4], wrap.rotation[5],
             wrap.rotation[6], wrap.rotation[7], wrap.rotation[8]},
            wrap.radius,
        });
        MRMujocoMuscleWrapGPU gpuWrap{};
        gpuWrap.bodyIndex = wrap.bodyIndex;
        gpuWrap.type = wrap.type;
        gpuWrap.localCenter = {
            wrap.centerX, wrap.centerY, wrap.centerZ, 0.0f,
        };
        gpuWrap.rotationRow0 = {
            wrap.rotation[0], wrap.rotation[1], wrap.rotation[2], 0.0f,
        };
        gpuWrap.rotationRow1 = {
            wrap.rotation[3], wrap.rotation[4], wrap.rotation[5], 0.0f,
        };
        gpuWrap.rotationRow2 = {
            wrap.rotation[6], wrap.rotation[7], wrap.rotation[8], 0.0f,
        };
        gpuWrap.radius = {wrap.radius, 0.0f, 0.0f, 0.0f};
        result.gpuWraps.push_back(gpuWrap);
    }
    result.gpuRoutes.reserve(result.routes.size());
    for (const RouteRecord& route : result.routes) {
        result.gpuRoutes.push_back({
            route.type,
            route.targetIndex,
            route.sideSiteIndex,
            0u,
        });
    }
    result.referenceMuscles.reserve(result.muscles.size());
    result.gpuMuscles.reserve(result.muscles.size());
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
        MRMujocoMuscleGPU gpuMuscle{};
        gpuMuscle.route = {
            muscle.routeOffset,
            muscle.routeCount,
            0u,
            0u,
        };
        gpuMuscle.lengthRangeAndAcceleration = {
            muscle.values[0],
            muscle.values[1],
            muscle.values[2],
            0.0f,
        };
        gpuMuscle.controlRange = {
            muscle.values[3],
            muscle.values[4],
            0.0f,
            0.0f,
        };
        for (std::size_t parameter = 0u; parameter < 10u; ++parameter) {
            (&gpuMuscle.gainParameters[parameter / 4u].x)[parameter % 4u] =
                muscle.values[5u + parameter];
            (&gpuMuscle.biasParameters[parameter / 4u].x)[parameter % 4u] =
                muscle.values[15u + parameter];
            (&gpuMuscle.dynamicParameters[parameter / 4u].x)[parameter % 4u] =
                muscle.values[25u + parameter];
        }
        result.gpuMuscles.push_back(gpuMuscle);
    }
    require(
        result.gpuSites.size() == result.sites.size() &&
            result.gpuWraps.size() == result.wraps.size() &&
            result.gpuRoutes.size() == result.routes.size() &&
            result.gpuMuscles.size() == result.muscles.size(),
        "MyoSim Metal source program packing is incomplete"
    );
    return result;
}

LoadedSupportContacts loadSupportContacts(
    const std::filesystem::path& path,
    const RigidHeader& rigid
) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "cannot open MyoSim support-contact payload " + path.string());
    LoadedSupportContacts result;
    readObject(input, result.header, "MyoSim support-contact header");
    require(result.header.magic == kSupportContactMagic &&
                result.header.payloadAbi == kPayloadAbi &&
                result.header.engineBodyCount == rigid.engineBodyCount &&
                result.header.sourceSha256 == rigid.sourceSha256 &&
                result.header.reserved0 == 0u &&
                result.header.contactCount >= 2u &&
                result.header.contactCount <= 32u,
            "MyoSim support-contact payload/header disagreement");
    result.records = readVector<SupportContactRecord>(
        input, result.header.contactCount, "MyoSim support-contact records"
    );
    require(input.peek() == std::char_traits<char>::eof(),
            "MyoSim support-contact payload has trailing bytes");
    const std::array<double, 3u> groundPoint{
        result.header.groundPointX,
        result.header.groundPointY,
        result.header.groundPointZ,
    };
    const std::array<double, 3u> groundNormal{
        result.header.groundNormalX,
        result.header.groundNormalY,
        result.header.groundNormalZ,
    };
    const double normalLength = std::sqrt(
        groundNormal[0] * groundNormal[0] +
        groundNormal[1] * groundNormal[1] +
        groundNormal[2] * groundNormal[2]
    );
    require(std::isfinite(groundPoint[0]) && std::isfinite(groundPoint[1]) &&
                std::isfinite(groundPoint[2]) && std::isfinite(normalLength) &&
                std::abs(normalLength - 1.0) <= 2.0e-4 &&
                std::isfinite(result.header.groundFriction) &&
                result.header.groundFriction >= 0.0f,
            "MyoSim support-contact ground plane is malformed");
    std::vector<std::uint32_t> sourceGeometryIds;
    sourceGeometryIds.reserve(result.records.size());
    for (const SupportContactRecord& record : result.records) {
        const std::array<double, 3u> localPoint{
            record.localPointX, record.localPointY, record.localPointZ,
        };
        const std::array<double, 3u> witness{
            record.worldWitnessX, record.worldWitnessY, record.worldWitnessZ,
        };
        const double witnessPlaneDistance =
            (witness[0] - groundPoint[0]) * groundNormal[0] +
            (witness[1] - groundPoint[1]) * groundNormal[1] +
            (witness[2] - groundPoint[2]) * groundNormal[2];
        require(record.bodyIndex < rigid.engineBodyCount &&
                    record.sourceGeometryIndex != MR_INVALID_INDEX &&
                    std::all_of(localPoint.begin(), localPoint.end(), [](const double value) {
                        return std::isfinite(value);
                    }) &&
                    std::all_of(witness.begin(), witness.end(), [](const double value) {
                        return std::isfinite(value);
                    }) &&
                    std::isfinite(record.friction) && record.friction >= 0.0f &&
                    std::isfinite(record.defaultSignedPlaneDistance) &&
                    std::abs(witnessPlaneDistance) <= 2.0e-4 &&
                    record.reserved0 == 0.0f && record.reserved1 == 0.0f,
                "MyoSim support-contact record is malformed");
        sourceGeometryIds.push_back(record.sourceGeometryIndex);
    }
    std::sort(sourceGeometryIds.begin(), sourceGeometryIds.end());
    require(std::adjacent_find(sourceGeometryIds.begin(), sourceGeometryIds.end()) ==
                sourceGeometryIds.end(),
            "MyoSim support-contact geometry identity is duplicated");
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
                result.header.reserved0 != 0u &&
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

LoadedSoftTissues loadSoftTissues(
    const std::filesystem::path& path,
    const RigidHeader& rigid
) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "cannot open BodyParts3D soft-tissue payload " + path.string());
    LoadedSoftTissues result;
    readObject(input, result.header, "BodyParts3D soft-tissue header");
    require(result.header.magic == kSoftTissueMagic &&
                result.header.payloadAbi == kSoftTissuePayloadAbi &&
                result.header.reserved0 != 0u &&
                result.header.sourceSha256 == rigid.sourceSha256 &&
                result.header.tissueCount > 0u && result.header.tissueCount <= 192u &&
                result.header.vertexCount > 0u && result.header.indexCount > 0u &&
                result.header.indexCount % 3u == 0u &&
                result.header.vertexCount <= 1'000'000u &&
                result.header.indexCount <= 6'000'000u,
            "BodyParts3D soft-tissue payload/header disagreement");
    result.records = readVector<SoftTissueRecord>(
        input, result.header.tissueCount, "BodyParts3D soft-tissue records"
    );
    result.vertices = readVector<SoftTissueVertex>(
        input, result.header.vertexCount, "BodyParts3D soft-tissue vertices"
    );
    result.indices = readVector<std::uint32_t>(
        input, result.header.indexCount, "BodyParts3D soft-tissue indices"
    );
    require(input.peek() == std::char_traits<char>::eof(),
            "BodyParts3D soft-tissue payload has trailing bytes");
    for (const SoftTissueVertex& vertex : result.vertices) {
        const float normalLength = std::sqrt(
            vertex.normalX * vertex.normalX +
            vertex.normalY * vertex.normalY +
            vertex.normalZ * vertex.normalZ
        );
        require(std::isfinite(vertex.positionX) && std::isfinite(vertex.positionY) &&
                    std::isfinite(vertex.positionZ) && std::isfinite(normalLength) &&
                    std::abs(normalLength - 1.0f) <= 2.0e-3f &&
                    std::isfinite(vertex.primaryWeight) &&
                    vertex.primaryWeight >= 0.0f && vertex.primaryWeight <= 1.0f,
                "BodyParts3D soft-tissue vertex is malformed");
    }
    std::vector<bool> stableIds(result.records.size() + 1u, false);
    for (const SoftTissueRecord& record : result.records) {
        const float primaryOrientationLength = std::sqrt(
            record.primaryQuaternionX * record.primaryQuaternionX +
            record.primaryQuaternionY * record.primaryQuaternionY +
            record.primaryQuaternionZ * record.primaryQuaternionZ +
            record.primaryQuaternionW * record.primaryQuaternionW
        );
        const float secondaryOrientationLength = std::sqrt(
            record.secondaryQuaternionX * record.secondaryQuaternionX +
            record.secondaryQuaternionY * record.secondaryQuaternionY +
            record.secondaryQuaternionZ * record.secondaryQuaternionZ +
            record.secondaryQuaternionW * record.secondaryQuaternionW
        );
        require(record.primaryBodyIndex < rigid.engineBodyCount &&
                    record.secondaryBodyIndex < rigid.engineBodyCount &&
                    record.primaryBodyIndex != record.secondaryBodyIndex &&
                    record.vertexCount > 0u &&
                    record.indexCount > 0u && record.indexCount % 3u == 0u &&
                    (record.layer == kSoftTissueLayerMuscle ||
                     record.layer == kSoftTissueLayerTendon) &&
                    record.firstVertex <= result.vertices.size() &&
                    record.vertexCount <= result.vertices.size() - record.firstVertex &&
                    record.firstIndex <= result.indices.size() &&
                    record.indexCount <= result.indices.size() - record.firstIndex &&
                    record.stableId > 0u && record.stableId < stableIds.size() &&
                    !stableIds[record.stableId] &&
                    std::isfinite(record.primaryTranslationX) &&
                    std::isfinite(record.primaryTranslationY) &&
                    std::isfinite(record.primaryTranslationZ) &&
                    std::isfinite(record.primaryUniformScale) && record.primaryUniformScale > 0.0f &&
                    std::isfinite(primaryOrientationLength) &&
                    std::abs(primaryOrientationLength - 1.0f) <= 2.0e-3f &&
                    std::isfinite(record.secondaryTranslationX) &&
                    std::isfinite(record.secondaryTranslationY) &&
                    std::isfinite(record.secondaryTranslationZ) &&
                    std::isfinite(record.secondaryUniformScale) && record.secondaryUniformScale > 0.0f &&
                    std::isfinite(secondaryOrientationLength) &&
                    std::abs(secondaryOrientationLength - 1.0f) <= 2.0e-3f,
                "BodyParts3D soft-tissue record is malformed");
        stableIds[record.stableId] = true;
        for (std::uint32_t offset = 0u; offset < record.indexCount; ++offset) {
            const std::uint32_t index = result.indices[record.firstIndex + offset];
            require(index >= record.firstVertex && index < record.firstVertex + record.vertexCount,
                    "BodyParts3D soft-tissue index escapes its source mesh");
        }
    }
    return result;
}

struct MuscleDrivenVisualState {
    std::vector<float> q;
    std::uint32_t stepCount = 0u;
    double maximumVelocityDelta = 0.0;
    double maximumConfigurationDelta = 0.0;
    std::uint32_t appliedWrapCount = 0u;
    bool supportContactApplied = false;
    std::uint32_t supportWitnessCount = 0u;
    std::uint32_t activeSupportContactCount = 0u;
    std::uint32_t maximumActiveSupportContactCount = 0u;
    double minimumSupportPlaneGapMeters = std::numeric_limits<double>::infinity();
    double supportSeedTranslationMeters = 0.0;
    double supportMaximumGpuCpuVelocityError = 0.0;
    double supportGpuElapsedMilliseconds = 0.0;
    std::string supportDeviceName;
    std::string supportMetalStatus = "not_attempted";
    // Full-body MyoSim force is evaluated by the articulated Metal sidecar
    // before the bounded Core state step. These counters are intentionally
    // separate from support-contact admission, which has a different current
    // capacity boundary.
    std::uint32_t muscleMetalStepCount = 0u;
    std::uint32_t muscleMetalForceRecordCount = 0u;
    double muscleMetalElapsedMilliseconds = 0.0;
    std::string muscleMetalDeviceName;
};

struct MetalMujocoVisualQueries {
    std::vector<MRArticulatedPointImpulseGPU> points;
    std::uint32_t bodyJacobianPointOffset = MR_INVALID_INDEX;
};

MetalMujocoVisualQueries makeMetalMujocoVisualQueries(
    const metalrobo::EngineModel& model
) {
    require(
        model.articulations.size() == 1u &&
            model.articulations.front().bodyCount == model.bodies.size() &&
            model.bodies.size() <=
                MR_ARTICULATED_OPERATOR_MAX_POINTS / 4u,
        "MyoSim Metal visual force queries exceed articulated-operator capacity"
    );
    MetalMujocoVisualQueries result;
    result.bodyJacobianPointOffset = 0u;
    result.points.reserve(4u * model.bodies.size());
    for (std::uint32_t body = 0u;
         body < model.bodies.size();
         ++body) {
        for (std::uint32_t probe = 0u; probe < 4u; ++probe) {
            MRArticulatedPointImpulseGPU point{};
            point.bodyIndex = body;
            point.localPoint = probe == 0u
                ? mr_float4{0.0f, 0.0f, 0.0f, 0.0f}
                : (probe == 1u
                    ? mr_float4{1.0f, 0.0f, 0.0f, 0.0f}
                    : (probe == 2u
                        ? mr_float4{0.0f, 1.0f, 0.0f, 0.0f}
                        : mr_float4{0.0f, 0.0f, 1.0f, 0.0f}));
            result.points.push_back(point);
        }
    }
    return result;
}

std::vector<float> packMetalConfiguration(
    const std::span<const double> configuration
) {
    std::vector<float> result;
    result.reserve(configuration.size());
    for (const double coordinate : configuration) {
        require(
            std::isfinite(coordinate) &&
                coordinate >= -static_cast<double>(
                    std::numeric_limits<float>::max()
                ) &&
                coordinate <= static_cast<double>(
                    std::numeric_limits<float>::max()
                ),
            "MyoSim visual configuration is not representable on Metal"
        );
        result.push_back(static_cast<float>(coordinate));
    }
    return result;
}

struct MetalMujocoForceStep {
    std::vector<float> generalizedForce;
    std::uint32_t appliedWrapCount = 0u;
    double elapsedMilliseconds = 0.0;
    std::string deviceName;
};

MetalMujocoForceStep evaluateMetalMujocoForce(
    const metalrobo::EngineModel& model,
    const LoadedMuscles& muscles,
    const MetalMujocoVisualQueries& queries,
    const std::span<const double> configuration,
    std::vector<MRMujocoMuscleStateGPU>& states,
    metalrobo::MetalArticulatedOperatorContext& context
) {
    const std::vector<float> q = packMetalConfiguration(configuration);
    const metalrobo::MetalArticulatedOperatorInput input{
        .articulationIndex = 0u,
        .environmentCount = 1u,
        .pointCount = queries.points.size(),
        .q = q,
        .points = queries.points,
        .mujoco = {
            .muscles = muscles.gpuMuscles,
            .states = states,
            .sites = muscles.gpuSites,
            .wraps = muscles.gpuWraps,
            .routeNodes = muscles.gpuRoutes,
            .bodyJacobianPointOffset = queries.bodyJacobianPointOffset,
        },
    };
    metalrobo::MetalArticulatedOperatorResult result;
    const auto diagnostics = context.run(model, input, result);
    require(
        diagnostics.succeeded() && diagnostics.dispatched &&
            diagnostics.published &&
            diagnostics.successfulEnvironmentCount == 1u &&
            diagnostics.failedEnvironmentCount == 0u &&
            result.mujocoResults.size() == muscles.gpuMuscles.size() &&
            result.mujocoActivationStates.size() == states.size() &&
            result.mujocoGeneralizedForces.size() == model.world.nv,
        "MyoSim Metal full-body force transaction failed: " +
            diagnostics.message
    );
    MetalMujocoForceStep output;
    output.generalizedForce = std::move(result.mujocoGeneralizedForces);
    output.elapsedMilliseconds = diagnostics.elapsedMilliseconds;
    output.deviceName = diagnostics.deviceName;
    for (const MRMujocoMuscleResultGPU& muscle : result.mujocoResults) {
        output.appliedWrapCount += muscle.appliedWrapCount;
    }
    states = std::move(result.mujocoActivationStates);
    return output;
}

std::array<double, 3u> crossProduct(
    const std::array<double, 3u>& left,
    const std::array<double, 3u>& right
) {
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}

std::array<double, 3u> normalizedVector(
    const std::array<double, 3u>& value,
    const char* context
) {
    const double length = std::sqrt(
        value[0] * value[0] + value[1] * value[1] + value[2] * value[2]
    );
    require(std::isfinite(length) && length > 1.0e-12,
            std::string(context) + " is degenerate");
    return {value[0] / length, value[1] / length, value[2] / length};
}

struct GroundAlignedSupport {
    std::vector<double> q;
    std::uint32_t witnessCount = 0u;
    double seedTranslationMeters = 0.0;
};

GroundAlignedSupport makeGroundAlignedSupport(
    const metalrobo::EngineModel& model,
    const LoadedSupportContacts& support
) {
    require(model.articulations.size() == 1u &&
                model.articulations.front().rootType == MR_ROOT_FLOATING,
            "MyoSim support contact requires one floating articulation");
    GroundAlignedSupport result;
    result.witnessCount = support.header.contactCount;
    result.q.assign(model.defaultQ.begin(), model.defaultQ.end());
    const MRArticulationGPU& articulation = model.articulations.front();
    require(articulation.qOffset + 3u <= result.q.size(),
            "MyoSim floating root position is unavailable for support alignment");
    const std::array<double, 3u> groundNormal = normalizedVector({
        support.header.groundNormalX,
        support.header.groundNormalY,
        support.header.groundNormalZ,
    }, "MyoSim source ground normal");
    double minimumGap = std::numeric_limits<double>::infinity();
    for (const SupportContactRecord& record : support.records) {
        minimumGap = std::min(
            minimumGap, static_cast<double>(record.defaultSignedPlaneDistance)
        );
    }
    require(std::isfinite(minimumGap) && minimumGap >= -1.0e-4 &&
                minimumGap <= 0.25,
            "MyoSim source support witnesses cannot form a bounded ground-aligned seed");
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
        result.q[articulation.qOffset + axis] -= groundNormal[axis] * minimumGap;
    }
    result.seedTranslationMeters = minimumGap;

    return result;
}

struct DynamicSourceSupportContacts {
    std::vector<metalrobo::MultiArticulatedIslandContact> contacts;
    std::vector<std::size_t> sourceRecordIndices;
    double minimumPlaneGapMeters = std::numeric_limits<double>::infinity();
};

DynamicSourceSupportContacts makeDynamicSourceSupportContacts(
    const metalrobo::EngineModel& model,
    const LoadedSupportContacts& support,
    const metalrobo::ArticulatedDynamicsConfig& dynamicsConfig,
    const std::span<const double> q,
    const std::span<const double> velocity,
    const std::span<const double> warmImpulses
) {
    require(model.articulations.size() == 1u && q.size() == model.world.nq &&
                velocity.size() == model.world.nv &&
                warmImpulses.size() == 3u * support.records.size(),
            "MyoSim dynamic source-support dimensions are inconsistent");
    const std::array<double, 3u> groundPoint{
        support.header.groundPointX, support.header.groundPointY, support.header.groundPointZ,
    };
    const std::array<double, 3u> groundNormal = normalizedVector({
        support.header.groundNormalX, support.header.groundNormalY, support.header.groundNormalZ,
    }, "MyoSim source ground normal");
    const std::array<double, 3u> contactNormal{
        -groundNormal[0], -groundNormal[1], -groundNormal[2],
    };
    const std::array<double, 3u> tangentReference =
        std::abs(contactNormal[0]) < 0.9
        ? std::array<double, 3u>{1.0, 0.0, 0.0}
        : std::array<double, 3u>{0.0, 1.0, 0.0};
    const double normalReferenceDot =
        contactNormal[0] * tangentReference[0] +
        contactNormal[1] * tangentReference[1] +
        contactNormal[2] * tangentReference[2];
    const std::array<double, 3u> tangentU = normalizedVector({
        tangentReference[0] - normalReferenceDot * contactNormal[0],
        tangentReference[1] - normalReferenceDot * contactNormal[1],
        tangentReference[2] - normalReferenceDot * contactNormal[2],
    }, "MyoSim dynamic support tangent");
    const std::array<double, 3u> tangentV = crossProduct(contactNormal, tangentU);
    std::vector<metalrobo::ArticulatedPointQuery> queries;
    queries.reserve(support.records.size());
    for (const SupportContactRecord& record : support.records) {
        queries.push_back({
            record.bodyIndex, {record.localPointX, record.localPointY, record.localPointZ},
        });
    }
    std::vector<metalrobo::ArticulatedPointKinematics> points(queries.size());
    std::vector<double> jacobians(queries.size() * 3u * model.world.nv, 0.0);
    const auto kinematics = metalrobo::computeArticulatedPointJacobians(
        model, 0u, q, velocity, queries, points, jacobians, dynamicsConfig
    );
    require(kinematics.succeeded(),
            "MyoSim dynamic source-support point kinematics failed");
    DynamicSourceSupportContacts result;
    constexpr double kActivationDistanceMeters = 0.002;
    for (std::size_t index = 0u; index < support.records.size(); ++index) {
        const SupportContactRecord& record = support.records[index];
        const std::array<double, 3u>& position = points[index].position;
        const double gap =
            (position[0] - groundPoint[0]) * groundNormal[0] +
            (position[1] - groundPoint[1]) * groundNormal[1] +
            (position[2] - groundPoint[2]) * groundNormal[2];
        require(std::isfinite(gap), "MyoSim dynamic source-support plane gap is non-finite");
        result.minimumPlaneGapMeters = std::min(result.minimumPlaneGapMeters, gap);
        if (gap > kActivationDistanceMeters) continue;
        require(record.friction > 0.0f,
                "MyoSim authored support contact has no tangential friction");
        const double normalRecoveryVelocity = gap < 0.0
            ? std::min(-gap / dynamicsConfig.timestep, 2.0)
            : 0.0;
        result.contacts.push_back({
            .endpointA = {
                metalrobo::MultiContactEndpointKind::articulatedBody,
                record.bodyIndex,
                {record.localPointX, record.localPointY, record.localPointZ},
            },
            // This is reprojected every step so the source witness remains a
            // unilateral plane contact rather than becoming a hidden fixed
            // foot weld at its initial world-space tangent coordinates.
            .endpointB = {
                metalrobo::MultiContactEndpointKind::staticWorld,
                MR_INVALID_INDEX,
                {
                    position[0] - gap * groundNormal[0],
                    position[1] - gap * groundNormal[1],
                    position[2] - gap * groundNormal[2],
                },
            },
            .normal = contactNormal,
            .tangentU = tangentU,
            .tangentV = tangentV,
            .targetVelocity = {normalRecoveryVelocity, 0.0, 0.0},
            .regularization = {1.0e-8, 1.0e-8, 1.0e-8},
            .warmImpulse = {
                warmImpulses[3u * index], warmImpulses[3u * index + 1u],
                warmImpulses[3u * index + 2u],
            },
            .friction = record.friction,
        });
        result.sourceRecordIndices.push_back(index);
    }
    return result;
}

MuscleDrivenVisualState integrateMuscleDrivenVisualState(
    const metalrobo::EngineModel& model,
    const LoadedMuscles& muscles,
    const double timestepSeconds,
    const std::uint32_t stepCount,
    const double activation,
    const LoadedSupportContacts* supportContacts
) {
    require(std::isfinite(timestepSeconds) &&
                timestepSeconds >= 1.0e-6 && timestepSeconds <= 1.0e-3,
            "muscle-driven visual step must be between 1 us and 1 ms");
    require(model.world.nv > 0u && model.defaultQ.size() == model.world.nq &&
                model.defaultV.size() == model.world.nv &&
                muscles.referenceMuscles.size() == muscles.muscles.size() &&
                muscles.gpuMuscles.size() == muscles.muscles.size() &&
                muscles.gpuSites.size() == muscles.sites.size() &&
                muscles.gpuWraps.size() == muscles.wraps.size() &&
                muscles.gpuRoutes.size() == muscles.routes.size(),
            "muscle-driven visual state has inconsistent MyoSim dimensions");
    require(std::isfinite(activation) && activation >= 0.0 && activation <= 1.0,
            "muscle-driven visual activation must be within [0, 1]");
    require(stepCount >= 1u && stepCount <= 64u,
            "muscle-driven visual step count must be in [1, 64]");

    std::vector<double> initialQ(model.defaultQ.begin(), model.defaultQ.end());
    const std::vector<double> initialV(model.defaultV.begin(), model.defaultV.end());
    std::optional<GroundAlignedSupport> support;
    if (supportContacts != nullptr) {
        support.emplace(makeGroundAlignedSupport(model, *supportContacts));
        initialQ = support->q;
    }
    MuscleDrivenVisualState result;
    result.stepCount = stepCount;
    metalrobo::ArticulatedDynamicsConfig dynamicsConfig;
    dynamicsConfig.timestep = timestepSeconds;
    std::vector<double> passiveQ = initialQ;
    std::vector<double> passiveV = initialV;
    std::vector<double> activeQ = initialQ;
    std::vector<double> activeV = initialV;
    const std::vector<double> zeroForce(model.world.nv, 0.0);
    const MetalMujocoVisualQueries metalQueries =
        makeMetalMujocoVisualQueries(model);
    std::vector<MRMujocoMuscleStateGPU> activeMuscleStates(
        muscles.gpuMuscles.size()
    );
    std::vector<MRMujocoMuscleStateGPU> passiveMuscleStates(
        muscles.gpuMuscles.size()
    );
    for (MRMujocoMuscleStateGPU& state : activeMuscleStates) {
        state.excitationAndActivation = {
            static_cast<float>(activation),
            static_cast<float>(activation),
            0.0f,
            0.0f,
        };
    }
    metalrobo::MetalArticulatedOperatorConfig activeMetalConfig{
        .pointJacobiansOnly = true,
        .mujocoActivationTimestepSeconds =
            static_cast<float>(timestepSeconds),
    };
    metalrobo::MetalArticulatedOperatorConfig passiveMetalConfig{
        .pointJacobiansOnly = true,
    };
    metalrobo::MetalArticulatedOperatorContext activeMetalContext(
        activeMetalConfig
    );
    metalrobo::MetalArticulatedOperatorContext passiveMetalContext(
        passiveMetalConfig
    );
    std::vector<double> passiveSupportWarm;
    std::vector<double> activeSupportWarm;
    if (supportContacts != nullptr) {
        passiveSupportWarm.assign(3u * supportContacts->records.size(), 0.0);
        activeSupportWarm.assign(3u * supportContacts->records.size(), 0.0);
        metalrobo::CompiledMetalMultiArticulatedContactProgram program;
        const auto compiled = metalrobo::compileMetalMultiArticulatedContactProgram(model, program);
        require(
            !compiled.succeeded() &&
                (compiled.status == metalrobo::MetalMultiArticulatedContactStatus::capacityOverflow ||
                 compiled.status == metalrobo::MetalMultiArticulatedContactStatus::unsupportedTopology),
            "MyoSim dynamic source-support Metal admission unexpectedly changed: " + compiled.message
        );
        result.supportDeviceName = "not_admitted";
        result.supportMetalStatus = "not_admitted_articulation_exceeds_metal_contact_bucket";
        result.supportWitnessCount = supportContacts->header.contactCount;
        result.supportSeedTranslationMeters = support->seedTranslationMeters;
    }
    metalrobo::QualityContactSolverConfig contactConfig;
    contactConfig.maximumIterations = 300u;
    contactConfig.kktTolerance = 1.0e-10;
    const auto applyDynamicSupport = [&] (
        std::vector<double>& q,
        std::vector<double>& velocity,
        std::vector<double>& warmImpulses,
        const bool isActive
    ) {
        const DynamicSourceSupportContacts contacts = makeDynamicSourceSupportContacts(
            model, *supportContacts, dynamicsConfig, q, velocity, warmImpulses
        );
        result.minimumSupportPlaneGapMeters = std::min(
            result.minimumSupportPlaneGapMeters, contacts.minimumPlaneGapMeters
        );
        if (isActive) {
            result.activeSupportContactCount = static_cast<std::uint32_t>(contacts.contacts.size());
            result.maximumActiveSupportContactCount = std::max(
                result.maximumActiveSupportContactCount, result.activeSupportContactCount
            );
        }
        if (contacts.contacts.empty()) return;
        metalrobo::MultiArticulatedContactProblem problem;
        const auto build = metalrobo::buildMultiArticulatedIslandContactProblem(
            model, q, velocity, {}, contacts.contacts, problem, dynamicsConfig
        );
        require(build.succeeded(), "MyoSim dynamic source-support FP64 contact construction failed");
        metalrobo::MultiArticulatedContactSolution solution;
        const auto solve = metalrobo::solveMultiArticulatedContactProblem(
            problem, solution, contactConfig
        );
        require(solve.succeeded() && solution.articulatedVelocity.size() == model.world.nv &&
                    solution.impulses.size() == 3u * contacts.contacts.size(),
                "MyoSim dynamic source-support FP64 contact solve failed");
        velocity = std::move(solution.articulatedVelocity);
        for (std::size_t contact = 0u; contact < contacts.sourceRecordIndices.size(); ++contact) {
            const std::size_t record = contacts.sourceRecordIndices[contact];
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                warmImpulses[3u * record + axis] = solution.impulses[3u * contact + axis];
            }
        }
        result.supportContactApplied = true;
    };
    for (std::uint32_t step = 0u; step < stepCount; ++step) {
        const MetalMujocoForceStep activatedForce = evaluateMetalMujocoForce(
            model,
            muscles,
            metalQueries,
            activeQ,
            activeMuscleStates,
            activeMetalContext
        );
        const MetalMujocoForceStep sourceDefaultPassiveForce =
            evaluateMetalMujocoForce(
                model,
                muscles,
                metalQueries,
                activeQ,
                passiveMuscleStates,
                passiveMetalContext
            );
        require(
            activatedForce.generalizedForce.size() == model.world.nv &&
                sourceDefaultPassiveForce.generalizedForce.size() ==
                    model.world.nv &&
                activatedForce.deviceName == sourceDefaultPassiveForce.deviceName,
            "MyoSim Metal force transactions returned incompatible outputs"
        );
        result.muscleMetalStepCount += 2u;
        result.muscleMetalForceRecordCount += static_cast<std::uint32_t>(
            muscles.gpuMuscles.size()
        );
        result.muscleMetalElapsedMilliseconds +=
            activatedForce.elapsedMilliseconds +
            sourceDefaultPassiveForce.elapsedMilliseconds;
        result.muscleMetalDeviceName = activatedForce.deviceName;
        result.appliedWrapCount += activatedForce.appliedWrapCount;
        std::vector<double> muscleForce(model.world.nv, 0.0);
        for (std::size_t index = 0u; index < muscleForce.size(); ++index) {
            muscleForce[index] = static_cast<double>(
                activatedForce.generalizedForce[index]
            ) - static_cast<double>(
                sourceDefaultPassiveForce.generalizedForce[index]
            );
        }
        require(std::all_of(muscleForce.begin(), muscleForce.end(), [](const double value) {
                    return std::isfinite(value);
                }),
                "MyoSim muscle force projection returned a non-finite generalized force");
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
        if (supportContacts != nullptr) {
            applyDynamicSupport(passiveQ, passiveV, passiveSupportWarm, false);
            applyDynamicSupport(activeQ, activeV, activeSupportWarm, true);
        }
    }
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
    if (supportContacts != nullptr) {
        require(std::isfinite(result.minimumSupportPlaneGapMeters),
                "MyoSim dynamic source-support did not evaluate a source plane gap");
    }
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
        // Surface registration is a presentation-only cue.  The source route
        // keeps its authored site records and force evaluation untouched.
        bool surfaceProjected = false;
        mr_float4 surfaceNormalWorld{};
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
                false,
                {},
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

mr_float4 normalizedDirection(const mr_float4 value, const char* context) {
    const float lengthSquared = dotPoint(value, value);
    require(std::isfinite(lengthSquared) && lengthSquared > 1.0e-10f,
            std::string(context) + " is degenerate");
    return scalePoint(value, 1.0f / std::sqrt(lengthSquared));
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

mr_float4 boneVertexNormalWorld(
    const BoneRecord& bone,
    const BoneVertex& vertex,
    const MRBodyStateGPU& body
) {
    const mr_float4 boneRotation{
        bone.quaternionX, bone.quaternionY, bone.quaternionZ, bone.quaternionW,
    };
    mr_float4 normal = rotatePoint(
        body.orientation,
        rotatePoint(boneRotation, {vertex.normalX, vertex.normalY, vertex.normalZ, 0.0f})
    );
    const float length = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    require(length > 1.0e-6f, "BodyParts3D attachment triangle has a degenerate normal");
    normal.x /= length;
    normal.y /= length;
    normal.z /= length;
    normal.w = 0.0f;
    return normal;
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
            mr_float4 closestNormal{};
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
                        const mr_float4 firstNormal = boneVertexNormalWorld(
                            bone, bones.vertices[first], bodies[bone.bodyIndex]
                        );
                        const mr_float4 secondNormal = boneVertexNormalWorld(
                            bone, bones.vertices[second], bodies[bone.bodyIndex]
                        );
                        const mr_float4 thirdNormal = boneVertexNormalWorld(
                            bone, bones.vertices[third], bodies[bone.bodyIndex]
                        );
                        closestNormal = {
                            firstNormal.x + secondNormal.x + thirdNormal.x,
                            firstNormal.y + secondNormal.y + thirdNormal.y,
                            firstNormal.z + secondNormal.z + thirdNormal.z,
                            0.0f,
                        };
                        const float normalLength = std::sqrt(
                            closestNormal.x * closestNormal.x +
                            closestNormal.y * closestNormal.y +
                            closestNormal.z * closestNormal.z
                        );
                        require(normalLength > 1.0e-6f,
                                "BodyParts3D attachment triangle averaged normal is degenerate");
                        closestNormal.x /= normalLength;
                        closestNormal.y /= normalLength;
                        closestNormal.z /= normalLength;
                    }
                }
            }
            if (closestDistanceSquared <= maximumProjectionDistanceSquared) {
                point.world = closest;
                point.surfaceProjected = true;
                point.surfaceNormalWorld = closestNormal;
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
    // ``position`` and ``target`` are calculated from the posed, rendered
    // geometry in world coordinates.  Parenting that world-space pose to the
    // articulated asset applies the root transform a second time, which makes
    // oblique anatomy reviews look off-centre and distant.  Keep the semantic
    // asset id for world validation, but make the camera genuinely world
    // anchored.
    camera.parentKind = MR_WORLD_SENSOR_PARENT_WORLD;
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

MRVisualMaterialGPUV2 makeMaterial(
    const mr_float4 color,
    const mr_float4 emission,
    const float roughness = 0.55f,
    const float clearcoat = 0.0f
) {
    MRVisualMaterialGPUV2 material{};
    material.baseColorAndOpacity = color;
    material.emissionAndStrength = emission;
    material.surface = {roughness, 0.02f, 1.0f, 1.0f};
    material.coatingAndAlphaCutoff = {clearcoat, 0.22f, 1.0f, 0.5f};
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

metalrobo::VisualLightRigV1 makeHumanAnatomyLightRig(
    const mr_float4 center,
    const mr_float4 cameraPosition,
    const float cameraDistance
) {
    // Use a camera-relative three-point rig.  The source meshes must remain
    // identical in every view, but a fixed world-space key left posterior and
    // lateral inspections underlit and made the tendon-to-bone interface read
    // as a flat colour boundary.  These broad, neutral softboxes present the
    // same source normals and triangles from each inspected camera direction.
    const float distance = std::max(cameraDistance, 0.35f);
    const float intensityScale = distance * distance;
    const mr_float4 view = normalizedDirection(
        subtractPoint(cameraPosition, center), "Human anatomy camera direction"
    );
    const mr_float4 target = addPoint(center, {0.0f, 0.0f, 0.04f * distance, 0.0f});
    const auto makeAreaLight = [&target, distance, intensityScale](
        const mr_float4 position,
        const mr_float4 color,
        const float intensity,
        const float width,
        const float height,
        const std::uint32_t stableId
    ) {
        MRVisualLightGPUV1 light{};
        light.positionAndRange = {position.x, position.y, position.z, 20.0f};
        const mr_float4 direction = normalizedDirection(
            subtractPoint(target, position), "Human anatomy softbox direction"
        );
        light.directionAndSpot = {direction.x, direction.y, direction.z, -1.0f};
        light.colorAndIntensity = {color.x, color.y, color.z, intensity * intensityScale};
        light.shape = {width * distance, height * distance, -1.0f, 0.08f};
        light.shadow = {1u, 0u, 0u, 0u};
        light.identity = {MR_VISUAL_LIGHT_RECTANGLE, MR_VISUAL_LIGHT_UNIT_NIT, 0u, stableId};
        return light;
    };

    const mr_float4 keyPosition = addPoint(
        cameraPosition, {0.24f * distance, -0.13f * distance, 0.31f * distance, 0.0f}
    );
    const mr_float4 fillPosition = addPoint(
        cameraPosition, {-0.32f * distance, 0.18f * distance, 0.10f * distance, 0.0f}
    );
    const mr_float4 rimPosition = addPoint(
        addPoint(center, scalePoint(view, -0.82f * distance)),
        {0.0f, 0.08f * distance, 0.42f * distance, 0.0f}
    );
    metalrobo::VisualLightRigV1 result;
    result.id = "human_anatomy_camera_relative_three_point";
    result.contentHash = "builtin:human-anatomy-camera-relative-three-point-v3";
    result.lights = {
        makeAreaLight(keyPosition, {1.0f, 0.99f, 0.96f, 0.0f}, 150.0f, 0.88f, 0.70f, 201u),
        makeAreaLight(fillPosition, {0.82f, 0.88f, 1.0f, 0.0f}, 54.0f, 0.96f, 0.78f, 202u),
        makeAreaLight(rimPosition, {1.0f, 0.95f, 0.86f, 0.0f}, 92.0f, 0.72f, 0.58f, 203u),
    };
    return result;
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

mr_float4 normalTangent(const mr_float4 normal) {
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

mr_float4 boneTangent(const BoneVertex& vertex) {
    return normalTangent({vertex.normalX, vertex.normalY, vertex.normalZ, 1.0f});
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

mr_float4 softTissueVertexWorld(
    const SoftTissueRecord& tissue,
    const SoftTissueVertex& vertex,
    const MRBodyStateGPU& body,
    const bool primary
) {
    const mr_float4 localRotation = primary
        ? mr_float4{
            tissue.primaryQuaternionX, tissue.primaryQuaternionY,
            tissue.primaryQuaternionZ, tissue.primaryQuaternionW,
        }
        : mr_float4{
            tissue.secondaryQuaternionX, tissue.secondaryQuaternionY,
            tissue.secondaryQuaternionZ, tissue.secondaryQuaternionW,
        };
    const mr_float4 localTranslation = primary
        ? mr_float4{
            tissue.primaryTranslationX, tissue.primaryTranslationY,
            tissue.primaryTranslationZ, 0.0f,
        }
        : mr_float4{
            tissue.secondaryTranslationX, tissue.secondaryTranslationY,
            tissue.secondaryTranslationZ, 0.0f,
        };
    const float localScale = primary
        ? tissue.primaryUniformScale : tissue.secondaryUniformScale;
    const mr_float4 local = addPoint(
        localTranslation,
        scalePoint(
            rotatePoint(localRotation, {vertex.positionX, vertex.positionY, vertex.positionZ, 0.0f}),
            localScale
        )
    );
    return addPoint(body.position, rotatePoint(body.orientation, local));
}

mr_float4 softTissueVertexNormalWorld(
    const SoftTissueRecord& tissue,
    const SoftTissueVertex& vertex,
    const MRBodyStateGPU& body,
    const bool primary
) {
    const mr_float4 localRotation = primary
        ? mr_float4{
            tissue.primaryQuaternionX, tissue.primaryQuaternionY,
            tissue.primaryQuaternionZ, tissue.primaryQuaternionW,
        }
        : mr_float4{
            tissue.secondaryQuaternionX, tissue.secondaryQuaternionY,
            tissue.secondaryQuaternionZ, tissue.secondaryQuaternionW,
        };
    mr_float4 normal = rotatePoint(
        body.orientation,
        rotatePoint(localRotation, {vertex.normalX, vertex.normalY, vertex.normalZ, 0.0f})
    );
    const float length = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    require(length > 1.0e-6f, "BodyParts3D soft-tissue normal is degenerate");
    normal.x /= length;
    normal.y /= length;
    normal.z /= length;
    normal.w = 0.0f;
    return normal;
}

GeometryRange appendSoftTissueGeometry(
    metalrobo::VisualAssetPackV2& pack,
    const LoadedSoftTissues& tissues,
    const SoftTissueRecord& tissue,
    const std::span<const MRBodyStateGPU> bodies
) {
    require(tissue.primaryBodyIndex < bodies.size() && tissue.secondaryBodyIndex < bodies.size(),
            "BodyParts3D soft-tissue body binding exceeds the rendered pose");
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
    for (std::uint32_t offset = 0u; offset < tissue.vertexCount; ++offset) {
        const SoftTissueVertex& source = tissues.vertices[tissue.firstVertex + offset];
        const float primaryWeight = source.primaryWeight;
        const float secondaryWeight = 1.0f - primaryWeight;
        const mr_float4 primaryPosition = softTissueVertexWorld(
            tissue, source, bodies[tissue.primaryBodyIndex], true
        );
        const mr_float4 secondaryPosition = softTissueVertexWorld(
            tissue, source, bodies[tissue.secondaryBodyIndex], false
        );
        const mr_float4 position = addPoint(
            scalePoint(primaryPosition, primaryWeight),
            scalePoint(secondaryPosition, secondaryWeight)
        );
        const mr_float4 primaryNormal = softTissueVertexNormalWorld(
            tissue, source, bodies[tissue.primaryBodyIndex], true
        );
        const mr_float4 secondaryNormal = softTissueVertexNormalWorld(
            tissue, source, bodies[tissue.secondaryBodyIndex], false
        );
        mr_float4 normal{
            primaryNormal.x * primaryWeight + secondaryNormal.x * secondaryWeight,
            primaryNormal.y * primaryWeight + secondaryNormal.y * secondaryWeight,
            primaryNormal.z * primaryWeight + secondaryNormal.z * secondaryWeight,
            0.0f,
        };
        const float normalLength = std::sqrt(
            normal.x * normal.x + normal.y * normal.y + normal.z * normal.z
        );
        require(normalLength > 1.0e-6f, "BodyParts3D blended soft-tissue normal is degenerate");
        normal.x /= normalLength;
        normal.y /= normalLength;
        normal.z /= normalLength;
        normal.w = 1.0f;
        pack.vertices.push_back({
            position,
            normal,
            normalTangent(normal),
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
    for (std::uint32_t offset = 0u; offset < tissue.indexCount; ++offset) {
        pack.indices.push_back(
            vertexBase + tissues.indices[tissue.firstIndex + offset] - tissue.firstVertex
        );
    }
    result.indexCount = tissue.indexCount;
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
    const LoadedSoftTissues* softTissuePayload,
    const std::span<const MRBodyStateGPU> bodies,
    const bool muscleDriven,
    const std::span<const std::uint32_t> requestedBoneBodyIndices,
    const std::span<const std::uint32_t> requestedSoftTissueStableIds,
    const SourceRouteCentrelines* sourceRouteCentrelines,
    std::uint32_t& renderedBodies,
    std::uint32_t& renderedSoftTissues,
    std::uint32_t& renderedRouteSegments
) {
    metalrobo::VisualAssetPackV2 pack;
    pack.id = bonePayload != nullptr
        ? "myosim_fullbody_articulated_bodyparts_bones_view"
        : "myosim_fullbody_articulated_marker_view";
    pack.sourceUri = bonePayload != nullptr
        ? (softTissuePayload != nullptr
            ? "numi://bodyparts3d/NHBONES1+NHTISS2+NHRIGID2+NHMYO1/articulated-anatomy-view"
            : "numi://bodyparts3d/NHBONES1+NHRIGID2+NHMYO1/articulated-bone-view")
        : "numi://myosim/NHRIGID2+NHMYO1/articulated-marker-view";
    pack.sourceContentHash = bonePayload != nullptr
        ? (softTissuePayload != nullptr
            ? "bodyparts3d-major-bones+right-posterior-chain+runtime-body-and-site-records"
            : "bodyparts3d-major-bones+runtime-body-and-site-records")
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
    if (softTissuePayload != nullptr) {
        pack.preprocessingProvenance +=
            "/exact_bodyparts3d_posterior_calf_surfaces_with_two_body_linear_blend_kinematic_binding";
    }
    if (sourceRouteCentrelines != nullptr) {
        pack.preprocessingProvenance +=
            "/cpu_fp64_mujoco_tangent_and_wrapped_arc_centreline_at_the_rendered_pose";
        if (sourceRouteCentrelines->surfaceProjectedAttachmentCount > 0u) {
            pack.preprocessingProvenance +=
                "/visual_only_nearest_bodyparts3d_triangle_attachment_projection";
        }
    }
    pack.materials.push_back(makeMaterial(
        {0.82f, 0.86f, 0.88f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, 0.52f
    ));
    pack.materials.push_back(makeMaterial(
        // Source muscle mesh detail is conveyed by real surface normals and
        // the anatomy light rig.  The former red emission lifted all shading
        // toward a flat, plastic appearance, so do not use self-illumination
        // as a substitute for an anatomical material.
        {0.56f, 0.018f, 0.014f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, 0.58f, 0.035f
    ));
    pack.materials.push_back(makeMaterial(
        // Exact force-route diagnostics must remain legible over the red
        // BodyParts3D muscle layer.  Cyan is intentionally reserved for this
        // opt-in source-route / attachment visual, never for a tendon mesh.
        {0.035f, 0.82f, 0.98f, 1.0f}, {0.0f, 0.20f, 0.34f, 0.55f}, 0.28f, 0.16f
    ));
    pack.materials.push_back(makeMaterial(
        // Keep osseous anatomy a cool, matte ivory.  It is intentionally
        // distinct from collagen so the calcaneal insertion can be judged at
        // a glance without a diagnostic outline or false geometry.
        {0.57f, 0.63f, 0.64f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, 0.62f
    ));
    pack.materials.push_back(makeMaterial(
        // Do not add red emission: it flattens the source muscle relief and
        // makes the layer look painted onto the skeleton rather than like an
        // anatomical surface.
        {0.50f, 0.022f, 0.014f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, 0.50f, 0.055f
    ));
    pack.materials.push_back(makeMaterial(
        // Tendon is warm, non-metallic collagen—not a glowing route or a gold
        // overlay.  Its deliberately separate value from bone makes the
        // source-continuous calcaneal insertion inspectable in a single frame.
        {0.91f, 0.75f, 0.53f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, 0.66f, 0.01f
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
            if (!requestedBoneBodyIndices.empty() &&
                !std::binary_search(
                    requestedBoneBodyIndices.begin(), requestedBoneBodyIndices.end(), bone.bodyIndex
                )) {
                continue;
            }
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
    renderedSoftTissues = 0u;
    if (softTissuePayload != nullptr) {
        for (const SoftTissueRecord& tissue : softTissuePayload->records) {
            if (!requestedSoftTissueStableIds.empty() &&
                !std::binary_search(
                    requestedSoftTissueStableIds.begin(),
                    requestedSoftTissueStableIds.end(), tissue.stableId
                )) {
                continue;
            }
            const GeometryRange geometry = appendSoftTissueGeometry(
                pack, *softTissuePayload, tissue, bodies
            );
            const bool isMuscle = tissue.layer == kSoftTissueLayerMuscle;
            appendInstance(
                geometry, isMuscle ? 4u : 5u,
                isMuscle ? kMuscleSurfaceSemantic : kTendonSurfaceSemantic,
                MR_VISUAL_BINDING_WORLD, MR_INVALID_INDEX,
                {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f},
                tissue.stableId
            );
            ++renderedSoftTissues;
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
                            bonePayload != nullptr ? 0.0016f : 0.0024f
                        ), 2u,
                        kRouteSemantic, MR_VISUAL_BINDING_WORLD, MR_INVALID_INDEX,
                        {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f},
                        stableRouteId++
                    );
                    ++renderedRouteSegments;
                }
            }
            // These partially intersect the registered bone surface, making
            // the resolved route's source origin/insertion visibly terminate
            // at it.  They are intentionally only endpoint cues, not a
            // fabricated tendon surface or altered force path.
            for (const std::size_t pointIndex : {std::size_t{0u}, route.points.size() - 1u}) {
                const SourceRouteCentreline::Point& point = route.points[pointIndex];
                if (!point.surfaceProjected) continue;
                constexpr float kAttachmentCapRadius = 0.0048f;
                constexpr float kAttachmentCapSurfaceOverlap = 0.0015f;
                appendInstance(
                    appendEllipsoid(
                        pack,
                        {kAttachmentCapRadius, kAttachmentCapRadius, kAttachmentCapRadius}
                    ),
                    2u, kSiteSemantic, MR_VISUAL_BINDING_WORLD, MR_INVALID_INDEX,
                    {
                        point.world.x + kAttachmentCapSurfaceOverlap * point.surfaceNormalWorld.x,
                        point.world.y + kAttachmentCapSurfaceOverlap * point.surfaceNormalWorld.y,
                        point.world.z + kAttachmentCapSurfaceOverlap * point.surfaceNormalWorld.z,
                        1.0f,
                    },
                    {0.0f, 0.0f, 0.0f, 1.0f}, stableRouteId++
                );
            }
        }
    }
    pack.contentHash = metalrobo::computeVisualAssetPackContentHash(pack);
    std::string reason;
    require(pack.valid(&reason), "native Human marker pack is invalid: " + reason);
    return pack;
}

struct CameraFraming {
    mr_float4 center{};
    float distance = 0.0f;
    float sourceExtentMeters = 0.0f;
    bool usesSourceGeometryBounds = false;
};

CameraFraming makeCameraFraming(
    const metalrobo::VisualAssetPackV2& pack,
    const std::span<const MRBodyStateGPU> bodies,
    const std::optional<std::uint32_t> focusBodyIndex
) {
    if (focusBodyIndex.has_value()) {
        require(*focusBodyIndex < bodies.size(), "MyoSim visual focus body index is out of bounds");
        // Focused inspections intentionally retain a close, deterministic
        // view.  The source-geometry bounds below are for the whole-body
        // presentation, where COM-only framing cropped the actual anatomy.
        const MRBodyStateGPU& focus = bodies[*focusBodyIndex];
        return {
            .center = {focus.position.x, focus.position.y, focus.position.z, 0.0f},
            .distance = 0.70f,
        };
    }

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
    double centroidX = 0.0;
    double centroidY = 0.0;
    double centroidZ = 0.0;
    std::size_t centroidVertexCount = 0u;
    std::size_t includedPrimitiveCount = 0u;
    const auto include = [&minimum, &maximum](const mr_float4 point) {
        require(std::isfinite(point.x) && std::isfinite(point.y) &&
                    std::isfinite(point.z),
                "MyoSim visual source geometry has a non-finite world bound");
        minimum.x = std::min(minimum.x, point.x);
        minimum.y = std::min(minimum.y, point.y);
        minimum.z = std::min(minimum.z, point.z);
        maximum.x = std::max(maximum.x, point.x);
        maximum.y = std::max(maximum.y, point.y);
        maximum.z = std::max(maximum.z, point.z);
    };
    for (std::size_t instanceIndex = 0u; instanceIndex < pack.instances.size(); ++instanceIndex) {
        const MRVisualInstanceGPUV2& instance = pack.instances[instanceIndex];
        require(instance.geometry.x <= pack.primitives.size() &&
                    instance.geometry.y <= pack.primitives.size() - instance.geometry.x &&
                    std::isfinite(instance.translationAndScale.w) &&
                    instance.translationAndScale.w > 0.0f,
                "MyoSim visual instance has an invalid framing range");
        const bool articulated =
            instance.binding.z == MR_VISUAL_BINDING_ARTICULATED_LINK;
        require(articulated || instance.binding.z == MR_VISUAL_BINDING_WORLD,
                "MyoSim visual framing only supports world or articulated-link bindings");
        const MRBodyStateGPU* body = nullptr;
        if (articulated) {
            require(instance.binding.y < bodies.size(),
                    "MyoSim visual framing articulated binding is out of bounds");
            body = &bodies[instance.binding.y];
        }
        const auto worldPoint = [&instance, body](const mr_float4 local) {
            const mr_float4 instancePoint = addPoint(
                instance.translationAndScale,
                rotatePoint(
                    instance.orientation,
                    scalePoint(local, instance.translationAndScale.w)
                )
            );
            return body == nullptr
                ? instancePoint
                : addPoint(body->position, rotatePoint(body->orientation, instancePoint));
        };
        for (std::uint32_t primitiveOffset = 0u;
             primitiveOffset < instance.geometry.y; ++primitiveOffset) {
            const MRVisualPrimitiveGPUV2& primitive =
                pack.primitives[instance.geometry.x + primitiveOffset];
            require(primitive.geometry.w == instanceIndex,
                    "MyoSim visual framing primitive/instance identity drifted");
            require(primitive.geometry.x <= pack.indices.size() &&
                        primitive.geometry.y <= pack.indices.size() - primitive.geometry.x,
                    "MyoSim visual framing primitive index range is invalid");
            // Use the actual rendered vertices for both bounds and target.
            // An AABB midpoint can land in empty space for an asymmetric
            // oblique anatomy view, which is why the prior frame showed most
            // of the body in one corner despite nominally correct bounds.
            for (std::uint32_t indexOffset = 0u;
                 indexOffset < primitive.geometry.y; ++indexOffset) {
                const std::uint32_t vertexIndex =
                    pack.indices[primitive.geometry.x + indexOffset];
                require(vertexIndex < pack.vertices.size(),
                        "MyoSim visual framing primitive references an invalid vertex");
                const mr_float4 point = worldPoint(pack.vertices[vertexIndex].position);
                include(point);
                centroidX += static_cast<double>(point.x);
                centroidY += static_cast<double>(point.y);
                centroidZ += static_cast<double>(point.z);
                ++centroidVertexCount;
            }
            ++includedPrimitiveCount;
        }
    }
    require(includedPrimitiveCount > 0u,
            "MyoSim visual source geometry has no primitives for whole-body framing");
    require(centroidVertexCount > 0u,
            "MyoSim visual source geometry has no vertices for whole-body framing");
    const float extent = std::max({
        maximum.x - minimum.x, maximum.y - minimum.y, maximum.z - minimum.z,
    });
    require(std::isfinite(extent) && extent > 1.0e-4f,
            "MyoSim visual source geometry has a degenerate whole-body bound");
    // The camera's vertical FOV at the reference focal length has a half-angle
    // of about 34 degrees.  This stand-off leaves a visible margin around the
    // exact imported mesh in every four-camera angle, without turning the
    // whole-body anatomy review into a distant thumbnail.
    return {
        .center = {
            static_cast<float>(centroidX / static_cast<double>(centroidVertexCount)),
            static_cast<float>(centroidY / static_cast<double>(centroidVertexCount)),
            static_cast<float>(centroidZ / static_cast<double>(centroidVertexCount)),
            0.0f,
        },
        // The old global 1.85 m lower bound was appropriate for a 1.7 m
        // whole-body specimen but reduced a selected ankle or wrist surface
        // to a thumbnail.  Retain the full-body stand-off through its actual
        // extent, while allowing a filtered anatomical insertion inspection
        // to fill the frame with a conservative 0.25 m lower bound.
        .distance = std::max(1.08f * extent, 0.25f),
        .sourceExtentMeters = extent,
        .usesSourceGeometryBounds = true,
    };
}

std::array<mr_float4, 4u> cameraPositions(
    const CameraFraming& framing
) {
    const mr_float4 center = framing.center;
    const float distance = framing.distance;
    return {{
        {center.x, center.y - distance, center.z + 0.10f * distance, 0.0f},
        {
            center.x + 0.72f * distance, center.y - 0.72f * distance,
            center.z + 0.16f * distance, 0.0f,
        },
        {center.x + distance, center.y, center.z + 0.16f * distance, 0.0f},
        {center.x, center.y + distance, center.z + 0.10f * distance, 0.0f},
    }};
}

metalrobo::WorldTemplate makeWorld(
    const metalrobo::EngineModel& model,
    const CameraFraming& framing,
    const std::uint32_t dimension,
    std::array<std::string, 4u>& cameraNames
) {
    const mr_float4 center = framing.center;
    const std::array<mr_float4, 4u> positions = cameraPositions(framing);
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
        makeCamera(cameraNames[0], positions[0], center, dimension),
        makeCamera(cameraNames[1], positions[1], center, dimension),
        makeCamera(cameraNames[2], positions[2], center, dimension),
        makeCamera(cameraNames[3], positions[3], center, dimension),
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

double parseMuscleActivation(const std::string& value) {
    std::size_t parsed = 0u;
    double result = 0.0;
    try {
        result = std::stod(value, &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("--muscle-activation must be a finite decimal from 0 through 1");
    }
    require(parsed == value.size() && std::isfinite(result) && result >= 0.0 && result <= 1.0,
            "--muscle-activation must be a finite decimal from 0 through 1");
    return result;
}

std::uint32_t parseMuscleStepCount(const std::string& value) {
    std::size_t parsed = 0u;
    unsigned long result = 0ul;
    try {
        result = std::stoul(value, &parsed, 10);
    } catch (const std::exception&) {
        throw std::runtime_error("--muscle-step-count must be an integer from 1 through 64");
    }
    require(parsed == value.size() && result >= 1ul && result <= 64ul,
            "--muscle-step-count must be an integer from 1 through 64");
    return static_cast<std::uint32_t>(result);
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
            std::optional<double> muscleActivation;
            std::optional<std::uint32_t> muscleStepCount;
            bool sourceRouteCentrelines = false;
            bool surfaceProjectSourceSites = false;
            std::vector<std::uint32_t> requestedSourceRouteMuscles;
            std::vector<std::uint32_t> requestedBoneBodyIndices;
            std::vector<std::uint32_t> requestedSoftTissueStableIds;
            std::optional<std::uint32_t> focusBodyIndex;
            std::optional<std::filesystem::path> softTissuePayloadPath;
            std::optional<std::filesystem::path> supportContactPayloadPath;
            std::uint32_t frameDimension = kDefaultFrameDimension;
            std::vector<std::string> positional;
            for (int index = 1; index < argc; ++index) {
                const std::string argument{argv[index]};
                if (argument == "--muscle-step-seconds") {
                    require(index + 1 < argc && !muscleStepSeconds.has_value(),
                            "--muscle-step-seconds requires one value and may be given only once");
                    muscleStepSeconds.emplace(parseMuscleStepSeconds(argv[++index]));
                } else if (argument == "--muscle-step-count") {
                    require(index + 1 < argc && !muscleStepCount.has_value(),
                            "--muscle-step-count requires one value and may be given only once");
                    muscleStepCount.emplace(parseMuscleStepCount(argv[++index]));
                } else if (argument == "--muscle-activation") {
                    require(index + 1 < argc && !muscleActivation.has_value(),
                            "--muscle-activation requires one value and may be given only once");
                    muscleActivation.emplace(parseMuscleActivation(argv[++index]));
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
                } else if (argument == "--soft-tissue-stable-id") {
                    require(index + 1 < argc,
                            "--soft-tissue-stable-id requires one source stable ID");
                    requestedSoftTissueStableIds.push_back(
                        parseSourceRouteIndex(argv[++index])
                    );
                } else if (argument == "--visible-bone-body-index") {
                    require(index + 1 < argc,
                            "--visible-bone-body-index requires one articulated body index");
                    requestedBoneBodyIndices.push_back(
                        parseSourceRouteIndex(argv[++index])
                    );
                } else if (argument == "--focus-body-index") {
                    require(index + 1 < argc && !focusBodyIndex.has_value(),
                            "--focus-body-index requires one body index and may be given only once");
                    focusBodyIndex.emplace(parseSourceRouteIndex(argv[++index]));
                } else if (argument == "--soft-tissue-payload") {
                    require(index + 1 < argc && !softTissuePayloadPath.has_value(),
                            "--soft-tissue-payload requires one path and may be given only once");
                    softTissuePayloadPath.emplace(argv[++index]);
                } else if (argument == "--support-contact-payload") {
                    require(index + 1 < argc && !supportContactPayloadPath.has_value(),
                            "--support-contact-payload requires one path and may be given only once");
                    supportContactPayloadPath.emplace(argv[++index]);
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
                          << " [--muscle-step-count <1..64>]"
                          << " [--muscle-activation <0..1>]"
                          << " [--source-route-centrelines] [--source-route-index <0..415>]..."
                          << " [--surface-project-source-sites]"
                          << " [--soft-tissue-payload <NHTISS2>]"
                          << " [--visible-bone-body-index <0..156>]..."
                          << " [--soft-tissue-stable-id <1..N>]..."
                          << " [--support-contact-payload <NHCNT1>]"
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
            std::sort(requestedBoneBodyIndices.begin(), requestedBoneBodyIndices.end());
            const auto duplicateBoneBody = std::adjacent_find(
                requestedBoneBodyIndices.begin(), requestedBoneBodyIndices.end()
            );
            require(duplicateBoneBody == requestedBoneBodyIndices.end(),
                    "--visible-bone-body-index values must be unique");
            require(std::all_of(
                        requestedBoneBodyIndices.begin(), requestedBoneBodyIndices.end(),
                        [&rigid](const std::uint32_t index) {
                            return index < rigid.header.engineBodyCount;
                        }
                    ),
                    "--visible-bone-body-index exceeds the source body count");
            std::sort(
                requestedSoftTissueStableIds.begin(),
                requestedSoftTissueStableIds.end()
            );
            const auto duplicateSoftTissue = std::adjacent_find(
                requestedSoftTissueStableIds.begin(), requestedSoftTissueStableIds.end()
            );
            require(duplicateSoftTissue == requestedSoftTissueStableIds.end(),
                    "--soft-tissue-stable-id values must be unique");
            std::optional<LoadedBones> bonePayload;
            if (bodypartsBoneVisual) {
                bonePayload.emplace(loadBones(positional[2], rigid.header));
            }
            require(requestedBoneBodyIndices.empty() || bonePayload.has_value(),
                    "--visible-bone-body-index requires a BodyParts3D bone payload");
            if (!requestedBoneBodyIndices.empty()) {
                for (const std::uint32_t bodyIndex : requestedBoneBodyIndices) {
                    const bool present = std::any_of(
                        bonePayload->records.begin(), bonePayload->records.end(),
                        [bodyIndex](const BoneRecord& bone) { return bone.bodyIndex == bodyIndex; }
                    );
                    require(present,
                            "--visible-bone-body-index has no source mesh in the supplied payload");
                }
            }
            std::optional<LoadedSoftTissues> softTissuePayload;
            if (softTissuePayloadPath.has_value()) {
                require(bodypartsBoneVisual,
                        "--soft-tissue-payload requires a BodyParts3D bone payload");
                softTissuePayload.emplace(loadSoftTissues(*softTissuePayloadPath, rigid.header));
                require(
                    bonePayload->header.reserved0 == softTissuePayload->header.reserved0,
                    "BodyParts3D bone and soft-tissue payloads have different visual registrations"
                );
            }
            require(requestedSoftTissueStableIds.empty() || softTissuePayload.has_value(),
                    "--soft-tissue-stable-id requires --soft-tissue-payload");
            if (!requestedSoftTissueStableIds.empty()) {
                for (const std::uint32_t stableId : requestedSoftTissueStableIds) {
                    const bool present = std::any_of(
                        softTissuePayload->records.begin(), softTissuePayload->records.end(),
                        [stableId](const SoftTissueRecord& tissue) {
                            return tissue.stableId == stableId;
                        }
                    );
                    require(present,
                            "--soft-tissue-stable-id is not present in the supplied payload");
                }
            }
            std::optional<LoadedSupportContacts> supportContactPayload;
            if (supportContactPayloadPath.has_value()) {
                require(muscleStepSeconds.has_value(),
                        "--support-contact-payload requires --muscle-step-seconds");
                supportContactPayload.emplace(loadSupportContacts(
                    *supportContactPayloadPath, rigid.header
                ));
            }
            require(!surfaceProjectSourceSites ||
                        (bodypartsBoneVisual && sourceRouteCentrelines),
                    "--surface-project-source-sites requires BodyParts3D bones and a source-route inspection");
            require(!muscleActivation.has_value() || muscleStepSeconds.has_value(),
                    "--muscle-activation requires --muscle-step-seconds");
            require(!muscleStepCount.has_value() || muscleStepSeconds.has_value(),
                    "--muscle-step-count requires --muscle-step-seconds");
            std::optional<MuscleDrivenVisualState> muscleDrivenState;
            std::span<const float> poseQ = rigid.model.defaultQ;
            if (muscleStepSeconds.has_value()) {
                muscleDrivenState.emplace(integrateMuscleDrivenVisualState(
                    rigid.model, musclePayload, *muscleStepSeconds,
                    muscleStepCount.value_or(1u),
                    muscleActivation.value_or(0.5),
                    supportContactPayload.has_value() ? &*supportContactPayload : nullptr
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
            std::uint32_t renderedBodies = 0u;
            std::uint32_t renderedSoftTissues = 0u;
            std::uint32_t renderedRouteSegments = 0u;
            bool anyRequestedRouteVisible = false;
            const metalrobo::VisualAssetPackV2 pack = makeMarkerPack(
                rigid.model, musclePayload,
                bonePayload.has_value() ? &*bonePayload : nullptr,
                softTissuePayload.has_value() ? &*softTissuePayload : nullptr,
                bodies,
                muscleDrivenState.has_value(),
                requestedBoneBodyIndices,
                requestedSoftTissueStableIds,
                resolvedRouteCentrelines.has_value() ? &*resolvedRouteCentrelines : nullptr,
                renderedBodies, renderedSoftTissues, renderedRouteSegments
            );
            require(requestedSoftTissueStableIds.empty() || renderedSoftTissues == requestedSoftTissueStableIds.size(),
                    "native Human visual soft-tissue selection did not render every requested source surface");
            require(requestedBoneBodyIndices.empty() || renderedBodies > 0u,
                    "native Human visual bone selection rendered no source mesh");
            const CameraFraming cameraFraming = makeCameraFraming(
                pack, bodies, focusBodyIndex
            );
            const std::array<mr_float4, 4u> positions = cameraPositions(cameraFraming);
            std::array<std::string, 4u> cameraNames;
            const metalrobo::WorldTemplate world = makeWorld(
                rigid.model, cameraFraming, frameDimension, cameraNames
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
            const std::filesystem::path outputDirectory{positional.back()};
            std::filesystem::create_directories(outputDirectory);
            const std::string stem = std::string(bodypartsBoneVisual
                ? "myosim-fullbody-articulated-bodyparts-bones"
                : "myosim-fullbody-articulated-markers") +
                (softTissuePayload.has_value() ? "-source-soft-tissues" : "") +
                (muscleDrivenState.has_value() ? "-muscle-driven" : "") +
                (supportContactPayload.has_value() ? "-source-support-contact" : "") +
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
                        makeHumanAnatomyLightRig(
                            cameraFraming.center, positions.front(), cameraFraming.distance
                        ), manifest, &reason
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
                            makeHumanAnatomyLightRig(
                                cameraFraming.center, positions[camera], cameraFraming.distance
                            ), cameraManifest, &reason
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
                const std::size_t muscleSurfacePixels = coverage(observation, kMuscleSurfaceSemantic);
                const std::size_t tendonSurfacePixels = coverage(observation, kTendonSurfaceSemantic);
                completeVisualCoverage = completeVisualCoverage &&
                    (bodypartsBoneVisual ? bonePixels > 0u : bodyPixels > 0u);
                anyRequestedRouteVisible = anyRequestedRouteVisible || routePixels > 0u;
                std::cout << "view=" << cameraNames[camera]
                          << " body_pixels=" << bodyPixels
                          << " bone_pixels=" << bonePixels
                          << " muscle_site_pixels=" << sitePixels
                          << " muscle_route_pixels=" << routePixels
                          << " muscle_surface_pixels=" << muscleSurfacePixels
                          << " tendon_surface_pixels=" << tendonSurfacePixels
                          << " frame=" << frame.string() << '\n';
            }
            require(completeVisualCoverage,
                    "one or more native Human frames have no linked-body coverage");
            require(!sourceRouteCentrelines || anyRequestedRouteVisible,
                    "requested source route is completely occluded from all native Human cameras");
            const bool sourceSupportContact = muscleDrivenState.has_value() &&
                muscleDrivenState->supportContactApplied;
            const std::string poseSource = !muscleDrivenState.has_value()
                ? "source_default_q_to_metal_kinematic_pose"
                : sourceSupportContact
                    ? "metal_all_416_mujoco_force_projection_and_activation_state_then_cpu_fp64_free_dynamics_and_dynamic_source_foot_witness_plane_contact_then_metal_kinematic_pose"
                    : "metal_all_416_mujoco_force_projection_and_activation_state_then_cpu_fp64_free_dynamics_then_metal_kinematic_pose";
            const std::string evidenceBoundary = !muscleDrivenState.has_value()
                ? (bodypartsBoneVisual
                    ? (softTissuePayload.has_value()
                        ? "metal_pose_snapshot_to_native_renderer_with_provisional_bodyparts_bone_registration_and_two_body_kinematic_source_soft_tissue_visuals_not_collision_or_live_rollout"
                        : "metal_pose_snapshot_to_native_renderer_with_provisional_bodyparts_bone_registration_not_collision_or_live_rollout")
                    : "metal_pose_snapshot_to_native_renderer_not_bodyparts_registration_or_live_rollout")
                : sourceSupportContact
                    ? (bodypartsBoneVisual
                        ? (softTissuePayload.has_value()
                            ? "bounded_multistep_all_416_mujoco_metal_force_projection_and_activation_state_then_cpu_fp64_dynamics_with_dynamic_source_foot_witness_plane_contact_and_metal_pose_with_provisional_bodyparts_bones_and_two_body_soft_tissue_visuals_metal_fullbody_contact_not_admitted_not_general_collision_stable_posture_or_live_rollout"
                            : "bounded_multistep_all_416_mujoco_metal_force_projection_and_activation_state_then_cpu_fp64_dynamics_with_dynamic_source_foot_witness_plane_contact_and_metal_pose_with_provisional_bodyparts_bones_metal_fullbody_contact_not_admitted_not_general_collision_stable_posture_or_live_rollout")
                        : "bounded_multistep_all_416_mujoco_metal_force_projection_and_activation_state_then_cpu_fp64_dynamics_with_dynamic_source_foot_witness_plane_contact_and_metal_pose_metal_fullbody_contact_not_admitted_not_general_collision_stable_posture_or_live_rollout")
                    : (bodypartsBoneVisual
                        ? (softTissuePayload.has_value()
                            ? "bounded_multistep_all_416_mujoco_metal_force_projection_and_activation_state_then_cpu_fp64_dynamics_to_metal_pose_snapshot_with_provisional_bodyparts_bone_registration_and_two_body_kinematic_source_soft_tissue_visuals_not_contact_or_live_rollout"
                            : "bounded_multistep_all_416_mujoco_metal_force_projection_and_activation_state_then_cpu_fp64_dynamics_to_metal_pose_snapshot_with_provisional_bodyparts_bone_registration_not_contact_or_live_rollout")
                        : "bounded_multistep_all_416_mujoco_metal_force_projection_and_activation_state_then_cpu_fp64_dynamics_to_metal_pose_snapshot_not_contact_or_live_rollout");
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
                      << " requested_bone_bodies=" << requestedBoneBodyIndices.size()
                      << " bodyparts_soft_tissues=" << renderedSoftTissues
                      << " requested_soft_tissue_surfaces=" << requestedSoftTissueStableIds.size()
                      << " soft_tissue_binding=" << (softTissuePayload.has_value()
                              ? "two_body_linear_blend_world_surface_snapshot"
                              : "none")
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
                      << " camera_framing=" << (cameraFraming.usesSourceGeometryBounds
                              ? "exact_rendered_source_geometry_bounds" : "focused_body_inspection")
                      << " camera_source_extent_m=" << cameraFraming.sourceExtentMeters
                      << " camera_distance_m=" << cameraFraming.distance
                      << " pose_stage_elapsed_ms=" << poseDiagnostics.elapsedMilliseconds
                      << " renderer_compile_ms_first_camera=" << rendererCompileMilliseconds
                      << " pose_source=" << poseSource
                      << " muscle_step_seconds=" << (muscleStepSeconds.has_value()
                              ? *muscleStepSeconds : 0.0)
                      << " muscle_step_count=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->stepCount : 0u)
                      << " muscle_activation=" << (muscleStepSeconds.has_value()
                              ? muscleActivation.value_or(0.5) : 0.0)
                      << " muscle_passive_baseline=" << (muscleDrivenState.has_value()
                              ? "source_default_activation_zero_subtracted" : "none")
                      << " muscle_step_applied_wraps=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->appliedWrapCount : 0u)
                      << " muscle_force_metal_device=\"" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->muscleMetalDeviceName : "none") << "\""
                      << " muscle_force_metal_transactions=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->muscleMetalStepCount : 0u)
                      << " muscle_force_metal_active_records=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->muscleMetalForceRecordCount : 0u)
                      << " muscle_force_metal_elapsed_ms=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->muscleMetalElapsedMilliseconds : 0.0)
                      << " muscle_step_max_velocity_delta=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->maximumVelocityDelta : 0.0)
                      << " muscle_step_max_configuration_delta=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->maximumConfigurationDelta : 0.0)
                      << " source_support_contact=" << (sourceSupportContact ? "true" : "false")
                      << " source_support_witnesses=" << (sourceSupportContact
                              ? muscleDrivenState->supportWitnessCount : 0u)
                      << " source_support_active_contacts=" << (sourceSupportContact
                              ? muscleDrivenState->activeSupportContactCount : 0u)
                      << " source_support_max_active_contacts=" << (sourceSupportContact
                              ? muscleDrivenState->maximumActiveSupportContactCount : 0u)
                      << " source_support_min_plane_gap_m=" << (sourceSupportContact
                              ? muscleDrivenState->minimumSupportPlaneGapMeters : 0.0)
                      << " source_support_seed_translation_m=" << (sourceSupportContact
                              ? muscleDrivenState->supportSeedTranslationMeters : 0.0)
                      << " source_support_metal_device=\"" << (sourceSupportContact
                              ? muscleDrivenState->supportDeviceName : "none") << "\""
                      << " source_support_metal_status=" << (sourceSupportContact
                              ? muscleDrivenState->supportMetalStatus : "not_requested")
                      << " source_support_metal_elapsed_ms=" << (sourceSupportContact
                              ? muscleDrivenState->supportGpuElapsedMilliseconds : 0.0)
                      << " source_support_max_gpu_cpu_velocity_error=" << (sourceSupportContact
                              ? muscleDrivenState->supportMaximumGpuCpuVelocityError : 0.0)
                      << " boundary=" << evidenceBoundary
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
