#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/MujocoMuscleReference.hpp"
#include "metalrobo/NumiHumanJointEquality.hpp"
#include "metalrobo/NumiHumanMuscleEquilibrium.hpp"
#include "metalrobo/NumiHumanTendon.hpp"
#include "metalrobo/NumiHumanTendonMetal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr std::array<char, 8u> kRigidMagic{'N', 'H', 'R', 'I', 'G', 'I', 'D', '2'};
constexpr std::array<char, 8u> kLegacyMuscleMagic{'N', 'H', 'M', 'Y', 'O', '1', '\0', '\0'};
constexpr std::array<char, 8u> kMuscleMagic{'N', 'H', 'M', 'Y', 'O', '2', '\0', '\0'};
constexpr std::uint32_t kRigidAbi = 1u;
constexpr std::uint32_t kLegacyMuscleAbi = 1u;
constexpr std::uint32_t kMuscleAbi = 2u;

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

struct MuscleArchitectureRecord {
    float optimalFiberLength = 0.0f;
    float tendonSlackLength = 0.0f;
    float tendonStrainAtOneNormalizedForce = 0.0f;
    float tendonStiffnessAtOneNormalizedForce = 0.0f;
    float tendonNormalizedForceAtToeEnd = 0.0f;
    float tendonCurviness = 0.0f;
    float normalizedFiberDamping = 0.0f;
    float fitNormalizedRmse = 0.0f;
};
#pragma pack(pop)

static_assert(sizeof(RigidHeader) == 80u);
static_assert(sizeof(SourcePoseRecord) == 28u);
static_assert(sizeof(MuscleHeader) == 76u);
static_assert(sizeof(SiteRecord) == 16u);
static_assert(sizeof(WrapRecord) == 64u);
static_assert(sizeof(RouteRecord) == 16u);
static_assert(sizeof(MuscleRecord) == 164u);
static_assert(sizeof(MuscleArchitectureRecord) == 32u);
static_assert(sizeof(MRWorldGPU) == 96u);
static_assert(sizeof(MRArticulationGPU) == 48u);
static_assert(sizeof(MRBodyPropertiesGPU) == 160u);
static_assert(sizeof(MRJointDescriptorGPU) == 144u);
static_assert(sizeof(MRDofPropertiesGPU) == 64u);

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
void readObject(std::istream& input, T& value, const char* description) {
    static_assert(std::is_trivially_copyable_v<T>);
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    require(input.good(), std::string("truncated ") + description);
}

template <typename T>
std::vector<T> readVector(std::istream& input, const std::size_t count, const char* description) {
    std::vector<T> result(count);
    if (count) {
        input.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(count * sizeof(T)));
        require(input.good(), std::string("truncated ") + description);
    }
    return result;
}

struct LoadedRigid {
    metalrobo::EngineModel model;
    RigidHeader header{};
    std::vector<std::uint32_t> sourceBodyToCore;
    std::vector<SourcePoseRecord> sourceDefaultPoses;
};

LoadedRigid loadRigid(const char* path) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), std::string("cannot open rigid payload ") + path);
    LoadedRigid result;
    readObject(input, result.header, "MyoSim rigid header");
    require(result.header.magic == kRigidMagic, "rigid payload magic is not NHRIGID2");
    require(result.header.payloadAbi == kRigidAbi, "unsupported MyoSim rigid payload ABI");
    require(result.header.engineAbi == MR_ENGINE_ABI_VERSION, "MyoSim rigid payload/Core engine ABI mismatch");
    require(result.header.reserved0 == 0u && result.header.rootBodyIndex == 0u, "invalid MyoSim rigid header reserved/root fields");
    require(result.header.sourceBodyCount > 0u && result.header.engineBodyCount >= result.header.sourceBodyCount,
            "invalid MyoSim rigid body counts");
    require(result.header.jointCount + 1u == result.header.engineBodyCount,
            "MyoSim rigid tree must have one inbound joint per non-root node");
    require(result.header.nq == result.header.nv + 1u, "MyoSim floating state dimensions are invalid");
    result.model.name = "numilab_human_myosim_fullbody_reference";
    readObject(input, result.model.world, "MyoSim world record");
    MRArticulationGPU articulation{};
    readObject(input, articulation, "MyoSim articulation record");
    result.model.articulations.push_back(articulation);
    result.model.bodies = readVector<MRBodyPropertiesGPU>(input, result.header.engineBodyCount, "MyoSim body records");
    result.model.joints = readVector<MRJointDescriptorGPU>(input, result.header.jointCount, "MyoSim joint records");
    result.model.dofs = readVector<MRDofPropertiesGPU>(input, result.header.nv, "MyoSim DoF records");
    result.model.defaultQ = readVector<float>(input, result.header.nq, "MyoSim default q");
    result.model.defaultV = readVector<float>(input, result.header.nv, "MyoSim default v");
    result.sourceBodyToCore = readVector<std::uint32_t>(input, result.header.sourceBodyCount, "source-to-Core body map");
    result.sourceDefaultPoses = readVector<SourcePoseRecord>(input, result.header.sourceBodyCount, "source default poses");
    require(input.peek() == std::char_traits<char>::eof(), "MyoSim rigid payload has trailing bytes");
    require(result.model.world.abiVersion == result.header.engineAbi &&
            result.model.world.bodyCount == result.header.engineBodyCount &&
            result.model.world.articulationCount == 1u && result.model.world.jointCount == result.header.jointCount &&
            result.model.world.nq == result.header.nq && result.model.world.nv == result.header.nv,
            "MyoSim rigid world/header disagreement");
    require(articulation.rootType == MR_ROOT_FLOATING && articulation.rootBody == 0u &&
            articulation.bodyCount == result.header.engineBodyCount && articulation.jointCount == result.header.jointCount &&
            articulation.nq == result.header.nq && articulation.nv == result.header.nv,
            "MyoSim floating articulation/header disagreement");
    for (const std::uint32_t body : result.sourceBodyToCore) {
        require(body < result.header.engineBodyCount, "source-to-Core body map is out of bounds");
    }
    std::string reason;
    require(result.model.valid(&reason), "MyoSim Core model invalid: " + reason);
    return result;
}

struct LoadedMuscles {
    MuscleHeader header{};
    std::vector<metalrobo::MujocoMuscleSite> sites;
    std::vector<metalrobo::MujocoWrapGeometry> wraps;
    std::vector<metalrobo::MujocoMuscleDefinition> muscles;
    std::vector<metalrobo::MujocoCompliantMuscleArchitecture> architectures;
    std::vector<MRMujocoMuscleSiteGPU> gpuSites;
    std::vector<MRMujocoMuscleWrapGPU> gpuWraps;
    std::vector<MRMujocoMuscleRouteNodeGPU> gpuRoutes;
    std::vector<MRMujocoMuscleGPU> gpuMuscles;
    std::vector<MRMujocoMuscleStateGPU> gpuStates;
    std::vector<double> oracleLength;
    std::vector<double> oracleForce;
    std::vector<metalrobo::MujocoMuscleSite> sourceSites;
    std::vector<metalrobo::MujocoMuscleDefinition> sourceMuscles;
    std::uint32_t tendonPointBindings = 0u;
    std::uint32_t tendonTriangleBindings = 0u;
    std::uint32_t tendonEnvelopeBindings = 0u;
    double maximumEndpointMigration = 0.0;
    metalrobo::NumiHumanTendonPayload tendonPayload;
};

metalrobo::MujocoRouteNodeType routeType(const std::uint32_t value) {
    switch (value) {
    case 1u: return metalrobo::MujocoRouteNodeType::site;
    case 2u: return metalrobo::MujocoRouteNodeType::sphere;
    case 3u: return metalrobo::MujocoRouteNodeType::cylinder;
    default: throw std::runtime_error("MyoSim route has an unknown node type");
    }
}

LoadedMuscles loadMuscles(const char* path, const RigidHeader& rigid) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), std::string("cannot open muscle payload ") + path);
    LoadedMuscles result;
    readObject(input, result.header, "MyoSim muscle header");
    const bool legacy = result.header.magic == kLegacyMuscleMagic &&
        result.header.payloadAbi == kLegacyMuscleAbi &&
        result.header.reserved0 == 0u && result.header.reserved1 == 0u;
    const bool compliant = result.header.magic == kMuscleMagic &&
        result.header.payloadAbi == kMuscleAbi &&
        result.header.reserved0 == result.header.muscleCount &&
        result.header.reserved1 == sizeof(MuscleArchitectureRecord);
    require(legacy || compliant,
            "unsupported or non-canonical NHMYO1/NHMYO2 muscle payload ABI");
    require(result.header.engineBodyCount == rigid.engineBodyCount && result.header.sourceSha256 == rigid.sourceSha256,
            "MyoSim muscle payload does not match rigid payload source");
    const std::vector<SiteRecord> sourceSites = readVector<SiteRecord>(input, result.header.siteCount, "MyoSim site records");
    const std::vector<WrapRecord> sourceWraps = readVector<WrapRecord>(input, result.header.wrapCount, "MyoSim wrap records");
    const std::vector<RouteRecord> routes = readVector<RouteRecord>(input, result.header.routeNodeCount, "MyoSim route records");
    const std::vector<MuscleRecord> sourceMuscles = readVector<MuscleRecord>(input, result.header.muscleCount, "MyoSim muscle records");
    const std::vector<MuscleArchitectureRecord> sourceArchitectures = compliant
        ? readVector<MuscleArchitectureRecord>(
            input, result.header.muscleCount, "MyoSim compliant architecture records"
        )
        : std::vector<MuscleArchitectureRecord>(result.header.muscleCount);
    require(input.peek() == std::char_traits<char>::eof(), "MyoSim muscle payload has trailing bytes");
    result.sites.reserve(sourceSites.size());
    result.gpuSites.reserve(sourceSites.size());
    for (const SiteRecord& source : sourceSites) {
        require(source.bodyIndex < rigid.engineBodyCount, "MyoSim site body index is out of bounds");
        result.sites.push_back({source.bodyIndex, {source.x, source.y, source.z}});
        MRMujocoMuscleSiteGPU gpuSite{};
        gpuSite.bodyIndex = source.bodyIndex;
        gpuSite.localPoint = {source.x, source.y, source.z, 0.0f};
        result.gpuSites.push_back(gpuSite);
    }
    result.wraps.reserve(sourceWraps.size());
    result.gpuWraps.reserve(sourceWraps.size());
    for (const WrapRecord& source : sourceWraps) {
        require(source.bodyIndex < rigid.engineBodyCount, "MyoSim wrap body index is out of bounds");
        const auto type = routeType(source.type);
        result.wraps.push_back({
            source.bodyIndex, type, {source.centerX, source.centerY, source.centerZ},
            {source.rotation[0], source.rotation[1], source.rotation[2], source.rotation[3], source.rotation[4],
             source.rotation[5], source.rotation[6], source.rotation[7], source.rotation[8]}, source.radius,
        });
        MRMujocoMuscleWrapGPU gpuWrap{};
        gpuWrap.bodyIndex = source.bodyIndex;
        gpuWrap.type = source.type;
        gpuWrap.localCenter = {
            source.centerX, source.centerY, source.centerZ, 0.0f,
        };
        gpuWrap.rotationRow0 = {
            source.rotation[0], source.rotation[1], source.rotation[2], 0.0f,
        };
        gpuWrap.rotationRow1 = {
            source.rotation[3], source.rotation[4], source.rotation[5], 0.0f,
        };
        gpuWrap.rotationRow2 = {
            source.rotation[6], source.rotation[7], source.rotation[8], 0.0f,
        };
        gpuWrap.radius = {source.radius, 0.0f, 0.0f, 0.0f};
        result.gpuWraps.push_back(gpuWrap);
    }
    result.gpuRoutes.reserve(routes.size());
    for (const RouteRecord& source : routes) {
        (void)routeType(source.type);
        require(source.reserved0 == 0u, "MyoSim route reserved field is nonzero");
        MRMujocoMuscleRouteNodeGPU gpuRoute{};
        gpuRoute.type = source.type;
        gpuRoute.targetIndex = source.targetIndex;
        gpuRoute.sideSiteIndex = source.sideSiteIndex;
        result.gpuRoutes.push_back(gpuRoute);
    }
    result.muscles.reserve(sourceMuscles.size());
    result.architectures.reserve(sourceMuscles.size());
    result.gpuMuscles.reserve(sourceMuscles.size());
    result.gpuStates.reserve(sourceMuscles.size());
    result.oracleLength.reserve(sourceMuscles.size());
    result.oracleForce.reserve(sourceMuscles.size());
    for (std::size_t muscleIndex = 0u; muscleIndex < sourceMuscles.size(); ++muscleIndex) {
        const MuscleRecord& source = sourceMuscles[muscleIndex];
        const MuscleArchitectureRecord& architecture = sourceArchitectures[muscleIndex];
        require(source.reserved0 == 0u && source.routeOffset <= routes.size() && source.routeCount <= routes.size() - source.routeOffset,
                "MyoSim muscle route range is invalid");
        metalrobo::MujocoMuscleDefinition definition;
        definition.route.reserve(source.routeCount);
        for (std::uint32_t index = 0; index < source.routeCount; ++index) {
            const RouteRecord& route = routes[source.routeOffset + index];
            require(route.reserved0 == 0u, "MyoSim route reserved field is nonzero");
            definition.route.push_back({routeType(route.type), route.targetIndex, route.sideSiteIndex});
        }
        definition.lengthRange = {source.values[0], source.values[1]};
        definition.accelerationScale = source.values[2];
        definition.controlRange = {source.values[3], source.values[4]};
        for (std::size_t index = 0; index < 10; ++index) {
            definition.gainParameters[index] = source.values[5 + index];
            definition.biasParameters[index] = source.values[15 + index];
            definition.dynamicParameters[index] = source.values[25 + index];
        }
        result.oracleLength.push_back(source.values[35]);
        result.oracleForce.push_back(source.values[36]);
        result.muscles.push_back(std::move(definition));
        result.architectures.push_back({
            architecture.optimalFiberLength,
            architecture.tendonSlackLength,
            architecture.tendonStrainAtOneNormalizedForce,
            architecture.tendonStiffnessAtOneNormalizedForce,
            architecture.tendonNormalizedForceAtToeEnd,
            architecture.tendonCurviness,
            architecture.normalizedFiberDamping,
            architecture.fitNormalizedRmse,
        });
        MRMujocoMuscleGPU gpuMuscle{};
        gpuMuscle.route = {source.routeOffset, source.routeCount, 0u, 0u};
        gpuMuscle.lengthRangeAndAcceleration = {
            source.values[0], source.values[1], source.values[2], 0.0f,
        };
        gpuMuscle.controlRange = {
            source.values[3], source.values[4], 0.0f, 0.0f,
        };
        for (std::size_t index = 0u; index < 10u; ++index) {
            (&gpuMuscle.gainParameters[index / 4u].x)[index % 4u] =
                source.values[5u + index];
            (&gpuMuscle.biasParameters[index / 4u].x)[index % 4u] =
                source.values[15u + index];
            (&gpuMuscle.dynamicParameters[index / 4u].x)[index % 4u] =
                source.values[25u + index];
        }
        gpuMuscle.compliantArchitecture0 = {
            architecture.optimalFiberLength,
            architecture.tendonSlackLength,
            architecture.tendonStrainAtOneNormalizedForce,
            architecture.tendonStiffnessAtOneNormalizedForce,
        };
        gpuMuscle.compliantArchitecture1 = {
            architecture.tendonNormalizedForceAtToeEnd,
            architecture.tendonCurviness,
            architecture.normalizedFiberDamping,
            architecture.fitNormalizedRmse,
        };
        result.gpuMuscles.push_back(gpuMuscle);
        MRMujocoMuscleStateGPU gpuState{};
        gpuState.excitationAndActivation = {0.5f, 0.5f, 0.0f, 0.0f};
        result.gpuStates.push_back(gpuState);
    }
    return result;
}

std::vector<std::byte> readBytes(const char* path);

metalrobo::NumiHumanJointEqualityPayload loadJointEqualities(
    const char* path, const RigidHeader& rigid
) {
    const std::vector<std::byte> bytes = readBytes(path);
    metalrobo::NumiHumanJointEqualityPayload payload;
    const auto diagnostics = metalrobo::decodeNumiHumanJointEqualityPayload(
        bytes, rigid.sourceSha256, payload
    );
    require(
        diagnostics.succeeded(),
        std::string("NHEQ decode failed: ") +
            metalrobo::numiHumanJointEqualityStatusName(diagnostics.status) +
            " index=" + std::to_string(diagnostics.failingIndex)
    );
    require(
        payload.nq == rigid.nq && payload.nv == rigid.nv,
        "NHEQ dimensions disagree with NHRIGID2"
    );
    return payload;
}

std::vector<std::byte> readBytes(const char* path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(input.is_open(), std::string("cannot open tendon payload ") + path);
    const std::streamsize size = input.tellg();
    require(size >= 0, "cannot determine tendon payload size");
    input.seekg(0, std::ios::beg);
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()), size);
        require(input.good(), "truncated tendon payload");
    }
    return bytes;
}

void applyNumiHumanTendonPayload(
    const char* path, const RigidHeader& rigid, LoadedMuscles& muscles
) {
    const std::vector<std::byte> bytes = readBytes(path);
    metalrobo::NumiHumanTendonPayload payload;
    const auto decode = metalrobo::decodeNumiHumanTendonPayload(
        bytes, rigid.sourceSha256, {}, payload
    );
    require(
        decode.succeeded(),
        std::string("NHTENDON decode failed: ") +
            metalrobo::numiHumanTendonStatusName(decode.status) +
            " index=" + std::to_string(decode.failingIndex)
    );
    require(
        payload.bodyCount == rigid.engineBodyCount,
        "NHTENDON body count disagrees with NHRIGID2"
    );
    muscles.sourceSites = muscles.sites;
    muscles.sourceMuscles = muscles.muscles;
    metalrobo::NumiHumanTendonResolvedProgram resolved;
    const auto diagnostics = metalrobo::resolveNumiHumanTendonProgram(
        payload, muscles.sourceSites, muscles.sourceMuscles, resolved
    );
    require(
        diagnostics.succeeded(),
        std::string("NHTENDON endpoint resolution failed: ") +
            metalrobo::numiHumanTendonStatusName(diagnostics.status) +
            " index=" + std::to_string(diagnostics.failingIndex)
    );
    muscles.sites = std::move(resolved.sites);
    muscles.muscles = std::move(resolved.muscles);
    muscles.tendonPointBindings = resolved.pointBindingCount;
    muscles.tendonTriangleBindings = resolved.triangleBindingCount;
    muscles.tendonEnvelopeBindings = resolved.envelopeBindingCount;
    muscles.maximumEndpointMigration = resolved.maximumEndpointMigration;
    muscles.tendonPayload = std::move(payload);

    muscles.gpuSites.clear();
    muscles.gpuSites.reserve(muscles.sites.size());
    for (const metalrobo::MujocoMuscleSite& site : muscles.sites) {
        MRMujocoMuscleSiteGPU gpu{};
        gpu.bodyIndex = site.bodyIndex;
        gpu.localPoint = {
            static_cast<float>(site.localPoint[0]),
            static_cast<float>(site.localPoint[1]),
            static_cast<float>(site.localPoint[2]), 0.0f,
        };
        muscles.gpuSites.push_back(gpu);
    }
    muscles.gpuRoutes.clear();
    for (std::size_t muscleIndex = 0u; muscleIndex < muscles.muscles.size(); ++muscleIndex) {
        MRMujocoMuscleGPU& gpuMuscle = muscles.gpuMuscles[muscleIndex];
        gpuMuscle.route.x = static_cast<std::uint32_t>(muscles.gpuRoutes.size());
        gpuMuscle.route.y = static_cast<std::uint32_t>(muscles.muscles[muscleIndex].route.size());
        for (const metalrobo::MujocoRouteNode& node : muscles.muscles[muscleIndex].route) {
            MRMujocoMuscleRouteNodeGPU gpu{};
            gpu.type = static_cast<std::uint32_t>(node.type);
            gpu.targetIndex = node.targetIndex;
            gpu.sideSiteIndex = node.sideSiteIndex;
            muscles.gpuRoutes.push_back(gpu);
        }
    }
}

double quaternionAngle(const std::array<double, 4>& left, const SourcePoseRecord& right) {
    const double rightNorm = std::sqrt(
        static_cast<double>(right.quaternionX) * right.quaternionX + static_cast<double>(right.quaternionY) * right.quaternionY +
        static_cast<double>(right.quaternionZ) * right.quaternionZ + static_cast<double>(right.quaternionW) * right.quaternionW
    );
    const double dot = std::abs((left[0] * right.quaternionX + left[1] * right.quaternionY +
                                 left[2] * right.quaternionZ + left[3] * right.quaternionW) / rightNorm);
    return 2.0 * std::acos(std::clamp(dot, 0.0, 1.0));
}

double vectorNorm(const std::array<double, 3>& value) {
    return std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
}

std::array<double, 3> quaternionRotate(
    const std::array<double, 4>& q, const std::array<double, 3>& value
) {
    const std::array<double, 3> axis{q[0], q[1], q[2]};
    const std::array<double, 3> twiceCross{
        2.0 * (axis[1] * value[2] - axis[2] * value[1]),
        2.0 * (axis[2] * value[0] - axis[0] * value[2]),
        2.0 * (axis[0] * value[1] - axis[1] * value[0]),
    };
    const std::array<double, 3> crossAgain{
        axis[1] * twiceCross[2] - axis[2] * twiceCross[1],
        axis[2] * twiceCross[0] - axis[0] * twiceCross[2],
        axis[0] * twiceCross[1] - axis[1] * twiceCross[0],
    };
    return {
        value[0] + q[3] * twiceCross[0] + crossAgain[0],
        value[1] + q[3] * twiceCross[1] + crossAgain[1],
        value[2] + q[3] * twiceCross[2] + crossAgain[2],
    };
}

struct MetalArticulatedMetrics {
    std::string deviceName;
    double maximumBodyPositionError = 0.0;
    double maximumBodyOrientationComponentError = 0.0;
    double maximumPointPositionError = 0.0;
    double maximumPointJacobianError = 0.0;
    double maximumMuscleLengthError = 0.0;
    double maximumMusclePathVelocityError = 0.0;
    double maximumMuscleForceError = 0.0;
    double maximumReferenceMuscleForce = 0.0;
    double maximumNormalizedTendonTension = 0.0;
    double maximumNormalizedEquilibriumResidual = 0.0;
    std::uint32_t maximumNormalizedEquilibriumResidualMuscle = 0u;
    double maximumMuscleGeneralizedForceError = 0.0;
    double maximumReferenceMuscleGeneralizedForce = 0.0;
    double maximumSummedGeneralizedForceError = 0.0;
    double maximumReferenceSummedGeneralizedForce = 0.0;
    double maximumActivationStepError = 0.0;
    double maximumTendonNodalForceParityError = 0.0;
    double maximumTendonForceResidual = 0.0;
    double maximumTendonMomentResidual = 0.0;
    double maximumTendonGeneralizedCorrection = 0.0;
    double activationTimestepSeconds = 0.0;
    std::uint32_t appliedMuscleWraps = 0u;
    std::uint32_t tendonTransferCount = 0u;
    std::uint32_t tendonEnvelopeTransferCount = 0u;
    bool tendonReplayByteIdentical = false;
};

MetalArticulatedMetrics verifyMetalArticulatedReference(
    const metalrobo::EngineModel& model,
    const LoadedMuscles& muscles
) {
    const MRArticulationGPU& articulation = model.articulations.at(0u);
    std::vector<float> q = model.defaultQ;
    std::vector<double> qReference(q.begin(), q.end());
    std::vector<float> velocity(articulation.nv, 0.0f);
    std::vector<double> referenceVelocity(articulation.nv, 0.0);
    for (std::size_t dof = 0u; dof < articulation.nv; ++dof) {
        velocity[dof] = 0.002f * std::sin(0.17f * static_cast<float>(dof + 1u));
        referenceVelocity[dof] = velocity[dof];
    }

    std::vector<MRArticulatedPointImpulseGPU> gpuPoints;
    std::vector<metalrobo::ArticulatedPointQuery> cpuPoints;
    gpuPoints.reserve(5u * articulation.bodyCount);
    cpuPoints.reserve(5u * articulation.bodyCount);
    for (std::size_t localBody = 0u; localBody < articulation.bodyCount; ++localBody) {
        const std::uint32_t globalBody = articulation.firstBody + static_cast<std::uint32_t>(localBody);
        const double phase = static_cast<double>(localBody + 1u);
        const std::array<double, 3> localPoint{
            0.004 * std::sin(0.31 * phase),
            0.003 * std::cos(0.47 * phase),
            0.002 * std::sin(0.59 * phase),
        };
        MRArticulatedPointImpulseGPU gpuPoint{};
        gpuPoint.bodyIndex = globalBody;
        gpuPoint.localPoint = {
            static_cast<float>(localPoint[0]), static_cast<float>(localPoint[1]),
            static_cast<float>(localPoint[2]), 0.0f,
        };
        gpuPoints.push_back(gpuPoint);
        cpuPoints.push_back({globalBody, localPoint});
    }
    const std::uint32_t bodyJacobianPointOffset = static_cast<std::uint32_t>(
        gpuPoints.size()
    );
    for (std::size_t localBody = 0u; localBody < articulation.bodyCount; ++localBody) {
        const std::uint32_t globalBody = articulation.firstBody + static_cast<std::uint32_t>(localBody);
        for (std::size_t probe = 0u; probe < 4u; ++probe) {
            const std::array<double, 3> localPoint = probe == 0u
                ? std::array<double, 3>{0.0, 0.0, 0.0}
                : (probe == 1u
                    ? std::array<double, 3>{1.0, 0.0, 0.0}
                    : (probe == 2u
                        ? std::array<double, 3>{0.0, 1.0, 0.0}
                        : std::array<double, 3>{0.0, 0.0, 1.0}));
            MRArticulatedPointImpulseGPU gpuPoint{};
            gpuPoint.bodyIndex = globalBody;
            gpuPoint.localPoint = {
                static_cast<float>(localPoint[0]), static_cast<float>(localPoint[1]),
                static_cast<float>(localPoint[2]), 0.0f,
            };
            gpuPoints.push_back(gpuPoint);
            cpuPoints.push_back({globalBody, localPoint});
        }
    }

    std::vector<metalrobo::ArticulatedBodyKinematics> cpuBodies(articulation.bodyCount);
    auto cpuDiagnostics = metalrobo::computeArticulatedBodyKinematics(
        model, 0u, qReference, referenceVelocity, cpuBodies
    );
    require(cpuDiagnostics.succeeded(), "MyoSim CPU body reference failed before Metal parity");
    std::vector<metalrobo::ArticulatedPointKinematics> cpuPointKinematics(cpuPoints.size());
    std::vector<double> cpuJacobians(cpuPoints.size() * 3u * articulation.nv);
    cpuDiagnostics = metalrobo::computeArticulatedPointJacobians(
        model, 0u, qReference, referenceVelocity, cpuPoints, cpuPointKinematics, cpuJacobians
    );
    require(cpuDiagnostics.succeeded(), "MyoSim CPU point/Jacobian reference failed before Metal parity");
    const metalrobo::MetalArticulatedOperatorInput input{
        .articulationIndex = 0u,
        .environmentCount = 1u,
        .pointCount = gpuPoints.size(),
        .q = q,
        .v = velocity,
        .points = gpuPoints,
        .mujoco = {
            .muscles = muscles.gpuMuscles,
            .states = muscles.gpuStates,
            .sites = muscles.gpuSites,
            .wraps = muscles.gpuWraps,
            .routeNodes = muscles.gpuRoutes,
            .bodyJacobianPointOffset = bodyJacobianPointOffset,
        },
    };
    metalrobo::MetalArticulatedOperatorResult kinematicsResult;
    const metalrobo::MetalArticulatedOperatorConfig kinematicsConfig{
        .pointJacobiansOnly = true,
        .mujocoActivationTimestepSeconds = 1.0e-5f,
    };
    const auto kinematicsDiagnostics = metalrobo::runMetalArticulatedOperator(
        model, input, kinematicsResult, kinematicsConfig
    );
    require(
        kinematicsDiagnostics.succeeded() && kinematicsDiagnostics.dispatched &&
            kinematicsDiagnostics.published &&
            kinematicsDiagnostics.successfulEnvironmentCount == 1u &&
            kinematicsDiagnostics.failedEnvironmentCount == 0u,
        std::string("MyoSim Metal kinematics/Jacobian operator failed: ") +
            metalrobo::metalArticulatedOperatorHostStatusName(kinematicsDiagnostics.status) +
            " " + kinematicsDiagnostics.message +
            " first_gpu_status=" + std::to_string(kinematicsDiagnostics.firstGPUStatusCode)
    );
    require(
        kinematicsResult.bodyPoses.size() == cpuBodies.size() &&
            kinematicsResult.pointWorld.size() == cpuPointKinematics.size() &&
            kinematicsResult.pointJacobians.size() == cpuJacobians.size() &&
            kinematicsResult.mujocoResults.size() == muscles.muscles.size() &&
            kinematicsResult.mujocoActivationStates.size() ==
                muscles.gpuStates.size() &&
            kinematicsResult.mujocoMuscleGeneralizedForces.size() ==
                muscles.muscles.size() * articulation.nv &&
            kinematicsResult.mujocoGeneralizedForces.size() == articulation.nv,
        "MyoSim Metal kinematics/Jacobian result layout is invalid"
    );

    MetalArticulatedMetrics metrics;
    metrics.deviceName = kinematicsDiagnostics.deviceName;
    if (muscles.tendonPayload.payloadAbi == 2u) {
        metalrobo::NumiHumanTendonMetalProgram tendonProgram;
        const auto packDiagnostics = metalrobo::makeNumiHumanTendonMetalProgram(
            muscles.tendonPayload, tendonProgram
        );
        require(
            packDiagnostics.succeeded() &&
                tendonProgram.bindings.size() == muscles.tendonPayload.bindings.size() &&
                tendonProgram.envelopes.size() == muscles.tendonPayload.envelopes.size(),
            std::string("NHTENDON2 Metal packing failed: ") +
                metalrobo::numiHumanTendonStatusName(packDiagnostics.status)
        );
        const metalrobo::NumiHumanTendonMetalInput tendonInput{
            .environmentCount = 1u,
            .dofCount = articulation.nv,
            .bodyPoseStride = articulation.bodyCount,
            .articulationFirstBody = articulation.firstBody,
            .pointJacobianStride = kinematicsResult.layout.dispatch.pointJacobianStride,
            .bodyJacobianPointOffset = bodyJacobianPointOffset,
            .muscleResults = kinematicsResult.mujocoResults,
            .bodyPoses = kinematicsResult.bodyPoses,
            .pointJacobians = kinematicsResult.pointJacobians,
        };
        metalrobo::NumiHumanTendonMetalResult tendonResult;
        const auto tendonDiagnostics = metalrobo::runMetalNumiHumanTendonTransfer(
            tendonProgram, tendonInput, tendonResult
        );
        require(
            tendonDiagnostics.succeeded() && tendonDiagnostics.dispatched &&
                tendonDiagnostics.published &&
                tendonResult.transfers.size() == tendonProgram.bindings.size() &&
                tendonResult.generalizedCorrections.size() ==
                    tendonProgram.bindings.size() * articulation.nv,
            std::string("NHTENDON2 Metal transfer failed: ") +
                metalrobo::numiHumanTendonMetalStatusName(tendonDiagnostics.status) +
                " " + tendonDiagnostics.message
        );
        metalrobo::NumiHumanTendonMetalResult replayResult;
        const auto replayDiagnostics = metalrobo::runMetalNumiHumanTendonTransfer(
            tendonProgram, tendonInput, replayResult
        );
        metrics.tendonReplayByteIdentical = replayDiagnostics.succeeded() &&
            replayResult.transfers.size() == tendonResult.transfers.size() &&
            replayResult.generalizedCorrections.size() == tendonResult.generalizedCorrections.size() &&
            std::memcmp(
                replayResult.transfers.data(), tendonResult.transfers.data(),
                tendonResult.transfers.size() * sizeof(tendonResult.transfers.front())
            ) == 0 &&
            std::memcmp(
                replayResult.generalizedCorrections.data(),
                tendonResult.generalizedCorrections.data(),
                tendonResult.generalizedCorrections.size() * sizeof(float)
            ) == 0;
        require(metrics.tendonReplayByteIdentical,
                "NHTENDON2 Metal transfer replay is not byte-identical");
        metrics.tendonTransferCount = static_cast<std::uint32_t>(
            tendonResult.transfers.size()
        );
        for (std::size_t index = 0u; index < tendonResult.transfers.size(); ++index) {
            const auto& transfer = tendonResult.transfers[index];
            metrics.maximumTendonForceResidual = std::max(
                metrics.maximumTendonForceResidual,
                static_cast<double>(transfer.residualsAndForce.x)
            );
            metrics.maximumTendonMomentResidual = std::max(
                metrics.maximumTendonMomentResidual,
                static_cast<double>(transfer.residualsAndForce.y)
            );
            metrics.maximumTendonGeneralizedCorrection = std::max(
                metrics.maximumTendonGeneralizedCorrection,
                static_cast<double>(transfer.residualsAndForce.z)
            );
            const auto& binding = muscles.tendonPayload.bindings[index];
            if (binding.mode != metalrobo::NumiHumanTendonAttachmentMode::registeredBoneDistributedEnvelope) {
                continue;
            }
            ++metrics.tendonEnvelopeTransferCount;
            const std::size_t localBody = binding.bodyIndex - articulation.firstBody;
            const mr_float4 orientation = kinematicsResult.bodyPoses[localBody].orientation;
            const std::array<double, 4> conjugate{
                -orientation.x, -orientation.y, -orientation.z, orientation.w,
            };
            const std::array<double, 3> worldForce{
                transfer.terminalWorldForce.x,
                transfer.terminalWorldForce.y,
                transfer.terminalWorldForce.z,
            };
            const std::array<double, 3> localForce = quaternionRotate(conjugate, worldForce);
            metalrobo::NumiHumanTendonTractionResult cpuTraction;
            const auto cpuTractionDiagnostics =
                metalrobo::evaluateNumiHumanTendonEnvelopeTraction(
                    binding,
                    muscles.tendonPayload.envelopes[binding.triangleIndex],
                    localForce, cpuTraction
                );
            require(cpuTractionDiagnostics.succeeded(),
                    "NHTENDON2 CPU traction failed during Metal parity");
            const std::array<double, 4> rotation{
                orientation.x, orientation.y, orientation.z, orientation.w,
            };
            for (std::size_t node = 0u; node < 4u; ++node) {
                const std::array<double, 3> expected = quaternionRotate(
                    rotation, cpuTraction.nodalForces[node]
                );
                for (std::size_t axis = 0u; axis < 3u; ++axis) {
                    metrics.maximumTendonNodalForceParityError = std::max(
                        metrics.maximumTendonNodalForceParityError,
                        std::abs(expected[axis] - static_cast<double>(
                            (&transfer.nodalWorldForces[node].x)[axis]
                        ))
                    );
                }
            }
        }
        for (const float correction : tendonResult.generalizedCorrections) {
            metrics.maximumTendonGeneralizedCorrection = std::max(
                metrics.maximumTendonGeneralizedCorrection,
                std::abs(static_cast<double>(correction))
            );
        }
    }
    for (std::size_t body = 0u; body < cpuBodies.size(); ++body) {
        const MRArticulatedBodyPoseGPU& gpuBody = kinematicsResult.bodyPoses[body];
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            metrics.maximumBodyPositionError = std::max(
                metrics.maximumBodyPositionError,
                std::abs(static_cast<double>((&gpuBody.position.x)[axis]) -
                         cpuBodies[body].centerOfMassPosition[axis])
            );
        }
        double sameSign = 0.0;
        double flippedSign = 0.0;
        for (std::size_t component = 0u; component < 4u; ++component) {
            sameSign = std::max(
                sameSign, std::abs(static_cast<double>((&gpuBody.orientation.x)[component]) -
                                   cpuBodies[body].orientation[component])
            );
            flippedSign = std::max(
                flippedSign, std::abs(static_cast<double>((&gpuBody.orientation.x)[component]) +
                                      cpuBodies[body].orientation[component])
            );
        }
        metrics.maximumBodyOrientationComponentError = std::max(
            metrics.maximumBodyOrientationComponentError, std::min(sameSign, flippedSign)
        );
    }
    for (std::size_t point = 0u; point < cpuPointKinematics.size(); ++point) {
        const mr_float4& gpuPoint = kinematicsResult.pointWorld[point].position;
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            metrics.maximumPointPositionError = std::max(
                metrics.maximumPointPositionError,
                std::abs(static_cast<double>((&gpuPoint.x)[axis]) -
                         cpuPointKinematics[point].position[axis])
            );
        }
    }
    for (std::size_t index = 0u; index < cpuJacobians.size(); ++index) {
        metrics.maximumPointJacobianError = std::max(
            metrics.maximumPointJacobianError,
            std::abs(static_cast<double>(kinematicsResult.pointJacobians[index]) - cpuJacobians[index])
        );
    }
    std::vector<double> expectedActuatorForces(muscles.muscles.size(), 0.0);
    std::vector<double> expectedPathVelocities(muscles.muscles.size(), 0.0);
    std::vector<std::vector<double>> expectedMuscleForces(
        muscles.muscles.size(), std::vector<double>(articulation.nv, 0.0)
    );
    for (std::size_t muscleIndex = 0u;
         muscleIndex < muscles.muscles.size();
         ++muscleIndex) {
        metalrobo::MujocoMuscleResult sourceResult;
        const auto sourceDiagnostics = metalrobo::evaluateMujocoMuscle(
            model, 0u, qReference, referenceVelocity, muscles.sites, muscles.wraps,
            muscles.muscles[muscleIndex], {.excitation = 0.5, .activation = 0.5},
            sourceResult
        );
        require(sourceDiagnostics.succeeded(),
                "MyoSim CPU path reference failed before compliant parity");
        double actuatorForce = sourceResult.actuatorForce;
        if (muscles.architectures[muscleIndex].optimalFiberLength > 0.0) {
            metalrobo::MujocoCompliantMuscleResult compliantResult;
            const auto compliantDiagnostics =
                metalrobo::evaluateMujocoCompliantMuscle(
                    sourceResult.path.length, sourceResult.path.velocity, 1.0e-5,
                    muscles.muscles[muscleIndex], muscles.architectures[muscleIndex],
                    {.excitation = 0.5, .activation = 0.5}, compliantResult
                );
            require(compliantDiagnostics.succeeded(),
                    "NHMYO2 CPU compliant equilibrium failed before Metal parity");
            actuatorForce = compliantResult.actuatorForce;
        }
        expectedActuatorForces[muscleIndex] = actuatorForce;
        metrics.maximumReferenceMuscleForce = std::max(
            metrics.maximumReferenceMuscleForce,
            std::abs(actuatorForce)
        );
        expectedPathVelocities[muscleIndex] = sourceResult.path.velocity;
        for (std::size_t dof = 0u; dof < articulation.nv; ++dof) {
            expectedMuscleForces[muscleIndex][dof] =
                actuatorForce * sourceResult.path.lengthJacobian[dof];
            metrics.maximumReferenceMuscleGeneralizedForce = std::max(
                metrics.maximumReferenceMuscleGeneralizedForce,
                std::abs(expectedMuscleForces[muscleIndex][dof])
            );
        }
    }
    for (std::size_t index = 0u;
         index < kinematicsResult.mujocoResults.size();
         ++index) {
        const MRMujocoMuscleResultGPU& gpu =
            kinematicsResult.mujocoResults[index];
        metrics.maximumMuscleLengthError = std::max(
            metrics.maximumMuscleLengthError,
            std::abs(static_cast<double>(
                gpu.pathForceAndActivationDerivative.x
            ) - muscles.oracleLength[index])
        );
        metrics.maximumMusclePathVelocityError = std::max(
            metrics.maximumMusclePathVelocityError,
            std::abs(static_cast<double>(
                gpu.pathForceAndActivationDerivative.y
            ) - expectedPathVelocities[index])
        );
        metrics.maximumMuscleForceError = std::max(
            metrics.maximumMuscleForceError,
            std::abs(static_cast<double>(
                gpu.pathForceAndActivationDerivative.z
            ) - expectedActuatorForces[index])
        );
        metrics.maximumNormalizedTendonTension = std::max(
            metrics.maximumNormalizedTendonTension,
            std::abs(static_cast<double>(
                gpu.fiberStateTendonForceResidual.z
            ))
        );
        const double equilibriumResidual = std::abs(static_cast<double>(
            gpu.fiberStateTendonForceResidual.w
        ));
        if (equilibriumResidual > metrics.maximumNormalizedEquilibriumResidual) {
            metrics.maximumNormalizedEquilibriumResidual = equilibriumResidual;
            metrics.maximumNormalizedEquilibriumResidualMuscle =
                static_cast<std::uint32_t>(index);
        }
        metrics.appliedMuscleWraps += gpu.appliedWrapCount;
    }
    std::vector<double> expectedGeneralizedForce(articulation.nv, 0.0);
    for (std::size_t muscleIndex = 0u;
         muscleIndex < muscles.muscles.size();
         ++muscleIndex) {
        for (std::size_t dof = 0u; dof < articulation.nv; ++dof) {
            const std::size_t gpuIndex = muscleIndex * articulation.nv + dof;
            metrics.maximumMuscleGeneralizedForceError = std::max(
                metrics.maximumMuscleGeneralizedForceError,
                std::abs(static_cast<double>(
                    kinematicsResult.mujocoMuscleGeneralizedForces[gpuIndex]
                ) - expectedMuscleForces[muscleIndex][dof])
            );
            expectedGeneralizedForce[dof] += expectedMuscleForces[muscleIndex][dof];
        }
    }
    for (std::size_t dof = 0u; dof < articulation.nv; ++dof) {
        metrics.maximumReferenceSummedGeneralizedForce = std::max(
            metrics.maximumReferenceSummedGeneralizedForce,
            std::abs(expectedGeneralizedForce[dof])
        );
        metrics.maximumSummedGeneralizedForceError = std::max(
            metrics.maximumSummedGeneralizedForceError,
            std::abs(static_cast<double>(
                kinematicsResult.mujocoGeneralizedForces[dof]
            ) - expectedGeneralizedForce[dof])
        );
    }
    // Use deliberately non-equilibrium activation values to verify that the
    // new Metal sidecar update is performing an actual temporal advance, not
    // merely echoing the input state. Force parity above remains tied to the
    // source-default 0.5/0.5 state.
    constexpr float kActivationTimestepSeconds = 1.0e-4f;
    std::vector<MRMujocoMuscleStateGPU> activationStates = muscles.gpuStates;
    for (std::size_t index = 0u; index < activationStates.size(); ++index) {
        activationStates[index].excitationAndActivation.x =
            0.2f + 0.2f * static_cast<float>(index % 4u);
        activationStates[index].excitationAndActivation.y =
            index % 2u == 0u ? 0.35f : 0.65f;
    }
    metalrobo::MetalArticulatedOperatorInput activationInput = input;
    activationInput.mujoco.states = activationStates;
    const metalrobo::MetalArticulatedOperatorConfig activationConfig{
        .pointJacobiansOnly = true,
        .mujocoActivationTimestepSeconds = kActivationTimestepSeconds,
    };
    metalrobo::MetalArticulatedOperatorResult activationResult;
    const auto activationDiagnostics = metalrobo::runMetalArticulatedOperator(
        model, activationInput, activationResult, activationConfig
    );
    require(
        activationDiagnostics.succeeded() && activationDiagnostics.dispatched &&
            activationDiagnostics.published &&
            activationResult.mujocoActivationStates.size() ==
                activationStates.size() &&
            activationResult.mujocoResults.size() == activationStates.size(),
        std::string("MyoSim Metal activation-step operator failed: ") +
            metalrobo::metalArticulatedOperatorHostStatusName(
                activationDiagnostics.status
            ) + " " + activationDiagnostics.message
    );
    // The reusable context is the path a bounded rollout uses. Run the same
    // source transaction through its retained arena and require byte-identical
    // activation publication before trusting it as the next device timestep.
    metalrobo::MetalArticulatedOperatorContext activationContext(
        activationConfig
    );
    metalrobo::MetalArticulatedOperatorResult activationContextResult;
    const auto activationContextDiagnostics = activationContext.run(
        model, activationInput, activationContextResult
    );
    require(
        activationContextDiagnostics.succeeded() &&
            activationContextDiagnostics.dispatched &&
            activationContextDiagnostics.published &&
            activationContextResult.mujocoActivationStates.size() ==
                activationResult.mujocoActivationStates.size() &&
            std::memcmp(
                activationContextResult.mujocoActivationStates.data(),
                activationResult.mujocoActivationStates.data(),
                activationResult.mujocoActivationStates.size() *
                    sizeof(MRMujocoMuscleStateGPU)
            ) == 0,
        std::string("MyoSim persistent Metal activation-step operator failed: ") +
            metalrobo::metalArticulatedOperatorHostStatusName(
                activationContextDiagnostics.status
            ) + " " + activationContextDiagnostics.message
    );
    metrics.activationTimestepSeconds = kActivationTimestepSeconds;
    for (std::size_t index = 0u;
         index < activationStates.size();
         ++index) {
        const float initialActivation =
            activationStates[index].excitationAndActivation.y;
        const float derivative = activationResult.mujocoResults[index]
            .pathForceAndActivationDerivative.w;
        const float expectedActivation = std::clamp(
            initialActivation + kActivationTimestepSeconds * derivative,
            0.0f,
            1.0f
        );
        const MRMujocoMuscleStateGPU& advanced =
            activationResult.mujocoActivationStates[index];
        metrics.maximumActivationStepError = std::max(
            metrics.maximumActivationStepError,
            std::abs(static_cast<double>(
                advanced.excitationAndActivation.y - expectedActivation
            ))
        );
        require(
            advanced.excitationAndActivation.x ==
                    activationStates[index].excitationAndActivation.x &&
                (muscles.architectures[index].optimalFiberLength > 0.0
                    ? advanced.excitationAndActivation.z > 0.0f
                    : advanced.excitationAndActivation.z == 0.0f &&
                        advanced.excitationAndActivation.w == 0.0f),
            "MyoSim Metal activation step corrupted the source state sidecar"
        );
    }
    require(
        metrics.maximumBodyPositionError < 2.0e-4 &&
            metrics.maximumBodyOrientationComponentError < 2.0e-4 &&
            metrics.maximumPointPositionError < 2.0e-4 &&
            metrics.maximumPointJacobianError < 5.0e-4 &&
            metrics.maximumMuscleLengthError < 2.0e-4 &&
            metrics.maximumMusclePathVelocityError < 2.0e-5 &&
            metrics.maximumMuscleForceError < std::max(
                5.0e-2, 2.0e-4 * metrics.maximumReferenceMuscleForce
            ) &&
            metrics.maximumMuscleGeneralizedForceError < std::max(
                5.0e-2,
                1.0e-4 * metrics.maximumReferenceMuscleGeneralizedForce
            ) &&
            metrics.maximumSummedGeneralizedForceError < std::max(
                2.0e-1,
                2.0e-4 * metrics.maximumReferenceSummedGeneralizedForce
            ) &&
            metrics.maximumActivationStepError < 2.0e-6 &&
            metrics.maximumTendonNodalForceParityError < 2.0e-2 &&
            metrics.maximumTendonForceResidual < 2.0e-2 &&
            metrics.maximumTendonMomentResidual < 2.0e-4 &&
            metrics.maximumTendonGeneralizedCorrection < 2.0e-2 &&
            metrics.appliedMuscleWraps == 90u,
        "MyoSim Metal kinematics/Jacobian/muscle-route parity exceeded FP32 tolerance: "
            "body=" + std::to_string(metrics.maximumBodyPositionError) +
            " orientation=" + std::to_string(metrics.maximumBodyOrientationComponentError) +
            " point=" + std::to_string(metrics.maximumPointPositionError) +
            " jacobian=" + std::to_string(metrics.maximumPointJacobianError) +
            " muscle_length=" + std::to_string(metrics.maximumMuscleLengthError) +
            " muscle_path_velocity=" + std::to_string(metrics.maximumMusclePathVelocityError) +
            " muscle_force=" + std::to_string(metrics.maximumMuscleForceError) +
            " muscle_force_scale=" +
                std::to_string(metrics.maximumReferenceMuscleForce) +
            " muscle_generalized_force=" + std::to_string(metrics.maximumMuscleGeneralizedForceError) +
            " muscle_generalized_force_scale=" +
                std::to_string(metrics.maximumReferenceMuscleGeneralizedForce) +
            " summed_generalized_force=" + std::to_string(metrics.maximumSummedGeneralizedForceError) +
            " summed_generalized_force_scale=" +
                std::to_string(metrics.maximumReferenceSummedGeneralizedForce) +
            " activation_step=" + std::to_string(metrics.maximumActivationStepError) +
            " tendon_nodal_force=" + std::to_string(metrics.maximumTendonNodalForceParityError) +
            " tendon_force_residual=" + std::to_string(metrics.maximumTendonForceResidual) +
            " tendon_moment_residual=" + std::to_string(metrics.maximumTendonMomentResidual) +
            " tendon_generalized_correction=" + std::to_string(metrics.maximumTendonGeneralizedCorrection) +
            " wraps=" + std::to_string(metrics.appliedMuscleWraps)
    );

    return metrics;
}

int run(
    const char* rigidPath, const char* musclePath,
    const char* tendonPath, const char* equalityPath,
    const bool runMetal, const bool runEquilibrium
) {
    const LoadedRigid rigid = loadRigid(rigidPath);
    LoadedMuscles muscles = loadMuscles(musclePath, rigid.header);
    if (tendonPath != nullptr) {
        applyNumiHumanTendonPayload(tendonPath, rigid.header, muscles);
    }
    const metalrobo::NumiHumanJointEqualityPayload equalities =
        equalityPath == nullptr
            ? metalrobo::NumiHumanJointEqualityPayload{}
            : loadJointEqualities(equalityPath, rigid.header);
    const auto& model = rigid.model;
    std::vector<double> q(model.defaultQ.begin(), model.defaultQ.end());
    std::vector<double> v(model.defaultV.begin(), model.defaultV.end());
    metalrobo::NumiHumanMuscleEquilibriumResult equilibrium;
    if (runEquilibrium) {
        const auto equilibriumDiagnostics =
            metalrobo::compileNumiHumanMuscleEquilibrium(
                model, 0u, q, muscles.sites, muscles.wraps, muscles.muscles,
                muscles.architectures, equalities.records, {}, equilibrium
            );
        require(
            equilibriumDiagnostics.succeeded(),
            std::string("full-body equilibrium compile failed: ") +
                metalrobo::numiHumanMuscleEquilibriumStatusName(
                    equilibriumDiagnostics.status
                ) + " index=" +
                std::to_string(equilibriumDiagnostics.failingIndex)
        );
    }
    std::vector<metalrobo::ArticulatedBodyKinematics> bodies(model.bodies.size());
    const auto bodyDiagnostics = metalrobo::computeArticulatedBodyKinematics(model, 0u, q, v, bodies);
    require(bodyDiagnostics.succeeded(), "MyoSim Core default body kinematics failed");
    double maxPositionError = 0.0;
    double maxOrientationError = 0.0;
    for (std::size_t index = 0; index < rigid.sourceBodyToCore.size(); ++index) {
        const auto& native = bodies[rigid.sourceBodyToCore[index]];
        const SourcePoseRecord& source = rigid.sourceDefaultPoses[index];
        maxPositionError = std::max(maxPositionError, vectorNorm({
            native.centerOfMassPosition[0] - source.positionX,
            native.centerOfMassPosition[1] - source.positionY,
            native.centerOfMassPosition[2] - source.positionZ,
        }));
        maxOrientationError = std::max(maxOrientationError, quaternionAngle(native.orientation, source));
    }
    require(maxPositionError <= 2.0e-5, "MyoSim Core default body pose does not match source");
    require(maxOrientationError <= 2.0e-5, "MyoSim Core default body orientation does not match source");
    std::vector<double> mass(model.world.nv * model.world.nv);
    const auto massDiagnostics = metalrobo::computeArticulatedMassMatrix(model, 0u, q, mass);
    require(massDiagnostics.succeeded(), "MyoSim Core mass matrix failed");
    std::vector<double> acceleration(model.world.nv);
    for (std::size_t index = 0; index < acceleration.size(); ++index) acceleration[index] = 0.003 * std::sin(static_cast<double>(index + 1u));
    std::vector<double> inverseForce(model.world.nv);
    const auto inverseDiagnostics = metalrobo::computeArticulatedInverseDynamics(model, 0u, q, v, acceleration, {}, inverseForce);
    require(inverseDiagnostics.succeeded(), "MyoSim Core inverse dynamics failed status=" +
            std::to_string(static_cast<std::uint32_t>(inverseDiagnostics.status)));
    std::vector<double> recovered(acceleration.size());
    const auto forwardDiagnostics = metalrobo::computeArticulatedForwardDynamics(model, 0u, q, v, inverseForce, {}, recovered);
    require(forwardDiagnostics.succeeded(), "MyoSim Core forward dynamics failed status=" +
            std::to_string(static_cast<std::uint32_t>(forwardDiagnostics.status)));
    double maxDynamicsError = 0.0;
    for (std::size_t index = 0; index < acceleration.size(); ++index) maxDynamicsError = std::max(maxDynamicsError, std::abs(recovered[index] - acceleration[index]));
    require(maxDynamicsError <= 2.0e-9, "MyoSim Core inverse/forward dynamics is inconsistent");
    double maxMuscleLengthError = 0.0;
    double maxMuscleForceError = 0.0;
    std::uint32_t appliedWraps = 0u;
    std::vector<double> muscleForce(model.world.nv, 0.0);
    std::vector<double> sourceMuscleForce(model.world.nv, 0.0);
    double maximumEnthesisForceResidual = 0.0;
    double maximumEnthesisMomentResidual = 0.0;
    for (std::size_t index = 0; index < muscles.muscles.size(); ++index) {
        metalrobo::MujocoMuscleResult result;
        const metalrobo::MujocoMuscleState state{.excitation = 0.5, .activation = 0.5};
        const auto diagnostics = metalrobo::projectMujocoMuscleForce(
            model, 0u, q, v, muscles.sites, muscles.wraps, muscles.muscles[index], state, muscleForce, &result
        );
        require(diagnostics.succeeded(), std::string("MyoSim muscle ") + std::to_string(index) + " failed: " +
                                           metalrobo::mujocoMuscleReferenceStatusName(diagnostics.status));
        maxMuscleLengthError = std::max(maxMuscleLengthError, std::abs(result.path.length - muscles.oracleLength[index]));
        maxMuscleForceError = std::max(maxMuscleForceError, std::abs(result.actuatorForce - muscles.oracleForce[index]));
        appliedWraps += result.path.appliedWrapCount;
        if (!muscles.tendonPayload.bindings.empty()) {
            require(result.path.centreline.size() >= 2u, "resolved tendon route has no endpoint direction");
            for (std::uint32_t endpoint = 0u; endpoint < 2u; ++endpoint) {
                const metalrobo::NumiHumanTendonBinding& binding =
                    muscles.tendonPayload.bindings[2u * index + endpoint];
                const auto& terminal = endpoint == 0u
                    ? result.path.centreline.front().world : result.path.centreline.back().world;
                const auto& adjacent = endpoint == 0u
                    ? result.path.centreline[1u].world
                    : result.path.centreline[result.path.centreline.size() - 2u].world;
                const MRArticulationGPU& articulation = model.articulations[0];
                require(
                    binding.bodyIndex >= articulation.firstBody &&
                    binding.bodyIndex < articulation.firstBody + articulation.bodyCount,
                    "resolved tendon endpoint body is outside the articulation"
                );
                const auto& pose = bodies[binding.bodyIndex - articulation.firstBody];
                std::array<std::array<double, 3>, 3> worldTriangle{};
                std::span<const std::array<double, 3>> worldTriangleSpan{};
                if (binding.mode == metalrobo::NumiHumanTendonAttachmentMode::registeredBoneTriangle) {
                    const auto& triangle = muscles.tendonPayload.triangles[binding.triangleIndex];
                    for (std::size_t vertex = 0u; vertex < 3u; ++vertex) {
                        const std::array<double, 3> rotated = quaternionRotate(
                            pose.orientation, triangle.localVertices[vertex]
                        );
                        for (std::size_t axis = 0u; axis < 3u; ++axis) {
                            worldTriangle[vertex][axis] = pose.centerOfMassPosition[axis] + rotated[axis];
                        }
                    }
                    worldTriangleSpan = worldTriangle;
                }
                metalrobo::NumiHumanTendonTractionResult traction;
                metalrobo::NumiHumanTendonDiagnostics tractionDiagnostics;
                if (binding.mode == metalrobo::NumiHumanTendonAttachmentMode::registeredBoneDistributedEnvelope) {
                    const std::array<double, 3> difference{
                        terminal[0] - adjacent[0], terminal[1] - adjacent[1], terminal[2] - adjacent[2],
                    };
                    const double length = vectorNorm(difference);
                    require(length > 1.0e-12, "distributed tendon endpoint has no terminal direction");
                    const std::array<double, 3> worldForce{
                        result.actuatorForce * difference[0] / length,
                        result.actuatorForce * difference[1] / length,
                        result.actuatorForce * difference[2] / length,
                    };
                    const std::array<double, 4> conjugate{
                        -pose.orientation[0], -pose.orientation[1], -pose.orientation[2], pose.orientation[3],
                    };
                    const std::array<double, 3> localForce = quaternionRotate(conjugate, worldForce);
                    tractionDiagnostics = metalrobo::evaluateNumiHumanTendonEnvelopeTraction(
                        binding, muscles.tendonPayload.envelopes[binding.triangleIndex],
                        localForce, traction
                    );
                } else {
                    tractionDiagnostics = metalrobo::evaluateNumiHumanTendonTraction(
                        binding, worldTriangleSpan, terminal, adjacent,
                        pose.centerOfMassPosition, result.actuatorForce, traction
                    );
                }
                require(
                    tractionDiagnostics.succeeded(),
                    std::string("tendon traction evaluation failed: ") +
                        metalrobo::numiHumanTendonStatusName(tractionDiagnostics.status)
                );
                maximumEnthesisForceResidual = std::max(
                    maximumEnthesisForceResidual, traction.forceResidual
                );
                maximumEnthesisMomentResidual = std::max(
                    maximumEnthesisMomentResidual, traction.momentResidual
                );
            }
        }
        if (!muscles.sourceMuscles.empty()) {
            const auto sourceDiagnostics = metalrobo::projectMujocoMuscleForce(
                model, 0u, q, v, muscles.sourceSites, muscles.wraps,
                muscles.sourceMuscles[index], state, sourceMuscleForce
            );
            require(sourceDiagnostics.succeeded(), "source endpoint comparison failed");
        }
        if (muscles.tendonTriangleBindings > 0u) {
            // Metal parity compares against the resolved CPU route. The source
            // oracle delta above remains reported as explicit endpoint migration.
            muscles.oracleLength[index] = result.path.length;
            muscles.oracleForce[index] = result.actuatorForce;
        }
    }
    if (muscles.tendonTriangleBindings == 0u) {
        require(maxMuscleLengthError <= 2.0e-5, "MyoSim native muscle paths do not match source default lengths");
        require(maxMuscleForceError <= 1.0e-2, "MyoSim native muscle forces do not match source default forces");
    }
    double maximumEndpointSingleScatterDifference = 0.0;
    if (!muscles.sourceMuscles.empty()) {
        for (std::size_t index = 0u; index < muscleForce.size(); ++index) {
            maximumEndpointSingleScatterDifference = std::max(
                maximumEndpointSingleScatterDifference,
                std::abs(muscleForce[index] - sourceMuscleForce[index])
            );
        }
        if (muscles.tendonTriangleBindings == 0u) {
            require(
                maximumEndpointSingleScatterDifference <= 1.0e-10,
                "source-point-preserving NHTENDON changed or duplicated the authoritative J^T force"
            );
        }
    }
    require(
        maximumEnthesisForceResidual <=
            (muscles.tendonEnvelopeBindings > 0u ? 1.0e-4 : 1.0e-9) &&
        maximumEnthesisMomentResidual <=
            (muscles.tendonEnvelopeBindings > 0u ? 1.0e-5 : 1.0e-4),
        "NHTENDON traction does not preserve endpoint force and moment: force=" +
            std::to_string(maximumEnthesisForceResidual) + " moment=" +
            std::to_string(maximumEnthesisMomentResidual)
    );
    std::vector<double> muscleAcceleration(model.world.nv);
    const auto muscleDynamics = metalrobo::computeArticulatedForwardDynamics(model, 0u, q, v, muscleForce, {}, muscleAcceleration);
    require(muscleDynamics.succeeded(), "MyoSim native muscle force did not drive Core forward dynamics");
    require(std::all_of(muscleAcceleration.begin(), muscleAcceleration.end(), [](const double value) { return std::isfinite(value); }),
            "MyoSim muscle-driven acceleration is non-finite");
    // A force vector and a forward-dynamics acceleration are not by themselves
    // evidence that the source muscles advance the articulated state. Compare
    // the same free floating full body for one deterministic 1 us Core step
    // with and without the complete 416-muscle generalized force. Gravity,
    // damping, timestep, and source default state remain identical.
    std::vector<double> passiveQ = q;
    std::vector<double> passiveV = v;
    std::vector<double> muscleDrivenQ = q;
    std::vector<double> muscleDrivenV = v;
    const std::vector<double> zeroForce(model.world.nv, 0.0);
    metalrobo::ArticulatedDynamicsConfig sensitivityConfig;
    sensitivityConfig.timestep = 1.0e-6;
    const auto passiveStep = metalrobo::integrateArticulatedState(
        model, 0u, passiveQ, passiveV, zeroForce, {}, sensitivityConfig
    );
    require(passiveStep.succeeded(), "MyoSim passive free-body integration failed");
    const auto muscleDrivenStep = metalrobo::integrateArticulatedState(
        model, 0u, muscleDrivenQ, muscleDrivenV, muscleForce, {}, sensitivityConfig
    );
    require(muscleDrivenStep.succeeded(), "MyoSim muscle-driven free-body integration failed");
    double maximumMuscleDrivenVelocityDelta = 0.0;
    double maximumMuscleDrivenConfigurationDelta = 0.0;
    for (std::size_t index = 0u; index < muscleDrivenV.size(); ++index) {
        maximumMuscleDrivenVelocityDelta = std::max(
            maximumMuscleDrivenVelocityDelta,
            std::abs(muscleDrivenV[index] - passiveV[index])
        );
    }
    for (std::size_t index = 0u; index < muscleDrivenQ.size(); ++index) {
        maximumMuscleDrivenConfigurationDelta = std::max(
            maximumMuscleDrivenConfigurationDelta,
            std::abs(muscleDrivenQ[index] - passiveQ[index])
        );
    }
    require(
        std::isfinite(maximumMuscleDrivenVelocityDelta) &&
            std::isfinite(maximumMuscleDrivenConfigurationDelta) &&
            maximumMuscleDrivenVelocityDelta > 1.0e-9 &&
            maximumMuscleDrivenConfigurationDelta > 1.0e-12,
        "MyoSim muscle force did not produce a distinguishable articulated state step"
    );
    const MetalArticulatedMetrics metal = runMetal
        ? verifyMetalArticulatedReference(model, muscles)
        : MetalArticulatedMetrics{};
    auto& output = std::cout << std::setprecision(12)
                             << "myosim_core_reference PASS"
                             << " source_bodies=" << rigid.header.sourceBodyCount
                             << " core_bodies=" << rigid.header.engineBodyCount
                             << " virtual_carriers=" << rigid.header.virtualBodyCount
                             << " nq=" << rigid.header.nq << " nv=" << rigid.header.nv
                             << " muscles=" << muscles.muscles.size()
                             << " route_sites=" << muscles.sites.size()
                             << " tendon_endpoints="
                             << muscles.tendonPointBindings + muscles.tendonTriangleBindings +
                                    muscles.tendonEnvelopeBindings
                             << " tendon_point_bindings=" << muscles.tendonPointBindings
                             << " tendon_triangle_bindings=" << muscles.tendonTriangleBindings
                             << " tendon_envelope_bindings=" << muscles.tendonEnvelopeBindings
                             << " tendon_max_endpoint_migration_m=" << muscles.maximumEndpointMigration
                             << " tendon_single_scatter_generalized_force_difference="
                             << maximumEndpointSingleScatterDifference
                             << " tendon_force_residual_n=" << maximumEnthesisForceResidual
                             << " tendon_moment_residual_nm=" << maximumEnthesisMomentResidual
                             << " wraps=" << muscles.wraps.size()
                             << " applied_wraps=" << appliedWraps
                             << " max_body_position_error_m=" << maxPositionError
                             << " max_body_orientation_error_rad=" << maxOrientationError
                             << " max_muscle_length_error_m=" << maxMuscleLengthError
                             << " max_muscle_force_error_n=" << maxMuscleForceError
                             << " muscle_driven_sensitivity_step_seconds=" << sensitivityConfig.timestep
                             << " muscle_driven_max_velocity_delta=" << maximumMuscleDrivenVelocityDelta
                             << " muscle_driven_max_configuration_delta=" << maximumMuscleDrivenConfigurationDelta
                             << " max_inverse_forward_error=" << maxDynamicsError
                             << " mass_min_pivot=" << massDiagnostics.minimumCholeskyPivot
                             << " mass_condition=" << massDiagnostics.estimatedMassMatrixCondition;
    if (runMetal) {
        output << " metal_stage=kinematics_jacobians_muscle_route_generalized_force"
               << " metal_device=\"" << metal.deviceName << "\""
               << " metal_max_body_position_error_m=" << metal.maximumBodyPositionError
               << " metal_max_body_orientation_component_error="
               << metal.maximumBodyOrientationComponentError
               << " metal_max_point_position_error_m=" << metal.maximumPointPositionError
               << " metal_max_point_jacobian_error=" << metal.maximumPointJacobianError
               << " metal_max_muscle_length_error_m="
               << metal.maximumMuscleLengthError
               << " metal_max_muscle_path_velocity_error_m_s="
               << metal.maximumMusclePathVelocityError
               << " metal_max_muscle_force_error_n="
               << metal.maximumMuscleForceError
               << " metal_max_reference_muscle_force_n="
               << metal.maximumReferenceMuscleForce
               << " metal_max_normalized_tendon_tension="
               << metal.maximumNormalizedTendonTension
               << " metal_max_normalized_equilibrium_residual="
               << metal.maximumNormalizedEquilibriumResidual
               << " metal_max_normalized_equilibrium_residual_muscle="
               << metal.maximumNormalizedEquilibriumResidualMuscle
               << " metal_max_muscle_generalized_force_error="
               << metal.maximumMuscleGeneralizedForceError
               << " metal_max_reference_muscle_generalized_force="
               << metal.maximumReferenceMuscleGeneralizedForce
               << " metal_max_summed_generalized_force_error="
               << metal.maximumSummedGeneralizedForceError
               << " metal_max_reference_summed_generalized_force="
               << metal.maximumReferenceSummedGeneralizedForce
               << " metal_activation_timestep_seconds="
               << metal.activationTimestepSeconds
               << " metal_max_activation_step_error="
               << metal.maximumActivationStepError
               << " metal_applied_wraps=" << metal.appliedMuscleWraps
               << " metal_tendon_transfers=" << metal.tendonTransferCount
               << " metal_tendon_envelope_transfers=" << metal.tendonEnvelopeTransferCount
               << " metal_tendon_max_nodal_force_parity_error_n="
               << metal.maximumTendonNodalForceParityError
               << " metal_tendon_max_force_residual_n=" << metal.maximumTendonForceResidual
               << " metal_tendon_max_moment_residual_nm=" << metal.maximumTendonMomentResidual
               << " metal_tendon_max_generalized_correction="
               << metal.maximumTendonGeneralizedCorrection
               << " metal_tendon_replay_byte_identical="
               << (metal.tendonReplayByteIdentical ? "true" : "false");
    }
    if (runEquilibrium) {
        output << " equilibrium_stage=bounded_pose_recruitment_compile"
               << " equilibrium_initial_normalized_residual_rms="
               << equilibrium.diagnostics.initialNormalizedResidualRms
               << " equilibrium_normalized_residual_rms="
               << equilibrium.diagnostics.normalizedResidualRms
               << " equilibrium_max_generalized_residual="
               << equilibrium.diagnostics.maximumGeneralizedForceResidual
               << " equilibrium_max_normalized_acceleration_residual="
               << equilibrium.diagnostics.maximumNormalizedAccelerationResidual
               << " equilibrium_max_normalized_acceleration_residual_dof="
               << equilibrium.diagnostics.maximumNormalizedResidualDof
               << " equilibrium_max_acceleration_residual="
               << equilibrium.diagnostics.maximumGeneralizedAccelerationResidual
               << " equilibrium_max_acceleration_residual_dof="
               << equilibrium.diagnostics.maximumAccelerationResidualDof
               << " equilibrium_active_muscles="
               << equilibrium.diagnostics.activeMuscleCount
               << " equilibrium_recruited_muscles="
               << equilibrium.diagnostics.recruitedMuscleCount
               << " equilibrium_max_activation="
               << equilibrium.diagnostics.maximumActivation
               << " equilibrium_pose_steps="
               << equilibrium.diagnostics.acceptedPoseSteps
               << " equilibrium_joint_equalities="
               << equilibrium.diagnostics.jointEqualityCount
               << " equilibrium_max_initial_equality_projection="
               << equilibrium.diagnostics.maximumInitialEqualityProjection
               << " equilibrium_max_equality_error="
               << equilibrium.diagnostics.maximumJointEqualityError
               << " equilibrium_max_equality_reaction="
               << equilibrium.diagnostics.maximumJointEqualityReaction
               << " equilibrium_active_position_limits="
               << equilibrium.diagnostics.activePositionLimitCount
               << " equilibrium_max_position_limit_reaction="
               << equilibrium.diagnostics.maximumPositionLimitReaction
               << " equilibrium_min_normalized_limit_margin="
               << equilibrium.diagnostics.minimumNormalizedPositionLimitMargin
               << " equilibrium_balanced="
               << (equilibrium.diagnostics.balanced ? "true" : "false");
    }
    output << "\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 7) {
            std::cerr << "usage: " << argv[0] << " <myosim-fullbody-core-reference.nhrigid> "
                      << "<myosim-fullbody-muscle-reference.nhmyo> "
                      << "[numi-human-tendon-endpoints.nhtendon] [--metal] "
                         "[myosim-fullbody-joint-equalities.nheq] "
                         "[--equilibrium]\n";
            return 2;
        }
        const char* tendonPath = nullptr;
        const char* equalityPath = nullptr;
        bool runMetal = false;
        bool runEquilibrium = false;
        for (int index = 3; index < argc; ++index) {
            if (std::string(argv[index]) == "--metal") {
                if (runMetal) return 2;
                runMetal = true;
            } else if (std::string(argv[index]) == "--equilibrium") {
                if (runEquilibrium) return 2;
                runEquilibrium = true;
            } else if (std::string(argv[index]).ends_with(".nheq") &&
                       equalityPath == nullptr) {
                equalityPath = argv[index];
            } else if (tendonPath == nullptr) {
                tendonPath = argv[index];
            } else {
                std::cerr << "usage: " << argv[0] << " <myosim-fullbody-core-reference.nhrigid> "
                          << "<myosim-fullbody-muscle-reference.nhmyo> "
                          << "[numi-human-tendon-endpoints.nhtendon] [--metal] "
                             "[myosim-fullbody-joint-equalities.nheq] "
                             "[--equilibrium]\n";
                return 2;
            }
        }
        return run(
            argv[1], argv[2], tendonPath, equalityPath, runMetal,
            runEquilibrium
        );
    } catch (const std::exception& error) {
        std::cerr << "myosim_core_reference FAIL: " << error.what() << "\n";
        return 1;
    }
}
