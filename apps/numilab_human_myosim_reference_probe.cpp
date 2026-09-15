#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/compensated_translation_gpu.h"
#include "metalrobo/MujocoMuscleReference.hpp"
#include "metalrobo/NumiHumanJointEquality.hpp"
#include "metalrobo/NumiHumanInitialState.hpp"
#include "metalrobo/NumiHumanMuscleEquilibrium.hpp"
#include "metalrobo/NumiHumanTendon.hpp"
#include "metalrobo/NumiHumanTendonMetal.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
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
    std::uint32_t tendonMigratedEnvelopeBindings = 0u;
    double maximumEndpointMigration = 0.0;
    double maximumTendonReferencePathDelta = 0.0;
    double maximumTendonArchitectureScaleChange = 0.0;
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

struct EqualityDerivativeAudit {
    std::uint32_t checkedCount = 0u;
    std::uint32_t maximumAbsoluteErrorIndex = MR_INVALID_INDEX;
    std::uint32_t maximumRelativeErrorIndex = MR_INVALID_INDEX;
    double maximumAbsoluteError = 0.0;
    double maximumRelativeError = 0.0;
};

constexpr double kJointEqualityDerivativeRelativeTolerance = 2.0e-7;

EqualityDerivativeAudit auditJointEqualityDerivatives(
    const std::span<const MRNumiHumanJointEqualityGPU> equalities,
    const std::span<const double> q,
    const std::string_view poseName
) {
    require(!equalities.empty(),
            "joint-equality derivative audit requires at least one equality");
    EqualityDerivativeAudit audit;
    for (std::size_t index = 0u; index < equalities.size(); ++index) {
        const auto& equality = equalities[index];
        metalrobo::NumiHumanJointEqualityEvaluation center;
        require(
            metalrobo::evaluateNumiHumanJointEquality(equality, q, center).succeeded(),
            "joint-equality derivative audit could not evaluate " +
                std::to_string(index) + " at " + std::string(poseName)
        );
        if (equality.indices.z == MR_INVALID_INDEX) {
            require(center.derivative == 0.0,
                    "constant joint equality has a nonzero derivative at " +
                        std::to_string(index));
            continue;
        }
        const double coordinate = q[equality.indices.z];
        const double h = 1.0e-5 * std::max(1.0, std::abs(coordinate));
        std::vector<double> positive(q.begin(), q.end());
        std::vector<double> negative(q.begin(), q.end());
        positive[equality.indices.z] += h;
        negative[equality.indices.z] -= h;
        metalrobo::NumiHumanJointEqualityEvaluation plus;
        metalrobo::NumiHumanJointEqualityEvaluation minus;
        require(
            metalrobo::evaluateNumiHumanJointEquality(equality, positive, plus).succeeded() &&
                metalrobo::evaluateNumiHumanJointEquality(equality, negative, minus).succeeded(),
            "joint-equality derivative finite-difference evaluation failed at " +
                std::to_string(index) + " at " + std::string(poseName)
        );
        const double finiteDifference =
            (plus.dependentTarget - minus.dependentTarget) / (2.0 * h);
        const double absoluteError = std::abs(finiteDifference - center.derivative);
        const double scale = std::max(
            {1.0, std::abs(finiteDifference), std::abs(center.derivative)}
        );
        const double relativeError = absoluteError / scale;
        if (absoluteError > audit.maximumAbsoluteError) {
            audit.maximumAbsoluteError = absoluteError;
            audit.maximumAbsoluteErrorIndex = static_cast<std::uint32_t>(index);
        }
        if (relativeError > audit.maximumRelativeError) {
            audit.maximumRelativeError = relativeError;
            audit.maximumRelativeErrorIndex = static_cast<std::uint32_t>(index);
        }
        require(relativeError <= kJointEqualityDerivativeRelativeTolerance,
                "joint-equality derivative finite-difference mismatch at " +
                    std::to_string(index) + " at " + std::string(poseName) +
                    " relative_error=" + std::to_string(relativeError));
        ++audit.checkedCount;
    }
    return audit;
}

int runJointEqualityDerivativeAudit(
    const char* rigidPath, const char* equalityPath
) {
    const LoadedRigid rigid = loadRigid(rigidPath);
    const metalrobo::NumiHumanJointEqualityPayload equalities =
        loadJointEqualities(equalityPath, rigid.header);
    std::vector<double> authoredQ(
        rigid.model.defaultQ.begin(), rigid.model.defaultQ.end()
    );
    const EqualityDerivativeAudit authored = auditJointEqualityDerivatives(
        equalities.records, authoredQ, "authored_q"
    );
    std::vector<double> projectedQ = authoredQ;
    double maximumProjection = 0.0;
    require(
        metalrobo::projectNumiHumanJointEqualities(
            equalities.records, projectedQ, &maximumProjection
        ).succeeded(),
        "joint-equality derivative audit could not project the authored pose"
    );
    const EqualityDerivativeAudit projected = auditJointEqualityDerivatives(
        equalities.records, projectedQ, "projected_q"
    );
    std::cout << std::setprecision(17)
              << "numi_human_joint_equality_derivative_audit="
              << "{\"schema\":\"numi.human.joint-equality-derivative-audit.v1\""
              << ",\"equality_count\":" << equalities.records.size()
              << ",\"maximum_projection_m_or_rad\":" << maximumProjection
              << ",\"authored_checked_count\":" << authored.checkedCount
              << ",\"authored_maximum_absolute_error\":"
              << authored.maximumAbsoluteError
              << ",\"authored_maximum_absolute_error_index\":"
              << authored.maximumAbsoluteErrorIndex
              << ",\"authored_maximum_relative_error\":"
              << authored.maximumRelativeError
              << ",\"authored_maximum_relative_error_index\":"
              << authored.maximumRelativeErrorIndex
              << ",\"projected_checked_count\":" << projected.checkedCount
              << ",\"projected_maximum_absolute_error\":"
              << projected.maximumAbsoluteError
              << ",\"projected_maximum_absolute_error_index\":"
              << projected.maximumAbsoluteErrorIndex
              << ",\"projected_maximum_relative_error\":"
              << projected.maximumRelativeError
              << ",\"projected_maximum_relative_error_index\":"
              << projected.maximumRelativeErrorIndex
              << ",\"relative_tolerance\":"
              << kJointEqualityDerivativeRelativeTolerance
              << ",\"passed\":true}\n";
    return 0;
}

[[nodiscard]] std::uint32_t parseUnsignedArgument(
    const char* const text,
    const char* const name
) {
    const char* const end = text + std::strlen(text);
    std::uint32_t value = 0u;
    const auto parsed = std::from_chars(text, end, value);
    require(
        parsed.ec == std::errc{} && parsed.ptr == end,
        std::string(name) + " must be an unsigned integer"
    );
    return value;
}

[[nodiscard]] std::vector<double> loadTraceArray(
    const std::string& contents,
    const std::size_t recordBegin,
    const std::size_t recordEnd,
    const std::string_view key,
    const std::size_t expectedCount
) {
    const std::string prefix{"\"" + std::string(key) + "\":["};
    const std::size_t beginWithPrefix = contents.find(prefix, recordBegin);
    require(
        beginWithPrefix != std::string::npos && beginWithPrefix < recordEnd,
        "requested stand-trace record has no " + std::string(key) + " array"
    );
    const std::size_t begin = beginWithPrefix + prefix.size();
    const std::size_t end = contents.find(']', begin);
    require(end != std::string::npos && end < recordEnd,
            "stand-trace " + std::string(key) + " array is truncated");

    const std::string_view encoded(contents.data() + begin, end - begin);
    std::vector<double> values;
    std::size_t cursor = 0u;
    while (cursor < encoded.size()) {
        const std::size_t comma = encoded.find(',', cursor);
        const std::size_t tokenEnd = comma == std::string_view::npos
            ? encoded.size()
            : comma;
        std::size_t tokenBegin = cursor;
        while (tokenBegin < tokenEnd && std::isspace(
            static_cast<unsigned char>(encoded[tokenBegin])
        )) {
            ++tokenBegin;
        }
        std::size_t trimmedEnd = tokenEnd;
        while (trimmedEnd > tokenBegin && std::isspace(
            static_cast<unsigned char>(encoded[trimmedEnd - 1u])
        )) {
            --trimmedEnd;
        }
        require(tokenBegin < trimmedEnd,
                "stand-trace " + std::string(key) + " has an empty element");
        const std::string token(encoded.substr(tokenBegin, trimmedEnd - tokenBegin));
        char* parsedEnd = nullptr;
        const double value = std::strtod(token.c_str(), &parsedEnd);
        require(
            parsedEnd != token.c_str() && *parsedEnd == '\0' && std::isfinite(value),
            "stand-trace " + std::string(key) + " has a non-finite or invalid element"
        );
        values.push_back(value);
        require(values.size() <= expectedCount,
                "stand-trace " + std::string(key) + " has too many elements");
        if (comma == std::string_view::npos) break;
        cursor = comma + 1u;
    }
    require(values.size() == expectedCount,
            "stand-trace " + std::string(key) + " count disagrees with NHRIGID2");
    return values;
}

struct TraceState {
    std::vector<double> q;
    std::vector<double> v;
};

[[nodiscard]] TraceState loadTraceState(
    const char* const tracePath,
    const std::uint32_t step,
    const std::size_t expectedQCount,
    const std::size_t expectedVCount
) {
    std::ifstream input(tracePath, std::ios::binary | std::ios::ate);
    require(input.is_open(), std::string("cannot open stand trace ") + tracePath);
    const std::streamsize size = input.tellg();
    require(size >= 0, "cannot determine stand trace size");
    input.seekg(0, std::ios::beg);
    std::string contents(static_cast<std::size_t>(size), '\0');
    if (!contents.empty()) {
        input.read(contents.data(), size);
        require(input.good(), "truncated stand trace");
    }

    const std::string stepPrefix =
        "{\"step\":" + std::to_string(step) + ",";
    const std::size_t recordBegin = contents.find(stepPrefix);
    require(recordBegin != std::string::npos,
            "requested step is absent from stand trace");
    const std::size_t nextRecord = contents.find("{\"step\":", recordBegin + 1u);
    const std::size_t recordEnd = nextRecord == std::string::npos
        ? contents.size()
        : nextRecord;
    return {
        loadTraceArray(contents, recordBegin, recordEnd, "q", expectedQCount),
        loadTraceArray(contents, recordBegin, recordEnd, "v", expectedVCount),
    };
}

[[nodiscard]] std::vector<double> loadTraceConfiguration(
    const char* const tracePath,
    const std::uint32_t step,
    const std::size_t expectedCount
) {
    return loadTraceState(tracePath, step, expectedCount, expectedCount - 1u).q;
}

struct SymmetricTwoByTwoSpectrum {
    double minimumEigenvalue = 0.0;
    double maximumEigenvalue = 0.0;
    double condition = 0.0;
};

[[nodiscard]] SymmetricTwoByTwoSpectrum spectrum(
    const double diagonal0,
    const double offDiagonal,
    const double diagonal1
) {
    const double trace = diagonal0 + diagonal1;
    const double radius = std::hypot(diagonal0 - diagonal1, 2.0 * offDiagonal);
    SymmetricTwoByTwoSpectrum result{
        .minimumEigenvalue = 0.5 * (trace - radius),
        .maximumEigenvalue = 0.5 * (trace + radius),
    };
    require(
        std::isfinite(result.minimumEigenvalue) &&
            std::isfinite(result.maximumEigenvalue) &&
            result.minimumEigenvalue > 0.0,
        "equality/limit Delassus matrix is not positive definite"
    );
    result.condition = result.maximumEigenvalue / result.minimumEigenvalue;
    require(std::isfinite(result.condition),
            "equality/limit Delassus condition is non-finite");
    return result;
}

int runTraceEqualityLimitRankAudit(
    const char* const rigidPath,
    const char* const equalityPath,
    const char* const tracePath,
    const char* const stepArgument,
    const char* const equalityArgument
) {
    constexpr double kBoundaryTolerance = 1.0e-5;
    constexpr double kResponseRegularization = 1.0e-7;
    const std::uint32_t step = parseUnsignedArgument(stepArgument, "trace step");
    const std::uint32_t equalityIndex =
        parseUnsignedArgument(equalityArgument, "equality index");
    const LoadedRigid rigid = loadRigid(rigidPath);
    const metalrobo::NumiHumanJointEqualityPayload equalities =
        loadJointEqualities(equalityPath, rigid.header);
    require(equalityIndex < equalities.records.size(),
            "equality index is outside NHEQ1");
    const std::vector<double> q = loadTraceConfiguration(
        tracePath, step, rigid.header.nq
    );
    const MRNumiHumanJointEqualityGPU& equality =
        equalities.records[equalityIndex];
    require(
        equality.indices.y < rigid.header.nv &&
            equality.indices.x < rigid.header.nq,
        "equality dependent indices are outside NHRIGID2"
    );
    const MRDofPropertiesGPU& limit = rigid.model.dofs.at(equality.indices.y);
    require(
        limit.qIndex != MR_INVALID_INDEX && limit.qIndex < rigid.header.nq &&
            (limit.flags & MR_DOF_FLAG_POSITION_LIMIT) != 0u,
        "equality dependent DoF has no source position limit"
    );

    metalrobo::NumiHumanJointEqualityEvaluation evaluation;
    require(
        metalrobo::evaluateNumiHumanJointEquality(equality, q, evaluation).succeeded(),
        "could not evaluate trace equality"
    );
    double traceFiniteDifference = 0.0;
    double traceDerivativeAbsoluteError = 0.0;
    double traceDerivativeRelativeError = 0.0;
    if (equality.indices.z != MR_INVALID_INDEX) {
        require(
            equality.indices.z < rigid.header.nq &&
                equality.indices.w < rigid.header.nv,
            "equality master indices are outside NHRIGID2"
        );
        const double coordinate = q[equality.indices.z];
        const double h = 1.0e-5 * std::max(1.0, std::abs(coordinate));
        std::vector<double> positive(q);
        std::vector<double> negative(q);
        positive[equality.indices.z] += h;
        negative[equality.indices.z] -= h;
        metalrobo::NumiHumanJointEqualityEvaluation plus;
        metalrobo::NumiHumanJointEqualityEvaluation minus;
        require(
            metalrobo::evaluateNumiHumanJointEquality(equality, positive, plus).succeeded() &&
                metalrobo::evaluateNumiHumanJointEquality(equality, negative, minus).succeeded(),
            "trace equality derivative finite-difference evaluation failed"
        );
        traceFiniteDifference =
            (plus.dependentTarget - minus.dependentTarget) / (2.0 * h);
        traceDerivativeAbsoluteError =
            std::abs(traceFiniteDifference - evaluation.derivative);
        traceDerivativeRelativeError = traceDerivativeAbsoluteError / std::max(
            {1.0, std::abs(traceFiniteDifference), std::abs(evaluation.derivative)}
        );
        require(
            traceDerivativeRelativeError <= kJointEqualityDerivativeRelativeTolerance,
            "trace equality derivative finite-difference mismatch"
        );
    } else {
        require(evaluation.derivative == 0.0,
                "constant trace equality has a nonzero derivative");
    }
    const double position = q[limit.qIndex];
    const double lowerDistance = position - static_cast<double>(limit.limits.x);
    const double upperDistance = static_cast<double>(limit.limits.y) - position;
    const bool lowerBoundary = std::abs(lowerDistance) <= std::abs(upperDistance);
    const double boundaryDistance = lowerBoundary ? lowerDistance : upperDistance;
    require(
        std::isfinite(lowerDistance) && std::isfinite(upperDistance) &&
            std::abs(boundaryDistance) <= kBoundaryTolerance,
        "trace equality dependent coordinate is not near a source limit"
    );
    const double limitSign = lowerBoundary ? 1.0 : -1.0;

    const std::size_t nv = rigid.header.nv;
    std::vector<double> rows(2u * nv, 0.0);
    rows[equality.indices.y] = 1.0;
    if (equality.indices.w != MR_INVALID_INDEX) {
        require(equality.indices.w < rigid.header.nv,
                "equality master velocity is outside NHRIGID2");
        rows[equality.indices.w] = -evaluation.derivative;
    }
    rows[nv + equality.indices.y] = limitSign;
    std::vector<double> responses(2u * nv, 0.0);
    metalrobo::ArticulatedDynamicsConfig dynamicsConfig{};
    dynamicsConfig.gravity = {
        rigid.model.world.gravityAndTimestep.x,
        rigid.model.world.gravityAndTimestep.y,
        rigid.model.world.gravityAndTimestep.z,
    };
    dynamicsConfig.timestep = rigid.model.world.gravityAndTimestep.w;
    const auto responseDiagnostics =
        metalrobo::computeArticulatedInverseMassResponses(
            rigid.model, 0u, q, rows, responses, dynamicsConfig
        );
    require(
        responseDiagnostics.succeeded(),
        "FP64 equality/limit inverse-mass response failed status=" +
            std::to_string(static_cast<std::uint32_t>(responseDiagnostics.status))
    );

    double delassus00 = 0.0;
    double delassus01 = 0.0;
    double delassus10 = 0.0;
    double delassus11 = 0.0;
    double rowNorm0Squared = 0.0;
    double rowNorm1Squared = 0.0;
    double rowDot = 0.0;
    for (std::size_t dof = 0u; dof < nv; ++dof) {
        const double equalityRow = rows[dof];
        const double limitRow = rows[nv + dof];
        delassus00 += equalityRow * responses[dof];
        delassus01 += equalityRow * responses[nv + dof];
        delassus10 += limitRow * responses[dof];
        delassus11 += limitRow * responses[nv + dof];
        rowNorm0Squared += equalityRow * equalityRow;
        rowNorm1Squared += limitRow * limitRow;
        rowDot += equalityRow * limitRow;
    }
    const double symmetricOffDiagonal = 0.5 * (delassus01 + delassus10);
    const SymmetricTwoByTwoSpectrum physical = spectrum(
        delassus00, symmetricOffDiagonal, delassus11
    );
    const SymmetricTwoByTwoSpectrum regularized = spectrum(
        delassus00 + kResponseRegularization, symmetricOffDiagonal,
        delassus11 + kResponseRegularization
    );
    const double rowCosine = rowDot / std::sqrt(rowNorm0Squared * rowNorm1Squared);
    const double delassusCorrelation = symmetricOffDiagonal /
        std::sqrt(delassus00 * delassus11);
    require(
        std::isfinite(rowCosine) && std::isfinite(delassusCorrelation),
        "equality/limit correlation is non-finite"
    );

    std::cout << std::setprecision(17)
              << "numi_human_trace_equality_limit_rank_audit="
              << "{\"schema\":\"numi.human.trace-equality-limit-rank-audit.v1\""
              << ",\"trace_step\":" << step
              << ",\"equality_index\":" << equalityIndex
              << ",\"dependent_q\":" << equality.indices.x
              << ",\"dependent_v\":" << equality.indices.y
              << ",\"master_q\":" << equality.indices.z
              << ",\"master_v\":" << equality.indices.w
              << ",\"dependent_position\":" << position
              << ",\"lower_distance\":" << lowerDistance
              << ",\"upper_distance\":" << upperDistance
              << ",\"limit_boundary\":\""
              << (lowerBoundary ? "lower" : "upper") << "\""
              << ",\"equality_target\":" << evaluation.dependentTarget
              << ",\"equality_position_error\":" << evaluation.positionError
              << ",\"equality_derivative\":" << evaluation.derivative
              << ",\"trace_derivative_finite_difference\":"
              << traceFiniteDifference
              << ",\"trace_derivative_absolute_error\":"
              << traceDerivativeAbsoluteError
              << ",\"trace_derivative_relative_error\":"
              << traceDerivativeRelativeError
              << ",\"row_cosine\":" << rowCosine
              << ",\"delassus_00\":" << delassus00
              << ",\"delassus_01\":" << delassus01
              << ",\"delassus_10\":" << delassus10
              << ",\"delassus_11\":" << delassus11
              << ",\"delassus_correlation\":" << delassusCorrelation
              << ",\"physical_min_eigenvalue\":" << physical.minimumEigenvalue
              << ",\"physical_max_eigenvalue\":" << physical.maximumEigenvalue
              << ",\"physical_condition\":" << physical.condition
              << ",\"regularization\":" << kResponseRegularization
              << ",\"regularized_min_eigenvalue\":"
              << regularized.minimumEigenvalue
              << ",\"regularized_max_eigenvalue\":"
              << regularized.maximumEigenvalue
              << ",\"regularized_condition\":" << regularized.condition
              << ",\"fp64_minimum_cholesky_pivot\":"
              << responseDiagnostics.minimumCholeskyPivot
              << ",\"source_identity_matched\":true"
              << ",\"full_active_set_qualified\":false}"
              << '\n';
    return 0;
}

// This is deliberately an operator snapshot, not a dynamics replacement. It
// retains every bilateral equality and every source limit that is already near
// an authored boundary in a retained production trace, then measures their
// FP64 inverse-mass coupling together. Contact, friction, force RHS, and time
// integration remain outside this audit and are reported as such.
enum class TraceEqualityLimitRowKind : std::uint32_t {
    equality,
    lowerLimit,
    upperLimit,
};

struct TraceEqualityLimitRow {
    TraceEqualityLimitRowKind kind = TraceEqualityLimitRowKind::equality;
    std::uint32_t sourceIndex = MR_INVALID_INDEX;
    std::uint32_t qIndex = MR_INVALID_INDEX;
    std::uint32_t vIndex = MR_INVALID_INDEX;
    double derivative = 0.0;
    double boundaryDistance = 0.0;
    bool outwardAtTrace = false;
};

[[nodiscard]] const char* traceEqualityLimitRowKindName(
    const TraceEqualityLimitRowKind kind
) {
    switch (kind) {
    case TraceEqualityLimitRowKind::equality: return "equality";
    case TraceEqualityLimitRowKind::lowerLimit: return "lower_limit";
    case TraceEqualityLimitRowKind::upperLimit: return "upper_limit";
    }
    return "unknown";
}

struct PivotedCholeskyAudit {
    std::vector<std::size_t> order;
    std::size_t rank = 0u;
    double cutoff = 0.0;
    double minimumAcceptedPivot = std::numeric_limits<double>::infinity();
    double firstRejectedPivot = 0.0;
    double maximumNegativeResidual = 0.0;
};

// A deterministic diagonal-pivoted Cholesky inspection of a symmetric PSD
// Gram matrix. It reports numerical rank at a declared relative pivot cutoff;
// it intentionally neither regularizes nor changes the underlying operator.
[[nodiscard]] PivotedCholeskyAudit pivotedCholeskyAudit(
    const std::span<const double> symmetricGram,
    const std::size_t dimension,
    const double relativeCutoff
) {
    require(
        dimension > 0u && symmetricGram.size() == dimension * dimension &&
            std::isfinite(relativeCutoff) && relativeCutoff > 0.0,
        "invalid symmetric Gram audit dimensions or cutoff"
    );
    PivotedCholeskyAudit result;
    result.order.resize(dimension);
    std::vector<double> residual(dimension, 0.0);
    std::vector<double> factor(dimension * dimension, 0.0);
    double maximumDiagonal = 0.0;
    for (std::size_t row = 0u; row < dimension; ++row) {
        result.order[row] = row;
        const double diagonal = symmetricGram[row * dimension + row];
        require(std::isfinite(diagonal), "Gram diagonal is non-finite");
        residual[row] = diagonal;
        maximumDiagonal = std::max(maximumDiagonal, diagonal);
    }
    require(maximumDiagonal > 0.0,
            "Gram audit has no positive diagonal response");
    result.cutoff = maximumDiagonal * relativeCutoff;
    for (std::size_t column = 0u; column < dimension; ++column) {
        std::size_t pivot = column;
        for (std::size_t candidate = column + 1u;
             candidate < dimension;
             ++candidate) {
            if (residual[candidate] > residual[pivot]) pivot = candidate;
        }
        if (pivot != column) {
            std::swap(result.order[pivot], result.order[column]);
            std::swap(residual[pivot], residual[column]);
            for (std::size_t previous = 0u; previous < column; ++previous) {
                std::swap(
                    factor[pivot * dimension + previous],
                    factor[column * dimension + previous]
                );
            }
        }
        const double pivotValue = residual[column];
        if (!(pivotValue > result.cutoff)) {
            result.firstRejectedPivot = pivotValue;
            break;
        }
        const double root = std::sqrt(pivotValue);
        require(std::isfinite(root) && root > 0.0,
                "Gram Cholesky pivot is non-finite");
        factor[column * dimension + column] = root;
        result.minimumAcceptedPivot = std::min(
            result.minimumAcceptedPivot, pivotValue
        );
        ++result.rank;
        for (std::size_t row = column + 1u; row < dimension; ++row) {
            double value = symmetricGram[
                result.order[row] * dimension + result.order[column]
            ];
            for (std::size_t previous = 0u; previous < column; ++previous) {
                value -= factor[row * dimension + previous] *
                    factor[column * dimension + previous];
            }
            const double entry = value / root;
            require(std::isfinite(entry), "Gram Cholesky entry is non-finite");
            factor[row * dimension + column] = entry;
            residual[row] -= entry * entry;
            if (residual[row] < 0.0) {
                result.maximumNegativeResidual = std::max(
                    result.maximumNegativeResidual, -residual[row]
                );
            }
        }
    }
    if (result.rank == dimension) result.firstRejectedPivot = 0.0;
    return result;
}

struct TraceEqualityLimitPairAudit {
    bool available = false;
    std::size_t equalityRow = 0u;
    std::size_t limitRow = 0u;
    double correlation = 0.0;
    double minimumEigenvalue = 0.0;
    double maximumEigenvalue = 0.0;
    double condition = 0.0;
};

[[nodiscard]] TraceEqualityLimitPairAudit traceEqualityLimitPairAudit(
    const std::span<const double> gram,
    const std::size_t dimension,
    const std::size_t equalityRow,
    const std::size_t limitRow
) {
    require(
        gram.size() == dimension * dimension && equalityRow < dimension &&
            limitRow < dimension,
        "invalid equality/limit pair audit indices"
    );
    const double diagonal0 = gram[equalityRow * dimension + equalityRow];
    const double diagonal1 = gram[limitRow * dimension + limitRow];
    const double offDiagonal = 0.5 * (
        gram[equalityRow * dimension + limitRow] +
        gram[limitRow * dimension + equalityRow]
    );
    require(
        std::isfinite(diagonal0) && std::isfinite(diagonal1) &&
            std::isfinite(offDiagonal) && diagonal0 > 0.0 && diagonal1 > 0.0,
        "equality/limit pair has invalid Delassus diagonal"
    );
    const double trace = diagonal0 + diagonal1;
    const double radius = std::hypot(diagonal0 - diagonal1, 2.0 * offDiagonal);
    TraceEqualityLimitPairAudit result;
    result.available = true;
    result.equalityRow = equalityRow;
    result.limitRow = limitRow;
    result.correlation = offDiagonal / std::sqrt(diagonal0 * diagonal1);
    result.minimumEigenvalue = 0.5 * (trace - radius);
    result.maximumEigenvalue = 0.5 * (trace + radius);
    result.condition = result.minimumEigenvalue > 0.0
        ? result.maximumEigenvalue / result.minimumEigenvalue
        : std::numeric_limits<double>::infinity();
    require(
        std::isfinite(result.correlation) && std::isfinite(result.minimumEigenvalue) &&
            std::isfinite(result.maximumEigenvalue),
        "equality/limit pair spectrum is non-finite"
    );
    return result;
}

void writeTraceEqualityLimitPairJson(
    std::ostream& output,
    const TraceEqualityLimitPairAudit& pair,
    const std::span<const TraceEqualityLimitRow> rows
) {
    if (!pair.available) {
        output << "null";
        return;
    }
    const TraceEqualityLimitRow& equality = rows[pair.equalityRow];
    const TraceEqualityLimitRow& limit = rows[pair.limitRow];
    output << "{\"equality_index\":" << equality.sourceIndex
           << ",\"equality_dependent_v\":" << equality.vIndex
           << ",\"limit_dof\":" << limit.sourceIndex
           << ",\"limit_q\":" << limit.qIndex
           << ",\"limit_boundary\":\""
           << traceEqualityLimitRowKindName(limit.kind) << "\""
           << ",\"limit_boundary_distance\":" << limit.boundaryDistance
           << ",\"limit_outward_at_trace\":"
           << (limit.outwardAtTrace ? "true" : "false")
           << ",\"correlation\":" << pair.correlation
           << ",\"minimum_eigenvalue\":" << pair.minimumEigenvalue
           << ",\"maximum_eigenvalue\":" << pair.maximumEigenvalue
           << ",\"condition\":";
    if (std::isfinite(pair.condition)) output << pair.condition;
    else output << "null";
    output << '}';
}

void writeTraceEqualityLimitRowJson(
    std::ostream& output,
    const TraceEqualityLimitRow& row
) {
    output << "{\"kind\":\"" << traceEqualityLimitRowKindName(row.kind)
           << "\",\"source_index\":" << row.sourceIndex
           << ",\"q_index\":" << row.qIndex
           << ",\"v_index\":" << row.vIndex;
    if (row.kind == TraceEqualityLimitRowKind::equality) {
        output << ",\"derivative\":" << row.derivative;
    } else {
        output << ",\"boundary_distance\":" << row.boundaryDistance
               << ",\"outward_at_trace\":"
               << (row.outwardAtTrace ? "true" : "false");
    }
    output << '}';
}

int runTraceEqualityLimitActiveSetAudit(
    const char* const rigidPath,
    const char* const equalityPath,
    const char* const tracePath,
    const char* const stepArgument
) {
    constexpr double kBoundaryTolerance = 1.0e-5;
    constexpr double kRelativePivotCutoff = 1.0e-10;
    const std::uint32_t step = parseUnsignedArgument(stepArgument, "trace step");
    const LoadedRigid rigid = loadRigid(rigidPath);
    const metalrobo::NumiHumanJointEqualityPayload equalities =
        loadJointEqualities(equalityPath, rigid.header);
    const TraceState trace = loadTraceState(
        tracePath, step, rigid.header.nq, rigid.header.nv
    );
    const EqualityDerivativeAudit derivativeAudit = auditJointEqualityDerivatives(
        equalities.records, trace.q, "trace_q"
    );
    const std::size_t nv = rigid.header.nv;
    std::vector<TraceEqualityLimitRow> metadata;
    metadata.reserve(equalities.records.size() + nv);
    std::vector<double> rows;
    rows.reserve((equalities.records.size() + nv) * nv);
    const auto appendRow = [&](const TraceEqualityLimitRow& row) {
        metadata.push_back(row);
        rows.resize(rows.size() + nv, 0.0);
    };

    for (std::size_t equalityIndex = 0u;
         equalityIndex < equalities.records.size();
         ++equalityIndex) {
        const MRNumiHumanJointEqualityGPU& equality = equalities.records[equalityIndex];
        metalrobo::NumiHumanJointEqualityEvaluation evaluation;
        require(
            metalrobo::evaluateNumiHumanJointEquality(
                equality, trace.q, evaluation
            ).succeeded(),
            "could not evaluate full equality set at trace state"
        );
        require(
            equality.indices.x < rigid.header.nq && equality.indices.y < nv &&
                (equality.indices.w == MR_INVALID_INDEX || equality.indices.w < nv),
            "trace equality indices are outside NHRIGID2"
        );
        appendRow({
            .kind = TraceEqualityLimitRowKind::equality,
            .sourceIndex = static_cast<std::uint32_t>(equalityIndex),
            .qIndex = equality.indices.x,
            .vIndex = equality.indices.y,
            .derivative = evaluation.derivative,
        });
        const std::size_t offset = (metadata.size() - 1u) * nv;
        rows[offset + equality.indices.y] = 1.0;
        if (equality.indices.w != MR_INVALID_INDEX) {
            rows[offset + equality.indices.w] = -evaluation.derivative;
        }
    }

    std::size_t outwardNearBoundaryLimitCount = 0u;
    for (std::size_t dof = 0u; dof < nv; ++dof) {
        const MRDofPropertiesGPU& properties = rigid.model.dofs[dof];
        if ((properties.flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u) continue;
        require(
            properties.qIndex != MR_INVALID_INDEX && properties.qIndex < rigid.header.nq &&
                std::isfinite(properties.limits.x) && std::isfinite(properties.limits.y) &&
                properties.limits.x < properties.limits.y,
            "trace source limit is malformed"
        );
        const double position = trace.q[properties.qIndex];
        const double lowerDistance = position - static_cast<double>(properties.limits.x);
        const double upperDistance = static_cast<double>(properties.limits.y) - position;
        const bool lowerNear = std::abs(lowerDistance) <= kBoundaryTolerance;
        const bool upperNear = std::abs(upperDistance) <= kBoundaryTolerance;
        if (!lowerNear && !upperNear) continue;
        require(!(lowerNear && upperNear),
                "trace source limit is simultaneously near both boundaries");
        const bool lower = lowerNear;
        const bool outward = lower ? trace.v[dof] < 0.0 : trace.v[dof] > 0.0;
        outwardNearBoundaryLimitCount += outward ? 1u : 0u;
        appendRow({
            .kind = lower ? TraceEqualityLimitRowKind::lowerLimit
                          : TraceEqualityLimitRowKind::upperLimit,
            .sourceIndex = static_cast<std::uint32_t>(dof),
            .qIndex = properties.qIndex,
            .vIndex = static_cast<std::uint32_t>(dof),
            .boundaryDistance = lower ? lowerDistance : upperDistance,
            .outwardAtTrace = outward,
        });
        const std::size_t offset = (metadata.size() - 1u) * nv;
        rows[offset + dof] = lower ? 1.0 : -1.0;
    }
    require(metadata.size() > equalities.records.size(),
            "trace active-set audit found no near-boundary source limit");

    std::vector<double> responses(rows.size(), 0.0);
    metalrobo::ArticulatedDynamicsConfig dynamicsConfig{};
    dynamicsConfig.gravity = {
        rigid.model.world.gravityAndTimestep.x,
        rigid.model.world.gravityAndTimestep.y,
        rigid.model.world.gravityAndTimestep.z,
    };
    dynamicsConfig.timestep = rigid.model.world.gravityAndTimestep.w;
    const auto responseDiagnostics = metalrobo::computeArticulatedInverseMassResponses(
        rigid.model, 0u, trace.q, rows, responses, dynamicsConfig
    );
    require(
        responseDiagnostics.succeeded(),
        "FP64 full equality/limit inverse-mass response failed status=" +
            std::to_string(static_cast<std::uint32_t>(responseDiagnostics.status))
    );

    const std::size_t rowCount = metadata.size();
    std::vector<double> gram(rowCount * rowCount, 0.0);
    double maximumAsymmetry = 0.0;
    double maximumAbsoluteOffDiagonalCorrelation = 0.0;
    for (std::size_t row = 0u; row < rowCount; ++row) {
        for (std::size_t column = 0u; column < rowCount; ++column) {
            double value = 0.0;
            for (std::size_t dof = 0u; dof < nv; ++dof) {
                value += rows[row * nv + dof] * responses[column * nv + dof];
            }
            require(std::isfinite(value), "full equality/limit Delassus entry is non-finite");
            gram[row * rowCount + column] = value;
        }
    }
    for (std::size_t row = 0u; row < rowCount; ++row) {
        require(gram[row * rowCount + row] > 0.0,
                "full equality/limit Delassus diagonal is not positive");
        for (std::size_t column = row + 1u; column < rowCount; ++column) {
            const double symmetric = 0.5 * (
                gram[row * rowCount + column] +
                gram[column * rowCount + row]
            );
            maximumAsymmetry = std::max(
                maximumAsymmetry,
                std::abs(gram[row * rowCount + column] -
                         gram[column * rowCount + row])
            );
            gram[row * rowCount + column] = symmetric;
            gram[column * rowCount + row] = symmetric;
        }
    }
    std::vector<double> normalizedGram(rowCount * rowCount, 0.0);
    for (std::size_t row = 0u; row < rowCount; ++row) {
        for (std::size_t column = 0u; column < rowCount; ++column) {
            const double normalized = gram[row * rowCount + column] /
                std::sqrt(
                    gram[row * rowCount + row] *
                    gram[column * rowCount + column]
                );
            require(std::isfinite(normalized),
                    "normalized full equality/limit Delassus entry is non-finite");
            normalizedGram[row * rowCount + column] = normalized;
            if (row != column) {
                maximumAbsoluteOffDiagonalCorrelation = std::max(
                    maximumAbsoluteOffDiagonalCorrelation, std::abs(normalized)
                );
            }
        }
    }
    const PivotedCholeskyAudit factor = pivotedCholeskyAudit(
        normalizedGram, rowCount, kRelativePivotCutoff
    );
    std::size_t retainedEqualityRowCount = 0u;
    std::size_t retainedLimitRowCount = 0u;
    for (std::size_t index = 0u; index < factor.rank; ++index) {
        const TraceEqualityLimitRow& row = metadata[factor.order[index]];
        if (row.kind == TraceEqualityLimitRowKind::equality) {
            ++retainedEqualityRowCount;
        } else {
            ++retainedLimitRowCount;
        }
    }

    TraceEqualityLimitPairAudit mostCoupled;
    TraceEqualityLimitPairAudit implicated;
    for (std::size_t equalityRow = 0u; equalityRow < rowCount; ++equalityRow) {
        if (metadata[equalityRow].kind != TraceEqualityLimitRowKind::equality) continue;
        for (std::size_t limitRow = 0u; limitRow < rowCount; ++limitRow) {
            if (metadata[limitRow].kind == TraceEqualityLimitRowKind::equality) continue;
            TraceEqualityLimitPairAudit candidate = traceEqualityLimitPairAudit(
                gram, rowCount, equalityRow, limitRow
            );
            if (!mostCoupled.available ||
                std::abs(candidate.correlation) > std::abs(mostCoupled.correlation)) {
                mostCoupled = candidate;
            }
            if (metadata[equalityRow].sourceIndex == 43u &&
                metadata[limitRow].sourceIndex == 113u) {
                implicated = candidate;
            }
        }
    }
    require(mostCoupled.available,
            "full equality/limit audit found no equality-limit pair");

    std::cout << std::setprecision(17)
              << "numi_human_trace_equality_limit_active_set_audit="
              << "{\"schema\":\"numi.human.trace-equality-limit-active-set-audit.v1\""
              << ",\"trace_step\":" << step
              << ",\"equality_row_count\":" << equalities.records.size()
              << ",\"near_boundary_limit_row_count\":"
              << (rowCount - equalities.records.size())
              << ",\"outward_near_boundary_limit_row_count\":"
              << outwardNearBoundaryLimitCount
              << ",\"operator_row_count\":" << rowCount
              << ",\"velocity_dof_count\":" << nv
              << ",\"equality_derivative_checked_count\":"
              << derivativeAudit.checkedCount
              << ",\"equality_derivative_maximum_absolute_error\":"
              << derivativeAudit.maximumAbsoluteError
              << ",\"equality_derivative_maximum_absolute_error_index\":"
              << derivativeAudit.maximumAbsoluteErrorIndex
              << ",\"equality_derivative_maximum_relative_error\":"
              << derivativeAudit.maximumRelativeError
              << ",\"equality_derivative_maximum_relative_error_index\":"
              << derivativeAudit.maximumRelativeErrorIndex
              << ",\"gram_maximum_asymmetry\":" << maximumAsymmetry
              << ",\"normalized_gram_maximum_absolute_off_diagonal\":"
              << maximumAbsoluteOffDiagonalCorrelation
              << ",\"normalized_pivot_relative_cutoff\":"
              << kRelativePivotCutoff
              << ",\"normalized_pivot_rank\":" << factor.rank
              << ",\"normalized_pivot_minimum_accepted\":"
              << factor.minimumAcceptedPivot
              << ",\"normalized_pivot_first_rejected\":"
              << factor.firstRejectedPivot
              << ",\"normalized_pivot_maximum_negative_residual\":"
              << factor.maximumNegativeResidual
              << ",\"normalized_pivot_retained_equality_row_count\":"
              << retainedEqualityRowCount
              << ",\"normalized_pivot_retained_near_boundary_limit_row_count\":"
              << retainedLimitRowCount
              << ",\"normalized_pivot_first_rejected_row\":";
    if (factor.rank < rowCount) {
        writeTraceEqualityLimitRowJson(
            std::cout, metadata[factor.order[factor.rank]]
        );
    } else {
        std::cout << "null";
    }
    std::cout
              << ",\"fp64_minimum_cholesky_pivot\":"
              << responseDiagnostics.minimumCholeskyPivot
              << ",\"most_coupled_equality_limit_pair\":";
    writeTraceEqualityLimitPairJson(std::cout, mostCoupled, metadata);
    std::cout << ",\"implicated_equality_43_limit_113_pair\":";
    writeTraceEqualityLimitPairJson(std::cout, implicated, metadata);
    std::cout << ",\"source_identity_matched\":true"
              << ",\"contact_rows_included\":false"
              << ",\"friction_rows_included\":false"
              << ",\"muscle_and_passive_force_rhs_included\":false"
              << ",\"time_integrated\":false"
              << ",\"full_active_set_qualified\":false"
              << ",\"scope\":\"read_only_FP64_equality_plus_near_boundary_limit_operator_snapshot\""
              << "}\n";
    return 0;
}

void applyNumiHumanTendonPayload(
    const char* path, const LoadedRigid& rigid, LoadedMuscles& muscles
) {
    const std::vector<std::byte> bytes = readBytes(path);
    metalrobo::NumiHumanTendonPayload payload;
    const auto decode = metalrobo::decodeNumiHumanTendonPayload(
        bytes, rigid.header.sourceSha256, {}, payload
    );
    require(
        decode.succeeded(),
        std::string("NHTENDON decode failed: ") +
            metalrobo::numiHumanTendonStatusName(decode.status) +
            " index=" + std::to_string(decode.failingIndex)
    );
    require(
        payload.bodyCount == rigid.header.engineBodyCount,
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
    if (resolved.migratedEnvelopeBindingCount > 0u) {
        std::vector<double> referenceQ(
            rigid.model.defaultQ.begin(), rigid.model.defaultQ.end()
        );
        metalrobo::NumiHumanTendonReferenceCalibration calibration;
        const auto calibrationDiagnostics =
            metalrobo::calibrateNumiHumanMigratedTendonReference(
                rigid.model, 0u, referenceQ, muscles.wraps,
                muscles.sourceSites, muscles.sourceMuscles,
                resolved.sites, resolved.muscles, muscles.architectures,
                payload, calibration
            );
        require(
            calibrationDiagnostics.succeeded(),
            std::string("NHTENDON reference calibration failed: ") +
                metalrobo::numiHumanTendonStatusName(calibrationDiagnostics.status) +
                " index=" + std::to_string(calibrationDiagnostics.failingIndex) +
                " path_delta_m=" + (
                    calibrationDiagnostics.failingIndex < calibration.pathLengthDeltas.size()
                    ? std::to_string(calibration.pathLengthDeltas[
                        calibrationDiagnostics.failingIndex
                    ])
                    : std::string("unavailable")
                ) +
                " maximum_absolute_path_delta_m=" +
                    std::to_string(calibration.maximumAbsolutePathLengthDelta) +
                " maximum_architecture_scale_change=" +
                    std::to_string(calibration.maximumArchitectureScaleChange) +
                " optimal_fiber_length_m=" + (
                    calibrationDiagnostics.failingIndex < muscles.architectures.size()
                    ? std::to_string(muscles.architectures[
                        calibrationDiagnostics.failingIndex
                    ].optimalFiberLength)
                    : std::string("unavailable")
                ) +
                " tendon_slack_length_m=" + (
                    calibrationDiagnostics.failingIndex < muscles.architectures.size()
                    ? std::to_string(muscles.architectures[
                        calibrationDiagnostics.failingIndex
                    ].tendonSlackLength)
                    : std::string("unavailable")
                )
        );
        for (std::size_t index = 0u; index < muscles.oracleLength.size(); ++index) {
            muscles.oracleLength[index] += calibration.pathLengthDeltas[index];
        }
        muscles.maximumTendonReferencePathDelta =
            calibration.maximumAbsolutePathLengthDelta;
        muscles.maximumTendonArchitectureScaleChange =
            calibration.maximumArchitectureScaleChange;
    }
    muscles.sites = std::move(resolved.sites);
    muscles.muscles = std::move(resolved.muscles);
    muscles.tendonPointBindings = resolved.pointBindingCount;
    muscles.tendonTriangleBindings = resolved.triangleBindingCount;
    muscles.tendonEnvelopeBindings = resolved.envelopeBindingCount;
    muscles.tendonMigratedEnvelopeBindings = resolved.migratedEnvelopeBindingCount;
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
        gpuMuscle.lengthRangeAndAcceleration.x = static_cast<float>(
            muscles.muscles[muscleIndex].lengthRange[0]
        );
        gpuMuscle.lengthRangeAndAcceleration.y = static_cast<float>(
            muscles.muscles[muscleIndex].lengthRange[1]
        );
        gpuMuscle.compliantArchitecture0.x = static_cast<float>(
            muscles.architectures[muscleIndex].optimalFiberLength
        );
        gpuMuscle.compliantArchitecture0.y = static_cast<float>(
            muscles.architectures[muscleIndex].tendonSlackLength
        );
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
    if (muscles.tendonPayload.payloadAbi == 2u || muscles.tendonPayload.payloadAbi == 3u) {
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
            if (binding.mode != metalrobo::NumiHumanTendonAttachmentMode::registeredBoneDistributedEnvelope &&
                binding.mode != metalrobo::NumiHumanTendonAttachmentMode::registeredBoneMigratedDistributedEnvelope) {
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
    // exact first-order-hold Metal sidecar update is performing an actual
    // temporal advance, not merely echoing the input state. Force parity above
    // remains tied to the source-default 0.5/0.5 state.
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
    // Run the same state through the explicitly versioned ABI 2 path. This
    // keeps the legacy update available while the Human production path uses
    // ABI 3 for timestep refinement.
    auto legacyConfig = activationConfig;
    legacyConfig.mujocoActivationExactFirstOrderHold = false;
    metalrobo::MetalArticulatedOperatorResult legacyResult;
    const auto legacyDiagnostics = metalrobo::runMetalArticulatedOperator(
        model, activationInput, legacyResult, legacyConfig
    );
    require(
        legacyDiagnostics.succeeded() &&
            legacyResult.mujocoActivationStates.size() == activationStates.size(),
        std::string("MyoSim legacy activation-step operator failed: ") +
            metalrobo::metalArticulatedOperatorHostStatusName(
                legacyDiagnostics.status
            ) + " " + legacyDiagnostics.message
    );
    for (std::size_t index = 0u; index < activationStates.size(); ++index) {
        const float initial = activationStates[index].excitationAndActivation.y;
        const float derivative = activationResult.mujocoResults[index]
            .pathForceAndActivationDerivative.w;
        const float expected = std::clamp(
            initial + kActivationTimestepSeconds * derivative, 0.0f, 1.0f
        );
        require(
            std::abs(double(
                legacyResult.mujocoActivationStates[index].excitationAndActivation.y
            ) - expected) < 2.0e-6,
            "MyoSim ABI 2 legacy activation update changed"
        );
    }
    metrics.activationTimestepSeconds = kActivationTimestepSeconds;
    for (std::size_t index = 0u;
         index < activationStates.size();
         ++index) {
        const float initialActivation =
            activationStates[index].excitationAndActivation.y;
        const float derivative = activationResult.mujocoResults[index]
            .pathForceAndActivationDerivative.w;
        const float control = std::clamp(
            activationStates[index].excitationAndActivation.x, 0.0f, 1.0f
        );
        const float excess = control - initialActivation;
        const float expectedActivation = std::clamp(
            (excess == 0.0f || derivative == 0.0f)
                ? initialActivation
                : control - excess * std::exp(
                    -kActivationTimestepSeconds / (excess / derivative)
                ),
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
    // NHMYO2 continuation begins with a positive, already accepted fibre
    // length. The old activation kernel treated z/w as legacy reserved words
    // and silently froze activation after the first step.
    auto continuedStates = activationResult.mujocoActivationStates;
    for (auto& state : continuedStates) {
        state.excitationAndActivation.x = 1.0f - state.excitationAndActivation.x;
    }
    activationInput.mujoco.states = continuedStates;
    metalrobo::MetalArticulatedOperatorResult continued, continuedReplay;
    const auto continuedDiagnostics = activationContext.run(model, activationInput, continued);
    const auto replayDiagnostics = activationContext.run(model, activationInput, continuedReplay);
    require(continuedDiagnostics.succeeded() && replayDiagnostics.succeeded() &&
                continued.mujocoActivationStates.size() == continuedStates.size() &&
                continued.mujocoResults.size() == continuedStates.size() &&
                continuedReplay.mujocoActivationStates.size() == continuedStates.size(),
            "MyoSim initialized-fibre continuation failed");
    std::size_t initializedFibres = 0u, changedActivations = 0u;
    double continuationError = 0.0;
    for (std::size_t index = 0; index < continuedStates.size(); ++index) {
        const auto& before = continuedStates[index].excitationAndActivation;
        const auto& after = continued.mujocoActivationStates[index].excitationAndActivation;
        const float control = std::clamp(before.x, 0.0f, 1.0f);
        const float excess = control - before.y;
        const float derivative =
            continued.mujocoResults[index].pathForceAndActivationDerivative.w;
        const float expected = std::clamp(
            (excess == 0.0f || derivative == 0.0f)
                ? before.y
                : control - excess * std::exp(
                    -kActivationTimestepSeconds / (excess / derivative)
                ),
            0.0f,
            1.0f
        );
        continuationError = std::max(continuationError, std::abs(double(after.y - expected)));
        initializedFibres += before.z > 0.0f ? 1u : 0u;
        changedActivations += after.y != before.y ? 1u : 0u;
        require(after.x == before.x && std::isfinite(after.z) && after.z >= 0.0f &&
                    std::isfinite(after.w), "MyoSim continuation corrupted the fibre state");
    }
    require(continuationError < 1.0e-6 && changedActivations > 0u,
            "MyoSim initialized-fibre activation did not advance with its source derivative");
    require(std::memcmp(continued.mujocoActivationStates.data(),
                continuedReplay.mujocoActivationStates.data(),
                continuedStates.size() * sizeof(MRMujocoMuscleStateGPU)) == 0,
            "MyoSim initialized-fibre continuation replay differs");
    std::cout << "muscle_activation_continuation initialized_fibres=" << initializedFibres
              << " changed_activations=" << changedActivations
              << " maximum_error=" << continuationError << " replay=byte_exact\n";
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
        applyNumiHumanTendonPayload(tendonPath, rigid, muscles);
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
                if (binding.mode == metalrobo::NumiHumanTendonAttachmentMode::registeredBoneDistributedEnvelope ||
                    binding.mode == metalrobo::NumiHumanTendonAttachmentMode::registeredBoneMigratedDistributedEnvelope) {
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
        if (muscles.tendonTriangleBindings > 0u || muscles.tendonMigratedEnvelopeBindings > 0u) {
            // Metal parity compares against the resolved CPU route. The source
            // oracle delta above remains reported as explicit endpoint migration.
            muscles.oracleLength[index] = result.path.length;
            muscles.oracleForce[index] = result.actuatorForce;
        }
    }
    if (muscles.tendonTriangleBindings == 0u && muscles.tendonMigratedEnvelopeBindings == 0u) {
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
        if (muscles.tendonTriangleBindings == 0u && muscles.tendonMigratedEnvelopeBindings == 0u) {
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
                             << " tendon_migrated_envelope_bindings="
                             << muscles.tendonMigratedEnvelopeBindings
                             << " tendon_max_endpoint_migration_m=" << muscles.maximumEndpointMigration
                             << " tendon_max_reference_path_delta_m="
                             << muscles.maximumTendonReferencePathDelta
                             << " tendon_max_architecture_scale_change="
                             << muscles.maximumTendonArchitectureScaleChange
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
               << " equilibrium_global_activation_polish_iterations="
               << equilibrium.diagnostics.globalActivationPolishIterations
               << " equilibrium_accepted_global_activation_polish_steps="
               << equilibrium.diagnostics.acceptedGlobalActivationPolishSteps
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

// Same admitted FP32 pose, independent native FP64 route evaluation and
// Metal path evaluation. This does not advance a body or grant reset authority.
int runPreparedPathReference(const char* rigidPath, const char* musclePath, const char* initialPath, std::uint64_t timestepOverride = 0u, bool pairedGeometry = false) {
    const LoadedRigid rigid = loadRigid(rigidPath);
    const LoadedMuscles muscles = loadMuscles(musclePath, rigid.header);
    const auto& model = rigid.model;
    const auto& articulation = model.articulations.at(0u);
    metalrobo::NumiHumanInitialState initial;
    std::string error;
    require(metalrobo::decodeNumiHumanInitialState(readBytes(initialPath), articulation.nq,
        articulation.nv, static_cast<std::uint32_t>(muscles.gpuMuscles.size()), rigid.header.sourceSha256,
        initial, error), "prepared path input: " + error);
    require(timestepOverride <= 1'000'000u, "fibre-reference timestep exceeds one second");
    const auto exactNanoseconds = timestepOverride != 0u ? timestepOverride * 1000u
        : metalrobo::numiHumanInitialStateTimestepNanoseconds(initial);
    const float timestep = static_cast<float>(exactNanoseconds * 1e-9);
    require(std::isfinite(timestep) && timestep > 0, "invalid fibre-reference timestep");
    std::vector<MRArticulatedPointImpulseGPU> points;
    for (std::uint32_t body = 0; body < articulation.bodyCount; ++body) {
        for (unsigned probe = 0; probe < 4u; ++probe) {
            MRArticulatedPointImpulseGPU point{};
            point.bodyIndex = articulation.firstBody + body;
            if (probe == 1u) point.localPoint.x = 1;
            if (probe == 2u) point.localPoint.y = 1;
            if (probe == 3u) point.localPoint.z = 1;
            points.push_back(point);
        }
    }
    const auto initialQ = initial.q, initialV = initial.v;
    const auto initialMuscles = initial.muscles;
    std::vector<MRCompensatedRootTranslationGPU> rootTranslations;
    if (pairedGeometry) rootTranslations.push_back(initial.rootTranslation.value_or(
        mrCompensatedTranslationFromProjection({initial.q[0], initial.q[1], initial.q[2], 0.0f})));
    const auto initialRoots = rootTranslations;
    const metalrobo::MetalArticulatedOperatorInput input{
        .articulationIndex = 0u, .environmentCount = 1u, .pointCount = points.size(),
        .q = initial.q, .rootTranslations = rootTranslations, .v = initial.v, .points = points,
        .mujoco = {.muscles = muscles.gpuMuscles, .states = initial.muscles,
            .sites = muscles.gpuSites, .wraps = muscles.gpuWraps,
            .routeNodes = muscles.gpuRoutes, .bodyJacobianPointOffset = 0u}};
    const metalrobo::MetalArticulatedOperatorConfig config{
        .pointJacobiansOnly = true, .mujocoActivationTimestepSeconds = timestep};
    std::size_t admissionControls = 0u;
    if (pairedGeometry) {
        const auto denied = [&](const metalrobo::MetalArticulatedOperatorInput& rejectedInput,
                                const metalrobo::MetalArticulatedOperatorConfig& rejectedConfig) {
            metalrobo::MetalArticulatedOperatorResult sentinel;
            sentinel.generalizedImpulse = {123.0f};
            const auto result = metalrobo::runMetalArticulatedOperator(model, rejectedInput, sentinel, rejectedConfig);
            require(!result.succeeded() && !result.dispatched && !result.published &&
                sentinel.generalizedImpulse == std::vector<float>{123.0f},
                "read-only paired admission accepted incompatible input or changed output");
            ++admissionControls;
        };
        auto invalidConfig = config; invalidConfig.pointJacobiansOnly = false;
        denied(input, invalidConfig);
        MRMillardMuscleGPU dummyMillard{};
        auto invalidInput = input;
        invalidInput.millard.muscles = std::span<const MRMillardMuscleGPU>(&dummyMillard, 1u);
        denied(invalidInput, config);
        auto invalidRoots = rootTranslations;
        invalidRoots[0].reference.w = 1.0f;
        invalidInput = input; invalidInput.rootTranslations = invalidRoots;
        denied(invalidInput, config);
        auto invalidQ = initial.q;
        invalidQ[0] = std::nextafter(invalidQ[0], invalidQ[0] == 0.0f
            ? 1.0f : 0.0f);
        invalidInput = input; invalidInput.q = invalidQ;
        denied(invalidInput, config);
        invalidRoots = rootTranslations; invalidRoots.push_back(rootTranslations[0]);
        invalidInput = input; invalidInput.rootTranslations = invalidRoots;
        denied(invalidInput, config);
        invalidInput = input;
        invalidInput.residentContinuation.previousTransactionFingerprint = 1u;
        invalidInput.residentContinuation.previousPhysicsGeneration = 1u;
        denied(invalidInput, config);
    }
    metalrobo::MetalArticulatedOperatorResult gpu;
    metalrobo::MetalArticulatedOperatorDiagnostics diagnostics;
    if (pairedGeometry) {
        metalrobo::MetalArticulatedOperatorContext context(config);
        diagnostics = context.run(model, input, gpu);
        require(diagnostics.succeeded() && diagnostics.published,
            "prepared paired context evaluation failed: " + diagnostics.message);
        metalrobo::MetalArticulatedOperatorResult replay;
        const auto replayDiagnostics = context.run(model, input, replay);
        const auto identical = []<typename T>(const std::vector<T>& left, const std::vector<T>& right) {
            return left.size() == right.size() && (left.empty() ||
                std::memcmp(left.data(), right.data(), left.size() * sizeof(T)) == 0);
        };
        const auto pairedEqual = [&](const metalrobo::MetalArticulatedOperatorResult& other) {
            return identical(gpu.rootTranslations, other.rootTranslations) &&
                identical(gpu.bodyPoses, other.bodyPoses) && identical(gpu.bodyPositionLow, other.bodyPositionLow) &&
                identical(gpu.pointWorld, other.pointWorld) && identical(gpu.pointPositionLow, other.pointPositionLow) &&
                identical(gpu.pointJacobians, other.pointJacobians) &&
                identical(gpu.mujocoResults, other.mujocoResults) &&
                identical(gpu.mujocoActivationStates, other.mujocoActivationStates) &&
                identical(gpu.mujocoMuscleGeneralizedForces, other.mujocoMuscleGeneralizedForces) &&
                identical(gpu.mujocoGeneralizedForces, other.mujocoGeneralizedForces);
        };
        const auto noStand = [](const metalrobo::MetalArticulatedOperatorResult& value,
                                const metalrobo::MetalArticulatedOperatorDiagnostics& info) {
            return value.standQ.empty() && value.standRootTranslations.empty() &&
                value.standV.empty() && value.standStatuses.empty() && info.layout.standStatusElements == 0u;
        };
        require(replayDiagnostics.succeeded() && replayDiagnostics.published &&
            identical(gpu.rootTranslations, initialRoots) && pairedEqual(replay),
            "prepared paired resident reuse changed source evaluation bytes");
        // Reuse the same context across both geometry modes. Retained allocations
        // must become logically absent in legacy mode, then be reinitialized from
        // the authored input rather than a previous private constitutive result.
        auto legacyInput = input; legacyInput.rootTranslations = {};
        metalrobo::MetalArticulatedOperatorResult legacy;
        const auto legacyDiagnostics = context.run(model, legacyInput, legacy);
        require(legacyDiagnostics.succeeded() && legacyDiagnostics.published &&
            legacy.rootTranslations.empty() && legacy.bodyPositionLow.empty() &&
            legacy.pointPositionLow.empty() && noStand(legacy, legacyDiagnostics),
            "paired to legacy transition retained paired or physical state");
        metalrobo::MetalArticulatedOperatorResult pairedAgain;
        const auto pairedAgainDiagnostics = context.run(model, input, pairedAgain);
        require(pairedAgainDiagnostics.succeeded() && pairedAgainDiagnostics.published && pairedEqual(pairedAgain),
            "legacy to paired transition changed authored source evaluation bytes");
        require(identical(initial.q, initialQ) && identical(initial.v, initialV) &&
            identical(initial.muscles, initialMuscles) && identical(rootTranslations, initialRoots),
            "read-only paired evaluation changed input source state");
        require(gpu.bodyPositionLow.size() == gpu.bodyPoses.size() &&
            gpu.pointPositionLow.size() == gpu.pointWorld.size() && noStand(gpu, diagnostics) &&
            noStand(replay, replayDiagnostics) && noStand(pairedAgain, pairedAgainDiagnostics),
            "read-only paired source evaluation acquired physical Stand state");
        std::cout << "prepared_compensated_admission=pass negative_controls=" << admissionControls
                  << " repeated_input=bitwise mode_transition=paired_legacy_paired_bitwise"
                     " source_state=unchanged physical_steps=0 accepted_root_publications=0\n";
    } else {
        diagnostics = metalrobo::runMetalArticulatedOperator(model, input, gpu, config);
    }
    require(diagnostics.succeeded() && diagnostics.dispatched && diagnostics.published,
        "prepared Metal path evaluation failed: " + diagnostics.message);
    require(gpu.mujocoResults.size() == muscles.muscles.size(), "prepared path result extent");
    std::vector<double> q(initial.q.begin(), initial.q.end());
    const std::vector<double> v(initial.v.begin(), initial.v.end());
    if (pairedGeometry) {
        const auto& root = rootTranslations[0];
        for (std::size_t axis = 0u; axis < 3u; ++axis)
            q[axis] = static_cast<double>(static_cast<long double>((&root.reference.x)[axis]) +
                (&root.displacement.x)[axis] + (&root.correction.x)[axis]);
    }
    std::vector<metalrobo::ArticulatedBodyKinematics> referenceBodies(articulation.bodyCount);
    require(metalrobo::computeArticulatedBodyKinematics(model, 0u, q, v, referenceBodies).succeeded(),
        "prepared body reference");
    for (std::size_t body = 0; body < referenceBodies.size(); ++body) {
        const auto& expected = referenceBodies[body];
        const auto& actual = gpu.bodyPoses[body];
        std::cout << std::setprecision(17) << "prepared_body={\"index\":" << body
                  << ",\"native_position\":[" << expected.centerOfMassPosition[0] << ','
                  << expected.centerOfMassPosition[1] << ',' << expected.centerOfMassPosition[2]
                  << "],\"metal_position\":[" << actual.position.x << ',' << actual.position.y << ',' << actual.position.z
                  << "],\"native_orientation\":[" << expected.orientation[0] << ',' << expected.orientation[1] << ','
                  << expected.orientation[2] << ',' << expected.orientation[3]
                  << "],\"metal_position_low\":["
                  << (pairedGeometry ? gpu.bodyPositionLow[body].x : 0.0f) << ','
                  << (pairedGeometry ? gpu.bodyPositionLow[body].y : 0.0f) << ','
                  << (pairedGeometry ? gpu.bodyPositionLow[body].z : 0.0f)
                  << "],\"metal_orientation\":[" << actual.orientation.x << ',' << actual.orientation.y << ','
                  << actual.orientation.z << ',' << actual.orientation.w << "]}\n";
    }
    const auto siteFromPose = [&] (std::uint32_t siteIndex, bool useGPU) {
        const auto& site = muscles.sites[siteIndex];
        const auto localBody = site.bodyIndex - articulation.firstBody;
        const auto& p = gpu.bodyPoses[localBody];
        const auto& r = referenceBodies[localBody];
        const std::array<double, 4> rotation = useGPU
            ? std::array<double, 4>{p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w}
            : r.orientation;
        auto value = quaternionRotate(rotation, site.localPoint);
        for (std::size_t axis = 0; axis < 3; ++axis)
            value[axis] += useGPU ? static_cast<double>((&p.position.x)[axis]) +
                (pairedGeometry ? (&gpu.bodyPositionLow[localBody].x)[axis] : 0.0) : r.centerOfMassPosition[axis];
        return value;
    };
    for (std::size_t i = 0; i < muscles.muscles.size(); ++i) {
        const auto& route = muscles.muscles[i].route;
        double nativeLength = 0, gpuPoseLength = 0;
        bool unwrapped = true;
        for (std::size_t node = 0; node + 1 < route.size(); ++node) {
            if (route[node].type != metalrobo::MujocoRouteNodeType::site ||
                route[node+1].type != metalrobo::MujocoRouteNodeType::site) { unwrapped = false; break; }
            const auto a = siteFromPose(route[node].targetIndex, false);
            const auto b = siteFromPose(route[node+1].targetIndex, false);
            const auto c = siteFromPose(route[node].targetIndex, true);
            const auto d = siteFromPose(route[node+1].targetIndex, true);
            nativeLength += std::hypot(b[0]-a[0], b[1]-a[1], b[2]-a[2]);
            gpuPoseLength += std::hypot(d[0]-c[0], d[1]-c[1], d[2]-c[2]);
        }
        if (unwrapped) std::cout << std::setprecision(17) << "prepared_pose_path={\"index\":" << i
            << ",\"native_pose_fp64_m\":" << nativeLength << ",\"metal_pose_fp64_m\":" << gpuPoseLength
            << ",\"metal_pose_fp32_m\":" << gpu.mujocoResults[i].pathForceAndActivationDerivative.x << "}\n";
    }
    double maximumError = 0, maximumForceError = 0, maximumSamePathForceError = 0;
    double maximumNormalizedForceError = 0, maximumResidual = 0, maximumPublicationErrorUlps = 0;
    std::size_t maximumIndex = 0;
    for (std::size_t i = 0; i < muscles.muscles.size(); ++i) {
        metalrobo::MujocoMuscleResult cpu;
        require(metalrobo::evaluateMujocoMuscle(model, 0u, q, v, muscles.sites, muscles.wraps,
            muscles.muscles[i], {}, cpu).succeeded(), "prepared native FP64 path evaluation");
        const double length = gpu.mujocoResults[i].pathForceAndActivationDerivative.x;
        const double delta = std::abs(length - cpu.path.length);
        require(std::isfinite(delta) && length > 0, "prepared path finite positive length");
        if (delta > maximumError) { maximumError = delta; maximumIndex = i; }
        const auto accepted = initial.muscles[i].excitationAndActivation;
        const metalrobo::MujocoCompliantMuscleState state{
            accepted.x, accepted.y, accepted.z, accepted.w};
        metalrobo::MujocoCompliantMuscleResult sourceForce, samePathForce;
        const double dt = timestep;
        require(metalrobo::evaluateMujocoCompliantMuscle(cpu.path.length, cpu.path.velocity, dt,
            muscles.muscles[i], muscles.architectures[i], state, sourceForce).succeeded(),
            "prepared FP64 fibre reference failed");
        require(metalrobo::evaluateMujocoCompliantMuscle(length,
            gpu.mujocoResults[i].pathForceAndActivationDerivative.y, dt,
            muscles.muscles[i], muscles.architectures[i], state, samePathForce).succeeded(),
            "prepared same-path FP64 fibre reference failed");
        const double gpuForce = gpu.mujocoResults[i].pathForceAndActivationDerivative.z;
        const double forceError = std::abs(gpuForce - sourceForce.actuatorForce);
        const double samePathError = std::abs(gpuForce - samePathForce.actuatorForce);
        require(std::isfinite(forceError) && std::isfinite(samePathError), "nonfinite fibre force comparison");
        maximumForceError = std::max(maximumForceError, forceError);
        maximumSamePathForceError = std::max(maximumSamePathForceError, samePathError);
        const auto& definition = muscles.muscles[i];
        const double forceScale = definition.gainParameters[2] < 0
            ? definition.gainParameters[3] / definition.accelerationScale : definition.gainParameters[2];
        maximumNormalizedForceError = std::max(maximumNormalizedForceError, samePathError / forceScale);
        const auto fiber = gpu.mujocoResults[i].fiberStateTendonForceResidual;
        maximumResidual = std::max(maximumResidual, std::abs(static_cast<double>(fiber.w)));
        const double publicationError = std::abs(static_cast<double>(fiber.x) - accepted.z - dt * fiber.y);
        const double ulp = std::nextafter(fiber.x, std::numeric_limits<float>::infinity()) - fiber.x;
        require(std::isfinite(publicationError) && ulp > 0, "invalid fibre publication");
        maximumPublicationErrorUlps = std::max(maximumPublicationErrorUlps, publicationError / ulp);
        std::cout << std::setprecision(17) << "prepared_path={\"index\":" << i
                  << ",\"native_fp64_m\":" << cpu.path.length << ",\"metal_fp32_m\":" << length
                  << ",\"absolute_error_m\":" << delta
                  << ",\"native_fp64_force_n\":" << sourceForce.actuatorForce
                  << ",\"same_path_fp64_force_n\":" << samePathForce.actuatorForce
                  << ",\"metal_fp32_force_n\":" << gpuForce
                  << ",\"metal_fiber_residual\":" << gpu.mujocoResults[i].fiberStateTendonForceResidual.w
                  << "}\n";
    }
    // A numerical route agreement budget, not experimental accuracy.
    const bool pathPassed = maximumError <= 2.0e-6;
    const bool fiberPassed = maximumNormalizedForceError <= 1.0e-5 && maximumResidual <= 1.0e-5 &&
        maximumPublicationErrorUlps <= 0.501;
    const bool passed = pathPassed && fiberPassed;
    std::cout << "prepared_path_summary={\"maximum_error_m\":" << maximumError
              << ",\"maximum_index\":" << maximumIndex
              << ",\"maximum_force_error_n\":" << maximumForceError
              << ",\"maximum_same_path_force_error_n\":" << maximumSamePathForceError
              << ",\"timestep_seconds\":" << timestep
              << ",\"maximum_normalized_same_path_force_error\":" << maximumNormalizedForceError
              << ",\"maximum_fiber_residual\":" << maximumResidual
              << ",\"maximum_fiber_publication_error_ulps\":" << maximumPublicationErrorUlps
              << ",\"path_passed\":" << (pathPassed ? "true" : "false")
              << ",\"fiber_passed\":" << (fiberPassed ? "true" : "false")
              << ",\"normalized_force_tolerance\":1e-5,\"publication_tolerance_ulps\":0.501"
              << ",\"geometry_mode\":\"" << (pairedGeometry ? "paired_read_only" : "legacy_scalar") << "\""
              << ",\"physical_steps\":0,\"accepted_root_publications\":0"
              << ",\"tolerance_m\":2e-6,\"passed\":"
              << (passed ? "true" : "false") << "}\n";
    return passed ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 4 && std::string(argv[2]).ends_with(".nheq") &&
            std::string(argv[3]) == "--equality-derivative-audit") {
            return runJointEqualityDerivativeAudit(argv[1], argv[2]);
        }
        if (argc == 7 &&
            std::string(argv[3]) == "--trace-equality-limit-rank-audit") {
            return runTraceEqualityLimitRankAudit(
                argv[1], argv[2], argv[4], argv[5], argv[6]
            );
        }
        if (argc == 6 &&
            std::string(argv[3]) == "--trace-equality-limit-active-set-audit") {
            return runTraceEqualityLimitActiveSetAudit(
                argv[1], argv[2], argv[4], argv[5]
            );
        }
        if ((argc == 5 || argc == 7) && (std::string(argv[3]) == "--prepared-paths" ||
            std::string(argv[3]) == "--prepared-compensated-paths")) {
            std::uint64_t timestepOverride = 0u;
            if (argc == 7) {
                require(std::string(argv[5]) == "--timestep-us", "expected --timestep-us");
                const char* end = argv[6] + std::strlen(argv[6]);
                const auto parsed = std::from_chars(argv[6], end, timestepOverride);
                require(parsed.ec == std::errc{} && parsed.ptr == end && timestepOverride > 0u,
                    "timestep must be a positive integer microsecond count");
            }
            return runPreparedPathReference(argv[1], argv[2], argv[4], timestepOverride,
                std::string(argv[3]) == "--prepared-compensated-paths");
        }
        if (argc < 3 || argc > 7) {
            std::cerr << "usage: " << argv[0] << " <myosim-fullbody-core-reference.nhrigid> "
                      << "<myosim-fullbody-muscle-reference.nhmyo> "
                      << "[numi-human-tendon-endpoints.nhtendon] [--metal] "
                         "[myosim-fullbody-joint-equalities.nheq] "
                         "[--equilibrium] | <rigid> <NHEQ1> --equality-derivative-audit | "
                         "<rigid> <NHEQ1> --trace-equality-limit-rank-audit "
                         "<trace> <step> <equality-index> | "
                         "<rigid> <NHEQ1> --trace-equality-limit-active-set-audit "
                         "<trace> <step> | "
                         "--prepared-paths/--prepared-compensated-paths "
                         "<prepared.nhinit> [--timestep-us N]\n";
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
