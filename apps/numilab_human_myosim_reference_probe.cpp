#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/MujocoMuscleReference.hpp"

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
constexpr std::array<char, 8u> kMuscleMagic{'N', 'H', 'M', 'Y', 'O', '1', '\0', '\0'};
constexpr std::uint32_t kRigidAbi = 1u;
constexpr std::uint32_t kMuscleAbi = 1u;

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
#pragma pack(pop)

static_assert(sizeof(RigidHeader) == 80u);
static_assert(sizeof(SourcePoseRecord) == 28u);
static_assert(sizeof(MuscleHeader) == 76u);
static_assert(sizeof(SiteRecord) == 16u);
static_assert(sizeof(WrapRecord) == 64u);
static_assert(sizeof(RouteRecord) == 16u);
static_assert(sizeof(MuscleRecord) == 164u);
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
    std::vector<MRMujocoMuscleSiteGPU> gpuSites;
    std::vector<MRMujocoMuscleWrapGPU> gpuWraps;
    std::vector<MRMujocoMuscleRouteNodeGPU> gpuRoutes;
    std::vector<MRMujocoMuscleGPU> gpuMuscles;
    std::vector<MRMujocoMuscleStateGPU> gpuStates;
    std::vector<double> oracleLength;
    std::vector<double> oracleForce;
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
    require(result.header.magic == kMuscleMagic, "muscle payload magic is not NHMYO1");
    require(result.header.payloadAbi == kMuscleAbi && result.header.reserved0 == 0u && result.header.reserved1 == 0u,
            "unsupported or non-canonical MyoSim muscle payload ABI");
    require(result.header.engineBodyCount == rigid.engineBodyCount && result.header.sourceSha256 == rigid.sourceSha256,
            "MyoSim muscle payload does not match rigid payload source");
    const std::vector<SiteRecord> sourceSites = readVector<SiteRecord>(input, result.header.siteCount, "MyoSim site records");
    const std::vector<WrapRecord> sourceWraps = readVector<WrapRecord>(input, result.header.wrapCount, "MyoSim wrap records");
    const std::vector<RouteRecord> routes = readVector<RouteRecord>(input, result.header.routeNodeCount, "MyoSim route records");
    const std::vector<MuscleRecord> sourceMuscles = readVector<MuscleRecord>(input, result.header.muscleCount, "MyoSim muscle records");
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
    result.gpuMuscles.reserve(sourceMuscles.size());
    result.gpuStates.reserve(sourceMuscles.size());
    result.oracleLength.reserve(sourceMuscles.size());
    result.oracleForce.reserve(sourceMuscles.size());
    for (const MuscleRecord& source : sourceMuscles) {
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
        result.gpuMuscles.push_back(gpuMuscle);
        MRMujocoMuscleStateGPU gpuState{};
        gpuState.excitationAndActivation = {0.5f, 0.5f, 0.0f, 0.0f};
        result.gpuStates.push_back(gpuState);
    }
    return result;
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

struct MetalArticulatedMetrics {
    std::string deviceName;
    double maximumBodyPositionError = 0.0;
    double maximumBodyOrientationComponentError = 0.0;
    double maximumPointPositionError = 0.0;
    double maximumPointJacobianError = 0.0;
    double maximumMuscleLengthError = 0.0;
    double maximumMuscleForceError = 0.0;
    double maximumMuscleGeneralizedForceError = 0.0;
    double maximumSummedGeneralizedForceError = 0.0;
    std::uint32_t appliedMuscleWraps = 0u;
};

MetalArticulatedMetrics verifyMetalArticulatedReference(
    const metalrobo::EngineModel& model,
    const LoadedMuscles& muscles
) {
    const MRArticulationGPU& articulation = model.articulations.at(0u);
    std::vector<float> q = model.defaultQ;
    std::vector<double> qReference(q.begin(), q.end());
    std::vector<double> zeroVelocity(articulation.nv, 0.0);

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
        model, 0u, qReference, zeroVelocity, cpuBodies
    );
    require(cpuDiagnostics.succeeded(), "MyoSim CPU body reference failed before Metal parity");
    std::vector<metalrobo::ArticulatedPointKinematics> cpuPointKinematics(cpuPoints.size());
    std::vector<double> cpuJacobians(cpuPoints.size() * 3u * articulation.nv);
    cpuDiagnostics = metalrobo::computeArticulatedPointJacobians(
        model, 0u, qReference, zeroVelocity, cpuPoints, cpuPointKinematics, cpuJacobians
    );
    require(cpuDiagnostics.succeeded(), "MyoSim CPU point/Jacobian reference failed before Metal parity");
    const metalrobo::MetalArticulatedOperatorInput input{
        .articulationIndex = 0u,
        .environmentCount = 1u,
        .pointCount = gpuPoints.size(),
        .q = q,
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
            kinematicsResult.mujocoMuscleGeneralizedForces.size() ==
                muscles.muscles.size() * articulation.nv &&
            kinematicsResult.mujocoGeneralizedForces.size() == articulation.nv,
        "MyoSim Metal kinematics/Jacobian result layout is invalid"
    );

    MetalArticulatedMetrics metrics;
    metrics.deviceName = kinematicsDiagnostics.deviceName;
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
        metrics.maximumMuscleForceError = std::max(
            metrics.maximumMuscleForceError,
            std::abs(static_cast<double>(
                gpu.pathForceAndActivationDerivative.z
            ) - muscles.oracleForce[index])
        );
        metrics.appliedMuscleWraps += gpu.appliedWrapCount;
    }
    std::vector<double> expectedGeneralizedForce(articulation.nv, 0.0);
    for (std::size_t muscleIndex = 0u;
         muscleIndex < muscles.muscles.size();
         ++muscleIndex) {
        std::vector<double> expectedMuscleForce(articulation.nv, 0.0);
        metalrobo::MujocoMuscleResult expectedResult;
        const auto forceDiagnostics = metalrobo::projectMujocoMuscleForce(
            model, 0u, qReference, zeroVelocity, muscles.sites, muscles.wraps,
            muscles.muscles[muscleIndex], {.excitation = 0.5, .activation = 0.5},
            expectedMuscleForce, &expectedResult
        );
        require(forceDiagnostics.succeeded(), "MyoSim CPU force reference failed before Metal parity");
        for (std::size_t dof = 0u; dof < articulation.nv; ++dof) {
            const std::size_t gpuIndex = muscleIndex * articulation.nv + dof;
            metrics.maximumMuscleGeneralizedForceError = std::max(
                metrics.maximumMuscleGeneralizedForceError,
                std::abs(static_cast<double>(
                    kinematicsResult.mujocoMuscleGeneralizedForces[gpuIndex]
                ) - expectedMuscleForce[dof])
            );
            expectedGeneralizedForce[dof] += expectedMuscleForce[dof];
        }
    }
    for (std::size_t dof = 0u; dof < articulation.nv; ++dof) {
        metrics.maximumSummedGeneralizedForceError = std::max(
            metrics.maximumSummedGeneralizedForceError,
            std::abs(static_cast<double>(
                kinematicsResult.mujocoGeneralizedForces[dof]
            ) - expectedGeneralizedForce[dof])
        );
    }
    require(
        metrics.maximumBodyPositionError < 2.0e-4 &&
            metrics.maximumBodyOrientationComponentError < 2.0e-4 &&
            metrics.maximumPointPositionError < 2.0e-4 &&
            metrics.maximumPointJacobianError < 5.0e-4 &&
            metrics.maximumMuscleLengthError < 2.0e-4 &&
            metrics.maximumMuscleForceError < 5.0e-2 &&
            metrics.maximumMuscleGeneralizedForceError < 5.0e-2 &&
            metrics.maximumSummedGeneralizedForceError < 2.0e-1 &&
            metrics.appliedMuscleWraps == 90u,
        "MyoSim Metal kinematics/Jacobian/muscle-route parity exceeded FP32 tolerance: "
            "body=" + std::to_string(metrics.maximumBodyPositionError) +
            " orientation=" + std::to_string(metrics.maximumBodyOrientationComponentError) +
            " point=" + std::to_string(metrics.maximumPointPositionError) +
            " jacobian=" + std::to_string(metrics.maximumPointJacobianError) +
            " muscle_length=" + std::to_string(metrics.maximumMuscleLengthError) +
            " muscle_force=" + std::to_string(metrics.maximumMuscleForceError) +
            " muscle_generalized_force=" + std::to_string(metrics.maximumMuscleGeneralizedForceError) +
            " summed_generalized_force=" + std::to_string(metrics.maximumSummedGeneralizedForceError) +
            " wraps=" + std::to_string(metrics.appliedMuscleWraps)
    );

    return metrics;
}

int run(const char* rigidPath, const char* musclePath, const bool runMetal) {
    const LoadedRigid rigid = loadRigid(rigidPath);
    const LoadedMuscles muscles = loadMuscles(musclePath, rigid.header);
    const auto& model = rigid.model;
    std::vector<double> q(model.defaultQ.begin(), model.defaultQ.end());
    std::vector<double> v(model.defaultV.begin(), model.defaultV.end());
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
    }
    require(maxMuscleLengthError <= 2.0e-5, "MyoSim native muscle paths do not match source default lengths");
    require(maxMuscleForceError <= 1.0e-2, "MyoSim native muscle forces do not match source default forces");
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
               << " metal_max_muscle_force_error_n="
               << metal.maximumMuscleForceError
               << " metal_max_muscle_generalized_force_error="
               << metal.maximumMuscleGeneralizedForceError
               << " metal_max_summed_generalized_force_error="
               << metal.maximumSummedGeneralizedForceError
               << " metal_applied_wraps=" << metal.appliedMuscleWraps;
    }
    output << "\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3 && argc != 4) {
            std::cerr << "usage: " << argv[0] << " <myosim-fullbody-core-reference.nhrigid> "
                      << "<myosim-fullbody-muscle-reference.nhmyo> [--metal]\n";
            return 2;
        }
        const bool runMetal = argc == 4;
        if (runMetal && std::string(argv[3]) != "--metal") {
            std::cerr << "usage: " << argv[0] << " <myosim-fullbody-core-reference.nhrigid> "
                      << "<myosim-fullbody-muscle-reference.nhmyo> [--metal]\n";
            return 2;
        }
        return run(argv[1], argv[2], runMetal);
    } catch (const std::exception& error) {
        std::cerr << "myosim_core_reference FAIL: " << error.what() << "\n";
        return 1;
    }
}
