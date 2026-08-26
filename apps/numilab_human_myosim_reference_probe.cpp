#include "metalrobo/ArticulatedDynamics.hpp"
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
    for (const SiteRecord& source : sourceSites) {
        require(source.bodyIndex < rigid.engineBodyCount, "MyoSim site body index is out of bounds");
        result.sites.push_back({source.bodyIndex, {source.x, source.y, source.z}});
    }
    result.wraps.reserve(sourceWraps.size());
    for (const WrapRecord& source : sourceWraps) {
        require(source.bodyIndex < rigid.engineBodyCount, "MyoSim wrap body index is out of bounds");
        result.wraps.push_back({
            source.bodyIndex, routeType(source.type), {source.centerX, source.centerY, source.centerZ},
            {source.rotation[0], source.rotation[1], source.rotation[2], source.rotation[3], source.rotation[4],
             source.rotation[5], source.rotation[6], source.rotation[7], source.rotation[8]}, source.radius,
        });
    }
    result.muscles.reserve(sourceMuscles.size());
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

int run(const char* rigidPath, const char* musclePath) {
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
    std::cout << std::setprecision(12)
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
              << " max_inverse_forward_error=" << maxDynamicsError
              << " mass_min_pivot=" << massDiagnostics.minimumCholeskyPivot
              << " mass_condition=" << massDiagnostics.estimatedMassMatrixCondition
              << "\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "usage: " << argv[0] << " <myosim-fullbody-core-reference.nhrigid> "
                      << "<myosim-fullbody-muscle-reference.nhmyo>\n";
            return 2;
        }
        return run(argv[1], argv[2]);
    } catch (const std::exception& error) {
        std::cerr << "myosim_core_reference FAIL: " << error.what() << "\n";
        return 1;
    }
}
