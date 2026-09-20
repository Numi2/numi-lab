#include "metalrobo/NumiHumanTensionNetwork.hpp"
#include "metalrobo/ArticulatedDynamics.hpp"

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
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using metalrobo::NumiHumanTensionNetworkElement;
using metalrobo::NumiHumanTensionNetworkLoad;
using metalrobo::NumiHumanTensionNetworkNode;
using metalrobo::NumiHumanTensionNetworkResult;

constexpr std::array<char, 8u> kRigidMagic{
    'N', 'H', 'R', 'I', 'G', 'I', 'D', '2'};
constexpr std::array<char, 8u> kHoodMagic{
    'N', 'H', 'H', 'O', 'O', 'D', '2', '\0'};

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

struct HoodHeader {
    std::array<char, 8u> magic{};
    std::uint32_t payloadAbi = 0u;
    std::uint32_t rayCount = 0u;
    std::uint32_t nodeCount = 0u;
    std::uint32_t elementCount = 0u;
    std::uint32_t inputCount = 0u;
    std::uint32_t rayRecordBytes = 0u;
    std::uint32_t nodeRecordBytes = 0u;
    std::uint32_t elementRecordBytes = 0u;
    std::uint32_t inputRecordBytes = 0u;
    std::array<std::uint8_t, 32u> sourceSha256{};
};

struct HoodRayRecord {
    std::uint32_t nodeOffset = 0u;
    std::uint32_t nodeCount = 0u;
    std::uint32_t elementOffset = 0u;
    std::uint32_t elementCount = 0u;
    std::uint32_t inputOffset = 0u;
    std::uint32_t inputCount = 0u;
    std::uint32_t side = 0u;
    std::uint32_t digit = 0u;
};

struct HoodNodeRecord {
    std::uint32_t sourceSite = 0u;
    std::uint32_t sourceBody = 0u;
    std::uint32_t coreBody = 0u;
    std::uint32_t flags = 0u;
    std::uint32_t role = 0u;
    float localX = 0.0f;
    float localY = 0.0f;
    float localZ = 0.0f;
};

struct HoodElementRecord {
    std::uint32_t nodeA = 0u;
    std::uint32_t nodeB = 0u;
    std::uint32_t bundle = 0u;
    std::uint32_t provenance = 0u;
    float restScale = 0.0f;
    float youngModulus = 0.0f;
    float area = 0.0f;
};

struct HoodInputRecord {
    std::uint32_t node = 0u;
    std::uint32_t sourceMuscle = 0u;
    std::uint32_t proximalSourceSite = 0u;
    std::uint32_t proximalCoreBody = 0u;
    std::uint32_t flags = 0u;
    float proximalLocalX = 0.0f;
    float proximalLocalY = 0.0f;
    float proximalLocalZ = 0.0f;
    float sourceOracleForce = 0.0f;
};
#pragma pack(pop)

static_assert(sizeof(RigidHeader) == 80u);
static_assert(sizeof(SourcePoseRecord) == 28u);
static_assert(sizeof(HoodHeader) == 76u);
static_assert(sizeof(HoodRayRecord) == 32u);
static_assert(sizeof(HoodNodeRecord) == 32u);
static_assert(sizeof(HoodElementRecord) == 28u);
static_assert(sizeof(HoodInputRecord) == 36u);

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
std::vector<T> readVector(
    std::istream& input, const std::size_t count, const char* description
) {
    std::vector<T> result(count);
    if (count != 0u) {
        input.read(reinterpret_cast<char*>(result.data()),
                   static_cast<std::streamsize>(count * sizeof(T)));
        require(input.good(), std::string("truncated ") + description);
    }
    return result;
}

struct RigidPoseTable {
    RigidHeader header{};
    metalrobo::EngineModel model;
    std::vector<double> q;
    std::vector<double> v;
    std::vector<SourcePoseRecord> poseByCoreBody;
};

RigidPoseTable loadRigidPoseTable(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "cannot open NHRIGID2 payload");
    RigidPoseTable result;
    readObject(input, result.header, "NHRIGID2 header");
    require(result.header.magic == kRigidMagic &&
                result.header.payloadAbi == 1u,
            "unsupported NHRIGID2 payload");
    require(result.header.sourceBodyCount > 0u &&
                result.header.engineBodyCount >= result.header.sourceBodyCount &&
                result.header.nq == result.header.nv + 1u,
            "invalid NHRIGID2 dimensions");
    result.model.name = "numilab_human_extensor_hood_source";
    readObject(input, result.model.world, "NHRIGID2 world");
    MRArticulationGPU articulation{};
    readObject(input, articulation, "NHRIGID2 articulation");
    result.model.articulations.push_back(articulation);
    result.model.bodies = readVector<MRBodyPropertiesGPU>(
        input, result.header.engineBodyCount, "NHRIGID2 bodies");
    result.model.joints = readVector<MRJointDescriptorGPU>(
        input, result.header.jointCount, "NHRIGID2 joints");
    result.model.dofs = readVector<MRDofPropertiesGPU>(
        input, result.header.nv, "NHRIGID2 dofs");
    const auto defaultQ = readVector<float>(
        input, result.header.nq, "NHRIGID2 default q");
    const auto defaultV = readVector<float>(
        input, result.header.nv, "NHRIGID2 default v");
    result.model.defaultQ = defaultQ;
    result.model.defaultV = defaultV;
    result.q.assign(defaultQ.begin(), defaultQ.end());
    result.v.assign(defaultV.begin(), defaultV.end());
    const auto sourceToCore = readVector<std::uint32_t>(
        input, result.header.sourceBodyCount, "NHRIGID2 source body map");
    const auto sourcePoses = readVector<SourcePoseRecord>(
        input, result.header.sourceBodyCount, "NHRIGID2 source poses");
    require(input.peek() == std::char_traits<char>::eof(),
            "NHRIGID2 payload has trailing bytes");
    result.poseByCoreBody.resize(result.header.engineBodyCount);
    for (std::size_t index = 0u; index < sourceToCore.size(); ++index) {
        require(sourceToCore[index] < result.poseByCoreBody.size(),
                "NHRIGID2 source body map is out of bounds");
        result.poseByCoreBody[sourceToCore[index]] = sourcePoses[index];
    }
    std::string reason;
    require(result.model.valid(&reason),
            "NHRIGID2 EngineModel is invalid: " + reason);
    return result;
}

struct LoadedHood {
    HoodHeader header{};
    std::vector<HoodRayRecord> rays;
    std::vector<HoodNodeRecord> nodes;
    std::vector<HoodElementRecord> elements;
    std::vector<HoodInputRecord> inputs;
};

LoadedHood loadHood(
    const std::filesystem::path& path, const RigidHeader& rigid
) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "cannot open NHHOOD2 payload");
    LoadedHood result;
    readObject(input, result.header, "NHHOOD2 header");
    require(result.header.magic == kHoodMagic &&
                result.header.payloadAbi == 2u,
            "unsupported NHHOOD2 payload");
    require(result.header.rayCount == 8u &&
                result.header.rayRecordBytes == sizeof(HoodRayRecord) &&
                result.header.nodeRecordBytes == sizeof(HoodNodeRecord) &&
                result.header.elementRecordBytes == sizeof(HoodElementRecord) &&
                result.header.inputRecordBytes == sizeof(HoodInputRecord) &&
                result.header.sourceSha256 == rigid.sourceSha256,
            "NHHOOD2 shape/source does not match NHRIGID2");
    result.rays = readVector<HoodRayRecord>(
        input, result.header.rayCount, "NHHOOD2 rays");
    result.nodes = readVector<HoodNodeRecord>(
        input, result.header.nodeCount, "NHHOOD2 nodes");
    result.elements = readVector<HoodElementRecord>(
        input, result.header.elementCount, "NHHOOD2 elements");
    result.inputs = readVector<HoodInputRecord>(
        input, result.header.inputCount, "NHHOOD2 inputs");
    require(input.peek() == std::char_traits<char>::eof(),
            "NHHOOD2 payload has trailing bytes");
    for (std::size_t index = 0u; index < result.rays.size(); ++index) {
        const auto& ray = result.rays[index];
        const std::uint32_t expectedSide = index / 4u;
        const std::uint32_t expectedDigit = 2u + index % 4u;
        require(ray.side == expectedSide && ray.digit == expectedDigit &&
                    ray.nodeCount == (ray.digit == 5u ? 12u : 10u) &&
                    ray.elementCount == (ray.digit == 5u ? 14u : 12u) &&
                    ray.inputCount == (ray.digit == 5u ? 5u : 4u) &&
                    ray.nodeOffset + ray.nodeCount <= result.nodes.size() &&
                    ray.elementOffset + ray.elementCount <= result.elements.size() &&
                    ray.inputOffset + ray.inputCount <= result.inputs.size(),
                "NHHOOD2 ray identity/range is invalid");
        if (index != 0u) {
            const auto& previous = result.rays[index - 1u];
            require(ray.nodeOffset == previous.nodeOffset + previous.nodeCount &&
                        ray.elementOffset == previous.elementOffset + previous.elementCount &&
                        ray.inputOffset == previous.inputOffset + previous.inputCount,
                    "NHHOOD2 ray ranges are not contiguous");
        }
    }
    return result;
}

std::array<double, 3u> rotate(
    const SourcePoseRecord& pose, const std::array<double, 3u>& value
) {
    const std::array<double, 3u> q{pose.quaternionX,
                                  pose.quaternionY,
                                  pose.quaternionZ};
    const double qw = pose.quaternionW;
    const std::array<double, 3u> cross{
        q[1] * value[2] - q[2] * value[1],
        q[2] * value[0] - q[0] * value[2],
        q[0] * value[1] - q[1] * value[0],
    };
    const std::array<double, 3u> second{
        q[1] * cross[2] - q[2] * cross[1],
        q[2] * cross[0] - q[0] * cross[2],
        q[0] * cross[1] - q[1] * cross[0],
    };
    return {
        value[0] + 2.0 * (qw * cross[0] + second[0]),
        value[1] + 2.0 * (qw * cross[1] + second[1]),
        value[2] + 2.0 * (qw * cross[2] + second[2]),
    };
}

std::array<double, 3u> worldPoint(
    const std::vector<SourcePoseRecord>& poses, const std::uint32_t body,
    const std::array<double, 3u>& local
) {
    require(body < poses.size(), "NHHOOD2 Core body is out of bounds");
    const auto rotated = rotate(poses[body], local);
    return {rotated[0] + poses[body].positionX,
            rotated[1] + poses[body].positionY,
            rotated[2] + poses[body].positionZ};
}

std::array<double, 3u> localPoint(
    const std::vector<SourcePoseRecord>& poses, const std::uint32_t body,
    const std::array<double, 3u>& world
) {
    require(body < poses.size(), "NHHOOD2 Core body is out of bounds");
    SourcePoseRecord inverse = poses[body];
    inverse.quaternionX = -inverse.quaternionX;
    inverse.quaternionY = -inverse.quaternionY;
    inverse.quaternionZ = -inverse.quaternionZ;
    return rotate(inverse, {
        world[0] - poses[body].positionX,
        world[1] - poses[body].positionY,
        world[2] - poses[body].positionZ,
    });
}

double norm(const std::array<double, 3u>& value) {
    return std::hypot(value[0], value[1], value[2]);
}

double distance(
    const std::array<double, 3u>& a, const std::array<double, 3u>& b
) {
    return norm({a[0] - b[0], a[1] - b[1], a[2] - b[2]});
}

struct Fixture {
    std::vector<NumiHumanTensionNetworkNode> nodes;
    std::vector<NumiHumanTensionNetworkElement> elements;
    std::vector<NumiHumanTensionNetworkLoad> loads;
    double maximumSourceOracleForce = 0.0;
    double minimumSourceElementLength = std::numeric_limits<double>::infinity();
    double maximumSourceElementLength = 0.0;
};

Fixture makeFixture(
    const LoadedHood& hood, const HoodRayRecord& ray,
    const std::vector<SourcePoseRecord>& poses,
    const bool useSourceOracleForce
) {
    Fixture result;
    result.nodes.reserve(ray.nodeCount);
    for (std::uint32_t index = 0u; index < ray.nodeCount; ++index) {
        const auto& source = hood.nodes[ray.nodeOffset + index];
        require((source.flags & 2u) != 0u && source.role == index,
                "NHHOOD2 node lacks exact initializer or canonical role order");
        result.nodes.push_back({
            worldPoint(poses, source.coreBody,
                       {source.localX, source.localY, source.localZ}),
            (source.flags & 1u) != 0u,
        });
    }
    require(result.nodes.size() >= 3u && result.nodes[0].fixed &&
                result.nodes[1].fixed && result.nodes[2].fixed &&
                hood.nodes[ray.nodeOffset].coreBody !=
                    hood.nodes[ray.nodeOffset + 1u].coreBody &&
                hood.nodes[ray.nodeOffset + 1u].coreBody !=
                    hood.nodes[ray.nodeOffset + 2u].coreBody &&
                hood.nodes[ray.nodeOffset].coreBody !=
                    hood.nodes[ray.nodeOffset + 2u].coreBody,
            "NHHOOD2 ray lacks distinct middle/distal/metacarpal bone anchors");
    result.elements.reserve(ray.elementCount);
    for (std::uint32_t index = 0u; index < ray.elementCount; ++index) {
        const auto& source = hood.elements[ray.elementOffset + index];
        require(source.nodeA >= ray.nodeOffset &&
                    source.nodeA < ray.nodeOffset + ray.nodeCount &&
                    source.nodeB >= ray.nodeOffset &&
                    source.nodeB < ray.nodeOffset + ray.nodeCount &&
                    source.nodeA != source.nodeB && source.provenance == 1u &&
                    source.restScale > 0.0f && source.restScale <= 1.0f &&
                    source.youngModulus > 0.0f && source.area > 0.0f,
                "NHHOOD2 element is invalid");
        const std::uint32_t a = source.nodeA - ray.nodeOffset;
        const std::uint32_t b = source.nodeB - ray.nodeOffset;
        const double sourceLength = distance(
            result.nodes[a].position, result.nodes[b].position);
        require(sourceLength >= 1.0e-4 && sourceLength <= 0.1,
                "NHHOOD2 source element length is anatomically implausible");
        result.minimumSourceElementLength = std::min(
            result.minimumSourceElementLength, sourceLength);
        result.maximumSourceElementLength = std::max(
            result.maximumSourceElementLength, sourceLength);
        result.elements.push_back({
            a, b, source.restScale * sourceLength,
            source.youngModulus, source.area,
        });
    }
    constexpr double literatureInputForce = 2.9;
    result.loads.reserve(ray.inputCount);
    for (std::uint32_t index = 0u; index < ray.inputCount; ++index) {
        const auto& source = hood.inputs[ray.inputOffset + index];
        require(source.node >= ray.nodeOffset &&
                    source.node < ray.nodeOffset + ray.nodeCount &&
                    (source.flags & 0xFFu) == 1u &&
                    (source.flags >> 8u) > 0u &&
                    std::isfinite(source.sourceOracleForce) &&
                    source.sourceOracleForce >= 0.0f,
                "NHHOOD2 muscle input is invalid");
        const std::uint32_t node = source.node - ray.nodeOffset;
        const auto proximal = worldPoint(
            poses, source.proximalCoreBody,
            {source.proximalLocalX, source.proximalLocalY,
             source.proximalLocalZ});
        std::array<double, 3u> force{
            proximal[0] - result.nodes[node].position[0],
            proximal[1] - result.nodes[node].position[1],
            proximal[2] - result.nodes[node].position[2],
        };
        const double forceNorm = norm(force);
        require(forceNorm > 1.0e-8,
                "NHHOOD2 muscle direction is degenerate");
        const double appliedForce = useSourceOracleForce
            ? static_cast<double>(source.sourceOracleForce)
            : literatureInputForce;
        for (double& component : force) component *= appliedForce / forceNorm;
        result.loads.push_back({node, force});
        result.maximumSourceOracleForce = std::max(
            result.maximumSourceOracleForce,
            static_cast<double>(source.sourceOracleForce));
    }
    return result;
}

bool bitwiseEqual(
    const NumiHumanTensionNetworkResult& a,
    const NumiHumanTensionNetworkResult& b
) {
    return a.position.size() == b.position.size() &&
        a.elementTension.size() == b.elementTension.size() &&
        std::memcmp(a.position.data(), b.position.data(),
                    a.position.size() * sizeof(a.position[0])) == 0 &&
        std::memcmp(a.elementTension.data(), b.elementTension.data(),
                    a.elementTension.size() * sizeof(double)) == 0 &&
        std::memcmp(&a.strainEnergy, &b.strainEnergy,
                    sizeof(double)) == 0;
}

NumiHumanTensionNetworkResult solve(const Fixture& fixture) {
    NumiHumanTensionNetworkResult result;
    metalrobo::NumiHumanTensionNetworkConfig config;
    config.maximumIterations = 512u;
    config.forceTolerance = 2.0e-7;
    const auto diagnostics = metalrobo::solveNumiHumanTensionNetwork(
        fixture.nodes, fixture.elements, fixture.loads, result, config);
    require(diagnostics.succeeded(),
            std::string("source-posed hood solve failed: ") +
                metalrobo::numiHumanTensionNetworkStatusName(
                    diagnostics.status) +
                " index=" + std::to_string(diagnostics.failingIndex));
    return result;
}

std::vector<double> projectTransferGeneralizedForce(
    const RigidPoseTable& rigid, const LoadedHood& hood,
    const HoodRayRecord& ray, const Fixture& fixture,
    const NumiHumanTensionNetworkResult& network
) {
    std::vector<metalrobo::ArticulatedPointQuery> queries;
    std::vector<std::array<double, 3u>> pointForces;
    for (std::uint32_t index = 0u; index < ray.nodeCount; ++index) {
        const auto& source = hood.nodes[ray.nodeOffset + index];
        if ((source.flags & 1u) == 0u) continue;
        queries.push_back({
            source.coreBody, {source.localX, source.localY, source.localZ}});
        const auto& reaction = network.fixedReactionForce[index];
        pointForces.push_back({-reaction[0], -reaction[1], -reaction[2]});
    }
    for (std::uint32_t index = 0u; index < ray.inputCount; ++index) {
        const auto& source = hood.inputs[ray.inputOffset + index];
        const auto& inputNode = hood.nodes[source.node];
        const std::uint32_t localNode = source.node - ray.nodeOffset;
        queries.push_back({
            inputNode.coreBody,
            localPoint(rigid.poseByCoreBody, inputNode.coreBody,
                       network.position[localNode]),
        });
        const auto& load = fixture.loads[index].force;
        pointForces.push_back({-load[0], -load[1], -load[2]});
    }
    require(queries.size() == pointForces.size() && !queries.empty(),
            "source-posed hood produced no point-force transfer");
    std::vector<metalrobo::ArticulatedPointKinematics> kinematics(
        queries.size());
    std::vector<double> jacobians(
        queries.size() * 3u * rigid.header.nv, 0.0);
    const auto diagnostics = metalrobo::computeArticulatedPointJacobians(
        rigid.model, 0u, rigid.q, rigid.v, queries, kinematics, jacobians);
    require(diagnostics.succeeded(),
            "source-posed hood point-Jacobian projection failed");
    std::vector<double> generalized(rigid.header.nv, 0.0);
    for (std::size_t point = 0u; point < queries.size(); ++point) {
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            const std::size_t row = (point * 3u + axis) * rigid.header.nv;
            for (std::uint32_t dof = 0u; dof < rigid.header.nv; ++dof) {
                generalized[dof] +=
                    jacobians[row + dof] * pointForces[point][axis];
            }
        }
    }
    require(std::all_of(generalized.begin(), generalized.end(),
                        [](const double value) { return std::isfinite(value); }),
            "source-posed hood generalized force is nonfinite");
    return generalized;
}

int run(
    const std::filesystem::path& rigidPath,
    const std::filesystem::path& hoodPath
) {
    const auto rigid = loadRigidPoseTable(rigidPath);
    const auto hood = loadHood(hoodPath, rigid.header);
    double maximumFreeResidual = 0.0;
    double maximumForceClosure = 0.0;
    double maximumMomentClosure = 0.0;
    double maximumSourceOracleForce = 0.0;
    double minimumSourceElementLength = std::numeric_limits<double>::infinity();
    double maximumSourceElementLength = 0.0;
    double totalStrainEnergy = 0.0;
    std::uint32_t totalActiveElements = 0u;
    bool replayBitwise = true;
    bool rollbackVerified = true;
    std::vector<double> combinedGeneralizedForce(rigid.header.nv, 0.0);
    double sourceMaximumFreeResidual = 0.0;
    double sourceMaximumForceClosure = 0.0;
    double sourceMaximumMomentClosure = 0.0;
    double sourceTotalStrainEnergy = 0.0;
    std::uint32_t sourceTotalActiveElements = 0u;
    bool sourceReplayBitwise = true;
    std::vector<double> rayMaximumInternalGeneralizedForce;
    rayMaximumInternalGeneralizedForce.reserve(hood.rays.size());
    for (const auto& ray : hood.rays) {
        const auto fixture = makeFixture(
            hood, ray, rigid.poseByCoreBody, false);
        minimumSourceElementLength = std::min(
            minimumSourceElementLength, fixture.minimumSourceElementLength);
        maximumSourceElementLength = std::max(
            maximumSourceElementLength, fixture.maximumSourceElementLength);
        const auto result = solve(fixture);
        const auto replay = solve(fixture);
        replayBitwise = replayBitwise && bitwiseEqual(result, replay);
        auto invalid = fixture.elements;
        invalid.front().nodeB = invalid.front().nodeA;
        auto rejected = result;
        const auto rejectedDiagnostics =
            metalrobo::solveNumiHumanTensionNetwork(
                fixture.nodes, invalid, fixture.loads, rejected);
        rollbackVerified = rollbackVerified &&
            !rejectedDiagnostics.succeeded() && bitwiseEqual(result, rejected);
        maximumFreeResidual = std::max(
            maximumFreeResidual, result.maximumFreeNodeResidual);
        maximumForceClosure = std::max(
            maximumForceClosure, norm(result.forceClosureResidual));
        maximumMomentClosure = std::max(
            maximumMomentClosure, norm(result.momentClosureResidual));
        maximumSourceOracleForce = std::max(
            maximumSourceOracleForce, fixture.maximumSourceOracleForce);
        totalStrainEnergy += result.strainEnergy;
        totalActiveElements += result.activeElementCount;
        const auto sourceFixture = makeFixture(
            hood, ray, rigid.poseByCoreBody, true);
        const auto sourceResult = solve(sourceFixture);
        const auto sourceReplay = solve(sourceFixture);
        sourceReplayBitwise = sourceReplayBitwise &&
            bitwiseEqual(sourceResult, sourceReplay);
        sourceMaximumFreeResidual = std::max(
            sourceMaximumFreeResidual,
            sourceResult.maximumFreeNodeResidual);
        sourceMaximumForceClosure = std::max(
            sourceMaximumForceClosure,
            norm(sourceResult.forceClosureResidual));
        sourceMaximumMomentClosure = std::max(
            sourceMaximumMomentClosure,
            norm(sourceResult.momentClosureResidual));
        sourceTotalStrainEnergy += sourceResult.strainEnergy;
        sourceTotalActiveElements += sourceResult.activeElementCount;
        const auto generalized = projectTransferGeneralizedForce(
            rigid, hood, ray, sourceFixture, sourceResult);
        double rayMaximum = 0.0;
        for (std::size_t dof = 6u; dof < generalized.size(); ++dof) {
            rayMaximum = std::max(rayMaximum, std::abs(generalized[dof]));
        }
        rayMaximumInternalGeneralizedForce.push_back(rayMaximum);
        for (std::size_t dof = 0u; dof < generalized.size(); ++dof) {
            combinedGeneralizedForce[dof] += generalized[dof];
        }
    }
    require(combinedGeneralizedForce.size() > 97u,
            "source-posed hood requires canonical MyoSim full-body DoFs");
    const double rootForceResidual = std::hypot(
        combinedGeneralizedForce[0], combinedGeneralizedForce[1],
        combinedGeneralizedForce[2]);
    const double rootMomentResidual = std::hypot(
        combinedGeneralizedForce[3], combinedGeneralizedForce[4],
        combinedGeneralizedForce[5]);
    double maximumInternalGeneralizedForce = 0.0;
    for (std::size_t dof = 6u; dof < combinedGeneralizedForce.size(); ++dof) {
        maximumInternalGeneralizedForce = std::max(
            maximumInternalGeneralizedForce,
            std::abs(combinedGeneralizedForce[dof]));
    }
    const double accumulatedRootClosureTolerance =
        2.0e-6 * static_cast<double>(hood.rays.size());
    const bool everyRayTransfersForce = std::all_of(
        rayMaximumInternalGeneralizedForce.begin(),
        rayMaximumInternalGeneralizedForce.end(),
        [](const double value) { return value > 1.0e-8; });
    require(replayBitwise && sourceReplayBitwise && rollbackVerified &&
                maximumFreeResidual <= 2.0e-7 &&
                maximumForceClosure <= 2.0e-6 &&
                maximumMomentClosure <= 2.0e-7 &&
                totalActiveElements >= 48u && totalStrainEnergy > 0.0 &&
                sourceMaximumFreeResidual <= 2.0e-7 &&
                sourceMaximumForceClosure <= 2.0e-6 &&
                sourceMaximumMomentClosure <= 2.0e-7 &&
                sourceTotalActiveElements >= 48u &&
                sourceTotalStrainEnergy > 0.0 &&
                minimumSourceElementLength >= 1.0e-4 &&
                maximumSourceElementLength <= 0.1 &&
                rootForceResidual <= accumulatedRootClosureTolerance &&
                rootMomentResidual <= accumulatedRootClosureTolerance &&
                maximumInternalGeneralizedForce > 1.0e-8 &&
                everyRayTransfersForce,
            "source-posed extensor hood qualification gate failed: free=" +
                std::to_string(maximumFreeResidual) +
                " closure_force=" + std::to_string(maximumForceClosure) +
                " closure_moment=" + std::to_string(maximumMomentClosure) +
                " root_force=" + std::to_string(rootForceResidual) +
                " root_moment=" + std::to_string(rootMomentResidual) +
                " internal=" +
                std::to_string(maximumInternalGeneralizedForce) +
                " source_free=" +
                std::to_string(sourceMaximumFreeResidual));
    std::cout << std::setprecision(12)
              << "numi_human_extensor_hood_source=passed"
              << " rays=" << hood.rays.size()
              << " hands=2 digits_per_hand=4"
              << " nodes=" << hood.nodes.size()
              << " elements=" << hood.elements.size()
              << " muscle_inputs=" << hood.inputs.size()
              << " active_elements=" << totalActiveElements
              << " strain_energy_j=" << totalStrainEnergy
              << " max_free_residual_n=" << maximumFreeResidual
              << " max_force_closure_residual_n=" << maximumForceClosure
              << " max_moment_closure_residual_nm=" << maximumMomentClosure
              << " max_source_oracle_force_n=" << maximumSourceOracleForce
              << " minimum_source_element_length_m="
              << minimumSourceElementLength
              << " maximum_source_element_length_m="
              << maximumSourceElementLength
              << " source_oracle_active_elements="
              << sourceTotalActiveElements
              << " source_oracle_strain_energy_j="
              << sourceTotalStrainEnergy
              << " source_oracle_max_free_residual_n="
              << sourceMaximumFreeResidual
              << " source_oracle_max_force_closure_residual_n="
              << sourceMaximumForceClosure
              << " source_oracle_max_moment_closure_residual_nm="
              << sourceMaximumMomentClosure
              << " root_generalized_force_residual_n=" << rootForceResidual
              << " root_generalized_moment_residual_nm=" << rootMomentResidual
              << " maximum_internal_generalized_force="
              << maximumInternalGeneralizedForce
              << " right_fifth_mcp_abduction_generalized_force_nm="
              << combinedGeneralizedForce[59]
              << " left_fifth_mcp_abduction_generalized_force_nm="
              << combinedGeneralizedForce[97]
              << " ray_maximum_internal_generalized_force=";
    for (std::size_t index = 0u;
         index < rayMaximumInternalGeneralizedForce.size(); ++index) {
        if (index != 0u) std::cout << ',';
        std::cout << (index < 4u ? 'R' : 'L') << (2u + index % 4u)
                  << ':' << rayMaximumInternalGeneralizedForce[index];
    }
    std::cout
              << " applied_input_force_each_n=2.9"
              << " replay=bitwise source_oracle_replay=bitwise rollback=verified"
              << " boundary=source_posed_cpu_fp64_point_jacobian_transfer_not_yet_live_muscle_force_replacement\n";
    return 0;
}

} // namespace

int main(const int argc, const char* const* argv) {
    try {
        if (argc != 3) {
            std::cerr << "usage: " << argv[0]
                      << " <myosim-fullbody-core-reference.nhrigid>"
                      << " <myosim-fullbody-extensor-hood.nhhood>\n";
            return 2;
        }
        return run(argv[1], argv[2]);
    } catch (const std::exception& error) {
        std::cerr << "numi_human_extensor_hood_source FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
