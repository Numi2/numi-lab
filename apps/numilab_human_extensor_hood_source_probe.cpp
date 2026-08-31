#include "metalrobo/NumiHumanTensionNetwork.hpp"

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
    'N', 'H', 'H', 'O', 'O', 'D', '1', '\0'};

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
    std::uint32_t handCount = 0u;
    std::uint32_t nodeCount = 0u;
    std::uint32_t elementCount = 0u;
    std::uint32_t inputCount = 0u;
    std::uint32_t handRecordBytes = 0u;
    std::uint32_t nodeRecordBytes = 0u;
    std::uint32_t elementRecordBytes = 0u;
    std::uint32_t inputRecordBytes = 0u;
    std::array<std::uint8_t, 32u> sourceSha256{};
};

struct HoodHandRecord {
    std::uint32_t nodeOffset = 0u;
    std::uint32_t nodeCount = 0u;
    std::uint32_t elementOffset = 0u;
    std::uint32_t elementCount = 0u;
    std::uint32_t inputOffset = 0u;
    std::uint32_t inputCount = 0u;
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
static_assert(sizeof(HoodHandRecord) == 24u);
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
    constexpr std::uint64_t prefixBytes = 96u + 48u;
    const std::uint64_t tableOffset = sizeof(RigidHeader) + prefixBytes +
        160ull * result.header.engineBodyCount +
        144ull * result.header.jointCount +
        64ull * result.header.nv + 4ull * result.header.nq +
        4ull * result.header.nv;
    input.seekg(static_cast<std::streamoff>(tableOffset), std::ios::beg);
    require(input.good(), "NHRIGID2 table offset is out of bounds");
    const auto sourceToCore = readVector<std::uint32_t>(
        input, result.header.sourceBodyCount, "NHRIGID2 source body map");
    const auto sourcePoses = readVector<SourcePoseRecord>(
        input, result.header.sourceBodyCount, "NHRIGID2 source poses");
    require(input.peek() == std::char_traits<char>::eof(),
            "NHRIGID2 payload has trailing bytes");
    result.poseByCoreBody.resize(result.header.engineBodyCount);
    std::vector<bool> assigned(result.header.engineBodyCount, false);
    for (std::size_t index = 0u; index < sourceToCore.size(); ++index) {
        require(sourceToCore[index] < result.poseByCoreBody.size(),
                "NHRIGID2 source body map is out of bounds");
        result.poseByCoreBody[sourceToCore[index]] = sourcePoses[index];
        assigned[sourceToCore[index]] = true;
    }
    return result;
}

struct LoadedHood {
    HoodHeader header{};
    std::vector<HoodHandRecord> hands;
    std::vector<HoodNodeRecord> nodes;
    std::vector<HoodElementRecord> elements;
    std::vector<HoodInputRecord> inputs;
};

LoadedHood loadHood(
    const std::filesystem::path& path, const RigidHeader& rigid
) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "cannot open NHHOOD1 payload");
    LoadedHood result;
    readObject(input, result.header, "NHHOOD1 header");
    require(result.header.magic == kHoodMagic &&
                result.header.payloadAbi == 1u,
            "unsupported NHHOOD1 payload");
    require(result.header.handCount == 2u &&
                result.header.handRecordBytes == sizeof(HoodHandRecord) &&
                result.header.nodeRecordBytes == sizeof(HoodNodeRecord) &&
                result.header.elementRecordBytes == sizeof(HoodElementRecord) &&
                result.header.inputRecordBytes == sizeof(HoodInputRecord) &&
                result.header.sourceSha256 == rigid.sourceSha256,
            "NHHOOD1 shape/source does not match NHRIGID2");
    result.hands = readVector<HoodHandRecord>(
        input, result.header.handCount, "NHHOOD1 hands");
    result.nodes = readVector<HoodNodeRecord>(
        input, result.header.nodeCount, "NHHOOD1 nodes");
    result.elements = readVector<HoodElementRecord>(
        input, result.header.elementCount, "NHHOOD1 elements");
    result.inputs = readVector<HoodInputRecord>(
        input, result.header.inputCount, "NHHOOD1 inputs");
    require(input.peek() == std::char_traits<char>::eof(),
            "NHHOOD1 payload has trailing bytes");
    for (std::size_t index = 0u; index < result.hands.size(); ++index) {
        const auto& hand = result.hands[index];
        require(hand.nodeOffset + hand.nodeCount <= result.nodes.size() &&
                    hand.elementOffset + hand.elementCount <= result.elements.size() &&
                    hand.inputOffset + hand.inputCount <= result.inputs.size(),
                "NHHOOD1 hand range is out of bounds");
        if (index != 0u) {
            const auto& previous = result.hands[index - 1u];
            require(hand.nodeOffset == previous.nodeOffset + previous.nodeCount &&
                        hand.elementOffset == previous.elementOffset + previous.elementCount &&
                        hand.inputOffset == previous.inputOffset + previous.inputCount,
                    "NHHOOD1 hand ranges are not contiguous");
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
    require(body < poses.size(), "NHHOOD1 Core body is out of bounds");
    const auto rotated = rotate(poses[body], local);
    return {rotated[0] + poses[body].positionX,
            rotated[1] + poses[body].positionY,
            rotated[2] + poses[body].positionZ};
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
};

Fixture makeFixture(
    const LoadedHood& hood, const HoodHandRecord& hand,
    const std::vector<SourcePoseRecord>& poses
) {
    Fixture result;
    result.nodes.reserve(hand.nodeCount);
    for (std::uint32_t index = 0u; index < hand.nodeCount; ++index) {
        const auto& source = hood.nodes[hand.nodeOffset + index];
        require((source.flags & 2u) != 0u && source.role == index,
                "NHHOOD1 node lacks exact initializer or canonical role order");
        result.nodes.push_back({
            worldPoint(poses, source.coreBody,
                       {source.localX, source.localY, source.localZ}),
            (source.flags & 1u) != 0u,
        });
    }
    result.elements.reserve(hand.elementCount);
    for (std::uint32_t index = 0u; index < hand.elementCount; ++index) {
        const auto& source = hood.elements[hand.elementOffset + index];
        require(source.nodeA >= hand.nodeOffset &&
                    source.nodeA < hand.nodeOffset + hand.nodeCount &&
                    source.nodeB >= hand.nodeOffset &&
                    source.nodeB < hand.nodeOffset + hand.nodeCount &&
                    source.nodeA != source.nodeB && source.provenance == 1u &&
                    source.restScale > 0.0f && source.restScale <= 1.0f &&
                    source.youngModulus > 0.0f && source.area > 0.0f,
                "NHHOOD1 element is invalid");
        const std::uint32_t a = source.nodeA - hand.nodeOffset;
        const std::uint32_t b = source.nodeB - hand.nodeOffset;
        result.elements.push_back({
            a, b, source.restScale * distance(
                result.nodes[a].position, result.nodes[b].position),
            source.youngModulus, source.area,
        });
    }
    constexpr double literatureInputForce = 2.9;
    result.loads.reserve(hand.inputCount);
    for (std::uint32_t index = 0u; index < hand.inputCount; ++index) {
        const auto& source = hood.inputs[hand.inputOffset + index];
        require(source.node >= hand.nodeOffset &&
                    source.node < hand.nodeOffset + hand.nodeCount &&
                    source.flags == 1u && std::isfinite(source.sourceOracleForce) &&
                    source.sourceOracleForce >= 0.0f,
                "NHHOOD1 muscle input is invalid");
        const std::uint32_t node = source.node - hand.nodeOffset;
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
                "NHHOOD1 muscle direction is degenerate");
        for (double& component : force) {
            component *= literatureInputForce / forceNorm;
        }
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
    double totalStrainEnergy = 0.0;
    std::uint32_t totalActiveElements = 0u;
    bool replayBitwise = true;
    bool rollbackVerified = true;
    for (const auto& hand : hood.hands) {
        const auto fixture = makeFixture(
            hood, hand, rigid.poseByCoreBody);
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
    }
    require(replayBitwise && rollbackVerified &&
                maximumFreeResidual <= 2.0e-7 &&
                maximumForceClosure <= 2.0e-6 &&
                maximumMomentClosure <= 2.0e-7 &&
                totalActiveElements >= 12u && totalStrainEnergy > 0.0,
            "source-posed extensor hood qualification gate failed");
    std::cout << std::setprecision(12)
              << "numi_human_extensor_hood_source=passed"
              << " hands=" << hood.hands.size()
              << " nodes=" << hood.nodes.size()
              << " elements=" << hood.elements.size()
              << " muscle_inputs=" << hood.inputs.size()
              << " active_elements=" << totalActiveElements
              << " strain_energy_j=" << totalStrainEnergy
              << " max_free_residual_n=" << maximumFreeResidual
              << " max_force_closure_residual_n=" << maximumForceClosure
              << " max_moment_closure_residual_nm=" << maximumMomentClosure
              << " max_source_oracle_force_n=" << maximumSourceOracleForce
              << " applied_input_force_each_n=2.9"
              << " replay=bitwise rollback=verified"
              << " boundary=source_posed_cpu_fp64_literature_load_not_live_whole_body_force_transaction\n";
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
