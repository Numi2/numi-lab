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

constexpr std::array<char, 8u> kRigidMagic{
    'N', 'H', 'R', 'I', 'G', 'I', 'D', '2'};
constexpr std::array<char, 8u> kAdmMagic{
    'N', 'H', 'A', 'D', 'M', '1', '\0', '\0'};

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

struct AdmHeader {
    std::array<char, 8u> magic{};
    std::uint32_t payloadAbi = 0u;
    std::uint32_t handCount = 0u;
    std::uint32_t recordBytes = 0u;
    std::array<std::uint8_t, 32u> bodyPartsArchiveSha256{};
};

struct AdmRecord {
    std::uint32_t side = 0u;
    std::uint32_t originBody = 0u;
    std::uint32_t insertionBody = 0u;
    std::uint32_t flags = 0u;
    std::array<float, 3u> originLocal{};
    std::array<float, 3u> insertionLocal{};
    float optimalFibreLength = 0.0f;
    float externalTendonLength = 0.0f;
    float pcsaMm2 = 0.0f;
    float forceLow = 0.0f;
    float forceNominal = 0.0f;
    float forceHigh = 0.0f;
};
#pragma pack(pop)

static_assert(sizeof(RigidHeader) == 80u);
static_assert(sizeof(SourcePoseRecord) == 28u);
static_assert(sizeof(AdmHeader) == 52u);
static_assert(sizeof(AdmRecord) == 64u);

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

struct RigidModel {
    RigidHeader header{};
    metalrobo::EngineModel model;
    std::vector<double> q;
    std::vector<double> v;
};

RigidModel loadRigid(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "cannot open NHRIGID2 payload");
    RigidModel result;
    readObject(input, result.header, "NHRIGID2 header");
    require(result.header.magic == kRigidMagic &&
                result.header.payloadAbi == 1u &&
                result.header.sourceBodyCount > 0u &&
                result.header.engineBodyCount >= result.header.sourceBodyCount &&
                result.header.nq == result.header.nv + 1u,
            "unsupported NHRIGID2 payload");
    result.model.name = "numilab_human_adm_feasibility";
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
    const auto q = readVector<float>(input, result.header.nq, "NHRIGID2 q");
    const auto v = readVector<float>(input, result.header.nv, "NHRIGID2 v");
    result.model.defaultQ = q;
    result.model.defaultV = v;
    result.q.assign(q.begin(), q.end());
    result.v.assign(v.begin(), v.end());
    static_cast<void>(readVector<std::uint32_t>(
        input, result.header.sourceBodyCount, "NHRIGID2 source body map"));
    static_cast<void>(readVector<SourcePoseRecord>(
        input, result.header.sourceBodyCount, "NHRIGID2 source poses"));
    require(input.peek() == std::char_traits<char>::eof(),
            "NHRIGID2 payload has trailing bytes");
    std::string reason;
    require(result.model.valid(&reason),
            "NHRIGID2 EngineModel is invalid: " + reason);
    return result;
}

struct AdmPayload {
    AdmHeader header{};
    std::vector<AdmRecord> hands;
};

AdmPayload loadAdm(const std::filesystem::path& path, const RigidModel& rigid) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "cannot open NHADM1 payload");
    AdmPayload result;
    readObject(input, result.header, "NHADM1 header");
    require(result.header.magic == kAdmMagic &&
                result.header.payloadAbi == 1u &&
                result.header.handCount == 2u &&
                result.header.recordBytes == sizeof(AdmRecord),
            "unsupported NHADM1 payload");
    result.hands = readVector<AdmRecord>(
        input, result.header.handCount, "NHADM1 hand records");
    require(input.peek() == std::char_traits<char>::eof(),
            "NHADM1 payload has trailing bytes");
    for (std::size_t index = 0u; index < result.hands.size(); ++index) {
        const auto& hand = result.hands[index];
        const auto finitePositive = [](const float value) {
            return std::isfinite(value) && value > 0.0f;
        };
        require(hand.side == index && (hand.flags & 1u) != 0u &&
                    hand.originBody < rigid.model.bodies.size() &&
                    hand.insertionBody < rigid.model.bodies.size() &&
                    hand.originBody != hand.insertionBody &&
                    std::all_of(hand.originLocal.begin(), hand.originLocal.end(),
                                [](const float value) { return std::isfinite(value); }) &&
                    std::all_of(hand.insertionLocal.begin(), hand.insertionLocal.end(),
                                [](const float value) { return std::isfinite(value); }) &&
                    finitePositive(hand.optimalFibreLength) &&
                    finitePositive(hand.externalTendonLength) &&
                    finitePositive(hand.pcsaMm2) &&
                    finitePositive(hand.forceLow) &&
                    hand.forceLow <= hand.forceNominal &&
                    hand.forceNominal <= hand.forceHigh,
                "NHADM1 hand record is invalid");
    }
    return result;
}

double norm(const std::array<double, 3u>& value) {
    return std::hypot(value[0], value[1], value[2]);
}

struct Evaluation {
    std::array<double, 2u> length{};
    std::array<double, 2u> targetMomentArm{};
    std::array<double, 2u> finiteDifferenceMomentArm{};
    std::array<double, 2u> requiredForce{};
    std::array<double, 2u> correctedResidual{};
    std::array<double, 2u> rootForceResidual{};
    std::array<double, 2u> rootMomentResidual{};
    std::array<double, 2u> maximumCollateralForce{};
};

struct PointResult {
    std::vector<metalrobo::ArticulatedPointKinematics> points;
    std::vector<double> jacobians;
};

PointResult evaluatePoints(
    const RigidModel& rigid, const AdmPayload& adm,
    const std::vector<double>& q
) {
    std::vector<metalrobo::ArticulatedPointQuery> queries;
    queries.reserve(4u);
    for (const auto& hand : adm.hands) {
        queries.push_back({hand.originBody,
            {hand.originLocal[0], hand.originLocal[1], hand.originLocal[2]}});
        queries.push_back({hand.insertionBody,
            {hand.insertionLocal[0], hand.insertionLocal[1], hand.insertionLocal[2]}});
    }
    PointResult result;
    result.points.resize(queries.size());
    result.jacobians.resize(
        queries.size() * 3u * rigid.header.nv, 0.0);
    const auto diagnostics = metalrobo::computeArticulatedPointJacobians(
        rigid.model, 0u, q, rigid.v, queries, result.points,
        result.jacobians);
    require(diagnostics.succeeded(), "ADM point-Jacobian evaluation failed");
    return result;
}

double segmentLength(const PointResult& points, const std::size_t hand) {
    const auto& origin = points.points[2u * hand].position;
    const auto& insertion = points.points[2u * hand + 1u].position;
    return norm({insertion[0] - origin[0], insertion[1] - origin[1],
                 insertion[2] - origin[2]});
}

Evaluation evaluate(
    const RigidModel& rigid, const AdmPayload& adm,
    const std::array<double, 2u>& residual
) {
    constexpr std::array<std::uint32_t, 2u> targetDof{59u, 97u};
    constexpr double epsilon = 1.0e-6;
    Evaluation result;
    const auto points = evaluatePoints(rigid, adm, rigid.q);
    for (std::size_t hand = 0u; hand < adm.hands.size(); ++hand) {
        const auto& origin = points.points[2u * hand].position;
        const auto& insertion = points.points[2u * hand + 1u].position;
        std::array<double, 3u> direction{
            insertion[0] - origin[0], insertion[1] - origin[1],
            insertion[2] - origin[2]};
        result.length[hand] = norm(direction);
        require(result.length[hand] > 1.0e-4,
                "ADM registered route is degenerate");
        for (double& value : direction) value /= result.length[hand];
        std::vector<double> unitGeneralized(rigid.header.nv, 0.0);
        for (std::uint32_t dof = 0u; dof < rigid.header.nv; ++dof) {
            for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
                const std::size_t originRow =
                    ((2u * hand) * 3u + axis) * rigid.header.nv;
                const std::size_t insertionRow =
                    ((2u * hand + 1u) * 3u + axis) * rigid.header.nv;
                unitGeneralized[dof] += direction[axis] *
                    (points.jacobians[originRow + dof] -
                     points.jacobians[insertionRow + dof]);
            }
        }
        result.targetMomentArm[hand] = unitGeneralized[targetDof[hand]];
        require(std::abs(result.targetMomentArm[hand]) > 1.0e-5,
                "ADM route has negligible fifth-MCP abduction leverage");
        result.requiredForce[hand] =
            -residual[hand] / result.targetMomentArm[hand];
        result.correctedResidual[hand] = residual[hand] +
            result.requiredForce[hand] * result.targetMomentArm[hand];
        double rootForceSquared = 0.0;
        double rootMomentSquared = 0.0;
        double maximumCollateral = 0.0;
        for (std::uint32_t dof = 0u; dof < rigid.header.nv; ++dof) {
            const double force = result.requiredForce[hand] * unitGeneralized[dof];
            if (dof < 3u) rootForceSquared += force * force;
            else if (dof < 6u) rootMomentSquared += force * force;
            else if (dof != targetDof[hand]) {
                maximumCollateral = std::max(maximumCollateral, std::abs(force));
            }
        }
        result.rootForceResidual[hand] = std::sqrt(rootForceSquared);
        result.rootMomentResidual[hand] = std::sqrt(rootMomentSquared);
        result.maximumCollateralForce[hand] = maximumCollateral;

        const auto qIndex = rigid.model.dofs[targetDof[hand]].qIndex;
        require(qIndex != MR_INVALID_INDEX && qIndex < rigid.q.size(),
                "ADM target DoF has no generalized coordinate");
        auto qMinus = rigid.q;
        auto qPlus = rigid.q;
        qMinus[qIndex] -= epsilon;
        qPlus[qIndex] += epsilon;
        const double derivative =
            (segmentLength(evaluatePoints(rigid, adm, qPlus), hand) -
             segmentLength(evaluatePoints(rigid, adm, qMinus), hand)) /
            (2.0 * epsilon);
        result.finiteDifferenceMomentArm[hand] = -derivative;
    }
    return result;
}

bool bitwiseEqual(const Evaluation& a, const Evaluation& b) {
    return std::memcmp(&a, &b, sizeof(Evaluation)) == 0;
}

int run(
    const std::filesystem::path& rigidPath,
    const std::filesystem::path& admPath,
    const std::array<double, 2u>& residual
) {
    const auto rigid = loadRigid(rigidPath);
    const auto adm = loadAdm(admPath, rigid);
    require(rigid.header.nv > 97u && residual[0] < 0.0 && residual[1] < 0.0,
            "ADM feasibility requires canonical negative fifth-MCP residuals");
    const auto result = evaluate(rigid, adm, residual);
    const auto replay = evaluate(rigid, adm, residual);
    require(bitwiseEqual(result, replay),
            "ADM feasibility evaluation did not replay bitwise");
    double maximumJacobianError = 0.0;
    bool forceBandFeasible = true;
    for (std::size_t hand = 0u; hand < 2u; ++hand) {
        maximumJacobianError = std::max(
            maximumJacobianError,
            std::abs(result.targetMomentArm[hand] -
                     result.finiteDifferenceMomentArm[hand]));
        require(std::abs(result.correctedResidual[hand]) <= 1.0e-12 &&
                    result.rootForceResidual[hand] <= 1.0e-9 &&
                    result.rootMomentResidual[hand] <= 1.0e-9,
                "ADM point-force projection failed force/moment closure");
        forceBandFeasible = forceBandFeasible &&
            result.requiredForce[hand] > 0.0 &&
            result.requiredForce[hand] <= adm.hands[hand].forceHigh;
    }
    require(maximumJacobianError <= 2.0e-8,
            "ADM analytic moment arm disagrees with the FP64 central-difference oracle");
    const double bilateralMomentArmRelativeDifference =
        std::abs(std::abs(result.targetMomentArm[0]) -
                 std::abs(result.targetMomentArm[1])) /
        std::max(std::abs(result.targetMomentArm[0]),
                 std::abs(result.targetMomentArm[1]));
    require(bilateralMomentArmRelativeDifference <= 0.25,
            "ADM bilateral moment-arm parity exceeds 25 percent");
    std::cout << std::setprecision(12)
              << "numi_human_adm_feasibility="
              << (forceBandFeasible ? "qualified" : "rejected_wrong_sign_or_capacity")
              << " qualified=" << (forceBandFeasible ? "true" : "false")
              << " right_route_length_m=" << result.length[0]
              << " left_route_length_m=" << result.length[1]
              << " right_moment_arm_m=" << result.targetMomentArm[0]
              << " left_moment_arm_m=" << result.targetMomentArm[1]
              << " right_required_force_n=" << result.requiredForce[0]
              << " left_required_force_n=" << result.requiredForce[1]
              << " right_corrected_residual_nm=" << result.correctedResidual[0]
              << " left_corrected_residual_nm=" << result.correctedResidual[1]
              << " force_capacity_low_n=" << adm.hands[0].forceLow
              << " force_capacity_nominal_n=" << adm.hands[0].forceNominal
              << " force_capacity_high_n=" << adm.hands[0].forceHigh
              << " maximum_jacobian_oracle_error_m=" << maximumJacobianError
              << " bilateral_moment_arm_relative_difference="
              << bilateralMomentArmRelativeDifference
              << " maximum_collateral_generalized_force="
              << std::max(result.maximumCollateralForce[0],
                          result.maximumCollateralForce[1])
              << " root_force_residual_n="
              << std::max(result.rootForceResidual[0],
                          result.rootForceResidual[1])
              << " root_moment_residual_nm="
              << std::max(result.rootMomentResidual[0],
                          result.rootMomentResidual[1])
              << " replay=bitwise"
              << " boundary=source_inferred_cpu_fp64_feasibility_not_yet_live_hill_type_actuation_or_static_equilibrium\n";
    return 0;
}

} // namespace

int main(const int argc, const char* const* argv) {
    try {
        if (argc != 5) {
            std::cerr << "usage: " << argv[0]
                      << " <myosim-fullbody-core-reference.nhrigid>"
                      << " <bodyparts-adm-inference.nhadm>"
                      << " <right-fifth-mcp-residual-nm>"
                      << " <left-fifth-mcp-residual-nm>\n";
            return 2;
        }
        const std::array<double, 2u> residual{
            std::stod(argv[3]), std::stod(argv[4])};
        require(std::isfinite(residual[0]) && std::isfinite(residual[1]),
                "ADM residual inputs are non-finite");
        return run(argv[1], argv[2], residual);
    } catch (const std::exception& error) {
        std::cerr << "numi_human_adm_feasibility FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
