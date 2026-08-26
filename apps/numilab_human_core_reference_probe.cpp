#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/OpenSimSpatialTransform.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr std::array<char, 8u> kMagic{
    'N', 'H', 'R', 'I', 'G', 'I', 'D', '1',
};
constexpr std::uint32_t kPayloadAbi = 1u;

#pragma pack(push, 1)
struct PayloadHeader {
    std::array<char, 8u> magic{};
    std::uint32_t payloadAbi = 0u;
    std::uint32_t engineAbi = 0u;
    std::uint32_t sourceBodyCount = 0u;
    std::uint32_t engineBodyCount = 0u;
    std::uint32_t jointCount = 0u;
    std::uint32_t nq = 0u;
    std::uint32_t nv = 0u;
    std::uint32_t functionProgramCount = 0u;
    std::uint32_t reserved0 = 0u;
    std::array<std::uint8_t, 32u> sourceSha256{};
};
#pragma pack(pop)

static_assert(sizeof(PayloadHeader) == 76u);
static_assert(sizeof(MRWorldGPU) == 96u);
static_assert(sizeof(MRArticulationGPU) == 48u);
static_assert(sizeof(MRBodyPropertiesGPU) == 160u);
static_assert(sizeof(MRJointDescriptorGPU) == 144u);
static_assert(sizeof(MRDofPropertiesGPU) == 64u);
static_assert(sizeof(MROpenSimSpatialTransformGPU) == 2512u);

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
void readObject(std::istream& input, T& value, const char* what) {
    static_assert(std::is_trivially_copyable_v<T>);
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    require(input.good(), std::string("truncated ") + what);
}

template <typename T>
std::vector<T> readVector(
    std::istream& input,
    const std::size_t count,
    const char* what
) {
    std::vector<T> values(count);
    if (!values.empty()) {
        input.read(
            reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T))
        );
        require(input.good(), std::string("truncated ") + what);
    }
    return values;
}

std::string hexSha256(const std::array<std::uint8_t, 32u>& value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size() * 2u);
    for (const std::uint8_t byte : value) {
        result.push_back(digits[byte >> 4u]);
        result.push_back(digits[byte & 0x0fu]);
    }
    return result;
}

metalrobo::EngineModel loadReference(const char* path, PayloadHeader& header) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), std::string("cannot open payload ") + path);
    readObject(input, header, "payload header");
    require(header.magic == kMagic, "payload magic is not NHRIGID1");
    require(header.payloadAbi == kPayloadAbi, "unsupported Numi Human payload ABI");
    require(header.engineAbi == MR_ENGINE_ABI_VERSION, "payload/Core engine ABI mismatch");
    require(header.reserved0 == 0u, "payload reserved field is nonzero");
    require(header.sourceBodyCount > 0u, "payload has no source bodies");
    require(
        header.engineBodyCount == header.sourceBodyCount + 1u,
        "payload must contain exactly one synthetic fixed-root anchor"
    );
    require(
        header.jointCount == header.sourceBodyCount,
        "payload tree must have one inbound joint per source body"
    );
    require(header.nq == header.nv && header.nv > 0u, "payload generalized state is invalid");

    metalrobo::EngineModel model;
    model.name = "numilab_human_rajagopal_core_reference";
    readObject(input, model.world, "world record");
    MRArticulationGPU articulation{};
    readObject(input, articulation, "articulation record");
    model.articulations.push_back(articulation);
    model.bodies = readVector<MRBodyPropertiesGPU>(
        input, header.engineBodyCount, "body records"
    );
    model.joints = readVector<MRJointDescriptorGPU>(
        input, header.jointCount, "joint records"
    );
    model.dofs = readVector<MRDofPropertiesGPU>(input, header.nv, "DoF records");
    model.functionBasedJointPrograms.reserve(header.functionProgramCount);
    for (std::uint32_t index = 0u; index < header.functionProgramCount; ++index) {
        std::uint32_t jointIndex = MR_INVALID_INDEX;
        std::uint32_t reserved = 0u;
        MROpenSimSpatialTransformGPU packed{};
        readObject(input, jointIndex, "FunctionBased joint index");
        readObject(input, reserved, "FunctionBased reserved field");
        readObject(input, packed, "FunctionBased program");
        require(reserved == 0u, "FunctionBased program reserved field is nonzero");
        const auto decoded = metalrobo::unpackOpenSimSpatialTransformGPU(packed);
        require(decoded.succeeded(), "FunctionBased program is not canonical");
        model.functionBasedJointPrograms.push_back({jointIndex, decoded.transform});
    }
    model.defaultQ = readVector<float>(input, header.nq, "default q");
    model.defaultV = readVector<float>(input, header.nv, "default v");
    require(input.peek() == std::char_traits<char>::eof(), "payload contains trailing bytes");

    require(model.world.abiVersion == header.engineAbi, "world/header ABI mismatch");
    require(model.world.bodyCount == header.engineBodyCount, "world/header body count mismatch");
    require(model.world.articulationCount == 1u, "payload must contain one articulation");
    require(model.world.jointCount == header.jointCount, "world/header joint count mismatch");
    require(model.world.nq == header.nq && model.world.nv == header.nv, "world/header state mismatch");
    require(
        articulation.rootType == MR_ROOT_FIXED && articulation.rootBody == 0u &&
            articulation.bodyCount == header.engineBodyCount &&
            articulation.jointCount == header.jointCount &&
            articulation.nq == header.nq && articulation.nv == header.nv,
        "articulation/header tree mismatch"
    );
    std::string reason;
    require(model.valid(&reason), "Core reference model invalid: " + reason);
    return model;
}

bool allFinite(const std::vector<double>& values) {
    return std::all_of(values.begin(), values.end(), [](const double value) {
        return std::isfinite(value);
    });
}

} // namespace

int main(const int argc, char** argv) {
    try {
        require(argc == 2, "usage: metalrobo_numilab_human_core_reference_probe <payload.nhrigid>");
        PayloadHeader header{};
        const metalrobo::EngineModel model = loadReference(argv[1], header);
        std::vector<double> q(model.defaultQ.begin(), model.defaultQ.end());
        std::vector<double> v(model.defaultV.size(), 0.0);
        std::vector<double> acceleration(model.defaultV.size(), 0.0);
        for (std::size_t index = 0u; index < v.size(); ++index) {
            v[index] = 0.01 * std::sin(0.37 * static_cast<double>(index + 1u));
            acceleration[index] = 0.02 * std::cos(0.23 * static_cast<double>(index + 1u));
        }

        metalrobo::ArticulatedDynamicsConfig config;
        config.gravity = {
            model.world.gravityAndTimestep.x,
            model.world.gravityAndTimestep.y,
            model.world.gravityAndTimestep.z,
        };
        config.timestep = model.world.gravityAndTimestep.w;
        config.applyBodyDamping = false;
        std::vector<metalrobo::ArticulatedBodyKinematics> kinematics(model.bodies.size());
        const auto kinematicsDiagnostics = metalrobo::computeArticulatedBodyKinematics(
            model, 0u, q, v, kinematics, config
        );
        require(kinematicsDiagnostics.succeeded(), "full Rajagopal kinematics failed");

        std::vector<double> massMatrix(v.size() * v.size(), 0.0);
        const auto massDiagnostics = metalrobo::computeArticulatedMassMatrix(
            model, 0u, q, massMatrix, config
        );
        require(massDiagnostics.succeeded(), "full Rajagopal mass-matrix assembly failed");
        require(
            massDiagnostics.minimumCholeskyPivot > 0.0 && allFinite(massMatrix),
            "full Rajagopal mass matrix is not finite positive definite"
        );

        double massSymmetryError = 0.0;
        for (std::size_t row = 0u; row < v.size(); ++row) {
            for (std::size_t column = 0u; column < v.size(); ++column) {
                massSymmetryError = std::max(
                    massSymmetryError,
                    std::abs(massMatrix[row * v.size() + column] -
                             massMatrix[column * v.size() + row])
                );
            }
        }
        require(massSymmetryError < 1.0e-10, "full Rajagopal mass matrix is asymmetric");

        std::vector<double> generalizedForce(v.size(), 0.0);
        const auto inverseDiagnostics = metalrobo::computeArticulatedInverseDynamics(
            model, 0u, q, v, acceleration, {}, generalizedForce, config
        );
        require(inverseDiagnostics.succeeded(), "full Rajagopal inverse dynamics failed");
        std::vector<double> recoveredAcceleration(v.size(), 0.0);
        const auto forwardDiagnostics = metalrobo::computeArticulatedForwardDynamics(
            model, 0u, q, v, generalizedForce, {}, recoveredAcceleration, config
        );
        require(forwardDiagnostics.succeeded(), "full Rajagopal forward dynamics failed");
        double forwardInverseError = 0.0;
        for (std::size_t index = 0u; index < acceleration.size(); ++index) {
            forwardInverseError = std::max(
                forwardInverseError,
                std::abs(recoveredAcceleration[index] - acceleration[index])
            );
        }
        require(
            allFinite(generalizedForce) && allFinite(recoveredAcceleration) &&
                forwardInverseError < 2.0e-7,
            "full Rajagopal forward/inverse closure exceeded tolerance"
        );

        metalrobo::ArticulatedInvariants invariants{};
        const auto invariantDiagnostics = metalrobo::computeArticulatedInvariants(
            model, 0u, q, v, invariants, config
        );
        require(invariantDiagnostics.succeeded(), "full Rajagopal invariant evaluation failed");
        require(
            std::isfinite(invariants.kineticEnergy) &&
                std::isfinite(invariants.potentialEnergy) &&
                std::isfinite(invariants.totalEnergy),
            "full Rajagopal invariants are non-finite"
        );

        std::cout << std::scientific << std::setprecision(6)
                  << "numilab_human_core_reference=ok"
                  << " source_sha256=" << hexSha256(header.sourceSha256)
                  << " source_bodies=" << header.sourceBodyCount
                  << " engine_bodies=" << header.engineBodyCount
                  << " joints=" << header.jointCount
                  << " nq=" << header.nq
                  << " nv=" << header.nv
                  << " function_based_programs=" << header.functionProgramCount
                  << " mass_pivot=" << massDiagnostics.minimumCholeskyPivot
                  << " mass_symmetry_error=" << massSymmetryError
                  << " forward_inverse_error=" << forwardInverseError
                  << " kinetic_energy=" << invariants.kineticEnergy
                  << " total_energy=" << invariants.totalEnergy
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "numilab_human_core_reference=failed " << error.what() << '\n';
        return 1;
    }
}
