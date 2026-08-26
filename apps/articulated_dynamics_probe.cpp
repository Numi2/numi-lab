#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/G1.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

mr_float4 f4(
    const double x,
    const double y,
    const double z,
    const double w = 0.0
) {
    return {
        static_cast<float>(x),
        static_cast<float>(y),
        static_cast<float>(z),
        static_cast<float>(w),
    };
}

MRDofPropertiesGPU rootDof(const std::uint32_t localDof) {
    MRDofPropertiesGPU result{};
    result.articulationIndex = 0u;
    result.jointIndex = MR_INVALID_INDEX;
    result.qIndex = localDof < 3u
        ? localDof
        : MR_INVALID_INDEX;
    result.vIndex = localDof;
    result.localDof = localDof;
    result.flags = MR_DOF_FLAG_ROOT;
    return result;
}

MRDofPropertiesGPU passiveJointDof(
    const std::uint32_t jointIndex,
    const std::uint32_t qIndex,
    const std::uint32_t vIndex
) {
    MRDofPropertiesGPU result{};
    result.articulationIndex = 0u;
    result.jointIndex = jointIndex;
    result.qIndex = qIndex;
    result.vIndex = vIndex;
    return result;
}

mr_float4 quaternionFromRpy(
    const double roll,
    const double pitch,
    const double yaw
) {
    const double cr = std::cos(0.5 * roll);
    const double sr = std::sin(0.5 * roll);
    const double cp = std::cos(0.5 * pitch);
    const double sp = std::sin(0.5 * pitch);
    const double cy = std::cos(0.5 * yaw);
    const double sy = std::sin(0.5 * yaw);
    return f4(
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy
    );
}

MRBodyPropertiesGPU body(
    const std::uint32_t parent,
    const std::uint32_t inboundJoint,
    const double mass,
    const std::array<double, 3> centerOfMassInLink,
    const std::array<double, 3> inertia
) {
    MRBodyPropertiesGPU result{};
    result.articulationIndex = 0u;
    result.parentBody = parent;
    result.inboundJoint = inboundJoint;
    result.motionType = MR_MOTION_DYNAMIC;
    result.massAndInverseMass = f4(mass, 1.0 / mass, 0.0, 0.0);
    // Metadata remains the URDF link-frame offset. Dynamics translation and
    // joint anchors are COM-based.
    result.centerOfMass = f4(
        centerOfMassInLink[0],
        centerOfMassInLink[1],
        centerOfMassInLink[2]
    );
    result.inertiaRow0 = f4(inertia[0], 0.0, 0.0);
    result.inertiaRow1 = f4(0.0, inertia[1], 0.0);
    result.inertiaRow2 = f4(0.0, 0.0, inertia[2]);
    result.inverseInertiaRow0 = f4(1.0 / inertia[0], 0.0, 0.0);
    result.inverseInertiaRow1 = f4(0.0, 1.0 / inertia[1], 0.0);
    result.inverseInertiaRow2 = f4(0.0, 0.0, 1.0 / inertia[2]);
    result.dampingAndSpeedLimits = f4(0.0, 0.0, 1.0e6, 1.0e6);
    return result;
}

metalrobo::EngineModel makeFreeBodyModel() {
    metalrobo::EngineModel model;
    model.name = "fp64_free_body";
    MRArticulationGPU articulation{};
    articulation.rootBody = 0u;
    articulation.rootType = MR_ROOT_FLOATING;
    articulation.firstBody = 0u;
    articulation.bodyCount = 1u;
    articulation.firstJoint = 0u;
    articulation.jointCount = 0u;
    articulation.qOffset = 0u;
    articulation.nq = 7u;
    articulation.vOffset = 0u;
    articulation.nv = 6u;
    model.articulations.push_back(articulation);
    for (std::uint32_t localDof = 0u;
         localDof < 6u;
         ++localDof) {
        model.dofs.push_back(rootDof(localDof));
    }
    model.bodies.push_back(body(
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
        2.5,
        {0.17, -0.08, 0.04},
        {0.7, 1.1, 1.6}
    ));
    model.defaultQ = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    model.defaultV.assign(6u, 0.0f);
    return model;
}

metalrobo::EngineModel makePendulumModel(const double length) {
    metalrobo::EngineModel model;
    model.name = "analytic_one_link_pendulum";
    MRArticulationGPU articulation{};
    articulation.rootBody = 0u;
    articulation.rootType = MR_ROOT_FIXED;
    articulation.firstBody = 0u;
    articulation.bodyCount = 2u;
    articulation.firstJoint = 0u;
    articulation.jointCount = 1u;
    articulation.qOffset = 0u;
    articulation.nq = 1u;
    articulation.vOffset = 0u;
    articulation.nv = 1u;
    model.articulations.push_back(articulation);
    model.bodies.push_back(body(
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
        3.0,
        {0.0, 0.0, 0.0},
        {0.3, 0.4, 0.5}
    ));
    model.bodies.push_back(body(
        0u,
        0u,
        1.7,
        {length, 0.0, 0.0},
        {0.08, 0.11, 0.16}
    ));

    MRJointDescriptorGPU joint{};
    joint.parentBody = 0u;
    joint.childBody = 1u;
    joint.jointType = MR_JOINT_REVOLUTE;
    joint.qOffset = 0u;
    joint.nq = 1u;
    joint.vOffset = 0u;
    joint.nv = 1u;
    joint.axis0 = f4(0.0, 0.0, 1.0);
    joint.parentAnchor = f4(0.0, 0.0, 0.0);
    // The joint is length metres in the -x direction from child COM.
    joint.childAnchor = f4(-length, 0.0, 0.0);
    joint.parentRotation = f4(0.0, 0.0, 0.0, 1.0);
    joint.childRotation = f4(0.0, 0.0, 0.0, 1.0);
    model.joints.push_back(joint);
    model.dofs.push_back(passiveJointDof(0u, 0u, 0u));
    model.defaultQ = {0.0f};
    model.defaultV = {0.0f};
    return model;
}

metalrobo::OpenSimSpatialTransformDefinition rajagopalWalkerKneeTransform() {
    // Exact walker_knee_r source coefficients from the pinned
    // RajagopalLaiUhlrich2023 model. This is intentionally the real
    // FunctionBased program rather than a surrogate one-DoF screw.
    using Axis = metalrobo::OpenSimSpatialAxisDefinition;
    using Function = metalrobo::OpenSimFunctionDefinition;
    using Kind = metalrobo::OpenSimFunctionKind;
    const auto axis = [](
                          const std::array<double, 3> direction,
                          const Function& function
                      ) {
        return Axis{
            .axis = direction,
            .coordinateIndex = 0u,
            .function = function,
        };
    };
    metalrobo::OpenSimSpatialTransformDefinition source{};
    source.coordinateCount = 1u;
    source.axes = {
        axis({1.0, 0.0, 0.0}, {.kind = Kind::linear, .coefficients = {1.0, 0.0}}),
        axis({0.0, 0.0, 1.0}, {.kind = Kind::polynomial, .coefficients = {
            0.010832094539863, -0.025218325501241,
            -0.032847810398852, 0.079100011967027,
            -1.473252350900463e-08,
        }}),
        axis({0.0, 1.0, 0.0}, {.kind = Kind::polynomial, .coefficients = {
            0.025165762727423, -0.16948005139054,
            0.369499348688249, -4.430358308836305e-08,
        }}),
        axis({1.0, 0.0, 0.0}, {.kind = Kind::polynomial, .coefficients = {
            0.0001590447878850381, -0.001015149915669,
            0.001817510974968, 2.64142664519923e-05,
            -7.746563532471892e-07,
        }}),
        axis({0.0, 1.0, 0.0}, {.kind = Kind::polynomial, .coefficients = {
            -0.0005796878052338684, 0.005079765745626,
            -0.011442375726364, 0.003936908668844,
            -2.516350383213525e-05,
        }}),
        axis({0.0, 0.0, 1.0}, {.kind = Kind::polynomial, .coefficients = {
            0.001208086889206, -0.004453611224706,
            0.000611649407298173, 0.006265429606387,
            -1.461912533723326e-05,
        }}),
    };
    return source;
}

metalrobo::EngineModel makeFunctionBasedWalkerKneeModel(
    const double length
) {
    metalrobo::EngineModel model;
    model.name = "rajagopal_walker_knee_function_based_reference";
    model.world.abiVersion = MR_ENGINE_ABI_VERSION;
    model.world.bodyCount = 2u;
    model.world.articulationCount = 1u;
    model.world.jointCount = 1u;
    model.world.nq = 1u;
    model.world.nv = 1u;
    model.world.pairCapacity = 1u;
    model.world.contactCapacity = 1u;
    model.world.constraintCapacity = 1u;
    model.world.islandCapacity = 1u;
    model.world.solverType = MR_SOLVER_REFERENCE_FP64;
    model.world.frictionConeType = MR_FRICTION_CONE_ELLIPTIC;
    model.world.gravityAndTimestep = f4(0.0, -9.81, 0.0, 1.0 / 1000.0);
    model.world.solverScales = f4(1.0e-8, 1.0e-9, 2.0, 1.0e-4);

    MRArticulationGPU articulation{};
    articulation.rootBody = 0u;
    articulation.rootType = MR_ROOT_FIXED;
    articulation.firstBody = 0u;
    articulation.bodyCount = 2u;
    articulation.firstJoint = 0u;
    articulation.jointCount = 1u;
    articulation.qOffset = 0u;
    articulation.nq = 1u;
    articulation.vOffset = 0u;
    articulation.nv = 1u;
    model.articulations.push_back(articulation);
    model.bodies.push_back(body(
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
        9.0,
        {0.0, 0.0, 0.0},
        {1.3, 1.4, 1.5}
    ));
    model.bodies.push_back(body(
        0u,
        0u,
        2.1,
        {length, 0.0, 0.0},
        {0.12, 0.16, 0.19}
    ));

    MRJointDescriptorGPU joint{};
    joint.parentBody = 0u;
    joint.childBody = 1u;
    joint.jointType = MR_JOINT_FUNCTION_BASED;
    joint.qOffset = 0u;
    joint.nq = 1u;
    joint.vOffset = 0u;
    joint.nv = 1u;
    joint.parentAnchor = f4(0.0, 0.0, 0.0);
    joint.childAnchor = f4(-length, 0.0, 0.0);
    joint.parentRotation = f4(0.0, 0.0, 0.0, 1.0);
    joint.childRotation = f4(0.0, 0.0, 0.0, 1.0);
    model.joints.push_back(joint);
    model.dofs.push_back(passiveJointDof(0u, 0u, 0u));
    const auto compiled = metalrobo::compileOpenSimSpatialTransform(
        rajagopalWalkerKneeTransform()
    );
    if (!compiled.succeeded()) {
        throw std::runtime_error("Rajagopal walker knee compilation failed");
    }
    model.functionBasedJointPrograms.push_back({
        .jointIndex = 0u,
        .transform = compiled.transform,
    });
    model.defaultQ = {0.0f};
    model.defaultV = {0.0f};
    return model;
}

metalrobo::EngineModel makeFloatingChainModel() {
    metalrobo::EngineModel model;
    model.name = "floating_two_joint_chain";
    MRArticulationGPU articulation{};
    articulation.rootBody = 0u;
    articulation.rootType = MR_ROOT_FLOATING;
    articulation.firstBody = 0u;
    articulation.bodyCount = 3u;
    articulation.firstJoint = 0u;
    articulation.jointCount = 2u;
    articulation.qOffset = 0u;
    articulation.nq = 9u;
    articulation.vOffset = 0u;
    articulation.nv = 8u;
    model.articulations.push_back(articulation);
    for (std::uint32_t localDof = 0u;
         localDof < 6u;
         ++localDof) {
        model.dofs.push_back(rootDof(localDof));
    }
    model.bodies.push_back(body(
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
        4.2,
        {0.03, -0.02, 0.06},
        {0.62, 0.83, 0.97}
    ));
    model.bodies.push_back(body(
        0u,
        0u,
        1.3,
        {0.22, 0.01, -0.02},
        {0.09, 0.14, 0.17}
    ));
    model.bodies.push_back(body(
        1u,
        1u,
        0.8,
        {0.16, -0.015, 0.025},
        {0.05, 0.07, 0.095}
    ));

    MRJointDescriptorGPU joint0{};
    joint0.parentBody = 0u;
    joint0.childBody = 1u;
    joint0.jointType = MR_JOINT_REVOLUTE;
    joint0.qOffset = 7u;
    joint0.nq = 1u;
    joint0.vOffset = 6u;
    joint0.nv = 1u;
    joint0.axis0 = f4(0.0, 1.0, 0.25);
    joint0.parentAnchor = f4(0.31, 0.07, -0.04);
    joint0.childAnchor = f4(-0.22, -0.01, 0.02);
    joint0.parentRotation = quaternionFromRpy(0.08, -0.12, 0.04);
    joint0.childRotation = quaternionFromRpy(-0.03, 0.06, 0.02);
    model.joints.push_back(joint0);
    model.dofs.push_back(passiveJointDof(0u, 7u, 6u));

    MRJointDescriptorGPU joint1{};
    joint1.parentBody = 1u;
    joint1.childBody = 2u;
    joint1.jointType = MR_JOINT_REVOLUTE;
    joint1.qOffset = 8u;
    joint1.nq = 1u;
    joint1.vOffset = 7u;
    joint1.nv = 1u;
    joint1.axis0 = f4(0.2, -0.1, 1.0);
    joint1.parentAnchor = f4(0.24, -0.03, 0.08);
    joint1.childAnchor = f4(-0.16, 0.015, -0.025);
    joint1.parentRotation = quaternionFromRpy(-0.07, 0.02, 0.11);
    joint1.childRotation = quaternionFromRpy(0.04, -0.05, 0.03);
    model.joints.push_back(joint1);
    model.dofs.push_back(passiveJointDof(1u, 8u, 7u));
    model.defaultQ = {
        0.1f, -0.2f, 0.4f,
        0.0f, 0.0f, 0.0f, 1.0f,
        0.0f, 0.0f,
    };
    model.defaultV.assign(8u, 0.0f);
    return model;
}

double maximumError(
    const std::span<const double> left,
    const std::span<const double> right
) {
    if (left.size() != right.size()) {
        return std::numeric_limits<double>::infinity();
    }
    double result = 0.0;
    for (std::size_t index = 0u; index < left.size(); ++index) {
        result = std::max(result, std::abs(left[index] - right[index]));
    }
    return result;
}

double vectorNorm(const std::array<double, 3>& value) {
    return std::sqrt(
        value[0] * value[0] +
        value[1] * value[1] +
        value[2] * value[2]
    );
}

double vectorDifferenceNorm(
    const std::array<double, 3>& left,
    const std::array<double, 3>& right
) {
    return std::sqrt(
        (left[0] - right[0]) * (left[0] - right[0]) +
        (left[1] - right[1]) * (left[1] - right[1]) +
        (left[2] - right[2]) * (left[2] - right[2])
    );
}

void require(
    const bool condition,
    const std::string& message
) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main() {
    try {
        metalrobo::ArticulatedDynamicsConfig config;
        config.gravity = {0.3, -1.7, 0.5};
        config.applyBodyDamping = false;

        // Free-body consistency, including gravity and an external COM wrench.
        const metalrobo::EngineModel freeModel = makeFreeBodyModel();
        std::vector<double> freeQ{
            0.2, -0.4, 0.7,
            0.0, 0.0, 0.0, 1.0,
        };
        std::vector<double> freeV{
            0.4, -0.2, 0.7,
            1.2, -0.8, 0.5,
        };
        std::vector<double> freeForce(6u, 0.0);
        std::vector<double> freeAcceleration(6u, 0.0);
        std::vector<metalrobo::ArticulatedBodyWrench> freeWrenches(1u);
        freeWrenches[0].force = {1.4, -0.6, 0.8};
        freeWrenches[0].torque = {0.3, -0.5, 0.7};
        const auto freeDiagnostics =
            metalrobo::computeArticulatedForwardDynamics(
                freeModel,
                0u,
                freeQ,
                freeV,
                freeForce,
                freeWrenches,
                freeAcceleration,
                config
            );
        require(freeDiagnostics.succeeded(), "free-body solve failed");
        const double mass = freeModel.bodies[0].massAndInverseMass.x;
        const std::array<double, 3> expectedLinear{
            config.gravity[0] + freeWrenches[0].force[0] / mass,
            config.gravity[1] + freeWrenches[0].force[1] / mass,
            config.gravity[2] + freeWrenches[0].force[2] / mass,
        };
        const std::array<double, 3> omega{
            freeV[3], freeV[4], freeV[5],
        };
        const std::array<double, 3> inertia{
            freeModel.bodies[0].inertiaRow0.x,
            freeModel.bodies[0].inertiaRow1.y,
            freeModel.bodies[0].inertiaRow2.z,
        };
        const std::array<double, 3> angularMomentum{
            inertia[0] * omega[0],
            inertia[1] * omega[1],
            inertia[2] * omega[2],
        };
        const std::array<double, 3> gyroscopic{
            omega[1] * angularMomentum[2] -
                omega[2] * angularMomentum[1],
            omega[2] * angularMomentum[0] -
                omega[0] * angularMomentum[2],
            omega[0] * angularMomentum[1] -
                omega[1] * angularMomentum[0],
        };
        const std::array<double, 3> expectedAngular{
            (freeWrenches[0].torque[0] - gyroscopic[0]) / inertia[0],
            (freeWrenches[0].torque[1] - gyroscopic[1]) / inertia[1],
            (freeWrenches[0].torque[2] - gyroscopic[2]) / inertia[2],
        };
        double freeBodyError = 0.0;
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            freeBodyError = std::max(
                freeBodyError,
                std::abs(freeAcceleration[axis] - expectedLinear[axis])
            );
            freeBodyError = std::max(
                freeBodyError,
                std::abs(
                    freeAcceleration[3u + axis] -
                    expectedAngular[axis]
                )
            );
        }
        require(
            freeBodyError < 2.0e-12,
            "free-body analytical consistency exceeded tolerance"
        );

        // Analytic one-link pendulum.
        constexpr double pendulumLength = 0.73;
        constexpr double theta = 0.37;
        constexpr double appliedTorque = 0.42;
        const metalrobo::EngineModel pendulumModel =
            makePendulumModel(pendulumLength);
        std::vector<double> pendulumQ{theta};
        std::vector<double> pendulumV{0.0};
        std::vector<double> pendulumTau{appliedTorque};
        std::vector<double> pendulumAcceleration(1u, 0.0);
        metalrobo::ArticulatedDynamicsConfig pendulumConfig = config;
        pendulumConfig.gravity = {0.0, -9.81, 0.0};
        const auto pendulumDiagnostics =
            metalrobo::computeArticulatedForwardDynamics(
                pendulumModel,
                0u,
                pendulumQ,
                pendulumV,
                pendulumTau,
                {},
                pendulumAcceleration,
                pendulumConfig
            );
        require(
            pendulumDiagnostics.succeeded(),
            "one-link pendulum solve failed"
        );
        const double storedPendulumMass =
            pendulumModel.bodies[1].massAndInverseMass.x;
        const double storedPendulumIzz =
            pendulumModel.bodies[1].inertiaRow2.z;
        const double storedPendulumLength =
            -pendulumModel.joints[0].childAnchor.x;
        const double expectedPendulumAcceleration =
            (
                appliedTorque -
                storedPendulumMass * 9.81 * storedPendulumLength *
                    std::cos(theta)
            ) /
            (
                storedPendulumIzz +
                storedPendulumMass *
                    storedPendulumLength * storedPendulumLength
            );
        const double pendulumError = std::abs(
            pendulumAcceleration[0] -
            expectedPendulumAcceleration
        );
        require(
            pendulumError < 2.0e-11,
            "one-link analytical acceleration exceeded tolerance"
        );

        // Real source-derived FunctionBased CustomJoint admission. The
        // point at the child joint frame must reproduce the source
        // SpatialTransform translation, velocity H*qdot, and H's linear
        // Jacobian exactly; forward/inverse closure also exercises Hdot in
        // the recursive acceleration and bias paths.
        constexpr double kneeLength = 0.25;
        const metalrobo::EngineModel walkerKneeModel =
            makeFunctionBasedWalkerKneeModel(kneeLength);
        std::string walkerKneeReason;
        require(
            walkerKneeModel.valid(&walkerKneeReason),
            "FunctionBased EngineModel validation failed: " + walkerKneeReason
        );
        constexpr double kneeCoordinate = 0.43;
        constexpr double kneeVelocity = -0.71;
        const std::vector<double> walkerKneeQ{kneeCoordinate};
        const std::vector<double> walkerKneeV{kneeVelocity};
        const auto walkerKneeEvaluation =
            metalrobo::evaluateOpenSimSpatialTransform(
                walkerKneeModel.functionBasedJointPrograms[0u].transform,
                {kneeCoordinate},
                {kneeVelocity}
            );
        require(
            walkerKneeEvaluation.succeeded(),
            "source walker knee evaluation failed"
        );
        std::vector<metalrobo::ArticulatedPointKinematics>
            walkerKneePoint(1u);
        std::vector<double> walkerKneeJacobian(3u, 0.0);
        const std::vector<metalrobo::ArticulatedPointQuery>
            walkerKneeQuery{{
                .bodyIndex = 1u,
                .localPoint = {-kneeLength, 0.0, 0.0},
            }};
        const auto walkerKneePointDiagnostics =
            metalrobo::computeArticulatedPointJacobians(
                walkerKneeModel,
                0u,
                walkerKneeQ,
                walkerKneeV,
                walkerKneeQuery,
                walkerKneePoint,
                walkerKneeJacobian,
                config
            );
        require(
            walkerKneePointDiagnostics.succeeded(),
            "FunctionBased point/Jacobian evaluation failed: status=" +
                std::to_string(
                    static_cast<std::uint32_t>(
                        walkerKneePointDiagnostics.status
                    )
                )
        );
        double walkerKneeKinematicError = 0.0;
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            walkerKneeKinematicError = std::max(
                walkerKneeKinematicError,
                std::abs(
                    walkerKneePoint[0u].position[axis] -
                    walkerKneeEvaluation.translation[axis]
                )
            );
            walkerKneeKinematicError = std::max(
                walkerKneeKinematicError,
                std::abs(
                    walkerKneePoint[0u].linearVelocity[axis] -
                    walkerKneeEvaluation.motionSubspace[0u].linear[axis] *
                        kneeVelocity
                )
            );
            walkerKneeKinematicError = std::max(
                walkerKneeKinematicError,
                std::abs(
                    walkerKneeJacobian[axis] -
                    walkerKneeEvaluation.motionSubspace[0u].linear[axis]
                )
            );
        }
        require(
            walkerKneeKinematicError < 2.0e-12,
            "FunctionBased source kinematics/Jacobian mismatch"
        );
        std::vector<double> walkerKneeForce(1u, 0.0);
        const std::vector<double> walkerKneeAcceleration{0.62};
        const auto walkerKneeInverseDiagnostics =
            metalrobo::computeArticulatedInverseDynamics(
                walkerKneeModel,
                0u,
                walkerKneeQ,
                walkerKneeV,
                walkerKneeAcceleration,
                {},
                walkerKneeForce,
                config
            );
        require(
            walkerKneeInverseDiagnostics.succeeded(),
            "FunctionBased inverse dynamics failed"
        );
        std::vector<double> walkerKneeRecoveredAcceleration(1u, 0.0);
        const auto walkerKneeForwardDiagnostics =
            metalrobo::computeArticulatedForwardDynamics(
                walkerKneeModel,
                0u,
                walkerKneeQ,
                walkerKneeV,
                walkerKneeForce,
                {},
                walkerKneeRecoveredAcceleration,
                config
            );
        const double walkerKneeForwardInverseError = std::abs(
            walkerKneeRecoveredAcceleration[0u] - 0.62
        );
        require(
            walkerKneeForwardDiagnostics.succeeded() &&
                walkerKneeForwardInverseError < 2.0e-11,
            "FunctionBased forward/inverse closure failed"
        );

        // Floating multi-link forward/inverse and dense spatial mass matrix.
        const metalrobo::EngineModel chainModel =
            makeFloatingChainModel();
        std::vector<double> chainQ{
            0.12, -0.16, 0.42,
            0.08, -0.03, 0.05, 0.9951437083,
            0.31, -0.27,
        };
        const double chainQuaternionNorm = std::sqrt(
            chainQ[3] * chainQ[3] +
            chainQ[4] * chainQ[4] +
            chainQ[5] * chainQ[5] +
            chainQ[6] * chainQ[6]
        );
        for (std::size_t index = 3u; index < 7u; ++index) {
            chainQ[index] /= chainQuaternionNorm;
        }
        std::vector<double> chainV{
            0.2, -0.1, 0.15,
            0.5, -0.35, 0.22,
            0.7, -0.45,
        };
        std::vector<double> requestedAcceleration{
            -0.3, 0.25, 0.4,
            0.15, -0.2, 0.33,
            -0.6, 0.8,
        };
        std::vector<metalrobo::ArticulatedBodyWrench> chainWrenches(3u);
        chainWrenches[1].force = {0.6, -0.4, 0.2};
        chainWrenches[1].torque = {0.03, 0.05, -0.02};
        chainWrenches[2].force = {-0.2, 0.1, 0.35};
        chainWrenches[2].torque = {-0.04, 0.02, 0.01};
        std::vector<double> recoveredForce(8u, 0.0);
        const auto inverseDiagnostics =
            metalrobo::computeArticulatedInverseDynamics(
                chainModel,
                0u,
                chainQ,
                chainV,
                requestedAcceleration,
                chainWrenches,
                recoveredForce,
                config
            );
        require(
            inverseDiagnostics.succeeded(),
            "multi-link inverse dynamics failed"
        );
        std::vector<double> recoveredAcceleration(8u, 0.0);
        const auto forwardDiagnostics =
            metalrobo::computeArticulatedForwardDynamics(
                chainModel,
                0u,
                chainQ,
                chainV,
                recoveredForce,
                chainWrenches,
                recoveredAcceleration,
                config
            );
        require(
            forwardDiagnostics.succeeded(),
            "multi-link forward dynamics failed"
        );
        const double forwardInverseError =
            maximumError(requestedAcceleration, recoveredAcceleration);
        require(
            forwardInverseError < 2.0e-11,
            "multi-link forward/inverse consistency exceeded tolerance"
        );

        std::vector<double> chainMassMatrix(64u, 0.0);
        const auto chainMassDiagnostics =
            metalrobo::computeArticulatedMassMatrix(
                chainModel,
                0u,
                chainQ,
                chainMassMatrix,
                config
            );
        require(
            chainMassDiagnostics.succeeded(),
            "multi-link mass matrix failed"
        );
        double massSymmetryError = 0.0;
        for (std::size_t row = 0u; row < 8u; ++row) {
            for (std::size_t column = 0u; column < 8u; ++column) {
                massSymmetryError = std::max(
                    massSymmetryError,
                    std::abs(
                        chainMassMatrix[row * 8u + column] -
                        chainMassMatrix[column * 8u + row]
                    )
                );
            }
        }
        require(
            massSymmetryError < 1.0e-14 &&
                chainMassDiagnostics.minimumCholeskyPivot > 0.0,
            "multi-link mass matrix symmetry/SPD gate failed"
        );

        // Finite/quaternion/limit rejection must not mutate output.
        std::vector<double> lower(chainQ.size(), -std::numeric_limits<double>::infinity());
        std::vector<double> upper(chainQ.size(), std::numeric_limits<double>::infinity());
        lower[7] = -0.5;
        upper[7] = 0.5;
        lower[8] = -0.5;
        upper[8] = 0.5;
        metalrobo::ArticulatedDynamicsConfig limitedConfig = config;
        limitedConfig.limits.lower = lower;
        limitedConfig.limits.upper = upper;
        std::vector<double> invalidQ = chainQ;
        invalidQ[7] = 0.8;
        std::vector<double> transactionalOutput(8u, 91.0);
        const auto limitDiagnostics =
            metalrobo::computeArticulatedForwardDynamics(
                chainModel,
                0u,
                invalidQ,
                chainV,
                recoveredForce,
                chainWrenches,
                transactionalOutput,
                limitedConfig
            );
        require(
            limitDiagnostics.status ==
                metalrobo::ArticulatedDynamicsStatus::
                    jointLimitViolation &&
                std::ranges::all_of(
                    transactionalOutput,
                    [](const double value) {
                        return value == 91.0;
                    }
                ),
            "joint limit rejection was not transactional"
        );
        invalidQ = chainQ;
        invalidQ[6] = 0.4;
        const auto quaternionDiagnostics =
            metalrobo::computeArticulatedForwardDynamics(
                chainModel,
                0u,
                invalidQ,
                chainV,
                recoveredForce,
                chainWrenches,
                transactionalOutput,
                config
            );
        require(
            quaternionDiagnostics.status ==
                metalrobo::ArticulatedDynamicsStatus::
                    invalidQuaternion,
            "invalid quaternion was not rejected"
        );
        std::vector<double> nonfiniteV = chainV;
        nonfiniteV[2] = std::numeric_limits<double>::quiet_NaN();
        const auto nonfiniteDiagnostics =
            metalrobo::computeArticulatedForwardDynamics(
                chainModel,
                0u,
                chainQ,
                nonfiniteV,
                recoveredForce,
                chainWrenches,
                transactionalOutput,
                config
            );
        require(
            nonfiniteDiagnostics.status ==
                metalrobo::ArticulatedDynamicsStatus::
                    nonfiniteInput &&
                std::ranges::all_of(
                    transactionalOutput,
                    [](const double value) {
                        return value == 91.0;
                    }
                ),
            "non-finite rejection was not transactional"
        );

        // Actual pinned G1 topology and COM-anchor mapping.
        const metalrobo::EngineModel g1 =
            metalrobo::makeUnitreeG1EngineModel();
        bool g1ComAnchors = true;
        for (const MRJointDescriptorGPU& joint : g1.joints) {
            const MRBodyPropertiesGPU& child = g1.bodies[joint.childBody];
            g1ComAnchors =
                g1ComAnchors &&
                std::abs(
                    static_cast<double>(joint.childAnchor.x) +
                    child.centerOfMass.x
                ) < 2.0e-7 &&
                std::abs(
                    static_cast<double>(joint.childAnchor.y) +
                    child.centerOfMass.y
                ) < 2.0e-7 &&
                std::abs(
                    static_cast<double>(joint.childAnchor.z) +
                    child.centerOfMass.z
                ) < 2.0e-7;
        }
        require(g1ComAnchors, "G1 child anchors are not COM shifted");
        std::vector<double> g1Q(
            g1.defaultQ.begin(),
            g1.defaultQ.end()
        );
        std::vector<double> g1V(35u, 0.0);
        std::vector<double> g1Acceleration(35u, 0.0);
        for (std::size_t index = 0u; index < 35u; ++index) {
            g1V[index] =
                0.03 * std::sin(0.31 * static_cast<double>(index + 1u));
            g1Acceleration[index] =
                0.09 * std::cos(0.27 * static_cast<double>(index + 1u));
        }
        std::vector<double> g1Tau(35u, 0.0);
        metalrobo::ArticulatedDynamicsConfig g1Config = config;
        g1Config.gravity = {0.0, 0.0, -9.81};
        const auto g1InverseDiagnostics =
            metalrobo::computeArticulatedInverseDynamics(
                g1,
                0u,
                g1Q,
                g1V,
                g1Acceleration,
                {},
                g1Tau,
                g1Config
            );
        require(
            g1InverseDiagnostics.succeeded(),
            "actual G1 topology inverse dynamics failed"
        );
        std::vector<double> g1RecoveredAcceleration(35u, 0.0);
        const auto g1ForwardDiagnostics =
            metalrobo::computeArticulatedForwardDynamics(
                g1,
                0u,
                g1Q,
                g1V,
                g1Tau,
                {},
                g1RecoveredAcceleration,
                g1Config
            );
        require(
            g1ForwardDiagnostics.succeeded(),
            "actual G1 topology forward dynamics failed"
        );
        const double g1ForwardInverseError =
            maximumError(g1Acceleration, g1RecoveredAcceleration);
        require(
            g1ForwardInverseError < 5.0e-9,
            "G1 forward/inverse consistency exceeded tolerance"
        );

        // Deterministic-random acceptance: armature must be the exact same
        // generalized inertia in CRBA, RNEA, invariants, and every state.
        metalrobo::EngineModel zeroArmatureG1 = g1;
        for (MRDofPropertiesGPU& dof : zeroArmatureG1.dofs) {
            dof.drive.z = 0.0f;
        }
        std::mt19937_64 armatureGenerator{0x4d4554414c524f42ull};
        std::uniform_real_distribution<double> signedUnit(-1.0, 1.0);
        std::uniform_real_distribution<double> positiveUnit(0.0, 1.0);
        double armatureMassError = 0.0;
        double armatureInverseError = 0.0;
        double armatureEnergyError = 0.0;
        constexpr std::size_t armatureSamples = 12u;
        for (std::size_t sample = 0u;
             sample < armatureSamples;
             ++sample) {
            metalrobo::EngineModel armedG1 = zeroArmatureG1;
            for (std::size_t dof = 6u;
                 dof < armedG1.dofs.size();
                 ++dof) {
                armedG1.dofs[dof].drive.z = static_cast<float>(
                    0.001 + 0.249 * positiveUnit(armatureGenerator)
                );
            }

            std::vector<double> sampleQ(
                armedG1.defaultQ.begin(),
                armedG1.defaultQ.end()
            );
            sampleQ[0] = 0.2 * signedUnit(armatureGenerator);
            sampleQ[1] = 0.2 * signedUnit(armatureGenerator);
            sampleQ[2] += 0.1 * signedUnit(armatureGenerator);
            std::array<double, 4> rootQuaternion{
                signedUnit(armatureGenerator),
                signedUnit(armatureGenerator),
                signedUnit(armatureGenerator),
                signedUnit(armatureGenerator),
            };
            double quaternionMagnitude = 0.0;
            for (const double value : rootQuaternion) {
                quaternionMagnitude += value * value;
            }
            quaternionMagnitude = std::sqrt(quaternionMagnitude);
            for (std::size_t axis = 0u; axis < 4u; ++axis) {
                sampleQ[3u + axis] =
                    rootQuaternion[axis] / quaternionMagnitude;
            }
            for (std::size_t dof = 6u;
                 dof < armedG1.dofs.size();
                 ++dof) {
                const MRDofPropertiesGPU& properties =
                    armedG1.dofs[dof];
                const double midpoint =
                    0.5 * (
                        static_cast<double>(properties.limits.x) +
                        properties.limits.y
                    );
                const double halfRange =
                    0.5 * (
                        static_cast<double>(properties.limits.y) -
                        properties.limits.x
                    );
                sampleQ[properties.qIndex] =
                    midpoint +
                    0.7 * halfRange *
                        signedUnit(armatureGenerator);
            }
            std::vector<double> sampleV(35u, 0.0);
            std::vector<double> sampleAcceleration(35u, 0.0);
            for (std::size_t dof = 0u; dof < 35u; ++dof) {
                sampleV[dof] =
                    1.7 * signedUnit(armatureGenerator);
                sampleAcceleration[dof] =
                    3.1 * signedUnit(armatureGenerator);
            }

            std::vector<double> massWithArmature(35u * 35u, 0.0);
            std::vector<double> massWithoutArmature(35u * 35u, 0.0);
            const auto massWithDiagnostics =
                metalrobo::computeArticulatedMassMatrix(
                    armedG1,
                    0u,
                    sampleQ,
                    massWithArmature,
                    g1Config
                );
            const auto massWithoutDiagnostics =
                metalrobo::computeArticulatedMassMatrix(
                    zeroArmatureG1,
                    0u,
                    sampleQ,
                    massWithoutArmature,
                    g1Config
                );
            require(
                massWithDiagnostics.succeeded() &&
                    massWithoutDiagnostics.succeeded(),
                "randomized armature mass-matrix evaluation failed"
            );
            for (std::size_t row = 0u; row < 35u; ++row) {
                for (std::size_t column = 0u;
                     column < 35u;
                     ++column) {
                    const double expected =
                        row == column
                            ? static_cast<double>(
                                  armedG1.dofs[row].drive.z
                              )
                            : 0.0;
                    armatureMassError = std::max(
                        armatureMassError,
                        std::abs(
                            massWithArmature[row * 35u + column] -
                            massWithoutArmature[row * 35u + column] -
                            expected
                        )
                    );
                }
            }

            std::vector<double> tauWithArmature(35u, 0.0);
            std::vector<double> tauWithoutArmature(35u, 0.0);
            const auto inverseWithDiagnostics =
                metalrobo::computeArticulatedInverseDynamics(
                    armedG1,
                    0u,
                    sampleQ,
                    sampleV,
                    sampleAcceleration,
                    {},
                    tauWithArmature,
                    g1Config
                );
            const auto inverseWithoutDiagnostics =
                metalrobo::computeArticulatedInverseDynamics(
                    zeroArmatureG1,
                    0u,
                    sampleQ,
                    sampleV,
                    sampleAcceleration,
                    {},
                    tauWithoutArmature,
                    g1Config
                );
            require(
                inverseWithDiagnostics.succeeded() &&
                    inverseWithoutDiagnostics.succeeded(),
                "randomized armature inverse-dynamics evaluation failed"
            );
            for (std::size_t dof = 0u; dof < 35u; ++dof) {
                armatureInverseError = std::max(
                    armatureInverseError,
                    std::abs(
                        tauWithArmature[dof] -
                        tauWithoutArmature[dof] -
                        static_cast<double>(
                            armedG1.dofs[dof].drive.z
                        ) *
                        sampleAcceleration[dof]
                    )
                );
            }

            metalrobo::ArticulatedInvariants invariantsWithArmature;
            metalrobo::ArticulatedInvariants invariantsWithoutArmature;
            const auto energyWithDiagnostics =
                metalrobo::computeArticulatedInvariants(
                    armedG1,
                    0u,
                    sampleQ,
                    sampleV,
                    invariantsWithArmature,
                    g1Config
                );
            const auto energyWithoutDiagnostics =
                metalrobo::computeArticulatedInvariants(
                    zeroArmatureG1,
                    0u,
                    sampleQ,
                    sampleV,
                    invariantsWithoutArmature,
                    g1Config
                );
            require(
                energyWithDiagnostics.succeeded() &&
                    energyWithoutDiagnostics.succeeded(),
                "randomized armature invariant evaluation failed"
            );
            double expectedEnergyDelta = 0.0;
            for (std::size_t dof = 0u; dof < 35u; ++dof) {
                expectedEnergyDelta +=
                    0.5 *
                    static_cast<double>(
                        armedG1.dofs[dof].drive.z
                    ) *
                    sampleV[dof] * sampleV[dof];
            }
            armatureEnergyError = std::max(
                armatureEnergyError,
                std::abs(
                    invariantsWithArmature.kineticEnergy -
                    invariantsWithoutArmature.kineticEnergy -
                    expectedEnergyDelta
                )
            );
        }
        require(
            armatureMassError < 5.0e-13 &&
                armatureInverseError < 5.0e-13 &&
                armatureEnergyError < 5.0e-13,
            "randomized armature operator identities exceeded tolerance"
        );

        // Unforced floating-chain implicit midpoint conservation.
        std::vector<double> driftQ = chainQ;
        std::vector<double> driftV = chainV;
        std::vector<double> zeroForce(8u, 0.0);
        metalrobo::ArticulatedDynamicsConfig driftConfig = config;
        driftConfig.gravity = {0.0, 0.0, 0.0};
        driftConfig.timestep = 5.0e-4;
        driftConfig.integrator =
            metalrobo::ArticulatedIntegrator::implicitMidpoint;
        driftConfig.nonlinearIterations = 8u;
        driftConfig.nonlinearTolerance = 2.0e-11;
        metalrobo::ArticulatedInvariants initialInvariants;
        auto invariantDiagnostics =
            metalrobo::computeArticulatedInvariants(
                chainModel,
                0u,
                driftQ,
                driftV,
                initialInvariants,
                driftConfig
            );
        require(
            invariantDiagnostics.succeeded(),
            "initial invariant evaluation failed"
        );
        std::uint32_t maximumMidpointIterations = 0u;
        double maximumMidpointResidual = 0.0;
        constexpr std::uint32_t driftSteps = 400u;
        for (std::uint32_t step = 0u; step < driftSteps; ++step) {
            const auto stepDiagnostics =
                metalrobo::integrateArticulatedState(
                    chainModel,
                    0u,
                    driftQ,
                    driftV,
                    zeroForce,
                    {},
                    driftConfig
                );
            require(
                stepDiagnostics.succeeded(),
                "implicit midpoint drift step failed"
            );
            maximumMidpointIterations = std::max(
                maximumMidpointIterations,
                stepDiagnostics.nonlinearIterations
            );
            maximumMidpointResidual = std::max(
                maximumMidpointResidual,
                stepDiagnostics.nonlinearResidual
            );
        }
        metalrobo::ArticulatedInvariants finalInvariants;
        invariantDiagnostics =
            metalrobo::computeArticulatedInvariants(
                chainModel,
                0u,
                driftQ,
                driftV,
                finalInvariants,
                driftConfig
            );
        require(
            invariantDiagnostics.succeeded(),
            "final invariant evaluation failed"
        );
        const double relativeEnergyDrift =
            std::abs(
                finalInvariants.totalEnergy -
                initialInvariants.totalEnergy
            ) /
            std::max(
                std::abs(initialInvariants.totalEnergy),
                1.0e-12
            );
        const double relativeLinearMomentumDrift =
            vectorDifferenceNorm(
                finalInvariants.linearMomentum,
                initialInvariants.linearMomentum
            ) /
            std::max(
                vectorNorm(initialInvariants.linearMomentum),
                1.0e-12
            );
        const double relativeAngularMomentumDrift =
            vectorDifferenceNorm(
                finalInvariants.angularMomentum,
                initialInvariants.angularMomentum
            ) /
            std::max(
                vectorNorm(initialInvariants.angularMomentum),
                1.0e-12
            );
        const double finalQuaternionNorm = std::sqrt(
            driftQ[3] * driftQ[3] +
            driftQ[4] * driftQ[4] +
            driftQ[5] * driftQ[5] +
            driftQ[6] * driftQ[6]
        );
        require(
            relativeEnergyDrift < 2.0e-7 &&
                relativeLinearMomentumDrift < 2.0e-8 &&
                relativeAngularMomentumDrift < 2.0e-7 &&
                std::abs(finalQuaternionNorm - 1.0) < 2.0e-14,
            "floating-chain conservation gate exceeded"
        );

        std::cout
            << std::scientific << std::setprecision(6)
            << "articulated=cpu_fp64_analytic_jacobian_rnea"
            << " free_body_error=" << freeBodyError
            << " pendulum_error=" << pendulumError
            << " function_based_walker_knee_kinematic_error="
            << walkerKneeKinematicError
            << " function_based_walker_knee_forward_inverse_error="
            << walkerKneeForwardInverseError
            << " forward_inverse_error=" << forwardInverseError
            << " mass_symmetry_error=" << massSymmetryError
            << " min_mass_pivot="
            << chainMassDiagnostics.minimumCholeskyPivot
            << " g1_com_anchors=yes"
            << " g1_nq=" << g1.world.nq
            << " g1_nv=" << g1.world.nv
            << " g1_forward_inverse_error=" << g1ForwardInverseError
            << " armature_mass_error=" << armatureMassError
            << " armature_inverse_error=" << armatureInverseError
            << " armature_energy_error=" << armatureEnergyError
            << " armature_samples=" << armatureSamples
            << " energy_drift=" << relativeEnergyDrift
            << " linear_momentum_drift="
            << relativeLinearMomentumDrift
            << " angular_momentum_drift="
            << relativeAngularMomentumDrift
            << " q_norm_error="
            << std::abs(finalQuaternionNorm - 1.0)
            << " midpoint_iterations=" << maximumMidpointIterations
            << " midpoint_residual=" << maximumMidpointResidual
            << " limit_rejection=transactional"
            << " nonfinite_rejection=transactional"
            << " finite=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "metalrobo_articulated_dynamics_probe: "
            << error.what() << '\n';
        return 1;
    }
}
