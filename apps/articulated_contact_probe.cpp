#include "metalrobo/ArticulatedContact.hpp"
#include "metalrobo/G1.hpp"
#include "metalrobo/QualityContactSolver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
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

MRBodyPropertiesGPU body(
    const std::uint32_t parent,
    const std::uint32_t inboundJoint,
    const double mass,
    const std::array<double, 3> inertia
) {
    MRBodyPropertiesGPU result{};
    result.articulationIndex = 0u;
    result.parentBody = parent;
    result.inboundJoint = inboundJoint;
    result.motionType = MR_MOTION_DYNAMIC;
    result.massAndInverseMass = f4(mass, 1.0 / mass, 0.0, 0.0);
    result.inertiaRow0 = f4(inertia[0], 0.0, 0.0);
    result.inertiaRow1 = f4(0.0, inertia[1], 0.0);
    result.inertiaRow2 = f4(0.0, 0.0, inertia[2]);
    result.inverseInertiaRow0 =
        f4(1.0 / inertia[0], 0.0, 0.0);
    result.inverseInertiaRow1 =
        f4(0.0, 1.0 / inertia[1], 0.0);
    result.inverseInertiaRow2 =
        f4(0.0, 0.0, 1.0 / inertia[2]);
    result.dampingAndSpeedLimits =
        f4(0.0, 0.0, 1.0e6, 1.0e6);
    return result;
}

metalrobo::EngineModel makeFreeBodyModel() {
    metalrobo::EngineModel model;
    model.name = "analytic_contact_free_body";
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
        {0.7, 1.1, 1.6}
    ));
    model.defaultQ = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    model.defaultV.assign(6u, 0.0f);
    return model;
}

metalrobo::EngineModel makePendulumModel(
    const double length,
    const double mass,
    const double inertiaZ
) {
    metalrobo::EngineModel model;
    model.name = "analytic_contact_pendulum";
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
        1.0,
        {0.2, 0.3, 0.4}
    ));
    model.bodies.push_back(body(
        0u,
        0u,
        mass,
        {0.08, 0.11, inertiaZ}
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
    joint.childAnchor = f4(-length, 0.0, 0.0);
    joint.parentRotation = f4(0.0, 0.0, 0.0, 1.0);
    joint.childRotation = f4(0.0, 0.0, 0.0, 1.0);
    model.joints.push_back(joint);
    model.dofs.push_back(passiveJointDof(0u, 0u, 0u));
    model.defaultQ = {0.0f};
    model.defaultV = {0.0f};
    return model;
}

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
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
        result = std::max(
            result,
            std::abs(left[index] - right[index])
        );
    }
    return result;
}

double dot(
    const std::array<double, 3>& left,
    const std::array<double, 3>& right
) {
    return
        left[0] * right[0] +
        left[1] * right[1] +
        left[2] * right[2];
}

std::array<double, 3> cross(
    const std::array<double, 3>& left,
    const std::array<double, 3>& right
) {
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}

std::array<double, 3> plus(
    const std::array<double, 3>& left,
    const std::array<double, 3>& right
) {
    return {
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
    };
}

std::array<double, 3> scale(
    const std::array<double, 3>& value,
    const double multiplier
) {
    return {
        multiplier * value[0],
        multiplier * value[1],
        multiplier * value[2],
    };
}

std::array<double, 3> rotate(
    const std::array<double, 4>& q,
    const std::array<double, 3>& value
) {
    const std::array<double, 3> vectorPart{q[0], q[1], q[2]};
    const std::array<double, 3> twiceCross =
        scale(cross(vectorPart, value), 2.0);
    return plus(
        value,
        plus(
            scale(twiceCross, q[3]),
            cross(vectorPart, twiceCross)
        )
    );
}

double matrixBilinear(
    const std::span<const double> matrix,
    const std::size_t dimension,
    const std::span<const double> left,
    const std::span<const double> right
) {
    double result = 0.0;
    for (std::size_t row = 0u; row < dimension; ++row) {
        for (std::size_t column = 0u;
             column < dimension;
             ++column) {
            result +=
                left[row] *
                matrix[row * dimension + column] *
                right[column];
        }
    }
    return result;
}

} // namespace

int main() {
    try {
        metalrobo::ArticulatedDynamicsConfig config;
        config.gravity = {0.0, 0.0, 0.0};
        config.applyBodyDamping = false;

        // Floating rigid body: prove the full off-COM inverse effective mass,
        // not only a center-point scalar.
        const metalrobo::EngineModel freeModel = makeFreeBodyModel();
        const std::vector<double> freeQ{
            0.2, -0.4, 0.7,
            0.0, 0.0, 0.0, 1.0,
        };
        const std::vector<double> freeV{
            0.3, -0.2, 0.1,
            -0.4, 0.5, 0.7,
        };
        const std::array<double, 3> radius{0.4, -0.2, 0.3};
        const metalrobo::ArticulatedPointQuery freePoint{
            0u,
            radius,
        };
        metalrobo::ArticulatedPointKinematics freePointKinematics;
        std::array<double, 18> freePointJacobian{};
        auto dynamicsDiagnostics =
            metalrobo::computeArticulatedPointJacobians(
                freeModel,
                0u,
                freeQ,
                freeV,
                std::span(&freePoint, 1u),
                std::span(&freePointKinematics, 1u),
                freePointJacobian,
                config
            );
        require(
            dynamicsDiagnostics.succeeded(),
            "free-body point Jacobian failed"
        );
        double freeVelocityIdentityError = 0.0;
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            double value = 0.0;
            for (std::size_t dof = 0u; dof < 6u; ++dof) {
                value +=
                    freePointJacobian[axis * 6u + dof] *
                    freeV[dof];
            }
            freeVelocityIdentityError = std::max(
                freeVelocityIdentityError,
                std::abs(
                    value -
                    freePointKinematics.linearVelocity[axis]
                )
            );
        }
        require(
            freeVelocityIdentityError < 2.0e-15,
            "free-body Jv point-velocity identity failed"
        );

        metalrobo::ArticulatedContact freeContact;
        freeContact.bodyA = 0u;
        freeContact.localPointA = radius;
        freeContact.localPointB =
            freePointKinematics.position;
        freeContact.normal = {0.0, 0.0, -1.0};
        freeContact.tangentU = {1.0, 0.0, 0.0};
        freeContact.tangentV = {0.0, -1.0, 0.0};
        freeContact.regularization = {
            1.0e-12,
            1.0e-12,
            1.0e-12,
        };
        freeContact.targetVelocity = {0.1, -0.2, 0.3};
        freeContact.warmImpulse = {0.4, 0.1, -0.1};
        metalrobo::ArticulatedContactProblem freeProblem;
        auto contactDiagnostics =
            metalrobo::buildArticulatedContactProblem(
                freeModel,
                0u,
                freeQ,
                freeV,
                std::span(&freeContact, 1u),
                freeProblem,
                config,
                true
            );
        require(
            contactDiagnostics.succeeded() &&
                contactDiagnostics
                    .maximumDenseInverseAdapterResidual < 1.0e-12 &&
                contactDiagnostics
                    .maximumDelassusAsymmetry < 1.0e-12 &&
                freeProblem.conic.contacts[0].targetVelocity ==
                    freeContact.targetVelocity &&
                freeProblem.conic.contacts[0].regularization ==
                    freeContact.regularization &&
                freeProblem.conic.contacts[0].warmImpulse ==
                    freeContact.warmImpulse,
            "free-body contact construction failed"
        );

        // A validated evaluated-IR frame may carry small FP32 norm error.
        // The generalized adapter must preserve those fingerprinted
        // coordinates verbatim instead of silently re-orthonormalizing them.
        metalrobo::ArticulatedContact fingerprintedFrameContact =
            freeContact;
        constexpr double frameScale = 1.0001;
        fingerprintedFrameContact.normal =
            {0.0, 0.0, -frameScale};
        fingerprintedFrameContact.tangentV =
            {0.0, -frameScale, 0.0};
        metalrobo::ArticulatedContactProblem fingerprintedFrameProblem;
        contactDiagnostics =
            metalrobo::buildArticulatedContactProblem(
                freeModel,
                0u,
                freeQ,
                freeV,
                std::span(&fingerprintedFrameContact, 1u),
                fingerprintedFrameProblem,
                config
            );
        require(
            contactDiagnostics.succeeded() &&
                std::abs(
                    fingerprintedFrameProblem.contactJacobian[2] -
                    frameScale * freeProblem.contactJacobian[2]
                ) < 2.0e-15 &&
                std::abs(
                    fingerprintedFrameProblem.contactJacobian[
                        2u * freeModel.articulations[0].nv + 1u
                    ] -
                    frameScale * freeProblem.contactJacobian[
                        2u * freeModel.articulations[0].nv + 1u
                    ]
                ) < 2.0e-15,
            "fingerprinted contact frame was reinterpreted"
        );

        const std::array<std::array<double, 3>, 3> frame{{
            {0.0, 0.0, -1.0},
            {1.0, 0.0, 0.0},
            {0.0, -1.0, 0.0},
        }};
        const std::array<double, 3> inverseInertia{
            1.0 / static_cast<double>(
                freeModel.bodies[0].inertiaRow0.x
            ),
            1.0 / static_cast<double>(
                freeModel.bodies[0].inertiaRow1.y
            ),
            1.0 / static_cast<double>(
                freeModel.bodies[0].inertiaRow2.z
            ),
        };
        double freeDelassusError = 0.0;
        for (std::size_t column = 0u; column < 3u; ++column) {
            const std::array<double, 3> angularImpulse =
                cross(radius, frame[column]);
            const std::array<double, 3> angularVelocity{
                inverseInertia[0] * angularImpulse[0],
                inverseInertia[1] * angularImpulse[1],
                inverseInertia[2] * angularImpulse[2],
            };
            const std::array<double, 3> pointResponse = plus(
                scale(frame[column], 1.0 / 2.5),
                cross(angularVelocity, radius)
            );
            for (std::size_t row = 0u; row < 3u; ++row) {
                freeDelassusError = std::max(
                    freeDelassusError,
                    std::abs(
                        freeProblem.delassus[row * 3u + column] -
                        dot(frame[row], pointResponse)
                    )
                );
            }
        }
        require(
            freeDelassusError < 2.0e-15,
            "off-COM free-body Delassus is not analytic result"
        );

        const std::array<double, 3> testImpulse{0.8, -0.3, 0.2};
        std::array<double, 6> freeDeltaVelocity{};
        std::array<double, 3> freeDeltaContact{};
        contactDiagnostics =
            metalrobo::computeArticulatedContactImpulseResponse(
                freeProblem,
                testImpulse,
                freeDeltaVelocity,
                freeDeltaContact
            );
        require(
            contactDiagnostics.succeeded() &&
                contactDiagnostics.maximumActionResidual < 2.0e-15,
            "free-body inverse mass action failed"
        );

        std::array<double, 3> appliedForce{};
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            appliedForce = plus(
                appliedForce,
                scale(frame[axis], -testImpulse[axis])
            );
        }
        const std::array<double, 3> appliedTorque =
            cross(radius, appliedForce);
        const std::array<double, 6> expectedFreeDelta{
            appliedForce[0] / 2.5,
            appliedForce[1] / 2.5,
            appliedForce[2] / 2.5,
            inverseInertia[0] * appliedTorque[0],
            inverseInertia[1] * appliedTorque[1],
            inverseInertia[2] * appliedTorque[2],
        };
        const double freeImpulseError = maximumError(
            freeDeltaVelocity,
            expectedFreeDelta
        );
        require(
            freeImpulseError < 2.0e-15,
            "free-body impulse generalized response is incorrect"
        );

        // Fixed-root pendulum: prove the analytic tangent Jacobian and scalar
        // Delassus L^2 / (Izz + m L^2).
        constexpr double pendulumLength = 0.73;
        constexpr double pendulumMass = 1.9;
        constexpr double pendulumInertia = 0.14;
        constexpr double pendulumAngle = 0.37;
        constexpr double pendulumRate = -0.62;
        const metalrobo::EngineModel pendulum = makePendulumModel(
            pendulumLength,
            pendulumMass,
            pendulumInertia
        );
        const std::array<double, 1> pendulumQ{pendulumAngle};
        const std::array<double, 1> pendulumV{pendulumRate};
        const metalrobo::ArticulatedPointQuery pendulumPoint{
            1u,
            {0.0, 0.0, 0.0},
        };
        metalrobo::ArticulatedPointKinematics pendulumPointKinematics;
        std::array<double, 3> pendulumJacobian{};
        dynamicsDiagnostics =
            metalrobo::computeArticulatedPointJacobians(
                pendulum,
                0u,
                pendulumQ,
                pendulumV,
                std::span(&pendulumPoint, 1u),
                std::span(&pendulumPointKinematics, 1u),
                pendulumJacobian,
                config
            );
        require(
            dynamicsDiagnostics.succeeded(),
            "pendulum point Jacobian failed"
        );
        const std::array<double, 3> expectedPendulumJacobian{
            -pendulumLength * std::sin(pendulumAngle),
            pendulumLength * std::cos(pendulumAngle),
            0.0,
        };
        const double pendulumJacobianError = maximumError(
            pendulumJacobian,
            expectedPendulumJacobian
        );
        require(
            pendulumJacobianError < 2.0e-8,
            "pendulum analytic point Jacobian failed"
        );

        metalrobo::ArticulatedContact pendulumContact;
        pendulumContact.bodyA = 1u;
        pendulumContact.localPointB =
            pendulumPointKinematics.position;
        pendulumContact.normal = {
            -std::sin(pendulumAngle),
            std::cos(pendulumAngle),
            0.0,
        };
        pendulumContact.tangentU = {0.0, 0.0, 1.0};
        pendulumContact.tangentV = {
            std::cos(pendulumAngle),
            std::sin(pendulumAngle),
            0.0,
        };
        pendulumContact.regularization = {
            1.0e-12,
            1.0e-12,
            1.0e-12,
        };
        metalrobo::ArticulatedContactProblem pendulumProblem;
        contactDiagnostics =
            metalrobo::buildArticulatedContactProblem(
                pendulum,
                0u,
                pendulumQ,
                pendulumV,
                std::span(&pendulumContact, 1u),
                pendulumProblem,
                config,
                true
            );
        require(
            contactDiagnostics.succeeded(),
            "pendulum contact construction failed"
        );
        const double expectedPendulumDelassus =
            pendulumLength * pendulumLength /
            (
                pendulumInertia +
                pendulumMass *
                    pendulumLength * pendulumLength
            );
        const double pendulumDelassusError = std::abs(
            pendulumProblem.delassus[0] -
            expectedPendulumDelassus
        );
        require(
            pendulumDelassusError < 2.0e-8 &&
                std::abs(pendulumProblem.delassus[4]) < 1.0e-15 &&
                std::abs(pendulumProblem.delassus[8]) < 1.0e-15,
            "pendulum analytic Delassus failed"
        );

        // Armature must propagate through the retained mass factor into both
        // Delassus and the matrix-free impulse response.
        metalrobo::EngineModel armoredPendulum = pendulum;
        armoredPendulum.dofs[0].drive.z = 0.37f;
        metalrobo::ArticulatedContactProblem armoredPendulumProblem;
        contactDiagnostics =
            metalrobo::buildArticulatedContactProblem(
                armoredPendulum,
                0u,
                pendulumQ,
                pendulumV,
                std::span(&pendulumContact, 1u),
                armoredPendulumProblem,
                config,
                true
            );
        require(
            contactDiagnostics.succeeded(),
            "armature contact construction failed"
        );
        const double armature =
            armoredPendulum.dofs[0].drive.z;
        const double barePendulumInertia =
            pendulumInertia +
            pendulumMass *
                pendulumLength * pendulumLength;
        const double expectedArmoredDelassus =
            pendulumLength * pendulumLength /
            (barePendulumInertia + armature);
        const double armatureDelassusError = std::abs(
            armoredPendulumProblem.delassus[0] -
            expectedArmoredDelassus
        );
        const std::array<double, 3> normalImpulse{1.0, 0.0, 0.0};
        std::array<double, 1> barePendulumVelocityDelta{};
        std::array<double, 1> armoredPendulumVelocityDelta{};
        std::array<double, 3> barePendulumContactDelta{};
        std::array<double, 3> armoredPendulumContactDelta{};
        const auto bareImpulseDiagnostics =
            metalrobo::computeArticulatedContactImpulseResponse(
                pendulumProblem,
                normalImpulse,
                barePendulumVelocityDelta,
                barePendulumContactDelta
            );
        const auto armoredImpulseDiagnostics =
            metalrobo::computeArticulatedContactImpulseResponse(
                armoredPendulumProblem,
                normalImpulse,
                armoredPendulumVelocityDelta,
                armoredPendulumContactDelta
            );
        const double armatureImpulseRatio =
            armoredPendulumVelocityDelta[0] /
            barePendulumVelocityDelta[0];
        const double expectedArmatureImpulseRatio =
            barePendulumInertia /
            (barePendulumInertia + armature);
        const double armatureImpulseRatioError = std::abs(
            armatureImpulseRatio -
            expectedArmatureImpulseRatio
        );
        require(
            bareImpulseDiagnostics.succeeded() &&
                armoredImpulseDiagnostics.succeeded() &&
                armatureDelassusError < 2.0e-8 &&
                armatureImpulseRatioError < 2.0e-8 &&
                armoredPendulumProblem.delassus[0] <
                    pendulumProblem.delassus[0],
            "armature did not change contact effective mass consistently"
        );

        // Actual pinned G1: prove body-pose/point consistency, floating-base
        // columns, tree sparsity, Jv, two-foot Delassus reciprocity, exact
        // inverse-mass action, and a converged solver-to-velocity update.
        const metalrobo::EngineModel g1 =
            metalrobo::makeUnitreeG1EngineModel();
        const auto& metadata = metalrobo::unitreeG1Metadata();
        std::vector<double> g1Q(
            g1.defaultQ.begin(),
            g1.defaultQ.end()
        );
        std::vector<double> g1V(
            g1.articulations[0].nv,
            0.0
        );
        for (std::size_t dof = 0u; dof < g1V.size(); ++dof) {
            g1V[dof] =
                0.11 * std::sin(0.31 * static_cast<double>(dof + 1u));
        }
        std::array<metalrobo::ArticulatedPointQuery, 2> footQueries{};
        for (std::size_t foot = 0u; foot < 2u; ++foot) {
            footQueries[foot].bodyIndex =
                metadata.feet[foot].bodyIndex;
            footQueries[foot].localPoint = {
                metadata.feet[foot].solePosition.x,
                metadata.feet[foot].solePosition.y,
                metadata.feet[foot].solePosition.z,
            };
        }
        std::array<metalrobo::ArticulatedPointKinematics, 2>
            footKinematics{};
        std::vector<double> footJacobians(
            2u * 3u * g1V.size(),
            0.0
        );
        dynamicsDiagnostics =
            metalrobo::computeArticulatedPointJacobians(
                g1,
                0u,
                g1Q,
                g1V,
                footQueries,
                footKinematics,
                footJacobians,
                config
            );
        require(
            dynamicsDiagnostics.succeeded(),
            "G1 foot point Jacobians failed"
        );
        std::vector<metalrobo::ArticulatedBodyKinematics>
            g1BodyKinematics(g1.bodies.size());
        dynamicsDiagnostics =
            metalrobo::computeArticulatedBodyKinematics(
                g1,
                0u,
                g1Q,
                g1V,
                g1BodyKinematics,
                config
            );
        require(
            dynamicsDiagnostics.succeeded(),
            "G1 body kinematics failed"
        );

        double g1PoseError = 0.0;
        double g1VelocityIdentityError = 0.0;
        double g1FloatingColumnError = 0.0;
        double g1TreeSparsityError = 0.0;
        const std::array<std::array<double, 3>, 3> worldBasis{{
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0},
        }};
        const std::array<double, 3> rootPosition{
            g1Q[0], g1Q[1], g1Q[2],
        };
        for (std::size_t foot = 0u; foot < 2u; ++foot) {
            const std::uint32_t bodyIndex =
                footQueries[foot].bodyIndex;
            const auto& bodyKinematics =
                g1BodyKinematics[bodyIndex];
            const auto expectedPoint = plus(
                bodyKinematics.centerOfMassPosition,
                rotate(
                    bodyKinematics.orientation,
                    footQueries[foot].localPoint
                )
            );
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                g1PoseError = std::max(
                    g1PoseError,
                    std::abs(
                        expectedPoint[axis] -
                        footKinematics[foot].position[axis]
                    )
                );
                double velocity = 0.0;
                for (std::size_t dof = 0u;
                     dof < g1V.size();
                     ++dof) {
                    velocity +=
                        footJacobians[
                            (foot * 3u + axis) * g1V.size() + dof
                        ] *
                        g1V[dof];
                }
                g1VelocityIdentityError = std::max(
                    g1VelocityIdentityError,
                    std::abs(
                        velocity -
                        footKinematics[foot].linearVelocity[axis]
                    )
                );
                for (std::size_t translation = 0u;
                     translation < 3u;
                     ++translation) {
                    const double expected =
                        axis == translation ? 1.0 : 0.0;
                    g1FloatingColumnError = std::max(
                        g1FloatingColumnError,
                        std::abs(
                            footJacobians[
                                (foot * 3u + axis) * g1V.size() +
                                translation
                            ] -
                            expected
                        )
                    );
                }
            }
            const std::array<double, 3> rootToPoint{
                footKinematics[foot].position[0] - rootPosition[0],
                footKinematics[foot].position[1] - rootPosition[1],
                footKinematics[foot].position[2] - rootPosition[2],
            };
            for (std::size_t angular = 0u;
                 angular < 3u;
                 ++angular) {
                const auto expected =
                    cross(worldBasis[angular], rootToPoint);
                for (std::size_t axis = 0u; axis < 3u; ++axis) {
                    g1FloatingColumnError = std::max(
                        g1FloatingColumnError,
                        std::abs(
                            footJacobians[
                                (foot * 3u + axis) * g1V.size() +
                                3u + angular
                            ] -
                            expected[axis]
                        )
                    );
                }
            }

            std::vector<bool> ancestorDof(g1V.size(), false);
            std::uint32_t cursor = bodyIndex;
            while (cursor != g1.articulations[0].rootBody) {
                const std::uint32_t jointIndex =
                    g1.bodies[cursor].inboundJoint;
                const MRJointDescriptorGPU& joint =
                    g1.joints[jointIndex];
                if (joint.nv == 1u) {
                    ancestorDof[
                        joint.vOffset -
                        g1.articulations[0].vOffset
                    ] = true;
                }
                cursor = joint.parentBody;
            }
            for (std::size_t dof = 6u;
                 dof < g1V.size();
                 ++dof) {
                if (!ancestorDof[dof]) {
                    for (std::size_t axis = 0u;
                         axis < 3u;
                         ++axis) {
                        g1TreeSparsityError = std::max(
                            g1TreeSparsityError,
                            std::abs(
                                footJacobians[
                                    (foot * 3u + axis) *
                                        g1V.size() +
                                    dof
                                ]
                            )
                        );
                    }
                }
            }
        }
        require(
            g1PoseError < 2.0e-15 &&
                g1VelocityIdentityError < 2.0e-15 &&
                g1FloatingColumnError < 2.0e-15 &&
                g1TreeSparsityError == 0.0,
            "G1 analytic foot Jacobian identities failed"
        );

        std::vector<double> g1FreeVelocity(g1V.size(), 0.0);
        g1FreeVelocity[2] = -0.6;
        std::array<metalrobo::ArticulatedContact, 2> footContacts{};
        for (std::size_t foot = 0u; foot < 2u; ++foot) {
            footContacts[foot].bodyA =
                footQueries[foot].bodyIndex;
            footContacts[foot].localPointA =
                footQueries[foot].localPoint;
            footContacts[foot].localPointB =
                footKinematics[foot].position;
            footContacts[foot].normal = {0.0, 0.0, -1.0};
            footContacts[foot].tangentU = {1.0, 0.0, 0.0};
            footContacts[foot].tangentV = {0.0, -1.0, 0.0};
            footContacts[foot].regularization = {
                1.0e-8,
                1.0e-8,
                1.0e-8,
            };
            footContacts[foot].friction = 0.8;
        }
        metalrobo::ArticulatedContactProblem g1Problem;
        contactDiagnostics =
            metalrobo::buildArticulatedContactProblem(
                g1,
                0u,
                g1Q,
                g1FreeVelocity,
                footContacts,
                g1Problem,
                config,
                true
            );
        const auto g1BuildDiagnostics = contactDiagnostics;
        require(
            contactDiagnostics.succeeded() &&
                contactDiagnostics
                    .maximumDenseInverseAdapterResidual < 1.0e-10 &&
                contactDiagnostics
                    .maximumDelassusAsymmetry < 1.0e-10,
            "G1 two-foot contact construction failed"
        );
        const std::size_t contactRows = 6u;
        double g1DelassusSymmetryError = 0.0;
        for (std::size_t row = 0u; row < contactRows; ++row) {
            require(
                g1Problem.delassus[row * contactRows + row] >= 0.0,
                "G1 Delassus has a negative diagonal"
            );
            for (std::size_t column = 0u;
                 column < contactRows;
                 ++column) {
                g1DelassusSymmetryError = std::max(
                    g1DelassusSymmetryError,
                    std::abs(
                        g1Problem.delassus[row * contactRows + column] -
                        g1Problem.delassus[column * contactRows + row]
                    )
                );
            }
        }
        const std::array<double, 6> reciprocityA{
            0.3, -0.2, 0.1, 0.7, 0.4, -0.5,
        };
        const std::array<double, 6> reciprocityB{
            -0.6, 0.8, 0.2, 0.1, -0.3, 0.9,
        };
        const double g1ReciprocityError = std::abs(
            matrixBilinear(
                g1Problem.delassus,
                contactRows,
                reciprocityA,
                reciprocityB
            ) -
            matrixBilinear(
                g1Problem.delassus,
                contactRows,
                reciprocityB,
                reciprocityA
            )
        );
        std::array<double, 6> contactAction{};
        std::vector<double> transposeAction(g1V.size(), 0.0);
        contactDiagnostics =
            metalrobo::applyArticulatedContactJacobian(
                g1Problem,
                g1V,
                contactAction
            );
        require(
            contactDiagnostics.succeeded(),
            "G1 J action failed"
        );
        contactDiagnostics =
            metalrobo::applyArticulatedContactJacobianTranspose(
                g1Problem,
                reciprocityA,
                transposeAction
            );
        require(
            contactDiagnostics.succeeded(),
            "G1 J transpose action failed"
        );
        double leftAdjoint = 0.0;
        double rightAdjoint = 0.0;
        for (std::size_t row = 0u; row < contactRows; ++row) {
            leftAdjoint += contactAction[row] * reciprocityA[row];
        }
        for (std::size_t dof = 0u; dof < g1V.size(); ++dof) {
            rightAdjoint += g1V[dof] * transposeAction[dof];
        }
        const double g1JacobianAdjointError =
            std::abs(leftAdjoint - rightAdjoint);
        require(
            g1DelassusSymmetryError == 0.0 &&
                g1ReciprocityError < 2.0e-14 &&
                g1JacobianAdjointError < 2.0e-14,
            "G1 Delassus reciprocity failed"
        );

        metalrobo::QualityContactSolverConfig qualityConfig;
        qualityConfig.maximumIterations = 200u;
        qualityConfig.kktTolerance = 1.0e-10;
        const metalrobo::QualityContactSolution solution =
            metalrobo::solveQualityContactProblem(
                g1Problem.conic,
                qualityConfig
            );
        require(
            solution.converged(),
            "G1 generalized two-foot quality solve failed"
        );
        std::vector<double> g1ImpulseVelocity =
            g1FreeVelocity;
        contactDiagnostics =
            metalrobo::applyArticulatedContactImpulses(
                g1Problem,
                solution.impulses,
                g1ImpulseVelocity
            );
        require(
            contactDiagnostics.succeeded(),
            "G1 transactional impulse application failed"
        );
        const double g1SolverVelocityError = maximumError(
            g1ImpulseVelocity,
            solution.velocity
        );
        require(
            g1SolverVelocityError < 3.0e-13,
            "G1 solver velocity and inverse-mass update disagree"
        );

        // Failure paths must not partially publish either a problem or state.
        metalrobo::ArticulatedContactProblem sentinelProblem;
        sentinelProblem.articulationIndex = 99u;
        sentinelProblem.nv = 123u;
        sentinelProblem.contactJacobian = {4.0, 5.0};
        auto invalidContact = freeContact;
        invalidContact.tangentU = {0.0, 0.0, -2.0};
        contactDiagnostics =
            metalrobo::buildArticulatedContactProblem(
                freeModel,
                0u,
                freeQ,
                freeV,
                std::span(&invalidContact, 1u),
                sentinelProblem,
                config
            );
        require(
            !contactDiagnostics.succeeded() &&
                sentinelProblem.articulationIndex == 99u &&
                sentinelProblem.nv == 123u &&
                sentinelProblem.contactJacobian ==
                    std::vector<double>({4.0, 5.0}),
            "contact problem failure was not transactional"
        );
        auto selfContact = freeContact;
        selfContact.bodyB = selfContact.bodyA;
        contactDiagnostics =
            metalrobo::buildArticulatedContactProblem(
                freeModel,
                0u,
                freeQ,
                freeV,
                std::span(&selfContact, 1u),
                sentinelProblem,
                config
            );
        require(
            contactDiagnostics.status ==
                metalrobo::ArticulatedContactStatus::invalidContact &&
                sentinelProblem.articulationIndex == 99u &&
                sentinelProblem.nv == 123u &&
                sentinelProblem.contactJacobian ==
                    std::vector<double>({4.0, 5.0}),
            "same-body contact was not rejected transactionally"
        );
        std::vector<double> sentinelVelocity = freeV;
        auto invalidImpulses = testImpulse;
        invalidImpulses[1] =
            std::numeric_limits<double>::quiet_NaN();
        contactDiagnostics =
            metalrobo::applyArticulatedContactImpulses(
                freeProblem,
                invalidImpulses,
                sentinelVelocity
            );
        require(
            !contactDiagnostics.succeeded() &&
                sentinelVelocity == freeV,
            "contact impulse failure was not transactional"
        );

        std::cout
            << std::scientific
            << std::setprecision(6)
            << "articulated_contact=analytic_fp64"
            << " free_jv_error=" << freeVelocityIdentityError
            << " free_delassus_error=" << freeDelassusError
            << " free_impulse_error=" << freeImpulseError
            << " pendulum_jacobian_error="
            << pendulumJacobianError
            << " pendulum_delassus_error="
            << pendulumDelassusError
            << " armature_delassus_error="
            << armatureDelassusError
            << " armature_impulse_ratio_error="
            << armatureImpulseRatioError
            << " g1_pose_error=" << g1PoseError
            << " g1_jv_error=" << g1VelocityIdentityError
            << " g1_base_column_error=" << g1FloatingColumnError
            << " g1_tree_sparsity_error=" << g1TreeSparsityError
            << " g1_delassus_symmetry="
            << g1DelassusSymmetryError
            << " g1_dense_adapter_residual="
            << g1BuildDiagnostics
                .maximumDenseInverseAdapterResidual
            << " g1_reciprocity=" << g1ReciprocityError
            << " g1_jacobian_adjoint="
            << g1JacobianAdjointError
            << " g1_solver_velocity_error="
            << g1SolverVelocityError
            << " g1_quality_kkt="
            << solution.scaledKktCertificate
            << " frame_semantics=verbatim"
            << " transactionality=yes"
            << " status=ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "articulated_contact_probe failed: "
                  << error.what() << '\n';
        return 1;
    }
}
