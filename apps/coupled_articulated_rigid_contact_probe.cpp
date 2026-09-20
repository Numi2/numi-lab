#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/CoupledArticulatedRigidContact.hpp"
#include "metalrobo/SurgicalAssets.hpp"
#include "metalrobo/SurgicalPSM.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

double norm(const std::span<const double> values) {
    double squared = 0.0;
    for (const double value : values) {
        squared += value * value;
    }
    return std::sqrt(squared);
}

double norm(const std::array<double, 3>& values) {
    return norm(std::span<const double>(values));
}

double maximumDifference(
    const std::span<const double> left,
    const std::span<const double> right
) {
    require(left.size() == right.size(), "comparison dimensions differ");
    double result = 0.0;
    for (std::size_t index = 0u; index < left.size(); ++index) {
        result = std::max(
            result,
            std::abs(left[index] - right[index])
        );
    }
    return result;
}

double maximumDifference(
    const metalrobo::CoupledRigidBodyVelocity& left,
    const metalrobo::CoupledRigidBodyVelocity& right
) {
    return std::max(
        maximumDifference(left.linear, right.linear),
        maximumDifference(left.angular, right.angular)
    );
}

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

std::array<double, 3> scale(
    const double value,
    const std::array<double, 3>& vector
) {
    return {
        value * vector[0],
        value * vector[1],
        value * vector[2],
    };
}

std::array<double, 3> add(
    const std::array<double, 3>& left,
    const std::array<double, 3>& right
) {
    return {
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
    };
}

std::array<double, 3> subtract(
    const std::array<double, 3>& left,
    const std::array<double, 3>& right
) {
    return {
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    };
}

struct JawContactGeometry {
    std::array<double, 3> position{};
    std::array<double, 3> localPoint{};
    std::array<double, 3> normal{};
    std::array<double, 3> tangentU{};
    std::array<double, 3> tangentV{};
};

JawContactGeometry selectResponsiveJawFrame(
    const metalrobo::EngineModel& model,
    const std::span<const double> q
) {
    constexpr std::uint32_t jawBody = 7u;
    // One transverse rail of gripper1's grooved insert is a stable, authored
    // collision witness. Shape local positions are already COM-relative.
    const MRShapeGPU& jawTip = model.shapes[15u];
    require(
        jawTip.bodyIndex == jawBody &&
            jawTip.shapeType == MR_SHAPE_CAPSULE,
        "surgical PSM jaw-tip collision witness changed"
    );
    const std::array<double, 3> localPoint{
        jawTip.localPosition.x,
        jawTip.localPosition.y,
        jawTip.localPosition.z,
    };
    const std::array<metalrobo::ArticulatedPointQuery, 1> query{{
        {jawBody, localPoint},
    }};
    std::array<metalrobo::ArticulatedPointKinematics, 1> point{};
    std::vector<double> jacobian(3u * model.world.nv, 0.0);
    const std::vector<double> zeroVelocity(model.world.nv, 0.0);
    const auto kinematics =
        metalrobo::computeArticulatedPointJacobians(
            model,
            0u,
            q,
            zeroVelocity,
            query,
            point,
            jacobian
        );
    require(
        kinematics.succeeded(),
        "PSM jaw point Jacobian query failed"
    );

    std::size_t strongestAxis = 0u;
    double strongestNorm = -1.0;
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
        const double rowNorm = norm(std::span<const double>(
            jacobian.data() + axis * model.world.nv,
            model.world.nv
        ));
        if (rowNorm > strongestNorm) {
            strongestAxis = axis;
            strongestNorm = rowNorm;
        }
    }
    require(
        strongestNorm > 1.0e-5,
        "PSM jaw point has no responsive contact axis"
    );

    static constexpr std::array<std::array<double, 3>, 3>
        normals{{
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0},
        }};
    static constexpr std::array<std::array<double, 3>, 3>
        tangentU{{
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0},
            {1.0, 0.0, 0.0},
        }};
    static constexpr std::array<std::array<double, 3>, 3>
        tangentV{{
            {0.0, 0.0, 1.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
        }};
    return {
        point[0].position,
        localPoint,
        normals[strongestAxis],
        tangentU[strongestAxis],
        tangentV[strongestAxis],
    };
}

MRBodyStateGPU makeNeedleState(
    const JawContactGeometry& geometry,
    const metalrobo::CurvedSutureNeedleAsset& needle,
    const std::array<double, 3>& rigidLocalPoint
) {
    MRBodyStateGPU state{};
    const std::array<double, 3> center =
        subtract(geometry.position, rigidLocalPoint);
    const std::array<double, 3> pointVelocity = add(
        scale(-0.35, geometry.normal),
        scale(0.08, geometry.tangentU)
    );
    state.position = f4(center[0], center[1], center[2], 1.0);
    state.orientation = f4(0.0, 0.0, 0.0, 1.0);
    state.linearVelocityAndInverseMass = f4(
        pointVelocity[0],
        pointVelocity[1],
        pointVelocity[2],
        needle.rigid.body.massAndInverseMass.y
    );
    state.angularVelocity = f4(0.0, 0.0, 0.0);
    state.inverseInertiaWorldRow0 =
        needle.rigid.body.inverseInertiaRow0;
    state.inverseInertiaWorldRow1 =
        needle.rigid.body.inverseInertiaRow1;
    state.inverseInertiaWorldRow2 =
        needle.rigid.body.inverseInertiaRow2;
    state.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
    state.flagsAndIndices[1] = MR_INVALID_INDEX;
    state.flagsAndIndices[2] = 41021u;
    return state;
}

metalrobo::CoupledRigidBodyVelocity sentinelRigidVelocity() {
    return {
        {71.0, 72.0, 73.0},
        {81.0, 82.0, 83.0},
    };
}

bool equal(
    const metalrobo::CoupledRigidBodyVelocity& left,
    const metalrobo::CoupledRigidBodyVelocity& right
) {
    return left.linear == right.linear &&
        left.angular == right.angular;
}

} // namespace

int main() {
    try {
        const metalrobo::EngineModel model =
            metalrobo::makeDvrkPsmLargeNeedleDriverEngineModel();
        const std::vector<double> q(
            model.defaultQ.begin(),
            model.defaultQ.end()
        );
        std::vector<double> freeArticulationVelocity(
            model.world.nv,
            0.0
        );
        const JawContactGeometry geometry =
            selectResponsiveJawFrame(model, q);

        const metalrobo::CurvedSutureNeedleAsset needle =
            metalrobo::makeCurvedSutureNeedleAsset({
                .bodyIndex = 0u,
                .materialIndex = 0u,
                .slotGenerationBase = 410210u,
                .collisionGroup = 1u,
                .collisionMask = ~0u,
                .motionType = MR_MOTION_DYNAMIC,
            });
        const std::uint32_t graspShapeIndex =
            (
                needle.metadata.graspShapeBegin +
                needle.metadata.graspShapeEnd
            ) / 2u;
        require(
            graspShapeIndex >= needle.metadata.graspShapeBegin &&
                graspShapeIndex < needle.metadata.graspShapeEnd &&
                graspShapeIndex < needle.rigid.shapes.size(),
            "procedural needle grasp-zone metadata is invalid"
        );
        const MRShapeGPU& graspShape =
            needle.rigid.shapes[graspShapeIndex];
        const std::array<double, 3> rigidLocalPoint{
            graspShape.localPosition.x,
            graspShape.localPosition.y,
            graspShape.localPosition.z,
        };
        const std::array<MRBodyStateGPU, 1> rigidBodies{{
            makeNeedleState(geometry, needle, rigidLocalPoint),
        }};
        metalrobo::CoupledArticulatedRigidContact contact;
        contact.articulatedBody = 7u;
        contact.rigidBody = 0u;
        contact.localPointArticulated = geometry.localPoint;
        contact.localPointRigid = rigidLocalPoint;
        contact.normal = geometry.normal;
        contact.tangentU = geometry.tangentU;
        contact.tangentV = geometry.tangentV;
        contact.regularization = {1.0e-9, 1.0e-9, 1.0e-9};
        contact.friction = std::sqrt(
            static_cast<double>(
                model.materials[
                    model.shapes[15u].materialIndex
                ].friction.y
            ) *
            static_cast<double>(needle.rigid.material.friction.y)
        );
        const std::array<
            metalrobo::CoupledArticulatedRigidContact,
            1
        > contacts{{contact}};

        metalrobo::QualityContactSolverConfig solverConfig;
        solverConfig.maximumIterations = 500u;
        solverConfig.kktTolerance = 1.0e-11;

        std::vector<double> postArticulation(
            model.world.nv,
            -19.0
        );
        std::array<metalrobo::CoupledRigidBodyVelocity, 1>
            postRigid{{sentinelRigidVelocity()}};
        const auto diagnostics =
            metalrobo::solveCoupledArticulatedRigidContactsCpu(
                model,
                0u,
                q,
                freeArticulationVelocity,
                rigidBodies,
                contacts,
                postArticulation,
                postRigid,
                {},
                solverConfig
            );
        require(
            diagnostics.succeeded(),
            "coupled PSM/needle solve failed: " +
                diagnostics.failure
        );
        require(
            diagnostics.impulses.size() == 3u &&
                diagnostics.impulses[0] > 0.0,
            "coupled solve produced no compressive impulse"
        );
        const double impulseNorm = norm(diagnostics.impulses);
        const double articulatedVelocityChange =
            norm(postArticulation);
        const std::array<double, 3> freeRigidLinear{
            rigidBodies[0].linearVelocityAndInverseMass.x,
            rigidBodies[0].linearVelocityAndInverseMass.y,
            rigidBodies[0].linearVelocityAndInverseMass.z,
        };
        const double rigidLinearVelocityChange = norm(subtract(
            postRigid[0].linear,
            freeRigidLinear
        ));
        const double rigidAngularVelocityChange =
            norm(postRigid[0].angular);
        require(
            impulseNorm > 1.0e-12 &&
                articulatedVelocityChange > 1.0e-12 &&
                rigidLinearVelocityChange > 1.0e-12 &&
                rigidAngularVelocityChange > 1.0e-12,
            "contact did not produce a two-way velocity response"
        );
        require(
            diagnostics.freeContactVelocity.size() == 3u &&
                diagnostics.postContactVelocity.size() == 3u &&
                diagnostics.freeContactVelocity[0] < -0.3 &&
                diagnostics.postContactVelocity[0] >
                    diagnostics.freeContactVelocity[0] + 0.1,
            "coupled impulse did not correct closing contact velocity"
        );
        require(
            diagnostics.quality.maximumPrimalConeViolation <=
                1.0e-10 &&
                diagnostics.quality.scaledKktCertificate <= 1.1e-11,
            "exact circular Coulomb certificate failed"
        );

        std::vector<double> replayArticulation(
            model.world.nv,
            std::numeric_limits<double>::quiet_NaN()
        );
        std::array<metalrobo::CoupledRigidBodyVelocity, 1>
            replayRigid{{sentinelRigidVelocity()}};
        const auto replay =
            metalrobo::solveCoupledArticulatedRigidContactsCpu(
                model,
                0u,
                q,
                freeArticulationVelocity,
                rigidBodies,
                contacts,
                replayArticulation,
                replayRigid,
                {},
                solverConfig
            );
        require(replay.succeeded(), "deterministic replay failed");
        const double replayError = std::max({
            maximumDifference(postArticulation, replayArticulation),
            maximumDifference(postRigid[0], replayRigid[0]),
            maximumDifference(
                diagnostics.impulses,
                replay.impulses
            ),
        });
        require(
            replayError == 0.0,
            "coupled contact replay is not deterministic"
        );

        auto invalidContacts = contacts;
        invalidContacts[0].tangentV =
            scale(-1.0, invalidContacts[0].tangentV);
        std::vector<double> invalidArticulation(
            model.world.nv,
            123.0
        );
        std::array<metalrobo::CoupledRigidBodyVelocity, 1>
            invalidRigid{{sentinelRigidVelocity()}};
        const auto invalidFrame =
            metalrobo::solveCoupledArticulatedRigidContactsCpu(
                model,
                0u,
                q,
                freeArticulationVelocity,
                rigidBodies,
                invalidContacts,
                invalidArticulation,
                invalidRigid,
                {},
                solverConfig
            );
        require(
            invalidFrame.status ==
                metalrobo::CoupledArticulatedRigidContactStatus::
                    invalidContact &&
                std::ranges::all_of(
                    invalidArticulation,
                    [](const double value) {
                        return value == 123.0;
                    }
                ) &&
                equal(invalidRigid[0], sentinelRigidVelocity()),
            "invalid frame violated output transactionality"
        );

        auto staticBodies = rigidBodies;
        staticBodies[0].flagsAndIndices[0] = MR_MOTION_STATIC;
        std::ranges::fill(invalidArticulation, 124.0);
        invalidRigid[0] = sentinelRigidVelocity();
        const auto invalidRigidBody =
            metalrobo::solveCoupledArticulatedRigidContactsCpu(
                model,
                0u,
                q,
                freeArticulationVelocity,
                staticBodies,
                contacts,
                invalidArticulation,
                invalidRigid,
                {},
                solverConfig
            );
        require(
            invalidRigidBody.status ==
                metalrobo::CoupledArticulatedRigidContactStatus::
                    invalidRigidBody &&
                std::ranges::all_of(
                    invalidArticulation,
                    [](const double value) {
                        return value == 124.0;
                    }
                ) &&
                equal(invalidRigid[0], sentinelRigidVelocity()),
            "invalid rigid state violated output transactionality"
        );

        std::cout
            << std::setprecision(12)
            << "coupled_articulated_rigid_contact"
            << " model=" << model.name
            << " articulated_body=" << contact.articulatedBody
            << " needle_mass_kg=" << needle.rigid.massKg
            << " needle_grasp_shape=" << graspShapeIndex
            << " impulse_norm=" << impulseNorm
            << " normal_impulse=" << diagnostics.impulses[0]
            << " articulated_dv=" << articulatedVelocityChange
            << " rigid_linear_dv=" << rigidLinearVelocityChange
            << " rigid_angular_dv=" << rigidAngularVelocityChange
            << " pre_normal_velocity="
            << diagnostics.freeContactVelocity[0]
            << " post_normal_velocity="
            << diagnostics.postContactVelocity[0]
            << " inverse_residual="
            << diagnostics.maximumArticulationInverseResidual
            << " velocity_reconstruction_error="
            << diagnostics.maximumVelocityReconstructionError
            << " contact_velocity_consistency_error="
            << diagnostics.maximumContactVelocityConsistencyError
            << " cone_violation="
            << diagnostics.quality.maximumPrimalConeViolation
            << " kkt=" << diagnostics.quality.scaledKktCertificate
            << " replay_error=" << replayError
            << " status=ok\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "coupled_articulated_rigid_contact status=failed reason="
            << exception.what() << '\n';
        return 1;
    }
}
