#include "metalrobo/ArticulatedActuation.hpp"
#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/ArticulatedJointLimits.hpp"
#include "metalrobo/ArticulatedWorld.hpp"
#include "metalrobo/Franka.hpp"
#include "metalrobo/MetalArticulatedABA.hpp"
#include "metalrobo/Model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
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

bool close(
    const double left,
    const double right,
    const double tolerance = 2.0e-7
) {
    return std::abs(left - right) <= tolerance;
}

using Vec3 = std::array<double, 3>;
using Quaternion = std::array<double, 4>;

struct Pose {
    Vec3 position{};
    Quaternion orientation{0.0, 0.0, 0.0, 1.0};
};

Quaternion quaternion(const mr_float4 value) {
    return {value.x, value.y, value.z, value.w};
}

Quaternion normalized(const Quaternion value) {
    const double inverseNorm = 1.0 / std::sqrt(
        value[0] * value[0] +
        value[1] * value[1] +
        value[2] * value[2] +
        value[3] * value[3]
    );
    return {
        value[0] * inverseNorm,
        value[1] * inverseNorm,
        value[2] * inverseNorm,
        value[3] * inverseNorm,
    };
}

Quaternion multiply(
    const Quaternion left,
    const Quaternion right
) {
    return {
        left[3] * right[0] + left[0] * right[3] +
            left[1] * right[2] - left[2] * right[1],
        left[3] * right[1] - left[0] * right[2] +
            left[1] * right[3] + left[2] * right[0],
        left[3] * right[2] + left[0] * right[1] -
            left[1] * right[0] + left[2] * right[3],
        left[3] * right[3] - left[0] * right[0] -
            left[1] * right[1] - left[2] * right[2],
    };
}

Quaternion axisAngle(
    const mr_float4 axis,
    const double angle
) {
    const double axisNorm = std::sqrt(
        static_cast<double>(axis.x) * axis.x +
        static_cast<double>(axis.y) * axis.y +
        static_cast<double>(axis.z) * axis.z
    );
    const double scale = std::sin(0.5 * angle) / axisNorm;
    return {
        axis.x * scale,
        axis.y * scale,
        axis.z * scale,
        std::cos(0.5 * angle),
    };
}

Vec3 rotate(const Quaternion rawRotation, const Vec3 point) {
    const Quaternion rotation = normalized(rawRotation);
    const Quaternion conjugate{
        -rotation[0],
        -rotation[1],
        -rotation[2],
        rotation[3],
    };
    const Quaternion pure{point[0], point[1], point[2], 0.0};
    const Quaternion result = multiply(
        multiply(rotation, pure),
        conjugate
    );
    return {result[0], result[1], result[2]};
}

Vec3 add(const Vec3 left, const Vec3 right) {
    return {
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
    };
}

std::vector<Pose> legacyComPoses(
    const metalrobo::Model& model,
    const std::span<const double> q
) {
    std::vector<Pose> linkPoses(model.links.size());
    for (std::size_t index = 0u;
         index < model.joints.size();
         ++index) {
        const MRJointGPU& joint = model.joints[index];
        const std::size_t parent =
            static_cast<std::size_t>(joint.parentLink);
        const std::size_t child =
            static_cast<std::size_t>(joint.childLink);
        const Vec3 offset{
            joint.parentOffset.x,
            joint.parentOffset.y,
            joint.parentOffset.z,
        };
        linkPoses[child].position = add(
            linkPoses[parent].position,
            rotate(linkPoses[parent].orientation, offset)
        );
        const Quaternion jointFrame = multiply(
            linkPoses[parent].orientation,
            quaternion(joint.parentRotation)
        );
        linkPoses[child].orientation = normalized(
            multiply(jointFrame, axisAngle(joint.axis, q[index]))
        );
    }
    for (std::size_t body = 0u;
         body < model.links.size();
         ++body) {
        const MRLinkGPU& link = model.links[body];
        const Vec3 centerOfMass{
            link.massAndCOMX.y,
            link.massAndCOMX.z,
            link.massAndCOMX.w,
        };
        linkPoses[body].position = add(
            linkPoses[body].position,
            rotate(linkPoses[body].orientation, centerOfMass)
        );
    }
    return linkPoses;
}

struct KinematicsParity {
    double position = 0.0;
    double orientation = 0.0;
};

KinematicsParity compareLegacyKinematics(
    const metalrobo::Model& legacy,
    const metalrobo::EngineModel& model
) {
    KinematicsParity parity;
    const Vec3 fixedRootShift{
        legacy.links[0].massAndCOMX.y,
        legacy.links[0].massAndCOMX.z,
        legacy.links[0].massAndCOMX.w,
    };
    for (std::size_t sample = 0u; sample < 6u; ++sample) {
        std::vector<double> q(
            model.defaultQ.begin(),
            model.defaultQ.end()
        );
        for (std::size_t dof = 0u; dof < q.size(); ++dof) {
            const MRDofPropertiesGPU& properties = model.dofs[dof];
            const double candidate =
                q[dof] +
                0.11 *
                    std::sin(
                        0.53 * static_cast<double>(
                            1u + sample * q.size() + dof
                        )
                    );
            q[dof] = std::clamp(
                candidate,
                static_cast<double>(properties.limits.x) + 1.0e-3,
                static_cast<double>(properties.limits.y) - 1.0e-3
            );
        }
        const std::vector<Pose> expected =
            legacyComPoses(legacy, q);
        std::vector<double> zeroVelocity(model.world.nv, 0.0);
        std::vector<metalrobo::ArticulatedBodyKinematics> actual(
            model.world.bodyCount
        );
        const auto diagnostics =
            metalrobo::computeArticulatedBodyKinematics(
                model,
                0u,
                q,
                zeroVelocity,
                actual
            );
        require(
            diagnostics.succeeded(),
            "canonical Franka multi-state kinematics failed"
        );
        for (std::size_t body = 0u;
             body < actual.size();
             ++body) {
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                parity.position = std::max(
                    parity.position,
                    std::abs(
                        actual[body].centerOfMassPosition[axis] -
                        (expected[body].position[axis] -
                         fixedRootShift[axis])
                    )
                );
            }
            const Quaternion actualOrientation =
                actual[body].orientation;
            const double quaternionDot = std::abs(
                actualOrientation[0] *
                    expected[body].orientation[0] +
                actualOrientation[1] *
                    expected[body].orientation[1] +
                actualOrientation[2] *
                    expected[body].orientation[2] +
                actualOrientation[3] *
                    expected[body].orientation[3]
            );
            parity.orientation = std::max(
                parity.orientation,
                std::abs(1.0 - quaternionDot)
            );
        }
    }
    return parity;
}

metalrobo::ArticulatedDynamicsConfig dynamicsConfig(
    const metalrobo::EngineModel& model
) {
    metalrobo::ArticulatedDynamicsConfig config;
    config.gravity = {
        model.world.gravityAndTimestep.x,
        model.world.gravityAndTimestep.y,
        model.world.gravityAndTimestep.z,
    };
    config.timestep = model.world.gravityAndTimestep.w;
    config.applyBodyDamping = true;
    config.integrator =
        metalrobo::ArticulatedIntegrator::symplecticEuler;
    return config;
}

struct MetalParity {
    double acceleration = 0.0;
    double accelerationScaled = 0.0;
    double nextV = 0.0;
    double nextQ = 0.0;
};

MetalParity compareMetalABA(
    const metalrobo::EngineModel& model,
    const metalrobo::MetalArticulatedABAInput& input,
    const metalrobo::MetalArticulatedABAResult& result
) {
    const MRArticulationGPU& articulation =
        model.articulations[input.articulationIndex];
    const auto config = dynamicsConfig(model);
    MetalParity parity;
    for (std::size_t environment = 0u;
         environment < input.environmentCount;
         ++environment) {
        const std::size_t qBase =
            environment * articulation.nq;
        const std::size_t vBase =
            environment * articulation.nv;
        std::vector<double> q(
            input.q.begin() + static_cast<std::ptrdiff_t>(qBase),
            input.q.begin() + static_cast<std::ptrdiff_t>(
                qBase + articulation.nq
            )
        );
        std::vector<double> v(
            input.v.begin() + static_cast<std::ptrdiff_t>(vBase),
            input.v.begin() + static_cast<std::ptrdiff_t>(
                vBase + articulation.nv
            )
        );
        std::vector<double> effort(
            input.effort.begin() + static_cast<std::ptrdiff_t>(vBase),
            input.effort.begin() + static_cast<std::ptrdiff_t>(
                vBase + articulation.nv
            )
        );
        std::vector<double> acceleration(articulation.nv, 0.0);
        const auto forward =
            metalrobo::computeArticulatedForwardDynamics(
                model,
                input.articulationIndex,
                q,
                v,
                effort,
                {},
                acceleration,
                config
            );
        require(
            forward.succeeded(),
            "Franka CPU ABA reference failed"
        );
        for (std::size_t dof = 0u;
             dof < articulation.nv;
             ++dof) {
            const double error = std::abs(
                static_cast<double>(
                    result.acceleration[vBase + dof]
                ) - acceleration[dof]
            );
            parity.acceleration =
                std::max(parity.acceleration, error);
            parity.accelerationScaled = std::max(
                parity.accelerationScaled,
                error / std::max(1.0, std::abs(acceleration[dof]))
            );
            v[dof] += config.timestep * acceleration[dof];
            parity.nextV = std::max(
                parity.nextV,
                std::abs(
                    static_cast<double>(result.nextV[vBase + dof]) -
                    v[dof]
                )
            );
        }
        const auto integration =
            metalrobo::integrateArticulatedConfiguration(
                model,
                input.articulationIndex,
                q,
                v,
                config
            );
        require(
            integration.succeeded(),
            "Franka CPU configuration integration failed"
        );
        for (std::size_t coordinate = 0u;
             coordinate < articulation.nq;
             ++coordinate) {
            parity.nextQ = std::max(
                parity.nextQ,
                std::abs(
                    static_cast<double>(
                        result.nextQ[qBase + coordinate]
                    ) - q[coordinate]
                )
            );
        }
    }
    return parity;
}

} // namespace

int main() {
    try {
        const metalrobo::Model legacy =
            metalrobo::makeFrankaPandaModel();
        const metalrobo::EngineModel model =
            metalrobo::makeFrankaPandaEngineModel();
        std::string reason;
        require(
            model.valid(&reason),
            "EngineModel::valid rejected Franka: " + reason
        );
        require(
            model.world.abiVersion == MR_ENGINE_ABI_VERSION &&
                model.world.bodyCount ==
                    metalrobo::kFrankaPandaBodyCount &&
                model.world.jointCount ==
                    metalrobo::kFrankaPandaJointCount &&
                model.world.shapeCount ==
                    metalrobo::kFrankaPandaShapeCount &&
                model.world.nq == 7u &&
                model.world.nv == 7u &&
                model.articulations.size() == 1u &&
                model.articulations[0].rootType == MR_ROOT_FIXED,
            "canonical Franka topology/counts are wrong"
        );

        constexpr mr_u32 expectedDofFlags =
            MR_DOF_FLAG_ACTUATED |
            MR_DOF_FLAG_POSITION_LIMIT |
            MR_DOF_FLAG_VELOCITY_LIMIT |
            MR_DOF_FLAG_EFFORT_LIMIT |
            MR_DOF_FLAG_DRIVE;
        for (std::size_t index = 0u;
             index < model.joints.size();
             ++index) {
            const MRBodyPropertiesGPU& parent = model.bodies[index];
            const MRBodyPropertiesGPU& child =
                model.bodies[index + 1u];
            const MRJointDescriptorGPU& joint = model.joints[index];
            const MRJointGPU& legacyJoint = legacy.joints[index];
            const MRDofPropertiesGPU& dof = model.dofs[index];
            require(
                close(
                    joint.parentAnchor.x + parent.centerOfMass.x,
                    legacyJoint.parentOffset.x
                ) &&
                    close(
                        joint.parentAnchor.y + parent.centerOfMass.y,
                        legacyJoint.parentOffset.y
                    ) &&
                    close(
                        joint.parentAnchor.z + parent.centerOfMass.z,
                        legacyJoint.parentOffset.z
                    ) &&
                    close(
                        joint.childAnchor.x + child.centerOfMass.x,
                        0.0
                    ) &&
                    close(
                        joint.childAnchor.y + child.centerOfMass.y,
                        0.0
                    ) &&
                    close(
                        joint.childAnchor.z + child.centerOfMass.z,
                        0.0
                    ),
                "Franka joint anchors are not COM-correct"
            );
            require(
                dof.flags == expectedDofFlags &&
                    close(dof.limits.x, legacyJoint.limits.x) &&
                    close(dof.limits.y, legacyJoint.limits.y) &&
                    close(dof.limits.z, legacyJoint.limits.z) &&
                    close(dof.limits.w, legacyJoint.limits.w) &&
                    close(dof.drive.x, legacyJoint.drive.x) &&
                    close(dof.drive.y, legacyJoint.drive.y) &&
                    close(dof.drive.z, legacyJoint.drive.w) &&
                    close(dof.drive.w, 0.2),
                "Franka actuation/limit metadata changed in compilation"
            );
        }
        for (std::size_t index = 0u;
             index < model.shapes.size();
             ++index) {
            const MRShapeGPU& shape = model.shapes[index];
            const MRColliderGPU& legacyShape =
                legacy.colliders[index];
            const MRBodyPropertiesGPU& body =
                model.bodies[shape.bodyIndex];
            require(
                shape.shapeType == MR_SHAPE_SPHERE &&
                    close(
                        shape.localPosition.x + body.centerOfMass.x,
                        legacyShape.centerAndRadius.x
                    ) &&
                    close(
                        shape.localPosition.y + body.centerOfMass.y,
                        legacyShape.centerAndRadius.y
                    ) &&
                    close(
                        shape.localPosition.z + body.centerOfMass.z,
                        legacyShape.centerAndRadius.z
                    ) &&
                    close(
                        shape.dimensions.x,
                        legacyShape.centerAndRadius.w
                    ),
                "Franka collision compilation changed link geometry"
            );
        }
        const KinematicsParity kinematicsParity =
            compareLegacyKinematics(legacy, model);
        require(
            kinematicsParity.position < 5.0e-7 &&
                kinematicsParity.orientation < 5.0e-12,
            "canonical Franka multi-state FK changed the legacy chain"
        );

        std::vector<double> q(
            model.defaultQ.begin(),
            model.defaultQ.end()
        );
        std::vector<double> v(7u, 0.0);
        std::vector<double> expectedAcceleration(7u, 0.0);
        for (std::size_t index = 0u; index < 7u; ++index) {
            q[index] += 0.025 * std::sin(0.7 * (index + 1u));
            v[index] = 0.13 * std::cos(0.4 * (index + 1u));
            expectedAcceleration[index] =
                0.7 * std::sin(0.9 * (index + 1u));
        }
        std::vector<double> generalizedForce(7u, 0.0);
        const auto config = dynamicsConfig(model);
        const auto inverse =
            metalrobo::computeArticulatedInverseDynamics(
                model,
                0u,
                q,
                v,
                expectedAcceleration,
                {},
                generalizedForce,
                config
            );
        require(
            inverse.succeeded(),
            "canonical Franka FP64 inverse dynamics failed"
        );
        std::vector<double> recoveredAcceleration(7u, 0.0);
        const auto forward =
            metalrobo::computeArticulatedForwardDynamics(
                model,
                0u,
                q,
                v,
                generalizedForce,
                {},
                recoveredAcceleration,
                config
            );
        require(
            forward.succeeded(),
            "canonical Franka FP64 forward dynamics failed"
        );
        double forwardInverseError = 0.0;
        for (std::size_t index = 0u; index < 7u; ++index) {
            forwardInverseError = std::max(
                forwardInverseError,
                std::abs(
                    recoveredAcceleration[index] -
                    expectedAcceleration[index]
                )
            );
        }
        require(
            forwardInverseError < 2.0e-11,
            "canonical Franka forward/inverse consistency regressed"
        );

        std::vector<metalrobo::ArticulatedDofCommand> commands(7u);
        for (std::size_t index = 0u; index < commands.size(); ++index) {
            commands[index].mode =
                metalrobo::ArticulatedActuationMode::modelPD;
            commands[index].desiredPosition =
                static_cast<double>(model.defaultQ[index]) + 0.01;
        }
        const std::vector<double> resetQ(
            model.defaultQ.begin(),
            model.defaultQ.end()
        );
        const std::vector<double> resetV(7u, 0.0);

        std::vector<double> worldQ = resetQ;
        std::vector<double> worldV = resetV;
        const std::vector<double> zeroForce(7u, 0.0);
        metalrobo::ArticulatedWorldConfig worldConfig;
        metalrobo::ArticulatedWorldCache worldCache;
        const auto worldDiagnostics =
            metalrobo::stepArticulatedWorldCpu(
                model,
                0u,
                worldQ,
                worldV,
                zeroForce,
                {},
                {},
                {},
                worldConfig,
                worldCache
            );
        require(
            worldDiagnostics.succeeded() &&
                worldDiagnostics.collision.requiredRawContacts == 0u &&
                worldDiagnostics.contactCount == 0u &&
                worldDiagnostics.maximumNormalImpulse == 0.0 &&
                worldCache.step == 1u,
            "canonical Franka default world step admitted parent-child "
            "self-contact"
        );

        metalrobo::ArticulatedActuationResult actuation;
        const auto actuationDiagnostics =
            metalrobo::evaluateArticulatedActuation(
                model,
                0u,
                resetQ,
                resetV,
                commands,
                actuation
            );
        require(
            actuationDiagnostics.succeeded() &&
                actuationDiagnostics.stictionDofCount == 7u &&
                actuationDiagnostics.saturatedDofCount == 0u,
            "canonical Franka model drive is not executable"
        );

        std::vector<double> limitQ = resetQ;
        limitQ[0] =
            static_cast<double>(model.dofs[0].limits.y) - 5.0e-4;
        std::vector<double> freeVelocity(7u, 0.0);
        freeVelocity[0] = 1.0;
        metalrobo::ArticulatedJointLimitConfig limitConfig;
        limitConfig.timestep = model.world.gravityAndTimestep.w;
        std::vector<metalrobo::ArticulatedJointLimitRow> limitRows;
        const auto limitDiagnostics =
            metalrobo::compileArticulatedJointLimitRows(
                model,
                0u,
                limitQ,
                freeVelocity,
                limitRows,
                limitConfig
            );
        require(
            limitDiagnostics.succeeded() &&
                !limitRows.empty() &&
                limitRows.front().side ==
                    metalrobo::ArticulatedJointLimitSide::upper &&
                limitRows.front().localQIndex == 0u &&
                limitRows.front().localVIndex == 0u,
            "canonical Franka limits did not compile into constraints"
        );

        constexpr std::size_t environmentCount = 3u;
        std::vector<float> metalQ(environmentCount * 7u);
        std::vector<float> metalV(environmentCount * 7u);
        std::vector<float> metalEffort(environmentCount * 7u);
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            for (std::size_t dof = 0u; dof < 7u; ++dof) {
                const std::size_t index = environment * 7u + dof;
                metalQ[index] =
                    model.defaultQ[dof] +
                    static_cast<float>(
                        0.015 *
                        std::sin(
                            0.4 * static_cast<double>(
                                1u + environment * 7u + dof
                            )
                        )
                    );
                metalV[index] =
                    static_cast<float>(
                        0.08 *
                        std::cos(
                            0.3 * static_cast<double>(
                                1u + environment * 7u + dof
                            )
                        )
                    );
                metalEffort[index] =
                    static_cast<float>(
                        0.6 *
                        std::sin(
                            0.2 * static_cast<double>(
                                1u + environment * 7u + dof
                            )
                        )
                    );
            }
        }
        metalrobo::MetalArticulatedABAInput metalInput;
        metalInput.articulationIndex = 0u;
        metalInput.environmentCount = environmentCount;
        metalInput.q = metalQ;
        metalInput.v = metalV;
        metalInput.effort = metalEffort;
        metalInput.applyBodyDamping = true;
        metalrobo::MetalArticulatedABAContext metalContext;
        metalrobo::MetalArticulatedABAResult metalResult;
        const auto metalDiagnostics =
            metalContext.run(model, metalInput, metalResult);
        require(
            metalDiagnostics.succeeded(),
            "canonical Franka public Metal ABA failed: " +
                metalDiagnostics.message
        );
        const MetalParity parity =
            compareMetalABA(model, metalInput, metalResult);
        require(
            parity.accelerationScaled < 5.0e-5 &&
                parity.nextV < 5.0e-5 &&
                parity.nextQ < 5.0e-5,
            "canonical Franka Metal/FP64 parity regressed"
        );

        std::cout
            << "franka_engine_model=abi_v2"
            << " bodies=" << model.world.bodyCount
            << " dofs=" << model.world.nv
            << " shapes=" << model.world.shapeCount
            << " fk_position_error=" << kinematicsParity.position
            << " fk_orientation_error="
            << kinematicsParity.orientation
            << " forward_inverse_error=" << forwardInverseError
            << " metal_acceleration_scaled_error="
            << parity.accelerationScaled
            << " metal_next_v_error=" << parity.nextV
            << " metal_next_q_error=" << parity.nextQ
            << " device=\"" << metalDiagnostics.deviceName << "\""
            << " actuation=pass"
            << " limits=pass"
            << " com_frames=pass"
            << " parent_child_exclusions=pass"
            << " status=ok\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "franka_engine_model=failed reason=\""
            << exception.what() << "\"\n";
        return 1;
    }
}
