#include "metalrobo/ArticulatedActuation.hpp"
#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/ArticulatedJointLimits.hpp"
#include "metalrobo/ArticulatedWorld.hpp"
#include "metalrobo/MetalArticulatedABA.hpp"
#include "metalrobo/SurgicalPSM.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
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

bool close(
    const double left,
    const double right,
    const double tolerance = 2.0e-7
) {
    return std::abs(left - right) <= tolerance;
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

struct PointPair {
    std::array<double, 3> remoteCenter{};
    std::array<double, 3> insertionOrigin{};
};

PointPair queryRemoteCenterAndInsertionOrigin(
    const metalrobo::EngineModel& model,
    const metalrobo::SurgicalPSMModelMetadata& metadata,
    const std::span<const double> q
) {
    const MRBodyPropertiesGPU& insertionBody = model.bodies[3u];
    const std::array<metalrobo::ArticulatedPointQuery, 2> queries{{
        {
            metadata.remoteCenterBodyIndex,
            {
                metadata.remoteCenterLocalPosition.x,
                metadata.remoteCenterLocalPosition.y,
                metadata.remoteCenterLocalPosition.z,
            },
        },
        {
            3u,
            {
                -insertionBody.centerOfMass.x,
                -insertionBody.centerOfMass.y,
                -insertionBody.centerOfMass.z,
            },
        },
    }};
    std::array<metalrobo::ArticulatedPointKinematics, 2> points{};
    std::vector<double> jacobians(
        queries.size() * 3u * model.world.nv,
        0.0
    );
    const std::vector<double> zeroVelocity(model.world.nv, 0.0);
    const auto diagnostics =
        metalrobo::computeArticulatedPointJacobians(
            model,
            0u,
            q,
            zeroVelocity,
            queries,
            points,
            jacobians
        );
    require(
        diagnostics.succeeded(),
        "surgical PSM RCM point query failed"
    );
    return {
        points[0].position,
        points[1].position,
    };
}

double distance(
    const std::array<double, 3>& left,
    const std::array<double, 3>& right
) {
    double squared = 0.0;
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
        const double delta = left[axis] - right[axis];
        squared += delta * delta;
    }
    return std::sqrt(squared);
}

struct RcmGate {
    double radialError = 0.0;
    double centerDrift = 0.0;
    double insertionDeltaError = 0.0;
};

struct JawGapGate {
    double closedSurfaceGap = 0.0;
    double maximumSurfaceGap = 0.0;
    double minimumGapIncrease = std::numeric_limits<double>::infinity();
};

RcmGate verifyRemoteCenter(
    const metalrobo::EngineModel& model,
    const metalrobo::SurgicalPSMModelMetadata& metadata
) {
    RcmGate gate;
    std::array<double, 3> referenceCenter{};
    bool hasReference = false;

    for (std::size_t sample = 0u; sample < 5u; ++sample) {
        std::vector<double> q(
            model.defaultQ.begin(),
            model.defaultQ.end()
        );
        q[0] = -0.5 + 0.25 * static_cast<double>(sample);
        q[1] = 0.35 * std::sin(0.7 * static_cast<double>(sample + 1u));
        q[2] = 0.065 + 0.03 * static_cast<double>(sample);
        q[3] = 0.0;
        q[4] = 0.0;
        q[5] = 0.0;

        const PointPair low =
            queryRemoteCenterAndInsertionOrigin(model, metadata, q);
        gate.radialError = std::max(
            gate.radialError,
            std::abs(distance(
                low.remoteCenter,
                low.insertionOrigin
            ) - q[2])
        );
        if (!hasReference) {
            referenceCenter = low.remoteCenter;
            hasReference = true;
        }
        gate.centerDrift = std::max(
            gate.centerDrift,
            distance(referenceCenter, low.remoteCenter)
        );

        constexpr double insertionIncrement = 0.011;
        q[2] += insertionIncrement;
        const PointPair high =
            queryRemoteCenterAndInsertionOrigin(model, metadata, q);
        gate.insertionDeltaError = std::max(
            gate.insertionDeltaError,
            std::abs(
                distance(
                    low.insertionOrigin,
                    high.insertionOrigin
                ) - insertionIncrement
            )
        );
    }
    return gate;
}

JawGapGate verifyJawApertureGeometry(
    const metalrobo::EngineModel& model,
    const double maximumAperture
) {
    constexpr std::size_t jawOneTipShape = 15u;
    constexpr std::size_t jawTwoTipShape = 17u;
    constexpr std::size_t sampleCount = 17u;
    require(
        model.shapes.size() > jawTwoTipShape &&
            model.shapes[jawOneTipShape].bodyIndex == 7u &&
            model.shapes[jawTwoTipShape].bodyIndex == 8u &&
            model.shapes[jawOneTipShape].shapeType == MR_SHAPE_SPHERE &&
            model.shapes[jawTwoTipShape].shapeType == MR_SHAPE_SPHERE,
        "surgical PSM jaw-tip witnesses changed"
    );

    const MRShapeGPU& tipOne = model.shapes[jawOneTipShape];
    const MRShapeGPU& tipTwo = model.shapes[jawTwoTipShape];
    const std::array<metalrobo::ArticulatedPointQuery, 2> queries{{
        {
            tipOne.bodyIndex,
            {
                tipOne.localPosition.x,
                tipOne.localPosition.y,
                tipOne.localPosition.z,
            },
        },
        {
            tipTwo.bodyIndex,
            {
                tipTwo.localPosition.x,
                tipTwo.localPosition.y,
                tipTwo.localPosition.z,
            },
        },
    }};

    JawGapGate gate;
    double previousGap = 0.0;
    for (std::size_t sample = 0u; sample < sampleCount; ++sample) {
        const double aperture =
            maximumAperture * static_cast<double>(sample) /
            static_cast<double>(sampleCount - 1u);
        const auto logicalFloat =
            metalrobo::surgicalPSMDefaultLogicalPositionTargets();
        std::vector<double> logical(logicalFloat.begin(), logicalFloat.end());
        logical[metalrobo::kSurgicalPSMLogicalJawApertureIndex] =
            aperture;
        std::vector<double> q;
        const auto map =
            metalrobo::expandSurgicalPSMLogicalPositionTargets(
                model,
                logical,
                q
            );
        require(
            map.succeeded(),
            "surgical PSM jaw-gap command mapping failed"
        );

        const std::vector<double> zeroVelocity(model.world.nv, 0.0);
        std::array<metalrobo::ArticulatedPointKinematics, 2> points{};
        std::vector<double> jacobians(
            queries.size() * 3u * model.world.nv,
            0.0
        );
        const auto kinematics =
            metalrobo::computeArticulatedPointJacobians(
                model,
                0u,
                q,
                zeroVelocity,
                queries,
                points,
                jacobians
            );
        require(
            kinematics.succeeded(),
            "surgical PSM jaw-gap kinematics failed"
        );
        const double surfaceGap =
            distance(points[0].position, points[1].position) -
            static_cast<double>(tipOne.dimensions.x) -
            static_cast<double>(tipTwo.dimensions.x);
        if (sample == 0u) {
            gate.closedSurfaceGap = surfaceGap;
        } else {
            gate.minimumGapIncrease = std::min(
                gate.minimumGapIncrease,
                surfaceGap - previousGap
            );
        }
        previousGap = surfaceGap;
        gate.maximumSurfaceGap = surfaceGap;
    }
    return gate;
}

struct MetalParity {
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
            "surgical PSM CPU ABA reference failed"
        );
        for (std::size_t dof = 0u;
             dof < articulation.nv;
             ++dof) {
            const double error = std::abs(
                static_cast<double>(
                    result.acceleration[vBase + dof]
                ) - acceleration[dof]
            );
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
            "surgical PSM CPU configuration integration failed"
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
        const metalrobo::SurgicalPSMModelMetadata& metadata =
            metalrobo::surgicalPSMMetadata();
        const metalrobo::EngineModel model =
            metalrobo::makeDvrkPsmLargeNeedleDriverEngineModel();
        std::string reason;
        require(
            model.valid(&reason),
            "EngineModel::valid rejected surgical PSM: " + reason
        );
        require(
            model.world.abiVersion == MR_ENGINE_ABI_VERSION &&
                model.world.bodyCount ==
                    metalrobo::kSurgicalPSMBodyCount &&
                model.world.jointCount ==
                    metalrobo::kSurgicalPSMJointCount &&
                model.world.shapeCount ==
                    metalrobo::kSurgicalPSMShapeCount &&
                model.world.nq == metalrobo::kSurgicalPSMJointCount &&
                model.world.nv == metalrobo::kSurgicalPSMJointCount &&
                model.articulations.size() == 1u &&
                model.articulations[0].rootType == MR_ROOT_FIXED,
            "surgical PSM topology/counts are wrong"
        );
        require(
            metadata.orbitCommit ==
                    "6e47534f7d412e4be523116f250c992a63146883" &&
                metadata.orbitLicense == "BSD-3-Clause" &&
                metadata.dvrkCommit ==
                    "53a401d014e5ef8a7d5e3ad05f0680084507662c" &&
                metadata.dvrkLicenseCommit ==
                    "7e95680b9461009b745567f382d1b498eabc046b" &&
                metadata.toolModelNumber == "400006" &&
                close(metadata.orbitToolYawLinkMass, 0.1) &&
                close(metadata.orbitFixedToolTipMass, 0.1) &&
                metadata.independentJawCoordinates &&
                metadata.fixedToolTipMassFoldedIntoYaw &&
                !metadata.calibratedInertias &&
                !metadata.clinicallyValidated,
            "surgical PSM provenance/fidelity metadata changed"
        );
        require(
            close(model.bodies[6].massAndInverseMass.x, 0.2) &&
                close(model.bodies[6].massAndInverseMass.y, 5.0),
            "fixed ORBIT tooltip mass was not folded into yaw"
        );

        const auto logicalDefaultFloat =
            metalrobo::surgicalPSMDefaultLogicalPositionTargets();
        const std::vector<double> logicalDefault(
            logicalDefaultFloat.begin(),
            logicalDefaultFloat.end()
        );
        std::vector<double> mappedTargets{91.0, 92.0, 93.0};
        const auto defaultMap =
            metalrobo::expandSurgicalPSMLogicalPositionTargets(
                model,
                logicalDefault,
                mappedTargets
            );
        require(
            defaultMap.succeeded() &&
                mappedTargets.size() == model.defaultQ.size() &&
                close(
                    logicalDefault[
                        metalrobo::kSurgicalPSMLogicalJawApertureIndex
                    ],
                    static_cast<double>(model.defaultQ[7]) -
                        model.defaultQ[6]
                ),
            "surgical PSM logical default map failed"
        );
        for (std::size_t index = 0u;
             index < mappedTargets.size();
             ++index) {
            require(
                close(mappedTargets[index], model.defaultQ[index]),
                "surgical PSM logical default did not round-trip"
            );
        }

        std::vector<double> endpointLogical = logicalDefault;
        endpointLogical[
            metalrobo::kSurgicalPSMLogicalJawApertureIndex
        ] = 0.0;
        const auto closedMap =
            metalrobo::expandSurgicalPSMLogicalPositionTargets(
                model,
                endpointLogical,
                mappedTargets
            );
        require(
            closedMap.succeeded() &&
                mappedTargets[6] == 0.0 &&
                mappedTargets[7] == 0.0,
            "surgical PSM closed-jaw endpoint map failed"
        );

        endpointLogical[
            metalrobo::kSurgicalPSMLogicalJawApertureIndex
        ] = defaultMap.maximumJawAperture;
        const auto openMap =
            metalrobo::expandSurgicalPSMLogicalPositionTargets(
                model,
                endpointLogical,
                mappedTargets
            );
        require(
            openMap.succeeded() &&
                close(mappedTargets[6], model.dofs[6].limits.x) &&
                close(mappedTargets[7], model.dofs[7].limits.y),
            "surgical PSM open-jaw endpoint map failed"
        );
        const JawGapGate jawGap = verifyJawApertureGeometry(
            model,
            defaultMap.maximumJawAperture
        );
        require(
            std::abs(jawGap.closedSurfaceGap) < 5.0e-7 &&
                jawGap.minimumGapIncrease > 1.0e-4 &&
                jawGap.maximumSurfaceGap > 0.015,
            "surgical PSM jaw aperture is not physically monotonic"
        );

        const std::vector<double> sentinelTargets{
            101.0, 102.0, 103.0,
        };
        const auto requireTransactionalRejection =
            [&](const std::span<const double> logical,
                const metalrobo::SurgicalPSMCommandMapStatus expected) {
                std::vector<double> output = sentinelTargets;
                const auto diagnostics =
                    metalrobo::expandSurgicalPSMLogicalPositionTargets(
                        model,
                        logical,
                        output
                    );
                require(
                    diagnostics.status == expected &&
                        output == sentinelTargets,
                    "surgical PSM logical rejection was not transactional"
                );
            };

        requireTransactionalRejection(
            std::span<const double>{
                logicalDefault.data(),
                logicalDefault.size() - 1u,
            },
            metalrobo::SurgicalPSMCommandMapStatus::invalidDimensions
        );
        std::vector<double> invalidLogical = logicalDefault;
        invalidLogical[3] =
            std::numeric_limits<double>::quiet_NaN();
        requireTransactionalRejection(
            invalidLogical,
            metalrobo::SurgicalPSMCommandMapStatus::nonfiniteTarget
        );
        invalidLogical = logicalDefault;
        invalidLogical[
            metalrobo::kSurgicalPSMLogicalJawApertureIndex
        ] = -0.001;
        requireTransactionalRejection(
            invalidLogical,
            metalrobo::SurgicalPSMCommandMapStatus::negativeJawAperture
        );
        invalidLogical = logicalDefault;
        invalidLogical[
            metalrobo::kSurgicalPSMLogicalJawApertureIndex
        ] = defaultMap.maximumJawAperture + 1.0e-4;
        requireTransactionalRejection(
            invalidLogical,
            metalrobo::SurgicalPSMCommandMapStatus::
                physicalLimitViolation
        );
        invalidLogical = logicalDefault;
        invalidLogical[0] =
            static_cast<double>(model.dofs[0].limits.y) + 1.0e-4;
        requireTransactionalRejection(
            invalidLogical,
            metalrobo::SurgicalPSMCommandMapStatus::
                physicalLimitViolation
        );
        require(
            std::string{
                metalrobo::surgicalPSMCommandMapStatusName(
                    metalrobo::SurgicalPSMCommandMapStatus::
                        negativeJawAperture
                )
            } == "negative_jaw_aperture",
            "surgical PSM command-map status names changed"
        );

        constexpr mr_u32 expectedDofFlags =
            MR_DOF_FLAG_ACTUATED |
            MR_DOF_FLAG_POSITION_LIMIT |
            MR_DOF_FLAG_VELOCITY_LIMIT |
            MR_DOF_FLAG_EFFORT_LIMIT |
            MR_DOF_FLAG_DRIVE;
        std::size_t prismaticCount = 0u;
        for (std::size_t index = 0u;
             index < model.joints.size();
             ++index) {
            const MRJointDescriptorGPU& joint = model.joints[index];
            const MRDofPropertiesGPU& dof = model.dofs[index];
            const auto& source = metadata.joints[index];
            const MRBodyPropertiesGPU& child =
                model.bodies[joint.childBody];
            prismaticCount +=
                joint.jointType == MR_JOINT_PRISMATIC ? 1u : 0u;
            require(
                joint.jointType == source.jointType &&
                    joint.parentBody == source.parentBody &&
                    joint.childBody == source.childBody &&
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
                    ) &&
                    dof.flags == expectedDofFlags &&
                    close(dof.limits.x, source.lowerPosition) &&
                    close(dof.limits.y, source.upperPosition) &&
                    close(dof.limits.z, source.maximumVelocity) &&
                    close(dof.limits.w, source.maximumEffort) &&
                    close(dof.drive.x, source.stiffness) &&
                    close(dof.drive.y, source.damping) &&
                    close(dof.drive.z, source.armature) &&
                    close(dof.drive.w, 0.0),
                "surgical PSM joint/COM/drive compilation changed"
            );
        }
        require(
            prismaticCount == 1u &&
                model.joints[2].jointType == MR_JOINT_PRISMATIC,
            "surgical PSM does not retain its true insertion joint"
        );

        std::array<std::size_t, metalrobo::kSurgicalPSMBodyCount>
            bodyShapeCounts{};
        for (const MRShapeGPU& shape : model.shapes) {
            require(
                shape.flags == 0u &&
                    shape.collisionGroup == (1u << 8u) &&
                    (shape.collisionMask & (1u << 8u)) == 0u &&
                    (shape.collisionMask & 1u) != 0u,
                "surgical PSM self-collision mask changed"
            );
            ++bodyShapeCounts[shape.bodyIndex];
        }
        require(
            std::ranges::all_of(
                bodyShapeCounts,
                [](const std::size_t count) {
                    return count >= 2u;
                }
            ) &&
                model.shapes[6].bodyIndex == 3u &&
                model.shapes[6].shapeType == MR_SHAPE_CAPSULE &&
                close(
                    2.0 * (
                        model.shapes[6].dimensions.x +
                        model.shapes[6].dimensions.y
                    ),
                    metadata.classicShaftLength
                ),
            "surgical PSM compound primitive geometry changed"
        );

        const RcmGate rcm = verifyRemoteCenter(model, metadata);
        require(
            rcm.radialError < 2.0e-8 &&
                rcm.centerDrift < 2.0e-10 &&
                rcm.insertionDeltaError < 2.0e-8,
            "surgical PSM serial equivalent violated its remote center"
        );

        std::vector<double> q(
            model.defaultQ.begin(),
            model.defaultQ.end()
        );
        std::vector<double> v(model.world.nv, 0.0);
        std::vector<double> expectedAcceleration(model.world.nv, 0.0);
        for (std::size_t index = 0u; index < model.world.nv; ++index) {
            const double scale = index == 2u ? 0.01 : 0.05;
            q[index] += scale * std::sin(0.6 * (index + 1u));
            v[index] =
                (index == 2u ? 0.02 : 0.08) *
                std::cos(0.4 * (index + 1u));
            expectedAcceleration[index] =
                0.3 * std::sin(0.8 * (index + 1u));
        }
        std::vector<double> generalizedForce(model.world.nv, 0.0);
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
            "surgical PSM FP64 inverse dynamics failed"
        );
        std::vector<double> recoveredAcceleration(model.world.nv, 0.0);
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
            "surgical PSM FP64 forward dynamics failed"
        );
        double forwardInverseError = 0.0;
        for (std::size_t index = 0u; index < model.world.nv; ++index) {
            forwardInverseError = std::max(
                forwardInverseError,
                std::abs(
                    recoveredAcceleration[index] -
                    expectedAcceleration[index]
                )
            );
        }
        require(
            forwardInverseError < 2.0e-9,
            "surgical PSM forward/inverse consistency regressed"
        );

        const std::vector<double> resetQ(
            model.defaultQ.begin(),
            model.defaultQ.end()
        );
        const std::vector<double> resetV(model.world.nv, 0.0);
        std::vector<metalrobo::ArticulatedDofCommand> commands(
            model.world.nv
        );
        for (std::size_t index = 0u; index < commands.size(); ++index) {
            commands[index].mode =
                metalrobo::ArticulatedActuationMode::modelPD;
            commands[index].desiredPosition =
                resetQ[index] + (index == 2u ? 0.001 : 0.001);
        }
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
                actuation.generalizedEffort.size() == model.world.nv &&
                actuationDiagnostics.maximumActuatorEffort > 0.0,
            "surgical PSM actuator preset is not executable"
        );

        std::vector<double> limitQ = resetQ;
        limitQ[2] =
            static_cast<double>(model.dofs[2].limits.y) - 5.0e-4;
        std::vector<double> freeVelocity(model.world.nv, 0.0);
        freeVelocity[2] = 1.0;
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
                limitRows.size() == 1u &&
                limitRows.front().side ==
                    metalrobo::ArticulatedJointLimitSide::upper &&
                limitRows.front().localQIndex == 2u &&
                limitRows.front().localVIndex == 2u,
            "surgical PSM prismatic stop did not compile"
        );

        std::vector<double> worldQ = resetQ;
        std::vector<double> worldV = resetV;
        const std::vector<double> zeroForce(model.world.nv, 0.0);
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
                worldCache.step == 1u,
            "surgical PSM default world step failed"
        );

        constexpr std::size_t environmentCount = 3u;
        std::vector<float> metalQ(
            environmentCount * model.world.nq
        );
        std::vector<float> metalV(
            environmentCount * model.world.nv
        );
        std::vector<float> metalEffort(
            environmentCount * model.world.nv
        );
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            for (std::size_t dof = 0u;
                 dof < model.world.nv;
                 ++dof) {
                const std::size_t index =
                    environment * model.world.nv + dof;
                const float qScale = dof == 2u ? 0.004f : 0.012f;
                metalQ[index] =
                    model.defaultQ[dof] +
                    qScale * static_cast<float>(
                        std::sin(0.35 * (index + 1u))
                    );
                metalV[index] =
                    static_cast<float>(
                        (dof == 2u ? 0.01 : 0.04) *
                        std::cos(0.29 * (index + 1u))
                    );
                metalEffort[index] =
                    static_cast<float>(
                        0.025 * std::sin(0.21 * (index + 1u))
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
            "surgical PSM public Metal ABA failed: " +
                metalDiagnostics.message
        );
        const MetalParity parity =
            compareMetalABA(model, metalInput, metalResult);
        require(
            parity.accelerationScaled < 2.0e-4 &&
                parity.nextV < 2.0e-4 &&
                parity.nextQ < 2.0e-4,
            "surgical PSM Metal/FP64 parity regressed"
        );

        std::cout
            << "surgical_psm=abi_v2"
            << " model=\"" << model.name << "\""
            << " bodies=" << model.world.bodyCount
            << " dofs=" << model.world.nv
            << " shapes=" << model.world.shapeCount
            << " insertion_joint=prismatic"
            << " folded_tip_mass="
            << metadata.orbitFixedToolTipMass
            << " rcm_radial_error=" << rcm.radialError
            << " rcm_drift=" << rcm.centerDrift
            << " insertion_delta_error="
            << rcm.insertionDeltaError
            << " forward_inverse_error=" << forwardInverseError
            << " metal_acceleration_scaled_error="
            << parity.accelerationScaled
            << " metal_next_v_error=" << parity.nextV
            << " metal_next_q_error=" << parity.nextQ
            << " device=\"" << metalDiagnostics.deviceName << "\""
            << " jaw_aperture_max="
            << defaultMap.maximumJawAperture
            << " jaw_closed_surface_gap="
            << jawGap.closedSurfaceGap
            << " jaw_open_surface_gap="
            << jawGap.maximumSurfaceGap
            << " jaw_gap_step_min="
            << jawGap.minimumGapIncrease
            << " logical_jaw_map=pass"
            << " actuation=pass"
            << " limits=pass"
            << " world=pass"
            << " compound_collision=pass"
            << " independent_jaws=research"
            << " clinical_validation=none"
            << " status=ok\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "surgical_psm=failed reason=\""
            << exception.what() << "\"\n";
        return 1;
    }
}
