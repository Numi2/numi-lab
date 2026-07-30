#include "metalrobo/ArticulatedCollision.hpp"
#include "metalrobo/Collision.hpp"
#include "metalrobo/G1.hpp"
#include "metalrobo/QualityContactSolver.hpp"
#include "metalrobo/RigidBodyWorld.hpp"

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
#include <utility>
#include <vector>

namespace {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

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

Vec3 operator+(const Vec3 left, const Vec3 right) {
    return {
        left.x + right.x,
        left.y + right.y,
        left.z + right.z,
    };
}

Vec3 operator-(const Vec3 value) {
    return {-value.x, -value.y, -value.z};
}

Vec3 operator-(const Vec3 left, const Vec3 right) {
    return {
        left.x - right.x,
        left.y - right.y,
        left.z - right.z,
    };
}

Vec3 operator*(const Vec3 value, const double scale) {
    return {
        value.x * scale,
        value.y * scale,
        value.z * scale,
    };
}

double dot(const Vec3 left, const Vec3 right) {
    return
        left.x * right.x +
        left.y * right.y +
        left.z * right.z;
}

Vec3 cross(const Vec3 left, const Vec3 right) {
    return {
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
    };
}

double norm(const Vec3 value) {
    return std::sqrt(dot(value, value));
}

Vec3 normalized(const Vec3 value) {
    return value * (1.0 / norm(value));
}

Vec3 xyz(const mr_float4 value) {
    return {value.x, value.y, value.z};
}

Vec3 pointVelocity(
    const MRBodyStateGPU& body,
    const Vec3 point
) {
    return
        xyz(body.linearVelocityAndInverseMass) +
        cross(
            xyz(body.angularVelocity),
            point - xyz(body.position)
        );
}

std::pair<Vec3, Vec3> contactBasis(const Vec3 normal) {
    const Vec3 absolute{
        std::abs(normal.x),
        std::abs(normal.y),
        std::abs(normal.z),
    };
    Vec3 reference{};
    if (absolute.x <= absolute.y && absolute.x <= absolute.z) {
        reference = {1.0, 0.0, 0.0};
    } else if (absolute.y <= absolute.z) {
        reference = {0.0, 1.0, 0.0};
    } else {
        reference = {0.0, 0.0, 1.0};
    }
    const Vec3 tangentU = normalized(cross(reference, normal));
    return {tangentU, cross(normal, tangentU)};
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

double contactError(
    const metalrobo::ArticulatedContact& left,
    const metalrobo::ArticulatedContact& right
) {
    if (left.bodyA != right.bodyA ||
        left.bodyB != right.bodyB) {
        return std::numeric_limits<double>::infinity();
    }
    return std::max({
        maximumError(left.localPointA, right.localPointA),
        maximumError(left.localPointB, right.localPointB),
        maximumError(left.normal, right.normal),
        maximumError(left.tangentU, right.tangentU),
        maximumError(left.tangentV, right.tangentV),
        maximumError(
            left.targetVelocity,
            right.targetVelocity
        ),
        maximumError(left.regularization, right.regularization),
        maximumError(left.warmImpulse, right.warmImpulse),
        std::abs(left.friction - right.friction),
    });
}

std::vector<MRBodyStateGPU> makeCollisionStates(
    const metalrobo::EngineModel& model,
    const std::span<const metalrobo::ArticulatedBodyKinematics>
        kinematics
) {
    require(
        kinematics.size() == model.bodies.size(),
        "body kinematics/state count mismatch"
    );
    std::vector<MRBodyStateGPU> states(model.bodies.size() + 1u);
    for (std::size_t body = 0u; body < model.bodies.size(); ++body) {
        const auto& source = kinematics[body];
        const auto& properties = model.bodies[body];
        MRBodyStateGPU& state = states[body];
        state.position = f4(
            source.centerOfMassPosition[0],
            source.centerOfMassPosition[1],
            source.centerOfMassPosition[2],
            1.0
        );
        state.orientation = f4(
            source.orientation[0],
            source.orientation[1],
            source.orientation[2],
            source.orientation[3]
        );
        state.linearVelocityAndInverseMass = f4(
            source.linearVelocity[0],
            source.linearVelocity[1],
            source.linearVelocity[2],
            properties.massAndInverseMass.y
        );
        state.angularVelocity = f4(
            source.angularVelocity[0],
            source.angularVelocity[1],
            source.angularVelocity[2]
        );
        state.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
        state.flagsAndIndices[1] = 0u;
        state.flagsAndIndices[2] =
            static_cast<std::uint32_t>(body);
    }
    MRBodyStateGPU& ground = states.back();
    ground.position = f4(0.0, 0.0, 0.0, 1.0);
    ground.orientation = f4(0.0, 0.0, 0.0, 1.0);
    ground.flagsAndIndices[0] = MR_MOTION_STATIC;
    ground.flagsAndIndices[1] = MR_INVALID_INDEX;
    ground.flagsAndIndices[2] = MR_INVALID_INDEX;
    return states;
}

MRShapeGPU makeZUpGroundPlane(const std::uint32_t bodyIndex) {
    constexpr double sineHalfQuarterTurn =
        0.7071067811865475244;
    MRShapeGPU plane{};
    plane.bodyIndex = bodyIndex;
    plane.shapeType = MR_SHAPE_PLANE;
    plane.materialIndex = 0u;
    plane.collisionGroup = 1u;
    plane.collisionMask = ~0u;
    plane.slotGeneration = 9001u;
    plane.localPosition = f4(0.0, 0.0, 0.0, 1.0);
    // Collision planes are +Y in local axes. +90 degrees about X maps +Y
    // onto world +Z.
    plane.localRotation = f4(
        sineHalfQuarterTurn,
        0.0,
        0.0,
        sineHalfQuarterTurn
    );
    return plane;
}

} // namespace

int main() {
    try {
        const metalrobo::EngineModel model =
            metalrobo::makeUnitreeG1EngineModel();
        std::vector<double> q(
            model.defaultQ.begin(),
            model.defaultQ.end()
        );
        std::vector<double> zeroVelocity(
            model.articulations[0].nv,
            0.0
        );
        metalrobo::ArticulatedDynamicsConfig dynamicsConfig;
        dynamicsConfig.gravity = {0.0, 0.0, 0.0};
        dynamicsConfig.applyBodyDamping = false;

        // Determine the current sole clearance through the engine's analytic
        // point kinematics, then lower only the floating root enough to create
        // a controlled 0.5 mm overlap.
        std::vector<metalrobo::ArticulatedPointQuery> sphereQueries;
        std::vector<double> sphereRadii;
        for (const MRShapeGPU& shape : model.shapes) {
            if (
                (shape.flags & MR_SHAPE_FLAG_SIMULATION_DISABLED) == 0u &&
                shape.shapeType == MR_SHAPE_SPHERE
            ) {
                sphereQueries.push_back({
                    shape.bodyIndex,
                    {
                        shape.localPosition.x,
                        shape.localPosition.y,
                        shape.localPosition.z,
                    },
                });
                sphereRadii.push_back(shape.dimensions.x);
            }
        }
        require(
            sphereQueries.size() ==
                metalrobo::kUnitreeG1FootSphereCount,
            "G1 executable foot-sphere set changed"
        );
        std::vector<metalrobo::ArticulatedPointKinematics>
            defaultSphereKinematics(sphereQueries.size());
        std::vector<double> sphereJacobians(
            sphereQueries.size() * 3u * zeroVelocity.size(),
            0.0
        );
        auto dynamicsDiagnostics =
            metalrobo::computeArticulatedPointJacobians(
                model,
                0u,
                q,
                zeroVelocity,
                sphereQueries,
                defaultSphereKinematics,
                sphereJacobians,
                dynamicsConfig
            );
        require(
            dynamicsDiagnostics.succeeded(),
            "default G1 sphere kinematics failed"
        );
        double defaultMinimumBottom =
            std::numeric_limits<double>::infinity();
        for (std::size_t sphere = 0u;
             sphere < sphereQueries.size();
             ++sphere) {
            defaultMinimumBottom = std::min(
                defaultMinimumBottom,
                defaultSphereKinematics[sphere].position[2] -
                    sphereRadii[sphere]
            );
        }
        constexpr double desiredPenetration = 5.0e-4;
        const double rootLowering =
            defaultMinimumBottom + desiredPenetration;
        q[2] -= rootLowering;

        std::vector<double> freeVelocity(
            model.articulations[0].nv,
            0.0
        );
        freeVelocity[2] = -0.25;
        std::vector<metalrobo::ArticulatedBodyKinematics>
            bodyKinematics(model.bodies.size());
        dynamicsDiagnostics =
            metalrobo::computeArticulatedBodyKinematics(
                model,
                0u,
                q,
                freeVelocity,
                bodyKinematics,
                dynamicsConfig
            );
        require(
            dynamicsDiagnostics.succeeded(),
            "lowered G1 body kinematics failed"
        );
        std::vector<MRBodyStateGPU> states =
            makeCollisionStates(model, bodyKinematics);

        std::vector<MRShapeGPU> shapes;
        shapes.reserve(
            metalrobo::kUnitreeG1FootSphereCount + 1u
        );
        for (const MRShapeGPU& shape : model.shapes) {
            if (
                (shape.flags &
                 MR_SHAPE_FLAG_SIMULATION_DISABLED) == 0u &&
                shape.shapeType == MR_SHAPE_SPHERE
            ) {
                shapes.push_back(shape);
            }
        }
        shapes.push_back(makeZUpGroundPlane(
            static_cast<std::uint32_t>(states.size() - 1u)
        ));
        metalrobo::CollisionConfig collisionConfig;
        collisionConfig.capacities = {
            .pairCapacity = 64u,
            .rawContactCapacity = 64u,
            .manifoldCapacity = 64u,
        };
        metalrobo::PersistentManifoldCache manifoldCache;
        const metalrobo::CollisionFrame collision =
            metalrobo::collideCpuReference(
                shapes,
                states,
                collisionConfig,
                manifoldCache
            );
        require(
            collision.succeeded(),
            "G1 foot/ground CPU collision failed"
        );
        const metalrobo::ContactAssemblyResult assembly =
            metalrobo::assembleContactConstraints(
                collision,
                shapes,
                model.materials,
                states,
                32u
            );
        require(
            assembly.diagnostics.succeeded() &&
                !assembly.constraints.empty(),
            "G1 collision manifold/material assembly failed"
        );

        metalrobo::ArticulatedCollisionAdapterConfig adapterConfig;
        adapterConfig.contact.timestep = 1.0 / 240.0;
        adapterConfig.contact.errorReduction = 0.2;
        adapterConfig.contact.penetrationSlop = 1.0e-4;
        adapterConfig.contact.maxDepenetrationVelocity = 2.0;
        adapterConfig.qualityTangentialRegularization = 1.0e-9;
        adapterConfig.contactCapacity = 32u;
        const metalrobo::ArticulatedCollisionResult adaptation =
            metalrobo::adaptArticulatedContactConstraints(
                model,
                0u,
                assembly.constraints,
                states,
                adapterConfig
            );
        require(
            adaptation.succeeded() &&
                adaptation.contacts.size() ==
                    assembly.constraints.size(),
            "G1 common-to-articulated contact adaptation failed"
        );

        metalrobo::ArticulatedContactProblem problem;
        auto contactDiagnostics =
            metalrobo::buildArticulatedContactProblem(
                model,
                0u,
                q,
                freeVelocity,
                adaptation.contacts,
                problem,
                dynamicsConfig,
                true
            );
        require(
            contactDiagnostics.succeeded(),
            "G1 collision-produced generalized problem failed"
        );
        std::vector<double> freeContactVelocity(
            3u * adaptation.contacts.size(),
            0.0
        );
        contactDiagnostics =
            metalrobo::applyArticulatedContactJacobian(
                problem,
                freeVelocity,
                freeContactVelocity
            );
        require(
            contactDiagnostics.succeeded(),
            "G1 free contact-velocity action failed"
        );
        double reconstructedPointError = 0.0;
        double freeJvParityError = 0.0;
        for (std::size_t contact = 0u;
             contact < adaptation.contacts.size();
             ++contact) {
            const std::uint32_t sourceIndex =
                adaptation.sourceConstraintIndices[contact];
            require(
                sourceIndex < assembly.constraints.size(),
                "adapted source-constraint index is invalid"
            );
            const MRContactConstraintGPU& source =
                assembly.constraints[sourceIndex];
            const Vec3 point = xyz(source.pointAndSeparation);
            const auto& reconstructedA = problem.pointA[contact];
            const auto& reconstructedB = problem.pointB[contact];
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                reconstructedPointError = std::max({
                    reconstructedPointError,
                    std::abs(
                        reconstructedA.position[axis] -
                        (
                            axis == 0u ? point.x :
                            axis == 1u ? point.y :
                                         point.z
                        )
                    ),
                    std::abs(
                        reconstructedB.position[axis] -
                        (
                            axis == 0u ? point.x :
                            axis == 1u ? point.y :
                                         point.z
                        )
                    ),
                });
            }
            const Vec3 relativeVelocity =
                pointVelocity(states[source.bodyB], point) -
                pointVelocity(states[source.bodyA], point);
            const Vec3 normal{
                adaptation.contacts[contact].normal[0],
                adaptation.contacts[contact].normal[1],
                adaptation.contacts[contact].normal[2],
            };
            const Vec3 tangentU{
                adaptation.contacts[contact].tangentU[0],
                adaptation.contacts[contact].tangentU[1],
                adaptation.contacts[contact].tangentU[2],
            };
            const Vec3 tangentV{
                adaptation.contacts[contact].tangentV[0],
                adaptation.contacts[contact].tangentV[1],
                adaptation.contacts[contact].tangentV[2],
            };
            const std::array<Vec3, 3> directions{
                normal,
                tangentU,
                tangentV,
            };
            for (std::size_t row = 0u; row < 3u; ++row) {
                freeJvParityError = std::max(
                    freeJvParityError,
                    std::abs(
                        freeContactVelocity[3u * contact + row] -
                        dot(relativeVelocity, directions[row])
                    )
                );
            }
        }
        require(
            reconstructedPointError < 2.0e-7 &&
                freeJvParityError < 2.0e-7,
            "adapted local anchor/Jv does not reconstruct common contact"
        );
        metalrobo::QualityContactSolverConfig qualityConfig;
        qualityConfig.maximumIterations = 400u;
        qualityConfig.kktTolerance = 1.0e-10;
        const metalrobo::QualityContactSolution solution =
            metalrobo::solveQualityContactProblem(
                problem.conic,
                qualityConfig
            );
        require(
            solution.converged(),
            "G1 collision-produced quality contact solve failed"
        );
        std::vector<double> solvedContactVelocity(
            3u * adaptation.contacts.size(),
            0.0
        );
        contactDiagnostics =
            metalrobo::applyArticulatedContactJacobian(
                problem,
                solution.velocity,
                solvedContactVelocity
            );
        require(
            contactDiagnostics.succeeded(),
            "G1 solved contact-velocity action failed"
        );
        double minimumSolvedNormalVelocity =
            std::numeric_limits<double>::infinity();
        for (std::size_t contact = 0u;
             contact < adaptation.contacts.size();
             ++contact) {
            minimumSolvedNormalVelocity = std::min(
                minimumSolvedNormalVelocity,
                solvedContactVelocity[3u * contact]
            );
        }
        require(
            minimumSolvedNormalVelocity >= -2.0e-9,
            "G1 solved foot contact is still approaching"
        );

        std::vector<double> operatorVelocity = freeVelocity;
        contactDiagnostics =
            metalrobo::applyArticulatedContactImpulses(
                problem,
                solution.impulses,
                operatorVelocity
            );
        const double operatorVelocityError = maximumError(
            operatorVelocity,
            solution.velocity
        );
        require(
            contactDiagnostics.succeeded() &&
                operatorVelocityError < 2.0e-12,
            "G1 exact operator update disagrees with solver velocity"
        );

        // A physically equivalent common record with endpoints reversed must
        // adapt to the identical generalized record, including nonzero
        // tangent warm impulses and surface targets.
        MRContactConstraintGPU forward =
            assembly.constraints.front();
        forward.targetVelocityAndPreSolveNormal.x = 0.02f;
        forward.targetVelocityAndPreSolveNormal.y = -0.01f;
        forward.targetVelocityAndPreSolveNormal.z = 0.04f;
        forward.impulses = f4(0.4, 0.05, -0.03, 0.0);
        forward.response.x = 0.4f;
        forward.response.y = 0.1f;
        forward.response.z = 2.0e-6f;
        const auto forwardAdaptation =
            metalrobo::adaptArticulatedContactConstraints(
                model,
                0u,
                std::span(&forward, 1u),
                states,
                adapterConfig
            );
        require(
            forwardAdaptation.succeeded(),
            "forward endpoint semantics adaptation failed"
        );
        const Vec3 forwardNormal = normalized(xyz(forward.normal));
        const Vec3 forwardSurfaceTarget =
            xyz(forward.targetVelocityAndPreSolveNormal);
        const double penetrationTarget = std::min(
            adapterConfig.contact.maxDepenetrationVelocity,
            -adapterConfig.contact.errorReduction *
                std::min(
                    static_cast<double>(
                        forward.pointAndSeparation.w
                    ) +
                        adapterConfig.contact.penetrationSlop,
                    0.0
                ) /
                adapterConfig.contact.timestep
        );
        const double restitutionTarget =
            -static_cast<double>(forward.response.x) *
            forward.targetVelocityAndPreSolveNormal.w;
        const double expectedForwardNormalTarget =
            dot(forwardSurfaceTarget, forwardNormal) +
            std::max(penetrationTarget, restitutionTarget);
        const double targetRuleError = std::abs(
            forwardAdaptation.contacts[0].targetVelocity[0] -
            expectedForwardNormalTarget
        );
        const double expectedNormalRegularization =
            static_cast<double>(forward.response.z) /
                (
                    adapterConfig.contact.timestep *
                    adapterConfig.contact.timestep
                ) +
            adapterConfig.qualityTangentialRegularization;
        const double complianceRegularizationError = std::abs(
            forwardAdaptation.contacts[0].regularization[0] -
            expectedNormalRegularization
        );
        require(
            targetRuleError < 2.0e-8 &&
                complianceRegularizationError < 2.0e-15,
            "shared target/compliance conversion rules regressed"
        );

        MRContactConstraintGPU reversed = forward;
        std::swap(reversed.bodyA, reversed.bodyB);
        reversed.normal = f4(
            -forwardNormal.x,
            -forwardNormal.y,
            -forwardNormal.z
        );
        reversed.targetVelocityAndPreSolveNormal.x =
            -forward.targetVelocityAndPreSolveNormal.x;
        reversed.targetVelocityAndPreSolveNormal.y =
            -forward.targetVelocityAndPreSolveNormal.y;
        reversed.targetVelocityAndPreSolveNormal.z =
            -forward.targetVelocityAndPreSolveNormal.z;
        const auto [forwardU, forwardV] =
            contactBasis(forwardNormal);
        const Vec3 forwardImpulse =
            forwardNormal * forward.impulses.x +
            forwardU * forward.impulses.y +
            forwardV * forward.impulses.z;
        const Vec3 reversedNormal = -forwardNormal;
        const auto [reversedU, reversedV] =
            contactBasis(reversedNormal);
        const Vec3 reversedImpulse = -forwardImpulse;
        reversed.impulses = f4(
            dot(reversedImpulse, reversedNormal),
            dot(reversedImpulse, reversedU),
            dot(reversedImpulse, reversedV),
            0.0
        );
        const auto reversedAdaptation =
            metalrobo::adaptArticulatedContactConstraints(
                model,
                0u,
                std::span(&reversed, 1u),
                states,
                adapterConfig
            );
        const double endpointSwapError =
            reversedAdaptation.succeeded()
            ? contactError(
                forwardAdaptation.contacts[0],
                reversedAdaptation.contacts[0]
            )
            : std::numeric_limits<double>::infinity();
        require(
            reversedAdaptation.succeeded() &&
                reversedAdaptation.diagnostics
                    .swappedEndpointCount == 1u &&
                endpointSwapError < 2.0e-7,
            "endpoint swap/sign/basis semantics changed the contact"
        );

        // Prescribed kinematic velocity is removed from the generalized
        // target rather than silently treated as zero.
        std::vector<MRBodyStateGPU> kinematicStates = states;
        MRBodyStateGPU& movingGround =
            kinematicStates[forward.bodyB];
        movingGround.flagsAndIndices[0] = MR_MOTION_KINEMATIC;
        movingGround.linearVelocityAndInverseMass =
            f4(0.07, -0.04, 0.12, 0.0);
        const auto kinematicAdaptation =
            metalrobo::adaptArticulatedContactConstraints(
                model,
                0u,
                std::span(&forward, 1u),
                kinematicStates,
                adapterConfig
            );
        require(
            kinematicAdaptation.succeeded(),
            "kinematic target compensation adaptation failed"
        );
        const Vec3 groundVelocity{0.07, -0.04, 0.12};
        const Vec3 adaptedNormal = {
            forwardAdaptation.contacts[0].normal[0],
            forwardAdaptation.contacts[0].normal[1],
            forwardAdaptation.contacts[0].normal[2],
        };
        const Vec3 adaptedU = {
            forwardAdaptation.contacts[0].tangentU[0],
            forwardAdaptation.contacts[0].tangentU[1],
            forwardAdaptation.contacts[0].tangentU[2],
        };
        const Vec3 adaptedV = cross(adaptedNormal, adaptedU);
        const std::array<double, 3> expectedKinematicShift{
            -dot(groundVelocity, adaptedNormal),
            -dot(groundVelocity, adaptedU),
            -dot(groundVelocity, adaptedV),
        };
        std::array<double, 3> measuredKinematicShift{};
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            measuredKinematicShift[axis] =
                kinematicAdaptation.contacts[0]
                    .targetVelocity[axis] -
                forwardAdaptation.contacts[0]
                    .targetVelocity[axis];
        }
        const double kinematicCompensationError = maximumError(
            measuredKinematicShift,
            expectedKinematicShift
        );
        require(
            kinematicCompensationError < 2.0e-8,
            "kinematic point velocity compensation is incorrect"
        );

        auto capacityConfig = adapterConfig;
        capacityConfig.contactCapacity = 0u;
        const auto capacityFailure =
            metalrobo::adaptArticulatedContactConstraints(
                model,
                0u,
                std::span(&forward, 1u),
                states,
                capacityConfig
            );
        require(
            !capacityFailure.succeeded() &&
                capacityFailure.diagnostics.failure ==
                    metalrobo::ArticulatedCollisionFailure::
                        capacityOverflow &&
                capacityFailure.diagnostics.requiredContactCount ==
                    1u &&
                capacityFailure.contacts.empty(),
            "adapter capacity failure was not transactional"
        );

        auto tinyTimestepConfig = adapterConfig;
        tinyTimestepConfig.contact.timestep =
            std::numeric_limits<double>::denorm_min();
        const auto tinyTimestepFailure =
            metalrobo::adaptArticulatedContactConstraints(
                model,
                0u,
                std::span(&forward, 1u),
                states,
                tinyTimestepConfig
            );
        require(
            !tinyTimestepFailure.succeeded() &&
                tinyTimestepFailure.diagnostics.failure ==
                    metalrobo::ArticulatedCollisionFailure::
                        invalidConfiguration &&
                tinyTimestepFailure.contacts.empty(),
            "underflowing timestep published nonfinite regularization"
        );

        std::vector<MRBodyStateGPU> crossStates = states;
        crossStates[forward.bodyA].flagsAndIndices[1] = 1u;
        const auto crossFailure =
            metalrobo::adaptArticulatedContactConstraints(
                model,
                0u,
                std::span(&forward, 1u),
                crossStates,
                adapterConfig
            );
        require(
            !crossFailure.succeeded() &&
                crossFailure.diagnostics.failure ==
                    metalrobo::ArticulatedCollisionFailure::
                        crossArticulationContact &&
                crossFailure.contacts.empty(),
            "cross-articulation dynamic contact was not rejected"
        );

        std::vector<MRBodyStateGPU> unboundStates = states;
        MRBodyStateGPU& unboundGround =
            unboundStates[forward.bodyB];
        unboundGround.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
        unboundGround.flagsAndIndices[1] = MR_INVALID_INDEX;
        unboundGround.flagsAndIndices[2] = MR_INVALID_INDEX;
        unboundGround.linearVelocityAndInverseMass.w = 1.0f;
        const auto unboundFailure =
            metalrobo::adaptArticulatedContactConstraints(
                model,
                0u,
                std::span(&forward, 1u),
                unboundStates,
                adapterConfig
            );
        require(
            !unboundFailure.succeeded() &&
                unboundFailure.diagnostics.failure ==
                    metalrobo::ArticulatedCollisionFailure::
                        unboundDynamicBody &&
                unboundFailure.contacts.empty(),
            "unbound dynamic contact was not rejected"
        );

        MRContactConstraintGPU unknownFlag = forward;
        unknownFlag.flags |= 1u << 31u;
        const auto unknownFlagFailure =
            metalrobo::adaptArticulatedContactConstraints(
                model,
                0u,
                std::span(&unknownFlag, 1u),
                states,
                adapterConfig
            );
        require(
            !unknownFlagFailure.succeeded() &&
                unknownFlagFailure.diagnostics.failure ==
                    metalrobo::ArticulatedCollisionFailure::
                        invalidConstraint &&
                unknownFlagFailure.diagnostics
                    .failedConstraintIndex == 0u &&
                unknownFlagFailure.contacts.empty(),
            "unknown active constraint flag was accepted"
        );

        MRContactConstraintGPU disabledUnknownFlag = forward;
        disabledUnknownFlag.flags =
            MR_CONSTRAINT_FLAG_DISABLED | (1u << 31u);
        const auto disabledUnknownFlagFailure =
            metalrobo::adaptArticulatedContactConstraints(
                model,
                0u,
                std::span(&disabledUnknownFlag, 1u),
                states,
                adapterConfig
            );
        require(
            !disabledUnknownFlagFailure.succeeded() &&
                disabledUnknownFlagFailure.diagnostics.failure ==
                    metalrobo::ArticulatedCollisionFailure::
                        invalidConstraint &&
                disabledUnknownFlagFailure.diagnostics
                    .failedConstraintIndex == 0u &&
                disabledUnknownFlagFailure.contacts.empty(),
            "unknown disabled constraint flag was silently skipped"
        );

        std::cout
            << std::scientific
            << std::setprecision(6)
            << "g1_collision_contact=cpu_to_generalized_fp64"
            << " root_lowering=" << rootLowering
            << " raw_contacts=" << collision.rawContacts.size()
            << " manifolds=" << collision.manifoldHeaders.size()
            << " constraints=" << assembly.constraints.size()
            << " adapted=" << adaptation.contacts.size()
            << " penetration_max="
            << assembly.diagnostics.maximumPenetration
            << " target_normal_max="
            << adaptation.diagnostics.maximumNormalTargetVelocity
            << " solved_normal_min="
            << minimumSolvedNormalVelocity
            << " operator_velocity_error="
            << operatorVelocityError
            << " reconstructed_point_error="
            << reconstructedPointError
            << " free_jv_parity_error="
            << freeJvParityError
            << " quality_kkt="
            << solution.scaledKktCertificate
            << " endpoint_swap_error=" << endpointSwapError
            << " target_rule_error=" << targetRuleError
            << " compliance_regularization_error="
            << complianceRegularizationError
            << " kinematic_compensation_error="
            << kinematicCompensationError
            << " capacity_transactional=yes"
            << " tiny_timestep_rejected=yes"
            << " cross_articulation_rejected=yes"
            << " unbound_dynamic_rejected=yes"
            << " strict_constraint_flags=yes"
            << " status=ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "g1_collision_contact_probe failed: "
                  << error.what() << '\n';
        return 1;
    }
}
