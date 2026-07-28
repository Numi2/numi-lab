#include "metalrobo/ArticulatedRigidWorld.hpp"
#include "metalrobo/SurgicalAssets.hpp"
#include "metalrobo/SurgicalPSM.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Quaternion {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w = 1.0;
};

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Vec3 operator+(const Vec3 left, const Vec3 right) {
    return {
        left.x + right.x,
        left.y + right.y,
        left.z + right.z,
    };
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

double norm(const Vec3 value) {
    return std::sqrt(dot(value, value));
}

Vec3 vector(const mr_float4 value) {
    return {value.x, value.y, value.z};
}

Vec3 vector(const std::array<double, 3>& value) {
    return {value[0], value[1], value[2]};
}

Quaternion quaternion(const std::array<double, 4>& value) {
    return {value[0], value[1], value[2], value[3]};
}

Vec3 rotate(const Quaternion q, const Vec3 value) {
    const Vec3 imaginary{q.x, q.y, q.z};
    const Vec3 doubled{
        2.0 * (
            imaginary.y * value.z -
            imaginary.z * value.y
        ),
        2.0 * (
            imaginary.z * value.x -
            imaginary.x * value.z
        ),
        2.0 * (
            imaginary.x * value.y -
            imaginary.y * value.x
        ),
    };
    const Vec3 secondCross{
        imaginary.y * doubled.z -
            imaginary.z * doubled.y,
        imaginary.z * doubled.x -
            imaginary.x * doubled.z,
        imaginary.x * doubled.y -
            imaginary.y * doubled.x,
    };
    return {
        value.x + q.w * doubled.x + secondCross.x,
        value.y + q.w * doubled.y + secondCross.y,
        value.z + q.w * doubled.z + secondCross.z,
    };
}

mr_float4 f4(
    const Vec3 value,
    const double w = 0.0
) {
    return {
        static_cast<float>(value.x),
        static_cast<float>(value.y),
        static_cast<float>(value.z),
        static_cast<float>(w),
    };
}

Vec3 shapeCenter(
    const std::vector<metalrobo::ArticulatedBodyKinematics>&
        kinematics,
    const MRShapeGPU& shape
) {
    const auto body = std::ranges::find_if(
        kinematics,
        [&](const metalrobo::ArticulatedBodyKinematics& item) {
            return item.bodyIndex == shape.bodyIndex;
        }
    );
    require(body != kinematics.end(), "jaw body kinematics missing");
    return
        vector(body->centerOfMassPosition) +
        rotate(
            quaternion(body->orientation),
            vector(shape.localPosition)
        );
}

MRBodyStateGPU makeNeedleState(
    const metalrobo::CurvedSutureNeedleAsset& needle,
    const Vec3 position
) {
    MRBodyStateGPU result{};
    result.position = f4(position, 1.0);
    result.orientation = f4({0.0, 0.0, 0.0}, 1.0);
    result.linearVelocityAndInverseMass = f4(
        {0.0, 0.0, 0.0},
        needle.rigid.body.massAndInverseMass.y
    );
    result.angularVelocity = f4({0.0, 0.0, 0.0});
    result.inverseInertiaWorldRow0 =
        needle.rigid.body.inverseInertiaRow0;
    result.inverseInertiaWorldRow1 =
        needle.rigid.body.inverseInertiaRow1;
    result.inverseInertiaWorldRow2 =
        needle.rigid.body.inverseInertiaRow2;
    result.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
    result.flagsAndIndices[1] = MR_INVALID_INDEX;
    result.flagsAndIndices[2] = 41021u;
    return result;
}

std::vector<double> gravityCompensation(
    const metalrobo::EngineModel& model,
    const std::span<const double> q,
    const std::span<const double> v,
    const metalrobo::ArticulatedDynamicsConfig& config
) {
    std::vector<double> zeroAcceleration(v.size(), 0.0);
    std::vector<double> force(v.size(), 0.0);
    const auto diagnostics =
        metalrobo::computeArticulatedInverseDynamics(
            model,
            0u,
            q,
            v,
            zeroAcceleration,
            {},
            force,
            config
        );
    require(
        diagnostics.succeeded(),
        "gravity compensation inverse dynamics failed"
    );
    return force;
}

metalrobo::ArticulatedRigidWorldConfig worldConfig() {
    metalrobo::ArticulatedRigidWorldConfig config;
    config.dynamics.timestep = 5.0e-4;
    config.dynamics.gravity = {0.0, 0.0, -9.81};
    config.dynamics.integrator =
        metalrobo::ArticulatedIntegrator::symplecticEuler;
    config.rigidFreeMotion.integrator =
        metalrobo::FreeBodyIntegrator::symplecticEuler;
    config.collision.collision.environment = 91u;
    config.collision.collision.capacities = {
        .pairCapacity = 4096u,
        .rawContactCapacity = 4096u,
        .manifoldCapacity = 4096u,
    };
    config.collision.collision.manifoldBreakingSeparation =
        0.0015;
    config.collision.collision.manifoldBreakingTangential =
        0.0015;
    config.collision.collision.manifoldMergeDistance = 2.0e-5;
    config.collision.contact.contact.errorReduction = 0.08;
    config.collision.contact.contact.penetrationSlop = 1.0e-6;
    config.collision.contact.contact.maxDepenetrationVelocity =
        0.5;
    config.collision.contact.contactCapacity = 256u;
    config.collision.contact.qualityTangentialRegularization =
        1.0e-8;
    config.jointLimits.activationDistance = 0.0015;
    config.jointLimits.recoveryFraction = 0.1;
    config.jointLimits.maximumRecoverySpeed = 0.5;
    config.jointLimits.regularization = 1.0e-8;
    config.quality.maximumIterations = 1000u;
    config.quality.kktTolerance = 1.0e-5;
    config.maximumContactsPerBodyPair = 1u;
    config.grasp = {
        .enabled = true,
        .jawBodyA = 7u,
        .jawBodyB = 8u,
        .minimumNormalImpulse = 1.0e-9,
        .minimumFriction = 0.1,
        .maximumTangentialSlipSpeed = 0.05,
        .maximumOpposingNormalDot = -0.1,
        .requiredConsecutiveSteps = 3u,
    };
    return config;
}

bool sameManifolds(
    const std::span<const metalrobo::PersistentManifold> left,
    const std::span<const metalrobo::PersistentManifold> right
) {
    return left.size() == right.size() &&
        (
            left.empty() ||
            std::memcmp(
                left.data(),
                right.data(),
                left.size() *
                    sizeof(metalrobo::PersistentManifold)
            ) == 0
    );
}

bool sameContactCache(
    const std::span<
        const metalrobo::ArticulatedRigidContactCacheEntry> left,
    const std::span<
        const metalrobo::ArticulatedRigidContactCacheEntry> right
) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t index = 0u; index < left.size(); ++index) {
        if (!(left[index].warmStart.key ==
                right[index].warmStart.key) ||
            left[index].warmStart.worldImpulseOnB !=
                right[index].warmStart.worldImpulseOnB ||
            left[index].lastSeenStep != right[index].lastSeenStep) {
            return false;
        }
    }
    return true;
}

bool sameLimitCache(
    const std::span<
        const metalrobo::ArticulatedRigidJointLimitCacheEntry> left,
    const std::span<
        const metalrobo::ArticulatedRigidJointLimitCacheEntry> right
) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t index = 0u; index < left.size(); ++index) {
        if (left[index].stableKey != right[index].stableKey ||
            left[index].impulse != right[index].impulse ||
            left[index].lastSeenStep != right[index].lastSeenStep) {
            return false;
        }
    }
    return true;
}

bool sameGraspCache(
    const std::span<
        const metalrobo::ArticulatedRigidGraspCacheEntry> left,
    const std::span<
        const metalrobo::ArticulatedRigidGraspCacheEntry> right
) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t index = 0u; index < left.size(); ++index) {
        if (left[index].rigidBody != right[index].rigidBody ||
            left[index].identity != right[index].identity ||
            left[index].consecutiveQualifiedSteps !=
                right[index].consecutiveQualifiedSteps ||
            left[index].grasped != right[index].grasped ||
            left[index].lastSeenStep != right[index].lastSeenStep) {
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    try {
        const metalrobo::EngineModel model =
            metalrobo::makeDvrkPsmLargeNeedleDriverEngineModel();
        const metalrobo::CurvedSutureNeedleAsset needle =
            metalrobo::makeCurvedSutureNeedleAsset({
                .bodyIndex = 0u,
                .materialIndex = 0u,
                .slotGenerationBase = 410210u,
                .collisionGroup = 1u,
                .collisionMask = ~0u,
                .motionType = MR_MOTION_DYNAMIC,
            });
        MRMaterialGPU needleMaterial = needle.rigid.material;
        std::vector<double> q(
            model.defaultQ.begin(),
            model.defaultQ.end()
        );
        std::vector<double> v(model.world.nv, 0.0);
        q[6] = -0.018;
        q[7] = 0.018;

        std::vector<metalrobo::ArticulatedBodyKinematics>
            kinematics(model.articulations[0].bodyCount);
        const auto kinematicsDiagnostics =
            metalrobo::computeArticulatedBodyKinematics(
                model,
                0u,
                q,
                v,
                kinematics
            );
        require(
            kinematicsDiagnostics.succeeded(),
            "PSM kinematics failed"
        );
        const Vec3 jawA = shapeCenter(
            kinematics,
            model.shapes[15u]
        );
        const Vec3 jawB = shapeCenter(
            kinematics,
            model.shapes[17u]
        );
        const Vec3 jawMidpoint = (jawA + jawB) * 0.5;
        const std::uint32_t graspShapeIndex =
            (
                needle.metadata.graspShapeBegin +
                needle.metadata.graspShapeEnd
            ) / 2u;
        const MRShapeGPU& graspShape =
            needle.rigid.shapes[graspShapeIndex];
        std::array<MRBodyStateGPU, 1> rigidBodies{{
            makeNeedleState(
                needle,
                jawMidpoint - vector(graspShape.localPosition)
            ),
        }};
        const Vec3 initialNeedlePosition =
            vector(rigidBodies[0].position);

        metalrobo::ArticulatedRigidWorldConfig config =
            worldConfig();
        double mixedLimitImpulse = 0.0;
        {
            std::vector<double> mixedQ = q;
            std::vector<double> mixedV(v.size(), 0.0);
            mixedQ[0] =
                static_cast<double>(model.dofs[0].limits.y) -
                2.0e-4;
            mixedV[0] = 0.5;
            std::vector<
                metalrobo::ArticulatedBodyKinematics
            > mixedKinematics(model.articulations[0].bodyCount);
            const auto mixedKinematicsDiagnostics =
                metalrobo::computeArticulatedBodyKinematics(
                    model,
                    0u,
                    mixedQ,
                    mixedV,
                    mixedKinematics,
                    config.dynamics
                );
            require(
                mixedKinematicsDiagnostics.succeeded(),
                "mixed contact/limit kinematics failed"
            );
            const Vec3 mixedMidpoint =
                (
                    shapeCenter(
                        mixedKinematics,
                        model.shapes[15u]
                    ) +
                    shapeCenter(
                        mixedKinematics,
                        model.shapes[17u]
                    )
                ) * 0.5;
            std::array<MRBodyStateGPU, 1> mixedRigid{{
                makeNeedleState(
                    needle,
                    mixedMidpoint -
                        vector(graspShape.localPosition)
                ),
            }};
            std::vector<double> mixedForce =
                gravityCompensation(
                    model,
                    mixedQ,
                    mixedV,
                    config.dynamics
                );
            metalrobo::ArticulatedRigidWorldCache mixedCache;
            auto mixedConfig = config;
            mixedConfig.grasp.enabled = false;
            const auto mixed =
                metalrobo::stepArticulatedRigidWorldCpu(
                    model,
                    0u,
                    mixedQ,
                    mixedV,
                    mixedForce,
                    {},
                    std::span<const MRBodyPropertiesGPU>(
                        &needle.rigid.body,
                        1u
                    ),
                    mixedRigid,
                    needle.rigid.shapes,
                    std::span<const MRMaterialGPU>(
                        &needleMaterial,
                        1u
                    ),
                    {},
                    mixedConfig,
                    mixedCache
                );
            require(
                mixed.succeeded() &&
                    mixed.contactCount >= 2u &&
                    mixed.jointLimitCount >= 1u &&
                    mixed.maximumJointLimitImpulse > 0.0 &&
                    mixed.coupledSolve.contactImpulses.size() >=
                        6u &&
                    !mixed.coupledSolve.jointLimitImpulses.empty(),
                "actual PSM contact and active stop were not solved "
                "monolithically: " + mixed.coupledSolve.failure
            );
            mixedLimitImpulse =
                mixed.maximumJointLimitImpulse;
            mixedForce = gravityCompensation(
                model,
                mixedQ,
                mixedV,
                config.dynamics
            );
            const auto warmedMixed =
                metalrobo::stepArticulatedRigidWorldCpu(
                    model,
                    0u,
                    mixedQ,
                    mixedV,
                    mixedForce,
                    {},
                    std::span<const MRBodyPropertiesGPU>(
                        &needle.rigid.body,
                        1u
                    ),
                    mixedRigid,
                    needle.rigid.shapes,
                    std::span<const MRMaterialGPU>(
                        &needleMaterial,
                        1u
                    ),
                    {},
                    mixedConfig,
                    mixedCache
                );
            require(
                warmedMixed.succeeded() &&
                    warmedMixed.jointLimitCount >= 1u &&
                    warmedMixed.matchedJointLimitWarmStarts >= 1u,
                "joint-limit warm impulse did not persist into the "
                "next monolithic solve"
            );
            mixedLimitImpulse = std::max(
                mixedLimitImpulse,
                warmedMixed.maximumJointLimitImpulse
            );
        }
        metalrobo::ArticulatedRigidWorldCache cache;
        std::uint32_t maximumContacts = 0u;
        std::uint32_t maximumLimits = 0u;
        std::uint32_t maximumWarmMatches = 0u;
        std::uint32_t graspFrames = 0u;
        double maximumNormalImpulse = 0.0;
        double maximumKkt = 0.0;
        double maximumGraspSlip = 0.0;

        constexpr std::uint32_t holdSteps = 100u;
        constexpr std::uint32_t liftSteps = 200u;
        const double insertionTarget = q[2] + 0.0030;
        for (std::uint32_t step = 0u;
             step < holdSteps + liftSteps;
             ++step) {
            std::vector<double> force =
                gravityCompensation(model, q, v, config.dynamics);
            const double targetInsertion =
                step < holdSteps ? q[2] : insertionTarget;
            force[2] +=
                800.0 * (targetInsertion - q[2]) -
                40.0 * v[2];
            force[6] +=
                4.0 * (0.0 - q[6]) -
                0.04 * v[6];
            force[7] +=
                4.0 * (0.0 - q[7]) -
                0.04 * v[7];
            force[6] = std::clamp(force[6], -0.1, 0.1);
            force[7] = std::clamp(force[7], -0.1, 0.1);

            const auto diagnostics =
                metalrobo::stepArticulatedRigidWorldCpu(
                    model,
                    0u,
                    q,
                    v,
                    force,
                    {},
                    std::span<const MRBodyPropertiesGPU>(
                        &needle.rigid.body,
                        1u
                    ),
                    rigidBodies,
                    needle.rigid.shapes,
                    std::span<const MRMaterialGPU>(
                        &needleMaterial,
                        1u
                    ),
                    {},
                    config,
                    cache
                );
            require(
                diagnostics.succeeded(),
                "composed PSM/needle step failed at " +
                    std::to_string(step) +
                    " stage=" +
                    std::to_string(
                        static_cast<std::uint32_t>(
                            diagnostics.failure
                        )
                    ) +
                    " coupled=" +
                    diagnostics.coupledSolve.failure +
                    " contacts=" +
                    std::to_string(diagnostics.contactCount) +
                    " kkt=" +
                    std::to_string(
                        diagnostics.coupledSolve.quality.
                            scaledKktCertificate
                    ) +
                    " natural=" +
                    std::to_string(
                        diagnostics.coupledSolve.quality.
                            scaledNaturalResidual
                    ) +
                    " lipschitz=" +
                    std::to_string(
                        diagnostics.coupledSolve.quality.
                            lipschitzBound
                    ) +
                    " newton=" +
                    std::to_string(
                        diagnostics.coupledSolve.quality.
                            semismoothNewtonSteps
                    ) +
                    " gn=" +
                    std::to_string(
                        diagnostics.coupledSolve.quality.
                            gaussNewtonFallbackSteps
                    ) +
                    " pg=" +
                    std::to_string(
                        diagnostics.coupledSolve.quality.
                            projectedGradientFallbackSteps
                    )
            );
            maximumContacts =
                std::max(maximumContacts, diagnostics.contactCount);
            maximumLimits =
                std::max(maximumLimits, diagnostics.jointLimitCount);
            maximumWarmMatches = std::max(
                maximumWarmMatches,
                diagnostics.matchedContactWarmStarts
            );
            maximumNormalImpulse = std::max(
                maximumNormalImpulse,
                diagnostics.maximumNormalImpulse
            );
            maximumKkt = std::max(
                maximumKkt,
                diagnostics.coupledSolve.quality.
                    scaledKktCertificate
            );
            if (!diagnostics.graspEvidence.empty() &&
                diagnostics.graspEvidence[0].grasped) {
                ++graspFrames;
            }
            if (!diagnostics.graspEvidence.empty()) {
                maximumGraspSlip = std::max(
                    maximumGraspSlip,
                    diagnostics.graspEvidence[0].
                        maximumTangentialSlipSpeed
                );
            }
        }

        const Vec3 finalNeedlePosition =
            vector(rigidBodies[0].position);
        const Vec3 needleDisplacement =
            finalNeedlePosition - initialNeedlePosition;
        std::vector<metalrobo::ArticulatedBodyKinematics>
            finalKinematics(model.articulations[0].bodyCount);
        const auto finalKinematicsDiagnostics =
            metalrobo::computeArticulatedBodyKinematics(
                model,
                0u,
                q,
                v,
                finalKinematics,
                config.dynamics
            );
        require(
            finalKinematicsDiagnostics.succeeded(),
            "final PSM kinematics failed"
        );
        const Vec3 finalJawMidpoint =
            (
                shapeCenter(finalKinematics, model.shapes[15u]) +
                shapeCenter(finalKinematics, model.shapes[17u])
            ) * 0.5;
        const Vec3 jawDisplacement =
            finalJawMidpoint - jawMidpoint;
        const double jawTravel = norm(jawDisplacement);
        const double needleAlongJawTravel =
            jawTravel > 0.0
            ? dot(needleDisplacement, jawDisplacement) / jawTravel
            : 0.0;
        require(
            maximumContacts >= 2u,
            "two-jaw placement did not generate a contact patch"
        );
        require(
            maximumWarmMatches > 0u,
            "persistent contact impulses never rematched"
        );
        require(
            maximumNormalImpulse > 0.0,
            "composed solve generated no compressive impulse"
        );
        require(
            maximumKkt < 1.1e-5,
            "composed exact-cone KKT certificate is too large"
        );
        require(
            graspFrames > 0u,
            "two-sided physics evidence never qualified a grasp"
        );
        require(
            jawTravel > 5.0e-4 &&
                needleAlongJawTravel > 2.0e-4 &&
                needleDisplacement.z > 2.0e-4,
            "needle did not follow the instrument against gravity"
        );

        require(
            cache.graspEvidence.size() == 1u &&
                cache.graspEvidence[0].grasped,
            "final grasp cache is missing qualified needle evidence"
        );
        const std::uint64_t priorGraspIdentity =
            cache.graspEvidence[0].identity;
        std::vector<double> replacementQ = q;
        std::vector<double> replacementV = v;
        auto replacementRigid = rigidBodies;
        auto replacementShapes = needle.rigid.shapes;
        ++replacementShapes[0].slotGeneration;
        auto replacementCache = cache;
        std::vector<double> replacementForce = gravityCompensation(
            model,
            replacementQ,
            replacementV,
            config.dynamics
        );
        replacementForce[6] = std::clamp(
            replacementForce[6] +
                4.0 * (0.0 - replacementQ[6]) -
                0.04 * replacementV[6],
            -0.1,
            0.1
        );
        replacementForce[7] = std::clamp(
            replacementForce[7] +
                4.0 * (0.0 - replacementQ[7]) -
                0.04 * replacementV[7],
            -0.1,
            0.1
        );
        const auto replacementStep =
            metalrobo::stepArticulatedRigidWorldCpu(
                model,
                0u,
                replacementQ,
                replacementV,
                replacementForce,
                {},
                std::span<const MRBodyPropertiesGPU>(
                    &needle.rigid.body,
                    1u
                ),
                replacementRigid,
                replacementShapes,
                std::span<const MRMaterialGPU>(
                    &needleMaterial,
                    1u
                ),
                {},
                config,
                replacementCache
            );
        require(
            replacementStep.succeeded() &&
                replacementCache.graspEvidence.size() == 1u &&
                replacementCache.graspEvidence[0].identity !=
                    priorGraspIdentity &&
                replacementCache.graspEvidence[0].
                    consecutiveQualifiedSteps <= 1u &&
                !replacementCache.graspEvidence[0].grasped,
            "grasp dwell transferred across a rigid-shape generation"
        );

        const std::vector<double> qBefore = q;
        const std::vector<double> vBefore = v;
        const auto rigidBefore = rigidBodies;
        const std::uint64_t stepBefore = cache.step;
        const auto contactCacheBefore = cache.contactImpulses;
        const auto limitCacheBefore = cache.jointLimitImpulses;
        const auto graspCacheBefore = cache.graspEvidence;
        const std::vector<metalrobo::PersistentManifold>
            manifoldsBefore(
                cache.manifolds.entries().begin(),
                cache.manifolds.entries().end()
            );
        std::vector<double> invalidForce(v.size(), 0.0);
        invalidForce[0] =
            std::numeric_limits<double>::quiet_NaN();
        const auto rejected =
            metalrobo::stepArticulatedRigidWorldCpu(
                model,
                0u,
                q,
                v,
                invalidForce,
                {},
                std::span<const MRBodyPropertiesGPU>(
                    &needle.rigid.body,
                    1u
                ),
                rigidBodies,
                needle.rigid.shapes,
                std::span<const MRMaterialGPU>(
                    &needleMaterial,
                    1u
                ),
                {},
                config,
                cache
            );
        require(
            !rejected.succeeded() &&
                q == qBefore &&
                v == vBefore &&
                std::memcmp(
                    rigidBodies.data(),
                    rigidBefore.data(),
                    sizeof(rigidBodies)
                ) == 0 &&
                cache.step == stepBefore &&
                sameContactCache(
                    cache.contactImpulses,
                    contactCacheBefore
                ) &&
                sameLimitCache(
                    cache.jointLimitImpulses,
                    limitCacheBefore
                ) &&
                sameGraspCache(
                    cache.graspEvidence,
                    graspCacheBefore
                ) &&
                sameManifolds(
                    cache.manifolds.entries(),
                    manifoldsBefore
                ),
            "late composed-step rejection violated transactionality"
        );

        std::cout
            << std::setprecision(12)
            << "articulated_rigid_world"
            << " model=" << model.name
            << " steps=" << cache.step
            << " contacts_max=" << maximumContacts
            << " joint_limits_max=" << maximumLimits
            << " mixed_limit_impulse=" << mixedLimitImpulse
            << " warm_matches_max=" << maximumWarmMatches
            << " normal_impulse_max=" << maximumNormalImpulse
            << " grasp_frames=" << graspFrames
            << " needle_displacement_mm="
            << norm(needleDisplacement) * 1000.0
            << " needle_lift_mm="
            << needleAlongJawTravel * 1000.0
            << " needle_dz_mm="
            << needleDisplacement.z * 1000.0
            << " jaw_travel_mm=" << jawTravel * 1000.0
            << " kkt_max=" << maximumKkt
            << " grasp_slip_max=" << maximumGraspSlip
            << " grasp_identity_reset=pass"
            << " rollback=pass"
            << " status=ok\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "articulated_rigid_world status=failed reason="
            << exception.what() << '\n';
        return 1;
    }
}
