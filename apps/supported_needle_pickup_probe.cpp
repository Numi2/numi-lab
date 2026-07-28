#include "metalrobo/ArticulatedRigidWorld.hpp"
#include "metalrobo/Collision.hpp"
#include "metalrobo/SurgicalAssets.hpp"
#include "metalrobo/SurgicalPSM.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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

enum class Phase {
    settle,
    approach,
    close,
    dwell,
    lift,
};

constexpr std::array<std::uint32_t, 3> kSupportNeedleShapes{
    6u,
    9u,
    25u,
};
constexpr std::size_t kSupportButtonCount =
    2u * kSupportNeedleShapes.size();
constexpr double kSupportRadius = 0.00125;
constexpr double kInitialSupportPenetration = 1.0e-6;
constexpr double kSupportRadialOffset = 0.00035;
constexpr double kOpenJawCoordinate = 0.025;
constexpr double kApproachDistance = 0.004;
constexpr double kLiftDistance = 0.0085;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool finite(const double value) {
    return std::isfinite(value);
}

Vec3 operator+(const Vec3 left, const Vec3 right) {
    return {left.x + right.x, left.y + right.y, left.z + right.z};
}

Vec3 operator-(const Vec3 left, const Vec3 right) {
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

Vec3 operator*(const Vec3 value, const double scale) {
    return {value.x * scale, value.y * scale, value.z * scale};
}

double dot(const Vec3 left, const Vec3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
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

Quaternion quaternion(const mr_float4 value) {
    return {value.x, value.y, value.z, value.w};
}

Quaternion quaternion(const std::array<double, 4>& value) {
    return {value[0], value[1], value[2], value[3]};
}

Vec3 rotate(const Quaternion q, const Vec3 value) {
    const Vec3 imaginary{q.x, q.y, q.z};
    const Vec3 doubled{
        2.0 * (imaginary.y * value.z - imaginary.z * value.y),
        2.0 * (imaginary.z * value.x - imaginary.x * value.z),
        2.0 * (imaginary.x * value.y - imaginary.y * value.x),
    };
    const Vec3 secondCross{
        imaginary.y * doubled.z - imaginary.z * doubled.y,
        imaginary.z * doubled.x - imaginary.x * doubled.z,
        imaginary.x * doubled.y - imaginary.y * doubled.x,
    };
    return {
        value.x + q.w * doubled.x + secondCross.x,
        value.y + q.w * doubled.y + secondCross.y,
        value.z + q.w * doubled.z + secondCross.z,
    };
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

mr_float4 f4(const Vec3 value, const double w = 0.0) {
    return f4(value.x, value.y, value.z, w);
}

double lerp(
    const double start,
    const double end,
    const double fraction
) {
    return start + (end - start) * fraction;
}

Vec3 articulatedShapeCenter(
    const std::span<
        const metalrobo::ArticulatedBodyKinematics> kinematics,
    const MRShapeGPU& shape
) {
    const auto body = std::find_if(
        kinematics.begin(),
        kinematics.end(),
        [&](const metalrobo::ArticulatedBodyKinematics& item) {
            return item.bodyIndex == shape.bodyIndex;
        }
    );
    require(body != kinematics.end(), "articulated shape body is missing");
    return
        vector(body->centerOfMassPosition) +
        rotate(
            quaternion(body->orientation),
            vector(shape.localPosition)
        );
}

Vec3 jawMidpoint(
    const metalrobo::EngineModel& model,
    const std::span<const double> q,
    const std::span<const double> v,
    const metalrobo::ArticulatedDynamicsConfig& config
) {
    std::vector<metalrobo::ArticulatedBodyKinematics> kinematics(
        model.articulations[0].bodyCount
    );
    const auto diagnostics =
        metalrobo::computeArticulatedBodyKinematics(
            model,
            0u,
            q,
            v,
            kinematics,
            config
        );
    require(diagnostics.succeeded(), "PSM jaw kinematics failed");
    return (
        articulatedShapeCenter(kinematics, model.shapes[15u]) +
        articulatedShapeCenter(kinematics, model.shapes[17u])
    ) * 0.5;
}

MRBodyStateGPU dynamicNeedleState(
    const metalrobo::CurvedSutureNeedleAsset& needle,
    const Vec3 position
) {
    MRBodyStateGPU result{};
    result.position = f4(position, 1.0);
    result.orientation = f4(0.0, 0.0, 0.0, 1.0);
    result.linearVelocityAndInverseMass = f4(
        0.0,
        0.0,
        0.0,
        needle.rigid.body.massAndInverseMass.y
    );
    result.angularVelocity = f4(0.0, 0.0, 0.0);
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

MRBodyPropertiesGPU staticBodyProperties() {
    MRBodyPropertiesGPU result{};
    result.articulationIndex = MR_INVALID_INDEX;
    result.parentBody = MR_INVALID_INDEX;
    result.inboundJoint = MR_INVALID_INDEX;
    result.motionType = MR_MOTION_STATIC;
    result.dampingAndSpeedLimits = f4(0.0, 0.0, 0.0, 0.0);
    return result;
}

MRBodyStateGPU staticBodyState(const std::uint32_t generation) {
    MRBodyStateGPU result{};
    result.position = f4(0.0, 0.0, 0.0, 1.0);
    result.orientation = f4(0.0, 0.0, 0.0, 1.0);
    result.flagsAndIndices[0] = MR_MOTION_STATIC;
    result.flagsAndIndices[1] = MR_INVALID_INDEX;
    result.flagsAndIndices[2] = generation;
    return result;
}

MRMaterialGPU supportMaterial() {
    MRMaterialGPU result{};
    result.friction = f4(0.0, 0.0, 0.0, 0.0);
    result.response = f4(0.0, 0.05, 1.0e-7, 0.0);
    return result;
}

MRShapeGPU supportSphere(
    const std::uint32_t body,
    const std::uint32_t generation,
    const Vec3 center
) {
    MRShapeGPU result{};
    result.bodyIndex = body;
    result.shapeType = MR_SHAPE_SPHERE;
    result.materialIndex = 1u;
    result.collisionGroup = 2u;
    result.collisionMask = ~2u;
    result.slotGeneration = generation;
    result.localPosition = f4(center, 1.0);
    result.localRotation = f4(0.0, 0.0, 0.0, 1.0);
    result.dimensions = f4(kSupportRadius, 0.0, 0.0, 0.0);
    result.contactRestAndBoundingRadius =
        f4(2.0e-5, 0.0, kSupportRadius, 0.0);
    return result;
}

metalrobo::ArticulatedRigidWorldConfig worldConfig() {
    metalrobo::ArticulatedRigidWorldConfig config;
    config.dynamics.timestep = 5.0e-4;
    config.dynamics.gravity = {0.0, 0.0, -9.81};
    config.dynamics.integrator =
        metalrobo::ArticulatedIntegrator::symplecticEuler;
    config.rigidFreeMotion.integrator =
        metalrobo::FreeBodyIntegrator::symplecticEuler;
    config.collision.collision.environment = 92u;
    config.collision.collision.capacities = {
        .pairCapacity = 8192u,
        .rawContactCapacity = 8192u,
        .manifoldCapacity = 8192u,
    };
    config.collision.collision.manifoldBreakingSeparation = 0.0015;
    config.collision.collision.manifoldBreakingTangential = 0.0015;
    config.collision.collision.manifoldMergeDistance = 2.0e-5;
    config.collision.contact.contact.errorReduction = 0.0;
    config.collision.contact.contact.penetrationSlop = 1.0e-6;
    config.collision.contact.contact.maxDepenetrationVelocity = 0.5;
    config.collision.contact.contactCapacity = 512u;
    config.collision.contact.qualityTangentialRegularization = 1.0e-4;
    config.jointLimits.activationDistance = 0.0015;
    config.jointLimits.recoveryFraction = 0.1;
    config.jointLimits.maximumRecoverySpeed = 0.5;
    config.jointLimits.regularization = 1.0e-8;
    config.quality.maximumIterations = 4000u;
    config.quality.kktTolerance = 2.0e-5;
    // Each cradle button owns a distinct static body, so one retained witness
    // per body pair preserves every independent support and each jaw pinch.
    config.maximumContactsPerBodyPair = 1u;
    config.grasp = {
        .enabled = true,
        .jawBodyA = 7u,
        .jawBodyB = 8u,
        .minimumNormalImpulse = 1.0e-9,
        .minimumFriction = 0.1,
        .maximumTangentialSlipSpeed = 0.05,
        .maximumOpposingNormalDot = -0.1,
        .requiredConsecutiveSteps = 10u,
    };
    return config;
}

std::vector<double> inverseDynamicsForce(
    const metalrobo::EngineModel& model,
    const std::span<const double> q,
    const std::span<const double> v,
    const std::span<const double> acceleration,
    const metalrobo::ArticulatedDynamicsConfig& config
) {
    std::vector<double> force(v.size(), 0.0);
    const auto diagnostics =
        metalrobo::computeArticulatedInverseDynamics(
            model,
            0u,
            q,
            v,
            acceleration,
            {},
            force,
            config
        );
    require(
        diagnostics.succeeded(),
        "PSM computed-torque inverse dynamics failed"
    );
    return force;
}

std::vector<double> controllerForce(
    const metalrobo::EngineModel& model,
    const std::span<const double> q,
    const std::span<const double> v,
    const std::span<const double> targets,
    const metalrobo::ArticulatedDynamicsConfig& config
) {
    std::vector<double> acceleration(v.size(), 0.0);
    for (std::size_t dof = 0u;
         dof < std::min<std::size_t>(6u, acceleration.size());
         ++dof) {
        double stiffness = 400.0;
        double damping = 40.0;
        if (dof == 2u) {
            stiffness = 1600.0;
            damping = 80.0;
        }
        acceleration[dof] =
            stiffness * (targets[dof] - q[dof]) -
            damping * v[dof];
    }
    require(
        acceleration.size() >= 8u && targets.size() >= 8u,
        "PSM jaw controller dimensions are invalid"
    );
    const double aperture = q[7u] - q[6u];
    const double targetAperture = targets[7u] - targets[6u];
    const double apertureSpeed = v[7u] - v[6u];
    const double common = q[6u] + q[7u];
    const double targetCommon = targets[6u] + targets[7u];
    const double commonSpeed = v[6u] + v[7u];
    const double apertureAcceleration =
        4000.0 * (targetAperture - aperture) -
        120.0 * apertureSpeed;
    const double commonAcceleration =
        400.0 * (targetCommon - common) -
        40.0 * commonSpeed;
    acceleration[6u] =
        0.5 * (commonAcceleration - apertureAcceleration);
    acceleration[7u] =
        0.5 * (commonAcceleration + apertureAcceleration);

    std::vector<double> force = inverseDynamicsForce(
        model,
        q,
        v,
        acceleration,
        config
    );
    for (std::size_t dof = 0u; dof < force.size(); ++dof) {
        const double effortLimit = std::min(
            static_cast<double>(model.dofs[dof].limits.w),
            dof >= 6u ? 0.1 : std::numeric_limits<double>::infinity()
        );
        require(
            finite(effortLimit) && effortLimit > 0.0,
            "PSM jaw effort limit is invalid"
        );
        force[dof] = std::clamp(
            force[dof],
            -effortLimit,
            effortLimit
        );
    }
    return force;
}

double supportTriangleMargin(
    const metalrobo::CurvedSutureNeedleAsset& needle
) {
    const Vec3 a = vector(
        needle.rigid.shapes[kSupportNeedleShapes[0]].localPosition
    );
    const Vec3 b = vector(
        needle.rigid.shapes[kSupportNeedleShapes[1]].localPosition
    );
    const Vec3 c = vector(
        needle.rigid.shapes[kSupportNeedleShapes[2]].localPosition
    );
    const double denominator =
        (b.y - c.y) * (a.x - c.x) +
        (c.x - b.x) * (a.y - c.y);
    require(
        std::abs(denominator) > 1.0e-12,
        "support triangle is degenerate"
    );
    const double first =
        ((b.y - c.y) * -c.x + (c.x - b.x) * -c.y) /
        denominator;
    const double second =
        ((c.y - a.y) * -c.x + (a.x - c.x) * -c.y) /
        denominator;
    const double third = 1.0 - first - second;
    return std::min({first, second, third});
}

std::uint32_t observedSupportContacts(
    const std::span<const MRShapeGPU> rigidShapes,
    const std::span<const MRBodyStateGPU> rigidBodies,
    const std::size_t needleShapeCount
) {
    metalrobo::CollisionConfig config;
    config.environment = 9301u;
    config.capacities = {
        .pairCapacity = 512u,
        .rawContactCapacity = 512u,
        .manifoldCapacity = 512u,
    };
    config.manifoldBreakingSeparation = 5.0e-4;
    config.manifoldBreakingTangential = 5.0e-4;
    config.manifoldMergeDistance = 2.0e-5;
    metalrobo::PersistentManifoldCache cache;
    const metalrobo::CollisionFrame frame =
        metalrobo::collideCpuReference(
            rigidShapes,
            rigidBodies,
            config,
            cache
        );
    require(frame.succeeded(), "support observation collision failed");

    std::array<bool, kSupportButtonCount> touched{};
    for (const MRManifoldHeaderGPU& header : frame.manifoldHeaders) {
        if (header.pairAndCount[3] == 0u) {
            continue;
        }
        const std::uint32_t colliderA = header.pairAndCount[1];
        const std::uint32_t colliderB = header.pairAndCount[2];
        const std::uint32_t supportCollider =
            colliderA >= needleShapeCount ? colliderA : colliderB;
        if (supportCollider >= needleShapeCount &&
            supportCollider <
                needleShapeCount + touched.size()) {
            touched[supportCollider - needleShapeCount] = true;
        }
    }
    return static_cast<std::uint32_t>(std::count(
        touched.begin(),
        touched.end(),
        true
    ));
}

double pointSegmentDistance(
    const Vec3 point,
    const Vec3 start,
    const Vec3 end
) {
    const Vec3 segment = end - start;
    const double squaredLength = dot(segment, segment);
    if (!(squaredLength > 0.0)) {
        return norm(point - start);
    }
    const double parameter = std::clamp(
        dot(point - start, segment) / squaredLength,
        0.0,
        1.0
    );
    return norm(point - (start + segment * parameter));
}

double minimumSupportClearance(
    const std::span<const MRShapeGPU> rigidShapes,
    const std::span<const MRBodyStateGPU> rigidBodies,
    const std::size_t needleShapeCount
) {
    const MRBodyStateGPU& needleBody = rigidBodies[0u];
    const Quaternion needleOrientation =
        quaternion(needleBody.orientation);
    const Vec3 needlePosition = vector(needleBody.position);
    double minimum = std::numeric_limits<double>::infinity();
    for (std::size_t needleIndex = 0u;
         needleIndex < needleShapeCount;
         ++needleIndex) {
        const MRShapeGPU& needleShape = rigidShapes[needleIndex];
        require(
            needleShape.shapeType == MR_SHAPE_CAPSULE,
            "needle support clearance expects capsule decomposition"
        );
        const Vec3 center =
            needlePosition +
            rotate(
                needleOrientation,
                vector(needleShape.localPosition)
            );
        const Vec3 localAxis = rotate(
            quaternion(needleShape.localRotation),
            {0.0, needleShape.dimensions.y, 0.0}
        );
        const Vec3 worldAxis =
            rotate(needleOrientation, localAxis);
        for (std::size_t support = 0u;
         support < kSupportButtonCount;
             ++support) {
            const MRShapeGPU& supportShape =
                rigidShapes[needleShapeCount + support];
            const MRBodyStateGPU& supportBody =
                rigidBodies[supportShape.bodyIndex];
            const Vec3 supportCenter =
                vector(supportBody.position) +
                rotate(
                    quaternion(supportBody.orientation),
                    vector(supportShape.localPosition)
                );
            minimum = std::min(
                minimum,
                pointSegmentDistance(
                    supportCenter,
                    center - worldAxis,
                    center + worldAxis
                ) -
                    static_cast<double>(needleShape.dimensions.x) -
                    static_cast<double>(supportShape.dimensions.x)
            );
        }
    }
    return minimum;
}

const metalrobo::ArticulatedRigidGraspEvidence*
needleGraspEvidence(
    const metalrobo::ArticulatedRigidWorldStepDiagnostics& diagnostics
) {
    const auto found = std::find_if(
        diagnostics.graspEvidence.begin(),
        diagnostics.graspEvidence.end(),
        [](const metalrobo::ArticulatedRigidGraspEvidence& evidence) {
            return evidence.rigidBody == 0u;
        }
    );
    return found == diagnostics.graspEvidence.end()
        ? nullptr
        : &*found;
}

bool sameCache(
    const metalrobo::ArticulatedRigidWorldCache& left,
    const metalrobo::ArticulatedRigidWorldCache& right
) {
    const auto leftManifolds = left.manifolds.entries();
    const auto rightManifolds = right.manifolds.entries();
    if (left.step != right.step ||
        leftManifolds.size() != rightManifolds.size() ||
        (
            !leftManifolds.empty() &&
            std::memcmp(
                leftManifolds.data(),
                rightManifolds.data(),
                leftManifolds.size() *
                    sizeof(metalrobo::PersistentManifold)
            ) != 0
        ) ||
        left.contactImpulses.size() !=
            right.contactImpulses.size() ||
        left.jointLimitImpulses.size() !=
            right.jointLimitImpulses.size() ||
        left.graspEvidence.size() !=
            right.graspEvidence.size()) {
        return false;
    }
    for (std::size_t index = 0u;
         index < left.contactImpulses.size();
         ++index) {
        const auto& a = left.contactImpulses[index];
        const auto& b = right.contactImpulses[index];
        if (!(a.warmStart.key == b.warmStart.key) ||
            a.warmStart.worldImpulseOnB !=
                b.warmStart.worldImpulseOnB ||
            a.lastSeenStep != b.lastSeenStep) {
            return false;
        }
    }
    for (std::size_t index = 0u;
         index < left.jointLimitImpulses.size();
         ++index) {
        const auto& a = left.jointLimitImpulses[index];
        const auto& b = right.jointLimitImpulses[index];
        if (a.stableKey != b.stableKey ||
            a.impulse != b.impulse ||
            a.lastSeenStep != b.lastSeenStep) {
            return false;
        }
    }
    for (std::size_t index = 0u;
         index < left.graspEvidence.size();
         ++index) {
        const auto& a = left.graspEvidence[index];
        const auto& b = right.graspEvidence[index];
        if (a.rigidBody != b.rigidBody ||
            a.identity != b.identity ||
            a.consecutiveQualifiedSteps !=
                b.consecutiveQualifiedSteps ||
            a.grasped != b.grasped ||
            a.lastSeenStep != b.lastSeenStep) {
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
                .collisionMask = ~1u,
                .motionType = MR_MOTION_DYNAMIC,
            });
        require(
            needle.metadata.graspShapeBegin <= 14u &&
                14u < needle.metadata.graspShapeEnd,
            "canonical grasp shape left the authored grasp zone"
        );
        for (const std::uint32_t shape : kSupportNeedleShapes) {
            require(
                shape < needle.metadata.graspShapeBegin ||
                    shape >= needle.metadata.graspShapeEnd,
                "support button entered the needle grasp zone"
            );
        }
        const double triangleMargin = supportTriangleMargin(needle);
        require(
            triangleMargin > 0.15,
            "three-site cradle does not contain the needle COM"
        );

        metalrobo::ArticulatedRigidWorldConfig config =
            worldConfig();
        std::vector<double> placementQ(
            model.defaultQ.begin(),
            model.defaultQ.end()
        );
        std::vector<double> zeroV(model.world.nv, 0.0);
        placementQ[6u] = -0.018;
        placementQ[7u] = 0.018;
        const Vec3 pickupMidpoint =
            jawMidpoint(model, placementQ, zeroV, config.dynamics);
        const std::uint32_t graspShapeIndex = 14u;
        const Vec3 needlePosition =
            pickupMidpoint -
            vector(
                needle.rigid.shapes[graspShapeIndex].localPosition
            );

        std::vector<MRBodyPropertiesGPU> rigidProperties;
        rigidProperties.reserve(1u + kSupportButtonCount);
        rigidProperties.push_back(needle.rigid.body);
        for (std::size_t support = 0u;
             support < kSupportButtonCount;
             ++support) {
            rigidProperties.push_back(staticBodyProperties());
        }

        std::vector<MRBodyStateGPU> rigidBodies;
        rigidBodies.reserve(1u + kSupportButtonCount);
        rigidBodies.push_back(
            dynamicNeedleState(needle, needlePosition)
        );
        for (std::size_t support = 0u;
             support < kSupportButtonCount;
             ++support) {
            rigidBodies.push_back(staticBodyState(
                52021u + static_cast<std::uint32_t>(support)
            ));
        }

        std::vector<MRMaterialGPU> rigidMaterials{
            needle.rigid.material,
            supportMaterial(),
        };
        std::vector<MRShapeGPU> rigidShapes = needle.rigid.shapes;
        rigidShapes.reserve(
            rigidShapes.size() + kSupportButtonCount
        );
        for (std::size_t support = 0u;
             support < kSupportButtonCount;
             ++support) {
            const std::size_t supportSite = support / 2u;
            const double side =
                support % 2u == 0u ? -1.0 : 1.0;
            const MRShapeGPU& supportedNeedleShape =
                needle.rigid.shapes[
                    kSupportNeedleShapes[supportSite]
                ];
            const Vec3 supportedCenter =
                needlePosition +
                vector(supportedNeedleShape.localPosition);
            const double contactDistance =
                static_cast<double>(
                    supportedNeedleShape.dimensions.x
                ) +
                kSupportRadius +
                static_cast<double>(
                    supportedNeedleShape.
                        contactRestAndBoundingRadius.x
                ) +
                2.0e-5 -
                kInitialSupportPenetration;
            const Vec3 localCenter =
                vector(supportedNeedleShape.localPosition);
            const Vec3 radial{localCenter.x, localCenter.y, 0.0};
            const double radialLength = norm(radial);
            require(
                radialLength > 0.0 &&
                    contactDistance > kSupportRadialOffset,
                "support cradle radial construction is invalid"
            );
            const Vec3 outward =
                radial * (1.0 / radialLength);
            const double verticalDistance = std::sqrt(
                contactDistance * contactDistance -
                kSupportRadialOffset * kSupportRadialOffset
            );
            rigidShapes.push_back(supportSphere(
                1u + static_cast<std::uint32_t>(support),
                520210u + static_cast<std::uint32_t>(support),
                supportedCenter +
                    outward * (side * kSupportRadialOffset) -
                    Vec3{0.0, 0.0, verticalDistance}
            ));
        }

        std::vector<double> q = placementQ;
        std::vector<double> v(model.world.nv, 0.0);
        const double pickupInsertion = placementQ[2u];
        const double initialInsertion =
            pickupInsertion - kApproachDistance;
        q[2u] = initialInsertion;
        q[6u] = -kOpenJawCoordinate;
        q[7u] = kOpenJawCoordinate;

        const Vec3 initialNeedlePosition =
            vector(rigidBodies[0u].position);
        metalrobo::ArticulatedRigidWorldCache cache;
        std::uint32_t maximumSupportContacts = 0u;
        std::uint32_t settleSupportFrames = 0u;
        std::uint32_t qualifiedFrames = 0u;
        std::uint32_t graspedFrames = 0u;
        std::uint32_t graspedLiftFrames = 0u;
        std::uint32_t graspedLiftRun = 0u;
        std::uint32_t maximumGraspedLiftRun = 0u;
        bool finalLiftGrasped = false;
        std::uint32_t maximumSupportFreeRun = 0u;
        std::uint32_t supportFreeRun = 0u;
        std::uint32_t maximumWarmMatches = 0u;
        std::uint32_t maximumContacts = 0u;
        std::uint32_t prematureJawContactFrames = 0u;
        std::uint32_t approachAcquisitionContactFrames = 0u;
        std::uint32_t precloseQualifiedFrames = 0u;
        std::uint32_t maximumArticulatedDynamicContacts = 0u;
        std::uint32_t maximumArticulatedPrescribedContacts = 0u;
        std::uint32_t maximumDynamicDynamicContacts = 0u;
        std::uint32_t maximumDynamicPrescribedContacts = 0u;
        double maximumNormalImpulse = 0.0;
        double settleMaximumNormalImpulse = 0.0;
        double maximumKkt = 0.0;
        double maximumPenetration = 0.0;
        double maximumGraspSlip = 0.0;

        constexpr std::uint32_t settleSteps = 50u;
        constexpr std::uint32_t approachSteps = 180u;
        constexpr std::uint32_t closeSteps = 600u;
        constexpr std::uint32_t dwellSteps = 200u;
        constexpr std::uint32_t liftSteps = 2000u;

        const auto executePhase = [&](
            const Phase phase,
            const std::uint32_t steps
        ) {
            for (std::uint32_t localStep = 0u;
                 localStep < steps;
                 ++localStep) {
                const double fraction =
                    static_cast<double>(localStep + 1u) /
                    static_cast<double>(steps);
                std::vector<double> targets = placementQ;
                if (phase == Phase::settle) {
                    targets[2u] = initialInsertion;
                    targets[6u] = -kOpenJawCoordinate;
                    targets[7u] = kOpenJawCoordinate;
                } else if (phase == Phase::approach) {
                    targets[2u] = lerp(
                        initialInsertion,
                        pickupInsertion,
                        fraction
                    );
                    targets[6u] = -kOpenJawCoordinate;
                    targets[7u] = kOpenJawCoordinate;
                } else if (phase == Phase::close) {
                    const double jaw = lerp(
                        kOpenJawCoordinate,
                        0.0,
                        fraction
                    );
                    targets[2u] = pickupInsertion;
                    targets[6u] = -jaw;
                    targets[7u] = jaw;
                } else if (phase == Phase::dwell) {
                    targets[2u] = pickupInsertion;
                    targets[6u] = 0.0;
                    targets[7u] = 0.0;
                } else {
                    targets[2u] = lerp(
                        pickupInsertion,
                        pickupInsertion + kLiftDistance,
                        fraction
                    );
                    targets[6u] = 0.0;
                    targets[7u] = 0.0;
                }

                const std::vector<double> force = controllerForce(
                    model,
                    q,
                    v,
                    targets,
                    config.dynamics
                );
                const auto diagnostics =
                    metalrobo::stepArticulatedRigidWorldCpu(
                        model,
                        0u,
                        q,
                        v,
                        force,
                        {},
                        rigidProperties,
                        rigidBodies,
                        rigidShapes,
                        rigidMaterials,
                        {},
                        config,
                        cache
                    );
                require(
                    diagnostics.succeeded(),
                    "supported pickup step failed phase=" +
                        std::to_string(
                            static_cast<std::uint32_t>(phase)
                        ) +
                        " local_step=" +
                        std::to_string(localStep) +
                        " failure=" +
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
                const std::uint32_t supportContacts =
                    observedSupportContacts(
                        rigidShapes,
                        rigidBodies,
                        needle.rigid.shapes.size()
                    );
                maximumSupportContacts = std::max(
                    maximumSupportContacts,
                    supportContacts
                );
                maximumContacts = std::max(
                    maximumContacts,
                    diagnostics.contactCount
                );
                maximumArticulatedPrescribedContacts = std::max(
                    maximumArticulatedPrescribedContacts,
                    diagnostics.articulatedPrescribedContactCount
                );
                maximumArticulatedDynamicContacts = std::max(
                    maximumArticulatedDynamicContacts,
                    diagnostics.articulatedDynamicContactCount
                );
                maximumDynamicDynamicContacts = std::max(
                    maximumDynamicDynamicContacts,
                    diagnostics.dynamicDynamicContactCount
                );
                maximumDynamicPrescribedContacts = std::max(
                    maximumDynamicPrescribedContacts,
                    diagnostics.dynamicPrescribedContactCount
                );
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
                maximumPenetration = std::max(
                    maximumPenetration,
                    diagnostics.maximumPenetration
                );

                const auto* grasp =
                    needleGraspEvidence(diagnostics);
                if (grasp != nullptr) {
                    maximumGraspSlip = std::max(
                        maximumGraspSlip,
                        grasp->maximumTangentialSlipSpeed
                    );
                    if (grasp->qualifiedThisStep) {
                        ++qualifiedFrames;
                        if (phase == Phase::settle ||
                            phase == Phase::approach) {
                            ++precloseQualifiedFrames;
                        }
                    }
                    if (grasp->grasped) {
                        ++graspedFrames;
                        if (phase == Phase::lift) {
                            ++graspedLiftFrames;
                        }
                    }
                    if (grasp->jawAContact ||
                        grasp->jawBContact) {
                        if (phase == Phase::settle) {
                            ++prematureJawContactFrames;
                        } else if (phase == Phase::approach) {
                            ++approachAcquisitionContactFrames;
                            ++prematureJawContactFrames;
                        }
                    }
                }

                if (phase == Phase::settle) {
                    if (supportContacts >= 4u) {
                        ++settleSupportFrames;
                    }
                    settleMaximumNormalImpulse = std::max(
                        settleMaximumNormalImpulse,
                        diagnostics.maximumNormalImpulse
                    );
                }
                if (phase == Phase::lift) {
                    finalLiftGrasped =
                        grasp != nullptr && grasp->grasped;
                    if (finalLiftGrasped) {
                        ++graspedLiftRun;
                        maximumGraspedLiftRun = std::max(
                            maximumGraspedLiftRun,
                            graspedLiftRun
                        );
                    } else {
                        graspedLiftRun = 0u;
                    }
                    if (supportContacts == 0u) {
                        ++supportFreeRun;
                        maximumSupportFreeRun = std::max(
                            maximumSupportFreeRun,
                            supportFreeRun
                        );
                    } else {
                        supportFreeRun = 0u;
                    }
                }
            }
        };

        executePhase(Phase::settle, settleSteps);
        const Vec3 settledNeedlePosition =
            vector(rigidBodies[0u].position);
        require(
            maximumSupportContacts >= 4u &&
                settleSupportFrames >= settleSteps / 2u &&
                settleMaximumNormalImpulse > 0.0 &&
                norm(settledNeedlePosition - initialNeedlePosition) <
                    5.0e-4,
            "needle did not settle stably on the six-button fixture: "
                "contacts=" +
                std::to_string(maximumSupportContacts) +
                " support_frames=" +
                std::to_string(settleSupportFrames) +
                " normal_impulse=" +
                std::to_string(settleMaximumNormalImpulse) +
                " art_static=" +
                std::to_string(
                    maximumArticulatedPrescribedContacts
                ) +
                " art_dynamic=" +
                std::to_string(
                    maximumArticulatedDynamicContacts
                ) +
                " dynamic_static=" +
                std::to_string(
                    maximumDynamicPrescribedContacts
                ) +
                " kkt=" +
                std::to_string(maximumKkt) +
                " drift=" +
                std::to_string(norm(
                    settledNeedlePosition - initialNeedlePosition
                )) +
                " dz=" +
                std::to_string(
                    settledNeedlePosition.z -
                    initialNeedlePosition.z
                ) +
                " vz=" +
                std::to_string(
                    rigidBodies[0u].
                        linearVelocityAndInverseMass.z
                )
        );

        executePhase(Phase::approach, approachSteps);
        require(
            precloseQualifiedFrames == 0u &&
                prematureJawContactFrames <= 20u &&
                approachAcquisitionContactFrames <= 20u,
            "open-jaw approach formed a bilateral grasp or sustained "
            "unilateral contact: touch_frames=" +
                std::to_string(prematureJawContactFrames) +
                " approach_frames=" +
                std::to_string(approachAcquisitionContactFrames) +
                " qualified_frames=" +
                std::to_string(precloseQualifiedFrames)
        );
        executePhase(Phase::close, closeSteps);
        executePhase(Phase::dwell, dwellSteps);

        const std::uint32_t supportsAtLiftStart =
            observedSupportContacts(
                rigidShapes,
                rigidBodies,
                needle.rigid.shapes.size()
            );
        const Vec3 liftStartNeedlePosition =
            vector(rigidBodies[0u].position);
        const Quaternion liftStartNeedleOrientation =
            quaternion(rigidBodies[0u].orientation);
        const Vec3 liftStartJawMidpoint =
            jawMidpoint(model, q, v, config.dynamics);
        require(
            supportsAtLiftStart > 0u &&
                qualifiedFrames >=
                    config.grasp.requiredConsecutiveSteps &&
                graspedFrames > 0u,
            "closure did not transfer load from supported needle "
            "into a qualified bilateral grasp: supports=" +
                std::to_string(supportsAtLiftStart) +
                " qualified_frames=" +
                std::to_string(qualifiedFrames) +
                " grasped_frames=" +
                std::to_string(graspedFrames) +
                " q6=" + std::to_string(q[6u]) +
                " q7=" + std::to_string(q[7u]) +
                " v6=" + std::to_string(v[6u]) +
                " v7=" + std::to_string(v[7u]) +
                " slip=" + std::to_string(maximumGraspSlip) +
                " robot_fixture_max=" +
                std::to_string(
                    maximumArticulatedPrescribedContacts
                )
        );

        executePhase(Phase::lift, liftSteps);
        const Vec3 finalNeedlePosition =
            vector(rigidBodies[0u].position);
        const Vec3 finalJawMidpoint =
            jawMidpoint(model, q, v, config.dynamics);
        const Vec3 needleLift =
            finalNeedlePosition - liftStartNeedlePosition;
        const Vec3 jawLift =
            finalJawMidpoint - liftStartJawMidpoint;
        const double jawTravel = norm(jawLift);
        const double needleAlongJaw =
            jawTravel > 0.0
            ? dot(needleLift, jawLift) / jawTravel
            : 0.0;
        const double followRatio =
            jawTravel > 0.0 ? needleAlongJaw / jawTravel : 0.0;
        const double finalClearance = minimumSupportClearance(
            rigidShapes,
            rigidBodies,
            needle.rigid.shapes.size()
        );
        const std::uint32_t finalSupportContacts =
            observedSupportContacts(
                rigidShapes,
                rigidBodies,
                needle.rigid.shapes.size()
            );
        const Quaternion finalOrientation =
            quaternion(rigidBodies[0u].orientation);
        const double acquisitionOrientationDrift =
            2.0 * std::acos(std::clamp(
                std::abs(liftStartNeedleOrientation.w),
                0.0,
                1.0
            ));
        const double orientationDrift =
            2.0 * std::acos(std::clamp(
                std::abs(
                    liftStartNeedleOrientation.x *
                        finalOrientation.x +
                    liftStartNeedleOrientation.y *
                        finalOrientation.y +
                    liftStartNeedleOrientation.z *
                        finalOrientation.z +
                    liftStartNeedleOrientation.w *
                        finalOrientation.w
                ),
                0.0,
                1.0
            ));

        require(
            maximumWarmMatches > 0u,
            "no persistent contact warm start rematched"
        );
        require(
            maximumNormalImpulse > 0.0 &&
                maximumKkt < 2.2e-5 &&
                maximumPenetration < 5.0e-4,
            "pickup contact solve failed its impulse/KKT/penetration gate"
        );
        require(
            maximumArticulatedPrescribedContacts == 0u &&
                finalSupportContacts == 0u &&
                maximumSupportFreeRun >= 50u &&
                finalClearance > 5.0e-4,
            "needle did not unload and geometrically clear the fixture: "
            "robot_fixture=" +
                std::to_string(
                    maximumArticulatedPrescribedContacts
                ) +
                " contacts=" + std::to_string(finalSupportContacts) +
                " free_run=" +
                std::to_string(maximumSupportFreeRun) +
                " clearance=" + std::to_string(finalClearance)
        );
        require(
            finalLiftGrasped &&
                graspedLiftFrames == liftSteps &&
                maximumGraspedLiftRun == liftSteps &&
                needleLift.z > 1.0e-3 &&
                jawTravel > 2.0e-3 &&
                needleAlongJaw > 1.0e-3 &&
                followRatio > 0.35 &&
                orientationDrift < 0.35,
            "needle did not remain grasped while following the lift: "
            "grasped_lift_frames=" +
                std::to_string(graspedLiftFrames) +
                " grasped_lift_run=" +
                std::to_string(maximumGraspedLiftRun) +
                " final_grasp=" +
                std::to_string(finalLiftGrasped) +
                " dz=" + std::to_string(needleLift.z) +
                " jaw_travel=" + std::to_string(jawTravel) +
                " along=" + std::to_string(needleAlongJaw) +
                " follow_ratio=" + std::to_string(followRatio) +
                " orientation=" +
                std::to_string(orientationDrift)
        );

        const std::vector<double> qBefore = q;
        const std::vector<double> vBefore = v;
        const std::vector<MRBodyStateGPU> rigidBefore = rigidBodies;
        const metalrobo::ArticulatedRigidWorldCache cacheBefore = cache;
        std::vector<double> invalidForce(v.size(), 0.0);
        invalidForce[0u] =
            std::numeric_limits<double>::quiet_NaN();
        const auto rejected =
            metalrobo::stepArticulatedRigidWorldCpu(
                model,
                0u,
                q,
                v,
                invalidForce,
                {},
                rigidProperties,
                rigidBodies,
                rigidShapes,
                rigidMaterials,
                {},
                config,
                cache
            );
        require(
            !rejected.succeeded() &&
                q == qBefore &&
                v == vBefore &&
                rigidBodies.size() == rigidBefore.size() &&
                std::memcmp(
                    rigidBodies.data(),
                    rigidBefore.data(),
                    rigidBodies.size() *
                        sizeof(MRBodyStateGPU)
                ) == 0 &&
                sameCache(cache, cacheBefore),
            "mixed dynamic/static pickup rollback was not transactional"
        );

        std::cout
            << std::setprecision(12)
            << "supported_needle_pickup"
            << " model=" << model.name
            << " steps=" << cache.step
            << " support_buttons=" << kSupportButtonCount
            << " support_triangle_margin=" << triangleMargin
            << " support_contacts_max=" << maximumSupportContacts
            << " supports_at_lift=" << supportsAtLiftStart
            << " support_free_run=" << maximumSupportFreeRun
            << " fixture_clearance_mm=" << finalClearance * 1000.0
            << " contacts_max=" << maximumContacts
            << " art_dynamic_max="
            << maximumArticulatedDynamicContacts
            << " art_prescribed_max="
            << maximumArticulatedPrescribedContacts
            << " dynamic_dynamic_max="
            << maximumDynamicDynamicContacts
            << " dynamic_prescribed_max="
            << maximumDynamicPrescribedContacts
            << " warm_matches_max=" << maximumWarmMatches
            << " normal_impulse_max=" << maximumNormalImpulse
            << " qualified_frames=" << qualifiedFrames
            << " preclose_touch_frames="
            << prematureJawContactFrames
            << " grasped_frames=" << graspedFrames
            << " grasped_lift_frames=" << graspedLiftFrames
            << " grasped_lift_run=" << maximumGraspedLiftRun
            << " final_grasp=" << finalLiftGrasped
            << " needle_lift_mm=" << needleLift.z * 1000.0
            << " jaw_travel_mm=" << jawTravel * 1000.0
            << " follow_ratio=" << followRatio
            << " acquisition_orientation_rad="
            << acquisitionOrientationDrift
            << " orientation_drift_rad=" << orientationDrift
            << " grasp_slip_max=" << maximumGraspSlip
            << " penetration_max_mm=" << maximumPenetration * 1000.0
            << " kkt_max=" << maximumKkt
            << " grasp_shape=14"
            << " controller=computed_torque"
            << " no_weld=yes"
            << " ccd=conservative_discrete"
            << " rollback=pass"
            << " status=ok\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "supported_needle_pickup status=failed reason="
            << exception.what() << '\n';
        return 1;
    }
}
