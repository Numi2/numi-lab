#include "metalrobo/ArticulatedRigidCollision.hpp"
#include "metalrobo/SurgicalAssets.hpp"
#include "metalrobo/SurgicalPSM.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
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
    const double length = norm(value);
    require(length > 1.0e-12, "cannot normalize zero vector");
    return value * (1.0 / length);
}

Vec3 rotate(const Quaternion q, const Vec3 value) {
    const Vec3 vector{q.x, q.y, q.z};
    const Vec3 twiceCross = cross(vector, value) * 2.0;
    return
        value +
        twiceCross * q.w +
        cross(vector, twiceCross);
}

Quaternion quaternion(const mr_float4 value) {
    return {value.x, value.y, value.z, value.w};
}

Quaternion quaternion(const std::array<double, 4>& value) {
    return {value[0], value[1], value[2], value[3]};
}

Vec3 vector(const mr_float4 value) {
    return {value.x, value.y, value.z};
}

Vec3 vector(const std::array<double, 3>& value) {
    return {value[0], value[1], value[2]};
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

bool sameCache(
    const std::span<const metalrobo::PersistentManifold> left,
    const std::span<const metalrobo::PersistentManifold> right
) {
    return
        left.size() == right.size() &&
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

MRBodyStateGPU makeNeedleState(
    const metalrobo::CurvedSutureNeedleAsset& needle,
    const Vec3 position,
    const Vec3 velocity
) {
    MRBodyStateGPU result{};
    result.position = f4(position, 1.0);
    result.orientation = f4({0.0, 0.0, 0.0}, 1.0);
    result.linearVelocityAndInverseMass = f4(
        velocity,
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

MRBodyStateGPU makeSphereBody(
    const Vec3 position,
    const Vec3 velocity,
    const std::uint32_t motionType,
    const std::uint32_t generation
) {
    MRBodyStateGPU result{};
    result.position = f4(position, 1.0);
    result.orientation = f4({0.0, 0.0, 0.0}, 1.0);
    result.linearVelocityAndInverseMass = f4(
        velocity,
        motionType == MR_MOTION_DYNAMIC ? 1.0 : 0.0
    );
    result.angularVelocity = f4({0.0, 0.0, 0.0});
    if (motionType == MR_MOTION_DYNAMIC) {
        result.inverseInertiaWorldRow0 =
            {10.0F, 0.0F, 0.0F, 0.0F};
        result.inverseInertiaWorldRow1 =
            {0.0F, 10.0F, 0.0F, 0.0F};
        result.inverseInertiaWorldRow2 =
            {0.0F, 0.0F, 10.0F, 0.0F};
    }
    result.flagsAndIndices[0] = motionType;
    result.flagsAndIndices[1] = MR_INVALID_INDEX;
    result.flagsAndIndices[2] = generation;
    return result;
}

MRShapeGPU makeSphereShape(
    const std::uint32_t body,
    const std::uint32_t generation
) {
    MRShapeGPU result{};
    result.bodyIndex = body;
    result.shapeType = MR_SHAPE_SPHERE;
    result.materialIndex = 0u;
    result.collisionGroup = 1u;
    result.collisionMask = ~0u;
    result.slotGeneration = generation;
    result.localPosition = f4({0.0, 0.0, 0.0}, 1.0);
    result.localRotation = f4({0.0, 0.0, 0.0}, 1.0);
    result.dimensions = {
        0.5F,
        0.0F,
        0.0F,
        0.0F,
    };
    result.contactRestAndBoundingRadius = {
        0.0F,
        0.0F,
        0.5F,
        0.0F,
    };
    return result;
}

MRMaterialGPU makeSphereMaterial() {
    MRMaterialGPU result{};
    result.friction = {0.6F, 0.5F, 0.0F, 0.0F};
    result.response = {0.0F, 0.05F, 1.0e-8F, 0.0F};
    return result;
}

std::vector<metalrobo::ArticulatedRigidContactWarmStart>
makeWarmStarts(
    const metalrobo::ArticulatedRigidCollisionResult& collision
) {
    std::vector<metalrobo::ArticulatedRigidContactWarmStart>
        result;
    result.reserve(collision.contacts.size());
    for (std::size_t index = 0u;
         index < collision.contacts.size();
         ++index) {
        const auto& contact = collision.contacts[index];
        const Vec3 normal = vector(contact.normal);
        const Vec3 tangent = vector(contact.tangentU);
        const Vec3 impulse =
            normal * 1.0e-5 + tangent * 2.0e-6;
        result.push_back({
            .key = collision.metadata[index].key,
            .worldImpulseOnRigid = {
                impulse.x,
                impulse.y,
                impulse.z,
            },
        });
    }
    return result;
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
        const std::vector<double> velocity(model.world.nv, 0.0);
        std::vector<metalrobo::ArticulatedBodyKinematics>
            kinematics(model.articulations[0].bodyCount);
        const auto kinematicsDiagnostics =
            metalrobo::computeArticulatedBodyKinematics(
                model,
                0u,
                q,
                velocity,
                kinematics
            );
        require(
            kinematicsDiagnostics.succeeded(),
            "PSM kinematics failed"
        );

        constexpr std::uint32_t jawShapeIndex = 15u;
        require(
            jawShapeIndex < model.shapes.size() &&
                model.shapes[jawShapeIndex].shapeType ==
                    MR_SHAPE_SPHERE,
            "PSM distal jaw sphere changed"
        );
        const MRShapeGPU& jawShape =
            model.shapes[jawShapeIndex];
        const auto jawKinematics = std::ranges::find_if(
            kinematics,
            [&](const metalrobo::ArticulatedBodyKinematics& body) {
                return body.bodyIndex == jawShape.bodyIndex;
            }
        );
        require(
            jawKinematics != kinematics.end(),
            "jaw body kinematics missing"
        );
        const Quaternion jawOrientation =
            quaternion(jawKinematics->orientation);
        const Vec3 jawCenter =
            Vec3{
                jawKinematics->centerOfMassPosition[0],
                jawKinematics->centerOfMassPosition[1],
                jawKinematics->centerOfMassPosition[2],
            } +
            rotate(
                jawOrientation,
                vector(jawShape.localPosition)
            );

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
            graspShapeIndex < needle.rigid.shapes.size(),
            "needle grasp shape is out of range"
        );
        const MRShapeGPU& graspShape =
            needle.rigid.shapes[graspShapeIndex];
        require(
            graspShape.shapeType == MR_SHAPE_CAPSULE,
            "needle grasp shape is not a capsule"
        );
        const Vec3 graspAxis = normalized(rotate(
            quaternion(graspShape.localRotation),
            {0.0, 1.0, 0.0}
        ));
        const Vec3 reference =
            std::abs(graspAxis.z) < 0.8
            ? Vec3{0.0, 0.0, 1.0}
            : Vec3{1.0, 0.0, 0.0};
        const Vec3 contactNormal =
            normalized(cross(graspAxis, reference));
        constexpr double penetration = 8.0e-5;
        const double centerDistance =
            static_cast<double>(jawShape.dimensions.x) +
            static_cast<double>(graspShape.dimensions.x) -
            penetration;
        const Vec3 desiredGraspCenter =
            jawCenter + contactNormal * centerDistance;
        const Vec3 needlePosition =
            desiredGraspCenter -
            vector(graspShape.localPosition);
        const MRBodyStateGPU needleState = makeNeedleState(
            needle,
            needlePosition,
            contactNormal * -0.25
        );
        const std::array<MRBodyStateGPU, 1> rigidBodies{{
            needleState,
        }};

        metalrobo::ArticulatedRigidCollisionConfig config;
        config.collision.environment = 73u;
        config.contact.contact.timestep = 1.0 / 1000.0;
        config.contact.contact.errorReduction = 0.2;
        config.contact.contact.penetrationSlop = 1.0e-6;
        config.contact.contact.maxDepenetrationVelocity = 1.0;
        config.contact.contactCapacity = 128u;
        config.contact.qualityTangentialRegularization = 1.0e-9;

        metalrobo::PersistentManifoldCache cache;
        const auto first =
            metalrobo::collideArticulatedRigidContactsCpu(
                model,
                0u,
                q,
                velocity,
                needle.rigid.shapes,
                std::span<const MRMaterialGPU>(
                    &needle.rigid.material,
                    1u
                ),
                rigidBodies,
                cache,
                config
            );
        require(
            first.succeeded(),
            "PSM/needle collision failed: " +
                first.diagnostics.failure
        );
        require(
            !first.contacts.empty() &&
                first.contacts.size() == first.metadata.size(),
            "PSM/needle collision emitted no aligned contacts"
        );
        require(
            std::ranges::all_of(
                first.metadata,
                [&](const auto& metadata) {
                    return
                        metadata.articulatedBodyIndex ==
                            jawShape.bodyIndex &&
                        metadata.rigidBodyIndex == 0u &&
                        metadata.key.articulationIndex == 0u &&
                        metadata.key.articulatedGeneration != 0u &&
                        metadata.key.rigidGeneration != 0u;
                }
            ),
            "contact ownership or stable generation metadata is wrong"
        );
        for (std::size_t index = 1u;
             index < first.metadata.size();
             ++index) {
            require(
                !(first.metadata[index - 1u].key ==
                  first.metadata[index].key),
                "contact warm-start keys are not unique"
            );
        }

        const auto warmStarts = makeWarmStarts(first);
        const auto warmed =
            metalrobo::collideArticulatedRigidContactsCpu(
                model,
                0u,
                q,
                velocity,
                needle.rigid.shapes,
                std::span<const MRMaterialGPU>(
                    &needle.rigid.material,
                    1u
                ),
                rigidBodies,
                cache,
                config,
                warmStarts
            );
        require(
            warmed.succeeded() &&
                warmed.contacts.size() == first.contacts.size() &&
                warmed.diagnostics.matchedWarmStartCount ==
                    warmed.contacts.size(),
            "stable warm-start replay did not match every contact"
        );
        for (std::size_t index = 0u;
             index < warmed.contacts.size();
             ++index) {
            require(
                warmed.metadata[index].key ==
                    first.metadata[index].key &&
                std::abs(
                    warmed.contacts[index].warmImpulse[0] -
                    1.0e-5
                ) < 2.0e-11 &&
                std::abs(
                    warmed.contacts[index].warmImpulse[1] -
                    2.0e-6
                ) < 2.0e-11 &&
                std::abs(
                    warmed.contacts[index].warmImpulse[2]
                ) < 2.0e-11,
                "world-space warm impulse projection changed"
            );
        }

        std::vector<double> postArticulation(
            model.world.nv,
            -99.0
        );
        std::array<metalrobo::CoupledRigidBodyVelocity, 1>
            postRigid{};
        const std::array<
            metalrobo::CoupledArticulatedRigidContact,
            1
        > oneContact{{warmed.contacts.front()}};
        metalrobo::QualityContactSolverConfig solverConfig;
        solverConfig.maximumIterations = 500u;
        solverConfig.kktTolerance = 1.0e-10;
        const auto solve =
            metalrobo::solveCoupledArticulatedRigidContactsCpu(
                model,
                0u,
                q,
                velocity,
                rigidBodies,
                oneContact,
                postArticulation,
                postRigid,
                {},
                solverConfig
            );
        require(
            solve.succeeded() &&
                solve.impulses.size() == 3u &&
                solve.impulses[0] > 0.0,
            "collision-generated contact was not consumable by "
            "the coupled solver: " + solve.failure
        );

        const std::vector<metalrobo::PersistentManifold>
            cacheBeforeFailure(
                cache.entries().begin(),
                cache.entries().end()
            );
        auto overflowConfig = config;
        overflowConfig.contact.contactCapacity = 0u;
        const auto overflow =
            metalrobo::collideArticulatedRigidContactsCpu(
                model,
                0u,
                q,
                velocity,
                needle.rigid.shapes,
                std::span<const MRMaterialGPU>(
                    &needle.rigid.material,
                    1u
                ),
                rigidBodies,
                cache,
                overflowConfig
            );
        require(
            !overflow.succeeded() &&
                overflow.contacts.empty() &&
                overflow.metadata.empty() &&
                sameCache(cache.entries(), cacheBeforeFailure),
            "capacity failure published payload or manifold cache"
        );

        auto duplicateWarm = warmStarts;
        duplicateWarm.push_back(warmStarts.front());
        const auto invalidWarm =
            metalrobo::collideArticulatedRigidContactsCpu(
                model,
                0u,
                q,
                velocity,
                needle.rigid.shapes,
                std::span<const MRMaterialGPU>(
                    &needle.rigid.material,
                    1u
                ),
                rigidBodies,
                cache,
                config,
                duplicateWarm
            );
        require(
            invalidWarm.diagnostics.status ==
                metalrobo::ArticulatedRigidCollisionStatus::
                    invalidWarmStart &&
                invalidWarm.contacts.empty() &&
                invalidWarm.metadata.empty() &&
                sameCache(cache.entries(), cacheBeforeFailure),
            "duplicate warm key violated transactionality"
        );

        const std::array<MRBodyStateGPU, 3> sceneBodies{{
            makeSphereBody(
                {10.0, 0.0, 0.0},
                {0.5, 0.0, 0.0},
                MR_MOTION_DYNAMIC,
                51001u
            ),
            makeSphereBody(
                {10.9, 0.0, 0.0},
                {0.0, 0.0, 0.0},
                MR_MOTION_DYNAMIC,
                51002u
            ),
            makeSphereBody(
                {11.8, 0.0, 0.0},
                {-0.1, 0.0, 0.0},
                MR_MOTION_KINEMATIC,
                51003u
            ),
        }};
        const std::array<MRShapeGPU, 3> sceneShapes{{
            makeSphereShape(0u, 61001u),
            makeSphereShape(1u, 61002u),
            makeSphereShape(2u, 61003u),
        }};
        const std::array<MRMaterialGPU, 1> sceneMaterials{{
            makeSphereMaterial(),
        }};
        metalrobo::PersistentManifoldCache islandCache;
        const auto island =
            metalrobo::collideArticulatedRigidIslandContactsCpu(
                model,
                0u,
                q,
                velocity,
                sceneShapes,
                sceneMaterials,
                sceneBodies,
                islandCache,
                config
            );
        require(
            island.succeeded() &&
                island.contacts.size() == 2u &&
                island.diagnostics.dynamicDynamicContactCount == 1u &&
                island.diagnostics.dynamicPrescribedContactCount == 1u &&
                island.diagnostics.articulatedDynamicContactCount == 0u &&
                island.diagnostics.articulatedPrescribedContactCount == 0u,
            "full-scene adapter did not emit both rigid pair classes"
        );
        std::vector<
            metalrobo::ArticulatedRigidIslandContactWarmStart>
            islandWarmStarts;
        islandWarmStarts.reserve(island.contacts.size());
        for (const auto& metadata : island.metadata) {
            islandWarmStarts.push_back({
                .key = metadata.key,
                .worldImpulseOnB = {1.0e-5, 0.0, 0.0},
            });
        }
        const auto warmedIsland =
            metalrobo::collideArticulatedRigidIslandContactsCpu(
                model,
                0u,
                q,
                velocity,
                sceneShapes,
                sceneMaterials,
                sceneBodies,
                islandCache,
                config,
                islandWarmStarts
            );
        require(
            warmedIsland.succeeded() &&
                warmedIsland.diagnostics.matchedWarmStartCount ==
                    warmedIsland.contacts.size(),
            "full-scene warm starts did not rematch"
        );

        std::vector<double> islandPostArticulation(
            model.world.nv,
            -99.0
        );
        std::array<metalrobo::CoupledRigidBodyVelocity, 3>
            islandPostScene{};
        metalrobo::QualityContactSolverConfig islandSolverConfig;
        islandSolverConfig.maximumIterations = 1000u;
        islandSolverConfig.kktTolerance = 1.0e-9;
        const auto islandSolve =
            metalrobo::solveCoupledArticulatedRigidIslandCpu(
                model,
                0u,
                q,
                velocity,
                sceneBodies,
                warmedIsland.contacts,
                islandPostArticulation,
                islandPostScene,
                {},
                islandSolverConfig
            );
        require(
            islandSolve.succeeded() &&
                islandSolve.dynamicRigidBodyCount == 2u &&
                islandSolve.prescribedRigidBodyCount == 1u &&
                islandSolve.impulses.size() == 6u &&
                std::ranges::all_of(
                    islandPostArticulation,
                    [](const double item) {
                        return std::abs(item) < 1.0e-12;
                    }
                ) &&
                std::abs(
                    islandPostScene[2u].linear[0u] + 0.1
                ) < 1.0e-7 &&
                std::abs(islandPostScene[2u].linear[1u]) <
                    1.0e-12 &&
                std::abs(islandPostScene[2u].linear[2u]) <
                    1.0e-12 &&
                islandPostScene[2u].angular ==
                    std::array<double, 3>{0.0, 0.0, 0.0},
            "generic mixed solver did not consume both rigid pair classes: " +
                islandSolve.failure
        );

        std::cout
            << std::setprecision(12)
            << "articulated_rigid_collision"
            << " model=" << model.name
            << " articulated_shapes="
            << first.diagnostics.articulatedShapeCount
            << " rigid_shapes="
            << first.diagnostics.rigidShapeCount
            << " contacts=" << first.contacts.size()
            << " warm_matches="
            << warmed.diagnostics.matchedWarmStartCount
            << " penetration="
            << first.diagnostics.maximumPenetration
            << " normal_impulse=" << solve.impulses[0]
            << " cache_entries=" << cache.size()
            << " island_contacts=" << island.contacts.size()
            << " dynamic_dynamic="
            << island.diagnostics.dynamicDynamicContactCount
            << " dynamic_prescribed="
            << island.diagnostics.dynamicPrescribedContactCount
            << " island_warm_matches="
            << warmedIsland.diagnostics.matchedWarmStartCount
            << " status=ok\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "articulated_rigid_collision status=failed reason="
            << exception.what() << '\n';
        return 1;
    }
}
