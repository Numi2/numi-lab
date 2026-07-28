#include "metalrobo/RigidBodyWorld.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

mr_float4 f4(
    const float x,
    const float y,
    const float z,
    const float w = 0.0f
) {
    return {x, y, z, w};
}

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

MRBodyPropertiesGPU makeProperties(
    const MRMotionType motion,
    const float mass,
    const float inertia
) {
    MRBodyPropertiesGPU result{};
    result.articulationIndex = MR_INVALID_INDEX;
    result.parentBody = MR_INVALID_INDEX;
    result.inboundJoint = MR_INVALID_INDEX;
    result.motionType = motion;
    const float inverseMass = mass > 0.0f ? 1.0f / mass : 0.0f;
    const float inverseInertia =
        inertia > 0.0f ? 1.0f / inertia : 0.0f;
    result.massAndInverseMass = f4(mass, inverseMass, 0.0f, 0.0f);
    result.inertiaRow0 = f4(inertia, 0.0f, 0.0f);
    result.inertiaRow1 = f4(0.0f, inertia, 0.0f);
    result.inertiaRow2 = f4(0.0f, 0.0f, inertia);
    result.inverseInertiaRow0 =
        f4(inverseInertia, 0.0f, 0.0f);
    result.inverseInertiaRow1 =
        f4(0.0f, inverseInertia, 0.0f);
    result.inverseInertiaRow2 =
        f4(0.0f, 0.0f, inverseInertia);
    result.dampingAndSpeedLimits =
        f4(0.02f, 0.02f, 100.0f, 100.0f);
    return result;
}

MRBodyStateGPU makeState(
    const std::uint32_t index,
    const MRMotionType motion,
    const float y,
    const float inverseMass,
    const float inverseInertia
) {
    MRBodyStateGPU result{};
    result.position = f4(0.0f, y, 0.0f, 1.0f);
    result.orientation = f4(0.0f, 0.0f, 0.0f, 1.0f);
    result.linearVelocityAndInverseMass =
        f4(0.0f, 0.0f, 0.0f, inverseMass);
    result.inverseInertiaWorldRow0 =
        f4(inverseInertia, 0.0f, 0.0f);
    result.inverseInertiaWorldRow1 =
        f4(0.0f, inverseInertia, 0.0f);
    result.inverseInertiaWorldRow2 =
        f4(0.0f, 0.0f, inverseInertia);
    result.flagsAndIndices[0] = motion;
    result.flagsAndIndices[1] = MR_INVALID_INDEX;
    result.flagsAndIndices[2] = index;
    return result;
}

MRShapeGPU makeShape(
    const std::uint32_t body,
    const MRShapeType type,
    const float radius,
    const std::uint32_t generation
) {
    MRShapeGPU result{};
    result.bodyIndex = body;
    result.shapeType = type;
    result.materialIndex = 0u;
    result.collisionGroup = 1u;
    result.collisionMask = 1u;
    result.slotGeneration = generation;
    result.localPosition = f4(0.0f, 0.0f, 0.0f, 1.0f);
    result.localRotation = f4(0.0f, 0.0f, 0.0f, 1.0f);
    result.dimensions = f4(radius, 0.0f, 0.0f, 0.0f);
    result.contactRestAndBoundingRadius =
        f4(0.02f, 0.0f, radius, 0.0f);
    return result;
}

struct Scene {
    std::array<MRBodyPropertiesGPU, 3> properties;
    std::array<MRBodyStateGPU, 3> states;
    std::array<MRShapeGPU, 3> shapes;
    std::array<MRMaterialGPU, 1> materials;
    std::array<metalrobo::BodyWrench, 3> wrenches{};
};

Scene makeScene() {
    Scene result{};
    result.properties = {
        makeProperties(MR_MOTION_STATIC, 0.0f, 0.0f),
        makeProperties(MR_MOTION_DYNAMIC, 1.0f, 0.1f),
        makeProperties(MR_MOTION_DYNAMIC, 1.5f, 0.15f),
    };
    result.states = {
        makeState(0u, MR_MOTION_STATIC, 0.0f, 0.0f, 0.0f),
        makeState(1u, MR_MOTION_DYNAMIC, 0.5f, 1.0f, 10.0f),
        makeState(
            2u,
            MR_MOTION_DYNAMIC,
            1.5f,
            1.0f / 1.5f,
            1.0f / 0.15f
        ),
    };
    result.shapes = {
        makeShape(0u, MR_SHAPE_PLANE, 0.0f, 10u),
        makeShape(1u, MR_SHAPE_SPHERE, 0.5f, 11u),
        makeShape(2u, MR_SHAPE_SPHERE, 0.5f, 12u),
    };
    result.materials[0].friction = f4(0.8f, 0.6f, 0.0f, 0.02f);
    result.materials[0].response = f4(0.0f, 0.5f, 0.0f, 0.0f);
    return result;
}

metalrobo::RigidBodyWorldConfig makeConfig() {
    metalrobo::RigidBodyWorldConfig result;
    result.freeMotion.timestep = 1.0 / 240.0;
    result.freeMotion.gravity = f4(0.0f, -9.81f, 0.0f, 0.0f);
    result.freeMotion.integrator =
        metalrobo::FreeBodyIntegrator::symplecticEuler;
    result.contact.timestep = result.freeMotion.timestep;
    result.contact.velocityIterations = 64u;
    result.contact.enableEarlyExit = false;
    result.contact.deterministic = true;
    result.contact.errorReduction = 0.25;
    result.contact.penetrationSlop = 2.0e-4;
    result.collision.capacities = {
        .pairCapacity = 16u,
        .rawContactCapacity = 16u,
        .manifoldCapacity = 16u,
    };
    result.constraintCapacity = 16u;
    return result;
}

struct RunResult {
    Scene scene;
    metalrobo::RigidBodyWorldCache cache;
    double maximumPenetration = 0.0;
    std::uint32_t maximumContacts = 0u;
    std::uint32_t warmFrames = 0u;
    double maximumQualityKkt = 0.0;
};

RunResult runStack(
    const std::uint32_t stepCount = 1200u,
    const MRSolverType solverType = MR_SOLVER_THROUGHPUT_PGS
) {
    RunResult result{.scene = makeScene()};
    auto config = makeConfig();
    config.solverType = solverType;
    if (solverType == MR_SOLVER_QUALITY_NEWTON) {
        // The current quality QP has a three-dimensional exact Coulomb cone;
        // distinct static/dynamic, rolling, and torsional dimensions are
        // intentionally rejected.
        result.scene.materials[0].friction.x =
            result.scene.materials[0].friction.y;
        result.scene.materials[0].friction.w = 0.0f;
        config.quality.kktTolerance = 1.0e-10;
    }
    for (std::uint32_t step = 0u; step < stepCount; ++step) {
        const auto diagnostics = metalrobo::stepRigidBodyWorldCpu(
            result.scene.properties,
            result.scene.states,
            result.scene.shapes,
            result.scene.materials,
            result.scene.wrenches,
            config,
            result.cache
        );
        if (!diagnostics.succeeded()) {
            throw std::runtime_error(
                "world step failed status=" +
                std::to_string(diagnostics.code) +
                " free=" +
                std::to_string(diagnostics.freeMotion.code) +
                " collision=" +
                std::to_string(diagnostics.collision.code) +
                " assembly=" +
                std::to_string(diagnostics.assembly.code) +
                " solver=" +
                std::to_string(diagnostics.solver.code) +
                " quality=" +
                std::to_string(diagnostics.qualitySolver.code)
            );
        }
        result.maximumPenetration = std::max(
            result.maximumPenetration,
            diagnostics.maximumPenetration
        );
        result.maximumContacts = std::max(
            result.maximumContacts,
            diagnostics.contactCount
        );
        if (solverType == MR_SOLVER_QUALITY_NEWTON) {
            result.maximumQualityKkt = std::max(
                result.maximumQualityKkt,
                diagnostics.qualitySolver.scaledKktCertificate
            );
        }
        if (result.cache.impulses.size() >= 2u) {
            ++result.warmFrames;
        }
    }
    return result;
}

bool sameStates(
    const std::array<MRBodyStateGPU, 3>& left,
    const std::array<MRBodyStateGPU, 3>& right
) {
    return std::memcmp(left.data(), right.data(), sizeof(left)) == 0;
}

void verifyTransactionalOverflow() {
    Scene scene = makeScene();
    const auto original = scene.states;
    auto config = makeConfig();
    config.constraintCapacity = 1u;
    metalrobo::RigidBodyWorldCache cache;
    const auto diagnostics = metalrobo::stepRigidBodyWorldCpu(
        scene.properties,
        scene.states,
        scene.shapes,
        scene.materials,
        scene.wrenches,
        config,
        cache
    );
    require(
        diagnostics.code == MR_STEP_CONSTRAINT_CAPACITY_OVERFLOW,
        "constraint overflow status changed"
    );
    require(
        diagnostics.assembly.requiredConstraints == 2u,
        "constraint preflight count changed"
    );
    require(sameStates(scene.states, original), "overflow mutated state");
    require(cache.step == 0u, "overflow published cache state");
}

void verifyReducedManifoldAssemblyAndRestOffset() {
    std::array<MRBodyStateGPU, 2> bodies{
        makeState(0u, MR_MOTION_STATIC, 0.0f, 0.0f, 0.0f),
        makeState(1u, MR_MOTION_DYNAMIC, -0.5f, 1.0f, 6.0f),
    };
    std::array<MRShapeGPU, 2> shapes{
        makeShape(0u, MR_SHAPE_PLANE, 0.0f, 30u),
        makeShape(1u, MR_SHAPE_BOX, 0.5f, 31u),
    };
    shapes[0].contactRestAndBoundingRadius =
        f4(0.02f, 0.01f, 0.0f, 0.0f);
    shapes[1].dimensions = f4(0.5f, 0.5f, 0.5f, 0.0f);
    shapes[1].contactRestAndBoundingRadius =
        f4(0.03f, 0.02f, 0.9f, 0.0f);

    metalrobo::CollisionConfig collisionConfig;
    collisionConfig.capacities = {
        .pairCapacity = 8u,
        .rawContactCapacity = 16u,
        .manifoldCapacity = 8u,
    };
    metalrobo::PersistentManifoldCache manifoldCache;
    const auto collision = metalrobo::collideCpuReference(
        shapes,
        bodies,
        collisionConfig,
        manifoldCache
    );
    require(collision.succeeded(), "box/plane collision failed");
    require(
        collision.rawContacts.size() == 8u &&
            collision.manifoldHeaders.size() == 1u &&
            collision.manifoldHeaders[0].pairAndCount[3] == 4u,
        "box manifold was not reduced from eight to four"
    );

    std::array<MRMaterialGPU, 1> materials{};
    materials[0].friction = f4(0.7f, 0.5f, 0.0f, 0.0f);
    materials[0].response = f4(0.0f, 0.5f, 0.0f, 0.0f);
    const auto withRest = metalrobo::assembleContactConstraints(
        collision,
        shapes,
        materials,
        bodies,
        8u
    );
    auto zeroRestShapes = shapes;
    zeroRestShapes[0].contactRestAndBoundingRadius.y = 0.0f;
    zeroRestShapes[1].contactRestAndBoundingRadius.y = 0.0f;
    const auto withoutRest = metalrobo::assembleContactConstraints(
        collision,
        zeroRestShapes,
        materials,
        bodies,
        8u
    );
    require(
        withRest.diagnostics.succeeded() &&
            withoutRest.diagnostics.succeeded() &&
            withRest.constraints.size() == 4u &&
            withoutRest.constraints.size() == 4u,
        "solver assembly ignored reduced manifold cardinality"
    );
    for (std::size_t index = 0u;
         index < withRest.constraints.size();
         ++index) {
        require(
            withRest.constraints[index].featureKey ==
                    withoutRest.constraints[index].featureKey &&
                std::abs(
                    (
                        withRest.constraints[index]
                            .pointAndSeparation.w -
                        withoutRest.constraints[index]
                            .pointAndSeparation.w
                    ) +
                    0.03f
                ) < 2.0e-6f,
            "shape rest offsets did not shift solver separation"
        );
    }
    const auto overflow = metalrobo::assembleContactConstraints(
        collision,
        shapes,
        materials,
        bodies,
        3u
    );
    require(
        overflow.diagnostics.code ==
                MR_STEP_CONSTRAINT_CAPACITY_OVERFLOW &&
            overflow.diagnostics.requiredConstraints == 4u &&
            overflow.constraints.empty(),
        "reduced-manifold capacity preflight is not transactional"
    );
}

void verifySceneLevelIslandBatching() {
    constexpr std::uint32_t kDynamicBodies = 129u;
    std::vector<MRBodyPropertiesGPU> properties;
    std::vector<MRBodyStateGPU> states;
    std::vector<MRShapeGPU> shapes;
    std::vector<metalrobo::BodyWrench> wrenches(kDynamicBodies + 1u);
    properties.reserve(kDynamicBodies + 1u);
    states.reserve(kDynamicBodies + 1u);
    shapes.reserve(kDynamicBodies + 1u);
    properties.push_back(makeProperties(MR_MOTION_STATIC, 0.0f, 0.0f));
    states.push_back(
        makeState(0u, MR_MOTION_STATIC, 0.0f, 0.0f, 0.0f)
    );
    shapes.push_back(makeShape(0u, MR_SHAPE_PLANE, 0.0f, 100u));
    for (std::uint32_t index = 0u;
         index < kDynamicBodies;
         ++index) {
        const std::uint32_t bodyIndex = index + 1u;
        properties.push_back(
            makeProperties(MR_MOTION_DYNAMIC, 1.0f, 0.1f)
        );
        states.push_back(
            makeState(
                bodyIndex,
                MR_MOTION_DYNAMIC,
                0.5f,
                1.0f,
                10.0f
            )
        );
        states.back().position.x = 2.0f * static_cast<float>(index);
        shapes.push_back(
            makeShape(
                bodyIndex,
                MR_SHAPE_SPHERE,
                0.5f,
                101u + index
            )
        );
    }
    std::array<MRMaterialGPU, 1> materials{};
    materials[0].friction = f4(0.6f, 0.5f, 0.0f, 0.0f);
    materials[0].response = f4(0.0f, 0.5f, 0.0f, 0.0f);

    auto config = makeConfig();
    config.constraintCapacity = 256u;
    config.collision.capacities = {
        .pairCapacity = 256u,
        .rawContactCapacity = 256u,
        .manifoldCapacity = 256u,
    };
    metalrobo::RigidBodyWorldCache cache;
    const auto diagnostics = metalrobo::stepRigidBodyWorldCpu(
        properties,
        states,
        shapes,
        materials,
        wrenches,
        config,
        cache
    );
    require(
        diagnostics.succeeded() &&
            diagnostics.contactCount == kDynamicBodies &&
            diagnostics.solver.islandCount == kDynamicBodies,
        "scene-level contact islands were treated as one 128-row batch"
    );
}

void verifyQualityRejectsDivergentFriction() {
    Scene scene = makeScene();
    scene.materials[0].friction.w = 0.0f;
    const auto original = scene.states;
    auto config = makeConfig();
    config.solverType = MR_SOLVER_QUALITY_NEWTON;
    metalrobo::RigidBodyWorldCache cache;
    const auto diagnostics = metalrobo::stepRigidBodyWorldCpu(
        scene.properties,
        scene.states,
        scene.shapes,
        scene.materials,
        scene.wrenches,
        config,
        cache
    );
    require(
        diagnostics.code == MR_STEP_UNSUPPORTED,
        "quality mode silently changed static/dynamic friction semantics"
    );
    require(
        sameStates(scene.states, original) && cache.step == 0u,
        "unsupported quality friction mutated world state"
    );
}

} // namespace

int main() {
    try {
        const RunResult first = runStack();
        const RunResult second = runStack();
        const RunResult quality = runStack(
            240u,
            MR_SOLVER_QUALITY_NEWTON
        );
        require(
            sameStates(first.scene.states, second.scene.states),
            "deterministic replay diverged"
        );
        require(first.maximumContacts == 2u, "stack contacts missing");
        require(first.warmFrames > 1000u, "warm-start cache did not persist");
        require(
            first.maximumPenetration < 0.006,
            "stack penetration exceeded bound"
        );
        require(
            first.scene.states[1].position.y > 0.49f &&
                first.scene.states[2].position.y > 1.47f,
            "stack collapsed"
        );
        require(
            std::abs(
                first.scene.states[1]
                    .linearVelocityAndInverseMass.y
            ) < 0.03f &&
                std::abs(
                    first.scene.states[2]
                        .linearVelocityAndInverseMass.y
                ) < 0.03f,
            "stack did not settle"
        );
        require(
            quality.maximumContacts == 2u &&
                quality.maximumQualityKkt <= 1.1e-10 &&
                quality.scene.states[1].position.y > 0.49f &&
                quality.scene.states[2].position.y > 1.49f,
            "quality solver was not integrated into the world step"
        );
        verifyTransactionalOverflow();
        verifyReducedManifoldAssemblyAndRestOffset();
        verifySceneLevelIslandBatching();
        verifyQualityRejectsDivergentFriction();

        std::cout << std::scientific << std::setprecision(6)
                  << "pipeline=cpu_rigid_world"
                  << " steps=1200"
                  << " contacts_max=" << first.maximumContacts
                  << " warm_frames=" << first.warmFrames
                  << " penetration_max="
                  << first.maximumPenetration
                  << " bottom_y="
                  << first.scene.states[1].position.y
                  << " top_y="
                  << first.scene.states[2].position.y
                  << " bottom_vy="
                  << first.scene.states[1]
                         .linearVelocityAndInverseMass.y
                  << " top_vy="
                  << first.scene.states[2]
                         .linearVelocityAndInverseMass.y
                  << " quality_steps=240"
                  << " quality_kkt_max="
                  << quality.maximumQualityKkt
                  << " manifold_constraints=4"
                  << " rest_offsets=yes"
                  << " island_batched_contacts=129"
                  << " quality_friction_rejection=transactional"
                  << " deterministic=yes"
                  << " overflow_transactional=yes"
                  << " status=ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "metalrobo_rigid_body_world_probe: "
                  << error.what() << '\n';
        return 1;
    }
}
