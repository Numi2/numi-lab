#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/EngineModel.hpp"
#include "metalrobo/Franka.hpp"
#include "metalrobo/MetalWorld.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

double percentile(
    std::vector<double> values,
    const double probability
) {
    require(
        !values.empty() &&
            probability >= 0.0 &&
            probability <= 1.0,
        "invalid percentile request"
    );
    std::sort(values.begin(), values.end());
    const double position =
        probability * static_cast<double>(values.size() - 1u);
    const auto lower =
        static_cast<std::size_t>(std::floor(position));
    const auto upper =
        static_cast<std::size_t>(std::ceil(position));
    const double fraction =
        position - static_cast<double>(lower);
    return values[lower] +
        fraction * (values[upper] - values[lower]);
}

MRBodyStateGPU groundState() {
    MRBodyStateGPU state{};
    state.position.w = 1.0f;
    state.orientation.w = 1.0f;
    state.flagsAndIndices[0] = MR_MOTION_STATIC;
    state.flagsAndIndices[1] = MR_INVALID_INDEX;
    state.flagsAndIndices[2] = 0u;
    return state;
}

mr_float4 f4(
    const float x,
    const float y,
    const float z,
    const float w = 0.0f
) {
    return {x, y, z, w};
}

MRBodyPropertiesGPU cubeProperties() {
    constexpr float inertia = 1.0f / 600.0f;
    constexpr float inverseInertia = 600.0f;
    MRBodyPropertiesGPU body{};
    body.articulationIndex = MR_INVALID_INDEX;
    body.parentBody = MR_INVALID_INDEX;
    body.inboundJoint = MR_INVALID_INDEX;
    body.motionType = MR_MOTION_DYNAMIC;
    body.massAndInverseMass = f4(1.0f, 1.0f, 0.0f, 0.0f);
    body.inertiaRow0 = f4(inertia, 0.0f, 0.0f);
    body.inertiaRow1 = f4(0.0f, inertia, 0.0f);
    body.inertiaRow2 = f4(0.0f, 0.0f, inertia);
    body.inverseInertiaRow0 =
        f4(inverseInertia, 0.0f, 0.0f);
    body.inverseInertiaRow1 =
        f4(0.0f, inverseInertia, 0.0f);
    body.inverseInertiaRow2 =
        f4(0.0f, 0.0f, inverseInertia);
    body.dampingAndSpeedLimits =
        f4(0.01f, 0.01f, 100.0f, 100.0f);
    return body;
}

MRShapeGPU cubeShape(const std::uint32_t bodyIndex) {
    MRShapeGPU shape{};
    shape.bodyIndex = bodyIndex;
    shape.shapeType = MR_SHAPE_BOX;
    shape.materialIndex = 0u;
    shape.collisionGroup = 1u;
    shape.collisionMask = ~0u;
    shape.slotGeneration = 1u;
    shape.localPosition.w = 1.0f;
    shape.localRotation.w = 1.0f;
    shape.dimensions = f4(0.05f, 0.05f, 0.05f);
    shape.contactRestAndBoundingRadius =
        f4(0.002f, 0.0f, 0.08660254f);
    return shape;
}

} // namespace

int main() {
    try {
        constexpr std::size_t environmentCount = 4u;
        constexpr std::size_t controlStepCount = 12u;
        metalrobo::EngineModel model =
            metalrobo::makeFreeSphereEngineModel();
        metalrobo::CompiledWorld world;
        const auto compiled = metalrobo::compileMetalWorld(
            model,
            0u,
            world
        );
        require(
            compiled.succeeded() &&
                world.sceneBodyCount() == 1u &&
                world.eligiblePairCount() == 1u &&
                world.minimumCapacities().rawContacts >=
                    MR_METAL_WORLD_RAW_CONTACTS_PER_PAIR,
            "contact world compilation failed"
        );

        std::vector<float> q(
            environmentCount * world.nq()
        );
        std::vector<float> v(
            environmentCount * world.nv(),
            0.0f
        );
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            std::copy(
                model.defaultQ.begin(),
                model.defaultQ.end(),
                q.begin() +
                    static_cast<std::ptrdiff_t>(
                        environment * world.nq()
                    )
            );
            q[environment * world.nq() + 1u] =
                0.245f + 0.0005f * environment;
        }
        std::vector<float> efforts(
            controlStepCount * environmentCount * world.nv(),
            0.0f
        );
        std::vector<MRBodyStateGPU> scene(
            environmentCount,
            groundState()
        );
        const metalrobo::MetalWorldBatch batch{
            .environmentCount = environmentCount,
            .controlStepCount = controlStepCount,
            .initialQ = q,
            .initialV = v,
            .efforts = efforts,
            .initialSceneBodies = scene,
        };
        const metalrobo::MetalWorldStepConfig config{
            .timestepSeconds = 1.0f / 120.0f,
            .physicsSubsteps = 4u,
            .solverMode =
                metalrobo::MetalWorldSolverMode::throughputTGS,
            .velocityIterations = 2u,
            .finalVelocityIterations = 1u,
            .deterministic = true,
            .warmStart = true,
            .captureContactEvidence = true,
        };

        metalrobo::MetalWorldContext context;
        metalrobo::MetalWorldResult first;
        const auto firstDiagnostics =
            context.run(world, batch, config, first);
        require(
            firstDiagnostics.succeeded(),
            firstDiagnostics.message.c_str()
        );
        require(
            firstDiagnostics.successfulStepCount ==
                environmentCount * controlStepCount &&
                first.contactStatuses.size() ==
                    environmentCount * controlStepCount &&
                first.finalSceneBodies.size() ==
                    environmentCount &&
                first.environmentStatuses.size() ==
                    environmentCount &&
                first.contactEvidence.manifoldCounts.size() ==
                    environmentCount &&
                first.contactStatuses.back().solverIterations ==
                    config.velocityIterations +
                        config.finalVelocityIterations,
            "contact result dimensions or accounting are invalid"
        );
        require(
            std::any_of(
                first.contactStatuses.begin(),
                first.contactStatuses.end(),
                [](const MRMetalWorldContactStatusGPU& status) {
                    return status.activeContacts > 0u;
                }
            ),
            "resting sphere never produced a device contact"
        );
        std::uint64_t retainedPoints = 0u;
        std::uint64_t observedPoints = 0u;
        for (std::size_t step = 1u;
             step < controlStepCount;
             ++step) {
            for (std::size_t environment = 0u;
                 environment < environmentCount;
                 ++environment) {
                const auto& status =
                    first.contactStatuses[
                        step * environmentCount + environment
                    ];
                retainedPoints += status.retainedPoints;
                observedPoints +=
                    status.retainedPoints + status.newPoints;
            }
        }
        require(
            observedPoints > 0u &&
                100u * retainedPoints >=
                    99u * observedPoints,
            "resting persistent-manifold retention fell below 99 percent"
        );
        require(
            std::all_of(
                first.environmentStatuses.begin(),
                first.environmentStatuses.end(),
                [](const metalrobo::MetalWorldStatus& status) {
                    return
                        status.code == MR_STEP_SUCCESS &&
                        status.failedControlSteps == 0u &&
                        status.required.constraintBlocks > 0u &&
                        status.highWater.constraintBlocks > 0u &&
                        status.manifoldRetention >= 0.99f;
                }
            ),
            "per-environment contact summaries are incomplete"
        );

        metalrobo::MetalWorldResult replay;
        const auto replayDiagnostics =
            context.run(world, batch, config, replay);
        require(
            replayDiagnostics.succeeded() &&
                first.finalQ == replay.finalQ &&
                first.finalV == replay.finalV &&
                first.observations == replay.observations,
            "deterministic contact replay diverged"
        );

        metalrobo::EngineModel franka =
            metalrobo::makeFrankaPandaEngineModel();
        const MRShapeGPU witnessShape = franka.shapes.back();
        std::vector<double> frankaQ(
            franka.defaultQ.begin(),
            franka.defaultQ.end()
        );
        std::vector<double> frankaV(
            franka.defaultV.begin(),
            franka.defaultV.end()
        );
        const metalrobo::ArticulatedPointQuery witnessQuery{
            .bodyIndex = witnessShape.bodyIndex,
            .localPoint = {
                witnessShape.localPosition.x,
                witnessShape.localPosition.y,
                witnessShape.localPosition.z,
            },
        };
        metalrobo::ArticulatedPointKinematics witness{};
        std::vector<double> witnessJacobian(
            3u * franka.articulations[0].nv
        );
        const auto witnessDiagnostics =
            metalrobo::computeArticulatedPointJacobians(
                franka,
                0u,
                frankaQ,
                frankaV,
                std::span{&witnessQuery, 1u},
                std::span{&witness, 1u},
                witnessJacobian
            );
        require(
            witnessDiagnostics.succeeded(),
            "Franka witness kinematics failed"
        );
        const std::uint32_t cubeBody =
            static_cast<std::uint32_t>(franka.bodies.size());
        franka.bodies.push_back(cubeProperties());
        franka.shapes.push_back(cubeShape(cubeBody));
        franka.world.bodyCount =
            static_cast<std::uint32_t>(franka.bodies.size());
        franka.world.shapeCount =
            static_cast<std::uint32_t>(franka.shapes.size());

        metalrobo::CompiledWorld frankaWorld;
        const auto frankaCompiled =
            metalrobo::compileMetalWorld(
                franka,
                0u,
                frankaWorld
            );
        require(
            frankaCompiled.succeeded() &&
                frankaWorld.sceneBodyCount() == 1u &&
                std::none_of(
                    frankaWorld.eligiblePairs().begin(),
                    frankaWorld.eligiblePairs().end(),
                    [](const MRCompiledCollisionPairGPU& pair) {
                        return pair.pairClass ==
                            MR_COLLISION_PAIR_UNSUPPORTED;
                    }
                ),
            "Franka-plus-cube compilation failed"
        );

        MRBodyStateGPU cube{};
        cube.position = f4(
            static_cast<float>(witness.position[0]) +
                witnessShape.dimensions.x + 0.047f,
            static_cast<float>(witness.position[1]),
            static_cast<float>(witness.position[2]),
            1.0f
        );
        cube.orientation.w = 1.0f;
        cube.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
        cube.flagsAndIndices[1] = MR_INVALID_INDEX;
        cube.flagsAndIndices[2] = cubeBody;
        const std::vector<float> frankaEfforts(
            2u * frankaWorld.nv(),
            0.0f
        );
        const std::vector<MRBodyStateGPU> cubeScene{cube};
        const metalrobo::MetalWorldBatch frankaBatch{
            .environmentCount = 1u,
            .controlStepCount = 2u,
            .initialQ = franka.defaultQ,
            .initialV = franka.defaultV,
            .efforts = frankaEfforts,
            .initialSceneBodies = cubeScene,
        };
        metalrobo::MetalWorldResult frankaResult;
        const auto frankaRun = context.run(
            frankaWorld,
            frankaBatch,
            config,
            frankaResult
        );
        if (!frankaRun.succeeded()) {
            std::cerr
                << "franka_status="
                << frankaRun.firstGPUStatusCode
                << " environment="
                << frankaRun.firstFailingEnvironment
                << " step="
                << frankaRun.firstFailingControlStep;
            if (!frankaResult.contactStatuses.empty()) {
                const auto& failed =
                    frankaResult.contactStatuses[
                        frankaRun.firstFailingControlStep
                    ];
                std::cerr
                    << " contact_code=" << failed.code
                    << " substep=" << failed.physicsSubstep
                    << " pair=" << failed.firstFailingPair
                    << " constraint="
                    << failed.firstFailingConstraint
                    << " required_pairs="
                    << failed.requiredPairs
                    << " required_contacts="
                    << failed.requiredConstraints
                    << " min_pivot="
                    << failed.diagnostics.z
                    << " max_pivot="
                    << failed.diagnostics.w
                    << " factor_residual="
                    << failed.residuals.w;
                if (!frankaResult.contactEvidence.contacts.empty() &&
                    !frankaResult.contactEvidence
                         .evaluatedRows.empty()) {
                    const auto& contact =
                        frankaResult.contactEvidence.contacts[0];
                    const auto& row =
                        frankaResult.contactEvidence
                            .evaluatedRows[0];
                    const auto& metadata =
                        frankaResult.contactEvidence
                            .contactMetadata[0];
                    std::cerr
                        << " body_a=" << contact.bodyA
                        << " body_b=" << contact.bodyB
                        << " normal=(" << contact.normal.x
                        << ',' << contact.normal.y
                        << ',' << contact.normal.z << ')'
                        << " eval_dir=(" << row.direction.x
                        << ',' << row.direction.y
                        << ',' << row.direction.z << ')'
                        << " anchor_a=("
                        << metadata.localAnchorA.x << ','
                        << metadata.localAnchorA.y << ','
                        << metadata.localAnchorA.z << ')'
                        << " anchor_b=("
                        << metadata.localAnchorB.x << ','
                        << metadata.localAnchorB.y << ','
                        << metadata.localAnchorB.z << ')';
                }
            }
            std::cerr << '\n';
        }
        require(
            frankaRun.succeeded() &&
                frankaResult.contactStatuses.back()
                        .activeContacts > 0u,
            frankaRun.message.c_str()
        );
        const auto& frankaStatus =
            frankaResult.contactStatuses.back();
        const std::size_t contactBase = 0u;
        require(
            std::any_of(
                frankaResult.contactEvidence.contacts.begin() +
                    static_cast<std::ptrdiff_t>(contactBase),
                frankaResult.contactEvidence.contacts.begin() +
                    static_cast<std::ptrdiff_t>(
                        contactBase +
                        frankaStatus.activeContacts
                    ),
                [cubeBody](const MRContactConstraintGPU& contact) {
                    return contact.bodyA == cubeBody ||
                        contact.bodyB == cubeBody;
                }
            ),
            "Franka-plus-cube graph did not compile a mixed contact"
        );

        metalrobo::CompiledWorld overflowWorld;
        const auto overflowCompiled =
            metalrobo::compileMetalWorld(
                franka,
                0u,
                overflowWorld,
                metalrobo::MetalWorldCapacityProfile{
                    .rawContacts = 1u,
                }
            );
        require(
            overflowCompiled.succeeded() &&
                overflowWorld.capacities().rawContacts == 1u,
            "runtime-overflow world compilation failed"
        );
        std::vector<float> overflowQ(
            2u * overflowWorld.nq()
        );
        std::vector<float> overflowV(
            2u * overflowWorld.nv()
        );
        for (std::size_t environment = 0u;
             environment < 2u;
             ++environment) {
            std::copy(
                franka.defaultQ.begin(),
                franka.defaultQ.end(),
                overflowQ.begin() +
                    static_cast<std::ptrdiff_t>(
                        environment * overflowWorld.nq()
                    )
            );
            std::copy(
                franka.defaultV.begin(),
                franka.defaultV.end(),
                overflowV.begin() +
                    static_cast<std::ptrdiff_t>(
                        environment * overflowWorld.nv()
                    )
            );
        }
        MRBodyStateGPU clearCube = cube;
        clearCube.position.z += 2.0f;
        const std::vector<MRBodyStateGPU> overflowScene{
            cube,
            clearCube,
        };
        const std::vector<float> overflowEfforts(
            2u * overflowWorld.nv(),
            0.0f
        );
        const metalrobo::MetalWorldBatch overflowBatch{
            .environmentCount = 2u,
            .controlStepCount = 1u,
            .initialQ = overflowQ,
            .initialV = overflowV,
            .efforts = overflowEfforts,
            .initialSceneBodies = overflowScene,
        };
        metalrobo::MetalWorldResult overflowResult;
        const auto overflowRun = context.run(
            overflowWorld,
            overflowBatch,
            config,
            overflowResult
        );
        require(
            !overflowRun.succeeded() &&
                overflowRun.published &&
                overflowRun.failedStepCount == 1u &&
                overflowRun.successfulStepCount == 1u &&
                overflowResult.contactStatuses.size() == 2u &&
                overflowResult.contactStatuses[0].code ==
                    MR_STEP_CONTACT_CAPACITY_OVERFLOW &&
                overflowResult.contactStatuses[0]
                        .requiredRawContacts > 1u &&
                overflowResult.contactStatuses[1].code ==
                    MR_STEP_SUCCESS &&
                overflowResult.environmentStatuses.size() == 2u &&
                overflowResult.environmentStatuses[0].required
                        .rawContacts > 1u &&
                overflowResult.environmentStatuses[0].highWater
                        .rawContacts == 1u &&
                overflowResult.environmentStatuses[0]
                        .firstFailingPair != MR_INVALID_INDEX &&
                overflowResult.environmentStatuses[1].code ==
                    MR_STEP_SUCCESS,
            "per-environment capacity overflow was not exact and isolated"
        );
        require(
            std::equal(
                franka.defaultQ.begin(),
                franka.defaultQ.end(),
                overflowResult.finalQ.begin()
            ) &&
                std::equal(
                    franka.defaultV.begin(),
                    franka.defaultV.end(),
                    overflowResult.finalV.begin()
                ) &&
                overflowResult.finalSceneBodies[0].position.x ==
                    cube.position.x &&
                overflowResult.finalSceneBodies[0].position.y ==
                    cube.position.y &&
                overflowResult.finalSceneBodies[0].position.z ==
                    cube.position.z &&
                overflowResult.finalSceneBodies[0]
                        .linearVelocityAndInverseMass.x ==
                    cube.linearVelocityAndInverseMass.x &&
                overflowResult.finalSceneBodies[0]
                        .linearVelocityAndInverseMass.y ==
                    cube.linearVelocityAndInverseMass.y &&
                overflowResult.finalSceneBodies[0]
                        .linearVelocityAndInverseMass.z ==
                    cube.linearVelocityAndInverseMass.z &&
                overflowResult.finalSceneBodies[1].position.z <
                    clearCube.position.z,
            "capacity failure did not roll back only the affected environment"
        );

        metalrobo::CompiledWorld pairOverflowWorld;
        const auto pairOverflowCompiled =
            metalrobo::compileMetalWorld(
                franka,
                0u,
                pairOverflowWorld,
                metalrobo::MetalWorldCapacityProfile{
                    .candidatePairs = 1u,
                    .rawContacts = 64u,
                    .manifolds = 32u,
                    .constraintBlocks = 32u,
                    .constraintRows = 96u,
                    .islands = 16u,
                }
            );
        require(
            pairOverflowCompiled.succeeded(),
            "runtime pair-overflow world compilation failed"
        );
        metalrobo::MetalWorldResult pairOverflowResult;
        const auto pairOverflowRun = context.run(
            pairOverflowWorld,
            overflowBatch,
            config,
            pairOverflowResult
        );
        require(
            !pairOverflowRun.succeeded() &&
                pairOverflowRun.published &&
                pairOverflowRun.failedStepCount == 1u &&
                pairOverflowRun.successfulStepCount == 1u &&
                pairOverflowResult.contactStatuses[0].code ==
                    MR_STEP_PAIR_CAPACITY_OVERFLOW &&
                pairOverflowResult.contactStatuses[0]
                        .requiredPairs > 1u &&
                pairOverflowResult.environmentStatuses[0]
                        .highWater.candidatePairs == 1u &&
                pairOverflowResult.environmentStatuses[0]
                        .firstFailingPair != MR_INVALID_INDEX &&
                pairOverflowResult.environmentStatuses[1].code ==
                    MR_STEP_SUCCESS,
            "candidate-pair overflow was not exact and isolated"
        );

        constexpr std::uint32_t collidersPerLargeBody = 257u;
        metalrobo::EngineModel largePairModel =
            metalrobo::makeFreeSphereEngineModel();
        const MRShapeGPU sphereTemplate =
            largePairModel.shapes.back();
        largePairModel.shapes.clear();
        largePairModel.shapes.reserve(
            2u * collidersPerLargeBody
        );
        for (std::uint32_t body = 0u; body < 2u; ++body) {
            for (std::uint32_t local = 0u;
                 local < collidersPerLargeBody;
                 ++local) {
                MRShapeGPU shape = sphereTemplate;
                shape.bodyIndex = body;
                shape.slotGeneration = local + 1u;
                shape.localPosition.x =
                    local + 1u == collidersPerLargeBody
                    ? 0.0f
                    : body == 0u ? -100.0f : 100.0f;
                shape.dimensions.x = 0.01f;
                shape.contactRestAndBoundingRadius =
                    f4(0.001f, 0.0f, 0.01f);
                largePairModel.shapes.push_back(shape);
            }
        }
        largePairModel.world.shapeCount =
            static_cast<std::uint32_t>(
                largePairModel.shapes.size()
            );
        metalrobo::CompiledWorld largePairWorld;
        const auto largePairCompiled =
            metalrobo::compileMetalWorld(
                largePairModel,
                0u,
                largePairWorld,
                metalrobo::MetalWorldCapacityProfile{
                    .candidatePairs = 1u,
                    .rawContacts = 1u,
                    .manifolds = 1u,
                    .constraintBlocks = 1u,
                    .constraintRows = 3u,
                    .islands = 2u,
                }
            );
        require(
            largePairCompiled.succeeded() &&
                largePairWorld.eligiblePairCount() >
                    65536u,
            "large compiled-pair stream did not exceed the former ceiling"
        );
        const std::vector<float> largePairEfforts(
            largePairWorld.nv(),
            0.0f
        );
        std::vector<float> largePairQ =
            largePairModel.defaultQ;
        largePairQ[1] = 0.0f;
        const std::vector<MRBodyStateGPU> largePairScene{
            groundState(),
        };
        const metalrobo::MetalWorldBatch largePairBatch{
            .environmentCount = 1u,
            .controlStepCount = 1u,
            .initialQ = largePairQ,
            .initialV = largePairModel.defaultV,
            .efforts = largePairEfforts,
            .initialSceneBodies = largePairScene,
        };
        metalrobo::MetalWorldStepConfig largePairConfig = config;
        largePairConfig.physicsSubsteps = 1u;
        largePairConfig.velocityIterations = 1u;
        largePairConfig.finalVelocityIterations = 0u;
        largePairConfig.captureContactEvidence = false;
        metalrobo::MetalWorldContext largePairContext;
        metalrobo::MetalWorldResult largePairResult;
        const auto largePairRun = largePairContext.run(
            largePairWorld,
            largePairBatch,
            largePairConfig,
            largePairResult
        );
        require(
            largePairRun.succeeded() &&
                largePairResult.contactStatuses.size() == 1u &&
                largePairResult.contactStatuses[0]
                        .requiredPairs == 1u &&
                largePairResult.contactStatuses[0]
                        .activeContacts == 1u,
            "large device pair stream missed its final eligible pair"
        );

        constexpr std::size_t throughputEnvironments = 1024u;
        constexpr std::size_t throughputSteps = 4u;
        metalrobo::CompiledWorld throughputWorld;
        const auto throughputCompiled =
            metalrobo::compileMetalWorld(
                franka,
                0u,
                throughputWorld,
                metalrobo::MetalWorldCapacityProfile{
                    .candidatePairs = 64u,
                    .rawContacts = 64u,
                    .manifolds = 32u,
                    .constraintBlocks = 32u,
                    .constraintRows = 96u,
                    .islands = 16u,
                }
            );
        require(
            throughputCompiled.succeeded() &&
                throughputWorld.capacities().constraintBlocks ==
                    32u,
            "Franka throughput capacity-class compilation failed"
        );
        std::vector<float> throughputQ(
            throughputEnvironments * throughputWorld.nq()
        );
        std::vector<float> throughputV(
            throughputEnvironments * throughputWorld.nv()
        );
        std::vector<MRBodyStateGPU> throughputScene(
            throughputEnvironments,
            cube
        );
        for (std::size_t environment = 0u;
             environment < throughputEnvironments;
             ++environment) {
            std::copy(
                franka.defaultQ.begin(),
                franka.defaultQ.end(),
                throughputQ.begin() +
                    static_cast<std::ptrdiff_t>(
                        environment * throughputWorld.nq()
                    )
            );
            std::copy(
                franka.defaultV.begin(),
                franka.defaultV.end(),
                throughputV.begin() +
                    static_cast<std::ptrdiff_t>(
                        environment * throughputWorld.nv()
                    )
            );
        }
        const std::vector<float> throughputEfforts(
            throughputSteps * throughputEnvironments *
                throughputWorld.nv(),
            0.0f
        );
        const metalrobo::MetalWorldBatch throughputBatch{
            .environmentCount = throughputEnvironments,
            .controlStepCount = throughputSteps,
            .initialQ = throughputQ,
            .initialV = throughputV,
            .efforts = throughputEfforts,
            .initialSceneBodies = throughputScene,
        };
        metalrobo::MetalWorldStepConfig throughputConfig = config;
        throughputConfig.captureContactEvidence = false;
        throughputConfig.velocityIterations = 1u;
        throughputConfig.finalVelocityIterations = 1u;
        metalrobo::MetalWorldResult throughputWarmup;
        metalrobo::MetalWorldContext throughputContext;
        const auto warmup = throughputContext.run(
            throughputWorld,
            throughputBatch,
            throughputConfig,
            throughputWarmup
        );
        require(
            warmup.succeeded(),
            "Franka-plus-object throughput warmup failed"
        );
        constexpr std::size_t throughputSampleCount = 5u;
        std::vector<double> gpuMilliseconds;
        std::vector<double> wallMilliseconds;
        gpuMilliseconds.reserve(throughputSampleCount);
        wallMilliseconds.reserve(throughputSampleCount);
        metalrobo::MetalWorldResult throughputResult;
        metalrobo::MetalWorldDiagnostics throughputRun;
        for (std::size_t sample = 0u;
             sample < throughputSampleCount;
             ++sample) {
            const auto wallStart =
                std::chrono::steady_clock::now();
            throughputRun = throughputContext.run(
                throughputWorld,
                throughputBatch,
                throughputConfig,
                throughputResult
            );
            const auto wallEnd =
                std::chrono::steady_clock::now();
            require(
                throughputRun.succeeded(),
                "Franka-plus-object throughput run failed"
            );
            gpuMilliseconds.push_back(
                throughputRun.gpuElapsedMilliseconds
            );
            wallMilliseconds.push_back(
                std::chrono::duration<double, std::milli>(
                    wallEnd - wallStart
                ).count()
            );
        }
        const double environmentSteps =
            static_cast<double>(
                throughputEnvironments * throughputSteps
            );
        double totalGpuMilliseconds = 0.0;
        double totalWallMilliseconds = 0.0;
        for (const double milliseconds : gpuMilliseconds) {
            totalGpuMilliseconds += milliseconds;
        }
        for (const double milliseconds : wallMilliseconds) {
            totalWallMilliseconds += milliseconds;
        }
        const double gpuStepsPerSecond =
            1000.0 * environmentSteps *
            throughputSampleCount / totalGpuMilliseconds;
        const double wallStepsPerSecond =
            1000.0 * environmentSteps *
            throughputSampleCount / totalWallMilliseconds;
        require(
            throughputResult.environmentStatuses.size() ==
                throughputEnvironments,
            "throughput status aggregation is incomplete"
        );
        metalrobo::MetalWorldStageCounts highWater{};
        for (const auto& status :
             throughputResult.environmentStatuses) {
            highWater.candidatePairs = std::max(
                highWater.candidatePairs,
                status.highWater.candidatePairs
            );
            highWater.rawContacts = std::max(
                highWater.rawContacts,
                status.highWater.rawContacts
            );
            highWater.manifolds = std::max(
                highWater.manifolds,
                status.highWater.manifolds
            );
            highWater.constraintBlocks = std::max(
                highWater.constraintBlocks,
                status.highWater.constraintBlocks
            );
            highWater.constraintRows = std::max(
                highWater.constraintRows,
                status.highWater.constraintRows
            );
            highWater.islands = std::max(
                highWater.islands,
                status.highWater.islands
            );
            highWater.spillRows = std::max(
                highWater.spillRows,
                status.highWater.spillRows
            );
        }
        std::uint32_t throughputActiveContacts = 0u;
        for (const auto& status :
             throughputResult.contactStatuses) {
            throughputActiveContacts = std::max(
                throughputActiveContacts,
                status.activeContacts
            );
        }
        const double gpuBatchStepP50 =
            percentile(gpuMilliseconds, 0.50) /
            static_cast<double>(throughputSteps);
        const double gpuBatchStepP95 =
            percentile(gpuMilliseconds, 0.95) /
            static_cast<double>(throughputSteps);
        const double wallBatchStepP50 =
            percentile(wallMilliseconds, 0.50) /
            static_cast<double>(throughputSteps);
        const double wallBatchStepP95 =
            percentile(wallMilliseconds, 0.95) /
            static_cast<double>(throughputSteps);
        const bool throughputGatePassed =
            gpuStepsPerSecond >= 40000.0 &&
            wallStepsPerSecond >= 40000.0;

        std::cout
            << "metal_world_contact=ok"
            << " environments=" << environmentCount
            << " steps=" << controlStepCount
            << " gpu_ms="
            << firstDiagnostics.gpuElapsedMilliseconds
            << " retained_manifolds="
            << first.contactStatuses.back().retainedPoints
            << " franka_cube_contacts="
            << frankaStatus.activeContacts
            << " isolated_overflow_required_raw="
            << overflowResult.contactStatuses[0]
                   .requiredRawContacts
            << " isolated_overflow_required_pairs="
            << pairOverflowResult.contactStatuses[0]
                   .requiredPairs
            << " large_pair_stream="
            << largePairWorld.eligiblePairCount()
            << " large_pair_tail_contacts="
            << largePairResult.contactStatuses[0]
                   .activeContacts
            << " throughput_envs="
            << throughputEnvironments
            << " throughput_gpu_steps_per_s="
            << gpuStepsPerSecond
            << " throughput_wall_steps_per_s="
            << wallStepsPerSecond
            << " gpu_batch_step_p50_ms="
            << gpuBatchStepP50
            << " gpu_batch_step_p95_ms="
            << gpuBatchStepP95
            << " wall_batch_step_p50_ms="
            << wallBatchStepP50
            << " wall_batch_step_p95_ms="
            << wallBatchStepP95
            << " throughput_active_contacts="
            << throughputActiveContacts
            << " high_water_pairs="
            << highWater.candidatePairs
            << " high_water_raw=" << highWater.rawContacts
            << " high_water_manifolds="
            << highWater.manifolds
            << " high_water_constraints="
            << highWater.constraintBlocks
            << " high_water_rows="
            << highWater.constraintRows
            << " high_water_islands="
            << highWater.islands
            << " high_water_spill="
            << highWater.spillRows
            << " retained_bytes="
            << throughputContext.stats().retainedBufferBytes
            << " thermal=" << throughputRun.thermalState
            << " release_gate_40k="
            << (throughputGatePassed ? "pass" : "open")
            << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "metal_world_contact=failed reason=\""
            << exception.what() << "\"\n";
        return 1;
    }
}
