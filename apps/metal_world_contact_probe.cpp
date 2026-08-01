#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/EngineModel.hpp"
#include "metalrobo/Franka.hpp"
#include "metalrobo/GeometryCooker.hpp"
#include "metalrobo/HeterogeneousWorld.hpp"
#include "metalrobo/MetalWorld.hpp"

#include <algorithm>
#include <array>
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

void verifyCoupledLimitContactNumiSolver() {
    metalrobo::EngineModel model =
        metalrobo::makeFrankaPandaEngineModel();
    const auto limited = std::find_if(
        model.dofs.begin(),
        model.dofs.end(),
        [](const MRDofPropertiesGPU& dof) {
            return
                (dof.flags & MR_DOF_FLAG_POSITION_LIMIT) != 0u &&
                dof.qIndex != MR_INVALID_INDEX &&
                dof.vIndex != MR_INVALID_INDEX;
        }
    );
    require(
        limited != model.dofs.end(),
        "Franka has no scalar position-limit witness"
    );
    const std::uint32_t limitedV = limited->vIndex;
    const std::uint32_t limitedQ = limited->qIndex;
    const float upper = limited->limits.y;
    model.defaultQ[limitedQ] = upper - 5.0e-4f;
    model.defaultV[limitedV] = 0.5f;

    const MRShapeGPU witnessShape = model.shapes.back();
    const std::vector<double> q(
        model.defaultQ.begin(),
        model.defaultQ.end()
    );
    const std::vector<double> v(
        model.defaultV.begin(),
        model.defaultV.end()
    );
    const metalrobo::ArticulatedPointQuery query{
        .bodyIndex = witnessShape.bodyIndex,
        .localPoint = {
            witnessShape.localPosition.x,
            witnessShape.localPosition.y,
            witnessShape.localPosition.z,
        },
    };
    metalrobo::ArticulatedPointKinematics witness{};
    std::vector<double> jacobian(
        3u * model.articulations[0].nv
    );
    const auto kinematics =
        metalrobo::computeArticulatedPointJacobians(
            model,
            0u,
            q,
            v,
            std::span{&query, 1u},
            std::span{&witness, 1u},
            jacobian
        );
    require(
        kinematics.succeeded(),
        "coupled limit/contact witness kinematics failed"
    );

    const std::uint32_t cubeBody =
        static_cast<std::uint32_t>(model.bodies.size());
    model.bodies.push_back(cubeProperties());
    model.shapes.push_back(cubeShape(cubeBody));
    model.world.bodyCount =
        static_cast<std::uint32_t>(model.bodies.size());
    model.world.shapeCount =
        static_cast<std::uint32_t>(model.shapes.size());

    metalrobo::CompiledWorld world;
    const auto compiled = metalrobo::compileMetalWorld(
        model,
        0u,
        world
    );
    require(
        compiled.succeeded(),
        compiled.message.c_str()
    );
    const auto& program = world.model().constraintProgram;
    const auto upperCandidate = std::find_if(
        program.blocks.begin(),
        program.blocks.end(),
        [limitedV](const MRConstraintIRBlockGPU& block) {
            return block.type == MR_CONSTRAINT_LIMIT &&
                block.key.words[0] == 0x4c494d54u &&
                block.key.words[2] == limitedV &&
                block.key.words[3] == 1u;
        }
    );
    require(
        upperCandidate != program.blocks.end() &&
            std::count_if(
                program.blocks.begin(),
                program.blocks.end(),
                [limitedV](const MRConstraintIRBlockGPU& block) {
                    return block.type == MR_CONSTRAINT_LIMIT &&
                        block.key.words[0] == 0x4c494d54u &&
                        block.key.words[2] == limitedV;
                }
            ) == 2,
        "compiler did not emit stable paired limit candidates"
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

    const std::vector<float> efforts(world.nv(), 0.0f);
    const metalrobo::MetalWorldBatch batch{
        .environmentCount = 1u,
        .controlStepCount = 1u,
        .initialQ = model.defaultQ,
        .initialV = model.defaultV,
        .efforts = efforts,
        .initialSceneBodies = std::span{&cube, 1u},
    };
    const metalrobo::MetalWorldStepConfig config{
        .timestepSeconds = 1.0f / 240.0f,
        .physicsSubsteps = 1u,
        .executionMode =
            metalrobo::MetalWorldExecutionMode::numiSolver,
        .numiSolver = {
            .temporalSubsteps = 1u,
        },
        .deterministic = true,
        .warmStart = true,
        .captureContactEvidence = true,
    };
    metalrobo::MetalWorldContext context;
    metalrobo::MetalWorldResult result;
    const auto run = context.run(world, batch, config, result);
    if (!run.succeeded() && !result.contactStatuses.empty()) {
        const auto& status = result.contactStatuses.front();
        std::cerr
            << "coupled_limit_status=" << status.code
            << " constraint=" << status.firstFailingConstraint
            << " rows=" << status.requiredRows
            << " blocks=" << status.requiredConstraints
            << " residuals=(" << status.residuals.x << ','
            << status.residuals.y << ',' << status.residuals.z
            << ',' << status.residuals.w << ")\n";
    }
    require(run.succeeded(), run.message.c_str());
    require(
        result.finalQ[limitedQ] <=
            upper + config.jointLimitPositionSlop + 1.0e-5f,
        "NumiSolver crossed the coupled scalar upper limit"
    );

    const std::size_t required =
        result.contactStatuses.front().requiredConstraints;
    bool activeUpperLimit = false;
    bool activeContact = false;
    for (std::size_t index = 0u;
         index < required;
         ++index) {
        const auto& block = result.contactEvidence.blocks[index];
        const auto& contact = result.contactEvidence.contacts[index];
        if (block.type == MR_CONSTRAINT_LIMIT &&
            block.key.words[0] == 0x4c494d54u &&
            block.key.words[2] == limitedV &&
            block.key.words[3] == 1u &&
            (block.flags &
             MR_CONSTRAINT_IR_BLOCK_DISABLED) == 0u &&
            contact.impulses.x > 0.0f) {
            activeUpperLimit = true;
        }
        if (block.type == MR_CONSTRAINT_CONTACT &&
            contact.impulses.x > 0.0f) {
            activeContact = true;
        }
    }
    if (!activeUpperLimit || !activeContact) {
        std::cerr
            << "coupled_limit_evidence"
            << " final_q=" << result.finalQ[limitedQ]
            << " final_v=" << result.finalV[limitedV]
            << " required=" << required << '\n';
        for (std::size_t index = 0u;
             index < required;
             ++index) {
            const auto& block =
                result.contactEvidence.blocks[index];
            const auto& contact =
                result.contactEvidence.contacts[index];
            std::cerr
                << "  block=" << index
                << " type=" << block.type
                << " flags=" << block.flags
                << " key=(" << block.key.words[0]
                << ',' << block.key.words[1]
                << ',' << block.key.words[2]
                << ',' << block.key.words[3] << ')'
                << " impulse=(" << contact.impulses.x
                << ',' << contact.impulses.y
                << ',' << contact.impulses.z << ")\n";
        }
    }
    require(
        activeUpperLimit && activeContact,
        "limit and contact were not both active in NumiSolver"
    );
}

double verifyRetainedRodCoupling() {
    metalrobo::DualPsmNeedleThreadWorldConfig sourceConfig;
    sourceConfig.threadNodeCount = 9u;
    sourceConfig.threadLengthM = 0.12;
    metalrobo::HeterogeneousWorld source;
    const auto authored =
        metalrobo::makeDualDvrkPsmNeedleThreadHeterogeneousWorld(
            source,
            sourceConfig
        );
    require(authored.succeeded(), "rod world authoring failed");

    metalrobo::CompiledWorld world;
    const auto compiled = metalrobo::compileMetalWorld(source, world);
    require(
        compiled.succeeded() && world.rodCount() == 1u,
        "rod world compilation failed"
    );
    const std::vector<float> efforts(source.model.world.nv, 0.0f);
    const metalrobo::MetalWorldBatch freeBatch{
        .environmentCount = 1u,
        .controlStepCount = 1u,
        .initialQ = source.model.defaultQ,
        .initialV = source.model.defaultV,
        .efforts = efforts,
    };
    const metalrobo::MetalWorldBatch constrainedBatch{
        .environmentCount = 1u,
        .controlStepCount = 1u,
        .initialQ = source.model.defaultQ,
        .initialV = source.model.defaultV,
        .efforts = efforts,
        .initialSceneBodies = source.defaultSceneBodies,
    };
    metalrobo::MetalWorldStepConfig freeConfig;
    freeConfig.executionMode =
        metalrobo::MetalWorldExecutionMode::freeMotionABA;
    freeConfig.timestepSeconds = 1.0f / 1000.0f;
    freeConfig.physicsSubsteps = 1u;
    freeConfig.ccdMode = metalrobo::MetalWorldCCDMode::disabled;
    metalrobo::MetalWorldStepConfig constrainedConfig = freeConfig;
    constrainedConfig.executionMode =
        metalrobo::MetalWorldExecutionMode::numiSolver;
    constrainedConfig.numiSolver.temporalSubsteps = 1u;

    metalrobo::MetalWorldContext context;
    metalrobo::MetalWorldResult freeResult;
    metalrobo::MetalWorldResult fixedResult;
    const auto freeRun = context.run(
        world,
        freeBatch,
        freeConfig,
        freeResult
    );
    const auto fixedRun = context.run(
        world,
        constrainedBatch,
        constrainedConfig,
        fixedResult
    );
    require(
        freeRun.succeeded() && fixedRun.succeeded() &&
            fixedResult.statuses.size() == 1u &&
            fixedResult.statuses[0].code == MR_STEP_SUCCESS &&
            fixedResult.finalRodNodes.size() ==
                freeResult.finalRodNodes.size(),
        "fixed-budget retained rod solve failed"
    );

    const std::uint32_t attachedNode =
        source.rods[0].attachments[0].nodeIndex;
    double maximumRemoteResponse = 0.0;
    for (std::size_t node = 0u;
         node < fixedResult.finalRodNodes.size();
         ++node) {
        if (node == attachedNode) {
            continue;
        }
        const std::array<float, 3u> fixedVelocity{
            fixedResult.finalRodNodes[node].velocity.x,
            fixedResult.finalRodNodes[node].velocity.y,
            fixedResult.finalRodNodes[node].velocity.z,
        };
        const std::array<float, 3u> freeVelocity{
            freeResult.finalRodNodes[node].velocity.x,
            freeResult.finalRodNodes[node].velocity.y,
            freeResult.finalRodNodes[node].velocity.z,
        };
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            maximumRemoteResponse = std::max(
                maximumRemoteResponse,
                static_cast<double>(std::abs(
                    fixedVelocity[axis] - freeVelocity[axis]
                ))
            );
        }
    }
    require(
        maximumRemoteResponse > 1.0e-9,
        "rod attachment remained a local endpoint response"
    );

    auto residualConfig = constrainedConfig;
    residualConfig.numiSolver.iterationPolicy =
        metalrobo::NumiSolverIterationPolicy::residualConverged;
    metalrobo::MetalWorldResult residualResult;
    const auto residualRun = context.run(
        world,
        constrainedBatch,
        residualConfig,
        residualResult
    );
    require(
        residualRun.succeeded() &&
            residualResult.statuses.size() == 1u &&
            residualResult.statuses[0].code == MR_STEP_SUCCESS &&
            residualResult.numiConvergenceStatuses.size() == 1u,
        "residual-converged retained rod solve failed"
    );
    return maximumRemoteResponse;
}

} // namespace

int main(const int argc, char** argv) {
    try {
        if (argc == 2 &&
            std::strcmp(argv[1], "--coupled-limit-only") == 0) {
            verifyCoupledLimitContactNumiSolver();
            std::cout << "coupled_limit_contact_numisolver=pass\n";
            return 0;
        }
        constexpr std::size_t environmentCount = 4u;
        constexpr std::size_t controlStepCount = 12u;
        const double retainedRodResponse =
            verifyRetainedRodCoupling();
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
            .physicsSubsteps = 1u,
            .executionMode =
                metalrobo::MetalWorldExecutionMode::numiSolver,
            .numiSolver = {
                .temporalSubsteps = 4u,
            },
            .deterministic = true,
            .warmStart = true,
            .captureContactEvidence = true,
        };

        metalrobo::MetalWorldContext context;
        metalrobo::MetalWorldResult first;
        const auto firstDiagnostics =
            context.run(world, batch, config, first);
        if (!firstDiagnostics.succeeded()) {
            std::cerr
                << "initial_contact_failure"
                << " environment="
                << firstDiagnostics.firstFailingEnvironment
                << " control_step="
                << firstDiagnostics.firstFailingControlStep
                << " code="
                << firstDiagnostics.firstGPUStatusCode;
            const std::size_t statusIndex =
                static_cast<std::size_t>(
                    firstDiagnostics.firstFailingControlStep
                ) *
                    environmentCount +
                firstDiagnostics.firstFailingEnvironment;
            if (statusIndex < first.contactStatuses.size()) {
                const auto& status =
                    first.contactStatuses[statusIndex];
                std::cerr
                    << " contact_code=" << status.code
                    << " substep=" << status.physicsSubstep
                    << " constraint="
                    << status.firstFailingConstraint
                    << " iterations=" << status.solverIterations
                    << " residuals=("
                    << status.residuals.x << ','
                    << status.residuals.y << ','
                    << status.residuals.z << ','
                    << status.residuals.w << ')';
            }
            if (statusIndex < first.statuses.size()) {
                const auto& status = first.statuses[statusIndex];
                std::cerr
                    << " aba_code=" << status.abaCode
                    << " failing_substep="
                    << status.failingSubstep
                    << " failing_index=" << status.failingIndex
                    << " successful_substeps="
                    << status.successfulSubsteps;
            }
            std::cerr << '\n';
        }
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
                first.contactStatuses.back().solverIterations == 1u,
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
        auto residualConfig = config;
        residualConfig.numiSolver.iterationPolicy =
            metalrobo::NumiSolverIterationPolicy::
                residualConverged;
        residualConfig.ccdMode =
            metalrobo::MetalWorldCCDMode::disabled;
        metalrobo::MetalWorldResult residualResult;
        const auto residualDiagnostics = context.run(
            world,
            batch,
            residualConfig,
            residualResult
        );
        require(
            residualDiagnostics.succeeded(),
            residualDiagnostics.message.c_str()
        );
        require(
            residualResult.numiConvergenceStatuses.size() ==
                    environmentCount &&
                std::all_of(
                    residualResult.numiConvergenceStatuses.begin(),
                    residualResult.numiConvergenceStatuses.end(),
                    [](const MRUnifiedQualityStatusGPU& status) {
                        return
                            status.code ==
                                MR_UNIFIED_QUALITY_SUCCESS &&
                            std::isfinite(
                                status.certificates0.x
                            ) &&
                            std::isfinite(
                                status.certificates1.x
                            );
                    }
                ) &&
                std::all_of(
                    residualResult.environmentStatuses.begin(),
                    residualResult.environmentStatuses.end(),
                    [environmentCount](
                        const metalrobo::MetalWorldStatus& status
                    ) {
                        return
                            status.code == MR_STEP_SUCCESS &&
                            status
                                .maximumNumiNewtonIterations >
                                0u &&
                            status.maximumWorkerPackets ==
                                environmentCount &&
                            std::isfinite(
                                status
                                    .maximumNumiCertificates[0]
                            );
                    }
                ),
            "residual-converged NumiSolver was not promoted through MetalWorld"
        );
        metalrobo::MetalWorldContext asynchronousContext(
            metalrobo::MetalWorldConfig{
                .maximumInFlightSubmissions = 3u,
            }
        );
        std::array<metalrobo::MetalWorldSubmission, 3u>
            asynchronousSubmissions;
        for (auto& ticket : asynchronousSubmissions) {
            const auto submitted = asynchronousContext.submit(
                world,
                batch,
                config,
                ticket
            );
            require(
                submitted.succeeded() && ticket.valid(),
                "three-slot asynchronous submission failed"
            );
        }
        metalrobo::MetalWorldSubmission saturatedTicket;
        const auto saturatedSubmission =
            asynchronousContext.submit(
                world,
                batch,
                config,
                saturatedTicket
            );
        require(
            saturatedSubmission.status ==
                    metalrobo::MetalWorldHostStatus::contextBusy &&
                !saturatedTicket.valid(),
            "asynchronous ring did not report exact saturation"
        );
        for (auto& ticket : asynchronousSubmissions) {
            metalrobo::MetalWorldResult asynchronousResult;
            const auto completed = ticket.wait(
                asynchronousResult
            );
            require(
                completed.succeeded(),
                "asynchronous arena slot did not publish"
            );
        }
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

        metalrobo::EngineModel meshModel =
            metalrobo::makeFreeSphereEngineModel();
        const std::array<mr_float4, 4u> floorVertices{{
            f4(-2.0f, 0.0f, -2.0f, 1.0f),
            f4(2.0f, 0.0f, -2.0f, 1.0f),
            f4(2.0f, 0.0f, 2.0f, 1.0f),
            f4(-2.0f, 0.0f, 2.0f, 1.0f),
        }};
        const std::array<std::uint32_t, 6u> floorIndices{{
            0u, 2u, 1u,
            0u, 3u, 2u,
        }};
        const auto cookedFloor =
            metalrobo::cookTriangleMeshGeometry(
                meshModel,
                floorVertices,
                floorIndices
            );
        require(
            cookedFloor.succeeded(),
            cookedFloor.message.c_str()
        );
        MRShapeGPU& floorShape = meshModel.shapes.front();
        floorShape.shapeType = MR_SHAPE_TRIANGLE_MESH;
        floorShape.geometryOffset = cookedFloor.geometryIndex;
        floorShape.geometryCount = 1u;
        floorShape.dimensions = f4(1.0f, 1.0f, 1.0f);
        floorShape.contactRestAndBoundingRadius =
            f4(0.01f, 0.0f, 3.0f);
        metalrobo::CompiledWorld meshWorld;
        const auto meshCompiled = metalrobo::compileMetalWorld(
            meshModel,
            0u,
            meshWorld
        );
        require(
            meshCompiled.succeeded() &&
                meshWorld.geometryHeaders().size() == 1u &&
                meshWorld.geometryHeaders()[0].bvhCount > 0u &&
                meshWorld.eligiblePairs()[0].pairClass ==
                    MR_COLLISION_PAIR_MESH,
            "cooked mesh world compilation failed"
        );
        metalrobo::MetalWorldResult meshResult;
        const auto meshRun = context.run(
            meshWorld,
            batch,
            config,
            meshResult
        );
        if (!meshRun.succeeded() ||
            meshResult.contactStatuses.empty() ||
            meshResult.contactStatuses.back().activeContacts == 0u) {
            std::cerr
                << "mesh_status="
                << meshRun.firstGPUStatusCode
                << " environment="
                << meshRun.firstFailingEnvironment
                << " step="
                << meshRun.firstFailingControlStep;
            if (!meshResult.contactStatuses.empty()) {
                const auto& failed =
                    meshResult.contactStatuses.back();
                std::cerr
                    << " contact_code=" << failed.code
                    << " pair=" << failed.firstFailingPair
                    << " constraint="
                    << failed.firstFailingConstraint
                    << " active=" << failed.activeContacts
                    << " pairs=" << failed.requiredPairs
                    << " raw=" << failed.requiredRawContacts
                    << " manifolds="
                    << failed.requiredManifolds;
            }
            std::cerr << '\n';
        }
        require(
            meshRun.succeeded() &&
                meshResult.contactStatuses.back()
                        .activeContacts > 0u &&
                meshResult.contactStatuses.back()
                        .requiredMeshCandidates > 0u,
            meshRun.message.c_str()
        );

        // Exact convex-mesh CCD uses the same stackless BVH candidates but
        // advances against the actual sphere/triangle witnesses rather than
        // a mesh-wide or body-wide bounding sphere.
        meshModel.shapes.back().flags |=
            MR_SHAPE_FLAG_ENABLE_CCD;
        metalrobo::CompiledWorld meshCCDWorld;
        const auto meshCCDCompiled =
            metalrobo::compileMetalWorld(
                meshModel,
                0u,
                meshCCDWorld
            );
        require(
            meshCCDCompiled.succeeded(),
            "cooked mesh CCD world compilation failed"
        );
        std::vector<float> meshCCDQ = meshModel.defaultQ;
        meshCCDQ[1] = 0.5f;
        std::vector<float> meshCCDV = meshModel.defaultV;
        meshCCDV[1] = -60.0f;
        const std::vector<float> meshCCDEffort(
            meshCCDWorld.nv(),
            0.0f
        );
        const std::vector<MRBodyStateGPU> meshCCDScene{
            groundState(),
        };
        const metalrobo::MetalWorldBatch meshCCDBatch{
            .environmentCount = 1u,
            .controlStepCount = 1u,
            .initialQ = meshCCDQ,
            .initialV = meshCCDV,
            .efforts = meshCCDEffort,
            .initialSceneBodies = meshCCDScene,
        };
        metalrobo::MetalWorldStepConfig meshCCDConfig = config;
        meshCCDConfig.timestepSeconds = 1.0f / 60.0f;
        meshCCDConfig.physicsSubsteps = 1u;
        // Hybrid CCD already advances to impact within the authored step;
        // isolate that contract from NumiSolver temporal refinement here.
        meshCCDConfig.numiSolver.temporalSubsteps = 1u;
        meshCCDConfig.ccdMode =
            metalrobo::MetalWorldCCDMode::hybrid;
        meshCCDConfig.maxConservativeAdvancementIterations =
            32u;
        meshCCDConfig.captureContactEvidence = true;
        metalrobo::MetalWorldResult meshCCDResult;
        const auto meshCCDRun = context.run(
            meshCCDWorld,
            meshCCDBatch,
            meshCCDConfig,
            meshCCDResult
        );
        require(
            meshCCDRun.succeeded() &&
                meshCCDResult.contactStatuses.size() == 1u &&
                meshCCDResult.contactStatuses[0]
                        .requiredCCDCandidates == 1u &&
                meshCCDResult.contactStatuses[0]
                        .requiredCCDEvents == 1u &&
                meshCCDResult.contactStatuses[0]
                        .clusteredCCDImpacts == 1u &&
                meshCCDResult.contactStatuses[0]
                        .unresolvedCCDCount == 0u &&
                std::abs(
                    meshCCDResult.contactStatuses[0]
                        .eventTimes.x -
                    meshCCDConfig.timestepSeconds
                ) < 1.0e-7f &&
                meshCCDResult.contactStatuses[0]
                        .eventTimes.y == 0.0f &&
                meshCCDResult.contactStatuses[0]
                        .ccdAdvanceCount == 1u &&
                meshCCDResult.finalQ.size() > 1u &&
                meshCCDResult.finalQ[1u] > 0.049f &&
                std::isfinite(meshCCDResult.finalV[1u]),
            "hybrid convex-mesh CCD did not certify and constrain impact"
        );

        // One fast cylinder crossing a cooked convex closes the support-map,
        // robust GJK/MPR/EPA, swept broadphase, and exact-event CCD path in a
        // single device graph.
        metalrobo::EngineModel ccdModel =
            metalrobo::makeFreeSphereEngineModel();
        const std::array<mr_float4, 8u> convexVertices{{
            f4(-0.05f, -0.08f, -0.08f, 1.0f),
            f4(0.05f, -0.08f, -0.08f, 1.0f),
            f4(0.05f, 0.08f, -0.08f, 1.0f),
            f4(-0.05f, 0.08f, -0.08f, 1.0f),
            f4(-0.05f, -0.08f, 0.08f, 1.0f),
            f4(0.05f, -0.08f, 0.08f, 1.0f),
            f4(0.05f, 0.08f, 0.08f, 1.0f),
            f4(-0.05f, 0.08f, 0.08f, 1.0f),
        }};
        const std::array<std::uint32_t, 36u> convexIndices{{
            0u, 2u, 1u, 0u, 3u, 2u,
            4u, 5u, 6u, 4u, 6u, 7u,
            0u, 1u, 5u, 0u, 5u, 4u,
            1u, 2u, 6u, 1u, 6u, 5u,
            2u, 3u, 7u, 2u, 7u, 6u,
            3u, 0u, 4u, 3u, 4u, 7u,
        }};
        const auto cookedConvex =
            metalrobo::cookConvexGeometry(
                ccdModel,
                convexVertices,
                convexIndices
            );
        require(
            cookedConvex.succeeded(),
            cookedConvex.message.c_str()
        );
        MRShapeGPU& convexShape = ccdModel.shapes.front();
        convexShape.shapeType = MR_SHAPE_CONVEX;
        convexShape.geometryOffset =
            cookedConvex.geometryIndex;
        convexShape.geometryCount = 1u;
        convexShape.dimensions = f4(1.0f, 1.0f, 1.0f);
        convexShape.contactRestAndBoundingRadius =
            f4(0.002f, 0.0f, 0.13f);
        MRShapeGPU& ccdCylinderShape = ccdModel.shapes.back();
        ccdCylinderShape.shapeType = MR_SHAPE_CYLINDER;
        ccdCylinderShape.flags = MR_SHAPE_FLAG_ENABLE_CCD;
        ccdCylinderShape.dimensions =
            f4(0.025f, 0.06f, 0.0f);
        ccdCylinderShape.contactRestAndBoundingRadius =
            f4(0.002f, 0.0f, 0.066f);
        ccdModel.world.gravityAndTimestep =
            f4(0.0f, 0.0f, 0.0f, 1.0f / 60.0f);

        metalrobo::CompiledWorld ccdWorld;
        const auto ccdCompiled =
            metalrobo::compileMetalWorld(
                ccdModel,
                0u,
                ccdWorld
            );
        require(
            ccdCompiled.succeeded() &&
                ccdWorld.eligiblePairs().size() == 1u &&
                ccdWorld.eligiblePairs()[0].pairClass ==
                    MR_COLLISION_PAIR_CONVEX,
            "cylinder-convex CCD world compilation failed"
        );
        std::vector<float> ccdQ = ccdModel.defaultQ;
        ccdQ[0] = -0.25f;
        ccdQ[1] = 0.0f;
        std::vector<float> ccdV = ccdModel.defaultV;
        ccdV[0] = 30.0f;
        const std::vector<float> ccdEffort(
            ccdWorld.nv(),
            0.0f
        );
        const std::vector<MRBodyStateGPU> ccdScene{
            groundState(),
        };
        const metalrobo::MetalWorldBatch ccdBatch{
            .environmentCount = 1u,
            .controlStepCount = 1u,
            .initialQ = ccdQ,
            .initialV = ccdV,
            .efforts = ccdEffort,
            .initialSceneBodies = ccdScene,
        };
        metalrobo::MetalWorldStepConfig ccdConfig = config;
        ccdConfig.timestepSeconds = 1.0f / 60.0f;
        ccdConfig.physicsSubsteps = 1u;
        ccdConfig.numiSolver.temporalSubsteps = 1u;
        ccdConfig.ccdMode =
            metalrobo::MetalWorldCCDMode::hybrid;
        ccdConfig.maxConservativeAdvancementIterations = 32u;
        ccdConfig.captureContactEvidence = true;
        metalrobo::MetalWorldResult ccdResult;
        const auto ccdRun = context.run(
            ccdWorld,
            ccdBatch,
            ccdConfig,
            ccdResult
        );
        require(
            ccdRun.succeeded() &&
                ccdResult.contactStatuses.size() == 1u &&
                ccdResult.contactStatuses[0]
                        .requiredCCDCandidates == 1u &&
                ccdResult.contactStatuses[0]
                        .requiredCCDEvents == 1u &&
                ccdResult.contactStatuses[0]
                        .clusteredCCDImpacts == 1u &&
                ccdResult.contactStatuses[0]
                        .unresolvedCCDCount == 0u &&
                std::abs(
                    ccdResult.contactStatuses[0]
                        .eventTimes.x -
                    ccdConfig.timestepSeconds
                ) < 1.0e-7f &&
                ccdResult.contactStatuses[0]
                        .eventTimes.y == 0.0f &&
                ccdResult.contactStatuses[0]
                        .ccdAdvanceCount == 1u &&
                ccdResult.finalQ.size() > 0u &&
                ccdResult.finalQ[0u] < 0.1f &&
                std::isfinite(ccdResult.finalV[0u]),
            "hybrid cylinder-convex CCD did not certify and constrain impact"
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
        franka.bodyNames.push_back("contact_cube");
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
                    static_cast<std::ptrdiff_t>(
                        contactBase +
                        frankaResult.layout.contactDispatch.
                            authoredConstraintCount
                    ),
                frankaResult.contactEvidence.contacts.begin() +
                    static_cast<std::ptrdiff_t>(
                        contactBase +
                        frankaStatus.requiredConstraints
                    ),
                [cubeBody](const MRContactConstraintGPU& contact) {
                    return
                        (contact.flags &
                         MR_CONSTRAINT_FLAG_GENERALIZED) == 0u &&
                        (contact.bodyA == cubeBody ||
                         contact.bodyB == cubeBody);
                }
            ),
            "Franka-plus-cube graph did not compile a mixed contact"
        );

        metalrobo::EngineModel cylinderModel = franka;
        MRShapeGPU& cylinderShape = cylinderModel.shapes.back();
        cylinderShape.shapeType = MR_SHAPE_CYLINDER;
        cylinderShape.dimensions =
            f4(0.05f, 0.05f, 0.0f, 0.0f);
        cylinderShape.contactRestAndBoundingRadius =
            f4(0.002f, 0.0f, 0.07071068f, 0.0f);
        cylinderShape.flags |= MR_SHAPE_FLAG_ENABLE_CCD;
        metalrobo::CompiledWorld cylinderWorld;
        const auto cylinderCompiled =
            metalrobo::compileMetalWorld(
                cylinderModel,
                0u,
                cylinderWorld
            );
        require(
            cylinderCompiled.succeeded() &&
                std::any_of(
                    cylinderWorld.eligiblePairs().begin(),
                    cylinderWorld.eligiblePairs().end(),
                    [](const MRCompiledCollisionPairGPU& pair) {
                        return pair.pairClass ==
                            MR_COLLISION_PAIR_CONVEX;
                    }
                ),
            "Franka-plus-cylinder convex world compilation failed"
        );
        metalrobo::MetalWorldResult cylinderResult;
        metalrobo::MetalWorldStepConfig cylinderConfig = config;
        cylinderConfig.ccdMode =
            metalrobo::MetalWorldCCDMode::hybrid;
        const auto cylinderRun = context.run(
            cylinderWorld,
            frankaBatch,
            cylinderConfig,
            cylinderResult
        );
        require(
            cylinderRun.succeeded() &&
                cylinderResult.contactStatuses.back()
                        .activeContacts > 0u &&
                cylinderResult.contactStatuses.back()
                        .requiredCCDCandidates > 0u &&
                cylinderResult.contactStatuses.back()
                        .eventTimes.y == 0.0f &&
                cylinderResult.contactStatuses.back().code ==
                    MR_STEP_SUCCESS,
            "Franka cylinder hybrid CCD did not finish with evidence"
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
        largePairConfig.numiSolver.temporalSubsteps = 1u;
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
        const metalrobo::MetalWorldContextStats runtimeStats =
            context.stats();
        require(
            runtimeStats.pipelineCreationCount > 0u &&
                runtimeStats.pipelineCreationCount <
                    MR_RUNTIME_PIPELINE_COUNT,
            "contact execution plan did not prune or share its Metal pipelines"
        );

        std::cout
            << "metal_world_contact=ok"
            << " environments=" << environmentCount
            << " steps=" << controlStepCount
            << " gpu_ms="
            << firstDiagnostics.gpuElapsedMilliseconds
            << " retained_manifolds="
            << first.contactStatuses.back().retainedPoints
            << " retained_rod_response="
            << retainedRodResponse
            << " async_slots=3"
            << " plan_pipelines="
            << runtimeStats.pipelineCreationCount
            << "/" << MR_RUNTIME_PIPELINE_COUNT
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
            << " mesh_candidates="
            << meshResult.contactStatuses.back()
                   .requiredMeshCandidates
            << " mesh_ccd_events="
            << meshCCDResult.contactStatuses[0]
                   .requiredCCDEvents
            << " cylinder_convex_ccd_events="
            << ccdResult.contactStatuses[0]
                   .requiredCCDEvents
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
            << " solver=numisolver"
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
