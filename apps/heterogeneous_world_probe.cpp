#include "metalrobo/HeterogeneousWorld.hpp"
#include "metalrobo/MetalMultiArticulatedContact.hpp"
#include "metalrobo/MetalMultiArticulatedConstraints.hpp"
#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/MultiArticulatedContact.hpp"
#include "metalrobo/MultiArticulatedWorld.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

double norm(const std::array<double, 3>& value) {
    return std::sqrt(
        value[0] * value[0] +
        value[1] * value[1] +
        value[2] * value[2]
    );
}

} // namespace

int main() {
    try {
        metalrobo::DualPsmNeedleThreadWorldConfig config;
        config.threadNodeCount = 9u;
        config.threadLengthM = 0.12;
        metalrobo::HeterogeneousWorld world;
        const auto diagnostics =
            metalrobo::
                makeDualDvrkPsmNeedleThreadHeterogeneousWorld(
                    world,
                    config
                );
        require(
            diagnostics.succeeded(),
            "heterogeneous surgical cook failed: " +
                diagnostics.message
        );
        std::string reason;
        require(
            world.valid(&reason),
            "heterogeneous surgical world is invalid: " + reason
        );
        require(
            world.model.articulations.size() == 2u &&
                world.model.bodies.size() == 19u &&
                world.model.shapes.size() == 72u &&
                world.model.constraintProgram.blocks.size() == 14u &&
                world.sceneBodyIndices ==
                    std::vector<std::uint32_t>{18u} &&
                world.defaultSceneBodies.size() == 1u &&
                world.rods.size() == 1u &&
                world.rods[0].rigidBindings[0].bodyIndex == 0u,
            "heterogeneous owned streams changed"
        );

        // The persistent world must own all articulation and rod state in
        // one transaction. This free-motion slice intentionally exercises
        // lifetime/reset/publication before contact coupling below.
        metalrobo::CompiledWorld persistentWorld;
        const auto persistentCompile =
            metalrobo::compileMetalWorld(
                world,
                persistentWorld
            );
        require(
            persistentCompile.succeeded() &&
                persistentWorld.articulationCount() == 2u &&
                persistentWorld.rodCount() == 1u,
            "persistent heterogeneous MetalWorld did not compile: " +
                persistentCompile.message
        );
        const auto& compiledProgram =
            persistentWorld.model().constraintProgram;
        require(
            compiledProgram.blocks.size() == 17u &&
                persistentWorld.dynamicNodes().size() == 4u &&
                persistentWorld.minimumCapacities()
                        .constraintBlocks >=
                    17u,
            "persistent cooker did not promote the swage attachment "
            "into the common constraint graph"
        );
        for (std::size_t blockIndex = 14u;
             blockIndex < 17u;
             ++blockIndex) {
            const auto& block =
                compiledProgram.blocks[blockIndex];
            require(
                block.dimension == 1u &&
                    block.endpointCount == 2u &&
                    (block.flags &
                     MR_CONSTRAINT_IR_BLOCK_ROD_ATTACHMENT) !=
                        0u &&
                    compiledProgram.endpoints[
                        block.endpointOffset
                    ].jacobianKind ==
                        MR_CONSTRAINT_IR_JACOBIAN_ROD_NODE &&
                    compiledProgram.endpoints[
                        block.endpointOffset + 1u
                    ].jacobianKind ==
                        MR_CONSTRAINT_IR_JACOBIAN_BODY_LOCAL_POINT,
                "cooked swage row is not a typed rod/body scalar "
                "ConstraintIR block"
            );
        }
        const std::vector<float> persistentEffort(
            world.model.world.nv,
            0.0f
        );
        const metalrobo::MetalWorldBatch persistentBatch{
            .environmentCount = 1u,
            .controlStepCount = 1u,
            .initialQ = world.model.defaultQ,
            .initialV = world.model.defaultV,
            .efforts = persistentEffort,
        };
        metalrobo::MetalWorldStepConfig persistentConfig;
        persistentConfig.solverMode =
            metalrobo::MetalWorldSolverMode::freeMotionABA;
        persistentConfig.timestepSeconds = 1.0f / 1000.0f;
        persistentConfig.physicsSubsteps = 1u;
        persistentConfig.ccdMode =
            metalrobo::MetalWorldCCDMode::disabled;
        metalrobo::MetalWorldContext persistentContext;
        metalrobo::MetalWorldResult persistentResult;
        const auto persistentRun = persistentContext.run(
            persistentWorld,
            persistentBatch,
            persistentConfig,
            persistentResult
        );
        require(
            persistentRun.succeeded() &&
                persistentResult.finalRodNodes.size() ==
                    persistentWorld.rodNodeCount() &&
                persistentResult.finalRodEdges.size() ==
                    persistentWorld.rodEdgeCount() &&
                persistentResult.statuses.size() == 1u &&
                persistentResult.statuses[0].code ==
                    MR_STEP_SUCCESS &&
                persistentResult.finalRodNodes[4]
                        .position.z <
                    persistentWorld.defaultRodNodes()[4]
                        .position.z,
            "persistent multi-articulation rod step failed: " +
                persistentRun.message
        );

        // Exercise the same persistent state through the composed contact
        // graph. Even when this authored reset has no active rigid contact,
        // the fixed-capacity factor/Jacobian pass must execute both PSM
        // trees, assemble the global operator, and publish the rod in the
        // same transaction.
        const metalrobo::MetalWorldBatch persistentContactBatch{
            .environmentCount = 1u,
            .controlStepCount = 1u,
            .initialQ = world.model.defaultQ,
            .initialV = world.model.defaultV,
            .efforts = persistentEffort,
            .initialSceneBodies = world.defaultSceneBodies,
        };
        metalrobo::MetalWorldStepConfig
            persistentContactConfig = persistentConfig;
        persistentContactConfig.solverMode =
            metalrobo::MetalWorldSolverMode::throughputTGS;
        persistentContactConfig.captureContactEvidence = true;
        metalrobo::MetalWorldResult persistentContactResult;
        const auto persistentContactRun =
            persistentContext.run(
                persistentWorld,
                persistentContactBatch,
                persistentContactConfig,
                persistentContactResult
            );
        std::size_t persistentRodConstraints = 0u;
        bool persistentRodEndpoint = false;
        bool persistentRodIsland = false;
        if (!persistentContactResult.contactStatuses.empty()) {
            const std::size_t constraintCount =
                persistentContactResult.contactStatuses[0]
                    .requiredConstraints;
            for (std::size_t constraint = 0u;
                 constraint < constraintCount &&
                 constraint <
                     persistentContactResult
                         .contactEvidence.blocks.size();
                 ++constraint) {
                const auto& block =
                    persistentContactResult
                        .contactEvidence.blocks[constraint];
                if ((block.flags &
                     MR_CONSTRAINT_IR_BLOCK_ROD_ENDPOINT) ==
                    0u) {
                    continue;
                }
                ++persistentRodConstraints;
                const std::size_t endpoint =
                    block.endpointOffset;
                persistentRodEndpoint =
                    persistentRodEndpoint ||
                    (
                        endpoint <
                            persistentContactResult
                                .contactEvidence
                                .endpointRuntime.size() &&
                        persistentContactResult
                                .contactEvidence
                                .endpointRuntime[endpoint]
                                .ownerKind ==
                            MR_CONSTRAINT_IR_OWNER_ROD_EDGE
                    );
            }
            const std::size_t islandCount =
                persistentContactResult.contactStatuses[0]
                    .islandCount;
            for (std::size_t island = 0u;
                 island < islandCount &&
                 island <
                     persistentContactResult
                         .contactEvidence.islands.size();
                 ++island) {
                persistentRodIsland =
                    persistentRodIsland ||
                    persistentContactResult
                            .contactEvidence.islands[island]
                            .rodNodeCount != 0u;
            }
        }
        require(
            persistentContactRun.succeeded() &&
                persistentContactResult.statuses.size() == 1u &&
                persistentContactResult.statuses[0].code ==
                    MR_STEP_SUCCESS &&
                persistentContactResult.finalQ.size() ==
                    world.model.defaultQ.size() &&
                persistentContactResult.finalRodNodes.size() ==
                    persistentWorld.rodNodeCount() &&
                persistentRodConstraints != 0u &&
                persistentRodEndpoint &&
                persistentRodIsland,
            "persistent composed multi-articulation contact step failed: " +
                persistentContactRun.message +
                (
                    persistentContactResult.statuses.empty()
                    ? std::string{}
                    : " code=" +
                          std::to_string(
                              persistentContactResult.statuses[0]
                                  .code
                          ) +
                          " failing=" +
                          std::to_string(
                              persistentContactResult.statuses[0]
                                  .failingIndex
                          )
                ) +
                (
                    persistentContactResult.contactStatuses.empty()
                    ? std::string{}
                    : " contact=" +
                          std::to_string(
                              persistentContactResult
                                  .contactStatuses[0]
                                  .code
                          ) +
                          " pair=" +
                          std::to_string(
                              persistentContactResult
                                  .contactStatuses[0]
                                  .firstFailingPair
                          ) +
                          " constraint=" +
                          std::to_string(
                              persistentContactResult
                                  .contactStatuses[0]
                                  .firstFailingConstraint
                          ) +
                          " required=" +
                          std::to_string(
                              persistentContactResult
                                  .contactStatuses[0]
                                  .requiredConstraints
                          ) +
                          " stable=" +
                          std::to_string(
                              persistentContactResult
                                  .contactStatuses[0]
                                  .firstFailingStableKeyLow
                          ) +
                          "/" +
                          std::to_string(
                              persistentContactResult
                                  .contactStatuses[0]
                                  .firstFailingStableKeyHigh
                          )
                ) +
                " shape0=" +
                std::to_string(
                    world.model.shapes[0].shapeType
                ) +
                "/" +
                std::to_string(
                    world.model.shapes[0].bodyIndex
                ) +
                "/" +
                std::to_string(world.model.shapes[0].flags) +
                " pair0=" +
                std::to_string(
                    persistentWorld.eligiblePairs()[0].colliderA
                ) +
                "/" +
                std::to_string(
                    persistentWorld.eligiblePairs()[0].colliderB
                ) +
                "/" +
                std::to_string(
                    persistentWorld.eligiblePairs()[0].pairClass
                ) +
                " shapeB=" +
                std::to_string(
                    world.model.shapes[
                        persistentWorld.eligiblePairs()[0]
                            .colliderB
                    ].shapeType
                ) +
                "/" +
                std::to_string(
                    world.model.shapes[
                        persistentWorld.eligiblePairs()[0]
                            .colliderB
                    ].flags
                )
        );

        metalrobo::CompiledMetalMultiArticulatedProgram program;
        const auto programDiagnostics =
            metalrobo::compileMetalMultiArticulatedProgram(
                world.model,
                program
            );
        require(
            programDiagnostics.succeeded() &&
                program.valid() &&
                program.rowCount() == 14u,
            "heterogeneous multi-articulation program did not compile"
        );

        std::vector<double> q(
            world.model.defaultQ.begin(),
            world.model.defaultQ.end()
        );
        std::vector<double> v(
            world.model.defaultV.begin(),
            world.model.defaultV.end()
        );
        std::vector<double> force(world.model.world.nv, 0.0);
        metalrobo::MultiArticulationFactorCache cache;
        metalrobo::MultiArticulatedWorldConfig stepConfig;
        stepConfig.dynamics.timestep =
            world.model.world.gravityAndTimestep.w;
        stepConfig.solverIterations = 128u;
        stepConfig.solverTolerance = 1.0e-8;
        stepConfig.constraintResidual.residualTolerance =
            1.0e-7;
        const auto step =
            metalrobo::stepMultiArticulatedWorldCpu(
                world.model,
                q,
                v,
                force,
                {},
                cache,
                stepConfig
            );
        require(
            step.succeeded(),
            "heterogeneous articulation program did not execute"
        );

        std::ranges::fill(v, 0.0);
        const MRArticulationGPU& leftArticulation =
            world.model.articulations[0];
        const MRArticulationGPU& rightArticulation =
            world.model.articulations[1];
        v[leftArticulation.vOffset + 12u] = -0.2;
        v[leftArticulation.vOffset + 13u] = 0.2;
        v[rightArticulation.vOffset + 12u] = 0.2;
        v[rightArticulation.vOffset + 13u] = -0.2;
        metalrobo::MultiArticulatedIslandContact leftContact;
        leftContact.endpointA = {
            metalrobo::MultiContactEndpointKind::
                articulatedBody,
            leftArticulation.firstBody + 7u,
            {0.0, 0.0, 0.01},
        };
        leftContact.endpointB = {
            metalrobo::MultiContactEndpointKind::sceneBody,
            0u,
            {0.0, 0.0, 0.0},
        };
        leftContact.normal = {1.0, 0.0, 0.0};
        leftContact.tangentU = {0.0, 1.0, 0.0};
        leftContact.tangentV = {0.0, 0.0, 1.0};
        leftContact.regularization = {
            1.0e-6,
            1.0e-6,
            1.0e-6,
        };
        metalrobo::MultiArticulatedIslandContact rightContact =
            leftContact;
        rightContact.endpointA = {
            metalrobo::MultiContactEndpointKind::sceneBody,
            0u,
            {0.0, 0.0, 0.0},
        };
        rightContact.endpointB = {
            metalrobo::MultiContactEndpointKind::
                articulatedBody,
            rightArticulation.firstBody + 7u,
            {0.0, 0.0, 0.01},
        };
        const std::array<
            metalrobo::MultiArticulatedIslandContact,
            2
        > psmNeedleContacts{
            leftContact,
            rightContact,
        };
        metalrobo::MultiArticulatedContactProblem
            psmNeedleProblem;
        metalrobo::ArticulatedDynamicsConfig contactDynamics;
        contactDynamics.gravity = {0.0, 0.0, 0.0};
        contactDynamics.applyBodyDamping = false;
        const auto contactBuild =
            metalrobo::
                buildMultiArticulatedIslandContactProblem(
                    world.model,
                    q,
                    v,
                    world.defaultSceneBodies,
                    psmNeedleContacts,
                    psmNeedleProblem,
                    contactDynamics
                );
        metalrobo::ConstraintIREvaluationConfig
            equalityConfig;
        equalityConfig.timestep =
            world.model.world.gravityAndTimestep.w;
        const auto equalityProjection =
            metalrobo::
                projectMultiArticulatedContactThroughGeneralizedEqualities(
                    world.model,
                    psmNeedleProblem,
                    equalityConfig
                );
        metalrobo::MultiArticulatedContactSolution
            psmNeedleSolution;
        const auto contactSolve =
            metalrobo::solveMultiArticulatedContactProblem(
                psmNeedleProblem,
                psmNeedleSolution
            );
        require(
            contactBuild.succeeded() &&
                equalityProjection.succeeded() &&
                contactSolve.succeeded() &&
                psmNeedleProblem.articulatedNv ==
                    world.model.world.nv &&
                psmNeedleProblem.nv ==
                    world.model.world.nv + 6u &&
                psmNeedleProblem
                        .generalizedConstraintRowCount == 14u &&
                psmNeedleSolution
                        .generalizedConstraintImpulses.size() ==
                    14u &&
                contactSolve
                        .maximumGeneralizedConstraintResidual <
                    1.0e-9 &&
                psmNeedleSolution.impulses[0] > 0.0 &&
                psmNeedleSolution.impulses[3] > 0.0 &&
                psmNeedleSolution.quality.velocity[0] >
                    -2.0e-6 &&
                psmNeedleSolution.quality.velocity[3] >
                    -2.0e-6,
            "dual PSM and needle did not enter one coupled "
            "exact-cone island: build=" +
                std::to_string(contactBuild.succeeded()) +
                " projection=" +
                std::to_string(equalityProjection.succeeded()) +
                " solve=" +
                std::to_string(contactSolve.succeeded()) +
                " impulses=" +
                (
                    psmNeedleSolution.impulses.size() >= 4u
                    ? std::to_string(
                          psmNeedleSolution.impulses[0]
                      ) +
                          "/" +
                          std::to_string(
                              psmNeedleSolution.impulses[3]
                          )
                    : std::string{"missing"}
                )
        );

        metalrobo::CompiledMetalMultiArticulatedContactProgram
            contactProgram;
        const auto contactProgramDiagnostics =
            metalrobo::
                compileMetalMultiArticulatedContactProgram(
                    world.model,
                    contactProgram
                );
        require(
            contactProgramDiagnostics.succeeded() &&
                contactProgram.valid(),
            "heterogeneous Metal contact program did not compile"
        );
        const std::vector<float> q32(q.begin(), q.end());
        const std::vector<float> v32(v.begin(), v.end());
        auto metalPsmContacts = psmNeedleContacts;
        for (std::size_t contact = 0u;
             contact < metalPsmContacts.size();
             ++contact) {
            for (std::size_t row = 0u; row < 3u; ++row) {
                metalPsmContacts[contact].warmImpulse[row] =
                    psmNeedleSolution.impulses[
                        3u * contact + row
                    ];
            }
        }
        metalrobo::MetalMultiArticulatedContactInput metalInput;
        metalInput.environmentCount = 1u;
        metalInput.contactCount = psmNeedleContacts.size();
        metalInput.sceneBodyCount =
            world.defaultSceneBodies.size();
        metalInput.q = q32;
        metalInput.freeArticulationVelocity = v32;
        metalInput.sceneBodies = world.defaultSceneBodies;
        metalInput.contacts = metalPsmContacts;
        metalrobo::MetalMultiArticulatedContactConfig
            metalConfig;
        metalConfig.quality.maximumNewtonIterations = 64u;
        metalConfig.quality.maximumCGIterations = 128u;
        metalConfig.quality.convergenceTolerance = 3.0e-5F;
        metalrobo::MetalMultiArticulatedContactResult
            metalContact;
        const auto metalContactDiagnostics =
            metalrobo::solveMetalMultiArticulatedContacts(
                contactProgram,
                metalInput,
                metalContact,
                metalConfig
            );
        require(
            metalContactDiagnostics.succeeded() &&
                metalContact.layout.dispatch.totalNv ==
                    psmNeedleProblem.nv,
            "dual PSM/needle Metal contact graph failed: " +
                metalContactDiagnostics.message +
                (
                    metalContact.qualityStatuses.empty()
                    ? std::string{}
                    : " quality=" +
                          std::to_string(
                              metalContact.qualityStatuses[0]
                                  .code
                          ) +
                          ":" +
                          std::to_string(
                              metalContact.qualityStatuses[0]
                                  .diagnostics.x
                          ) +
                          "/" +
                          std::to_string(
                              metalContact.qualityStatuses[0]
                                  .diagnostics.w
                          ) +
                          " iterations=" +
                          std::to_string(
                              metalContact.qualityStatuses[0]
                                  .newtonIterations
                          ) +
                          "/" +
                          std::to_string(
                              metalContact.qualityStatuses[0]
                                  .cgIterations
                          ) +
                          " fallbacks=" +
                          std::to_string(
                              metalContact.qualityStatuses[0]
                                  .projectedGradientFallbacks
                          )
                )
        );
        double maximumMetalContactError = 0.0;
        double maximumMetalDelassusError = 0.0;
        for (std::size_t index = 0u;
             index < psmNeedleProblem.conic.delassus.size();
             ++index) {
            maximumMetalDelassusError = std::max(
                maximumMetalDelassusError,
                std::abs(
                    psmNeedleProblem.conic.delassus[index] -
                    metalContact.delassus[index]
                )
            );
        }
        for (std::size_t index = 0u;
             index < psmNeedleSolution.impulses.size();
             ++index) {
            maximumMetalContactError = std::max(
                maximumMetalContactError,
                std::abs(
                    psmNeedleSolution.impulses[index] -
                    metalContact.impulses[index]
                )
            );
        }
        for (std::size_t index = 0u;
             index <
                 psmNeedleSolution.generalizedVelocity.size();
             ++index) {
            maximumMetalContactError = std::max(
                maximumMetalContactError,
                std::abs(
                    psmNeedleSolution.generalizedVelocity[
                        index
                    ] -
                    metalContact.nextVelocity[index]
                )
            );
        }
        for (std::size_t index = 0u;
             index <
                 psmNeedleSolution
                     .generalizedConstraintImpulses.size();
             ++index) {
            maximumMetalContactError = std::max(
                maximumMetalContactError,
                std::abs(
                    psmNeedleSolution
                        .generalizedConstraintImpulses[index] -
                    metalContact.equalityImpulses[index]
                )
            );
        }
        require(
            maximumMetalContactError < 3.0e-3 &&
                metalContact.equalityStatuses.size() == 1u &&
                metalContact.equalityStatuses[0].code ==
                    MR_MULTI_CONTACT_EQUALITY_SUCCESS &&
                metalContact.equalityStatuses[0]
                        .diagnostics.x <
                    metalConfig.equalityResidualTolerance,
            "dual PSM/needle Metal contact disagrees with FP64: " +
                std::to_string(maximumMetalContactError) +
                " W=" +
                std::to_string(maximumMetalDelassusError) +
                " impulses=" +
                std::to_string(psmNeedleSolution.impulses[0]) +
                "/" +
                std::to_string(metalContact.impulses[0]) +
                "," +
                std::to_string(psmNeedleSolution.impulses[3]) +
                "/" +
                std::to_string(metalContact.impulses[3]) +
                " free=" +
                std::to_string(
                    psmNeedleProblem.conic
                        .freeContactVelocity[0]
                ) +
                "/" +
                std::to_string(
                    metalContact.freeContactVelocity[0]
                ) +
                "," +
                std::to_string(
                    psmNeedleProblem.conic
                        .freeContactVelocity[3]
                ) +
                "/" +
                std::to_string(
                    metalContact.freeContactVelocity[3]
                ) +
                " Wn=" +
                std::to_string(
                    psmNeedleProblem.conic.delassus[0]
                ) +
                "/" +
                std::to_string(metalContact.delassus[0]) +
                "," +
                std::to_string(
                    psmNeedleProblem.conic.delassus[
                        3u * 6u + 3u
                    ]
                ) +
                "/" +
                std::to_string(
                    metalContact.delassus[3u * 6u + 3u]
                ) +
                " quality=" +
                std::to_string(
                    metalContact.qualityStatuses[0].code
                ) +
                ":" +
                std::to_string(
                    metalContact.qualityStatuses[0]
                        .diagnostics.x
                )
        );
        auto coldConfig = metalConfig;
        coldConfig.quality.enableWarmStart = false;
        metalrobo::MetalMultiArticulatedContactResult
            coldContact;
        const auto coldDiagnostics =
            metalrobo::solveMetalMultiArticulatedContacts(
                contactProgram,
                metalInput,
                coldContact,
                coldConfig
            );
        require(
            !coldDiagnostics.succeeded() &&
                coldDiagnostics.status ==
                    metalrobo::
                        MetalMultiArticulatedContactStatus::
                            gpuEnvironmentFailure &&
                coldDiagnostics.published &&
                coldContact.statuses[0].code ==
                    MR_MULTI_CONTACT_QUALITY_FAILED &&
                coldContact.qualityStatuses[0].code ==
                    MR_METAL_QUALITY_DID_NOT_CONVERGE &&
                std::equal(
                    v32.begin(),
                    v32.end(),
                    coldContact.nextVelocity.begin()
                ),
            "ill-conditioned cold quality solve was not rejected "
            "transactionally"
        );

        metalrobo::HeterogeneousRodProgram rod = world.rods[0];
        require(
            rod.stepConfig.enableSelfCollision,
            "surgical heterogeneous rod did not own self-contact"
        );
        rod.defaultState.positions[4][1] += 0.005;
        metalrobo::DiscreteElasticRodStepConfig rodConfig =
            rod.stepConfig;
        rodConfig.gravity = {0.0, 0.0, 0.0};
        rodConfig.solverIterations = 256u;
        rodConfig.constraintTolerance = 1.0e-5;
        std::array<
            metalrobo::DiscreteRodAttachmentReaction,
            1
        > reactions{};
        const auto rodStep =
            metalrobo::stepDiscreteElasticRodCpu(
                rod.model,
                rod.defaultState,
                rod.attachments,
                rodConfig,
                reactions
            );
        require(
            rodStep.succeeded(),
            "heterogeneous rod sidecar failed: " +
                rodStep.message
        );
        require(
            reactions[0].bodyIndex ==
                metalrobo::kDiscreteRodNoRigidBody &&
                norm(reactions[0].averageForceOnTarget) > 0.0,
            "heterogeneous rod reaction evidence is empty"
        );

        metalrobo::HeterogeneousWorld replay;
        const auto replayDiagnostics =
            metalrobo::
                makeDualDvrkPsmNeedleThreadHeterogeneousWorld(
                    replay,
                    config
                );
        require(
            replayDiagnostics.succeeded() &&
                replay.fingerprint == world.fingerprint &&
                replay.sceneBodyIndices == world.sceneBodyIndices,
            "heterogeneous cook replay changed"
        );

        metalrobo::HeterogeneousWorld sentinel = world;
        const std::uint64_t sentinelFingerprint =
            sentinel.fingerprint;
        auto rejectedConfig = config;
        rejectedConfig.threadNodeCount = 1u;
        const auto rejected =
            metalrobo::
                makeDualDvrkPsmNeedleThreadHeterogeneousWorld(
                    sentinel,
                    rejectedConfig
                );
        require(
            !rejected.succeeded() &&
                sentinel.fingerprint == sentinelFingerprint &&
                sentinel.valid(),
            "failed heterogeneous cook published partial state"
        );

        std::cout
            << "heterogeneous_world=ok"
            << " components="
            << world.componentInstanceIds.size()
            << " articulations="
            << world.model.articulations.size()
            << " scene_bodies="
            << world.defaultSceneBodies.size()
            << " colliders=" << world.model.shapes.size()
            << " constraint_rows=" << program.rowCount()
            << " rods=" << world.rods.size()
            << " persistent_rod_nodes="
            << persistentResult.finalRodNodes.size()
            << " composed_constraints="
            << persistentContactResult.contactStatuses[0]
                   .requiredConstraints
            << " rod_constraints="
            << persistentRodConstraints
            << " contact_nv=" << psmNeedleProblem.nv
            << " needle_contact_impulses="
            << psmNeedleSolution.impulses[0] << "/"
            << psmNeedleSolution.impulses[3]
            << " equality_residual="
            << contactSolve
                   .maximumGeneralizedConstraintResidual
            << " metal_equality_residual="
            << metalContact.equalityStatuses[0].diagnostics.x
            << " metal_contact_error="
            << maximumMetalContactError
            << " metal_kkt="
            << metalContact.qualityStatuses[0].diagnostics.w
            << " fingerprint=" << world.fingerprint
            << " swage_force_n="
            << norm(reactions[0].averageForceOnTarget)
            << " deterministic=yes transactional=yes"
            << " cold_start_rejected=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "heterogeneous_world=failed reason=\""
            << error.what()
            << "\"\n";
        return 1;
    }
}
