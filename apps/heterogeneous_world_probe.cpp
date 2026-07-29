#include "metalrobo/HeterogeneousWorld.hpp"
#include "metalrobo/MetalMultiArticulatedContact.hpp"
#include "metalrobo/MetalMultiArticulatedConstraints.hpp"
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
        rod.defaultState.positions[4][1] += 0.005;
        metalrobo::DiscreteElasticRodStepConfig rodConfig;
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
