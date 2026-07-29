#include "metalrobo/EngineModelComposer.hpp"
#include "metalrobo/MetalMultiArticulatedContact.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
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

metalrobo::EngineModel makeTwoSpheres() {
    const metalrobo::EngineModel first =
        metalrobo::makeFreeSphereEngineModel();
    const metalrobo::EngineModel second =
        metalrobo::makeFreeSphereEngineModel();
    const std::array<metalrobo::EngineModelComponent, 2>
        components{{
            {&first, "sphere_a"},
            {&second, "sphere_b"},
        }};
    metalrobo::EngineModel output;
    metalrobo::EngineModelComposeConfig config;
    config.name = "metal_two_articulation_scene_contact";
    config.gravityAndTimestep = {
        0.0F, 0.0F, 0.0F, 1.0F / 1000.0F,
    };
    const auto diagnostics =
        metalrobo::composeEngineModels(
            components,
            output,
            config
        );
    require(
        diagnostics.succeeded(),
        "model composition failed: " + diagnostics.message
    );
    return output;
}

MRBodyStateGPU makeSceneBody() {
    MRBodyStateGPU body{};
    body.position = {0.0F, 1.0F, 0.0F, 1.0F};
    body.orientation = {0.0F, 0.0F, 0.0F, 1.0F};
    body.linearVelocityAndInverseMass = {
        0.0F, 0.0F, 0.0F, 1.0F,
    };
    body.inverseInertiaWorldRow0 = {
        1.0F, 0.0F, 0.0F, 0.0F,
    };
    body.inverseInertiaWorldRow1 = {
        0.0F, 1.0F, 0.0F, 0.0F,
    };
    body.inverseInertiaWorldRow2 = {
        0.0F, 0.0F, 1.0F, 0.0F,
    };
    body.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
    body.flagsAndIndices[1] = MR_INVALID_INDEX;
    body.flagsAndIndices[2] = MR_INVALID_INDEX;
    return body;
}

std::array<metalrobo::MultiArticulatedIslandContact, 2>
makeContacts(
    const metalrobo::EngineModel& model
) {
    const MRArticulationGPU& first =
        model.articulations[0];
    const MRArticulationGPU& second =
        model.articulations[1];
    metalrobo::MultiArticulatedIslandContact left;
    left.endpointA = {
        metalrobo::MultiContactEndpointKind::articulatedBody,
        first.rootBody,
        {0.25, 0.0, 0.0},
    };
    left.endpointB = {
        metalrobo::MultiContactEndpointKind::sceneBody,
        0u,
        {-0.25, 0.0, 0.0},
    };
    left.normal = {1.0, 0.0, 0.0};
    left.tangentU = {0.0, 1.0, 0.0};
    left.tangentV = {0.0, 0.0, 1.0};
    left.regularization = {
        1.0e-6,
        1.0e-6,
        1.0e-6,
    };
    left.friction = 0.7;

    metalrobo::MultiArticulatedIslandContact right = left;
    right.endpointA = {
        metalrobo::MultiContactEndpointKind::sceneBody,
        0u,
        {0.25, 0.0, 0.0},
    };
    right.endpointB = {
        metalrobo::MultiContactEndpointKind::articulatedBody,
        second.rootBody,
        {-0.25, 0.0, 0.0},
    };
    return {left, right};
}

template <typename T>
bool bitwiseEqual(
    const std::vector<T>& left,
    const std::vector<T>& right
) {
    return left.size() == right.size() &&
        (left.empty() ||
         std::memcmp(
             left.data(),
             right.data(),
             left.size() * sizeof(T)
         ) == 0);
}

} // namespace

int main() {
    try {
        constexpr std::size_t environments = 4u;
        constexpr std::size_t contactsPerEnvironment = 2u;
        const metalrobo::EngineModel model = makeTwoSpheres();
        require(
            model.articulations.size() == 2u &&
                model.world.nv == 12u,
            "unexpected composed model topology"
        );
        const MRArticulationGPU& first =
            model.articulations[0];
        const MRArticulationGPU& second =
            model.articulations[1];

        std::vector<float> q(
            environments * model.world.nq
        );
        std::vector<float> freeVelocity(
            environments * model.world.nv,
            0.0F
        );
        std::vector<MRBodyStateGPU> sceneBodies(
            environments,
            makeSceneBody()
        );
        std::vector<
            metalrobo::MultiArticulatedIslandContact
        > contacts;
        contacts.reserve(
            environments * contactsPerEnvironment
        );
        const auto contactTemplate = makeContacts(model);
        for (std::size_t environment = 0u;
             environment < environments;
             ++environment) {
            std::copy(
                model.defaultQ.begin(),
                model.defaultQ.end(),
                q.begin() +
                    static_cast<std::ptrdiff_t>(
                        environment * model.world.nq
                    )
            );
            q[environment * model.world.nq +
              first.qOffset] = -0.25F;
            q[environment * model.world.nq +
              second.qOffset] = 0.25F;
            freeVelocity[
                environment * model.world.nv +
                first.vOffset
            ] = 1.0F;
            freeVelocity[
                environment * model.world.nv +
                second.vOffset
            ] = -1.0F;
            contacts.insert(
                contacts.end(),
                contactTemplate.begin(),
                contactTemplate.end()
            );
        }

        metalrobo::MetalMultiArticulatedContactInput input;
        input.environmentCount = environments;
        input.contactCount = contactsPerEnvironment;
        input.sceneBodyCount = 1u;
        input.q = q;
        input.freeArticulationVelocity = freeVelocity;
        input.sceneBodies = sceneBodies;
        input.contacts = contacts;
        metalrobo::MetalMultiArticulatedContactConfig config;
        config.quality.maximumNewtonIterations = 64u;
        config.quality.maximumCGIterations = 128u;
        config.quality.convergenceTolerance = 3.0e-5F;

        metalrobo::MetalMultiArticulatedContactResult gpu;
        const auto diagnostics =
            metalrobo::solveMetalMultiArticulatedContacts(
                model,
                input,
                gpu,
                config
            );
        require(
            diagnostics.succeeded(),
            "Metal contact graph failed: " +
                diagnostics.message
        );
        require(
            diagnostics.dispatched &&
                diagnostics.published &&
                gpu.layout.dispatch.totalNv == 18u &&
                gpu.layout.dispatch.rowCount == 6u &&
                gpu.layout.pointDispatches.size() == 2u &&
                gpu.layout.inverseMassDispatches.size() == 4u,
            "Metal contact graph layout is incorrect"
        );

        double maximumVelocity = 0.0;
        for (std::size_t environment = 0u;
             environment < environments;
             ++environment) {
            const std::size_t velocityBase =
                environment * gpu.layout.dispatch.totalNv;
            const std::size_t rowBase =
                environment * gpu.layout.dispatch.rowCount;
            const std::size_t matrixBase =
                environment *
                gpu.layout.dispatch.rowCount *
                gpu.layout.dispatch.rowCount;
            require(
                gpu.statuses[environment].code ==
                    MR_MULTI_CONTACT_SUCCESS &&
                    gpu.pointStatuses[
                        0u * environments + environment
                    ].code ==
                        MR_ARTICULATED_OPERATOR_SUCCESS &&
                    gpu.pointStatuses[
                        1u * environments + environment
                    ].code ==
                        MR_ARTICULATED_OPERATOR_SUCCESS,
                "a healthy environment failed"
            );
            require(
                std::abs(
                    gpu.delassus[matrixBase + 0u] - 2.0F
                ) < 2.0e-4F &&
                std::abs(
                    gpu.delassus[matrixBase + 3u] + 1.0F
                ) < 2.0e-4F &&
                std::abs(
                    gpu.delassus[
                        matrixBase + 3u * 6u + 3u
                    ] - 2.0F
                ) < 2.0e-4F,
                "Metal mixed Delassus operator is incorrect"
            );
            require(
                std::abs(gpu.impulses[rowBase] - 1.0F) <
                    2.5e-3F &&
                std::abs(gpu.impulses[rowBase + 3u] - 1.0F) <
                    2.5e-3F,
                "Metal exact-cone impulses are incorrect"
            );
            for (std::size_t dof = 0u;
                 dof < gpu.layout.dispatch.totalNv;
                 ++dof) {
                maximumVelocity = std::max(
                    maximumVelocity,
                    std::abs(
                        static_cast<double>(
                            gpu.nextVelocity[
                                velocityBase + dof
                            ]
                        )
                    )
                );
            }
        }
        require(
            maximumVelocity < 2.5e-3,
            "Metal solution did not stop the coupled bodies"
        );

        std::vector<double> q64(
            q.begin(),
            q.begin() +
                static_cast<std::ptrdiff_t>(model.world.nq)
        );
        std::vector<double> velocity64(
            freeVelocity.begin(),
            freeVelocity.begin() +
                static_cast<std::ptrdiff_t>(model.world.nv)
        );
        metalrobo::MultiArticulatedContactProblem oracle;
        metalrobo::ArticulatedDynamicsConfig dynamics;
        dynamics.gravity = {0.0, 0.0, 0.0};
        dynamics.applyBodyDamping = false;
        const auto oracleBuild =
            metalrobo::
                buildMultiArticulatedIslandContactProblem(
                    model,
                    q64,
                    velocity64,
                    std::span<const MRBodyStateGPU>(
                        sceneBodies.data(),
                        1u
                    ),
                    std::span<
                        const metalrobo::
                            MultiArticulatedIslandContact
                    >(
                        contacts.data(),
                        contactsPerEnvironment
                    ),
                    oracle,
                    dynamics
                );
        require(
            oracleBuild.succeeded(),
            "FP64 mixed contact operator failed"
        );
        metalrobo::QualityContactSolverConfig cpuConfig;
        cpuConfig.kktTolerance = 1.0e-11;
        metalrobo::MultiArticulatedContactSolution cpu;
        const auto oracleSolve =
            metalrobo::solveMultiArticulatedContactProblem(
                oracle,
                cpu,
                cpuConfig
            );
        require(
            oracleSolve.succeeded(),
            "FP64 mixed exact-cone solve failed"
        );
        double maximumOracleError = 0.0;
        for (std::size_t index = 0u;
             index < cpu.generalizedVelocity.size();
             ++index) {
            maximumOracleError = std::max(
                maximumOracleError,
                std::abs(
                    cpu.generalizedVelocity[index] -
                    gpu.nextVelocity[index]
                )
            );
        }
        for (std::size_t index = 0u;
             index < cpu.impulses.size();
             ++index) {
            maximumOracleError = std::max(
                maximumOracleError,
                std::abs(
                    cpu.impulses[index] -
                    gpu.impulses[index]
                )
            );
        }
        require(
            maximumOracleError < 2.5e-3,
            "Metal result disagrees with the FP64 oracle"
        );

        metalrobo::MetalMultiArticulatedContactResult replay;
        const auto replayDiagnostics =
            metalrobo::solveMetalMultiArticulatedContacts(
                model,
                input,
                replay,
                config
            );
        require(
            replayDiagnostics.succeeded() &&
                bitwiseEqual(
                    gpu.nextVelocity,
                    replay.nextVelocity
                ) &&
                bitwiseEqual(gpu.impulses, replay.impulses) &&
                bitwiseEqual(gpu.delassus, replay.delassus) &&
                bitwiseEqual(gpu.statuses, replay.statuses),
            "same-device Metal replay changed"
        );

        metalrobo::MetalMultiArticulatedContactResult sentinel =
            gpu;
        const std::vector<float> acceptedVelocity =
            sentinel.nextVelocity;
        metalrobo::MetalMultiArticulatedContactInput invalid =
            input;
        invalid.contactCount = 0u;
        const auto rejected =
            metalrobo::solveMetalMultiArticulatedContacts(
                model,
                invalid,
                sentinel,
                config
            );
        require(
            !rejected.succeeded() &&
                !rejected.dispatched &&
                sentinel.nextVelocity == acceptedVelocity,
            "pre-dispatch failure mutated the accepted result"
        );

        std::vector<float> failedQ = q;
        constexpr std::size_t failedEnvironment = 2u;
        for (std::size_t component = 3u;
             component < 7u;
             ++component) {
            failedQ[
                failedEnvironment * model.world.nq +
                first.qOffset + component
            ] = 0.0F;
        }
        metalrobo::MetalMultiArticulatedContactInput injected =
            input;
        injected.q = failedQ;
        metalrobo::MetalMultiArticulatedContactResult partial;
        const auto injectedDiagnostics =
            metalrobo::solveMetalMultiArticulatedContacts(
                model,
                injected,
                partial,
                config
            );
        require(
            !injectedDiagnostics.succeeded() &&
                injectedDiagnostics.status ==
                    metalrobo::
                        MetalMultiArticulatedContactStatus::
                            gpuEnvironmentFailure &&
                injectedDiagnostics.published &&
                injectedDiagnostics.firstFailingEnvironment ==
                    failedEnvironment &&
                partial.statuses[failedEnvironment].code ==
                    MR_MULTI_CONTACT_POINT_JACOBIAN_FAILED,
            "GPU failure injection was not isolated"
        );
        for (std::size_t environment = 0u;
             environment < environments;
             ++environment) {
            if (environment == failedEnvironment) {
                const std::size_t outputBase =
                    environment *
                    partial.layout.dispatch.totalNv;
                for (std::size_t dof = 0u;
                     dof < model.world.nv;
                     ++dof) {
                    require(
                        partial.nextVelocity[outputBase + dof] ==
                            freeVelocity[
                                environment * model.world.nv +
                                dof
                            ],
                        "failed environment did not roll back"
                    );
                }
                continue;
            }
            require(
                partial.statuses[environment].code ==
                    MR_MULTI_CONTACT_SUCCESS,
                "failure injection poisoned a healthy environment"
            );
        }

        std::cout
            << "metal_multi_articulated_contact=ok"
            << " device=\"" << diagnostics.deviceName << "\""
            << " environments=" << environments
            << " articulations="
            << gpu.layout.dispatch.articulationCount
            << " scene_bodies="
            << gpu.layout.dispatch.sceneBodyCount
            << " total_nv=" << gpu.layout.dispatch.totalNv
            << " rows=" << gpu.layout.dispatch.rowCount
            << " impulse=" << gpu.impulses[0]
            << " max_velocity=" << maximumVelocity
            << " oracle_error=" << maximumOracleError
            << " deterministic=yes"
            << " transactional_failure=yes"
            << " elapsed_ms="
            << diagnostics.elapsedMilliseconds
            << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "metal_multi_articulated_contact=failed reason=\""
            << error.what() << "\"\n";
        return 1;
    }
}
