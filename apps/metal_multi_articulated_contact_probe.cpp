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

        metalrobo::CompiledMetalMultiArticulatedContactProgram
            program;
        const auto compiled =
            metalrobo::
                compileMetalMultiArticulatedContactProgram(
                    model,
                    program
                );
        require(
            compiled.succeeded() &&
                compiled.published &&
                program.valid() &&
                program.fingerprint() != 0u,
            "Metal contact program did not compile"
        );
        metalrobo::MetalMultiArticulatedContactResult gpu;
        const auto diagnostics =
            metalrobo::solveMetalMultiArticulatedContacts(
                program,
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

        metalrobo::MultiArticulatedIslandContact loopContact;
        loopContact.endpointA = {
            metalrobo::MultiContactEndpointKind::articulatedBody,
            first.rootBody,
            {0.0, 0.0, 0.0},
        };
        loopContact.endpointB = {
            metalrobo::MultiContactEndpointKind::staticWorld,
            MR_INVALID_INDEX,
            {0.0, 0.0, 0.0},
        };
        loopContact.normal = {1.0, 0.0, 0.0};
        loopContact.tangentU = {0.0, 1.0, 0.0};
        loopContact.tangentV = {0.0, 0.0, 1.0};
        loopContact.regularization = {
            1.0e-6,
            1.0e-6,
            1.0e-6,
        };
        constexpr double inverseRootTwo =
            0.70710678118654752440;
        metalrobo::MultiArticulatedPointEquality pointLoop;
        pointLoop.key.words[0] = 0x4d4c4f4fu;
        pointLoop.key.words[1] = 1u;
        pointLoop.endpointA = {
            metalrobo::MultiContactEndpointKind::articulatedBody,
            first.rootBody,
            {0.0, 0.0, 0.0},
        };
        pointLoop.endpointB = {
            metalrobo::MultiContactEndpointKind::articulatedBody,
            second.rootBody,
            {0.0, 0.0, 0.0},
        };
        pointLoop.axisX = {
            inverseRootTwo,
            inverseRootTwo,
            0.0,
        };
        pointLoop.axisY = {
            -inverseRootTwo,
            inverseRootTwo,
            0.0,
        };
        pointLoop.axisZ = {0.0, 0.0, 1.0};
        pointLoop.compliance = {
            1.0e-12,
            1.0e-12,
            1.0e-12,
        };
        pointLoop.positionStabilized = false;
        metalrobo::MultiArticulatedPointEquality bodyLoop =
            pointLoop;
        bodyLoop.key.words[0] = 0x4d424f44u;
        bodyLoop.key.words[1] = 2u;
        bodyLoop.endpointA = {
            metalrobo::MultiContactEndpointKind::articulatedBody,
            second.rootBody,
            {0.0, 0.0, 0.0},
        };
        bodyLoop.endpointB = {
            metalrobo::MultiContactEndpointKind::sceneBody,
            0u,
            {0.0, 0.0, 0.0},
        };
        bodyLoop.axisX = {0.0, 0.0, 1.0};
        bodyLoop.axisY = {
            inverseRootTwo,
            inverseRootTwo,
            0.0,
        };
        bodyLoop.axisZ = {
            -inverseRootTwo,
            inverseRootTwo,
            0.0,
        };
        metalrobo::MultiArticulatedPointEquality boundaryLoop =
            pointLoop;
        boundaryLoop.key.words[0] = 0x4d424e44u;
        boundaryLoop.key.words[1] = 3u;
        boundaryLoop.endpointA = {
            metalrobo::MultiContactEndpointKind::sceneBody,
            0u,
            {0.0, 0.0, 0.0},
        };
        boundaryLoop.endpointB = {
            metalrobo::MultiContactEndpointKind::sceneBody,
            1u,
            {0.0, 0.0, 0.0},
        };
        const std::array pointLoopTemplate{
            boundaryLoop,
            bodyLoop,
            pointLoop,
        };
        metalrobo::MultiArticulatedAngularEquality
            angularPoint;
        angularPoint.key.words[0] = 0x5a4c4f4fu;
        angularPoint.key.words[1] = 1u;
        angularPoint.endpointA = pointLoop.endpointA;
        angularPoint.endpointB = pointLoop.endpointB;
        angularPoint.axisX = pointLoop.axisX;
        angularPoint.axisY = pointLoop.axisY;
        angularPoint.axisZ = pointLoop.axisZ;
        angularPoint.compliance = pointLoop.compliance;
        angularPoint.positionStabilized = false;
        metalrobo::MultiArticulatedAngularEquality
            angularBody = angularPoint;
        angularBody.key.words[0] = 0x5a424f44u;
        angularBody.key.words[1] = 2u;
        angularBody.endpointA = bodyLoop.endpointA;
        angularBody.endpointB = bodyLoop.endpointB;
        angularBody.axisX = bodyLoop.axisX;
        angularBody.axisY = bodyLoop.axisY;
        angularBody.axisZ = bodyLoop.axisZ;
        metalrobo::MultiArticulatedAngularEquality
            angularBoundary = angularPoint;
        angularBoundary.key.words[0] = 0x5a424e44u;
        angularBoundary.key.words[1] = 3u;
        angularBoundary.endpointA = boundaryLoop.endpointA;
        angularBoundary.endpointB = boundaryLoop.endpointB;
        const std::array angularLoopTemplate{
            angularBoundary,
            angularBody,
            angularPoint,
        };

        std::vector<float> loopFreeVelocity(
            environments * model.world.nv,
            0.0F
        );
        std::vector<
            metalrobo::MultiArticulatedIslandContact
        > loopContacts(environments, loopContact);
        std::vector<MRBodyStateGPU> loopSceneBodies;
        loopSceneBodies.reserve(2u * environments);
        std::vector<
            metalrobo::MultiArticulatedPointEquality
        > pointLoops;
        pointLoops.reserve(
            environments * pointLoopTemplate.size()
        );
        std::vector<
            metalrobo::MultiArticulatedAngularEquality
        > angularLoops;
        angularLoops.reserve(
            environments * angularLoopTemplate.size()
        );
        for (std::size_t environment = 0u;
             environment < environments;
             ++environment) {
            loopFreeVelocity[
                environment * model.world.nv +
                first.vOffset
            ] = 1.0F;
            loopFreeVelocity[
                environment * model.world.nv +
                first.vOffset + 5u
            ] = 1.0F;
            loopFreeVelocity[
                environment * model.world.nv +
                second.vOffset + 5u
            ] = -0.5F;
            MRBodyStateGPU dynamicBody = makeSceneBody();
            dynamicBody.angularVelocity.z = 0.25F;
            loopSceneBodies.push_back(dynamicBody);
            MRBodyStateGPU boundary = makeSceneBody();
            boundary.position.y = 2.0F;
            boundary.linearVelocityAndInverseMass = {
                -0.2F, 0.0F, 0.0F, 0.0F,
            };
            boundary.inverseInertiaWorldRow0 = {};
            boundary.inverseInertiaWorldRow1 = {};
            boundary.inverseInertiaWorldRow2 = {};
            boundary.flagsAndIndices[0] =
                MR_MOTION_KINEMATIC;
            boundary.angularVelocity.z = -0.1F;
            loopSceneBodies.push_back(boundary);
            pointLoops.insert(
                pointLoops.end(),
                pointLoopTemplate.begin(),
                pointLoopTemplate.end()
            );
            angularLoops.insert(
                angularLoops.end(),
                angularLoopTemplate.begin(),
                angularLoopTemplate.end()
            );
        }
        metalrobo::MetalMultiArticulatedContactInput loopInput;
        loopInput.environmentCount = environments;
        loopInput.contactCount = 1u;
        loopInput.sceneBodyCount = 2u;
        loopInput.q = q;
        loopInput.freeArticulationVelocity =
            loopFreeVelocity;
        loopInput.sceneBodies = loopSceneBodies;
        loopInput.contacts = loopContacts;
        loopInput.pointEqualityCount =
            pointLoopTemplate.size();
        loopInput.pointEqualities = pointLoops;
        loopInput.angularEqualityCount =
            angularLoopTemplate.size();
        loopInput.angularEqualities = angularLoops;

        std::vector<double> loopFree64(
            model.world.nv,
            0.0
        );
        loopFree64[first.vOffset] = 1.0;
        loopFree64[first.vOffset + 5u] = 1.0;
        loopFree64[second.vOffset + 5u] = -0.5;
        metalrobo::MultiArticulatedContactProblem
            loopOracle;
        const auto loopOracleBuild =
            metalrobo::
                buildMultiArticulatedIslandContactProblem(
                    model,
                    q64,
                    loopFree64,
                    std::span<const MRBodyStateGPU>(
                        loopSceneBodies.data(),
                        2u
                    ),
                    std::span<
                        const metalrobo::
                            MultiArticulatedIslandContact
                    >(&loopContact, 1u),
                    loopOracle,
                    dynamics
                );
        const auto loopOracleProjection =
            metalrobo::
                projectMultiArticulatedContactThroughSpatialEqualities(
                    model,
                    q64,
                    loopOracle,
                    std::span<
                        const metalrobo::
                            MultiArticulatedPointEquality
                    >(
                        pointLoopTemplate.data(),
                        pointLoopTemplate.size()
                    ),
                    std::span<
                        const metalrobo::
                            MultiArticulatedAngularEquality
                    >(
                        angularLoopTemplate.data(),
                        angularLoopTemplate.size()
                    ),
                    config.equalityEvaluation,
                    dynamics
                );
        metalrobo::MultiArticulatedContactSolution
            loopOracleSolution;
        const auto loopOracleSolve =
            metalrobo::solveMultiArticulatedContactProblem(
                loopOracle,
                loopOracleSolution,
                cpuConfig
            );
        require(
            loopOracleBuild.succeeded() &&
                loopOracleProjection.succeeded() &&
                loopOracleSolve.succeeded(),
            "FP64 point-loop contact oracle failed"
        );

        metalrobo::MetalMultiArticulatedContactResult
            loopGPU;
        const auto loopDiagnostics =
            metalrobo::solveMetalMultiArticulatedContacts(
                program,
                loopInput,
                loopGPU,
                config
            );
        require(
                loopDiagnostics.succeeded() &&
                loopGPU.layout.dispatch.equalityRowCount ==
                    18u &&
                loopGPU.layout.dispatch
                        .staticEqualityRowCount == 0u &&
                loopGPU.equalityStatuses.size() ==
                    environments &&
                loopGPU.equalityStatuses[0].code ==
                    MR_MULTI_CONTACT_EQUALITY_SUCCESS,
            "Metal point-loop contact graph failed: " +
                loopDiagnostics.message +
                " gpu_status=" +
                std::to_string(
                    loopDiagnostics.firstGPUStatusCode
                ) +
                " equality_status=" +
                (
                    loopGPU.equalityStatuses.empty()
                    ? std::string{"empty"}
                    : std::to_string(
                          loopGPU.equalityStatuses[0].code
                      )
                ) +
                " residual=" +
                (
                    loopGPU.equalityStatuses.empty()
                    ? std::string{"empty"}
                    : std::to_string(
                          loopGPU.equalityStatuses[0]
                              .diagnostics.x
                      )
                )
        );
        double maximumPointLoopVelocityError = 0.0;
        double maximumPointLoopContactError = 0.0;
        double maximumPointLoopEqualityError = 0.0;
        for (std::size_t index = 0u;
             index <
                 loopOracleSolution
                     .generalizedVelocity.size();
             ++index) {
            maximumPointLoopVelocityError = std::max(
                maximumPointLoopVelocityError,
                std::abs(
                    loopOracleSolution
                        .generalizedVelocity[index] -
                    loopGPU.nextVelocity[index]
                )
            );
        }
        for (std::size_t index = 0u;
             index < loopOracleSolution.impulses.size();
             ++index) {
            maximumPointLoopContactError = std::max(
                maximumPointLoopContactError,
                std::abs(
                    loopOracleSolution.impulses[index] -
                    loopGPU.impulses[index]
                )
            );
        }
        for (std::size_t index = 0u;
             index <
                 loopOracleSolution
                     .generalizedConstraintImpulses.size();
             ++index) {
            maximumPointLoopEqualityError = std::max(
                maximumPointLoopEqualityError,
                std::abs(
                    loopOracleSolution
                        .generalizedConstraintImpulses[
                            index
                        ] -
                    loopGPU.equalityImpulses[index]
                )
            );
        }
        const double maximumPointLoopError = std::max({
            maximumPointLoopVelocityError,
            maximumPointLoopContactError,
            maximumPointLoopEqualityError,
        });
        const std::array angularVelocityIndices{
            static_cast<std::size_t>(first.vOffset + 5u),
            static_cast<std::size_t>(second.vOffset + 5u),
            static_cast<std::size_t>(model.world.nv + 5u),
        };
        double maximumAngularConstraintError = 0.0;
        for (const std::size_t index :
             angularVelocityIndices) {
            maximumAngularConstraintError = std::max(
                maximumAngularConstraintError,
                std::abs(
                    static_cast<double>(
                        loopGPU.nextVelocity[index]
                    ) + 0.1
                )
            );
        }
        require(
            maximumPointLoopError < 2.5e-3 &&
                maximumAngularConstraintError < 2.5e-4 &&
                loopGPU.equalityStatuses[0]
                        .diagnostics.x <
                    config.equalityResidualTolerance,
            "Metal point-loop result disagrees with FP64 "
            "error=" +
                std::to_string(maximumPointLoopError) +
                " velocity=" +
                std::to_string(
                    maximumPointLoopVelocityError
                ) +
                " contact=" +
                std::to_string(
                    maximumPointLoopContactError
                ) +
                " equality=" +
                std::to_string(
                    maximumPointLoopEqualityError
                ) +
                " angular=" +
                std::to_string(
                    maximumAngularConstraintError
                ) +
                " residual=" +
                std::to_string(
                    loopGPU.equalityStatuses[0]
                        .diagnostics.x
                )
        );
        metalrobo::MetalMultiArticulatedContactResult
            loopReplay;
        const auto loopReplayDiagnostics =
            metalrobo::solveMetalMultiArticulatedContacts(
                program,
                loopInput,
                loopReplay,
                config
            );
        require(
            loopReplayDiagnostics.succeeded() &&
                bitwiseEqual(
                    loopGPU.nextVelocity,
                    loopReplay.nextVelocity
                ) &&
                bitwiseEqual(
                    loopGPU.impulses,
                    loopReplay.impulses
                ) &&
                bitwiseEqual(
                    loopGPU.equalityImpulses,
                    loopReplay.equalityImpulses
                ),
            "same-device Metal point-loop replay changed"
        );

        metalrobo::MetalMultiArticulatedContactResult replay;
        const auto replayDiagnostics =
            metalrobo::solveMetalMultiArticulatedContacts(
                program,
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
                program,
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
                program,
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
            << " program=" << program.fingerprint()
            << " impulse=" << gpu.impulses[0]
            << " max_velocity=" << maximumVelocity
            << " oracle_error=" << maximumOracleError
            << " spatial_loop_rows="
            << loopGPU.layout.dispatch.equalityRowCount
            << " spatial_loop_error="
            << maximumPointLoopError
            << " angular_target_error="
            << maximumAngularConstraintError
            << " spatial_loop_residual="
            << loopGPU.equalityStatuses[0].diagnostics.x
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
