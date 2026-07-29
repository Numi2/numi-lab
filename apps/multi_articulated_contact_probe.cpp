#include "metalrobo/EngineModelComposer.hpp"
#include "metalrobo/MultiArticulatedContact.hpp"

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

bool bitwiseEqual(
    const std::vector<double>& left,
    const std::vector<double>& right
) {
    return left.size() == right.size() &&
        (left.empty() ||
         std::memcmp(
             left.data(),
             right.data(),
             left.size() * sizeof(double)
         ) == 0);
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
    metalrobo::EngineModel model;
    metalrobo::EngineModelComposeConfig config;
    config.name = "two_sphere_contact_oracle";
    config.gravityAndTimestep = {
        0.0F, 0.0F, 0.0F, 1.0F / 1000.0F,
    };
    const auto composed =
        metalrobo::composeEngineModels(
            components,
            model,
            config
        );
    require(
        composed.succeeded(),
        "sphere composition failed: " + composed.message
    );
    return model;
}

} // namespace

int main() {
    try {
        const metalrobo::EngineModel model =
            makeTwoSpheres();
        require(
            model.articulations.size() == 2u,
            "expected two articulations"
        );
        const MRArticulationGPU& first =
            model.articulations[0];
        const MRArticulationGPU& second =
            model.articulations[1];
        std::vector<double> q(
            model.defaultQ.begin(),
            model.defaultQ.end()
        );
        q[first.qOffset + 0u] = -0.25;
        q[first.qOffset + 1u] = 1.0;
        q[second.qOffset + 0u] = 0.25;
        q[second.qOffset + 1u] = 1.0;

        std::vector<double> freeVelocity(
            model.world.nv,
            0.0
        );
        freeVelocity[first.vOffset + 0u] = 1.0;
        freeVelocity[second.vOffset + 0u] = -1.0;

        metalrobo::ArticulatedContact contact;
        contact.bodyA = first.rootBody;
        contact.bodyB = second.rootBody;
        contact.localPointA = {0.25, 0.0, 0.0};
        contact.localPointB = {-0.25, 0.0, 0.0};
        contact.normal = {1.0, 0.0, 0.0};
        contact.tangentU = {0.0, 1.0, 0.0};
        contact.tangentV = {0.0, 0.0, 1.0};
        contact.regularization = {
            1.0e-12,
            1.0e-12,
            1.0e-12,
        };
        contact.friction = 0.7;

        metalrobo::ArticulatedDynamicsConfig dynamics;
        dynamics.gravity = {0.0, 0.0, 0.0};
        dynamics.applyBodyDamping = false;

        metalrobo::MultiArticulatedContactProblem problem;
        const auto built =
            metalrobo::buildMultiArticulatedContactProblem(
                model,
                q,
                freeVelocity,
                std::span<const metalrobo::ArticulatedContact>(
                    &contact,
                    1u
                ),
                problem,
                dynamics
            );
        require(
            built.succeeded(),
            std::string("contact build failed: ") +
                metalrobo::multiArticulatedContactStatusName(
                    built.status
                )
        );
        require(
            problem.factors.factors.size() == 2u &&
            problem.contactCount == 1u &&
            problem.nv == 12u &&
            std::abs(
                problem.conic.freeContactVelocity[0] +
                2.0
            ) < 1.0e-12 &&
            std::abs(problem.conic.delassus[0] - 2.0) <
                1.0e-12,
            "global contact operator is incorrect"
        );

        metalrobo::QualityContactSolverConfig quality;
        quality.kktTolerance = 1.0e-12;
        metalrobo::MultiArticulatedContactSolution solved;
        const auto solve =
            metalrobo::solveMultiArticulatedContactProblem(
                problem,
                solved,
                quality
            );
        require(
            solve.succeeded(),
            std::string("contact solve failed: ") +
                metalrobo::multiArticulatedContactStatusName(
                    solve.status
                ) + " / " + solved.quality.failure
        );
        require(
            std::abs(solved.impulses[0] - 1.0) <
                2.0e-9 &&
            std::abs(
                solved.generalizedVelocity[
                    first.vOffset + 0u
                ]
            ) < 2.0e-9 &&
            std::abs(
                solved.generalizedVelocity[
                    second.vOffset + 0u
                ]
            ) < 2.0e-9 &&
            solve.maximumContactVelocityResidual <
                1.0e-12,
            "exact-cone impulse did not stop equal masses"
        );

        metalrobo::MultiArticulatedContactSolution replay;
        const auto replayDiagnostics =
            metalrobo::solveMultiArticulatedContactProblem(
                problem,
                replay,
                quality
            );
        require(
            replayDiagnostics.succeeded() &&
            bitwiseEqual(
                solved.generalizedVelocity,
                replay.generalizedVelocity
            ) &&
            bitwiseEqual(solved.impulses, replay.impulses),
            "multi-articulation contact replay changed"
        );

        metalrobo::ArticulatedContact staticContact =
            contact;
        staticContact.bodyB =
            metalrobo::kArticulatedStaticWorld;
        staticContact.localPointA = {0.0, -0.25, 0.0};
        staticContact.localPointB = {-0.25, 0.75, 0.0};
        staticContact.normal = {0.0, -1.0, 0.0};
        staticContact.tangentU = {1.0, 0.0, 0.0};
        staticContact.tangentV = {0.0, 0.0, 1.0};
        std::ranges::fill(freeVelocity, 0.0);
        freeVelocity[first.vOffset + 1u] = -1.0;
        metalrobo::MultiArticulatedContactProblem
            staticProblem;
        const auto staticBuilt =
            metalrobo::buildMultiArticulatedContactProblem(
                model,
                q,
                freeVelocity,
                std::span<const metalrobo::ArticulatedContact>(
                    &staticContact,
                    1u
                ),
                staticProblem,
                dynamics
            );
        require(
            staticBuilt.succeeded() &&
            std::abs(
                staticProblem.conic
                    .freeContactVelocity[0] +
                1.0
            ) < 1.0e-12 &&
            std::abs(
                staticProblem.conic.delassus[0] -
                1.0
            ) < 1.0e-12,
            "articulation-static boundary is incorrect"
        );

        MRBodyStateGPU sceneBody{};
        sceneBody.position = {
            0.0F, 1.0F, 0.0F, 1.0F,
        };
        sceneBody.orientation = {
            0.0F, 0.0F, 0.0F, 1.0F,
        };
        sceneBody.linearVelocityAndInverseMass = {
            0.0F, 0.0F, 0.0F, 1.0F,
        };
        sceneBody.inverseInertiaWorldRow0 = {
            1.0F, 0.0F, 0.0F, 0.0F,
        };
        sceneBody.inverseInertiaWorldRow1 = {
            0.0F, 1.0F, 0.0F, 0.0F,
        };
        sceneBody.inverseInertiaWorldRow2 = {
            0.0F, 0.0F, 1.0F, 0.0F,
        };
        sceneBody.flagsAndIndices[0] = MR_MOTION_DYNAMIC;
        sceneBody.flagsAndIndices[1] = MR_INVALID_INDEX;
        sceneBody.flagsAndIndices[2] = MR_INVALID_INDEX;
        const float halfRootTwo =
            static_cast<float>(std::sqrt(0.5));
        MRBodyStateGPU orientationScene = sceneBody;
        orientationScene.orientation = {
            0.0F, 0.0F, halfRootTwo, halfRootTwo,
        };
        metalrobo::MultiArticulatedAngularEquality
            authoredSceneOrientation;
        authoredSceneOrientation.endpointA = {
            metalrobo::MultiContactEndpointKind::staticWorld,
            MR_INVALID_INDEX,
            {},
        };
        authoredSceneOrientation.endpointB = {
            metalrobo::MultiContactEndpointKind::sceneBody,
            0u,
            {},
        };
        const double inverseRootTwo = std::sqrt(0.5);
        authoredSceneOrientation.axisX = {
            inverseRootTwo,
            inverseRootTwo,
            0.0,
        };
        authoredSceneOrientation.axisY = {
            -inverseRootTwo,
            inverseRootTwo,
            0.0,
        };
        authoredSceneOrientation.axisZ = {0.0, 0.0, 1.0};
        auto authoredRelativeOrientation =
            authoredSceneOrientation;
        authoredRelativeOrientation.endpointA = {
            metalrobo::MultiContactEndpointKind::
                articulatedBody,
            first.rootBody,
            {},
        };
        std::array authoredOrientations{
            authoredSceneOrientation,
            authoredRelativeOrientation,
        };
        const std::array<std::array<double, 4>, 2>
            desiredOrientations{{
                {0.0, 0.0, 0.0, 1.0},
                {
                    0.0,
                    0.0,
                    inverseRootTwo,
                    inverseRootTwo,
                },
            }};
        const auto authoredOrientationDiagnostics =
            metalrobo::
                authorMultiArticulatedAngularOrientationErrors(
                    model,
                    q,
                    std::span<const MRBodyStateGPU>(
                        &orientationScene,
                        1u
                    ),
                    desiredOrientations,
                    authoredOrientations,
                    dynamics
                );
        require(
            authoredOrientationDiagnostics.succeeded() &&
                std::abs(
                    authoredOrientations[0].
                        orientationError[0]
                ) < 1.0e-12 &&
                std::abs(
                    authoredOrientations[0].
                        orientationError[1]
                ) < 1.0e-12 &&
                std::abs(
                    authoredOrientations[0].
                        orientationError[2] -
                    0.5 * std::acos(-1.0)
                ) < 1.0e-7 &&
                std::abs(
                    authoredOrientations[1].
                        orientationError[2]
                ) < 1.0e-7,
            "quaternion angular-error authoring used the wrong "
            "relative frame or axes"
        );
        MRBodyStateGPU antipodalScene = orientationScene;
        antipodalScene.orientation = {
            -orientationScene.orientation.x,
            -orientationScene.orientation.y,
            -orientationScene.orientation.z,
            -orientationScene.orientation.w,
        };
        auto antipodalOrientations = authoredOrientations;
        const auto antipodalDiagnostics =
            metalrobo::
                authorMultiArticulatedAngularOrientationErrors(
                    model,
                    q,
                    std::span<const MRBodyStateGPU>(
                        &antipodalScene,
                        1u
                    ),
                    desiredOrientations,
                    antipodalOrientations,
                    dynamics
                );
        require(
            antipodalDiagnostics.succeeded() &&
                antipodalOrientations[0].orientationError ==
                    authoredOrientations[0].orientationError,
            "antipodal quaternion changed the angular error"
        );
        MRBodyStateGPU piScene = sceneBody;
        piScene.orientation = {
            0.0F, 0.0F, -1.0F, 0.0F,
        };
        std::array piEquality{authoredSceneOrientation};
        const std::array<std::array<double, 4>, 1>
            identityRelative{{
                {0.0, 0.0, 0.0, 1.0},
            }};
        const auto piDiagnostics =
            metalrobo::
                authorMultiArticulatedAngularOrientationErrors(
                    model,
                    q,
                    std::span<const MRBodyStateGPU>(
                        &piScene,
                        1u
                    ),
                    identityRelative,
                    piEquality,
                    dynamics
                );
        require(
            piDiagnostics.succeeded() &&
                std::abs(
                    piEquality[0].orientationError[2] -
                    std::acos(-1.0)
                ) < 1.0e-12,
            "pi-angle quaternion sign tie is nondeterministic"
        );
        auto rejectedOrientations = authoredOrientations;
        const auto acceptedOrientationError =
            rejectedOrientations[0].orientationError;
        auto invalidDesired = desiredOrientations;
        invalidDesired[0] = {};
        const auto rejectedOrientationDiagnostics =
            metalrobo::
                authorMultiArticulatedAngularOrientationErrors(
                    model,
                    q,
                    std::span<const MRBodyStateGPU>(
                        &orientationScene,
                        1u
                    ),
                    invalidDesired,
                    rejectedOrientations,
                    dynamics
                );
        require(
            !rejectedOrientationDiagnostics.succeeded() &&
                rejectedOrientations[0].orientationError ==
                    acceptedOrientationError,
            "failed angular-error authoring mutated the "
            "accepted equality"
        );

        std::ranges::fill(freeVelocity, 0.0);
        freeVelocity[first.vOffset + 0u] = 1.0;
        freeVelocity[second.vOffset + 0u] = -1.0;
        metalrobo::MultiArticulatedIslandContact left;
        left.endpointA = {
            metalrobo::MultiContactEndpointKind::
                articulatedBody,
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
            1.0e-12,
            1.0e-12,
            1.0e-12,
        };
        metalrobo::MultiArticulatedIslandContact right =
            left;
        right.endpointA = {
            metalrobo::MultiContactEndpointKind::sceneBody,
            0u,
            {0.25, 0.0, 0.0},
        };
        right.endpointB = {
            metalrobo::MultiContactEndpointKind::
                articulatedBody,
            second.rootBody,
            {-0.25, 0.0, 0.0},
        };
        const std::array<
            metalrobo::MultiArticulatedIslandContact,
            2
        > heterogeneousContacts{left, right};
        metalrobo::MultiArticulatedContactProblem
            heterogeneousProblem;
        const auto heterogeneousBuilt =
            metalrobo::
                buildMultiArticulatedIslandContactProblem(
                    model,
                    q,
                    freeVelocity,
                    std::span<const MRBodyStateGPU>(
                        &sceneBody,
                        1u
                    ),
                    heterogeneousContacts,
                    heterogeneousProblem,
                    dynamics
                );
        require(
            heterogeneousBuilt.succeeded() &&
            heterogeneousProblem.nv == 18u &&
            heterogeneousProblem.articulatedNv == 12u &&
            heterogeneousProblem
                    .sceneBodyVelocityOffsets[0] == 12u &&
            std::abs(
                heterogeneousProblem.conic.delassus[
                    0u * 6u + 0u
                ] - 2.0
            ) < 1.0e-12 &&
            std::abs(
                heterogeneousProblem.conic.delassus[
                    0u * 6u + 3u
                ] + 1.0
            ) < 1.0e-12 &&
            std::abs(
                heterogeneousProblem.conic.delassus[
                    3u * 6u + 3u
                ] - 2.0
            ) < 1.0e-12,
            "two-articulation/scene-body operator is incorrect"
        );
        metalrobo::MultiArticulatedContactSolution
            heterogeneousSolved;
        const auto heterogeneousSolve =
            metalrobo::solveMultiArticulatedContactProblem(
                heterogeneousProblem,
                heterogeneousSolved,
                quality
            );
        require(
            heterogeneousSolve.succeeded() &&
            heterogeneousSolved.impulses.size() == 6u &&
            std::abs(heterogeneousSolved.impulses[0] - 1.0) <
                3.0e-9 &&
            std::abs(heterogeneousSolved.impulses[3] - 1.0) <
                3.0e-9 &&
            std::abs(
                heterogeneousSolved.articulatedVelocity[
                    first.vOffset
                ]
            ) < 3.0e-9 &&
            std::abs(
                heterogeneousSolved.articulatedVelocity[
                    second.vOffset
                ]
            ) < 3.0e-9 &&
            std::abs(
                heterogeneousSolved.sceneBodyVelocities[0]
                    .linear[0]
            ) < 3.0e-9 &&
            heterogeneousSolve
                    .maximumContactVelocityResidual <
                1.0e-12,
            "two robots and dynamic scene body did not share "
            "one exact-cone island"
        );

        MRBodyStateGPU kinematicBody = sceneBody;
        kinematicBody.flagsAndIndices[0] =
            MR_MOTION_KINEMATIC;
        kinematicBody.linearVelocityAndInverseMass = {
            0.25F, 0.0F, 0.0F, 0.0F,
        };
        kinematicBody.inverseInertiaWorldRow0 = {};
        kinematicBody.inverseInertiaWorldRow1 = {};
        kinematicBody.inverseInertiaWorldRow2 = {};
        std::ranges::fill(freeVelocity, 0.0);
        metalrobo::MultiArticulatedContactProblem
            kinematicProblem;
        const auto kinematicBuilt =
            metalrobo::
                buildMultiArticulatedIslandContactProblem(
                    model,
                    q,
                    freeVelocity,
                    std::span<const MRBodyStateGPU>(
                        &kinematicBody,
                        1u
                    ),
                    std::span<
                        const metalrobo::
                            MultiArticulatedIslandContact
                    >(&left, 1u),
                    kinematicProblem,
                    dynamics
                );
        metalrobo::MultiArticulatedContactSolution
            kinematicSolved;
        const auto kinematicSolve =
            metalrobo::solveMultiArticulatedContactProblem(
                kinematicProblem,
                kinematicSolved,
                quality
            );
        require(
            kinematicBuilt.succeeded() &&
            kinematicSolve.succeeded() &&
            kinematicProblem.nv == model.world.nv &&
            kinematicProblem.sceneBodyVelocityOffsets[0] ==
                MR_INVALID_INDEX &&
            std::abs(
                kinematicProblem
                    .prescribedContactVelocity[0] - 0.25
            ) < 1.0e-12 &&
            std::abs(kinematicSolved.impulses[0]) <
                1.0e-12 &&
            std::abs(
                kinematicSolved.sceneBodyVelocities[0]
                    .linear[0] - 0.25
            ) < 1.0e-12,
            "kinematic scene endpoint was not preserved as "
            "prescribed velocity"
        );

        std::ranges::fill(freeVelocity, 0.0);
        freeVelocity[first.vOffset] = 1.0;
        metalrobo::MultiArticulatedIslandContact
            loopContact;
        loopContact.endpointA = {
            metalrobo::MultiContactEndpointKind::
                articulatedBody,
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
            1.0e-12,
            1.0e-12,
            1.0e-12,
        };
        metalrobo::MultiArticulatedContactProblem loopProblem;
        const auto loopBuilt =
            metalrobo::
                buildMultiArticulatedIslandContactProblem(
                    model,
                    q,
                    freeVelocity,
                    {},
                    std::span<
                        const metalrobo::
                            MultiArticulatedIslandContact
                    >(&loopContact, 1u),
                    loopProblem,
                    dynamics
                );
        metalrobo::MultiArticulatedPointEquality loop;
        loop.key.words[0] = 0x4c4f4f50u;
        loop.key.words[1] = 1u;
        loop.endpointA = {
            metalrobo::MultiContactEndpointKind::
                articulatedBody,
            first.rootBody,
            {0.0, 0.0, 0.0},
        };
        loop.endpointB = {
            metalrobo::MultiContactEndpointKind::
                articulatedBody,
            second.rootBody,
            {0.0, 0.0, 0.0},
        };
        loop.compliance = {
            1.0e-12,
            1.0e-12,
            1.0e-12,
        };
        loop.positionStabilized = false;
        metalrobo::ConstraintIREvaluationConfig
            loopEvaluation;
        loopEvaluation.timestep = 1.0e-3;
        const auto loopProjected =
            metalrobo::
                projectMultiArticulatedContactThroughPointEqualities(
                    model,
                    q,
                    loopProblem,
                    std::span<
                        const metalrobo::
                            MultiArticulatedPointEquality
                    >(&loop, 1u),
                    loopEvaluation,
                    dynamics
                );
        metalrobo::MultiArticulatedContactSolution
            loopSolved;
        const auto loopSolve =
            metalrobo::solveMultiArticulatedContactProblem(
                loopProblem,
                loopSolved,
                quality
            );
        require(
            loopBuilt.succeeded() &&
                loopProjected.succeeded() &&
                loopSolve.succeeded() &&
                loopProblem.generalizedConstraintRowCount ==
                    3u &&
                loopSolved.generalizedConstraintImpulses
                        .size() == 3u &&
                loopSolved.impulses[0] > 0.0 &&
                std::abs(
                    loopSolved.generalizedVelocity[
                        first.vOffset
                    ]
                ) < 3.0e-9 &&
                std::abs(
                    loopSolved.generalizedVelocity[
                        second.vOffset
                    ]
                ) < 3.0e-9 &&
                loopSolve
                        .maximumGeneralizedConstraintResidual <
                    1.0e-10,
            "three-axis inter-articulation point loop did not "
            "share the contact Schur operator build=" +
                std::to_string(loopBuilt.succeeded()) +
                " project=" +
                std::string(
                    metalrobo::
                        multiArticulatedContactStatusName(
                            loopProjected.status
                        )
                ) +
                " solve=" +
                std::string(
                    metalrobo::
                        multiArticulatedContactStatusName(
                            loopSolve.status
                        )
                ) +
                " rows=" +
                std::to_string(
                    loopProblem
                        .generalizedConstraintRowCount
                ) +
                " residual=" +
                std::to_string(
                    loopSolve
                        .maximumGeneralizedConstraintResidual
                ) +
                " impulse=" +
                (
                    loopSolved.impulses.empty()
                    ? std::string{"empty"}
                    : std::to_string(
                          loopSolved.impulses[0]
                      )
                ) +
                " velocity=" +
                std::to_string(
                    loopSolved.generalizedVelocity[
                        first.vOffset
                    ]
                ) +
                "/" +
                std::to_string(
                    loopSolved.generalizedVelocity[
                        second.vOffset
                    ]
                ) +
                " equality_impulse=" +
                std::to_string(
                    loopSolved
                        .generalizedConstraintImpulses[0]
                ) +
                " coupling=" +
                std::to_string(
                    loopProblem
                        .generalizedConstraintContactCoupling[
                            0
                        ]
                ) +
                " G=" +
                std::to_string(
                    loopProblem
                        .generalizedConstraintJacobian[
                            first.vOffset
                        ]
                ) +
                "/" +
                std::to_string(
                    loopProblem
                        .generalizedConstraintJacobian[
                            second.vOffset
                        ]
                )
        );

        const std::vector<double> acceptedJacobian =
            problem.contactJacobian;
        const std::uint32_t acceptedCount =
            problem.contactCount;
        contact.tangentV = {0.0, 0.0, -1.0};
        const auto rejected =
            metalrobo::buildMultiArticulatedContactProblem(
                model,
                q,
                freeVelocity,
                std::span<const metalrobo::ArticulatedContact>(
                    &contact,
                    1u
                ),
                problem,
                dynamics
            );
        require(
            !rejected.succeeded() &&
            rejected.status ==
                metalrobo::MultiArticulatedContactStatus::
                    invalidContact &&
            problem.contactCount == acceptedCount &&
            bitwiseEqual(
                problem.contactJacobian,
                acceptedJacobian
            ),
            "failed contact build mutated accepted operator"
        );

        std::cout
            << "multi_articulated_contact=ok"
            << " articulations=" << built.articulationCount
            << " rows=" << built.rowCount
            << " normal_impulse=" << solved.impulses[0]
            << " delassus=" << problem.conic.delassus[0]
            << " asymmetry="
            << built.maximumDelassusAsymmetry
            << " factor_residual="
            << built.maximumFactorResidual
            << " heterogeneous_nv="
            << heterogeneousProblem.nv
            << " scene_impulses="
            << heterogeneousSolved.impulses[0] << "/"
            << heterogeneousSolved.impulses[3]
            << " solve_residual="
            << solved.quality.scaledKktCertificate
            << " loop_rows="
            << loopProblem.generalizedConstraintRowCount
            << " loop_residual="
            << loopSolve
                   .maximumGeneralizedConstraintResidual
            << " authored_angle_error="
            << authoredOrientations[0].orientationError[2]
            << " deterministic=yes transactional=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "multi-articulated contact probe failed: "
            << error.what() << '\n';
        return 1;
    }
}
