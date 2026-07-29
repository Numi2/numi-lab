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
            << " deterministic=yes transactional=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "multi-articulated contact probe failed: "
            << error.what() << '\n';
        return 1;
    }
}
