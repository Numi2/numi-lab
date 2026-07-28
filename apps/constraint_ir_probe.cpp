#include "metalrobo/ConstraintIR.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
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

bool near(
    const double left,
    const double right,
    const double tolerance = 2.0e-6
) {
    return std::abs(left - right) <= tolerance;
}

MRContactConstraintGPU makeContact(
    const std::uint64_t pairKey,
    const std::uint64_t featureKey,
    const std::uint32_t bodyA,
    const std::uint32_t bodyB,
    const float separation,
    const bool newImpact,
    const bool torsion
) {
    MRContactConstraintGPU contact{};
    contact.bodyA = bodyA;
    contact.bodyB = bodyB;
    contact.flags = newImpact
        ? MR_CONSTRAINT_FLAG_NEW_IMPACT
        : 0u;
    contact.islandIndex = 0u;
    contact.pairKey = pairKey;
    contact.featureKey = featureKey;
    contact.pointAndSeparation =
        {0.25F, 0.50F, -0.125F, separation};
    contact.normal = {0.0F, 1.0F, 0.0F, 0.0F};
    contact.friction = {
        newImpact ? 0.80F : 0.70F,
        newImpact ? 0.50F : 0.40F,
        0.0F,
        torsion ? 0.02F : 0.0F,
    };
    contact.response = {
        newImpact ? 0.50F : 0.10F,
        0.20F,
        newImpact ? 2.0e-5F : 1.0e-5F,
        0.0F,
    };
    contact.targetVelocityAndPreSolveNormal = {
        0.20F,
        0.10F,
        -0.15F,
        newImpact ? -2.0F : -0.20F,
    };
    contact.impulses = {};
    return contact;
}

std::uint32_t findBlockByPairLowWord(
    const metalrobo::ConstraintIR& ir,
    const std::uint32_t pairLowWord
) {
    for (std::uint32_t index = 0u;
         index < ir.blocks.size();
         ++index) {
        if (ir.blocks[index].key.words[1] == pairLowWord) {
            return index;
        }
    }
    throw std::runtime_error("probe could not find adapted block");
}

void requireFailedAndEmpty(
    const metalrobo::ConstraintIRV1AdapterResult& result,
    const metalrobo::ConstraintIRStatus expected,
    const std::string& label
) {
    require(!result.succeeded(), label + " unexpectedly succeeded");
    require(
        result.diagnostics.status == expected,
        label + " returned an unexpected status"
    );
    require(
        result.ir.empty() && result.preSolveVelocities.empty(),
        label + " published a partial payload"
    );
}

} // namespace

int main() {
    try {
        // Deliberately reverse stable-key order. The adapter must publish one
        // sorted transaction, not preserve arbitrary input order.
        const MRContactConstraintGPU highKey = makeContact(
            2u,
            22u,
            2u,
            3u,
            -0.012F,
            true,
            true
        );
        const MRContactConstraintGPU lowKey = makeContact(
            1u,
            11u,
            0u,
            1u,
            -0.003F,
            false,
            false
        );
        const std::array<MRContactConstraintGPU, 2> contacts{
            highKey,
            lowKey,
        };

        metalrobo::ConstraintIRV1AdapterConfig adapterConfig;
        adapterConfig.timeConstant = 0.02F;
        adapterConfig.dampingRatio = 1.0F;
        adapterConfig.dissipation = 3.0e-4F;
        adapterConfig.stictionTransitionVelocity = 1.0e-3F;
        const auto adapted =
            metalrobo::adaptV1ContactsToConstraintIR(
                contacts,
                adapterConfig
            );
        require(
            adapted.succeeded(),
            "valid v1 contact adaptation failed: " +
                adapted.diagnostics.message
        );
        require(
            metalrobo::validateConstraintIR(adapted.ir).succeeded(),
            "adapted constraint IR failed validation"
        );
        require(
            adapted.ir.blocks.size() == 2u &&
            adapted.ir.endpoints.size() == 4u &&
            adapted.ir.rows.size() == 7u &&
            adapted.ir.cones.size() == 2u &&
            adapted.ir.warmImpulses.size() == 7u,
            "adapted stream counts are wrong"
        );
        require(
            metalrobo::constraintIRKeyLess(
                adapted.ir.blocks[0].key,
                adapted.ir.blocks[1].key
            ),
            "adapter did not canonicalize stable-key order"
        );

        const std::uint32_t lowBlockIndex =
            findBlockByPairLowWord(adapted.ir, 1u);
        const std::uint32_t highBlockIndex =
            findBlockByPairLowWord(adapted.ir, 2u);
        const auto& lowBlock = adapted.ir.blocks[lowBlockIndex];
        const auto& highBlock = adapted.ir.blocks[highBlockIndex];
        require(
            lowBlock.dimension == 3u &&
            highBlock.dimension == 4u,
            "adapter did not preserve torsional dimensionality"
        );

        std::vector<float> relative(adapted.ir.rows.size(), 0.0F);
        for (std::size_t row = 0u;
             row < adapted.ir.rows.size();
             ++row) {
            relative[row] = adapted.ir.rows[row].targetVelocity;
        }
        // Force the lower-key contact into dynamic slip. The new-impact
        // contact remains inside the static region.
        relative[lowBlock.rowOffset + 1u] += 0.25F;

        metalrobo::ConstraintIREvaluationConfig evaluationConfig;
        evaluationConfig.timestep = 0.01;
        evaluationConfig.penetrationSlop = 1.0e-4;
        evaluationConfig.maximumDepenetrationVelocity = 2.0;
        evaluationConfig.minimumTimeConstantRatio = 2.0;
        evaluationConfig.stictionTransitionVelocity = 1.0e-3;
        const auto evaluation = metalrobo::evaluateConstraintIR(
            adapted.ir,
            {
                relative,
                adapted.preSolveVelocities,
            },
            evaluationConfig
        );
        require(
            evaluation.succeeded(),
            "shared semantic evaluation failed: " +
                evaluation.diagnostics.message
        );

        const auto qualityView =
            metalrobo::makeConstraintIREvaluationView(
                evaluation.evaluated,
                metalrobo::ConstraintIRConsumer::quality
            );
        const auto throughputView =
            metalrobo::makeConstraintIREvaluationView(
                evaluation.evaluated,
                metalrobo::ConstraintIRConsumer::throughput
            );
        require(
            qualityView.rows.data() == throughputView.rows.data() &&
            qualityView.cones.data() ==
                throughputView.cones.data() &&
            qualityView.endpoints.data() ==
                throughputView.endpoints.data() &&
            qualityView.semanticFingerprint ==
                throughputView.semanticFingerprint &&
            qualityView.semanticFingerprint ==
                metalrobo::fingerprintConstraintSemantics(
                    evaluation.evaluated
                ),
            "quality and throughput did not share evaluated semantics"
        );

        const auto& lowCone =
            evaluation.evaluated.cones[lowBlock.coneIndex];
        const auto& highCone =
            evaluation.evaluated.cones[highBlock.coneIndex];
        require(
            near(lowCone.effectiveFrictionU, 0.40) &&
            near(highCone.effectiveFrictionU, 0.80),
            "shared friction region selection is incorrect"
        );

        // A cache can be feasible for static friction and infeasible after
        // slip selects a narrower dynamic cone. Evaluation must project the
        // cache before either consumer can observe it.
        metalrobo::ConstraintIR narrowingWarmIR = adapted.ir;
        auto& narrowingCone =
            narrowingWarmIR.cones[lowBlock.coneIndex];
        narrowingCone.staticFrictionU = 0.80F;
        narrowingCone.staticFrictionV = 0.80F;
        narrowingCone.dynamicFrictionU = 0.50F;
        narrowingCone.dynamicFrictionV = 0.50F;
        narrowingWarmIR.warmImpulses[lowBlock.impulseOffset] =
            1.0F;
        narrowingWarmIR.warmImpulses[
            lowBlock.impulseOffset + 1u
        ] = 0.70F;
        require(
            metalrobo::validateConstraintIR(
                narrowingWarmIR
            ).succeeded(),
            "static-feasible warm cache was rejected"
        );
        const auto narrowingEvaluation =
            metalrobo::evaluateConstraintIR(
                narrowingWarmIR,
                {
                    relative,
                    adapted.preSolveVelocities,
                },
                evaluationConfig
            );
        require(
            narrowingEvaluation.succeeded(),
            "dynamic-cone warm projection failed"
        );
        const auto narrowingQualityView =
            metalrobo::makeConstraintIREvaluationView(
                narrowingEvaluation.evaluated,
                metalrobo::ConstraintIRConsumer::quality
            );
        const auto narrowingThroughputView =
            metalrobo::makeConstraintIREvaluationView(
                narrowingEvaluation.evaluated,
                metalrobo::ConstraintIRConsumer::throughput
            );
        const double projectedWarmNormal =
            narrowingQualityView.warmImpulses[
                lowBlock.impulseOffset
            ];
        const double projectedWarmTangent =
            narrowingQualityView.warmImpulses[
                lowBlock.impulseOffset + 1u
            ];
        require(
            near(projectedWarmNormal, 1.20) &&
            near(projectedWarmTangent, 0.60) &&
            std::abs(projectedWarmTangent / 0.50) <=
                projectedWarmNormal + 2.0e-6 &&
            narrowingQualityView.warmImpulses.data() ==
                narrowingThroughputView.warmImpulses.data() &&
            narrowingQualityView.semanticFingerprint ==
                narrowingThroughputView.semanticFingerprint &&
            narrowingQualityView.semanticFingerprint ==
                metalrobo::fingerprintConstraintSemantics(
                    narrowingEvaluation.evaluated
                ) &&
            narrowingQualityView.semanticFingerprint !=
                qualityView.semanticFingerprint,
            "evaluated warm cache is not a shared feasible projection"
        );

        const auto& highNormal =
            evaluation.evaluated.rows[highBlock.rowOffset];
        const double sourceNormalTarget =
            adapted.ir.rows[highBlock.rowOffset].targetVelocity;
        const double incoming =
            adapted.preSolveVelocities[highBlock.rowOffset] -
            sourceNormalTarget;
        const double expectedBounce = -0.50 * incoming;
        require(
            near(highCone.restitutionVelocity, expectedBounce) &&
            near(
                highNormal.targetVelocity,
                sourceNormalTarget + expectedBounce
            ),
            "new-impact restitution semantics are incorrect"
        );
        const double expectedRegularization =
            2.0e-5 / (0.01 * 0.01) +
            3.0e-4 / 0.01;
        require(
            near(
                highNormal.regularization,
                expectedRegularization
            ),
            "compliance/dissipation discretization is incorrect"
        );

        // Zero impulses with separating normal velocity are an exact
        // complementarity solution for both contacts.
        std::vector<float> postVelocity(
            evaluation.evaluated.rows.size(),
            0.0F
        );
        for (std::size_t row = 0u;
             row < postVelocity.size();
             ++row) {
            postVelocity[row] =
                evaluation.evaluated.rows[row].targetVelocity;
        }
        postVelocity[lowBlock.rowOffset] += 0.25F;
        postVelocity[highBlock.rowOffset] += 0.25F;
        std::vector<float> impulses(
            evaluation.evaluated.rows.size(),
            0.0F
        );
        metalrobo::ConstraintIRResidualConfig residualConfig;
        residualConfig.projectionStep = 1.0;
        residualConfig.impulseTolerance = 1.0e-7;
        residualConfig.residualTolerance = 1.0e-6;
        const auto qualityResidual =
            metalrobo::evaluateConstraintIRResidual(
                qualityView,
                postVelocity,
                impulses,
                residualConfig
            );
        const auto throughputResidual =
            metalrobo::evaluateConstraintIRResidual(
                throughputView,
                postVelocity,
                impulses,
                residualConfig
            );
        require(
            qualityResidual.succeeded() &&
            throughputResidual.succeeded() &&
            qualityResidual.withinTolerance(residualConfig) &&
            throughputResidual.withinTolerance(residualConfig) &&
            qualityResidual.maximumNaturalResidual ==
                throughputResidual.maximumNaturalResidual &&
            qualityResidual.maximumPrimalViolation ==
                throughputResidual.maximumPrimalViolation,
            "shared residual evaluator disagrees across consumers"
        );

        std::vector<float> badImpulse = impulses;
        badImpulse[highBlock.impulseOffset] = 0.10F;
        badImpulse[highBlock.impulseOffset + 1u] = 0.20F;
        const auto coneViolation =
            metalrobo::evaluateConstraintIRResidual(
                qualityView,
                postVelocity,
                badImpulse,
                residualConfig
            );
        require(
            coneViolation.succeeded() &&
            coneViolation.maximumPrimalViolation > 0.10 &&
            coneViolation.maximumNaturalResidual > 0.0 &&
            !coneViolation.withinTolerance(residualConfig),
            "residual evaluator missed an elliptic-cone violation"
        );

        // Analytic coupled-patch projection. In scaled coordinates the
        // projected candidate is (n, t, r) = (0, 2, 2). The closest point
        // satisfying |t| <= n and |r| <= n is
        // (4/3, 4/3, 4/3), so the natural residual is sqrt(16/3).
        const double torsionScale =
            std::sqrt(
                static_cast<double>(highCone.effectiveFrictionU) *
                highCone.effectiveFrictionV
            ) * highCone.torsionalLength;
        require(
            torsionScale > 0.0,
            "probe torsional patch unexpectedly degenerated"
        );
        std::vector<float> coupledVelocity = postVelocity;
        coupledVelocity[highBlock.rowOffset] =
            highNormal.targetVelocity;
        coupledVelocity[highBlock.rowOffset + 1u] =
            evaluation.evaluated.rows[
                highBlock.rowOffset + 1u
            ].targetVelocity -
            static_cast<float>(
                2.0 / highCone.effectiveFrictionU
            );
        coupledVelocity[highBlock.rowOffset + 2u] =
            evaluation.evaluated.rows[
                highBlock.rowOffset + 2u
            ].targetVelocity;
        coupledVelocity[highBlock.rowOffset + 3u] =
            evaluation.evaluated.rows[
                highBlock.rowOffset + 3u
            ].targetVelocity -
            static_cast<float>(2.0 / torsionScale);
        const auto coupledTorsionResidual =
            metalrobo::evaluateConstraintIRResidual(
                qualityView,
                coupledVelocity,
                impulses,
                residualConfig
            );
        const double expectedCoupledResidual =
            std::sqrt(16.0 / 3.0);
        require(
            coupledTorsionResidual.succeeded() &&
            coupledTorsionResidual.coupledTorsionContacts == 1u &&
            near(
                coupledTorsionResidual.maximumNaturalResidual,
                expectedCoupledResidual,
                2.0e-4
            ),
            "normal/tangent/torsion projection was not jointly solved"
        );

        std::uint32_t adversarialChecks = 0u;
        {
            std::vector<float> tinyStepVelocity = postVelocity;
            tinyStepVelocity[highBlock.rowOffset] =
                highNormal.targetVelocity - 1.0F;
            metalrobo::ConstraintIRResidualConfig tinyStepConfig =
                residualConfig;
            tinyStepConfig.projectionStep = 1.0e-6;
            tinyStepConfig.residualTolerance = 1.0e-5;
            const auto tinyStepResidual =
                metalrobo::evaluateConstraintIRResidual(
                    qualityView,
                    tinyStepVelocity,
                    impulses,
                    tinyStepConfig
                );
            require(
                tinyStepResidual.succeeded() &&
                near(
                    tinyStepResidual.maximumNaturalResidual,
                    1.0,
                    2.0e-6
                ) &&
                !tinyStepResidual.withinTolerance(tinyStepConfig),
                "tiny natural-map step produced false convergence"
            );
            ++adversarialChecks;
        }
        {
            metalrobo::EvaluatedConstraintIR malformed =
                evaluation.evaluated;
            malformed.blocks[0].dimension = 7u;
            const auto malformedView =
                metalrobo::makeConstraintIREvaluationView(
                    malformed,
                    metalrobo::ConstraintIRConsumer::quality
                );
            const auto diagnostic =
                metalrobo::evaluateConstraintIRResidual(
                    malformedView,
                    postVelocity,
                    impulses,
                    residualConfig
                );
            require(
                diagnostic.status ==
                    metalrobo::ConstraintIRStatus::
                        invalidResidualInput,
                "forged residual view reached unsafe block indexing"
            );
            ++adversarialChecks;
        }
        {
            auto staleView = qualityView;
            staleView.semanticFingerprint ^= 1u;
            const auto diagnostic =
                metalrobo::evaluateConstraintIRResidual(
                    staleView,
                    postVelocity,
                    impulses,
                    residualConfig
                );
            require(
                diagnostic.status ==
                    metalrobo::ConstraintIRStatus::
                        invalidResidualInput,
                "stale evaluated fingerprint was accepted"
            );
            ++adversarialChecks;
        }
        {
            metalrobo::ConstraintIR malformed = adapted.ir;
            malformed.blocks[1].key = malformed.blocks[0].key;
            const auto diagnostic =
                metalrobo::validateConstraintIR(malformed);
            require(
                diagnostic.status ==
                    metalrobo::ConstraintIRStatus::
                        nonCanonicalOrder,
                "duplicate stable key was accepted"
            );
            ++adversarialChecks;
        }
        {
            metalrobo::ConstraintIR malformed = adapted.ir;
            malformed.rows[
                lowBlock.rowOffset
            ].impulseUpper = 0.50F;
            require(
                metalrobo::validateConstraintIR(malformed).status ==
                    metalrobo::ConstraintIRStatus::invalidRow,
                "normal row cap was allowed to contradict its cone"
            );
            ++adversarialChecks;
        }
        {
            metalrobo::ConstraintIR malformed = adapted.ir;
            malformed.rows[
                lowBlock.rowOffset + 1u
            ].impulseUpper = 0.50F;
            require(
                metalrobo::validateConstraintIR(malformed).status ==
                    metalrobo::ConstraintIRStatus::invalidRow,
                "finite contact tangent bound was accepted"
            );
            ++adversarialChecks;
        }
        {
            metalrobo::ConstraintIR malformed = adapted.ir;
            malformed.rows[
                highBlock.rowOffset + 3u
            ].impulseLower = -0.50F;
            require(
                metalrobo::validateConstraintIR(malformed).status ==
                    metalrobo::ConstraintIRStatus::invalidRow,
                "finite contact torsion bound was accepted"
            );
            ++adversarialChecks;
        }
        {
            metalrobo::ConstraintIR malformed = adapted.ir;
            ++malformed.blocks[0].rowOffset;
            require(
                metalrobo::validateConstraintIR(malformed).status ==
                    metalrobo::ConstraintIRStatus::invalidRange,
                "overlapping/gapped row range was accepted"
            );
            ++adversarialChecks;
        }
        {
            metalrobo::ConstraintIR malformed = adapted.ir;
            malformed.rows[0].compliance =
                std::numeric_limits<float>::quiet_NaN();
            require(
                metalrobo::validateConstraintIR(malformed).status ==
                    metalrobo::ConstraintIRStatus::nonfiniteData,
                "NaN row data was accepted"
            );
            ++adversarialChecks;
        }
        {
            metalrobo::ConstraintIR malformed = adapted.ir;
            malformed.cones[0].dynamicFrictionU =
                malformed.cones[0].staticFrictionU + 0.1F;
            require(
                metalrobo::validateConstraintIR(malformed).status ==
                    metalrobo::ConstraintIRStatus::invalidCone,
                "dynamic friction above static friction was accepted"
            );
            ++adversarialChecks;
        }
        {
            metalrobo::ConstraintIR malformed = adapted.ir;
            malformed.warmImpulses[lowBlock.impulseOffset] = 0.1F;
            malformed.warmImpulses[
                lowBlock.impulseOffset + 1u
            ] = 0.2F;
            require(
                metalrobo::validateConstraintIR(malformed).status ==
                    metalrobo::ConstraintIRStatus::
                        infeasibleWarmStart,
                "out-of-cone warm start was accepted"
            );
            ++adversarialChecks;
        }
        {
            metalrobo::ConstraintIR unsupported = adapted.ir;
            unsupported.cones[0].adhesionImpulse = 0.01F;
            require(
                metalrobo::validateConstraintIR(unsupported).status ==
                    metalrobo::ConstraintIRStatus::
                        unsupportedSemantics,
                "unimplemented shifted adhesion cone was hidden"
            );
            ++adversarialChecks;
        }
        {
            const std::array<MRContactConstraintGPU, 2> duplicate{
                lowKey,
                lowKey,
            };
            requireFailedAndEmpty(
                metalrobo::adaptV1ContactsToConstraintIR(
                    duplicate,
                    adapterConfig
                ),
                metalrobo::ConstraintIRStatus::nonCanonicalOrder,
                "duplicate-key adapter transaction"
            );
            ++adversarialChecks;
        }
        {
            MRContactConstraintGPU rolling = lowKey;
            rolling.friction.z = 0.01F;
            const std::array<MRContactConstraintGPU, 1> stream{
                rolling,
            };
            requireFailedAndEmpty(
                metalrobo::adaptV1ContactsToConstraintIR(
                    stream,
                    adapterConfig
                ),
                metalrobo::ConstraintIRStatus::
                    unsupportedSemantics,
                "rolling adapter transaction"
            );
            ++adversarialChecks;
        }
        {
            auto invalidConfig = evaluationConfig;
            invalidConfig.timestep = 0.0;
            const auto failed = metalrobo::evaluateConstraintIR(
                adapted.ir,
                {
                    relative,
                    adapted.preSolveVelocities,
                },
                invalidConfig
            );
            require(
                !failed.succeeded() && failed.evaluated.empty() &&
                failed.diagnostics.status ==
                    metalrobo::ConstraintIRStatus::
                        invalidEvaluationConfig,
                "invalid evaluation published a partial payload"
            );
            ++adversarialChecks;
        }
        {
            std::vector<float> nonfinite = postVelocity;
            nonfinite[0] =
                std::numeric_limits<float>::infinity();
            const auto failed =
                metalrobo::evaluateConstraintIRResidual(
                    qualityView,
                    nonfinite,
                    impulses,
                    residualConfig
                );
            require(
                failed.status ==
                    metalrobo::ConstraintIRStatus::
                        invalidResidualInput,
                "non-finite residual input was accepted"
            );
            ++adversarialChecks;
        }

        // The adversarial copies above must not mutate the accepted source.
        require(
            metalrobo::validateConstraintIR(adapted.ir).succeeded(),
            "failed validation attempt mutated the source IR"
        );

        std::cout << std::setprecision(10)
                  << "constraint_ir=abi_v2"
                  << " blocks=" << adapted.ir.blocks.size()
                  << " rows=" << adapted.ir.rows.size()
                  << " endpoints=" << adapted.ir.endpoints.size()
                  << " cones=" << adapted.ir.cones.size()
                  << " fingerprint="
                  << qualityView.semanticFingerprint
                  << " shared_buffers=yes"
                  << " restitution_target="
                  << highNormal.targetVelocity
                  << " regularization="
                  << highNormal.regularization
                  << " equilibrium_residual="
                  << qualityResidual.maximumNaturalResidual
                  << " adversarial_cone_violation="
                  << coneViolation.maximumPrimalViolation
                  << " coupled_torsion_residual="
                  << coupledTorsionResidual.maximumNaturalResidual
                  << " projected_warm_normal="
                  << projectedWarmNormal
                  << " projected_warm_tangent="
                  << projectedWarmTangent
                  << " adversarial_checks="
                  << adversarialChecks
                  << " transactional=yes"
                  << " status=ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "constraint IR probe failed: "
                  << error.what() << '\n';
        return 1;
    }
}
