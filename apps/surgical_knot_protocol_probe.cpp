#include "metalrobo/SurgicalKnot.hpp"
#include "metalrobo/SurgicalPSM.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
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

} // namespace

int main() {
    try {
        const auto protocol =
            metalrobo::makeSurgeonsKnotInstrumentProtocol();
        const auto diagnostics =
            metalrobo::certifySurgeonsKnotInstrumentProtocol(protocol);
        require(
            diagnostics.succeeded(),
            "canonical protocol rejected: " + std::string(
                metalrobo::surgeonsKnotProtocolStatusName(
                    diagnostics.status
                )
            ) + " first=" +
                metalrobo::surgicalThrowStatusName(
                    diagnostics.firstDoubleThrow.status
                ) + " second=" +
                metalrobo::surgicalThrowStatusName(
                    diagnostics.reversingSingleThrow.status
                )
        );

        auto sameHanded = protocol;
        sameHanded.reversingSingleThrow.expectedWindingSign = 1;
        for (auto& sample :
             sameHanded.reversingSingleThrow.samples) {
            sample.workingJawCenterM[1] *= -1.0;
        }
        const auto sameHandedDiagnostics =
            metalrobo::certifySurgeonsKnotInstrumentProtocol(
                sameHanded
            );
        require(
            sameHandedDiagnostics.status ==
                metalrobo::SurgeonsKnotProtocolStatus::
                    invalidThrowSequence &&
                sameHandedDiagnostics.reversingSingleThrow.succeeded(),
            "same-handed second throw was not rejected structurally"
        );

        auto missedGate = protocol.firstDoubleThrow;
        missedGate.transferGateCenterM[1] +=
            2.0 * missedGate.transferGateRadiusM;
        const auto missedGateDiagnostics =
            metalrobo::certifySurgicalThrowPath(missedGate);
        require(
            missedGateDiagnostics.status ==
                metalrobo::SurgicalThrowStatus::
                    transferGateClearanceViolation,
            "off-centre tail transfer was not rejected"
        );

        auto missingWrap = protocol.firstDoubleThrow;
        missingWrap.expectedWholeTurns = 1u;
        const auto missingWrapDiagnostics =
            metalrobo::certifySurgicalThrowPath(missingWrap);
        require(
            missingWrapDiagnostics.status ==
                metalrobo::SurgicalThrowStatus::windingMismatch,
            "missing first-throw wrap was not rejected"
        );

        constexpr metalrobo::SurgicalKnotContactSpec contactSpec{
            .threadRadiusM = 1.0e-4,
            .contactMarginM = 5.0e-5,
            .separationToleranceM = 1.0e-9,
            .minimumMaterialEdgeSeparation = 2u,
            .minimumContactPairCount = 1u,
        };
        const std::array<metalrobo::SurgicalKnotPoint, 6u>
            separatedThread{{
                {0.000, 0.0, 0.0},
                {0.001, 0.0, 0.0},
                {0.002, 0.0, 0.0},
                {0.003, 0.0, 0.0},
                {0.004, 0.0, 0.0},
                {0.005, 0.0, 0.0},
            }};
        const auto separatedContact =
            metalrobo::certifySurgicalKnotContacts(
                separatedThread,
                contactSpec
            );
        require(
            separatedContact.status ==
                    metalrobo::SurgicalKnotContactStatus::
                        insufficientContacts &&
                separatedContact.contactPairCount == 0u,
            "a straight separated strand was accepted as a knot contact"
        );
        const std::array<metalrobo::SurgicalKnotPoint, 5u>
            intersectingThread{{
                {-0.001, 0.000, 0.0},
                {0.001, 0.000, 0.0},
                {0.002, 0.001, 0.0},
                {0.000, -0.001, 0.0},
                {0.000, 0.001, 0.0},
            }};
        const auto intersectingContact =
            metalrobo::certifySurgicalKnotContacts(
                intersectingThread,
                contactSpec
            );
        require(
            intersectingContact.status ==
                    metalrobo::SurgicalKnotContactStatus::
                        interpenetratingContact &&
                intersectingContact.interpenetratingPairCount > 0u,
            "an interpenetrating strand crossing was accepted as physical"
        );

        constexpr std::uint32_t kThreadNodeCount = 128u;
        constexpr double kThreadLengthM = 0.25;
        std::vector<metalrobo::SurgicalKnotPoint> threadRestNodes;
        threadRestNodes.reserve(kThreadNodeCount);
        for (std::uint32_t node = 0u;
             node < kThreadNodeCount;
             ++node) {
            threadRestNodes.push_back({
                kThreadLengthM * static_cast<double>(node) /
                    static_cast<double>(kThreadNodeCount - 1u),
                0.0,
                0.0,
            });
        }
        const metalrobo::SurgicalSutureMaterialPlanSpec materialSpec{
            // Intracorporeal technique guidance limits the prepared short end
            // to 20 mm. The working requirement covers the protocol's two
            // 12 mm-radius wraps, transfer, and finite jaw reserve.
            .targetFreeTailLengthM = 0.019,
            .freeTailToleranceM = 0.001,
            .minimumWorkingArcLengthM = 0.180,
            .minimumStitchArcLengthM = 0.006,
            .maximumDrawPerStrokeM = 0.025,
            .maximumStrokeCount = 16u,
        };
        const auto materialPlan =
            metalrobo::planSurgicalSuturePullThrough(
                threadRestNodes,
                25u,
                20u,
                materialSpec
            );
        const auto materialReplay =
            metalrobo::planSurgicalSuturePullThrough(
                threadRestNodes,
                25u,
                20u,
                materialSpec
            );
        require(
            materialPlan.succeeded() && materialReplay.succeeded() &&
                materialPlan.strokes.size() == 8u &&
                materialPlan.strokes.size() ==
                    materialReplay.strokes.size() &&
                std::abs(
                    materialPlan.target.freeTailLengthM -
                    materialSpec.targetFreeTailLengthM
                ) <= 1.0e-12 &&
                materialPlan.target.workingArcLengthM >=
                    materialSpec.minimumWorkingArcLengthM &&
                materialPlan.current.stitchArcLengthM >=
                    materialSpec.minimumStitchArcLengthM &&
                materialPlan.current.conservationErrorM <= 1.0e-12 &&
                materialPlan.target.conservationErrorM <= 1.0e-12,
            "source-sized suture did not produce a bounded knot pull plan"
        );
        for (std::size_t stroke = 0u;
             stroke < materialPlan.strokes.size();
             ++stroke) {
            const auto& accepted = materialPlan.strokes[stroke];
            const auto& replayed = materialReplay.strokes[stroke];
            require(
                accepted.index == replayed.index &&
                    accepted.drawLengthM == replayed.drawLengthM &&
                    accepted.freeTailBeforeM ==
                        replayed.freeTailBeforeM &&
                    accepted.freeTailAfterM ==
                        replayed.freeTailAfterM &&
                    accepted.drawLengthM <=
                        materialSpec.maximumDrawPerStrokeM &&
                    (stroke == 0u ||
                     accepted.freeTailBeforeM ==
                        materialPlan.strokes[stroke - 1u]
                            .freeTailAfterM),
                "pull-stroke material coordinates were not exact"
            );
        }
        const auto reversedMaterial =
            metalrobo::planSurgicalSuturePullThrough(
                threadRestNodes,
                20u,
                25u,
                materialSpec
            );
        require(
            reversedMaterial.status ==
                metalrobo::SurgicalSutureMaterialPlanStatus::
                    reversedTractOrder,
            "reversed two-bite material order was not rejected"
        );
        auto overPulledSpec = materialSpec;
        overPulledSpec.targetFreeTailLengthM = 0.210;
        const auto overPulled =
            metalrobo::planSurgicalSuturePullThrough(
                threadRestNodes,
                25u,
                20u,
                overPulledSpec
            );
        require(
            overPulled.status ==
                metalrobo::SurgicalSutureMaterialPlanStatus::overPulledTail,
            "over-pulled free tail was not rejected"
        );
        auto capacitySpec = materialSpec;
        capacitySpec.maximumStrokeCount = 7u;
        const auto insufficientStrokeCapacity =
            metalrobo::planSurgicalSuturePullThrough(
                threadRestNodes,
                25u,
                20u,
                capacitySpec
            );
        require(
            insufficientStrokeCapacity.status ==
                metalrobo::SurgicalSutureMaterialPlanStatus::
                    strokeCapacityExceeded,
            "undersized pull-stroke budget was not rejected"
        );

        const auto& psm = metalrobo::surgicalPSMMetadata();
        std::cout << std::setprecision(9)
            << "surgical_knot_protocol=ok"
            << " instrument_diameter_mm="
            << 1000.0 * psm.instrumentDiameter
            << " jaw_length_mm="
            << 1000.0 * psm.largeNeedleDriverJawLength
            << " first_signed_turns="
            << diagnostics.firstDoubleThrow.signedWindingTurns
            << " reversing_signed_turns="
            << diagnostics.reversingSingleThrow.signedWindingTurns
            << " first_minimum_instrument_clearance_m="
            << diagnostics.firstDoubleThrow
                .minimumInstrumentClearanceM
            << " reversing_minimum_instrument_clearance_m="
            << diagnostics.reversingSingleThrow
                .minimumInstrumentClearanceM
            << " first_transfer_gate_clearance_m="
            << diagnostics.firstDoubleThrow.transferGateClearanceM
            << " reversing_transfer_gate_clearance_m="
            << diagnostics.reversingSingleThrow
                .transferGateClearanceM
            << " maximum_jaw_center_speed_mps="
            << std::max(
                diagnostics.firstDoubleThrow.maximumWorkingJawSpeedMps,
                diagnostics.reversingSingleThrow
                    .maximumWorkingJawSpeedMps
            )
            << " first_cinch_gain_m="
            << diagnostics.firstDoubleThrow.finalCinchSeparationM -
                diagnostics.firstDoubleThrow.initialCinchSeparationM
            << " reversing_cinch_gain_m="
            << diagnostics.reversingSingleThrow.finalCinchSeparationM -
                diagnostics.reversingSingleThrow
                    .initialCinchSeparationM
            << " material_current_working_arc_m="
            << materialPlan.current.workingArcLengthM
            << " material_stitch_arc_m="
            << materialPlan.current.stitchArcLengthM
            << " material_current_free_tail_m="
            << materialPlan.current.freeTailLengthM
            << " material_required_draw_m="
            << materialPlan.requiredDrawLengthM
            << " material_target_working_arc_m="
            << materialPlan.target.workingArcLengthM
            << " material_target_free_tail_m="
            << materialPlan.target.freeTailLengthM
            << " pull_strokes=" << materialPlan.strokes.size()
            << " pull_maximum_stroke_m="
            << materialSpec.maximumDrawPerStrokeM
            << " negative_same_handed=reject"
            << " negative_missing_wrap=reject"
            << " negative_gate_miss=reject"
            << " negative_reversed_tract_order=reject"
            << " negative_over_pulled_tail=reject"
            << " negative_stroke_capacity=reject"
            << " boundary=instrument_trajectory_only_not_live_thread_or_robot"
            << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "surgical_knot_protocol=failed reason=\""
            << exception.what() << "\"\n";
        return 1;
    }
}
