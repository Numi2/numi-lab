#include "metalrobo/SurgicalKnot.hpp"
#include "metalrobo/SurgicalPSM.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

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
            << " negative_same_handed=reject"
            << " negative_missing_wrap=reject"
            << " negative_gate_miss=reject"
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
