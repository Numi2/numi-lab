#include "metalrobo/EngineModel.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"

#include <cstdlib>
#include <iostream>
#include <vector>

namespace {
void expect(const bool condition, const char* message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
}

void expectRejectedBeforeDispatch(
    metalrobo::MetalArticulatedOperatorContext& context,
    const metalrobo::EngineModel& model,
    const metalrobo::MetalArticulatedOperatorInput& input,
    const char* reason
) {
    metalrobo::MetalArticulatedOperatorSubmission submission;
    const std::vector<float> qBefore(input.q.begin(), input.q.end());
    const std::vector<float> vBefore(input.stand.v.begin(), input.stand.v.end());
    const auto result = context.submit(model, input, submission);
    expect(result.status ==
               metalrobo::MetalArticulatedOperatorHostStatus::invalidDimensions,
           "invalid observer configuration is rejected as invalid dimensions");
    expect(result.message.find(reason) != std::string::npos,
           "rejection comes from the requested root observer guard");
    expect(!result.dispatched && !result.published && !submission.valid(),
           "input-only validation rejects before dispatch or publication");
    expect(std::vector<float>(input.q.begin(), input.q.end()) == qBefore &&
               std::vector<float>(input.stand.v.begin(), input.stand.v.end()) == vBefore,
           "rejected admission does not mutate caller q/v");
}

metalrobo::MetalArticulatedOperatorInput baseInput(
    const metalrobo::EngineModel& model,
    const std::uint32_t stepCount
) {
    metalrobo::MetalArticulatedOperatorInput input;
    input.articulationIndex = 0u;
    input.environmentCount = 1u;
    input.q = model.defaultQ;
    input.v = model.defaultV;
    input.stand.stepCount = stepCount;
    input.stand.authoritativeStepCount = 100u;
    input.stand.v = model.defaultV;
    input.stand.enableContact = false;
    return input;
}
}  // namespace

int main() {
    using namespace metalrobo;
    const EngineModel model = makeFreeSphereEngineModel();

    MetalArticulatedOperatorConfig diagnosticsEnabled;
    diagnosticsEnabled.pointJacobiansOnly = true;
    diagnosticsEnabled.readStandConstraintDiagnostics = true;
    MetalArticulatedOperatorContext context(diagnosticsEnabled);

    auto halfConfigured = baseInput(model, 1u);
    halfConfigured.stand.rootMomentumDiagnosticFirstAcceptedStep = 12u;
    expectRejectedBeforeDispatch(context, model, halfConfigured,
        "root momentum observer requires");

    auto outsideHorizon = baseInput(model, 1u);
    outsideHorizon.stand.rootMomentumDiagnosticFirstAcceptedStep = 99u;
    outsideHorizon.stand.rootMomentumDiagnosticLastAcceptedStep = 101u;
    expectRejectedBeforeDispatch(context, model, outsideHorizon,
        "root momentum observer requires");

    auto overlappingBatch = baseInput(model, 2u);
    overlappingBatch.stand.stepIndexOffset = 8u;
    overlappingBatch.stand.rootMomentumDiagnosticFirstAcceptedStep = 10u;
    overlappingBatch.stand.rootMomentumDiagnosticLastAcceptedStep = 12u;
    expectRejectedBeforeDispatch(context, model, overlappingBatch,
        "root momentum observer requires one-step submissions");

    auto missingDiagnostics = baseInput(model, 1u);
    missingDiagnostics.stand.rootMomentumDiagnosticFirstAcceptedStep = 10u;
    missingDiagnostics.stand.rootMomentumDiagnosticLastAcceptedStep = 12u;
    MetalArticulatedOperatorConfig diagnosticsDisabled = diagnosticsEnabled;
    diagnosticsDisabled.readStandConstraintDiagnostics = false;
    MetalArticulatedOperatorContext diagnosticsDisabledContext(diagnosticsDisabled);
    expectRejectedBeforeDispatch(diagnosticsDisabledContext, model,
        missingDiagnostics, "root momentum observer requires");
    return 0;
}
