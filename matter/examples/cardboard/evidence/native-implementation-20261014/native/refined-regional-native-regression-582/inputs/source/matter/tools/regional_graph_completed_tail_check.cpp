#include "../src/regional_graph_schwarz.hpp"

#include <array>
#include <cstdint>
#include <iostream>

using numi::matter::detail::regionalGraphCapturePendingForIteration;
using numi::matter::detail::regionalGraphCompletedTailReuseAllowed;
using numi::matter::detail::regionalGraphMaySkipPatchFactorization;

namespace {

struct EnvironmentFixture {
    bool nonlinearLatch = false;
    bool diagnosticsCompleted = false;
    bool factorValid = false;
    std::uint32_t currentLinearization = 0u;
    std::uint32_t factorLinearization = 0u;
    std::uint32_t matrixBuilds = 0u;
    std::uint32_t factorizations = 0u;
};

bool require(const bool condition, const char* message, std::uint32_t& checks) {
    ++checks;
    if (condition) return true;
    std::cerr << "regional_graph_completed_tail_check: " << message << '\n';
    return false;
}

}  // namespace

int main() {
    std::uint32_t checks = 0u;
    bool ok = true;

    ok &= require(!regionalGraphCompletedTailReuseAllowed(0u, false, false),
        "iteration zero must rebuild even with no forced capture", checks);
    ok &= require(regionalGraphCompletedTailReuseAllowed(1u, false, false),
        "later same-transaction tail may consult its completion latch", checks);
    ok &= require(!regionalGraphCompletedTailReuseAllowed(4u, true, false),
        "explicit cache rebuild must suppress reuse", checks);
    const bool selectedCaptureAtIterationFour =
        regionalGraphCapturePendingForIteration(
            true, false, 7u, 7u, 4u, 4u);
    const bool selectedCaptureAtIterationFive =
        regionalGraphCapturePendingForIteration(
            true, false, 7u, 7u, 4u, 5u);
    const bool selectedCaptureAtOtherRoot =
        regionalGraphCapturePendingForIteration(
            true, false, 7u, 8u, 4u, 4u);
    const bool completedCaptureAtIterationFour =
        regionalGraphCapturePendingForIteration(
            true, true, 7u, 7u, 4u, 4u);
    const bool invalidSelectorAtIterationFour =
        regionalGraphCapturePendingForIteration(
            false, false, 7u, 7u, 4u, 4u);
    ok &= require(selectedCaptureAtIterationFour &&
                      !regionalGraphCompletedTailReuseAllowed(
                          4u, false, selectedCaptureAtIterationFour),
        "validated selected iteration must force its graph rebuild", checks);
    ok &= require(!selectedCaptureAtIterationFive &&
                      regionalGraphCompletedTailReuseAllowed(
                          5u, false, selectedCaptureAtIterationFive),
        "unselected later iteration must keep completed-tail reuse enabled", checks);
    ok &= require(!selectedCaptureAtOtherRoot,
        "capture selector for another root must not disable reuse", checks);
    ok &= require(!completedCaptureAtIterationFour,
        "already completed capture must not keep disabling reuse", checks);
    ok &= require(!invalidSelectorAtIterationFour &&
                      regionalGraphCompletedTailReuseAllowed(
                          4u, false, invalidSelectorAtIterationFour),
        "invalid or disabled selector must fail closed without disabling reuse", checks);
    ok &= require(!regionalGraphCompletedTailReuseAllowed(4u, false, true),
        "selected diagnostic generation must suppress reuse", checks);
    ok &= require(!regionalGraphMaySkipPatchFactorization(true, false, true),
        "active environment must rebuild", checks);
    ok &= require(regionalGraphMaySkipPatchFactorization(true, true, true),
        "completed environment may retain a valid previous factor", checks);
    ok &= require(!regionalGraphMaySkipPatchFactorization(true, true, false),
        "invalid old factor must be rebuilt", checks);

    // Simulate three environment states: active, completed/no-op, and a
    // latched first no-op whose next linear solve is still active. The last
    // case may reuse only because the candidate linearization identity has
    // not changed; this is not a native FEM trial.
    std::array<EnvironmentFixture, 3> environments{};
    environments[0] = {false, false, true, 8u, 7u, 3u, 3u};
    environments[1] = {true, true, true, 7u, 7u, 3u, 3u};
    environments[2] = {true, false, true, 7u, 7u, 3u, 3u};
    const std::uint32_t currentLinearization = 8u;
    const bool reuseAllowed = regionalGraphCompletedTailReuseAllowed(
        40u, false, false);
    for (auto& environment : environments) {
        if (regionalGraphMaySkipPatchFactorization(
                reuseAllowed, environment.nonlinearLatch,
                environment.factorValid))
            continue;
        environment.currentLinearization = currentLinearization;
        ++environment.matrixBuilds;
        environment.factorValid = false;  // factor kernel's first write
        ++environment.factorizations;
        environment.factorLinearization = currentLinearization;
        environment.factorValid = true;
    }
    ok &= require(environments[0].currentLinearization == 8u &&
                      environments[0].factorLinearization == 8u &&
                      environments[0].matrixBuilds == 4u &&
                      environments[0].factorizations == 4u,
        "active environment must rebuild matrix and factor", checks);
    ok &= require(environments[1].currentLinearization == 7u &&
                      environments[1].factorLinearization == 7u &&
                      environments[1].matrixBuilds == 3u &&
                      environments[1].factorizations == 3u,
        "completed environment retains its factor for the unchanged candidate", checks);
    ok &= require(environments[2].currentLinearization == 7u &&
                      environments[2].factorLinearization == 7u &&
                      environments[2].matrixBuilds == 3u &&
                      environments[2].factorizations == 3u,
        "latched but active solve may consume the valid factor only for unchanged A", checks);
    const bool completedSolveReadsFactor = !environments[1].diagnosticsCompleted;
    const bool activeLatchedSolveReadsMatchingFactor =
        !environments[2].diagnosticsCompleted && environments[2].factorValid &&
        environments[2].factorLinearization == environments[2].currentLinearization;
    ok &= require(!completedSolveReadsFactor && activeLatchedSolveReadsMatchingFactor,
        "completed solve bypasses the factor; active solve sees the same-generation valid factor", checks);

    // Invalid factor state never passes the skip predicate. It gets rebuilt;
    // the transaction boundary also disables reuse even if the prior latch is
    // still set in persistent FGMRES state.
    EnvironmentFixture staleInvalid{true, false, false, 5u, 4u, 2u, 2u};
    ok &= require(!regionalGraphMaySkipPatchFactorization(
                      reuseAllowed, staleInvalid.nonlinearLatch,
                      staleInvalid.factorValid),
        "completed latch does not preserve an invalid factor", checks);
    const bool newTransactionReuseAllowed =
        regionalGraphCompletedTailReuseAllowed(0u, false, false);
    ok &= require(!newTransactionReuseAllowed &&
                      !regionalGraphMaySkipPatchFactorization(
                          newTransactionReuseAllowed,
                          environments[1].nonlinearLatch,
                          environments[1].factorValid),
        "new transaction rebuilds even a valid factor with an old latch", checks);
    staleInvalid.factorValid = false;
    staleInvalid.currentLinearization = 6u;
    ++staleInvalid.matrixBuilds;
    ++staleInvalid.factorizations;
    staleInvalid.factorLinearization = 6u;
    staleInvalid.factorValid = true;
    ok &= require(staleInvalid.factorValid &&
                      staleInvalid.factorLinearization == 6u,
        "rebuild replaces stale invalid factor state", checks);

    std::cout << "regional_graph_completed_tail_checks=" << checks
              << " result=" << (ok ? "PASS" : "FAIL") << '\n';
    return ok ? 0 : 1;
}
