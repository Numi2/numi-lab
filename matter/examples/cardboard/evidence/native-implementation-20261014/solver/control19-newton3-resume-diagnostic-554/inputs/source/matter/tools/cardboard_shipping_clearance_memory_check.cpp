#include "cardboard_shipping_memory.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

void check(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void run() {
    using namespace numi::cardboard;
    std::uint64_t exactWorkspace = 0u;
    check(shippingAcceptedInsideClearanceWorkspaceBytes(
              1477728u, 528444u, 1054876u, exactWorkspace) &&
          exactWorkspace == 1028812212u,
          "compiled shipping-count workspace differs from the scale preflight");

    ShippingMemoryBudgetInput noClearance;
    noClearance.nodeCount = 528444u;
    noClearance.tetrahedronCount = 1477728u;
    noClearance.environmentCount = 1u;
    noClearance.fgmresRestart = 16u;
    noClearance.materialStateStride = 1u;
    noClearance.hostSourceMeshBytes = 100000000u;
    noClearance.hostCompiledWorldBytes = 200000000u;
    noClearance.hostGeneratedMetalBytes = 300000000u;
    noClearance.estimatedSnapshotPayloadBytes = 4000000u;
    noClearance.retainedCompleteCheckpointCount = 2u;
    noClearance.uploadStagingBytes = 50000000u;
    const auto baseline = estimateShippingMemoryBudget(noClearance);

    ShippingMemoryBudgetInput withClearance = noClearance;
    withClearance.acceptedInsideClearanceWorkspaceRequired = true;
    withClearance.acceptedInsideClearanceWorkspaceBytes = exactWorkspace;
    const auto included = estimateShippingMemoryBudget(withClearance);
    check(included.acceptedInsideClearanceWorkspaceRequired &&
          included.acceptedInsideClearanceWorkspaceAdmitted &&
          included.acceptedInsideClearanceWorkspaceBytes == exactWorkspace,
          "accepted inside-clearance scratch was not admitted into the host model");
    check(included.estimatedPeakDomainBytes - baseline.estimatedPeakDomainBytes ==
              exactWorkspace &&
          included.estimatedHostPeakBytes - baseline.estimatedHostPeakBytes ==
              exactWorkspace,
          "the geometry scratch does not overlap live host mesh/checkpoint allocations in the peak model");

    ShippingMemoryBudgetInput blocked = noClearance;
    blocked.acceptedInsideClearanceWorkspaceRequired = true;
    blocked.acceptedInsideClearanceWorkspaceAdmitted = false;
    blocked.acceptedInsideClearanceWorkspaceFailure =
        "synthetic workspace cap rejection";
    const auto rejected = estimateShippingMemoryBudget(blocked);
    check(!rejected.withinLimit &&
          !rejected.acceptedInsideClearanceWorkspaceAdmitted &&
          rejected.acceptedInsideClearanceWorkspaceFailure ==
              "synthetic workspace cap rejection",
          "failed clearance-workspace preflight did not fail closed");

    ShippingPrebuildMemoryScreen prebuild;
    prebuild.nominalNodeCount = 528444u;
    prebuild.nominalTetrahedronCount = 1477728u;
    prebuild.estimatedNodeCount = prebuild.nominalNodeCount;
    prebuild.estimatedTetrahedronCount = prebuild.nominalTetrahedronCount;
    ShippingPrebuildArenaOptions options;
    options.includeAcceptedInsideClearanceWorkspace = true;
    ShippingPrebuildArenaOptions optionsWithoutClearance = options;
    optionsWithoutClearance.includeAcceptedInsideClearanceWorkspace = false;
    const auto planningWithoutClearance =
        estimateShippingPrebuildMemoryBudget(prebuild, optionsWithoutClearance);
    const auto planning = estimateShippingPrebuildMemoryBudget(prebuild, options);
    check(planning.acceptedInsideClearanceWorkspaceRequired &&
          planning.acceptedInsideClearanceWorkspaceBytes ==
              inside_clearance_detail::maximumWorkspaceBytes,
          "prebuild model did not reserve the mapper's checked maximum scratch");
    check(planning.limitBytes == 16ull * 1024ull * 1024ull * 1024ull &&
          planning.estimatedPeakDomainBytes -
                  planningWithoutClearance.estimatedPeakDomainBytes ==
              inside_clearance_detail::maximumWorkspaceBytes,
          "prebuild peak accounting did not add the reserved workspace under the unchanged 16 GiB limit");

    ShippingPrebuildMemoryScreen oversized = prebuild;
    oversized.nominalTetrahedronCount =
        inside_clearance_detail::maximumIndexedTetrahedra + 1u;
    oversized.estimatedTetrahedronCount = oversized.nominalTetrahedronCount;
    const auto blockedPlan = estimateShippingPrebuildMemoryBudget(oversized, options);
    check(!blockedPlan.acceptedInsideClearanceWorkspaceAdmitted &&
          !blockedPlan.withinLimit &&
          !blockedPlan.acceptedInsideClearanceWorkspaceFailure.empty(),
          "prebuild plan above the topology cap did not fail closed with a specific blocker");

    std::cout << "source159 shipping-memory accounting tests: PASS\n";
}

} // namespace

int main() {
    try {
        run();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "source159 shipping-memory accounting tests: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
