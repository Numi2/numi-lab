#include "cardboard_shipping_memory.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {
using namespace numi::cardboard;

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class F>
void expectFailure(F&& function, const char* message) {
    bool failed = false;
    try { function(); }
    catch (const std::exception&) { failed = true; }
    require(failed, message);
}

void run() {
    const auto single = estimateShippingStaticMemoryLowerBound(1000u, 2000u, 1u);
    const auto doubleEnvironment =
        estimateShippingStaticMemoryLowerBound(1000u, 2000u, 2u);
    require(single.totalBytes > single.femNodeStatesBytes &&
            single.totalBytes > single.tetrahedraBytes,
            "static lower bound omitted its FEM or Krylov terms");
    require(doubleEnvironment.femNodeStatesBytes ==
                2u * single.femNodeStatesBytes &&
            doubleEnvironment.tetrahedraBytes ==
                2u * single.tetrahedraBytes &&
            doubleEnvironment.fgmresBasisBytes ==
                2u * single.fgmresBasisBytes,
            "static estimate did not scale with environment count");

    const auto prebuild = screenShippingPrebuildMemory(1.025, 0.0079);
    require(prebuild.referencePitchIntervals == 130u &&
                prebuild.requestedPitchIntervals == 130u &&
                prebuild.yIntervals == 48u &&
                prebuild.nominalNodeCount == kShippingReferenceNodes &&
                prebuild.nominalTetrahedronCount == kShippingReferenceTetrahedra &&
                prebuild.estimatedNodeCount == 566723u &&
                prebuild.estimatedTetrahedronCount == 1847160u &&
                prebuild.staticArenaScreen.femNodeStatesBytes == 45337840u &&
                prebuild.staticArenaScreen.tetrahedraBytes == 177327360u &&
                prebuild.staticArenaScreen.fgmresBasisBytes == 598459488u &&
                prebuild.staticArenaScreen.totalBytes == 821124688u &&
                prebuild.staticArenaScreen.environmentCount == 1u,
            "prebuild estimate differs from the exact shipping reference or 25% margin");
    ShippingPrebuildArenaOptions preflightOnlyOptions;
    const auto preflightOnlyBudget = estimateShippingPrebuildMemoryBudget(
        prebuild, preflightOnlyOptions);
    require(preflightOnlyBudget.withinLimit,
            "shipping preflight-only arena estimate exceeded the 16 GiB planning limit");
    ShippingPrebuildArenaOptions fullWorldOptions;
    fullWorldOptions.rigidProxyCount = 39u;
    fullWorldOptions.rigidBodyCount = 5u;
    fullWorldOptions.cohesiveFaceUpperBound =
        prebuild.nominalTetrahedronCount / 5u;
    fullWorldOptions.compactImmutableSurfaceMap = true;
    const auto fullDefaultCapacityBudget =
        estimateShippingPrebuildMemoryBudget(prebuild, fullWorldOptions);
    require(!fullDefaultCapacityBudget.withinLimit,
            "full shipping contact upper bound with default slots passed the 16 GiB gate");
    fullWorldOptions.deformableContactCapacity = 1048576u;
    const auto explicitCapacityBudget =
        estimateShippingPrebuildMemoryBudget(prebuild, fullWorldOptions);
    require(explicitCapacityBudget.withinLimit &&
                explicitCapacityBudget.rigidContactRuntimeBytes > 0u,
            "explicit contact-capacity prebuild estimate omitted tool-contact arenas");
    const auto longer = screenShippingPrebuildMemory(10.0, 0.0079);
    require(longer.estimatedNodeCount > prebuild.estimatedNodeCount &&
                longer.estimatedTetrahedronCount > prebuild.estimatedTetrahedronCount,
            "prebuild screen did not grow with requested flute intervals");
    const auto moreYIntervals = screenShippingPrebuildMemory(
        1.025, 0.0079, 1u, kShippingFgmresRestartCeiling, 64u, 1u);
    const auto thicker = screenShippingPrebuildMemory(1.025, 0.0079, 1u,
                                                       kShippingFgmresRestartCeiling,
                                                       kShippingReferenceYIntervals,
                                                       2u);
    require(moreYIntervals.yIntervals == 64u && thicker.thicknessSlices == 2u &&
                moreYIntervals.estimatedNodeCount > prebuild.estimatedNodeCount &&
                moreYIntervals.estimatedTetrahedronCount > prebuild.estimatedTetrahedronCount &&
                thicker.estimatedNodeCount > prebuild.estimatedNodeCount &&
                thicker.estimatedTetrahedronCount > prebuild.estimatedTetrahedronCount,
            "prebuild screen ignored Y-interval or thickness discretization");

    const ShippingRuntimeMemoryObservation within{
        5ull * 1024ull * 1024ull * 1024ull,
        6ull * 1024ull * 1024ull * 1024ull,
        20ull * 1024ull * 1024ull * 1024ull,
        7ull * 1024ull * 1024ull * 1024ull,
        true};
    enforceShippingPeakMemoryLimit(within);
    require(shippingObservedPeakBytes(within) == within.hostResidentBytes,
            "observed peak did not select largest sampled footprint");

    auto overLimit = within;
    overLimit.metalCurrentAllocatedBytes = 17ull * 1024ull * 1024ull * 1024ull;
    expectFailure([&] { enforceShippingPeakMemoryLimit(overLimit); },
                  "native memory gate admitted a measured limit violation");
    auto beforeRuntime = within;
    beforeRuntime.runtimeResidentBytes = 0u;
    expectFailure([&] { enforceShippingPeakMemoryLimit(beforeRuntime); },
                  "native memory gate treated pre-initialize lower bound as runtime measurement");
    expectFailure([&] {
        (void)estimateShippingStaticMemoryLowerBound(
            std::numeric_limits<std::uint64_t>::max(), 2u, 2u);
    }, "static estimate did not reject arithmetic overflow");
    expectFailure([&] {
        (void)estimateShippingStaticMemoryLowerBound(10u, 10u, 1u, 17u);
    }, "static estimate admitted restart above the native ceiling");
    expectFailure([&] { (void)screenShippingPrebuildMemory(0.0, 0.0079); },
                  "prebuild screen admitted a zero perimeter");
    expectFailure([&] {
        (void)screenShippingPrebuildMemory(1.025, 0.0079, 1u,
                                           kShippingFgmresRestartCeiling,
                                           kShippingYIntervalCeiling + 1u, 1u);
    }, "prebuild screen admitted an unsupported Y-interval discretization");

    ShippingMemoryBudgetInput shipping;
    shipping.nodeCount = kShippingReferenceNodes;
    shipping.tetrahedronCount = kShippingReferenceTetrahedra;
    shipping.cohesiveFaceCount = 100000u;
    shipping.contactPairCount = 10000u;
    shipping.activeContactCapacity = 4096u;
    shipping.deformableContactCapacity =
        8u * kShippingReferenceTetrahedra;
    shipping.rigidGeneralizedCapacity = 6u;
    shipping.rigidBodyCount = 5u;
    shipping.rigidProxyCount = 4u;
    shipping.materialStateStride = 27u;
    shipping.compactPreconditionerRegionCount = 3u;
    shipping.environmentCount = 1u;
    shipping.fgmresRestart = 16u;
    shipping.regionalGalerkinEnabled = false;
    shipping.hostSourceMeshBytes = 400ull * 1024ull * 1024ull;
    shipping.hostCompiledWorldBytes = 300ull * 1024ull * 1024ull;
    shipping.hostGeneratedMetalBytes = 32ull * 1024ull * 1024ull;
    shipping.retainedCompleteCheckpointCount = 2u;
    shipping.uploadStagingBytes = 128ull * 1024ull * 1024ull;
    shipping.estimatedSnapshotPayloadBytes =
        estimateShippingSnapshotPayloadBytes(shipping);
    const auto budget = estimateShippingMemoryBudget(shipping);
    const std::uint64_t referenceContact = shippingPersistentContactBytes(
        kShippingReferenceNodes, kShippingReferenceTetrahedra, 0u,
        8u * kShippingReferenceTetrahedra, 1u);
    require(budget.measuredBaseAndContactEnvelopeBytes >
                budget.measuredNoncontactRuntimeEnvelopeBytes &&
                budget.persistentContactBytes >= referenceContact &&
                budget.rigidContactRuntimeBytes > 0u &&
                budget.regionalGalerkinBytes == 0u &&
                budget.cohesiveRuntimeBytes > 0u &&
                budget.hostCheckpointBytes ==
                    2u * shipping.estimatedSnapshotPayloadBytes &&
                budget.snapshotReadbackBytes ==
                    shipping.estimatedSnapshotPayloadBytes &&
                budget.estimatedPeakDomainBytes ==
                    std::max(budget.estimatedMetalPeakBytes,
                             budget.estimatedHostPeakBytes) &&
                budget.withinLimit,
            "shipping arena budget omitted measured baseline, contact, cohesive, or checkpoint arenas");

    // At the measured reference dimensions, the decomposed baseline must
    // reproduce the native resident anchor before optional geometry is added.
    ShippingMemoryBudgetInput measured;
    measured.nodeCount = kShippingReferenceNodes;
    measured.tetrahedronCount = kShippingReferenceTetrahedra;
    measured.deformableContactCapacity =
        8u * kShippingReferenceTetrahedra;
    measured.fgmresRestart = kShippingFgmresRestartCeiling;
    const auto measuredBudget = estimateShippingMemoryBudget(measured);
    require(measuredBudget.measuredBaseAndContactEnvelopeBytes ==
                kShippingCase032RuntimeResidentBytes +
                    kShippingReferenceNodes * sizeof(nm_float4),
            "reference decomposition must retain case032 plus the new contact-force readback arena");
    require(measuredBudget.constraintContactForceReadbackBytes ==
                kShippingReferenceNodes * sizeof(nm_float4),
            "new contact-force readback arena size differs from runtime allocation");

    auto compactSurface = measured;
    compactSurface.compactImmutableSurfaceMap = true;
    compactSurface.surfaceFaceCount = 2u * compactSurface.tetrahedronCount;
    const auto compactBudget = estimateShippingMemoryBudget(compactSurface);
    require(compactBudget.persistentContactSurfaceBytes <
                measuredBudget.persistentContactSurfaceBytes &&
                compactBudget.persistentContactSlotBytes ==
                    measuredBudget.persistentContactSlotBytes,
            "compact immutable map estimate failed to reduce only the surface-domain arenas");
    auto missingCompactSurfaceCount = compactSurface;
    missingCompactSurfaceCount.surfaceFaceCount = 0u;
    expectFailure([&] {
        (void)estimateShippingMemoryBudget(missingCompactSurfaceCount);
    }, "compact immutable surface estimate silently used the legacy face count");
    auto impossibleCompactSurfaceCount = compactSurface;
    impossibleCompactSurfaceCount.surfaceFaceCount =
        4u * impossibleCompactSurfaceCount.tetrahedronCount + 1u;
    expectFailure([&] {
        (void)estimateShippingMemoryBudget(impossibleCompactSurfaceCount);
    }, "compact immutable surface estimate exceeded the all-tet face bound");
    auto reducedCapacity = measured;
    reducedCapacity.deformableContactCapacity = 131072u;
    const auto reducedCapacityBudget = estimateShippingMemoryBudget(reducedCapacity);
    require(reducedCapacityBudget.persistentContactSlotBytes <
                measuredBudget.persistentContactSlotBytes &&
                reducedCapacityBudget.persistentContactSurfaceBytes ==
                    measuredBudget.persistentContactSurfaceBytes,
            "explicit contact capacity estimate failed to isolate contact-slot savings");
    ShippingMemoryBudgetInput privateSurface;
    privateSurface.nodeCount = 128u;
    privateSurface.tetrahedronCount = 64u;
    privateSurface.deformableContactCapacity = 512u;
    privateSurface.fullContactArmijoEnabled = true;
    const auto legacyPrivateSurface = estimateShippingMemoryBudget(privateSurface);
    privateSurface.compactImmutableSurfaceMap = true;
    privateSurface.surfaceFaceCount = 48u;
    const auto compactPrivateSurface = estimateShippingMemoryBudget(privateSurface);
    require(compactPrivateSurface.fullContactArmijoBytes <
                legacyPrivateSurface.fullContactArmijoBytes,
            "private contact arena estimate ignored the compact immutable surface map");

    // A smaller three-role RAP case is admissible. At the shipping dimensions,
    // native's 256 MiB cap rejects it before runtime allocation. Capturing the
    // regional diagnostics also has an independent 64 MiB readback cap.
    ShippingMemoryBudgetInput rap;
    rap.nodeCount = 72000u;
    rap.tetrahedronCount = 80000u;
    rap.deformableContactCapacity = 8u * rap.tetrahedronCount;
    rap.regionalGalerkinEnabled = true;
    rap.compactPreconditionerRegionCount = 3u;
    const auto rapBudget = estimateShippingMemoryBudget(rap);
    require(rapBudget.regionalGalerkinBytes > 0u &&
            rapBudget.regionalGalerkinBlockFineBytes == 0u,
            "small regional Galerkin estimate omitted basis/coarse matrix bytes");
    auto blockRap = rap;
    blockRap.regionalGalerkinBlockFineEnabled = true;
    const auto blockRapBudget = estimateShippingMemoryBudget(blockRap);
    const std::uint64_t expectedBlockFineBytes =
        3u * blockRap.environmentCount * blockRap.nodeCount * sizeof(nm_float4);
    require(blockRapBudget.regionalGalerkinBlockFineBytes ==
                expectedBlockFineBytes &&
            blockRapBudget.regionalGalerkinBytes >=
                rapBudget.regionalGalerkinBytes + expectedBlockFineBytes,
            "selector-3 estimate omitted its 32-mode RAP envelope or 3x3 inverse block arena");
    auto contactRap = rap;
    contactRap.regionalGalerkinBlockFineEnabled = true;
    const auto contactRapBudget = estimateShippingMemoryBudget(contactRap);
    require(contactRapBudget.regionalGalerkinBytes ==
                blockRapBudget.regionalGalerkinBytes &&
            contactRapBudget.regionalGalerkinBlockFineBytes ==
                expectedBlockFineBytes,
            "selector-4 contact RAP estimate diverged from selector-3's 32-mode block envelope");
    auto blockWithoutRap = rap;
    blockWithoutRap.regionalGalerkinEnabled = false;
    blockWithoutRap.regionalGalerkinBlockFineEnabled = true;
    expectFailure([&] { (void)estimateShippingMemoryBudget(blockWithoutRap); },
                  "block-Jacobi fine arena was accepted without its regional RAP basis");
    rap.regionalGalerkinDiagnosticsCapture = true;
    expectFailure([&] { (void)estimateShippingMemoryBudget(rap); },
                  "regional Galerkin diagnostics passed the 64 MiB snapshot cap");
    auto shippingRap = measured;
    shippingRap.regionalGalerkinEnabled = true;
    shippingRap.compactPreconditionerRegionCount = 3u;
    expectFailure([&] { (void)estimateShippingMemoryBudget(shippingRap); },
                  "shipping-sized three-region RAP passed the runtime 256 MiB admission cap");
    auto shippingBlockRap = shippingRap;
    shippingBlockRap.regionalGalerkinBlockFineEnabled = true;
    expectFailure([&] { (void)estimateShippingMemoryBudget(shippingBlockRap); },
                  "shipping-sized selector-3 RAP plus block arena passed the runtime 256 MiB admission cap");
    auto shippingContactRap = shippingRap;
    shippingContactRap.regionalGalerkinBlockFineEnabled = true;
    expectFailure([&] { (void)estimateShippingMemoryBudget(shippingContactRap); },
                  "shipping-sized selector-4 contact RAP passed the runtime 256 MiB admission cap");
    auto privateContact = shipping;
    privateContact.rigidContactArmijoEnabled = true;
    privateContact.fullContactArmijoEnabled = true;
    privateContact.cohesiveArmijoEnabled = true;
    expectFailure([&] { (void)estimateShippingMemoryBudget(privateContact); },
                  "shipping budget admitted private Armijo contact above runtime limits");
    auto requiredButUnselected = shipping;
    requiredButUnselected.fullContactArmijoRequired = true;
    requiredButUnselected.rigidProxyCount = 39u;
    requiredButUnselected.contactPairCount =
        requiredButUnselected.nodeCount * requiredButUnselected.rigidProxyCount;
    requiredButUnselected.activeContactCapacity =
        requiredButUnselected.contactPairCount;
    const auto requiredBudget =
        estimateShippingMemoryBudget(requiredButUnselected);
    require(requiredBudget.requiredFullContactArmijoBytes > 0u &&
                requiredBudget.requiredRigidContactArmijoBytes > 0u &&
                requiredBudget.requiredCohesiveArmijoBytes > 0u &&
                !requiredBudget.requiredResidualDescentArenaAdmitted &&
                !requiredBudget.withinLimit &&
                requiredBudget.fullContactArmijoBytes == 0u,
            "required residual-descent arenas were omitted when the selector was not allocated");

    // Keep the estimator on the runtime's effective full-contact scope: it
    // delegates to capsule/box admission, which caps FEM nodes at 8,192.
    const auto fullContactNodeAdmission = [](const std::uint64_t nodes) {
        ShippingMemoryBudgetInput input;
        input.nodeCount = nodes;
        input.tetrahedronCount = 1u;
        input.cohesiveFaceCount = 1u;
        input.contactPairCount = 1u;
        input.activeContactCapacity = 1u;
        input.deformableContactCapacity = 1u;
        input.rigidProxyCount = 2u;
        input.environmentCount = 1u;
        input.fullContactArmijoRequired = true;
        return estimateShippingMemoryBudget(input);
    };
    const auto atFullContactNodeLimit = fullContactNodeAdmission(
        kShippingFullContactArmijoNodeLimit);
    require(atFullContactNodeLimit.requiredResidualDescentArenaAdmitted,
            "full-contact memory admission rejected the runtime's 8,192-node boundary");
    for (const std::uint64_t nodes : {
             kShippingFullContactArmijoNodeLimit + 1u,
             std::uint64_t{16384u}}) {
        const auto outsideFullContactNodeLimit = fullContactNodeAdmission(nodes);
        require(!outsideFullContactNodeLimit.requiredResidualDescentArenaAdmitted &&
                    !outsideFullContactNodeLimit.withinLimit,
                "shipping memory admission accepted a node count above the runtime full-contact limit");
    }

    auto outsideCohesiveLimit = shipping;
    outsideCohesiveLimit.cohesiveFaceCount = 4097u;
    outsideCohesiveLimit.cohesiveArmijoEnabled = true;
    expectFailure([&] { (void)estimateShippingMemoryBudget(outsideCohesiveLimit); },
                  "shipping budget admitted cohesive Armijo above runtime face cap");
    auto overflowingHost = shipping;
    overflowingHost.hostSourceMeshBytes =
        std::numeric_limits<std::uint64_t>::max();
    expectFailure([&] { (void)estimateShippingMemoryBudget(overflowingHost); },
                  "shipping budget did not check host-overlap byte arithmetic");
    auto rejectedBudget = budget;
    rejectedBudget.withinLimit = false;
    expectFailure([&] { enforceShippingMemoryBudget(rejectedBudget); },
                  "shipping memory gate admitted a budget marked over-limit");
}
} // namespace

int main() {
    try {
        run();
        std::cout << "shipping memory preflight checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "shipping memory preflight check failed: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
