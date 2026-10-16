#include "cardboard_shipping_memory.hpp"
#include "cardboard_box_assembly_controller.hpp"

#include <array>
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

void checkRuntimeEnvironmentAdmission() {
    const auto sparse = parseShippingSparsePrivateContactOptions("65536", "1");
    require(sparse.requested && sparse.parity && sparse.capacity == 65536u &&
                sparse.error == nullptr,
            "valid sparse runtime environment values did not parse");
    require(!parseShippingSparsePrivateContactOptions(nullptr, nullptr).requested,
            "absent sparse runtime options were treated as requested");
    for (const char* malformed : {"", "0", "+4", " 4", "4294967296"}) {
        require(parseShippingSparsePrivateContactOptions(malformed, nullptr).error != nullptr,
                "malformed sparse capacity was admitted by the preflight parser");
    }
    require(parseShippingSparsePrivateContactOptions(nullptr, "1").error != nullptr &&
                parseShippingSparsePrivateContactOptions("4", "true").error != nullptr,
            "sparse parity parser did not fail closed");

    const auto allSteps = parseShippingFullContactSelector("all");
    const auto oneStep = parseShippingFullContactSelector("0007");
    require(allSteps.requested && allSteps.selectsAllSteps &&
                allSteps.error == nullptr &&
                oneStep.requested && !oneStep.selectsAllSteps &&
                oneStep.controlStep == 7u && oneStep.error == nullptr,
            "valid full-contact selector syntax did not parse");
    for (const char* malformed : {"", "ALL", "-1", "1x", "4294967296"})
        require(parseShippingFullContactSelector(malformed).error != nullptr,
                "malformed full-contact selector was admitted by preflight");

    require(shippingRigidPrivatePairDomainAdmitted(65536u, false) &&
                !shippingRigidPrivatePairDomainAdmitted(65537u, false) &&
                shippingRigidPrivatePairDomainAdmitted(65537u, true) &&
                !shippingRigidPrivatePairDomainAdmitted(0u, true),
            "dense/sparse private-pair selector boundary is inconsistent");

    NMMatterDispatchGPU dispatch{};
    dispatch.rigidProxyCount = 3u;
    dispatch.rigidGeneralizedCapacity = 6u;
    dispatch.rigidQCapacity = 0u;
    std::array<NMRigidProxyGPU, 3u> proxies{};
    for (auto& proxy : proxies) {
        proxy.adaptiveObjectIndex = NM_INVALID_INDEX;
        proxy.generalizedFreeBodyIndex = NM_INVALID_INDEX;
    }
    proxies[0].shapeKind = NM_RIGID_CAPSULE;
    proxies[0].flags = NM_RIGID_PRESCRIBED_TRANSLATION;
    proxies[0].bodyIndex = 0u;
    proxies[1].shapeKind = NM_RIGID_BOX;
    proxies[1].flags = NM_RIGID_DYNAMIC;
    proxies[1].bodyIndex = 1u;
    proxies[1].sceneBodyIndex = 1u;
    proxies[1].generalizedFreeBodyIndex = 0u;
    proxies[2].shapeKind = NM_RIGID_BOX;
    proxies[2].flags = 0u;
    proxies[2].sceneBodyIndex = NM_INVALID_INDEX;
    require(shippingSparsePrivateBoundedFreeBoxLayout(dispatch, proxies),
            "exact bounded capsule/free-box/static-box layout was rejected");
    auto articulated = proxies;
    articulated[1].flags |= NM_RIGID_ARTICULATED;
    require(!shippingSparsePrivateBoundedFreeBoxLayout(dispatch, articulated),
            "articulated free box passed the sparse bounded free-body scope");
    auto generalizedQ = dispatch;
    generalizedQ.rigidQCapacity = 1u;
    require(!shippingSparsePrivateBoundedFreeBoxLayout(generalizedQ, proxies),
            "nonzero rigid Q capacity passed the sparse bounded free-body scope");

    const bool prebuildScopeCandidate =
        shippingPrebuildBoundedFreeBoxScopeCandidate(
            true, true, kShippingRigidProxyCount, kShippingRigidBodyCount,
            6u, kShippingRigidProxyCount, kShippingRigidBodyCount);
    require(prebuildScopeCandidate &&
                !shippingPrebuildBoundedFreeBoxScopeCandidate(
                    true, false, kShippingRigidProxyCount,
                    kShippingRigidBodyCount, 6u,
                    kShippingRigidProxyCount, kShippingRigidBodyCount) &&
                !shippingPrebuildBoundedFreeBoxScopeCandidate(
                    true, true, kShippingRigidProxyCount + 1u,
                    kShippingRigidBodyCount, 6u,
                    kShippingRigidProxyCount, kShippingRigidBodyCount),
            "prebuild bounded BOX candidate does not match the authored scene recipe");
    ShippingPrebuildArenaOptions sparseOptions;
    configureShippingPrebuildSparsePrivateContact(
        sparseOptions, "65536", nullptr, true, prebuildScopeCandidate);
    require(sparseOptions.sparsePrivateContactRequested &&
                sparseOptions.sparsePrivateContactCapacity == 65536u &&
                sparseOptions.sparsePrivateContactBoundedFreeBoxScope &&
                sparseOptions.runtimeConfigurationFailure.empty(),
            "prebuild sparse parser failed to propagate the explicit bounded request");
    ShippingPrebuildArenaOptions unselectedSparseOptions;
    configureShippingPrebuildSparsePrivateContact(
        unselectedSparseOptions, "65536", nullptr, false,
        prebuildScopeCandidate);
    require(!unselectedSparseOptions.runtimeConfigurationFailure.empty(),
            "sparse prebuild request was accepted without selected residual Armijo");
    ShippingPrebuildArenaOptions malformedSparseOptions;
    configureShippingPrebuildSparsePrivateContact(
        malformedSparseOptions, "65k", nullptr, true,
        prebuildScopeCandidate);
    require(malformedSparseOptions.sparsePrivateContactCapacity == 0u &&
                !malformedSparseOptions.runtimeConfigurationFailure.empty(),
            "malformed prebuild sparse option did not fail closed with its parse error");
}

void checkTiledCohesivePreflight() {
    ShippingPrebuildArenaOptions options;
    configureShippingTiledCohesiveTrial(options, nullptr, nullptr, false, false);
    require(!options.fullContactArmijoTiledCohesive &&
                options.runtimeConfigurationFailure.empty(),
            "absent tiled cohesive request changed prebuild accounting");
    configureShippingTiledCohesiveTrial(options, "1", "0003", true, false);
    require(options.fullContactArmijoTiledCohesive &&
                options.cohesiveTrialPageCapacity == 3u,
            "valid tiled cohesive page request did not parse");
    for (const char* malformed : {"", "0", "+1", " 1", "4294967296"}) {
        ShippingPrebuildArenaOptions bad;
        configureShippingTiledCohesiveTrial(bad, "1", malformed, true, false);
        require(!bad.fullContactArmijoTiledCohesive &&
                    bad.cohesiveTrialPageCapacity == 0u &&
                    !bad.runtimeConfigurationFailure.empty(),
                "malformed tiled page request admitted memory savings");
    }
    for (const auto& pair : std::array<std::array<bool, 2>, 2>{{
             {{false, false}}, {{true, true}}}}) {
        ShippingPrebuildArenaOptions bad;
        configureShippingTiledCohesiveTrial(bad, "1", nullptr,
                                           pair[0], pair[1]);
        require(!bad.fullContactArmijoTiledCohesive &&
                    !bad.runtimeConfigurationFailure.empty(),
                "conflicting tiled selector admitted memory savings");
    }
    ShippingPrebuildArenaOptions noTile;
    configureShippingTiledCohesiveTrial(noTile, nullptr, "1", true, false);
    require(!noTile.runtimeConfigurationFailure.empty(),
            "page override without tiling was admitted");
    ShippingMemoryBudgetInput input;
    input.nodeCount = 24u;
    input.tetrahedronCount = 6u;
    input.cohesiveFaceCount = 260835u;
    input.contactPairCount = 1u;
    input.activeContactCapacity = 1u;
    input.deformableContactCapacity = 1u;
    input.rigidProxyCount = 2u;
    input.environmentCount = 1u;
    input.fullContactArmijoRequired = true;
    input.deformableSelfContactEnabled = true;
    const auto legacy = estimateShippingMemoryBudget(input);
    require(!legacy.requiredResidualDescentArenaAdmitted &&
                legacy.requiredCohesiveArmijoBytes == 260835ull * 480u,
            "legacy high-face preflight did not retain its original limit");
    input.fullContactArmijoTiledCohesive = true;
    const auto tiled = estimateShippingMemoryBudget(input);
    require(tiled.requiredResidualDescentArenaAdmitted &&
                tiled.requiredCohesiveArmijoBytes == 87381ull * 96u &&
                tiled.cohesiveRuntimeBytes == legacy.cohesiveRuntimeBytes,
            "tiled preflight omitted force-page or persistent cohesive accounting");
    input.environmentCount = 4u;
    const auto multi = estimateShippingMemoryBudget(input);
    require(multi.requiredResidualDescentArenaAdmitted &&
                multi.requiredCohesiveArmijoBytes == 4ull * 21845u * 96u,
            "tiled preflight squared or omitted the environment stride");
    input.cohesiveTrialPageCapacity = 1u;
    require(estimateShippingMemoryBudget(input).requiredCohesiveArmijoBytes ==
                4ull * 96u,
            "explicit one-face trial page did not match runtime storage");
    input.nodeCount = kShippingFullContactArmijoNodeLimit + 1u;
    require(!estimateShippingMemoryBudget(input).requiredResidualDescentArenaAdmitted,
            "cohesive tiling bypassed the independent full-contact node cap");
    input.nodeCount = 24u;
    input.environmentCount = 1u;
    input.cohesiveTrialPageCapacity = 87382u;
    require(!estimateShippingMemoryBudget(input).requiredResidualDescentArenaAdmitted,
            "oversized force page bypassed the runtime 8 MiB cap");
    input.cohesiveTrialPageCapacity = 0u;
    input.fullContactArmijoAliasesCohesiveScratch = true;
    expectFailure([&] { (void)estimateShippingMemoryBudget(input); },
                  "conflicting cohesive alias and tiling were admitted");
}

void run() {
    checkTiledCohesivePreflight();
    checkRuntimeEnvironmentAdmission();
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
    const std::uint64_t expectedPrebuildSourceBytes = shippingMemoryCheckedAdd(
        shippingMemoryCheckedMultiply(
            shippingPrebuildRigidNodeUpperBound(prebuild), 256u),
        shippingMemoryCheckedMultiply(
            shippingPrebuildTetrahedronUpperBound(prebuild), 512u));
    const std::uint64_t expectedPrebuildCompiledBytes = shippingMemoryCheckedAdd(
        shippingMemoryCheckedMultiply(
            shippingPrebuildRigidNodeUpperBound(prebuild), 128u),
        shippingMemoryCheckedMultiply(
            shippingPrebuildTetrahedronUpperBound(prebuild), 256u));
    const std::uint64_t expectedPrebuildHostReserve = shippingMemoryCheckedAdd(
        shippingMemoryCheckedAdd(expectedPrebuildSourceBytes,
                                 expectedPrebuildCompiledBytes),
        64ull * 1024ull * 1024ull);
    require(preflightOnlyBudget.hostMeshAndCompileBytes ==
                expectedPrebuildHostReserve &&
            preflightOnlyBudget.uploadStagingBytes ==
                expectedPrebuildCompiledBytes &&
            preflightOnlyBudget.hostCheckpointBytes ==
                2u * preflightOnlyBudget.snapshotReadbackBytes &&
            preflightOnlyBudget.estimatedPeakDomainBytes ==
                preflightOnlyBudget.estimatedMetalPeakBytes +
                    preflightOnlyBudget.hostMeshAndCompileBytes +
                    preflightOnlyBudget.hostCheckpointBytes,
            "prebuild planning reserve omitted or miscounted modeled host/source/compile allocations");
    // Build the exact shared shipping tooling composition used by the probe:
    // three score families, each with a capsule punch and two rail boxes,
    // followed by platen, floor, and impactor slots.
    GlueMeshConfig sectionConfig;
    sectionConfig.length = 1.025;
    sectionConfig.pitch = 0.0079;
    sectionConfig.caliper = 0.004210;
    sectionConfig.linerThickness = 0.000277;
    sectionConfig.mediumThickness = 0.000191;
    sectionConfig.glueMinimumGap = 0.000100;
    sectionConfig.bondWidth = 0.000800;
    sectionConfig.nxPerPitch = 4u;
    sectionConfig.thicknessSlices = 1u;
    sectionConfig.allowPartialTerminalPitch = true;
    const auto section = buildConformingGlueCrossSection(sectionConfig);
    FEFCO0201BlankConfig blankConfig;
    blankConfig.panelLength = 0.300;
    blankConfig.panelWidth = 0.200;
    blankConfig.wallHeight = 0.200;
    blankConfig.manufacturerJointWidth = 0.025;
    blankConfig.slotKerf = 0.001;
    blankConfig.subdivisionsPerBand = 1u;
    const auto blank = buildFEFCO0201Blank(blankConfig, section);
    const ShippingScoreToolDefaults scoreDefaults;
    const double pitchOverhang = 0.25 * sectionConfig.pitch;
    const auto tooling = makeSequentialBoxScoreTooling(
        blank, scoreDefaults.punchRadiusM, pitchOverhang,
        pitchOverhang, pitchOverhang, 0.001, pitchOverhang,
        scoreDefaults.channelWidthM, scoreDefaults.supportRailWidthM);
    const auto scoreToolBodies = makeBoxToolBodyStates(tooling, 1u, 0.050);
    const std::uint64_t observedProxyCount = tooling.proxies.size() +
        kShippingAdditionalRigidProxyCount;
    const std::uint64_t observedBodySlotCount = scoreToolBodies.size() +
        kShippingAdditionalBodySlotCount;
    require(tooling.families.size() == kShippingScoreToolFamilyCount &&
                tooling.proxies.size() == kShippingScoreToolProxyCount &&
                scoreToolBodies.size() == kShippingScoreToolSceneBodyCount &&
                observedProxyCount == kShippingRigidProxyCount &&
                observedBodySlotCount == kShippingRigidBodyCount,
            "prebuild shipping rigid counts diverged from executable score tooling and scene composition");

    ShippingPrebuildArenaOptions fullWorldOptions;
    fullWorldOptions.rigidProxyCount = observedProxyCount;
    fullWorldOptions.rigidBodyCount = observedBodySlotCount;
    fullWorldOptions.cohesiveFaceUpperBound =
        prebuild.nominalTetrahedronCount / 5u;
    fullWorldOptions.compactImmutableSurfaceMap = true;
    const auto fullDefaultCapacityBudget =
        estimateShippingPrebuildMemoryBudget(prebuild, fullWorldOptions);
    require(!fullDefaultCapacityBudget.withinLimit &&
                fullDefaultCapacityBudget.estimatedMetalPeakBytes <
                    kShippingPeakMemoryLimitBytes &&
                fullDefaultCapacityBudget.estimatedHostPeakBytes <
                    kShippingPeakMemoryLimitBytes &&
                fullDefaultCapacityBudget.estimatedPeakDomainBytes >
                    kShippingPeakMemoryLimitBytes,
            "twelve-proxy prebuild model did not add distinct Metal and host allocations");
    fullWorldOptions.requireFullContactResidualDescent = true;
    const auto requiredFullDefaultCapacityBudget =
        estimateShippingPrebuildMemoryBudget(prebuild, fullWorldOptions);
    require(!requiredFullDefaultCapacityBudget.withinLimit &&
                !requiredFullDefaultCapacityBudget.requiredResidualDescentArenaAdmitted &&
                requiredFullDefaultCapacityBudget.requiredRigidContactArmijoBytes > 0u,
            "shipping estimate admitted required full-contact arenas beyond the native pair cap");
    require(requiredFullDefaultCapacityBudget.rigidContactPairDomainUpperBound ==
                kShippingReferencePreparedNodes * observedProxyCount &&
                requiredFullDefaultCapacityBudget.requiredFullContactArmijoBytes ==
                    7'260'122'128ull &&
                requiredFullDefaultCapacityBudget.estimatedPeakDomainBytes ==
                    31'115'178'266ull &&
                !requiredFullDefaultCapacityBudget.sparsePrivateContactRequested,
            "dense default changed its prepared-node pair domain or complete private arena inventory");

    ShippingPrebuildArenaOptions sparsePrebuildOptions = fullWorldOptions;
    sparsePrebuildOptions.requireFullContactResidualDescent = true;
    sparsePrebuildOptions.deformableSelfContactEnabled = true;
    configureShippingPrebuildSparsePrivateContact(
        sparsePrebuildOptions, "65536", nullptr, true, true);
    const auto sparsePrebuildBudget = estimateShippingPrebuildMemoryBudget(
        prebuild, sparsePrebuildOptions);
    const auto preparedPairDomain = static_cast<std::uint32_t>(
        kShippingReferencePreparedNodes * observedProxyCount);
    const auto sparsePrebuildAdmission =
        numi::matter::detail::admitSparsePrivateContact(
            1u, preparedPairDomain, 65536u, false, true, true);
    require(sparsePrebuildBudget.rigidContactPairDomainUpperBound ==
                preparedPairDomain &&
                preparedPairDomain >= 527964u * observedProxyCount &&
                sparsePrebuildBudget.sparsePrivateContactRequested &&
                sparsePrebuildBudget.sparsePrivateContactCapacity == 65536u &&
                sparsePrebuildBudget.sparsePrivateContactBoundedFreeBoxScope &&
                sparsePrebuildBudget.privateRigidContactAdmitted &&
                sparsePrebuildBudget.sparsePrivateContactBytes ==
                    sparsePrebuildAdmission.totalBytes &&
                sparsePrebuildAdmission.admitted(),
            "prebuild sparse-private budget did not carry the runtime capacity, bounded layout, or conservative prepared-node pair domain");
    require(!sparsePrebuildBudget.requiredResidualDescentArenaAdmitted &&
                !sparsePrebuildBudget.withinLimit &&
                sparsePrebuildBudget.requiredFullContactArmijoBytes > 0u &&
                sparsePrebuildBudget.requiredCohesiveArmijoBytes > 0u,
            "sparse prebuild incorrectly implied that the shipping full-contact/cohesive topology caps are available");

    auto missingBoundedScope = sparsePrebuildOptions;
    missingBoundedScope.sparsePrivateContactBoundedFreeBoxScope = false;
    const auto missingScopeBudget = estimateShippingPrebuildMemoryBudget(
        prebuild, missingBoundedScope);
    require(!missingScopeBudget.privateRigidContactAdmitted &&
                !missingScopeBudget.requiredResidualDescentArenaAdmitted &&
                missingScopeBudget.privateRigidContactFailure.find(
                    "free-rigid residual is limited") != std::string::npos,
            "sparse free-body prebuild admitted without the exact bounded BOX scope candidate");

    auto overflowCapacity = sparsePrebuildOptions;
    overflowCapacity.sparsePrivateContactCapacity =
        preparedPairDomain + 1ull;
    const auto overflowCapacityBudget = estimateShippingPrebuildMemoryBudget(
        prebuild, overflowCapacity);
    require(!overflowCapacityBudget.privateRigidContactAdmitted &&
                !overflowCapacityBudget.requiredResidualDescentArenaAdmitted,
            "prebuild sparse candidate capacity larger than its pair domain was admitted");

    auto invalidRuntimeOption = sparsePrebuildOptions;
    invalidRuntimeOption.runtimeConfigurationFailure =
        "synthetic malformed sparse-capacity selector";
    const auto invalidRuntimeOptionBudget = estimateShippingPrebuildMemoryBudget(
        prebuild, invalidRuntimeOption);
    require(!invalidRuntimeOptionBudget.withinLimit &&
                !invalidRuntimeOptionBudget.requiredResidualDescentArenaAdmitted &&
                invalidRuntimeOptionBudget.runtimeConfigurationFailure ==
                    invalidRuntimeOption.runtimeConfigurationFailure &&
                invalidRuntimeOptionBudget.requiredResidualDescentArenaFailure ==
                    invalidRuntimeOption.runtimeConfigurationFailure,
            "malformed sparse runtime selection did not fail closed in the prebuild screen");

    fullWorldOptions.deformableContactCapacity = 1048576u;
    const auto explicitCapacityBudget =
        estimateShippingPrebuildMemoryBudget(prebuild, fullWorldOptions);
    require(!explicitCapacityBudget.withinLimit &&
                !explicitCapacityBudget.requiredResidualDescentArenaAdmitted &&
                explicitCapacityBudget.estimatedPeakDomainBytes <
                    requiredFullDefaultCapacityBudget.estimatedPeakDomainBytes &&
                explicitCapacityBudget.rigidContactRuntimeBytes > 0u,
            "explicit deformable capacity did not reduce the estimate while preserving the rigid-pair rejection");
    const auto longer = screenShippingPrebuildMemory(10.0, 0.0079);
    require(longer.estimatedNodeCount > prebuild.estimatedNodeCount &&
                longer.estimatedTetrahedronCount > prebuild.estimatedTetrahedronCount,
            "prebuild screen did not grow with requested flute intervals");
    require(shippingPrebuildRigidNodeUpperBound(longer) ==
                longer.estimatedNodeCount &&
                shippingPrebuildTetrahedronUpperBound(longer) ==
                    longer.estimatedTetrahedronCount,
            "non-reference prebuild arena bounds did not use padded topology counts");
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
    shipping.rigidBodyCount = kShippingRigidBodyCount;
    shipping.rigidProxyCount = kShippingRigidProxyCount;
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
                    budget.estimatedMetalPeakBytes +
                        budget.hostMeshAndCompileBytes +
                        budget.hostCheckpointBytes &&
                budget.withinLimit,
            "shipping arena budget omitted measured baseline, contact, cohesive, or checkpoint arenas");

    // A modeled pre-allocation budget must add separate live host and Metal
    // allocations even when neither subtotal alone exceeds the device limit.
    ShippingMemoryBudgetInput distinctDomains;
    distinctDomains.nodeCount = 1000u;
    distinctDomains.tetrahedronCount = 2000u;
    const auto metalOnlyBudget = estimateShippingMemoryBudget(distinctDomains);
    distinctDomains.hostSourceMeshBytes =
        kShippingPeakMemoryLimitBytes - 1u;
    const auto distinctDomainsBudget =
        estimateShippingMemoryBudget(distinctDomains);
    require(metalOnlyBudget.estimatedMetalPeakBytes <
                kShippingPeakMemoryLimitBytes &&
                distinctDomainsBudget.estimatedMetalPeakBytes <
                    kShippingPeakMemoryLimitBytes &&
                distinctDomainsBudget.estimatedHostPeakBytes <
                    kShippingPeakMemoryLimitBytes &&
                distinctDomainsBudget.estimatedPeakDomainBytes >
                    kShippingPeakMemoryLimitBytes &&
                !distinctDomainsBudget.withinLimit,
            "distinct sub-limit Metal and host allocations were not rejected after summation");

    // Staging and one readback reserve are reported in both side subtotals,
    // but the combined inventory adds them only through the Metal envelope.
    ShippingMemoryBudgetInput sharedReserves;
    sharedReserves.nodeCount = 1000u;
    sharedReserves.tetrahedronCount = 2000u;
    sharedReserves.hostSourceMeshBytes = 40ull * 1024ull * 1024ull;
    sharedReserves.hostCompiledWorldBytes = 20ull * 1024ull * 1024ull;
    sharedReserves.uploadStagingBytes = 20ull * 1024ull * 1024ull;
    sharedReserves.estimatedSnapshotPayloadBytes = 8ull * 1024ull * 1024ull;
    sharedReserves.retainedCompleteCheckpointCount = 2u;
    const auto sharedReservesBudget =
        estimateShippingMemoryBudget(sharedReserves);
    require(sharedReservesBudget.estimatedHostPeakBytes ==
                sharedReservesBudget.hostMeshAndCompileBytes +
                    sharedReservesBudget.hostCheckpointBytes +
                    sharedReservesBudget.uploadStagingBytes +
                    sharedReservesBudget.snapshotReadbackBytes &&
                sharedReservesBudget.estimatedPeakDomainBytes ==
                    shippingCombinedAllocationPeakBytes(sharedReservesBudget) &&
                sharedReservesBudget.estimatedPeakDomainBytes ==
                    sharedReservesBudget.estimatedMetalPeakBytes +
                    sharedReservesBudget.hostMeshAndCompileBytes +
                    sharedReservesBudget.hostCheckpointBytes &&
                sharedReservesBudget.estimatedPeakDomainBytes <
                    sharedReservesBudget.estimatedMetalPeakBytes +
                        sharedReservesBudget.estimatedHostPeakBytes,
            "combined allocation estimate double-counted the shared staging/readback reserves");

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

    // Sparse private trials reduce only the trial sample arena. The canonical
    // pair domain remains dense and bounded to uint32 addressing; accepted
    // contact history and persistent surface/contact allocations stay intact.
    ShippingMemoryBudgetInput sparseFullContact;
    sparseFullContact.nodeCount = 128u;
    sparseFullContact.tetrahedronCount = 64u;
    sparseFullContact.cohesiveFaceCount = 1u;
    sparseFullContact.contactPairCount = 100000u;
    sparseFullContact.activeContactCapacity = 1024u;
    sparseFullContact.deformableContactCapacity = 512u;
    sparseFullContact.rigidGeneralizedCapacity = 6u;
    sparseFullContact.rigidProxyCount = 3u;
    sparseFullContact.environmentCount = 1u;
    sparseFullContact.fullContactArmijoRequired = true;
    sparseFullContact.deformableSelfContactEnabled = true;
    sparseFullContact.sparsePrivateContactRequested = true;
    sparseFullContact.sparsePrivateContactCapacity = 1024u;
    sparseFullContact.sparsePrivateContactBoundedFreeBoxScope = true;
    sparseFullContact.fullContactArmijoAliasesSurfaceScratch = true;
    sparseFullContact.fullContactArmijoAliasesCohesiveScratch = true;
    const auto sparseBudget = estimateShippingMemoryBudget(sparseFullContact);
    require(sparseBudget.privateRigidContactAdmitted,
            "valid bounded sparse private arena was rejected by rigid-pair admission");
    if (!sparseBudget.requiredResidualDescentArenaAdmitted)
        throw std::runtime_error(
            "valid bounded sparse private arena failed the required full-contact budget: " +
            sparseBudget.requiredResidualDescentArenaFailure);
    require(sparseBudget.sparsePrivateContactBytes > 0u &&
                sparseBudget.sparsePrivateContactBytes <=
                    numi::matter::detail::kSparsePrivateContactScratchBudgetBytes &&
                sparseBudget.persistentContactBytes > 0u &&
                sparseBudget.cohesiveRuntimeBytes > 0u &&
                sparseBudget.requiredFullContactArmijoBytes > 0u,
            "sparse budget omitted private or persistent arena bytes");
    auto unaliasedSparse = sparseFullContact;
    unaliasedSparse.fullContactArmijoAliasesSurfaceScratch = false;
    unaliasedSparse.fullContactArmijoAliasesCohesiveScratch = false;
    const auto unaliasedSparseBudget =
        estimateShippingMemoryBudget(unaliasedSparse);
    require(unaliasedSparseBudget.persistentContactBytes ==
                sparseBudget.persistentContactBytes &&
                unaliasedSparseBudget.cohesiveRuntimeBytes ==
                    sparseBudget.cohesiveRuntimeBytes &&
                unaliasedSparseBudget.requiredFullContactArmijoBytes >
                    sparseBudget.requiredFullContactArmijoBytes &&
                unaliasedSparseBudget.requiredCohesiveArmijoBytes > 0u &&
                sparseBudget.requiredCohesiveArmijoBytes == 0u,
            "private scratch aliases incorrectly changed persistent arenas or were ignored");
    // The former 480-byte slot model admitted this two-environment arena.
    // Its per-candidate status records add 48 bytes per slot, putting the
    // actual private allocation above the unchanged 128 MiB limit.
    auto oversizedPrivateContact = sparseFullContact;
    oversizedPrivateContact.environmentCount = 2u;
    oversizedPrivateContact.deformableContactCapacity = 131072u;
    const auto oversizedPrivateBudget =
        estimateShippingMemoryBudget(oversizedPrivateContact);
    require(oversizedPrivateBudget.requiredFullContactArmijoBytes ==
                138416192u &&
                !oversizedPrivateBudget.requiredResidualDescentArenaAdmitted,
            "private candidate status storage was omitted from shipping admission");
    oversizedPrivateContact.fullContactArmijoEnabled = true;
    expectFailure([&] {
        (void)estimateShippingMemoryBudget(oversizedPrivateContact);
    }, "enabled private contact arena exceeded 128 MiB without rejection");
    auto noFreeBoxException = sparseFullContact;
    noFreeBoxException.sparsePrivateContactBoundedFreeBoxScope = false;
    const auto rejectedSparseFreeBody = estimateShippingMemoryBudget(noFreeBoxException);
    require(!rejectedSparseFreeBody.privateRigidContactAdmitted &&
                !rejectedSparseFreeBody.requiredResidualDescentArenaAdmitted,
            "sparse free-rigid allocation was admitted without the exact bounded BOX scope");
    auto sparseParityOverflow = sparseFullContact;
    sparseParityOverflow.contactPairCount = 250000u;
    sparseParityOverflow.sparsePrivateContactParity = true;
    const auto rejectedSparseParity = estimateShippingMemoryBudget(sparseParityOverflow);
    require(!rejectedSparseParity.privateRigidContactAdmitted,
            "parity reference arena exceeded the sparse byte cap without rejection");

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
    requiredButUnselected.rigidProxyCount = kShippingRigidProxyCount;
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
    const auto fullContactNodeAdmission = [](const std::uint64_t nodes,
                                             const bool selfContact = true) {
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
        input.deformableSelfContactEnabled = selfContact;
        return estimateShippingMemoryBudget(input);
    };
    const auto atFullContactNodeLimit = fullContactNodeAdmission(
        kShippingFullContactArmijoNodeLimit);
    require(atFullContactNodeLimit.requiredResidualDescentArenaAdmitted,
            "full-contact memory admission rejected the runtime's 8,192-node boundary");
    const auto withoutSelfContact = fullContactNodeAdmission(
        kShippingFullContactArmijoNodeLimit, false);
    require(!withoutSelfContact.requiredResidualDescentArenaAdmitted,
            "full-contact memory admission ignored the required self-contact flag");
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
