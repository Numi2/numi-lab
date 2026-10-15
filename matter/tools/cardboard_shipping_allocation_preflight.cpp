// Exact CPU geometry and bounded-arena preflight for the current shipping
// blank. This deliberately does not initialize Metal or advance FEM state.

#include "cardboard_box_assembly_controller.hpp"
#include "cardboard_shipping_closure_mesh.hpp"
#include "cardboard_shipping_driver.hpp"
#include "cardboard_shipping_memory.hpp"
#include "cardboard_starch_interface_mesh.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace numi::cardboard;
using Face = std::array<std::uint32_t, 3>;

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::uint64_t regionalRAPBytes(const std::uint64_t nodeCount,
                               const std::uint64_t modeCount,
                               const std::uint64_t environmentCount,
                               const bool blockFine) {
    const std::uint64_t rapScalars =
        3u * nodeCount * modeCount +
        environmentCount * modeCount * modeCount +
        4u * environmentCount * modeCount + environmentCount;
    const std::uint64_t rapBytes = rapScalars * sizeof(nm_float4);
    const std::uint64_t blockFineBytes = blockFine
        ? 3u * environmentCount * nodeCount * sizeof(nm_float4)
        : 0u;
    return rapBytes + blockFineBytes;
}

ShippingMemoryBudget estimateScenario(const std::uint64_t nodeCount,
    const std::uint64_t tetrahedronCount,
    const std::uint64_t cohesiveFaceCount,
    const std::uint64_t pairCount,
    const std::uint64_t surfaceFaceCount,
    const std::uint64_t deformableCapacity,
    const bool compactSurfaceMap,
    const std::uint64_t materialStateStride,
    const bool sparsePrivate = false,
    const bool freeRigidUnknowns = true,
    const bool aliasSurfaceScratch = false,
    const bool aliasCohesiveScratch = false,
    const bool boundedOneFreeBoxScope = false) {
    ShippingMemoryBudgetInput input;
    input.nodeCount = nodeCount;
    input.tetrahedronCount = tetrahedronCount;
    input.cohesiveFaceCount = cohesiveFaceCount;
    input.contactPairCount = pairCount;
    input.activeContactCapacity = pairCount;
    input.deformableContactCapacity = deformableCapacity;
    input.surfaceFaceCount = compactSurfaceMap ? surfaceFaceCount : 0u;
    input.compactImmutableSurfaceMap = compactSurfaceMap;
    input.rigidGeneralizedCapacity = freeRigidUnknowns ? 6u : 0u;
    input.rigidBodyCount = kShippingRigidBodyCount;
    input.rigidProxyCount = kShippingRigidProxyCount;
    input.materialStateStride = materialStateStride;
    input.compactPreconditionerRegionCount = 3u;
    input.environmentCount = 1u;
    input.fgmresRestart = kShippingFgmresRestartCeiling;
    input.fullContactArmijoRequired = true;
    input.deformableSelfContactEnabled = true;
    input.sparsePrivateContactRequested = sparsePrivate;
    input.sparsePrivateContactCapacity = sparsePrivate ? 65536u : 0u;
    // This is an explicit scenario assumption, not inferred from the number
    // of generalized rows. Runtime still validates the full proxy/body layout.
    input.sparsePrivateContactBoundedFreeBoxScope = boundedOneFreeBoxScope;
    input.fullContactArmijoAliasesSurfaceScratch = aliasSurfaceScratch;
    input.fullContactArmijoAliasesCohesiveScratch = aliasCohesiveScratch;
    // Match the coarse host-side reserve used by the probe's prebuild
    // admission. These are source-level planning factors, not measured vector
    // capacities or an observed process footprint.
    input.hostSourceMeshBytes = shippingMemoryCheckedAdd(
        shippingMemoryCheckedMultiply(nodeCount, 256u),
        shippingMemoryCheckedMultiply(tetrahedronCount, 512u));
    input.hostCompiledWorldBytes = shippingMemoryCheckedAdd(
        shippingMemoryCheckedMultiply(nodeCount, 128u),
        shippingMemoryCheckedMultiply(tetrahedronCount, 256u));
    input.hostGeneratedMetalBytes = 64ull * 1024ull * 1024ull;
    input.uploadStagingBytes = input.hostCompiledWorldBytes;
    input.retainedCompleteCheckpointCount = 2u;
    input.estimatedSnapshotPayloadBytes =
        estimateShippingSnapshotPayloadBytes(input);
    return estimateShippingMemoryBudget(input);
}

void writeMemoryScenario(const char* name,
                         const ShippingMemoryBudget& budget,
                         const std::uint64_t snapshotBytes,
                         const std::uint64_t checkpointCount,
                         const std::uint64_t tangentCacheBytes,
                         const bool boundedOneFreeBoxScope = false) {
    std::cout << "    \"" << name << "\": {"
        << "\"modeled_metal_envelope_bytes\":" << budget.estimatedMetalPeakBytes
        << ",\"modeled_peak_domain_bytes\":" << budget.estimatedPeakDomainBytes
        << ",\"host_mesh_and_compile_bytes\":"
        << budget.hostMeshAndCompileBytes
        << ",\"host_checkpoint_payload_bytes\":"
        << budget.hostCheckpointBytes
        << ",\"modeled_host_subtotal_with_shared_reserves_bytes\":"
        << budget.estimatedHostPeakBytes
        << ",\"shared_upload_and_readback_reserves_counted_once\":true"
        << ",\"static_fem_solver_floor_bytes\":" << budget.staticFEMSolverFloorBytes
        << ",\"measured_scaled_noncontact_envelope_bytes\":"
        << budget.measuredNoncontactRuntimeEnvelopeBytes
        << ",\"persistent_contact_surface_bytes\":"
        << budget.persistentContactSurfaceBytes
        << ",\"persistent_deformable_slots_bytes\":"
        << budget.persistentContactSlotBytes
        << ",\"persistent_contact_auxiliary_bytes\":"
        << budget.persistentContactAuxiliaryBytes
        << ",\"persistent_rigid_contact_bytes\":"
        << budget.rigidContactRuntimeBytes
        << ",\"persistent_cohesive_bytes\":" << budget.cohesiveRuntimeBytes
        << ",\"required_private_full_contact_bytes\":"
        << budget.requiredFullContactArmijoBytes
        << ",\"required_private_rigid_contact_bytes\":"
        << budget.requiredRigidContactArmijoBytes
        << ",\"required_private_cohesive_bytes\":"
        << budget.requiredCohesiveArmijoBytes
        << ",\"sparse_private_contact_bytes\":"
        << budget.sparsePrivateContactBytes
        << ",\"bounded_one_free_box_six_dof_scope_assumed\":"
        << (boundedOneFreeBoxScope ? "true" : "false")
        << ",\"bounded_one_free_box_scope_note\":\""
        << (boundedOneFreeBoxScope
                ? "planning assumption only; runtime must validate one non-articulated free BOX and remaining proxy/body eligibility"
                : "not assumed")
        << "\""
        << ",\"private_rigid_contact_admitted\":"
        << (budget.privateRigidContactAdmitted ? "true" : "false")
        << ",\"private_rigid_contact_failure\":\""
        << budget.privateRigidContactFailure << "\""
        << ",\"private_arenas_admitted\":"
        << (budget.requiredResidualDescentArenaAdmitted ? "true" : "false")
        << ",\"private_arena_failure\":\""
        << budget.requiredResidualDescentArenaFailure
        << "\",\"snapshot_one_copy_bytes\":" << snapshotBytes
        << ",\"snapshot_checkpoint_count\":" << checkpointCount
        << ",\"snapshot_retained_host_bytes\":"
        << shippingMemoryCheckedMultiply(snapshotBytes, checkpointCount)
        << ",\"modeled_metal_envelope_plus_tangent_cache_bytes\":"
        << budget.estimatedMetalPeakBytes + tangentCacheBytes
        << ",\"modeled_combined_allocation_plus_tangent_cache_bytes\":"
        << shippingMemoryCheckedAdd(budget.estimatedPeakDomainBytes,
                                    tangentCacheBytes)
        << ",\"combined_allocation_model_within_16_gib\":"
        << (budget.estimatedPeakDomainBytes <= budget.limitBytes
                ? "true" : "false")
        << ",\"combined_allocation_model_is_measurement\":false"
        << ",\"combined_allocation_plus_tangent_cache_within_16_gib\":"
        << (shippingMemoryCheckedAdd(budget.estimatedPeakDomainBytes,
                                     tangentCacheBytes) <= budget.limitBytes
                ? "true" : "false")
        << ",\"required_private_arenas_admitted\":"
        << (budget.requiredResidualDescentArenaAdmitted ? "true" : "false")
        << ",\"within_shipping_admission_model\":"
        << (budget.withinLimit ? "true" : "false")
        << "}";
}

void run() {
    ShippingBoxDesignDefaults defaults;
    defaults.subdivisionsPerBand = 16u;
    const std::array<double, 3> zeroEngineeringOffset{0.0, 0.0, 0.0};
    const auto selected = selectShippingDesign(
        defaults, nullptr, &zeroEngineeringOffset, false);
    ShippingCorrugatedRecipe recipe;
    recipe.nxPerPitch = 8u; // current shipping probe setting
    recipe.thicknessSlices = 1u;
    const auto geometry = buildShippingBoxBlankGeometry(selected.design, recipe);

    StarchInterfaceMesh starchSource;
    starchSource.referenceNodesM.reserve(geometry.blank.points.size());
    for (const auto& point : geometry.blank.points)
        starchSource.referenceNodesM.push_back({point.x, point.y, point.z});
    starchSource.tetrahedra.reserve(geometry.blank.tetrahedra.size());
    for (const auto& sourceTet : geometry.blank.tetrahedra) {
        StarchInterfaceTet tet;
        for (std::size_t corner = 0u; corner < 4u; ++corner)
            tet.nodeIds[corner] = sourceTet.nodes[corner];
        tet.materialIndex = static_cast<std::uint8_t>(sourceTet.material);
        // Density only validates the topology preparation call. It does not
        // affect face counts; actual material assets are read by the probe.
        tet.densityKgPerM3 = tet.materialIndex == kStarchMaterialLiner ? 766.0 :
            tet.materialIndex == kStarchMaterialMedium ? 679.0 : 1500.0;
        starchSource.tetrahedra.push_back(tet);
    }
    const auto starch = prepareStarchInterfaceMesh(starchSource);

    std::vector<BoxBlankPoint> closureNodes;
    closureNodes.reserve(starch.referenceNodesM.size());
    for (const auto& point : starch.referenceNodesM)
        closureNodes.push_back({point.x, point.y, point.z});
    std::vector<ShippingClosureTet> closureTets;
    closureTets.reserve(starch.tetrahedra.size());
    for (const auto& sourceTet : starch.tetrahedra) {
        ShippingClosureTet tet;
        for (std::size_t corner = 0u; corner < 4u; ++corner)
            tet.nodes[corner] = static_cast<std::uint32_t>(
                sourceTet.nodeIds[corner]);
        tet.materialIndex = sourceTet.materialIndex;
        closureTets.push_back(tet);
    }
    // The activation-gap scalar does not alter triangle enumeration. It is a
    // positive geometry-builder placeholder, not an HMA material input.
    const auto hma = buildShippingClosureMortarMesh(
        geometry.blank, closureNodes, closureTets, 0.010, 0.0001, 1u,
        static_cast<std::uint32_t>(starch.pairedFaces.size() + 1u));

    std::map<Face, std::uint32_t> incidence;
    for (const auto& tet : closureTets) {
        const auto n = tet.nodes;
        std::array<Face, 4> faces{{
            {n[0], n[1], n[2]}, {n[0], n[1], n[3]},
            {n[0], n[2], n[3]}, {n[1], n[2], n[3]}}};
        for (Face face : faces) {
            std::sort(face.begin(), face.end());
            ++incidence[face];
        }
    }
    std::set<std::uint32_t> contactNodes;
    std::uint64_t boundaryFaceCount = 0u;
    for (const auto& [face, count] : incidence) {
        require(count <= 2u, "prepared shipping topology is nonmanifold");
        if (count == 1u) {
            ++boundaryFaceCount;
            contactNodes.insert(face.begin(), face.end());
        }
    }

    const ShippingScoreToolDefaults toolDefaults;
    const double pitchOverhang = 0.25 * recipe.pitchM;
    const auto scoreTools = makeSequentialBoxScoreTooling(
        geometry.blank, toolDefaults.punchRadiusM, pitchOverhang,
        pitchOverhang, pitchOverhang, 0.001, pitchOverhang,
        toolDefaults.channelWidthM, toolDefaults.supportRailWidthM);
    const std::uint64_t proxyCount = scoreTools.proxies.size() +
        kShippingAdditionalRigidProxyCount;
    const std::uint64_t bodyCount = kShippingRigidBodyCount;
    const std::uint64_t pairCount = contactNodes.size() * proxyCount;
    const std::uint64_t contactPairCapacity = starch.referenceNodesM.size() *
        proxyCount;
    const std::uint64_t cohesiveFaceCount = starch.pairedFaces.size() +
        hma.patches.size();
    const std::uint64_t defaultDeformableCapacity =
        8u * starch.tetrahedra.size();
    const std::uint64_t tangentCacheBytes =
        81u * sizeof(float) * starch.tetrahedra.size();
    const std::uint64_t tangentCacheValidBytes =
        sizeof(std::uint32_t) * starch.tetrahedra.size();

    require(geometry.blank.points.size() == kShippingReferenceNodes &&
            geometry.blank.tetrahedra.size() == kShippingReferenceTetrahedra,
            "shipping source geometry no longer matches its exact cardinality anchor");
    require(starch.pairedFaces.size() == 97984u &&
            starch.duplicatedGlueNodeCount == 75066u &&
            starch.referenceNodesM.size() == 528444u,
            "starch interface topology differs from the reviewed shipping mesh");
    require(hma.patches.size() == 162851u && hma.groups.size() == 9u,
            "HMA closure topology differs from the reviewed shipping geometry");
    require(boundaryFaceCount == 1054876u && contactNodes.size() == 527964u,
            "exterior contact map differs from the reviewed shipping mesh");
    require(proxyCount == 12u && bodyCount == 9u,
            "shipping score/scene tool composition changed");

    const auto defaultDense = estimateScenario(
        starch.referenceNodesM.size(), starch.tetrahedra.size(), cohesiveFaceCount,
        pairCount, boundaryFaceCount, defaultDeformableCapacity, false, 15u);
    const auto compactDefaultCapacity = estimateScenario(
        starch.referenceNodesM.size(), starch.tetrahedra.size(), cohesiveFaceCount,
        pairCount, boundaryFaceCount, defaultDeformableCapacity, true, 15u);
    const auto boundedCompact = estimateScenario(
        starch.referenceNodesM.size(), starch.tetrahedra.size(), cohesiveFaceCount,
        pairCount, boundaryFaceCount, 131072u, true, 15u);
    const auto boundedUncompacted = estimateScenario(
        starch.referenceNodesM.size(), starch.tetrahedra.size(), cohesiveFaceCount,
        pairCount, boundaryFaceCount, 131072u, false, 15u);
    const auto sparseKinematicAliased = estimateScenario(
        starch.referenceNodesM.size(), starch.tetrahedra.size(), cohesiveFaceCount,
        pairCount, boundaryFaceCount, 131072u, true, 15u,
        true, false, true, true);
    const auto sparseKinematicUnaliased = estimateScenario(
        starch.referenceNodesM.size(), starch.tetrahedra.size(), cohesiveFaceCount,
        pairCount, boundaryFaceCount, 131072u, true, 15u,
        true, false, false, false);
    const auto sparseFreeRigid = estimateScenario(
        starch.referenceNodesM.size(), starch.tetrahedra.size(), cohesiveFaceCount,
        pairCount, boundaryFaceCount, 131072u, true, 15u,
        true, true, true, true, true);

    ShippingMemoryBudgetInput snapshotDimensions;
    snapshotDimensions.nodeCount = starch.referenceNodesM.size();
    snapshotDimensions.tetrahedronCount = starch.tetrahedra.size();
    snapshotDimensions.cohesiveFaceCount = cohesiveFaceCount;
    snapshotDimensions.contactPairCount = pairCount;
    snapshotDimensions.deformableContactCapacity = 131072u;
    snapshotDimensions.surfaceFaceCount = boundaryFaceCount;
    snapshotDimensions.compactImmutableSurfaceMap = true;
    snapshotDimensions.rigidGeneralizedCapacity = 6u;
    snapshotDimensions.rigidBodyCount = bodyCount;
    snapshotDimensions.rigidProxyCount = proxyCount;
    snapshotDimensions.compactPreconditionerRegionCount = 3u;
    snapshotDimensions.environmentCount = 1u;
    snapshotDimensions.fgmresRestart = kShippingFgmresRestartCeiling;
    const std::array<std::uint64_t, 3> stateStrides{13u, 15u, 30u};
    std::array<std::uint64_t, 3> snapshotBytes{};
    for (std::size_t index = 0u; index < stateStrides.size(); ++index) {
        snapshotDimensions.materialStateStride = stateStrides[index];
        snapshotBytes[index] = estimateShippingSnapshotPayloadBytes(snapshotDimensions);
    }

    constexpr std::uint64_t sparseCandidateCapacity = 65536u;
    const std::uint64_t sparseSampleAndIdBytes = shippingMemoryCheckedMultiply(
        sparseCandidateCapacity,
        sizeof(NMContactSampleGPU) + sizeof(std::uint32_t));
    const std::uint64_t sparsePairMapBytes = shippingMemoryCheckedMultiply(
        pairCount, sizeof(std::uint32_t));
    const auto sparseAdmissionNoFreeRows =
        numi::matter::detail::admitSparsePrivateContact(
        1u, static_cast<std::uint32_t>(pairCount),
        static_cast<std::uint32_t>(sparseCandidateCapacity), false, false);
    const auto sparseAdmissionFreeRows =
        numi::matter::detail::admitSparsePrivateContact(
        1u, static_cast<std::uint32_t>(pairCount),
        static_cast<std::uint32_t>(sparseCandidateCapacity), false, true);
    const auto sparseAdmissionBoundedFreeBox =
        numi::matter::detail::admitSparsePrivateContact(
            1u, static_cast<std::uint32_t>(pairCount),
            static_cast<std::uint32_t>(sparseCandidateCapacity),
            false, true, true);
    require(sparseAdmissionNoFreeRows.admitted(),
            "shared runtime sparse admission rejected the no-free-rigid K=65536 case");
    require(!sparseAdmissionFreeRows.admitted() &&
            sparseAdmissionFreeRows.error ==
                numi::matter::detail::SparsePrivateContactAdmissionError::free_rigid_unknowns_unsupported,
            "shared runtime sparse admission did not reject generic free rigid unknowns");
    require(sparseAdmissionBoundedFreeBox.admitted(),
            "shared runtime sparse admission rejected the bounded one-free-BOX six-DOF case");
    require(sparseFreeRigid.privateRigidContactAdmitted &&
            sparseFreeRigid.sparsePrivateContactBytes ==
                sparseAdmissionBoundedFreeBox.totalBytes,
            "bounded free-BOX scenario failed to include its sparse trial arena in the model");
    require(sparseKinematicAliased.requiredFullContactArmijoBytes <
                sparseKinematicUnaliased.requiredFullContactArmijoBytes &&
            sparseKinematicAliased.persistentContactBytes ==
                sparseKinematicUnaliased.persistentContactBytes &&
            sparseKinematicAliased.cohesiveRuntimeBytes ==
                sparseKinematicUnaliased.cohesiveRuntimeBytes,
            "private scratch aliasing changed persistent allocations or failed to reduce private trial memory");
    const std::uint64_t selector2RAPBytes = regionalRAPBytes(
        starch.referenceNodesM.size(), 18u, 1u, false);
    const std::uint64_t selector4RAPBytes = regionalRAPBytes(
        starch.referenceNodesM.size(), 32u, 1u, true);

    const std::uint64_t hostSourceMeshBytes = shippingMemoryCheckedAdd(
        shippingMemoryCheckedMultiply(starch.referenceNodesM.size(), 256u),
        shippingMemoryCheckedMultiply(starch.tetrahedra.size(), 512u));
    const std::uint64_t hostCompiledWorldBytes = shippingMemoryCheckedAdd(
        shippingMemoryCheckedMultiply(starch.referenceNodesM.size(), 128u),
        shippingMemoryCheckedMultiply(starch.tetrahedra.size(), 256u));
    constexpr std::uint64_t hostGeneratedMetalBytes =
        64ull * 1024ull * 1024ull;

    std::cout
        << "{\n"
        << "  \"schema\": \"numi.cardboard.shipping-allocation-preflight.v6\",\n"
        << "  \"allocation_accounting\": {\"combined_formula\": \"modeled Metal envelope + host mesh/compile allocations + retained checkpoints\", \"shared_upload_and_readback_reserves_counted_once\": true, \"host_subtotal_is_not_an_observed_process_metric\": true, \"observed_runtime_and_host_counters_compared_by_max\": true},\n"
        << "  \"evidence_class\": \"exact CPU-built authored geometry plus allocation formulas read from source; modeled envelope only, not actual Metal peak, solver qualification, or physical qualification\",\n"
        << "  \"host_reserve_assumptions\": {\"basis\": \"same coarse reserve formulas as estimateShippingPrebuildMemoryBudget in cardboard_shipping_memory.hpp\", \"source_mesh_bytes\": " << hostSourceMeshBytes
        << ", \"source_mesh_formula\": \"256*nodes + 512*tetrahedra\", \"compiled_world_bytes\": " << hostCompiledWorldBytes
        << ", \"compiled_world_formula\": \"128*nodes + 256*tetrahedra\", \"generated_metal_bytes\": " << hostGeneratedMetalBytes
        << ", \"upload_staging_bytes\": " << hostCompiledWorldBytes
        << ", \"staging_formula\": \"equal to compiled-world reserve\", \"host_reserve_is_measurement\": false, \"capacity_accuracy\": \"coarse explicit planning assumption; actual vector capacities, allocator overhead, generated source/compiler peaks, and host/device peak remain unmeasured\"},\n"
        << "  \"source_anchor\": {\"base_nodes\": " << kShippingReferenceNodes
        << ", \"base_tetrahedra\": " << kShippingReferenceTetrahedra
        << ", \"source032_measured_runtime_resident_bytes\": "
        << kShippingCase032RuntimeResidentBytes
        << ", \"source032_is_full_shipping_world\": false},\n"
        << "  \"design\": {\"inside_m\": [0.3, 0.2, 0.2], \"zero_engineering_score_allowance_used_for_cardinality_only\": true, \"blank_perimeter_m\": 1.025, \"flute_pitch_m\": 0.0079, \"nx_per_pitch\": 8, \"y_intervals\": 48, \"thickness_slices\": 1, \"HMA_strip_width_m\": 0.01, \"HMA_builder_gap_placeholder_m\": 0.0001, \"HMA_gap_is_not_a_material_or_cure_calibration\": true},\n"
        << "  \"configuration_defaults\": {\"--compact-immutable-surfaces_default\": false, \"--deformable-contact-capacity_default\": \"legacy 8 slots per tetrahedron\", \"NM_FEM_FULL_CONTACT_REUSE_SELF_SURFACE_SCRATCH_default\": false, \"NM_FEM_FULL_CONTACT_REUSE_COHESIVE_SCRATCH_default\": false, \"recommended_cardinality_screen_requires_compact_flag\": true},\n"
        << "  \"topology\": {\"base_nodes\": " << geometry.blank.points.size()
        << ", \"tetrahedra\": " << starch.tetrahedra.size()
        << ", \"duplicated_glue_nodes\": " << starch.duplicatedGlueNodeCount
        << ", \"prepared_nodes\": " << starch.referenceNodesM.size()
        << ", \"starch_interface_faces_if_explicit_law_enabled\": " << starch.pairedFaces.size()
        << ", \"HMA_closure_faces_if_explicit_law_enabled\": " << hma.patches.size()
        << ", \"HMA_closure_groups\": " << hma.groups.size()
        << ", \"cohesive_faces_total_if_both_laws_enabled\": " << cohesiveFaceCount
        << ", \"exterior_faces_after_starch_split\": " << boundaryFaceCount
        << ", \"contact_eligible_exterior_nodes\": " << contactNodes.size() << "},\n"
        << "  \"tooling\": {\"score_tool_proxies\": " << scoreTools.proxies.size()
        << ", \"total_proxies_including_platen_floor_impactor\": " << proxyCount
        << ", \"scene_body_slots\": " << bodyCount << "},\n"
        << "  \"capacities_and_private_arrays\": {\"rigid_pair_rows\": " << pairCount
        << ", \"compiled_pair_upper_bound_from_all_prepared_nodes\": " << contactPairCapacity
        << ", \"deformable_capacity_default_8_per_tet\": " << defaultDeformableCapacity
        << ", \"deformable_capacity_bounded_candidate\": 131072"
        << ", \"tangent_cache_bytes_81_f32_plus_valid_u32_per_tet\": "
        << tangentCacheBytes + tangentCacheValidBytes
        << ", \"tangent_cache_values_bytes\": " << tangentCacheBytes
        << ", \"tangent_cache_validity_bytes\": " << tangentCacheValidBytes
        << ", \"dense_private_box_armijo_estimate_bytes\": "
        << boundedCompact.requiredRigidContactArmijoBytes
        << ", \"sparse_private_K65536_sample_plus_ID_bytes\": "
        << sparseSampleAndIdBytes
        << ", \"sparse_private_4P_pair_map_bytes\": " << sparsePairMapBytes
        << ", \"sparse_private_K65536_total_bytes_without_parity\": "
        << sparseAdmissionNoFreeRows.totalBytes
        << ", \"sparse_candidate_capacity\": " << sparseCandidateCapacity
        << ", \"dense_private_pair_limit\": "
        << numi::matter::detail::kDensePrivateContactPairLimit
        << ", \"dense_candidate_rejected_for_shipping_pair_domain\": "
        << (!numi::matter::detail::densePrivateContactPairDomainSupported(
                static_cast<std::uint32_t>(pairCount)) ? "true" : "false")
        << ", \"sparse_runtime_helper_admitted_without_free_rigid_rows\": "
        << (sparseAdmissionNoFreeRows.admitted() ? "true" : "false")
        << ", \"sparse_runtime_helper_failure_without_free_rigid_rows\": \""
        << numi::matter::detail::sparsePrivateContactAdmissionErrorText(
                sparseAdmissionNoFreeRows.error)
        << "\", \"sparse_runtime_helper_admitted_with_generic_six_free_rigid_rows\": "
        << (sparseAdmissionFreeRows.admitted() ? "true" : "false")
        << ", \"sparse_runtime_helper_failure_with_generic_six_free_rigid_rows\": \""
        << numi::matter::detail::sparsePrivateContactAdmissionErrorText(
                sparseAdmissionFreeRows.error)
        << "\", \"sparse_runtime_helper_admitted_with_bounded_one_free_BOX_six_DOF_scope\": "
        << (sparseAdmissionBoundedFreeBox.admitted() ? "true" : "false")
        << ", \"sparse_runtime_helper_failure_with_bounded_one_free_BOX_six_DOF_scope\": \""
        << numi::matter::detail::sparsePrivateContactAdmissionErrorText(
                sparseAdmissionBoundedFreeBox.error)
        << "\"},\n"
        << "  \"current_admission\": {\"node_limit\": 8192, \"nodes_rejected\": "
        << (starch.referenceNodesM.size() > 8192u ? "true" : "false")
        << ", \"tetrahedron_limit\": 32768, \"tetrahedra_rejected\": "
        << (starch.tetrahedra.size() > 32768u ? "true" : "false")
        << ", \"dense_private_pair_limit\": "
        << numi::matter::detail::kDensePrivateContactPairLimit
        << ", \"dense_pairs_rejected\": "
        << (!numi::matter::detail::densePrivateContactPairDomainSupported(
                static_cast<std::uint32_t>(pairCount)) ? "true" : "false")
        << ", \"sparse_pair_count_limit_65536_exists\": false"
        << ", \"cohesive_face_limit\": 4096, \"cohesive_faces_rejected_if_both_laws\": "
        << (cohesiveFaceCount > 4096u ? "true" : "false")
        << ", \"default_deformable_capacity_limit\": 131072, \"default_capacity_rejected\": "
        << (defaultDeformableCapacity > 131072u ? "true" : "false")
        << ", \"explicit_131072_capacity_passes_its_capacity_gate\": true"
        << ", \"proxy_limit\": 16, \"proxy_count_rejected\": false"
        << ", \"cohesive_full_contact_private_cap_bytes\": 8388608"
        << ", \"rigid_full_contact_private_cap_bytes\": 37748736"
        << ", \"FEM_full_contact_private_cap_bytes\": 134217728"
        << ", \"full_contact_admission_requires_deformable_self_contact\": true},\n"
        << "  \"memory_model\": {\n"
        << "    \"state_stride_snapshot_scenarios\": {\"environment_count\": 1, \"deformable_capacity\": 131072, \"contact_pair_count\": " << pairCount
        << ", \"13_state_snapshot_bytes\": " << snapshotBytes[0]
        << ", \"13_state_two_checkpoint_host_bytes\": " << 2u * snapshotBytes[0]
        << ", \"15_state_snapshot_bytes\": " << snapshotBytes[1]
        << ", \"15_state_two_checkpoint_host_bytes\": " << 2u * snapshotBytes[1]
        << ", \"30_state_snapshot_bytes\": " << snapshotBytes[2]
        << ", \"30_state_two_checkpoint_host_bytes\": " << 2u * snapshotBytes[2]
        << ", \"note\": \"current Haj-Ali bilinear material files use 15 states; 13 and 30 are stride sensitivities\"},\n"
        << "    \"modeled_scenarios\": {\n";
    writeMemoryScenario("default_capacity_compact_flag_off_stride15",
        defaultDense, snapshotBytes[1], 2u,
        tangentCacheBytes + tangentCacheValidBytes);
    std::cout << ",\n";
    writeMemoryScenario("default_capacity_compact_flag_on_stride15",
        compactDefaultCapacity, snapshotBytes[1], 2u,
        tangentCacheBytes + tangentCacheValidBytes);
    std::cout << ",\n";
    writeMemoryScenario("capacity_131072_compact_flag_on_stride15_best_case",
        boundedCompact, snapshotBytes[1], 2u,
        tangentCacheBytes + tangentCacheValidBytes);
    std::cout << ",\n";
    writeMemoryScenario("capacity_131072_compact_flag_off_stride15",
        boundedUncompacted, snapshotBytes[1], 2u,
        tangentCacheBytes + tangentCacheValidBytes);
    std::cout << ",\n";
    writeMemoryScenario("sparse_K65536_no_free_rigid_on_stride15",
        sparseKinematicAliased, snapshotBytes[1], 2u,
        tangentCacheBytes + tangentCacheValidBytes);
    std::cout << ",\n";
    writeMemoryScenario("sparse_K65536_no_free_rigid_off_stride15",
        sparseKinematicUnaliased, snapshotBytes[1], 2u,
        tangentCacheBytes + tangentCacheValidBytes);
    std::cout << ",\n";
    writeMemoryScenario("sparse_K65536_one_free_BOX_six_DOF_on_stride15",
        sparseFreeRigid, snapshotBytes[1], 2u,
        tangentCacheBytes + tangentCacheValidBytes, true);
    std::cout
        << "\n    },\n"
        << "    \"regional_preconditioner_selector_estimates\": {"
        << "\"selector_2_modes_18_bytes\":" << selector2RAPBytes
        << ",\"selector_2_runtime_cap_bytes\":268435456"
        << ",\"selector_2_cap_rejected\":"
        << (selector2RAPBytes > 268435456u ? "true" : "false")
        << ",\"selector_4_modes_32_plus_blockfine_bytes\":" << selector4RAPBytes
        << ",\"selector_4_runtime_cap_bytes\":268435456"
        << ",\"selector_4_cap_rejected\":"
        << (selector4RAPBytes > 268435456u ? "true" : "false")
        << "},\n"
        << "    \"measured_source032_runtime_resident_bytes\":"
        << kShippingCase032RuntimeResidentBytes
        << ",\"measurement_scope\":\"earlier base FEM source032 without shipping tools or cohesive faces; not the present full-world model\"\n"
        << "  },\n"
        << "  \"known_admission_blockers\": [\"node/tet/cohesive count gates remain exceeded for the full shipping topology\", \"default 8T deformable contact slots exceed cap; explicit 131072 only clears this one count gate\", \"dense private contact rejects the current pair domain; sparse K=65536 models an exact one non-articulated free BOX six-DOF scope, subject to runtime proxy/body validation\", \"full-contact FEM and cohesive count/arena gates remain exceeded\", \"selector 2 and selector 4 RAP arenas exceed 256 MiB\"],\n"
        << "  \"limits\": {\"memory_peak_is_model_not_actual_metal_allocation\": true, \"host_reserve_is_coarse_not_measured\": true, \"scratch_aliases_reduce_private_estimates_only\": true, \"persistent_allocations_and_runtime_count_caps_are_not_reduced_by_aliases\": true, \"generic_sparse_free_rigid_scope_is_rejected\": true, \"bounded_sparse_free_BOX_layout_is_only_a_modeled_assumption_until_runtime_validation\": true, \"no_solver_step_or_physical_qualification\": true}\n"
        << "}\n";
}

} // namespace

int main() {
    try {
        run();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cardboard_shipping_allocation_preflight: "
                  << error.what() << '\n';
        return 2;
    }
}
