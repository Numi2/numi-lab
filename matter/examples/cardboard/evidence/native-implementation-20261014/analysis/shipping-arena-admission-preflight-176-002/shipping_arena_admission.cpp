#include "cardboard_shipping_memory.hpp"
#include "numi/matter/detail/full_contact_armijo_scratch_budget.hpp"
#include "cardboard_cohesive_armijo_tiles.hpp"
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace numi::cardboard;
using namespace numi::matter::detail;

struct BufferRow {
    std::string name;
    std::uint64_t count = 0;
    std::uint64_t itemBytes = 0;
    std::uint64_t bytes = 0;
};
static void require(bool okay, const char* message) {
    if (!okay) throw std::runtime_error(message);
}
static std::uint64_t allocBytes(std::uint64_t count, std::uint64_t item) {
    if (!item || count > std::numeric_limits<std::size_t>::max() / item)
        throw std::overflow_error("typed allocation product overflows size_t");
    const auto logical = count * item;
    return std::max<std::uint64_t>({logical, item, 16u});
}
static std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    return shippingMemoryCheckedMultiply(a,b);
}
static std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    return shippingMemoryCheckedAdd(a,b);
}
static void row(std::vector<BufferRow>& rows, const char* name,
                std::uint64_t count, std::uint64_t item) {
    rows.push_back({name,count,item,allocBytes(count,item)});
}
static std::optional<std::uint64_t> parseMax(int argc, char** argv) {
    if (argc == 1) return std::nullopt;
    if (argc != 3 || std::string(argv[1]) != "--max-buffer-length")
        throw std::invalid_argument("usage: shipping-arena-admission [--max-buffer-length BYTES]");
    std::uint64_t value = 0;
    const std::string s(argv[2]);
    const auto r = std::from_chars(s.data(),s.data()+s.size(),value);
    if (r.ec != std::errc{} || r.ptr != s.data()+s.size() || value == 0)
        throw std::invalid_argument("max-buffer-length must be a positive uint64");
    return value;
}
static void emitRows(const std::vector<BufferRow>& rows,
                     std::optional<std::uint64_t> maxBytes) {
    std::cout << "[";
    for (std::size_t i=0;i<rows.size();++i) {
        if (i) std::cout << ",";
        const auto& b=rows[i];
        std::cout << "{\"name\":\"" << b.name << "\",\"elements\":" << b.count
                  << ",\"element_bytes\":" << b.itemBytes
                  << ",\"metal_checked_bytes\":" << b.bytes
                  << ",\"last_byte_exclusive\":" << b.bytes
                  << ",\"linear_index_max\":";
        if (b.count == 0u) std::cout << "null";
        else std::cout << (b.count - 1u);
        std::cout << ",\"u32_linear_index_fits\":"
                  << (b.count == 0u || b.count - 1u <= UINT32_MAX ? "true" : "false")
                  << ",\"max_buffer_length_status\":";
        if (!maxBytes) std::cout << "\"unmeasured\"";
        else std::cout << (b.bytes <= *maxBytes ? "\"within_supplied_limit\"" : "\"exceeds_supplied_limit\"");
        std::cout << "}";
    }
    std::cout << "]";
}
static std::vector<BufferRow> contactArmijoRows(
    std::uint64_t E, std::uint64_t P, std::uint64_t leaves,
    std::uint64_t C, std::uint64_t N, bool aliasSurface) {
    std::vector<BufferRow> rows;
    if (!aliasSurface) {
        row(rows,"surface_primitives",mul(E,P),sizeof(NMContinuumSurfacePrimitiveGPU));
        row(rows,"surface_bvh_bounds",mul(mul(E,leaves),4u),sizeof(nm_float4));
        for (const char* n : {"surface_sort_keys_a","surface_sort_keys_b","surface_sort_indices_a","surface_sort_indices_b"})
            row(rows,n,mul(E,P),sizeof(std::uint32_t));
        row(rows,"surface_active_counts",E,sizeof(std::uint32_t));
    }
    const auto EC=mul(E,C), EN=mul(E,N);
    row(rows,"deformable_candidates",EC,sizeof(NMDeformableContactCandidateGPU));
    row(rows,"candidate_counts",E,sizeof(std::uint32_t));
    row(rows,"candidate_statuses",EC,sizeof(NMMatterStatusGPU));
    row(rows,"deformable_contacts",EC,sizeof(NMDeformableContactGPU));
    row(rows,"deformable_active_indices",EC,sizeof(std::uint32_t));
    row(rows,"deformable_active_counts",E,sizeof(std::uint32_t));
    row(rows,"node_incidence",mul(EC,6u),sizeof(std::uint32_t));
    row(rows,"node_ranges",EN,sizeof(NMIncidenceRangeGPU));
    row(rows,"active_offsets",E,sizeof(std::uint32_t));
    row(rows,"global_active_indices",EC,sizeof(std::uint32_t));
    row(rows,"active_dispatch",1u,sizeof(NMIndirectDispatchGPU));
    return rows;
}
static ShippingMemoryBudgetInput budgetInput(std::uint64_t capacity,
                                               bool aliasSurface) {
    constexpr std::uint64_t N=528444u,T=1477728u,F=260835u,
        P=6335568u,SF=1054876u;
    ShippingMemoryBudgetInput x;
    x.nodeCount=N; x.tetrahedronCount=T; x.cohesiveFaceCount=F;
    x.contactPairCount=P; x.activeContactCapacity=P;
    x.deformableContactCapacity=capacity;
    x.surfaceFaceCount=SF; x.compactImmutableSurfaceMap=true;
    x.rigidGeneralizedCapacity=0u; x.rigidBodyCount=9u; x.rigidProxyCount=12u;
    x.materialStateStride=15u; x.compactPreconditionerRegionCount=3u;
    x.environmentCount=1u; x.fgmresRestart=kShippingFgmresRestartCeiling;
    x.fullContactArmijoRequired=true; x.deformableSelfContactEnabled=true;
    x.fullContactArmijoAliasesSurfaceScratch=aliasSurface;
    x.fullContactArmijoAliasesCohesiveScratch=false;
    x.fullContactArmijoTiledCohesive=true; x.cohesiveTrialPageCapacity=65536u;
    x.sparsePrivateContactRequested=true; x.sparsePrivateContactCapacity=65536u;
    x.sparsePrivateContactParity=false; x.sparsePrivateContactBoundedFreeBoxScope=false;
    x.hostSourceMeshBytes=add(mul(N,256u),mul(T,512u));
    x.hostCompiledWorldBytes=add(mul(N,128u),mul(T,256u));
    x.hostGeneratedMetalBytes=64ull*1024ull*1024ull;
    x.uploadStagingBytes=x.hostCompiledWorldBytes;
    x.retainedCompleteCheckpointCount=2u;
    x.estimatedSnapshotPayloadBytes=estimateShippingSnapshotPayloadBytes(x);
    return x;
}
int main(int argc, char** argv) {
    try {
        const auto maxBytes=parseMax(argc,argv);
        constexpr std::uint64_t E=1u,N=528444u,T=1477728u,F=260835u,
            C=131072u,P=6335568u,SF=1054876u,K=65536u;
        const std::uint64_t leaves=std::bit_ceil(SF);
        FullContactArmijoScratchBytes alias{}, noAlias{};
        if (!estimateFullContactArmijoScratchBytes(E,SF,leaves,C,N,true,alias) ||
            !estimateFullContactArmijoScratchBytes(E,SF,leaves,C,N,false,noAlias))
            throw std::overflow_error("full-contact scratch helper rejected input");
        auto aliasRows=contactArmijoRows(E,SF,leaves,C,N,true);
        auto noAliasRows=contactArmijoRows(E,SF,leaves,C,N,false);
        const auto sparse=admitSparsePrivateContact(1u,static_cast<std::uint32_t>(P),
            static_cast<std::uint32_t>(K),false,false,false);
        const auto sparseOneFreeBox=admitSparsePrivateContact(1u,static_cast<std::uint32_t>(P),
            static_cast<std::uint32_t>(K),false,true,true);
        const auto sparseGenericFree=admitSparsePrivateContact(1u,static_cast<std::uint32_t>(P),
            static_cast<std::uint32_t>(K),false,true,false);
        numi::matter::cardboard::cohesive_tiles::Layout tiled{};
        const auto tiledOK=numi::matter::cardboard::cohesive_tiles::makeLayout(
            1u,static_cast<std::uint32_t>(F),65536u,
            maxBytes.value_or(std::numeric_limits<std::uint64_t>::max()),tiled);
        const auto baseFloor=estimateShippingStaticMemoryLowerBound(N,T,1u,16u);
        auto defaultInput=budgetInput(8u*T,false);
        auto boundedInput=budgetInput(C,false);
        auto aliasInput=budgetInput(C,true);
        const auto defaultBudget=estimateShippingMemoryBudget(defaultInput);
        const auto boundedBudget=estimateShippingMemoryBudget(boundedInput);
        const auto aliasBudget=estimateShippingMemoryBudget(aliasInput);
        require(aliasBudget.requiredFullContactArmijoBytes == alias.total,
                "aliased shipping estimate must equal checked helper");
        require(boundedBudget.requiredFullContactArmijoBytes == noAlias.total,
                "unaliased shipping estimate must equal checked helper");
        FullContactArmijoScratchBytes e2fit{}, e2over{}, ov{};
        require(estimateFullContactArmijoScratchBytes(2u,0u,0u,127100u,0u,true,e2fit) &&
                e2fit.total<=kFullContactArmijoScratchBudgetBytes,"E2 fit boundary");
        require(estimateFullContactArmijoScratchBytes(2u,0u,0u,127101u,0u,true,e2over) &&
                e2over.total>kFullContactArmijoScratchBudgetBytes,"E2 first over");
        require(!estimateFullContactArmijoScratchBytes(
                    2u,std::numeric_limits<std::uint64_t>::max(),1u,1u,1u,false,ov),
                "overflow must fail closed");

        std::vector<BufferRow> baseRows;
        row(baseRows,"fem_node_states",mul(E,N),sizeof(NMFEMNodeStateGPU));
        row(baseRows,"tetrahedra",mul(E,T),sizeof(NMTetrahedronGPU));
        row(baseRows,"material_state_15",mul(mul(E,T),15u),sizeof(float));
        row(baseRows,"algorithmic_tangent_values",mul(mul(E,T),81u),sizeof(float));
        row(baseRows,"algorithmic_tangent_validity",mul(E,T),sizeof(std::uint32_t));
        const std::uint64_t unknowns=mul(2u,mul(E,N));
        row(baseRows,"fgmres_v_basis",mul(unknowns,17u),sizeof(nm_float4));
        row(baseRows,"fgmres_z_basis",mul(unknowns,16u),sizeof(nm_float4));
        row(baseRows,"rigid_contact_pair_samples",E*P,sizeof(NMContactSampleGPU));
        row(baseRows,"rigid_contact_pair_table",P,sizeof(NMContactPairGPU));
        row(baseRows,"rigid_contact_pair_candidate_history",E*P,sizeof(nm_float4));
        row(baseRows,"rigid_contact_pair_checkpoint_history",E*P,sizeof(nm_float4));
        row(baseRows,"rigid_contact_pair_accepted_history",E*P,sizeof(nm_float4));
        row(baseRows,"rigid_contact_active_slot_map",E*P,sizeof(std::uint32_t));
        row(baseRows,"deformable_contact_candidates",E*C,sizeof(NMDeformableContactCandidateGPU));
        row(baseRows,"deformable_contact",E*C,sizeof(NMDeformableContactGPU));
        row(baseRows,"deformable_contact_accepted_history",E*C,sizeof(NMDeformableContactHistoryGPU));
        row(baseRows,"deformable_contact_candidate_history",E*C,sizeof(NMDeformableContactHistoryGPU));
        row(baseRows,"deformable_contact_checkpoint_history",E*C,sizeof(NMDeformableContactHistoryGPU));
        row(baseRows,"cohesive_face_records_one_copy",E*F,sizeof(NMCohesiveFaceGPU));
        row(baseRows,"cohesive_force_vectors_one_copy",E*F,sizeof(NMCohesiveFaceVectorGPU));
        row(baseRows,"surface_primitives",E*SF,sizeof(NMContinuumSurfacePrimitiveGPU));
        row(baseRows,"surface_bvh_bounds",E*4u*leaves,sizeof(nm_float4));
        row(baseRows,"surface_sort_keys_a",E*SF,sizeof(std::uint32_t));
        row(baseRows,"surface_sort_keys_b",E*SF,sizeof(std::uint32_t));
        row(baseRows,"surface_sort_indices_a",E*SF,sizeof(std::uint32_t));
        row(baseRows,"surface_sort_indices_b",E*SF,sizeof(std::uint32_t));
        row(baseRows,"contact_node_ranges",E*N,sizeof(NMIncidenceRangeGPU));
        row(baseRows,"contact_node_incidence_six_per_C",E*C*6u,sizeof(std::uint32_t));
        const std::uint64_t logicalCornersPerEnvironment=mul(6u,C);
        const std::uint64_t paddedCornersPerEnvironment=std::bit_ceil(logicalCornersPerEnvironment);
        const std::uint64_t paddedCornerCount=mul(E,paddedCornersPerEnvironment);
        row(baseRows,"stable_corner_sort_keys_padded",paddedCornerCount,sizeof(std::uint64_t));
        row(baseRows,"stable_corner_sort_payloads_padded",paddedCornerCount,sizeof(std::uint32_t));
        row(baseRows,"regional_tangent_cache_values",mul(mul(E,T),81u),sizeof(float));
        row(baseRows,"regional_tangent_cache_validity",mul(E,T),sizeof(std::uint32_t));
        std::cout << "{\"schema\":\"numi.cardboard.shipping-arena-preflight.v1\","
                  << "\"inputs\":{\"E\":"<<E<<",\"N\":"<<N<<",\"T\":"<<T
                  <<",\"cohesive_faces\":"<<F<<",\"surface_faces\":"<<SF
                  <<",\"rigid_pair_rows\":"<<P<<",\"proxy_count\":12,\"body_slots\":9,\"deformable_C\":"<<C
                  <<",\"default_C_8T\":"<<8u*T<<",\"sparse_K\":"<<K
                  <<",\"restart\":16,\"material_state_stride\":15},"
                  << "\"evidence_class\":\"source-bound CPU arithmetic/layout preflight; no Metal device query, buffer allocation, solver admission, or physical qualification\","
                  << "\"runtime_scopes\":{"
                  << "\"ordinary_full_contact\":{\"node_limit\":8192,\"tet_limit\":32768,\"environment_limit\":4,\"contact_capacity_limit\":131072,\"proxy_limit\":16,\"one_static_explicit_FEM\":true,\"contact_required\":true,\"dense_pair_limit\":65536,\"sparse_pair_index_u32\":true,\"shipping_node_rejected\":true,\"shipping_tetrahedron_rejected\":true,\"shipping_C_rejected\":false,\"shipping_dense_pair_path_rejected\":true,\"sparse_K65536_no_free_rigid_admitted_by_byte_helper\":"<<(sparse.admitted()?"true":"false")<<",\"tile_cohesive_required_for_260835_faces\":true},"
                  << "\"true_residual_certification\":{\"node_limit\":4096,\"tet_limit\":32768,\"cohesive_direct_limit\":4096,\"rigid_generalized_capacity_must_be_zero\":true,\"shipping_node_rejected\":true,\"shipping_tetrahedron_rejected\":true,\"shipping_cohesive_count_rejected\":true}} ,"
                  << "\"integer_index_checks\":{"
                  << "\"E_times_N\":"<<E*N<<",\"E_times_T\":"<<E*T
                  <<",\"E_times_surface_faces\":"<<E*SF<<",\"bvh_leaves\":"<<leaves
                  <<",\"four_bounds_times_leaves\":"<<4u*leaves
                  <<",\"E_times_rigid_pairs\":"<<E*P
                  <<",\"E_times_C\":"<<E*C<<",\"six_incidence_times_C\":"<<6u*C
                  <<",\"six_corner_sort_payloads\":"<<mul(E,mul(6u,C))
                  <<",\"corner_sort_padded_per_environment\":"<<paddedCornersPerEnvironment
                  <<",\"corner_sort_total_padded\":"<<paddedCornerCount
                  <<",\"corner_sort_max_order_u32\":"<<(8u*(C-1u)+5u)
                  <<",\"corner_sort_max_payload_u32\":"<<(8u*(C-1u)+5u)
                  <<",\"E_times_cohesive_faces\":"<<E*F
                  <<",\"cohesive_face_slot_index_max\":"<<6u*F
                  <<",\"all_reported_dispatch_domains_fit_uint32\":"
                  <<((mul(E,N)<=UINT32_MAX && mul(E,T)<=UINT32_MAX && mul(E,P)<=UINT32_MAX &&
                      mul(E,C)<=UINT32_MAX && mul(E,mul(6u,C))<=UINT32_MAX &&
                      paddedCornerCount<=UINT32_MAX && mul(mul(E,leaves),4u)<=UINT32_MAX &&
                      mul(E,F)<=UINT32_MAX)?"true":"false")<<","
                  <<"\"pair_upper_bound_from_contact_nodes_times_proxies\":"<<527964u*12u
                  <<",\"compiled_pairs_within_upper_bound\":"<<(P<=527964u*12u?"true":"false")
                  <<"},\"static_fem_solver_floor\":{\"bytes\":"<<baseFloor.totalBytes
                  <<",\"node_states_bytes\":"<<baseFloor.femNodeStatesBytes
                  <<",\"tetrahedra_bytes\":"<<baseFloor.tetrahedraBytes
                  <<",\"fgmres_vectors_bytes\":"<<baseFloor.fgmresBasisBytes<<"},"
                  <<"\"buffers_known_extents\":";
        emitRows(baseRows,maxBytes);
        std::cout << ",\"arena_inventory_complete\":false"
                  << ",\"inventory_limit\":\"Named ABI extents only; not every runtime arena or simultaneous lifetime\""
                  << ",\"max_buffer_length_measured_on_device\":false"
                  << ",\"max_buffer_length\":";
        if(maxBytes) std::cout << *maxBytes; else std::cout << "null";
        std::cout << ",\"known_extents_pass_supplied_limit\":";
        if(!maxBytes) std::cout << "null";
        else {
            bool pass=true;
            for(const auto& b:baseRows) pass=pass && b.bytes<=*maxBytes;
            std::cout << (pass?"true":"false");
        }
        std::cout << ",\"full_contact_private_scratch\":{\"aliased_surface_bytes\":"<<alias.total
                  <<",\"aliased_surface_rows\":";
        emitRows(aliasRows,maxBytes);
        std::cout << ",\"unaliased_surface_bytes\":"<<noAlias.total
                  <<",\"unaliased_surface_rows\":";
        emitRows(noAliasRows,maxBytes);
        std::cout << ",\"unchanged_128_MiB_cap\":"<<kFullContactArmijoScratchBudgetBytes
                  <<",\"E1_alias_fits\":"<<(alias.total<=kFullContactArmijoScratchBudgetBytes?"true":"false")
                  <<",\"E1_unalias_fits\":"<<(noAlias.total<=kFullContactArmijoScratchBudgetBytes?"true":"false")
                  <<",\"exact_slot_status_bytes\":"<<sizeof(NMMatterStatusGPU)
                  <<"},\"sparse_private_rigid\":{\"admitted\":"<<(sparse.admitted()?"true":"false")
                  <<",\"scratch_bytes\":"<<sparse.totalBytes<<",\"cap_bytes\":"<<kSparsePrivateContactScratchBudgetBytes
                  <<",\"pair_slot_bytes\":"<<sparse.pairSlotBytes<<",\"K_sample_bytes\":"<<sparse.sampleBytes
                  <<",\"pair_domain_index_u32\":"<<(P<=UINT32_MAX?"true":"false")
                  <<",\"bounded_one_free_box_six_dof_admitted\":"
                  <<(sparseOneFreeBox.admitted()?"true":"false")
                  <<",\"bounded_one_free_box_scratch_bytes\":"<<sparseOneFreeBox.totalBytes
                  <<",\"generic_free_rigid_admitted\":"<<(sparseGenericFree.admitted()?"true":"false")
                  <<",\"generic_free_rigid_failure\":\""
                  <<sparsePrivateContactAdmissionErrorText(sparseGenericFree.error)<<"\""
                  <<"},\"tiled_cohesive\":{\"layout_admitted_by_helper\":"<<(tiledOK?"true":"false")
                  <<",\"page_capacity\":"<<tiled.pageCapacity<<",\"page_count\":"<<tiled.pages.size()
                  <<",\"page_bytes\":"<<tiled.pageBytes<<",\"private_page_cap_bytes\":"
                  <<numi::matter::cardboard::cohesive_tiles::kPrivatePageByteCap
                  <<",\"all_faces_covered\":"<<(tiledOK && tiled.pages.size()==(F+tiled.pageCapacity-1)/tiled.pageCapacity?"true":"false")
                  <<"},\"modeled_budget\":{"
                  <<"\"default_8T_capacity_peak_bytes\":"<<defaultBudget.estimatedPeakDomainBytes
                  <<",\"default_8T_model_within_16GiB\":"<<(defaultBudget.estimatedPeakDomainBytes<=kShippingPeakMemoryLimitBytes?"true":"false")
                  <<",\"explicit_C131072_unaliased_peak_bytes\":"<<boundedBudget.estimatedPeakDomainBytes
                  <<",\"explicit_C131072_aliased_peak_bytes\":"<<aliasBudget.estimatedPeakDomainBytes
                  <<",\"explicit_C_model_within_16GiB\":"<<(aliasBudget.estimatedPeakDomainBytes<=kShippingPeakMemoryLimitBytes?"true":"false")
                  <<",\"memory_model_is_measured_peak\":false,\"model_assumes_two_retained_checkpoints\":true"
                  <<",\"scratch_undercount_removed_bytes_E1_C131072_unaliased\":"<<(noAlias.total-(boundedBudget.requiredFullContactArmijoBytes-(48u*C+48u)))
                  <<"},\"current_verdict\":{\"modeled_bytes_can_be_under_16GiB\":"<<(aliasBudget.estimatedPeakDomainBytes<=kShippingPeakMemoryLimitBytes?"true":"false")
                  <<",\"runtime_admitted\":false,\"blockers\":[\"ordinary full-contact N=528444 exceeds 8192\",\"ordinary full-contact T=1477728 exceeds 32768\",\"true-residual N exceeds 4096 and rigid generalized rows are unsupported\",\"dense rigid private contact rejects 6335568 pair rows\",\"default self-contact C=8T exceeds 131072; explicit C only passes the count gate\",\"whole peak and Metal maxBufferLength remain unmeasured\"]}}\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "shipping-arena-preflight failed closed: " << e.what() << "\n";
        return 2;
    }
}
