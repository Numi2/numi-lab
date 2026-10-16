#!/usr/bin/env python3
"""Source-bound arithmetic and first-gate audit for the nominal shipping mesh.

This is a CPU evidence helper only. It does not construct a Matter world or
claim that the nominal topology has been admitted or physically assembled.
"""
from pathlib import Path
import hashlib
import json
import subprocess

B = Path('/Users/home/cardboard-implementation-20261014')
OUT = B / 'analysis/source175-sparse-private-scan-integration-001'
RUNTIME = OUT / 'candidate/matter/src/runtime.mm'
HEADER = OUT / 'candidate/matter/include/numi/matter/sparse_private_contact_admission.hpp'
SHARED = B / 'snapshots/source-checkpoint-warm-breakdown-173/matter/include/numi/matter/shared.h'
PREFLIGHT = B / 'native/shipping-memory-preflight-032/output/shipping-memory-native-preflight.json'
PROBE = OUT / 'validation/shipping-arena-size-probe'
PROBE_VALIDATION = OUT / 'validation/shipping_arena_size_probe_validation.json'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def round_min16(size: int) -> int:
    return max(size, 16)


def main() -> None:
    raw = subprocess.run([str(PROBE)], check=True, capture_output=True, text=True).stdout
    sizes = {}
    for line in raw.splitlines():
        name, value = line.split('=', 1)
        sizes[name] = int(value)

    # Nominal authored shipping topology supplied for this audit. The prior
    # native allocation preflight is a different, smaller 0-cohesive mesh.
    E, N, T, cohesive_faces = 1, 528_444, 1_477_728, 260_835
    surface_faces, contact_pairs = 1_054_876, 6_335_568
    sparse_capacity, scratch_budget = 65_536, 36 * 1024 * 1024

    blocks = (contact_pairs + 255) // 256
    sparse = {
        'sample_bytes': E * sparse_capacity * sizes['NMContactSampleGPU'],
        'pair_slot_bytes': E * contact_pairs * 4,
        'candidate_id_bytes': E * sparse_capacity * 4,
        'candidate_count_bytes': round_min16(E * 4),
        'block_offset_bytes': round_min16(E * blocks * 4),
        'blocks_per_environment': blocks,
    }
    sparse['total_bytes'] = sum(v for k, v in sparse.items() if k.endswith('_bytes'))
    sparse['budget_bytes'] = scratch_budget
    sparse['margin_bytes'] = scratch_budget - sparse['total_bytes']
    sparse['fits_one_environment'] = sparse['total_bytes'] <= scratch_budget
    sparse2 = {
        'sample_bytes': 2 * sparse_capacity * sizes['NMContactSampleGPU'],
        'pair_slot_bytes': 2 * contact_pairs * 4,
        'candidate_id_bytes': 2 * sparse_capacity * 4,
        'candidate_count_bytes': round_min16(2 * 4),
        'block_offset_bytes': round_min16(2 * blocks * 4),
    }
    sparse2['total_bytes'] = sum(sparse2.values())
    sparse2['margin_bytes'] = scratch_budget - sparse2['total_bytes']
    sparse2['fits_two_environments'] = sparse2['total_bytes'] <= scratch_budget

    face_leaf_count = 1 << (surface_faces - 1).bit_length()
    contact_capacity = 131_072
    unaliased_surface = (
        surface_faces * sizes['NMContinuumSurfacePrimitiveGPU']
        + 4 * face_leaf_count * 16  # four nm_float4 BVH bound arrays
        + 4 * surface_faces * 4  # two sort keys and two sort indices
    )
    capacity_scratch = contact_capacity * (
        sizes['NMDeformableContactCandidateGPU']
        + sizes['NMDeformableContactGPU']
        + 2 * 4 + 6 * 4
    )
    deformable_node_ranges = N * sizes['NMIncidenceRangeGPU']
    deformable_other = 4 * 4 + sizes['NMIndirectDispatchGPU']
    deformable_private_nonaliased = (
        unaliased_surface + capacity_scratch + deformable_node_ranges + deformable_other
    )
    deformable_private_aliased = capacity_scratch + deformable_node_ranges + deformable_other
    deformable_budget = 128 * 1024 * 1024

    preflight = json.loads(PREFLIGHT.read_text())
    old_n = preflight['mesh']['nodes']
    old_t = preflight['mesh']['tetrahedra']
    old_fgmres = preflight['static_lower_bound_bytes']['fgmres_vectors']
    fgmres_bytes_per_unknown = old_fgmres // (2 * old_n)
    assert old_fgmres % (2 * old_n) == 0
    assert fgmres_bytes_per_unknown == 528
    core_lower_bound = (
        N * sizes['NMFEMNodeStateGPU']
        + T * sizes['NMTetrahedronGPU']
        + (2 * N) * fgmres_bytes_per_unknown
    )

    page_capacity = (8 * 1024 * 1024) // sizes['NMCohesiveFaceVectorGPU']
    page_bytes = page_capacity * sizes['NMCohesiveFaceVectorGPU']
    page_count = (cohesive_faces + page_capacity - 1) // page_capacity

    result = {
        'schema': 'numi.cardboard.source175-shipping-admission-audit.v1',
        'evidence_class': 'source-bound admission and byte arithmetic; no world construction or physical qualification',
        'gpu_dispatch': False,
        'physical_transactions': 0,
        'sources': {
            str(RUNTIME): sha(RUNTIME),
            str(HEADER): sha(HEADER),
            str(SHARED): sha(SHARED),
            str(PREFLIGHT): sha(PREFLIGHT),
            str(PROBE): sha(PROBE),
            str(OUT / 'validation/shipping_arena_size_probe.cpp'): sha(
                OUT / 'validation/shipping_arena_size_probe.cpp'),
            str(OUT / 'validation/run_shipping_size_probe.py'): sha(
                OUT / 'validation/run_shipping_size_probe.py'),
            str(PROBE_VALIDATION): sha(PROBE_VALIDATION),
            str(OUT / 'validation/shipping_admission_audit.py'): sha(
                OUT / 'validation/shipping_admission_audit.py'),
        },
        'topology': {
            'environments': E,
            'nodes': N,
            'tetrahedra': T,
            'cohesive_faces': cohesive_faces,
            'surface_faces_used_as_primitive_count_assumption': surface_faces,
            'contact_pairs': contact_pairs,
        },
        'first_explicit_full_contact_armijo_gate': {
            'kind': 'runtime scope cap',
            'condition': 'femNodeCount > 8192',
            'observed_target_nodes': N,
            'limit': 8192,
            'result': 'rejects',
            'source': str(RUNTIME),
            'source_lines': [10915, 10920],
            'reason': 'fem_node_cap_8192',
            'call_chain': [
                {'function': 'femFullContactResidualArmijoScope', 'lines': [11141, 11146]},
                {'function': 'femCapsuleBoxResidualArmijoScope', 'lines': [10885, 10920]},
            ],
        },
        'next_scale_gates_after_node_cap': [
            {
                'condition': 'tetrahedronCount > 32768',
                'target': T,
                'limit': 32768,
                'result': 'rejects',
                'source_lines': [10921, 10924],
            },
            {
                'condition': 'deformable scratch supportedCounts requires femNodeCount <= 16384 and tetrahedronCount <= 32768',
                'target': {'nodes': N, 'tetrahedra': T},
                'limits': {'nodes': 16384, 'tetrahedra': 32768},
                'result': 'rejects',
                'source_lines': [6857, 6862],
            },
            {
                'condition': 'deformableContactCapacity <= 131072',
                'limit': 131072,
                'result': 'conditional on compiled dispatch capacity; scope rejects zero or greater than limit',
                'source_lines': [11161, 11181],
            },
            {
                'condition': 'full-contact cohesive arena: either tiled face force page <= 8 MiB or untiled cohesiveFaceCount <= 4096',
                'target_faces': cohesive_faces,
                'tile_page_capacity_faces_at_96_bytes_each': page_capacity,
                'tile_page_bytes': page_bytes,
                'tile_page_count': page_count,
                'result': 'tiled page arithmetic fits; non-tiled branch rejects',
                'source_lines': [11147, 11160],
            },
        ],
        'struct_sizes_bytes': sizes,
        'known_target_array_bytes': {
            'one_fem_node_state_array': N * sizes['NMFEMNodeStateGPU'],
            'accepted_plus_candidate_node_state_arrays_if_both_materialized': 2 * N * sizes['NMFEMNodeStateGPU'],
            'one_tetrahedron_array': T * sizes['NMTetrahedronGPU'],
            'accepted_plus_candidate_tetrahedron_arrays_if_both_materialized': 2 * T * sizes['NMTetrahedronGPU'],
            'one_cohesive_face_state_array': cohesive_faces * sizes['NMCohesiveFaceGPU'],
            'accepted_plus_candidate_cohesive_face_arrays_if_both_materialized': 2 * cohesive_faces * sizes['NMCohesiveFaceGPU'],
            'one_contact_pair_topology_array': contact_pairs * sizes['NMContactPairGPU'],
            'dense_one_sample_per_pair_array': contact_pairs * sizes['NMContactSampleGPU'],
            'sparse_sample_capacity_array': sparse_capacity * sizes['NMContactSampleGPU'],
        },
        'sparse_private_contact_arena': sparse,
        'sparse_private_contact_two_environment_counterfactual': sparse2,
        'deformable_contact_scratch_model': {
            'primitive_count_assumption': surface_faces,
            'leaf_count_bit_ceil': face_leaf_count,
            'capacity_at_existing_max': contact_capacity,
            'surface_arrays_unaliased_bytes': unaliased_surface,
            'contact_capacity_arrays_bytes': capacity_scratch,
            'node_incidence_ranges_bytes': deformable_node_ranges,
            'dispatch_and_small_counts_bytes': deformable_other,
            'private_total_if_surface_not_aliased': deformable_private_nonaliased,
            'private_total_if_surface_aliased': deformable_private_aliased,
            'existing_budget_bytes': deformable_budget,
            'unaliased_result': 'rejects budget',
            'aliased_result': 'fits arithmetic budget, but alias identity and all source/topology constraints require runtime proof',
            'source_lines': [6843, 6882],
        },
        'extrapolated_known_static_lower_bound': {
            'old_preflight_mesh': {'nodes': old_n, 'tetrahedra': old_t},
            'preflight_fgmres_bytes_per_unknown': fgmres_bytes_per_unknown,
            'formula_bytes': 'N*80 + T*96 + (2*N)*528, E=1',
            'target_bytes': core_lower_bound,
            'limit_bytes': preflight['peak_limit_bytes'],
            'within_16_gib': core_lower_bound <= preflight['peak_limit_bytes'],
            'interpretation': 'only the same three categories as the prior native preflight lower bound, scaled arithmetically; excludes many runtime/device/host/checkpoint/contact/compiler arenas and is not a peak forecast',
        },
        'prior_native_memory_observation': {
            'mesh': preflight['mesh'],
            'runtime_resident_bytes': preflight['native_observation_bytes']['runtime_resident_bytes'],
            'metal_current_allocated_bytes': preflight['native_observation_bytes']['metal_current_allocated_bytes'],
            'metal_recommended_max_working_set_bytes': preflight['native_observation_bytes']['metal_recommended_max_working_set_bytes'],
            'within_componentwise_16_gib': preflight['native_observation_bytes']['within_16gib_componentwise_limit'],
            'qualification_allowed': preflight['qualification_allowed'],
            'scope_warning': 'prior observation is for 453378 nodes and zero cohesive faces, not the 528444-node cohesive target',
        },
        'decision': {
            'first_blocker': 'full-contact residual Armijo scope rejects at node cap 8192; target has 528444 nodes',
            'sparse_pair_scan': 'one-environment pair-domain scratch fits 36 MiB, but does not remove node/tet/deformable/cohesive scope gates',
            'shipping_16_gib_admission': 'not established; this source-bound audit does not raise any limit or forecast a complete target peak',
        },
    }
    out = OUT / 'shipping-admission-audit-001'
    out.mkdir(exist_ok=True)
    (out / 'report.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({
        'first_gate': 'fem_node_cap_8192',
        'sparse_bytes_e1': sparse['total_bytes'],
        'sparse_fits_e1': sparse['fits_one_environment'],
        'sparse_bytes_e2': sparse2['total_bytes'],
        'sparse_fits_e2': sparse2['fits_two_environments'],
        'deformable_unaliased_bytes': deformable_private_nonaliased,
        'deformable_aliased_bytes': deformable_private_aliased,
        'target_core_lower_bound_bytes': core_lower_bound,
        'gpu_dispatch': False,
    }, sort_keys=True))


if __name__ == '__main__':
    main()
