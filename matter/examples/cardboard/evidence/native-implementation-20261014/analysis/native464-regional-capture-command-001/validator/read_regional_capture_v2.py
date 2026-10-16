#!/usr/bin/env python3
"""Strict source151 regional CGLS capture reader v2.

The v1 parser remains an immutable upstream implementation. This wrapper adds
independent graph checksum/topology and pivot-permutation checks and refuses to
admit augmented RAP without a separately captured *consuming* restart cycle.
The v1 manifest records the producing-cycle tag; production consumes it at c+1.
No native replay or physical qualification is implied by parsing this file.
"""
from __future__ import annotations
import hashlib, json, math, struct
from pathlib import Path
from typing import Any
import read_regional_capture_v1 as v1

CaptureError = v1.CaptureError


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fnv32_arrays(arrays: dict[str, memoryview], names: tuple[str, ...]) -> int:
    h = 2166136261
    for name in names:
        data = arrays[name]
        if len(data) % 4:
            raise CaptureError(f"checksum array is not uint32 aligned: {name}")
        for off in range(0, len(data), 4):
            word = int.from_bytes(data[off:off+4], 'little')
            h = ((h ^ word) * 16777619) & 0xffffffff
    return h


def _floats_finite(data: memoryview) -> bool:
    if len(data) % 4:
        return False
    return all(math.isfinite(row[0]) for row in struct.iter_unpack('<f', data))


def _is_permutation(words: tuple[int, ...], size: int) -> bool:
    return len(words) == size and sorted(words) == list(range(size))


def _augmented_cycle_admission(tag: tuple[int, int, int, int],
                               expected_prefix: tuple[int, int, int],
                               mode_valid: int, factor_valid: int) -> dict[str, Any]:
    return {
        'producer_generation_tag': tag,
        'root_substep_newton_match': tag[:3] == expected_prefix,
        'producer_restart_cycle': tag[3],
        'consumer_restart_cycle': None,
        'required_relation': 'producer_cycle + 1 == consumed_restart_cycle',
        'mode_valid': mode_valid, 'factor_valid': factor_valid,
        'augmented_data_admissible_for_same_newton': False,
        'reason': 'consumed_restart_cycle_not_captured_independently',
    }


def _check_pivot_permutations(words: tuple[int, ...], env_count: int,
                              total_per_env: int,
                              segments: tuple[tuple[int, int], ...],
                              *, enabled_envs: tuple[bool, ...], label: str) -> bool:
    if len(words) != env_count * total_per_env or len(enabled_envs) != env_count:
        raise CaptureError(f"{label} pivot dimensions mismatch")
    for e in range(env_count):
        if not enabled_envs[e]:
            continue
        base = e * total_per_env
        for offset, n in segments:
            p = words[base+offset:base+offset+n]
            if len(p) != n or sorted(p) != list(range(n)):
                return False
    return True


def _topology_hashes(state_arrays: dict[str, memoryview], env: int,
                     nodes: int, tets: int) -> tuple[list[str], list[str]]:
    tet_bytes = state_arrays['accepted_tetrahedra']
    node_bytes = state_arrays['accepted_nodes']
    env_tet_bytes = tets * 96
    env_node_bytes = nodes * 80
    topo_hashes=[]; free_hashes=[]
    for e in range(env):
        tbase=e*env_tet_bytes; nbase=e*env_node_bytes
        conn=bytearray()
        for i in range(tets):
            conn.extend(tet_bytes[tbase+i*96:tbase+i*96+16])
        topology=struct.pack('<II',nodes,tets)+conn
        free=bytearray(nodes)
        for i in range(nodes):
            fixed=struct.unpack_from('<f',node_bytes,nbase+i*80+44)[0]
            if fixed not in (0.0,1.0):
                raise CaptureError(f"nonbinary source fixed tag in state capture env={e} node={i}")
            free[i]=1 if fixed==0.0 else 0
        topo_hashes.append(_sha(topology)); free_hashes.append(_sha(bytes(free)))
    return topo_hashes,free_hashes


def read_capture(path: str | Path) -> dict[str, Any]:
    """Validate bytes, joins, graph identity and replay gates for a v1 capture."""
    mpath=Path(path)
    result=v1.read_capture(mpath)
    manifest=v1._strict_json(mpath)
    _, payload=v1._local_payload(mpath,manifest.get('payload'))
    arrays=v1._validate_arrays(manifest,payload,strict_dtype=True)
    binding=manifest['candidate_state_binding']
    smpath=mpath.parent/binding['manifest_file']
    sm=v1._strict_json(smpath)
    _, spayload=v1._local_payload(smpath,sm.get('payload'))
    state_arrays=v1._validate_arrays(sm,spayload,strict_dtype=False)
    entries={a['name']:a for a in manifest['arrays']}
    env=manifest['environment_count']; nodes=manifest['node_count']; tets=manifest['tetrahedron_count']

    # All float inputs consumed by base/graph operator replay must be finite.
    finite_names=('P','effective_P','AP','base_RAP_lu','base_coarse_rhs',
                  'base_coarse_coefficients','graph_original_matrices','graph_patch_lu')
    finite_by_array={name:_floats_finite(arrays[name]) for name in finite_names if name in arrays}
    all_operator_finite=all(finite_by_array.values())

    # Enforce real row permutations, but preserve invalid-factor captures as
    # observational evidence rather than treating their zeroed pivot scratch as valid.
    modes=manifest['base_mode_count']
    base_valid=tuple(v1._u32_words(arrays['base_RAP_valid']))
    base_pivots=v1._u32_words(arrays['base_RAP_pivots'])
    base_perm=_check_pivot_permutations(
        base_pivots,env,modes,((0,modes),),
        enabled_envs=tuple(x==1 for x in base_valid),label='base RAP')
    aug_perm=None
    aug_enabled=manifest.get('same_newton_restart_augmentation_enabled') is True
    if aug_enabled:
        aug_modes=manifest['augmented_mode_count']
        aug_valid=tuple(v1._u32_words(arrays['augmented_RAP_valid']))
        aug_pivots=v1._u32_words(arrays['augmented_RAP_pivots'])
        aug_perm=_check_pivot_permutations(
            aug_pivots,env,aug_modes,((0,aug_modes),),
            enabled_envs=tuple(x==1 for x in aug_valid),label='augmented RAP')

    graph=manifest.get('graph_schwarz')
    graph_enabled=isinstance(graph,dict) and graph.get('enabled') is True
    graph_checksum_ok=None; graph_topology_ok=None; graph_mask_ok=None
    graph_permutations_ok=None; graph_finite=None
    if graph_enabled:
        names=('graph_patches','graph_rows','graph_patch_nodes',
               'graph_node_patch_offsets','graph_node_patch_values')
        computed=_fnv32_arrays(arrays,names)
        graph_checksum_ok=computed==graph.get('layout_checksum_fnv1a32')
        if not graph_checksum_ok:
            raise CaptureError('selector-5 graph layout checksum does not match exact captured arrays')
        top_hashes,free_hashes=_topology_hashes(state_arrays,env,nodes,tets)
        graph_topology_ok=all(x==graph.get('source_topology_sha256') for x in top_hashes)
        graph_mask_ok=all(x==graph.get('source_free_mask_sha256') for x in free_hashes)
        potential=_sha(bytes([1])*nodes)
        if graph.get('potential_node_mask_sha256') != potential:
            raise CaptureError('selector-5 potential mask digest is not the runtime all-covered mask')
        if not graph_topology_ok or not graph_mask_ok:
            raise CaptureError('selector-5 topology/free-mask digest differs from joined accepted state')
        patch_words=v1._u32_words(arrays['graph_patches'])
        patch_count=graph['patch_count']; pivot_total=graph['pivot_count']
        patch_desc=[tuple(patch_words[4*i:4*i+4]) for i in range(patch_count)]
        segments=tuple((p[3],3*p[1]) for p in patch_desc)
        gvalid=v1._u32_words(arrays['graph_factor_valid'])
        gpivots=v1._u32_words(arrays['graph_pivots'])
        if len(gvalid)!=env*patch_count or len(gpivots)!=env*pivot_total:
            raise CaptureError('selector-5 validity/pivot extent mismatch')
        graph_permutations_ok=True
        for e in range(env):
            for p,(pivot_off,n) in enumerate(segments):
                if gvalid[e*patch_count+p] == 1 and not _is_permutation(
                        gpivots[e*pivot_total+pivot_off:e*pivot_total+pivot_off+n],n):
                    graph_permutations_ok=False
        graph_finite=all(_floats_finite(arrays[n]) for n in ('graph_original_matrices','graph_patch_lu'))

    # Critical fail-closed distinction: restart_generation.w is the producer
    # cycle c; source151's consumer runs at c+1. The capture contains no
    # independently selected/consumed cycle, so even a matching first-three
    # word tag and valid factor cannot qualify augmented RAP for replay.
    generation_status=[]
    if aug_enabled:
        tags=v1._u32_words(arrays['restart_generation'])
        mode_valid=v1._u32_words(arrays['restart_mode_valid'])
        aug_valid=v1._u32_words(arrays['augmented_RAP_valid'])
        for e in range(env):
            tag=tuple(tags[4*e:4*e+4])
            row=_augmented_cycle_admission(
                tag,(manifest['control_step'],manifest['physics_substep'],manifest['newton_iteration']),
                mode_valid[e],aug_valid[e])
            row['environment']=e
            generation_status.append(row)
    return {
        **result,
        'reader_schema':'numi.matter.base_cgls_regional_state_capture.reader.v2',
        'graph_layout_checksum_recomputed':graph_checksum_ok,
        'graph_source_topology_matches_joined_state':graph_topology_ok,
        'graph_source_free_mask_matches_joined_state':graph_mask_ok,
        'graph_patch_pivots_are_permutations_when_valid':graph_permutations_ok,
        'base_RAP_pivots_are_permutations_when_valid':base_perm,
        'augmented_RAP_pivots_are_permutations_when_valid':aug_perm,
        'operator_arrays_finite':finite_by_array,
        'all_operator_arrays_finite':all_operator_finite,
        'graph_arrays_finite':graph_finite,
        'augmented_generation':generation_status,
        'base_regional_arrays_admission':'structurally validated; any native operator replay must independently verify action and currentness',
        'augmented_regional_arrays_admission':'UNADMITTED: consumed restart cycle is not recorded; do not infer it from producer tag',
        'native_or_physical_qualification':False,
    }
