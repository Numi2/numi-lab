#!/usr/bin/env python3
"""Recompute bounded graph-Schwarz arena arithmetic; no source or device access."""
import json
N, T, E, fixed = 20682, 89088, 2, 324
P, patch_nodes, M = 128, 64, 32
free = N - fixed
patch_slots = P * patch_nodes
dofs = 3 * patch_nodes
matrices = P * dofs * dofs
pivots = P * dofs
rows = P * dofs
metadata = P*16 + patch_slots*4 + rows*16 + (N+1)*4 + patch_slots*4 + N*4
per_env = 2*matrices*4 + pivots*4 + patch_slots*16 + P*4
graph = E*per_env + metadata
block = E*N*3*16
diagonal = E*N*16
regional = (N*M + 2*E*N*M + E*M*M + 4*E*M + E*M + E)*16
q, r = divmod(free, P)
full_matrix_floats = 9*(r*(q+1)**2 + (P-r)*q*q)
full_metadata = P*16 + free*4 + 3*free*16 + (N+1)*4 + free*4 + N*4
full_graph = E*(2*full_matrix_floats*4 + 3*free*4 + free*16 + P*4) + full_metadata
print(json.dumps({
    'target': {'nodes':N,'tets':T,'environments':E,'fixed_nodes':fixed,'free_nodes':free,'basis_modes_assumed':M},
    'partial_128x64': {
        'patch_matrix_floats_per_environment':matrices,
        'metadata_bytes':metadata,
        'graph_device_bytes':graph,
        'graph_MiB':graph/(1024*1024),
        'patch_slots':patch_slots,
        'minimum_uncovered_free_nodes':max(0,free-patch_slots),
        'block_inverse_extra_bytes':block,
        'diagonal_bytes':diagonal,
        'regional_P_AP_coarse_bytes':regional,
        'graph_block_coarse_subtotal_bytes':graph+block+regional,
        'factorization_flop_order_2env':2*E*P*dofs**3/3,
        'factor_threadgroup_search_scratch_bytes':3*256*4,
        'solve_threadgroup_float_arrays_bytes':3*768*4
    },
    'full_free_node_coverage_equal_128_dense_patches': {
        'patches_160_nodes':r,'patches_159_nodes':P-r,
        'matrix_floats_per_environment':full_matrix_floats,
        'graph_device_bytes':full_graph,
        'graph_MiB':full_graph/(1024*1024),
        'cap_bytes':100663296,
        'excess_bytes':full_graph-100663296
    }
},indent=2,sort_keys=True))
