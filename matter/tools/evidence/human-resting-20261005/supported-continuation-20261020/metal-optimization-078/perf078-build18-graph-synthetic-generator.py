#!/usr/bin/env python3
"""Create a tiny, deterministic NMSGKC1 fixture for the graph Schwarz GPU checker.

The layout arrays are reused verbatim from the retained C++ production-helper
NMGRO export. The operator is synthetic: identity plus tetrahedron-edge graph
Laplacian, restricted to free nodes. It is not captured Human physics evidence.
"""
from __future__ import annotations
import array
import hashlib
import json
import math
import random
import struct
import sys
from pathlib import Path

ROOT = Path('/Users/numi/numi-human-aortic-contact-opt-20261019')
EVIDENCE = ROOT / 'matter/examples/cardboard/evidence/native-implementation-20261014/analysis/selector5-native-kernel-check-001/canonical182-input'
GRAPH_INPUT = EVIDENCE / 'graph-layout-input.bin'
GRAPH_LAYOUT = EVIDENCE / 'graph-layout-output.bin'
OUT = Path('/tmp/perf078-build18-graph-synthetic')
OUT.mkdir(parents=True, exist_ok=True)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unpack_u32s(data: bytes, count: int):
    return list(struct.unpack('<' + 'I' * count, data))

# Read source topology/free mask and the already-exported output of
# buildRegionalGraphSchwarzLayout. We require exact dimensions/mask consistency.
graw = GRAPH_INPUT.read_bytes()
if graw[:8] != b'NMGRA001':
    raise SystemExit('bad retained graph-layout input magic')
version, node_count, tet_count, environment_count = struct.unpack_from('<4I', graw, 8)
if version != 1 or environment_count != 2:
    raise SystemExit('unexpected retained graph-layout input header')
cur = 24
free_mask = list(graw[cur:cur + node_count]); cur += node_count
if any(x not in (0, 1) for x in free_mask):
    raise SystemExit('source free mask is not binary')
if len(graw) - cur != 16 * tet_count:
    raise SystemExit('retained topology byte count mismatch')
tets = [struct.unpack_from('<4I', graw, cur + 16 * i) for i in range(tet_count)]
if any(len(set(t)) != 4 or any(v >= node_count for v in t) for t in tets):
    raise SystemExit('retained topology has invalid tetrahedra')

lraw = GRAPH_LAYOUT.read_bytes()
if lraw[:8] != b'NMGRO001':
    raise SystemExit('bad production-helper layout export magic')
h = struct.unpack_from('<9I', lraw, 8)
(v, ln, lt, le, patch_count, patch_node_count, dof_count,
 matrix_float_count, pivot_count) = h
if (v, ln, lt, le) != (1, node_count, tet_count, environment_count):
    raise SystemExit('production-helper layout dimensions differ from source topology')
cur = 44
layout_arrays = {}
for name, expected_count, width in (
    ('patches', patch_count, 16),
    ('patch_nodes', patch_node_count, 4),
    ('rows', dof_count, 16),
    ('node_patch_offsets', node_count + 1, 4),
    ('node_patch_values', patch_node_count, 4),
    ('node_coverage', node_count, 4),
):
    count = struct.unpack_from('<Q', lraw, cur)[0]; cur += 8
    if count != expected_count or cur + width * count > len(lraw):
        raise SystemExit(f'production-helper {name} extent mismatch')
    layout_arrays[name] = lraw[cur:cur + width * count]
    cur += width * count
if cur + 8 != len(lraw):
    raise SystemExit('production-helper layout has trailing bytes')
layout_device_bytes = struct.unpack_from('<Q', lraw, cur)[0]
coverage = unpack_u32s(layout_arrays['node_coverage'], node_count)
if any(bool(free_mask[i]) != bool(coverage[i]) for i in range(node_count)):
    raise SystemExit('source free mask and production-helper coverage disagree')
patches = [struct.unpack_from('<4I', layout_arrays['patches'], 16 * i)
           for i in range(patch_count)]
patch_nodes = unpack_u32s(layout_arrays['patch_nodes'], patch_node_count)
if len(patch_nodes) != patch_node_count or any(node >= node_count for node in patch_nodes):
    raise SystemExit('production-helper patch node map is invalid')

# Assemble a source-topology-derived, isotropic SPD operator K = I + L, where
# each tet contributes a positive spring to each of its six edges. Every patch
# matrix below is the exact principal submatrix K[patchNodes, patchNodes].
edge_weights: dict[tuple[int, int], float] = {}
for tet in tets:
    for i in range(4):
        for j in range(i):
            a, b = sorted((tet[i], tet[j]))
            edge_weights[(a, b)] = edge_weights.get((a, b), 0.0) + (1.0 / 6.0)
diagonal = [1.0 for _ in range(node_count)]
adjacency: list[list[tuple[int, float]]] = [[] for _ in range(node_count)]
for (a, b), w in sorted(edge_weights.items()):
    diagonal[a] += w
    diagonal[b] += w
    adjacency[a].append((b, -w))
    adjacency[b].append((a, -w))

# Build deterministic orthonormal regional modes on source-free xyz DOFs.
rng = random.Random(20261020)
free_dofs = [3 * node + axis for node in range(node_count) if free_mask[node]
             for axis in range(3)]
mode_count = 26
modes: list[list[float]] = []
for mode in range(mode_count):
    vector = [0.0] * (3 * node_count)
    for dof in free_dofs:
        vector[dof] = rng.uniform(-1.0, 1.0)
    # Two-pass modified Gram-Schmidt keeps the synthetic R matrix well-scaled.
    for _ in range(2):
        for prior in modes:
            dot = math.fsum(vector[i] * prior[i] for i in free_dofs)
            for i in free_dofs:
                vector[i] -= dot * prior[i]
    norm = math.sqrt(math.fsum(vector[i] * vector[i] for i in free_dofs))
    if not (math.isfinite(norm) and norm > 1.0e-12):
        raise SystemExit('synthetic regional basis lost rank')
    modes.append([value / norm for value in vector])

# Apply K matrix-free in node xyz order.
ap_modes: list[list[float]] = []
for vector in modes:
    applied = [0.0] * (3 * node_count)
    for node in range(node_count):
        for axis in range(3):
            value = diagonal[node] * vector[3 * node + axis]
            for neighbor, offdiag in adjacency[node]:
                value += offdiag * vector[3 * neighbor + axis]
            applied[3 * node + axis] = value
    ap_modes.append(applied)

rap = [[0.0] * mode_count for _ in range(mode_count)]
for row in range(mode_count):
    for column in range(mode_count):
        rap[row][column] = math.fsum(
            modes[row][dof] * ap_modes[column][dof] for dof in free_dofs)

# Match nm_fgmres_regional_restart_factor_rap's final row-permutation format.
lu = [row[:] for row in rap]
pivots = list(range(mode_count))
scale = max(abs(x) for row in lu for x in row)
pivot_floor = max(scale * 1.0e-10, 1.0e-30)
for k in range(mode_count):
    pivot_row = max(range(k, mode_count), key=lambda row: abs(lu[row][k]))
    if not (abs(lu[pivot_row][k]) > pivot_floor):
        raise SystemExit('synthetic RAP is singular')
    if pivot_row != k:
        lu[k], lu[pivot_row] = lu[pivot_row], lu[k]
        pivots[k], pivots[pivot_row] = pivots[pivot_row], pivots[k]
    pivot = lu[k][k]
    for row in range(k + 1, mode_count):
        lu[row][k] /= pivot
        lower = lu[row][k]
        for column in range(k + 1, mode_count):
            lu[row][column] -= lower * lu[k][column]

# RHS is deterministic and nonconstant, so coarse projection and graph solve
# both produce finite nonzero work. Fixed nodes remain zero in the source RHS.
basis_values = [0.0] * (4 * node_count)
for node in range(node_count):
    if not free_mask[node]:
        continue
    xyz = (
        math.sin(0.17 * (node + 1)) + 0.003 * node,
        math.cos(0.11 * (node + 2)) - 0.002 * node,
        math.sin(0.07 * (node + 3)) + math.cos(0.13 * node),
    )
    for axis in range(3):
        basis_values[4 * node + axis] = xyz[axis]

regional_basis = [0.0] * (mode_count * node_count * 4)
regional_ap = [0.0] * (mode_count * node_count * 4)
for mode in range(mode_count):
    for node in range(node_count):
        base = (mode * node_count + node) * 4
        for axis in range(3):
            regional_basis[base + axis] = modes[mode][3 * node + axis]
            regional_ap[base + axis] = ap_modes[mode][3 * node + axis]

# Pack exact principal blocks in patch row-major interleaved xyz ordering.
patch_matrices = array.array('f', [0.0]) * matrix_float_count
for patch_index, (node_offset, local_nodes, matrix_offset, pivot_offset) in enumerate(patches):
    if local_nodes == 0 or local_nodes > 256:
        raise SystemExit('production-helper patch size violates native kernel bound')
    nodes = patch_nodes[node_offset:node_offset + local_nodes]
    if len(nodes) != local_nodes or len(set(nodes)) != local_nodes:
        raise SystemExit('production-helper patch slice is malformed')
    local_index = {node: i for i, node in enumerate(nodes)}
    size = 3 * local_nodes
    for local_row, node in enumerate(nodes):
        for axis in range(3):
            row = 3 * local_row + axis
            patch_matrices[matrix_offset + row * size + row] = diagonal[node]
    for (a, b), w in edge_weights.items():
        ia = local_index.get(a); ib = local_index.get(b)
        if ia is None or ib is None:
            continue
        for axis in range(3):
            ra, rb = 3 * ia + axis, 3 * ib + axis
            patch_matrices[matrix_offset + ra * size + rb] = -w
            patch_matrices[matrix_offset + rb * size + ra] = -w

# Strictly little-endian host input ABI.
def pack_floats(values):
    result = array.array('f', values)
    if sys.byteorder != 'little':
        result.byteswap()
    return result.tobytes()

def pack_u32(values):
    return struct.pack('<' + 'I' * len(values), *values)

header = struct.pack('<8s11I', b'NMSGKC1\0', 1, node_count, tet_count,
    mode_count, patch_count, patch_node_count, dof_count,
    matrix_float_count, pivot_count, environment_count, 0)
arrays = [
    ('rap_lu', '<f4', [mode_count, mode_count], pack_floats(x for row in lu for x in row)),
    ('rap_pivots', '<u4', [mode_count], pack_u32(pivots)),
    ('rap_valid', '<u4', [1], pack_u32([1])),
    ('basis', '<f4x4', [node_count, 4], pack_floats(basis_values)),
    ('regional_basis', '<f4x4', [mode_count, node_count, 4], pack_floats(regional_basis)),
    ('regional_ap', '<f4x4', [mode_count, node_count, 4], pack_floats(regional_ap)),
    ('patches', '<u4x4', [patch_count, 4], layout_arrays['patches']),
    ('patch_nodes', '<u4', [patch_node_count], layout_arrays['patch_nodes']),
    ('node_patch_offsets', '<u4', [node_count + 1], layout_arrays['node_patch_offsets']),
    ('node_patch_values', '<u4', [patch_node_count], layout_arrays['node_patch_values']),
    ('node_coverage', '<u4', [node_count], layout_arrays['node_coverage']),
    ('patch_matrices', '<f4', [matrix_float_count], pack_floats(patch_matrices)),
]
blob = bytearray(header)
array_meta = []
for name, dtype, shape, payload in arrays:
    offset = len(blob)
    blob.extend(payload)
    array_meta.append({'name': name, 'dtype': dtype, 'shape': shape,
                       'offset': offset, 'byte_count': len(payload),
                       'sha256': digest(payload)})
input_path = OUT / 'selector5-kernel-input.bin'
input_path.write_bytes(blob)
manifest = {
    'schema': 'numi.matter.selector5-kernel-check-input.v1',
    'qualification': 'synthetic native-kernel fixture only; not captured Human operator evidence',
    'layout_source': str(GRAPH_LAYOUT),
    'layout_source_sha256': digest(lraw),
    'topology_source': str(GRAPH_INPUT),
    'topology_source_sha256': digest(graw),
    'layout_device_bytes': layout_device_bytes,
    'operator': 'K = I + sum over each source tet of six positive unit/6 edge springs, applied isotropically to xyz; patch matrices are exact K principal blocks',
    'coarse_operator': 'R = B^T K B, LU/pivots formed in FP64 then stored FP32',
    'fixed_candidate_nodes': sum(not bit for bit in free_mask),
    'fixed_candidate_rule': 'candidate tags are supplied by the checker from node_coverage == 0, matching the retained source free mask',
    'dimensions': {'node_count': node_count, 'tetrahedron_count': tet_count,
        'mode_count': mode_count, 'patch_count': patch_count,
        'patch_node_count': patch_node_count, 'dof_count': dof_count,
        'matrix_float_count': matrix_float_count, 'pivot_count': pivot_count,
        'source_environment_count': environment_count, 'selected_environment': 0},
    'bundle_file': input_path.name,
    'bundle_byte_count': len(blob),
    'bundle_sha256': digest(blob),
    'arrays': array_meta,
}
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
# Cheap CPU integrity checks, not a native GPU run.
if len(blob) != len(header) + sum(item['byte_count'] for item in array_meta):
    raise SystemExit('serialized array sizes do not cover NMSGKC1 bundle')
if not all(math.isfinite(x) for x in patch_matrices):
    raise SystemExit('patch matrices contain nonfinite entries')
print(json.dumps({'input': str(input_path), 'bytes': len(blob),
    'sha256': digest(blob), 'dimensions': manifest['dimensions'],
    'fixed_candidate_nodes': manifest['fixed_candidate_nodes'],
    'layout_device_bytes': layout_device_bytes}, sort_keys=True))
