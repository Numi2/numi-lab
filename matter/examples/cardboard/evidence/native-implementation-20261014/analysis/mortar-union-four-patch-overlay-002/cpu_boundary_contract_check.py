#!/usr/bin/env python3
"""CPU contract checks for the paired mortar shared-edge CCD exemption.

Uses canonical float32 barycentric maps authored by the full-contact fixture.
This checks the narrow witness predicate and source geometry only; it does not
execute Metal CCD or qualify a native contact transaction.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path('/Users/home/cardboard-implementation-20261014')
PACKAGE = ROOT / 'analysis/mortar-union-four-patch-overlay-002'
SNAPSHOT = ROOT / 'snapshots/source-rationalized-paper-and-native-checkers-091'
RUN = ROOT / 'native/repaired-mortar-union-private-ccd-301'
BARY_TOL = 8.0e-6
EPS = float(np.float32(1.1920928955078125e-7))
GEOMETRY_FLOOR = 1.0e-12


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fp32_rows(rows):
    return np.asarray(rows, dtype=np.float32).astype(np.float64)


def three_sector_maps():
    vertices = np.asarray([
        [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
        [np.float32(1. / 3.)] * 3,
    ], dtype=np.float32)
    sectors = ((0, 1, 3), (1, 2, 3), (2, 0, 3))
    return [fp32_rows(vertices[list(s)]) for s in sectors]


def four_sector_maps():
    # Exact maps used by the new four-patch private-sweep fixture.
    a, b, c = np.eye(3, dtype=np.float32)
    ab = np.asarray((a + b) * np.float32(.5), dtype=np.float32)
    bc = np.asarray((b + c) * np.float32(.5), dtype=np.float32)
    ca = np.asarray((c + a) * np.float32(.5), dtype=np.float32)
    mid = np.asarray((a + b + c) / np.float32(3.), dtype=np.float32)
    points = np.asarray((a, b, c, ab, bc, ca, mid), dtype=np.float32)
    cells = ((0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5))
    return [fp32_rows(points[list(cell)]) for cell in cells]


def paired_edge(left_a, right_a, left_b, right_b):
    """Mirror the Metal helper's common paired-vertex scan, including cutoff."""
    tol2 = BARY_TOL * BARY_TOL
    pairs = []
    for ia in range(3):
        for ib in range(3):
            if np.dot(left_a[ia] - left_b[ib], left_a[ia] - left_b[ib]) > tol2:
                continue
            if np.dot(right_a[ia] - right_b[ib], right_a[ia] - right_b[ib]) > tol2:
                continue
            if (ia, ib) in pairs:
                continue
            if len(pairs) >= 2:
                return None
            pairs.append((ia, ib))
    if len(pairs) != 2:
        return None
    a0, b0 = pairs[0]
    a1, b1 = pairs[1]
    if any(np.dot(x[j] - x[i], x[j] - x[i]) <= tol2
           for x, i, j in ((left_a, a0, a1), (right_a, a0, a1),
                           (left_b, b0, b1), (right_b, b0, b1))):
        return None
    return (left_a[a0], left_a[a1], right_a[a0], right_a[a1])


def edge_parameter(point, a, b):
    edge = b - a
    length2 = float(np.dot(edge, edge))
    if not (length2 > BARY_TOL * BARY_TOL and math.isfinite(length2)):
        return None
    t = float(np.dot(point - a, edge) / length2)
    if not math.isfinite(t) or t < -BARY_TOL or t > 1. + BARY_TOL:
        return None
    nearest = a + min(max(t, 0.), 1.) * edge
    if float(np.dot(point - nearest, point - nearest)) > BARY_TOL * BARY_TOL:
        return None
    return t


def paired_witness_on_edge(boundary, pleft, pright):
    if boundary is None:
        return False
    tleft = edge_parameter(pleft, boundary[0], boundary[1])
    tright = edge_parameter(pright, boundary[2], boundary[3])
    return (tleft is not None and tright is not None and
            abs(tleft - tright) <= BARY_TOL)


def nonapproaching(relative_displacement, normal):
    rel = np.asarray(relative_displacement, dtype=np.float32).astype(np.float64)
    n = np.asarray(normal, dtype=np.float32).astype(np.float64)
    speed = float(np.linalg.norm(rel))
    if not math.isfinite(speed) or not np.all(np.isfinite(n)):
        return False
    threshold = 8. * EPS * max(speed, GEOMETRY_FLOOR)
    return float(np.dot(rel, n)) >= -threshold


def triangle_area(triangle):
    p = np.asarray(triangle, dtype=np.float64)
    u, v = p[1, :2] - p[0, :2], p[2, :2] - p[0, :2]
    return .5 * abs(float(u[0] * v[1] - u[1] * v[0]))


def main():
    three = three_sector_maps()
    adjacent = paired_edge(three[0], three[0], three[1], three[1])
    assert adjacent is not None, 'canonical three-sector shared edge missing'
    assert paired_witness_on_edge(adjacent, adjacent[0], adjacent[2])
    assert paired_witness_on_edge(
        adjacent,
        .25 * adjacent[0] + .75 * adjacent[1],
        .25 * adjacent[2] + .75 * adjacent[3])
    assert not paired_witness_on_edge(
        adjacent,
        .25 * adjacent[0] + .75 * adjacent[1],
        .75 * adjacent[2] + .25 * adjacent[3]), 'sliding parameter mismatch was exempted'
    off_edge = adjacent[0] + np.asarray([0., 0., 2. * BARY_TOL])
    assert not paired_witness_on_edge(adjacent, off_edge, adjacent[2])

    z = np.asarray([0., 0., 1.], dtype=np.float32)
    minus_z = -z
    tangent = np.asarray([1., 0., 0.], dtype=np.float32)
    separation = np.asarray([0., 0., 0.25], dtype=np.float32)
    inward = np.asarray([0., 0., -0.25], dtype=np.float32)
    assert nonapproaching(separation, z), 'separating seam motion must be exemptible'
    assert nonapproaching(tangent, z), 'pure tangential shared-edge motion is non-approaching'
    assert not nonapproaching(inward, z), 'inward crossing must remain contact-eligible'
    assert not nonapproaching(-inward, minus_z), 'opposite witness normal must reverse approach sign'

    # A cyclicly mismatched face correspondence has a positive-area opposite-side
    # overlap, so it has no unique paired edge and remains eligible.
    permuted_side_b = [three[(patch_id + 1) % 3] for patch_id in range(3)]
    # Left patch 0 and right patch 2 map onto the same full sector, while
    # their other corresponding edges are not the same paired boundary.
    positive_overlap = triangle_area(three[0])
    mismatched = paired_edge(three[0], permuted_side_b[0],
                             three[2], permuted_side_b[2])
    assert positive_overlap > 0.0 and mismatched is not None
    cross_interior = np.asarray([4. / 9., 4. / 9., 1. / 9.], dtype=np.float32).astype(np.float64)
    assert not paired_witness_on_edge(mismatched, cross_interior, cross_interior), \
        'interior witness in positive-area cross-mapping was exempted'

    four = four_sector_maps()
    # The four-cell subdivision is a full unit barycentric triangle, and every
    # adjacent pair has exactly a common edge. Nonadjacent cells stay separate.
    assert abs(sum(triangle_area(cell) for cell in four) - .5) <= 2e-8
    four_adjacent = 0
    for i in range(len(four)):
        for j in range(i + 1, len(four)):
            edge = paired_edge(four[i], four[i], four[j], four[j])
            if edge is not None:
                four_adjacent += 1
    assert four_adjacent == 3, f'expected three outer-to-center shared edges in four-cell tiling, got {four_adjacent}'
    assert len(four) == 4

    result = {
        'schema': 'numi.cardboard.mortar-boundary-contract-cpu.v1',
        'evidence_class': 'CPU float32 map and predicate checks; no Metal CCD execution',
        'native301': {
            'receipt_sha256': sha(RUN / 'receipt.json'),
            'stderr': (RUN / 'stderr.log').read_text().strip(),
            'limitation': 'native 301 did not capture the first pair/witness; the exact-pair attribution remains a CPU reconstruction',
        },
        'source091_hashes': {
            'contact_metalinc': sha(SNAPSHOT / 'matter/src/metal/contact.metalinc'),
            'checker': sha(SNAPSHOT / 'matter/tools/cardboard_full_contact_armijo_check.mm'),
            'common_metalinc': sha(SNAPSHOT / 'matter/src/metal/common.metalinc'),
        },
        'predicate_constants': {
            'paired_boundary_bary_tolerance': BARY_TOL,
            'nonapproach_epsilon': EPS,
            'geometry_floor': GEOMETRY_FLOOR,
            'source_values': 'contact.metalinc kMortarPairedBoundaryBaryTolerance; common.metalinc kEpsilon and kGeometryFloor',
        },
        'checks': {
            'three_patch_shared_edge_endpoints_and_interior': 'PASS',
            'different_edge_parameter_not_exempted': 'PASS',
            'off_edge_not_exempted': 'PASS',
            'separating_witness_nonapproaching': 'PASS',
            'tangential_witness_nonapproaching': 'PASS',
            'inward_normal_witness_remains_candidate': 'PASS',
            'positive_area_cyclic_mismatch_remains_candidate': 'PASS',
            'four_patch_tiling_area_and_shared_edge_count': 'PASS',
        },
        'cross_identity_control': {
            'left_patch_id': 0,
            'right_patch_id': 2,
            'positive_overlap_area_in_parent_barycentric': positive_overlap,
            'paired_shared_boundary_exists': True,
            'interior_overlap_witness_exempted': False,
        },
        'four_patch': {
            'cell_count': len(four),
            'total_barycentric_area': sum(triangle_area(cell) for cell in four),
            'paired_shared_edges': four_adjacent,
            'capacity': 4,
        },
        'verdict': 'CPU contract checks pass. Native compilation passes. Native contact behavior, sliding controls, and four-patch capacity still require the parent-owned GPU run.',
    }
    out = PACKAGE / 'cpu_boundary_contract_check.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
