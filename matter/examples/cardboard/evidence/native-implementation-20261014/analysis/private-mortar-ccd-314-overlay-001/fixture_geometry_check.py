#!/usr/bin/env python3
"""Independent arithmetic checks for the private mortar CCD overlay fixtures."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PATCH_BOUNDARY_FIRST_BARY = 0.35
PARENT_EDGE_LENGTH_M = 4.0e-3
COLLISION_THICKNESS_M = 0.01 * 1.0e-3
GAP_M = 2.0e-4
MOVED_VERTEX_END_Z_M = 4.0e-4
UNCOVERED_VERTEX_END_Z_M = 2.8e-4
SHARED_MAP_TOL = 8.0e-6
IN_PLANE_OFFSET_M = 4.0e-9

# The partial patch occupies first-barycentric-coordinate x >= 0.35.
# The covered-positive case moves only the x=1 corner. On x < 0.35, its
# maximum displacement is 0.4 mm * 0.35 = 0.14 mm, so 0.06 mm remains.
covered_max_z = MOVED_VERTEX_END_Z_M * PATCH_BOUNDARY_FIRST_BARY
covered_complement_clearance = GAP_M - covered_max_z
assert covered_complement_clearance > COLLISION_THICKNESS_M

# The positive-gap near-contact case instead moves the x=0 corner. It is in
# the uncovered complement, starts with a 0.2 mm gap, and crosses the opposing
# plane during the linear endpoint path. The crossing is not initially touching.
near_contact_end_gap = GAP_M - UNCOVERED_VERTEX_END_Z_M
near_contact_patch_edge_z = UNCOVERED_VERTEX_END_Z_M * 0.65
near_contact_patch_min_clearance = GAP_M - near_contact_patch_edge_z
assert GAP_M > COLLISION_THICKNESS_M
assert near_contact_end_gap < 0.0
assert near_contact_patch_min_clearance > COLLISION_THICKNESS_M

# This case targets the paired-witness barycentric tolerance: shifting one
# parent by 4 nm over a 4 mm edge induces a 1e-6 barycentric offset, below the
# current 8e-6 witness tolerance. All base-to-trial displacements are zero.
static_off_edge_bary_offset = IN_PLANE_OFFSET_M / PARENT_EDGE_LENGTH_M
assert static_off_edge_bary_offset < SHARED_MAP_TOL
assert IN_PLANE_OFFSET_M < COLLISION_THICKNESS_M

result = {
    "schema": "numi.cardboard.private_mortar_fixture_geometry.v1",
    "classification": "CPU fixture arithmetic only; no native CCD result",
    "partial_patch_first_bary_coverage": f"x >= {PATCH_BOUNDARY_FIRST_BARY}",
    "collision_thickness_m": COLLISION_THICKNESS_M,
    "partial_covered_crossing": {
        "start_gap_m": GAP_M,
        "maximum_uncovered_left_z_m": covered_max_z,
        "minimum_uncovered_clearance_m": covered_complement_clearance,
        "clearance_exceeds_collision_thickness": True,
    },
    "partial_uncovered_near_contact": {
        "start_gap_m": GAP_M,
        "end_gap_at_moving_uncovered_vertex_m": near_contact_end_gap,
        "minimum_active_patch_clearance_m": near_contact_patch_min_clearance,
        "linear_path_crosses_only_in_uncovered_complement": True,
    },
    "static_off_edge_initial_overlap": {
        "all_displacements_zero": True,
        "initial_normal_separation_m": 0.0,
        "parent_lateral_offset_m": IN_PLANE_OFFSET_M,
        "induced_barycentric_offset": static_off_edge_bary_offset,
        "paired_witness_tolerance": SHARED_MAP_TOL,
        "off_edge_offset_is_inside_current_witness_tolerance": True,
        "native_expected": "contact failure; test whether paired-boundary witness exemption masks initial off-edge collision",
    },
}
print(json.dumps(result, sort_keys=True, indent=2))
