#!/usr/bin/env python3
"""CPU checks for the source-derived selector-5 base-map CGLS overlay.

This calls the separately sealed source151 algebraic reference. It does not
emulate Metal arithmetic or claim native-CGLS parity.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

PACKAGE = Path(__file__).resolve().parents[1]
REFERENCE = Path("/Users/home/cardboard-implementation-20261014/analysis/regional-rap-selector5-reference-151-002/selector5_graph_reference.py")
REFERENCE_HASH = ""
sys.path.insert(0, str(REFERENCE.parent))
spec = importlib.util.spec_from_file_location("selector5_graph_reference", REFERENCE)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load sealed source151 graph reference")
ref = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)
rap = ref.rap


def dot(a, b):
    return math.fsum(float(x) * float(y) for x, y in zip(a, b))


def build_capture():
    nodes = 5
    dofs = 3 * nodes
    free = (True, True, True, True, False)
    A = [[0.0] * dofs for _ in range(dofs)]
    for i in range(12):
        A[i][i] = 7.0 + 0.07 * i
        for j in range(12):
            if i != j:
                # Deterministic diagonally dominant nonsymmetric operator.
                A[i][j] = 0.013 * ((i * 11 + j * 7) % 9 - 4)
    # Force a real row pivot in the first patch so its transpose solve must
    # apply the captured final-permutation convention, not an identity LU.
    A[0][0] = 0.05
    A[3][0] = 2.5
    for i in range(12, dofs):
        A[i][i] = 1.0
    P = [[0.0, 0.0] for _ in range(dofs)]
    for i in range(12):
        P[i][0] = 0.7 + 0.031 * (i + 1)
        P[i][1] = -0.2 + 0.017 * ((3 * i + 2) % 11)
    AP = [[dot(A[i], [P[k][m] for k in range(dofs)]) for m in range(2)]
          for i in range(dofs)]
    R = [[math.fsum(P[i][a] * AP[i][b] for i in range(dofs))
          for b in range(2)] for a in range(2)]
    base_factor = rap.factor_rap(R)
    if not base_factor.valid:
        raise AssertionError("manufactured base RAP unexpectedly singular")

    patch_nodes = ((0, 1, 2), (1, 2, 3))
    patches = []
    packed = []
    for members in patch_nodes:
        idx = [3 * node + axis for node in members for axis in range(3)]
        local = [[A[i][j] for j in idx] for i in idx]
        factor = rap.factor_rap(local)
        if not factor.valid:
            raise AssertionError("manufactured overlapping patch unexpectedly singular")
        if max(abs(local[i][j] - local[j][i])
               for i in range(len(local)) for j in range(len(local))) == 0.0:
            raise AssertionError("manufactured local patch is accidentally symmetric")
        patches.append(ref.GraphPatch(tuple(members), local, factor))
        packed.extend(members)
    memberships = [[] for _ in range(nodes)]
    for index, node in enumerate(packed):
        memberships[node].append(index)
    offsets = [0]
    values = []
    coverage = []
    for items in memberships:
        values.extend(items)
        offsets.append(len(values))
        coverage.append(len(items))
    return ref.GraphCapture(
        node_count=nodes,
        free_mask=free,
        patches=tuple(patches),
        patch_nodes=tuple(packed),
        node_patch_offsets=tuple(offsets),
        node_patch_values=tuple(values),
        node_coverage=tuple(coverage),
        P=P,
        AP=AP,
        base_factor=base_factor,
        mode_stride=2,
        augmentation_enabled=False,
        completed_tail_latch=False,
        identity=("source151-synthetic", "device", "world", "candidate", 0, 1, 2, 0),
    )


def check_reference_adjoint():
    c = build_capture()
    if all(tuple(p.factor.pivots) == tuple(range(len(p.factor.pivots)))
           for p in c.patches):
        raise AssertionError("manufactured local factors did not exercise a row pivot")
    x = [math.sin(i * 0.37) + 0.03 * i for i in range(c.dof_count)]
    y = [math.cos(i * 0.23) - 0.02 * i for i in range(c.dof_count)]
    sx = ref.apply_base_S(c, x)
    sty = ref.apply_base_S_transpose(c, y)
    lhs, rhs = dot(sx, y), dot(x, sty)
    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
    if rel > 2e-13:
        raise AssertionError(f"source151 S/S^T adjoint identity failed: rel={rel:.3g}")
    for node in (4,):
        if sx[3 * node:3 * node + 3] != [0.0] * 3:
            raise AssertionError("S did not zero fixed-node coordinates")
        if sty[3 * node:3 * node + 3] != [0.0] * 3:
            raise AssertionError("S^T did not zero fixed-node coordinates")
    return {"case": "overlapping_nonsymmetric_pivoted_patches", "adjoint_relative_error": rel,
            "fixed_output_zero": True, "selected_patch_count": len(c.patches),
            "local_row_pivot_exercised": True}


def check_invalid_and_generation_controls():
    c = build_capture()
    bad_patch = ref.GraphPatch(c.patches[0].nodes, c.patches[0].original,
                               rap.Factor(c.patches[0].factor.lu,
                                          c.patches[0].factor.pivots, False))
    invalid = ref.GraphCapture(**{**c.__dict__, "patches": (bad_patch, c.patches[1])})
    failures = []
    for call in (ref.apply_base_S, ref.apply_base_S_transpose):
        try:
            call(invalid, [1.0] * c.dof_count)
        except ref.GraphReferenceError:
            failures.append(True)
        else:
            failures.append(False)
    if failures != [True, True]:
        raise AssertionError("invalid graph factor did not fail closed in both directions")
    try:
        ref.require_base_view(c, request_actual_consumed_view=True)
    except ref.GraphReferenceError:
        generation_rejected = True
    else:
        generation_rejected = False
    augmented = ref.GraphCapture(**{**c.__dict__, "augmentation_enabled": True})
    try:
        ref.require_base_view(augmented, request_actual_consumed_view=True)
    except ref.GraphReferenceError:
        augmentation_rejected = True
    else:
        augmentation_rejected = False
    if not generation_rejected or not augmentation_rejected:
        raise AssertionError("reference admitted unsupported consumed/augmented view")
    return {"case": "invalid_factor_and_unproven_view", "invalid_factor_S_and_ST_rejected": True,
            "actual_consumed_view_rejected": True, "augmented_view_rejected": True}


def check_pivoted_nonsymmetric_rap_transpose():
    matrix = [[0.05, 2.0], [1.0, 3.0]]
    factor = rap.factor_rap(matrix)
    if not factor.valid or tuple(factor.pivots) != (1, 0):
        raise AssertionError("manufactured RAP did not exercise row permutation")
    rhs = [0.7, -1.3]
    solution = rap.solve_rap_transpose(factor, rhs)
    residual = [
        math.fsum(matrix[j][i] * solution[j] for j in range(2)) - rhs[i]
        for i in range(2)
    ]
    residual_l2 = math.sqrt(dot(residual, residual))
    if residual_l2 > 2e-14:
        raise AssertionError(f"pivoted nonsymmetric RAP transpose residual {residual_l2:.3g}")
    return {"case": "pivoted_nonsymmetric_base_RAP_transpose",
            "pivots": list(factor.pivots), "residual_l2": residual_l2}


def check_production_contract():
    runtime = (PACKAGE / "source/matter/src/runtime.mm").read_text()
    fgmres = (PACKAGE / "source/matter/src/metal/fgmres.metalinc").read_text()
    cgls = (PACKAGE / "source/matter/src/metal/base_cgls.metalinc").read_text()
    required_runtime = [
        'std::strcmp(rightScale, "regional5") == 0',
        "state.mixedSolverValue.executionBudgets.z == 5u",
        "!state.femRegionalRestartAugmentationEnabled",
        "state.dispatch.rigidGeneralizedCapacity == 0u",
        '"nm_fgmres_base_cgls_validate_regional_graph"',
        '"nm_fgmres_base_cgls_solve_regional_rap"',
        "applySelectedCGLSRightMap(state.femBaseCGLSP",
        "applyRegionalBaseST(state.femBaseCGLSS",
        "applySelectedCGLSRightMap(state.femBaseCGLSX",
        "regional5_base_graph",
    ]
    missing = [term for term in required_runtime if term not in runtime]
    if missing:
        raise AssertionError(f"runtime source contract missing {missing}")
    if "cglsMode == 1u &&" not in fgmres or "cglsSelected[environment] != 1u" not in fgmres:
        raise AssertionError("CGLS graph dispatch is not explicitly selected-env gated")
    if "nm_fgmres_base_cgls_graph_h_transpose" not in cgls or "nm_fgmres_base_cgls_regional_solve_rap_transpose" not in cgls:
        raise AssertionError("exact transpose kernel path is incomplete")
    rap_solve = re.search(
        r"kernel void nm_fgmres_base_cgls_solve_regional_rap\(([\s\S]*?)\n\}",
        cgls)
    if rap_solve is None:
        raise AssertionError("selected-only regional RAP solve is missing")
    rap_body = rap_solve.group(1)
    for required in ("selected[environment] != 1u", "valid[environment] == 0u",
                     "source >= modeCount", "seen & (1u << source)",
                     "kBaseCGLSErrorInvalidRegionalMap",
                     "kBaseCGLSErrorNonfinite"):
        if required not in rap_body:
            raise AssertionError(f"CGLS regional RAP failure latch missing {required}")
    if '"nm_fgmres_regional_solve_rap"' in runtime[
        runtime.find("const auto applyRegionalBaseS"):
        runtime.find("const auto applyRegionalBaseST")]:
        raise AssertionError("regional CGLS forward map uses silent ordinary RAP fallback")
    if 'does not describe the regional5 graph map' not in runtime:
        raise AssertionError("regional5 binary capture without its map is not rejected")
    if re.search(r"NM_FEM_BASE_CGLS_RIGHT_SCALE[\s\S]{0,150}regional5", runtime) is None:
        raise AssertionError("regional5 selector parser is not present")
    return {"case": "source-contract", "selector5_base_only": True,
            "pure_fem_gate": True, "same_newton_resident_map": True,
            "completed_latch_bypassed_only_in_cgls_mode": True,
            "binary_capture_fail_closed": True,
            "regional_rap_solver_failure_latches_private_cgls_error": True}


def main():
    expected_ref_hash = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    result = {
        "schema": "numi.cgls.regional5_right_map_cpu_checks.v1",
        "source151_reference_sha256": expected_ref_hash,
        "source_overlay_files": {},
        "checks": [check_reference_adjoint(), check_invalid_and_generation_controls(),
                   check_pivoted_nonsymmetric_rap_transpose(),
                   check_production_contract()],
        "qualification": "CPU algebraic/source-contract checks only; no Metal compilation or native CGLS result",
    }
    for rel in ("source/matter/src/runtime.mm", "source/matter/src/metal/fgmres.metalinc",
                "source/matter/src/metal/base_cgls.metalinc"):
        result["source_overlay_files"][rel] = hashlib.sha256((PACKAGE / rel).read_bytes()).hexdigest()
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))

if __name__ == "__main__":
    main()
