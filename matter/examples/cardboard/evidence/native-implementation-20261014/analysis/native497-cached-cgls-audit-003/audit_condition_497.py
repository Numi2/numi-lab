#!/usr/bin/env python3
"""Direct solve and conditioning estimate for the exact cached operator in run 497.

All inputs are the retained native captures. The matrix is reconstructed from
the captured FP32 tangent cache using the existing source-bound FP64 replay.
This does not rerun constitutive physics or qualify the full coupled solver.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, onenormest, splu

BASE = Path("/Users/home/cardboard-implementation-20261014")
DEFAULT_CASE = BASE / "solver/macmini-warm-cgls-control20-diagnostic-replay-497"
DEFAULT_SOURCE = BASE / "snapshots/source-admission-encoder-182-003"
DEFAULT_ARTIFACTS = BASE / "analysis/source182-host-artifacts-003"
DEFAULT_READER = BASE / "analysis/base-cgls-state-reader-abi-repair-148-001/tools/read_state_capture.py"
PRIOR_001 = BASE / "analysis/native497-cached-cgls-audit-001/report-v2.json"
PRIOR_002 = BASE / "analysis/native497-cached-cgls-audit-002/report-v4.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError(f"duplicate key {k!r} in {path}")
            out[k] = v
        return out
    def reject(token):
        raise ValueError(f"nonstandard JSON token {token} in {path}")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def vector_metrics(A, b: np.ndarray, x: np.ndarray) -> dict[str, Any]:
    x64 = np.asarray(x, dtype=np.float64).reshape(-1)
    b64 = np.asarray(b, dtype=np.float64).reshape(-1)
    r = b64 - np.asarray(A @ x64).reshape(-1)
    abs_A = abs(A)
    component_den = np.asarray(abs_A @ np.abs(x64)).reshape(-1) + np.abs(b64)
    component_ratio = np.zeros_like(r)
    nz = component_den > 0.0
    component_ratio[nz] = np.abs(r[nz]) / component_den[nz]
    component_ratio[~nz] = np.where(np.abs(r[~nz]) == 0.0, 0.0, np.inf)
    norm_a1 = float(np.asarray(abs_A.sum(axis=0)).max())
    norm_ainf = float(np.asarray(abs_A.sum(axis=1)).max())
    norm_b1, norm_x1 = float(np.linalg.norm(b64, 1)), float(np.linalg.norm(x64, 1))
    norm_binf, norm_xinf = float(np.linalg.norm(b64, np.inf)), float(np.linalg.norm(x64, np.inf))
    denom1 = norm_a1 * norm_x1 + norm_b1
    denomi = norm_ainf * norm_xinf + norm_binf
    return {
        "finite": bool(np.isfinite(x64).all() and np.isfinite(r).all()),
        "residual_l2": float(np.linalg.norm(r)),
        "residual_l1": float(np.linalg.norm(r, 1)),
        "residual_linf": float(np.linalg.norm(r, np.inf)),
        "rhs_l2": float(np.linalg.norm(b64)),
        "solution_l2": float(np.linalg.norm(x64)),
        "normwise_backward_error_1": float(np.linalg.norm(r, 1) / denom1) if denom1 else None,
        "normwise_backward_error_inf": float(np.linalg.norm(r, np.inf) / denomi) if denomi else None,
        "componentwise_backward_error_inf": float(np.max(component_ratio)),
        "residual_over_rhs_l2": float(np.linalg.norm(r) / np.linalg.norm(b64)) if np.linalg.norm(b64) else None,
        "residual_vector": r,
    }


def compensated_global_stiffness(operator_module, header: dict, records: list[dict],
                                  environment: int, free_nodes: np.ndarray,
                                  node_count: int) -> sparse.csr_matrix:
    """Re-sum the same FP64 element-entry terms with math.fsum per global slot."""
    descriptor = {int(m["index"]): m for m in header["materials"]}
    constraints = np.zeros(node_count, dtype=np.int32)
    for row in records:
        if row.get("record") != "node" or int(row["environment"]) != environment:
            continue
        tag = operator_module.captured_f32_scalar(row["center"]["constraint_tag"], "constraint tag")
        if not tag.is_integer():
            raise ValueError("nonintegral fixed/free tag during compensated replay")
        constraints[int(row["node_id"])] = int(tag)
    fixed = np.isin(constraints, (1, 3))
    if not np.array_equal(np.flatnonzero(~fixed), free_nodes):
        raise ValueError("fixed/free mask differs from source-bound assembly")
    reduced = np.full(node_count, -1, dtype=np.int64)
    reduced[free_nodes] = np.arange(free_nodes.size, dtype=np.int64)
    dof_map = np.full((node_count, 3), -1, dtype=np.int64)
    dof_map[free_nodes] = 3 * reduced[free_nodes, None] + np.arange(3)[None, :]
    sums: dict[tuple[int, int], list[float]] = {}
    dt_ref: float | None = None
    tet_seen = 0
    for tet in records:
        if tet.get("record") != "tetrahedron" or int(tet["environment"]) != environment:
            continue
        if not bool(int(tet["tet_flags"]) & 1):
            raise ValueError("inactive tetrahedron in complete captured operator")
        desc = descriptor.get(int(tet["material_index"]))
        if desc is None or desc.get("name") != tet.get("material_name"):
            raise ValueError("material descriptor mismatch in compensated replay")
        meta = operator_module.captured_f32_array(
            tet["center"]["dt_seconds_bulk_modulus_kind_frame_active"], (4,), "material metadata")
        dt, _, _, frame_active_num = map(float, meta)
        if dt_ref is None: dt_ref = dt
        elif dt != dt_ref: raise ValueError("mixed timestep in compensated replay")
        inv = operator_module.captured_f32_array(tet["inverse_rest_rows_per_m"], (3, 3), "inverse rest")
        volume = operator_module.captured_f32_scalar(tet["rest_volume_m3"], "rest volume")
        tangent = operator_module.captured_f32_array(
            tet["cached_algorithmic_tangent_local"]["matrix_row_major_local_dP_dF_Pa_per_dimensionless_F"],
            (81,), "tangent cache").reshape(9, 9)
        quaternion = operator_module.captured_f32_array(tet["material_frame_quaternion_xyzw"], (4,), "frame")
        frame_active = frame_active_num == 1.0
        k_local = operator_module.local_tet_stiffness(inv, volume, tangent, quaternion, frame_active)
        conn = np.asarray(tet["nodes"], dtype=np.int64)
        local_dofs = np.array([dof_map[int(n), axis] for n in conn for axis in range(3)], dtype=np.int64)
        for lr, gr in enumerate(local_dofs):
            if gr < 0: continue
            for lc, gc in enumerate(local_dofs):
                if gc < 0: continue
                value = dt * dt * float(k_local[lr, lc])
                if value != 0.0:
                    sums.setdefault((int(gr), int(gc)), []).append(value)
        tet_seen += 1
    if dt_ref is None or tet_seen == 0:
        raise ValueError("no captured active tetrahedra for compensated assembly")
    keys = list(sums)
    rows = np.fromiter((k[0] for k in keys), dtype=np.int64, count=len(keys))
    cols = np.fromiter((k[1] for k in keys), dtype=np.int64, count=len(keys))
    vals = np.fromiter((math.fsum(sums[k]) for k in keys), dtype=np.float64, count=len(keys))
    out = sparse.coo_matrix((vals, (rows, cols)), shape=(3 * len(free_nodes),) * 2).tocsr()
    out.sum_duplicates()
    return out


def main_run(args: argparse.Namespace) -> dict[str, Any]:
    case, source, artifacts, reader = (p.resolve(strict=True) for p in
        (args.case, args.source_snapshot, args.source_artifacts, args.state_reader))
    prior001, prior002 = args.prior001.resolve(strict=True), args.prior002.resolve(strict=True)
    if sha256(prior002) != args.expected_prior002_report_sha256:
        raise ValueError("prior002 hash changed from the task's sealed binding")
    prior002_manifest = prior002.parent / "manifest.json"
    if sha256(prior002_manifest) != args.expected_prior002_manifest_sha256:
        raise ValueError("prior002 manifest changed from the task's sealed binding")

    base_audit = load_module("audit497_base", args.base_audit.resolve(strict=True))
    # Re-run the sealed input/identity gates; this returns only evidence and makes no changes.
    baseline = base_audit.run(SimpleNamespace(case=case, source_snapshot=source,
        source_artifacts=artifacts, state_reader=reader))
    state_manifest = case / "diagnostics/capture-base-cgls-capture/capture.state.json"
    state_doc = base_audit.strict_json(state_manifest)
    payload_path = state_manifest.parent / state_doc["payload"]["file"]
    payload = payload_path.read_bytes()
    cgls_path = case / "diagnostics/capture-base-cgls-capture/capture"
    cgls_header, cgls = base_audit.parse_cgls_capture(cgls_path)
    fd_path = case / "diagnostics/capture-operator-fd-capture/capture"
    records = base_audit.strict_jsonl(fd_path)
    header = records[0]
    integrity = [r for r in records if r.get("record") == "integrity"]
    if len(integrity) != 1: raise ValueError("FD capture lacks one integrity record")
    source_tools = source / "matter/tools"
    op = load_module("cached_operator_497_direct", source_tools / "cardboard_cached_operator_audit.py")
    mass, dt2k, A, _, meta = op.validate_and_assemble(header,
        [r for r in records if r.get("record") == "node"],
        [r for r in records if r.get("record") == "tetrahedron"], integrity[0], 0)
    if int(cgls["selected"][0]) != 1 or int(cgls["selected"][1]) != 0:
        raise ValueError("expected selected env0 and unselected env1 for this direct-solve analysis")
    if A.shape != (2538, 2538) or A.nnz != 71046:
        raise ValueError(f"captured free operator shape changed: {A.shape}, nnz={A.nnz}")
    if not np.isfinite(A.data).all() or not np.isfinite(mass.data).all() or not np.isfinite(dt2k.data).all():
        raise ValueError("nonfinite reconstructed matrix data")

    E, N, U = (int(state_doc[k]) for k in ("environment_count", "node_count", "unknown_count"))
    row_start = 0
    desc = {a["name"]: a for a in state_doc["arrays"]}
    def state_float(name: str, shape: tuple[int, ...]) -> np.ndarray:
        d = desc[name]
        data = payload[d["offset"]:d["offset"] + d["byte_count"]]
        return np.frombuffer(data, dtype="<f4").reshape(shape).copy()
    x0_all = state_float("original_newton_correction", (U, 4))
    b_all = state_float("base_residual_rhs", (U, 4))
    accepted = state_float("accepted_nodes", (E * N, 20))
    x0 = x0_all[row_start + meta["free_nodes"], :3].astype(np.float64).reshape(-1)
    b = b_all[row_start + meta["free_nodes"], :3].astype(np.float64).reshape(-1)
    X = cgls["x"][row_start + meta["free_nodes"], :3].astype(np.float64).reshape(-1)
    native_r = cgls["r"][row_start + meta["free_nodes"], :3].astype(np.float64).reshape(-1)

    # General nonsymmetric sparse LU; no symmetry assumption is introduced.
    lu = splu(A.tocsc(), permc_spec="COLAMD", diag_pivot_thresh=1.0)
    x_star = np.asarray(lu.solve(b), dtype=np.float64)
    refinement = []
    for i in range(4):
        residual = b - np.asarray(A @ x_star).reshape(-1)
        norm_before = float(np.linalg.norm(residual))
        if norm_before == 0.0: break
        correction = np.asarray(lu.solve(residual), dtype=np.float64)
        candidate = x_star + correction
        norm_after = float(np.linalg.norm(b - np.asarray(A @ candidate).reshape(-1)))
        refinement.append({"iteration": i + 1, "residual_before_l2": norm_before,
                           "residual_after_l2": norm_after,
                           "correction_l2": float(np.linalg.norm(correction))})
        if norm_after >= norm_before:
            break
        x_star = candidate

    solve_rows = {
        "captured_original_Newton_x0": vector_metrics(A, b, x0),
        "captured_total_warm_CGLS_X": vector_metrics(A, b, X),
        "FP64_sparse_LU_solution": vector_metrics(A, b, x_star),
        "FP32_rounded_FP64_solution": vector_metrics(A, b, x_star.astype(np.float32).astype(np.float64)),
    }
    xstar_norm = float(np.linalg.norm(x_star))
    for label, value in (("captured_original_Newton_x0", x0), ("captured_total_warm_CGLS_X", X)):
        diff = value - x_star
        solve_rows[label]["forward_error_l2_vs_LU_solution"] = float(np.linalg.norm(diff))
        solve_rows[label]["relative_forward_error_l2_vs_LU_solution"] = (
            float(np.linalg.norm(diff)) / xstar_norm if xstar_norm else None)
    solve_rows["FP32_rounded_FP64_solution"]["forward_error_l2_vs_LU_solution"] = float(
        np.linalg.norm(x_star.astype(np.float32).astype(np.float64) - x_star))
    solve_rows["FP32_rounded_FP64_solution"]["relative_forward_error_l2_vs_LU_solution"] = (
        solve_rows["FP32_rounded_FP64_solution"]["forward_error_l2_vs_LU_solution"] / xstar_norm
        if xstar_norm else None)

    # Hager/Higham one-norm estimator applied to the LU inverse. It is an estimate,
    # not a rigorous upper bound; no symmetry is assumed.
    inv_op = LinearOperator(A.shape, dtype=np.float64,
        matvec=lambda v: lu.solve(np.asarray(v, dtype=np.float64)),
        rmatvec=lambda v: lu.solve(np.asarray(v, dtype=np.float64), trans="T"),
        matmat=lambda X_: lu.solve(np.asarray(X_, dtype=np.float64)),
        rmatmat=lambda X_: lu.solve(np.asarray(X_, dtype=np.float64), trans="T"))
    inv_norm_estimates = [float(onenormest(inv_op, t=2, itmax=5)) for _ in range(3)]
    norm_a1 = float(np.asarray(abs(A).sum(axis=0)).max())
    norm_a_inf = float(np.asarray(abs(A).sum(axis=1)).max())
    piv_r = np.asarray(lu.perm_r, dtype=np.int64)
    piv_c = np.asarray(lu.perm_c, dtype=np.int64)
    udiag = np.abs(lu.U.diagonal())
    max_a = float(np.max(np.abs(A.data)))
    max_u = float(np.max(np.abs(lu.U.data)))

    # Compare the ordinary COO duplicate-entry summation against accurate summation
    # of the same FP64 element terms. This isolates only global duplicate coalescing.
    k_comp = compensated_global_stiffness(op, header, records, 0,
                                           meta["free_nodes"], N)
    dK = (k_comp - dt2k).tocsr()
    dK.eliminate_zeros()
    A_comp = (mass + k_comp).tocsr()
    x_comp = np.asarray(splu(A_comp.tocsc(), permc_spec="COLAMD").solve(b), dtype=np.float64)
    comp_delta_x = x_comp - x_star
    residual_float32 = {}
    A32 = A.astype(np.float32).tocsr()
    b32 = b.astype(np.float32)
    for name, vec in (("x0", x0), ("total_X", X), ("LU_solution_rounded_to_FP32", x_star.astype(np.float32).astype(np.float64))):
        v32 = np.asarray(vec, dtype=np.float32)
        # This is a CSR FP32 arithmetic proxy, not the native element/gather order.
        r32 = np.asarray(b32 - A32 @ v32, dtype=np.float32)
        r64 = b - np.asarray(A @ vec).reshape(-1)
        residual_float32[name] = {
            "csr_float32_residual_l2": float(np.linalg.norm(r32.astype(np.float64))),
            "fp64_replay_residual_l2_for_same_stored_vector": float(np.linalg.norm(r64)),
            "difference_l2": float(np.linalg.norm(r32.astype(np.float64) - r64)),
            "scope_note": "float32 sparse-CSR arithmetic proxy; not native Metal element/gather accumulation order",
        }

    # Inertia/mask evidence from captured constraints and operator assembly.
    constraints = meta["constraint_tag_counts"]
    free_dof = int(A.shape[0])
    fixed_nodes = int(meta["fixed_node_count"])
    if free_dof != 3 * int(meta["free_node_count"]):
        raise ValueError("free xyz row count does not match free node mask")
    if int(np.count_nonzero(cgls["mask"][row_start:row_start + N, :3])) != free_dof:
        raise ValueError("captured CGLS xyz mask does not match reconstructed free xyz DOFs")

    # Drop numpy vectors from internal metric objects after extracting scalar findings.
    for row in solve_rows.values():
        row.pop("residual_vector", None)
    report = {
        "schema": "numi.cardboard.native497_cached_operator_direct_solve_conditioning.v1",
        "verdict": "CPU-only sparse direct solve and condition estimate for one captured FP32-derived FEM operator; not a constitutive reference, native FP64 execution, or full coupled/physical qualification",
        "capture": {
            "native_case": str(case), "root": state_doc["root"],
            "newton_iteration": state_doc["newton_iteration"],
            "environment": 0, "environment_count": E,
            "nodes_per_environment": N, "tets_per_environment": int(state_doc["tetrahedron_count"]),
            "operator_dimension_free_xyz_only": free_dof, "operator_nnz": int(A.nnz),
            "dt_seconds": float(meta["dt_seconds"]),
            "world_fingerprint": state_doc["world_fingerprint_u64_hex"],
            "device_program_fingerprint": state_doc["device_program_fingerprint_u64_hex"],
            "fixed_node_count": fixed_nodes, "free_node_count": int(meta["free_node_count"]),
            "constraint_tag_counts": constraints,
            "fixed_xyz_dofs_eliminated": 3 * fixed_nodes,
            "field_backing_dofs_excluded": True,
            "rigid_contact_cohesive_unknowns_excluded": True,
            "mass_diagonal_min_max": [float(mass.diagonal().min()), float(mass.diagonal().max())],
            "captured_cgls_selected_words": [int(v) for v in cgls["selected"]],
            "selected_environment_only": "env0 is selected; env1 is unselected private scratch and is excluded from solve/condition metrics",
        },
        "operator_provenance": {
            "A": "M + dt^2 K reconstructed in binary64 from captured FP32 algorithmic dP/dF cache and geometry using the frozen source182 helper",
            "mass": "lumped captured nodal masses on free xyz rows, diagonal",
            "stiffness": "captured full unsymmetric local tangent; no symmetrization",
            "not_claimed": ["constitutive re-evaluation", "native binary64 operator action", "full coupled residual Jacobian", "whole-run convergence"],
        },
        "sparse_direct_solve": {
            "method": "SciPy SuperLU sparse LU on CSC A, COLAMD ordering, default pivoting threshold 1.0",
            "iterative_refinement": refinement,
            "solve_vectors": solve_rows,
            "lu_pivot_diagnostics": {
                "row_permutation_length": int(piv_r.size), "row_permutation_unique": bool(np.unique(piv_r).size == piv_r.size),
                "column_permutation_length": int(piv_c.size), "column_permutation_unique": bool(np.unique(piv_c).size == piv_c.size),
                "row_permutation_moved_count": int(np.count_nonzero(piv_r != np.arange(piv_r.size))),
                "column_permutation_moved_count": int(np.count_nonzero(piv_c != np.arange(piv_c.size))),
                "row_permutation_sha256_le_i64": hashlib.sha256(piv_r.astype("<i8").tobytes()).hexdigest(),
                "column_permutation_sha256_le_i64": hashlib.sha256(piv_c.astype("<i8").tobytes()).hexdigest(),
                "abs_U_diagonal_min": float(udiag.min()), "abs_U_diagonal_max": float(udiag.max()),
                "abs_U_diagonal_min_over_max": float(udiag.min() / udiag.max()),
                "max_abs_A_entry": max_a, "max_abs_U_entry": max_u,
                "U_to_A_max_entry_ratio": max_u / max_a if max_a else None,
            },
        },
        "conditioning": {
            "method": "Hager-Higham 1-norm estimator via scipy.sparse.linalg.onenormest applied to SuperLU solve and transpose solve LinearOperator",
            "A_one_norm_exact_sparse": norm_a1,
            "A_infinity_norm_exact_sparse": norm_a_inf,
            "inverse_one_norm_estimates": inv_norm_estimates,
            "condition_one_estimates": [norm_a1 * v for v in inv_norm_estimates],
            "interpretation": "onenormest returns an estimate/lower estimate for inverse norm, not a certified upper bound; condition values are diagnostic estimates, not rigorous bounds",
            "parameter_t": 2, "iteration_limit": 5, "repeated_runs": 3,
        },
        "assembly_and_precision": {
            "global_duplicate_entry_resummation": {
                "method": "recompute the same FP64 local element entries from the captured FP32 cache, then math.fsum duplicate global (row,column) terms",
                "changed_stiffness_entries": int(dK.nnz),
                "stiffness_difference_l2_from_source_COO_assembly": float(sparse.linalg.norm(dK)),
                "stiffness_difference_max_abs": float(np.max(np.abs(dK.data))) if dK.nnz else 0.0,
                "relative_stiffness_difference_l2": float(sparse.linalg.norm(dK) / max(sparse.linalg.norm(dt2k), 1e-300)),
                "compensated_solution_difference_l2": float(np.linalg.norm(comp_delta_x)),
                "compensated_solution_relative_to_direct_l2": float(np.linalg.norm(comp_delta_x) / max(np.linalg.norm(x_star), 1e-300)),
                "compensated_residual_l2": float(np.linalg.norm(b - np.asarray(A_comp @ x_comp).reshape(-1))),
                "scope_note": "only global accumulation of already-computed FP64 element entries changes; local tensor/geometry arithmetic remains binary64 and identical",
            },
            "float32_sparse_action_proxy": residual_float32,
            "separate_mass_and_stiffness": {
                "mass_nnz": int(mass.nnz), "dt2_stiffness_nnz": int(dt2k.nnz),
                "mass_diagonal_l2": float(np.linalg.norm(mass.diagonal())),
                "dt2_stiffness_l2_frobenius": float(sparse.linalg.norm(dt2k)),
                "operator_l2_frobenius": float(sparse.linalg.norm(A)),
                "relative_skew_norm": float(meta["relative_skew_norm"]),
            },
        },
        "input_bindings": {
            "prior_audit_001": {"path": str(prior001), "sha256": sha256(prior001)},
            "prior_audit_002_adjudicated_report_v4": {"path": str(prior002), "sha256": sha256(prior002)},
            "prior_audit_002_manifest": {"path": str(prior002_manifest), "sha256": sha256(prior002_manifest)},
            "source_snapshot_manifest": {"path": str(source / "frozen-source.json"), "sha256": sha256(source / "frozen-source.json")},
            "cached_operator_helper": {"path": str(source_tools / "cardboard_cached_operator_audit.py"), "sha256": sha256(source_tools / "cardboard_cached_operator_audit.py")},
            "state_reader": {"path": str(reader), "sha256": sha256(reader)},
            "state_manifest": {"path": str(state_manifest), "sha256": sha256(state_manifest)},
            "state_payload": {"path": str(payload_path), "sha256": sha256(payload_path), "bytes": payload_path.stat().st_size},
            "cgls_capture": {"path": str(cgls_path), "sha256": sha256(cgls_path), "bytes": cgls_path.stat().st_size},
            "operator_fd_capture": {"path": str(fd_path), "sha256": sha256(fd_path), "bytes": fd_path.stat().st_size},
        },
        "limits": [
            "Direct solve and condition estimates apply only to the reconstructed environment-0 free-mechanical-xyz cached FEM block at captured root/Newton.",
            "The exact operator coefficients come from a captured FP32 production cache; this is not a constitutive FP64 oracle.",
            "Fixed xyz DOFs are eliminated, and all field backing, rigid, contact and cohesive rows are excluded.",
            "The condition number is estimated in the one norm; no certified upper bound or independent full-system conditioning result is provided.",
            "Global compensated resummation tests only duplicate-entry assembly; it does not emulate native Metal element/gather order or FP32 atomic arithmetic.",
            "The native run remains a failed/incomplete coupon segment with zero accepted controls; no convergence or physical qualification is implied.",
        ],
        "baseline_report_summary": {
            "baseline_verdict": baseline["verdict"],
            "source_binding": baseline["source_binding"],
            "native_trace_control": baseline["native_trace"].get("environments", []),
        },
        "native_run_binding": {
            "case": baseline["run"]["case"],
            "prediction_sha256": baseline["run"]["prediction_sha256"],
            "receipt_sha256": baseline["run"]["receipt_sha256"],
            "stderr_sha256": baseline["run"]["stderr_sha256"],
            "exit_code": baseline["run"]["exit_code"],
            "timed_out": baseline["run"]["timed_out"],
            "source_inputs_unchanged": baseline["run"]["source_inputs_unchanged"],
            "receipt_diagnostic_files_verified": baseline["run"]["receipt_diagnostic_files_verified"],
            "prediction_inputs_all_verified": baseline["run"]["verified_prediction_inputs"]["all_declared_prediction_inputs_verified"],
            "verified_prediction_input_count": baseline["run"]["verified_prediction_inputs"]["verified_count"],
            "note": "This reused the sealed 001 input-validation gate; no native run was launched.",
        },
    }
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, default=DEFAULT_CASE)
    p.add_argument("--source-snapshot", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--source-artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    p.add_argument("--state-reader", type=Path, default=DEFAULT_READER)
    p.add_argument("--base-audit", type=Path, default=BASE / "analysis/native497-cached-cgls-audit-001/audit_native497.py")
    p.add_argument("--prior001", type=Path, default=PRIOR_001)
    p.add_argument("--prior002", type=Path, default=PRIOR_002)
    p.add_argument("--expected-prior002-report-sha256", default="c6b8070c6e29c46840000ab40a43877aa9733d412a7c3d5668f14a9e3b0aa1fb")
    p.add_argument("--expected-prior002-manifest-sha256", default="f04b224849cf6127c56b5e34823e3eb14c35f57c758b774c87935a6c619bff46")
    p.add_argument("--output", type=Path, default=BASE / "analysis/native497-cached-cgls-audit-003/report-v2.json")
    args = p.parse_args()
    try:
        report = main_run(args)
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            if args.output.exists(): raise ValueError(f"refusing to overwrite {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        else:
            print(encoded, end="")
    except Exception as e:
        print(f"native497 conditioning audit rejected input: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
