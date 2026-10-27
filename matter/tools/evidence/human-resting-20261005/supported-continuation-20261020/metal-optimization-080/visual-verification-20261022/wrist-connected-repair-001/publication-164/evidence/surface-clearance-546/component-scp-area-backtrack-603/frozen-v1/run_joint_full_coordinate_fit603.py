"""Full-coordinate construction adapter retaining every existing 458 goal row."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import contextlib
import os

import numpy as np
import component_scp_backtracking603 as _component_scp603
from scipy import sparse

from joint_objective_560 import retain_joint_rows560
D = Path("/Users/numi/Work/numi-human-connected-surface-069")
ENTRY = D / "D/D/component-lp-decomposition-490/run_matched_component_fit_490.py"
BASE_ACTIVE_COMPONENT = D / "D/contained-clearance-545/component_lp_active545.py"
ACTIVE_COMPONENT = Path(__file__).with_name("component_lp_partial_timeout564.py")
SURFACE_HELPER = D / "D/surface-clearance-546/surface_objective_546.py"
EXPECTED_ENTRY_SHA = "7719b5042d06abc8211400bd82fc5fed89d3d388e5a1c0794ddd7a0e084bf693"
EXPECTED_BASE_ACTIVE_SHA = "37c8a7e22f240be1a669e0f29712b0dfb068861fd89c9b6b2a8955c8ffdf39ed"
EXPECTED_ACTIVE_SHA = "a24aa58178394e25d7b38c7c35a36c93eaf109a666ace7773c0ca13d33250818"
EXPECTED_SURFACE_HELPER_SHA = "dd6f70afab9cb51f3658d8f4fc9d65a13a6f70517e536a9e81c4c2a098750516"
EXPECTED_JOINT_HELPER_SHA = "09ce0c4f47e69142de6ce394576edd3ef0ee6f4b946824a476f5425919c492c8"
CAPTURE_HELPER = D / "D/surface-clearance-546/raw-joint-capture-563/raw_joint_capture563.py"
EXPECTED_CAPTURE_HELPER_SHA = "fa7df8771a37200c4ed30e61d92ebf1b6227eed9da8a5ff45ab54b4ff299975e"
COMPONENT_SCP_HELPER = Path(__file__).with_name("component_scp_backtracking603.py")
EXPECTED_COMPONENT_SCP_SHA = "f02aed5e462779e85a5d4df46a38a68bcf7eb3972c9dbf565830c4919b387be6"
PARTIAL_TIMEOUT_FLAG = "--allow-partial-timeout-zero-fallback-564"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if sha(SURFACE_HELPER) != EXPECTED_SURFACE_HELPER_SHA:
    raise RuntimeError("pinned 546 outside-depth helper hash changed")
_surface_spec = importlib.util.spec_from_file_location("surface_objective_546_for_joint560", SURFACE_HELPER)
_surface_module = importlib.util.module_from_spec(_surface_spec)
_surface_spec.loader.exec_module(_surface_module)
select_outside_depth_rows546 = _surface_module.select_outside_depth_rows546
if sha(CAPTURE_HELPER) != EXPECTED_CAPTURE_HELPER_SHA:
    raise RuntimeError("pinned 563 raw capture helper hash changed")
_capture_spec = importlib.util.spec_from_file_location("raw_joint_capture563_for_562", CAPTURE_HELPER)
_capture_module = importlib.util.module_from_spec(_capture_spec)
_capture_spec.loader.exec_module(_capture_module)


def _flag_value(flag, default=None):
    if flag not in sys.argv:
        return default
    index = sys.argv.index(flag)
    if index + 1 >= len(sys.argv):
        raise ValueError(f"missing value after {flag}")
    return sys.argv[index + 1]


def _matched_input_pins():
    pins = {}
    file_flags = (
        "--base-candidate", "--raw-trial", "--self-pairs", "--area-reference-npz",
        "--bary-owner", "--point-target-npz", "--point-target-report",
        "--face-target-npz", "--face-target-report", "--point-depth-rows-npz",
        "--point-depth-rows-report", "--point-depth-owner", "--source-base-npz",
        "--fixed-boundary-npz", "--fixed-boundary-report", "--fixed-boundary-owner",
        "--pair-owner", "--area-owner", "--matched-inputs-json-490", "--audit159-490",
    )
    for flag in file_flags:
        value = _flag_value(flag)
        if value is None:
            continue
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{flag} input does not exist: {path}")
        pins[str(path)] = sha(path)
    matched = _flag_value("--matched-inputs-json-490")
    if matched is not None:
        document = json.loads(Path(matched).read_text())
        for section in ("inputs", "source_pins"):
            for record in document.get(section, {}).values():
                if not isinstance(record, dict) or "path" not in record or "sha256" not in record:
                    continue
                path = Path(record["path"]).resolve()
                expected = str(record["sha256"])
                if not path.is_file() or sha(path) != expected:
                    raise ValueError(f"matched input pin mismatch: {path}")
                pins[str(path)] = expected
    return pins


def _basis_snapshot_for_partial(token):
    return {
        "base_candidate": token["base_candidate"], "base_sha256": token["base_sha256"],
        "capture_helper_sha256": token["capture_helper_sha256"],
        "output_scope_rows": token["output_scope_rows"], "anchor_rows": token["anchor_rows"],
        "control_parent_rows": token["control_parent_rows"],
        "control_scope_positions": token["control_scope_positions"],
        "anchor_scope_positions": token["anchor_scope_positions"],
        "basis_csr": token["basis_csr"], "output_dimension": token["output_dimension"],
        "control_dimension": token["control_dimension"], "units": token["units"],
        "_base_points_world_m_f32": token["base_points_world_m_f32"].copy(),
    }


def capture_partial_solver_direction(result, partial, basis_token, capture_module, records,
                                     fit_receipt_factory):
    """Capture a failed-but-feasible partial direction without changing failure status."""
    delta = np.asarray(partial["delta"], dtype=np.float64).reshape(-1)
    q = np.asarray(partial["basis_coordinates"], dtype=np.float64).reshape(-1)
    if not np.isfinite(delta).all() or not np.isfinite(q).all():
        raise ValueError("partial timeout direction contains nonfinite coordinates")
    if delta.shape != (basis_token["output_dimension"],):
        raise ValueError("partial timeout delta does not match captured full output scope")
    if q.shape != (basis_token["control_dimension"],):
        raise ValueError("partial timeout basis coordinates do not match captured controls")
    csr = basis_token["basis_csr"]
    Q = sparse.csr_matrix((csr["data"], csr["indices"], csr["indptr"]), shape=csr["shape"])
    expanded = sparse.kron(Q, sparse.eye(3, format="csr"), format="csr")
    reconstruction_error = float(np.max(np.abs(np.asarray(expanded @ q).reshape(-1) - delta), initial=0.0))
    if reconstruction_error > 1e-7:
        raise ValueError("partial timeout delta does not reconstruct through captured basis")
    anchors = np.asarray(basis_token["anchor_scope_positions"], dtype=np.int64)
    anchor_delta = delta.reshape((-1, 3))[anchors]
    anchor_max = float(np.max(np.abs(anchor_delta), initial=0.0))
    if anchor_max > 1e-12:
        raise ValueError("partial timeout direction moves fixed anchor rows")
    receipt_input = dict(result)
    receipt_input["partial_timeout_fallback"] = partial["partial_receipt"]
    fit_receipt = fit_receipt_factory(receipt_input)
    partial_status = partial["partial_receipt"]
    zeroed_components = list(partial_status.get("zeroed_components", []))
    timed_out_components = list(partial_status.get("timed_out_components", []))
    budget_components = list(partial_status.get("budget_exhausted_components", []))
    if not zeroed_components:
        raise ValueError("partial fallback receipt has no zeroed components")
    if timed_out_components and budget_components:
        stage = "joint_lp_partial_timeout_and_budget_zero_fallback"
    elif timed_out_components:
        stage = "joint_lp_partial_timeout_zero_fallback"
    else:
        stage = "joint_lp_partial_budget_exhausted_zero_fallback"
    component = int(zeroed_components[0]) if len(zeroed_components) == 1 else None
    record = {
        "status": "PARTIAL_FEASIBLE_TIMEOUT_DIRECTION_CAPTURED_UNADMITTED",
        "delta_mm": delta.copy(), "basis_coordinates": q.copy(),
        "basis_snapshot": _basis_snapshot_for_partial(basis_token),
        "solver_status": int(result.get("status", 1)),
        "solver_message": str(result.get("message", "")), "solver_objective": None,
        "solver_wall_seconds": result.get("wall_seconds"), "delta_shape": list(delta.shape),
        "delta_captured": True, "stage": stage,
        "component": component,
        "fit_receipt": fit_receipt,
        "basis_coordinate_reconstruction_residual_max_abs_mm": reconstruction_error,
        "raw_anchor_delta_max_mm": anchor_max,
        "partial_fallback_receipt": partial["partial_receipt"],
        "feasibility_certificate": partial["feasibility_certificate"],
        "solver_receipt": {
            "success": False, "status": int(result.get("status", 1)),
            "message": str(result.get("message", "")),
            "partial_feasible_direction_emitted_as_offline_construction_only": True,
            "global_lp_converged": False, "retained_gain_budget_claimed": False,
            "partial_fallback_receipt": partial["partial_receipt"],
            "feasibility_certificate": partial["feasibility_certificate"],
        },
    }
    records.append(record)


def wrap_solver_for_raw_capture(original_solve, capture_module, basis_tokens, records,
                                fit_receipt_factory, *, partial_timeout_opt_in=False):
    def wrapped(*args, **kwargs):
        kwargs["allow_partial_timeout_zero_fallback"] = bool(partial_timeout_opt_in)
        result = original_solve(*args, **kwargs)
        if isinstance(result, dict):
            partial = result.pop("_partial_timeout_direction_564", None)
            if partial is not None:
                if not partial_timeout_opt_in:
                    raise ValueError("partial timeout direction appeared without explicit opt-in")
                if not basis_tokens:
                    raise ValueError("partial timeout direction has no captured full basis token")
                capture_partial_solver_direction(
                    result, partial, basis_tokens[-1], capture_module, records, fit_receipt_factory
                )
            elif result.get("success"):
                if not basis_tokens:
                    raise ValueError("successful joint LP return has no captured basis token")
                capture_module.capture_solver_return(
                    result, basis_tokens[-1], fit_receipt=fit_receipt_factory(result),
                    stage="joint_lp_raw_before_area_alpha", component=None, records=records
                )
        return result
    return wrapped

def main():
    if sys.argv.count(PARTIAL_TIMEOUT_FLAG) != 1:
        raise ValueError(f"explicit opt-in flag required exactly once: {PARTIAL_TIMEOUT_FLAG}")
    sys.argv.remove(PARTIAL_TIMEOUT_FLAG)
    partial_timeout_opt_in = True
    helper_path = Path(__file__).with_name("joint_objective_560.py")
    frozen = {
        ENTRY: EXPECTED_ENTRY_SHA,
        BASE_ACTIVE_COMPONENT: EXPECTED_BASE_ACTIVE_SHA,
        ACTIVE_COMPONENT: EXPECTED_ACTIVE_SHA,
        SURFACE_HELPER: EXPECTED_SURFACE_HELPER_SHA,
        CAPTURE_HELPER: EXPECTED_CAPTURE_HELPER_SHA,
        COMPONENT_SCP_HELPER: EXPECTED_COMPONENT_SCP_SHA,
    }
    if any(sha(path) != expected for path, expected in frozen.items()):
        raise RuntimeError("a pinned 490/545/546/563/564 owner source changed")
    helper_sha = sha(helper_path)
    if helper_sha != EXPECTED_JOINT_HELPER_SHA:
        raise RuntimeError("joint objective helper hash changed")

    spec = importlib.util.spec_from_file_location("fit490_joint564", ENTRY)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    original_load = entry._load_module
    receipt = {}
    capture_records = []
    basis_tokens = []
    base_candidate_path = Path(_flag_value("--base-candidate")).resolve()
    matched_pins = _matched_input_pins()

    def make_fit_receipt(solver_result):
        row_receipt = dict(receipt)
        row_receipt["pinned_row_adapter_sha256"] = helper_sha
        row_receipt["joint_goal_rows_retained"] = True
        workload = solver_result.get("component_decomposition_475", {})
        if not isinstance(workload, dict) or not workload:
            raise ValueError("successful joint solver return lacks component workload receipt")
        labels = list(receipt.get("row_kinds", []))
        if not labels:
            if receipt.get("surface_soft_rows_retained", 0):
                labels.append("face")
            if receipt.get("point_soft_rows_retained", 0):
                labels.append("point")
        if not labels:
            raise ValueError("joint solver return lacks point/face goal labels")
        policy_flags = (
            "--trust-mm", "--source-cap-mm", "--area-floor", "--max-edge-gradient",
            "--margin-fraction", "--area-scp-iterations", "--time-limit-s",
            "--lp-method", "--retained-gain-fraction",
        )
        policy = {flag.lstrip("-").replace("-", "_"): _flag_value(flag) for flag in policy_flags}
        partial = solver_result.get("partial_timeout_fallback")
        if partial:
            row_receipt["partial_timeout_feasibility_only"] = True
        policy.update({
            "partial_timeout_zero_fallback_opt_in": bool(partial_timeout_opt_in),
            "max_total_wall_s": 120.0,
            "component_wall_budget_policy": workload.get("wall_budget_allocation_policy"),
            "soft_target_regression_opt_in": "--allow-soft-target-regression-485" in sys.argv,
            "native_admission": False,
        })
        return {
            "fit_entry_path": str(Path(__file__).resolve()),
            "fit_entry_sha256": self_sha,
            "objective_contract": (
                "the 490/458 builder C, h, weights, row order, and metadata are unchanged; "
                "this is a feasible partial timeout-fallback construction direction, not a "
                "converged or globally optimal LP"
                if solver_result.get("partial_timeout_fallback")
                else "the 490/458 builder C, h, weights, row order, and metadata are returned unchanged for both point and face goals"
            ),
            "goal_labels": labels,
            "delta_semantics": "absolute_base_relative_millimeters",
            "matched_input_pins": dict(matched_pins),
            "row_receipt": row_receipt,
            "workload_receipt": workload,
            "policy_receipt": policy,
        }

    def load490(path):
        runner = original_load(path)
        if Path(path).resolve() != entry.RUNNER_485.resolve():
            return runner
        original_runner_load = runner.load

        def load485(name, module_path):
            module = original_runner_load(name, module_path)
            if name == "component_lp_decomposition_485_for_run":
                spec_active = importlib.util.spec_from_file_location(
                    "component_partial_timeout564", ACTIVE_COMPONENT
                )
                active = importlib.util.module_from_spec(spec_active)
                spec_active.loader.exec_module(active)
                module.solve_by_components = wrap_solver_for_raw_capture(
                    active.solve_by_components,
                    _capture_module,
                    basis_tokens,
                    capture_records,
                    make_fit_receipt,
                    partial_timeout_opt_in=partial_timeout_opt_in,
                )
                receipt["active_component_adapter_source_sha256"] = sha(ACTIVE_COMPONENT)
                receipt["partial_timeout_zero_fallback_opt_in"] = partial_timeout_opt_in
                receipt["capture_helper_source_sha256"] = sha(CAPTURE_HELPER)
                return module
            if name != "matched458_for_485":
                return module
            install = module.install_objective

            def install_joint(owner, objective, crossing):
                result = install(owner, objective, crossing)
                component_receipt = _component_scp603.install(owner, _flag_value("--self-pairs"))
                receipt["component_scp_backtracking603"] = component_receipt
                basis = owner.build_immediate_parent_basis
                build = owner.build_clearance_rows
                depth = owner.build_output_node_direction_rows

                def full_basis(base, support, anchors):
                    B, parent, basis_report = basis(base, support, anchors)
                    scope = np.asarray(basis_report["output_scope_rows"], dtype=np.int64)
                    free = np.flatnonzero(~np.isin(scope, anchors))
                    Q = sparse.csr_matrix(
                        (np.ones(len(free)), (free, np.arange(len(free)))),
                        shape=(len(scope), len(free)),
                    )
                    basis_report.update(
                        control_parent_rows=scope[free].tolist(),
                        matrix_shape=list(Q.shape),
                        matrix_nnz=Q.nnz,
                        semantics=(
                            "independent emitted-node construction coordinates on the "
                            "existing expanded scope; source ancestry and native "
                            "mechanical coordinates unchanged"
                        ),
                        construction_full_output_coordinates=True,
                    )
                    control_rows = scope[free]
                    receipt["construction_coordinate_nodes"] = int(len(free))
                    basis_token = _capture_module.capture_basis(
                        base_candidate_path,
                        scope,
                        np.asarray(anchors, dtype=np.int64),
                        control_rows,
                        basis_matrix=Q,
                    )
                    basis_tokens.append(basis_token)
                    receipt["captured_basis_summary"] = {
                        "base_candidate": basis_token["base_candidate"],
                        "base_sha256": basis_token["base_sha256"],
                        "output_scope_count": len(basis_token["output_scope_rows"]),
                        "anchor_count": len(basis_token["anchor_rows"]),
                        "control_count": len(basis_token["control_parent_rows"]),
                        "basis_csr_shape": basis_token["basis_csr"]["shape"],
                        "basis_csr_representation": basis_token["basis_csr"]["representation"],
                        "units": basis_token["units"],
                    }
                    return Q, control_rows, basis_report

                def joint_rows(*args, **kwargs):
                    C, h, weights, metadata, proof = retain_joint_rows560(
                        build, *args, **kwargs
                    )
                    receipt.update(proof)
                    return C, h, weights, metadata

                def outside_rows(node_count, support, node_rows, normals, lower, typed_ids=None):
                    rows, dirs, lo, ids, proof = select_outside_depth_rows546(
                        node_rows, normals, lower, typed_ids
                    )
                    receipt.update(proof)
                    return depth(node_count, support, rows, dirs, lo, ids)

                owner.build_immediate_parent_basis = full_basis
                owner.build_clearance_rows = joint_rows
                owner.build_output_node_direction_rows = outside_rows
                return result

            module.install_objective = install_joint
            return module

        runner.load = load485
        return runner

    entry._load_module = load490
    self_sha = sha(__file__)
    capture_sha = sha(CAPTURE_HELPER)
    try:
        with open(os.devnull, "w") as sink:
            with contextlib.redirect_stdout(sink):
                entry.main()
    finally:
        output = Path(sys.argv[sys.argv.index("--output") + 1])
        if capture_records:
            final_receipt = capture_records[-1]["fit_receipt"]
            capture_dir = output / "raw-joint-capture-563"
            summary_path, summary = _capture_module.finalize_capture(
                base_candidate_path,
                capture_dir,
                capture_records,
                fit_receipt=final_receipt,
            )
            receipt["raw_joint_capture_summary_path"] = str(summary_path)
            receipt["raw_joint_capture_summary_sha256"] = sha(summary_path)
            receipt["raw_joint_capture_status"] = summary.get("status")
            receipt["raw_joint_capture_captured_delta_count"] = int(
                summary.get("successful_delta_count", 0)
            )
    if (sha(__file__) != self_sha or sha(helper_path) != helper_sha
            or sha(CAPTURE_HELPER) != capture_sha or sha(COMPONENT_SCP_HELPER) != EXPECTED_COMPONENT_SCP_SHA):
        raise RuntimeError("joint or raw capture adapter changed during fit invocation")
    if any(sha(path) != expected for path, expected in frozen.items()):
        raise RuntimeError("a pinned 490/545/546/563/564 owner changed during fit invocation")
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    report["full_coordinate_partial_timeout564"] = {
        **receipt,
        "source_sha256": self_sha,
        "joint_objective_helper_sha256": helper_sha,
        "pinned_owner_sha256": {
            "matched_490_entry": EXPECTED_ENTRY_SHA,
            "545_component_source": EXPECTED_BASE_ACTIVE_SHA,
            "564_partial_timeout_component_adapter": EXPECTED_ACTIVE_SHA,
            "546_outside_depth_selector": EXPECTED_SURFACE_HELPER_SHA,
            "560_joint_row_adapter": EXPECTED_JOINT_HELPER_SHA,
            "563_raw_capture_helper": EXPECTED_CAPTURE_HELPER_SHA,
            "603_component_scp_backtracking_helper": EXPECTED_COMPONENT_SCP_SHA,
        },
        "construction_only": True,
        "native_admission": False,
        "max_total_wall_s_unchanged": 120.0,
        "lp_equations_and_gate_sources_unchanged": True,
        "raw_joint_directions_are_unadmitted": True,
        "raw_joint_capture_record_count": len(capture_records),
        "partial_timeout_direction_count": sum(
            1 for record in capture_records
            if str(record.get("stage", "")).startswith("joint_lp_partial_")
        ),
        "partial_timeout_is_solver_failure_not_global_success": True,
        "retained_gain_budget_claimed_for_partial": False,
        "accepted_physical_steps": 0,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
