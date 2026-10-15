#!/usr/bin/env python3
"""Read-only audit for the private CGLS snapshot emitted by source142.

This checks native layout/status/row masking and, when supplied, compares the
captured action on the native x and r against a separately retained rich
operator capture.  The action comparison is deliberately not promoted to a
whole-matrix identity or solver/physical qualification.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any
import numpy as np

MAGIC = 0x534C4743
HEADER_WORDS = 8
SEGMENT_COUNT = 12
VECTOR_NAMES = ("x", "r", "s", "p", "Ax", "b", "mask")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def align16(value: int) -> int:
    return (value + 15) & ~15


def parse_snapshot(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    raw = Path(path).read_bytes()
    if len(raw) < HEADER_WORDS * 4:
        raise ValueError("CGLS snapshot is shorter than its eight-word header")
    magic, version, env_count, nodes_per_env, unknown_count, cap, nsegments, reserved = struct.unpack_from("<8I", raw)
    if magic != MAGIC or version != 1 or nsegments != SEGMENT_COUNT or reserved != 0:
        raise ValueError(f"unsupported CGLS header: {(magic, version, nsegments, reserved)}")
    if env_count == 0 or nodes_per_env == 0 or unknown_count != 2 * env_count * nodes_per_env:
        raise ValueError("CGLS header violates U = 2 * E * N")
    if cap not in (32, 64):
        raise ValueError(f"unsupported CGLS cap {cap}")
    specs: list[tuple[str, str, int, tuple[int, ...]]] = []
    for name in VECTOR_NAMES:
        specs.append((name, "<f4", 4 * unknown_count, (unknown_count, 4)))
    specs.extend([
        ("row_environment", "<u4", unknown_count, (unknown_count,)),
        ("scalars", "<f4", 4 * env_count, (env_count, 4)),
        ("control", "<u4", 4 * env_count, (env_count, 4)),
        ("errors", "<u4", env_count, (env_count,)),
        ("selected", "<u4", env_count, (env_count,)),
    ])
    arrays: dict[str, np.ndarray] = {}
    segments: list[dict[str, int | str]] = []
    offset = HEADER_WORDS * 4
    for name, dtype, count, shape in specs:
        start = align16(offset)
        if start > len(raw) or any(raw[offset:start]):
            raise ValueError(f"nonzero/truncated alignment padding before {name}")
        byte_count = count * 4
        end = start + byte_count
        if end > len(raw):
            raise ValueError(f"truncated CGLS segment {name}")
        arrays[name] = np.frombuffer(raw, dtype=np.dtype(dtype), count=count, offset=start).reshape(shape).copy()
        segments.append({"name": name, "offset": start, "bytes": byte_count})
        offset = end
    if offset != len(raw):
        raise ValueError(f"unexpected trailing bytes: expected {offset}, got {len(raw)}")
    return ({"magic_hex": f"0x{magic:08x}", "version": version,
             "environment_count": env_count, "nodes_per_environment": nodes_per_env,
             "unknown_count": unknown_count, "iteration_cap": cap,
             "segment_count": nsegments, "file_bytes": len(raw), "sha256": sha(path),
             "segments": segments}, arrays)


def compare(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.shape != bb.shape:
        return {"shape_match": False, "left_shape": list(aa.shape), "right_shape": list(bb.shape)}
    d = aa - bb
    if not np.isfinite(d).all():
        return {"shape_match": True, "finite": False,
                "nonfinite_count": int(d.size - np.count_nonzero(np.isfinite(d)))}
    dn, bn = float(np.linalg.norm(d)), float(np.linalg.norm(bb))
    return {"shape_match": True, "finite": True, "difference_l2": dn,
            "reference_l2": bn, "relative_l2": dn / bn if bn else None,
            "max_abs": float(np.max(np.abs(d))) if d.size else 0.0,
            "threshold": None, "meaning": "descriptive action comparison; no noise-floor or qualification gate"}


def compare_float32_words(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Compare values after canonical FP32 conversion, including signed-zero bits."""
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    if aa.shape != bb.shape:
        return {"shape_match": False, "left_shape": list(aa.shape), "right_shape": list(bb.shape)}
    aw, bw = aa.view(np.uint32), bb.view(np.uint32)
    mismatch = np.flatnonzero(aw != bw)
    return {
        "shape_match": True,
        "left_count": int(aa.size),
        "bitwise_word_match": bool(mismatch.size == 0),
        "mismatch_count": int(mismatch.size),
        "first_mismatch_index": int(mismatch[0]) if mismatch.size else None,
        "first_left_word_hex": f"0x{int(aw[mismatch[0]]):08x}" if mismatch.size else None,
        "first_right_word_hex": f"0x{int(bw[mismatch[0]]):08x}" if mismatch.size else None,
        "numeric_max_abs_difference_after_fp32_cast": float(np.max(np.abs(aa.astype(np.float64) - bb.astype(np.float64)))) if aa.size else 0.0,
        "meaning": "FP32-cast word comparison only; source JSON may not preserve signed-zero spelling, so this is reported rather than treated as an unconditional capture-integrity gate",
    }


def raw_zero(a: np.ndarray) -> bool:
    return bool(np.all(np.asarray(a).view(np.uint32) == 0))


def existing_action_gate(result: dict[str, Any]) -> dict[str, Any]:
    relative = result.get("relative_l2")
    reference_norm = result.get("reference_l2")
    resolved = (isinstance(relative, (int, float)) and np.isfinite(relative) and
                isinstance(reference_norm, (int, float)) and np.isfinite(reference_norm) and
                float(reference_norm) > 0.0)
    return {"threshold_relative_l2": 0.01,
            "applicable_for_nonzero_finite_reference_action": bool(resolved),
            "relative_error": relative,
            "absolute_difference_l2": result.get("difference_l2"),
            "reference_action_l2": reference_norm,
            "within_existing_1_percent_gate": bool(resolved and relative <= 0.01),
            "meaning": "applies the existing 1% assembled-action gate to this bound nonzero vector action; it is not a whole-matrix or solver certificate and adds no absolute noise-floor claim"}


def load_audit_module(snapshot: Path):
    path = snapshot / "matter/tools/cardboard_cached_operator_audit.py"
    if not path.is_file():
        raise ValueError(f"missing source-bound operator audit module {path}")
    spec = importlib.util.spec_from_file_location("source_bound_cardboard_operator_audit", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import source-bound operator audit module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def verify_prediction_inputs(module, prediction_path: Path, prediction: dict,
                             snapshot: Path) -> dict[str, Any]:
    if Path(prediction.get("source_snapshot", "")).resolve() != snapshot.resolve():
        raise ValueError("prediction source_snapshot does not equal retained source snapshot")
    inputs = prediction.get("inputs_sha256")
    if not isinstance(inputs, dict):
        raise ValueError("prediction lacks source/input hash map")
    runtime_value = prediction.get("runtime_dir")
    runtime = Path(runtime_value) if isinstance(runtime_value, str) and runtime_value else None
    checked, missing = 0, []
    for relative, expected in inputs.items():
        candidate, _ = module._manifest_file(relative, snapshot, runtime, prediction)
        if candidate is None or not candidate.is_file():
            missing.append(relative)
            continue
        if module.sha256(candidate) != expected:
            raise ValueError(f"prediction input hash mismatch: {relative}")
        checked += 1
    if missing:
        raise ValueError(f"prediction inputs unavailable: {missing[:8]}")
    return {"input_count": len(inputs), "verified_count": checked,
            "all_prediction_inputs_verified": checked == len(inputs)}


def parse_markers(stderr: Path, prefix: str) -> list[dict]:
    out = []
    for line in stderr.read_text(errors="replace").splitlines():
        if line.startswith(prefix):
            out.append(json.loads(line.split("=", 1)[1]))
    return out


def summarize_pretarget_trace(run_records: list[dict], baseline_records: list[dict],
                              root: int, target_iteration: int) -> dict[str, Any]:
    def indexed(records: list[dict]) -> dict[int, dict]:
        chosen = [r for r in records if int(r.get("root", -1)) == root and
                  0 <= int(r.get("newton_iteration", -1)) < target_iteration]
        out: dict[int, dict] = {}
        duplicates: list[int] = []
        for r in chosen:
            it = int(r["newton_iteration"])
            if it in out:
                duplicates.append(it)
            out[it] = r
        if duplicates:
            raise ValueError(f"duplicate pretarget Armijo records for root {root}: {sorted(set(duplicates))}")
        return out
    actual = indexed(run_records)
    baseline = indexed(baseline_records)
    iters = sorted(set(actual) | set(baseline))
    mismatches = [i for i in iters if i not in actual or i not in baseline or actual.get(i) != baseline.get(i)]
    canonical_actual = [json.dumps(actual[i], sort_keys=True, separators=(",", ":"), allow_nan=False)
                        for i in sorted(actual)]
    canonical_baseline = [json.dumps(baseline[i], sort_keys=True, separators=(",", ":"), allow_nan=False)
                          for i in sorted(baseline)]
    missing = list(range(target_iteration))
    missing_actual = [i for i in missing if i not in actual]
    missing_baseline = [i for i in missing if i not in baseline]
    return {
        "classification": "exact parsed residual-Armijo record parity before the selected Newton; it does not prove every internal solver state or physical history byte-identical",
        "root": root, "target_newton_iteration_exclusive": target_iteration,
        "actual_record_count": len(actual), "baseline_record_count": len(baseline),
        "expected_iteration_count": target_iteration,
        "missing_actual_iterations": missing_actual, "missing_baseline_iterations": missing_baseline,
        "mismatch_iterations": mismatches,
        "actual_canonical_record_sha256": hashlib.sha256("\n".join(canonical_actual).encode()).hexdigest(),
        "baseline_canonical_record_sha256": hashlib.sha256("\n".join(canonical_baseline).encode()).hexdigest(),
        "exact_pretarget_trace_parity": bool(not mismatches and not missing_actual and not missing_baseline and len(actual) == target_iteration),
    }


def summarize_coupled_capture_scope(stderr: Path, root: int, iteration: int) -> dict[str, Any]:
    skipped = parse_markers(stderr, "fem_coupled_capture_skipped=")
    captured = parse_markers(stderr, "fem_coupled_capture=")
    selected_skips = [m for m in skipped if int(m.get("root", -1)) == root and
                      int(m.get("iteration", -1)) == iteration]
    selected_captures = [m for m in captured if int(m.get("root", -1)) == root and
                         int(m.get("iteration", -1)) == iteration]
    return {
        "root": root, "newton_iteration": iteration,
        "completed_markers": selected_captures,
        "skipped_markers": selected_skips,
        "classification": ("completed rich operator capture marker present" if selected_captures else
                            "runtime emitted a bounded skip marker" if selected_skips else
                            "no completed or skipped rich capture marker; absence is not interpreted as success"),
    }


def summarize_armijo(records: list[dict], environment: int, root: int,
                     iteration: int) -> dict[str, Any]:
    selected = [m for m in records
        if int(m.get("root", -1)) == root and int(m.get("newton_iteration", -1)) == iteration]
    result: dict[str, Any] = {
        "record_count_for_selected_root_newton": len(selected),
        "records": [],
        "accepted_non_noop_CGLS_trial": False,
        "classification": "ordinary residual Armijo evidence only; no original Newton certificate transfers to the CGLS direction",
    }
    for record in selected:
        environments = record.get("environments", [])
        row = environments[environment] if isinstance(environments, list) and environment < len(environments) else None
        is_cgls = record.get("direction_kind") == "base_cgls"
        accepted_non_noop = bool(is_cgls and isinstance(row, dict) and
            row.get("accepted") is True and row.get("certified_noop") is False and
            isinstance(row.get("selected_alpha"), (int, float)) and
            float(row["selected_alpha"]) > 0.0)
        result["records"].append({
            "direction_kind": record.get("direction_kind"),
            "selected_environment": row,
            "cgls_direction": is_cgls,
            "accepted_non_noop_CGLS_trial": accepted_non_noop,
        })
        result["accepted_non_noop_CGLS_trial"] |= accepted_non_noop
    return result


def bind_run(run_dir: Path, source_snapshot: Path, module):
    prediction_path = run_dir / "prediction.json"
    receipt_path = run_dir / "receipt.json"
    stderr_path = run_dir / "stderr.log"
    for p in (prediction_path, receipt_path, stderr_path):
        if not p.is_file():
            raise ValueError(f"run is incomplete; missing {p}")
    pred = json.loads(prediction_path.read_text())
    rec = json.loads(receipt_path.read_text())
    if rec.get("source_inputs_unchanged") is not True:
        raise ValueError(f"{run_dir} receipt does not attest unchanged inputs")
    if rec.get("timed_out") is True:
        raise ValueError(f"{run_dir} receipt says the process timed out")
    checked = verify_prediction_inputs(module, prediction_path, pred, source_snapshot)
    return pred, rec, stderr_path, checked


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    cgls_binary = args.binary.resolve()
    source_snapshot = args.source_snapshot.resolve()
    for p in (cgls_binary, run_dir / "output/compiled.nmatterpack", run_dir / "output/manifest.json"):
        if not p.is_file():
            raise ValueError(f"required native result input missing: {p}")
    source_module, source_module_path = load_audit_module(source_snapshot)
    pred, rec, stderr_path, input_checks = bind_run(run_dir, source_snapshot, source_module)
    snapshot_header, arrays = parse_snapshot(cgls_binary)
    env_count, N, U = (snapshot_header[k] for k in ("environment_count", "nodes_per_environment", "unknown_count"))
    if snapshot_header["iteration_cap"] != int(pred["environment"].get("NM_FEM_BASE_CGLS_ITERATIONS", -1)):
        raise ValueError("snapshot CGLS cap disagrees with sealed environment setting")
    if int(pred["environment"].get("NM_FEM_BASE_CGLS_CAPTURE_NEWTON", -1)) < 0:
        raise ValueError("prediction has invalid CGLS capture selector")

    cgls_captures = parse_markers(stderr_path, "fem_base_cgls_capture=")
    cgls_directions = parse_markers(stderr_path, "fem_base_cgls_direction=")
    iteration = int(pred["environment"]["NM_FEM_BASE_CGLS_CAPTURE_NEWTON"])
    selector = pred["environment"].get("NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR")
    if not isinstance(selector, str) or re.fullmatch(r"\d+:\d+", selector) is None:
        raise ValueError("prediction must bind CGLS root through the Cauchy fallback selector")
    expected_root, selector_iteration = (int(v) for v in selector.split(":"))
    if selector_iteration != iteration:
        raise ValueError("Cauchy and CGLS selectors name different Newton iterations")
    # Current source142 uses the Cauchy selector's root when CGLS capture only
    # supplies a Newton selector.  Bind the unique marker root/iteration pair.
    matching = [m for m in cgls_captures if m.get("newton_iteration") == iteration]
    matching_dir = [m for m in cgls_directions if m.get("newton_iteration") == iteration]
    if len(matching) != 1 or len(matching_dir) != 1:
        raise ValueError("expected exactly one CGLS capture and direction marker for configured Newton")
    cm, dm = matching[0], matching_dir[0]
    root = int(cm.get("root", -1))
    if dm.get("root") != root:
        raise ValueError("CGLS capture and direction marker roots differ")
    if root != expected_root:
        raise ValueError("CGLS root differs from the sealed selector")
    if cm.get("complete") is not True or cm.get("physical_state_changed") is not False:
        raise ValueError("CGLS capture is incomplete or claims physical mutation")
    if int(cm.get("bytes", -1)) != cgls_binary.stat().st_size or int(cm.get("expected_bytes", -1)) != cgls_binary.stat().st_size:
        raise ValueError("CGLS marker byte counts do not match snapshot")
    if dm.get("direction_kind") != "truncated_unregularized_CGLS" or dm.get("original_newton_certificate_applies_to_direction") is not False:
        raise ValueError("direction marker has unexpected kind or transfers prior certificate")
    if int(dm.get("iteration_budget", -1)) != snapshot_header["iteration_cap"]:
        raise ValueError("direction marker cap disagrees with snapshot")

    compiled_pack = run_dir / "output/compiled.nmatterpack"
    run_manifest = json.loads((run_dir / "output/manifest.json").read_text())
    run_world = int(run_manifest["compiled_world_fingerprint"])
    run_world_hex = f"{run_world:016x}"
    reference = None
    reference_free_nodes = None
    action_checks: dict[str, Any] = {"available": False,
        "reason": "no independently bound same-world/root/Newton rich operator capture supplied"}
    if args.reference_run_dir is not None:
        ref_dir = args.reference_run_dir.resolve()
        ref_snapshot_pred = json.loads((ref_dir / "prediction.json").read_text())
        ref_snapshot = Path(ref_snapshot_pred["source_snapshot"]).resolve()
        ref_module, ref_module_path = load_audit_module(ref_snapshot)
        ref_pred, ref_rec, ref_stderr, ref_inputs = bind_run(ref_dir, ref_snapshot, ref_module)
        ref_capture = ref_dir / "operator-capture"
        header, nodes, tets, integrity = ref_module.capture_records(ref_capture)
        ref_env = args.reference_environment
        if ref_env is None:
            ref_env = args.operator_environment
        ref_bound = ref_module.bind_inputs(ref_capture, ref_dir / "prediction.json",
            ref_dir / "receipt.json", ref_stderr, ref_env, source_snapshot_override=ref_snapshot)
        ref_audit = ref_module.validate_and_assemble(header, nodes, tets, integrity, ref_env)
        mass, dt2K, A, _, meta = ref_audit
        ref_manifest = json.loads((ref_dir / "output/manifest.json").read_text())
        ref_world_hex = f"{int(ref_manifest['compiled_world_fingerprint']):016x}"
        pack_equal = sha(compiled_pack) == sha(ref_dir / "output/compiled.nmatterpack")
        same_world = run_world_hex == ref_world_hex == header.get("world_fingerprint_hex")
        same_dims = (env_count == int(header["environment_count"]) and
                     N == int(header["node_count_per_environment"]) and
                     int(header["tetrahedron_count_per_environment"]) == int(ref_manifest["mesh"]["tetrahedra"]))
        same_selector = (int(header["root"]) == root and int(header["newton_iteration"]) == iteration)
        if not (pack_equal and same_world and same_dims and same_selector):
            action_checks = {"available": False,
                "reason": "reference capture binding does not match CGLS compiled world/dimensions/root/Newton",
                "compiled_pack_hash_equal": pack_equal, "world_fingerprint_equal": same_world,
                "dimensions_equal": same_dims, "root_newton_equal": same_selector}
        else:
            env = args.operator_environment
            if env != args.reference_environment and args.reference_environment is not None:
                raise ValueError("operator and reference environments must match")
            if env >= env_count:
                raise ValueError("reference environment outside native snapshot")
            free_nodes = np.asarray(meta["free_nodes"], dtype=np.int64)
            reference_free_nodes = free_nodes
            rows = env * N + free_nodes
            x = arrays["x"][rows, :3].astype(np.float64).reshape(-1)
            r = arrays["r"][rows, :3].astype(np.float64).reshape(-1)
            b = arrays["b"][rows, :3].astype(np.float64).reshape(-1)
            native_ax = arrays["Ax"][rows, :3].astype(np.float64).reshape(-1)
            native_s = arrays["s"][rows, :3].astype(np.float64).reshape(-1)
            ref_rhs = np.asarray(meta["native_rhs_free_impulse_Ns"], dtype=np.float64).reshape(-1)
            if ref_rhs.shape != b.shape:
                raise ValueError("reference RHS and CGLS free vector shapes differ")
            ref_ax = np.asarray(A @ x).reshape(-1)
            ref_atr = np.asarray(A.T @ r).reshape(-1)
            action_checks = {
                "available": True,
                "classification": "same-byte-compiled-world and same root/Newton rich capture; directional action parity only, not whole-matrix identity",
                "reference_run": str(ref_dir), "reference_capture_sha256": sha(ref_capture),
                "reference_source_module_sha256": sha(ref_module_path),
                "reference_prediction_inputs": ref_inputs,
                "reference_binding_source_snapshot": str(ref_snapshot),
                "compiled_pack_sha256": sha(compiled_pack),
                "compiled_pack_hash_equal": pack_equal,
                "world_fingerprint_hex": run_world_hex,
                "root": root, "newton_iteration": iteration, "environment": env,
                "free_node_count": int(meta["free_node_count"]), "operator_shape": list(A.shape), "operator_nnz": int(A.nnz),
                "native_b_vs_reference_center_rhs": compare(b, ref_rhs),
                "native_b_fp32_words_vs_reference_rhs": compare_float32_words(b, ref_rhs),
                "native_Ax_vs_reference_Ax_on_native_x": compare(native_ax, ref_ax),
                "native_s_vs_reference_ATr_on_native_r": compare(native_s, ref_atr),
                "native_recurrence_r_vs_b_minus_native_Ax": compare(r, b - native_ax),
                "reference_linear_residual_l2_from_native_Ax": float(np.linalg.norm(b - ref_ax)),
                "reference_action_residual_l2_from_native_Ax": float(np.linalg.norm(native_ax - ref_ax)),
                "threshold": None,
            }
            action_checks["native_Ax_existing_assembled_1_percent_gate"] = existing_action_gate(
                action_checks["native_Ax_vs_reference_Ax_on_native_x"])
            action_checks["native_s_existing_assembled_1_percent_gate"] = existing_action_gate(
                action_checks["native_s_vs_reference_ATr_on_native_r"])
            reference = {"capture_sha256": sha(ref_capture), "source_snapshot": str(ref_snapshot),
                         "world_fingerprint_hex": ref_world_hex, "binding": ref_bound["capture_run_fingerprints_match"]}

    baseline_result: dict[str, Any] = {"available": False,
        "reason": "no pretarget baseline run supplied"}
    baseline_records: list[dict] = []
    if args.baseline_run_dir is not None:
        baseline_dir = args.baseline_run_dir.resolve()
        baseline_pred_path = baseline_dir / "prediction.json"
        baseline_receipt_path = baseline_dir / "receipt.json"
        baseline_stderr_path = baseline_dir / "stderr.log"
        if not all(p.is_file() for p in (baseline_pred_path, baseline_receipt_path, baseline_stderr_path)):
            raise ValueError(f"baseline run is incomplete: {baseline_dir}")
        baseline_pred = json.loads(baseline_pred_path.read_text())
        baseline_snapshot = Path(baseline_pred.get("source_snapshot", "")).resolve()
        baseline_module, baseline_module_path = load_audit_module(baseline_snapshot)
        baseline_pred, baseline_rec, baseline_stderr_path, baseline_input_checks = bind_run(
            baseline_dir, baseline_snapshot, baseline_module)
        baseline_manifest_path = baseline_dir / "output/manifest.json"
        if not baseline_manifest_path.is_file():
            raise ValueError(f"baseline compiled-world manifest missing: {baseline_manifest_path}")
        baseline_manifest = json.loads(baseline_manifest_path.read_text())
        baseline_world_hex = f"{int(baseline_manifest['compiled_world_fingerprint']):016x}"
        baseline_pack_path = baseline_dir / "output/compiled.nmatterpack"
        baseline_records = parse_markers(baseline_stderr_path, "fem_residual_armijo_iteration=")
        baseline_result = {
            "available": True, "run_dir": str(baseline_dir),
            "source_snapshot": str(baseline_snapshot),
            "source_module_sha256": sha(baseline_module_path),
            "prediction_sha256": sha(baseline_dir / "prediction.json"),
            "receipt_sha256": sha(baseline_dir / "receipt.json"),
            "stderr_sha256": sha(baseline_stderr_path),
            "prediction_inputs": baseline_input_checks,
            "run_exit_code": baseline_rec.get("exit_code"),
            "compiled_world_fingerprint_hex": baseline_world_hex,
            "compiled_pack_sha256": sha(baseline_pack_path) if baseline_pack_path.is_file() else None,
            "same_compiled_world": baseline_world_hex == run_world_hex,
            "same_compiled_pack": bool(baseline_pack_path.is_file() and sha(baseline_pack_path) == sha(compiled_pack)),
        }

    selected = arrays["selected"]
    if np.any(selected > 1) or int(np.count_nonzero(selected == 1)) != 1:
        raise ValueError("snapshot must select exactly one environment")
    env = int(np.flatnonzero(selected == 1)[0])
    if env != args.operator_environment:
        raise ValueError("selected environment differs from requested operator environment")
    direction_environments = dm.get("environments", [])
    if not isinstance(direction_environments, list) or len(direction_environments) != env_count:
        raise ValueError("CGLS direction marker environment vector does not match snapshot E")
    direction_environment = direction_environments[env]
    expected_env = np.concatenate((np.repeat(np.arange(env_count, dtype=np.uint32), N),
                                   np.repeat(np.arange(env_count, dtype=np.uint32), N)))
    if not np.array_equal(arrays["row_environment"], expected_env):
        raise ValueError("row-environment map is not exact E-by-N mechanical/field layout")
    for name in VECTOR_NAMES:
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"CGLS vector {name} contains nonfinite data")
        if np.any(arrays[name][:, 3].view(np.uint32) != 0):
            raise ValueError(f"CGLS vector {name} has nonzero w padding")
    full_mech = env_count * N
    field_rows = np.arange(full_mech, U)
    field_checks = {n: raw_zero(arrays[n][field_rows]) for n in (*VECTOR_NAMES,)}
    if not all(field_checks.values()):
        raise ValueError(f"field backing block is not exact-zero: {field_checks}")
    quiet_checks = {}
    for qenv in range(env_count):
        if qenv == env: continue
        node_rows = np.arange(qenv * N, (qenv + 1) * N)
        rows = np.r_[node_rows, full_mech + node_rows]
        checks = {n: raw_zero(arrays[n][rows]) for n in VECTOR_NAMES}
        checks.update({"errors_zero": bool(arrays["errors"][qenv] == 0),
                       "scalars_zero": raw_zero(arrays["scalars"][qenv]),
                       "control_idle": bool(arrays["control"][qenv, 0] == 0),
                       "selected_zero": bool(arrays["selected"][qenv] == 0)})
        if not all(checks.values()):
            raise ValueError(f"quiet environment {qenv} violates zero/idle contract: {checks}")
        quiet_checks[str(qenv)] = checks

    status, completed, cap, selected_word = (int(v) for v in arrays["control"][env])
    error_bits = int(arrays["errors"][env])
    status_valid = status in (2, 3) and error_bits == 0 and selected_word == 1 and completed <= cap
    mask = arrays["mask"][:full_mech, :3]
    if np.any((mask != 0.0) & (mask != 1.0)):
        raise ValueError("mechanical free mask is outside {0,1}")
    if np.any(arrays["mask"][full_mech:, :3] != 0.0):
        raise ValueError("field backing mask is nonzero")
    reference_mask_check = {"available": False,
        "reason": "no same-world/root/Newton rich operator capture supplied"}
    if reference_free_nodes is not None:
        expected_mask = np.zeros((N, 3), dtype=np.float32)
        expected_mask[reference_free_nodes, :] = 1.0
        selected_node_mask = arrays["mask"][env * N:(env + 1) * N, :3]
        reference_mask_check = {
            "available": True,
            "classification": "exact selected-environment free-node mask comparison against the bound rich operator capture",
            "free_node_count": int(reference_free_nodes.size),
            "native_free_node_count": int(np.count_nonzero(selected_node_mask[:, 0])),
            "exact_match": bool(np.array_equal(selected_node_mask, expected_mask)),
            "mismatch_node_count": int(np.count_nonzero(np.any(selected_node_mask != expected_mask, axis=1))),
        }

    armijo_records = parse_markers(stderr_path, "fem_residual_armijo_iteration=")
    armijo_result = summarize_armijo(armijo_records, env, root, iteration)
    pretarget = summarize_pretarget_trace(armijo_records, baseline_records, root, iteration) if baseline_result["available"] else {
        "exact_pretarget_trace_parity": False, "classification": "not assessed; baseline run absent"}
    if baseline_result["available"]:
        pretarget["same_compiled_world"] = baseline_result["same_compiled_world"]
        pretarget["same_compiled_pack"] = baseline_result["same_compiled_pack"]
        pretarget["pretarget_trace_parity_with_compiled_identity"] = bool(
            pretarget["exact_pretarget_trace_parity"] and
            baseline_result["same_compiled_world"] and baseline_result["same_compiled_pack"])
    coupled_capture_scope = summarize_coupled_capture_scope(stderr_path, root, iteration)
    report = {
        "schema": "numi.cardboard.native_cgls_snapshot_audit.v2",
        "classification": "read-only snapshot/layout/status audit with optional same-world directional operator replay; not a physical solve or solver certificate",
        "run_binding": {"run_dir": str(run_dir), "source_snapshot": str(source_snapshot),
            "prediction_sha256": sha(run_dir / "prediction.json"), "receipt_sha256": sha(run_dir / "receipt.json"),
            "stderr_sha256": sha(stderr_path), "source_module_sha256": sha(source_module_path),
            "prediction_inputs": input_checks, "run_exit_code": rec.get("exit_code"), "timed_out": rec.get("timed_out"),
            "CGLS_root": root, "CGLS_newton_iteration": iteration,
            "binary_sha256": snapshot_header["sha256"], "binary_bytes": snapshot_header["file_bytes"]},
        "snapshot_header": snapshot_header,
        "native_state": {"selected_environment": env, "status": status,
            "status_name": {2:"exhausted",3:"zero_normal_residual"}.get(status,"invalid_or_breakdown"),
            "completed_iterations": completed, "cap": cap, "error_bits": error_bits,
            "selected_word": selected_word, "status_valid": bool(status_valid),
            "direction_marker_selected_environment": direction_environment,
            "scalars_gamma_alpha_beta_bnorm2": arrays["scalars"][env].astype(float).tolist(),
            "x_l2": float(np.linalg.norm(arrays["x"][env*N:(env+1)*N,:3].astype(np.float64))),
            "r_l2": float(np.linalg.norm(arrays["r"][env*N:(env+1)*N,:3].astype(np.float64))),
            "s_l2": float(np.linalg.norm(arrays["s"][env*N:(env+1)*N,:3].astype(np.float64))),
            "p_l2": float(np.linalg.norm(arrays["p"][env*N:(env+1)*N,:3].astype(np.float64))),
            "Ax_l2": float(np.linalg.norm(arrays["Ax"][env*N:(env+1)*N,:3].astype(np.float64))),
            "b_l2": float(np.linalg.norm(arrays["b"][env*N:(env+1)*N,:3].astype(np.float64)))},
        "scope_checks": {"field_backing_exact_zero": field_checks,
            "quiet_environment_exact_zero_idle": quiet_checks,
            "selected_mask_free_row_count": int(np.count_nonzero(
                arrays["mask"][env*N:(env+1)*N, 0])),
            "selected_mask_vs_reference": reference_mask_check,
            "original_newton_certificate_applies_to_CGLS_direction": False},
        "armijo_trial": armijo_result,
        "pretarget_history_comparison": {"baseline": baseline_result, **pretarget},
        "target_rich_operator_capture_scope": coupled_capture_scope,
        "reference_operator": reference,
        "action_comparison": action_checks,
        "limitations": [
            "Any coupled-capture skip is reported from this run's own root/Newton skip marker; an absent marker is not treated as a completed capture.",
            "If supplied, 419 CSR replay is checked only on native x and r, with same compiled world, pack, root, Newton, E, and N; this is directional action parity, not a proof of full matrix equality.",
            "All action differences are report-only and have no qualified noise-floor or acceptance threshold.",
            "The CGLS direction does not inherit the original Newton/FGMRES certificate and this audit does not accept or qualify a physical transaction."
        ]}
    report["analyzer_binding"] = {
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha(Path(__file__).resolve()),
        "argv": list(sys.argv),
        "python_version": sys.version,
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--source-snapshot", type=Path, required=True)
    ap.add_argument("--operator-environment", type=int, required=True)
    ap.add_argument("--reference-run-dir", type=Path,
                    help="optional source-bound 419 run containing operator-capture JSONL")
    ap.add_argument("--reference-environment", type=int)
    ap.add_argument("--baseline-run-dir", type=Path,
                    help="source-bound feature-off run used for exact residual-Armijo record parity before the selected Newton")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists(): raise SystemExit(f"refusing to overwrite existing output {out}")
    try:
        report = run(args)
    except Exception as exc:
        out.mkdir(parents=True, exist_ok=False)
        (out / "failure.json").write_text(json.dumps({"schema":"numi.cardboard.native_cgls_snapshot_audit_failure.v2","error":str(exc)},indent=2)+"\n")
        raise
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    manifest = {"files": {p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in out.iterdir()}}
    (out / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"result":"AUDIT_WRITTEN","output":str(out),
        "status_valid":report["native_state"]["status_valid"],
        "selected_environment":report["native_state"]["selected_environment"],
        "root":report["run_binding"]["CGLS_root"],
        "newton_iteration":report["run_binding"]["CGLS_newton_iteration"],
        "reference_operator_available":report["action_comparison"]["available"]},indent=2))
    return 0

if __name__ == "__main__": raise SystemExit(main())
