#!/usr/bin/env python3
"""Independent sparse FP64 replay of native497's captured cached FEM operator.

The matrix is rebuilt from the exact captured FP32 production tangent cache and
geometry. This is an operator replay, not a constitutive-reference solve, a
native FP64 execution, or physical/solver qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

BASE = Path("/Users/home/cardboard-implementation-20261014")
DEFAULT_CASE = BASE / "solver/macmini-warm-cgls-control20-diagnostic-replay-497"
DEFAULT_SOURCE = BASE / "snapshots/source-admission-encoder-182-003"
DEFAULT_ARTIFACTS = BASE / "analysis/source182-host-artifacts-003"
DEFAULT_STATE_READER = BASE / "analysis/base-cgls-state-reader-abi-repair-148-001/tools/read_state_capture.py"
MAGIC = 0x534C4743


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            value[key] = item
        return value

    def reject(token):
        raise ValueError(f"nonstandard JSON token {token} in {path}")

    result = json.loads(path.read_text(encoding="utf-8"),
                        object_pairs_hook=pairs, parse_constant=reject)
    if not isinstance(result, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return result


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load source-bound module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _align16(value: int) -> int:
    return (value + 15) & ~15


def parse_cgls_capture(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Parse exact source182 CGLS capture versions 1..3 with checked extents."""
    raw = path.read_bytes()
    if len(raw) < 32:
        raise ValueError("CGLS capture is shorter than its 8-word header")
    magic, version, envs, nodes, unknowns, cap, nsegments, reserved = struct.unpack_from("<8I", raw)
    if magic != MAGIC or version not in (1, 2, 3) or reserved != 0:
        raise ValueError(f"unsupported CGLS header {(magic, version, reserved)}")
    if envs < 1 or envs > 2 or nodes < 1 or unknowns != 2 * envs * nodes:
        raise ValueError("CGLS shape does not satisfy admitted U=2*E*N layout")
    if cap not in (32, 64):
        raise ValueError("unexpected CGLS iteration cap")
    if nsegments != (13 if version in (2, 3) else 12):
        raise ValueError("CGLS segment count disagrees with version")
    if version == 3 and nsegments != 13:
        raise ValueError("warm-start CGLS v3 must carry the right-scale rows")

    specs: list[tuple[str, str, int, tuple[int, ...]]] = []
    vector_names = ("x", "r", "s", "p", "q", "b", "mask")
    for name in vector_names:
        specs.append((name, "<f4", unknowns * 4, (unknowns, 4)))
    specs.extend([
        ("row_environment", "<u4", unknowns, (unknowns,)),
        ("scalars", "<f4", envs * 4, (envs, 4)),
        ("control", "<u4", envs * 4, (envs, 4)),
        ("errors", "<u4", envs, (envs,)),
        ("selected", "<u4", envs, (envs,)),
    ])
    if nsegments == 13:
        specs.append(("right_scale_rows", "<f4", envs * nodes * 3 * 4,
                      (envs, nodes, 3, 4)))

    arrays: dict[str, np.ndarray] = {}
    segments = []
    cursor = 32
    for name, dtype, count, shape in specs:
        start = _align16(cursor)
        if start > len(raw) or any(raw[cursor:start]):
            raise ValueError(f"invalid/truncated alignment padding before {name}")
        size = count * 4
        end = start + size
        if end > len(raw):
            raise ValueError(f"truncated CGLS segment {name}")
        arrays[name] = np.frombuffer(raw, dtype=np.dtype(dtype), count=count,
                                     offset=start).reshape(shape).copy()
        segments.append({"name": name, "offset": start, "byte_count": size,
                         "dtype": dtype, "shape": list(shape)})
        cursor = end
    if cursor != len(raw):
        raise ValueError(f"unexpected CGLS trailing bytes: expected {cursor}, got {len(raw)}")
    for name, values in arrays.items():
        if values.dtype.kind == "f" and not np.isfinite(values).all():
            raise ValueError(f"nonfinite values in CGLS segment {name}")
    return ({"magic_hex": f"0x{magic:08x}", "version": version,
             "environment_count": envs, "node_count_per_environment": nodes,
             "unknown_count": unknowns, "iteration_cap": cap,
             "segment_count": nsegments, "file_bytes": len(raw),
             "sha256": sha256(path), "segments": segments}, arrays)


def strict_jsonl(path: Path) -> list[dict[str, Any]]:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line, object_pairs_hook=pairs,
                                   parse_constant=lambda token: (_ for _ in ()).throw(
                                       ValueError(f"nonstandard JSON token {token}")))
            except Exception as error:
                raise ValueError(f"invalid strict JSONL line {line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            records.append(value)
    return records


def pack_f32(values: Any) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes(order="C")


def pack_u32(values: Any) -> bytes:
    return np.asarray(values, dtype="<u4").tobytes(order="C")


def canonical_zero_words(raw: bytes, float_word_positions: set[int]) -> bytes:
    words = list(struct.unpack("<" + "I" * (len(raw) // 4), raw))
    for index in float_word_positions:
        if (words[index] & 0x7FFFFFFF) == 0:
            words[index] = 0
    return struct.pack("<" + "I" * len(words), *words)


def compare_native_record(actual: bytes, expected: bytes,
                          float_positions: set[int]) -> str:
    if actual == expected:
        return "exact"
    if len(actual) == len(expected) and canonical_zero_words(actual, float_positions) == \
            canonical_zero_words(expected, float_positions):
        return "signed_zero_only"
    return "different"


def compare_state_to_operator_capture(state_doc: dict[str, Any], state_payload: bytes,
                                      records: list[dict[str, Any]]) -> dict[str, Any]:
    desc = {x["name"]: x for x in state_doc["arrays"]}

    def raw(name: str) -> bytes:
        item = desc[name]
        return state_payload[item["offset"]:item["offset"] + item["byte_count"]]

    e_count, n_count, t_count, stride = (state_doc[k] for k in
        ("environment_count", "node_count", "tetrahedron_count", "material_state_stride"))
    tallies = {
        "accepted_nodes_exact": 0, "candidate_nodes_exact": 0,
        "accepted_nodes_signedzero_only": 0, "candidate_nodes_signedzero_only": 0,
        "accepted_tetrahedra_exact": 0, "candidate_tetrahedra_exact": 0,
        "accepted_tetrahedra_signedzero_only": 0, "candidate_tetrahedra_signedzero_only": 0,
        "accepted_history_exact": 0, "cache_valid_exact": 0, "cache_values_exact": 0,
        "newton_correction_xyz_exact": 0, "base_rhs_float4_exact": 0,
    }
    mismatches: dict[str, list] = {}
    seen_nodes: set[tuple[int, int]] = set()
    seen_tets: set[tuple[int, int]] = set()
    node_float_positions = set(range(20))
    # Tetrahedron ABI: nodes uint4, inverseRest float4 x3, identity uint4, frame float4.
    tet_float_positions = set(range(4, 16)) | set(range(20, 24))
    history = np.frombuffer(raw("accepted_material_history"), dtype="<f4").reshape(
        e_count, t_count, stride)
    correction = np.frombuffer(raw("original_newton_correction"), dtype="<f4").reshape(
        2 * e_count * n_count, 4)
    rhs = np.frombuffer(raw("base_residual_rhs"), dtype="<f4").reshape(
        2 * e_count * n_count, 4)
    cache = np.frombuffer(raw("algorithmic_tangent_cache"), dtype="<f4").reshape(
        e_count, t_count, 9, 9)
    cache_valid = np.frombuffer(raw("algorithmic_tangent_cache_valid"), dtype="<u4").reshape(
        e_count, t_count)

    counts = {"node": 0, "tetrahedron": 0}
    for rec in records[1:]:
        kind = rec.get("record")
        if kind == "node":
            e, node = int(rec["environment"]), int(rec["node_id"])
            key = (e, node)
            if key in seen_nodes or not (0 <= e < e_count and 0 <= node < n_count):
                raise ValueError(f"duplicate/out-of-range operator node {key}")
            seen_nodes.add(key); counts["node"] += 1
            expected_accepted = pack_f32(rec["accepted"]["raw_state_float4s"])
            expected_candidate = pack_f32(rec["center"]["raw_state_float4s"])
            idx = e * n_count + node
            aoff = desc["accepted_nodes"]["offset"] + idx * 80
            coff = desc["candidate_nodes"]["offset"] + idx * 80
            got_a = state_payload[aoff:aoff + 80]
            got_c = state_payload[coff:coff + 80]
            cmp_a = compare_native_record(got_a, expected_accepted, node_float_positions)
            cmp_c = compare_native_record(got_c, expected_candidate, node_float_positions)
            tallies[f"accepted_nodes_{'exact' if cmp_a == 'exact' else 'signedzero_only' if cmp_a == 'signed_zero_only' else 'different'}"] += 1
            tallies[f"candidate_nodes_{'exact' if cmp_c == 'exact' else 'signedzero_only' if cmp_c == 'signed_zero_only' else 'different'}"] += 1
            if cmp_a == "different": mismatches.setdefault("accepted_nodes", []).append([e, node])
            if cmp_c == "different": mismatches.setdefault("candidate_nodes", []).append([e, node])
            actual_corr = correction[idx]
            expected_corr = np.asarray([*rec["correction_delta_v_m_s"], 0.0], dtype="<f4")
            if actual_corr.tobytes() == expected_corr.tobytes():
                tallies["newton_correction_xyz_exact"] += 1
            else: mismatches.setdefault("newton_correction_xyz", []).append([e, node])
            expected_rhs = np.asarray(rec["center_residual_impulse_Ns"], dtype="<f4")
            if rhs[idx].tobytes() == expected_rhs.tobytes():
                tallies["base_rhs_float4_exact"] += 1
            else: mismatches.setdefault("base_rhs_float4", []).append([e, node])
        elif kind == "tetrahedron":
            e, tet = int(rec["environment"]), int(rec["tet_id"])
            key = (e, tet)
            if key in seen_tets or not (0 <= e < e_count and 0 <= tet < t_count):
                raise ValueError(f"duplicate/out-of-range operator tetrahedron {key}")
            seen_tets.add(key); counts["tetrahedron"] += 1
            inv = rec["inverse_rest_rows_per_m"]
            row_words = [*inv[0], rec["rest_volume_m3"], *inv[1], 0.0, *inv[2], 0.0]
            ident = [rec["material_index"], rec["object_index"],
                     rec["topology_generation"], rec["tet_flags"]]
            expected_tet = (pack_u32(rec["nodes"]) + pack_f32(row_words) +
                            pack_u32(ident) + pack_f32(rec["material_frame_quaternion_xyzw"]))
            idx = e * t_count + tet
            aoff = desc["accepted_tetrahedra"]["offset"] + idx * 96
            coff = desc["candidate_tetrahedra"]["offset"] + idx * 96
            got_a, got_c = state_payload[aoff:aoff + 96], state_payload[coff:coff + 96]
            cmp_a = compare_native_record(got_a, expected_tet, tet_float_positions)
            cmp_c = compare_native_record(got_c, expected_tet, tet_float_positions)
            tallies[f"accepted_tetrahedra_{'exact' if cmp_a == 'exact' else 'signedzero_only' if cmp_a == 'signed_zero_only' else 'different'}"] += 1
            tallies[f"candidate_tetrahedra_{'exact' if cmp_c == 'exact' else 'signedzero_only' if cmp_c == 'signed_zero_only' else 'different'}"] += 1
            if cmp_a == "different": mismatches.setdefault("accepted_tetrahedra", []).append([e, tet])
            if cmp_c == "different": mismatches.setdefault("candidate_tetrahedra", []).append([e, tet])
            hist_values = rec["center"]["accepted_state"]
            if len(hist_values) > stride: raise ValueError("accepted material history exceeds state stride")
            expected_hist = np.asarray(hist_values + [0.0] * (stride - len(hist_values)), dtype="<f4")
            if history[e, tet].tobytes() == expected_hist.tobytes():
                tallies["accepted_history_exact"] += 1
            else: mismatches.setdefault("accepted_history", []).append([e, tet])
            cache_row = rec["cached_algorithmic_tangent_local"]
            want_valid = 1 if cache_row.get("valid") is True else 0
            if int(cache_valid[e, tet]) == want_valid:
                tallies["cache_valid_exact"] += 1
            else: mismatches.setdefault("cache_valid", []).append([e, tet])
            if want_valid:
                expected_cache = np.asarray(cache_row[
                    "matrix_row_major_local_dP_dF_Pa_per_dimensionless_F"], dtype="<f4").reshape(9, 9)
                if cache[e, tet].tobytes() == expected_cache.tobytes():
                    tallies["cache_values_exact"] += 1
                else: mismatches.setdefault("cache_values", []).append([e, tet])
        elif kind == "integrity":
            continue
        else:
            raise ValueError(f"unexpected operator capture record type {kind!r}")
    if counts != {"node": e_count * n_count, "tetrahedron": e_count * t_count}:
        raise ValueError(f"operator capture record counts incomplete: {counts}")
    return {"counts": counts, "expected_counts": {"node": e_count*n_count,
            "tetrahedron": e_count*t_count}, "exact_word_tallies": tallies,
            "mismatches": {k: v[:20] for k, v in mismatches.items()},
            "all_geometry_cache_rhs_and_history_joins_exact_or_signedzero_only": not mismatches}


def resolve_prediction_inputs(prediction: dict[str, Any], source: Path,
                             artifacts: Path, case: Path) -> dict[str, Any]:
    """Verify prediction inputs in the mirrored B tree without rewriting it."""
    external = prediction.get("external_inputs")
    if not isinstance(external, list): raise ValueError("prediction external input map missing")
    by_base: dict[str, list[str]] = {}
    for remote in external:
        if isinstance(remote, str): by_base.setdefault(Path(remote).name, []).append(remote)

    def mirrored_external(remote: str) -> tuple[Path | None, str | None]:
        anchors = ("/Users/n/cardboard-implementation-20261014-host-macmini/",
                   "/Users/n/cardboard-implementation-20261014/")
        for anchor in anchors:
            if remote.startswith(anchor):
                rel = remote[len(anchor):]
                if rel.startswith("build-source182-003/"):
                    return artifacts / Path(rel).relative_to("build-source182-003"), None
                candidate = BASE / rel
                if candidate.is_file():
                    return candidate, None
                # The checkpoint case was mirrored locally with an explicit
                # host qualifier in its directory name. This is the one
                # source-declared path alias; do not generalize by basename.
                remote_prefix = "solver/warm-cgls-control19-checkpoint-480/"
                if rel.startswith(remote_prefix):
                    local_rel = "solver/macmini-warm-cgls-control19-checkpoint-480/" + \
                        rel[len(remote_prefix):]
                    alias = BASE / local_rel
                    if alias.is_file():
                        return alias, "remote_solver_checkpoint_to_local_macmini_qualified_case"
                return candidate, None
        return None, None

    checked, missing, mismatched, aliases, verified = 0, [], [], [], []
    classes = {"source": 0, "runtime": 0, "external": 0}
    for key, expected in prediction.get("inputs_sha256", {}).items():
        candidate = None
        if key.startswith("source/"):
            candidate = source / key[len("source/"):]
            classes["source"] += 1
        elif key.startswith("runtime/"):
            candidate = artifacts / key[len("runtime/"):]
            classes["runtime"] += 1
        elif key.startswith("external/"):
            remotes = by_base.get(Path(key).name, [])
            if len(remotes) != 1:
                missing.append({"key": key, "reason": "external basename is not unique",
                                "candidates": len(remotes)})
                continue
            candidate, alias = mirrored_external(remotes[0])
            classes["external"] += 1
            if alias is not None:
                aliases.append({"key": key, "remote": remotes[0],
                                "local": str(candidate), "rule": alias})
        else:
            missing.append({"key": key, "reason": "unrecognized prediction input key"})
            continue
        if candidate is None or not candidate.is_file():
            missing.append({"key": key, "candidate": str(candidate) if candidate else None})
        else:
            actual = sha256(candidate)
            if actual != expected:
                mismatched.append({"key": key, "path": str(candidate),
                                   "expected_sha256": expected, "actual_sha256": actual})
            else:
                checked += 1
                verified.append({"key": key, "path": str(candidate), "sha256": actual})
    if mismatched:
        raise ValueError(f"prediction input hash mismatch: {mismatched[:3]}")
    return {"declared_count": len(prediction.get("inputs_sha256", {})),
            "verified_count": checked, "missing": missing[:20], "missing_count": len(missing),
            "verified_by_class": classes, "explicit_local_path_aliases": aliases,
            "verified_inputs": verified,
            "all_declared_prediction_inputs_verified": checked == len(prediction.get("inputs_sha256", {}))}


def compare_vector(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    a = np.asarray(actual, dtype=np.float64).reshape(-1)
    b = np.asarray(expected, dtype=np.float64).reshape(-1)
    d = a - b
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    diff = float(np.linalg.norm(d))
    return {"rows": int(a.size), "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
            "actual_l2": norm_a, "reconstructed_l2": norm_b,
            "difference_l2": diff,
            "relative_to_reconstructed": diff / norm_b if norm_b > 0 else None,
            "max_abs_difference": float(np.max(np.abs(d))) if d.size else 0.0}


def norm_or_zero(vector: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float64)))


def warm_residual_decomposition(A, b: np.ndarray, x0: np.ndarray,
                                total_candidate: np.ndarray) -> dict[str, np.ndarray]:
    """Apply v3 semantics: X is total x0+Sy, B starts at b-A*x0."""
    b64 = np.asarray(b, dtype=np.float64).reshape(-1)
    x064 = np.asarray(x0, dtype=np.float64).reshape(-1)
    x64 = np.asarray(total_candidate, dtype=np.float64).reshape(-1)
    if b64.shape != x064.shape or x064.shape != x64.shape or A.shape != (b64.size, b64.size):
        raise ValueError("warm residual vectors and operator have inconsistent extents")
    delta = x64 - x064
    initial = b64 - np.asarray(A @ x064).reshape(-1)
    final = b64 - np.asarray(A @ x64).reshape(-1)
    predicted = initial - np.asarray(A @ delta).reshape(-1)
    return {"delta": delta, "initial_residual": initial,
            "final_residual": final, "residual_from_delta": predicted,
            "identity_error": final - predicted}


def residual_evidence(operator_module, state_doc: dict[str, Any], payload: bytes,
                      cgls_header: dict[str, Any], cgls: dict[str, np.ndarray],
                      fd_records: list[dict[str, Any]], environment: int,
                      direction_trace: dict[str, Any]) -> dict[str, Any]:
    header = fd_records[0]
    nodes = [r for r in fd_records if r.get("record") == "node"]
    tets = [r for r in fd_records if r.get("record") == "tetrahedron"]
    integrity_rows = [r for r in fd_records if r.get("record") == "integrity"]
    if len(integrity_rows) != 1: raise ValueError("FD capture must contain one integrity record")
    mass, dt2_stiffness, A, fd_direction, meta = operator_module.validate_and_assemble(
        header, nodes, tets, integrity_rows[0], environment)
    E, N, U = (state_doc[k] for k in ("environment_count", "node_count", "unknown_count"))
    if cgls_header["environment_count"] != E or cgls_header["node_count_per_environment"] != N or cgls_header["unknown_count"] != U:
        raise ValueError("CGLS and sidecar dimensions differ")
    free_nodes = meta["free_nodes"]
    row_start = environment * N
    field_base = E * N
    vec_arrays = cgls
    state_arrays = {a["name"]: a for a in state_doc["arrays"]}

    def state_view(name: str, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
        desc = state_arrays[name]
        block = payload[desc["offset"]:desc["offset"] + desc["byte_count"]]
        return np.frombuffer(block, dtype=np.dtype(dtype)).reshape(shape).copy()

    x0_full = state_view("original_newton_correction", "<f4", (U, 4))
    original_b_full = state_view("base_residual_rhs", "<f4", (U, 4))
    x0 = x0_full[row_start + free_nodes, :3].astype(np.float64).reshape(-1)
    b = original_b_full[row_start + free_nodes, :3].astype(np.float64).reshape(-1)
    X_all = vec_arrays["x"]
    X = X_all[row_start + free_nodes, :3].astype(np.float64).reshape(-1)
    decomposition = warm_residual_decomposition(A, b, x0, X)
    delta = decomposition["delta"]
    native_B = vec_arrays["b"][row_start + free_nodes, :3].astype(np.float64).reshape(-1)
    native_R = vec_arrays["r"][row_start + free_nodes, :3].astype(np.float64).reshape(-1)
    native_Q = vec_arrays["q"][row_start + free_nodes, :3].astype(np.float64).reshape(-1)

    Mx0 = np.asarray(mass @ x0).reshape(-1)
    Kx0 = np.asarray(dt2_stiffness @ x0).reshape(-1)
    AX0 = Mx0 + Kx0
    r0 = decomposition["initial_residual"]
    Md = np.asarray(mass @ delta).reshape(-1)
    Kd = np.asarray(dt2_stiffness @ delta).reshape(-1)
    Ad = Md + Kd
    MX = np.asarray(mass @ X).reshape(-1)
    KX = np.asarray(dt2_stiffness @ X).reshape(-1)
    AX = MX + KX
    r_final = decomposition["final_residual"]
    r_from_refined_residual = decomposition["residual_from_delta"]
    direct_identity_error = decomposition["identity_error"]
    finite_resid = np.isfinite(np.concatenate((r0, r_final, r_from_refined_residual))).all()
    if not finite_resid: raise ValueError("FP64 sparse replay produced a nonfinite residual")

    trace_env = next((e for e in direction_trace.get("environments", [])
                      if int(e.get("environment", -1)) == environment), None)
    if trace_env is None: raise ValueError("native CGLS trace lacks selected environment")
    cgls_control = vec_arrays["control"][environment].astype(np.uint32).tolist()
    selected_word = int(vec_arrays["selected"][environment])
    if selected_word not in (0, 1):
        raise ValueError(f"invalid CGLS selected word for environment {environment}: {selected_word}")
    if selected_word == 0:
        # The v3 arrays are private per-environment scratch. When selection is
        # zero, their zeroed B/R/X/Q values are not an installed candidate and
        # must not be compared with the nonzero Newton-center residual.
        scratch_norms = {
            name: norm_or_zero(vec_arrays[name][row_start + free_nodes, :3].astype(np.float64).reshape(-1))
            for name in ("x", "r", "q", "b")
        }
        base_action = np.asarray(A @ x0).reshape(-1)
        base_residual = b - base_action
        return {
            "source_semantics": {
                "operator": "A = M + dt^2 K reconstructed in FP64 from captured FP32 production dP/dF cache and candidate geometry",
                "state_x0": "original Newton correction captured before CGLS; not a constitutive reference solution",
                "base_residual": "original captured RHS minus reconstructed cached-operator action at x0",
                "CGLS_v3_payload": "unselected per-environment scratch only; no CGLS candidate or residual is installed for this environment",
                "scope": "mechanical free FEM rows only; no rigid/contact/cohesive block is reconstructed",
            },
            "operator": {"dimension": int(A.shape[0]), "nnz": int(A.nnz),
                         "free_nodes": int(meta["free_node_count"]),
                         "fixed_nodes": int(meta["fixed_node_count"]),
                         "dt_seconds": float(meta["dt_seconds"]),
                         "cached_tangent_valid_count": int(state_doc["tangent_cache"]["valid_entry_count"] // E),
                         "relative_skew_norm": float(meta["relative_skew_norm"]),
                         "native_original_newton_action_parity": meta["action_parity"],
                         "mass_diagonal_min_max": [float(mass.diagonal().min()), float(mass.diagonal().max())],
                         "source_fp32_cache_is_not_recomputed": True},
            "CGLS_state": {"selected_word": 0,
                           "control_u32_status_iterations_cap_selected": cgls_control,
                           "errors": int(vec_arrays["errors"][environment]),
                           "native_direction_trace": trace_env,
                           "direction_installed": False},
            "vector_norms": {"original_b_l2": norm_or_zero(b), "x0_l2": norm_or_zero(x0),
                             "base_residual_b_minus_Ax0_l2": norm_or_zero(base_residual),
                             "unselected_private_CGLS_scratch_l2": scratch_norms},
            "native_vector_comparisons": {"applicable": False,
                "reason": "selected word is zero; private CGLS scratch is not an installed direction",
                "unselected_private_scratch_l2": scratch_norms},
            "mass_stiffness_separation": {"base_x0": {
                "M_x0_l2": norm_or_zero(Mx0), "dt2_K_x0_l2": norm_or_zero(Kx0),
                "A_x0_l2": norm_or_zero(base_action), "base_residual_l2": norm_or_zero(base_residual),
                "b_minus_Ax0_cancellation_fraction": norm_or_zero(base_residual) /
                    (norm_or_zero(b) + norm_or_zero(base_action))
                    if norm_or_zero(b) + norm_or_zero(base_action) > 0 else None}},
            "algebraic_consistency": {"CGLS_checks_applicable": False,
                "reason": "environment was not selected; no CGLS update was installed"},
            "limits": [
                "This environment had no installed CGLS direction; its zero scratch arrays are not a residual-reduction result.",
                "The replay matrix is derived from the captured FP32 production tangent cache; no constitutive law is reevaluated.",
                "This is an offline operator reconstruction, not native FP64 execution or whole-coupon qualification.",
            ],
        }
    residual_stats = {
        "source_semantics": {
            "operator": "A = M + dt^2 K reconstructed in FP64 from captured FP32 production dP/dF cache and captured candidate geometry",
            "state_x0": "original Newton correction captured before CGLS; not a constitutive reference solution",
            "CGLS_v3_X": "total captured candidate x0+S*y per source182 writer semantics",
            "CGLS_v3_B": "fresh residual b-A*x0 in CGLS scratch; compared against FP64 reconstruction",
            "CGLS_v3_R": "CGLS recurrence residual; compared against fresh FP64 b-A*X and recurrence identity",
            "scope": "mechanical free FEM rows only; field backing and constrained rows must be zero; no rigid/contact/cohesive unknown block is reconstructed here",
        },
        "operator": {"dimension": int(A.shape[0]), "nnz": int(A.nnz),
                     "free_nodes": int(meta["free_node_count"]),
                     "fixed_nodes": int(meta["fixed_node_count"]),
                     "dt_seconds": float(meta["dt_seconds"]),
                     "cached_tangent_valid_count": int(state_doc["tangent_cache"]["valid_entry_count"] // E),
                     "relative_skew_norm": float(meta["relative_skew_norm"]),
                     "native_original_newton_action_parity": meta["action_parity"],
                     "mass_diagonal_min_max": [float(mass.diagonal().min()), float(mass.diagonal().max())],
                     "source_fp32_cache_is_not_recomputed": True},
        "CGLS_state": {"selected_word": selected_word,
                       "control_u32_status_iterations_cap_selected": cgls_control,
                       "errors": int(vec_arrays["errors"][environment]),
                       "native_direction_trace": trace_env,
                       "native_q_is_AX_after_total_candidate_by_source_order": True},
        "vector_norms": {
            "original_b_l2": norm_or_zero(b),
            "x0_l2": norm_or_zero(x0),
            "candidate_total_X_l2": norm_or_zero(X),
            "delta_X_minus_x0_l2": norm_or_zero(delta),
            "r0_fp64_l2": norm_or_zero(r0),
            "r_final_fp64_l2": norm_or_zero(r_final),
            "residual_reduction_ratio_fp64": norm_or_zero(r_final) / norm_or_zero(r0) if norm_or_zero(r0) > 0 else None,
        },
        "native_vector_comparisons": {
            "captured_B_vs_FP64_b_minus_Ax0": compare_vector(native_B, r0),
            "captured_R_vs_FP64_b_minus_A_total_X": compare_vector(native_R, r_final),
            "captured_Q_vs_FP64_A_total_X": compare_vector(native_Q, AX),
            "captured_R_vs_FP64_r0_minus_A_delta": compare_vector(native_R, r_from_refined_residual),
        },
        "mass_stiffness_separation": {
            "x0": {"M_x0_l2": norm_or_zero(Mx0), "dt2_K_x0_l2": norm_or_zero(Kx0),
                   "A_x0_l2": norm_or_zero(AX0), "r0_l2": norm_or_zero(r0),
                   "b_minus_Ax0_cancellation_fraction": norm_or_zero(r0) /
                     (norm_or_zero(b) + norm_or_zero(AX0)) if norm_or_zero(b) + norm_or_zero(AX0) > 0 else None},
            "delta": {"M_delta_l2": norm_or_zero(Md), "dt2_K_delta_l2": norm_or_zero(Kd),
                      "A_delta_l2": norm_or_zero(Ad)},
            "total_X": {"M_X_l2": norm_or_zero(MX), "dt2_K_X_l2": norm_or_zero(KX),
                        "A_X_l2": norm_or_zero(AX), "r_final_l2": norm_or_zero(r_final),
                        "b_minus_AX_cancellation_fraction": norm_or_zero(r_final) /
                          (norm_or_zero(b) + norm_or_zero(AX)) if norm_or_zero(b) + norm_or_zero(AX) > 0 else None},
            "interpretation_limit": "small residual relative to b and A*x indicates subtraction cancellation in this captured operator action; it is not by itself a condition-number estimate",
        },
        "algebraic_consistency": {
            "r_final_equals_r0_minus_A_delta_fp64_max_abs": float(np.max(np.abs(direct_identity_error))) if direct_identity_error.size else 0.0,
            "r_final_equals_r0_minus_A_delta_fp64_l2": norm_or_zero(direct_identity_error),
            "linear_residual_trace_l2": trace_env.get("linear_residual_l2"),
            "trace_vs_fp64_final_residual_relative": (
                abs(float(trace_env["linear_residual_l2"]) - norm_or_zero(r_final)) /
                norm_or_zero(r_final) if norm_or_zero(r_final) > 0 else None),
        },
        "limits": [
            "The replay matrix is derived from the captured FP32 production tangent cache; no constitutive law is reevaluated.",
            "FP64 sparse arithmetic is an offline reconstruction, not a native FP64 operator execution.",
            "A small residual/cancellation fraction does not establish poor conditioning; no singular-value estimate is included unless separately requested.",
            "This audit concerns one selected Newton and the admitted FEM-only CGLS block, not the full contact/coupled physical operator or the whole failed coupon.",
        ],
    }
    return residual_stats


def run(args: argparse.Namespace) -> dict[str, Any]:
    case = args.case.resolve(strict=True)
    source = args.source_snapshot.resolve(strict=True)
    artifacts = args.source_artifacts.resolve(strict=True)
    state_manifest = case / "diagnostics/capture-base-cgls-capture/capture.state.json"
    cgls_binary = case / "diagnostics/capture-base-cgls-capture/capture"
    fd_capture = case / "diagnostics/capture-operator-fd-capture/capture"
    prediction_path, receipt_path = case / "prediction.json", case / "receipt.json"
    stderr_path = case / "stderr.log"
    for path in (state_manifest, cgls_binary, fd_capture, prediction_path, receipt_path, stderr_path):
        if not path.is_file(): raise ValueError(f"required native497 input missing: {path}")

    prediction, receipt = strict_json(prediction_path), strict_json(receipt_path)
    state_reader = load_module("native497_state_reader", args.state_reader.resolve(strict=True))
    reader_validation = state_reader.validate_capture(state_manifest)
    state_doc = strict_json(state_manifest)
    state_payload_path = state_manifest.parent / state_doc["payload"]["file"]
    state_payload = state_payload_path.read_bytes()
    if receipt.get("source_inputs_unchanged") is not True or receipt.get("timed_out") is not False:
        raise ValueError("native497 receipt is not a clean, unchanged-input terminal capture")
    if receipt.get("exit_code") != 2:
        raise ValueError("native497 failure run exit differs from retained expected failure receipt")
    input_check = resolve_prediction_inputs(prediction, source, artifacts, case)

    cgls_header, cgls_arrays = parse_cgls_capture(cgls_binary)
    if cgls_header["version"] != 3:
        raise ValueError("native497 is not the source-defined warm-start CGLS v3 snapshot")
    if (cgls_header["environment_count"], cgls_header["node_count_per_environment"],
            cgls_header["unknown_count"]) != (state_doc["environment_count"],
            state_doc["node_count"], state_doc["unknown_count"]):
        raise ValueError("CGLS capture and sidecar dimensions differ")

    source_tools = source / "matter/tools"
    sys.path.insert(0, str(source_tools))
    operator_module_path = source_tools / "cardboard_cached_operator_audit.py"
    operator_module = load_module("native497_cached_operator_audit", operator_module_path)
    fd_records = strict_jsonl(fd_capture)
    if not fd_records or fd_records[0].get("record") != "header":
        raise ValueError("operator FD capture lacks header")
    fd_header = fd_records[0]
    if (fd_header.get("root"), fd_header.get("newton_iteration")) != (
            state_doc["root"], state_doc["newton_iteration"]):
        raise ValueError("FD capture and CGLS state sidecar root/Newton differ")
    if fd_header.get("world_fingerprint_hex") != state_doc["world_fingerprint_u64_hex"] or \
       fd_header.get("execution_fingerprint_hex") != state_doc["device_program_fingerprint_u64_hex"]:
        raise ValueError("FD capture and state sidecar fingerprint identity differs")
    state_join = compare_state_to_operator_capture(state_doc, state_payload, fd_records)
    if not state_join["all_geometry_cache_rhs_and_history_joins_exact_or_signedzero_only"]:
        raise ValueError("state sidecar and FD capture do not join on required geometry/cache/RHS/history fields")

    trace_records = []
    for line in stderr_path.read_text(encoding="utf-8", errors="replace").splitlines():
        prefix = "fem_base_cgls_direction="
        if line.startswith(prefix):
            trace_records.append(json.loads(line[len(prefix):], parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))))
    chosen = [x for x in trace_records if x.get("root") == state_doc["root"] and
              x.get("newton_iteration") == state_doc["newton_iteration"]]
    if len(chosen) != 1: raise ValueError("expected exactly one selected native CGLS direction trace")
    cgls_trace = chosen[0]
    if cgls_trace.get("warm_start_refinement") is not True or \
       cgls_trace.get("candidate_represents") != "x0_plus_Sy":
        raise ValueError("native trace does not declare the expected warm-start total-candidate semantics")
    if cgls_trace.get("original_newton_certificate_applies_to_direction") is not False:
        raise ValueError("CGLS trace incorrectly transfers the original Newton certificate")

    per_env = []
    for environment in range(state_doc["environment_count"]):
        per_env.append(residual_evidence(operator_module, state_doc, state_payload,
            cgls_header, cgls_arrays, fd_records, environment, cgls_trace))

    diagnostic_files = receipt.get("diagnostic_capture_validation", {}).get("files", [])
    verified_receipt_files = []
    for row in diagnostic_files:
        rel = Path(row["path"])
        candidate = case / rel
        if not candidate.is_file():
            raise ValueError(f"receipt-bound diagnostic capture file missing: {rel}")
        actual_sha = sha256(candidate)
        if candidate.stat().st_size != row["bytes"] or actual_sha != row["sha256"]:
            raise ValueError(f"receipt-bound diagnostic capture mismatch: {rel}")
        verified_receipt_files.append({"path": str(candidate), "bytes": row["bytes"], "sha256": actual_sha})

    return {
        "schema": "numi.cardboard.native497_cached_cgls_operator_audit.v1",
        "verdict": "CPU-only reconstruction of the captured FP32 cached FEM operator; not an independent constitutive reference, native FP64 run, or physical qualification",
        "run": {"case": str(case), "prediction_sha256": sha256(prediction_path),
                "receipt_sha256": sha256(receipt_path), "stderr_sha256": sha256(stderr_path),
                "exit_code": receipt["exit_code"], "timed_out": receipt["timed_out"],
                "source_inputs_unchanged": receipt["source_inputs_unchanged"],
                "verified_prediction_inputs": input_check,
                "receipt_diagnostic_files_verified": verified_receipt_files},
        "source_binding": {"source_snapshot": str(source),
            "frozen_source_manifest_sha256": sha256(source / "frozen-source.json"),
            "cached_operator_module_sha256": sha256(operator_module_path),
            "state_reader_sha256": sha256(args.state_reader),
            "state_manifest_sha256": sha256(state_manifest),
            "state_payload_sha256": sha256(state_payload_path),
            "cgls_binary_sha256": sha256(cgls_binary),
            "operator_fd_capture_sha256": sha256(fd_capture)},
        "capture_identity": {"root": state_doc["root"], "newton_iteration": state_doc["newton_iteration"],
            "environment_count": state_doc["environment_count"], "node_count": state_doc["node_count"],
            "tetrahedron_count": state_doc["tetrahedron_count"], "unknown_count": state_doc["unknown_count"],
            "dt_seconds": state_doc["dt_seconds"], "world_fingerprint": state_doc["world_fingerprint_u64_hex"],
            "device_program_fingerprint": state_doc["device_program_fingerprint_u64_hex"],
            "operator_fd_world_fingerprint": fd_header["world_fingerprint_hex"],
            "operator_fd_execution_fingerprint": fd_header["execution_fingerprint_hex"]},
        "state_reader_validation": reader_validation,
        "state_to_operator_capture_join": state_join,
        "cgls_binary_header": cgls_header,
        "native_trace": cgls_trace,
        "per_environment_operator_replay": per_env,
        "interpretation": [
            "All reconstructed entries originate in native497's captured FP32 tangent cache and geometry; no constitutive law is reevaluated.",
            "Comparisons against FP64 sparse arithmetic quantify cached-operator reconstruction and subtraction behavior only.",
            "Residual subtraction cancellation and any optional condition estimate are separate diagnostics; neither is a solver or physical qualification.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--source-snapshot", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--state-reader", type=Path, default=DEFAULT_STATE_READER)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = run(args)
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            if args.output.exists():
                raise ValueError(f"refusing to overwrite existing report {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        else:
            print(encoded, end="")
    except Exception as error:
        print(f"native497 cached-operator audit rejected input: {type(error).__name__}: {error}",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
