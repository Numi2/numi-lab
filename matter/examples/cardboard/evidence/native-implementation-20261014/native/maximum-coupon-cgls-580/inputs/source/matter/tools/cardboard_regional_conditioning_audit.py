#!/usr/bin/env python3
"""Bounded CPU audit of a sealed regional-Galerkin conditioning capture.

The capture is diagnostic data from a rejected physical step.  This tool checks
the run/source bindings and raw arenas, then evaluates the captured coarse
operator and its one-sided Ritz residuals.  It does not reproduce the full
FEM operator or diagnose the nonlinear/FGMRES failure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
from typing import Any


SCHEMA = "numi.cardboard.regional-conditioning.v1"
MAX_ENVIRONMENTS = 4
MAX_NODES = 100_000
MAX_MODES = 32
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_ARENA_BYTES = 256 * 1024 * 1024
MAX_SEALED_INPUT_COUNT = 10_000
EXPECTED_LAYOUT = (
    "P mode-node; effective_P/AP environment-mode-node; "
    "RAP pivoted LU environment-row-column; block_inverse environment-node-row"
)
EXPECTED_VALUE_FILES = {
    "candidate_nodes.bin": ("f", 80),
    "P.bin": ("f", 16),
    "effective_P.bin": ("f", 16),
    "AP.bin": ("f", 16),
    "pivoted_RAP.bin": ("f", 4),
    "pivots.bin": ("I", 4),
    "factor_valid.bin": ("I", 4),
    "node_block_inverse.bin": ("f", 16),
}


class AuditError(ValueError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise AuditError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _safe_child(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise AuditError(f"unsafe relative path {relative!r}")
    root_resolved = root.resolve()
    candidate = (root_resolved / rel).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise AuditError(f"path escapes bound root: {relative!r}")
    return candidate


def _verify_outer_run(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    prediction = _read_json(run_dir / "prediction.json")
    receipt = _read_json(run_dir / "receipt.json")
    result = _read_json(run_dir / "output" / "result.json")

    if manifest.get("accepted_physical_state") is not False:
        raise AuditError("conditioning capture must be marked non-accepted")
    expected_fp = str(manifest.get("source_physics_fingerprint", ""))
    if not expected_fp or str(result.get("compiled_world_fingerprint", "")) != expected_fp:
        raise AuditError("outer result fingerprint does not match the conditioning capture")
    if result.get("status") != "failed" or result.get("accepted_steps") != 1:
        raise AuditError("outer run is not the expected one-accepted-step rejected run")
    if receipt.get("execution_started") is not True or receipt.get("source_inputs_unchanged") is not True:
        raise AuditError("outer receipt does not confirm execution and unchanged inputs")
    if receipt.get("timed_out") is not False or receipt.get("exit_code") != 2:
        raise AuditError("outer receipt does not show the expected non-timeout rejection")

    if prediction.get("environment", {}).get("NM_FEM_RESIDUAL_ARMIJO_ROOT") != "all":
        raise AuditError("required all-root residual Armijo environment binding is missing")
    argv = prediction.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise AuditError("prediction argv is missing or malformed")
    flags = set(argv)
    required_flags = {
        "--accurate-paper-projection",
        "--scaled-paper-residuals",
        "--without-self-contact",
        "--reference-displacement-gradient",
        "--compensated-paper-arithmetic",
        "--safeguarded-paper-initializer",
    }
    missing = sorted(required_flags - flags)
    if missing:
        raise AuditError(f"required source/numerical flags missing from argv: {missing}")
    values: dict[str, str] = {}
    for index, item in enumerate(argv[:-1]):
        if item.startswith("--"):
            values[item] = argv[index + 1]
    expected_values = {
        "--preset": "literature2009",
        "--line-search-steps": "24",
        "--residual-tolerance": "1e-6",
        "--newton-iterations": "20",
        "--fgmres-restart": "16",
        "--fgmres-iterations": "256",
        "--fgmres-preconditioner": "regional-galerkin-block",
        "--steps": "12",
        "--hold-steps": "3",
        "--unload-steps": "6",
        "--relax-steps": "3",
        "--release-steps": "3",
        "--nx": "8",
        "--ny": "2",
        "--thickness-slices": "1",
        "--bend-angle": ".375",
        "--dt": ".0001",
        "--material-iterations": "16",
    }
    bad = {key: (values.get(key), expected) for key, expected in expected_values.items()
           if values.get(key) != expected}
    if bad:
        raise AuditError(f"prediction argv differs from the conditioning-case protocol: {bad}")

    # Verify every declared source/runtime/external input against its sealed
    # root.  This binds the analysis to the captured binary and authored code.
    source_snapshot = Path(str(prediction.get("source_snapshot", "")))
    runtime_dir = Path(str(prediction.get("runtime_dir", "")))
    if not source_snapshot.is_dir() or not runtime_dir.is_dir():
        raise AuditError("sealed source snapshot or runtime directory is unavailable")
    external_paths = prediction.get("external_inputs")
    input_hashes = prediction.get("inputs_sha256")
    if not isinstance(external_paths, list) or not isinstance(input_hashes, dict):
        raise AuditError("sealed input hash map is missing")
    external_by_name: dict[str, Path] = {}
    for item in external_paths:
        path = Path(str(item))
        if path.name in external_by_name:
            raise AuditError(f"ambiguous external input basename: {path.name}")
        external_by_name[path.name] = path
    verified = 0
    if len(input_hashes) > MAX_SEALED_INPUT_COUNT:
        raise AuditError("sealed input hash map exceeds audit bound")
    for key, expected_hash in input_hashes.items():
        if not isinstance(key, str) or not isinstance(expected_hash, str):
            raise AuditError("malformed source hash entry")
        prefix, separator, relative = key.partition("/")
        if not separator:
            raise AuditError(f"unknown sealed input namespace: {key}")
        if prefix == "source":
            path = _safe_child(source_snapshot, relative)
        elif prefix == "runtime":
            path = _safe_child(runtime_dir, relative)
        elif prefix == "external":
            path = external_by_name.get(Path(relative).name)
            if path is None:
                raise AuditError(f"unmapped external input: {key}")
        else:
            raise AuditError(f"unknown sealed input namespace: {prefix}")
        if not path.is_file() or _sha256(path) != expected_hash:
            raise AuditError(f"sealed input hash mismatch: {key}")
        verified += 1
    status_header = _safe_child(source_snapshot, "matter/include/numi/matter/shared.h")
    try:
        header_text = status_header.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise AuditError(f"cannot read sealed status enum: {exc}") from exc
    status_match = re.search(r"\bNM_STATUS_NONLINEAR_SOLVER_FAILURE\s*=\s*10u\b", header_text)
    if status_match is None:
        raise AuditError("sealed source does not bind status code 10 to nonlinear solver failure")

    observation_path = run_dir / "output" / "observations.csv"
    try:
        with observation_path.open(newline="", encoding="utf-8") as stream:
            observations = list(csv.DictReader(stream))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise AuditError(f"cannot read sealed-run observations: {exc}") from exc
    rejected_rows = [row for row in observations
                     if row.get("step") == str(manifest.get("control_step"))
                     and row.get("environment") == "0"]
    if len(rejected_rows) != 1:
        raise AuditError("expected exactly one environment-0 observation for captured control step")
    rejected = rejected_rows[0]
    if (rejected.get("status_code") != "10" or rejected.get("step_accepted") != "0" or
            rejected.get("certificate_raw_accepted_flag") != "0"):
        raise AuditError("control-step observation does not confirm a rejected nonlinear-solver step")
    if rejected.get("fgmres_iterations") != "256":
        raise AuditError("control-step FGMRES iteration count differs from sealed prediction")
    return {
        "verified_input_count": verified,
        "source_snapshot": str(source_snapshot),
        "runtime_dir": str(runtime_dir),
        "result_status": result["status"],
        "accepted_steps": result["accepted_steps"],
        "receipt_exit_code": receipt["exit_code"],
        "rejected_observation": {
            "step": int(rejected["step"]),
            "environment": int(rejected["environment"]),
            "status_code": int(rejected["status_code"]),
            "status_name_from_sealed_source": "NM_STATUS_NONLINEAR_SOLVER_FAILURE",
            "fgmres_iterations": int(rejected["fgmres_iterations"]),
            "raw_certificate_accepted_flag": int(rejected["certificate_raw_accepted_flag"]),
            "step_accepted": int(rejected["step_accepted"]),
            "certificate_residual_raw": rejected.get("certificate_residual"),
            "certificate_correction_raw": rejected.get("certificate_correction"),
            "status_diagnostic_x_raw": rejected.get("status_diagnostic_x"),
            "status_diagnostic_y_raw": rejected.get("status_diagnostic_y"),
            "free_force_imbalance_l2_N_raw": rejected.get("free_force_imbalance_l2_N"),
        },
        "required_flags": sorted(required_flags),
        "argv_values": expected_values,
    }


def _decode(path: Path, code: str, expected_count: int, stride: int) -> list[Any]:
    expected_bytes = expected_count * stride
    if expected_bytes > MAX_FILE_BYTES:
        raise AuditError(f"arena exceeds per-file bound: {path.name}")
    try:
        actual_bytes = path.stat().st_size
    except OSError as exc:
        raise AuditError(f"cannot stat {path}: {exc}") from exc
    if actual_bytes != expected_bytes:
        raise AuditError(f"arena byte length mismatch for {path.name}: {actual_bytes} != {expected_bytes}")
    if actual_bytes > MAX_FILE_BYTES:
        raise AuditError(f"arena exceeds per-file bound: {path.name}")
    item_size = struct.calcsize("<" + code)
    if stride % item_size:
        raise AuditError(f"invalid scalar stride for {path.name}")
    with path.open("rb") as stream:
        data = stream.read(expected_bytes + 1)
    if len(data) != expected_bytes:
        raise AuditError(f"short or growing arena read for {path.name}")
    try:
        values = [entry[0] for entry in struct.iter_unpack("<" + code, data)]
    except struct.error as exc:
        raise AuditError(f"cannot decode {path.name}: {exc}") from exc
    if code == "f" and any(not math.isfinite(value) for value in values):
        raise AuditError(f"nonfinite float in {path.name}")
    return values


def _matmul_gram(left: list[float], right: list[float], modes: int, width: int) -> list[list[float]]:
    out = [[0.0] * modes for _ in range(modes)]
    rows_left = [left[i * width:(i + 1) * width] for i in range(modes)]
    rows_right = [right[i * width:(i + 1) * width] for i in range(modes)]
    for i in range(modes):
        li = rows_left[i]
        for j in range(modes):
            rj = rows_right[j]
            total = 0.0
            for k in range(width):
                total += float(li[k]) * float(rj[k])
            out[i][j] = total
    return out


def _jacobi_symmetric(matrix: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    n = len(matrix)
    a = [row[:] for row in matrix]
    vectors = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    scale = max((abs(x) for row in a for x in row), default=0.0)
    tolerance = max(1e-15, scale * 1e-14)
    for _ in range(max(1, 80 * n * n)):
        p, q, largest = 0, 0, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > largest:
                    p, q, largest = i, j, abs(a[i][j])
        if largest <= tolerance:
            break
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        tau = (aqq - app) / (2.0 * apq)
        t = math.copysign(1.0, tau) / (abs(tau) + math.sqrt(1.0 + tau * tau)) if tau else 1.0
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c
        a[p][p] = app - t * apq
        a[q][q] = aqq + t * apq
        a[p][q] = a[q][p] = 0.0
        for k in range(n):
            if k != p and k != q:
                akp, akq = a[k][p], a[k][q]
                a[k][p] = a[p][k] = c * akp - s * akq
                a[k][q] = a[q][k] = s * akp + c * akq
            vkp, vkq = vectors[k][p], vectors[k][q]
            vectors[k][p] = c * vkp - s * vkq
            vectors[k][q] = s * vkp + c * vkq
    else:
        raise AuditError("symmetric eigensolver did not converge")
    order = sorted(range(n), key=lambda i: a[i][i])
    values = [a[i][i] for i in order]
    ordered_vectors = [[vectors[row][col] for col in order] for row in range(n)]
    return values, ordered_vectors


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    a = [matrix[i][:] + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-20:
            raise AuditError("basis Gram matrix is singular")
        a[col], a[pivot] = a[pivot], a[col]
        factor = a[col][col]
        for j in range(col, n + 1):
            a[col][j] /= factor
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            for j in range(col, n + 1):
                a[row][j] -= factor * a[col][j]
    return [a[i][n] for i in range(n)]


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _lu_relative_reconstruction(factors: list[float], permutation: list[int], original: list[list[float]]) -> float:
    n = len(original)
    if sorted(permutation) != list(range(n)):
        raise AuditError("pivot table is not a permutation")
    lu = [factors[i * n:(i + 1) * n] for i in range(n)]
    product = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                l_value = 1.0 if i == k else (lu[i][k] if i > k else 0.0)
                u_value = lu[k][j] if k <= j else 0.0
                total += l_value * u_value
            product[i][j] = total
    target = [original[permutation[i]][:] for i in range(n)]
    error = math.sqrt(sum((product[i][j] - target[i][j]) ** 2
                          for i in range(n) for j in range(n)))
    denominator = math.sqrt(sum(target[i][j] ** 2 for i in range(n) for j in range(n)))
    if denominator == 0.0:
        raise AuditError("pivoted RAP has zero norm")
    return error / denominator


def _audit_one_sided(eb: list[float], ab: list[float], modes: int,
                     width: int, gram: list[list[float]],
                     vectors: list[list[float]], eigvalues: list[float],
                     which: int) -> dict[str, float]:
    q = [vectors[row][which] for row in range(modes)]
    norm_sq = sum(q[i] * gram[i][j] * q[j] for i in range(modes) for j in range(modes))
    qscale = 1.0 / math.sqrt(norm_sq)
    q = [value * qscale for value in q]
    lam = eigvalues[which]
    v = [0.0] * width
    av = [0.0] * width
    for mode, coefficient in enumerate(q):
        ebrow = eb[mode * width:(mode + 1) * width]
        abrow = ab[mode * width:(mode + 1) * width]
        for k in range(width):
            v[k] += coefficient * ebrow[k]
            av[k] += coefficient * abrow[k]
    residual = [av[k] - lam * v[k] for k in range(width)]
    rnorm = _norm(residual)
    avnorm = _norm(av)
    etr = [sum(eb[i * width + k] * residual[k] for k in range(width))
           for i in range(modes)]
    projected_coefficients = _solve(gram, etr)
    projected_norm_sq = sum(projected_coefficients[i] * gram[i][j] * projected_coefficients[j]
                            for i in range(modes) for j in range(modes))
    projected_norm = math.sqrt(max(0.0, projected_norm_sq))
    orthogonal_sq = max(0.0, rnorm * rnorm - projected_norm * projected_norm)
    return {
        "ritz_value": lam,
        "full_space_A_invariance_residual_relative_to_Aq": rnorm / max(avnorm, 1e-300),
        "residual_projected_onto_coarse_basis_fraction": projected_norm / max(rnorm, 1e-300),
        "residual_orthogonal_to_coarse_basis_fraction": math.sqrt(orthogonal_sq) / max(rnorm, 1e-300),
    }


def audit_capture(capture_dir: Path, run_dir: Path | None = None) -> dict[str, Any]:
    capture_dir = capture_dir.resolve()
    manifest = _read_json(capture_dir / "manifest.json")
    if manifest.get("schema") != SCHEMA:
        raise AuditError("unsupported conditioning manifest schema")
    if manifest.get("layout") != EXPECTED_LAYOUT:
        raise AuditError("unexpected arena layout string")
    if manifest.get("byte_order") != "native_little_endian":
        raise AuditError("capture is not explicitly little-endian")
    environments = manifest.get("environment_count")
    nodes = manifest.get("fem_node_count")
    modes = manifest.get("mode_count")
    if not all(isinstance(x, int) for x in (environments, nodes, modes)):
        raise AuditError("malformed capture dimensions")
    if not (1 <= environments <= MAX_ENVIRONMENTS and 1 <= nodes <= MAX_NODES and
            1 <= modes <= MAX_MODES):
        raise AuditError("capture dimensions exceed bounded audit limits")

    expected_counts = {
        "candidate_nodes.bin": environments * nodes,
        "P.bin": modes * nodes,
        "effective_P.bin": environments * modes * nodes,
        "AP.bin": environments * modes * nodes,
        "pivoted_RAP.bin": environments * modes * modes,
        "pivots.bin": environments * modes,
        "factor_valid.bin": environments,
        "node_block_inverse.bin": environments * nodes * 3,
    }
    file_rows = manifest.get("files")
    if not isinstance(file_rows, list):
        raise AuditError("manifest files list is missing")
    rows_by_name: dict[str, dict[str, Any]] = {}
    for row in file_rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise AuditError("malformed file metadata entry")
        name = row["name"]
        if name in rows_by_name:
            raise AuditError(f"duplicate arena name: {name}")
        if name not in EXPECTED_VALUE_FILES or Path(name).name != name:
            raise AuditError(f"unexpected arena name: {name}")
        rows_by_name[name] = row
    if set(rows_by_name) != set(EXPECTED_VALUE_FILES):
        raise AuditError("arena set does not match the required capture schema")
    expected_total_bytes = sum(expected_counts[name] * EXPECTED_VALUE_FILES[name][1]
                               for name in EXPECTED_VALUE_FILES)
    if expected_total_bytes > MAX_TOTAL_ARENA_BYTES:
        raise AuditError("raw arenas exceed the aggregate audit bound")

    decoded: dict[str, list[Any]] = {}
    raw_hashes: dict[str, str] = {}
    for name, (code, stride) in EXPECTED_VALUE_FILES.items():
        row = rows_by_name[name]
        count = expected_counts[name]
        if row.get("count") != count or row.get("stride_bytes") != stride:
            raise AuditError(f"count/stride mismatch for {name}")
        if row.get("bytes") != count * stride:
            raise AuditError(f"manifest byte count mismatch for {name}")
        path = _safe_child(capture_dir, name)
        if not path.is_file() or path.stat().st_size != count * stride:
            raise AuditError(f"raw file is missing or has the wrong size: {name}")
        actual_hash = _sha256(path)
        if actual_hash != row.get("sha256"):
            raise AuditError(f"raw arena SHA-256 mismatch: {name}")
        raw_hashes[name] = actual_hash
        decoded[name] = _decode(path, code, count, stride)

    if any(v not in (0, 1) for v in decoded["factor_valid.bin"]):
        raise AuditError("factor_valid has values outside the boolean domain")
    if decoded["factor_valid.bin"] != [1] * environments:
        raise AuditError("one or more exported RAP factors were invalid")
    pivots = decoded["pivots.bin"]
    for env in range(environments):
        block = pivots[env * modes:(env + 1) * modes]
        if sorted(block) != list(range(modes)):
            raise AuditError(f"pivot block {env} is not a row permutation")

    block_values = decoded["node_block_inverse.bin"]
    block_valid = []
    for env in range(environments):
        valid_nodes = 0
        for node in range(nodes):
            first = ((env * nodes + node) * 3) * 4
            row_flags = [block_values[first + r * 4 + 3] for r in range(3)]
            if row_flags not in ([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]):
                raise AuditError("node-block validity marker has unexpected row pattern")
            valid_nodes += int(row_flags[0] == 1.0)
        block_valid.append(valid_nodes)

    p = decoded["P.bin"]
    effective = decoded["effective_P.bin"]
    ap = decoded["AP.bin"]
    width = nodes * 4
    authored_gram = _matmul_gram(p, p, modes, width)
    authored_gram_eigenvalues, _ = _jacobi_symmetric(
        [[0.5 * (authored_gram[i][j] + authored_gram[j][i]) for j in range(modes)]
         for i in range(modes)])
    if authored_gram_eigenvalues[0] <= 0.0:
        raise AuditError("authored P basis Gram matrix is not positive definite")
    authored_gram_identity_error = math.sqrt(sum(
        (authored_gram[i][j] - (1.0 if i == j else 0.0)) ** 2
        for i in range(modes) for j in range(modes)))
    env_metrics = []
    for env in range(environments):
        offset = env * modes * width
        eb = effective[offset:offset + modes * width]
        ab = ap[offset:offset + modes * width]
        gram = _matmul_gram(eb, eb, modes, width)
        rap = _matmul_gram(eb, ab, modes, width)
        sym = [[0.5 * (rap[i][j] + rap[j][i]) for j in range(modes)]
               for i in range(modes)]
        gram_eigenvalues, gram_eigenvectors = _jacobi_symmetric(gram)
        if gram_eigenvalues[0] <= 0.0:
            raise AuditError(f"effective basis Gram matrix is not positive definite in environment {env}")
        gram_inverse_sqrt = [[
            sum(gram_eigenvectors[i][k] * gram_eigenvectors[j][k] /
                math.sqrt(gram_eigenvalues[k]) for k in range(modes))
            for j in range(modes)] for i in range(modes)]
        whitened = [[
            sum(gram_inverse_sqrt[i][a] * sym[a][b] * gram_inverse_sqrt[b][j]
                for a in range(modes) for b in range(modes))
            for j in range(modes)] for i in range(modes)]
        eigvalues, whitened_vectors = _jacobi_symmetric(whitened)
        eigvectors = [[sum(gram_inverse_sqrt[i][j] * whitened_vectors[j][k]
                           for j in range(modes))
                       for k in range(modes)] for i in range(modes)]
        if not eigvalues or eigvalues[0] <= 0.0:
            raise AuditError(f"symmetric RAP is not positive definite in environment {env}")
        nonsym_num = math.sqrt(sum((rap[i][j] - rap[j][i]) ** 2
                                   for i in range(modes) for j in range(modes)))
        rap_norm = math.sqrt(sum(rap[i][j] ** 2 for i in range(modes) for j in range(modes)))
        factor_offset = env * modes * modes
        pivot_offset = env * modes
        lu_error = _lu_relative_reconstruction(
            decoded["pivoted_RAP.bin"][factor_offset:factor_offset + modes * modes],
            pivots[pivot_offset:pivot_offset + modes], rap)
        min_mode = _audit_one_sided(eb, ab, modes, width, gram, eigvectors, eigvalues, 0)
        max_mode = _audit_one_sided(eb, ab, modes, width, gram, eigvectors, eigvalues, modes - 1)
        gram_identity_error = math.sqrt(sum((gram[i][j] - (1.0 if i == j else 0.0)) ** 2
                                            for i in range(modes) for j in range(modes)))
        env_metrics.append({
            "environment": env,
            "authored_P_basis_gram_frobenius_distance_from_identity": authored_gram_identity_error,
            "authored_P_basis_gram_condition_number": authored_gram_eigenvalues[-1] / authored_gram_eigenvalues[0],
            "basis_gram_frobenius_distance_from_identity": gram_identity_error,
            "symmetric_RAP_min_eigenvalue": eigvalues[0],
            "symmetric_RAP_max_eigenvalue": eigvalues[-1],
            "symmetric_RAP_condition_number": eigvalues[-1] / eigvalues[0],
            "RAP_relative_nonsymmetry": nonsym_num / max(rap_norm, 1e-300),
            "pivoted_LU_relative_reconstruction_error_against_unpivoted_FP64_reduction": lu_error,
            "smallest_Ritz_mode": min_mode,
            "largest_Ritz_mode": max_mode,
            "valid_node_block_count": block_valid[env],
        })

    outer = None
    if run_dir is None:
        # output/rejected-conditioning-*/manifest.json -> run directory
        run_dir = capture_dir.parent.parent
    outer = _verify_outer_run(run_dir.resolve(), manifest)
    return {
        "schema": "numi.cardboard.regional-conditioning-audit.v1",
        "classification": "rejected_step_diagnostic_only",
        "accepted_physical_state": False,
        "control_step": manifest.get("control_step"),
        "source_physics_fingerprint": str(manifest.get("source_physics_fingerprint")),
        "device_program_fingerprint": str(manifest.get("device_program_fingerprint")),
        "dimensions": {"environments": environments, "nodes": nodes, "modes": modes},
        "raw_arena_sha256": raw_hashes,
        "outer_artifact_sha256": {
            name: _sha256(run_dir.resolve() / name)
            for name in ("prediction.json", "receipt.json", "stdout.log", "stderr.log")
            if (run_dir.resolve() / name).is_file()
        } | {
            "output/result.json": _sha256(run_dir.resolve() / "output" / "result.json"),
            "output/observations.csv": _sha256(run_dir.resolve() / "output" / "observations.csv"),
            "output/conditioning-manifest.json": _sha256(capture_dir / "manifest.json"),
        },
        "outer_binding": outer,
        "factor_valid": decoded["factor_valid.bin"],
        "valid_node_block_count_per_environment": block_valid,
        "environments": env_metrics,
        "interpretation": {
            "coarse_factor": "all exported pivoted RAP factors are valid and reconstruct the captured coarse matrix within the reported FP32 capture/reduction error",
            "coarse_conditioning": "the captured symmetric RAP matrices are positive definite with the reported condition numbers; this does not establish conditioning or convergence of the full FEM operator",
            "one_sided_residual": "Ritz A-invariance residuals compare captured AP times coarse Ritz vectors against the same vector times its projected Ritz value; the projected/orthogonal fractions are not an FGMRES right-hand-side residual",
            "cause_limit": "the capture contains no full operator, Krylov basis, or rejected linear-system right-hand side; it cannot identify the first failing solver subsystem or attribute the run rejection to RAP/LU",
            "physical_limit": "candidate_nodes and conditioning arenas are diagnostic data from a rejected step, not accepted geometry or physical evidence",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path, help="directory containing manifest.json and raw arenas")
    parser.add_argument("--run-dir", type=Path, help="outer run directory; defaults to capture_dir/../../")
    parser.add_argument("--output", type=Path, help="write result JSON here; otherwise print it")
    args = parser.parse_args(argv)
    try:
        result = audit_capture(args.capture_dir, args.run_dir)
    except AuditError as exc:
        print(f"conditioning audit rejected input: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
