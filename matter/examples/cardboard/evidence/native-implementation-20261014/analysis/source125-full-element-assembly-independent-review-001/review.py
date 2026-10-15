#!/usr/bin/env python3
"""Independent integrity and scalar-contraction review of audit 150-002."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import ast
import zipfile

BASE = Path("/Users/home/cardboard-implementation-20261014")
V1 = BASE / "analysis/source125-full-element-assembly-fp64-audit-150-001"
V2 = BASE / "analysis/source125-full-element-assembly-fp64-verification-150-002"
OUT = BASE / "analysis/source125-full-element-assembly-independent-review-001"
CAP = BASE / "solver/two-stage-first-failure-rich-fd-419/operator-capture"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def f32_array(values):
    return [f32(value) for value in values]


def npz_arrays(path: Path) -> dict[str, dict]:
    """Read the numeric NPY subset used by the retained NPZ files."""
    arrays = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            raw = archive.read(member)
            if raw[:6] != b"\x93NUMPY":
                raise ValueError(f"invalid NPY member in {path}: {member}")
            major = raw[6]
            if major == 1:
                header_len = struct.unpack_from("<H", raw, 8)[0]
                header_start = 10
            elif major in (2, 3):
                header_len = struct.unpack_from("<I", raw, 8)[0]
                header_start = 12
            else:
                raise ValueError(f"unsupported NPY version {major}")
            header = ast.literal_eval(raw[header_start:header_start + header_len].decode("latin1").strip())
            payload = raw[header_start + header_len:]
            if header["fortran_order"]:
                raise ValueError(f"unexpected Fortran-order numeric payload in {path}:{member}")
            descr = header["descr"]
            shape = header["shape"]
            count = math.prod(shape) if shape else 1
            formats = {"<f8": (8, "d"), "<f4": (4, "f"), "<i8": (8, "q"),
                       "<i4": (4, "i"), "<u4": (4, "I"), "|b1": (1, "?")}
            if descr.startswith("|S"):
                width = int(descr[2:])
                values = [payload[i:i + width] for i in range(0, len(payload), width)]
            else:
                if descr not in formats:
                    raise ValueError(f"unsupported dtype {descr} in {path}:{member}")
                width, fmt = formats[descr]
                if len(payload) != count * width:
                    raise ValueError(f"NPY payload extent mismatch in {path}:{member}")
                values = list(struct.unpack("<" + fmt * count, payload))
            arrays[member[:-4]] = {"shape": shape, "descr": descr, "values": values}
    return arrays


def nested(values, shape):
    if len(shape) == 1:
        return values[:]
    stride = math.prod(shape[1:])
    return [nested(values[i:i + stride], shape[1:]) for i in range(0, len(values), stride)]


def verify_manifest(root: Path, manifest_path: Path) -> dict:
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in doc["files"].items():
        path = root / name
        if sha(path) != expected:
            raise ValueError(f"bound file hash mismatch: {path}")
    for name, expected in doc["inputs"].items():
        path = Path(name)
        if sha(path) != expected:
            raise ValueError(f"bound input hash mismatch: {path}")
    return doc


def fsum_products(left, right) -> float:
    return math.fsum(float(a) * float(b) for a, b in zip(left, right))


def qrotation(q):
    x, y, z, w = f32_array(q)
    norm = math.sqrt(math.fsum(v * v for v in (x, y, z, w)))
    x, y, z, w = (v / norm for v in (x, y, z, w))
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def load_capture():
    nodes = []
    tets = []
    integrity = None
    with CAP.open("r", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        for line in stream:
            record = json.loads(line)
            if record.get("record") == "node" and record["environment"] == 0:
                nodes.append(record)
            elif record.get("record") == "tetrahedron" and record["environment"] == 0:
                tets.append(record)
            elif record.get("record") == "integrity":
                integrity = record
    nodes.sort(key=lambda row: row["node_id"])
    tets.sort(key=lambda row: row["tet_id"])
    if len(nodes) != 882 or len(tets) != 1920:
        raise ValueError(f"unexpected captured mesh dimensions: {len(nodes)} nodes, {len(tets)} tets")
    if integrity is None or integrity.get("production_inputs_bitwise_unchanged") is not True or integrity.get("cached_tangent_snapshot_bitwise_guard_included") is not True:
        raise ValueError("419 capture integrity footer is absent or does not certify unchanged inputs")
    if [row["node_id"] for row in nodes] != list(range(882)) or [row["tet_id"] for row in tets] != list(range(1920)):
        raise ValueError("captured node or tetrahedron IDs are not unique dense ranges")
    return header, nodes, tets, integrity


def matmul_scalar(a, b):
    nrow, inner, ncol = len(a), len(b), len(b[0])
    return [
        [math.fsum(float(a[i][k]) * float(b[k][j]) for k in range(inner))
         for j in range(ncol)]
        for i in range(nrow)
    ]


def local_b(inv_rows, q, frame_active):
    inv = [[f32(v) for v in row] for row in inv_rows]
    gradients = [[-math.fsum(inv[r][c] for r in range(3)) for c in range(3)], *inv]
    rot = qrotation(q) if frame_active else [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    grad_rot = matmul_scalar(gradients, rot)
    b = [[0.0] * 12 for _ in range(9)]
    for a in range(3):
        for component in range(3):
            flat_row = 3 * a + component
            for local_node in range(4):
                for displacement in range(3):
                    b[flat_row][3 * local_node + displacement] = (
                        grad_rot[local_node][component] if a == displacement else 0.0
                    )
    return b


def local_stiffness(b, tangent, volume):
    # All products and reductions are scalar Python float operations; no BLAS,
    # NumPy matmul, einsum, or source helper contraction is used here.
    bt_c = [
        [math.fsum(b[k][i] * float(tangent[k][j]) for k in range(9)) for j in range(9)]
        for i in range(12)
    ]
    return [
        [volume * math.fsum(bt_c[i][k] * b[k][j] for k in range(9)) for j in range(12)]
        for i in range(12)
    ]


def add_matrix(entries, i, j, value):
    entries.setdefault((i, j), []).append(value)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v2m = verify_manifest(V2, V2 / "manifest.json")
    v1m = verify_manifest(V1, V1 / "manifest.json")
    tangent_review = BASE / "analysis/source125-full-tangent-independent-review-001"
    trm = json.loads((tangent_review / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in trm["bound_inputs"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"tangent independent-review input hash mismatch: {name}")
    review_files = {
        "README.md": trm["readme_sha256"],
        "report.json": trm["report_sha256"],
        "verification-details.json": trm["details_sha256"],
        "verify.py": trm["verifier_sha256"],
    }
    for name, expected in review_files.items():
        if sha(tangent_review / name) != expected:
            raise ValueError(f"tangent independent-review artifact hash mismatch: {name}")
    rerun_manifest = BASE / "analysis/review147-independent-rerun-001/manifest.json"
    if sha(rerun_manifest) != trm["rerun_manifest_sha256"]:
        raise ValueError("tangent independent-review rerun manifest hash mismatch")
    tangent_audit = BASE / "analysis/source125-full-tangent-fp64-audit-147-001"
    tangent_manifest = verify_manifest(tangent_audit, tangent_audit / "manifest.json")
    tangent_report = json.loads((tangent_audit / "report.json").read_text(encoding="utf-8"))
    cap_hash = sha(CAP)
    if cap_hash != "f2d42314cfee0428ecd285ebb9dbd5da1556624e3e92af1d12240fd8106b309a":
        raise ValueError("419 capture hash changed")
    h, nodes, tets, capture_integrity = load_capture()
    if (h.get("schema") != "numi.cardboard.fem-newton-capture.v1" or
            h.get("root") != 1 or h.get("newton_iteration") != 48 or
            h.get("environment_count") != 2 or
            h.get("node_count_per_environment") != 882 or
            h.get("tetrahedron_count_per_environment") != 1920 or
            h.get("center_residual_source") != "pre_fgmres_private_copy" or
            h.get("cached_algorithmic_tangent_capture", {}).get("available") is not True):
        raise ValueError("419 capture header does not match the audited linearization")
    matrices = npz_arrays(tangent_audit / "matrices.npz")
    cached_array = matrices["native_C_Pa"]
    independent_array = matrices["independent_C_Pa"]
    cached = nested(cached_array["values"], cached_array["shape"])
    independent = nested(independent_array["values"], independent_array["shape"])
    if cached_array["shape"] != (1920, 9, 9) or independent_array["shape"] != (1920, 9, 9):
        raise ValueError("captured tangent arrays have unexpected dimensions")
    if any(not math.isfinite(value) for matrix in cached + independent for row in matrix for value in row) or not all(matrices["valid_projection"]["values"]):
        raise ValueError("tangent arrays contain invalid input")

    free_nodes = []
    for node in nodes:
        tag = f32(node["center"]["constraint_tag"])
        if tag not in (1., 3.):
            free_nodes.append(node["node_id"])
    if len(free_nodes) != 846:
        raise ValueError(f"unexpected free-node count {len(free_nodes)}")
    node_map = {node: i for i, node in enumerate(free_nodes)}
    dimension = 3 * len(free_nodes)
    mass = [f32(nodes[i]["center"]["mass_kg"]) for i in free_nodes]
    cached_entries: dict[tuple[int, int], list[float]] = {}
    independent_entries: dict[tuple[int, int], list[float]] = {}
    max_local_column_rel = 0.0
    max_local_matrix_rel = 0.0
    max_local_column_abs = 0.0
    element_count = 0
    for tet_id, record in enumerate(tets):
        if record["tet_id"] != tet_id:
            raise ValueError("tetrahedron IDs are not a dense ordered range")
        if record.get("cached_algorithmic_tangent_local", {}).get("valid") is not True:
            raise ValueError(f"captured native tangent is invalid at tetrahedron {tet_id}")
        captured_tangent = [f32(v) for v in record["cached_algorithmic_tangent_local"]
                            ["matrix_row_major_local_dP_dF_Pa_per_dimensionless_F"]]
        if len(captured_tangent) != 81:
            raise ValueError(f"captured tangent width is invalid at tetrahedron {tet_id}")
        native_tangent = [float(v) for row in cached[tet_id] for v in row]
        if captured_tangent != native_tangent:
            raise ValueError(f"147 cached tangent input differs from native 419 capture at tetrahedron {tet_id}")
        meta = f32_array(record["center"]["dt_seconds_bulk_modulus_kind_frame_active"])
        volume = f32(record["rest_volume_m3"])
        b = local_b(record["inverse_rest_rows_per_m"], record["material_frame_quaternion_xyzw"], meta[3] == 1.0)
        kn = local_stiffness(b, cached[tet_id], volume)
        kr = local_stiffness(b, independent[tet_id], volume)
        local_err = [[kn[i][j] - kr[i][j] for j in range(12)] for i in range(12)]
        for col in range(12):
            numerator = math.sqrt(math.fsum(local_err[row][col] ** 2 for row in range(12)))
            denominator = math.sqrt(math.fsum(kr[row][col] ** 2 for row in range(12)))
            if denominator == 0.0:
                raise ValueError(f"zero reference element column at tet {tet_id}, column {col}")
            max_local_column_rel = max(max_local_column_rel, numerator / denominator)
            max_local_column_abs = max(max_local_column_abs, numerator)
        denom_matrix = math.sqrt(math.fsum(kr[i][j] ** 2 for i in range(12) for j in range(12)))
        numer_matrix = math.sqrt(math.fsum(local_err[i][j] ** 2 for i in range(12) for j in range(12)))
        max_local_matrix_rel = max(max_local_matrix_rel, numer_matrix / denom_matrix)
        dt2 = meta[0] * meta[0]
        local_nodes = record["nodes"]
        for a in range(4):
            if local_nodes[a] not in node_map:
                continue
            for ca in range(3):
                row = 3 * node_map[local_nodes[a]] + ca
                rloc = 3 * a + ca
                for bb in range(4):
                    if local_nodes[bb] not in node_map:
                        continue
                    for cb in range(3):
                        col = 3 * node_map[local_nodes[bb]] + cb
                        cloc = 3 * bb + cb
                        add_matrix(cached_entries, row, col, dt2 * kn[rloc][cloc])
                        add_matrix(independent_entries, row, col, dt2 * kr[rloc][cloc])
        element_count += 1

    keys = set(cached_entries) | set(independent_entries)
    col_sq = [0.0] * dimension
    ref_col_sq = [0.0] * dimension
    delta_sq = 0.0
    ref_matrix_sq = 0.0
    independent_matrix_values = {}
    cached_matrix_values = {}
    for key in keys:
        row, col = key
        nvalue = math.fsum(cached_entries.get(key, ()))
        rvalue = math.fsum(independent_entries.get(key, ()))
        if row == col:
            nvalue += mass[row // 3]
            rvalue += mass[row // 3]
        cached_matrix_values[key] = nvalue
        independent_matrix_values[key] = rvalue
        diff = nvalue - rvalue
        delta_sq += diff * diff
        ref_matrix_sq += rvalue * rvalue
        col_sq[col] += diff * diff
        ref_col_sq[col] += rvalue * rvalue
    col_rel = [math.sqrt(col_sq[i]) / math.sqrt(ref_col_sq[i]) for i in range(dimension)]
    assembled_max_col = max(col_rel)
    assembled_relative = math.sqrt(delta_sq) / math.sqrt(ref_matrix_sq)
    if element_count != 1920 or dimension != 2538:
        raise ValueError("unexpected assembled dimensions")

    # Compare the independent scalar assembly to the retained sparse matrices.
    first_a = npz_arrays(V1 / "independent_A_free_csr.npz")
    previous_cached = npz_arrays(BASE / "analysis/captured-base-state-cauchy-419-001/A_free_csr.npz")
    def sparse_relative_difference(entries, reference):
        err_sq = 0.0
        ref_sq = 0.0
        data = reference["data"]["values"]
        indices = reference["indices"]["values"]
        indptr = reference["indptr"]["values"]
        shape = tuple(reference["shape"]["values"])
        if shape != (dimension, dimension):
            raise ValueError(f"unexpected saved sparse shape: {shape}")
        represented = {}
        for row in range(shape[0]):
            start, end = indptr[row], indptr[row + 1]
            for idx in range(start, end):
                col, value = indices[idx], float(data[idx])
                represented[(row, col)] = value
        for row, col in set(entries) | set(represented):
            value = represented.get((row, col), 0.0)
            got = math.fsum(entries.get((row, col), ()))
            if row == col:
                got += mass[row // 3]
            err_sq += (got - value) ** 2
            ref_sq += value * value
        return math.sqrt(err_sq / ref_sq)

    # V1's independent A includes the same independent tangent plus mass;
    # the prior captured A uses cached tangent plus mass.
    ref_sparse_delta = sparse_relative_difference(independent_entries, first_a)
    cached_sparse_delta = sparse_relative_difference(cached_entries, previous_cached)
    published = json.loads((V2 / "report.json").read_text(encoding="utf-8"))
    metric_differences = {
        "max_element_column_relative_error": abs(max_local_column_rel - published["max_element_column_relative_error"]),
        "max_element_matrix_relative_error": abs(max_local_matrix_rel - published["max_element_matrix_relative_error"]),
        "max_assembled_column_relative_error": abs(assembled_max_col - published["max_assembled_column_relative_error"]),
        "assembled_matrix_relative_error": abs(assembled_relative - published["assembled_matrix_relative_error"]),
    }
    if any(not math.isfinite(v) or v > 1.0e-10 for v in metric_differences.values()):
        raise ValueError(f"independent scalar metrics do not reproduce 150-002: {metric_differences}")
    report = {
        "schema": "numi.cardboard.full_element_assembly_independent_review.v1",
        "inputs": {
            "verification_150_manifest_sha256": sha(V2 / "manifest.json"),
            "verification_150_report_sha256": sha(V2 / "report.json"),
            "source_150_manifest_sha256": sha(V1 / "manifest.json"),
            "tangent_147_manifest_sha256": sha(tangent_audit / "manifest.json"),
            "tangent_independent_review_manifest_sha256": sha(tangent_review / "manifest.json"),
            "capture_419_sha256": cap_hash,
        },
        "integrity": {
            "verification_150_declared_files_checked": len(v2m["files"]),
            "verification_150_declared_inputs_checked": len(v2m["inputs"]),
            "source_150_declared_files_checked": len(v1m["files"]),
            "source_150_declared_inputs_checked": len(v1m["inputs"]),
            "tangent_147_declared_files_checked": len(tangent_manifest["files"]),
            "tangent_147_declared_inputs_checked": len(tangent_manifest["inputs"]),
            "tangent_independent_review_artifacts_checked": len(review_files) + 1,
            "tangent_independent_review_bound_inputs_checked": len(trm["bound_inputs"]),
            "capture_record_counts": {"environment0_nodes": len(nodes), "environment0_tetrahedra": len(tets)},
            "native_cached_tangent_matrices_exactly_join_419": element_count,
            "capture_header_identity_checked": True,
            "capture_integrity_footer": capture_integrity,
            "tangent_independent_review_artifacts_checked": len(review_files) + 1,
            "warnings_or_numpy_fp_exceptions": 0,
        },
        "scalar_reconstruction": {
            "contraction_method": "independent scalar Python math.fsum loops; no BLAS, NumPy matmul/einsum, or source helper contraction",
            "local_matrix_count": element_count,
            "local_columns": element_count * 12,
            "max_element_column_relative_error": max_local_column_rel,
            "max_element_matrix_relative_error": max_local_matrix_rel,
            "max_element_column_absolute_error": max_local_column_abs,
            "assembled_dimension": dimension,
            "max_assembled_column_relative_error": assembled_max_col,
            "assembled_matrix_relative_error": assembled_relative,
            "independent_matrix_vs_first_report_relative_difference": ref_sparse_delta,
            "cached_matrix_vs_prior_capture_relative_difference": cached_sparse_delta,
            "differences_from_150_report": metric_differences,
            "stated_gates_reproduced": max_local_column_rel <= 1.0e-3 and assembled_max_col <= 0.01,
        },
        "classification": "Independent numerical reproduction of the analytic same-center element/frame/assembly audit only. No native all-column action sweep, finite-difference noise resolution, solver convergence or physical qualification.",
        "limitations": [
            "The input is one captured environment/control/Newton center and one source law state.",
            "No independent cohesive interfaces are represented; fixture is explicit finite elastic starch geometry with original substrate connectivity.",
            "The tangent source uses the separately audited FP64 projection and derivative arrays; this scalar review tests their element and assembly propagation.",
            "Signed-zero-only tetrahedron record differences versus JSON capture remain as described in the source audit; this review uses captured numerical fields, not an invented geometry qualification.",
        ],
        "reviewer_attempts": [
            {
                "attempt": "001",
                "outcome": "no numerical analysis",
                "cause": "the available system Python has neither NumPy nor SciPy; the successful review uses a small standard-library NPY/NPZ reader and scalar contractions",
                "log": "attempt-001-missing-numpy.log",
            },
            {
                "attempt": "002",
                "outcome": "reviewer-script caught its own indexing error before any report was written",
                "cause": "initial scalar B-matrix construction used the wrong Kronecker-delta index and produced zero reference columns; corrected to the source convention delta(a, displacement)",
                "log": "attempt-002-review-indexing-failure.log",
            },
            {
                "attempt": "003",
                "outcome": "reviewer script stopped before report output",
                "cause": "the 150-002 report stores metrics at the top level; the review initially looked for a nested scalar_reconstruction object",
                "log": "attempt-003-report-schema-adaptation.log",
            },
        ],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
