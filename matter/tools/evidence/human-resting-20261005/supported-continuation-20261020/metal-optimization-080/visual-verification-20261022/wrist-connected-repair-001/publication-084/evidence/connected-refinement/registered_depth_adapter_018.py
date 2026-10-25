"""Prolong registered O/I/D depth fields onto frozen 017 affine refinement."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path("/Users/n/Work/numi-human-connected-surface-069")
DEFAULTS = {
    "base_candidate": ROOT / "full-connected-clearance-150/run-017-point-constrained-soft-continuation/candidate.npz",
    "refined_candidate": ROOT / "full-connected-clearance-150/run-017-conforming-crossing-refinement/run-003/refined-candidate.npz",
    "field_asset": ROOT / "fine-layer-fields-127/baseline-coarse-to-detail-layer-fields.npz",
    "depth_owner": ROOT / "connected-implicit-layer-155/build_signed_registered_depth_fields_156.py",
    "layer_assembler": ROOT / "coupled-layer-trial-128/assemble_layers_128.py",
    "coarse_faces": ROOT / "publication-071/evidence/exact-transfer-075/run-001/source-preflight/accepted-coarse-faces.csv.gz",
}
PINNED_SHA256 = {
    "base_candidate": "3858c079054c4a87829fc554757104d74cb818073b1c8414acfd2d117fb3dbdd",
    "refined_candidate": "7a3a6fb208ad7e6af96fecf4bdcbab5aee8c3b340958a66b39676ea45a8176ec",
    "field_asset": "ff5d936d9c5f14df7e78a596134a03c5a6fd4f5c09a517203643ded28da8f5c1",
    "depth_owner": "3ffb8e68a5427f99d27acc6a9583de4897552166f459752a60112cbf97bfa0f8",
    "layer_assembler": "9d169a15458d272048a917c3e5cc3d91a45d5aabe0fd211746f972cb2390cde9",
    "coarse_faces": "8270d42f41289b6a077e33aae6ffb6af9ddfa40d90ceeb7a5bd93d4f8d4cb79a",
}

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def require_pinned_file(path, expected_sha256, label):
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    return actual

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def validate_prolongation(indptr, indices, weights, n_output, n_source):
    """Construct strict CSR for one nonnegative convex row per output node."""
    ip, ix = np.asarray(indptr), np.asarray(indices)
    w = np.asarray(weights, dtype=np.float64)
    if ip.ndim != 1 or len(ip) != n_output + 1 or not np.issubdtype(ip.dtype, np.integer):
        raise ValueError("prolongation indptr must be integer with n_output+1 entries")
    if ix.ndim != 1 or not np.issubdtype(ix.dtype, np.integer) or w.ndim != 1 or len(ix) != len(w):
        raise ValueError("prolongation indices/weights have invalid shapes or types")
    if ip[0] != 0 or ip[-1] != len(ix) or np.any(ip[1:] < ip[:-1]):
        raise ValueError("prolongation indptr is not a valid CSR pointer")
    if len(ix) == 0 or np.any(ix < 0) or np.any(ix >= n_source):
        raise ValueError("prolongation source row is out of range")
    if not np.isfinite(w).all() or np.any(w < 0.0):
        raise ValueError("prolongation weights must be finite and nonnegative")
    B = csr_matrix((w, ix.astype(np.int64, copy=False), ip.astype(np.int64, copy=False)),
                   shape=(n_output, n_source), dtype=np.float64)
    B.sum_duplicates()
    sums = np.asarray(B.sum(axis=1)).ravel()
    if np.any(B.indptr[1:] == B.indptr[:-1]) or not np.allclose(sums, 1.0, rtol=0.0, atol=2e-14):
        raise ValueError("each prolongation row must be nonempty and convex")
    return B

def prolong_depth_fields(source_depths, B):
    """Apply the source CSR to rows [skin depth, fat depth]."""
    H = np.asarray(source_depths)
    if H.ndim != 2 or H.shape != (B.shape[1], 2):
        raise ValueError("source depths must have shape (n_source, 2)")
    if not np.issubdtype(H.dtype, np.number) or not np.isfinite(H).all() or np.any(H <= 0.0):
        raise ValueError("source skin/fat depths must be finite and positive")
    result = np.asarray(B @ H.astype(np.float64, copy=False), dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError("prolonged skin/fat depths must remain finite and positive")
    return result

def load_registered_depths(base_candidate_path=DEFAULTS["base_candidate"],
                            refined_candidate_path=DEFAULTS["refined_candidate"],
                            field_asset_path=DEFAULTS["field_asset"],
                            depth_owner_path=DEFAULTS["depth_owner"],
                            layer_assembler_path=DEFAULTS["layer_assembler"],
                            coarse_faces_path=DEFAULTS["coarse_faces"],
                            expected_sha256=PINNED_SHA256):
    """Use owner 156 for source rows, then the refinement's exact sparse B."""
    paths = {
        "base_candidate": Path(base_candidate_path),
        "refined_candidate": Path(refined_candidate_path),
        "field_asset": Path(field_asset_path),
        "depth_owner": Path(depth_owner_path),
        "layer_assembler": Path(layer_assembler_path),
        "coarse_faces": Path(coarse_faces_path),
    }
    pins = {}
    for key, path in paths.items():
        if key not in expected_sha256:
            raise ValueError(f"no expected SHA-256 supplied for {key}")
        pins[key] = require_pinned_file(path, expected_sha256[key], key)

    base = np.load(paths["base_candidate"], allow_pickle=False)
    refined = np.load(paths["refined_candidate"], allow_pickle=False)
    fields = np.load(paths["field_asset"], allow_pickle=False)
    base_points = np.asarray(base["points_world_m"])
    out_points = np.asarray(refined["points_world_m"])
    if base_points.ndim != 2 or base_points.shape[1] != 3 or out_points.ndim != 2 or out_points.shape[1] != 3:
        raise ValueError("candidate point arrays must be N by 3")
    n_source, n_output = len(base_points), len(out_points)
    if base_points.dtype != np.float32 or out_points.dtype != np.float32:
        raise ValueError("the depth owner inputs require Float32 candidate points")
    if n_output < n_source or not np.array_equal(out_points[:n_source], base_points):
        raise ValueError("refinement does not preserve the original Float32 point prefix")
    for key in ("node_kind", "node_source_row"):
        if key not in base.files or key not in refined.files or not np.array_equal(refined[key][:n_source], base[key]):
            raise ValueError(f"refinement changed original {key} prefix")

    B = validate_prolongation(refined["source_prolongation_indptr"],
                              refined["source_prolongation_node_rows"],
                              refined["source_prolongation_weights"], n_output, n_source)
    for row in range(n_source):
        a, b = B.indptr[row:row+2]
        if b-a != 1 or B.indices[a] != row or B.data[a] != 1.0:
            raise ValueError(f"prolongation prefix row {row} is not identity")
    projected_points = np.column_stack([B @ base_points[:, k].astype(np.float64) for k in range(3)])
    if not np.array_equal(projected_points.astype(np.float32), out_points):
        err = float(np.max(np.abs(projected_points - out_points.astype(np.float64))))
        raise ValueError(f"prolongation does not reconstruct Float32 points; max error={err}")

    owner = _load_module("registered_depth_owner_156", paths["depth_owner"])
    assembler = _load_module("registered_layer_assembler_128", paths["layer_assembler"])
    coarse_faces, face_csv_raw_sha = assembler.face_csv(paths["coarse_faces"])
    if face_csv_raw_sha != "7311267392275649abeb3ecaf6a5ca43eb4d67cf2a9b7f13a9c692587c36b1e2":
        raise ValueError("decompressed coarse face table SHA-256 mismatch")
    source_depths = owner.registered_depths(base, fields, coarse_faces)
    if source_depths.shape != (n_source, 2):
        raise ValueError("registered_depths owner returned an unexpected source shape")
    refined_depths = prolong_depth_fields(source_depths, B)
    if not np.array_equal(refined_depths[:n_source], source_depths):
        raise ValueError("identity-prefix depths are not bitwise unchanged")
    report = {
        "status": "REGISTERED_DEPTHS_PROLONGED",
        "operation": "B @ owner156_registered_depths",
        "base_node_count": int(n_source), "refined_node_count": int(n_output),
        "prolongation_shape": [int(n_output), int(n_source)],
        "prolongation_nnz": int(B.nnz), "generated_node_count": int(n_output-n_source),
        "generated_node_kind_counts": {
            str(int(k)): int(np.count_nonzero(refined["refinement_node_kind"][n_source:] == k))
            for k in np.unique(refined["refinement_node_kind"][n_source:])
        },
        "convex_weight_sum_min": float(np.min(np.asarray(B.sum(axis=1)).ravel())),
        "convex_weight_sum_max": float(np.max(np.asarray(B.sum(axis=1)).ravel())),
        "source_depth_min_m": source_depths.min(axis=0).tolist(),
        "source_depth_max_m": source_depths.max(axis=0).tolist(),
        "refined_depth_min_m": refined_depths.min(axis=0).tolist(),
        "refined_depth_max_m": refined_depths.max(axis=0).tolist(),
        "prefix_depth_bitwise_equal": True,
        "prolongation_reconstructs_float32_points_bitwise": True,
        "input_sha256": pins, "coarse_face_csv_raw_sha256": face_csv_raw_sha,
        "physical_steps": 0, "geometry_admitted": False,
        "limitations": [
            "Generated-node fields are convex sparse interpolation of registered source-node fields.",
            "This adaptation does not qualify volume, bone clearance, mass, or tissue physics."
        ],
    }
    return refined_depths, report

def write_depth_asset(output_dir, **kwargs):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    depths, report = load_registered_depths(**kwargs)
    report["elapsed_seconds"] = time.monotonic() - start
    asset = out / "registered-depths-refined.npz"
    np.savez_compressed(asset, registered_depths_m=depths,
                        hskin_euclidean_m=depths[:, 0], hfat_euclidean_m=depths[:, 1],
                        output_node_row=np.arange(len(depths), dtype=np.int32))
    report["output_sha256"] = sha256_file(asset)
    report["output_path"] = str(asset)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", type=Path)
    args = ap.parse_args()
    print(json.dumps(write_depth_asset(args.output_dir), indent=2))

if __name__ == "__main__":
    main()

