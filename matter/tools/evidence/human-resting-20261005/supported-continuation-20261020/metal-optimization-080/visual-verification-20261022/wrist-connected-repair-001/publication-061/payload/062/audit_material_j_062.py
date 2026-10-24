from pathlib import Path
import csv, hashlib, json
import numpy as np

E = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
OUT = E / "root-reference-relative-j-062"
REFERENCE = OUT / "source-preflight/identity-reference-points.csv"
BASELINE = E / "native-candidate020-all-bank-containment-035/source-preflight-003/initial-skin-fem-nodes.csv"
CANDIDATE = E / "bone-clearance-fit-037-componentwise-035affine-002/candidate.npz"
MATERIALS = {
    "skin": Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/materials/human_skin_dermis_frequency_specific_effective.nmatter"),
    "fat": Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/materials/human_subcutaneous_adipose_low_strain_effective.nmatter"),
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_xyz(rows):
    return np.asarray([[float(row[k]) for k in ("x_m", "y_m", "z_m")] for row in rows], dtype=np.float64)

def descriptor_bounds(path):
    line = next(line for line in path.read_text().splitlines() if "limits {" in line)
    values = {}
    for key in ("minimum_J", "maximum_J"):
        values[key] = float(line.split(key + " =", 1)[1].split(";", 1)[0].strip())
    return values

ref_rows = list(csv.DictReader(REFERENCE.open()))
assert len(ref_rows) == 7233
assert [int(row["local_volume_node"]) for row in ref_rows] == list(range(7233))
reference = read_xyz(ref_rows)
base_rows = list(csv.DictReader(BASELINE.open()))
base_rows.sort(key=lambda row: (int(row["layer"]), int(row["row"])))
assert len(base_rows) == len(reference)
baseline = read_xyz(base_rows)
with np.load(CANDIDATE) as arrays:
    current = arrays["points_world_f32"].astype(np.float32).astype(np.float64)
    tetrahedra = arrays["tetrahedra"].astype(np.int64)
    material_ids = arrays["material_ids"].astype(np.int64)
    global_node_ids = arrays["global_node_ids"].astype(np.int64)
assert current.shape == reference.shape == (7233, 3)
assert tetrahedra.shape == (28632, 4)
assert np.array_equal(global_node_ids, np.arange(1666, 8899))
material_values, material_counts = np.unique(material_ids, return_counts=True)
assert np.array_equal(material_values, np.asarray([1, 2]))
assert np.array_equal(material_counts, np.asarray([14316, 14316]))

# Local rows are a repeated O/I/D triplet per column. This proves which band
# each generated material occupies without assuming material IDs start at zero.
node_layer = np.arange(7233) // 2411
layer_patterns = {}
for material_id in (1, 2):
    pattern = np.sort(node_layer[tetrahedra[material_ids == material_id]], axis=1)
    unique, counts = np.unique(pattern, axis=0, return_counts=True)
    layer_patterns[str(material_id)] = {"O_I_D_layer_patterns": unique.tolist(), "tet_counts": counts.tolist()}

bounds = {name: descriptor_bounds(path) for name, path in MATERIALS.items()}
assert bounds["skin"] == bounds["fat"] == {"minimum_J": 0.35, "maximum_J": 2.5}
D_reference = (reference[tetrahedra[:, 1:]] - reference[tetrahedra[:, :1]]).transpose(0, 2, 1)
inv_D_reference = np.linalg.inv(D_reference)
results = {}
for case, points in (("baseline_native_current", baseline), ("candidate_037_002", current)):
    D_current = (points[tetrahedra[:, 1:]] - points[tetrahedra[:, :1]]).transpose(0, 2, 1)
    J = np.linalg.det(D_current @ inv_D_reference)
    results[case] = {}
    for material_id, name in ((1, "skin"), (2, "fat")):
        rows = np.flatnonzero(material_ids == material_id)
        material_J = J[rows]
        lower = bounds[name]["minimum_J"]
        upper = bounds[name]["maximum_J"]
        results[case][name] = {
            "material_id": material_id,
            "tet_count": int(len(rows)),
            "J_quantiles_0_1_50_99_100": np.quantile(material_J, [0, .01, .5, .99, 1]).tolist(),
            "below_minimum_J_count": int(np.count_nonzero(material_J < lower)),
            "above_maximum_J_count": int(np.count_nonzero(material_J > upper)),
            "inverted_count": int(np.count_nonzero(material_J <= 0)),
            "min_J": float(material_J.min()),
            "max_J": float(material_J.max()),
            "tetrahedron_rows_in_combined_candidate_array": rows[np.argsort(material_J)[:8]].tolist(),
            "overmaximum_tetrahedron_rows_in_combined_candidate_array": rows[np.argsort(material_J)[-8:][::-1]].tolist(),
        }

report = {
    "status": "REFERENCE_RELATIVE_MATERIAL_SPECIFIC_DIAGNOSTIC_NOT_ADMISSION",
    "reference_points_sha256": sha(REFERENCE),
    "baseline_current_csv_sha256": sha(BASELINE),
    "candidate_sha256": sha(CANDIDATE),
    "materials": {name: {**bounds[name], "path": str(path), "sha256": sha(path)} for name, path in MATERIALS.items()},
    "nodes": int(len(current)),
    "tetrahedra": int(len(tetrahedra)),
    "tetrahedron_row_space": "0-based row index into the combined candidate tetrahedra/material_ids arrays; not a native global FEM cell ID",
    "material_id_counts": {str(int(k)): int(v) for k, v in zip(material_values, material_counts)},
    "material_role_proof_from_layer_rows": layer_patterns,
    "material_id_mapping": {"1": "skin O/I band", "2": "fat I/D band"},
    "reference_vs_native_current_max_m": float(np.linalg.norm(reference - baseline, axis=1).max()),
    "results": results,
    "limitations": [
        "Evaluates emitted Float32 candidate/current points in Float64 against owner reference point geometry; not a bitwise replay of compiled Metal inverse-rest arithmetic.",
        "Checks elementwise J against descriptor bounds only; it does not establish contact, anatomy, constitutive history, or accepted native runtime behavior.",
        "Candidate 037-002 is rejected; no native admission is claimed.",
    ],
}
report_path = OUT / "material-specific-reference-J-062.json"
report_path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"script": str(Path(__file__)), "script_sha256": sha(Path(__file__)), "report": str(report_path), "report_sha256": sha(report_path), "candidate_sha256": sha(CANDIDATE), "results": results, "role_patterns": layer_patterns}, indent=2))
