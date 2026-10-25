from pathlib import Path
import sys, importlib.util, json, hashlib, numpy as np
D = Path("/Users/numi/Work/numi-human-connected-surface-069")
source = D / "coupled-layer-trial-128/assemble_layers_128.py"
spec = importlib.util.spec_from_file_location("assembler", source)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
C = D / "connected-source-unfold-146/run-002-corrected-metadata/candidate.npz"
BASE = D / "fine-layer-fields-127/baseline-coarse-to-detail-layer-fields.npz"
CAP115 = D / "fine-layer-fields-127/capacity-115-coarse-to-detail-layer-fields.npz"
OUT = D / "nonmatching-layer-comparison-142/run-005-full146-baseline084"
OUT.mkdir(exist_ok=False, parents=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
z = np.load(C); a = np.load(BASE); b = np.load(CAP115)
P = z["points_world_m"]; F = z["source_O_faces"][:, [0, 2, 1]]; n = len(P)
# Explicitly confirm that the baseline and 115 fields refer to the same coarse column rows.
for key in ("coarse_column_ids", "coarse_source_global_node_ids_O", "coarse_source_global_node_ids_I", "coarse_source_global_node_ids_D", "coarse_donor_body"):
    if not np.array_equal(a[key], b[key]):
        raise ValueError(f"coarse source row identity mismatch: {key}")
I = a["coarse_I_reference_m"]; Dp = a["coarse_D_reference_m"]; nc = len(I)
if I.shape != (2411, 3) or Dp.shape != (2411, 3): raise ValueError("unexpected baseline-084 layer shape")
Fc, _ = m.face_csv(m.DEFAULT_FACE_CSV_GZ); Fc = Fc[:, [0, 2, 1]]
if Fc.ndim != 2 or Fc.shape[1] != 3 or (Fc.size and (Fc.min() < 0 or Fc.max() >= nc)): raise ValueError("coarse face/node domain mismatch")
X = np.vstack((P, I, Dp)); kind = z["node_kind"]; sr = z["node_source_row"]
caps = []
for edge, occ in m.edges_of_faces(F).items():
    if len(occ) == 1:
        _, u, v = occ[0]
        if kind[u] != 0 or kind[v] != 0 or sr[u] < 0 or sr[v] < 0:
            raise ValueError(f"outer boundary is not exact coarse ancestry: {edge}")
        ui = n + int(sr[u]); vi = n + int(sr[v])
        caps.extend(((v, u, ui), (v, ui, vi)))
caps = np.asarray(caps, dtype=np.int64).reshape((-1, 3))
skin = np.vstack((F, Fc[:, [0, 2, 1]] + n, caps))
boundary = np.asarray([(v[0][1], v[0][2]) for v in m.edges_of_faces(Fc).values() if len(v) == 1], dtype=np.int64).reshape((-1, 2))
fat = m.shell_faces(Fc, boundary, nc, n, n + nc)
report = {
    "candidate_input": str(C), "candidate_sha256": sha(C),
    "baseline084_fields_input": str(BASE), "baseline084_fields_sha256": sha(BASE),
    "capacity115_fields_comparison_input": str(CAP115), "capacity115_fields_sha256": sha(CAP115),
    "assembler_source": str(source), "assembler_sha256": sha(source),
    "builder_source": str(Path(__file__).resolve()), "builder_sha256": sha(Path(__file__).resolve()),
    "method": "full O from corrected run-002; independent baseline-084 coarse I/D fields; exact coarse-row identity crosscheck against 115 field export; no geometry optimization",
    "accepted_physical_steps": 0, "geometry_admitted": False, "bone_clearance_audited": False,
    "point_count": int(len(X)), "outer_node_count": int(n), "inner_node_count": int(nc),
    "baseline084_skin_volume_m3": m.signed_volume(X, skin),
    "baseline084_fat_volume_m3": m.signed_volume(X, fat),
    "max_I_shift_vs_capacity115_m": float(np.linalg.norm(I - b["capacity115_coarse_I_reference_m"], axis=1).max()),
    "max_D_shift_vs_capacity115_m": float(np.linalg.norm(Dp - b["capacity115_coarse_D_reference_m"], axis=1).max()),
    "skin_topology": m.shell_topology(skin), "fat_topology": m.shell_topology(fat), "audits": {}
}
for name, faces in (("O", F), ("I", Fc + n), ("D", Fc + n + nc), ("skin_shell", skin), ("fat_shell", fat)):
    result = m.exact_self_audit(X, faces)
    np.save(OUT / (name + "-pairs.npy"), result.pop("pairs_array"))
    report["audits"][name] = result
np.savez_compressed(OUT / "candidate.npz", points_world_m=X, outer_faces=F, inner_faces=Fc+n, deep_faces=Fc+n+nc,
                    skin_shell_faces=skin, fat_shell_faces=fat, outer_node_kind=kind,
                    outer_node_source_row=sr, inner_coarse_source_row=np.arange(nc), deep_coarse_source_row=np.arange(nc))
report["output_candidate_sha256"] = sha(OUT / "candidate.npz")
(OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
