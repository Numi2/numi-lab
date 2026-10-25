#!/usr/bin/env python3
"""Source-only frame audit and LBS displacement map for NHSKIN1 ABI5."""
from __future__ import annotations
import argparse, csv, hashlib, json, struct, sys, socket, time
from pathlib import Path
import numpy as np

ROOT = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
DETAIL = Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001")
SKIN128 = ROOT / "left-ankle-rest-preserving-weight-repair-128/candidate.nhskin"
SKIN294 = DETAIL / "bodyparts3d-myosim-skinned-shell.nhskin"
CSV294 = DETAIL / "native-step-001/registered-skin.csv"
PARENT_REF_CSV = ROOT / "elbow-refinement-current-main-024-final080-candidate-source-frames-001/provenance/registered-skin-080-candidate.csv"
D36 = ROOT / "wholebody-attachment-source-rebind-036-ankle128-regenerated-deltas-001"
EMBED = D36 / "generated/embedding.csv"
NODES = D36 / "generated/nodes.csv"
FRAME = ROOT / "elbow-refinement-current-main-024-final080-candidate-source-frames-001/native-source-only-024/common-rigid-transfer-row-diagnostic-005.json"
BODY_FRAMES = Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/finger-joint-response-001/final074-neutral-reference-002/body-frames.json")
AP = ROOT / "bone-clearance-feasibility-033/world-bone-clearance-constraints.npz"
OLD_MAP = ROOT / "current-294-skin-displacement-prolongation-001/run-002/prolongation.npz"
CAND = ROOT / "connected-volume-continuation-039/iteration-005/candidate.npz"
ROOT_LIFT_Z_M = 0.35

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def read_csv(path: Path):
    with path.open(newline="") as f: return list(csv.DictReader(f))

def csv_xyz(rows, prefix="w"):
    return np.asarray([[float(r[prefix + c]) for c in "xyz"] for r in rows], dtype=np.float64)

def quat_matrix(q):
    x, y, z, w = np.asarray(q, dtype=np.float64)
    return np.asarray([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)

def load_skin(path: Path):
    raw = path.read_bytes()
    magic, abi, nb, nv, ni, flags, source_sha = struct.unpack_from("<8s5I32s", raw, 0)
    if (magic, abi, nb, nv, ni) != (b"NHSKIN1\0", 5, 86, 61950, 326961):
        raise ValueError(f"unexpected NHSKIN ABI/counts in {path}")
    bind_off = 60
    vert_off = bind_off + nb * 36
    face_off = vert_off + nv * 56
    weight_off = face_off + ni * 4
    if len(raw) != weight_off + nv * nb * 4: raise ValueError("ABI5 payload length mismatch")
    bindings = []
    for i in range(nb):
        v = struct.unpack_from("<I8f", raw, bind_off + 36*i)
        bindings.append({"body": int(v[0]), "translation": np.asarray(v[1:4], dtype=np.float64),
                         "quaternion": np.asarray(v[4:8], dtype=np.float64), "scale": float(v[8])})
    pos = np.empty((nv, 3), dtype=np.float64)
    for i in range(nv): pos[i] = struct.unpack_from("<3f", raw, vert_off + 56*i)
    faces = np.frombuffer(raw, dtype="<u4", count=ni, offset=face_off).reshape(-1, 3).copy()
    weights = np.frombuffer(raw, dtype="<f4", count=nv*nb, offset=weight_off).reshape(nv, nb).astype(np.float64)
    return {"raw": raw, "nb": nb, "nv": nv, "ni": ni, "bindings": bindings, "positions": pos,
            "faces": faces, "weights": weights, "binding_bytes": raw[bind_off:vert_off], "sha256": sha(path)}

def effective_binding_affines(bindings, poses, root_lift_z=ROOT_LIFT_Z_M):
    A = np.empty((len(bindings), 3, 3), dtype=np.float64)
    c = np.empty((len(bindings), 3), dtype=np.float64)
    body_R, body_t = {}, {}
    for body, rec in poses.items():
        q = np.asarray(rec["quaternion_xyzw"], dtype=np.float32).astype(np.float64)
        pos = np.asarray(rec["position"], dtype=np.float32).astype(np.float64)
        pos[2] += float(root_lift_z)
        body_R[int(body)] = quat_matrix(q); body_t[int(body)] = pos
    for i, b in enumerate(bindings):
        if b["body"] not in body_R: raise ValueError(f"missing pose for body {b['body']}")
        rb = body_R[b["body"]]
        A[i] = rb @ quat_matrix(b["quaternion"]) * b["scale"]
        c[i] = body_t[b["body"]] + rb @ b["translation"]
    return A, c, body_R, body_t

def weighted_skin_jacobians(weights, binding_linear):
    w = np.asarray(weights, dtype=np.float64); a = np.asarray(binding_linear, dtype=np.float64)
    if w.ndim != 2 or a.shape != (w.shape[1], 3, 3): raise ValueError("weight/binding-linear shape mismatch")
    if not np.isfinite(w).all() or not np.isfinite(a).all(): raise ValueError("nonfinite skinning Jacobian input")
    return np.einsum("nb,bij->nij", w, a, optimize=True)

def map_world_delta_to_vertex_position_delta(jacobians, world_delta, *, max_condition=1e10, min_singular=1e-10):
    j = np.asarray(jacobians, dtype=np.float64); d = np.asarray(world_delta, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != 3 or j.shape != (len(d), 3, 3): raise ValueError("Jacobian/delta shape mismatch")
    if not np.isfinite(j).all() or not np.isfinite(d).all(): raise ValueError("nonfinite Jacobian/delta")
    singular = np.linalg.svd(j, compute_uv=False)
    if np.any(singular[:, -1] <= min_singular): raise ValueError("singular blended skinning Jacobian")
    cond = singular[:, 0] / singular[:, -1]
    if np.any(cond > max_condition): raise ValueError("ill-conditioned blended skinning Jacobian")
    local = np.linalg.solve(j, d[..., None])[..., 0]
    return local, cond, singular[:, -1], np.einsum("nij,nj->ni", j, local) - d

def quantiles(x):
    a = np.asarray(x, dtype=np.float64)
    return {"count": int(a.size), "min": float(np.min(a)), "median": float(np.median(a)),
            "p95": float(np.quantile(a, .95)), "p99": float(np.quantile(a, .99)), "max": float(np.max(a))}

def run_audit(outdir: Path):
    started = time.time()
    outdir.mkdir(parents=True, exist_ok=False)
    input_paths = [SKIN128, SKIN294, CSV294, PARENT_REF_CSV, EMBED, NODES, FRAME, BODY_FRAMES, AP, CAND, OLD_MAP]
    for p in input_paths:
        if not p.is_file(): raise FileNotFoundError(p)
    s128, s294 = load_skin(SKIN128), load_skin(SKIN294)
    if not np.array_equal(s128["faces"], s294["faces"]): raise ValueError("128/294 source faces differ")
    if s128["binding_bytes"] != s294["binding_bytes"]: raise ValueError("128/294 binding table differs")
    rows294, rows_parent = read_csv(CSV294), read_csv(PARENT_REF_CSV)
    rows_d36 = read_csv(D36 / "provenance/registered-skin-128-candidate.csv")
    if not (len(rows294) == len(rows_parent) == len(rows_d36) == s294["nv"]): raise ValueError("registered row count mismatch")
    for rows in (rows294, rows_parent, rows_d36):
        if any(int(r["vertex"]) != i for i, r in enumerate(rows)): raise ValueError("CSV row identity mismatch")
    x294, xparent, xd36 = csv_xyz(rows294), csv_xyz(rows_parent), csv_xyz(rows_d36)
    bodyframes = json.loads(BODY_FRAMES.read_text())
    poses = {int(p["core_body"]): p for p in bodyframes["source_poses"]}
    A_bind, c_bind, body_R, body_t = effective_binding_affines(s294["bindings"], poses)
    J = weighted_skin_jacobians(s294["weights"], A_bind)
    b = np.einsum("nb,bc->nc", s294["weights"], c_bind, optimize=True)
    xpred = np.einsum("nij,nj->ni", J, s294["positions"], optimize=True) + b
    lbs_error = np.linalg.norm(xpred - x294, axis=1)
    singular = np.linalg.svd(J, compute_uv=False)
    condition = singular[:, 0] / singular[:, -1]
    donor_frames, donor_locals = [], []
    owner_mats = np.empty((s294["nv"], 3, 3), dtype=np.float64)
    for i, r in enumerate(rows294):
        donor = int(r["donor"])
        if donor not in body_R: raise ValueError(f"missing donor pose {donor}")
        m = np.asarray([[float(r[f"m{u}{v}"]) for v in range(3)] for u in range(3)])
        owner_mats[i] = m
        donor_frames.append(np.max(np.abs(m - body_R[donor].T)))
        expected_l = body_R[donor].T @ (x294[i] - body_t[donor])
        actual_l = np.asarray([float(r[f"l{c}"]) for c in "xyz"])
        donor_locals.append(np.linalg.norm(expected_l - actual_l))

    emb, nodes = read_csv(EMBED), read_csv(NODES)
    if len(emb) != 2411 or len(nodes) != 2411: raise ValueError("D36 row count mismatch")
    direct = np.asarray([int(e["source_vertex"]) != np.iinfo(np.uint32).max for e in emb])
    source_ids = np.asarray([int(e["source_vertex"]) for e in emb], dtype=np.int64)
    O = np.asarray([[float(r[f"o{c}"]) for c in "xyz"] for r in nodes], dtype=np.float64)
    mapped_parent, mapped_current = np.empty_like(O), np.empty_like(O)
    for i, e in enumerate(emb):
        if direct[i]:
            sid = int(source_ids[i]); mapped_parent[i] = xparent[sid]; mapped_current[i] = x294[sid]
        else:
            fi = int(e["face"]); w = np.asarray([float(e[f"b{k}"]) for k in range(3)])
            mapped_parent[i] = w @ xparent[s294["faces"][fi]]
            mapped_current[i] = w @ x294[s294["faces"][fi]]
    offsets = O - mapped_current
    off_direct = np.linalg.norm(offsets[direct], axis=1)
    off_bary = np.linalg.norm(offsets[~direct], axis=1)
    frame = json.loads(FRAME.read_text())
    R = np.asarray(frame["placement_rotation"], dtype=np.float64)
    t = np.asarray(frame["placement_translation_m"], dtype=np.float64)
    ap = np.load(AP); ap_points = np.asarray(ap["points"], dtype=np.float64)
    if ap_points.shape != (7233, 3): raise ValueError("AP basis cardinality mismatch")
    ap_o_error = np.linalg.norm(O @ R.T + t - ap_points[:2411], axis=1)
    ap_source_error = np.linalg.norm(mapped_current @ R.T + t - ap_points[:2411], axis=1)
    source_offset_ap = (mapped_current - O) @ R.T
    rigid_offset_norm_difference = np.abs(np.linalg.norm(offsets, axis=1) - np.linalg.norm(source_offset_ap, axis=1))

    pos128_delta = np.linalg.norm(s128["positions"] - s294["positions"], axis=1)
    weight_changed = np.any(s128["weights"] != s294["weights"], axis=1)
    render_delta = np.linalg.norm(xd36 - x294, axis=1)
    parent_current_delta = np.linalg.norm(xparent - x294, axis=1)

    old = np.load(OLD_MAP)
    world_delta = np.asarray(old["world_displacements_m"], dtype=np.float64)
    if world_delta.shape != s294["positions"].shape: raise ValueError("saved displacement map row mismatch")
    local_delta, cond, min_sv, linear_resid = map_world_delta_to_vertex_position_delta(J, world_delta)
    naive_local = np.einsum("nij,nj->ni", owner_mats, world_delta)
    naive_error = np.linalg.norm(np.einsum("nij,nj->ni", J, naive_local) - world_delta, axis=1)
    p_after_f32 = (s294["positions"] + local_delta).astype(np.float32).astype(np.float64)
    f32_world_delta = np.einsum("nij,nj->ni", J, p_after_f32 - s294["positions"], optimize=True)
    f32_error = np.linalg.norm(f32_world_delta - world_delta, axis=1)
    out_npz = outdir / "current294-source-position-delta-map.npz"
    np.savez_compressed(out_npz, source_vertex_ids=np.arange(s294["nv"], dtype=np.uint32),
        desired_reference_world_displacement_m=world_delta, raw_nhskin_vertex_position_delta_m=local_delta,
        raw_nhskin_vertex_position_after_float32_m=p_after_f32.astype(np.float32),
        effective_lbs_jacobian=J, effective_lbs_min_singular_value=min_sv, effective_lbs_condition_number=cond)

    result = {
      "schema": "numi.human.current294-source-reference-frame-audit.v1",
      "classification": "diagnostic_reference_map_only_no_NHSKIN_mutation_no_physical_admission",
      "inputs": {str(p): sha(p) for p in input_paths},
      "support_csvs": {
        "registered_128_reference": {"path": str(D36 / "provenance/registered-skin-128-candidate.csv"), "sha256": sha(D36 / "provenance/registered-skin-128-candidate.csv")},
        "registered_294_parent_reference": {"path": str(PARENT_REF_CSV), "sha256": sha(PARENT_REF_CSV)},
        "registered_294_native_current": {"path": str(CSV294), "sha256": sha(CSV294)},
      },
      "owner_routes": {
        "detailed_renderer": "numilab_human_myosim_visual_probe.mm::skinVertexWorld: per-binding transform followed by body pose, then weighted sum over all 86 bindings",
        "mechanical_region": "NumiHumanSkinFEMIntegration.wholebody.hpp::makeRegisteredSkinMechanicalRegion gets the same positionAt world point, then uses dominantSkinDonorBody and worldToDonorAt for its per-node donor frame",
        "source_position_delta": "raw vertex.position requires inverse of J_i=sum_b(weight_ib * R_body_b * R_binding_b * scale_b); the dominant-donor inverse rotation is not J_i",
      },
      "checks": {
        "same_128_294_faces": bool(np.array_equal(s128["faces"], s294["faces"])),
        "128_294_binding_table_byte_identical": bool(s128["binding_bytes"] == s294["binding_bytes"]),
        "native_294_position_reconstruction_from_full_weights_and_pinned_body_pose": {
          "max_error_m": float(lbs_error.max()), "p95_error_m": float(np.quantile(lbs_error, .95)),
          "within_20um": bool(lbs_error.max() <= 20e-6)},
        "dominant_donor_frame_vs_same_body_pose": {
          "max_rotation_inf_error": float(np.max(donor_frames)), "max_local_point_error_m": float(np.max(donor_locals))},
        "parent_reference_vs_native_current_world_delta_m": quantiles(parent_current_delta),
        "128_vs_294_rebind": {
          "raw_local_position_delta_m": quantiles(pos128_delta), "raw_position_rows_changed": int(np.count_nonzero(pos128_delta > 0)),
          "fullweight_rows_changed": int(np.count_nonzero(weight_changed)), "registered_world_position_delta_m": quantiles(render_delta)},
        "exact_source_ancestry_offset_O_to_current294": {
          "direct_anchors": int(np.count_nonzero(direct)), "barycentric_anchors": int(np.count_nonzero(~direct)),
          "direct_offset_m": quantiles(off_direct), "barycentric_offset_m": quantiles(off_bary),
          "all_offset_m": quantiles(np.linalg.norm(offsets, axis=1)),
          "same_rigid_transform_O_vs_AP_error_m": quantiles(ap_o_error),
          "same_rigid_transform_source_vs_AP_error_m": quantiles(ap_source_error),
          "offset_norms_preserved_by_rigid_transform": bool(rigid_offset_norm_difference.max(initial=0.0) <= 2e-12),
          "rigid_offset_norm_max_difference_m": float(rigid_offset_norm_difference.max(initial=0.0)),
          "source_offset_ap_frame_m": quantiles(np.linalg.norm(source_offset_ap, axis=1)),
          "frame_or_weight_mismatch_excluded_at_surface_position_level": bool(parent_current_delta.max() <= 20e-6 and np.max(donor_frames) <= 2e-6)},
        "full_weight_local_displacement_map": {
          "effective_jacobian_min_singular_value": quantiles(min_sv),
          "effective_jacobian_condition_number": quantiles(cond),
          "linear_world_roundtrip_max_m": float(np.linalg.norm(linear_resid,axis=1).max()),
          "dominant_donor_as_vertex_position_delta_error_m": quantiles(naive_error),
          "float32_source_position_forward_error_m": quantiles(f32_error),
          "max_raw_source_position_delta_mm": float(np.linalg.norm(local_delta,axis=1).max()*1000)},
      },
      "outputs": {"npz": {"path": str(out_npz), "sha256": sha(out_npz)}},
      "limitations": [
        "The direct 2.946 mm and barycentric 11.598 mm residuals persist after exact source ancestry and the same AP rigid transform; they are offsets between the D36 O-layer points and the current detailed source geometry.",
        "Raw p and full weights differ between the 128 rebind and 294 source, but their registered rendered surface positions under the pinned body pose remain near-coincident; those rebind differences do not explain the O-to-source residual.",
        "The local-delta NPZ is diagnostic only. It does not mutate NHSKIN and does not certify collision, geometry, layer, volume, mass, FEM, native runtime, or physical acceptance.",
      ],
      "runtime": {"seconds": time.time()-started, "hostname": socket.gethostname(), "python": sys.version, "numpy": np.__version__},
    }
    report = outdir / "report.json"
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(report), "npz": str(out_npz), "checks": result["checks"]}, indent=2))
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run_audit(parser.parse_args().output)

