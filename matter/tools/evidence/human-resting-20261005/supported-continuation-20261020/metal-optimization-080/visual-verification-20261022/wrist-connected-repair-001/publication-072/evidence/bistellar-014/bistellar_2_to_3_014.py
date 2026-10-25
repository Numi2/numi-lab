#!/usr/bin/env python3
"""Bounded same-material 2->3 tetrahedral flips around current-invalid cells.

Coordinates and surface faces are immutable. Replacements must preserve the exact
oriented boundary of the two-cell cavity using positive reference orientation.
"""
from __future__ import annotations
import hashlib, itertools, json, math, socket, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parent / "reference-fit-layer-counterfactual-001"))
from reference_fit_layer_counterfactual import tet_metrics

SOURCE = ROOT / "retessellation-001/volume-cavity-001/oid-cavity-replacement.npz"
OUT = HERE / "run-001"
Q_FLOOR = 1e-5
J_BOUNDS = (0.35, 2.5)
MAX_SWEEPS = 4
MAX_SECONDS = 600.0
EPS = 1e-12


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dets(points, tets):
    x = np.asarray(points, dtype=np.float64)[np.asarray(tets, dtype=np.int64)]
    return np.einsum("ij,ij->i", x[:, 1]-x[:, 0],
                     np.cross(x[:, 2]-x[:, 0], x[:, 3]-x[:, 0]))


def orient_reference_positive(tets, reference):
    out = np.asarray(tets, dtype=np.int64).copy()
    d = dets(reference, out)
    if np.any(d == 0):
        raise ValueError("zero reference determinant")
    neg = d < 0
    if np.any(neg):
        tmp = out[neg, 2].copy()
        out[neg, 2] = out[neg, 3]
        out[neg, 3] = tmp
    if np.any(dets(reference, out) <= 0):
        raise ValueError("could not orient positive in trusted reference")
    return out


def perm_sign3(values):
    idx = [sorted(values).index(x) for x in values]
    inversions = sum(idx[i] > idx[j] for i in range(3) for j in range(i+1, 3))
    return -1 if inversions % 2 else 1


def tet_boundary_chain(tets, reference):
    t = orient_reference_positive(tets, reference)
    chain = defaultdict(int)
    for a, b, c, d in t:
        for face in ((b, c, d), (a, d, c), (a, b, d), (a, c, b)):
            key = tuple(sorted(map(int, face)))
            chain[key] += perm_sign3(face)
    return {k: v for k, v in chain.items() if v}


def shared_face_entries(tets, reference):
    T = orient_reference_positive(tets, reference)
    entries = defaultdict(list)
    for ci, (a, b, c, d) in enumerate(T):
        for face, opposite in (((b,c,d),a), ((a,d,c),b), ((a,b,d),c), ((a,c,b),d)):
            key = tuple(sorted(map(int, face)))
            entries[key].append((ci, tuple(map(int, face)), int(opposite),
                                 perm_sign3(face)))
    return entries


def topology_indices(tets, reference):
    faces = shared_face_entries(tets, reference)
    edge_cells = defaultdict(set)
    cell_owners = defaultdict(set)
    for ci, tet in enumerate(tets):
        cell_owners[tuple(sorted(map(int, tet)))].add(ci)
        for a, b in itertools.combinations(map(int, tet), 2):
            edge_cells[tuple(sorted((a, b)))].add(ci)
    return faces, edge_cells, cell_owners


def line_crosses_face_interior(points, u, v, face):
    p = np.asarray(points, dtype=np.float64)
    a, b, c = p[np.asarray(face, dtype=np.int64)]
    pu, pv = p[int(u)], p[int(v)]
    n = np.cross(b-a, c-a)
    du, dv = float(np.dot(n, pu-a)), float(np.dot(n, pv-a))
    den = du-dv
    if abs(den) <= 1e-30 or du * dv >= 0:
        return False
    t = du/den
    if not (0.0 < t < 1.0):
        return False
    q = pu + t*(pv-pu)
    M = np.column_stack((b-a, c-a))
    try:
        uv = np.linalg.lstsq(M, q-a, rcond=None)[0]
    except np.linalg.LinAlgError:
        return False
    bary = np.array([1.0-uv.sum(), uv[0], uv[1]])
    return bool(np.min(bary) > 1e-10)


def two_to_three(face, u, v):
    a, b, c = map(int, face)
    u, v = int(u), int(v)
    return np.asarray([
        [u, v, a, b],
        [u, v, b, c],
        [u, v, c, a],
    ], dtype=np.int64)


def evaluate_flip(pair, shared_face, opposites, T, mats, Xr64, Xr32, Xc32,
                  cell_owners):
    i, j = map(int, pair)
    reason = None
    if int(mats[i]) != int(mats[j]):
        return None, "material_mismatch"
    u, v = map(int, opposites)
    face = tuple(map(int, shared_face))
    raw = two_to_three(face, u, v)
    new = orient_reference_positive(raw, Xr64)
    # Fixed combinatorics must preserve exact directed cavity boundary.
    old_boundary = tet_boundary_chain(T[[i, j]], Xr64)
    new_boundary = tet_boundary_chain(new, Xr64)
    if old_boundary != new_boundary:
        return None, "oriented_boundary_mismatch"
    if not line_crosses_face_interior(Xr64, u, v, face):
        return None, "nonconvex_reference_bipyramid"
    # The six vertex sets must form distinct simplices.
    keys = [tuple(sorted(map(int, q))) for q in new]
    if len(set(keys)) != 3:
        return None, "duplicate_new_cells"
    if any(cell_owners.get(k, set()) - {i, j} for k in keys):
        return None, "duplicate_existing_cell"

    # Reference quality/positivity uses both double and actual Float32 coordinates.
    dr64 = dets(Xr64, new)
    dr32, _, qr32 = tet_metrics(Xr32, new)
    if np.any(dr64 <= 0) or np.any(dr32 <= 0):
        return None, "reference_nonpositive"
    if np.any(qr32 < Q_FLOOR):
        return None, "reference_quality_floor"
    # Current-pose replacement must be positive and above the quality floor.
    dc32, _, qc32 = tet_metrics(Xc32, new)
    if np.any(dc32 <= 0):
        return None, "current_nonpositive"
    if np.any(qc32 < Q_FLOOR):
        return None, "current_quality_floor"
    jr = dc32 / dr32
    lo, hi = J_BOUNDS
    new_j_bad = int(np.count_nonzero((jr < lo) | (jr > hi)))
    old_dr = dets(Xr32, T[[i, j]])
    old_dc, _, _ = tet_metrics(Xc32, T[[i, j]])
    old_j = old_dc / old_dr
    old_j_bad = int(np.count_nonzero((old_j < lo) | (old_j > hi)))
    if new_j_bad > old_j_bad:
        return None, "jacobian_bound_count_increase"
    old_bad = int(np.count_nonzero(old_dc <= 0))
    new_bad = int(np.count_nonzero(dc32 <= 0))
    if not (old_bad > new_bad):
        return None, "not_strict_invalid_reduction"
    # The pair boundary equality implies volume equality; also check numerically.
    v_old = float(np.sum(dets(Xr64, T[[i,j]]))/6.0)
    v_new = float(np.sum(dr64)/6.0)
    if abs(v_new-v_old) > max(1e-24, abs(v_old)*1e-9):
        return None, "reference_volume_mismatch"
    score = (old_bad-new_bad, old_j_bad-new_j_bad,
             float(np.min(qr32)), float(np.min(qc32)),
             -new_j_bad, -float(np.max(np.maximum(lo-jr, jr-hi))))
    return {
        "new_tets": new,
        "old_bad": old_bad,
        "new_bad": new_bad,
        "old_j_bad": old_j_bad,
        "new_j_bad": new_j_bad,
        "j_ratio": jr,
        "reference_det64": dr64,
        "reference_det32": dr32,
        "current_det32": dc32,
        "reference_q32": qr32,
        "current_q32": qc32,
        "reference_volume_delta_m3": v_new-v_old,
        "score": score,
    }, None


def summarize_state(T, mats, Xr64, Xr32, Xc32):
    dr32, _, qr32 = tet_metrics(Xr32, T)
    dc32, _, qc32 = tet_metrics(Xc32, T)
    jr = dc32/dr32
    lo, hi = J_BOUNDS
    return {
        "cell_count": int(len(T)),
        "reference_nonpositive_f32": int(np.count_nonzero(dr32 <= 0)),
        "reference_quality_below_1e-5_f32": int(np.count_nonzero(qr32 < Q_FLOOR)),
        "current_nonpositive_f32": int(np.count_nonzero(dc32 <= 0)),
        "current_quality_below_1e-5_f32": int(np.count_nonzero(qc32 < Q_FLOOR)),
        "current_J_below_0.35": int(np.count_nonzero(jr < lo)),
        "current_J_above_2.5": int(np.count_nonzero(jr > hi)),
        "reference_q_min_f32": float(qr32.min()),
        "current_q_min_f32": float(qc32.min()),
        "current_J_min_positive_reference": float(np.min(jr[dr32 > 0], initial=np.inf)),
        "current_J_max": float(np.max(jr, initial=-np.inf)),
    }


def run(output=OUT):
    start = time.time()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    src_sha = sha(SOURCE)
    z = np.load(SOURCE)
    Xr64 = np.asarray(z["reference_points_AP_m"], dtype=np.float64)
    Xc64 = np.asarray(z["target_points_AP_m"], dtype=np.float64)
    T = orient_reference_positive(np.asarray(z["candidate_tetrahedra"], dtype=np.int64), Xr64)
    mats = np.asarray(z["candidate_material_ids"], dtype=np.int64).copy()
    if len(T) != len(mats) or T.shape[1] != 4:
        raise ValueError("input tet/material domain mismatch")
    Xr32 = Xr64.astype(np.float32).astype(np.float64)
    Xc32 = Xc64.astype(np.float32).astype(np.float64)
    if not np.array_equal(Xr64.astype(np.float32), Xr32.astype(np.float32)):
        raise AssertionError("reference Float32 round trip mismatch")
    if not np.array_equal(Xc64.astype(np.float32), Xc32.astype(np.float32)):
        raise AssertionError("current Float32 round trip mismatch")
    baseline = summarize_state(T, mats, Xr64, Xr32, Xc32)
    if baseline["current_nonpositive_f32"] != 126:
        raise ValueError(f"pinned input changed: expected126 current invalid cells, got {baseline['current_nonpositive_f32']}")
    if baseline["reference_nonpositive_f32"] != 0:
        raise ValueError("baseline reference mesh is not positive")
    uid = np.arange(len(T), dtype=np.int64)
    next_uid = len(T)
    accepted = []
    rejected = defaultdict(int)
    pair_checks = 0
    sweeps = []
    deadline = start + MAX_SECONDS

    for sweep in range(1, MAX_SWEEPS+1):
        sweep_start = time.time()
        accepted_sweep = 0
        tried_sweep = 0
        while time.time() < deadline:
            state = summarize_state(T, mats, Xr64, Xr32, Xc32)
            bad_ids = set(np.flatnonzero(dets(Xc32, T) <= 0).tolist())
            if not bad_ids:
                break
            face_inc, edge_inc, cell_owners = topology_indices(T, Xr64)
            candidates = []
            for fkey, inc in face_inc.items():
                if len(inc) != 2:
                    continue
                (i, face_i, u, sign_i), (j, face_j, v, sign_j) = inc
                if i not in bad_ids and j not in bad_ids:
                    continue
                if sign_i == sign_j:
                    rejected["shared_face_not_oppositely_oriented"] += 1
                    continue
                if mats[i] != mats[j]:
                    rejected["material_mismatch"] += 1
                    continue
                if tuple(sorted((u,v))) in edge_inc and edge_inc[tuple(sorted((u,v)))] - {i,j}:
                    rejected["new_edge_exists_outside_pair"] += 1
                    continue
                if not line_crosses_face_interior(Xr64, u, v, fkey):
                    rejected["nonconvex_reference_bipyramid"] += 1
                    continue
                pair_checks += 1
                tried_sweep += 1
                result, reason = evaluate_flip((i,j), fkey, (u,v), T, mats,
                                               Xr64, Xr32, Xc32, cell_owners)
                if reason:
                    rejected[reason] += 1
                    continue
                candidates.append((result["score"], fkey, i, j, u, v, result))
            if not candidates:
                break
            # Choose deterministic best improving flip; rebuild all adjacency after it.
            candidates.sort(key=lambda q: (q[0], tuple(-x for x in q[1]), -q[2], -q[3]), reverse=True)
            score, face, i, j, u, v, result = candidates[0]
            before_bad = int(state["current_nonpositive_f32"])
            old_uids = [int(uid[i]), int(uid[j])]
            old_mat = int(mats[i])
            new_tets = result["new_tets"]
            # Remove two cells and append three; all other connectivity/materials unchanged.
            keep = np.ones(len(T), dtype=bool)
            keep[[i,j]] = False
            T = np.concatenate((T[keep], new_tets), axis=0)
            mats = np.concatenate((mats[keep], np.full(3, old_mat, dtype=np.int64)))
            new_uids = list(range(next_uid, next_uid+3))
            next_uid += 3
            uid = np.concatenate((uid[keep], np.asarray(new_uids, dtype=np.int64)))
            after = summarize_state(T, mats, Xr64, Xr32, Xc32)
            if after["current_nonpositive_f32"] >= before_bad:
                raise AssertionError("accepted flip did not strictly reduce global current invalid cells")
            accepted.append({
                "sweep": sweep, "old_cell_uids": old_uids, "old_initial_rows": [int(x) for x in old_uids if x < len(z["candidate_tetrahedra"])],
                "new_cell_uids": new_uids, "shared_face": list(map(int, face)),
                "opposite_vertices": [int(u), int(v)], "material_id": old_mat,
                "removed_current_nonpositive": result["old_bad"],
                "new_current_nonpositive": result["new_bad"],
                "old_J_out_of_bounds": result["old_j_bad"],
                "new_J_out_of_bounds": result["new_j_bad"],
                "new_J_ratios": result["j_ratio"].tolist(),
                "new_reference_q_f32": result["reference_q32"].tolist(),
                "new_current_q_f32": result["current_q32"].tolist(),
                "reference_volume_delta_m3": result["reference_volume_delta_m3"],
                "state_before": state, "state_after": after
            })
            accepted_sweep += 1
            # Loop restarts and reconstructs adjacency from the new mesh.
        sweeps.append({"sweep": sweep, "accepted": accepted_sweep,
                       "candidate_pair_checks": tried_sweep,
                       "elapsed_seconds": time.time()-sweep_start})
        if accepted_sweep == 0 or time.time() >= deadline:
            break

    final = summarize_state(T, mats, Xr64, Xr32, Xc32)
    if len(accepted) == 0:
        classification = "no_admissible_2_to_3_flip_found"
    elif final["current_nonpositive_f32"] < baseline["current_nonpositive_f32"]:
        classification = "partial_current_invalid_cell_reduction_topology_counterfactual"
    else:
        classification = "rejected_no_global_improvement"
    # Outer source positions, face topology and all O/I/D coordinates are bitwise preserved.
    report = {
        "schema": "numi.human.connected-mesh-same-material-2-to-3-flip.v1",
        "classification": classification,
        "source": {"path": str(SOURCE), "sha256": src_sha},
        "method": {
            "kind": "same-material bistellar 2-to-3 flips on internal faces touching current-invalid cells",
            "max_sweeps": MAX_SWEEPS, "max_seconds": MAX_SECONDS,
            "candidate_scope": "entire connected tetrahedral mesh; only internal faces incident to current-F32-nonpositive cells",
            "orientation": "all old/new cells oriented using trusted positive reference coordinates; current pose never determines orientation",
            "fixed": ["all reference/current point coordinates", "O/I/D face arrays", "material IDs of untouched cells"],
            "acceptance": ["exact oriented cavity boundary equality", "reference F32 q >= 1e-5", "current F32 determinant > 0 and q >= 1e-5",
                           "current local J-out-of-range count does not increase", "strict global reduction of current nonpositive cells",
                           "same material", "no duplicate cells", "new opposite edge absent outside pair"],
            "J_bounds": list(J_BOUNDS),
            "quality_definition": "pinned reference_fit_layer_counterfactual.tet_metrics; q is 12*(3*abs(volume))^(2/3)/sum(edge_length_squared)"
        },
        "baseline": baseline, "final": final, "accepted_flips": accepted,
        "sweeps": sweeps,
        "rejections": dict(sorted(rejected.items())),
        "counts": {"accepted": len(accepted), "candidate_pair_checks": pair_checks,
                   "changed_cells_removed": 2*len(accepted), "changed_cells_added": 3*len(accepted)},
        "invariants": {
            "reference_points_bitwise_unchanged": True,
            "current_points_bitwise_unchanged": True,
            "outer_face_arrays_bitwise_unchanged": True,
            "all_untouched_tet_connectivity_preserved": True,
            "all_untouched_material_ids_preserved": True,
        },
        "limits": ["This is a topology-only counterfactual, not a qualified physical repair.",
                   "The current state is the source-target O surface with original I/D positions.",
                   "Positive cells and bounded determinant ratios do not establish conservative mass ancestry or native material acceptance.",
                   "No whole-body collision, detailed-surface, constitutive, or runtime admission is inferred."],
        "runtime": {"hostname": socket.gethostname(), "python": sys.version,
                    "numpy": np.__version__, "seconds": time.time()-start}
    }
    out_tets = OUT_TETS = output / "candidate-tetrahedra.npz"
    np.savez_compressed(out_tets, reference_points_AP_m=Xr64, target_points_AP_m=Xc64,
                        candidate_tetrahedra=T.astype(np.int32),
                        candidate_material_ids=mats.astype(np.int8),
                        candidate_cell_uids=uid.astype(np.int64),
                        candidate_O_faces=np.asarray(z["candidate_O_faces"]),
                        candidate_I_faces=np.asarray(z["candidate_I_faces"]),
                        candidate_D_faces=np.asarray(z["candidate_D_faces"]))
    report["output"] = {"path": str(out_tets), "sha256": sha(out_tets)}
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"classification": classification, "accepted": len(accepted),
                      "baseline": baseline, "final": final,
                      "rejections": dict(sorted(rejected.items())),
                      "report": str(output/"report.json"),
                      "output": str(out_tets), "output_sha256": sha(out_tets),
                      "seconds": time.time()-start}, indent=2))


if __name__ == "__main__":
    run()
