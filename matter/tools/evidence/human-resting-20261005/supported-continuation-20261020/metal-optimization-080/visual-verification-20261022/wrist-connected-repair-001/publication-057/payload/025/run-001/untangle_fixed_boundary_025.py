#!/usr/bin/env python3
"""Bounded generic fixed-boundary tet untangle for 017 run-002; diagnostic only."""
from pathlib import Path
import hashlib, json, time
import numpy as np
from scipy.optimize import minimize, Bounds

W = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
OUTDIR = W / "connected-skin-safe-flow-025-generic-untangle" / "run-001"
OUTDIR.mkdir(parents=True, exist_ok=True)
MESH = W / "E/connected-skin-mechanical-transfer-013/boundary-refinement-002/candidate.npz"
PROPOSAL = W / "connected-skin-safe-flow-017/run-002/connected-rest-proposal.npz"
SCRIPT = Path(__file__)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def det_grad(x, tt):
    p = x[tt]
    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    e3 = p[:, 3] - p[:, 0]
    g1 = np.cross(e2, e3)
    g2 = np.cross(e3, e1)
    g3 = np.cross(e1, e2)
    det = np.einsum("ij,ij->i", e1, g1)
    grad = np.stack((-g1 - g2 - g3, g1, g2, g3), axis=1)
    return det, grad

def max_tet_edge(x, tt):
    p = x[tt]
    lengths = []
    for a, b in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
        lengths.append(np.linalg.norm(p[:,a]-p[:,b], axis=1))
    return np.max(np.stack(lengths, axis=1), axis=1)

t0 = time.monotonic()
assert sha(MESH) == "1849d24c14f8c0ddfedf3aa131877bdd38a852f9e48baebcf900a0efadf75db1"
assert sha(PROPOSAL) == "64f40e915703e35950ba2d3ab95b01a5e75681616c316cace25518c3a0ccb054"
m = np.load(MESH)
p = np.load(PROPOSAL)
T = np.asarray(m["tetrahedra"], dtype=np.int64)
Xbase = np.asarray(m["points_world_m"], dtype=np.float32).astype(np.float64)
Xref = np.asarray(p["points_world_m"], dtype=np.float32).astype(np.float64)
assert T.shape == (253425, 4) and Xbase.shape == Xref.shape == (56514, 3)
assert np.array_equal(T, p["tetrahedra"])
# The 017 proposal intentionally moves active O rows; keep it fixed relative to Xref.

# The declared physical shell plus tagged construction closure must equal the
# tetrahedral boundary. Hold both fixed during this volume-only diagnostic.
allfaces = np.sort(T[:, [[1,2,3],[0,3,2],[0,1,3],[0,2,1]]].reshape(-1,3), axis=1)
unique_faces, face_counts = np.unique(allfaces, axis=0, return_counts=True)
tet_boundary = unique_faces[face_counts == 1]
declared_boundary = np.unique(np.vstack((m["physical_shell_faces"], m["construction_closure_faces"])), axis=0)
assert {tuple(x) for x in np.sort(tet_boundary, axis=1)} == {tuple(x) for x in np.sort(declared_boundary, axis=1)}
boundary_nodes = np.unique(tet_boundary)
fixed = np.zeros(len(Xref), dtype=bool)
fixed[boundary_nodes] = True
fixed[np.asarray(m["physical_outer_node_ids"], dtype=np.int64)] = True
fixed[np.asarray(p["protected_node_ids"], dtype=np.int64)] = True

d_base, _ = det_grad(Xbase, T)
d_ref, _ = det_grad(Xref, T)
assert np.all(d_base > 0), "013 reference must be positive for signed relative checks"
bad = np.flatnonzero(d_ref <= 0)
assert len(bad) == 78, f"pinned 017 failure count changed: {len(bad)}"
bad_nodes = T[bad]
free_nodes = np.unique(bad_nodes[~fixed[bad_nodes]])
assert len(free_nodes) == 86
affected = np.flatnonzero(np.any(np.isin(T, free_nodes), axis=1))
assert len(affected) == 1011
assert np.all(np.any(np.isin(T[bad], free_nodes), axis=1))
loc = np.full(len(Xref), -1, dtype=np.int64)
loc[free_nodes] = np.arange(len(free_nodes), dtype=np.int64)
Tl = T[affected]
local = loc[Tl]
Lref = max_tet_edge(Xref, Tl)
scale = np.maximum(Lref ** 3, np.finfo(np.float64).tiny)
qbase = d_base / (max_tet_edge(Xbase, T) ** 3)
qref = d_ref / (max_tet_edge(Xref, T) ** 3)
assert float(qbase.min()) >= 1e-5 and int(np.sum(qref <= 0)) == 78

def unpack(u):
    x = Xref.copy()
    x[free_nodes] += np.asarray(u).reshape(-1, 3) * 0.001
    return x

def cfun(u, target):
    x = unpack(u)
    d, _ = det_grad(x, Tl)
    return d / scale - target

def cjac(u, target):
    x = unpack(u)
    d, g = det_grad(x, Tl)
    J = np.zeros((len(affected), 3 * len(free_nodes)), dtype=np.float64)
    for corner in range(4):
        keep = local[:, corner] >= 0
        rr = np.flatnonzero(keep)
        vv = local[keep, corner]
        cols = 3 * vv[:, None] + np.arange(3)[None, :]
        J[rr[:, None], cols] = g[keep, corner] * (0.001 / scale[keep, None])
    return J

# Validate the analytic signed-determinant Jacobian before optimization.
rng = np.random.default_rng(250)
direction = rng.normal(size=3 * len(free_nodes))
direction /= np.linalg.norm(direction)
step = 1e-3
fd = (cfun(step * direction, 1e-5) - cfun(-step * direction, 1e-5)) / (2 * step)
ad = cjac(np.zeros_like(direction), 1e-5) @ direction
jac_rel = float(np.linalg.norm(fd - ad) / max(1.0, np.linalg.norm(fd)))
assert jac_rel < 1e-5, jac_rel

trace = []
solve_started = time.monotonic()
last_feasible = None
statuses = []
for trial, target in enumerate((1.005e-5, 1.02e-5, 1.05e-5, 1.10e-5), start=1):
    start_u = np.zeros(3 * len(free_nodes)) if last_feasible is None else last_feasible
    def callback(u):
        elapsed = time.monotonic() - solve_started
        if elapsed > 520:
            raise TimeoutError("total optimizer budget exceeded")
        cc = cfun(u, target)
        trace.append({"trial": trial, "target_q": target, "iteration": len([r for r in trace if r["trial"] == trial]),
                      "elapsed_s": elapsed, "minimum_constraint": float(cc.min()),
                      "max_node_move_mm": float(np.linalg.norm(np.asarray(u).reshape(-1,3),axis=1).max())})
    try:
        res = minimize(lambda u: (0.5 * float(u @ u), u), start_u, jac=True, method="SLSQP",
                       bounds=Bounds(np.full(start_u.shape, -3.0), np.full(start_u.shape, 3.0)),
                       constraints=[{"type": "ineq", "fun": lambda u: cfun(u, target),
                                     "jac": lambda u: cjac(u, target)}],
                       callback=callback,
                       options={"maxiter": 80, "ftol": 1e-10, "disp": False})
    except TimeoutError as ex:
        statuses.append({"trial": trial, "target_q": target, "status": "bounded-timeout", "error": str(ex)})
        break
    cmin = float(cfun(res.x, target).min())
    statuses.append({"trial": trial, "target_q": target, "success": bool(res.success),
                     "status": int(res.status), "message": str(res.message), "iterations": int(res.nit),
                     "minimum_linearized_constraint": cmin, "objective_mm2": float(res.fun)})
    if cmin >= -1e-10:
        last_feasible = np.asarray(res.x).copy()
        xf = unpack(last_feasible).astype(np.float32).astype(np.float64)
        df, _ = det_grad(xf, T)
        qf = df / (max_tet_edge(xf, T) ** 3)
        if float(qf.min()) >= 1e-5 and int(np.sum(df <= 0)) == 0:
            break
    else:
        break

report = {
    "schema": "numi.connected-skin-fixed-boundary-local-untangle.v1",
    "status": "candidate_only_not_admitted",
    "method": "generic local SLSQP signed-tetrahedron-determinant untangler adapted from owner 196; fixes every declared tetrahedral boundary node, physical O node, and 017 protected node; frees only 86 interior nodes from the 78 inverted cells and constrains their complete 1,011-tet incident star",
    "inputs": {"013_mesh": str(MESH), "013_sha256": sha(MESH), "017_proposal": str(PROPOSAL),
               "017_sha256": sha(PROPOSAL), "script": str(SCRIPT), "script_sha256": sha(SCRIPT)},
    "domain": {"nodes": len(Xref), "tets": len(T), "region_attribute_values": np.unique(m["region_attributes"]).astype(int).tolist(),
               "typed_O_I_D_maps_available": False, "attachment_bank_stitch_masks_available": False,
               "declared_tet_boundary_faces": int(len(tet_boundary)),
               "declared_boundary_matches_tet_boundary": True},
    "pre_solve": {"inverted_tets": int(len(bad)), "base013_nonpositive_tets": int(np.sum(d_base <= 0)),
                  "base013_min_normalized_det": float(qbase.min()), "proposal017_min_normalized_det": float(qref.min()),
                  "proposal017_min_relative_J_on_bad": float(np.min(d_ref[bad] / d_base[bad])),
                  "interior_free_nodes": int(len(free_nodes)), "free_node_rows": free_nodes.tolist(),
                  "complete_incident_tets": int(len(affected)),
                  "protected_O_and_boundary_fixed": True, "analytic_jacobian_relative_error": jac_rel},
    "optimization": {"targets": [1.005e-5, 1.02e-5, 1.05e-5, 1.10e-5], "coordinate_bounds_mm": [-3.0, 3.0],
                     "trials": statuses, "iteration_trace": trace, "elapsed_s": time.monotonic()-solve_started},
    "post": {},
    "limits": ["No O/I/D layer decomposition or independent layer boundary maps are present in 013/017; this is not an O/I/D layer-refit result.",
               "013 has no attachment/bank/stitch node masks. Only declared tet-shell nodes, physical O nodes, and the 017 protected-node IDs were fixed.",
               "Positive determinants and normalized quality do not certify collision clearance, material ancestry, regional mass, source transfer, or native admission.",
               "Parent volume fractions are not recomputed for the changed interior geometry; do not reuse them as conservative mass transfer."],
    "elapsed_s": time.monotonic()-t0
}
if last_feasible is not None:
    Xnew = unpack(last_feasible).astype(np.float32)
    Xnew64 = Xnew.astype(np.float64)
    dn, _ = det_grad(Xnew64, T)
    qn = dn / max_tet_edge(Xnew64, T) ** 3
    actual_changed = np.flatnonzero(np.any(Xnew != np.asarray(p["points_world_m"], dtype=np.float32), axis=1))
    proposal_fixed_equal = bool(np.array_equal(Xnew[fixed], np.asarray(p["points_world_m"], dtype=np.float32)[fixed]))
    nonfree_equal = bool(np.array_equal(Xnew[np.setdiff1d(np.arange(len(Xnew)), free_nodes)],
                                         np.asarray(p["points_world_m"], dtype=np.float32)[np.setdiff1d(np.arange(len(Xnew)), free_nodes)]))
    report["post"] = {"float32_nonpositive_tets": int(np.sum(dn <= 0)),
                      "float32_below_1e-5": int(np.sum(qn < 1e-5)),
                      "float32_min_q": float(qn.min()), "float32_min_det_m3": float(dn.min()),
                      "all_fixed_nodes_byte_identical_to_017": proposal_fixed_equal,
                      "all_nodes_outside_free_set_byte_identical_to_017": nonfree_equal,
                      "changed_nodes_vs_017": int(len(actual_changed)),
                      "max_free_adjustment_mm": float(np.linalg.norm(Xnew64[free_nodes] - Xref[free_nodes], axis=1).max()*1000),
                      "max_total_displacement_from_013_mm": float(np.linalg.norm(Xnew64-Xbase, axis=1).max()*1000),
                      "signed_total_volume_delta_m3": float(np.sum(dn-d_base)/6.0),
                      "positive_quality_gate_pass": bool(np.all(dn > 0) and np.all(qn >= 1e-5)),
                      "fixed_boundary_gate_pass": bool(proposal_fixed_equal and nonfree_equal)}
    out = OUTDIR / "candidate.npz"
    # Preserve geometry and explicit ancestry row domains; omit stale volume fractions/costs.
    keys = ("parent_008_tet_rows", "parent_005_tet_rows", "region_attributes",
            "physical_outer_faces", "physical_O_parent_face_rows",
            "physical_O_parent_detailed_source_face_ids", "physical_O_parent_surface_ids",
            "physical_O_parent_source271_face_rows", "physical_O_parent_barycentric",
            "physical_outer_node_ids", "physical_shell_faces", "physical_shell_parent_face_rows",
            "construction_closure_faces", "parent_008_tet_rows")
    payload = {k: np.asarray(m[k]) for k in dict.fromkeys(keys) if k in m.files}
    payload.update({"points_world_m": Xnew, "tetrahedra": T,
                    "protected_node_ids": np.asarray(p["protected_node_ids"]),
                    "untangle_free_node_ids": free_nodes,
                    "untangle_constrained_tet_rows": affected,
                    "proposal_node_displacements_m": np.asarray(p["node_displacements_m"]),
                    "candidate_node_displacements_from_013_m": Xnew64-Xbase})
    np.savez_compressed(out, **payload)
    report["output"] = {"candidate_npz": str(out), "candidate_sha256": sha(out)}
else:
    report["post"] = {"candidate_emitted": False, "reason": "no SLSQP iterate met the linearized constraints"}
report_path = OUTDIR / "report.json"
report_path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"status": report["status"], "bad_before": len(bad), "free_nodes": len(free_nodes),
                  "incident_tets": len(affected), "optimizer_trials": statuses,
                  "post": report["post"], "output": report.get("output"), "elapsed_s": report["elapsed_s"]}, indent=2), flush=True)

