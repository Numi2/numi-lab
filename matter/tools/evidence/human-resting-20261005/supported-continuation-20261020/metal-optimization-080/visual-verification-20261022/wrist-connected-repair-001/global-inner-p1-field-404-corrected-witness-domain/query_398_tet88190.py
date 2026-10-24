#!/usr/bin/env python3
"""Reproduce bounded 398/tet-88190 query against current registered NHSKIN."""
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
CAND = ROOT / "global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/local-quality-cavities-398/candidate-with-transfer.npz"
POSE = ROOT / "wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-003/registered-skin-poses.json"
REBIND = ROOT / "regional-refinement-compose-255/tools/rebind_human_skin_reference.py"
BODYADM = ROOT / "exact-plane-filter-owner-264/src/numilab_human/bodyparts_adm.py"
ASSETS = [
    ROOT / "left-ankle-rest-preserving-weight-repair-128/candidate.nhskin",
    Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin"),
]
sys.path.insert(0, str(BODYADM.parent.parent))
spec = importlib.util.spec_from_file_location("rebind_human_skin_reference", REBIND)
rebind = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rebind)
from numilab_human.bodyparts_adm import prepare_nearest_surface_index

mesh = np.load(CAND)
X, T = mesh["points_world_m"], mesh["tetrahedra"]
tet_row = 88190
node_ids = T[tet_row]
bary = np.array([0.7473772446, 0.0, 0.2526227554, 0.0], dtype=np.float64)
q = bary @ X[node_ids]
snapshot = json.loads(POSE.read_text())
out = {"candidate_sha256": hashlib.sha256(CAND.read_bytes()).hexdigest(),
       "pose_sha256": hashlib.sha256(POSE.read_bytes()).hexdigest(),
       "tet_row": int(tet_row), "node_ids": node_ids.tolist(),
       "node_xyz_world_m": X[node_ids].tolist(), "barycentric": bary.tolist(),
       "query_world_m": q.tolist(), "assets": []}
for asset in ASSETS:
    raw = asset.read_bytes()
    payload = rebind.decode(raw)
    xyz = rebind.registered_positions_float32(payload, snapshot, frame="current")
    F = np.asarray(payload["inds"], dtype=np.int64)
    idx = prepare_nearest_surface_index(xyz, F)
    distance, closest, face_row = idx.closest(q)
    closest = np.asarray(closest, dtype=np.float64)
    face = F[face_row]
    tri = xyz[face].astype(np.float64)
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    normal /= np.linalg.norm(normal)
    signed = float((q - closest) @ normal)
    def winding(point):
        rel = xyz[F].astype(np.float64) - point[None, None, :]
        a, b, c = rel[:, 0], rel[:, 1], rel[:, 2]
        la, lb, lc = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1), np.linalg.norm(c, axis=1)
        num = np.einsum("ij,ij->i", a, np.cross(b, c))
        den = (la*lb*lc + np.einsum("ij,ij->i", a,b)*lc +
               np.einsum("ij,ij->i", b,c)*la + np.einsum("ij,ij->i", c,a)*lb)
        return float((2*np.arctan2(num, den)).sum()/(4*np.pi))
    E = np.sort(np.concatenate((F[:,[0,1]], F[:,[1,2]], F[:,[2,0]]), axis=0), axis=1)
    _, counts = np.unique(E, axis=0, return_counts=True)
    out["assets"].append({
        "path": str(asset), "sha256": hashlib.sha256(raw).hexdigest(),
        "emitted_current_xyz_sha256": hashlib.sha256(xyz.tobytes()).hexdigest(),
        "vertices": int(payload["nv"]), "faces": int(len(F)),
        "closest_distance_m": float(distance), "closest_point_world_m": closest.tolist(),
        "closest_face_row": int(face_row), "closest_face_vertex_ids": face.tolist(),
        "oriented_face_normal": normal.tolist(), "signed_normal_offset_m": signed,
        "winding_at_query": winding(q),
        "winding_at_offsets_m": {str(d): winding(q+d*normal) for d in (-.001,-.0001,.0001,.001)},
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
    })
print(json.dumps(out, indent=2, sort_keys=True))

