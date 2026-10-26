import importlib.util
import unittest
from pathlib import Path
import numpy as np
from scipy import sparse
from whole_face_objective_429 import build_whole_face_clearance_rows

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
OWNER = ROOT / "D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
spec = importlib.util.spec_from_file_location("frozen_surface_witness_408", OWNER)
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)

class BaryOwner:
    @staticmethod
    def build_barycentric_exit_system(points, faces, rows, weights, normals, targets, *, anchor_rows=()):
        rows = np.asarray(rows, dtype=np.int64)
        tri = np.asarray(faces, dtype=np.int64)[rows]
        weights = np.asarray(weights, dtype=np.float64)
        n = len(rows)
        B = sparse.coo_matrix((weights.ravel(),
            (np.repeat(np.arange(n), 3), tri.ravel())), shape=(n, len(points))).tocsr()
        sample = np.asarray(B @ points).reshape(-1, 3)
        delta = np.asarray(targets, dtype=np.float64) - sample
        g = weights[:, :, None] * np.asarray(normals)[:, None, :]
        h = np.einsum("ij,ij->i", np.asarray(normals), delta)
        return B, tri, g, h, None

def empty_points():
    return {"point_rows": np.empty(0, np.int64),
        "point_targets_f32_m": np.empty((0, 3), np.float32),
        "point_directions": np.empty((0, 3), np.float64),
        "point_report": {}, "point_npz": {}}

def targets(points, faces, face_rows, endpoints=None):
    rows = np.asarray(face_rows, dtype=np.int64)
    nodes = np.asarray(faces, dtype=np.int64)[rows]
    bary = np.full((len(rows), 3), 1/3, dtype=np.float64)
    sample = np.einsum("ij,ijk->ik", bary, np.asarray(points)[nodes])
    if endpoints is None:
        endpoints = sample + np.tile([1.0, 0.0, 0.0], (len(rows), 1))
    return {"face_rows": rows, "face_nodes": nodes, "barycentric_weights": bary,
        "face_samples_m": sample/1000.0,
        "face_targets_f32_m": (np.asarray(endpoints)/1000.0).astype(np.float32),
        "face_directions": np.tile([1.0, 0.0, 0.0], (len(rows), 1)),
        "face_report": {}, "face_npz": {
            "bank_body": np.arange(len(rows), dtype=np.int64)+20,
            "bank_stable_bone_id": np.arange(len(rows), dtype=np.int64)+100,
            "bank_triangle_stable_id": np.arange(len(rows), dtype=np.int64)+200}}

def build(points, target, point=None, anchors=()):
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    point = empty_points() if point is None else point
    return build_whole_face_clearance_rows(np.asarray(points, dtype=np.float64),
        faces, np.arange(len(points)), np.asarray(anchors, dtype=np.int64),
        point, target, BaryOwner, owner.build_clearance_rows)

class Tests(unittest.TestCase):
    def test_barycentric_sample_exit_does_not_clear_every_corner(self):
        p = np.asarray([[0,0,0], [2,0,0], [-2,0,0]], dtype=np.float64)
        t = targets(p, [[0,1,2]], [0], endpoints=[[1,0,0]])
        old = owner.build_clearance_rows(p, [[0,1,2]], np.arange(3),
            np.empty(0, np.int64), empty_points(), t, BaryOwner)
        move = np.tile([1.0,0,0], 3)
        self.assertGreaterEqual(float(np.asarray(old[0] @ move).reshape(-1)[0]), float(old[1][0])-1e-6)
        C,h,_,m = build(p,t)
        self.assertEqual([r["corner_row"] for r in m], [0,2])
        self.assertTrue(np.allclose(h, [1,3]))
        self.assertFalse(np.all(np.asarray(C @ move).ravel() >= h))
        # A zero projected deficit is omitted as well as the negative deficit above.
        pzero = np.asarray([[0,0,0], [float(np.float32(0.002))*1000,0,0], [-1,0,0]], dtype=np.float64)
        tzero = targets(pzero, [[0,1,2]], [0], endpoints=[[2,0,0]])
        _, hz, _, mz = build(pzero, tzero)
        self.assertEqual([r["corner_row"] for r in mz], [0,2])
        self.assertTrue(np.all(hz > 0))

    def test_coherent_translation_satisfies_all_corner_rows(self):
        p = np.zeros((3,3))
        C,h,_,_ = build(p, targets(p, [[0,1,2]], [0], endpoints=[[1,0,0]]))
        self.assertEqual(len(h), 3)
        self.assertTrue(np.allclose(C @ np.tile([1.0,0,0],3), h))

    def test_duplicate_pairs_share_one_unique_face_mass(self):
        p = np.zeros((3,3))
        C,h,w,m = build(p, targets(p, [[0,1,2]], [0,0],
            endpoints=[[1,0,0],[2,0,0]]))
        self.assertEqual(len(h), 6)
        self.assertAlmostEqual(float(w.sum()), 1.0)
        masses = {}
        for x,r in zip(w,m): masses[r["bank_body"]] = masses.get(r["bank_body"],0)+float(x)
        self.assertTrue(all(abs(x-.5)<1e-12 for x in masses.values()))
        self.assertTrue(np.allclose(w, 1/6))

    def test_anchor_has_zero_coefficients_and_retained_slack(self):
        p = np.zeros((3,3))
        C,h,w,m = build(p, targets(p, [[0,1,2]], [0], endpoints=[[1,0,0]]), anchors=[0])
        i = next(i for i,r in enumerate(m) if r["corner_row"] == 0)
        self.assertEqual(C.getrow(i).nnz, 0)
        self.assertAlmostEqual(float(h[i]), 1.0, places=6)
        self.assertGreater(float(w[i]), 0)

    def test_empty_targets_keep_shape(self):
        p = np.zeros((3,3))
        C,h,w,m = build(p, targets(p, [[0,1,2]], []))
        self.assertEqual(C.shape, (0,9))
        self.assertEqual((len(h),len(w),len(m)), (0,0,0))

    def test_point_rows_are_preserved_and_share_mass(self):
        p = np.zeros((3,3))
        point = {"point_rows": np.array([0], np.int64),
            "point_targets_f32_m": np.array([[.001,0,0]], np.float32),
            "point_directions": np.array([[1.,0,0]], np.float64),
            "point_report": {}, "point_npz": {}}
        C,h,w,m = build(p, targets(p, [[0,1,2]], [0], endpoints=[[1,0,0]]), point)
        self.assertEqual(float(C[0,0]), 1.0)
        self.assertAlmostEqual(float(h[0]), 1.0, places=6)
        self.assertAlmostEqual(float(w[0]), .5)
        self.assertAlmostEqual(float(w[1:].sum()), .5)
        self.assertEqual(m[0]["kind"], "point")

if __name__ == "__main__":
    unittest.main(verbosity=2)
