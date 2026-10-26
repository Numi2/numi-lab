import importlib.util
import unittest
from pathlib import Path
import numpy as np
from scipy import sparse
from whole_face_objective_435 import build_face_bone_grouped_clearance_rows, select_one_plane_per_face_bone

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
OWNER = ROOT / "D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
spec = importlib.util.spec_from_file_location("frozen_surface_witness_408_for_435", OWNER)
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

def targets(points, faces, face_rows, directions=None, endpoints=None, bodies=None, bones=None, tri_ids=None):
    rows=np.asarray(face_rows,dtype=np.int64); nodes=np.asarray(faces,dtype=np.int64)[rows]
    bary=np.full((len(rows),3),1/3,dtype=np.float64)
    sample=np.einsum("ij,ijk->ik",bary,np.asarray(points)[nodes])
    if directions is None: directions=np.tile([1.,0,0],(len(rows),1))
    directions=np.asarray(directions,dtype=float)
    if endpoints is None: endpoints=sample+directions
    count=len(rows)
    return {"face_rows":rows,"face_nodes":nodes,"barycentric_weights":bary,
        "face_samples_m":sample/1000.0,
        "face_targets_f32_m":(np.asarray(endpoints)/1000.0).astype(np.float32),
        "face_directions":directions,"face_report":{},
        "face_npz":{"bank_body":np.asarray(bodies if bodies is not None else np.full(count,20),dtype=np.int64),
        "bank_stable_bone_id":np.asarray(bones if bones is not None else np.full(count,100),dtype=np.int64),
        "bank_triangle_stable_id":np.asarray(tri_ids if tri_ids is not None else np.arange(count)+200,dtype=np.int64)}}

def build(points, face_targets, point=None, anchors=()):
    faces=np.asarray([[0,1,2]],dtype=np.int64); point=empty_points() if point is None else point
    return build_face_bone_grouped_clearance_rows(np.asarray(points,dtype=float),faces,
        np.arange(len(points)),np.asarray(anchors,dtype=np.int64),point,face_targets,
        BaryOwner,owner.build_clearance_rows)

class Tests(unittest.TestCase):
    def test_opposing_triangle_planes_reduce_to_one_group_plane(self):
        p=np.asarray([[0,0,0],[2,0,0],[-2,0,0]],float)
        t=targets(p,[[0,1,2]],[0,0],directions=[[1,0,0],[-1,0,0]],
            endpoints=[[2,0,0],[-1,0,0]],bodies=[43,43],bones=[16,16],tri_ids=[100,101])
        selected,_,info=select_one_plane_per_face_bone(p,t)
        self.assertEqual(selected.tolist(),[1])
        self.assertEqual(info[1]["candidate_source_indices"],[0,1])
        C,h,w,m=build(p,t)
        self.assertEqual(len(m),2)
        self.assertEqual({x["source_index"] for x in m},{1})
        self.assertTrue(all(np.allclose(x["proposal_direction_world_unit"],[-1,0,0]) for x in m))
        self.assertAlmostEqual(float(h.max()),3.0,places=5)
        self.assertAlmostEqual(float(w.sum()),1.0)

    def test_tie_uses_lower_source_index_and_is_deterministic(self):
        p=np.asarray([[0,0,0],[0,1,0],[0,-1,0]],float)
        t=targets(p,[[0,1,2]],[0,0],directions=[[1,0,0],[-1,0,0]],
            endpoints=[[1,0,0],[-1,0,0]],bodies=[1,1],bones=[2,2],tri_ids=[4,5])
        a=select_one_plane_per_face_bone(p,t)[0]; b=select_one_plane_per_face_bone(p,t)[0]
        self.assertEqual(a.tolist(),[0]); self.assertTrue(np.array_equal(a,b))

    def test_point_rows_are_exact_frozen_408_rows(self):
        p=np.zeros((3,3))
        t=targets(p,[[0,1,2]],[0,0],directions=[[1,0,0],[1,0,0]],
            endpoints=[[1,0,0],[2,0,0]],bodies=[2,2],bones=[3,3])
        point={"point_rows":np.array([0],np.int64),
            "point_targets_f32_m":np.array([[.001,0,0]],np.float32),
            "point_directions":np.array([[1.,0,0]],np.float64),"point_report":{},"point_npz":{}}
        full=owner.build_clearance_rows(p,[[0,1,2]],np.arange(3),np.empty(0,np.int64),point,t,BaryOwner)
        C,h,w,m=build(p,t,point)
        self.assertEqual((C[:1]!=full[0][:1]).nnz,0)
        self.assertTrue(np.array_equal(h[:1],full[1][:1]))
        self.assertTrue(np.array_equal(w[:1],full[2][:1]))
        self.assertEqual(m[0],full[3][0])
        self.assertAlmostEqual(float(w[1:].sum()),.5)

    def test_anchor_keeps_zero_coefficients_and_positive_slack_row(self):
        p=np.zeros((3,3))
        t=targets(p,[[0,1,2]],[0],directions=[[-1,0,0]],endpoints=[[-1,0,0]],bodies=[2],bones=[3])
        C,h,w,m=build(p,t,anchors=[0])
        row=next(i for i,x in enumerate(m) if x["corner_row"]==0)
        self.assertEqual(C.getrow(row).nnz,0); self.assertGreater(float(h[row]),0); self.assertGreater(float(w[row]),0)

    def test_empty_targets_keep_shape(self):
        p=np.zeros((3,3)); t=targets(p,[[0,1,2]],[])
        C,h,w,m=build(p,t)
        self.assertEqual(C.shape,(0,9)); self.assertEqual((len(h),len(w),len(m)),(0,0,0))

if __name__=="__main__": unittest.main(verbosity=2)
