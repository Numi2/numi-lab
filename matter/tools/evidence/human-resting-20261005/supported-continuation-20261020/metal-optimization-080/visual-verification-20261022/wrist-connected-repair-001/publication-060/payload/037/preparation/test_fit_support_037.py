import io, unittest, tempfile
from pathlib import Path
import numpy as np
from fit_support_037 import (expand_one_ring, merge_quality_rows, discover_quality_rows,
    parse_all_bank_csv, transform_inside_constraints, layer_floor_mm, ActiveQualityRows,
    normalized_tet_quality_jacobian, projected_layer_thickness_mm)

class Fit037HelperTests(unittest.TestCase):
    def test_one_ring_does_not_recurse(self):
        faces=np.array([[0,1,2],[2,3,4],[4,5,6]],dtype=np.int64)
        self.assertEqual(expand_one_ring([0],faces).tolist(),[0,1,2])
        self.assertEqual(expand_one_ring([0,4],faces).tolist(),[0,1,2,3,4,5,6])

    def test_quality_restarts_keep_all_discovered_rows(self):
        affected=np.array([10,11,12,13],dtype=np.int64)
        first=discover_quality_rows(affected,np.array([2.,.8,2.,2.]),np.array([2.,.7,2.,2.]),1.)
        active=merge_quality_rows([11,20],first)
        second=discover_quality_rows(affected,np.array([2.,2.,.5,2.]),np.array([2.,2.,.6,2.]),1.)
        active=merge_quality_rows(active,second)
        self.assertEqual(active.tolist(),[11,12,20])

    def _csv(self, lines):
        header="local_node,global_node,layer,row,role_mask,bank_body,bank_stable_id,classification,inside_rays,outside_rays,ambiguous,nearest_bone,nearest_triangle,nearest_distance_m,signed_face_distance_m,query_local_x,query_local_y,query_local_z,closest_x,closest_y,closest_z\n"
        return io.StringIO(header+"\n".join(lines)+"\n")

    def test_per_bone_rows_preserve_same_node_and_unresolved(self):
        rows, unresolved, summary=parse_all_bank_csv(self._csv([
            "7,100,0,7,1,23,5,inside,3,0,0,5,50,1,-1,0,0,0,1,0,0",
            "7,100,0,7,1,23,6,inside,3,0,0,6,60,1,-1,0,0,0,0,1,0",
            "7,100,0,7,1,23,9,ambiguous,1,1,1,9,90,1,0,0,0,0,0,0,1",
            "8,101,1,8,2,24,4,outside,0,3,0,4,40,2,2,0,0,0,0,0,2",
        ]),node_count=10)
        self.assertEqual(len(rows),2)
        self.assertEqual([(r["local_node"],r["body_index"],r["bone_stable_id"]) for r in rows],[(7,23,5),(7,23,6)])
        self.assertEqual(len(unresolved),1)
        self.assertEqual(summary["inside_constraints"],2)
        np.testing.assert_allclose(rows[0]["outward_local"],[1,0,0])
        np.testing.assert_allclose(rows[1]["outward_local"],[0,1,0])

    def test_duplicate_key_and_mismatched_nearest_bone_fail(self):
        base="7,100,0,7,1,23,5,inside,3,0,0,5,50,1,-1,0,0,0,1,0,0"
        with self.assertRaisesRegex(ValueError,"duplicate"):
            parse_all_bank_csv(self._csv([base,base]))
        bad=base.replace(",5,50,1,",",6,50,1,")
        with self.assertRaisesRegex(ValueError,"nearest bone"):
            parse_all_bank_csv(self._csv([bad]))

    def test_zero_distance_is_unresolved_not_a_fabricated_normal(self):
        line="7,100,0,7,1,23,5,inside,3,0,0,5,50,0,0,0,0,0,0,0,0"
        rows, unresolved, _=parse_all_bank_csv(self._csv([line]))
        self.assertEqual(rows,[])
        self.assertEqual(unresolved[0]["reason"],"zero_or_unresolved_direction")

    def test_world_transform_rotates_pairwise_constraints(self):
        rows=[{"local_node":0,"body_index":23,"bone_stable_id":5,"nearest_triangle":50,
               "query_local":np.zeros(3),"closest_local":np.array([1.,0.,0.]),
               "outward_local":np.array([1.,0.,0.]),"inside_depth_m":1.}]
        poses={23:{"current":{"orientation_world_xyzw":[0.,0.,np.sqrt(.5),np.sqrt(.5)]}}}
        out=transform_inside_constraints(rows,np.array([[10.,2.,3.]]),poses)
        np.testing.assert_allclose(out["constraint_closest_world_m"],[[10.,3.,3.]],atol=1e-12)
        np.testing.assert_allclose(out["constraint_outward_normal_world"],[[0.,1.,0.]],atol=1e-12)


    def test_active_quality_scans_full_domain_and_checkpoints_restart(self):
        affected=np.array([10,11,12,13],dtype=np.int64)
        manager=ActiveQualityRows(affected,[11],1.0)
        new=manager.scan(np.array([2.,2.,.5,2.]),np.array([2.,2.,.6,2.]))
        self.assertEqual(new.tolist(),[12])
        # A later accepted iterate discovers another row; the first discovery remains active.
        new=manager.scan(np.array([2.,2.,2.,.4]),np.array([2.,2.,2.,.5]))
        self.assertEqual(new.tolist(),[13])
        self.assertEqual(manager.active.tolist(),[11,12,13])
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/"state.npz"
            manager.checkpoint(out,x_mm=np.zeros((2,3)),variable_local_rows=[4,5],
                               q_float64=np.array([2.,2.,2.,.4]),
                               q_emitted_float32=np.array([2.,2.,2.,.5]))
            z=np.load(out)
            self.assertEqual(z["affected_tet_rows"].tolist(),[10,11,12,13])
            self.assertEqual(z["active_tet_rows"].tolist(),[11,12,13])
            self.assertEqual(int(z["scan_count"]),2)
        with self.assertRaisesRegex(ValueError,"cover every affected"):
            manager.checkpoint("unused.npz",x_mm=np.zeros((2,3)),variable_local_rows=[4,5],
                               q_float64=np.array([1.]),q_emitted_float32=np.array([1.]))

    def test_all_bank_reader_accepts_pathlike(self):
        line="7,100,0,7,1,23,5,inside,3,0,0,5,50,1,-1,0,0,0,1,0,0"
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"capture.csv"
            path.write_text(self._csv([line]).getvalue())
            rows,unresolved,summary=parse_all_bank_csv(path,node_count=10)
        self.assertEqual(len(rows),1)
        self.assertEqual(unresolved,[])
        self.assertEqual(summary["inside_constraints"],1)


    def test_normalized_quality_jacobian_on_nonzero_dofs(self):
        # A scalene fixture gives a unique longest edge, avoiding the max-edge kink.
        V=np.array([[[0.,0.,0.],[1.3,0.,0.],[0.,1.0,0.],[0.,0.,0.8]]])
        q,g=normalized_tet_quality_jacobian(V)
        self.assertGreater(float(q[0]),0.0)
        # Check real movable coordinates, not zero-gradient coordinates.
        for vertex,axis in [(1,0),(2,1),(3,2)]:
            self.assertGreater(abs(float(g[0,vertex,axis])),1e-3)
            h=1e-6
            plus=V.copy();minus=V.copy()
            plus[0,vertex,axis]+=h;minus[0,vertex,axis]-=h
            qp=normalized_tet_quality_jacobian(plus)[0][0]
            qm=normalized_tet_quality_jacobian(minus)[0][0]
            numeric=(qp-qm)/(2*h)
            self.assertAlmostEqual(float(g[0,vertex,axis]),float(numeric),delta=2e-7)
        with self.assertRaisesRegex(ValueError,"finite"):
            normalized_tet_quality_jacobian(np.full((1,4,3),np.nan))

    def test_layer_projection_mm_converts_float32_emitted_meter_geometry(self):
        outer=np.array([[0.,0.,0.002]],dtype=np.float32).astype(np.float64)
        inner=np.array([[0.,0.,0.0014]],dtype=np.float32).astype(np.float64)
        n=np.array([[0.,0.,1.]])
        projection=projected_layer_thickness_mm(outer,inner,n)
        self.assertAlmostEqual(float(projection[0]),0.6,places=5)
        floor=layer_floor_mm(np.array([0.0003]))
        self.assertAlmostEqual(float(floor[0]),0.06,places=12)
        self.assertGreaterEqual(float(projection[0]),float(floor[0]))

    def test_thickness_floor_units_and_collapse(self):
        floors=layer_floor_mm(np.array([0.0003,0.01,0.0]))
        np.testing.assert_allclose(floors,[0.06,0.10,0.0])
        with self.assertRaisesRegex(ValueError,"thickness"):
            layer_floor_mm(np.array([-1e-6]))

if __name__ == "__main__": unittest.main()
