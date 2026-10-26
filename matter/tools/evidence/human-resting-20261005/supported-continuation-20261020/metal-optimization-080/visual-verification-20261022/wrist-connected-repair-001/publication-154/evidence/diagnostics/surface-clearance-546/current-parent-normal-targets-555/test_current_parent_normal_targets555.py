import sys
from pathlib import Path
import unittest
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import current_parent_normal_helper555 as helper
import run_current_parent_normal_targets555 as runner
import recover_current_parent_normal_outputs555 as recovery


class CurrentParentNormalTests(unittest.TestCase):
    def test_centroid_uses_orientation_signed_current_parent_face(self):
        pp=np.array([[0,0,0],[2,0,0],[0,2,0]],np.float32)
        cp=np.vstack([pp,np.array([[2/3,2/3,0]],np.float32)])
        F=np.array([[0,1,2]],np.int64)
        n,prov=helper.parent_normal_for_node(3,parent_points=pp,parent_faces=F,child_points=cp,
            node_kind=np.array([0,0,0,2],np.int8),parent_face_indptr=np.array([0,0,0,0,1]),
            parent_face_rows=np.array([0]),parent_face_barycentric=np.array([[1/3,1/3,1/3]]),
            outward_sign=-1)
        np.testing.assert_allclose(n,[0,0,-1],atol=1e-14)
        self.assertEqual(prov["method"],"immediate-parent-face-area-weighted-oriented-normal")

    def test_edge_midpoint_is_area_weighted_two_face_bisector(self):
        pp=np.array([[0,0,0],[2,0,0],[0,1,0],[0,0,4]],np.float32)
        cp=np.vstack([pp,np.array([[1,0,0]],np.float32)])
        F=np.array([[0,1,2],[1,0,3]],np.int64)
        n,prov=helper.parent_normal_for_node(4,parent_points=pp,parent_faces=F,child_points=cp,
            node_kind=np.array([0,0,0,0,1],np.int8),parent_face_indptr=np.array([0,0,0,0,0,2]),
            parent_face_rows=np.array([0,1]),
            parent_face_barycentric=np.array([[.5,.5,0],[.5,.5,0]]),outward_sign=-1)
        expected=np.array([0,-4,-1],float);expected/=np.linalg.norm(expected)
        np.testing.assert_allclose(n,expected,atol=1e-14)
        self.assertEqual(prov["parent_face_rows"],[0,1])

    def test_prefix_and_unknown_kind_reject(self):
        pp=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
        cp=np.vstack([pp,np.array([[1/3,1/3,0]],np.float32)])
        F=np.array([[0,1,2]],np.int64); kinds=np.array([0,0,0,2],np.int8)
        kwargs=dict(parent_points=pp,parent_faces=F,child_points=cp,node_kind=kinds,
            parent_face_indptr=np.array([0,0,0,0,1]),parent_face_rows=np.array([0]),
            parent_face_barycentric=np.array([[1/3,1/3,1/3]]))
        with self.assertRaisesRegex(ValueError,"appended"):
            helper.parent_normal_field([2],old_prefix_count=3,**kwargs,outward_sign=1)
        with self.assertRaisesRegex(ValueError,"not an appended"):
            helper.parent_normal_for_node(1,**kwargs,outward_sign=1)

    def test_wrong_barycentric_point_rejected(self):
        pp=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
        cp=np.vstack([pp,np.array([[.4,.3,0]],np.float32)])
        with self.assertRaisesRegex(ValueError,"does not match"):
            helper.parent_normal_for_node(3,parent_points=pp,parent_faces=np.array([[0,1,2]]),
                child_points=cp,node_kind=np.array([0,0,0,2]),parent_face_indptr=np.array([0,0,0,0,1]),
                parent_face_rows=np.array([0]),parent_face_barycentric=np.array([[1/3,1/3,1/3]]),outward_sign=1)

    def test_duplicate_or_degenerate_ancestry_rejected(self):
        pp=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
        cp=np.vstack([pp,np.array([[.5,.5,0]],np.float32)])
        F=np.array([[0,1,2],[0,1,3]],np.int64)
        args=dict(parent_points=pp,parent_faces=F,child_points=cp,node_kind=np.array([0,0,0,0,1]),
            parent_face_indptr=np.array([0,0,0,0,0,2]),parent_face_rows=np.array([0,0]),
            parent_face_barycentric=np.array([[.5,.5,0],[.5,.5,0]]),outward_sign=1)
        with self.assertRaisesRegex(ValueError,"duplicate"):
            helper.parent_normal_for_node(4,**args)



    def test_pinned_303_orientation_owner_resolves_current_parent_sign(self):
        sign=runner.resolve_coarse_outward_sign(
            runner.SOURCE303,runner.ORIENT)
        self.assertEqual(sign,-1)

    def test_actual_completed_545_run_uses_signed_parent_vectors(self):
        root=runner.ROOT
        base=root/"D/surface-clearance-546/round-020-retained-sat552"
        refined=root/"D/surface-clearance-546/connected-source-refinement-553/refine-frozen30886-002"
        out=root/"D/surface-clearance-546/current-parent-normal-targets-555/targets-001"
        report,arrays,rows,snorm,sprov,current,cprov,sign=recovery.verify_completed_outputs(
            base/"candidate.npz",base/"full-check/audit-130/summary.json",
            refined/"candidate.npz",refined/"audit-130/summary.json",
            refined/"audit-130/query",out/"nearest-union-exits.json",out/"nearest-union-exits.npz")
        self.assertEqual(len(rows),1248)
        self.assertEqual(sign,-1)
        self.assertEqual(set(rows),set(current))
        self.assertEqual(set(rows),set(snorm))
        np.testing.assert_array_equal(arrays["proposal_direction_world_unit"],
                                      np.asarray([current[r] for r in rows]))
        self.assertEqual(report["outputs"]["npz_sha256"],recovery.sha(out/"nearest-union-exits.npz"))

    def test_builder_failure_before_outputs_preserves_original_error(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as d:
            report=Path(d)/"targets.json"; arrays=Path(d)/"targets.npz"
            def fail(_args): raise ValueError("simulated builder preflight failure")
            with self.assertRaisesRegex(ValueError,"simulated builder preflight failure"):
                runner.invoke_and_annotate_builder(fail,[],report,arrays,
                    source_normals={},source_provenance={},parent_normals={},parent_provenance={},
                    orientation_sign=-1,input_metadata={})
            self.assertFalse(report.exists());self.assertFalse(arrays.exists())

    def test_runtime_postprocess_refreshes_npz_hash_and_keeps_status(self):
        import json
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as d:
            report_path=Path(d)/"targets.json"; arrays_path=Path(d)/"targets.npz"
            def build(_args):
                record={"query_row":4,"status":"PASS_FIRST_UNION_EXIT",
                    "target_exactly_outside_all185":True,"exact_endpoint_bank_count":185,
                    "proposal_direction_world_unit":[1.,0.,0.],
                    "source_surface_normal_world_unit":[1.,0.,0.]}
                report_path.write_text(json.dumps({"targets":[record],"inputs":{},"outputs":{},
                    "status":"PASS_FIRST_UNION_EXIT_TARGETS_ONLY"}))
                np.savez_compressed(arrays_path,query_rows=np.array([4]),
                    proposal_direction_world_unit=np.array([[1.,0,0]]),
                    source_surface_normal_world_unit=np.array([[1.,0,0]]),
                    depth_covector_world_unit=np.full((1,3),np.nan),
                    target_exactly_outside_all185=np.array([True]))
            result=runner.invoke_and_annotate_builder(build,[],report_path,arrays_path,
                source_normals={4:np.array([0.,1,0])},source_provenance={4:{"owner":"303"}},
                parent_normals={4:np.array([1.,0,0])},parent_provenance={4:{"parent":[7]}},
                orientation_sign=-1,input_metadata={"parent_sha256":"abc"})
            self.assertEqual(result["status"],"PASS_FIRST_UNION_EXIT_TARGETS_ONLY")
            self.assertEqual(result["direction_mode"],"current-parent-normal")
            self.assertEqual(result["outputs"]["npz_sha256"],runner.sha(arrays_path))
            self.assertEqual(result["outputs"]["current_parent_normal_postprocessor_sha256"],runner.sha(runner.RUNNER))
            self.assertEqual(result["inputs"]["parent_sha256"],"abc")
            with np.load(arrays_path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z["source_surface_normal_world_unit"],[[0,1,0]])
                np.testing.assert_array_equal(z["proposal_direction_world_unit"],[[1,0,0]])

    def test_output_decoration_keeps_reference_and_proposal_separate(self):
        report={"targets":[{"query_row":4,"status":"INCOMPLETE","target_exactly_outside_all185":False}]}
        arrays={"query_rows":np.array([4]),"proposal_direction_world_unit":np.array([[1.,0,0]]),
                "source_surface_normal_world_unit":np.array([[1.,0,0]]),
                "depth_covector_world_unit":np.full((1,3),np.nan),
                "target_exactly_outside_all185":np.array([False])}
        decorated,z=runner.decorate_target_outputs(report,arrays,
            source_normals={4:np.array([0.,1,0])},source_provenance={4:{"owner":"303"}},
            parent_normals={4:np.array([1.,0,0])},parent_provenance={4:{"parent":[7]}},
            orientation_sign=-1)
        np.testing.assert_array_equal(z["proposal_direction_world_unit"],[[1,0,0]])
        np.testing.assert_array_equal(z["source_surface_normal_world_unit"],[[0,1,0]])
        np.testing.assert_array_equal(z["current_parent_normal_world_unit"],[[1,0,0]])
        self.assertFalse(z["depth_covector_used_mask"][0])
        self.assertEqual(decorated["direction_mode"],"current-parent-normal")
        self.assertEqual(decorated["targets"][0]["status"],"INCOMPLETE")

    def test_output_decoration_rejects_fabricated_depth_covectors(self):
        report={"targets":[{"query_row":4}]}
        arrays={"query_rows":np.array([4]),"proposal_direction_world_unit":np.array([[1.,0,0]]),
                "source_surface_normal_world_unit":np.array([[0.,1,0]]),
                "depth_covector_world_unit":np.array([[0.,0.,1.]])}
        with self.assertRaisesRegex(ValueError,"fabricated"):
            runner.decorate_target_outputs(report,arrays,source_normals={4:np.array([0.,1,0])},
                source_provenance={4:{}},parent_normals={4:np.array([1.,0,0])},
                parent_provenance={4:{}},orientation_sign=-1)


if __name__=="__main__":unittest.main(verbosity=2)

