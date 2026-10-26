import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

VERIFIER=Path(__file__).with_name("verify_construction_candidate_494.py")
spec=importlib.util.spec_from_file_location("verify494",VERIFIER)
verify494=importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify494)

class GuardCountContractTest(unittest.TestCase):
    def fixture(self, root):
        npz=root/"guard.npz"; report_path=root/"guard.json"
        np.savez(npz,
          skin_face_rows=np.array([1,2],dtype=np.int64),
          bank_body=np.array([4,4],dtype=np.int64),
          stable_bone_id=np.array([9,9],dtype=np.int64),
          bank_triangle_stable_id=np.array([20,21],dtype=np.int64),
          fixed_triangles_world_mm=np.zeros((2,3,3),dtype=np.float64))
        report_path.write_text("{}")
        hnpz=verify494.sha(npz); hreport=verify494.sha(report_path)
        base="base-sha"; faces="faces-sha"; points="points-sha"; owner="owner-sha"
        typed=[
          {"skin_face_row":1,"bank_body":4,"stable_bone_id":9,"bank_triangle_stable_id":20},
          {"skin_face_row":2,"bank_body":4,"stable_bone_id":9,"bank_triangle_stable_id":21}]
        intro={"skin_face_row":3,"bank_body":4,"stable_bone_id":9,
               "bank_triangle_stable_id":22,"current_crossing":True,"constraint_emitted":False}
        resolved={**typed[1],"current_crossing":False,"constraint_emitted":True}
        scope={"retained_guard_pairs":2,"313_rows":12,"prior_guard_pairs":1,
               "prior_guards_filtered_as_current":0,"resolved_pairs_added_as_guards":1,
               "resolved_pairs_retained":1,"current_pairs_removed_from_union":0,
               "union_pairs_before_filter":2,"introduced_current_pairs":1,
               "current_complete_pair_count":10,"previous_complete_pair_count":10,
               "current_complete_185_bank_audit_validated":True,
               "current_matches_embedded_037_faces":True,"same_topology_bytewise":True}
        report={"status":"COMPLETE_TYPED_INTRODUCED_SKIN_BONE_BARRIER_EXPORT",
          "geometry_or_native_admission":False,"base_candidate_sha256":base,
          "point_count":3,"face_count":5,"candidate_points_world_m_shape":[3,3],
          "candidate_points_world_m_array_sha256":points,"bank_count":185,"expected_bank_count":185,
          "introduced_pair_count":2,"typed_pairs":typed,
          "introduced_pairs_previous_to_current":[intro],
          "resolved_pairs_previous_to_current":[resolved],"removed_current_pairs":[],
          "construction_scope":scope}
        inp={"enabled":True,"construction_only":True,"geometry_or_native_admission":False,
          "report_status":report["status"],"npz_path":str(npz),"npz_sha256":hnpz,
          "report_path":str(report_path),"report_sha256":hreport,"base_candidate_sha256":base,
          "candidate_sha256":base,"point_count":3,"face_count":5,"base_faces_array_sha256":faces,
          "typed_pair_count":2,"linear_row_count":12,"units":"world-frame millimetres",
          "313_owner_sha256":owner,"row_margin_fraction":0.25}
        fixed={"passed":True,"certificate":"UNCLAMPED_313_FIXED_BONE_ALPHA_ROWS",
          "row_count":12,"checked_rows":12,"violated_rows":0,"max_violation_mm":0.0,
          "tolerance_mm":2e-8,"anchor_max_abs_scale":0.0,
          "lp_positive_lower_clamped_rows":0,"lp_max_positive_lower_clamp_mm":0.0,
          "input_receipt":inp}
        fit={"enabled":True,"geometry_or_native_admission":False,"base_candidate_sha256":base,
          "base_faces_array_sha256":faces,"npz_sha256":hnpz,"report_sha256":hreport,
          "npz_path":str(npz),"point_count":3,"face_count":5,"typed_pair_count":2,
          "units":"world-frame millimetres","report_status":report["status"],"owner_sha256":owner}
        arrays=np.load(npz,allow_pickle=False)
        kwargs={"npz_path":npz,"report_path":report_path,"base_sha":base,
          "point_count":3,"face_count":5,"base_faces_sha":faces,"base_points_sha":points,
          "base_surface_count":10}
        return fixed,report,arrays,fit,kwargs

    def test_actual_counts_are_derived_from_paired_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            fixed,report,arrays,fit,kwargs=self.fixture(Path(td))
            got=verify494.validate_fixed_guard_bundle(fixed,report,arrays,fit,**kwargs)
            self.assertEqual(got["typed_pairs"],2)
            self.assertEqual(got["linear_rows"],12)

    def test_wrong_certificate_row_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            fixed,report,arrays,fit,kwargs=self.fixture(Path(td))
            bad=copy.deepcopy(fixed); bad["row_count"]=13
            with self.assertRaisesRegex(AssertionError,"fixed-bone alpha guard certificate failed"):
                verify494.validate_fixed_guard_bundle(bad,report,arrays,fit,**kwargs)

if __name__=="__main__":
    unittest.main()
