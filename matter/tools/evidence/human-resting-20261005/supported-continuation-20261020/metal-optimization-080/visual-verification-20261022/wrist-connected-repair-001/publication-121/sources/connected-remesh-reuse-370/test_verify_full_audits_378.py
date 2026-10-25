from __future__ import annotations
import importlib.util, sys, unittest
from pathlib import Path
import numpy as np

path=Path(__file__).with_name("verify_full_audits_378.py")
spec=importlib.util.spec_from_file_location("verify_full_audits_378",path)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)

class Verifier378Tests(unittest.TestCase):
    def test_all_implicated_operations_are_excluded_by_either_face(self):
        selector={"operations":[
            {"selected":True,"face_rows":[4,7]},
            {"selected":True,"face_rows":[8,9]},
            {"selected":True,"face_rows":[12,13]},
            {"selected":False,"face_rows":[14,15]},
        ]}
        safe,excluded,rows=mod.safe_operations(selector,np.asarray([[7,10],[20,8]],dtype=np.int64))
        self.assertEqual([op["face_rows"] for op in safe],[[12,13]])
        self.assertEqual([op["face_rows"] for op in excluded],[[4,7],[8,9]])
        self.assertEqual(rows,{7,8,10,20})

    def test_self_witness_shape_must_be_exact_integer_n_by_two(self):
        for a in (np.asarray([1,2]),np.asarray([[1,2,3]]),np.asarray([[1.0,2.0]])):
            with self.assertRaises(mod.VerificationError):
                mod.safe_operations({"operations":[]},a)

    def test_full_report_rejects_stale_candidate_sha(self):
        report=self.report()
        with self.assertRaisesRegex(mod.VerificationError,"candidate SHA"):
            mod.validate_full_159(report,"different",1)

    def test_full_report_rejects_stale_face_shape(self):
        with self.assertRaisesRegex(mod.VerificationError,"face-domain"):
            mod.validate_full_159(self.report(),"base",2)

    def test_full_report_rejects_missing_typed_bank(self):
        report=self.report();report["banks"].pop();report["bank_count"]=184
        with self.assertRaises(mod.VerificationError):
            mod.validate_full_159(report,"base",1)

    def test_full_report_requires_explicit_null_bank_exclusion(self):
        report=self.report();report["banks"][1]["skin_aabb_active"]=1
        with self.assertRaisesRegex(mod.VerificationError,"AABB exclusion"):
            mod.validate_full_159(report,"base",1)

    def test_exact_pair_identity_set_rejects_duplicate_pair(self):
        report=self.report()
        report["banks"][0]["result"]={"count":2,"triangle_pairs":[[0,1],[0,1]]}
        with self.assertRaisesRegex(mod.VerificationError,"duplicate exact pair"):
            mod.validate_full_159(report,"base",1)

    @staticmethod
    def report():
        banks=[]
        for i in range(185):
            body,bone=1+i,1000+i
            if i==0:
                result={"count":0,"triangle_pairs":[]}
                row={"body":body,"stable_bone_id":bone,"skin_faces":1,"bank_faces":1,"result":result}
            else:
                row={"body":body,"stable_bone_id":bone,"skin_faces":1,"bank_faces":1,
                     "skin_aabb_active":0,"bank_aabb_active":0,"result":None}
            banks.append(row)
        return {"status":"COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT","complete":True,
                "bank_count":185,"expected_bank_count":185,"candidate_sha256":"base",
                "surface_face_count":1,"banks":banks}

if __name__=="__main__": unittest.main(verbosity=2)
