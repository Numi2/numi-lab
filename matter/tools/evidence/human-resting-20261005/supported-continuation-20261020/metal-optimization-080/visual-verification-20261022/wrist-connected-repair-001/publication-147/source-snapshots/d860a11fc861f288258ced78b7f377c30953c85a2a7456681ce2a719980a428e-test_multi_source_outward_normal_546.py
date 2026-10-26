from __future__ import annotations
import importlib.util
import json
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
EXT = ROOT / "D/source-outward-proposal-500/build_outward_targets_multi_source_546.py"
CANDIDATE = ROOT / "D/surface-clearance-546/topology/round-006-wide-chain-025/round-002/candidate.npz"
DETAIL = ROOT / "publication-072/evidence/run-006-direct-detail-p1-075/direct-detail-p1-075.npz"
ORIENTATION = ROOT / "digit-source-diagnosis-110/source-path-orientation-audit-112.json"
FACES = (11827, 14133, 14730, 15305)
EXPECTED_IDS = {11827:[5120,5135], 14133:[5529,5530], 14730:[6103,6135], 15305:[6103,6135]}


def load_extension():
    spec=importlib.util.spec_from_file_location("multi_source_546_test",EXT)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod


def fixture(second_vertices=None):
    if second_vertices is None:
        # Equal-area source normals +Z and +X.
        second_vertices=[[0,0,0],[0,1,0],[0,0,1]]
    points=np.asarray([[0,0,0],[1,0,0],[0,1,0],*second_vertices],np.float64)
    c={
        "face_kind":np.asarray([2],np.int8),
        "parent_fine_detail_face_row":np.asarray([-1],np.int32),
        "parent_fine_subface_row":np.asarray([-1],np.int32),
        "parent_coarse_face_row":np.asarray([-1],np.int32),
        "parent_source_face_marker":np.asarray([-1],np.int32),
        "face_source_ancestry_indptr":np.asarray([0,2],np.int64),
        "face_source_ancestry_face_rows":np.asarray([0,1],np.int64),
        "face_fine_detail_ancestry_indptr":np.asarray([0,2],np.int64),
        "face_fine_detail_ancestry_rows":np.asarray([0,1],np.int64),
        "face_fine_subface_ancestry_indptr":np.asarray([0,2],np.int64),
        "face_fine_subface_ancestry_rows":np.asarray([10,11],np.int64),
        "selected_fine_subface_rows":np.asarray([10,11],np.int64),
        "parent_fine_detail_face_row_semantics":np.asarray(["original NHSKIN face row; -1 for multi-source ancestry"]),
        "face_fine_detail_ancestry_semantics":np.asarray(["CSR union of source fine detail face rows; singular row -1 for cross-detail flips"]),
    }
    d={"source_surface_faces_detail_rows":np.asarray([[0,1,2],[3,4,5]],np.int64),
       "detail_baseline_world_m":points}
    return c,d,{"fine_outward_sign":1}


class MultiSourceOutwardNormal546Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.m=load_extension()

    def test_area_weighted_union_normal_and_complete_provenance(self):
        c,d,o=fixture();n,p=self.m.multi_parent_source_outward_normal(0,c,d,o)
        np.testing.assert_allclose(n,[2**-.5,0,2**-.5],rtol=0,atol=1e-14)
        self.assertEqual(p["registered_fine_detail_face_rows"],[0,1])
        self.assertEqual(p["registered_source_face_rows"],[0,1])
        self.assertEqual(p["registered_fine_subface_rows"],[10,11])
        self.assertAlmostEqual(sum(p["area_weights"]),1.0,places=14)
        self.assertGreater(p["minimum_open_hemisphere_dot"],0)

    def test_rejects_opposing_or_cancelled_parent_normals(self):
        c,d,o=fixture(second_vertices=[[0,1,0],[1,0,0],[0,0,0]])
        with self.assertRaisesRegex(ValueError,"cancel|hemisphere"):
            self.m.multi_parent_source_outward_normal(0,c,d,o)

    def test_rejects_non_sentinel_scalar_parent(self):
        c,d,o=fixture();c["parent_fine_detail_face_row"][0]=0
        with self.assertRaisesRegex(ValueError,"-1 scalar sentinels"):
            self.m.multi_parent_source_outward_normal(0,c,d,o)

    def test_rejects_detail_union_outside_source_union(self):
        c,d,o=fixture();c["face_source_ancestry_indptr"]=np.asarray([0,1]);c["face_source_ancestry_face_rows"]=np.asarray([0])
        with self.assertRaisesRegex(ValueError,"union and registered source-face union disagree"):
            self.m.multi_parent_source_outward_normal(0,c,d,o)

    def test_rejects_unregistered_subface_id(self):
        c,d,o=fixture();c["face_fine_subface_ancestry_rows"][1]=99
        with self.assertRaisesRegex(ValueError,"outside registered selected subface domain"):
            self.m.multi_parent_source_outward_normal(0,c,d,o)

    def test_real_four_multi_source_faces_use_all_typed_parents(self):
        owner=self.m.load_module(self.m.OWNER500)
        with np.load(CANDIDATE,allow_pickle=False) as zz:c={k:np.asarray(zz[k]).copy() for k in zz.files}
        with np.load(DETAIL,allow_pickle=False) as d:
            orientation_record=json.loads(ORIENTATION.read_text())
            self.m.sha(self.m.OWNER500)==self.m.OWNER500_SHA
            owner.validate_source_orientation_input_pins(orientation_record)
            orient=owner.validate_source_outward_orientation(orientation_record)
            for face in FACES:
                expected=EXPECTED_IDS[face]
                self.assertEqual(c["face_kind"][face],2)
                self.assertEqual(int(c["parent_fine_detail_face_row"][face]),-1)
                self.assertEqual(int(c["parent_source_face_marker"][face]),-1)
                n,p=self.m.multi_parent_source_outward_normal(face,c,d,orient)
                self.assertEqual(p["registered_fine_detail_face_rows"],expected)
                self.assertEqual(len(p["area_weights"]),2)
                self.assertGreater(p["minimum_open_hemisphere_dot"],0)
                self.assertAlmostEqual(float(np.linalg.norm(n)),1.0,places=14)

    def test_single_parent_dispatch_uses_original_500_function(self):
        owner=self.m.load_module(self.m.OWNER500)
        original=owner.registered_source_outward_face_normal
        wrapped_original=self.m.install_extension(owner)
        self.assertIs(wrapped_original,original)
        with np.load(CANDIDATE,allow_pickle=False) as zz:c={k:np.asarray(zz[k]).copy() for k in zz.files}
        with np.load(DETAIL,allow_pickle=False) as d, np.load(owner.BASE,allow_pickle=False) as b:
            orientation_record=json.loads(ORIENTATION.read_text())
            orient=owner.validate_source_outward_orientation(orientation_record)
            row=next(i for i in range(len(c["face_kind"])) if int(c["face_kind"][i])==2 and int(c["parent_fine_detail_face_row"][i])>=0)
            expected=original(row,c,d,b,orient)
            actual=owner.registered_source_outward_face_normal(row,c,d,b,orient)
            np.testing.assert_array_equal(actual[0],expected[0])
            self.assertEqual(actual[1],expected[1])


if __name__=="__main__": unittest.main(verbosity=2)
