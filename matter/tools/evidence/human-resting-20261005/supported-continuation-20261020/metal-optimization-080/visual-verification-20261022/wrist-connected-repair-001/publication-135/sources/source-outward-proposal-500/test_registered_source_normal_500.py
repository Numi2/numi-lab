from __future__ import annotations
import hashlib, importlib.util, json, sys, unittest
from pathlib import Path
import numpy as np

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
HELPER=ROOT/"D/source-outward-proposal-500/build_outward_nearest_union_targets_500.py"
CANDIDATE=ROOT/"D/refreshed-connected-continuation-492/run-001/iteration-002/check/backtrack-373/pass-013/candidate.npz"
BASE=ROOT/"full-connected-clearance-150/run-028-current-target-connected-fit-001/candidate.npz"
ORIENTATION=ROOT/"digit-source-diagnosis-110/source-path-orientation-audit-112.json"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_helper():
    spec=importlib.util.spec_from_file_location("helper500_test",HELPER)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

class RegisteredSourceNormal500Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.m=load_helper()

    def fixture(self, *, marker=89, parent=89, ancestry=(89,)):
        ptr=[0,len(ancestry)]
        candidate={"face_kind":np.array([0],np.int8),
          "parent_fine_detail_face_row":np.array([-1],np.int32),
          "parent_coarse_face_row":np.array([parent],np.int32),
          "parent_source_face_marker":np.array([marker],np.int32),
          "face_source_ancestry_indptr":np.asarray(ptr,np.int64),
          "face_source_ancestry_face_rows":np.asarray(ancestry,np.int64)}
        base={"source_O_faces":np.tile(np.array([[0,1,2]],np.int64),(90,1)),
          "points_world_m":np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])}
        return candidate,base,{"coarse_outward_sign":-1}

    def test_kind0_requires_single_matching_typed_coarse_ancestry(self):
        c,b,o=self.fixture()
        n,prov=self.m.registered_source_outward_face_normal(0,c,{},b,o)
        np.testing.assert_allclose(n,[0.,0.,-1.],rtol=0,atol=1e-15)
        self.assertEqual(prov,{"kind":"unique_coarse_source_face_ancestry_kind0","row":89,"output_face_kind":0})
        for bad in [self.fixture(ancestry=(88,))[0], self.fixture(marker=88)[0], self.fixture(parent=88)[0], self.fixture(ancestry=(89,90))[0]]:
            with self.assertRaises(ValueError): self.m.registered_source_outward_face_normal(0,bad,{},b,o)

    def test_kind0_missing_ancestry_fails_closed(self):
        c,b,o=self.fixture(); c.pop("face_source_ancestry_indptr"); c.pop("face_source_ancestry_face_rows")
        with self.assertRaisesRegex(ValueError,"lacks typed coarse-source ancestry"):
            self.m.registered_source_outward_face_normal(0,c,{},b,o)

    def test_actual_failed_face146_normal_matches_direct_kind1_coarse_resolution(self):
        self.assertEqual(sha(CANDIDATE),"aef14b798367c7a8e76aeb605942453022eefe68670019044d59b9a58f3444a1")
        self.assertEqual(sha(BASE),"2f93d0159f5e1ece1fa377da71e52e4edb7ee60f4b61e483a0390733ca404c11")
        self.assertEqual(sha(ORIENTATION),"e43908188f134182635ac36a9a1602abad1918df374a04f3f7cc9eb1812d7497")
        orientation=self.m.validate_source_outward_orientation(json.loads(ORIENTATION.read_text()))
        self.assertEqual(int(orientation["coarse_outward_sign"]),-1)
        with np.load(CANDIDATE,allow_pickle=False) as z, np.load(BASE,allow_pickle=False) as b:
            self.assertEqual(int(z["face_kind"][146]),0)
            self.assertEqual(int(z["parent_coarse_face_row"][146]),89)
            self.assertEqual(int(z["parent_source_face_marker"][146]),89)
            ptr=z["face_source_ancestry_indptr"]
            self.assertEqual(z["face_source_ancestry_face_rows"][int(ptr[146]):int(ptr[147])].tolist(),[89])
            n0,p0=self.m.registered_source_outward_face_normal(146,z,{},b,orientation)
            direct={k:np.asarray(z[k]).copy() for k in ["face_kind","parent_fine_detail_face_row","parent_coarse_face_row"]}
            direct["face_kind"][146]=1
            n1,p1=self.m.registered_source_outward_face_normal(146,direct,{},b,orientation)
            np.testing.assert_allclose(n0,n1,rtol=0,atol=1e-15)
            self.assertEqual(p0["row"],89); self.assertEqual(p1,{"kind":"original_coarse_O_face","row":89})

if __name__=="__main__": unittest.main(verbosity=2)
