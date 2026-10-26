from __future__ import annotations
import copy
import os
import unittest
from pathlib import Path
import numpy as np
from refresh_guards_multi_parent_546 import validate_ancestry


def sample():
    points = np.asarray([[0,0,0],[1,0,0],[0,1,0]], np.float32)
    ids = lambda x: np.asarray(x, np.int64)
    cand = {
        "points_world_m": points,
        "source_O_faces": np.asarray([[0,1,2]], np.int32),
        "area_reference_037_points_world_m_f64": points.astype(np.float64),
        "area_reference_037_source_O_faces": np.asarray([[0,1,2]], np.int32),
        "face_kind": np.asarray([2], np.int32),
        "parent_fine_detail_face_row": ids([-1]),
        "parent_fine_subface_row": ids([-1]),
        "parent_coarse_face_row": ids([-1]),
        "parent_source_face_marker": ids([-1]),
        "refinement_face_parent_face_row": ids([-1]),
        "refinement_face_parent_barycentric": np.full((1,3,3), np.nan),
        "face_source_ancestry_indptr": ids([0,2]),
        "face_source_ancestry_face_rows": ids([4,5]),
        "face_fine_subface_ancestry_indptr": ids([0,2]),
        "face_fine_subface_ancestry_rows": ids([8,9]),
        "face_fine_detail_ancestry_indptr": ids([0,2]),
        "face_fine_detail_ancestry_rows": ids([4,5]),
        "refinement_face_parent_ancestry_indptr": ids([0,2]),
        "refinement_face_parent_ancestry_face_rows": ids([0,1]),
        "parent_fine_detail_face_row_semantics": np.asarray(["original NHSKIN face row; -1 for multi-source ancestry"]),
        "parent_fine_subface_row_semantics": np.asarray(["run-026 fine_subfaces row; -1 when not unique"]),
        "face_fine_detail_ancestry_semantics": np.asarray(["CSR union of source fine detail face rows; singular row -1 for cross-detail flips"]),
        "refinement_face_parent_ancestry_semantics": np.asarray(["CSR union of immediate parent face rows; singular row -1 and singular barycentrics NaN when multiple parents apply"]),
        "refinement_face_parent_barycentric_semantics": np.asarray(["single-parent affine map only; NaN explicitly marks multi-parent patch flips"]),
    }
    parent = {k: np.array(v, copy=True) for k,v in cand.items()}
    parent["source_O_faces"] = np.asarray([[0,1,2],[1,3,2]], np.int32)
    parent["face_source_ancestry_face_rows"] = ids([4,5])
    parent["face_fine_subface_ancestry_rows"] = ids([8,9])
    parent["face_fine_detail_ancestry_rows"] = ids([4,5])
    surface = {"banks":[{"result":{"triangle_pairs":[[0,7]]}}]}
    return cand,parent,surface


class MultiParentRefreshTests(unittest.TestCase):
    def test_accepts_explicit_two_parent_union(self):
        c,p,s=sample()
        r=validate_ancestry(c,p,s)
        self.assertTrue(r["crossing_fine_detail_ancestry_union_validated"])
        self.assertEqual(r["crossing_faces_with_multi_source_union"],1)
        self.assertEqual(r["crossing_faces_with_multi_parent_csr"],1)

    def test_rejects_scalarized_detail_parent(self):
        c,p,s=sample();c["parent_fine_detail_face_row"][0]=4
        with self.assertRaisesRegex(ValueError,"fine-detail scalar"):
            validate_ancestry(c,p,s)

    def test_rejects_unregistered_source_id(self):
        c,p,s=sample();c["face_source_ancestry_face_rows"][1]=99
        with self.assertRaisesRegex(ValueError,"absent from pinned registered parent domain"):
            validate_ancestry(c,p,s)

    def test_rejects_parent_id_outside_face_domain(self):
        c,p,s=sample();c["refinement_face_parent_ancestry_face_rows"][1]=2
        with self.assertRaisesRegex(ValueError,"outside registered row range"):
            validate_ancestry(c,p,s)

    def test_single_parent_affine_map_and_scalar(self):
        c,p,s=sample()
        c["face_source_ancestry_indptr"]=np.array([0,1]);c["face_source_ancestry_face_rows"]=np.array([4])
        c["face_fine_subface_ancestry_indptr"]=np.array([0,1]);c["face_fine_subface_ancestry_rows"]=np.array([8])
        c["face_fine_detail_ancestry_indptr"]=np.array([0,1]);c["face_fine_detail_ancestry_rows"]=np.array([4])
        c["refinement_face_parent_ancestry_indptr"]=np.array([0,1]);c["refinement_face_parent_ancestry_face_rows"]=np.array([1])
        c["parent_fine_detail_face_row"][0]=4;c["parent_fine_subface_row"][0]=8
        c["parent_source_face_marker"][0]=4;c["refinement_face_parent_face_row"][0]=1
        c["refinement_face_parent_barycentric"][0]=np.eye(3)
        r=validate_ancestry(c,p,s)
        self.assertTrue(r["all_scalar_summaries_match_csr_unions"])

    def test_optional_real_candidate(self):
        path=os.environ.get("MP546_CANDIDATE")
        if not path:return
        with np.load(path,allow_pickle=False) as z:c={k:np.asarray(z[k]).copy() for k in z.files}
        surface_path=Path(os.environ["MP546_AUDIT159"])
        import json
        r=validate_ancestry(c,c,json.loads(surface_path.read_text()))
        self.assertGreater(r["crossing_face_count"],0)
        self.assertTrue(r["crossing_fine_detail_ancestry_union_validated"])


if __name__=="__main__":
    unittest.main()
