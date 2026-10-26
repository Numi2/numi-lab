import unittest
import numpy as np
from merge_introduced_skin_bone_guards556 import merge_pair_arrays, pair_map


def payload(rows):
    # face, body, bone, stable triangle ID, x offset for the triangle.
    return {
        "skin_face_rows":np.asarray([r[0] for r in rows],dtype=np.int64),
        "bank_body":np.asarray([r[1] for r in rows],dtype=np.int64),
        "stable_bone_id":np.asarray([r[2] for r in rows],dtype=np.int64),
        "bank_triangle_stable_id":np.asarray([r[3] for r in rows],dtype=np.int64),
        "fixed_triangles_world_mm":np.asarray([
            [[r[4],0.,0.],[r[4],1.,0.],[r[4],0.,1.]] for r in rows
        ],dtype=np.float64).reshape(-1,3,3),
    }


def fake_canonical(pairs):
    rows=[(face,body,bone,tri,float(tri)) for body,bone,face,tri in pairs]
    rows.sort(key=lambda r:(r[1],r[2],r[0],r[3]))
    return payload(rows)


class MergeGuard556Tests(unittest.TestCase):
    def test_deduplicates_identity_and_emits_canonical_typed_order(self):
        old=payload([(4,43,16,133,133.),(8,65,99,207,207.)])
        new=payload([(8,65,99,207,207.),(3,43,16,88,88.)])
        merged,meta=merge_pair_arrays(old,new,10,fake_canonical)
        got=pair_map(merged,10)
        self.assertEqual(set(got),{(4,43,16,133),(8,65,99,207),(3,43,16,88)})
        self.assertEqual(meta["overlap_pair_count_deduplicated"],1)
        self.assertEqual(meta["merged_pair_count"],3)
        self.assertEqual(list(zip(merged["bank_body"],merged["stable_bone_id"],
                                  merged["skin_face_rows"],merged["bank_triangle_stable_id"])),
                         [(43,16,3,88),(43,16,4,133),(65,99,8,207)])

    def test_conflicting_bank_geometry_for_duplicate_identity_rejects(self):
        with self.assertRaisesRegex(ValueError,"bank geometry differs"):
            merge_pair_arrays(payload([(4,43,16,133,133.)]),
                              payload([(4,43,16,133,134.)]),10,fake_canonical)

    def test_duplicate_and_out_of_range_identities_reject(self):
        with self.assertRaisesRegex(ValueError,"duplicate typed pair"):
            pair_map(payload([(4,43,16,133,133.),(4,43,16,133,133.)]),10)
        with self.assertRaisesRegex(ValueError,"outside"):
            pair_map(payload([(10,43,16,133,133.)]),10)

    def test_noninteger_ids_and_wrong_triangle_dtype_reject(self):
        bad=payload([(4,43,16,133,133.)]);bad["stable_bone_id"]=bad["stable_bone_id"].astype(np.float64)
        with self.assertRaisesRegex(ValueError,"integer array"):pair_map(bad,10)
        bad=payload([(4,43,16,133,133.)]);bad["fixed_triangles_world_mm"]=bad["fixed_triangles_world_mm"].astype(np.float32)
        with self.assertRaisesRegex(ValueError,"Float64"):pair_map(bad,10)


if __name__=="__main__":unittest.main()

