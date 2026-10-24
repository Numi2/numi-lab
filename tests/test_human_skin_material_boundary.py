import unittest
import numpy as np
from human_skin_material_boundary import certify_material_boundaries as certify


class MaterialBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.x=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1],[0,0,-.5]],np.float32)
        self.t=np.array([[0,1,2,3],[0,2,1,5],[5,2,1,4]],np.int64)
        self.m=np.array([1,2,2],np.uint32)
        self.o=np.array([[1,2,3],[0,3,2],[0,1,3]],np.int64)
        self.i=np.array([[0,2,1]],np.int64)
        self.d=np.array([[0,5,1],[0,2,5],[2,1,4],[5,4,1],[5,2,4]],np.int64)

    def run_case(self, **kw):
        args=dict(points=self.x,tetrahedra=self.t,material_ids=self.m,
                  outer_faces=self.o,interface_faces=self.i,deep_faces=self.d)
        args.update(kw)
        return certify(**args)

    def test_independent_boundary_sizes_and_conservative_volume(self):
        r=self.run_case()
        self.assertEqual([len(r[k]) for k in ("outer_faces","interface_faces","deep_faces")],[3,1,5])
        self.assertTrue(r["boundary_incidence_pass"])
        self.assertFalse(r["native_admission"])
        self.assertAlmostEqual(r["material_volume_m3"]["1"],1/6)
        self.assertAlmostEqual(r["material_volume_m3"]["2"],1/6)
        self.assertFalse(r["construction_only_facets_remaining"])

    def test_permutation_preserves_canonical_faces(self):
        a=self.run_case()
        b=self.run_case(tetrahedra=self.t[[2,0,1]],material_ids=self.m[[2,0,1]],
                        outer_faces=self.o[::-1],deep_faces=self.d[::-1])
        for k in ("outer_faces","interface_faces","deep_faces"):
            np.testing.assert_array_equal(a[k],b[k])

    def test_construction_cannot_masquerade_as_physical_surface(self):
        r=self.run_case(outer_faces=self.o[1:],construction_faces=self.o[:1])
        self.assertEqual(r["construction_facet_count"],1)
        self.assertFalse(r["native_admission"])
        self.assertTrue(r["construction_only_facets_remaining"])
        with self.assertRaisesRegex(ValueError,"construction role"):
            self.run_case(construction_faces=self.o[:1])

    def test_missing_exterior_and_internal_faces(self):
        for role in ("outer_faces","interface_faces","deep_faces"):
            with self.subTest(role=role),self.assertRaisesRegex(ValueError,"missing facet"):
                self.run_case(**{role:getattr(self,{"outer_faces":"o","interface_faces":"i","deep_faces":"d"}[role])[:0]})

    def test_wrong_material_role_or_orientation(self):
        with self.assertRaisesRegex(ValueError,"wrong material"):
            self.run_case(outer_faces=self.d)
        with self.assertRaisesRegex(ValueError,"orientation"):
            self.run_case(interface_faces=self.i[:,[0,2,1]])
        with self.assertRaisesRegex(ValueError,"orientation"):
            self.run_case(deep_faces=self.d[:,[0,2,1]])

    def test_duplicate_tables_tets_and_nonmanifold_faces(self):
        with self.assertRaisesRegex(ValueError,"duplicate facet"):
            self.run_case(outer_faces=np.vstack([self.o,self.o[:1]]))
        with self.assertRaisesRegex(ValueError,"duplicate tetrahedron"):
            self.run_case(tetrahedra=np.vstack([self.t,self.t[:1]]),material_ids=np.r_[self.m,1])
        x=np.vstack([self.x,[[0,0,2]]])
        with self.assertRaisesRegex(ValueError,"nonmanifold"):
            self.run_case(points=x,tetrahedra=np.vstack([self.t,[[0,1,2,6]]]),
                          material_ids=np.r_[self.m,1])

    def test_float32_collapse_and_negative_determinant(self):
        with self.assertRaisesRegex(ValueError,"Float32 tetrahedron quality"):
            self.run_case(tetrahedra=self.t[:,[0,2,1,3]])
        x=self.x.copy();x[3]=[0,0,1e-12]
        with self.assertRaisesRegex(ValueError,"Float32 tetrahedron quality"):
            self.run_case(points=x)

    def test_invalid_identity_and_nonfinite_geometry(self):
        with self.assertRaisesRegex(ValueError,"integer"):
            self.run_case(tetrahedra=self.t.astype(float))
        bad=self.o.copy();bad[0,0]=999
        with self.assertRaisesRegex(ValueError,"out-of-range"):
            self.run_case(outer_faces=bad)
        x=self.x.copy();x[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,"finite Float32"):
            self.run_case(points=x)
        with self.assertRaisesRegex(ValueError,"material identity"):
            self.run_case(material_ids=np.array([1,2,3]))

    def test_construction_requires_real_exterior_and_unique_rows(self):
        with self.assertRaisesRegex(ValueError,"not exterior"):
            self.run_case(construction_faces=self.i)
        with self.assertRaisesRegex(ValueError,"duplicate"):
            self.run_case(outer_faces=self.o[1:],construction_faces=np.vstack([self.o[:1],self.o[:1]]))
        with self.assertRaisesRegex(ValueError,"orientation"):
            self.run_case(outer_faces=self.o[1:],construction_faces=self.o[:1,[0,2,1]])


if __name__ == "__main__":
    unittest.main()

