import unittest
import numpy as np
from reference_map_gate import MappingError, validate_reference_parent_map

class ReferenceMapGateTests(unittest.TestCase):
    def setUp(self):
        self.X=np.asarray([[0.,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1]],float)
        self.T=np.asarray([[0,1,2,3],[1,2,3,4]],np.int64)
        self.rows=np.asarray([0,1],np.int64)
        self.B=np.asarray([[.25,.25,.25,.25],[.1,.2,.3,.4]])
        self.domain="original_013_tet_rows:sha256=baseline-pin"
        self.C=np.einsum("ni,nij->nj",self.B,self.X[self.T[self.rows]])

    def test_valid_typed_positive_pullback(self):
        out=validate_reference_parent_map(node_parent_tet_rows=self.rows,node_barycentric=self.B,
            parent_tetrahedra=self.T,deformed_parent_points=self.X,reference_parent_points=self.X,candidate_node_points=self.C,supplied_domain=self.domain,
            expected_domain=self.domain,expected_node_count=2)
        self.assertEqual(out["validated_nodes"],2)

    def test_wrong_integer_row_domain_rejected_even_when_ids_in_range(self):
        with self.assertRaisesRegex(MappingError,"domain mismatch"):
            validate_reference_parent_map(node_parent_tet_rows=self.rows,node_barycentric=self.B,
                parent_tetrahedra=self.T,deformed_parent_points=self.X,reference_parent_points=self.X,candidate_node_points=self.C,
                supplied_domain="018_merged_input_tet_rows",expected_domain=self.domain,
                expected_node_count=2)

    def test_missing_ancestry_rejected(self):
        with self.assertRaisesRegex(MappingError,"explicit per-node"):
            validate_reference_parent_map(node_parent_tet_rows=None,node_barycentric=None,
                parent_tetrahedra=self.T,deformed_parent_points=self.X,reference_parent_points=self.X,candidate_node_points=self.C,
                supplied_domain=self.domain,expected_domain=self.domain,expected_node_count=2)

    def test_nonpositive_deformed_parent_rejected(self):
        folded=self.X.copy();folded[[1,2]]=folded[[2,1]]
        with self.assertRaisesRegex(MappingError,"nonpositive deformed"):
            validate_reference_parent_map(node_parent_tet_rows=np.asarray([0],np.int64),
                node_barycentric=np.asarray([[.25,.25,.25,.25]]),parent_tetrahedra=self.T,
                deformed_parent_points=folded,reference_parent_points=self.X,candidate_node_points=np.asarray([[.25,.25,.25]]),supplied_domain=self.domain,
                expected_domain=self.domain,expected_node_count=1)

    def test_bad_barycentric_or_range_rejected(self):
        with self.assertRaisesRegex(MappingError,"barycentrics"):
            validate_reference_parent_map(node_parent_tet_rows=self.rows,
                node_barycentric=np.asarray([[.25,.25,.25,.25],[.5,.5,.5,-.5]]),
                parent_tetrahedra=self.T,deformed_parent_points=self.X,reference_parent_points=self.X,candidate_node_points=self.C,supplied_domain=self.domain,
                expected_domain=self.domain,expected_node_count=2)
        with self.assertRaisesRegex(MappingError,"out-of-domain"):
            validate_reference_parent_map(node_parent_tet_rows=np.asarray([0,2],np.int64),
                node_barycentric=self.B,parent_tetrahedra=self.T,deformed_parent_points=self.X,reference_parent_points=self.X,candidate_node_points=self.C,
                supplied_domain=self.domain,expected_domain=self.domain,expected_node_count=2)


    def test_in_range_but_wrong_identity_row_rejected_by_reconstruction(self):
        wrong=np.asarray([1],np.int64)
        with self.assertRaisesRegex(MappingError,"does not reconstruct candidate node"):
            validate_reference_parent_map(node_parent_tet_rows=wrong,
                node_barycentric=self.B[:1],parent_tetrahedra=self.T,
                deformed_parent_points=self.X,reference_parent_points=self.X,
                candidate_node_points=self.C[:1],supplied_domain=self.domain,
                expected_domain=self.domain,expected_node_count=1)

    def test_nonfinite_or_nonpositive_point_tolerance_rejected(self):
        for tol in (float("nan"),float("inf"),0.0,-1.0):
            with self.subTest(tol=tol),self.assertRaisesRegex(MappingError,"point tolerance"):
                validate_reference_parent_map(node_parent_tet_rows=self.rows,
                    node_barycentric=self.B,parent_tetrahedra=self.T,
                    deformed_parent_points=self.X,reference_parent_points=self.X,
                    candidate_node_points=self.C,supplied_domain=self.domain,
                    expected_domain=self.domain,expected_node_count=2,point_tolerance_m=tol)

if __name__ == "__main__":
    unittest.main()
