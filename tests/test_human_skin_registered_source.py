import unittest
import numpy as np
from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location('human_skin_registered_source',Path(__file__).resolve().parents[1]/'tools/human_skin_registered_source.py')
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
audit_registered_outer_source=owner.audit_registered_outer_source

class OuterSourceGuardTests(unittest.TestCase):
    def fixture(self):
        source=np.array([[0.,0.,0.],[0.,1.,0.],[0.,1.,0.],[2.,0.,0.]],float)
        source_faces=np.array([[0,1,2],[1,3,2]],int)
        # Two outer faces: detailed replacement face 0 and retained coarse face 1.
        outer=source.copy()
        faces=np.array([[0,1,2],[1,3,2]],int)
        parent=np.array([-1,77],int)
        detailed=np.array([0,-1],int)
        direct=np.array([0,1,2,3],int)
        return dict(outer_points=outer,surface_faces=faces,source_points=source,
            detailed_source_faces=source_faces,parent_surface_face_ids=parent,
            detailed_source_face_ids=detailed,
            required_replacement_face_rows=[0],
            required_boundary_nodes=[0,1,2,3],node_source_vertex_ids=direct)
    def run_audit(self, **patch):
        a=self.fixture();a.update(patch);return audit_registered_outer_source(**a)
    def test_mixed_retained_parent_and_detailed_replacement_pass(self):
        r=self.run_audit()
        self.assertTrue(r["passed"],r)
        self.assertEqual(r["counts"]["face_domain_parent_only"],1)
    def test_replacement_cannot_use_parent_id_as_detailed_identity(self):
        r=self.run_audit(parent_surface_face_ids=np.array([77,88]),detailed_source_face_ids=np.array([-1,-1]))
        self.assertFalse(r["passed"])
        self.assertIn("replacement_face_lacks_typed_detailed_source_face_identity",r["issues"])
    def test_wrong_detailed_face_id_is_caught_by_corner_ancestry(self):
        r=self.run_audit(detailed_source_face_ids=np.array([1,-1]))
        self.assertFalse(r["passed"])
        self.assertIn("replacement_face_corner_source_identity_mismatch",r["issues"])
    def test_projected_row_requires_explicit_face_bary_residual(self):
        a=self.fixture();a["node_source_vertex_ids"]=np.array([0,-1,2,3])
        a["node_source_face_rows"]=np.array([-1,0,-1,-1])
        a["node_source_barycentric"]=np.array([[np.nan]*3,[0.,1.,0.],[np.nan]*3,[np.nan]*3])
        a["node_source_residual_world_m"]=np.zeros((4,3))
        self.assertTrue(audit_registered_outer_source(**a)["passed"])
    def test_direct_id_with_projection_residual_is_not_silently_accepted(self):
        a=self.fixture();res=np.zeros((4,3));res[0,0]=1e-3;a["node_source_residual_world_m"]=res
        r=audit_registered_outer_source(**a)
        self.assertFalse(r["passed"])
        self.assertIn("direct_source_identity_has_projection_residual_without_projected_embedding",r["issues"])
    def test_missing_replacement_boundary_mapping_fails(self):
        a=self.fixture();a["node_source_vertex_ids"]=np.array([0,1,-1,3])
        a["required_boundary_nodes"]=[0,1,2]
        r=audit_registered_outer_source(**a)
        self.assertFalse(r["passed"])
        self.assertIn("required_physical_boundary_node_unmapped_or_not_reconstructed",r["issues"])
    def test_parent_faces_outside_replacement_region_remain_allowed(self):
        a=self.fixture();a["required_replacement_face_rows"]=[0]
        self.assertTrue(audit_registered_outer_source(**a)["passed"])
    def test_repeated_parent_source_identity_is_allowed_for_subdivided_faces(self):
        r=self.run_audit(parent_surface_face_ids=np.array([77,77]),
            detailed_source_face_ids=np.array([0,1]),
            required_replacement_face_rows=[0,1])
        self.assertTrue(r["passed"],r)
    def test_detailed_face_identity_out_of_domain_fails(self):
        r=self.run_audit(detailed_source_face_ids=np.array([99,-1]))
        self.assertFalse(r["passed"])
        self.assertIn("detailed_source_face_identity_out_of_range",r["issues"])
    def test_float_or_nan_identity_arrays_are_rejected_before_cast(self):
        cases = [
            ("surface_faces", np.array([[0.,1.,2.],[1.,3.,2.]]), "surface_faces_dtype_not_integer"),
            ("detailed_source_faces", np.array([[0.,1.,2.],[1.,3.,2.]]), "detailed_source_faces_dtype_not_integer"),
            ("parent_surface_face_ids", np.array([-1.,77.]), "parent_surface_face_ids_dtype_not_integer"),
            ("detailed_source_face_ids", np.array([np.nan,-1.]), "detailed_source_face_ids_dtype_not_integer"),
            ("node_source_vertex_ids", np.array([0.,1.,2.,3.]), "node_source_vertex_ids_dtype_not_integer"),
            ("node_source_face_rows", np.array([-1.,0.,-1.,-1.]), "node_source_face_rows_dtype_not_integer"),
            ("node_source_edge_vertices", np.array([[-1.,-1.]]*4), "node_source_edge_vertices_dtype_not_integer"),
            ("required_replacement_face_rows", np.array([0.]), "required_replacement_face_rows_dtype_not_integer"),
            ("required_boundary_nodes", np.array([0.,1.]), "required_boundary_nodes_dtype_not_integer"),
        ]
        for name, value, issue in cases:
            with self.subTest(name=name):
                r=self.run_audit(**{name:value})
                self.assertFalse(r["passed"])
                self.assertIn(issue,r["issues"])
    def test_identity_array_shapes_are_checked_without_reshaping(self):
        cases = [
            ("surface_faces", np.array([[0,1],[1,3]],dtype=np.int64), "surface_faces_shape_invalid"),
            ("detailed_source_faces", np.array([[0,1],[1,3]],dtype=np.int64), "detailed_source_faces_shape_invalid"),
            ("parent_surface_face_ids", np.array([[-1,77]],dtype=np.int64), "parent_surface_face_ids_shape_invalid"),
            ("detailed_source_face_ids", np.array([[0,-1]],dtype=np.int64), "detailed_source_face_ids_shape_invalid"),
            ("node_source_vertex_ids", np.array([[0,1,2,3]],dtype=np.int64), "node_source_vertex_ids_shape_invalid"),
            ("node_source_face_rows", np.array([[-1,0,-1,-1]],dtype=np.int64), "node_source_face_rows_shape_invalid"),
            ("node_source_edge_vertices", np.array([-1,-1,-1,-1],dtype=np.int64), "node_source_edge_vertices_shape_invalid"),
            ("required_replacement_face_rows", np.array([[0]],dtype=np.int64), "required_replacement_face_rows_shape_invalid"),
            ("required_boundary_nodes", np.array([[0,1]],dtype=np.int64), "required_boundary_nodes_shape_invalid"),
        ]
        for name, value, issue in cases:
            with self.subTest(name=name):
                r=self.run_audit(**{name:value})
                self.assertFalse(r["passed"])
                self.assertIn(issue,r["issues"])
    def test_nonfinite_tolerance_is_rejected(self):
        for value in (np.nan, np.inf, -np.inf, 0.0, -1.0, np.array([1e-7])):
            with self.subTest(value=value):
                r=self.run_audit(point_tolerance_m=value)
                self.assertFalse(r["passed"])
                self.assertIn("point_tolerance_not_finite_positive_scalar",r["issues"])
    def test_nonfinite_residual_on_required_mapped_node_fails(self):
        residual=np.zeros((4,3));residual[2,1]=np.nan
        r=self.run_audit(node_source_residual_world_m=residual)
        self.assertFalse(r["passed"])
        self.assertIn("mapped_node_has_nonfinite_source_residual",r["issues"])
        self.assertIn("source_point_plus_residual_does_not_reconstruct_physical_outer_node",r["issues"])
        self.assertIn("required_physical_boundary_node_unmapped_or_not_reconstructed",r["issues"])

if __name__=="__main__":unittest.main()
