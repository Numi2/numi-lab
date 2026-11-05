"""Conservation and rejection controls for initialization-only mesh construction."""
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from simplify_human_skin_fat_boundary import compose_local_cell_mass,load_reference_cell_mass,build_triangle_bins,query_triangle_bins,quotient_closure,compose_face_ancestry,attachment_nodes,volume_boundary_nodes,interior_contraction_directions,unchanged_cavity_boundary

class LocalMassConstruction(unittest.TestCase):
 def test_split_keeps_physical_mass_and_density(self):
  actual=compose_local_cell_mass([.01],[15],[15,15],[3e-6,7e-6],{15:(900,1100)})
  np.testing.assert_allclose(actual,[.003,.007],rtol=2e-15,atol=0)
  self.assertAlmostEqual(float(actual.sum()),.01,places=16)

 def test_tiny_last_cell_does_not_absorb_large_cell_roundoff(self):
  volumes=np.r_[np.linspace(1e-9,3e-9,31),1e-19]
  mass=float(volumes.sum()*950)
  actual=compose_local_cell_mass([mass],[16],np.full(len(volumes),16),volumes,{16:(850,950)})
  np.testing.assert_allclose(actual/volumes,950.,rtol=64*np.finfo(float).eps,atol=0.)
  self.assertLessEqual(abs(float(actual.sum())-mass),2*np.spacing(mass))

 def test_material_mass_cannot_cross_an_interface(self):
  actual=compose_local_cell_mass([.01,.009],[15,16],[16,15,15],
   [1e-5,4e-6,6e-6],{15:(900,1100),16:(850,950)})
  self.assertAlmostEqual(float(actual[0]),.009,places=16)
  self.assertAlmostEqual(float(actual[1:].sum()),.01,places=16)

 def test_volume_change_cannot_be_hidden_by_global_reallocation(self):
  original=np.array([.0095])
  with self.assertRaisesRegex(ValueError,"density bounds"):
   compose_local_cell_mass(original,[16],[16],[9.8e-6],{16:(850,950)})
  np.testing.assert_array_equal(original,[.0095])

 def test_lower_density_limit_is_enforced(self):
  with self.assertRaisesRegex(ValueError,"density bounds"):
   compose_local_cell_mass([.009],[15],[15],[1.01e-5],{15:(900,1100)})

 def test_lost_material_is_not_silently_merged(self):
  with self.assertRaisesRegex(ValueError,"loses a material"):
   compose_local_cell_mass([1.,1.],[15,16],[15],[.002],{15:(900,1100)})

 def test_invalid_domains_are_rejected(self):
  for mass,material,volume in [([0],[15],[1]),([np.nan],[15],[1]),([1],[15],[-1]),([1],[15,15],[1])]:
   with self.subTest(mass=mass,volume=volume),self.assertRaises(ValueError):
    compose_local_cell_mass(mass,material,[15],volume,{15:(900,1100)})

 def test_reference_flow_requires_exact_identity_and_complete_cells(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"flow.csv"
   p.write_text("child_row,source_parent_row,mass_kg\n0,4,.003\n0,5,.007\n1,6,.009\n")
   digest=hashlib.sha256(p.read_bytes()).hexdigest()
   np.testing.assert_allclose(load_reference_cell_mass(p,digest,2),[.01,.009],rtol=2e-15)
   with self.assertRaisesRegex(ValueError,"SHA mismatch"):
    load_reference_cell_mass(p,"0"*64,2)
   with self.assertRaisesRegex(ValueError,"cover every"):
    load_reference_cell_mass(p,digest,3)

 def test_fractional_row_or_negative_source_mass_is_invalid(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"flow.csv"
   for row in ["0.5,4,.01","0,4,-.01"]:
    p.write_text("child_row,source_parent_row,mass_kg\n"+row+"\n")
    with self.subTest(row=row),self.assertRaisesRegex(ValueError,"invalid reference flow"):
     load_reference_cell_mass(p,hashlib.sha256(p.read_bytes()).hexdigest(),1)


class ConnectedPassControls(unittest.TestCase):
 def test_size_bins_conservatively_cover_intersecting_aabbs(self):
  rng=np.random.default_rng(318)
  xyz=rng.normal(size=(200,3,3))*10.**rng.uniform(-8,1,size=(200,1,1))
  points=xyz.reshape(-1,3);faces=np.arange(len(points)).reshape(-1,3)
  bins=build_triangle_bins(points,faces)
  for tri in xyz[::7]:
   expected=np.flatnonzero(np.all(xyz.max(1)>=tri.min(0),axis=1)&np.all(xyz.min(1)<=tri.max(0),axis=1))
   found=query_triangle_bins(bins,tri)
   self.assertTrue(set(expected)<=set(found))

 def test_movement_padding_keeps_candidates_after_an_accepted_neighbor(self):
  points=np.array([[0.,0,0],[0,1,0],[0,0,1]])
  bins=build_triangle_bins(points,np.array([[0,1,2]]))
  moved=points+[.25,0,0]
  self.assertIn(0,query_triangle_bins(bins,moved,.25))
  with self.assertRaises(ValueError):query_triangle_bins(bins,moved,-1)

 def test_deleted_faces_are_not_indexed_and_empty_queries_are_valid(self):
  points=np.array([[0.,0,0],[1,0,0],[0,1,0]])
  bins=build_triangle_bins(points,np.array([[0,0,0],[0,1,2]]))
  np.testing.assert_array_equal(query_triangle_bins(bins,points),[1])
  self.assertEqual(len(query_triangle_bins([],points)),0)

 def test_chained_quotient_resolves_and_cycles_reject(self):
  np.testing.assert_array_equal(quotient_closure([1,2,2,3]),[2,2,2,3])
  for q in [[1,0],[1,2,0],[0,4]]:
   with self.subTest(q=q),self.assertRaises(ValueError):quotient_closure(q)

 def test_repeated_face_repairs_retain_all_source_ancestors(self):
  ops=[{"source_face_rows":{"O":[0,1,2]}},{"source_face_rows":{"O":[2,3]}},
       {"source_face_rows":{"I":[4,5]}}]
  out=compose_face_ancestry(ops,"O")
  self.assertEqual(out[2],[0,1,2,3]);self.assertEqual(out[3],[0,1,2,3])
  self.assertEqual(out[0],[0,1,2]);self.assertEqual(compose_face_ancestry(ops,"D"),{})


class AttachmentConstructionControls(unittest.TestCase):
 def test_exact_source_attachment_rows_are_protected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"attachments.csv"
   p.write_text("stable_attachment_id,donor_body,stable_source_vertex,source_domain\n7,20,1,D37_coarse_parent\n")
   digest=hashlib.sha256(p.read_bytes()).hexdigest()
   self.assertEqual(attachment_nodes(p,digest,np.array([4,9])),{9})
   with self.assertRaisesRegex(ValueError,"SHA mismatch"):attachment_nodes(p,"0"*64,np.array([4,9]))
   with self.assertRaisesRegex(ValueError,"source domain"):attachment_nodes(p,digest,np.array([4,-1]))

 def test_duplicate_or_foreign_attachment_identities_are_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"attachments.csv";header="stable_attachment_id,donor_body,stable_source_vertex,source_domain\n"
   for data in ["7,20,1,wrong\n","7,20,1,D37_coarse_parent\n7,20,0,D37_coarse_parent\n"]:
    p.write_text(header+data)
    with self.subTest(data=data),self.assertRaisesRegex(ValueError,"identity or source domain"):
     attachment_nodes(p,hashlib.sha256(p.read_bytes()).hexdigest(),np.array([4,9]))


class InteriorOnlyControls(unittest.TestCase):
 def setUp(self):
  # A tetrahedron split around one interior node: all four outer facets stay exact.
  self.old=np.array([[4,1,2,3],[0,4,2,3],[0,1,4,3],[0,1,2,4]])
  self.outer=np.array([[0,1,2,3]])
 def test_only_actual_interior_node_can_be_retired(self):
  np.testing.assert_array_equal(volume_boundary_nodes(self.old,np.zeros(4,int)),[0,1,2,3])
  self.assertEqual(interior_contraction_directions(0,4,{0,1,2,3},set()),[(4,0)])
 def test_protected_attachment_is_never_retired(self):
  self.assertEqual(interior_contraction_directions(4,0,{0,1,2,3},{4}),[])
  self.assertEqual(interior_contraction_directions(4,0,{0,1,2,3},{0}),[(4,0)])
 def test_material_interface_is_fixed_even_if_not_rendered(self):
  np.testing.assert_array_equal(volume_boundary_nodes(self.old,[15,15,15,16]),np.arange(5))
 def test_exact_boundary_chain_survives_interior_elimination(self):
  self.assertTrue(unchanged_cavity_boundary(self.old,self.outer))
 def test_open_or_orientation_changed_replacement_is_rejected(self):
  self.assertFalse(unchanged_cavity_boundary(self.old,self.old[:-1]))
  self.assertFalse(unchanged_cavity_boundary(self.old,self.outer[:,[1,0,2,3]]))
 def test_nonmanifold_volume_rejects(self):
  with self.assertRaisesRegex(ValueError,"nonmanifold"):
   volume_boundary_nodes(np.tile(self.outer,(3,1)),np.zeros(3,int))

if __name__=="__main__":unittest.main()
