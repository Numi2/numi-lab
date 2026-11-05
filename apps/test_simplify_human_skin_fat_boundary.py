"""Conservation and rejection controls for initialization-only mesh construction."""
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from simplify_human_skin_fat_boundary import compose_local_cell_mass,load_reference_cell_mass

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

if __name__=="__main__":unittest.main()
