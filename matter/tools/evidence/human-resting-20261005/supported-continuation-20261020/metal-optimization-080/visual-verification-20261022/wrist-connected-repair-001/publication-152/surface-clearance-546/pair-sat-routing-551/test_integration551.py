import importlib.util,unittest
from pathlib import Path
import numpy as np
from fit_pair_sat_planes551 import install_pair_sat551,PLANES,AUDIT
O=Path('/Users/numi/Work/numi-human-connected-surface-069/D/D/fixed-bone-outward-plane-458/fixed_bone_outward_plane_458.py')
def original():
 s=importlib.util.spec_from_file_location('builder458test551',O);o=importlib.util.module_from_spec(s);s.loader.exec_module(o);return o
class TestIntegration(unittest.TestCase):
 def test_signed_rows_match_finite_pair_support(self):
  o=original();install_pair_sat551(o);rng=np.random.default_rng(551)
  negative=0
  for i in range(64):
   skin=rng.normal(size=(3,3))*.5+np.array([260.,-960.,130.]);bone=rng.normal(size=(3,3))*.5+np.array([260.,-960.,130.])
   pair=dict(face_row=0,body=109,bone_id=114,triangle_id=115)
   r=o.build_fixed_bone_outward_plane_rows(skin,np.array([[0,1,2]]),np.arange(3),[],[pair],{(109,114,115):bone})
   n=PLANES['normal_world_unit'][0];lower=PLANES['corner_lower_mm'][0];margin=PLANES['margin_mm'][0]
   np.testing.assert_allclose(lower,float(np.max(bone@n))+margin-skin@n,rtol=0,atol=2e-13)
   np.testing.assert_allclose(r['G'].toarray(),np.kron(np.eye(3),n[None,:]),rtol=0,atol=0)
   negative+=np.count_nonzero(lower<0)
   delta=(lower+1e-7)[:,None]*n
   self.assertGreaterEqual(float(np.min(r['G']@delta.ravel()-r['lower'])),-1e-13)
   self.assertGreater(float(np.min((skin+delta)@n)-np.max(bone@n)),0.)
   self.assertEqual(r['guard_metadata'][0]['plane_selection'],'finite_pair_minimum_deficit_sat_551')
  self.assertGreater(negative,0);self.assertEqual(AUDIT['pair_count'],1)
 def test_anchor_and_identity_contracts_preserved(self):
  o=original();install_pair_sat551(o)
  skin=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
  bone=skin+np.array([.2,.2,0.])
  pair=dict(face_row=0,body=1,bone_id=2,triangle_id=3)
  r=o.build_fixed_bone_outward_plane_rows(skin,np.array([[0,1,2]]),np.arange(3),[0],[pair],{(1,2,3):bone})
  self.assertEqual(r['G'].shape,(3,9));self.assertTrue(np.all(r['G'].toarray()[0]==0))
  self.assertEqual(r['guard_metadata'][0]['body'],1)
  with self.assertRaisesRegex(ValueError,'duplicate'):o.build_fixed_bone_outward_plane_rows(skin,np.array([[0,1,2]]),np.arange(3),[],[pair,pair],{(1,2,3):bone})
if __name__=='__main__':unittest.main()
