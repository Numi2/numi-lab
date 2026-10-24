import copy,unittest
import numpy as np
from recover_hand_joint_influences_020 import recover_hand_rows
class InfluenceScopeTests(unittest.TestCase):
 def fixtures(self):
  binds=np.zeros(6,dtype=[('body','u4'),('v','f4',(8,))]);binds['body']=[43,49,54,93,99,104]
  c=dict(nv=4,binds=binds,source_sha256=b'ancestor',fingerprint=1,
         weights=np.array([[0,1,0,0,0,0],[0,0,0,0,1,0],[1,0,0,0,0,0],[0,1,0,0,0,0]],dtype=np.float32))
  h=copy.deepcopy(c);h['weights'][0]=[0,0,1,0,0,0];h['weights'][1]=[0,0,0,0,0,1]
  # A forearm-dominated historical row must not be reclassified as hand.
  h['weights'][3]=[.9,.1,0,0,0,0]
  return c,h
 def test_both_hands_gain_finger_support_without_neighbor_change(self):
  c,h=self.fixtures();w,selected,sides=recover_hand_rows(c,h)
  np.testing.assert_array_equal(selected,[True,True,False,False])
  self.assertEqual(sides,{'right':[0],'left':[1]})
  np.testing.assert_array_equal(w[:2],h['weights'][:2]);np.testing.assert_array_equal(w[2:],c['weights'][2:])
 def test_registered_lineage_mismatch_is_rejected(self):
  c,h=self.fixtures();h['fingerprint']=2
  with self.assertRaisesRegex(ValueError,'lineage'):recover_hand_rows(c,h)
 def test_binding_identity_mismatch_is_rejected(self):
  c,h=self.fixtures();h['binds']['body'][2]=55
  with self.assertRaisesRegex(ValueError,'identities'):recover_hand_rows(c,h)
if __name__=='__main__':unittest.main()
