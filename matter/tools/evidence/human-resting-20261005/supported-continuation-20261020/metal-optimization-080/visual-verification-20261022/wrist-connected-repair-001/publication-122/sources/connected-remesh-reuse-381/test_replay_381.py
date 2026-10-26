from __future__ import annotations
import importlib.util,sys,unittest
from pathlib import Path
import numpy as np

p=Path(__file__).with_name("replay_381.py")
s=importlib.util.spec_from_file_location("replay_381",p)
m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)

class Replay381Tests(unittest.TestCase):
 def test_only_expected_edge_with_exact_face_pair_is_eligible(self):
  data={"source_O_faces":np.asarray([[0,1,2],[1,0,3],[4,5,6],[5,4,7]],dtype=np.int64),
        "refinement_face_parent_face_row":np.asarray([11,11,12,12])}
  ops=[{"edge":[0,1],"face_rows":[0,1],"parent_row":11},
       {"edge":[4,5],"face_rows":[0,1],"parent_row":11},
       {"edge":[0,1],"face_rows":[0,1],"parent_row":12}]
  good,bad=m.eligible_on_current_topology(data,ops)
  self.assertEqual(good,[ops[0]])
  self.assertEqual([x["reason"] for x in bad],["edge_adjacency_or_pair_rows_changed","immediate_parent_mismatch"])

 def test_flips_must_not_touch_changed_or_missing_expected_parent(self):
  data={"source_O_faces":np.asarray([[0,1,2],[1,0,3]],dtype=np.int64),
        "refinement_face_parent_face_row":np.asarray([8,8])}
  op={"edge":[0,1],"face_rows":[0,1],"parent_row":9}
  good,bad=m.eligible_on_current_topology(data,[op])
  self.assertEqual(good,[])
  self.assertEqual(bad[0]["reason"],"immediate_parent_mismatch")

 def test_signature_normalizes_edge_order_but_retains_pair_rows(self):
  self.assertEqual(m.sig({"edge":[9,2],"face_rows":[5,6],"parent_row":1}),
                   m.sig({"edge":[2,9],"face_rows":[5,6],"parent_row":1}))
  self.assertNotEqual(m.sig({"edge":[2,9],"face_rows":[6,5],"parent_row":1}),
                      m.sig({"edge":[2,9],"face_rows":[5,6],"parent_row":1}))

if __name__=="__main__":unittest.main(verbosity=2)
