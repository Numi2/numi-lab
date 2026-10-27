from pathlib import Path
import unittest,json
import numpy as np
from merge_saved_field602 import merge
N=Path(__file__).resolve().parent.parent
class ReplaySafety(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  paths=[N/"component-topology-599/run-001/final/state.json",N/"connected-topology-596/run-001/final/state.json",N/"connected-fit-598/run-002/retained-regions/final/state.json"]
  cls.payloads=[]
  for p in paths:
   with np.load(json.loads(p.read_text())["candidate"],allow_pickle=False)as z:cls.payloads.append({k:z[k].copy()for k in z.files})
 def test_baseline_coordinate_mismatch_rejected(self):
  a,b,c=[dict(x)for x in self.payloads];b["points_world_m"]=b["points_world_m"].copy();b["points_world_m"][0,0]+=np.float32(.001)
  with self.assertRaisesRegex(ValueError,"baseline coordinate"):merge(a,b,c)
 def test_saved_topology_mismatch_rejected(self):
  a,b,c=[dict(x)for x in self.payloads];c["source_O_faces"]=c["source_O_faces"].copy();c["source_O_faces"][0,[0,1]]=c["source_O_faces"][0,[1,0]]
  with self.assertRaisesRegex(ValueError,"saved proposal topology"):merge(a,b,c)
 def test_source_mapping_mismatch_rejected(self):
  a,b,c=[dict(x)for x in self.payloads];a["source_prolongation_weights"]=a["source_prolongation_weights"].copy();a["source_prolongation_weights"][0]+=.01
  with self.assertRaisesRegex(ValueError,"point domain/provenance"):merge(a,b,c)
 def test_nan_sentinels_preserved(self):
  a,b,c=self.payloads;d=merge(a,b,c)
  self.assertEqual(d["node_parent_coarse_edge_parameter"].tobytes(),a["node_parent_coarse_edge_parameter"].tobytes())
  self.assertEqual(d["points_world_m"].tobytes(),c["points_world_m"].tobytes())
if __name__=="__main__":unittest.main()
