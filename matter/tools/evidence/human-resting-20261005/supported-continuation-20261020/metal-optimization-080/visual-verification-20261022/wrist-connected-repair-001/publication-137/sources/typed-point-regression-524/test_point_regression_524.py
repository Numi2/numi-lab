import unittest
import numpy as np
from point_regression_witnesses_524 import regressed_point_faces

def report(items):
    return dict(inside_node_bank_pairs=len(items), inside_witnesses=[
        dict(output_node=i, query_row=str(i), bank_body=str(b), bank_stable_id=str(k),
             nearest_distance_m=str(d), classification="inside") for i,b,k,d in items])

class RegressionTests(unittest.TestCase):
    F = np.array([[0,1,2], [1,3,2]],dtype=np.int64)
    def test_existing_improvement_is_not_regression(self):
        rows,faces=regressed_point_faces(report([(1,2,3,.002)]),report([(1,2,3,.001)]),self.F,4)
        self.assertEqual(rows.tolist(),[]); self.assertEqual(faces.tolist(),[])
    def test_existing_deepening_selected(self):
        rows,faces=regressed_point_faces(report([(1,2,3,.001)]),report([(1,2,3,.002)]),self.F,4)
        self.assertEqual(rows.tolist(),[1]); self.assertEqual(faces.tolist(),[0,1])
    def test_new_bank_identity_selected(self):
        rows,_=regressed_point_faces(report([(1,2,3,.002)]),report([(1,2,4,.001)]),self.F,4)
        self.assertEqual(rows.tolist(),[1])
    def test_zero_baseline_retains_old_behavior(self):
        rows,_=regressed_point_faces(report([]),report([(0,2,3,.001)]),self.F,4)
        self.assertEqual(rows.tolist(),[0])
    def test_duplicate_rejected(self):
        bad=report([(1,2,3,.001),(1,2,3,.002)])
        with self.assertRaises(ValueError):regressed_point_faces(report([]),bad,self.F,4)
    def test_incomplete_rejected(self):
        bad=report([(1,2,3,.001)]); bad["inside_node_bank_pairs"]=2
        with self.assertRaises(ValueError):regressed_point_faces(report([]),bad,self.F,4)
if __name__=="__main__":unittest.main()
