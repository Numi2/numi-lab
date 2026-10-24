import sys,unittest
from pathlib import Path
D=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/global-skin-crossing-patches-012")
sys.path.insert(0,str(D))
import verify_patches_012 as v
class OffsetValidationTests(unittest.TestCase):
    def test_valid_offsets_partition_rows(self):
        v.check_offsets("x",[0,2,2,5],5,3)
    def test_reject_nonzero_start(self):
        with self.assertRaises(ValueError):v.check_offsets("x",[1,2],2)
    def test_reject_decreasing_offsets(self):
        with self.assertRaises(ValueError):v.check_offsets("x",[0,2,1],1)
    def test_reject_wrong_terminal(self):
        with self.assertRaises(ValueError):v.check_offsets("x",[0,1],2)
    def test_reject_wrong_component_count(self):
        with self.assertRaises(ValueError):v.check_offsets("x",[0,1,2],2,1)
    def test_frozen_full_artifact(self):
        result=v.run()
        self.assertEqual(result["metrics"]["patches"],9)
        self.assertEqual(result["metrics"]["selected_pairs"],11251) if "selected_pairs" in result["metrics"] else self.assertEqual(result["metrics"]["canonical_pairs"],11251)
        self.assertEqual(result["metrics"]["exact_hit_points"],22502)
if __name__=="__main__":unittest.main(verbosity=2)
