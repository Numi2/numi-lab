from pathlib import Path
import importlib.util,sys,unittest
import numpy as np
p=Path(__file__).with_name('build_partial_depth_targets545_root.py')
s=importlib.util.spec_from_file_location('partial545test',p)
m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)

class TargetSubsetTests(unittest.TestCase):
    def test_real_inside_subset_preserves_full_domain(self):
        full=[2,4,6,9]
        self.assertEqual(m.validate_target_subset(full,np.array([2,6],dtype=np.int64)),[2,6])
        self.assertEqual(full,[2,4,6,9])

    def test_malformed_or_noninside_subsets_fail_closed(self):
        bads=[np.array([],dtype=np.int64),np.array([2,2],dtype=np.int64),
              np.array([6,2],dtype=np.int64),np.array([2,3],dtype=np.int64),
              np.array([2.0]),np.array([[2]],dtype=np.int64),np.array([-1],dtype=np.int64),
              np.array([2**63],dtype=np.uint64)]
        for bad in bads:
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                m.validate_target_subset([2,4,6,9],bad)

if __name__=='__main__':unittest.main()
