import tempfile
import unittest
from pathlib import Path
import numpy as np

from compare_capacity_prefix_009_010 import compare_runs

def make_run(root, part_name, interval, faces, bary, low, zero):
    root.mkdir(parents=True)
    partdir=root/'constraint-parts';partdir.mkdir()
    np.savez(partdir/part_name,tet_ids=np.asarray([4],np.int64),face_ids=np.asarray(faces,np.int64),
             bank_ids=np.asarray([2],np.uint16),poly_offsets=np.asarray([0,len(bary)],np.int64),
             polygon_barycentric=np.asarray(bary,np.float64),
             insufficient_capacity_polygon_vertex_rows=np.asarray(low,np.int64),
             zero_capacity_polygon_vertex_rows=np.asarray(zero,np.int64))
    meta={'file':part_name,'bank_id':2,'flat_face_range':list(interval)}
    cp={'faces_processed':interval[1],'parts':[meta]}
    (root/'checkpoint.json').write_text(__import__('json').dumps(cp))

class CapacityPrefixCompareTests(unittest.TestCase):
    def test_exact_canonical_rows_match_independent_of_checkpoint_files(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);b=np.asarray([[.2,.3,.1,.4],[.5,.1,.2,.2]],np.float64)
            make_run(p/'a','part-00000.npz',(0,2),[1],b,[0,1],[0])
            make_run(p/'b','part-00400.npz',(0,2),[1],b,[0,1],[0])
            report=compare_runs(p/'a',p/'b',2)
            self.assertEqual(report['status'],'PASS_EXACT_COMMON_PREFIX_ROWS')

    def test_row_data_difference_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);a=np.asarray([[.2,.3,.1,.4],[.5,.1,.2,.2]],np.float64)
            b=a.copy();b[0,0]+=1e-10;b[0,1]-=1e-10
            make_run(p/'a','part-00000.npz',(0,2),[1],a,[0,1],[0])
            make_run(p/'b','part-00000.npz',(0,2),[1],b,[0,1],[0])
            report=compare_runs(p/'a',p/'b',2)
            self.assertEqual(report['status'],'FAIL_COMMON_PREFIX_ROW_DIFFERENCE')

    def test_zero_hit_chunk_ranges_must_match(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);b=np.asarray([[.2,.3,.1,.4],[.5,.1,.2,.2]],np.float64)
            make_run(p/'a','part-00000.npz',(0,2),[1],b,[0,1],[0])
            make_run(p/'b','part-00000.npz',(0,1),[0],b,[0,1],[0])
            with self.assertRaisesRegex(ValueError,'not cover|part ranges differ'):
                compare_runs(p/'a',p/'b',2)

if __name__=='__main__':unittest.main()
