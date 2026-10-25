import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("depth_adapter_018", HERE / "registered_depth_adapter_018.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class ProlongedDepthTests(unittest.TestCase):
    def setUp(self):
        # Four identity rows, one edge midpoint, one face centroid on 0,1,2.
        ip = np.array([0,1,2,3,4,6,9], dtype=np.int64)
        ix = np.array([0,1,2,3,0,1,0,1,2], dtype=np.int32)
        w = np.array([1,1,1,1,.5,.5,1/3,1/3,1/3], dtype=np.float64)
        self.B = m.validate_prolongation(ip, ix, w, 6, 4)

    def test_constant_field_reproduces_on_every_node(self):
        H = np.tile([0.002, 0.007], (4,1))
        np.testing.assert_allclose(m.prolong_depth_fields(H,self.B),
                                   np.tile([0.002,0.007], (6,1)),rtol=0,atol=2e-18)

    def test_affine_field_reproduces_at_midpoint_and_centroid(self):
        x = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        H = np.column_stack((0.01 + .001*x[:,0] + .002*x[:,1] + .003*x[:,2],
                             0.02 + .004*x[:,0] - .001*x[:,1] + .002*x[:,2]))
        got = m.prolong_depth_fields(H,self.B)
        xo = np.vstack((x, [.5,0.,0.], [1/3,1/3,0.]))
        expected = np.column_stack((0.01 + .001*xo[:,0] + .002*xo[:,1] + .003*xo[:,2],
                                    0.02 + .004*xo[:,0] - .001*xo[:,1] + .002*xo[:,2]))
        np.testing.assert_allclose(got,expected,rtol=0,atol=1e-16)

    def test_prefix_remains_bitwise_and_depths_stay_positive(self):
        rng=np.random.default_rng(18)
        H=0.0005+rng.random((4,2))*0.01
        got=m.prolong_depth_fields(H,self.B)
        self.assertTrue(np.array_equal(got[:4],H))
        self.assertTrue(np.isfinite(got).all())
        self.assertTrue(np.all(got>0))

    def test_invalid_nonconvex_map_rejected(self):
        with self.assertRaises(ValueError):
            m.validate_prolongation([0,2],[0,1],[1.1,-.1],1,2)

    def test_source_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"input.bin";p.write_bytes(b"pinned source")
            with self.assertRaisesRegex(ValueError,"SHA-256 mismatch"):
                m.require_pinned_file(p,"0"*64,"fixture input")

if __name__ == "__main__":
    unittest.main(verbosity=2)

