import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "pointworld_cuda_oracle.py"
SPEC = importlib.util.spec_from_file_location("pointworld_cuda_oracle", SCRIPT)
ORACLE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ORACLE)


class PointWorldOracleTest(unittest.TestCase):
    def test_array_fingerprint_binds_shape_dtype_and_bytes(self):
        original = np.arange(12, dtype=np.float32).reshape(3, 4)
        self.assertEqual(
            ORACLE._array_fingerprint(original),
            ORACLE._array_fingerprint(original.copy()),
        )
        self.assertNotEqual(
            ORACLE._array_fingerprint(original),
            ORACLE._array_fingerprint(original.reshape(2, 6)),
        )
        self.assertNotEqual(
            ORACLE._array_fingerprint(original),
            ORACLE._array_fingerprint(original.astype(np.float64)),
        )

    def test_atomic_outputs_are_complete_and_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = {"stage.encoder": np.eye(3, dtype=np.float32)}
            npz = root / "oracle.npz"
            receipt = root / "oracle.json"
            ORACLE._atomic_npz(npz, arrays)
            ORACLE._atomic_json(receipt, {"contract": "PointWorldCUDAOracleV1"})
            with np.load(npz, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["stage.encoder"], arrays["stage.encoder"])
            self.assertEqual(
                json.loads(receipt.read_text(encoding="utf-8"))["contract"],
                "PointWorldCUDAOracleV1",
            )
            self.assertEqual(set(root.glob("oracle.*")), {receipt, npz})


if __name__ == "__main__":
    unittest.main()
