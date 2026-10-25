"""Focused regression entrypoint for the frozen 222 sparse clearance builder."""
import importlib.util
import unittest
from pathlib import Path

SOURCE = Path(__file__).with_name("propose_joint_clearance_witnesses_222.py")

spec = importlib.util.spec_from_file_location("proposal222", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {SOURCE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Proposal222Tests(unittest.TestCase):
    def test_lexicographic_sparse_clearance_and_geometry_guards(self):
        module.self_test()


if __name__ == "__main__":
    unittest.main()
