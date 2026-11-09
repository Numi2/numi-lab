"""Reject damaged, incomplete, and redistributed compressed reference data."""
import copy
import gzip
import json
from pathlib import Path
import tempfile

from large_spatial_reference import read_reference, compare

root = Path(__file__).resolve().parents[2]
path = root / "docs/evidence/midpoint-first-failure-20261109/compact-reference-parity-n2-compact-006.json"
original = json.loads(path.read_text())
with tempfile.TemporaryDirectory() as directory:
    target = Path(directory) / "reference.json.gz"
    def write(value):
        with gzip.open(target, "wt") as out:
            json.dump(value, out)
    def rejected(value):
        write(value)
        try:
            read_reference(target, 2, 2e-6)
        except (ValueError, KeyError):
            return
        raise AssertionError("corrupted reference accepted")
    write(original)
    reference = read_reference(target, 2, 2e-6)
    assert compare(reference, read_reference(path, 2, 2e-6))["sampled_force_trace"] == 0
    wrong = copy.deepcopy(original)
    wrong["contact_area_weights_m2"][0] += 1e-8
    wrong["contact_area_weights_m2"][1] -= 1e-8
    rejected(wrong)
    wrong = copy.deepcopy(original); wrong["trace"].pop(); rejected(wrong)
    wrong = copy.deepcopy(original); wrong["trace"][5][0] += .000001; rejected(wrong)
    wrong = copy.deepcopy(original); wrong["mass_kg"] = float("nan"); rejected(wrong)
    wrong = copy.deepcopy(original); wrong["min_gap_m"] = 1e-7; rejected(wrong)
    wrong = copy.deepcopy(original); wrong["impulse_Ns"] *= 2; rejected(wrong)
print("large spatial reference validation passed")
