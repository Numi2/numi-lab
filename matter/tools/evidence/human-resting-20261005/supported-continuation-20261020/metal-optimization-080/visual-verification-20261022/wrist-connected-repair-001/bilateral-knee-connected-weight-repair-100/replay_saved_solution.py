#!/usr/bin/env python3
"""Exact source-asset replay; no native simulation is performed here."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "tools/rebind_human_skin_reference.py").is_file())
SPEC = importlib.util.spec_from_file_location("skin_rebind", ROOT / "tools/rebind_human_skin_reference.py")
rb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rb)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional new NHSKIN file; existing files are never overwritten.")
    args = parser.parse_args()
    baseline = HERE.parent / "forefoot-connected-weight-repair-091/candidate.nhskin"
    raw = baseline.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "a50e68939206d96a6c5e5e05d8f3f4d8d76b4bd46085e3dc6430f01dddfbc192"
    saved = HERE / "sparse-solution.npz"
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == "d18f7cb5963c081e1f9bab04825d5161eed6b77d6281328cc84dacf1439214b0"
    payload = rb.decode(raw)
    w = payload["weights"].astype(float)
    z = np.load(saved, allow_pickle=False)
    sel, pairs, delta = z["selected"], z["pairs"], z["delta"]
    w[sel, pairs[:,0]] += delta
    w[sel, pairs[:,1]] -= delta
    a, b = rb.reference_transforms(payload, json.loads((HERE / "neutral.json").read_text()), [0,0,.35])
    blob, report = rb.rebind(payload, w, a, b)
    digest = hashlib.sha256(blob).hexdigest()
    assert digest == "27368e4bb9a0f59f41e67241a61b17de515a6ffb651b36dce6d47a81902869f3"
    assert blob == (HERE / "candidate.nhskin").read_bytes()
    if args.output:
        with args.output.open("xb") as output:
            output.write(blob)
    print(json.dumps({"exact_replay": True, "output_sha256": digest,
                      "changed_rows": len(report["changed_rows"]),
                      "reference_error_max_m": report["reference_error_max_m"],
                      "physical_admission": False}))

if __name__ == "__main__":
    main()
