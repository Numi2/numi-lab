"""Fail closed before the registered n16 comparison; does not alter its gates."""
import hashlib
import json
import math
from pathlib import Path

p = Path(__file__).resolve().parent
root = p.parents[2]
paths = {
    "source_sha256": root / "tests/contact/fem_p2_full_restart.py",
    "model_sha256": root / "tests/contact/fem_p2_dynamics.py",
    "reference_sha256": root / "docs/evidence/contact-p2-n8-order18-20261108/fine.json",
    "prerequisite_sha256": root / "docs/evidence/contact-p2-n8-restart-full-20261108/analysis.json",
    "prerequisite_restart_sha256": root / "docs/evidence/contact-p2-n8-restart-full-20261108/fine.json",
}
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def fail(message):
    raise SystemExit("pre-analysis audit failed: " + message)
def finite_tree(value, label):
    if isinstance(value, float) and not math.isfinite(value):
        fail(label + " contains a nonfinite value")
    if isinstance(value, list):
        for index, item in enumerate(value):
            finite_tree(item, f"{label}[{index}]")
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, label + "." + key)

def read(path):
    value = json.loads(path.read_text())
    finite_tree(value, str(path))
    return value

identity = {key: sha(path) for key, path in paths.items()}
prerequisite = read(paths["prerequisite_sha256"])
if prerequisite.get("prerequisite_pass") is not True:
    fail("n8 prerequisite did not pass")
if prerequisite.get("reference_sha256") != identity["reference_sha256"] or prerequisite.get("restart_sha256") != identity["prerequisite_restart_sha256"]:
    fail("n8 prerequisite inputs changed")
reference = read(paths["reference_sha256"])
if len(reference.get("force_bins_N", [])) != 10 or not isinstance(reference.get("impulse_Ns"), (int, float)):
    fail("n8 comparison output has an unexpected shape")
output = {"inputs": identity, "phases": {}}
for variant in ("coarse", "fine"):
    manifest = read(p / f"{variant}-manifest.json")
    artifact = p / f"{variant}.json"
    if manifest.get("variant") != variant or manifest.get("exit") != 0 or manifest.get("limit_reason") is not None:
        fail(variant + " did not complete without a resource stop")
    if any(manifest.get(k) != v for k, v in identity.items()):
        fail(variant + " input identity differs")
    if manifest.get("output_sha256") != sha(artifact):
        fail(variant + " output hash differs")
    data = read(artifact)
    trace = data.get("trace", [])
    bins = data.get("force_bins_N", [])
    if data.get("n") != 16 or data.get("variant") != variant or len(bins) != 10 or len(trace) != 56 or any(not isinstance(row, list) or len(row) < 14 for row in trace):
        fail(variant + " output has an unexpected shape")
    if any(not isinstance(x, (int, float)) for x in bins) or max(bins) <= 0 or not isinstance(data.get("impulse_Ns"), (int, float)) or abs(data["impulse_Ns"]) <= 0:
        fail(variant + " has an invalid normalization")
    times = [row[0] for row in trace]
    if abs(times[0] - .0045) > 1e-15 or abs(times[-1] - .01) > 1e-15 or any(a >= b for a, b in zip(times, times[1:])):
        fail(variant + " trace times are invalid")
    output["phases"][variant] = {"manifest_sha256": sha(p / f"{variant}-manifest.json"), "output_sha256": sha(artifact)}
output["analysis_script_sha256"] = sha(p / "analyze.py")
output["audit_script_sha256"] = sha(Path(__file__))
output["audit_pass"] = True
print(json.dumps(output, indent=2, allow_nan=False))
