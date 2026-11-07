"""CPU-only construction/rejection tests; no physical policy qualification."""
import json
from pathlib import Path
import struct
import subprocess
import tempfile

here = Path(__file__).resolve().parent
prior = here.parent / "g1-episode-local-20261107"
count = 1000*8*29
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    baseline = tmp / "baseline.f32"
    output = tmp / "result"
    def invoke():
        return subprocess.run(["python3", str(here / "make_interventions.py"), str(baseline),
            str(prior / "policy-4081008.trace.tsv"), str(prior / "policy-4081008.stdout"), str(output)],
            capture_output=True, text=True)
    baseline.write_bytes(struct.pack("<"+"f"*count, *([0.0]*count)))
    proc = invoke()
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads(proc.stdout)
    assert (manifest["trigger_step"], manifest["first_action_step"], manifest["last_action_step"]) == (142, 143, 152)
    for name in ("positive", "negative"):
        path = output.with_name(output.name+"-"+name+".f32")
        values = struct.unpack("<"+"f"*count, path.read_bytes())
        indexes = manifest["arms"][name]["indices"]
        assert len(indexes) == 10 and sum(v != 0 for v in values) == 10
        assert all(abs(values[i]-manifest["arms"][name]["delta"]) < 1e-6 for i in indexes)
        path.unlink()
    baseline.write_bytes(struct.pack("<"+"f"*count, *([0.95]*count)))
    proc = invoke()
    assert proc.returncode != 0 and "leave [-1,1]" in proc.stderr
    assert not list(tmp.glob("result-*.f32"))
print("synthetic intervention construction and atomic rejection passed")
