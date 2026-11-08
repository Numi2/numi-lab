import hashlib
import json
import pathlib
import subprocess
root = pathlib.Path(__file__).resolve().parent
checkout = root.parents[2]
link = json.loads((root.parent / "mixed-failing-row-witness-20261108/build.json").read_text())[2]["command"]
records = []
def run(cmd):
    proc = subprocess.run([str(x) for x in cmd], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    records.append({"command": list(map(str, cmd)), "returncode": proc.returncode, "output": proc.stdout})
    (root / "build.json").write_text(json.dumps(records, indent=2) + "\n")
    if proc.returncode:
        raise SystemExit(proc.returncode)
run(["python3", root / "prepare.py"])
run(["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-O3", "-I", checkout / "include", "-c", root / "candidate.metal", "-o", root / "candidate.air"])
cmd = [str(root / "candidate.air") if x.endswith("/baseline.air") else str(root / "candidate.metallib") if x.endswith("/baseline.metallib") else x for x in link]
run(cmd)
(root / "inputs.sha256").write_text("".join(hashlib.sha256((root / x).read_bytes()).hexdigest() + "  " + x + "\n" for x in ("candidate.metal", "candidate.metallib")))
