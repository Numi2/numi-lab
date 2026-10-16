import hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((r/"manifest.json").read_text())
for p,s in m["files"].items():
 q=r/p
 if not q.is_file() or q.stat().st_size!=s["bytes"] or h(q)!=s["sha256"]:raise SystemExit("sealed mismatch "+p)
for p,d in json.loads((r/"input-bindings.json").read_text()).items():
 q=Path(p)
 if not q.is_file() or h(q)!=d:raise SystemExit("input mismatch "+p)
print("integrity verified")
