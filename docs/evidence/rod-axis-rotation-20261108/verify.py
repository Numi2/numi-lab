import json
from pathlib import Path
r=json.loads(Path("results.json").read_text())
assert set(r)=={"baseline-x","baseline-y","candidate-x","candidate-y"}
for key,exit in (("baseline-x",1),("baseline-y",1),("candidate-x",0),("candidate-y",1)):
 item=r[key];assert item["exit"]==exit and not item["stderr"]
 lines=item["stdout"].splitlines();assert len(lines)==3
 for name,line in zip(("symmetric","asymmetric","moving"),lines):assert "case="+name in line
 if key.startswith("baseline"):
  assert "accepted=1" in lines[0] and "state_gate=0 force_gate=0 passed=0" in lines[0]
  assert all("accepted=0" in line and "gpu=4" in line for line in lines[1:])
 elif key=="candidate-x":assert all("state_gate=1 force_gate=1 passed=1" in line for line in lines)
 else:assert all("accepted=0" in line and "gpu=4" in line for line in lines)
assert r["baseline-x"]["stdout"]==r["baseline-y"]["stdout"]
print("Rotated baseline, direct-solve outcomes and rejection evidence verified.")
