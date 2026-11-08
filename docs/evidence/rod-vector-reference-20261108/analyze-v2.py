import json
from pathlib import Path
p=Path(__file__).parent
items=[json.loads(s) for s in (p/"results-v2.jsonl").read_text().splitlines()];assert len(items)==116
r={}
for a,b in zip(items[::2],items[1::2]):
 assert a["repeat"]==0 and b["repeat"]==1 and a["exit"]==b["exit"]==0 and not a["stderr"] and not b["stderr"] and a["stdout"]==b["stdout"]
 v=[json.loads(s) for s in a["stdout"].splitlines()];assert len(v)==3
 r[(a["arm"],a["axis"],a["offset"],a["kind"])]=v
for kind in ("bent","transverse"):
 assert r[("guarded","x","0",kind)]==r[("vector","x","0",kind)]
summary={}
for arm in ("vector","guarded"):
 for off in ("0","0.5","100"):
  z=[e for (a,axis,o,k),v in r.items() if a==arm and o==off and k=="axial" for e in v]
  summary[f"{arm}@{off}"]={"pass":sum(e["passed"] for e in z),"accepted_failed":sum(e["accepted"] and not e["passed"] for e in z),"rejected":sum(not e["accepted"] for e in z)}
for off in ("0","0.5"):
 assert summary["guarded@"+off]=={"pass":27,"accepted_failed":0,"rejected":0}
assert summary["guarded@100"]=={"pass":0,"accepted_failed":0,"rejected":27}
for axis in ("oblique3","oblique4","oblique5"):
 for off in ("0","0.5"):
  assert all(x["passed"] for x in r[("guarded",axis,off,"axial")])
(p/"analysis-v2.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
