import json,math
from pathlib import Path
p=Path(__file__).parent
norm=lambda v:math.sqrt(sum(x*x for x in v))
raw=[json.loads(s) for s in (p/"raw-v3.jsonl").read_text().splitlines()];assert len(raw)==96
prior=[json.loads(s) for s in (p/"prior-results-v3.jsonl").read_text().splitlines() if json.loads(s)["arm"]=="guarded"];assert len(prior)==48
for a,b in zip(raw[::2],raw[1::2]):
 assert a["repeat"]==0 and b["repeat"]==1 and a["exit"]==b["exit"]==0 and not a["stderr"] and not b["stderr"] and a["stdout"]==b["stdout"]
for a,b in zip([x for x in raw if x["arm"]=="guarded"],prior):
 d=json.loads(a["stdout"]);d.pop("localres5");d.pop("localres10");assert d==json.loads(b["stdout"])
out=[]
for a in raw[::2]:
 d=json.loads(a["stdout"]);d["arm"]=a["arm"]
 if d["accepted"]:
  vals=[norm(v) for v,pi in zip(d["localres5"],d["pinned"]) if not pi]
  diffs=[norm([x-y for x,y in zip(u,v)]) for u,v,pi in zip(d["localres5"],d["localres10"],d["pinned"]) if not pi]
  d["max_local_residual"]=max(vals);d["max_gradient_disagreement"]=max(diffs);assert d["max_gradient_disagreement"]<=1e-4
 out.append(d)
g=[d for d in out if d["arm"]=="guarded"];r=[d for d in out if d["arm"]=="residual"]
assert len(g)==len(r)==24 and all(d["accepted"] and d["max_local_residual"]>.001 for d in g) and all(not d["accepted"] for d in r)
summary={"guarded_accepted":24,"guarded_accepted_over_local_gate":24,"residual_rejected":24,"max_local_residual_m_s":max(d["max_local_residual"] for d in g),"max_gradient_disagreement_m_s":max(d["max_gradient_disagreement"] for d in g)}
(p/"analysis-v3.json").write_text(json.dumps({"summary":summary,"cases":out},indent=2)+"\n");print(json.dumps(summary))
