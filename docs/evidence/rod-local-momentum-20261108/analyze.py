import json,math
from pathlib import Path
p=Path(__file__).parent
raw=[json.loads(x) for x in (p/"raw.jsonl").read_text().splitlines()];assert len(raw)==120
norm=lambda a:math.sqrt(sum(v*v for v in a))
sub=lambda a,b:[x-y for x,y in zip(a,b)]
rows=[]
for a,b in zip(raw[::2],raw[1::2]):
 assert a["repeat"]==0 and b["repeat"]==1 and a["exit"]==b["exit"]==0 and not a["stderr"] and not b["stderr"] and a["stdout"]==b["stdout"]
 d=json.loads(a["stdout"]);d["arm"]=a["arm"]
 if d["accepted"]:
  d["residual"]=norm(d["residual5"]);d["gradient_agreement"]=norm(sub(d["residual10"],d["residual5"]));d["material_minus_stretch_force"]=norm(sub(d["gradient5"],d["stretch_gradient"]))
  assert d["gradient_agreement"]<=1e-4
  d["pass"]=d["residual"]<=.001
 rows.append(d)
for axis in ("x","oblique"):
 z=next(x for x in rows if x["arm"]=="guarded" and x["axis"]==axis and x["bend"]==0)
 assert z["accepted"] and z["pass"]
 # Deliberate velocity corruption must be detected by the same residual gate.
 assert norm([z["residual5"][0]+.002,z["residual5"][1],z["residual5"][2]])>.001
for axis in ("x","oblique"):
 z=next(x for x in rows if x["arm"]=="guarded" and x["axis"]==axis and abs(x["bend"]-1e-7)<1e-18)
 assert z["accepted"] and not z["pass"]
summary={}
for arm in ("production","guarded"):
 z=[x for x in rows if x["arm"]==arm];summary[arm]={"accepted":sum(x["accepted"] for x in z),"accepted_failed":sum(x["accepted"] and not x["pass"] for x in z),"rejected":sum(not x["accepted"] for x in z),"maximum_gradient_disagreement":max(x.get("gradient_agreement",0) for x in z)}
(p/"analysis.json").write_text(json.dumps({"summary":summary,"cases":rows},indent=2)+"\n");print(json.dumps(summary,indent=2))
