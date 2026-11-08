import collections,json
from pathlib import Path
p=Path(__file__).parent
items=[json.loads(x) for x in (p/"results.jsonl").read_text().splitlines()]
assert len(items)==120
for a,b in zip(items[::2],items[1::2]):
 assert a["repeat"]==0 and b["repeat"]==1 and a["exit"]==b["exit"]==0 and not a["stderr"] and not b["stderr"]
 assert a["stdout"]==b["stdout"], (a["arm"],a["axis"],a["offset"],a["kind"])
results={}
for x in items[::2]:
 rows=[json.loads(s) for s in x["stdout"].splitlines()];assert len(rows)==3
 results[(x["arm"],x["axis"],x["offset"],x["kind"])]=rows
for kind in ("bent","transverse"):
 assert results[("prior","x","0",kind)]==results[("vector","x","0",kind)]
for off in ("0","0.5"):
 for axis in ("x","y","z","minusx","oblique1","oblique2"):
  assert all(r["passed"] for r in results[("vector",axis,off,"axial")])
summary={}
for arm in ("production","prior","vector"):
 for off in ("0","0.5","100"):
  rows=[r for (a,axis,o,k),v in results.items() if a==arm and o==off and k=="axial" for r in v]
  summary[arm+"@"+off]={"pass":sum(r["passed"] for r in rows),"accepted_failed":sum(r["accepted"] and not r["passed"] for r in rows),"rejected":sum(not r["accepted"] for r in rows)}
(p/"analysis.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
