import json,math
from pathlib import Path
p=Path(__file__).parent
norm=lambda a:math.sqrt(sum(x*x for x in a))
rows=[json.loads(x) for x in (p/"raw-v2.jsonl").read_text().splitlines()];assert len(rows)==60
results=[]
for a,b in zip(rows[::2],rows[1::2]):
 assert a["repeat"]==0 and b["repeat"]==1 and a["exit"]==b["exit"]==0 and not a["stderr"] and not b["stderr"] and a["stdout"]==b["stdout"]
 d=json.loads(a["stdout"])
 if d["accepted"]:
  e=norm(d["residual5"]);delta=norm([x-y for x,y in zip(d["residual10"],d["residual5"])])
  assert delta<=1e-4 and e<=.001
 results.append(d)
assert sum(x["accepted"] for x in results)==2
for axis in ("x","oblique"):
 assert next(x for x in results if x["axis"]==axis and x["bend"]==0)["accepted"]
s=[json.loads(x) for x in (p/"straight-v2.jsonl").read_text().splitlines()];assert len(s)==48
for a,b in zip(s[::2],s[1::2]):
 assert a["repeat"]==0 and b["repeat"]==1 and a["exit"]==b["exit"]==0 and not a["stderr"] and not b["stderr"] and a["stdout"]==b["stdout"]
 assert not json.loads(a["stdout"])["accepted"]
summary={"microbend_accepted":2,"microbend_rejected":28,"longer_chain_accepted":0,"longer_chain_rejected":24}
(p/"analysis-v2.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary))
