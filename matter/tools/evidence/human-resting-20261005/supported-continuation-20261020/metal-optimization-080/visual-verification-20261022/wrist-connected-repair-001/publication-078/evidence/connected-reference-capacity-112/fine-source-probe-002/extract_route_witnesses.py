from __future__ import annotations
import gzip,json,hashlib
from pathlib import Path
D=Path("/Users/n/Work/numi-human-connected-surface-069/connected-reference-capacity-112")
run=D/"run-002/exact-segment-audits.jsonl.gz";report=D/"fine-source-probe-002/report.json"
cols={204,215,217,218,219,231,233,271,337,1467,1990,2024,2187,2261,2262}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
base=json.loads(report.read_text()); bycol={c:[] for c in cols}
with gzip.open(run,"rt") as f:
 for line in f:
  r=json.loads(line);c=int(r["column_row"])
  if c not in cols:continue
  q=r.get("result",{}); hits=q.get("proper_hits",[]); unresolved=q.get("unresolved_witnesses",[])
  if not hits and not unresolved:continue
  bycol[c].append({"route":r["route"],"body":int(r["body"]),"stable_bone_id":int(r["bone"]),"status":q.get("status"),"reason":q.get("reason"),"proper_hit_count":int(q.get("proper_hit_count",0)),"proper_hits":hits,"unresolved_witnesses":unresolved,"equal_parameter_groups":q.get("equal_parameter_groups",[]),"endpoint_parity_status":q.get("endpoint_parity_status")})
rows=[]
for x in base["results"]:
 c=x["column_row"]
 rows.append({"column_row":c,"nearest_detail_face_row":x["detail_source_face_row"],"nearest_detail_vertex_rows":x["detail_vertex_rows"],"O_to_nearest_source_surface_m":x["distance_O_to_detail_surface_m"],"events":bycol[c]})
out={"schema":"fine-source-probe-route-witnesses-v1","source_probe_report_sha256":sha(report),"source_run112_jsonl_gz_sha256":sha(run),"scope":"Frozen exact OI/ID/OD event witnesses for the same 15 columns, copied from run-002 and not recomputed.","results":rows}
p=D/"fine-source-probe-002/route-witnesses.json";p.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps({"path":str(p),"sha256":sha(p),"columns":len(rows),"events":sum(len(x["events"]) for x in rows)},indent=2))
