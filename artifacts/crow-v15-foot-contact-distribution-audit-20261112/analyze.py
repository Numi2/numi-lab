from pathlib import Path
import collections,hashlib,json
A=Path("artifacts/crow-v15-foot-contact-distribution-audit-20261112");P=Path("artifacts/crow-v15-knee-warp-contact-timing-audit-20261112");out={"inputs":{},"results":{},"scope":"One previously used seed and lane, two-second window, positive ground contacts, simulated segment labels. Descriptive and post-hoc. Contact and state timings may differ; speed is not stance-integrated slip."}
for arm in ("baseline","phase"):
 p=P/("analysis-"+arm+".json");raw=p.read_bytes();out["inputs"][arm]={"path":str(p),"sha256":hashlib.sha256(raw).hexdigest()};d=json.loads(raw);g=collections.defaultdict(lambda:{"normal_impulse_ns":0.,"high_speed_impulse_ns":0.,"samples":0})
 for x in d["samples"]:
  row=g[x["body"]];row["normal_impulse_ns"]+=x["normal_impulse_ns"];row["high_speed_impulse_ns"]+=x["normal_impulse_ns"]*(x["tangential_speed_m_s"]>.05);row["samples"]+=1
 for row in g.values():row["high_fraction"]=row["high_speed_impulse_ns"]/row["normal_impulse_ns"]
 out["results"][arm]=dict(sorted(g.items()))
(A/"analysis.json").write_text(json.dumps(out,indent=2)+"\n")
