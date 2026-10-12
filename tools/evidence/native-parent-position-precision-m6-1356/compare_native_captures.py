import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser()
ap.add_argument("--parent",required=True,type=Path)
ap.add_argument("--candidate",required=True,type=Path)
ap.add_argument("--output",required=True,type=Path)
ap.add_argument("--allow-target-change",action="store_true")
a=ap.parse_args()
assert not a.output.exists()
sys.path.insert(0,"/Users/n/numi-human-left-pq-publication-1344/src")
from numilab_human import common_atlas_skin_clearance as ca
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inventory=Path("/Users/n/numi-human-resting-evidence-20261005/native-complete-skin-containment-audit-890/pair-summary-v3.csv")
keys,*_=ca._load_target_inventory(inventory)
keys=set(keys)|{(51005,s) for s in (151,152,159,160,161,162,163,164)}
expected=keys|{(51007,1)}
targets={(51005,132),(51005,134)}
r={"parent":str(a.parent.resolve()),"candidate":str(a.candidate.resolve()),"scope":"Exact native positions/topology/accepted-pose and physiology comparison. Intersection qualification is separate.","captures":{},"input_sha256":{str(Path(__file__).resolve()):sha(__file__),str(inventory):sha(inventory),str(Path(ca.__file__).resolve()):sha(ca.__file__)},"complete":False}
def save():a.output.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
def pin(p):r["input_sha256"][str(p.resolve())]=sha(p)
for scene in (a.parent,a.candidate):
 ep=scene/"baseline/execution.json";e=json.loads(ep.read_text())
 assert e["returncode"]==0 and not e["changed_inputs"] and e["native_argv_matches_prepared_cli_preview"] and e["native_environment_matches_prepared_cli_preview"]
 pin(ep);pin(scene/"baseline/run-declaration.json")
for step in (0,3807,4767,5023,5599,6207,7423,8000):
 pp=a.parent/f"baseline/native-run/accepted-geometry/step-{step}.mrvpack"
 qp=a.candidate/f"baseline/native-run/accepted-geometry/step-{step}.mrvpack"
 pr=pp.with_suffix(".receipt.json");qr=qp.with_suffix(".receipt.json")
 for p in (pp,qp,pr,qr):pin(p)
 pc=json.loads(pr.read_text());qc=json.loads(qr.read_text())
 assert pc["pack_file_sha256"]==sha(pp) and qc["pack_file_sha256"]==sha(qp)
 assert pc["physical_endpoint"]==qc["physical_endpoint"]=="accepted"
 p,ps,_=ca._pack_surfaces(pp,keys);q,qs,_=ca._pack_surfaces(qp,keys)
 assert set(ps)==set(qs)==expected
 changes=[];topology=[]
 for k in sorted(ps):
  f=np.asarray(ps[k]["faces"]);g=np.asarray(qs[k]["faces"])
  if not np.array_equal(f,g):topology.append(list(k));continue
  u=np.unique(f)
  if not np.array_equal(p[u],q[u]):
   changes.append({"key":list(k),"changed_vertices":int(np.count_nonzero(np.any(p[u]!=q[u],axis=1))),"max_absolute_coordinate_difference_m":float(np.max(np.abs(p[u].astype(float)-q[u].astype(float))))})
 poses=pc.get("accepted_body_poses")==qc.get("accepted_body_poses") and len(pc.get("accepted_body_poses",[]))==157
 unexpected=[x for x in changes if not a.allow_target_change or tuple(x["key"]) not in targets]
 r["captures"][str(step)]={"surface_count":len(ps),"changed_surfaces":changes,"topology_mismatches":topology,"accepted_157_poses_exact":poses,"unexpected_geometry_changes":unexpected,"passed":poses and not topology and not unexpected}
 save()
pp=a.parent/"baseline/native-run/resting-coupled.csv";qp=a.candidate/"baseline/native-run/resting-coupled.csv"
pin(pp);pin(qp)
r["physiology_trace_bytes_exact"]=pp.read_bytes()==qp.read_bytes()
r["physiology_rows"]=max(0,len(qp.read_text().splitlines())-1)
r["complete"]=True
r["passed"]=r["physiology_trace_bytes_exact"] and all(v["passed"] for v in r["captures"].values())
save()
print(json.dumps({"passed":r["passed"],"physiology_exact":r["physiology_trace_bytes_exact"],"captures":len(r["captures"]),"changed_surface_counts":{k:len(v["changed_surfaces"]) for k,v in r["captures"].items()}}))
