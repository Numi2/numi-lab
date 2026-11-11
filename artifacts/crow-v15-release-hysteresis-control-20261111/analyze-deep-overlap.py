from pathlib import Path
import gzip,json,collections
A=Path("artifacts/crow-v15-release-hysteresis-control-20261111/autonomous-20261126");geometry={x["step"]:x for x in json.loads((A/"clearance-details.json").read_text())};endpoint={};impulse=collections.defaultdict(float);coverage=collections.Counter()
with gzip.open(A/"run.stderr.gz","rt") as f:
 for line in f:
  if line.startswith("CROW_SUPPORT_SAMPLE "):
   x=json.loads(line.split(" ",1)[1]);t=x["step"]
   if x["lane"]==5 and t in geometry:endpoint[t]=[c for c in x["contacts"] if {c["a_name"],c["b_name"]}=={"crow_left_front_toe_1_distal","crow_course_perch"}]
  elif line.startswith("CROW_TWO_TOE_SUBSTEP "):
   z=line.split();t,k,e=map(int,z[1:4]);
   if e==5 and t in geometry:coverage[t,k]+=1
  elif line.startswith("CROW_SCENE_CONTACT "):
   z=line.split();t,k,e,a,b=map(int,z[1:6]);
   if e==5 and t in geometry and {a,b}=={20,40}:impulse[t]+=float(z[6])
rows=[]
for t,g in sorted(geometry.items()):
 if g["toe1_distal_gap_m"]<-.005:
  assert t in endpoint and all(coverage[t,k]==1 for k in range(16));rows.append({"step":t,"sat_separation_m":g["toe1_distal_gap_m"],"endpoint_pair_count":len(endpoint[t]),"max_endpoint_depth_m":max((-c["separation"] for c in endpoint[t]),default=None),"positive_normal_impulse_all_substeps_Ns":impulse[t]})
def spans(seq):
 out=[];start=prev=None
 for v in seq:
  if start is None:start=prev=v
  elif v==prev+1:prev=v
  else:out.append([start,prev]);start=prev=v
 if start is not None:out.append([start,prev])
 return out
no=[r["step"] for r in rows if not r["endpoint_pair_count"] and r["positive_normal_impulse_all_substeps_Ns"]<=1e-7]
x={"lane":5,"deep_overlap_threshold_m":.005,"deep_endpoint_count":len(rows),"deep_no_endpoint_pair_no_substep_impulse_count":len(no),"deep_no_pair_no_impulse_spans":spans(no),"worst_sat_separation_m":min((r["sat_separation_m"] for r in rows),default=None),"rows":rows,"scope":"Independent oriented-box SAT at accepted endpoint; positive impulse telemetry covers all 16 substeps and endpoint pair logs include zero-impulse contacts. CPU box-box collision is separately checked at selected matching replay states. Endpoint/contact temporal offset and GPU broadphase/solver cause require further isolation; this is not evidence of a corrected collider."};(A/"deep-overlap-analysis.json").write_text(json.dumps(x,indent=2)+"\n");print(json.dumps({k:v for k,v in x.items() if k!="rows"},indent=2))
