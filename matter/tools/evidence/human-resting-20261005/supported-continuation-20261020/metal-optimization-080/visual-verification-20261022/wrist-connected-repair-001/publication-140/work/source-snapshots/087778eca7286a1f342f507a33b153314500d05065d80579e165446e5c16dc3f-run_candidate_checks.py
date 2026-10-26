#!/usr/bin/env python3
"""Run existing construction arithmetic and exact owners from one pinned baseline."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, subprocess, sys, time
import numpy as np

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
BASE=ROOT/"D/connected-quality-patch-538/run-001/candidate.npz"
BASE_SHA="2f52328e1105dad576717eae872391324a1172ceec3f74f23fae591891078c6e"
BASE130=ROOT/"D/connected-quality-patch-538/root-exact-run001/audit-130/summary.json"
BASE159=ROOT/"D/connected-quality-patch-538/root-exact-run001/audit-159/report.json"
OWNER=ROOT/"D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
OWNER_SHA="5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2"
R0=ROOT/"full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz"
R0_SHA="70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31"
AUDITS=[("audit-130",ROOT/"audit_connected_outer_geometry_130.py","52d45f287162d22dca417793c1393bcfbbe506c5ce85b828f009b4fd7ccd94b9"),("audit-159",ROOT/"audit_skin_bone_surfaces_159.py","de5f61b69d4b5621749df181f167b046a9d088daad081bf3dc453cef9f824ad1")]
SLOT=Path("/tmp/numi-human-geometry-slot.py")

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b):
 a=np.ascontiguousarray(a); b=np.ascontiguousarray(b)
 return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def mod(p):
 s=importlib.util.spec_from_file_location("owner408_focused",p)
 m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
def boundary(f):
 counts={}
 for t in np.asarray(f):
  if len(set(map(int,t)))!=3:raise ValueError("repeated face vertex")
  for i,j in zip(t,np.roll(t,-1)):
   e=tuple(sorted((int(i),int(j))));counts[e]=counts.get(e,0)+1
 if any(v>2 for v in counts.values()):raise ValueError("nonmanifold edge")
 return {e for e,v in counts.items() if v==1}
def validate_banks(j,expected_sha):
 if j.get("candidate_sha256")!=expected_sha or j.get("complete") is not True or j.get("expected_bank_count")!=185 or j.get("bank_count")!=185:
  raise ValueError("incomplete or stale bone audit")
 rows=j.get("banks",[])
 if len(rows)!=185 or len({(int(b["body"]),int(b["stable_bone_id"])) for b in rows})!=185:raise ValueError("duplicate or missing typed banks")
 total=0;counts={}
 for b in rows:
  r=b.get("result")
  if r is None:
   if int(b.get("skin_aabb_active",0)) and int(b.get("bank_aabb_active",0)):raise ValueError("active bank has no result")
   n=0
  else:
   pairs=r.get("triangle_pairs")
   if not isinstance(pairs,list) or int(r.get("count",-1))!=len(pairs):raise ValueError("bone pair count mismatch")
   n=len(pairs)
  counts[str(b["body"])+"/"+str(b["stable_bone_id"])]=n;total+=n
 return total,counts

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--candidate",required=True);ap.add_argument("--output-dir",required=True);ap.add_argument("--arithmetic-only",action="store_true");a=ap.parse_args()
 cp=Path(a.candidate);out=Path(a.output_dir)
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);start=time.monotonic()
 result={"candidate_path":str(cp),"candidate_sha256":sha(cp),"baseline_sha256":BASE_SHA,"native_admission":False,"accepted_physical_steps":0,"commands":[]}
 pins={BASE:BASE_SHA,OWNER:OWNER_SHA,R0:R0_SHA,**{p:h for _,p,h in AUDITS},cp:result["candidate_sha256"],Path(__file__):sha(__file__)}
 try:
  for p,h in pins.items():
   if sha(p)!=h:raise ValueError("source/input hash mismatch: "+str(p))
  b=load(BASE);c=load(cp);x=c["points_world_m"];xb=b["points_world_m"];f=c["source_O_faces"]
  if x.dtype!=np.float32 or x.shape!=xb.shape or not np.isfinite(x).all():raise ValueError("invalid emitted point domain")
  if f.dtype.kind not in "iu" or f.ndim!=2 or f.shape[1]!=3 or np.any(f<0) or np.any(f>=len(x)):raise ValueError("invalid face indices")
  if not same(f,c["area_reference_037_source_O_faces"]):raise ValueError("fixed-reference topology differs")
  anchors=b["typed_coarse_anchor_output_rows_018"]
  if not same(x[anchors],xb[anchors]):raise ValueError("typed anchors moved")
  if boundary(f)!=boundary(b["source_O_faces"]):raise ValueError("boundary edge set changed")
  owner=mod(OWNER);bm=xb.astype(np.float64)*1000.;xm=x.astype(np.float64)*1000.;ref=c["area_reference_037_points_world_m_f64"]*1000.
  area=owner.full_f32_area_gate(x,bm,ref,f,.1)
  result["full_f32_area_gate"]=area
  if area.get("pass") is not True:raise ValueError("full Float32 area/orientation gate failed")
  support=np.flatnonzero(np.any(x!=xb,axis=1))
  # No cached old-topology row is reused. Build current edge rows from actual support.
  if len(support):
   A,rhs,er=owner.build_edge_gradient_constraints(bm,f,support,.5)
   slack=np.asarray(A@(xm-bm)[support].ravel()).ravel()-rhs
   emin=float(slack.min(initial=1e30))
   if not np.isfinite(slack).all() or emin < -1e-7:raise ValueError("fresh edge-gradient gate failed")
  else:er={"active_edges":0,"inequality_rows":0,"zero_coordinate_step":True};emin=0.
  with np.load(BASE,allow_pickle=False) as z:source,levels=owner.compose_source_reference(z,R0)
  step=float(np.linalg.norm(xm-bm,axis=1).max(initial=0.))
  distance=float((np.linalg.norm(x.astype(np.float64)-source,axis=1)*1000.).max(initial=0.))
  if step>1.+1e-6 or distance>12.+1e-6:raise ValueError("trust/source cap failed")
  result.update({"arithmetic_pass":True,"anchor_rows":int(len(anchors)),"changed_point_rows":support.tolist(),"changed_face_rows":np.flatnonzero(np.any(f!=b["source_O_faces"],axis=1)).tolist(),"full_f32_area_gate":area,"fresh_edge_gate":er,"minimum_edge_slack_mm":emin,"maximum_step_mm":step,"maximum_source_distance_mm":distance,"source_reference":levels})
  if not a.arithmetic_only:
   for name,script,want in AUDITS:
    od=out/name;argv=[sys.executable,str(SLOT),sys.executable,str(script),str(cp),str(od)]
    if name=="audit-159":argv+=["--max-seconds","600"]
    t=time.monotonic()
    with (out/(name+".log")).open("w") as log:p=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
    result["commands"].append({"argv":argv,"returncode":p.returncode,"seconds":time.monotonic()-t,"script_sha256":want})
    if p.returncode:raise RuntimeError(name+" owner failed")
    if name=="audit-130":
     j=json.loads((od/"summary.json").read_text())
     if j.get("candidate_sha256")!=result["candidate_sha256"] or j.get("complete_surface_audit") is not True:raise ValueError("stale/incomplete self audit")
     pairs=np.load(od/"surface-self-pairs.npy",allow_pickle=False)
     if len(pairs)!=j.get("exact_PLC_surface_pairs"):raise ValueError("self pair count mismatch")
     result.update({"self_pairs":len(pairs),"contained_pairs":j.get("inside_node_bank_pairs")})
     if len(pairs):raise ValueError("exact self intersections")
    else:
     n,counts=validate_banks(json.loads((od/"report.json").read_text()),result["candidate_sha256"])
     bn,bc=validate_banks(json.loads(BASE159.read_text()),BASE_SHA)
     result.update({"surface_pairs":n,"surface_delta":n-bn,"contained_delta":result["contained_pairs"]-267,"per_bank_deltas":{k:counts[k]-bc[k] for k in counts if counts[k]!=bc[k]},"complete_banks":185})
   result["construction_progress"]=(result["surface_pairs"]<=2981 and result["contained_pairs"]<=267 and (result["surface_pairs"]<2981 or result["contained_pairs"]<267))
  for p,h in pins.items():
   if sha(p)!=h:raise ValueError("input/source changed during verification: "+str(p))
  result["status"]="ARITHMETIC_PASS" if a.arithmetic_only else "COMPLETE_EXACT_CONSTRUCTION_CHECKS"
 except Exception as exc:
  result["status"]="FAILED";result["error"]=str(exc)
 finally:
  result["seconds"]=time.monotonic()-start
  result["source_pins"]={str(p):h for p,h in pins.items()}
  (out/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
  print(json.dumps({k:v for k,v in result.items() if k in ["status","error","surface_pairs","contained_pairs","self_pairs","seconds","candidate_sha256","construction_progress"]}))
 if result["status"]=="FAILED":raise SystemExit(1)
if __name__=="__main__":main()
