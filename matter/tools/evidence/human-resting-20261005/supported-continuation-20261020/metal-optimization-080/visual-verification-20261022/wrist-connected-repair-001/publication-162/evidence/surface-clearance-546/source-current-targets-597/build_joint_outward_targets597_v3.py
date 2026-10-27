"""Reuse the 545 spherical-cap selector for current/source outward face targets.
Only construction directions change; 500 exact union rays and 185-bank endpoint
classification remain mandatory. No mesh or physical state is selected here.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,math
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
EXT=ROOT/"D/source-outward-proposal-500/build_outward_targets_multi_source_546.py"
EXT_SHA="dd61f0fb87f12c132166515a1e058e144d493a835cadbb2a47e6a2c8574e1e68"
DEPTH=ROOT/"D/contained-clearance-545/build_direct_depth_targets545.py"
DEPTH_SHA="2864d9547d2715c61c9c27b17bcfd11fac250197c47f4fd4870f8f6898cecb60"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def current_face_outward(points,faces,row,sign):
 x=np.asarray(points);f=np.asarray(faces);row=int(row)
 if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():raise ValueError("finite emitted F32 points required")
 if f.ndim!=2 or f.shape[1]!=3 or f.dtype.kind not in "iu" or row<0 or row>=len(f) or np.any(f[row]<0)or np.any(f[row]>=len(x)):raise ValueError("invalid face")
 if sign not in(-1,1):raise ValueError("unresolved O-to-I sign")
 t=x[f[row]].astype(float);v=np.cross(t[1]-t[0],t[2]-t[0]);n=float(np.linalg.norm(v))
 if not np.isfinite(n)or n<=0:raise ValueError("degenerate current face")
 return sign*v/n
def joint_direction(depth,source,current):
 d,proof=depth.depth_maximizing_unit_direction(np.asarray(source),np.asarray(current),minimum_cosine=.05)
 if min(float(d@source),float(d@current))<.05:raise ValueError("joint outward direction failed")
 return d,proof
def select_with_fallback(depth,source,current):
 try:
  d,proof=joint_direction(depth,source,current)
  return d,{**proof,"joint_source_current_feasible":True}
 except ValueError as exc:
  # Keep this target's existing source-only goal; do not drop or reweight it.
  # The unresolved reversed face is separately recorded for patch reconstruction.
  source=np.asarray(source,dtype=float)
  if source.shape!=(3,)or not np.isfinite(source).all()or abs(np.linalg.norm(source)-1)>2e-10:raise
  return source.copy(),{"joint_source_current_feasible":False,"joint_failure":str(exc),
   "method":"pinned 500 source-outward policy retained for unresolved current/source fold",
   "selected_source_hemisphere_cosine":1.0,"selected_depth_cosine":float(source@current)}
def strict_records(value,path=()):
 if isinstance(value,float)and not math.isfinite(value):
  if value==math.inf and path[-2:]==("target_selection","free_gap_m"):return None
  raise ValueError("unexpected nonfinite target data at "+"/".join(map(str,path)))
 if isinstance(value,dict):
  r={k:strict_records(v,path+(k,))for k,v in value.items()}
  if path==("target_selection",)and value.get("free_gap_m")==math.inf:
   r["free_gap_is_unbounded"]=True
  return r
 if isinstance(value,list):return[strict_records(v,path+(i,))for i,v in enumerate(value)]
 return value
def terminal_summary597(report,report_sha256):
 n=int(report["crossing_pairs"]);passing=int(report["passing_targets"])
 return {"status":"PASS_CERTIFIED_DIRECTION_TARGETS_ONLY"if passing==n else"INCOMPLETE_TARGETS",
  "passing_targets":passing,"crossing_pairs":n,"report_sha256":report_sha256}
def main(argv=None):
 argv=list(sys.argv[1:]if argv is None else argv)
 ap=argparse.ArgumentParser(add_help=False);ap.add_argument("--candidate",type=Path,required=True);ap.add_argument("--surface-report",type=Path,required=True);ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--direction-mode",required=True)
 a,_=ap.parse_known_args(argv)
 if a.direction_mode!="source-outward":raise ValueError("owner invocation must use source-outward; wrapper discloses operative joint direction")
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 for p,h in((EXT,EXT_SHA),(DEPTH,DEPTH_SHA)):
  if sha(p)!=h:raise ValueError("owner pin changed")
 ext=load("extension597",EXT);depth=load("depth597",DEPTH)
 if sha(ext.OWNER500)!=ext.OWNER500_SHA:raise ValueError("500 owner pin changed")
 owner=ext.load_module(ext.OWNER500);ext.install_extension(owner)
 original=owner.registered_source_outward_face_normal
 context={};provenance={}
 def registered(face,candidate,detail,base,orientation):
  s,pr=original(face,candidate,detail,base,orientation)
  c=current_face_outward(candidate["points_world_m"],candidate["source_O_faces"],face,orientation["coarse_outward_sign"])
  context.clear();context.update(face=int(face),source=s,current=c)
  pr={**pr,"current_outward_normal_world_unit":c.tolist(),"current_source_outward_dot":float(c@s),"coarse_outward_sign_from_orientation_owner":int(orientation["coarse_outward_sign"])}
  provenance[int(face)]={"source":s.tolist(),"current":c.tolist(),"source_provenance":pr}
  return s,pr
 def choose(mode,nearest,source):
  if mode!="source-outward"or not context:raise ValueError("joint wrapper invoked outside guarded face context")
  if not np.array_equal(source,context["source"]):raise ValueError("source face context mismatch")
  d,proof=select_with_fallback(depth,source,context["current"])
  provenance[context["face"]]["joint_selection"]=proof
  return d
 owner.registered_source_outward_face_normal=registered;owner.choose_proposal_direction=choose
 owner.main(argv)
 out=a.output_dir;rp=out/"report.json";records=out/"target-records.jsonl";npz=out/"targets.npz"
 r=json.loads(rp.read_text());rows=[json.loads(v)for v in records.read_text().splitlines()if v.strip()]
 passed=[]
 for v in rows:
  face=int(v["face_row"]);p=provenance.get(face)
  if p:v["source_current_outward_selection"]=p
  if str(v.get("status","")).startswith("PASS_"):
   if p is None or"joint_selection"not in p:raise ValueError("passing row lacks selector provenance")
   d=np.asarray(v["proposal_direction_world_unit"]);s=np.asarray(p["source"]);c=np.asarray(p["current"])
   if p["joint_selection"]["joint_source_current_feasible"]:
    if min(float(d@s),float(d@c))<.05:raise ValueError("passing direction violates joint cone")
   elif float(d@s)<1-2e-10:raise ValueError("fold exception changed source-only goal")
   v["direction_source"]=("545 current/source selector with cosine floors 0.05"if p["joint_selection"]["joint_source_current_feasible"]else"pinned 500 source-only direction retained for explicit unresolved current/source fold")
   passed.append(v)
 records.write_text("".join(json.dumps(strict_records(v),sort_keys=True,allow_nan=False)+"\n"for v in rows))
 r.update(direction_mode="source-current-outward-with-source-fold-exceptions",pinned_500_invocation_direction_mode="source-outward",
  direction_method="maximize current outward cosine within source cap when feasible; retain pinned 500 source-only goal for explicit infeasible fold witnesses; unchanged exact union-ray and all185 endpoint checks")
 r["joint_outward_direction_count"]=sum(bool(v["source_current_outward_selection"]["joint_selection"]["joint_source_current_feasible"])for v in passed)
 r["source_only_fold_exception_count"]=len(passed)-r["joint_outward_direction_count"]
 r["source_only_fold_exception_face_rows"]=sorted({int(v["face_row"])for v in passed if not v["source_current_outward_selection"]["joint_selection"]["joint_source_current_feasible"]})
 r["diagnostic_interval_serialization"]="positive-infinite target_selection.free_gap_m stored as null with free_gap_is_unbounded=true; no finite quantity or gate changed"
 r["source_current_direction_extension"]={"wrapper":str(Path(__file__).resolve()),"wrapper_sha256":sha(__file__),"selector":str(DEPTH),"selector_sha256":DEPTH_SHA,"source_extension":str(EXT),"source_extension_sha256":EXT_SHA,"owner500":str(ext.OWNER500),"owner500_sha256":sha(ext.OWNER500),"candidate_sha256":sha(a.candidate),"surface_report_sha256":sha(a.surface_report),"passing_rows":len(passed),"all_rows":len(rows),"target_records_sha256":sha(records),"targets_npz_sha256":sha(npz),"geometry_changed":False,"native_admission":False,"accepted_physical_steps":0}
 rp.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps(terminal_summary597(r,sha(rp))),flush=True)
 if len(passed)!=r["crossing_pairs"]:raise RuntimeError("incomplete joint-outward coverage; retain failures")
if __name__=="__main__":main()

