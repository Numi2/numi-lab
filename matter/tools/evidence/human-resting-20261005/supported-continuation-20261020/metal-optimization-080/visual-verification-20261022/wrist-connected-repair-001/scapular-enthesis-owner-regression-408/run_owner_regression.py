#!/usr/bin/env python3
import hashlib,json,math,sys,time
from pathlib import Path
ROOT=Path("/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/current-manifest-recovery-004")
OWNER=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/source-route-weighting-owner-385/validation/src")
sys.path.insert(0,str(OWNER))
from numilab_human import model
SOURCE_SHA="280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975"
BASE=ROOT/"bilateral-all-upper-limb-bone-compile-001/baseline"
CAND=ROOT/"bilateral-all-upper-limb-bone-compile-001/candidate"
MAN=ROOT/"tendon-current-001/numi-human-tendon-attachments.manifest.json"
CMP=ROOT/"bilateral-all-upper-limb-bone-compile-001/compile-comparison-003.json"
def digest(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def endpoint(rows,muscle,kind):
 found=[r for r in rows if r["muscle"]==muscle and r["endpoint"]==kind]
 assert len(found)==1,(muscle,kind,len(found))
 return found[0]
def nearest(point,surface):
 best=(math.inf,-1)
 for i,tri in enumerate(surface["triangles"]):
  cp,_=model._tendon_closest_point_on_triangle(point,[surface["vertices"][v] for v in tri])
  d=math.dist(point,cp)
  if d<best[0]:best=(d,i)
 return best
def eval_one(surfaces,row):
 point=row["source_local_point_m"]; body=row["body_index"]; choices=[]
 for original in surfaces.get(body,[]):
  surface=dict(original); surface["body_index"]=body
  env,reason=model._numi_human_tendon_surface_envelope(point,surface,.012,.012,4.0,False)
  distance=env["surface_distance_m"] if env else nearest(point,surface)[0]
  choices.append((env is not None,distance,env,reason,surface))
 if not choices:return {"accepted":False,"reason":"body_has_no_registered_bone_surface"}
 ok,d,env,reason,s=min(choices,key=lambda x:(not x[0],x[1]))
 return {"accepted":ok,"reason":reason,"distance_m":d,"member_id":s.get("member_id"),"stable_id":s.get("stable_id"),
  "surface_patch_method":env.get("surface_patch_method") if env else None,
  "patch_radius_m":env.get("patch_radius_m") if env else None,
  "force_residual":env.get("force_residual") if env else None,
  "moment_residual_m":env.get("moment_residual_m") if env else None,
  "force_amplification":env.get("sampled_total_force_amplification") if env else None,
  "source_point_unchanged":bool(env and env.get("resolved_local_point_m")==point)}
def main():
 start=time.monotonic(); manifest=json.loads(MAN.read_text()); rows=manifest["endpoints"]
 cases={
  "DELT2 scapular insertion":endpoint(rows,"DELT2","insertion"),
  "TRIlong scapular origin":endpoint(rows,"TRIlong","origin"),
  "BIClong scapular origin":endpoint(rows,"BIClong","origin"),
  "CORB existing scapular control":endpoint(rows,"CORB","origin"),
  "BICshort existing scapular control":endpoint(rows,"BICshort","origin"),
  "PECM1 existing humeral control":endpoint(rows,"PECM1","origin")}
 result={};pins={}
 for label,directory in (("baseline",BASE),("candidate",CAND)):
  surfaces,meta,payload=model._numi_human_bone_envelope_surfaces(directory,SOURCE_SHA)
  pins[label]={"payload_sha256":meta["sha256"],"manifest_sha256":meta["manifest"]["sha256"],"path":str(payload)}
  result[label]={name:eval_one(surfaces,row) for name,row in cases.items()}
 checks={}
 for name in list(cases)[:3]:
  b=result["baseline"][name];c=result["candidate"][name]
  checks[name]={"baseline_rejected_by_12mm_gate":b["reason"]=="surface_distance_exceeds_gate" and b["distance_m"]>.012,
   "candidate_connected_patch_admitted":c["accepted"] and c["distance_m"]<=.012,
   "authored_endpoint_preserved":c["source_point_unchanged"]}
 for name in list(cases)[3:]:
  checks[name]={"baseline_admitted":result["baseline"][name]["accepted"],
   "candidate_still_admitted":result["candidate"][name]["accepted"],
   "authored_endpoint_preserved":result["candidate"][name]["source_point_unchanged"]}
 passed=all(all(v.values()) for v in checks.values())
 comparison=json.loads(CMP.read_text())
 report={"schema":"numi.human.scapular-enthesis-owner-regression.v1",
  "status":"passed_offline_owner_regression" if passed else "failed_offline_owner_regression",
  "admission":False,
  "evidence_boundary":"Offline existing-owner regression on unchanged source-authored route points. Candidate rigid bone registration remains provisional; this is not an independent enthesis measurement, coupled tissue qualification, force run, or native admission.",
  "owner_module":str(OWNER/"numilab_human/model.py"),"owner_module_sha256":digest(OWNER/"numilab_human/model.py"),
  "source_archive_sha256":SOURCE_SHA,"attachment_manifest_sha256":digest(MAN),
  "candidate_registration_sha256":"b42e92c21f49b2fb7612447f0f0678295cae894ddffbe80cdf11f210ccee6a73",
  "compile_comparison":{"sha256":digest(CMP),"status":comparison.get("status"),"admission":comparison.get("admission")},
  "owner_gates":{"max_surface_distance_m":.012,"max_patch_radius_m":.012,"max_force_amplification":4.0,"migrate_endpoint_to_surface":False},
  "artifacts":pins,"checks":checks,"results":result,"elapsed_seconds":time.monotonic()-start}
 out=Path(__file__).resolve().parent/"owner-regression-report.json"
 out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":report["status"],"report":str(out),"checks":checks},indent=2))
 if not passed:raise SystemExit(1)
if __name__=="__main__":main()

