"""Verify complete 597 target artifacts left before a terminal-summary failure."""
from pathlib import Path
import argparse,hashlib,json,numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(),parse_constant=lambda v:(_ for _ in()).throw(ValueError("nonfinite JSON "+v)))
def verify(folder,state_path):
 s=read(state_path)
 for k in("candidate","point_summary","surface_report"):
  if sha(s[k])!=s[k+"_sha256"]:raise ValueError("state pin "+k)
 rp=folder/"report.json";cp=folder/"target-records.jsonl";zp=folder/"targets.npz";r=read(rp);e=r["source_current_direction_extension"]
 if e["candidate_sha256"]!=s["candidate_sha256"]or e["surface_report_sha256"]!=s["surface_report_sha256"]:raise ValueError("target baseline differs")
 for k in("wrapper","selector","source_extension","owner500"):
  if sha(e[k])!=e[k+"_sha256"]:raise ValueError("source changed "+k)
 if sha(zp)!=e["targets_npz_sha256"]or sha(cp)!=e["target_records_sha256"]:raise ValueError("artifact pin")
 records=[json.loads(a,parse_constant=lambda v:(_ for _ in()).throw(ValueError(v)))for a in cp.read_text().splitlines()if a.strip()]
 n=r["crossing_pairs"]
 if r["passing_targets"]!=n or len(records)!=n or r["banks"]!=185 or r["failure_counts"]or not r["matched_159_complete"]:raise ValueError("incomplete exact target coverage")
 with np.load(zp,allow_pickle=False)as z:
  if z["target_world_f32_m"].dtype!=np.float32 or z["target_world_f32_m"].shape!=(n,3)or not np.isfinite(z["target_world_f32_m"]).all()or not np.all(z["target_exactly_outside_all185"]):raise ValueError("invalid target field")
  actual_dots=[];joint=0;exceptions=[]
  for i,v in enumerate(records):
   if not v["status"].startswith("PASS_")or v["pair_index"]!=int(z["pair_index"][i]):raise ValueError("record identity/status")
   sel=v["target_selection"]
   if v.get("target_exactly_outside_all185")is not True or sel["classified_bank_count"]!=185 or sel["target_exactly_outside_all_banks"]is not True or not sel["status"].startswith("PASS_"):raise ValueError("incomplete all185 endpoint certificate")
   for key,zkey in(("target_world_f32_m","target_world_f32_m"),("proposal_direction_world_unit","proposal_direction_world_unit"),("inside_sample_world_m","inside_sample_world_m"),("source_surface_normal_world_unit","source_surface_normal_world_unit")):
    if not np.array_equal(np.asarray(v[key]),z[zkey][i]):raise ValueError("NPZ/record mismatch "+key)
   p=v["source_current_outward_selection"];source=np.asarray(p["source"]);current=np.asarray(p["current"]);d=np.asarray(v["proposal_direction_world_unit"]);a=z["target_world_f32_m"][i].astype(float)-z["inside_sample_world_m"][i];a/=np.linalg.norm(a)
   if p["joint_selection"]["joint_source_current_feasible"]:
    joint+=1
    if min(float(d@source),float(d@current))<.05 or min(float(a@source),float(a@current))<=0:raise ValueError("joint proposal or actual emitted direction")
    actual_dots.append([float(a@source),float(a@current)])
   else:
    exceptions.append(int(v["face_row"]))
    if not np.array_equal(d,source)or float(a@source)<=0:raise ValueError("source exception changed goal")
 if joint!=r["joint_outward_direction_count"]or n-joint!=r["source_only_fold_exception_count"]:raise ValueError("counts mismatch")
 return {"status":"PASS_CERTIFIED_DIRECTION_TARGETS_ONLY","classification":"complete construction targets independently verified after wrapper terminal-summary KeyError; no geometry or physical admission","state":str(state_path),"state_sha256":sha(state_path),"candidate_sha256":s["candidate_sha256"],"source_pins":{str(Path(__file__).resolve()):sha(__file__),**{e[k]:e[k+"_sha256"]for k in("wrapper","selector","source_extension","owner500")}},"artifact_pins":{str(p):sha(p)for p in(rp,cp,zp)},"target_count":n,"exact_endpoint_bank_count":185,"joint_source_current_targets":joint,"source_only_exception_targets":n-joint,"source_only_exception_faces":sorted(set(exceptions)),"minimum_actual_joint_source_cosine":min(x[0]for x in actual_dots),"minimum_actual_joint_current_cosine":min(x[1]for x in actual_dots),"native_admission":False,"accepted_physical_steps":0}
def main():
 p=argparse.ArgumentParser();p.add_argument("--folder",type=Path,required=True);p.add_argument("--state",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 r=verify(a.folder,a.state);a.output.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+"\n");print(json.dumps({k:r[k]for k in("status","target_count","joint_source_current_targets","source_only_exception_targets","minimum_actual_joint_source_cosine","minimum_actual_joint_current_cosine")}))
if __name__=="__main__":main()

