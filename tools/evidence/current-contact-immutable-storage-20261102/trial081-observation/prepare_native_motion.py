from pathlib import Path
import json,hashlib
S=Path(__file__).resolve().parent.parent;N=Path(__file__).resolve().parent
R=Path("/Users/numi/Work/skin-fat-native-integration-209")
def pin(p):
 p=Path(p)
 with p.open("rb") as f: digest=hashlib.file_digest(f,"sha256").hexdigest()
 return {"path":str(p),"sha256":digest,"bytes":p.stat().st_size}
base=S/"native-import-079/install-motion.plan.json";plan=json.loads(base.read_text());newbuild=S/"immutable-contact-storage-080/retry-002/build-source-device-manifest.json";build=json.loads(newbuild.read_text());assert build["status"]=="PASS" and all(s["exit_code"]==0 for s in build["stages"])
assert pin(build["binary"]["path"])==build["binary"]
for k,v in build["runtime_dependencies"].items():assert pin(k)==v
# No source checks are skipped: the full motion command performs the complete
# source import itself. 078 is reference evidence, not a new-binary preflight.
seen=set()
def verify(x):
 if isinstance(x,dict):
  if "path" in x and "sha256" in x and x["path"] not in seen:
   seen.add(x["path"]);assert pin(x["path"])["sha256"]==x["sha256"],x["path"]
  for k,v in x.items():
   if isinstance(k,str) and k.startswith("/") and isinstance(v,dict) and "sha256" in v and k not in seen:
    seen.add(k);assert pin(k)["sha256"]==v["sha256"],k
   verify(v)
 elif isinstance(x,list):
  for v in x:verify(v)
verify(plan["input_pins"]);verify(plan["role_input_pins"])
old=str(S/"native-import-079")
def rewrite(x):
 if isinstance(x,str):return x.replace(old,str(N))
 if isinstance(x,list):return [rewrite(v) for v in x]
 if isinstance(x,dict):return {k:rewrite(v) for k,v in x.items()}
 return x
stage=rewrite(plan["stages"][0]);stage["environment"]["NUMI_HUMAN_CONTACT_WORKING_SET_GUARD"]="1"
plan["stages"]=[stage];plan["native_binary"]=build["binary"];plan["runtime_dependencies"]={k:{"sha256":v["sha256"],"bytes":v["bytes"]} for k,v in build["runtime_dependencies"].items()};plan["build_manifest"]=pin(newbuild);plan["status"]="FULL_NATIVE_MOTION_PLAN_CURRENT_STORAGE_BUILD_COMPLETE_SOURCE_CHECKS_REQUIRED";plan["source_preflight_reuse"]={"reused":False,"reason":"native binary and adapter storage changed; full motion run repeats all source admission checks","reference_execution":plan.pop("source_preflight_execution")};plan["source_qualifier"]=pin(Path(__file__));plan["retained_prior_plan"]=pin(base);plan["physical_equations_changed"]=False;plan["contact_rows_removed"]=0;plan["measurement_scope"]="two-step full-scene release diagnostic; explicit memory budget; no sustained qualification"
(N/"install-motion.plan.json").write_text(json.dumps(plan,indent=2)+"\n");print("plan_ready",pin(N/"install-motion.plan.json"),"verified_inputs",len(seen),flush=True)
