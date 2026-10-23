from pathlib import Path
import json, hashlib, subprocess, os, time, datetime
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E=R/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
D=E/"local-remesh-native-integration-131"
B=E/"wholebody-attachment-source-rebind-036-ankle128-regenerated-deltas-001"
old=B/"native-diagnostic-036-1ms-008x125us-001"
N=D/"native-diagnostic-131-1ms-008x125us-001"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
if not (D/"build-manifest.json").is_file():raise SystemExit("missing successful build")
N.mkdir(exist_ok=False)
m=json.loads((old/"launch-manifest.snapshot.json").read_text())
def replace(s):return str(s).replace(str(old),str(N)).replace(str(B/"build-main-0a693-004"),str(D/"build-native-001"))
m["argv"]=[replace(s) for s in m["argv"]]
m["environment"]={k:replace(v) for k,v in m["environment"].items()}
m["run_id"]=N.name
m["scope"]="Same-domain skin/fat local remesh native mechanics diagnostic; no new arm surface or anatomy qualification"
m["local_volume_remesh_input_sha256"]=sha(D/"source-overlay/apps/NumiHumanD36Ring0RemeshInput.hpp")
m["base_candidate_identity_sha256"]=sha(B/"D36-candidate-identity.json")
m["base_launch_snapshot_path"]=str(old/"launch-manifest.snapshot.json")
m["base_launch_snapshot_sha256"]=sha(old/"launch-manifest.snapshot.json")
m["base_input_hashes"]=m.pop("input_hashes")
for k in ("compile_result_sha256", "link_result_sha256"):m[k]=sha(D/"build-native-001"/(k.replace("_result_sha256","-result.json")))
m["candidate_build_manifest_sha256"]=sha(D/"build-manifest.json")
m["binary_sha256"]=sha(m["argv"][0])
inputs={p for p in m["argv"] if Path(p).is_file()}
inputs.update(v for v in m["environment"].values() if Path(v).is_file())
inputs.update(str(p) for p in (D/"source-overlay/apps").iterdir() if p.is_file())
# DYLD_LIBRARY_PATH is a directory. Hash the selected library file explicitly.
libraries=[]
for folder in m["environment"].get("DYLD_LIBRARY_PATH", "").split(":"):
    p=Path(folder)/"libmetalrobo.dylib"
    if p.is_file():libraries.append(str(p.resolve()))
if len(libraries)!=1:raise SystemExit("runtime library selection is ambiguous")
inputs.update(libraries)
m["actual_runtime_libraries"]=libraries
m["input_hashes"]={p:sha(p) for p in sorted(inputs)}
m["argv_sha256"]=hashlib.sha256(json.dumps(m["argv"],sort_keys=True).encode()).hexdigest()
m["environment_sha256"]=hashlib.sha256(json.dumps(m["environment"],sort_keys=True).encode()).hexdigest()
m["created_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
m["physical_qualification"]=False
snap=N/"launch-manifest.snapshot.json";snap.write_text(json.dumps(m,indent=2)+"\n")
env=dict(os.environ,**m["environment"]);start=time.monotonic();err=None
with (N/"stdout.log").open("wb") as out,(N/"stderr.log").open("wb") as er:
    try:r=subprocess.run(m["argv"],env=env,cwd=R,stdout=out,stderr=er);rc=r.returncode
    except Exception as ex:err=repr(ex);rc=None
receipt={"launch_manifest_sha256":sha(snap),"returncode":rc,"launch_error":err,"elapsed_seconds":time.monotonic()-start,"stdout_sha256":sha(N/"stdout.log"),"stderr_sha256":sha(N/"stderr.log"),"physical_qualification":False}
(N/"execution-receipt.raw.json").write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps(receipt),flush=True)
