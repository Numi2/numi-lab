from pathlib import Path
import json,hashlib,subprocess,os,time
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019");E=R/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001";D=E/"sparse-contact-arithmetic-138";B=D/"build";base=E/"wholebody-attachment-source-rebind-036-ankle128-regenerated-deltas-001/native-diagnostic-036-1ms-008x125us-001";old=json.loads((base/"launch-manifest.snapshot.json").read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,lib,shader in [("sparse-matched-001",str(B),B/"candidate.metallib")]:
 N=D/name;N.mkdir(exist_ok=False)
 m=json.loads(json.dumps(old));m["argv"]=[str(x).replace(str(base),str(N)) for x in m["argv"]];m["environment"]={k:str(v).replace(str(base),str(N)) for k,v in m["environment"].items()};m["environment"]["DYLD_LIBRARY_PATH"]=lib;m["environment"]["NUMI_MATTER_METALLIB_OVERRIDE"]=str(shader)
 m["scope"]="Matched 1ms native D36 mechanics regression of sparse bone-contact force dispatch; not anatomical or long-run qualification";m["run_id"]=name;m["input_hashes"]={str(p):sha(p) for p in [Path(m["argv"][0]),Path(lib)/"libmetalrobo.dylib",shader]};m["base_manifest_sha256"]=sha(base/"launch-manifest.snapshot.json");(N/"launch-manifest.snapshot.json").write_text(json.dumps(m,indent=2))
 t=time.monotonic()
 with (N/"stdout.log").open("wb") as out,(N/"stderr.log").open("wb") as err:p=subprocess.run(m["argv"],cwd=R,env=dict(os.environ,**m["environment"]),stdout=out,stderr=err,timeout=500)
 r={"returncode":p.returncode,"wall_seconds":time.monotonic()-t,"scope":m["scope"],"stdout_sha256":sha(N/"stdout.log"),"stderr_sha256":sha(N/"stderr.log")};(N/"execution-receipt.json").write_text(json.dumps(r,indent=2));print(name,json.dumps(r),flush=True)
 if p.returncode:break
