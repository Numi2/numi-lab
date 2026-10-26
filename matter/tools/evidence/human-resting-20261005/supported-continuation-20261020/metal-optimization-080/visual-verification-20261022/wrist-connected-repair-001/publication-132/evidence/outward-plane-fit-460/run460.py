from pathlib import Path
import json,hashlib,subprocess,time,os
D=Path("/Users/numi/Work/numi-human-connected-surface-069");PY="/Users/n/numi-human-prep-venv-20261005/bin/python";sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
P=D/"D/outward-plane-fit-460/run-001-launch";P.mkdir(parents=True,exist_ok=False)
cur=D/"D/constraint-project-backtrack-456/run-001/backtrack-373/pass-006/candidate.npz";prep=D/"D/connected-continuation-prep-457/run-001"
assert sha(cur)=="1ac004ef6f6646ce4720360c50b7fd9855336a4967c3e97b199cf9bc0350049a"
wrapper=D/"D/D/fixed-bone-outward-plane-458/run_grouped_fit_458.py";helper=wrapper.with_name("fixed_bone_outward_plane_458.py")
a=json.loads((D/"D/reseeded-fit-434/run-001-launch/argv.json").read_text());a[1]=str(wrapper)
updates={"--base-candidate":cur,"--raw-trial":cur,"--area-reference-npz":cur,"--self-pairs":prep/"self-seeds/merged-self-pair-seeds.npy","--point-target-npz":prep/"point-targets-390/nearest-union-exits.npz","--point-target-report":prep/"point-targets-390/nearest-union-exits.json","--face-target-npz":prep/"surface-targets-335/targets.npz","--face-target-report":prep/"surface-targets-335/report.json","--point-depth-rows-npz":D/"D/point-depth-refresh-457/run-001/point-depth-rows.npz","--point-depth-rows-report":D/"D/point-depth-refresh-457/run-001/365-compatible-report.json","--fixed-boundary-npz":D/"D/fixed-bone-guard-refresh-457/run-001/introduced-skin-bone-barriers.npz","--fixed-boundary-report":D/"D/fixed-bone-guard-refresh-457/run-001/report.json","--output":D/"barycentric-connected-clearance-170/outward-plane-fit-460/run-001"}
for k,v in updates.items():a[a.index(k)+1]=str(v)
audit=cur.parent/"audit-159/report.json";env={"NUMI_FIXED_BONE_OUTWARD_AUDIT159_PATH_458":str(audit),"NUMI_FIXED_BONE_OUTWARD_AUDIT159_SHA256_458":sha(audit)}
assert json.loads(audit.read_text())["candidate_sha256"]==sha(cur)
pins={str(p):sha(p)for p in [wrapper,helper,prep/"preparation-report.json",audit,Path(__file__)]+[p for p in updates.values()if p.is_file()]}
(P/"argv.json").write_text(json.dumps(a,indent=2)+"\n");(P/"preflight.json").write_text(json.dumps(dict(pins=pins,env=env,variant="actual outward bone triangle planes458; matched456; unchanged hard physics construction constraints",native_admission=False,accepted_physical_steps=0),indent=2)+"\n")
start=time.monotonic()
with(P/"execution.log").open("x")as log:r=subprocess.run([a[0],"/tmp/numi-human-geometry-slot.py",*a],env={**os.environ,**env},stdout=log,stderr=subprocess.STDOUT)
assert all(sha(p)==v for p,v in pins.items())
result=dict(returncode=r.returncode,seconds=time.monotonic()-start,runtime_pins_unchanged=True,native_admission=False,accepted_physical_steps=0)
(P/"execution.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
