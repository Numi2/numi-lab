from pathlib import Path
import json,hashlib,subprocess,time
D=Path("/Users/numi/Work/numi-human-connected-surface-069");PY="/Users/n/numi-human-prep-venv-20261005/bin/python";sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
P=D/"D/refresh-438";P.mkdir(exist_ok=False)
cur=D/"D/seed-preserving-backtrack-436/run-001/backtrack-373/pass-015";old=D/"D/postfit-backtrack-427/run-001/backtrack-373/pass-004"
def run(name,args,slot=True):
 argv=[PY,"/tmp/numi-human-geometry-slot.py",PY,*map(str,args)]if slot else[PY,*map(str,args)]
 (P/(name+"-argv.json")).write_text(json.dumps(argv,indent=2));t=time.monotonic()
 with(P/(name+".log")).open("x")as f:r=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT)
 (P/(name+"-execution.json")).write_text(json.dumps(dict(returncode=r.returncode,seconds=time.monotonic()-t),indent=2));assert r.returncode==0
depth=D/"D/point-depth-refresh-438/run-001"
a=[D/"D/point-depth-refresh-413/refresh_point_depth_rows.py","--candidate",cur/"candidate.npz","--point-summary",cur/"audit-130/summary.json","--surface-report",cur/"audit-159/report.json","--query-dir",cur/"audit-130/query","--output-dir",depth]
for key,p in {"candidate":cur/"candidate.npz","point-summary":cur/"audit-130/summary.json","surface-report":cur/"audit-159/report.json","query-receipt":cur/"audit-130/query/query-receipt.json"}.items():a+=["--expected-"+key+"-sha256",sha(p)]
a+=["--expected-owner-sha256","dc8078515dd12b2edc62c0bc75759a7cb9472c9914e4704ecd41e04c4d5378e1"]
run("depth",a)
guards=D/"D/fixed-bone-guard-refresh-438/run-001";prior=D/"D/fixed-bone-guard-refresh-428/run-001"
a=[D/"D/fixed-bone-guard-refresh-413/refresh_fixed_bone_guards.py","--previous-candidate",old/"candidate.npz","--previous-surface-report",old/"audit-159/report.json","--prior-guard-report",prior/"report.json","--prior-guard-npz",prior/"introduced-skin-bone-barriers.npz","--candidate",cur/"candidate.npz","--surface-report",cur/"audit-159/report.json","--output-dir",guards];run("guards",a)
trials=[]
for name,runname in [("seed-preserving-backtrack-436","run-001"),("uniform-control-432","run-001")]:
 for p in sorted((D/"D"/name/runname/"backtrack-373").glob("pass-*/audit-130/surface-self-pairs.npy")):
  summ=p.parent/"summary.json"
  if json.loads(summ.read_text()).get("exact_PLC_surface_pairs",0)<=0:continue
  c=p.parent.parent/"candidate.npz";trials.append(dict(candidate_path=str(c),candidate_sha256=sha(c),audit130_path=str(summ),audit130_sha256=sha(summ),pairs_path=str(p),pairs_sha256=sha(p)))
mp=P/"failed-trials.json";mp.write_text(json.dumps(dict(trials=trials),indent=2))
prior=D/"D/connected-continuation-prep-433/run-001/self-seeds"
items={"candidate":cur/"candidate.npz","audit130":cur/"audit-130/summary.json","audit159":cur/"audit-159/report.json","prior-seed-report":prior/"report.json","prior-seed-pairs":prior/"merged-self-pair-seeds.npy","failed-trials-manifest":mp,"point-depth-npz":depth/"point-depth-rows.npz","point-depth-report":depth/"365-compatible-report.json"}
a=[D/"D/prepare_connected_continuation_inputs_417.py"]
for k,v in items.items():a+=["--"+k,str(v),"--"+k+"-sha256",sha(v)]
a+=["--query-dir",str(cur/"audit-130/query"),"--output-root",str(D/"D/connected-continuation-prep-438/run-001")]
run("prep",a,False)
print("438 inputs refreshed")
