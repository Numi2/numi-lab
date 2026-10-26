from pathlib import Path
import json,hashlib,subprocess,time,os,sys
D=Path("/Users/numi/Work/numi-human-connected-surface-069");PY="/Users/n/numi-human-prep-venv-20261005/bin/python"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
P=D/"D/area-blocker-fit-497/run-001-launch";P.mkdir(parents=True,exist_ok=False)
wrapper=D/"D/D/area-blocker-zero-bounds-496/run_matched_fit_496.py";helper=wrapper.with_name("area_blocker_zero_bounds_496.py")
assert sha(wrapper)==sys.argv[1];assert sha(helper)=="92c7e6a3bbd53611c3dca55ffec59189ff83614ecdb9bc415309782a176ab5e4"
a=json.loads((D/"D/combined-fit-486/run-001-launch/argv.json").read_text());a[1]=str(wrapper);out=D/"barycentric-connected-clearance-170/area-blocker-fit-497/run-001";a[a.index("--output")+1]=str(out)
a.extend(["--matched-inputs-json-490",str(D/"D/D/component-lp-decomposition-490/matched_inputs_476.json"),"--audit159-490",a[a.index("--base-candidate")+1].replace("candidate.npz","audit-159/report.json"),"--zero-bound-486-area-blockers-496"])
pins={str(p):sha(p)for p in [wrapper,helper,Path(__file__)]}
(P/"argv.json").write_text(json.dumps(a,indent=2)+"\n");(P/"preflight.json").write_text(json.dumps(dict(pins=pins,native_admission=False,accepted_physical_steps=0),indent=2)+"\n")
start=time.monotonic()
with(P/"fit.log").open("x")as f:r=subprocess.run([PY,"/tmp/numi-human-geometry-slot.py",*a],stdout=f,stderr=subprocess.STDOUT)
assert all(sha(p)==h for p,h in pins.items())
(P/"execution.json").write_text(json.dumps(dict(returncode=r.returncode,seconds=time.monotonic()-start,pins_unchanged=True,native_admission=False,accepted_physical_steps=0),indent=2)+"\n")
if r.returncode:raise RuntimeError("497fit failed")
report=json.loads((out/"report.json").read_text())
if report.get("status")=="CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED":
 checker=D/"D/checked-proposal-459/run_checked_proposal_459.py";assert sha(checker)=="f1c0c9828757b2760186da40e12beba0bde9ac5185d9b1fe8521eeb16ebe5072"
 a=[PY,str(checker),str(D/"D/outward-plane-backtrack-461/run-001/backtrack-373/selected-result.json"),str(out),str(D/"D/connected-continuation-prep476/run-001"),str(D/"D/fixed-bone-guard-refresh476/run-001"),str(D/"D/area-blocker-backtrack-498/run-001")]
 (P/"check-argv.json").write_text(json.dumps(a,indent=2)+"\n")
 with(P/"check.log").open("x")as f:r=subprocess.run(a,stdout=f,stderr=subprocess.STDOUT)
 assert all(sha(p)==h for p,h in pins.items())
 print("checked",r.returncode)
