from pathlib import Path
import json,hashlib,subprocess,time,sys
D=Path("/Users/numi/Work/numi-human-connected-surface-069");PY="/Users/n/numi-human-prep-venv-20261005/bin/python";sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
entry=Path(sys.argv[1]);expected=sys.argv[2];tag=sys.argv[3];flags=sys.argv[4:]
assert sha(entry)==expected
old=D/"D/area-blocker-fit-506/run-001-launch";a=json.loads((old/"argv.json").read_text());a[1]=str(entry)
P=D/f"D/finite-triangle-fit-511/{tag}-launch";P.mkdir(parents=True,exist_ok=False)
out=D/f"barycentric-connected-clearance-170/finite-triangle-fit-511/{tag}"
a[a.index("--output")+1]=str(out);a.extend(flags)
pins={str(entry):expected,str(Path(__file__)):sha(__file__)}
for i,v in enumerate(a[2:]):
 p=Path(v)
 if p.is_file():pins[str(p)]=sha(p)
(P/"argv.json").write_text(json.dumps(a,indent=2)+"\n");(P/"preflight.json").write_text(json.dumps(dict(pins=pins,native_admission=False,accepted_physical_steps=0),indent=2)+"\n")
t=time.monotonic()
with(P/"execution.log").open("x")as f:r=subprocess.run([PY,"/tmp/numi-human-geometry-slot.py",*a],stdout=f,stderr=subprocess.STDOUT)
assert all(sha(p)==v for p,v in pins.items())
(P/"execution.json").write_text(json.dumps(dict(returncode=r.returncode,seconds=time.monotonic()-t,pins_unchanged=True,native_admission=False,accepted_physical_steps=0),indent=2)+"\n")
if r.returncode:raise RuntimeError("511fitfailed")
fr=json.loads((out/"report.json").read_text())
print("fit",fr.get("status"),flush=True)
if fr.get("status")=="CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED":
 sel=D/"D/refreshed-connected-continuation-492/run-001/iteration-002/check/backtrack-373/selected-result.json";I=D/"D/refreshed-connected-continuation-492/run-001/iteration-003/refresh"
 checker=D/"D/checked-proposal-459/run_checked_proposal_459.py";assert sha(checker)=="f1c0c9828757b2760186da40e12beba0bde9ac5185d9b1fe8521eeb16ebe5072"
 a=[PY,str(checker),str(sel),str(out),str(I/"connected-continuation-prep500/run-001"),str(I/"fixed-bone-guard-refresh491/run-001"),str(D/f"D/finite-triangle-backtrack-512/{tag}")]
 (P/"check-argv.json").write_text(json.dumps(a,indent=2)+"\n");t=time.monotonic()
 with(P/"check.log").open("x")as f:r=subprocess.run(a,stdout=f,stderr=subprocess.STDOUT)
 assert all(sha(p)==v for p,v in pins.items())
 (P/"check-execution.json").write_text(json.dumps(dict(returncode=r.returncode,seconds=time.monotonic()-t,pins_unchanged=True),indent=2)+"\n")
 print("checked",r.returncode,flush=True)
