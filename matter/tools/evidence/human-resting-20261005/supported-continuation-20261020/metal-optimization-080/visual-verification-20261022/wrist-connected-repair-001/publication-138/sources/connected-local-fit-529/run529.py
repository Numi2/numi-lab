from pathlib import Path
import json,hashlib,subprocess,time,sys
D=Path("/Users/numi/Work/numi-human-connected-surface-069");PY="/Users/n/numi-human-prep-venv-20261005/bin/python";sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
entry=Path(sys.argv[1]);assert sha(entry)==sys.argv[2];tag=sys.argv[3];flags=sys.argv[4:]
sel=D/"D/finite-triangle-backtrack-512/all-pairs-001/backtrack-373/selected-result.json";s=json.loads(sel.read_text());cur=Path(s["candidate"]);audit=Path(s["surface_report"])
fresh=D/"D/refresh-515/run-001";prep=fresh/"connected-continuation-prep491/run-001";guard=fresh/"fixed-bone-guard-refresh491/run-001";depth=fresh/"point-depth-refresh491/run-001"
pr=json.loads((prep/"preparation-report.json").read_text());assert pr["candidate_sha256"]==sha(cur)
P=D/f"D/connected-local-fit-529/{tag}-launch";P.mkdir(parents=True,exist_ok=False)
cfg=json.loads((D/"D/area-blocker-fit-506/run-001-launch/matched-inputs.json").read_text())
paths=dict(candidate=cur,audit159=audit,face_target_npz=Path(pr["335"]["npz_path"]),point_target_npz=Path(pr["390"]["npz_path"]));cfg["inputs"]={k:dict(path=str(p),sha256=sha(p))for k,p in paths.items()};config=P/"matched-inputs.json";config.write_text(json.dumps(cfg,indent=2)+"\n")
out=D/f"barycentric-connected-clearance-170/connected-local-fit-529/{tag}"
a=json.loads((D/"D/finite-triangle-fit-511/all-pairs-001-launch/argv.json").read_text());a[1]=str(entry)
# Replace the previous wrapper opt-ins with the requested entry's explicit options.
for flag in ["--anchor-compatible-finite-triangle-513","--anchor-compatible-finite-triangle-all-pairs-513"]:
 a.remove(flag)
updates={"--base-candidate":cur,"--raw-trial":cur,"--area-reference-npz":cur,"--self-pairs":pr["self_seeds"]["path"],"--point-target-npz":paths["point_target_npz"],"--point-target-report":pr["390"]["report_path"],"--face-target-npz":paths["face_target_npz"],"--face-target-report":pr["335"]["report_path"],"--point-depth-rows-npz":depth/"point-depth-rows.npz","--point-depth-rows-report":depth/"365-compatible-report.json","--fixed-boundary-npz":guard/"introduced-skin-bone-barriers.npz","--fixed-boundary-report":guard/"report.json","--matched-inputs-json-490":config,"--audit159-490":audit,"--output":out}
for k,v in updates.items():a[a.index(k)+1]=str(v)
a.extend(flags)
pins={str(entry):sha(entry),str(Path(__file__)):sha(__file__)}
for v in a[2:]:
 p=Path(v)
 if p.is_file():pins[str(p)]=sha(p)
(P/"argv.json").write_text(json.dumps(a,indent=2)+"\n");(P/"preflight.json").write_text(json.dumps(dict(pins=pins,native_admission=False,accepted_physical_steps=0),indent=2)+"\n")
t=time.monotonic()
with(P/"execution.log").open("x")as f:r=subprocess.run([PY,"/tmp/numi-human-geometry-slot.py",*a],stdout=f,stderr=subprocess.STDOUT)
assert all(sha(p)==v for p,v in pins.items())
(P/"execution.json").write_text(json.dumps(dict(returncode=r.returncode,seconds=time.monotonic()-t,pins_unchanged=True),indent=2)+"\n")
if r.returncode:raise RuntimeError("517fitfailed")
fr=json.loads((out/"report.json").read_text());print("fit",fr.get("status"),flush=True)
if fr.get("status")=="CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED":
 checker=D/"D/typed-point-regression-524/run_checked_proposal_525.py";assert sha(checker)=="64845a64a4fc5fe1c71f6b5114ca4d589a1f0957f68649f78069a6ae0ac4a345"
 a=[PY,str(checker),str(sel),str(out),str(prep),str(guard),str(D/f"D/connected-local-backtrack-530/{tag}")]
 (P/"check-argv.json").write_text(json.dumps(a,indent=2)+"\n");t=time.monotonic()
 with(P/"check.log").open("x")as f:r=subprocess.run(a,stdout=f,stderr=subprocess.STDOUT)
 assert all(sha(p)==v for p,v in pins.items())
 (P/"check-execution.json").write_text(json.dumps(dict(returncode=r.returncode,seconds=time.monotonic()-t,pins_unchanged=True),indent=2)+"\n");print("checked",r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
