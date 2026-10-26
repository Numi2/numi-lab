from pathlib import Path
import json,hashlib,subprocess,time,sys
D=Path("/Users/numi/Work/numi-human-connected-surface-069");PY="/Users/n/numi-human-prep-venv-20261005/bin/python";sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
R=D/"D/refreshed-connected-continuation-492/run-001";I=R/"iteration-003";sel=R/"iteration-002/check/backtrack-373/selected-result.json";s=json.loads(sel.read_text());cur=Path(s["candidate"]);audit=Path(s["surface_report"])
prep=I/"refresh/connected-continuation-prep500/run-001";guard=I/"refresh/fixed-bone-guard-refresh491/run-001";depth=I/"refresh/point-depth-refresh491/run-001"
pr=json.loads((prep/"preparation-report.json").read_text());assert pr["candidate_sha256"]==sha(cur);assert pr["335"]["passing_targets"]==3090;assert pr["390"]["passing_targets"]==283
wrapper=D/"D/D/area-blocker-zero-bounds-505/run_matched_fit_505.py";helper=wrapper.with_name("area_blocker_zero_bounds_505.py")
assert sha(wrapper)==sys.argv[1];assert sha(helper)==sys.argv[2]
P=D/"D/area-blocker-fit-506/run-001-launch";P.mkdir(parents=True,exist_ok=False)
cfg=json.loads((D/"D/D/component-lp-decomposition-490/matched_inputs_476.json").read_text());paths=dict(candidate=cur,audit159=audit,face_target_npz=Path(pr["335"]["npz_path"]),point_target_npz=Path(pr["390"]["npz_path"]));cfg["inputs"]={k:dict(path=str(p),sha256=sha(p))for k,p in paths.items()};config=P/"matched-inputs.json";config.write_text(json.dumps(cfg,indent=2)+"\n")
out=D/"barycentric-connected-clearance-170/area-blocker-fit-506/run-001";a=json.loads((D/"D/combined-fit-486/run-001-launch/argv.json").read_text());a[1]=str(wrapper)
updates={"--base-candidate":cur,"--raw-trial":cur,"--area-reference-npz":cur,"--self-pairs":pr["self_seeds"]["path"],"--point-target-npz":paths["point_target_npz"],"--point-target-report":pr["390"]["report_path"],"--face-target-npz":paths["face_target_npz"],"--face-target-report":pr["335"]["report_path"],"--point-depth-rows-npz":depth/"point-depth-rows.npz","--point-depth-rows-report":depth/"365-compatible-report.json","--fixed-boundary-npz":guard/"introduced-skin-bone-barriers.npz","--fixed-boundary-report":guard/"report.json","--output":out}
for k,v in updates.items():a[a.index(k)+1]=str(v)
a.extend(["--matched-inputs-json-490",str(config),"--audit159-490",str(audit),"--zero-bound-486-area-blockers-505"])
pins={str(p):sha(p)for p in [wrapper,helper,Path(__file__),config,prep/"preparation-report.json",audit]+[Path(v)for v in updates.values()if Path(v).is_file()]}
(P/"argv.json").write_text(json.dumps(a,indent=2)+"\n");(P/"preflight.json").write_text(json.dumps(dict(pins=pins,native_admission=False,accepted_physical_steps=0),indent=2)+"\n")
t=time.monotonic()
with(P/"execution.log").open("x")as f:r=subprocess.run([PY,"/tmp/numi-human-geometry-slot.py",*a],stdout=f,stderr=subprocess.STDOUT)
assert all(sha(p)==v for p,v in pins.items())
(P/"execution.json").write_text(json.dumps(dict(returncode=r.returncode,seconds=time.monotonic()-t,pins_unchanged=True,native_admission=False,accepted_physical_steps=0),indent=2)+"\n")
if r.returncode:raise RuntimeError("506fitfailed")
fr=json.loads((out/"report.json").read_text())
if fr.get("status")=="CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED":
 checker=D/"D/checked-proposal-459/run_checked_proposal_459.py";assert sha(checker)=="f1c0c9828757b2760186da40e12beba0bde9ac5185d9b1fe8521eeb16ebe5072"
 a=[PY,str(checker),str(sel),str(out),str(prep),str(guard),str(D/"D/area-blocker-backtrack-507/run-001")];(P/"check-argv.json").write_text(json.dumps(a,indent=2)+"\n")
 with(P/"check.log").open("x")as f:r=subprocess.run(a,stdout=f,stderr=subprocess.STDOUT)
 assert all(sha(p)==v for p,v in pins.items())
 print("checked",r.returncode)
