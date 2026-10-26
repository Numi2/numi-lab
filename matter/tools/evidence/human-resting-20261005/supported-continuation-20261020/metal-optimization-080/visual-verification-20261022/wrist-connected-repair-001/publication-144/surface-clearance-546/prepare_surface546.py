from pathlib import Path
import importlib.util,argparse,json,numpy as np
D=Path("/Users/numi/Work/numi-human-connected-surface-069");W=D/"D/contained-clearance-545";N=D/"D/surface-clearance-546";N.mkdir(exist_ok=True)
s=importlib.util.spec_from_file_location("cont545",W/"continue545.py");m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--state",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);ap.add_argument("--prior-context",type=Path,required=True);ap.add_argument("--seed-source-candidate",type=Path);ap.add_argument("--seed-source-audit130",type=Path);a=ap.parse_args();O=a.out;O.mkdir(parents=True,exist_ok=False)
 st=m.read(a.state);base=Path(st["candidate"]);q=Path(st["point_summary"]).parent/"query";prior=m.read(a.prior_context)
 for k in ("candidate","point_summary","surface_report"):assert m.sha(st[k])==st[k+"_sha256"]
 r=m.read(st["point_summary"]);assert r["exact_PLC_surface_pairs"]==0 and r["inside_node_bank_pairs"]==0 and r["complete_surface_audit"]
 m.write(O/"baseline.json",st)
 trial=a.seed_source_candidate or W/"fit-001/candidate.npz";ta=a.seed_source_audit130 or W/"check-001-diagnostic/audit-130/summary.json"
 args=[m.PY,D/"D/focused-clearance-541/constraints/refresh_guards_541.py","--candidate",base,"--audit130",st["point_summary"],"--audit159",st["surface_report"],"--query-dir",q,"--ancestry-parent-candidate",base,"--seed-source-candidate",trial,"--seed-source-audit130",ta,"--seed-source-pairs",ta.parent/"surface-self-pairs.npy","--prior-guard-candidate",prior["candidate"],"--prior-guard-report",prior["guard_report"],"--prior-guard-npz",prior["guard_npz"],"--output-dir",O/"constraints","--expected-candidate-sha256",m.sha(base),"--expected-audit130-sha256",m.sha(st["point_summary"]),"--expected-audit159-sha256",m.sha(st["surface_report"])]
 m.run(args,O/"refresh",True);rr=m.read(O/"constraints/refresh-report.json")
 z=np.load(base,allow_pickle=False);pz=np.load(prior["candidate"],allow_pickle=False);assert np.array_equal(z["source_O_faces"],pz["source_O_faces"])
 pairs=np.unique(np.vstack([np.load(prior["seed_pairs_path"]),np.load(rr["seed_pairs_path"])]),axis=0);seed=O/"constraints/merged-self-pair-seeds.npy";np.save(seed,pairs)
 owner=D/"publication-101/evidence/204/source/fit/self_separation_026.py";assert m.sha(owner)=="c67daf7c9fd2292e32ae88dba215a74350e1cbf83c669bc299a64b86d12e99f3"
 rows,_=m.loadmod(owner).build_pair_rows(z["points_world_m"].astype(float)*1000,z["source_O_faces"],pairs,margin_fraction=.25);rhs=max((float(x["lower"])for x in rows),default=0);assert rhs<=2e-8
 sr=O/"constraints/seed-preflight.json";m.write(sr,dict(schema="numi.human.continuation-self-seeds.v1",status="PASS_REBUILT_CONSTRUCTION_SEEDS_ONLY",candidate_sha256=m.sha(base),zero_step_feasible=True,zero_step_max_rhs_mm=rhs,rebuilt_linear_row_count=len(rows),output_pairs_path=str(seed),output_pairs_sha256=m.sha(seed),merged_unique_pair_count=len(pairs),pair_owner_sha256=m.sha(owner),prior_seed_path=prior["seed_pairs_path"],prior_seed_sha256=m.sha(prior["seed_pairs_path"]),new_trial_seed_path=rr["seed_pairs_path"],new_trial_seed_sha256=m.sha(rr["seed_pairs_path"]),native_admission=False,accepted_physical_steps=0))
 m.write(O/"constraint-context.json",dict(candidate=str(base),seed_pairs_path=str(seed),seed_report=str(sr),guard_npz=rr["fixed_guard_npz_path"],guard_report=rr["fixed_guard_report_path"]))
 args=m.read(W/"depth-argv.json");changes={"--candidate":base,"--point-summary":st["point_summary"],"--surface-report":st["surface_report"],"--query-dir":q,"--output-dir":O/"depth","--expected-candidate-sha256":m.sha(base),"--expected-point-summary-sha256":m.sha(st["point_summary"]),"--expected-surface-report-sha256":m.sha(st["surface_report"]),"--expected-query-receipt-sha256":m.sha(q/"query-receipt.json")}
 for k,v in changes.items():args[args.index(k)+1]=str(v)
 m.run(args,O/"depth",True)
 m.run([m.PY,D/"D/source-outward-proposal-500/build_outward_nearest_union_targets_500.py","--candidate",base,"--surface-report",st["surface_report"],"--query-receipt",q/"query-receipt.json","--output-dir",O/"surface-targets","--clearance-m",".00005","--direction-mode","outward-nearest"],O/"surfaces",True)
 m.write(O/"prepared.json",dict(status="FRESH_CONSTRAINTS_DEPTH_SURFACE_TARGETS_PREPARED",state=str(a.state),context=str(O/"constraint-context.json"),native_admission=False,accepted_physical_steps=0))
 print(json.dumps(m.read(O/"prepared.json")),flush=True)
if __name__=="__main__":main()
