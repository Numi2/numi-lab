from pathlib import Path
import numpy as np,json,hashlib,os,subprocess,time,importlib.util
D=Path("/Users/numi/Work/numi-human-connected-surface-069");O=D/"D/constraint-project-backtrack-456/run-001";O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=json.loads((D/"D/point-first-backtrack-446/run-001/backtrack-config.json").read_text())
selected=json.loads((D/"D/point-first-backtrack-446/run-001/backtrack-373/selected-result.json").read_text())
base_state=json.loads((D/"D/fixed-guard-backtrack-450/run-003/backtrack-373/selected-result.json").read_text());cfg["state"]={**cfg["state"],**{k:base_state[k] for k in ("candidate","point_summary","surface_report")}};base=Path(cfg["state"]["candidate"]);trial=Path(selected["candidate"])
a=np.load(base);b=np.load(trial);X=a["points_world_m"];Y=b["points_world_m"];F=a["source_O_faces"];anchors=a["typed_coarse_anchor_output_rows_018"]
assert sha(base)=="32daa86b90f18f19f20790c4bf64e5b772c9ee0b65646df48a724ffa5d70c5e0"
assert sha(trial)=="1f6fddce0536d32da528565ab02b114f5198490426991d57b0ff85aface8d3de"
e=np.unique(np.sort(np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1),axis=0)
delta=Y.astype(float)-X.astype(float);amp=np.linalg.norm(delta,axis=1);den=np.maximum(amp[e[:,0]],amp[e[:,1]])
cost=np.ones(len(e));np.divide(.1*np.linalg.norm(X[e[:,0]]-X[e[:,1]],axis=1),den,out=cost,where=den>0);cost=np.minimum(1,cost)
ctx=O/"context.npz";np.savez_compressed(ctx,faces=F,edges=e,edge_scale_budget=cost,points_baseline_m=X,proposal_displacement_m=delta,anchors=anchors)
helper=D/"D/graded-cut-step-447/graded_cut_step_447.py";assert sha(helper)=="522152a4df248d230ddc66652e3071a2283c29ca73bccc52efbe1789e2ee2b9e"
prep=D/"D/connected-continuation-prep-454/run-001";guard=D/"D/fixed-bone-guard-refresh-454/run-001";gn=guard/"introduced-skin-bone-barriers.npz";gr=guard/"report.json"
assert json.loads(gr.read_text())["base_candidate_sha256"]==sha(base)
pj=json.loads((prep/"preparation-report.json").read_text());cfg.update(self_pair_seed_path=pj["self_seeds"]["path"],self_pair_seed_sha256=pj["self_seeds"]["sha256"],self_pair_seed_report_sha256=pj["self_seeds"]["report_sha256"],self_pair_seed_count=pj["self_seeds"]["pair_count"],area_reference_npz=str(base),area_reference_sha256=sha(base))
cfg["fit_support_rows"]=sorted(set(cfg["fit_support_rows"])|set(json.loads((D/"D/grouped-backtrack-442/run-001/backtrack-config.json").read_text())["fit_support_rows"]))
env=dict(NUMI_GRADED_LOCAL_STEP_CONTEXT_423=str(ctx),NUMI_GRADED_LOCAL_STEP_CONTEXT_SHA256_423=sha(ctx),NUMI_GRADED_LOCAL_STEP_KAPPA_423=".1",
NUMI_GRADED_SELF_SEED_PATH_431=cfg["self_pair_seed_path"],NUMI_GRADED_SELF_SEED_SHA256_431=cfg["self_pair_seed_sha256"],
NUMI_GRADED_SELF_SEED_PREFLIGHT_PATH_431=str(prep/"self-seeds/report.json"),NUMI_GRADED_SELF_SEED_PREFLIGHT_SHA256_431=sha(prep/"self-seeds/report.json"),
NUMI_GRADED_SELF_SEED_CACHE_431=str(O/"seed-rows.npz"),NUMI_GRADED_SELF_SEED_REPORT_431=str(O/"seed-scale-report.json"),
NUMI_GRADED_FIXED_GUARD_NPZ_PATH_447=str(gn),NUMI_GRADED_FIXED_GUARD_NPZ_SHA256_447=sha(gn),NUMI_GRADED_FIXED_GUARD_REPORT_PATH_447=str(gr),
NUMI_GRADED_FIXED_GUARD_REPORT_SHA256_447=sha(gr),NUMI_GRADED_FIXED_GUARD_BASE_CANDIDATE_SHA256_447=sha(base))
os.environ.update(env)
sp=importlib.util.spec_from_file_location("guard447",helper);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
bad=np.unique([x["skin_face_row"]for x in json.loads(gr.read_text())["resolved_pairs_previous_to_current"]])
scales=np.minimum(1.,np.divide(.001,amp,out=np.ones(len(X)),where=amp>0));scales[anchors]=0;t=time.monotonic()
context=m.cuts431.cuts.g.load_context_423()
seeds,packed,cache_path,report_path,meta=m.cuts431._load_seed_rows(F,context)
key=json.dumps(meta,sort_keys=True,separators=(",",":"))
guards,guard_receipt=m.load_fixed_guard_rows(ctx,F,context)
scales,proof=m.solve_with_fixed_guard_rows(m.cuts431,F,scales,anchors,context["edges"],context["budget"],delta*1000.,[],packed,key,guards,guard_receipt)
report=dict(status="CONSTRAINT_PROJECTION_WITHOUT_WITNESS_HALVING",seed_cut_lp=proof,seed_pair_count=len(seeds),seed_rows_total=len(packed["lower"]),explicit_failure_face_rows=bad.tolist(),native_admission=False,accepted_physical_steps=0)
np.save(O/"initial-scales.npy",scales);(O/"preprojection.json").write_text(json.dumps(report,indent=2)+"\n")
cfg.pop("initial_scale",None);cfg.update(trial=str(trial),fit_candidate_sha256=sha(trial),original_fit_candidate_sha256=cfg["fit_candidate_sha256"],derived_trial_selected_result=str(D/"D/point-first-backtrack-446/run-001/backtrack-373/selected-result.json"),helper=str(helper),helper_sha256=sha(helper),out=str(O/"backtrack-373"),initial_scales=str(O/"initial-scales.npy"),local_step_context_npz=str(ctx),local_step_context_sha256=sha(ctx))
cp=O/"backtrack-config.json";cp.write_text(json.dumps(cfg,indent=2)+"\n")
runner=D/"barycentric-connected-clearance-170/no-progress-backtracker-373/run_local_witness_backtracking_373.py"
pins={str(p):sha(p)for p in [base,trial,helper,ctx,gr,gn,cp,Path(__file__)]};(O/"launch.json").write_text(json.dumps(dict(pins=pins,env=env,exact_failure_face_count=len(bad),native_admission=False,accepted_physical_steps=0),indent=2)+"\n")
with(O/"execution.log").open("x")as log:r=subprocess.run(["/Users/n/numi-human-prep-venv-20261005/bin/python",str(runner),str(cp)],stdout=log,stderr=subprocess.STDOUT,timeout=1800)
assert all(sha(k)==v for k,v in pins.items())
result=dict(returncode=r.returncode,seconds=time.monotonic()-t,selected_result_exists=(O/"backtrack-373/selected-result.json").exists(),runtime_pins_unchanged=True)
(O/"execution.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
