from pathlib import Path
import numpy as np,json,hashlib,os,subprocess,time,importlib.util
D=Path("/Users/numi/Work/numi-human-connected-surface-069");O=D/"D/fixed-guard-backtrack-450/run-003";O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=json.loads((D/"D/grouped-backtrack-442/run-001/backtrack-config.json").read_text())
selected=json.loads((D/"D/grouped-backtrack-442/run-001/backtrack-373/selected-result.json").read_text())
base=Path(cfg["state"]["candidate"]);trial=Path(selected["candidate"])
a=np.load(base);b=np.load(trial);X=a["points_world_m"];Y=b["points_world_m"];F=a["source_O_faces"];anchors=a["typed_coarse_anchor_output_rows_018"]
assert sha(base)=="1429bbe58c4e27d7d483fa4583b747ec42cf679e04505e24f43c2ffbb5e091e1"
assert sha(trial)=="3b2f3ce58e4e0d7e1883f937f203af8bb0a5630cd502c5131cc7c4cc4b7e2b07"
e=np.unique(np.sort(np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1),axis=0)
delta=Y.astype(float)-X.astype(float);amp=np.linalg.norm(delta,axis=1);den=np.maximum(amp[e[:,0]],amp[e[:,1]])
cost=np.ones(len(e));np.divide(.1*np.linalg.norm(X[e[:,0]]-X[e[:,1]],axis=1),den,out=cost,where=den>0);cost=np.minimum(1,cost)
ctx=O/"context.npz";np.savez_compressed(ctx,faces=F,edges=e,edge_scale_budget=cost,points_baseline_m=X,proposal_displacement_m=delta,anchors=anchors)
helper=D/"D/graded-cut-step-447/graded_cut_step_447.py";assert sha(helper)=="522152a4df248d230ddc66652e3071a2283c29ca73bccc52efbe1789e2ee2b9e"
prep=D/"D/connected-continuation-prep-438/run-001";guard=D/"D/fixed-bone-guard-refresh-449/run-001";gn=guard/"introduced-skin-bone-barriers.npz";gr=guard/"report.json"
assert sha(gn)=="7e9ca02dfbe0b6f29bbf20b6f261cc60388c7ef476dcce6213f5f3d2e48e0ece" and sha(gr)=="ddbd34ea34428123e3b82c17412a5819b27f111c2f7d463f056daf68316da6be"
env=dict(NUMI_GRADED_LOCAL_STEP_CONTEXT_423=str(ctx),NUMI_GRADED_LOCAL_STEP_CONTEXT_SHA256_423=sha(ctx),NUMI_GRADED_LOCAL_STEP_KAPPA_423=".1",
NUMI_GRADED_SELF_SEED_PATH_431=cfg["self_pair_seed_path"],NUMI_GRADED_SELF_SEED_SHA256_431=cfg["self_pair_seed_sha256"],
NUMI_GRADED_SELF_SEED_PREFLIGHT_PATH_431=str(prep/"self-seeds/report.json"),NUMI_GRADED_SELF_SEED_PREFLIGHT_SHA256_431=sha(prep/"self-seeds/report.json"),
NUMI_GRADED_SELF_SEED_CACHE_431=str(O/"seed-rows.npz"),NUMI_GRADED_SELF_SEED_REPORT_431=str(O/"seed-scale-report.json"),
NUMI_GRADED_FIXED_GUARD_NPZ_PATH_447=str(gn),NUMI_GRADED_FIXED_GUARD_NPZ_SHA256_447=sha(gn),NUMI_GRADED_FIXED_GUARD_REPORT_PATH_447=str(gr),
NUMI_GRADED_FIXED_GUARD_REPORT_SHA256_447=sha(gr),NUMI_GRADED_FIXED_GUARD_BASE_CANDIDATE_SHA256_447=sha(base))
os.environ.update(env)
sp=importlib.util.spec_from_file_location("guard447",helper);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
bad=np.unique([x["skin_face_row"]for x in json.loads(gr.read_text())["resolved_pairs_previous_to_current"]])
scales=np.ones(len(X));scales[anchors]=0;t=time.monotonic()
scales,report=m.reduce_witness_node_scales(F,scales,bad,np.empty((0,2),np.int64),anchors=anchors,zero_below=cfg["zero_below"])
np.save(O/"initial-scales.npy",scales);(O/"preprojection.json").write_text(json.dumps(report,indent=2)+"\n")
cfg.pop("initial_scale",None);cfg.update(trial=str(trial),fit_candidate_sha256=sha(trial),original_fit_candidate_sha256=cfg["fit_candidate_sha256"],derived_trial_selected_result=str(D/"D/grouped-backtrack-442/run-001/backtrack-373/selected-result.json"),helper=str(helper),helper_sha256=sha(helper),out=str(O/"backtrack-373"),initial_scales=str(O/"initial-scales.npy"),local_step_context_npz=str(ctx),local_step_context_sha256=sha(ctx))
cp=O/"backtrack-config.json";cp.write_text(json.dumps(cfg,indent=2)+"\n")
runner=D/"barycentric-connected-clearance-170/no-progress-backtracker-373/run_local_witness_backtracking_373.py"
pins={str(p):sha(p)for p in [base,trial,helper,ctx,gr,gn,cp,Path(__file__)]};(O/"launch.json").write_text(json.dumps(dict(pins=pins,env=env,exact_failure_face_count=len(bad),native_admission=False,accepted_physical_steps=0),indent=2)+"\n")
with(O/"execution.log").open("x")as log:r=subprocess.run(["/Users/n/numi-human-prep-venv-20261005/bin/python",str(runner),str(cp)],stdout=log,stderr=subprocess.STDOUT,timeout=1800)
assert all(sha(k)==v for k,v in pins.items())
result=dict(returncode=r.returncode,seconds=time.monotonic()-t,selected_result_exists=(O/"backtrack-373/selected-result.json").exists(),runtime_pins_unchanged=True)
(O/"execution.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
