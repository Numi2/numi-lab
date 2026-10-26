"""Run the existing exact backtracker with pinned witness-scale repair and matched inputs."""
from pathlib import Path
import sys,json,hashlib,os,subprocess,time,argparse,math
import numpy as np
D=Path("/Users/numi/Work/numi-human-connected-surface-069")
PY="/Users/n/numi-human-prep-venv-20261005/bin/python"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
FROZEN_545=D/'D/contained-clearance-545/run_checked_batch_context545.py'
FROZEN_545_SHA256='ca1074c583144ac86c6b44e9c382ab7fc04ab32749eede794fb5fd31dc0d8f21'
def parse_cli(argv=None):
    ap=argparse.ArgumentParser(description="Checked 545 proposal backtracking with explicit internal construction trust")
    ap.add_argument("state");ap.add_argument("fit");ap.add_argument("context");ap.add_argument("out")
    ap.add_argument("--construction-trust-mm",type=float,default=1.0)
    cli=ap.parse_args(argv)
    if not math.isfinite(cli.construction_trust_mm) or cli.construction_trust_mm<=0:raise ValueError("construction trust must be finite and positive")
    return cli
def apply_construction_trust(cfg,cli):
    cfg=dict(cfg)
    cfg["construction_trust_mm"]=float(cli.construction_trust_mm)
    return cfg
def final_checker_argv(cli,checker,candidate,state,out):
    return [PY,str(checker),"--candidate",str(candidate),"--baseline-state",str(state),"--output-dir",str(out),"--trust-mm",format(float(cli.construction_trust_mm),".17g")]
def main():
    cli=parse_cli()
    assert sha(FROZEN_545)==FROZEN_545_SHA256, "frozen 545 context wrapper changed"
    state,fit,context_path,out=map(Path,(cli.state,cli.fit,cli.context,cli.out));out.mkdir(parents=True,exist_ok=False)
    W=Path(__file__).parent;context=json.loads(context_path.read_text());guard=Path(context["guard_npz"]).parent
    st=json.loads(state.read_text());base=Path(st["candidate"]);fz=fit/"candidate.npz";fr=fit/"report.json";report=json.loads(fr.read_text())
    if report["status"]!="CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED":raise ValueError("fit has no proposal")
    assert report["inputs"]["base_candidate_sha256"]==sha(base)
    assert report["native_admission"]is False and report["accepted_physical_steps"]==0
    a=np.load(base,allow_pickle=False);b=np.load(fz,allow_pickle=False);assert set(a.files)<=set(b.files)
    for k in a.files:
        if k=="points_world_m":continue
        if a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes():continue
        assert k.startswith(("local_witness_cluster_222_","local_witness_cluster_230_","local_witness_cluster_354_","local_witness_cluster_315_")),k
    X=a["points_world_m"].astype(float);Y=b["points_world_m"].astype(float);f=a["source_O_faces"];anchors=a["typed_coarse_anchor_output_rows_018"]
    assert np.isfinite(X).all()and np.isfinite(Y).all()and a["points_world_m"].dtype==b["points_world_m"].dtype==np.float32
    assert X[anchors].tobytes()==Y[anchors].tobytes()
    support=b["local_witness_cluster_222_support_rows"];changed=np.flatnonzero(np.any(X!=Y,axis=1));assert np.isin(changed,support).all()
    s=json.loads(Path(context["seed_report"]).read_text());assert s["candidate_sha256"]==sha(base)
    p={"self_seeds":{"path":s["output_pairs_path"],"sha256":s["output_pairs_sha256"],"report_sha256":sha(context["seed_report"]),"pair_count":s["merged_unique_pair_count"]}}
    cfg=apply_construction_trust(json.loads((D/"D/postfit-backtrack-427/run-001/backtrack-config.json").read_text()),cli)
    cfg.pop("initial_scales",None)
    cfg.pop("local_step_context_report",None)
    cfg.pop("local_step_context_report_sha256",None)
    cfg["helper"]=str(D/"D/graded-cut-step-447/graded_cut_step_447.py")
    cfg["helper_sha256"]="522152a4df248d230ddc66652e3071a2283c29ca73bccc52efbe1789e2ee2b9e"
    cfg["state"]={k:str(v)for k,v in dict(candidate=base,point_summary=Path(st["point_summary"]),surface_report=Path(st["surface_report"]),parent_fit=fr).items()}
    cfg.update(trial=str(fz),out=str(out/"backtrack-373"),parent_fit=str(fr),fit_candidate_sha256=sha(fz),fit_report_sha256=sha(fr),fit_support_rows=support.tolist(),
               area_reference_npz=str(base),area_reference_sha256=sha(base),area_reference_points_key="area_reference_037_points_world_m_f64",area_reference_faces_key="area_reference_037_source_O_faces",self_pair_seed_path=p["self_seeds"]["path"],self_pair_seed_sha256=p["self_seeds"]["sha256"],self_pair_seed_report_sha256=p["self_seeds"]["report_sha256"],self_pair_seed_count=p["self_seeds"]["pair_count"],initial_scale=1.)
    e=np.unique(np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
    delta=Y-X;amp=np.linalg.norm(delta,axis=1);den=np.maximum(amp[e[:,0]],amp[e[:,1]]);cost=np.ones(len(e));np.divide(.1*np.linalg.norm(X.astype(np.float32)[e[:,0]]-X.astype(np.float32)[e[:,1]],axis=1),den,out=cost,where=den>0);cost=np.minimum(1,cost);assert np.isfinite(cost).all()and np.all(cost>0)
    ctx=out/"context.npz";np.savez_compressed(ctx,faces=f,edges=e,edge_scale_budget=cost,points_baseline_m=X.astype(np.float32),proposal_displacement_m=delta,anchors=anchors)
    cfg.update(local_step_context_npz=str(ctx),local_step_context_sha256=sha(ctx),local_step_context_kappa=.1)
    cp=out/"backtrack-config.json";cp.write_text(json.dumps(cfg,indent=2)+"\n")
    runner=str(Path(__file__).with_name('run_local_witness_backtracking_546.py'))
    assert sha(runner)=='278e59c0e4c97cb428ac2ffbd3d9f39f8ec4c07c4077dad1d7d2f5fa6873962e'
    assert sha(cfg["helper"])==cfg["helper_sha256"]
    env=os.environ.copy();env.update(NUMI_GRADED_LOCAL_STEP_CONTEXT_423=str(ctx),NUMI_GRADED_LOCAL_STEP_CONTEXT_SHA256_423=sha(ctx),NUMI_GRADED_LOCAL_STEP_KAPPA_423=".1")
    seed_preflight=Path(context["seed_report"])
    env.update(NUMI_GRADED_SELF_SEED_PATH_431=cfg["self_pair_seed_path"],NUMI_GRADED_SELF_SEED_SHA256_431=cfg["self_pair_seed_sha256"],NUMI_GRADED_SELF_SEED_PREFLIGHT_PATH_431=str(seed_preflight),NUMI_GRADED_SELF_SEED_PREFLIGHT_SHA256_431=sha(seed_preflight),NUMI_GRADED_SELF_SEED_CACHE_431=str(out/"seed-rows.npz"),NUMI_GRADED_SELF_SEED_REPORT_431=str(out/"seed-scale-report.json"))
    guard_npz=guard/"fixed-skin-bone-guards.npz";guard_report=guard/"report.json"
    assert json.loads(guard_report.read_text())["base_candidate_sha256"]==sha(base)
    env.update(NUMI_GRADED_FIXED_GUARD_NPZ_PATH_447=str(guard_npz),NUMI_GRADED_FIXED_GUARD_NPZ_SHA256_447=sha(guard_npz),NUMI_GRADED_FIXED_GUARD_REPORT_PATH_447=str(guard_report),NUMI_GRADED_FIXED_GUARD_REPORT_SHA256_447=sha(guard_report),NUMI_GRADED_FIXED_GUARD_BASE_CANDIDATE_SHA256_447=sha(base))
    import importlib.util
    os.environ.update(env)
    spec=importlib.util.spec_from_file_location("project447",cfg["helper"]);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    context=helper.cuts431.cuts.g.load_context_423()
    seeds,packed,cache,seed_report,meta=helper.cuts431._load_seed_rows(f,context)
    key=json.dumps(meta,sort_keys=True,separators=(",",":"))
    guards,guard_receipt=helper.load_fixed_guard_rows(ctx,f,context)
    scales=np.ones(len(X));scales[anchors]=0
    scales,proof=helper.solve_with_fixed_guard_rows(helper.cuts431,f,scales,anchors,context["edges"],context["budget"],delta*1000.,[],packed,key,guards,guard_receipt)
    np.save(out/"initial-scales.npy",scales)
    (out/"preprojection.json").write_text(json.dumps(dict(status="CONSTRAINT_PROJECTION_WITHOUT_WITNESS_HALVING",seed_cut_lp=proof,native_admission=False,accepted_physical_steps=0),indent=2)+"\n")
    cfg.pop("initial_scale",None);cfg["initial_scales"]=str(out/"initial-scales.npy")
    cp.write_text(json.dumps(cfg,indent=2)+"\n")
    checker=D/'D/contained-clearance-545/run_candidate_checks545.py'
    checker_sha='0c6ce073c8ab615330f0ec993db44087c43d84d1283c4fb6458cf81a49d2195a'
    assert sha(checker)==checker_sha
    pins={str(x):sha(x)for x in [state,base,fz,fr,cp,ctx,runner,Path(runner).with_name("construction_trust_546.py"),Path(cfg["helper"]),seed_preflight,Path(cfg["self_pair_seed_path"]),guard_npz,guard_report,checker,FROZEN_545,Path(__file__)]}
    argv=[PY,str(runner),str(cp)]
    (out/"launch.json").write_text(json.dumps(dict(argv=argv,pins=pins,env={k:v for k,v in env.items()if k.startswith("NUMI_GRADED_")},base_sha256=sha(base),fit_sha256=sha(fz),typed_anchors_unchanged=len(anchors),changed_within_support=True,construction_trust_mm=float(cli.construction_trust_mm),required_final_checker_argv=final_checker_argv(cli,checker,"<selected candidate>",state,out/"final-check"),global_R0_source_cap_mm=12.0,native_admission=False,accepted_physical_steps=0),indent=2)+"\n")
    start=time.monotonic()
    with(out/"execution.log").open("x")as log:r=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    assert all(sha(k)==v for k,v in pins.items())
    result=dict(selected_result_exists=(out/"backtrack-373/selected-result.json").exists(),returncode=r.returncode,seconds=time.monotonic()-start,unchanged_runtime_pins=True,native_admission=False,accepted_physical_steps=0)
    (out/"execution.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
    if r.returncode != 0 or not result["selected_result_exists"]:
        raise SystemExit(r.returncode or 1)
    selected=json.loads((out/"backtrack-373/selected-result.json").read_text())
    checker_out=out/"final-check"
    checker_argv=final_checker_argv(cli,checker,selected["candidate"],state,checker_out)
    with (out/"final-check.log").open("x") as log:cr=subprocess.run(checker_argv,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    check_report=json.loads((checker_out/"report.json").read_text()) if (checker_out/"report.json").exists() else {}
    result.update(final_checker_argv=checker_argv,final_checker_returncode=cr.returncode,final_checker_status=check_report.get("status"),construction_trust_mm=float(cli.construction_trust_mm))
    (out/"execution.json").write_text(json.dumps(result,indent=2)+"\n")
    if cr.returncode!=0 or check_report.get("status")!="COMPLETE_EXACT_CONSTRUCTION_CHECKS":raise RuntimeError("545 final arithmetic checker failed")
    assert all(sha(k)==v for k,v in pins.items())
if __name__=="__main__":main()
