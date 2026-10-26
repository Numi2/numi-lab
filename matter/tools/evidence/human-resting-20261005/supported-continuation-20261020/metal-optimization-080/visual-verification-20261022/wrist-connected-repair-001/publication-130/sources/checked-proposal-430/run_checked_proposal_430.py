"""Run the existing exact backtracker with pinned witness-scale repair and matched inputs."""
from pathlib import Path
import sys,json,hashlib,os,subprocess,time
import numpy as np
D=Path("/Users/numi/Work/numi-human-connected-surface-069")
PY="/Users/n/numi-human-prep-venv-20261005/bin/python"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    state,fit,prep,out=map(Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=False)
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
    p=json.loads((prep/"preparation-report.json").read_text());assert p["candidate_sha256"]==sha(base)
    cfg=json.loads((D/"D/postfit-backtrack-427/run-001/backtrack-config.json").read_text())
    cfg.pop("initial_scales",None)
    cfg.pop("local_step_context_report",None)
    cfg.pop("local_step_context_report_sha256",None)
    cfg["state"]={k:str(v)for k,v in dict(candidate=base,point_summary=Path(st["point_summary"]),surface_report=Path(st["surface_report"]),parent_fit=fr).items()}
    cfg.update(trial=str(fz),out=str(out/"backtrack-373"),parent_fit=str(fr),fit_candidate_sha256=sha(fz),fit_report_sha256=sha(fr),fit_support_rows=support.tolist(),
               area_reference_npz=str(base),area_reference_sha256=sha(base),self_pair_seed_path=p["self_seeds"]["path"],self_pair_seed_sha256=p["self_seeds"]["sha256"],self_pair_seed_report_sha256=p["self_seeds"]["report_sha256"],self_pair_seed_count=p["self_seeds"]["pair_count"],initial_scale=1.)
    e=np.unique(np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
    delta=Y-X;amp=np.linalg.norm(delta,axis=1);den=np.maximum(amp[e[:,0]],amp[e[:,1]]);cost=np.ones(len(e));np.divide(.1*np.linalg.norm(X[e[:,0]]-X[e[:,1]],axis=1),den,out=cost,where=den>0);cost=np.minimum(1,cost);assert np.isfinite(cost).all()and np.all(cost>0)
    ctx=out/"context.npz";np.savez_compressed(ctx,faces=f,edges=e,edge_scale_budget=cost,points_baseline_m=X,proposal_displacement_m=delta,anchors=anchors)
    cfg.update(local_step_context_npz=str(ctx),local_step_context_sha256=sha(ctx),local_step_context_kappa=.1)
    cp=out/"backtrack-config.json";cp.write_text(json.dumps(cfg,indent=2)+"\n")
    runner=D/"barycentric-connected-clearance-170/no-progress-backtracker-373/run_local_witness_backtracking_373.py"
    assert sha(runner)=="a91b410c979701a29ab32f376637db566abb745e066b629c000f6a37fea13212"
    assert sha(cfg["helper"])==cfg["helper_sha256"]
    env=os.environ.copy();env.update(NUMI_GRADED_LOCAL_STEP_CONTEXT_423=str(ctx),NUMI_GRADED_LOCAL_STEP_CONTEXT_SHA256_423=sha(ctx),NUMI_GRADED_LOCAL_STEP_KAPPA_423=".1")
    pins={str(x):sha(x)for x in [state,base,fz,fr,cp,ctx,runner,Path(cfg["helper"]),Path(__file__)]}
    argv=[PY,str(runner),str(cp)]
    (out/"launch.json").write_text(json.dumps(dict(argv=argv,pins=pins,base_sha256=sha(base),fit_sha256=sha(fz),typed_anchors_unchanged=len(anchors),changed_within_support=True,native_admission=False,accepted_physical_steps=0),indent=2)+"\n")
    start=time.monotonic()
    with(out/"execution.log").open("x")as log:r=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
    assert all(sha(k)==v for k,v in pins.items())
    result=dict(returncode=r.returncode,seconds=time.monotonic()-start,unchanged_runtime_pins=True,native_admission=False,accepted_physical_steps=0)
    (out/"execution.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
if __name__=="__main__":main()
