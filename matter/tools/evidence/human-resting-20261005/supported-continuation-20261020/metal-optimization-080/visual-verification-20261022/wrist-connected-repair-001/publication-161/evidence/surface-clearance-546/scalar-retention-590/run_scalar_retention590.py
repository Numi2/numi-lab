#!/usr/bin/env python3
"""Scalar-alpha LP warm start followed by the unchanged 582 actual gates."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
HERE=Path(__file__).resolve().parent
CORE=ROOT/"D/surface-clearance-546/local-witness-retention-582/retain_local_witnesses582.py"
CORE_SHA="08bb82034180636d0ff798ad503735b6a50327bf52cc3c62bffb75d0265ff57a"
OWNER=ROOT/"D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
OWNER_SHA="5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2"
R0=ROOT/"full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz"
R0_SHA="70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31"
HELPER=HERE/"scalar_alpha_retention590.py"


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and np.ascontiguousarray(a).tobytes()==np.ascontiguousarray(b).tobytes()
def loadnpz(p):
    with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(str(path))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state",type=Path,required=True)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--capture-report",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--max-edge-gradient",type=float,default=.5)
    p.add_argument("--trust-mm",type=float,default=4.)
    p.add_argument("--max-lp-calls",type=int,default=8)
    p.add_argument("--lp-time-limit-s",type=float,default=5.)
    args=p.parse_args(argv)
    started=time.monotonic(); out=args.out.resolve()
    if out.exists(): raise FileExistsError(out)
    if not (0 < args.max_edge_gradient <= .5): raise ValueError("edge-gradient bound must be positive and no weaker than the existing 0.5 gate")
    if not (0 < args.trust_mm <= 4.): raise ValueError("trust bound must be positive and no weaker than the existing 4 mm gate")
    for path,want,name in ((CORE,CORE_SHA,"582"),(OWNER,OWNER_SHA,"408"),(R0,R0_SHA,"R0")):
        if sha(path)!=want: raise ValueError(f"pinned {name} source/input changed")
    state=json.loads(args.state.read_text()); base_path=Path(state["candidate"])
    pins={str(args.state.resolve()):sha(args.state),str(base_path.resolve()):sha(base_path),
          str(args.target.resolve()):sha(args.target),str(args.capture_report.resolve()):sha(args.capture_report),
          str(CORE):CORE_SHA,str(OWNER):OWNER_SHA,str(R0):R0_SHA,str(HELPER):sha(HELPER),str(Path(__file__).resolve()):sha(__file__)}
    if state.get("candidate_sha256")!=pins[str(base_path.resolve())]: raise ValueError("base candidate does not match state pin")
    if state.get("point_summary_sha256")!=sha(state["point_summary"]) or state.get("surface_report_sha256")!=sha(state["surface_report"]): raise ValueError("baseline exact evidence is stale")
    capture=json.loads(args.capture_report.read_text())
    target_sha=pins[str(args.target.resolve())]
    if (capture.get("candidate_path")!=str(args.target.resolve()) or capture.get("candidate_sha256")!=target_sha
        or capture.get("base_candidate")!=str(base_path) or capture.get("base_sha256")!=state["candidate_sha256"]
        or capture.get("candidate_arrays_copied_from_base_except_points_world_m") is not True
        or capture.get("accepted_physical_steps")!=0):
        raise ValueError("raw target capture is not matched to the pinned baseline or construction-only")
    base=loadnpz(base_path); target=loadnpz(args.target)
    if set(base)!=set(target): raise ValueError("base and target NPZ schemas differ")
    for key in base:
        if key!="points_world_m" and not same(base[key],target[key]):
            raise ValueError(f"target changed noncoordinate payload: {key}")
    x=np.asarray(base["points_world_m"]); xt=np.asarray(target["points_world_m"]); faces=np.asarray(base["source_O_faces"])
    if x.dtype!=np.float32 or xt.dtype!=np.float32 or x.shape!=xt.shape or not np.isfinite(x).all() or not np.isfinite(xt).all():
        raise ValueError("base/target points must be aligned finite Float32")
    if not np.array_equal(faces,target["source_O_faces"]): raise ValueError("base/target face topology differs")
    anchors=np.asarray(base["typed_coarse_anchor_output_rows_018"],dtype=np.int64)
    if not same(x[anchors],xt[anchors]): raise ValueError("saved target moves typed anchors")
    free=np.flatnonzero(np.any(x!=xt,axis=1)).astype(np.int64)
    if not len(free): raise ValueError("saved target has no mutable nodes")
    delta_mm=(xt[free].astype(np.float64)-x[free].astype(np.float64))*1000.
    base_mm=x.astype(np.float64)*1000.
    owner=load_module("owner408_for590",OWNER)
    with np.load(base_path,allow_pickle=False) as archive: source,source_levels=owner.compose_source_reference(archive,R0)
    source_mm=np.asarray(source,dtype=np.float64)*1000.
    if source_mm.shape!=base_mm.shape or not np.isfinite(source_mm).all(): raise ValueError("composed source-reference domain invalid")
    E,rhs,edge_report=owner.build_edge_gradient_constraints(base_mm,faces,free,args.max_edge_gradient)
    helper=load_module("scalar_alpha_math590",HELPER)
    alpha,lp_report=helper.optimize_scalar_alpha590(
        E,rhs,base_mm,source_mm,base["area_reference_037_points_world_m_f64"]*1000.,
        faces,free,delta_mm,trust_mm=args.trust_mm,source_cap_mm=12.,
        edge_gradient=args.max_edge_gradient,max_lp_calls=args.max_lp_calls,
        lp_time_limit_s=args.lp_time_limit_s)
    out.mkdir(parents=True,exist_ok=False)
    proposal=x.copy()
    proposal[free]=(x[free].astype(np.float64)+alpha[:,None]*(xt[free].astype(np.float64)-x[free].astype(np.float64))).astype(np.float32)
    if same(proposal,x):
        report={"status":"SCALAR_LP_NO_REPRESENTABLE_MOVE","base_candidate_sha256":pins[str(base_path.resolve())],
                "raw_target_sha256":target_sha,"lp":lp_report,"source_pins":pins,
                "native_admission":False,"accepted_physical_steps":0,"seconds":time.monotonic()-started}
        (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
        print(json.dumps({"status":report["status"],"lp_calls":lp_report["lp_calls"]}));return
    scalar_target={k:v.copy() for k,v in base.items()};scalar_target["points_world_m"]=proposal
    target_out=out/"scalar-target.npz"
    with target_out.open("xb") as stream: np.savez_compressed(stream,**scalar_target)
    np.save(out/"alpha.npy",alpha,allow_pickle=False)
    if not same(loadnpz(target_out)["points_world_m"],proposal): raise AssertionError("Float32 scalar target emission mismatch")
    core=load_module("frozen582_for590",CORE)
    core_out=out/"582-witness-retention"
    old_argv=sys.argv
    sys.argv=[str(CORE),"--state",str(args.state),"--target",str(target_out),"--out",str(core_out),
              "--max-edge-gradient",format(args.max_edge_gradient,".17g"),"--trust-mm",format(args.trust_mm,".17g")]
    core_error=None
    try:
        core.main()
    except (Exception,SystemExit) as exc:
        core_error=f"{type(exc).__name__}: {exc}"
    finally:
        sys.argv=old_argv
    core_report=core_out/"report.json"
    final={"status":"SCALAR_ALPHA_582_COMPLETE" if core_error is None else "SCALAR_ALPHA_582_FAILED",
           "scope":"scalar-alpha LP proposal followed by unchanged frozen582 actual Float32 area/source/trust/self checks and witness-halving fallback",
           "base_state":str(args.state.resolve()),"base_state_sha256":pins[str(args.state.resolve())],
           "base_candidate":str(base_path.resolve()),"base_candidate_sha256":pins[str(base_path.resolve())],
           "raw_target":str(args.target.resolve()),"raw_target_sha256":target_sha,
           "capture_report":str(args.capture_report.resolve()),"capture_report_sha256":pins[str(args.capture_report.resolve())],
           "scalar_target":str(target_out.resolve()),"scalar_target_sha256":sha(target_out),
           "alpha_path":str((out/"alpha.npy").resolve()),"alpha_sha256":sha(out/"alpha.npy"),
           "mutable_node_count":int(len(free)),"typed_anchor_count":int(len(anchors)),
           "edge_constraint_receipt":edge_report,"source_reference_levels":source_levels,
           "lp":lp_report,"core_582_report":str(core_report.resolve()) if core_report.exists() else None,
           "core_582_report_sha256":sha(core_report) if core_report.exists() else None,
           "core_582_error":core_error,"source_pins":pins,
           "final_545_130_159_required":True,"native_admission":False,"accepted_physical_steps":0,
           "seconds":time.monotonic()-started}
    for path,want in pins.items():
        if sha(path)!=want: raise ValueError("input/source changed during scalar retention: "+path)
    (out/"report.json").write_text(json.dumps(final,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({k:final[k] for k in ("status","mutable_node_count","typed_anchor_count","core_582_error","seconds")}))
    if core_error is not None: raise SystemExit(1)

if __name__=="__main__": main()
