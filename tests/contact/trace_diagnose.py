#!/usr/bin/env python3
"""Offline per-Newton diagnostics from bounded read-only state snapshots."""
import argparse,json
from pathlib import Path
import numpy as np
from diagnose import evaluate
p=argparse.ArgumentParser();p.add_argument("trace",type=Path);p.add_argument("--h",type=float,required=True);p.add_argument("--arm",choices=["ipc","al"],required=True);a=p.parse_args()
records={}
for line in a.trace.open():
    if '"state_arenas"' not in line:continue
    x=json.loads(line[line.index("{"):]);records.setdefault((x["iteration"],x["stage"]),x)
def arenas(x):return {k:bytes.fromhex(v) for k,v in x["state_arenas"].items()}
def array(x,k,width):return np.frombuffer(arenas(x)[k],"<f4").reshape(-1,width).astype(np.float64)
def data_for(x,correction):
    z=arenas(x)
    d={dst:z[src] for src,dst in [("accepted_nodes","fem_accepted_nodes"),("working_nodes","fem_candidate_nodes"),("tetrahedra","tetrahedra_accepted"),("contact_samples","contact_samples"),("rhs_workspace","full_rhs_before_fgmres")]}
    d["fem_solution_correction"]=correction
    d["center_production_jdelta_action"]=bytes(len(z["rhs_workspace"]))
    return d
rows=[]
for it in sorted({k[0] for k in records}):
    pre=records.get((it,"fully_assembled_rhs"));post=records.get((it,"before_multiplier_update"))
    if not pre or not post:continue
    corr=arenas(post)["correction"];d=data_for(pre,corr)
    oracle=evaluate(d,a.h,a.arm,capture=f"{a.trace.name}:{it}")
    before=records.get((it,"before_contact_limits"),pre);limited=records.get((it,"after_rigid_limit"),post)
    row={"iteration":it,"native_reassembled_rhs_l2_Ns":oracle["native_rhs_l2_Ns"],"fp64_reassembled_rhs_l2_Ns":oracle["fp64_rhs_l2_Ns"],
         "rhs_difference_l2_Ns":oracle["rhs_difference_l2_Ns"],"fp64_linear_residual_b_minus_Jdelta_Ns":oracle["fp64_linear_residual_l2_Ns"],
         "native_inner_residual_telemetry":post["fgmres"][0]["diagnostics"][0],"native_nonlinear_state":post["fgmres"][0]["nonlinear"],
         "correction_l2_mps":float(np.linalg.norm(np.frombuffer(corr,"<f4").reshape(-1,4)[:pre["fem_node_count"],:3])),
         "alpha_before_contact":before["alpha"],"alpha_after_rigid_clearance":limited["alpha"],"accepted_alpha":post["alpha"],
         "object_line_search":post["object_line_search"],"min_captured_contact_gap_m":None,"dual_stages":[]}
    sf=array(pre,"contact_samples",40);si=np.frombuffer(arenas(pre)["contact_samples"],"<u4").reshape(-1,40)
    gaps=[float(s[7]+(s[31] if ids[3]&512 else 0)) for s,ids in zip(sf,si) if ids[3]&1]
    if gaps:row["min_captured_contact_gap_m"]=min(gaps)
    for stage in ["before_multiplier_update","after_multiplier_update"]:
        rec=records.get((it,stage))
        if not rec:continue
        z=arenas(rec);accepted=array(rec,"accepted_dual",4)[:,3];candidate=array(rec,"candidate_dual",4)[:,3]
        dr={"stage":stage,"accepted_max_impulse_Ns":float(accepted.max(initial=0)),"candidate_max_impulse_Ns":float(candidate.max(initial=0))}
        if a.arm=="al":
            # Re-evaluate the same equations using this working configuration
            # and its current trial multipliers; the samples themselves have
            # not yet been rebuilt at these two boundaries.
            cur=array(rec,"working_nodes",24);samples=np.frombuffer(z["contact_samples"],"<f4").reshape(-1,40).copy()
            ids=np.frombuffer(z["contact_samples"],"<u4").reshape(-1,40)
            for j,(sample,identity) in enumerate(zip(samples,ids)):
                if identity[3]&1:
                    sample[7]=cur[int(identity[0]),2]
                    sample[27]=candidate[j]/float(np.float32(a.h))
            dd=data_for(rec,bytes(len(corr)));dd["contact_samples"]=samples.astype("<f4").tobytes()
            dr["fp64_recomputed_with_current_trial_multiplier_l2_Ns"]=evaluate(dd,a.h,a.arm)["fp64_rhs_l2_Ns"]
        row["dual_stages"].append(dr)
    rows.append(row)
print(json.dumps({"schema":"numi.fem-plane.newton-diagnostic.v1","trace":str(a.trace),"arm":a.arm,"h":a.h,"iterations":rows,
    "limits":["FP64 evaluates retained discrete equations and Jv only; no rollout or fallback.",
    "fully_assembled_rhs precedes Krylov scratch reuse; correction is from the same Newton iteration.",
    "Multiplier-boundary samples are stale by design; AL residuals there are explicitly recomputed from working nodes and trial duals.",
    "Only the first record per iteration/stage is used; deterministic replay duplicates are excluded."]},indent=2))
