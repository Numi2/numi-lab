#!/usr/bin/env python3
"""Elastic construction-only Phase-I diagnostic for 396's appended F32 rows."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


class CaptureComplete(Exception):
    pass


def _csr(a):
    return sparse.csr_matrix(a,dtype=np.float64)


def _csr_sha(a):
    a=_csr(a)
    return hashlib.sha256(a.indptr.tobytes()+a.indices.tobytes()+a.data.tobytes()).hexdigest()


def build_elastic_problem(A,b,lo,hi,M,phase_start,phase_stop,source_indices,pair_keys):
    """Relax only the declared duplicate precision rows; every other row/bound is hard."""
    A=_csr(A); M=_csr(M)
    b=np.asarray(b,dtype=np.float64).reshape(-1)
    lo=np.asarray(lo,dtype=np.float64).reshape(-1); hi=np.asarray(hi,dtype=np.float64).reshape(-1)
    source_indices=np.asarray(source_indices,dtype=np.int64).reshape(-1)
    nrow,nout=A.shape
    if len(b)!=nrow or len(lo)!=nout or len(hi)!=nout or M.shape[0]!=nout or M.shape[1]==0:
        raise ValueError("A/b/bounds/M dimensions mismatch")
    if not all(np.isfinite(x).all() for x in (A.data,b,lo,hi,M.data)) or np.any(lo>hi):
        raise ValueError("nonfinite matrix or inverted output bounds")
    start,stop=int(phase_start),int(phase_stop)
    if start<0 or stop<=start or stop>nrow or stop-start!=len(source_indices) or len(pair_keys)!=len(source_indices):
        raise ValueError("precision row range/source identity mismatch")
    if np.any(source_indices<0) or np.any(source_indices>=start):
        raise ValueError("precision source rows must be earlier hard rows")
    soft=np.arange(start,stop,dtype=np.int64)
    original=A[source_indices]
    duplicate=A[soft]
    difference=(duplicate-original).tocsr(); difference.eliminate_zeros()
    if difference.nnz:
        raise ValueError("precision rows do not duplicate their original hard-row coefficients")
    extra=b[soft]-b[source_indices]
    if not np.isfinite(extra).all() or np.any(extra<=0):
        raise ValueError("precision duplicate RHS must strengthen, not replace, each original row")
    groups={}
    for key in pair_keys:
        groups[key]=groups.get(key,0)+1
    weights=np.asarray([1.0/(groups[key]*float(ex)) for key,ex in zip(pair_keys,extra)],dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights<=0):
        raise ValueError("invalid normalized pair-balanced slack weights")

    AM=(A@M).tocsr()
    hard_mask=np.ones(nrow,dtype=bool); hard_mask[soft]=False
    hard_A=AM[hard_mask]
    hard_b=b[hard_mask]
    bound_A=sparse.vstack((M,-M),format="csr")
    bound_b=np.concatenate((lo,-hi))
    hbase=sparse.vstack((hard_A,bound_A),format="csr")
    rbase=np.concatenate((hard_b,bound_b))
    phase_A=AM[soft]
    ncontrol=M.shape[1]; nsoft=len(soft)
    hard_aug=sparse.hstack((hbase,sparse.csr_matrix((hbase.shape[0],nsoft))),format="csr")
    phase_aug=sparse.hstack((phase_A,sparse.eye(nsoft,format="csr")),format="csr")
    aub=-sparse.vstack((hard_aug,phase_aug),format="csr")
    bub=-np.concatenate((rbase,b[soft]))
    objective=np.concatenate((np.zeros(ncontrol,dtype=np.float64),weights))
    bounds=[(None,None)]*ncontrol+[(0.0,None)]*nsoft
    return {
        "A_ub":aub,"b_ub":bub,"objective":objective,"bounds":bounds,
        "A":A,"b":b,"lo":lo,"hi":hi,"M":M,"AM":AM,
        "hard_mask":hard_mask,"soft_rows":soft,"source_indices":source_indices,
        "extra_rhs":extra,"pair_keys":list(pair_keys),"weights":weights,
        "hard_constraint_count":int(hbase.shape[0]),"ncontrol":int(ncontrol),
    }


def solve_elastic_phase_one(problem,*,time_limit_s=60.0,method="highs-ipm"):
    t=time.monotonic()
    result=linprog(problem["objective"],A_ub=problem["A_ub"],b_ub=problem["b_ub"],
                   bounds=problem["bounds"],method=method,
                   options={"time_limit":float(time_limit_s),"presolve":True})
    elapsed=time.monotonic()-t
    report={"status":int(result.status),"success":bool(result.success),"message":str(result.message),
            "iterations":int(getattr(result,"nit",0)),"wall_seconds":elapsed,
            "method":method,"time_limit_s":float(time_limit_s),
            "objective_normalized_pair_mean_fractional_shortfall":None,
            "minimum_original_hard_residual":None,"minimum_precision_relaxed_residual":None,
            "maximum_output_bound_violation":None,"positive_slack_row_count":None}
    arrays={}
    if result.success:
        ncontrol=problem["ncontrol"]; d=np.asarray(problem["M"]@result.x[:ncontrol]).reshape(-1)
        slack=np.asarray(result.x[ncontrol:],dtype=np.float64)
        A=problem["A"]; b=problem["b"]; soft=problem["soft_rows"]
        hard_res=np.asarray(A[problem["hard_mask"]]@d).reshape(-1)-b[problem["hard_mask"]]
        phase_res=np.asarray(A[soft]@d).reshape(-1)+slack-b[soft]
        bound_error=max(float(np.max(np.maximum(problem["lo"]-d,0),initial=0)),
                        float(np.max(np.maximum(d-problem["hi"],0),initial=0)))
        if (not np.isfinite(d).all() or not np.isfinite(slack).all()
                or np.min(hard_res,initial=np.inf)<-2e-7
                or np.min(phase_res,initial=np.inf)<-2e-7
                or np.min(slack,initial=0)<0 or bound_error>2e-7):
            raise RuntimeError("elastic primal failed hard-row, relaxed-row, or output-bound certificate")
        extras=problem["extra_rhs"]
        fractions=slack/extras
        report.update({
            "objective_normalized_pair_mean_fractional_shortfall":float(np.dot(problem["weights"],slack)),
            "minimum_original_hard_residual":float(np.min(hard_res,initial=np.inf)),
            "minimum_precision_relaxed_residual":float(np.min(phase_res,initial=np.inf)),
            "maximum_output_bound_violation":bound_error,
            "positive_slack_row_count":int(np.count_nonzero(slack>1e-10)),
            "maximum_fractional_shortfall":float(np.max(fractions,initial=0)),
            "fractional_shortfall_median":float(np.median(fractions)),
        })
        marginals=np.asarray(result.ineqlin.marginals,dtype=np.float64)
        soft_dual=np.abs(marginals[problem["hard_constraint_count"]:])
        arrays={"basis_control_q_diagnostic_only":np.asarray(result.x[:ncontrol],dtype=np.float64),
                "precision_slack_mm":slack,"precision_extra_rhs_mm":extras,
                "precision_fractional_shortfall":fractions,"precision_dual_abs":soft_dual,
                "precision_duplicate_A_row_indices":soft,
                "precision_original_A_row_indices":problem["source_indices"]}
        report["objective_semantics"]="sum over pairs of mean(slack_i/added_rhs_i); only appended precision duplicates are softened"
        report["candidate_emitted"]=False
        report["native_admission"]=False
    return report,arrays,result


def run_capture(argv_json:Path,output_dir:Path,*,time_limit_s=60.0,method="highs-ipm"):
    argv=json.loads(argv_json.read_text()); script=Path(argv[1])
    sys.path.insert(0,str(script.parent))
    spec=importlib.util.spec_from_file_location("driver396_elastic_capture",script)
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    captured={}
    def capture(A,b,target,lower,upper,C,h,weights,M,*args,**kwargs):
        if _csr(C).shape[0]!=0:
            raise ValueError("expected 396 Phase-I assembly with soft clearance rows omitted")
        local=inspect.currentframe().f_back.f_locals
        receipt=local.get("precision_phase_one_receipt")
        if not isinstance(receipt,dict): raise ValueError("Phase-I receipt missing at solver boundary")
        source_indices=np.asarray(local.get("precision_phase_one_source_indices"),dtype=np.int64)
        report_rows=json.loads(Path(local["args"].precision_phase_one_report).read_text())["rows"]
        pair_keys=[]
        pair_findings=[]
        for source_i in source_indices:
            row=report_rows[int(source_i)]
            faces=tuple(sorted(map(int,row["pair_face_rows"])))
            kind=str(row["pair_kind"])
            key=(kind,faces)
            pair_keys.append(key)
            pair_findings.append({"pair_kind":kind,"face_rows":list(faces),
                                 "pair_row_label":str(local["self_pair_rows"][int(source_i)]["label"])})
        problem=build_elastic_problem(A,b,lower,upper,M,
                                      receipt["duplicate_self_row_start"],
                                      receipt["duplicate_self_row_stop"],
                                      source_indices,pair_keys)
        report,arrays,result=solve_elastic_phase_one(problem,time_limit_s=time_limit_s,method=method)
        controls=local.get("basis_report",{}).get("control_parent_rows",[])
        if len(controls)*3!=problem["ncontrol"]:
            raise ValueError("immediate-parent control-row identity does not match basis columns")
        report.update({
            "classification":"read-only elastic precision Phase-I diagnostic; no candidate emitted",
            "all_original_rows_remain_hard":True,
            "relaxed_row_range_A":[int(receipt["duplicate_self_row_start"]),int(receipt["duplicate_self_row_stop"])],
            "relaxed_row_count":int(len(source_indices)),
            "original_hard_row_count":int(problem["A"].shape[0]-len(source_indices)),
            "soft_clearance_rows_omitted":int(receipt["soft_clearance_rows_omitted"]),
            "input_hashes":{str(k):str(v) for k,v in local["input_hashes"].items()},
            "phase_one_receipt":receipt,
            "A_shape":list(problem["A"].shape),"A_nnz":int(problem["A"].nnz),"A_sha256":_csr_sha(problem["A"]),
            "M_shape":list(problem["M"].shape),"M_nnz":int(problem["M"].nnz),"M_sha256":_csr_sha(problem["M"]),
            "basis_control_parent_rows":list(map(int,controls)),
            "support_nodes":int(len(local["support"])),
            "native_anchor_count":int(len(local["anchors"])),
            "fixed_prefix_nodes":int(local["fixed_prefix_nodes"]),
            "row_categories":{"self_pair_including_precision":int(len(local["self_pair_rows"])),
                               "fixed_bone":int(len(local["fixed_boundary_rows"])),
                               "point_depth":int(len(local["point_depth_barrier_rows"])),
                               "area":int(len(local["area_faces"])),
                               "edge_gradient":int(_csr(local["edge_gradient_A"]).shape[0])},
        })
        if result.success:
            scale=problem["extra_rhs"]; sl=arrays["precision_slack_mm"]
            group={}
            for j,key in enumerate(pair_keys):
                agg=group.setdefault(key,{"row_indices":[],"weighted_cost":0.0,"slack_sum_mm":0.0,
                                          "extra_rhs_sum_mm":0.0,"max_fractional_shortfall":0.0,
                                          "rows":[]})
                agg["row_indices"].append(j); agg["weighted_cost"]+=float(problem["weights"][j]*sl[j])
                agg["slack_sum_mm"]+=float(sl[j]); agg["extra_rhs_sum_mm"]+=float(scale[j])
                frac=float(sl[j]/scale[j]); agg["max_fractional_shortfall"]=max(agg["max_fractional_shortfall"],frac)
                src=int(source_indices[j]); row=local["self_pair_rows"][src]
                nodes=np.asarray(row["nodes"],dtype=np.int64); coeff=np.asarray(row["coefficients"],dtype=np.float64)
                ag=problem["AM"][int(receipt["duplicate_self_row_start"])+j]
                cols=ag.indices
                control_nodes=sorted({int(controls[int(c)//3]) for c in cols})
                rec={"duplicate_A_row":int(receipt["duplicate_self_row_start"]+j),
                     "source_A_row":src,"source_026_row":int(source_indices[j]),
                     "pair_kind":key[0],"face_rows":list(key[1]),"label":str(row["label"]),
                     "source_csr_nodes":nodes.tolist(),"source_csr_coefficients_mm":coeff.tolist(),
                     "added_rhs_mm":float(scale[j]),"slack_mm":float(sl[j]),
                     "fractional_shortfall":frac,"absolute_dual":float(arrays["precision_dual_abs"][j]),
                     "immediate_control_parent_rows":control_nodes}
                agg["rows"].append(rec)
            pair_rank=[]
            for key,agg in group.items():
                local_ids=np.asarray(agg["row_indices"],dtype=np.int64)
                mat=problem["AM"][problem["soft_rows"][local_ids]]
                cols=np.unique(mat.indices)
                dense=mat[:,cols].toarray() if len(cols) else np.empty((len(local_ids),0))
                agg["control_support_rank"]=int(np.linalg.matrix_rank(dense)) if dense.size else 0
                agg["control_support_column_count"]=int(len(cols))
                pair_rank.append({k:v for k,v in agg.items()})
            pair_rank.sort(key=lambda x:(-x["weighted_cost"],-x["max_fractional_shortfall"],x["face_rows"]))
            report["offending_pairs_top64"]=pair_rank[:64]
            rows=sorted([r for g in pair_rank for r in g["rows"]],
                        key=lambda r:(-r["fractional_shortfall"],-r["absolute_dual"],r["face_rows"]))
            report["offending_rows_top128"]=rows[:128]
            dual_rows=sorted(rows,key=lambda r:(-r["absolute_dual"],-r["fractional_shortfall"],r["face_rows"]))
            report["largest_precision_row_duals_top64"]=dual_rows[:64]
            arrays["phase_pair_face_rows"]=np.asarray([x["face_rows"] for x in pair_findings],dtype=np.int64)
            arrays["phase_pair_kind"]=np.asarray([x["pair_kind"] for x in pair_findings],dtype="U32")
            arrays["source_csr_node_indptr"]=np.asarray([0]+list(np.cumsum([len(local["self_pair_rows"][int(i)]["nodes"]) for i in source_indices])),dtype=np.int64)
            arrays["source_csr_nodes"]=np.concatenate([np.asarray(local["self_pair_rows"][int(i)]["nodes"],dtype=np.int64) for i in source_indices])
            arrays["source_csr_coefficients_mm"]=np.concatenate([np.asarray(local["self_pair_rows"][int(i)]["coefficients"],dtype=np.float64) for i in source_indices],axis=0)
        captured.update(report=report,arrays=arrays)
        raise CaptureComplete()
    module.solve_sparse_l1_clearance_in_basis=capture
    sys.argv=[str(script),*argv[2:]]
    try: module.main()
    except CaptureComplete: pass
    if "report" not in captured: raise RuntimeError("driver did not reach reduced-basis solver")
    output_dir.mkdir(parents=True,exist_ok=False)
    (output_dir/"elastic-phase-one-report.json").write_text(json.dumps(captured["report"],indent=2,sort_keys=True)+"\n")
    if captured["arrays"]: np.savez_compressed(output_dir/"elastic-phase-one-primal-diagnostic.npz",**captured["arrays"])
    return captured["report"]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--argv-json",required=True); p.add_argument("--output-dir",required=True)
    p.add_argument("--time-limit-s",type=float,default=60.0); p.add_argument("--lp-method",default="highs-ipm")
    a=p.parse_args()
    r=run_capture(Path(a.argv_json),Path(a.output_dir),time_limit_s=a.time_limit_s,method=a.lp_method)
    print(json.dumps(r,indent=2,sort_keys=True))


if __name__=="__main__": main()
