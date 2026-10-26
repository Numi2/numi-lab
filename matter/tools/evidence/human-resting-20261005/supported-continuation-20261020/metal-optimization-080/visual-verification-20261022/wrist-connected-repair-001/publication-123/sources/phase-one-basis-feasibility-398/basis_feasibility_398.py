#!/usr/bin/env python3
"""Read-only feasibility comparison for 396 precision Phase-I output mapping."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


class CaptureComplete(Exception):
    pass


def _csr_sha(a):
    a=sparse.csr_matrix(a,dtype=np.float64)
    return hashlib.sha256(a.indptr.tobytes()+a.indices.tobytes()+a.data.tobytes()).hexdigest()


def _certificate(A,b,lo,hi,d):
    d=np.asarray(d,dtype=np.float64).reshape(-1)
    hard=np.asarray(A@d).reshape(-1)-b
    bound=max(float(np.max(np.maximum(lo-d,0),initial=0.0)),
              float(np.max(np.maximum(d-hi,0),initial=0.0)))
    min_hard=float(np.min(hard,initial=np.inf))
    return {
        "passed": bool(np.isfinite(d).all() and min_hard>=-2e-7 and bound<=2e-7),
        "minimum_hard_residual": min_hard,
        "maximum_output_bound_violation": bound,
        "primal_linf": float(np.max(np.abs(d),initial=0.0)),
        "primal_l2": float(np.linalg.norm(d)),
    }


def compare_feasibility(A,b,lo,hi,M,*,time_limit_s=90.0,method="highs-ipm"):
    """Compare same A d>=b and lo<=d<=hi in q where d=Mq and in free d."""
    A=sparse.csr_matrix(A,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64).reshape(-1)
    lo=np.asarray(lo,dtype=np.float64).reshape(-1)
    hi=np.asarray(hi,dtype=np.float64).reshape(-1)
    M=sparse.csr_matrix(M,dtype=np.float64)
    n=A.shape[1]
    if A.shape[0]!=len(b) or len(lo)!=n or len(hi)!=n or M.shape[0]!=n or M.shape[1]==0:
        raise ValueError("A/b/bounds/output-map dimensions mismatch")
    if not all(np.isfinite(x).all() for x in (A.data,b,lo,hi,M.data)) or np.any(lo>hi):
        raise ValueError("nonfinite rows or inverted bounds")
    opts={"time_limit":float(time_limit_s),"presolve":True}
    t=time.monotonic()
    reduced_A=sparse.vstack((A@M,M,-M),format="csr")
    reduced_b=np.concatenate((b,lo,-hi))
    qr=linprog(np.zeros(M.shape[1]),A_ub=-reduced_A,b_ub=-reduced_b,
               bounds=[(None,None)]*M.shape[1],method=method,options=opts)
    reduced_seconds=time.monotonic()-t
    t=time.monotonic()
    fr=linprog(np.zeros(n),A_ub=-A,b_ub=-b,bounds=list(zip(lo,hi)),
               method=method,options=opts)
    full_seconds=time.monotonic()-t
    report={
        "classification":"read-only construction feasibility diagnostic; no candidate emitted; no mapping/native admission change",
        "same_hard_rows_and_output_bounds":True,
        "lp_method":method,"time_limit_s":float(time_limit_s),
        "matrix":{"A_shape":list(A.shape),"A_nnz":int(A.nnz),"A_sha256":_csr_sha(A),
                  "M_shape":list(M.shape),"M_nnz":int(M.nnz),"M_sha256":_csr_sha(M),
                  "b_sha256":hashlib.sha256(b.tobytes()).hexdigest(),
                  "lower_sha256":hashlib.sha256(lo.tobytes()).hexdigest(),
                  "upper_sha256":hashlib.sha256(hi.tobytes()).hexdigest()},
        "reduced_basis":{"status":int(qr.status),"success":bool(qr.success),
                          "message":str(qr.message),"iterations":int(getattr(qr,"nit",0)),
                          "wall_seconds":reduced_seconds},
        "full_output_coordinates":{"status":int(fr.status),"success":bool(fr.success),
                                    "message":str(fr.message),"iterations":int(getattr(fr,"nit",0)),
                                    "wall_seconds":full_seconds},
    }
    arrays={}
    if qr.success:
        q=np.asarray(qr.x,dtype=np.float64)
        d=np.asarray(M@q).reshape(-1)
        report["reduced_basis"]["output_primal_certificate"]=_certificate(A,b,lo,hi,d)
        if not report["reduced_basis"]["output_primal_certificate"]["passed"]:
            raise RuntimeError("reduced LP primal failed exact output-space certificate")
        arrays["reduced_control_primal_q"]=q
        arrays["reduced_output_primal_d"]=d
    if fr.success:
        d=np.asarray(fr.x,dtype=np.float64)
        report["full_output_coordinates"]["primal_certificate"]=_certificate(A,b,lo,hi,d)
        if not report["full_output_coordinates"]["primal_certificate"]["passed"]:
            raise RuntimeError("full-space LP primal failed output-space certificate")
        arrays["full_output_primal_d"]=d
    return report,arrays


def run_capture(argv_json:Path,output_dir:Path,time_limit_s=30.0):
    argv=json.loads(argv_json.read_text())
    script=Path(argv[1])
    sys.path.insert(0,str(script.parent))
    spec=importlib.util.spec_from_file_location("driver396_phase_capture",script)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    captured={}
    def capture(A,b,target,lower,upper,C,h,weights,output_map,*args,**kwargs):
        if sparse.csr_matrix(C).shape[0] != 0:
            raise ValueError("expected Phase-I assembly with soft clearance rows omitted")
        local=__import__("inspect").currentframe().f_back.f_locals
        report,primal=compare_feasibility(A,b,lower,upper,output_map,time_limit_s=time_limit_s)
        report["phase_one_receipt"]=local.get("precision_phase_one_receipt")
        report["assembly_rows"]={
            "hard_rows":int(sparse.csr_matrix(A).shape[0]),
            "self_pair_rows":int(len(local.get("self_pair_rows",()))),
            "fixed_boundary_rows":int(len(local.get("fixed_boundary_rows",()))),
            "point_depth_rows":int(len(local.get("point_depth_barrier_rows",()))),
            "area_rows":int(len(local.get("area_faces",()))),
            "edge_gradient_rows":int(sparse.csr_matrix(local.get("edge_gradient_A")).shape[0]),
            "support_nodes":int(len(local.get("support",()))),
            "fixed_typed_anchor_rows":int(len(local.get("anchors",()))),
        }
        captured["report"]=report
        captured["arrays"]=primal
        raise CaptureComplete()
    module.solve_sparse_l1_clearance_in_basis=capture
    sys.argv=[str(script),*argv[2:]]
    try:
        module.main()
    except CaptureComplete:
        pass
    if "report" not in captured:
        raise RuntimeError("driver did not reach the basis solver")
    output_dir.mkdir(parents=True,exist_ok=False)
    report_path=output_dir/"feasibility-comparison.json"
    report_path.write_text(json.dumps(captured["report"],indent=2,sort_keys=True)+"\n")
    if captured["arrays"]:
        np.savez_compressed(output_dir/"primal-certificates.npz",**captured["arrays"])
    return captured["report"]


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--argv-json",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--time-limit-s",type=float,default=30.0)
    a=p.parse_args()
    report=run_capture(Path(a.argv_json),Path(a.output_dir),time_limit_s=a.time_limit_s)
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
