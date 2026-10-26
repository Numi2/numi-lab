"""Construction-only point-priority pass for one metadata-selected C row (535)."""
from __future__ import annotations
import hashlib, inspect, time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

SCHEMA="numi.human.point_priority_pass_535.v1"
OWNER=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py")
OWNER_SHA="5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2"
DECOMP=Path("/Users/numi/Work/numi-human-connected-surface-069/D/D/component-lp-decomposition-485/component_lp_decomposition_485.py")
DECOMP_SHA="b8632e0734ff0163e854b9c0c5fbfe33c6d12c7b17cbb7935c5028c8df78ef74"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def csr(a):
    x=sparse.csr_matrix(a,dtype=np.float64,copy=True)
    x.sum_duplicates(); x.eliminate_zeros(); x.sort_indices()
    if not np.isfinite(x.data).all(): raise ValueError("nonfinite sparse input")
    return x

def same_sparse(a,b):
    a=csr(a); b=csr(b)
    if a.shape!=b.shape: return False
    d=(a-b).tocsr(); d.sum_duplicates(); d.eliminate_zeros()
    return d.nnz==0

def identify_priority_point_row(metadata,point_prefix_count,query_row):
    if not isinstance(metadata,(list,tuple)) or not all(isinstance(x,dict) for x in metadata):
        raise ValueError("clearance metadata must be row dictionaries")
    if isinstance(point_prefix_count,(bool,np.bool_)) or not isinstance(point_prefix_count,(int,np.integer)):
        raise ValueError("point_prefix_count must be integer")
    p=int(point_prefix_count)
    if not 0<=p<=len(metadata): raise ValueError("point prefix count out of range")
    if any(metadata[i].get("kind")!="point" for i in range(p)): raise ValueError("bad point prefix")
    if any(x.get("kind")=="point" for x in metadata[p:]): raise ValueError("point rows are not a prefix")
    found=[i for i,x in enumerate(metadata[:p]) if x.get("query_row")==int(query_row)]
    if len(found)!=1: raise ValueError("query_row does not identify exactly one point-prefix row")
    return int(found[0])

def patch_owner_clearance_builder(owner,state,query_row):
    path=Path(getattr(owner,"__file__","")).resolve()
    if path!=OWNER.resolve() or sha(path)!=OWNER_SHA: raise ValueError("535 408 owner pin mismatch")
    if getattr(owner,"_priority_535_builder_hook",False): raise RuntimeError("535 builder hook already installed")
    original=owner.build_clearance_rows; sig=inspect.signature(original)
    def wrapped(*args,**kwargs):
        bound=sig.bind(*args,**kwargs); C,h,w,meta=original(*args,**kwargs)
        prefix=len(np.asarray(bound.arguments["point_targets"].get("point_rows",())).reshape(-1))
        row=identify_priority_point_row(meta,prefix,query_row); C=csr(C); h=np.asarray(h,float).reshape(-1)
        if C.shape[0]!=len(meta) or h.shape!=(len(meta),): raise ValueError("C/h/metadata mismatch")
        capture={"query_row":int(query_row),"point_prefix_count":prefix,"priority_row_index":row,
                 "priority_c_functional":C.getrow(row).copy(),"priority_target_mm":float(h[row]),
                 "metadata":dict(meta[row]),"clearance_row_count":int(C.shape[0])}
        old=state.get("capture")
        if old and (old["priority_row_index"]!=row or old["point_prefix_count"]!=prefix or
                    old["priority_target_mm"]!=capture["priority_target_mm"] or
                    not same_sparse(old["priority_c_functional"],capture["priority_c_functional"])):
            raise RuntimeError("535 target row changed during fit")
        state["capture"]=capture
        return C,h,w,meta
    owner.build_clearance_rows=wrapped; owner._priority_535_builder_hook=True
    def restore():
        owner.build_clearance_rows=original; owner._priority_535_builder_hook=False
    return {"owner_path":str(path),"owner_sha256":OWNER_SHA,"query_row":int(query_row),"restore":restore}

def fail(msg,status=2,receipt=None,wall=0.):
    return {"success":False,"status":int(status),"message":str(msg),"nit":0,"objective":None,
            "wall_seconds":float(wall),"lp_method":None,"point_priority_535":receipt or {}}

def solve_with_point_priority(solver,A,b,target,lo,hi,C,h,weights,M,time_limit_s,
        slack_tolerance=1e-9,allow_stage1_feasible_timeout_fallback=False,lp_method="highs",
        retained_gain_fraction=None,*,point_prefix_count,priority_row_index,
        priority_c_functional=None,priority_query_row=None,retention_fraction=.95,
        priority_time_fraction=.20):
    """Maximize a point C-row under original A/b/bounds/M; retain a fraction in pooled LP."""
    start=time.monotonic(); A=csr(A); C=csr(C); M=csr(M)
    b=np.asarray(b,float).reshape(-1); target=np.asarray(target,float).reshape(-1)
    lo=np.asarray(lo,float).reshape(-1); hi=np.asarray(hi,float).reshape(-1)
    h=np.asarray(h,float).reshape(-1); w=np.asarray(weights,float).reshape(-1)
    no,nq=M.shape; m=len(h); p=int(point_prefix_count); ri=int(priority_row_index); T=float(time_limit_s)
    if A.shape!=(len(b),no) or C.shape!=(m,no) or target.shape!=(no,) or lo.shape!=(no,) or hi.shape!=(no,) or w.shape!=(m,):
        raise ValueError("535 LP dimensions do not align")
    if not all(np.isfinite(x).all() for x in (b,target,lo,hi,h,w)) or np.any(lo>hi) or np.any(h<=0) or np.any(w<0):
        raise ValueError("535 LP values invalid")
    retain=float(retention_fraction); tf=float(priority_time_fraction)
    if not (0<p<=m and 0<=ri<p): raise ValueError("priority row outside point prefix")
    if not 0<retain<=1 or not 0<tf<1 or not np.isfinite(T) or T<=0: raise ValueError("invalid retention/time policy")
    crow=C.getrow(ri)
    if priority_c_functional is not None and not same_sparse(crow,priority_c_functional):
        raise ValueError("priority C functional differs from selected row")
    cmap=csr(crow@M)
    if not crow.nnz or not cmap.nnz: raise ValueError("priority row has no reachable output-map control")
    H=csr(sparse.vstack((A@M,M,-M),format="csr")); rhs=np.r_[b,lo,-hi]
    tlim=T*tf; poollim=T*(1.-tf/2.)
    t0=time.monotonic()
    try:
        opt=linprog(-np.asarray(cmap.toarray()).reshape(-1),A_ub=-H,b_ub=-rhs,
                    bounds=[(None,None)]*nq,method=lp_method,
                    options={"time_limit":tlim,"presolve":True})
    except Exception as e:
        return fail("priority LP exception; pooled solve skipped",receipt={"status":"PRIORITY_EXCEPTION","error":repr(e)},wall=time.monotonic()-start)
    tw=time.monotonic()-t0
    if not opt.success or opt.x is None:
        return fail("priority LP failed; pooled solve skipped",int(getattr(opt,"status",2)),
            {"status":"PRIORITY_FAILED","lp_status":int(getattr(opt,"status",2)),
             "lp_message":str(getattr(opt,"message","")),"time_limit_s":tlim},time.monotonic()-start)
    qstar=np.asarray(opt.x,float); ds=np.asarray(M@qstar).reshape(-1)
    hr=np.asarray(A@ds).reshape(-1)-b
    berr=max(float(np.max(np.maximum(lo-ds,0),initial=0)),float(np.max(np.maximum(ds-hi,0),initial=0)))
    best=float(np.asarray(crow@ds).reshape(-1)[0])
    if np.min(hr,initial=np.inf)<-2e-7 or berr>2e-7 or best < -2e-7:
        return fail("priority primal failed hard/bound certificate",receipt={"status":"PRIORITY_CERTIFICATE_FAILED",
            "max_progress_mm":best,"minimum_hard_residual":float(np.min(hr,initial=np.inf)),"max_bound_error":berr},wall=time.monotonic()-start)
    floor=float(retain*max(0.,best)); Aaug=sparse.vstack((A,crow),format="csr"); baug=np.r_[b,floor]
    t1=time.monotonic()
    try:
        out=solver(Aaug,baug,target,lo,hi,C,h,w,M,poollim,slack_tolerance=slack_tolerance,
            allow_stage1_feasible_timeout_fallback=allow_stage1_feasible_timeout_fallback,
            lp_method=lp_method,retained_gain_fraction=retained_gain_fraction)
    except Exception as e:
        return fail("pooled solver exception after priority pass",receipt={"status":"POOLED_EXCEPTION","error":repr(e)},wall=time.monotonic()-start)
    pw=time.monotonic()-t1
    rec={"schema":SCHEMA,"classification":"offline construction-only point-priority objective floor",
         "query_row":priority_query_row,"point_prefix_count":p,"priority_row_index":ri,
         "priority_target_mm":float(h[ri]),"max_progress_mm":best,"max_fraction":best/float(h[ri]),
         "retention_fraction":retain,"retained_floor_mm":floor,"priority_lp_status":int(opt.status),
         "priority_wall_seconds":tw,"priority_time_limit_s":tlim,"pooled_wall_seconds":pw,
         "pooled_per_stage_limit_s":poollim,"combined_two_stage_budget_bound_s":tlim+2*poollim,
         "original_hard_row_count":int(A.shape[0]),"added_offline_construction_row_count":1,
         "original_A_b_preserved":True,"bounds_C_h_weights_M_preserved":True,
         "basis_coordinates_preserved":True,"native_admission":False}
    if not isinstance(out,dict) or not out.get("success"):
        out=dict(out) if isinstance(out,dict) else {"success":False,"message":"non-dict solver result"}
        rec["status"]="PRIORITY_PASSED_POOLED_SOLVER_FAILED"; out["point_priority_535"]=rec; return out
    d=np.asarray(out.get("delta"),float).reshape(-1); q=np.asarray(out.get("basis_coordinates"),float).reshape(-1)
    s=np.asarray(out.get("normalized_slack"),float).reshape(-1)
    if d.shape!=(no,) or q.shape!=(nq,) or s.shape!=(m,): raise ValueError("pooled output shape mismatch")
    prog=np.asarray(C@d).reshape(-1); hres=np.asarray(A@d).reshape(-1)-b; tres=prog+h*s-h
    pp=float(np.asarray(crow@d).reshape(-1)[0])
    be=max(float(np.max(np.maximum(lo-d,0),initial=0)),float(np.max(np.maximum(d-hi,0),initial=0)),float(np.max(np.maximum(-s,0),initial=0)))
    maperr=float(np.max(np.abs(M@q-d),initial=0))
    cert=out.get("stage1_output_certificate")
    cert_failed=not isinstance(cert,dict) or not bool(cert.get("passed",False))
    objective=out.get("normalized_slack_objective")
    objective_error=(abs(float(objective)-float(w@s)) if objective is not None else 0.0)
    if (np.min(hres,initial=np.inf)<-2e-7 or be>2e-7 or np.min(tres,initial=np.inf)<-2e-7 or
            pp<floor-2e-7 or maperr>2e-9 or cert_failed or objective_error>max(2e-7,abs(float(w@s))*1e-9)):
        rec.update({"status":"POOLED_CERTIFICATE_FAILED","minimum_original_hard_residual":float(np.min(hres,initial=np.inf)),
                    "minimum_target_residual":float(np.min(tres,initial=np.inf)),"max_bound_error":be,
                    "priority_floor_residual_mm":pp-floor,"basis_map_max_abs_error":maperr,
                    "stage1_output_certificate_passed":(None if not isinstance(cert,dict) else bool(cert.get("passed"))),
                    "weighted_slack_objective_error":objective_error})
        out.update({"success":False,"message":"535 pooled result failed original-space certificate","point_priority_535":rec}); return out
    rec.update({"status":"PRIORITY_AND_POOLED_CERTIFICATES_PASSED","achieved_progress_mm":pp,
                "achieved_fraction":pp/float(h[ri]),"priority_floor_residual_mm":pp-floor,
                "minimum_original_hard_residual":float(np.min(hres,initial=np.inf)),
                "minimum_target_residual":float(np.min(tres,initial=np.inf)),"max_bound_error":be,
                "basis_map_max_abs_error":maperr,"weighted_target_slack_recomputed":float(w@s)})
    out["point_priority_535"]=rec
    return out

def patch_component_decomposition(decomposition,state,*,retention_fraction=.95,priority_time_fraction=.20):
    path=Path(getattr(decomposition,"__file__","")).resolve()
    if path!=DECOMP.resolve() or sha(path)!=DECOMP_SHA: raise ValueError("535 485 pin mismatch")
    if getattr(decomposition,"_priority_535_installed",False): raise RuntimeError("535 component hook already installed")
    original=decomposition.solve_by_components; sig=inspect.signature(original)
    def wrapped(original_solver,*args,**kwargs):
        cap=state.get("capture")
        if not isinstance(cap,dict): raise RuntimeError("535 ran before owner target metadata capture")
        bound=sig.bind_partial(original_solver,*args,**kwargs)
        A=csr(bound.arguments["A"]); C=csr(bound.arguments["C"]); M=csr(bound.arguments["output_map"])
        ri=int(cap["priority_row_index"]); prefix=int(cap["point_prefix_count"])
        if ri>=prefix or ri>=C.shape[0] or not same_sparse(C.getrow(ri),cap["priority_c_functional"]):
            raise ValueError("535 captured row differs from global C")
        AM=decomposition._csr(A@M,"535 A@M"); CM=decomposition._csr(C@M,"535 C@M")
        groups,labels,_=decomposition._components_from_rows((AM,CM,M),M.shape[1])
        clabels=decomposition._row_support_labels(CM,labels,"535 C@M")
        row_groups=[np.flatnonzero(clabels==i) for i in range(len(groups))]
        pc=int(clabels[ri])
        if pc<0: raise ValueError("priority row has no active component")
        calls={"i":0}; receipts=[]
        def local_solver(*la,**lk):
            ci=calls["i"]; calls["i"]+=1
            if ci>=len(row_groups): raise RuntimeError("535 saw excess component solver calls")
            if ci!=pc: return original_solver(*la,**lk)
            loc=np.flatnonzero(row_groups[ci]==ri)
            if len(loc)!=1: raise RuntimeError("535 cannot map global row into local component")
            local_prefix=int(np.count_nonzero(row_groups[ci]<prefix))
            out=solve_with_point_priority(original_solver,*la,point_prefix_count=local_prefix,
                priority_row_index=int(loc[0]),priority_query_row=cap["query_row"],
                retention_fraction=retention_fraction,priority_time_fraction=priority_time_fraction,**lk)
            receipts.append(out.get("point_priority_535",{})); return out
        if bool(getattr(original_solver,"__soft_target_no_regression_470__",False)):
            local_solver.__soft_target_no_regression_470__=True
        out=original(local_solver,*args,**kwargs)
        out["point_priority_535"]={"schema":SCHEMA,"classification":"offline priority pass in owning sparse component",
            "captured_row":{k:v for k,v in cap.items() if k!="priority_c_functional"},
            "priority_component":pc,"retention_fraction":float(retention_fraction),
            "priority_time_fraction":float(priority_time_fraction),"component_receipts":receipts,
            "original_hard_rows_and_bounds_preserved":True,"extra_row_is_offline_objective_constraint":True,
            "native_admission":False}
        return out
    decomposition.solve_by_components=wrapped; decomposition._priority_535_installed=True
    def restore():
        decomposition.solve_by_components=original; decomposition._priority_535_installed=False
    return {"schema":SCHEMA,"decomposition_path":str(path),"decomposition_sha256":DECOMP_SHA,
            "retention_fraction":float(retention_fraction),"priority_time_fraction":float(priority_time_fraction),
            "hook":"485.solve_by_components; only metadata-selected component is prioritized","restore":restore}

