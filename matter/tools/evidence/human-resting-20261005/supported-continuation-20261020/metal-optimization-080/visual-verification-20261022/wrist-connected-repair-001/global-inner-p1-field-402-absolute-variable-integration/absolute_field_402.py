from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _int_array(value, name, ndim):
    a=np.asarray(value)
    if a.ndim!=ndim or (a.size and not np.issubdtype(a.dtype,np.integer)):
        raise ValueError(f"{name} must be an integer array with ndim={ndim}")
    return a.astype(np.int64,copy=False)

def solve_absolute_obstacle_field(p0, tetrahedra, constraint_tet_ids,
                                  constraint_barycentric, fixed_ids,
                                  fixed_values, node_costs, target,
                                  max_value=1.0, time_limit_s=120.0):
    """311-compatible weighted-minimum solve with absolute nodal unknowns.

    The objective is the existing weighted minimum c^T phi. p0 is validated
    and retained as a diagnostic initializer only; it is never a bound or
    objective term. Explicit fixed values are the only equality constraints.
    Obstacle CSR rows are assembled vectorially over four tet corners.
    """
    try: max_value=float(max_value)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError("max_value must be finite and positive") from exc
    if not np.isfinite(max_value) or max_value<=0:
        raise ValueError("max_value must be finite and positive")
    try: target=float(target);time_limit_s=float(time_limit_s)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError("target and time limit must be finite") from exc
    if not np.isfinite(target) or target<0 or target>max_value:
        raise ValueError("invalid target")
    if not np.isfinite(time_limit_s) or time_limit_s<=0:
        raise ValueError("invalid time limit")
    p0=np.asarray(p0,dtype=np.float64)
    if p0.ndim!=1 or not np.isfinite(p0).all() or np.any(p0<0) or np.any(p0>max_value):
        raise ValueError("initializer must be finite and within absolute field bounds")
    T=_int_array(tetrahedra,"tetrahedra",2)
    ids=_int_array(constraint_tet_ids,"constraint_tet_ids",1)
    fixed=_int_array(fixed_ids,"fixed_ids",1)
    if T.shape[1:]!=(4,) or np.any(T<0) or np.any(T>=len(p0)):
        raise ValueError("invalid tetrahedron table")
    if np.any(ids<0) or np.any(ids>=len(T)):
        raise ValueError("invalid constraint tet IDs")
    B=np.asarray(constraint_barycentric,dtype=np.float64)
    if B.shape!=(len(ids),4) or not np.isfinite(B).all() or np.any(B<0):
        raise ValueError("invalid barycentric constraints")
    if len(B) and np.max(np.abs(B.sum(axis=1)-1))>1e-10:
        raise ValueError("constraint barycentric rows must sum to one")
    fv=np.asarray(fixed_values,dtype=np.float64);costs=np.asarray(node_costs,dtype=np.float64)
    if fixed.shape!=fv.shape or np.any(fixed<0) or np.any(fixed>=len(p0)):
        raise ValueError("invalid explicit fixed-value map")
    if len(np.unique(fixed))!=len(fixed) or not np.isfinite(fv).all() or np.any(fv<0) or np.any(fv>max_value):
        raise ValueError("fixed IDs must be unique with values in [0,max_value]")
    if costs.shape!=p0.shape or not np.isfinite(costs).all() or np.any(costs<=0):
        raise ValueError("node costs must be finite and positive")
    fixed_mask=np.zeros(len(p0),dtype=bool);fixed_mask[fixed]=True
    fixed_value=np.zeros(len(p0),dtype=np.float64);fixed_value[fixed]=fv
    free_ids=np.flatnonzero(~fixed_mask)
    free_index=np.full(len(p0),-1,dtype=np.int64);free_index[free_ids]=np.arange(len(free_ids))
    TN=T[ids] if len(ids) else np.empty((0,4),dtype=np.int64)
    fixed_part=np.sum(B*np.where(fixed_mask[TN],fixed_value[TN],0.0),axis=1) if len(ids) else np.empty(0)
    free_coeff=np.where(~fixed_mask[TN],B,0.0) if len(ids) else np.empty((0,4))
    capacity=fixed_part+max_value*free_coeff.sum(axis=1)
    impossible=np.flatnonzero(capacity<target-1e-12)
    if len(impossible):
        return None,{"pass":False,"status":"fixed_support_conflict","constraints":int(len(ids)),
          "conflict_rows":impossible[:64].tolist(),"conflict_tet_ids":ids[impossible[:64]].tolist(),
          "minimum_capacity":float(capacity[impossible].min()),"target":target,
          "initializer_as_lower_bound":False}
    active_rows=np.flatnonzero(fixed_part<target-1e-12)
    out=np.zeros_like(p0);out[fixed]=fv
    if not len(active_rows):
        values=np.einsum("ij,ij->i",B,out[TN]) if len(ids) else np.empty(0)
        bad=np.flatnonzero(~np.isfinite(values)|(values<target-1e-8))
        if len(bad):
            return None,{"pass":False,"status":"fixed_value_verification_failed","failed_rows":bad[:64].tolist(),
              "constraints":int(len(ids)),"minimum_constraint_value":float(np.nanmin(values))}
        return out,{"pass":True,"status":"already_satisfied_or_no_free_support_needed","constraints":int(len(ids)),
          "active_rows":0,"changed_nodes":int(np.count_nonzero(out!=p0)),"objective":float(costs@out),
          "initializer_role":"diagnostic_only_not_bound_or_objective","absolute_bounds":[0.0,max_value],
          "fixed_values_exact":True}
    W=free_coeff[active_rows]
    support_count=np.count_nonzero(W>0,axis=1)
    if np.any(support_count==0):
        bad=active_rows[support_count==0]
        return None,{"pass":False,"status":"fixed_support_conflict","constraints":int(len(ids)),
          "active_rows":int(len(active_rows)),"conflict_rows":bad[:64].tolist(),
          "conflict_tet_ids":ids[bad[:64]].tolist(),"target":target,"initializer_as_lower_bound":False}
    scales=np.max(W,axis=1)
    rr,corner=np.nonzero(W>0)
    global_tet_rows=active_rows[rr]
    cols=free_index[TN[global_tet_rows,corner]]
    data=-W[rr,corner]/scales[rr]
    rhs=(fixed_part[active_rows]-target)/scales
    A=coo_matrix((data,(rr,cols)),shape=(len(active_rows),len(free_ids))).tocsr()
    result=linprog(costs[free_ids],A_ub=A,b_ub=rhs,
      bounds=[(0.0,max_value)]*len(free_ids),method="highs",
      options={"time_limit":time_limit_s,"primal_feasibility_tolerance":1e-9,
               "dual_feasibility_tolerance":1e-9})
    report={"pass":bool(result.success),"status":str(result.message),"solver_status":int(result.status),
      "iterations":int(getattr(result,"nit",0)),"constraints":int(len(ids)),"active_rows":int(len(active_rows)),
      "sparse_shape":[int(A.shape[0]),int(A.shape[1])],"sparse_nnz":int(A.nnz),
      "initializer_role":"diagnostic_only_not_bound_or_objective","absolute_bounds":[0.0,max_value],
      "fixed_node_count":int(len(fixed)),"target":target,
      "objective_policy":"preserved 311 weighted minimum c^T phi"}
    if not result.success or result.x is None:return None,report
    sx=np.asarray(result.x,dtype=np.float64)
    if sx.shape!=(len(free_ids),) or not np.isfinite(sx).all() or np.any(sx<0) or np.any(sx>max_value):
        report.update({"pass":False,"status":"solver_output_validation_failed","failure":"nonfinite_or_out_of_bounds_solver_vector"})
        return None,report
    out[free_ids]=sx
    values=np.einsum("ij,ij->i",B,out[TN])
    bad=np.flatnonzero(~np.isfinite(values)|(values<target-1e-8))
    if not np.isfinite(out).all() or np.any(out<0) or np.any(out>max_value) or not np.array_equal(out[fixed],fv) or len(bad):
        report.update({"pass":False,"status":"solver_output_validation_failed","failure":"postsolve_field_or_constraint_check",
          "verification_failures":int(len(bad)),"failed_rows":bad[:64].tolist(),
          "minimum_constraint_value":float(np.nanmin(values)) if len(values) else None})
        return None,report
    report.update({"changed_nodes":int(np.count_nonzero(out!=p0)),"objective":float(costs@out),
      "minimum_constraint_value":float(values.min()) if len(values) else None,
      "maximum_field_value":float(out.max()) if len(out) else None,"fixed_values_exact":True})
    return out,report

def reactivate_skipped_tets(tetrahedra, field_values, considered_tet_ids, threshold):
    T=_int_array(tetrahedra,"tetrahedra",2);considered=_int_array(considered_tet_ids,"considered_tet_ids",1)
    p=np.asarray(field_values,dtype=np.float64)
    try: threshold=float(threshold)
    except (TypeError,ValueError,OverflowError) as exc: raise ValueError("threshold must be finite") from exc
    if not np.isfinite(threshold):raise ValueError("threshold must be finite")
    if p.ndim!=1 or not np.isfinite(p).all() or T.shape[1:]!=(4,) or np.any(T<0) or np.any(T>=len(p)):
        raise ValueError("invalid mesh or nonfinite field")
    if np.any(considered<0) or np.any(considered>=len(T)) or len(np.unique(considered))!=len(considered):
        raise ValueError("invalid considered tet IDs")
    mask=np.ones(len(T),dtype=bool);mask[considered]=False;skipped=np.flatnonzero(mask)
    if not len(skipped):return skipped
    return skipped[np.min(p[T[skipped]],axis=1)<threshold]

def certify_skipped_tets(tetrahedra, field_values, considered_tet_ids, threshold):
    T=_int_array(tetrahedra,"tetrahedra",2);considered=_int_array(considered_tet_ids,"considered_tet_ids",1)
    new=reactivate_skipped_tets(T,field_values,considered,threshold)
    return {"pass":bool(len(new)==0),"skipped_tet_count":int(len(T)-len(considered)),
      "reactivate_tet_ids":new.tolist(),"threshold":float(threshold),
      "proof":"P1 field is a convex barycentric combination on each tet; min nodal value bounds every point in that tet."}
