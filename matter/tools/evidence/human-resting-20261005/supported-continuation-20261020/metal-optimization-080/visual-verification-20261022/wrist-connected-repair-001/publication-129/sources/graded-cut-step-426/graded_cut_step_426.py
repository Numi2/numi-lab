"""Construction-only scale-field LP using current exact witnesses and frozen SAT rows."""
from pathlib import Path
import importlib.util, hashlib
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
BASE=Path("/Users/numi/Work/numi-human-connected-surface-069")
G=BASE/"D/graded-local-step-423/graded_local_step_423.py"
SAT=BASE/"barycentric-connected-clearance-170/connected-local-continuation-189-run001/round-005/proposal/round-001/fit/self_separation_026.py"
def load(p,name,h):
    if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError("owner hash changed")
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
g=load(G,"graded423","72cb043f1849a66f6829b1767717d574d0e25fd286bf40cfeab1c96a9cab3b8f")
sat=load(SAT,"sat026","c67daf7c9fd2292e32ae88dba215a74350e1cbf83c669bc299a64b86d12e99f3")
_PAIRS=set()
def solve_scale_field(faces, upper, anchors, edges, budgets, displacement_mm, pair_rows):
    n=len(upper);rr=[];cc=[];vv=[];rhs=[]
    def add(nodes,values,bound):
        values=np.asarray(values,float);mx=np.max(np.abs(values),initial=0)
        if mx<1e-16:
            if bound < -1e-10:raise ValueError("infeasible constant row")
            return
        i=len(rhs);rr.extend([i]*len(nodes));cc.extend(map(int,nodes));vv.extend((values/mx).tolist());rhs.append(float(bound/mx))
    for row in pair_rows:
        nodes=row["nodes"];v=np.einsum("ij,ij->i",row["coefficients"],displacement_mm[nodes])
        add(nodes,-v,-row["lower"])
    mask=np.zeros(n,bool);mask[np.asarray(anchors,int)]=True
    for (u,v),b in zip(edges,budgets):
        if b>=1 or mask[u] or mask[v]:continue
        add([u,v],[1.,-1.],b);add([u,v],[-1.,1.],b)
    amp=np.linalg.norm(displacement_mm,axis=1);w=np.maximum(amp,1e-8);w/=max(w.max(),1e-8)
    ub=np.asarray(upper,float).copy();ub[np.asarray(anchors,int)]=0
    A=coo_matrix((vv,(rr,cc)),shape=(len(rhs),n)).tocsr();b=np.asarray(rhs)
    result=linprog(-w,A_ub=A,b_ub=b,bounds=np.column_stack([np.zeros(n),ub]),method="highs",
                   options={"time_limit":15.,"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9})
    if not result.success:raise ValueError("scale LP failed:"+result.message)
    x=np.minimum(ub,np.maximum(0,result.x))
    residual=float(np.max(A@x-b,initial=0))
    if residual>1e-8:raise ValueError("scale LP row certificate failed")
    return x,dict(lp_success=True,lp_iterations=int(result.nit),lp_rows=len(rhs),lp_variables=n,
                  normalized_max_row_violation=residual,retained_weighted_scale=float(w@x),possible_weighted_scale=float(w@ub))
def reduce_witness_node_scales(faces,step_scales,bad_face_rows,self_pair_face_rows,*,anchors=(),zero_below=1/128):
    upper,report=g.reduce_witness_node_scales(faces,step_scales,bad_face_rows,self_pair_face_rows,anchors=anchors,zero_below=zero_below)
    c=g.load_context_423()
    pairs=np.asarray(self_pair_face_rows,int).reshape(-1,2)
    _PAIRS.update(tuple(sorted(map(int,p)))for p in pairs)
    if not _PAIRS:return upper,report
    import os
    with np.load(os.environ["NUMI_GRADED_LOCAL_STEP_CONTEXT_423"],allow_pickle=False)as z:
        baseline=z["points_baseline_m"].astype(float)*1000;delta=z["proposal_displacement_m"].astype(float)*1000
    rows,records=sat.build_pair_rows(baseline,faces,np.asarray(sorted(_PAIRS),int),margin_fraction=.25)
    if max((r["lower"]for r in rows),default=0)>2e-8:raise ValueError("baseline SAT zero step infeasible")
    out,proof=solve_scale_field(faces,upper,anchors,c["edges"],c["budget"],delta,rows)
    report.update(status="GRADED_WITNESS_SAT_SCALE_LP_ONLY",policy="183_423_THEN_026_SCALE_LP",sat_pair_count=len(_PAIRS),sat_row_count=len(rows),scale_lp=proof)
    return out,report
