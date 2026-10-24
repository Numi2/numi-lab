#!/usr/bin/env python3
"""Bounded sequential phase-I repair for the 017 run-002 local inverted tet star."""
from pathlib import Path
import hashlib, json, time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack

W=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
OUT=W/"connected-skin-safe-flow-025-generic-untangle"/"run-002"
OUT.mkdir(parents=True,exist_ok=True)
MESH=W/"E/connected-skin-mechanical-transfer-013/boundary-refinement-002/candidate.npz"
PROP=W/"connected-skin-safe-flow-017/run-002/connected-rest-proposal.npz"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(MESH)=="1849d24c14f8c0ddfedf3aa131877bdd38a852f9e48baebcf900a0efadf75db1"
assert sha(PROP)=="64f40e915703e35950ba2d3ab95b01a5e75681616c316cace25518c3a0ccb054"
m=np.load(MESH);p=np.load(PROP);T=np.asarray(m["tetrahedra"],np.int64)
Xbase=np.asarray(m["points_world_m"],np.float32).astype(np.float64)
Xref=np.asarray(p["points_world_m"],np.float32).astype(np.float64)
def detgrad(x,tt):
 q=x[tt];e1=q[:,1]-q[:,0];e2=q[:,2]-q[:,0];e3=q[:,3]-q[:,0]
 g1=np.cross(e2,e3);g2=np.cross(e3,e1);g3=np.cross(e1,e2)
 return np.einsum("ij,ij->i",e1,g1),np.stack((-g1-g2-g3,g1,g2,g3),axis=1)
def maxedge(x,tt):
 q=x[tt]
 return np.max(np.stack([np.linalg.norm(q[:,a]-q[:,b],axis=1) for a,b in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))],axis=1),axis=1)
assert T.shape==(253425,4) and Xbase.shape==Xref.shape==(56514,3)
assert np.array_equal(T,np.asarray(p["tetrahedra"]))
fac=np.sort(T[:,[[1,2,3],[0,3,2],[0,1,3],[0,2,1]]].reshape(-1,3),axis=1)
uf,fc=np.unique(fac,axis=0,return_counts=True);boundary=uf[fc==1]
declared=np.unique(np.vstack((m["physical_shell_faces"],m["construction_closure_faces"])),axis=0)
assert {tuple(r) for r in np.sort(boundary,axis=1)}=={tuple(r) for r in np.sort(declared,axis=1)}
fixed=np.zeros(len(Xref),bool);fixed[np.unique(boundary)]=1
fixed[np.asarray(m["physical_outer_node_ids"],np.int64)]=1
fixed[np.asarray(p["protected_node_ids"],np.int64)]=1
d0,_=detgrad(Xbase,T);dref,_=detgrad(Xref,T)
bad=np.flatnonzero(dref<=0);assert len(bad)==78
free=np.unique(T[bad][~fixed[T[bad]]]);assert len(free)==86
affected=np.flatnonzero(np.any(np.isin(T,free),axis=1));assert len(affected)==1011
Tl=T[affected];loc=np.full(len(Xref),-1,np.int64);loc[free]=np.arange(len(free))
locT=loc[Tl];L0=maxedge(Xref,Tl);sc=L0**3;target=1.005e-5
u=np.zeros((len(free),3),dtype=np.float64) # millimetres relative to 017
trace=[];t0=time.monotonic();status="bounded-iteration-limit"
max_iter=20;max_abs_mm=3.0;step_box_mm=0.5;l1_weight=1e-5

def vals_and_jac(u_now):
    x=Xref.copy();x[free]+=u_now*0.001
    d,g=detgrad(x,Tl);q=d/sc
    rows=[];cols=[];data=[]
    for c in range(4):
        keep=locT[:,c]>=0;rr=np.flatnonzero(keep);vv=locT[keep,c]
        for a in range(3):
            rows.append(rr);cols.append(3*vv+a);data.append(g[keep,c,a]*(0.001/sc[keep]))
    rr=np.concatenate(rows);cc=np.concatenate(cols);dd=np.concatenate(data)
    J=csr_matrix((dd,(rr,cc)),shape=(len(Tl),3*len(free)))
    return x,q,J

def merit(q):
    v=np.maximum(target-q,0.0)
    return (float(v.max(initial=0.0)),float(v.sum()),int(np.count_nonzero(v)))

for it in range(max_iter):
    if time.monotonic()-t0>520:
        status="bounded-timeout";break
    _,q,J=vals_and_jac(u)
    before=merit(q)
    # Uniform nonnegative phase-I slack, with a small L1 move tie-break.
    # Variables are [delta_mm(3N), slack, abs_delta_mm(3N)].
    n=3*len(free); eye=csr_matrix(np.eye(n))
    zcol=csr_matrix((np.ones(len(Tl)),(np.arange(len(Tl)),np.zeros(len(Tl)))),shape=(len(Tl),1))
    A1=hstack((-J,-zcol,csr_matrix((len(Tl),n))),format="csr")
    b1=q-target
    A2=hstack((eye,csr_matrix((n,1)),-eye),format="csr")
    A3=hstack((-eye,csr_matrix((n,1)),-eye),format="csr")
    A=vstack((A1,A2,A3),format="csr")
    b=np.concatenate((b1,np.zeros(2*n)))
    c=np.concatenate((np.zeros(n),np.array([1.0]),np.full(n,l1_weight)))
    current=u.reshape(-1)
    stepbounds=[]
    for xcur in current:
        lo=max(-step_box_mm,-max_abs_mm-xcur)
        hi=min(step_box_mm,max_abs_mm-xcur)
        stepbounds.append((lo,hi))
    bounds=stepbounds+[(0.0,None)]+[(0.0,None)]*n
    lp=linprog(c,A_ub=A,b_ub=b,bounds=bounds,method="highs")
    if not lp.success:
        status="phase1-lp-failed";trace.append({"iteration":it,"lp_status":int(lp.status),"message":lp.message});break
    delta=lp.x[:n].reshape(-1,3)
    slack=float(lp.x[n])
    alphas=(1.0,0.5,0.25,0.125,0.0625,0.03125)
    chosen=None
    for alpha in alphas:
        trial=u+alpha*delta
        _,qtrial,_=vals_and_jac(trial)
        after=merit(qtrial)
        if (after[0] < before[0]-1e-12) or (abs(after[0]-before[0])<=1e-12 and after[1] < before[1]-1e-12):
            chosen=(alpha,trial,qtrial,after);break
    if chosen is None:
        status="phase1-stagnation"
        trace.append({"iteration":it,"phase1_slack":slack,"before_merit":before,"accepted":False,
                      "max_lp_step_mm":float(np.abs(delta).max()),"max_abs_u_mm":float(np.abs(u).max())})
        break
    alpha,u,qnew,after=chosen
    trace.append({"iteration":it,"phase1_slack":slack,"before_merit":before,"after_merit":after,
                  "accepted_alpha":alpha,"max_lp_step_mm":float(np.abs(delta).max()),
                  "max_abs_u_mm":float(np.abs(u).max()),"max_node_move_mm":float(np.linalg.norm(u,axis=1).max()),
                  "elapsed_s":time.monotonic()-t0})
    xf=Xref.copy();xf[free]+=u*0.001;xf=xf.astype(np.float32).astype(np.float64)
    dg,_=detgrad(xf,T);qg=dg/maxedge(xf,T)**3
    if np.all(dg>0) and np.all(qg>=1e-5):
        status="positive-f32-qfloor"
        break

Xnew=Xref.copy();Xnew[free]+=u*0.001;Xnew32=Xnew.astype(np.float32);Xnew64=Xnew32.astype(np.float64)
dn,_=detgrad(Xnew64,T);qn=dn/maxedge(Xnew64,T)**3
dold,_=detgrad(Xbase,T);qold=dold/maxedge(Xbase,T)**3
fixed_equal=bool(np.array_equal(Xnew32[fixed],np.asarray(p["points_world_m"],np.float32)[fixed]))
nonfree=np.setdiff1d(np.arange(len(Xref)),free)
nonfree_equal=bool(np.array_equal(Xnew32[nonfree],np.asarray(p["points_world_m"],np.float32)[nonfree]))
out=OUT/"candidate.npz"
payload={k:np.asarray(m[k]) for k in ("parent_008_tet_rows","parent_005_tet_rows","region_attributes",
 "physical_outer_faces","physical_O_parent_face_rows","physical_O_parent_detailed_source_face_ids",
 "physical_O_parent_surface_ids","physical_O_parent_source271_face_rows","physical_O_parent_barycentric",
 "physical_outer_node_ids","physical_shell_faces","physical_shell_parent_face_rows","construction_closure_faces") if k in m.files}
payload.update({"points_world_m":Xnew32,"tetrahedra":T,"protected_node_ids":np.asarray(p["protected_node_ids"]),
 "untangle_free_node_ids":free,"untangle_constrained_tet_rows":affected,
 "proposal_node_displacements_m":np.asarray(p["node_displacements_m"]),
 "candidate_node_displacements_from_013_m":Xnew64-Xbase})
np.savez_compressed(out,**payload)
report={"schema":"numi.connected-skin-fixed-boundary-local-untangle.v1","status":"candidate_only_not_admitted",
 "method":"sequential phase-I LP using the same analytic signed-tet determinant Jacobian as owner 196; fixes every declared tetrahedral boundary node, physical O node, and 017 protected node; only the 86 interior nodes in the bad-cell set move; all 1,011 incident tets are constrained",
 "inputs":{"013_mesh":str(MESH),"013_sha256":sha(MESH),"017_proposal":str(PROP),"017_sha256":sha(PROP),
           "script":str(Path(__file__)),"script_sha256":sha(Path(__file__))},
 "pre":{"bad_tets":int(len(bad)),"free_nodes":int(len(free)),"incident_tets":int(len(affected)),
        "base_013_nonpositive":int(np.sum(dold<=0)),"base_013_min_q":float(qold.min()),
        "proposal_017_nonpositive":int(np.sum(dref<=0)),"proposal_017_min_q":float((dref/maxedge(Xref,T)**3).min()),
        "proposal_017_min_J_bad":float(np.min(dref[bad]/dold[bad])),"boundary_face_count":int(len(boundary)),
        "declared_boundary_matches_tet_boundary":True},
 "optimization":{"phase1_target_q":target,"max_abs_coordinate_move_mm":max_abs_mm,"step_box_mm":step_box_mm,
                 "iterations":trace,"final_status":status,"elapsed_s":time.monotonic()-t0},
 "post":{"F32_nonpositive":int(np.sum(dn<=0)),"F32_below_qfloor":int(np.sum(qn<1e-5)),
         "F32_min_det_m3":float(dn.min()),"F32_min_q":float(qn.min()),
         "bad_cell_count":int(np.sum(dn<=0)),"fixed_boundary_byte_identical_to_017":fixed_equal,
         "all_nodes_outside_free_set_byte_identical_to_017":nonfree_equal,
         "max_adjustment_from_017_mm":float(np.linalg.norm(Xnew64[free]-Xref[free],axis=1).max()),
         "max_abs_coordinate_adjustment_mm":float(np.abs(u).max()),
         "max_displacement_from_013_mm":float(np.linalg.norm(Xnew64-Xbase,axis=1).max()*1000),
         "signed_total_volume_delta_m3":float(np.sum(dn-dold)/6.0),
         "positive_f32_qfloor_pass":bool(np.all(dn>0)&np.all(qn>=1e-5)),
         "fixed_boundary_pass":bool(fixed_equal and nonfree_equal)},
 "limits":["No O/I/D role maps or attachment/bank/stitch node masks exist in these inputs; this is a generic volume diagnostic, not a layer refit.",
           "Parent volume fractions are not recomputed; do not reuse them as conservative material transfer.",
           "No contact, collision, detailed-source transfer, mass, or native admission claim."]}
report["output"]={"candidate_npz":str(out),"candidate_sha256":sha(out)}
(OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"status":status,"iterations":len(trace),"final_merit":trace[-1].get("after_merit") if trace else None,
 "qfloor_pass":report["post"]["positive_f32_qfloor_pass"],"f32_nonpositive":report["post"]["F32_nonpositive"],
 "f32_min_q":report["post"]["F32_min_q"],"max_adjustment_from_017_mm":report["post"]["max_adjustment_from_017_mm"],
 "output":report["output"],"elapsed_s":report["optimization"]["elapsed_s"]},indent=2),flush=True)

