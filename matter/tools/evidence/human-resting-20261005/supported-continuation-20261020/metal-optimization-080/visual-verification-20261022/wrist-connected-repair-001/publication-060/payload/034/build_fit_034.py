#!/usr/bin/env python3
from pathlib import Path
import json,csv,hashlib,time,os
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
OUT=E/"bone-clearance-fit-034"; OUT.mkdir(exist_ok=True)
AP=E/"bone-clearance-feasibility-033/world-bone-clearance-constraints.npz"
PP=E/"native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003/candidate.npz"
POSE=E/"native-candidate020-bone-witness-029/source-preflight-002/registered-skin-poses.json"
DONOR=E/"wholebody-attachment-source-rebind-037-candidate020-identity004-002/generated/donor-frames-fullweights.csv"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=np.load(AP); p=np.load(PP)
X0=np.asarray(a["points"],np.float32).astype(np.float64)
T=np.asarray(a["tetrahedra"],np.int64); mat=np.asarray(a["material_ids"],np.int64)
assert X0.shape==(7233,3) and T.shape==(28632,4) and np.array_equal(T,p["tetrahedra"]) and np.array_equal(mat,p["material_ids"])
assert len(a["constrained_nodes"])==138 and len(np.unique(a["constrained_nodes"]))==138
hnode=np.asarray(a["constrained_nodes"],np.int64); hcol=np.asarray(a["columns"],np.int64); hlay=np.asarray(a["layers"],np.int64)
assert np.array_equal(hnode//2411,hlay) and np.array_equal(hnode%2411,hcol)
OF=np.asarray(p["O_faces"],np.int64)
# One-ring I/D support, but only the 12 confirmed O witnesses can move.
ring=set(map(int,hcol))
hset=set(ring)
for f in OF:
    if any(int(v) in hset for v in f): ring.update(map(int,f))
ring=np.asarray(sorted(ring),np.int64)
assert len(ring)==415
ovars=np.unique(hnode[hlay==0]); ivars=2411+ring; dvars=4822+ring
vn=np.unique(np.r_[ovars,ivars,dvars]).astype(np.int64); N=len(vn); nd=3*N
assert N==842
vm=np.full(len(X0),-1,np.int64); vm[vn]=np.arange(N)
Ttouch=np.flatnonzero(np.isin(T,vn).any(axis=1)); Tseed=np.flatnonzero(np.isin(T,hnode).any(axis=1))
assert len(Ttouch)==6959 and len(Tseed)==1306
Tq=T[Ttouch]; tv=vm[Tq]
Xmm=X0*1000; x0=Xmm[vn].copy()
layer=vn//2411; col=vn%2411
penalty=np.choose(layer,[4.0,2.0,1.0])
pairs=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],np.int64)
qfloor=1.25e-5
def qjac(x, rows, jac=True):
    xx=Xmm.copy();xx[vn]=x.reshape(N,3)
    V=xx[Tq[rows]]
    u=V[:,1]-V[:,0];v=V[:,2]-V[:,0];w=V[:,3]-V[:,0]
    det=np.einsum("ij,ij->i",u,np.cross(v,w))
    gd1=np.cross(v,w);gd2=np.cross(w,u);gd3=np.cross(u,v);gd0=-(gd1+gd2+gd3)
    gd=np.stack([gd0,gd1,gd2,gd3],axis=1)
    ev=np.stack([V[:,i]-V[:,j] for i,j in pairs],axis=1)
    e2=np.einsum("ijk,ijk->ij",ev,ev); em=np.argmax(e2,axis=1); L=np.sqrt(np.maximum(e2[np.arange(len(rows)),em],1e-30))
    q=det/L**3
    if not jac:return q
    G=gd/L[:,None,None]**3
    ep=V[np.arange(len(rows)),pairs[em,0]]-V[np.arange(len(rows)),pairs[em,1]]
    cc=(-3*det/L**5)[:,None]
    np.add.at(G,(np.arange(len(rows)),pairs[em,0]),cc*ep)
    np.add.at(G,(np.arange(len(rows)),pairs[em,1]),-cc*ep)
    J=np.zeros((len(rows),nd),float); rr=np.arange(len(rows))
    for k in range(4):
        ix=tv[rows,k];ok=ix>=0
        if np.any(ok):J[rr[ok,None],3*ix[ok,None]+np.arange(3)[None,:]]+=G[ok,k,:]
    return q,J
# Layer ordering in each touched column, projected along its baseline layer vector.
O0=X0[:2411];I0=X0[2411:4822];D0=X0[4822:]
sv=O0[ring]-I0[ring];fv=I0[ring]-D0[ring]
sl=np.linalg.norm(sv,axis=1);fl=np.linalg.norm(fv,axis=1)
assert np.all(sl>1e-5)&np.all(fl>1e-5)
sn=sv/sl[:,None];fn=fv/fl[:,None]
smin=np.minimum(.2*sl,.10);fmin=np.minimum(.2*fl,.10)
hw=vm[hnode]; closest=np.asarray(a["closest_world"],float)*1000
normal=np.asarray(a["outward_normals"],float); margin=float(a["inferred_margin_m"])*1000
assert np.all(hw>=0)
# Weak displacement regularization across the surface one-ring; fixed outside rows anchor the patch.
edge_set=set()
for f in OF:
    for u,v in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
        for off in (0,2411,4822):
            x,y=off+int(u),off+int(v)
            ix,iy=int(vm[x]),int(vm[y])
            if ix>=0 or iy>=0:edge_set.add((ix,iy))
edges=np.asarray(sorted(edge_set),np.int64)
def obj(x):
    dx=x.reshape(N,3)-x0
    val=np.sum(penalty[:,None]*dx*dx)
    if len(edges):
        ia,ib=edges.T; da=np.zeros((len(edges),3));db=np.zeros_like(da)
        ma=ia>=0;mb=ib>=0;da[ma]=dx[ia[ma]];db[mb]=dx[ib[mb]]
        val+=.08*np.sum((da-db)**2)
    return float(val)
def objjac(x):
    dx=x.reshape(N,3)-x0;g=2*penalty[:,None]*dx
    if len(edges):
        ia,ib=edges.T;da=np.zeros((len(edges),3));db=np.zeros_like(da)
        ma=ia>=0;mb=ib>=0;da[ma]=dx[ia[ma]];db[mb]=dx[ib[mb]]
        z=.16*(da-db)
        np.add.at(g,ia[ma],z[ma]);np.add.at(g,ib[mb],-z[mb])
    return g.ravel()
Jplane=np.zeros((len(hnode),nd))
for k,ix in enumerate(hw):Jplane[k,3*ix:3*ix+3]=normal[k]
Jlay=np.zeros((2*len(ring),nd))
for k,c in enumerate(ring):
    oi,ii,di=int(vm[c]),int(vm[2411+c]),int(vm[4822+c])
    if oi>=0:Jlay[k,3*oi:3*oi+3]+=sn[k]
    Jlay[k,3*ii:3*ii+3]-=sn[k]
    Jlay[len(ring)+k,3*ii:3*ii+3]+=fn[k]
    Jlay[len(ring)+k,3*di:3*di+3]-=fn[k]
def full(x):
    xx=Xmm.copy();xx[vn]=x.reshape(N,3);return xx
def linfun(x):
    xx=full(x)
    return np.r_[np.einsum("ij,ij->i",xx[hnode]-closest,normal)-margin,
                 np.einsum("ij,ij->i",xx[ring]-xx[2411+ring],sn)-smin,
                 np.einsum("ij,ij->i",xx[2411+ring]-xx[4822+ring],fn)-fmin]
def con(active):
    active=np.asarray(active,np.int64);cache={"x":None,"q":None,"J":None,"l":None}
    def get(x,j):
        if cache["x"] is None or not np.array_equal(x,cache["x"]):
            cache["x"]=x.copy();cache["q"],cache["J"]=qjac(x,active,True);cache["l"]=linfun(x)
        return np.vstack((cache["J"]*1e5,np.vstack((Jplane,Jlay)))) if j else np.r_[cache["q"]*1e5-qfloor*1e5,cache["l"]]
    return {"type":"ineq","fun":lambda x:get(x,False),"jac":lambda x:get(x,True)}
def metrics(X):
    z=np.asarray(X,np.float32).astype(np.float64);v=z[T]
    det=np.linalg.det(v[:,1:]-v[:,:1])
    el=np.stack([np.linalg.norm(v[:,i]-v[:,j],axis=1) for i,j in pairs],axis=1)
    q=det/np.maximum(el.max(axis=1)**3,1e-30)
    vol={str(int(k)):float(det[mat==k].sum()/6) for k in np.unique(mat)}
    return {"nonpositive_tets":int(np.sum(det<=0)),"below_q1e5":int(np.sum(q<1e-5)),"min_det_m3":float(det.min()),"min_normalized_q":float(q.min()),"q_p01":float(np.percentile(q,1)),"material_volume_m3":vol},q,z
def Xfromx(x):return full(x)/1000
# Preflight exact rows, base quality, donor map, and analytic Jacobian.
qbase=qjac(x0.ravel(),np.arange(len(Ttouch)),False)
base_report,_,_=metrics(X0)
assert base_report["nonpositive_tets"]==0 and base_report["below_q1e5"]==0
rows=list(csv.DictReader(open(DONOR)));assert len(rows)==2411 and all(int(r["row"])==i for i,r in enumerate(rows))
donor_by_col=np.array([int(r["donor"]) for r in rows],np.int64)
poses={int(r["body_index"]):r["current"] for r in json.load(open(POSE))["all_bodies"]}
assert all(int(b) in poses for b in donor_by_col[ring])
# Analytic q derivative sanity on three seed cells and active coordinates.
test=np.unique(Tseed[:3]); local=np.searchsorted(Ttouch,test); qv,J=qjac(x0.ravel(),local,True)
for d in [0,nd//2,nd-1]:
    e=np.zeros(nd);e[d]=1e-4
    numeric=(qjac(x0.ravel()+e,local,False)-qjac(x0.ravel()-e,local,False))/(2e-4)
    assert np.max(np.abs(numeric-J[:,d]))<2e-5
pre={"status":"pass","variables":N,"dofs":nd,"one_ring_columns":len(ring),"witnesses":len(hnode),"witness_layers":{str(int(k)):int(np.sum(hlay==k)) for k in np.unique(hlay)},"directly_touched_tets":len(Tseed),"all_affected_tets":len(Ttouch),"affected_baseline_min_q":float(qbase.min()),"base_allcell_metrics":base_report,"input_sha256":{"constraint_npz":sha(AP),"physical_topology_npz":sha(PP),"pose_json":sha(POSE),"donor_frames":sha(DONOR)}}
(OUT/"preflight.json").write_text(json.dumps(pre,indent=2)+"\n")
print("PREFLIGHT",json.dumps(pre),flush=True)
if os.environ.get("FIT_PREFLIGHT_ONLY")=="1":raise SystemExit(0)
# Minimal projected push is retained as a comparison only, never the candidate.
direct=np.zeros((len(hnode),3))
for k,i in enumerate(hnode):direct[k]=closest[k]+margin*normal[k]-Xmm[i]
xd=x0.copy();xd[hw]=Xmm[hnode]+direct
qd=qjac(xd.ravel(),np.arange(len(Ttouch)),False)
direct_diag={"max_node_delta_mm":float(np.linalg.norm(direct,axis=1).max()),"incident_nonpositive":int(np.sum(qd<=0)),"incident_below1e5":int(np.sum(qd<1e-5))}
print("DIRECT_DIAGNOSTIC",json.dumps(direct_diag),flush=True)
active=np.unique(np.searchsorted(Ttouch,Tseed))
xcur=x0.ravel().copy(); start=time.monotonic();budget=float(os.environ.get("FIT_BUDGET_S","105"));count=0;last=xcur.copy()
class StopBudget(Exception):pass
def cb(x):
    global count,last
    count+=1;last=x.copy()
    if count%2==0:
        np.savez_compressed(OUT/"solver-checkpoint.npz",x_mm=x,variable_local_rows=vn,active_tet_rows=Ttouch[active])
        (OUT/"solver-progress.json").write_text(json.dumps({"iteration":count,"elapsed_s":time.monotonic()-start,"objective":obj(x),"active_quality_constraints":int(len(active))})+"\n")
    if time.monotonic()-start>budget:raise StopBudget()
solve=[]
for attempt in range(3):
    bounds=[(z-15.,z+15.) for z in x0.ravel()]
    try:
        r=minimize(obj,xcur,jac=objjac,method="SLSQP",bounds=bounds,constraints=[con(active)],callback=cb,options={"maxiter":80,"ftol":2e-7,"disp":False})
        xcur=r.x.copy(); sr={"success":bool(r.success),"status":int(r.status),"message":str(r.message),"iterations":int(r.nit),"objective":float(r.fun)}
    except StopBudget:
        xcur=last.copy();sr={"success":False,"status":-2,"message":"bounded wall-clock stop; retained last callback iterate","iterations":count,"objective":obj(xcur)}
    qa=qjac(xcur,np.arange(len(Ttouch)),False); bad=np.flatnonzero(qa<qfloor*1.02);new=np.setdiff1d(bad,active)
    ll=linfun(xcur);solve.append({"attempt":attempt,"solver":sr,"active_quality_rows":len(active),"patch_quality_below_margin":len(bad),"added_rows":len(new),"min_bone_or_layer_constraint":float(ll.min()),"min_layer_order":float(ll[len(hnode):].min())})
    print("SOLVE",json.dumps(solve[-1]),flush=True)
    if sr["status"]==-2 or not len(new) or time.monotonic()-start>budget:break
    active=np.unique(np.r_[active,new])
Xcand=Xfromx(xcur); mm,qfinal,Xcand=metrics(Xcand)
base,_,_=metrics(X0)
gaps=np.einsum("ij,ij->i",Xcand[hnode]-np.asarray(a["closest_world"]),normal)
plane_ok=bool(np.all(gaps>=float(a["inferred_margin_m"])-2e-7))
of=Xcand[:2411];ii=Xcand[2411:4822];dd=Xcand[4822:]
sproj=np.einsum("ij,ij->i",of[ring]-ii[ring],sn);fproj=np.einsum("ij,ij->i",ii[ring]-dd[ring],fn)
layer_ok=bool(np.all(sproj>0)&np.all(fproj>0))
vol0=base["material_volume_m3"];vol1=mm["material_volume_m3"]
base_mass={1:3.4525707761420956,2:10.487108964465744}
rho={int(k):base_mass[int(k)]/float(vol0[k]) for k in vol0}
dmass={str(int(k)):(vol1[k]-vol0[k])*rho[int(k)] for k in vol1}
delta=Xcand-X0;moved=np.flatnonzero(np.linalg.norm(delta,axis=1)>0)
# Per-node donor-local vectors use row-aligned D37 donor IDs and the exact current donor pose rotations.
donor=np.asarray([int(rows[int(i%2411)]["donor"]) for i in moved],np.int64)
local=[]
for i,bid,dw in zip(moved,donor,delta[moved]):
    R=Rotation.from_quat(poses[int(bid)]["orientation_world_xyzw"]).as_matrix()
    local.append(R.T@dw)
local=np.asarray(local,float)
report={"schema":"numi.human.d37-bone-clearance-joint-layer-fit.v1","status":"candidate_only_not_admitted","input_sha256":{"constraint_npz":sha(AP),"physical_topology_npz":sha(PP),"pose_json":sha(POSE),"donor_frames":sha(DONOR)},"fit":{"method":"joint 3-D world-coordinate SLSQP; analytic normalized tetra quality Jacobian; active-set recheck over all touched cells","fixed":"all nodes outside 12 confirmed O witnesses and 415 connected I/D columns","variable_nodes":N,"dofs":nd,"witnesses":len(hnode),"witness_layers":pre["witness_layers"],"columns_in_one_ring":len(ring),"affected_tets":len(Ttouch),"seed_tets":len(Tseed),"q_barrier":qfloor,"bone_margin_m":float(a["inferred_margin_m"]),"coordinate_bounds_mm":15.0,"objective":"weighted squared displacement (O=4,I=2,D=1) plus weak one-ring edge displacement regularizer","solve_attempts":solve,"elapsed_s":time.monotonic()-start,"direct_projection_diagnostic":direct_diag},"checks":{"baseline_float32":base,"candidate_float32_allcell":mm,"bone_plane_min_gap_m":float(gaps.min()),"bone_plane_violations":int(np.sum(gaps<float(a["inferred_margin_m"])-2e-7)),"bone_plane_pass":plane_ok,"minimum_skin_order_projection_m":float(sproj.min()),"minimum_fat_order_projection_m":float(fproj.min()),"layer_order_pass":layer_ok,"allcell_positive_q1e5_pass":bool(mm["nonpositive_tets"]==0 and mm["below_q1e5"]==0)},"geometry":{"moved_nodes":len(moved),"moved_layer_counts":{str(k):int(np.sum(moved//2411==k)) for k in range(3)},"max_displacement_m":float(np.linalg.norm(delta,axis=1).max()),"max_witness_displacement_m":float(np.linalg.norm(delta[hnode],axis=1).max()),"rms_moved_displacement_m":float(np.sqrt(np.mean(np.sum(delta[moved]**2,axis=1))) if len(moved) else 0),"plane_gap_p01_m":float(np.percentile(gaps,1)),"plane_gap_median_m":float(np.median(gaps)),"donor_local_vectors":"computed as current-pose rotation transpose times world displacement using D37 2411-row donor table"},"mass":{"material_volume_baseline_m3":vol0,"material_volume_candidate_m3":vol1,"volume_delta_m3":{k:vol1[k]-vol0[k] for k in vol1},"reference_fixed_density_kg_m3":{str(k):rho[k] for k in rho},"fixed_density_mass_delta_diagnostic_kg":dmass,"conservative_mass_owner_updated":False,"density_rescaled":False},"limits":["Bone planes are necessary witness constraints only; they do not replace a fresh registered-solid BVH containment audit.","No external self/retained-surface audit was rerun.","Detailed-source embedding and physical O ancestry are not regenerated.","Donor-local displacement is provided for moved rows, but the native owner/topology and conservative material map are not updated.","This candidate is unadmitted and does not modify frozen 030 evidence."]}
np.savez_compressed(OUT/"candidate.npz",points_world_f32=Xcand.astype(np.float32),tetrahedra=T.astype(np.int32),material_ids=mat.astype(np.int8),global_node_ids=np.asarray(a["global_nodes"]),variable_local_rows=vn,variable_layer=layer,variable_column=col,witness_local_rows=hnode,witness_global_node_ids=np.asarray(a["global_nodes"])[hnode],witness_layer=hlay,witness_column=hcol,witness_body_index=np.asarray(a["bodies"]),witness_inside_depth_m=np.asarray(a["inside_depth_m"]),witness_closest_world_m=np.asarray(a["closest_world"]),witness_outward_normal=np.asarray(a["outward_normals"]),witness_final_plane_gap_m=gaps,donor_body_by_column=donor_by_col,changed_local_rows=moved,changed_world_delta_m=delta[moved],changed_donor_body_ids=donor,changed_donor_local_delta_m=local,skin_faces=OF.astype(np.int32),inner_faces=p["I_faces"],deep_faces=p["D_faces"])
(OUT/"fit-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print("FINAL",json.dumps({"path":str(OUT/"candidate.npz"),"report":str(OUT/"fit-report.json"),"fit":report["fit"],"checks":report["checks"],"geometry":report["geometry"],"mass":report["mass"]},indent=2),flush=True)

