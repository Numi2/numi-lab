from pathlib import Path
import hashlib, json, time, itertools, sys, os
import numpy as np
from scipy.optimize import minimize
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
OUT=E/"o12-outward-repair-021"
IP=E/"baseline-depth-transition-014/candidate.npz"
RP=E/"depth014-exact-material-routes-015/route-report.json"
P18=E/"surface-normal-bone-exits-018/outward-exit-planes.npz"
P17=E/"outward-skin-bone-exits-017/outward-exit-planes.npz"
HELP=E/"bone-clearance-fit-040-shared-columns/column_fit_support.py"
for p in (IP,RP,P18,P17,HELP): assert p.is_file(), p
sys.path.insert(0,str(HELP.parent))
import column_fit_support as cs
N=2411; MARGIN=1.0e-4; QMIN=1.0e-5
z=np.load(IP,allow_pickle=False); X=z["points_world_f32"].astype(np.float64); T=z["tetrahedra"].astype(np.int64); mat=z["material_ids"]
assert X.shape==(3*N,3) and T.shape==(28632,4) and np.all(T>=0) and T.max()<3*N
p18=np.load(P18,allow_pickle=False);p17=np.load(P17,allow_pickle=False)
ids=p18["constraint_local_node"].astype(np.int64)
assert ids.tolist()==[204,215,217,218,219,231,233,271,337,2187,2261,2262]
H=[]; normals=[]
for c in ids:
    src=p17 if int(c)==271 else p18
    j=int(np.flatnonzero(src["constraint_local_node"]==c)[0])
    H.append(src["constraint_closest_world_m"][j]); normals.append(src["constraint_outward_normal_world"][j])
H=np.asarray(H,float); normals=np.asarray(normals,float)
assert np.allclose(np.linalg.norm(normals,axis=1),1,atol=2e-8)
route=json.loads(RP.read_text())
route_cols=sorted({int(w["row"]) for w in route["route_witnesses"] if w["segment"]==0})
assert route_cols==[204,215,217,218,219,231,233,271],route_cols
route_set=set(route_cols)
# Column graph induced by actual three-layer tetrahedra.
adj=[set() for _ in range(N)]
for tet in T:
    c=np.unique(tet%N)
    for ii in range(len(c)):
        for jj in range(ii+1,len(c)):
            a,b=int(c[ii]),int(c[jj]);adj[a].add(b);adj[b].add(a)
seeds=set(map(int,ids));dist=np.full(N,999,np.int32);front=list(seeds)
for c in front:dist[c]=0
for d in range(1,4):
    nxt=set()
    for c in front:
        for q in adj[c]:
            if dist[q]==999:nxt.add(q)
    for q in nxt:dist[q]=d
    front=nxt
free=np.flatnonzero(dist<=1).astype(np.int64); collar=np.flatnonzero(dist==2).astype(np.int64)
assert all(dist[c]<=1 for c in seeds)
idx=np.full(N,-1,np.int64);idx[free]=np.arange(len(free))
seedlist=np.asarray(sorted(seeds),np.int64);sidx=np.full(N,-1,np.int64);sidx[seedlist]=np.arange(len(seedlist))
# Include every cell touching a movable column, so all nonlinear quality consequences are constrained.
rows=np.flatnonzero(np.isin(T%N,free).any(axis=1)).astype(np.int64);TT=T[rows]
assert len(rows)>0
# Local graph regularizer includes fixed collar edges.
edge_set=set()
for c in free:
    for q in adj[int(c)]:
        if dist[q]<=2: edge_set.add(tuple(sorted((int(c),int(q)))))
edges=np.asarray(sorted(edge_set),np.int64)
edge_i=idx[edges[:,0]]; edge_j=idx[edges[:,1]]
# Layer vectors from the frozen positive 014 source.
O=X[:N]; I=X[N:2*N]; D=X[2*N:]
vs=I-O;vf=D-I
ls=np.linalg.norm(vs[seedlist],axis=1);lf=np.linalg.norm(vf[seedlist],axis=1)
assert np.all(ls>0) and np.all(lf>0)
# Variables are d[free,xyz] in mm, then skin/fat shorten lengths in mm on the 12 seed columns.
nd=3*len(free); na=len(seedlist); nb=len(seedlist); nv=nd+na+nb

def unpack(x): return x[:nd].reshape(-1,3),x[nd:nd+na],x[nd+na:]

def emit64(x):
    d,a,b=unpack(x); P=X.copy()
    delta=np.zeros((N,3),float);delta[free]=d*1e-3
    P[:N]=O+delta
    P[N:2*N]=I+delta
    P[2*N:]=D+delta
    # Interpolate original per-column layer vectors, shortening each target skin/fat interval without changing source IDs.
    P[N+seedlist] -= (a*1e-3/ls)[:,None]*vs[seedlist]
    P[2*N+seedlist] -= (a*1e-3/ls)[:,None]*vs[seedlist]
    P[2*N+seedlist] -= (b*1e-3/lf)[:,None]*vf[seedlist]
    return P

def obj(x):
    d,a,b=unpack(x)
    val=float(np.sum(d*d)+0.5*np.sum(a*a+b*b))
    gd=2*d.copy()
    left=d[edge_i]
    ok=edge_j>=0
    right=np.zeros_like(left)
    right[ok]=d[edge_j[ok]]
    diff=left-right
    gamma=3.0
    val+=gamma*float(np.sum(diff*diff))
    np.add.at(gd,edge_i,2*gamma*diff)
    np.add.at(gd,edge_j[ok],-2*gamma*diff[ok])
    return val

def obj_jac(x):
    d,a,b=unpack(x);gd=2*d.copy()
    left=d[edge_i];ok=edge_j>=0;right=np.zeros_like(left)
    right[ok]=d[edge_j[ok]]
    diff=left-right;gamma=3.0
    np.add.at(gd,edge_i,2*gamma*diff);np.add.at(gd,edge_j[ok],-2*gamma*diff[ok])
    return np.r_[gd.ravel(),a,b]

# Fixed endpoint rows: seed O always; I also for the eight measured O-I crossings.
end_specs=[]
for j,c in enumerate(seedlist):
    end_specs.append((j,0))
    if int(c) in route_set:end_specs.append((j,1))
end_seed=np.asarray([s for s,l in end_specs],np.int64)
end_layer=np.asarray([l for s,l in end_specs],np.int64)

def plane_fun(x):
    d,a,b=unpack(x)
    vals=[]; jac=[]
    for sj,layer in zip(end_seed,end_layer):
        c=int(seedlist[sj]); n=normals[sj]; h=H[sj]
        p=O[c]+(d[idx[c]]*1e-3 if idx[c]>=0 else 0)
        # Reconstruct endpoint after common translation and skin shortening.
        if layer==1:p=p+vs[c]-(a[sj]*1e-3/ls[sj])*vs[c]
        vals.append((np.dot(n,p-h)-MARGIN)*1000.0)
        g=np.zeros(nv,float)
        k=int(idx[c])
        if k>=0:g[3*k:3*k+3]=n
        if layer==1:g[nd+sj]=-np.dot(n,vs[c]/ls[sj])
        jac.append(g)
    return np.asarray(vals),np.asarray(jac)

def quality_fun_jac(x):
    P=emit64(x);q,G=cs.normalized_tet_quality_and_gradient(P[TT])
    # Scale q rows for conditioning.
    qv=(q-QMIN)*1e5
    J=np.zeros((len(rows),nv),float)
    cols=TT%N
    for k in range(4):
        c=cols[:,k];l=TT[:,k]//N;g=G[:,k,:]
        ix=idx[c];mask=ix>=0
        for ax in range(3):
            np.add.at(J,(np.flatnonzero(mask),3*ix[mask]+ax),g[mask,ax]*1e-3*1e5)
        ss=sidx[c];mask=(ss>=0)&((l==1)|(l==2))
        if np.any(mask):
            u=vs[c[mask]]/np.linalg.norm(vs[c[mask]],axis=1)[:,None]
            deriv=-np.einsum("ij,ij->i",g[mask],u)*1e-3*1e5
            np.add.at(J,(np.flatnonzero(mask),nd+ss[mask]),deriv)
        mask=(ss>=0)&(l==2)
        if np.any(mask):
            u=vf[c[mask]]/np.linalg.norm(vf[c[mask]],axis=1)[:,None]
            deriv=-np.einsum("ij,ij->i",g[mask],u)*1e-3*1e5
            np.add.at(J,(np.flatnonzero(mask),nd+na+ss[mask]),deriv)
    return qv,J

def qfun(x):return quality_fun_jac(x)[0]
def qjac(x):return quality_fun_jac(x)[1]
def pfun(x):return plane_fun(x)[0]
def pjac(x):return plane_fun(x)[1]

# Directional finite-difference preflight for the exact optimizer callbacks.
rng=np.random.default_rng(21021)
xtest=rng.normal(0.0,0.03,size=nv)
for sj in range(na):
    xtest[nd+sj]=min(float(bounds_value) if False else 0.0,0.0)
# Keep shortening test variables in a small physically local range.
xtest[nd:]=0.01
vtest=rng.normal(size=nv);vtest/=np.linalg.norm(vtest);eps=2e-5
fd_obj=(obj(xtest+eps*vtest)-obj(xtest-eps*vtest))/(2*eps)
ana_obj=float(np.dot(obj_jac(xtest),vtest))
fd_plane=(pfun(xtest+eps*vtest)-pfun(xtest-eps*vtest))/(2*eps)
ana_plane=pjac(xtest)@vtest
fd_q=(qfun(xtest+eps*vtest)-qfun(xtest-eps*vtest))/(2*eps)
ana_q=qjac(xtest)@vtest
checks={"objective_directional_abs_error":abs(float(fd_obj-ana_obj)),
        "plane_directional_max_abs_error":float(np.max(np.abs(fd_plane-ana_plane))),
        "quality_directional_max_abs_error":float(np.max(np.abs(fd_q-ana_q)))}
if checks["objective_directional_abs_error"]>2e-5 or checks["plane_directional_max_abs_error"]>2e-5 or checks["quality_directional_max_abs_error"]>1e-3:
    raise RuntimeError("analytic derivative preflight failed: "+repr(checks))
if os.environ.get("O12_DERIVATIVE_PREFLIGHT_ONLY")=="1":
    print(json.dumps({"derivative_checks":checks,"free_columns":int(len(free)),"affected_tets":int(len(rows))},indent=2))
    raise SystemExit(0)

# Bounds preserve positive layer thickness and keep local movement bounded.
lo_s=np.minimum(.2*ls,.0001);lo_f=np.minimum(.2*lf,.0001)
bounds=[(-8.0,8.0)]*nd+[(0.0,max(0.0,float(ls[i]*1000-lo_s[i]*1000))) for i in range(na)]+[(0.0,max(0.0,float(lf[i]*1000-lo_f[i]*1000))) for i in range(nb)]
x0=np.zeros(nv,float)
# Independent source-row preflight: record initial plane sides and affected-cell quality.
q0,_=cs.normalized_tet_quality_and_gradient(X[TT]);p0=pfun(x0)
start=time.monotonic();deadline=start+28.0
last_x=x0.copy();iters=0
class Deadline(Exception):pass
def callback(xk):
    global last_x,iters
    last_x=np.array(xk,copy=True);iters+=1
    if time.monotonic()>deadline:raise Deadline()
cons=[{"type":"ineq","fun":qfun,"jac":qjac},{"type":"ineq","fun":pfun,"jac":pjac}]
try:
    res=minimize(obj,x0,jac=obj_jac,method="SLSQP",bounds=bounds,constraints=cons,callback=callback,options={"maxiter":35,"ftol":1e-10,"disp":True})
    x=np.array(res.x,copy=True);term={"success":bool(res.success),"message":str(res.message),"nit":int(res.nit),"fun":float(res.fun)}
except Deadline:
    x=last_x;term={"success":False,"message":"bounded_wallclock_stop","nit":iters,"fun":float(obj(x))}
P64=emit64(x);Y=P64.astype(np.float32);Y64=Y.astype(np.float64)
qa,_=cs.normalized_tet_quality_and_gradient(Y64[TT]);qfull,_=cs.normalized_tet_quality_and_gradient(Y64[T]);ps=pfun(x)
# Recompute F32 plane margins from the actual emitted positions.
plane_slacks=[]
for sj,layer in zip(end_seed,end_layer):
 c=int(seedlist[sj]);node=layer*N+c
 plane_slacks.append(float(np.dot(normals[sj],Y64[node]-H[sj])-MARGIN))
plane_slacks=np.asarray(plane_slacks)
P_OUT=OUT/"candidate.npz"
np.savez_compressed(P_OUT,points_world_f32=Y,tetrahedra=T.astype(np.int32),material_ids=mat)
inputs=[IP,RP,P18,P17,HELP]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={"schema":"numi.human.o12-outward-local-repair.021","status":"GEOMETRY_CONSTRUCTION_DIAGNOSTIC_NOT_ADMITTED","inputs":{str(p):sha(p) for p in inputs},"candidate_sha256":sha(P_OUT),"parameters":{"outer_exit_source":"018 surface-normal exact exits; 017 I-to-O directional exit for column271 due grazing 018 normal","seed_columns":ids.tolist(),"route_hit_columns":route_cols,"constrained_endpoints":[{"column":int(seedlist[s]),"layer":"O" if l==0 else "I"} for s,l in zip(end_seed,end_layer)],"free_column_count":int(len(free)),"fixed_collar_ring":2,"affected_tet_count":int(len(rows)),"q_min":QMIN,"clearance_margin_m":MARGIN,"shared_column_translation_bound_mm":8.0,"objective":"minimum squared local shared-column displacement plus smoothness and bounded skin/fat shortening"},"derivative_checks":checks,"solver":term,"initial":{"min_affected_q":float(q0.min()),"min_all_q":float(cs.normalized_tet_quality_and_gradient(X[T])[0].min()),"plane_slack_mm":(p0).tolist()},"candidate":{"moved_source_rows":int(np.any(Y!=X.astype(np.float32),axis=1).sum()),"max_displacement_mm":float(np.linalg.norm(Y64-X,axis=1).max()*1000),"free_columns_nonzero":int(np.sum(np.linalg.norm(unpack(x)[0],axis=1)>1e-6)),"min_affected_q":float(qa.min()),"min_global_q":float(qfull.min()),"q_below_1e-5_affected":int(np.sum(qa<0)),"plane_slack_m":plane_slacks.tolist(),"plane_violations_tol1e-7m":int(np.sum(plane_slacks < -1e-7)),"skin_shortening_mm":unpack(x)[1].tolist(),"fat_shortening_mm":unpack(x)[2].tolist()},"limitations":["No exact whole-volume or all-bone requery has run on this candidate.","No reference-coordinate pullback, conservative material transfer, density/mass closure, or native admission.","Plane rows are construction constraints from exact emitted bone exits; only the existing eight O-I route hits are constrained at both O and I endpoints.","No existing source IDs, face topology, or material rows changed."]}
(OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"derivative_checks":checks,"solver":term,"free_columns":int(len(free)),"affected_tets":int(len(rows)),"min_q_affected":float(qa.min()),"min_q_global":float(qfull.min()),"plane_violations":int(np.sum(plane_slacks < -1e-7)),"max_move_mm":float(np.linalg.norm(Y64-X,axis=1).max()*1000),"candidate":str(P_OUT),"sha256":report["candidate_sha256"],"report":str(OUT/"report.json")},indent=2))
