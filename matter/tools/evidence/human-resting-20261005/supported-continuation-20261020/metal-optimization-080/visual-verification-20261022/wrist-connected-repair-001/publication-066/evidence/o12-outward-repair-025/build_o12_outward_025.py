from pathlib import Path
import csv, hashlib, json, os, sys, time
import numpy as np
from scipy.optimize import minimize

E = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
OUT = E / "o12-outward-repair-025"
if OUT.exists() and any(p.name != Path(__file__).name for p in OUT.iterdir()):
    raise SystemExit(f"refusing to overwrite existing outputs in directory: {OUT}")
OUT.mkdir(parents=True, exist_ok=True)
IP = E / "baseline-depth-transition-014/candidate.npz"
PREV = E / "o12-outward-repair-022/candidate.npz"
ROUTE = E / "route-audit-consolidation-019/run-022-matched-query-023-corrected-001/route-report.json"
EXIT18 = E / "surface-normal-bone-exits-018/outward-exit-planes.npz"
EXIT17 = E / "outward-skin-bone-exits-017/outward-exit-planes.npz"
QUERY = E / "native-source-only-rest-geometry-022-query-023/source-preflight/all-bank-containment.csv"
BODIES = E / "native-source-only-rest-geometry-022-query-023/source-preflight/exact-emitted.bodies.csv"
NODEMAP = E / "native-source-only-rest-geometry-022-query-023/source-preflight/exact-emitted.nodes.csv"
HELP = E / "bone-clearance-fit-040-shared-columns/column_fit_support.py"
EXPECTED = {
    IP: "f5b0e33933ddf0867715a877ade75279659034362ae8f586435ccbf00609f42c",
    PREV: "d02bccfbba026edb1a564c9a477d0d31deb704cd96b66091764e21ca32e7452a",
    ROUTE: "2f4af36c838ecb2060db94ef7b1bc2b41b0ce9c4fc21e3e7597408c8eace73c7",
    EXIT18: "1d54b19e89f17b32cfc3b325c90650bbc2faf41788151dfd2301682d41d48ebe",
    EXIT17: "3737e93d9d81c6ebac9a8116ee7e0addfe67553e438194cb006d575651b25312",
    QUERY: "17825aed54a12442dc08826e8a42616a3bff2844961d95f0bbcef7c8d2f2ba7a",
    BODIES: "dae52ab190e02586421b56381b03a4453aff1058edca7b6a42afcb9051b50b90",
    NODEMAP: "f82c65e1d8b839d86505ee2095e97e56790812d155f73fa83aed114a121e55cb",
    HELP: "2de0857455c25dc04bf28c83429ba751ced58439a03bfb30494e510c1a64bd16",
}
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
for p,want in EXPECTED.items():
    if not p.is_file(): raise FileNotFoundError(p)
    got=sha(p)
    if got!=want: raise RuntimeError(f"pinned input hash mismatch {p}: {got} != {want}")
sys.path.insert(0,str(HELP.parent))
import column_fit_support as cs

N=2411
QMIN=1.0e-5
Q_SCALE=1.0e3
MARGIN=1.0e-4
z=np.load(IP,allow_pickle=False)
X=z["points_world_f32"].astype(np.float64)
T=z["tetrahedra"].astype(np.int64)
mat=z["material_ids"]
assert X.shape==(3*N,3) and T.shape==(28632,4) and T.min()>=0 and T.max()<3*N
zp=np.load(PREV,allow_pickle=False)
Prev=zp["points_world_f32"].astype(np.float64)
assert np.array_equal(zp["tetrahedra"],T.astype(np.int32))
assert np.array_equal(zp["material_ids"],mat)
route=json.loads(ROUTE.read_text())
assert len(route["route_witnesses"])==16

# Build the immutable direct-exit seed planes from native-validated 017/018 receipts.
p18=np.load(EXIT18,allow_pickle=False); p17=np.load(EXIT17,allow_pickle=False)
seed_cols=p18["constraint_local_node"].astype(np.int64)
assert seed_cols.tolist()==[204,215,217,218,219,231,233,271,337,2187,2261,2262]
exit_rows=[]
for c in seed_cols:
    src=p17 if int(c)==271 else p18
    j=int(np.flatnonzero(src["constraint_local_node"]==c)[0])
    h=np.asarray(src["constraint_closest_world_m"][j],np.float64)
    n=np.asarray(src["constraint_outward_normal_world"][j],np.float64)
    assert abs(np.linalg.norm(n)-1.0)<2e-8
    for layer in (0,1,2):
        exit_rows.append({"node":int(layer*N+c),"column":int(c),"layer":layer,"h":h,"n":n,
                          "kind":"seed_outward_exit","bank_stable_id":int(src["constraint_bone_stable_id"][j]),
                          "body_index":int(src["constraint_body_index"][j]),"triangle_stable_id":None})

# Route first-hit constraints are oriented toward the endpoint classified outside.
route_rows=[]
route_cols=set()
for wi,w in enumerate(route["route_witnesses"]):
    if not w.get("is_first_route_hit",False) or w.get("first_hit_ambiguous",True):
        raise RuntimeError(f"route witness {wi} is not a unique exact first hit")
    if w.get("exact_first_hit",{}).get("classification")!="proper_interior_hit":
        raise RuntimeError(f"route witness {wi} is not a proper interior hit")
    row=int(w["row"]); seg=str(w["segment_name"])
    if seg=="O-I": layers=(0,1)
    elif seg=="I-D": layers=(1,2)
    else: raise RuntimeError(f"unknown segment {seg}")
    start_class=str(w["start_classification"]); end_class=str(w["end_classification"])
    n=np.asarray(w["unit_exterior_normal_world"],np.float64)
    h=np.asarray(w["plane_point_world_m"],np.float64)
    if start_class=="outside": side=np.asarray(w["segment_start_world_f32_m"],np.float64)
    elif end_class=="outside": side=np.asarray(w["segment_end_world_f32_m"],np.float64)
    else: raise RuntimeError(f"route witness {wi} has no outside endpoint classification")
    if float(np.dot(n,side-h))<0.0: n=-n
    if float(np.dot(n,side-h))<=0.0: raise RuntimeError(f"route witness {wi} exterior normal not oriented toward outside endpoint")
    route_cols.add(row)
    for layer in layers:
        route_rows.append({"node":int(layer*N+row),"column":row,"layer":layer,"h":h,"n":n,
                           "kind":"exact_first_route_hit","route_witness":wi,"segment":seg,
                           "body_index":int(w["body"]),"bank_stable_id":int(w["bone_stable_id"]),
                           "triangle_stable_id":int(w["triangle_stable_id"]),"outside_endpoint":("start" if start_class=="outside" else "end")})
assert len(route_rows)==32

# Transform the 023 inside-node nearest exits with the native-emitted raw body affine.
with BODIES.open(newline="") as f: body_rows={int(r["body_index"]):r for r in csv.DictReader(f)}
with NODEMAP.open(newline="") as f: node_rows={int(r["local_node"]):r for r in csv.DictReader(f)}
assert len(node_rows)==3*N and set(node_rows)==set(range(3*N))
inside_rows=[]
with QUERY.open(newline="") as f:
    for q in csv.DictReader(f):
        if q["classification"]!="inside": continue
        if q["ambiguous"]!="0": raise RuntimeError("inside query marked ambiguous")
        c=int(q["row"]); layer=int(q["layer"]); body=int(q["bank_body"])
        if not 0<=c<N or layer not in (0,1,2): raise RuntimeError(f"bad inside node identity {q}")
        gidx=layer*N+c
        nr=node_rows[gidx]
        if int(nr["global_node"])!=int(q["global_node"]) or int(nr["layer"])!=layer or int(nr["row"])!=c:
            raise RuntimeError(f"023 AP global/local node mapping mismatch at local {gidx}: query={q}, map={nr}")
        mapped_world=np.asarray([float(nr[f"{a}_m"]) for a in "xyz"],np.float64)
        if not np.array_equal(mapped_world.astype(np.float32),Prev[gidx].astype(np.float32)):
            raise RuntimeError(f"023 exact-emitted node map does not match 022 candidate local row {gidx}")
        br=body_rows[body]
        M=np.asarray([[float(br[f"world_to_body_m{i}{j}"]) for j in range(3)] for i in range(3)],np.float64)
        pos=np.asarray([float(br[f"position_{a}_m"]) for a in "xyz"],np.float64)
        off=np.asarray([float(br[f"com_offset_{a}_m"]) for a in "xyz"],np.float64)
        qlocal=np.asarray([float(q[f"query_local_{a}"]) for a in "xyz"],np.float64)
        clocal=np.asarray([float(q[f"closest_{a}"]) for a in "xyz"],np.float64)
        qworld=pos+np.linalg.solve(M,qlocal+off)
        h=pos+np.linalg.solve(M,clocal+off)
        delta=np.linalg.solve(M,clocal-qlocal)
        dn=float(np.linalg.norm(delta))
        if not dn>0: raise RuntimeError(f"zero inside-exit vector {q}")
        n=delta/dn
        actual=Prev[gidx]
        err=float(np.linalg.norm(qworld-actual))
        if err>2e-7: raise RuntimeError(f"023 inside point does not match 022 candidate at node {layer*N+c}: {err}m")
        if float(np.dot(n,h-qworld))<=0: raise RuntimeError(f"inside-exit normal points inward for node {layer*N+c}")
        inside_rows.append({"node":gidx,"column":c,"layer":layer,"h":h,"n":n,
                            "kind":"023_native_inside_nearest_exit","body_index":body,
                            "bank_stable_id":int(q["bank_stable_id"]),"triangle_stable_id":int(q["nearest_triangle"]),
                            "global_node":int(q["global_node"]),"query_world_match_error_m":err,
                            "inside_distance_m":float(q["nearest_distance_m"]),"signed_face_distance_m":float(q["signed_face_distance_m"])})
assert len(inside_rows)==17

# Retire only the backside nearest-exit construction plane uniquely shown infeasible by the 024 phase-I LP.
inside_query_rows_input=len(inside_rows)
dropped_conflict_rows=[r for r in inside_rows if r["node"]==5039 and r["body_index"]==61 and r["bank_stable_id"]==97 and r["triangle_stable_id"]==172]
if len(dropped_conflict_rows)!=1: raise RuntimeError(f"expected only the LP-identified D217 backside nearest-exit row, got {len(dropped_conflict_rows)}")
inside_rows=[r for r in inside_rows if r not in dropped_conflict_rows]
plane_rows=exit_rows+route_rows+inside_rows
plane_nodes=np.asarray([r["node"] for r in plane_rows],np.int64)
plane_h=np.asarray([r["h"] for r in plane_rows],np.float64)
plane_n=np.asarray([r["n"] for r in plane_rows],np.float64)
assert plane_h.shape==plane_n.shape==(len(plane_rows),3)
route_cols=sorted(route_cols)
inside_cols=sorted({int(r["column"]) for r in inside_rows})
shape_cols=np.asarray(sorted(set(map(int,seed_cols))|set(route_cols)|set(inside_cols)),np.int64)
target_cols=set(map(int,shape_cols))

# Build a graph on existing tetrahedral columns to give the repair one intact collar ring.
adj=[set() for _ in range(N)]
for tet in T:
    cc=np.unique(tet%N)
    for i in range(len(cc)):
        for j in range(i+1,len(cc)):
            a,b=int(cc[i]),int(cc[j]);adj[a].add(b);adj[b].add(a)
dist=np.full(N,999,np.int32);front=list(sorted(target_cols))
for c in front: dist[c]=0
for depth in range(1,4):
    nxt=set()
    for c in front:
        for v in adj[c]:
            if dist[v]==999: nxt.add(v)
    for c in nxt: dist[c]=depth
    front=nxt
free=np.flatnonzero(dist<=1).astype(np.int64)
collar=np.flatnonzero(dist==2).astype(np.int64)
idx=np.full(N,-1,np.int64);idx[free]=np.arange(len(free))
sidx=np.full(N,-1,np.int64);sidx[shape_cols]=np.arange(len(shape_cols))
rows=np.flatnonzero(np.isin(T%N,free).any(axis=1)).astype(np.int64)
TT=T[rows]
edge_set=set()
for c in free:
    for v in adj[int(c)]:
        if dist[v]<=2: edge_set.add(tuple(sorted((int(c),int(v)))))
edge_arr=np.asarray(sorted(edge_set),np.int64)
edge_i=idx[edge_arr[:,0]].copy();edge_j=idx[edge_arr[:,1]].copy()
# Put each free endpoint first so fixed-ring neighbors use edge_j=-1, never idx[-1].
swap=edge_i<0
edge_i[swap],edge_j[swap]=edge_j[swap].copy(),edge_i[swap].copy()
assert np.all(edge_i>=0) and np.all(edge_j< len(free))
O=X[:N];I=X[N:2*N];D=X[2*N:]
vs=I-O;vf=D-I
ls=np.linalg.norm(vs[shape_cols],axis=1);lf=np.linalg.norm(vf[shape_cols],axis=1)
assert np.all(ls>0) and np.all(lf>0)
nd=3*len(free); ns=len(shape_cols); nv=nd+2*ns

def unpack(x): return x[:nd].reshape(-1,3),x[nd:nd+ns],x[nd+ns:]
def emit64(x):
    d,a,b=unpack(x)
    P=X.copy()
    delta=np.zeros((N,3),np.float64);delta[free]=d*1e-3
    P[:N]=O+delta
    P[N:2*N]=I+delta
    P[2*N:]=D+delta
    P[N+shape_cols]-=(a*1e-3/ls)[:,None]*vs[shape_cols]
    P[2*N+shape_cols]-=(a*1e-3/ls)[:,None]*vs[shape_cols]
    P[2*N+shape_cols]-=(b*1e-3/lf)[:,None]*vf[shape_cols]
    return P

def plane_fun_jac(x):
    P=emit64(x)
    vals=(np.einsum("ij,ij->i",plane_n,P[plane_nodes]-plane_h)-MARGIN)*1000.0
    J=np.zeros((len(plane_rows),nv),np.float64)
    cols=plane_nodes%N; layers=plane_nodes//N
    free_idx=idx[cols];mask=free_idx>=0
    rr=np.flatnonzero(mask)
    J[rr,3*free_idx[mask]+0]=plane_n[mask,0]
    J[rr,3*free_idx[mask]+1]=plane_n[mask,1]
    J[rr,3*free_idx[mask]+2]=plane_n[mask,2]
    sj=sidx[cols];mask=(sj>=0)&(layers>=1);rr=np.flatnonzero(mask)
    if len(rr): J[rr,nd+sj[mask]]=-np.einsum("ij,ij->i",plane_n[mask],vs[cols[mask]]/ls[sj[mask],None])
    mask=(sj>=0)&(layers==2);rr=np.flatnonzero(mask)
    if len(rr): J[rr,nd+ns+sj[mask]]=-np.einsum("ij,ij->i",plane_n[mask],vf[cols[mask]]/lf[sj[mask],None])
    return vals,J

def quality_fun_jac(x):
    P=emit64(x);q,G=cs.normalized_tet_quality_and_gradient(P[TT])
    qv=(q-QMIN)*Q_SCALE
    J=np.zeros((len(rows),nv),np.float64)
    cols=TT%N;layers=TT//N
    for k in range(4):
        c=cols[:,k];layer=layers[:,k];g=G[:,k,:]
        ix=idx[c];mask=ix>=0;rr=np.flatnonzero(mask)
        for ax in range(3): np.add.at(J,(rr,3*ix[mask]+ax),g[mask,ax]*1e-3*Q_SCALE)
        ss=sidx[c];mask=(ss>=0)&(layer>=1);rr=np.flatnonzero(mask)
        if len(rr):
            u=vs[c[mask]]/ls[ss[mask],None]
            der=-np.einsum("ij,ij->i",g[mask],u)*1e-3*Q_SCALE
            np.add.at(J,(rr,nd+ss[mask]),der)
        mask=(ss>=0)&(layer==2);rr=np.flatnonzero(mask)
        if len(rr):
            u=vf[c[mask]]/lf[ss[mask],None]
            der=-np.einsum("ij,ij->i",g[mask],u)*1e-3*Q_SCALE
            np.add.at(J,(rr,nd+ns+ss[mask]),der)
    return qv,J

def obj(x):
    d,a,b=unpack(x);p=plane_fun_jac(x)[0];viol=np.minimum(p,0.0)
    left=d[edge_i];ok=edge_j>=0;right=np.zeros_like(left);right[ok]=d[edge_j[ok]]
    return float(viol@viol+1e-6*(d.ravel()@d.ravel()+0.5*(a@a+b@b))+3e-6*np.sum((left-right)**2))
def obj_jac(x):
    d,a,b=unpack(x);p,Pj=plane_fun_jac(x);viol=np.minimum(p,0.0)
    g=Pj.T@(2.0*viol)
    gd=g[:nd].reshape(-1,3)+2e-6*d
    ga=g[nd:nd+ns]+1e-6*a;gb=g[nd+ns:]+1e-6*b
    left=d[edge_i];ok=edge_j>=0;right=np.zeros_like(left);right[ok]=d[edge_j[ok]]
    diff=left-right;gamma=3e-6
    np.add.at(gd,edge_i,2*gamma*diff);np.add.at(gd,edge_j[ok],-2*gamma*diff[ok])
    return np.r_[gd.ravel(),ga,gb]
qfun=lambda x:quality_fun_jac(x)[0]
qjac=lambda x:quality_fun_jac(x)[1]
pfun=lambda x:plane_fun_jac(x)[0]
pjac=lambda x:plane_fun_jac(x)[1]

# Recover cumulative 022 parameters over positive 014; this verifies 022 is only a starting iterate.
d0=(Prev[free]-O[free])*1000.0
a0=np.zeros(ns);b0=np.zeros(ns)
for sj,c0 in enumerate(shape_cols):
    c=int(c0);delta=d0[idx[c]]*1e-3
    us=vs[c]/ls[sj];uf=vf[c]/lf[sj]
    a0[sj]=-float(np.dot(Prev[N+c]-I[c]-delta,us))*1000.0
    b0[sj]=-float(np.dot(Prev[2*N+c]-D[c]-delta+a0[sj]*1e-3*us,uf))*1000.0
x0=np.r_[d0.ravel(),a0,b0]
roundtrip=emit64(x0).astype(np.float32)
roundtrip_error=float(np.max(np.abs(roundtrip.astype(np.float64)-Prev)))
if roundtrip_error>2e-7: raise RuntimeError(f"022 cumulative-parameter replay mismatch {roundtrip_error}m")
start_plane=pfun(x0)
q0=qfun(x0)/Q_SCALE+QMIN
# Directional derivative checks on frozen arrays.
rng=np.random.default_rng(24024);xt=x0.copy();v=rng.normal(size=nv);v/=np.linalg.norm(v);eps=2e-5
fdp=(pfun(xt+eps*v)-pfun(xt-eps*v))/(2*eps);anp=pjac(xt)@v
fdq=(qfun(xt+eps*v)-qfun(xt-eps*v))/(2*eps);anq=qjac(xt)@v
fdo=(obj(xt+eps*v)-obj(xt-eps*v))/(2*eps);ano=float(obj_jac(xt)@v)
deriv={"plane_max_abs":float(np.max(np.abs(fdp-anp))),"quality_max_abs":float(np.max(np.abs(fdq-anq))),"objective_abs":abs(float(fdo-ano))}
if deriv["plane_max_abs"]>2e-5 or deriv["quality_max_abs"]>1e-3 or deriv["objective_abs"]>2e-5: raise RuntimeError(f"024 derivative preflight failed {deriv}")
if os.environ.get("O12_024_DERIVATIVE_ONLY")=="1":
    print(json.dumps({"derivative":deriv,"constraint_rows":len(plane_rows),"free_columns":len(free),"shape_columns":shape_cols.tolist(),"affected_tets":len(rows)},indent=2));raise SystemExit(0)

lo_s=np.minimum(.2*ls,.0001);lo_f=np.minimum(.2*lf,.0001)
bounds=[(-8.0,8.0)]*nd+[(0.0,max(0.0,float(ls[i]*1000-lo_s[i]*1000))) for i in range(ns)]+[(0.0,max(0.0,float(lf[i]*1000-lo_f[i]*1000))) for i in range(ns)]
bad=[(i,float(v),tuple(map(float,bounds[i]))) for i,v in enumerate(x0) if not float(bounds[i][0])<=float(v)<=float(bounds[i][1])]
projection_max_mm=max((max(bounds[i][0]-v,0.0,v-bounds[i][1]) for i,v,_ in bad),default=0.0)
if projection_max_mm>1e-3: raise RuntimeError(f"022 cumulative parameters exceed 024 bounds beyond tiny Float32 fit residue: {bad[:20]}")
raw_x0=x0.copy()
x0=np.asarray([np.clip(v,lo,hi) for v,(lo,hi) in zip(x0,bounds)],np.float64)
roundtrip=emit64(x0).astype(np.float32)
roundtrip_error=float(np.max(np.abs(roundtrip.astype(np.float64)-Prev)))
if roundtrip_error>2e-7: raise RuntimeError(f"bounded cumulative 022 replay mismatch {roundtrip_error}m")
start_plane=pfun(x0)
q0=qfun(x0)/Q_SCALE+QMIN
def merit(x):
    pv=pfun(x);qv=qfun(x);d,a,b=unpack(x)
    return float(np.sum(np.minimum(pv,0.0)**2)+1e4*np.sum(np.minimum(qv,0.0)**2)+1e-9*(d.ravel()@d.ravel()+a@a+b@b))
def feasible(x): return bool(np.min(qfun(x))>=-1e-8 and np.min(pfun(x))>=-1e-4)
best_merit_x=x0.copy();best_merit=merit(x0)
best_feasible_x=x0.copy() if feasible(x0) else None
best_feasible_obj=obj(x0) if best_feasible_x is not None else float("inf")
last_x=x0.copy();iters=0;start=time.monotonic();deadline=start+28.0
class Deadline(Exception): pass
def callback(xk):
    global last_x,iters,best_merit_x,best_merit,best_feasible_x,best_feasible_obj
    xk=np.asarray(xk,np.float64).copy();last_x=xk;iters+=1;m=merit(xk)
    if m<best_merit:best_merit=m;best_merit_x=xk.copy()
    if feasible(xk) and obj(xk)<best_feasible_obj:best_feasible_obj=obj(xk);best_feasible_x=xk.copy()
    if time.monotonic()>deadline: raise Deadline()
cons=[{"type":"ineq","fun":qfun,"jac":qjac},{"type":"ineq","fun":pfun,"jac":pjac}]
try:
    res=minimize(obj,x0,jac=obj_jac,method="SLSQP",bounds=bounds,constraints=cons,callback=callback,options={"maxiter":35,"ftol":1e-10,"disp":True})
    terminal_x=np.asarray(res.x,np.float64).copy();solver={"success":bool(res.success),"message":str(res.message),"nit":int(res.nit),"fun":float(res.fun)}
except Deadline:
    terminal_x=last_x.copy();solver={"success":False,"message":"bounded_wallclock_stop","nit":iters,"fun":float(obj(terminal_x))}
if merit(terminal_x)<best_merit:best_merit=merit(terminal_x);best_merit_x=terminal_x.copy()
if feasible(terminal_x) and obj(terminal_x)<best_feasible_obj:best_feasible_obj=obj(terminal_x);best_feasible_x=terminal_x.copy()
selected_x=best_feasible_x.copy() if best_feasible_x is not None else best_merit_x.copy()
selection="best_feasible_iterate" if best_feasible_x is not None else "best_merit_iterate"
P64=emit64(selected_x);Y=P64.astype(np.float32);Y64=Y.astype(np.float64)
qa=cs.normalized_tet_quality_and_gradient(Y64[TT])[0];qfull=cs.normalized_tet_quality_and_gradient(Y64[T])[0]
plane_slack=np.einsum("ij,ij->i",plane_n,Y64[plane_nodes]-plane_h)-MARGIN
P_OUT=OUT/"candidate.npz"
np.savez_compressed(P_OUT,points_world_f32=Y,tetrahedra=T.astype(np.int32),material_ids=mat)
np.savez_compressed(OUT/"construction-planes.npz",node_ids=plane_nodes,points_world_m=plane_h,normals_world=plane_n,
                    margins_m=np.full(len(plane_rows),MARGIN,np.float64),kinds=np.asarray([r["kind"] for r in plane_rows]),
                    columns=np.asarray([r["column"] for r in plane_rows],np.int32),layers=(plane_nodes//N).astype(np.int8),
                    bank_body=np.asarray([r.get("body_index",-1) for r in plane_rows],np.int32),
                    bank_stable_id=np.asarray([r.get("bank_stable_id",-1) for r in plane_rows],np.int32),
                    triangle_stable_id=np.asarray([r.get("triangle_stable_id") if r.get("triangle_stable_id") is not None else -1 for r in plane_rows],np.int64),
                    route_witness=np.asarray([r.get("route_witness",-1) for r in plane_rows],np.int32),
                    endpoint_side=np.asarray([r.get("outside_endpoint","") for r in plane_rows]))
np.savez_compressed(OUT/"solver-state.npz",initial_x=x0,terminal_x=terminal_x,selected_x=selected_x,
                    best_merit_x=best_merit_x,best_feasible_x=(best_feasible_x if best_feasible_x is not None else np.empty(0)),
                    best_merit=np.asarray(best_merit),best_feasible_objective=np.asarray(best_feasible_obj),
                    cumulative_replay_max_error_m=np.asarray(roundtrip_error))
changed=np.any(Y!=X.astype(np.float32),axis=1)
free_nodes=np.concatenate([free,N+free,2*N+free])
outside_mask=np.ones(3*N,bool);outside_mask[free_nodes]=False
outside_exact=bool(np.array_equal(Y[outside_mask],X.astype(np.float32)[outside_mask]))
report={
 "schema":"numi.human.o12-outward-local-repair.024",
 "status":"GEOMETRY_CONSTRUCTION_CANDIDATE_NOT_PHYSICALLY_ADMITTED",
 "supersedes":"o12-outward-repair-024 only by omitting the single LP-proven backside nearest-exit construction row; all other inputs, planes, bounds, q guards and optimizer settings are unchanged",
 "source_sha256":sha(Path(__file__)),"inputs":{str(p):sha(p) for p in EXPECTED},"candidate_sha256":sha(P_OUT),
 "solver":solver,"selection":selection,"derivative_checks":deriv,
 "construction":{"base_candidate_014_sha256":EXPECTED[IP],"cumulative_start_candidate_022_sha256":EXPECTED[PREV],"cumulative_replay_max_error_m":roundtrip_error,"cumulative_parameter_projection_max_mm":projection_max_mm,
                "seed_columns":seed_cols.tolist(),"route_witness_groups":len(route["route_witnesses"]),"route_segment_columns":route_cols,
                "inside_query_rows_input":inside_query_rows_input,"inside_query_rows_active":len(inside_rows),"inside_columns":inside_cols,"shape_columns":shape_cols.tolist(),
                "excluded_plane_construction_rows":[{"kind":r["kind"],"node":r["node"],"column":r["column"],"layer":r["layer"],"body_index":r["body_index"],"bank_stable_id":r["bank_stable_id"],"triangle_stable_id":r["triangle_stable_id"],"reason":"unique positive-slack phase-I dual conflict in 024; nearest exit lies backside of validated outer/first-entry constraints"} for r in dropped_conflict_rows],
                "free_columns":int(len(free)),"fixed_collar_columns":int(len(collar)),"affected_tets":int(len(rows)),
                "construction_plane_rows":len(plane_rows),"plane_kinds":{k:sum(r["kind"]==k for r in plane_rows) for k in sorted(set(r["kind"] for r in plane_rows))},
                "clearance_margin_m":MARGIN,"q_min":QMIN,"d_bounds_mm":[-8.0,8.0],"skin_floor_m_min":float(lo_s.min()),"fat_floor_m_min":float(lo_f.min()),
                "start_min_affected_q":float(q0.min()),"start_min_all_q":float(cs.normalized_tet_quality_and_gradient(Prev[T])[0].min()),
                "start_min_plane_slack_m":float(np.min(start_plane)*1e-3)},
 "candidate":{"moved_nodes":int(changed.sum()),"max_node_displacement_mm":float(np.linalg.norm(Y64-X,axis=1).max()*1000),
              "outside_free_columns_bitwise_unchanged":outside_exact,"min_affected_q_f32":float(qa.min()),"min_global_q_f32":float(qfull.min()),
              "affected_q_below_1e-5":int(np.sum(qa<QMIN)),"min_plane_slack_m_f32":float(plane_slack.min()),
              "plane_violations_below_minus_0.1um":int(np.sum(plane_slack < -1e-7)),"plane_slacks_m_f32":plane_slack.tolist(),
              "skin_shortening_mm":unpack(selected_x)[1].tolist(),"fat_shortening_mm":unpack(selected_x)[2].tolist()},
 "plane_row_sources":[{k:v for k,v in r.items() if k not in ("h","n")} for r in plane_rows],
 "limitations":["Plane rows are construction constraints; no refreshed all-bank native query or exact full-volume route audit has run on candidate 024.",
                "No reference-coordinate pullback, conservative material transfer, density/mass closure, or native physics admission.",
                "Source topology, tetrahedra, and material rows are unchanged; only local current/reference point arrays in this construction candidate are displaced."]}
(OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"candidate":str(P_OUT),"candidate_sha256":report["candidate_sha256"],"report":str(OUT/"report.json"),"solver":solver,"selection":selection,
                  "plane_rows":len(plane_rows),"free_columns":len(free),"shape_columns":shape_cols.tolist(),"affected_tets":len(rows),
                  "min_affected_q":float(qa.min()),"min_global_q":float(qfull.min()),"min_plane_slack_m":float(plane_slack.min()),
                  "plane_violations":int(np.sum(plane_slack < -1e-7)),"max_displacement_mm":report["candidate"]["max_node_displacement_mm"],"derivative_checks":deriv},indent=2))
