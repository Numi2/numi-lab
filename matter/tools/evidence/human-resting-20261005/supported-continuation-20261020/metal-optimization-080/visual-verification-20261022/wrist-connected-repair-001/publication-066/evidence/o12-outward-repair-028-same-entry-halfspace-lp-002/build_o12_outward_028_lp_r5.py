from pathlib import Path
import csv, hashlib, json, os, sys, time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack, eye

E = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
OUT = E / "o12-outward-repair-028-same-entry-halfspace-lp-002"
if OUT.exists() and any(p.name != Path(__file__).name for p in OUT.iterdir()):
    raise SystemExit("refusing to overwrite non-empty 028 output directory")
OUT.mkdir(parents=True, exist_ok=True)
BASE = E / "baseline-depth-transition-014/candidate.npz"
CAND = E / "o12-outward-repair-027-f32-gated/candidate.npz"
STATE = E / "o12-outward-repair-027-f32-gated/solver-state.npz"
PLANES = E / "o12-outward-repair-027-f32-gated/construction-planes.npz"
REPORT = E / "o12-outward-repair-027-f32-gated/report.json"
ROUTE = E / "route-audit-027-matched-query-028/route-report.json"
TRI = E / "bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight/exact-emitted.triangles.csv"
BODIES = E / "bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight/exact-emitted.bodies.csv"
HELP = E / "bone-clearance-fit-040-shared-columns/column_fit_support.py"
EXPECTED = {
 BASE: "f5b0e33933ddf0867715a877ade75279659034362ae8f586435ccbf00609f42c",
 CAND: "ccfeb27e6e939249e4c5146f77722bf2477d1f8377300bde8cc82086f44bbe1a",
 STATE: "9694fc0b3b1145d6fb2ce1aa835e47437fb6d6d290c4148de870d0a81d58ef26",
 PLANES: "9fc4f9b06e26c1ee8af02e4d06434c0da1e9ac56e6f101669bf025aa211aa545",
 REPORT: "3dc2a9d1bbcd534fafc7770c2c0a83434d25dc563d858b768e16c3230514e71e",
 ROUTE: "7d9108c9fa03161bd39cb23cfc98e7e587386ebeb5372ac1fdcd5c6705d1b972",
 TRI: "0d3418064af1d4f34d19a1ec91992afceacaaa70b6c982d4a2637ded1d829eac",
 BODIES: "dae52ab190e02586421b56381b03a4453aff1058edca7b6a42afcb9051b50b90",
 HELP: "2de0857455c25dc04bf28c83429ba751ced58439a03bfb30494e510c1a64bd16",
}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
if EXPECTED[ROUTE] == "MATCHED_ROUTE_SHA_TO_FILL":
 raise RuntimeError("pin matched-query route report SHA before running")
for p,want in EXPECTED.items():
 if not p.is_file(): raise FileNotFoundError(p)
 got=sha(p)
 if got != want: raise RuntimeError(f"pin mismatch {p}: {got} != {want}")
sys.path.insert(0,str(HELP.parent))
import column_fit_support as cs
N=2411
QMIN=1.06e-5
Q_SCALE=1.0e3
MARGIN=1.0e-4

z=np.load(BASE,allow_pickle=False)
X=z["points_world_f32"].astype(np.float64)
T=z["tetrahedra"].astype(np.int64)
mat=z["material_ids"]
assert X.shape==(3*N,3) and T.shape==(28632,4)
cand=np.load(CAND,allow_pickle=False)["points_world_f32"].astype(np.float32)
assert cand.shape==X.shape

# Preserve the exact 84 prior 027 construction rows byte-for-byte as arrays.
pz=np.load(PLANES,allow_pickle=False)
old_nodes=pz["node_ids"].astype(np.int64)
old_h=pz["points_world_m"].astype(np.float64)
old_n=pz["normals_world"].astype(np.float64)
old_m=pz["margins_m"].astype(np.float64)
assert len(old_nodes)==84 and old_h.shape==old_n.shape==(84,3)

route=json.loads(ROUTE.read_text())
assert route["counts"]["proper_hits"]==4
assert route["uncertainties"]==0
assert route["columns_with_hits"]==2
bodies={}
with BODIES.open(newline="") as f:
 for r in csv.DictReader(f):
  bodies[int(r["body_index"])]=r
need=set()
for w in route["route_witnesses"]:
 assert w["start_classification"]=="outside" and w["end_classification"]=="outside"
 assert len(w["all_hits"])==2 and not w["first_hit_ambiguous"]
 for hit in w["all_hits"]:
  assert hit["classification"]=="proper_interior_hit"
  need.add((int(w["body"]),int(w["bone_stable_id"]),int(hit["triangle_stable_id"])))
triangles={}
with TRI.open(newline="") as f:
 for r in csv.DictReader(f):
  key=(int(r["body_index"]),int(r["bone_stable_id"]),int(r["triangle_stable_id"]))
  if key in need:
   if key in triangles: raise RuntimeError(f"duplicate stable bank triangle {key}")
   triangles[key]=r
if set(triangles)!=need: raise RuntimeError(f"missing exact hit triangles {need-set(triangles)}")

def xyz_from_row(r,prefix):
 return np.asarray([float(r[prefix+"x_m"]),float(r[prefix+"y_m"]),float(r[prefix+"z_m"])],np.float64)
def local_corners(r):
 return np.asarray([[float(r[f"v{k}{a}_m"]) for a in "xyz"] for k in range(3)],np.float64)

new_rows=[]
route_summary=[]
for wi,w in enumerate(route["route_witnesses"]):
 row=int(w["row"]); seg=w["segment_name"]
 layers=(0,1) if seg=="O-I" else ((1,2) if seg=="I-D" else None)
 if layers is None: raise RuntimeError(f"unknown segment kind {seg}")
 if not np.array_equal(cand[layers[0]*N+row],np.asarray(w["segment_start_world_f32_m"],np.float32)): raise RuntimeError(f"route start node mapping differs from 027 candidate at column {row}/{seg}")
 if not np.array_equal(cand[layers[1]*N+row],np.asarray(w["segment_end_world_f32_m"],np.float32)): raise RuntimeError(f"route end node mapping differs from 027 candidate at column {row}/{seg}")
 ps=np.asarray(w["segment_start_world_f32_m"],np.float64)
 pe=np.asarray(w["segment_end_world_f32_m"],np.float64)
 body=int(w["body"]); bone=int(w["bone_stable_id"])
 br=bodies[body]
 M=np.asarray([[float(br[f"world_to_body_m{i}{j}"]) for j in range(3)] for i in range(3)],np.float64)
 pos=np.asarray([float(br[f"position_{a}_m"]) for a in "xyz"],np.float64)
 off=np.asarray([float(br[f"com_offset_{a}_m"]) for a in "xyz"],np.float64)
 hits=sorted(w["all_hits"],key=lambda h:float(h["t"]))
 if len(hits)!=2: raise RuntimeError(f"expected paired crossing hits, got {len(hits)} for column {row}/{seg}")
 # Choose the first actual entry triangle as a local separating plane and place BOTH endpoints
 # on the known exterior side. Independent entry-start/exit-end rows do not separate the segment.
 hit=hits[0]; tid=int(hit["triangle_stable_id"]); t=float(hit["t"])
 tr=triangles[(body,bone,tid)]
 loc=local_corners(tr)
 world=pos+np.linalg.solve(M,(loc+off).T).T
 nn=np.cross(world[1]-world[0],world[2]-world[0])
 area2=float(np.linalg.norm(nn))
 if not area2>0: raise RuntimeError(f"degenerate entry triangle {body}/{bone}/{tid}")
 nn/=area2
 hitpoint=ps+t*(pe-ps)
 plane_err=float(nn@(hitpoint-world[0]))
 if abs(plane_err)>2e-7: raise RuntimeError(f"route hit/triangle plane mismatch {body}/{bone}/{tid}: {plane_err}m")
 e0=world[1]-world[0];e1=world[2]-world[0];v=hitpoint-world[0]
 gram=np.array([[e0@e0,e0@e1],[e0@e1,e1@e1]],np.float64)
 uv=np.linalg.solve(gram,np.array([e0@v,e1@v],np.float64))
 bary=np.array([1.0-uv[0]-uv[1],uv[0],uv[1]])
 if float(bary.min()) < -2e-6 or float(bary.max())>1+2e-6: raise RuntimeError(f"non-interior entry barycentric for {body}/{bone}/{tid}: {bary}")
 if float(nn@(ps-hitpoint))<0: nn=-nn
 interior_t=0.5*(float(hits[0]["t"])+float(hits[-1]["t"]))
 interior_point=ps+interior_t*(pe-ps)
 start_side=float(nn@(ps-hitpoint));end_side=float(nn@(pe-hitpoint));mid_side=float(nn@(interior_point-hitpoint))
 if start_side<=0 or mid_side>=0: raise RuntimeError(f"entry plane does not separate exterior start from route interior: col={row}/{seg} start={start_side} mid={mid_side}")
 for endpoint_name,endpoint_layer,endpoint_point in (("start",layers[0],ps),("end",layers[1],pe)):
  node=endpoint_layer*N+row
  endpoint_side=float(nn@(endpoint_point-hitpoint))
  new_rows.append({"node":node,"h":hitpoint.tolist(),"n":nn.copy().tolist(),"margin":MARGIN,
                   "kind":"027_first_entry_plane_both_endpoints_exterior","column":row,"layer":endpoint_layer,
                   "body":body,"bone":bone,"triangle":tid,"route_witness":wi,
                   "segment":seg,"outside_endpoint":endpoint_name,"hit_t":t,
                   "hit_barycentric":bary.tolist(),"plane_error_m":plane_err,
                   "endpoint_signed_distance_m":endpoint_side,"entry_start_signed_distance_m":start_side,
                   "route_mid_signed_distance_m":mid_side})
 route_summary.append({"column":row,"segment":seg,"body":body,"bone":bone,
                       "policy":"both endpoints on first-entry triangle exterior halfspace",
                       "entry_triangle":tid,"entry_t":t,"entry_barycentric":bary.tolist(),
                       "plane_error_m":plane_err,"start_signed_distance_m":start_side,
                       "end_signed_distance_m":end_side,"interior_probe_t":interior_t,
                       "interior_signed_distance_m":mid_side,
                       "all_hit_triangles":[int(h["triangle_stable_id"]) for h in hits]})
assert len(new_rows)==4

plane_nodes=np.r_[old_nodes,np.asarray([r["node"] for r in new_rows],np.int64)]
plane_h=np.vstack([old_h,np.asarray([r["h"] for r in new_rows],np.float64)])
plane_n=np.vstack([old_n,np.asarray([r["n"] for r in new_rows],np.float64)])
margins=np.r_[old_m,np.asarray([r["margin"] for r in new_rows],np.float64)]
assert len(plane_nodes)==88

# Same 027 one-ring free set, collar and quality affected rows.
shape_cols=np.asarray(json.loads(REPORT.read_text())["construction"]["shape_columns"],np.int64)
adj=[set() for _ in range(N)]
for tet in T:
 cc=np.unique(tet%N)
 for i in range(len(cc)):
  for j in range(i+1,len(cc)):
   a,b=int(cc[i]),int(cc[j]);adj[a].add(b);adj[b].add(a)
dist=np.full(N,999,np.int32);front=list(sorted(map(int,shape_cols)))
for c in front: dist[c]=0
for depth in range(1,4):
 nxt=set()
 for c in front:
  for v in adj[c]:
   if dist[v]==999:nxt.add(v)
 for c in nxt:dist[c]=depth
 front=nxt
free=np.flatnonzero(dist<=1).astype(np.int64)
collar=np.flatnonzero(dist==2).astype(np.int64)
idx=np.full(N,-1,np.int64);idx[free]=np.arange(len(free))
sidx=np.full(N,-1,np.int64);sidx[shape_cols]=np.arange(len(shape_cols))
rows=np.flatnonzero(np.isin(T%N,free).any(axis=1)).astype(np.int64)
TT=T[rows]
O=X[:N];I=X[N:2*N];D=X[2*N:]
vs=I-O;vf=D-I
ls=np.linalg.norm(vs[shape_cols],axis=1);lf=np.linalg.norm(vf[shape_cols],axis=1)
nd=3*len(free);ns=len(shape_cols);nv=nd+2*ns
def unpack(x): return x[:nd].reshape(-1,3),x[nd:nd+ns],x[nd+ns:]
def emit64(x):
 d,a,b=unpack(x);P=X.copy();delta=np.zeros((N,3),np.float64);delta[free]=d*1e-3
 P[:N]=O+delta;P[N:2*N]=I+delta;P[2*N:]=D+delta
 P[N+shape_cols]-=(a*1e-3/ls)[:,None]*vs[shape_cols]
 P[2*N+shape_cols]-=(a*1e-3/ls)[:,None]*vs[shape_cols]
 P[2*N+shape_cols]-=(b*1e-3/lf)[:,None]*vf[shape_cols]
 return P
def plane_fun_jac(x):
 P=emit64(x)
 vals=(np.einsum("ij,ij->i",plane_n,P[plane_nodes]-plane_h)-margins)*1000.0
 J=np.zeros((len(plane_nodes),nv),np.float64)
 cols=plane_nodes%N;layers=plane_nodes//N;fi=idx[cols];mask=fi>=0;rr=np.flatnonzero(mask)
 for ax in range(3):J[rr,3*fi[mask]+ax]=plane_n[mask,ax]
 si=sidx[cols];mask=(si>=0)&(layers>=1);rr=np.flatnonzero(mask)
 if len(rr):J[rr,nd+si[mask]]=-np.einsum("ij,ij->i",plane_n[mask],vs[cols[mask]]/ls[si[mask],None])
 mask=(si>=0)&(layers==2);rr=np.flatnonzero(mask)
 if len(rr):J[rr,nd+ns+si[mask]]=-np.einsum("ij,ij->i",plane_n[mask],vf[cols[mask]]/lf[si[mask],None])
 return vals,J
def quality_fun_jac(x):
 P=emit64(x);q,G=cs.normalized_tet_quality_and_gradient(P[TT]);qv=(q-QMIN)*Q_SCALE
 J=np.zeros((len(rows),nv),np.float64)
 cols=TT%N;layers=TT//N
 for k in range(4):
  c=cols[:,k];layer=layers[:,k];g=G[:,k,:];ix=idx[c];mask=ix>=0;rr=np.flatnonzero(mask)
  for ax in range(3):np.add.at(J,(rr,3*ix[mask]+ax),g[mask,ax]*1e-3*Q_SCALE)
  ss=sidx[c];mask=(ss>=0)&(layer>=1);rr=np.flatnonzero(mask)
  if len(rr):
   u=vs[c[mask]]/ls[ss[mask],None];der=-np.einsum("ij,ij->i",g[mask],u)*1e-3*Q_SCALE
   np.add.at(J,(rr,nd+ss[mask]),der)
  mask=(ss>=0)&(layer==2);rr=np.flatnonzero(mask)
  if len(rr):
   u=vf[c[mask]]/lf[ss[mask],None];der=-np.einsum("ij,ij->i",g[mask],u)*1e-3*Q_SCALE
   np.add.at(J,(rr,nd+ns+ss[mask]),der)
 return qv,J

state=np.load(STATE,allow_pickle=False)
x0=state["selected_x"].astype(np.float64)
assert x0.shape==(nv,)
replay=emit64(x0).astype(np.float32)
if not np.array_equal(replay,cand): raise RuntimeError(f"027 selected-x F32 replay mismatch max={np.max(np.abs(replay-cand))}")
p0,Jp=plane_fun_jac(x0);q0,Jq=quality_fun_jac(x0)
assert len(p0)==88 and len(q0)==len(rows)
rng=np.random.default_rng(28028);v=rng.normal(size=nv);v/=np.linalg.norm(v);eps=1e-5
fdp=(plane_fun_jac(x0+eps*v)[0]-plane_fun_jac(x0-eps*v)[0])/(2*eps)
fdq=(quality_fun_jac(x0+eps*v)[0]-quality_fun_jac(x0-eps*v)[0])/(2*eps)
deriv={"plane_max_abs":float(np.max(np.abs(fdp-Jp@v))),"quality_max_abs":float(np.max(np.abs(fdq-Jq@v)))}
if deriv["plane_max_abs"]>2e-5 or deriv["quality_max_abs"]>1e-3: raise RuntimeError(f"028 LP directional derivative check failed {deriv}")
if os.environ.get("O12_028_PRELIGHT_ONLY")=="1":
 print(json.dumps({"preflight":"pass","prior_plane_rows":len(old_nodes),"added_endpoint_rows":len(new_rows),"plane_derivatives":deriv["plane_max_abs"],"quality_derivatives":deriv["quality_max_abs"],"free_columns":len(free),"shape_columns":shape_cols.tolist(),"affected_tets":len(rows),"endpoint_groups":route_summary},indent=2));raise SystemExit(0)

lo_s=np.minimum(.2*ls,.0001);lo_f=np.minimum(.2*lf,.0001)
lo=np.r_[np.full(nd,-8.0),np.zeros(ns),np.zeros(ns)]
hi=np.r_[np.full(nd,8.0),ls*1000.0-lo_s*1000.0,lf*1000.0-lo_f*1000.0]
if np.any(x0<lo-1e-7) or np.any(x0>hi+1e-7): raise RuntimeError("027 selected parameter vector violates same 027 bounds")

# LP: first-order quality guards, exact linear plane guards, and minimum L-infinity parameter correction.
Arows=np.vstack([-Jp,-Jq])
brows=np.r_[p0,q0]
base=Arows
Aineq=np.zeros((len(Arows)+2*nv,nv+1),np.float64)
Aineq[:len(Arows),:nv]=base
Aineq[len(Arows):len(Arows)+nv,:nv]=np.eye(nv)
Aineq[len(Arows):len(Arows)+nv,nv]=-1.0
Aineq[len(Arows)+nv:len(Arows)+2*nv,:nv]=-np.eye(nv)
Aineq[len(Arows)+nv:,nv]=-1.0
bineq=np.r_[brows,np.zeros(2*nv)]
bounds=[(float(lo[i]-x0[i]),float(hi[i]-x0[i])) for i in range(nv)]+[(0.0,None)]
c=np.zeros(nv+1);c[-1]=1.0
t0=time.monotonic()
res=linprog(c,A_ub=csr_matrix(Aineq),b_ub=bineq,bounds=bounds,method="highs")
elapsed=time.monotonic()-t0
report={
 "schema":"numi.human.o12-endpoint-plane-lp.028",
 "status":"LINEARIZED_FEASIBILITY_ONLY_NOT_GEOMETRY_OR_PHYSICAL_ADMISSION",
 "source_sha256":sha(Path(__file__)),
 "inputs":{str(p):sha(p) for p in EXPECTED},
 "prior_plane_rows_preserved":int(len(old_nodes)),
 "new_endpoint_plane_rows":new_rows,
 "endpoint_route_groups":route_summary,
 "derivative_checks":deriv,
 "lp":{"success":bool(res.success),"status":int(res.status),"message":str(res.message),"elapsed_seconds":elapsed,
       "n_variables":int(nv),"n_plane_rows":int(len(p0)),"n_linearized_quality_rows":int(len(q0)),
       "minimum_linf_parameter_correction_mm":float(res.x[-1]) if res.success else None,
       "minimum_plane_value_at_start_mm":float(np.min(p0)),"minimum_linearized_quality_value_at_start":float(np.min(q0)),
       "minimum_slack_if_success":float(np.min(bineq-Aineq@res.x)) if res.success else None,
       "objective_parameter_units":"mm for all 206 variables, L-infinity"},
 "source":{"free_columns":int(len(free)),"shape_columns":shape_cols.tolist(),"fixed_collar_columns":int(len(collar)),
           "affected_tets":int(len(rows)),"same_d_bounds_mm":[-8.0,8.0],
           "skin_floor_m_min":float(lo_s.min()),"fat_floor_m_min":float(lo_f.min()),
           "q_linearization_target":QMIN,"q_scale":Q_SCALE},
 "limitations":["Only a local first-order LP feasibility probe at emitted 027 coordinates; both endpoints are constrained to the exterior halfspace of each first-entry triangle. No nonlinear solve, F32 candidate selection, refreshed route audit, full skin/tet clearance, material/reference rebuild, mass transfer, or physical/native admission."]
}
if res.success:
 dx=res.x[:nv]
 xlp=x0+dx
 plp,_=plane_fun_jac(xlp);qlp,_=quality_fun_jac(xlp)
 P32=emit64(xlp).astype(np.float32).astype(np.float64)
 qf32=cs.normalized_tet_quality_and_gradient(P32[T])[0]
 pf32=np.einsum("ij,ij->i",plane_n,P32[plane_nodes]-plane_h)-margins
 if os.environ.get("O12_LP_REPORT_ONLY") != "1":
  np.savez_compressed(OUT/"lp-proposal.npz",x_lp=xlp,delta_x=dx,points_world_f32=P32.astype(np.float32),tetrahedra=T.astype(np.int32),material_ids=mat)
 report["lp"].update({"max_abs_parameter_correction_mm":float(np.max(np.abs(dx))),
                       "min_plane_linearized_mm":float(plp.min()),"min_quality_linearized":float(qlp.min()),
                       "emitted_f32_min_global_q":float(qf32.min()),"emitted_f32_min_plane_slack_m":float(pf32.min()),
                       "emitted_f32_quality_pass_1e-5":bool(qf32.min()>=1e-5),
                       "emitted_f32_plane_pass_minus_0.1um":bool(pf32.min()>=-1e-7),
                       "max_node_displacement_mm":float(np.linalg.norm(P32-X,axis=1).max()*1000.0)})
else:
 # Elastic LP for a compact conflict diagnostic, with plane/quality row slacks in their native scales.
 nr=len(Arows);nsl=nr
 Aelastic=np.zeros((nr,nv+nsl),np.float64);Aelastic[:,:nv]=Arows;Aelastic[:,nv:]=-np.eye(nsl)
 c_el=np.r_[np.zeros(nv),np.ones(nsl)]
 be=linprog(c_el,A_ub=csr_matrix(Aelastic),b_ub=brows,
            bounds=[(float(lo[i]-x0[i]),float(hi[i]-x0[i])) for i in range(nv)]+[(0.0,None)]*nsl,
            method="highs")
 if be.success:
  sl=be.x[nv:]
  order=np.argsort(sl)[::-1]
  report["elastic_conflict"]={"success":True,"minimum_sum_row_slack_native_units":float(sl.sum()),
      "top_nonzero_rows":[{"type":"plane" if i<len(p0) else "quality","row":int(i if i<len(p0) else i-len(p0)),
                           "slack":float(sl[i]),"plane_metadata":({"node":int(plane_nodes[i]),"kind":str(pz["kinds"][i]) if i<len(old_nodes) else "new_endpoint_exterior","margin_m":float(margins[i])} if i<len(p0) else None)}
                          for i in order[:20] if sl[i]>1e-9]}
 else:
  report["elastic_conflict"]={"success":False,"message":str(be.message)}
(OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"out":str(OUT),"source_sha256":report["source_sha256"],"status":report["status"],
                  "lp":report["lp"],"elastic_conflict":report.get("elastic_conflict")},indent=2))

