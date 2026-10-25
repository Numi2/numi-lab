import hashlib, importlib.util, json, subprocess, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import minimize, linprog
D=Path("/Users/n/Work/numi-human-connected-surface-069"); R=D/"full-connected-clearance-150"
BASE=R/"run-009-certified-source-normal-union-rays/endpoint-candidate.npz"
PROP=R/"run-013-first-exit-proposals/proposal-002/report.json"; FACE=R/"run-013-first-exit-proposals/face-audit-001/report.json"
SAT=R/"triangle_translation_intervals_160.py"; PLANE=D/"clearance-linearization-164/run-001/nodal-clearance-planes.npz"; PLANE_REPORT=D/"clearance-linearization-164/run-001/report.json"
OUT=R/"run-016-point-constrained-soft-fit"; AUDIT=D/"audit_connected_outer_geometry_130.py"
PRED=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py")
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
PY="/Users/n/numi-human-prep-venv-20261005/bin/python"
PINS={"base":"70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31","proposal":"2d9bfa692bc8ad903e49698cfcd5bdc7dcbee8a371550747088068ca893f292f","face":"26083de305f88c310ea12c159a6eb8e6ec2504bcd08b3d37eb5589f8ceea8171","sat":"b9e2dc94bdd4a49d3c2635e312a5d5b5b9a35786ed68695fb4fd96f7317e865d","predicate":"f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d","planes":"482439e08d1763d2f3d70006f0272d07f73e5fc9135f232811b0d7095c0ba5ed"}
CAP=.006; INC=.0015; MAXIT=4; MAXSOLVE=90.
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadpred():
 sys.path.insert(0,str(E/"incremental-exact-audit-owner-253/src"));name="numilab_human._soft016";s=importlib.util.spec_from_file_location(name,PRED);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def selfgate(m,Y,F):
 rec,_=m.float32_surface_records(np.asarray(Y,np.float32),F,source_seam_groups=());r=m._audit_self_pairs_touching_faces(rec,set(range(len(F))));return r,np.asarray(r.get("triangle_pairs",[]),np.int64).reshape(-1,2)
def write_candidate(p,z,Y):
 d={k:z[k] for k in z.files};d["points_world_m"]=np.asarray(Y,np.float32)
 with Path(p).open("xb") as f:np.savez_compressed(f,**d)
def main():
 if OUT.exists():raise FileExistsError(OUT)
 t0=time.monotonic()
 for k,p in {"base":BASE,"proposal":PROP,"face":FACE,"sat":SAT,"predicate":PRED,"planes":PLANE}.items():
  if sha(p)!=PINS[k]:raise RuntimeError(f"input hash mismatch {k}")
 pr=json.loads(PROP.read_text());r164=json.loads(PLANE_REPORT.read_text())
 if r164["npz_sha256"]!=PINS["planes"] or not r164["all_baseline_feasible"] or r164["rows"]!=2915:raise RuntimeError("164 receipt mismatch")
 if pr["candidate_sha256"]!=PINS["base"] or pr["face_audit_sha256"]!=PINS["face"] or pr["exact_current_pair_count"]!=1868:raise RuntimeError("proposal lineage mismatch")
 z=np.load(BASE,allow_pickle=False);X=np.asarray(z["points_world_m"],np.float32).astype(float);Y=X.copy();R0=np.asarray(z["points_before_joint_self_pair_repair_m"],float);F=np.asarray(z["source_O_faces"],np.int32);P=np.load(PLANE,allow_pickle=False)
 if X.shape!=(8776,3) or F.shape!=(17507,3) or not np.array_equal(X.astype(np.float32),P["baseline_points_world_m_f32"]):raise RuntimeError("base shape/plane mismatch")
 if np.linalg.norm(Y-R0,axis=1).max()>.006+2e-8:raise RuntimeError("source cap mismatch")
 targets=[];unresolved=[]
 for q in pr["per_face_summaries"]:
  d=float(q["first_current_overlap_exit"]["candidate_first_exit_distance_m"]);n=np.asarray(q["outward_world_unit"],float);f=int(q["surface_face_row"])
  if abs(np.linalg.norm(n)-1)>1e-10:raise RuntimeError("invalid target normal")
  if d+50e-6>=.006-1e-12:unresolved.append({"face":f,"exit_m":d});continue
  v=X[F[f]];targets.append((f,n*(d+50e-6),.5*np.linalg.norm(np.cross(v[1]-v[0],v[2]-v[0]))))
 if len(pr["per_face_summaries"])!=388 or len(targets)!=295:raise RuntimeError("target count mismatch")
 OUT.mkdir(parents=True);mod=loadpred();sr,sp=selfgate(mod,Y,F)
 if not sr.get("complete_surface_audit") or len(sp):raise RuntimeError("starting self gate failed")
 adj=[set() for _ in X];elen={}
 for tri in F:
  a,b,c=map(int,tri)
  for i,j in ((a,b),(b,c),(c,a)):
   if i>j:i,j=j,i
   adj[i].add(j);adj[j].add(i);elen[(i,j)]=np.linalg.norm(X[i]-X[j])
 levels={int(v):0 for f,_,_ in targets for v in F[f]};front=set(levels)
 for depth in range(1,4):
  nxt={u for v in front for u in adj[v]}-set(levels);levels.update({v:depth for v in nxt});front=nxt
 anchors=set(map(int,z["coarse_boundary_anchor_rows"]));pins=set(map(int,np.flatnonzero(z["all_inherited_exit_target_mask"])));fixed=anchors|pins
 free=np.asarray(sorted(v for v,d in levels.items() if d<=2 and v not in fixed),np.int64);loc={int(v):i for i,v in enumerate(free)}
 ar=[];ac=[];av=[];used=[];T=[];areas=[]
 for f,t,a in targets:
  cols=[loc[int(v)] for v in F[f] if int(v) in loc]
  if not cols:continue
  row=len(used)
  for j in cols:ar.append(row);ac.append(j);av.append(1/3)
  used.append(f);T.append(t);areas.append(a)
 A=sparse.coo_matrix((av,(ar,ac)),shape=(len(used),len(free))).tocsr();T=np.asarray(T);areas=np.asarray(areas);w=np.clip(areas/max(np.median(areas),1e-30),.05,20);sw=np.sqrt(w)
 br=[];bc=[];bv=[];ew=[]
 for (a,b),L in elen.items():
  if L<=0 or not((a in loc or b in loc) and a in levels and b in levels):continue
  row=len(ew)
  if a in loc:br.append(row);bc.append(loc[a]);bv.append(1.)
  if b in loc:br.append(row);bc.append(loc[b]);bv.append(-1.)
  ew.append(1/max(L,1e-9))
 B=sparse.coo_matrix((bv,(br,bc)),shape=(len(ew),len(free))).tocsr();se=np.sqrt(np.asarray(ew))
 pids=P["typed_ids"];norm=P["normals_world"];lower=P["lower_delta_from_702_m"];nodes=pids[:,0].astype(np.int64)
 con=[]
 for row,n,b in zip(nodes,norm,lower):
  if int(row) not in loc:
   if float(n@(Y[row]-X[row]))+1e-12<float(b):raise RuntimeError(f"fixed plane violation node {row}")
  else:con.append((int(row),np.asarray(n,float),float(b)))
 cr=[];cc=[];cv=[];rhs=[]
 for row,n,b in con:
  k=len(rhs);cr.extend([k]*3);cc.extend([3*loc[row]+a for a in range(3)]);cv.extend(n.tolist());rhs.append((b-float(n@(Y[row]-X[row])))*1e6)
 C=sparse.coo_matrix((cv,(cr,cc)),shape=(len(rhs),3*len(free))).tocsr();rhs=np.asarray(rhs)
 report={"schema":"numi.human.point-constrained-first-exit-soft-fit.v1","status":"RUNNING","classification":"unadmitted_geometry_construction_candidate","inputs":{"candidate_sha256":sha(BASE),"proposal_sha256":sha(PROP),"face_audit_sha256":sha(FACE),"sat_helper_sha256":sha(SAT),"exact_predicate_sha256":sha(PRED),"clearance_planes_sha256":sha(PLANE),"clearance_planes_report_sha256":sha(PLANE_REPORT),"clearance_plane_owner_sha256":r164["owner_sha256"]},"initial":{"complete_exact_self":True,"exact_self_pairs":0,"face_bank_pairs":1868,"crossing_faces":388,"max_cumulative_from_source_m":float(np.linalg.norm(Y-R0,axis=1).max()),"plane_rows":len(pids),"plane_constraints_on_free_nodes":len(rhs),"fixed_plane_rows":len(pids)-len(rhs),"minimum_plane_slack_m":float(np.min(-lower))},"method":{"finite_target_faces":len(used),"unresolved_cap_faces":unresolved,"fixed_exit_rows":sorted(pins),"fixed_anchor_rows":len(anchors),"free_nodes":len(free),"free_rings":2,"source_cap_m":.006,"increment_cap_m":INC,"point_constraint_count":len(rhs),"optimizer":"smooth L-BFGS-B target projected with node-wise Dykstra onto frozen point planes and source-radius balls"},"iterations":[],"geometry_admitted":False,"accepted_physical_steps":0}
 total=0.
 for it in range(MAXIT):
  faces=np.asarray(used,np.int64);resid=T-np.mean(Y[F[faces]]-X[F[faces]],axis=1);lam=.08;eta=1e-4
  def fg(v):
   d=v.reshape(-1,3);a=(A@d-resid)*sw[:,None];b=(B@d)*se[:,None]
   val=.5*np.sum(a*a)+.5*lam*np.sum(b*b)+.5*eta*np.sum(d*d)
   g=A.T@(a*sw[:,None])+lam*B.T@(b*se[:,None])+eta*d
   return float(val),np.asarray(g).ravel()
  rem=np.maximum(0,.006-np.linalg.norm(Y-R0,axis=1)-5e-6);step=np.minimum(INC,rem[free])/np.sqrt(3)
  bd=[(-float(v),float(v)) for v in step for _ in range(3)]
  ts=time.monotonic();sm=minimize(fg,np.zeros(3*len(free)),jac=True,method="L-BFGS-B",bounds=bd,options={"maxiter":180,"maxfun":6000,"maxcor":12,"maxls":30,"ftol":1e-12,"gtol":1e-8});ss=time.monotonic()-ts;total+=ss
  nd=3*len(free);target_um=sm.x.reshape(-1,3)*1e6
  # Dykstra projects independently per node onto its frozen point halfspaces and a
  # Euclidean step ball. The intersection is feasible at zero because each
  # previous accepted state passed the exact point gate and 164 starts feasible.
  bynode={}
  for row,n,b in con:
   rhsnode=(b-float(n@(Y[row]-X[row])))*1e6
   bynode.setdefault(row,[]).append((n,rhsnode))
  proj_um=np.zeros((len(free),3));proj_max_sweeps=0;proj_max_violation_um=0.;projection_ok=True
  tproj=time.monotonic()
  for node,idx in loc.items():
   remnode=max(0.,CAP-float(np.linalg.norm(Y[node]-R0[node]))-5e-6)
   radius_um=min(INC,remnode)*1e6
   hs=bynode.get(node,[])
   N=np.asarray([v[0] for v in hs],float).reshape(-1,3)
   h=np.asarray([v[1] for v in hs],float)
   x=target_um[idx].copy();pcorr=np.zeros_like(N);bcorr=np.zeros(3);node_ok=False;maxv=0.
   for sweep in range(1,1001):
    previous=x.copy()
    for j in range(len(h)):
     y=x+pcorr[j];violation=h[j]-float(N[j]@y)
     zproj=y+max(0.,violation)*N[j]
     pcorr[j]=y-zproj;x=zproj
    y=x+bcorr;yn=float(np.linalg.norm(y))
    zproj=y*(radius_um/yn) if yn>radius_um and yn>0 else y
    bcorr=y-zproj;x=zproj
    hv=np.maximum(h-N@x,0.) if len(h) else np.zeros(0)
    maxv=float(hv.max(initial=0.))
    if maxv<=1.e-7 and float(np.linalg.norm(x))<=radius_um+1.e-7 and float(np.linalg.norm(x-previous))<=1.e-7:
     node_ok=True;break
   proj_max_sweeps=max(proj_max_sweeps,sweep);proj_max_violation_um=max(proj_max_violation_um,maxv)
   if not node_ok:projection_ok=False;break
   proj_um[idx]=x
  proj_s=time.monotonic()-tproj;total+=proj_s
  if not projection_ok:
   report["iterations"].append({"iteration":it+1,"smooth_success":bool(sm.success),"smooth_message":str(sm.message),"smooth_seconds":ss,"projection_status":"DYKSTRA_NOT_CONVERGED","projection_sweeps":proj_max_sweeps,"maximum_violation_um":proj_max_violation_um,"projection_seconds":proj_s})
   break
  delta=np.zeros_like(Y);delta[free]=proj_um*1e-6
  predicted_slack=C@proj_um.ravel()-rhs
  if float(np.min(predicted_slack,initial=0.)) < -1.e-6:raise RuntimeError("Dykstra point-plane residual")
  before=float(np.sqrt(np.sum(w[:,None]*resid**2)));backs=[];accepted=False
  for k in range(9):
   alpha=.5**k;trial=(Y+alpha*delta).astype(np.float32).astype(float);capok=bool(np.all(np.linalg.norm(trial-R0,axis=1)<=.006+2e-8));fixedok=bool(np.array_equal(trial[list(fixed)],Y[list(fixed)]))
   slack=np.einsum("ij,ij->i",norm,trial[nodes]-X[nodes])-lower;mins=float(slack.min())
   if not capok or not fixedok or mins< -2e-8:
    backs.append({"alpha":alpha,"accepted":False,"reason":"cap_fixed_or_linear_plane","min_plane_slack_m":mins});continue
   sr,sp=selfgate(mod,trial,F);spth=OUT/f"iteration-{it+1:02d}-self-alpha-{k:02d}.npy";np.save(spth,sp)
   if not sr.get("complete_surface_audit") or len(sp):
    backs.append({"alpha":alpha,"accepted":False,"reason":"exact_self","complete":bool(sr.get("complete_surface_audit")),"pairs":len(sp),"witness_path":str(spth)});continue
   cp=OUT/f"iteration-{it+1:02d}-candidate-alpha-{k:02d}.npz";write_candidate(cp,z,trial)
   ao=OUT/f"iteration-{it+1:02d}-point-audit-alpha-{k:02d}"
   audit=subprocess.run([PY,str(AUDIT),str(cp),str(ao)],capture_output=True,text=True,timeout=75)
   (OUT/f"iteration-{it+1:02d}-audit-alpha-{k:02d}.log").write_text(audit.stdout+audit.stderr)
   if audit.returncode or not (ao/"summary.json").is_file():
    backs.append({"alpha":alpha,"accepted":False,"reason":"point_audit_error","returncode":audit.returncode,"audit_path":str(ao)});continue
   q=json.loads((ao/"summary.json").read_text());after=float(np.sqrt(np.sum(w[:,None]*(T-np.mean(trial[F[faces]]-X[F[faces]],axis=1))**2)))
   pp=q.get("complete_surface_audit") is True and int(q.get("inside_node_bank_pairs",-1))==0 and int(q.get("query_pair_count",-1))>=len(trial)
   ok=pp and after<before-1e-10
   backs.append({"alpha":alpha,"accepted":bool(ok),"reason":None if ok else ("point_gate" if not pp else "no_target_progress"),"point_audit_status":q.get("status"),"inside_node_bank_pairs":q.get("inside_node_bank_pairs"),"inside_nodes":q.get("inside_nodes"),"query_pair_count":q.get("query_pair_count"),"min_plane_slack_m":mins,"soft_before_m":before,"soft_after_m":after,"audit_path":str(ao),"candidate_sha256":sha(cp)})
   if ok:
    Y=trial;accepted=True
    report["iterations"].append({"iteration":it+1,"smooth_success":bool(sm.success),"smooth_message":str(sm.message),"smooth_nit":int(sm.nit),"smooth_seconds":ss,"projection_status":"DYKSTRA_FEASIBLE","projection_max_sweeps":proj_max_sweeps,"projection_max_violation_um":proj_max_violation_um,"projection_seconds":proj_s,"accepted_alpha":alpha,"backtracking":backs,"soft_before_m":before,"soft_after_m":after,"max_cumulative_m":float(np.linalg.norm(Y-R0,axis=1).max()),"changed_nodes":int(np.count_nonzero(np.any(Y!=X,axis=1))),"minimum_plane_slack_m":mins,"point_audit_path":str(ao),"candidate_sha256":sha(cp)})
    break
  if not accepted:
   report["iterations"].append({"iteration":it+1,"smooth_success":bool(sm.success),"smooth_message":str(sm.message),"smooth_nit":int(sm.nit),"smooth_seconds":ss,"projection_status":"DYKSTRA_FEASIBLE","projection_max_sweeps":proj_max_sweeps,"projection_max_violation_um":proj_max_violation_um,"projection_seconds":proj_s,"backtracking":backs,"status":"NO_EXACT_SELF_POINT_SAFE_PROGRESS"});break
  if total>=MAXSOLVE:report["iterations"].append({"iteration":it+1,"status":"SOLVER_BUDGET_EXCEEDED_AFTER_ACCEPT"});break
 cand=OUT/"candidate.npz";write_candidate(cand,z,Y);slack=np.einsum("ij,ij->i",norm,Y[nodes]-X[nodes])-lower;cum=np.linalg.norm(Y-R0,axis=1)
 report["final"]={"candidate_path":str(cand),"candidate_sha256":sha(cand),"max_cumulative_from_source_m":float(cum.max()),"cumulative_6mm_cap_pass":bool(np.all(cum<=.006+2e-8)),"changed_nodes":int(np.count_nonzero(np.any(Y!=X,axis=1))),"weighted_soft_residual_m":float(np.sqrt(np.sum(w[:,None]*(T-np.mean(Y[F[np.asarray(used)]]-X[F[np.asarray(used)]],axis=1))**2))),"target_face_rows":used,"linear_point_plane_rows":len(rhs),"minimum_linear_plane_slack_m":float(slack.min()),"all_linear_point_planes_pass":bool(slack.min()>=-2e-8),"all_candidate_array_keys_preserved":list(z.files)}
 report["solve_seconds_including_LP_excluding_exact_audits"]=total;report["wall_seconds"]=time.monotonic()-t0
 report["status"]="CONSTRAINED_SOFT_FIT_PARTIAL" if any("accepted_alpha" in v for v in report["iterations"]) else "NO_ITERATION_ACCEPTED"
 (OUT/"fit-report.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps({"status":report["status"],"candidate":str(cand),"candidate_sha256":sha(cand),"report":str(OUT/"fit-report.json"),"solver_seconds":total,"iterations":report["iterations"],"max_cumulative_m":float(cum.max()),"min_plane_slack_m":float(slack.min())},indent=2))
if __name__=="__main__":main()
