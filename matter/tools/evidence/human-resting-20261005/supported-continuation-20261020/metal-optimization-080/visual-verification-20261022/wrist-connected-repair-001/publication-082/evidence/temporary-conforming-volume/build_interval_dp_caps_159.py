#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.util, json, pathlib, subprocess, sys, time
from collections import defaultdict, Counter
from functools import lru_cache
import numpy as np
R=pathlib.Path("/Users/n/Work/numi-human-connected-surface-069")
OUT=R/"temporary-outer-solid-159/run-004-interval-dp"
SRC=R/"full-connected-clearance-150/run-004/pass-003/candidate.npz"
SRC_SHA="d2beb7c9cd71ae7969d140b7783bfc5d4254e2ab8c51f77f9974d7d6e9e056ab"
PRIOR=R/"temporary-outer-solid-155/run-001/construct_outer_solid_155.py"
HELPER=R/"coupled-layer-trial-128/assemble_layers_128.py"
E=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
PRED=E/"collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py"
PI=E/"incremental-exact-audit-owner-253/src"
TETGEN=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/arm-local-pocket-remesh-116/local-remesh-118/tetgen-v1.6.0/tetgen")
# Correct the path explicitly; fail closed if the pinned binary is absent.
TETGEN=pathlib.Path(str(TETGEN).replace("20261022","20261022"))
LIMIT=20.0

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def need(x,m):
 if not x: raise ValueError(m)
def mod(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def rec(ids,rid,L):
 t=tuple(L[i] for i in ids)
 return (t,tuple(min(p[k] for p in t) for k in range(3)),tuple(max(p[k] for p in t) for k in range(3)),rid,tuple(ids))
def vol(X,F):
 T=X[F];return float(np.einsum("ij,ij->i",T[:,0],np.cross(T[:,1],T[:,2])).sum()/6)

def main():
 OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic();deadline=start+LIMIT
 report={"status":"STARTED","method":"memoized interval polygon DP with exact complete-source collision blockers",
 "source_candidate":str(SRC),"source_sha256_expected":SRC_SHA,
 "source_exclusion":"all fixed source O triangles checked with exact Float32 predicate",
 "caps_are_temporary_construction_only_not_physical_skin":True,"time_limit_seconds":LIMIT,"tetgen_executed":False}
 try:
  need(sha(SRC)==SRC_SHA,"source hash mismatch")
  for p in (PRIOR,HELPER,PRED,TETGEN):need(p.is_file(),f"missing {p}")
  ear=mod(PRIOR,"ear155");mesh=mod(HELPER,"layer128dp")
  sys.path.insert(0,str(PI));pred=mod(PRED,"numilab_human.cardiac_cavity_intersections")
  with np.load(SRC,allow_pickle=False) as z:A={k:z[k] for k in z.files}
  X=np.asarray(A["points_world_m"],np.float64);X32=X.astype(np.float32);Fs=np.asarray(A["source_O_faces"],np.int64)
  need(X.shape==(8776,3) and Fs.shape==(17507,3),"source dimensions mismatch")
  F=Fs[:,[0,2,1]].copy();loops,directed=ear.boundary_loops(F)
  need(sorted(map(len,loops))==[21,24] and len(directed)==45,"boundary loop mismatch")
  loops=sorted(loops,key=lambda x:(len(x),tuple(map(int,x))))
  occ=defaultdict(list)
  for f,t in enumerate(F):
   for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):occ[tuple(sorted((int(a),int(b))))].append(f)
  bedges={tuple(sorted((int(a),int(b)))) for a,b in directed};forbidden=set(occ)-bedges
  source,_=pred.float32_surface_records(X32,F,source_seam_groups=())
  index=pred._prepare_surface_aabb(source);L=tuple(pred.float32_point_lattice_key(p) for p in X32)
  exact={};blockers=[];stats={"unique_tested_triangles":0,"source_aabb_pairs":0,"blocked_triangles":0}
  loop_reports=[];caps_by_loop=[]
  for li,la in enumerate(loops):
   loop=np.asarray(la,np.int64);n=len(loop);Q,area,eps,center,basis=ear.project_loop(X,loop);sgn=1. if area>0 else -1.
   boundary={tuple(sorted((i,(i+1)%n))) for i in range(n)};segments=[(i,(i+1)%n) for i in range(n)]
   diagcache={};tricache={};reject=Counter()
   def onseg(p,a,b):return abs(ear.cross2(a,b,p))<=eps and np.dot(p-a,p-b)<=eps
   def diag(i,j):
    key=tuple(sorted((int(i),int(j))))
    if key in diagcache:return diagcache[key]
    if key in boundary:diagcache[key]=(True,"boundary");return diagcache[key]
    if tuple(sorted((int(loop[i]),int(loop[j])))) in forbidden:
     diagcache[key]=(False,"source_internal_edge");return diagcache[key]
    a,b=Q[i],Q[j]
    for u,v in segments:
     if u in (i,j) or v in (i,j):continue
     c,d=Q[u],Q[v]
     if ear.seg_intersect(a,b,c,d,eps) or any((onseg(c,a,b),onseg(d,a,b),onseg(a,c,d),onseg(b,c,d))):
      diagcache[key]=(False,"boundary_cross_or_contact");return diagcache[key]
    if any(onseg(Q[m],a,b) for m in range(n) if m not in (i,j)):
     diagcache[key]=(False,"through_other_polygon_vertex");return diagcache[key]
    if not ear.point_in_polygon((a+b)*.5,Q,eps):
     diagcache[key]=(False,"outside_polygon");return diagcache[key]
    diagcache[key]=(True,"internal");return diagcache[key]
   def legaltri(i,k,j):
    key=(i,k,j)
    if key in tricache:return tricache[key]
    if sgn*ear.cross2(Q[i],Q[k],Q[j])<=eps:
     tricache[key]=False;reject["projected_degenerate_or_wrong_winding"]+=1;return False
    for e in ((i,k),(k,j),(i,j)):
     ok,why=diag(*e)
     if not ok:tricache[key]=False;reject["diagonal:"+why]+=1;return False
    if not ear.point_in_polygon((Q[i]+Q[k]+Q[j])/3,Q,eps):
     tricache[key]=False;reject["centroid_outside"]+=1;return False
    if any(ear.point_in_tri(Q[m],Q[i],Q[k],Q[j],sgn,eps) for m in range(n) if m not in (i,k,j)):
     tricache[key]=False;reject["contains_other_boundary_vertex"]+=1;return False
    ids=tuple(int(loop[x]) for x in (i,k,j));P=X32[list(ids)].astype(np.float64)
    if np.linalg.norm(np.cross(P[1]-P[0],P[2]-P[0]))==0:
     tricache[key]=False;reject["F32_degenerate_triangle"]+=1;return False
    sk=tuple(sorted(ids))
    if sk not in exact:
     rid=len(F)+len(exact);out=pred._audit_pair_prepared_first(index,[rec(ids,rid,L)],same_surface=True)
     pairs=out.get("triangle_pairs",[])
     hits=sorted({int(a if int(a)<len(F) else b) for a,b in pairs if int(a)<len(F) or int(b)<len(F)})
     exact[sk]=hits;stats["unique_tested_triangles"]+=1;stats["source_aabb_pairs"]+=int(out.get("aabb_candidate_pairs",0))
     if hits:
      stats["blocked_triangles"]+=1;blockers.append({"loop":li,"triangle":list(sk),"source_face_rows":hits})
    if exact[sk]:
     tricache[key]=False;reject["exact_source_intersection"]+=1;return False
    tricache[key]=True;return True
   dead=[]
   @lru_cache(None)
   def solve(i,j):
    if time.monotonic()>deadline:raise TimeoutError("20s interval-DP limit")
    if j<=i+1:return ()
    for k in range(i+1,j):
     if not legaltri(i,k,j):continue
     a=solve(i,k)
     if a is None:continue
     b=solve(k,j)
     if b is not None:return a+b+((i,k,j),)
    dead.append((i,j));return None
   try:ans=solve(0,n-1)
   except TimeoutError:
    report.update({"status":"INTERVAL_DP_TIME_LIMIT","loop_reports":loop_reports,"failed_loop":li,
     "candidate_triples":len(tricache),"interval_subproblems":solve.cache_info().currsize,
     "blocked_source_triangles":blockers,"exact_source_counters":stats,"tetgen_executed":False})
    break
   item={"loop":li,"vertex_count":n,"projected_area_m2":float(area),"interval_subproblems":solve.cache_info().currsize,
    "candidate_triples":len(tricache),"dead_interval_count":len(dead),"dead_interval_sample":[list(x) for x in dead[:80]],
    "rejections":dict(sorted(reject.items()))}
   loop_reports.append(item)
   if ans is None:
    report.update({"status":"EXHAUSTIVE_NO_VALID_PROJECTED_TRIANGULATION","failed_loop":li,
     "limitation":"Exhaustive only over this projected interval-triangulation class and its declared source blockers; not proof no nonprojected disk exists.",
     "loop_reports":loop_reports,"blocked_source_triangle_witnesses":blockers,"exact_source_counters":stats,
     "tetgen_executed":False})
    break
   area2=sum(abs(ear.cross2(Q[i],Q[k],Q[j]))*.5 for i,k,j in ans)
   need(abs(area2-abs(area))<=max(1e-12,abs(area)*1e-10),"triangulated projected area mismatch")
   item["triangulated_projected_area_m2"]=float(area2)
   caps_by_loop.append(np.asarray([[int(loop[i]),int(loop[j]),int(loop[k])] for i,j,k in ans],np.int64)[:,[0,2,1]])
  if report["status"] in ("EXHAUSTIVE_NO_VALID_PROJECTED_TRIANGULATION","INTERVAL_DP_TIME_LIMIT"):pass
  else:
   need(len(caps_by_loop)==2,"not both loops solved")
   caps=np.vstack(caps_by_loop);SF=np.vstack((F,caps)).astype(np.int64)
   need(caps.shape==(41,3),"cap count mismatch")
   topo=mesh.shell_topology(SF);need(topo["bad_oriented_edge_count"]==0,f"bad closed-shell oriented edges {topo['bad_edge_witnesses'][:8]}")
   v=vol(X,SF);need(v>0 and np.isfinite(v),"closed shell volume not positive")
   npz=OUT/"interval-dp-capped-solid.npz"
   np.savez_compressed(npz,source_points_world_m=X,source_O_faces_original=Fs.astype(np.int32),
    shell_source_faces_outward=F.astype(np.int32),temporary_cap_faces=caps.astype(np.int32),shell_faces_closed=SF.astype(np.int32),
    caps_are_temporary_construction_only=np.asarray([True]),input_candidate_sha256=np.asarray([SRC_SHA]))
   poly=OUT/"interval-dp-capped-solid.poly"
   with poly.open("w") as f:
    f.write("# Temporary construction caps, not physical skin.\n");f.write(f"{len(X)} 3 0 0\n")
    for i,p in enumerate(X,1):f.write(f"{i} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g}\n")
    f.write(f"{len(SF)} 1\n")
    for fi,t in enumerate(SF,1):f.write(f"1 0 {fi}\n3 {int(t[0])+1} {int(t[1])+1} {int(t[2])+1}\n")
    f.write("0\n0\n")
   audit=mesh.exact_self_audit(X32,SF);pairs=np.asarray(audit["pairs_array"],np.int64).reshape(-1,2)
   np.save(OUT/"exact-self-pairs.npy",pairs.astype(np.int32))
   report.update({"status":"CAP_BUILT","loop_reports":loop_reports,"blocked_source_triangle_witnesses":blockers,
    "exact_source_counters":stats,"closed_face_count":len(SF),"closed_oriented_topology":topo,"signed_volume_m3":v,
    "exact_float32_full_shell_audit_complete":bool(audit["complete"]),"exact_float32_self_pair_count":len(pairs),
    "candidate_npz":str(npz),"candidate_npz_sha256":sha(npz),"exact_pair_sha256":sha(OUT/"exact-self-pairs.npy"),
    "poly_sha256":sha(poly),"tetgen_executed":False})
   if audit["complete"] and len(pairs)==0:
    cmd=[str(TETGEN),"-pYq1.4S512",poly.name]
    try:
     cp=subprocess.run(cmd,cwd=OUT,text=True,capture_output=True,timeout=45);ret,out,err,timed=cp.returncode,cp.stdout,cp.stderr,False
    except subprocess.TimeoutExpired as e:
     ret=None;out=e.stdout.decode(errors="replace") if isinstance(e.stdout,bytes) else (e.stdout or "")
     err=e.stderr.decode(errors="replace") if isinstance(e.stderr,bytes) else (e.stderr or "");timed=True
    report.update({"tetgen_executed":True,"tetgen_command":cmd,"tetgen_return_code":ret,"tetgen_stdout":out,
     "tetgen_stderr":err,"tetgen_timed_out":timed,"tetgen_binary_sha256":sha(TETGEN)})
    report["status"]="EXACT_CLOSED_CAP_PASSED_TETGEN_ATTEMPTED"
   else:report["status"]="CLOSED_CAP_REJECTED_OR_INCOMPLETE"
   report["admission"]="none; temporary caps/TetGen are construction-only, not physical skin qualification"
 except Exception as e:
  report.update({"status":"FAILED_CLOSED","failure_type":type(e).__name__,"failure":str(e)})
 report["elapsed_seconds"]=time.monotonic()-start
 (OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__":main()

