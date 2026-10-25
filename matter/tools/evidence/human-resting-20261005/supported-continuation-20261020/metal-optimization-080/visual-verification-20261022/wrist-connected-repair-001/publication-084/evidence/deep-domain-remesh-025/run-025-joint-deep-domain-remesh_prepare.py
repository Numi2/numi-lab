#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, pathlib, subprocess, time
from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

B=pathlib.Path("/Users/numi/Work/numi-human-connected-surface-069/connected-volume-extraction-157/true-o-foundation-159")
R=pathlib.Path("/Users/numi/Work/numi-human-connected-surface-069/whole-reference-tetgen-084-bounded")
OUT=B/"run-025-joint-deep-domain-remesh"/"attempt-001"
CP=B/"run-020-refined-snap-layer-extraction/deep-positive-retained-checkpoint.npz"
NEG=B/"run-020-refined-snap-layer-extraction/deep-positive-excluded-side-ledger.npz"
ALL=B/"run-020-refined-snap-layer-extraction/deep-positive-all-side-checkpoint.npz"
REF=B/"run-013-conforming-field-refinement/refined-foundation-checkpoint.npz"
TETGEN=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/arm-local-pocket-remesh-116/local-remesh-118/tetgen-v1.6.0/tetgen")
MAX_SECONDS=60
MAX_STEINER=10000
MAX_CELLS=1_000_000
TARGET_VOLUME=0.020009312767966532

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def need(ok,msg):
 if not ok: raise RuntimeError(msg)
def detq(P,T):
 X=np.asarray(P,dtype=np.float64)[np.asarray(T,dtype=np.int64)]
 a=X[:,1]-X[:,0]; b=X[:,2]-X[:,0]; c=X[:,3]-X[:,0]
 det=np.einsum("ij,ij->i",a,np.cross(b,c))
 longest2=np.zeros(len(T),dtype=np.float64)
 for i in range(4):
  for j in range(i+1,4):
   e=X[:,i]-X[:,j]; longest2=np.maximum(longest2,np.einsum("ij,ij->i",e,e))
 q=det/np.maximum(longest2,np.finfo(np.float64).tiny)**1.5
 return det,q
def face_rows(T):
 T=np.asarray(T,dtype=np.int64)
 # For a positive tetrahedron, these ordered faces point out of its interior.
 return np.stack((T[:,[1,2,3]],T[:,[0,3,2]],T[:,[0,1,3]],T[:,[0,2,1]]),axis=1).reshape(-1,3).astype(np.int32)
def groups_of_faces(F,owner):
 keys=np.sort(np.asarray(F,dtype=np.int32),axis=1)
 order=np.lexsort((keys[:,2],keys[:,1],keys[:,0]))
 ks=keys[order]; own=np.asarray(owner,dtype=np.int64)[order]
 starts=np.r_[0,1+np.flatnonzero(np.any(ks[1:]!=ks[:-1],axis=1))]
 ends=np.r_[starts[1:],len(order)]; counts=ends-starts
 return ks,own,order,starts,counts
def struct_keys(K):
 K=np.ascontiguousarray(K,dtype=np.int32)
 dtype=np.dtype([("a","<i4"),("b","<i4"),("c","<i4")])
 return K.view(dtype).reshape(-1)
def tet_boundary(P,T):
 F=face_rows(T); owners=np.repeat(np.arange(len(T),dtype=np.int64),4)
 keys,own,order,starts,counts=groups_of_faces(F,owners)
 if np.any(counts>2):
  raise RuntimeError(f"nonmanifold tetra face incidence: groups>2={int(np.count_nonzero(counts>2))}")
 ix=order[starts[counts==1]]
 return F[ix],owners[ix],keys[starts[counts==1]],keys[starts[counts==2]],own[starts[counts==2]],counts[starts==2]
def edge_components(F):
 n=len(F); E=np.concatenate((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]),axis=0)
 eo=np.tile(np.arange(n,dtype=np.int64),3); ek=np.sort(E,axis=1)
 order=np.lexsort((ek[:,1],ek[:,0])); es=ek[order]; eo=eo[order]
 starts=np.r_[0,1+np.flatnonzero(np.any(es[1:]!=es[:-1],axis=1))]
 ends=np.r_[starts[1:],len(order)]; counts=ends-starts
 if np.any(counts!=2):
  u,c=np.unique(counts,return_counts=True); detail={int(a):int(b) for a,b in zip(u,c)}
  raise RuntimeError(f"boundary surface is not closed 2-manifold by edges: {detail}")
 a=eo[starts]; b=eo[starts+1]
 graph=coo_matrix((np.ones(2*len(a),dtype=np.int8),(np.r_[a,b],np.r_[b,a])),shape=(n,n)).tocsr()
 nc,labels=connected_components(graph,directed=False,return_labels=True)
 return int(nc),labels

def parse_nodes(path):
 lines=path.read_text().splitlines(); head=next(i for i,l in enumerate(lines) if l.strip() and not l.lstrip().startswith("#"))
 n,dim,nattr,nmark=map(int,lines[head].split()[:4]); rows=[]
 for line in lines[head+1:]:
  s=line.strip()
  if not s or s.startswith("#"): continue
  v=s.split()
  if len(v)<4+ nattr+nmark: continue
  idx=int(v[0]); xyz=[float(x) for x in v[1:4]]; off=4+nattr
  mark=int(v[off]) if nmark else 0
  rows.append((idx,xyz,mark))
  if len(rows)==n: break
 need(len(rows)==n,f"TetGen node rows parsed {len(rows)} != {n}")
 ids=np.array([r[0] for r in rows],dtype=np.int64); X=np.array([r[1] for r in rows],dtype=np.float64); m=np.array([r[2] for r in rows],dtype=np.int64)
 return ids,X,m
def parse_eles(path,node_ids):
 lines=path.read_text().splitlines(); head=next(i for i,l in enumerate(lines) if l.strip() and not l.lstrip().startswith("#"))
 n,nodes,attrs=map(int,lines[head].split()[:3]); rows=[]
 for line in lines[head+1:]:
  s=line.strip()
  if not s or s.startswith("#"): continue
  v=s.split(); rows.append([int(x) for x in v[1:5]])
  if len(rows)==n: break
 need(len(rows)==n,f"TetGen elements parsed {len(rows)} != {n}")
 idmap={int(x):i for i,x in enumerate(node_ids)}
 T=np.array([[idmap[int(x)] for x in row] for row in rows],dtype=np.int64)
 return T

def main():
 OUT.mkdir(parents=True,exist_ok=True); t0=time.monotonic()
 report={"status":"STARTED","construction_only":True,"geometry_admitted":False,"physical_or_native_admission":False,"simulation_allowed":False,
  "method":"quality remesh of frozen retained deep-positive domain boundary (outer O plus D); negative excluded-side components supply TetGen holes",
  "time_limit_seconds":MAX_SECONDS,"max_steiner_points":MAX_STEINER,"max_cells":MAX_CELLS,
  "target_retained_volume_m3":TARGET_VOLUME,"tetgen_executed":False,"inputs":{}}
 rp=OUT/"report.json"; rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 try:
  for p in (CP,NEG,ALL,REF,TETGEN): need(p.is_file(),f"missing input {p}")
  expected={CP:"cb8f5637b8b51a28c4d4a392f3292072fa0cbc32cd1200fbad648b7ad3e30243",
   NEG:"be2c2f905b01ba789143a833295b1aebb052109c6a6f0740c00ae5b77e6dd7e9",
   ALL:"be19752bfc315cc92568d824abf0681da5c685dbede26e9272dc46c4e5ac49d9",
   REF:"cf51ebb33721a351be149f822f49f0737af136e66eb95a39924f10bbe6e3f179"}
  for p,h in expected.items(): need(sha(p)==h,f"input hash mismatch: {p}")
  report["inputs"]={str(p):sha(p) for p in (CP,NEG,ALL,REF)}; report["tetgen_binary_sha256"]=sha(TETGEN)
  with np.load(CP,allow_pickle=False) as z:C={k:z[k] for k in z.files}
  with np.load(NEG,allow_pickle=False) as z:N={k:z[k] for k in z.files}
  with np.load(ALL,allow_pickle=False) as z:A={k:z[k] for k in z.files}
  with np.load(REF,allow_pickle=False) as z:Q={k:z[k] for k in z.files}
  P=np.asarray(C["points_world_m"],np.float32); T=np.asarray(C["tetrahedra_zero_based"],np.int64)
  need(P.shape==(250855,3) and T.shape==(613806,4),f"retained input shape mismatch {P.shape},{T.shape}")
  need(np.all(T>=0) and np.all(T<len(P)),"retained tet node index out of range")
  need(len(C["registered_levelsets_m"])==len(P) and len(C["parent_foundation_tet_rows"])==len(T),"retained checkpoint array length mismatch")
  det,q=detq(P,T); need(np.all(det>0),f"retained mesh not positive in emitted F32: {int(np.count_nonzero(det<=0))}")
  input_v=float(np.sum(det,dtype=np.float64)/6); input_q={"min":float(q.min()),"p01":float(np.quantile(q,.01)),"p05":float(np.quantile(q,.05)),"median":float(np.median(q)),"lt_1e-5":int(np.count_nonzero(q<1e-5)),"lt_1e-4":int(np.count_nonzero(q<1e-4))}
  need(abs(input_v-TARGET_VOLUME)<=1e-10*TARGET_VOLUME,f"retained volume differs from pinned target: {input_v}")
  need(np.array_equal(P[:len(Q["points_world_m"])],np.asarray(Q["points_world_m"],np.float32)),"run020 original/refined point prefix differs from run013 checkpoint")
  need(np.array_equal(N["excluded_tetrahedra"],A["tetrahedra_zero_based"][N["excluded_child_rows"]]),"excluded-side tetrahedra disagree with all-side checkpoint")
  need(np.all(N["excluded_side_signs"]==-1) and np.all(A["side_signs"][N["excluded_child_rows"]]==-1),"excluded ledger contains nonnegative side labels")
  # Check the run013 original boundary and exact O/cap identities.
  EF=np.asarray(Q["boundary_faces_zero_based"],np.int32); O=np.asarray(Q["source_O_faces"],np.int32); CAP=np.asarray(Q["temporary_cap_faces"],np.int32)
  caps=np.asarray(Q["boundary_face_is_temporary_cap"],bool)
  need(EF.ndim==2 and EF.shape[1]==3 and len(caps)==len(EF),"run013 boundary schema mismatch")
  exp_lookup={tuple(sorted(map(int,f))):i for i,f in enumerate(EF)}
  o_lookup={tuple(sorted(map(int,f))):i for i,f in enumerate(O)}
  cap_lookup={tuple(sorted(map(int,f))):i for i,f in enumerate(CAP)}
  BF,BOwner,bkeys,negkeys,negowners,negcounts=tet_boundary(P,T)
  nb=len(BF)
  expected_idx=np.array([exp_lookup.get(tuple(sorted(map(int,f))),-1) for f in BF],dtype=np.int64)
  matched=expected_idx>=0
  Orow=np.array([o_lookup.get(tuple(sorted(map(int,f))),-1) for f in BF],dtype=np.int64)
  Crow=np.array([cap_lookup.get(tuple(sorted(map(int,f))),-1) for f in BF],dtype=np.int64)
  kind=np.full(nb,2,dtype=np.int8); source=np.full(nb,-1,dtype=np.int32); marker=np.zeros(nb,dtype=np.int32)
  for i in range(nb):
   if Orow[i]>=0: kind[i]=0;source[i]=Orow[i];marker[i]=1+int(Orow[i])
   elif Crow[i]>=0: kind[i]=1;source[i]=Crow[i];marker[i]=1_000_000+int(Crow[i])
   elif expected_idx[i]>=0 and bool(caps[expected_idx[i]]):
    kind[i]=1;source[i]=int(Q["boundary_face_shell_rows"][expected_idx[i]]);marker[i]=1_000_000+int(source[i])
   elif expected_idx[i]>=0 and not bool(caps[expected_idx[i]]):
    raise RuntimeError(f"run013 outer boundary face unmatched to source_O_faces: boundary row {int(expected_idx[i])}")
  # Every required original O facet must remain an exact retained boundary facet.
  got_o={tuple(sorted(map(int,f))) for f in BF[Orow>=0]}
  missing_o=[i for i,f in enumerate(O) if tuple(sorted(map(int,f))) not in got_o]
  need(not missing_o,f"retained positive domain is missing {len(missing_o)} original O PLC faces; first={missing_o[:12]}")
  # Every source temporary cap present in retained boundary is labeled separately.
  missing_caps=[i for i,f in enumerate(CAP) if tuple(sorted(map(int,f))) not in {tuple(sorted(map(int,g))) for g in BF}]
  # Closed boundary edge incidence and surface components.
  nbc,bcomp=edge_components(BF)
  boundary_vol=float(np.einsum("ij,ij->i",P[BF[:,0]].astype(np.float64),np.cross(P[BF[:,1]].astype(np.float64),P[BF[:,2]].astype(np.float64))).sum()/6)
  need(abs(boundary_vol-input_v)<=2e-10*input_v,f"oriented boundary volume mismatch {boundary_vol} vs {input_v}")
  # Negative excluded domain components from actual excluded child tets; each D interface component gets one seed in its most negative source cell.
  NT=np.asarray(N["excluded_tetrahedra"],np.int64); nf=face_rows(NT); no=np.repeat(np.arange(len(NT),dtype=np.int64),4)
  nk,nn,ni,ns,nc=groups_of_faces(nf,no)
  if np.any(nc>2): raise RuntimeError(f"excluded-side nonmanifold face incidence groups={int(np.count_nonzero(nc>2))}")
  pair_starts=ns[nc==2]; ordern=ni
  pn=np.column_stack((nn[pair_starts],nn[pair_starts+1])) if len(pair_starts) else np.empty((0,2),dtype=np.int64)
  graph=coo_matrix((np.ones(2*len(pn),dtype=np.int8),(np.r_[pn[:,0],pn[:,1]],np.r_[pn[:,1],pn[:,0]])),shape=(len(NT),len(NT))).tocsr()
  nneg,labels=connected_components(graph,directed=False,return_labels=True)
  # Map retained D boundary triangles to their exact negative-side incident child cell and component.
  negb_start=ns[nc==1]; negb_keys=nk[negb_start]; negb_owner=nn[negb_start]
  drows=np.flatnonzero(kind==2); dkeys=np.sort(BF[drows],axis=1).astype(np.int32)
  if len(drows):
   pos=np.searchsorted(struct_keys(negb_keys),struct_keys(dkeys))
   valid=(pos<len(negb_keys))
   valid[valid]=np.all(negb_keys[pos[valid]]==dkeys[valid],axis=1)
   need(np.all(valid),f"{int(np.count_nonzero(~valid))} retained D-boundary facets have no matching excluded-side face")
   dtet=negb_owner[pos]; dcomp=labels[dtet]
  else:
   dtet=np.empty(0,dtype=np.int64);dcomp=np.empty(0,dtype=np.int64)
  dcount=np.bincount(dcomp,minlength=nneg) if len(dcomp) else np.zeros(nneg,dtype=np.int64)
  # Identify negative components touching original O or temporary caps; keep this explicit for the head-connected exterior review.
  negb_expected=np.array([exp_lookup.get(tuple(map(int,k)),-1) for k in negb_keys],dtype=np.int64)
  touch_o=np.zeros(nneg,dtype=np.int64);touch_cap=np.zeros(nneg,dtype=np.int64)
  for j,ei in enumerate(negb_expected):
   if ei>=0:
    comp=int(labels[negb_owner[j]])
    if caps[ei]:touch_cap[comp]+=1
    else:touch_o[comp]+=1
  negphi=np.asarray(A["scalar_values_m"],np.float64)[NT]
  meanphi=negphi.mean(axis=1)
  seed_ids=[];seed_points=[];seed_child=[];comp_rows=[]
  for c in range(nneg):
   ids=np.flatnonzero(labels==c)
   if not len(ids): continue
   best=int(ids[np.argmin(meanphi[ids])])
   if dcount[c]>0:
    seed_ids.append(c); seed_child.append(best); seed_points.append(P[NT[best]].astype(np.float64).mean(axis=0))
   comp_rows.append({"component":int(c),"excluded_tet_count":int(len(ids)),"volume_m3":float(np.sum(np.abs(detq(P,NT[ids])[0]),dtype=np.float64)/6),
    "deepest_cell":best,"deepest_cell_mean_phiD_m":float(meanphi[best]),"D_interface_face_count":int(dcount[c]),"touches_original_O_face_count":int(touch_o[c]),"touches_temporary_cap_face_count":int(touch_cap[c])})
  need(len(seed_points)>0,"no excluded negative component has a D interface; hole seed set would be empty")
  seeds=np.asarray(seed_points,dtype=np.float64)
  need(np.all(np.isfinite(seeds)),"nonfinite hole seed")
  need(np.all(np.asarray([meanphi[i] for i in seed_child])<0),"hole seed source cell centroid is not in negative phiD")
  # Verify boundary is closed, all O facets exact, D facets have excluded-side mates, and hole seeds are sourced from negative-domain cells.
  report.update({"status":"PREFLIGHT_PASS","retained_point_count":int(len(P)),"retained_tet_count":int(len(T)),"retained_emitted_f32_volume_m3":input_v,"retained_emitted_f32_q":input_q,
   "retained_boundary_face_count":int(nb),"retained_boundary_vertex_count":int(len(np.unique(BF))),"retained_boundary_component_count":int(nbc),
   "source_run013_boundary_face_count":int(len(EF)),"source_O_face_count":int(len(O)),"source_O_faces_exactly_present":True,
   "temporary_cap_face_count":int(len(CAP)),"temporary_caps_missing_from_retained_boundary":int(len(missing_caps)),"boundary_faces_matching_run013_original_boundary":int(matched.sum()),
   "D_interface_face_count":int(len(drows)),"D_interface_faces_matched_to_excluded_negative_cells":int(len(dtet)),"oriented_retained_boundary_volume_m3":boundary_vol,
   "excluded_negative_child_count":int(len(NT)),"excluded_negative_connected_component_count":int(nneg),"hole_seed_count":int(len(seeds)),"hole_seeds_from_components_with_D_boundary_only":True,
   "negative_components":comp_rows,"components_touching_temp_caps_count":int(np.count_nonzero(touch_cap)),"components_touching_original_O_count":int(np.count_nonzero(touch_o)),
   "warning":"Temporary cap and head-connected/exterior contact counts are reported per excluded component. This is a construction-only TetGen seed classification, not a biological tissue boundary claim."})
  # Save the exact oriented PLC boundary and source maps used below.
  boundary_vertex_rows=np.unique(BF)
  vmap=np.full(len(P),-1,dtype=np.int32);vmap[boundary_vertex_rows]=np.arange(len(boundary_vertex_rows),dtype=np.int32)
  localBF=vmap[BF]
  dorder=np.flatnonzero(kind==2)
  for k,bi in enumerate(dorder): marker[bi]=2_000_000+k
  need(np.all(marker>0),"unassigned boundary marker")
  np.savez_compressed(OUT/"boundary-preflight.npz",boundary_vertex_rows=boundary_vertex_rows.astype(np.int32),boundary_faces_zero_based=BF.astype(np.int32),boundary_face_parent_tet_rows=BOwner.astype(np.int32),boundary_face_kind=kind,boundary_face_source_rows=source,boundary_face_markers=marker,
   hole_seed_points_m=seeds,hole_component_ids=np.asarray(seed_ids,dtype=np.int32),hole_source_excluded_child_rows=np.asarray([N["excluded_child_rows"][Nidx] for Nidx in seed_child],dtype=np.int32),hole_source_excluded_tet_rows=np.asarray(seed_child,dtype=np.int32),
   retained_volume_m3=np.array([input_v],dtype=np.float64))
  # Write a small, provenance-pinned PLC: only boundary vertices are sent to TetGen; interior refinement points are intentionally omitted.
  poly=OUT/"retained-deep-domain.poly"
  with poly.open("w") as f:
   f.write(f"{len(boundary_vertex_rows)} 3 0 1\n")
   for local,old in enumerate(boundary_vertex_rows,1):
    x=P[int(old)]; f.write(f"{local} {float(x[0]):.9g} {float(x[1]):.9g} {float(x[2]):.9g} {int(old)+1}\n")
   f.write(f"{len(BF)} 1\n")
   for tri,m in zip(localBF,marker):
    f.write(f"1 0 {int(m)}\n3 {int(tri[0])+1} {int(tri[1])+1} {int(tri[2])+1}\n")
   f.write(f"{len(seeds)}\n")
   for i,x in enumerate(seeds,1): f.write(f"{i} {x[0]:.17g} {x[1]:.17g} {x[2]:.17g}\n")
   f.write("0\n")
  report["poly_path"]=str(poly);report["poly_sha256"]=sha(poly);report["boundary_preflight_npz_sha256"]=sha(OUT/"boundary-preflight.npz")
  report["poly_point_count"]=int(len(boundary_vertex_rows));report["poly_facet_count"]=int(len(BF));report["poly_hole_count"]=int(len(seeds)); report["status"]="TETGEN_RUNNING"
  rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
  prefix=OUT/"deep-remesh"; cmd=[str(TETGEN),"-pYq1.4S10000",str(poly)]
  start=time.monotonic(); timed=False
  try:
   cp=subprocess.run(cmd,cwd=OUT,capture_output=True,text=True,timeout=MAX_SECONDS)
  except subprocess.TimeoutExpired as e:
   timed=True; cp=None; stdout=e.stdout or ""; stderr=e.stderr or ""; rc=None
  else: stdout=cp.stdout;stderr=cp.stderr;rc=cp.returncode
  (OUT/"tetgen.stdout.txt").write_text(stdout);(OUT/"tetgen.stderr.txt").write_text(stderr)
  report.update({"tetgen_executed":True,"tetgen_command":cmd,"tetgen_return_code":rc,"tetgen_timed_out":timed,"tetgen_elapsed_seconds":time.monotonic()-start,"tetgen_stdout_sha256":sha(OUT/"tetgen.stdout.txt"),"tetgen_stderr_sha256":sha(OUT/"tetgen.stderr.txt")})
  nodefile=OUT/"retained-deep-domain.1.node"; elefile=OUT/"retained-deep-domain.1.ele"; facefile=OUT/"retained-deep-domain.1.face"
  if timed or rc!=0 or not nodefile.exists() or not elefile.exists():
   report["status"]="TETGEN_FAILED_OR_TIMED_OUT"; report["exit_or_timeout"]={"timed_out":timed,"return_code":rc,"node_exists":nodefile.exists(),"ele_exists":elefile.exists()}; rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); return
  nodeids,X64,nodemarks=parse_nodes(nodefile); Tout=parse_eles(elefile,nodeids)
  need(len(Tout)<=MAX_CELLS,f"TetGen output cell cap exceeded: {len(Tout)}")
  need(len(X64)-len(boundary_vertex_rows)<=MAX_STEINER,f"TetGen Steiner cap exceeded: {len(X64)-len(boundary_vertex_rows)}")
  X32=X64.astype(np.float32);raw_det,raw_q=detq(X32,Tout)
  neg_orientation=int(np.count_nonzero(raw_det<0));zero=int(np.count_nonzero(raw_det==0))
  Tnorm=Tout.copy();flip=raw_det<0;Tnorm[flip]=Tnorm[flip][:,[1,0,2,3]]
  det32,q32=detq(X32,Tnorm);det64,q64=detq(X64,Tnorm)
  need(np.all(det32>0),f"TetGen emitted F32 nonpositive cells after orientation normalization: {int(np.count_nonzero(det32<=0))}")
  vol32=float(np.sum(det32,dtype=np.float64)/6);vol64=float(np.sum(det64,dtype=np.float64)/6)
  # Exact source-O and input-facet checks via TetGen point/facet markers.
  expected_point_markers=boundary_vertex_rows.astype(np.int64)+1
  point_marker_counts={int(x):int(np.count_nonzero(nodemarks==x)) for x in expected_point_markers}
  bad_point_markers={k:v for k,v in point_marker_counts.items() if v!=1}
  need(not bad_point_markers,f"lost or duplicated input boundary point markers: {list(bad_point_markers.items())[:12]}")
  output_input_rows=np.full(len(X64),-1,dtype=np.int32)
  for i,m in enumerate(nodemarks):
   if int(m)>0 and int(m)-1<len(vmap) and vmap[int(m)-1]>=0: output_input_rows[i]=int(m)-1
  output_facet_markers={}
  if facefile.exists():
   fl=facefile.read_text().splitlines(); h=next(i for i,l in enumerate(fl) if l.strip() and not l.lstrip().startswith("#"));nf,nmark=map(int,fl[h].split()[:2]);
   for line in fl[h+1:]:
    s=line.strip()
    if not s or s.startswith("#"):continue
    a=s.split()
    if len(a)<5:continue
    m=int(a[4]) if nmark else 0
    if m:output_facet_markers[m]=output_facet_markers.get(m,0)+1
    if len(output_facet_markers)>0 and sum(output_facet_markers.values())>=nf:break
  missing_markers=[int(m) for m in marker if output_facet_markers.get(int(m),0)!=1]
  # Map every O point/facet source ID through emitted node/facet markers.
  o_markers=marker[kind==0]; missing_o_markers=[int(m) for m in o_markers if output_facet_markers.get(int(m),0)!=1]
  # Mesh boundary closure independent of .face parser.
  outBF,_,_,_,_,_=tet_boundary(X32,Tnorm); outbc,outbc_labels=edge_components(outBF)
  np.savez_compressed(OUT/"retained-deep-domain-remesh.npz",points_world_m=X32,points_world_m_f64=X64,tetrahedra_zero_based=Tnorm.astype(np.int32),raw_tetrahedra_zero_based=Tout.astype(np.int32),orientation_flipped_rows=np.flatnonzero(flip).astype(np.int32),material_ids=np.zeros(len(Tnorm),dtype=np.int8),
   output_node_ids=nodeids.astype(np.int64),output_node_markers=nodemarks.astype(np.int32),output_input_point_rows=output_input_rows,input_boundary_vertex_rows=boundary_vertex_rows.astype(np.int32),boundary_faces_zero_based=outBF.astype(np.int32),input_boundary_face_markers=marker.astype(np.int32),input_boundary_face_kind=kind,input_boundary_face_source_rows=source,
   input_boundary_face_parent_tet_rows=BOwner.astype(np.int32),hole_seed_points_m=seeds,hole_component_ids=np.asarray(seed_ids,dtype=np.int32),hole_source_excluded_child_rows=np.asarray([N["excluded_child_rows"][Nidx] for Nidx in seed_child],dtype=np.int32),
   original_O_faces=O.astype(np.int32),temporary_cap_faces=CAP.astype(np.int32),construction_only=np.array([True]))
  report.update({"status":"TETGEN_EMITTED_AUDIT_PENDING","output_node_count":int(len(X64)),"output_tet_count":int(len(Tnorm)),"output_steiner_count":int(len(X64)-len(boundary_vertex_rows)),"output_boundary_component_count":int(outbc),
   "raw_negative_orientation_count":neg_orientation,"raw_zero_det_f32_count":zero,"orientation_normalized_rows":int(flip.sum()),"emitted_f32_nonpositive_count":int(np.count_nonzero(det32<=0)),"emitted_f32_q":{"min":float(q32.min()),"p01":float(np.quantile(q32,.01)),"p05":float(np.quantile(q32,.05)),"median":float(np.median(q32)),"lt_1e-5":int(np.count_nonzero(q32<1e-5)),"lt_1e-4":int(np.count_nonzero(q32<1e-4))},
   "emitted_f64_volume_m3":vol64,"emitted_f32_volume_m3":vol32,"volume_delta_vs_retained_m3":vol32-input_v,"volume_delta_relative":(vol32-input_v)/input_v,
   "output_face_marker_count":len(output_facet_markers),"input_boundary_facets_missing_or_nonunique_in_output":len(missing_markers),"missing_boundary_marker_sample":missing_markers[:20],"original_O_facets_missing_or_nonunique":len(missing_o_markers),"original_O_marker_sample":missing_o_markers[:20],
   "output_npz_path":str(OUT/"retained-deep-domain-remesh.npz"),"output_npz_sha256":sha(OUT/"retained-deep-domain-remesh.npz"),"elapsed_seconds":time.monotonic()-t0})
  if not np.all(det32>0) or abs(vol32-input_v)>2e-8*input_v or outbc!=nbc or missing_o_markers or missing_markers:
   report["status"]="REJECTED_OUTPUT_GATE"
  else: report["status"]="REMESH_GEOMETRY_GATES_PASS_CONSTRUCTION_ONLY"
  rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 except Exception as e:
  report["status"]="PREFLIGHT_OR_RUN_FAILURE";report["failure_type"]=type(e).__name__;report["failure_message"]=str(e);report["elapsed_seconds"]=time.monotonic()-t0
  rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
  raise
 finally:
  if rp.exists():
   report["report_sha256_before_final_write"]=sha(rp)

if __name__ == "__main__":
 main()
