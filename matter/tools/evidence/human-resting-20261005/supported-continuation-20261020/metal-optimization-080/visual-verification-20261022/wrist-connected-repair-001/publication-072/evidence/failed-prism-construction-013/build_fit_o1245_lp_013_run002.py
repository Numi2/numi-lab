#!/usr/bin/env python3
"""O1245 direct-source star, annulus, and corrected coned O/I/D cavity diagnostic."""
from __future__ import annotations
import hashlib,json,socket,sys,time
from fractions import Fraction
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent.parent; WORK=HERE.parent; OUT=HERE/"source-star-013-run-002"
sys.path[:0]=[str(WORK),str(WORK/"reference-fit-layer-counterfactual-001"),str(HERE)]
import reference_frame_mapping as ref
import connected_interior_reference_fit as fit
from retriangulate_outer_target_patches import directed_boundary

BASE=HERE/"retessellation-001/volume-cavity-001/oid-cavity-replacement.npz"
STAR=HERE/"source-star-002/report.json"
LAYER=WORK/"reference-fit-layer-counterfactual-001/run-001/whole-layer-reference-fit.npz"
N=2411
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def dets(x,t):
 q=np.asarray(x,float)[np.asarray(t,int)]
 return np.linalg.det(np.stack((q[:,1]-q[:,0],q[:,2]-q[:,0],q[:,3]-q[:,0]),axis=2))
def perm_sign(v):
 ix=[sorted(v).index(x) for x in v]
 inv=sum(ix[i]>ix[j] for i in range(len(ix)) for j in range(i+1,len(ix)))
 return -1 if inv%2 else 1
def positive(t,x):
 out=np.asarray(t,dtype=np.int64).copy();d=dets(x,out);bad=d<0
 if bad.any():out[bad,2],out[bad,3]=out[bad,3].copy(),out[bad,2].copy()
 return out
def boundary_chain(t,x):
 acc={}
 for a,b,c,d in positive(t,x):
  for tri in ((b,c,d),(a,d,c),(a,b,d),(a,c,b)):
   k=tuple(sorted(map(int,tri)));acc[k]=acc.get(k,0)+perm_sign(tri)
 return {k:v for k,v in acc.items() if v}
def directed_cycle(f):
 _,edges=directed_boundary(np.asarray(f,dtype=np.int64));out={}
 for a,b in edges:
  if int(a) in out:raise ValueError("branched directed boundary")
  out[int(a)]=int(b)
 if not out:raise ValueError("empty boundary")
 start=min(out);cy=[start];cur=out[start]
 while cur!=start:
  if cur in cy or cur not in out or len(cy)>len(out):raise ValueError("boundary not one cycle")
  cy.append(cur);cur=out[cur]
 if len(cy)!=len(edges):raise ValueError("multiple boundary cycles")
 return cy
def closest_triangle(p,a,b,c):
 ab=b-a;ac=c-a;ap=p-a;d1=float(ab@ap);d2=float(ac@ap)
 if d1<=0 and d2<=0:return np.array([1.,0,0]),a
 bp=p-b;d3=float(ab@bp);d4=float(ac@bp)
 if d3>=0 and d4<=d3:return np.array([0.,1,0]),b
 vc=d1*d4-d3*d2
 if vc<=0 and d1>=0 and d3<=0:
  v=d1/(d1-d3);return np.array([1-v,v,0]),a+v*ab
 cp=p-c;d5=float(ab@cp);d6=float(ac@cp)
 if d6>=0 and d5<=d6:return np.array([0.,0,1]),c
 vb=d5*d2-d1*d6
 if vb<=0 and d2>=0 and d6<=0:
  w=d2/(d2-d6);return np.array([1-w,0,w]),a+w*ac
 va=d3*d6-d5*d4
 if va<=0 and d4-d3>=0 and d5-d6>=0:
  w=(d4-d3)/((d4-d3)+(d5-d6));return np.array([0,1-w,w]),b+w*(c-b)
 den=1/(va+vb+vc);v=vb*den;w=vc*den;bary=np.array([1-v-w,v,w])
 return bary,bary@np.stack((a,b,c))
def project(p,faces,points):
 best=None
 for i,t in enumerate(faces):
  b,q=closest_triangle(p,*points[t]);d=float(np.linalg.norm(p-q))
  if best is None or d<best[0]:best=(d,i,t,b,q)
 return best
def tri_cost(x,t):
 q=x[np.asarray(t,int)];e=np.array([np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[1]),np.linalg.norm(q[0]-q[2])]);a=np.linalg.norm(np.cross(q[1]-q[0],q[2]-q[0]))
 if a<=1e-15:return float("inf")
 return float((e@e)/a+1e-3*(e@e)/(0.01**2))
def annulus_candidates(outer,inner,x):
 m,n=len(outer),len(inner);result=[]
 for phase in range(n):
  inn=inner[phase:]+inner[:phase];dp={(0,0):(0.,[])}
  for i in range(m+1):
   for j in range(n+1):
    if (i,j) not in dp:continue
    score,path=dp[(i,j)]
    for ni,nj,tri in ((i+1,j,(outer[i%m],outer[(i+1)%m],inn[j%n])) if i<m else (None,None,None),
                      (i,j+1,(outer[i%m],inn[(j+1)%n],inn[j%n])) if j<n else (None,None,None)):
     if ni is None:continue
     c=tri_cost(x,tri)
     if not np.isfinite(c):continue
     key=(ni,nj);v=(score+c,path+[tri])
     if key not in dp or v[0]<dp[key][0]:dp[key]=v
  if (m,n) in dp:result.append({"phase":phase,"cost":dp[(m,n)][0],"faces":np.asarray(dp[(m,n)][1],dtype=np.int64)})
 return result
def edge_counts(f):
 d={}
 for t in np.asarray(f,int):
  for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
   e=tuple(sorted((int(a),int(b))));d[e]=d.get(e,0)+1
 return d
def cavity_rows(tets,faces,face_ids,layer):
 rows=[]
 for fid in face_ids:
  cols=set(map(int,faces[int(fid)]));allowed={c+k*N for c in cols for k in (layer,layer+1)}
  hit=[i for i,t in enumerate(tets) if set(map(int,t)).issubset(allowed)]
  if len(hit)!=3:raise ValueError(f"old face {fid} layer {layer} has {len(hit)} cells")
  rows+=hit
 if len(set(rows))!=len(rows):raise ValueError("overlapping cavity rows")
 return sorted(rows)
def side(chain,cycle,layer):
 d={}
 for i,u in enumerate(cycle):
  v=cycle[(i+1)%len(cycle)];q={layer*N+u,layer*N+v,(layer+1)*N+u,(layer+1)*N+v}
  hit={k:s for k,s in chain.items() if set(k)<=q}
  if len(hit)!=2:raise ValueError(f"side quad {(u,v)} layer{layer} has {len(hit)} facets")
  d.update(hit)
 return d
def old_diag(sidechain,u,v,layer):
 q={layer*N+u,layer*N+v,(layer+1)*N+u,(layer+1)*N+v}
 faces=[set(k) for k in sidechain if set(k)<=q]
 if len(faces)!=2 or len(faces[0]&faces[1])!=2:raise ValueError("bad old side diagonal")
 a,b=map(int,faces[0]&faces[1])
 lo,hi=(a,b) if a//N==layer else (b,a)
 return (lo%N,hi%N)
def node(layer,col):
 if col<N:return layer*N+col
 return 7233+layer*6+(col-N)
def face_at_layer(faces,L):return np.asarray([[node(L,c) for c in tri] for tri in faces],int)
def cap_rel(chain,faces,layer):
 vals=[]
 for tri in faces:
  ids=tuple(node(layer,int(c)) for c in tri);k=tuple(sorted(ids));vals.append(chain[k]*perm_sign(ids))
 if len(set(vals))!=1:raise ValueError("cap orientation inconsistent")
 return vals[0]
def prism_cells(la,lb,lc,ha,hb,hc):
 # Standard oriented 3-tet split of a triangular prism. The middle tet
 # must use [la,hb,lb,hc]; [la,lb,hb,hc] has the opposite orientation.
 return np.asarray([[la,lb,lc,hc],[la,hb,lb,hc],[la,ha,hb,hc]],dtype=np.int64)
def prism_order_candidates(tri,L,Xr,Xc):
 import itertools
 out=[]
 for order in itertools.permutations(map(int,tri)):
  a,b,c=order;la,lb,lc=node(L,a),node(L,b),node(L,c);ha,hb,hc=node(L+1,a),node(L+1,b),node(L+1,c)
  t=prism_cells(la,lb,lc,ha,hb,hc)
  dr=dets(Xr,t);dc=dets(Xc,t)
  if np.any(dr<=1e-24):continue
  edges={}
  for i,u in enumerate(order):
   for v in order[i+1:]:edges[tuple(sorted((u,v)))]=(u,v)
  scale=max(float(np.linalg.norm(Xr[u]-Xr[v])) for u,v in ((la,lb),(lb,lc),(lc,la),(ha,hb),(hb,hc),(hc,ha)))
  score=(-float(np.min(dr))/(scale**3+1e-300),-float(np.min(dc))/(scale**3+1e-300))
  out.append({"order":order,"cells":t,"directions":edges,"score":score,
              "reference_determinants":dr.tolist(),"target_determinants":dc.tolist()})
 return out
def solve_prism_orders(tris,L,Xr,Xc,outer_edges,sidechain,outer_cycle,max_nodes=100000):
 domains=[]
 for fi,tri in enumerate(tris):
  opts=prism_order_candidates(tri,L,Xr,Xc)
  if not opts:raise ValueError(f"surface triangle {fi} has no all-positive reference 3-tet prism decomposition")
  domains.append(opts)
 incidence={}
 for fi,tri in enumerate(tris):
  for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
   incidence.setdefault(tuple(sorted((int(a),int(b)))),[]).append(fi)
 if any(len(v)>2 for v in incidence.values()):raise ValueError("patch edge has >2 incident surface faces")
 fixed={}
 for e in outer_edges:
  u,v=next((outer_cycle[i],outer_cycle[(i+1)%len(outer_cycle)]) for i in range(len(outer_cycle)) if tuple(sorted((outer_cycle[i],outer_cycle[(i+1)%len(outer_cycle)])))==e)
  fixed[e]=old_diag(sidechain,u,v,L)
 def propagate(ds):
  changed=True
  while changed:
   changed=False
   for e,faces in incidence.items():
    dirs=None
    for fi in faces:
     avail={o["directions"][e] for o in ds[fi] if e in o["directions"]}
     dirs=avail if dirs is None else dirs&avail
    if e in fixed:dirs=dirs&{fixed[e]}
    if not dirs:return None
    for fi in faces:
     nd=[o for o in ds[fi] if o["directions"].get(e) in dirs]
     if not nd:return None
     if len(nd)!=len(ds[fi]):ds[fi]=nd;changed=True
  return ds
 visits=0
 def search(ds):
  nonlocal visits
  visits+=1
  if visits>max_nodes:return None
  ds=propagate(ds)
  if ds is None:return None
  unresolved=[i for i,d in enumerate(ds) if len(d)>1]
  if not unresolved:return [d[0] for d in ds]
  fi=min(unresolved,key=lambda i:len(ds[i]))
  for opt in sorted(ds[fi],key=lambda o:o["score"]):
   nxt=[list(d) for d in ds];nxt[fi]=[opt]
   sol=search(nxt)
   if sol is not None:return sol
  return None
 sol=search([list(d) for d in domains])
 if sol is None:raise ValueError(f"no compatible 3-tet prism choices; search_nodes={visits}")
 return sol,visits
def main():
 t0=time.time();OUT.mkdir(parents=True,exist_ok=False)
 z=np.load(BASE);sr=json.load(open(STAR));lf=np.load(LAYER);sk=ref.load_skin(fit.DETAIL_SKIN)
 Xr0=np.asarray(z["reference_points_AP_m"],float);Xc0=np.asarray(z["target_points_AP_m"],float)
 T0=np.asarray(z["candidate_tetrahedra"],int);M0=np.asarray(z["candidate_material_ids"],int);F0=np.asarray(z["candidate_O_faces"],int)
 if Xr0.shape!=(7233,3) or Xc0.shape!=(7233,3) or T0.shape!=(28632,4):raise ValueError("unexpected base dimensions")
 centercol=1245;faceids=np.flatnonzero(np.any(F0==centercol,axis=1));oldstar=F0[faceids]
 if len(faceids)!=18:raise ValueError(f"O1245 star count changed: {len(faceids)}")
 source_faces=np.asarray(sk["faces"],int);source_center=int(sr["target"]["source_vertex_id"]);detailed=np.asarray(lf["source_current_AP_world_m"],float)
 source_star_ids=np.flatnonzero(np.any(source_faces==source_center,axis=1));source_star=source_faces[source_star_ids]
 if len(source_star_ids)!=6:raise ValueError("detailed source star is not six triangles")
 outer_cycle=directed_cycle(oldstar);inner_cycle=directed_cycle(source_star)
 pre=np.load(HERE/"run-003/outer-target-preflight.npz");direct=np.asarray(pre["source_vertex_ids"],np.uint32)
 source_to_col={int(s):i for i,s in enumerate(direct) if int(s)!=np.iinfo(np.uint32).max}
 if source_to_col.get(source_center)!=centercol:raise ValueError("center source mapping mismatch")
 if any(int(s) in source_to_col for s in inner_cycle):raise ValueError("inner direct source vertex already has a coarse column")
 ring_cols={int(s):N+i for i,s in enumerate(inner_cycle)};source_to_col.update(ring_cols)
 srcstar_cols=np.asarray([[source_to_col[int(s)] for s in tri] for tri in source_star],int)
 ring_xyz=np.asarray([detailed[int(s)] for s in inner_cycle],float)
 # Reference ring positions use closest-point P1 projection onto the original O-star; target O is exact current294.
 refO=[];refI=[];refD=[];curO=[];curI=[];curD=[];projection=[]
 for sid,p in zip(inner_cycle,ring_xyz):
  dr,ir,tr,br,qr=project(p,oldstar,Xr0[:N]);dc,ic,tc,bc,qc=project(p,oldstar,Xc0[:N])
  # Keep the detailed material O coordinate in the reference and current maps;
  # transfer only the existing local column offsets through parent barycentrics.
  ro=p;ri=ro+br@(Xr0[tr+N]-Xr0[tr]);rd=ri+br@(Xr0[tr+2*N]-Xr0[tr+N])
  co=p;ci=co+bc@(Xc0[tc+N]-Xc0[tc]);cd=ci+bc@(Xc0[tc+2*N]-Xc0[tc+N])
  refO.append(ro);refI.append(ri);refD.append(rd);curO.append(co);curI.append(ci);curD.append(cd)
  projection.append({"source_vertex_id":int(sid),"reference_parent_face":int(faceids[ir]),"reference_bary":br.tolist(),"reference_projection_residual_m":dr,
    "target_parent_face":int(faceids[ic]),"target_bary":bc.tolist(),"target_projection_residual_m":dc})
 refextra=np.vstack((refO,refI,refD));curextra=np.vstack((curO,curI,curD))
 Xr=np.vstack((Xr0,refextra));Xc=np.vstack((Xc0,curextra))
 target_surface=np.vstack((Xc0[:N],ring_xyz)).astype(np.float32).astype(np.float64)
 trials=annulus_candidates(outer_cycle,[ring_cols[int(s)] for s in inner_cycle],target_surface)
 if len(trials)!=6:raise ValueError(f"expected six zipper phases; found {len(trials)}")
 patchtries=[];pred=fit.load_exact_predicate();Fbase=F0
 counts0=edge_counts(Fbase);boundary0={e for e,n in counts0.items() if n==1}
 for tr in trials:
  patch=np.vstack((srcstar_cols,tr["faces"]))
  if len(patch)!=30:raise ValueError("patch must contain 6 source and 24 annulus faces")
  oriented={}
  for tri in patch:
   for aa,bb in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
    key=tuple(sorted((int(aa),int(bb))));oriented.setdefault(key,[]).append((int(aa),int(bb)))
  same_direction=[{"edge":k,"directions":v} for k,v in oriented.items() if len(v)==2 and v[0]==v[1]]
  if same_direction:raise ValueError(f"patch has same-oriented internal edges: {same_direction[:30]}")
  if set(directed_boundary(patch)[1])!=set(directed_boundary(oldstar)[1]):
   patchtries.append({"phase":tr["phase"],"cost":tr["cost"],"boundary_ok":False});continue
  F=Fbase.copy();F[faceids]=patch[:18];F=np.vstack((F,patch[18:]))
  cnt=edge_counts(F);bdy={e for e,n in cnt.items() if n==1};nonman=[e for e,n in cnt.items() if n>2]
  if bdy!=boundary0 or nonman:
   patchtries.append({"phase":tr["phase"],"cost":tr["cost"],"boundary_ok":True,"global_manifold_ok":False});continue
  verts=tuple(tuple(Fraction.from_float(float(x)) for x in row) for row in target_surface)
  rec=pred._records(verts,F.tolist());changed=set(map(int,faceids))|set(range(len(Fbase),len(F)))
  aud=pred._audit_self_pairs_touching_faces(rec,changed)
  area=.5*np.linalg.norm(np.cross(target_surface[F[:,1]]-target_surface[F[:,0]],target_surface[F[:,2]]-target_surface[F[:,0]]),axis=1)
  patchtries.append({"phase":tr["phase"],"cost":float(tr["cost"]),"boundary_ok":True,"global_manifold_ok":True,
    "exact_pair_count":int(aud["count"]),"pairs":[list(map(int,p)) for p in aud.get("triangle_pairs",[])],
    "min_patch_area_m2":float(area[np.r_[faceids,np.arange(len(Fbase),len(F))]].min()),"patch_faces":patch.tolist(),"all_faces":F.tolist()})
 valid=[x for x in patchtries if x.get("global_manifold_ok")]
 if not valid:raise ValueError("no topologically valid annulus")
 valid.sort(key=lambda x:(x["exact_pair_count"],x["cost"],-x["min_patch_area_m2"]))
 # Test prism compatibility for every already-gated annulus phase before choosing.
 # This is a finite topology search, not a geometric reorientation or relaxation.
 topology_outer_edges={tuple(sorted((outer_cycle[i],outer_cycle[(i+1)%len(outer_cycle)]))) for i in range(len(outer_cycle))}
 topology_sides={}
 for LL in (0,1):
  rr=cavity_rows(T0,Fbase,faceids,LL)
  if len(rr)!=54:raise ValueError(f"old cavity layer {LL}: {len(rr)} cells, expected54")
  topology_sides[LL]=side(boundary_chain(T0[rr],Xr0),outer_cycle,LL)
 feasible=[];topology_trials=[]
 for candidate in valid:
  candidate_patch=np.asarray(candidate["patch_faces"],int)
  layer_solutions={}
  try:
   for LL in (0,1):
    layer_solutions[LL]=solve_prism_orders(candidate_patch,LL,Xr0,Xc0,topology_outer_edges,topology_sides[LL],outer_cycle)
   feasible.append((candidate,layer_solutions))
   topology_trials.append({"phase":int(candidate["phase"]),"exact_pair_count":int(candidate["exact_pair_count"]),
                          "status":"compatible","search_nodes_by_layer":{str(k):int(v[1]) for k,v in layer_solutions.items()}})
  except ValueError as exc:
   topology_trials.append({"phase":int(candidate["phase"]),"exact_pair_count":int(candidate["exact_pair_count"]),
                           "status":"incompatible","reason":str(exc)})
 if not feasible:
  (OUT/"topology-search.json").write_text(json.dumps({"status":"no_compatible_prism_phase","trials":topology_trials,
    "note":"No geometry candidate or LP was emitted; every tested phase had already passed exact surface/manifold checks."},indent=2,sort_keys=True)+"\n")
  raise ValueError("no compatible prism decomposition across exact-surface annulus phases")
 chosen,preselected_orders=feasible[0];chosen["prism_compatibility_trials"]=topology_trials
 patch=np.asarray(chosen["patch_faces"],int);F=np.asarray(chosen["all_faces"],int)
 # Replace matching O/I/D surface star and preserve all exterior faces/rows.
 Fsurf=[face_at_layer(F,L) for L in range(3)]
 # Cavity rows and inherited outer side triangles.
 rows={};sides={};oldrel={}
 for L in (0,1):
  rr=cavity_rows(T0,Fbase,faceids,L); 
  if len(rr)!=54:raise ValueError(f"old cavity layer {L}: {len(rr)} cells, expected54")
  oldb=boundary_chain(T0[rr],Xr0);sides[L]=side(oldb,outer_cycle,L)
  rows[L]=rr;oldrel[L]=(cap_rel(oldb,oldstar,L),cap_rel(oldb,oldstar,L+1))
 # Edge diagonal choices: preserve exact old perimeter diagonals; use deterministic endpoint order internally.
 outer_edges={tuple(sorted((outer_cycle[i],outer_cycle[(i+1)%len(outer_cycle)]))) for i in range(len(outer_cycle))}
 edgetables={}
 for L in (0,1):
  d={}
  for tri in patch:
   for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
    e=tuple(sorted((int(a),int(b))))
    if e in d:continue
    if e in outer_edges:
     u,v=next((outer_cycle[i],outer_cycle[(i+1)%len(outer_cycle)]) for i in range(len(outer_cycle)) if tuple(sorted((outer_cycle[i],outer_cycle[(i+1)%len(outer_cycle)])))==e)
     lo,hi=old_diag(sides[L],u,v,L)
     d[e]=tuple(sorted((node(L,lo),node(L+1,hi))))
    else:
     lo,hi=e
     d[e]=tuple(sorted((node(L,lo),node(L+1,hi))))
  edgetables[L]=d
 xr=list(Xr);xc=list(Xc);newcells={};metas={};prism_solve_meta={}
 for L in (0,1):
  order_solution,search_nodes=preselected_orders[L]
  cells=np.vstack([o["cells"] for o in order_solution]).astype(np.int64);meta=[]
  for j,o in enumerate(order_solution):
   for k in range(3):meta.append({"layer":L,"patch_face":j,"simplex_in_prism":k,"order":list(o["order"]),"material_id":int(M0[rows[L][0]])})
  prism_solve_meta[str(L)]={"search_nodes":search_nodes,"orders":[list(o["order"]) for o in order_solution],"reference_determinants":[o["reference_determinants"] for o in order_solution],"target_determinants":[o["target_determinants"] for o in order_solution]}
  bnew=boundary_chain(cells,np.asarray(xr))
  expected=set(sides[L])
  for LL in (L,L+1):
   expected|={tuple(sorted(map(int,t))) for t in face_at_layer(patch,LL)}
  if set(bnew)!=expected:raise ValueError(f"replacement layer {L} cavity boundary triangle set mismatch missing={sorted(expected-set(bnew))[:40]} extra={sorted(set(bnew)-expected)[:40]} counts={len(expected)}/{len(bnew)}")
  if {k:v for k,v in bnew.items() if k in sides[L]}!=sides[L]:raise ValueError(f"replacement layer {L} changed fixed side boundary orientation")
  low=cap_rel(bnew,patch,L);high=cap_rel(bnew,patch,L+1)
  if (low,high)!=oldrel[L]:raise ValueError(f"cap orientation changed in layer{L}: {(low,high)} vs {oldrel[L]}")
  newcells[L]=cells;metas[L]=meta
 # Preserve outside cell rows; overwrite first54 per layer and append the remainder.
 T=T0.copy();M=M0.copy();extra_t=[];extra_m=[];cellmap=[]
 for L in (0,1):
  rr=rows[L];cells=newcells[L];mid=int(M0[rr[0]])
  T[rr]=cells[:len(rr)];M[rr]=mid
  for k in range(len(rr)):cellmap.append({"output_row":int(rr[k]),"row_kind":"repurposed_old_cavity_slot","layer":L,"patch_face":metas[L][k]["patch_face"],"material_id":mid})
  for k,t in enumerate(cells[len(rr):],start=len(rr)):
   outrow=len(T)+len(extra_t);extra_t.append(t);extra_m.append(mid)
   cellmap.append({"output_row":int(outrow),"row_kind":"appended_cone_cell","layer":L,"patch_face":metas[L][k]["patch_face"],"material_id":mid})
 if extra_t:T=np.vstack((T,np.asarray(extra_t,int)));M=np.r_[M,np.asarray(extra_m,int)]
 Xr=np.asarray(xr);Xc=np.asarray(xc)
 # Bounded exact per-node determinant LP: O and exterior rings fixed.
 sys.path.insert(0,str(HERE/"source-star-011"))
 from tet_determinant_lp import coordinate_repair
 free_nodes=list(range(7239,7251))+[3656,6067]
 edge_parent={**{7239+i:7233+i for i in range(6)},
              **{7245+i:7239+i for i in range(6)},
              3656:1245,6067:3656}
 lp_initial={"reference":Xr.copy(),"current":Xc.copy()}
 edge_base={}
 for child,parent in edge_parent.items():
  for name,pts in lp_initial.items():
   edge_base[(int(child),name)]=float(np.linalg.norm(pts[int(child)]-pts[int(parent)]))
 def layer_bounds(node_id, pose_points):
  if int(node_id) not in edge_parent:return None
  parent=int(edge_parent[int(node_id)]); rows=[]; rhs=[]
  for name,pts in pose_points.items():
   v=pts[int(node_id)]-pts[parent]; length=float(np.linalg.norm(v))
   base=edge_base[(int(node_id),name)]
   if length<=1e-12 or base<=1e-12:continue
   direction=v/length
   lower=.25*base; upper=4.0*base
   rows.extend((direction,-direction))
   rhs.extend((lower-length,length-upper))
  if not rows:return None
  return np.asarray(rows,float),np.asarray(rhs,float)
 raw_ref=dets(Xr,T); raw_cur=dets(Xc,T)
 orient=np.sign(raw_ref); orient[orient==0]=np.sign(raw_cur[orient==0]); orient[orient==0]=1.
 lp_result=coordinate_repair({"reference":Xr,"current":Xc},T,free_nodes,
   determinant_floor=1e-22,trust_radius=0.004,orientation_signs=orient,
   max_sweeps=4,emission_dtype={"reference":None,"current":np.float32},
   linear_constraints_provider=layer_bounds)
 Xr=lp_result.points_by_pose["reference"];Xc=lp_result.points_by_pose["current"]
 lp_meta={"sweeps":int(lp_result.sweeps),"accepted_any":bool(lp_result.accepted),
   "emitted_f32_global_hinge_before":float(lp_result.emitted_merit_before),
   "emitted_f32_global_hinge_after":float(lp_result.emitted_merit_after),
   "free_node_ids":free_nodes,"displacement_by_node_m":{
     str(n):{pose:(lp_result.points_by_pose[pose][n]-lp_initial[pose][n]).tolist()
             for pose in lp_initial} for n in free_nodes},
   "steps":[{"node_id":int(s.node_id),"status":s.status,"phase1_slack":float(s.phase1_slack),
     "local_hinge_before":float(s.normalized_merit_before),
     "local_hinge_after":float(s.normalized_merit_after),
     "delta_m":s.delta.tolist(),"message":s.message} for s in lp_result.steps]}


 dref=dets(Xr,T);dcur=dets(Xc,T);dcur32=dets(Xc.astype(np.float32).astype(np.float64),T)
 qref=fit.tet_metrics(Xr,T)[2];qcur=fit.tet_metrics(Xc,T)[2]
 # Cavity volume deltas, not conservative mass ancestry.
 vol={}
 for L in (0,1):
  vol[str(L)]={"material_id":int(M0[rows[L][0]]),"old_reference_m3":float(dets(Xr0,T0[rows[L]]).sum()/6),
    "new_reference_m3":float(dets(Xr,newcells[L]).sum()/6),"old_target_m3":float(dets(Xc0,T0[rows[L]]).sum()/6),
    "new_target_m3":float(dets(Xc,newcells[L]).sum()/6)}
  vol[str(L)]["reference_delta_m3"]=vol[str(L)]["new_reference_m3"]-vol[str(L)]["old_reference_m3"]
  vol[str(L)]["target_delta_m3"]=vol[str(L)]["new_target_m3"]-vol[str(L)]["old_target_m3"]
 report={"schema":"numi.d36.o1245-detailed-source-connected-cavity.v1","classification":"bounded_topology_counterfactual_not_admitted",
  "builder_sha256":sha(Path(__file__)),
  "inputs":{"base_cavity_sha256":sha(BASE),"source_star_report_sha256":sha(STAR),"source_target_preflight_sha256":sha(HERE/"run-003/outer-target-preflight.npz"),
    "skin294_sha256":sk["sha256"],"whole_layer_fit_sha256":sha(LAYER),"exact_predicate_sha256":sha(fit.PRED)},
  "surface":{"center_column":centercol,"center_source_vertex":source_center,"old_faces":18,"new_faces":30,"old_face_ids":faceids.tolist(),
    "detail_source_face_ids":source_star_ids.tolist(),"detail_ring_source_vertex_ids":list(map(int,inner_cycle)),"detail_ring_projection":projection,
    "annulus_trials":[{k:v for k,v in t.items() if k not in ("all_faces","patch_faces")} for t in patchtries],
    "selected_phase":chosen["phase"],"exact_emitted_float32_local_self_pairs":chosen["exact_pair_count"],"exact_pairs":chosen["pairs"],
    "outside_face_rows_unchanged":True,"global_boundary_unchanged":True,"global_nonmanifold_edges":0,
    "new_corner_ancestry":"Six source-star faces retain exact NHSKIN294 face IDs. Annulus corners have direct NHSKIN294 vertex identities; no parent face is asserted."},
  "volume":{"old_cells_per_material_layer":{str(L):len(rows[L]) for L in (0,1)},"new_prism_cells_per_layer":{str(L):len(newcells[L]) for L in (0,1)},"prism_order_solutions":prism_solve_meta,
    "exact_fixed_side_boundary_preserved":True,"cavity_volume_by_layer":vol,"full_reference_nonpositive_cells":int(np.count_nonzero(dref<=0)),
    "reference_min_quality":float(qref.min()),"full_target_nonpositive_cells":int(np.count_nonzero(dcur<=0)),
    "target_float32_nonpositive_cells":int(np.count_nonzero(dcur32<=0)),"target_min_quality":float(qcur.min()),
    "added_nodes":int(len(Xr)-len(Xr0)),"added_cells":int(len(T)-len(T0)),"target_I_D_offsets":"P1-interpolated existing column offsets; construction seed only, no mass ancestry.",
    "source_reference_ring":"Closest-point projection onto old reference O star; current ring O uses exact detailed294 source positions.",
    "outside_tet_rows_bitwise_unchanged":True,"cell_row_map":"Construction row slots only; not conservative material ancestry."},
  "limitations":["This does not prove an outer-surface fold: the unbounded shared-apex certificate is a volume-topology obstruction.","Exact outer self-pair and fixed side-chain checks do not establish tissue/bone clearance, mass conservation, constitutive response, or native/physical admission.","The 72-cell previous cavity audit remains separate; this 18-face O1245 cavity is a new diagnostic construction.","Any target nonpositive cells reject the volume candidate; no acceptance floor was relaxed."],
  "runtime":{"host":socket.gethostname(),"python":sys.version,"numpy":np.__version__,"seconds":time.time()-t0}}
 report["volume"]["local_determinant_lp"]=lp_meta
 out=OUT/"candidate.npz"
 np.savez_compressed(out,reference_points_AP_m=Xr,target_points_AP_m=Xc,old_tetrahedra=T0.astype(np.int32),candidate_tetrahedra=T.astype(np.int32),
  old_material_ids=M0.astype(np.int8),candidate_material_ids=M.astype(np.int8),old_O_faces=F0.astype(np.int32),candidate_O_faces=F.astype(np.int32),
  candidate_I_faces=Fsurf[1].astype(np.int32),candidate_D_faces=Fsurf[2].astype(np.int32),candidate_patch_faces=patch.astype(np.int32),
  changed_face_ids=np.r_[faceids,np.arange(len(F0),len(F),dtype=np.int64)].astype(np.int32),ring_source_vertex_ids=np.asarray(inner_cycle,np.int32),
  ring_outer_columns=np.asarray([ring_cols[int(s)] for s in inner_cycle],np.int32),new_cell_map_json=np.asarray(json.dumps(cellmap,sort_keys=True)))
 report["output"]={"path":str(out),"sha256":sha(out)};(OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"report":str(OUT/"report.json"),"candidate":str(out),"sha256":report["output"]["sha256"],
   "phase":chosen["phase"],"surface_pairs":chosen["exact_pair_count"],"reference_bad":int(np.count_nonzero(dref<=0)),
   "target_bad":int(np.count_nonzero(dcur<=0)),"target_f32_bad":int(np.count_nonzero(dcur32<=0)),
   "added_cells":len(T)-len(T0),"minq":float(qcur.min())},indent=2))
if __name__=="__main__":main()
