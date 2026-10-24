#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,collections
import numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
src=E/"bilateral-source-skin-cap-reconstruction-211/candidate-surface-unadmitted.npz"
out=E/"bilateral-source-skin-cap-reconstruction-212"
out.mkdir(exist_ok=True)
dst=out/"candidate-surface-welded-unadmitted.npz"
rp=out/"report-212.json"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(src)=="9c33d81c4d16985051db78234e4e36d233f3130a02c60ffd8ca7f1d04cc76884"
z=np.load(src,allow_pickle=False); X=z["outer_world_actual007_m"]; F0=z["surface_faces"].astype(np.int64); F=F0.copy(); N=len(X); nf=len(F0); face_orig=np.arange(nf,dtype=np.int64)
tri=X[F0].astype(np.float64)
area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
bad=np.flatnonzero(area==0.0); assert len(bad)==60
pairs=set()
for i in bad:
 t=F0[i]; found=[]
 for a,b in ((0,1),(1,2),(2,0)):
  u,v=int(t[a]),int(t[b])
  if np.array_equal(X[u],X[v]):found.append(tuple(sorted((u,v))))
 assert len(found)==1,(int(i),found)
 pairs.add(found[0])
pairs=sorted(pairs); assert len(pairs)==30
protected=set(map(int,z["protected_cut_vertex_ids"])); assert not any(a in protected or b in protected for a,b in pairs)
guide=z["source_guide_face_ids"]; bary=z["source_guide_barycentric"]
for a,b in pairs:
 assert guide[a]==guide[b] and np.array_equal(bary[a],bary[b])
pg=collections.defaultdict(set)
for a,b in pairs:pg[a].add(b);pg[b].add(a)
assert max(map(len,pg.values()))<=2
components=[]; seen=set()
for v in sorted(pg):
 if v in seen:continue
 q=[v];seen.add(v); comp=[]
 while q:
  u=q.pop();comp.append(u)
  for w in sorted(pg[u]):
   if w not in seen:seen.add(w);q.append(w)
 assert len(comp) in (2,3) and sum(len(pg[u]) for u in comp)//2==len(comp)-1
 components.append(comp)
alias=np.arange(N,dtype=np.int64)
def rep(a):
 a=int(a)
 while alias[a]!=a:alias[a]=alias[alias[a]];a=int(alias[a])
 return a
def topo(F):
 adj=collections.defaultdict(set); opp=collections.defaultdict(set); ec=collections.Counter()
 for t in F:
  a,b,c=map(int,t)
  for u,v,w in ((a,b,c),(b,c,a),(c,a,b)):
   e=tuple(sorted((u,v))); ec[e]+=1;opp[e].add(w);adj[u].add(v);adj[v].add(u)
 return adj,opp,ec
def boundary(F):
 _,_,ec=topo(F);return {e for e,n in ec.items() if n==1}
b0=boundary(F);assert len(b0)==45
steps=[]; removed=0
for comp in components:
 master=min((i for i in comp if i<16821),default=min(comp))
 stack=[master]; seen2={master}; parent={}
 while stack:
  u=stack.pop()
  for v in sorted(pg[u]):
   if v not in seen2:seen2.add(v);parent[v]=u;stack.append(v)
 assert seen2==set(comp)
 def depth(v):
  d=0
  while v in parent:v=parent[v];d+=1
  return d
 for v in sorted(parent,key=depth,reverse=True):
  u=rep(parent[v]); w=rep(v)
  if u==w:continue
  assert u not in protected and w not in protected and np.array_equal(X[u],X[w])
  adj,opp,ec=topo(F); e=tuple(sorted((u,w)))
  inter=adj[u]&adj[w]; le=opp[e]; inc=ec[e]
  assert inc==2 and inter==le,(u,w,inc,inter,le)
  F2=F.copy();F2[F2==w]=u
  keep=np.array([len(set(map(int,t)))==3 for t in F2])
  gone=np.flatnonzero(~keep)
  orig_gone=face_orig[gone]
  assert len(gone)>0 and np.all(area[orig_gone]==0.0) and set(map(int,orig_gone))<=set(map(int,bad))
  F=F2[keep];face_orig=face_orig[keep];removed+=len(gone);alias[w]=u
  steps.append({"master":int(u),"alias":int(w),"incidence":int(inc),"link_legal":True,"removed_faces":int(len(gone))})
assert removed==60 and len(F)==nf-60
keep_rows=face_orig
assert set(map(int,keep_rows))==set(map(int,np.setdiff1d(np.arange(nf),bad)))
expect=np.asarray([[rep(v) for v in t] for t in F0[keep_rows]],dtype=np.int64)
assert np.array_equal(F,expect)
V=X[F].astype(np.float64); area2=np.linalg.norm(np.cross(V[:,1]-V[:,0],V[:,2]-V[:,0]),axis=1)
assert np.all(area2>0) and np.array_equal(area2,area[keep_rows])
assert boundary(F)==b0
adj,opp,ec=topo(F); hist=collections.Counter(ec.values())
assert max(ec.values())==2 and hist==collections.Counter({2:70779,1:45})
# Check every used vertex link is a single cycle or path.
lg=collections.defaultdict(lambda:collections.defaultdict(set))
for t in F:
 a,b,c=map(int,t)
 for v,u,w in ((a,b,c),(b,c,a),(c,a,b)):lg[v][u].add(w);lg[v][w].add(u)
bv={v for e in b0 for v in e}
for v,g in lg.items():
 ds=[len(ns) for ns in g.values()]
 assert all(d in (1,2) for d in ds)
 assert sum(d==1 for d in ds)==(2 if v in bv else 0)
 ns=set(g); seen3={next(iter(ns))}; q=list(seen3)
 while q:
  u=q.pop()
  for w in g[u]:
   if w not in seen3:seen3.add(w);q.append(w)
 assert ns==seen3
# Every face metadata array follows surviving rows. Keep vertex rows untouched; aliases are explicit.
outarr={}
for k in z.files:
 a=z[k]
 if k=="surface_faces":continue
 if k=="changed_face_ids":
  outarr["input_changed_face_ids"]=a.astype(np.int64)
  m=np.full(nf,-1,dtype=np.int64);m[keep_rows]=np.arange(len(keep_rows))
  mapped=m[a.astype(np.int64)];outarr["changed_face_ids"]=mapped[mapped>=0]
 elif a.ndim>0 and a.shape[0]==nf:outarr[k]=a[keep_rows]
 else:outarr[k]=a
outarr["surface_faces"]=F
outarr["vertex_master_rows"]=alias.copy()
outarr["collapsed_vertex_pairs"]=np.asarray(pairs,dtype=np.int64)
outarr["removed_zero_area_face_rows_input"]=bad.astype(np.int64)
outarr["removed_zero_area_face_vertices_input"]=F0[bad]
for k in ("surface_face_parent_223","surface_face_kind","surface_face_source_side","surface_face_parent_surface_ids","surface_face_detailed_source_ids","surface_face_original_d36_row"):
 if k in z.files:outarr["removed_"+k]=z[k][bad]
np.savez_compressed(dst,**outarr)
w=np.load(dst,allow_pickle=False)
assert w["surface_faces"].shape==(nf-60,3) and np.array_equal(w["outer_world_actual007_m"],X)
assert np.array_equal(w["protected_cut_vertex_ids"],z["protected_cut_vertex_ids"])
groups=collections.Counter((int(z["surface_face_source_side"][i]),int(z["surface_face_kind"][i])) for i in bad)
report={"physical_admission":False,"scope":"exact zero-length projection-edge topology cleanup only; no coordinate motion or physical qualification","input":str(src),"input_sha256":sha(src),"output":str(dst),"output_sha256":sha(dst),"script":"/tmp/clean_211_duplicate_projection_212.py","script_sha256":sha("/tmp/clean_211_duplicate_projection_212.py"),"input_vertices":N,"retained_vertex_rows":N,"alias_rows":int(np.sum(alias!=np.arange(N))),"input_faces":nf,"output_faces":len(F),"removed_faces":removed,"removed_face_groups":{f"{a}:{b}":int(n) for (a,b),n in sorted(groups.items())},"zero_edges":len(pairs),"components":len(components),"sequential_link_checks":len(steps),"collapse_steps":steps,"all_collapses_link_legal":True,"aliases_have_identical_guide_face_and_bary":True,"geometry_change_m":0.0,"all_remaining_face_areas_positive":True,"minimum_double_area_m2":float(area2.min()),"edge_incidence_histogram":{str(k):int(v) for k,v in sorted(hist.items())},"boundary_edges":len(b0),"boundary_edge_ids_exact":True,"protected_cut_ids_exact":True,"vertex_links_manifold":True,"remaining_gates":["exact source self and all-nearby tissue crossings","bone crossings","surface-to-volume transfer","independent inner/deep layers","conservative material mass","native mechanics"]}
rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"candidate":str(dst),"candidate_sha256":report["output_sha256"],"report":str(rp),"script_sha256":report["script_sha256"],"faces":len(F),"removed":removed,"min_area2":report["minimum_double_area_m2"],"boundary":len(b0),"edge_hist":report["edge_incidence_histogram"]},indent=2,sort_keys=True))
