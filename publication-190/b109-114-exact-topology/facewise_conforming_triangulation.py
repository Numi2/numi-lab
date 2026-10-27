#!/usr/bin/env python3
"""Face-wise exact-endpoint constrained triangulation for separate 109/114 patches."""
from pathlib import Path
from collections import defaultdict,Counter
from fractions import Fraction
from scipy.spatial import Delaunay
import numpy as np, json, hashlib, time, traceback
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
C=Path('/Users/numi/Work/numi-human-connected-surface-069/D/publication-182-stage-001/combined-653/candidate.npz')
GRAPH=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b109-114/attempt-002-exact-contours/exact-contour-graph.json')
PATCH_DIR=S/'per-bank-workers-20261027/b109-114/attempt-001-boundary-feasibility/patch-components'
OUT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b109-114/attempt-002-exact-contours')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rat(s):return Fraction(s)
def dec(x):
 if isinstance(x,str) and '/' in x:return Fraction(x)
 if isinstance(x,list):return tuple(dec(y) for y in x)
 return x
def orient(a,b,c):return float((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
def orient_exact(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def make_tri(a,b,c,p2):
 o=orient(p2[a],p2[b],p2[c])
 if abs(o)<1e-18:raise ValueError(f'degenerate reference triangle {(a,b,c)} area2={o}')
 return (a,b,c) if o>0 else (a,c,b)
def proper_cross(e,f,p2):
 a,b=e;c,d=f
 if len({a,b,c,d})<4:return False
 o1,o2=orient(p2[a],p2[b],p2[c]),orient(p2[a],p2[b],p2[d])
 o3,o4=orient(p2[c],p2[d],p2[a]),orient(p2[c],p2[d],p2[b])
 return o1*o2 < -1e-26 and o3*o4 < -1e-26
def edge_map(tris):
 out=defaultdict(list)
 for i,t in enumerate(tris):
  for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):out[tuple(sorted((int(a),int(b))))].append(i)
 return out
def on_exact(p,a,b):
 return orient_exact(a,b,p)==0 and min(a[0],b[0])<=p[0]<=max(a[0],b[0]) and min(a[1],b[1])<=p[1]<=max(a[1],b[1])
def seg_intersect_exact(a,b,c,d):
 o1,o2,o3,o4=orient_exact(a,b,c),orient_exact(a,b,d),orient_exact(c,d,a),orient_exact(c,d,b)
 if o1==0 and on_exact(c,a,b):return True
 if o2==0 and on_exact(d,a,b):return True
 if o3==0 and on_exact(a,c,d):return True
 if o4==0 and on_exact(b,c,d):return True
 return (o1>0)!=(o2>0) and (o3>0)!=(o4>0)
def validate_constraints(constraints,pexact):
 constraints=list(dict.fromkeys(tuple(sorted(map(int,e))) for e in constraints))
 for ix,e in enumerate(constraints):
  a,b=e
  for k,p in enumerate(pexact):
   if k not in e and on_exact(p,pexact[a],pexact[b]):
    raise ValueError(f'exact constraint T-junction: point {k} lies on edge {e}')
  for f in constraints[ix+1:]:
   if e==f:continue
   shared=set(e)&set(f)
   if not seg_intersect_exact(pexact[e[0]],pexact[e[1]],pexact[f[0]],pexact[f[1]]):continue
   if not shared:raise ValueError(f'exact crossing/overlap between constraints {e} and {f}')
   if len(shared)==1:
    sh=next(iter(shared));u=next(x for x in e if x!=sh);v=next(x for x in f if x!=sh)
    if orient_exact(pexact[sh],pexact[u],pexact[v])==0 and (on_exact(pexact[u],pexact[sh],pexact[v]) or on_exact(pexact[v],pexact[sh],pexact[u])):
     raise ValueError(f'exact collinear overlap at shared endpoint: {e} {f}')
 return constraints
def constrained_triangulate(p2,constraints,max_flips=5000):
 if len(p2)<3:raise ValueError('less than three local points')
 T=[make_tri(*map(int,t),p2) for t in Delaunay(p2).simplices]
 if len({v for t in T for v in t})!=len(p2):raise ValueError('Delaunay omitted one or more local points')
 protected=set();flip_count=0
 for target0 in constraints:
  target=tuple(sorted(map(int,target0)))
  if target[0]==target[1]:raise ValueError('collapsed local constraint')
  for _ in range(max_flips):
   edges=edge_map(T)
   if target in edges:protected.add(target);break
   crossing=[e for e in edges if proper_cross(e,target,p2)];switched=False
   for e in sorted(crossing):
    if e in protected or len(edges[e])!=2:continue
    i,j=edges[e];c=next(v for v in T[i] if v not in e);d=next(v for v in T[j] if v not in e);new=tuple(sorted((c,d)))
    if new in edges or not proper_cross(e,new,p2) or proper_cross(new,target,p2):continue
    if any(proper_cross(new,k,p2) for k in protected):continue
    t1,t2=make_tri(c,d,e[0],p2),make_tri(d,c,e[1],p2)
    if abs((orient(*(p2[x] for x in T[i]))+orient(*(p2[x] for x in T[j])))-(orient(*(p2[x] for x in t1))+orient(*(p2[x] for x in t2))))>1e-10:continue
    T[i],T[j]=t1,t2;flip_count+=1;switched=True;break
   if not switched:raise ValueError(f'cannot insert constraint {target}; crossing edges={crossing[:8]}')
  else:raise ValueError(f'flip cap reached for {target}')
 edges=edge_map(T)
 missing=[tuple(sorted(map(int,e))) for e in constraints if tuple(sorted(map(int,e))) not in edges]
 if missing:raise ValueError(f'local constraints missing after CDT: {missing[:8]}')
 all_edges=list(edges)
 for i,e in enumerate(all_edges):
  for f in all_edges[i+1:]:
   if proper_cross(e,f,p2):raise ValueError(f'local triangulation edges cross: {e} {f}')
 return T,flip_count
def f32(x):return np.float32(float(x))
def exact_cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def one_component_boundary(F,parent_faces):
 edge_faces=defaultdict(list);node_faces=defaultdict(list)
 parent_set=set(map(int,parent_faces))
 for fi,t in enumerate(F):
  for v in t:node_faces[int(v)].append(fi)
  for u,v in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):edge_faces[tuple(sorted((int(u),int(v))))].append(fi)
 p_edges=defaultdict(list)
 for fi in parent_faces:
  t=F[int(fi)]
  for u,v in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):p_edges[tuple(sorted((int(u),int(v))))].append(int(fi))
 boundary=[]
 for e,inside in p_edges.items():
  if any(g not in parent_set for g in edge_faces[e]) or len(edge_faces[e])==1:boundary.append(e)
 graph=defaultdict(list)
 for u,v in boundary:graph[u].append(v);graph[v].append(u)
 unseen=set(graph);comps=[]
 while unseen:
  st=min(unseen);unseen.remove(st);q=[st];c=[]
  while q:
   v=q.pop();c.append(v)
   for w in graph[v]:
    if w in unseen:unseen.remove(w);q.append(w)
  comps.append(c)
 if len(comps)!=1 or not all(len(graph[v])==2 for v in graph):raise ValueError(f'patch outer boundary is not one simple cycle: loops={list(map(len,comps))}, degrees={dict(Counter(map(len,graph.values())))}')
 return boundary,edge_faces,node_faces
started=time.monotonic()
g=json.loads(GRAPH.read_text())
if g['status']!='EXACT_CONTOUR_GRAPH_COMPLETE' or g['candidate_sha256']!=sha(C):raise RuntimeError('exact contour report is stale or failed')
with np.load(C,allow_pickle=False) as z:Z={k:z[k].copy() for k in z.files}
P32=np.asarray(Z['points_world_m'],np.float32);F=np.asarray(Z['source_O_faces'],np.int64)
ep_by_root=defaultdict(list)
for r in g['endpoint_records']:ep_by_root[int(r['root'])].append(r)
reports=[]
for ci in (0,1):
 cg=g['per_component_graphs'][ci]
 pfile=PATCH_DIR/f'component-{ci:02d}-face-rows.npy'
 parent_faces=sorted(map(int,np.load(pfile,allow_pickle=False)))
 parent_set=set(parent_faces);edge_rows=cg['contour_edge_roots']
 try:
  degree_hist={int(k):int(v) for k,v in cg['endpoint_degree_histogram'].items()}
  all_cycles=bool(cg['endpoint_graph_component_count']>=1 and set(degree_hist)=={2} and cg['collapsed_segment_count']==0 and cg['duplicate_contour_edge_count']==0)
  if not all_cycles:raise ValueError(f'exact contour graph has a branch/collapse/duplicate: degree_hist={degree_hist}, graph_components={cg["endpoint_graph_component_sizes"]}')
  boundary_edges,edgefaces,nodefaces=one_component_boundary(F,parent_faces)
  roots=sorted({int(r) for e in edge_rows for r in e['endpoint_roots']})
  points={('v',int(v)):P32[int(v)].copy() for v in sorted({int(v) for fi in parent_faces for v in F[fi]})}
  key_for_root={root:('x',root) for root in roots}
  endpoint_record_root={}
  for root in roots:
   rows=ep_by_root[root]
   if len({tuple(r['local_m']) for r in rows})!=1 or len({tuple(r['world_from_skin_barycentric_m']) for r in rows})!=1:
    raise ValueError(f'exact endpoint root {root} has conflicting rational coordinates')
   endpoint_record_root[root]=rows[0]
   world=tuple(rat(x) for x in rows[0]['world_from_skin_barycentric_m'])
   points[key_for_root[root]]=np.asarray([f32(x) for x in world],dtype=np.float32)
  point_bary=defaultdict(dict)
  for fi in parent_faces:
   for j,v in enumerate(F[fi]):
    b=[Fraction(0),Fraction(0),Fraction(0)];b[j]=Fraction(1)
    point_bary[int(fi)][('v',int(v))]=tuple(b)
  def put_face(fi,key,b):
   b=tuple(b)
   if sum(b)!=1 or min(b)<0:raise ValueError(f'bad exact barycentric {(fi,key,b)}')
   old=point_bary[fi].get(key)
   if old is not None and old!=b:raise ValueError(f'canonical endpoint has different barycentrics in face {fi}')
   point_bary[fi][key]=b
  # Apply each saved exact skin-side endpoint barycentric to its intersected face.
  for root in roots:
   key=key_for_root[root]
   for r in ep_by_root[root]:
    sf=int(r['skin_face_row'])
    if sf in parent_set:put_face(sf,key,tuple(rat(x) for x in r['skin_barycentrics']))
   feats={dec(r['skin_feature']) for r in ep_by_root[root] if r['skin_feature'] is not None}
   for feat in feats:
    if feat[0]=='skin-edge':
     _,lo,hi,t=feat;edge=tuple(sorted((int(lo),int(hi))))
     for nb in edgefaces[edge]:
      if nb not in parent_set:continue
      nodes=[int(v) for v in F[nb]];b=[Fraction(0),Fraction(0),Fraction(0)]
      b[nodes.index(int(lo))]=1-t;b[nodes.index(int(hi))]=t;put_face(nb,key,b)
    elif feat[0]=='skin-vertex':
     v=int(feat[1])
     for nb in nodefaces[v]:
      if nb not in parent_set:continue
      nodes=[int(vv) for vv in F[nb]];b=[Fraction(0),Fraction(0),Fraction(0)];b[nodes.index(v)]=Fraction(1);put_face(nb,key,b)
  segments_by_face=defaultdict(list)
  for rec in edge_rows:
   a,b=map(int,rec['endpoint_roots']);sf=int(rec['skin_face_row']);bone=int(rec['bone_face_row'])
   if a==b:raise ValueError(f'collapsed exact contour segment pair {rec["pair_index"]}')
   segments_by_face[sf].append((key_for_root[a],key_for_root[b],bone,int(rec['pair_index'])))
  child_keys=[];child_parent=[];child_bary_exact=[];per_face=[];flips_total=0;contour_counts=Counter()
  for fi in parent_faces:
   keys_face=sorted(point_bary[fi],key=lambda k:(0,k[1]) if k[0]=='v' else (1,k[1]))
   idx={k:i for i,k in enumerate(keys_face)};local_bary=[point_bary[fi][k] for k in keys_face]
   pexact=[(b[1],b[2]) for b in local_bary];p2=np.asarray([[float(x),float(y)] for x,y in pexact],dtype=np.float64)
   if len(set(pexact))!=len(pexact):raise ValueError(f'distinct endpoint identities collapse in exact reference plane of face {fi}')
   contours=[];constraints=[]
   for a,b,bone,pair_idx in segments_by_face.get(fi,[]):
    e=tuple(sorted((idx[a],idx[b])));contours.append((e,bone,pair_idx));constraints.append(e);contour_counts[(a,b)]+=1
   tri_nodes=[int(v) for v in F[fi]];boundary_local=[]
   for j,k in ((0,1),(1,2),(2,0)):
    other=3-j-k;seq=[]
    for qkey,b in zip(keys_face,local_bary):
     if b[other]==0:seq.append((b[k],idx[qkey]))
    seq.sort()
    if len(seq)<2 or seq[0][1]!=idx[('v',tri_nodes[j])] or seq[-1][1]!=idx[('v',tri_nodes[k])]:
     raise ValueError(f'parent face {fi} boundary edge {(tri_nodes[j],tri_nodes[k])} lost endpoint')
    for (_,u),(_,v) in zip(seq,seq[1:]):boundary_local.append(tuple(sorted((u,v))))
   constraints=validate_constraints(list(dict.fromkeys(constraints+boundary_local)),pexact)
   if len(keys_face)==3 and not contours:
    triids=[idx[('v',int(v))] for v in F[fi]];tris=[make_tri(*triids,p2)];flips=0
   else:tris,flips=constrained_triangulate(p2,constraints)
   flips_total+=flips;emap=edge_map(tris)
   signed=[exact_cross(pexact[t[0]],pexact[t[1]],pexact[t[2]]) for t in tris]
   if any(a<=0 for a in signed) or sum(signed,Fraction(0))!=Fraction(1):raise ValueError(f'exact parent-face coverage/orientation failure for {fi}: sum2={sum(signed,Fraction(0))}')
   if any(len(v) not in (1,2) for v in emap.values()):raise ValueError(f'parent face {fi} has invalid local edge incidence')
   for e,bone,pair_idx in contours:
    if len(emap.get(tuple(sorted(e)),[]))!=2:raise ValueError(f'contour edge incidence not 2 in parent face {fi}: pair {pair_idx}, edge {e}')
   for t in tris:
    child_keys.append([keys_face[x] for x in t]);child_parent.append(fi);child_bary_exact.append([local_bary[x] for x in t])
   per_face.append({'parent_face_row':fi,'point_count':len(keys_face),'child_triangle_count':len(tris),'contour_segments':len(contours),'local_constraint_count':len(constraints),'exact_reference_double_area':str(sum(signed,Fraction(0))),'flips':flips})
  patch_nodes=sorted({int(v) for fi in parent_faces for v in F[fi]})
  roots_in_patch=sorted(roots)
  node_keys=[('v',v) for v in patch_nodes]+[key_for_root[r] for r in roots_in_patch]
  node_index={k:i for i,k in enumerate(node_keys)}
  child_faces=np.asarray([[node_index[k] for k in tri] for tri in child_keys],np.int64)
  counts=Counter()
  for tri in child_faces:
   for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):counts[tuple(sorted((int(a),int(b))))]+=1
  if any(v not in (1,2) for v in counts.values()):raise ValueError(f'global patch edge incidence invalid: {[(e,n) for e,n in counts.items() if n not in (1,2)][:8]}')
  old_boundary={tuple(sorted((node_index[('v',u)],node_index[('v',v)]))) for u,v in boundary_edges}
  new_boundary={e for e,n in counts.items() if n==1}
  if new_boundary!=old_boundary:raise ValueError(f'fixed outer boundary changed: missing {len(old_boundary-new_boundary)}, added {len(new_boundary-old_boundary)}')
  contour_global=sorted({tuple(sorted((node_index[key_for_root[int(a)]],node_index[key_for_root[int(b)]]))) for a,b in ((x['endpoint_roots'][0],x['endpoint_roots'][1]) for x in edge_rows)})
  if any(counts.get(e)!=2 for e in contour_global):raise ValueError('global contour constraint incidence is not two')
  # Verify all per-face contour constraints are unique exact segments.
  if len(contour_global)!=len(edge_rows):raise ValueError(f'duplicate canonical contour segments: {len(edge_rows)} pair segments vs {len(contour_global)} unique edges')
  # Preserve source and refinement ancestry per child face.
  src_ptr=[0];src_rows=[];sub_ptr=[0];sub_rows=[];det_ptr=[0];det_rows=[]
  for fi in child_parent:
   a,b=map(int,Z['face_source_ancestry_indptr'][fi:fi+2]);src_rows.extend(map(int,Z['face_source_ancestry_face_rows'][a:b]));src_ptr.append(len(src_rows))
   a,b=map(int,Z['face_fine_subface_ancestry_indptr'][fi:fi+2]);sub_rows.extend(map(int,Z['face_fine_subface_ancestry_rows'][a:b]));sub_ptr.append(len(sub_rows))
   a,b=map(int,Z['face_fine_detail_ancestry_indptr'][fi:fi+2]);det_rows.extend(map(int,Z['face_fine_detail_ancestry_rows'][a:b]));det_ptr.append(len(det_rows))
  out_points=np.asarray([points[k] for k in node_keys],np.float32)
  child_bary=np.asarray([[[float(x) for x in b] for b in tri] for tri in child_bary_exact],np.float64)
  point_parent=np.full(len(node_keys),-1,np.int64);point_bary=np.zeros((len(node_keys),3),np.float64)
  endpoint_ancestry=[]
  for root in roots_in_patch:
   key=key_for_root[root];uses=[(int(fi),b) for fi,m in point_bary_by_face.items() for k,b in m.items() if k==key] if False else []
   uses=[(int(fi),point_bary_exact) for fi,m in globals_point_bary.items() for k,point_bary_exact in m.items() if k==key] if False else []
   # Choose representative source face/bary from the exact rational endpoint records.
   rec=endpoint_record_root[root];rep_face=int(rec['skin_face_row']);rep_bary=tuple(rat(x) for x in rec['skin_barycentrics'])
   ni=node_index[key];point_parent[ni]=rep_face;point_bary[ni]=[float(x) for x in rep_bary]
   endpoint_ancestry.append({'endpoint_root':root,'node_row':ni,'representative_parent_candidate_face_row':rep_face,'representative_skin_barycentrics':[str(x) for x in rep_bary],'exact_local_m':rec['local_m'],'exact_world_from_skin_barycentric_m':rec['world_from_skin_barycentric_m'],'skin_feature':rec['skin_feature'],'bone_feature':rec['bone_feature']})
  subdir=OUT/f'component-{ci:02d}';subdir.mkdir(exist_ok=True)
  outpath=subdir/'conforming-patch.npz'
  np.savez_compressed(outpath,points_world_m_f32=out_points,patch_faces=child_faces,source_candidate_node_rows=np.asarray([k[1] if k[0]=='v' else -1 for k in node_keys],np.int64),point_parent_candidate_face_rows=point_parent,point_parent_candidate_barycentric_f64=point_bary,face_parent_candidate_rows=np.asarray(child_parent,np.int64),face_parent_barycentric_f64=child_bary,face_source_ancestry_indptr=np.asarray(src_ptr,np.int64),face_source_ancestry_face_rows=np.asarray(src_rows,np.int64),face_fine_subface_ancestry_indptr=np.asarray(sub_ptr,np.int64),face_fine_subface_ancestry_rows=np.asarray(sub_rows,np.int64),face_fine_detail_ancestry_indptr=np.asarray(det_ptr,np.int64),face_fine_detail_ancestry_face_rows=np.asarray(det_rows,np.int64),patch_outer_boundary_edges=np.asarray(sorted(old_boundary),np.int64),contour_edges=np.asarray(contour_global,np.int64).reshape(-1,2),original_patch_face_rows=np.asarray(parent_faces,np.int64),exact_endpoint_component_ids=np.asarray([-1]*len(patch_nodes)+roots_in_patch,np.int64))
  side=subdir/'rational-ancestry.json'
  side.write_text(json.dumps({'schema':'numi.human.bank109_114.rational_patch_ancestry.v1','candidate_sha256':sha(C),'exact_contour_graph_sha256':sha(GRAPH),'component_id':ci,'endpoint_nodes':endpoint_ancestry,'per_child_face_rational_barycentrics':[{'child_face_row':i,'parent_candidate_face_row':int(fi),'vertex_barycentrics':[[str(x) for x in b] for b in tri]} for i,(fi,tri) in enumerate(zip(child_parent,child_bary_exact))]},indent=2)+'\n')
  reports.append({'component_id':ci,'status':'CONFORMING_LOCAL_TRIANGULATION_PASS','patch_parent_face_count':len(parent_faces),'crossing_pairs':cg['exact_pair_count'],'contour_cycles':cg['endpoint_graph_component_sizes'],'contour_segment_count':len(edge_rows),'canonical_endpoint_count':len(roots),'output_node_count':len(out_points),'added_endpoint_node_count':len(roots),'child_face_count':len(child_faces),'outer_boundary_edge_count':len(boundary_edges),'outer_boundary_unchanged':new_boundary==old_boundary,'global_edge_incidence_histogram':dict(Counter(counts.values())),'all_contour_edges_incidence_two':all(counts.get(e)==2 for e in contour_global),'exact_parent_face_double_area':True,'constrained_triangulator_edge_flips':flips_total,'patch_npz_path':str(outpath),'patch_npz_sha256':sha(outpath),'rational_ancestry_path':str(side),'rational_ancestry_sha256':sha(side),'per_parent_face':per_face})
 except Exception as e:
  reports.append({'component_id':ci,'status':'CONFORMING_LOCAL_TRIANGULATION_FAILED','patch_parent_face_count':len(parent_faces),'crossing_pairs':cg['exact_pair_count'],'contour_cycles':cg['endpoint_graph_component_sizes'],'failure':repr(e),'traceback':traceback.format_exc()})
final={'schema':'numi.human.bank109_114.facewise_conforming_triangulation.v1','candidate_path':str(C),'candidate_sha256':sha(C),'exact_contour_graph_path':str(GRAPH),'exact_contour_graph_sha256':sha(GRAPH),'component_results':reports,'all_components_triangulated':len(reports)==2 and all(x['status']=='CONFORMING_LOCAL_TRIANGULATION_PASS' for x in reports),'existing_candidate_coordinates_moved':False,'existing_candidate_modified':False,'new_endpoint_coordinates':'Exact rational world points from skin barycentrics rounded to Float32 metres; only private patch outputs contain them.','physical_or_clearance_claim':False,'audit_rerun':False,'scope':'Face-wise constrained triangulation of the two fixed face-row patches using exact rational contour endpoints and inherited source ancestry; no full candidate or audit emitted.','elapsed_seconds':time.monotonic()-started}
(OUT/'facewise-triangulation-report.json').write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in final.items() if k!='component_results'},indent=2,allow_nan=False))
print(json.dumps([{k:v for k,v in x.items() if k!='per_parent_face' and k!='traceback'} for x in reports],indent=2,allow_nan=False))

