#!/usr/bin/env python3
import csv,gzip,hashlib,json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np

BASE=Path('/Users/numi/Work/numi-human-connected-surface-069')
CUT=BASE/'connected-source-patch-cut-123/run-019-route-harmonized-old-input'
COARSE=BASE/'publication-072/evidence/whole-reference-TetGen-084/whole-skin-fat-reference-084-candidate.npz'
DETAIL=BASE/'publication-072/evidence/run-006-direct-detail-p1-075/direct-detail-p1-075.npz'
CSV=BASE/'publication-071/evidence/exact-transfer-075/run-001/source-preflight/accepted-coarse-reference.csv.gz'
OUT=BASE/'connected-source-patch-cut-123/run-019-route-harmonized-old-emission'

if OUT.exists(): raise RuntimeError(f'output directory already exists: {OUT}')
OUT.mkdir(parents=True,exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def edge(a,b):
 a=int(a);b=int(b);return (a,b) if a<b else (b,a)

co=np.load(COARSE,allow_pickle=False); de=np.load(DETAIL,allow_pickle=False); z=np.load(CUT/'source-patch-cut.npz',allow_pickle=False); cr=json.load(open(CUT/'source-patch-cut-report.json'))
F=np.asarray(co['source_O_faces'],dtype=np.int64); fine=np.asarray(z['fine_subfaces'],dtype=np.int64); fine_parent=np.asarray(z['fine_subface_parent_rows'],dtype=np.int32)
coarse_points=np.asarray(co['reference_points_m'],dtype=np.float64); detail_points=np.asarray(de['detail_baseline_world_m'],dtype=np.float64)
nco=len(coarse_points); nd=len(detail_points)
coords=np.vstack([detail_points,np.asarray(z['virtual_anchor_points_world_m'],dtype=np.float64)])
nvirtual=len(coords)-nd
rows=[None]*2411
with gzip.open(CSV,'rt',newline='') as f:
 for r in csv.DictReader(f): rows[int(r['coarse_row'])]=r
if any(x is None for x in rows): raise RuntimeError('accepted coarse reference table incomplete')
stable=np.asarray(de['detail_source_vertex_id'],dtype=np.int64);id_to_row={int(v):i for i,v in enumerate(stable)}
coarse_anchor={}
direct_residual={}
for i,r in enumerate(rows):
 sid=int(r['source_direct_id'])
 if sid>=0 and sid in id_to_row:
  dr=id_to_row[sid];coarse_anchor[i]=dr
  cp=np.array([float(r[f'world_{a}_m']) for a in 'xyz'])
  direct_residual[i]=float(np.linalg.norm(detail_points[dr]-cp))
for cr0,node in zip(z['virtual_anchor_coarse_rows'],z['virtual_anchor_nodes']):
 coarse_anchor[int(cr0)]=int(node)
if not coarse_anchor: raise RuntimeError('no accepted direct or virtual anchors')
if len(set(coarse_anchor.values()))!=len(coarse_anchor): raise RuntimeError('anchor map not injective')

# Segment/path rows are one-to-one with the report descriptor order.
ip=z['route_path_indptr']; flat=z['route_path_nodes']
paths=[flat[ip[i]:ip[i+1]].astype(int).tolist() for i in range(len(ip)-1)]
segments=[]
for j,meta in enumerate(cr['routed_segments']):
 x=dict(meta);x['path_nodes']=paths[j];segments.append(x)
loops=cr['patch_loops']
cut_edges=set(map(tuple,np.asarray(z['routed_cut_edges'],dtype=np.int64).tolist()))
edgefaces=defaultdict(list)
for fi,t in enumerate(fine):
 for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):edgefaces[edge(a,b)].append(fi)
# Partition detailed source faces after all routed boundary cuts.
seen=set(); fcomps=[]
for start in range(len(fine)):
 if start in seen:continue
 stack=[start];seen.add(start); comp=[]
 while stack:
  f=stack.pop();comp.append(f);t=fine[f]
  for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
   ee=edge(a,b)
   if ee in cut_edges:continue
   for g in edgefaces[ee]:
    if g not in seen:seen.add(g);stack.append(g)
 fcomps.append(comp)
face_owner=np.empty(len(fine),dtype=np.int32)
for ci,fs in enumerate(fcomps):face_owner[fs]=ci
def directed_edge(tri,e):
 for i in range(3):
  a,b=int(tri[i]),int(tri[(i+1)%3])
  if edge(a,b)==e:return (a,b)
 return None

selected_loop={}
selection_rows=[]
for rec in loops:
 if rec.get('status')!='closed_disjoint_source_edge_cycle':continue
 ci=int(rec['component']);li=int(rec['loop'])
 seg=[x for x in segments if int(x['component'])==ci and int(x['loop'])==li]
 needed=[]
 for s in seg:
  p=s['path_nodes'];needed.extend(zip(p[:-1],p[1:]))
 candidates=[]
 for fc,faces in enumerate(fcomps):
  okay=True
  for a,b in needed:
   fs=edgefaces.get(edge(a,b),[])
   inner=[f for f in fs if face_owner[f]==fc]
   if len(inner)!=1 or directed_edge(fine[inner[0]],edge(a,b))!=(int(b),int(a)):
    okay=False;break
  if okay:candidates.append(fc)
 if len(candidates)==1:
  selected_loop[(ci,li)]=candidates[0];rec['selected_fine_side_component']=candidates[0]
 else:
  rec['selected_fine_side_component']=None
 rec['orientation_side_candidate_count']=len(candidates)
 selection_rows.append({'coarse_component':ci,'loop':li,'fine_side_candidates':candidates,
  'selected_face_count':len(fcomps[candidates[0]]) if len(candidates)==1 else None,
  'selected_area_m2':float(sum(.5*np.linalg.norm(np.cross(coords[fine[f,1]]-coords[fine[f,0]],coords[fine[f,2]]-coords[fine[f,0]])) for f in fcomps[candidates[0]])) if len(candidates)==1 else None})

bycomp=defaultdict(list)
for r in loops:bycomp[int(r['component'])].append(r)
valid_components={}
for ci,recs in bycomp.items():
 if all(r.get('status')=='closed_disjoint_source_edge_cycle' and (ci,int(r['loop'])) in selected_loop for r in recs):
  valid_components[ci]=sorted({selected_loop[(ci,int(r['loop']))] for r in recs})
patch_coarse=set()
for ci,recs in bycomp.items():
 if ci in valid_components:
  patch_coarse.update(map(int,recs[0]['coarse_faces']))
patch_fine=set()
for cids in valid_components.values():
 for fc in cids:patch_fine.update(fcomps[fc])
# Prevent double extraction across coarse components.
if len(patch_fine)!=sum(len(fcomps[fc]) for cids in valid_components.values() for fc in cids):
 raise RuntimeError('selected fine patches overlap across coarse components')

# Keep the original coarse boundary fixed. Map every fine source-path vertex
# onto its original straight coarse edge by normalized source-path arclength.
coarse_to_node={}
for rec in loops:
 if rec.get('status')!='closed_disjoint_source_edge_cycle' or int(rec['component']) not in valid_components:continue
 for cr0,anc in zip(rec['coarse_loop'],rec['anchor_nodes']):
  cr0=int(cr0);anc=int(anc)
  old=coarse_to_node.get(cr0)
  if old is not None and old!=anc:raise RuntimeError(f'conflicting source anchor for coarse row {cr0}')
  coarse_to_node[cr0]=anc
source_node_to_coarse={}
for cr0,node in coarse_to_node.items():
 old=source_node_to_coarse.get(int(node))
 if old is not None and old!=cr0:raise RuntimeError(f'source anchor {node} aliases coarse rows {old} and {cr0}')
 source_node_to_coarse[int(node)]=int(cr0)

edge_path={}
path_target={}
path_parameter={}
path_source_edge={}
path_target_conflicts=[]
for s in segments:
 if int(s['component']) not in valid_components:continue
 cr0,cr1=map(int,(s['coarse_start'],s['coarse_end']))
 key=edge(cr0,cr1)
 if key in edge_path:raise RuntimeError(f'duplicate patch boundary edge {key}')
 p=list(map(int,s['path_nodes']))
 if p[0]!=coarse_anchor[cr0] or p[-1]!=coarse_anchor[cr1]:
  raise RuntimeError(f'path endpoint/accepted-anchor mismatch on coarse edge {cr0}-{cr1}')
 xyz=coords[np.asarray(p,dtype=np.int64)]
 lengths=np.linalg.norm(np.diff(xyz,axis=0),axis=1)
 total=float(lengths.sum())
 if not np.isfinite(total) or total<=0:raise RuntimeError(f'zero/nonfinite source arclength for coarse edge {cr0}-{cr1}')
 cum=np.concatenate(([0.0],np.cumsum(lengths)))/total
 if abs(float(cum[-1])-1.0)>1e-12:raise RuntimeError('path arclength normalization failure')
 for j,node in enumerate(p):
  node=int(node);t=float(cum[j])
  if j==0 or j==len(p)-1:
   expected=cr0 if j==0 else cr1
   prior=source_node_to_coarse.get(node)
   if prior is not None and prior!=expected:raise RuntimeError(f'path endpoint source node {node} conflicts with coarse rows {prior}/{expected}')
   source_node_to_coarse[node]=expected
  else:
   if node in source_node_to_coarse:raise RuntimeError(f'path interior source node {node} is also a coarse anchor')
   target=(1.0-t)*coarse_points[cr0]+t*coarse_points[cr1]
   old=path_target.get(node)
   if old is not None:
    d=float(np.linalg.norm(old-target))
    if d>1e-9:path_target_conflicts.append({'source_node':node,'first_edge':path_source_edge[node],'second_edge':[cr0,cr1],'target_disagreement_m':d})
    if d<=1e-9: continue
   else:
    path_target[node]=target
    path_parameter[node]=t
    path_source_edge[node]=(cr0,cr1)
 edge_path[key]=(cr0,cr1,p)
if path_target_conflicts:raise RuntimeError(f'incompatible shared path targets: {path_target_conflicts[:5]}')

# Check each selected fine component has exactly the routed paths as its boundary.
expected_boundary_by_fc=defaultdict(set)
for s in segments:
 ci,li=int(s['component']),int(s['loop'])
 fc=selected_loop.get((ci,li))
 if fc is None:continue
 p=list(map(int,s['path_nodes']))
 for a,b in zip(p[:-1],p[1:]):expected_boundary_by_fc[fc].add(edge(a,b))
actual_boundary_by_fc={}
for fc in sorted({f for cids in valid_components.values() for f in cids}):
 counts=Counter()
 for fi in fcomps[fc]:
  t=fine[fi]
  for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):counts[edge(a,b)]+=1
 actual={e for e,n in counts.items() if n==1}
 expected=expected_boundary_by_fc[fc]
 actual_boundary_by_fc[fc]=actual
 if actual!=expected:
  raise RuntimeError(f'fine patch boundary mismatch component {fc}: actual={len(actual)} expected={len(expected)} missing={len(expected-actual)} extra={len(actual-expected)}')

# Assign coarse-edge sample nodes and solve a positive graph-harmonic displacement
# for each selected fine patch interior. This is a construction blend only.
points=[p.copy() for p in coarse_points]
node_meta=[{'kind':0,'source_row':i,'source_virtual_row':-1,'edge_a':-1,'edge_b':-1,'edge_t':np.nan,'parent_coarse_face':-1,'source_displacement_m':0.0} for i in range(nco)]
path_output={}
for node in sorted(path_target):
 path_output[node]=len(points)
 points.append(path_target[node].copy())
 a,b=path_source_edge[node]
 node_meta.append({'kind':3 if node>=nd else 2,'source_row':node if node<nd else -1,
  'source_virtual_row':node-nd if node>=nd else -1,'edge_a':a,'edge_b':b,'edge_t':path_parameter[node],
  'parent_coarse_face':-1,'source_displacement_m':float(np.linalg.norm(path_target[node]-coords[node]))})

# Map source boundary vertices to the untouched coarse endpoint rows.
fine_node_output={int(node):int(row) for node,row in source_node_to_coarse.items()}
for node,out in path_output.items():
 if node in fine_node_output:raise RuntimeError(f'path sample {node} conflicts with endpoint mapping')
 fine_node_output[node]=out

harmonic_records=[]
patch_components=sorted({fc for cids in valid_components.values() for fc in cids})
for fc in patch_components:
 face_ids=np.asarray(fcomps[fc],dtype=np.int64)
 tris=fine[face_ids].astype(np.int64)
 patch_nodes=np.unique(tris)
 bnodes=np.asarray(sorted(set(map(int,patch_nodes)) & set(source_node_to_coarse.keys()) | (set(map(int,patch_nodes)) & set(path_output.keys()))),dtype=np.int64)
 # Every patch boundary vertex is prescribed by a routed source path.
 boundary_edges=actual_boundary_by_fc[fc]
 bset=set(map(int,bnodes))
 for a,b in boundary_edges:
  if int(a) not in bset or int(b) not in bset:
   raise RuntimeError(f'boundary edge endpoint lacks arclength target in fine component {fc}: {(int(a),int(b))}')
 interior=np.asarray([int(n) for n in patch_nodes if int(n) not in bset],dtype=np.int64)
 src=coords[patch_nodes]
 if len(interior):
  from scipy.sparse import coo_matrix
  from scipy.sparse.linalg import spsolve
  loc={int(n):i for i,n in enumerate(patch_nodes)}
  deg=np.zeros(len(patch_nodes),dtype=np.float64)
  rr=[];cc=[];vv=[]
  edge_set=set()
  for t in tris:
   for aa,bb in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
    aa=int(aa);bb=int(bb);ee=edge(aa,bb)
    if ee in edge_set:continue
    edge_set.add(ee)
    ll=float(np.linalg.norm(coords[aa]-coords[bb]))
    if not np.isfinite(ll) or ll<=1e-15:raise RuntimeError(f'zero/nonfinite source edge in fine patch {fc}: {ee}')
    w=1.0/ll
    ia,ib=loc[aa],loc[bb]
    deg[ia]+=w;deg[ib]+=w
    rr.extend((ia,ib));cc.extend((ib,ia));vv.extend((-w,-w))
  rr.extend(range(len(patch_nodes)));cc.extend(range(len(patch_nodes)));vv.extend(deg.tolist())
  L=coo_matrix((vv,(rr,cc)),shape=(len(patch_nodes),len(patch_nodes))).tocsr()
  bidx=np.asarray([loc[int(n)] for n in bnodes],dtype=np.int64)
  iidx=np.asarray([loc[int(n)] for n in interior],dtype=np.int64)
  prescribed=np.empty((len(bnodes),3),dtype=np.float64)
  for j,n in enumerate(bnodes):
   if int(n) in source_node_to_coarse: prescribed[j]=coarse_points[source_node_to_coarse[int(n)]]-coords[int(n)]
   else: prescribed[j]=path_target[int(n)]-coords[int(n)]
  if len(bidx)==0:raise RuntimeError(f'fine patch component {fc} has no Dirichlet boundary')
  disp=np.zeros((len(patch_nodes),3),dtype=np.float64);disp[bidx]=prescribed
  disp[iidx]=spsolve(L[iidx][:,iidx],-L[iidx][:,bidx]@prescribed)
  if not np.isfinite(disp).all():raise RuntimeError(f'nonfinite harmonic displacement in fine component {fc}')
  for n in interior:
   ni=int(n);out=len(points);fine_node_output[ni]=out
   q=coords[ni]+disp[loc[ni]]
   points.append(q)
   node_meta.append({'kind':4 if ni>=nd else 1,'source_row':ni if ni<nd else -1,
    'source_virtual_row':ni-nd if ni>=nd else -1,'edge_a':-1,'edge_b':-1,'edge_t':np.nan,
    'parent_coarse_face':-1,'source_displacement_m':float(np.linalg.norm(disp[loc[ni]]))})
  harmonic_records.append({'fine_component':int(fc),'patch_faces':int(len(face_ids)),'patch_nodes':int(len(patch_nodes)),
   'boundary_nodes':int(len(bnodes)),'interior_nodes':int(len(interior)),
   'max_boundary_displacement_m':float(np.linalg.norm(prescribed,axis=1).max(initial=0)),
   'max_interior_harmonic_displacement_m':float(np.linalg.norm(disp[iidx],axis=1).max(initial=0)) if len(iidx) else 0.0})
 else:
  for n in patch_nodes:
   ni=int(n)
   if ni in fine_node_output:continue
   raise RuntimeError(f'unmapped zero-interior patch vertex {ni} in component {fc}')

# Make sure every vertex used by selected source faces is now represented.
for fi in patch_fine:
 for n in fine[int(fi)]:
  if int(n) not in fine_node_output:raise RuntimeError(f'no hybrid node map for fine source vertex {int(n)}')

faces=[];face_kind=[];parent_coarse=[];parent_fine=[];parent_marker=[];center_parent=[]
def append_face(tri,kind,pc=-1,pf=-1,marker=-1):
 faces.append(list(map(int,tri)));face_kind.append(kind);parent_coarse.append(pc);parent_fine.append(pf);parent_marker.append(marker)

# Keep untouched coarse faces and subdivide each outside collar face exactly in
# its original plane; all inserted boundary samples lie on original coarse edges.
for fi,tri0 in enumerate(F):
 tri=list(map(int,tri0))
 if fi in patch_coarse:continue
 split_edges=[];poly=[]
 for i in range(3):
  u,v=tri[i],tri[(i+1)%3];key=edge(u,v)
  if key in edge_path:
   cu,cv,p=edge_path[key]
   if (u,v)==(cu,cv):ordered=list(map(int,p))
   elif (u,v)==(cv,cu):ordered=list(map(int,p[::-1]))
   else:raise RuntimeError('collar edge orientation mismatch')
   chain=[u]+[fine_node_output[n] for n in ordered[1:-1]]+[v]
   poly.extend(chain[:-1]);split_edges.append((chain,int(tri[(i+2)%3])))
  else:poly.append(u)
 if len(set(poly))!=len(poly):raise RuntimeError(f'collar polygon repeats nodes on coarse face {fi}')
 if not split_edges:
  append_face(tri,0,fi,-1,fi)
 elif len(split_edges)==1:
  chain,opp=split_edges[0]
  for j in range(len(chain)-1):append_face([chain[j],chain[j+1],opp],1,fi,-1,fi)
 else:
  center=np.mean(coarse_points[np.asarray(tri,dtype=np.int64)],axis=0)
  center_id=len(points);points.append(center)
  node_meta.append({'kind':5,'source_row':-1,'source_virtual_row':-1,'edge_a':-1,'edge_b':-1,'edge_t':np.nan,'parent_coarse_face':fi,'source_displacement_m':0.0})
  center_parent.append(fi)
  for j in range(len(poly)):append_face([center_id,poly[j],poly[(j+1)%len(poly)]],1,fi,-1,fi)

# Replace selected coarse patch faces with the selected fine source faces,
# reversed to match the compact O-surface orientation.
for fi in sorted(patch_fine):
 t=fine[int(fi)]
 append_face([fine_node_output[int(t[0])],fine_node_output[int(t[2])],fine_node_output[int(t[1])]],2,int(fine_parent[fi]),int(fi),int(fine_parent[fi]))

P=np.asarray(points,dtype=np.float64);T=np.asarray(faces,dtype=np.int32)
if T.ndim!=2 or T.shape[1]!=3:raise RuntimeError('bad emitted face shape')
used=np.unique(T);remap=np.full(len(P),-1,dtype=np.int64);remap[used]=np.arange(len(used))
Tc=remap[T].astype(np.int32);Pc=P[used]
meta=[node_meta[int(i)] for i in used]
node_kind=np.asarray([m['kind'] for m in meta],dtype=np.int8)
node_source=np.asarray([m['source_row'] for m in meta],dtype=np.int64)
node_virtual_coarse=np.asarray([m['source_virtual_row'] for m in meta],dtype=np.int32)
node_edge_a=np.asarray([m['edge_a'] for m in meta],dtype=np.int32)
node_edge_b=np.asarray([m['edge_b'] for m in meta],dtype=np.int32)
node_edge_t=np.asarray([m['edge_t'] for m in meta],dtype=np.float64)
node_parent_coarse=np.asarray([m['parent_coarse_face'] for m in meta],dtype=np.int32)
node_source_displacement=np.asarray([m['source_displacement_m'] for m in meta],dtype=np.float64)
face_kind=np.asarray(face_kind,dtype=np.int8);parent_coarse=np.asarray(parent_coarse,dtype=np.int32);parent_fine=np.asarray(parent_fine,dtype=np.int32);parent_marker=np.asarray(parent_marker,dtype=np.int32)
# Topological manifold, area, and orientation diagnostics. These are construction
# checks only and do not replace the exact self-intersection predicate.
ec=Counter()
for t in Tc:
 for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):ec[edge(a,b)]+=1
boundary=[e for e,n in ec.items() if n==1]
nonman=[(e,n) for e,n in ec.items() if n>2]
areas=np.empty(len(Tc),dtype=np.float64);zero=[]
for i,t in enumerate(Tc):
 a,b,c=Pc[t];areas[i]=0.5*np.linalg.norm(np.cross(b-a,c-a))
 if not np.isfinite(areas[i]) or areas[i]<=1e-16:zero.append(i)
coarse_normal_cos=[];patch_source_normal_cos=[]
for i,k in enumerate(face_kind):
 if k in (0,1):
  pc=int(parent_coarse[i]);old=F[pc]
  a,b,c=Pc[Tc[i]];oa,ob,oc=coarse_points[old]
  n=np.cross(b-a,c-a);on=np.cross(ob-oa,oc-oa);den=np.linalg.norm(n)*np.linalg.norm(on)
  coarse_normal_cos.append(float(np.dot(n,on)/den) if den>0 else -1.0)
 elif k==2:
  fi=int(parent_fine[i]);t0=fine[fi]
  a,b,c=Pc[Tc[i]]
  sa,sb,sc=coords[t0]
  n=np.cross(b-a,c-a);sn=-np.cross(sb-sa,sc-sa);den=np.linalg.norm(n)*np.linalg.norm(sn)
  patch_source_normal_cos.append(float(np.dot(n,sn)/den) if den>0 else -1.0)
report={
 'status':'hybrid_O_surface_emitted_arclength_anchored_harmonic_source_patch_topology_only',
 'interpretation':'coarse boundary vertices remain fixed; fine boundary paths are mapped by normalized source arclength to original straight coarse edges; selected source-patch interiors use positive inverse-edge-length harmonic displacement; outside collar is an exact parent-face partition. No exact self-intersection, bone clearance, I/D conformity, tetrahedron, material-transfer, mass, or native admission gate run.',
 'inputs':{'cut_report':str(CUT/'source-patch-cut-report.json'),'cut_report_sha256':sha(CUT/'source-patch-cut-report.json'),
  'cut_npz':str(CUT/'source-patch-cut.npz'),'cut_npz_sha256':sha(CUT/'source-patch-cut.npz'),
  'coarse084':str(COARSE),'coarse084_sha256':sha(COARSE),'direct_detail075':str(DETAIL),'direct_detail075_sha256':sha(DETAIL),
  'accepted_coarse_csv_gz':str(CSV),'accepted_coarse_csv_gz_sha256':sha(CSV)},
 'anchor_mapping':{'rule':'accepted_csv source_direct_id -> detailed vertex row; otherwise explicit CSV source face+barycentric virtual anchor',
  'accepted_direct_rows':len(direct_residual),'virtual_anchor_rows':z['virtual_anchor_coarse_rows'].astype(int).tolist(),
  'direct_residual_max_m':max(direct_residual.values(),default=0),'direct_residual_p99_m':float(np.quantile(list(direct_residual.values()),.99)) if direct_residual else 0},
 'topology':{'coarse_O_faces_input':len(F),'seed_coarse_components':len(bycomp),'selected_source_patch_components':len(valid_components),
  'untouched_seed_components':sorted(set(bycomp)-set(valid_components)),'routed_loops':len(selected_loop),
  'selected_fine_faces':len(patch_fine),
  'harmonic_components':harmonic_records,
  'path_target_sample_count':len(path_target),
  'path_target_min_segment_m':float(min((np.linalg.norm(path_target[a]-path_target[b]) for _,(_,_,p) in edge_path.items() for a,b in zip(p[1:-1],p[2:-1])),default=0.0)),
  'max_source_path_target_displacement_m':float(max((np.linalg.norm(path_target[n]-coords[n]) for n in path_target),default=0.0)),
  'unchanged_coarse_boundary_node_count':len(coarse_to_node),
  'patch_boundary_exact_path_match':True,'collar_faces':int(np.count_nonzero(face_kind==1)),
  'untouched_coarse_faces':int(np.count_nonzero(face_kind==0)),'output_O_faces':len(Tc),
  'output_O_nodes':len(Pc),'source_fine_nodes_in_output':int(np.count_nonzero(node_kind==1)),
  'virtual_anchor_nodes_in_output':int(np.count_nonzero(node_kind==2)),
  'collar_centroid_nodes':int(np.count_nonzero(node_kind==5)),
  'arclength_boundary_samples':int(np.count_nonzero((node_kind==2)|(node_kind==3))),
  'harmonic_detail_interior_nodes':int(np.count_nonzero((node_kind==1)|(node_kind==4))),
  'max_detail_node_displacement_m':float(node_source_displacement.max(initial=0.0)),
  'boundary_edge_count':len(boundary),'nonmanifold_edge_count':len(nonman),
  'nonmanifold_edge_witnesses':[[list(e),n] for e,n in nonman[:100]],
  'zero_or_tiny_face_count':len(zero),'min_face_area_m2':float(areas.min()),
  'coarse_descendant_normal_cos_min':float(min(coarse_normal_cos)) if coarse_normal_cos else None,
  'coarse_descendant_normal_cos_p01':float(np.quantile(coarse_normal_cos,.01)) if coarse_normal_cos else None,
  'fine_patch_output_vs_reversed_source_normal_cos_min':float(min(patch_source_normal_cos)) if patch_source_normal_cos else None,
  'fine_patch_output_vs_reversed_source_normal_cos_p01':float(np.quantile(patch_source_normal_cos,.01)) if patch_source_normal_cos else None,
  'component_side_selection':selection_rows},
 'patch_loop_outcomes':[{'component':int(r['component']),'loop':r.get('loop'),'status':r['status'],
  'selected_side':r.get('selected_fine_side_component'),'side_candidates':r.get('orientation_side_candidate_count')} for r in loops],
 'outputs':{'candidate_npz':str(OUT/'hybrid-O-surface-129-old-harmonized.npz')}
}
# Preserve each route's output chain and normalized source-arclength parameters.
edge_keys=sorted(edge_path)
chain_indptr=[0];chain_outputs=[];chain_source_nodes=[];chain_parameters=[]
for k in edge_keys:
 cr0,cr1,p=edge_path[k]
 chain_outputs.extend([cr0]+[fine_node_output[n] for n in p[1:-1]]+[cr1])
 chain_source_nodes.extend(p)
 xyz=coords[np.asarray(p,dtype=np.int64)];lengths=np.linalg.norm(np.diff(xyz,axis=0),axis=1)
 tv=np.concatenate(([0.0],np.cumsum(lengths)))/float(lengths.sum())
 chain_parameters.extend(tv.tolist())
 chain_indptr.append(len(chain_source_nodes))
np.savez_compressed(OUT/'hybrid-O-surface-129-old-harmonized.npz',points_world_m=Pc,source_O_faces=Tc,
 node_kind=node_kind,node_source_row=node_source,node_source_virtual_row=node_virtual_coarse,
 node_parent_coarse_face_row=node_parent_coarse,node_parent_coarse_edge_a=node_edge_a,node_parent_coarse_edge_b=node_edge_b,
 node_parent_coarse_edge_parameter=node_edge_t,node_source_displacement_m=node_source_displacement,
 face_kind=face_kind,parent_coarse_face_row=parent_coarse,parent_fine_detail_face_row=parent_fine,parent_source_face_marker=parent_marker,
 coarse_boundary_anchor_rows=np.asarray(sorted(coarse_to_node),dtype=np.int32),
 coarse_boundary_anchor_source_nodes=np.asarray([coarse_to_node[i] for i in sorted(coarse_to_node)],dtype=np.int32),
 coarse_boundary_edge_endpoints=np.asarray([list(k) for k in edge_keys],dtype=np.int32),
 coarse_boundary_edge_path_indptr=np.asarray(chain_indptr,dtype=np.int64),
 coarse_boundary_edge_path_output_nodes=np.asarray(chain_outputs,dtype=np.int32),
 coarse_boundary_edge_path_source_nodes=np.asarray(chain_source_nodes,dtype=np.int32),
 coarse_boundary_edge_path_parameters=np.asarray(chain_parameters,dtype=np.float64),
 selected_coarse_face_rows=np.asarray(sorted(patch_coarse),dtype=np.int32),
 selected_fine_subface_rows=np.asarray(sorted(patch_fine),dtype=np.int32),
 failed_or_unselected_coarse_components=np.asarray(sorted(set(bycomp)-set(valid_components)),dtype=np.int32))
report['outputs']['candidate_sha256']=sha(OUT/'hybrid-O-surface-129-old-harmonized.npz')
(OUT/'hybrid-O-surface-129-old-harmonized-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'out':str(OUT),'hash':report['outputs']['candidate_sha256'],'topology':report['topology'],
 'selection_counts':len(selection_rows),'valid_components':len(valid_components)},indent=2))
