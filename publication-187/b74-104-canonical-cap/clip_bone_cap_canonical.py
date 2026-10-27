import sys,json,hashlib,importlib.util
from pathlib import Path
from collections import Counter,defaultdict,deque
from fractions import Fraction
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
S=ROOT/'D/surface-clearance-546/component-sat-selector-638'
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
helper_path=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
exact_path=E/'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py'
sys.path.insert(0,str(E/'incremental-exact-audit-owner-253/src'))
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
h=module('bankhelper',helper_path)
ROOTC=ROOT/'D/publication-182-stage-001/combined-653/candidate.npz'
out=S/'per-bank-workers-20261027/b74-104/steiner-multinode-006-bone-cap-canonical'
Z=np.load(ROOTC,allow_pickle=False);X=np.asarray(Z['points_world_m'],np.float32);F=np.asarray(Z['source_O_faces'],np.int64)
basehash=hashlib.sha256(ROOTC.read_bytes()).hexdigest()
W=json.load(open(S/'per-bank-workers-20261027/b74-104/baseline-witnesses.json'))
G=json.load(open(S/'per-bank-workers-20261027/b74-104/steiner-multinode-002-segment-guided/segment-graph-report.json'))
BND=json.load(open(S/'per-bank-workers-20261027/b74-104/steiner-multinode-001/boundary-feasibility.json'))
old_rows=list(map(int,W['skin_child_faces']));loop=[int(x['node']) for x in BND['fixed_boundary_vertices']]
if G['candidate_sha256']!=basehash:raise RuntimeError('exact segment graph does not match 653 candidate')
node_bary={}
for fr in old_rows:
 for n,b in zip(F[fr],Z['refinement_face_parent_barycentric'][fr]):node_bary[int(n)]=np.asarray(b,np.float64)
if set(node_bary)!=set(loop):raise RuntimeError('source-face boundary ancestry mismatch')
corners=[next(n for n,b in node_bary.items() if np.array_equal(b,np.eye(3)[i])) for i in range(3)]
EVID=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
T,ids,gids,bodies,M,off,pos,keys,pins,_=h.read_bank_inputs(EVID,E)
bankmask=(ids[:,0]==74)&(ids[:,1]==104);B=np.asarray(T[bankmask],np.float32);bid=np.asarray(ids[bankmask,3],np.int64)
R=M[74];O=off[74];P=pos[74]
to_local=lambda v:(np.asarray(v,dtype=np.float64)-P)@R.T-O
to_world=lambda v:(R.T@(np.asarray(v,dtype=np.float64)+O)+P).astype(np.float32)
base_local=to_local(X)
# Parent source-face plane and barycentric projection basis.
corner_xyz=base_local[np.asarray(corners)]
plane_n=np.cross(corner_xyz[1]-corner_xyz[0],corner_xyz[2]-corner_xyz[0]);plane_n/=np.linalg.norm(plane_n)
basis=np.column_stack([corner_xyz[1]-corner_xyz[0],corner_xyz[2]-corner_xyz[0]])
def source_bary(p):
 uv=np.linalg.lstsq(basis,np.asarray(p)-corner_xyz[0],rcond=None)[0]
 projected=corner_xyz[0]+basis@uv
 return np.asarray([1-uv.sum(),uv[0],uv[1]],np.float64),float(np.linalg.norm(projected-p))
def cross2(a,b,c):return float((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
def area2(poly):return float(sum(poly[i,0]*poly[(i+1)%len(poly),1]-poly[(i+1)%len(poly),0]*poly[i,1] for i in range(len(poly)))*0.5)
def point_in_poly(p,poly):
 inside=False
 for i in range(len(poly)):
  a=poly[i];b=poly[(i+1)%len(poly)]
  if abs(cross2(a,b,p))<1e-12 and min(a[0],b[0])-1e-12<=p[0]<=max(a[0],b[0])+1e-12 and min(a[1],b[1])-1e-12<=p[1]<=max(a[1],b[1])+1e-12:return True
  if ((a[1]>p[1])!=(b[1]>p[1])) and p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0]:inside=not inside
 return inside
def proper_intersects(a,b,c,d):
 o1=cross2(a,b,c);o2=cross2(a,b,d);o3=cross2(c,d,a);o4=cross2(c,d,b)
 return o1*o2<-1e-24 and o3*o4<-1e-24
def on_segment(p,a,b,tol=1e-12):
 ab=b-a;ap=p-a
 return abs(ab[0]*ap[1]-ab[1]*ap[0])<=tol and float(ap@ab)>tol and float(ap@ab)<float(ab@ab)-tol
# Exact witness contour in source barycentric coordinates.
gpoints=np.asarray(G['intersection_points_barycentric'],np.float64)
graph_edges={tuple(sorted(map(int,e))) for e in G['segments_point_indices']}
npts=len(gpoints);adj=[[] for _ in range(npts)]
for a,b in graph_edges:adj[a].append(b);adj[b].append(a)
if any(len(q)!=2 for q in adj):raise RuntimeError('exact crossing witness graph is not a degree-2 loop')
cycle=[0];prev=-1;cur=0
while True:
 nxt=next(q for q in adj[cur] if q!=prev)
 if nxt==cycle[0]:break
 if nxt in cycle:raise RuntimeError('exact witness contour closes before covering all vertices')
 cycle.append(nxt);prev,cur=cur,nxt
if len(cycle)!=npts:raise RuntimeError('exact crossing witness graph has multiple components')
contour_xy=gpoints[np.asarray(cycle),1:]
if area2(contour_xy)<0:cycle=list(reversed(cycle));contour_xy=contour_xy[::-1]
contour_area=abs(area2(contour_xy))
# Deduplicate exact source-bank mesh vertices and verify closed edge incidence.
vmap={};vertices=[];face_vids=[]
for tri in B:
 row=[]
 for q in tri:
  key=tuple(np.asarray(q,np.float32).view(np.uint32).tolist())
  if key not in vmap:vmap[key]=len(vertices);vertices.append(np.asarray(q,np.float64))
  row.append(vmap[key])
 face_vids.append(row)
V=np.asarray(vertices,np.float64);face_vids=np.asarray(face_vids,np.int64)
edge_faces=defaultdict(list);directed=defaultdict(list);vertex_faces=defaultdict(set)
for fi,face in enumerate(face_vids):
 for v in face:vertex_faces[int(v)].add(fi)
 for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):
  k=tuple(sorted((int(a),int(b))));edge_faces[k].append(fi);directed[k].append((int(a),int(b)))
closed_edges=all(len(v)==2 for v in edge_faces.values())
oriented_edges=all(len(v)==2 and v[0]==(v[1][1],v[1][0]) for v in directed.values())
# Clip each bone triangle to either side of the source-face plane.
plane_tol=2e-9
dvert=(V-corner_xyz[0])@plane_n
edgepoint_cache={}
def get_cut_edge(va,vb):
 lo,hi=sorted((int(va),int(vb)));key=(lo,hi)
 if key not in edgepoint_cache:
  d0=float(dvert[lo]);d1=float(dvert[hi])
  if abs(d0)<=plane_tol: q=V[lo].copy()
  elif abs(d1)<=plane_tol:q=V[hi].copy()
  elif abs(d0-d1)<1e-24:raise RuntimeError('parallel bone edge requested as a cut')
  else:q=V[lo]+(d0/(d0-d1))*(V[hi]-V[lo])
  edgepoint_cache[key]=q
 q=edgepoint_cache[key]
 if abs(float(dvert[lo]))<=plane_tol:tok=('v',lo)
 elif abs(float(dvert[hi]))<=plane_tol:tok=('v',hi)
 else:tok=('x',lo,hi)
 tok=canonical_contour_token(tok,q)
 return tok,(gpoints_local[int(tok[1])].copy() if tok[0]=='c' else q.copy())
def point_for(tok):
 if tok[0]=='v':return V[int(tok[1])]
 if tok[0]=='c':return gpoints_local[int(tok[1])]
 return edgepoint_cache[(int(tok[1]),int(tok[2]))]
gpoints_local=np.asarray([b@base_local[np.asarray(corners)] for b in gpoints],np.float64)
endpoint_mapping_witnesses=[]
endpoint_mapping_tolerance_m=1e-6
def canonical_contour_token(tok,q):
 d=np.linalg.norm(gpoints_local-np.asarray(q,dtype=np.float64),axis=1)
 pid=int(np.argmin(d));dist=float(d[pid])
 if dist<=endpoint_mapping_tolerance_m:
  tied=np.flatnonzero(np.abs(d-dist)<=1e-12)
  if len(tied)!=1:raise RuntimeError(f'ambiguous contour endpoint mapping for {tok}: {tied.tolist()}')
  endpoint_mapping_witnesses.append({'source_token':list(tok),'contour_point_id':pid,'distance_m':dist})
  return ('c',pid)
 return tok
# Associate exact contour vertices with every intersecting stable bone face from the exact pair witnesses.
face_graph_points=defaultdict(set)
for rec in G['segment_crossing_pairs']:
 bid0=int(rec['bone_triangle_row'])
 for pid in rec['point_ids']:face_graph_points[bid0].add(int(pid))
plane_mismatch=[]
for pid,p in enumerate(gpoints_local):
 bary,err=source_bary(p)
 if err>5e-8:plane_mismatch.append({'point_id':pid,'residual_m':err})
faceid_to_index={int(x):i for i,x in enumerate(bid)}
def clip_face(fi,side):
 vids=list(map(int,face_vids[fi])); poly=[]
 for v in vids:
  tok=('v',v);tok=canonical_contour_token(tok,V[v])
  poly.append((tok,gpoints_local[int(tok[1])].copy() if tok[0]=='c' else V[v].copy()))
 def dist(tok,q):
  return float(np.dot(q-corner_xyz[0],plane_n))
 def inside(item):return side*dist(item[0],item[1])>=-plane_tol
 outpoly=[]
 for idx,curitem in enumerate(poly):
  previtem=poly[idx-1];pin=inside(previtem);cin=inside(curitem)
  if pin and cin:outpoly.append(curitem)
  elif pin and not cin:outpoly.append(get_cut_edge(previtem[0][1],curitem[0][1]))
  elif not pin and cin:
   outpoly.append(get_cut_edge(previtem[0][1],curitem[0][1]));outpoly.append(curitem)
 # Remove consecutive duplicate tokens.
 clean=[]
 for it in outpoly:
  if not clean or clean[-1][0]!=it[0]:clean.append(it)
 if len(clean)>1 and clean[0][0]==clean[-1][0]:clean.pop()
 if len(clean)<3:return []
 # Split the plane-cut edge with exact segment witness points on this bone face.
 stable=int(bid[fi]);pids=sorted(face_graph_points.get(stable,set()))
 if pids:
  changed=True
  for qi in range(len(clean)):
   qj=(qi+1)%len(clean);a_tok,a=clean[qi];b_tok,b=clean[qj]
   da=dist(a_tok,a);db=dist(b_tok,b)
   if abs(da)>5e-8 or abs(db)>5e-8:continue
   vec=b-a;ll=float(vec@vec)
   if ll<=1e-20:continue
   line=[];bad=[]
   for pid in pids:
    p=gpoints_local[pid];t=float((p-a)@vec/ll);res=float(np.linalg.norm(p-(a+t*vec)))
    if -1e-5<=t<=1+1e-5 and res<=1e-6:line.append((t,pid,res))
    elif res>1e-6 and -1e-5<=t<=1+1e-5:bad.append((pid,res))
   line.sort()
   if len(line)<2:continue
   # Require exact endpoint witnesses to span this clipped cut segment.
   endpoint_ok=(np.linalg.norm(gpoints_local[line[0][1]]-a)<=1e-6 and np.linalg.norm(gpoints_local[line[-1][1]]-b)<=1e-6)
   if not endpoint_ok and np.linalg.norm(gpoints_local[line[-1][1]]-a)<=1e-6 and np.linalg.norm(gpoints_local[line[0][1]]-b)<=1e-6:
    line=list(reversed(line));endpoint_ok=True
   if not endpoint_ok:continue
   # Replace this plane edge by the ordered exact witness subsegments.
   # Canonicalize both endpoint identities as well as every interior contour point.
   interior=[(('c',pid),gpoints_local[pid].copy()) for _,pid,_ in line[1:-1]]
   edge_start=(('c',int(line[0][1])),gpoints_local[int(line[0][1])].copy())
   edge_end=(('c',int(line[-1][1])),gpoints_local[int(line[-1][1])].copy())
   # Rotate around the cyclic polygon so qi->qj is the first edge; this also
   # handles qj==0 without duplicating the polygon through Python slice wrap.
   rot=clean[qi:]+clean[:qi]
   if len(rot)<2:raise RuntimeError('clipped polygon lost a plane-cut edge')
   clean=[edge_start]+interior+[edge_end]+rot[2:]
   if clean and clean[0][0]==clean[-1][0]:clean.pop()
   changed=False
   break
 # Return convex clipped polygon tokens; later fan from a non-plane bone vertex.
 return clean
def polygon_to_triangles(poly,fi):
 if len(poly)<3:return []
 # Pivot at a bone vertex strictly off the cutting plane; clipped polygon is convex.
 pivots=[i for i,(t,q) in enumerate(poly) if t[0]=='v' and abs(float(np.dot(q-corner_xyz[0],plane_n)))>5e-8]
 if not pivots:return []
 p=pivots[0];rot=poly[p:]+poly[:p];res=[]
 for k in range(1,len(rot)-1):
  tri=[rot[0][0],rot[k][0],rot[k+1][0]]
  pts=[rot[0][1],rot[k][1],rot[k+1][1]]
  if len(set(tri))<3:continue
  if np.linalg.norm(np.cross(pts[1]-pts[0],pts[2]-pts[0]))<1e-16:continue
  res.append((tri,int(bid[fi])))
 return res
def side_mesh(side):
 tris=[];facecounts=Counter();bad_faces=[]
 for fi in range(len(B)):
  poly=clip_face(fi,side)
  if len(poly)>=3:tris.extend(polygon_to_triangles(poly,fi))
 # Edge-based connected components.
 edge_to_tris=defaultdict(list)
 for ti,(tri,boneid) in enumerate(tris):
  for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):edge_to_tris[tuple(sorted((a,b)))].append(ti)
 nbr=[set() for _ in tris]
 for e,ts in edge_to_tris.items():
  if len(ts)==2:nbr[ts[0]].add(ts[1]);nbr[ts[1]].add(ts[0])
 unseen=set(range(len(tris)));comps=[]
 while unseen:
  seed=min(unseen);q=[seed];unseen.remove(seed);comp=[]
  while q:
   t=q.pop();comp.append(t)
   for u in nbr[t]:
    if u in unseen:unseen.remove(u);q.append(u)
  comps.append(comp)
 compdata=[]
 target={tuple(sorted((('c',a),('c',b)))) for a,b in graph_edges}
 for comp in comps:
  ec=Counter()
  for ti in comp:
   tri=tris[ti][0]
   for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):ec[tuple(sorted((a,b)))]+=1
  bd={e for e,n in ec.items() if n==1};badinc={str(e):n for e,n in ec.items() if n>2 or (n==1 and e not in target)}
  faceids=[tris[ti][1] for ti in comp]
  compdata.append({'face_indices':comp,'face_count':len(comp),'boundary_edge_count':len(bd),'boundary_edges_match_exact_contour':bd==target,'internal_edge_incidence_valid':not badinc and all(n in (1,2) for n in ec.values()),'boundary_mismatch':{'missing':[str(e) for e in sorted(target-bd,key=str)],'extra':[str(e) for e in sorted(bd-target,key=str)]},'bad_edge_incidence':badinc,'bone_face_rows':faceids})
 return tris,compdata
# Cap projection checks.
def token_bary(token):
 if token[0]=='c':return np.asarray(gpoints[int(token[1])],np.float64),0.0
 p=point_for(token);return source_bary(p)
def project_cap(tris,comp,side):
 selected=[tris[i] for i in comp]
 tokens=sorted({t for tri,_ in selected for t in tri},key=str)
 bary_by_token={};residual_by_token={}
 for t in tokens:bary_by_token[t],residual_by_token[t]=token_bary(t)
 xy_by_token={t:bary_by_token[t][1:] for t in tokens}
 mesh_edges=Counter()
 for tri,boneid in selected:
  for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):mesh_edges[tuple(sorted((a,b)))]+=1
 target={tuple(sorted((('c',a),('c',b)))) for a,b in graph_edges}
 boundary={e for e,c in mesh_edges.items() if c==1}
 edge_ok=boundary==target and all(c in (1,2) for c in mesh_edges.values())
 bary_valid=all(np.min(b)>=-1e-8 and np.max(b)<=1+1e-8 and abs(float(np.sum(b))-1)<1e-8 and point_in_poly(xy_by_token[t],contour_xy) for t,b in bary_by_token.items())
 # Projected face areas, orientation and interior tests.
 signed=[];centroid_ok=True
 projected_tris=[]
 for tri,_ in selected:
  q=np.asarray([xy_by_token[t] for t in tri]);a=cross2(q[0],q[1],q[2])*0.5;signed.append(a);projected_tris.append(q)
  if not point_in_poly(q.mean(axis=0),contour_xy):centroid_ok=False
 orient_ok=bool(signed) and (all(x>1e-14 for x in signed) or all(x<-1e-14 for x in signed))
 proj_area=float(sum(abs(x) for x in signed))
 unique_edges=sorted(mesh_edges);crossings=[];tjs=[]
 for i,e in enumerate(unique_edges):
  a=xy_by_token[e[0]];b=xy_by_token[e[1]]
  for f in unique_edges[i+1:]:
   if set(e)&set(f):continue
   c=xy_by_token[f[0]];d=xy_by_token[f[1]]
   if proper_intersects(a,b,c,d):crossings.append([str(e),str(f)])
  for t,p in xy_by_token.items():
   if t in e:continue
   if on_segment(p,a,b):tjs.append([str(t),str(e)])
  if crossings:break
 planar_ok=edge_ok and bary_valid and centroid_ok and orient_ok and not crossings and not tjs and abs(proj_area-contour_area)<=1e-10
 # Physical cap area and all lineage information.
 surface_area=0.0
 for tri,_ in selected:
  p=[point_for(t) for t in tri]
  surface_area+=float(np.linalg.norm(np.cross(p[1]-p[0],p[2]-p[0]))*0.5)
 return {'side_sign':side,'face_indices':comp,'selected_face_count':len(selected),'unique_token_count':len(tokens),'physical_surface_area_m2':surface_area,'projected_area_barycentric2':proj_area,'contour_area_barycentric2':contour_area,'projected_area_error':proj_area-contour_area,'boundary_edge_count':len(boundary),'boundary_edges_match_exact_contour':boundary==target,'internal_edge_incidence_valid':edge_ok,'barycentric_source_ancestry_valid':bary_valid,'projected_orientation_consistent':orient_ok,'all_face_centroids_inside_contour':centroid_ok,'projected_edge_crossings':crossings,'projected_t_junctions':tjs,'planar_pass':planar_ok,'tokens':tokens,'bary_by_token':bary_by_token,'residual_by_token':residual_by_token,'bone_faces':selected,'projected_tris':projected_tris}
# Prepare contour polygon and exact bone-face pair input.
if contour_area<=0:raise RuntimeError('exact segment contour has zero area')
if not BND['fixed_boundary_vertices_all_outside'] or not BND['fixed_boundary_edges_all_disjoint_from_bone']:raise RuntimeError('fixed boundary not certified')
if len(G['segment_crossing_pairs'])!=25 or len(graph_edges)!=25:raise RuntimeError('expected exact 25-segment contour input')
report={'schema':'numi.human.b74_104_bone_clipped_cap_planar.v1','input_candidate_path':str(ROOTC),'input_candidate_sha256':basehash,'segment_graph_report_path':str(S/'per-bank-workers-20261027/b74-104/steiner-multinode-002-segment-guided/segment-graph-report.json'),'segment_graph_report_sha256':hashlib.sha256((S/'per-bank-workers-20261027/b74-104/steiner-multinode-002-segment-guided/segment-graph-report.json').read_bytes()).hexdigest(),'body':74,'stable_bone_id':104,'fixed_boundary_nodes':loop,'fixed_boundary_vertices_all_outside':True,'fixed_boundary_edges_all_disjoint_from_bone':True,'exact_contour_segments':len(graph_edges),'exact_contour_vertices':npts,'exact_contour_area_barycentric2':contour_area,'max_contour_source_plane_projection_residual_m':G['max_projection_residual_m'],'bone_triangle_count':len(B),'bone_unique_vertex_count':len(V),'bone_unique_edge_count':len(edge_faces),'bone_mesh_closed_edge_incidence':closed_edges,'bone_mesh_consistently_oriented':oriented_edges,'plane_vertex_tolerance_m':plane_tol,'contour_point_plane_mismatches':plane_mismatch,'boundary_side_results':[],'planar_cap_candidates':[],'status':'PREFLIGHT'}
all_side_data={}
for side in (1,-1):
 tris,comps=side_mesh(side);all_side_data[side]=(tris,comps)
 summ=[]
 for c in comps:
  # Preserve component boundary evidence and only project components with exact contour boundary.
  if c['boundary_edges_match_exact_contour'] and c['internal_edge_incidence_valid']:
   q=project_cap(tris,c['face_indices'],side)
   report['planar_cap_candidates'].append({k:v for k,v in q.items() if k not in ('tokens','bary_by_token','residual_by_token','bone_faces','projected_tris')})
   if q['planar_pass']:summ.append(q)
  else:summ.append(None)
 report['boundary_side_results'].append({'side_sign':side,'clipped_triangle_count':len(tris),'connected_component_count':len(comps),'components':[ {k:v for k,v in c.items() if k!='face_indices'} for c in comps]})
 if summ:
  all_side_data[str(side)+'_valid']=summ
valid=[]
for side in (1,-1):
 valid.extend(q for q in all_side_data.get(str(side)+'_valid',[]) if q is not None)
if valid:
 valid.sort(key=lambda q:q['physical_surface_area_m2'])
chosen=valid[0] if valid else None
report['bone_mesh_closed_edge_incidence']=closed_edges
report['bone_mesh_consistently_oriented']=oriented_edges
report['selected_cap_side_sign']=chosen['side_sign'] if chosen else None
report['selected_cap_face_count']=chosen['selected_face_count'] if chosen else 0
report['selected_cap_physical_area_m2']=chosen['physical_surface_area_m2'] if chosen else None
report['selected_cap_planar_pass']=bool(chosen)
report['status']='PASS_PLANAR_CAP' if chosen else 'FAIL_NO_COHERENT_CLIPPED_BONE_CAP'
(out/'cap-planar-report.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
if chosen:
 # Persist explicit bone-face and source-face ancestry for the selected closed cap.
 tokens=chosen['tokens'];token_index={t:i for i,t in enumerate(tokens)}
 types={'v':0,'c':1,'x':2}
 token_kind=np.asarray([types[t[0]] for t in tokens],np.int8)
 token_a=np.asarray([int(t[1]) for t in tokens],np.int64)
 token_b=np.asarray([int(t[2]) if len(t)>2 else -1 for t in tokens],np.int64)
 face_idx=np.asarray([[token_index[t] for t in tri] for tri,boneid in chosen['bone_faces']],np.int32)
 face_bone=np.asarray([boneid for tri,boneid in chosen['bone_faces']],np.int64)
 face_bary=np.asarray([[chosen['bary_by_token'][t][0] for t in tri] for tri,boneid in chosen['bone_faces']],np.float64)
 face_source_residual=np.asarray([[chosen['residual_by_token'][t] for t in tri] for tri,boneid in chosen['bone_faces']],np.float64)
 source_pos=np.asarray([point_for(t) for t in tokens],np.float64)
 np.savez_compressed(out/'clipped-bone-cap-planar.npz',token_kind=token_kind,token_a=token_a,token_b=token_b,source_local_m=source_pos,source_face_barycentric=np.asarray([chosen['bary_by_token'][t][0] for t in tokens]),cap_face_token_indices=face_idx,cap_face_bone_triangle_stable_row=face_bone,cap_face_source_barycentric=face_bary,cap_face_source_projection_residual_m=face_source_residual)
 print(json.dumps({k:v for k,v in report.items() if k not in ('boundary_side_results','planar_cap_candidates')},indent=2))
else:
 print(json.dumps({k:v for k,v in report.items() if k not in ('boundary_side_results','planar_cap_candidates')},indent=2))
 raise SystemExit(0)

