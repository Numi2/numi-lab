#!/usr/bin/env python3
"""One face-wise CDT prototype for the 115/118 exact contour; patch only."""
from pathlib import Path
from fractions import Fraction
from collections import defaultdict, Counter
import hashlib, importlib.util, json, sys, time
import numpy as np
from scipy.spatial import Delaunay
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
C=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz'
A=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/audit-159/report.json'
G=S/'per-bank-workers-20261027/b115-118/attempt-005/local-contour-geometry.json'
O=S/'per-bank-workers-20261027/b115-118/attempt-005/conforming-patch-003'
AUDIT=Path('/Users/numi/Work/numi-human-connected-surface-069/audit_skin_bone_surfaces_159.py')
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def bary(p,tri):
 a,b,c=tri;v0=b-a;v1=c-a;v2=p-a
 d00=v0@v0;d01=v0@v1;d11=v1@v1;d20=v2@v0;d21=v2@v1;den=d00*d11-d01*d01
 if abs(den)<1e-20:raise ValueError('degenerate parent triangle')
 v=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den
 return np.asarray([1-v-w,v,w],dtype=np.float64)
def orient2(a,b,c):
 x=p2[b]-p2[a];y=p2[c]-p2[a];return float(x[0]*y[1]-x[1]*y[0])
def segcross(e,f):
 a,b=e;c,d=f
 if len({a,b,c,d})<4:return False
 o1=orient2(a,b,c);o2=orient2(a,b,d);o3=orient2(c,d,a);o4=orient2(c,d,b)
 return o1*o2 < -1e-28 and o3*o4 < -1e-28
def make_tri(a,b,c):
 o=orient2(a,b,c)
 if abs(o)<1e-16:raise ValueError(f'degenerate 2D triangle {a,b,c,o}')
 return (a,b,c) if o>0 else (a,c,b)
def edge_map(ts):
 out=defaultdict(list)
 for i,t in enumerate(ts):
  for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):out[tuple(sorted((a,b)))].append(i)
 return out
def constrained_triangulate(p2, constraints, max_flips=5000):
 if len(p2)<3:raise ValueError('less than 3 vertices')
 T=[make_tri(*map(int,t)) for t in Delaunay(p2).simplices]
 if len({v for t in T for v in t})!=len(p2):raise ValueError('Delaunay omitted one or more points')
 protected=set();flips=[]
 for target0 in constraints:
  target=tuple(sorted(map(int,target0)))
  if target[0]==target[1]:raise ValueError('collapsed constraint')
  for _ in range(max_flips):
   edges=edge_map(T)
   if target in edges:protected.add(target);break
   crossing=[e for e in edges if segcross(e,target)]
   switched=False
   for e in sorted(crossing):
    if e in protected or len(edges[e])!=2:continue
    i,j=edges[e];c=next(v for v in T[i] if v not in e);d=next(v for v in T[j] if v not in e)
    new=tuple(sorted((c,d)))
    if new in edges or not segcross(e,new) or segcross(new,target) or any(segcross(new,k) for k in protected):continue
    aa,bb=e
    try:t1=make_tri(c,d,aa);t2=make_tri(d,c,bb)
    except ValueError:continue
    if abs((orient2(*T[i])+orient2(*T[j]))-(orient2(*t1)+orient2(*t2)))>1e-10:continue
    T[i]=t1;T[j]=t2;flips.append({'target':target,'old_edge':e,'new_edge':new});switched=True;break
   if not switched:raise ValueError(f'no convex flip can insert constraint {target}; crossing={crossing[:12]}')
  else:raise ValueError(f'flip cap exceeded for constraint {target}')
 edges=edge_map(T)
 missing=[e for e in constraints if tuple(sorted(map(int,e))) not in edges]
 if missing:raise ValueError(f'missing constraints {missing[:12]}')
 # Ensure the straight-line graph is embedded and has no un-noded edge crossings.
 ee=list(edges)
 for i,e in enumerate(ee):
  for f in ee[i+1:]:
   if segcross(e,f):raise ValueError(f'nonadjacent triangulation edges cross: {e}, {f}')
 return T,flips

def slice_hits(tri,n,d,other):
 vals=tri@n+d;pts=[];eps=1e-8
 for i in range(3):
  j=(i+1)%3;u,v=vals[i],vals[j]
  if abs(u)<=eps:pts.append(tri[i])
  if u*v<0:
   t=u/(u-v);pts.append(tri[i]+t*(tri[j]-tri[i]))
 uniq=[]
 for p in pts:
  if not any(np.linalg.norm(p-q)<1e-5 for q in uniq):uniq.append(p)
 return uniq
def inside_tri(p,t):
 b=bary(p,t);return bool(np.min(b)>=-1e-7 and np.max(b)<=1+1e-7)
def tri_intersections(a,b):
 na=np.cross(a[1]-a[0],a[2]-a[0]);nb=np.cross(b[1]-b[0],b[2]-b[0]);la=np.linalg.norm(na);lb=np.linalg.norm(nb)
 if la==0 or lb==0:return []
 na/=la;nb/=lb;da=-na@a[0];db=-nb@b[0]
 pts=[p for p in slice_hits(a,nb,db,b) if inside_tri(p,b)]+[p for p in slice_hits(b,na,da,a) if inside_tri(p,a)]
 uniq=[]
 for p in pts:
  if not any(np.linalg.norm(p-q)<1e-4 for q in uniq):uniq.append(p)
 return uniq
def point_seg_dist(p,a,b):
 v=b-a;d=v@v;t=0 if d==0 else min(1,max(0,float((p-a)@v/d)))
 return float(np.linalg.norm(p-(a+t*v)))
def on_segment(p,a,b,tol):return point_seg_dist(p,a,b)<=tol
started=time.monotonic();O.mkdir(parents=True,exist_ok=False)
report={'schema':'numi.human.bank115_118.conforming_patch_trial.v1','status':'STARTED','candidate_path':str(C),'candidate_sha256':sha(C),'audit159_sha256':sha(A),'contour_report_sha256':sha(G),'patch_candidate_path':str(O/'patch-candidate.npz'),'geometry_or_native_admission':False,'limitations':['Patch-only topology prototype, not integrated into full mesh. No all-bank, whole-body self, physical-transfer, or native admission check.']}
try:
 audit=loadmod('bank115_patch_audit',AUDIT); helper=loadmod('bank115_patch_helper',HELPER); exact=audit.exact_module()
 with np.load(C,allow_pickle=False) as z:
  Z={k:z[k].copy() for k in z.files}
 P=Z['points_world_m'].astype(np.float64);F=Z['source_O_faces'].astype(np.int64);Pmm=P*1000.0
 base=json.loads(A.read_text());bank=next(b for b in base['banks'] if (b['body'],b['stable_bone_id'])==(115,118));pairs=[tuple(map(int,p)) for p in bank['result']['triangle_pairs']];cross=sorted({sf for sf,bf in pairs})
 # Whole connected one-vertex-ring patch around exact crossing skin faces.
 edgefaces=defaultdict(list);nodefaces=defaultdict(list)
 for fi,t in enumerate(F):
  for v in t:nodefaces[int(v)].append(fi)
  for u,v in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):edgefaces[tuple(sorted((int(u),int(v))))].append(fi)
 patch=set(cross)
 for fi in cross:
  for v in F[fi]:patch.update(nodefaces[int(v)])
 parent_faces=sorted(patch);parent_set=set(parent_faces)
 # Patch boundary must be a simple closed cycle and all exact crossing faces must be interior to it.
 patch_edge_faces=defaultdict(list)
 for fi in parent_faces:
  t=F[fi]
  for u,v in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):patch_edge_faces[tuple(sorted((int(u),int(v))))].append(fi)
 boundary_edges=[]
 for e,inside in patch_edge_faces.items():
  if any(g not in parent_set for g in edgefaces[e]):boundary_edges.append(e)
  elif len(edgefaces[e])==1:boundary_edges.append(e)
 bgraph=defaultdict(list)
 for u,v in boundary_edges:bgraph[u].append(v);bgraph[v].append(u)
 boundary_components=[];unseen=set(bgraph)
 while unseen:
  st=min(unseen);unseen.remove(st);q=[st];comp=[]
  while q:
   v=q.pop();comp.append(v)
   for w in bgraph[v]:
    if w in unseen:unseen.remove(w);q.append(w)
  boundary_components.append(comp)
 boundary_ok=(len(boundary_components)==1 and all(len(bgraph[v])==2 for v in bgraph))
 patch_nodes=sorted({int(v) for fi in parent_faces for v in F[fi]})
 # Original patch points keep their exact candidate rows; appended contour nodes remain float64.
 points={('v',v):Pmm[v].copy() for v in patch_nodes}
 point_bary_by_face=defaultdict(dict)
 for fi in parent_faces:
  for j,v in enumerate(F[fi]):
   bc=np.zeros(3);bc[j]=1;point_bary_by_face[fi][('v',int(v))]=bc
 # Load pinned bone and use the report's exact crossing pair IDs.
 bankdir=audit.E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
 T,ids,gids,bodies,M,off,pos,keys,pins,unused=helper.read_bank_inputs(bankdir,audit.E)
 bmask=(ids[:,0]==115)&(ids[:,1]==118);bone_ids=ids[bmask,3].astype(np.int64);bone_local=T[bmask].astype(np.float64)
 bone_world=((bone_local+off[115])@M[115]+pos[115])*1000.0
 bone_index={int(fid):i for i,fid in enumerate(bone_ids)}
 report['bank_115_118']={'face_count':int(len(bone_ids)),'exact_cross_pair_count':len(pairs),'crossing_skin_face_count':len(cross),'crossing_skin_face_rows':cross,'crossing_bone_face_rows':sorted({bf for sf,bf in pairs})}
 # Insert points globally; micron bins join independently reconstructed endpoints along shared skin edges.
 point_cluster={};point_parent={};point_error_max=0.0
 def key_for(p):
  p=np.asarray(p,dtype=np.float64);d=np.linalg.norm(Pmm-p,axis=1);nearest=int(np.argmin(d))
  if d[nearest]<1e-6:return ('v',nearest)
  k=tuple(np.rint(p*100000.0).astype(np.int64))
  if k not in point_cluster:point_cluster[k]=p.copy()
  else:
   discrepancy=float(np.linalg.norm(point_cluster[k]-p));report['max_endpoint_cluster_spread_mm']=max(report.get('max_endpoint_cluster_spread_mm',0.0),discrepancy)
   if discrepancy>0.00002:raise ValueError(f'contour endpoints do not weld within 2 microns: {discrepancy} mm')
  return ('p',k)
 def add_in_face(fi,p,key=None):
  p=np.asarray(p,dtype=np.float64);tri=Pmm[F[fi]];bc=bary(p,tri);proj=bc@tri;err=float(np.linalg.norm(proj-p));
  if err>0.005:raise ValueError(f'contour point is {err} mm off skin face {fi}')
  if float(np.min(bc)) < -2e-4 or float(np.max(bc))>1.0002:raise ValueError(f'contour point outside skin face {fi}: {bc.tolist()}')
  # endpoint projection error is checked against the local tolerance above
  if key is None:key=key_for(p)
  if key[0]=='p': points.setdefault(key,point_cluster[key[1]].copy())
  else: points.setdefault(key,Pmm[int(key[1])].copy())
  if key in point_bary_by_face[fi]:
   if np.linalg.norm(point_bary_by_face[fi][key]-bc)>0.002:raise ValueError('inconsistent repeated barycentric endpoint')
  point_bary_by_face[fi][key]=bc
  # If point lies on a skin edge, propagate the same node to every incident face in this patch.
  z=np.flatnonzero(np.abs(bc)<2e-5)
  if len(z)==1:
   keep=[j for j in range(3) if j!=int(z[0])];e=tuple(sorted((int(F[fi,keep[0]]),int(F[fi,keep[1]]))))
   for nb in edgefaces[e]:
    if nb in parent_set and nb!=fi:
     nbc=bary(points[key],Pmm[F[nb]]);err2=float(np.linalg.norm(nbc@Pmm[F[nb]]-points[key]))
     if err2>0.005:raise ValueError(f'edge endpoint not conforming in adjacent face {nb}: {err2} mm')
     point_bary_by_face[nb][key]=nbc
  return key,bc
 # Contour segments from exact pair intersections; add the isolated point-contact witness too.
 geom=json.loads(G.read_text());segments_by_face=defaultdict(list);contour_segments=[];contact_points=[]
 for s in geom['crossing_contour_approx']['segments']:
  fi=int(s['skin_face']);pa=np.asarray(s['a_mm'],float);pb=np.asarray(s['b_mm'],float)
  ka,_=add_in_face(fi,pa);kb,_=add_in_face(fi,pb)
  if ka==kb:raise ValueError(f'contour segment collapsed after endpoint welding: face {fi}')
  segments_by_face[fi].append((ka,kb));contour_segments.append((pa,pb,fi,int(s['bone_face'])))
 # Reconstruct the known single point contact pair and make its point a triangle vertex.
 pcontact_pair=(31456,286);wa=F[pcontact_pair[0]];ta=Pmm[wa];tb=bone_world[bone_index[pcontact_pair[1]]]
 nA=np.cross(ta[1]-ta[0],ta[2]-ta[0]);nB=np.cross(tb[1]-tb[0],tb[2]-tb[0]);nA/=np.linalg.norm(nA);nB/=np.linalg.norm(nB)
 hits=[p for p in slice_hits(ta,nB,-nB@tb[0],tb) if inside_tri(p,tb)]+[p for p in slice_hits(tb,nA,-nA@ta[0],ta) if inside_tri(p,ta)]
 uniq=[]
 for p in hits:
  if not any(np.linalg.norm(p-q)<1e-4 for q in uniq):uniq.append(p)
 if len(uniq)!=1:raise ValueError(f'expected one isolated point for {pcontact_pair}, found {len(uniq)}')
 kc,_=add_in_face(pcontact_pair[0],uniq[0]);contact_points.append({'skin_face':pcontact_pair[0],'bone_face':pcontact_pair[1],'point_key':str(kc),'point_world_mm':uniq[0].tolist()})
 # Build every current patch face independently in its own barycentric plane, sharing all boundary nodes.
 all_child_keys=[];all_child_parent=[];all_child_bary=[];all_contour_key_edges=[];face_reports=[];flips_total=0
 original_boundary=set(boundary_edges)
 for fi in parent_faces:
  # points are ordered deterministically, original vertices precede inserted nodes.
  keys_face=list(point_bary_by_face[fi])
  keys_face.sort(key=lambda k:(0,int(k[1])) if k[0]=='v' else (1,k[1]))
  idx={k:i for i,k in enumerate(keys_face)}
  local_bary=np.asarray([point_bary_by_face[fi][k] for k in keys_face])
  global_p2=local_bary[:,1:]
  # Constraints: all contour segments in this parent face and the split parent-triangle boundary.
  constraints=[];contour_local=[]
  for ka,kb in segments_by_face.get(fi,[]):
   e=tuple(sorted((idx[ka],idx[kb])));constraints.append(e);contour_local.append(e)
  # Collect nodes lying on each original edge, sorted by edge parameter.
  boundary_local=[]
  tri_nodes=list(map(int,F[fi]))
  for j,k in ((0,1),(1,2),(2,0)):
   v0=tri_nodes[j];v1=tri_nodes[k];seq=[]
   for qk,bc in zip(keys_face,local_bary):
    if abs(float(bc[(j+2)%3]))<2e-5: # wrong for arbitrary edge; replaced by endpoint/line projection below
     pass
    pp=points[qk]
    t=Pmm[v1]-Pmm[v0];den=float(t@t)
    u=0.0 if den==0 else float((pp-Pmm[v0])@t/den)
    dist=point_seg_dist(pp,Pmm[v0],Pmm[v1])
    if -2e-5<=u<=1+2e-5 and dist<=0.005:seq.append((u,idx[qk]))
   seq=sorted({ii:(u,ii) for u,ii in seq}.values())
   if not seq or seq[0][1]!=idx[('v',v0)] or seq[-1][1]!=idx[('v',v1)]:
    # Edge orientation can be opposite to canonical face-edge order; sort and validate endpoints as a set.
    seq=sorted(seq)
   if len(seq)<2 or {seq[0][1],seq[-1][1]}!={idx[('v',v0)],idx[('v',v1)]}:
    raise ValueError(f'parent face {fi} boundary edge lost an endpoint: {(v0,v1,seq)}')
   for (_,a),(_,b) in zip(seq,seq[1:]):boundary_local.append(tuple(sorted((a,b))))
  constraints=list(dict.fromkeys(constraints+boundary_local))
  # Reject intersecting constraint edges unless they share their exact contour endpoint.
  p2=global_p2
  for i,e in enumerate(constraints):
   for f in constraints[i+1:]:
    if segcross(e,f):raise ValueError(f'crossing constraints inside parent face {fi}: {e} vs {f}')
  if len(keys_face)==3 and not contour_local:
   tris=[make_tri(*(idx[('v',int(v))] for v in F[fi]))];flips=[]
  else:tris,flips=constrained_triangulate(p2,constraints)
  flips_total+=len(flips)
  edge_counts=edge_map(tris)
  child_area=sum(orient2(*t)*0.5 for t in tris)
  if abs(child_area-0.5)>1e-9:raise ValueError(f'parent face {fi} coverage mismatch {child_area}')
  if any(len(v) not in (1,2) for v in edge_counts.values()):raise ValueError(f'parent face {fi} nonmanifold local edge incidence')
  for e in contour_local:
   if len(edge_counts.get(tuple(sorted(e)),[]))!=2:raise ValueError(f'contour edge not interior to parent face {fi}: {e}')
  # Convert children into patch-global nodes with per-face parent barycentric coordinates.
  for t in tris:
   kk=[keys_face[i] for i in t]
   all_child_keys.append(kk);all_child_parent.append(fi);all_child_bary.append(np.asarray([local_bary[i] for i in t]))
  for e in contour_local:all_contour_key_edges.append((keys_face[e[0]],keys_face[e[1]]))
  face_reports.append({'parent_face_row':fi,'point_count':len(keys_face),'child_triangle_count':len(tris),'constraint_count':len(constraints),'contour_constraint_count':len(contour_local),'flips':len(flips),'planar_area':child_area,'unused_points':sorted(set(range(len(keys_face)))-{v for t in tris for v in t})})
 # Construct deterministic patch-global node table, reusing every original candidate row.
 new_keys=sorted((k for k in points if k[0]=='p'),key=lambda k:k[1]);node_keys=[('v',v) for v in patch_nodes]+new_keys;node_index={k:i for i,k in enumerate(node_keys)}
 out_points=np.asarray([points[k]*1e-3 for k in node_keys],dtype=np.float64)
 out_source=np.asarray([int(k[1]) if k[0]=='v' else -1 for k in node_keys],dtype=np.int64)
 out_kind=np.asarray([0 if k[0]=='v' else 1 for k in node_keys],dtype=np.int8)
 child_faces=np.asarray([[node_index[k] for k in tri] for tri in all_child_keys],dtype=np.int64)
 child_parent=np.asarray(all_child_parent,dtype=np.int64);child_bary=np.asarray(all_child_bary,dtype=np.float64)
 # Parent/source ancestry copied per emitted child triangle; barycentric parent map retained explicitly.
 def inherited(prefix,ptr_name,rows_name):
  ptr=Z[ptr_name];vals=Z[rows_name];outptr=[0];out=[]
  for fi in child_parent:
   out.extend(map(int,vals[ptr[int(fi)]:ptr[int(fi)+1]]));outptr.append(len(out))
  return np.asarray(outptr,np.int64),np.asarray(out,dtype=vals.dtype)
 sptr,srows=inherited('source','face_source_ancestry_indptr','face_source_ancestry_face_rows')
 subptr,subrows=inherited('sub','face_fine_subface_ancestry_indptr','face_fine_subface_ancestry_rows')
 detptr,detrows=inherited('detail','face_fine_detail_ancestry_indptr','face_fine_detail_ancestry_rows')
 # Global local patch edge incidence must preserve precisely the original fixed outer boundary.
 ecounts=defaultdict(int)
 for t in child_faces:
  for u,v in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):ecounts[tuple(sorted((int(u),int(v))))]+=1
 inv_boundary={tuple(sorted((node_index[('v',u)],node_index[('v',v)]))) for u,v in boundary_edges}
 emitted_boundary={e for e,n in ecounts.items() if n==1}
 nonmanifold=[(e,n) for e,n in ecounts.items() if n not in (1,2)]
 if emitted_boundary!=inv_boundary:raise ValueError(f'patch outer boundary changed: missing={len(inv_boundary-emitted_boundary)}, added={len(emitted_boundary-inv_boundary)}')
 if nonmanifold:raise ValueError(f'patch has nonmanifold output edges: {nonmanifold[:8]}')
 # No T-junctions among patch output nodes and edges.
 t_junctions=[]
 pmm=out_points*1000.0
 for e in ecounts:
  a,b=pmm[list(e)];v=b-a;den=float(v@v)
  for j,p in enumerate(pmm):
   if j in e:continue
   u=0.0 if den==0 else float((p-a)@v/den)
   if 1e-8<u<1-1e-8 and point_seg_dist(p,a,b)<1e-5:t_junctions.append((e,j,u))
 if t_junctions:raise ValueError(f'patch contains T junctions: {t_junctions[:12]}')
 # Local shape checks: positive orientation and no tiny sliver below 1e-6 parent-area fraction.
 parent_tri=Pmm[F[child_parent]];child_tri=pmm[child_faces];pn=np.cross(parent_tri[:,1]-parent_tri[:,0],parent_tri[:,2]-parent_tri[:,0]);cn=np.cross(child_tri[:,1]-child_tri[:,0],child_tri[:,2]-child_tri[:,0]);parea=np.linalg.norm(pn,axis=1)*0.5;carea=np.linalg.norm(cn,axis=1)*0.5;orient=np.einsum('ij,ij->i',pn,cn)/np.maximum(np.einsum('ij,ij->i',pn,pn),1e-30);rel=carea/np.maximum(parea,1e-30)
 minangle=[];aspect=[]
 for t in child_tri:
  lens=[np.linalg.norm(t[(i+1)%3]-t[i]) for i in range(3)];ar=np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0]))/2
  aspect.append(max(lens)**2/(4*np.sqrt(3)*max(ar,1e-30)))
  ang=[]
  for i in range(3):
   u=t[(i+1)%3]-t[i];v=t[(i+2)%3]-t[i];ang.append(np.degrees(np.arccos(np.clip((u@v)/(max(np.linalg.norm(u)*np.linalg.norm(v),1e-30)),-1,1))))
  minangle.append(min(ang))
 shape_pass=bool(np.all(orient>0) and float(rel.min())>=1e-6 and float(max(aspect))<1e5 and float(min(minangle))>0.001)
 if not shape_pass:raise ValueError(f'local shape gate failed: min_relative_area={rel.min()} max_aspect={max(aspect)} min_angle={min(minangle)}')
 # Local exact check only: patch triangles vs the 384 triangles in bone 118.
 tris_world_mm=pmm[child_faces];lo,hi=tris_world_mm.min(1),tris_world_mm.max(1);blo,bhi=bone_world.min(1),bone_world.max(1)
 si=np.flatnonzero(np.all(hi>=blo.min(0),axis=1)&np.all(lo<=bhi.max(0),axis=1))
 rlo,rhi=lo[si].min(0),hi[si].max(0);bi=np.flatnonzero(np.all(bone_world.max(1)>=rlo,axis=1)&np.all(bone_world.min(1)<=rhi,axis=1))
 def exact_records(triangles,face_ids):
  pts=[tuple(Fraction.from_float(float(v)) for v in p) for p in np.asarray(triangles).reshape(-1,3)];ff=np.arange(len(pts),dtype=np.int64).reshape(-1,3)
  return [(tr,l,h,int(face_ids[i]),ids0) for i,(tr,l,h,_rid,ids0) in enumerate(exact._records(pts,ff))]
 cr=exact._audit_pair(exact_records(tris_world_mm[si],si),exact_records(bone_world[bi],bone_ids[bi]),same_surface=False)
 exact_pairs=[tuple(map(int,p)) for p in cr['triangle_pairs']]
 # Any exact local skin/bone intersection must lie on a child mesh edge and on the reconstructed contour graph.
 conform_fail=[];contact_records=[]
 for cf,bf in exact_pairs:
  st=tris_world_mm[cf];bt=bone_world[bone_index[bf]];hits=tri_intersections(st,bt)
  if not hits:conform_fail.append({'child_face':cf,'bone_face':bf,'reason':'exact pair but no floating reconstruction'});continue
  if len(hits)==1:
   p=hits[0];on_edge=any(point_seg_dist(p,st[i],st[(i+1)%3])<0.005 for i in range(3))
   on_contour=any(point_seg_dist(p,a,b)<0.005 for a,b,_,_ in contour_segments) or any(np.linalg.norm(p-q)<0.005 for q in [x['point_world_mm'] for x in contact_points])
   if not on_edge or not on_contour:conform_fail.append({'child_face':cf,'bone_face':bf,'reason':'point intersection lies in child interior or off contour'})
   else:contact_records.append({'child_face':cf,'bone_face':bf,'kind':'vertex_or_edge_point'})
  else:
   far=max(((np.linalg.norm(hits[i]-hits[j]),i,j) for i in range(len(hits)) for j in range(i+1,len(hits))),default=(0,0,0));p,q=hits[far[1]],hits[far[2]]
   matching=[]
   for i in range(3):
    a,b=st[i],st[(i+1)%3]
    if point_seg_dist(p,a,b)<0.005 and point_seg_dist(q,a,b)<0.005:matching.append(i)
   on_contour=any(point_seg_dist(p,a,b)<0.005 and point_seg_dist(q,a,b)<0.005 for a,b,_,_ in contour_segments)
   if not matching or not on_contour:conform_fail.append({'child_face':cf,'bone_face':bf,'reason':'intersection segment crosses child interior or misses constrained contour','intersection_mm':[p.tolist(),q.tolist()]})
   else:contact_records.append({'child_face':cf,'bone_face':bf,'kind':'conforming_edge_contact','child_edge_index':matching[0]})
 if conform_fail:raise ValueError(f'local exact conformity failed in {len(conform_fail)} pairs; first={conform_fail[:8]}')
 # Verify all inserted points have a parent candidate-face coordinate for source ancestry.
 # Existing source nodes use -1; inserted contour points retain a canonical parent row and barycentric coordinate.
 point_parent_rows=np.full(len(node_keys),-1,dtype=np.int64);point_parent_bary=np.zeros((len(node_keys),3),dtype=np.float64)
 for fi,dict0 in point_bary_by_face.items():
  for k,bc in dict0.items():
   if k[0]=='p':
    ix=node_index[k]
    if point_parent_rows[ix]<0:point_parent_rows[ix]=int(fi);point_parent_bary[ix]=bc
 if np.any((out_source<0)&(point_parent_rows<0)):raise ValueError('an inserted contour node lost its parent ancestry')
 # Save a patch-only candidate with a fixed-boundary integration map and inherited ancestry.
 outer_local=np.asarray(sorted(inv_boundary),np.int64)
 contour_local_edges=np.asarray(sorted({tuple(sorted((node_index[a],node_index[b]))) for a,b in all_contour_key_edges}),np.int64).reshape(-1,2)
 patch_npz=O/'patch-candidate.npz'
 np.savez_compressed(patch_npz,points_world_m_f64=out_points,patch_faces=child_faces,source_candidate_node_rows=out_source,point_kind=out_kind,point_parent_candidate_face_rows=point_parent_rows,point_parent_candidate_barycentric=point_parent_bary,patch_outer_boundary_edges=outer_local,patch_outer_boundary_node_rows=np.asarray(sorted({v for e in inv_boundary for v in e}),np.int64),contour_edges=contour_local_edges,face_parent_candidate_rows=child_parent,face_parent_barycentric=child_bary,face_source_ancestry_indptr=sptr,face_source_ancestry_face_rows=srows,face_fine_subface_ancestry_indptr=subptr,face_fine_subface_ancestry_rows=subrows,face_fine_detail_ancestry_indptr=detptr,face_fine_detail_ancestry_rows=detrows,original_patch_face_rows=np.asarray(parent_faces,np.int64))
 report.update({'status':'PATCH_TRIANGULATION_AND_LOCAL_CONFORMITY_PASS','patch_candidate_sha256':sha(patch_npz),'patch_counts':{'original_patch_faces':len(parent_faces),'crossing_original_faces':len(cross),'output_patch_faces':int(len(child_faces)),'output_patch_nodes':int(len(out_points)),'original_patch_nodes':len(patch_nodes),'inserted_contour_nodes':len(new_keys),'fixed_boundary_edges':len(boundary_edges),'fixed_boundary_components':len(boundary_components),'fixed_boundary_degree_histogram':dict(Counter(len(v) for v in bgraph.values())),'contour_segments':len(contour_segments),'unique_contour_mesh_edges':len(contour_local_edges),'point_contact_constraints':contact_points,'exact_local_pair_count_after_triangulation':len(exact_pairs),'exact_local_contact_pairs_edge_or_vertex_only':len(contact_records),'exact_local_conformity_failures':len(conform_fail),'delaunay_edge_flips':flips_total},'geometry_checks':{'boundary_unchanged':True,'boundary_edges_preserved':True,'patch_coverage_per_parent_face':True,'all_output_internal_edge_incidence_two':True,'no_t_junctions':True,'all_child_orientations_positive':True,'minimum_child_to_parent_area_ratio':float(rel.min()),'maximum_triangle_aspect_measure':float(max(aspect)),'minimum_triangle_angle_degrees':float(min(minangle)),'local_area_shape_gate_pass':shape_pass,'outer_boundary_nodes_exactly_reused':True},'ancestry_checks':{'every_child_face_has_candidate_parent':True,'every_child_face_inherits_source_fine_and_detail_ancestry':True,'every_inserted_point_has_candidate_parent_barycentric':True},'local_exact_check':{'bank':[115,118],'bank_faces_checked_after_aabb':int(len(bi)),'patch_faces_aabb_active':int(len(si)),'exact_pair_count':len(exact_pairs),'contacts_only_on_contour_edges_or_vertices':True,'exact_pairs':[list(p) for p in exact_pairs]},'per_parent_face':face_reports,'elapsed_seconds':time.monotonic()-started})
except Exception as e:
 report.update({'status':'PATCH_TRIAL_FAILED','failure':repr(e),'elapsed_seconds':time.monotonic()-started})
(O/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('per_parent_face','local_exact_check')},indent=2,allow_nan=False))
