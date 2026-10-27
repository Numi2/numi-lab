from pathlib import Path
from fractions import Fraction
from collections import defaultdict, Counter, deque
import hashlib, importlib.util, json, sys, time
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
C=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz'
A=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/audit-159/report.json'
TOPO=S/'per-bank-workers-20261027/b115-118/attempt-005/local-topology-report.json'
OUT=S/'per-bank-workers-20261027/b115-118/attempt-005/local-contour-geometry.json'
AUDIT=Path('/Users/numi/Work/numi-human-connected-surface-069/audit_skin_bone_surfaces_159.py')
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-2025-10-05/supported-continuation-2025-10-20/metal-optimization-080/visual-verification-2026-10-22/wrist-connected-repair-001')
# Use authoritative paths resolved from the audit source, to avoid guessing an evidence directory.
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
audit=loadmod('bank115_local_audit',AUDIT); E=audit.E; helper=loadmod('bank115_local_helper',HELPER); exact=audit.exact_module()
base=json.loads(A.read_text()); bank=next(b for b in base['banks'] if (b['body'],b['stable_bone_id'])==(115,118)); pairs=[tuple(map(int,p)) for p in bank['result']['triangle_pairs']]
cross=sorted({p[0] for p in pairs})
with np.load(C,allow_pickle=False) as z:
 P=z['points_world_m'].astype(np.float64); F=z['source_O_faces'].astype(np.int64); REF=z['area_reference_037_points_world_m_f64'].astype(np.float64)
 ancptr=z['face_source_ancestry_indptr']; ancrows=z['face_source_ancestry_face_rows'];
 # The ancestry is required to remain traceable for any future local remesh.
 fparent=z['refinement_face_parent_face_row']; fsubptr=z['face_fine_subface_ancestry_indptr']; fsubrows=z['face_fine_subface_ancestry_rows'];
 # edge and node adjacency
edgefaces=defaultdict(list); nodefaces=defaultdict(list)
for fi,t in enumerate(F):
 for v in t: nodefaces[int(v)].append(fi)
 for u,v in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):edgefaces[tuple(sorted((int(u),int(v))))].append(fi)
crossset=set(cross);patch=set(cross)
for f in cross:
 for v in F[f]:patch.update(nodefaces[int(v)])
patch_rows=np.asarray(sorted(patch),dtype=np.int64)
patch_edges=defaultdict(list)
for f in patch_rows:
 t=F[f]
 for u,v in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):patch_edges[tuple(sorted((int(u),int(v))))].append(int(f))
boundary_edges=[];nonmanifold_edges=[]
for e,inside_faces in patch_edges.items():
 outside=[g for g in edgefaces[e] if g not in patch]
 if outside: boundary_edges.append((e,inside_faces,outside))
 if len(edgefaces[e])!=2:nonmanifold_edges.append((e,len(edgefaces[e])))
boundary_nodes=sorted({v for e,_,_ in boundary_edges for v in e})
# exact local one-bank check only on the connected patch; no all-bank or global-self audit.
bankdir=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
T,ids,gids,bodies,M,off,pos,keys,pins,unused=helper.read_bank_inputs(bankdir,E)
mask=(ids[:,0]==115)&(ids[:,1]==118); faceids=ids[mask,3].astype(np.int64); local_bone=T[mask].astype(np.float64)
world_bone=((local_bone+off[115])@M[115]+pos[115])*1000.0
skin_world=P*1000.0; patch_tri=skin_world[F[patch_rows]]; bone_tri=world_bone
# AABB filter exact one-bank pairs.
slo,shi=patch_tri.min(1),patch_tri.max(1); blo,bhi=bone_tri.min(1),bone_tri.max(1)
si=np.flatnonzero(np.all(shi>=blo.min(0),axis=1)&np.all(slo<=bhi.max(0),axis=1))
rlo,rhi=slo[si].min(0),shi[si].max(0); bi=np.flatnonzero(np.all(bone_tri.max(1)>=rlo,axis=1)&np.all(bone_tri.min(1)<=rhi,axis=1))
def records(tris,faceids0):
 points=[tuple(Fraction.from_float(float(v)) for v in p) for p in np.asarray(tris).reshape(-1,3)]
 faces=np.arange(len(points),dtype=np.int64).reshape(-1,3)
 return [(tri,lo,hi,int(faceids0[i]),v_ids) for i,(tri,lo,hi,_r,v_ids) in enumerate(exact._records(points,faces))]
# Direct pair exact test over local AABB candidates.
sr=records(patch_tri[si],patch_rows[si]); br=records(bone_tri[bi],faceids[bi]);res=exact._audit_pair(sr,br,same_surface=False)
local_pairs=[(int(a),int(b)) for a,b in res['triangle_pairs']]
# The crossing contour segment is reconstructed from each exact witness pair using floating geometry only.
def slice_edges_by_plane(tri,n,d,other):
 pts=[]; scale=max(1.0,float(np.max(np.abs(tri))),float(np.max(np.abs(other))))
 eps=1e-7*scale
 signed=tri@n+d
 for i in range(3):
  j=(i+1)%3; u,v=signed[i],signed[j]
  if abs(u)<=eps:
   p=tri[i]
   if np.linalg.norm(np.cross(other[1]-other[0],p-other[0])) <= eps*max(1.0,np.linalg.norm(other[1]-other[0])):
    pts.append(p)
  if u*v<-(eps*eps):
   t=u/(u-v); pts.append(tri[i]+t*(tri[j]-tri[i]))
 uniq=[]
 for p in pts:
  if not any(np.linalg.norm(p-q)<1e-5 for q in uniq):uniq.append(p)
 return uniq
def in_tri(p,tol,tri):
 a,b,c=tri; v0=b-a;v1=c-a;v2=p-a
 d00=v0@v0;d01=v0@v1;d11=v1@v1;d20=v2@v0;d21=v2@v1;den=d00*d11-d01*d01
 if abs(den)<1e-20:return False
 v=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den;u=1-v-w
 return min(u,v,w)>=-tol
segments=[];degenerate=[]
for sf,bf in pairs:
 ta=skin_world[F[sf]]; tb=world_bone[np.flatnonzero(faceids==bf)[0]]
 na=np.cross(ta[1]-ta[0],ta[2]-ta[0]); nb=np.cross(tb[1]-tb[0],tb[2]-tb[0]); nn=np.linalg.norm(na);mn=np.linalg.norm(nb)
 if nn==0 or mn==0:degenerate.append([sf,bf,'zero_normal']);continue
 na/=nn;nb/=mn; da=-na@ta[0];db=-nb@tb[0]
 Apts=slice_edges_by_plane(ta,nb,db,tb);Bpts=slice_edges_by_plane(tb,na,da,ta)
 valid=[p for p in Apts if in_tri(p,1e-5,tb)]+[p for p in Bpts if in_tri(p,1e-5,ta)]
 uv=[]
 for p in valid:
  if not any(np.linalg.norm(p-q)<1e-4 for q in uv):uv.append(p)
 if len(uv)>=2:
  best=max(((np.linalg.norm(uv[i]-uv[j]),i,j) for i in range(len(uv)) for j in range(i+1,len(uv))),default=(0,0,0))
  if best[0]>1e-5:segments.append({'skin_face':sf,'bone_face':bf,'a_mm':uv[best[1]].tolist(),'b_mm':uv[best[2]].tolist(),'length_mm':float(best[0])})
  else:degenerate.append([sf,bf,'point_contact'])
 else:degenerate.append([sf,bf,'no_segment_reconstructed',len(uv)])
# Quantized segment endpoint graph: only used to inspect contour connectivity; not an exact topological proof.
def key(p):return tuple(np.rint(np.asarray(p)*1000).astype(np.int64)) # micron bins in mm units? values mm, *1000 => micron
adj=defaultdict(list)
for i,s in enumerate(segments):adj[key(s['a_mm'])].append(i);adj[key(s['b_mm'])].append(i)
verts=sorted(adj);unseen=set(verts);contours=[]
while unseen:
 st=min(unseen);unseen.remove(st);q=[st];comp=[]
 while q:
  v=q.pop();comp.append(v)
  for eidx in adj[v]:
   s=segments[eidx]
   w=key(s['b_mm']) if key(s['a_mm'])==v else key(s['a_mm'])
   if w in unseen:unseen.remove(w);q.append(w)
 contours.append({'endpoint_count':len(comp),'degree_histogram':dict(Counter(len(adj[v]) for v in comp)),'endpoints':len([v for v in comp if len(adj[v])==1]),'branches':len([v for v in comp if len(adj[v])>2])})
# Outer boundary node-to-bone distances in pinned bank-local coordinates.
skin_local=((P-pos[115])@M[115].T-off[115])*1000.0
bone_local=local_bone*1000.0
# Triangle point distances from point projection and three segment distances.
def point_tri_distance(p,tri):
 a,b,c=tri;ab=b-a;ac=c-a;n=np.cross(ab,ac);nn=n@n
 if nn<1e-24:return min(point_seg_dist(p,a,b),point_seg_dist(p,b,c),point_seg_dist(p,c,a))
 q=p-(n@ (p-a))/nn*n
 v0=ab;v1=ac;v2=q-a;d00=v0@v0;d01=v0@v1;d11=v1@v1;d20=v2@v0;d21=v2@v1;den=d00*d11-d01*d01
 v=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den;u=1-v-w
 if min(u,v,w)>=-1e-12:return float(np.linalg.norm(p-q))
 return min(point_seg_dist(p,a,b),point_seg_dist(p,b,c),point_seg_dist(p,c,a))
def point_seg_dist(p,a,b):
 v=b-a;d=v@v
 t=0 if d==0 else min(1,max(0,float((p-a)@v/d)))
 return float(np.linalg.norm(p-(a+t*v)))
dists=[]
for v in boundary_nodes:
 ds=[point_tri_distance(skin_local[v],tri) for tri in bone_local]
 dists.append((int(v),float(min(ds))))
# current shape ratio on the 57-face patch relative to fixed 037 reference.
cur=skin_world[F[patch_rows]];ref=REF[F[patch_rows]]*1000
ca=np.cross(cur[:,1]-cur[:,0],cur[:,2]-cur[:,0]);ra=np.cross(ref[:,1]-ref[:,0],ref[:,2]-ref[:,0]);den=np.einsum('ij,ij->i',ra,ra);ratio=np.einsum('ij,ij->i',ca,ra)/np.maximum(den,1e-30)
report={'schema':'numi.human.bank115_118.local_contour_geometry.v1','candidate_path':str(C),'candidate_sha256':sha(C),'audit159_sha256':sha(A),'bank_export_path':str(bankdir),'exact_predicate_sha256':sha(audit.EXACT),'local_patch':{'crossing_skin_face_count':len(cross),'one_vertex_ring_face_count':len(patch_rows),'one_ring_non_crossing_face_count':len(patch_rows)-len(cross),'crossing_exact_pairs_full_bank':len(pairs),'crossing_exact_pairs_patch_local':len(local_pairs),'patch_boundary_edge_count':len(boundary_edges),'patch_boundary_node_count':len(boundary_nodes),'nonmanifold_patch_edges':[list(x) for x in nonmanifold_edges],'fixed_outer_boundary_nodes':boundary_nodes,'boundary_node_min_distance_to_bone_mm':dists,'minimum_boundary_node_distance_mm':min((d for _,d in dists),default=None),'maximum_boundary_node_distance_mm':max((d for _,d in dists),default=None),'boundary_edge_adjacent_faces_crossing':sorted({g for _,_,out in boundary_edges for g in out}&crossset)},'crossing_contour_approx':{'pair_segment_count':len(segments),'degenerate_pair_reconstructions':degenerate,'micron_quantized_endpoint_components':contours,'endpoint_degree_histogram':dict(Counter(len(adj[v]) for v in adj)),'segments':segments},'ancestry':{'crossing_source_ancestry_rows':sorted({int(v) for f in cross for v in ancrows[ancptr[f]:ancptr[f+1]]}),'crossing_immediate_parent_face_rows':sorted({int(fparent[f]) for f in cross}),'crossing_fine_subface_ancestry_rows':sorted({int(v) for f in cross for v in fsubrows[fsubptr[f]:fsubptr[f+1]]})},'shape_gate':{'patch_faces_checked':len(patch_rows),'minimum_current_oriented_ratio_vs_fixed_037':float(np.min(ratio)),'negative_or_zero_faces':int(np.sum(ratio<=0)),'all_positive':bool(np.all(ratio>0))},'limitations':['Approximate segment reconstruction and boundary distance are diagnostic; exact pair reports remain authoritative. No topology edits, global bank audit, whole-body self audit, or physical admission.']}
OUT.write_text(json.dumps(report,indent=2)+"\n")
summary={k:v for k,v in report.items() if k not in ('crossing_contour_approx','ancestry')}
summary['crossing_contour_approx']={k:v for k,v in report['crossing_contour_approx'].items() if k!='segments'}
summary['ancestry']=report['ancestry']; print(json.dumps(summary,indent=2))
