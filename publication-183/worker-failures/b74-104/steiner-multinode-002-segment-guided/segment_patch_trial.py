import sys,json,hashlib,importlib.util
from pathlib import Path
from fractions import Fraction
from collections import Counter,defaultdict
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
S=ROOT/'D/surface-clearance-546/component-sat-selector-638'
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
helper_path=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
exact_path=E/'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py'
sys.path.insert(0,str(E/'incremental-exact-audit-owner-253/src'))
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
ex=module('numilab_human.cardiac_cavity_intersections',exact_path)
h=module('bankhelper',helper_path)
C=ROOT/'D/publication-182-stage-001/combined-653/candidate.npz'
out=S/'per-bank-workers-20261027/b74-104/steiner-multinode-002-segment-guided'
Z=np.load(C,allow_pickle=False);X=np.asarray(Z['points_world_m'],np.float32);F=np.asarray(Z['source_O_faces'],np.int64)
basehash=hashlib.sha256(C.read_bytes()).hexdigest()
graph=json.load(open(out/'segment-graph-report.json'))
if graph['candidate_sha256']!=basehash:raise RuntimeError('segment graph candidate hash mismatch')
W=json.load(open(S/'per-bank-workers-20261027/b74-104/baseline-witnesses.json'))
boundary=json.load(open(S/'per-bank-workers-20261027/b74-104/steiner-multinode-001/boundary-feasibility.json'))
old_rows=list(map(int,W['skin_child_faces']))
loop=[int(x['node']) for x in boundary['fixed_boundary_vertices']]
EVID=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
T,ids,gids,bodies,M,off,pos,keys,pins,_=h.read_bank_inputs(EVID,E)
mask=(ids[:,0]==74)&(ids[:,1]==104);B=T[mask];bid=ids[mask,3];R=M[74];O=off[74];P=pos[74]
to_local=lambda v:(np.asarray(v,dtype=np.float64)-P)@R.T-O
to_world=lambda v:(R.T@(np.asarray(v,dtype=np.float64)+O)+P).astype(np.float32)
base_local=to_local(X)
def records(v,faces,face_ids):
 out=[]
 for fid,face in zip(face_ids,faces):
  tri=tuple(tuple(Fraction.from_float(float(x)) for x in v[int(k)]) for k in face)
  out.append((tri,tuple(min(q[a] for q in tri) for a in range(3)),tuple(max(q[a] for q in tri) for a in range(3)),int(fid),tuple(map(int,face))))
 return out
def closest(p,t):
 a,b,c=t;ab=b-a;ac=c-a;ap=p-a;d1=ab@ap;d2=ac@ap
 if d1<=0 and d2<=0:return a
 bp=p-b;d3=ab@bp;d4=ac@bp
 if d3>=0 and d4<=d3:return b
 vc=d1*d4-d3*d2
 if vc<=0 and d1>=0 and d3<=0:return a+(d1/(d1-d3))*ab
 cp=p-c;d5=ab@cp;d6=ac@cp
 if d6>=0 and d5<=d6:return c
 vb=d5*d2-d1*d6
 if vb<=0 and d2>=0 and d6<=0:return a+(d2/(d2-d6))*ac
 va=d3*d6-d3*d5
 if va<=0 and d4-d3>=0 and d5-d6>=0:return b+((d4-d3)/((d4-d3)+(d5-d6)))*(c-b)
 den=1/(va+vb+vc);v=vb*den;w=vc*den
 return a+ab*v+ac*w
def cross2(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def area2(poly):
 return float(sum(poly[i,0]*poly[(i+1)%len(poly),1]-poly[(i+1)%len(poly),0]*poly[i,1] for i in range(len(poly)))*0.5)
def point_in_poly(p,poly):
 inside=False
 for i in range(len(poly)):
  a=poly[i];b=poly[(i+1)%len(poly)]
  if abs(cross2(a,b,p))<1e-12 and min(a[0],b[0])-1e-12<=p[0]<=max(a[0],b[0])+1e-12 and min(a[1],b[1])-1e-12<=p[1]<=max(a[1],b[1])+1e-12:return True
  if ((a[1]>p[1])!=(b[1]>p[1])) and p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0]:inside=not inside
 return inside
def proper_intersects(a,b,c,d):
 o1=cross2(a,b,c);o2=cross2(a,b,d);o3=cross2(c,d,a);o4=cross2(c,d,b)
 return (o1*o2<-1e-20) and (o3*o4<-1e-20)
def poly_centroid(poly):
 cross=[]
 for i in range(len(poly)):cross.append(poly[i,0]*poly[(i+1)%len(poly),1]-poly[(i+1)%len(poly),0]*poly[i,1])
 a=sum(cross)*0.5
 if abs(a)<1e-16:return np.mean(poly,axis=0)
 c=np.zeros(2)
 for i in range(len(poly)):c+=(poly[i]+poly[(i+1)%len(poly)])*cross[i]
 return c/(6*a)
def is_star(poly,center):
 ang=np.arctan2(poly[:,1]-center[1],poly[:,0]-center[0]);ang=np.unwrap(ang)
 dif=np.diff(np.r_[ang,ang[0]+2*np.pi])
 return bool(np.min(dif)>1e-8),dif.tolist()
def ear_clip(poly):
 n=len(poly);remaining=list(range(n));tri=[]
 guard=0
 while len(remaining)>3 and guard<n*n:
  guard+=1;clipped=False
  for q in range(len(remaining)):
   ia=remaining[(q-1)%len(remaining)];ib=remaining[q];ic=remaining[(q+1)%len(remaining)]
   a,b,c=poly[[ia,ib,ic]]
   if cross2(a,b,c)<=1e-14:continue
   blocked=False
   for j in remaining:
    if j in (ia,ib,ic):continue
    p=poly[j]
    if cross2(a,b,p)>=-1e-13 and cross2(b,c,p)>=-1e-13 and cross2(c,a,p)>=-1e-13:
     blocked=True;break
   if not blocked:
    tri.append([ia,ib,ic]);remaining.pop(q);clipped=True;break
  if not clipped:raise RuntimeError('ear clipping failed on exact intersection contour')
 if len(remaining)!=3:raise RuntimeError('ear clipping did not close contour')
 tri.append(remaining)
 return tri
def pointloc(local):
 return ex.point_location_prepared(tuple(Fraction.from_float(float(x)) for x in local),locprep)['location']
# Full local-104 bank data and exact point-location accelerator.
bankrec=records(B.reshape(-1,3),np.arange(3*len(B),dtype=np.int64).reshape(-1,3),bid)
locprep=ex.prepare_point_location(bankrec)
# Reconfirm the exact input crossing set before using its segment graph.
skin_in=records(base_local,F[old_rows],old_rows)
base_check=ex._audit_pair(skin_in,bankrec,same_surface=False)
if int(base_check['count'])!=graph['raw_crossing_pairs']:raise RuntimeError(f'exact input pair count differs from segment graph: {base_check["count"]} vs {graph["raw_crossing_pairs"]}')
# Exact crossing endpoints form one closed degree-2 contour.
edges=[tuple(map(int,e)) for e in graph['segments_point_indices']]
npts=len(graph['intersection_points_barycentric']);adj=[[] for _ in range(npts)]
for a,b in edges:adj[a].append(b);adj[b].append(a)
degrees=[len(x) for x in adj]
if any(x!=2 for x in degrees):raise RuntimeError(f'exact crossing contour is not a closed degree-2 loop: {Counter(degrees)}')
cycle=[0];prev=-1;cur=0
while True:
 opts=[x for x in adj[cur] if x!=prev]
 if not opts:raise RuntimeError('open exact crossing contour')
 nxt=opts[0]
 if nxt==cycle[0]:break
 if nxt in cycle:raise RuntimeError('exact crossing graph closes before visiting every node')
 cycle.append(nxt);prev,cur=cur,nxt
if len(cycle)!=npts:raise RuntimeError(f'exact crossing graph has multiple closed components: visited {len(cycle)} of {npts}')
# Source-face ancestry for all original boundary nodes and exact contour points.
node_bary={}
for fr in old_rows:
 for n,b in zip(F[fr],Z['refinement_face_parent_barycentric'][fr]):node_bary[int(n)]=np.asarray(b,np.float64)
if set(node_bary)!=set(loop):raise RuntimeError('source-face boundary identity mismatch')
corners=[next(n for n,b in node_bary.items() if np.array_equal(b,np.eye(3)[i])) for i in range(3)]
outer_bary=np.asarray([node_bary[n] for n in loop])
inner_bary_all=np.asarray(graph['intersection_points_barycentric'],np.float64)
inner_bary=inner_bary_all[cycle]
outer_xy=outer_bary[:,1:];inner_xy=inner_bary[:,1:]
outer_area=area2(outer_xy);inner_area=area2(inner_xy)
if outer_area<0:loop=list(reversed(loop));outer_bary=outer_bary[::-1];outer_xy=outer_xy[::-1];outer_area=-outer_area
if inner_area<0:cycle=list(reversed(cycle));inner_bary=inner_bary[::-1];inner_xy=inner_xy[::-1];inner_area=-inner_area
if inner_area<=0:raise RuntimeError('exact intersection contour has zero projected area')
if any(not point_in_poly(p,outer_xy) for p in inner_xy):raise RuntimeError('exact crossing contour exits the certified fixed boundary')
for i in range(npts):
 for j in range(i+1,npts):
  if j in (i,(i+1)%npts) or i==(j+1)%npts:continue
  if proper_intersects(inner_xy[i],inner_xy[(i+1)%npts],inner_xy[j],inner_xy[(j+1)%npts]):raise RuntimeError(f'exact projected contour is self-crossing at segments {i},{j}')
center=poly_centroid(inner_xy)
outer_star,outer_angle_diffs=is_star(outer_xy,center)
inner_star,inner_angle_diffs=is_star(inner_xy,center)
if not outer_star or not inner_star:raise RuntimeError(f'constrained annulus angular sweep requires star-shaped nested loops; outer={outer_star}, inner={inner_star}')
# Build a constrained annulus zipper that preserves every one of the 25 exact bone/skin segments.
def angle_order(poly):
 a=np.mod(np.arctan2(poly[:,1]-center[1],poly[:,0]-center[0]),2*np.pi)
 return list(np.argsort(a)),a
oo,oa=angle_order(outer_xy);ii,ia=angle_order(inner_xy);cur_o=oo[-1];cur_i=ii[-1];adv_o=0;adv_i=0;ann=[]
while adv_o<len(oo) or adv_i<len(ii):
 to=oa[oo[adv_o]] if adv_o<len(oo) else np.inf
 ti=ia[ii[adv_i]] if adv_i<len(ii) else np.inf
 if to<ti:
  j=oo[adv_o];ann.append([('o',cur_o),('o',j),('i',cur_i)]);cur_o=j;adv_o+=1
 else:
  j=ii[adv_i];ann.append([('o',cur_o),('i',j),('i',cur_i)]);cur_i=j;adv_i+=1
# Polygon cap triangulation uses exact contour constraints; centroids will be fitted to the bone exterior.
cap=ear_clip(inner_xy)
base_tris=ann+[[('i',int(a)),('i',int(b)),('i',int(c))] for a,b,c in cap]
def token_bary(t):return outer_bary[t[1]] if t[0]=='o' else inner_bary[t[1]]
def token_node(t):return loop[t[1]] if t[0]=='o' else len(X)+t[1]
# Verify the planar patch is a conforming covering of the fixed outer triangle.
base_area=0.0
for tri in base_tris:
 xy=np.asarray([token_bary(t)[1:] for t in tri])
 a=abs(cross2(xy[0],xy[1],xy[2]))*0.5
 if a<=1e-14:raise RuntimeError('constrained planar patch contains a degenerate base triangle')
 base_area+=a
if abs(base_area-outer_area)>max(1e-10,outer_area*1e-8):raise RuntimeError(f'constrained patch does not cover the boundary exactly: area {base_area}, outer {outer_area}')
# Outward normal for exact contour vertices, certified by exact point location on both sides.
def normal_outside_at(q,bone_ids):
 normals=[]
 for boneid in sorted(set(map(int,bone_ids))):
  bi=np.flatnonzero(bid==boneid)
  if len(bi)!=1:continue
  t=B[int(bi[0])];n=np.cross(t[1]-t[0],t[2]-t[0]);norm=np.linalg.norm(n)
  if norm==0:continue
  n=n/norm;chosen=None
  for eps in (1e-5,3e-5,1e-4):
   plus=pointloc(q+eps*n);minus=pointloc(q-eps*n)
   if plus=='outside' and minus=='inside':chosen=n;break
   if plus=='inside' and minus=='outside':chosen=-n;break
  if chosen is not None:normals.append(chosen)
 if not normals:raise RuntimeError(f'cannot orient an exterior bone normal at exact segment point {q.tolist()}')
 avg=np.sum(normals,axis=0);norm=np.linalg.norm(avg)
 if norm<1e-8:avg=normals[0];norm=np.linalg.norm(avg)
 return avg/norm
# Closest point and outward fit for all face-centroid nodes.
def nearest_bone(p):
 best=(1e99,None,None)
 for j,t in enumerate(B):
  q=closest(p,t);d=float(np.linalg.norm(p-q))
  if d<best[0]:best=(d,j,q)
 return best
def fit_point(bary,kind,bone_ids=()):
 affine_world=np.asarray(bary@X[np.asarray(corners)].astype(np.float64),np.float64)
 local=to_local(affine_world);state=pointloc(local);d,j,q=nearest_bone(local);moved=local.copy();method='affine-outside'
 if kind=='segment':
  q=local.copy();n=normal_outside_at(q,bone_ids);moved=q+0.00020*n;method='exact-intersection-contour-plus-0.20mm-outward'
 elif state=='inside':
  if d<=1e-12:raise RuntimeError('inside centroid has zero distance to bone boundary')
  moved=q+(q-local)/d*0.00020;method='nearest-bone-exterior-plus-0.20mm'
 elif state=='boundary':
  normal_ids=[int(bid[j])]
  n=normal_outside_at(q,normal_ids);moved=q+0.00020*n;method='boundary-normal-plus-0.20mm'
 elif state=='outside' and d<0.00020:
  if d<=1e-12:raise RuntimeError('outside boundary-fit centroid has zero nearest distance')
  moved=q+(local-q)/d*0.00020;method='nearest-bone-exterior-plus-0.20mm'
 elif state!='outside':raise RuntimeError(f'centroid point location unknown: {state}')
 world=to_world(moved);replayed=to_local(world);post=pointloc(replayed)
 displacement=float(np.linalg.norm(world.astype(np.float64)-affine_world))
 if displacement>0.004000001:raise RuntimeError(f'geometry-driven point exceeds 4 mm construction trust: {displacement*1000:.6f} mm')
 return world,{'source_barycentric':bary.tolist(),'source_location':state,'post_location':post,'fit_method':method,'nearest_bone_triangle':int(bid[j]),'nearest_boundary_distance_mm':d*1000,'world_f32_m':world.tolist(),'affine_source_world_m':affine_world.tolist(),'displacement_mm':displacement*1000,'bone_ids_for_contour_normal':list(map(int,bone_ids))}
# Segment point incident bone triangles from the exact pair segments.
incident=defaultdict(set)
for pair in graph['segment_crossing_pairs']:
 for pid0 in pair['point_ids']:incident[int(pid0)].add(int(pair['bone_triangle_row']))
# Convert each base constrained triangle centroid to a source-face barycentric node.
new_barys=[b.copy() for b in inner_bary]
new_specs=[]
for i,b in enumerate(inner_bary):new_specs.append(('segment',tuple(sorted(incident[cycle[i]]))))
for tri in base_tris:
 new_barys.append(np.mean([token_bary(t) for t in tri],axis=0));new_specs.append(('centroid',()))
new_barys=np.asarray(new_barys,np.float64)
new_world=[];node_motion=[]
for i,(b,spec) in enumerate(zip(new_barys,new_specs)):
 world,meta=fit_point(b,spec[0],spec[1]);new_world.append(world);meta.update({'new_node':len(X)+i,'node_role':spec[0]});node_motion.append(meta)
new_world=np.asarray(new_world,np.float32);trial_world=np.vstack([X,new_world])
trial_local=to_local(trial_world)
# Refine each constrained base triangle at its fitted centroid node, preserving all source edges.
fan=[];fan_bary=[]
for k,tri in enumerate(base_tris):
 cnode=len(X)+npts+k;nd=[token_node(t) for t in tri];bd=[token_bary(t) for t in tri];cb=new_barys[npts+k]
 children=[[nd[0],nd[1],cnode],[nd[1],nd[2],cnode],[nd[2],nd[0],cnode]]
 child_bary=[[bd[0],bd[1],cb],[bd[1],bd[2],cb],[bd[2],bd[0],cb]]
 for f,bf in zip(children,child_bary):
  xy=np.asarray(bf)[:,1:]
  if cross2(xy[0],xy[1],xy[2])<0:f[1],f[2]=f[2],f[1];bf[1],bf[2]=bf[2],bf[1]
  xyz=trial_local[np.asarray(f)];parent_normal=np.cross(base_local[corners[1]]-base_local[corners[0]],base_local[corners[2]]-base_local[corners[0]])
  if float(np.dot(np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0]),parent_normal))<0:f[1],f[2]=f[2],f[1];bf[1],bf[2]=bf[2],bf[1]
  fan.append(f);fan_bary.append(bf)
fan=np.asarray(fan,np.int64);fan_bary=np.asarray(fan_bary,np.float64);Fcount=len(F);fan_rows=old_rows+list(range(Fcount,Fcount+len(fan)-4))
trial_faces=F.copy();trial_faces[old_rows]=fan[:4];trial_faces=np.vstack([trial_faces,fan[4:]])
# Boundary and edge incidence, including every constrained inner contour edge.
patch_edges=Counter(tuple(sorted((int(a),int(b)))) for f in fan for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])))
full_edges=Counter(tuple(sorted((int(a),int(b)))) for f in trial_faces for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])))
outer_inc={f'{loop[i]}-{loop[(i+1)%len(loop)]}':{'patch':patch_edges[tuple(sorted((loop[i],loop[(i+1)%len(loop)])))],'full':full_edges[tuple(sorted((loop[i],loop[(i+1)%len(loop)])))]} for i in range(len(loop))}
inner_edge_ids=[(len(X)+i,len(X)+(i+1)%npts) for i in range(npts)]
inner_inc=[{'edge_nodes':list(e),'patch_incidence':patch_edges[tuple(sorted(e))],'full_incidence':full_edges[tuple(sorted(e))]} for e in inner_edge_ids]
if any(x['patch']!=1 or x['full']!=2 for x in outer_inc.values()):raise RuntimeError(f'fixed boundary incidence failed: {outer_inc}')
if any(x['patch_incidence']!=2 or x['full_incidence']!=2 for x in inner_inc):raise RuntimeError('exact intersection segment constraints are not conforming in output patch')
# Local shape and 037-reference area metrics; report even for a failed exact screen.
def tri_area(v,f):return float(np.linalg.norm(np.cross(v[f[1]]-v[f[0]],v[f[2]]-v[f[0]]))*0.5)
def tri_aspect(v,f):
 e=[np.linalg.norm(v[f[1]]-v[f[0]]),np.linalg.norm(v[f[2]]-v[f[1]]),np.linalg.norm(v[f[0]]-v[f[2]])]
 a=tri_area(v,f)
 return float(max(e)**2/(2*np.sqrt(3)*a)) if a>0 else float('inf')
refsrc=np.asarray(Z['area_reference_037_points_world_m_f64'],np.float64)
def reference_positions(barys,nodes=None):
 if nodes is None:return np.asarray([b@refsrc[np.asarray(corners)] for b in barys],np.float64)
 return np.asarray([refsrc[int(n)] if n<len(X) else barys[k]@refsrc[np.asarray(corners)] for k,n in enumerate(nodes)],np.float64)
old_ref=refsrc
base_area_vals=np.asarray([tri_area(X,f) for f in F[old_rows]])
base_ref_vals=np.asarray([tri_area(old_ref,f) for f in F[old_rows]])
base_ratios=base_area_vals/np.maximum(base_ref_vals,1e-30)
fan_ref_world=np.zeros((len(X)+len(new_barys),3),np.float64);fan_ref_world[:len(X)]=refsrc
for i,b in enumerate(new_barys):fan_ref_world[len(X)+i]=b@refsrc[np.asarray(corners)]
fan_area_vals=np.asarray([tri_area(trial_world,f) for f in fan])
fan_ref_vals=np.asarray([tri_area(fan_ref_world,f) for f in fan])
fan_ratios=fan_area_vals/np.maximum(fan_ref_vals,1e-30)
fan_aspects=np.asarray([tri_aspect(trial_world,f) for f in fan])
base_aspects=np.asarray([tri_aspect(X,f) for f in F[old_rows]])
new_ref_dist=np.asarray([np.linalg.norm(new_world[i].astype(np.float64)-new_barys[i]@refsrc[np.asarray(corners)]) for i in range(len(new_barys))])
old_patch_nodes=sorted(set(int(n) for f in F[old_rows] for n in f))
old_ref_dist=np.asarray([np.linalg.norm(X[n].astype(np.float64)-refsrc[n]) for n in old_patch_nodes])
shape={'base_face_count':len(old_rows),'base_total_skin_area_mm2':float(base_area_vals.sum()*1e6),'base_total_037_reference_area_mm2':float(base_ref_vals.sum()*1e6),'base_total_area_ratio':float(base_area_vals.sum()/max(base_ref_vals.sum(),1e-30)),'base_max_face_area_ratio':float(base_ratios.max()),'base_max_aspect_ratio':float(base_aspects.max()),'base_min_angle_deg':None,'base_max_patch_node_distance_to_037_mm':float(old_ref_dist.max()),'trial_face_count':len(fan),'trial_total_skin_area_mm2':float(fan_area_vals.sum()*1e6),'trial_total_037_reference_area_mm2':float(fan_ref_vals.sum()*1e6),'trial_total_area_ratio':float(fan_area_vals.sum()/max(fan_ref_vals.sum(),1e-30)),'trial_max_face_area_ratio':float(fan_ratios.max()),'trial_p95_face_area_ratio':float(np.percentile(fan_ratios,95)),'trial_max_aspect_ratio':float(fan_aspects.max()),'trial_p95_aspect_ratio':float(np.percentile(fan_aspects,95)),'trial_min_angle_deg':None,'trial_max_new_node_distance_to_037_mm':float(new_ref_dist.max()),'max_new_node_source_displacement_mm':float(max(x['displacement_mm'] for x in node_motion))}
def min_angle(v,f):
 pts=v[f];ls=[np.linalg.norm(pts[1]-pts[0]),np.linalg.norm(pts[2]-pts[1]),np.linalg.norm(pts[0]-pts[2])]
 out=[]
 for i in range(3):
  a=pts[(i+1)%3]-pts[i];b=pts[(i+2)%3]-pts[i];out.append(float(np.degrees(np.arccos(np.clip((a@b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)))))
 return min(out)
shape['base_min_angle_deg']=float(min(min_angle(X,f) for f in F[old_rows]))
shape['trial_min_angle_deg']=float(min(min_angle(trial_world,f) for f in fan))
# Exact local skin/bone screen and exact witnesses on the actual Float32 patch.
skinrec=records(trial_local,fan,fan_rows);bank_check=ex._audit_pair(skinrec,bankrec,same_surface=False)
residual=[]
by_face={int(r):i for i,r in enumerate(fan_rows)}
for skinrow,boneid in bank_check['triangle_pairs']:
 fi=by_face[int(skinrow)];s=trial_local[fan[fi]];bi=np.flatnonzero(bid==int(boneid))
 if len(bi)!=1:continue
 b=B[int(bi[0])]
 raw=ex.triangle_intersection_points(tuple(tuple(Fraction.from_float(float(v)) for v in q) for q in s),tuple(tuple(Fraction.from_float(float(v)) for v in q) for q in b));uniq=[]
 for p in raw:
  if p not in uniq:uniq.append(p)
 residual.append({'skin_face_row':int(skinrow),'stable_bone_triangle_row':int(boneid),'intersection_vertices_body74_local_m':[[float(v) for v in p] for p in uniq]})
sample_locs=[pointloc(to_local(p)) for p in new_world]
report={'schema':'numi.human.per_bank_worker_segment_guided_patch.v1','input_candidate_path':str(C),'input_candidate_sha256':basehash,'base_candidate_status':'COMBINED_CONSTRUCTION_ONLY_653','segment_graph_report_path':str(out/'segment-graph-report.json'),'segment_graph_report_sha256':hashlib.sha256((out/'segment-graph-report.json').read_bytes()).hexdigest(),'boundary_feasibility_path':str(S/'per-bank-workers-20261027/b74-104/steiner-multinode-001/boundary-feasibility.json'),'fixed_boundary_nodes':loop,'fixed_boundary_vertices_all_outside':True,'fixed_boundary_edges_all_disjoint_from_bone':True,'source_fine_detail_face':10342,'source_fine_subface_row':10344,'immediate_parent_face_row':22189,'exact_input_raw_child_pairs':int(base_check['count']),'exact_intersection_segments':len(edges),'exact_intersection_unique_points':npts,'exact_intersection_graph_degrees':degrees,'exact_contour_cycle_order':cycle,'exact_contour_projected_area_barycentric2':inner_area,'outer_projected_area_barycentric2':outer_area,'outer_and_inner_star_shaped':{'outer':outer_star,'inner':inner_star},'max_exact_contour_projection_residual_m':graph['max_projection_residual_m'],'constrained_planar_base_triangles':len(base_tris),'constrained_segments_preserved':len(inner_inc),'annulus_triangles':len(ann),'interior_cap_triangles_before_centroid_split':len(cap),'centroid_nodes_added':len(base_tris),'new_nodes_count':len(new_barys),'new_faces_count':len(fan),'boundary_edge_incidence':outer_inc,'exact_contour_edge_incidence':inner_inc,'node_motion':node_motion,'all_new_node_point_locations':sample_locs,'shape_metrics':shape,'raw_exact_bank104_child_pairs_after':int(bank_check['count']),'bank104_pair_rows_after':bank_check['triangle_pairs'],'bank104_residual_exact_witnesses':residual,'local_exact_bank104_pass':int(bank_check['count'])==0 and all(x=='outside' for x in sample_locs),'bank_input_pins':pins,'accepted_physical_steps':0,'geometry_admitted':False,'full_audits_run':False,'local_self_test_run':False}
# Preserve exact evidence on any failed local exact bank gate and stop before unrelated work.
if not report['local_exact_bank104_pass']:
 patch=out/'segment-guided-failed-patch.npz'
 np.savez_compressed(patch,points_world_m=trial_world,source_O_faces=trial_faces,trial_patch_faces=fan,trial_patch_output_rows=np.asarray(fan_rows,np.int64),trial_patch_source_barycentric_vertices=fan_bary,trial_boundary_nodes=np.asarray(loop,np.int64),trial_contour_nodes=np.arange(len(X),len(X)+npts,dtype=np.int64),trial_contour_source_barycentric=inner_bary,trial_new_node_source_barycentric=new_barys,trial_new_node_delta_world_m=np.asarray([np.asarray(x['world_f32_m'])-np.asarray(x['affine_source_world_m']) for x in node_motion],np.float32))
 report.update({'status':'FAIL_EXACT_LOCAL_SKIN_BONE','failed_patch_path':str(patch),'failed_patch_sha256':hashlib.sha256(patch.read_bytes()).hexdigest(),'local_self_test_run':False,'next_method':'The exact intersection loop was constrained and the interior collar was fitted to bone-104 exterior; inspect the preserved exact child witnesses and shape metrics before any further method.'})
 (out/'segment-guided-report.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('node_motion','bank_input_pins','bank104_residual_exact_witnesses','bank104_pair_rows_after','exact_contour_cycle_order','boundary_edge_incidence','exact_contour_edge_incidence','all_new_node_point_locations')},indent=2))
 raise SystemExit(0)
# Exact self-pair check on every patch face against the complete output skin mesh.
allrec,_=ex.float32_surface_records(trial_world,trial_faces,source_seam_groups=())
self_check=ex._audit_self_pairs_touching_faces(allrec,set(fan_rows))
report.update({'local_self_test_run':True,'local_self_pair_count':int(self_check['count']),'local_self_pair_rows':self_check['triangle_pairs']})
if int(self_check['count']):
 report.update({'status':'FAIL_EXACT_LOCAL_SELF','next_method':'The segment-guided patch cleared bone 104 but has exact local skin self pairs; preserve the exact pairs and do not emit a candidate.'})
 (out/'segment-guided-report.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('node_motion','bank_input_pins','bank104_residual_exact_witnesses','bank104_pair_rows_after','exact_contour_cycle_order','boundary_edge_incidence','exact_contour_edge_incidence','all_new_node_point_locations')},indent=2))
 raise SystemExit(0)
# Shape gate prevents passing clearance by introducing a new spike.
shape_ok=(shape['trial_total_area_ratio']<=1.25*shape['base_total_area_ratio'] and shape['trial_max_new_node_distance_to_037_mm']<=shape['base_max_patch_node_distance_to_037_mm']+0.25 and shape['trial_max_aspect_ratio']<=max(3.0,1.5*shape['base_max_aspect_ratio']))
if not shape_ok:
 report.update({'status':'FAIL_LOCAL_SHAPE_BLOWUP','shape_gate_pass':False,'local_self_pair_count':int(self_check['count']),'next_method':'Exact local bank and self screens pass, but the candidate exceeds the conservative local area/reference-distance/aspect gate; do not accept clearance by creating a spike.'})
 (out/'segment-guided-report.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('node_motion','bank_input_pins','bank104_residual_exact_witnesses','bank104_pair_rows_after','exact_contour_cycle_order','boundary_edge_incidence','exact_contour_edge_incidence','all_new_node_point_locations')},indent=2))
 raise SystemExit(0)
# Fully typed candidate: append barycentric Steiner rows and replace the four source-face children.
N=len(X);new_count=len(new_barys);payload={k:np.array(Z[k]) for k in Z.files}
face_fields={'face_kind':2,'parent_coarse_face_row':-1,'parent_fine_detail_face_row':10342,'parent_source_face_marker':10342,'parent_fine_subface_row':10344,'refinement_face_parent_face_row':22189,'refinement_face_parent_ancestry_face_rows':22189}
for key,val in face_fields.items():
 a=payload[key].copy();a[old_rows]=val;payload[key]=np.concatenate([a,np.full((len(fan)-4,)+a.shape[1:],val,dtype=a.dtype)],axis=0)
for key in ['source_O_faces','area_reference_037_source_O_faces']:
 a=payload[key].copy();a[old_rows]=fan[:4];payload[key]=np.vstack([a,fan[4:]])
payload['refinement_face_parent_barycentric'][old_rows]=fan_bary[:4]
payload['refinement_face_parent_barycentric']=np.concatenate([payload['refinement_face_parent_barycentric'],fan_bary[4:]],axis=0)
for ptrkey,rowkey,newseg in [('face_source_ancestry_indptr','face_source_ancestry_face_rows',[10342]),('face_fine_subface_ancestry_indptr','face_fine_subface_ancestry_rows',[10344]),('face_fine_detail_ancestry_indptr','face_fine_detail_ancestry_rows',[10342]),('refinement_face_parent_ancestry_indptr','refinement_face_parent_ancestry_face_rows',[22189])]:
 oldptr=np.asarray(Z[ptrkey],np.int64);oldvals=np.asarray(Z[rowkey]);segs=[]
 for faceid in range(Fcount):segs.append(np.asarray(newseg,dtype=oldvals.dtype) if faceid in old_rows else oldvals[oldptr[faceid]:oldptr[faceid+1]])
 segs.extend([np.asarray(newseg,dtype=oldvals.dtype) for _ in range(len(fan)-4)])
 ptr=np.zeros(Fcount+len(fan)-4+1,np.int64);flat=[]
 for i,seg in enumerate(segs):flat.extend(seg.tolist());ptr[i+1]=len(flat)
 payload[ptrkey]=ptr;payload[rowkey]=np.asarray(flat,dtype=oldvals.dtype)
node_keys=['points_world_m','node_kind','node_source_row','node_source_virtual_row','node_parent_coarse_face_row','node_parent_coarse_edge_a','node_parent_coarse_edge_b','node_parent_coarse_edge_parameter','node_source_displacement_m','points_before_connected_source_unfold_m','node_connected_source_unfold_displacement_m','points_before_joint_self_pair_repair_m','node_joint_self_pair_repair_displacement_m','node_box_constrained_source_fidelity_delta_m','node_source_fidelity_target_mask','all_inherited_exit_target_mask','refinement_node_kind','refinement_node_parent_edge_rows','refinement_node_parent_edge_weights','local_witness_cluster_199_delta_m','local_witness_cluster_199_source_reference_m','area_reference_037_points_world_m_f64','local_witness_cluster_222_delta_m','local_witness_cluster_222_source_reference_R0_m','local_witness_cluster_222_area_reference_037_m']
for key in node_keys:
 a=payload[key];new=[]
 for i,bary in enumerate(new_barys):
  if key=='points_world_m':v=trial_world[N+i]
  elif key=='node_kind':v=np.asarray(-1,dtype=a.dtype)
  elif key in ['node_source_row','node_source_virtual_row','node_parent_coarse_face_row','node_parent_coarse_edge_a','node_parent_coarse_edge_b','node_parent_coarse_edge_parameter']:v=np.asarray(-1,dtype=a.dtype)
  elif key=='node_source_displacement_m':v=np.asarray(node_motion[i]['displacement_mm']/1000.0,dtype=a.dtype)
  elif key in ['node_source_fidelity_target_mask','all_inherited_exit_target_mask']:v=np.asarray(False,dtype=a.dtype)
  elif key=='refinement_node_kind':v=np.asarray(2,dtype=a.dtype)
  elif key in ['refinement_node_parent_edge_rows','refinement_node_parent_edge_weights']:v=np.asarray([-1,-1],dtype=a.dtype)
  else:v=np.tensordot(bary,a[np.asarray(corners)],axes=(0,0)).astype(a.dtype)
  new.append(np.asarray(v,dtype=a.dtype).reshape((1,)+a.shape[1:]))
 payload[key]=np.concatenate([a]+new,axis=0)
ptr=np.asarray(Z['refinement_node_parent_face_indptr'],np.int64)
payload['refinement_node_parent_face_indptr']=np.concatenate([ptr,ptr[-1]+np.arange(1,new_count+1,dtype=np.int64)])
payload['refinement_node_parent_face_rows']=np.concatenate([Z['refinement_node_parent_face_rows'],np.full(new_count,22189,dtype=Z['refinement_node_parent_face_rows'].dtype)])
payload['refinement_node_parent_face_barycentric']=np.concatenate([Z['refinement_node_parent_face_barycentric'],new_barys],axis=0)
imm_idx=[];imm_w=[];src_idx=[];src_w=[];src_ptr=np.asarray(Z['source_prolongation_indptr'],np.int64);imm_ptr=np.asarray(Z['immediate_refinement_prolongation_indptr'],np.int64)
for bary in new_barys:
 imm_idx.extend(corners);imm_w.extend(bary.tolist());terms={}
 for j,node in enumerate(corners):
  a,b=map(int,src_ptr[node:node+2])
  for k,w in zip(Z['source_prolongation_node_rows'][a:b],Z['source_prolongation_weights'][a:b]):terms[int(k)]=terms.get(int(k),0.0)+float(bary[j])*float(w)
 terms={k:v for k,v in sorted(terms.items()) if abs(v)>1e-15}
 if not np.isclose(sum(terms.values()),1.0,rtol=0,atol=1e-12):raise RuntimeError('composed source interpolation row does not sum to one')
 src_idx.extend(terms);src_w.extend(terms.values());src_ptr=np.append(src_ptr,int(src_ptr[-1])+len(terms));imm_ptr=np.append(imm_ptr,int(imm_ptr[-1])+3)
payload['source_prolongation_indptr']=src_ptr;payload['source_prolongation_node_rows']=np.concatenate([Z['source_prolongation_node_rows'],np.asarray(src_idx,dtype=Z['source_prolongation_node_rows'].dtype)])
payload['source_prolongation_weights']=np.concatenate([Z['source_prolongation_weights'],np.asarray(src_w,dtype=Z['source_prolongation_weights'].dtype)])
payload['immediate_refinement_prolongation_indptr']=imm_ptr;payload['immediate_refinement_prolongation_node_rows']=np.concatenate([Z['immediate_refinement_prolongation_node_rows'],np.asarray(imm_idx,dtype=Z['immediate_refinement_prolongation_node_rows'].dtype)])
payload['immediate_refinement_prolongation_weights']=np.concatenate([Z['immediate_refinement_prolongation_weights'],np.asarray(imm_w,dtype=Z['immediate_refinement_prolongation_weights'].dtype)])
delta=np.zeros((N+new_count,3),np.float32)
for i,bary in enumerate(new_barys):delta[N+i]=trial_world[N+i]-np.asarray(bary@X[np.asarray(corners)].astype(np.float64),np.float32)
payload['worker_b74_104_delta_world_m']=delta
payload['worker_b74_104_scope_nodes']=np.asarray(loop+list(range(N,N+new_count)),np.int32)
payload['worker_b74_104_scope_faces']=np.asarray(fan_rows,np.int32)
payload['worker_b74_104_source_candidate_sha256']=np.asarray([basehash])
payload['worker_b74_104_method']=np.asarray(['fixed six-node source-face boundary; constrained 25-edge exact bone/skin intersection contour; annulus and clipped interior cap fitted to bone exterior with exact local gates and barycentric ancestry'])
payload['worker_b74_104_construction_trust_limit_m']=np.asarray([0.004],np.float64)
payload['worker_b74_104_new_node_parent_face_row']=np.full(new_count,22189,np.int64)
payload['worker_b74_104_new_node_parent_barycentric']=new_barys
payload['refinement_node_kind_semantics']=np.asarray(['0=preexisting; 1=selected shared-edge midpoint; 2=local interior Steiner node; non-affine coordinate offsets are separately recorded in worker_b74_104_delta_world_m'])
payload['refinement_node_parent_face_barycentric_semantics']=np.asarray(['local face-interior Steiner ancestry in the immediate parent input face; actual geometry offset is separately recorded in worker_b74_104_delta_world_m'])
payload['source_prolongation_semantics']=np.asarray(['rows map output nodes through immediate refinement to preserved source rows; each local Steiner row is barycentric interpolation and its construction offset is separately recorded'])
payload['immediate_refinement_prolongation_semantics']=np.asarray(['rows map output nodes to immediate parent by identity, edge midpoint, or face-interior barycentric interpolation; construction offset is separately recorded'])
payload['refinement_face_parent_barycentric_semantics']=np.asarray(['output child vertices retain source barycentric ancestry in pre-refinement parent face; non-affine Steiner position offsets are separately recorded'])
payload['refinement_boundary_semantics']=np.asarray(['fixed six-node boundary identities and coordinates preserved; exact intersection contour is a conforming interior edge loop with source barycentric ancestry'])
for key in Z.files:
 a=Z[key]
 if a.ndim and a.shape[0]==N and payload[key].shape[0]!=N+new_count:raise RuntimeError(f'node-aligned payload mismatch {key}: {payload[key].shape}')
 if a.ndim and a.shape[0]==Fcount and payload[key].shape[0]!=Fcount+len(fan)-4:raise RuntimeError(f'face-aligned payload mismatch {key}: {payload[key].shape}')
for key in ['refinement_node_parent_face_indptr','source_prolongation_indptr','immediate_refinement_prolongation_indptr']:
 if payload[key].shape[0]!=N+new_count+1:raise RuntimeError(f'node CSR pointer mismatch {key}')
for key in ['face_source_ancestry_indptr','face_fine_subface_ancestry_indptr','face_fine_detail_ancestry_indptr','refinement_face_parent_ancestry_indptr']:
 if payload[key].shape[0]!=Fcount+len(fan)-4+1:raise RuntimeError(f'face CSR pointer mismatch {key}')
if not np.array_equal(payload['source_O_faces'],payload['area_reference_037_source_O_faces']):raise RuntimeError('area-reference source faces differ from candidate')
for fr in fan_rows:
 if int(payload['parent_fine_detail_face_row'][fr])!=10342 or int(payload['parent_source_face_marker'][fr])!=10342 or int(payload['refinement_face_parent_face_row'][fr])!=22189:raise RuntimeError('face ancestry mismatch')
cand=out/'candidate-segment-guided-001.npz';np.savez_compressed(cand,**payload)
report.update({'status':'PASS_LOCAL_EXACT_BANK_SELF_AND_SHAPE','local_self_pair_count':0,'candidate_path':str(cand),'candidate_sha256':hashlib.sha256(cand.read_bytes()).hexdigest(),'candidate_point_count':N+new_count,'candidate_face_count':Fcount+len(fan)-4,'payload_key_count':len(payload),'face_ancestry_update':'replace four patch faces with constrained segment-guided patch faces; each output face maps to fine-detail 10342, fine-subface 10344, immediate input row 22189, and per-vertex source barycentrics','node_ancestry_update':'append source barycentric refinement-node-kind=2 rows; immediate interpolation to three parent face corners; composed source prolongation and Float32 geometry deltas recorded','full_audits_run':False,'shape_gate_pass':True})
(out/'segment-guided-report.json').write_text(json.dumps(report,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('node_motion','bank_input_pins','bank104_residual_exact_witnesses','bank104_pair_rows_after','exact_contour_cycle_order','boundary_edge_incidence','exact_contour_edge_incidence','all_new_node_point_locations')},indent=2))

