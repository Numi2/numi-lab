from pathlib import Path
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib, importlib.util, json, sys, time, traceback
import numpy as np
from scipy.optimize import minimize

S=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638")
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
C=S/"partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz"
AUDIT159=C.parent/"audit-159/report.json"
W=S/"per-bank-workers-20261027/b115-118/attempt-006-exact-endpoint-identity/wider-fit-trial-001/topology"
OUT=Path(__file__).resolve().parent
HELPER=Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
AUDIT_TOOL=ROOT/"audit_skin_bone_surfaces_159.py"
BANKDIR=E/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
FIT_MM=1.0
SCALE_M=FIT_MM/1000.0
CLEARANCE_MM=0.05
AREA_FLOOR=0.10
MAX_ASPECT=50.0
MIN_ANGLE_DEG=0.5
MAXITER=250
SMOOTH_WEIGHT=0.50
CAND_PIN="7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af"
WIDE_PIN="67c90a682da5213bb7f26d166565b1a7e36d502dff504ab2ffd7ebf734ed14e4"

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path)
 m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def frac(x):return Fraction.from_float(float(x))
def exact_tri(tri):return tuple(tuple(frac(x) for x in p) for p in np.asarray(tri).reshape(3,3))
def exact_barycentric(p,tri,exact):
 n=exact._cross(exact._sub(tri[1],tri[0]),exact._sub(tri[2],tri[0]))
 axes=exact._axes(n)
 a,b,c,q=[tuple(v[k] for k in axes) for v in (tri[0],tri[1],tri[2],p)]
 den=exact._orient2(a,b,c)
 if den==0:raise ValueError("exact parent face is degenerate")
 return (exact._orient2(b,c,q)/den,exact._orient2(c,a,q)/den,exact._orient2(a,b,q)/den)
def closest_point_triangle(p,a,b,c):
 ab=b-a;ac=c-a;ap=p-a
 d1=float(ab@ap);d2=float(ac@ap)
 if d1<=0 and d2<=0:return a.copy()
 bp=p-b;d3=float(ab@bp);d4=float(ac@bp)
 if d3>=0 and d4<=d3:return b.copy()
 vc=d1*d4-d3*d2
 if vc<=0 and d1>=0 and d3<=0:return a+(d1/(d1-d3))*ab
 cp=p-c;d5=float(ab@cp);d6=float(ac@cp)
 if d6>=0 and d5<=d6:return c.copy()
 vb=d5*d2-d1*d6
 if vb<=0 and d2>=0 and d6<=0:return a+(d2/(d2-d6))*ac
 va=d3*d6-d5*d4
 if va<=0 and d4-d3>=0 and d5-d6>=0:return b+((d4-d3)/((d4-d3)+(d5-d6)))*(c-b)
 den=1.0/(va+vb+vc);v=vb*den;w=vc*den
 return a+ab*v+ac*w
def exact_records_mesh(points,faces,face_ids,vertex_ids,exact):
 pts=[tuple(frac(x) for x in p) for p in np.asarray(points).reshape(-1,3)]
 out=[]
 for fid,face in zip(face_ids,faces):
  ids=tuple(map(int,face));tri=tuple(pts[i] for i in ids)
  n=exact._cross(exact._sub(tri[1],tri[0]),exact._sub(tri[2],tri[0]))
  if not any(n):raise ValueError(f"exact degenerate emitted triangle {fid}")
  lo=tuple(min(p[k] for p in tri) for k in range(3));hi=tuple(max(p[k] for p in tri) for k in range(3))
  out.append((tri,lo,hi,int(fid),tuple(int(vertex_ids[i]) for i in ids)))
 return out
def ancestry_repeat(indptr,rows,parent_rows):
 outp=[0];outr=[]
 for p in parent_rows:
  vals=rows[int(indptr[int(p)]):int(indptr[int(p)+1])]
  for _ in range(3):
   outr.extend(map(int,vals));outp.append(len(outr))
 return np.asarray(outp,np.int64),np.asarray(outr,np.int64)
def metrics(points,faces,parents,P,F):
 tri=np.asarray(points,dtype=np.float32)[faces].astype(np.float64)
 par=np.asarray(P,dtype=np.float32)[F[parents]].astype(np.float64)
 pn=np.cross(par[:,1]-par[:,0],par[:,2]-par[:,0])
 cn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
 pn2=np.einsum("ij,ij->i",pn,pn);plen=np.sqrt(np.maximum(pn2,1e-300))
 area_rel=np.linalg.norm(cn,axis=1)/plen
 orient=np.einsum("ij,ij->i",pn,cn)/np.maximum(pn2,1e-300)
 angles=[];aspects=[]
 for t in tri:
  lens=[float(np.linalg.norm(t[(i+1)%3]-t[i])) for i in range(3)]
  area=float(np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0]))*0.5)
  aspects.append(max(lens)**2/(4*np.sqrt(3)*max(area,1e-30)))
  aa=[]
  for i in range(3):
   u=t[(i+1)%3]-t[i];v=t[(i+2)%3]-t[i]
   aa.append(float(np.degrees(np.arccos(np.clip((u@v)/max(np.linalg.norm(u)*np.linalg.norm(v),1e-30),-1,1)))))
  angles.append(min(aa))
 return {"face_count":int(len(faces)),"minimum_unsigned_parent_area_ratio":float(area_rel.min()),
  "minimum_signed_orientation_ratio":float(orient.min()),"minimum_angle_degrees":float(min(angles)),
  "p05_angle_degrees":float(np.percentile(angles,5)),"maximum_aspect_measure":float(max(aspects)),
  "p95_aspect_measure":float(np.percentile(aspects,95)),
  "orientation_positive":bool(np.all(orient>0))}
def disk_boundary(faces):
 edge_uses=Counter();adj=defaultdict(set)
 for f in np.asarray(faces,np.int64):
  for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
   a=int(a);b=int(b);edge=tuple(sorted((a,b)));edge_uses[edge]+=1;adj[a].add(b);adj[b].add(a)
 if any(v not in (1,2) for v in edge_uses.values()):raise ValueError("parent patch has nonmanifold edge incidence")
 boundary=sorted(e for e,n in edge_uses.items() if n==1)
 verts=set(adj);seen=set();components=0
 for v in verts:
  if v in seen:continue
  components+=1;stack=[v];seen.add(v)
  while stack:
   u=stack.pop()
   for w in adj[u]:
    if w not in seen:seen.add(w);stack.append(w)
 bdeg=Counter()
 for a,b in boundary:bdeg[a]+=1;bdeg[b]+=1
 euler=len(verts)-len(edge_uses)+len(faces)
 return boundary,{"vertices":len(verts),"edges":len(edge_uses),"faces":int(len(faces)),
  "boundary_edges":len(boundary),"boundary_degree_histogram":{str(k):int(v) for k,v in Counter(bdeg.values()).items()},
  "connected_components":components,"euler_characteristic":int(euler),
  "is_single_disk":bool(components==1 and euler==1 and all(v==2 for v in bdeg.values()))}

started=time.monotonic()
report={"schema":"numi.human.bank115_118.bone-aware-centroid-fan.v1","status":"STARTED",
 "candidate_path":str(C),"wide_topology_path":str(W/"patch-candidate.npz"),
 "method":"Replace the exact-contour subdivision with a per-source-face interior barycentric vertex and three child faces; retain the exact source boundary and CSR ancestry; fit only interior nodes against exact bone-derived halfspaces and positive parent-oriented area constraints.",
 "candidate_modified":False,"geometry_or_native_admission":False,"fit_scale_mm":FIT_MM,
 "constraints":{"bone_clearance_mm":CLEARANCE_MM,"child_projected_area_ratio_floor":AREA_FLOOR,
   "SLSQP_ftol":1e-9,"SLSQP_maxiter":MAXITER,"smoothness_weight":SMOOTH_WEIGHT,
   "emitted_shape_gate":{"min_angle_degrees":MIN_ANGLE_DEG,"max_aspect_measure":MAX_ASPECT,"min_parent_relative_area_ratio":AREA_FLOOR}}}
try:
 if sha(C)!=CAND_PIN:raise RuntimeError("accepted candidate pin changed")
 if sha(W/"patch-candidate.npz")!=WIDE_PIN:raise RuntimeError("source patch pin changed")
 Z=npz(C);Q=npz(W/"patch-candidate.npz")
 P=np.asarray(Z["points_world_m"],np.float32);F=np.asarray(Z["source_O_faces"],np.int64)
 source_parent_rows=np.asarray(Q["original_patch_face_rows"],np.int64)
 if len(np.unique(source_parent_rows))!=len(source_parent_rows):raise RuntimeError("duplicate source parent face rows")
 if np.any((source_parent_rows<0)|(source_parent_rows>=len(F))):raise RuntimeError("source parent face row outside candidate")
 source_faces=F[source_parent_rows]
 boundary_src,topo=disk_boundary(source_faces)
 source_vertices=np.unique(source_faces.reshape(-1))
 qsrc=np.asarray(Q["source_candidate_node_rows"],np.int64)
 qb=np.asarray(Q["patch_outer_boundary_edges"],np.int64)
 if np.any(qsrc[qb]<0):raise RuntimeError("fixed boundary contains a generated node without direct source identity")
 q_boundary={tuple(sorted((int(qsrc[a]),int(qsrc[b])))) for a,b in qb}
 if set(boundary_src)!=q_boundary or len(qb)!=len(boundary_src):raise RuntimeError("source-face disk boundary differs from pinned 47-edge boundary")
 if not topo["is_single_disk"] or len(boundary_src)!=47:raise RuntimeError("source patch is not the expected one-loop disk")
 src_ind=np.asarray(Z["face_source_ancestry_indptr"],np.int64)
 src_rows=np.asarray(Z["face_source_ancestry_face_rows"],np.int64)
 sub_ind=np.asarray(Z["face_fine_subface_ancestry_indptr"],np.int64)
 sub_rows=np.asarray(Z["face_fine_subface_ancestry_rows"],np.int64)
 det_ind=np.asarray(Z["face_fine_detail_ancestry_indptr"],np.int64)
 det_rows=np.asarray(Z["face_fine_detail_ancestry_rows"],np.int64)
 if any(src_ind[r+1]<=src_ind[r] for r in source_parent_rows):raise RuntimeError("a parent face lacks source-face ancestry")
 report.update({"input_pins":{"candidate_sha256":sha(C),"wide_topology_sha256":sha(W/"patch-candidate.npz"),
    "candidate_surface_report_sha256":sha(AUDIT159)},
    "disk_preflight":topo|{"boundary_matches_pinned_topology":True,"all_parent_faces_have_source_ancestry":True,
       "source_parent_face_rows":int(len(source_parent_rows))},
    "source_face_ancestry_parent_count":int(sum(src_ind[r+1]-src_ind[r] for r in source_parent_rows))})

 # New topology: one interior point in every source parent face, joined to the three
 # source edges. A half-centroid / half-incenter barycentric point keeps area spread
 # bounded while improving element aspect over a pure centroid fan.
 src_to_local={int(v):i for i,v in enumerate(source_vertices)}
 nsrc=len(source_vertices);nparent=len(source_parent_rows)
 init_world=np.empty((nsrc+nparent,3),np.float64)
 init_world[:nsrc]=P[source_vertices].astype(np.float64)
 center_bary=np.empty((nparent,3),np.float64)
 for j,fi in enumerate(source_parent_rows):
  tri=P[F[int(fi)]].astype(np.float64)
  edge_opposite=np.asarray([np.linalg.norm(tri[2]-tri[1]),np.linalg.norm(tri[0]-tri[2]),np.linalg.norm(tri[1]-tri[0])])
  if not np.isfinite(edge_opposite).all() or edge_opposite.sum()<=0:raise RuntimeError(f"bad source face edges at row {int(fi)}")
  incenter=edge_opposite/edge_opposite.sum()
  center_bary[j]=0.5*incenter+0.5/3.0
  init_world[nsrc+j]=center_bary[j]@tri
 patch_faces=[];face_parent=[];face_bary=[]
 unit_barys=np.eye(3,dtype=np.float64)
 for j,fi in enumerate(source_parent_rows):
  va,vb,vc=map(int,F[int(fi)]);a,b,c=(src_to_local[va],src_to_local[vb],src_to_local[vc]);m=nsrc+j
  patch_faces.extend(([a,b,m],[b,c,m],[c,a,m]))
  mu=center_bary[j]
  face_parent.extend((int(fi),int(fi),int(fi)))
  face_bary.extend((np.asarray([unit_barys[0],unit_barys[1],mu]),
                    np.asarray([unit_barys[1],unit_barys[2],mu]),
                    np.asarray([unit_barys[2],unit_barys[0],mu])))
 patch_faces=np.asarray(patch_faces,np.int64);face_parent=np.asarray(face_parent,np.int64);face_bary=np.asarray(face_bary,np.float64)
 boundary_edges=np.asarray([[src_to_local[a],src_to_local[b]] for a,b in boundary_src],np.int64)
 fixed_nodes=np.unique(boundary_edges.reshape(-1))
 psource=np.r_[source_vertices,np.full(nparent,-1,np.int64)]
 pparent=np.r_[np.full(nsrc,-1,np.int64),source_parent_rows]
 pbary=np.vstack([np.zeros((nsrc,3),np.float64),center_bary])
 source_csr=ancestry_repeat(src_ind,src_rows,face_parent)
 sub_csr=ancestry_repeat(sub_ind,sub_rows,face_parent)
 detail_csr=ancestry_repeat(det_ind,det_rows,face_parent)
 patch_data={"points_world_m_f32":init_world.astype(np.float32),"patch_faces":patch_faces,
  "source_candidate_node_rows":psource,"point_parent_candidate_face_rows":pparent,
  "point_parent_candidate_barycentric_f64":pbary,"face_parent_candidate_rows":face_parent,
  "face_parent_barycentric_f64":face_bary,"face_source_ancestry_indptr":source_csr[0],
  "face_source_ancestry_face_rows":source_csr[1],"face_fine_subface_ancestry_indptr":sub_csr[0],
  "face_fine_subface_ancestry_rows":sub_csr[1],"face_fine_detail_ancestry_indptr":detail_csr[0],
  "face_fine_detail_ancestry_face_rows":detail_csr[1],"patch_outer_boundary_edges":boundary_edges,
  "original_patch_face_rows":source_parent_rows}
 pre_topo=OUT/"pre-fit-centroid-fan-patch.npz"
 np.savez_compressed(pre_topo,**patch_data)
 pre_quality=metrics(patch_data["points_world_m_f32"],patch_faces,face_parent,P,F)
 report["construction"]={"points":int(len(init_world)),"faces":int(len(patch_faces)),
  "source_nodes":int(nsrc),"new_face_barycentric_nodes":int(nparent),
  "fixed_boundary_edges":int(len(boundary_edges)),"fixed_boundary_nodes":int(len(fixed_nodes)),
  "center_barycentric_min":center_bary.min(axis=0).tolist(),"center_barycentric_max":center_bary.max(axis=0).tolist(),
  "pre_fit_float32_shape":pre_quality,"pre_fit_patch_sha256":sha(pre_topo)}
 if not (pre_quality["orientation_positive"] and pre_quality["minimum_unsigned_parent_area_ratio"]>=AREA_FLOOR
         and pre_quality["minimum_angle_degrees"]>=MIN_ANGLE_DEG
         and pre_quality["maximum_aspect_measure"]<=MAX_ASPECT):
  report["status"]="PRIVATE_REJECTED_BASELINE_FAN_QUALITY";raise RuntimeError("new fan topology fails its emitted-F32 baseline shape gates")

 # Load the exact bank surface and construct outward normals from its closed orientation.
 helper=loadmod("bone_aware_fan_bank_helper",HELPER);audit_tool=loadmod("bone_aware_fan_audit_owner",AUDIT_TOOL);exact=audit_tool.exact_module()
 T,ids,gids,bodies,M,off,pos,bankkeys,bankpins,unused=helper.read_bank_inputs(BANKDIR,E)
 bankmask=(ids[:,0]==115)&(ids[:,1]==118)
 bone_raw=np.asarray(T[bankmask],np.float64);bone_stable=np.asarray(ids[bankmask,3],np.int64)
 if not len(bone_raw) or len(np.unique(bone_stable))!=len(bone_stable):raise RuntimeError("bank 115/118 triangle inventory invalid")
 bone_map={int(v):i for i,v in enumerate(bone_stable)}
 bone_normals=[];signed6=0.0;edge_uses=defaultdict(list)
 bone_keys=[]
 for bi,tri in enumerate(bone_raw):
  a,b,c=tri;n=np.cross(b-a,c-a);ln=float(np.linalg.norm(n))
  if not np.isfinite(ln) or ln<=0:raise RuntimeError(f"degenerate bone triangle {bone_stable[bi]}")
  bone_normals.append(n/ln);signed6+=float(a@np.cross(b,c))
  keys=tuple(exact.float32_point_lattice_key(v) for v in tri);bone_keys.append(keys)
  for i,j in ((0,1),(1,2),(2,0)):edge_uses[tuple(sorted((keys[i],keys[j])))].append((bi,keys[i],keys[j]))
 if any(len(v)!=2 for v in edge_uses.values()):raise RuntimeError("bone input edge closure/orientation preflight failed")
 if any(v[0][1:]==v[1][1:] for v in edge_uses.values()):raise RuntimeError("bone input has inconsistent edge orientation")
 if abs(signed6)<1e-18:raise RuntimeError("bone signed volume is unresolved")
 outward=np.asarray(bone_normals)*(1.0 if signed6>0 else -1.0)
 bone_records=audit_tool.records(bone_raw,bone_stable,exact)
 prepared_bone=exact.prepare_point_location(bone_records)
 skin_local_all=(P.astype(np.float64)-pos[115])@M[115].T-off[115]
 init_local=(init_world-pos[115])@M[115].T-off[115]

 # Build affine halfspaces from exact skin/bone intersection points. Intersection points
 # constrain the reconstructed fan surface but are not inserted as patch vertices or edges.
 surf=json.loads(AUDIT159.read_text())
 if surf.get("candidate_sha256")!=sha(C) or not surf.get("complete"):raise RuntimeError("159 report does not match pinned candidate")
 result=next(x for x in surf["banks"] if (int(x["body"]),int(x["stable_bone_id"]))==(115,118))["result"]
 pairs=[(int(f),int(tid)) for f,tid in result.get("triangle_pairs",[])]
 parent_set=set(map(int,source_parent_rows))
 missed=[(f,t) for f,t in pairs if f not in parent_set]
 if missed:raise RuntimeError(f"the fixed 79-face patch does not cover all exact 115/118 pairs; missed={missed[:8]}")
 halfspaces=[];endpoint_count=0
 fan_child_index={int(fi):3*j for j,fi in enumerate(source_parent_rows)}
 def normal_for_hit(bary_bone,bi):
  zeros=[i for i,w in enumerate(bary_bone) if w==0]
  incident={bi}
  if len(zeros)>=2:
   ix=next(i for i,w in enumerate(bary_bone) if w==1);key=bone_keys[bi][ix]
   incident.update(k for k,ks in enumerate(bone_keys) if key in ks)
  elif len(zeros)==1:
   ix=[i for i in range(3) if i!=zeros[0]];edge={bone_keys[bi][ix[0]],bone_keys[bi][ix[1]]}
   incident.update(k for k,ks in enumerate(bone_keys) if edge.issubset(set(ks)))
  nn=np.sum(outward[sorted(incident)],axis=0);ln=float(np.linalg.norm(nn))
  if ln<1e-10:raise RuntimeError("bone corner outward normals cancel at exact witness")
  return nn/ln
 def hit_fan_constraint(face_row,lam):
  idx=fan_child_index[int(face_row)]
  mu=tuple(frac(v) for v in center_bary[idx//3])
  for opp in range(3):
   t=lam[opp]/mu[opp]
   rest=[lam[k]-t*mu[k] for k in range(3) if k!=opp]
   if all(v>=0 for v in rest):
    if opp==2: child=idx+0;nodes=patch_faces[child];weights=(rest[0],rest[1],t)
    elif opp==0: child=idx+1;nodes=patch_faces[child];weights=(rest[0],rest[1],t)
    else: child=idx+2;nodes=patch_faces[child];weights=(rest[0],rest[1],t)
    w=np.asarray([float(v) for v in weights],np.float64)
    if abs(float(w.sum())-1)>1e-8 or np.min(w)<-1e-9:raise RuntimeError("fan barycentric endpoint mapping invalid")
    return nodes,w,int(child)
  raise RuntimeError("exact crossing point cannot be assigned to a barycentric fan child")
 for sf,tid in pairs:
  bi=bone_map.get(tid)
  if bi is None:raise RuntimeError(f"missing stable bone triangle {tid}")
  ts=exact_tri(skin_local_all[F[sf]]);tb=exact_tri(bone_raw[bi])
  hits=sorted(set(exact.triangle_intersection_points(ts,tb)))
  if not hits:raise RuntimeError(f"reported pair has no exact intersection points: {(sf,tid)}")
  for hit in hits:
   lam=exact_barycentric(hit,ts,exact);bb=exact_barycentric(hit,tb,exact)
   if any(v<0 or v>1 for v in lam) or sum(lam)!=1:raise RuntimeError("exact skin hit barycentric is outside parent face")
   nodes,w,child=hit_fan_constraint(sf,lam);n=normal_for_hit(bb,bi)
   q=np.asarray([float(x) for x in hit],np.float64)
   halfspaces.append((nodes,w,q,n,"exact_intersection",int(sf),int(tid),int(child)))
   endpoint_count+=1

 # Any source vertex or fan vertex classified inside/on the bone gets a nearest-face
 # outward plane. Exact intersection point constraints cover the remaining cut surface.
 inside_rows=[]
 for node,p in enumerate(init_local):
  cls=exact.point_location_prepared(tuple(frac(x) for x in p),prepared_bone)["location"]
  if cls in ("inside","boundary"):
   best=np.inf;bestq=None;bestbi=-1
   for bi,tri in enumerate(bone_raw):
    q=closest_point_triangle(p,*tri);d=float(np.linalg.norm(q-p))
    if d<best:best=d;bestq=q;bestbi=bi
   n=outward[bestbi]
   halfspaces.append((np.asarray([node],np.int64),np.asarray([1.0]),bestq,n,"inside_source_or_fan_node",int(node),int(bone_stable[bestbi]),-1))
   inside_rows.append({"node":int(node),"location":cls,"nearest_bone_id":int(bone_stable[bestbi]),"distance_mm":best*1000})
 if not halfspaces:raise RuntimeError("no bone constraints were derived from exact witnesses or inside points")
 fixed_set=set(map(int,fixed_nodes))
 fixed_half=[i for i,h in enumerate(halfspaces) if all(int(n) in fixed_set for n in h[0]) and float(h[3]@(np.sum(init_local[h[0]]*h[1][:,None],axis=0)-h[2]))<CLEARANCE_MM/1000]
 report["bone_constraint_preflight"]={"exact_crossing_pair_count":len(pairs),"exact_intersection_point_count":endpoint_count,
  "inside_or_boundary_vertex_count":len(inside_rows),"halfspace_count":len(halfspaces),"fixed_only_violated_constraints":len(fixed_half),
  "inside_or_boundary_nodes":inside_rows[:64]}
 if fixed_half:raise RuntimeError(f"fixed boundary alone violates {len(fixed_half)} bone halfspaces")

 # Graph-regularized minimum displacement with exact-parent orientation/area constraints.
 edges=set()
 for tri in patch_faces:
  for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):edges.add(tuple(sorted((int(a),int(b)))))
 movable=np.asarray([i for i in range(len(init_local)) if i not in fixed_set],np.int64)
 if not len(movable):raise RuntimeError("fan patch has no movable interior nodes")
 vmap={int(node):j for j,node in enumerate(movable)};nvar=len(movable)
 L=np.zeros((nvar,nvar),np.float64)
 for a,b in edges:
  ia=vmap.get(a);ib=vmap.get(b)
  if ia is not None:L[ia,ia]+=1
  if ib is not None:L[ib,ib]+=1
  if ia is not None and ib is not None:L[ia,ib]-=1;L[ib,ia]-=1
 H=np.eye(nvar)+SMOOTH_WEIGHT*L;H3=np.kron(H,np.eye(3))
 child_parent_local=skin_local_all[F[face_parent]]
 parent_norm=np.cross(child_parent_local[:,1]-child_parent_local[:,0],child_parent_local[:,2]-child_parent_local[:,0])
 parent_len=np.linalg.norm(parent_norm,axis=1)
 if np.any(parent_len==0):raise RuntimeError("degenerate parent source face")
 parent_unit=parent_norm/parent_len[:,None]
 def unpack(y):
  x=init_local.copy();x[movable]+=np.asarray(y).reshape(nvar,3)*SCALE_M;return x
 def objective(y):
  a=np.asarray(y);return float(0.5*a@H3@a)
 def objective_jac(y):return H3@np.asarray(y)
 def half_fun(y):
  x=unpack(y);v=[]
  for nodes,w,q,n,*_ in halfspaces:
   p=np.sum(x[nodes]*w[:,None],axis=0);v.append((float(n@(p-q))-CLEARANCE_MM/1000)/SCALE_M)
  return np.asarray(v,np.float64)
 def half_jac(y):
  J=np.zeros((len(halfspaces),3*nvar),np.float64)
  for r,(nodes,w,q,n,*_) in enumerate(halfspaces):
   for node,weight in zip(nodes,w):
    j=vmap.get(int(node))
    if j is not None:J[r,3*j:3*j+3]+=float(weight)*n
  return J
 def shape_fun(y):
  x=unpack(y);out=np.empty(len(patch_faces),np.float64)
  for i,f in enumerate(patch_faces):
   a,b,c=x[f];cn=np.cross(b-a,c-a)
   out[i]=float(parent_unit[i]@cn)/parent_len[i]-AREA_FLOOR
  return out
 def shape_jac(y):
  x=unpack(y);J=np.zeros((len(patch_faces),3*nvar),np.float64)
  for i,f in enumerate(patch_faces):
   ia,ib,ic=map(int,f);a,b,c=x[ia],x[ib],x[ic];n=parent_unit[i];den=parent_len[i]
   grads=(np.cross(n,c-b),np.cross(c-a,n),np.cross(n,b-a))
   for node,g in zip((ia,ib,ic),grads):
    j=vmap.get(node)
    if j is not None:J[i,3*j:3*j+3]=SCALE_M*g/den
  return J
 y0=np.zeros(3*nvar,np.float64)
 start_shape=shape_fun(y0);start_half=half_fun(y0)
 res=minimize(objective,y0,jac=objective_jac,method="SLSQP",
  constraints=[{"type":"ineq","fun":half_fun,"jac":half_jac},{"type":"ineq","fun":shape_fun,"jac":shape_jac}],
  bounds=[(-2.0,2.0)]*(3*nvar),
  options={"maxiter":MAXITER,"ftol":1e-9,"disp":False})
 fit_local=unpack(res.x);fit_world=((fit_local+off[115])@M[115]+pos[115]).astype(np.float32)
 fit_world[fixed_nodes]=patch_data["points_world_m_f32"][fixed_nodes]
 fit_path=OUT/"private-bone-aware-centroid-fan-patch.npz"
 save={k:v for k,v in patch_data.items() if k!="points_world_m_f32"}
 save.update({"points_world_m_f32":fit_world,"node_displacement_body_local_m_f64":fit_local-init_local,
  "parent_source_face_rows":source_parent_rows,"center_barycentric_f64":center_bary})
 np.savez_compressed(fit_path,**save)
 report["optimizer"]={"success":bool(res.success),"message":str(res.message),"iterations":int(res.nit),
  "objective":float(res.fun),"movable_nodes":int(nvar),"fixed_nodes":int(len(fixed_nodes)),
  "shape_constraint_count":int(len(patch_faces)),"bone_halfspace_count":int(len(halfspaces)),
  "initial_min_shape_constraint":float(start_shape.min()),"final_min_shape_constraint":float(shape_fun(res.x).min()),
  "initial_min_bone_halfspace_scaled":float(start_half.min()),"final_min_bone_halfspace_scaled":float(half_fun(res.x).min()),
  "max_displacement_mm":float(np.max(np.linalg.norm(fit_local-init_local,axis=1))*1000),
  "rms_displacement_mm":float(np.sqrt(np.mean(np.sum((fit_local-init_local)**2,axis=1)))*1000),
  "fit_patch_path":str(fit_path),"fit_patch_sha256":sha(fit_path)}
 # Emitted Float32 shape gate.
 fit_quality=metrics(fit_world,patch_faces,face_parent,P,F)
 fit_shape_pass=bool(fit_quality["orientation_positive"] and fit_quality["minimum_unsigned_parent_area_ratio"]>=AREA_FLOOR
  and fit_quality["minimum_angle_degrees"]>=MIN_ANGLE_DEG and fit_quality["maximum_aspect_measure"]<=MAX_ASPECT)
 report["emitted_f32_shape_gate"]={**fit_quality,"pass":fit_shape_pass}
 report["fixed_boundary"]={"bitwise_preserved":bool(np.array_equal(fit_world[fixed_nodes].view(np.uint32),patch_data["points_world_m_f32"][fixed_nodes].view(np.uint32))),
  "edge_count":int(len(boundary_edges)),"node_count":int(len(fixed_nodes))}
 # Exact local bone and point-location gate.
 emitted_local=(fit_world.astype(np.float64)-pos[115])@M[115].T-off[115]
 fit_patch_records=audit_tool.records(emitted_local[patch_faces],np.arange(len(patch_faces),dtype=np.int64),exact)
 bone_pairs=exact._audit_pair(fit_patch_records,bone_records,same_surface=False)
 loc_counts=Counter(exact.point_location_prepared(tuple(frac(x) for x in p),prepared_bone)["location"] for p in emitted_local)
 report["exact_local_bone_gate"]={"bank":[115,118],"pair_count":int(bone_pairs["count"]),
  "pairs":bone_pairs["triangle_pairs"],"patch_vertex_location_counts":dict(loc_counts),
  "all_vertices_outside":loc_counts.get("inside",0)==0 and loc_counts.get("boundary",0)==0,
  "pass":int(bone_pairs["count"])==0 and loc_counts.get("inside",0)==0 and loc_counts.get("boundary",0)==0}
 # Exact patch internal self and patch versus AABB-overlapping source complement.
 vertex_ids=[int(v) for v in source_vertices]+[int(len(P)+j) for j in range(nparent)]
 patch_self_records=exact_records_mesh(emitted_local,patch_faces,np.arange(len(patch_faces)),vertex_ids,exact)
 self_internal=exact._audit_pair(patch_self_records,patch_self_records,same_surface=True)
 comp_rows=np.asarray([i for i in range(len(F)) if i not in parent_set],np.int64)
 comp_tri=skin_local_all[F[comp_rows]]
 tpatch=emitted_local[patch_faces];lo=tpatch.min(axis=(0,1));hi=tpatch.max(axis=(0,1))
 clo=comp_tri.min(axis=1);chi=comp_tri.max(axis=1)
 active=np.flatnonzero(np.all(chi>=lo,axis=1)&np.all(clo<=hi,axis=1))
 comp_ids=comp_rows[active]
 comp_records=exact_records_mesh(skin_local_all,F[comp_ids],len(patch_faces)+comp_ids,np.arange(len(P),dtype=np.int64),exact)
 self_comp=exact._audit_pair(patch_self_records,comp_records,same_surface=True)
 report["exact_local_self_gate"]={"patch_internal_pair_count":int(self_internal["count"]),
  "patch_internal_pairs":self_internal["triangle_pairs"],"complement_aabb_active_count":int(len(active)),
  "patch_vs_complement_pair_count":int(self_comp["count"]),"patch_vs_complement_pairs":self_comp["triangle_pairs"],
  "pass":int(self_internal["count"])==0 and int(self_comp["count"])==0}
 report["status"]="PRIVATE_BONE_AWARE_CENTROID_FAN_LOCAL_GATES_PASS" if (
  bool(res.success) and fit_shape_pass and report["exact_local_bone_gate"]["pass"] and
  report["exact_local_self_gate"]["pass"] and report["fixed_boundary"]["bitwise_preserved"]
 ) else "PRIVATE_BONE_AWARE_CENTROID_FAN_REJECTED"
 report["accepted_candidate_modified"]=False
except Exception as e:
 report.update({"status":report.get("status") if report.get("status") not in ("STARTED","") else "PRIVATE_BONE_AWARE_CENTROID_FAN_ABORTED",
  "failure":repr(e),"traceback":traceback.format_exc()})
report["elapsed_seconds"]=time.monotonic()-started
report["script_path"]=str(Path(__file__));report["script_sha256"]=sha(Path(__file__))
try:
 report["input_pins"]=report.get("input_pins",{})
 report_path=OUT/"report.json"
 report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
 print(json.dumps({k:v for k,v in report.items() if k not in ("traceback","exact_local_bone_gate","exact_local_self_gate")},indent=2,allow_nan=False))
except Exception:
 print(json.dumps({"report_write_failure":traceback.format_exc(),"report":report},indent=2))
