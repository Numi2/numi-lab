#!/usr/bin/env python3
"""Guarded selective-fan sparse LP for the pinned Human 43/16 patch.
Default invocation is a no-op readiness receipt. A real solve requires both
--run-fit and --slot-granted and must be launched through the shared geometry
slot wrapper. This file has not been used to run a solver or geometry audit.
"""
from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction
import argparse,hashlib,importlib.util,json,os,sys,time,traceback
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
C=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz'
AUDIT159=C.parent/'audit-159/report.json'
PATCH=S/'per-bank-workers-20261027/b43-16/attempt-001/complement-filled-patch-face-rows.npy'
DIAG=S/'per-bank-workers-20261027/b43-16/attempt-001/local-diagnostic.json'
OUT=Path(__file__).resolve().parent
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
AUDIT_TOOL=ROOT/'audit_skin_bone_surfaces_159.py'
BANKDIR=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
BODY=43; STABLE_BONE=16
CLEARANCE_MM=0.05; AREA_FLOOR=0.10; MIN_ANGLE_DEG=0.5; MAX_ASPECT=50.0
TRUST_L1_MM=3.0; SLACK_PENALTY=1000.0; LP_TIME_LIMIT_S=90.0
CAND_PIN='7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af'
AUDIT_PIN='c480d498bc1fb0f361e817b7f785b7b425ba9f40c0e181f7176ae375103cce8e'
DIAG_PIN='adcfeedb5219cb7547b377baa98cbfb83d7ecc64b08b2df1b153bef7ed1aedda'
PATCH_PIN='ad2fb608a536ab8396b781baab4b01f62fc8ae1da81ad533fcb6a4c281131085'
MAPPING_RESIDUAL_TOL_M=1.0e-6


def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def npz(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def frac(x):return Fraction.from_float(float(x))

def exact_tri(tri):return tuple(tuple(frac(x) for x in p) for p in np.asarray(tri).reshape(3,3))

def exact_barycentric(p,tri,exact):
 n=exact._cross(exact._sub(tri[1],tri[0]),exact._sub(tri[2],tri[0]))
 axes=exact._axes(n);a,b,c,q=[tuple(v[k] for k in axes) for v in (tri[0],tri[1],tri[2],p)]
 den=exact._orient2(a,b,c)
 if den==0:raise ValueError('exact parent face is degenerate')
 return (exact._orient2(b,c,q)/den,exact._orient2(c,a,q)/den,exact._orient2(a,b,q)/den)

def closest_point_triangle(p,a,b,c):
 ab=b-a;ac=c-a;ap=p-a;d1=float(ab@ap);d2=float(ac@ap)
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

def ancestry_repeat(indptr,rows,parent_rows):
 outp=[0];outr=[]
 for p in parent_rows:
  vals=rows[int(indptr[int(p)]):int(indptr[int(p)+1])]
  if len(vals)==0:raise ValueError(f'empty source ancestry at candidate face {p}')
  outr.extend(map(int,vals));outp.append(len(outr))
 return np.asarray(outp,np.int64),np.asarray(outr,np.int64)

def disk_boundary(faces):
 uses=Counter();adj=defaultdict(set)
 for f in np.asarray(faces,np.int64):
  for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
   a=int(a);b=int(b);e=tuple(sorted((a,b)));uses[e]+=1;adj[a].add(b);adj[b].add(a)
 if any(v not in (1,2) for v in uses.values()):raise ValueError('patch has nonmanifold edge incidence')
 boundary=sorted(e for e,n in uses.items() if n==1);degree=Counter()
 for a,b in boundary:degree[a]+=1;degree[b]+=1
 seen=set();components=0
 for v in adj:
  if v in seen:continue
  components+=1;todo=[v];seen.add(v)
  while todo:
   q=todo.pop()
   for w in adj[q]:
    if w not in seen:seen.add(w);todo.append(w)
 chi=len(adj)-len(uses)+len(faces)
 topo={'vertices':len(adj),'edges':len(uses),'faces':int(len(faces)),'boundary_edges':len(boundary),
       'boundary_nodes':len(degree),'boundary_degree_histogram':{str(k):int(v) for k,v in Counter(degree.values()).items()},
       'components':components,'euler_characteristic':int(chi),
       'is_single_disk':bool(components==1 and chi==1 and all(v==2 for v in degree.values()))}
 return boundary,topo

def tri_shape(t,parent_tri):
 pn=np.cross(parent_tri[1]-parent_tri[0],parent_tri[2]-parent_tri[0]);pn2=float(pn@pn);plen=np.sqrt(max(pn2,1e-300))
 cn=np.cross(t[1]-t[0],t[2]-t[0]);ratio=float(np.linalg.norm(cn)/plen);orient=float((pn@cn)/max(pn2,1e-300))
 lens=[float(np.linalg.norm(t[(i+1)%3]-t[i])) for i in range(3)]
 area=float(np.linalg.norm(cn)*0.5);aspect=max(lens)**2/(4*np.sqrt(3)*max(area,1e-30))
 aa=[]
 for i in range(3):
  u=t[(i+1)%3]-t[i];v=t[(i+2)%3]-t[i]
  aa.append(float(np.degrees(np.arccos(np.clip((u@v)/max(np.linalg.norm(u)*np.linalg.norm(v),1e-30),-1,1)))))
 return {'area_ratio':ratio,'orientation':orient,'min_angle':min(aa),'aspect':float(aspect)}

def metrics(points,faces,parent_rows,P,F):
 tris=np.asarray(points,np.float32)[faces].astype(np.float64);pars=np.asarray(P,np.float32)[F[parent_rows]].astype(np.float64)
 vals=[tri_shape(t,p) for t,p in zip(tris,pars)]
 return {'face_count':len(vals),'minimum_unsigned_parent_area_ratio':float(min(v['area_ratio'] for v in vals)),
         'minimum_signed_orientation_ratio':float(min(v['orientation'] for v in vals)),
         'minimum_angle_degrees':float(min(v['min_angle'] for v in vals)),
         'maximum_aspect_measure':float(max(v['aspect'] for v in vals)),
         'orientation_positive':bool(all(v['orientation']>0 for v in vals))}

def exact_records_mesh(points,faces,face_ids,vertex_ids,exact):
 pts=[tuple(frac(x) for x in p) for p in np.asarray(points).reshape(-1,3)];out=[]
 for fid,face in zip(face_ids,faces):
  ids=tuple(map(int,face));tri=tuple(pts[i] for i in ids)
  n=exact._cross(exact._sub(tri[1],tri[0]),exact._sub(tri[2],tri[0]))
  if not any(n):raise ValueError(f'exact degenerate emitted triangle {fid}')
  lo=tuple(min(p[k] for p in tri) for k in range(3));hi=tuple(max(p[k] for p in tri) for k in range(3))
  out.append((tri,lo,hi,int(fid),tuple(int(vertex_ids[i]) for i in ids)))
 return out

def run_fit():
 started=time.monotonic();report={'schema':'numi.human.bank43_16.selective-sparse-lp.v1','status':'STARTED',
  'candidate_modified':False,'geometry_or_native_admission':False,'physical_transfer_steps':0,
  'fit_parameters':{'clearance_mm':CLEARANCE_MM,'trust_region_per_node_l1_mm':TRUST_L1_MM,
   'area_floor':AREA_FLOOR,'min_angle_degrees':MIN_ANGLE_DEG,'max_aspect':MAX_ASPECT,
   'bone_slack_penalty_per_mm':SLACK_PENALTY,'linprog_method':'highs','time_limit_seconds':LP_TIME_LIMIT_S},
  'output_directory':str(OUT)}
 try:
  pins={'candidate.npz':sha(C),'audit-159/report.json':sha(AUDIT159),
        'local-diagnostic.json':sha(DIAG),'complement-filled-patch-face-rows.npy':sha(PATCH)}
  expect={'candidate.npz':CAND_PIN,'audit-159/report.json':AUDIT_PIN,
          'local-diagnostic.json':DIAG_PIN,'complement-filled-patch-face-rows.npy':PATCH_PIN}
  if pins!=expect:raise RuntimeError('pinned input SHA mismatch: '+json.dumps(pins,sort_keys=True))
  Z=npz(C);P=np.asarray(Z['points_world_m'],np.float32);F=np.asarray(Z['source_O_faces'],np.int64)
  parent_rows=np.asarray(np.load(PATCH,allow_pickle=False),np.int64);parent_set=set(map(int,parent_rows))
  if len(parent_rows)!=434 or len(parent_set)!=434:raise RuntimeError('selective patch row inventory mismatch')
  diag=json.load(open(DIAG));dpatch=diag['patches'][0]
  crossing_faces=set(map(int,dpatch['crossing_skin_faces']))
  known_inside_source={int(v['node_row']) for v in dpatch['inside_crossing_nodes']}
  surf=json.load(open(AUDIT159))
  if surf.get('candidate_sha256')!=pins['candidate.npz'] or not surf.get('complete'):raise RuntimeError('exact audit-159 not complete/pinned')
  bresult=next(v['result'] for v in surf['banks'] if (int(v['body']),int(v['stable_bone_id']))==(BODY,STABLE_BONE))
  baseline_pairs={(int(f),int(t)) for f,t in bresult.get('triangle_pairs',[])}
  if len(baseline_pairs)!=249 or any(f not in parent_set for f,t in baseline_pairs):raise RuntimeError('exact 43/16 pair set mismatch or uncovered pair')
  source_faces=F[parent_rows];boundary_src,topo=disk_boundary(source_faces)
  if not topo['is_single_disk'] or len(boundary_src)!=54:raise RuntimeError('pinned source patch no longer a 54-edge disk')
  src_i=np.asarray(Z['face_source_ancestry_indptr'],np.int64);src_r=np.asarray(Z['face_source_ancestry_face_rows'],np.int64)
  sub_i=np.asarray(Z['face_fine_subface_ancestry_indptr'],np.int64);sub_r=np.asarray(Z['face_fine_subface_ancestry_rows'],np.int64)
  det_i=np.asarray(Z['face_fine_detail_ancestry_indptr'],np.int64);det_r=np.asarray(Z['face_fine_detail_ancestry_rows'],np.int64)
  source_vertices=np.unique(source_faces.reshape(-1));local_of={int(v):i for i,v in enumerate(source_vertices)}
  anchor_mask=np.asarray(Z['node_source_fidelity_target_mask'],bool)|np.asarray(Z['all_inherited_exit_target_mask'],bool)
  anchor_source_rows=set(map(int,np.flatnonzero(anchor_mask)))
  patch_anchor_rows=anchor_source_rows&set(map(int,source_vertices))
  # Baseline-select fan parents using the same half-incenter/half-centroid F32 construction as preflight.
  centers_bary={};centers_world64={};fan_good=set();fan_bad=set();quality_failures={}
  for fi,tri_ids in zip(parent_rows,source_faces):
   tri=P[tri_ids].astype(np.float64)
   opp=np.asarray([np.linalg.norm(tri[2]-tri[1]),np.linalg.norm(tri[0]-tri[2]),np.linalg.norm(tri[1]-tri[0])])
   if not np.isfinite(opp).all() or opp.sum()<=0:raise RuntimeError(f'invalid parent face {int(fi)}')
   mu=0.5*opp/opp.sum()+0.5/3.0;center=mu@tri;center_f32=center.astype(np.float32).astype(np.float64)
   centers_bary[int(fi)]=mu;centers_world64[int(fi)]=center
   q=[tri_shape(np.asarray([tri[a],tri[b],center_f32]),tri) for a,b in ((0,1),(1,2),(2,0))]
   good=all(v['area_ratio']>=AREA_FLOOR and v['orientation']>0 and v['min_angle']>=MIN_ANGLE_DEG and v['aspect']<=MAX_ASPECT for v in q)
   (fan_good if good else fan_bad).add(int(fi))
   if not good:quality_failures[int(fi)]={'parent_min_angle_degrees':float(min(tri_shape(tri,tri)['min_angle'] for _ in [0])),'fan_min_angle_degrees':float(min(v['min_angle'] for v in q)),'fan_max_aspect':float(max(v['aspect'] for v in q))}
  if (len(fan_good),len(fan_bad))!=(385,49):raise RuntimeError(f'selective fan topology changed good/bad counts {len(fan_good)}/{len(fan_bad)}')
  crossing_good=crossing_faces&fan_good;excluded_pairs=sorted(pair for pair in baseline_pairs if pair[0] in fan_bad)
  target_pairs=sorted(pair for pair in baseline_pairs if pair[0] in fan_good)
  if (len(target_pairs),len(excluded_pairs))!=(233,16):raise RuntimeError('targetable/excluded pair counts changed')
  # Build local source nodes, selected fans, retained excluded parent faces, and full CSR ancestry.
  nsrc=len(source_vertices);center_rows=sorted(fan_good);center_local={fi:nsrc+j for j,fi in enumerate(center_rows)}
  init_world=np.empty((nsrc+len(center_rows),3),np.float64);init_world[:nsrc]=P[source_vertices].astype(np.float64)
  for fi in center_rows:init_world[center_local[fi]]=centers_world64[fi]
  init_world_f32=init_world.astype(np.float32)
  zero_step_source_bitwise=bool(np.array_equal(init_world_f32[:nsrc].view(np.uint32),P[source_vertices].view(np.uint32)))
  if not zero_step_source_bitwise:raise RuntimeError('zero-step source coordinate control is not bitwise identical')
  patch_faces=[];face_parent=[];face_bary=[];fan_children_by_parent={};fan_child_rows=[]
  unit=np.eye(3,dtype=np.float64)
  for fi,tri in zip(parent_rows,source_faces):
   fi=int(fi);a,b,c=(local_of[int(n)] for n in tri)
   if fi in fan_good:
    m=center_local[fi];start=len(patch_faces);mu=centers_bary[fi]
    patch_faces.extend(([a,b,m],[b,c,m],[c,a,m]));face_parent.extend([fi]*3)
    face_bary.extend((np.asarray([unit[0],unit[1],mu]),np.asarray([unit[1],unit[2],mu]),np.asarray([unit[2],unit[0],mu])))
    fan_children_by_parent[fi]=[start,start+1,start+2];fan_child_rows.extend([start,start+1,start+2])
   else:
    start=len(patch_faces);patch_faces.append([a,b,c]);face_parent.append(fi);face_bary.append(unit.copy())
  patch_faces=np.asarray(patch_faces,np.int64);face_parent=np.asarray(face_parent,np.int64);face_bary=np.asarray(face_bary,np.float64);fan_child_rows=np.asarray(fan_child_rows,np.int64)
  if len(patch_faces)!=1204:raise RuntimeError('selective patch emitted-face count mismatch')
  boundary_local=np.asarray([[local_of[a],local_of[b]] for a,b in boundary_src],np.int64)
  mixed_boundary,mixed_topo=disk_boundary(patch_faces)
  if not mixed_topo['is_single_disk'] or set(mixed_boundary)!={tuple(sorted(map(int,e))) for e in boundary_local}:raise RuntimeError('selective mixed patch does not preserve the fixed one-loop disk boundary')
  fixed_boundary=set(map(int,boundary_local.reshape(-1)))
  frozen_source_rows={int(n) for fi in fan_bad for n in F[fi]}|patch_anchor_rows
  frozen_local={local_of[n] for n in frozen_source_rows}
  fixed_source_local=fixed_boundary|frozen_local
  # Source-face and fine ancestry are repeated unchanged for each selected child; bad parents stay one face.
  source_csr=ancestry_repeat(src_i,src_r,face_parent);sub_csr=ancestry_repeat(sub_i,sub_r,face_parent);detail_csr=ancestry_repeat(det_i,det_r,face_parent)
  point_source=np.r_[source_vertices,np.full(len(center_rows),-1,np.int64)]
  point_parent=np.r_[np.full(nsrc,-1,np.int64),np.asarray(center_rows,np.int64)]
  point_bary=np.vstack([np.zeros((nsrc,3),np.float64),np.asarray([centers_bary[f] for f in center_rows])])
  fan_parent_array=np.asarray([face_parent[i] for i in fan_child_rows],np.int64)
  fan_pre=metrics(init_world_f32,patch_faces[fan_child_rows],fan_parent_array,P,F)
  if not (fan_pre['orientation_positive'] and fan_pre['minimum_unsigned_parent_area_ratio']>=AREA_FLOOR and fan_pre['minimum_angle_degrees']>=MIN_ANGLE_DEG and fan_pre['maximum_aspect_measure']<=MAX_ASPECT):raise RuntimeError('selected fan children fail emitted-F32 baseline gates')
  report.update({'input_pins':pins,'baseline_exact_pair_count':len(baseline_pairs),
    'fan_classification':{'good_parent_faces':len(fan_good),'excluded_parent_faces':len(fan_bad),'crossing_pairs_targetable':len(target_pairs),
      'crossing_pairs_excluded_unresolved':len(excluded_pairs),'excluded_pairs':excluded_pairs,'excluded_source_rows':sorted(fan_bad),
      'known_inside_nodes_frozen_by_exclusions':sorted(known_inside_source&frozen_source_rows),
      'known_inside_nodes_frozen_count':len(known_inside_source&frozen_source_rows),
      'source_anchor_rows_in_patch':sorted(patch_anchor_rows),'source_anchor_rows_frozen':True},
    'patch_topology':{'source_parent_disk':topo,'selective_output_disk':mixed_topo,
      'patch_face_count_after_selective_fan':len(patch_faces),'fan_child_count':len(fan_child_rows),
      'preserved_source_ancestry_entries':len(source_csr[1]),'fixed_boundary_edges':len(boundary_local),
      'excluded_source_faces_bitwise_unchanged':True,'zero_step_source_coordinate_bitwise_control':zero_step_source_bitwise,
      'source_anchor_rows_in_patch':sorted(patch_anchor_rows)},
    'baseline_emitted_f32_fan_shape':fan_pre})

  # Load only the pinned 43/16 target surface and exact local predicate implementation.
  helper=loadmod('selective_fan_bank_helper',HELPER);audit_tool=loadmod('selective_fan_audit_owner',AUDIT_TOOL);exact=audit_tool.exact_module()
  T,ids,gids,bodies,M,off,pos,bankkeys,bankpins,unused=helper.read_bank_inputs(BANKDIR,E)
  mask=(ids[:,0]==BODY)&(ids[:,1]==STABLE_BONE);bone_raw=np.asarray(T[mask],np.float64);bone_stable=np.asarray(ids[mask,3],np.int64)
  if len(bone_raw)==0 or len(np.unique(bone_stable))!=len(bone_stable):raise RuntimeError('43/16 bone triangle inventory invalid')
  bone_map={int(v):i for i,v in enumerate(bone_stable)};bone_normals=[];signed6=0.0;bone_edges=defaultdict(list);bone_keys=[]
  for bi,tri in enumerate(bone_raw):
   a,b,c=tri;n=np.cross(b-a,c-a);ln=float(np.linalg.norm(n))
   if not np.isfinite(ln) or ln<=0:raise RuntimeError(f'degenerate bank triangle {bone_stable[bi]}')
   bone_normals.append(n/ln);signed6+=float(a@np.cross(b,c));keys=tuple(exact.float32_point_lattice_key(v) for v in tri);bone_keys.append(keys)
   for i,j in ((0,1),(1,2),(2,0)):bone_edges[tuple(sorted((keys[i],keys[j])))].append((bi,keys[i],keys[j]))
  if any(len(v)!=2 for v in bone_edges.values()) or any(v[0][1:]==v[1][1:] for v in bone_edges.values()) or abs(signed6)<1e-18:raise RuntimeError('bank closed orientation preflight failed')
  outward=np.asarray(bone_normals)*(1.0 if signed6>0 else -1.0)
  bone_records=audit_tool.records(bone_raw,bone_stable,exact);prepared_bone=exact.prepare_point_location(bone_records)
  skin_local_all=(P.astype(np.float64)-pos[BODY])@M[BODY].T-off[BODY]
  init_local=(init_world_f32.astype(np.float64)-pos[BODY])@M[BODY].T-off[BODY]

  def normal_for_hit(bary_bone,bi):
   zeros=[i for i,w in enumerate(bary_bone) if w==0];incident={bi}
   if len(zeros)>=2:
    ix=next(i for i,w in enumerate(bary_bone) if w==1);key=bone_keys[bi][ix]
    incident.update(k for k,ks in enumerate(bone_keys) if key in ks)
   elif len(zeros)==1:
    ix=[i for i in range(3) if i!=zeros[0]];edge={bone_keys[bi][ix[0]],bone_keys[bi][ix[1]]}
    incident.update(k for k,ks in enumerate(bone_keys) if edge.issubset(set(ks)))
   nn=np.sum(outward[sorted(incident)],axis=0);ln=float(np.linalg.norm(nn))
   if ln<1e-10:raise RuntimeError('outward bank normals cancel at witness')
   return nn/ln

  halfspaces=[];mapping_residuals=[];pair_hit_count=0
  def map_hit(face_row,lam,hit):
   children=fan_children_by_parent[int(face_row)];mu=tuple(frac(v) for v in centers_bary[int(face_row)])
   child_slot={0:1,1:2,2:0};edge_order={0:(1,2),1:(2,0),2:(0,1)}
   for k in range(3):
    t=lam[k]/mu[k];rem={j:lam[j]-t*mu[j] for j in range(3) if j!=k}
    order=edge_order[k];r0=rem[order[0]];r1=rem[order[1]]
    if r0>=0 and r1>=0:
     child=children[child_slot[k]];nodes=patch_faces[child]
     weights=(r0,r1,t)
     w=np.asarray([float(v) for v in weights],np.float64)
     if abs(float(w.sum())-1)>1e-8 or np.min(w)<-1e-9:raise RuntimeError('invalid endpoint fan barycentrics')
     q=np.asarray([float(v) for v in hit],np.float64);mapped=np.sum(init_local[nodes]*w[:,None],axis=0)
     residual=float(np.linalg.norm(mapped-q));mapping_residuals.append(residual)
     if residual>MAPPING_RESIDUAL_TOL_M:raise RuntimeError(f'endpoint-to-fan initial residual {residual} m')
     return nodes,w,int(child),q
   raise RuntimeError(f'exact hit does not map to any fan child on parent {face_row}')

  for sf,tid in target_pairs:
   bi=bone_map.get(int(tid))
   if bi is None:raise RuntimeError(f'missing exact bank triangle id {tid}')
   ts=exact_tri(skin_local_all[F[int(sf)]]);tb=exact_tri(bone_raw[bi])
   hits=sorted(set(exact.triangle_intersection_points(ts,tb)))
   if not hits:raise RuntimeError(f'exact audit pair has no reconstructed intersection {sf}/{tid}')
   for hit in hits:
    lam=exact_barycentric(hit,ts,exact);bb=exact_barycentric(hit,tb,exact)
    if any(v<0 or v>1 for v in lam) or sum(lam)!=1:raise RuntimeError('skin exact hit barycentric invalid')
    nodes,w,child,q=map_hit(sf,lam,hit);n=normal_for_hit(bb,bi)
    halfspaces.append({'nodes':nodes,'weights':w,'q':q,'normal':n,'kind':'exact_intersection',
      'source_face':int(sf),'bone_triangle':int(tid),'fan_child':int(child)})
    pair_hit_count+=1

  # Classify every point in the new patch against the target bank. Newly created
  # inside fan centers expand the one-edge-ring active set before the LP is built.
  inside_records=[];inside_center_faces=set();inside_source_rows=[]
  for ni,p0 in enumerate(init_local):
   loc=exact.point_location_prepared(tuple(frac(x) for x in p0),prepared_bone)['location']
   if loc not in ('inside','boundary'):continue
   if ni<nsrc:inside_source_rows.append(int(source_vertices[ni]))
   else:
    fi=int(point_parent[ni]);inside_center_faces.add(fi)
   best=float('inf');bestq=None;bestbi=-1
   for bi,tri in enumerate(bone_raw):
    q0=closest_point_triangle(p0,*tri);dd=float(np.linalg.norm(q0-p0))
    if dd<best:best=dd;bestq=q0;bestbi=bi
   inside_records.append({'node':int(ni),'source_node_row':int(source_vertices[ni]) if ni<nsrc else -1,
     'parent_face_row':int(point_parent[ni]),'location':loc,'nearest_bone_id':int(bone_stable[bestbi]),'distance_mm':1000*best,
     'q':bestq,'normal':outward[bestbi]})

  # Edge adjacency for the selected fan subcomplex.
  edge_uses=defaultdict(list);node_inc=defaultdict(set)
  for fi,tri in zip(parent_rows,source_faces):
   fi=int(fi);a,b,c=map(int,tri)
   for n in (a,b,c):node_inc[n].add(fi)
   for u,v in ((a,b),(b,c),(c,a)):edge_uses[tuple(sorted((u,v)))].append(fi)
  good_adj=defaultdict(set)
  for uses in edge_uses.values():
   if len(uses)==2 and uses[0] in fan_good and uses[1] in fan_good:
    a,b=uses;good_adj[a].add(b);good_adj[b].add(a)
  core=set(crossing_good)
  frozen_source_rows={int(n) for fi in fan_bad for n in F[fi]}|{int(n) for edge in boundary_src for n in edge}|patch_anchor_rows
  for n in inside_source_rows:
   if n not in frozen_source_rows:core.update(node_inc[n]&fan_good)
  core.update(inside_center_faces&fan_good)
  active=set(core)
  for fi in list(core):active.update(good_adj[fi])
  active&=fan_good
  active_source_rows={int(n) for fi in active for n in F[fi]}-frozen_source_rows
  movable_local=set(local_of[n] for n in active_source_rows)
  movable_local.update(center_local[fi] for fi in active)
  movable=np.asarray(sorted(movable_local),np.int64)
  if len(movable)==0:raise RuntimeError('selective region has no movable points')
  vmap={int(node):j for j,node in enumerate(movable)};nmov=len(movable);nd=3*nmov
  # Every excluded parent face and every boundary node is hard frozen.
  excluded_local={local_of[int(n)] for fi in fan_bad for n in F[fi]}
  anchor_local={local_of[n] for n in patch_anchor_rows}
  if set(movable_local)&(set(map(int,boundary_local.reshape(-1)))|excluded_local|anchor_local):raise RuntimeError('fixed boundary/excluded source vertex entered movable set')
  # Every inside/boundary point contributes a plane. Fixed points keep explicit
  # penalized slack, so unresolved frozen witnesses are retained and visible.
  for rec in inside_records:
   ni=rec['node'];halfspaces.append({'nodes':np.asarray([ni],np.int64),'weights':np.asarray([1.0]),'q':np.asarray(rec['q']),
      'normal':np.asarray(rec['normal']),'kind':'inside_or_boundary_point','source_face':int(rec['parent_face_row']),
      'bone_triangle':int(rec['nearest_bone_id']),'fan_child':-1,'source_node_row':int(rec['source_node_row']),
      'location':rec['location'],'distance_mm':float(rec['distance_mm'])})
  # Fixed-only constraints remain soft witnesses; no coordinate, shape or boundary slack is added.
  fixed_violations=[]
  for ri,h in enumerate(halfspaces):
   p0=np.sum(init_local[np.asarray(h['nodes'],np.int64)]*np.asarray(h['weights'])[:,None],axis=0)
   margin=(float(np.asarray(h['normal'])@(p0-np.asarray(h['q'])))*1000.0)-CLEARANCE_MM
   if all(int(n) not in vmap for n in h['nodes']) and margin<0:fixed_violations.append({'row':ri,'kind':h['kind'],'source_face':h['source_face'],'bone_triangle':h['bone_triangle'],'margin_mm':margin})

  # The preflight showed no global complement faces touch this one-ring support;
  # fail closed if that invariant changes, rather than omit old-face protection.
  movable_source_global=[int(source_vertices[int(i)]) for i in movable if int(i)<nsrc]
  patch_rows_set=set(map(int,parent_rows))
  if movable_source_global:
   extmask=np.any(np.isin(F,np.asarray(movable_source_global,np.int64)),axis=1)
   external_old_rows=set(map(int,np.flatnonzero(extmask)))-patch_rows_set
  else:external_old_rows=set()
  if external_old_rows:raise RuntimeError(f'movable nodes touch {len(external_old_rows)} old faces outside the patch; nonworsening model needs extension')
  for fi in fan_bad:
   tri_local=[local_of[int(n)] for n in F[int(fi)]]
   if any(n in vmap for n in tri_local):raise RuntimeError('excluded old face would be affected')

  # Hard linearized projected-area floor on every newly generated child touched
  # by a movable point. Exact emitted Float32 angle/aspect/area are checked later.
  area_rows=[]
  for child in fan_child_rows:
   child=int(child);nodes=patch_faces[child]
   if not any(int(n) in vmap for n in nodes):continue
   par=int(face_parent[child]);tri0=init_local[nodes];ptri=skin_local_all[F[par]]
   pn=np.cross(ptri[1]-ptri[0],ptri[2]-ptri[0]);plen=float(np.linalg.norm(pn))
   if plen<=0:raise RuntimeError(f'degenerate source parent plane {par}')
   n=pn/plen;a,b,c=tri0;cross0=np.cross(b-a,c-a);g0=float(n@cross0)/plen
   grads=(np.cross(n,c-b)/plen,np.cross(c-a,n)/plen,np.cross(n,b-a)/plen)
   area_rows.append({'nodes':nodes,'g0':g0,'grads':grads,'child':child,'source_face':par})

  nbone=len(halfspaces);nabs=nd;nvar=nd+nabs+nbone
  cobj=np.zeros(nvar,np.float64);cobj[nd:nd+nabs]=1.0;cobj[nd+nabs:]=SLACK_PENALTY
  rr=[];cc=[];vv=[];bb=[];row=0
  def addrow(vals,rhs):
   nonlocal row
   for col,val in vals:
    if val!=0.0:rr.append(row);cc.append(int(col));vv.append(float(val))
   bb.append(float(rhs));row+=1
  # |delta_k| <= u_k.
  for k in range(nd):addrow(((k,1.0),(nd+k,-1.0)),0.0);addrow(((k,-1.0),(nd+k,-1.0)),0.0)
  # Per-point L1 trust region <= 3 mm; this implies Euclidean displacement <= 3 mm.
  for j in range(nmov):
   for sx in (-1.0,1.0):
    for sy in (-1.0,1.0):
     for sz in (-1.0,1.0):addrow(((3*j,sx),(3*j+1,sy),(3*j+2,sz)),TRUST_L1_MM)
  initial_margins=[]
  for ri,h in enumerate(halfspaces):
   nodes=np.asarray(h['nodes'],np.int64);weights=np.asarray(h['weights'],np.float64);normal=np.asarray(h['normal'],np.float64);q=np.asarray(h['q'],np.float64)
   p0=np.sum(init_local[nodes]*weights[:,None],axis=0);margin_mm=1000.0*float(normal@(p0-q));initial_margins.append(margin_mm-CLEARANCE_MM)
   vals=[]
   for node,w in zip(nodes,weights):
    j=vmap.get(int(node))
    if j is not None:
     for ax in range(3):vals.append((3*j+ax,-float(w*normal[ax])))
   vals.append((nd+nabs+ri,-1.0))
   addrow(vals,margin_mm-CLEARANCE_MM)
  for ar in area_rows:
   vals=[]
   for node,g in zip(ar['nodes'],ar['grads']):
    j=vmap.get(int(node))
    if j is not None:
     for ax in range(3):vals.append((3*j+ax,-0.001*float(g[ax])))
   addrow(vals,float(ar['g0']-AREA_FLOOR))
  Aub=coo_matrix((vv,(rr,cc)),shape=(row,nvar)).tocsr();bub=np.asarray(bb,np.float64)
  bounds=[(None,None)]*nd+[(0.0,None)]*nabs+[(0.0,None)]*nbone
  report['sparse_lp']={'movable_points':nmov,'coordinate_dofs':nd,'absolute_motion_slacks':nabs,
    'penalized_bone_witness_slacks':nbone,'variables':nvar,'inequality_rows':row,'matrix_nnz':int(Aub.nnz),
    'targetable_exact_pair_count':len(target_pairs),'exact_endpoint_constraint_count':sum(1 for h in halfspaces if h['kind']=='exact_intersection'),
    'inside_or_boundary_plane_count':sum(1 for h in halfspaces if h['kind']=='inside_or_boundary_point'),
    'area_rows':len(area_rows),'trust_rows':8*nmov,'fixed_only_violated_planes':fixed_violations,
    'initial_worst_plane_margin_mm':float(min(initial_margins,default=float('inf'))),
    'slack_penalty_per_mm':SLACK_PENALTY,'hard_constraints':['3 mm per-node L1 displacement trust region','fixed boundary bitwise','excluded source-face vertices bitwise','linearized child projected area ratio >= 0.10']}
  res=linprog(cobj,A_ub=Aub,b_ub=bub,bounds=bounds,method='highs',options={'time_limit':LP_TIME_LIMIT_S,'presolve':True})
  report['lp_result']={'success':bool(res.success),'status':int(res.status),'message':str(res.message),'iterations':int(getattr(res,'nit',-1)),'objective':float(res.fun) if res.fun is not None else None}
  if not res.success:raise RuntimeError(f'bounded sparse LP did not solve: {res.message}')
  delta_mm_full=np.asarray(res.x[:nd]).reshape(nmov,3)
  bone_slack_opt=np.asarray(res.x[nd+nabs:])
  trial_scales=(1.0,0.5,0.25,0.125,0.0625)
  shape_trials=[];selected=None
  for alpha in trial_scales:
   trial_delta_mm=delta_mm_full*alpha
   trial_local_m=trial_delta_mm*0.001
   trial_world=trial_local_m@np.linalg.inv(M[BODY].T)
   trial64=init_world_f32.astype(np.float64);trial64[movable]+=trial_world
   trial_world_f32=trial64.astype(np.float32)
   trial_shape=metrics(trial_world_f32,patch_faces[fan_child_rows],fan_parent_array,P,F)
   trial_pass=bool(trial_shape['orientation_positive'] and trial_shape['minimum_unsigned_parent_area_ratio']>=AREA_FLOOR and trial_shape['minimum_angle_degrees']>=MIN_ANGLE_DEG and trial_shape['maximum_aspect_measure']<=MAX_ASPECT)
   shape_trials.append({'alpha':float(alpha),'pass':trial_pass,'shape':trial_shape})
   if trial_pass:
    selected=(float(alpha),trial_delta_mm,trial_local_m,trial_world,trial_world_f32,trial_shape)
    break
  report['shape_backtracking']={'direction':'single LP optimum direction; scale only, no reoptimization',
   'tested_alphas':[float(x['alpha']) for x in shape_trials],'trials':shape_trials,
   'alpha_selected':None if selected is None else selected[0],
   'exact_bone_or_self_checks_used_to_select_alpha':False}
  if selected is None:
   failed_path=OUT/'shape-rejected-lp-direction.npz'
   np.savez_compressed(failed_path,movable_local_nodes=movable,lp_direction_delta_mm=delta_mm_full,
    bone_slack_opt_mm=bone_slack_opt,trial_alphas=np.asarray(trial_scales,np.float64))
   report['shape_backtracking']['retained_direction_path']=str(failed_path)
   report['shape_backtracking']['retained_direction_sha256']=sha(failed_path)
   raise RuntimeError('all bounded step scales failed emitted Float32 shape gates')
  alpha_selected,delta_mm,delta_local_m,delta_world,emitted,new_shape=selected
  # Recompute slack at the selected emitted step; an alpha backtrack can increase it.
  emitted_local=(emitted.astype(np.float64)-pos[BODY])@M[BODY].T-off[BODY]
  bone_slack=np.asarray([max(0.0,CLEARANCE_MM-1000.0*float(np.asarray(h['normal'])@(np.sum(emitted_local[np.asarray(h['nodes'],np.int64)]*np.asarray(h['weights'])[:,None],axis=0)-np.asarray(h['q'])))) for h in halfspaces],np.float64)
  l1=np.sum(np.abs(delta_mm),axis=1);euclid=np.linalg.norm(delta_mm,axis=1)
  fixed_local=set(range(len(init_world)))-set(map(int,movable))
  if not np.array_equal(emitted[np.asarray(sorted(fixed_boundary),np.int64)].view(np.uint32),init_world_f32[np.asarray(sorted(fixed_boundary),np.int64)].view(np.uint32)):raise RuntimeError('fixed boundary changed in Float32 emission')
  if not np.array_equal(emitted[np.asarray(sorted(frozen_local),np.int64)].view(np.uint32),init_world_f32[np.asarray(sorted(frozen_local),np.int64)].view(np.uint32)):raise RuntimeError('excluded-face source node changed in Float32 emission')
  # Patch output carries one candidate-face ancestry span for each retained or child face.
  patch_save={'points_world_m_f32':emitted,'patch_faces':patch_faces,'point_source_candidate_node_rows':point_source,
    'point_parent_candidate_face_rows':point_parent,'point_parent_candidate_barycentric_f64':point_bary,
    'face_parent_candidate_rows':face_parent,'face_parent_barycentric_f64':face_bary,
    'face_source_ancestry_indptr':source_csr[0],'face_source_ancestry_face_rows':source_csr[1],
    'face_fine_subface_ancestry_indptr':sub_csr[0],'face_fine_subface_ancestry_rows':sub_csr[1],
    'face_fine_detail_ancestry_indptr':detail_csr[0],'face_fine_detail_ancestry_face_rows':detail_csr[1],
    'patch_outer_boundary_edges':boundary_local,'original_patch_face_rows':parent_rows,
    'excluded_source_face_rows':np.asarray(sorted(fan_bad),np.int64),'source_anchor_candidate_node_rows':np.asarray(sorted(patch_anchor_rows),np.int64),'movable_local_node_rows':movable,
    'bone_halfspace_slack_mm':bone_slack,'bone_halfspace_slack_optimum_mm':bone_slack_opt,
    'lp_full_step_delta_body_local_mm_f64':np.pad(delta_mm_full,((0,len(init_world)-nmov),(0,0)))}
  all_delta=np.zeros_like(init_world,dtype=np.float64);all_delta[movable]=delta_local_m;patch_save['node_delta_body_local_m_f64']=all_delta
  out_npz=OUT/'selective-sparse-lp-patch.npz';np.savez_compressed(out_npz,**patch_save)
  report['emission']={'path':str(out_npz),'sha256':sha(out_npz),'shape_backtrack_alpha':float(alpha_selected),'zero_step_source_coordinate_bitwise_control':zero_step_source_bitwise,'float32_emission_rule':'(base_f32 promoted to float64 + delta_m_world) cast once',
    'max_l1_displacement_mm':float(max(l1,default=0.0)),'max_euclidean_displacement_mm':float(max(euclid,default=0.0)),
    'trust_region_pass':bool(max(l1,default=0.0)<=TRUST_L1_MM+1e-7),'fixed_boundary_bitwise':True,'excluded_nodes_bitwise':True,
    'candidate_file_modified':False}
  shape_pass=bool(new_shape['orientation_positive'] and new_shape['minimum_unsigned_parent_area_ratio']>=AREA_FLOOR and new_shape['minimum_angle_degrees']>=MIN_ANGLE_DEG and new_shape['maximum_aspect_measure']<=MAX_ASPECT)
  report['emitted_f32_new_child_shape_gate']={**new_shape,'pass':shape_pass}
  if not shape_pass:raise RuntimeError('emitted Float32 new fan children fail area/angle/aspect gate')
  # All excluded old faces remain exact source coordinates; no outside-patch old face is touched.
  report['affected_old_face_gate']={'excluded_parent_faces_unchanged':True,'outside_patch_old_face_count':len(external_old_rows),'outside_old_faces_nonworsened':len(external_old_rows)==0}

  # Exact local target-bank check on emitted Float32 geometry. Pair IDs are mapped
  # back to candidate source-face ancestry; unresolved exclusions remain explicit.
  emitted_local=(emitted.astype(np.float64)-pos[BODY])@M[BODY].T-off[BODY]
  fit_records=audit_tool.records(emitted_local[patch_faces],np.arange(len(patch_faces),dtype=np.int64),exact)
  exact_bone_pairs=exact._audit_pair(fit_records,bone_records,same_surface=False)
  raw_child_pairs=sorted((int(f),int(t)) for f,t in exact_bone_pairs.get('triangle_pairs',[]))
  mapped_parent_pairs={(int(face_parent[f]),int(t)) for f,t in raw_child_pairs}
  new_pairs=sorted(mapped_parent_pairs-baseline_pairs);remaining=sorted(mapped_parent_pairs)
  unresolved_excluded=sorted(set(excluded_pairs)&mapped_parent_pairs)
  reduction=bool(not new_pairs and len(mapped_parent_pairs)<len(baseline_pairs))
  emitted_point_locations=[]
  for ni,p0 in enumerate(emitted_local):
   loc=exact.point_location_prepared(tuple(frac(x) for x in p0),prepared_bone)['location']
   if loc in ('inside','boundary'):emitted_point_locations.append({'node':int(ni),'source_node_row':int(point_source[ni]),'parent_face_row':int(point_parent[ni]),'location':loc})
  report['exact_local_bone_check']={'raw_emitted_child_bone_pair_count':len(raw_child_pairs),
    'unique_mapped_parent_bone_pair_count':len(mapped_parent_pairs),'baseline_parent_bone_pair_count':len(baseline_pairs),
    'new_parent_bone_pairs':new_pairs,'remaining_parent_bone_pairs':[list(x) for x in remaining],
    'excluded_pairs_remaining':[list(x) for x in unresolved_excluded],'excluded_pair_count_remaining':len(unresolved_excluded),
    'strict_exact_identity_reduction':reduction,'inside_or_boundary_emitted_point_count':len(emitted_point_locations),
    'inside_or_boundary_emitted_points':emitted_point_locations,
    'pass_zero_pairs_and_zero_inside':bool(len(raw_child_pairs)==0 and not emitted_point_locations)}

  # Exact internal patch and patch/complement self screen using the owner predicate.
  vertex_ids=[int(v) for v in source_vertices]+[int(len(P)+j) for j in range(len(center_rows))]
  self_records=exact_records_mesh(emitted_local,patch_faces,np.arange(len(patch_faces)),vertex_ids,exact)
  self_internal=exact._audit_pair(self_records,self_records,same_surface=True)
  comp_rows=np.asarray([i for i in range(len(F)) if i not in parent_set],np.int64)
  comp_tri=skin_local_all[F[comp_rows]];pt=emitted_local[patch_faces];lo=pt.min(axis=(0,1));hi=pt.max(axis=(0,1))
  clo=comp_tri.min(axis=1);chi=comp_tri.max(axis=1);active_comp=np.flatnonzero(np.all(chi>=lo,axis=1)&np.all(clo<=hi,axis=1));comp_ids=comp_rows[active_comp]
  comp_records=exact_records_mesh(skin_local_all,F[comp_ids],len(patch_faces)+comp_ids,np.arange(len(P),dtype=np.int64),exact)
  self_comp=exact._audit_pair(self_records,comp_records,same_surface=True)
  report['exact_local_self_check']={'patch_internal_pair_count':int(self_internal['count']),
    'patch_internal_pairs':self_internal.get('triangle_pairs',[]),'complement_aabb_active_count':int(len(comp_ids)),
    'patch_vs_complement_pair_count':int(self_comp['count']),'patch_vs_complement_pairs':self_comp.get('triangle_pairs',[]),
    'pass_zero_local_self_pairs':bool(int(self_internal['count'])==0 and int(self_comp['count'])==0)}
  report['bone_slack_residuals']=[{'kind':h['kind'],'source_face':h['source_face'],'bone_triangle':h['bone_triangle'],
    'slack_mm':float(bone_slack[i])} for i,h in enumerate(halfspaces) if float(bone_slack[i])>1e-7]
  report['exact_hit_mapping']={'targetable_pair_count':len(target_pairs),'enumerated_intersection_endpoint_count':pair_hit_count,
    'max_initial_endpoint_mapping_residual_m':float(max(mapping_residuals,default=0.0)),'tolerance_m':MAPPING_RESIDUAL_TOL_M}
  report['limitations']=['Excluded/frozen 16 exact bank-pair witnesses and inside source nodes remain unresolved when reported.',
    'This output is a private patch artifact, not a full candidate and not geometry or physical admission.',
    'Only 43/16 exact local bone plus local skin self are checked; no 185-bank no-worsening audit or whole-body self audit is run.',
    'A strict mapped parent-bone identity reduction with no new identities is required to retain as a useful repair trial.']
  if reduction and report['exact_local_self_check']['pass_zero_local_self_pairs']:
   report['status']='PRIVATE_EXACT_IDENTITY_REDUCTION_WITH_UNRESOLVED_WITNESSES' if unresolved_excluded or emitted_point_locations else 'PRIVATE_LOCAL_BONE_AND_SELF_PASS_NOT_ADMISSION'
  else:report['status']='PRIVATE_REJECTED_NO_STRICT_EXACT_IDENTITY_REDUCTION_OR_SELF_REGRESSION'
 except Exception as e:
  report.update({'status':'PRIVATE_ABORTED_OR_REJECTED','failure':repr(e),'traceback':traceback.format_exc()})
 report['elapsed_seconds']=time.monotonic()-started;report['script_path']=str(Path(__file__));report['script_sha256']=sha(Path(__file__))
 report_path=OUT/'fit-report.json';report['report_path']=str(report_path)
 report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('traceback',)},indent=2,allow_nan=False))


def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--run-fit',action='store_true');ap.add_argument('--slot-granted',action='store_true')
 a=ap.parse_args()
 if not (a.run_fit and a.slot_granted):
  print(json.dumps({'status':'PREPARED_NOT_EXECUTED','script':str(Path(__file__).resolve()),
   'required_authorized_command':'python selective_sparse_fit.py --run-fit --slot-granted',
   'heavy_execution_guard':'Both flags are required; launch only through /tmp/numi-human-geometry-slot.py after slot authorization.'},indent=2))
  return 0
 run_fit();return 0

if __name__=='__main__':raise SystemExit(main())
