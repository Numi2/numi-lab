#!/usr/bin/env python3
"""Build the selective 109/114 fan topology and DOF receipt; no fit or audit."""
from pathlib import Path
from collections import defaultdict
from fractions import Fraction
import hashlib,json
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
D=S/'per-bank-workers-20261027/b109-114/attempt-004-bone-aware-centroid-fan-001'
C=Path('/Users/numi/Work/numi-human-connected-surface-069/D/publication-182-stage-001/combined-653/candidate.npz')
Q=D/'fan-source-patch.npz';W=D/'actual-crossing-witness-map.json';R=D/'fan-quality-diagnostic.json'
OUT=D/'selective-fit-preflight.json';PATCH=D/'selective-pre-fit-patch.npz';EX=D/'selective-excluded-witnesses.json'
BASE='7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af'
GRAPH='524e1c62a7768b5ec9ae50a2b7bc5ac65bd2303ff9d03355486bf784cfc932ab'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(points,faces,parents,P,F):
 tri=points[faces].astype(np.float64);par=P[F[parents]].astype(np.float64)
 pn=np.cross(par[:,1]-par[:,0],par[:,2]-par[:,0]);cn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
 den=np.einsum('ij,ij->i',pn,pn);ori=np.einsum('ij,ij->i',pn,cn)/den;ratio=np.linalg.norm(cn,axis=1)/np.sqrt(den)
 ang=[];asp=[]
 for t in tri:
  lens=[np.linalg.norm(t[(i+1)%3]-t[i]) for i in range(3)];area=np.linalg.norm(np.cross(t[1]-t[0],t[2]-t[0]))/2
  asp.append(float(max(lens)**2/(4*np.sqrt(3)*max(area,1e-300))))
  vals=[]
  for i in range(3):
   u=t[(i+1)%3]-t[i];v=t[(i+2)%3]-t[i]
   vals.append(float(np.degrees(np.arccos(np.clip((u@v)/max(np.linalg.norm(u)*np.linalg.norm(v),1e-300),-1,1)))))
  ang.append(min(vals))
 return {'faces':len(faces),'min_parent_area_ratio':float(ratio.min()),'min_orientation_ratio':float(ori.min()),'all_orientation_positive':bool(np.all(ori>0)),'min_angle_deg':min(ang),'max_aspect':max(asp)}
def main():
 if sha(C)!=BASE:raise RuntimeError('candidate pin mismatch')
 if sha(W)!='a81eef5507def4d62f36ec507f01f1cbfff4d5f0ec96627a8b780dcebe676d35':raise RuntimeError('witness map pin mismatch')
 with np.load(C,allow_pickle=False) as z:
  P=z['points_world_m'].astype(np.float32);F=z['source_O_faces'].astype(np.int64)
  source_ind=z['face_source_ancestry_indptr'].astype(np.int64);source_rows=z['face_source_ancestry_face_rows'].astype(np.int64)
  sub_ind=z['face_fine_subface_ancestry_indptr'].astype(np.int64);sub_rows=z['face_fine_subface_ancestry_rows'].astype(np.int64)
  detail_ind=z['face_fine_detail_ancestry_indptr'].astype(np.int64);detail_rows=z['face_fine_detail_ancestry_rows'].astype(np.int64)
 S0=np.load(Q,allow_pickle=False);G=json.loads(W.read_text());diag=json.loads(R.read_text())
 comps=[S0['component_parent_face_rows_00'].astype(int),S0['component_parent_face_rows_01'].astype(int)]
 parent=S0['original_patch_face_rows'].astype(int);sv=S0['source_candidate_node_rows'].astype(int);sv=sv[sv>=0]
 if len(parent)!=277 or [len(x) for x in comps]!=[239,38] or len(sv)!=172:raise RuntimeError('source patch inventory changed')
 outerloc=np.unique(S0['patch_outer_boundary_edges'].astype(int));boundary={int(sv[i]) for i in outerloc}
 viol={int(x['parent_face_row']):x for x in diag['fan_shape_violating_parents']}
 unsafe=sorted(fi for fi,x in viol.items() if x['crossing_parent'])
 if unsafe!=[34552,34553,34554,34555,34705,34706,34707,34708]:raise RuntimeError('crossing fan unsafe-face set changed')
 frozen=set(boundary)
 for fi in unsafe:frozen.update(map(int,F[fi]))
 free=set(map(int,sv))-frozen
 by=defaultdict(list)
 for x in G['witnesses']:by[int(x['skin_face_row'])].append(x)
 crossing=sorted(by);shape_unsafe=set(viol);shape_safe=sorted(set(crossing)-shape_unsafe)
 blocked={};selected=[]
 for fi in shape_safe:
  bad=[]
  for x in by[fi]:
   lam=[Fraction(v) for v in x['skin_barycentrics']]
   center=Fraction(x['fan_child_weights'][2])
   support=any(v>0 and int(node) in free for node,v in zip(F[fi],lam)) or center>0
   if not support:bad.append(x)
  if bad:blocked[fi]=bad
  else:selected.append(fi)
 if len(selected)!=95 or len(blocked)!=8:raise RuntimeError(f'expected 95 fully supported safe faces, selected={len(selected)}, blocked={len(blocked)}')
 selected=set(selected);selected_w=[x for fi in sorted(selected) for x in by[fi]]
 local={int(v):i for i,v in enumerate(sv)};points=[P[v].copy() for v in sv];faces=[];parents=[];barys=[];new_nodes=[];eye=np.eye(3)
 for fi0 in parent:
  fi=int(fi0);a,b,c=map(int,F[fi]);tri=P[[a,b,c]].astype(np.float64)
  if fi in selected:
   le=np.asarray([np.linalg.norm(tri[2]-tri[1]),np.linalg.norm(tri[0]-tri[2]),np.linalg.norm(tri[1]-tri[0])]);mu=.5*le/le.sum()+np.ones(3)/6
   node=len(points);points.append((mu@tri).astype(np.float32));new_nodes.append((node,fi,mu.tolist()))
   faces.extend(([local[a],local[b],node],[local[b],local[c],node],[local[c],local[a],node]))
   barys.extend((np.asarray([eye[0],eye[1],mu]),np.asarray([eye[1],eye[2],mu]),np.asarray([eye[2],eye[0],mu])))
   parents.extend((fi,fi,fi))
  else:faces.append((local[a],local[b],local[c]));barys.append(eye.copy());parents.append(fi)
 points=np.asarray(points,np.float32);faces=np.asarray(faces,np.int64);parents=np.asarray(parents,np.int64);barys=np.asarray(barys,np.float64)
 fan_mask=np.isin(parents,np.asarray(sorted(selected),np.int64))
 fan_quality=metrics(points,faces[fan_mask],parents[fan_mask],P,F)
 unrefined=np.asarray([fi for fi in parent if int(fi) not in selected],np.int64)
 baseline_unrefined=metrics(P,F[unrefined],unrefined,P,F)
 quality=metrics(points,faces,parents,P,F);csr={}
 for name,ind,rows in (('source',source_ind,source_rows),('subface',sub_ind,sub_rows),('detail',detail_ind,detail_rows)):
  ptr=[0];out=[]
  for fi in parents:out.extend(map(int,rows[ind[fi]:ind[fi+1]]));ptr.append(len(out))
  csr[name]=(np.asarray(ptr,np.int64),np.asarray(out,np.int64))
 node_rows=np.r_[sv,np.full(len(points)-len(sv),-1,np.int64)]
 node_parent=np.r_[np.full(len(sv),-1,np.int64),np.asarray([x[1] for x in new_nodes],np.int64)]
 node_bary=np.vstack([np.zeros((len(sv),3)),np.asarray([x[2] for x in new_nodes],np.float64)])
 boundary_local=S0['patch_outer_boundary_edges'].astype(np.int64)
 np.savez_compressed(PATCH,points_world_m_f32=points,patch_faces=faces,face_parent_candidate_rows=parents,face_parent_barycentric_f64=barys,
  node_source_candidate_rows=node_rows,node_parent_candidate_face_rows=node_parent,node_parent_candidate_barycentric_f64=node_bary,
  patch_outer_boundary_edges=boundary_local,original_patch_face_rows=parent,selected_crossing_parent_face_rows=np.asarray(sorted(selected),np.int64),
  fixed_boundary_source_node_rows=np.asarray(sorted(boundary),np.int64),frozen_unsafe_parent_vertex_rows=np.asarray(sorted(frozen-boundary),np.int64),
  face_source_ancestry_indptr=csr['source'][0],face_source_ancestry_face_rows=csr['source'][1],
  face_fine_subface_ancestry_indptr=csr['subface'][0],face_fine_subface_ancestry_rows=csr['subface'][1],
  face_fine_detail_ancestry_indptr=csr['detail'][0],face_fine_detail_ancestry_rows=csr['detail'][1])
 excluded=[]
 for fi in crossing:
  if fi in selected:continue
  reason='fan_shape_gate' if fi in viol else 'fixed_support'
  bad={(int(x['pair_index']),int(x['root']),int(x['endpoint_record_id'])) for x in blocked.get(fi,[])}
  for x in by[fi]:
   key=(int(x['pair_index']),int(x['root']),int(x['endpoint_record_id']))
   excluded.append({'parent_face_row':fi,'reason':reason,'fixed_only_support':key in bad,'crossing':True,'witness':x})
 EX.write_text(json.dumps({'schema':'numi.human.b109_114.selective_fan_excluded_witnesses.v1','excluded':excluded},indent=2,sort_keys=True)+'\n')
 affected=[int(fi) for fi in parent if any(int(v) in free for v in F[int(fi)])]
 report={'schema':'numi.human.b109_114.selective_fit_preflight.v1','candidate_sha256':sha(C),'source_fan_sha256':sha(Q),'witness_map_sha256':sha(W),'shape_diagnostic_sha256':sha(R),
  'method':'Float32 incenter/centroid fans only on shape-safe crossing parents whose endpoints have nonfrozen support; other local faces remain unrefined. Freeze vertices of the eight unsafe crossing parents plus fixed outer boundary. Noncrossing incident source faces may move and are included in no-deterioration checks.',
  'source':{'component_face_counts':[len(x) for x in comps],'parent_faces':len(parent),'source_vertices':len(sv),'outer_boundary_edges':len(boundary_local),'outer_boundary_vertices':len(boundary)},
  'selection':{'crossing_parent_faces':len(crossing),'shape_unsafe_crossing_faces':unsafe,'shape_safe_crossing_faces':len(shape_safe),'fixed_support_blocked_safe_faces':sorted(blocked),'selected_fully_supported_crossing_faces':len(selected),'selected_face_rows':sorted(selected),'selected_endpoint_witnesses':len(selected_w),'excluded_witnesses':len(excluded),'excluded_witness_file':str(EX),'excluded_witness_sha256':sha(EX)},
  'degrees_of_freedom':{'frozen_source_nodes':len(frozen),'frozen_boundary_nodes':len(boundary),'frozen_unsafe_parent_node_count':len(frozen-boundary),'movable_source_nodes':len(free),'source_faces_affected_by_movable_nodes':len(affected),'affected_parent_face_rows':affected,'generated_fan_centers':len(selected)},
  'topology':{'components_remain_disconnected':True,'output_local_node_count':len(points),'output_local_face_count':len(faces),'fan_parent_faces':len(selected),'unrefined_parent_faces':len(parent)-len(selected),'fixed_boundary_edges':len(boundary_local),'typed_parent_barycentric_source_subface_detail_ancestry':True,'patch_path':str(PATCH),'patch_sha256':sha(PATCH)},
  'pre_fit_mixed_patch_shape':quality,'newly_generated_fan_shape':fan_quality,'unrefined_faces_baseline_shape':baseline_unrefined,'gates':{'source_boundary_fixed_by_construction':True,'fit_run':False,'geometry_audit_run':False,'slot_acquired':False,'physical_steps':0}}
 OUT.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'report':str(OUT),'report_sha256':sha(OUT),'patch':str(PATCH),'patch_sha256':sha(PATCH),'excluded':str(EX),'excluded_sha256':sha(EX),'selection':report['selection'],'degrees_of_freedom':{k:v for k,v in report['degrees_of_freedom'].items() if not k.endswith('rows')},'pre_fit_mixed_patch_shape':quality,'newly_generated_fan_shape':fan_quality,'unrefined_faces_baseline_shape':baseline_unrefined},indent=2))
if __name__=='__main__':main()
