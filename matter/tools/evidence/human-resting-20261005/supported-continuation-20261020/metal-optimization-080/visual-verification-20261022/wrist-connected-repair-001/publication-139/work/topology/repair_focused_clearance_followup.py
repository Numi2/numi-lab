#!/usr/bin/env python3
"""Bounded source-compatible 2:2 patch flips from crossing/quality evidence.
No coordinate movement or full exact audit; output is a construction candidate only.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path('/Users/n/Work/numi-human-connected-surface-069')
DEFAULT_CANDIDATE=ROOT/'D/connected-quality-patch-538/run-001/candidate.npz'
DEFAULT_AUDIT=ROOT/'D/connected-quality-patch-538/root-exact-run001/audit-159/report.json'
EXPECTED_CANDIDATE_SHA='2f52328e1105dad576717eae872391324a1172ceec3f74f23fae591891078c6e'
EXPECTED_AUDIT_SHA='9103bc0e4f05defa505c2d259e73b90b4a4f95d4e8fb10dd5d18d85f2963b9d8'
REPLAY_534=ROOT/'D/connected-quality-patch-534/replay_surviving_flips_534.py'
REPLAY_534_SHA='4559fd425a3300528f49e5211afbd633674910ecaa0d6671408ee0e9d145e504'
FOCUS_BANK_PAIR_COUNTS={(109,114):516,(115,118):383,(65,99):357,(43,16):355,(124,123):299,(59,95):299}
REPLAY_405=ROOT/'D/local-two-parent-retriangulation-405/cross_seam_search_405.py'
OWNER_402=ROOT/'D/local-two-parent-retriangulation-402/two_parent_patch_flips_402.py'
REPLAY_SHA='e995a2f460e60cb62fb24b80af400678757f62a2b0c1b51141bf0ef387993967'
OWNER_SHA='f68271083bbb97905487f7571ded8264fde1b575759feaf4f5aadf9c42076764'
# A two-face patch may lose at most 0.75 degrees, but cannot emit a sub-2-degree face.
MAX_MIN_ANGLE_LOSS_DEG=0.75
MIN_PATCH_ANGLE_DEG=2.0
MIN_AREA_RATIO=0.10
LOW_AREA_SEED_RATIO=0.15
SHARP_SEED_ANGLE_DEG=5.0
MIN_RATIO_IMPROVEMENT=0.01
MIN_COS_IMPROVEMENT=0.05
MIN_ANGLE_IMPROVEMENT_DEG=0.5
ALLOWED_CHANGED_KEYS={
 'source_O_faces','area_reference_037_source_O_faces',
 'face_source_ancestry_indptr','face_source_ancestry_face_rows',
 'face_fine_subface_ancestry_indptr','face_fine_subface_ancestry_rows',
 'face_fine_subface_ancestry_semantics',
 'face_fine_detail_ancestry_indptr','face_fine_detail_ancestry_rows','face_fine_detail_ancestry_semantics',
 'refinement_face_parent_ancestry_indptr','refinement_face_parent_ancestry_face_rows','refinement_face_parent_ancestry_semantics',
 'refinement_face_parent_face_row','refinement_face_parent_barycentric','refinement_face_parent_barycentric_semantics',
 'parent_coarse_face_row','parent_source_face_marker'
}

def sha_file(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def loadmod(name,path):
 sp=importlib.util.spec_from_file_location(name,path)
 if sp is None or sp.loader is None:raise RuntimeError(f'cannot load {path}')
 m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

def identical(a,b):
 x,y=np.ascontiguousarray(a),np.ascontiguousarray(b)
 return x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()

def csr_rows(ptr,vals,n):
 ptr=np.asarray(ptr);vals=np.asarray(vals)
 if ptr.shape!=(n+1,) or int(ptr[0])!=0 or int(ptr[-1])!=len(vals) or np.any(ptr[1:]<ptr[:-1]):raise ValueError('invalid ancestry CSR')
 return [list(map(int,vals[int(ptr[i]):int(ptr[i+1])])) for i in range(n)]

def face_quality(points_mm,ref_mm,faces,ref_faces):
 f=np.asarray(faces,dtype=np.int64);rf=np.asarray(ref_faces,dtype=np.int64)
 x=np.asarray(points_mm,dtype=np.float64);r=np.asarray(ref_mm,dtype=np.float64)
 c=np.cross(x[f[:,1]]-x[f[:,0]],x[f[:,2]]-x[f[:,0]])
 q=np.cross(r[rf[:,1]]-r[rf[:,0]],r[rf[:,2]]-r[rf[:,0]])
 nc=np.linalg.norm(c,axis=1);nr=np.linalg.norm(q,axis=1)
 cos=np.einsum('ij,ij->i',c,q)/(nc*nr);ratio=nc/nr
 tri=x[f];e0=np.linalg.norm(tri[:,1]-tri[:,0],axis=1);e1=np.linalg.norm(tri[:,2]-tri[:,1],axis=1);e2=np.linalg.norm(tri[:,0]-tri[:,2],axis=1)
 lengths=np.stack([e0,e1,e2],axis=1)
 with np.errstate(invalid='ignore',divide='ignore'):
  a0=np.arccos(np.clip((e2*e2+e0*e0-e1*e1)/(2*e2*e0),-1,1))
  a1=np.arccos(np.clip((e0*e0+e1*e1-e2*e2)/(2*e0*e1),-1,1))
  a2=np.arccos(np.clip((e1*e1+e2*e2-e0*e0)/(2*e1*e2),-1,1))
 angles=np.rad2deg(np.stack([a0,a1,a2],axis=1))
 return {'normal_cos':cos,'area_ratio':ratio,'min_angle_deg':np.min(angles,axis=1),'edge_lengths_mm':lengths}

def crossing_face_rows(report,nfaces):
 if report.get('status')!='COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT' or report.get('complete') is not True or report.get('expected_bank_count')!=185 or report.get('bank_count')!=185:
  raise ValueError('surface report is not a complete exact 185-bank audit')
 out=set()
 for bank in report.get('banks',[]):
  for pair in (bank.get('result') or {}).get('triangle_pairs') or []:
   if len(pair)!=2:raise ValueError('malformed exact pair')
   face=int(pair[0])
   if face<0 or face>=nfaces:raise ValueError('exact pair references invalid face row')
   out.add(face)
 return out

def seed_face_rows(c,quality,crossing):
 ratio=quality['area_ratio'];cos=quality['normal_cos'];ang=quality['min_angle_deg']
 # Negative cosine alone is deliberately insufficient. Use exact crossing witnesses,
 # or a measured near-floor area ratio, or the joint negative-cosine/sharp-angle signal.
 low_area=set(map(int,np.flatnonzero(ratio<LOW_AREA_SEED_RATIO)))
 sharp_fold=set(map(int,np.flatnonzero((cos<=0.0)&(ang<SHARP_SEED_ANGLE_DEG))))
 return set(crossing)|low_area|sharp_fold,low_area,sharp_fold

def topology(faces):
 em=defaultdict(list)
 for fi,t in enumerate(np.asarray(faces,dtype=np.int64)):
  if len(set(map(int,t)))!=3:raise ValueError(f'degenerate face indices row {fi}')
  for a,b in zip(t,np.roll(t,-1)):em[(min(int(a),int(b)),max(int(a),int(b)))].append(int(fi))
 bad={e:rows for e,rows in em.items() if len(rows)>2};boundary={e for e,rows in em.items() if len(rows)==1}
 return em,boundary,bad

def compatible(c,i,j,src,sub,detail):
 return (int(c['face_kind'][i])==int(c['face_kind'][j]) and
         int(c['face_kind'][i])>=0 and
         int(c['parent_fine_detail_face_row'][i])==int(c['parent_fine_detail_face_row'][j])>=0 and
         int(c['parent_fine_subface_row'][i])==int(c['parent_fine_subface_row'][j])>=0 and
         int(c['parent_source_face_marker'][i])==int(c['parent_source_face_marker'][j]) and
         int(c['parent_coarse_face_row'][i])==int(c['parent_coarse_face_row'][j]) and
         src[i]==src[j] and sub[i]==sub[j] and detail[i]==detail[j])

def evaluate_op(c,op,quality_before,points_mm,ref_mm):
 rows=list(map(int,op['face_rows'])); nf=np.asarray(op['new_faces'],dtype=np.int64)
 qc=face_quality(points_mm[None] if False else points_mm,ref_mm,np.asarray(c['source_O_faces']),np.asarray(c['area_reference_037_source_O_faces']))
 before={k:np.asarray(v)[rows] for k,v in qc.items() if k in ('normal_cos','area_ratio','min_angle_deg')}
 afterq=face_quality(points_mm,ref_mm,nf,nf)
 after={k:np.asarray(v) for k,v in afterq.items() if k in ('normal_cos','area_ratio','min_angle_deg')}
 old_ratio=float(before['area_ratio'].min());new_ratio=float(after['area_ratio'].min())
 old_cos=float(before['normal_cos'].min());new_cos=float(after['normal_cos'].min())
 old_ang=float(before['min_angle_deg'].min());new_ang=float(after['min_angle_deg'].min())
 if np.any(after['area_ratio']<MIN_AREA_RATIO):return None,'fixed037_area_floor'
 if np.any(after['normal_cos']<=0):return None,'new_face_not_aligned_to_fixed037'
 if new_ang<MIN_PATCH_ANGLE_DEG or new_ang-old_ang < -MAX_MIN_ANGLE_LOSS_DEG-1e-10:return None,'angle_loss_bound'
 delta=(new_ratio-old_ratio,new_cos-old_cos,new_ang-old_ang)
 if not (delta[0]>=MIN_RATIO_IMPROVEMENT or delta[1]>=MIN_COS_IMPROVEMENT or delta[2]>=MIN_ANGLE_IMPROVEMENT_DEG):return None,'no_measured_quality_gain'
 score=(delta[0],delta[1],delta[2])
 return {'score':score,'before':{'minimum_area_ratio':old_ratio,'minimum_normal_cosine':old_cos,'minimum_angle_deg':old_ang},'after':{'minimum_area_ratio':new_ratio,'minimum_normal_cosine':new_cos,'minimum_angle_deg':new_ang},'deltas':{'area_ratio':delta[0],'normal_cosine':delta[1],'min_angle_deg':delta[2]}},'eligible'

def build_531_reference(base,report,max_flips=24):
 replay=loadmod('cross_seam_search_405_for_531',REPLAY_405);owner=loadmod('two_parent_patch_flips_402_for_531',OWNER_402)
 if sha_file(REPLAY_405)!=REPLAY_SHA or sha_file(OWNER_402)!=OWNER_SHA:raise ValueError('pinned 405/402 owner changed')
 owner.validate_candidate(base)
 F=np.asarray(base['source_O_faces']);RF=np.asarray(base['area_reference_037_source_O_faces']);X=np.asarray(base['points_world_m']).astype(np.float64)*1000;R=np.asarray(base['area_reference_037_points_world_m_f64'])*1000
 if not np.array_equal(F,RF):raise ValueError('current/fixed-037 topology mismatch')
 before_edges,boundary_before,bad=topology(F)
 if bad:raise ValueError('input has nonmanifold edges')
 cross=crossing_face_rows(report,len(F));q=face_quality(X,R,F,RF);seeds,low,sharp=seed_face_rows(base,q,cross)
 src=csr_rows(base['face_source_ancestry_indptr'],base['face_source_ancestry_face_rows'],len(F));sub=csr_rows(base['face_fine_subface_ancestry_indptr'],base['face_fine_subface_ancestry_rows'],len(F));detail=replay.detail_csr(base,set())
 candidate_edges=[]
 crossing_counts=defaultdict(int)
 for bank in report.get('banks',[]):
  for pair in (bank.get('result') or {}).get('triangle_pairs') or []:crossing_counts[int(pair[0])]+=1
 # Owner's oriented edge incidence is required by candidate_flip.
 owner_em=owner.edge_map(F)
 for edge,inc in owner_em.items():
  rows=sorted(int(v[0]) for v in inc)
  if len(rows)!=2 or not (set(rows)&seeds):continue
  i,j=rows
  if not compatible(base,i,j,src,sub,detail):continue
  op,why=owner.candidate_flip(base,edge,inc,{int(base['parent_fine_detail_face_row'][i])},set(),-MAX_MIN_ANGLE_LOSS_DEG,0.0,owner_em)
  if op is None:continue
  qop,why2=evaluate_op(base,op,q,X,R)
  if qop is None:continue
  if any(r in low for r in op['face_rows']):
   priority=(0,float(min(q['area_ratio'][list(op['face_rows'])])))
  elif any(r in sharp for r in op['face_rows']):
   priority=(1,float(min(q['min_angle_deg'][list(op['face_rows'])])))
  else:
   priority=(2,-float(sum(crossing_counts.get(int(r),0) for r in op['face_rows'])))
  qop['priority_class']=priority[0]
  qop['priority_measure']=priority[1]
  candidate_edges.append((priority,qop['score'],edge,op,qop,why))
 # Lowest-area floor risks first, then negative sharp folds, then exact crossing density.
 # Within a class, prefer measured quality improvement; deterministic edge tie break.
 candidate_edges.sort(key=lambda x:(x[0][0],x[0][1],-x[1][0],-x[1][1],-x[1][2],x[2],x[3]['face_rows']))
 out={k:np.array(v,copy=True) for k,v in base.items()};selected=[];reserved_nodes=set();reserved_faces=set();overlap=0;rejected_stale=0
 for priority,score,edge,op,qual,owner_status in candidate_edges:
  if len(selected)>=max_flips:break
  rows=list(map(int,op['face_rows']));quad=set(map(int,np.asarray(out['source_O_faces'])[rows].ravel()))
  if set(rows)&reserved_faces or quad&reserved_nodes:
   overlap+=1;continue
  current_em=owner.edge_map(out['source_O_faces']);inc=current_em.get(edge,[])
  if sorted(int(v[0]) for v in inc)!=sorted(rows) or not np.array_equal(out['source_O_faces'][rows],np.asarray([base['source_O_faces'][r] for r in rows])):
   rejected_stale+=1;continue
  op2,why=owner.candidate_flip(out,edge,inc,{int(out['parent_fine_detail_face_row'][rows[0]])},set(),-MAX_MIN_ANGLE_LOSS_DEG,0.0,current_em)
  if op2 is None or op2['new_faces']!=op['new_faces']:
   rejected_stale+=1;continue
  out=replay.apply_op(out,op2)
  selected.append({'edge':list(edge),'face_rows':rows,'old_faces':np.asarray(base['source_O_faces'])[rows].tolist(),'new_faces':op2['new_faces'],'source_detail':op2['source_detail_face_row'],'source_subface':op2['source_subface_row'],'material_kind':int(base['face_kind'][rows[0]]),'source_ancestry_union':sorted(set(src[rows[0]])), 'subface_ancestry_union':sorted(set(sub[rows[0]])),'immediate_parent_union':op2['immediate_parent_union'],'quality':qual,'owner_reason':owner_status})
  reserved_nodes|=quad;reserved_faces|=set(rows)

 after_edges,boundary_after,bad_after=topology(out['source_O_faces'])
 if bad_after or boundary_after!=boundary_before:raise ValueError('selected batch changed manifold status or open boundary')
 if len({tuple(sorted(map(int,t))) for t in out['source_O_faces']})!=len(out['source_O_faces']):raise ValueError('selected batch introduced duplicate triangle')
 if not identical(out['points_world_m'],base['points_world_m']):raise ValueError('coordinate array changed')
 if not np.array_equal(out['source_O_faces'],out['area_reference_037_source_O_faces']):raise ValueError('matched fixed-037 connectivity not synchronized')
 changed=[k for k in base if k not in out or not identical(base[k],out[k])];added=sorted(set(out)-set(base))
 if set(changed)-ALLOWED_CHANGED_KEYS or set(added)-ALLOWED_CHANGED_KEYS:raise ValueError(f'unexpected noncoordinate array changes: {sorted(set(changed)-ALLOWED_CHANGED_KEYS)} {sorted(set(added)-ALLOWED_CHANGED_KEYS)}')
 qafter=face_quality(X,R,out['source_O_faces'],out['area_reference_037_source_O_faces'])
 return out,{
  'schema':'numi.human.connected-quality-patch-531.v1','status':'LOCAL_CONSTRUCTION_CANDIDATE_PENDING_FULL_AUDIT',
  'geometry_admitted':False,'native_admission':False,'accepted_physical_steps':0,
  'candidate_seed_count':len(seeds),'exact_crossing_face_count':len(cross),'low_area_seed_count':len(low),'negative_cosine_sharp_seed_count':len(sharp),
  'input_negative_fixed037_cosine_count':int(np.sum(q['normal_cos']<=0)),'output_negative_fixed037_cosine_count':int(np.sum(qafter['normal_cos']<=0)),
  'input_faces_with_area_ratio_below_0_15':int(np.sum(q['area_ratio']<LOW_AREA_SEED_RATIO)),'output_faces_with_area_ratio_below_0_15':int(np.sum(qafter['area_ratio']<LOW_AREA_SEED_RATIO)),
  'input_min_area_ratio':float(q['area_ratio'].min()),'output_min_area_ratio':float(qafter['area_ratio'].min()),
  'candidate_edge_count_after_material_and_seed_filter':len(candidate_edges),'selected_flip_count':len(selected),'selected_flips':selected,
  'candidate_proposal_overlap_skips':overlap,'stale_or_rejected_after_prior_flips':rejected_stale,
  'candidate_proposal_face_rows':[{'face_rows':list(map(int,op['face_rows'])),'edge':list(edge),'priority_class':int(qop['priority_class']),'priority_measure':float(qop['priority_measure']),'quality_deltas':qop['deltas']} for priority,score,edge,op,qop,reason in candidate_edges],
  'coordinates_bitwise_unchanged':True,'outer_boundary_edge_set_unchanged':True,'nonmanifold_edges_after':0,'matched_037_face_connectivity_synchronized':True,
  'changed_face_rows':sorted(reserved_faces),'changed_candidate_arrays':changed,'added_candidate_arrays':added,
  'eligibility_policy':{'seed':'current complete exact 159 face witness OR fixed-037 area ratio<0.15 OR cosine<=0 with min angle<5 degrees; cosine alone never seeds a flip','source_compatibility':'same face_kind, singular source detail, singular subface, source marker, coarse parent and identical source/detail/subface CSR rows; immediate parent may differ and is recorded as multi-parent CSR union','owner':'402 local fixed-coordinate flip guard plus 405 ancestry replay owner','min_angle_loss_bound_deg':MAX_MIN_ANGLE_LOSS_DEG,'post_flip_min_angle_deg':MIN_PATCH_ANGLE_DEG,'post_flip_area_ratio_floor':MIN_AREA_RATIO},
  'input_candidate_path':'provided by CLI','input_candidate_sha256':'provided by CLI','input_surface_report_path':'provided by CLI','input_surface_report_sha256':'provided by CLI',
  'owner_405_sha256':sha_file(REPLAY_405),'owner_402_sha256':sha_file(OWNER_402),
  'exact_full_self_surface_and_185_bank_audits':'NOT_RUN_PENDING_ROOT_SLOT',
  'scope_limit':'Candidate-only topology patch; exact self, full 185-bank bone, layer transfer, volume and native gates are not evaluated.'}

def focused_crossing_rows(report,nfaces):
 if report.get('status')!='COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT' or report.get('complete') is not True or report.get('expected_bank_count')!=185 or report.get('bank_count')!=185: raise ValueError('incomplete exact 185-bank report')
 requested=set(FOCUS_BANK_PAIR_COUNTS); seen=set(); rows=set(); counts={}
 banks=report.get('banks')
 if not isinstance(banks,list): raise ValueError('missing bank table')
 for bank in banks:
  key=(int(bank.get('body',-1)),int(bank.get('stable_bone_id',-1)))
  if key not in requested: continue
  if key in seen: raise ValueError('duplicate focused bank')
  seen.add(key); result=bank.get('result') or {}; pairs=result.get('triangle_pairs')
  if not isinstance(pairs,list): raise ValueError('focused bank lacks exact triangle-pair rows')
  count=int(result.get('count',len(pairs)))
  if count!=len(pairs): raise ValueError(f'focused report count mismatch for {key}')
  counts[key]=count
  for pair in pairs:
   if not isinstance(pair,(list,tuple)) or len(pair)!=2: raise ValueError('malformed focused pair')
   row=int(pair[0])
   if row<0 or row>=nfaces: raise ValueError('focused skin face row out of range')
   rows.add(row)
 if seen!=requested: raise ValueError(f'missing requested typed focus bank IDs: {sorted(requested-seen)}')
 return rows,counts

def edge_one_ring(faces,seeds):
 em=defaultdict(list)
 for i,t in enumerate(np.asarray(faces,dtype=np.int64)):
  for a,b in zip(t,np.roll(t,-1)): em[(min(int(a),int(b)),max(int(a),int(b)))].append(i)
 scope=set(seeds)
 for rs in em.values():
  if len(rs)>2: raise ValueError('nonmanifold edge')
  if len(rs)==2 and (rs[0] in seeds or rs[1] in seeds): scope.update(rs)
 return scope

def local_eval(op,q,x,r):
 rows=list(map(int,op['face_rows'])); nf=np.asarray(op['new_faces'],dtype=np.int64)
 old={k:np.asarray(q[k])[rows] for k in ('area_ratio','normal_cos','min_angle_deg')}
 nq=face_quality(x,r,nf,nf); new={k:np.asarray(nq[k]) for k in old}
 orat,nrat=float(old['area_ratio'].min()),float(new['area_ratio'].min())
 oc,nc=float(old['normal_cos'].min()),float(new['normal_cos'].min())
 oa,na=float(old['min_angle_deg'].min()),float(new['min_angle_deg'].min())
 if nrat<MIN_AREA_RATIO or nc<=0 or na<MIN_PATCH_ANGLE_DEG or na-oa < -MAX_MIN_ANGLE_LOSS_DEG-1e-10:return None
 d=(nrat-orat,nc-oc,na-oa)
 if not (d[0]>=MIN_RATIO_IMPROVEMENT or d[1]>=MIN_COS_IMPROVEMENT or d[2]>=MIN_ANGLE_IMPROVEMENT_DEG):return None
 return {'score':d,'before':{'minimum_area_ratio':orat,'minimum_normal_cosine':oc,'minimum_angle_deg':oa},'after':{'minimum_area_ratio':nrat,'minimum_normal_cosine':nc,'minimum_angle_deg':na},'deltas':{'area_ratio':d[0],'normal_cosine':d[1],'min_angle_deg':d[2]}}

def build(base,report,max_flips=24):
 replay=loadmod('cross_seam_search_405_for_541',REPLAY_405)
 replay534=loadmod('replay_surviving_flips_534_for_541',REPLAY_534)
 owner=loadmod('two_parent_patch_flips_402_for_541',OWNER_402)
 if sha_file(REPLAY_405)!=REPLAY_SHA or sha_file(OWNER_402)!=OWNER_SHA or sha_file(REPLAY_534)!=REPLAY_534_SHA: raise ValueError('pinned topology owner changed')
 owner.validate_candidate(base)
 F=np.asarray(base['source_O_faces']); RF=np.asarray(base['area_reference_037_source_O_faces'])
 X=np.asarray(base['points_world_m'],dtype=np.float64)*1000; R=np.asarray(base['area_reference_037_points_world_m_f64'],dtype=np.float64)*1000
 if not np.array_equal(F,RF): raise ValueError('current/fixed-037 topology mismatch')
 _,boundary0,bad=topology(F)
 if bad: raise ValueError('input has nonmanifold edges')
 focus,counts=focused_crossing_rows(report,len(F)); scope=edge_one_ring(F,focus)
 q=face_quality(X,R,F,RF)
 low=set(map(int,np.flatnonzero(q['area_ratio']<LOW_AREA_SEED_RATIO))) & scope
 sharp=set(map(int,np.flatnonzero((q['normal_cos']<=0)&(q['min_angle_deg']<SHARP_SEED_ANGLE_DEG)))) & scope
 quality_seeds=low|sharp
 src=csr_rows(base['face_source_ancestry_indptr'],base['face_source_ancestry_face_rows'],len(F))
 sub=csr_rows(base['face_fine_subface_ancestry_indptr'],base['face_fine_subface_ancestry_rows'],len(F))
 det=replay.detail_csr(base,set()); em=owner.edge_map(F); candidates=[]; cross_count=defaultdict(int)
 for bank in report['banks']:
  key=(int(bank.get('body',-1)),int(bank.get('stable_bone_id',-1)))
  if key in FOCUS_BANK_PAIR_COUNTS:
   for pair in (bank.get('result') or {}).get('triangle_pairs') or []: cross_count[int(pair[0])]+=1
 for edge,inc in em.items():
  rows=sorted(int(v[0]) for v in inc)
  if len(rows)!=2 or not set(rows)<=scope or not ((set(rows)&focus) or (set(rows)&quality_seeds)): continue
  i,j=rows
  if not compatible(base,i,j,src,sub,det): continue
  op,why=owner.candidate_flip(base,edge,inc,{int(base['parent_fine_detail_face_row'][i])},set(),-MAX_MIN_ANGLE_LOSS_DEG,0.0,em)
  if op is None: continue
  qop=local_eval(op,q,X,R)
  if qop is None: continue
  if any(r in low for r in rows): priority=(0,float(min(q['area_ratio'][rows])))
  elif any(r in sharp for r in rows): priority=(1,float(min(q['min_angle_deg'][rows])))
  else: priority=(2,-float(sum(cross_count.get(r,0) for r in rows)))
  candidates.append((priority,qop['score'],edge,op,qop,why))
 candidates.sort(key=lambda z:(z[0][0],z[0][1],-z[1][0],-z[1][1],-z[1][2],z[2],z[3]['face_rows']))
 selected=[]; used_faces=set(); used_nodes=set(); overlap=0
 for priority,score,edge,op,qual,why in candidates:
  if max_flips>0 and len(selected)>=max_flips: break
  rows=list(map(int,op['face_rows'])); nodes=set(map(int,F[rows].ravel()))
  if not set(rows)<=scope: raise ValueError('operation escaped declared scope')
  if set(rows)&used_faces or nodes&used_nodes: overlap+=1; continue
  selected.append({'edge':list(edge),'face_rows':rows,'old_faces':F[rows].tolist(),'new_faces':op['new_faces'],'source_detail':op['source_detail_face_row'],'source_subface':op['source_subface_row'],'material_kind':int(base['face_kind'][rows[0]]),'source_ancestry_union':sorted(set(src[rows[0]])|set(src[rows[1]])),'subface_ancestry_union':sorted(set(sub[rows[0]])|set(sub[rows[1]])),'immediate_parent_union':op['immediate_parent_union'],'quality':qual,'owner_reason':why})
  used_faces.update(rows);used_nodes.update(nodes)
 out=replay534.apply_ops_batch(base,selected,replay)
 _,boundary1,bad1=topology(out['source_O_faces'])
 if bad1 or boundary1!=boundary0: raise ValueError('boundary/manifold invariant failed')
 if not identical(base['points_world_m'],out['points_world_m']): raise ValueError('point coordinates changed')
 if not np.array_equal(out['source_O_faces'],out['area_reference_037_source_O_faces']): raise ValueError('fixed037 topology mismatch')
 changed=set(map(int,np.flatnonzero(np.any(F!=out['source_O_faces'],axis=1))))
 expected={r for op in selected for r in op['face_rows']}
 if changed!=expected or not changed<=scope: raise ValueError('changed rows escaped focus scope')
 outside=np.ones(len(F),dtype=bool); outside[list(changed)]=False
 for k,a in base.items():
  b=out[k]; aa=np.asarray(a); bb=np.asarray(b)
  if aa.ndim and aa.shape[0]==len(F) and bb.shape==aa.shape and not np.array_equal(aa[outside],bb[outside],equal_nan=True): raise ValueError(f'face row changed outside patch: {k}')
 for ptr,val in [('face_source_ancestry_indptr','face_source_ancestry_face_rows'),('face_fine_subface_ancestry_indptr','face_fine_subface_ancestry_rows'),('face_fine_detail_ancestry_indptr','face_fine_detail_ancestry_rows'),('refinement_face_parent_ancestry_indptr','refinement_face_parent_ancestry_face_rows')]:
  if ptr not in base or ptr not in out: continue
  for i in np.flatnonzero(outside):
   a0,a1=map(int,base[ptr][i:i+2]); b0,b1=map(int,out[ptr][i:i+2])
   if not np.array_equal(base[val][a0:a1],out[val][b0:b1]): raise ValueError(f'ancestry row changed outside patch: {i}')
 anchors=np.asarray(base['typed_coarse_anchor_output_rows_018'],dtype=np.int64)
 if anchors.size!=237 or not identical(base['points_world_m'][anchors],out['points_world_m'][anchors]): raise ValueError('typed anchor invariant failed')
 changedkeys=[k for k in base if not identical(base[k],out[k])]; added=sorted(set(out)-set(base))
 if set(changedkeys)-ALLOWED_CHANGED_KEYS or set(added)-ALLOWED_CHANGED_KEYS: raise ValueError('undeclared candidate array changed')
 return out,{'schema':'numi.human.focused-clearance-topology-541.v1','status':'FOCUSED_LOCAL_CONSTRUCTION_CANDIDATE_PENDING_ROOT_EXACT_AUDIT','geometry_admitted':False,'native_admission':False,'accepted_physical_steps':0,'focus_body_stable_bone_pairs':[{'body':b,'stable_bone_id':s,'exact_pair_count':counts[(b,s)]} for b,s in sorted(FOCUS_BANK_PAIR_COUNTS)],'focus_exact_pair_count':int(sum(counts.values())),'focus_witness_face_count':len(focus),'declared_edge_one_ring_face_count':len(scope),'quality_defect_seed_count':len(quality_seeds),'candidate_edge_count':len(candidates),'selected_flip_count':len(selected),'selected_flips':selected,'overlap_skips':overlap,'changed_face_rows':sorted(changed),'changed_candidate_arrays':changedkeys,'added_candidate_arrays':added,'coordinates_bitwise_unchanged':True,'typed_anchor_count':237,'typed_anchor_coordinates_bitwise_unchanged':True,'outside_focus_faces_bitwise_unchanged':True,'outside_focus_ancestry_rows_unchanged':True,'boundary_edge_set_unchanged':True,'nonmanifold_edges_after':0,'matched_037_face_connectivity_synchronized':True,'replay_405_sha256':sha_file(REPLAY_405),'replay_534_sha256':sha_file(REPLAY_534),'owner_402_sha256':sha_file(OWNER_402),'exact_self_surface_and_185_bank_audits':'NOT_RUN_PENDING_ROOT_SLOT','area_and_edge_gates':'NOT_RECOMPUTED_BY_THIS_BUILDER','scope_rule':'six exact body/stable-bone banks plus one edge-ring; every flip touches a focus witness or a halo quality defect and both faces remain in scope','scope_limit':'fixed-coordinate local topology candidate only; full exact and native gates remain pending'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',default=str(DEFAULT_CANDIDATE));ap.add_argument('--surface-report',default=str(DEFAULT_AUDIT));ap.add_argument('--output-dir',required=True);ap.add_argument('--max-flips',type=int,default=24)
 a=ap.parse_args();cp=Path(a.candidate);rp=Path(a.surface_report);od=Path(a.output_dir)
 if a.max_flips<0:raise ValueError('max-flips must be nonnegative')
 with np.load(cp,allow_pickle=False) as z:base={k:z[k] for k in z.files}
 report=json.loads(rp.read_text())
 if report.get('candidate_sha256')!=sha_file(cp):raise ValueError('surface audit candidate SHA does not match input')
 out,meta=build(base,report,a.max_flips)
 if od.exists():raise FileExistsError(od)
 od.mkdir(parents=True)
 dest=od/'candidate.npz';np.savez_compressed(dest,**out)
 meta.update({'input_candidate_path':str(cp),'input_candidate_sha256':sha_file(cp),'input_surface_report_path':str(rp),'input_surface_report_sha256':sha_file(rp),'candidate_output_path':str(dest),'candidate_output_sha256':sha_file(dest)})
 rpout=od/'report.json';rpout.write_text(json.dumps(meta,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({**meta,'report_path':str(rpout),'report_sha256':sha_file(rpout)},indent=2,sort_keys=True,allow_nan=False))
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--candidate',required=True);ap.add_argument('--surface-report',required=True)
 ap.add_argument('--candidate-sha256',required=True);ap.add_argument('--surface-report-sha256',required=True)
 ap.add_argument('--output-dir',required=True);ap.add_argument('--max-flips',type=int,default=0)
 a=ap.parse_args();cp=Path(a.candidate);rp=Path(a.surface_report);od=Path(a.output_dir);t0=time.perf_counter()
 if a.max_flips<0:raise ValueError('max-flips must be zero (unlimited) or positive')
 for path,want,label in ((cp,a.candidate_sha256,'candidate'),(rp,a.surface_report_sha256,'surface report')):
  if not path.is_file():raise FileNotFoundError(path)
  if sha_file(path)!=want:raise ValueError(f'{label} SHA mismatch')
 with np.load(cp,allow_pickle=False) as z:base={k:z[k] for k in z.files}
 report=json.loads(rp.read_text())
 if report.get('candidate_sha256')!=a.candidate_sha256:raise ValueError('complete 185-bank audit is not bound to candidate')
 if report.get('status')!='COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT' or report.get('complete') is not True or report.get('expected_bank_count')!=185 or report.get('bank_count')!=185:raise ValueError('input must be a complete exact 185-bank report')
 focused_crossing_rows(report,len(base['source_O_faces']))
 out,meta=build(base,report,a.max_flips)
 if od.exists():raise FileExistsError(od)
 od.mkdir(parents=True);dest=od/'candidate.npz';np.savez_compressed(dest,**out)
 meta.update({'input_candidate_path':str(cp),'input_candidate_sha256':a.candidate_sha256,'input_surface_report_path':str(rp),'input_surface_report_sha256':a.surface_report_sha256,'candidate_output_path':str(dest),'candidate_output_sha256':sha_file(dest),'builder_source_path':str(Path(__file__)),'builder_source_sha256':sha_file(__file__),'generation_elapsed_seconds':time.perf_counter()-t0,'max_flips_argument':int(a.max_flips),'zero_limit_means_unlimited':True})
 rpout=od/'report.json';rpout.write_text(json.dumps(meta,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'report_path':str(rpout),'report_sha256':sha_file(rpout),**meta},indent=2,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
