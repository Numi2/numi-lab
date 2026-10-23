#!/usr/bin/env python3
"""Generate bounded, explicitly new inferred retargeting proposals (never headers)."""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
OUT=E/'capitate-attachment-correspondence-001/source-slot-provenance-002'
C=E/'joint-bmap-refinement-candidate-011-mechanical119-integration-001/source-C-reference-offset-diagnostic-014-exact-f32-residual-001'
REPORT=E/'capitate-attachment-correspondence-001/capitate-attachment-correspondence.json'
SLOT=OUT/'source-slot-provenance-audit.json'
REG=E/'native-pose-checks/connected-hand-registration069i/registered-skin.csv'
PAY=E/'finger-profile-069i-left-ring-web-retreat/replacement-bilateral-hands038.nhskin'
BIND=E/'arm-skin-weight-audit.json'
sys.path.insert(0,'/Users/n/numi-human-anatomy-finish-1378/src')
from numilab_human.skin_source_payload_preflight import decode_payload

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def closest_point_to_all_triangles(p,T):
 a,b,c=T[:,0],T[:,1],T[:,2]
 e0=b-a; e1=c-a; v=p-a
 d00=np.einsum('ij,ij->i',e0,e0); d01=np.einsum('ij,ij->i',e0,e1); d11=np.einsum('ij,ij->i',e1,e1)
 d20=np.einsum('ij,ij->i',v,e0); d21=np.einsum('ij,ij->i',v,e1)
 den=d00*d11-d01*d01; ok=den>1e-24
 x=np.divide(d11*d20-d01*d21,den,out=np.zeros_like(den),where=ok)
 y=np.divide(d00*d21-d01*d20,den,out=np.zeros_like(den),where=ok)
 z=1-x-y; B=np.stack([z,x,y],1); Q=a+x[:,None]*e0+y[:,None]*e1
 d2=np.einsum('ij,ij->i',Q-p,Q-p)
 inside=ok&(z>=-1e-10)&(x>=-1e-10)&(y>=-1e-10)
 d2=np.where(inside,d2,np.inf); B=np.where(inside[:,None],B,np.nan)
 for i,j in ((0,1),(1,2),(2,0)):
  e=T[:,j]-T[:,i]; dd=np.einsum('ij,ij->i',e,e)
  t=np.clip(np.divide(np.einsum('ij,ij->i',p-T[:,i],e),dd,out=np.zeros_like(dd),where=dd>1e-24),0,1)
  q=T[:,i]+t[:,None]*e; d=np.einsum('ij,ij->i',q-p,q-p)
  b2=np.zeros_like(B); b2[:,i]=1-t; b2[:,j]=t
  take=d<d2; d2=np.where(take,d,d2); B=np.where(take[:,None],b2,B)
 return np.sqrt(d2),B

def family(name,side):
 suffix='_r' if side=='right' else '_l'
 return name.endswith(suffix) and ('mc_' in name or any(k in name for k in ('capitate','scaphoid','lunate','triquetrum','pisiform','hamate','trapezium','trapezoid')))

base=json.loads(REPORT.read_text()); slot=json.loads(SLOT.read_text())
name_by_body={int(x['i']):x['name'] for x in json.loads(BIND.read_text())['bindings']}
with REG.open(newline='') as f: csv_rows=[{**r,'vertex':int(r['vertex']),'donor':int(r['donor'])} for r in csv.DictReader(f)]
rows_by_donor={}
for r in csv_rows: rows_by_donor.setdefault(r['donor'],[]).append(r)
payload=decode_payload(PAY.read_bytes()); binding_ids=np.asarray(payload['bindings_u'][:,0],np.int64)
with np.load(C/'coarse-surface.npz',allow_pickle=False) as z:
 F=z['faces'].astype(np.int64); face_parent=z['source_face_parent'].astype(np.int64); face_kind=z['source_face_kind'].astype(int)
 inferred=z['source_face_parent_inferred_mask'].astype(bool); ap=z['source_face_parent_ancestry_indptr']; ai=z['source_face_parent_ancestry_ids']
with np.load(C/'candidate-layers-C014.npz',allow_pickle=False) as z:
 D=z['deep_world'].astype(float); outward=z['outward_normals'].astype(float)
with np.load(C/'coarse-to-source-C014.npz',allow_pickle=False) as z:
 W=z['expanded_source_weights86'].astype(float)
T=D[F]; area2=np.linalg.norm(np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]),axis=1)
node_owner=np.argmax(W,axis=1)
owner_names=np.asarray([[name_by_body[int(binding_ids[node_owner[v]])] for v in tri] for tri in F],dtype=object)
valid=(area2>1e-12)&(~inferred)&((ap[1:]-ap[:-1])==1)
records=[]; proposals=[]; local_limit_m=0.002
for s in base['sites']:
 donor=int(s['donor_body']); donor_name=name_by_body[donor]; side='right' if donor_name.endswith('_r') else 'left'
 col=np.flatnonzero(binding_ids==donor)
 if len(col)!=1: raise ValueError(f'nonunique binding for donor {donor}')
 col=int(col[0]); p=np.asarray(s['target_world_in_069i_frame_m'],float)
 distances,bary=closest_point_to_all_triangles(p,T)
 side_palm=np.asarray([all(family(str(n),side) for n in names) for names in owner_names])
 eligible=valid&side_palm
 cap_weight=np.einsum('fi,fi->f',bary,W[F,col])
 anyface=np.flatnonzero(eligible)
 positive=eligible&(cap_weight>0.0)
 def describe(mask):
  ix=np.flatnonzero(mask)
  if not len(ix): return None
  j=int(ix[np.argmin(distances[ix])]); q=np.einsum('i,ij->j',bary[j],T[j])
  triN=np.cross(T[j,1]-T[j,0],T[j,2]-T[j,0]); triN/=np.linalg.norm(triN)
  avg=outward[F[j]].mean(axis=0); avg/=np.linalg.norm(avg)
  return {'face':j,'nodes':F[j].tolist(),'barycentric':bary[j].tolist(),'point_world_m':q.tolist(),
   'distance_from_frozen_target_m':float(distances[j]),'face_region_label_diagnostic_only':str(np.asarray(np.load(C/'coarse-surface.npz',allow_pickle=False)['face_region'])[j]),
   'corner_dominant_bone_names':owner_names[j].tolist(),'interpolated_frozen_capitate_weight':float(cap_weight[j]),
   'corner_frozen_capitate_weights':W[F[j],col].tolist(),'source_face_parent':int(face_parent[j]),
   'source_face_kind':int(face_kind[j]),'source_parent_inferred':bool(inferred[j]),
   'source_parent_ancestry_ids':ai[ap[j]:ap[j+1]].astype(int).tolist(),
   'triangle_unit_normal':triN.tolist(),'mean_authored_outward_normal':avg.tolist(),
   'triangle_normal_dot_authored_outward':float(triN@avg),
   'area_m2':float(area2[j]*0.5)}
 nearest_hand=describe(eligible)
 nearest_cap=describe(positive)
 rec={'legacy_site_index':int(s['legacy_site_index']),'legacy_reduced_vertex':int(s['legacy_reduced_vertex']),
  'historical_source_vertex_id_not_used_as_current_identity':int(s['source_vertex']),
  'migrates_from_stable_id':s['stable_id'],'frozen_donor_body':donor,'frozen_donor_name':donor_name,
  'frozen_target_world_m':p.tolist(),'original_kA_N_per_m':float(s['original_kA_N_per_m']),
  'original_tributary_area_m2':float(s['baseline_tributary_area_m2']),
  'nearest_same_side_carpal_palm_exact_ancestry':nearest_hand,
  'nearest_same_side_carpal_palm_with_positive_capitate_weight':nearest_cap,
  'proposal_status':'unresolved_no_proposal'}
 # Bounded proposal criteria: exact source parent, same-side carpal/palm face,
 # positive capitate contribution, close migration, and consistent outward orientation.
 if nearest_cap and nearest_cap['distance_from_frozen_target_m']<=local_limit_m and nearest_cap['triangle_normal_dot_authored_outward']>=0.90:
  candidate=nearest_cap; q=np.asarray(candidate['point_world_m'],float)
  donor_rows=rows_by_donor[donor]
  M=[]; tvec=[]
  for row in donor_rows:
   m=np.array([[float(row[f'm{i}{j}']) for j in range(3)] for i in range(3)])
   w=np.array([float(row[k]) for k in ('wx','wy','wz')]); l=np.array([float(row[k]) for k in ('lx','ly','lz')])
   M.append(m); tvec.append(l-m@w)
  M=np.asarray(M); tvec=np.asarray(tvec); M0=np.median(M,axis=0); t0=np.median(tvec,axis=0)
  frame_spread={'rows':int(len(M)),'M_spread':float(np.max(np.abs(M-M[0]))),'t_spread_m':float(np.max(np.abs(tvec-tvec[0])))}
  newlocal=M0@q+t0; recovered=np.linalg.solve(M0,newlocal-t0); frame_recon=float(np.linalg.norm(recovered-q))
  oldname=s['stable_id']; newname=oldname+':inferred-retarget:C014-face-'+str(candidate['face'])
  proposal={'proposal_type':'new_inferred_retargeting_not_identity_recovery','proposal_stable_id':newname,
    'migrates_from_stable_id':oldname,'migrates_from_source_vertex_id_historical_only':int(s['source_vertex']),
    'source_identity_recovery_claim':False,'donor_body':donor,'donor_name':donor_name,
    'old_zero_strain_target_world_m':p.tolist(),'old_donor_local_target_m':s['original_donor_local_target_m'],'new_zero_strain_target_world_m':q.tolist(),
    'new_donor_local_target_m':newlocal.tolist(),'target_shift_m':float(np.linalg.norm(q-p)),
    'donor_local_roundtrip_error_m':frame_recon,'donor_frame_consistency':frame_spread,'frame_convention':'CSV local = M @ world + t; new target local is M @ barycentric_D_world + t','original_kA_N_per_m_preserved':float(s['original_kA_N_per_m']),
    'original_tributary_area_m2_preserved':float(s['baseline_tributary_area_m2']),
    'surface_binding':{'face':candidate['face'],'nodes':candidate['nodes'],'barycentric':candidate['barycentric'],
      'source_face_parent':candidate['source_face_parent'],'source_face_kind':candidate['source_face_kind'],
      'source_parent_ancestry_ids':candidate['source_parent_ancestry_ids'],'source_parent_inferred':False},
    'anatomical_witness':{'same_side':side,'corner_dominant_bone_names':candidate['corner_dominant_bone_names'],
      'region_interpretation':'metacarpal palm; based on named current corner owners, not the stale face_region string',
      'interpolated_capitate_weight':candidate['interpolated_frozen_capitate_weight'],
      'triangle_unit_normal':candidate['triangle_unit_normal'],
      'mean_authored_outward_normal':candidate['mean_authored_outward_normal'],
      'normal_dot':candidate['triangle_normal_dot_authored_outward']},
    'status':'proposal_only_requires_compiler_owner_review'}
  rec['proposal_status']='proposal_only_requires_compiler_owner_review'; rec['proposal']=proposal; proposals.append(proposal)
 elif nearest_cap is None:
  rec['proposal_reason']='No exact-ancestry same-side carpal/palm triangle with positive capitate contribution was found.'
 elif nearest_cap['distance_from_frozen_target_m']>local_limit_m:
  rec['proposal_reason']='Nearest exact-ancestry same-side carpal/palm triangle with positive capitate contribution exceeds the declared 2 mm locality screen.'
 else:
  rec['proposal_reason']='Candidate triangle failed the authored outward-normal orientation screen.'
 records.append(rec)

report={'schema':'numi.human.capitate.inferred-retargeting-proposals.v1',
 'status':'new inferred reference proposals only; no compiler/header emission or admission',
 'inputs':{str(p):sha(p) for p in (REPORT,SLOT,REG,PAY,BIND,C/'coarse-surface.npz',C/'candidate-layers-C014.npz',C/'coarse-to-source-C014.npz')},
 'method':{'surface':'current candidate011/C014 deep boundary D','candidate_face_rule':'all three corners have dominant named same-side carpal or metacarpal bone owner; face source parent is exact/non-inferred and has one ancestry parent; interpolated frozen capitate weight is positive',
  'locality_screen_m':local_limit_m,'normal_orientation_screen_dot_min':0.90,
  'locality_screen_is':'proposal-selection heuristic only, not a physical admission gate',
  'detailed_face_mapping':'Candidate face index, nodes, barycentric coordinates, source_face_parent/kind, ancestry, and exact/inferred flag are recorded. No raw reassigned source slot is used as a current identity.'},
 'proposal_count':len(proposals),'proposals':proposals,'site_diagnostics':records,
 'unchanged_original_evidence':'The original six-site correspondence report remains unchanged and unresolved (0/6 identity-preserving candidates). These are separately named new inferred retargeting proposals, not recovered original attachments.',
 'limits':['No attachment record was written to an application header.','No physical or solver admission is claimed.','Compiler owner must verify stable-ID migration semantics and active C014 surface/barycentric mapping before any integration.','The frozen kA and area are copied unchanged; no stiffness, area, density, or donor substitutions are made.']}
out=OUT/'inferred-capitate-retargeting-proposals.json'; out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
back=json.loads(out.read_text())
assert back['proposal_count']==len(proposals)
for p in proposals:
 b=np.asarray(p['surface_binding']['barycentric']); assert np.all(np.isfinite(b)) and abs(float(b.sum())-1)<1e-12 and b.min()>=-1e-10
 assert p['source_identity_recovery_claim'] is False and p['status']=='proposal_only_requires_compiler_owner_review'
 assert p['donor_local_roundtrip_error_m']<1e-8
print(json.dumps({'output':str(out),'output_sha256':sha(out),'proposal_count':len(proposals),'proposal_sites':[p['migrates_from_stable_id'] for p in proposals],
 'sites':[{'site':r['legacy_site_index'],'status':r['proposal_status'],'nearest_exact_same_side_m':None if r['nearest_same_side_carpal_palm_exact_ancestry'] is None else r['nearest_same_side_carpal_palm_exact_ancestry']['distance_from_frozen_target_m'],
 'nearest_capitate_influenced_m':None if r['nearest_same_side_carpal_palm_with_positive_capitate_weight'] is None else r['nearest_same_side_carpal_palm_with_positive_capitate_weight']['distance_from_frozen_target_m'],
 'capitate_weight':None if r['nearest_same_side_carpal_palm_with_positive_capitate_weight'] is None else r['nearest_same_side_carpal_palm_with_positive_capitate_weight']['interpolated_frozen_capitate_weight']} for r in records]},indent=2))
