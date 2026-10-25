from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np

BASE=Path('/Users/n/Work/numi-human-connected-surface-069/connected-volume-extraction-157/true-o-foundation-159')
FOUNDATION=BASE/'run-003-conversion/true-o-tetgen-foundation.npz'
RUN004=BASE/'run-004-registered-fields/registered-levelsets.npz'
CHECKPOINT=BASE/'run-013-conforming-field-refinement/refined-foundation-checkpoint.npz'
FIELDS=BASE/'run-013-conforming-field-refinement/registered-levelsets.npz'
DAG=BASE/'run-013-conforming-field-refinement/point-interpolation-dag.npz'
OUT=BASE/'run-029-closed-volume-field-repair/run-002'
CLOSED_REPORT=BASE/'run-028-closed-volume-sign/run-002/report.json'
OUTER=Path('/Users/n/Work/numi-human-connected-surface-069/full-connected-clearance-150/run-004/pass-003/candidate.npz')
EXPECTED={
 'foundation':'f4f50a725d51b0dd5d0433c6078e6913c64fe0e9cf4ea676aa0f01fe8ce76ddb',
 'run004':'8530e57f83de88862da6f3c258b0ee68befe9b83096316556b808d42750e2e81',
 'checkpoint':'cf51ebb33721a351be149f822f49f0737af136e66eb95a39924f10bbe6e3f179',
 'fields':'67ad75b444382ba864f244dc57b1ac073664379a409963eb6813972fa6091769',
 'dag':'f5b5385e14c26219bba54c189b3a784264cf356f36e8c3e6c1a151d520e68c76',
 'closed_report':'9675dc5a3736ea4b417183aae49e454441b2ef0e91c6756a1be0fc5576765e23',
 'outer':'d2beb7c9cd71ae7969d140b7783bfc5d4254e2ab8c51f77f9974d7d6e9e056ab',
}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=False)
 paths={'foundation':FOUNDATION,'run004':RUN004,'checkpoint':CHECKPOINT,'fields':FIELDS,'dag':DAG,'closed_report':CLOSED_REPORT,'outer':OUTER}
 pins={k:sha(v) for k,v in paths.items()}
 if pins!=EXPECTED: raise RuntimeError(f'input pin mismatch: {pins}')
 certificate=json.loads(CLOSED_REPORT.read_text())
 if certificate.get('negative_phi_outer_rows')!=670 or not certificate.get('all_negative_rows_inside_closed_volume') or certificate.get('exact_location_counts')!={'boundary':0,'inside':670,'outside':0}:
  raise RuntimeError('closed-volume exact membership certificate does not cover all negative samples')
 with np.load(FOUNDATION,allow_pickle=False) as z: V={k:z[k] for k in z.files}
 with np.load(RUN004,allow_pickle=False) as z: B={k:z[k] for k in z.files}
 with np.load(CHECKPOINT,allow_pickle=False) as z: C={k:z[k] for k in z.files}
 with np.load(FIELDS,allow_pickle=False) as z: F={k:z[k] for k in z.files}
 with np.load(DAG,allow_pickle=False) as z: D={k:z[k] for k in z.files}
 with np.load(OUTER,allow_pickle=False) as z: O={k:z[k] for k in z.files}
 Q=np.asarray(C['points_world_m'],dtype=np.float32)
 lev=np.asarray(F['registered_levelsets_m'],dtype=np.float64)
 if not np.array_equal(lev,np.asarray(C['registered_levelsets_m'],dtype=np.float64)):
  raise RuntimeError('field NPZ/checkpoint registered values differ')
 neg=np.flatnonzero(lev[:,0]<0)
 if len(neg)!=670: raise RuntimeError('negative row set changed')
 OP=np.asarray(O['points_world_m'],dtype=np.float32).astype(np.float64)
 OF=np.asarray(O['source_O_faces'],dtype=np.int64)[:,[0,2,1]]
 base_face=np.asarray(B['nearest_outer_face_row'],dtype=np.int64)
 base_bary=np.asarray(B['nearest_outer_barycentric'],dtype=np.float64)
 base_d2=np.asarray(B['outer_distance_squared_m2'],dtype=np.float64)
 newrows=np.asarray(D['new_node_rows'],dtype=np.int64)
 newface=np.asarray(D['nearest_outer_face_rows'],dtype=np.int64)
 newbary=np.asarray(D['nearest_outer_barycentrics'],dtype=np.float64)
 newh=np.asarray(D['sampled_skin_fat_depths_m'],dtype=np.float64)
 new_lookup={int(r):i for i,r in enumerate(newrows)}
 hskin=np.empty(len(neg),dtype=np.float64); hfat=np.empty(len(neg),dtype=np.float64)
 dist=np.empty(len(neg),dtype=np.float64); dist_abs_resid=np.empty(len(neg),dtype=np.float64)
 details=[]
 corrected=lev.copy()
 for j,rr in enumerate(neg):
  r=int(rr)
  if r<len(V['points_world_m']):
   f=int(base_face[r]); bary=np.asarray(base_bary[r],dtype=np.float64)
   d2=float(base_d2[r])
   hs=float(B['interpolated_field_depths_m'][r,0]); hf=float(B['interpolated_field_depths_m'][r,1])
   kind='run004-original-foundation-sample'
   tri=OP[OF[f]]
   nearest=bary@tri
   distance=float(np.sqrt(max(d2,0.0)))
   reconstructed_distance=float(np.linalg.norm(Q[r].astype(np.float64)-nearest))
   reconstruction_error=abs(distance-reconstructed_distance)
  else:
   k=new_lookup.get(r)
   if k is None: raise RuntimeError(f'new row {r} lacks exact nearest-face sample provenance')
   f=int(newface[k]); bary=np.asarray(newbary[k],dtype=np.float64)
   if not 0<=f<len(OF) or not np.isfinite(bary).all() or abs(float(bary.sum())-1)>1e-10 or bary.min()<-1e-10:
    raise RuntimeError(f'bad nearest-face barycentrics row={r} face={f} bary={bary}')
   tri=OP[OF[f]]
   nearest=bary@tri
   distance=float(np.linalg.norm(Q[r].astype(np.float64)-nearest))
   hs,hf=map(float,newh[k])
   reconstruction_error=abs(distance-abs(float(lev[r,0])))
   kind='run013-exact-refinement-sample'
  old_phi=lev[r]
  # Independently reconstruct the retained inferred hskin/hfat identities.
  hs_identity=float(old_phi[1]+old_phi[0]); hf_identity=float(old_phi[2]-old_phi[1])
  hskin[j]=hs_identity; hfat[j]=hf_identity; dist[j]=distance
  dist_abs_resid[j]=abs(distance-abs(float(old_phi[0])))
  if abs(hs_identity-hs)>2e-10 or abs(hf_identity-hf)>2e-10:
   raise RuntimeError(f'unchanged depth provenance mismatch at row {r}: identity {(hs_identity,hf_identity)} sampled {(hs,hf)}')
  if dist_abs_resid[j]>1.1e-7:
   raise RuntimeError(f'nearest source distance differs from |phiO| beyond saved interpolation bound at row {r}: {dist_abs_resid[j]}')
  if hs_identity<=0 or hf_identity<=0 or distance<=0:
   raise RuntimeError(f'nonpositive field depth/distance at row {r}: hskin={hs_identity} hfat={hf_identity} d={distance}')
  corrected[r,0]=distance
  corrected[r,1]=hs_identity-distance
  corrected[r,2]=hs_identity+hf_identity-distance
  details.append({'query_row':r,'sample_kind':kind,'nearest_outer_face_row':f,
    'nearest_outer_barycentrics':bary.tolist(),'distance_to_registered_open_O_m':distance,
    'distance_abs_phi_outer_residual_m':dist_abs_resid[j],
    'hskin_unchanged_m':hs_identity,'hfat_unchanged_m':hf_identity,
    'old_registered_phi_m':old_phi.tolist(),'corrected_registered_phi_m':corrected[r].tolist()})
 if not np.isfinite(corrected).all(): raise RuntimeError('corrected registered fields contain nonfinite values')
 # Save a separate checkpoint: all geometry/topology/ancestry bytes are unchanged; only 670 field rows differ.
 cp_out=OUT/'corrected-foundation-checkpoint.npz'
 np.savez_compressed(cp_out,**{**C,'registered_levelsets_m':corrected})
 cp_sha=sha(cp_out)
 fields_out=OUT/'corrected-registered-levelsets.npz'
 np.savez_compressed(fields_out,registered_levelsets_m=corrected,
   outer_levelset_m=corrected[:,0],skin_levelset_m=corrected[:,1],total_levelset_m=corrected[:,2],
   source_checkpoint_sha256=np.asarray([cp_sha]),parent_registered_fields_sha256=np.asarray([pins['fields']]),
   closed_volume_certificate_sha256=np.asarray([pins['closed_report']]),
   corrected_query_rows=neg.astype(np.int32),distance_to_registered_open_O_m=dist,
   unchanged_hskin_m=hskin,unchanged_hfat_m=hfat)
 fs=sha(fields_out)
 report={'status':'CLOSED_VOLUME_SIGN_CORRECTION_EMITTED_CONSTRUCTION_ONLY',
   'input_sha256':pins,'closed_volume_report_sha256':pins['closed_report'],
   'output_checkpoint':str(cp_out),'output_checkpoint_sha256':cp_sha,
   'output_fields':str(fields_out),'output_fields_sha256':fs,
   'corrected_rows':int(len(neg)),'all_670_certified_inside_closed_O_plus_caps':True,
   'distance_policy':'Reused the exact registered nearest-open-O face/barycentric sample associated with each original field sample; independently reconstructed Euclidean distance from emitted F32 query point and pinned source O triangle. No nearest-normal sign is used.',
   'distance_check':{'max_abs_distance_minus_abs_original_phi_m':float(dist_abs_resid.max(initial=0)),
      'p99_abs_distance_minus_abs_original_phi_m':float(np.quantile(dist_abs_resid,.99)),
      'min_m':float(dist.min()),'max_m':float(dist.max())},
   'unchanged_depths':{'hskin_min_m':float(hskin.min()),'hskin_max_m':float(hskin.max()),
      'hfat_min_m':float(hfat.min()),'hfat_max_m':float(hfat.max()),
      'max_abs_hskin_vs_original_sampled_depth_m':float(max(abs(float(details[j]['hskin_unchanged_m'])-float(newh[new_lookup[int(neg[j])],0])) for j in range(len(neg)) if int(neg[j]) in new_lookup) if any(int(x) in new_lookup for x in neg) else 0.0),
      'hskin_policy':'unchanged algebraic source thickness hskin=old phi_skin+old phi_outer',
      'hfat_policy':'unchanged algebraic source thickness hfat=old phi_total-old phi_skin'},
   'formula':'For rows exact-classified inside: phiO=+d, phiI=hskin-d, phiD=hskin+hfat-d. Rows outside the negative set are bytewise unchanged.',
   'changed_components':int(np.count_nonzero(corrected!=lev)),
   'unchanged_node_geometry_and_tetrahedra':True,'native_physics_or_mass_admission':False,
   'records':details}
 report_path=OUT/'report.json'; report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({k:report[k] for k in ('status','output_checkpoint','output_checkpoint_sha256','output_fields','output_fields_sha256','corrected_rows','distance_check','unchanged_depths','changed_components')},indent=2))
 print('report',report_path,'sha256',sha(report_path))
if __name__=='__main__': main()
