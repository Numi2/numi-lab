from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
B=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b65-99')
RUN=B/'attempt-022-boundary-owner-flip-001/run-001'
BASE=B/'root-layer-guard-002/backtrack-0p25/candidate.npz'
FULL=RUN/'candidate.npz'; OPS=RUN/'receipt.json'
LAYER=B.parent/'physical-layer-diagnostic-001/guarded-65-quarter/skin-fat-layers-541.npz'
LREPORT=LAYER.parent/'report.json'
PATCH=B/'attempt-021-boundary-expansion-001/run-001/iteration-002/rows.json'
BOUND=B/'attempt-021-boundary-expansion-001/run-001/iteration-002/preflight.json'
OWNER=Path('/Users/numi/Work/numi-human-connected-surface-069/D/local-two-parent-retriangulation-402/two_parent_patch_flips_402.py')
OUT=RUN/'layer-filter-retry002'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def vol(a,b,c,d):return np.einsum('ij,ij->i',np.cross(b-a,c-a),d-a)/6.0
def check_layer(z,faces,rows,outerkey,innerkey,fieldkey):
 src=np.asarray(faces,dtype=np.int64);q=src[rows]
 order=np.argsort(q,axis=1);f=np.take_along_axis(q,order,axis=1);a,b,c=f.T
 inv=((q[:,0]>q[:,1]).astype(np.int8)+(q[:,0]>q[:,2]).astype(np.int8)+(q[:,1]>q[:,2]).astype(np.int8));parity=inv&1
 outer=np.asarray(z[outerkey],np.float64);inner=np.asarray(z[innerkey],np.float64)
 v0=vol(outer[a],outer[b],outer[c],inner[c]);v1=vol(outer[a],outer[b],inner[b],inner[c]);v2=vol(outer[a],inner[a],inner[b],inner[c])
 field=np.asarray(z[fieldkey],np.float64);tol=np.maximum(1e-8,np.maximum.reduce([field[a],field[b],field[c]]))**3*1e-12
 expected=np.where(parity,1.0,-1.0)
 authored=(np.isfinite(v0)&np.isfinite(v1)&np.isfinite(v2)&(np.abs(v0)>tol)&(np.abs(v1)>tol)&(np.abs(v2)>tol)&(np.sign(v0)==expected)&(np.sign(v1)==-expected)&(np.sign(v2)==expected))
 ids=np.stack([np.stack([a,b,c,len(outer)+c],axis=1),np.stack([a,b,len(outer)+b,len(outer)+c],axis=1),np.stack([a,len(outer)+a,len(outer)+b,len(outer)+c],axis=1)],axis=1).reshape(-1,4)
 xyz64=np.concatenate([outer,inner]);tet64=vol(xyz64[ids[:,0]],xyz64[ids[:,1]],xyz64[ids[:,2]],xyz64[ids[:,3]])
 reverse=tet64<0;ids[reverse,0],ids[reverse,1]=ids[reverse,1].copy(),ids[reverse,0].copy()
 xyz32=np.concatenate([outer.astype(np.float32).astype(np.float64),inner.astype(np.float32).astype(np.float64)])
 p=xyz32[ids];v=vol(p[:,0],p[:,1],p[:,2],p[:,3]);long=np.zeros(len(p))
 for i in range(4):
  for j in range(i+1,4):long=np.maximum(long,np.linalg.norm(p[:,i]-p[:,j],axis=1))
 nd=6*v/np.maximum(long,1e-300)**3
 amin=[]
 surf=inner[q];area=.5*np.linalg.norm(np.cross(surf[:,1]-surf[:,0],surf[:,2]-surf[:,0]),axis=1)
 for i in range(len(rows)):
  sl=slice(3*i,3*i+3)
  amin.append({'face_row':int(rows[i]),'authored_fail':bool(not authored[i]),
   'authored_signed_volumes_m3':[float(v0[i]),float(v1[i]),float(v2[i])],
   'f32_nonpositive_tet_ids':[int(3*int(rows[i])+j) for j in range(3) if v[sl][j]<=0],
   'f32_conditioned_tet_ids_lt_1e-5':[int(3*int(rows[i])+j) for j in range(3) if nd[sl][j]<1e-5],
   'f32_min_normalized_det':float(nd[sl].min()),'inner_face_area_m2':float(area[i])})
 return amin
def main():
 OUT.mkdir(exist_ok=False);r=json.loads(OPS.read_text());base=load(BASE);allc=load(FULL);z=load(LAYER)
 lp=json.loads(LREPORT.read_text())
 if r['pins']['base']!='a69f3b0db4c84cfae63177dd5aa96ec339b128870f4d0261c43d56e39d2a64db':raise ValueError('candidate base pin mismatch')
 if lp['pins']['current_candidate']['sha256']!=r['pins']['base']:raise ValueError('layer output current candidate does not match base')
 if not np.array_equal(z['outer_faces'],base['source_O_faces'][:,[0,2,1]]):raise ValueError('541 outer faces differ from pinned base with layer winding')
 rows=np.asarray(json.loads(PATCH.read_text()),np.int64)
 if not np.array_equal(base['points_world_m'],allc['points_world_m']):raise ValueError('flip candidate moved coordinates')
 owner=mod('owner402_layer_filter',OWNER);n=len(base['source_O_faces'])
 safe=[];excluded=[];details=[]
 for oi,op in enumerate(r['operations']):
  gr=[int(rows[x]) for x in op['face_rows']]
  old={};new={}
  old['skin']=check_layer(z,z['outer_faces'],gr,'outer_points_f64','inner_points_f64','skin_thickness_transferred_m')
  old['fat']=check_layer(z,z['outer_faces'],gr,'inner_points_f64','deep_points_f64','fat_thickness_transferred_m')
  newfaces=allc['source_O_faces'][:,[0,2,1]]
  new['skin']=check_layer(z,newfaces,gr,'outer_points_f64','inner_points_f64','skin_thickness_transferred_m')
  new['fat']=check_layer(z,newfaces,gr,'inner_points_f64','deep_points_f64','fat_thickness_transferred_m')
  fail=[]
  for phase,measure in [('old',old),('new',new)]:
   for layer,items in measure.items():
    for item in items:
     if item['authored_fail'] or item['f32_nonpositive_tet_ids'] or item['f32_conditioned_tet_ids_lt_1e-5'] or item['inner_face_area_m2']<=0:
      fail.append({'phase':phase,'layer':layer,'face_row':item['face_row'],'authored_fail':item['authored_fail'],'f32_nonpositive_tet_ids':item['f32_nonpositive_tet_ids'],'f32_conditioned_tet_ids_lt_1e-5':item['f32_conditioned_tet_ids_lt_1e-5'],'f32_min_normalized_det':item['f32_min_normalized_det'],'inner_face_area_m2':item['inner_face_area_m2']})
  rec={'operation_index':oi,'local_patch_rows':op['face_rows'],'global_face_rows':gr,'edge':op['edge'],'new_diagonal':op['new_diagonal'],'layer_old':old,'layer_new':new,'failure_witnesses':fail,'layer_gate_pass':not fail}
  details.append(rec)
  if fail:excluded.append(oi)
  else:safe.append(oi)
 safe_rows=sorted({int(rows[x]) for oi in safe for x in r['operations'][oi]['face_rows']})
 ex_rows=sorted({int(rows[x]) for oi in excluded for x in r['operations'][oi]['face_rows']})
 out={k:np.array(v,copy=True) for k,v in base.items()}
 csr=(('face_source_ancestry_indptr','face_source_ancestry_face_rows'),('face_fine_subface_ancestry_indptr','face_fine_subface_ancestry_rows'),('refinement_face_parent_ancestry_indptr','refinement_face_parent_ancestry_face_rows'))
 for oi in safe:
  op=r['operations'][oi]
  for local in op['face_rows']:
   g=int(rows[local])
   out['source_O_faces'][g]=allc['source_O_faces'][g]
   out['area_reference_037_source_O_faces'][g]=allc['area_reference_037_source_O_faces'][g]
   for k in ('refinement_face_parent_face_row','refinement_face_parent_barycentric','parent_coarse_face_row','parent_source_face_marker'):
    out[k][g]=allc[k][g]
 for pk,vk in csr:
  old=owner.csr_rows(base[pk],base[vk],n);new=owner.csr_rows(allc[pk],allc[vk],n)
  for g in safe_rows:old[g]=new[g]
  out[pk],out[vk]=owner.csr_build(old,np.asarray(base[vk]).dtype)
 for k in ('refinement_face_parent_ancestry_semantics','refinement_face_parent_barycentric_semantics'):
  if safe and k in allc:out[k]=np.array(allc[k],copy=True)
 emitted=np.flatnonzero(np.any(out['source_O_faces']!=base['source_O_faces'],axis=1)).tolist()
 if emitted!=safe_rows:raise ValueError(f'safe subset changed-row mismatch: {emitted} vs {safe_rows}')
 report={'schema':'numi.human.b65-99.per-flip-layer-filter.v1','status':'LAYER_FILTERED_SAFE_SUBSET_READY' if safe else 'STOP_NO_LAYER_SAFE_FLIPS','base_candidate_sha256':sha(BASE),'full_eight_flip_candidate_sha256':sha(FULL),'layer_npz_path':str(LAYER),'layer_npz_sha256':sha(LAYER),'layer_report_path':str(LREPORT),'layer_report_sha256':sha(LREPORT),'layer_candidate_base_sha256':lp['pins']['current_candidate']['sha256'],'patch_rows_sha256':sha(PATCH),'operation_count':len(r['operations']),'safe_operation_indices':safe,'excluded_operation_indices':excluded,'safe_face_rows':safe_rows,'excluded_face_rows':ex_rows,'operations':details,'safe_subset_candidate_written':False,'geometry_admitted':False,'native_admission':False,'accepted_physical_steps':0,'limitations':['This is a local inherited offset-prism orientation and Float32-conditioning check, not physical solver admission.','An operation is excluded if any old or new affected skin/fat prism is already invalid or ill-conditioned.']}
 if safe:
  cand=OUT/'candidate-layer-safe-subset.npz';np.savez_compressed(cand,**out)
  report['safe_subset_candidate_written']=True;report['candidate_path']=str(cand);report['candidate_sha256']=sha(cand)
  report['safe_subset_face_count']=len(safe_rows)
 (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'status':report['status'],'safe_operation_indices':safe,'excluded_operation_indices':excluded,'safe_face_rows':safe_rows,'excluded_face_rows':ex_rows,'candidate_sha256':report.get('candidate_sha256'),'layer_report_sha256':report['layer_report_sha256'],'failure_witness_count':sum(len(x['failure_witnesses']) for x in details)},indent=2))
if __name__=='__main__':main()
