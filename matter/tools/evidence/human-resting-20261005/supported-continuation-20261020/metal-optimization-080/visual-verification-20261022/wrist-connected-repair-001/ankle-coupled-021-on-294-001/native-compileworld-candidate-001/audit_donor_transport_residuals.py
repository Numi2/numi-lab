#!/usr/bin/env python3
"""Compare baseline/candidate 294 surface LBS points against the same coarse donor transport."""
import csv, hashlib, json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/Users/n/numi-human-anatomy-finish-1378/src')
from numilab_human.skin_source_payload_preflight import decode_payload

ROOT=Path('/Users/n/numi-human-aortic-contact-opt-20261019')
E=ROOT/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
OUT=E/'ankle-coupled-021-on-294-001/native-compileworld-candidate-001'
BASE=Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin')
CAND=E/'ankle-coupled-021-on-294-001/ankle-coupled-021-on-294.nhskin'
RED=Path('/Users/n/numi-human-separation-20261018/C/wholebody-integration-295/build-wholebody-016/source/apps/NumiHumanWholeBodySkinReduction.hpp')
NPZ=E/'ankle-coupled-021-on-294-001/main294-commonmode-layer-candidate-001/candidate-layers.npz'
CSV=Path('/Users/n/numi-human-separation-20261018/C/wholebody-integration-295/native-wholebody-001/registered-skin.csv')
POSES={
 'reference':E/'native-registered-skin-pose-clean-014/registered-reference-elevation-010-014/registered-skin-poses.json',
 'dorsiflexion':E/'native-registered-joint-reference-pairs-016/ankle-dorsiflexion-020-ref010/registered-skin-poses.json',
 'plantarflexion':E/'native-registered-joint-reference-pairs-016/ankle-plantarflexion-020-ref010/registered-skin-poses.json',
}
POSE_CSV={
 'dorsiflexion':E/'native-registered-joint-reference-pairs-016/ankle-dorsiflexion-020-ref010/registered-skin.csv',
 'plantarflexion':E/'native-registered-joint-reference-pairs-016/ankle-plantarflexion-020-ref010/registered-skin.csv',
}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def numbers(s): return np.array([float(x.strip().rstrip('f')) for x in s.split(',') if x.strip()],float)
def parse_int_array(text,name):
 m=re.search(r'\b'+re.escape(name)+r'\s*=\s*\{(.*?)\}\s*;',text,re.S)
 if not m: raise ValueError('missing '+name)
 return np.array([int(x) for x in re.findall(r'(?<![A-Za-z_])(\d+)(?:u)?',m.group(1))],np.int64)
def parse_vec_array(text,name):
 pos=text.index(name); end=text.index('};',pos)
 vals=[numbers(x) for x in re.findall(r'std::array<double,3>\{([^}]*)\}',text[pos:end])]
 return np.asarray(vals,float)
def rot(q):
 x,y,z,w=map(float,q); n=np.sqrt(x*x+y*y+z*z+w*w); x,y,z,w=x/n,y/n,z/n,w/n
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def posed(d,p):
 V=np.asarray(d['vertices_f'][:,:3],float); W=np.asarray(d['full_weights'],float)
 B=np.asarray(d['bindings_u'][:,0],int); BF=np.asarray(d['bindings_f'],float)
 state={int(x['body_index']):x for x in p['bodies']}; out=np.zeros_like(V); done=set()
 for j,b in enumerate(B):
  b=int(b)
  if b in done: continue
  done.add(b); cur=state[b]['current']; Rc=rot(cur['orientation_world_xyzw']); t=np.asarray(cur['position_world_m'])
  Rb=rot(BF[j,4:8]); local=(V@Rb.T)*BF[j,8]+BF[j,1:4]
  out+=W[:,j,None]*(local@Rc.T+t)
 return out.astype(np.float32)
def transported(local,pose,body):
 b={int(x['body_index']):x for x in pose['bodies']}[int(body)]['current']
 return (local@rot(b['orientation_world_xyzw']).T+np.asarray(b['position_world_m'])).astype(np.float32)
def stats(x):
 x=np.asarray(x,float)
 return {'count':int(x.size),'p50_mm':float(np.percentile(x,50)),'p95_mm':float(np.percentile(x,95)),'p99_mm':float(np.percentile(x,99)),'max_mm':float(x.max())}
def main():
 text=RED.read_text()
 sv=parse_int_array(text,'kWholeBodySkinSourceVertices')
 framepat=r'WholeBodySkinDonorFrameV1\{(\d+)u,\{([^}]*)\},\{([^}]*)\},\{([^}]*)\},\{([^}]*)\}\}'
 frames=[(int(m.group(1)),numbers(m.group(2)),numbers(m.group(3)),numbers(m.group(4)),numbers(m.group(5))) for m in re.finditer(framepat,text)]
 dl=parse_vec_array(text,'kWholeBodySkinMechanicalDonorLocalDeltas')
 dw=parse_vec_array(text,'kWholeBodySkinMechanicalWorldDeltas')
 if len(sv)!=2332 or len(frames)!=len(sv) or dl.shape!=(len(sv),3) or dw.shape!=(len(sv),3): raise ValueError('clean 294 header shape mismatch')
 donors=np.array([x[0] for x in frames],np.int64)
 base_local=np.asarray([x[1] for x in frames])+dl
 with np.load(NPZ,allow_pickle=False) as z: A={k:np.array(z[k]) for k in z.files}
 cand_local=np.asarray(A['outer_donor_local_m'],float)
 if A['donor_body_indices'].shape!=(len(sv),) or not np.array_equal(donors,A['donor_body_indices']): raise ValueError('candidate donor owners differ from clean 294 rows')
 d0=decode_payload(BASE.read_bytes());d1=decode_payload(CAND.read_bytes())
 if d0['vertex_count']!=d1['vertex_count'] or d0['binding_count']!=86 or d1['binding_count']!=86: raise ValueError('payload shape/binding mismatch')
 if not np.array_equal(d0['indices'],d1['indices']): raise ValueError('candidate topology differs from base')
 changed=np.any(np.asarray(d0['vertices_f'][:,:3],np.float32)!=np.asarray(d1['vertices_f'][:,:3],np.float32),axis=1)|np.any(np.asarray(d0['full_weights'],np.float32)!=np.asarray(d1['full_weights'],np.float32),axis=1)
 affected=np.flatnonzero(np.isin(sv,np.flatnonzero(changed)))
 if sha(CAND)!='d07b27e7d9d6952a3bc33d41535693b45f107f629e69520c2afbfb990ed1237f': raise ValueError('candidate NHSKIN hash changed')
 if sha(CSV)!='4dd63097a035f5f385712307f2336ef6fd46c964d1b93a2e85d84193ac12d327': raise ValueError('base 294 registered CSV changed')
 pose_data={k:json.loads(p.read_text()) for k,p in POSES.items()}
 allresults={}
 ref=pose_data['reference']
 base_ref=np.empty((len(sv),3),np.float32);cand_ref=np.empty_like(base_ref)
 for b in np.unique(donors):
  m=donors==b; base_ref[m]=transported(base_local[m],ref,b);cand_ref[m]=transported(cand_local[m],ref,b)
 base_header=np.asarray([x[3] for x in frames])+dw
 max_base_ref=float(np.max(np.linalg.norm(base_ref-base_header,axis=1))*1000)
 max_cand_ref=float(np.max(np.linalg.norm(cand_ref-np.asarray(A['outer_world_m'],np.float32),axis=1))*1000)
 # Validate unposed reference source evaluator against the 294 CSV (xyz columns 2:5).
 csvref=np.loadtxt(CSV,delimiter=',',skiprows=1,usecols=(0,2,3,4),dtype=np.float32)
 ref_lbs=posed(d0,ref); referr=np.linalg.norm(ref_lbs[csvref[:,0].astype(int)]-csvref[:,1:],axis=1)*1000
 allresults['pose_frame_validation']={'base_294_reference_LBS_vs_registered_csv_max_mm':float(referr.max()),'base_294_reference_LBS_vs_registered_csv_p99_mm':float(np.percentile(referr,99)),'base_donor_transport_vs_header_outer_max_mm':max_base_ref,'candidate_donor_transport_vs_layer_outer_max_mm':max_cand_ref,'direct_source_rows':int(len(sv)),'candidate_changed_source_rows':int(np.sum(changed)),'affected_direct_coarse_rows':affected.astype(int).tolist()}
 for label,p in pose_data.items():
  V0=posed(d0,p);V1=posed(d1,p)
  R0=np.empty((len(sv),3),np.float32);R1=np.empty_like(R0)
  for b in np.unique(donors):
   m=donors==b;R0[m]=transported(base_local[m],p,b);R1[m]=transported(cand_local[m],p,b)
  r0=np.linalg.norm(V0[sv]-R0,axis=1)*1000
  r1=np.linalg.norm(V1[sv]-R1,axis=1)*1000
  item={'baseline_all_rows':stats(r0),'candidate_all_rows':stats(r1),'baseline_affected_coarse_rows':stats(r0[affected]),'candidate_affected_coarse_rows':stats(r1[affected]),'affected_rows':[]}
  for row in affected:
   item['affected_rows'].append({'coarse_row':int(row),'source_vertex':int(sv[row]),'donor_body':int(donors[row]),'baseline_residual_mm':float(r0[row]),'candidate_residual_mm':float(r1[row]),'candidate_minus_baseline_mm':float(r1[row]-r0[row])})
  for row in (1759,1773):
   item.setdefault('reported_witness_rows',[]).append({'coarse_row':row,'source_vertex':int(sv[row]),'donor_body':int(donors[row]),'baseline_residual_mm':float(r0[row]),'candidate_residual_mm':float(r1[row]),'candidate_minus_baseline_mm':float(r1[row]-r0[row])})
  if label in POSE_CSV:
   a=np.loadtxt(POSE_CSV[label],delimiter=',',skiprows=1,usecols=(0,2,3,4),dtype=np.float32)
   nativeerr=np.linalg.norm(V0[a[:,0].astype(int)]-a[:,1:],axis=1)*1000
   item['base_LBS_vs_saved_native_pose_CSV_max_mm']=float(nativeerr.max())
   item['base_LBS_vs_saved_native_pose_CSV_p99_mm']=float(np.percentile(nativeerr,99))
  allresults[label]=item
 report={'schema':'numi.human.ankle-294-commonmode-donor-residual-comparison.v1','status':'candidate_only_diagnostic_not_dynamically_qualified','method':'At the same matched registered rigid-body snapshots, evaluate baseline 294 and candidate NHSKIN with the exact 86-binding LBS. Compare each mapped 2332 source vertex with the same donor-local coarse outer point transported by that donor body. Baseline donor-local coordinates are clean 294 donorLocalPoint plus its authored mechanical donor-local delta; candidate coordinates are the exact candidate-layers outer_donor_local_m. Distances are magnitudes in mm. This compares source-render to coarse rigid-donor transport, not FEM motion or contact.','inputs':{'baseline_payload':str(BASE),'baseline_payload_sha256':sha(BASE),'candidate_payload':str(CAND),'candidate_payload_sha256':sha(CAND),'clean_294_reduction_header':str(RED),'clean_294_reduction_header_sha256':sha(RED),'candidate_layers_npz':str(NPZ),'candidate_layers_npz_sha256':sha(NPZ),'base_registered_csv_sha256':sha(CSV),'reference_pose':str(POSES['reference']),'reference_pose_sha256':sha(POSES['reference']),'dorsiflexion_pose':str(POSES['dorsiflexion']),'dorsiflexion_pose_sha256':sha(POSES['dorsiflexion']),'plantarflexion_pose':str(POSES['plantarflexion']),'plantarflexion_pose_sha256':sha(POSES['plantarflexion'])},'measurements':allresults,'limits':['No FEM simulation, layer deformation, dynamic contact, or native admission is tested.','The matched snapshots are registration-only body poses; this isolates the donor rigid-transport versus surface-LBS mismatch.','Unaffected coarse rows are retained in global distributions; affected row set is determined by exact source-position or 86-weight changes in the candidate payload.']}
 out=OUT/'donor-transport-residual-comparison.json';out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');json.loads(out.read_text())
 print(json.dumps({'report':str(out),'sha256':sha(out),'validation':allresults['pose_frame_validation'],'rows_1759_1773':{k:v['reported_witness_rows'] for k,v in allresults.items() if isinstance(v,dict) and 'reported_witness_rows' in v}},indent=2))
if __name__=='__main__':main()
