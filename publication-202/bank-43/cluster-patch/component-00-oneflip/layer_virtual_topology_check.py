#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,time
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
W=S/'per-bank-workers-20261027/b43-16/attempt-025-witness-cluster-topology-001/component-00-oneflip'
BASE=S/'per-bank-workers-20261027/b65-99/root-layer-guard-002/backtrack-0p25/candidate.npz'
C=W/'candidate-private.npz'
L=S/'per-bank-workers-20261027/physical-layer-diagnostic-001/guarded-65-quarter/skin-fat-layers-541.npz'
BPIN='a69f3b0db4c84cfae63177dd5aa96ec339b128870f4d0261c43d56e39d2a64db'
CPIN='1918739307097c6c51f7bfd867cf645b7b05aa2450b7af18655272a3630272ac'
LPIN='78916a0c05eb130e5efb02174342295c2d8d4444eae963c7b36fc43a2a1b17fc'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def signed(v,sgn):
 p0,p1,p2,p3=v;return float(sgn*np.dot(p1-p0,np.cross(p2-p0,p3-p0))/6)
def tets(face,out,inn):
 a,b,c=sorted(map(int,face));parity=(int(face[0]>face[1])+int(face[0]>face[2])+int(face[1]>face[2]))&1;ex=1. if parity else -1.
 return [
  ([(a,out[a]),(b,out[b]),(c,out[c]),(c,inn[c])],ex),
  ([(a,out[a]),(b,out[b]),(b,inn[b]),(c,inn[c])],-ex),
  ([(a,out[a]),(a,inn[a]),(b,inn[b]),(c,inn[c])],ex)]
def main():
 t=time.time()
 if sha(BASE)!=BPIN or sha(C)!=CPIN or sha(L)!=LPIN:raise ValueError('layer candidate/base/file pin mismatch')
 with np.load(BASE,allow_pickle=False) as z: F0=np.asarray(z['source_O_faces'],np.int64);P0=np.asarray(z['points_world_m'],np.float32)
 with np.load(C,allow_pickle=False) as z: F1=np.asarray(z['source_O_faces'],np.int64);P1=np.asarray(z['points_world_m'],np.float32)
 if not np.array_equal(P0.view(np.uint32),P1.view(np.uint32)):raise ValueError('topology flip moved point rows')
 changed=np.flatnonzero(np.any(F0!=F1,axis=1))
 if changed.tolist()!=[19114,19126]:raise ValueError('unexpected topology diff '+str(changed.tolist()))
 with np.load(L,allow_pickle=False) as z:
  outer=np.asarray(z['outer_points_f64'],np.float64)*1000.;inner=np.asarray(z['inner_points_f64'],np.float64)*1000.;deep=np.asarray(z['deep_points_f64'],np.float64)*1000.;lf=np.asarray(z['outer_faces'],np.int64)
 if not np.array_equal(lf,F0[:,[0,2,1]]):raise ValueError('pinned outer layer face identity mismatch')
 intervals=[]; allok=True
 for li,(outp,inp) in enumerate(((outer,inner),(inner,deep))):
  base=[];trial=[]
  for row in changed:
   for verts,sgn in tets(F0[row],outp,inp):base.append({'row':int(row),'slot':len(base)%3,'signed_mm3':signed([v[1] for v in verts],sgn)})
   for verts,sgn in tets(F1[row],outp,inp):trial.append({'row':int(row),'slot':len(trial)%3,'signed_mm3':signed([v[1] for v in verts],sgn)})
  bvals=np.asarray([x['signed_mm3'] for x in base]);tvals=np.asarray([x['signed_mm3'] for x in trial])
  ok=int(np.count_nonzero(tvals<=0))<=int(np.count_nonzero(bvals<=0)) and float(tvals.sum())>=float(bvals.sum())-1e-12 and np.all(np.isfinite(tvals))
  allok &= bool(ok)
  intervals.append({'layer_interval':li,'base_tetra_signed_mm3':base,'trial_tetra_signed_mm3':trial,'base_nonpositive_count':int(np.count_nonzero(bvals<=0)),'trial_nonpositive_count':int(np.count_nonzero(tvals<=0)),'base_signed_sum_mm3':float(bvals.sum()),'trial_signed_sum_mm3':float(tvals.sum()),'min_trial_signed_mm3':float(tvals.min()),'local_construction_gate':bool(ok)})
 r={'schema':'numi.human.component0.virtual_layer_topology_check.v1','status':'PASS_LOCAL_LAYER_TOPOLOGY_CONSTRUCTION_ONLY' if allok else 'FAIL_LOCAL_LAYER_TOPOLOGY_GUARD','base_sha256':sha(BASE),'candidate_sha256':sha(C),'layer_file_sha256':sha(L),'changed_face_rows':changed.tolist(),'candidate_points_bitwise_same':True,'virtual_same_flip_applied_to_outer_inner_deep_topologies':True,'outer_points_match_a69':bool(np.array_equal(outer/1000.,P0.astype(np.float64))),'intervals':intervals,'physical_layer_file_modified':False,'physical_transfer_steps':0,'geometry_admitted':False,'elapsed_seconds':time.time()-t,'limitations':['Only the two-face local layer tetrahedra were compared; this is a virtual same-row connectivity remesh check, not physical material-transfer admission.','The existing guarded layer archive still stores the original topology and was not changed.']}
 out=W/'layer-virtual-topology-check.json';out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':r['status'],'intervals':[{'interval':x['layer_interval'],'base_nonpositive':x['base_nonpositive_count'],'trial_nonpositive':x['trial_nonpositive_count'],'base_sum':x['base_signed_sum_mm3'],'trial_sum':x['trial_signed_sum_mm3'],'min_trial':x['min_trial_signed_mm3']} for x in intervals]},indent=2));print('receipt_sha256',sha(out))
if __name__=='__main__':main()
