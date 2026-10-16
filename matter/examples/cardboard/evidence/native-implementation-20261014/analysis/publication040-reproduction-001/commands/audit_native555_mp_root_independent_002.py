from pathlib import Path
import json,hashlib,sys,datetime
import numpy as np
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');d=q/'analysis/native555-mp-root-independent-adjudication-002';d.mkdir()
paths={
 'capture':q/'solver/control19-newton3-resume-diagnostic-555/diagnostics/capture-operator-fd-capture/capture',
 'mp_arrays':q/'analysis/native555-global-ideal-fd-mp-full-006/arrays.npz',
 'fp64_arrays':q/'analysis/native555-actual19-material-replay-001/environment-0/arrays.npz',
 'legacy_arrays':q/'analysis/native555-global-ideal-fd-002/assembled_fd.npz',
 'native_vectors':q/'analysis/native555-cached-operator-001/operator.vectors.npz',
 'agent_erratum':q/'analysis/native555-global-ideal-fd-mp-factor2-erratum-004/report.json',
 'parser':q/'analysis/source198-strict-layer-replay-preparation-001/strict_layers.py',
 'reference':q/'snapshots/source-stable-green-reference-206-buildfix-002/matter/tools/cardboard_bilinear_fp64_reference.py',
 'compensated_reference':q/'snapshots/source-stable-green-reference-206-buildfix-002/matter/tools/cardboard_compensated_fp64_reference.py',
 'geometry_parser':q/'snapshots/source-stable-green-reference-206-buildfix-002/matter/tools/cardboard_cached_operator_audit.py'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();before={k:sha(p) for k,p in paths.items()}
assert before['capture']=='c0a111c510ef4c5891ace3bb58f65af48740733d7b46987be370561d1c22ab53'
assert before['mp_arrays']=='d6d6bb54b7e4a4e9644f5aea1b25bfeeef37a5433ce45302e0adc887db67644f'
assert before['agent_erratum']=='f013f321eee3b011135f9e4348923566be274d4c5b3cd235fad073ea97e2f25b'
sys.path[:0]=[str(paths['parser'].parent),str(paths['reference'].parent)];import strict_layers as parser
header,nr,tr,_=parser.geometry.capture_records(paths['capture'])
nodes=sorted((r for r in nr if r['environment']==0),key=lambda r:r['node_id']);tets=sorted((r for r in tr if r['environment']==0),key=lambda r:r['tet_id'])
assert len(nodes)==882 and len(tets)==1920
mp=np.load(paths['mp_arrays']);fp=np.load(paths['fp64_arrays']);old=np.load(paths['legacy_arrays']);native=np.load(paths['native_vectors'])
free=np.asarray(mp['free_mask'],bool);assert np.array_equal(np.flatnonzero(free),native['free_node_ids'])
dt=float(np.float32(header['dt_seconds']));mass=np.array([r['center']['mass_kg'] for r in nodes],np.float32).astype(float)
dv=np.array([r['correction_delta_v_m_s'] for r in nodes],np.float32).astype(float)
inverse=np.array([t['inverse_rest_rows_per_m'] for t in tets],np.float32).astype(float);volume=np.array([t['rest_volume_m3'] for t in tets],np.float32).astype(float)
ids=np.array([t['nodes'] for t in tets],int)
rot=[]
for t in tets:
 if t['center']['dt_seconds_bulk_modulus_kind_frame_active'][3]!=1:rot.append(np.eye(3));continue
 x,y,z,w=np.array(t['material_frame_quaternion_xyzw'],np.float32).astype(float);n=np.sqrt(x*x+y*y+z*z+w*w);x/=n;y/=n;z/=n;w/=n
 rot.append(np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]))
rot=np.array(rot)
def gather_scalar(local):
 result=np.zeros((882,3));max_balance=0.
 for t in range(1920):
  world=local[t]@rot[t].T
  f=np.zeros((4,3))
  for j in range(1,4):
   for a in range(3):
    f[j,a]=-volume[t]*sum(world[a,k]*inverse[t,j-1,k] for k in range(3))
  f[0]=-np.sum(f[1:],axis=0);max_balance=max(max_balance,float(np.max(np.abs(np.sum(f,axis=0)))))
  for j in range(4):result[ids[t,j]]+=f[j]
 return result,max_balance
def error(x,y):
 delta=(x-y)[free];target=y[free]
 return {'relative_l2':float(np.linalg.norm(delta)/np.linalg.norm(target)),'absolute_l2_Ns':float(np.linalg.norm(delta)),'maximum_abs_Ns':float(np.max(np.abs(delta))),'target_l2_Ns':float(np.linalg.norm(target))}
ift=mp['mp_center_tangent_all_tets'];assert ift.shape==(1920,3,3) and np.isfinite(ift).all()
f,bal=gather_scalar(ift);center=dt*dt*f-mass[:,None]*dv
mp_checks={}
for pw in (26,28):
 f,b=gather_scalar(mp[f'mp_secant_p{pw}']);action=dt*f-mass[:,None]*dv
 mp_checks[str(pw)]={'vs_independent_center_IFT':error(action,center),'vs_saved_secant_action':error(action,mp[f'mp_assembled_dR_p{pw}']),'element_force_balance_max_N':b}
ref=fp['reference_C'];H=np.array([t['center']['F_direction_local_per_s'] for t in tets],np.float32).astype(float)
ch=np.einsum('tij,tj->ti',ref,H.reshape(1920,9)).reshape(1920,3,3)
rel=np.linalg.norm((ch-ift).reshape(1920,9),axis=1)/np.linalg.norm(ift.reshape(1920,9),axis=1)
nativeA=np.zeros((882,3));nativeA[free]=native['native_action_Ns'].reshape(-1,3)
r={'schema':'numi.cardboard.root-independent-MP-derivative-adjudication.v1','input_paths':{k:str(p) for k,p in paths.items()},'input_sha256':before,'cohort':{'paper':1728,'starch':192,'tets':1920,'free_nodes':int(free.sum())},'independent_scalar_force_assembly':True,'independent_unit_quaternion_frame_transform':True,'FP64_C_H_vs_MP_IFT':{'maximum_relative':float(rel.max()),'rows_above_1e_minus_6':int(np.sum(rel>1e-6)),'all_rows_pass':bool(np.all(rel<=1e-6))},'MP_center_vs_saved_analytic_residual_derivative':error(center,old['linear_residual_derivative']),'native_A_delta_vs_negative_MP_center_residual_derivative':error(nativeA,-center),'MP_secants':mp_checks,'smooth_IFT_gate_pass':bool(np.all(rel<=1e-6)),'native_assembled_action_magnitude_gate_pass':error(nativeA,-center)['relative_l2']<=.01,'native_empirical_finite_difference_noise_floor_qualified':False,'complete_coupon_or_mesh_time_convergence_qualified':False,'limits':['Cross-audit reuses the source-bound MP projection arrays; it independently recomputes frame transforms and scalar tet/nodal actions, not the 90-digit nonlinear projections.','The factor-of-two FP64 secant report is retained with its erratum; this audit uses MP secants only.','Magnitude gates do not establish the native FP32 finite-difference noise floor or repair the complete nonlinear trajectory.'],'all_inputs_unchanged':all(sha(p)==before[k] for k,p in paths.items()),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
assert r['smooth_IFT_gate_pass'] and r['native_assembled_action_magnitude_gate_pass'] and r['all_inputs_unchanged']
assert mp_checks['28']['vs_independent_center_IFT']['absolute_l2_Ns']<mp_checks['26']['vs_independent_center_IFT']['absolute_l2_Ns']
(d/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
(d/'manifest.json').write_text(json.dumps({'schema':'numi.cardboard.analysis-manifest.v1','files':{'report.json':sha(d/'report.json')},'inputs_sha256':before},indent=2)+'\n')
print(json.dumps(r),flush=True)
