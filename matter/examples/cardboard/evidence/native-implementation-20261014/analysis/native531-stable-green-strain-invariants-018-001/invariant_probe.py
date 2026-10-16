import json, hashlib, sys, time
from pathlib import Path
import numpy as np
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
PK=Q/'analysis/native531-stable-green-strain-invariants-018-001'
sys.path.insert(0,str(Q/'analysis/native531-stable-green-strain-reference-017-001/reference'))
import cardboard_bilinear_fp64_reference as ref
import cardboard_cached_operator_audit as geo
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def oldstrain(f):
 e=(np.einsum('...ki,...kj->...ij',f,f)-np.eye(3))/2
 return e[...,(0,1,2,1,0,0),(0,1,2,2,2,1)]
def stablestrain(f):
 d=f-np.eye(3,dtype=f.dtype);e=(d+np.swapaxes(d,-1,-2)+np.einsum('...ki,...kj->...ij',d,d))/2
 return e[...,(0,1,2,1,0,0),(0,1,2,2,2,1)]
def decstrain(f):
 from decimal import Decimal,localcontext
 with localcontext() as c:
  c.prec=100;a=[[Decimal.from_float(float(f[i,j])) for j in range(3)] for i in range(3)];o=[]
  for i,j in [(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]:
   v=sum(a[k][i]*a[k][j] for k in range(3))-(Decimal(1) if i==j else Decimal(0));o.append(float(v/2))
  return np.array(o)
def norm(x):return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))
def st(p,F,z,strain_fn):
 s=p[10]*np.einsum('ij,j->i',ref.stiffness(p),strain_fn(F)-z[:6]);S=np.array([[s[0],s[5],s[4]],[s[5],s[1],s[3]],[s[4],s[3],s[2]]]);return F@S,S
def energy(p,F,z,strain_fn):
 _,S=st(p,F,z,strain_fn);v=strain_fn(F)-z[:6];E=np.array([[v[0],v[5],v[4]],[v[5],v[1],v[3]],[v[4],v[3],v[2]]]);return .5*float(np.trace(E@S))
arrpath=Q/'analysis/native531-strict-fp64-replay-001/layers/arrays.npz';ninepath=Q/'analysis/native531-fp64-ninebasis-003/arrays.npz';predpath=Q/'solver/control17-root17-newton34-replay-531/prediction.json';pred=json.loads(predpath.read_text());capture=Path(pred['environment']['NM_FEM_OPERATOR_FD_CAPTURE_PATH']);header,nodes,tets,_=geo.capture_records(capture);par=np.asarray(header['parameters_by_environment'][0],dtype=np.float64);specs={int(x['index']):x for x in header['materials']};papers=[t for t in tets if int(t['environment'])==0 and int(t['material_index']) in (0,1)]
near=[]
for exp in (20,24,28,30,32,36,40,44,48,50):
 mo=mn=0.
 for seed in range(20):
  rng=np.random.default_rng(9700+seed);F=np.eye(3)+rng.normal(size=(3,3))*2.**(-exp);exact=decstrain(F);mo=max(mo,norm(oldstrain(F)[0]-exact));mn=max(mn,norm(stablestrain(F)[0]-exact))
 near.append({'power':exp,'max_old_l2_error_vs_100digit_decimal':mo,'max_stable_l2_error_vs_100digit_decimal':mn})
rng=np.random.default_rng(20261016);rot=[];obj=[];ener=[]
for i in range(64):
 A=rng.normal(size=(3,3));Qm,Rm=np.linalg.qr(A)
 if np.linalg.det(Qm)<0:Qm[:,0]*=-1
 F=np.eye(3)+rng.normal(size=(3,3))*.02
 rot.append({'old':norm(oldstrain(Qm)[0]),'stable':norm(stablestrain(Qm)[0]),'orth':norm(Qm.T@Qm-np.eye(3))})
 obj.append({'old':norm(oldstrain(Qm@F)[0]-oldstrain(F)[0]),'stable':norm(stablestrain(Qm@F)[0]-stablestrain(F)[0]),'delta':norm(oldstrain(F)[0]-stablestrain(F)[0])})
 name='hajali2009_liner_hill_bilinear_fit' if i%2==0 else 'hajali2009_medium_hill_bilinear_fit';mi=next(k for k,v in specs.items() if v['name']==name);off=int(specs[mi]['parameter_offset']);p=par[off:off+28];z=np.zeros(15);H=rng.normal(size=(3,3));P,_=st(p,F,z,stablestrain);pd=float(np.sum(P*H));es=[]
 for eps in (1e-4,1e-5,1e-6,1e-7):
  fd=(energy(p,F+eps*H,z,stablestrain)-energy(p,F-eps*H,z,stablestrain))/(2*eps);es.append({'epsilon':eps,'abs':abs(fd-pd),'rel':abs(fd-pd)/max(abs(pd),1e-300)})
 ener.append(es)
frame=[]
for t in papers:
 active=float(t['center']['dt_seconds_bulk_modulus_kind_frame_active'][3])==1.0;R=geo.captured_material_frame(t,active);mi=int(t['material_index']);off=int(specs[mi]['parameter_offset']);p=par[off:off+28];z=np.zeros(15)
 po,_=st(p,R,z,oldstrain);pn,_=st(p,R,z,stablestrain);wo=po@R.T;wn=pn@R.T
 frame.append({'tet':int(t['tet_id']),'q':t['material_frame_quaternion_xyzw'],'orth':norm(R.T@R-np.eye(3)),'strain_old':norm(oldstrain(R)[0]),'strain_stable':norm(stablestrain(R)[0]),'worldP_old':norm(wo),'worldP_stable':norm(wn),'deltaP':norm(wo-wn)})
report={'schema':'numi.cardboard.native531-stable-green-strain-invariants.v1','classification':'CPU arithmetic invariants only; captured quaternion analyzed; direct native frame-matrix rows are absent, no native noise-floor, production change, calibration, or trajectory qualification','bindings':{'source203_reference_sha256':sha(Q/'snapshots/source-repeated-damped-fallback-203/matter/tools/cardboard_bilinear_fp64_reference.py'),'stable_reference_sha256':sha(Q/'analysis/native531-stable-green-strain-reference-017-001/reference/cardboard_bilinear_fp64_reference.py'),'capture_sha256':sha(capture),'prediction_sha256':sha(predpath),'strict_arrays_sha256':sha(arrpath),'ninebasis_arrays_sha256':sha(ninepath)},'method':{'stable_equation':'D=F-I; E=(D+D^T+D^T D)/2','decimal_oracle':'100-digit Decimal on exact binary64 F entries, strain only','frame_path':'source femMaterialInput(I)=I*R; femMaterialOutput(P)=P*R^T using captured quaternion to reconstruct R in FP64 because this capture omits direct FP32 matrix rows; z=0 isolates reconstructed-frame contribution','energy':'W=0.5 trace(E S); finite differences against P:H for 64 moderate F/H using captured liner/medium laws'},'near_identity':{'samples_per_power':20,'results':near},'rotation_objectivity':{'n':64,'max_rotation_strain_old':max(x['old'] for x in rot),'max_rotation_strain_stable':max(x['stable'] for x in rot),'max_old_objectivity':max(x['old'] for x in obj),'max_stable_objectivity':max(x['stable'] for x in obj),'max_finite_strain_delta':max(x['delta'] for x in obj),'max_rotation_orthogonality':max(x['orth'] for x in rot)},'energy_derivative':{'n':len(ener),'max_relative_by_epsilon':{str(e):max(es[j]['rel'] for es in ener) for j,e in enumerate((1e-4,1e-5,1e-6,1e-7))},'all':ener},'reconstructed_frame_rest_probe':{'paper_tets':len(frame),'rotated_frames':sum(norm(np.asarray(x['q'][:3]))>0 for x in frame),'max_orthogonality':max(x['orth'] for x in frame),'max_frame_strain_old':max(x['strain_old'] for x in frame),'max_frame_strain_stable':max(x['strain_stable'] for x in frame),'max_world_P_old_Pa_at_zero_state':max(x['worldP_old'] for x in frame),'max_world_P_stable_Pa_at_zero_state':max(x['worldP_stable'] for x in frame),'max_world_P_delta_Pa':max(x['deltaP'] for x in frame),'rows':frame},'limitations':['Frame probe reconstructs R in FP64 from captured FP32 quaternions because this capture has no material_frame_rows_float32 field. It measures reconstructed-frame strain only, not native GPU matrix roundoff, actual accepted-history stress, or a simulator rest-state defect.','Decimal oracle tests only strain arithmetic.','Synthetic rotations and energy checks are arithmetic invariants, not native/material qualification.','No native FP32 repeatability or absolute noise floor.']}
(PK/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(PK/'run.json').write_text(json.dumps({'cpu_only':True,'near_identity_samples':200,'rotation_objectivity_samples':64,'energy_samples':64,'captured_frames':len(frame)},indent=2)+'\n')
print(json.dumps({'output':str(PK),'frames':len(frame),'rotated':report['reconstructed_frame_rest_probe']['rotated_frames'],'max_orth':report['reconstructed_frame_rest_probe']['max_orthogonality'],'maxP_old':report['reconstructed_frame_rest_probe']['max_world_P_old_Pa_at_zero_state'],'maxP_stable':report['reconstructed_frame_rest_probe']['max_world_P_stable_Pa_at_zero_state'],'rot_old':report['rotation_objectivity']['max_old_objectivity'],'rot_stable':report['rotation_objectivity']['max_stable_objectivity'],'energy':report['energy_derivative']['max_relative_by_epsilon']},indent=2))
