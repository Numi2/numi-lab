import json, hashlib, sys, time
from pathlib import Path
import numpy as np
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
PK=Q/'analysis/native531-stable-green-strain-invariants-018-002'
PK.mkdir(parents=True,exist_ok=True)
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
arrpath=Q/'analysis/native531-strict-fp64-replay-001/layers/arrays.npz';ninepath=Q/'analysis/native531-fp64-ninebasis-003/arrays.npz';arr=np.load(arrpath);nine=np.load(ninepath);paper_ids=nine['paper_tet_ids'];paper_index={int(v):i for i,v in enumerate(paper_ids)};framepath=Q/'analysis/native531-recovered-fp32-material-frames-001/frames.npz';frame_archive=np.load(framepath);frame_by_id={int(tid):frame_archive['frame_rows_f32'][i].astype(np.float64) for i,tid in enumerate(frame_archive['tet_id'])};predpath=Q/'solver/control17-root17-newton34-replay-531/prediction.json';pred=json.loads(predpath.read_text());capture=Path(pred['environment']['NM_FEM_OPERATOR_FD_CAPTURE_PATH']);header,nodes,tets,_=geo.capture_records(capture);par=np.asarray(header['parameters_by_environment'][0],dtype=np.float64);specs={int(x['index']):x for x in header['materials']};papers=[t for t in tets if int(t['environment'])==0 and int(t['material_index']) in (0,1)]
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
 active=float(t['center']['dt_seconds_bulk_modulus_kind_frame_active'][3])==1.0;R=frame_by_id[int(t['tet_id'])] if active else np.eye(3);Qrot=geo.rotation_from_xyzw(geo.captured_f32_array(t['material_frame_quaternion_xyzw'],(4,),f"tet{t['tet_id']} q")) if active else np.eye(3);mi=int(t['material_index']);off=int(specs[mi]['parameter_offset']);p=par[off:off+28];z=np.zeros(15)
 po,_=st(p,R,z,oldstrain);pn,_=st(p,R,z,stablestrain);pideal,_=st(p,Qrot,z,stablestrain);wo=po@R.T;wn=pn@R.T;wideal=pideal@Qrot.T;oh=np.asarray(arr['accepted_paper_history'][paper_index[int(t['tet_id'])]],dtype=np.float64);pold,_=st(p,R,oh,stablestrain);poldideal,_=st(p,Qrot,oh,stablestrain);world_oldstate=pold@R.T;world_oldstate_ideal=poldideal@Qrot.T
 frame.append({'tet':int(t['tet_id']),'q':t['material_frame_quaternion_xyzw'],'orth':norm(R.T@R-np.eye(3)),'R_vs_quaternion_rotation':norm(R-Qrot),'strain_old':norm(oldstrain(R)[0]),'strain_stable':norm(stablestrain(R)[0]),'worldP_old':norm(wo),'worldP_stable':norm(wn),'worldP_ideal':norm(wideal),'deltaP':norm(wo-wn),'worldP_stable_zero_minus_ideal':norm(wn-wideal),'worldP_stable_oldstate':norm(world_oldstate),'worldP_stable_oldstate_ideal':norm(world_oldstate_ideal),'worldP_oldstate_rounding_delta':norm(world_oldstate-world_oldstate_ideal)})
report={'schema':'numi.cardboard.native531-stable-green-strain-invariants.v2','classification':'CPU arithmetic invariants only; captured quaternion analyzed; direct native frame-matrix rows are absent, no native noise-floor, production change, calibration, or trajectory qualification','bindings':{'source203_reference_sha256':sha(Q/'snapshots/source-repeated-damped-fallback-203/matter/tools/cardboard_bilinear_fp64_reference.py'),'stable_reference_sha256':sha(Q/'analysis/native531-stable-green-strain-reference-017-001/reference/cardboard_bilinear_fp64_reference.py'),'capture_sha256':sha(capture),'recovered_frame_rows_sha256':sha(framepath),'prediction_sha256':sha(predpath),'strict_arrays_sha256':sha(arrpath),'ninebasis_arrays_sha256':sha(ninepath)},'method':{'stable_equation':'D=F-I; E=(D+D^T+D^T D)/2','decimal_oracle':'100-digit Decimal on exact binary64 F entries, strain only','frame_path':'source femMaterialInput(I)=I*R; femMaterialOutput(P)=P*R^T using exact recovered FP32 rows keyed by tet_id; also compares with ideal quaternion-derived orthogonal Q. Both zero-state and exact accepted-history z are reported, but neither is a coupon state at rest.','energy':'W=0.5 trace(E S); finite differences against P:H for 64 moderate F/H using captured liner/medium laws'},'near_identity':{'samples_per_power':20,'results':near},'rotation_objectivity':{'n':64,'max_rotation_strain_old':max(x['old'] for x in rot),'max_rotation_strain_stable':max(x['stable'] for x in rot),'max_old_objectivity':max(x['old'] for x in obj),'max_stable_objectivity':max(x['stable'] for x in obj),'max_finite_strain_delta':max(x['delta'] for x in obj),'max_rotation_orthogonality':max(x['orth'] for x in rot)},'energy_derivative':{'n':len(ener),'max_relative_by_epsilon':{str(e):max(es[j]['rel'] for es in ener) for j,e in enumerate((1e-4,1e-5,1e-6,1e-7))},'all':ener},'native_frame_rest_probe':{'paper_tets':len(frame),'rotated_frames':sum(norm(np.asarray(x['q'][:3]))>0 for x in frame),'max_orthogonality':max(x['orth'] for x in frame),'max_R_vs_quaternion_rotation':max(x['R_vs_quaternion_rotation'] for x in frame),'max_frame_strain_old':max(x['strain_old'] for x in frame),'max_frame_strain_stable':max(x['strain_stable'] for x in frame),'max_world_P_old_Pa_at_zero_state':max(x['worldP_old'] for x in frame),'max_world_P_stable_Pa_at_zero_state':max(x['worldP_stable'] for x in frame),'max_world_P_delta_Pa':max(x['deltaP'] for x in frame),'max_zero_state_rounding_stress_delta_Pa':max(x['worldP_stable_zero_minus_ideal'] for x in frame),'max_accepted_history_world_stress_Pa':max(x['worldP_stable_oldstate'] for x in frame),'max_accepted_history_rounding_delta_Pa':max(x['worldP_oldstate_rounding_delta'] for x in frame),'rows':frame},'limitations':['Frame archive provides recovered native float32 rows but is a separately hash-bound recovery; compare exact rows with ideal quaternion rotation. Probe uses F_local=R and reports z=0 and accepted-history z separately. This does not establish a trajectory rest state or solver error.','Decimal oracle tests only strain arithmetic.','Synthetic rotations and energy checks are arithmetic invariants, not native/material qualification.','No native FP32 repeatability or absolute noise floor.']}
(PK/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(PK/'run.json').write_text(json.dumps({'cpu_only':True,'near_identity_samples':200,'rotation_objectivity_samples':64,'energy_samples':64,'captured_frames':len(frame)},indent=2)+'\n')
print(json.dumps({'output':str(PK),'frames':len(frame),'rotated':report['native_frame_rest_probe']['rotated_frames'],'max_orth':report['native_frame_rest_probe']['max_orthogonality'],'max_RQ':report['native_frame_rest_probe']['max_R_vs_quaternion_rotation'],'maxP_old':report['native_frame_rest_probe']['max_world_P_old_Pa_at_zero_state'],'maxP_stable':report['native_frame_rest_probe']['max_world_P_stable_Pa_at_zero_state'],'maxPdeltaZero':report['native_frame_rest_probe']['max_zero_state_rounding_stress_delta_Pa'],'maxPdeltaHistory':report['native_frame_rest_probe']['max_accepted_history_rounding_delta_Pa'],'rot_old':report['rotation_objectivity']['max_old_objectivity'],'rot_stable':report['rotation_objectivity']['max_stable_objectivity'],'energy':report['energy_derivative']['max_relative_by_epsilon']},indent=2))
