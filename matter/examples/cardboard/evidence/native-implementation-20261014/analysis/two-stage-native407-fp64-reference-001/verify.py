from pathlib import Path
import hashlib,json,struct,sys,shutil
import numpy as np
B=Path('/Users/home/cardboard-implementation-20261014');O=B/'analysis/two-stage-native407-fp64-reference-001';O.mkdir(exist_ok=False)
old=B/'analysis/dimensionless-material674-point-390-strict-reference-addendum-001';shutil.copy2(old/'bilinear15_reference.py',O/'bilinear15_reference.py');sys.path.insert(0,str(O));import bilinear15_reference as ref
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p401=B/'native/source392-tet858-point-401/output/point.json';p407=B/'native/two-stage-material858-point-407/output/point.json';j=json.loads(p407.read_text());prior=json.loads(p401.read_text())
for k in ['package_sha256','candidate_nodes_sha256','conditioning_manifest_sha256','accepted_state_sha256','medium_asset_sha256','physics_fingerprint','tet_id','dt_s','temperature_K']:assert prior[k]==j[k],k
assert prior['raw_160_f32_words'][3:30]==j['raw_160_f32_words'][3:30]
assert j['projection_converged'] and j['direct_stress_valid'] and j['tangent_valid'] and j['local_updates']==14 and j['terminal_failure']==0
decode=lambda xs:np.asarray([struct.unpack('<f',struct.pack('<I',int(s,16)))[0] for s in xs],dtype=np.float64)
w=decode(j['raw_160_f32_words']);F=(w[3:12]+w[12:21]).reshape(3,3);native=w[56:71]+w[88:103];stress=w[30:39].reshape(3,3);tangent=w[138:147].reshape(3,3)
params=np.asarray(json.loads((B/'analysis/source380-generalized-material-point-001/native_point_input_674_v2.json').read_text())['cases'][0]['parameters'],dtype=np.float32).astype(np.float64);accepted=np.zeros(15);h=np.zeros((3,3));h[0,0]=1;floor=params[26]
z,it,r=ref.project(params[None,:],F[None,:,:],accepted[None,:]);z=z[0]
def metrics(z):
 r=ref.residual(params,F,accepted,z);J=ref.jacobian(params,F,accepted,z);scale=np.maximum(np.maximum(np.abs(accepted),np.abs(z)),floor);delta=np.linalg.solve(J,-r)
 return dict(raw_residual_inf=float(np.max(np.abs(r))),raw_limit=float(2e-8*(1+np.max(np.abs(z)))),scaled_correction_inf=float(np.max(np.abs(delta)/scale))),delta
history=[]
for extra in range(8):
 m,d=metrics(z);history.append(m)
 if m['raw_residual_inf']<=m['raw_limit'] and m['scaled_correction_inf']<=2e-8:break
 z=z+d
else:raise RuntimeError('independent reference did not meet unchanged gates')
def compare(a,b):return dict(abs_l2=float(np.linalg.norm(a-b)),relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)),max_abs=float(np.max(np.abs(a-b))))
native_metrics,_=metrics(native);sref=ref.stress(params,F,z);tref=ref.tangent(params,F,accepted,z,h)
comparisons=dict(state=compare(native,z),stress=compare(stress,sref),tangent_declared_local_Fxx=compare(tangent,tref))
report=dict(status='PASS_NATIVE407_SINGLE_POINT_ANALYTIC_REFERENCE',bindings={str(p):sha(p) for p in [p401,p407,O/'bilinear15_reference.py']},native_iterations=14,native_metal_residual=j['projection_residual'],independent_reference_initial_iterations=int(it[0]),extra_reference_update_metrics=history,native_candidate_promoted_fp64=native_metrics,comparisons=comparisons,limits=['One local material point and one declared analytic direction. Not complete finite-difference/noise-floor, element/assembly or fullcoupon qualification.','The current constitutive law does not depend on Fdot; native point preserves Hdot=H/dt.','Retained candidate still lacks firstfailuregeneration binding.'],physical_steps=0,physical_validation=False)
for k in ['stress','tangent_declared_local_Fxx']:assert comparisons[k]['relative_l2']<=1e-3,(k,comparisons[k])
assert native_metrics['raw_residual_inf']<=native_metrics['raw_limit'] and native_metrics['scaled_correction_inf']<=2e-8
(O/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(O/'vectors.json').write_text(json.dumps(dict(parameters=params.tolist(),accepted=accepted.tolist(),F=F.tolist(),native_state=native.tolist(),reference_state=z.tolist(),native_local_stress=stress.tolist(),reference_local_stress=sref.tolist(),native_local_tangent=tangent.tolist(),reference_local_tangent=tref.tolist()),indent=2)+'\n');(O/'verify.py').write_bytes(Path(__file__).read_bytes());(O/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2)+'\n');print(json.dumps(report,indent=2))
