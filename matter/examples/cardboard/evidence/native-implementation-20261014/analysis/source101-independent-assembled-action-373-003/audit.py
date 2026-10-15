from pathlib import Path
import hashlib,importlib.util,json,datetime
import numpy as np
B=Path('/Users/home/cardboard-implementation-20261014');O=B/'analysis/source101-independent-assembled-action-373-003';O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
refpath=B/'analysis/bilinear15-fp64-replay-267/bilinear15_reference.py'
fdpath=B/'analysis/source101-capture360-directional-fd-001/compare_directional_fd.py'
geometrypath=B/'snapshots/source-boundary-reset-order-101/matter/tools/cardboard_cached_operator_audit.py'
ref=module(refpath,'independent_paper');fd=module(fdpath,'capture_helpers');geo=module(geometrypath,'geometry_helpers')
cap=B/'solver/reset-augmentation-first-failure-capture-360/operator-capture'
assert sha(cap)=='548dce37fbe5a28a609dd6bc45318926072c878774ea0caffa3237d925cebb33'
records=[json.loads(line,parse_int=fd.parse_int) for line in cap.read_text().splitlines() if line.strip()]
h=records[0];ts=sorted([x for x in records if x.get('record')=='tetrahedron' and x['environment']==0],key=lambda x:x['tet_id']);ns=sorted([x for x in records if x.get('record')=='node' and x['environment']==0],key=lambda x:x['node_id'])
assert len(ts)==1920 and len(ns)==882
dv=fd.f32([x['correction_delta_v_m_s'] for x in ns]);mass=fd.f32([x['center']['mass_kg'] for x in ns]);free=np.array([x['center']['constraint_tag']==0 for x in ns]);dt=float(fd.f32([h['dt_seconds_by_environment'][0]])[0])
F=np.stack([fd.pair(t['center'],'F_local') for t in ts]);H=[];rotations=[];inverses=[]
for t in ts:
 inv=fd.f32(t['inverse_rest_rows_per_m']);rot=geo.rotation_from_xyzw(fd.f32(t['material_frame_quaternion_xyzw'])) if t['center']['dt_seconds_bulk_modulus_kind_frame_active'][3]==1 else np.eye(3)
 d=dv[np.array(t['nodes'],int)];H.append(np.column_stack((d[1]-d[0],d[2]-d[0],d[3]-d[0]))@inv@rot);rotations.append(rot);inverses.append(inv)
H=np.array(H);mats=np.array([t['material_index'] for t in ts]);paper=np.flatnonzero(mats<2);glue=np.flatnonzero(mats==2);assert len(paper)==1728 and len(glue)==192
ptab=fd.f32(h['parameters_by_environment'][0]);assert len(ptab)==59
p=np.stack([ptab[int(mats[i])*28:(int(mats[i])+1)*28] for i in paper]);old=np.stack([fd.f32(ts[i]['center']['accepted_state']) for i in paper])
z,it,res,ok,fail=fd.project_resilient(ref,p,F[paper],old);assert ok.all(),fail
refinement=[]
scale=np.array([1e4]*6+[1]*6+[1e4]*3)
for refinement_iteration in range(5):
 r=ref.residual(p,F[paper],old,z)
 corr=np.linalg.solve(ref.jacobian(p,F[paper],old,z),-r[...,None])[...,0]
 state_scale=np.maximum(np.maximum(np.abs(old),np.abs(z)),p[:,26,None]);cnorm=np.max(np.abs(corr)/state_scale,axis=1);selected=np.flatnonzero(cnorm>2e-8)
 refinement.append(dict(iteration=refinement_iteration,rows_above_2e_minus8=int(len(selected)),maximum_scaled_correction=float(cnorm.max())))
 if len(selected)==0:break
 assert refinement_iteration<4,'bounded center correction refinement exhausted'
 baseline=np.max(np.abs(r[selected])*scale,axis=1);pending=np.ones(len(selected),bool);alpha=np.ones(len(selected));next_z=z[selected].copy()
 for trial in range(24):
  candidate=z[selected]+alpha[:,None]*corr[selected]
  nr=np.max(np.abs(ref.residual(p[selected],F[paper][selected],old[selected],candidate))*scale,axis=1)
  accepted=pending&np.isfinite(nr)&(nr<baseline);next_z[accepted]=candidate[accepted];pending[accepted]=False
  if not pending.any():break
  alpha[pending]*=.5
 assert not pending.any(),'bounded center correction refinement failed descent'
 z[selected]=next_z
res=np.max(np.abs(ref.residual(p,F[paper],old,z))*scale,axis=1)
assert np.max(res)<=2e-10
dP=np.empty_like(F);dP[paper]=ref.tangent(p,F[paper],old,z,H[paper])
scale=np.array([1e4]*6+[1]*6+[1e4]*3);correction=np.linalg.solve(ref.jacobian(p,F[paper],old,z),-ref.residual(p,F[paper],old,z)[...,None])[...,0]
mu,lam=ptab[57],ptab[58]
fg=F[glue];hg=H[glue];invf=np.linalg.inv(fg);itf=np.swapaxes(invf,1,2);logj=np.log(np.linalg.det(fg));trace=np.einsum('nij,nji->n',invf,hg)
dP[glue]=mu*hg+(mu-lam*logj)[:,None,None]*(itf@np.swapaxes(hg,1,2)@itf)+lam*trace[:,None,None]*itf
def glue_stress(f):
 invt=np.swapaxes(np.linalg.inv(f),-1,-2);return mu*(f-invt)+lam*np.log(np.linalg.det(f))[...,None,None]*invt
complex_derivative=np.imag(glue_stress(fg.astype(complex)+1e-25j*hg))/1e-25
glue_error=float(np.linalg.norm(complex_derivative-dP[glue])/np.linalg.norm(dP[glue]));assert glue_error<1e-12
forces=np.zeros((len(ts),4,3));gather=np.zeros((len(ns),3))
for i,t in enumerate(ts):
 world=dP[i]@rotations[i].T;force=-float(fd.f32([t['rest_volume_m3']])[0])*world@inverses[i].T
 forces[i,1:]=force.T;forces[i,0]=-np.sum(force.T,axis=0)
 for slot,node in enumerate(t['nodes']):gather[int(node)]+=forces[i,slot]
action=mass[:,None]*dv-dt*dt*gather;action[~free]=0
native_root=B/'analysis/native-operator-term-replay-result-373';receipt=json.loads((native_root/'receipt.json').read_text());assert all(c['exit_code']==0 for c in receipt['commands'].values()) and receipt['prepared_package_unchanged']
native=np.fromfile(native_root/'result/cached-action.float4.bin',dtype='<f4').reshape(2,882,4)[0,:,:3].astype(float)
delta=native-action;err=float(np.linalg.norm(delta));den=float(np.linalg.norm(action));assert den>0
np.savez_compressed(O/'reference-vectors.npz',independent_material_action=dP,independent_element_forces=forces,independent_assembled_action=action,native_action=native,paper_projection_residual=res,paper_final_scaled_correction=np.max(np.abs(correction)/np.maximum(np.maximum(np.abs(old),np.abs(z)),p[:,26,None]),axis=1))
report=dict(schema='numi.cardboard.independent-assembled-action.v1',scope='One rejected source101 capture360 direction, environment0, all1920tets. Independent FP64 Hill/bilinear paper equations and analytic finite starch neo-Hookean law, then independent frame/element/gather/mass assembly. No production tangent coefficients used.',physical_transactions=0,supersedes_correction_gate_only=['source101-independent-assembled-action-373-001','source101-independent-assembled-action-373-002'],correction_erratum='Earlier audits used the fixed FP64 residual weights for their correction metric. This audit uses max(abs(accepted),abs(candidate),parameter26) per state, matching the production scaled-correction coordinates. Earlier action vectors remain retained comparison data, not proof of the production correction gate.',paper_correction_scaling='max(abs(delta_state)/max(abs(old_state),abs(candidate_state),parameter26))',center_correction_refinement=refinement,paper_correction_limit=2e-8,paper_final_correction_gate_passed=bool(np.max(np.abs(correction)/np.maximum(np.maximum(np.abs(old),np.abs(z)),p[:,26,None]))<=2e-8),paper_rows=len(paper),starch_rows=len(glue),paper_projection_success=int(ok.sum()),maximum_paper_scaled_residual=float(np.max(res)),maximum_paper_final_scaled_correction=float(np.max(np.abs(correction)/np.maximum(np.maximum(np.abs(old),np.abs(z)),p[:,26,None]))),maximum_paper_iterations=int(np.max(it)),starch_analytic_vs_complex_step_relative_error=glue_error,dt_seconds=dt,absolute_action_error_Ns=err,reference_action_l2_Ns=den,native_action_l2_Ns=float(np.linalg.norm(native)),relative_assembled_action_error=err/den,comparison_gate=0.01,comparison_within_one_percent=err/den<=.01,absolute_noise_floor_measured=False,general_solver_qualified=False,input_bindings={str(p):sha(p) for p in [cap,refpath,fdpath,geometrypath,native_root/'receipt.json',native_root/'result/cached-action.float4.bin']},finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');(O/'audit.py').write_bytes(Path(__file__).read_bytes())
(O/'manifest.json').write_text(json.dumps(dict(schema='numi.cardboard.audit-evidence.v1',files={str(p.relative_to(O)):sha(p) for p in sorted(O.iterdir()) if p.is_file()}),indent=2)+'\n');print(json.dumps(report,indent=2))
