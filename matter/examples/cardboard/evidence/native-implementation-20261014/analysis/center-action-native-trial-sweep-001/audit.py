from pathlib import Path
import argparse,hashlib,importlib.util,json,struct
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('capture',type=Path);ap.add_argument('--center',type=Path,required=True);ap.add_argument('--reader',type=Path,required=True);ap.add_argument('--center-reader',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
spec=importlib.util.spec_from_file_location('strict_reader',a.reader);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);c=r.Capture(a.capture)
pre='candidate_before_private_contact_rebuild';full='full_residual_before_armijo_measure';measure='armijo_measurement_before_decision'
spec2=importlib.util.spec_from_file_location('center_reader',a.center_reader);cr=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(cr);center_capture=cr.Capture(a.center)
join_keys=('source_physics_fingerprint','device_program_fingerprint','root','iteration','control_step','physics_substep','physics_substeps','microtick','dt_seconds_f32_bits_u32','environment_count','node_count','tetrahedron_count','cohesive_face_count','contact_pair_count','deformable_contact_capacity')
for key in join_keys:assert c.manifest[key]==center_capture.manifest[key],key
assert center_capture.manifest['rhs_preserved_before_fgmres'] and 'center_operator_action' in center_capture.manifest
center_names={'full_rhs_before_fgmres','center_production_jdelta_action','center_operator_diagnostic_status','fgmres_vector_row_map','contact_history_used_by_newton_center','deformable_history_used_by_newton_center'}
def raw(name,phase):return center_capture.raw(name) if name in center_names else c.raw(name,phase)
# Raw bytes must bind the candidate and correction, not only scalar generation labels.
for center_name,trial_name in [('fem_candidate_nodes','fem_linearization_nodes'),('fem_solution_correction','fem_solution_correction'),('material_history_accepted','material_history_accepted'),('tetrahedra_candidate','tetrahedra_candidate')]:
 assert bytes(center_capture.raw(center_name))==c.raw(trial_name,'candidate_before_private_contact_rebuild'),(center_name,trial_name)

def x4(name,phase):
 m=center_capture.arrays[name] if name in center_names else c.arrays[phase,name];assert m['dtype']=='float32x4' and m['element_bytes']==16
 return np.frombuffer(raw(name,phase),dtype='<f4').reshape(-1,4).astype(np.float64)
def stats(v):
 v=np.asarray(v,dtype=np.float64);assert np.isfinite(v).all()
 return dict(count=v.size,norm=float(np.linalg.norm(v)),max_abs=float(np.max(np.abs(v))) if v.size else 0.)
def hraw(name,phase):return hashlib.sha256(c.raw(name,phase)).hexdigest()
n=c.manifest['unknown_count'];E=c.manifest['environment_count'];N=c.manifest['node_count'];nodes=np.frombuffer(c.raw('fem_linearization_nodes',pre),dtype='<f4').reshape(-1,20);assert len(nodes)==E*N
row=np.frombuffer(raw('fgmres_vector_row_map',pre),dtype='<u4').reshape(-1,4);assert row.shape==(n,4)
b=x4('full_rhs_before_fgmres',pre);j=x4('center_production_jdelta_action',pre);rt=x4('residual_before_armijo_measure',full);assert b.shape==j.shape==rt.shape==(n,4)
delta=x4('fem_solution_correction',pre);assert delta.shape==(n,4)
alpha=float(x4('fem_armijo_metrics',pre)[0,2]);assert 0<alpha<=1
status=raw('center_operator_diagnostic_status',pre);assert len(status)==E*48
statuses=[]
for off in range(0,len(status),48):
 u=struct.unpack_from('<8I',status,off);f=struct.unpack_from('<4f',status,off+32);statuses.append(dict(code=u[0],indices=list(u[1:]),diagnostics=list(f)));assert u[0]==0
histories={}
for center,trial in [('contact_history_used_by_newton_center','contact_histories_used_by_private_trial'),('deformable_history_used_by_newton_center','deformable_histories_used_by_private_trial')]:
 x,y=bytes(raw(center,pre)),c.raw(trial,pre);histories[center]=dict(center_sha256=hashlib.sha256(x).hexdigest(),trial_sha256=hashlib.sha256(y).hexdigest(),bytes_equal=x==y,bytes=len(x));assert x==y
# Explicit row identity makes reserve padding and generalized scalar lanes visible.
mask=np.zeros((n,4),bool);groups={}
for i,(env,block,local,source) in enumerate(row):
 assert env<E
 if block==1:
  assert local<N and i==env*N+local
  free=nodes[i,11]<=.5 and nodes[i,3]>0
  if free:mask[i,:3]=True
 elif block==2:assert local<N and i==E*N+env*N+local
 elif block==3:mask[i,0]=True
 else:raise ValueError('unknown row block')
# Mechanical norm used by the full-contact Armijo path; fields are separately shown.
res0=-b;trial_physical=-rt;observed=trial_physical-res0;predicted=alpha*j;error=observed-predicted
for label,sel in [('armijo_free_mechanical_and_rigid',mask),('all_exported_lanes',np.ones_like(mask)),('fem_xyz',np.repeat((row[:,1]==1)[:,None],4,axis=1)&np.array([1,1,1,0],bool)),('field_channels',np.repeat((row[:,1]==2)[:,None],4,axis=1)),('free_rigid_generalized',np.repeat((row[:,1]==3)[:,None],4,axis=1)&np.array([1,0,0,0],bool))]:
 obs,pred=observed[sel],predicted[sel];err=error[sel];den=float(np.linalg.norm(pred));groups[label]=dict(center_residual=stats(res0[sel]),trial_residual=stats(trial_physical[sel]),production_action=stats(j[sel]),observed_increment=stats(obs),linear_predicted_increment=stats(pred),linear_prediction_error=stats(err),relative_to_predicted_increment=float(np.linalg.norm(err))/den if den else None,cosine=float(np.dot(obs,pred)/(np.linalg.norm(obs)*np.linalg.norm(pred))) if np.linalg.norm(obs)*np.linalg.norm(pred)>0 else None)
actual_nodes=np.frombuffer(c.raw('fem_armijo_candidate_nodes',pre),dtype='<f4').reshape(-1,20).astype(np.float64);center=nodes.astype(np.float64);dv=actual_nodes[:,4:7]-center[:,4:7];expected_dv=alpha*delta[:E*N,:3];dx=actual_nodes[:,:3]-center[:,:3];dt=float(c.manifest['dt_seconds']);du=actual_nodes[:,16:19]-center[:,16:19]
report=dict(schema='numi.cardboard.center_action_linear_prediction.sign_corrected.v1',capture=c.summary(),center_capture=center_capture.summary(),center_reader_sha256=hashlib.sha256(a.center_reader.read_bytes()).hexdigest(),joined_generation_keys=list(join_keys),reader_sha256=hashlib.sha256(a.reader.read_bytes()).hexdigest(),alpha=alpha,dt=dt,operator_statuses=statuses,history_identity=histories,groups=groups,representability=dict(velocity_increment=stats(dv),alpha_delta=stats(expected_dv),velocity_rounding=stats(dv-expected_dv),position_increment=stats(dx),dt_velocity_increment=stats(dt*dv),position_vs_rate_increment=stats(dx-dt*dv),reference_displacement_increment=stats(du),reference_vs_rate_increment=stats(du-dt*dv)),limitations=['Both center and private buffers are Newton right-hand sides b=-R; physical residuals are negated before comparing, so observed physical increment is (-b_trial)-(-b_center). One selected finite trial is observational only; no epsilon sweep or complete derivative qualification.','The exported all-lane statistics include fixed/reserved rows. Armijo active-row mask is reported independently.','Contact branch/feature changes and endpoint roundoff require independent follow-up; mismatch alone does not identify the first faulty layer.'],solver_correctness_qualified=False,physical_assembly_qualified=False)
assert not a.output.exists();a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(alpha=alpha,active=groups['armijo_free_mechanical_and_rigid'],representability=report['representability']),indent=2))
