#!/usr/bin/env python3
"""Complete the independent endpoint secant by inserting only the recovered tet1691 plus root."""
import datetime, gzip, hashlib, importlib.util, json, pathlib, math
import numpy as np
BASE=pathlib.Path('/Users/home/cardboard-implementation-20261014')
PKG=BASE/'analysis/source125-medium1691-independent-endpoint-004'
CAP=BASE/'solver/two-stage-first-failure-rich-fd-419/operator-capture'
SWEEP=BASE/'analysis/native-residual-fd-sweep-result-424'
CENTER=BASE/'analysis/operator-fd-e2-adapter-001/coupon419-003/environment-0.npz'
REPORT003=BASE/'analysis/source125-kinematics-and-native424-force-audit-003/report.json'
ROW003=BASE/'analysis/source125-kinematics-and-native424-force-audit-003/per-paper-tet.jsonl.gz'
D=importlib.util.spec_from_file_location('diagnose',PKG/'diagnose.py'); diag=importlib.util.module_from_spec(D); D.loader.exec_module(diag)
a=diag.load(diag.SCRIPT003,'audit003_reassembly'); ref=diag.load(diag.REFPATH,'ref15_reassembly')

records=[json.loads(line) for line in CAP.read_text().splitlines() if line.strip()]
h=records[0]; nodes_json=sorted((r for r in records if r.get('record')=='node' and r.get('environment')==0),key=lambda r:int(r['node_id']))
tets=sorted((r for r in records if r.get('record')=='tetrahedron' and r.get('environment')==0),key=lambda r:int(r['tet_id']))
assert len(nodes_json)==882 and len(tets)==1920
nnode,ntet=882,1920; dt=float(np.float32(h['dt_seconds_by_environment'][0])); eps=float(np.float32(1/16384))
invrest=np.asarray([a.f32(t['inverse_rest_rows_per_m']) for t in tets],dtype=np.float64)
volume=a.f32([t['rest_volume_m3'] for t in tets])
frames32=np.asarray([a.frame_from_xyzw(t['material_frame_quaternion_xyzw']) for t in tets],dtype=np.float32)
frames=frames32.astype(np.float64)
frame_active=np.asarray([int(a.f32([t['center']['dt_seconds_bulk_modulus_kind_frame_active'][3]])[0])==1 for t in tets])
ids=np.asarray([int(t['material_index']) for t in tets],dtype=np.int32)
materials={int(m['index']):m for m in h['materials']}; ptab=a.f32(h['parameters_by_environment'][0])
params=np.zeros((ntet,28),dtype=np.float64); accepted=np.zeros((ntet,15),dtype=np.float64)
for i,t in enumerate(tets):
 m=materials[ids[i]]; off=int(m['parameter_offset']); n=len(m['parameter_names']); params[i,:n]=ptab[off:off+n]
 if ids[i] in (0,1): accepted[i]=a.f32(t['center']['accepted_state'])
paper_ids=np.flatnonzero(np.isin(ids,[0,1])); glue=np.flatnonzero(ids==2)

def load_nodes(stem):
 r=np.fromfile(SWEEP/'result'/f'{stem}-nodes.bin',dtype='<f4'); assert r.size==2*nnode*5*4
 return r.reshape(2,nnode,5,4)[0]
node={'center':load_nodes('sample-0-0'),'plus':load_nodes('sample-6-0'),'minus':load_nodes('sample-6-1')}
F={}; H={}; V={}; U={}; high={}; low={}
for side in node:
 F[side],_,V[side],U[side],high[side],low[side]=a.reconstruct_kinematics(node[side],tets,invrest,frames32,frame_active)
# Retain exact captured center high/low, as in003.
for i,t in enumerate(tets):
 F['center'][i]=a.f32(t['center']['F_local'])+a.f32(t['center'].get('F_local_low',np.zeros((3,3))))
Hdir=(F['plus']-F['minus'])/(2*eps); Vdir=(V['plus']-V['minus'])/(2*eps)

# Read serialized 003 per-paper stresses. Only substitute the independently
# recovered plus stress for the single failed paper endpoint, after strict gates.
rows=[json.loads(line) for line in gzip.open(ROW003,'rt',encoding='utf-8')]
assert len(rows)==1728 and [r['tet_id'] for r in rows]==paper_ids.tolist()
sol=json.loads((PKG/'result.json').read_text()); rec=sol['recovered_endpoint']; assert rec['strict_residual_gate'] and rec['strict_correction_gate']
zplus=np.asarray(rec['state'],dtype=np.float64); p1691=params[1691]; old1691=accepted[1691]
plus_stress_1691=ref.stress(p1691[None,:],F['plus'][1691][None,:,:],zplus[None,:])[0]
P={side:np.full((ntet,3,3),np.nan,dtype=np.float64) for side in ('center','plus','minus')}
for r in rows:
 ti=int(r['tet_id'])
 for side in P:
  value=r['center_projection'] if side=='center' else r['endpoints'][side]
  stress=value.get('independent_stress_Pa') if side!='center' else None
  if side=='center':
   # Center stress is recomputed from the independently projected center state.
   pass
  elif stress is not None: P[side][ti]=np.asarray(stress,dtype=np.float64)
P['plus'][1691]=plus_stress_1691
# Independent center projections for all paper rows are recomputed under unchanged equations/history.
for mi in (0,1):
 group=np.flatnonzero(ids[paper_ids]==mi); globals_=paper_ids[group]
 old=accepted[globals_]; ff=F['center'][globals_]; pp=params[globals_]
 z,its,rn=ref.project(pp,ff,old,maxiter=128)
 S=np.array([1e4]*6+[1]*6+[1e4]*3,dtype=np.float64)
 J=np.stack([ref.jacobian(pp[j],ff[j],old[j],z[j]) for j in range(len(group))])
 rr=ref.residual(pp,ff,old,z); corr=np.linalg.solve(J,-rr[...,None])[...,0]
 dynamic=np.maximum(np.maximum(np.abs(old),np.abs(z)),pp[:,26,None])
 corr_n=np.max(np.abs(corr)/dynamic,axis=1); res_n=np.max(np.abs(rr*S),axis=1)
 assert np.all(res_n<=2e-10) and np.all(corr_n<=2e-8)
 P['center'][globals_]=ref.stress(pp,ff,z)
# Independent analytic glue response on all three exact endpoint F's.
for side in P:
 P[side][glue]=a.glue_stress(F[side][glue],params[glue,1],params[glue,2])

def elem_force(stress): return a.elemental_force(stress,invrest,volume,frames,frame_active)
def scatter(element): return a.scatter_elements(element,tets,nnode)
force={side:elem_force(P[side]) for side in P}
assert all(np.all(np.isfinite(force[s])) for s in force)
free=np.asarray([float(a.f32([n['center']['constraint_tag']])[0])==0 for n in nodes_json])
mass=a.f32([n['center']['mass_kg'] for n in nodes_json]); npz=np.load(CENTER)
ind_A=np.asarray(npz['independent_A_delta_v_impulse_Ns'],dtype=np.float64); native_A=np.asarray(npz['native_production_A_delta_v_impulse_Ns'],dtype=np.float64)
dR_ind={}; dR_native_force={}; dR_native_cap={}; native_force={}; native_residual={}
for side,stem in [('center','sample-0-0'),('plus','sample-6-0'),('minus','sample-6-1')]:
 fb=np.fromfile(SWEEP/'result'/f'{stem}-forces.bin',dtype='<f4').reshape(2,ntet,4,4)[0,:,:,:3].astype(np.float64)
 rb=np.fromfile(SWEEP/'result'/f'{stem}-residual.bin',dtype='<f4').reshape(2,nnode,4)[0,:,:3].astype(np.float64)
 native_force[side]=fb; native_residual[side]=rb
for side in ('plus','minus'):
 dv=V[side]-V['center']
 dR_ind[side]=dt*(scatter(force[side])-scatter(force['center']))-mass[:,None]*dv
 dR_native_force[side]=dt*(scatter(native_force[side])-scatter(native_force['center']))-mass[:,None]*dv
 dR_native_cap[side]=native_residual[side]-native_residual['center']
 for x in (dR_ind[side],dR_native_force[side],dR_native_cap[side]): x[~free]=0
sec_ind=-(dR_ind['plus']-dR_ind['minus'])/(2*eps)
sec_native=-(dR_native_cap['plus']-dR_native_cap['minus'])/(2*eps)
sec_native_formula=-(dR_native_force['plus']-dR_native_force['minus'])/(2*eps)
for x in (sec_ind,sec_native,sec_native_formula): x[~free]=0

# Full center tangent on the realized native endpoint direction.
P_tangent=np.full((ntet,3,3),np.nan)
for mi in (0,1):
 subset=paper_ids[ids[paper_ids]==mi]; pp=params[subset]; old=accepted[subset]; ff=F['center'][subset]
 z,_,_=ref.project(pp,ff,old,maxiter=128)
 P_tangent[subset]=ref.tangent(pp,ff,old,z,Hdir[subset])
P_tangent[glue]=a.glue_tangent_action(F['center'][glue],Hdir[glue],params[glue,1],params[glue,2])
force_tangent=elem_force(P_tangent)
action_tangent=mass[:,None]*Vdir-dt*scatter(force_tangent); action_tangent[~free]=0

report={'schema':'numi.cardboard.source125-medium1691-completed-endpoint-secants.v1','status':'complete_independent_endpoint_secant_recovered_by_continuation_diagnostic_only',
 'qualification':False,'scope':'exact native424 plus/minus node endpoints at epsilon 2^-14; source125 environment0; fixed accepted material history; all1920 tets; CPU independent FP64',
 'bindings':{'source125':str(diag.SOURCE),'capture419_sha256':diag.sha(CAP),'native424_receipt_sha256':diag.sha(SWEEP/'receipt.json'),
  'native424_audit_sha256':diag.sha(BASE/'analysis/native424-residual-fd-sweep-audit-001/report.json'),
  'audit003_report_sha256':diag.sha(REPORT003),'audit003_rows_sha256':diag.sha(ROW003),
  'tet1691_continuation_result_sha256':diag.sha(PKG/'result.json'),'reference_sha256':diag.sha(diag.REFPATH)},
 'tet1691_plus_recovery':{'accepted_history_unchanged':True,'strict_endpoint_residual':rec['scaled_residual_inf'],
  'strict_endpoint_dynamic_correction':rec['dynamic_scaled_correction_inf'],'branch':rec['branch'],
  'center_branch':diag.branch(ref,p1691,old1691,ref.project(p1691[None],F['center'][1691][None],old1691[None],maxiter=128)[0][0]),
  'minus_branch':diag.branch(ref,p1691,old1691,ref.project(p1691[None],F['minus'][1691][None],old1691[None],maxiter=128)[0][0]),
  'native_vs_independent_element_force':a.metrics(native_force['plus'][1691],force['plus'][1691]),
  'independent_plus_stress_Pa':plus_stress_1691.tolist()},
 'complete_assembled_comparison':{
  'all_tets_independent_plus_minus_projection_complete':True,'paper_tet_count':1728,'glue_tet_count':192,'total_tet_count':1920,
  'native_endpoint_force_formula_vs_captured_residual':{
    'plus':a.metrics(dR_native_force['plus'][free],dR_native_cap['plus'][free]),
    'minus':a.metrics(dR_native_force['minus'][free],dR_native_cap['minus'][free])},
  'independent_endpoint_secant_vs_independent_center_A_nominal':a.metrics(sec_ind[free],ind_A[free]),
  'independent_endpoint_secant_vs_native_center_A_nominal':a.metrics(sec_ind[free],native_A[free]),
  'native_captured_endpoint_secant_vs_independent_endpoint_secant':a.metrics(sec_native[free],sec_ind[free]),
  'native_force_formula_secant_vs_independent_endpoint_secant':a.metrics(sec_native_formula[free],sec_ind[free]),
  'native_captured_secant_vs_native_center_A':a.metrics(sec_native[free],native_A[free]),
  'independent_endpoint_secant_vs_center_tangent_realized_direction':a.metrics(sec_ind[free],action_tangent[free]),
  'native_captured_secant_vs_center_tangent_realized_direction':a.metrics(sec_native[free],action_tangent[free]),
  'independent_center_A_vs_native_center_A':a.metrics(ind_A[free],native_A[free]),
  'finite_direction':{'epsilon':eps,'F_realized_direction_vs_dt_times_H_relative':1.054 if False else report003 if False else None},
  'omitted_tet_ids':[]},
 'branch_and_interpretation':{'observed_lambda_active_predicate':'state[12] > accepted[12] remains active at center/minus/recovered plus for tet1691',
  'scope_warning':'The predicate checked here is not a complete branch signature. The continuation crosses a narrow regularized yield region (yield gap falls from center scale 1.24e-3 to plus 4.05e-8).',
  'conclusion':'The independent plus endpoint has a strict root on a continuation-connected branch with unchanged equations and accepted history. The direct elastic-start Newton takes a negative/inactive multiplier step and stalls at scaled residual about90.813; continuation from the converged center requires 4 substeps (8 also succeeds), and from minus requires 8 substeps. The miss is an independent solver initialization/path failure, not evidence that no endpoint root exists. Completing the common rows changes the full assembled secant comparison; this is one finite perturbation and is not a derivative qualification or a pure FP32 noise-floor estimate.'},
 'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
# Replace placeholder with measured direction metrics summarized from sealed003.
base_report=json.loads(REPORT003.read_text())
report['complete_assembled_comparison']['finite_direction']={
 'epsilon':eps,
 'realized_dv_vs_nominal_correction_relative':base_report['assembled_residual_and_fd']['realized_dv_direction_vs_nominal_correction']['relative_to_reference'],
 'realized_dF_vs_nominal_dt_times_captured_H_relative':base_report['assembled_residual_and_fd']['realized_dF_direction_vs_nominal_dt_times_captured_H']['relative_to_reference'],
 'observed_plastic_multiplier_predicate_flips_on_valid_rows':0}
(PKG/'completed-secants.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'tet1691':report['tet1691_plus_recovery'],
 'secant':report['complete_assembled_comparison'],'output':str(PKG/'completed-secants.json')},indent=2,allow_nan=False))
