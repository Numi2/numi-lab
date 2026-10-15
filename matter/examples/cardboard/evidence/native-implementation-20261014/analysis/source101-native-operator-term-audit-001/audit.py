from pathlib import Path
import datetime,hashlib,importlib.util,json
import numpy as np
B=Path('/Users/home/cardboard-implementation-20261014');I=B/'analysis/native-operator-term-replay-result-373';O=B/'analysis/source101-native-operator-term-audit-001';O.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=B/'snapshots/source-boundary-reset-order-101/matter/tools/cardboard_cached_operator_audit.py'
spec=importlib.util.spec_from_file_location('owner',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
receipt=json.loads((I/'receipt.json').read_text());assert all(r.get('exit_code')==0 for r in receipt['commands'].values());assert receipt['prepared_package_unchanged']
prior=B/'analysis/native-cached-operator-replay-result-368/result';r=I/'result'
for name in ['cached-action.float4.bin','fresh-action.float4.bin','captured-fresh-action.float4.bin']:
 assert (r/name).read_bytes()==(prior/name).read_bytes(),name
capture=B/'solver/reset-augmentation-first-failure-capture-360/operator-capture';h,nodes,tets,integrity=m.capture_records(capture)
assert h['production_inputs_bitwise_unchanged'] and integrity['production_inputs_bitwise_unchanged']
f32=lambda v:np.asarray(v,dtype=np.float32).astype(np.float64)
def metric(a,b):
 a=np.asarray(a);b=np.asarray(b);d=a-b;na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b));nd=float(np.linalg.norm(d))
 return dict(reference_l2=nb,native_l2=na,absolute_l2_error=nd,relative_to_reference=None if nb==0 else nd/nb,maximum_absolute_error=float(np.max(np.abs(d))),units='force_per_velocity_direction_before_dt_squared for elements; Ns for actions',resolved_noise_floor=False)
N=int(h['node_count_per_environment']);T=int(h['tetrahedron_count_per_environment']);E=int(h['environment_count'])
cached=np.fromfile(r/'cached-elements.float4x4.bin',dtype='<f4').reshape(E,T,4,4).astype(np.float64);fresh=np.fromfile(r/'fresh-elements.float4x4.bin',dtype='<f4').reshape(E,T,4,4).astype(np.float64)
mass=np.fromfile(r/'mass-only-action.float4.bin',dtype='<f4').reshape(E,N,4).astype(np.float64)
normal=np.fromfile(r/'cached-action.float4.bin',dtype='<f4').reshape(E,N,4).astype(np.float64)
assert all(np.isfinite(a).all() for a in [cached,fresh,mass,normal]);rows=[];vectors={}
for e in range(E):
 ns=sorted([x for x in nodes if x['environment']==e],key=lambda x:x['node_id']);ts=sorted([x for x in tets if x['environment']==e],key=lambda x:x['tet_id'])
 assert len(ns)==N and len(ts)==T
 dv=f32([x['correction_delta_v_m_s'] for x in ns]);masses=f32([x['center']['mass_kg'] for x in ns]);free=np.array([x['center']['constraint_tag']==0 for x in ns]);dt=float(f32([h['dt_seconds_by_environment'][e]])[0])
 refs=np.zeros((T,4,3));native_gather=np.zeros((N,3));reference_gather=np.zeros((N,3));per=[]
 for t in ts:
  ti=t['tet_id'];conn=np.array(t['nodes'],int);inv=f32(t['inverse_rest_rows_per_m']);vol=float(f32([t['rest_volume_m3']])[0]);c=f32(t['cached_algorithmic_tangent_local']['matrix_row_major_local_dP_dF_Pa_per_dimensionless_F']).reshape(9,9)
  active=t['center']['dt_seconds_bulk_modulus_kind_frame_active'][3]==1;rot=m.rotation_from_xyzw(f32(t['material_frame_quaternion_xyzw'])) if active else np.eye(3)
  d=dv[conn];grad=np.column_stack((d[1]-d[0],d[2]-d[0],d[3]-d[0]))@inv;local=grad@rot;dp=(c@local.reshape(9)).reshape(3,3)@rot.T;force=-vol*dp@inv.T
  refs[ti,1:]=force.T;refs[ti,0]=-np.sum(force.T,axis=0)
  for slot,ni in enumerate(conn):
   native_gather[ni]+=cached[e,ti,slot,:3];reference_gather[ni]+=refs[ti,slot]
  per.append(dict(tet_id=ti,material=t['material_name'],**metric(cached[e,ti,:,:3],refs[ti])))
 massref=masses[:,None]*dv;massref[~free]=0
 assembled=massref-dt*dt*reference_gather;assembled[~free]=0
 gatherref=massref-dt*dt*native_gather;gatherref[~free]=0
 row=dict(environment=e,dt_seconds=dt,element_force_transform=metric(cached[e,:,:,:3],refs),mass_term=metric(mass[e,:,:3],massref),gather_same_native_elements=metric(normal[e,:,:3],gatherref),combined_cached_action=metric(normal[e,:,:3],assembled),worst_element_rows=sorted(per,key=lambda x:x['relative_to_reference'] or 0,reverse=True)[:20])
 rows.append(row)
 (O/f'environment-{e}-per-element.json').write_text(json.dumps(per,indent=2)+'\n')
 vectors.update({f'env{e}_element_reference':refs,f'env{e}_gather_native_elements_fp64':gatherref,f'env{e}_mass_reference':massref,f'env{e}_combined_reference':assembled})
np.savez_compressed(O/'vectors.npz',**vectors)
report=dict(schema='numi.cardboard.native-operator-term-audit.v1',verdict='DIAGNOSTIC_COMPARISON_ONLY',normal_combined_action_bitwise_equal_368=True,physical_transactions=0,independent_scope='Independent FP64 frame/element/gather assembly using exact captured production FP32 tangent coefficients; not an independent constitutive derivative or a measured FP32 noise floor.',input_bindings={str(p):sha(p) for p in [source,capture,I/'manifest.json',I/'receipt.json']},environments=rows,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');(O/'audit.py').write_bytes(Path(__file__).read_bytes());files={str(p.relative_to(O)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(O.rglob('*')) if p.is_file()};(O/'manifest.json').write_text(json.dumps(dict(schema='numi.cardboard.audit-evidence.v1',files=files),indent=2)+'\n');print(json.dumps(report,indent=2))
