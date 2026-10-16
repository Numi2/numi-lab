from pathlib import Path
import hashlib,json
import numpy as np
B=Path('/Users/home/cardboard-implementation-20261014');O=B/'analysis/native174-fp64-direction-units-audit-001';assert not O.exists();O.mkdir()
P=B/'solver/final-newton-fd-capture-469/diagnostics/final-newton'
A=B/'analysis/native469-bilinear-layer-replay-002/arrays.npz'
creator=B/'analysis/bilinear-capture-layer-audit-source166-003/candidate/matter/tools/cardboard_bilinear_capture_layers.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(P)=='cf42bdb145317b24a044e8e7561a99dc987df44761b841a08d7f3b28e6cd96d2'
assert sha(A)=='4361b6aa9f0d269a9733b0bd8f508dbc1204624451611816512dad02c303d27c'
text=creator.read_text();assert "reference_directional_action=analytic" in text and "direction=np.stack([f32(t['center']['F_direction_local_per_s']) for t in tets])" in text
f32=lambda x:np.asarray(x,dtype=np.float32).astype(np.float64)
tets={};header=None
for line in P.open():
 r=json.loads(line)
 if r['record']=='header':header=r
 if r['record']=='tetrahedron' and r['environment']==0:
  assert r['tet_id'] not in tets;tets[r['tet_id']]=r
assert set(tets)==set(range(1920))
a=np.load(A);dt=float(f32(header['dt_seconds']));eps=float(f32(header['epsilon_values'][0]))
direction=np.stack([f32(tets[i]['center']['F_direction_local_per_s']) for i in range(1920)])
assert direction.shape==(1920,3,3) and np.isfinite(direction).all()
ideal=eps*dt*direction
assert np.array_equal(ideal,a['s1_ideal_F_delta'])
action=np.einsum('tij,tj->ti',a['reference_C'],direction.reshape(1920,9)).reshape(1920,3,3)
stored=a['reference_directional_action'];assert np.array_equal(action,stored,equal_nan=True)
np.savez_compressed(O/'captured-gradient-direction.npz',tet_ids=np.arange(1920),material=a['material'],F_direction_local_per_s=direction,dt_seconds=np.array(dt),captured_epsilon=np.array(eps))
report={'schema':'numi.cardboard.fp64-direction-units-audit.v1','capture_sha256':sha(P),'source_arrays_sha256':sha(A),'creator_sha256':sha(creator),
 'correct_direction':{'source':'center.F_direction_local_per_s','units':'s^-1','rows':1920,'all_finite':True,'component_min_max':[float(direction.min()),float(direction.max())],'bitwise_ideal_epsilon_dt_deltaF_match':True},
 'incorrect_array':{'name':'reference_directional_action','definition':'C @ F_direction_local_per_s','units':'Pa/s','bitwise_source_reconstruction_match':True,'finite_components':int(np.isfinite(stored).sum()),'nonfinite_components':int((~np.isfinite(stored)).sum()),'finite_component_min_max':[float(np.nanmin(stored)),float(np.nanmax(stored))]},
 'invalid_attempts':['native174-fp64-derivative-qualification-002/run-006 directional sweep','native174-fp64-derivative-qualification-002/run-007 directional sweep','agent small-row profiles derived from same wrong direction'],
 'scope':'Units/input preparation defect in unintegrated diagnostic, not evidence of constitutive nonconvergence. Already-saved center/tangent/MP comparisons do not use the mistaken directional array and retain their separate scope. No production changes.',
 'physical_validation':False}
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');(O/'audit.py').write_bytes(Path(__file__).read_bytes());(O/'manifest.json').write_text(json.dumps({'files_sha256':{p.name:sha(p) for p in sorted(O.iterdir())}},indent=2)+'\n')
print(json.dumps({'manifest_sha256':sha(O/'manifest.json'),'gradient_direction_sha256':sha(O/'captured-gradient-direction.npz'),'direction_range':report['correct_direction']['component_min_max']}))
