from pathlib import Path
import json,hashlib,numpy as np
e=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"); p=e/'bone-clearance-fit-041-exact-material-routes'
d=json.loads((p/'route-report.json').read_text()); z=np.load(p/'construction-plane-constraints.npz')
planes=d['construction_planes_world']; n=0
for j,a in enumerate(planes):
 assert np.array_equal(z['plane_point_world_m'][j],a['plane_point_world_m'])
 assert np.array_equal(z['plane_outside_normal_world'][j],a['unit_exterior_normal_world'])
 assert z['plane_target_layer_mask'][j]==a['target_layer_mask']
 assert z['plane_row'][j]==a['row']
 mask=a['target_layer_mask']; layers=[i for i in range(3) if mask&(1<<i)]
 idx=np.flatnonzero(z['constraint_plane_index']==j)
 assert sorted(z['constraint_endpoint_layer'][idx].tolist())==layers
 for k in idx:
  assert z['constraint_endpoint_node_row'][k]==a['row']
  assert np.array_equal(z['constraint_hit_point_world_m'][k],a['plane_point_world_m'])
  assert np.array_equal(z['constraint_outside_normal_world'][k],a['unit_exterior_normal_world'])
  assert z['constraint_margin_m'][k]==a['clearance_m']
  assert z['constraint_endpoint_global_node'][k]==z['plane_start_global_node'][j]+(int(z['constraint_endpoint_layer'][k])-int(z['plane_segment_code'][j]))*2411
 n+=len(idx)
assert n==125
assert np.max(abs(np.linalg.norm(z['plane_outside_normal_world'],axis=1)-1))<1e-14
startdist=np.einsum('ij,ij->i',z['plane_start_world_f32_m']-z['plane_point_world_m'],z['plane_outside_normal_world'])
assert startdist.min()>0
r={'status':'PASS_ADAPTER_EXACT_PLANES_AND_LAYER_EXPANSION','planes':len(planes),'constraints':n,'min_exterior_start_distance_m':float(startdist.min()),'report_sha256':hashlib.sha256((p/'route-report.json').read_bytes()).hexdigest(),'npz_sha256':hashlib.sha256((p/'construction-plane-constraints.npz').read_bytes()).hexdigest(),'limitation':'Construction constraints only; not full surface or physical acceptance'}
(e/'root-independent-route-adapter-041.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

