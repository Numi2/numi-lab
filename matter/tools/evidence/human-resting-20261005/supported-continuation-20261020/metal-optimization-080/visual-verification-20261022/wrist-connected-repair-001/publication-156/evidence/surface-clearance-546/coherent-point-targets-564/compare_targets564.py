"""Compare matched target proposals only; does not choose/admit a mesh."""
from pathlib import Path
import importlib.util,json,sys,numpy as np,hashlib
N=Path(__file__).resolve().parent.parent
s=importlib.util.spec_from_file_location('checker564',N.parent/'contained-clearance-545/run_candidate_checks545.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
o=c.mod(c.OWNER);base=N/'conditioned-connected-refinement-558/refine-all-001/candidate.npz'
direct=N/'conditioned-connected-refinement-558/prepared-001/point-targets/nearest-union-exits.npz'
out=Path(__file__).parent/'targets-001/nearest-union-exits.npz'
paths=[Path(__file__),base,direct,out,c.OWNER,c.R0]
pins={str(p):c.sha(p)for p in paths}
with np.load(base)as z:R0,_=o.compose_source_reference(z,c.R0);X=z['points_world_m']
with np.load(direct)as z:d={k:z[k]for k in z.files}
with np.load(out)as z:u={k:z[k]for k in z.files}
if not np.array_equal(d['query_rows'],u['query_rows']):raise ValueError('different rows')
ids=d['query_rows'];dot=np.sum(d['proposal_direction_world_unit']*d['source_surface_normal_world_unit'],axis=1)
r={'candidate_sha256':c.sha(base),'source_pins':pins,'point_targets':len(ids),
'direct_directions_opposing_registered_normal_count':int((dot<-.05).sum()),'comparison_scope':'target proposals only; no mesh or physics admission',
'native_admission':False,'accepted_physical_steps':0}
for k,z in [('direct',d),('source_outward',u)]:
 goal=z['target_world_f32_m'].astype(float);distance=np.linalg.norm(goal-R0[ids],axis=1)*1000
 step=np.linalg.norm(goal-X[ids].astype(float),axis=1)*1000
 r[k]={'endpoint_step_p0_p50_p90_p100_mm':np.percentile(step,[0,50,90,100]).tolist(),'endpoints_over_4mm_step':int((step>4+1e-6).sum()),
 'source_distance_p0_p50_p90_p100_mm':np.percentile(distance,[0,50,90,100]).tolist(),'endpoints_over_12mm_source_cap':int((distance>12+1e-6).sum()),
 'all185_endpoint_certification':bool(np.all(z['target_exactly_outside_all185']))}
if any(c.sha(p)!=h for p,h in pins.items()):raise ValueError('input changed')
(Path(__file__).parent/'target-comparison.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r),flush=True)
