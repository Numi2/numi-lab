from pathlib import Path
import json,csv,hashlib,runpy
from collections import Counter
import numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=E/'route-audit-counter-fix-030';B=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
reports={n:json.loads((D/n/'tet-edge-report.json').read_text())for n in ['baseline035','candidate014','candidate027']}
def groups(r):return {(w['edge_index'],w['body'],w['bone_stable_id'])for w in r['witnesses']}
g={n:groups(r)for n,r in reports.items()}
X=np.load(E/'o12-outward-repair-027-f32-gated/candidate.npz')['points_world_f32']
base=np.load(D/'baseline-035.npz')['points_world_f32'];ap=np.load(E/'bone-clearance-feasibility-033/world-bone-clearance-constraints.npz')
changed=np.any(X.view(np.uint32)!=base.view(np.uint32),axis=1)
r=reports['candidate027'];assert r['counts']['proper_hits']==len(r['witnesses'])==3732
assert len({w['edge_index']for w in r['witnesses']})==1533 and len(g['candidate027'])==1787
for w in r['witnesses']:assert np.array_equal(ap['global_nodes'][w['endpoint_local_nodes']],w['endpoint_global_nodes'])
raw=np.loadtxt(B/'exact-emitted.triangles.csv',delimiter=',',skiprows=1)
assert raw.shape==(469136,23)
assert sha(B/'exact-emitted.triangles.csv')=='0d3418064af1d4f34d19a1ec91992afceacaaa70b6c982d4a2637ded1d829eac'
tri=raw[:,5:14].astype(np.float32).astype(float).reshape(-1,3,3);ids=raw[:,1:5].astype(np.int64)
bodies={int(a['body_index']):a for a in csv.DictReader((B/'exact-emitted.bodies.csv').open())}
exact=runpy.run_path(str(D/'canonical/matter/root_exact_segment_witness.py'))['classify_segment_triangle']
samples=[]
for wi in np.linspace(0,len(r['witnesses'])-1,48,dtype=int):
 w=r['witnesses'][wi];body=bodies[w['body']]
 pos=np.array([float(body['position_'+c+'_m'])for c in 'xyz'],np.float32).astype(float)
 quat=np.array([float(body['orientation_q'+c])for c in 'xyzw'],np.float32).astype(float);quat[:3]*=-1
 M=np.column_stack([v+2*np.cross(quat[:3],np.cross(quat[:3],v)+quat[3]*v)for v in np.eye(3)])
 off=np.array([float(body['com_offset_'+c+'_m'])for c in 'xyz'])
 ends=(X[w['endpoint_local_nodes']].astype(float)-pos)@M.T-off
 mask=(ids[:,0]==w['body'])&(ids[:,1]==w['bone_stable_id'])&(ids[:,3]==w['triangle_stable_id'])
 assert mask.sum()==1
 hit=exact(ends[0],ends[1],tri[mask][0])
 assert hit['classification']==w['classification']=='proper_interior_hit'
 assert abs(hit['t']-w['exact_result']['t'])<1e-12
 samples.append({'witness_index':int(wi),'edge_index':w['edge_index'],'body':w['body'],'bone':w['bone_stable_id'],'triangle':w['triangle_stable_id'],'t':hit['t']})
unchanged_groups={(w['edge_index'],w['body'],w['bone_stable_id'])for w in r['witnesses']if not changed[w['endpoint_local_nodes']].any()}
banks=Counter()
for edge,body,bone in g['candidate027']:banks[(body,bone)]+=1
out={'status':'48_INDEPENDENT_EXACT_EDGE_WITNESSES_PASS_COMPARISON_RETAINS_CLEARANCE_FAILURE','changed_nodes':int(changed.sum()),'all_candidate027_crossing_groups':len(g['candidate027']),'groups_with_both_endpoints_unchanged_from_baseline':len(unchanged_groups),'baseline_to_027':{'removed':len(g['baseline035']-g['candidate027']),'new':len(g['candidate027']-g['baseline035']),'retained':len(g['candidate027']&g['baseline035'])},'014_to_027':{'removed':len(g['candidate014']-g['candidate027']),'new':len(g['candidate027']-g['candidate014']),'retained':len(g['candidate027']&g['candidate014'])},'groups_by_bank':[{'body':b,'bone_stable_id':s,'groups':n}for (b,s),n in banks.most_common()],'independent_samples':samples,'accepted_physical_steps':0,'physical_qualification':False,'inputs':{str(D/n/'tet-edge-report.json'):sha(D/n/'tet-edge-report.json')for n in reports},'limitations':['Exact edge witnesses are genuine crossings, but count is not penetration volume or physiological impact.','Node containment plus edge screening is not full face/tet solid certification.','As-run029proper_hitscounterwasinflated; individualwitnesses/uniqueedgegroupsremainunchanged.']}
(D/'root-independent-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items()if k not in ['independent_samples','inputs','groups_by_bank']},indent=2))

