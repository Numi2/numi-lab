from pathlib import Path
import numpy as np,json,hashlib,importlib.util,sys
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';r=D/'D/postfit-backtrack-403/run-001/backtrack-373';state=r/'selected-result.json';j=json.loads(state.read_text());c=Path(j['candidate']);base=H/'combined-clearance-394/run-001/candidate.npz';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(c)==j['candidate_sha256']=='442f80600304c9474794aa034f4efe8418c15cfe1256ee364c11ac586853ca98'
a=np.load(base,allow_pickle=False);b=np.load(c,allow_pickle=False)
for k in a.files:
 if k!='points_world_m':assert a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes(),k
anchors=a['coarse_boundary_anchor_rows'];assert len(anchors)==237 and np.array_equal(a['points_world_m'][anchors],b['points_world_m'][anchors])
raw=np.load(H/'retained-gain-continuation-401/run-003/raw-lp-candidate.npz',allow_pickle=False);sup=raw['local_witness_cluster_222_lp_support_rows']
outside=np.setdiff1d(np.arange(len(a['points_world_m'])),sup);assert np.array_equal(a['points_world_m'][outside],b['points_world_m'][outside])
p=D/'D/retained-gain-continuation-401/propose_joint_clearance_witnesses_401.py';sys.path.insert(0,str(p.parent));sp=importlib.util.spec_from_file_location('root403',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
ref,receipt=m.compose_source_reference(a,D/'full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz')
area=m.full_f32_area_gate(b['points_world_m'],a['points_world_m'].astype(float)*1000,a['area_reference_037_points_world_m_f64']*1000,a['source_O_faces'],.1);assert area['pass']
source=np.linalg.norm(b['points_world_m'].astype(float)-ref,axis=1)*1000;step=np.linalg.norm(b['points_world_m'].astype(float)-a['points_world_m'].astype(float),axis=1)*1000;assert source.max()<=12+1e-6 and step.max()<=1+1e-6
q=json.loads(Path(j['point_summary']).read_text());assert q['exact_PLC_surface_pairs']==0 and q['complete_surface_audit']
s=json.loads(Path(j['surface_report']).read_text());assert s['complete'] and len(s['banks'])==185 and sum(x['result']['count']for x in s['banks']if x.get('result'))==3338
assert j['point_merit']['after_inside_pairs']==458 and not j['point_merit']['introduced_typed_pairs']
out=dict(status='ROOT_INDEPENDENT_CONSTRUCTION_PASS',candidate=str(c),candidate_sha256=sha(c),state_sha256=sha(state),base_candidate_sha256=sha(base),source_arrays_bitwise_unchanged=len(a.files)-1,anchors237_unchanged=True,support_nodes=len(sup),changed_nodes=int(np.count_nonzero(step)),outside_support_bitwise_unchanged=True,max_step_mm=float(step.max()),max_source_displacement_mm=float(source.max()),area=area,source_reference_receipt=receipt,surface_pairs=3338,inside_pairs=458,self_pairs=0,point_summary_sha256=sha(j['point_summary']),surface_report_sha256=sha(j['surface_report']),native_admission=False,accepted_physical_steps=0)
(r/'root-independent-construction-verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
