from pathlib import Path
import numpy as np,json,hashlib,importlib.util,sys
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';r=D/'D/local-two-parent-retriangulation-405/run-011-replay-405-four-flips';c=r/'candidate.npz'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(c)=='6c115d5beef83989bed2af6309771b8d34cece7e30eaaf3f99f397ee3a304fda'
a=np.load(H/'fine-point-basis-enrichment-405/run-001/candidate.npz',allow_pickle=False);b=np.load(c,allow_pickle=False)
changed=[k for k in a.files if a[k].dtype!=b[k].dtype or a[k].shape!=b[k].shape or a[k].tobytes()!=b[k].tobytes()]
assert set(changed)=={'source_O_faces','refinement_face_parent_face_row','refinement_face_parent_barycentric','refinement_face_parent_barycentric_semantics','area_reference_037_source_O_faces'}
assert np.array_equal(a['points_world_m'],b['points_world_m'])
base=np.load(D/'D/postfit-backtrack-403/run-001/backtrack-373/pass-002/candidate.npz')
anchors=b['typed_coarse_anchor_output_rows_018'];assert len(anchors)==237 and np.array_equal(base['points_world_m'][anchors],b['points_world_m'][anchors])
faces=b['source_O_faces'];old=a['source_O_faces'];rows=np.flatnonzero(np.any(faces!=old,axis=1));assert len(rows)==8
assert np.array_equal(faces,b['area_reference_037_source_O_faces'])
def edges(f):
 e=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1);return np.unique(e,axis=0,return_counts=True)
e,n=edges(faces);eo,no=edges(old);assert n.max()==2 and np.array_equal(e[n==1],eo[no==1]);assert len(np.unique(np.sort(faces,axis=1),axis=0))==len(faces)
ip=b['refinement_face_parent_ancestry_indptr'];ix=b['refinement_face_parent_ancestry_face_rows'];assert len(ip)==len(faces)+1 and ip[0]==0 and ip[-1]==len(ix) and np.all(np.diff(ip)>0)
p=D/'D/retained-gain-continuation-401/propose_joint_clearance_witnesses_401.py';sys.path.insert(0,str(p.parent));sp=importlib.util.spec_from_file_location('rootTopo405',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
ref,receipt=m.compose_source_reference(b,D/'full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz')
area=m.full_f32_area_gate(b['points_world_m'],b['points_world_m'].astype(float)*1000,b['area_reference_037_points_world_m_f64']*1000,faces,.1);assert area['pass']
source=np.linalg.norm(b['points_world_m'].astype(float)-ref,axis=1)*1000;assert source.max()<=12+1e-6
ap=D/'D/postfit-backtrack-405-topology-full-audits/run-001';j=json.load(open(ap/'audit-130/summary.json'));s=json.load(open(ap/'audit-159/report.json'))
assert j['exact_PLC_surface_pairs']==0 and j['complete_surface_audit'];assert s['complete'] and len(s['banks'])==185 and sum(v['result']['count']for v in s['banks']if v.get('result'))==3324
prev=json.load(open(D/'D/postfit-backtrack-404/run-002/backtrack-373/pass-003/audit-130/summary.json')) if (D/'D/postfit-backtrack-404/run-002/backtrack-373/pass-003/audit-130/summary.json').exists() else None
def pairs(j):return {(int(w['output_node']),int(w['bank_body']),int(w['bank_stable_id'])) for w in j['inside_witnesses']}
if prev is None:
 st=json.load(open(D/'D/postfit-backtrack-404/run-002/backtrack-373/selected-result.json'));prev=json.load(open(st['point_summary']))
introduced=sorted(pairs(j)-pairs(prev));removed=sorted(pairs(prev)-pairs(j))
out=dict(status='ROOT_INDEPENDENT_CONSTRUCTION_CHECK_PASS',candidate=str(c),candidate_sha256=sha(c),changed_existing_arrays=changed,unchanged_array_count=len(a.files)-len(changed),changed_face_rows=rows.tolist(),points_bitwise_unchanged_from_unshrunk405=True,typed_anchors_unchanged=237,boundary_edges_unchanged=int(np.count_nonzero(n==1)),manifold=True,no_duplicate_faces=True,matched037_faces=True,parent_union_csr_valid=True,area=area,max_source_displacement_mm=float(source.max()),self_pairs=0,surface_pairs=3324,inside_pairs=len(pairs(j)),introduced_inside_pairs_vs404=introduced,removed_inside_pairs_vs404=removed,point_summary_sha256=sha(ap/'audit-130/summary.json'),surface_report_sha256=sha(ap/'audit-159/report.json'),accepted_physical_steps=0,native_admission=False,physical_transfer_pending=True)
(r/'root-independent-construction-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in ['removed_inside_pairs_vs404','area'] }));print('removed',len(removed),'area',area)
