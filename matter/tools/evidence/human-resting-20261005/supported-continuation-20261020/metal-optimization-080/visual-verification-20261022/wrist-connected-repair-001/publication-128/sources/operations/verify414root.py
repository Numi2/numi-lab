from pathlib import Path
import numpy as np,json,hashlib,importlib.util,sys
D=Path('/Users/numi/Work/numi-human-connected-surface-069');r=D/'D/local-two-parent-retriangulation-414/run-001-pass002';c=r/'search/selected/candidate.npz';p0=D/'D/postfit-backtrack-413/run-001/backtrack-373/pass-002/candidate.npz';base=D/'D/postfit-backtrack-410/run-001/backtrack-373/pass-002/candidate.npz'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(c)=='3221a63a44cd76d25374fd5be42f25ec8c3ef0f87a80ca6415d077e6add92f01'
a=np.load(p0);b=np.load(c);z=np.load(base)
assert set(a.files)==set(b.files)
changed=[k for k in a.files if a[k].dtype!=b[k].dtype or a[k].shape!=b[k].shape or a[k].tobytes()!=b[k].tobytes()]
assert set(changed)=={'source_O_faces','refinement_face_parent_face_row','refinement_face_parent_barycentric','area_reference_037_source_O_faces','refinement_face_parent_ancestry_indptr','refinement_face_parent_ancestry_face_rows'}
assert a['points_world_m'].tobytes()==b['points_world_m'].tobytes()
anchors=b['typed_coarse_anchor_output_rows_018'];assert len(anchors)==237 and z['points_world_m'][anchors].tobytes()==b['points_world_m'][anchors].tobytes()
F=b['source_O_faces'];G=a['source_O_faces'];rows=np.flatnonzero(np.any(F!=G,axis=1));assert 1<=len(rows)<=4
assert np.array_equal(F,b['area_reference_037_source_O_faces'])
def edges(f):
 directed=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);e=np.sort(directed,axis=1);u,inv,n=np.unique(e,axis=0,return_inverse=True,return_counts=True);signed=np.bincount(inv,weights=np.where(directed[:,0]<directed[:,1],1,-1));return u,n,signed
e,n,sg=edges(F);eo,no,sgo=edges(G);assert n.max()==2 and np.all(sg[n==2]==0) and np.array_equal(e[n==1],eo[no==1])
assert len(np.unique(np.sort(F,axis=1),axis=0))==len(F)
ip=b['refinement_face_parent_ancestry_indptr'];ix=b['refinement_face_parent_ancestry_face_rows'];oldip=a['refinement_face_parent_ancestry_indptr'];oldix=a['refinement_face_parent_ancestry_face_rows']
assert len(ip)==len(F)+1 and ip[0]==0 and ip[-1]==len(ix) and np.all(np.diff(ip)>0) and ix.min()>=0
for i in range(len(F)):
 before=set(oldix[oldip[i]:oldip[i+1]].tolist());after=set(ix[ip[i]:ip[i+1]].tolist())
 assert (before<=after if i in set(rows)else before==after),i
p=D/'D/retained-gain-continuation-401/propose_joint_clearance_witnesses_401.py';sys.path.insert(0,str(p.parent));sp=importlib.util.spec_from_file_location('rootTopo414',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
ref,refreceipt=m.compose_source_reference(b,D/'full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz')
area=m.full_f32_area_gate(b['points_world_m'],z['points_world_m'].astype(float)*1000,b['area_reference_037_points_world_m_f64']*1000,F,.1);assert area['pass']
source=np.linalg.norm(b['points_world_m'].astype(float)-ref,axis=1)*1000;step=np.linalg.norm(b['points_world_m'].astype(float)-z['points_world_m'],axis=1)*1000;assert source.max()<=12+1e-6 and step.max()<=1+1e-6
ap=r/'full-audits';j=json.load(open(ap/'audit-130/summary.json'));s=json.load(open(ap/'audit-159/report.json'))
assert j['candidate_sha256']==sha(c) and j['exact_PLC_surface_pairs']==0 and j['complete_surface_audit'];assert s['candidate_sha256']==sha(c) and s['complete'] and len(s['banks'])==185 and sum((v.get('result')or{}).get('count',0)for v in s['banks'])==3254
prev=json.load(open(base.parent/'audit-130/summary.json'))
def pairs(j):return {(int(w['output_node']),int(w['bank_body']),int(w['bank_stable_id']))for w in j['inside_witnesses']}
introduced=sorted(pairs(j)-pairs(prev));removed=sorted(pairs(prev)-pairs(j));assert len(pairs(j))==391 and not introduced
def depth(j):return [float(w['nearest_distance_m'])for w in j['inside_witnesses']]
assert max(depth(j))<=max(depth(prev)) and sum(v*v for v in depth(j))<=sum(v*v for v in depth(prev))
oldsurf=json.load(open(base.parent/'audit-159/report.json'))
def surface(j):return {(int(v['body']),int(v['stable_bone_id']),int(x[0]),int(x[1]))for v in j['banks']for x in (v.get('result')or{}).get('triangle_pairs',[])}
oldset=surface(oldsurf);newset=surface(s);unchanged_old={v for v in oldset if v[2]not in rows};unchanged_new={v for v in newset if v[2]not in rows}
out=dict(status='ROOT_INDEPENDENT_CONSTRUCTION_CHECK_PASS',candidate=str(c),candidate_sha256=sha(c),base_candidate_sha256=sha(base),changed_existing_arrays=changed,unchanged_array_count=len(a.files)-len(changed),changed_face_rows=rows.tolist(),points_bitwise_unchanged_from_pass002=True,typed_anchors_unchanged=237,boundary_edges_unchanged=int(np.count_nonzero(n==1)),manifold=True,no_winding_conflict=True,no_duplicate_faces=True,matched037_faces=True,parent_union_csr_valid=True,area=area,max_source_displacement_mm=float(source.max()),max_step_from410_mm=float(step.max()),self_pairs=0,surface_pairs=3254,inside_pairs=391,introduced_inside_pairs_vs410=introduced,removed_inside_pairs_vs410=removed,unchanged_face_introduced_surface_pairs=len(unchanged_new-unchanged_old),unchanged_face_resolved_surface_pairs=len(unchanged_old-unchanged_new),point_summary_sha256=sha(ap/'audit-130/summary.json'),surface_report_sha256=sha(ap/'audit-159/report.json'),accepted_physical_steps=0,native_admission=False,physical_transfer_pending=True)
(r/'root-independent-construction-verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items()if k not in ['area','removed_inside_pairs_vs410']}))

