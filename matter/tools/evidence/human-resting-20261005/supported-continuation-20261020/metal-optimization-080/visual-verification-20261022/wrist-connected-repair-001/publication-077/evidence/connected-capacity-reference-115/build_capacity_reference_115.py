from pathlib import Path
import numpy as np,csv,json,hashlib,sys
D=Path("/Users/numi/Work/numi-human-connected-surface-069");P=D/'connected-capacity-reference-115';assert not P.exists();P.mkdir()
source=D/'whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz'
rowsfile=D/'connected-reference-capacity-112/run-001/column-capacity-proposals.csv'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(source)=='5a35470223ce12343cc217be7822d182802e99f1c423bb52b8409f153f9a49db'
assert sha(rowsfile)=='ec00f14dbec8852e7eb8b3a5e3299e6c2657fa81b1e50aa96d20f51bd6a8b8c9'
z=np.load(source);O=z['current037_points_m_f32'];R=z['reference_points_m'];T=z['tetrahedra_zero_based'];M=z['material_ids'];X=O.astype(float).copy()
rows=list(csv.DictReader(rowsfile.open()));changed=[]
for r in rows:
 if r['capacity_status']!='PROPOSED_D_SHORTENING':continue
 c=int(r['column_row']);node=c+4822
 X[node]=[float(r['D_safe_'+k+'_m'])for k in 'xyz'];changed.append(node)
assert len(changed)==176 and np.array_equal(X[:4822],O[:4822])
NATIVE=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")/'native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003/candidate.npz'
assert sha(NATIVE)=='640b7bc56013748b26d5e4e12790f85a472602ee6a4732f5773243df6e2147cc'
nt=np.load(NATIVE)['old_tetrahedra']
delta=X[:7233]-O[:7233].astype(float);gen={}
for node,parent,w in zip(z['newnode_ancestry_output_row'],z['newnode_ancestry_source_tet_row'],z['newnode_ancestry_barycentric']):
 gen.setdefault(int(node),[]).append(np.asarray(w)@delta[nt[int(parent)]])
disagreements=[]
for node,values in gen.items():
 v=np.stack(values);spread=float(np.max(np.linalg.norm(v-v[0],axis=1)));disagreements.append({'node':node,'max_parent_displacement_disagreement_m':spread,'ancestry_choices':len(v)})
 if spread>1e-10:continue
 X[node]+=v[0]
X32=X.astype(np.float32)
sys.path.insert(0,str(D/'connected-cell-trust-111/matter/tools'))
from human_skin_fat_connected_repair import cell_values_gradients
J,_,q,_=cell_values_gradients(X32,R,T)
def dets(Q):
 V=Q[T].astype(float);return np.einsum('ij,ij->i',V[:,1]-V[:,0],np.cross(V[:,2]-V[:,0],V[:,3]-V[:,0]))
vol0=dets(R)/6;vol1=dets(X)/6
bad=np.flatnonzero(J<=0);thin=np.flatnonzero(q<1e-5)
out=P/'candidate.npz';np.savez_compressed(out,points_world_f32=X32,proposed_reference_points_m=R+(X-O.astype(float)),source_reference_points_m=R,source_points_world_f32=O,tetrahedra=T,material_ids=M,source_input_row=z['source_input_row'],cell_uids=z['cell_uids'],modified_D_rows=np.asarray(changed),generated_ancestry_disagreement_rows=np.asarray([r['node']for r in disagreements if r['max_parent_displacement_disagreement_m']>1e-10]))
report={'source_sha256':sha(source),'capacity_proposals_sha256':sha(rowsfile),'candidate_sha256':sha(out),
 'source_outer_and_inner_skin_points_unchanged':bool(np.array_equal(X32[:4822],O[:4822])),
 'D_original_rows_proposed':len(changed),'generated_nodes':len(gen),'generated_parent_choices':int(len(z['newnode_ancestry_output_row'])),
 'generated_ancestry_disagreements':disagreements,
 'invalid_original_connectivity_cells':int(len(bad)),'subthreshold_original_connectivity_cells':int(len(thin)),
 'J_min_relative_to_original_reference':float(J.min()),'J_max_relative_to_original_reference':float(J.max()),
 'q_min':float(q.min()),'invalid_cell_uids':z['cell_uids'][bad].tolist(),
 'material_signed_volume_before_after_m3':{str(m):[float(vol0[M==m].sum()),float(vol1[M==m].sum())]for m in [1,2]},
 'accepted_physical_steps':0,'geometry_admitted':False,'mass_ledger_reusable':False,
 'interpretation':'Whole-mesh capacity-limited inner-reference proposal, not a force or per-step physics change. O/I stay unchanged, D follows certified source path entries, generated points use every recorded source-parent displacement. Old connectivity/material mass cannot be promoted if invalid, nonmatching ancestry, full surfaces or solid checks fail. Reference change requires remeshing and conservative mass/donor mapping before native admission.'}
(P/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ['generated_ancestry_disagreements','invalid_cell_uids']},indent=2))
