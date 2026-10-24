"""Independent F32/node-accounting check. Source-only, no physical admission."""
from pathlib import Path
import csv,hashlib,json
from collections import Counter
import numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D=E/'native-source-only-rest-geometry-022-query-023/source-preflight'
C=E/'o12-outward-repair-022/candidate.npz'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
 with p.open() as f:return list(csv.DictReader(f))
def xyz(rs):return np.array([[float(r[c+'_m']) for c in 'xyz'] for r in rs],np.float32)
X=np.load(C,allow_pickle=False)['points_world_f32']
exact=rows(D/'exact-emitted.nodes.csv'); init=rows(D/'initial-skin-fem-nodes.csv')
base=rows(E/'native-candidate020-all-bank-containment-035/source-preflight-003/initial-skin-fem-nodes.csv')
assert len(exact)==len(init)==len(base)==7233
assert [int(r['global_node']) for r in exact]==[int(r['global_node']) for r in base]
assert np.array_equal(xyz(exact).view(np.uint32),X.view(np.uint32))
assert np.array_equal(xyz(init).view(np.uint32),xyz(base).view(np.uint32))
q=rows(D/'all-bank-containment.csv'); nodes=rows(D/'all-bank-containment.csv.nodes.csv')
assert len(nodes)==7233
for r in nodes:
 assert int(r['total_bone_banks'])==185
 assert sum(int(r[k]) for k in ['aabb_proven_outside_banks','parity_outside_banks','inside_banks','ambiguous_banks','invalid_banks'])==185
counts=Counter(r['classification'] for r in q)
inside=[{k:int(r[k]) for k in ['local_node','global_node','layer','row','bank_body','bank_stable_id']} for r in q if r['classification']=='inside']
assert sum(int(r['inside_banks']) for r in nodes)==len(inside)==17
assert sum(int(r['invalid_banks']) for r in nodes)==0
assert sum(int(r['ambiguous_banks']) for r in nodes)==0
out={
 'status':'INDEPENDENT_STATIC_QUERY_REVIEW_FAIL_CLEARANCE',
 'candidate_sha256':sha(C),'exact_f32_matches_candidate':True,
 'initial_node_export_is_pre_override_baseline':True,
 'initial_export_mismatch_rows_vs_candidate':int(np.any(xyz(init)!=X,axis=1).sum()),
 'all_bank_count':185,'node_count':7233,'query_classifications':dict(counts),
 'inside_by_layer':dict(Counter(str(r['layer'])for r in inside)),
 'inside_witnesses':inside,'ambiguous_banks':0,'invalid_banks':0,
 'accepted_physical_steps':0,'physical_qualification':False,
 'limitations':['Exact emitted nodes, not the pre-override initial-node export, own query geometry.','Node containment does not establish edge/surface/tet clearance.','Native query frame reconstruction is separately checked by the consolidated route owner.'],
 'inputs':{str(f):sha(f)for f in [C,D/'exact-emitted.nodes.csv',D/'initial-skin-fem-nodes.csv',D/'all-bank-containment.csv',D/'all-bank-containment.csv.nodes.csv']}
}
O=E/'root-independent-static-query-023.json';O.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

