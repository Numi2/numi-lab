#!/usr/bin/env python3
import json, hashlib, sys
from pathlib import Path
import numpy as np
D=Path(sys.argv[1]); E=D.parent
P=D/'validation-001/zero-capacity-witness-index-010.npz'
J=P.with_suffix('.json')
meta=json.loads(J.read_text())
assert meta['schema']=='numi.human.p1-zero-capacity-witness-index.v1'
assert meta['npz_sha256']==hashlib.sha256(P.read_bytes()).hexdigest()
assert meta['source_checkpoint_sha256']==hashlib.sha256((D/'run-capacity-010/checkpoint.json').read_bytes()).hexdigest()
with np.load(P,allow_pickle=False) as z:
    a={k:z[k] for k in z.files}
mesh=E/'global-inner-p1-active-layer-refinement-008/candidate-with-transfer-partial-field.npz'
with np.load(mesh,allow_pickle=False) as z:
    tets=np.asarray(z['tetrahedra'],np.int64)
    fixed=np.zeros(len(z['points_world_m']),bool); fixed[np.asarray(z['physical_outer_node_ids'],np.int64)]=True
inv=json.loads((E/'registered-surrounding-tissue-native013-191/inventory.json').read_text())['banks']
off=np.r_[0,np.cumsum([int(x['triangles']) for x in inv],dtype=np.int64)]
keys=np.column_stack((a['key_bank_ids'],a['key_local_face_ids'],a['key_flat_face_ids'],a['key_tet_ids'])).astype(np.int64)
assert keys.shape[1]==4 and len(np.unique(keys,axis=0))==len(keys)
assert len(keys)==meta['unique_key_count']==13709
rowoffs=a['key_witness_row_offsets']; counts=a['key_stored_witness_row_counts']
assert rowoffs.shape==(len(keys)+1,) and rowoffs[0]==0 and np.array_equal(np.diff(rowoffs),counts)
assert rowoffs[-1]==len(a['barycentric'])==meta['stored_unique_barycentric_rows']==24960
assert a['support_node_ids'].shape==(len(a['barycentric']),4)
assert np.all(a['zero_capacity_mask']) and np.all(a['free_capacity']<=1e-12)
assert np.all(np.isfinite(a['barycentric'])) and np.all(a['barycentric']>=-2e-12)
assert np.max(np.abs(a['barycentric'].sum(axis=1)-1))<=2e-12
assert np.array_equal(keys[:,2],off[keys[:,0]]+keys[:,1])
for i,k in enumerate(keys):
    b,lf,ff,t=map(int,k)
    assert b<len(inv) and lf<int(inv[b]['triangles']) and ff==int(off[b]+lf) and t<len(tets)
    s=int(rowoffs[i]);e=int(rowoffs[i+1]); nodes=tets[t]
    exp=np.where(a['barycentric'][s:e]>0,nodes[None,:],np.iinfo(np.uint32).max).astype(np.uint32)
    assert np.array_equal(a['support_node_ids'][s:e],exp), (i,k)
    assert np.all(np.sum(a['barycentric'][s:e]*~fixed[nodes][None,:],axis=1)<=1e-12)
    assert np.all(fixed[np.unique(nodes[a['barycentric'][s:e].max(axis=0)>0])])
bo=a['bank_affected_tet_offsets']; bt=a['bank_affected_tet_ids'].astype(np.int64)
assert bo.shape==(len(inv)+1,) and bo[0]==0 and bo[-1]==len(bt) and np.all(np.diff(bo)>=0)
summary=meta['bank_summaries']; assert len(summary)==len(inv)==689
assert sum(int(x['zero_capacity_rows']) for x in summary)==meta['raw_zero_capacity_rows']==24960
assert sum(int(x['low_capacity_rows']) for x in summary)==1616231
assert np.array_equal(a['bank_zero_capacity_rows'],np.asarray([x['zero_capacity_rows'] for x in summary],np.uint64))
assert np.array_equal(a['bank_low_capacity_rows'],np.asarray([x['low_capacity_rows'] for x in summary],np.uint64))
for b,x in enumerate(summary):
    tids=np.unique(keys[keys[:,0]==b,3])
    got=bt[bo[b]:bo[b+1]]
    assert np.array_equal(got,tids)
    assert int(x['affected_tet_count'])==len(tids)
    assert int(x['affected_key_count'])==int(np.count_nonzero(keys[:,0]==b))
print(json.dumps({'status':'PASS_ZERO_CAPACITY_INDEX_INTEGRITY','unique_bank_face_tet_keys':len(keys),'stored_zero_support_rows':len(a['barycentric']),'zero_capacity_rows':int(meta['raw_zero_capacity_rows']),'low_capacity_rows':1616231,'banks':len(inv),'banks_with_zero_capacity':int(np.count_nonzero(a['bank_zero_capacity_rows'])),'banks_with_affected_tets':int(np.count_nonzero(np.diff(bo))),'affected_bank_tet_pairs':len(bt),'index_npz_sha256':meta['npz_sha256'],'index_json_sha256':hashlib.sha256(J.read_bytes()).hexdigest()},indent=2))
