from pathlib import Path
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import hashlib,json,numpy as np
S=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027")
paths=[S/"b124-123/root-depth-quadratic-strain-001/candidate-private.npz",S/"batch-six-depth-strain-001/safe-components-001/candidate.npz",S/"batch-four-depth-strain-001/safe-components-001/candidate.npz",S/"b124-123/root-strain-backtrack-001/alpha-0.5/candidate.npz"]
wanted=["e111fa302a41f7c80e278219c43451a8e56452b563d638855c0e477744a83be1","d677167f6706bfbdb5411f571161d00aba0ac099d4958b29676c2713134a4c66","c3a15db2ec5d7660526f131b97ef71c954a16df4e9d1d18600e0ddeb48d1e778","bdeaa6fc6f348f08b84a288b0bba17b0a2f8861d7f86c23ff0a916ed3adbb939"]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert [sha(p) for p in paths]==wanted
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
b,s,f,a=map(load,paths)
for d in (s,f,a):
 assert set(d)==set(b) and all(d[k].shape==b[k].shape and d[k].dtype==b[k].dtype and d[k].tobytes()==b[k].tobytes() for k in b if k!="points_world_m")
x=b["points_world_m"];faces=b["source_O_faces"]
si=np.flatnonzero(np.any(s["points_world_m"]!=x,axis=1));fi=np.flatnonzero(np.any(f["points_world_m"]!=x,axis=1));ai=np.flatnonzero(np.any(a["points_world_m"]!=x,axis=1))
e=np.unique(np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0);mask=np.zeros(len(x),bool);mask[si]=True;edge=e[np.all(mask[e],axis=1)];adj=sparse.csr_matrix((np.ones(2*len(edge)),(np.r_[edge[:,0],edge[:,1]],np.r_[edge[:,1],edge[:,0]])),shape=(len(x),len(x)))[si][:,si];n,labels=connected_components(adj,directed=False)
components=[si[labels==i] for i in range(n)];unique=[c for c in components if not np.intersect1d(c,fi).size]
assert len(unique)==1 and len(unique[0])==176 and len(fi)==244 and len(ai)==174
assert not np.intersect1d(ai,np.r_[fi,unique[0]]).size
y=x.copy();y[fi]=f["points_world_m"][fi];y[unique[0]]=s["points_world_m"][unique[0]];y[ai]=a["points_world_m"][ai];b["points_world_m"]=y
O=S/"combined-multibank-strain-001";O.mkdir(exist_ok=False)
with (O/"candidate.npz").open("xb") as q:np.savez_compressed(q,**b)
r={"status":"PRIVATE_COMBINATION_PENDING_EXACT_AUDITS","input_sha256":wanted,"candidate_sha256":sha(O/"candidate.npz"),"selected_0p15mm_nodes":len(fi),"selected_0p05mm_nodes":len(unique[0]),"selected_backtracked_124_nodes":len(ai),"geometry_admitted":False,"accepted_physical_steps":0};(O/"build-report.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r))
