from pathlib import Path
import csv,hashlib,json,numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
C=Path(__file__).parent; S=C.parents[1]
B=S/'per-bank-workers-20261027/b124-123/root-depth-quadratic-strain-001'
OUT=C/'safe-components-001';OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
base=B/'candidate-private.npz';trial=C/'candidate-private.npz'
if sha(base)!='e111fa302a41f7c80e278219c43451a8e56452b563d638855c0e477744a83be1' or sha(trial)!='40dbb7464134404071afbc244cb7a30f52cf2d1da6f1d7dafdeea03a564a2c02':raise ValueError('input pin')
with np.load(base,allow_pickle=False) as z:A={k:z[k].copy() for k in z.files}
with np.load(trial,allow_pickle=False) as z:
 if set(A)!=set(z.files) or not all(A[k].shape==z[k].shape and A[k].dtype==z[k].dtype and A[k].tobytes()==z[k].tobytes() for k in A if k!='points_world_m'):raise ValueError('unexpected metadata changes')
 y=z['points_world_m'].copy()
x=A['points_world_m'];f=A['source_O_faces'];ids=np.flatnonzero(np.any(x!=y,axis=1));mask=np.zeros(len(x),bool);mask[ids]=True
e=np.unique(np.sort(np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1),axis=0);e=e[np.all(mask[e],axis=1)]
adj=sparse.csr_matrix((np.ones(2*len(e)),(np.r_[e[:,0],e[:,1]],np.r_[e[:,1],e[:,0]])),shape=(len(x),len(x)))[ids][:,ids]
count,labels=connected_components(adj,directed=False); pairs=np.load(C/'audit-130/surface-self-pairs.npy')
selected=[];details=[]
for i in range(count):
 nodes=ids[labels==i];rows=np.flatnonzero(np.any(np.isin(f,nodes),axis=1));rset=set(map(int,rows))
 witness=[p.tolist() for p in pairs if int(p[0]) in rset or int(p[1]) in rset]
 keep=not witness
 if keep:selected.extend(map(int,nodes))
 details.append({'component':i,'node_count':len(nodes),'affected_face_count':len(rows),'touching_self_pair_count':len(witness),'retained':keep})
selected=np.array(sorted(selected),dtype=np.int64);out=x.copy();out[selected]=y[selected];A['points_world_m']=out;p=OUT/'candidate.npz'
with p.open('xb') as fp:np.savez_compressed(fp,**A)
rep={'status':'PRIVATE_SELECTED_COMPONENTS_PENDING_EXACT_AUDITS','base_sha256':sha(base),'trial_sha256':sha(trial),'candidate_sha256':sha(p),'self_witness_count_in_trial':len(pairs),'selected_node_count':len(selected),'components':details,'geometry_admitted':False,'physical_steps':0}
(OUT/'build-report.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2))
