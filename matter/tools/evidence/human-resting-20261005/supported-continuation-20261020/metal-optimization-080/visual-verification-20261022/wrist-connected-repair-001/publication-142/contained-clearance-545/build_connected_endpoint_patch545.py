"""Connected harmonic construction patch with exact point endpoint constraints.

This prepares geometry only; it never supplies forces or simulation state.
Every position outside the declared patch, every typed anchor, and already
cleared endpoint operations stay fixed. Native/physical admission remains false.
"""
from pathlib import Path
import argparse,hashlib,json,numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def solve_patch(x,faces,rows,targets,anchors,protected,rings):
 e=np.unique(np.sort(np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
 L=np.linalg.norm(x[e[:,0]].astype(float)-x[e[:,1]],axis=1);assert np.all(L>0)
 weight=1/(L*L)
 A=sparse.coo_matrix((np.ones(len(e)*2),(np.r_[e[:,0],e[:,1]],np.r_[e[:,1],e[:,0]])),shape=(len(x),len(x))).tocsr()
 domain=np.zeros(len(x),bool);domain[rows]=True
 for _ in range(rings):domain|=np.asarray(A@domain.astype(float)).ravel()>0
 fixed=np.zeros(len(x),bool);fixed[np.r_[anchors,protected,rows]]=True
 free=np.flatnonzero(domain&~fixed)
 lap=sparse.coo_matrix((np.r_[weight,weight,-weight,-weight],(np.r_[e[:,0],e[:,1],e[:,0],e[:,1]],np.r_[e[:,0],e[:,1],e[:,1],e[:,0]])),shape=(len(x),len(x))).tocsr()
 d=np.zeros((len(x),3),float);d[rows]=targets.astype(float)-x[rows].astype(float)
 if len(free):d[free]=spsolve(lap[free][:,free],-(lap[free]@d))
 assert np.isfinite(d).all()
 y=(x.astype(float)+d).astype(np.float32);y[rows]=targets;y[anchors]=x[anchors];y[protected]=x[protected]
 return y,dict(rings=rings,patch_nodes=np.flatnonzero(domain).tolist(),free_nodes=free.tolist(),target_nodes=rows.tolist(),protected_endpoint_nodes=protected.tolist(),laplacian='positive inverse squared rest-edge weights; Dirichlet target and fixed outer boundary',linear_solve_residual_max=float(np.abs((lap@d)[free]).max(initial=0.)))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--state',required=True,type=Path);ap.add_argument('--targets',required=True,type=Path);ap.add_argument('--protected-operations',type=Path);ap.add_argument('--rings',type=int,required=True);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args()
 st=json.loads(a.state.read_text());assert sha(st['candidate'])==st['candidate_sha256'];z=np.load(st['candidate'],allow_pickle=False);t=np.load(a.targets,allow_pickle=False)
 assert t['pass_mask'].all() and t['target_exactly_outside_all185'].all()
 rows=t['query_rows'].astype(np.int64);anchors=z['typed_coarse_anchor_output_rows_018'];assert not np.isin(rows,anchors).any()
 protected=np.asarray(json.loads(a.protected_operations.read_text())['retained_node_rows'],np.int64)if a.protected_operations else np.empty(0,np.int64)
 assert not np.isin(rows,protected).any()
 y,receipt=solve_patch(z['points_world_m'],z['source_O_faces'],rows,t['target_world_f32_m'],anchors,protected,a.rings)
 a.out.mkdir(parents=True,exist_ok=False);np.savez_compressed(a.out/'candidate.npz',**{**{k:z[k]for k in z.files},'points_world_m':y})
 receipt.update(base_candidate=st['candidate'],base_sha256=st['candidate_sha256'],targets=str(a.targets),targets_sha256=sha(a.targets),source_sha256=sha(__file__),candidate_sha256=sha(a.out/'candidate.npz'),construction_only=True,native_admission=False,accepted_physical_steps=0)
 (a.out/'report.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'rings':a.rings,'patch_nodes':len(receipt['patch_nodes']),'free_nodes':len(receipt['free_nodes'])}))
if __name__=='__main__':main()
