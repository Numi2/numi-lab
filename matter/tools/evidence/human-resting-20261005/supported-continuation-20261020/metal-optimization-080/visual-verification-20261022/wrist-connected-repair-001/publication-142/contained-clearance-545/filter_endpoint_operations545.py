"""Retain safe endpoint moves in one connected mesh; remove only witnessed operations.

This is a construction proposal. Exact self and all-bank audits are mandatory.
No interpolation or smoothing of target points is performed: each operation is
retained at its certified outside endpoint or removed back to the pinned base.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,numpy as np
W=Path(__file__).parent;D=W.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--state',type=Path,required=True);a.add_argument('--targets',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--remove-face-rows',type=Path);a.add_argument('--prior-operations',type=Path);q=a.parse_args()
 st=json.loads(q.state.read_text());base=Path(st['candidate']);assert sha(base)==st['candidate_sha256'];z=np.load(base,allow_pickle=False);t=np.load(q.targets,allow_pickle=False)
 x=z['points_world_m'];F=z['source_O_faces'];rows=t['query_rows'].astype(np.int64);assert t['pass_mask'].all() and t['target_exactly_outside_all185'].all()
 assert not np.isin(rows,z['typed_coarse_anchor_output_rows_018']).any()
 kept=np.ones(len(rows),bool)
 if q.prior_operations:kept=np.asarray(json.loads(q.prior_operations.read_text())['retained_mask'],bool);assert kept.shape==(len(rows),)
 history=[]
 if q.remove_face_rows:
  f=np.load(q.remove_face_rows);assert f.dtype.kind in 'iu' and np.all(f>=0)and np.all(f<len(F));nodes=np.unique(F[f]);drop=kept&np.isin(rows,nodes);kept[drop]=False;history.append(dict(kind='exact_geometry_witness_faces',faces=f.tolist(),removed_nodes=rows[drop].tolist()))
 sp=importlib.util.spec_from_file_location('checks545',W/'run_candidate_checks545.py');check=importlib.util.module_from_spec(sp);sp.loader.exec_module(check);owner=check.mod(check.OWNER)
 r0,_=owner.compose_source_reference(z,check.R0);bm=x.astype(float)*1000;ref=z['area_reference_037_points_world_m_f64']*1000
 edges=np.unique(np.sort(np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1),axis=0);L=np.linalg.norm(bm[edges[:,0]]-bm[edges[:,1]],axis=1)
 for iteration in range(len(rows)+1):
  y=x.copy();y[rows[kept]]=t['target_world_f32_m'][kept];delta=(y.astype(float)-x.astype(float))*1000
  gate=owner.full_f32_area_gate(y,bm,ref,F,.1)
  badfaces=np.union1d(gate['bad_current_oriented_face_rows_first_64'],gate['bad_fixed_037_reference_face_rows_first_64']).astype(np.int64)
  # Compute all failing face IDs, including beyond report's witness truncation.
  c=np.cross(y[F[:,1]].astype(float)-y[F[:,0]],y[F[:,2]].astype(float)-y[F[:,0]])
  cb=np.cross(x[F[:,1]].astype(float)-x[F[:,0]],x[F[:,2]].astype(float)-x[F[:,0]])
  cr=np.cross(ref[F[:,1]]-ref[F[:,0]],ref[F[:,2]]-ref[F[:,0]])
  badfaces=np.flatnonzero((np.einsum('ij,ij->i',c,cb)/np.einsum('ij,ij->i',cb,cb)<.1)|(np.linalg.norm(c,axis=1)/np.linalg.norm(cr,axis=1)*1e6<.1))
  bade=np.flatnonzero(np.max(np.abs(delta[edges[:,0]]-delta[edges[:,1]]),axis=1)>.5*L+1e-8)
  badn=np.flatnonzero((np.linalg.norm(y.astype(float)-r0,axis=1)>.012000001)|(np.linalg.norm(delta,axis=1)>1.000001))
  nodes=np.unique(np.concatenate([F[badfaces].ravel(),edges[bade].ravel(),badn]))
  drop=kept&np.isin(rows,nodes)
  history.append(dict(iteration=iteration,bad_area_faces=badfaces.tolist(),bad_edge_rows=bade.tolist(),bad_bound_nodes=badn.tolist(),removed_nodes=rows[drop].tolist(),remaining_operations=int(kept.sum())))
  if not len(badfaces)and not len(bade)and not len(badn):break
  if not drop.any():raise RuntimeError('bad geometry cannot be attributed to remaining operations')
  kept[drop]=False
 else:raise RuntimeError('filter did not terminate')
 q.out.mkdir(parents=True,exist_ok=False);np.savez_compressed(q.out/'candidate.npz',**{**{k:z[k]for k in z.files},'points_world_m':y})
 record=dict(status='PROPOSAL_REQUIRES_EXACT_CHECKS',base_candidate=str(base),base_sha256=sha(base),target_npz=str(q.targets),target_sha256=sha(q.targets),candidate_sha256=sha(q.out/'candidate.npz'),source_sha256=sha(__file__),retained_mask=kept.tolist(),retained_node_rows=rows[kept].tolist(),removed_node_rows=rows[~kept].tolist(),retained_operations=int(kept.sum()),history=history,construction_only=True,native_admission=False,accepted_physical_steps=0)
 (q.out/'operations.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:record[k]for k in ['status','retained_operations','candidate_sha256']}))
if __name__=='__main__':main()
