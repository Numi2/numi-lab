"""Project measured patch area/edge construction constraints, retaining exact targets."""
from pathlib import Path
import argparse,json,hashlib,numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def oriented_grad(p,normal,den):
 return np.cross(np.array([p[1]-p[2],p[2]-p[0],p[0]-p[1]]),normal)/den

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--patch-dir',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args();r=json.loads((a.patch_dir/'report.json').read_text())
 assert sha(r['base_candidate'])==r['base_sha256'];z=np.load(a.patch_dir/'candidate.npz',allow_pickle=False);b=np.load(r['base_candidate'],allow_pickle=False)
 X=b['points_world_m'].astype(float)*1000;Y=z['points_world_m'].astype(float)*1000;F=z['source_O_faces'];free=np.zeros(len(X),bool);free[np.array(r['free_nodes'],int)]=True
 R=z['area_reference_037_points_world_m_f64']*1000;refmag=np.linalg.norm(np.cross(R[F[:,1]]-R[F[:,0]],R[F[:,2]]-R[F[:,0]]),axis=1)
 ref=np.cross(X[F[:,1]]-X[F[:,0]],X[F[:,2]]-X[F[:,0]]);den=np.einsum('ij,ij->i',ref,ref);assert np.all(den>0)
 edges=np.unique(np.sort(np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1),axis=0);L=np.linalg.norm(X[edges[:,0]]-X[edges[:,1]],axis=1)
 history=[]
 for it in range(120):
  # Emission is part of the constraint, so no hidden double-precision proposal is selected.
  Y=(Y*.001).astype(np.float32).astype(float)*1000
  cr=np.cross(Y[F[:,1]]-Y[F[:,0]],Y[F[:,2]]-Y[F[:,0]]);ratio=np.einsum('ij,ij->i',cr,ref)/den
  mag=np.linalg.norm(cr,axis=1)/refmag
  bad=np.flatnonzero(ratio<.105);badmag=np.flatnonzero(mag<.105)
  delta=Y-X;badedges=np.flatnonzero(np.max(np.abs(delta[edges[:,0]]-delta[edges[:,1]]),axis=1)>.495*L+1e-8)
  history.append(dict(iteration=it,minimum_oriented_ratio=float(ratio.min()),area_faces=bad.tolist(),magnitude_faces=badmag.tolist(),edge_count=len(badedges)))
  if not len(bad)and not len(badmag)and not len(badedges):break
  for fi in bad:
   ns=F[fi];g=oriented_grad(Y[ns],ref[fi],den[fi]);g[~free[ns]]=0;norm=float(np.sum(g*g))
   if norm<1e-24:raise RuntimeError(f'area witness {fi} has no free correction direction')
   current=float(np.cross(Y[ns[1]]-Y[ns[0]],Y[ns[2]]-Y[ns[0]])@ref[fi]/den[fi])
   if current<.115:Y[ns]+=((.115-current)/norm)*g
  for fi in badmag:
   ns=F[fi];c=np.cross(Y[ns[1]]-Y[ns[0]],Y[ns[2]]-Y[ns[0]]);normc=float(np.linalg.norm(c));assert normc>0
   g=oriented_grad(Y[ns],c/normc,refmag[fi]);g[~free[ns]]=0;norm=float(np.sum(g*g))
   if norm<1e-24:raise RuntimeError(f'magnitude witness {fi} has no free correction')
   current=normc/refmag[fi]
   if current<.115:Y[ns]+=((.115-current)/norm)*g
  for ei in badedges:
   ns=edges[ei];n=int(free[ns].sum())
   if not n:raise RuntimeError('edge witness has no free correction')
   d=(Y[ns[0]]-X[ns[0]])-(Y[ns[1]]-X[ns[1]])
   err=d-np.clip(d,-.49*L[ei],.49*L[ei])
   if free[ns[0]]:Y[ns[0]]-=err/n
   if free[ns[1]]:Y[ns[1]]+=err/n
 else:raise RuntimeError('bounded patch projection did not converge')
 y=(Y*.001).astype(np.float32);fixed=np.flatnonzero(~free);assert np.array_equal(y[fixed],z['points_world_m'][fixed])
 a.out.mkdir(parents=True,exist_ok=False);np.savez_compressed(a.out/'candidate.npz',**{**{k:z[k]for k in z.files},'points_world_m':y})
 (a.out/'report.json').write_text(json.dumps(dict(parent_patch=str(a.patch_dir),parent_patch_sha256=sha(a.patch_dir/'candidate.npz'),base_candidate=r['base_candidate'],base_sha256=r['base_sha256'],source_sha256=sha(__file__),history=history,target_nodes_bitwise_unchanged=True,all_fixed_nodes_bitwise_unchanged=True,candidate_sha256=sha(a.out/'candidate.npz'),construction_only=True,native_admission=False,accepted_physical_steps=0),indent=2)+'\n');print('projected',len(history),'iterations')
if __name__=='__main__':main()
