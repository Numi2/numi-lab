"""Finite-witness patch fitting through the existing lexicographic LP owner276."""
from pathlib import Path
import numpy as np,json,hashlib,runpy,time
from scipy import sparse
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bary(p,t):
 a=t[1]-t[0];b=t[2]-t[0];v=p-t[0]
 gram=np.array([[a@a,a@b],[a@b,b@b]])
 uv=np.linalg.solve(gram,np.array([a@v,b@v]))
 return np.r_[1.-uv.sum(),uv]
def contact_rows(x,f,z,banks,inventory):
 rr=[];cc=[];vv=[];rhs=[];source=[];row=0
 volume_sign={}
 for bi in np.unique(z['bank_ids']):
  key=inventory[int(bi)]['key'];v=banks[key+'_vertices'].astype(float);bf=banks[key+'_faces'];t=v[bf]
  volume=float(np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()/6)
  volume_sign[int(bi)]=1. if volume>0 else -1.
 for k,(bi,fi,oi) in enumerate(zip(z['bank_ids'],z['bank_local_face_ids'],z['physical_O_face_ids'])):
  key=inventory[int(bi)]['key'];bt=banks[key+'_vertices'][banks[key+'_faces'][fi]].astype(float)
  n=np.cross(bt[1]-bt[0],bt[2]-bt[0]);n*=volume_sign[int(bi)]/np.linalg.norm(n)
  tri=x[f[oi]];lo,hi=z['exact_hit_offsets'][k:k+2]
  for point in z['exact_hit_points_m'][lo:hi]:
   w=bary(point,tri)
   if min(w)<-1e-7 or max(w)>1+1e-7:raise ValueError('witness outside physical O face')
   w=np.maximum(w,0);w/=w.sum()
   for j,node in enumerate(f[oi]):
    for axis in range(3):rr.append(row);cc.append(3*int(node)+axis);vv.append(w[j]*n[axis])
   rhs.append(.02);source.append((k,-1));row+=1
  # Add only projected vertex/face penetration constraints whose closest point lies within this finite bank triangle.
  for j,node in enumerate(f[oi]):
   signed=float((tri[j]-bt[0])@n)
   if signed>=0:continue
   w=bary(tri[j]-signed*n,bt)
   if min(w)<0 or max(w)>1:continue
   for axis in range(3):rr.append(row);cc.append(3*int(node)+axis);vv.append(n[axis])
   rhs.append(-signed*1000+.02);source.append((k,j));row+=1
 A=sparse.coo_matrix((vv,(rr,cc)),shape=(row,3*len(x))).tocsr()
 return A,np.array(rhs),np.asarray(source,np.int64),volume_sign
def fit(x,f,A,b):
 active=np.zeros(len(x),bool);active[np.unique(A.indices//3)]=True
 edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
 for _ in range(3):active[edges[np.any(active[edges],axis=1)].ravel()]=True
 ids=np.flatnonzero(active);cols=(3*ids[:,None]+np.arange(3)).ravel();A=A[:,cols];n=A.shape[1]
 selected=edges[np.any(active[edges],axis=1)];lookup=np.full(len(x),-1);lookup[ids]=np.arange(len(ids))
 gr=[];gc=[];gv=[]
 for j in range(2):
  ok=lookup[selected[:,j]]>=0
  for axis in range(3):
   gr.extend(3*np.flatnonzero(ok)+axis);gc.extend(3*lookup[selected[ok,j]]+axis);gv.extend(np.full(sum(ok),1. if j==0 else -1.))
 G=sparse.coo_matrix((gv,(gr,gc)),shape=(3*len(selected),n)).tocsr();ne=G.shape[0];nc=A.shape[0]
 I=sparse.eye(n);J=sparse.eye(ne);Z=lambda r,c:sparse.csr_matrix((r,c))
 matrix=sparse.vstack([
  sparse.hstack([-A,Z(nc,n),Z(nc,ne),-sparse.eye(nc)]),
  sparse.hstack([I,-I,Z(n,ne),Z(n,nc)]),sparse.hstack([-I,-I,Z(n,ne),Z(n,nc)]),
  sparse.hstack([G,Z(ne,n),-J,Z(ne,nc)]),sparse.hstack([-G,Z(ne,n),-J,Z(ne,nc)])],format='csr')
 rhs=np.r_[-b,np.zeros(2*n+2*ne)];c=np.r_[np.zeros(n),np.ones(n),np.full(ne,2.),np.zeros(nc)]
 bounds=[(-25.,25.)]*n+[(0,None)]*(n+ne+nc)
 owner=E/'skin-repair-lexicographic-lp-276/tools/human_skin_repair_lp.py'
 solve=runpy.run_path(str(owner))['solve_contact_lp_lexicographic']
 res,info=solve(c,matrix,rhs,bounds,2*n+ne,nc,time_limit=60.,slack_sum_tolerance=1e-7)
 if not res.success:raise RuntimeError(json.dumps(info))
 delta=np.zeros_like(x);delta[ids]=res.x[:n].reshape(-1,3)/1000
 return delta,ids,info,owner
def main():
 begun=time.monotonic();out=D/'run-003';assert not out.exists();out.mkdir()
 mesh=E/'global-inner-p1-active-layer-refinement-008/candidate-with-transfer-partial-field.npz'
 pairs=E/'global-inner-p1-full-domain-clipping-009/validation-001/zero-support-physical-O-face-pairs-010.npz'
 bank=E/'registered-surrounding-tissue-native013-191/registered-tissue-banks.npz';ip=bank.with_name('inventory.json')
 m=np.load(mesh);z=np.load(pairs);banks=np.load(bank);inv=json.loads(ip.read_text())['banks']
 x=m['points_world_m'].astype(float);f=m['physical_outer_faces']
 A,b,witness,sign=contact_rows(x,f,z,banks,inv);delta,ids,info,owner=fit(x,f,A,b)
 new=(x+delta).astype(np.float32);actual=new.astype(float)-x;tri=x[f];nt=new[f].astype(float)
 n0=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n1=np.cross(nt[:,1]-nt[:,0],nt[:,2]-nt[:,0])
 cos=np.einsum('ij,ij->i',n0,n1)/(np.linalg.norm(n0,axis=1)*np.linalg.norm(n1,axis=1))
 moved=np.flatnonzero(np.any(new!=m['points_world_m'],axis=1))
 np.savez_compressed(out/'outer-envelope-proposal.npz',points_world_m=new,physical_outer_faces=f,changed_node_ids=moved,
  node_displacements_m=actual,active_node_ids=ids,constraint_witness_rows=witness,constraint_rhs_mm=b,
  constraint_values_emitted_mm=A@(actual.ravel()*1000))
 report={'status':'FINITE_WITNESS_CONSTRUCTION_ONLY','lp':info,'constraint_count':len(b),'moved_nodes':len(moved),
  'max_displacement_mm':float(np.linalg.norm(actual,axis=1).max()*1000),'surface_normal_flips':int(sum(cos<=0)),
  'minimum_surface_normal_cosine':float(cos.min()),'constraint_min_slack_emitted_mm':float(np.min(A@(actual.ravel()*1000)-b)),
  'bank_signed_volume_orientation_signs':sign,'input_sha256':{str(p):sha(p) for p in (mesh,pairs,bank,ip,owner,Path(__file__))},
  'physical_admission':False,'native_admission':False,'needs_exact_gate':True,'elapsed_s':time.monotonic()-begun}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()

