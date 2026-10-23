
from pathlib import Path
import importlib.util,json,hashlib,numpy as np,time
from scipy.optimize import minimize, least_squares
from scipy.sparse import lil_matrix
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019');E=R/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001';P=Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/finger-joint-response-001');O=E/'bilateral-knee-connected-weight-repair-100'
sp=importlib.util.spec_from_file_location('rb',R/'tools/rebind_human_skin_reference.py');rb=importlib.util.module_from_spec(sp);sp.loader.exec_module(rb)
src=E/'forefoot-connected-weight-repair-091/candidate.nhskin';p=rb.decode(src.read_bytes());F=p['inds'];W=p['weights'].astype(float);by={int(b['body']):i for i,b in enumerate(p['binds'])}
baseline=json.loads((O/'baseline091-knee-strain.json').read_text());selected=[];side_ids=[]
for side,pair in [('right',(131,136)),('left',(145,150))]:
 bad=baseline['poses']['knee_1p0'][side]['bad_faces']+baseline['poses']['knee_1p5'][side]['bad_faces']
 seed=np.unique(F[np.unique(bad)]);incident=np.any(np.isin(F,seed),axis=1);sel=np.unique(F[incident])
 # Keep variables only on that anatomical chain; incidental neighbours stay fixed.
 sel=sel[(W[sel,by[pair[0]]]+W[sel,by[pair[1]]])>.5]
 selected.extend(sel.tolist());side_ids.extend([pair]*len(sel))
sel=np.array(selected,dtype=np.int64);pairs=np.array([[by[a],by[b]] for a,b in side_ids]);assert len(np.unique(sel))==len(sel)
faceids=np.flatnonzero(np.any(np.isin(F,sel),axis=1));LF=F[faceids]
used=np.unique(LF);lookup=np.full(len(W),-1,dtype=np.int64);lookup[used]=np.arange(len(used));lfaces=lookup[LF];lselected=lookup[sel]
a0,b0=rb.reference_transforms(p,json.loads((P/'neutral.json').read_text()),[0,0,.35]);X=rb.positions(p,a0,b0)
T=X[LF];A=np.stack((T[:,1]-T[:,0],T[:,2]-T[:,0]),axis=2);_,RR=np.linalg.qr(A,mode='reduced');invR=np.linalg.inv(RR);area=np.linalg.norm(np.cross(A[:,:,0],A[:,:,1]),axis=1)
cache={}
for name in ['knee_0p5','knee_1p0','knee_1p5']:
 pose=E/'joint-regional-quality-088-expanded-poses'/(name+'.json')
 a,b=rb.reference_transforms(p,json.loads(pose.read_text()),[0,0,.35]);cache[name]=(a,b,rb.positions(p,a,b)[used])
idx=np.arange(len(sel));wbase=W[sel].copy()
def metrics(delta,summary=False):
 w=wbase.copy();w[idx,pairs[:,0]]+=delta;w[idx,pairs[:,1]]-=delta
 M=np.einsum('nk,kij->nij',w,a0);t=w@b0
 x=np.linalg.solve(M,(X[sel]-t)[...,None])[...,0]
 rows=[];report={}
 for name,(a,b,Y0) in cache.items():
  Y=Y0.copy();Y[lselected]=np.einsum('nk,kij,nj->ni',w,a,x)+w@b
  T=Y[lfaces];B=np.stack((T[:,1]-T[:,0],T[:,2]-T[:,0]),axis=2)
  ss=np.linalg.svd(B@invR,compute_uv=False);ar=np.linalg.norm(np.cross(B[:,:,0],B[:,:,1]),axis=1)/area
  rows.extend([ss[:,-1]-.355,2.39-ss[:,0],ar-.255])
  report[name]={'min_stretch':float(ss[:,-1].min()),'max_stretch':float(ss[:,0].max()),'min_area_ratio':float(ar.min()),'bad_faces':faceids[(ss[:,-1]<.35)|(ss[:,0]>2.4)|(ar<.25)].tolist()}
 return report if summary else np.concatenate(rows)
bounds=[(max(-.15,-W[i,a]),min(.15,W[i,b])) for i,(a,b) in zip(sel,pairs)]

from scipy.sparse import vstack, eye
d0=np.zeros(len(sel));start=time.perf_counter()
# A residual row touches only the at-most-three variable vertices in its
# triangle. Coloring this sparse Jacobian avoids a body-sized numerical scan.
sparse=lil_matrix((len(cache)*3*len(LF),len(sel)),dtype=int)
vmap={int(v):i for i,v in enumerate(sel)}
for fi,face in enumerate(LF):
 for v in face:
  if int(v) in vmap:
   for poseidx in range(len(cache)):
    for component in range(3):
     sparse[poseidx*3*len(LF)+component*len(LF)+fi,vmap[int(v)]]=1
sparse=vstack((sparse.tocsr(),eye(len(sel))),format='csr')
def residual(d):
 return np.r_[np.minimum(metrics(d),0.),1e-4*d]
print(json.dumps({'variables':len(sel),'faces':len(LF),'method':'sparse colored trust-region hinge constraints'}),flush=True)
sol=least_squares(residual,d0,bounds=np.array(bounds).T,jac_sparsity=sparse,
                  method='trf',tr_solver='lsmr',x_scale=.01,max_nfev=100,
                  ftol=1e-11,xtol=1e-11,gtol=1e-10)
report={'success':bool(sol.success),'message':str(sol.message),'evaluations':int(sol.nfev),
        'seconds':time.perf_counter()-start,'variables':len(sel),'face_count':len(LF),
        'max_weight_delta':float(np.abs(sol.x).max()),'minimum_constraint':float(metrics(sol.x).min()),
        'local_metrics':metrics(sol.x,True),'physical_admission':False}
np.savez(O/'sparse-solution.npz',delta=sol.x,selected=sel,pairs=pairs,source_faces=faceids)
if report['minimum_constraint']>-1e-6:
 w=W.copy();w[sel,pairs[:,0]]+=sol.x;w[sel,pairs[:,1]]-=sol.x
 blob,rr=rb.rebind(p,w,a0,b0);(O/'candidate.nhskin').write_bytes(blob);report['rebind']=rr;report['candidate_sha256']=hashlib.sha256(blob).hexdigest()
(O/'sparse-fit-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
