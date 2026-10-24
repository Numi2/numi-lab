"""Tiny scalar-depth transition repair; unadmitted local reconstruction diagnostic."""
from pathlib import Path
import numpy as np,json,time,hashlib
from scipy.optimize import minimize
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");OUT=E/'baseline-depth-transition-014';OUT.mkdir(exist_ok=True)
ip=E/'bone-clearance-fit-040-shared-columns/attempt-004/fit-input.npz';cp=E/'baseline-layer-depth-013/candidate.npz'
z=np.load(ip);c=np.load(cp);B=z['points_baseline_f32'].astype(float);X=c['points_world_f32'].astype(float);T=c['tetrahedra'];N=2411
vf=B[2*N:]-B[N:2*N];lf=np.linalg.norm(vf,axis=1)
seed=np.array([693,728,729,1218,1219,1248]);cols=np.unique(T[np.any(np.isin(T%N,seed),axis=1)]%N)
affected=np.flatnonzero(np.any(np.isin(T,cols+2*N),axis=1));TT=T[affected]
t0=np.einsum('ij,ij->i',X[2*N+cols]-X[N+cols],vf[cols])/lf[cols]**2
rp=E/'baseline-exact-material-routes-010/route-report.json'
planes=json.loads(rp.read_text())['construction_planes_world']
nodes=[];hits=[];normals=[];margins=[]
for p in planes:
 for layer in [1,2] if p['target_layer_mask']==6 else [2]:
  nodes.append(layer*N+p['row']);hits.append(p['plane_point_world_m']);normals.append(p['unit_exterior_normal_world']);margins.append(p['clearance_m'])
nodes=np.r_[np.array(nodes,int),z['constraint_endpoint_node_rows'][z['constraint_kind']==1]]
H=np.vstack([np.array(hits,float),z['constraint_plane_point_world_m'][z['constraint_kind']==1]])
NM=np.vstack([np.array(normals,float),z['constraint_normal_world'][z['constraint_kind']==1]])
margin=np.r_[margins,z['constraint_clearance_m'][z['constraint_kind']==1]]
kind=np.r_[np.zeros(len(hits),int),np.ones(12,int)]
pr=np.flatnonzero((kind==0)&np.isin(nodes,cols+2*N));mapper={int(v):i for i,v in enumerate(cols)};pc=np.array([mapper[int(nodes[r]%N)] for r in pr],int)
def make(t):
 P=X.copy();P[2*N+cols]=P[N+cols]+t[:,None]*vf[cols];return P
def quality(P,TT):
 V=P[TT];D=np.stack([V[:,i]-V[:,0] for i in(1,2,3)],axis=2);det=np.linalg.det(D)
 e2=np.max(np.stack([np.sum((V[:,i]-V[:,j])**2,axis=1) for i,j in[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]],axis=1),axis=1)
 return det/e2**1.5,det
def con(t):
 P=make(t);qq,_=quality(P,TT);slack=np.einsum('ij,ij->i',NM[pr],P[nodes[pr]]-H[pr])-margin[pr]
 return np.r_[(qq-1.25e-5)*1e3,slack*1e3]
def obj(t):return np.sum((t-1)**2)
st=time.monotonic();sol=minimize(obj,t0,jac=lambda t:2*(t-1),method='SLSQP',bounds=list(zip(np.minimum(.2*lf[cols],.0001)/lf[cols],np.ones(len(cols)))),constraints=[{'type':'ineq','fun':con}],options={'maxiter':80,'ftol':1e-12})
Y=make(sol.x).astype(np.float32).astype(float);qq,det=quality(Y,T);slack=np.einsum('ij,ij->i',NM,Y[nodes]-H)-margin
p=OUT/'candidate.npz';assert not p.exists();np.savez_compressed(p,points_world_f32=Y.astype(np.float32),tetrahedra=T,material_ids=c['material_ids'])
r={'status':'LOCAL_DEPTH_TRANSITION_DIAGNOSTIC_NOT_ADMITTED','inputs':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in[ip,cp,rp]},'candidate_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'elapsed_s':time.monotonic()-st,'solver_success':bool(sol.success),'message':sol.message,'iterations':sol.nit,'columns':cols.tolist(),'touched_cells':len(affected),'fractions_before':t0.tolist(),'fractions_after':sol.x.tolist(),'nonpositive_tets':int(np.sum(det<=0)),'q_below_1e_5':int(np.sum(qq<1e-5)),'minimum_q':float(qq.min()),'route_constraint_violations_F32':int(np.sum(slack[kind==0]<-2e-7)),'O12_violations_F32':int(np.sum(slack[kind==1]<-2e-7)),'outer_nodes_moved':int(np.any(Y[:N]!=B[:N],axis=1).sum()),'limitations':['Old reference J/material/mass/ancestry not qualified.','O12 planes remain uncorrected; no full solid clearance or native admission.']}
(OUT/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
