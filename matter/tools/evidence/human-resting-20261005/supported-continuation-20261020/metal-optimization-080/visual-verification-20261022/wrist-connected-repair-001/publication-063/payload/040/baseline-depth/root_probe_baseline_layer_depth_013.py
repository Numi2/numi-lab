"""Diagnostic local layer-depth reconstruction; not native or mass admission."""
from pathlib import Path
import numpy as np, json,hashlib,time
from scipy.optimize import linprog
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
OUT=E/'baseline-layer-depth-013'
OUT.mkdir(exist_ok=True)
ip=E/'bone-clearance-fit-040-shared-columns/attempt-004/fit-input.npz'
z=np.load(ip,allow_pickle=False);X=z['points_baseline_f32'].astype(float);T=z['tetrahedra'].astype(int);mat=z['material_ids']
N=2411;O=X[:N];vs=X[N:2*N]-O;vf=X[2*N:]-X[N:2*N]
lens=np.linalg.norm(vs,axis=1);lenf=np.linalg.norm(vf,axis=1)
assert np.all(lens>0) and np.all(lenf>0)
lo_s=np.minimum(.2*lens,.0001)/lens;lo_f=np.minimum(.2*lenf,.0001)/lenf
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
Y=X.copy();rows=[];st=time.monotonic()
for c in np.unique(nodes[kind==0]%N):
    rr=np.flatnonzero((kind==0)&(nodes%N==c));layer=nodes[rr]//N
    assert np.all((layer==1)|(layer==2))
    A=np.column_stack((NM[rr]@vs[c],(NM[rr]@vf[c])*(layer==2)))*1000
    b=(margin[rr]-np.einsum('ij,ij->i',NM[rr],O[c]-H[rr]))*1000
    result=linprog([-1.,-1.],A_ub=-A,b_ub=-b,bounds=[(lo_s[c],1.),(lo_f[c],1.)],method='highs')
    rec={'column':int(c),'constraints':rr.tolist(),'success':bool(result.success),'status':int(result.status),'message':result.message,'baseline_skin_depth_m':float(lens[c]),'baseline_fat_depth_m':float(lenf[c]),'fraction_floors':[float(lo_s[c]),float(lo_f[c])]}
    if result.success:
        s,t=result.x;Y[N+c]=O[c]+s*vs[c];Y[2*N+c]=O[c]+s*vs[c]+t*vf[c]
        rec.update(fractions=[float(s),float(t)],minimum_constraint_slack_mm=float(np.min(A@result.x-b)))
    rows.append(rec)
Y=Y.astype(np.float32).astype(float)
V=Y[T];D=np.stack([V[:,i]-V[:,0] for i in [1,2,3]],axis=2);det=np.linalg.det(D)
edge2=np.max(np.stack([np.sum((V[:,i]-V[:,j])**2,axis=1) for i,j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]],axis=1),axis=1)
q=det/edge2**1.5
slack=np.einsum('ij,ij->i',NM,Y[nodes]-H)-margin
p=OUT/'candidate.npz';assert not p.exists()
np.savez_compressed(p,points_world_f32=Y.astype(np.float32),tetrahedra=T.astype(np.int32),material_ids=mat)
r={'schema':'numi.human.local-depth-feasibility.v1','status':'DIAGNOSTIC_NOT_ADMITTED','elapsed_s':time.monotonic()-st,'construction':'true baseline010 route planes; fixed outer skin; shorten original O-I and I-D directions independently; maximize retained fractional depths; no old-reference strain constraints','inputs':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [ip,rp]},'candidate_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'columns':rows,'solved_columns':sum(a['success'] for a in rows),'infeasible_columns':[a['column'] for a in rows if not a['success']],'outer_nodes_moved':int(np.any(Y[:N]!=X[:N],axis=1).sum()),'inner_nodes_moved':int(np.any(Y[N:2*N]!=X[N:2*N],axis=1).sum()),'deep_nodes_moved':int(np.any(Y[2*N:]!=X[2*N:],axis=1).sum()),'nonpositive_tets':int(np.sum(det<=0)),'q_below_1e_5':int(np.sum(q<1e-5)),'minimum_q':float(q.min()),'route_constraint_violations_F32':int(np.sum(slack[kind==0]<-2e-7)),'O12_violations_F32':int(np.sum(slack[kind==1]<-2e-7)),'minimum_gap_m':float(slack.min()),'limits':['O12 baseline outer penetrations deliberately retained and reported, not concealed.','Plane feasibility is not solid containment or topology validation.','Reference, conservative mass, ancestry, detailed transfer and native material admission remain unchanged and unqualified.','If tet inversion results, this proves depth clipping alone is inadequate and does not justify emitting this candidate.']}
(OUT/'report.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='columns'},indent=2))
