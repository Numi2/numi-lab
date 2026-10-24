from pathlib import Path
import hashlib, json, os, time, math
import numpy as np
from scipy.optimize import minimize

BASE=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/global-inner-p1-edge-support-403')
INPUT=BASE/'quality-bistellar-004/candidate-geometry-only.npz'
INPUT_SHA='ff882067af08d4d749d516fdf51fc8f81bb83f85e611f352e394d00c67c52932'
REF403=BASE/'build-002/candidate.npz'
REF403_SHA='9c8c23c8229c46494d19f6fc12d6131667b901864dfab21cd1ef7de6a2fc6744'
REF398=BASE.parent/'global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/local-quality-cavities-398/candidate-with-transfer.npz'
REF398_SHA='8abe15f941dbc01659aee693a07e3edf66f32617c784cf0334410ca11a357890'
OUTDIR=BASE/'quality-coordinate-005'
TARGET=1.0e-5
NODE_ROWS=np.array([55529,55530], dtype=np.int64)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def dets_q(points, tets):
    p=np.asarray(points[tets],dtype=np.float64)
    d=np.einsum('ij,ij->i',p[:,1]-p[:,0],np.cross(p[:,2]-p[:,0],p[:,3]-p[:,0]))
    e0=np.linalg.norm(p[:,1]-p[:,0],axis=1)
    e1=np.linalg.norm(p[:,2]-p[:,0],axis=1)
    e2=np.linalg.norm(p[:,3]-p[:,0],axis=1)
    e3=np.linalg.norm(p[:,2]-p[:,1],axis=1)
    e4=np.linalg.norm(p[:,3]-p[:,1],axis=1)
    e5=np.linalg.norm(p[:,3]-p[:,2],axis=1)
    L=np.maximum.reduce([e0,e1,e2,e3,e4,e5])
    return d/(L**3), d, L

def bary_of(point, tet_points):
    a=np.asarray(tet_points,dtype=np.float64)
    M=np.column_stack((a[1]-a[0],a[2]-a[0],a[3]-a[0]))
    uvw=np.linalg.solve(M,np.asarray(point,dtype=np.float64)-a[0])
    return np.array([1.0-uvw.sum(),uvw[0],uvw[1],uvw[2]],dtype=np.float64)

start=time.time()
if OUTDIR.exists():
    raise SystemExit(f'fail-closed: output directory already exists: {OUTDIR}')
for p,h in ((INPUT,INPUT_SHA),(REF403,REF403_SHA),(REF398,REF398_SHA)):
    if not p.is_file() or sha(p)!=h:
        raise SystemExit(f'fail-closed input pin mismatch: {p} actual={sha(p) if p.exists() else "missing"} expected={h}')
OUTDIR.mkdir(parents=True,exist_ok=False)
z=np.load(INPUT)
r403=np.load(REF403)
r398=np.load(REF398)
X0=z['points_world_m']
T=z['tetrahedra'].astype(np.int64,copy=False)
if X0.dtype!=np.float32 or X0.shape!=(55553,3) or T.shape!=(248662,4):
    raise SystemExit(f'unexpected input geometry shape/dtype {X0.shape}/{X0.dtype} {T.shape}')
if not np.array_equal(X0,r403['points_world_m']):
    raise SystemExit('004 point rows do not match pinned 403 build-002 node prefix')
for key in ('region_attributes','physical_outer_node_ids','physical_outer_faces','physical_shell_faces','construction_closure_faces'):
    if not np.array_equal(z[key],r403[key]): raise SystemExit(f'004 immutable boundary/material field differs from pinned 403 build-002: {key}')
X=np.array(X0,copy=True)
allq0,alld0,allL0=dets_q(X,T)
if np.any(alld0<=0): raise SystemExit(f'input has {int(np.sum(alld0<=0))} nonpositive oriented determinants')
node_ids=NODE_ROWS.tolist()
stars={int(v):np.flatnonzero(np.any(T==v,axis=1)) for v in node_ids}
if set(node_ids).intersection(set(T[stars[node_ids[0]]].ravel())) - set(node_ids) != set():
    pass
shared=set(stars[node_ids[0]].tolist()) & set(stars[node_ids[1]].tolist())
star_rows=np.unique(np.concatenate([stars[v] for v in node_ids]))
star_tets=T[star_rows]
if not shared:
    # independent points are expected, but solve the same objective with shared rows handled anyway
    pass
radii=[]; minedges=[]
for v in node_ids:
    rows=stars[v]
    other=T[rows].reshape(-1)
    other=other[other!=v]
    lengths=np.linalg.norm(X0[other].astype(np.float64)-X0[v].astype(np.float64),axis=1)
    m=float(lengths.min())
    minedges.append(m)
    radii.append(min(0.0005,0.05*m))
radii=np.asarray(radii,dtype=np.float64)
q0,_,_=dets_q(X,star_tets)
# Ensure ordering in the local tet union matches fixed input rows.
initial_min=float(q0.min())

def apply_z(zv):
    pts=np.array(X0,dtype=np.float64,copy=True)
    for i,v in enumerate(node_ids): pts[v]=X0[v].astype(np.float64)+radii[i]*zv[3*i:3*i+3]
    return pts

def local_q(zv):
    pts=apply_z(zv)
    return dets_q(pts,star_tets)[0]

def objective(w):
    zz=w[:6]; t=w[6]
    return -float(t)+1e-6*float(np.dot(zz,zz))

def cons_quality(w):
    return local_q(w[:6])/TARGET-float(w[6])

def cons_ball_0(w): return 1.0-float(np.dot(w[0:3],w[0:3]))
def cons_ball_1(w): return 1.0-float(np.dot(w[3:6],w[3:6]))
cons=[{'type':'ineq','fun':cons_quality},{'type':'ineq','fun':cons_ball_0},{'type':'ineq','fun':cons_ball_1}]
w0=np.zeros(7,dtype=np.float64); w0[6]=initial_min/TARGET
res=minimize(objective,w0,method='SLSQP',bounds=[(-1,1)]*6+[(0,100)],constraints=cons,options={'maxiter':800,'ftol':1e-12,'disp':False})
Xd64=apply_z(res.x[:6])
Xd=np.asarray(Xd64,dtype=np.float32)
# Only the two nominated support points may move, and displacement is measured after F32 emission.
changed=np.flatnonzero(np.any(Xd!=X0,axis=1))
actual_dx=Xd[NODE_ROWS].astype(np.float64)-X0[NODE_ROWS].astype(np.float64)
actual_norms=np.linalg.norm(actual_dx,axis=1)
qstar,dstar,Lstar=dets_q(Xd,star_tets)
qglobal,dglobal,Lglobal=dets_q(Xd,T)
quality_pass=bool(np.all(qstar>=TARGET-2e-12) and np.all(dstar>0) and np.all(actual_norms<=radii+1e-12))
# Keep the diagnostic even if no admissible candidate was found.
report={
 'schema':'numi.human.global-inner-p1-quality-coordinate-005.report.v1',
 'native_admission':False,'physical_admission':False,
 'input_candidate_path':str(INPUT),'input_candidate_sha256':INPUT_SHA,
 'source403_path':str(REF403),'source403_sha256':REF403_SHA,
 'source398_path':str(REF398),'source398_sha256':REF398_SHA,
 'target_normalized_determinant':TARGET,
 'nodes':node_ids,'input_star_rows':{str(v):stars[v].tolist() for v in node_ids},
 'shared_star_rows':sorted(shared),'star_tet_rows':star_rows.tolist(),
 'initial_star_min_q':initial_min,'initial_star_q_by_row':{str(int(r)):float(q) for r,q in zip(star_rows,q0)},
 'min_incident_edge_m':minedges,'displacement_radius_m':radii.tolist(),
 'optimizer':{'method':'SLSQP max-min normalized determinant','success':bool(res.success),'status':int(res.status),'message':str(res.message),'iterations':int(res.nit),'objective':float(res.fun),'min_q_target_ratio_float64':float(res.x[6]),'normalized_displacement_variables':res.x[:6].tolist(),'predicted_star_q_min_float64':float(local_q(res.x[:6]).min())},
 'float32_emission':{'changed_node_rows':changed.tolist(),'euclidean_displacement_m':actual_norms.tolist(),'star_q_by_row':{str(int(r)):float(q) for r,q in zip(star_rows,qstar)},'star_det_by_row_m3':{str(int(r)):float(d) for r,d in zip(star_rows,dstar)},'star_max_edge_m_by_row':{str(int(r)):float(L) for r,L in zip(star_rows,Lstar)},'star_min_q':float(qstar.min()),'global_min_q':float(qglobal.min()),'global_count_below_floor':int(np.sum(qglobal<TARGET)),'global_nonpositive_count':int(np.sum(dglobal<=0))},
 'geometry_quality_gate_pass':quality_pass,
 'elapsed_seconds':time.time()-start,
 'limitations':['quality-only geometry candidate; no material mass, transfer, field, boundary, native, or physical admission claim']}

if quality_pass and set(changed.tolist()).issubset(set(node_ids)):
    # Find an original398 containing tet for each moved point and compose barycentric source ancestry.
    X398=np.asarray(r398['points_world_m'],dtype=np.float64)
    T398=np.asarray(r398['tetrahedra'],dtype=np.int64)
    lo=np.min(X398[T398],axis=1); hi=np.max(X398[T398],axis=1)
    node_parent_rows=[]; node_parent_bary=[]; new_ancestry={}
    for v in node_ids:
        p=Xd[v].astype(np.float64)
        cand=np.flatnonzero(np.all(lo<=p+2e-8,axis=1)&np.all(hi>=p-2e-8,axis=1))
        choices=[]
        for rr in cand:
            ids=T398[rr]
            try: b=bary_of(p,X398[ids])
            except np.linalg.LinAlgError: continue
            if b.min()>=-2e-9 and b.max()<=1+2e-9 and abs(float(b.sum())-1)<=2e-12:
                choices.append((int(rr),b))
        if not choices:
            report['geometry_quality_gate_pass']=False
            report.setdefault('ancestry_failures',[]).append({'node':v,'candidate_398_rows_examined':int(len(cand)),'reason':'no containing original398 tetra for emitted F32 coordinate'})
            continue
        rr,b=min(choices,key=lambda x:x[0])
        ids=T398[rr]
        accum={}
        ptr=r398['node_source_indptr']; sids=r398['node_source_indices']; sw=r398['node_source_weights']
        for ni,bw in zip(ids,b):
            if bw==0: continue
            a0,a1=int(ptr[ni]),int(ptr[ni+1])
            for sid,ww in zip(sids[a0:a1],sw[a0:a1]):
                accum[int(sid)]=accum.get(int(sid),0.0)+float(bw)*float(ww)
        items=sorted((sid,w) for sid,w in accum.items() if abs(w)>1e-16)
        total=sum(w for _,w in items)
        items=[(sid,w/total) for sid,w in items]
        new_ancestry[v]=items
        node_parent_rows.append(rr); node_parent_bary.append(b.tolist())
    if len(node_parent_rows)==len(node_ids):
        ptr403=np.asarray(r403['node_source_indptr'],dtype=np.int64)
        ids403=np.asarray(r403['node_source_indices'],dtype=np.int64)
        w403=np.asarray(r403['node_source_weights'],dtype=np.float64)
        outptr=[0]; outids=[]; outw=[]
        for row in range(len(Xd)):
            if row in new_ancestry:
                items=new_ancestry[row]
            else:
                a,b=int(ptr403[row]),int(ptr403[row+1]); items=list(zip(ids403[a:b].tolist(),w403[a:b].tolist()))
            outids.extend(int(sid) for sid,_ in items); outw.extend(float(w) for _,w in items); outptr.append(len(outids))
        out={k:np.array(z[k]) for k in z.files}
        out['points_world_m']=Xd
        out['node_source_indptr']=np.asarray(outptr,dtype=np.int64)
        out['node_source_indices']=np.asarray(outids,dtype=np.int64)
        out['node_source_weights']=np.asarray(outw,dtype=np.float64)
        out['coordinate_fit_node_rows']=NODE_ROWS
        out['coordinate_fit_parent_398_tet_rows']=np.asarray(node_parent_rows,dtype=np.int64)
        out['coordinate_fit_parent_398_barycentric']=np.asarray(node_parent_bary,dtype=np.float64)
        out['coordinate_fit_displacements_m']=actual_dx.astype(np.float64)
        # Verify every output mapping and immutable geometry/table pins before writing.
        if not np.array_equal(out['points_world_m'][np.setdiff1d(np.arange(len(X0)),NODE_ROWS)],X0[np.setdiff1d(np.arange(len(X0)),NODE_ROWS)]): raise SystemExit('unexpected non-support coordinate change')
        for k in ('tetrahedra','region_attributes','physical_outer_node_ids','physical_outer_faces','physical_shell_faces','construction_closure_faces','parent_input_candidate_offsets','parent_input_candidate_rows'):
            if not np.array_equal(out[k],z[k]): raise SystemExit(f'input immutable field changed: {k}')
        if not np.allclose(np.add.reduceat(out['node_source_weights'],out['node_source_indptr'][:-1]),1,atol=2e-12):
            # reduceat mishandles empty rows; verify robustly below instead.
            for row in range(len(Xd)):
                a,b=out['node_source_indptr'][row:row+2]
                if not np.isclose(out['node_source_weights'][a:b].sum(),1,atol=2e-12): raise SystemExit(f'bad node source CSR closure row {row}')
        for row in range(len(Xd)):
            a,b=out['node_source_indptr'][row:row+2]
            if not np.isclose(out['node_source_weights'][a:b].sum(),1,atol=2e-12): raise SystemExit(f'bad node source CSR closure row {row}')
        candpath=OUTDIR/'candidate.npz'; tmp=candpath.with_suffix('.npz.tmp')
        with open(tmp,'wb') as f: np.savez_compressed(f,**out)
        os.replace(tmp,candpath)
        report['candidate_path']=str(candpath); report['candidate_sha256']=sha(candpath)
        report['source_ancestry']={'method':'barycentric composition through a containing original398 tetrahedron','node_source_rows_replaced':node_ids,'parent_398_tet_rows':node_parent_rows,'parent_barycentric':node_parent_bary,'max_csr_row_sum_error':float(max(abs(out['node_source_weights'][out['node_source_indptr'][i]:out['node_source_indptr'][i+1]].sum()-1) for i in range(len(Xd))))}
    else:
        report['geometry_quality_gate_pass']=False

reportpath=OUTDIR/'optimization-report.json'
tmp=reportpath.with_suffix('.json.tmp')
tmp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
os.replace(tmp,reportpath)
print(json.dumps({'report':str(reportpath),'report_sha256':sha(reportpath),'candidate':report.get('candidate_path'),'candidate_sha256':report.get('candidate_sha256'),'geometry_quality_gate_pass':report['geometry_quality_gate_pass'],'optimizer':report['optimizer'],'float32':report['float32_emission'],'ancestry':report.get('source_ancestry',report.get('ancestry_failures'))},indent=2))
