from pathlib import Path
import sys,csv,json,hashlib,importlib.util,time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import minimize
D=Path('/Users/numi/Work/numi-human-connected-surface-069')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def solve_connected(P,F,node_rows,normals,gaps,ring_count=3,regularization=.05):
    n=len(P); edges=np.unique(np.sort(np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1),axis=0)
    A=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
    selected=set(map(int,node_rows)); frontier=set(selected)
    for _ in range(ring_count):
        new=set()
        for r in frontier:new.update(map(int,A.indices[A.indptr[r]:A.indptr[r+1]]))
        frontier=new-selected;selected.update(new)
    free=np.array(sorted(selected),np.int64);loc=np.full(n,-1,np.int64);loc[free]=np.arange(len(free))
    K=(sparse.diags(np.asarray(A.sum(axis=1)).ravel()+regularization)-A)[free][:,free].tocsc()
    factor=splu(K);ir=loc[node_rows]
    rhs=np.zeros((len(free),len(ir)));rhs[ir,np.arange(len(ir))]=1
    response=factor.solve(rhs);S=response[ir,:]*(normals@normals.T);S=.5*(S+S.T)
    def fun(x):return .5*x@S@x-gaps@x,S@x-gaps
    result=minimize(fun,np.zeros(len(ir)),jac=True,bounds=[(0,None)]*len(ir),method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-15,'gtol':1e-12,'maxls':40})
    U=np.zeros_like(P);U[free]=response@(result.x[:,None]*normals)
    residual=np.einsum('ij,ij->i',normals,U[node_rows])-gaps
    if residual.min(initial=0)<-1e-8:raise ValueError(f'plane feasibility failed {residual.min()}')
    return U,dict(free_nodes=len(free),constraints=len(ir),minimum_constraint_slack_m=float(residual.min(initial=0)),
      max_translation_m=float(np.linalg.norm(U,axis=1).max(initial=0)),dual_success=bool(result.success),dual_message=str(result.message),dual_iterations=int(result.nit))
def main():
    C,Q,O=map(Path,sys.argv[1:4]);assert not O.exists();O.mkdir(parents=True)
    clearance=float(sys.argv[4]) if len(sys.argv)>4 else .00005
    assert clearance>=0 and np.isfinite(clearance)
    r=json.load(open(Q/'summary.json'));assert r['candidate_sha256']==sha(C) and r['exact_PLC_surface_pairs']==0
    z=np.load(C);data={k:z[k] for k in z.files};P=np.asarray(z['points_world_m'],float);X=P.astype(np.float32)
    A=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
    sys.path.insert(0,str(A.parent));import human_skin_fat_bone_audit as audit
    bank=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
    tri,ids,gids,bodies,rot,off,pos,keys,pins,exact=audit.read_bank_inputs(bank,E/'route-audit-counter-fix-030/canonical/matter')
    rows=[];normals=[];gaps=[];witness=[]
    for v in r['inside_witnesses']:
        i=int(v['query_row']);body=int(v['bank_body']);local=np.array([float(v['query_local_'+a]) for a in 'xyz'])
        expected=(X[i].astype(float)-pos[body])@rot[body].T-off[body]
        assert np.allclose(local,expected,rtol=0,atol=1e-12)
        closest=np.array([float(v['closest_'+a]) for a in 'xyz']);d=closest-local;length=np.linalg.norm(d);assert length>0
        nl=d/length;nw=rot[body].T@nl;scale=np.linalg.norm(nw);nw/=scale
        bound=(nl@(closest+off[body])+(rot[body].T@nl)@pos[body]+clearance)/scale
        rows.append(i);normals.append(nw);gaps.append(bound-nw@P[i])
        witness.append(dict(node=i,body=body,bone=int(v['bank_stable_id']),normal_world=nw.tolist(),plane_minimum_world_m=float(bound),nearest_distance_local_m=float(length)))
    assert rows
    U,stats=solve_connected(P,z['source_O_faces'],np.array(rows),np.array(normals),np.array(gaps))
    data['points_world_m']=P+U;data['node_connected_clearance_displacement_m']=U;data['points_before_connected_clearance_m']=P
    np.savez_compressed(O/'candidate.npz',**data)
    report=dict(status='CONNECTED_OUTER_CLEARANCE_CONSTRUCTION_ONLY',input_sha256=sha(C),query_summary_sha256=sha(Q/'summary.json'),
      script_sha256=sha(__file__),candidate_sha256=sha(O/'candidate.npz'),clearance_m=clearance,solver=stats,
      constraints=witness,material_and_native_admission=False,accepted_physical_steps=0)
    (O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='constraints'}))
if __name__=='__main__':main()
