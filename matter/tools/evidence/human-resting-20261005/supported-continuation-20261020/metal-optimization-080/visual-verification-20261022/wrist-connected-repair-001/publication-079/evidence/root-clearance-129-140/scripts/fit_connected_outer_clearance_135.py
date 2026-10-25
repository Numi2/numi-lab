from pathlib import Path
import sys,csv,json,hashlib,importlib.util,time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import minimize
D=Path('/Users/numi/Work/numi-human-connected-surface-069')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def solve_connected(P,F,node_rows,normals,gaps,ring_count=3,regularization=.05,separation_rows=None,separation_rhs=None):
    n=len(P); edges=np.unique(np.sort(np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1),axis=0)
    A=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
    selected=set(map(int,node_rows)); frontier=set(selected)
    for _ in range(ring_count):
        new=set()
        for r in frontier:new.update(map(int,A.indices[A.indptr[r]:A.indptr[r+1]]))
        frontier=new-selected;selected.update(new)
    if separation_rows is not None:
        selected.update(map(int,np.unique(separation_rows.indices//3)))
    free=np.array(sorted(selected),np.int64);loc=np.full(n,-1,np.int64);loc[free]=np.arange(len(free))
    K=(sparse.diags(np.asarray(A.sum(axis=1)).ravel()+regularization)-A)[free][:,free].tocsc()
    factor=splu(K)
    rr=np.repeat(np.arange(len(node_rows)),3);cc=(3*np.asarray(node_rows)[:,None]+np.arange(3)).ravel()
    C=sparse.csr_matrix((normals.ravel(),(rr,cc)),shape=(len(node_rows),3*n))
    target=np.asarray(gaps)
    if separation_rows is not None:
        C=sparse.vstack([C,separation_rows],format='csr');target=np.r_[target,separation_rhs]
    Cs=[C[:,3*free+k] for k in range(3)]
    response=[factor.solve(Ck.T.toarray()) for Ck in Cs]
    S=sum(Ck@Rk for Ck,Rk in zip(Cs,response));S=.5*(S+S.T)
    def fun(x):return .5*x@S@x-target@x,S@x-target
    result=minimize(fun,np.zeros(len(target)),jac=True,bounds=[(0,None)]*len(target),method='L-BFGS-B',options={'maxiter':4000,'ftol':1e-15,'gtol':1e-12,'maxls':50,'maxcor':30})
    U=np.zeros_like(P)
    for k in range(3):U[free,k]=response[k]@result.x
    residual=np.asarray(C@U.reshape(-1)).ravel()-target
    if residual.min(initial=0)<-1e-8:raise ValueError(f'plane feasibility failed {residual.min()}')
    return U,dict(free_nodes=len(free),constraints=len(target),bone_constraints=len(node_rows),surface_rows=len(target)-len(node_rows),minimum_constraint_slack_m=float(residual.min(initial=0)),
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
    constraint_inputs=[{'candidate_sha256':sha(C),'query_summary_sha256':sha(Q/'summary.json')}]
    if len(sys.argv)>7:
        for item in json.load(open(sys.argv[7])):
            Cextra,Qextra=map(Path,(item['candidate'],item['query_dir']))
            rextra=json.load(open(Qextra/'summary.json'));assert rextra['candidate_sha256']==sha(Cextra) and rextra['exact_PLC_surface_pairs']==0
            zextra=np.load(Cextra);assert np.array_equal(zextra['source_O_faces'],z['source_O_faces'])
            for key in ('node_kind','node_source_row'):assert np.array_equal(zextra[key],z[key])
            for v in rextra['inside_witnesses']:
                assert np.array_equal(zextra['points_world_m'].astype(np.float32)[int(v['query_row'])],np.array(v['point_m'],np.float32))
            r['inside_witnesses'].extend(rextra['inside_witnesses'])
            constraint_inputs.append({'candidate_sha256':sha(Cextra),'query_summary_sha256':sha(Qextra/'summary.json')})
    retained={}
    for v in r['inside_witnesses']:
        retained[(int(v['query_row']),int(v['bank_body']),int(v['bank_stable_id']))]=v
    r['inside_witnesses']=list(retained.values())
    rows=[];normals=[];gaps=[];witness=[]
    for v in r['inside_witnesses']:
        i=int(v['query_row']);body=int(v['bank_body']);local=np.array([float(v['query_local_'+a]) for a in 'xyz'])
        expected=(np.array(v['point_m'],float)-pos[body])@rot[body].T-off[body]
        assert np.allclose(local,expected,rtol=0,atol=1e-12)
        closest=np.array([float(v['closest_'+a]) for a in 'xyz']);d=closest-local;length=np.linalg.norm(d);assert length>0
        nl=d/length;nw=rot[body].T@nl;scale=np.linalg.norm(nw);nw/=scale
        bound=(nl@(closest+off[body])+(rot[body].T@nl)@pos[body]+clearance)/scale
        rows.append(i);normals.append(nw);gaps.append(bound-nw@P[i])
        witness.append(dict(node=i,body=body,bone=int(v['bank_stable_id']),normal_world=nw.tolist(),plane_minimum_world_m=float(bound),nearest_distance_local_m=float(length)))
    assert rows
    extraC=None;extraB=None;separators=[]
    if len(sys.argv)>6:
        pairs_path=Path(sys.argv[5]);trial_path=Path(sys.argv[6])
        trial=np.load(trial_path)['points_world_m'];pairs=np.load(pairs_path)
        sys.path.insert(0,str(D/'publication-078/canonical/matter/tools'))
        from human_skin_fat_surface_separation import build_pair_inequalities
        rowsC=[];rowsB=[];identity=sparse.eye(len(P),format='csr')
        for a,b in pairs:
            ci,bi,meta=build_pair_inequalities(z['source_O_faces'][a],z['source_O_faces'][b],P,identity,P,P,candidate_detail_positions=trial,pair_id=[int(a),int(b)])
            rowsC.append(ci);rowsB.append(bi-ci@P.reshape(-1));separators.append(meta)
        extraC=sparse.vstack(rowsC,format='csr');extraB=np.concatenate(rowsB)
    U,stats=solve_connected(P,z['source_O_faces'],np.array(rows),np.array(normals),np.array(gaps),separation_rows=extraC,separation_rhs=extraB)
    if stats['max_translation_m']>.005:
        (O/'rejected-solve.json').write_text(json.dumps(dict(status='REJECTED_AT_DISPLACEMENT_GATE',maximum_allowed_m=.005,solver=stats),indent=2)+'\n')
        raise ValueError('connected clearance displacement exceeds 5mm construction bound')
    data['points_world_m']=P+U;data['node_connected_clearance_displacement_m']=U;data['points_before_connected_clearance_m']=P
    np.savez_compressed(O/'candidate.npz',**data)
    report=dict(status='CONNECTED_OUTER_CLEARANCE_CONSTRUCTION_ONLY',input_sha256=sha(C),query_summary_sha256=sha(Q/'summary.json'),
      constraint_input_assets=constraint_inputs,script_sha256=sha(__file__),candidate_sha256=sha(O/'candidate.npz'),clearance_m=clearance,solver=stats,
      constraints=witness,surface_separators=separators,material_and_native_admission=False,accepted_physical_steps=0)
    (O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ('constraints','surface_separators')}))
if __name__=='__main__':main()
