"""Connected static skin/fat source repair; never a runtime physics solver."""
from pathlib import Path
import numpy as np,json,sys,hashlib,importlib.util,time
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.optimize import minimize
D=Path('/Users/numi/Work/numi-human-connected-surface-069')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def solve_translations(O,F,planes,rings=3):
    N=len(O);edges=np.unique(np.sort(np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1),axis=0)
    adj=sparse.coo_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(N,N)).tocsr()
    active=set(int(p['column_row'])for p in planes)
    for _ in range(rings):
        active.update(int(j)for i in list(active)for j in adj.indices[adj.indptr[i]:adj.indptr[i+1]])
    free=np.asarray(sorted(active),np.int64);index=np.full(N,-1,np.int64);index[free]=np.arange(len(free))
    # A fixed source contour bounds the connected authoring patch. The regularizer
    # is dimensionless graph construction energy, not an extra physical force.
    K=(sparse.diags(np.asarray(adj.sum(axis=1)).ravel()+.05)-adj)[free][:,free].tocsc()
    lu=splu(K);cols=np.asarray([index[int(p['column_row'])]for p in planes])
    normals=np.asarray([p['normal']for p in planes],float);b=np.asarray([p['deficit_m']for p in planes],float)*1000.
    Ecols=np.zeros((len(free),len(planes)));Ecols[cols,np.arange(len(planes))]=1.
    H=lu.solve(Ecols);S=H[cols,:]*(normals@normals.T);S=(S+S.T)*.5
    cp=minimize(lambda l:(.5*l@S@l-b@l,S@l-b),np.zeros(len(b)),jac=True,bounds=[(0,None)]*len(b),method='L-BFGS-B',options={'ftol':1e-15,'gtol':1e-11,'maxiter':2000,'maxls':100})
    U=np.zeros((N,3));U[free]=H@(cp.x[:,None]*normals)*.001
    slack=np.einsum('ij,ij->i',normals,U[[int(p['column_row'])for p in planes]])-b*.001
    assert np.min(slack)>=-2e-9,('infeasible construction',cp.message,float(slack.min()))
    return U,{'rings':rings,'free_columns':len(free),'solver_success':bool(cp.success),'solver_message':str(cp.message),'iterations':int(cp.nit),'minimum_plane_slack_m':float(slack.min()),'outside_patch_exact_zero':bool(np.array_equal(U[np.setdiff1d(np.arange(N),free)],np.zeros((N-len(free),3))))}
def main():
    plane_path,out=map(Path,sys.argv[1:]);assert not out.exists();out.mkdir(parents=True)
    # Deliberately flexible reader only for the construction-owner records; all
    # required plane values and source point identities are validated below.
    obj=json.loads(plane_path.read_text());assert sha(plane_path)=='717921214fee88ea7891d5c6c93f67596be1c16c4ada26772ccba24351b9eaef';records=obj['results'];planes=[]
    base=D/'connected-capacity-reference-115/candidate.npz'
    old=D/'whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz'
    assert sha(base)=='abff4290774cc96a17f6c68a21fe86fc3933a1878e0bd6512733e99589fb6dd9'
    assert sha(old)=='5a35470223ce12343cc217be7822d182802e99f1c423bb52b8409f153f9a49db'
    z=np.load(old);v=np.load(base);X=v['points_world_f32'];R=v['proposed_reference_points_m']
    for r in records:
        assert r['status']=='PASS';row=int(r['query_row']);col=int(r['column_row']);layer=r['layer'];assert layer in('O','I')
        assert row==col+({'O':0,'I':2411}[layer]) and 0<=col<2411
        plane=r['report']['plane'];n=np.asarray(plane['unit_normal_world'],float);target=float(plane['minimum_signed_distance_m'])
        assert n.shape==(3,)and np.isfinite(n).all()and abs(np.linalg.norm(n)-1)<1e-8
        assert np.allclose(X[row].astype(float),np.asarray(r['world_start_m']),atol=0,rtol=0)
        planes.append({'column_row':col,'query_row':row,'layer':layer,'normal':n.tolist(),'target':target,'deficit_m':target-n@X[row].astype(float),'body':r['body'],'bone_stable_id':r['bone_stable_id']})
    assert len(planes)>=12 and len({p['column_row']for p in planes if p['layer']=='O'})==12
    U,solve=solve_translations(X[:2411],z['source_O_faces'],planes)
    delta=np.tile(U,(3,1));Y=X.astype(float).copy();Y[:7233]+=delta
    native=E/'native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003/candidate.npz'
    nt=np.load(native)['old_tetrahedra'];generated={}
    for node,parent,w in zip(z['newnode_ancestry_output_row'],z['newnode_ancestry_source_tet_row'],z['newnode_ancestry_barycentric']):
        generated.setdefault(int(node),[]).append(np.asarray(w)@delta[nt[int(parent)]])
    for node,values in generated.items():
        assert np.max(np.linalg.norm(np.stack(values)-values[0],axis=1))<=1e-10;Y[node]+=values[0]
    Y=Y.astype(np.float32);NR=R+(Y.astype(float)-X.astype(float))
    # Identical layer translations preserve the existing column OI and ID vectors
    # in float64; emitted rounding is measured, not assumed exact.
    Ierr=(Y[2411:4822].astype(float)-Y[:2411])-(X[2411:4822].astype(float)-X[:2411])
    Derr=(Y[4822:7233].astype(float)-Y[2411:4822])-(X[4822:7233].astype(float)-X[2411:4822])
    # Use the retained physical-detail displacement map to keep both surfaces in
    # the same proposal; no render-only replacement is installed.
    from scipy.sparse import load_npz
    Wpath=D/'whole-reference-tetgen-084-bounded/connectivity-001/run-001/transfer-map-075-084-002/Wnew-detail-to-084-nodes.npz'
    sys.path.insert(0,str(E/'E/accepted-transfer-row-adapter-075'));from load_transfer_075 import load_transfer075
    P075=E/'accepted-transfer-row-export-075/run-001/source-preflight'
    W075,detail,coarse,DF=load_transfer075(P075/'accepted-transfer-rows.csv',P075/'accepted-coarse-reference.csv',P075/'accepted-coarse-faces.csv',E/'bilateral-hand-functional-rebind-020/run-001/candidate.nhskin',
      expected_sha256={'transfer_rows':'71e0ff115f74f1c072dc39e25d8c0de869c892dbfaf3424c792c2f674172fb14','coarse_reference':'9e26673335c571d5920f5cd43e52daf4b240de72feadf6d9e1cef92cf80807a2','coarse_faces':'7311267392275649abeb3ecaf6a5ca43eb4d67cf2a9b7f13a9c692587c36b1e2','detail_payload':'7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb'})
    W=load_npz(Wpath);DT=(detail.astype(float)+W@(Y.astype(float)-z['current037_points_m_f32'].astype(float))).astype(np.float32)
    C=out/'candidate.npz';np.savez_compressed(C,points_world_f32=Y,proposed_reference_points_m=NR,source_reference_points_m=R,source_points_world_f32=X,tetrahedra=z['tetrahedra_zero_based'],material_ids=z['material_ids'],source_input_row=z['source_input_row'],column_displacements_m=U,detailed_skin_points_world_f32=DT,detailed_skin_faces=DF,surface_faces_zero_based=z['surface_faces_zero_based'],surface_face_markers=z['surface_face_markers'])
    sys.path.insert(0,str(D/'connected-cell-trust-111/matter/tools'));from human_skin_fat_connected_repair import cell_values_gradients
    J,_,q,_=cell_values_gradients(Y,z['reference_points_m'],z['tetrahedra_zero_based'])
    sys.path.insert(0,str(E/'incremental-exact-audit-owner-253/src'));pred=E/'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py'
    spec=importlib.util.spec_from_file_location('numilab_human._source122',pred);s=importlib.util.module_from_spec(spec);sys.modules[spec.name]=s;spec.loader.exec_module(s)
    result={'source_plane_sha256':sha(plane_path),'candidate_sha256':sha(C),'construction_solver':solve,'plane_constraints':planes,'maximum_column_translation_m':float(np.linalg.norm(U,axis=1).max()),'maximum_emitted_OI_vector_change_m':float(np.linalg.norm(Ierr,axis=1).max()),'maximum_emitted_ID_vector_change_m':float(np.linalg.norm(Derr,axis=1).max()),'old_connectivity_invalid_cells':int((J<=0).sum()),'old_connectivity_below_quality':int((q<1e-5).sum()),'surfaces':{},'accepted_physical_steps':0,'geometry_admitted':False,'interpretation':'Simultaneous connected source-reference construction with identical translations of all three column layers. Source-exit constraints and topology anchoring are authoring constraints; never runtime locks or forces. Native mass/ancestry/transfer and full clearance remain required.'}
    for label,pts,faces in [('PLC',Y,z['surface_faces_zero_based']),('detailed',DT,DF)]:
        t=time.monotonic();rec,_=s.float32_surface_records(pts,faces,source_seam_groups=());a=s._audit_self_pairs_touching_faces(rec,set(range(len(faces))));pairs=np.asarray(a['triangle_pairs'],np.int64).reshape(-1,2);np.save(out/(label+'-surface-pairs.npy'),pairs);result['surfaces'][label]={'self_pairs':len(pairs),'complete':a['complete_surface_audit'],'wall_seconds':time.monotonic()-t}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='plane_constraints'},indent=2))
if __name__=='__main__':main()
