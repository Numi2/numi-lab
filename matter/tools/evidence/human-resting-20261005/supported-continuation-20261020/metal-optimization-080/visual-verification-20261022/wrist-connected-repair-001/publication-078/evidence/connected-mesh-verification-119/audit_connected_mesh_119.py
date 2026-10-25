from pathlib import Path
import sys,json,csv,hashlib,time,subprocess,importlib.util,collections
import numpy as np
D=Path('/Users/numi/Work/numi-human-connected-surface-069')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    C,P=map(Path,sys.argv[1:]);assert C.is_file() and not P.exists();P.mkdir(parents=True)
    started=time.monotonic();z=np.load(C,allow_pickle=False);X=z['points_world_f32'];R=z['reference_points_m'];T=z['tetrahedra'];mat=z['material_ids']
    assert X.dtype==np.float32 and X.ndim==2 and X.shape[1]==3 and R.shape==X.shape and np.isfinite(X).all() and np.isfinite(R).all()
    assert T.ndim==2 and T.shape[1]==4 and T.min()>=0 and T.max()<len(X)
    sys.path.insert(0,str(D/'connected-cell-trust-111/matter/tools'))
    from human_skin_fat_connected_repair import cell_values_gradients
    J,_,q,_=cell_values_gradients(X,R,T)
    report={'candidate_sha256':sha(C),'points':len(X),'cells':len(T),'invalid_cells':int((J<=0).sum()),'below_quality_1e_minus5':int((q<1e-5).sum()),'q_min':float(q.min()),'J_min':float(J.min()),'accepted_physical_steps':0,'geometry_admitted':False}
    (P/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    # PLC surfaces include the O/I/D interfaces and side caps, excluding internal
    # tetrahedral faces whose TetGen marker is zero.
    F=z['surface_faces_zero_based'];M=z['surface_face_markers'];F=F[M!=0]
    pred=E/'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py'
    sys.path.insert(0,str(E/'incremental-exact-audit-owner-253/src'))
    spec=importlib.util.spec_from_file_location('numilab_human._connected119',pred);s=importlib.util.module_from_spec(spec);sys.modules[spec.name]=s;spec.loader.exec_module(s)
    records,_=s.float32_surface_records(X,F,source_seam_groups=())
    sr=s._audit_self_pairs_touching_faces(records,set(range(len(F))))
    pairs=np.asarray(sr['triangle_pairs'],np.int64).reshape(-1,2);np.save(P/'surface-self-pairs.npy',pairs)
    report.update(exact_PLC_surface_pairs=len(pairs),complete_surface_audit=sr['complete_surface_audit'],surface_predicate_sha256=sha(pred))
    # Cell quality is reported independently; a surface crossing stops the
    # downstream audit because its reconstruction is the next required change.
    if len(pairs):
        report.update(status='STOPPED_AT_SURFACE_GATE',wall_seconds=time.monotonic()-started)
        (P/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return
    bank=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
    exporter=E/'E/generic-static-query-042/prepare_generic_query_points.py'
    binary=Path('/tmp/generic-query-overlay-build-042-work/matter/numi-matter-human-skin-fat-static-query')
    assert sha(binary)=='5dc7b5ce2a96f58bd03ccc85e38645d9da2ec8be105a532c461a3e0cc6d25460'
    Q=P/'query';Q.mkdir()
    args=[PY,str(exporter),'--candidate-npz',str(C),'--input-sha256',sha(C),'--field','points_world_f32','--output',str(Q/'points.csv')]
    cp=subprocess.run(args,capture_output=True,text=True);(Q/'prepare.log').write_text(cp.stdout+cp.stderr);assert cp.returncode==0
    args=[str(binary),str(bank/'exact-emitted.triangles.csv'),str(bank/'exact-emitted.bodies.csv'),str(Q/'points.csv'),str(Q/'pairs.csv'),str(Q/'nodes.csv'),'185']
    cp=subprocess.run(args,capture_output=True,text=True);(Q/'query.log').write_text(cp.stdout+cp.stderr);assert cp.returncode==0
    (Q/'query-receipt.json').write_text(json.dumps({'argv':args,'binary_sha256':sha(binary),'points_sha256':sha(Q/'points.csv'),'pairs_sha256':sha(Q/'pairs.csv'),'nodes_sha256':sha(Q/'nodes.csv'),'query_rows':len(X),'identity_domain':'mechanics_node_row','physical_steps':0},indent=2)+'\n')
    nodes=list(csv.DictReader((Q/'nodes.csv').open()));query_pairs=list(csv.DictReader((Q/'pairs.csv').open()))
    assert len(nodes)==len(X)
    point_rows=list(csv.DictReader((Q/'points.csv').open()))
    assert len(point_rows)==len(X)
    for i,(r,nr) in enumerate(zip(point_rows,nodes)):
        assert int(r['query_row'])==i and int(nr['query_row'])==i
        assert int(nr['total_bone_banks'])==185
        assert sum(int(nr[k])for k in ['aabb_proven_outside_banks','parity_outside_banks','inside_banks','ambiguous_banks','invalid_banks'])==185
        assert int(nr['ambiguous_banks'])==0 and int(nr['invalid_banks'])==0
        for k,key in enumerate(('x_bits_u32','y_bits_u32','z_bits_u32')):
            assert int(r[key])==int(X.view(np.uint32)[i,k])
    assert all(r['classification']in('inside','outside')and int(r['ambiguous'])==0 for r in query_pairs)
    inside=[r for r in query_pairs if r['classification']=='inside']
    report.update(inside_node_bank_pairs=len(inside),inside_nodes=len({int(r['query_row'])for r in inside}),query_pair_count=len(query_pairs))
    A=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
    sys.path.insert(0,str(A.parent));import human_skin_fat_bone_audit as audit
    tri,ids,gids,bodies,rot,off,pos,keys,pins,exact=audit.read_bank_inputs(bank,E/'route-audit-counter-fix-030/canonical/matter')
    local={int(body):(X.astype(float)-pos[int(body)])@rot[int(body)].T-off[int(body)]for body in rot}
    actual=set()
    for body,bone in keys:
        body=int(body);bone=int(bone);bank_tri=tri[(ids[:,0]==body)&(ids[:,1]==bone)]
        active=np.all(local[body]>=bank_tri.min(axis=(0,1)),axis=1)&np.all(local[body]<=bank_tri.max(axis=(0,1)),axis=1)
        actual.update((int(r),body,bone)for r in np.flatnonzero(active))
    returned={(int(r['query_row']),int(r['bank_body']),int(r['bank_stable_id']))for r in query_pairs}
    assert actual==returned and len(returned)==len(query_pairs)
    er=audit.audit_unique_tet_edges(np.arange(len(X)),T,ids,tri,local,exact['classify_segment_triangle'],node_identity_domain='query_row')
    (P/'tet-edge-report.json').write_text(json.dumps(er,indent=2)+'\n')
    report.update(edge_bank_groups_with_hits=er['edge_bank_groups_with_hits'],edge_counts=er['counts'],wall_seconds=time.monotonic()-started,status='GEOMETRY_DIAGNOSTIC_NOT_ADMITTED')
    (P/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
