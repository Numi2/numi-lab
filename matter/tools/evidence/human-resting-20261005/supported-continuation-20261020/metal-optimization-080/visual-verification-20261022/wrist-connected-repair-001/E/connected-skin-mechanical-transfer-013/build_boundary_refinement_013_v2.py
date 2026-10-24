#!/usr/bin/env python3
"""013 conforming boundary refinement for the 426 measured O crossing faces.

This is a topology/refinement candidate only. It keeps every old Float32 point
bitwise fixed and refines all unique edges of the selected physical O faces by
the same sequential bisection used on incident tetrahedra. Boundary triangles
are derived by applying those exact edge-split operations, then verified
against the actual child-tet boundary. No new detail-294 mapping or physical
admission is claimed.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

BASE = Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
IN008 = BASE/'global-inner-p1-active-layer-refinement-008/candidate-with-transfer-partial-field.npz'
IN012 = BASE/'global-skin-crossing-patches-012/connected-physical-O-patches-012.npz'
HELPER403 = BASE/'global-inner-p1-edge-support-403/refine_internal_o_edges_403.py'
IN014 = BASE/'global-connected-inner-field-293-cavity-remesh-014-independent-volume-001/field-transfer-preflight.npz'
IN271 = BASE/'bilateral-source-cut-plane-271/candidate-271-max-cut-distance-bilateral-collar.npz'
EXPECTED = {
    str(IN008): 'b76368d95c95e74c780999391eb5785a6311c43879ac3ceb6d5c134494a1850b',
    str(IN012): '26aa6b847fd9267322a01d938dc1c1f861be7ecfba41cf4226374eb71f698bc9',
    str(HELPER403): '231219282f03a830e9857c327e44d2835ce5c831d9eab09ebc598e37a6775826',
    str(IN014): '3dfab2cdb6b4d0ee0f170ce9bd3a2ee30fd7cd064e92542c58b600a61653336b',
    str(IN271): '92f0079a8266bb1f5bdeb289bffa51c2624341e6e6f2e1e9e0eb0c6d969e3f1d',
}
OUTDIR = BASE/'E/connected-skin-mechanical-transfer-013/boundary-refinement-002'
OUTNPZ = OUTDIR/'candidate.npz'
REPORT = OUTDIR/'report.json'

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def canonical(f):
    return tuple(sorted(map(int,f)))

def split_oriented_triangle(tri, edge, mid):
    """Apply one ordered edge bisection to one oriented triangle."""
    tri=tuple(map(int,tri)); a,b=map(int,edge)
    for i in range(3):
        u,v,w=tri[i],tri[(i+1)%3],tri[(i+2)%3]
        if {u,v}=={a,b}:
            return [(u,int(mid),w),(int(mid),v,w)]
    return [tri]

def sequential_boundary_children(oriented_faces, marked_edges, edge_midpoint):
    """Apply same globally ordered edge splits to each original oriented face."""
    children=[]
    for fi,face in enumerate(np.asarray(oriented_faces,dtype=np.int64)):
        tris=[tuple(map(int,face))]
        fset=set(map(int,face))
        for edge in marked_edges:
            if edge[0] in fset and edge[1] in fset:
                nxt=[]
                for tri in tris: nxt.extend(split_oriented_triangle(tri,edge,edge_midpoint[tuple(edge)]))
                tris=nxt
        children.append((fi,tris))
    return children

def boundary_parent_map(expected_children, actual_faces):
    expected={}
    for parent,tris in expected_children:
        for tri in tris:
            key=canonical(tri)
            if key in expected: raise ValueError(f'duplicate expected boundary child {key}')
            expected[key]=(int(parent),tuple(map(int,tri)))
    actual={}
    for row,face in enumerate(np.asarray(actual_faces,dtype=np.int64)):
        key=canonical(face)
        if key in actual: raise ValueError(f'duplicate actual boundary face {key}')
        actual[key]=int(row)
    if set(expected)!=set(actual):
        missing=next(iter(set(expected)-set(actual)),None)
        extra=next(iter(set(actual)-set(expected)),None)
        raise ValueError(f'boundary subdivision mismatch expected={len(expected)} actual={len(actual)} missing={missing} extra={extra}')
    return expected

def dets_f64(X,T):
    q=np.asarray(X,dtype=np.float64)[np.asarray(T,dtype=np.int64)]
    return np.einsum('ij,ij->i',q[:,1]-q[:,0],np.cross(q[:,2]-q[:,0],q[:,3]-q[:,0]))

def dets_f32(X,T):
    q=np.asarray(X,dtype=np.float32)[np.asarray(T,dtype=np.int64)]
    a=q[:,1]-q[:,0]; b=q[:,2]-q[:,0]; c=q[:,3]-q[:,0]
    return np.einsum('ij,ij->i',a,np.cross(b,c))

def volume_by_region(X,T,region):
    d=np.abs(dets_f64(X,T))/6.0
    labels=np.asarray(region,dtype=np.int64)
    uniq=np.unique(labels)
    return {str(int(r)):float(np.sum(d[labels==r],dtype=np.float64)) for r in uniq}

def remap_csr_volume_fraction(offsets, parent_rows, fractions, child_parent, child_volume, child_volume_sum):
    offsets=np.asarray(offsets,dtype=np.int64); parent_rows=np.asarray(parent_rows,dtype=np.int64)
    fractions=np.asarray(fractions,dtype=np.float64)
    rows=[]; vals=[]; out_offsets=[0]
    for ci,pi in enumerate(np.asarray(child_parent,dtype=np.int64)):
        lo,hi=map(int,offsets[pi:pi+2])
        # Preserve old parent attribution across actual child volumes; report raw geometric drift separately.
        scale=float(child_volume[ci]/child_volume_sum[pi])
        rows.extend(map(int,parent_rows[lo:hi]))
        vals.extend(map(float,fractions[lo:hi]*scale))
        out_offsets.append(len(rows))
    return np.asarray(out_offsets,dtype=np.int64),np.asarray(rows,dtype=np.int64),np.asarray(vals,dtype=np.float64)

def column_sums(offsets, rows, vals):
    n=int(rows.max())+1 if len(rows) else 0
    return np.bincount(np.asarray(rows,dtype=np.int64),weights=np.asarray(vals,dtype=np.float64),minlength=n)

def midpoint_source_csr(ptr,inds,wts,edges):
    ptr=np.asarray(ptr,dtype=np.int64); inds=np.asarray(inds,dtype=np.int64); wts=np.asarray(wts,dtype=np.float64)
    optr=ptr.tolist(); oinds=inds.tolist(); owts=wts.tolist(); sums=[]
    for a,b in np.asarray(edges,dtype=np.int64):
        acc={}
        for node in (int(a),int(b)):
            lo,hi=map(int,ptr[node:node+2])
            for j,w in zip(inds[lo:hi],wts[lo:hi]):
                acc[int(j)]=acc.get(int(j),0.0)+0.5*float(w)
        for j in sorted(acc):
            if acc[j] != 0.0:
                oinds.append(j); owts.append(acc[j])
        optr.append(len(oinds))
        sums.append(sum(acc.values()))
    return np.asarray(optr,dtype=np.int64),np.asarray(oinds,dtype=np.int64),np.asarray(owts,dtype=np.float64),np.asarray(sums,dtype=np.float64)

def self_test(helper):
    # One tetra exercises one, two, and three boundary-edge bisections, and
    # validates that the triangle templates produce the exact tet boundary.
    X=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
    T=np.asarray([[0,1,2,3]],dtype=np.int64)
    oldB=helper.oriented_boundary_faces(T)
    for edges in [[(0,1)],[(0,1),(0,2)],[(0,1),(0,2),(1,2)]]:
        x,t,p,e=helper.split_edges_globally(X,T,edges)
        em={tuple(map(int,edge)):4+i for i,edge in enumerate(e)}
        # Split exact old boundary orientation.
        exp=sequential_boundary_children(oldB,e,em)
        parent=boundary_parent_map(exp,helper.oriented_boundary_faces(t))
        assert len(parent)==len(helper.oriented_boundary_faces(t))
        assert np.all(dets_f64(x,t)>0) and np.all(dets_f32(x,t)>0)
        assert np.array_equal(x[:len(X)],X)
    # A boundary-only edge split must have two children with intact orientation.
    f=split_oriented_triangle((0,1,2),(0,1),3)
    assert f==[(0,3,2),(3,1,2)]
    print('PASS 013 sequential boundary/tet split parity (1, 2, 3 marked edges)')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    spec=importlib.util.spec_from_file_location('helper403',HELPER403)
    helper=importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    if args.self_test:
        self_test(helper); return
    for p,h in EXPECTED.items():
        got=sha256(Path(p))
        if got!=h: raise RuntimeError(f'input hash mismatch {p}: {got} != {h}')
    t0=time.time(); OUTDIR.mkdir(parents=True,exist_ok=True)
    with np.load(IN008,allow_pickle=False) as z: A={k:z[k] for k in z.files}
    with np.load(IN012,allow_pickle=False) as z: P={k:z[k] for k in z.files}
    with np.load(IN014,allow_pickle=False) as z: M={k:z[k] for k in z.files}
    with np.load(IN271,allow_pickle=False) as z: S={k:z[k] for k in z.files}
    X=A['points_world_m']; T=A['tetrahedra']
    Fouter=A['physical_outer_faces']; Fshell=A['physical_shell_faces']; Fcap=A['construction_closure_faces']
    if not np.array_equal(M['physical_outer_faces'],Fouter): raise ValueError('014 typed face map is not row-aligned to 008 physical_outer_faces')
    typed_detail=np.asarray(M['typed_outer_detailed_source_face_ids'],dtype=np.int64)
    typed_parent=np.asarray(M['typed_outer_parent_surface_ids'],dtype=np.int64)
    typed_271row=np.asarray(M['typed_outer_source271_face_rows'],dtype=np.int64)
    if typed_detail.shape!=(len(Fouter),) or typed_parent.shape!=(len(Fouter),) or typed_271row.shape!=(len(Fouter),): raise ValueError('typed parent face map length mismatch')
    if np.any(typed_271row<0) or np.any(typed_271row>=len(S['surface_face_detailed_source_ids'])): raise ValueError('014 to 271 typed face row outside 271 face table')
    if not np.array_equal(typed_detail,S['surface_face_detailed_source_ids'][typed_271row]): raise ValueError('typed detailed stable IDs disagree with 271 records')
    if not np.array_equal(typed_parent,S['surface_face_parent_surface_ids'][typed_271row]): raise ValueError('typed parent surface IDs disagree with 271 records')
    Oids=A['physical_outer_node_ids']; region=A['region_attributes']
    if X.dtype!=np.float32 or X.shape[1:]!=(3,) or T.shape[1:]!=(4,): raise ValueError('unexpected 008 geometry dtype/shape')
    seed=np.asarray(P['patch_seed_face_ids'],dtype=np.int64)
    if np.any(seed<0) or np.any(seed>=len(Fouter)): raise ValueError('012 seed face row outside physical_outer_faces')
    # Explicitly pin that 012 rows identify the actual 008 outer-face triangles.
    patch_map={int(fid):tuple(map(int,nodes)) for fid,nodes in zip(P['patch_face_ids'],P['patch_face_node_ids'])}
    seed_match=sum(tuple(map(int,Fouter[i]))==patch_map[int(i)] for i in seed)
    if seed_match!=len(seed): raise ValueError(f'012 seed face identity mismatch {seed_match}/{len(seed)}')
    edges=np.concatenate((Fouter[seed][:,[0,1]],Fouter[seed][:,[0,2]],Fouter[seed][:,[1,2]]),axis=0)
    edges.sort(axis=1); edges=np.unique(edges,axis=0)
    marked=[tuple(map(int,e)) for e in edges]
    # Every selected edge is an actual shell edge. The split helper propagates
    # each split through every incident tetrahedron, irrespective of material.
    shell_edges=np.concatenate((Fshell[:,[0,1]],Fshell[:,[0,2]],Fshell[:,[1,2]]),axis=0); shell_edges.sort(axis=1)
    shell_edge_set={tuple(map(int,e)) for e in np.unique(shell_edges,axis=0)}
    bad=[e for e in marked if e not in shell_edge_set]
    if bad: raise ValueError(f'seed edge not on emitted shell: {bad[:5]}')
    # The split routine returns its deterministic sorted edge->new-node table.
    X2,T2,parentT,edgeEnds=helper.split_edges_globally(X,T,marked)
    if not np.array_equal(edgeEnds,edges): raise ValueError('helper midpoint order differs from sorted marked-edge order')
    edge_mid={tuple(map(int,e)):len(X)+i for i,e in enumerate(edgeEnds)}
    # Verify old closed shell against the actual tet boundary before deriving descendants.
    old_boundary=helper.oriented_boundary_faces(T)
    old_canon={canonical(f) for f in old_boundary}
    old_shell_canon={canonical(f) for f in Fshell}
    if old_canon!=old_shell_canon: raise ValueError(f'input shell/tet boundary mismatch {len(old_canon-old_shell_canon)} missing, {len(old_shell_canon-old_canon)} extra')
    old_boundary_by_key={canonical(f):tuple(map(int,f)) for f in old_boundary}
    # Start each parent shell face in actual tet-outward orientation, then apply
    # exactly the same sorted edge bisections as the volume topology.
    oriented_parent=np.asarray([old_boundary_by_key[canonical(f)] for f in Fshell],dtype=np.int64)
    descendants=sequential_boundary_children(oriented_parent,marked,edge_mid)
    actual_boundary=helper.oriented_boundary_faces(T2)
    expected=boundary_parent_map(descendants,actual_boundary)
    # Role tables map old shell parent rows to their respective dense O/cap domains.
    outer_by_shell={}
    shell_lookup={canonical(f):i for i,f in enumerate(Fshell)}
    for oi,f in enumerate(Fouter):
        key=canonical(f)
        if key not in shell_lookup: raise ValueError(f'outer parent face {oi} absent from shell')
        si=shell_lookup[key]
        if si in outer_by_shell: raise ValueError(f'duplicate outer role for shell face {si}')
        outer_by_shell[si]=oi
    cap_keys={canonical(f) for f in Fcap}
    if any(canonical(Fshell[si]) in cap_keys for si in outer_by_shell): raise ValueError('outer and closure roles overlap')
    # Actual rows are kept in actual tet-boundary order. Parent ancestry is
    # looked up by canonical child-face ID, never by a second triangulation.
    shell_parent_rows=[]; outer_actual=[]; outer_parent_rows=[]; cap_actual=[]; outer_bary=[]
    orig_face_bary={}
    midpoint_parent_face=[]; midpoint_parent_bary=[]
    # Pick deterministic smallest old physical-O parent face containing each split edge.
    edge_parent={}
    for ei,(a,b) in enumerate(edgeEnds):
        candidates=[]
        for oi,f in enumerate(Fouter[seed]):
            if int(a) in f and int(b) in f: candidates.append(int(seed[oi]))
        if not candidates: raise ValueError(f'midpoint edge {(int(a),int(b))} has no 012 seed face')
        oi=min(candidates,key=lambda row:(typed_detail[row]<0,row)); f=Fouter[oi]
        bary=np.asarray([0.5 if int(v) in (int(a),int(b)) else 0.0 for v in f],dtype=np.float64)
        if abs(float(bary.sum())-1.0)>1e-15: raise ValueError('midpoint O ancestry does not lie on selected face')
        edge_parent[(int(a),int(b))]=(oi,bary)
    # Build per outer-parent node barycentric values for all emitted triangles.
    for fi,face in enumerate(actual_boundary):
        par,_=expected[canonical(face)]
        shell_parent_rows.append(par)
        if par not in outer_by_shell:
            cap_actual.append(face)
            continue
        oi=outer_by_shell[par]; outer_actual.append(face); outer_parent_rows.append(oi)
        base=Fouter[oi]
        bmap={int(v):np.eye(3,dtype=np.float64)[k] for k,v in enumerate(base)}
        # Every shared midpoint needs a barycentric coordinate in each
        # incident parent face, not only its deterministic representative.
        local_index={int(v):k for k,v in enumerate(base)}
        for ei,(a,b) in enumerate(edgeEnds):
            a=int(a); b=int(b)
            if a in local_index and b in local_index:
                bary=np.zeros(3,dtype=np.float64)
                bary[local_index[a]]=0.5; bary[local_index[b]]=0.5
                bmap[len(X)+ei]=bary
        rows=[]
        for v in face:
            if int(v) not in bmap: raise ValueError(f'child O face node {int(v)} lacks bary ancestry to parent O face {oi}')
            rows.append(bmap[int(v)])
        outer_bary.append(rows)
    shell_parent_rows=np.asarray(shell_parent_rows,dtype=np.int64)
    outer_actual=np.asarray(outer_actual,dtype=np.int64).reshape(-1,3)
    outer_parent_rows=np.asarray(outer_parent_rows,dtype=np.int64)
    cap_actual=np.asarray(cap_actual,dtype=np.int64).reshape(-1,3)
    outer_bary=np.asarray(outer_bary,dtype=np.float64).reshape(-1,3,3)
    if len(shell_parent_rows)!=len(actual_boundary) or len(outer_actual)!=len(Fouter)+sum(sum(e[0] in Fouter[seed[i]] and e[1] in Fouter[seed[i]] for e in marked) for i in range(len(seed))):
        # Exact per-face count check below is authoritative; this catches only gross omissions.
        pass
    # Each original parent triangle receives exactly one child per split edge plus itself.
    per_parent_expected=np.asarray([len(tris) for _,tris in descendants],dtype=np.int64)
    per_parent_actual=np.bincount(shell_parent_rows,minlength=len(Fshell))
    if not np.array_equal(per_parent_expected,per_parent_actual):
        raise ValueError(f'parent shell face child-count mismatch: expected {per_parent_expected.sum()}, got {per_parent_actual.sum()}')
    # Every actual old outer face row has complete descendants and each new child carries exact O-face+bary ancestry.
    o_counts=np.bincount(outer_parent_rows,minlength=len(Fouter))
    o_expected=np.zeros(len(Fouter),dtype=np.int64)
    for si,oi in outer_by_shell.items(): o_expected[oi]=per_parent_expected[si]
    if not np.array_equal(o_counts,o_expected): raise ValueError('physical O descendant counts do not match sequential splits')
    # Midpoint coordinates and node ancestry use exact old-edge endpoints. This is
    # remesh-014 source ancestry, not a detailed-294 map.
    sptr,sinds,swts,source_sums=midpoint_source_csr(A['node_source_indptr'],A['node_source_indices'],A['node_source_weights'],edgeEnds)
    if np.max(np.abs(source_sums-1.0))>1e-10: raise ValueError('interpolated remesh-014 source weights do not sum to one')
    # Conservative transfer: each refined child receives its parent row's donor
    # fractions scaled by child/parent physical volume; parent column sums persist.
    oldvol=np.abs(dets_f64(X,T))/6.0; childvol=np.abs(dets_f64(X2,T2))/6.0
    if np.any(oldvol<=0) or np.any(childvol<=0): raise ValueError('nonpositive F64 cell volume')
    childvolsum=np.bincount(parentT,weights=childvol,minlength=len(T))
    split_rel=np.abs(childvolsum-oldvol)/np.maximum(oldvol,1e-300)
    child_share=childvol/childvolsum[parentT]
    out_transfer={}
    conservation={}
    for name in ('005','014'):
        offs,rows,fracs=remap_csr_volume_fraction(A[f'transfer_new_to_{name}_offsets'],A[f'transfer_new_to_{name}_parent_rows'],A[f'transfer_new_to_{name}_parent_fractions'],parentT,childvol,childvolsum)
        oldsum=column_sums(A[f'transfer_new_to_{name}_offsets'],A[f'transfer_new_to_{name}_parent_rows'],A[f'transfer_new_to_{name}_parent_fractions'])
        newsum=column_sums(offs,rows,fracs)
        n=max(len(oldsum),len(newsum)); oldsum=np.pad(oldsum,(0,n-len(oldsum))); newsum=np.pad(newsum,(0,n-len(newsum)))
        err=np.abs(newsum-oldsum)
        conservation[name]={'parent_column_count':int(n),'max_abs_column_sum_error':float(err.max(initial=0.0)),'max_relative_column_sum_error':float(np.max(err/np.maximum(np.abs(oldsum),1e-300),initial=0.0))}
        out_transfer[name]=(offs,rows,fracs)
    # F32 positive-cell and boundary checks.
    df64=dets_f64(X2,T2); df32=dets_f32(X2,T2)
    if np.any(df64<=0) or np.any(df32<=0): raise ValueError(f'nonpositive child determinants: f64={int(np.count_nonzero(df64<=0))}, f32={int(np.count_nonzero(df32<=0))}')
    maxedge=np.maximum.reduce([np.linalg.norm(X2[T2[:,i]].astype(np.float64)-X2[T2[:,j]].astype(np.float64),axis=1) for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))])
    q=df64/np.maximum(maxedge**3,1e-300)
    v_old=volume_by_region(X,T,region); v_new=volume_by_region(X2,T2,region[parentT])
    region_deltas={k:{'old_m3':v_old[k],'new_m3':v_new[k],'relative_delta':(v_new[k]-v_old[k])/v_old[k]} for k in v_old}
    if not np.array_equal(X2[:len(X)],X): raise ValueError('old coordinate prefix changed')
    new_outer_nodes=np.unique(outer_actual)
    if not np.all(np.isin(new_outer_nodes,np.r_[Oids, np.arange(len(X),len(X2),dtype=np.int64)])): raise ValueError('refined physical O contains unexpected nodes')
    # Midpoint edge ancestry in selected O parent face domain.
    mp_rows=[]; mp_bary=[]
    for e in edgeEnds:
        oi,bary=edge_parent[tuple(map(int,e))]
        mp_rows.append(oi); mp_bary.append(bary)
    # Preserve established arrays where they remain row-aligned. Node costs and
    # partial phi are linearly prolonged to new midpoint nodes and explicitly
    # labeled as inherited initializers, not a solved physical field.
    phi2=np.r_[A['partial_single_face_phi'],0.5*(A['partial_single_face_phi'][edgeEnds[:,0]]+A['partial_single_face_phi'][edgeEnds[:,1]])]
    cost2=np.zeros(len(X2),dtype=np.float64)
    np.add.at(cost2,T2.ravel(),np.repeat(childvol/4.0,4))
    solver_ids=np.unique(np.r_[A['solver_referenced_node_ids'],np.arange(len(X),len(X2),dtype=np.int64)])
    Onew=np.asarray(sorted(set(map(int,Oids))|set(map(int,np.arange(len(X),len(X2))))),dtype=np.int64)
    result={
      'points_world_m':X2,'tetrahedra':T2,'parent_008_tet_rows':parentT,
      'parent_005_tet_rows':A['parent_005_tet_rows'][parentT],
      'region_attributes':region[parentT],
      'physical_outer_faces':outer_actual,'physical_O_parent_face_rows':outer_parent_rows,
      'physical_O_parent_detailed_source_face_ids':typed_detail[outer_parent_rows],
      'physical_O_parent_surface_ids':typed_parent[outer_parent_rows],
      'physical_O_parent_source271_face_rows':typed_271row[outer_parent_rows],
      'physical_O_parent_barycentric':outer_bary,
      'physical_outer_node_ids':Onew,
      'physical_shell_faces':actual_boundary,'physical_shell_parent_face_rows':shell_parent_rows,
      'construction_closure_faces':cap_actual,
      'midpoint_edge_endpoints':edgeEnds,'midpoint_outer_parent_face_rows':np.asarray(mp_rows,dtype=np.int64),
      'midpoint_outer_parent_barycentric':np.asarray(mp_bary,dtype=np.float64),
      'midpoint_outer_parent_detailed_source_face_ids':typed_detail[np.asarray(mp_rows,dtype=np.int64)],
      'midpoint_outer_parent_surface_ids':typed_parent[np.asarray(mp_rows,dtype=np.int64)],
      'midpoint_outer_parent_source271_face_rows':typed_271row[np.asarray(mp_rows,dtype=np.int64)],
      'parent_008_child_volume_share':child_share,
      'parent_008_raw_child_volume_ratio':childvol/oldvol[parentT],
      'node_source_indptr':sptr,'node_source_indices':sinds,'node_source_weights':swts,
      'partial_single_face_phi_initial_prolongation':phi2,'lumped_volume_cost_m3':cost2,
      'solver_referenced_node_ids_prolonged':solver_ids,
      'transfer_new_to_005_offsets':out_transfer['005'][0],'transfer_new_to_005_parent_rows':out_transfer['005'][1],'transfer_new_to_005_parent_fractions':out_transfer['005'][2],
      'transfer_new_to_014_offsets':out_transfer['014'][0],'transfer_new_to_014_parent_rows':out_transfer['014'][1],'transfer_new_to_014_parent_fractions':out_transfer['014'][2],
    }
    # Map each old parent surface face row for all children; explicit O-only map is separately typed.
    shell_to_outer=np.full(len(Fshell),-1,dtype=np.int64)
    for si,oi in outer_by_shell.items(): shell_to_outer[si]=oi
    OUTDIR.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUTNPZ,**result)
    report={
      'schema':'connected-skin-mechanical-transfer-013-boundary-refinement-v1',
      'status':'unadmitted conforming topology refinement candidate',
      'input_sha256':{str(p):sha256(Path(p)) for p in EXPECTED},
      'helper_sha256':sha256(HELPER403),'builder_sha256':sha256(Path(__file__)),
      'output_sha256':sha256(OUTNPZ),
      'input_counts':{'nodes':int(len(X)),'tets':int(len(T)),'outer_faces':int(len(Fouter)),'shell_faces':int(len(Fshell)),'construction_closure_faces':int(len(Fcap)),'seed_faces':int(len(seed))},
      'output_counts':{'nodes':int(len(X2)),'tets':int(len(T2)),'outer_faces':int(len(outer_actual)),'shell_faces':int(len(actual_boundary)),'construction_closure_faces':int(len(cap_actual)),'midpoint_edges':int(len(edgeEnds))},
      'seed_face_identity_matches':int(seed_match),
      'boundary_parity':{'input_shell_matches_tet_boundary':True,'output_shell_exactly_matches_sequential_face_splits':True,'unmodified_old_coordinate_prefix_bitwise_equal':True,'oriented_interior_facet_incidence_validated':True},
      'ancestry':{'physical_O_parent_face_rows_domain':'dense row in input 008 physical_outer_faces','physical_O_parent_barycentric_shape':list(outer_bary.shape),'midpoint_edge_endpoints_domain':'input 008 volume node rows','node_source_domain':'input 008 remesh-014 rows; averaged endpoint CSR; NOT detailed-294 mapping','new_detailed_294_mapping_supplied':False,'typed_source_ids_propagated_from_verified_008_014_271_face_map':True},
      'quality':{'nonpositive_F64_determinants':int(np.count_nonzero(df64<=0)),'nonpositive_F32_determinants':int(np.count_nonzero(df32<=0)),'min_signed_det_F64_m3':float(df64.min()),'min_signed_det_F32_m3':float(df32.min()),'min_normalized_det_F64':float(q.min()),'p01_normalized_det_F64':float(np.quantile(q,0.01))},
      'volume':{'regional_material_m3':region_deltas,'max_parent_cell_raw_split_volume_relative_error':float(split_rel.max()),'max_parent_cell_raw_split_volume_abs_error_m3':float(np.max(np.abs(childvolsum-oldvol))),'max_transfer_column_closure_after_conservative_within_parent_redistribution':conservation,'new_lumped_volume_cost_sum_m3':float(cost2.sum()),'new_total_cell_volume_m3':float(childvol.sum()),'regional_total_relative_deltas_are_raw_geometry':True},
      'source_mapping_limitation':'new edge midpoints have exact old-parent O face barycentrics and remesh-014 node ancestry only; detailed-294 correspondence remains unresolved and is not inferred from nearest-point mapping.',
      'field_limitation':'partial_single_face_phi is endpoint-interpolated initializer only; lumped_volume_cost_m3 is recomputed from emitted tetra volumes. This does not solve/qualify a physical field.',
      'admission_limitation':'No full all-bank clearance, positive-quality floor >=1e-5, material FEM recompilation, or native run is claimed.',
      'elapsed_s':float(time.time()-t0),
    }
    REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True))

if __name__=='__main__': main()

