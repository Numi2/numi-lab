#!/usr/bin/env python3
"""Exact audit of opposite-material native/candidate tet overlaps for 085c."""
from pathlib import Path
from collections import defaultdict
import csv, hashlib, itertools, json, sys, time
import numpy as np

ROOT = Path('/Users/n/Work/numi-human-connected-surface-069/whole-reference-tetgen-084-bounded')
CAND = ROOT / 'connectivity-001/run-001/candidate-tetrahedra-compacted.npz'
EXPORT = Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/connected-skin-parent-mass-transfer-067/source-contribution-export-072/run-003/source-preflight')
NODES = EXPORT / 'compiled-skin-fem-nodes.csv'
TETS = EXPORT / 'compiled-skin-fem-tets.csv'
STACKED = ROOT / 'native-parent-transfer-085b/plane_filter_085b.py'
HALFSPACE = ROOT / 'native-parent-transfer-085b/plane_filter_085.py'
OUT = Path('/Users/n/Work/numi-human-connected-surface-069/whole-reference-tetgen-084-bounded/native-parent-transfer-085c-cross-audit-001')
EXPECTED = {
    CAND: '5a35470223ce12343cc217be7822d182802e99f1c423bb52b8409f153f9a49db',
    NODES: '26c607f9835241ef5d274169e4ac2aeae9c0f6ab7ad94a06b307ca25844862c9',
    TETS: 'a7dc2d97713e43ed2bfae2aa805bb5281fdf2361108cf208df4cbdea10084461',
    STACKED: 'ae3cbe1e9f79a580b3a0f22b5126a2d91f11293874f9c867f46a83e81d6f3f61',
    HALFSPACE: '0ba4109b943f7e1b4312ca339f8e5f2f931b81413daca4074fa2be42f3b9c305',
}
TOL = 2e-10
MAT = {1:15, 2:16}
COMB = np.asarray(list(itertools.combinations(range(8),3)), dtype=np.int8)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def csvrows(p):
    with Path(p).open(newline='') as f: return list(csv.DictReader(f))
def vol6(v): return abs(float(np.linalg.det(v[1:]-v[0])))/6.
def intersection_points(a,b,halfspaces):
    na,ba=halfspaces(a); nb,bb=halfspaces(b)
    if na is None or nb is None:return np.empty((0,3))
    n=np.vstack((na,nb)); h=np.r_[ba,bb]; mats=n[COMB]; rhs=h[COMB]; det=np.linalg.det(mats)
    keep=np.abs(det)>=1e-10
    try: xs=np.linalg.solve(mats[keep],rhs[keep,:,None])[:,:,0]
    except np.linalg.LinAlgError:
        vals=[]
        for m,r in zip(mats[keep],rhs[keep]):
            try: vals.append(np.linalg.solve(m,r))
            except np.linalg.LinAlgError: pass
        xs=np.asarray(vals,dtype=np.float64).reshape((-1,3))
    pts=[]
    for x in xs:
        if np.max(n@x-h)<=TOL and not any(np.max(np.abs(x-y))<TOL for y in pts):pts.append(x)
    return np.asarray(pts,dtype=np.float64).reshape((-1,3))
def nearest_triangle(p, tris):
    if len(tris)==0:return float('inf'),-1
    a=tris[:,0]; b=tris[:,1]; c=tris[:,2]
    ab=b-a; ac=c-a; ap=p-a
    d00=np.einsum('ij,ij->i',ab,ab); d01=np.einsum('ij,ij->i',ab,ac); d11=np.einsum('ij,ij->i',ac,ac)
    d20=np.einsum('ij,ij->i',ap,ab); d21=np.einsum('ij,ij->i',ap,ac)
    den=d00*d11-d01*d01
    valid=np.abs(den)>1e-30
    v=np.zeros(len(tris)); w=np.zeros(len(tris))
    v[valid]=(d11[valid]*d20[valid]-d01[valid]*d21[valid])/den[valid]
    w[valid]=(d00[valid]*d21[valid]-d01[valid]*d20[valid])/den[valid]
    proj=a+v[:,None]*ab+w[:,None]*ac
    dist=np.full(len(tris),np.inf)
    inside=valid&(v>=0)&(w>=0)&(v+w<=1)
    dist[inside]=np.linalg.norm(p-proj[inside],axis=1)
    for x,y in ((a,b),(b,c),(c,a)):
        e=y-x; ee=np.einsum('ij,ij->i',e,e); t=np.zeros(len(tris)); ok=ee>0
        t[ok]=np.clip(np.einsum('ij,ij->i',p-x[ok],e[ok])/ee[ok],0.,1.)
        q=x+t[:,None]*e
        dist=np.minimum(dist,np.linalg.norm(p-q,axis=1))
    k=int(np.argmin(dist)); return float(dist[k]),k

def face_interface(tets, mats):
    faces=defaultdict(list)
    for ci,t in enumerate(tets):
        for omit in range(4):
            f=tuple(sorted(np.delete(t,omit).tolist()))
            faces[f].append((ci,int(mats[ci])))
    out=[]; nonmanifold=0
    for f,owners in faces.items():
        if len({m for _,m in owners})>1: out.append(f)
        if len(owners)>2: nonmanifold+=1
    return np.asarray(out,dtype=np.int64).reshape((-1,3)),nonmanifold

def main():
    t0=time.monotonic()
    if (OUT/'run-001').exists(): raise RuntimeError('refusing to overwrite run-001')
    for p,h in EXPECTED.items():
        actual=sha(p)
        if actual!=h: raise RuntimeError(f'pin mismatch {p}: {actual}')
    sys.path.insert(0,str(STACKED.parent))
    from plane_filter_085 import halfspaces
    from plane_filter_085b import tetra_intersection_volume_stacked
    z=np.load(CAND,allow_pickle=False)
    points=np.asarray(z['reference_points_m'],dtype=np.float64)
    child_t=np.asarray(z['tetrahedra_zero_based'],dtype=np.int64)
    child_m=np.asarray(z['material_ids'],dtype=np.int64)
    cell_uid=np.asarray(z['cell_uids'],dtype=np.int64)
    source_i=np.asarray(z['source_I_faces'],dtype=np.int64)
    node_rows=csvrows(NODES); tet_rows=[r for r in csvrows(TETS) if int(r['identity_active_bit'])==1]
    gids=np.asarray([int(r['global_node']) for r in node_rows],np.int64)
    local=np.asarray([int(r['object_local_node']) for r in node_rows],np.int64)
    native_xyz=np.asarray([[float(r[f'rest_{a}']) for a in 'xyz'] for r in node_rows],np.float64)
    old_t=np.asarray([[int(r[f'source_node{j}']) for j in range(4)] for r in tet_rows],np.int64)
    old_m=np.asarray([int(r['material_index']) for r in tet_rows],np.int64)
    old_row=np.asarray([int(r['source_tet_row']) for r in tet_rows],np.int64)
    if not np.array_equal(local,np.arange(len(local))) or old_t.min()<0 or old_t.max()>=len(local):raise RuntimeError('native node-index identity failure')
    if len(child_t)!=32796 or len(old_t)!=28632 or not np.array_equal(z['source_global_node_ids'],gids):raise RuntimeError('candidate/native identity dimensions differ')
    child_xyz=points[child_t]; old_xyz=native_xyz[old_t]
    child_vol=np.asarray([vol6(v) for v in child_xyz]); old_vol=np.asarray([vol6(v) for v in old_xyz])
    child_native_m=np.asarray([MAT[int(m)] for m in child_m],np.int64)
    cand_min,cand_max=child_xyz.min(axis=1),child_xyz.max(axis=1)
    old_min,old_max=old_xyz.min(axis=1),old_xyz.max(axis=1)
    order=np.argsort(old_max[:,0],kind='mergesort'); sorted_max=old_max[order,0]
    cross_aabb=0; positives=[]
    for ci in range(len(child_t)):
        first=int(np.searchsorted(sorted_max,cand_min[ci,0]-TOL,side='left'))
        ids=order[first:]
        if len(ids):
            keep=(old_min[ids,0]<=cand_max[ci,0]+TOL)
            keep &= (old_m[ids] != child_native_m[ci])
            keep &= np.all(old_max[ids,1:]>=cand_min[ci,None,1:]-TOL,axis=1)
            keep &= np.all(old_min[ids,1:]<=cand_max[ci,None,1:]+TOL,axis=1)
            ids=ids[keep]
        for oi in ids:
            cross_aabb+=1
            v=tetra_intersection_volume_stacked(child_xyz[ci],old_xyz[int(oi)],TOL,TOL)
            cutoff=max(min(child_vol[ci],old_vol[int(oi)])*1e-13,1e-25)
            if v>cutoff:
                pts=intersection_points(child_xyz[ci],old_xyz[int(oi)],halfspaces)
                positives.append((int(ci),int(oi),float(v),float(cutoff),pts))
        if (ci+1)%5000==0: print(json.dumps({'children':ci+1,'cross_material_aabb_pairs':cross_aabb,'positive_pairs':len(positives),'elapsed_s':time.monotonic()-t0}),flush=True)

    cand_if, cand_nonmanifold=face_interface(child_t,child_m)
    native_if, native_nonmanifold=face_interface(old_t,old_m)
    cand_if_xyz=points[cand_if] if len(cand_if) else np.empty((0,3,3))
    native_if_xyz=native_xyz[native_if] if len(native_if) else np.empty((0,3,3))
    source_i_xyz=points[source_i]
    interface_set={tuple(sorted(f.tolist())) for f in cand_if}
    witnesses=[]; detail_arrays=[]
    for ci,oi,v,cutoff,pts in positives:
        center=pts.mean(axis=0) if len(pts) else .5*(child_xyz[ci].mean(axis=0)+old_xyz[oi].mean(axis=0))
        cd,cr=nearest_triangle(center,cand_if_xyz)
        nd,nr=nearest_triangle(center,native_if_xyz)
        sd,sr=nearest_triangle(center,source_i_xyz)
        source_face=source_i[sr].tolist() if sr>=0 else None
        native_face=native_if[nr].tolist() if nr>=0 else None
        candidate_face=cand_if[cr].tolist() if cr>=0 else None
        witness={
            'candidate_child_row':ci,'candidate_cell_uid':int(cell_uid[ci]),'candidate_material_id':int(child_m[ci]),
            'candidate_native_material_id':int(child_native_m[ci]),'native_parent_row':oi,'native_source_tet_row':int(old_row[oi]),
            'native_material_id':int(old_m[oi]),'overlap_volume_m3':v,'positive_cutoff_m3':cutoff,
            'overlap_over_cutoff':v/cutoff,'candidate_child_volume_m3':float(child_vol[ci]),'native_parent_volume_m3':float(old_vol[oi]),
            'fraction_of_candidate_child':v/float(child_vol[ci]),'fraction_of_native_parent':v/float(old_vol[oi]),
            'intersection_points_m':pts.tolist(),'intersection_centroid_m':center.tolist(),
            'candidate_tet_reference_points_m':child_xyz[ci].tolist(),'native_tet_reference_points_m':old_xyz[oi].tolist(),
            'candidate_I_PLC_nearest_face_row':int(sr),'candidate_I_PLC_nearest_face_vertex_rows':source_face,
            'candidate_I_PLC_distance_from_overlap_centroid_m':sd,
            'candidate_material_interface_nearest_face_vertex_rows':candidate_face,'candidate_material_interface_distance_m':cd,
            'candidate_nearest_interface_face_is_source_I_PLC':bool(tuple(sorted(candidate_face)) in interface_set) if candidate_face else False,
            'native_material_interface_nearest_face_vertex_rows':native_face,'native_material_interface_distance_m':nd,
        }
        witnesses.append(witness)
        detail_arrays.append((ci,oi,v,cutoff,center))
    run=OUT/'run-001'; run.mkdir(parents=True,exist_ok=False)
    report={
        'status':'CROSS_MATERIAL_EXACT_WITNESS_AUDIT_COMPLETE',
        'admission':{'mass_transfer':False,'physical_geometry':False,'native_physics':False},
        'inputs_sha256':{str(p):sha(p) for p in EXPECTED},
        'clip':{'method':'pinned unculled 085b stacked exact tetra intersection; no plane rejection','inside_tolerance_m':TOL,'unique_tolerance_m':TOL,'positive_cutoff':'max(min(Vchild,Vnative)*1e-13,1e-25)'},
        'counts':{'candidate_children':len(child_t),'native_parent_tets':len(old_t),'cross_material_aabb_pairs_exactly_clipped':cross_aabb,'positive_cross_material_pairs':len(witnesses),'candidate_material_interface_faces':len(cand_if),'candidate_interface_nonmanifold_faces':cand_nonmanifold,'native_material_interface_faces':len(native_if),'native_interface_nonmanifold_faces':native_nonmanifold,'source_I_PLC_faces':len(source_i)},
        'witnesses':witnesses,
        'summary':{'total_cross_material_overlap_volume_m3':float(sum(x[2] for x in detail_arrays)),
                   'max_fraction_of_candidate_child':max((w['fraction_of_candidate_child'] for w in witnesses),default=0.),
                   'min_candidate_I_PLC_distance_m':min((w['candidate_I_PLC_distance_from_overlap_centroid_m'] for w in witnesses),default=None),
                   'min_candidate_interface_distance_m':min((w['candidate_material_interface_distance_m'] for w in witnesses),default=None),
                   'min_native_interface_distance_m':min((w['native_material_interface_distance_m'] for w in witnesses),default=None),
                   'elapsed_seconds':time.monotonic()-t0},
        'limitations':['This isolates only opposite-material AABB pairs; it does not compute complete child or native-parent coverage.','The candidate I-face and native tet material-interface distances are diagnostic nearest-triangle distances; they do not repartition overlap volume or establish a valid typed transfer.']}
    (run/'cross-material-witnesses.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(run/'cross-material-witnesses.npz',
        candidate_child_rows=np.asarray([x[0] for x in detail_arrays],np.int32),
        native_parent_rows=np.asarray([x[1] for x in detail_arrays],np.int32),
        overlap_volumes_m3=np.asarray([x[2] for x in detail_arrays],np.float64),
        positive_cutoffs_m3=np.asarray([x[3] for x in detail_arrays],np.float64),
        intersection_centroids_m=np.asarray([x[4] for x in detail_arrays],np.float64).reshape((-1,3)))
    print(json.dumps(report['summary'],indent=2),flush=True)

if __name__=='__main__': main()
