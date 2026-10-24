#!/usr/bin/env python3
import csv, json, hashlib, itertools, math, os, time
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull, QhullError

ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
IN=ROOT/"wholebody-attachment-source-rebind-037-candidate020-identity004-002/generated"
OUT=ROOT/"native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003"
OUT.mkdir(parents=True,exist_ok=True)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_int_csv(path, usecols):
    return np.loadtxt(path, delimiter=",", skiprows=1, usecols=usecols, dtype=np.int64, ndmin=2)
t0=time.time()
Ntab=np.genfromtxt(IN/"nodes.csv",delimiter=",",names=True,dtype=None,encoding="utf-8")
O=np.column_stack([Ntab["ox"],Ntab["oy"],Ntab["oz"]]).astype(np.float32)
I=np.column_stack([Ntab["ix"],Ntab["iy"],Ntab["iz"]]).astype(np.float32)
D=np.column_stack([Ntab["dx"],Ntab["dy"],Ntab["dz"]]).astype(np.float32)
n=len(O)
X=np.concatenate([O,I,D]).astype(np.float32)
F=read_int_csv(IN/"faces.csv",(0,1,2,3,4,5))
T=read_int_csv(IN/"tetrahedra.csv",(0,1,2,3,4,5,6,7,8))
# T columns: row,a,b,c,d,material,parent_tet,parent_surface_face,stable_source_face
tets=T[:,1:5].copy(); mats=T[:,5].copy(); stable=T[:,8].copy(); parent=T[:,6].copy()
face_rows=F[:,0]; Fnodes=F[:,1:4]; fstable=F[:,5]
target_col=2262; target_stable=89542; target_face_row=4643; threshold=1e-5
patch_face_mask=np.any(Fnodes==target_col,axis=1)
patch_face_rows=face_rows[patch_face_mask]
patch_stable=fstable[patch_face_mask]
if not np.any((face_rows==target_face_row)&(fstable==target_stable)&patch_face_mask):
    raise RuntimeError("typed target face identity mismatch")
if len(np.unique(patch_stable)) != len(patch_stable): raise RuntimeError("duplicate stable identity in local patch")
rows_by_stable=defaultdict(list)
for r,s in enumerate(stable.tolist()): rows_by_stable[int(s)].append(r)
patch_tet_rows=[]
face_tet_rows={}
for fr,sid in zip(patch_face_rows.tolist(),patch_stable.tolist()):
    rr=np.asarray(rows_by_stable[int(sid)],dtype=np.int64)
    if len(rr)!=6 or sorted(mats[rr].tolist()) != [1,1,1,2,2,2]:
        raise RuntimeError(f"expected 3 skin+3 fat tets at face {fr}, stable {sid}: {rr.tolist()}")
    face_tet_rows[int(sid)]=rr
    patch_tet_rows.extend(rr.tolist())
patch_tet_rows=np.array(sorted(set(patch_tet_rows)),dtype=np.int64)
if len(patch_tet_rows)!=6*len(patch_face_rows): raise RuntimeError("patch tet cardinality mismatch")

EDGES=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
FACETS=((1,2,3),(0,3,2),(0,1,3),(0,2,1))
def det6(tt, xyz=X):
    p=xyz[np.asarray(tt,dtype=np.int64)].astype(np.float64)
    return float(np.linalg.det(np.stack([p[1]-p[0],p[2]-p[0],p[3]-p[0]],axis=1)))
def orient(tt, xyz=X):
    tt=list(map(int,tt)); d=det6(tt,xyz)
    if d<0: tt[0],tt[1]=tt[1],tt[0]; d=-d
    if not d>0: raise RuntimeError("zero tet while generating")
    return tt
def qcell(tt,xyz=X):
    d=det6(tt,xyz)
    p=xyz[np.asarray(tt,dtype=np.int64)].astype(np.float64)
    le=max(float(np.linalg.norm(p[i]-p[j])) for i,j in EDGES)
    return d/(le**3) if le>0 else -np.inf
# Original registered surface face orientation is retained; choose a total node priority.
neighbors=sorted(set(int(x) for f in Fnodes[patch_face_mask] for x in f if int(x)!=target_col))
orders=[]
for insert in range(len(neighbors)+1):
    order=neighbors[:insert]+[target_col]+neighbors[insert:]
    rank={v:i for i,v in enumerate(order)}
    rank[target_col]=insert
    def tri_tets(fnodes, mat):
        a,b,c=sorted(map(int,fnodes),key=lambda v:(rank.get(v,v+len(neighbors)+1),v))
        if mat==1:
            raw=[(a,b,c,c+n),(b,a,b+n,c+n),(a,a+n,b+n,c+n)]
        elif mat==2:
            raw=[(a+n,b+n,c+n,c+2*n),(b+n,a+n,b+2*n,c+2*n),(a+n,a+2*n,b+2*n,c+2*n)]
        else: raise RuntimeError("unexpected material")
        return [orient(z) for z in raw]
    cand=tets.copy()
    for sid,rr in face_tet_rows.items():
        fridx=int(np.flatnonzero(fstable==sid)[0])
        fnodes=Fnodes[fridx]
        for mat in (1,2):
            rrm=rr[mats[rr]==mat]
            if len(rrm)!=3: raise RuntimeError("bad material patch")
            cand[rrm]=np.asarray(tri_tets(fnodes,mat),dtype=np.int64)
    qs=np.array([qcell(cand[r]) for r in patch_tet_rows])
    orders.append((float(qs.min()),int(insert),order,cand,qs))
# Use the best globally conforming total ordering of the target column.
best=max(orders,key=lambda z:(z[0],-abs(z[1]-neighbors.index(2245) if 2245 in neighbors else z[1])))
new_tets=best[3]
changed=np.flatnonzero(np.any(new_tets!=tets,axis=1))
if not set(changed.tolist()).issubset(set(patch_tet_rows.tolist())):
    raise RuntimeError("changed rows escaped the complete incident-star patch")
# Ensure every tetrahedron is positively oriented in F32-emitted coordinates.
vol64=np.empty(len(tets)); q64=np.empty(len(tets))
for r,tt in enumerate(new_tets):
    vol64[r]=det6(tt)/6.0; q64[r]=qcell(tt)
if np.min(vol64)<=0 or np.min(q64)<threshold:
    bad=np.flatnonzero((vol64<=0)|(q64<threshold))
    raise RuntimeError(f"quality gate failure rows={bad[:10].tolist()}, minq={q64.min():.9g}")
# Face incidence/material boundary accounting.
def oriented_face_key(face):
    vals=list(map(int,face)); s=sorted(vals)
    inv=sum(vals[i]>vals[j] for i in range(3) for j in range(i+1,3))
    return tuple(s), (-1 if inv%2 else 1)
def boundary_oriented(rows, ttarr):
    uses=defaultdict(list)
    for r in rows:
        a,b,c,d=map(int,ttarr[int(r)])
        for ff in ((b,c,d),(a,d,c),(a,b,d),(a,c,b)):
            k,sgn=oriented_face_key(ff); uses[k].append(sgn)
    out={}
    for k,ss in uses.items():
        if len(ss)==1: out[k]=ss[0]
        elif len(ss)==2 and sum(ss)==0: continue
        else: raise RuntimeError(f"nonmanifold/orientation cavity face {k} uses={ss}")
    return out
old_cav=boundary_oriented(patch_tet_rows,tets)
new_cav=boundary_oriented(patch_tet_rows,new_tets)
if old_cav!=new_cav:
    missing=set(old_cav)-set(new_cav); extra=set(new_cav)-set(old_cav)
    raise RuntimeError(f"oriented cavity boundary mismatch missing={list(missing)[:5]} extra={list(extra)[:5]}")
def material_faces(ttarr):
    uses=defaultdict(list)
    for r,tt in enumerate(ttarr):
        a,b,c,d=map(int,tt); m=int(mats[r])
        for ff in ((b,c,d),(a,d,c),(a,b,d),(a,c,b)):
            k=tuple(sorted(ff)); uses[k].append(m)
    return {k:tuple(sorted(v)) for k,v in uses.items() if len(v)==1 or len(set(v))>1}
old_mf=material_faces(tets); new_mf=material_faces(new_tets)
if old_mf!=new_mf:
    raise RuntimeError(f"exterior/material interface facet mismatch: {sum(old_mf.get(k)!=new_mf.get(k) for k in set(old_mf)|set(new_mf))}")
# Explicit O/I/D role tables stay byte-identical.
role_faces={role:(Fnodes+off) for role,off in (("O",0),("I",n),("D",2*n))}
# Geometry volume per material must remain unchanged.
oldvol=np.array([det6(tt)/6 for tt in tets]); newvol=vol64
material_volumes={}
for m in sorted(set(mats.tolist())):
    material_volumes[str(int(m))]={"old_m3":float(oldvol[mats==m].sum()),"new_m3":float(newvol[mats==m].sum()),"delta_m3":float(newvol[mats==m].sum()-oldvol[mats==m].sum())}
# Exact local geometric overlaps; cached X and candidate connectivity, only affected star.
def bary(ids,p):
    a=X[np.asarray(ids,dtype=np.int64)].astype(np.float64)
    A=np.stack([a[0]-a[3],a[1]-a[3],a[2]-a[3]],axis=1)
    u=np.linalg.solve(A,np.asarray(p,dtype=np.float64)-a[3])
    return np.array([u[0],u[1],u[2],1.0-u.sum()])
def inside(ids,p,tol=2e-10): return bool(np.min(bary(ids,p))>=-tol)
def intersect_volume(t1,t2):
    ids1=np.asarray(t1,dtype=np.int64); ids2=np.asarray(t2,dtype=np.int64)
    a=X[ids1].astype(np.float64); b=X[ids2].astype(np.float64)
    scale=max(np.linalg.norm(a[i]-a[j]) for i,j in EDGES)
    scale=max(scale,max(np.linalg.norm(b[i]-b[j]) for i,j in EDGES))
    if scale<=0:return 0.0
    origin=(a.mean(axis=0)+b.mean(axis=0))*0.5
    pts=[]
    for p in a:
        if inside(ids2,p):pts.append((p-origin)/scale)
    for p in b:
        if inside(ids1,p):pts.append((p-origin)/scale)
    def edge_plane(srcids,otherids):
        src=X[srcids].astype(np.float64); oth=X[otherids].astype(np.float64)
        for i,j in EDGES:
            p0,p1=src[i],src[j]
            for ff in FACETS:
                tri=oth[np.asarray(ff)]
                nrm=np.cross(tri[1]-tri[0],tri[2]-tri[0])
                d0=float(np.dot(nrm,p0-tri[0])); d1=float(np.dot(nrm,p1-tri[0]))
                if d0*d1>=0:continue
                u=d0/(d0-d1)
                if u<=1e-12 or u>=1-1e-12:continue
                p=p0+u*(p1-p0)
                if inside(otherids,p):pts.append((p-origin)/scale)
    edge_plane(ids1,ids2); edge_plane(ids2,ids1)
    if len(pts)<4:return 0.0
    keep=[]
    for p in pts:
        if not any(np.dot(p-q,p-q)<=1e-18 for q in keep):keep.append(p)
    if len(keep)<4:return 0.0
    try:return float(ConvexHull(np.asarray(keep),qhull_options="Qt Qx").volume*(scale**3))
    except QhullError:return 0.0
oldrows=patch_tet_rows
newrows=patch_tet_rows
old_by_mat={m:[r for r in oldrows if mats[r]==m] for m in (1,2)}
new_by_mat={m:[r for r in newrows if mats[r]==m] for m in (1,2)}
records=[]
raw_child_cover=[]; raw_parent_cover=[]
for m in (1,2):
    for nr in new_by_mat[m]:
        nv=abs(det6(new_tets[nr]))/6.0
        row_sum=0.0
        for orow in old_by_mat[m]:
            ov=intersect_volume(new_tets[nr],tets[orow])
            if ov>max(1e-24,abs(det6(tets[orow]))/6*1e-14):
                records.append((int(nr),int(orow),float(ov),float(ov/(abs(det6(tets[orow]))/6.0))))
                row_sum+=ov
        raw_child_cover.append(abs(row_sum-nv)/max(nv,1e-30))
    for orow in old_by_mat[m]:
        ov=abs(det6(tets[orow]))/6.0
        colsum=sum(rec[2] for rec in records if rec[1]==int(orow))
        raw_parent_cover.append(abs(colsum-ov)/max(ov,1e-30))
max_child=max(raw_child_cover,default=0.0); max_parent=max(raw_parent_cover,default=0.0)
if max(max_child,max_parent)>5e-7:
    raise RuntimeError(f"local overlap coverage fails child={max_child:g} parent={max_parent:g}")
# Compose full conservative transfer CSR: identity for untouched; overlap fractions (by old-cell volume) for patch children.
by_child=defaultdict(list)
for nr,orow,ov,frac in records: by_child[nr].append((orow,frac,ov))
offsets=[0]; parent_rows=[]; fractions=[]; overlap_vols=[]
changed_set=set(changed.tolist())
for nr in range(len(tets)):
    if nr not in changed_set:
        parent_rows.append(nr); fractions.append(1.0); overlap_vols.append(float(oldvol[nr]))
    else:
        for orow,frac,ov in sorted(by_child[nr]):
            parent_rows.append(orow); fractions.append(frac); overlap_vols.append(ov)
    offsets.append(len(parent_rows))
# Parent columns sum to 1; child rows need not.
cols=np.bincount(np.asarray(parent_rows,dtype=np.int64),weights=np.asarray(fractions),minlength=len(tets))
if np.max(np.abs(cols-1.0))>5e-7: raise RuntimeError(f"CSR old-parent column closure {np.max(np.abs(cols-1.0)):g}")
# Deterministic arbitrary old cell masses validate conservative assignment.
old_mass=np.arange(1,len(tets)+1,dtype=np.float64)**1.137
new_mass=np.zeros(len(tets),dtype=np.float64)
for ci in range(len(tets)):
    for k in range(offsets[ci],offsets[ci+1]):
        new_mass[ci]+=fractions[k]*old_mass[parent_rows[k]]
mass_rel=abs(new_mass.sum()-old_mass.sum())/old_mass.sum()
if mass_rel>5e-7:raise RuntimeError(f"synthetic mass transfer residual {mass_rel:g}")
# Save full mesh and ancestry.
candidate_path=OUT/"candidate.npz"
np.savez_compressed(candidate_path,
    points_world_f32=X, tetrahedra=new_tets.astype(np.int32), material_ids=mats.astype(np.int8),
    stable_source_face_ids=stable.astype(np.int64), legacy_parent_tet_rows=np.where(np.isin(np.arange(len(tets)),changed),-1,parent).astype(np.int64),
    changed_tet_rows=changed.astype(np.int64), patch_surface_face_rows=patch_face_rows.astype(np.int64),
    patch_stable_source_face_ids=patch_stable.astype(np.int64),
    parent_csr_offsets=np.asarray(offsets,dtype=np.int64), parent_csr_old_tet_rows=np.asarray(parent_rows,dtype=np.int64),
    parent_csr_old_volume_fractions=np.asarray(fractions,dtype=np.float64), parent_csr_overlap_volumes_m3=np.asarray(overlap_vols,dtype=np.float64),
    old_parent_tet_ids=parent.astype(np.int64), parent_surface_face_ids=T[:,7].astype(np.int64),
    O_world_m=O,I_world_m=I,D_world_m=D,
    O_faces=role_faces["O"].astype(np.int32),I_faces=role_faces["I"].astype(np.int32),D_faces=role_faces["D"].astype(np.int32),
    old_tetrahedra=tets.astype(np.int32), old_material_ids=mats.astype(np.int8))
changed_csv=OUT/"changed-connectivity.csv"
with open(changed_csv,"w",newline="") as f:
    w=csv.writer(f);w.writerow(["tet_row","material","stable_source_face_id","old_parent_tet_id","new_single_parent_tet_id","a","b","c","d"])
    for r in changed:
        w.writerow([int(r),int(mats[r]),int(stable[r]),int(parent[r]),-1,*map(int,new_tets[r])])
report={
 "schema":"numi.human.d37.conforming-prism-rank-retriangulation.v1",
 "status":"bounded local candidate; not native/physical admission",
 "inputs":{"nodes_csv_sha256":sha(IN/"nodes.csv"),"faces_csv_sha256":sha(IN/"faces.csv"),"tetrahedra_csv_sha256":sha(IN/"tetrahedra.csv")},
 "target":{"surface_face_row":target_face_row,"stable_source_face_id":target_stable,"column_node_id":target_col,"affected_surface_face_rows":patch_face_rows.tolist(),"affected_stable_source_face_ids":patch_stable.tolist(),"rank_insertion_index":best[1],"rank_order_over_local_neighbor_rows":best[2]},
 "counts":{"node_rows_per_layer":int(n),"surface_faces":int(len(F)),"tetrahedra":int(len(tets)),"changed_tetrahedron_rows":int(len(changed)),"affected_prism_faces":int(len(patch_face_rows)),"csr_entries":int(len(parent_rows))},
 "gates":{"all_float32_signed_volumes_positive":bool(np.all(newvol>0)),"min_normalized_det_over_longest_edge_cubed":float(q64.min()),"quality_floor":threshold,"all_tets_meet_quality_floor":bool(np.all(q64>=threshold)),
 "oriented_local_cavity_boundary_exact":True,"global_exterior_and_material_interface_facets_exact":True,"O_I_D_positions_exact":True,"O_I_D_face_tables_exact":True,
 "local_old_new_overlap_child_coverage_max_relative_error":float(max_child),"local_old_new_overlap_parent_coverage_max_relative_error":float(max_parent),
 "old_parent_column_fraction_closure_max_abs_error":float(np.max(np.abs(cols-1.0))),"synthetic_nonuniform_mass_relative_residual":float(mass_rel)},
 "material_volume_m3":material_volumes,
 "quality":{"old_global_min_q":float(min(qcell(tt) for tt in tets)),"new_global_min_q":float(q64.min()),"old_patch_min_q":float(min(qcell(tets[r]) for r in patch_tet_rows)),"new_patch_min_q":float(min(q64[r] for r in patch_tet_rows))},
 "changed_connectivity":[{"tet_row":int(r),"material":int(mats[r]),"stable_source_face_id":int(stable[r]),"legacy_parent_tet_id":int(parent[r]),"new_nodes":new_tets[r].tolist()} for r in changed],
 "limitations":["No bone-clearance retest was run; existing known D-row bone penetration remains unresolved.","No native header/build/runtime qualification is claimed.","For changed cells, legacy single-parent IDs are set to -1; use the conservative CSR mapping."],
 "elapsed_seconds":time.time()-t0
}
report_path=OUT/"report.json";report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
manifest={
 "schema":"numi.human.d37.conforming-prism-rank-retriangulation.freeze.v1",
 "files":{p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in (candidate_path,changed_csv,report_path,Path(__file__))},
 "builder_source":str(Path(__file__)),
 "input_dir":str(IN),
 "status":"candidate frozen by deterministic input/output hashing"
}
(OUT/"freeze-manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
print(json.dumps({"output":str(OUT),"candidate_sha256":sha(candidate_path),"report_sha256":sha(report_path),"manifest_sha256":sha(OUT/"freeze-manifest.json"),"elapsed_s":time.time()-t0,"changed_tets":len(changed),"faces":len(patch_face_rows),"minq":float(q64.min()),"patch_old_minq":float(min(qcell(tets[r]) for r in patch_tet_rows)),"patch_new_minq":float(min(q64[r] for r in patch_tet_rows)),"max_coverage":max(max_child,max_parent)},indent=2))

