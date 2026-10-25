#!/usr/bin/env python3
"""Bounded local thickness-preserving levelset gauge test (construction only)."""
from __future__ import annotations
import hashlib, importlib, json, sys, time, types
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull, QhullError

ROOT = Path("/Users/n/Work/numi-human-connected-surface-069")
HERE = Path(__file__).resolve().parent
OUT = HERE / "output-018"
OUT.mkdir(parents=True, exist_ok=False)
FACET_NPZ = ROOT / "joint-interface-facet-extraction-164/run-015-nonzero-sign-regularized/run-001-joint-unsnapped/joint-interface-facets.npz"
CHECKPOINT = ROOT / "connected-volume-extraction-157/true-o-foundation-159/run-029-closed-volume-field-repair/run-003-nonzero-regularization/output-002/regularized-foundation-checkpoint.npz"
FIELDS = ROOT / "connected-volume-extraction-157/true-o-foundation-159/run-029-closed-volume-field-repair/run-003-nonzero-regularization/output-002/regularized-registered-levelsets.npz"
PAIRS = ROOT / "joint-interface-facet-contraction-165/run-004-global-short-edge-batch/output-001/localization-003.json"
PRED = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py")
CUTTER = ROOT / "joint-interface-facet-extraction-164/canonical/matter/tools/human_skin_fat_layer_reconstruction.py"
QUALITY = ROOT / "publication-085/canonical/matter/tools/human_skin_fat_layer_reconstruction.py"

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def poly_volume(points4, fields3, cuts):
    """Convex volume after sequential affine halfspace clips of one tetra."""
    verts=[]
    for i in range(4):
        verts.append((np.asarray(points4[i],dtype=np.float64),np.asarray(fields3[i],dtype=np.float64)))
    faces=[[verts[i] for i in face] for face in ((0,1,2),(0,1,3),(0,2,3),(1,2,3))]
    for field_id, keep_sign in cuts:
        clipped=[]
        plane_points=[]
        for face in faces:
            if len(face)<3: continue
            out=[]
            for j in range(len(face)):
                p=face[j]; q=face[(j+1)%len(face)]
                vp=float(keep_sign*p[1][field_id]); vq=float(keep_sign*q[1][field_id])
                pin=vp>=0.0; qin=vq>=0.0
                if pin and qin:
                    out.append(q)
                elif pin and not qin:
                    t=vp/(vp-vq)
                    x=p[0]+t*(q[0]-p[0]); v=p[1]+t*(q[1]-p[1]); v[field_id]=0.0
                    r=(x,v); out.append(r); plane_points.append(r)
                elif not pin and qin:
                    t=vp/(vp-vq)
                    x=p[0]+t*(q[0]-p[0]); v=p[1]+t*(q[1]-p[1]); v[field_id]=0.0
                    r=(x,v); out.extend((r,q)); plane_points.append(r)
            # Remove adjacent duplicates by point bytes.
            cleaned=[]
            for x in out:
                if not cleaned or not np.array_equal(cleaned[-1][0],x[0]): cleaned.append(x)
            if len(cleaned)>1 and np.array_equal(cleaned[0][0],cleaned[-1][0]): cleaned.pop()
            if len(cleaned)>=3: clipped.append(cleaned)
        # Exact-zero vertices on faces clipped away are part of the section too.
        for face in faces:
            for v in face:
                if float(v[1][field_id])==0.0:
                    # Include only if on the new poly boundary; extra zero points
                    # remain coplanar and are harmless in the cap hull.
                    plane_points.append(v)
        unique=[]
        for v in plane_points:
            if not any(np.linalg.norm(v[0]-u[0]) <= 1e-14 for u in unique): unique.append(v)
        if len(unique)>=3:
            xyz=np.asarray([u[0] for u in unique]); center=xyz.mean(axis=0)
            # Estimate a normal from the most separated pair and a third point.
            normal=None
            for a in range(1,len(xyz)):
                for b in range(a+1,len(xyz)):
                    cr=np.cross(xyz[a]-xyz[0],xyz[b]-xyz[0])
                    if np.linalg.norm(cr)>1e-24:
                        normal=cr/np.linalg.norm(cr); break
                if normal is not None: break
            if normal is not None:
                uvec=xyz[0]-center; uvec/=np.linalg.norm(uvec)
                vvec=np.cross(normal,uvec)
                angles=np.arctan2((xyz-center)@vvec,(xyz-center)@uvec)
                order=np.argsort(angles)
                clipped.append([unique[int(i)] for i in order])
        faces=clipped
        if not faces: return 0.0
    allp=[]
    for face in faces:
        allp.extend([v[0] for v in face])
    unique_xyz=[]
    for p in allp:
        if not any(np.linalg.norm(p-u)<=1e-14 for u in unique_xyz): unique_xyz.append(p)
    if len(unique_xyz)<4: return 0.0
    xyz=np.asarray(unique_xyz,dtype=np.float64)
    try:
        return float(ConvexHull(xyz).volume)
    except QhullError:
        return 0.0

def region_volumes(P,T,O,I,D,rows):
    skin=0.0; fat=0.0
    for ti in rows:
        tet=T[int(ti)]; pp=P[tet].astype(np.float64)
        vv=np.column_stack((O[tet],I[tet],D[tet])).astype(np.float64)
        # Local edited tets are certified O-nonnegative, so skin is I>=0.
        skin += poly_volume(pp,vv,((1,+1),))
        fat += poly_volume(pp,vv,((1,-1),(2,+1)))
    return {"skin_m3":skin,"fat_m3":fat}

def qstats(points, facets):
    tri=np.asarray(points,dtype=np.float32)[np.asarray(facets,dtype=np.int64)].astype(np.float64)
    a=tri[:,1]-tri[:,0]; b=tri[:,2]-tri[:,0]; c=tri[:,2]-tri[:,1]
    den=np.maximum.reduce((np.einsum("ij,ij->i",a,a),np.einsum("ij,ij->i",b,b),np.einsum("ij,ij->i",c,c)))
    return np.linalg.norm(np.cross(a,b),axis=1)/np.maximum(den,np.finfo(float).tiny)

def audit_local(P,F):
    package="numilab_human"
    pkg=types.ModuleType(package)
    pkg.__path__=[str(PRED.parent),str(PRED.parents[3]/"incremental-exact-audit-owner-253/src/numilab_human")]
    sys.modules[package]=pkg
    mod=importlib.import_module("numilab_human.cardiac_cavity_intersections")
    rec,_=mod.float32_surface_records(np.asarray(P,dtype=np.float32),np.asarray(F,dtype=np.int64),source_seam_groups=())
    return mod._audit_self_pairs_touching_faces(rec,set(range(len(F))))

def do_extract(cutter, P, T, I, D, ids):
    return cutter.extract_joint_zero_level_facets(P,T[ids],I,D,global_vertex_ids=np.arange(len(P),dtype=np.int64),parent_tet_ids=ids)

t0=time.monotonic()
for p in (FACET_NPZ,CHECKPOINT,FIELDS,PAIRS,PRED,CUTTER,QUALITY):
    if not p.is_file(): raise FileNotFoundError(p)
fac=np.load(FACET_NPZ,allow_pickle=False); cp=np.load(CHECKPOINT,allow_pickle=False); field=np.load(FIELDS,allow_pickle=False)
P=cp["points_world_m"]; T=cp["tetrahedra_zero_based"].astype(np.int64)
if P.dtype!=np.float32: raise ValueError("foundation must remain exact emitted F32")
if not np.array_equal(P,fac["points_world_m_f32"][:len(P)]): raise ValueError("facet/foundation F32 prefix mismatch")
O=field["outer_levelset_m"].astype(np.float64); I=field["skin_levelset_m"].astype(np.float64); D=field["total_levelset_m"].astype(np.float64)
if not (O.shape==I.shape==D.shape==(len(P),)): raise ValueError("field/source domain mismatch")
if not np.isfinite(np.column_stack((O,I,D))).all(): raise ValueError("nonfinite source field")
if np.any(D-I<=0.0): raise ValueError("baseline fat field not positive")
base_i=np.load(QUALITY,allow_pickle=False) if False else None
sys.path.insert(0,str(QUALITY.parent))
from human_skin_fat_layer_reconstruction import facet_incident_tetrahedron_quality_upper_bound
qfull=facet_incident_tetrahedron_quality_upper_bound(fac["points_world_m_f32"],fac["facets_zero_based"])
qbad_facets=np.flatnonzero(qfull<1.0e-5)
q_local_tets=np.unique(fac["facet_parent_local_input_tet_rows"][qbad_facets].astype(np.int64))
pairdoc=json.loads(PAIRS.read_text())
pair_tets=set()
for pair in pairdoc["pairs"]:
    for name in ("face_a","face_b"):
        pair_tets.add(int(pair[name]["input_local_tet_row"]))
seed_tets=np.unique(np.r_[q_local_tets,np.fromiter(pair_tets,dtype=np.int64)])
seed_nodes=np.unique(T[seed_tets])
target_rows=np.unique(np.r_[seed_nodes[np.abs(I[seed_nodes])<4.0e-6],
                            seed_nodes[np.abs(D[seed_nodes])<4.0e-6], 15278])
if target_rows.tolist()!=[10056,15094,15278,15880,18012,25615,31041]:
    raise ValueError(f"unexpected target witness rows {target_rows.tolist()}")
moddelta=[]
for row in target_rows:
    near=[]
    if int(row)==15278:
        near.append((1,float(I[row])))
    elif abs(I[row])<4e-6:
        near.append((1,float(I[row])))
    if abs(D[row])<4e-6:
        near.append((2,float(D[row])))
    if len(near)!=1: raise ValueError(f"row {row} has ambiguous target fields {near}")
    k,val=near[0]
    target_abs={10056:8.0e-6,15094:10.0e-6,15278:105.0e-6}.get(int(row),4.0e-6)
    target=float(np.copysign(target_abs,val if val!=0 else 1.0))
    delta=target-val
    hskin=float(O[row]+I[row])
    cap=float("nan")  # scalar gauge delta is not the geometric movement bound
    if O[row]<=0.0 or O[row]-delta<=0.0: raise ValueError(f"row {row} outer field is not safely positive")
    moddelta.append((int(row),int(k),val,target,delta,hskin,cap))
rows=np.asarray([x[0] for x in moddelta],dtype=np.int64)
deltas=np.asarray([x[4] for x in moddelta],dtype=np.float64)
O1=O.copy(); I1=I.copy(); D1=D.copy()
O1[rows]-=deltas; I1[rows]+=deltas; D1[rows]+=deltas
# Exact local proof: affected tets contain no negative O value and changed rows
# are strictly positive; every zero-valued O vertex remains untouched.
touch=np.flatnonzero(np.any(np.isin(T,rows),axis=1))
Ot=O[T[touch]]; O1t=O1[T[touch]]
if np.any(Ot<0.0) or not np.array_equal(np.where(Ot<0,-1,np.where(Ot>0,1,0)),
                                        np.where(O1t<0,-1,np.where(O1t>0,1,0))):
    raise ValueError("outer zero-set proof failed on touched cells")
if np.any(O[rows]==0.0) or not np.array_equal(np.flatnonzero(O==0.0),np.flatnonzero(O1==0.0)):
    raise ValueError("outer zero-valued source rows changed")
if not np.array_equal(np.sign(I),np.sign(I1)) or not np.array_equal(np.sign(D),np.sign(D1)):
    raise ValueError("field sign pattern changed")
hskin0=O+I; hskin1=O1+I1; hfat0=D-I; hfat1=D1-I1
hs_err=float(np.max(np.abs(hskin1-hskin0),initial=0)); hf_err=float(np.max(np.abs(hfat1-hfat0),initial=0))
if hs_err>2e-18 or hf_err>2e-18: raise ValueError(f"thickness arithmetic residual too large: {hs_err},{hf_err}")
# Every source edge incident to an edited node: exact before/after root motion.
edges=np.concatenate((T[:,[0,1]],T[:,[0,2]],T[:,[0,3]],T[:,[1,2]],T[:,[1,3]],T[:,[2,3]]),axis=0)
edges=np.sort(edges,axis=1)
edges=edges[np.isin(edges[:,0],rows)|np.isin(edges[:,1],rows)]
edges=np.unique(edges,axis=0)
cut_moves={}; cut_edge_count={}
for iface,(f0,f1) in enumerate(((I,I1),(D,D1)),start=1):
 move=[]
 for a,b in edges:
    va,vb=float(f0[a]),float(f0[b]); wa,wb=float(f1[a]),float(f1[b])
    if va*vb<0.0:
        if not wa*wb<0.0: raise ValueError(f"interface {iface} changed crossing status on edge {(int(a),int(b))}")
        t=abs(va)/(abs(va)+abs(vb)); u=abs(wa)/(abs(wa)+abs(wb))
        x=P[a].astype(np.float64)+t*(P[b].astype(np.float64)-P[a].astype(np.float64))
        y=P[a].astype(np.float64)+u*(P[b].astype(np.float64)-P[a].astype(np.float64))
        move.append((float(np.linalg.norm(y-x)),int(a),int(b),float(t),float(u)))
 cut_moves[str(iface)]={"crossing_edges":len(move),"max_displacement_m":max((x[0] for x in move),default=0.0),
   "max_witness":max(move,default=(0.0,-1,-1,float("nan"),float("nan"))),"all_edges":move}
 cut_edge_count[str(iface)]=len(move)
max_move=max(v["max_displacement_m"] for v in cut_moves.values())
# Bound actual cutpoint motion by local skin thickness at the moved cut.
move_fracs={}
for iface, moves in cut_moves.items():
    vals=[]
    for displacement,a,b,t,u in moves["all_edges"]:
        h=float((1.0-u)*hskin0[a]+u*hskin0[b])
        if not np.isfinite(h) or h<=0.0: raise ValueError(f"nonpositive local hskin at interface {iface}, edge {(a,b)}")
        vals.append(displacement/h)
    move_fracs[iface]=max(vals,default=0.0)
max_move_frac=max(move_fracs.values(),default=0.0)
if max_move_frac>0.10: raise ValueError(f"actual interface cutpoint motion exceeds 10% construction ceiling: {max_move_frac}")
# Local ring: all tets touching a vertex of a tet incident to an edited node.
ring_nodes=np.unique(T[touch])
local_ids=np.flatnonzero(np.any(np.isin(T,ring_nodes),axis=1)).astype(np.int64)
# Exact local material-volume change: only touched tets can change; O is
# nonnegative throughout them so skin is I>=0 and fat is I<=0,D>=0.
vol_before=region_volumes(P,T,O,I,D,touch)
vol_after=region_volumes(P,T,O1,I1,D1,touch)
# Extract pre/post local surfaces from the same source cell ring.
spec=importlib.util.spec_from_file_location("joint_facet_cutter_164",CUTTER)
cutter=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=cutter
spec.loader.exec_module(cutter)
before=do_extract(cutter,P,T,I,D,local_ids)
after=do_extract(cutter,P,T,I1,D1,local_ids)
q0=qstats(before.points_float32,before.facets)
q1=qstats(after.points_float32,after.facets)
a0=audit_local(before.points_float32,before.facets)
a1=audit_local(after.points_float32,after.facets)
# Save editable candidate field arrays, never altering source assets.
cand=OUT/"common-field-gauge-candidate.npz"
np.savez_compressed(cand,outer_levelset_m=O1,skin_levelset_m=I1,total_levelset_m=D1,
                    modified_source_rows=rows,delta_m=deltas)
report={
 "status":"LOCAL_DIAGNOSTIC_COMPLETE",
 "construction_only":True,"native_or_physical_admission":False,
 "input_pins":{"facet_npz_sha256":sha(FACET_NPZ),"checkpoint_sha256":sha(CHECKPOINT),
   "field_npz_sha256":sha(FIELDS),"source_pair_localization_sha256":sha(PAIRS),
   "exact_predicate_sha256":sha(PRED),"cutter_sha256":sha(CUTTER),"quality_helper_sha256":sha(QUALITY)},
 "candidate":{"path":str(cand),"sha256":sha(cand),"edited_rows":rows.tolist()},
 "edit":{"rule":"phiO'=phiO-delta, phiI'=phiI+delta, phiD'=phiD+delta",
   "targets":[{"row":r,"field_id":k,"before_m":v,"target_m":tg,"delta_m":d,
     "hskin_m":hs,"scalar_delta_fraction_hskin_diagnostic_only":abs(d)/hs} for r,k,v,tg,d,hs,cap in moddelta],
   "max_abs_delta_m":float(np.max(np.abs(deltas),initial=0)),
   "nodewise_thickness":{"hskin_max_abs_residual_m":hs_err,"hfat_max_abs_residual_m":hf_err,
     "hskin_unchanged_within_2e-18_m":hs_err<=2e-18,"hfat_unchanged_within_2e-18_m":hf_err<=2e-18},
   "outer_zero_set":{"zero_source_rows_unchanged":True,"touched_tet_count":int(len(touch)),
     "all_touched_outer_values_nonnegative":bool(np.all(Ot>=0.0)),
     "touched_tet_sign_and_zero_pattern_unchanged":True,
     "proof":"Only strictly positive outer vertices were changed, every touched tet has no negative outer vertex, and all zero outer vertices are untouched; the PL outer zero set on those cells is the same convex hull of zero vertices."},
   "source_edge_cut_motion":cut_moves,
   "max_displacement_fraction_hskin":max_move_frac,"per_interface_max_displacement_fraction_hskin":move_fracs,
   "construction_ceiling_fraction_hskin":0.10,"exceeds_earlier_0p75pct_screen":max_move_frac>0.0075,
   "anatomical_or_physical_acceptance":False},
 "local_field_defined_material_volume":{"affected_tet_count":int(len(touch)),
   "skin_before_m3":vol_before["skin_m3"],"skin_after_m3":vol_after["skin_m3"],
   "skin_delta_m3":vol_after["skin_m3"]-vol_before["skin_m3"],
   "fat_before_m3":vol_before["fat_m3"],"fat_after_m3":vol_after["fat_m3"],
   "fat_delta_m3":vol_after["fat_m3"]-vol_before["fat_m3"],
   "scope":"All other source tets are bitwise unaffected, so these local deltas equal global field-defined regional volume deltas; absolute global volume totals were not recomputed."},
 "local_surface_check":{"source_tet_count":int(len(local_ids)),
   "before":{"facet_count":int(len(before.facets)),"q_min":float(q0.min()),"q_below_1e-5":int(np.count_nonzero(q0<1e-5)),
      "self_pair_count":int(a0["count"]),"self_pair_examples":a0.get("triangle_pairs",[])[:50]},
   "after":{"facet_count":int(len(after.facets)),"q_min":float(q1.min()),"q_below_1e-5":int(np.count_nonzero(q1<1e-5)),
      "self_pair_count":int(a1["count"]),"self_pair_examples":a1.get("triangle_pairs",[])[:50]},
   "complete_global_self_audit":False,"complete_global_quality_audit":False},
 "runtime_seconds":time.monotonic()-t0}
(OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
np.savez_compressed(OUT/"local-before-facets.npz",points=before.points_float32,facets=before.facets,interface_ids=before.interface_ids,
                    facet_parent_indptr=before.facet_parent_indptr,facet_parent_tet_rows=before.facet_parent_tet_rows)
np.savez_compressed(OUT/"local-after-facets.npz",points=after.points_float32,facets=after.facets,interface_ids=after.interface_ids,
                    facet_parent_indptr=after.facet_parent_indptr,facet_parent_tet_rows=after.facet_parent_tet_rows)
print(json.dumps({"status":report["status"],"targets":report["edit"]["targets"],
 "outer_zero_set":report["edit"]["outer_zero_set"],"source_edge_cut_motion":cut_moves,
 "volume":report["local_field_defined_material_volume"],"surface":report["local_surface_check"],
 "runtime_seconds":report["runtime_seconds"]},indent=2))

