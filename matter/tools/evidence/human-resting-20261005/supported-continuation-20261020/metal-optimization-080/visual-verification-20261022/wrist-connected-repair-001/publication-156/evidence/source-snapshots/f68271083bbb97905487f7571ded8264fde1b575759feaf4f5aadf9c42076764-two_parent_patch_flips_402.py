#!/usr/bin/env python3
"""Fixed-coordinate local surface flips with explicit two-parent patch ownership.

The selected source-detail patches are processed as a coordinated patch. Their
outer rings and inter-parent seam edges are locked. Flips remain inside a
single source-detail/subface/material region, while immediate-parent lineage is
recorded as an explicit CSR union when the replacement face spans two prior
faces. This emits construction candidates only; exact local/full geometry and
all other gates remain external requirements.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

EPS=np.finfo(np.float64).eps

class PatchError(ValueError):
    pass

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()

def sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def edge_map(faces):
    out={}
    for fi,tri in enumerate(faces):
        for a,b in zip(tri,np.roll(tri,-1)):
            a,b=int(a),int(b)
            out.setdefault((min(a,b),max(a,b)),[]).append((fi,a,b))
    return out

def area_vector(points, faces):
    out=np.zeros(3,dtype=np.float64)
    for tri in faces:
        p=np.asarray(points)[np.asarray(tri,dtype=np.int64)].astype(np.float64)
        out += .5*np.cross(p[1]-p[0],p[2]-p[0])
    return out

def normal_area(points, tri):
    p=np.asarray(points)[np.asarray(tri,dtype=np.int64)].astype(np.float64)
    n=np.cross(p[1]-p[0],p[2]-p[0])
    q=float(np.linalg.norm(n))
    return (n/q if q>0 else np.zeros(3)),.5*q

def min_angle(points, tri):
    a,b,c=np.asarray(points)[np.asarray(tri,dtype=np.int64)].astype(np.float64)
    e0=np.linalg.norm(b-a); e1=np.linalg.norm(c-b); e2=np.linalg.norm(a-c)
    if min(e0,e1,e2)<=0:return 0.0
    x=np.arccos(np.clip(np.dot(b-a,c-a)/(e0*e2),-1.,1.))
    y=np.arccos(np.clip(np.dot(a-b,c-b)/(e0*e1),-1.,1.))
    return float(np.rad2deg(min(x,y,np.pi-x-y)))

def chart(points, normal, rows):
    n=np.asarray(normal,dtype=np.float64); q=float(np.linalg.norm(n))
    if not np.isfinite(q) or q<=64*EPS: raise PatchError("unstable chart normal")
    n=n/q; axis=np.eye(3)[int(np.argmin(np.abs(n)))]
    u=np.cross(n,axis); u/=np.linalg.norm(u); v=np.cross(n,u)
    p=np.asarray(points)[np.asarray(rows,dtype=np.int64)].astype(np.float64)
    return np.column_stack((p@u,p@v))

def strict_convex(poly):
    if poly.shape!=(4,2) or not np.isfinite(poly).all():return False
    e=np.roll(poly,-1,axis=0)-poly
    scale=float(np.max(np.sum(e*e,axis=1)))
    if scale<=0:return False
    c=np.asarray([e[i,0]*e[(i+1)%4,1]-e[i,1]*e[(i+1)%4,0] for i in range(4)])
    tol=128*EPS*scale
    return bool(np.all(c>tol) or np.all(c < -tol))

def csr_rows(ptr, vals, n):
    ptr=np.asarray(ptr); vals=np.asarray(vals)
    if ptr.shape!=(n+1,) or int(ptr[0])!=0 or int(ptr[-1])!=len(vals) or np.any(ptr[1:]<ptr[:-1]):
        raise PatchError("invalid ancestry CSR")
    return [list(map(int,vals[int(ptr[i]):int(ptr[i+1])])) for i in range(n)]

def csr_build(rows,dtype=np.int64):
    ptr=np.zeros(len(rows)+1,dtype=np.int64); vals=[]
    for i,row in enumerate(rows):
        vals.extend(sorted(set(map(int,row))))
        ptr[i+1]=len(vals)
    return ptr,np.asarray(vals,dtype=dtype)

def array_identical(a,b):
    x=np.ascontiguousarray(a); y=np.ascontiguousarray(b)
    return x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()

def validate_candidate(c):
    keys=("points_world_m","source_O_faces","area_reference_037_points_world_m_f64",
          "area_reference_037_source_O_faces","face_kind","parent_fine_detail_face_row",
          "parent_fine_subface_row","parent_coarse_face_row","parent_source_face_marker",
          "refinement_face_parent_face_row","refinement_face_parent_barycentric",
          "face_source_ancestry_indptr","face_source_ancestry_face_rows",
          "face_fine_subface_ancestry_indptr","face_fine_subface_ancestry_rows")
    missing=[k for k in keys if k not in c]
    if missing:raise PatchError(f"missing arrays: {missing}")
    x=np.asarray(c["points_world_m"]); f=np.asarray(c["source_O_faces"])
    r=np.asarray(c["area_reference_037_points_world_m_f64"]); rf=np.asarray(c["area_reference_037_source_O_faces"])
    n=len(f)
    if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():raise PatchError("points must be finite emitted F32 Nx3")
    if f.ndim!=2 or f.shape[1]!=3 or not np.issubdtype(f.dtype,np.integer) or (f.size and (f.min()<0 or f.max()>=len(x))):raise PatchError("invalid integer surface faces")
    if rf.shape!=f.shape or not np.array_equal(rf,f):raise PatchError("037 connectivity must match current faces")
    if r.dtype!=np.float64 or r.shape!=x.shape or not np.isfinite(r).all():raise PatchError("matched-037 coordinates must be finite F64 Nx3")
    for k in ("face_kind","parent_fine_detail_face_row","parent_fine_subface_row","parent_coarse_face_row","parent_source_face_marker","refinement_face_parent_face_row"):
        if np.asarray(c[k]).shape!=(n,) or not np.issubdtype(np.asarray(c[k]).dtype,np.signedinteger):raise PatchError(f"{k} must be signed face rows")
    if np.asarray(c["refinement_face_parent_barycentric"]).shape!=(n,3,3):raise PatchError("parent barycentric array must be Fx3x3")
    csr_rows(c["face_source_ancestry_indptr"],c["face_source_ancestry_face_rows"],n)
    csr_rows(c["face_fine_subface_ancestry_indptr"],c["face_fine_subface_ancestry_rows"],n)
    if "refinement_face_parent_ancestry_indptr" in c:
        csr_rows(c["refinement_face_parent_ancestry_indptr"],c["refinement_face_parent_ancestry_face_rows"],n)

def patch_edges(faces, detail, selected):
    em=edge_map(faces); outer=set(); seam=set()
    sel=set(selected)
    for edge,inc in em.items():
        local=[v for v in inc if v[0] in sel]
        if not local:continue
        if len(local)==1:outer.add(edge)
        elif len(local)==2:
            if int(detail[local[0][0]])!=int(detail[local[1][0]]):seam.add(edge)
        else:raise PatchError(f"selected patch has nonmanifold edge {edge}")
    return outer,seam

def candidate_flip(c, edge, inc, requested, locked, min_gain, min_dot, complete_edge_map=None):
    x=c["points_world_m"]; r=c["area_reference_037_points_world_m_f64"]; f=c["source_O_faces"]
    detail=c["parent_fine_detail_face_row"]; sub=c["parent_fine_subface_row"]; kind=c["face_kind"]
    if edge in locked:return None,"locked_outer_or_seam"
    if len(inc)!=2:return None,"not_interior_edge"
    i,u,v=inc[0]; j,u2,v2=inc[1]
    if (u2,v2)!=(v,u):return None,"directed_winding_mismatch"
    if int(detail[i])!=int(detail[j]) or int(detail[i]) not in requested:return None,"source_detail_mismatch"
    if int(sub[i])!=int(sub[j]):return None,"source_subface_mismatch"
    if int(kind[i])!=int(kind[j]):return None,"material_parent_mismatch"
    shared={u,v}; a=next(int(q) for q in f[i] if int(q) not in shared); b=next(int(q) for q in f[j] if int(q) not in shared)
    if len({u,v,a,b})!=4:return None,"non_four_vertex_quad"
    em=complete_edge_map if complete_edge_map is not None else edge_map(f)
    if (min(a,b),max(a,b)) in em:return None,"alternate_diagonal_exists"
    new=(np.asarray([u,b,a],dtype=f.dtype),np.asarray([b,v,a],dtype=f.dtype))
    old=(f[i],f[j])
    nc=[normal_area(x,t)[0] for t in old]; nr=[normal_area(r,t)[0] for t in old]
    if min(np.linalg.norm(n) for n in nc+nr)<=0:return None,"degenerate_old_face"
    ac=nc[0]+nc[1]; ar=nr[0]+nr[1]
    if np.linalg.norm(ac)<=64*EPS or np.linalg.norm(ar)<=64*EPS:return None,"ambiguous_old_normal"
    ac/=np.linalg.norm(ac); ar/=np.linalg.norm(ar)
    if min(float(np.dot(n,ac)) for n in nc)<=min_dot:return None,"current_old_orientation"
    if min(float(np.dot(n,ar)) for n in nr)<=min_dot:return None,"037_old_orientation"
    poly=[v,a,u,b]
    try:
        if not strict_convex(chart(x,ac,poly)) or not strict_convex(chart(r,ar,poly)):return None,"nonconvex_quad"
    except PatchError:return None,"chart_failure"
    for pts,avg,label in ((x,ac,"current"),(r,ar,"037")):
        new_na=[normal_area(pts,t) for t in new]
        scale2=max(float(np.sum((np.asarray(pts[p],np.float64)-np.asarray(pts[q],np.float64))**2))
                   for p,q in ((u,v),(v,a),(a,u),(u,b),(b,v)))
        if min(a0 for _,a0 in new_na)<=128*EPS*scale2:return None,f"{label}_nonpositive_new_face"
        if min(float(np.dot(n,avg)) for n,_ in new_na)<=min_dot:return None,f"{label}_new_orientation"
        delta=float(np.linalg.norm(area_vector(pts,new)-area_vector(pts,old)))
        if delta>2048*EPS*scale2:return None,f"{label}_patch_area_vector_changed"
    if min(float(np.dot(normal_area(x,t)[0],normal_area(r,t)[0])) for t in new)<=min_dot:return None,"current_037_orientation_disagreement"
    oldang=min(min_angle(x,t) for t in old); newang=min(min_angle(x,t) for t in new); gain=newang-oldang
    if gain<=min_gain:return None,"no_min_angle_gain"
    return {"face_rows":[int(i),int(j)],"edge":[int(min(u,v)),int(max(u,v))],
            "new_diagonal":[int(min(a,b)),int(max(a,b))],"new_faces":[new[0].tolist(),new[1].tolist()],
            "source_detail_face_row":int(detail[i]),"source_subface_row":int(sub[i]),
            "immediate_parent_union":sorted(set(map(int,c["refinement_face_parent_face_row"][[i,j]]))),
            "old_min_angle_deg":oldang,"new_min_angle_deg":newang,"angle_gain_deg":gain},"candidate"

def propose_patch_flips(candidate, source_detail_rows, seed_face_rows, *, max_flips=32, max_rounds=3, min_angle_gain_deg=1e-8, min_normal_dot=0.0):
    validate_candidate(candidate)
    if isinstance(max_flips,bool) or not isinstance(max_flips,int) or max_flips<0:raise PatchError("max_flips must be nonnegative integer")
    if isinstance(max_rounds,bool) or not isinstance(max_rounds,int) or max_rounds<0:raise PatchError("max_rounds must be nonnegative integer")
    requested=tuple(sorted(set(map(int,source_detail_rows))))
    if not requested or len(requested)!=len(source_detail_rows):raise PatchError("source detail rows must be nonempty and unique")
    f=np.asarray(candidate["source_O_faces"]); detail=np.asarray(candidate["parent_fine_detail_face_row"])
    nface=len(f); selected=np.flatnonzero(np.isin(detail,requested)).tolist()
    if not selected or any(not np.any(detail==row) for row in requested):raise PatchError("requested source detail patch missing")
    seeds=set(map(int,seed_face_rows)); selected_set=set(selected)
    if not seeds or any(i<0 or i>=nface or i not in selected_set for i in seeds):raise PatchError("seed rows must belong to selected patches")
    outer,seams=patch_edges(f,detail,selected); locked=outer|seams
    out={k:np.array(v,copy=True) for k,v in candidate.items()}; accepted=[]; rejected={}; active=seeds
    for round_idx in range(max_rounds):
        if len(accepted)>=max_flips or not active:break
        em=edge_map(out["source_O_faces"]); seen=set(); eligible=[]
        for seed in sorted(active):
            for a,b in zip(out["source_O_faces"][seed],np.roll(out["source_O_faces"][seed],-1)):
                edge=(min(int(a),int(b)),max(int(a),int(b))); inc=em.get(edge,[])
                key=tuple(sorted(q[0] for q in inc)) if len(inc)==2 else None
                if key is not None and key in seen:continue
                if key is not None:seen.add(key)
                op,why=candidate_flip(out,edge,inc,set(requested),locked,min_angle_gain_deg,min_normal_dot,em)
                if op is None:rejected[why]=rejected.get(why,0)+1
                else:eligible.append(op)
        eligible.sort(key=lambda q:(-q["angle_gain_deg"],q["source_detail_face_row"],q["face_rows"],q["edge"]))
        chosen=[]; usedf=set(); usedn=set()
        for op in eligible:
            if len(accepted)+len(chosen)>=max_flips:break
            fr=set(op["face_rows"]); nodes={int(v) for t in op["new_faces"] for v in t}
            if fr&usedf or nodes&usedn:continue
            chosen.append(op); usedf|=fr; usedn|=nodes
        if not chosen:break
        for op in chosen:
            i,j=op["face_rows"]
            out["source_O_faces"][i]=op["new_faces"][0]; out["source_O_faces"][j]=op["new_faces"][1]
            out["area_reference_037_source_O_faces"][i]=op["new_faces"][0]; out["area_reference_037_source_O_faces"][j]=op["new_faces"][1]
            accepted.append(op)
        em=edge_map(out["source_O_faces"]); active=set(); changed={r for op in chosen for r in op["face_rows"]}
        for inc in em.values():
            ids={r[0] for r in inc}
            if ids&changed:active.update(ids)
    if accepted:
        nface=len(f)
        src=csr_rows(candidate["face_source_ancestry_indptr"],candidate["face_source_ancestry_face_rows"],nface)
        sub=csr_rows(candidate["face_fine_subface_ancestry_indptr"],candidate["face_fine_subface_ancestry_rows"],nface)
        if "refinement_face_parent_ancestry_indptr" in candidate:
            par=csr_rows(candidate["refinement_face_parent_ancestry_indptr"],candidate["refinement_face_parent_ancestry_face_rows"],nface)
        else:par=[[int(v)] if int(v)>=0 else [] for v in np.asarray(candidate["refinement_face_parent_face_row"])]
        multi=set()
        for op in accepted:
            i,j=op["face_rows"]; su=sorted(set(src[i])|set(src[j])); fu=sorted(set(sub[i])|set(sub[j])); pu=sorted(set(par[i])|set(par[j]))
            for row in (i,j):src[row]=su;sub[row]=fu;par[row]=pu
            out["refinement_face_parent_face_row"][[i,j]]=pu[0] if len(pu)==1 else -1
            if len(pu)>1:
                out["refinement_face_parent_barycentric"][[i,j]]=np.nan;multi.update((i,j))
            for key in ("parent_coarse_face_row","parent_source_face_marker"):
                if key in out and int(out[key][i])!=int(out[key][j]):out[key][[i,j]]=-1
        out["face_source_ancestry_indptr"],out["face_source_ancestry_face_rows"]=csr_build(src,np.asarray(candidate["face_source_ancestry_face_rows"]).dtype)
        out["face_fine_subface_ancestry_indptr"],out["face_fine_subface_ancestry_rows"]=csr_build(sub,np.asarray(candidate["face_fine_subface_ancestry_rows"]).dtype)
        out["refinement_face_parent_ancestry_indptr"],out["refinement_face_parent_ancestry_face_rows"]=csr_build(par,np.int64)
        out["refinement_face_parent_ancestry_semantics"]=np.asarray(["CSR union of immediate parent face rows; singular row -1 and singular barycentrics NaN when multiple parents apply"])
        out["refinement_face_parent_barycentric_semantics"]=np.asarray(["single-parent affine map only; NaN explicitly marks multi-parent patch flips"])
    newf=out["source_O_faces"]; outer2,seams2=patch_edges(newf,detail,selected)
    if outer2!=outer or seams2!=seams:raise PatchError("outer boundary or source-parent seam changed")
    em=edge_map(newf)
    if any(len(v)>2 for v in em.values()):raise PatchError("nonmanifold edge introduced")
    if len({tuple(sorted(map(int,t))) for t in newf})!=len(newf):raise PatchError("duplicate face introduced")
    if not np.array_equal(newf,out["area_reference_037_source_O_faces"]):raise PatchError("037 connectivity mismatch")
    if not np.array_equal(out["points_world_m"],candidate["points_world_m"]):raise PatchError("coordinates moved")
    changed=sorted({r for op in accepted for r in op["face_rows"]})
    for row in changed:
        for pts,label in ((out["points_world_m"],"current"),(out["area_reference_037_points_world_m_f64"],"037")):
            _,a=normal_area(pts,newf[row])
            if not np.isfinite(a) or a<=0:raise PatchError(f"{label} face {row} is not positive")
    allowed={"source_O_faces","area_reference_037_source_O_faces","face_source_ancestry_indptr","face_source_ancestry_face_rows","face_fine_subface_ancestry_indptr","face_fine_subface_ancestry_rows","refinement_face_parent_face_row","refinement_face_parent_barycentric","refinement_face_parent_ancestry_indptr","refinement_face_parent_ancestry_face_rows","refinement_face_parent_ancestry_semantics","refinement_face_parent_barycentric_semantics","parent_coarse_face_row","parent_source_face_marker"}
    for k,v in candidate.items():
        if k not in allowed and not (out[k].dtype==v.dtype and out[k].shape==v.shape and np.ascontiguousarray(out[k]).tobytes()==np.ascontiguousarray(v).tobytes()):raise PatchError(f"unapproved array mutation: {k}")
    report={"status":"LOCAL_PATCH_RETRIANGULATION_PROPOSED" if accepted else "NO_ELIGIBLE_LOCAL_FLIPS","geometry_admitted":False,"physical_steps":0,
            "local_exact_self_audit_required":True,"full_exact_self_and_all185_gates_required_before_admission":True,
            "source_detail_rows":list(requested),"selected_face_count":len(selected),"seed_face_rows":sorted(seeds),
            "accepted_flip_count":len(accepted),"changed_face_rows":changed,"multi_parent_changed_face_rows":sorted(multi) if accepted else [],
            "locked_outer_edge_count":len(outer),"locked_outer_edges":[list(e) for e in sorted(outer)],
            "locked_inter_parent_seam_edge_count":len(seams),"locked_inter_parent_seam_edges":[list(e) for e in sorted(seams)],
            "patch_boundary_and_seam_preserved":True,"points_bitwise_unchanged":True,"matched_037_faces_synchronized":True,
            "source_face_ancestry_union_regenerated":bool(accepted),"immediate_parent_union_csr_regenerated":bool(accepted),
            "rejection_reason_counts":rejected,"operations":accepted}
    return out,report

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--candidate",required=True);ap.add_argument("--source-detail-rows",required=True);ap.add_argument("--seed-face-rows",required=True);ap.add_argument("--output-dir",required=True);ap.add_argument("--max-flips",type=int,default=32);ap.add_argument("--max-rounds",type=int,default=3)
    a=ap.parse_args(); cp=Path(a.candidate);od=Path(a.output_dir)
    if od.exists():raise SystemExit("output directory exists")
    od.mkdir(parents=True)
    with np.load(cp,allow_pickle=False) as z:c={k:z[k] for k in z.files}
    src=json.loads(Path(a.source_detail_rows).read_text());seeds=json.loads(Path(a.seed_face_rows).read_text())
    out,rep=propose_patch_flips(c,src,seeds,max_flips=a.max_flips,max_rounds=a.max_rounds)
    outpath=od/"candidate.npz";np.savez_compressed(outpath,**out)
    rep.update(candidate_input_sha256=sha_file(cp),candidate_output_sha256=sha_file(outpath),points_sha256=sha_array(out["points_world_m"]),input_faces_sha256=sha_array(c["source_O_faces"]),output_faces_sha256=sha_array(out["source_O_faces"]),source_detail_rows_file_sha256=sha_file(a.source_detail_rows),seed_rows_file_sha256=sha_file(a.seed_face_rows),builder_sha256=sha_file(Path(__file__)),output_candidate_path=str(outpath))
    (od/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in rep.items() if k not in ("operations","locked_outer_edges","locked_inter_parent_seam_edges")},indent=2))
if __name__=="__main__":main()
