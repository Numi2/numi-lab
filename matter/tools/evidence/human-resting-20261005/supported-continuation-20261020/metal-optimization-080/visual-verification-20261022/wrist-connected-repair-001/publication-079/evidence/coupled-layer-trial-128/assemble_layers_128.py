#!/usr/bin/env python3
"""Construct an O/I/D layered surface proposal for a pinned hybrid O mesh.

This is static geometric construction only. It does not create tetrahedra,
material ancestry, mass transfer, or native admission.
"""
from __future__ import annotations

import argparse, csv, gzip, hashlib, importlib.util, json, sys, time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE.parent
DEFAULT_OUTER = D / "connected-source-patch-cut-123/run-017-expanded-target-hybrid/hybrid-O-surface-128.npz"
DEFAULT_BASELINE = D / "connected-capacity-reference-115/candidate.npz"
DEFAULT_FIELDS = D / "fine-layer-fields-127/capacity-115-coarse-to-detail-layer-fields.npz"
DEFAULT_FACE_CSV_GZ = D / "publication-071/evidence/exact-transfer-075/run-001/source-preflight/accepted-coarse-faces.csv.gz"
DEFAULT_OUTPUT = HERE / "run-001"
E = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
HELPER = D / "connected-layer-reconstruction-127/matter/tools/human_skin_fat_layer_reconstruction.py"
PREDICATE = E / "collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py"
PREDICATE_IMPORTS = E / "incremental-exact-audit-owner-253/src"

PINS = {
    "baseline115_sha256": "abff4290774cc96a17f6c68a21fe86fc3933a1878e0bd6512733e99589fb6dd9",
    "capacity_fields_sha256": "4278c32fc30f2ec2ce67131e6883782f5f266d6f04877deaec4743cf3a196145",
    "helper_sha256": "2a7f7ed02a5c90af11e09ffd8b8d2a34b2f3ff9c3b08ecb08514cccb29dc565e",
}

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def need(ok, message):
    if not ok: raise ValueError(message)

def face_csv(path: Path):
    raw = gzip.open(path,"rb").read() if path.suffix==".gz" else path.read_bytes()
    raw_hash = hashlib.sha256(raw).hexdigest()
    rows=[]
    import io
    for r in csv.DictReader(io.StringIO(raw.decode("utf-8"))):
        rows.append((int(r["vertex0"]),int(r["vertex1"]),int(r["vertex2"])))
    return np.asarray(rows,dtype=np.int64), raw_hash

def edges_of_faces(F):
    edgefaces=defaultdict(list)
    for fi,t in enumerate(F):
        for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
            edgefaces[tuple(sorted((int(a),int(b))))].append((fi,int(a),int(b)))
    return edgefaces

def shell_faces(Fout, boundary_edges, n, offset_outer, offset_inner):
    """Close one oriented layer shell using the directed open boundary edges."""
    inner = Fout[:,[0,2,1]] + offset_inner
    outer = Fout + offset_outer
    cap=[]
    for a,b in boundary_edges:
        a=int(a)+offset_outer; b=int(b)+offset_outer
        ai=int(a-offset_outer)+offset_inner
        bi=int(b-offset_outer)+offset_inner
        cap.extend(((b,a,ai),(b,ai,bi)))
    return np.vstack((outer,inner,np.asarray(cap,dtype=np.int64).reshape(-1,3)))

def shell_topology(F):
    edge=defaultdict(list)
    for t in F:
        for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
            key=tuple(sorted((int(a),int(b))))
            edge[key].append(1 if a<b else -1)
    counts=Counter(len(v) for v in edge.values())
    bad=[(e,v) for e,v in edge.items() if len(v)!=2 or sum(v)!=0]
    return {"edge_incidence_histogram":dict(counts),"edge_count":len(edge),
            "bad_oriented_edge_count":len(bad),"bad_edge_witnesses":[[list(e),v] for e,v in bad[:20]]}

def signed_volume(X,F):
    T=X[F].astype(np.float64)
    return float(np.einsum("ij,ij->i",T[:,0],np.cross(T[:,1],T[:,2])).sum()/6.0)

def exact_self_audit(X,F):
    sys.path.insert(0,str(PREDICATE_IMPORTS))
    name="numilab_human.cardiac_cavity_intersections"
    spec=importlib.util.spec_from_file_location(name,PREDICATE)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    rec,_=mod.float32_surface_records(np.asarray(X,dtype=np.float32),np.asarray(F,dtype=np.int64),source_seam_groups=())
    out=mod._audit_self_pairs_touching_faces(rec,set(range(len(F))))
    pairs=np.asarray(out["triangle_pairs"],dtype=np.int64).reshape(-1,2)
    return {"pairs":int(len(pairs)),"complete":bool(out.get("complete_surface_audit",False)),
            "pairs_array":pairs}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outer-npz",type=Path,default=DEFAULT_OUTER)
    ap.add_argument("--baseline-npz",type=Path,default=DEFAULT_BASELINE)
    ap.add_argument("--fields-npz",type=Path,default=DEFAULT_FIELDS)
    ap.add_argument("--coarse-faces-csv-gz",type=Path,default=DEFAULT_FACE_CSV_GZ)
    ap.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    ap.add_argument("--expected-outer-sha256")
    ap.add_argument("--orientation-sign",type=int,choices=(-1,1),default=-1)
    ap.add_argument("--no-exact-audit",action="store_true")
    ap.add_argument("--baseline-offset-mode",action="store_true",help="preserve 115 baseline I/D vectors exactly; do not normal-reconstruct active patch")
    args=ap.parse_args()
    for p in (args.outer_npz,args.baseline_npz,args.fields_npz,args.coarse_faces_csv_gz,HELPER,PREDICATE):
        need(p.is_file(),f"missing pinned input: {p}")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    outsha=sha256(args.outer_npz)
    if args.expected_outer_sha256: need(outsha==args.expected_outer_sha256,"outer hash mismatch")
    bsha=sha256(args.baseline_npz); fsha=sha256(args.fields_npz); hsha=sha256(HELPER)
    need(bsha==PINS["baseline115_sha256"],"baseline 115 hash mismatch")
    need(fsha==PINS["capacity_fields_sha256"],"capacity-115 field asset hash mismatch")
    need(hsha==PINS["helper_sha256"],"layer helper hash mismatch")
    faces_source, faces_csv_sha=face_csv(args.coarse_faces_csv_gz)
    need(faces_csv_sha=="7311267392275649abeb3ecaf6a5ca43eb4d67cf2a9b7f13a9c692587c36b1e2",
         "accepted coarse face CSV content hash mismatch")

    spec=importlib.util.spec_from_file_location("layer_reconstruction_127",HELPER)
    helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)

    with np.load(args.outer_npz,allow_pickle=False) as z: O={k:z[k] for k in z.files}
    with np.load(args.baseline_npz,allow_pickle=False) as z: B={k:z[k] for k in z.files}
    with np.load(args.fields_npz,allow_pickle=False) as z: A={k:z[k] for k in z.files}
    P=np.asarray(O["points_world_m"],dtype=np.float64); F=np.asarray(O["source_O_faces"],dtype=np.int64)
    nk=np.asarray(O["node_kind"],dtype=np.int64); sr=np.asarray(O["node_source_row"],dtype=np.int64)
    n=len(P)
    need(P.shape==(n,3) and len(nk)==n and F.ndim==2 and F.shape[1]==3,"outer schema mismatch")
    need(np.isfinite(P).all() and F.min()>=0 and F.max()<n,"invalid outer mesh")
    need(np.isin(nk,[0,1,2,5]).all(),f"unsupported node kinds {np.unique(nk).tolist()}")
    need(faces_source.shape==(4777,3),"coarse face count mismatch")
    need(B["source_reference_points_m"].shape==(7360,3),"115 baseline shape mismatch")
    need(A["capacity115_coarse_I_reference_m"].shape==(2411,3),"coarse field shape mismatch")
    need(A["capacity115_coarse_D_reference_m"].shape==(2411,3),"deep field shape mismatch")
    need(A["capacity115_fine_skin_offset_I_minus_O_m"].shape==(61950,3),"fine skin offset shape mismatch")
    need(A["capacity115_fine_fat_offset_D_minus_I_m"].shape==(61950,3),"fine fat offset shape mismatch")

    Ocoarse=np.asarray(A["capacity115_coarse_O_reference_m"],dtype=np.float64)
    Icoarse=np.asarray(A["capacity115_coarse_I_reference_m"],dtype=np.float64)
    Dcoarse=np.asarray(A["capacity115_coarse_D_reference_m"],dtype=np.float64)
    hs_coarse=np.asarray(A["capacity115_coarse_hskin_euclidean_m"],dtype=np.float64)
    hf_coarse=np.asarray(A["capacity115_coarse_hfat_euclidean_m"],dtype=np.float64)
    Ofields=np.zeros((n,3),dtype=np.float64); Ibase=np.zeros_like(Ofields); Dbase=np.zeros_like(Ofields)
    hs=np.zeros(n,dtype=np.float64); hf=np.zeros(n,dtype=np.float64)
    kind_counts={str(int(k)):int(np.sum(nk==k)) for k in np.unique(nk)}
    direct_res=[]
    # Kinds 0,1,2,5 are explicit provenance cases from emitter 128.
    for i,k in enumerate(nk):
        if k==0:
            r=int(sr[i]);need(0<=r<2411,f"kind0 source row invalid at {i}")
            Ofields[i]=Ocoarse[r];Ibase[i]=Icoarse[r];Dbase[i]=Dcoarse[r]
            hs[i]=hs_coarse[r];hf[i]=hf_coarse[r];direct_res.append(float(np.linalg.norm(P[i]-Ocoarse[r])))
        elif k==1:
            r=int(sr[i]);need(0<=r<61950,f"kind1 detail row invalid at {i}")
            need(bool(A["fine_active_mask"][r]),f"kind1 row {i} maps inactive detail {r}")
            di=np.asarray(A["capacity115_fine_skin_offset_I_minus_O_m"][r],dtype=np.float64)
            dd=np.asarray(A["capacity115_fine_fat_offset_D_minus_I_m"][r],dtype=np.float64)
            need(np.isfinite(di).all() and np.isfinite(dd).all(),f"nonfinite fine offsets at {i}")
            Ofields[i]=P[i];Ibase[i]=P[i]+di;Dbase[i]=Ibase[i]+dd
            hs[i]=float(A["capacity115_fine_hskin_euclidean_m"][r]);hf[i]=float(A["capacity115_fine_hfat_euclidean_m"][r])
        elif k==2:
            a=int(O["node_parent_coarse_edge_a"][i]);b=int(O["node_parent_coarse_edge_b"][i]);t=float(O["node_parent_coarse_edge_parameter"][i])
            need(0<=a<2411 and 0<=b<2411 and a!=b and np.isfinite(t) and 0<=t<=1,f"invalid coarse edge provenance at {i}")
            Ofields[i]=(1-t)*Ocoarse[a]+t*Ocoarse[b]
            Ibase[i]=(1-t)*Icoarse[a]+t*Icoarse[b]
            Dbase[i]=(1-t)*Dcoarse[a]+t*Dcoarse[b]
            hs[i]=(1-t)*hs_coarse[a]+t*hs_coarse[b];hf[i]=(1-t)*hf_coarse[a]+t*hf_coarse[b]
            direct_res.append(float(np.linalg.norm(P[i]-Ofields[i])))
        elif k==5:
            fr=int(O["node_parent_coarse_face_row"][i]);need(0<=fr<len(faces_source),f"invalid coarse face parent at {i}")
            tri=faces_source[fr]
            Ofields[i]=Ocoarse[tri].mean(axis=0);Ibase[i]=Icoarse[tri].mean(axis=0);Dbase[i]=Dcoarse[tri].mean(axis=0)
            hs[i]=hs_coarse[tri].mean();hf[i]=hf_coarse[tri].mean()
            direct_res.append(float(np.linalg.norm(P[i]-Ofields[i])))
    # Kind 0 and generated boundary rows must be a parent-domain point match.
    max_parent_res=float(max(direct_res,default=0.0))
    need(max_parent_res<=1.0e-9,f"coarse parent reconstruction residual {max_parent_res:g} m exceeds 1 nm")
    Ofields=P.copy()  # preserve the exact emitted hybrid O coordinates
    di=Ibase-Ofields;dd=Dbase-Ibase
    need(np.isfinite(hs).all() and np.all(hs>0) and np.isfinite(hf).all() and np.all(hf>0),
         "all output vertices need finite positive capacity-115 thickness fields")

    # The active region is the union of face_kind==2 detail patch faces.
    fk=np.asarray(O["face_kind"],dtype=np.int64);need(fk.shape==(len(F),),"face kind shape mismatch")
    active=(np.asarray([],dtype=np.int64) if args.baseline_offset_mode else np.unique(F[fk==2]))
    edgefaces=edges_of_faces(F)
    interface_edges=[]; boundary_halfedges=[]; nonmanifold=[]
    for e,occ in edgefaces.items():
        if len(occ)>2: nonmanifold.append((e,len(occ)))
        kinds={int(fk[q[0]]) for q in occ}
        if len(occ)==2 and 2 in kinds and len(kinds)>1: interface_edges.append(e)
        if len(occ)==1: boundary_halfedges.append((occ[0][1],occ[0][2],occ[0][0],int(fk[occ[0][0]])))
    need(not nonmanifold,f"outer mesh has nonmanifold edges {nonmanifold[:8]}")
    boundary=(np.asarray([],dtype=np.int64) if args.baseline_offset_mode else np.unique(np.asarray([v for e in interface_edges for v in e],dtype=np.int64)))
    if not args.baseline_offset_mode:
        need(len(active)>0 and len(boundary)>0,"active detail patch or interface boundary is empty")
        need(np.isin(boundary,active).all(),"detail patch interface endpoints must be active vertices")
    # Fixed O/I/D rows are exactly their baseline values on the active patch boundary.
    rec=helper.reconstruct_connected_layers(
        P,F,di,dd,boundary,P[boundary],Ibase[boundary],Dbase[boundary],
        skin_depth_m=hs,fat_depth_m=hf,orientation_sign=args.orientation_sign,
        active_vertex_indices=active)
    I=rec.inner_points;Ddeep=rec.deep_points
    # Keep exact unchanged baseline rows outside the patch, and exact prescribed boundary.
    need(np.array_equal(I[np.setdiff1d(np.arange(n),active)],Ibase[np.setdiff1d(np.arange(n),active)]),
         "outside-patch I changed")
    need(np.array_equal(Ddeep[np.setdiff1d(np.arange(n),active)],Dbase[np.setdiff1d(np.arange(n),active)]),
         "outside-patch D changed")
    need(np.array_equal(I[boundary],Ibase[boundary]) and np.array_equal(Ddeep[boundary],Dbase[boundary]),
         "Dirichlet layer boundary changed")

    # Construct closed material shells by capping the 45 (or current-input) open O boundary edges.
    open_half=[(a,b,fi) for a,b,fi,k in boundary_halfedges]
    open_edges=np.asarray([(a,b) for a,b,_ in open_half],dtype=np.int64).reshape(-1,2)
    # Reverse O faces when emitter winding points toward I; normal-derived reconstruction uses same sign.
    Fout=F if args.orientation_sign==1 else F[:,[0,2,1]]
    # Rebuild oriented boundary halfedges after choosing physical outward orientation.
    out_occ=edges_of_faces(Fout); out_boundary=[]
    for e,occ in out_occ.items():
        if len(occ)==1: out_boundary.append((occ[0][1],occ[0][2]))
    shell_n=2*n
    skinF=shell_faces(Fout,np.asarray(out_boundary,dtype=np.int64).reshape(-1,2),n,0,n)
    # The same directed outward boundary edge is used on the I outer surface of fat.
    fatF=shell_faces(Fout,np.asarray(out_boundary,dtype=np.int64).reshape(-1,2),n,n,2*n)
    # Shell vertex domains are [O,I] and [I,D].
    allX=np.vstack((P,I,Ddeep))
    skin_top=shell_topology(skinF);fat_top=shell_topology(fatF)
    skin_vol=signed_volume(allX.astype(np.float64),skinF)
    fat_vol=signed_volume(allX.astype(np.float64),fatF)
    need(len(out_boundary)==len(open_half), "oriented boundary edge count changed")
    # Store both double construction and exact emitted F32 views.
    f32=np.vstack((P.astype(np.float32),I.astype(np.float32),Ddeep.astype(np.float32)))
    allX32=f32
    O32=f32[:n];I32=f32[n:2*n];D32=f32[2*n:]
    # Face area and layer offset diagnostics in the emitted F32 domain.
    def area_stats(X):
        q=X[F].astype(np.float64);a=.5*np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1)
        return {"min_m2":float(a.min(initial=np.inf)),"nonpositive":int(np.count_nonzero(~np.isfinite(a)|(a<=0))),"count":int(len(a))}
    # Audit all exact self-pairs in the emitted Float32 lattice. This is not a PLC proof.
    audit={}
    if not args.no_exact_audit:
        t=time.monotonic()
        for label,X,faces in (("O",O32,Fout),("I",I32,Fout),("D",D32,Fout),
                              ("skin_shell",allX32,skinF),("fat_shell",allX32,fatF)):
            a=exact_self_audit(X,faces)
            pairs=a.pop("pairs_array")
            np.save(args.output_dir/f"{label}-self-pairs.npy",pairs)
            audit[label]={**a,"wall_seconds":time.monotonic()-t}
            t=time.monotonic()

    np.savez_compressed(args.output_dir/"layer-candidate.npz",
        outer_points_f64=P,inner_points_f64=I,deep_points_f64=Ddeep,
        outer_points_f32=O32,inner_points_f32=I32,deep_points_f32=D32,
        outer_faces=Fout.astype(np.int32),inner_faces=Fout[:,[0,2,1]].astype(np.int32),
        deep_faces=Fout.astype(np.int32),active_vertex_rows=active.astype(np.int32),
        fixed_interface_rows=boundary.astype(np.int32),boundary_edge_rows=np.asarray(out_boundary,dtype=np.int32),
        skin_shell_faces=skinF.astype(np.int32),fat_shell_faces=fatF.astype(np.int32),
        skin_depth_m=hs,fat_depth_m=hf,node_kind=nk.astype(np.int8),node_source_row=sr.astype(np.int32))
    report={
       "status":"CONSTRUCTION_ONLY","geometry_admitted":False,"tetgen_run":False,"material_or_mass_mapping_created":False,
       "baseline_offset_mode":bool(args.baseline_offset_mode),
       "inputs":{"outer_npz":str(args.outer_npz),"outer_sha256":outsha,"baseline115_npz":str(args.baseline_npz),"baseline115_sha256":bsha,
                 "capacity_fields_npz":str(args.fields_npz),"capacity_fields_sha256":fsha,"coarse_faces_gz":str(args.coarse_faces_csv_gz),
                 "coarse_face_csv_content_sha256":faces_csv_sha,"helper":str(HELPER),"helper_sha256":hsha,
                 "predicate_module":str(PREDICATE),"predicate_sha256":sha256(PREDICATE)},
       "counts":{"nodes":int(n),"outer_faces":int(len(F)),"node_kind":kind_counts,"active_patch_vertices":int(len(active)),
                 "fixed_patch_interface_vertices":int(len(boundary)),"patch_transition_edges":len(interface_edges),
                 "open_outer_boundary_edges":len(open_half),"closed_skin_shell_faces":len(skinF),"closed_fat_shell_faces":len(fatF)},
       "field_mapping":{"kind0_direct_max_position_residual_m":max_parent_res,
            "kind1_fine_detail_rows":int(np.count_nonzero(nk==1)),"kind2_coarse_edge_rows":int(np.count_nonzero(nk==2)),
            "kind5_coarse_face_centroid_rows":int(np.count_nonzero(nk==5)),
            "orientation_sign":args.orientation_sign,"orientation_basis":"chosen from capacity-115 signed I-O versus emitted O winding; inward raw winding requires -1",
            "hskin_min_m":float(hs.min()),"hskin_max_m":float(hs.max()),"hfat_min_m":float(hf.min()),"hfat_max_m":float(hf.max())},
       "reconstruction":rec.diagnostics,"shell_topology":{"skin":skin_top,"fat":fat_top},
       "shell_signed_volume_m3":{"skin":skin_vol,"fat":fat_vol},
       "face_area_f32":{"outer":area_stats(O32),"inner":area_stats(I32),"deep":area_stats(D32)},
       "exact_self_audit_float32":audit,
       "limits":["Inferred geometric thickness fields; not measured anatomy.",
                 "Harmonic layer construction is not a volume mesh, conservative map, mass transfer, or physical/native admission.",
                 "No all-bank bone clearance or volume-cell quality check was run."]}
    (args.output_dir/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    (args.output_dir/"README.txt").write_text(
        "Construction-only O/I/D surface proposal. No volume cells, mass transfer, bone clearance, or native physics admission.\n"+
        f"Outer input SHA-256: {outsha}\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()

