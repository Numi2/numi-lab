#!/usr/bin/env python3
"""One bounded sparse LP for coupled O/I/D static layer untangling.
Construction diagnostics only; no tetrahedra, mass map, or physical admission.
"""
from __future__ import annotations
import json, hashlib, importlib.util, sys, time, pathlib, collections, traceback
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

ROOT=pathlib.Path("/Users/numi/Work/numi-human-connected-surface-069")
E=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
RUN=ROOT/"coupled-layer-trial-128/run-003"
OUT=ROOT/"coupled-layer-clearance-133/run-003-gradient-caps-005-reduced-domain"
SEP_PATH=ROOT/"publication-079/canonical/matter/tools/human_skin_fat_surface_separation.py"
GRAD_PATH=ROOT/"connected_face_gradient_guard_141.py"
POST=ROOT/"coupled-layer-clearance-133/run-002/post-audit-002"
PRIOR=ROOT/"coupled-layer-clearance-133/run-002"
PREVIOUS_OUT=ROOT/"coupled-layer-clearance-133/run-003-gradient-caps-004"
PRED=E/"collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py"
PRED_IMPORTS=E/"incremental-exact-audit-owner-253/src"
INPUT_SHA="a420a94e7485a226bfc605736da600b2787b80fe1c42a3303e7c09c88676a7b5"
PAIR_MM=0.051
COLUMN_MM=0.05
SMOOTH_WEIGHT=0.05
TIME_LIMIT_S=60.0

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def need(ok,msg):
    if not ok: raise RuntimeError(msg)

def loadmod(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

need(not OUT.exists(),f"refusing to overwrite {OUT}")
need(RUN.joinpath("layer-candidate.npz").is_file(),"missing run-003 input")
need(sha(RUN/"layer-candidate.npz")==INPUT_SHA,"run-003 input hash mismatch")
need(SEP_PATH.is_file() and PRED.is_file() and GRAD_PATH.is_file(),"missing exact owner helper")
sep=loadmod("human_skin_fat_surface_separation_133",SEP_PATH)
grad=loadmod("connected_face_gradient_guard_141",GRAD_PATH)
sys.path.insert(0,str(PRED_IMPORTS))
pred=loadmod("numilab_human.cardiac_cavity_intersections",PRED)
OUT.mkdir(parents=True,exist_ok=False)
start=time.monotonic()
z=np.load(RUN/"layer-candidate.npz",allow_pickle=False)
O=np.asarray(z["outer_points_f32"],dtype=np.float64)
I=np.asarray(z["inner_points_f32"],dtype=np.float64)
D=np.asarray(z["deep_points_f32"],dtype=np.float64)
F=np.asarray(z["outer_faces"],dtype=np.int64)
n=len(O); nf=len(F)
need(O.shape==I.shape==D.shape==(n,3),"layer point shape mismatch")
need(F.ndim==2 and F.shape[1]==3 and int(F.min())>=0 and int(F.max())<n,"face domain invalid")
need(np.array_equal(z["inner_faces"],F[:,[0,2,1]]) and np.array_equal(z["deep_faces"],F),"layer face topology mismatch")

def classify_shell_pairs(path,label):
    raw=np.load(path).astype(np.int64).reshape(-1,2)
    groups={k:[] for k in ("outer_inner","outer_outer","inner_inner","cap_inner","cap_outer","cap_cap")}
    def section(face):
        return 0 if face<nf else (1 if face<2*nf else 2)
    for aa,bb in raw:
        a,b=int(aa),int(bb); sa,sb=section(a),section(b)
        if {sa,sb}=={0,1}: key="outer_inner"
        elif sa==sb==0: key="outer_outer"
        elif sa==sb==1: key="inner_inner"
        elif {sa,sb}=={1,2}: key="cap_inner"
        elif {sa,sb}=={0,2}: key="cap_outer"
        elif sa==sb==2: key="cap_cap"
        else: raise RuntimeError(f"{label} has unclassified pair {a}/{b}")
        groups[key].append((a,b))
    return {k:np.asarray(v,dtype=np.int64).reshape(-1,2) for k,v in groups.items()}

def unique_pairs(*tables):
    arrays=[np.asarray(x,dtype=np.int64).reshape(-1,2) for x in tables if np.asarray(x).size]
    return np.unique(np.vstack(arrays),axis=0) if arrays else np.empty((0,2),dtype=np.int64)

def section_cross_rows(pairs, section_a, section_b, label):
    out=[]
    for aa,bb in np.asarray(pairs,dtype=np.int64).reshape(-1,2):
        a,b=int(aa),int(bb); sa=0 if a<nf else (1 if a<2*nf else 2); sb=0 if b<nf else (1 if b<2*nf else 2)
        if (sa,sb)==(section_a,section_b):
            out.append((a-section_a*nf,b-section_b*nf))
        elif (sa,sb)==(section_b,section_a):
            out.append((b-section_a*nf,a-section_b*nf))
        else:
            raise RuntimeError(f"{label} pair {a}/{b} is not in requested sections")
    return unique_pairs(np.asarray(out,dtype=np.int64).reshape(-1,2))

skin0=classify_shell_pairs(RUN/"skin_shell-self-pairs.npy","baseline skin shell")
fat0=classify_shell_pairs(RUN/"fat_shell-self-pairs.npy","baseline fat shell")
need(len(skin0["outer_inner"])==428,"baseline skin O/I pair inventory differs from run-003 receipt")
need(len(fat0["outer_inner"])==216,"baseline fat I/D pair inventory differs from run-003 receipt")
D_base=np.load(RUN/"D-self-pairs.npy").astype(np.int64).reshape(-1,2)
need(len(fat0["inner_inner"])==len(D_base)==55,"baseline D/D pair inventory mismatch")
need(set(map(tuple,section_cross_rows(fat0["inner_inner"],1,1,"baseline D/D")))==set(map(tuple,D_base)),
     "baseline fat shell D/D rows do not match standalone D self-pair inventory")
capD_base=fat0["cap_inner"]
need(len(capD_base)==3,"unexpected baseline cap/D exact pair inventory")

skin_new=classify_shell_pairs(POST/"skin_shell-pairs.npy","attempt-002 skin shell")
fat_new=classify_shell_pairs(POST/"fat_shell-pairs.npy","attempt-002 fat shell")
skin={
 "outer_inner":unique_pairs(section_cross_rows(skin0["outer_inner"],0,1,"baseline O/I"),np.load(POST/"O_I-pairs.npy")),
 "inner_inner":unique_pairs(section_cross_rows(skin0["inner_inner"],1,1,"baseline I/I"),section_cross_rows(skin_new["inner_inner"],1,1,"new skin I/I"),np.load(POST/"I-pairs.npy")),
 "cap_inner":unique_pairs(skin0["cap_inner"],skin_new["cap_inner"]),
 "outer_outer":unique_pairs(skin_new["outer_outer"]),
 "cap_outer":unique_pairs(skin_new["cap_outer"]),
 "cap_cap":unique_pairs(skin_new["cap_cap"])}
fat={
 "outer_inner":unique_pairs(section_cross_rows(fat0["outer_inner"],0,1,"baseline I/D"),np.load(POST/"I_D-pairs.npy")),
 "outer_outer":unique_pairs(section_cross_rows(fat_new["outer_outer"],0,0,"new I/I"),np.load(POST/"I-pairs.npy")),
 "inner_inner":unique_pairs(section_cross_rows(fat0["inner_inner"],1,1,"baseline D/D"),section_cross_rows(fat_new["inner_inner"],1,1,"new D/D"),np.load(POST/"D-pairs.npy")),
 "cap_inner":unique_pairs(fat0["cap_inner"],fat_new["cap_inner"]),
 "cap_outer":unique_pairs(fat_new["cap_outer"]),
 "cap_cap":unique_pairs(fat_new["cap_cap"])}
need(len(skin["outer_outer"])==0 and len(skin["cap_outer"])==0 and len(skin["cap_cap"])==0 and
     len(fat["cap_outer"])==0 and len(fat["cap_cap"])==0,
     "new witness set contains a fixed O/O or unhandled cap category")
# I pairs from baseline I self, new direct audit, and both shell encodings.
I_pairs=unique_pairs(np.load(RUN/"I-self-pairs.npy"),np.load(POST/"I-pairs.npy"),
                     skin["inner_inner"],fat["outer_outer"])
D_pairs=unique_pairs(D_base,fat["inner_inner"])
cap_pairs=fat["cap_inner"]
skin_cap_pairs=skin["cap_inner"]

# Full exact O/D crossing audit; other pair lists are from the complete run-003
# closed-shell audits and are re-audited after the solve.
recO,_=pred.float32_surface_records(O.astype(np.float32),F)
recD,_=pred.float32_surface_records(D.astype(np.float32),F)
od=pred._audit_pair(recO,recD,same_surface=False)
OD_current=np.asarray(od["triangle_pairs"],dtype=np.int64).reshape(-1,2)
need(int(od["count"])==len(OD_current),"O/D baseline exact crossing count inconsistent")
OD=unique_pairs(OD_current,np.load(POST/"O_D-pairs.npy"))
np.save(OUT/"O-D-exact-pairs.npy",OD)
need(len(OD)<=10000,"unexpected O/D pair volume")
O_normals=np.cross(O[F[:,1]]-O[F[:,0]],O[F[:,2]]-O[F[:,0]])
face_norm_len=np.linalg.norm(O_normals,axis=1)
need(np.isfinite(face_norm_len).all() and np.all(face_norm_len>0),"degenerate outer face")
O_face_n=O_normals/face_norm_len[:,None]

# Build the exact witness face/node inventory, plus one face-neighbor ring.
witness_face_rows=set()
def add_faces(pairs, aoffset=0, boffset=0):
    for a,b in pairs:
        witness_face_rows.add(int(a)-aoffset)
        witness_face_rows.add(int(b)-boffset)
add_faces(skin["outer_inner"],0,0)
add_faces(fat["outer_inner"],0,0)
add_faces(I_pairs,0,0)
add_faces(OD,0,0)
for a,b in D_pairs:
    witness_face_rows.add(int(a)); witness_face_rows.add(int(b))
# Include skin cap/I interface faces and cap/D interface faces.
for ca,ib in skin_cap_pairs:
    ca,ib=int(ca),int(ib)
    if ca<2*nf: ca,ib=ib,ca
    need(ca>=2*nf and nf<=ib<2*nf,"skin cap/I pair row order invalid")
    witness_face_rows.add(ib-nf)
    captri=z["skin_shell_faces"][ca].astype(np.int64)
    for gv in captri:
        gv=int(gv)
        row=gv if 0<=gv<n else (gv-n if n<=gv<2*n else None)
        if row is not None:
            hits=np.flatnonzero(np.any(F==row,axis=1))
            if len(hits): witness_face_rows.add(int(hits[0]))
# Include cap/D interface faces by source D face and incident O boundary-face
for a,b in cap_pairs:
    ca,db=(int(a),int(b)) if int(a)>=2*nf else (int(b),int(a))
    drow=db-nf
    need(nf<=db<2*nf and ca>=2*nf,"cap/D pair face domains invalid")
    witness_face_rows.add(drow)
    captri=z["fat_shell_faces"][ca].astype(np.int64)
    for gv in captri:
        if n<=gv<2*n:
            row=int(gv-n)
            # Add an incident outer face for this cap endpoint when available.
            hits=np.flatnonzero(np.any(F==row,axis=1))
            if len(hits): witness_face_rows.add(int(hits[0]))
need(witness_face_rows and min(witness_face_rows)>=0 and max(witness_face_rows)<nf,"bad witness face set")
witness_nodes=set(map(int,F[np.asarray(sorted(witness_face_rows),dtype=np.int64)].reshape(-1)))
# One-ring expansion over the actual indexed outer topology.
neighbors=[set() for _ in range(n)]
for tri in F:
    a,b,c=map(int,tri)
    neighbors[a].update((b,c));neighbors[b].update((a,c));neighbors[c].update((a,b))
affected=set(witness_nodes)
for i in tuple(witness_nodes): affected.update(neighbors[i])
affected_rows=np.asarray(sorted(affected),dtype=np.int64)
# The compact solve keeps witness vertices plus two indexed graph rings free in
# both I and D. Outer O stays fixed. affected_rows remains the original one-ring
# set for provenance; free_rows is the expanded variable workset.
free_nodes=set(affected)
for row in tuple(affected):
    free_nodes.update(neighbors[row])
free_rows=np.asarray(sorted(free_nodes),dtype=np.int64)
need(len(free_rows)>=len(affected_rows) and set(affected_rows).issubset(set(free_rows)),
     "expanded free workset lost original affected rows")

# Area-weighted O normals, oriented by the already-frozen outward O face winding.
vertex_n=np.zeros((n,3),dtype=np.float64)
for corner in range(3):
    np.add.at(vertex_n,F[:,corner],O_normals)
vertex_len=np.linalg.norm(vertex_n,axis=1)
need(np.isfinite(vertex_len).all() and np.all(vertex_len[free_rows]>1e-14),
     "free workset vertex has undefined area-weighted outer normal")
vertex_n[free_rows]/=vertex_len[free_rows,None]

# Variable domain: all I and D nodes, in mm displacements from exact emitted-F32 baseline.
nd=6*n
def idx(layer,node,axis):
    node=int(node);axis=int(axis)
    if layer=="I": return 3*node+axis
    if layer=="D": return 3*n+3*node+axis
    if layer=="O": return None
    raise ValueError(layer)

geo_r=[];geo_c=[];geo_v=[];geo_b=[];geo_meta=[]
def add_ineq(layer_a,node_a,layer_b,node_b,normal,margin_m,metadata):
    """Add n.(Xa-Xb)>=margin, expressed as a sparse upper-bound row in mm."""
    na=np.asarray(normal,dtype=np.float64)
    need(na.shape==(3,) and np.isfinite(na).all() and abs(float(np.linalg.norm(na))-1.0)<1e-7,
         "constraint normal is not unit length")
    va=({"O":O,"I":I,"D":D}[layer_a][int(node_a)])
    vb=({"O":O,"I":I,"D":D}[layer_b][int(node_b)])
    gap=float(np.dot(na,va-vb))
    coeff=collections.defaultdict(float)
    if layer_a!="O":
        for k in range(3): coeff[idx(layer_a,node_a,k)]-=float(na[k])
    if layer_b!="O":
        for k in range(3): coeff[idx(layer_b,node_b,k)]+=float(na[k])
    row=len(geo_b)
    rhs=(gap-float(margin_m))*1000.0
    nz=[(c,v) for c,v in coeff.items() if v!=0.0]
    if not nz:
        if rhs < -1e-7:
            raise RuntimeError(f"fixed violated inequality: {metadata}, rhs_mm={rhs}")
        geo_meta.append({**metadata,"row":row,"status":"constant_satisfied","baseline_gap_m":gap,
                         "margin_m":float(margin_m),"rhs_mm":rhs})
        geo_b.append(rhs); return
    for c,v in nz: geo_r.append(row);geo_c.append(c);geo_v.append(v)
    geo_b.append(rhs)
    geo_meta.append({**metadata,"row":row,"status":"variable","baseline_gap_m":gap,
                     "margin_m":float(margin_m),"rhs_mm":rhs})

pair_margin_m=PAIR_MM*1e-3
# O/I first-entry-side constraints from complete shell crossing witnesses.
for pair in skin["outer_inner"]:
    a,b=map(int,pair); fa=a; fb=b
    need(0<=fa<nf and 0<=fb<nf,"bad O/I shell face rows")
    normal=O_face_n[fa]
    for ia,va in enumerate(F[fa]):
        for ib,vb in enumerate(F[fb]):
            add_ineq("O",int(va),"I",int(vb),normal,pair_margin_m,
                     {"kind":"O_I","face_a":fa,"face_b":fb,"corner_a":ia,"corner_b":ib})
# I/D outer-to-inner constraints.
for pair in fat["outer_inner"]:
    a,b=map(int,pair); fa=a; fb=b
    need(0<=fa<nf and 0<=fb<nf,"bad I/D shell face rows")
    normal=O_face_n[fa]
    for ia,va in enumerate(F[fa]):
        for ib,vb in enumerate(F[fb]):
            add_ineq("I",int(va),"D",int(vb),normal,pair_margin_m,
                     {"kind":"I_D","face_a":fa,"face_b":fb,"corner_a":ia,"corner_b":ib})
# Direct O/D constraints not represented in the two closed adjacent-layer shells.
for fa,fb in OD:
    fa=int(fa);fb=int(fb);normal=O_face_n[fa]
    for ia,va in enumerate(F[fa]):
        for ib,vb in enumerate(F[fb]):
            add_ineq("O",int(va),"D",int(vb),normal,pair_margin_m,
                     {"kind":"O_D","face_a":fa,"face_b":fb,"corner_a":ia,"corner_b":ib})

# Same-layer I/I witnesses from the baseline and rejected candidate audits.
identity=sparse.eye(n,format="csr",dtype=np.float64)
candidate_prior=np.load(PRIOR/"joint-layer-candidate.npz",allow_pickle=False)
Iprior=np.asarray(candidate_prior["inner_points_f32"],dtype=np.float64)
ii_separator_details=[]
for fa,fb in I_pairs:
    fa=int(fa);fb=int(fb);idsa=F[fa];idsb=F[fb]
    selected=sep.select_baseline_separator(
        O[idsa],O[idsb],idsa,idsb,identity[idsa],identity[idsb],
        candidate_a=Iprior[idsa],candidate_b=Iprior[idsb])
    ii_separator_details.append({"face_a":fa,"face_b":fb,"axis_source":selected.axis_source,
                                 "baseline_min_gap_m":selected.baseline_min_gap_m,"margin_m":selected.margin_m,
                                 "candidate_violation_m":selected.candidate_violation_m,
                                 "shared_vertex_ids":list(selected.shared_vertex_ids),"normal":selected.normal.tolist()})
    for ia,va in enumerate(idsb):
        for ib,vb in enumerate(idsa):
            add_ineq("I",int(va),"I",int(vb),selected.normal,selected.margin_m,
                     {"kind":"I_I_baseline_SAT","face_a":fa,"face_b":fb,"corner_a":ia,"corner_b":ib})

# Same-layer D/D witnesses: choose the frozen helper's baseline SAT separator
# from the corresponding O-layer triangle pair, then preserve its margin.
identity=sparse.eye(n,format="csr",dtype=np.float64)
dd_separator_details=[]
for fa,fb in D_pairs:
    fa=int(fa);fb=int(fb);idsa=F[fa];idsb=F[fb]
    wa=identity[idsa];wb=identity[idsb]
    selected=sep.select_baseline_separator(
        O[idsa],O[idsb],idsa,idsb,wa,wb,
        candidate_a=D[idsa],candidate_b=D[idsb])
    dd_separator_details.append({"face_a":fa,"face_b":fb,"axis_source":selected.axis_source,
                                 "baseline_min_gap_m":selected.baseline_min_gap_m,
                                 "margin_m":selected.margin_m,
                                 "candidate_violation_m":selected.candidate_violation_m,
                                 "shared_vertex_ids":list(selected.shared_vertex_ids),
                                 "normal":selected.normal.tolist()})
    for ia,va in enumerate(idsb):
        for ib,vb in enumerate(idsa):
            add_ineq("D",int(va),"D",int(vb),selected.normal,selected.margin_m,
                     {"kind":"D_D_baseline_SAT","face_a":fa,"face_b":fb,
                      "corner_a":ia,"corner_b":ib,"axis_source":selected.axis_source})

# Skin closure cap/I witnesses: baseline-valid SAT axes over the actual shell faces.
skin_shell=np.asarray(z["skin_shell_faces"],dtype=np.int64)
layer_points=np.vstack((O,I));candidate_layer=np.vstack((O,Iprior))
layer_identity=sparse.eye(2*n,format="csr",dtype=np.float64)
skin_cap_details=[]
for ca,ib in skin_cap_pairs:
    ca,ib=int(ca),int(ib)
    if ca<2*nf: ca,ib=ib,ca
    need(ca>=2*nf and nf<=ib<2*nf,"skin cap/I pair ordering invalid")
    cap_row=ca-2*nf;inner_row=ib-nf
    cap_ids=skin_shell[ca].astype(np.int64);inner_ids=(n+F[inner_row]).astype(np.int64)
    selected=sep.select_baseline_separator(
        layer_points[cap_ids],layer_points[inner_ids],cap_ids,inner_ids,
        layer_identity[cap_ids],layer_identity[inner_ids],
        candidate_a=candidate_layer[cap_ids],candidate_b=candidate_layer[inner_ids])
    cap_points=[]
    for gv in cap_ids:
        if int(gv)<n: cap_points.append(("O",int(gv)))
        elif int(gv)<2*n: cap_points.append(("I",int(gv)-n))
        else: raise RuntimeError(f"unexpected skin cap vertex id {gv}")
    for ia,(la,ra) in enumerate(cap_points):
        for ibc,gb in enumerate(F[inner_row]):
            add_ineq(la,ra,"I",int(gb),selected.normal,selected.margin_m,
                     {"kind":"skin_cap_I_baseline_SAT","cap_face":cap_row,"I_face":inner_row,
                      "cap_corner":ia,"I_corner":ibc,"axis_source":selected.axis_source})
    skin_cap_details.append({"cap_face":cap_row,"I_face":inner_row,"axis_source":selected.axis_source,
                             "margin_m":selected.margin_m,"baseline_min_gap_m":selected.baseline_min_gap_m,
                             "candidate_violation_m":selected.candidate_violation_m,
                             "normal":selected.normal.tolist(),"shared_vertex_ids":list(selected.shared_vertex_ids)})

# Paired open-boundary cap witnesses: use the current D face plane as a
# fixed separating axis, oriented outward by its corresponding O face normal.
# Every cap vertex must stay on/outside that deep-face plane; shared D vertices
# are retained as exact zero rows, not omitted.
cap_details=[]
fat_shell=np.asarray(z["fat_shell_faces"],dtype=np.int64)
for ca,db in cap_pairs:
    ca,db=int(ca),int(db)
    if ca<2*nf: ca,db=db,ca
    need(ca>=2*nf and nf<=db<2*nf,"cap/D pair row order invalid")
    cap_face=ca-2*nf; dface=db-nf
    drows=F[dface]
    dn=np.cross(D[drows[1]]-D[drows[0]],D[drows[2]]-D[drows[0]])
    dl=float(np.linalg.norm(dn))
    need(np.isfinite(dl) and dl>0,"degenerate baseline D face at cap witness")
    dn/=dl
    if float(np.dot(dn,O_face_n[dface]))<0: dn=-dn
    alignment=float(np.dot(dn,O_face_n[dface]))
    need(alignment>0.25,f"cap/D plane is not outward-aligned: {alignment}")
    capg=fat_shell[ca]
    cap_local=[]
    for gv in capg:
        gv=int(gv)
        if n<=gv<2*n: cap_local.append(("I",gv-n))
        elif 2*n<=gv<3*n: cap_local.append(("D",gv-2*n))
        else: raise RuntimeError(f"unexpected cap vertex layer id {gv}")
    for i,(la,ra) in enumerate(cap_local):
        for j,rb in enumerate(drows):
            add_ineq(la,int(ra),"D",int(rb),dn,0.0,
                     {"kind":"paired_cap_D_plane","cap_face":cap_face,"d_face":dface,
                      "cap_corner":i,"d_corner":j,"normal_alignment":alignment})
    cap_details.append({"cap_face":cap_face,"d_face":dface,"normal":dn.tolist(),
                        "normal_alignment":alignment,
                        "cap_vertices":[[la,int(r)] for la,r in cap_local],"d_vertices":drows.tolist()})

# Positive construction ordering on the complete free workset (two rings).
for row in free_rows:
    normal=vertex_n[row]
    add_ineq("O",int(row),"I",int(row),normal,COLUMN_MM*1e-3,
             {"kind":"column_O_I_50um","node":int(row)})
    add_ineq("I",int(row),"D",int(row),normal,COLUMN_MM*1e-3,
             {"kind":"column_I_D_50um","node":int(row)})

Ageo=sparse.coo_matrix((np.asarray(geo_v,dtype=np.float64),
                        (np.asarray(geo_r,dtype=np.int64),np.asarray(geo_c,dtype=np.int64))),
                       shape=(len(geo_b),nd)).tocsr()
Ageo.sum_duplicates();Ageo.eliminate_zeros();Ageo.sort_indices()
bgeo=np.asarray(geo_b,dtype=np.float64)
need(Ageo.shape[0]==len(bgeo)==len(geo_meta),"geometric row count mismatch")

# Sufficient P1 deformation-gradient bounds on every baseline I and D face.
# The exact guard produces six tangent-gradient components per face in m^-1;
# LP variables are mm, hence the 1e-3 conversion. Fixed columns are eliminated
# exactly; rows with no free gradient coefficient are identically 0 <= limit.
def require_rank2(P, label):
    Eface=np.stack((P[F[:,1]]-P[F[:,0]],P[F[:,2]]-P[F[:,0]]),axis=2)
    sv=np.linalg.svd(Eface,compute_uv=False)
    ratio=sv[:,1]/sv[:,0]
    bad=np.flatnonzero(~np.isfinite(ratio)|(ratio<=64*np.finfo(np.float64).eps))
    need(not len(bad),f"{label} has rank-deficient face {int(bad[0]) if len(bad) else -1}")
    return {"minimum_singular_ratio":float(ratio.min()),"maximum_condition":float((1.0/ratio).max()),
            "worst_face_row":int(np.argmin(ratio))}
rankI=require_rank2(I,"I baseline");rankD=require_rank2(D,"D baseline")
gradI,grad_limit,gradI_diag=grad.build_face_gradient_guard(I,F,max_frobenius=0.4)
gradD,gradD_limit,gradD_diag=grad.build_face_gradient_guard(D,F,max_frobenius=0.4)
need(abs(grad_limit-gradD_limit)<1e-15 and abs(grad_limit-0.4/np.sqrt(6))<1e-15,
     "unexpected face gradient guard limit")
Ggrad=sparse.block_diag((gradI*1e-3,gradD*1e-3),format="csr")
Agrad_all=sparse.vstack((Ggrad,-Ggrad),format="csr")
need(Agrad_all.shape==(24*nf,nd) and np.isfinite(Agrad_all.data).all(),
     "full gradient guard shape/data invalid")

# Map each free I/D coordinate to its original full displacement column. All
# other I/D coordinates remain exactly at the frozen baseline; O has no variables.
free_I_dof=(3*free_rows[:,None]+np.arange(3,dtype=np.int64)).reshape(-1)
free_D_dof=3*n+free_I_dof
free_global_dof=np.concatenate((free_I_dof,free_D_dof)).astype(np.int64)
nd_free=len(free_global_dof)
need(nd_free==6*len(free_rows) and len(np.unique(free_global_dof))==nd_free,
     "free displacement column map malformed")
global_to_free=np.full(nd,dtype=np.int64,fill_value=-1)
global_to_free[free_global_dof]=np.arange(nd_free,dtype=np.int64)

# Keep every geometric inequality, including rows that become fixed-only. A
# fixed-only violated row is a hard stop, never silently dropped.
Ageo_free=Ageo[:,free_global_dof].tocsr()
Ageo_free.eliminate_zeros();Ageo_free.sort_indices()
geometric_row_nnz=np.diff(Ageo_free.indptr)
fixed_geom_rows=np.flatnonzero(geometric_row_nnz==0)
fixed_geom_bad=fixed_geom_rows[bgeo[fixed_geom_rows] < -1e-7]
if len(fixed_geom_bad):
    (OUT/"fixed-geometric-row-rejection.json").write_text(json.dumps({
        "status":"REJECT_FIXED_VIOLATED_GEOMETRIC_ROW",
        "rows":[{"row":int(r),"rhs_mm":float(bgeo[r]),"metadata":geo_meta[int(r)]}
                for r in fixed_geom_bad],
        "free_rows":free_rows.tolist(),
    },sort_keys=True,indent=2)+"\n")
    raise SystemExit("fixed-only geometric inequality is violated; not solving")

# Retain all gradient rows that can change. The exact helper orders six rows per
# face, then the same rows with negative sign, for each of the two layers.
Agrad_all_free=Agrad_all[:,free_global_dof].tocsr()
grad_keep=np.diff(Agrad_all_free.indptr)>0
faces_incident_free=np.any(np.isin(F,free_rows),axis=1)
expected_grad_keep=np.tile(np.repeat(faces_incident_free,6),4)
need(np.array_equal(grad_keep,expected_grad_keep),
     "zero-gradient row classification disagrees with incident-face topology")
Agrad_free=Agrad_all_free[grad_keep].tocsr()
Agrad_free.eliminate_zeros();Agrad_free.sort_indices()
bgrad=np.full(Agrad_free.shape[0],grad_limit,dtype=np.float64)

gradient_guard_receipt={"skin_faces":gradI_diag,"fat_faces":gradD_diag,
    "skin_rank2":rankI,"fat_rank2":rankD,"componentwise_limit":grad_limit,
    "max_frobenius":0.4,"rows_per_layer_full":int(gradI.shape[0]),
    "combined_signed_rows_full":int(Agrad_all.shape[0]),
    "combined_signed_rows_retained":int(Agrad_free.shape[0]),
    "combined_signed_rows_dropped_zero":int(Agrad_all.shape[0]-Agrad_free.shape[0]),
    "incident_free_faces_per_layer":int(np.count_nonzero(faces_incident_free)),
    "units":"unknowns_mm; guard matrix includes m-to-mm 1e-3 scale",
    "face_count_per_layer":int(nf)}

# Full indexed mesh edge graph. Only edges with at least one free endpoint can
# contribute to the L1 displacement-gradient objective; fixed-fixed edges are
# exact zeros and their auxiliary variables/rows are omitted.
edges=np.unique(np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0)
need(edges.ndim==2 and edges.shape[1]==2,"edge graph malformed")
edge_touch=np.any(np.isin(edges,free_rows),axis=1)
edges_touch=edges[edge_touch]
er=np.repeat(np.arange(len(edges),dtype=np.int64),2)
ec=edges.reshape(-1)
ev=np.tile(np.asarray([1.0,-1.0]),len(edges))
B=sparse.coo_matrix((ev,(er,ec)),shape=(len(edges),n)).tocsr()
B_touch=B[edge_touch][:,free_rows].tocsr()
B3=sparse.kron(B_touch,sparse.eye(3,format="csr"),format="csr")
H=sparse.block_diag((B3,B3),format="csr")
qcount=H.shape[0]
total=2*nd_free+qcount
Id=sparse.eye(nd_free,format="csr")
Zabs=sparse.csr_matrix((nd_free,qcount))
Aabs=sparse.vstack((sparse.hstack((Id,-Id,Zabs),format="csr"),
                    sparse.hstack((-Id,-Id,Zabs),format="csr")),format="csr")
m=H.shape[0]
Zs=sparse.csr_matrix((m,nd_free))
Iq=sparse.eye(qcount,format="csr")
Asmooth=sparse.vstack((sparse.hstack((H,Zs,-Iq),format="csr"),
                       sparse.hstack((-H,Zs,-Iq),format="csr")),format="csr")
Ageom_full=sparse.hstack((Ageo_free,sparse.csr_matrix((Ageo_free.shape[0],nd_free+qcount))),format="csr")
Agrad_full=sparse.hstack((Agrad_free,sparse.csr_matrix((Agrad_free.shape[0],nd_free+qcount))),format="csr")
Aub=sparse.vstack((Aabs,Asmooth,Ageom_full,Agrad_full),format="csr")
bub=np.concatenate((np.zeros(Aabs.shape[0]+Asmooth.shape[0]),bgeo,bgrad))
c=np.concatenate((np.zeros(nd_free),np.ones(nd_free),SMOOTH_WEIGHT*np.ones(qcount)))
bounds=[(None,None)]*nd_free+[(0.0,None)]*nd_free+[(0.0,None)]*qcount

previous_counts=json.loads((PREVIOUS_OUT/"model-preflight.json").read_text())["counts"]
full_qcount=6*len(edges)
full_model_counts={
    "LP_variables":int(2*nd+full_qcount),
    "LP_rows":int(2*nd+12*len(edges)+Ageo.shape[0]+Agrad_all.shape[0]),
    "LP_nnz":int(4*nd+36*len(edges)+Ageo.nnz+Agrad_all.nnz),
}
need(all(full_model_counts[k]==int(previous_counts[k]) for k in full_model_counts),
     f"reconstructed full model counts differ from frozen attempt: {full_model_counts} vs {previous_counts}")
reduced_model_counts={
    "LP_variables":int(total),"LP_rows":int(Aub.shape[0]),"LP_nnz":int(Aub.nnz),
    "displacement_variables":int(nd_free),"abs_aux_variables":int(nd_free),
    "smooth_aux_variables":int(qcount),"free_nodes_per_layer":int(len(free_rows)),
    "two_ring_free_nodes":int(len(free_rows)),"original_one_ring_nodes":int(len(affected_rows)),
    "smooth_edges_retained":int(len(edges_touch)),"smooth_edges_total":int(len(edges)),
    "geometric_rows_retained":int(Ageo_free.shape[0]),
    "geometric_rows_fixed_only":int(len(fixed_geom_rows)),
    "gradient_rows_retained":int(Agrad_free.shape[0]),
    "gradient_rows_dropped_zero":int(Agrad_all.shape[0]-Agrad_free.shape[0]),
    "gradient_faces_incident_to_free_nodes_per_layer":int(np.count_nonzero(faces_incident_free))}
reduction_receipt={
    "full_attempt_004_counts":full_model_counts,
    "reduced_model_counts":reduced_model_counts,
    "free_domain":"all geometric witness vertices plus two indexed edge-neighbor rings, shared by I and D",
    "outer_layer":"fixed; not a variable block",
    "elimination":"I/D displacement columns outside free domain fixed exactly at baseline; all geometric inequality rows retained, fixed-only violated rows reject before solve; zero gradient rows dropped; fixed-fixed smoothness edges omitted",
    "free_rows_sha256":hashlib.sha256(np.asarray(free_rows,dtype="<i8").tobytes()).hexdigest(),
    "free_global_dof_sha256":hashlib.sha256(np.asarray(free_global_dof,dtype="<i8").tobytes()).hexdigest(),
    "fixed_geometric_row_count":int(len(fixed_geom_rows)),
    "fixed_geometric_violation_count":int(len(fixed_geom_bad))}
need(Aub.shape==(len(bub),total) and np.isfinite(Aub.data).all() and np.isfinite(bub).all(),
     "LP sparse model shape/data invalid")

# Save the exact constructed model/witness inventory before optimization.
sparse.save_npz(OUT/"joint-lp-A-ub.npz",Aub,compressed=True)
np.savez_compressed(OUT/"joint-lp-inputs.npz",
    c=c,b_ub=bub,affected_rows=affected_rows,free_rows=free_rows,
    free_global_dof=free_global_dof,gradient_rows_kept=np.flatnonzero(grad_keep),
    gradient_rows_dropped=np.flatnonzero(~grad_keep),outer_face_normals=O_face_n,
    geometric_A_ub=Ageo,geometric_b_ub=bgeo,
    O_I_pairs=skin["outer_inner"][:,[0,1]] if len(skin["outer_inner"]) else np.empty((0,2),dtype=np.int64),
    I_D_pairs=fat["outer_inner"][:,[0,1]] if len(fat["outer_inner"]) else np.empty((0,2),dtype=np.int64),
    O_D_pairs=OD,D_D_pairs=D_pairs,cap_D_pairs=cap_pairs)
(OUT/"geometric-witness-rows.json").write_text(json.dumps(geo_meta,sort_keys=True,indent=2)+"\n")
(OUT/"model-preflight.json").write_text(json.dumps({
    "input_sha256":INPUT_SHA,"separation_helper_sha256":sha(SEP_PATH),"predicate_sha256":sha(PRED),
    "attempt002_report_sha256":sha(POST/"report.json"),"gradient_guard_sha256":sha(GRAD_PATH),
    "attempt002_witness_sha256":{p.name:sha(p) for p in sorted(POST.glob("*.npy"))},
    "run003_report_sha256":sha(RUN/"report.json"),
    "counts":{"nodes":n,"faces":nf,"O_I_pairs":len(skin["outer_inner"]),
        "I_D_pairs":len(fat["outer_inner"]),"O_D_pairs":len(OD),"D_D_pairs":len(D_pairs),
        "cap_D_pairs":len(cap_pairs),"affected_one_ring_nodes":len(affected_rows),
        "free_two_ring_nodes":len(free_rows),
        "geometric_rows":Ageo.shape[0],"geometric_nnz":Ageo.nnz,"mesh_edges":len(edges),
        "smooth_edges_retained":len(edges_touch),"gradient_rows_retained":Agrad_free.shape[0],
        "gradient_rows_dropped_zero":int(Agrad_all.shape[0]-Agrad_free.shape[0]),
        "LP_variables":total,"LP_rows":Aub.shape[0],"LP_nnz":Aub.nnz},
    "reduction":reduction_receipt,
    "choices":{"pair_margin_m":pair_margin_m,"column_ordering_m":COLUMN_MM*1e-3,
        "smoothness_weight_per_mm":SMOOTH_WEIGHT,"smoothness_graph":"all unique edges of the indexed outer topology",
        "displacement_l1":"unit weight per I/D node coordinate, mm units",
        "same_layer_D_separator":"human_skin_fat_surface_separation.select_baseline_separator on O faces",
        "cap_D_separator":"fixed baseline-D-face plane oriented outward by O-face normal; zero margin; all 9 pairs retained"},
    "I_I_separators":ii_separator_details,"D_D_separators":dd_separator_details,
    "skin_cap_I_separators":skin_cap_details,"cap_D_separators":cap_details,
    "face_gradient_guard":gradient_guard_receipt,
},sort_keys=True,indent=2)+"\n")

t0=time.monotonic()
res=linprog(c,A_ub=Aub,b_ub=bub,bounds=bounds,method="highs",
            options={"time_limit":TIME_LIMIT_S,"presolve":True})
solve_s=time.monotonic()-t0
report={"status":"LP_SOLVED_NO_GEOMETRY_ADMISSION","lp":{"success":bool(res.success),"status":int(res.status),
    "message":str(res.message),"objective":None if res.fun is None else float(res.fun),
    "iterations":None if not hasattr(res,"nit") else int(res.nit),"wall_seconds":solve_s,
    "time_limit_seconds":TIME_LIMIT_S,"max_inequality_violation_mm":None},
    "model":json.loads((OUT/"model-preflight.json").read_text()),
    "source":{"run003_layer_candidate_sha256":INPUT_SHA,"separation_helper_sha256":sha(SEP_PATH),
              "exact_predicate_sha256":sha(PRED)},"post_audit":None}
if res.x is None:
    report["status"]="LP_NO_PRIMAL_SOLUTION"
    (OUT/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps(report["lp"],sort_keys=True))
    raise SystemExit(0)

x_compact=np.asarray(res.x[:nd_free],dtype=np.float64)
x=np.zeros(nd,dtype=np.float64)
x[free_global_dof]=x_compact
dI=x[:3*n].reshape(n,3)*1e-3
dD=x[3*n:].reshape(n,3)*1e-3
I64=I+dI;D64=D+dD
I32=I64.astype(np.float32);D32=D64.astype(np.float32);O32=O.astype(np.float32)
np.savez_compressed(OUT/"joint-layer-candidate.npz",
    outer_points_f64=O,inner_points_f64=I64,deep_points_f64=D64,
    outer_points_f32=O32,inner_points_f32=I32,deep_points_f32=D32,
    outer_faces=z["outer_faces"],inner_faces=z["inner_faces"],deep_faces=z["deep_faces"],
    skin_shell_faces=z["skin_shell_faces"],fat_shell_faces=z["fat_shell_faces"],
    active_vertex_rows=free_rows.astype(np.int32),original_affected_one_ring_rows=affected_rows.astype(np.int32),
    free_global_dof=free_global_dof,global_displacement_mm=x, input_node_kind=z["node_kind"],
    input_node_source_row=z["node_source_row"],displacement_I_m=dI,displacement_D_m=dD,
    LP_solution_compact=res.x,LP_constraint_slack_mm=(bub-Aub@res.x),
    LP_success=np.asarray([bool(res.success)]),LP_status=np.asarray([int(res.status)]),
    LP_message=np.asarray([str(res.message)]),LP_iterations=np.asarray([int(getattr(res,"nit",-1))]),
    LP_objective=np.asarray([np.nan if res.fun is None else float(res.fun)]))
maxv=float(np.max(Aub@res.x-bub,initial=-np.inf))
report["lp"]["max_inequality_violation_mm"]=maxv
report["displacement"]={
    "I_max_mm":float(np.linalg.norm(dI,axis=1).max(initial=0)*1000),
    "D_max_mm":float(np.linalg.norm(dD,axis=1).max(initial=0)*1000),
    "I_mean_affected_mm":float(np.linalg.norm(dI[affected_rows],axis=1).mean()*1000),
    "D_mean_affected_mm":float(np.linalg.norm(dD[affected_rows],axis=1).mean()*1000),
    "nonzero_I_nodes":int(np.count_nonzero(np.linalg.norm(dI,axis=1)>1e-12)),
    "nonzero_D_nodes":int(np.count_nonzero(np.linalg.norm(dD,axis=1)>1e-12)),
}
# Verify exact post-F32 50um per-column inequalities.
out_gaps=np.einsum("ij,ij->i",vertex_n[free_rows],O32[free_rows].astype(float)-I32[free_rows].astype(float))
fat_gaps=np.einsum("ij,ij->i",vertex_n[free_rows],I32[free_rows].astype(float)-D32[free_rows].astype(float))
report["ordering_f32"]={"checked_free_workset_count":int(len(free_rows)),"original_affected_count":int(len(affected_rows)),
    "OI_min_mm":float(out_gaps.min(initial=np.inf)*1000),"ID_min_mm":float(fat_gaps.min(initial=np.inf)*1000),
    "OI_below_50um":int(np.count_nonzero(out_gaps<COLUMN_MM*1e-3-1e-10)),
    "ID_below_50um":int(np.count_nonzero(fat_gaps<COLUMN_MM*1e-3-1e-10))}
# Exact full self/cross audits using the same exact owner.
def exact_self(X,faces):
    rec,_=pred.float32_surface_records(np.asarray(X,dtype=np.float32),np.asarray(faces,dtype=np.int64),source_seam_groups=())
    out=pred._audit_self_pairs_touching_faces(rec,set(range(len(faces))))
    pairs=np.asarray(out["triangle_pairs"],dtype=np.int64).reshape(-1,2)
    return {"count":len(pairs),"complete":bool(out.get("complete_surface_audit",False)),"pairs":pairs}
def exact_cross(A,FA,B,FB):
    ra,_=pred.float32_surface_records(np.asarray(A,dtype=np.float32),np.asarray(FA,dtype=np.int64),source_seam_groups=())
    rb,_=pred.float32_surface_records(np.asarray(B,dtype=np.float32),np.asarray(FB,dtype=np.int64),source_seam_groups=())
    out=pred._audit_pair(ra,rb,same_surface=False)
    return {"count":int(out["count"]),"pairs":np.asarray(out["triangle_pairs"],dtype=np.int64).reshape(-1,2),
            "aabb_candidate_pairs":int(out["aabb_candidate_pairs"])}
Fout=np.asarray(z["outer_faces"],dtype=np.int64)
skinF=np.asarray(z["skin_shell_faces"],dtype=np.int64);fatF=np.asarray(z["fat_shell_faces"],dtype=np.int64)
all_skin=np.vstack((O32,I32));all_fat=np.vstack((I32,D32))
post={}
for label,X,Faces in (("O",O32,Fout),("I",I32,Fout),("D",D32,Fout),
                      ("skin_shell",all_skin,skinF),("fat_shell",all_fat,fatF)):
    t=time.monotonic();r=exact_self(X,Faces);np.save(OUT/f"{label}-self-pairs.npy",r["pairs"])
    post[label]={"pairs":r["count"],"complete":r["complete"],"seconds":time.monotonic()-t}
for label,A,FA,B,FB in (("O_I",O32,Fout,I32,Fout),
                        ("I_D",I32,Fout,D32,Fout),
                        ("O_D",O32,Fout,D32,Fout)):
    t=time.monotonic();r=exact_cross(A,FA,B,FB);np.save(OUT/f"{label}-cross-pairs.npy",r["pairs"])
    post[label]={k:v for k,v in r.items() if k!="pairs"};post[label]["seconds"]=time.monotonic()-t
report["post_audit"]=post
report["status"]="CONSTRUCTION_ONLY"
report["geometry_gates"]={
    "all_individual_and_closed_shell_exact_self_audits_zero":all(post[k]["pairs"]==0 for k in ("O","I","D","skin_shell","fat_shell")),
    "all_pairwise_layer_cross_audits_zero":all(post[k]["count"]==0 for k in ("O_I","I_D","O_D")),
    "affected_column_f32_ordering_50um":report["ordering_f32"]["OI_below_50um"]==0 and report["ordering_f32"]["ID_below_50um"]==0,
    "LP_full_model_feasible":bool(res.success) and maxv<=1e-6,
}
report["geometry_admitted"]=bool(all(report["geometry_gates"].values()))
report["limits"]=[
 "Static inferred layer construction only; hskin/hfat fields are not measured anatomy.",
 "The LP preserves one frozen SAT axis per witnessed pair; exact full audits are mandatory and are reported separately.",
 "No tetrahedral remesh, material-parent map, mass transfer, native simulation, or physical admission was performed."
]
report["source"]["runner_sha256"]=sha(pathlib.Path(__file__))
report["wall_seconds_total"]=time.monotonic()-start
(OUT/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
print(json.dumps({"status":report["status"],"lp":report["lp"],"displacement":report["displacement"],
                  "ordering_f32":report["ordering_f32"],"geometry_gates":report["geometry_gates"],
                  "post_audit":post,"wall_seconds_total":report["wall_seconds_total"]},sort_keys=True))

