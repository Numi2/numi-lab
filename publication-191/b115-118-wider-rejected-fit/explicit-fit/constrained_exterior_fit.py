from __future__ import annotations
import hashlib, importlib.util, json, math, sys, time, traceback
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

S=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638")
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
C=S/"partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz"
A=S/"partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/audit-159/report.json"
BASE_TOPO=S/"per-bank-workers-20261027/b115-118/attempt-006-exact-endpoint-identity/patch-trial-001"
WIDE_TOPO=S/"per-bank-workers-20261027/b115-118/attempt-006-exact-endpoint-identity/wider-fit-trial-001/topology"
OUT=S/"per-bank-workers-20261027/b115-118/attempt-006-exact-endpoint-identity/wider-fit-trial-001/explicit-fit"
AUDIT_TOOL=ROOT/"audit_skin_bone_surfaces_159.py"
HELPER=Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
EXACT=E/"collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py"
BANKDIR=E/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
FIT_MM=0.05
SCALE_M=FIT_MM/1000.0
AREA_RATIO_MIN=1e-6
CAND_PIN="7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af"
BASE_PATCH_PIN="840569c103e174269e9461d170d5cbccfb3f664da262c5798e027acacf8e617f"
WIDE_PATCH_PIN="67c90a682da5213bb7f26d166565b1a7e36d502dff504ab2ffd7ebf734ed14e4"

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def loadmod(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def frac(x):return Fraction.from_float(float(x))
def exact_tri(tri):return tuple(tuple(frac(x) for x in p) for p in np.asarray(tri).reshape(3,3))
def barycentric(p,tri,exact):
    n=exact._cross(exact._sub(tri[1],tri[0]),exact._sub(tri[2],tri[0]))
    axes=exact._axes(n)
    a,b,c,q=[tuple(v[k] for k in axes) for v in (tri[0],tri[1],tri[2],p)]
    den=exact._orient2(a,b,c)
    if den==0:raise ValueError("degenerate exact triangle")
    return (exact._orient2(b,c,q)/den,exact._orient2(c,a,q)/den,exact._orient2(a,b,q)/den)

def feature_for(bary,vertex_ids,vertex_keys,side):
    zeros=[i for i,w in enumerate(bary) if w==0]
    if len(zeros)>=2:
        i=next(i for i,w in enumerate(bary) if w==1)
        return ("skin-vertex",int(vertex_ids[i])) if side=="skin" else ("bone-vertex",115,118,vertex_keys[i])
    if len(zeros)==1:
        i,j=[x for x in range(3) if x!=zeros[0]]
        if side=="skin":
            vi,vj=int(vertex_ids[i]),int(vertex_ids[j]);lo,hi=sorted((vi,vj))
            return ("skin-edge",lo,hi,bary[j] if vj==hi else bary[i])
        ki,kj=vertex_keys[i],vertex_keys[j];lo,hi=sorted((ki,kj))
        return ("bone-emitted-edge",115,118,lo,hi,bary[j] if kj==hi else bary[i])
    return None

def closest_point_triangle(p,a,b,c):
    ab=b-a;ac=c-a;ap=p-a
    d1=float(ab@ap);d2=float(ac@ap)
    if d1<=0 and d2<=0:return a.copy()
    bp=p-b;d3=float(ab@bp);d4=float(ac@bp)
    if d3>=0 and d4<=d3:return b.copy()
    vc=d1*d4-d3*d2
    if vc<=0 and d1>=0 and d3<=0:return a+(d1/(d1-d3))*ab
    cp=p-c;d5=float(ab@cp);d6=float(ac@cp)
    if d6>=0 and d5<=d6:return c.copy()
    vb=d5*d2-d1*d6
    if vb<=0 and d2>=0 and d6<=0:return a+(d2/(d2-d6))*ac
    va=d3*d6-d5*d4
    if va<=0 and (d4-d3)>=0 and (d5-d6)>=0:return b+((d4-d3)/((d4-d3)+(d5-d6)))*(c-b)
    denom=1.0/(va+vb+vc);v=vb*denom;w=vc*denom
    return a+ab*v+ac*w

def exact_records_mesh(points,faces,face_ids,vertex_ids,exact):
    pts=[tuple(frac(x) for x in p) for p in np.asarray(points).reshape(-1,3)]
    out=[]
    for tri_id,face in zip(face_ids,faces):
        ids=tuple(map(int,face));tri=tuple(pts[i] for i in ids)
        n=exact._cross(exact._sub(tri[1],tri[0]),exact._sub(tri[2],tri[0]))
        if not any(n):raise ValueError(f"exact degenerate triangle in records: {tri_id}")
        lo=tuple(min(p[k] for p in tri) for k in range(3))
        hi=tuple(max(p[k] for p in tri) for k in range(3))
        out.append((tri,lo,hi,int(tri_id),tuple(int(vertex_ids[i]) for i in ids)))
    return out

started=time.monotonic()
report={"schema":"numi.human.bank115_118.wide-constrained-exterior-fit.v1","status":"STARTED",
        "candidate_path":str(C),"candidate_sha256":None,
        "wide_topology_path":str(WIDE_TOPO/"patch-candidate.npz"),"wide_topology_sha256":None,
        "wide_topology_report_path":str(WIDE_TOPO/"report.json"),"wide_topology_report_sha256":None,
        "base_topology_report_path":str(BASE_TOPO/"report.json"),"base_topology_report_sha256":None,
        "output_patch_path":str(OUT/"private-constrained-fit-patch.npz"),
        "fit_target_mm":FIT_MM,"fit_method":"SLSQP constrained displacement objective with explicit per-child orientation/area inequalities and bone-side halfspaces",
        "geometry_or_native_admission":False}
try:
    if sha(C)!=CAND_PIN:raise RuntimeError("candidate pin changed")
    if sha(WIDE_TOPO/"patch-candidate.npz")!=WIDE_PATCH_PIN:raise RuntimeError("wide topology patch pin changed")
    if sha(BASE_TOPO/"patch-candidate.npz")!=BASE_PATCH_PIN:raise RuntimeError("base topology patch pin changed")
    report.update({"candidate_sha256":sha(C),"wide_topology_sha256":sha(WIDE_TOPO/"patch-candidate.npz"),
                   "wide_topology_report_sha256":sha(WIDE_TOPO/"report.json"),
                   "base_topology_report_sha256":sha(BASE_TOPO/"report.json")})
    audit_tool=loadmod("wide_fit_audit_owner",AUDIT_TOOL)
    helper=loadmod("wide_fit_bank_helper",HELPER)
    exact=audit_tool.exact_module()
    with np.load(C,allow_pickle=False) as z:Z={k:z[k].copy() for k in z.files}
    with np.load(WIDE_TOPO/"patch-candidate.npz",allow_pickle=False) as z:Q={k:z[k].copy() for k in z.files}
    with np.load(BASE_TOPO/"patch-candidate.npz",allow_pickle=False) as z:QB={k:z[k].copy() for k in z.files}
    P32=np.asarray(Z["points_world_m"],dtype=np.float32)
    F=np.asarray(Z["source_O_faces"],dtype=np.int64)
    Ppatch=np.asarray(Q["points_world_m_f32"],dtype=np.float32)
    PF=np.asarray(Q["patch_faces"],dtype=np.int64)
    src_rows=np.asarray(Q["source_candidate_node_rows"],dtype=np.int64)
    endpoint_roots=np.asarray(Q["exact_endpoint_component_ids"],dtype=np.int64)
    boundary_edges=np.asarray(Q["patch_outer_boundary_edges"],dtype=np.int64)
    parent_face=np.asarray(Q["face_parent_candidate_rows"],dtype=np.int64)
    parent_set=set(map(int,Q["original_patch_face_rows"]))
    if len(Ppatch)!=len(src_rows) or len(PF)!=len(parent_face):raise RuntimeError("wide topology row alignment failed")

    # Preserve all nodes from the prior 25-edge boundary in addition to the wider fixed edge.
    base_boundary_edges=np.asarray(QB["patch_outer_boundary_edges"],dtype=np.int64)
    base_source_rows=np.asarray(QB["source_candidate_node_rows"],dtype=np.int64)
    base_boundary_source_nodes=sorted({int(base_source_rows[int(n)]) for e in base_boundary_edges for n in e
                                       if base_source_rows[int(n)]>=0})
    source_to_node={int(s):i for i,s in enumerate(src_rows) if s>=0}
    if any(s not in source_to_node for s in base_boundary_source_nodes):
        raise RuntimeError("wide patch lost one or more original 25-edge boundary nodes")
    wide_boundary_nodes=sorted({int(x) for e in boundary_edges for x in e})
    fixed_nodes=sorted(set(wide_boundary_nodes)|{source_to_node[s] for s in base_boundary_source_nodes})
    fixed_set=set(fixed_nodes)
    report["boundary_preservation"]={"original_fixed_outer_edges":int(len(base_boundary_edges)),
        "original_fixed_outer_source_nodes":len(base_boundary_source_nodes),
        "wide_fixed_outer_edges":int(len(boundary_edges)),
        "wide_fixed_outer_nodes":len(wide_boundary_nodes),
        "fixed_node_union":len(fixed_nodes)}

    T,ids,gids,bodies,M,off,pos,bankkeys,bankpins,unused=helper.read_bank_inputs(BANKDIR,E)
    mask=(ids[:,0]==115)&(ids[:,1]==118)
    bone_stable=ids[mask,3].astype(np.int64)
    bone_raw=T[mask].astype(np.float64)
    bone_map={int(s):i for i,s in enumerate(bone_stable)}
    bone_keys=[[exact.float32_point_lattice_key(v) for v in tri] for tri in bone_raw]
    bone_exact=[exact_tri(t) for t in bone_raw]
    bone_normals=[]
    edge_uses=defaultdict(list);signed6=0.0
    for fi,tri in enumerate(bone_raw):
        a,b,c=tri;n=np.cross(b-a,c-a);ln=float(np.linalg.norm(n))
        if ln==0:raise RuntimeError(f"degenerate bank face {fi}")
        bone_normals.append(n/ln);signed6+=float(a@np.cross(b,c))
        keys=bone_keys[fi]
        for i,j in ((0,1),(1,2),(2,0)):
            edge_uses[tuple(sorted((keys[i],keys[j])))].append((fi,keys[i],keys[j]))
    incidence=Counter(len(v) for v in edge_uses.values())
    if any(len(v)!=2 for v in edge_uses.values()):raise RuntimeError(f"bone edge closure failed: {dict(incidence)}")
    if any(v[0][1:]==v[1][1:] for v in edge_uses.values()):raise RuntimeError("bone triangle orientation inconsistent at exact edge")
    if abs(signed6)<1e-18:raise RuntimeError("bone signed volume too small")
    orient_sign=1.0 if signed6>0 else -1.0
    outward=np.asarray(bone_normals)*orient_sign
    bone_records=audit_tool.records(bone_raw,bone_stable,exact)
    prepared_bone=exact.prepare_point_location(bone_records)
    skin_local_all=(P32.astype(np.float64)-pos[115])@M[115].T-off[115]
    init_local=(Ppatch.astype(np.float64)-pos[115])@M[115].T-off[115]

    # Reconstruct the same exact endpoint components and map them to the wide patch node rows.
    audit_doc=json.loads(A.read_text())
    pairs=[tuple(map(int,p)) for p in next(b for b in audit_doc["banks"] if (b["body"],b["stable_bone_id"])==(115,118))["result"]["triangle_pairs"]]
    endpoint_rows=[]
    for sf,bone_id in pairs:
        bi=bone_map[bone_id];ts=exact_tri(skin_local_all[F[sf]]);tb=bone_exact[bi]
        raw=sorted(set(exact.triangle_intersection_points(ts,tb)))
        if len(raw)<2:raise RuntimeError(f"exact segment missing {(sf,bone_id)}")
        axis=max(range(3),key=lambda k:max(p[k] for p in raw)-min(p[k] for p in raw))
        ends=[min(raw,key=lambda p:p[axis]),max(raw,key=lambda p:p[axis])]
        for p in ends:
            sb=barycentric(p,ts,exact);bb=barycentric(p,tb,exact)
            skin_ids=[int(v) for v in F[sf]]
            sfkey=feature_for(sb,skin_ids,None,"skin")
            bfkey=feature_for(bb,None,bone_keys[bi],"bone")
            endpoint_rows.append({"id":len(endpoint_rows),"sf":sf,"bone":bone_id,"p":p,
                                  "sfeat":sfkey,"bfeat":bfkey})
    uf=list(range(len(endpoint_rows)))
    def find(i):
        while uf[i]!=i:uf[i]=uf[uf[i]];i=uf[i]
        return i
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb:uf[max(ra,rb)]=min(ra,rb)
    feature_owner={}
    for r in endpoint_rows:
        for f in (r["sfeat"],r["bfeat"]):
            if f is not None:
                if f in feature_owner:union(r["id"],feature_owner[f])
                else:feature_owner[f]=r["id"]
    roots={i:find(i) for i in range(len(endpoint_rows))}
    by_root=defaultdict(list)
    for r in endpoint_rows:by_root[roots[r["id"]]].append(r)
    root_to_node={int(root):i for i,root in enumerate(endpoint_roots) if root>=0}
    if set(root_to_node)!=set(by_root):raise RuntimeError("wide endpoint keys do not match exact contour")
    for root,rows in by_root.items():
        if len({r["p"] for r in rows})!=1:raise RuntimeError(f"exact root {root} coordinate conflict")

    # Outward normals at exact contour keys from their actual incident bank faces.
    endpoint_normals={}
    for root,rows in by_root.items():
        incident={bone_map[r["bone"]] for r in rows}
        for r in rows:
            f=r["bfeat"]
            if f and f[0]=="bone-emitted-edge":
                edge={f[3],f[4]}
                incident.update(i for i,keys in enumerate(bone_keys) if edge.issubset(set(keys)))
            elif f and f[0]=="bone-vertex":
                key=f[3];incident.update(i for i,keys in enumerate(bone_keys) if key in keys)
        n=np.sum(outward[sorted(incident)],axis=0);ln=float(np.linalg.norm(n))
        if ln<1e-10:raise RuntimeError(f"normal cancellation at endpoint key {root}")
        endpoint_normals[root]=n/ln

    # Point-location state for source vertices to be moved; keep fixed boundaries exterior.
    initial_source_locations=[]
    for i,src in enumerate(src_rows):
        if src<0:continue
        state=exact.point_location_prepared(tuple(frac(x) for x in init_local[i]),prepared_bone)["location"]
        initial_source_locations.append((i,int(src),state))
        if i in fixed_set and state!="outside":
            raise RuntimeError(f"fixed source/boundary point {i} is {state} to bank 115/118")
    init_state_counts=Counter(s for _,_,s in initial_source_locations)

    # Halfspace constraints for exact contour endpoints and originally inside/boundary source nodes.
    clearance=FIT_MM/1000.0
    halfspaces=[]
    for root,node in root_to_node.items():
        q=np.asarray([float(x) for x in by_root[root][0]["p"]],dtype=np.float64)
        n=endpoint_normals[root]
        halfspaces.append((node,q,n,"endpoint",root))
    for node,src,state in initial_source_locations:
        if state not in ("inside","boundary"):continue
        if node in fixed_set:raise RuntimeError(f"fixed node {node} must move to exit the bone")
        p=init_local[node];best=math.inf;bestq=None;bestface=-1
        for fi,tri in enumerate(bone_raw):
            q=closest_point_triangle(p,tri[0],tri[1],tri[2]);d=float(np.linalg.norm(q-p))
            if d<best:best=d;bestq=q;bestface=fi
        halfspaces.append((node,bestq,outward[bestface],"source_inside",src))
    movable=np.asarray([i for i in range(len(Ppatch)) if i not in fixed_set],dtype=np.int64)
    vmap={int(node):j for j,node in enumerate(movable)}
    if len(movable)==0:raise RuntimeError("wider patch has no movable nodes")
    # Graph Laplacian regularizes only the explicit constrained optimum; no harmonic
    # displacement field is used as the solution.
    edges=set()
    for tri in PF:
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            edges.add(tuple(sorted((int(a),int(b)))))
    L=np.zeros((len(movable),len(movable)),dtype=np.float64)
    for a,b in edges:
        ia=vmap.get(a);ib=vmap.get(b)
        if ia is not None:L[ia,ia]+=1.0
        if ib is not None:L[ib,ib]+=1.0
        if ia is not None and ib is not None:
            L[ia,ib]-=1.0;L[ib,ia]-=1.0
    smooth_weight=0.5
    H=np.eye(len(movable))+smooth_weight*L
    H3=np.kron(H,np.eye(3))
    x0=init_local.copy()
    nvar=len(movable)
    child_parent_local=skin_local_all[F[parent_face]]
    parent_norm=np.cross(child_parent_local[:,1]-child_parent_local[:,0],
                         child_parent_local[:,2]-child_parent_local[:,0])
    parent_norm_len=np.linalg.norm(parent_norm,axis=1)
    if np.any(parent_norm_len==0):raise RuntimeError("degenerate parent source face in widened patch")
    parent_unit=parent_norm/parent_norm_len[:,None]
    area_floor=AREA_RATIO_MIN
    def unpack(y):
        x=x0.copy()
        x[movable]+=np.asarray(y).reshape(nvar,3)*SCALE_M
        return x
    def objective(y):return float(0.5*np.asarray(y)@H3@np.asarray(y))
    def objective_jac(y):return H3@np.asarray(y)
    def half_fun(y):
        x=unpack(y);vals=[]
        for node,q,n,kind,ident in halfspaces:
            vals.append((float(n@(x[node]-q))-clearance)/SCALE_M)
        return np.asarray(vals)
    def half_jac(y):
        J=np.zeros((len(halfspaces),3*nvar),dtype=np.float64)
        for r,(node,q,n,kind,ident) in enumerate(halfspaces):
            j=vmap.get(node)
            if j is not None:J[r,3*j:3*j+3]=n
        return J
    def shape_fun(y):
        x=unpack(y);vals=np.empty(len(PF),dtype=np.float64)
        for i,face in enumerate(PF):
            a,b,c=x[face]
            double_area=float(parent_unit[i]@np.cross(b-a,c-a))
            vals[i]=double_area/parent_norm_len[i]-area_floor
        return vals
    def shape_jac(y):
        x=unpack(y);J=np.zeros((len(PF),3*nvar),dtype=np.float64)
        for i,face in enumerate(PF):
            ia,ib,ic=map(int,face);a,b,c=x[ia],x[ib],x[ic];n=parent_unit[i]
            grads=(np.cross(n,c-b),np.cross(c-a,n),np.cross(n,b-a))
            for node,g in zip((ia,ib,ic),grads):
                j=vmap.get(node)
                if j is not None:J[i,3*j:3*j+3]=SCALE_M*g/parent_norm_len[i]
        return J
    y0=np.zeros((nvar,3),dtype=np.float64).reshape(-1)
    initial_shape=shape_fun(y0)
    initial_half=half_fun(y0)
    res=minimize(objective,y0,jac=objective_jac,method="SLSQP",
                 constraints=[{"type":"ineq","fun":half_fun,"jac":half_jac},
                              {"type":"ineq","fun":shape_fun,"jac":shape_jac}],
                 options={"maxiter":100,"ftol":1e-8,"disp":False})
    fitted_local=unpack(res.x)
    fit_world=((fitted_local+off[115])@M[115]+pos[115]).astype(np.float32)
    fit_world[fixed_nodes]=Ppatch[fixed_nodes]
    fit_npz=OUT/"private-constrained-fit-patch.npz"
    np.savez_compressed(fit_npz,points_world_m_f32=fit_world,patch_faces=PF,
        source_candidate_node_rows=src_rows,
        point_parent_candidate_face_rows=Q["point_parent_candidate_face_rows"],
        point_parent_candidate_barycentric_f64=Q["point_parent_candidate_barycentric_f64"],
        face_parent_candidate_rows=parent_face,face_parent_barycentric_f64=Q["face_parent_barycentric_f64"],
        face_source_ancestry_indptr=Q["face_source_ancestry_indptr"],face_source_ancestry_face_rows=Q["face_source_ancestry_face_rows"],
        face_fine_subface_ancestry_indptr=Q["face_fine_subface_ancestry_indptr"],face_fine_subface_ancestry_rows=Q["face_fine_subface_ancestry_rows"],
        face_fine_detail_ancestry_indptr=Q["face_fine_detail_ancestry_indptr"],face_fine_detail_ancestry_face_rows=Q["face_fine_detail_ancestry_face_rows"],
        patch_outer_boundary_edges=boundary_edges,contour_edges=Q["contour_edges"],
        original_patch_face_rows=Q["original_patch_face_rows"],exact_endpoint_component_ids=endpoint_roots,
        node_displacement_body_local_m_f64=fitted_local-init_local)
    displacement=(fitted_local-init_local)
    report["optimizer"]={"method":"SLSQP explicit constrained displacement optimization",
        "success":bool(res.success),"message":str(res.message),"iterations":int(res.nit),
        "objective":float(res.fun),"movable_nodes":int(nvar),"fixed_nodes":len(fixed_nodes),
        "explicit_bone_halfspace_count":len(halfspaces),"explicit_face_orientation_area_constraint_count":int(len(PF)),
        "initial_min_shape_constraint":float(initial_shape.min()),
        "final_min_shape_constraint":float(shape_fun(res.x).min()),
        "initial_min_bone_halfspace_constraint_scaled":float(initial_half.min()) if len(initial_half) else None,
        "final_min_bone_halfspace_constraint_scaled":float(half_fun(res.x).min()) if len(halfspaces) else None,
        "smoothness_penalty_weight":smooth_weight,
        "max_node_displacement_mm":float(np.max(np.linalg.norm(displacement,axis=1))*1000.0),
        "rms_node_displacement_mm":float(np.sqrt(np.mean(np.sum(displacement*displacement,axis=1)))*1000.0),
        "output_patch_sha256":sha(fit_npz)}

    # Emitted Float32 shape gate against each original source parent face.
    child_world=fit_world[PF].astype(np.float64)
    parent_world=P32[F[parent_face]].astype(np.float64)
    pn=np.cross(parent_world[:,1]-parent_world[:,0],parent_world[:,2]-parent_world[:,0])
    cn=np.cross(child_world[:,1]-child_world[:,0],child_world[:,2]-child_world[:,0])
    pnorm2=np.einsum("ij,ij->i",pn,pn);parent_area=np.sqrt(pnorm2)*0.5;child_area=np.linalg.norm(cn,axis=1)*0.5
    orient=np.einsum("ij,ij->i",pn,cn)/np.maximum(pnorm2,1e-30)
    rel=child_area/np.maximum(parent_area,1e-30)
    aspects=[];mins=[]
    for tri in child_world:
        lens=[float(np.linalg.norm(tri[(i+1)%3]-tri[i])) for i in range(3)]
        area=float(np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))*0.5)
        aspects.append(max(lens)**2/(4*np.sqrt(3)*max(area,1e-30)))
        aa=[]
        for i in range(3):
            u=tri[(i+1)%3]-tri[i];v=tri[(i+2)%3]-tri[i]
            aa.append(float(np.degrees(np.arccos(np.clip((u@v)/(max(np.linalg.norm(u)*np.linalg.norm(v),1e-30)),-1,1)))))
        mins.append(min(aa))
    shape_pass=bool(np.all(orient>0) and float(rel.min())>=AREA_RATIO_MIN and max(aspects)<1e5 and min(mins)>0.001)
    report["shape_gate_emitted_f32"]={"positive_orientation":bool(np.all(orient>0)),
        "inverted_child_rows":np.flatnonzero(orient<=0).astype(int).tolist(),
        "minimum_relative_area":float(rel.min()),"maximum_aspect_measure":float(max(aspects)),
        "minimum_angle_degrees":float(min(mins)),"pass":shape_pass}

    # Exact local bone intersection and point-location screen on emitted Float32 patch.
    moved_local_f32=(fit_world.astype(np.float64)-pos[115])@M[115].T-off[115]
    patch_bone_records=audit_tool.records(moved_local_f32[PF],np.arange(len(PF)),exact)
    exact_bone=exact._audit_pair(patch_bone_records,bone_records,same_surface=False)
    vertex_loc=Counter(exact.point_location_prepared(tuple(frac(x) for x in p),prepared_bone)["location"]
                       for p in moved_local_f32)
    report["exact_local_bone_gate"]={"bank":[115,118],"pair_count":int(exact_bone["count"]),
        "pairs":exact_bone["triangle_pairs"],"patch_vertex_location_counts":dict(vertex_loc),
        "all_patch_vertices_outside":vertex_loc.get("inside",0)==0 and vertex_loc.get("boundary",0)==0,
        "pass":int(exact_bone["count"])==0 and vertex_loc.get("inside",0)==0 and vertex_loc.get("boundary",0)==0}

    # Exact local self: patch internal and patch versus AABB-overlapping complement.
    global_vertex_ids=[int(src) if src>=0 else int(len(P32)+endpoint_roots[i]) for i,src in enumerate(src_rows)]
    patch_self_records=exact_records_mesh(moved_local_f32,PF,np.arange(len(PF)),global_vertex_ids,exact)
    patch_self=exact._audit_pair(patch_self_records,patch_self_records,same_surface=True)
    comp_rows=np.asarray([i for i in range(len(F)) if i not in parent_set],dtype=np.int64)
    comp_tri=skin_local_all[F[comp_rows]]
    tri_patch=moved_local_f32[PF]
    lo,hi=tri_patch.min(axis=(0,1)),tri_patch.max(axis=(0,1))
    clo,chi=comp_tri.min(axis=1),comp_tri.max(axis=1)
    active=np.flatnonzero(np.all(chi>=lo,axis=1)&np.all(clo<=hi,axis=1))
    comp_rows_active=comp_rows[active]
    comp_records=exact_records_mesh(skin_local_all,F[comp_rows_active],
        len(PF)+comp_rows_active,np.arange(len(P32)),exact)
    patch_comp=exact._audit_pair(patch_self_records,comp_records,same_surface=True)
    report["exact_local_self_gate"]={"patch_internal_pair_count":int(patch_self["count"]),
        "patch_internal_pairs":patch_self["triangle_pairs"],"complement_aabb_active_count":int(len(active)),
        "patch_vs_complement_pair_count":int(patch_comp["count"]),
        "patch_vs_complement_pairs":patch_comp["triangle_pairs"],
        "pass":int(patch_self["count"])==0 and int(patch_comp["count"])==0}

    report["fixed_boundary_checks"]={"wider_fixed_boundary_node_bits_preserved":bool(
        np.array_equal(fit_world[wide_boundary_nodes].view(np.uint32),Ppatch[wide_boundary_nodes].view(np.uint32))),
        "original_25_edge_source_node_bits_preserved":bool(
        np.array_equal(fit_world[[source_to_node[s] for s in base_boundary_source_nodes]].view(np.uint32),
                       Ppatch[[source_to_node[s] for s in base_boundary_source_nodes]].view(np.uint32)))}
    report["status"]="PRIVATE_WIDE_FIT_LOCAL_GATES_PASS" if (
        bool(res.success) and shape_pass and report["exact_local_bone_gate"]["pass"] and
        report["exact_local_self_gate"]["pass"] and all(report["fixed_boundary_checks"].values())
    ) else "PRIVATE_WIDE_FIT_REJECTED"
    report["accepted_candidate_modified"]=False
    report["limitations"]=["One private widened local fit only; no full mesh or all-bank audit.",
        "Exact bone and self checks cover bank 115/118 and this patch plus AABB-overlapping complement only.",
        "No whole-solid containment, physical transfer, or native admission is claimed.",
        "Bone source vertex indices are absent from the pinned triangle CSV; exact emitted coordinate keys identify candidate bone edges."]
except Exception as e:
    report.update({"status":"PRIVATE_WIDE_FIT_TRIAL_ERROR","failure":repr(e),
                   "traceback":traceback.format_exc(),"elapsed_seconds":time.monotonic()-started})
report["script_path"]=str(Path(__file__));report["script_sha256"]=sha(Path(__file__))
report["elapsed_seconds"]=time.monotonic()-started
(OUT/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
print(json.dumps({k:v for k,v in report.items() if k not in ("traceback","exact_local_bone_gate","exact_local_self_gate")},indent=2,allow_nan=False))
