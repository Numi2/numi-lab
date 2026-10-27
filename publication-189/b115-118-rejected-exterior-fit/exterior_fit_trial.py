from __future__ import annotations
import hashlib, importlib.util, json, math, sys, time, traceback
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np

S=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638")
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
C=S/"partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz"
A=S/"partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/audit-159/report.json"
TOPO=S/"per-bank-workers-20261027/b115-118/attempt-006-exact-endpoint-identity/patch-trial-001"
TOPO_REPORT=TOPO/"report.json"
TOPO_NPZ=TOPO/"patch-candidate.npz"
TOPO_ANCESTRY=TOPO/"rational-ancestry.json"
OUT=S/"per-bank-workers-20261027/b115-118/attempt-006-exact-endpoint-identity/exterior-fit-trial-001"
AUDIT_TOOL=ROOT/"audit_skin_bone_surfaces_159.py"
HELPER=Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
EXACT=E/"collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py"
BANKDIR=E/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
FIT_MM=0.05
FIT_M=FIT_MM/1000.0
CAND_PIN="7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af"
TOPO_REPORT_PIN="bcd2748f0cde4347fc1dabb5fb31e8fac70b8b41f9eff03b3632d0352e199711"
TOPO_NPZ_PIN="840569c103e174269e9461d170d5cbccfb3f664da262c5798e027acacf8e617f"
TOPO_ANCESTRY_PIN="9c13b765b6abf56b5f5943e0f16a08c11c5851a5c84450edf0320135a553b578"

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def loadmod(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def frac(x): return Fraction.from_float(float(x))
def exact_tri(tri): return tuple(tuple(frac(x) for x in p) for p in np.asarray(tri).reshape(3,3))
def barycentric(p,tri,exact):
    n=exact._cross(exact._sub(tri[1],tri[0]),exact._sub(tri[2],tri[0]))
    axes=exact._axes(n)
    a,b,c,q=[tuple(v[k] for k in axes) for v in (tri[0],tri[1],tri[2],p)]
    den=exact._orient2(a,b,c)
    if den==0: raise ValueError("degenerate exact triangle")
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
        if not any(n): raise ValueError(f"exact degenerate triangle in records: {tri_id}")
        lo=tuple(min(p[k] for p in tri) for k in range(3))
        hi=tuple(max(p[k] for p in tri) for k in range(3))
        out.append((tri,lo,hi,int(tri_id),tuple(int(vertex_ids[i]) for i in ids)))
    return out

started=time.monotonic()
report={"schema":"numi.human.bank115_118.exterior-fit-local-gates.v1","status":"STARTED",
        "candidate_path":str(C),"candidate_sha256":None,"topology_patch_path":str(TOPO_NPZ),
        "topology_patch_sha256":None,"topology_report_path":str(TOPO_REPORT),
        "topology_report_sha256":None,"rational_ancestry_path":str(TOPO_ANCESTRY),
        "rational_ancestry_sha256":None,"output_patch_path":str(OUT/"private-fit-patch.npz"),
        "fit_mm":FIT_MM,"fixed_boundary_edge_count":25,"geometry_or_native_admission":False}
try:
    if sha(C)!=CAND_PIN or sha(TOPO_REPORT)!=TOPO_REPORT_PIN or sha(TOPO_NPZ)!=TOPO_NPZ_PIN or sha(TOPO_ANCESTRY)!=TOPO_ANCESTRY_PIN:
        raise RuntimeError("pinned candidate or topology artifacts changed")
    report.update({"candidate_sha256":sha(C),"topology_patch_sha256":sha(TOPO_NPZ),
                   "topology_report_sha256":sha(TOPO_REPORT),"rational_ancestry_sha256":sha(TOPO_ANCESTRY)})
    topo_doc=json.loads(TOPO_REPORT.read_text())
    if topo_doc.get("status")!="LOCAL_PATCH_TOPOLOGY_PASS":
        raise RuntimeError("topology input did not pass")
    audit_tool=loadmod("exterior_fit_audit_owner",AUDIT_TOOL)
    helper=loadmod("exterior_fit_bank_helper",HELPER)
    exact=audit_tool.exact_module()
    with np.load(C,allow_pickle=False) as z: Z={k:z[k].copy() for k in z.files}
    with np.load(TOPO_NPZ,allow_pickle=False) as z: Q={k:z[k].copy() for k in z.files}
    P32=np.asarray(Z["points_world_m"],dtype=np.float32)
    F=np.asarray(Z["source_O_faces"],dtype=np.int64)
    Ppatch=np.asarray(Q["points_world_m_f32"],dtype=np.float32)
    PF=np.asarray(Q["patch_faces"],dtype=np.int64)
    source_rows=np.asarray(Q["source_candidate_node_rows"],dtype=np.int64)
    endpoint_roots=np.asarray(Q["exact_endpoint_component_ids"],dtype=np.int64)
    boundary_edges=np.asarray(Q["patch_outer_boundary_edges"],dtype=np.int64)
    parent_face_rows=np.asarray(Q["face_parent_candidate_rows"],dtype=np.int64)
    if len(Ppatch)!=len(source_rows) or len(PF)!=len(parent_face_rows):
        raise RuntimeError("topology patch row alignment invalid")
    T,ids,gids,bodies,M,off,pos,bankkeys,bankpins,unused=helper.read_bank_inputs(BANKDIR,E)
    bm=(ids[:,0]==115)&(ids[:,1]==118)
    bone_stable=ids[bm,3].astype(np.int64)
    bone_raw=T[bm].astype(np.float64)
    bone_map={int(s):i for i,s in enumerate(bone_stable)}
    bone_keys=[[exact.float32_point_lattice_key(p) for p in t] for t in bone_raw]
    bone_exact=[exact_tri(t) for t in bone_raw]
    skin_local_all=(P32.astype(np.float64)-pos[115])@M[115].T-off[115]
    audit_doc=json.loads(A.read_text())
    bank_doc=next(b for b in audit_doc["banks"] if (b["body"],b["stable_bone_id"])==(115,118))
    pairs=[tuple(map(int,p)) for p in bank_doc["result"]["triangle_pairs"]]
    if len(pairs)!=46: raise RuntimeError("pinned pair count changed")

    # Reconstruct the same rational endpoint keys and verify patch-node alignment.
    endpoint_rows=[]
    for sf,bone_id in pairs:
        bi=bone_map[bone_id]
        ts=exact_tri(skin_local_all[F[sf]])
        tb=bone_exact[bi]
        pts=sorted(set(exact.triangle_intersection_points(ts,tb)))
        if len(pts)<2: raise RuntimeError(f"segment missing in pair {(sf,bone_id)}")
        axis=max(range(3),key=lambda k:max(p[k] for p in pts)-min(p[k] for p in pts))
        ends=[min(pts,key=lambda p:p[axis]),max(pts,key=lambda p:p[axis])]
        for ordinal,p in enumerate(ends):
            sb=barycentric(p,ts,exact);bb=barycentric(p,tb,exact)
            sv=[int(v) for v in F[sf]]
            sfeat=feature_for(sb,sv,None,"skin")
            bfeat=feature_for(bb,None,bone_keys[bi],"bone")
            world=tuple(sum(sb[j]*frac(P32[sv[j],k]) for j in range(3)) for k in range(3))
            endpoint_rows.append({"id":len(endpoint_rows),"sf":sf,"bone":bone_id,"p":p,
                                  "sb":sb,"bb":bb,"sfeat":sfeat,"bfeat":bfeat,"world":world})
    parent=list(range(len(endpoint_rows)))
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb:parent[max(ra,rb)]=min(ra,rb)
    owners={}
    for r in endpoint_rows:
        for f in (r["sfeat"],r["bfeat"]):
            if f is None:continue
            if f in owners:union(r["id"],owners[f])
            else:owners[f]=r["id"]
    roots={i:find(i) for i in range(len(endpoint_rows))}
    by_root=defaultdict(list)
    for r in endpoint_rows: by_root[roots[r["id"]]].append(r)
    root_to_node={int(root):i for i,root in enumerate(endpoint_roots) if root>=0}
    if set(root_to_node)!=set(by_root):
        raise RuntimeError("exact endpoint key set does not match topology patch")
    for root,rows in by_root.items():
        if len({r["p"] for r in rows})!=1 or len({r["world"] for r in rows})!=1:
            raise RuntimeError(f"exact endpoint key {root} has conflicting rational coordinates")
        node=root_to_node[root]
        expected=np.asarray([np.float32(float(x)) for x in rows[0]["world"]],dtype=np.float32)
        if not np.array_equal(expected.view(np.uint32),Ppatch[node].view(np.uint32)):
            raise RuntimeError(f"emitted endpoint row differs from exact identity {root}")

    # Exact emitted bone surface orientation and closed-edge check.
    edge_uses=defaultdict(list)
    signed6=0.0
    normals=[]
    for fi,tri in enumerate(bone_raw):
        a,b,c=tri
        n=np.cross(b-a,c-a);norm=float(np.linalg.norm(n))
        if norm==0:raise RuntimeError(f"degenerate bone input triangle row {fi}")
        normals.append(n/norm)
        signed6+=float(a@np.cross(b,c))
        keys=bone_keys[fi]
        for i,j in ((0,1),(1,2),(2,0)):
            ka,kb=keys[i],keys[j]
            edge_uses[tuple(sorted((ka,kb)))].append((fi,ka,kb))
    incidence_hist=Counter(len(v) for v in edge_uses.values())
    if any(len(v)!=2 for v in edge_uses.values()):
        raise RuntimeError(f"bank 115/118 edge closure failed: {dict(incidence_hist)}")
    if any(v[0][1:]==v[1][1:] for v in edge_uses.values()):
        raise RuntimeError("bank 115/118 edge winding is inconsistent")
    if abs(signed6)<1e-18:raise RuntimeError("bank 115/118 signed volume too small to orient normals")
    orientation_sign=1.0 if signed6>0 else -1.0
    outward_normals=np.asarray(normals)*orientation_sign

    # Prepare exact bank point-location and local query records.
    bone_records=audit_tool.records(bone_raw,bone_stable,exact)
    prepared_bone=exact.prepare_point_location(bone_records)
    def local_of_world(points):
        return (np.asarray(points,dtype=np.float32).astype(np.float64)-pos[115])@M[115].T-off[115]
    init_local=local_of_world(Ppatch)
    initial_loc=[]
    boundary_nodes=sorted({int(x) for e in boundary_edges for x in e})
    boundary_set=set(boundary_nodes)
    for node in range(len(Ppatch)):
        if source_rows[node]<0: continue
        p=tuple(frac(x) for x in init_local[node])
        loc=exact.point_location_prepared(p,prepared_bone)["location"]
        initial_loc.append({"node_row":node,"candidate_node_row":int(source_rows[node]),"location":loc})
        if node in boundary_set and loc!="outside":
            raise RuntimeError(f"fixed outer-boundary node {node} is {loc} bone-side")
    initial_counts=Counter(r["location"] for r in initial_loc)

    # Outward normal for each exact endpoint from all actual bank faces incident to its feature.
    endpoint_normals={}
    for root,rows in by_root.items():
        face_set={bone_map[r["bone"]] for r in rows}
        for r in rows:
            f=r["bfeat"]
            if f and f[0]=="bone-emitted-edge":
                edge={f[3],f[4]}
                face_set.update(i for i,keys in enumerate(bone_keys) if edge.issubset(set(keys)))
            elif f and f[0]=="bone-vertex":
                v=f[3]
                face_set.update(i for i,keys in enumerate(bone_keys) if v in keys)
        nvec=np.sum(outward_normals[sorted(face_set)],axis=0)
        nlen=float(np.linalg.norm(nvec))
        if nlen<1e-10:raise RuntimeError(f"outward normal cancelled at exact key {root}")
        endpoint_normals[root]=nvec/nlen

    # One discrete harmonic lift: fixed 25-edge boundary, exterior-directed
    # endpoint targets, and minimum-distance corrections for interior source nodes
    # that lie inside/on bone. Remaining interior nodes take a graph-Laplacian fit.
    adjacency=[set() for _ in range(len(Ppatch))]
    for tri in PF:
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            adjacency[int(a)].add(int(b));adjacency[int(b)].add(int(a))
    fixed_disp={i:np.zeros(3,dtype=np.float64) for i in boundary_nodes}
    for root,node in root_to_node.items():
        fixed_disp[node]=FIT_M*endpoint_normals[root]
    # Candidate source nodes inside the bone are moved to the closest surface plus
    # a small outward offset; fixed boundary nodes were already proven outside.
    for row in initial_loc:
        node=row["node_row"];state=row["location"]
        if state not in ("inside","boundary"):continue
        p=init_local[node]
        nearest=None;best=math.inf;nearest_face=-1
        for fi,tri in enumerate(bone_raw):
            q=closest_point_triangle(p,tri[0],tri[1],tri[2])
            d=float(np.linalg.norm(q-p))
            if d<best:best=d;nearest=q;nearest_face=fi
        n=outward_normals[nearest_face]
        target=(nearest-p)+FIT_M*n
        if node in boundary_set:
            raise RuntimeError(f"fixed boundary node {node} requires motion to exit bone")
        fixed_disp[node]=target
    known=set(fixed_disp)
    unknown=[i for i in range(len(Ppatch)) if i not in known]
    dloc=np.zeros_like(init_local)
    for i,d in fixed_disp.items():dloc[i]=d
    if unknown:
        col={node:i for i,node in enumerate(unknown)}
        L=np.zeros((len(unknown),len(unknown)),dtype=np.float64)
        rhs=np.zeros((len(unknown),3),dtype=np.float64)
        for node in unknown:
            i=col[node];neigh=adjacency[node]
            if not neigh:raise RuntimeError(f"isolated local patch node {node}")
            L[i,i]=len(neigh)
            for nb in neigh:
                if nb in col:L[i,col[nb]]-=1.0
                else:rhs[i]+=dloc[nb]
        dloc[unknown]=np.linalg.solve(L,rhs)
    moved_local=init_local+dloc
    # Map local displacements back to world, round each patch node once to emitted F32.
    moved_world=((moved_local+off[115])@M[115]+pos[115]).astype(np.float32)
    moved_world[boundary_nodes]=Ppatch[boundary_nodes]
    if not np.array_equal(moved_world[boundary_nodes].view(np.uint32),Ppatch[boundary_nodes].view(np.uint32)):
        raise RuntimeError("fixed outer-boundary Float32 coordinates changed")
    fit_npz=OUT/"private-fit-patch.npz"
    np.savez_compressed(fit_npz,points_world_m_f32=moved_world,patch_faces=PF,
                        source_candidate_node_rows=source_rows,
                        point_parent_candidate_face_rows=Q["point_parent_candidate_face_rows"],
                        point_parent_candidate_barycentric_f64=Q["point_parent_candidate_barycentric_f64"],
                        face_parent_candidate_rows=parent_face_rows,
                        face_parent_barycentric_f64=Q["face_parent_barycentric_f64"],
                        face_source_ancestry_indptr=Q["face_source_ancestry_indptr"],
                        face_source_ancestry_face_rows=Q["face_source_ancestry_face_rows"],
                        face_fine_subface_ancestry_indptr=Q["face_fine_subface_ancestry_indptr"],
                        face_fine_subface_ancestry_rows=Q["face_fine_subface_ancestry_rows"],
                        face_fine_detail_ancestry_indptr=Q["face_fine_detail_ancestry_indptr"],
                        face_fine_detail_ancestry_face_rows=Q["face_fine_detail_ancestry_face_rows"],
                        patch_outer_boundary_edges=boundary_edges,
                        contour_edges=Q["contour_edges"],
                        original_patch_face_rows=Q["original_patch_face_rows"],
                        exact_endpoint_component_ids=endpoint_roots,
                        node_displacement_body_local_m_f64=dloc)
    report["fit"]={"method":"one discrete harmonic displacement field over the 149-face patch",
                   "contour_target_offset_mm":FIT_MM,
                   "initial_source_node_point_locations":dict(initial_counts),
                   "fixed_boundary_nodes":len(boundary_nodes),
                   "moved_nodes":int(np.count_nonzero(np.linalg.norm(dloc,axis=1)>0)),
                   "maximum_node_displacement_mm":float(np.max(np.linalg.norm(dloc,axis=1))*1000.0),
                   "rms_node_displacement_mm":float(np.sqrt(np.mean(np.sum(dloc*dloc,axis=1)))*1000.0),
                   "bone_closed_exact_coordinate_edge_incidence":dict(incidence_hist),
                   "bone_signed_volume_sign_for_outward_normals":"positive" if signed6>0 else "negative",
                   "patch_fit_sha256":sha(fit_npz)}

    # Emitted-F32 shape gate against each original parent face.
    child_tri=moved_world[PF].astype(np.float64)
    parent_tri=P32[F[parent_face_rows]].astype(np.float64)
    pn=np.cross(parent_tri[:,1]-parent_tri[:,0],parent_tri[:,2]-parent_tri[:,0])
    cn=np.cross(child_tri[:,1]-child_tri[:,0],child_tri[:,2]-child_tri[:,0])
    pnorm2=np.einsum("ij,ij->i",pn,pn)
    area_parent=np.sqrt(pnorm2)*0.5
    area_child=np.linalg.norm(cn,axis=1)*0.5
    orientation=np.einsum("ij,ij->i",pn,cn)/np.maximum(pnorm2,1e-30)
    relative_area=area_child/np.maximum(area_parent,1e-30)
    aspect=[];angles=[]
    for tri in child_tri:
        lens=[float(np.linalg.norm(tri[(i+1)%3]-tri[i])) for i in range(3)]
        ar=float(np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))*0.5)
        aspect.append(max(lens)**2/(4*np.sqrt(3)*max(ar,1e-30)))
        aa=[]
        for i in range(3):
            u=tri[(i+1)%3]-tri[i];v=tri[(i+2)%3]-tri[i]
            aa.append(float(np.degrees(np.arccos(np.clip((u@v)/(max(np.linalg.norm(u)*np.linalg.norm(v),1e-30)),-1,1)))))
        angles.append(min(aa))
    shape={"positive_orientation":bool(np.all(orientation>0)),
           "minimum_relative_area":float(relative_area.min()),
           "maximum_aspect_measure":float(max(aspect)),
           "minimum_angle_degrees":float(min(angles)),
           "pass":bool(np.all(orientation>0) and float(relative_area.min())>=1e-6 and
                       float(max(aspect))<1e5 and float(min(angles))>0.001)}
    report["shape_gate_emitted_f32"]=shape

    # Exact local bone surface intersection screen plus exact exterior point-location.
    moved_local_f32=local_of_world(moved_world)
    patch_records=audit_tool.records(moved_local_f32[PF],np.arange(len(PF),dtype=np.int64),exact)
    bone_intersections=exact._audit_pair(patch_records,bone_records,same_surface=False)
    patch_vertex_locations=[]
    for ni,p in enumerate(moved_local_f32):
        q=tuple(frac(x) for x in p)
        state=exact.point_location_prepared(q,prepared_bone)["location"]
        patch_vertex_locations.append({"node_row":ni,"location":state})
    vertex_locs=Counter(x["location"] for x in patch_vertex_locations)
    report["exact_local_bone_gate"]={"bank":[115,118],"bone_triangle_count":len(bone_raw),
        "exact_triangle_intersection_pair_count":int(bone_intersections["count"]),
        "exact_triangle_pairs":bone_intersections["triangle_pairs"],
        "patch_vertex_locations":dict(vertex_locs),
        "all_patch_vertices_outside":vertex_locs.get("inside",0)==0 and vertex_locs.get("boundary",0)==0,
        "pass":int(bone_intersections["count"])==0 and vertex_locs.get("inside",0)==0 and vertex_locs.get("boundary",0)==0}

    # Local self screen: patch vs itself, then patch vs every complement face whose
    # exact AABB can overlap the patch's local bounding box.
    patch_vertex_ids=[]
    for i,src in enumerate(source_rows):
        patch_vertex_ids.append(int(src) if src>=0 else int(len(P32)+endpoint_roots[i]))
    patch_records_self=exact_records_mesh(moved_local_f32,PF,np.arange(len(PF)),patch_vertex_ids,exact)
    patch_self=exact._audit_pair(patch_records_self,patch_records_self,same_surface=True)
    in_patch=set(map(int,Q["original_patch_face_rows"]))
    comp_rows=np.asarray([i for i in range(len(F)) if i not in in_patch],dtype=np.int64)
    comp_tri=skin_local_all[F[comp_rows]]
    patch_tri_local=moved_local_f32[PF]
    plo=patch_tri_local.min(axis=(0,1));phi=patch_tri_local.max(axis=(0,1))
    clo=comp_tri.min(axis=1);chi=comp_tri.max(axis=1)
    active=np.flatnonzero(np.all(chi>=plo,axis=1)&np.all(clo<=phi,axis=1))
    comp_rows_active=comp_rows[active]
    comp_faces_active=F[comp_rows_active]
    comp_records=exact_records_mesh(skin_local_all,comp_faces_active,
        len(PF)+comp_rows_active,P32.shape[0]*[0] if False else np.arange(len(P32)),exact)
    # exact_records_mesh receives full node coordinates and candidate face indices; its vertex map is identity.
    patch_vs_comp=exact._audit_pair(patch_records_self,comp_records,same_surface=True)
    report["exact_local_self_gate"]={"patch_internal_pair_count":int(patch_self["count"]),
        "patch_internal_pairs":patch_self["triangle_pairs"],
        "complement_faces_aabb_active":int(len(active)),
        "patch_vs_complement_pair_count":int(patch_vs_comp["count"]),
        "patch_vs_complement_pairs":patch_vs_comp["triangle_pairs"],
        "pass":int(patch_self["count"])==0 and int(patch_vs_comp["count"])==0}

    # Record parent/source input and output identities. The accepted candidate is untouched.
    report["status"]="PRIVATE_FIT_LOCAL_GATES_PASS" if (
        shape["pass"] and report["exact_local_bone_gate"]["pass"] and report["exact_local_self_gate"]["pass"]
    ) else "PRIVATE_FIT_REJECTED_LOCAL_GATE_FAILURE"
    report["accepted_candidate_modified"]=False
    report["trial_patch_sha256"]=sha(fit_npz)
    report["limitations"]=["One private local fit only; not all-bank or full-body audit.",
        "No containment of whole solids, physical transfer, native admission, or external qualification was claimed.",
        "Exact Float32 bone and local self predicates are scoped to this bank and the 149-face patch plus its AABB-overlapping complement."]
except Exception as e:
    report.update({"status":"PRIVATE_FIT_REJECTED_TRIAL_ERROR","failure":repr(e),
                   "traceback":traceback.format_exc(),"elapsed_seconds":time.monotonic()-started})
report["script_path"]=str(Path(__file__))
report["script_sha256"]=sha(Path(__file__))
report["elapsed_seconds"]=time.monotonic()-started
(OUT/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
print(json.dumps({k:v for k,v in report.items() if k not in ("traceback","exact_local_self_gate","exact_local_bone_gate")},indent=2,allow_nan=False))
