from __future__ import annotations
import hashlib, importlib.util, json, math, sys, time, traceback
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay

S = Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638")
ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
C = S / "partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz"
A = S / "partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/audit-159/report.json"
OUT = S / "per-bank-workers-20261027/b43-16/attempt-005-exact-endpoint-identity/patch-trial-001"
AUDIT_TOOL = ROOT / "audit_skin_bone_surfaces_159.py"
HELPER = Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
E = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
EXACT = E / "collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py"
BANKDIR = E / "bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
CAND_PIN = "7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af"
AUDIT_PIN = "c480d498bc1fb0f361e817b7f785b7b425ba9f40c0e181f7176ae375103cce8e"

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def loadmod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

def frac(x):
    return Fraction.from_float(float(x))

def exact_tri(tri):
    return tuple(tuple(frac(x) for x in p) for p in np.asarray(tri).reshape(3, 3))

def barycentric(p, tri, exact):
    n = exact._cross(exact._sub(tri[1], tri[0]), exact._sub(tri[2], tri[0]))
    axes = exact._axes(n)
    a, b, c, q = [tuple(v[k] for k in axes) for v in (tri[0], tri[1], tri[2], p)]
    den = exact._orient2(a, b, c)
    if den == 0:
        raise ValueError("degenerate exact triangle")
    return (exact._orient2(b, c, q) / den,
            exact._orient2(c, a, q) / den,
            exact._orient2(a, b, q) / den)

def feature_for(bary, vertex_ids, vertex_keys, side, body, bone):
    zeros = [i for i, w in enumerate(bary) if w == 0]
    if len(zeros) >= 2:
        i = next(i for i, w in enumerate(bary) if w == 1)
        if side == "skin":
            return ("skin-vertex", int(vertex_ids[i]))
        return ("bone-vertex", int(body), int(bone), vertex_keys[i])
    if len(zeros) == 1:
        i, j = [x for x in range(3) if x != zeros[0]]
        if side == "skin":
            vi, vj = int(vertex_ids[i]), int(vertex_ids[j])
            lo, hi = sorted((vi, vj))
            t = bary[j] if vj == hi else bary[i]
            return ("skin-edge", lo, hi, t)
        ki, kj = vertex_keys[i], vertex_keys[j]
        lo, hi = sorted((ki, kj))
        t = bary[j] if kj == hi else bary[i]
        return ("bone-emitted-edge", int(body), int(bone), lo, hi, t)
    return None

def ratstr(q):
    return f"{q.numerator}/{q.denominator}"

def keyjson(x):
    if isinstance(x, Fraction):
        return ratstr(x)
    if isinstance(x, tuple):
        return [keyjson(v) for v in x]
    return x

def f32(q):
    return np.float32(float(q))

def orient(a, b, c):
    return float((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))

def make_tri(a, b, c, p2):
    o = orient(p2[a], p2[b], p2[c])
    if abs(o) < 1e-18:
        raise ValueError(f"degenerate reference triangle {(a,b,c)} area2={o}")
    return (a, b, c) if o > 0 else (a, c, b)

def proper_cross(e, f, p2):
    a, b = e; c, d = f
    if len({a, b, c, d}) < 4:
        return False
    o1, o2 = orient(p2[a], p2[b], p2[c]), orient(p2[a], p2[b], p2[d])
    o3, o4 = orient(p2[c], p2[d], p2[a]), orient(p2[c], p2[d], p2[b])
    return o1*o2 < -1e-26 and o3*o4 < -1e-26

def edge_map(tris):
    out = defaultdict(list)
    for i, t in enumerate(tris):
        for a, b in ((t[0],t[1]), (t[1],t[2]), (t[2],t[0])):
            out[tuple(sorted((int(a),int(b))))].append(i)
    return out

def constrained_triangulate(p2, constraints, max_flips=5000):
    if len(p2) < 3:
        raise ValueError("less than three face nodes")
    T = [make_tri(*map(int,t), p2) for t in Delaunay(p2).simplices]
    if len({v for t in T for v in t}) != len(p2):
        raise ValueError("Delaunay omitted one or more local points")
    protected, flip_count = set(), 0
    for target0 in constraints:
        target = tuple(sorted(map(int,target0)))
        if target[0] == target[1]:
            raise ValueError("collapsed local constraint")
        for _ in range(max_flips):
            edges = edge_map(T)
            if target in edges:
                protected.add(target)
                break
            crossing = [e for e in edges if proper_cross(e,target,p2)]
            switched = False
            for e in sorted(crossing):
                if e in protected or len(edges[e]) != 2:
                    continue
                i,j = edges[e]
                c = next(v for v in T[i] if v not in e)
                d = next(v for v in T[j] if v not in e)
                new = tuple(sorted((c,d)))
                if new in edges or not proper_cross(e,new,p2) or proper_cross(new,target,p2):
                    continue
                if any(proper_cross(new,k,p2) for k in protected):
                    continue
                t1, t2 = make_tri(c,d,e[0],p2), make_tri(d,c,e[1],p2)
                if abs((orient(*(p2[x] for x in T[i]))+orient(*(p2[x] for x in T[j]))) -
                       (orient(*(p2[x] for x in t1))+orient(*(p2[x] for x in t2)))) > 1e-10:
                    continue
                T[i],T[j] = t1,t2
                flip_count += 1
                switched = True
                break
            if not switched:
                raise ValueError(f"cannot insert constraint {target}; crossing edges={crossing[:8]}")
        else:
            raise ValueError(f"flip cap reached for {target}")
    edges = edge_map(T)
    missing = [tuple(sorted(map(int,e))) for e in constraints if tuple(sorted(map(int,e))) not in edges]
    if missing:
        raise ValueError(f"local constraints missing after CDT: {missing[:8]}")
    for i,e in enumerate(edges):
        for f in list(edges)[i+1:]:
            if proper_cross(e,f,p2):
                raise ValueError(f"local triangulation edges cross: {e} {f}")
    return T, flip_count

started = time.monotonic()
report = {
    "schema": "numi.human.bank43_16.exact-endpoint-facewise-patch-topology.v1",
    "status": "STARTED",
    "candidate_path": str(C), "candidate_sha256": None,
    "audit_report_path": str(A), "audit_report_sha256": None,
    "exact_predicate_path": str(EXACT), "exact_predicate_sha256": None,
    "script_path": str(Path(__file__)), "script_sha256": None,
    "patch_candidate_path": str(OUT / "patch-candidate.npz"),
    "rational_ancestry_path": str(OUT / "rational-ancestry.json"),
    "geometry_or_native_admission": False,
    "scope": "One local face-wise topology integration over the retained two-ring 502-face patch; no all-bank or self audit.",
}
try:
    if sha(C) != CAND_PIN or sha(A) != AUDIT_PIN:
        raise RuntimeError("pinned candidate or audit changed")
    report["candidate_sha256"] = sha(C)
    report["audit_report_sha256"] = sha(A)
    report["exact_predicate_sha256"] = sha(EXACT)
    audit_tool = loadmod("exact_patch_audit_owner", AUDIT_TOOL)
    helper = loadmod("exact_patch_bank_helper", HELPER)
    exact = audit_tool.exact_module()
    with np.load(C, allow_pickle=False) as z:
        Z = {k: z[k].copy() for k in z.files}
    P32 = np.asarray(Z["points_world_m"], dtype=np.float32)
    P = P32.astype(np.float64)
    F = np.asarray(Z["source_O_faces"], dtype=np.int64)
    T, ids, gids, bodies, M, off, pos, bankkeys, bankpins, unused = helper.read_bank_inputs(BANKDIR,E)
    bmask = (ids[:,0]==43) & (ids[:,1]==16)
    bone_stable = ids[bmask,3].astype(np.int64)
    bone_raw = T[bmask].astype(np.float64)
    bone_row_for_id = {int(v): i for i,v in enumerate(bone_stable)}
    bone_local = [exact_tri(t) for t in bone_raw]
    bone_vertex_keys = [[exact.float32_point_lattice_key(v) for v in tri] for tri in bone_raw]
    skin_local_all = (P-pos[43]) @ M[43].T - off[43]
    skin_local = skin_local_all
    audit_doc = json.loads(A.read_text())
    bank_doc = next(x for x in audit_doc["banks"] if (x["body"],x["stable_bone_id"])==(43,16))
    pairs = [tuple(map(int,p)) for p in bank_doc["result"]["triangle_pairs"]]
    if len(pairs) != 249:
        raise RuntimeError(f"expected 249 pinned triangle pairs, got {len(pairs)}")

    endpoint_rows, pair_segments = [], []
    for sf, bone_id in pairs:
        bi = bone_row_for_id[bone_id]
        ts, tb = exact_tri(skin_local[F[sf]]), bone_local[bi]
        raw = exact.triangle_intersection_points(ts,tb)
        uniq = sorted(set(raw))
        if len(uniq) < 2:
            raise RuntimeError(f"pair {(sf,bone_id)} has no exact segment")
        p0 = uniq[0]
        direction = next(exact._sub(p,p0) for p in uniq[1:] if p != p0)
        if not all(not any(exact._cross(direction,exact._sub(p,p0))) for p in uniq[1:]):
            raise RuntimeError(f"pair {(sf,bone_id)} has non-segment exact intersection")
        axis = max(range(3), key=lambda k:max(p[k] for p in uniq)-min(p[k] for p in uniq))
        endpoints = [min(uniq,key=lambda p:p[axis]), max(uniq,key=lambda p:p[axis])]
        local_ids=[]
        skin_ids=[int(v) for v in F[sf]]
        for ordinal,p in enumerate(endpoints):
            sb=barycentric(p,ts,exact)
            bb=barycentric(p,tb,exact)
            if min(sb)<0 or max(sb)>1 or min(bb)<0 or max(bb)>1:
                raise RuntimeError(f"exact endpoint outside pair {(sf,bone_id)}")
            sfeat=feature_for(sb,skin_ids,None,"skin",43,16)
            bfeat=feature_for(bb,None,bone_vertex_keys[bi],"bone",43,16)
            world=tuple(sum(sb[j]*frac(P32[skin_ids[j],k]) for j in range(3)) for k in range(3))
            local_ids.append(len(endpoint_rows))
            endpoint_rows.append({"id":len(endpoint_rows),"sf":sf,"bone":bone_id,"ordinal":ordinal,
                                  "p":p,"sb":sb,"bb":bb,"sfeat":sfeat,"bfeat":bfeat,"world":world})
        pair_segments.append((sf,bone_id,local_ids))
    parent=list(range(len(endpoint_rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[max(ra,rb)]=min(ra,rb)
    feature_owner={}
    for r in endpoint_rows:
        for feat in (r["sfeat"],r["bfeat"]):
            if feat is not None:
                if feat in feature_owner: union(r["id"],feature_owner[feat])
                else: feature_owner[feat]=r["id"]
    roots={i:find(i) for i in range(len(endpoint_rows))}
    by_root=defaultdict(list)
    for r in endpoint_rows: by_root[roots[r["id"]]].append(r)
    if not by_root:
        raise RuntimeError("exact feature union produced no endpoint roots")
    for root, rows in by_root.items():
        if len({r["p"] for r in rows})!=1 or len({r["world"] for r in rows})!=1:
            raise RuntimeError(f"endpoint feature union has exact-coordinate disagreement at root {root}")
    contour_degree=Counter(); contour_adjacency=defaultdict(set)
    for sf,bone_id,ids0 in pair_segments:
        ra,rb=find(ids0[0]),find(ids0[1])
        if ra==rb: raise RuntimeError(f"collapsed exact contour segment {(sf,bone_id)}")
        contour_degree[ra]+=1; contour_degree[rb]+=1
        contour_adjacency[ra].add(rb); contour_adjacency[rb].add(ra)
    if any(d!=2 for d in contour_degree.values()):
        raise RuntimeError(f"exact endpoint contour is branched: {dict(Counter(contour_degree.values()))}")
    contour_unseen=set(contour_adjacency); contour_components=[]
    while contour_unseen:
        st=min(contour_unseen);contour_unseen.remove(st);q=[st];cc=[]
        while q:
            r=q.pop();cc.append(r)
            for n in contour_adjacency[r]:
                if n in contour_unseen:contour_unseen.remove(n);q.append(n)
        contour_components.append(cc)

    # Reconstruct the retained 434-face topology patch plus its two edge rings.
    # The retained seed already includes the bounded complement-fill faces.
    cross=sorted({sf for sf,_ in pairs})
    edgefaces=defaultdict(list); nodefaces=defaultdict(list)
    for fi,t in enumerate(F):
        for v in t: nodefaces[int(v)].append(fi)
        for u,v in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
            edgefaces[tuple(sorted((int(u),int(v))))].append(fi)
    def patch_boundary_rows(p0):
        rows=[]
        for e,fs in edgefaces.items():
            ii=[f for f in fs if f in p0];oo=[f for f in fs if f not in p0]
            if ii and oo: rows.append((e,ii,oo))
            elif ii and len(fs)==1: rows.append((e,ii,[]))
        return rows
    patch_seed_path=S/"per-bank-workers-20261027/b43-16/attempt-001/complement-filled-patch-face-rows.npy"
    patch=set(map(int,np.load(patch_seed_path,allow_pickle=False)))
    if len(patch)!=434 or not set(cross)<=patch:
        raise RuntimeError(f"pinned retained seed mismatch: faces={len(patch)}, crossing subset={set(cross)<=patch}")
    ring_face_add_counts=[]
    for _ring in range(2):
        ring_rows=patch_boundary_rows(patch)
        add_faces={f for _,_,oo in ring_rows for f in oo}-patch
        ring_face_add_counts.append(len(add_faces))
        patch.update(add_faces)
    reconstructed_patch_face_count=len(patch)
    if ring_face_add_counts!=[40,28] or reconstructed_patch_face_count!=502:
        raise RuntimeError(f"retained patch reconstruction mismatch: additions={ring_face_add_counts}, faces={reconstructed_patch_face_count}")
    parent_faces=sorted(patch); parent_set=set(parent_faces)
    patch_edge_faces=defaultdict(list)
    for fi in parent_faces:
        t=F[fi]
        for u,v in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
            patch_edge_faces[tuple(sorted((int(u),int(v))))].append(fi)
    boundary_edges=[]
    for e,inside in patch_edge_faces.items():
        if any(g not in parent_set for g in edgefaces[e]) or len(edgefaces[e])==1:
            boundary_edges.append(e)
    bgraph=defaultdict(list)
    for u,v in boundary_edges: bgraph[u].append(v); bgraph[v].append(u)
    unseen=set(bgraph); bcomps=[]
    while unseen:
        seed=min(unseen); unseen.remove(seed); stack=[seed]; comp=[]
        while stack:
            v=stack.pop(); comp.append(v)
            for w in bgraph[v]:
                if w in unseen: unseen.remove(w); stack.append(w)
        bcomps.append(comp)
    if len(bcomps)!=1 or any(len(bgraph[v])!=2 for v in bgraph):
        raise RuntimeError("one-ring patch boundary is not one simple cycle")
    patch_nodes=sorted({int(v) for fi in parent_faces for v in F[fi]})
    points={("v",v):P32[v].copy() for v in patch_nodes}
    point_bary=defaultdict(dict)
    for fi in parent_faces:
        for j,v in enumerate(F[fi]):
            b=[Fraction(0),Fraction(0),Fraction(0)]; b[j]=Fraction(1)
            point_bary[fi][("v",int(v))]=tuple(b)
    endpoint_key={root:("x",root) for root in by_root}
    for root, rows in by_root.items():
        wp=rows[0]["world"]
        points[endpoint_key[root]]=np.asarray([f32(x) for x in wp],dtype=np.float32)
    def put_face(fi,key,b):
        b=tuple(b)
        if sum(b)!=1 or min(b)<0:
            raise RuntimeError(f"bad exact face barycentric coordinate {(fi,key,b)}")
        old=point_bary[fi].get(key)
        if old is not None and old!=b:
            raise RuntimeError(f"same canonical node has different exact face barycentrics in face {fi}")
        point_bary[fi][key]=b
    # Each endpoint is added by exact face barycentrics; shared skin-edge nodes are
    # propagated to every incident face in the fixed one-ring patch.
    for r in endpoint_rows:
        root=roots[r["id"]]; key=endpoint_key[root]
        sf=r["sf"];put_face(sf,key,r["sb"])
        feat=r["sfeat"]
        if feat and feat[0]=="skin-edge":
            _,lo,hi,t=feat
            for nb in edgefaces[(lo,hi)]:
                if nb not in parent_set: continue
                nodes=[int(v) for v in F[nb]]
                b=[Fraction(0),Fraction(0),Fraction(0)]
                b[nodes.index(lo)]=1-t;b[nodes.index(hi)]=t
                put_face(nb,key,b)
        elif feat and feat[0]=="skin-vertex":
            v=feat[1]
            for nb in nodefaces[v]:
                if nb not in parent_set: continue
                nodes=[int(x) for x in F[nb]]
                b=[Fraction(0),Fraction(0),Fraction(0)];b[nodes.index(v)]=Fraction(1)
                put_face(nb,key,b)

    segments_by_face=defaultdict(list)
    for sf,bone_id,ids0 in pair_segments:
        a,b=(endpoint_key[roots[i]] for i in ids0)
        if a==b: raise RuntimeError(f"exact contour segment collapsed {(sf,bone_id)}")
        segments_by_face[sf].append((a,b,bone_id))

    all_child_keys=[]; all_child_parent=[]; all_child_bary_exact=[]
    per_face=[]; flips_total=0
    for fi in parent_faces:
        keys_face=sorted(point_bary[fi],key=lambda k:(0,k[1]) if k[0]=="v" else (1,k[1]))
        idx={k:i for i,k in enumerate(keys_face)}
        local_bary=[point_bary[fi][k] for k in keys_face]
        p2=np.asarray([[float(b[1]),float(b[2])] for b in local_bary],dtype=np.float64)
        if len({tuple(v) for v in p2})!=len(p2):
            raise RuntimeError(f"distinct exact nodes collapse in reference plane of face {fi}")
        constraints=[]; contour_local=[]
        for a,b,bone_id in segments_by_face.get(fi,[]):
            e=tuple(sorted((idx[a],idx[b])))
            contour_local.append(e);constraints.append(e)
        tri_nodes=[int(v) for v in F[fi]]
        boundary_local=[]
        for j,k in ((0,1),(1,2),(2,0)):
            other=3-j-k
            seq=[]
            for qkey,b in zip(keys_face,local_bary):
                if b[other]==0:
                    seq.append((b[k],idx[qkey]))
            seq.sort()
            if len(seq)<2 or seq[0][1]!=idx[("v",tri_nodes[j])] or seq[-1][1]!=idx[("v",tri_nodes[k])]:
                raise RuntimeError(f"parent face {fi} boundary edge {(tri_nodes[j],tri_nodes[k])} lost an endpoint")
            for (_,a),(_,b) in zip(seq,seq[1:]):
                boundary_local.append(tuple(sorted((a,b))))
        constraints=list(dict.fromkeys(constraints+boundary_local))
        for i,e in enumerate(constraints):
            for f in constraints[i+1:]:
                if proper_cross(e,f,p2):
                    raise RuntimeError(f"crossing constraints in parent face {fi}: {e} {f}")
        if len(keys_face)==3 and not contour_local:
            tri_ids=[idx[("v",int(v))] for v in F[fi]]
            tris=[make_tri(tri_ids[0],tri_ids[1],tri_ids[2],p2)]
            flips=0
        else:
            tris,flips=constrained_triangulate(p2,constraints)
        flips_total+=flips
        emap=edge_map(tris)
        area=sum(orient(*(p2[x] for x in t))*0.5 for t in tris)
        if abs(area-0.5)>1e-10:
            raise RuntimeError(f"parent face {fi} coverage error {area}")
        if any(len(v) not in (1,2) for v in emap.values()):
            raise RuntimeError(f"parent face {fi} has invalid local edge incidence")
        for e in contour_local:
            if len(emap.get(tuple(sorted(e)),[]))!=2:
                raise RuntimeError(f"contour edge is not internal in parent face {fi}: {e}")
        for t in tris:
            all_child_keys.append([keys_face[x] for x in t])
            all_child_parent.append(fi)
            all_child_bary_exact.append([local_bary[x] for x in t])
        per_face.append({"parent_face_row":fi,"point_count":len(keys_face),"child_triangle_count":len(tris),
                         "contour_constraint_count":len(contour_local),"constraint_count":len(constraints),
                         "reference_area":area,"flips":flips})
    # Output node IDs preserve candidate rows for originals and add one F32 row per exact key.
    new_roots=sorted(by_root)
    node_keys=[("v",v) for v in patch_nodes]+[endpoint_key[r] for r in new_roots]
    node_index={k:i for i,k in enumerate(node_keys)}
    out_points=np.asarray([points[k] for k in node_keys],dtype=np.float32)
    child_faces=np.asarray([[node_index[k] for k in tri] for tri in all_child_keys],dtype=np.int64)
    child_parent=np.asarray(all_child_parent,dtype=np.int64)
    child_bary=np.asarray([[[float(x) for x in b] for b in tri] for tri in all_child_bary_exact],dtype=np.float64)
    edge_counts=Counter()
    for tri in child_faces:
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            edge_counts[tuple(sorted((int(a),int(b))))]+=1
    nonmanifold=[(e,n) for e,n in edge_counts.items() if n not in (1,2)]
    old_boundary={tuple(sorted((node_index[("v",u)],node_index[("v",v)]))) for u,v in boundary_edges}
    new_boundary={e for e,n in edge_counts.items() if n==1}
    if new_boundary!=old_boundary:
        raise RuntimeError(f"outer boundary changed: missing={len(old_boundary-new_boundary)}, added={len(new_boundary-old_boundary)}")
    if nonmanifold:
        raise RuntimeError(f"output patch nonmanifold edges: {nonmanifold[:8]}")
    contour_global_edges=sorted({tuple(sorted((node_index[a],node_index[b])))
                                 for sf,rows in segments_by_face.items() for a,b,_ in rows})
    if any(edge_counts.get(e)!=2 for e in contour_global_edges):
        raise RuntimeError("one or more emitted contour constraints do not have two incident child faces")
    # Inherited ancestry for each child face and exact rational sidecar.
    src_ptr,src_rows=[0],[]
    sub_ptr,sub_rows=[0],[]
    det_ptr,det_rows=[0],[]
    for fi in child_parent:
        a,b=map(int,Z["face_source_ancestry_indptr"][int(fi):int(fi)+2])
        src_rows.extend(map(int,Z["face_source_ancestry_face_rows"][a:b]));src_ptr.append(len(src_rows))
        a,b=map(int,Z["face_fine_subface_ancestry_indptr"][int(fi):int(fi)+2])
        sub_rows.extend(map(int,Z["face_fine_subface_ancestry_rows"][a:b]));sub_ptr.append(len(sub_rows))
        a,b=map(int,Z["face_fine_detail_ancestry_indptr"][int(fi):int(fi)+2])
        det_rows.extend(map(int,Z["face_fine_detail_ancestry_rows"][a:b]));det_ptr.append(len(det_rows))
    parent_node_rows=np.asarray([int(k[1]) if k[0]=="v" else -1 for k in node_keys],dtype=np.int64)
    point_parent_rows=np.full(len(node_keys),-1,dtype=np.int64)
    point_parent_bary=np.zeros((len(node_keys),3),dtype=np.float64)
    exact_point_ancestry=[]
    for root in new_roots:
        key=endpoint_key[root]
        uses=[(fi,b) for fi,m in point_bary.items() for k,b in m.items() if k==key]
        if not uses: raise RuntimeError(f"exact endpoint key {root} has no parent face")
        fi,b=uses[0]
        ni=node_index[key]
        point_parent_rows[ni]=fi;point_parent_bary[ni]=[float(x) for x in b]
        exact_point_ancestry.append({"node_row":ni,"exact_endpoint_key":root,
                                     "representative_parent_candidate_face_row":fi,
                                     "barycentric_on_representative_parent_face":[ratstr(x) for x in b],
                                     "exact_world_endpoint_m":[ratstr(x) for x in by_root[root][0]["world"]],
                                     "source_feature_keys":[keyjson(by_root[root][0]["sfeat"]),
                                                            keyjson(by_root[root][0]["bfeat"])]})
    exact_face_bary=[]
    for child_row,(fi,tri_bary) in enumerate(zip(child_parent,all_child_bary_exact)):
        exact_face_bary.append({"child_face_row":child_row,"parent_candidate_face_row":int(fi),
                                "vertex_barycentrics":[[ratstr(x) for x in b] for b in tri_bary]})
    # Validate inherited CSR and Float32 distinctness/degeneracy before any NPZ is emitted.
    csr_summary={}
    for name,ptr,rows in (("source",src_ptr,src_rows),("fine_subface",sub_ptr,sub_rows),("fine_detail",det_ptr,det_rows)):
        counts=np.diff(np.asarray(ptr,dtype=np.int64))
        valid=(len(ptr)==len(child_parent)+1 and len(ptr)>0 and int(ptr[0])==0
               and int(ptr[-1])==len(rows) and bool(np.all(counts>0))
               and not any(int(v)<0 for v in rows))
        csr_summary[name]={"valid":bool(valid),"child_count":len(child_parent),
                           "empty_child_rows":int(np.sum(counts==0)),
                           "negative_ancestry_rows":int(sum(int(v)<0 for v in rows)),
                           "ancestry_id_set_count":len({int(v) for v in rows})}
    csr_valid=all(v["valid"] for v in csr_summary.values())
    float_keys=defaultdict(list)
    for root in new_roots:
        ni=node_index[endpoint_key[root]]
        bits=tuple(int(v) for v in out_points[ni].view(np.uint32))
        float_keys[bits].append(int(root))
    collision_groups=[{"emitted_f32_bits":list(bits),"distinct_exact_roots":roots}
                      for bits,roots in float_keys.items() if len(roots)>1]
    collision_witnesses=[]
    for g in collision_groups:
        roots=g["distinct_exact_roots"];anchor=by_root[roots[0]][0]["world"]
        for root in roots:
            exact_world=by_root[root][0]["world"]
            delta2=sum((exact_world[k]-anchor[k])**2 for k in range(3))
            emitted=out_points[node_index[endpoint_key[root]]]
            collision_witnesses.append({"root":root,"colliding_root_group":roots,
                "exact_world_endpoint_m":[ratstr(x) for x in exact_world],
                "emitted_world_f32_m":[float(x) for x in emitted],
                "emitted_f32_bits":list(g["emitted_f32_bits"]),
                "exact_squared_separation_from_group_anchor_m2":ratstr(delta2),
                "separation_from_group_anchor_m":math.sqrt(float(delta2)),
                "incident_pair_endpoints":[{"skin_face":int(r["sf"]),"bone_triangle":int(r["bone"]),
                    "endpoint_ordinal":int(r["ordinal"]),"skin_feature":keyjson(r["sfeat"]),
                    "bone_feature":keyjson(r["bfeat"])} for r in by_root[root]]})
    degenerate_child_faces=[]
    for ci,tri_ids in enumerate(child_faces):
        t=exact_tri(out_points[tri_ids])
        if not any(exact._cross(exact._sub(t[1],t[0]),exact._sub(t[2],t[0]))):
            degenerate_child_faces.append(int(ci))
    graph_receipt=OUT.parent/"receipt.json"
    graph_doc=json.loads(graph_receipt.read_text())
    f32check=graph_doc["float32_emission_recheck"]
    ancestry_sidecar=OUT/"facewise-topology-ancestry.json"
    ancestry_doc={"schema":"numi.human.bank43_16.facewise-topology-ancestry.v1",
        "candidate_sha256":sha(C),
        "source_ancestry_csr":{"source":{"indptr":src_ptr,"rows":src_rows},
            "fine_subface":{"indptr":sub_ptr,"rows":sub_rows},
            "fine_detail":{"indptr":det_ptr,"rows":det_rows}},
        "exact_endpoint_nodes":exact_point_ancestry,
        "child_face_parent_candidate_rows":list(map(int,child_parent)),
        "child_face_parent_barycentrics":exact_face_bary}
    ancestry_sidecar.write_text(json.dumps(ancestry_doc,indent=2)+chr(10))
    report["facewise_topology_attempt"]={"status":"FACEWISE_TOPOLOGY_BUILT_IN_MEMORY",
        "crossing_pairs":len(pairs),"crossing_skin_faces":len(cross),
        "reconstructed_retained_two_ring_patch_faces":reconstructed_patch_face_count,
        "patch_seed_sha256":sha(patch_seed_path),"ring_face_add_counts":ring_face_add_counts,
        "fixed_boundary_edges":len(boundary_edges),"fixed_boundary_components":len(bcomps),
        "contour_segments":len(pair_segments),"exact_endpoint_roots":len(new_roots),
        "contour_graph_degree_histogram":dict(Counter(contour_degree.values())),
        "contour_graph_components":len(contour_components),"contour_graph_component_vertex_counts":sorted(Counter(len(c) for c in contour_components).items()),"child_patch_faces":int(len(child_faces)),
        "parent_face_count":len(parent_faces),"contour_edges_unique":len(contour_global_edges),
        "outer_boundary_unchanged":new_boundary==old_boundary,
        "parent_face_area_coverage_pass":all(abs(x["reference_area"]-0.5)<1e-10 for x in per_face),
        "csr_ancestry":csr_summary,"csr_ancestry_all_valid":csr_valid,
        "distinct_exact_roots_collapsed_after_float32_emission":collision_groups,
        "exact_sub_ulp_collision_witnesses":collision_witnesses,
        "exact_degenerate_emitted_child_face_count":len(degenerate_child_faces),
        "exact_degenerate_emitted_child_face_rows_first_100":degenerate_child_faces[:100],
        "prior_endpoint_rounding_recheck":{"components_checked":f32check["components_checked_after_emission"],
          "skin_failures":f32check["endpoint_pair_checks_failing_skin_triangle"],
          "bone_failures":f32check["endpoint_pair_checks_failing_bone_triangle"],
          "distinct_roots_collapsed":f32check["distinct_exact_components_collapsed_to_one_Float32_point"]},
        "ancestry_sidecar_path":str(ancestry_sidecar),
        "ancestry_sidecar_sha256":sha(ancestry_sidecar),"patch_candidate_emitted":False}
    if not csr_valid: raise RuntimeError("source/fine CSR ancestry gate failed")
    if collision_groups: raise RuntimeError("distinct exact contour roots collapse to identical Float32 output coordinates")
    if degenerate_child_faces: raise RuntimeError("Float32-emitted child patch contains exactly degenerate faces")
    patch_path=OUT/"patch-candidate.npz"
    np.savez_compressed(
        patch_path, points_world_m_f32=out_points, patch_faces=child_faces,
        source_candidate_node_rows=parent_node_rows,
        point_parent_candidate_face_rows=point_parent_rows,
        point_parent_candidate_barycentric_f64=point_parent_bary,
        face_parent_candidate_rows=child_parent,
        face_parent_barycentric_f64=child_bary,
        face_source_ancestry_indptr=np.asarray(src_ptr,dtype=np.int64),
        face_source_ancestry_face_rows=np.asarray(src_rows,dtype=np.int64),
        face_fine_subface_ancestry_indptr=np.asarray(sub_ptr,dtype=np.int64),
        face_fine_subface_ancestry_rows=np.asarray(sub_rows,dtype=np.int64),
        face_fine_detail_ancestry_indptr=np.asarray(det_ptr,dtype=np.int64),
        face_fine_detail_ancestry_face_rows=np.asarray(det_rows,dtype=np.int64),
        patch_outer_boundary_edges=np.asarray(sorted(old_boundary),dtype=np.int64),
        contour_edges=np.asarray(contour_global_edges,dtype=np.int64).reshape(-1,2),
        original_patch_face_rows=np.asarray(parent_faces,dtype=np.int64),
        exact_endpoint_component_ids=np.asarray([-1]*len(patch_nodes)+new_roots,dtype=np.int64),
    )
    rational_path=OUT/"rational-ancestry.json"
    rational_path.write_text(json.dumps({"schema":"numi.human.bank43_16.rational-local-patch-ancestry.v1",
        "candidate_sha256":sha(C),"exact_predicate_sha256":sha(EXACT),
        "endpoint_nodes":exact_point_ancestry,"child_face_parent_barycentrics":exact_face_bary},indent=2)+"\n")
    report.update({
        "status":"LOCAL_PATCH_TOPOLOGY_PASS",
        "patch_candidate_sha256":sha(patch_path),
        "rational_ancestry_sha256":sha(rational_path),
        "counts":{"exact_crossing_pairs":len(pairs),"crossing_skin_faces":len(cross),
                  "retained_two_ring_parent_faces":len(parent_faces),"fixed_outer_boundary_edges":len(boundary_edges),
                  "fixed_boundary_components":len(bcomps),"original_patch_nodes":len(patch_nodes),
                  "exact_endpoint_keys":len(new_roots),"emitted_nodes_added_once_per_exact_key":len(new_roots),
                  "child_patch_faces":int(len(child_faces)),"contour_segments":len(pair_segments),
                  "contour_edges_unique":len(contour_global_edges),"local_edge_flips":flips_total},
        "checks":{"all_exact_endpoint_cycles_closed":all(d==2 for d in contour_degree.values()),"one_output_node_per_exact_endpoint_key":True,
                  "exact_rational_skin_barycentrics_preserved":True,"candidate_source_ancestry_inherited":True,
                  "fixed_outer_boundary_unchanged":True,"all_parent_faces_reference_area_covered":True,
                  "all_patch_edges_have_incidence_one_or_two":True,"all_contour_edges_have_incidence_two":True},
        "per_parent_face":per_face,
        "bank_clearance_checked":False,"skin_self_intersection_checked":False,
        "limitations":["Topology-only local patch trial; no crossing, containment, self-intersection, all-bank or native physical admission check.",
                       "All endpoint coordinates are independently emitted as Float32 world metres; this trial checks topology and face coverage, not bone clearance.",
                       "Bone source vertex indices are absent from the pinned triangle CSV; exact emitted coordinate keys stand in for explicit bone edge IDs."]
    })
except Exception as e:
    report.update({"status":"LOCAL_PATCH_TOPOLOGY_FAILED","failure":repr(e),"traceback":traceback.format_exc(),"elapsed_seconds":time.monotonic()-started})
report["script_sha256"]=sha(Path(__file__))
report["elapsed_seconds"]=time.monotonic()-started
(OUT/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
print(json.dumps({k:v for k,v in report.items() if k!="per_parent_face"},indent=2,allow_nan=False))
