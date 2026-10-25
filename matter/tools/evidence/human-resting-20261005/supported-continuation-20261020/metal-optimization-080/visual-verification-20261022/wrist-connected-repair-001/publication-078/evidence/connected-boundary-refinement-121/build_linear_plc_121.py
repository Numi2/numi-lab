from __future__ import annotations
import collections, hashlib, importlib.util, json, math, socket, sys, time, subprocess
from pathlib import Path

import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
OUTROOT = ROOT / "connected-boundary-refinement-121"
RUN = OUTROOT / "run-003"
REFERENCE = ROOT / "connected-capacity-reference-115/candidate.npz"
POLY115 = ROOT / "connected-layer-remesh-116/run-002/whole-skin-fat-reference.poly"
COMPACT = ROOT / "whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz"
TEMPLATE = ROOT / "connected-boundary-refinement-114/run-002-source-normal-all-146/surface-proposal.npz"
TEMPLATE_REPORT = TEMPLATE.with_name("proposal-report.json")
BATCH114 = ROOT / "digit-source-diagnosis-110/batch-146-curved-source-midpoints-quality-114.json"
TOOL121 = OUTROOT / "matter/tools/human_skin_fat_boundary_refinement_121.py"
TOOL114 = ROOT / "connected-boundary-refinement-114/matter/tools/human_skin_fat_boundary_refinement.py"
PREDICATE = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py")
PREDICATE_SRC = PREDICATE.parents[3] / "incremental-exact-audit-owner-253/src"
EXPECTED = {
    "reference115": "abff4290774cc96a17f6c68a21fe86fc3933a1878e0bd6512733e99589fb6dd9",
    "poly115": "20e005ac365f494361ac56a9699d0a0f3f41d054a55ed2ef77568fd00f392bb1",
    "compact084": "5a35470223ce12343cc217be7822d182802e99f1c423bb52b8409f153f9a49db",
    "template114_npz": "fdcb88d99dd488afb59b5f49ad3b6f5975f37fc46479f74dd8d31e24a0e5d5b8",
    "template114_report": "a636f31846c8a2f21ea543ca5028d2db6726aae8a146dd2d3026540149c45492",
    "batch114": "aa683f25a629f867d20decaa5a8222535142901c2807ba2361748605979b741a",
    "helper114": "a94f203efba2f0019ef1553a687b93798884e4b64b6d2434e5a690da42035d1d",
}

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def array_sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode("ascii"))
    h.update(json.dumps(list(a.shape), separators=(",", ":")).encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()

def cyclic_key(face):
    a, b, c = map(int, face)
    return min((a, b, c), (b, c, a), (c, a, b))

def read_poly(path):
    lines = [line.strip() for line in path.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if len(lines[0].split()) != 4:
        raise RuntimeError("unexpected .poly point header")
    n, dim, attrs, bmarkers = map(int, lines[0].split())
    if (n, dim, attrs, bmarkers) != (7233, 3, 0, 1):
        raise RuntimeError(f"unexpected 115 PLC point header {(n, dim, attrs, bmarkers)}")
    point_records = []
    for i in range(1, 1+n):
        q = lines[i].split()
        if len(q) != 5 or int(q[0]) != i:
            raise RuntimeError(f"malformed original .poly point record {i}")
        point_records.append((int(q[0]), np.asarray([float(x) for x in q[1:4]], np.float64), int(q[4])))
    fi = 1+n
    if len(lines[fi].split()) != 2 or lines[fi].split()[1] != "1":
        raise RuntimeError("unexpected 115 PLC facet header")
    nf = int(lines[fi].split()[0])
    fi += 1
    faces, markers = [], []
    for row in range(nf):
        facet = lines[fi].split()
        if len(facet) != 3 or facet[0:2] != ["1", "0"]:
            raise RuntimeError(f"unsupported facet record at row {row}")
        marker = int(facet[2])
        tri = lines[fi+1].split()
        if len(tri) != 4 or tri[0] != "3":
            raise RuntimeError(f"unsupported facet triangle at row {row}")
        ids = [int(v)-1 for v in tri[1:]]
        if min(ids) < 0 or max(ids) >= n or len(set(ids)) != 3:
            raise RuntimeError(f"invalid source facet vertices at row {row}")
        faces.append(ids); markers.append(marker); fi += 2
    return np.asarray([r[1] for r in point_records]), np.asarray([r[2] for r in point_records], np.int64), np.asarray(faces, np.int32), np.asarray(markers, np.int32), lines[fi:]

def edge_incidence(faces):
    use = collections.defaultdict(list)
    for f in np.asarray(faces, np.int64):
        a,b,c=map(int,f)
        for u,v in ((a,b),(b,c),(c,a)):
            use[tuple(sorted((u,v)))].append((u,v))
    counts=collections.Counter(len(v) for v in use.values())
    orientation_conflicts=sum(1 for v in use.values() if len(v)==2 and v[0] != v[1][::-1])
    boundary=sum(1 for v in use.values() if len(v)==1)
    nonmanifold=sum(1 for v in use.values() if len(v)>2)
    return {"unique_edges":len(use),"incidence_histogram":{str(k):v for k,v in sorted(counts.items())},
            "boundary_edges":boundary,"nonmanifold_edges":nonmanifold,
            "two_face_orientation_conflicts":orientation_conflicts}

def main():
    if RUN.exists():
        raise RuntimeError(f"refusing existing output directory {RUN}")
    pin_paths = {
        "reference115":REFERENCE, "poly115":POLY115, "compact084":COMPACT,
        "template114_npz":TEMPLATE, "template114_report":TEMPLATE_REPORT,
        "batch114":BATCH114, "helper114":TOOL114,
    }
    actual={k:sha(v) for k,v in pin_paths.items()}
    if actual != EXPECTED:
        raise RuntimeError(f"frozen input pins changed: {actual}")
    if sha(TOOL121) != "1e5b46ad79681d7c22f7f4635bbb171746a826d961aedfff7fe956b5971a30c8":
        raise RuntimeError("isolated 121 subdivision helper changed")
    if not PREDICATE.exists(): raise RuntimeError("exact predicate source missing")
    pred_sha=sha(PREDICATE)

    r=np.load(REFERENCE,allow_pickle=False)
    compact=np.load(COMPACT,allow_pickle=False)
    template=np.load(TEMPLATE,allow_pickle=False)
    poly_points, point_markers, source_poly_faces, source_poly_markers, poly_tail=read_poly(POLY115)
    ref=np.asarray(r["proposed_reference_points_m"][:7233],dtype=np.float64)
    if ref.shape!=(7233,3) or not np.array_equal(ref,poly_points):
        raise RuntimeError("115 reference prefix differs from exact whole-115 PLC points")
    if not np.array_equal(np.asarray(r["source_reference_points_m"][:7233]),np.asarray(compact["reference_points_m"][:7233])):
        raise RuntimeError("115 source reference differs from pinned 084 source reference")
    template_prefix=np.asarray(template["reference_points_m"][:7360])
    if not np.array_equal(template_prefix,np.asarray(r["proposed_reference_points_m"])):
        raise RuntimeError("114 topology template does not share exact 115 reference prefix")

    # Recover each source sheet in exact facet-marker parent-row order.
    faces_by_layer={}; source_poly_by_layer={}; markers_by_layer={}; winding_parity_by_layer={}
    for li, layer in enumerate(("O","I","D"),start=1):
        code=li*100000
        idx=np.flatnonzero((source_poly_markers>code)&(source_poly_markers<code+100000))
        rows=source_poly_markers[idx]-code-1
        if len(idx)!=4777 or not np.array_equal(np.sort(rows),np.arange(4777)):
            raise RuntimeError(f"{layer} 115 marker coverage is incomplete")
        ordered=np.empty((4777,3),np.int32)
        for ix,row in zip(idx,rows):
            ordered[int(row)]=source_poly_faces[int(ix)]
        source=np.asarray(compact[f"source_{layer}_faces"],np.int32)
        def orient_cycle(face):
            a,b,c=map(int,face)
            return min((a,b,c),(b,c,a),(c,a,b))
        def winding_parity(face, other):
            if orient_cycle(face)==orient_cycle(other): return 1
            if orient_cycle(face)==orient_cycle(np.asarray(other)[::-1]): return -1
            return 0
        parity=np.asarray([winding_parity(a,b) for a,b in zip(ordered,source)],dtype=np.int8)
        if np.any(parity==0):
            raise RuntimeError(f"{layer} 115 source-face marker mapping disagrees with 084 parent vertex sets")
        if len(np.unique(parity))!=1:
            raise RuntimeError(f"{layer} 115 parent-face winding is inconsistent relative to 084")
        # Subdivide with 084's oriented rows to reproduce 114 combinatorics,
        # then reverse every child whose marker-matched 115 parent was reversed.
        source_poly_by_layer[layer]=ordered
        faces_by_layer[layer]=source.copy()
        winding_parity_by_layer[layer]=int(parity[0])
        markers_by_layer[layer]=code+1+np.arange(len(ordered),dtype=np.int32)
    cap_idx=np.flatnonzero(source_poly_markers>=400001)
    cap_by_marker={int(source_poly_markers[i]):source_poly_faces[i].copy() for i in cap_idx}
    cap_markers=np.asarray(template["cap_face_markers"],dtype=np.int32)
    if len(cap_markers)!=180 or len(np.unique(cap_markers))!=180 or set(map(int,cap_markers))!=set(cap_by_marker):
        raise RuntimeError("115 cap marker inventory differs from 114 template")
    cap_faces=np.asarray([cap_by_marker[int(marker)] for marker in cap_markers],dtype=np.int32)
    cap_parent_rows=np.asarray(template["cap_face_parent_rows"],dtype=np.int64)

    edges=np.asarray(template["selected_outer_edges"],dtype=np.int64)
    layer_edges=np.asarray(template["selected_edge_layer_endpoint_rows"],dtype=np.int64)
    old_midrows=np.asarray(template["selected_edge_midpoint_rows"],dtype=np.int64)
    if edges.shape!=(146,2) or layer_edges.shape!=(146,3,2) or old_midrows.shape!=(146,3):
        raise RuntimeError("114 selected-edge row map dimensions changed")
    if not np.array_equal(np.sort(edges,axis=1),np.sort(layer_edges[:,0,:],axis=1)):
        raise RuntimeError("114 O edge list and explicit O endpoint map disagree")
    if len(np.unique(layer_edges.reshape(-1,2),axis=0))!=438:
        raise RuntimeError("selected O/I/D edge pairs are not unique")
    if len(np.unique(old_midrows))!=438 or set(map(int,old_midrows.ravel()))!=set(range(7360,7798)):
        raise RuntimeError("114 template midpoint row identity changed")
    if np.any(layer_edges<0) or np.any(layer_edges>=7233):
        raise RuntimeError("selected edge endpoint outside 115 source PLC node domain")

    # The 114 candidate supplies only topology, parent rows, and explicit endpoint maps.
    # Its 438 registered curved sample coordinates are intentionally not reused.
    original_edge_checks={}
    for li,layer in enumerate(("O","I","D")):
        incidence=collections.defaultdict(list)
        for fi,f in enumerate(source_poly_by_layer[layer]):
            a,b,c=map(int,f)
            for u,v in ((a,b),(b,c),(c,a)):
                incidence[tuple(sorted((u,v)))].append((fi,u,v))
        cap_edge_set={tuple(sorted((int(t[j]),int(t[(j+1)%3])))) for t in cap_faces for j in range(3)}
        layer_edge_keys=[]
        for ei,(a,b) in enumerate(layer_edges[:,li,:]):
            key=tuple(sorted((int(a),int(b))))
            uses=incidence.get(key,[])
            if len(uses)!=2 or uses[0][1:]!=uses[1][1:][::-1]:
                raise RuntimeError(f"{layer} selected edge {ei} lacks two oppositely oriented source faces")
            if key in cap_edge_set:
                raise RuntimeError(f"{layer} selected edge {ei} is also a side-cap edge")
            layer_edge_keys.append(key)
        if len(set(layer_edge_keys))!=146:
            raise RuntimeError(f"{layer} selected source edge duplicates were found")
        original_edge_checks[layer]={"selected_edges":len(layer_edge_keys),"two_face_oppositely_oriented":True,"cap_edge_overlap":0}
    # New row order is edge order from the pinned 114 map, with O/I/D rows per edge.
    new_midrows=(7233+np.arange(438,dtype=np.int64).reshape(146,3))
    midpoints=np.empty((146,3,3),np.float64)
    linear_records={}
    reference_pin=hashlib.sha256(np.ascontiguousarray(ref).tobytes()).hexdigest()
    midpoint_deviations={layer:[] for layer in ("O","I","D")}
    for ei in range(146):
        for li,layer in enumerate(("O","I","D")):
            a,b=map(int,layer_edges[ei,li])
            value=(ref[a]+ref[b])*0.5
            midpoints[ei,li]=value
            old=int(old_midrows[ei,li])
            midpoint_deviations[layer].append(float(np.linalg.norm(value-template["reference_points_m"][old])))
            edge=tuple(sorted((a,b)))
            if edge in linear_records.setdefault(layer,{}):
                raise RuntimeError(f"duplicate {layer} edge record")
            linear_records[layer][edge]={
                "source_kind":"registered_linear_midpoint","layer":layer,
                "endpoint_rows":[a,b],"parameter":0.5,
                "sample_point_world_m":value.tolist(),
                "reference_field_sha256":reference_pin,
            }
    points=np.concatenate((ref,midpoints.reshape(438,3)),axis=0)
    if points.shape!=(7671,3) or points.dtype!=np.float64:
        raise RuntimeError("new PLC point array shape/precision changed")
    if not np.array_equal(points[:7233],ref):
        raise RuntimeError("original source coordinates were modified")
    edge_midpoint_maps={}
    for li,layer in enumerate(("O","I","D")):
        edge_midpoint_maps[layer]={tuple(sorted(map(int,edge))):int(mid) for edge,mid in zip(layer_edges[:,li,:],new_midrows[:,li])}
        if len(edge_midpoint_maps[layer])!=146:
            raise RuntimeError(f"{layer} midpoint row map incomplete")

    sys.path.insert(0,str(OUTROOT/"matter/tools"))
    from human_skin_fat_boundary_refinement_121 import subdivide_oriented_surface_edges
    layer_results={}
    for li,layer in enumerate(("O","I","D")):
        rowmap=edge_midpoint_maps[layer]
        expected_rows=np.asarray(new_midrows[:,li],dtype=np.int64)
        layer_results[layer]=subdivide_oriented_surface_edges(
            points,faces_by_layer[layer],rowmap,linear_records[layer],
            original_point_count=7233,parent_face_markers=markers_by_layer[layer],
            layer=layer,expected_midpoint_rows=expected_rows)
        if not np.array_equal(layer_results[layer].child_parent_face_markers,
                              markers_by_layer[layer][layer_results[layer].child_parent_face_rows]):
            raise RuntimeError(f"{layer} child parent marker propagation failed")

    output_layer_faces={}
    for layer in ("O","I","D"):
        child_faces=layer_results[layer].faces.copy()
        if winding_parity_by_layer[layer] == -1:
            child_faces=child_faces[:,[0,2,1]]
        output_layer_faces[layer]=child_faces
    all_faces=np.concatenate([output_layer_faces[x] for x in ("O","I","D")]+[cap_faces],axis=0).astype(np.int32)
    all_markers=np.concatenate([layer_results[x].child_parent_face_markers for x in ("O","I","D")]+[cap_markers]).astype(np.int32)
    parent_rows=np.concatenate([layer_results[x].child_parent_face_rows for x in ("O","I","D")]+[cap_parent_rows]).astype(np.int64)
    surface_ids=np.concatenate([np.full(len(layer_results[x].faces),i,dtype=np.int8) for i,x in enumerate(("O","I","D"))]+[np.full(len(cap_faces),3,np.int8)])
    if all_faces.shape!=(15387,3) or len(all_markers)!=len(all_faces) or np.any(all_faces<0) or np.any(all_faces>=len(points)):
        raise RuntimeError("refined full PLC facet table is malformed")

    # Match the separately frozen 114 topology template after replacing only its midpoint row IDs.
    old_to_new={int(old):int(new) for old,new in zip(old_midrows.ravel(),new_midrows.ravel())}
    tf=np.asarray(template["surface_faces_zero_based"],dtype=np.int64)
    remapped=tf.copy()
    for old,new in old_to_new.items(): remapped[tf==old]=new
    def face_counter(face_array,marker_array):
        # Topology comparison is unoriented: the exact .poly parent winding is
        # retained, while the 114 combinatoric template may globally reverse O.
        return collections.Counter((int(m),tuple(sorted(map(int,f)))) for f,m in zip(face_array,marker_array))
    if face_counter(remapped,template["surface_face_markers"]) != face_counter(all_faces,all_markers):
        raise RuntimeError("121 linear midpoint subdivision differs from pinned 114 tri1/2/3 topology")
    if not np.array_equal(template["surface_face_surface_ids"],surface_ids):
        raise RuntimeError("114 topology template surface identity/order changed")
    if not np.array_equal(template["surface_face_markers"],all_markers):
        raise RuntimeError("121 output marker sequence changed from 114 topology template")
    if not np.array_equal(template["surface_face_parent_rows"],parent_rows):
        raise RuntimeError("121 face-parent row sequence changed from 114 topology template")

    # Parent area/orientation are invariant within binary64 midpoint rounding.
    parent_area={}
    max_area_rel=0.0; min_child_cos=1.0
    for li,layer in enumerate(("O","I","D")):
        source=source_poly_by_layer[layer]; result=layer_results[layer]
        pc=points[source]
        parent_area2=np.linalg.norm(np.cross(pc[:,1]-pc[:,0],pc[:,2]-pc[:,0]),axis=1)
        cc=points[output_layer_faces[layer]]
        child_area2=np.linalg.norm(np.cross(cc[:,1]-cc[:,0],cc[:,2]-cc[:,0]),axis=1)
        sums=np.bincount(result.child_parent_face_rows,weights=child_area2,minlength=len(source))
        rel=np.abs(sums-parent_area2)/np.maximum(parent_area2,np.finfo(float).tiny)
        dots=np.einsum("ij,ij->i",np.cross(cc[:,1]-cc[:,0],cc[:,2]-cc[:,0]),
                       np.cross(pc[result.child_parent_face_rows,1]-pc[result.child_parent_face_rows,0],
                                pc[result.child_parent_face_rows,2]-pc[result.child_parent_face_rows,0]))
        cos=dots/np.maximum(child_area2*parent_area2[result.child_parent_face_rows],np.finfo(float).tiny)
        max_area_rel=max(max_area_rel,float(np.max(rel)))
        min_child_cos=min(min_child_cos,float(np.min(cos)))
        parent_area[layer]={"max_relative_area_partition_error":float(np.max(rel)),
                            "minimum_child_normal_cosine":float(np.min(cos)),
                            "all_parent_rows_covered":bool(np.array_equal(np.unique(result.child_parent_face_rows),np.arange(len(source))))}
    if max_area_rel>2e-12 or min_child_cos<=0.0:
        raise RuntimeError(f"linear split changes parent area/orientation: {max_area_rel}/{min_child_cos}")

    before_chain=edge_incidence(source_poly_faces)
    after_chain=edge_incidence(all_faces)
    # Interior edge subdivision must preserve the complete PLC incidence/orientation ledger.
    for key in ("boundary_edges","nonmanifold_edges","two_face_orientation_conflicts"):
        if before_chain[key]!=after_chain[key]:
            raise RuntimeError(f"PLC boundary incidence changed at {key}: {before_chain} -> {after_chain}")

    # Current037 midpoint field is saved as a separate linear interpolation, not as a source curve.
    cur=np.asarray(r["points_world_f32"][:7233],dtype=np.float32)
    cur_mids=np.empty((146,3,3),np.float32)
    for ei in range(146):
        for li in range(3):
            a,b=map(int,layer_edges[ei,li])
            cur_mids[ei,li]=((cur[a].astype(np.float64)+cur[b].astype(np.float64))*0.5).astype(np.float32)
    current_points=np.concatenate((cur,cur_mids.reshape(438,3)),axis=0)
    if current_points.shape!=(7671,3) or current_points.dtype!=np.float32:
        raise RuntimeError("current037 linear midpoint field shape changed")
    old_template_midpoints=template["reference_points_m"][old_midrows]
    max_curve_delta={layer:float(max(midpoint_deviations[layer])) for layer in ("O","I","D")}
    max_reference_midpoint_line_error_m=0.0
    for ei in range(146):
        for li in range(3):
            a,b=map(int,layer_edges[ei,li]); m=points[int(new_midrows[ei,li])]
            e=ref[b]-ref[a]
            den=float(e@e)
            t=float((m-ref[a])@e/den) if den else float("nan")
            projected=ref[a]+t*e
            max_reference_midpoint_line_error_m=max(max_reference_midpoint_line_error_m,float(np.linalg.norm(m-projected)))

    RUN.mkdir(parents=True)
    poly_out=RUN/"whole-skin-fat-reference-121.poly"
    out_lines=["# 121: linear registered edge subdivision; original 115 PLC reference points unchanged.",
               "7671 3 0 1"]
    for i,p in enumerate(points,1):
        marker=int(point_markers[i-1]) if i<=7233 else i
        out_lines.append(f"{i} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g} {marker}")
    out_lines.append(f"{len(all_faces)} 1")
    for tri,marker in zip(all_faces,all_markers):
        out_lines.append(f"1 0 {int(marker)}")
        out_lines.append("3 "+" ".join(str(int(v)+1) for v in tri))
    out_lines.extend(poly_tail)
    poly_out.write_text("\n".join(out_lines)+"\n")
    candidate=RUN/"candidate.npz"
    np.savez_compressed(
        candidate, points_reference_m=points, points_current037_f32=current_points,
        surface_faces_zero_based=all_faces, surface_face_markers=all_markers,
        surface_face_parent_rows=parent_rows, surface_face_surface_ids=surface_ids,
        source_O_faces=output_layer_faces["O"], source_I_faces=output_layer_faces["I"],
        source_D_faces=output_layer_faces["D"],
        source_O_child_parent_rows=layer_results["O"].child_parent_face_rows,
        source_I_child_parent_rows=layer_results["I"].child_parent_face_rows,
        source_D_child_parent_rows=layer_results["D"].child_parent_face_rows,
        selected_edge_source_column_ids=template["selected_edge_global_column_ids"],
        selected_edge_layer_endpoint_rows=layer_edges,
        selected_edge_midpoint_rows=new_midrows,
        midpoint_source_kind=np.asarray(["registered_linear_midpoint"]),
        cap_faces_zero_based=cap_faces, cap_face_markers=cap_markers,
        original_115_point_count=np.asarray([7233],dtype=np.int32),
    )
    pred_dir=PREDICATE_SRC
    if not pred_dir.exists(): raise RuntimeError("exact predicate source import root missing")
    sys.path.insert(0,str(pred_dir))
    spec=importlib.util.spec_from_file_location("numilab_human._exact_plc_121",PREDICATE)
    if spec is None or spec.loader is None: raise RuntimeError("unable to load frozen exact self-audit owner")
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    audit_points=points.astype(np.float32)
    started=time.monotonic()
    records,_=module.float32_surface_records(audit_points,all_faces,source_seam_groups=())
    self_audit=module._audit_self_pairs_touching_faces(records,set(range(len(all_faces))))
    pairs=np.asarray(self_audit["triangle_pairs"],dtype=np.int64).reshape(-1,2)
    np.save(RUN/"exact-self-pairs.npy",pairs)
    self_audit_seconds=time.monotonic()-started
    report={
        "status":"PLC_SELF_GATE_PASS" if len(pairs)==0 and self_audit.get("complete_surface_audit") else "PLC_SELF_GATE_FAIL",
        "scope":"Actual 115 reference PLC, topology-only linear midpoint split on 146 explicitly mapped O/I/D edges (438 new points).",
        "interpretation":"Construction/refinement input only. No TetGen or cell-quality, material-transfer, bone, native, or physical admission is claimed.",
        "host":socket.gethostname(),
        "input_sha256":{**actual,"helper121":sha(TOOL121),"predicate315":pred_sha},
        "inputs":{"reference_field":"115 candidate.npz::proposed_reference_points_m[:7233]",
                  "topology_endpoint_map":"114 run-002 surface-proposal.npz::selected_edge_layer_endpoint_rows",
                  "face_topology_template":"114 run-002 surface-proposal.npz; only child connectivity/parent marker ancestry reused",
                  "exact_base_PLC":"116 run-002 whole-skin-fat-reference.poly"},
        "counts":{"source_points":7233,"new_linear_midpoints":438,"output_points":len(points),
                  "selected_edges":146,"selected_edges_by_layer":{"O":146,"I":146,"D":146},
                  "output_facets":len(all_faces),"facets_by_surface":{k:int(np.count_nonzero(surface_ids==i)) for i,k in enumerate(("O","I","D","side_caps"))}},
        "edge_maps":original_edge_checks,
        "source_face_winding_against_084":winding_parity_by_layer,
        "linear_midpoint":{"parameter":0.5,"source_claim":"registered linear chord midpoint only; no detailed-skin curve claim",
                           "maximum_delta_from_114_registered_source_samples_m":max_curve_delta,
                           "maximum_F64_distance_from_source_edge_line_m":max_reference_midpoint_line_error_m,
                           "original_115_prefix_bitwise_unchanged":bool(np.array_equal(points[:7233],ref))},
        "parent_geometry":{"per_layer":parent_area,"maximum_relative_area_partition_error":max_area_rel,
                           "minimum_child_normal_cosine":min_child_cos,
                           "all_146_edges_two_face_oppositely_oriented_by_layer":True,
                           "selected_edges_on_side_caps":0,"source_chain_before":before_chain,"source_chain_after":after_chain},
        "face_provenance":{"parent_face_rows_and_original_markers_preserved":True,
                           "same_unoriented_tri1_tri2_tri3_connectivity_as_114_after_midpoint_row_remap":True,
                           "exact_115_parent_winding_preserved_per_child":True,
                           "template_orientation_difference_is_only_uniform_per_layer":winding_parity_by_layer},
        "exact_self_gate":{"predicate":"315 exact rational triangle classifier on emitted Float32 coordinates, complete all-facet pair query",
                           "predicate_source":str(PREDICATE),"predicate_sha256":pred_sha,
                           "seconds":self_audit_seconds,**{k:v for k,v in self_audit.items() if k not in ("triangle_pairs",)},
                           "pair_count":int(len(pairs)),"pair_file":"exact-self-pairs.npy"},
        "poly_sha256":sha(poly_out),"candidate_sha256":sha(candidate),
        "TetGen":{"status":"not_run_pending_parent_review","requested_command":"-pq1.2/10AS512"},
    }
    if report["status"]!="PLC_SELF_GATE_PASS":
        report["TetGen"]={"status":"not_run_self_gate_failed","requested_command":"-pq1.2/10AS512"}
        (RUN/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        print(json.dumps(report,indent=2,sort_keys=True))
        raise SystemExit(3)

    # Geometry-only TetGen diagnostic, only after the complete exact PLC self gate.
    tetgen=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/arm-local-pocket-remesh-116/local-remesh-118/tetgen-v1.6.0/tetgen")
    if not tetgen.is_file():
        raise RuntimeError(f"pinned TetGen binary is missing: {tetgen}")
    cmd=[str(tetgen),"-pq1.2/10AS512",str(poly_out)]
    start_tet=time.monotonic()
    try:
        cp=subprocess.run(cmd,cwd=RUN,capture_output=True,text=True,timeout=45)
        rc=cp.returncode; stdout=cp.stdout; stderr=cp.stderr; timed_out=False
    except subprocess.TimeoutExpired as exc:
        rc=None; stdout=exc.stdout.decode() if isinstance(exc.stdout,bytes) else (exc.stdout or "")
        stderr=exc.stderr.decode() if isinstance(exc.stderr,bytes) else (exc.stderr or "")
        timed_out=True
    (RUN/"tetgen.stdout.log").write_text(stdout)
    (RUN/"tetgen.stderr.log").write_text(stderr)
    tetrec={"status":"timeout" if timed_out else ("completed" if rc==0 else "nonzero_exit"),
            "command":cmd,"binary_sha256":sha(tetgen),"return_code":rc,"timed_out":timed_out,
            "wall_seconds":time.monotonic()-start_tet}
    mesh_npz=None; current_mesh=None
    node_path=poly_out.with_suffix(".1.node"); ele_path=poly_out.with_suffix(".1.ele")
    if rc==0 and node_path.exists() and ele_path.exists():
        def read_table(path):
            rows=[]
            for line in path.read_text().splitlines():
                line=line.split("#",1)[0].strip()
                if line: rows.append(line.split())
            return rows
        nr=read_table(node_path); er=read_table(ele_path)
        nh,dim,nattr,nmark=map(int,nr[0][:4])
        if dim!=3 or nh!=len(nr)-1: raise RuntimeError("TetGen node table header/count mismatch")
        outids=np.asarray([int(row[0]) for row in nr[1:]],np.int64)
        if not np.array_equal(outids,np.arange(1,nh+1)): raise RuntimeError("TetGen node identifiers are not dense one-based rows")
        outref=np.asarray([[float(x) for x in row[1:4]] for row in nr[1:]],np.float64)
        eh,ncorn,eattr=map(int,er[0][:3])
        if ncorn!=4 or eh!=len(er)-1: raise RuntimeError("TetGen element table header/count mismatch")
        outT=np.asarray([[int(x)-1 for x in row[1:5]] for row in er[1:]],np.int64)
        if outT.min()<0 or outT.max()>=len(outref): raise RuntimeError("TetGen element references invalid node row")
        outmat=np.asarray([int(float(row[5])) if eattr else 0 for row in er[1:]],np.int32)
        from collections import Counter
        tetrec.update({"node_count":len(outref),"tet_count":len(outT),"material_id_counts":{str(k):int(v) for k,v in Counter(outmat).items()}})
        # Reference-domain positivity and scaled-volume quality are auditable without
        # claiming the separate transferred current037 field for any Steiner rows.
        xyz=outref[outT]
        signed=np.einsum("ij,ij->i",xyz[:,1]-xyz[:,0],np.cross(xyz[:,2]-xyz[:,0],xyz[:,3]-xyz[:,0]))/6.0
        tetrec["reference_domain_positive_cells"]=int(np.count_nonzero(signed>0))
        tetrec["reference_domain_nonpositive_cells"]=int(np.count_nonzero(signed<=0))
        tetrec["reference_domain_min_signed_volume_m3"]=float(np.min(signed))
        if np.all(signed>0):
            sys.path.insert(0,str(ROOT/"connected-cell-trust-111/matter/tools"))
            from human_skin_fat_connected_repair import cell_values_gradients
            Xref32=outref.astype(np.float32)
            Jref,_,qref,_=cell_values_gradients(Xref32,outref,outT)
            tetrec.update({"reference_identity_emitted_F32_invalid_J":int(np.count_nonzero(Jref<=0)),
                           "reference_identity_emitted_F32_q_below_1e_minus5":int(np.count_nonzero(qref<1e-5)),
                           "reference_identity_emitted_F32_q_min":float(np.min(qref)),
                           "reference_identity_emitted_F32_J_min":float(np.min(Jref)),
                           "reference_identity_emitted_F32_J_max":float(np.max(Jref))})
        plclock={tuple(map(float,x)):i for i,x in enumerate(points)}
        plcrow=np.asarray([plclock.get(tuple(map(float,x)),-1) for x in outref],np.int64)
        exact_all=bool(np.all(plcrow>=0) and len(set(map(int,plcrow)))==len(plcrow))
        tetrec["all_tetgen_nodes_exactly_map_to_121_PLC"]=exact_all
        if exact_all:
            cur_out=current_points[plcrow]
            plc_to_out=np.empty(len(points),dtype=np.int64);plc_to_out[plcrow]=np.arange(len(outref),dtype=np.int64)
            out_faces=plc_to_out[all_faces]
            mesh_npz=RUN/"tetgen-mesh.npz"
            np.savez_compressed(mesh_npz,reference_points_m=outref,points_world_f32=cur_out.astype(np.float32),
                                tetrahedra=outT,material_ids=outmat,output_node_ids=outids,
                                plc_input_row=plcrow,surface_faces_zero_based=out_faces,
                                surface_face_markers=all_markers,surface_face_parent_rows=parent_rows)
            current_mesh={"status":"mapped_by_exact_PLC_vertex_identity","candidate_npz":str(mesh_npz),"candidate_sha256":sha(mesh_npz)}
            Jcur,_,qcur,_=cell_values_gradients(cur_out.astype(np.float32),outref,outT)
            current_mesh.update({"invalid_cells":int(np.count_nonzero(Jcur<=0)),
                                 "q_below_1e_minus5":int(np.count_nonzero(qcur<1e-5)),
                                 "q_min":float(np.min(qcur)),"J_min":float(np.min(Jcur)),"J_max":float(np.max(Jcur))})
        else:
            current_mesh={"status":"not_mapped_no_qualified_Steiner_ancestry",
                          "unmapped_output_node_count":int(np.count_nonzero(plcrow<0)),
                          "interpretation":"Current037 values are known only at exact PLC input vertices; no nearest or guessed interpolation is used."}
        tetrec["mesh_candidate"]=current_mesh
    report["TetGen"]=tetrec
    report["status"]="PLC_SELF_GATE_PASS_TETGEN_DIAGNOSTIC"
    (RUN/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))

if __name__=="__main__": main()

