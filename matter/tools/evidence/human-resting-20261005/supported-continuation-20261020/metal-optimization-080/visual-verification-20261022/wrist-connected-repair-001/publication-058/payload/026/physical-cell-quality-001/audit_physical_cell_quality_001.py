#!/usr/bin/env python3
"""Read-only D37 physical-cell quality and local palm-prism split diagnostic."""
import csv, hashlib, itertools, json
from pathlib import Path
import numpy as np

ROOT = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
OUT = ROOT / "native-candidate020-failure-witness-026" / "physical-cell-quality-001"
D37 = ROOT / "wholebody-attachment-source-rebind-037-candidate020-identity004-002" / "generated"
COORDS = ROOT / "native-candidate020-failure-witness-026" / "run-001" / "initial-skin-fem-nodes.csv"
WITNESS = ROOT / "skin-contact-backtracking-diagnostic-145" / "initial-witness-146" / "native-source-only-001" / "initial-tet-29419.json"
FAIL = ROOT / "skin-contact-backtracking-diagnostic-145" / "failure-analysis.json"
POSES = ROOT / "native-candidate020-failure-witness-026" / "run-001" / "registered-skin-poses.json"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def cell_metrics(points, tet_rows):
    X = points[tet_rows].astype(np.float64)
    signed_det = np.einsum("ij,ij->i", X[:,1]-X[:,0], np.cross(X[:,2]-X[:,0], X[:,3]-X[:,0]))
    edge = np.stack([np.linalg.norm(X[:,i]-X[:,j], axis=1)
                     for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))], axis=1)
    longest = edge.max(axis=1)
    q = np.abs(signed_det) / longest**3
    face2 = np.stack([np.linalg.norm(np.cross(X[:,b]-X[:,a], X[:,c]-X[:,a]),axis=1)
                      for a,b,c in ((1,2,3),(0,2,3),(0,1,3),(0,1,2))], axis=1)
    altitude = np.abs(signed_det[:,None]) / face2
    aspect = longest / altitude.min(axis=1)
    return signed_det, q, longest, altitude, aspect

def boundary_faces(tets):
    counts = {}
    for t in tets:
        a,b,c,d = map(int,t)
        for f in ((a,b,c),(a,b,d),(a,c,d),(b,c,d)):
            key=tuple(sorted(f)); counts[key]=counts.get(key,0)+1
    return {k for k,v in counts.items() if v==1}

def prism_side_diagonals(tets, face_vertices):
    """Return internal-prism quad diagonal per source-face edge, inferred from boundary triangles."""
    I=[2411+int(v) for v in face_vertices]
    D=[4822+int(v) for v in face_vertices]
    bfaces=boundary_faces(tets)
    result={}
    for i,j in ((0,1),(1,2),(2,0)):
        quad={I[i],I[j],D[i],D[j]}
        tris=[set(f) for f in bfaces if set(f)<=quad]
        if len(tris)!=2:
            result[f"{face_vertices[i]}-{face_vertices[j]}"]={"triangles":len(tris),"diagonal":None}
            continue
        diag=tuple(sorted(tris[0]&tris[1]))
        result[f"{face_vertices[i]}-{face_vertices[j]}"]={"triangles":2,"diagonal":diag}
    return result

tets_csv=rows(D37/"tetrahedra.csv")
nodes_csv=rows(D37/"nodes.csv")
donor_csv=rows(D37/"donor-frames-fullweights.csv")
embed_csv=rows(D37/"embedding.csv")
faces_csv=rows(D37/"faces.csv")
coord_csv=rows(COORDS)
T=np.asarray([[int(r[k]) for k in ("a","b","c","d")] for r in tets_csv],dtype=np.int64)
global_ids=np.asarray([int(r["global_node"]) for r in coord_csv],dtype=np.int64)
if len(coord_csv)!=7233 or not np.array_equal(global_ids, np.arange(1666,1666+7233)):
    raise RuntimeError("compiled-coordinate node domain does not match pinned D37 local node IDs")
points=np.asarray([[float(r[k]) for k in ("x_m","y_m","z_m")] for r in coord_csv],dtype=np.float32)
if int(T.max())>=len(points):
    raise RuntimeError("D37 tet references missing compiled coordinate row")
signed_det,q,longest,altitude,aspect=cell_metrics(points,T)
materials=np.asarray([int(r["material"]) for r in tets_csv])
donor_rows=np.asarray([int(r["donor"]) for r in donor_csv])
node_donors=donor_rows[T%2411]
modal_donor=np.asarray([min(np.unique(d,return_counts=True)[0][np.unique(d,return_counts=True)[1]==np.unique(d,return_counts=True)[1].max()])
                        for d in node_donors],dtype=np.int64)
# Exact D37-to-native object row correspondence from captured witness.
wit=json.loads(WITNESS.read_text())
local=int(wit["local_tet"]); global_tet=int(wit["global_tet"]); elem_offset=int(wit["object_element_offset"])
if global_tet-elem_offset != local or local>=len(tets_csv):
    raise RuntimeError("native global/object-local tet mapping is inconsistent")
if int(tets_csv[local]["tet"]) != local:
    raise RuntimeError("D37 tet table row identity is not its local row")
row=tets_csv[local]
if [int(row[k]) for k in ("a","b","c","d")] != list(map(int,wit["nodes_local"])):
    raise RuntimeError("D37 cell node IDs differ from captured native witness")
if (int(row["parent_tet"]),int(row["parent_surface_face"]),int(row["stable_source_face"])) != (25186,3776,89542):
    raise RuntimeError("unexpected D37 source ancestry for failing cell")
# The JSON witness stores coordinates directly; check whether its *_det_m3 field is determinant or volume.
WX=np.asarray(wit["current_xyz_m"],dtype=np.float64)
wdet=float(np.dot(WX[1]-WX[0],np.cross(WX[2]-WX[0],WX[3]-WX[0])))
wvol=abs(wdet)/6.0
# Per-material quality summary and deterministic minimum rows.
material_summary={}
for m in sorted(set(materials.tolist())):
    ix=np.flatnonzero(materials==m)
    k=int(ix[np.argmin(q[ix])])
    material_summary[str(m)]={"count":int(len(ix)),"q_min":float(q[k]),"q_p01":float(np.quantile(q[ix],.01)),
        "q_median":float(np.median(q[ix])),"min_local_tet_row":k,"min_signed_det_m3":float(signed_det[k]),
        "min_tet_volume_m3":float(abs(signed_det[k])/6),"min_longest_edge_mm":float(longest[k]*1000),
        "min_aspect_L_over_min_altitude":float(aspect[k])}
# Donor-grouped distribution uses deterministic modal donor across the 4 tet vertices; it is a grouping key, not an anatomical-region label.
donor_summary=[]
for d in sorted(set(modal_donor.tolist())):
    ix=np.flatnonzero(modal_donor==d)
    if len(ix)<10: continue
    k=int(ix[np.argmin(q[ix])])
    donor_summary.append({"modal_donor_id":int(d),"cell_count":int(len(ix)),
        "skin_cell_count":int(np.sum(materials[ix]==1)),"fat_cell_count":int(np.sum(materials[ix]==2)),
        "q_min":float(q[k]),"q_p01":float(np.quantile(q[ix],.01)),"q_median":float(np.median(q[ix])),
        "min_local_tet_row":k,"q_below_1e-4_count":int(np.sum(q[ix]<1e-4))})
donor_summary.sort(key=lambda r:(r["q_min"],r["modal_donor_id"]))
# Source/layer ancestry and thickness columns at the failed cell.
node_info=[]
for n in T[local]:
    layer=int(n//2411); rr=int(n%2411); weights=np.asarray([float(donor_csv[rr]["w"+str(i)]) for i in range(86)])
    node_info.append({"volume_node":int(n),"layer":["O","I","D"][layer],"reduced_row":rr,
        "source_vertex_id":int(embed_csv[rr]["source_vertex"]),"donor_body_id":int(donor_csv[rr]["donor"]),
        "max_weight_lane":int(np.argmax(weights)),"max_weight":float(weights.max()),
        "skin_thickness_mm":float(nodes_csv[rr]["skin_thickness_m"])*1000,
        "fat_thickness_mm":float(nodes_csv[rr]["fat_thickness_m"])*1000})
column_info=[]
for rr in (2245,2261,2262):
    c=nodes_csv[rr]
    V=np.asarray([[float(c[k]) for k in names] for names in (("ox","oy","oz"),("ix","iy","iz"),("dx","dy","dz"))])
    s=V[1]-V[0]; f=V[2]-V[1]
    angle=np.degrees(np.arccos(np.clip(np.dot(s,f)/(np.linalg.norm(s)*np.linalg.norm(f)),-1,1)))
    column_info.append({"reduced_row":rr,"source_vertex_id":int(embed_csv[rr]["source_vertex"]),
        "donor_body_id":int(donor_csv[rr]["donor"]),"skin_OI_length_mm":float(np.linalg.norm(s)*1000),
        "fat_ID_length_mm":float(np.linalg.norm(f)*1000),"OI_ID_angle_deg":float(angle),
        "declared_skin_thickness_mm":float(c["skin_thickness_m"])*1000,
        "declared_fat_thickness_mm":float(c["fat_thickness_m"])*1000})
# Enumerate the six staircase-compatible tetrahedralizations of the failing fat triangular prism.
face_rows=[r for r in faces_csv if int(r["stable_source_face"])==89542]
if len(face_rows)!=1:
    raise RuntimeError("stable source face 89542 does not resolve to exactly one D37 surface face")
face_vertices=[int(face_rows[0][k]) for k in ("a","b","c")]
fat_prism=[r for r in tets_csv if int(r["stable_source_face"])==89542 and int(r["material"])==2]
if len(fat_prism)!=3:
    raise RuntimeError("source face 89542 does not own exactly three fat prism tets")
current_prism=np.asarray([[int(r[k]) for k in ("a","b","c","d")] for r in fat_prism],dtype=np.int64)
current_q=cell_metrics(points,current_prism)[1]
variants={}
for perm in itertools.permutations(range(3)):
    i,j,k=perm
    v=np.asarray([[2411+face_vertices[i],2411+face_vertices[j],2411+face_vertices[k],4822+face_vertices[k]],
                  [2411+face_vertices[i],2411+face_vertices[j],4822+face_vertices[j],4822+face_vertices[k]],
                  [2411+face_vertices[i],4822+face_vertices[i],4822+face_vertices[j],4822+face_vertices[k]]],dtype=np.int64)
    key=tuple(sorted(tuple(sorted(map(int,t))) for t in v))
    if key in variants: continue
    sd,vq,vl,va,vasp=cell_metrics(points,v)
    boundary=boundary_faces(v)
    base_boundary=boundary_faces(current_prism)
    Itri=tuple(sorted(2411+x for x in face_vertices)); Dtri=tuple(sorted(4822+x for x in face_vertices))
    fixed_end_faces=Itri in boundary and Dtri in boundary
    side_diags=prism_side_diagonals(v,face_vertices)
    variants[key]={"permutation":[face_vertices[x] for x in perm],"tetrahedra":v.tolist(),
        "q_each":vq.tolist(),"q_min":float(vq.min()),"signed_det_each_m3":sd.tolist(),
        "total_abs_det_over_6_m3":float(np.abs(sd).sum()/6),"total_volume_delta_vs_current_m3":float(np.abs(sd).sum()/6-np.abs(cell_metrics(points,current_prism)[0]).sum()/6),
        "fixed_I_and_D_end_triangles_preserved":bool(fixed_end_faces),
        "boundary_facet_count":len(boundary),"changed_side_boundary_facets_vs_current":len(boundary.symmetric_difference(base_boundary)),
        "side_diagonals":side_diags}
current_key=tuple(sorted(tuple(sorted(map(int,t))) for t in current_prism))
current_variant=variants.get(current_key)
# Find the adjacent registered surface faces and current fat-prism diagonal on each shared edge.
adjacency=[]
for edge in ((face_vertices[0],face_vertices[1]),(face_vertices[1],face_vertices[2]),(face_vertices[2],face_vertices[0])):
    e=set(edge); neighbors=[r for r in faces_csv if r is not face_rows[0] and e.issubset({int(r[k]) for k in ("a","b","c")})]
    for nr in neighbors:
        sf=int(nr["stable_source_face"])
        nt=[r for r in tets_csv if int(r["stable_source_face"])==sf and int(r["material"])==2]
        diag=prism_side_diagonals(np.asarray([[int(t[k]) for k in ("a","b","c","d")] for t in nt],dtype=np.int64),[int(nr[k]) for k in ("a","b","c")]) if len(nt)==3 else {}
        adjacency.append({"shared_source_edge":list(edge),"neighbor_stable_source_face":sf,"neighbor_fat_tet_count":len(nt),"neighbor_side_diagonals":diag})
compat=[]
for key,v in variants.items():
    matches=True
    for n in adjacency:
        ekey=f"{n['shared_source_edge'][0]}-{n['shared_source_edge'][1]}"
        own=v["side_diagonals"].get(ekey,{}).get("diagonal")
        # The helper edge keys follow the local source face order; make undirected key.
        if own is None or n["neighbor_fat_tet_count"]!=3:
            matches=False; break
        nd=None
        for side in n["neighbor_side_diagonals"].values():
            if side.get("diagonal") is not None and set(side["diagonal"]).issubset({2411+n["shared_source_edge"][0],2411+n["shared_source_edge"][1],4822+n["shared_source_edge"][0],4822+n["shared_source_edge"][1]}):
                nd=side["diagonal"]
        if nd is None or tuple(own)!=tuple(nd):
            matches=False; break
    if matches: compat.append({"q_min":v["q_min"],"permutation":v["permutation"],"key":list(key)})
# Current cell geometric measures.
cell_nodes=[]
for n in T[local]:
    c=coord_csv[int(n)]
    cell_nodes.append([float(c[k]) for k in ("x_m","y_m","z_m")])
X=np.asarray(cell_nodes,dtype=np.float64)
det=float(np.dot(X[1]-X[0],np.cross(X[2]-X[0],X[3]-X[0])))
edge=np.asarray([np.linalg.norm(X[i]-X[j]) for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))])
long=float(edge.max())
face_area2=np.asarray([np.linalg.norm(np.cross(X[b]-X[a],X[c]-X[a])) for a,b,c in ((1,2,3),(0,2,3),(0,1,3),(0,1,2))])
alt=abs(det)/face_area2
report={
 "schema":"numi.human.physical-cell-quality-audit.026.v1","status":"READ_ONLY_DIAGNOSTIC","candidate":"wholebody-attachment-source-rebind-037-candidate020-identity004-002",
 "input_sha256":{"D37_tetrahedra.csv":sha(D37/"tetrahedra.csv"),"D37_nodes.csv":sha(D37/"nodes.csv"),
   "D37_donor_frames_fullweights.csv":sha(D37/"donor-frames-fullweights.csv"),"D37_embedding.csv":sha(D37/"embedding.csv"),
   "D37_faces.csv":sha(D37/"faces.csv"),"compiled_initial_skin_fem_nodes.csv":sha(COORDS),
   "initial_tet_witness.json":sha(WITNESS),"failure_analysis.json":sha(FAIL),"registered_skin_poses.json":sha(POSES)},
 "native_failure_mapping":{"global_tet":global_tet,"object_index":int(wit["object_index"]),"object_name":wit["object_name"],
   "object_element_offset":elem_offset,"object_local_tet":local,"mapping_identity":"global_tet = object_element_offset + object_local_tet",
   "D37_table_row":int(row["tet"]),"D37_material_code":int(row["material"]),"material_name":"fat (D37 report: 14,316 skin and 14,316 fat cells)",
   "D37_parent_tet":int(row["parent_tet"]),"historical_parent_surface_face":int(row["parent_surface_face"]),
   "stable_source_face":int(row["stable_source_face"]),"surface_face_row":int(face_rows[0]["face"]),
   "surface_face_reduced_vertices":face_vertices,"volume_node_ids":list(map(int,T[local])),"node_lineage":node_info,
   "failure_reason":json.loads(FAIL.read_text())["failure_reason"],"failure_reason_mask":int(json.loads(FAIL.read_text())["failure_reason_mask"]),
   "witness_current_J":float(wit["current_j_double_from_f32_nodes"]),"witness_validity_min_J":float(wit["validity_min_j"]),
   "witness_validity_max_J":float(wit["validity_max_j"]),"newton_diagnostic_trial_J":2.5066757202,"last_current_J_before_rejected_trial":2.4964625835,
   "accepted_physical_steps":int(json.loads(FAIL.read_text())["accepted_steps"]),"initial_tet_witness_current_det_m3_field":float(wit["current_det_m3"]),
   "initial_tet_witness_reference_det_m3_field":float(wit["reference_det_m3"]),
   "witness_explicit_signed_determinant_m3":wdet,"witness_explicit_tetra_volume_m3":wvol,
   "json_field_semantics":"current_det_m3/reference_det_m3 equal abs(det)/6 from listed xyz; the key is named determinant but stores tetra volume.",
   "compiled_csv_signed_determinant_m3":det,"compiled_csv_tetra_volume_m3":abs(det)/6,
   "compiled_csv_q_abs_det_over_longest_edge_cubed":float(q[local]),
   "compiled_csv_longest_edge_mm":long*1000,"compiled_csv_min_altitude_um":float(alt.min()*1e6),
   "compiled_csv_longest_edge_over_min_altitude":float(long/alt.min()),"compiled_csv_positive_orientation":bool(det>0)},
 "wholemesh_quality_from_compiled_F32_positions":{"cells":len(T),"positive_signed_determinants":int(np.sum(signed_det>0)),
   "nonpositive_signed_determinants":int(np.sum(signed_det<=0)),"q_definition":"abs(det)/longest_edge^3",
   "q_min":float(q.min()),"q_p01":float(np.quantile(q,.01)),"q_median":float(np.median(q)),"q_min_local_tet_row":int(q.argmin()),
   "per_material_code":material_summary,"modal_donor_groups_top15_by_q_min":donor_summary[:15],
   "donor_grouping_note":"modal donor across four tet vertex columns, deterministic lowest-ID tie break; donor ID is not a verified anatomical-region label."},
 "affected_layer_columns":{"rows":column_info,"failed_cell_node_ids":[int(n) for n in T[local]],
   "metric_note":"O/I/D thicknesses are measured from D37 generated column coordinates; compiled quality uses pinned native compiled F32 node coordinates."},
 "fat_prism_alternate_split_diagnostic":{"surface_face_row":int(face_rows[0]["face"]),"stable_source_face":89542,
   "current_fat_tet_rows":[int(r["tet"]) for r in fat_prism],"current_fat_tet_ids":current_prism.tolist(),
   "current_q_each":current_q.tolist(),"current_q_min":float(current_q.min()),"unique_staircase_variants":list(variants.values()),
   "number_of_local_variants":len(variants),"adjacent_registered_faces":adjacency,
   "variants_conforming_to_all_three_unchanged_neighbor_side_diagonals":compat,
   "interpretation":"A local prism diagonal change preserves the I/D end triangles but can alter shared internal side-face triangulations; any quality-gaining variant needs propagation to adjacent prisms and the owner must rebuild/check same-material topology, exact boundary/interface facets, positive F32 cells, and conservative mass."},
 "conclusion":{"initial_cell_is_inverted_or_below_quality_floor":bool(det<=0 or q[local]<1e-5),
   "failure_is_initial_geometry_gate":False,"failure_is_newton_trial_upper_J_gate":True,
   "interpretation":"The cell begins with J=1 and positive F32 geometry, above the recorded q floor. It is highly slender and has a source vertex inside the left metacarpal region according to the existing source/bone witness, so geometry/contact conditioning may contribute; this audit does not assign the rejected static Newton trial to mesh shape versus contact/muscle loading.",
   "no_mesh_or_native_changes":True}
}
(OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"output":str(OUT/"report.json"),"hash":sha(OUT/"report.json"),"failed_cell":report["native_failure_mapping"],"material_quality":material_summary,"prism_variants":len(variants),"compatible_neighbor_variants":len(compat)},indent=2))
