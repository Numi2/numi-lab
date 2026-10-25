#!/usr/bin/env python3
"""Read-only head-vs-O/I/D feasibility summary from frozen 088 outputs."""
from __future__ import annotations
import csv, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

W = Path("/Users/n/Work/numi-human-connected-surface-069")
RUN = W / "whole-reference-tetgen-084-bounded/connected-clearance-088/run-001"
IT = RUN / "iteration-003"
Q = IT / "query"
SOURCE = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight")
NODES = SOURCE / "exact-emitted.nodes.csv"
BODIES = SOURCE / "exact-emitted.bodies.csv"
ONTOLOGY = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.manifest.json")
OUT = Path(__file__).with_name("report.json")
HEAD_BODY = 23
LAYERS = {0: "O_outer", 1: "I_interface", 2: "D_deep"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def qstats(v):
    a = np.asarray(v, dtype=np.float64)
    return {"count": int(a.size), "min_mm": float(np.min(a)), "p25_mm": float(np.quantile(a,.25)), "median_mm": float(np.quantile(a,.5)), "p75_mm": float(np.quantile(a,.75)), "max_mm": float(np.max(a))} if a.size else {"count": 0}

with np.load(IT / "candidate.npz") as z:
    C = z["points_world_f32"].copy()
with np.load(IT / "input.npz") as z:
    I0 = z["points_world_f32"].copy()
with np.load(RUN / "baseline/candidate.npz") as z:
    B0 = z["points_world_f32"].copy()
with np.load(RUN / "iteration-003/input.npz") as z:
    reference = z["reference_points_m"].copy()
points_rows = list(csv.DictReader((Q / "points.csv").open()))
queries = np.asarray([[float(r[k]) for k in ("x_m","y_m","z_m")] for r in points_rows], np.float32)
assert C.shape == I0.shape == B0.shape == queries.shape == (7360,3)
assert np.array_equal(C, queries), "query point rows must be exact emitted candidate F32"

with np.load(IT / "candidate.npz") as z:
    assert z["tetrahedra"].shape == (32796,4)
    assert z["material_ids"].shape == (32796,)

src_rows = list(csv.DictReader(NODES.open()))
assert len(src_rows) == 7233
assert all(int(r["local_node"]) == i for i,r in enumerate(src_rows))
assert all(int(r["layer"]) == i//2411 and int(r["row"]) == i%2411 for i,r in enumerate(src_rows))
S = np.asarray([[float(r[k]) for k in ("x_m","y_m","z_m")] for r in src_rows], np.float32)

with open(Q / "query-receipt.json") as f: qr = json.load(f)
query_pairs = list(csv.DictReader((Q / "pairs.csv").open()))
base_pairs = list(csv.DictReader((RUN / "baseline/query/pairs.csv").open()))
node_rows = list(csv.DictReader((Q / "nodes.csv").open()))
assert len(query_pairs) == 2076
assert int(qr["query_rows"]) == 7360
aabb_outside = sum(int(r["aabb_proven_outside_banks"]) for r in node_rows)
assert len(query_pairs) + aabb_outside == 7360*185

head_pairs = [r for r in query_pairs if int(r["bank_body"]) == HEAD_BODY and r["classification"] == "inside"]
base_head = [r for r in base_pairs if int(r["bank_body"]) == HEAD_BODY and r["classification"] == "inside"]
byrow = defaultdict(list)
for r in query_pairs:
    if int(r["bank_body"]) == HEAD_BODY:
        byrow[int(r["query_row"])].append(r)
head_byrow = defaultdict(list)
for r in head_pairs: head_byrow[int(r["query_row"])].append(r)
head_row_set = set(head_byrow)
base_head_rows = {int(r["query_row"]) for r in base_head}

ordinary_cols = sorted({r % 2411 for r in head_row_set if r < 7233})
D_cols = sorted({r-4822 for r in head_row_set if 4822 <= r < 7233})
assert len(D_cols) == 75 and len(ordinary_cols) == 75

# Exact body 23 current transform used by the query owner: M * (world - position) - COM offset.
bodies = list(csv.DictReader(BODIES.open()))
body = next(r for r in bodies if int(r["body_index"]) == HEAD_BODY)
M = np.asarray([[float(body["world_to_body_m"+str(i)+str(j)]) for j in range(3)] for i in range(3)])
pos = np.asarray([float(body["position_"+k+"_m"]) for k in "xyz"])
offset = np.asarray([float(body["com_offset_"+k+"_m"]) for k in "xyz"])
def local(x): return M @ (np.asarray(x,dtype=np.float64)-pos) - offset
# Check transform against exact query-local values for all hit rows.
transform_residuals=[]
for row, hits in head_byrow.items():
    qrow=next(x for x in points_rows if int(x["query_row"])==row)
    world=np.asarray([float(qrow[k]) for k in ("x_m","y_m","z_m")])
    for hit in hits:
        qloc=np.asarray([float(hit["query_local_"+k]) for k in "xyz"])
        transform_residuals.append(float(np.linalg.norm(local(world)-qloc)))
assert max(transform_residuals) < 1e-8

# Endpoint classification per ordinary source column. O/I/D rows are exact local_node order.
head_banks=sorted({int(r["bank_stable_id"]) for r in query_pairs if int(r["bank_body"])==HEAD_BODY})
endpoint_counts=Counter()
columns=[]
for c in ordinary_cols:
    layer_status={}
    for layer,name in LAYERS.items():
        row=layer*2411+c
        hits=head_byrow.get(row,[])
        # Pairs omit exactly those bank pairs that were AABB-proven outside.
        layer_status[name] = "inside" if hits else ("queried_no_inside" if byrow.get(row) else "aabb_proven_outside_all_head_banks")
        if hits: endpoint_counts[name]+=1
    columns.append({"source_column_row":int(c),"source_global_nodes":[int(src_rows[l*2411+c]["global_node"]) for l in range(3)],"endpoint_status":layer_status})

# Source/current same-column spacing on the 75 columns whose D endpoint is inside a head bank.
spacing={}
for label,arr in (("035_exact_emitted_source",S),("088_iteration003_input",I0),("088_iteration003_candidate",C)):
    oi=np.linalg.norm(arr[D_cols] - arr[np.asarray(D_cols)+2411],axis=1)*1000
    idv=np.linalg.norm(arr[np.asarray(D_cols)+2411] - arr[np.asarray(D_cols)+4822],axis=1)*1000
    spacing[label]={"O_to_I":qstats(oi),"I_to_D":qstats(idv)}

displacements={}
for label,arr in (("088_baseline_candidate",B0),("088_iteration003_input",I0),("088_iteration003_candidate",C)):
    displacements[label]={}
    for layer,name in LAYERS.items():
        ix=np.asarray([layer*2411+c for c in D_cols],dtype=np.int64)
        displacements[label][name+"_from_035_emitted_source"]=qstats(np.linalg.norm(arr[ix]-S[ix],axis=1)*1000)
displacements["088_iteration003_delta_from_start"]={}
for layer,name in LAYERS.items():
    ix=np.asarray([layer*2411+c for c in D_cols],dtype=np.int64)
    displacements["088_iteration003_delta_from_start"][name]=qstats(np.linalg.norm(C[ix]-I0[ix],axis=1)*1000)

# Nearest-face plane projection along each I-D column, using the saved closest point and
# the normal direction from that closest point toward the D query. This is a local plane
# diagnostic only; it is not a finite-triangle or whole-solid clearance test.
plane=[]
for c in D_cols:
    drow=4822+c
    hit=next(r for r in head_byrow[drow] if int(r["bank_stable_id"]) in (28,29,30,32,33))
    closest=np.asarray([float(hit["closest_"+k]) for k in "xyz"])
    D=local(C[drow]); I=local(C[2411+c]); O=local(C[c])
    vec=D-closest; norm=float(np.linalg.norm(vec)); n=vec/norm
    dO=float((O-closest)@n)*1000; dI=float((I-closest)@n)*1000; dD=float((D-closest)@n)*1000
    frac=(-dI/(dD-dI)) if dI < 0 < dD else None
    plane.append({"column":int(c),"head_stable_id":int(hit["bank_stable_id"]),"nearest_distance_mm":float(hit["nearest_distance_m"])*1000,"signed_face_distance_mm":float(hit["signed_face_distance_m"])*1000,"O_projection_mm":dO,"I_projection_mm":dI,"D_projection_mm":dD,"I_to_D_plane_crossing_fraction":None if frac is None else float(frac)})

bank_summaries={}
for bid in sorted({int(r["bank_stable_id"]) for r in head_pairs}):
    rr=[r for r in head_pairs if int(r["bank_stable_id"])==bid]
    bank_summaries[str(bid)]={"inside_rows":len(rr),"query_rows":[int(r["query_row"]) for r in rr],"nearest_distance_mm":qstats([float(r["nearest_distance_m"])*1000 for r in rr]),"signed_face_distance_mm":qstats([float(r["signed_face_distance_m"])*1000 for r in rr])}

with open(ONTOLOGY) as f: ontology=json.load(f)
head_anchors=[a for a in ontology["source"]["anchors"] if int(a["core_body_index"])==HEAD_BODY]
assert head_anchors and all(a["myosim_body"]=="head" for a in head_anchors)

sha_inputs={}
for name,p in {
    "iteration003_candidate":IT/"candidate.npz",
    "iteration003_input":IT/"input.npz",
    "iteration003_points_csv":Q/"points.csv",
    "iteration003_pairs_csv":Q/"pairs.csv",
    "iteration003_nodes_csv":Q/"nodes.csv",
    "iteration003_query_receipt":Q/"query-receipt.json",
    "baseline_candidate":RUN/"baseline/candidate.npz",
    "baseline_pairs_csv":RUN/"baseline/query/pairs.csv",
    "035_exact_emitted_nodes":NODES,
    "039_exact_emitted_bodies":BODIES,
    "head_ontology_manifest":ONTOLOGY,
    "analysis_source":Path(__file__),
}.items(): sha_inputs[name]={"path":str(p),"sha256":sha(p)}
sha_inputs["query_binary"]={"path":qr["argv"][0],"sha256":qr["binary_sha256"]}
sha_inputs["query_triangles"]={"path":qr["argv"][0+1],"sha256":qr["triangles_sha256"],"sha_from_query_receipt":True}

frac=np.asarray([x["I_to_D_plane_crossing_fraction"] for x in plane if x["I_to_D_plane_crossing_fraction"] is not None],dtype=float)
report={
 "schema":"numi.head-layer-feasibility.090.v1",
 "scope":"Read-only analysis of frozen 088 iteration-003 static 185-bank query; no geometry edited, no query rerun, no physical/native qualification.",
 "inputs":sha_inputs,
 "head_identity":{"body_index":HEAD_BODY,"myosim_body":"head","ontology_manifest_anchors":len(head_anchors),"queried_head_stable_bone_ids_with_hits":sorted(int(k) for k in bank_summaries)},
 "query_integrity":{"candidate_query_points_bitwise_equal_float32":True,"all_bank_query_rows":int(qr["query_rows"]),"bank_count":185,"stored_non_aabb_pair_rows":len(query_pairs),"aabb_proven_outside_bank_pairs":aabb_outside,"pair_partition_exact":True,"inside_all_banks":sum(int(r["inside_banks"]) for r in node_rows),"ambiguous_all_banks":sum(int(r["ambiguous_banks"]) for r in node_rows),"invalid_all_banks":sum(int(r["invalid_banks"]) for r in node_rows),"query_transform_max_local_residual_m":max(transform_residuals),"query_mode_physical_steps":qr.get("physical_steps")},
 "head_endpoint_counts_iteration003":{"inside_pair_count":len(head_pairs),"unique_inside_query_rows":len(head_row_set),"ordinary_source_columns_with_any_head_inside_endpoint":len(ordinary_cols),"ordinary_O_inside_columns":endpoint_counts["O_outer"],"ordinary_I_inside_columns":endpoint_counts["I_interface"],"ordinary_D_inside_columns":endpoint_counts["D_deep"],"generated_node_inside_rows":[{"query_row":r,"stable_bone_ids":[int(x["bank_stable_id"]) for x in head_byrow[r]]} for r in sorted(head_row_set) if r>=7233],"per_head_stable_bone":bank_summaries,"ordinary_columns":columns},
 "head_endpoint_counts_baseline":{"inside_pair_count":len(base_head),"unique_inside_query_rows":len(base_head_rows),"by_layer":{"O_outer":sum(0<=r<2411 for r in base_head_rows),"I_interface":sum(2411<=r<4822 for r in base_head_rows),"D_deep":sum(4822<=r<7233 for r in base_head_rows),"generated":sum(r>=7233 for r in base_head_rows)},"added_rows_vs_baseline":sorted(head_row_set-base_head_rows),"removed_rows_vs_baseline":sorted(base_head_rows-head_row_set)},
 "spacing_on_75_D_hit_columns":spacing,
 "movement_on_75_D_hit_columns_mm":displacements,
 "nearest_face_plane_projection_diagnostic":{"method":"Transform same-column candidate O/I/D points with the exact body-23 world-to-body matrix and position/COM offset from 039; for each D hit use the saved nearest point and vector from that point toward D as the plane normal, then project O/I/D onto that infinite plane.","is_finite_triangle_or_solid_clearance":False,"crossing_count_I_to_D":int(frac.size),"columns":len(plane),"crossing_fraction_stats":qstats(frac),"D_nearest_distance_stats":qstats([x["nearest_distance_mm"] for x in plane]),"I_to_D_length_stats":spacing["088_iteration003_candidate"]["I_to_D"],"rows":plane},
 "interpretation":{"finding":"The frozen query localizes nearly all head containment to the deep D endpoint: iteration 003 has 75 ordinary D-inside columns, one additional I-inside endpoint, zero O-inside columns, plus one generated boundary-node hit (row 7237). Baseline already had 72 D-inside rows, one I row, and one generated row; iteration 003 adds four rows and removes one, net +3. The 088 iteration step itself moved points on these columns by at most about 0.0271 mm, while the 035-to-088 starting geometry already differs by millimetres.","layer_scale":"On the same D-hit columns, exact 035 source O-I median spacing is 2.451 mm and I-D is 10.256 mm; iteration-003 candidate medians are 2.252 mm and 10.277 mm. This makes the deep-layer spacing/gap a plausible contributor and argues against a broad outer-skin O penetration, but is not enough to distinguish a misregistered D layer from an insufficient local head gap.","plane_limit":"The closest-face plane projection is only a directional local diagnostic. It is not a finite-face intersection, full skull/head containment certificate, or anatomical clearance measure.","conclusion":"The available evidence localizes the issue to D placement and/or its thickness relative to the head banks; it does not establish whether the D registration or the local permissible gap is the root cause. Do not interpret these static point classifications as physical acceptance."}
}
OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(OUT)
print("sha256",sha(OUT))
print("report summary",json.dumps({"baseline_inside":report["head_endpoint_counts_baseline"],"candidate_inside":{k:v for k,v in report["head_endpoint_counts_iteration003"].items() if k!="ordinary_columns" and k!="per_head_stable_bone"},"spacing":{k:{layer:report["spacing_on_75_D_hit_columns"][k][layer]["median_mm"] for layer in ("O_to_I","I_to_D")} for k in report["spacing_on_75_D_hit_columns"]},"plane_crosses":len(frac)},sort_keys=True))
