from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from outward_ray_exit_098 import build_source_outward_field

OUT = HERE / "source-field-audit-002"
if OUT.exists():
    raise SystemExit(f"refusing to overwrite existing audit directory: {OUT}")
OUT.mkdir(parents=True)
COMPACT = Path("/Users/n/Work/numi-human-connected-surface-069/whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz")
SOURCE = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003/candidate.npz")
PINS = {
    str(COMPACT): "5a35470223ce12343cc217be7822d182802e99f1c423bb52b8409f153f9a49db",
    str(SOURCE): "640b7bc56013748b26d5e4e12790f85a472602ee6a4732f5773243df6e2147cc",
}
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()
for p, expected in PINS.items():
    actual=sha(Path(p))
    if actual != expected: raise SystemExit(f"input SHA mismatch: {p}: {actual}")

with np.load(COMPACT, allow_pickle=False) as z:
    source_ids=z["source_global_node_ids"]
    input_rows=z["source_input_row"]
    gen_rows=z["newnode_ancestry_output_row"]
    gen_parents=z["newnode_ancestry_source_tet_row"]
    bary=z["newnode_ancestry_barycentric"]
    material_ids=z["newnode_ancestry_material_id"]
    compact_tets=z["tetrahedra_zero_based"]
    compact_materials=z["material_ids"]
    compact_R=z["reference_points_m"]
    compact_X=z["current037_points_m_f32"]
with np.load(SOURCE, allow_pickle=False) as z:
    O=z["O_world_m"]
    D=z["D_world_m"]
    native_tets=z["old_tetrahedra"]
    native_materials=z["old_material_ids"]
    native_points=z["points_world_f32"]
if native_points.shape != (len(source_ids),3) or native_tets.shape[1] != 4:
    raise SystemExit("unexpected native source topology domain")
if compact_tets.shape != (32796,4) or len(compact_R) != 7360 or len(compact_X) != 7360:
    raise SystemExit("unexpected compact candidate topology/coordinate domain")
# Match the actual compacted reference frame used by the bone query.
# The older O_world_m/D_world_m arrays are authored-source coordinates and
# are not substituted for the compiled/posed reference node rows.
source_R=np.empty((len(source_ids),3),dtype=np.float64)
direct=np.flatnonzero(input_rows>=0)
assert len(direct)==len(source_ids) and np.array_equal(np.sort(input_rows[direct]),np.arange(len(source_ids)))
source_R[input_rows[direct]]=compact_R[direct]
O=source_R[:2411].copy();D=source_R[4822:7233].copy()
field, field_report = build_source_outward_field(
    source_global_node_ids=source_ids,
    output_source_input_row=input_rows,
    outer_points_by_column_m=O,
    deep_points_by_column_m=D,
    native_parent_tetrahedra=native_tets,
    generated_output_rows=gen_rows,
    generated_parent_tet_rows=gen_parents,
    generated_parent_barycentric=bary,
    column_count=2411,
    direction_tolerance=1e-8,
    strict_parent_agreement=False,
)
generated_unique=np.unique(gen_rows)
# Preserve all raw ancestry rows and the NaN-marked field: no averaging or fallback.
field_npz=OUT/"source-outward-field.npz"
np.savez_compressed(
    field_npz,
    source_outward_direction_by_compact_output_row=field,
    source_global_node_ids=source_ids,
    source_input_row_by_compact_output_row=input_rows,
    generated_output_rows=generated_unique,
    generated_parent_output_rows=gen_rows,
    generated_parent_native_tet_rows=gen_parents,
    generated_parent_material_ids=material_ids,
    generated_parent_barycentric_raw=bary,
)
out_report={
    "schema":"source-outward-field-audit-098-v1",
    "inputs":{
        "compact_candidate_npz":{"path":str(COMPACT),"sha256":PINS[str(COMPACT)]},
        "source_candidate_npz":{"path":str(SOURCE),"sha256":PINS[str(SOURCE)]},
        "compact_point_domain":"084 compacted reference_points_m; source_input_row maps exact source rows to compact rows",
        "direction_coordinate_frame":"same compiled reference frame as 084 reference_points_m, not legacy O_world_m/D_world_m metadata",
        "native_parent_tet_domain":"old_tetrahedra rows in native source candidate; vertex indices are source-input rows 0..7232",
        "global_source_ids_semantics":"identity labels only; source-input row order carries topology and O/I/D column index",
    },
    "compact_counts":{
        "nodes":int(len(input_rows)),"tetrahedra":int(len(compact_tets)),
        "materials":{str(int(k)):int(v) for k,v in zip(*np.unique(compact_materials,return_counts=True))},
        "direct_source_rows":int(np.count_nonzero(input_rows>=0)),
        "generated_output_rows":int(len(generated_unique)),
        "generated_parent_ancestry_choices":int(len(gen_rows)),
        "source_global_id_min":int(source_ids.min()),"source_global_id_max":int(source_ids.max()),
    },
    "ancestry_numeric":{
        "raw_barycentric_min":float(np.min(bary)),
        "raw_barycentric_max":float(np.max(bary)),
        "max_abs_row_sum_error":float(np.max(np.abs(bary.sum(axis=1)-1))),
        "accepted_roundoff_tolerance":1e-10,
        "weights_clamped_or_renormalized":False,
    },
    "field_summary":{
        "finite_rows":int(np.isfinite(field).all(axis=1).sum()),
        "unresolved_rows":field_report["unresolved_output_rows"],
        "generated_direction_disagreement_count":int(len(field_report["generated_direction_disagreements"])),
        "maximum_parent_direction_spread":field_report["maximum_generated_parent_direction_spread"],
        "strict_parent_agreement":False,
        "direction_definition":"normalize(compacted reference O[input_row % 2411] - compacted reference D[input_row % 2411]), after explicit source-input row inverse mapping",
        "generated_field":"raw barycentric prolongation through every declared native parent tet; normalize each candidate; compare all candidate directions per generated node",
        "anatomical_outward_claim":False,
        "full_field_diagnostic":field_report,
    },
    "geometry_gates":{
        "source_field_only":True,
        "ray_queries_run":False,
        "mesh_or_source_changed":False,
        "anatomical_outward_or_bone_clearance_claim":False,
    },
    "output":{
        "npz":str(field_npz),"sha256":sha(field_npz),
        "keys":["source_outward_direction_by_compact_output_row","source_global_node_ids","source_input_row_by_compact_output_row","generated_output_rows","generated_parent_output_rows","generated_parent_native_tet_rows","generated_parent_material_ids","generated_parent_barycentric_raw"],
    },
}
rp=OUT/"source-field-audit.json"
rp.write_text(json.dumps(out_report,indent=2,sort_keys=True)+"\n")
manifest={
    "script_path":str(Path(__file__).resolve()),
    "script_sha256":sha(Path(__file__).resolve()),
    "helper_path":str(HERE/"outward_ray_exit_098.py"),
    "helper_sha256":sha(HERE/"outward_ray_exit_098.py"),
    "test_path":str(HERE/"test_outward_ray_exit_098.py"),
    "test_sha256":sha(HERE/"test_outward_ray_exit_098.py"),
    "report_path":str(rp),"report_sha256":sha(rp),
    "npz_path":str(field_npz),"npz_sha256":sha(field_npz),
    "input_pins":PINS,
}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
print(json.dumps({"report":str(rp),"manifest":str(OUT/"manifest.json"),"nodes":len(input_rows),"direct":int(np.count_nonzero(input_rows>=0)),"generated":len(generated_unique),"choices":len(gen_rows),"unresolved":field_report["unresolved_output_rows"],"disagreement_count":len(field_report["generated_direction_disagreements"]),"spread":field_report["maximum_generated_parent_direction_spread"],"bary_min":float(bary.min()),"bary_max_sum_error":float(np.max(np.abs(bary.sum(axis=1)-1)))},indent=2))

