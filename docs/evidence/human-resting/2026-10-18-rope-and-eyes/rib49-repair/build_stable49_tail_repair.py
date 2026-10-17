import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/Users/n/numi-human-final-integration-publish-1397/src")
from numilab_human import composed_bone_body_frame as cbb
from numilab_human.cardiac_cavity_geometry import analyze_topology

root = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378")
srcdir = root / "rib50-published-source-owner-reproducer-952" / "compiled"
src = srcdir / "bodyparts3d-myosim-major-bones.nhbones"
srcman = srcdir / "bodyparts3d-myosim-major-bones.manifest.json"
default_outdir = root / "rope-suspension-reference-probe-20261018-001" / "stable49-subulp-edge-retirement-001"
outdir = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else default_outdir
out = outdir / "bodyparts3d-myosim-major-bones.nhbones"
manout = outdir / "bodyparts3d-myosim-major-bones.manifest.json"
side = outdir / "stable49-face-ancestry.json"
reportpath = outdir / "repair-report.json"
def sha(b): return hashlib.sha256(b).hexdigest()
raw = src.read_bytes()
assert sha(raw) == "6a1ece3a3a282873175356df7c3b0430e9657d3322fef1e992eba74b1d1e80bb", sha(raw)
mr = srcman.read_bytes()
m = json.loads(mr)
header, records, voff, ioff = cbb._decode_bones(raw)
assert len(records) == 185 and header[2] == 185
ri = 48
r = records[ri]
body_index, vertex_first, vertex_count, index_first, index_count, stable_id = r[:6]
assert stable_id == 49 and body_index == 20 and vertex_count == 11668
assert index_count == 26436 and index_count // 3 == 8812
assert ioff + header[4] * 4 == len(raw)
indices_all = np.frombuffer(raw, dtype="<u4", count=header[4], offset=ioff).copy()
vertex_bytes = raw[voff:ioff]
assert len(vertex_bytes) == header[3] * cbb._BONE_VERTEX.size
allv = np.frombuffer(vertex_bytes, dtype="<f4").reshape(header[3], 6).copy()
faces = indices_all[index_first:index_first+index_count].reshape(-1, 3).copy()
assert np.all(faces < header[3])
f0, f1, keep, remove = 8798, 8799, 7527, 11530
assert faces[f0].tolist() == [vertex_first+7527, vertex_first+7525, vertex_first+11530]
assert faces[f1].tolist() == [vertex_first+11530, vertex_first+7522, vertex_first+7527]
a = allv[vertex_first+keep,:3].astype(np.float64)
b = allv[vertex_first+remove,:3].astype(np.float64)
edge_m = float(np.linalg.norm(a-b))
assert 1.30e-8 < edge_m < 1.40e-8, edge_m
rowmeta = [x for x in m["body_local_bakes"]["rows"] if x.get("channel_index")==13 and x.get("core_body_index")==20 and x.get("compiled_vertex_count")==11668]
assert len(rowmeta)==1
interface_set = {int(pair[1]) for pair in rowmeta[0]["interface_vertex_pairs"]}
assert keep not in interface_set and remove not in interface_set
faces[faces == vertex_first+remove] = vertex_first+keep
keep_mask = np.ones(len(faces), dtype=bool)
keep_mask[[f0,f1]] = False
faces_out = faces[keep_mask]
assert len(faces_out)==8810
assert not np.any((faces_out[:,0]==faces_out[:,1]) | (faces_out[:,1]==faces_out[:,2]) | (faces_out[:,2]==faces_out[:,0]))
chunks=[]; newrecords=list(records); cursor=0
face_origins=np.flatnonzero(keep_mask).astype(int).tolist()
for j, rec in enumerate(records):
    first,count=rec[3],rec[4]
    rowidx=indices_all[first:first+count]
    nr=list(rec)
    if j==ri:
        assert first==index_first and count==index_count
        rowidx=faces_out.reshape(-1).astype("<u4",copy=False)
        nr[3]=cursor; nr[4]=int(rowidx.size)
    elif j>ri:
        nr[3]=first-6
    else:
        nr[3]=first
    newrecords[j]=tuple(nr)
    chunks.append(np.asarray(rowidx,dtype="<u4").tobytes())
    cursor += int(np.asarray(rowidx).size)
assert cursor==header[4]-6
new_all_indices=np.frombuffer(b"".join(chunks),dtype="<u4")
for j,rec in enumerate(records):
    if j==ri: continue
    nr=newrecords[j]
    assert np.array_equal(indices_all[rec[3]:rec[3]+rec[4]],new_all_indices[nr[3]:nr[3]+nr[4]]), j
newheader=list(header); newheader[4]=header[4]-6
payload=bytearray(cbb._BONE_HEADER.pack(*newheader))
for rec in newrecords: payload.extend(cbb._BONE_RECORD.pack(*rec))
payload.extend(vertex_bytes); payload.extend(b"".join(chunks)); payload=bytes(payload)
assert len(payload)==len(raw)-24
# Decode and validate completely before creating output files.
h2,rs2,vo2,io2=cbb._decode_bones(payload)
r2=rs2[ri]
assert h2[4]==header[4]-6 and r2[5]==49 and r2[4]==26430
v2bytes=payload[vo2:io2]; assert v2bytes==vertex_bytes
f2=np.frombuffer(payload,dtype="<u4",count=r2[4],offset=io2+r2[3]*4).reshape(-1,3)
f2local=f2.astype(np.int64)-vertex_first
before_vertices=allv[vertex_first:vertex_first+vertex_count,:3]
before_faces=np.frombuffer(raw,dtype="<u4",count=index_count,offset=ioff+index_first*4).reshape(-1,3).astype(np.int64)-vertex_first
after_vertices=np.frombuffer(v2bytes,dtype="<f4").reshape(header[3],6)[vertex_first:vertex_first+vertex_count,:3]
before_top=analyze_topology(before_vertices.tolist(),before_faces.tolist())
after_top=analyze_topology(after_vertices.tolist(),f2local.tolist())
assert before_top.get("boundary_edge_count")==0 and after_top.get("boundary_edge_count")==0
assert len(before_top.get("nonmanifold_edges",[]))==0 and len(after_top.get("nonmanifold_edges",[]))==0
assert len(before_top.get("orientation_defect_edges",[]))==0 and len(after_top.get("orientation_defect_edges",[]))==0
assert len(after_top.get("degenerate_face_ids",[]))==0 and len(after_top.get("duplicate_face_ids",[]))==0
assert len(after_top.get("vertex_manifold_defect_ids",[]))==0 and after_top.get("face_component_count")==1
assert after_top.get("face_count")==8810 and after_top.get("euler_characteristic")==2
used=np.unique(f2local); remap=np.full(vertex_count,-1,dtype=np.int64); remap[used]=np.arange(len(used))
compact_top=analyze_topology(after_vertices[used].tolist(),remap[f2local].tolist())
assert compact_top.get("closed_oriented_manifold_candidate") is True, {k:compact_top.get(k) for k in ("boundary_edge_count","face_component_count","closed_oriented_manifold_candidate")}
pose=tuple(float(x) for x in records[ri][6:14])
after_rows=np.frombuffer(v2bytes,dtype="<f4").reshape(header[3],6)[vertex_first:vertex_first+vertex_count]
world=np.asarray([cbb._row_body_local(tuple(map(float,row)),pose)[:3] for row in after_rows],dtype="<f4").astype(np.float64)
areas=np.linalg.norm(np.cross(world[f2local][:,1]-world[f2local][:,0],world[f2local][:,2]-world[f2local][:,0]),axis=1)*0.5
assert np.all(np.isfinite(world)) and float(areas.min())>0.0
decl=root/"intercostal313-rib48-rib49-source-export-006-reader-rebind-004"/m["body_local_bakes"]["declaration_file"]
assert sha(decl.read_bytes())==m["body_local_bakes"]["declaration_sha256"]
# Update only current stable-49 output-count declarations; preserve all upstream historical hashes.
rowmeta[0]["compiled_triangle_count"]=8810
m["body_local_bakes"]["payload_sha256"]=sha(payload)
m["payload"]["sha256"]=sha(payload); m["payload"]["bytes"]=len(payload); m["payload"]["index_count"]=header[4]-6
def update_counts(node):
    if isinstance(node,dict):
        sid=node.get("stable_id")
        is49=(sid==49 or (node.get("channel_index")==13 and node.get("core_body_index")==20 and node.get("compiled_vertex_count")==11668))
        if is49:
            if node.get("compiled_vertex_count")==11668 and "compiled_triangle_count" in node: node["compiled_triangle_count"]=8810
            if node.get("new_compiled_vertex_count")==11668 and "new_compiled_triangle_count" in node: node["new_compiled_triangle_count"]=8810
        for value in node.values(): update_counts(value)
    elif isinstance(node,list):
        for value in node: update_counts(value)
update_counts(m)
ancestry={
 "schema":"numi.human.stable49-subulp-tail-edge-retirement.v1",
 "input_payload_sha256":sha(raw),"input_manifest_sha256":sha(mr),
 "input_stable_id":49,"input_record_index":ri,"input_compiled_face_count":8812,"output_compiled_face_count":8810,
 "preserved_output_face_to_input_compiled_face":face_origins,
 "retired_input_compiled_face_ordinals":[f0,f1],
 "retired_face_row_local_vertices":{"8798":[7527,7525,11530],"8799":[11530,7522,7527]},
 "canonical_vertex_row_local":keep,"retired_vertex_row_local":remove,
 "vertex_positions_normals_allocations_unchanged":True,
 "shared_interface_vertex_count":len(interface_set),"collapsed_vertices_in_shared_interface":0,
 "upstream_partition_ancestry_sha256":m["body_local_bakes"]["partition_ancestry_sha256"],
 "upstream_per_face_source_parent_csr_available_in_input":False,
 "ancestry_limit":"Input NHBONES lacks upstream per-face parent CSR. This sidecar binds compiled-output face ordinals and preserves the upstream ancestry digest; it does not invent BodyParts3D source-face parents.",
 "operation":"Replace row-local index 11530 with existing row-local vertex 7527, then retire only the two resulting repeated-index faces 8798 and 8799. Keep allocated vertex arrays and all records/attachments/mass unchanged.",
 "edge_length_source_m":edge_m,
 "world_failure":{"step":31,"local_face":8798,"global_vertex_indices":[159564,159562,163567],"first_and_third_world_positions_equal":True,"world_z_m":1.8305658102035522,"float32_ulp_at_world_z_m":float(np.spacing(np.float32(1.8305658102035522)))}
}
side_bytes=(json.dumps(ancestry,sort_keys=True,separators=(",",":"))+"\n").encode()
m["stable49_subulp_tail_edge_repair"]={
 "schema":ancestry["schema"],"sidecar_file":side.name,"sidecar_sha256":sha(side_bytes),
 "candidate_payload_sha256":sha(payload),"input_payload_sha256":sha(raw),"input_manifest_sha256":sha(mr),
 "stable_id":49,"input_actual_compiled_triangles":8812,"output_compiled_triangles":8810,
 "retired_compiled_face_ordinals":[f0,f1],"anatomical_admission":False,
 "mass_attachment_or_material_change":False,
 "validation_scope":"Source topology and accepted step-0 upright Float32 transform only; native motion/replay remains required."
}
manbytes=(json.dumps(m,sort_keys=True,indent=2)+"\n").encode()
# Final writes only after all geometry/schema checks above succeed.
outdir.mkdir(parents=True,exist_ok=False)
out.write_bytes(payload); side.write_bytes(side_bytes); manout.write_bytes(manbytes); shutil.copy2(decl,outdir/decl.name)
report={
 "status":"CANDIDATE_REPAIRED_LOCAL_SOURCE_TOPOLOGY_PENDING_NATIVE",
 "input_payload_path":str(src),"input_payload_sha256":sha(raw),"input_manifest_path":str(srcman),"input_manifest_sha256":sha(mr),
 "candidate_payload_path":str(out),"candidate_payload_sha256":sha(payload),
 "candidate_manifest_path":str(manout),"candidate_manifest_sha256":sha(manbytes),
 "ancestry_path":str(side),"ancestry_sha256":sha(side_bytes),
 "stable_id":49,"body_index":20,"input_actual_face_count":8812,"input_manifest_declared_face_count_stale":8818,
 "candidate_face_count":8810,"allocated_row_vertices_preserved":11668,"global_allocated_vertex_count_preserved":header[3],
 "retired_compiled_face_ordinals":[f0,f1],"canonicalized_row_local_edge":[keep,remove],
 "source_edge_length_m":edge_m,"world_z_float32_ulp_m":float(np.spacing(np.float32(1.8305658102035522))),
 "shared_interface_vertex_count":len(interface_set),"collapsed_shared_interface_vertices":0,
 "before_topology":{k:before_top.get(k) for k in ("vertex_count","face_count","edge_count","boundary_edge_count","boundary_loop_count","nonmanifold_edges","orientation_defect_edges","degenerate_face_ids","duplicate_face_ids","vertex_manifold_defect_ids","face_component_count","euler_characteristic","closed_oriented_manifold_candidate")},
 "after_topology":{k:after_top.get(k) for k in ("vertex_count","face_count","edge_count","boundary_edge_count","boundary_loop_count","nonmanifold_edges","orientation_defect_edges","degenerate_face_ids","duplicate_face_ids","vertex_manifold_defect_ids","face_component_count","euler_characteristic","closed_oriented_manifold_candidate")},
 "referenced_surface_topology_after":{k:compact_top.get(k) for k in ("vertex_count","face_count","edge_count","boundary_edge_count","boundary_loop_count","nonmanifold_edges","orientation_defect_edges","degenerate_face_ids","duplicate_face_ids","vertex_manifold_defect_ids","face_component_count","euler_characteristic","closed_oriented_manifold_candidate")},
 "step0_upright_min_triangle_area_m2":float(areas.min()),"step0_upright_zero_area_faces":int(np.count_nonzero(areas==0)),
 "vertex_position_normal_fields_and_allocation_bytes_identical":True,
 "other_row_index_sequences_exact":True,"full_per_face_bodyparts3d_parent_ancestry_available":False,
 "evidence_boundary":"Local Float32 precision repair candidate only; not native/runtime/anatomical admission."
}
reportpath.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
print(json.dumps({"payload":str(out),"payload_sha256":sha(payload),"manifest":str(manout),"manifest_sha256":sha(manbytes),"ancestry":str(side),"ancestry_sha256":sha(side_bytes),"report":str(reportpath),"report_sha256":sha(reportpath.read_bytes()),"edge_m":edge_m,"before":before_top,"after":after_top,"step0_min_area":float(areas.min()),"interfaces":len(interface_set)},sort_keys=True))

