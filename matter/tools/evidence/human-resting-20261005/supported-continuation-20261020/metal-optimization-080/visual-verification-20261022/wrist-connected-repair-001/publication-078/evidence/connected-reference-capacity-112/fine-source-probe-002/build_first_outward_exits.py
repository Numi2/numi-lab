from __future__ import annotations
import csv,json,hashlib,sys,importlib.util
from pathlib import Path
import numpy as np
D=Path("/Users/n/Work/numi-human-connected-surface-069")
OUT=D/"connected-reference-capacity-112/fine-source-probe-002/first-outward-exits-003"
OUT.mkdir(parents=True,exist_ok=False)
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
COMPACT=D/"whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz"
BASE=D/"connected-cell-witness-103/run-001/baseline"
BANK=E/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
FIELD=D/"closest-exit-clearance-098/source-field-audit-001/source-outward-field.npz"
FIELD_REPORT=D/"closest-exit-clearance-098/source-field-audit-001/source-field-audit.json"
HELPER=D/"closest-exit-clearance-098/outward_ray_exit_098.py"
AUDIT=Path("/Users/n/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
CAP=D/"connected-reference-capacity-112/human_skin_fat_reference_capacity.py"
PRED=D/"connected-cell-trust-107/matter/root_exact_segment_witness.py"
PROBE=D/"connected-reference-capacity-112/fine-source-probe-002/report.json"
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def pin(p,h):
 g=sha(p)
 if g!=h:raise ValueError(f"SHA mismatch {p}: {g}")
 return g
pins={
 "084_compact":pin(COMPACT,"5a35470223ce12343cc217be7822d182802e99f1c423bb52b8409f153f9a49db"),
 "103_candidate":pin(BASE/"candidate.npz","ddb86fbf07183adce0194c59a1acfc5456c96cb2efc709eb587e5489d3d59163"),
 "098_field":pin(FIELD,"048f4f37354475bba48bd9427e8fb6f9723da22e1b5f2c3cf4de7485e386570a"),
 "098_field_report":pin(FIELD_REPORT,"33a7090fec99e497cbe38feaa1b04aa6b58421c75c9ce219b75821704d29c7b7"),
 "098_helper":pin(HELPER,"712b6aa304893cf1ad3aca1edde4ea913d4ab60cb5c0eb3617f5f09732e1743f"),
 "probe_report":pin(PROBE,"d6ae04efd7c788ef289a0d60f16b2b91c77af79ac023897c00ea11bacbb2e447"),
 "exact_predicate":pin(PRED,"d6a2efe451e7743b659d74c274d1372a5ed2a765511a8671b2c1a2370b7d03a2")}
with np.load(COMPACT,allow_pickle=False) as z:
 P=np.asarray(z["current037_points_m_f32"],dtype=np.float32)
 input_rows=np.asarray(z["source_input_row"],dtype=np.int64)
 source_ids=np.asarray(z["source_global_node_ids"],dtype=np.int64)
inv=np.full(len(source_ids),-1,dtype=np.int64)
for outrow,sr in enumerate(input_rows):
 if sr>=0:inv[sr]=outrow
if np.any(inv<0) or len(source_ids)!=7233 or P.shape!=(7360,3):raise ValueError("unexpected compact source domains")
with np.load(FIELD,allow_pickle=False) as z:
 field=np.asarray(z["source_outward_direction_by_compact_output_row"],dtype=np.float64)
 field_ids=np.asarray(z["source_global_node_ids"],dtype=np.int64)
 field_rows=np.asarray(z["source_input_row_by_compact_output_row"],dtype=np.int64)
if not np.array_equal(field_ids,source_ids) or not np.array_equal(field_rows,input_rows) or field.shape!=(7360,3):raise ValueError("098 source field domain mismatch")
probe=json.loads(PROBE.read_text())
requests=[]
for col in probe["results"]:
 c=int(col["column_row"])
 for bank in col["bank_comparisons"]:
  key=(int(bank["body"]),int(bank["stable_bone_id"]))
  for layer,label in ((0,"O"),(1,"I")):
   status=bank["O_endpoint_status"] if layer==0 else bank["I_endpoint_status"]
   if status=="inside":requests.append((c,label,layer,key))
if sum(x[1]=="O" for x in requests)!=12:raise ValueError("expected 12 O-inside relations")
with open(BASE/"query/pairs.csv",newline="") as f:
 qrows=list(csv.DictReader(f))
qmap={(int(r["query_row"]),int(r["bank_body"]),int(r["bank_stable_id"])):r["classification"] for r in qrows}
for c,label,layer,key in requests:
 if qmap.get((c+layer*2411,*key),"outside")!="inside":raise ValueError(f"endpoint class mismatch {c}/{label}/{key}")
sys.path.insert(0,str(AUDIT.parent))
import human_skin_fat_bone_audit as audit
tri,ids,gids,bodies,Mats,offsets,positions,bankkeys,bankpins,exact=audit.read_bank_inputs(BANK,D/"connected-cell-trust-107/matter")
if len(bankkeys)!=185:raise ValueError("incomplete typed bank inventory")
spec=importlib.util.spec_from_file_location("outward098",HELPER)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
sys.path.insert(0,str(CAP.parent))
import human_skin_fat_reference_capacity as cap
records=[]
for c,label,layer,key in requests:
 body,bone=key;mask=(ids[:,0]==body)&(ids[:,1]==bone)
 bt=tri[mask];bid=ids[mask,3].astype(np.int64)
 bank={"triangles_local":bt,"triangle_stable_ids":bid,"expected_triangle_count":len(bt),
 "aabb_min_local":bt.min(axis=(0,1)),"aabb_max_local":bt.max(axis=(0,1)),
 "world_to_local":Mats[body],"body_position_world":positions[body],"com_offset":offsets[body]}
 b=cap.prepare_bone_bank(bank)
 queryrow=c+layer*2411;start_world=P[queryrow].astype(np.float64);outrow=int(inv[queryrow])
 direction_world=field[outrow];direction_local=Mats[body]@direction_world;direction_local=direction_local/np.linalg.norm(direction_local)
 start_local=cap.world_to_bank_local(start_world,body_position_world=b.body_position_world,world_to_local=b.world_to_local,com_offset=b.com_offset)
 end_local,ep=mod.ray_endpoint_outside_aabb(start_local,direction_local,b.aabb_min,b.aabb_max,margin_m=2e-5)
 result=mod.certify_first_outward_exit(start_status="inside",start_local=start_local,end_local=end_local,
 source_direction_local=direction_local,triangles_local=b.triangles,triangle_stable_ids=b.stable_ids,
 bank_aabb_min_local=b.aabb_min,bank_aabb_max_local=b.aabb_max,expected_bank_triangle_count=b.expected_triangle_count,
 world_start=start_world,world_to_local=b.world_to_local,clearance_m=5e-5,
 exact_classifier=exact["classify_segment_triangle"],exact_classifier_sha256=pins["exact_predicate"])
 hit=result.get("selected_first_exit")
 records.append({"column_row":c,"layer":label,"query_row":queryrow,"body":body,"bone_stable_id":bone,
 "complete_bank_triangle_count":len(bt),"world_start_m":start_world.tolist(),"source_direction_world_unit":direction_world.tolist(),
 "local_source_direction_unit":direction_local.tolist(),"clearance_m":5e-5,"status":result["status"],
 "first_exit_distance_m":None if hit is None else float(hit["t"]*result["segment_length_m"]),
 "report":result,"aabb_endpoint":ep})
report={"schema":"098-source-OD-first-outward-exit-planes-003","status":"CONSTRUCTION_PLANES_ONLY",
"input_pins":pins,"direction_definition":"Pinned 098 source field normalize(O-D) at source input row, prolonged through declared native parent tetrahedra; field audit reports 7,233 direct +127 generated, zero unresolved, max parent direction spread 4.61e-15.",
"scope":"First outward exit for every O- or I-inside relation among the 15 selected columns, against each implicated complete typed bank. Exact start classes were rechecked against frozen 103 all-bank endpoint query. 50um helper clearance is the existing construction margin, not an anatomical thickness target.",
"limitations":["O-D direction is a construction field, not anatomical outward proof.","Planes are per complete bank, not whole-surface/tet clearance or native/physical admission.","Overlapping constraints across banks remain separate."],"results":records}
p=OUT/"planes.json";p.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"path":str(p),"sha256":sha(p),"rows":len(records),"O_rows":sum(r["layer"]=="O" for r in records),"I_rows":sum(r["layer"]=="I" for r in records),"passes":sum(r["status"]=="PASS" for r in records),"failures":[{"column":r["column_row"],"layer":r["layer"],"bank":[r["body"],r["bone_stable_id"]],"reason":r["report"]["reason"]} for r in records if r["status"]!="PASS"],"exit_distances_mm":[None if r["first_exit_distance_m"] is None else round(r["first_exit_distance_m"]*1000,6) for r in records]},indent=2))
