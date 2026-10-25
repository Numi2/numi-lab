from __future__ import annotations
import csv,hashlib,json,sys,gzip,time,math
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np

D=Path("/Users/numi/Work/numi-human-connected-surface-069")
HERE=Path(__file__).resolve().parent
OUT=HERE/"run-002"
assert not OUT.exists()
OUT.mkdir(parents=True)
BASE=D/"connected-cell-witness-103/run-001/baseline"
CAND=BASE/"candidate.npz"
COMPACT=D/"whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz"
QUERY=BASE/"query"
BANK=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight")
AUDIT_PATH=Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
RAY_DIR=D/"connected-cell-trust-107/matter/tools"
PRED=D/"connected-cell-trust-107/matter/root_exact_segment_witness.py"
PY=sys.executable
EXPECTED={
 "candidate": "ddb86fbf07183adce0194c59a1acfc5456c96cb2efc709eb587e5489d3d59163",
 "compacted084": "5a35470223ce12343cc217be7822d182802e99f1c423bb52b8409f153f9a49db",
 "query_pairs": "95023ef49a865473ab8baaca65da37bc45d78d5b4adf9e16ee7effa604bad4eb",
 "query_nodes": "e49ef0e61ed9551bc3782e3bb8def6a45b5b2b32124ce1c7b7dabb9e286badef",
 "query_points": "f15ea806a61c6a70fb4d051c932df22ac173b1f39e3f60662fad396362a8b58d",
 "predicate": "d6a2efe451e7743b659d74c274d1372a5ed2a765511a8671b2c1a2370b7d03a2",
 "ray_helper": "116a32e8080c8470c36d3424d52e2deae17838bb46e93b28eec9d75db1ff5fc1",
 "query_binary": "5dc7b5ce2a96f58bd03ccc85e38645d9da2ec8be105a532c461a3e0cc6d25460"
}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def pin(p,name):
 actual=sha(p)
 exp=EXPECTED.get(name)
 if exp is not None and actual!=exp:raise ValueError(f"{name} SHA mismatch: {actual}")
 return actual
if pin(CAND,"candidate")!=EXPECTED["candidate"]:raise RuntimeError("candidate pin")
if pin(COMPACT,"compacted084")!=EXPECTED["compacted084"]:raise RuntimeError("084 pin")
if pin(QUERY/"pairs.csv","query_pairs")!=EXPECTED["query_pairs"]:raise RuntimeError("query pair pin")
if pin(QUERY/"nodes.csv","query_nodes")!=EXPECTED["query_nodes"]:raise RuntimeError("query node pin")
if pin(QUERY/"points.csv","query_points")!=EXPECTED["query_points"]:raise RuntimeError("query point pin")
if pin(PRED,"predicate")!=EXPECTED["predicate"]:raise RuntimeError("exact predicate pin")
if pin(RAY_DIR/"human_skin_fat_bone_clearance.py","ray_helper")!=EXPECTED["ray_helper"]:raise RuntimeError("ray helper pin")
# Load exact triangle/body rows via the source audit owner; it checks F32 replay,
# raw body-frame rotations, typed-bank inventory, and complete exact helper hash.
sys.path.insert(0,str(AUDIT_PATH.parent))
import human_skin_fat_bone_audit as audit
tri,ids,gids,bodies,matrices,offsets,positions,bankkeys,bank_pins,exact=audit.read_bank_inputs(
    BANK,D/"connected-cell-trust-107/matter")
if len(bankkeys)!=185:raise ValueError("expected exactly 185 typed banks")
if exact["classify_segment_triangle"] is None:raise ValueError("missing pinned exact classifier")
import importlib.util
spec=importlib.util.spec_from_file_location("capacity_exact_t_helper",RAY_DIR/"human_skin_fat_bone_clearance.py")
ray=importlib.util.module_from_spec(spec);spec.loader.exec_module(ray)
sys.path.insert(0,str(HERE))
import human_skin_fat_reference_capacity as cap
# Bind exact helper functions to explicit source hashes.
classifier=exact["classify_segment_triangle"]
exact_parameter=ray._exact_t_barycentric
clearance_m=5e-5  # Existing 107 construction margin, not a biological thickness.
z=np.load(CAND,allow_pickle=False)
points=np.asarray(z["points_world_f32"],dtype=np.float32)
if points.shape!=(7360,3):raise ValueError(f"expected 7360 baseline world rows, got {points.shape}")
if z["tetrahedra"].shape!=(32796,4):raise ValueError("baseline tet topology changed")
z2=np.load(COMPACT,allow_pickle=False)
points084=np.asarray(z2["current037_points_m_f32"],dtype=np.float32)
if not np.array_equal(points.view(np.uint32),points084.view(np.uint32)):
 raise ValueError("103 baseline positions differ from exact 084 compacted current input")
if points.dtype!=np.float32:raise ValueError("baseline emitted points must be Float32")
# Validate the point CSV including its exact F32 bit representation.
with (QUERY/"points.csv").open(newline="") as f:
 qrows=list(csv.DictReader(f))
if len(qrows)!=len(points):raise ValueError("baseline query point row count mismatch")
receipt=json.loads((QUERY/"query-receipt.json").read_text())
if receipt.get("binary_sha256")!=EXPECTED["query_binary"]:raise ValueError("endpoint query binary is not the pinned 042 static-query producer")
if receipt.get("query_rows")!=len(points) or receipt.get("identity_domain")!="mechanics_node_row":raise ValueError("endpoint query receipt domain/count mismatch")
for i,row in enumerate(qrows):
 if int(row["query_row"])!=i:raise ValueError("query point row ordering mismatch")
 bits=np.asarray([int(row[k]) for k in ("x_bits_u32","y_bits_u32","z_bits_u32")],dtype=np.uint32)
 if not np.array_equal(points[i].view(np.uint32),bits):
  raise ValueError(f"candidate/query point bit mismatch at row {i}")
# Exact endpoint classifications are carried by complete 185-bank summary plus
# one row for each bank whose query point fell within that bank's AABB.
node_rows=list(csv.DictReader((QUERY/"nodes.csv").open(newline="")))
pair_rows=list(csv.DictReader((QUERY/"pairs.csv").open(newline="")))
if len(node_rows)!=len(points):raise ValueError("query summary coverage changed")
bankset={(int(x[0]),int(x[1])) for x in bankkeys}
pairmap={}
for r in pair_rows:
 key=(int(r["query_row"]),int(r["bank_body"]),int(r["bank_stable_id"]))
 if key in pairmap:raise ValueError(f"duplicate endpoint bank row {key}")
 if (key[1],key[2]) not in bankset:raise ValueError(f"unknown typed bank in endpoint query {key}")
 if r["classification"] not in ("inside","outside") or int(r["ambiguous"]):
  raise ValueError(f"non-binary endpoint classification {key}")
 pairmap[key]=r
endpoint={}
endpoint_pair_counts=[]
max_local_residual=0.
max_local_witness=None
for i,nr in enumerate(node_rows):
 if int(nr["query_row"])!=i or int(nr["total_bone_banks"])!=185:
  raise ValueError(f"query summary identity mismatch at row {i}")
 amb=int(nr["ambiguous_banks"]);inv=int(nr["invalid_banks"])
 if amb or inv:raise ValueError(f"endpoint query has ambiguous/invalid bank rows at {i}: {amb}/{inv}")
 present=[k for k in pairmap if k[0]==i]
 count=len(present);aabb=int(nr["aabb_proven_outside_banks"])
 if count+aabb!=185:raise ValueError(f"endpoint 185-bank coverage mismatch at row {i}: {count}+{aabb}")
 if count!=int(nr["parity_query_banks"]):raise ValueError(f"query pair count differs from summary at row {i}")
 endpoint_pair_counts.append(count)
 for body,bone in bankset:
  k=(i,body,bone)
  if k not in pairmap:
   endpoint[k]="outside"  # precisely: query proved outside complete bank AABB
   continue
  r=pairmap[k];endpoint[k]=r["classification"]
  # Reproduce query's raw world-to-local row transform for every reported bank/node.
  p=points[i].astype(np.float64)
  local=(p-positions[body])@matrices[body].T-offsets[body]
  qlocal=np.asarray([float(r["query_local_"+c]) for c in "xyz"],dtype=np.float64)
  residual=float(np.linalg.norm(local-qlocal))
  if residual>max_local_residual:max_local_residual=residual;max_local_witness={"query_row":i,"body":body,"bone":bone}
if len(endpoint)!=len(points)*len(bankset):raise ValueError("typed endpoint classifications are not complete")
# Cache each complete bank's exact triangles, stable IDs, AABB, and frame once.
prepared={}
bank_candidate_counts={}
for body0,bone0 in bankkeys:
 body,bone=int(body0),int(bone0)
 mask=(ids[:,0]==body)&(ids[:,1]==bone)
 bt=tri[mask];bid=ids[mask,3].astype(np.int64)
 if len(bt)==0:raise ValueError(f"empty typed bank {(body,bone)}")
 record={"triangles_local":bt,"triangle_stable_ids":bid,"expected_triangle_count":len(bt),
   "aabb_min_local":bt.min(axis=(0,1)),"aabb_max_local":bt.max(axis=(0,1)),
   "world_to_local":matrices[body],"body_position_world":positions[body],"com_offset":offsets[body]}
 prepared[(body,bone)]=cap.prepare_bone_bank(record)
# Verify the exact query-local conversion on all observed endpoint-bank pairs.
if max_local_residual>2e-8:raise ValueError(f"endpoint body-frame parity exceeds 107 tolerance: {max_local_residual:.9g} m")
# The conservative numerical broadphase padding only widens candidate segment
# boxes; final hits are still decided by the pinned exact binary-rational predicate.
paths={"OI":(0,2411),"ID":(2411,4822),"OD":(0,4822)}
node_count=2411
local_points={}
broadphase_counts=Counter()
for key,bank in prepared.items():
 body=key[0]
 local_points[key]=(points[:7233].astype(np.float64)-positions[body])@matrices[body].T-offsets[body]
 local_points[key]=local_points[key].reshape(3,node_count,3)
# Keep every exact candidate-segment receipt; disjoint segment/AABB pairs are
# counted after proving both endpoints outside all-bank AABB query.
jsonl=OUT/"exact-segment-audits.jsonl.gz"
column_records=[{"column_row":i,"O_world_m":points[i].tolist(),"I_world_m":points[i+2411].tolist(),"D_world_m":points[i+4822].tolist(),
 "OI_entries":[],"ID_entries":[],"OD_entries":[],"OI_failures":[],"ID_failures":[],"OD_failures":[]} for i in range(node_count)]
segment_count=Counter();status_count=Counter();cross_bank_ties=Counter()
start=time.monotonic()
with gzip.open(jsonl,"wt",encoding="utf-8",compresslevel=6) as out:
 for (body,bone),bank in sorted(prepared.items()):
  lo,hi=bank.aabb_min,bank.aabb_max
  lp=local_points[(body,bone)]
  # Factorial guard is a conservative Float32 arithmetic broadphase bound,
  # not a physical margin. Query-local residual is also included.
  scale=max(1.,float(np.max(np.abs(points[:7233]))),float(np.max(np.abs(positions[body]))),
            float(np.max(np.abs(lo))),float(np.max(np.abs(hi))),float(np.max(np.abs(offsets[body]))))
  broad_pad=max(8*np.finfo(np.float32).eps*scale,max_local_residual+64*np.finfo(np.float64).eps*scale)
  bank_candidate_counts[(body,bone)]=0
  for route,(first,second) in paths.items():
   A=lp[first//node_count] if first>=2411 else lp[0]
   B=lp[second//node_count] if second>=2411 else lp[0]
   # Correct layer indexing (O=0, I=1, D=2).
   layer_a={"OI":0,"ID":1,"OD":0}[route]
   layer_b={"OI":1,"ID":2,"OD":2}[route]
   A=lp[layer_a];B=lp[layer_b]
   seglo=np.minimum(A,B);seghi=np.maximum(A,B)
   boxhit=np.all(seghi>=lo-broad_pad,axis=1)&np.all(seglo<=hi+broad_pad,axis=1)
   for col in range(node_count):
    sa=endpoint[(layer_a*node_count+col,body,bone)]
    sb=endpoint[(layer_b*node_count+col,body,bone)]
    if not boxhit[col]:
     if sa!="outside" or sb!="outside":
      rec={"column_row":col,"route":route,"body":body,"bone":bone,"status":"REJECT",
           "reason":"segment AABB disjoint while endpoint query is not outside"}
      out.write(json.dumps(rec,separators=(",",":"))+"\n")
      column_records[col][route+"_"+("failures" if route!="OD" else "failures")].append(rec)
      segment_count[route]+=1;status_count["broadphase_contradiction"]+=1
      continue
     broadphase_counts[route]+=1
     continue
    # Exact bank-wide triangle candidate clipping for this one segment.
    result=cap.audit_exact_segment(points[layer_a*node_count+col],points[layer_b*node_count+col],
       start_status=sa,end_status=sb,bank=bank,exact_classifier=classifier,exact_parameter=exact_parameter,
       exact_classifier_sha256=EXPECTED["predicate"],exact_parameter_sha256=EXPECTED["ray_helper"])
    segment_count[route]+=1;status_count[(route,result["status"])]+=1
    bank_candidate_counts[(body,bone)]+=1
    rec={"column_row":col,"route":route,"body":body,"bone":bone,"status":result["status"],
         "result":result}
    out.write(json.dumps(rec,separators=(",",":"))+"\n")
    if result["status"]!="PASS":
     column_records[col][route+"_failures"].append({"body":body,"bone":bone,"reason":result["reason"],
        "unresolved_witnesses":result["unresolved_witnesses"],"equal_parameter_groups":result["equal_parameter_groups"]})
    if result["first_entry"] is not None:
     column_records[col][route+"_entries"].append({
      "body":body,"bone":bone,"t_numerator":result["first_entry"]["t_numerator"],
      "t_denominator":result["first_entry"]["t_denominator"],"t":result["first_entry"]["t"],
      "distance_m":result["first_entry"]["t"]*result["segment_length_m"],
      "triangle_stable_id":result["first_entry"]["triangle_stable_id"],
      "hit_world_m":result["first_entry"]["hit_world"]})
# Global first-event selection across all typed banks. Cross-bank ties remain
# unresolved; this implementation does not coalesce shared geometric features.
def earliest(entries, route, column):
 if not entries:return None,False
 ordered=sorted(entries,key=lambda r:(int(r["t_numerator"])/int(r["t_denominator"]),r["body"],r["bone"]))
 # Use exact Fraction comparison, not the float sort key, for final order/ties.
 from fractions import Fraction
 ordered=sorted(entries,key=lambda r:(Fraction(int(r["t_numerator"]),int(r["t_denominator"])),r["body"],r["bone"]))
 first=Fraction(int(ordered[0]["t_numerator"]),int(ordered[0]["t_denominator"]))
 tied=[r for r in ordered if Fraction(int(r["t_numerator"]),int(r["t_denominator"]))==first]
 return ordered[0],len(tied)>1
columns_path=OUT/"column-capacity-proposals.csv"
with columns_path.open("w",newline="") as f:
 w=csv.writer(f)
 w.writerow(["column_row","O_to_I_first_entry_bank_count","I_to_D_first_entry_bank_count","O_to_D_first_entry_bank_count",
  "OD_first_entry_body","OD_first_entry_bone","OD_first_entry_triangle","OD_first_entry_t",
  "OD_soft_path_capacity_m_after_50um_construction_clearance","OI_path_intersects_or_endpoint_inside",
  "O_endpoint_inside_bank_count","I_endpoint_inside_bank_count","OI_equal_t_tie","ID_equal_t_tie","OD_equal_t_tie","capacity_status",
  "ID_first_entry_body","ID_first_entry_bone","ID_first_entry_triangle","ID_first_entry_distance_m",
  "D_action","D_safe_parameter_from_I","D_safe_x_m","D_safe_y_m","D_safe_z_m",
  "O_to_D_I_line_perp_distance_m","O_to_D_projected_I_parameter","exact_segment_failures"])
 for c in column_records:
  oi,oitie=earliest(c["OI_entries"],"OI",c["column_row"])
  ide,idtie=earliest(c["ID_entries"],"ID",c["column_row"])
  ode,odtie=earliest(c["OD_entries"],"OD",c["column_row"])
  fails=len(c["OI_failures"])+len(c["ID_failures"])+len(c["OD_failures"])
  O=points[c["column_row"]].astype(np.float64);I=points[c["column_row"]+2411].astype(np.float64);D0=points[c["column_row"]+4822].astype(np.float64)
  od=D0-O;od2=float(od@od)
  proj=float(((I-O)@od)/od2) if od2>0 else float("nan")
  perp=float(np.linalg.norm((I-O)-proj*od)) if od2>0 else float("nan")
  capacity=float(ode["distance_m"]-clearance_m) if ode is not None else float("nan")
  skin_cross=bool(c["OI_entries"])
  action="unchanged";safe_alpha=float("nan");safeD=D0.copy()
  if fails or oitie or idtie or odtie:
   state="WITHHELD_UNRESOLVED"
  elif any(endpoint[(c["column_row"],b,bo)]!="outside" for b,bo in bankset) or any(endpoint[(c["column_row"]+2411,b,bo)]!="outside" for b,bo in bankset):
   state="REQUIRES_SKIN_RECONSTRUCTION";skin_cross=True
  elif skin_cross:
   state="REQUIRES_SKIN_RECONSTRUCTION"
  elif ide is not None:
   state="NO_FEASIBLE_D_SHORTENING" if ide["distance_m"]<=clearance_m else "PROPOSED_D_SHORTENING"
   if state=="PROPOSED_D_SHORTENING":
    length=float(np.linalg.norm(D0-I));safe_alpha=(ide["distance_m"]-clearance_m)/length
    safeD=I+safe_alpha*(D0-I);action="shorten_on_original_I_to_D_segment"
   else:action="withheld"
  else:state="NO_I_TO_D_ENTRY_D_UNCHANGED"
  o_inside_count=sum(endpoint[(c["column_row"],b,bo)]=="inside" for b,bo in bankset)
  i_inside_count=sum(endpoint[(c["column_row"]+2411,b,bo)]=="inside" for b,bo in bankset)
  w.writerow([c["column_row"],len(c["OI_entries"]),len(c["ID_entries"]),len(c["OD_entries"]),
   ode["body"] if ode else "",ode["bone"] if ode else "",ode["triangle_stable_id"] if ode else "",
   ode["t"] if ode else "",capacity if ode else "",int(skin_cross),o_inside_count,i_inside_count,int(oitie),int(idtie),int(odtie),state,
   ide["body"] if ide else "",ide["bone"] if ide else "",ide["triangle_stable_id"] if ide else "",
   ide["distance_m"] if ide else "",action,safe_alpha,*safeD.tolist(),perp,proj,fails])
report={
 "schema":"connected-reference-capacity-112-run-v2","input_candidate":str(CAND),
 "input_candidate_sha256":sha(CAND),"input_084_compacted_sha256":sha(COMPACT),
 "endpoint_query_pair_sha256":sha(QUERY/"pairs.csv"),"endpoint_query_nodes_sha256":sha(QUERY/"nodes.csv"),
 "endpoint_query_points_sha256":sha(QUERY/"points.csv"),"query_receipt_sha256":sha(QUERY/"query-receipt.json"),
 "typed_bank_pins":bank_pins,"exact_classifier_sha256":EXPECTED["predicate"],
 "rational_parameter_source_sha256":EXPECTED["ray_helper"],
 "endpoint_count":len(points),"O_I_D_columns":node_count,"complete_typed_bank_count":len(bankset),
 "endpoint_bank_status_rows":len(endpoint),"endpoint_query_pairs":len(pair_rows),
 "endpoint_query_pair_min_per_node":min(endpoint_pair_counts),"endpoint_query_pair_max_per_node":max(endpoint_pair_counts),
 "endpoint_ambiguities_invalid":0,"max_query_local_transform_residual_m":max_local_residual,
 "max_query_local_residual_witness":max_local_witness,"aabb_broadphase_pad_rule":"max(8*eps32*coordinate_scale, measured_query_local_residual+64*eps64*scale); widens candidate boxes only",
 "exact_segments_audited":dict(segment_count),"aabb_disjoint_segment_proofs":dict(broadphase_counts),
 "exact_segment_status_counts":{f"{k[0]}:{k[1]}":v for k,v in status_count.items()},
 "cross_bank_equal_parameter_policy":"fail closed; no cross-bank geometric-feature coalescing",
 "declared_clearance_m":clearance_m,"clearance_source":"existing 107 construction parameter in first_exit_contact, 50 micrometres; not a biological thickness target",
 "output_columns_csv":str(columns_path),"output_columns_sha256":sha(columns_path),
 "output_exact_segment_jsonl_gz":str(jsonl),"output_exact_segment_jsonl_gz_sha256":sha(jsonl),
 "column_count":len(column_records),"run_wall_seconds":time.monotonic()-start,
 "scope":"offline reference path-capacity audit only; O/I/D proposals retain original O and I, D changes only along the original I-D segment; no mass/topology/native/biological admission"
}
(OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
pins={"candidate":sha(CAND),"084_compacted":sha(COMPACT),
 "query_pairs":sha(QUERY/"pairs.csv"),"query_nodes":sha(QUERY/"nodes.csv"),
 "query_points":sha(QUERY/"points.csv"),"query_receipt":sha(QUERY/"query-receipt.json"),
 "predicate":sha(PRED),"ray_parameter_helper":sha(RAY_DIR/"human_skin_fat_bone_clearance.py"),
 "bank_triangles":bank_pins["exact-emitted.triangles.csv"],"bank_nodes":bank_pins["exact-emitted.nodes.csv"],
 "bank_bodies":bank_pins["exact-emitted.bodies.csv"],"bank_inventory":bank_pins["all-bank-containment.csv.banks.csv"],
 "script_sha256":sha(__file__),"helper_sha256":sha(HERE/"human_skin_fat_reference_capacity.py")}
(OUT/"input-pins.json").write_text(json.dumps(pins,indent=2,sort_keys=True)+"\n")
print(json.dumps({"report":str(OUT/"report.json"),"segments":dict(segment_count),
 "statuses":{f"{k[0]}:{k[1]}":v for k,v in status_count.items()},
 "aabb_disjoint":dict(broadphase_counts),"wall_s":time.monotonic()-start},sort_keys=True))
