#!/usr/bin/env python3
"""Sequentially replay pinned 368 and 378 flip edge whitelists on a 375 point state.

No edge is substituted. 358 re-evaluates each allowed edge under current and
matched-037 orientation/area gates; invalid operations are reported and omitted.
This writes a topology proposal only and runs no geometry/bone audits.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
FLIP_PATH=ROOT/"D/connected-remesh-reuse-358/same_parent_flips_358.py"
REPLAY368=ROOT/"barycentric-connected-clearance-170/same-parent-remesh-368/run-001/replay-report.json"
CAND368=ROOT/"barycentric-connected-clearance-170/same-parent-remesh-368/run-001/candidate.npz"
AUDIT368=ROOT/"barycentric-connected-clearance-170/same-parent-remesh-368/run-001/audit-159/report.json"
SUBSET378=ROOT/"barycentric-connected-clearance-170/bone-aware-remesh-370/run-002-self-safe-subset/subset-report.json"
CAND378=ROOT/"barycentric-connected-clearance-170/bone-aware-remesh-370/run-002-self-safe-subset/candidate.npz"
BASE362=ROOT/"barycentric-connected-clearance-170/coherent-support-continuation-362/uniform-scale-recovery/backtrack/pass-001/candidate.npz"
PINS={
 "flip358":"a08e1f7e6392854f72ffab09af6f2b1f2afe885b1a84f586afe71160ff3ccbee",
 "replay368":"fe945162d424bb24eca9d3afe63a1ada56800b7c449523805e3a724bc11feeb0",
 "candidate368":"9eb42003871a9a95bed5c56906f97bdc3c3f996486ddade3a10fd9bac0437a88",
 "audit368":"d5674389a55adeae71c20f9f3c81ad05c04ca8fbf2c4167351973fdefaf0c97d",
 "subset378":"9ed0c96f96dc5cefcc07b13980445ffc04135fbc5739dca049bd4d2d42679104",
 "candidate378":"d60b58c1babc739a4a0bc53ad43c4c0623d0181a8b80dc2ac00a02d6ba11b630",
 "base362":"814b879cb8f3af988aeaa0540d183222d942e95f5597434b6328f714d35e14d3",
}
class ReplayError(ValueError): pass

def sha(path:Path)->str:
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def arrhash(a:np.ndarray)->str:
 return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def import_flip():
 if sha(FLIP_PATH)!=PINS["flip358"]:raise ReplayError("358 source pin mismatch")
 spec=importlib.util.spec_from_file_location("same_parent_flips_358_for_381",FLIP_PATH)
 if spec is None or spec.loader is None:raise ReplayError("cannot load 358")
 mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 return mod

def sig(op:Mapping[str,Any])->tuple:
 return tuple(sorted(map(int,op["edge"]))),tuple(map(int,op["face_rows"])),int(op["parent_row"])

def _edge_face_map(faces:np.ndarray)->dict[tuple[int,int],tuple[int,...]]:
 found={}
 for row,tri in enumerate(np.asarray(faces)):
  for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
   e=tuple(sorted((int(a),int(b))))
   found.setdefault(e,[]).append(int(row))
 return {k:tuple(sorted(v)) for k,v in found.items()}

def eligible_on_current_topology(data:Mapping[str,np.ndarray],ops:Sequence[Mapping[str,Any]])->tuple[list[Mapping[str,Any]],list[dict]]:
 faces=np.asarray(data["source_O_faces"]); parents=np.asarray(data["refinement_face_parent_face_row"])
 edge_rows=_edge_face_map(faces); eligible=[]; excluded=[]
 for op in ops:
  edge=tuple(sorted(map(int,op["edge"]))); rows=tuple(map(int,op["face_rows"])); parent=int(op["parent_row"])
  if edge_rows.get(edge)!=rows:
   excluded.append({"edge":list(edge),"face_rows":list(rows),"reason":"edge_adjacency_or_pair_rows_changed"})
   continue
  if any(r<0 or r>=len(faces) for r in rows) or parents[list(rows)].tolist()!=[parent,parent]:
   excluded.append({"edge":list(edge),"face_rows":list(rows),"reason":"immediate_parent_mismatch"})
   continue
  eligible.append(op)
 return eligible,excluded

def _load_npz(path:Path)->dict[str,np.ndarray]:
 with np.load(path,allow_pickle=False) as z:return {k:np.asarray(z[k]).copy() for k in z.files}

def _changed_face_rows(a:np.ndarray,b:np.ndarray)->set[int]:
 if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3:raise ReplayError("face array shapes changed")
 return set(map(int,np.flatnonzero(np.any(a!=b,axis=1))))

def _target_rows(report:dict,face_count:int)->list[int]:
 if report.get("status")!="COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT" or report.get("complete") is not True or report.get("bank_count")!=185 or report.get("expected_bank_count")!=185 or len(report.get("banks",[]))!=185 or report.get("surface_face_count")!=face_count:
  raise ReplayError("368 baseline is not a complete 185-bank audit on expected face domain")
 rows=set()
 keys=set()
 for bank in report["banks"]:
  key=(bank.get("body"),bank.get("stable_bone_id"))
  if type(key[0]) is not int or type(key[1]) is not int or key in keys:raise ReplayError("malformed or duplicate typed bank")
  keys.add(key)
  result=bank.get("result")
  if result is None:
   if bank.get("skin_aabb_active")!=0 or bank.get("bank_aabb_active")!=0:raise ReplayError("null typed bank lacks AABB exclusion")
  else:
   pairs=result.get("triangle_pairs",[])
   if result.get("count")!=len(pairs):raise ReplayError("159 pair count mismatch")
   for p in pairs:
    if not isinstance(p,list) or len(p)!=2 or type(p[0]) is not int or not 0<=p[0]<face_count:raise ReplayError("malformed 159 pair")
    rows.add(p[0])
 if len(keys)!=185:raise ReplayError("incomplete typed-bank inventory")
 return sorted(rows)

def _eq(a:np.ndarray,b:np.ndarray)->bool:
 aa,bb=np.asarray(a),np.asarray(b)
 if aa.shape!=bb.shape or aa.dtype!=bb.dtype:return False
 if np.issubdtype(aa.dtype,np.floating):return bool(np.array_equal(aa,bb,equal_nan=True))
 return bool(np.array_equal(aa,bb))

def _assert_input_matches_362(candidate:dict[str,np.ndarray]):
 base=_load_npz(BASE362)
 required=("source_O_faces","node_kind","node_source_row","node_source_virtual_row",
  "node_parent_coarse_face_row","node_parent_coarse_edge_a","node_parent_coarse_edge_b",
  "node_parent_coarse_edge_parameter","face_kind","parent_coarse_face_row",
  "parent_fine_detail_face_row","parent_source_face_marker","parent_fine_subface_row",
  "coarse_boundary_edge_endpoints","coarse_boundary_edge_path_indptr",
  "coarse_boundary_edge_path_output_nodes","coarse_boundary_edge_path_source_nodes",
  "coarse_boundary_edge_path_parameters","face_source_ancestry_indptr",
  "face_source_ancestry_face_rows","face_fine_subface_ancestry_indptr",
  "face_fine_subface_ancestry_rows","refinement_node_kind","refinement_node_parent_edge_rows",
  "refinement_node_parent_edge_weights","refinement_node_parent_face_indptr",
  "refinement_node_parent_face_rows","refinement_node_parent_face_barycentric",
  "refinement_face_parent_face_row","refinement_face_parent_barycentric",
  "source_prolongation_indptr","source_prolongation_node_rows","source_prolongation_weights",
  "prior_input_parent__source_prolongation_indptr","prior_input_parent__source_prolongation_node_rows",
  "prior_input_parent__source_prolongation_weights","prior_input_006__source_prolongation_indptr",
  "prior_input_006__source_prolongation_node_rows","prior_input_006__source_prolongation_weights",
  "prior_run028__source_prolongation_indptr","prior_run028__source_prolongation_node_rows",
  "prior_run028__source_prolongation_weights","immediate_refinement_prolongation_indptr",
  "immediate_refinement_prolongation_node_rows","immediate_refinement_prolongation_weights",
  "area_reference_037_points_world_m_f64","area_reference_037_source_O_faces",
  "coarse_boundary_anchor_rows","coarse_boundary_anchor_source_nodes",
  "typed_coarse_anchor_output_rows_018","typed_coarse_anchor_source_ids_018")
 for key in required:
  if key not in candidate or key not in base or not _eq(candidate[key],base[key]):
   raise ReplayError(f"375 input topology/ancestry differs from exact 362 input: {key}")
 for key in candidate:
  if "anchor" in key and (key not in base or not _eq(candidate[key],base[key])):
   raise ReplayError(f"375 changed pinned native anchor identity array: {key}")
 if candidate["points_world_m"].shape!=base["points_world_m"].shape:
  raise ReplayError("375 point domain differs from 362")

def _verify_ancestry_union(before:dict[str,np.ndarray],after:dict[str,np.ndarray],op:Mapping[str,Any]):
 rows=list(map(int,op["face_rows"]))
 for ptr,val in (("face_source_ancestry_indptr","face_source_ancestry_face_rows"),
                 ("face_fine_subface_ancestry_indptr","face_fine_subface_ancestry_rows")):
  u=set(map(int,np.asarray(before[val])[np.asarray(before[ptr])[rows[0]]:np.asarray(before[ptr])[rows[0]+1]])) | set(map(int,np.asarray(before[val])[np.asarray(before[ptr])[rows[1]]:np.asarray(before[ptr])[rows[1]+1]]))
  for r in rows:
   got=set(map(int,np.asarray(after[val])[np.asarray(after[ptr])[r]:np.asarray(after[ptr])[r+1]]))
   if got!=u:raise ReplayError(f"{ptr} ancestry union mismatch on {rows}")

def replay(data:dict[str,np.ndarray],ops:list[dict],seeds:list[int],flip:Any,expected_faces:np.ndarray,stage:str)->tuple[dict,dict]:
 eligible,excluded=eligible_on_current_topology(data,ops)
 expected_sigs={sig(op) for op in eligible}
 edges=[list(map(int,op["edge"])) for op in eligible]
 out,report=flip.propose_flip_batch(data,seeds,max_flips=len(edges),min_angle_gain_deg=1e-8,
                                    min_normal_dot=0.0,allowed_edges=edges)
 generated=report.get("operations",[])
 generated_sigs={sig(op) for op in generated}
 if not generated_sigs.issubset(expected_sigs):
  raise ReplayError(f"{stage} produced a flip outside its pinned whitelist")
 if len(generated)!=len(generated_sigs):raise ReplayError(f"{stage} generated duplicate operations")
 for op in generated:
  rows=list(map(int,op["face_rows"]))
  if not np.array_equal(out["source_O_faces"][rows],expected_faces[rows]):
   raise ReplayError(f"{stage} replacement faces differ from pinned source topology")
  _verify_ancestry_union(data,out,op)
 if not np.array_equal(out["points_world_m"],data["points_world_m"]):raise ReplayError(f"{stage} changed points")
 if not np.array_equal(out["source_O_faces"],out["area_reference_037_source_O_faces"]):raise ReplayError(f"{stage} desynchronized 037 connectivity")
 if not _eq(out["area_reference_037_points_world_m_f64"],data["area_reference_037_points_world_m_f64"]):raise ReplayError(f"{stage} changed 037 reference points")
 mutable={"source_O_faces","area_reference_037_source_O_faces","refinement_face_parent_barycentric",
  "face_source_ancestry_indptr","face_source_ancestry_face_rows","face_fine_subface_ancestry_indptr",
  "face_fine_subface_ancestry_rows","parent_coarse_face_row","parent_fine_detail_face_row",
  "parent_source_face_marker","parent_fine_subface_row","same_parent_flip_349_face_rows",
  "same_parent_flip_349_pair_face_rows","same_parent_flip_349_parent_rows","same_parent_flip_349_semantics"}
 allowed_created={"same_parent_flip_349_face_rows","same_parent_flip_349_pair_face_rows","same_parent_flip_349_parent_rows","same_parent_flip_349_semantics"}
 if set(out)-set(data) - allowed_created:raise ReplayError(f"{stage} added unexpected array keys: {sorted(set(out)-set(data)-allowed_created)}")
 if set(data)-set(out):raise ReplayError(f"{stage} removed array keys: {sorted(set(data)-set(out))}")
 for key in data:
  if key not in mutable and not _eq(data[key],out[key]):raise ReplayError(f"{stage} changed non-topology array {key}")
 report["expected_pinned_operation_count"]=len(ops)
 report["eligible_current_topology_count"]=len(eligible)
 report["excluded_topology_conflicts"]=excluded
 report["generated_valid_count"]=len(generated)
 report["invalid_current_or_037_gate_count"]=len(eligible)-len(generated)
 report["stage_geometry_admitted"]=False
 return out,report

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--candidate-375",type=Path,required=True)
 p.add_argument("--expected-candidate-375-sha256",required=True)
 p.add_argument("--output-dir",type=Path,required=True)
 a=p.parse_args()
 if a.output_dir.exists():raise FileExistsError(a.output_dir)
 if sha(REPLAY368)!=PINS["replay368"] or sha(CAND368)!=PINS["candidate368"] or sha(AUDIT368)!=PINS["audit368"] or sha(SUBSET378)!=PINS["subset378"] or sha(CAND378)!=PINS["candidate378"] or sha(BASE362)!=PINS["base362"]:
  raise ReplayError("one or more source pins changed")
 input_sha=sha(a.candidate_375)
 if input_sha!=a.expected_candidate_375_sha256:raise ReplayError("375 candidate SHA mismatch")
 replay368=json.loads(REPLAY368.read_text()); subset378=json.loads(SUBSET378.read_text())
 report159=json.loads(AUDIT368.read_text())
 if replay368.get("status")!="REVALIDATED_359_FLIP_SUBSET_ON_362_COORDINATES" or replay368.get("input_362_candidate_sha256")!=PINS["base362"]:
  raise ReplayError("368 replay receipt is not pinned to expected 362 source")
 if subset378.get("status")!="SELF_WITNESS_FILTERED_358_SUBSET_PROPOSAL" or subset378.get("retained_flip_count")!=55:
  raise ReplayError("378 subset report is not the expected 55-edge subset")
 base=_load_npz(a.candidate_375);_assert_input_matches_362(base)
 face_count=len(base["source_O_faces"])
 stage1_ops=replay368.get("operations",[])
 if len(stage1_ops)!=replay368.get("accepted_pair_count"):raise ReplayError("368 accepted operation count mismatch")
 stage1_expected=_load_npz(CAND368)["source_O_faces"]
 flip=import_flip()
 seeds1=list(map(int,replay368["seed_face_rows"]))
 after1,r1=replay(base,stage1_ops,seeds1,flip,stage1_expected,"368")
 # Stage 2 applies 378 edges only where the exact expected face pair survived stage 1.
 expected2=_load_npz(CAND378)["source_O_faces"]
 sigs=subset378.get("retained_operation_signatures")
 if not isinstance(sigs,list) or len(sigs)!=55:raise ReplayError("378 operation signature inventory mismatch")
 stage2_ops=[{"edge":x["edge"],"face_rows":x["face_rows"],"parent_row":x["parent_row"]} for x in sigs]
 seeds2=_target_rows(report159,face_count)
 after2,r2=replay(after1,stage2_ops,seeds2,flip,expected2,"378")
 generated1=r1.get("operations",[]);generated2=r2.get("operations",[])
 # Preserve aggregate execution provenance after stage 2 overwrites the 358 arrays.
 all_ops=generated1+generated2
 after2["same_parent_flip_349_face_rows"]=np.asarray([r for op in all_ops for r in op["face_rows"]],dtype=np.int32)
 after2["same_parent_flip_349_pair_face_rows"]=np.asarray([op["face_rows"] for op in all_ops],dtype=np.int32).reshape((-1,2))
 after2["same_parent_flip_349_parent_rows"]=np.asarray([op["parent_row"] for op in all_ops],dtype=np.int64)
 out=a.output_dir;a.output_dir.mkdir(parents=True)
 candidate_out=out/"candidate.npz"
 with candidate_out.open("xb") as f:np.savez_compressed(f,**after2)
 summary={
  "status":"ORDERED_368_THEN_378_REPLAY_PROPOSAL",
  "geometry_admitted":False,"native_admission":False,"accepted_physical_steps":0,
  "input_375":str(a.candidate_375),"input_375_sha256":input_sha,
  "base_362_sha256":PINS["base362"],"replay368_report_sha256":PINS["replay368"],
  "audit368_159_sha256":PINS["audit368"],"subset378_report_sha256":PINS["subset378"],
  "candidate378_sha256":PINS["candidate378"],"flip358_sha256":PINS["flip358"],
  "stage_368":r1,"stage_378":r2,
  "stage_368_generated_count":len(generated1),"stage_378_generated_count":len(generated2),
  "stage_368_expected_count":len(stage1_ops),"stage_378_expected_count":len(stage2_ops),
  "final_candidate":str(candidate_out),"final_candidate_sha256":sha(candidate_out),
  "coordinates_bitwise_unchanged":True,"037_reference_points_bitwise_unchanged":True,
  "full_current_ref_area_self_and_185_bank_audits_required":True,
  "limitations":"Only pinned flips are considered. Current/reference-invalid operations and topology conflicts are excluded; no replacements are generated.",
 }
 (out/"replay-381-report.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":summary["status"],"stage1":len(generated1),"stage2":len(generated2),"candidate_sha256":summary["final_candidate_sha256"]},sort_keys=True))

if __name__=="__main__":main()
