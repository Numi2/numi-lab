#!/usr/bin/env python3
"""Verify a completed 370/378 proposal against independent complete 159 reports.

This script reads existing reports/candidates only; it does not run collision,
self-intersection, or geometry audits. It requires the completed full reports.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
import numpy as np

class VerificationError(ValueError):
    pass

def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def array_equal_bitsafe(a: np.ndarray, b: np.ndarray) -> bool:
    aa,bb=np.asarray(a),np.asarray(b)
    if aa.shape!=bb.shape or aa.dtype!=bb.dtype: return False
    if np.issubdtype(aa.dtype,np.floating):
        return bool(np.array_equal(aa,bb,equal_nan=True))
    return bool(np.array_equal(aa,bb))

def safe_operations(selector: dict[str,Any], self_pairs: np.ndarray) -> tuple[list[dict],list[dict],set[int]]:
    pairs=np.asarray(self_pairs)
    if pairs.ndim!=2 or pairs.shape[1]!=2 or not np.issubdtype(pairs.dtype,np.integer):
        raise VerificationError("self witnesses must be integer Nx2")
    rows=set(map(int,pairs.ravel()))
    selected=[op for op in selector.get("operations",[]) if op.get("selected") is True]
    safe,excluded=[],[]
    for op in selected:
        fr=op.get("face_rows")
        if (not isinstance(fr,list) or len(fr)!=2 or any(type(x)is not int for x in fr)):
            raise VerificationError("selected flip has malformed face rows")
        (excluded if rows.intersection(fr) else safe).append(op)
    return safe,excluded,rows

def validate_full_159(report: dict[str,Any], candidate_sha: str, face_count: int) -> dict[tuple[int,int],set[tuple[int,int]]]:
    if (report.get("status")!="COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT"
        or report.get("complete") is not True
        or report.get("bank_count")!=185 or report.get("expected_bank_count")!=185
        or report.get("candidate_sha256")!=candidate_sha
        or report.get("surface_face_count")!=face_count):
        raise VerificationError("159 report status, completeness, candidate SHA, or face-domain mismatch")
    banks=report.get("banks")
    if not isinstance(banks,list) or len(banks)!=185:
        raise VerificationError("159 report must enumerate all 185 typed banks")
    out={}
    for bank in banks:
        body,bone=bank.get("body"),bank.get("stable_bone_id")
        if type(body)is not int or type(bone)is not int:
            raise VerificationError("159 typed bank identity malformed")
        key=(body,bone)
        if key in out: raise VerificationError(f"duplicate typed bank {key}")
        if bank.get("skin_faces")!=face_count: raise VerificationError(f"skin face domain mismatch for {key}")
        result=bank.get("result")
        ids=set()
        if result is None:
            if (bank.get("skin_aabb_active")!=0 or bank.get("bank_aabb_active")!=0
                or type(bank.get("bank_faces")) is not int or bank["bank_faces"]<=0):
                raise VerificationError(f"null result lacks explicit complete AABB exclusion for {key}")
        else:
            if not isinstance(result,dict): raise VerificationError(f"malformed result for {key}")
            pairs=result.get("triangle_pairs")
            if not isinstance(pairs,list) or type(result.get("count")) is not int or len(pairs)!=result["count"]:
                raise VerificationError(f"exact pair count mismatch for {key}")
            for pair in pairs:
                if (not isinstance(pair,list) or len(pair)!=2 or any(type(v)is not int for v in pair)
                    or not 0<=pair[0]<face_count):
                    raise VerificationError(f"malformed exact pair for {key}")
                if tuple(pair) in ids: raise VerificationError(f"duplicate exact pair for {key}")
                ids.add(tuple(pair))
        out[key]=ids
    if len(out)!=185: raise VerificationError("typed bank inventory incomplete")
    return out

def edge_set(faces: np.ndarray) -> set[tuple[int,int]]:
    counts={}
    for tri in np.asarray(faces):
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            e=tuple(sorted((int(a),int(b))))
            counts[e]=counts.get(e,0)+1
    if any(v>2 for v in counts.values()):
        raise VerificationError("surface has edge incidence greater than two")
    return {e for e,n in counts.items() if n==1}

def parent_boundary_rings(faces: np.ndarray, parents: np.ndarray) -> dict[int,set[tuple[int,int]]]:
    f=np.asarray(faces); p=np.asarray(parents)
    if p.shape!=(len(f),): raise VerificationError("parent-row map shape mismatch")
    out={}
    for parent in sorted(set(map(int,p))):
        counts={}
        for tri in f[p==parent]:
            for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
                e=tuple(sorted((int(a),int(b))))
                counts[e]=counts.get(e,0)+1
        out[parent]={e for e,n in counts.items() if n==1}
    return out

def csr_row(data:dict[str,np.ndarray], ptrkey:str, valkey:str, row:int)->set[int]:
    ptr=np.asarray(data[ptrkey]); vals=np.asarray(data[valkey])
    if ptr.ndim!=1 or vals.ndim!=1 or len(ptr)==0 or ptr[0]!=0 or ptr[-1]!=len(vals) or np.any(ptr[1:]<ptr[:-1]):
        raise VerificationError(f"malformed ancestry CSR {ptrkey}")
    return set(map(int,vals[int(ptr[row]):int(ptr[row+1])]))

def verify(base_path:Path,candidate_path:Path,before_path:Path,after_path:Path,
           selector_path:Path,subset_path:Path,self_pairs_path:Path)->dict[str,Any]:
    base_sha, candidate_sha=sha_file(base_path),sha_file(candidate_path)
    with np.load(base_path,allow_pickle=False) as z: base={k:np.asarray(z[k]).copy() for k in z.files}
    with np.load(candidate_path,allow_pickle=False) as z: cand={k:np.asarray(z[k]).copy() for k in z.files}
    before=json.loads(before_path.read_text()); after=json.loads(after_path.read_text())
    selector=json.loads(selector_path.read_text()); subset=json.loads(subset_path.read_text())
    self_pairs=np.load(self_pairs_path,allow_pickle=False)
    if selector.get("candidate_sha256")!=base_sha or subset.get("base_candidate_sha256")!=base_sha:
        raise VerificationError("selector/subset is not tied to exact base bytes")
    if selector.get("output_candidate_sha256")!=subset.get("selector_370_candidate_sha256"):
        raise VerificationError("selector output candidate pin mismatch")
    if subset.get("output_candidate_sha256")!=candidate_sha:
        raise VerificationError("subset candidate SHA mismatch")
    faces0=np.asarray(base.get("source_O_faces")); faces1=np.asarray(cand.get("source_O_faces"))
    if faces0.ndim!=2 or faces0.shape[1]!=3 or faces1.shape!=faces0.shape:
        raise VerificationError("base/output face shapes differ")
    before_pairs=validate_full_159(before,base_sha,len(faces0))
    after_pairs=validate_full_159(after,candidate_sha,len(faces1))
    if set(before_pairs)!=set(after_pairs): raise VerificationError("before/after typed bank identities differ")
    safe,excluded,witness_rows=safe_operations(selector,self_pairs)
    safe_sigs={(tuple(sorted(map(int,o["edge"]))),tuple(map(int,o["face_rows"])),int(o["parent_row"])) for o in safe}
    recorded={(tuple(sorted(map(int,o["edge"]))),tuple(map(int,o["face_rows"])),int(o["parent_row"])) for o in subset.get("retained_operation_signatures",[])}
    if safe_sigs!=recorded or len(safe)!=subset.get("retained_flip_count"):
        raise VerificationError("subset retained operations differ from witness-filtered selector")
    selected_faces=[int(r) for op in safe for r in op["face_rows"]]
    if len(selected_faces)!=len(set(selected_faces)):
        raise VerificationError("retained flips overlap face rows")
    changed=set(map(int,np.flatnonzero(np.any(faces0!=faces1,axis=1))))
    predicted_changed=set(selected_faces)
    if changed!=predicted_changed: raise VerificationError("output changed face rows differ from retained flip rows")
    if changed & witness_rows: raise VerificationError("output retains a flip touching 130 self witness")
    # These are the only arrays a 358 topology flip is allowed to alter.
    mutable={
        "source_O_faces","area_reference_037_source_O_faces","refinement_face_parent_barycentric",
        "face_source_ancestry_indptr","face_source_ancestry_face_rows",
        "face_fine_subface_ancestry_indptr","face_fine_subface_ancestry_rows",
        "parent_coarse_face_row","parent_fine_detail_face_row","parent_source_face_marker","parent_fine_subface_row",
        "same_parent_flip_349_face_rows","same_parent_flip_349_pair_face_rows","same_parent_flip_349_parent_rows",
        "same_parent_flip_349_semantics",
    }
    if set(base)!=set(cand): raise VerificationError("candidate array key inventory changed")
    unexpected=[]
    for key in base:
        if key not in mutable and not array_equal_bitsafe(base[key],cand[key]): unexpected.append(key)
    if unexpected: raise VerificationError(f"unexpected point/source/native-anchor/reference changes: {unexpected}")
    if not array_equal_bitsafe(faces1,cand["area_reference_037_source_O_faces"]):
        raise VerificationError("current and matched-037 connectivity diverged")
    if not array_equal_bitsafe(base["area_reference_037_points_world_m_f64"],cand["area_reference_037_points_world_m_f64"]):
        raise VerificationError("matched-037 reference geometry changed")
    if not np.array_equal(base["refinement_face_parent_face_row"],cand["refinement_face_parent_face_row"]):
        raise VerificationError("immediate parent face IDs changed")
    be0,be1=edge_set(faces0),edge_set(faces1)
    if be0!=be1: raise VerificationError("exact surface boundary edge set changed")
    if len(be0)!=45: raise VerificationError(f"expected preserved 45 open boundary edges, found {len(be0)}")
    if parent_boundary_rings(faces0,base["refinement_face_parent_face_row"]) != parent_boundary_rings(faces1,cand["refinement_face_parent_face_row"]):
        raise VerificationError("immediate-parent boundary rings changed")
    # Exact source-face and fine-subface ancestry on each flipped pair must be
    # the union of the two original face rows, as defined by 358.
    ancestry_checks=0
    for op in safe:
        rows=list(map(int,op["face_rows"]))
        for ptr,val in (("face_source_ancestry_indptr","face_source_ancestry_face_rows"),
                        ("face_fine_subface_ancestry_indptr","face_fine_subface_ancestry_rows")):
            union=csr_row(base,ptr,val,rows[0])|csr_row(base,ptr,val,rows[1])
            if any(csr_row(cand,ptr,val,r)!=union for r in rows):
                raise VerificationError(f"{ptr} is not preserved as exact pair ancestry union on {rows}")
        ancestry_checks+=1
    # Match each selected operation's per-bank prediction to exact identities in
    # the complete before/after reports, then reconcile every full bank count.
    bank_old_new={key:[0,0] for key in before_pairs}
    op_checks=0
    for op in safe:
        rows=set(map(int,op["face_rows"]))
        predictions={(int(x["body"]),int(x["stable_bone_id"])):x for x in op["old_new_by_typed_bank"]}
        for key in before_pairs:
            old_ids=sorted(tid for face,tid in before_pairs[key] if face in rows)
            new_ids=sorted(tid for face,tid in after_pairs[key] if face in rows)
            pred=predictions.get(key,{"old_pair_count":0,"new_pair_count":0,"old_triangle_ids":[],"new_triangle_ids":[]})
            if (len(old_ids)!=pred["old_pair_count"] or old_ids!=sorted(pred["old_triangle_ids"])
                or len(new_ids)!=pred["new_pair_count"] or new_ids!=sorted(pred["new_triangle_ids"])):
                raise VerificationError(f"selector pair-identity prediction mismatch for operation {op['edge']} bank {key}")
            bank_old_new[key][0]+=pred["old_pair_count"]
            bank_old_new[key][1]+=pred["new_pair_count"]
        op_checks+=1
    per_bank=[]
    total_before=total_after=0
    removed_total=introduced_total=0
    changed_pair_identity_rows=set()
    for key in sorted(before_pairs):
        old_set,new_set=before_pairs[key],after_pairs[key]
        removed=old_set-new_set; introduced=new_set-old_set
        if any(face not in changed for face,_ in removed|introduced):
            raise VerificationError(f"159 pair identity changed outside flipped face rows for bank {key}")
        old_count,new_count=len(old_set),len(new_set)
        pred_old,pred_new=bank_old_new[key]
        predicted_final=old_count-pred_old+pred_new
        if predicted_final!=new_count:
            raise VerificationError(f"final 159 count mismatch for bank {key}: predicted {predicted_final}, actual {new_count}")
        if new_count>old_count:
            raise VerificationError(f"typed bank pair count increased for {key}")
        total_before+=old_count; total_after+=new_count
        removed_total+=len(removed); introduced_total+=len(introduced)
        per_bank.append({"body":key[0],"stable_bone_id":key[1],"before_count":old_count,
                         "predicted_removed_count":pred_old,"predicted_introduced_count":pred_new,
                         "after_count":new_count,"exact_removed_pair_identities":[list(x) for x in sorted(removed)],
                         "exact_introduced_pair_identities":[list(x) for x in sorted(introduced)]})
    return {
        "status":"FULL_159_AUDIT_MATCHES_378_SELECTOR_PREDICTION",
        "geometry_admitted":False,"native_admission":False,"accepted_physical_steps":0,
        "base_candidate_sha256":base_sha,"output_candidate_sha256":candidate_sha,
        "before_159_sha256":sha_file(before_path),"after_159_sha256":sha_file(after_path),
        "selector_report_sha256":sha_file(selector_path),"subset_report_sha256":sha_file(subset_path),
        "complete_typed_banks":185,"matched_selector_operations":op_checks,
        "matched_ancestry_union_operations":ancestry_checks,"unchanged_nonface_arrays":len(base)-len(mutable),
        "changed_face_rows":sorted(changed),"open_boundary_edges_before_after":len(be0),
        "total_pairs_before":total_before,"total_pairs_after":total_after,
        "exact_pair_identities_removed":removed_total,"exact_pair_identities_introduced":introduced_total,
        "all_bank_count_predictions_match":True,"all_pair_identity_predictions_match":True,
        "per_typed_bank":per_bank,
        "limitations":"This verifies reports against the selector prediction; it does not promote geometry or native admission.",
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-candidate",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--before-159",type=Path,required=True)
    p.add_argument("--after-159",type=Path,required=True)
    p.add_argument("--selector-370",type=Path,required=True)
    p.add_argument("--subset-378",type=Path,required=True)
    p.add_argument("--self-pairs",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=verify(a.base_candidate,a.candidate,a.before_159,a.after_159,a.selector_370,a.subset_378,a.self_pairs)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":result["status"],"pairs_before":result["total_pairs_before"],
                      "pairs_after":result["total_pairs_after"],"removed":result["exact_pair_identities_removed"],
                      "introduced":result["exact_pair_identities_introduced"]},sort_keys=True))

if __name__=="__main__": main()
