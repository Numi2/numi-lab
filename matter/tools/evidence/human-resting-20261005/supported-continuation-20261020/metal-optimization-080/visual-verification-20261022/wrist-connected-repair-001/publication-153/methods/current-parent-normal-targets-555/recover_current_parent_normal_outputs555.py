#!/usr/bin/env python3
"""Postprocess completed 555 endpoint outputs without rerunning geometry."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
sys.path.insert(0,str(HERE))
import current_parent_normal_helper555 as helper
import run_current_parent_normal_targets555 as launcher


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def read_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def reconstruct_registered_source_normal(record, *, atol=2e-12):
    outer=record.get("source_normal_provenance")
    if not isinstance(outer,dict) or "registered_source_reference" not in outer:
        raise ValueError("original report lacks nested registered 303 normal provenance")
    provenance=outer["registered_source_reference"]
    faces=provenance.get("source_face_area_weights")
    if not isinstance(faces,list) or not faces:
        raise ValueError("registered 303 normal provenance has no contributing face rows")
    if len(faces)!=int(provenance.get("unique_incident_face_count",-1)):
        raise ValueError("registered normal face count differs from recorded source ancestry")
    total=0.0;acc=np.zeros(3,dtype=np.float64)
    for face in faces:
        area=float(face["source037_area_m2"])
        normal=np.asarray(face["registered_source_normal_world_unit"],dtype=np.float64)
        if not np.isfinite(area) or area<=0 or normal.shape!=(3,) or not np.isfinite(normal).all():
            raise ValueError("invalid registered 303 source normal face record")
        if not np.isclose(np.linalg.norm(normal),1.,rtol=0,atol=atol):
            raise ValueError("registered source face normal is not unit length")
        acc+=area*normal;total+=area
    recorded=float(provenance["source037_area_sum_m2"])
    if not np.isclose(total,recorded,rtol=1e-13,atol=1e-18):
        raise ValueError("recorded source037 area sum does not match source face records")
    resultant=float(np.linalg.norm(acc))
    if not np.isfinite(resultant) or resultant<=1e-14:
        raise ValueError("registered source face normals cancel")
    ratio=resultant/total
    if not np.isclose(ratio,float(provenance["resultant_over_area_sum"]),rtol=1e-12,atol=1e-14):
        raise ValueError("registered normal resultant ratio differs from provenance")
    return acc/resultant,provenance


def verify_completed_outputs(parent_candidate, parent_summary, candidate, point_summary,
                            query_dir, report_path, arrays_path):
    parent_sha=sha(parent_candidate);candidate_sha=sha(candidate)
    report=json.loads(Path(report_path).read_text());arrays=read_npz(arrays_path)
    if report.get("inputs",{}).get("candidate_sha256")!=candidate_sha:
        raise ValueError("completed 555 report is not bound to supplied candidate")
    if report.get("inputs",{}).get("point_summary_sha256")!=sha(point_summary):
        raise ValueError("completed 555 report is not bound to supplied point summary")
    if report.get("inputs",{}).get("query_receipt_sha256")!=sha(Path(query_dir)/"query-receipt.json"):
        raise ValueError("completed 555 report is not bound to supplied 130 query receipt")
    if report.get("inputs",{}).get("query_points_sha256")!=sha(Path(query_dir)/"points.csv"):
        raise ValueError("completed 555 report is not bound to supplied query points")
    if report.get("inputs",{}).get("query_nodes_sha256")!=sha(Path(query_dir)/"nodes.csv") or report.get("inputs",{}).get("query_pairs_sha256")!=sha(Path(query_dir)/"pairs.csv"):
        raise ValueError("completed 555 report is not bound to supplied 130 query rows")
    if report.get("outputs",{}).get("npz_sha256")!=sha(arrays_path):
        raise ValueError("completed NPZ differs from original report output pin")
    if report.get("outputs",{}).get("npz")!=str(Path(arrays_path)):
        raise ValueError("completed report output path differs from supplied NPZ")
    if report.get("inputs",{}).get("source303_sha256")!=launcher.SOURCE303_SHA:
        raise ValueError("completed report used a different source303 owner")
    if report.get("inputs",{}).get("source_orientation_report_sha256")!=launcher.ORIENTATION_SHA:
        raise ValueError("completed report used a different orientation report")
    if report.get("inputs",{}).get("current_parent_candidate_sha256") not in (None,parent_sha):
        raise ValueError("completed report names a different immediate parent candidate")
    orient=launcher.resolve_coarse_outward_sign(
        launcher.SOURCE303,launcher.ORIENT)
    if orient!=-1:raise ValueError("pinned current O-face outward sign is not -1")

    wrapper=launcher.load_pinned(launcher.WRAPPER553,launcher.WRAPPER553_SHA,"source_refinement_validator_for_recovery555")
    parent=launcher._npz_dict(parent_candidate);child=launcher._npz_dict(candidate)
    parent_sum=json.loads(Path(parent_summary).read_text());child_sum=json.loads(Path(point_summary).read_text())
    rows,old_count,new_count=wrapper.validate_new_only_prefix(parent,child,parent_sum,child_sum,parent_sha,candidate_sha)
    report_rows=[int(r["query_row"]) for r in report.get("targets",[])]
    if report_rows!=rows:raise ValueError("completed report targets do not match appended inside nodes")
    if not np.array_equal(np.asarray(arrays.get("query_rows"),dtype=np.int64),np.asarray(rows,dtype=np.int64)):
        raise ValueError("completed NPZ target rows differ from 130 witnesses")
    if not np.asarray(arrays.get("target_exactly_outside_all185"),dtype=bool).all() or not np.asarray(arrays.get("pass_mask"),dtype=bool).all():
        raise ValueError("completed endpoints lack all-185 pass masks")
    if report.get("passing_targets")!=len(rows) or report.get("all_bank_ray_results")!=len(rows):
        raise ValueError("completed endpoint report has incomplete target counts")

    current,prov=helper.parent_normal_field(rows,parent_points=parent["points_world_m"],
        parent_faces=parent["source_O_faces"],child_points=child["points_world_m"],
        node_kind=child["refinement_node_kind"],
        parent_face_indptr=child["refinement_node_parent_face_indptr"],
        parent_face_rows=child["refinement_node_parent_face_rows"],
        parent_face_barycentric=child["refinement_node_parent_face_barycentric"],
        old_prefix_count=old_count,outward_sign=orient)
    proposal=np.asarray(arrays["proposal_direction_world_unit"],dtype=np.float64)
    expected=np.asarray([current[row] for row in rows],dtype=np.float64)
    if proposal.shape!=expected.shape or not np.array_equal(proposal,expected):
        raise ValueError("completed proposal vectors are not the signed immediate-parent normals")
    if np.isfinite(np.asarray(arrays.get("depth_covector_world_unit"),dtype=np.float64)).any():
        raise ValueError("unexpected finite depth covectors in current-parent target output")

    source_normals={};source_prov={}
    child_points=np.asarray(child["points_world_m"],dtype=np.float32)
    target_world=np.asarray(arrays["target_world_f32_m"],dtype=np.float32)
    for i,(row,record) in enumerate(zip(rows,report["targets"])):
        if record.get("status")!="PASS_FIRST_UNION_EXIT" or record.get("target_exactly_outside_all185") is not True or int(record.get("exact_endpoint_bank_count",-1))!=185:
            raise ValueError(f"completed target row {row} did not pass its 185-bank endpoint check")
        start=np.asarray(record["start_world_f32_m"],dtype=np.float32)
        if not np.array_equal(start.view(np.uint8),child_points[row].view(np.uint8)):
            raise ValueError(f"target row {row} starts from a different candidate point")
        target_record=np.asarray(record["target_world_f32_m"],dtype=np.float32)
        if not np.array_equal(target_record.view(np.uint8),target_world[i].view(np.uint8)):
            raise ValueError(f"target row {row} JSON/NPZ endpoints differ")
        src,source_record=reconstruct_registered_source_normal(record)
        source_normals[row]=src
        source_prov[row]=source_record
        if not np.allclose(np.asarray(record["proposal_direction_world_unit"],dtype=np.float64),current[row],rtol=0,atol=1e-12):
            raise ValueError(f"target row {row} JSON direction does not match current parent normal")
        if not np.isclose(float(record["target_distance_m"]),
                          float(np.linalg.norm(target_world[i].astype(np.float64)-start.astype(np.float64))),
                          rtol=1e-6,atol=2e-8):
            raise ValueError(f"target row {row} recorded step distance differs from endpoint")

    return report,arrays,rows,source_normals,source_prov,current,prov,orient


def recover(parent_candidate, parent_summary, candidate, point_summary, query_dir,
            source_report, source_npz, output_dir):
    source_report=Path(source_report);source_npz=Path(source_npz);out=Path(output_dir)
    if out.exists():raise FileExistsError(out)
    original_report_sha=sha(source_report);original_npz_sha=sha(source_npz)
    report,arrays,rows,source_normals,source_prov,current,current_prov,orient=verify_completed_outputs(
        parent_candidate,parent_summary,candidate,point_summary,query_dir,source_report,source_npz)
    out.mkdir(parents=True)
    preserved=out/"preserved-unannotated-original";preserved.mkdir()
    shutil.copy2(source_report,preserved/"nearest-union-exits.original.json")
    shutil.copy2(source_npz,preserved/"nearest-union-exits.original.npz")
    preserved_record={
        "classification":"preserved original outputs from completed target stage before metadata repair",
        "report_sha256":sha(preserved/"nearest-union-exits.original.json"),
        "npz_sha256":sha(preserved/"nearest-union-exits.original.npz"),
        "reported_original_report_sha256":original_report_sha,
        "reported_original_npz_sha256":original_npz_sha,
    }
    out_report=out/"nearest-union-exits.json";out_npz=out/"nearest-union-exits.npz"
    ordered_source=np.asarray([source_normals[row] for row in rows],dtype=np.float64)
    ordered_current=np.asarray([current[row] for row in rows],dtype=np.float64)
    for record in report["targets"]:
        row=int(record["query_row"])
        record["proposal_direction_world_unit"]=current[row].tolist()
        record["current_parent_normal_world_unit"]=current[row].tolist()
        record["registered_source_normal_reference_world_unit"]=source_normals[row].tolist()
        record["source_surface_normal_world_unit"]=source_normals[row].tolist()
        record["source_normal_provenance"]=source_prov[row]
        record["current_parent_normal_provenance"]=current_prov[row]
        record["direction_method"]="current-parent-normal"
    report["schema"]="numi.human.refined-point-current-parent-normal-targets-555.v1"
    report["direction_mode"]="current-parent-normal"
    report["direction_method"]=(
        "Proposal rays were verified bitwise equal to immediate-parent face area-weighted normals "
        "oriented by the validated 303 coarse outward sign -1. Registered 303 source normals are "
        "reconstructed from the recorded 303 face-normal and 037-area provenance and are retained "
        "as reference only. Endpoint arrays and coordinates are unchanged from the completed 545 run."
    )
    report["normal_method"]=(
        "Registered source-normal references are reconstructed from the completed call's nested 303 "
        "typed-ancestry face-normal records and matched source037 area weights. Proposal directions "
        "are immediate-parent-face normals from the 555 helper."
    )
    report["current_parent_normal_orientation_sign_from_303"]=orient
    report["registered_source_normals_are_reference_only"]=True
    report["depth_covector_used"]=False
    report["native_admission"]=False;report["physical_steps"]=0
    report["output_annotation_status"]="PASS_POSTPROCESS_ONLY_RECOVERY"
    report["postprocess_recovery"]={
        "geometry_or_endpoint_recomputed":False,
        "proposal_directions_verified_exactly_against_immediate_parent_normal_recomputation":True,
        "original_endpoint_rows_and_all185_statuses_preserved":True,
        "preserved_original_outputs":preserved_record,
    }
    report["inputs"]["current_parent_candidate_sha256"]=sha(parent_candidate)
    report["inputs"]["current_parent_faces_sha256"]=hashlib.sha256(
        np.ascontiguousarray(read_npz(parent_candidate)["source_O_faces"]).tobytes()).hexdigest()
    report["inputs"]["immediate_parent_normal_helper_sha256"]=sha(launcher.HELPER)
    report["inputs"]["current_parent_normal_postprocessor_sha256"]=sha(launcher.RUNNER)
    arrays["proposal_direction_world_unit"]=ordered_current
    arrays["current_parent_normal_world_unit"]=ordered_current
    arrays["current_parent_normal_available_mask"]=np.ones(len(rows),dtype=np.bool_)
    arrays["source_surface_normal_world_unit"]=ordered_source
    arrays["registered_source_normal_reference_world_unit"]=ordered_source
    arrays["source_normal_available_mask"]=np.ones(len(rows),dtype=np.bool_)
    arrays["depth_covector_used_mask"]=np.zeros(len(rows),dtype=np.bool_)
    with out_npz.open("wb") as f:np.savez_compressed(f,**arrays)
    outputs=report.setdefault("outputs",{})
    outputs["npz"]=str(out_npz);outputs["npz_sha256"]=sha(out_npz)
    outputs["current_parent_normal_postprocessor_sha256"]=sha(Path(__file__))
    outputs["current_parent_normal_helper_sha256"]=sha(launcher.HELPER)
    out_report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    receipt={
        "schema":"numi.human.current_parent_normal_target_postprocess_recovery_555.v1",
        "status":"PASS_POSTPROCESS_ONLY_RECOVERY",
        "classification":"reannotation of completed exact endpoint outputs; no new endpoint query or geometry operation",
        "source_report_sha256":original_report_sha,"source_npz_sha256":original_npz_sha,
        "recovered_report_sha256":sha(out_report),"recovered_npz_sha256":sha(out_npz),
        "candidate_sha256":sha(candidate),"parent_candidate_sha256":sha(parent_candidate),
        "target_count":len(rows),"rows_are_appended_inside_refinement_nodes":True,
        "verified_proposal_equals_current_parent_normal_bitwise":True,
        "coarse_outward_sign_from_pinned_303":orient,
        "registered_source_normal_provenance_reconstructed_for_all_rows":True,
        "all185_endpoint_rows_and_targets_unchanged":True,
        "depth_covectors_used":False,"native_admission":False,"accepted_physical_steps":0,
        "preserved_original_outputs":preserved_record,
        "recovery_script_sha256":sha(Path(__file__)),
        "source_owner_hashes":{
            "source303":launcher.SOURCE303_SHA,"orientation_report":launcher.ORIENTATION_SHA,
            "target_builder545":launcher.BUILDER_SHA,"helper555":sha(launcher.HELPER),
        },
    }
    receipt_path=out/"postprocess-recovery-555.json"
    receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-candidate",required=True,type=Path)
    p.add_argument("--parent-point-summary",required=True,type=Path)
    p.add_argument("--candidate",required=True,type=Path)
    p.add_argument("--point-summary",required=True,type=Path)
    p.add_argument("--query-dir",required=True,type=Path)
    p.add_argument("--source-report",required=True,type=Path)
    p.add_argument("--source-npz",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    a=p.parse_args(argv)
    receipt=recover(a.parent_candidate,a.parent_point_summary,a.candidate,a.point_summary,
                    a.query_dir,a.source_report,a.source_npz,a.output_dir)
    print(json.dumps(receipt,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

