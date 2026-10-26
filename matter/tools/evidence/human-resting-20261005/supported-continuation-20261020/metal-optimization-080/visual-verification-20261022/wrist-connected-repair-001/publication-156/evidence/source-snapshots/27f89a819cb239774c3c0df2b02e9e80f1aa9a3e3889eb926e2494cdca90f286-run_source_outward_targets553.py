#!/usr/bin/env python3
"""Build registered-source-normal, all-185 targets for new refined inside nodes.

This invokes the frozen 545 first-union-exit builder in source-outward mode and
installs the pinned 546 multi-parent source-normal extension in memory. It
writes target proposals only; it does not move candidate points or certify the
whole connected mesh.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
PY=Path("/Users/n/numi-human-prep-venv-20261005/bin/python")
BUILDER=ROOT/"D/contained-clearance-545/build_direct_depth_targets545.py"
BUILDER_SHA="2864d9547d2715c61c9c27b17bcfd11fac250197c47f4fd4870f8f6898cecb60"
INSTALLER=ROOT/"D/surface-clearance-546/direct-depth-targets-546/run_direct_depth_targets546.py"
INSTALLER_SHA="0708d259c7bc7dd6341b8cac737f6516ac611011e6d4f599e05530cf91106ac5"
EXTENSION=ROOT/"D/source-outward-proposal-500/build_outward_targets_multi_source_546.py"
EXTENSION_SHA="dd61f0fb87f12c132166515a1e058e144d493a835cadbb2a47e6a2c8574e1e68"
OWNER500_SHA="454501303db7c6420ff3744729c62c723c208aaf9348a53534137c62b7e6afbf"
ANCHOR_KEYS=("coarse_boundary_anchor_rows","coarse_boundary_anchor_source_nodes",
             "typed_coarse_anchor_output_rows_018","typed_coarse_anchor_source_ids_018")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def load_pinned(path,expected,name):
    actual=sha(path)
    if actual!=expected:raise ValueError(f"pinned source SHA mismatch for {path}: {actual}")
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise ImportError(f"cannot load pinned source: {path}")
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def validate_direction_mode(mode):
    if mode!="source-outward":
        raise ValueError("553 only supports the registered source-outward direction")

def _load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def validate_new_only_prefix(parent_arrays,child_arrays,parent_summary,child_summary,
                              parent_candidate_sha,child_candidate_sha):
    """Prove target-eligible inside witnesses are appended nodes only."""
    if parent_summary.get("candidate_sha256")!=parent_candidate_sha:
        raise ValueError("parent 130 summary is not bound to parent candidate")
    if child_summary.get("candidate_sha256")!=child_candidate_sha:
        raise ValueError("child 130 summary is not bound to refined candidate")
    if parent_summary.get("complete_surface_audit") is not True or parent_summary.get("exact_PLC_surface_pairs")!=0:
        raise ValueError("parent needs a complete self-clear 130 point report")
    if int(parent_summary.get("inside_node_bank_pairs",-1))!=0 or int(parent_summary.get("inside_nodes",-1))!=0:
        raise ValueError("parent contains inside rows; targets would not be new-node-only")
    if child_summary.get("complete_surface_audit") is not True or child_summary.get("exact_PLC_surface_pairs")!=0:
        raise ValueError("refined candidate needs a complete self-clear 130 point report")
    old=np.asarray(parent_arrays.get("points_world_m"));new=np.asarray(child_arrays.get("points_world_m"))
    if old.dtype!=np.float32 or new.dtype!=np.float32 or old.ndim!=2 or old.shape[1:]!=(3,) or new.ndim!=2 or new.shape[1:]!=(3,):
        raise ValueError("parent/refined point arrays must be Float32 Nx3")
    if int(parent_summary.get("points",-1))!=len(old):
        raise ValueError("parent 130 point count differs from parent candidate")
    if len(new)<=len(old) or not np.array_equal(new[:len(old)].view(np.uint8),old.view(np.uint8)):
        raise ValueError("refinement did not preserve a strict bitwise parent point prefix")
    if int(child_summary.get("points",-1))!=len(new):
        raise ValueError("child 130 summary point count differs from refined candidate")
    inside=child_summary.get("inside_witnesses")
    if not isinstance(inside,list):raise ValueError("child point report lacks inside witness rows")
    rows=sorted({int(r["output_node"]) for r in inside})
    if any(n<len(old) or n>=len(new) for n in rows):
        raise ValueError("an inside witness is not an appended refinement node")
    if int(child_summary.get("inside_nodes",-1))!=len(rows):
        raise ValueError("child inside-node count does not match unique witness rows")
    kinds=np.asarray(child_arrays.get("refinement_node_kind"))
    if kinds.shape!=(len(new),) or kinds.dtype.kind not in "iu":
        raise ValueError("refined candidate lacks aligned typed refinement node kinds")
    if rows and any(int(kinds[n]) not in (1,2) for n in rows):
        raise ValueError("inside target rows are not typed edge-midpoint/face-centroid nodes")
    return rows,len(old),len(new)

def validate_target_artifacts(report,arrays_path,expected_rows):
    if report.get("direction_mode")!="source-outward":
        raise ValueError("545 target report did not use source-outward direction mode")
    if report.get("status")!="PASS_FIRST_UNION_EXIT_TARGETS_ONLY":
        raise ValueError("545 did not certify every source-outward endpoint")
    targets=report.get("targets")
    if not isinstance(targets,list):raise ValueError("target report lacks rows")
    report_rows=sorted(int(r["query_row"]) for r in targets)
    if report_rows!=list(expected_rows):raise ValueError("target report rows differ from new inside-node rows")
    for row in targets:
        if row.get("status")!="PASS_FIRST_UNION_EXIT" or row.get("target_exactly_outside_all185") is not True or row.get("exact_endpoint_bank_count")!=185:
            raise ValueError(f"target row {row.get(query_row)} lacks exact all-185 endpoint certification")
        if row.get("direction_method") not in (None,"Registered source-outward direction."):
            # The source-normal provenance can be nested; do not allow a depth
            # covector to be silently described as the actual ray direction.
            diag=row.get("direction_compatibility",{})
            if diag.get("method")!="registered_source_outward_normal":
                raise ValueError("target direction metadata is not registered source-outward")
    z=_load_npz(arrays_path)
    if not np.array_equal(np.asarray(z.get("query_rows"),dtype=np.int64),np.asarray(expected_rows,dtype=np.int64)):
        raise ValueError("target NPZ row identities differ from new inside nodes")
    if not np.asarray(z.get("pass_mask"),dtype=bool).all() or not np.asarray(z.get("target_exactly_outside_all185"),dtype=bool).all():
        raise ValueError("target NPZ does not certify all rows outside all 185 banks")
    if not np.asarray(z.get("source_normal_available_mask"),dtype=bool).all():
        raise ValueError("target NPZ omits registered source-normal provenance")
    directions=np.asarray(z.get("proposal_direction_world_unit"),dtype=np.float64)
    normals=np.asarray(z.get("source_surface_normal_world_unit"),dtype=np.float64)
    if directions.shape!=(len(expected_rows),3) or normals.shape!=directions.shape or not np.isfinite(directions).all() or not np.isfinite(normals).all():
        raise ValueError("source-outward direction arrays have invalid shape or values")
    if not np.allclose(directions,normals,rtol=0.,atol=1e-12):
        raise ValueError("proposal rays are not the registered source-surface normals")
    return z

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--direction-mode",required=True,choices=("source-outward",))
    ap.add_argument("--parent-candidate",required=True,type=Path)
    ap.add_argument("--parent-point-summary",required=True,type=Path)
    ap.add_argument("--candidate",required=True,type=Path)
    ap.add_argument("--point-summary",required=True,type=Path)
    ap.add_argument("--query-dir",required=True,type=Path)
    ap.add_argument("--output-dir",required=True,type=Path)
    ap.add_argument("--clearance-m",type=float,default=5e-5)
    a=ap.parse_args(argv);validate_direction_mode(a.direction_mode)
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    paths={"parent_candidate":a.parent_candidate,"parent_point_summary":a.parent_point_summary,
           "candidate":a.candidate,"point_summary":a.point_summary,
           "query_receipt":a.query_dir/"query-receipt.json",
           "query_points":a.query_dir/"points.csv","query_nodes":a.query_dir/"nodes.csv",
           "query_pairs":a.query_dir/"pairs.csv","builder":BUILDER,
           "multi_parent_installer":INSTALLER,"multi_parent_extension":EXTENSION}
    pins={k:sha(v) for k,v in paths.items()}
    parent_arrays=_load_npz(a.parent_candidate);candidate_arrays=_load_npz(a.candidate)
    parent_summary=json.loads(a.parent_point_summary.read_text());summary=json.loads(a.point_summary.read_text())
    parent_sha=pins["parent_candidate"];candidate_sha=pins["candidate"]
    new_rows,old_count,new_count=validate_new_only_prefix(parent_arrays,candidate_arrays,parent_summary,summary,parent_sha,candidate_sha)
    # Typed coarse-anchor IDs must remain byte-identical across subdivision.
    for key in ANCHOR_KEYS:
        if key in parent_arrays and (key not in candidate_arrays or parent_arrays[key].dtype!=candidate_arrays[key].dtype or parent_arrays[key].shape!=candidate_arrays[key].shape or parent_arrays[key].tobytes()!=candidate_arrays[key].tobytes()):
            raise ValueError(f"refined candidate changed typed anchor metadata: {key}")
    installer=load_pinned(INSTALLER,INSTALLER_SHA,"direct_depth_installer_for_source_outward553")
    extension=load_pinned(EXTENSION,EXTENSION_SHA,"multi_parent_source_normal_546_for553")
    owner500=Path(extension.OWNER500)
    if sha(owner500)!=OWNER500_SHA or extension.OWNER500_SHA!=OWNER500_SHA:
        raise ValueError("pinned source-normal owner 500 hash mismatch")
    paths["source_normal_owner500"]=owner500;pins["source_normal_owner500"]=sha(owner500)
    builder=load_pinned(BUILDER,BUILDER_SHA,"source_outward_point_builder545_for553")
    face_provenance={}
    original_loadmod=builder.loadmod
    def load_with_multi_parent(path,name):
        module=original_loadmod(path,name)
        if Path(path).resolve()==Path(builder.SOURCE303).resolve():
            installer.install_source303_extension(module,extension,face_provenance)
        return module
    builder.loadmod=load_with_multi_parent
    builder_args=["--candidate",str(a.candidate),"--point-summary",str(a.point_summary),
        "--query-dir",str(a.query_dir),"--output-dir",str(a.output_dir),
        "--clearance-m",format(a.clearance_m,".17g"),"--direction-mode","source-outward"]
    builder.main(builder_args)
    target_report_path=a.output_dir/"nearest-union-exits.json";target_npz=a.output_dir/"nearest-union-exits.npz"
    if not target_report_path.is_file() or not target_npz.is_file():raise RuntimeError("545 omitted source-outward target outputs")
    target_report=json.loads(target_report_path.read_text())
    target_arrays=validate_target_artifacts(target_report,target_npz,new_rows)
    # Verify every source pin again before receipt emission.
    if {k:sha(v) for k,v in paths.items()}!=pins:raise RuntimeError("a pinned input/owner changed during target construction")
    out=a.output_dir/"source-outward-launcher-553.json"
    receipt={"schema":"numi.human.refined_source_outward_targets_553.v1",
      "status":"PASS_NEW_REFINEMENT_INSIDE_ROWS_SOURCE_OUTWARD_ENDPOINTS_ONLY",
      "candidate_sha256":candidate_sha,"parent_candidate_sha256":parent_sha,
      "point_summary_sha256":pins["point_summary"],"parent_point_summary_sha256":pins["parent_point_summary"],
      "query_receipt_sha256":pins["query_receipt"],"target_report_sha256":sha(target_report_path),"target_npz_sha256":sha(target_npz),
      "old_point_prefix_count":old_count,"refined_point_count":new_count,"new_inside_target_rows":new_rows,
      "target_rows_are_all_appended_refinement_nodes":True,"parent_had_zero_inside_node_bank_pairs":True,
      "old_point_prefix_bitwise_unchanged":True,"typed_anchor_arrays_unchanged":True,
      "direction_mode":"source-outward","proposal_vectors_equal_registered_source_surface_normals":True,
      "target_endpoint_rows_exactly_outside_all185":True,"multi_parent_source_normal_extension":{
        "installer_path":str(INSTALLER),"installer_sha256":pins["multi_parent_installer"],
        "extension_path":str(EXTENSION),"extension_sha256":pins["multi_parent_extension"],
        "owner500_path":str(owner500),"owner500_sha256":sha(owner500),
        "multi_parent_source_face_rows":sorted(face_provenance),"multi_parent_source_face_count":len(face_provenance),
        "all_candidate_source_provenance_retained":True},
      "source_pins":pins,"target_builder_sha256":BUILDER_SHA,
      "targets_are_proposals_only":True,"candidate_coordinates_changed":False,
      "whole_mesh_clearance_claim":False,"native_admission":False,"accepted_physical_steps":0}
    out.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n")
    if {k:sha(v) for k,v in paths.items()}!=pins:raise RuntimeError("a pinned input/owner changed before exit")
    print(json.dumps({"status":receipt["status"],"target_count":len(new_rows),"candidate_sha256":candidate_sha,
      "target_report":str(target_report_path),"target_npz":str(target_npz),"receipt":str(out),"receipt_sha256":sha(out),
      "source_outward_directions":True,"all185_endpoint_targets":True,"geometry_changed":False},indent=2))

if __name__=="__main__":main()
