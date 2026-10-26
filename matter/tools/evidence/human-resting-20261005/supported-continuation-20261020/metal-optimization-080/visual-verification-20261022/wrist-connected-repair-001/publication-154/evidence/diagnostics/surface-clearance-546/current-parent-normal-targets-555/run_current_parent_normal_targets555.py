#!/usr/bin/env python3
"""Proposal-only all-185 union-exit targets along immediate-parent normals.

This adapter reuses frozen 545 ray/endpoint logic. For newly appended refinement
nodes only, proposal directions come from immediate parent O-face winding,
oriented by the pinned 303 coarse outward sign. Registered 303 normals remain a
separate reference/provenance field. This does not move coordinates or admit a
mesh.
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
DIR=ROOT/"D/surface-clearance-546/current-parent-normal-targets-555"
SOURCE303=Path("/Users/n/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/source-outward-proposal-300/build_outward_nearest_union_targets_303.py")
ORIENT=Path("/Users/n/Work/numi-human-connected-surface-069/digit-source-diagnosis-110/source-path-orientation-audit-112.json")
WRAPPER553=ROOT/"D/surface-clearance-546/connected-source-refinement-553/run_source_outward_targets553.py"
WRAPPER553_SHA="27f89a819cb239774c3c0df2b02e9e80f1aa9a3e3889eb926e2494cdca90f286"
BUILDER=ROOT/"D/contained-clearance-545/build_direct_depth_targets545.py"
BUILDER_SHA="2864d9547d2715c61c9c27b17bcfd11fac250197c47f4fd4870f8f6898cecb60"
INSTALLER=ROOT/"D/surface-clearance-546/direct-depth-targets-546/run_direct_depth_targets546.py"
INSTALLER_SHA="0708d259c7bc7dd6341b8cac737f6516ac611011e6d4f599e05530cf91106ac5"
EXTENSION=ROOT/"D/source-outward-proposal-500/build_outward_targets_multi_source_546.py"
EXTENSION_SHA="dd61f0fb87f12c132166515a1e058e144d493a835cadbb2a47e6a2c8574e1e68"
OWNER500=ROOT/"D/source-outward-proposal-500/build_outward_nearest_union_targets_500.py"
OWNER500_SHA="454501303db7c6420ff3744729c62c723c208aaf9348a53534137c62b7e6afbf"
SOURCE303_SHA="c8fe5b7be2efb2c6867a2cde178ebc35280814c37f90eec5aea47e964fc17a1f"
ORIENTATION_SHA="e43908188f134182635ac36a9a1602abad1918df374a04f3f7cc9eb1812d7497"
COARSE_OUTWARD_SIGN=-1
HELPER=DIR/"current_parent_normal_helper555.py"
HELPER_SHA="431f4af0b1308a1c8d90f22cf9be65e5dd13b97491ba6ddfd3b4c269bf56e5e5"
RUNNER=Path(__file__).resolve()
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
    if spec is None or spec.loader is None:raise ImportError(f"cannot load pinned source {path}")
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:raise ImportError(f"cannot load helper {path}")
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


def _npz_dict(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def decorate_target_outputs(report, arrays, *, source_normals, source_provenance,
                            parent_normals, parent_provenance, orientation_sign):
    """Separate proposed parent normals from registered source-normal references."""
    rows=[int(r["query_row"]) for r in report.get("targets",[])]
    if len(rows)!=len(set(rows)) or not set(rows).issubset(source_normals) or not set(rows).issubset(parent_normals):
        raise ValueError("target output rows lack normal/provenance maps")
    if int(orientation_sign) not in (-1,1):
        raise ValueError("current-parent direction has no pinned orientation sign")
    for rec in report["targets"]:
        row=int(rec["query_row"])
        pn=np.asarray(parent_normals[row],dtype=np.float64)
        sn=np.asarray(source_normals[row],dtype=np.float64)
        if pn.shape!=(3,) or sn.shape!=(3,) or not np.isfinite(pn).all() or not np.isfinite(sn).all():
            raise ValueError(f"invalid normal vectors for target row {row}")
        rec["proposal_direction_world_unit"]=pn.tolist()
        rec["current_parent_normal_world_unit"]=pn.tolist()
        rec["registered_source_normal_reference_world_unit"]=sn.tolist()
        rec["source_surface_normal_world_unit"]=sn.tolist()
        rec["source_normal_provenance"]=source_provenance[row]
        rec["current_parent_normal_provenance"]=parent_provenance[row]
        rec["direction_method"]="current-parent-normal"
    report["schema"]="numi.human.refined-point-current-parent-normal-targets-555.v1"
    report["direction_mode"]="current-parent-normal"
    report["direction_method"]=(
        "Proposal rays use the area-weighted oriented normals of the immediate parent O faces "
        "for appended edge-midpoint and face-centroid nodes. The orientation sign is the pinned "
        "303 coarse outward sign. Registered source normals are retained separately as reference "
        "and provenance. The existing exact first-union-exit selector and all-185 endpoint "
        "classification are unchanged; this is a construction target, not mesh admission."
    )
    report["normal_method"]=(
        "Registered source normals were computed by the pinned 303 typed-ancestry owner and are "
        "reference-only. Proposal direction is computed from immediate parent face coordinates "
        "and node-to-parent barycentric ancestry."
    )
    report["current_parent_normal_orientation_sign_from_303"]=int(orientation_sign)
    report["registered_source_normals_are_reference_only"]=True
    report["depth_covector_used"]=False
    report["native_admission"]=False
    report["physical_steps"]=0

    expected=np.asarray(rows,dtype=np.int64)
    out=dict(arrays)
    if not np.array_equal(np.asarray(out.get("query_rows"),dtype=np.int64),expected):
        raise ValueError("NPZ target row order differs from report rows")
    current=np.asarray([parent_normals[row] for row in rows],dtype=np.float64).reshape(-1,3)
    registered=np.asarray([source_normals[row] for row in rows],dtype=np.float64).reshape(-1,3)
    if not np.allclose(np.asarray(out["proposal_direction_world_unit"]),current,rtol=0,atol=1e-12):
        raise ValueError("545 proposal vectors differ from the injected current-parent normals")
    depth=np.asarray(out.get("depth_covector_world_unit"),dtype=np.float64)
    if depth.shape!=(len(rows),3) or np.isfinite(depth).any():
        raise ValueError("current-parent mode must not contain fabricated 357 depth covectors")
    out["proposal_direction_world_unit"]=current
    out["current_parent_normal_world_unit"]=current
    out["current_parent_normal_available_mask"]=np.ones(len(rows),dtype=np.bool_)
    out["source_surface_normal_world_unit"]=registered
    out["registered_source_normal_reference_world_unit"]=registered
    out["source_normal_available_mask"]=np.ones(len(rows),dtype=np.bool_)
    out["depth_covector_used_mask"]=np.zeros(len(rows),dtype=np.bool_)
    return report,out


def resolve_coarse_outward_sign(source303_path, orientation_path):
    """Read the coarse winding sign from the validated, pinned 303 owner."""
    if Path(source303_path).resolve()!=SOURCE303.resolve() or Path(orientation_path).resolve()!=ORIENT.resolve():
        raise ValueError("source orientation paths differ from pinned 303 inputs")
    if sha(source303_path)!=SOURCE303_SHA or sha(orientation_path)!=ORIENTATION_SHA:
        raise ValueError("registered source orientation inputs do not match 303 pins")
    source=load_pinned(source303_path,SOURCE303_SHA,"source303_orientation_for_current_parent555")
    result=source.validate_source_outward_orientation(json.loads(Path(orientation_path).read_text()))
    sign=int(result.get("coarse_outward_sign",0))
    if sign!=COARSE_OUTWARD_SIGN:
        raise ValueError(f"registered 303 coarse outward sign changed: {sign}")
    return sign


def invoke_and_annotate_builder(builder_call, builder_args, report_path, arrays_path, *,
                                source_normals, source_provenance, parent_normals,
                                parent_provenance, orientation_sign, input_metadata):
    """Run builder, annotate artifacts when available, and preserve its failure."""
    build_error=None
    try:
        builder_call(builder_args)
    except BaseException as exc:
        build_error=exc
    report=None
    post_error=None
    report_path=Path(report_path);arrays_path=Path(arrays_path)
    if report_path.is_file() and arrays_path.is_file():
        try:
            report=json.loads(report_path.read_text())
            arrays=_npz_dict(arrays_path)
            report,arrays=decorate_target_outputs(report,arrays,
                source_normals=source_normals,source_provenance=source_provenance,
                parent_normals=parent_normals,parent_provenance=parent_provenance,
                orientation_sign=orientation_sign)
            report.setdefault("inputs",{}).update(input_metadata)
            with arrays_path.open("wb") as f:
                np.savez_compressed(f,**arrays)
            outputs=report.setdefault("outputs",{})
            outputs["npz"]=str(arrays_path)
            outputs["npz_sha256"]=sha(arrays_path)
            outputs["current_parent_normal_postprocessor_sha256"]=sha(RUNNER)
            outputs["current_parent_normal_helper_sha256"]=sha(HELPER)
            if build_error is not None:
                report["status"]="INCOMPLETE_BUILDER_ERROR_AFTER_OUTPUT"
                report["builder_error"]=repr(build_error)
            else:
                report["output_annotation_status"]="PASS"
            report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        except BaseException as exc:
            post_error=exc
            try:
                partial=json.loads(report_path.read_text())
                partial["direction_mode"]="current-parent-normal"
                partial["direction_method"]="Current-parent-normal target attempt; output annotation incomplete."
                partial["native_admission"]=False;partial["physical_steps"]=0
                partial["output_annotation_status"]="FAILED"
                partial["output_annotation_error"]=repr(exc)
                if build_error is not None:partial["builder_error"]=repr(build_error)
                partial["status"]="INCOMPLETE_CURRENT_PARENT_NORMAL_OUTPUT_ANNOTATION"
                report_path.write_text(json.dumps(partial,indent=2,sort_keys=True)+"\n")
            except BaseException:
                pass
    elif report_path.is_file():
        try:
            partial=json.loads(report_path.read_text())
            partial["direction_mode"]="current-parent-normal"
            partial["direction_method"]="Current-parent-normal target attempt; target-array output was not emitted."
            partial["native_admission"]=False;partial["physical_steps"]=0
            partial["output_annotation_status"]="INCOMPLETE_NO_ARRAYS"
            partial["status"]="INCOMPLETE_CURRENT_PARENT_NORMAL_ARRAY_OUTPUT"
            if build_error is not None:partial["builder_error"]=repr(build_error)
            report_path.write_text(json.dumps(partial,indent=2,sort_keys=True)+"\n")
        except BaseException as exc:
            post_error=exc
    if build_error is not None:
        raise build_error
    if post_error is not None:
        raise post_error
    if report is None or not report_path.is_file() or not arrays_path.is_file():
        raise RuntimeError("545 union-exit builder did not emit complete target report and arrays")
    return report


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--direction-mode",required=True,choices=("current-parent-normal",))
    ap.add_argument("--parent-candidate",required=True,type=Path)
    ap.add_argument("--parent-point-summary",required=True,type=Path)
    ap.add_argument("--candidate",required=True,type=Path)
    ap.add_argument("--point-summary",required=True,type=Path)
    ap.add_argument("--query-dir",required=True,type=Path)
    ap.add_argument("--output-dir",required=True,type=Path)
    ap.add_argument("--clearance-m",type=float,default=5e-5)
    cli=ap.parse_args(argv)
    if cli.output_dir.exists():raise FileExistsError(cli.output_dir)

    wrap=load_pinned(WRAPPER553,WRAPPER553_SHA,"source_outward_wrapper553_for_current_parent555")
    parent=_npz_dict(cli.parent_candidate);child=_npz_dict(cli.candidate)
    parent_summary=json.loads(cli.parent_point_summary.read_text())
    child_summary=json.loads(cli.point_summary.read_text())
    parent_sha=sha(cli.parent_candidate);child_sha=sha(cli.candidate)
    target_rows,old_count,new_count=wrap.validate_new_only_prefix(
        parent,child,parent_summary,child_summary,parent_sha,child_sha)
    if not target_rows:raise ValueError("there are no new inside rows to target")
    for key in ANCHOR_KEYS:
        if key in parent and (key not in child or parent[key].dtype!=child[key].dtype or
                parent[key].shape!=child[key].shape or parent[key].tobytes()!=child[key].tobytes()):
            raise ValueError(f"refinement changed typed anchor array {key}")

    if sha(HELPER)!=HELPER_SHA:raise ValueError("current-parent helper source pin mismatch")
    helper=load_module(HELPER,"current_parent_normal_helper555_for_runtime")
    paths={
        "parent_candidate":cli.parent_candidate,"parent_point_summary":cli.parent_point_summary,
        "candidate":cli.candidate,"point_summary":cli.point_summary,
        "query_receipt":cli.query_dir/"query-receipt.json",
        "query_points":cli.query_dir/"points.csv","query_nodes":cli.query_dir/"nodes.csv",
        "query_pairs":cli.query_dir/"pairs.csv","wrapper553":WRAPPER553,
        "target_builder545":BUILDER,"installer546":INSTALLER,"extension546":EXTENSION,
        "source_normal_owner500":OWNER500,"parent_normal_helper555":HELPER,
        "current_parent_launcher555":RUNNER,
    }
    installer=load_pinned(INSTALLER,INSTALLER_SHA,"direct_depth_installer_for_current_parent555")
    extension=load_pinned(EXTENSION,EXTENSION_SHA,"multi_parent_source_extension_for_current_parent555")
    if Path(extension.OWNER500).resolve()!=OWNER500.resolve():
        raise ValueError("multi-parent extension references a different 500 owner")
    if sha(Path(extension.OWNER500))!=OWNER500_SHA or extension.OWNER500_SHA!=OWNER500_SHA:
        raise ValueError("pinned 500 source-normal owner mismatch")
    builder=load_pinned(BUILDER,BUILDER_SHA,"union_exit_builder545_for_current_parent555")
    paths["source303_owner"]=Path(builder.SOURCE303)
    paths["source_orientation_report"]=Path(builder.ORIENT)
    pins={k:sha(v) for k,v in paths.items()}
    orientation_sign={"value":resolve_coarse_outward_sign(paths["source303_owner"],paths["source_orientation_report"])}
    source_modules={};face_provenance={}
    original_loadmod=builder.loadmod
    def load_with_multi_parent(path,name):
        module=original_loadmod(path,name)
        if Path(path).resolve()==Path(builder.SOURCE303).resolve():
            installer.install_source303_extension(module,extension,face_provenance)
            source_modules["303"]=module
        return module
    builder.loadmod=load_with_multi_parent
    original_normals=builder.vertex_source_outward_normals
    registered_normals={};registered_provenance={};parent_normals={};parent_provenance={};orientation_sign={}
    def current_parent_direction(candidate,source303,detail,base,orientation,rows):
        refs,ref_prov=original_normals(candidate,source303,detail,base,orientation,rows)
        sign=int(orientation.get("coarse_outward_sign",0))
        if sign!=COARSE_OUTWARD_SIGN:
            raise ValueError(f"pinned owner coarse outward sign changed: {sign}")
        current,prov=helper.parent_normal_field(
            rows,parent_points=parent["points_world_m"],parent_faces=parent["source_O_faces"],
            child_points=candidate["points_world_m"],node_kind=candidate["refinement_node_kind"],
            parent_face_indptr=candidate["refinement_node_parent_face_indptr"],
            parent_face_rows=candidate["refinement_node_parent_face_rows"],
            parent_face_barycentric=candidate["refinement_node_parent_face_barycentric"],
            old_prefix_count=old_count,outward_sign=sign)
        registered_normals.update({int(k):np.asarray(v,dtype=np.float64).copy() for k,v in refs.items()})
        registered_provenance.update(ref_prov)
        parent_normals.update(current);parent_provenance.update(prov);orientation_sign["value"]=sign
        combined={int(k):{"registered_source_reference":ref_prov[int(k)],
                          "current_parent_normal":prov[int(k)]} for k in rows}
        return current,combined
    builder.vertex_source_outward_normals=current_parent_direction

    args=["--candidate",str(cli.candidate),"--point-summary",str(cli.point_summary),
          "--query-dir",str(cli.query_dir),"--output-dir",str(cli.output_dir),
          "--clearance-m",format(cli.clearance_m,".17g"),"--direction-mode","source-outward"]
    report_path=cli.output_dir/"nearest-union-exits.json"
    arrays_path=cli.output_dir/"nearest-union-exits.npz"
    input_metadata={
        "current_parent_candidate_sha256":parent_sha,
        "current_parent_faces_sha256":hashlib.sha256(
            np.ascontiguousarray(parent["source_O_faces"]).tobytes()).hexdigest(),
        "immediate_parent_normal_helper_sha256":pins["parent_normal_helper555"],
        "current_parent_normal_postprocessor_sha256":pins["current_parent_launcher555"],
    }
    report=invoke_and_annotate_builder(builder.main,args,report_path,arrays_path,
        source_normals=registered_normals,source_provenance=registered_provenance,
        parent_normals=parent_normals,parent_provenance=parent_provenance,
        orientation_sign=orientation_sign.get("value",0),input_metadata=input_metadata)

    if set(parent_normals)!=set(target_rows) or set(registered_normals)!=set(target_rows):
        raise RuntimeError("not every appended inside node has both current and registered source normals")
    if not all(r.get("status")=="PASS_FIRST_UNION_EXIT" and
               r.get("target_exactly_outside_all185") is True and
               int(r.get("exact_endpoint_bank_count",-1))==185 for r in report["targets"]):
        raise RuntimeError("not every current-parent-normal endpoint passed all185")
    if {k:sha(v) for k,v in paths.items()}!=pins:
        raise RuntimeError("a pinned source/input changed during target construction")
    receipt={
        "schema":"numi.human.refined_current_parent_normal_targets_555.v1",
        "status":"PASS_CURRENT_PARENT_NORMAL_UNION_EXIT_TARGETS_ONLY",
        "classification":"proposal-only; no coordinate movement, mesh admission, or physical steps",
        "direction_mode":"current-parent-normal",
        "direction_source":"immediate parent candidate source_O_faces, oriented by pinned 303 coarse outward sign",
        "registered_source_normal_role":"reference and provenance only; not used as proposal direction",
        "source_orientation_sign":orientation_sign["value"],
        "pinned_coarse_outward_sign":COARSE_OUTWARD_SIGN,
        "parent_candidate_sha256":parent_sha,"candidate_sha256":child_sha,
        "parent_point_summary_sha256":pins["parent_point_summary"],
        "point_summary_sha256":pins["point_summary"],"query_receipt_sha256":pins["query_receipt"],
        "target_report_sha256":sha(report_path),"target_npz_sha256":sha(arrays_path),
        "target_rows":target_rows,"target_count":len(target_rows),
        "all_target_rows_are_appended_refinement_nodes":True,
        "point_prefix_count":old_count,"refined_point_count":new_count,
        "old_point_prefix_bitwise_unchanged":True,"typed_anchor_arrays_unchanged":True,
        "all185_endpoint_targets_passed":True,"depth_covectors_used":False,
        "endpoint_certification_mode":"exact 545 first-union-exit with complete 185-bank endpoint classification",
        "native_admission":False,"accepted_physical_steps":0,"source_pins":pins,
    }
    receipt_path=cli.output_dir/"current-parent-normal-targets-555.json"
    receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return 0


if __name__=="__main__":
    raise SystemExit(main())

