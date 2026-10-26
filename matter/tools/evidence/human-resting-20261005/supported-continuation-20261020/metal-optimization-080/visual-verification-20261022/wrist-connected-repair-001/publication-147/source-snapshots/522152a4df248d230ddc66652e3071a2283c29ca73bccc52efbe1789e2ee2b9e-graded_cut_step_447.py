"""Opt-in 431 alpha projection preserving full self seeds and pinned 313 guards.

All fixed-bone rows are loaded from an explicitly hashed 314 export and are
rebuilt from its typed records with the pinned 313/026 owners. They enter the
same alpha LP as failed-pair rows; emitted geometry still requires exact gates.
Required in addition to 431/423 environment:
  NUMI_GRADED_FIXED_GUARD_NPZ_PATH_447
  NUMI_GRADED_FIXED_GUARD_NPZ_SHA256_447
  NUMI_GRADED_FIXED_GUARD_REPORT_PATH_447
  NUMI_GRADED_FIXED_GUARD_REPORT_SHA256_447
  NUMI_GRADED_FIXED_GUARD_BASE_CANDIDATE_SHA256_447
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import numpy as np

BASE=Path("/Users/numi/Work/numi-human-connected-surface-069")
CUTS431=BASE/"D/graded-cut-step-431/graded_cut_step_431.py"
CUTS431_SHA256="615b579d5a7c5d70ae788cf905c2d09b117a130c828717c0794c3e8eff30d37b"
FIXED_VALIDATOR=BASE/"D/retained-gain-continuation-401/propose_joint_clearance_witnesses_401.py"
FIXED_VALIDATOR_SHA256="2f1eda3f0d46d0f5497a1a04075b5dc2c3b37afe906d0389ff636fca8abc5869"
FIXED_OWNER=BASE/"connected-fixed-boundary-313/fixed_bone_triangle_rows_313.py"
FIXED_OWNER_SHA256="dc9d4f953ecede00567c59cbbb6b6b63e0a3556b97972689de499cbce29d4c2e"
ROW_TOL_MM=2e-8


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def _load_pinned(path, expected_sha, module_name):
    if not Path(path).is_file() or _sha(path)!=expected_sha:
        raise ValueError(f"447 pinned source hash mismatch: {path}")
    spec=importlib.util.spec_from_file_location(module_name,path)
    if spec is None or spec.loader is None: raise ValueError(f"447 cannot load {path}")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


cuts431=_load_pinned(CUTS431,CUTS431_SHA256,"graded_cut_step_431_pinned_for_447")
fixed_validation=_load_pinned(FIXED_VALIDATOR,FIXED_VALIDATOR_SHA256,"fixed_boundary_validation_401_pinned_for_447")
fixed_owner=_load_pinned(FIXED_OWNER,FIXED_OWNER_SHA256,"fixed_triangle_rows_313_pinned_for_447")


def _require_env(name):
    value=os.environ.get(name)
    if not value: raise ValueError(f"447 requires {name}")
    return value


def load_fixed_guard_rows(context_npz_path, faces, context):
    """Authenticate 449-style 314 payload and rebuild its rows with pinned 313."""
    npz_path=Path(_require_env("NUMI_GRADED_FIXED_GUARD_NPZ_PATH_447"))
    npz_sha=_require_env("NUMI_GRADED_FIXED_GUARD_NPZ_SHA256_447").lower()
    report_path=Path(_require_env("NUMI_GRADED_FIXED_GUARD_REPORT_PATH_447"))
    report_sha=_require_env("NUMI_GRADED_FIXED_GUARD_REPORT_SHA256_447").lower()
    candidate_sha=_require_env("NUMI_GRADED_FIXED_GUARD_BASE_CANDIDATE_SHA256_447").lower()
    if not npz_path.is_file() or _sha(npz_path)!=npz_sha:
        raise ValueError("447 fixed-guard NPZ path/hash mismatch")
    if not report_path.is_file() or _sha(report_path)!=report_sha:
        raise ValueError("447 fixed-guard report path/hash mismatch")
    with np.load(context_npz_path,allow_pickle=False) as z:
        points_f32=np.asarray(z["points_baseline_m"])
        context_faces=np.asarray(z["faces"])
    if points_f32.dtype!=np.dtype(np.float32):
        raise ValueError("447 pinned context baseline points must be Float32")
    if context_faces.dtype.kind not in "iu" or not np.array_equal(context_faces,faces):
        raise ValueError("447 runtime faces differ from fixed-guard context topology")
    report=json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("base_candidate_sha256")!=candidate_sha:
        raise ValueError("447 fixed-guard base candidate SHA differs from pinned input")
    with np.load(npz_path,allow_pickle=False) as z:
        arrays={name:np.asarray(z[name]).copy() for name in z.files}
    records,receipt=fixed_validation.validate_fixed_boundary_payload(
        arrays,report,candidate_sha256=candidate_sha,candidate_points=points_f32,
        candidate_faces=context_faces,npz_path=npz_path)
    rows,typed=fixed_validation.build_fixed_boundary_barriers(
        points_f32.astype(np.float64)*1000.0,context_faces.astype(np.int64,copy=False),
        records,cuts431.cuts.sat,fixed_owner,0.25)
    if not rows or not typed:
        raise ValueError("447 authenticated fixed-boundary export produced no typed guard rows")
    receipt.update({"enabled":True,"report_path":str(report_path),"report_sha256":report_sha,
        "candidate_sha256":candidate_sha,"npz_path":str(npz_path),"npz_sha256":npz_sha,
        "typed_pair_count":int(len(typed)),"linear_row_count":int(len(rows)),
        "313_owner_path":str(FIXED_OWNER),"313_owner_sha256":FIXED_OWNER_SHA256,
        "row_margin_fraction":0.25,"construction_only":True})
    return rows,receipt


def validate_rows(rows,node_count):
    """Validate global-node 026 row structure before LP assembly."""
    checked=[]
    for index,row in enumerate(rows):
        if not isinstance(row,dict): raise ValueError(f"447 row {index} is not a mapping")
        raw_nodes=np.asarray(row.get("nodes"))
        coeff=np.asarray(row.get("coefficients"),dtype=np.float64)
        lower=row.get("lower")
        if (raw_nodes.ndim!=1 or raw_nodes.dtype.kind not in "iu" or len(raw_nodes)==0
                or coeff.shape!=(len(raw_nodes),3) or not np.isfinite(coeff).all()
                or isinstance(lower,(bool,np.bool_)) or not np.isscalar(lower)
                or not np.isfinite(float(lower))):
            raise ValueError(f"447 row {index} is malformed")
        nodes=raw_nodes.astype(np.int64,copy=False)
        if np.any(nodes<0) or np.any(nodes>=node_count) or len(np.unique(nodes))!=len(nodes):
            raise ValueError(f"447 row {index} has invalid global node IDs")
        if float(lower)>ROW_TOL_MM:
            raise ValueError(f"447 row {index} is infeasible at the zero-displacement base")
        checked.append({"nodes":nodes,"coefficients":coeff,"lower":float(lower),
                        **({"label":row["label"]} if "label" in row else {})})
    return checked


def fixed_guard_certificate(rows, displacement_mm, alpha, anchors=(), tolerance_mm=ROW_TOL_MM):
    """Certify original, unclamped fixed rows and anchor values in millimetres."""
    d=np.asarray(displacement_mm,dtype=np.float64); x=np.asarray(alpha,dtype=np.float64)
    if d.ndim!=2 or d.shape[1]!=3 or x.shape!=(len(d),) or not np.isfinite(d).all() or not np.isfinite(x).all():
        raise ValueError("447 fixed-guard certificate inputs are malformed")
    rows=validate_rows(rows,len(d)); residuals=[]
    for row in rows:
        nodes=row["nodes"]
        value=float(np.einsum("ij,ij->",row["coefficients"],d[nodes]*x[nodes,None]))
        residuals.append(value-row["lower"])
    residuals=np.asarray(residuals,dtype=np.float64)
    violated=np.flatnonzero(residuals < -float(tolerance_mm))
    anchor_ids=np.asarray(anchors,dtype=np.int64).reshape(-1)
    if len(anchor_ids) and (np.any(anchor_ids<0) or np.any(anchor_ids>=len(x))):
        raise ValueError("447 anchor outside node domain")
    anchor_max=float(np.abs(x[anchor_ids]).max(initial=0.0))
    passed=not len(violated) and anchor_max<=1e-12
    return {"passed":bool(passed),"row_count":int(len(rows)),
        "checked_rows":int(len(rows)),"violated_rows":int(len(violated)),
        "max_violation_mm":float(np.maximum(-residuals,0.0).max(initial=0.0)),
        "min_slack_mm":float(residuals.min(initial=np.inf)),
        "tolerance_mm":float(tolerance_mm),"anchor_rows_checked":anchor_ids.tolist(),
        "anchor_max_abs_scale":anchor_max,"certificate":"UNCLAMPED_313_FIXED_BONE_ALPHA_ROWS"}


def solve_with_fixed_guard_rows(owner, faces, upper, anchors, edges, budgets,
                                displacement_mm, failed_rows, packed, cache_key,
                                guard_rows, guard_receipt=None):
    """Run immutable 431 seed projector with added permanent fixed guard rows."""
    n=len(upper); failed_rows=validate_rows(failed_rows,n); original=validate_rows(guard_rows,n)
    lp_guards=[]; clamped=0; max_clamp=0.0
    for row in original:
        lp_row=dict(row)
        # 431's LP expects exact zero-step feasibility. Tiny positive rhs values
        # within its accepted row tolerance are projected to zero for LP use;
        # the final certificate below checks every original rhs unchanged.
        if 0.0<float(lp_row["lower"])<=owner.ROW_TOL_MM:
            max_clamp=max(max_clamp,float(lp_row["lower"])); lp_row["lower"]=0.0; clamped+=1
        lp_guards.append(lp_row)
    alpha,proof=owner._solve_with_seed_cuts(faces,upper,anchors,edges,budgets,
        displacement_mm,failed_rows+lp_guards,packed,cache_key)
    cert=fixed_guard_certificate(original,displacement_mm,alpha,anchors,owner.ROW_TOL_MM)
    cert.update({"lp_positive_lower_clamped_rows":int(clamped),
                 "lp_max_positive_lower_clamp_mm":float(max_clamp),
                 "input_receipt":guard_receipt or {"enabled":False}})
    if not cert["passed"]:
        raise ValueError("447 final original fixed-guard row/anchor certificate failed")
    proof["fixed_guard_certificate"]=cert
    return alpha,proof


def reduce_witness_node_scales(faces,step_scales,bad_face_rows,self_pair_face_rows,*,anchors=(),zero_below=1/128):
    """431 scale projection plus full self-seed and typed fixed-bone guard rows."""
    context=cuts431.cuts.g.load_context_423()
    faces=np.asarray(faces,dtype=np.int64)
    upper,report=cuts431.cuts.g.reduce_witness_node_scales(
        faces,step_scales,bad_face_rows,self_pair_face_rows,anchors=anchors,zero_below=zero_below)
    pairs_now=np.asarray(self_pair_face_rows,dtype=np.int64)
    if pairs_now.size==0: pairs_now=np.empty((0,2),dtype=np.int64)
    if pairs_now.ndim!=2 or pairs_now.shape[1]!=2: raise ValueError("447 failed-pair ids must have shape (K,2)")
    before=len(cuts431._FAILED_PAIRS)
    cuts431._FAILED_PAIRS.update(tuple(sorted(map(int,p))) for p in pairs_now)
    context_path=Path(os.environ["NUMI_GRADED_LOCAL_STEP_CONTEXT_423"])
    with np.load(context_path,allow_pickle=False) as z:
        baseline=np.asarray(z["points_baseline_m"],dtype=np.float64)*1000.0
        displacement=np.asarray(z["proposal_displacement_m"],dtype=np.float64)*1000.0
    pair_ids=np.asarray(sorted(cuts431._FAILED_PAIRS),dtype=np.int64).reshape(-1,2)
    failed_rows,failed_records=(cuts431.cuts.sat.build_pair_rows(
        baseline,faces,pair_ids,margin_fraction=cuts431.MARGIN_FRACTION) if len(pair_ids) else ([],[]))
    seeds,packed,cache_path,report_path,meta=cuts431._load_seed_rows(faces,context)
    key=json.dumps(meta,sort_keys=True,separators=(",",":"))
    guards,guard_receipt=load_fixed_guard_rows(context_path,faces,context)
    out,proof=solve_with_fixed_guard_rows(cuts431,faces,upper,anchors,context["edges"],
        context["budget"],displacement,failed_rows,packed,key,guards,guard_receipt)
    report.update(status="GRADED_426_WITH_FULL_SELF_SEEDS_AND_FIXED_BONE_GUARDS_447",
        policy="423_PLUS_026_FAILED_AND_FULL_SEED_CUTS_PLUS_PINNED_313_FIXED_BONE_ROWS",
        seed_pair_path=meta["seed_path"],seed_pair_sha256=meta["seed_sha256"],
        seed_pair_count=meta["seed_pair_count"],seed_row_cache_path=str(cache_path),
        seed_row_cache_sha256=cuts431._sha(cache_path),seed_linear_row_count=int(len(packed["lower"])),
        baseline_points_sha256=meta["baseline_points_sha256"],faces_sha256=meta["faces_sha256"],
        sat_owner_sha256=cuts431.SAT_SHA256,failed_pair_count=int(len(pair_ids)),
        failed_pair_rows=int(len(failed_rows)),new_failed_pairs_this_call=int(len(cuts431._FAILED_PAIRS)-before),
        failed_pair_records=failed_records,seed_cut_lp=proof,native_admission=False,accepted_physical_steps=0)
    cuts431._write_report(report_path,report)
    return out,report
