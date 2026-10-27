"""Prepared-input adapter and bounded selector entrypoint.

Default execution is a preparation-only preflight. Pass --solve explicitly to
run small, capped per-incidence-component MILPs. The result is not an emitted
mesh and does not replace any 603/545/130/159 acceptance gate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import __version__ as scipy_version

HERE = Path(__file__).resolve().parent
D = Path("/Users/numi/Work/numi-human-connected-surface-069")
DEFAULT_PREPARED_DIR = D / "D/surface-clearance-546/hard-face-feasibility-634/prepared-from764-001/prepared"
BASE_DIR = DEFAULT_PREPARED_DIR.parent
CURRENT_INPUTS = BASE_DIR / "current-inputs.json"
FIT_ARGV = DEFAULT_PREPARED_DIR / "fit-argv.json"
EXPECTED_CANDIDATE_SHA = "d0d8ba18026d37a1ea64a247e09e4f6e972db0b157d64ad075851ab76c1c4ca4"
EXPECTED_BOUNDARY_NPZ_SHA = "2112c14a3e3e900bc040c9e1d775d22e2f2e4e8e3137d45291654e960857a3c9"
EXPECTED_CURRENT_INPUTS_SHA = "f47a4bddc657209a384f0e45901ef648438a29a3fc09b6a40eabeabdb582eaaa"
EXPECTED_FIT_ARGV_SHA = "a8a811f0d43539f9175dbd9dead2bb3bf6fda908678f15a0303f364f996334a2"
ACTIVE_STATE_PATH = None
EXPECTED_BOUNDARY_NPZ_SHA = "2112c14a3e3e900bc040c9e1d775d22e2f2e4e8e3137d45291654e960857a3c9"
EXPECTED_CURRENT_INPUTS_SHA = "f47a4bddc657209a384f0e45901ef648438a29a3fc09b6a40eabeabdb582eaaa"
EXPECTED_FIT_ARGV_SHA = "a8a811f0d43539f9175dbd9dead2bb3bf6fda908678f15a0303f364f996334a2"
ACTIVE_STATE_PATH = None
EXPECTED_SOURCE_SHA = "70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31"
EXPECTED_AUDIT159_SHA = "980264f285e805a4edb6d0a70b8fe706d72560e94557900613e4a2bab9603b36"
EXPECTED_FACE_TARGET_SHA = "076d744efe0e6d650eb535dce2b91152f6bc84a824197ee812e92731caececdf"
EXPECTED_POINT_REPORT_SHA = "bf5084ccb67eb69b3517f67e35d8594012168aac5bd5e1e7b55598132a8f60c0"
EXPECTED_FACE_REPORT_SHA = "a63b3aeecafbf380d7082757ede4a4bd883fedf26428d8adf82640457701fad4"
EXPECTED_BOUNDARY_REPORT_SHA = "316909b9f9db5dace03dae6d6264437fb8db377d8f2d852f5c754d86b85b196f"
EXPECTED_SELF_PAIRS_SHA = "3c141192ae2e093d6951958c727dac2139bff4969fc9995a86d8a45ee55fdaaa"
EXPECTED_STATE_SHA = "d550f351dadbffa26085960c377dd5754dc7d175ac33d31de9d80821ec266b4d"
EXPECTED_POINT_TARGET_SHA = "f6557b7e8840873f7700c053a404932372f3f1464bef617bcaedbd669b08df43"
SAT_PATH = D / "D/surface-clearance-546/pair-sat-routing-551/pair_sat_routing_551.py"
SAT_SHA = "27431e9ed3e92886394b59f1156a743e508fba8c81cf8b1ff94a7d80e3ef201f"
BOX_PATH = D / "D/surface-clearance-546/directional-source-bounds-636/directional_bounds636_v2.py"
BOX_SHA = "fb3e9cda895b086cf2f5f15ca28f35b43e84c912c73db443e3c7d04cacbd10a8"
OWNER408_PATH = D / "D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
OWNER408_SHA = "5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2"
RUNNER458_PATH = D / "D/D/fixed-bone-outward-plane-458/run_grouped_fit_458.py"
RUNNER458_SHA = "08cd0e2681bb5ae16a85a0384514bb30739f8d62a06ea2ab1bdfcd896be4ef93"
HELPER638_PATH = HERE / "connected_patch_sat_selector638.py"

MAX_TOTAL_SECONDS = 60.0
MAX_COMPONENT_SECONDS = 5.0
MAX_NODE_LIMIT = 25000
MAX_TOTAL_BINARY = 20000
MAX_COMPONENT_BINARY = 6000
MAX_COMPONENT_ROWS = 20000
MAX_TOTAL_ROWS = 100000
MAX_PAIRS = 1200
FEASIBILITY_TOL_MM = 2e-9


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(a: np.ndarray, dtype) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=dtype).tobytes()).hexdigest()


def load_module(name: str, path: Path, expected_sha: str):
    actual=sha(path)
    if actual != expected_sha:
        raise RuntimeError(f"source pin mismatch: {path} expected {expected_sha} got {actual}")
    spec=importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(f"cannot load {path}")
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


def arg_path(argv: list[str], key: str) -> Path:
    if argv.count(key) != 1:
        raise ValueError(f"fit argv must contain exactly one {key}")
    i=argv.index(key)
    if i+1 >= len(argv): raise ValueError(f"missing path after {key}")
    return Path(argv[i+1]).resolve()


def stable_json(path: Path):
    return json.loads(path.read_text())


def configure_preparation(prepared_dir: Path, current_inputs_path: Path | None = None, state_path: Path | None = None):
    """Bind preparation paths and content hashes from an immutable prep bundle."""
    global BASE_DIR, CURRENT_INPUTS, FIT_ARGV, ACTIVE_STATE_PATH
    global EXPECTED_CANDIDATE_SHA, EXPECTED_AUDIT159_SHA, EXPECTED_FACE_TARGET_SHA
    global EXPECTED_POINT_REPORT_SHA, EXPECTED_FACE_REPORT_SHA, EXPECTED_BOUNDARY_REPORT_SHA
    global EXPECTED_SELF_PAIRS_SHA, EXPECTED_POINT_TARGET_SHA, EXPECTED_STATE_SHA
    global EXPECTED_BOUNDARY_NPZ_SHA, EXPECTED_CURRENT_INPUTS_SHA, EXPECTED_FIT_ARGV_SHA
    prepared_dir=Path(prepared_dir).resolve()
    if not prepared_dir.is_dir():
        raise FileNotFoundError(f"prepared directory does not exist: {prepared_dir}")
    BASE_DIR=prepared_dir.parent
    CURRENT_INPUTS=Path(current_inputs_path).resolve() if current_inputs_path else BASE_DIR/"current-inputs.json"
    FIT_ARGV=prepared_dir/"fit-argv.json"
    manifest=stable_json(CURRENT_INPUTS)
    fit_args=stable_json(FIT_ARGV)
    if not isinstance(manifest,dict) or not isinstance(fit_args,list):
        raise ValueError("prepared manifests have invalid shape")
    argv=fit_args[1:]
    flags=("--base-candidate","--source-base-npz","--point-target-npz","--point-target-report",
           "--face-target-npz","--face-target-report","--fixed-boundary-npz","--fixed-boundary-report",
           "--self-pairs","--audit159-490","--matched-inputs-json-490")
    paths={flag:arg_path(argv,flag) for flag in flags}
    def scalar(flag):
        if argv.count(flag)!=1: raise ValueError(f"fit argv must contain exactly one {flag}")
        j=argv.index(flag)
        if j+1>=len(argv): raise ValueError(f"missing scalar after {flag}")
        return argv[j+1]
    if abs(float(scalar("--trust-mm"))-4.0)>1e-12 or abs(float(scalar("--source-cap-mm"))-12.0)>1e-12:
        raise ValueError("selector preflight is pinned to the original 4 mm trust / 12 mm source balls")
    if int(scalar("--fixed-prefix-nodes"))!=0:
        raise ValueError("selector preflight does not support a fixed-prefix override")
    if str(paths["--base-candidate"])!=str(Path(manifest["candidate"]).resolve()):
        raise ValueError("fit argv candidate path differs from current-inputs.json")
    if sha(paths["--base-candidate"])!=manifest.get("candidate_sha256"):
        raise ValueError("candidate SHA differs from current-inputs.json")
    if str(paths["--audit159-490"])!=str(Path(manifest["surface_report"]).resolve()):
        raise ValueError("fit argv audit159 path differs from current-inputs.json")
    if sha(paths["--audit159-490"])!=manifest.get("surface_report_sha256"):
        raise ValueError("audit159 SHA differs from current-inputs.json")
    matched=stable_json(paths["--matched-inputs-json-490"])
    roles=matched.get("inputs",{})
    role_flags={"candidate":"--base-candidate","audit159":"--audit159-490",
                "point_target_npz":"--point-target-npz","face_target_npz":"--face-target-npz"}
    for role,flag in role_flags.items():
        rec=roles.get(role,{})
        path=paths[flag]
        if str(Path(rec.get("path","")).resolve())!=str(path) or rec.get("sha256")!=sha(path):
            raise ValueError(f"490 matched-inputs mismatch for {role}")
    if sha(paths["--source-base-npz"])!=EXPECTED_SOURCE_SHA:
        raise ValueError("source endpoint does not match pinned registered source")
    state_expected=Path(manifest["state"]).resolve()
    ACTIVE_STATE_PATH=Path(state_path).resolve() if state_path else state_expected
    EXPECTED_STATE_SHA=manifest.get("state_sha256")
    if not EXPECTED_STATE_SHA or sha(ACTIVE_STATE_PATH)!=EXPECTED_STATE_SHA:
        raise ValueError("selected state path is not byte-identical to manifest state")
    EXPECTED_CANDIDATE_SHA=manifest["candidate_sha256"]
    EXPECTED_AUDIT159_SHA=manifest["surface_report_sha256"]
    EXPECTED_POINT_TARGET_SHA=sha(paths["--point-target-npz"])
    EXPECTED_POINT_REPORT_SHA=sha(paths["--point-target-report"])
    EXPECTED_FACE_TARGET_SHA=sha(paths["--face-target-npz"])
    EXPECTED_FACE_REPORT_SHA=sha(paths["--face-target-report"])
    EXPECTED_BOUNDARY_NPZ_SHA=sha(paths["--fixed-boundary-npz"])
    EXPECTED_BOUNDARY_REPORT_SHA=sha(paths["--fixed-boundary-report"])
    EXPECTED_SELF_PAIRS_SHA=sha(paths["--self-pairs"])
    EXPECTED_CURRENT_INPUTS_SHA=sha(CURRENT_INPUTS)
    EXPECTED_FIT_ARGV_SHA=sha(FIT_ARGV)
    return {"prepared_dir":str(prepared_dir),"current_inputs":str(CURRENT_INPUTS),
            "fit_argv":str(FIT_ARGV),"state":str(ACTIVE_STATE_PATH),
            "candidate_sha256":EXPECTED_CANDIDATE_SHA,"state_sha256":EXPECTED_STATE_SHA,
            "audit159_sha256":EXPECTED_AUDIT159_SHA,"matched_inputs":str(paths["--matched-inputs-json-490"])}


def configure_preparation(prepared_dir: Path, current_inputs_path: Path | None = None, state_path: Path | None = None):
    """Bind preparation paths and content hashes from an immutable prep bundle."""
    global BASE_DIR, CURRENT_INPUTS, FIT_ARGV, ACTIVE_STATE_PATH
    global EXPECTED_CANDIDATE_SHA, EXPECTED_AUDIT159_SHA, EXPECTED_FACE_TARGET_SHA
    global EXPECTED_POINT_REPORT_SHA, EXPECTED_FACE_REPORT_SHA, EXPECTED_BOUNDARY_REPORT_SHA
    global EXPECTED_SELF_PAIRS_SHA, EXPECTED_POINT_TARGET_SHA, EXPECTED_STATE_SHA
    global EXPECTED_BOUNDARY_NPZ_SHA, EXPECTED_CURRENT_INPUTS_SHA, EXPECTED_FIT_ARGV_SHA
    prepared_dir=Path(prepared_dir).resolve()
    if not prepared_dir.is_dir():
        raise FileNotFoundError(f"prepared directory does not exist: {prepared_dir}")
    BASE_DIR=prepared_dir.parent
    CURRENT_INPUTS=Path(current_inputs_path).resolve() if current_inputs_path else BASE_DIR/"current-inputs.json"
    FIT_ARGV=prepared_dir/"fit-argv.json"
    manifest=stable_json(CURRENT_INPUTS)
    fit_args=stable_json(FIT_ARGV)
    if not isinstance(manifest,dict) or not isinstance(fit_args,list):
        raise ValueError("prepared manifests have invalid shape")
    argv=fit_args[1:]
    flags=("--base-candidate","--source-base-npz","--point-target-npz","--point-target-report",
           "--face-target-npz","--face-target-report","--fixed-boundary-npz","--fixed-boundary-report",
           "--self-pairs","--audit159-490","--matched-inputs-json-490")
    paths={flag:arg_path(argv,flag) for flag in flags}
    if str(paths["--base-candidate"])!=str(Path(manifest["candidate"]).resolve()):
        raise ValueError("fit argv candidate path differs from current-inputs.json")
    if sha(paths["--base-candidate"])!=manifest.get("candidate_sha256"):
        raise ValueError("candidate SHA differs from current-inputs.json")
    if str(paths["--audit159-490"])!=str(Path(manifest["surface_report"]).resolve()):
        raise ValueError("fit argv audit159 path differs from current-inputs.json")
    if sha(paths["--audit159-490"])!=manifest.get("surface_report_sha256"):
        raise ValueError("audit159 SHA differs from current-inputs.json")
    matched=stable_json(paths["--matched-inputs-json-490"])
    roles=matched.get("inputs",{})
    role_flags={"candidate":"--base-candidate","audit159":"--audit159-490",
                "point_target_npz":"--point-target-npz","face_target_npz":"--face-target-npz"}
    for role,flag in role_flags.items():
        rec=roles.get(role,{})
        path=paths[flag]
        if str(Path(rec.get("path","")).resolve())!=str(path) or rec.get("sha256")!=sha(path):
            raise ValueError(f"490 matched-inputs mismatch for {role}")
    if sha(paths["--source-base-npz"])!=EXPECTED_SOURCE_SHA:
        raise ValueError("source endpoint does not match pinned registered source")
    state_expected=Path(manifest["state"]).resolve()
    ACTIVE_STATE_PATH=Path(state_path).resolve() if state_path else state_expected
    EXPECTED_STATE_SHA=manifest.get("state_sha256")
    if not EXPECTED_STATE_SHA or sha(ACTIVE_STATE_PATH)!=EXPECTED_STATE_SHA:
        raise ValueError("selected state path is not byte-identical to manifest state")
    EXPECTED_CANDIDATE_SHA=manifest["candidate_sha256"]
    EXPECTED_AUDIT159_SHA=manifest["surface_report_sha256"]
    EXPECTED_POINT_TARGET_SHA=sha(paths["--point-target-npz"])
    EXPECTED_POINT_REPORT_SHA=sha(paths["--point-target-report"])
    EXPECTED_FACE_TARGET_SHA=sha(paths["--face-target-npz"])
    EXPECTED_FACE_REPORT_SHA=sha(paths["--face-target-report"])
    EXPECTED_BOUNDARY_NPZ_SHA=sha(paths["--fixed-boundary-npz"])
    EXPECTED_BOUNDARY_REPORT_SHA=sha(paths["--fixed-boundary-report"])
    EXPECTED_SELF_PAIRS_SHA=sha(paths["--self-pairs"])
    EXPECTED_CURRENT_INPUTS_SHA=sha(CURRENT_INPUTS)
    EXPECTED_FIT_ARGV_SHA=sha(FIT_ARGV)
    return {"prepared_dir":str(prepared_dir),"current_inputs":str(CURRENT_INPUTS),
            "fit_argv":str(FIT_ARGV),"state":str(ACTIVE_STATE_PATH),
            "candidate_sha256":EXPECTED_CANDIDATE_SHA,"state_sha256":EXPECTED_STATE_SHA,
            "audit159_sha256":EXPECTED_AUDIT159_SHA,"matched_inputs":str(paths["--matched-inputs-json-490"])}


def prepare_current_inputs():
    h=load_module("sat551_for_selector638", SAT_PATH, SAT_SHA)
    box=load_module("bounds636_for_selector638", BOX_PATH, BOX_SHA)
    owner=load_module("owner408_for_selector638", OWNER408_PATH, OWNER408_SHA)
    runner=load_module("runner458_for_selector638", RUNNER458_PATH, RUNNER458_SHA)
    fit_args=stable_json(FIT_ARGV)
    manifest=stable_json(CURRENT_INPUTS)
    if not isinstance(fit_args,list) or not isinstance(manifest,dict):
        raise ValueError("current preparation manifests malformed")
    args_by=fit_args[1:]
    base=arg_path(args_by,"--base-candidate")
    source=arg_path(args_by,"--source-base-npz")
    point_npz=arg_path(args_by,"--point-target-npz")
    point_report=arg_path(args_by,"--point-target-report")
    face_npz=arg_path(args_by,"--face-target-npz")
    face_report=arg_path(args_by,"--face-target-report")
    boundary_npz=arg_path(args_by,"--fixed-boundary-npz")
    boundary_report=arg_path(args_by,"--fixed-boundary-report")
    self_pairs_path=arg_path(args_by,"--self-pairs")
    if args_by.count("--immediate-parent-basis") != 1:
        raise ValueError("prepared 603 fit does not use the pinned immediate-parent full-coordinate basis")
    if str(base) != manifest.get("candidate") or sha(base) != EXPECTED_CANDIDATE_SHA:
        raise ValueError("prepared-from764 candidate path/hash changed")
    if sha(source) != EXPECTED_SOURCE_SHA:
        raise ValueError("source-base hash differs from prepared 764 source")
    if sha(point_npz) != EXPECTED_POINT_TARGET_SHA or sha(face_npz) != EXPECTED_FACE_TARGET_SHA:
        raise ValueError("prepared point/face targets changed")
    if sha(point_report) != EXPECTED_POINT_REPORT_SHA or sha(face_report) != EXPECTED_FACE_REPORT_SHA:
        raise ValueError("prepared point/face target reports changed")
    if sha(boundary_report) != EXPECTED_BOUNDARY_REPORT_SHA or sha(self_pairs_path) != EXPECTED_SELF_PAIRS_SHA:
        raise ValueError("prepared fixed-boundary report or self-pair seeds changed")
    state_path=Path(ACTIVE_STATE_PATH or manifest["state"]).resolve()
    if sha(state_path) != EXPECTED_STATE_SHA or manifest.get("state_sha256") != EXPECTED_STATE_SHA:
        raise ValueError("prepared 764 selected state changed")
    audit=Path(manifest["surface_report"]).resolve()
    if sha(audit) != EXPECTED_AUDIT159_SHA:
        raise ValueError("current complete 159 report changed")
    if sha(CURRENT_INPUTS) != EXPECTED_CURRENT_INPUTS_SHA or sha(FIT_ARGV) != EXPECTED_FIT_ARGV_SHA:
        raise ValueError("prepared 764 manifests changed")
    if sha(boundary_npz) != EXPECTED_BOUNDARY_NPZ_SHA:
        raise ValueError("prepared fixed-boundary NPZ changed")

    os.environ["NUMI_FIXED_BONE_OUTWARD_CANDIDATE_SHA256_458"]=EXPECTED_CANDIDATE_SHA
    os.environ["NUMI_FIXED_BONE_OUTWARD_AUDIT159_SHA256_458"]=EXPECTED_AUDIT159_SHA
    crossing=runner.prepare_crossing_inputs(base, audit)
    if int(crossing["receipt"]["complete_159_pair_count"]) != len(crossing["pairs"]):
        raise ValueError("prepared current crossing inventory is not the pinned 764 pairs")

    zbase=np.load(base,allow_pickle=False)
    try:
        points=np.asarray(zbase["points_world_m"],dtype=np.float32)
        faces=np.asarray(zbase["source_O_faces"],dtype=np.int64)
        targets_p,targets_f=owner.load_clearance_target_inputs(
            point_npz,point_report,face_npz,face_report,base,points,faces)
        with np.load(boundary_npz,allow_pickle=False) as z:
            fixed_arrays={k:np.asarray(z[k]).copy() for k in z.files}
        fixed_report=stable_json(boundary_report)
        fixed_records,_fixed_receipt=owner.validate_fixed_boundary_payload(
            fixed_arrays,fixed_report,candidate_sha256=sha(base),candidate_points=points,
            candidate_faces=faces,npz_path=boundary_npz)
        fixed_face_rows=np.asarray([int(r["skin_face_row"]) for r in fixed_records],dtype=np.int64)
        if "typed_coarse_anchor_output_rows_018" not in zbase.files:
            raise ValueError("candidate lacks typed fixed anchor rows")
        anchors=np.asarray(zbase["typed_coarse_anchor_output_rows_018"],dtype=np.int64)
        raw_identity=np.asarray(crossing["pair_identity_array"],dtype=np.int64)
        identity=np.asarray([[r["face_row"],r["body"],r["bone_id"],r["triangle_id"]] for r in crossing["pairs"]],dtype=np.int64).reshape(-1,4)
        if identity.shape != (len(crossing["pairs"]),4) or len(np.unique(identity,axis=0)) != len(identity):
            raise ValueError("current 159 pair identities are duplicated or malformed")
        if not np.array_equal(raw_identity,identity[:,[1,2,0,3]]):
            raise ValueError("458 raw identity columns are not body,bone,face,triangle")
        if not np.array_equal(points.astype(np.float64)*1000.0,crossing["points_mm"]):
            raise ValueError("458 crossing points do not match candidate F32 coordinates")
        if not np.array_equal(faces,crossing["faces"]):
            raise ValueError("458 crossing face order differs from candidate")
        pair_nodes=faces[identity[:,0]]
        if np.any(pair_nodes<0) or np.any(pair_nodes>=len(points)):
            raise ValueError("current crossing contains invalid face node IDs")
        raw_pairs=np.load(self_pairs_path,allow_pickle=False)
        has_targets=bool(len(targets_p["point_rows"]) or len(targets_f["face_rows"]) or len(fixed_records))
        pairs=owner.validate_self_pair_input(raw_pairs,len(faces),has_targets)
        core_nodes=np.unique(np.concatenate((
            faces[pairs].reshape(-1),
            targets_p["point_rows"],
            targets_f["face_nodes"].reshape(-1),
            faces[fixed_face_rows].reshape(-1),
        ))).astype(np.int64)
        core_faces=np.flatnonzero(np.isin(faces,core_nodes).any(axis=1)).astype(np.int64)
        initial_support=np.unique(faces[core_faces].reshape(-1)).astype(np.int64)
        if not len(initial_support): raise ValueError("empty initial support")
        _B, _controls, basis_report=owner.build_immediate_parent_basis(zbase,initial_support,anchors)
        support=np.asarray(basis_report["output_scope_rows"],dtype=np.int64)
        if support.ndim!=1 or len(np.unique(support))!=len(support):
            raise ValueError("408 returned malformed ordered scope")
        if np.any(~np.isin(pair_nodes.reshape(-1),support)):
            raise ValueError("crossing nodes are absent from exact 603 408-basis support")
        source_ref_m,source_receipt=owner.compose_source_reference(zbase,source)
        points_mm=points.astype(np.float64)*1000.0
        source_mm=np.asarray(source_ref_m,dtype=np.float64)*1000.0
        lo408,hi408,old_box_receipt=owner.source_safe_trust_bounds(
            points_mm,source_mm,support,anchors,4.0,12.0)
        lo,hi,bounds_receipt=box.directional_bounds(
            points_mm,source_mm,support,anchors,4.0,12.0)
        if lo.shape != hi.shape or lo.shape != lo408.shape:
            raise ValueError("636 bounds and 408 support domains differ")
        if np.any(lo > lo408+2e-10) or np.any(hi < hi408-2e-10):
            raise ValueError("636 directional bounds do not contain the 408 box")
        anchor_mask=np.isin(support,anchors)
        if np.any(np.abs(lo.reshape(-1,3)[anchor_mask]) > 1e-10) or np.any(np.abs(hi.reshape(-1,3)[anchor_mask]) > 1e-10):
            raise ValueError("636 anchor bounds are not fixed at zero")
        scope_hash=array_sha(support,"<i8")
        lo_hash=array_sha(lo,"<f8")
        hi_hash=array_sha(hi,"<f8")
        ppos={int(node):i for i,node in enumerate(support)}
        lb=lo.reshape(-1,3);ub=hi.reshape(-1,3)
        lower_by_node={int(node):lb[i].copy() for i,node in enumerate(support)}
        upper_by_node={int(node):ub[i].copy() for i,node in enumerate(support)}
        options=[]
        axis_total=[];axis_feasible=[];worst_unmet=[]
        for pi,(ident,nodes) in enumerate(zip(identity,pair_nodes)):
            skin=points_mm[nodes]
            body,bone,face,tri=map(int,crossing["pair_identity_array"][pi])
            if face!=int(ident[0]): raise ValueError("crossing identity order mismatch")
            bone_tri=np.asarray(crossing["bone_triangles_mm"][(body,bone,tri)],dtype=np.float64)
            axes=h._signed_sat_axes(skin,bone_tri)
            axis_total.append(len(axes))
            pi_options=[];best_unmet=np.inf
            for a in axes:
                n=np.asarray(a["normal_world_unit"],dtype=np.float64)
                if n.shape!=(3,) or not np.isfinite(n).all() or abs(np.linalg.norm(n)-1.0)>2e-8:
                    raise ValueError("551 returned a malformed SAT axis")
                q=float(np.max(bone_tri@n))
                margin=float(h._float32_world_m_emission_margin_mm(skin,bone_tri,n))
                rhs=q+margin-skin@n
                maxdot=np.asarray([
                    np.sum(np.where(n>=0.0,n*upper_by_node[int(node)],n*lower_by_node[int(node)]))
                    for node in nodes
                ],dtype=np.float64)
                unmet=np.maximum(rhs-maxdot,0.0)
                best_unmet=min(best_unmet,float(unmet.max(initial=0.0)))
                if float(unmet.max(initial=0.0))<=FEASIBILITY_TOL_MM:
                    pi_options.append(selector.AxisOption(
                        pi,n.copy(),rhs.copy(),str(a["axis_source"]),int(a["axis_index"]),
                        int(a["axis_sign"]),int(a.get("axis_source_order",0)),
                        int(a.get("axis_sign_order",0)),int(a.get("extra_sign",1)),
                        float(unmet.max(initial=0.0))))
            options.append(pi_options)
            axis_feasible.append(len(pi_options))
            worst_unmet.append(float(best_unmet))
        return {
            "candidate_path":str(base),"candidate_sha256":sha(base),
            "source_path":str(source),"source_sha256":sha(source),
            "audit159_path":str(audit),"audit159_sha256":sha(audit),
            "point_target_npz_path":str(point_npz),"point_target_npz_sha256":sha(point_npz),
            "face_target_npz_path":str(face_npz),"face_target_npz_sha256":sha(face_npz),
            "boundary_npz_path":str(boundary_npz),"boundary_npz_sha256":sha(boundary_npz),
            "boundary_report_path":str(boundary_report),"boundary_report_sha256":sha(boundary_report),
            "self_pairs_path":str(self_pairs_path),"self_pairs_sha256":sha(self_pairs_path),
            "point_target_report_path":str(point_report),"point_target_report_sha256":sha(point_report),
            "face_target_report_path":str(face_report),"face_target_report_sha256":sha(face_report),
            "current_inputs_path":str(CURRENT_INPUTS),"current_inputs_sha256":sha(CURRENT_INPUTS),
            "fit_argv_path":str(FIT_ARGV),"fit_argv_sha256":sha(FIT_ARGV),
            "selected_state_path":str(state_path),"selected_state_sha256":sha(state_path),
            "pair_identity":identity.copy(),"pair_nodes":pair_nodes.copy(),
            "options":options,"axis_total":np.asarray(axis_total,dtype=np.int64),
            "axis_feasible":np.asarray(axis_feasible,dtype=np.int64),
            "best_axis_unmet_mm":np.asarray(worst_unmet,dtype=np.float64),
            "lower_by_node":lower_by_node,"upper_by_node":upper_by_node,
            "support_rows":support.copy(),"anchors":anchors.copy(),
            "bounds_receipt":bounds_receipt,"old_408_box_receipt":old_box_receipt,
            "source_composition_receipt":source_receipt,
            "scope_sha256":scope_hash,"lower_sha256":lo_hash,"upper_sha256":hi_hash,
            "crossing_receipt":crossing["receipt"],
            "runner_sources":{
                str(SAT_PATH):sha(SAT_PATH),str(BOX_PATH):sha(BOX_PATH),
                str(OWNER408_PATH):sha(OWNER408_PATH),str(RUNNER458_PATH):sha(RUNNER458_PATH),
                str(HELPER638_PATH):sha(HELPER638_PATH),
                str(Path(__file__).resolve()):sha(Path(__file__).resolve()),
            },
        }
    finally:
        zbase.close()


def build_preflight(data):
    ids=data["pair_identity"]
    nodes=data["pair_nodes"]
    comps=selector.partition_pair_components(nodes)
    records=[]
    total_bin=0;total_rows=0
    for ci,idx in enumerate(comps):
        pcount=len(idx)
        binaries=int(data["axis_feasible"][idx].sum())
        rows=binaries*3+int(np.count_nonzero(data["axis_feasible"][idx]))
        variables=3*len(np.unique(nodes[idx]))+binaries
        total_bin+=binaries;total_rows+=rows
        records.append({
            "component_index":ci,"pair_count":pcount,
            "min_pair_identity":ids[int(idx.min())].tolist(),
            "pair_indices":idx.tolist(),"unique_skin_nodes":int(len(np.unique(nodes[idx]))),
            "feasible_axis_binary_count":binaries,"estimated_constraints":rows,
            "estimated_variables":variables,"max_candidate_axes_per_pair":int(data["axis_total"][idx].max(initial=0)),
            "individually_axis_feasible_pair_count":int(np.count_nonzero(data["axis_feasible"][idx])),
            "pair_count_with_no_box_feasible_axis":int(np.count_nonzero(data["axis_feasible"][idx]==0)),
        })
    records.sort(key=lambda r:(-r["pair_count"],-r["feasible_axis_binary_count"],tuple(r["min_pair_identity"])))
    over=[r for r in records if r["feasible_axis_binary_count"]>MAX_COMPONENT_BINARY or r["estimated_constraints"]>MAX_COMPONENT_ROWS]
    return {
        "schema":"numi.connected_patch_sat_selector638.preflight.v1",
        "pair_count":int(len(ids)),"unique_skin_node_count":int(len(np.unique(nodes))),
        "component_count":len(records),"candidate_axis_count":int(data["axis_total"].sum()),
        "individually_box_feasible_axis_count":int(data["axis_feasible"].sum()),
        "pairs_with_no_box_feasible_axis":int(np.count_nonzero(data["axis_feasible"]==0)),
        "max_best_axis_unmet_mm":float(data["best_axis_unmet_mm"].max(initial=0.0)),
        "component_binary_count_total":total_bin,"estimated_constraint_count_total":total_rows,
        "box_infeasible_pair_identities":[{"identity":ids[i].tolist(),"candidate_axis_count":int(data["axis_total"][i]),
            "feasible_axis_count":int(data["axis_feasible"][i]),"least_axis_unmet_mm":float(data["best_axis_unmet_mm"][i])}
            for i in np.flatnonzero(data["axis_feasible"]==0)],
        "components_over_hard_capacity":over,"components":records,
        "limits":{
            "max_total_seconds":MAX_TOTAL_SECONDS,"max_component_seconds":MAX_COMPONENT_SECONDS,
            "max_node_limit":MAX_NODE_LIMIT,"max_total_binary":MAX_TOTAL_BINARY,
            "max_component_binary":MAX_COMPONENT_BINARY,"max_component_constraints":MAX_COMPONENT_ROWS,
            "max_total_constraints":MAX_TOTAL_ROWS,
            "max_pairs":MAX_PAIRS,
            "hard_wall_cap_total_seconds":MAX_TOTAL_SECONDS,
            "hard_wall_cap_per_component_seconds":MAX_COMPONENT_SECONDS,
            "max_solver_nodes":MAX_NODE_LIMIT,
        },
        "inputs":{
            "candidate":{"path":data["candidate_path"],"sha256":data["candidate_sha256"]},
            "source_base":{"path":data["source_path"],"sha256":data["source_sha256"]},
            "complete_159_report":{"path":data["audit159_path"],"sha256":data["audit159_sha256"]},
            "current_crossing_pairs":data["crossing_receipt"],
            "point_target_npz":{"path":data["point_target_npz_path"],"sha256":data["point_target_npz_sha256"]},
            "face_target_npz":{"path":data["face_target_npz_path"],"sha256":data["face_target_npz_sha256"]},
            "point_target_report":{"path":data["point_target_report_path"],"sha256":data["point_target_report_sha256"]},
            "face_target_report":{"path":data["face_target_report_path"],"sha256":data["face_target_report_sha256"]},
            "boundary_npz":{"path":data["boundary_npz_path"],"sha256":data["boundary_npz_sha256"]},
            "boundary_report":{"path":data["boundary_report_path"],"sha256":data["boundary_report_sha256"]},
            "self_pair_seeds":{"path":data["self_pairs_path"],"sha256":data["self_pairs_sha256"]},
            "current_inputs":{"path":data["current_inputs_path"],"sha256":data["current_inputs_sha256"]},
            "fit_argv":{"path":data["fit_argv_path"],"sha256":data["fit_argv_sha256"]},
            "selected_state":{"path":data["selected_state_path"],"sha256":data["selected_state_sha256"]},
            "basis_scope_rows_count":int(len(data["support_rows"])),
            "basis_scope_rows_sha256":data["scope_sha256"],
            "directional636_lower_sha256":data["lower_sha256"],
            "directional636_upper_sha256":data["upper_sha256"],
            "bounds636":data["bounds_receipt"],
            "old408_box":data["old_408_box_receipt"],
            "source_composition":data["source_composition_receipt"],
            "sources":data["runner_sources"],
        },
        "claim_limit":"Box-only simultaneous finite-pair projection feasibility. No area, edge, fixed-boundary, self-intersection, containment, full-185-bank, or physical acceptance certification.",
        "native_admission":False,"accepted_physical_steps":0,
    }


def solve_all(data, preflight, out, seconds_total, seconds_component, node_limit):
    if not (0 < seconds_total <= MAX_TOTAL_SECONDS and 0 < seconds_component <= MAX_COMPONENT_SECONDS and 0 < node_limit <= MAX_NODE_LIMIT):
        raise ValueError("requested solve limits exceed hard selector caps")
    if len(data["pair_identity"])>MAX_PAIRS: raise ValueError("pair inventory exceeds hard cap")
    if preflight["component_binary_count_total"]>MAX_TOTAL_BINARY:
        raise ValueError("estimated binary count exceeds hard selector cap; no solve started")
    if preflight["estimated_constraint_count_total"]>MAX_TOTAL_ROWS:
        raise ValueError("estimated row count exceeds hard selector cap; no solve started")
    if preflight["components_over_hard_capacity"]:
        raise ValueError("one or more components exceed hard row/variable caps; no solve started")
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+seconds_total
    progress=out/"components.jsonl"
    selected_flags=np.zeros(len(data["pair_identity"]),dtype=bool)
    selected_normals=np.zeros((len(selected_flags),3),dtype=np.float64)
    selected_source=np.full(len(selected_flags),"",dtype="U32")
    selected_axis=np.full(len(selected_flags),-1,dtype=np.int64)
    selected_sign=np.zeros(len(selected_flags),dtype=np.int64)
    selected_source_order=np.full(len(selected_flags),-1,dtype=np.int64)
    selected_sign_order=np.full(len(selected_flags),-1,dtype=np.int64)
    selected_extra_sign=np.zeros(len(selected_flags),dtype=np.int64)
    disp_rows=[];disp_values=[];component_reports=[]
    all_m_rows=[]
    for rec in preflight["components"]:
        ci=int(rec["component_index"]);idx=np.asarray(rec["pair_indices"],dtype=np.int64)
        remain=deadline-time.monotonic()
        if remain<=0:
            component_reports.append({"component_index":ci,"status":"SKIPPED_TOTAL_TIME_CAP","pair_count":len(idx)})
            continue
        model=selector.build_component_milp(idx,data["pair_nodes"],data["pair_identity"],data["options"],
            data["lower_by_node"],data["upper_by_node"],data["anchors"])
        budget=min(seconds_component,remain)
        if model["binary_variable_count"]>MAX_COMPONENT_BINARY or model["constraint_count"]>MAX_COMPONENT_ROWS:
            component_reports.append({"component_index":ci,"status":"SKIPPED_COMPONENT_CAP","pair_count":len(idx),
                "binary_variable_count":model["binary_variable_count"],"constraint_count":model["constraint_count"]})
            continue
        result=selector.solve_component(model,budget,node_limit)
        got=[]
        if result["has_incumbent"]:
            for record in result["selected"]:
                pi=int(record["pair_index"]);op=record["axis"]
                selected_flags[pi]=True;selected_normals[pi]=op.normal
                selected_source[pi]=op.source;selected_axis[pi]=op.axis_index;selected_sign[pi]=op.axis_sign
                selected_source_order[pi]=op.source_order;selected_sign_order[pi]=op.sign_order;selected_extra_sign[pi]=op.extra_sign
                got.append(pi)
            for local,node in enumerate(model["nodes"]):
                disp_rows.append(int(node));disp_values.append(result["node_displacements_mm"][local].copy())
        all_m_rows.extend(model["big_m_receipts"])
        comp_report={"component_index":ci,"status":result["status"],"message":result["message"],
            "has_incumbent":result["has_incumbent"],"selected_pair_count":len(got),
            "selected_pair_indices":got,"pair_count":len(idx),"binary_variable_count":model["binary_variable_count"],
            "coordinate_variable_count":model["coordinate_variable_count"],"constraint_count":model["constraint_count"],
            "time_limit_s":budget,"elapsed_s":result["elapsed_s"],"mip_gap":result["mip_gap"],
            "mip_node_count":result["mip_node_count"],"objective":result.get("objective")}
        component_reports.append(comp_report)
        with progress.open("a") as f:f.write(json.dumps(comp_report,sort_keys=True,allow_nan=False)+"\n")
    np.savez_compressed(out/"selection.npz",
        pair_identity=data["pair_identity"],pair_nodes=data["pair_nodes"],selected=selected_flags,
        selected_normal_world_unit=selected_normals,selected_axis_source=selected_source,
        selected_axis_index=selected_axis,selected_axis_sign=selected_sign,
        selected_axis_source_order=selected_source_order,selected_axis_sign_order=selected_sign_order,
        selected_axis_extra_sign=selected_extra_sign,
        displacement_node_rows=np.asarray(disp_rows,dtype=np.int64),
        displacement_mm=np.asarray(disp_values,dtype=np.float64).reshape(-1,3))
    if all_m_rows:
        np.savez_compressed(out/"row-specific-big-m.npz",
            pair_identity=np.asarray([r["pair_identity"] for r in all_m_rows],dtype=np.int64),
            corner=np.asarray([r["corner"] for r in all_m_rows],dtype=np.int8),
            node_row=np.asarray([r["node"] for r in all_m_rows],dtype=np.int64),
            rhs_mm=np.asarray([r["rhs_mm"] for r in all_m_rows],dtype=np.float64),
            box_min_dot_mm=np.asarray([r["box_min_dot_mm"] for r in all_m_rows],dtype=np.float64),
            big_m_mm=np.asarray([r["big_m_mm"] for r in all_m_rows],dtype=np.float64),
            axis_source=np.asarray([r["axis_source"] for r in all_m_rows],dtype="U32"),
            axis_index=np.asarray([r["axis_index"] for r in all_m_rows],dtype=np.int64),
            axis_sign=np.asarray([r["axis_sign"] for r in all_m_rows],dtype=np.int64),
            normal_world_unit=np.asarray([r["normal"] for r in all_m_rows],dtype=np.float64).reshape(-1,3))
    report={
        "schema":"numi.connected_patch_sat_selector638.result.v1",
        "preflight":preflight,"component_results":component_reports,
        "selected_pair_count":int(selected_flags.sum()),
        "all_components_proven_optimal":bool(all(r.get("status")==0 for r in component_reports if r.get("has_incumbent")) and
            all(r.get("has_incumbent") for r in component_reports if r.get("status") not in ("SKIPPED_TOTAL_TIME_CAP","SKIPPED_COMPONENT_CAP")) and
            not any(isinstance(r.get("status"),str) and r["status"].startswith("SKIPPED_") for r in component_reports)),
        "elapsed_total_s":time.monotonic()-started,
        "selection_npz":"selection.npz",
        "selection_npz_sha256":sha(out/"selection.npz"),
        "row_specific_big_m_npz":"row-specific-big-m.npz" if all_m_rows else None,
        "row_specific_big_m_npz_sha256":sha(out/"row-specific-big-m.npz") if all_m_rows else None,
        "claim_limit":preflight["claim_limit"],
        "native_admission":False,"accepted_physical_steps":0,
    }
    (out/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prepared-dir",type=Path,default=DEFAULT_PREPARED_DIR,help="prepared directory containing fit-argv.json")
    parser.add_argument("--current-inputs-json",type=Path,default=None,help="defaults to prepared-dir parent/current-inputs.json")
    parser.add_argument("--state-path",type=Path,default=None,help="optional byte-identical selected state copy")
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--preflight-only",action="store_true")
    group.add_argument("--solve",action="store_true",help="run bounded per-component MILPs")
    parser.add_argument("--max-total-seconds",type=float,default=45.0)
    parser.add_argument("--component-seconds",type=float,default=2.5)
    parser.add_argument("--node-limit",type=int,default=10000)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(f"output already exists: {args.output}")
    configure_preparation(args.prepared_dir,args.current_inputs_json,args.state_path)
    data=prepare_current_inputs()
    preflight=build_preflight(data)
    if args.preflight_only or not args.solve:
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/"preflight.json").write_text(json.dumps(preflight,indent=2,sort_keys=True,allow_nan=False)+"\n")
        print(json.dumps({"status":"PREFLIGHT_ONLY","output":str(args.output),"pairs":preflight["pair_count"],
             "components":preflight["component_count"],"candidate_axes":preflight["candidate_axis_count"],
             "feasible_axes":preflight["individually_box_feasible_axis_count"],
             "estimated_binaries":preflight["component_binary_count_total"],
             "over_capacity_components":len(preflight["components_over_hard_capacity"]),
             "claim_limit":preflight["claim_limit"]},sort_keys=True))
        return
    result=solve_all(data,preflight,args.output,args.max_total_seconds,args.component_seconds,args.node_limit)
    print(json.dumps({"status":"SELECTOR_COMPLETE","output":str(args.output),
        "selected_pair_count":result["selected_pair_count"],
        "all_components_proven_optimal":result["all_components_proven_optimal"],
        "elapsed_total_s":result["elapsed_total_s"],"claim_limit":result["claim_limit"]},sort_keys=True))


if __name__=="__main__":
    import connected_patch_sat_selector638 as selector
    main()
