#!/usr/bin/env python3
"""Isolated joint fit preparation for the checked 585 affine midpoint pilot.

This is a temporary construction branch: it requires exact self0/full185 audits,
but permits contained midpoint rows only with their already candidate-matched
all185-certified point exits. The ordinary surface-only 546 gate is untouched.
"""
from __future__ import annotations
import argparse, importlib.util, json, shutil, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
D=Path("/Users/numi/Work/numi-human-connected-surface-069/D")
N=D/"surface-clearance-546"
PYTHON=Path("/Users/n/numi-human-prep-venv-20261005/bin/python")
STATE_DEFAULT=N/"edge-only-refinement-585/pilot-008/baseline-check/baseline-state.json"
PRIOR_DEFAULT=N/"connected-continuation-584/run-001/round-001/emitted-fit/prepare/constraint-context.json"
PARENT_DEFAULT=N/"connected-topology-580/run-003/lineage/candidate.npz"
POINT_NPZ_DEFAULT=N/"edge-only-refinement-585/pilot-008/union-targets/nearest-union-exits.npz"
POINT_REPORT_DEFAULT=POINT_NPZ_DEFAULT.with_suffix(".json")
DEPTH_NPZ_DEFAULT=N/"edge-only-refinement-585/pilot-008/depth/point-depth-rows.npz"
DEPTH_REPORT_DEFAULT=N/"edge-only-refinement-585/pilot-008/depth/365-compatible-report.json"
FIT_TEMPLATE_DEFAULT=N/"connected-continuation-584/run-001/round-001/emitted-fit/fit-argv.json"
MATCHED_TEMPLATE_DEFAULT=N/"connected-continuation-584/run-001/round-001/emitted-fit/matched-inputs.json"
ADAPTER=N/"refresh_surface_constraints546.py"
ADAPTER_SHA="a9586a81d92f117040e774c980b6028168432386b2e9babd96a7b7f725175d09"
TARGET_BUILDER=D/"source-outward-proposal-500/build_outward_targets_multi_source_546.py"
TARGET_BUILDER_SHA="dd61f0fb87f12c132166515a1e058e144d493a835cadbb2a47e6a2c8574e1e68"
PREP_HELPER=HERE/"joint_refined595.py"
WRAPPER_PATH=N/"partial-timeout-fallback564/frozen-v2/run_joint_full_coordinate_fit564.py"

def loadmod(name,path):
    sp=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--state",type=Path,default=STATE_DEFAULT);ap.add_argument("--prior-context",type=Path,default=PRIOR_DEFAULT)
    ap.add_argument("--ancestry-parent-candidate",type=Path,default=PARENT_DEFAULT)
    ap.add_argument("--point-target-npz",type=Path,default=POINT_NPZ_DEFAULT);ap.add_argument("--point-target-report",type=Path,default=POINT_REPORT_DEFAULT)
    ap.add_argument("--point-depth-rows-npz",type=Path,default=DEPTH_NPZ_DEFAULT);ap.add_argument("--point-depth-rows-report",type=Path,default=DEPTH_REPORT_DEFAULT)
    ap.add_argument("--fit-template",type=Path,default=FIT_TEMPLATE_DEFAULT);ap.add_argument("--matched-template",type=Path,default=MATCHED_TEMPLATE_DEFAULT)
    ap.add_argument("--out",type=Path,required=True);ap.add_argument("--trust-mm",type=float,default=4.0)
    ap.add_argument("--prepare",action="store_true",help="execute 546 topology refresh and unchanged multi-parent 500 face target builder")
    ap.add_argument("--run-fit",action="store_true",help="run pinned 564 joint fit after successful preparation")
    a=ap.parse_args()
    if a.run_fit and not a.prepare: raise ValueError("--run-fit requires --prepare")
    if a.out.exists(): raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    owner=loadmod("continue546_joint_refined595",N/"continue_surface_round546.py")
    h=loadmod("joint_refined595_helper",PREP_HELPER)
    st=owner.read_json(a.state);prior=owner.read_json(a.prior_context)
    base=Path(st["candidate"]); point_path=Path(st["point_summary"]); surface_path=Path(st["surface_report"])
    for key,path in [("candidate",base),("point_summary",point_path),("surface_report",surface_path)]:
        if owner.sha(path)!=st[key+"_sha256"]: raise ValueError(f"state hash mismatch for {key}")
    summary=owner.read_json(point_path);surface=owner.read_json(surface_path)
    with np.load(base,allow_pickle=False) as z: arrays={k:np.asarray(z[k]).copy() for k in z.files}
    with np.load(a.point_target_npz,allow_pickle=False) as z: point_targets={k:np.asarray(z[k]).copy() for k in z.files}
    point_report=owner.read_json(a.point_target_report)
    with np.load(a.point_depth_rows_npz,allow_pickle=False) as z: depth={k:np.asarray(z[k]).copy() for k in z.files}
    depth_report=owner.read_json(a.point_depth_rows_report)
    report=h.validate_joint_refined_inputs(state=st,point_summary=summary,surface_report=surface,arrays=arrays,
        candidate_sha256=owner.sha(base),point_target_report=point_report,point_targets=point_targets,
        point_target_sha256=owner.sha(a.point_target_npz),point_depth_report=depth_report,
        depth_npz=depth,depth_sha256=owner.sha(a.point_depth_rows_npz),prior_context=prior,
        prior_candidate_sha256=owner.sha(Path(prior["candidate"])),
        ancestry_parent_sha256=owner.sha(a.ancestry_parent_candidate))
    if report["ancestry_parent_candidate_sha256"]!=owner.sha(Path(prior["candidate"])):
        raise ValueError("ancestry parent differs from prior seed/guard context")
    query_dir=point_path.parent/"query"
    query_receipt=query_dir/"query-receipt.json"
    if not query_receipt.is_file(): raise FileNotFoundError(query_receipt)
    prior_surface=Path(prior.get("surface_report",""))
    if not prior_surface.is_file() or owner.sha(prior_surface)!=prior.get("surface_report_sha256"):
        raise ValueError("prior exact 159 context/report is missing or stale")
    if owner.sha(ADAPTER)!=ADAPTER_SHA: raise ValueError("pinned 546 topology refresh adapter changed")
    if owner.sha(TARGET_BUILDER)!=TARGET_BUILDER_SHA: raise ValueError("pinned 500 multi-parent target builder changed")
    owner.write_json(a.out/"preflight.json",{**report,"state_path":str(a.state),"state_sha256":owner.sha(a.state),
        "point_summary_path":str(point_path),"point_summary_sha256":owner.sha(point_path),
        "surface_report_path":str(surface_path),"surface_report_sha256":owner.sha(surface_path),
        "point_target_path":str(a.point_target_npz),"point_target_sha256":owner.sha(a.point_target_npz),
        "point_target_report_path":str(a.point_target_report),"point_target_report_sha256":owner.sha(a.point_target_report),
        "point_depth_rows_path":str(a.point_depth_rows_npz),"point_depth_rows_sha256":owner.sha(a.point_depth_rows_npz),
        "point_depth_report_path":str(a.point_depth_rows_report),"point_depth_report_sha256":owner.sha(a.point_depth_rows_report),
        "prior_context_path":str(a.prior_context),"prior_context_sha256":owner.sha(a.prior_context),
        "ancestry_parent_candidate":str(a.ancestry_parent_candidate),"ancestry_parent_sha256":owner.sha(a.ancestry_parent_candidate),
        "mode":"explicit_joint_refined_temporary_construction","geometry_or_native_admission":False})
    if not a.prepare:
        print(json.dumps({"status":"PREFLIGHT_PASS_JOINT_REFINED_UNSELECTED","out":str(a.out),"inside":report["inside_node_bank_pairs"]}),flush=True);return
    constraints=a.out/"constraints"
    refresh_argv=[PYTHON,ADAPTER,"--candidate",base,"--audit130",point_path,"--audit159",surface_path,
      "--query-dir",query_dir,"--ancestry-parent-candidate",a.ancestry_parent_candidate,
      "--prior-candidate",prior["candidate"],"--prior-surface-report",prior_surface,
      "--prior-guard-candidate",prior["candidate"],"--prior-guard-npz",prior["guard_npz"],
      "--prior-guard-report",prior["guard_report"],"--prior-seed-pairs",prior["seed_pairs_path"],
      "--prior-seed-report",prior["seed_report"],"--failed-seed-candidate",base,
      "--failed-seed-audit130",point_path,"--failed-seed-pairs",point_path.parent/"surface-self-pairs.npy",
      "--output-dir",constraints]
    srcs=(ADAPTER,N/"refresh_guards_multi_parent_546.py",base,point_path,surface_path,query_receipt,
          a.ancestry_parent_candidate,prior_surface,Path(prior["candidate"]),Path(prior["guard_npz"]),
          Path(prior["guard_report"]),Path(prior["seed_pairs_path"]),Path(prior["seed_report"]))
    owner.run_stage(refresh_argv,a.out/"refresh-stage",slot=True,source_paths=srcs,timeout=600)
    ctx=owner.read_json(constraints/"refresh-report.json")
    for key in ("seed_pairs_path","seed_report_path","fixed_guard_npz_path","fixed_guard_report_path"):
        if not Path(ctx[key]).is_file(): raise ValueError(f"refresh output missing {key}")
    targets_dir=a.out/"surface-targets"
    owner.run_stage([PYTHON,TARGET_BUILDER,"--candidate",base,"--surface-report",surface_path,
        "--query-receipt",query_receipt,"--output-dir",targets_dir,"--clearance-m",".00005",
        "--direction-mode","outward-nearest"],a.out/"surface-targets-stage",slot=True,
        source_paths=(TARGET_BUILDER,base,surface_path,query_receipt),timeout=600)
    # Revalidate the target builder output is bound to this complete all185 report.
    face_report=owner.read_json(targets_dir/"report.json")
    with np.load(targets_dir/"targets.npz",allow_pickle=False) as z: face_targets={k:np.asarray(z[k]) for k in z.files}
    if face_report.get("inputs",{}).get("candidate_sha256")!=owner.sha(base) or face_report.get("matched_159_complete") is not True:
        raise ValueError("500 face targets do not match current exact all185 report")
    if len(face_targets.get("face_rows",[]))!=len(face_targets.get("target_world_f32_m",[])):
        raise ValueError("500 face target arrays have inconsistent row count")
    matched=a.out/"matched-inputs.json"
    inputs={"candidate":base,"audit159":surface_path,"face_target_npz":targets_dir/"targets.npz","point_target_npz":a.point_target_npz}
    owner.matched_inputs_from_template(a.matched_template,matched,inputs)
    fit_template=json.loads(a.fit_template.read_text())
    fit_argv=h.fit_argv_from_template(fit_template,candidate=base,audit159=surface_path,
       point_target_npz=a.point_target_npz,point_target_report=a.point_target_report,
       face_target_npz=targets_dir/"targets.npz",face_target_report=targets_dir/"report.json",
       point_depth_npz=a.point_depth_rows_npz,point_depth_report=a.point_depth_rows_report,
       self_pairs=Path(ctx["seed_pairs_path"]),fixed_guard_npz=Path(ctx["fixed_guard_npz_path"]),
       fixed_guard_report=Path(ctx["fixed_guard_report_path"]),matched_inputs=matched,
       output=a.out/"fit",trust_mm=a.trust_mm)
    wrapper=Path(fit_argv[1])
    if wrapper!=WRAPPER_PATH: raise ValueError("fit template uses unexpected 564 wrapper")
    argv_path=a.out/"fit-argv.json"; owner.write_json(argv_path,fit_argv)
    prep={"status":"JOINT_REFINED_TARGETS_AND_GUARDS_PREPARED_UNADMITTED","baseline_state":str(a.state),
       "baseline_candidate_sha256":owner.sha(base),"inside_node_bank_pairs":report["inside_node_bank_pairs"],
       "self_pairs":0,"surface_target_rows":len(face_targets["face_rows"]),
       "point_target_rows":len(point_targets["query_rows"]),"point_targets_outside_all185":True,
       "seed_report":ctx["seed_report_path"],"guard_report":ctx["fixed_guard_report_path"],
       "matched_inputs":str(matched),"fit_argv":str(argv_path),"fit_wrapper":str(wrapper),
       "fit_wrapper_sha256":owner.sha(wrapper),"trust_mm":a.trust_mm,"source_cap_mm":12.0,
       "area_floor":0.1,"max_edge_gradient":0.5,"native_admission":False,"accepted_physical_steps":0}
    owner.write_json(a.out/"prepared.json",prep)
    if a.run_fit:
        source_paths=(wrapper,Path(fit_argv[1]),a.fit_template,a.matched_template,base,surface_path,
            a.point_target_npz,a.point_target_report,a.point_depth_rows_npz,a.point_depth_rows_report,
            targets_dir/"targets.npz",targets_dir/"report.json",Path(ctx["seed_pairs_path"]),
            Path(ctx["seed_report_path"]),Path(ctx["fixed_guard_npz_path"]),Path(ctx["fixed_guard_report_path"]))
        owner.run_stage(fit_argv,a.out/"fit-stage",slot=True,source_paths=source_paths,timeout=600)
        print(json.dumps({"status":"FIT_STAGE_FINISHED_UNADMITTED","out":str(a.out),"prepared":prep}),flush=True)
    else: print(json.dumps({"status":prep["status"],"out":str(a.out),"prepared":prep}),flush=True)

if __name__=="__main__": main()

