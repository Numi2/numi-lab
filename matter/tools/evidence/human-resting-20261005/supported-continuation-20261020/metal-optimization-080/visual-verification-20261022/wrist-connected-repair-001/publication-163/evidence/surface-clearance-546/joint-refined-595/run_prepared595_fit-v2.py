#!/usr/bin/env python3
"""Run only the already prepared 595 fit under the shared geometry child slot."""
from __future__ import annotations
import importlib.util, json, shutil, sys
from pathlib import Path

D=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546")
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D")
P=D/"joint-refined-595/run-002"
RECEIPT=P/"preparation-receipt.json"
STAGE=P/"fit-stage"
MIN_FREE=256*1024*1024

def loadmod(name,path):
    sp=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    if STAGE.with_suffix(".launch.json").exists() or (P/"fit").exists():
        raise FileExistsError("595 fit output already exists; preserving existing attempt")
    rec=json.loads(RECEIPT.read_text())
    if rec.get("fit_launched") is not False or rec.get("physical_steps") != 0:
        raise ValueError("prepared receipt is not an unlaunched zero-step proposal")
    for role,item in rec["files"].items():
        path=Path(item["path"])
        if not path.is_file() or digest(path)!=item["sha256"]:
            raise ValueError(f"595 preparation input/source changed: {role}")
    prepared=json.loads((P/"prepared.json").read_text())
    argv=json.loads((P/"fit-argv.json").read_text())
    state_path=Path(prepared["baseline_state"])
    state=json.loads(state_path.read_text())
    point_summary=Path(state["point_summary"])
    surface_report=Path(state["surface_report"])
    if digest(point_summary)!=state["point_summary_sha256"] or digest(surface_report)!=state["surface_report_sha256"]:
        raise ValueError("baseline audit reports changed after preparation")
    if prepared.get("status")!="JOINT_REFINED_TARGETS_AND_GUARDS_PREPARED_UNADMITTED":
        raise ValueError("595 prepared status changed")
    if prepared.get("fit_wrapper_sha256") != digest(Path(argv[1])):
        raise ValueError("564 fit wrapper pin differs from prepared report")
    for flag,expected in (("--trust-mm","4.0"),("--source-cap-mm","12.0"),("--area-floor","0.1"),("--max-edge-gradient","0.5")):
        if float(argv[argv.index(flag)+1]) != float(expected):
            raise ValueError(f"unexpected fit policy for {flag}")
    if "--immediate-parent-basis" not in argv or "--allow-soft-target-regression-485" not in argv:
        raise ValueError("joint fit lost the prepared basis/objective contract")
    free=shutil.disk_usage(D).free
    if free < MIN_FREE:
        raise RuntimeError(f"disk reserve below 256 MiB; have {free//(1024*1024)} MiB")
    owner_path=D/"continue_surface_round546.py"
    owner=loadmod("continue_surface_round546_fit595",owner_path)
    sources=[
        owner_path,
        D/"joint-refined-595/prepare_joint_refined595-v2.py",
        D/"joint-refined-595/joint_refined595.py",
        D/"joint-refined-595/make_mixed_targets595.py",
        Path(argv[1]),
        D/"connected-continuation-584/run-001/round-001/emitted-fit/fit-argv.json",
        D/"connected-continuation-584/run-001/round-001/emitted-fit/matched-inputs.json",
        state_path, point_summary, surface_report,
        Path(rec["files"]["baseline_state"]["path"]),
        Path(rec["files"]["candidate"]["path"]),
        Path(rec["files"]["prior_context"]["path"]),
        Path(rec["files"]["ancestry_parent_candidate"]["path"]),
        Path(rec["files"]["surface_targets_npz"]["path"]), Path(rec["files"]["surface_targets_report"]["path"]),
        Path(rec["files"]["mixed_point_target_npz"]["path"]), Path(rec["files"]["mixed_point_target_report"]["path"]),
        Path(rec["files"]["point_depth_npz"]["path"]), Path(rec["files"]["point_depth_report"]["path"]),
        Path(rec["files"]["seed_pairs"]["path"]), Path(rec["files"]["seed_report"]["path"]),
        Path(rec["files"]["guard_npz"]["path"]), Path(rec["files"]["guard_report"]["path"]),
        Path(rec["files"]["constraint_context"]["path"]), Path(rec["files"]["matched_inputs"]["path"]),
        Path(rec["files"]["fit_argv"]["path"]), Path(rec["files"]["fit_wrapper"]["path"]),
        Path(rec["files"]["refresh_adapter"]["path"]), Path(rec["files"]["refresh_multiparent_shim"]["path"]),
        Path(rec["files"]["surface_target_builder"]["path"]),
        D/"source-outward-targets-589/pilot-008/nearest-union-exits.npz",
        D/"source-outward-targets-589/pilot-008/nearest-union-exits.json",
        D/"source-outward-targets-589/depth-maximizing-pilot-008/nearest-union-exits.npz",
        D/"source-outward-targets-589/depth-maximizing-pilot-008/nearest-union-exits.json",
        D/"mixed-target-591/emitted-001/candidate.npz",
        D/"mixed-target-591/emitted-001/report.json",
    ]
    sources=list(dict.fromkeys(sources))
    owner.run_stage(argv,STAGE,slot=True,source_paths=sources,timeout=600)
    after=shutil.disk_usage(D).free
    result={"status":"FIT_STAGE_FINISHED_UNADMITTED","fit_argv":str(P/"fit-argv.json"),
        "fit_argv_sha256":digest(P/"fit-argv.json"),"fit_output":argv[argv.index("--output")+1],
        "disk_free_before_bytes":free,"disk_free_after_bytes":after,"fit_stage":str(STAGE),
        "fit_stage_log":str(STAGE.with_suffix(".log")),"native_admission":False,"accepted_physical_steps":0}
    out=P/"fit-launch-receipt.json";out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__": main()

