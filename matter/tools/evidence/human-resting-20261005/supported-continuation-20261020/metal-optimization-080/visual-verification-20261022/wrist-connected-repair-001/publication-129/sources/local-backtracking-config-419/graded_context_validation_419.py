"""Strict external-context validation for the 423 reducer under 419 dispatch."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
GRADED_HELPER=ROOT/"D/graded-local-step-423/graded_local_step_423.py"
GRADED_SHA256="72cb043f1849a66f6829b1767717d574d0e25fd286bf40cfeab1c96a9cab3b8f"
GRADED_ENTRY="reduce_witness_node_scales"

def sha256_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_graded_context_419(config):
    if config.get("uniform_failure_backtracking") is not False:
        raise ValueError("423 graded local mode requires explicit false uniform flag")
    if config.get("helper")!=str(GRADED_HELPER) or config.get("helper_sha256")!=GRADED_SHA256 or config.get("helper_entry")!=GRADED_ENTRY:
        raise ValueError("423 helper source/path/entry pin mismatch")
    if not GRADED_HELPER.is_file() or sha256_file(GRADED_HELPER)!=GRADED_SHA256:
        raise ValueError("423 helper file hash mismatch")
    required=("local_step_context_npz","local_step_context_sha256","local_step_context_report",
              "local_step_context_report_sha256","local_step_context_kappa")
    if any(k not in config for k in required):
        raise ValueError("423 mode requires explicit context NPZ/report path and SHA pins plus kappa")
    context=Path(config["local_step_context_npz"]);report_path=Path(config["local_step_context_report"])
    if not context.is_file() or sha256_file(context)!=config["local_step_context_sha256"]:
        raise ValueError("423 context NPZ path/hash mismatch")
    if not report_path.is_file() or sha256_file(report_path)!=config["local_step_context_report_sha256"]:
        raise ValueError("423 context report path/hash mismatch")
    report=json.loads(report_path.read_text())
    kappa=float(config["local_step_context_kappa"])
    if not np.isfinite(kappa) or not 0<kappa<=1 or report.get("kappa")!=kappa:
        raise ValueError("423 context kappa does not match its pinned report")
    state=config.get("state")
    if not isinstance(state,dict) or not isinstance(state.get("candidate"),str) or not isinstance(config.get("trial"),str):
        raise ValueError("423 mode requires a pinned base candidate and trial")
    base=Path(state["candidate"]);trial=Path(config["trial"])
    if not base.is_file() or not trial.is_file():raise ValueError("423 base candidate or trial file is missing")
    base_sha=sha256_file(base);trial_sha=sha256_file(trial)
    if report.get("base_candidate")!=str(base) or report.get("base_sha256")!=base_sha:
        raise ValueError("423 context base candidate binding mismatch")
    if report.get("trial")!=str(trial) or report.get("trial_sha256")!=trial_sha:
        raise ValueError("423 context trial binding mismatch")
    if report.get("context")!=str(context) or report.get("context_sha256")!=config["local_step_context_sha256"]:
        raise ValueError("423 context report payload binding mismatch")
    with np.load(base,allow_pickle=False) as b, np.load(trial,allow_pickle=False) as t, np.load(context,allow_pickle=False) as c:
        required_arrays={"faces","edges","edge_scale_budget","points_baseline_m","proposal_displacement_m","anchors"}
        if not required_arrays.issubset(c.files):raise ValueError("423 context NPZ missing required arrays")
        f=np.asarray(b["source_O_faces"]);p=np.asarray(b["points_world_m"],dtype=np.float64);q=np.asarray(t["points_world_m"],dtype=np.float64)
        anchors=np.asarray(b["typed_coarse_anchor_output_rows_018"],dtype=np.int64)
        cf=np.asarray(c["faces"]);cp=np.asarray(c["points_baseline_m"]);cd=np.asarray(c["proposal_displacement_m"])
        ca=np.asarray(c["anchors"],dtype=np.int64);edges=np.asarray(c["edges"],dtype=np.int64)
        if not np.array_equal(cf,f) or not np.array_equal(cp,p):raise ValueError("423 context topology/base points do not match source candidate")
        if not np.array_equal(cd,q-p):raise ValueError("423 context proposal displacement does not match trial")
        if not np.array_equal(np.sort(ca),np.sort(anchors)):raise ValueError("423 context anchors do not match source candidate")
        e=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[0,2]]),axis=0).astype(np.int64);e.sort(axis=1);expected_edges=np.unique(e,axis=0)
        if not np.array_equal(edges,expected_edges):raise ValueError("423 context edges do not match source topology")
        lengths=np.linalg.norm(p[edges[:,0]]-p[edges[:,1]],axis=1)
        delta=q-p;amplitudes=np.maximum(np.linalg.norm(delta[edges[:,0]],axis=1),np.linalg.norm(delta[edges[:,1]],axis=1))
        expected=np.ones(len(edges),dtype=np.float64);moving=amplitudes>0
        expected[moving]=np.minimum(1.0,kappa*lengths[moving]/amplitudes[moving])
        budgets=np.asarray(c["edge_scale_budget"],dtype=np.float64)
        if budgets.shape!=expected.shape or not np.allclose(budgets,expected,rtol=2e-12,atol=2e-15):
            raise ValueError("423 edge scale budget is not derived from pinned geometry")
    return {"mode":"graded_local_423","helper_path":str(GRADED_HELPER),"helper_sha256":GRADED_SHA256,
            "helper_entry":GRADED_ENTRY,"context_npz":str(context),"context_sha256":config["local_step_context_sha256"],
            "context_report":str(report_path),"context_report_sha256":config["local_step_context_report_sha256"],
            "context_kappa":kappa,"base_candidate_sha256":base_sha,"trial_sha256":trial_sha,
            "geometry_certified":False,"native_admission":False,"accepted_physical_steps":0}
