"""Receipt-first coordinator for the matched 2102 topology-025 v5 chain."""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
BASE = ROOT / "D/surface-clearance-546/topology/round-037-v3-from2102"
STATE = ROOT / "D/surface-clearance-546/local-witness-retention-567/retained-552-001/final/state.json"
EXPECTED_STATE_SHA = "d1a83165e2a069b5f53e2ef89bd52b54b02731eeb0f164cc7a1dcabf461076a4"
RUNNER = BASE / "sources-v5/run_quality_policy025_variant_v5.py"
PYTHON = Path("/Users/n/numi-human-prep-venv-20261005/bin/python")
SLOT = Path("/tmp/numi-human-geometry-slot.py")
OUT = BASE / "run-005-v5"
STAGE = BASE / "run-005-v5-stage"

def loadmod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module); return module

def main():
    owner = loadmod("continue_surface_round546_for_topology037_v5", ROOT / "D/surface-clearance-546/continue_surface_round546.py")
    if owner.sha(STATE) != EXPECTED_STATE_SHA: raise ValueError("2102 state file changed")
    state = owner.read_json(STATE)
    candidate = Path(state["candidate"]).resolve(); audit130 = Path(state["point_summary"]).resolve(); audit159 = Path(state["surface_report"]).resolve()
    self_pairs = audit130.parent / "surface-self-pairs.npy"
    for path, key in ((candidate,"candidate_sha256"),(audit130,"point_summary_sha256"),(audit159,"surface_report_sha256")):
        if owner.sha(path) != state[key]: raise ValueError(f"state-pinned input hash mismatch: {path}")
    pairs = np.load(self_pairs,allow_pickle=False)
    if pairs.shape != (0,2) or not np.issubdtype(pairs.dtype,np.integer): raise ValueError("input is not a zero-self state")
    summary = owner.read_json(audit130); surface = owner.read_json(audit159)
    if summary.get("candidate_sha256") != state["candidate_sha256"] or summary.get("complete_surface_audit") is not True or summary.get("exact_PLC_surface_pairs") != 0:
        raise ValueError("point audit does not certify the matched zero-self candidate")
    if surface.get("candidate_sha256") != state["candidate_sha256"] or surface.get("complete") is not True or surface.get("expected_bank_count") != 185 or surface.get("bank_count") != 185:
        raise ValueError("surface audit is not complete and matched to all 185 banks")
    if OUT.exists(): raise FileExistsError(f"refusing existing chain output root: {OUT}")
    stage_paths=(STAGE.with_suffix(".launch.json"),STAGE.with_suffix(".execution.json"),STAGE.with_suffix(".log"))
    if any(p.exists() for p in stage_paths): raise FileExistsError(f"stage receipt already exists: {STAGE}")
    argv=[PYTHON,RUNNER,"--chain","--candidate",candidate,"--audit159",audit159,"--audit130",audit130,"--self-pairs130",self_pairs,
          "--output-root",OUT,"--expected-candidate-sha256",state["candidate_sha256"],"--expected-audit159-sha256",owner.sha(audit159),
          "--expected-audit130-sha256",owner.sha(audit130),"--expected-self-pairs130-sha256",owner.sha(self_pairs),
          "--rounds","3","--max-candidates","2048","--max-accept","64","--prefilter-factor","1"]
    sources=(Path(__file__),RUNNER,RUNNER.parent/"quality_policy025_variant.py",ROOT/"D/surface-clearance-546/topology/chain_exact_flip_batches_025.py",
             ROOT/"D/surface-clearance-546/topology/topology-bone-aware-025.py",ROOT/"D/focused-clearance-541/topology/repair_focused_clearance_541.py",
             ROOT/"D/connected-remesh-reuse-370/select_bone_reducing_flips_370.py",ROOT/"D/surface-clearance-546/continue_surface_round546.py",SLOT,
             STATE,candidate,audit130,audit159,self_pairs)
    status=owner.run_stage(argv,STAGE,slot=True,source_paths=sources,timeout=3600)
    chain=owner.read_json(OUT/"chain-summary.json")
    print(json.dumps({"stage_status":status,"results":chain.get("results"),"final_state":chain.get("final_state"),"chain_summary":str(OUT/"chain-summary.json")},sort_keys=True),flush=True)

if __name__=="__main__": main()
