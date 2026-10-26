"""Opt-in wrapper over pinned 458 targets and 408 physical hard constraints."""
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
RUNNER458=Path("/Users/numi/Work/numi-human-connected-surface-069/D/D/fixed-bone-outward-plane-458/run_grouped_fit_458.py")
SOLVER470=HERE/"soft_target_nonregression_470.py"
RUNNER458_SHA256="08cd0e2681bb5ae16a85a0384514bb30739f8d62a06ea2ab1bdfcd896be4ef93"
SOLVER470_SHA256="990e126932186e33c0a5363a6ac34fa967e211f4bc55977567582e2b89b8bb79"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""): h.update(block)
 return h.hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 if spec is None or spec.loader is None: raise ImportError(f"cannot import pinned module: {path}")
 module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module); return module
def load_dependencies():
 if sha(RUNNER458)!=RUNNER458_SHA256: raise ValueError("470 refuses changed 458 wrapper")
 if sha(SOLVER470)!=SOLVER470_SHA256: raise ValueError("470 solver helper changed after review")
 return load("runner458_for_470",RUNNER458),load("solver470",SOLVER470)
def install_variant(owner,objective,crossing,runner=None,solver=None):
 if runner is None or solver is None:
  loaded_runner,loaded_solver=load_dependencies()
  runner=loaded_runner if runner is None else runner
  solver=loaded_solver if solver is None else solver
 state=runner.install_objective(owner,objective,crossing)
 state["solver_470"]=solver.install_solver(owner)
 return state
def main():
 runner,solver=load_dependencies(); state_box={}
 def install(owner,objective,crossing):
  state=install_variant(owner,objective,crossing,runner=runner,solver=solver)
  state_box.clear(); state_box.update(state); return state
 runner.install_objective=install
 runner.main()
 out=runner.arg_value("--output")/"report.json"
 report=json.loads(out.read_text())
 report["soft_target_nonregression_470"]={
  "variant":"408 reduced-basis LP without artificial C*d>=0 rows",
  "objective_and_hard_guard_installer":"pinned 458",
  "row_counts":state_box.get("objective"),
  "solver_receipt":state_box.get("solver_470"),
  "negative_soft_target_progress_is_reported":True,
  "physical_clearance_claim":False,"construction_only":True,"native_admission":False,
  "wrapper_sha256":sha(Path(__file__)),"solver_helper_sha256":sha(SOLVER470),
  "pinned_458_wrapper_sha256":sha(RUNNER458)}
 out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 if sha(SOLVER470)!=SOLVER470_SHA256 or sha(RUNNER458)!=RUNNER458_SHA256:
  raise RuntimeError("pinned 470/458 source changed during run")
if __name__=="__main__": main()
