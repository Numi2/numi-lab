"""Retain safe connected regions from a saved transferred field; offline construction only."""
from pathlib import Path
import importlib.util,json,sys
N=Path(__file__).resolve().parent.parent
sp=importlib.util.spec_from_file_location("stage565",N/"continue_surface_round546.py")
m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
P=N/"parent-field-retention-565"
T=P/"regional-549-001"
base=N/"connected-reference-degeneracy-repair-557/current-best029-004/state.json"
raw=P/"raw-transfer-001"
ctx=N/"round-033-565-refresh/constraints/constraint-context.json"
desc=P/"prepare-001/549-transfer-descriptor.json"
ann=P/"prepare-001/annotate_549_transfer565.py"
owner=N.parent/"component-area-backtracking-549/component_area_backtracking549.py"
checker=N.parent/"contained-clearance-545/run_candidate_checks545.py"
pins={str(p):m.sha(p) for p in (Path(__file__),base,raw/"report.json",raw/"candidate.npz",ctx,desc,owner,ann,checker)}
m.write_json(P/"regional-549-001-input-pins.json",{"source_pins":pins,"native_admission":False,"accepted_physical_steps":0})
m.run_stage([sys.executable,owner,"--raw-dir",raw,"--context",ctx,"--output",T,"--trust-mm","4","--exact-self"],
P/"regional-549-001-stage",slot=True,source_paths=(Path(__file__),owner),timeout=600)
m.run_stage([sys.executable,ann,"--549-report",T/"report.json","--raw-report",raw/"report.json","--descriptor",desc],
P/"regional-549-001-annotation",slot=False,source_paths=(ann,),timeout=60)
if not all(m.sha(p)==v for p,v in pins.items()):raise ValueError("matched source or input changed")
report=json.loads((T/"report.json").read_text())
if report.get("status")!="CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED":raise ValueError("no qualified construction proposal")
m.run_stage([sys.executable,checker,"--candidate",T/"candidate.npz","--baseline-state",base,
"--trust-mm","4","--output-dir",P/"regional-549-001-full-check"],
P/"regional-549-001-full-check-stage",slot=False,source_paths=(checker,),timeout=600)
r=json.loads((P/"regional-549-001-full-check/report.json").read_text())
print(json.dumps({k:r.get(k) for k in ["status","surface_pairs","contained_pairs","self_pairs","construction_progress"]}),flush=True)

