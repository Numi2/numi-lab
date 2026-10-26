"""Full existing gates for ancestry-repaired, geometry-identical 2106 checkpoint."""
from pathlib import Path
import importlib.util,sys,json
N=Path(__file__).resolve().parent.parent
sp=importlib.util.spec_from_file_location('c566',N/'continue_surface_round546.py');m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
T=N/'affine-ancestry-refresh-566/from2106-001';C=T/'candidate.npz'
S=N/'parent-field-retention-565/current-best-2122-001/state.json'
check=N.parent/'contained-clearance-545/run_candidate_checks545.py'
m.run_stage([sys.executable,check,'--candidate',C,'--baseline-state',S,'--trust-mm','4','--output-dir',T/'full-check'],T/'full-check-stage',slot=False,source_paths=(Path(__file__),check),timeout=600)
r=json.loads((T/'full-check/report.json').read_text());print(json.dumps({k:r.get(k)for k in ['status','surface_pairs','contained_pairs','self_pairs','construction_progress']}),flush=True)
if r.get('construction_progress') and r.get('contained_pairs')==0 and r.get('self_pairs')==0:
 f=N/'finalize_existing_checks546.py'
 m.run_stage([sys.executable,f,'--candidate',C,'--baseline-state',S,'--point-summary',T/'full-check/audit-130/summary.json',
 '--surface-report',T/'full-check/audit-159/report.json','--out',T/'final','--trust-mm','4'],T/'final-stage',slot=False,source_paths=(f,),timeout=60)

