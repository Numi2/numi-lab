from pathlib import Path
import sys,json,importlib.util
N=Path(__file__).resolve().parent.parent
sp=importlib.util.spec_from_file_location('st567',N/'continue_surface_round546.py');m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
P=N/'local-witness-retention-567';T=P/'run-001'
S=N/'affine-ancestry-refresh-566/from2106-001/final/state.json'
target=N/'parent-field-retention-565/run-001/candidate.npz'
owner=P/'retain_local_witnesses567.py'
m.run_stage([sys.executable,owner,'--state',S,'--target',target,'--out',T],P/'run-001-stage',slot=True,source_paths=(Path(__file__),owner,S,target),timeout=600)
checker=N.parent/'contained-clearance-545/run_candidate_checks545.py'
m.run_stage([sys.executable,checker,'--candidate',T/'candidate.npz','--baseline-state',S,'--trust-mm','4','--output-dir',T/'full-check'],T/'full-check-stage',slot=False,source_paths=(checker,),timeout=600)
r=json.loads((T/'full-check/report.json').read_text());print(json.dumps({k:r.get(k)for k in ['status','surface_pairs','contained_pairs','self_pairs','construction_progress']}),flush=True)
if r.get('construction_progress') and r.get('contained_pairs')==0 and r.get('self_pairs')==0:
 f=N/'finalize_existing_checks546.py'
 m.run_stage([sys.executable,f,'--candidate',T/'candidate.npz','--baseline-state',S,'--point-summary',T/'full-check/audit-130/summary.json','--surface-report',T/'full-check/audit-159/report.json','--out',T/'final','--trust-mm','4'],T/'final-stage',slot=False,source_paths=(f,),timeout=60)

