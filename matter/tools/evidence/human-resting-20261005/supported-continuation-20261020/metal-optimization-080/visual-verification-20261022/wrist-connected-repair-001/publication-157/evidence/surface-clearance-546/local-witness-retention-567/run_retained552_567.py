"""Retain only measured improving regions of 567; existing552 and545 owners."""
from pathlib import Path
import importlib.util,json,sys
N=Path(__file__).resolve().parent.parent
sp=importlib.util.spec_from_file_location('r567552',N/'continue_surface_round546.py');m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
T=N/'local-witness-retention-567/retained-552-001';P=N/'local-witness-retention-567/run-001'
S=N/'affine-ancestry-refresh-566/from2106-001/final/state.json';ctx=N/'context-refresh-568/from2106-001/constraints/constraint-context.json'
owner=N/'retain_surface_components552.py'
m.run_stage([sys.executable,owner,'--state',S,'--proposal',P,'--full-check',P/'full-check','--context',ctx,'--output',T],T.with_name(T.name+'-stage'),slot=False,source_paths=(Path(__file__),owner,ctx),timeout=60)
r=json.loads((T/'report.json').read_text());print(json.dumps({'predicted_surface_pairs':r['predicted_surface_pairs'],'selected':[z['component']for z in r['choices']if z['selected_trial']]}),flush=True)
check=N.parent/'contained-clearance-545/run_candidate_checks545.py'
m.run_stage([sys.executable,check,'--candidate',T/'candidate.npz','--baseline-state',S,'--trust-mm','4','--output-dir',T/'full-check'],T/'full-check-stage',slot=False,source_paths=(check,),timeout=600)
r=json.loads((T/'full-check/report.json').read_text());print(json.dumps({k:r.get(k)for k in ['status','surface_pairs','contained_pairs','self_pairs','construction_progress']}),flush=True)
if r.get('construction_progress') and r.get('contained_pairs')==0 and r.get('self_pairs')==0:
 f=N/'finalize_existing_checks546.py'
 m.run_stage([sys.executable,f,'--candidate',T/'candidate.npz','--baseline-state',S,'--point-summary',T/'full-check/audit-130/summary.json','--surface-report',T/'full-check/audit-159/report.json','--out',T/'final','--trust-mm','4'],T/'final-stage',slot=False,source_paths=(f,),timeout=60)

