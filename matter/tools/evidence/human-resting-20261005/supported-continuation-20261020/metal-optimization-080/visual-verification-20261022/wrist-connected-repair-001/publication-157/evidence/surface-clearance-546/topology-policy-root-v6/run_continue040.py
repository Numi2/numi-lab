"""Execute the root-reviewed lazy-bound policy, then existing lineage/geometry gates."""
from pathlib import Path
import importlib.util,json,sys
N=Path(__file__).resolve().parent.parent
sp=importlib.util.spec_from_file_location('rt025v5',N/'continue_surface_round546.py');m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
S=N/'topology/round-039-root-policy-v6/final/state.json'
s=json.loads(S.read_text())
for k in ['candidate','point_summary','surface_report']:
 if m.sha(s[k])!=s[k+'_sha256']:raise ValueError('changed input '+k)
C=Path(s['candidate']);A=Path(s['surface_report']);H=Path(s['point_summary']);P=H.parent/'surface-self-pairs.npy'
T=N/'topology/round-040-root-policy-v6';owner=Path(__file__).with_name('run_quality_policy025_variant.py')
argv=[sys.executable,owner,'--chain','--candidate',C,'--audit159',A,'--audit130',H,'--self-pairs130',P,
'--expected-candidate-sha256',m.sha(C),'--expected-audit159-sha256',m.sha(A),'--expected-audit130-sha256',m.sha(H),'--expected-self-pairs130-sha256',m.sha(P),
'--output-root',T,'--rounds','8','--max-candidates','2048','--prefilter-factor','1','--max-accept','64','--progress-every','128']
m.run_stage(argv,N/'topology/round-040-root-policy-v6-stage',slot=True,source_paths=(Path(__file__),owner,owner.with_name('quality_policy025_variant.py'),S),timeout=600)
r=json.loads((T/'chain-summary.json').read_text());print(json.dumps({'results':r['results']}),flush=True)
f=r['final_state'];before=r['results'][0]['before_pairs']
if f['surface_pair_count_composed']>=before:raise SystemExit(0)
lineage=N/'affine-ancestry-refresh-566/rebuild_affine_rows566.py'
m.run_stage([sys.executable,lineage,'--candidate',f['candidate'],'--out',T/'lineage'],T/'lineage-stage',slot=True,source_paths=(lineage,),timeout=60)
check=N.parent/'contained-clearance-545/run_candidate_checks545.py'
m.run_stage([sys.executable,check,'--candidate',T/'lineage/candidate.npz','--baseline-state',S,'--trust-mm','4','--output-dir',T/'full-check'],T/'full-check-stage',slot=False,source_paths=(check,),timeout=600)
r=json.loads((T/'full-check/report.json').read_text());print(json.dumps({k:r.get(k)for k in ['status','surface_pairs','contained_pairs','self_pairs','construction_progress']}),flush=True)
if r.get('construction_progress') and r.get('contained_pairs')==0 and r.get('self_pairs')==0:
 final=N/'finalize_existing_checks546.py'
 m.run_stage([sys.executable,final,'--candidate',T/'lineage/candidate.npz','--baseline-state',S,'--point-summary',T/'full-check/audit-130/summary.json','--surface-report',T/'full-check/audit-159/report.json','--out',T/'final','--trust-mm','4'],T/'final-stage',slot=False,source_paths=(final,),timeout=60)

