from pathlib import Path
import importlib.util,sys,json,shutil
N=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546');T=N/'prolongation-witness-seeds-561'
s=importlib.util.spec_from_file_location('stages561',N/'continue_surface_round546.py')
m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
F=T/'frozen-v2';F.mkdir(exist_ok=False)
for p in T.glob('*.py'):shutil.copy2(p,F/p.name)
m.write_json(F/'source-pins.json',{str(p):m.sha(p)for p in F.glob('*.py')})
h=T/'map_prolongation_witnesses561_v2.py'
a=[sys.executable,str(h),'--source-candidate',str(N/'round-031-refined-connected557/fit/candidate.npz'),
'--source-audit130',str(N/'round-031-refined-connected557/full-check/audit-130/summary.json'),
'--source-pairs',str(N/'round-031-refined-connected557/full-check/audit-130/surface-self-pairs.npy'),
'--context',str(N/'conditioned-connected-refinement-558/prepared-001/constraint-context.json'),
'--output',str(T/'run-002')]
m.run_stage(a,T/'run-002-stage',slot=True,source_paths=(h,N/'refresh_surface_constraints546.py'),timeout=240)
r=json.loads((T/'run-002/receipt.json').read_text());print(json.dumps({k:r[k]for k in ['status','merged_pair_count','mapped_witnesses','unmapped_witnesses','zero_step_max_rhs_mm','context']}))
