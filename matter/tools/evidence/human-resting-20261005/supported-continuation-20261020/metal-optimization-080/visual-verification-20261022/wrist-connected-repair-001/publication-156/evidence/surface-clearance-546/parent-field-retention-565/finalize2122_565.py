from pathlib import Path
import importlib.util,sys,json
N=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546')
sp=importlib.util.spec_from_file_location('f565',N/'continue_surface_round546.py')
m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
T=N/'parent-field-retention-565';p=T/'regional-549-001-full-check'
m.run_stage([sys.executable,N/'finalize_existing_checks546.py','--candidate',T/'regional-549-001/candidate.npz',
'--baseline-state',N/'connected-reference-degeneracy-repair-557/current-best029-004/state.json',
'--point-summary',p/'audit-130/summary.json','--surface-report',p/'audit-159/report.json',
'--out',T/'current-best-2122-001','--trust-mm','4'],T/'finalize-2122-001',slot=False,
source_paths=(N/'finalize_existing_checks546.py',),timeout=60)
s=json.loads((T/'current-best-2122-001/state.json').read_text())
print(json.dumps(s,indent=2))

