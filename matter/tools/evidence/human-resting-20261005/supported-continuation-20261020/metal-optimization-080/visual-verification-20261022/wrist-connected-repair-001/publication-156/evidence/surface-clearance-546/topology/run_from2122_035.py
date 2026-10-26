"""Fresh fixed-coordinate topology batch on the newly deformed 2122 checkpoint."""
from pathlib import Path
import json,sys,importlib.util
N=Path(__file__).resolve().parent.parent
sp=importlib.util.spec_from_file_location('top566',N/'continue_surface_round546.py')
m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
S=N/'parent-field-retention-565/current-best-2122-001/state.json'
s=json.loads(S.read_text())
C=Path(s['candidate']);A=Path(s['surface_report']);H=Path(s['point_summary']);P=H.parent/'surface-self-pairs.npy'
T=N/'topology/round-035-from2122'
owner=N/'topology/chain_exact_flip_batches_025.py'
argv=[sys.executable,owner,'--candidate',C,'--audit159',A,'--audit130',H,'--self-pairs130',P,
'--expected-candidate-sha256',m.sha(C),'--expected-audit159-sha256',m.sha(A),
'--expected-audit130-sha256',m.sha(H),'--expected-self-pairs130-sha256',m.sha(P),
'--output-root',T,'--rounds','3','--max-candidates','2048','--prefilter-factor','1',
'--max-accept','64','--progress-every','128']
m.run_stage(argv,N/'topology/round-035-stage',slot=True,
source_paths=(Path(__file__),owner,N/'topology/topology-bone-aware-025.py',S),timeout=600)
r=json.loads((T/'chain-summary.json').read_text())
print(json.dumps({'results':r['results'],'final_state':r['final_state']},indent=2),flush=True)

