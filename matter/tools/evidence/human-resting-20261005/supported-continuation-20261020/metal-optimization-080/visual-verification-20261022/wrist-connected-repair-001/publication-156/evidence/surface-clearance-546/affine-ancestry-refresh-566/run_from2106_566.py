from pathlib import Path
import importlib.util,sys,json
N=Path(__file__).resolve().parent.parent
sp=importlib.util.spec_from_file_location('st566',N/'continue_surface_round546.py');m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
o=N/'affine-ancestry-refresh-566/rebuild_affine_rows566.py'
m.run_stage([sys.executable,o,'--candidate',N/'topology/round-035-from2122/round-003/candidate.npz',
'--out',N/'affine-ancestry-refresh-566/from2106-001'],N/'affine-ancestry-refresh-566/from2106-001-stage',
slot=True,source_paths=(Path(__file__),o,o.parent.parent/'connected-reference-degeneracy-repair-557/frozen-v4/diagnose_reference_degeneracy_557.py'),timeout=60)
r=json.loads((o.parent/'from2106-001/report.json').read_text());print(json.dumps({'changed_fields':r['changed_fields'],'repair':r['repair']}))

