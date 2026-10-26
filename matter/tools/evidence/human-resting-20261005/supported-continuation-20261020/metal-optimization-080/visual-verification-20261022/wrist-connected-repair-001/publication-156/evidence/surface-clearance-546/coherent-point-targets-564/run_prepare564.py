from pathlib import Path
import sys,importlib.util,json
N=Path(__file__).resolve().parent.parent
s=importlib.util.spec_from_file_location('stage564',N/'continue_surface_round546.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
h=N/'connected-source-refinement-553/run_source_outward_targets553.py'
B=N/'conditioned-connected-refinement-558/refine-all-001'
parent=N/'connected-reference-degeneracy-repair-557/unselected-topology027-004'
a=[sys.executable,str(h),'--direction-mode','source-outward','--parent-candidate',str(parent/'diagnostic-proposal.npz'),
'--parent-point-summary',str(parent/'full-check/audit-130/summary.json'),'--candidate',str(B/'candidate.npz'),
'--point-summary',str(B/'audit-130/summary.json'),'--query-dir',str(B/'audit-130/query'),
'--output-dir',str(Path(__file__).parent/'targets-001'),'--clearance-m','.00005']
m.run_stage(a,Path(__file__).parent/'targets-001-stage',slot=True,source_paths=(Path(__file__),h),timeout=120)
print(json.dumps({'status':'SOURCE_OUTWARD_ENDPOINT_PROPOSALS_ONLY','output':str(Path(__file__).parent/'targets-001')}),flush=True)
