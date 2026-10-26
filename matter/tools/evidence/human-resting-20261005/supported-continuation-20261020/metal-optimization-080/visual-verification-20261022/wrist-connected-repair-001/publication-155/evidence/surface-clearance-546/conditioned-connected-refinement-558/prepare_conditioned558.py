"""Prepare hash-bound existing targets for the conditioned connected branch; no fit/admission."""
from pathlib import Path
import importlib.util,json,sys
N=Path(__file__).resolve().parent.parent
s=importlib.util.spec_from_file_location('stages558',N/'continue_surface_round546.py')
m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
B=N/'conditioned-connected-refinement-558/refine-all-001'
P=N/'conditioned-connected-refinement-558/prepared-001'
P.mkdir(exist_ok=False)
state={k:str(p)for k,p in {'candidate':B/'candidate.npz','point_summary':B/'audit-130/summary.json','surface_report':B/'audit-159/report.json'}.items()}
state.update({k+'_sha256':m.sha(v)for k,v in list(state.items())})
state.update(candidate_selected=False,native_admission=False,accepted_physical_steps=0,temporary_unselected_construction=True,continuation_mode='point_continuation')
m.write_json(B/'state.json',state)
argv=json.loads((N.parent/'contained-clearance-545/depth-argv.json').read_text())
Q=B/'audit-130/query'
changes={'--candidate':state['candidate'],'--point-summary':state['point_summary'],'--surface-report':state['surface_report'],
'--query-dir':Q,'--output-dir':P/'depth','--expected-candidate-sha256':state['candidate_sha256'],
'--expected-point-summary-sha256':state['point_summary_sha256'],'--expected-surface-report-sha256':state['surface_report_sha256'],
'--expected-query-receipt-sha256':m.sha(Q/'query-receipt.json')}
for k,v in changes.items():m.replace_cli_value(argv,k,v)
m.run_stage(argv,P/'depth-stage',slot=True,source_paths=(Path(__file__),Path(argv[1])),timeout=600)
h=N/'direct-depth-targets-546/run_direct_depth_targets546.py'
m.run_stage([sys.executable,str(h),'--candidate',state['candidate'],'--point-summary',state['point_summary'],'--query-dir',str(Q),
'--output-dir',str(P/'point-targets'),'--clearance-m','.00005','--direction-mode','direct-depth',
'--point-depth-rows-npz',str(P/'depth/point-depth-rows.npz'),'--point-depth-rows-report',str(P/'depth/365-compatible-report.json'),
'--minimum-depth-cosine','.05'],P/'point-targets-stage',slot=True,source_paths=(h,),timeout=600)
h=N.parent/'source-outward-proposal-500/build_outward_targets_multi_source_546.py'
m.run_stage([sys.executable,str(h),'--candidate',state['candidate'],'--surface-report',state['surface_report'],
'--query-receipt',str(Q/'query-receipt.json'),'--output-dir',str(P/'surface-targets'),
'--clearance-m','.00005','--direction-mode','outward-nearest'],P/'surface-targets-stage',slot=True,source_paths=(h,),timeout=600)
m.write_json(P/'prepared.json',{'state':str(B/'state.json'),'state_sha256':m.sha(B/'state.json'),
'status':'MATCHED_TARGETS_PREPARED_NOT_FITTED','native_admission':False,'accepted_physical_steps':0})
print(json.dumps({'status':'MATCHED_TARGETS_PREPARED_NOT_FITTED','output':str(P)}),flush=True)
