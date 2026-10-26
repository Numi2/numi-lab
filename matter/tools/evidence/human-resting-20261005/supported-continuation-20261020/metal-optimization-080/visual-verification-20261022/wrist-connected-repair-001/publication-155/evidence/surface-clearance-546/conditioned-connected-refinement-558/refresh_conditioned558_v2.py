from pathlib import Path
import importlib.util,json,sys
N=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546')
s=importlib.util.spec_from_file_location('stages558refresh',N/'continue_surface_round546.py')
m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
P=N/'conditioned-connected-refinement-558/prepared-001'
state=json.loads((N/'conditioned-connected-refinement-558/refine-all-001/state.json').read_text())
prior=json.loads((N/'round-026-guard-cuts556/constraint-context.json').read_text())
base=state['candidate'];h=N/'refresh_surface_constraints546.py'
a=[sys.executable,str(h),'--candidate',base,'--audit130',state['point_summary'],'--audit159',state['surface_report'],
'--query-dir',str(Path(state['point_summary']).parent/'query'),
'--ancestry-parent-candidate',str(N/'connected-reference-degeneracy-repair-557/unselected-topology027-004/diagnostic-proposal.npz'),'--prior-candidate',prior['candidate'],
'--prior-surface-report',prior['surface_report'],'--prior-guard-candidate',prior['candidate'],
'--prior-guard-npz',prior['guard_npz'],'--prior-guard-report',prior['guard_report'],
'--prior-seed-pairs',prior['seed_pairs_path'],'--prior-seed-report',prior['seed_report'],
'--failed-seed-candidate',base,'--failed-seed-audit130',state['point_summary'],
'--failed-seed-pairs',str(Path(state['point_summary']).parent/'surface-self-pairs.npy'),
'--output-dir',str(P/'constraints-002'),'--point-continuation']
m.run_stage(a,P/'refresh-stage-002',slot=True,source_paths=(h,N/'refresh_guards_multi_parent_546.py'),timeout=600)
rr=json.loads((P/'constraints-002/refresh-report.json').read_text())
ctx={**{k:state[k]for k in ['candidate','candidate_sha256','point_summary','point_summary_sha256','surface_report','surface_report_sha256']},
'seed_pairs_path':rr['seed_pairs_path'],'seed_report':rr['seed_report_path'],
'guard_npz':rr['fixed_guard_npz_path'],'guard_report':rr['fixed_guard_report_path'],
'native_admission':False,'accepted_physical_steps':0}
m.write_json(P/'constraint-context.json',ctx)
print(json.dumps({'status':'MATCHED_CONSTRAINTS_PREPARED','context':str(P/'constraint-context.json')}),flush=True)
