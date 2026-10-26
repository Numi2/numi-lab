"""Prepare the existing joint fit on558's exact current topology; no geometry admission."""
from pathlib import Path
import argparse,importlib.util,json,sys
N=Path(__file__).resolve().parent.parent
s=importlib.util.spec_from_file_location('joint_stages558',N/'continue_surface_round546.py')
m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
ap=argparse.ArgumentParser();ap.add_argument('--entry',required=True,type=Path);ap.add_argument('--context',type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--iterations',type=int,default=2);ap.add_argument('--run',action='store_true');a=ap.parse_args()
T=a.output;T.mkdir(exist_ok=False)
B=N/'conditioned-connected-refinement-558/refine-all-001';P=N/'conditioned-connected-refinement-558/prepared-001'
state=json.loads((B/'state.json').read_text());context_path=a.context or P/'constraint-context.json';ctx=json.loads(context_path.read_text())
for k in ['candidate','point_summary','surface_report']:
 if m.sha(state[k])!=state[k+'_sha256']:raise ValueError('stale state '+k)
matched=json.loads((N/'refined-point-001/matched-point-fit-inputs.json').read_text())
paths={'candidate':Path(state['candidate']),'audit159':Path(state['surface_report']),
'face_target_npz':P/'surface-targets/targets.npz','point_target_npz':P/'point-targets/nearest-union-exits.npz'}
if set(matched['inputs'])!=set(paths):raise ValueError('matched roles')
matched['inputs']={k:{'path':str(p),'sha256':m.sha(p)}for k,p in paths.items()}
m.write_json(T/'matched-inputs.json',matched)
argv=json.loads((N/'refined-point-001/point-fit-001-argv.prepared.json').read_text());argv[1]=str(a.entry)
changes={'--base-candidate':state['candidate'],'--raw-trial':state['candidate'],'--area-reference-npz':state['candidate'],
'--self-pairs':ctx['seed_pairs_path'],'--point-target-npz':P/'point-targets/nearest-union-exits.npz',
'--point-target-report':P/'point-targets/nearest-union-exits.json','--face-target-npz':P/'surface-targets/targets.npz',
'--face-target-report':P/'surface-targets/report.json','--point-depth-rows-npz':P/'depth/point-depth-rows.npz',
'--point-depth-rows-report':P/'depth/365-compatible-report.json','--fixed-boundary-npz':ctx['guard_npz'],
'--fixed-boundary-report':ctx['guard_report'],'--matched-inputs-json-490':T/'matched-inputs.json',
'--audit159-490':state['surface_report'],'--area-scp-iterations':a.iterations,'--output':T/'fit'}
for k,v in changes.items():m.replace_cli_value(argv,k,v)
m.write_json(T/'fit-argv.json',argv)
pins={str(p):m.sha(p)for p in [Path(__file__),a.entry,B/'state.json',context_path,*paths.values(),
P/'point-targets/nearest-union-exits.json',P/'surface-targets/report.json',P/'depth/point-depth-rows.npz',
P/'depth/365-compatible-report.json',Path(ctx['seed_pairs_path']),Path(ctx['guard_npz']),Path(ctx['guard_report'])]}
m.write_json(T/'input-pins.json',{'source_pins':pins,'native_admission':False,'accepted_physical_steps':0,'temporary_unselected_construction':True})
if not a.run:print(json.dumps({'status':'PREPARED_NOT_RUN','argv':str(T/'fit-argv.json')}));raise SystemExit
m.run_stage(argv,T/'fit-stage',slot=True,source_paths=(Path(__file__),a.entry,),timeout=600)
if not all(m.sha(p)==v for p,v in pins.items()):raise ValueError('matched input/source changed')
proposal=json.loads((T/'fit/report.json').read_text())
if proposal.get('status')!='CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED' or not (T/'fit/candidate.npz').is_file():
 m.write_json(T/'early-stop.json',{'status':'STOPPED_AT_FIT_WITHOUT_CANDIDATE','fit_status':proposal.get('status'),
 'fit_report':str(T/'fit/report.json'),'fit_report_sha256':m.sha(T/'fit/report.json'),
 'native_admission':False,'accepted_physical_steps':0})
 print(json.dumps({'status':'STOPPED_AT_FIT_WITHOUT_CANDIDATE','fit_status':proposal.get('status')}),flush=True)
 raise SystemExit(2)
check=N.parent/'contained-clearance-545/run_candidate_checks545.py'
m.run_stage([sys.executable,str(check),'--candidate',str(T/'fit/candidate.npz'),'--baseline-state',str(B/'state.json'),
'--trust-mm','4','--output-dir',str(T/'full-check')],T/'full-check-stage',slot=False,source_paths=(check,),timeout=600)
r=json.loads((T/'full-check/report.json').read_text())
print(json.dumps({k:r.get(k)for k in ['status','surface_pairs','contained_pairs','self_pairs','construction_progress']}),flush=True)
