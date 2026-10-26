from pathlib import Path
import importlib.util,json,sys
W=Path(__file__).parent;sp=importlib.util.spec_from_file_location('continuation545',W/'continue545.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
O=W/'round-004';args=m.read(O/'fit.launch.json')['argv'][2:]
args[args.index('--output')+1]=str(O/'retry-fit');args[args.index('--retained-gain-fraction')+1]='.95'
m.run(args,O/'retry-fit',True)
m.run([m.PY,W/'run_checked_batch_context545.py',W/'round-003/state.json',O/'retry-fit',O/'constraint-context.json',O/'retry-backtrack'],O/'retry-backtrack')
s=m.read(O/'retry-backtrack/backtrack-373/selected-result.json')
m.run([m.PY,W/'run_candidate_checks545.py','--candidate',s['candidate'],'--baseline-state',W/'round-003/state.json','--output-dir',O/'retry-arithmetic','--arithmetic-only'],O/'retry-arithmetic')
n={k:s[k]for k in ('candidate','point_summary','surface_report')};n.update({k+'_sha256':m.sha(v)for k,v in list(n.items())});n.update(native_admission=False,accepted_physical_steps=0);m.write(O/'retry-state.json',n)
a=m.read(n['point_summary']);sr=m.read(n['surface_report']);m.write(O/'retry-progress.json',dict(before_inside=160,after_inside=a['inside_node_bank_pairs'],surface_pairs=sum((b.get('result')or{}).get('count',0)for b in sr['banks']),self_pairs=a['exact_PLC_surface_pairs'],candidate_sha256=n['candidate_sha256'],state=str(O/'retry-state.json'),native_admission=False,accepted_physical_steps=0))
print(json.dumps(m.read(O/'retry-progress.json')),flush=True)
