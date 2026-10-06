from pathlib import Path
import json, subprocess, os
root=Path(__file__).parent
repo=Path('/Users/home/numi-cardboard-20261006')
for plan_name in ['nonlinear-iterate-diagnostic-plan.json','final-identity-regressions-plan.json']:
 plan=json.loads((root/plan_name).read_text())
 for trial in plan['trials']:
  env=os.environ.copy()
  if plan_name.startswith('nonlinear'): env['NM_FEM_NEWTON_TRACE_ROOT']='0'
  with (root/(trial['name']+'-driver.log')).open('x') as stream:
   result=subprocess.run(['python3',str(root/'run_native_trace.py'),trial['name'],*trial['args']],cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
  print(trial['name'],result.returncode,flush=True)
  print((root/(trial['name']+'-driver.log')).read_text()[-1000:],flush=True)
  if plan_name.startswith('nonlinear'):
   cmd=['python3','matter/tools/cardboard_newton_trace.py',str(root/'receipts'/trial['name']/'stderr.log'),'--root','0','--near-full-alpha','.99','--growth-limit','1.1','--output',str(root/(trial['name']+'-newton-analysis.json'))]
   decoded=subprocess.run(cmd,cwd=repo)
   print('decoder',decoded.returncode,flush=True)
   if decoded.returncode: raise SystemExit('Trace integrity failure')
  elif result.returncode: raise SystemExit('Identity regression failed')
