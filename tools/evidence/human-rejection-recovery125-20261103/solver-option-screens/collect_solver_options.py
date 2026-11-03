from pathlib import Path
import json,re,csv,statistics,sys
D=Path(__file__).parent
result={}
for name in sys.argv[1:]:
 t=D/name;execution=json.loads((t/'install-motion.execution.json').read_text());times=[];profiles=[]
 with(t/'install-motion.stdout.log').open()as f:
  for l in f:
   if 'resting_FEM_endpoint_capture=begin'in l:break
   if l.startswith('resting_native_profile '):
    d=dict(x.split('=',1)for x in l.split()[1:]if'='in x);profiles.append(d);times.append(float(d['gpu_ms'])/1000)
 p=t/'install-motion/resting-coupled.csv'
 with p.open() if p.exists() else open('/dev/null') as f:rows=list(csv.DictReader(f))
 cert=[]
 with(t/'install-motion.stderr.log').open()as f:
  for l in f:
   if l.startswith('fem_newton_terminal_certificate='):
    v=json.loads(l.split('=',1)[1]);cert.append({'root':v['root'],'status':v['status'],'fgmres':v['fgmres']})
 result[name]={'native_exit':execution['exit_code'],'elapsed_s_including_initialization_and_failed_attempt':execution['elapsed_s'],'resource_stop_reason':execution['resource_stop_reason'],'accepted_steps':len(rows),'accepted_time_s':float(rows[-1]['time_s'])if rows else 0,'accepted_GPU_seconds':times,'accepted_GPU_mean_seconds':statistics.mean(times)if times else None,'GPU_only_RTF':len(rows)*0.000015625/sum(times)if times else None,'elapsed_inclusive_RTF':len(rows)*0.000015625/execution['elapsed_s'],'last_terminal_certificate':cert[-1]if cert else None,'profiles':profiles,'claims':'Screen only; unbalanced release transient; no resting/motion/performance qualification; endpoint field comparison still required.'}
(D/'solver-option-screens-result.json').write_text(json.dumps(result,indent=2)+'\n')
for k,v in result.items():print(k,{a:v[a]for a in['native_exit','accepted_steps','accepted_GPU_mean_seconds','GPU_only_RTF','elapsed_inclusive_RTF']})
