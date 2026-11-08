import json, math
from pathlib import Path
p=Path(__file__).resolve().parent
baseline=json.loads((p/'expanded-baseline.json').read_text())['cases']; candidate=json.loads((p/'expanded-candidate.json').read_text())['cases']
assert len(baseline)==len(candidate)==40
fail=[];base_fail=[];finite_identical=True;success=0
for a,b in zip(baseline,candidate):
 assert (a['angle'],a['mu'],a['case'])==(b['angle'],b['mu'],b['case'])
 assert a['success'] and b['success'] and a['repeat'] and b['repeat']
 if a['case']==4:
  if a['contacts'] or b['contacts']:fail.append((b['angle'],b['mu'],'unexpected no-contact control contacts',a['contacts'],b['contacts']))
  continue
 if not a['contacts'] or not b['contacts']:fail.append((b['angle'],b['mu'],'missing contact',a['contacts'],b['contacts']))
 if b['mu']>0:
  if a['v']!=b['v'] or a['q']!=b['q'] or a['contacts']!=b['contacts'] or a['gpu_status']!=b['gpu_status']:
   finite_identical=False;fail.append((b['angle'],b['mu'],b['case'],'finite-friction changed'))
  continue
 initial=[[1,0,0,0,0,0],[-1,0,0,0,0,0],[0,0,1,0,0,0],[0,0,-1,0,0,0]][b['case']]
 be=max(abs(x-y) for x,y in zip(b['v'],initial));ae=max(abs(x-y) for x,y in zip(a['v'],initial))
 if be>1e-6:fail.append((b['angle'],b['case'],'candidate velocity error',be,b['v']))
 if ae>1e-6:base_fail.append((a['angle'],a['case'],ae))
 success+=1
print('candidate_zero_friction_passed',success-len([x for x in fail if 'candidate velocity error' in x]),'of',success)
print('baseline_zero_friction_failed',len(base_fail),'of',success,base_fail)
print('finite_friction_identical',finite_identical)
print('failed',fail)
if fail: raise SystemExit(1)
print('PASS')
