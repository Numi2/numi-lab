import json,math
from pathlib import Path
p=Path(__file__).resolve().parent
runs={a:json.loads((p/(a+'.json')).read_text())['cases'] for a in ('baseline','candidate')}
report={}
for arm,cases in runs.items():
 assert len(cases)==6
 results=[]
 for c in cases:
  assert c['success'] and c['repeat'] and c['contacts']==len(c['rows'])==2
  v=c['cube_velocity'];g=c['ground_velocity'];initial=(c['case']-1,-float(c['active']),0,0,0,0)
  residual=0.; J=0.
  for r in c['rows']:
   assert r['bodyA']==0 and r['bodyB']==2 and r['tangent']==[0,0]
   x,y,z=r['point'];y-=c['initial_y'];n=r['direction'];w=v[3:]
   cross=(w[1]*z-w[2]*y,w[2]*x-w[0]*z,w[0]*y-w[1]*x)
   vn=sum(n[j]*(v[j]+cross[j]-g[j]) for j in range(3));e=vn-r['target']+r['reg']*r['lambda']
   residual=max(residual,abs(e) if r['lambda']>1e-8 else max(-e,0));J+=r['lambda']
  poserr=max(abs(v[j]-initial[j]) for j in range(6)) if not c['active'] else None
  momentum=abs((v[1]-initial[1])-J)
  results.append({'active':c['active'],'case':c['case'],'residual':residual,'reported_error':abs(c['solver_residual']-residual),'momentum_error':momentum,'impulse':J,'sliding_velocity_error':poserr})
 report[arm]=results
 print(arm,'slide errors',[x['sliding_velocity_error'] for x in results[:3]],'active normal residual',[x['residual'] for x in results[3:]],'max impulse balance error',max(x['momentum_error'] for x in results))
worst=max(abs(runs['candidate'][j]['cube_velocity'][k]-runs['candidate'][1]['cube_velocity'][k]) for j in (0,2) for k in (1,3,4,5))
worstactive=max(abs(runs['candidate'][j]['cube_velocity'][k]-runs['candidate'][4]['cube_velocity'][k]) for j in (3,5) for k in (1,3,4,5))
assert worst<2e-5 and worstactive<2e-5
assert all(x['sliding_velocity_error']<1e-6 and x['impulse']==0 for x in report['candidate'][:3])
assert all(x['impulse']>0 and x['residual']<.002 for x in report['candidate'][3:])
assert all(x['reported_error']<2e-5 and x['momentum_error']<2e-5 for x in report['candidate'])
(p/'analysis.json').write_text(json.dumps(report,indent=2))
print('candidate scene checks PASS; tangential dependence',worst,worstactive)
