import json,math
from pathlib import Path
p=Path(__file__).parent
def sub(a,b):return [x-y for x,y in zip(a,b)]
def add(a,b):return [x+y for x,y in zip(a,b)]
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
out={}
for arm in ('baseline','candidate'):
 cases=json.loads((p/f'{arm}.json').read_text())['cases'];assert len(cases)==24
 groups={}
 for c in cases:groups.setdefault((c['angle'],c['iterations'],c['active']),{})[c['case']]=c
 assert len(groups)==8 and all(set(g)=={0,1,2} for g in groups.values())
 rows=[]
 for key,g in groups.items():
  ref=g[1];ang,it,active=key
  for case,c in g.items():
   v=c['v'];u=c['support_v'];init=[case-1,-.5 if active else .1,0]
   momentum=max(abs(v[j]+2*u[j]-init[j]-2*([0,.1,0][j])) for j in range(3))
   tang=max(abs(v[0]-init[0]),abs(v[2]),abs(u[0]),abs(u[2]))
   no_change=max(abs(v[j]-init[j]) for j in range(3));no_change=max(no_change,max(abs(u[j]-[0,.1,0][j]) for j in range(3)),*(abs(x) for x in v[3:6]+u[3:6]))
   pair=max(abs(v[j]-ref['v'][j]) for j in (1,3,4,5));pair=max(pair,max(abs(u[j]-ref['support_v'][j]) for j in range(6)))
   pair=max(pair,*(abs(a['lambda']-b['lambda']) for a,b in zip(c['rows'],ref['rows'])))
   residual=[]
   for r in c['rows']:
    assert r['bodyA']==1 and r['bodyB']==2
    point=r['point'];ra=sub(point,[0,c['initial_y'],0]);rb=sub(point,[0,1,0]);va=add(v[:3],cross(v[3:6],ra));vb=add(u[:3],cross(u[3:6],rb));vn=dot(sub(vb,va),r['direction']);equation=vn-r['target']+r['reg']*r['lambda'];residual.append(abs(equation) if r['lambda']>1e-8 else max(-equation,0.))
   residual_max=max(residual,default=float('inf'));rows.append({'angle':ang,'iterations':it,'active':active,'case':case,'success':c['success'],'repeat':c['repeat'],'contacts':c['contacts'],'impulse':sum(r['lambda'] for r in c['rows']),'momentum_error':momentum,'tangent_error':tang,'free_velocity_error':no_change,'matched_response_error':pair,'independent_normal_residual':residual_max,'solver_normal_residual':c['solver_residual'],'residual_disagreement':abs(residual_max-c['solver_residual'])})
 out[arm]={}
 for it in (2,32):
  subset=[r for r in rows if r['iterations']==it]
  out[arm][str(it)]={'cases':len(subset),'success':sum(r['success'] for r in subset),'repeat':sum(r['repeat'] for r in subset),'contacts':sum(r['contacts']>0 for r in subset),'positive_impulse_closing':sum(r['active'] and r['impulse']>1e-8 for r in subset),'free_velocity_failures':sum(not r['active'] and r['free_velocity_error']>1e-6 for r in subset),'normal_residual_failures':sum(r['independent_normal_residual']>.002 for r in subset),'max_momentum_error':max(r['momentum_error'] for r in subset),'max_tangent_error':max(r['tangent_error'] for r in subset),'max_matched_response_error':max(r['matched_response_error'] for r in subset),'max_normal_residual':max(r['independent_normal_residual'] for r in subset),'max_residual_disagreement':max(r['residual_disagreement'] for r in subset)}
 out[arm]['rows']=rows
(p/'analysis.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
for arm in out:print(arm,json.dumps({k:v for k,v in out[arm].items() if k!='rows'},sort_keys=True))
