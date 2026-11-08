from pathlib import Path
import json, math
p=Path(__file__).resolve().parent
s={a:json.loads((p/(a+'.json')).read_text())['cases'] for a in ('baseline','candidate')}
assert all(len(v)==18 for v in s.values())
out={}
for arm,cases in s.items():
 results=[]
 for c in cases:
  assert c['success'] and c['repeat'] and c['contacts']==len(c['rows'])==2
  v=c['v']; err=0; impulse=0; checks=[]
  for r in c['rows']:
   assert r['bodyA']==0 and r['bodyB']==1 and r['tangent']==[0,0]
   x,y,z=r['point'];y-=c['initial_y']; n=r['direction']; w=v[3:]
   cross=(w[1]*z-w[2]*y, w[2]*x-w[0]*z, w[0]*y-w[1]*x)
   vn=sum(n[j]*(v[j]+cross[j]) for j in range(3)); eq=vn-r['target']+r['reg']*r['lambda']
   residual=abs(eq) if r['lambda']>1e-8 else max(-eq,0.)
   err=max(err,residual);impulse+=r['lambda'];checks.append({'vn':vn,'equation':eq,'residual':residual})
  mismatch=abs(err-c['solver_residual']); momerr=abs((v[1]+1)-impulse)
  theta=c['angle'];ca=math.cos(theta);sa=math.sin(theta); wb=(ca*w[0]+sa*w[1],-sa*w[0]+ca*w[1],w[2])
  ix=(.02**2+.03**2)/3;iy=(.2**2+.03**2)/3;iz=(.2**2+.02**2)/3
  initialK=.5*((c['case']-1)**2+1); finalK=.5*(sum(x*x for x in v[:3])+ix*wb[0]**2+iy*wb[1]**2+iz*wb[2]**2)
  results.append({'angle':theta,'iterations':c['iterations'],'case':c['case'],'normal_residual':err,'reported':c['solver_residual'],'residual_discrepancy':mismatch,'momentum_error_Ns':momerr,'impulse_Ns':impulse,'energy_change_J':finalK-initialK,'vn_and_equation':checks})
 out[arm]=results
 print(arm,'max residual mismatch',max(r['residual_discrepancy'] for r in results),'max momentum error',max(r['momentum_error_Ns'] for r in results))
 print(arm,'max residual by iterations',[(it,max(r['normal_residual'] for r in results if r['iterations']==it)) for it in (2,8,32)])
 print(arm,'energy change bounds',min(r['energy_change_J'] for r in results),max(r['energy_change_J'] for r in results))
for arm,cases in s.items():
 worst=0
 for k in (0,9):
  for offset in (0,3,6):
   ref=cases[k+offset+1]['v']
   for c in (cases[k+offset],cases[k+offset+2]):
    worst=max(worst,*[abs(c['v'][j]-ref[j]) for j in (1,3,4,5)])
 print(arm,'max normal/angular response change under tangent variation',worst)
 out[arm+'_tangent_response_difference']=worst
assert all(x['residual_discrepancy']<2e-5 and x['momentum_error_Ns']<2e-5 for x in out['candidate'])
assert out['candidate_tangent_response_difference']<2e-5
assert max(x['normal_residual'] for x in out['candidate'] if x['iterations']==32)<.002
(p/'analysis.json').write_text(json.dumps(out,indent=2))
print('candidate analytic checks PASS')
