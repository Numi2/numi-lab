import copy,json,math
from pathlib import Path
prior=json.loads(Path('trace.json').read_text())['cases'];cases=json.loads(Path('work-trace.json').read_text())['cases']
control=json.loads(Path('work-control.json').read_text())['cases'];prior_control=json.loads(Path('control.json').read_text())['cases']
assert len(prior)==len(cases)==len(control)==len(prior_control)==60
out=[];max_pre_error=0.;unchanged=0
for p,c,old_ctrl,ctrl in zip(prior,cases,prior_control,control):
 for baseline,new in ((p,c),(old_ctrl,ctrl)):
  tmp=copy.deepcopy(new)
  for row in tmp['rows']:row.pop('pre_relative')
  assert tmp==baseline
 unchanged+=1
 key={k:c[k] for k in ('iterations','mu','initial_vn','dir')};normal=tangent=0.;rows=[]
 for r in c['rows']:
  imp=r['impulse'];pre=r['pre_relative'];post=r['post_relative']
  assert all(math.isfinite(x) for x in [*imp,*pre,*post])
  w=[imp[i]*(pre[i]+post[i])*.5 for i in range(3)]
  normal+=w[0];tangent+=w[1]+w[2];rows.append({'normal_J':w[0],'tangent_J':w[1]+w[2]})
  if not (r['flags']&8) and r['body_a']==0 and r['body_b']==1:
   initial=([1.,0.,0.],[-1.,0.,0.],[0.,0.,1.],[0.,0.,-1.],[0.,0.,0.])[c['dir']][:];initial[1]=c['initial_vn']
   n=r['normal'];t=r['tangent'];u=[n[1]*t[2]-n[2]*t[1],n[2]*t[0]-n[0]*t[2],n[0]*t[1]-n[1]*t[0]]
   reconstructed=[sum(a*b for a,b in zip(initial,v)) for v in (n,t,u)]
   err=max(abs(x-y) for x,y in zip(reconstructed,pre));max_pre_error=max(max_pre_error,err)
   assert err<1e-5,(key,err,pre,reconstructed)
 out.append({**key,'normal_J':normal,'tangential_J':tangent,'total_J':normal+tangent,'rows':rows})
result={'cases':len(cases),'unchanged_control_and_trace_cases':unchanged,'max_sphere_pre_velocity_error_m_s':max_pre_error,'normal_J_range':[min(x['normal_J'] for x in out),max(x['normal_J'] for x in out)],'tangential_J_range':[min(x['tangential_J'] for x in out),max(x['tangential_J'] for x in out)],'total_J_range':[min(x['total_J'] for x in out),max(x['total_J'] for x in out)],'work_by_case':out}
Path('work-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='work_by_case'},indent=2))
