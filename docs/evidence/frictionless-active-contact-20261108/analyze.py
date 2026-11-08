import json, math
from pathlib import Path
p=Path(__file__).parent
out={}
for arm in ('baseline','candidate'):
 data=json.loads((p/f'{arm}.json').read_text())['cases']
 assert len(data)==24
 groups={}
 for c in data:
  key=(c['angle'],c['speed'],c['iterations'])
  assert c['case'] not in groups.setdefault(key,{})
  groups[key][c['case']]=c
 assert len(groups)==8 and all(set(g)=={0,1,2} for g in groups.values())
 rows=[]
 for key,g in groups.items():
  a,s,it=key; ref=g[1]['v']; rx=-math.copysign(.2,a)*math.cos(a)+.02*math.sin(a)
  for i,c in g.items():
   v=c['v']; tangent=max(abs(v[0]-(i-1)),abs(v[2])); drift=max(abs(v[j]-ref[j]) for j in (1,3,4,5))
   corner=[v[1]+v[5]*rx-v[3]*z for z in (-.03,.03)]
   target=-c['gap']/.001
   rows.append({'angle':a,'speed':s,'iterations':it,'case':i,'success':c['success'],'repeat':c['repeat'],'contacts':c['contacts'],'normal_delta':v[1]-s,'tangent_error':tangent,'vertical_spin_abs':abs(v[4]),'paired_velocity_error':drift,'min_corner_v':min(corner),'target_v':target,'bound_violation':max(0.,target-min(corner))})
 out[arm]={}
 for it in (2,32):
  subset=[r for r in rows if r['iterations']==it]
  out[arm][str(it)]={'cases':len(subset),'success':sum(r['success'] for r in subset),'repeat':sum(r['repeat'] for r in subset),'contact':sum(r['contacts']>0 for r in subset),'active_response':sum(r['normal_delta']>1e-6 for r in subset),'max_tangent_error':max(r['tangent_error'] for r in subset),'max_vertical_spin':max(r['vertical_spin_abs'] for r in subset),'max_paired_velocity_error':max(r['paired_velocity_error'] for r in subset),'max_corner_bound_violation':max(r['bound_violation'] for r in subset),'bound_failed':sum(r['bound_violation']>.002 for r in subset)}
 out[arm]['rows']=rows
(p/'analysis.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
for arm in out:
 print(arm,json.dumps({k:v for k,v in out[arm].items() if k!='rows'},sort_keys=True))
