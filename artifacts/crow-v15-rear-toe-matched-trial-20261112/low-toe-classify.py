from pathlib import Path
import importlib.util,json,collections,math
p=Path('artifacts/crow-v15-rear-toe-matched-trial-20261112')
s=importlib.util.spec_from_file_location('a','artifacts/crow-v15-whole-gait-audit-20261111/audit.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
paths={'held':p/'candidate-20261127-rebound-control','candidate':p/'candidate-20261127-rebound-preload'}
output={}
for label,path in paths.items():
 sup,sub,b,com=a.load(path);rows=[]
 for e in range(8):
  count=collections.Counter()
  for t in range(201,301):
   for side,side_name in enumerate(('left','right')):
    gap=[]
    for n in a.names(side_name):
     z=b[e,t,n];x,y,q,w=z[3:7];r=(2*(x*q-y*w),2*(y*q+x*w),1-2*(x*x+y*y));h=(.010,.01725,.0092) if n.endswith('_foot') else (.009,.0025,.0092);gap.append(z[2]-sum(abs(i)*j for i,j in zip(r,h)))
    if not (gap[0]>.001 and min(gap[1:])<=.001):continue
    count['low_all']+=1
    toe=[sum(sub[e,t,k][5+side*8:13+side*8]) for k in range(16)];foot=[sub[e,t,k][2+side] for k in range(16)]
    if max(toe)>1e-7:count['toe_loaded_somewhere_in_interval']+=1
    elif max(foot)>1e-7:count['sole_loaded_only_in_interval']+=1
    else:count['foot_unloaded_entire_interval']+=1
    if toe[-1]>1e-7:count['toe_loaded_final_substep']+=1
    if foot[-1]<=1e-7:count['foot_unloaded_final_substep']+=1
  assert count['low_all']==count['toe_loaded_somewhere_in_interval']+count['sole_loaded_only_in_interval']+count['foot_unloaded_entire_interval']
  rows.append({'lane':e,**dict(count)})
 output[label]=rows
output['scope']='Geometry is at each endpoint; contact load categories use the same 20 ms interval or its final recorded substep, not synchronized continuous toe clearance. Low geometric clearance alone is not evidence of dragging. A loaded toe with a lifted sole can be normal push-off; an unloaded low toe may be nearing or leaving ground. Neither this grouping nor endpoint point speeds alone measures integrated slip.'
(p/'low-toe-classification.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
