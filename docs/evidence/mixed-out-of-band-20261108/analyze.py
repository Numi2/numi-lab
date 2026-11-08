from pathlib import Path
import json,math
p=Path(__file__).resolve().parent
names=('ordinary','witness','forced','old')
data={n:json.loads((p/(n+'.stdout')).read_text()) for n in names}
data['guard']=json.loads((p/'guard-2.stdout').read_text())
def exposed(x):
 if isinstance(x,dict):return {k:exposed(v) for k,v in x.items() if k not in ('scratch','row_trace')}
 if isinstance(x,list):return [exposed(v) for v in x]
 return x
def diffs(a,b,path=''):
 if isinstance(a,dict) and isinstance(b,dict):
  for k in sorted(set(a)|set(b)):
   if k not in ('scratch','row_trace'):yield from diffs(a.get(k),b.get(k),path+'/'+k)
 elif isinstance(a,list) and isinstance(b,list):
  if len(a)!=len(b):yield [path+'/length',len(a),len(b)]
  for i,(x,y) in enumerate(zip(a,b)):yield from diffs(x,y,path+'/'+str(i))
 elif a!=b:yield [path,a,b]
def comparison(a,b):
 ds=list(diffs(a,b));nums=[abs(x[1]-x[2]) for x in ds if all(isinstance(y,(int,float)) and not isinstance(y,bool) and math.isfinite(y) for y in x[1:])]
 return {'exact':len(ds)==0,'difference_count':len(ds),'max_numeric_difference_mixed_units':max(nums,default=0),'first_differences':ds[:20]}
events=[]
for i,s in enumerate(data['guard']['steps']):
 rods=[r for r in s['rows'] if r['flags']&8];rigids=[r for r in s['rows'] if not r['flags']&8]
 if s['code']==0 and len(rods)==2 and len(rigids)==1:
  idx=[tuple(int(x) for x in r['scratch'][:3]) for r in rods]
  valid=all(all(float(int(x))==x for x in r['scratch'][:3]) for r in rods)
  ns=max(x for pair in idx for x in pair[:2])-min(x for pair in idx for x in pair[:2])
  es=max(x[2] for x in idx)-min(x[2] for x in idx)
  marker=[r['scratch'][3] for r in s['rows']]
  events.append({'step':i,'indices':idx,'node_span':ns,'edge_span':es,'markers':marker,'valid_indices':valid,'qualifies':valid and (ns>2 or es>1) and marker==[8,8,8]})
summary={n:{k:d[k] for k in ('success','failed','repeat','exact_trace','capture_noninvasive')} for n,d in data.items()}
out={'runs':summary,'guard_vs_ordinary':comparison(data['ordinary'],data['guard']),'witness_vs_ordinary':comparison(data['ordinary'],data['witness']),'ordinary_vs_forced':comparison(data['ordinary'],data['forced']),'ordinary_vs_old':comparison(data['ordinary'],data['old']),'steps_with_guard_denied':events,'qualified_guard_step_count':sum(e['qualifies'] for e in events),'strict_guard_gate':bool(events) and any(e['qualifies'] for e in events) and exposed(data['ordinary'])==exposed(data['guard']) and all(data['guard'][k] for k in ('repeat','exact_trace','capture_noninvasive'))}
(p/'analysis.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='steps_with_guard_denied'},indent=2))
if not out['strict_guard_gate']:raise SystemExit(1)
