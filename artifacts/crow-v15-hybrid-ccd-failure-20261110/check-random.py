import math,json,gzip
from pathlib import Path
A=Path('artifacts/crow-v15-hybrid-ccd-failure-20261110')
s=(A/'sat.py').read_text();exec(s[s.index('def dot'):s.index('def pose')])
def oracle(x,y):
 h,c,q=x[:3],x[3:6],x[6:10];k,d,r=y[:3],y[3:6],y[6:10];q=[v/math.sqrt(dot(q,q)) for v in q];r=[v/math.sqrt(dot(r,r)) for v in r];u,v=axes(q),axes(r);delta=[a-b for a,b in zip(c,d)];best=-1e9
 for w in u+v+[cross(a,b) for a in u for b in v]:
  n=math.sqrt(dot(w,w))
  if n<1e-10:continue
  w=[a/n for a in w];g=abs(dot(delta,w))-sum(z*abs(dot(a,w)) for z,a in zip(h,u))-sum(z*abs(dot(a,w)) for z,a in zip(k,v));best=max(best,g)
 return best
results={'cases':0,'strict_overlap_count':0,'strict_separation_count':0,'false_separation_strict_1e-5':0,'false_overlap_strict_1e-4':0,'bad_status':0,'max_pair_order_difference':0.,'max_gpu_minus_oracle':-1e9,'max_abs_difference':0.,'examples':[]}
with gzip.open(A/'random-sat.tsv.gz','rt') as f:
 for line in f:
  a=list(map(float,line.split()));i=int(a[0]);x=a[1:11];y=a[11:21];g,rev,ok,ok2=a[21:];o=oracle(x,y);results['cases']+=1;results['strict_overlap_count']+=o < -1e-5;results['strict_separation_count']+=o>1e-4;bad=o < -1e-5 and (g>0 or rev>0);results['false_separation_strict_1e-5']+=bad;results['false_overlap_strict_1e-4']+=o>1e-4 and (g<=0 or rev<=0);results['bad_status']+=ok!=1 or ok2!=1;results['max_pair_order_difference']=max(results['max_pair_order_difference'],abs(g-rev));results['max_gpu_minus_oracle']=max(results['max_gpu_minus_oracle'],g-o,rev-o);results['max_abs_difference']=max(results['max_abs_difference'],abs(g-o),abs(rev-o))
  if bad and len(results['examples'])<5:results['examples'].append({'index':i,'oracle':o,'gpu':g,'reversed':rev})
(A/'random-analysis.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
