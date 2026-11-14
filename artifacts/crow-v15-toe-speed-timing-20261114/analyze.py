from pathlib import Path
import gzip,json
R=Path('artifacts');out=[]
for name in ['crow-v15-return-toe-closure-rate-20261113','crow-v15-stance-pitch-support-20261114']:
 trace=R/name/'trace.log.gz';lanes={0:[],1:[]};fall={}
 with gzip.open(trace,'rt') as f:
  for line in f:
   x=line.split()
   if not x or x[0]!='step':continue
   e,t=int(x[1]),int(x[2]);qi=x.index('q');vi=x.index('v');q=list(map(float,x[qi+1:vi]));v=list(map(float,x[vi+1:]));
   if q[2]<.1 and e not in fall:fall[e]=t
   if max(abs(v[j-1]) for j in range(23,39))>=11.99:lanes[e].append(t)
 for e in (0,1):out.append({'trial':name,'lane':e,'first_root_below_0_1_zero_based':fall.get(e),'threshold_steps_zero_based':lanes[e],'all_threshold_steps_after_fall':bool(lanes[e]) and fall.get(e) is not None and min(lanes[e])>fall[e]})
p=Path(__file__).resolve().parent/'timing.json';p.write_text(json.dumps({'rows':out,'scope':'accepted control states only; substep values unobserved'},indent=2)+'\n');print(p.read_text())
