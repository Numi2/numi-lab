from pathlib import Path
import gzip,json,statistics,math
p=Path('artifacts/crow-v15-common-command-pronation-screen-20261112'); samples={e:[] for e in range(8)}
with gzip.open(p/'candidate-20261126-rebound-flexed/run.stderr.gz','rt') as f:
 for l in f:
  if l.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);e=x['lane'];t=x['step']
   if 650<=t<=1200:samples[e].append((x['q'][11],x['q'][12]))
result=[]
for e,x in samples.items():
 assert len(x)==551,(e,len(x))
 l=[z[0] for z in x];r=[z[1] for z in x]
 result.append({'lane':e,'mean_left_rad':statistics.mean(l),'mean_right_rad':statistics.mean(r),'mean_q_difference_rad':statistics.mean(a-b for a,b in x),'left_range_rad':[min(l),max(l)],'right_range_rad':[min(r),max(r)],'near_left_limit_count':sum(abs(z)>.34 for z in l),'near_right_limit_count':sum(abs(z)>.34 for z in r),'samples':len(x)})
(p/'joint-response.json').write_text(json.dumps({'interval_s':[13,24],'source':'accepted 50 Hz endpoint coordinates; action target saturation is not recorded','rows':result},indent=2)+'\n')
for x in result:print(x)
