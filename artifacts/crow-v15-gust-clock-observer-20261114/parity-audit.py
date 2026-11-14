from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-gust-clock-observer-20261114');paths=[Path('artifacts/crow-v15-airborne-gust-episode-clock-20261114/turn-20261126-rebound-gustclock/run.stderr.gz'),A/'turn-20261126-rebound-clockaudit/run.stderr.gz'];data=[]
for p in paths:
 d={'state':{},'wrench':{},'duplicates':[]}
 with gzip.open(p,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);k=(x['lane'],x['step']);typ='state';val=x
   elif line.startswith('CROW_PRE_AERO_ROOT '):
    x=line.split();k=tuple(map(int,(x[3],x[1],x[2])));typ='wrench';val=x[4:]
   else:continue
   if k in d[typ]:d['duplicates'].append([typ,k])
   d[typ][k]=val
 data.append(d)
out={}
for typ in ('state','wrench'):
 a,b=data[0][typ],data[1][typ];keys=set(a)|set(b);bad=[k for k in keys if k not in a or k not in b or a[k]!=b[k]];out[typ]={'previous_count':len(a),'replay_count':len(b),'mismatch_or_missing_count':len(bad),'first_mismatches':sorted(bad)[:10]}
out['duplicates']=[x['duplicates'] for x in data];out['exact_full_state_and_wrench_parity']=not any(out['duplicates']) and all(out[k]['mismatch_or_missing_count']==0 for k in ('state','wrench')) and out['state']['previous_count']==14408 and out['wrench']['previous_count']==8*16*16;out['scope']='matched replay under two observational hosts, same used seed and same physical disturbance; not an independent robustness trial';(A/'turn-20261126-rebound-clockaudit'/'parity-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
