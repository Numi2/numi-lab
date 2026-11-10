import gzip,json
from pathlib import Path
old='artifacts/crow-v15-bounded-rebound-dev-i-20261110/development-i-bounded-rebound.json.gz';new='artifacts/crow-v15-long-perch-i-20261110/development-i-long-perch.json.gz'
with gzip.open(old,'rt') as f:a=json.load(f)['payload']['frames']
with gzip.open(new,'rt') as f:b=json.load(f)['payload']['frames']
count=min(len(a),len(b),5000);diff=[];maxerr=0.0
for x,y in zip(a[:count],b[:count]):
 if x['step']!=y['step']:diff.append({'step':x['step'],'reason':'step mismatch'});continue
 for key in ('q','v'):
  if x[key]!=y[key]:
   delta=max(abs(u-v) for u,v in zip(x[key],y[key]));maxerr=max(maxerr,delta)
   if len(diff)<5:diff.append({'step':x['step'],'field':key,'max_abs_diff':delta})
result={'compared_frames':count,'fields':['step','q','v'],'exact_equal':not diff and count==5000,'max_abs_diff':maxerr,'first_differences':diff,'note':'Task identity and terminal flags differ because of episode horizon; compares accepted physical q/v only.'}
Path('artifacts/crow-v15-long-perch-i-20261110/prefix-parity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
