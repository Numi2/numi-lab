from pathlib import Path
import gzip,json,sys
reference=Path(sys.argv[1]);candidate=Path(sys.argv[2]);release=int(sys.argv[3]);out=Path(sys.argv[4])
with gzip.open(reference,'rt') as f:a=json.load(f)['payload']['frames']
with gzip.open(candidate,'rt') as f:b=json.load(f)['payload']['frames']
keys=('step','q','v','actor_observation','accepted_actions');diff=[];n=0
for x,y in zip(a,b):
 if x['step']>=release:break
 n+=1
 for k in keys:
  if x[k]!=y[k] and len(diff)<10:diff.append({'step':x['step'],'field':k})
r={'reference':str(reference),'candidate':str(candidate),'expected_frames':release-1,'checked_frames':n,'equal_prefix':n==release-1 and not diff,'first_differences':diff,'scope':list(keys)};out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
