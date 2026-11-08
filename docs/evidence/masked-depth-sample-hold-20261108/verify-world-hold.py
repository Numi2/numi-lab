from array import array
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parent;data=Path(sys.argv[1]);envs=2;stride=38526;pixels=19200
def load(s):
 a=array('f');a.frombytes((data/('run1.'+s)).read_bytes());return a
inputs=load('inputs');actor=load('actor')
r=array('I');r.frombytes((data/'run1.reset').read_bytes())
log=(root/'run1.log').read_text();errors={int(i):int(e) for i,e in re.findall(r'transition\[(\d+)\] done=\d+ timeout=\d+ physics_error=(\d+)',log)}
assert len(errors)==60
checked=0;visible=0;by_status={'accepted':0,'rejected':0};bad=[]
for step in range(1,30):
 if step%5==0:continue
 for env in range(envs):
  i=step*envs+env
  if r[i] or errors[i-envs]:continue
  a=((i-envs)*stride)+102;b=i*stride+102
  old=actor[a:a+pixels];new=inputs[b:b+pixels]
  checked+=1;visible+=int(any(x<1.0 for x in old));by_status['rejected' if errors[i] else 'accepted']+=1
  if old!=new:bad.append({'step':step,'env':env,'different':sum(x!=y for x,y in zip(old,new))})
result={'checked_contiguous_accepted_predecessor_nonsampled_observations':checked,'visible_rows':visible,'current_outcomes':by_status,'different_rows':bad,'passed':checked>0 and visible>0 and not bad,'note':'No reset and previous step accepted; comparisons use the most recent visible actor depth frame versus the next policy input. Frame cadence is batch global.'}
(root/'world-hold-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
