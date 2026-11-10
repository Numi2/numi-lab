from collections import defaultdict
from pathlib import Path
import json,math
A=Path('artifacts/crow-v15-balance-response-20261110');rows=defaultdict(lambda:defaultdict(lambda:{'frontL':0.,'frontR':0.,'rearL':0.,'rearR':0.,'up':0.,'other':0.}));states=defaultdict(dict);status={}
for line in (A/'trace.log').open():
 s=line.split()
 if not s:continue
 if s[0]=='case_result':status[int(s[2])]=' '.join(s[3:])
 elif s[0]=='trace':
  arm,t=map(int,s[1:3]);v=list(map(float,s[3:]));q=v[:7];vel=v[7:13];pitch=math.asin(max(-1,min(1,2*(q[6]*q[4]-q[5]*q[3]))));states[arm][t]={'x':q[0],'z':q[2],'pitch':pitch,'vx':vel[0],'vz':vel[2],'pitch_rate':vel[4],'hips':v[13:15]}
 elif s[0]=='contact':
  arm,t=int(s[2]),int(s[3]);name=s[4];imp=float(s[5]);nx=float(s[7]);nz=float(s[8]);r=rows[arm][t]
  if imp<=1e-8:continue
  side='L' if '_left_' in name else 'R' if '_right_' in name else '?'
  if '_toe' not in name and '_foot' not in name:r['other']+=imp
  if '_front_toe' in name and nx<-.5:r['front'+side]+=imp
  if '_rear_toe' in name and nx>.5:r['rear'+side]+=imp
  if ('_toe' in name or '_foot' in name) and nz<-.5:r['up']+=imp
out={}
for arm,delta in enumerate((-.06,-.03,0,.03,.06)):
 data=states[arm];streak=best=count=0
 for t in range(80):
  r=rows[arm][t];ok=all(r[k]>.00005 for k in ('frontL','frontR','rearL','rearR')) and r['up']>.011;streak=streak+1 if ok else 0;best=max(best,streak);count+=ok
 out[str(arm)]={'hip_target_delta_rad':delta,'status':status.get(arm),'sample_count':len(data),'start':data.get(0),'end':data.get(79),'opposed_supported_samples':count,'longest_opposed_supported_streak':best,'nonfoot_impulse_Ns':sum(r['other'] for r in rows[arm].values())}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for arm,x in out.items():
 a=x['start'];b=x['end'];print(arm,'delta',x['hip_target_delta_rad'],'status',x['status'],'n',x['sample_count'],'pitch change',round(b['pitch']-a['pitch'],5),'final rate',b['pitch_rate'],'x change',round(b['x']-a['x'],5),'opposed',x['opposed_supported_samples'],'max streak',x['longest_opposed_supported_streak'],'other',x['nonfoot_impulse_Ns'])
