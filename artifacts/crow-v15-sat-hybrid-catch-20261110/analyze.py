from collections import defaultdict
import json
from pathlib import Path
A=Path('artifacts/crow-v15-sat-hybrid-catch-20261110'); rows=defaultdict(lambda:defaultdict(lambda:{'frontL':0.,'frontR':0.,'rearL':0.,'rearR':0.,'up':0.,'other':0.,'toe':None,'z':None,'vz':None}));status={}
for line in (A/'trace.log').open():
 s=line.split()
 if not s: continue
 if s[0]=='case_result':status[(int(s[1]),int(s[2]))]=' '.join(s[3:])
 elif s[0]=='trace':
  cs,arm,t=map(int,s[1:4]);r=rows[cs,arm][t];r['z']=float(s[5]);r['vz']=float(s[6]);r['toe']=list(map(float,s[7:15]))
 elif s[0]=='contact':
  cs,arm,t=map(int,s[1:4]);name=s[4];imp=float(s[5]);nx=float(s[7]);nz=float(s[8]);r=rows[cs,arm][t]
  if imp<=1e-8:continue
  side='L' if '_left_' in name else 'R' if '_right_' in name else '?'
  if '_toe' not in name and '_foot' not in name:r['other']+=imp
  if '_front_toe' in name and nx<-.5:r['front'+side]+=imp
  if '_rear_toe' in name and nx>.5:r['rear'+side]+=imp
  if ('_toe' in name or '_foot' in name) and nz<-.5:r['up']+=imp
out={}
for cs in range(4):
 for arm in range(3):
  data=rows[cs,arm];streak=longest=0;first=None;other=0.;supported_after=0
  for t,r in sorted(data.items()):
   if t<100:
    ok=all(r[k]>.00005 for k in ('frontL','frontR','rearL','rearR')) and r['up']>.011
    streak=streak+1 if ok else 0;longest=max(longest,streak)
    if ok and first is None:first=t
    other+=r['other']
   elif r['up']>.011 and abs(r['vz'])<.12:supported_after+=1
  last=data[max(data)] if data else {};out[f'{cs}/{arm}']={'status':status.get((cs,arm)),'max_opposed_bilateral_supported_streak':longest,'first_qualified_step':first,'nonfoot_impulse_before_release':other,'supported_postrelease_samples':supported_after,'end_z':last.get('z'),'end_vz':last.get('vz'),'end_toe_max_abs':max(map(abs,last.get('toe') or [0]))}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for arm in range(3):
 print('arm',arm,[(cs,out[f'{cs}/{arm}']) for cs in range(4)])
