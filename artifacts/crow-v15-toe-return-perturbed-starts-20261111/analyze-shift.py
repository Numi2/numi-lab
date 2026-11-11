from pathlib import Path
import gzip,subprocess,json
D=Path('artifacts/crow-v15-toe-return-perturbed-starts-20261111');desired={80,120,160,180,200,250,300,400,500};states={};poses={}
with gzip.open(D/'trace.log.gz','rt') as f:
 for l in f:
  a=l.split()
  if len(a)<4 or a[0] not in ('step','pose'):continue
  e,t=int(a[1]),int(a[2])
  if t not in desired:continue
  if a[0]=='step':
   i=a.index('q');j=a.index('v');states[e,t]=a[i+1:j]+a[j+1:]
  elif a[0]=='pose' and a[3] in ('crow_left_foot','crow_right_foot'):poses[e,t,a[3]]=list(map(float,a[4:7]))
s=''.join(f'{e} {t} '+' '.join(v)+'\n' for (e,t),v in sorted(states.items()));p=subprocess.run(['build/crow-v15-hip-roll-filtered-prototype-20261111/kinematics'],input=s,text=True,stdout=subprocess.PIPE,check=True)
com={};foot={};err=[]
for l in p.stdout.splitlines():
 a=l.split();e,t=map(int,a[1:3]);
 if a[0]=='com':com[e,t]=list(map(float,a[4:7]))
 elif a[0]=='foot':
  name=a[3];v=list(map(float,a[4:7]));foot[e,t,name]=v;ref=poses[e,t,name];err.extend(abs(v[i]-ref[i]) for i in range(3))
rows=[]
for (e,t),c in sorted(com.items()):
 r=foot[e,t,'crow_right_foot'];rows.append({'lane':e,'step_zero_based':t,'com':c,'right_foot':r,'com_minus_support_foot_x_m':c[0]-r[0],'com_minus_support_foot_y_m':c[1]-r[1]})
r={'max_reconstructed_foot_position_error_m':max(err) if err else None,'rows':rows,'limitations':'Foot center is not the center of pressure or support polygon; observations at sampled endpoints do not establish stability or causation.'};(D/'shift-analysis.json').write_text(json.dumps(r,indent=2)+'\n')
for z in rows:
 if z['step_zero_based'] in (80,160,180,250):print(z['lane'],z['step_zero_based'],round(z['com_minus_support_foot_y_m'],5),round(z['com_minus_support_foot_x_m'],5))
print('max error',max(err) if err else None)
