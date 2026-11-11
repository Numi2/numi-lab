from pathlib import Path
import gzip,json,subprocess,math,os
D=Path('artifacts/crow-v15-release-joint-audit-20261111');src=Path('artifacts/crow-v15-release-pitch-gentle-screen-20261111/trace.log.gz');desired={110,120,150,200,250};states={};poses={}
with gzip.open(src,'rt') as f:
 for l in f:
  a=l.split()
  if len(a)<4:continue
  if a[0]=='step' and int(a[1]) in (0,1,2,3,4,5) and int(a[2]) in desired:
   e,t=map(int,a[1:3]);i=a.index('q');j=a.index('v');states[e,t]=a[i+1:j]+a[j+1:]
  elif a[0]=='pose' and int(a[1]) in (0,1,2,3,4,5) and int(a[2]) in desired and a[3] in ('crow_left_foot','crow_right_foot'):
   poses[int(a[1]),int(a[2]),a[3]]=list(map(float,a[4:]))
s=''.join(f'{e} {t} '+ ' '.join(v)+'\n' for (e,t),v in sorted(states.items()))
p=subprocess.run(['build/crow-v15-landing-com-diagnostic-20261111/kinematics'],input=s,text=True,stdout=subprocess.PIPE,check=True,env={**os.environ,'DYLD_LIBRARY_PATH':str(Path.cwd()/'build/crow-v15-two-segment-substep-observer-20261111')})
com={};foot={};errs=[]
for l in p.stdout.splitlines():
 a=l.split();e,t=map(int,a[1:3]);
 if a[0]=='com':com[e,t]=list(map(float,a[4:7]))
 elif a[0]=='body' and a[3] in ('crow_left_foot','crow_right_foot'):
  name=a[3];v=list(map(float,a[4:]));foot[e,t,name]=v;ref=poses[e,t,name];errs.extend(abs(v[i]-ref[i]) for i in range(3))
rows=[]
for (e,t),c in sorted(com.items()):
 right=foot[e,t,'crow_right_foot'];left=foot[e,t,'crow_left_foot'];rows.append({'lane':e,'step_zero_based':t,'com':c,'right_foot':right[:3],'left_foot':left[:3],'com_minus_right_foot_y_m':c[1]-right[1],'com_minus_right_foot_x_m':c[0]-right[0]})
r={'max_reconstructed_foot_position_error_m':max(errs),'rows':rows,'interpretation':'Foot center is not a center of pressure or a support polygon. Lateral and fore-aft COM offsets describe geometry but do not prove the cause of the fall; release also changes several joint controls. The model audit has pitch-only hip, knee, ankle and toe joints, not an active leg abduction joint.'}
(D/'analysis.json').write_text(json.dumps(r,indent=2)+'\n')
for z in rows:print(z['lane'],z['step_zero_based'],'COM-right x,y',round(z['com_minus_right_foot_x_m'],4),round(z['com_minus_right_foot_y_m'],4))
print('max error',max(errs))
