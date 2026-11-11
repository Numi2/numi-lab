from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-two-segment-gait-lift-screen-20261111');P=Path('artifacts/crow-v15-two-segment-gait-direction-screen-20261111');state={};ref={};sub=collections.defaultdict(lambda:[0.]*5);cover=collections.Counter();errors=[]
def read(p,target,withsub):
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);target[x['lane'],x['step']]=x
   elif withsub and l.startswith('CROW_TWO_TOE_SUBSTEP '):
    a=l.split();t,k,e=map(int,a[1:4]);cover[e,t,k]+=1
    for i in range(5):sub[e,t][i]+=float(a[6+i])
   elif withsub and 'failed' in l.lower():errors.append(l.strip())
read(P/'stderr.gz',ref,False);read(A/'stderr.gz',state,True);out=[]
for e in range(8):
 samples=[state.get((e,t)) for t in range(101,202)];complete=all(samples);diff=sum(state.get((e,t))!=ref.get((e,t)) for t in range(1,203)) if e%4==1 else None
 if complete:
  disp=samples[-1]['q'][0]-samples[0]['q'][0];speed=max(math.sqrt(sum(z*z for z in x['v'][:3])) for x in samples);tilt=max(math.acos(max(-1.,min(1.,1.-2*(x['q'][3]**2+x['q'][4]**2)))) for x in samples);air=sum(sub[e,t][2]+sub[e,t][3]<=1e-7 for t in range(101,202));nonfoot=sum(sub[e,t][4]>1e-7 for t in range(101,202));coverage=sum(cover[e,t,k]!=1 for t in range(101,202) for k in range(16))
 else:disp=speed=tilt=air=nonfoot=coverage=None
 out.append({'lane':e,'hip_coefficient':[.50,.30,.30,.30][e%4],'knee_coefficient':[.80,.40,.60,.80][e%4],'complete':complete,'baseline_sample_differences':diff,'forward_displacement_m':disp,'maximum_speed_m_s':speed,'maximum_tilt_rad':tilt,'unsupported_intervals':air,'nonfoot_impulse_intervals':nonfoot,'missing_substeps':coverage,'pass':bool(complete and disp>=.015 and speed<=.5 and tilt<=.35 and air==0 and nonfoot==0 and coverage==0 and (diff is None or diff==0))})
try:
 data=json.loads((A/'stdout').read_text());failed=data['failed_environment_steps'];termination=data['termination_count']
except Exception as ex:failed=None;termination=None;errors.append(str(ex))
for row in out:row['pass']=bool(row['pass'] and failed==0 and termination==0 and not errors)
report={'lanes':out,'failed_environment_steps':failed,'termination_count':termination,'errors':errors[-5:],'scope':'development directional screen; no full foot geometry qualification'};(A/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
