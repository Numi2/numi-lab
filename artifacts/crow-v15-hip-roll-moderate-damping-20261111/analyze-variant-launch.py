from pathlib import Path
import gzip,json,collections,math,sys
A=Path('artifacts/crow-v15-hip-roll-moderate-damping-20261111')/sys.argv[1];cmd=json.loads((A/'command.json').read_text());nsub=int(cmd[cmd.index('--physics-substeps')+1]);sup={};sub={};caps=collections.defaultdict(list);coverage=collections.Counter();vel={}
with gzip.open(A/'run.stderr.gz','rt') as f:
 for l in f:
  if l.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);sup[x['lane'],x['step']]=x
  elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
   a=l.split();t,k,e=map(int,a[1:4]);sub[e,t,k]=list(map(float,a[6:]));coverage[e,t,k]+=1
  elif l.startswith('CROW_TOE_V_SUBSTEP '):
   a=l.split();t,k,e=map(int,a[1:4]);v=list(map(float,a[4:]));vel[e,t,k]=v
   if max(map(abs,v))>=11.99:caps[e].append((t,k))
def impulse(e,t):
 arr=[sub.get((e,t,k)) for k in range(nsub)]
 return None if any(x is None for x in arr) else [sum(x[i] for x in arr) for i in range(5)]
rows=[]
for e in range(8):
 first=next((t for t in range(348,400) if all((e,u) in sup and impulse(e,u) is not None and sum(impulse(e,u)[2:5])<=1e-7 and sup[e,u]['v'][2]>0 for u in (t,t+1))),None)
 recontact=next((t for t in range(first+2,501) if impulse(e,t) is not None and sum(impulse(e,t)[2:5])>1e-7),None) if first else None
 cap={name:sum(a<=t<=b for t,k in caps[e]) for name,a,b in [('settle',1,200),('walk',201,301),('launch',302,399),('after_400',400,500)]}
 q=sup.get((e,500),{}).get('q');v=sup.get((e,500),{}).get('v');rows.append({'lane':e,'first_two_unloaded_rising_interval':first,'first_ground_recontact':recontact,'height_gain_347_to_500_m':q[2]-sup[e,347]['q'][2] if q else None,'forward_speed_at_500_m_s':v[0] if v else None,'tilt_at_500_rad':math.acos(max(-1.,min(1.,1-2*(q[3]**2+q[4]**2)))) if q else None,'toe_speed_cap_substeps_by_phase':cap,'last_cap':caps[e][-1] if caps[e] else None,'missing_or_duplicate_substeps':sum(coverage[e,t,k]!=1 for t in range(1,501) for k in range(nsub)),'missing_velocity_substeps':sum((e,t,k) not in vel for t in range(1,501) for k in range(nsub))})
out={'rows':rows,'limitations':'Wing-on only, teacher-guided development starts; unloaded intervals do not prove toe geometry clearance or leg causality. Velocity samples are end-of-substep, not within-substep maxima. No flight recovery, release, supported landing or autonomous course conclusion.'};(A/'launch-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(rows,indent=2))
