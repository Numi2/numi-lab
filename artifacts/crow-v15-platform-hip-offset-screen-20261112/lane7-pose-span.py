from pathlib import Path
import subprocess,gzip,json,math,statistics
A=Path('artifacts/crow-v15-platform-hip-offset-screen-20261112');D=A/'autonomous-20261126-posture';p=subprocess.Popen(['cat',*[str(x) for x in sorted(D.glob('run.stderr.gz.part[0-9][0-9][0-9]'))]],stdout=subprocess.PIPE);rows=[]
with gzip.open(p.stdout,'rt') as f:
 for l in f:
  if l.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);t=x['step']
   if x['lane']==7 and 3800<=t<=4800:
    q=x['q'];v=x['v'];tilt=math.acos(max(-1,min(1,1-2*(q[3]*q[3]+q[4]*q[4]))));speed=math.sqrt(sum(z*z for z in v[:3]));rows.append({'step':t,'tilt_rad':tilt,'root_speed_m_s':speed,'vertical_speed_m_s':v[2]})
assert p.wait()==0 and len(rows)==1001
r={'scope':'First-episode endpoint samples only, not continuous substep extrema; used development start and old flight law.','samples':len(rows),'max_tilt_rad':max(x['tilt_rad'] for x in rows),'max_root_speed_m_s':max(x['root_speed_m_s'] for x in rows),'p95_root_speed_m_s':sorted(x['root_speed_m_s'] for x in rows)[int(.95*(len(rows)-1))],'samples_over_tilt_0_35':sum(x['tilt_rad']>.35 for x in rows),'samples_over_speed_0_12':sum(x['root_speed_m_s']>.12 for x in rows),'rows':rows}
(A/'lane7-pose-span.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='rows'})
