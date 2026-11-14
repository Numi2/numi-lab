from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-ankle-balance-diagnostic-20261114');src=Path('artifacts/crow-v15-ankle-clearance-20261114/trace.log.gz');data={}
with gzip.open(src,'rt') as f:
 for line in f:
  a=line.split()
  if a and a[0] in ('balance','pitchcontrol','event','step','rearAnkle','rearKnee','hipcommand') and a[1]=='1' and int(a[2]) in (220,230,240,250,260,270,280,290):data.setdefault(int(a[2]),{})[a[0]]=a[3:]
rows=[]
for t,d in sorted(data.items()):
 x=d['step'];qi=x.index('q');vi=x.index('v');q=list(map(float,x[qi+1:vi]));v=list(map(float,x[vi+1:]));b=list(map(float,d['balance']));p=list(map(float,d['pitchcontrol']))
 rows.append({'zero_based_tick':t,'height_m':q[2],'tilt_rad':math.acos(max(-1.,min(1.,1-2*(q[3]**2+q[4]**2)))),'root_velocity_m_s':v[:3],'com_relative_foot_x_m':b[2],'com_relative_foot_y_m':b[3],'com_relative_foot_vx_m_s':b[4],'com_relative_foot_vy_m_s':b[5],'pitch_requested_rad':p[-2],'pitch_applied_rad':p[-1],'hip_target_rad':float(d['hipcommand'][0]),'knee_offset_rad':float(d['rearKnee'][0]),'ankle_offset_rad':float(d['rearAnkle'][0])})
out={'source':str(src),'lane':1,'rows':rows,'scope':'Native model COM/accepted-state association only; contact/state epoch not independently aligned. Does not establish causality.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
