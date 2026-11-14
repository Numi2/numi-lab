from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-stance-paired-balance-20261114');src=Path('artifacts/crow-v15-stance-pitch-authority-20261114/trace.log.gz');data={}
with gzip.open(src,'rt') as f:
 for l in f:
  a=l.split()
  if a and a[0] in ('pitchcontrol','balance','event') and int(a[1]) in (0,1) and int(a[2]) in (260,265,270,275,280,290,300):data.setdefault(int(a[2]),{}).setdefault(int(a[1]),{})[a[0]]=list(map(float,a[3:]))
rows=[]
for t,x in sorted(data.items()):
 a=x[0];b=x[1];rows.append({'zero_based_tick':t,'candidate_minus_control_pitch_correction_rad':b['pitchcontrol'][-1]-a['pitchcontrol'][-1],'candidate_minus_control_body_pitch_rad':b['pitchcontrol'][2]-a['pitchcontrol'][2],'candidate_minus_control_com_x_m':b['balance'][2]-a['balance'][2],'candidate_minus_control_clearance_m':b['event'][10]-a['event'][10]})
out={'source':str(src),'rows':rows,'scope':'Paired, identical-prefix state association; other dynamic responses and contact changes can mediate it. Not proof of a stable intervention.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
