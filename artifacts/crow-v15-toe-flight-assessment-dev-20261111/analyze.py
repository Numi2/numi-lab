from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-perch-strong-balance-screen-20261111');O=Path('artifacts/crow-v15-toe-flight-assessment-dev-20261111')
a=json.loads((A/'analysis.json').read_text());end={x['lane']:x['first_scene_contact']['step']-1 for x in a['lanes']}
r=[{'lane':e,'start_step':356,'end_step':end[e],'max_abs_toe_speed_rad_s':0.,'toe_values_ge_5_rad_s':0,'toe_values_ge_10_rad_s':0,'toe_samples':0,'min_endpoint_toe_angle_rad':float('inf'),'max_endpoint_toe_angle_rad':-float('inf'),'angle_samples':0,'endpoint_values_outside_joint_limits':0} for e in range(8)]
for p in sorted((A/'trial').glob('stderr-*.gz')):
 with gzip.open(p,'rt') as f:
  for line in f:
   if line.startswith('CROW_TOE_V_SUBSTEP '):
    x=line.split();t=int(x[1]);e=int(x[3]);
    if 356<=t<=end[e]:
     v=list(map(float,x[4:]));assert len(v)==16
     row=r[e];row['max_abs_toe_speed_rad_s']=max(row['max_abs_toe_speed_rad_s'],max(map(abs,v)));row['toe_values_ge_5_rad_s']+=sum(abs(z)>=5 for z in v);row['toe_values_ge_10_rad_s']+=sum(abs(z)>=10 for z in v);row['toe_samples']+=len(v)
   elif line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);e=x['lane'];t=x['step']
    if 356<=t<=end[e]:
     q=x['q'][23:39];assert len(q)==16
     row=r[e];row['min_endpoint_toe_angle_rad']=min(row['min_endpoint_toe_angle_rad'],min(q));row['max_endpoint_toe_angle_rad']=max(row['max_endpoint_toe_angle_rad'],max(q));row['endpoint_values_outside_joint_limits']+=sum(z<-.3001 or z>1.2001 for z in q);row['angle_samples']+=len(q)
for row in r:assert row['toe_samples']==(row['end_step']-355)*16*16
(O/'assessment.json').write_text(json.dumps({'source':'artifacts/crow-v15-perch-strong-balance-screen-20261111','rows':r,'scope':'Teacher-guided development starts, first episode only, from post-handoff step 356 through control interval preceding first observed course contact. Speeds are all substeps; joint angles are control-endpoint samples. The configured toe limits are -0.3 to 1.2 rad and 12 rad/s; being within limits does not establish natural toe posture, contact-free flight on other starts, or biological realism.'},indent=2)+'\n')
print([(x['lane'],round(x['max_abs_toe_speed_rad_s'],3),x['toe_values_ge_10_rad_s'],x['endpoint_values_outside_joint_limits']) for x in r])
