from pathlib import Path
import gzip,json,math,subprocess
A=Path('artifacts/crow-v15-platform-arrival-state-audit-20261112');S=Path('artifacts/crow-v15-localized-route-autonomous-eval-20261111/autonomous-20261126');support=json.loads((S/'perch-analysis.json').read_text())['rows'];first={x['lane']:x['first_perch_foot_impulse_step'] for x in support}; parts=sorted(S.glob('run.stderr.gz.part[0-9][0-9][0-9]'));p=subprocess.Popen(['cat',*[str(x) for x in parts]],stdout=subprocess.PIPE);samples={}
with gzip.open(p.stdout,'rt') as z:
 for line in z:
  if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
  x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step'];t0=first.get(e)
  if t0 is not None and t in (t0-1,t0,t0+1,t0+25,t0+50,t0+100) and t<=support[e]['first_episode_steps']:samples[e,t]=x
assert p.wait()==0
rows=[]
for e in range(8):
 t=first[e];a=[]
 for step in (t-1,t,t+1,t+25,t+50,t+100):
  if (e,step) not in samples:continue
  x=samples[e,step];q=x['q'];v=x['v'];xx,yy,zz,w=q[3:7];roll=math.atan2(2*(w*xx+yy*zz),1-2*(xx*xx+yy*yy));pitch=math.asin(max(-1,min(1,2*(w*yy-zz*xx))));yaw=math.atan2(2*(w*zz+xx*yy),1-2*(yy*yy+zz*zz));a.append({'step':step,'root_position_m':q[:3],'root_linear_velocity_m_s':v[:3],'root_angular_velocity_rad_s':v[3:6],'roll_pitch_yaw_rad':[roll,pitch,yaw],'toe_mean_proximal_rad':[sum(q[i] for i in (23,25,27,29))/4,sum(q[i] for i in (31,33,35,37))/4]})
 rows.append({'lane':e,'first_perch_contact_step':t,'physical_failure':support[e]['physical_failure'],'last_25_supported_opposed':support[e]['last_25_half_weight_bilateral_opposed'],'observations':a})
r={'rows':rows,'scope':'Previously used seed/layout. Contact timing comes from full-substep perch audit; positions are world coordinates without platform-frame correction. State samples are control-step endpoints, not exact contact time; policy outputs are not applied torques. Correlation of approach states with outcome is not causal.'};(A/'analysis.json').write_text(json.dumps(r,indent=2)+'\n')
for x in rows:
 a=x['observations'][0];print(x['lane'],'failure',x['physical_failure'],'supported',x['last_25_supported_opposed'],'precontact_v',*[round(v,3) for v in a['root_linear_velocity_m_s']],'rpy',*[round(v,3) for v in a['roll_pitch_yaw_rad']])
