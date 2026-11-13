from pathlib import Path
import gzip,json
R=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts')
A=R/'crow-v15-wing-presolve-audit-20261113';A.mkdir(exist_ok=True)
p=R/'crow-v15-aero-wrench-observer-20261113/push-wing-20261126-rebound-aero/run.stderr.gz'
d={};pre={}
with gzip.open(p,'rt') as f:
 for l in f:
  z=l.split()
  if z and z[0] in ('CROW_PRESOLVE_V','CROW_KIN_STATE') and int(z[1])==369 and 2<=int(z[3])<=7:
   e,k=int(z[3]),int(z[2]);key=(e,k)
   if z[0]=='CROW_PRESOLVE_V':
    assert key not in pre;pre[key]=list(map(float,z[5:]))[8:10]
   else:
    assert key not in d;nq=int(z[4]);d[key]=list(map(float,z[6+nq:]))[8:10]
rows=[]
for e in range(2,8):
 for k in range(3):
  for j in range(2):
   a=d[e,k][j];b=pre[e,k][j];c=d[e,k+1][j];rows.append({'lane':e,'step':369,'substep':k,'wing':j,'start_velocity_rad_s':a,'precontact_velocity_rad_s':b,'next_start_velocity_rad_s':c,'precontact_delta_rad_s':b-a,'post_presolve_delta_rad_s':c-b})
out={'source':'parity-validated aero observer, used seed 20261126; first three substeps at the prior speed peak, selected after observing the peak','scope':'Pre-contact velocity change combines native actuation, aero, gravity, other forces and numerical integration; next-start minus pre-contact includes contact solve and other post-presolve effects. Not a calibrated biological limit or an attribution to any single force.','rows':rows};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
(A/'RESULT.md').write_text('''# Where the observed flap-speed transient develops\n\nIn the earlier parity-validated smooth-release push run, inspect step 369, substeps 0–2, lanes 2–7, both flap DOFs. This interval was selected after the peak was observed and is descriptive, not a preregistered test. Each flap gained 8.79–12.96 rad/s between its recorded substep start and pre-contact velocity. The subsequent pre-contact-to-next-substep-start difference was at most 0.049 rad/s in magnitude across these samples. The sample period is 1.25 ms.\n\nThus, for this window, the major increase is already present before contact resolution; the subsequent change is comparatively small. This does not distinguish drive torque, aerodynamic force, gravity, other pre-contact effects, or numerical integration, and does not establish physiological validity. The underlying full traces, observer provenance and parity checks remain in ../crow-v15-aero-wrench-observer-20261113/.\n''')
print(len(rows),'records',min(x['precontact_delta_rad_s'] for x in rows),max(x['precontact_delta_rad_s'] for x in rows),max(abs(x['post_presolve_delta_rad_s']) for x in rows))
