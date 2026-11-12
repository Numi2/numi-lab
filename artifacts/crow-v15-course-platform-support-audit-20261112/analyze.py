from pathlib import Path
import collections, gzip, json, subprocess
A=Path('artifacts/crow-v15-localized-route-autonomous-eval-20261111/autonomous-20261126')
D=Path('artifacts/crow-v15-course-platform-support-audit-20261112')
first=json.loads((A/'run.stdout').read_text())['first_episode_navigation']
ends=first['steps_by_environment']
parts=sorted(A.glob('run.stderr.gz.part[0-9][0-9][0-9]'))
# Stream archived parts, without restoring a large raw log.
p=subprocess.Popen(['cat',*[str(x) for x in parts]],stdout=subprocess.PIPE)
sums=collections.defaultdict(lambda:[0.0]*7)
with gzip.open(p.stdout,'rt') as f:
 for line in f:
  if not line.startswith('CROW_SCENE_CONTACT '): continue
  z=line.split(); t,k,e,ba,bb=map(int,z[1:6])
  if e>=len(ends) or t>ends[e]+1 or 40 not in (ba,bb): continue
  b=bb if ba==40 else ba; sign=1. if ba==40 else -1.
  normal,tan1,tan2=map(float,z[6:9]); nx,ny,nz=map(float,z[13:16]);tx,ty,tz=map(float,z[16:19]); vz=nx*ty-ny*tx
  side=0 if b==11 or 19<=b<=26 else 1 if b==15 or 27<=b<=34 else -1
  x=sums[e,t]
  if side<0: x[6]+=normal; continue
  x[side]+=sign*(nz*normal+tz*tan1+vz*tan2)
  x[2+side]+=normal; x[4+side]+=(tan1*tan1+tan2*tan2)**0.5
if p.wait()!=0: raise SystemExit('archive stream failed')
def longest(xs):
 best=run=0
 for x in xs:
  run=run+1 if x else 0; best=max(best,run)
 return best
out=[]; half=.5*.45006*9.81*.02
for e,end in enumerate(ends):
 x=[sums[e,t] for t in range(1,end+1)]
 supported=[v[0]+v[1]>=half and v[6]<=1e-7 for v in x]
 bilat=[s and v[2]>1e-7 and v[3]>1e-7 for s,v in zip(supported,x)]
 single=[s and ((v[2]>1e-7) != (v[3]>1e-7)) for s,v in zip(supported,x)]
 final=x[-25:]
 out.append({'lane':e,'episode_steps':end,'waypoints':first['maximum_waypoints_by_environment'][e], 'physical_failure':first['physical_failure_by_environment'][e], 'max_supported_intervals':longest(supported),'max_bilateral_supported_intervals':longest(bilat),'max_single_foot_supported_intervals':longest(single),'last_25_supported_intervals':sum(supported[-25:]),'last_25_nonfoot_contact_intervals':sum(v[6]>1e-7 for v in final),'last_25_mean_upward_impulse_Ns':sum(v[0]+v[1] for v in final)/len(final),'last_25_foot_normal_impulses_Ns':[sum(v[2+i] for v in final) for i in range(2)],'last_25_foot_tangent_impulse_magnitudes_Ns':[sum(v[4+i] for v in final) for i in range(2)]})
r={'source':'crow-v15-localized-route-autonomous-eval-20261111/autonomous-20261126','platform_half_width_m':.10,'mass_kg':.45006,'control_interval_s':.02,'support_threshold':'upward foot impulse >= half body weight over interval, no nonfoot normal impulse','interpretation':'Development seed and layout. Impulses measured over full physics substeps; tangential magnitudes are summed per contact and cannot establish slip, friction coefficient or grip. These data do not test intentional toe release, takeoff, flight-controller qualification or held-out layouts. Consult source audit for contact coverage and toe velocity caps.','rows':out}
(D/'analysis.json').write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r,indent=2))
