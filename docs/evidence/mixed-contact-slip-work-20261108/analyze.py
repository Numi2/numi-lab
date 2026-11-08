"""Preregistered mixed-contact direction and instrumentation checks."""
import copy,json,math
from pathlib import Path
b=json.loads(Path('control.json').read_text())['cases']
t=json.loads(Path('trace.json').read_text())['cases']
assert len(b)==len(t)==60,(len(b),len(t))
def vdot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
checks={'instrument_equivalence_failures':[],'replay_failures':[],'bad_status':[],'bad_momentum':[],
 'invalid_cones':[],'frictionless_tangent_impulses':[],'positive_tangent_work':[],
 'normal_residual_exceedances':[],'sliding_direction_failures_64':[],
 'sphere_independent_velocity_mismatch':[]}
maxima={'positive_tangent_work_J':0.,'normal_residual_m_s':0.,'sphere_velocity_mismatch_m_s':0.,'angle_error_rad_64':0.,'relative_cone_bound_error_64':0.}
sliding_rows=0;active_friction_rows=0
for a,c in zip(b,t):
 key={k:c[k] for k in ('iterations','mu','initial_vn','dir')}
 plain=copy.deepcopy(c)
 for row in plain['rows']:row.pop('post_relative')
 orig=copy.deepcopy(a)
 for row in orig['rows']:row.pop('post_relative')
 if plain!=orig:checks['instrument_equivalence_failures'].append(key)
 if not c['replay']:checks['replay_failures'].append(key)
 if not c['accepted'] or c['status']!=0 or not c['mixed_island'] or c['rod_contacts']!=2 or c['rigid_contacts']!=1:checks['bad_status'].append(key)
 if c['candidate_momentum_error_uninterpretable_on_rejection']>2e-5:checks['bad_momentum'].append(key)
 normal=c['normal_residual'];maxima['normal_residual_m_s']=max(maxima['normal_residual_m_s'],normal)
 if not math.isfinite(normal) or normal>.002:checks['normal_residual_exceedances'].append({**key,'residual':normal})
 for i,r in enumerate(c['rows']):
  n,u,v=r['impulse'];vel=r['post_relative'];all_values=[n,u,v,*vel,*r['normal'],*r['tangent']]
  if not all(math.isfinite(x) for x in all_values):checks['invalid_cones'].append({**key,'row':i,'nonfinite':True});continue
  tang=math.hypot(u,v);cone=c['mu']*n
  if n < -1e-8 or tang>cone+1e-7:checks['invalid_cones'].append({**key,'row':i,'normal':n,'tangent':tang,'limit':cone})
  if c['mu']==0 and tang>1e-7:checks['frictionless_tangent_impulses'].append({**key,'row':i,'tangent':tang})
  if c['mu']>0:
   if n>1e-8:active_friction_rows+=1
   signed=u*vel[1]+v*vel[2];maxima['positive_tangent_work_J']=max(maxima['positive_tangent_work_J'],signed)
   if signed>1e-6:checks['positive_tangent_work'].append({**key,'row':i,'value_J':signed})
   speed=math.hypot(vel[1],vel[2])
   if c['iterations']==64 and speed>1e-3 and tang>1e-5:
    sliding_rows+=1
    cosine=max(-1.,min(1.,-(u*vel[1]+v*vel[2])/(tang*speed)))
    angle=math.acos(cosine);bound=abs(tang-cone)/max(cone,1e-30)
    maxima['angle_error_rad_64']=max(maxima['angle_error_rad_64'],angle)
    maxima['relative_cone_bound_error_64']=max(maxima['relative_cone_bound_error_64'],bound)
    if angle>.01 or bound>.01:checks['sliding_direction_failures_64'].append({**key,'row':i,'angle_rad':angle,'relative_bound_error':bound,'speed_m_s':speed,'impulse':tang})
  if not (r['flags']&8) and r['body_a']==0 and r['body_b']==1:
   state=c['final_v'];rot=cross(state[3:6],r['anchor_b']);point=[state[j]+rot[j] for j in range(3)]
   nvec=r['normal'];tan=r['tangent'];binormal=cross(nvec,tan)
   independent=[vdot(point,nvec),vdot(point,tan),vdot(point,binormal)]
   err=max(abs(x-y) for x,y in zip(independent,vel));maxima['sphere_velocity_mismatch_m_s']=max(maxima['sphere_velocity_mismatch_m_s'],err)
   if err>1e-5:checks['sphere_independent_velocity_mismatch'].append({**key,'row':i,'error_m_s':err,'independent':independent,'reported':vel})
summary={'case_count':len(t),'row_count':sum(len(x['rows']) for x in t),'active_friction_rows':active_friction_rows,'sliding_rows_64_checked':sliding_rows,'maxima':maxima,'counts':{k:len(v) for k,v in checks.items()},'checks':checks}
Path('analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='checks'},indent=2))
