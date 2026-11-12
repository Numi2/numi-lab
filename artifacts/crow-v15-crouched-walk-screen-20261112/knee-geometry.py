from pathlib import Path
import importlib.util,math,json,statistics
p=Path('artifacts/crow-v15-crouched-walk-screen-20261112');sp=importlib.util.spec_from_file_location('a','artifacts/crow-v15-whole-gait-audit-20261111/audit.py');a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
paths={'held':Path('artifacts/crow-v15-small-support-stroke-20261112/sham-20261127-paired'),'candidate':Path('artifacts/crow-v15-crouched-walk-screen-20261112/candidate-20261127-rebound')}
def rot(q,v):
 x,y,z,w=q;u=(x,y,z);d=sum(u[i]*v[i] for i in range(3));c=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]);s=w*w-sum(x*x for x in u);return [s*v[i]+2*d*u[i]+2*w*c[i] for i in range(3)]
def point(b,v):
 q=rot(b[3:7],v);return [q[i]+b[i] for i in range(3)]
def angle(h,k,v):
 x=[h[i]-k[i] for i in range(3)];y=[v[i]-k[i] for i in range(3)];d=sum(x[i]*y[i] for i in range(3))/(math.sqrt(sum(i*i for i in x))*math.sqrt(sum(i*i for i in y)));return math.degrees(math.acos(max(-1,min(1,d))))
out={}
for label,path in paths.items():
 sup,sub,b,com=a.load(path); rows=[];errors=[];relation=[];allangles=[]
 for e in range(8):
  vals=[];loaded=[];unloaded=[]
  for t in range(201,301):
   for side,sd in enumerate(('left','right')):
    thigh=b[e,t,'crow_'+sd+'_thigh'];shank=b[e,t,'crow_'+sd+'_shank'];hip=point(thigh,[0,0,.0275*1.15]);knee=point(thigh,[0,0,-.0275*1.15]);kneecheck=point(shank,[0,0,.0275*1.15]);ankle=point(shank,[0,0,-.0275*1.15]);errors.append(math.dist(knee,kneecheck));v=angle(hip,knee,ankle);vals.append(v);allangles.append(v);q=sup[e,t]['q'][15 if side==0 else 18];relation.append(abs(v-(180-abs(math.degrees(q)))));load=sum(sub[e,t,k][2+side] for k in range(16));(loaded if load>1e-7 else unloaded).append(v)
  rows.append({'lane':e,'n':len(vals),'minimum_deg':min(vals),'maximum_deg':max(vals),'median_deg':statistics.median(vals),'interval_loaded_median_deg':statistics.median(loaded) if loaded else None,'interval_unloaded_median_deg':statistics.median(unloaded) if unloaded else None})
 out[label]={'rows':rows,'all_minimum_deg':min(allangles),'all_maximum_deg':max(allangles),'all_median_deg':statistics.median(allangles),'max_parent_child_anchor_error_m':max(errors),'max_angle_vs_revolute_error_deg':max(relation)}
out['reference']={'species':'black-billed magpie, Pica pica as named in source','doi':'10.1242/jeb.203.14.2159','walking_stance_knee_min_deg_mean_sd':[72,12],'walking_stance_n':26,'walking_recovery_knee_min_deg_mean_sd':[50,12],'walking_recovery_n':20,'walking_pre_stance_knee_max_deg_mean_sd':[143,9],'walking_pre_stance_n':34,'qualification':'Table 1 is means and standard deviations over observed events, not a population range. Different species, speed, surface, landmark estimation, stance/recovery phase and sample design; current full-window geometric extremes are not phase-matched estimates. Do not infer fit or choose physical limits from this comparison alone.'}
out['geometry']={'segment_local_half_length_m':.0275*1.15,'knee_configured_limits_rad':[-.1,1.55],'angle_at_positive_configured_limit_deg':180-math.degrees(1.55),'limit_scope':'Configured constraint limit from inherited source geometry. Solver tolerance and actual joint coordinate range are not tested by this geometric calculation.'}
out['scope']='800 endpoints per arm, both knees, world 3D angle between vectors hip-to-knee and ankle-to-knee from native FK and parent-child joint anchors. Not projected onto sagittal plane, filtered, or event-normalized like the published video. Interval load is normal impulse at any substep, not necessarily load at endpoint. FK anchor identity and q relation checked independently; source model geometry is estimated, not specimen data.'
(p/'knee-geometry.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
