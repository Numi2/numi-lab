from pathlib import Path
import collections,gzip,json,math,statistics
A=Path('artifacts/crow-v15-landing-hover-calibration-20261111');R=Path('artifacts/crow-v15-v1-toe-contact-memory-20261111');
def samples(p):
 states={};pSub={};clear=[];fatal=[]
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);states[x['step']]=x
   elif l.startswith('CROW_PERCH_SUBSTEP '):
    x=l.split();pSub[tuple(map(int,x[1:4]))]=(int(x[4]),int(x[5]),list(map(float,x[6:])))
   elif l.startswith('CROW_CLEARANCE_SAMPLE '):clear.append(json.loads(l.split(' ',1)[1]))
   elif l.startswith('CROW_CONTACT_TRACE '):fatal.append(json.loads(l.split(' ',1)[1]))
 return states,pSub,clear,fatal
reference,_,_,_=samples(R/'route.stderr.gz');out={}
for name in ('o18','o21','o24','o27'):
 s,sub,c,f=samples(A/(name+'.stderr.gz'));j=json.loads((A/(name+'.stdout')).read_text());diff=[t for t in range(1,1306) if t not in s or t not in reference or any(s[t][k]!=reference[t][k] for k in ('q','v'))];missing=[(t,k) for t in range(1,2201) for k in range(16) if (t,k,0) not in sub];overflow=[k for k,(count,cap,v) in sub.items() if count>cap or len(v)!=21];
 with gzip.open(A/(name+'.json.gz'),'rt') as stream:r=json.load(stream)['payload']
 frame=r['frames'][0];idx=r['body_names'].index('crow_course_perch')*r['body_state_stride'];perch=frame['body_states'][idx:idx+3];target=(perch[0]-.035,perch[1]+.55);rows=[s[t] for t in range(1900,2201) if t in s];valid=not(diff or missing or overflow) and len(rows)==301 and j['termination_count']==0 and j['failed_environment_steps']==0
 errors=[(x['q'][0]-target[0],x['q'][1]-target[1]) for x in rows];qerr=[math.hypot(*v) for v in errors];vx=[x['v'][0] for x in rows];tilt=[math.acos(max(-1.,min(1.,1.-2.*(x['q'][3]**2+x['q'][4]**2)))) for x in rows];z=[x['q'][2] for x in rows];touch=any(sum(v[2:4])+v[4]>1e-7 for t,(_,_,v) in sub.items() if t[0]>1305)
 out[name]={'valid':valid,'pre_waypoint_5_q_v_differences':len(diff),'missing_perch_substeps':len(missing),'perch_overflows':len(overflow),'termination_count':j['termination_count'],'failed_environment_steps':j['failed_environment_steps'],'perch_contact_seen':touch,'target_xy_m':target,'window_samples':len(rows),'mean_x_error_m':statistics.mean(v[0] for v in errors) if rows else None,'mean_y_error_m':statistics.mean(v[1] for v in errors) if rows else None,'planar_rms_error_m':math.sqrt(statistics.mean(v*v for v in qerr)) if rows else None,'max_planar_error_m':max(qerr) if rows else None,'mean_forward_velocity_m_s':statistics.mean(vx) if rows else None,'rms_forward_velocity_m_s':math.sqrt(statistics.mean(v*v for v in vx)) if rows else None,'max_tilt_rad':max(tilt) if rows else None,'mean_height_m':statistics.mean(z) if rows else None,'minimum_sampled_obstacle_clearance':c[-1] if c else None,'first_fatal_contact':f[0] if f else None}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for n,x in out.items():print(n,{k:x[k] for k in ('valid','pre_waypoint_5_q_v_differences','termination_count','perch_contact_seen','mean_x_error_m','mean_y_error_m','planar_rms_error_m','rms_forward_velocity_m_s','max_tilt_rad')})
