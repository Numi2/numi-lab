from pathlib import Path
import gzip,json,math,sys
A=Path('artifacts/crow-v15-equal-launch-onset-20261113');seed=int(sys.argv[1]);assert seed in (20261126,20261127)
D=A/('straight-'+str(seed)+'-rebound-equal');refpath=(Path('artifacts/crow-v15-launch-aero-observer-20261113/observed-window.json.gz') if seed==20261126 else Path('artifacts/crow-v15-seed27-launch-attribution-20261113/push-window.json.gz'))
def load(p):
 j=json.load(gzip.open(p,'rt'));d={'state':{(x['lane'],x['step']):x for x in j['state']},'errors':j['errors']}
 for k,n in [('target',2),('contact',3),('toe',3),('leg',3),('aero',3)]:
  d[k]={tuple(x[:n]):x[n:] for x in j[k]};assert len(d[k])==len(j[k])
 return d
x=load(A/('straight-'+str(seed)+'-window.json.gz'));r=load(refpath);par=[]
for e in range(8):
 stop=500 if e<7 else 368;bad=[]
 for t in range(1,stop+1):
  a=x['state'].get((e,t));b=r['state'].get((e,t))
  if a is None or b is None or any(a[k]!=b[k] for k in ('q','v','policy_action','contacts')):bad.append(t)
 par.append({'lane':e,'through':stop,'mismatch_count':len(bad),'first_mismatches':bad[:5]})
rows=[]
for e in (6,7):
 missing={k:sum((e,t,i) not in x[k] for t in range(335,501) for i in range(16)) for k in ('contact','toe','leg')};miss_a=sum((e,t,i) not in x['aero'] for t in range(335,401) for i in range(16));sm=sum((e,t) not in x['state'] for t in range(1,501));tm=sum((e,t) not in x['target'] for t in range(335,501))
 def loaded(t):return any(sum(x['contact'][e,t,k][2:4])>1e-7 for k in range(16))
 first=next((t for t in range(371,501) if not loaded(t)),None) if not missing['contact'] else None
 three=next((t for t in range(371,499) if not any(loaded(u) for u in range(t,t+3))),None) if not missing['contact'] else None
 later=next((t for t in range(first+1,501) if loaded(t)),None) if first else None
 toe=sum(max(map(abs,x['toe'][e,t,i]))>=11.99 for t in range(371,501) for i in range(16)) if not missing['toe'] else None
 nonfoot=sum(abs(x['contact'][e,t,i][4]) for t in range(371,501) for i in range(16)) if not missing['contact'] else None
 aer=sum(x['aero'][e,t,i][3] for t in range(371,first) for i in range(16))*.02/16 if first is not None and first<=401 and not miss_a else None
 flap=max((abs(x['state'][e,t]['q'][q]-x['state'][e,370]['q'][q]) for t in range(371,first) for q in (9,10)),default=0.) if first else None
 rows.append({'lane':e,'missing_substeps':missing,'missing_launch_aero':miss_a,'missing_states':sm,'missing_targets':tm,'first_unloaded_interval':first,'first_three_unloaded_intervals':three,'first_recontact':later,'toe_speed_threshold_substeps':toe,'nonfoot_abs_impulse_Ns':nonfoot,'modeled_pre_release_aero_vertical_impulse_Ns':aer,'max_pre_release_flap_displacement_rad':flap})
inputs=json.loads((D/'inputs.json').read_text());refinputs=json.loads(((Path('artifacts/crow-v15-launch-aero-observer-20261113/low-straight-20261126-rebound-early-observer') if seed==20261126 else Path('artifacts/crow-v15-seed27-launch-attribution-20261113/push-20261127-rebound-seed27'))/'inputs.json').read_text());same=all(inputs[k]==refinputs[k] for k in ('host_sha256','frozen_flight_sha256','landing_policy_sha256','route_policy_sha256'));meta=json.loads((D/'run.stdout').read_text())['first_episode_navigation'];ok=same and not x['errors'] and all(a['mismatch_count']==0 for a in par) and all(meta['steps_by_environment'][e]==500 and not meta['physical_failure_by_environment'][e] for e in (6,7)) and all(a['first_unloaded_interval'] and a['first_three_unloaded_intervals'] and a['first_recontact'] is None and not any(a['missing_substeps'].values()) and a['missing_launch_aero']==a['missing_states']==a['missing_targets']==a['toe_speed_threshold_substeps']==0 and a['nonfoot_abs_impulse_Ns']<=1e-7 for a in rows)
out={'seed':seed,'scope':'paired development seed screen; not qualified flight or landing','unchanged_inputs_except_task_shader':same,'parity':par,'rows':rows,'observer_errors':x['errors'],'physical_failure_by_environment':meta['physical_failure_by_environment'],'passed_both':bool(ok)};(A/('analysis-'+str(seed)+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
