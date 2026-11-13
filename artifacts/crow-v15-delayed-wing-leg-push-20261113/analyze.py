from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-delayed-wing-leg-push-20261113');D={}
for arm in ('push','hold'):
 with gzip.open(A/(arm+'-window.json.gz'),'rt') as f:j=json.load(f)
 d={'errors':j['errors'],'aero_count':len(j['aero'])}
 d['state']={(x['lane'],x['step']):x for x in j['state']}
 for key,n in (('target',2),('contact',3),('toe',3),('leg',3),('aero',3)):
  d[key]={tuple(x[:n]):x[n:] for x in j[key]};assert len(d[key])==len(j[key])
 D[arm]=d
keys=json.loads((A/'fk-keys.json').read_text());vals={};h=[]
for line in (A/'fk-results.tsv').read_text().splitlines():
 z=line.split()
 if z[0]=='C':
  idx=int(z[1]);v=list(map(float,z[3:]));assert len(v)==7 and all(map(math.isfinite,v));vals[idx]=v
 else:h.append(z[0])
assert h==['543203944cce27056406a8c9f3a8a9f55016f404ef310c11dbfa497c72e83aaf','5c278558f4c9ae3cac66b2db510b864e324e313d75f11b605602433f9494d063']
assert len(vals)==len(keys)+2
translation=max(abs((vals[len(keys)+1][4+j]-vals[len(keys)][4+j])-[.3,-.2,.1][j]) for j in range(3));assert translation<1e-8
com={tuple(k):vals[i] for i,k in enumerate(keys)}
parity=[]
for e in range(8):
 stop=500 if e<6 else 368;bad=[]
 for t in range(1,stop+1):
  a=D['push']['state'].get((e,t));b=D['hold']['state'].get((e,t));
  if a is None or b is None or any(a[k]!=b[k] for k in ('q','v','policy_action','contacts')):bad.append(t)
 parity.append({'lane':e,'through':stop,'mismatch_count':len(bad),'first_mismatches':bad[:5]})
rows=[]
for e in (6,7):
 r={'lane':e};targets=[]
 for t in range(371,401):
  a=D['push']['target'].get((e,t));b=D['hold']['target'].get((e,t));
  if a and b:targets.extend(abs(a[i]-b[i]) for i in (8,9,10,11))
 r['wing_target_comparison_samples']=len(targets);r['wing_target_max_abs_difference_rad']=max(targets,default=None)
 r['leg_target_snapshots']={str(t):{arm:d['target'][e,t][13:19] for arm,d in D.items()} for t in (370,375,380,390)};r['max_actual_flap_position_difference_371_400_rad']=max(abs(D['push']['state'][e,t]['q'][j]-D['hold']['state'][e,t]['q'][j]) for t in range(371,401) for j in (9,10));r['max_actual_flap_velocity_difference_371_400_rad_s']=max(abs(D['push']['state'][e,t]['v'][j]-D['hold']['state'][e,t]['v'][j]) for t in range(371,401) for j in (8,9))
 for arm,d in D.items():
  cover={k:sum((e,t,s) not in d[k] for t in range(335,501) for s in range(16)) for k in ('contact','toe','leg')};target_missing=sum((e,t) not in d['target'] for t in range(335,501));state_missing=sum((e,t) not in d['state'] for t in range(1,501));
  def loaded(t):return any(sum(d['contact'][e,t,s][2:4])>1e-7 for s in range(16))
  release=next((t for t in range(371,499) if not any(loaded(u) for u in range(t,t+3))),None);recontact=next((t for t in range(release+3,501) if loaded(t)),None) if release else None
  impulse=sum(sum(d['contact'][e,t,s][:2]) for t in range(371,391) for s in range(16));nonfoot=sum(abs(d['contact'][e,t,s][4]) for t in range(371,501) for s in range(16));caps=sum(max(map(abs,d['toe'][e,t,s]))>=11.99 for t in range(371,501) for s in range(16));base=com[arm,e,370];end=com[arm,e,390];peak=max((com[arm,e,t][3]-base[3],t) for t in range(371,501));release_vz=com[arm,e,release][6] if release else None
  aero_missing=sum((e,t,k) not in d['aero'] for t in range(335,401) for k in range(16)); aero_impulse=sum(d['aero'][e,t,k][3] for t in range(371,378) for k in range(16))*(.02/16) if not aero_missing else None; flap_delta=max(abs(d['state'][e,t]['q'][j]-d['state'][e,370]['q'][j]) for t in range(371,378) for j in (9,10)); r[arm]={'missing_aero_substeps_335_400':aero_missing,'modeled_aero_vertical_impulse_371_377_Ns':aero_impulse,'max_actual_flap_displacement_371_377_rad':flap_delta,'missing_substeps':cover,'missing_target_steps':target_missing,'missing_states':state_missing,'first_three_unloaded_intervals':release,'first_recontact':recontact,'foot_normal_impulse_371_390_Ns':impulse,'nonfoot_abs_impulse_371_500_Ns':nonfoot,'toe_threshold_substeps_371_500':caps,'com_vz_step370_m_s':base[6],'com_vz_step390_m_s':end[6],'com_peak_rise_371_500_m_and_tick':peak,'com_vz_at_release_m_s':release_vz,'modeled_mass_kg':base[0]}
 r['push_minus_hold_foot_impulse_Ns']=r['push']['foot_normal_impulse_371_390_Ns']-r['hold']['foot_normal_impulse_371_390_Ns'];r['push_minus_hold_com_vz_step390_m_s']=r['push']['com_vz_step390_m_s']-r['hold']['com_vz_step390_m_s'];rows.append(r)
meta={arm:json.loads((A/(arm+'-20261126-rebound-delay/run.stdout')).read_text())['first_episode_navigation'] for arm in D}
out={'scope':'same used seed and delayed-wing paired extension versus held crouch; not flight or biological qualification','parity':parity,'rows':rows,'observer_errors':{k:v['errors'] for k,v in D.items()},'aero_wrench_records_335_500':{k:v['aero_count'] for k,v in D.items()},'com_velocity_translation_calibration_max_error':translation,'fk_executable_and_library_sha256':h,'first_episode_steps':{a:m['steps_by_environment'] for a,m in meta.items()},'physical_failure':{a:m['physical_failure_by_environment'] for a,m in meta.items()}};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
