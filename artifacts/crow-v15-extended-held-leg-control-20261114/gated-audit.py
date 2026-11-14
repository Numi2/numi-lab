from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-extended-held-leg-control-20261114');O=Path('artifacts/crow-v15-release-gated-wing-20261114');D={}
for arm in ('push','hold'):
 x=json.load(gzip.open((O if arm=='push' else A)/(arm+'-window.json.gz'),'rt'));D[arm]={'state':{(v['lane'],v['step']):v for v in x['state']},'target':{tuple(v[:2]):v[2:] for v in x['target']},'errors':x['errors']}
 for k in ('contact','toe','leg','aero'):D[arm][k]={tuple(v[:3]):v[3:] for v in x[k]}
previous_hold=json.load(gzip.open(O/'hold-window.json.gz','rt'));previous_state={(v['lane'],v['step']):v for v in previous_hold['state']};previous_target={tuple(v[:2]):v[2:] for v in previous_hold['target']}
analysis=json.loads((A/'analysis-corrected.json').read_text());bounds=json.loads((O/'wing-limits-audit.json').read_text());br={(r['arm'],r['lane']):r for r in bounds['rows']};byarm={(r['arm'],r['lane']):r for r in analysis['rows']};rows=[]
for e in range(8):
 d=D['push'];h=D['hold'];p=byarm['push',e];c=byarm['hold',e];release=p['first_three_interval_unload'];keys=[(e,t,k) for t in range(371,release or 371) for k in range(16)];aero=sum(d['aero'][k][3]*.02/16 for k in keys) if keys and all(k in d['aero'] for k in keys) else None
 flap=max((abs(d['state'][e,t]['q'][j]-d['state'][e,370]['q'][j]) for t in range(371,release or 371) if (e,t) in d['state'] for j in (9,10)),default=None)
 ks=[(e,t,k) for t in range(371,377) for k in range(16)];delta=sum(sum(d['contact'][k][:2])-sum(h['contact'][k][:2]) for k in ks) if all(k in d['contact'] and k in h['contact'] for k in ks) else None
 nonfoot=sum(abs(d['contact'][e,t,k][4]) for t in range(371,501) for k in range(16) if (e,t,k) in d['contact']);caps=sum(max(map(abs,d['toe'][e,t,k]))>=11.99 for t in range(371,501) for k in range(16) if (e,t,k) in d['toe'])
 diffs={};missing=[]
 for t in range(335,501):
  a=d['target'].get((e,t));b=h['target'].get((e,t))
  if a is None or b is None or len(a)!=40 or len(b)!=40:missing.append(t);continue
  for i,(x,y) in enumerate(zip(a,b)):
   if x!=y:diffs.setdefault(i,[]).append(t)
 pre_nonleg=[(i,t) for i,ts in diffs.items() if i not in range(13,19) for t in ts if release is None or t<release];wingfirst=min((t for i,ts in diffs.items() if i in range(6,12) for t in ts),default=None)
 prior_bad=[t for t in range(1,411) if D['hold']['state'].get((e,t))!=previous_state.get((e,t))];target_bad=[t for t in range(335,411) if D['hold']['target'].get((e,t))!=previous_target.get((e,t))];extended_bad=[t for t in range(412,501) if any(abs(D['hold']['target'][e,t][13+i]-v)>1e-5 for i,v in enumerate((-.435,1.03,-.68,-.435,1.03,-.68)))]
 parity=analysis['parity'][e];limits=br['push',e];integrity=not prior_bad and not target_bad and not extended_bad and not parity['state_missing'] and not parity['state_mismatch'] and not missing and not pre_nonleg and not p['missing_or_nonfinite_substep_count'] and not c['missing_or_nonfinite_substep_count'] and not p['physical_failure'] and not c['physical_failure'] and analysis['shared_nonshader_inputs'] and not any(D[a]['errors'] for a in D)
 passed=integrity and release is not None and c['first_three_interval_unload'] is None and p['first_later_contact_or_missing'] is None and aero is not None and abs(aero)<=.005 and flap is not None and flap<=.01 and delta is not None and delta>0 and nonfoot<=1e-7 and caps==0 and not limits['accepted_state_limit_hit'] and not limits['accepted_velocity_limit_hit']
 rows.append({'lane':e,'prior_hold_state_mismatch':prior_bad,'prior_hold_target_mismatch':target_bad,'extended_hold_target_mismatch':extended_bad,'integrity_pass':integrity,'first_target_difference_by_dof':{str(i):ts[0] for i,ts in diffs.items()},'nonleg_command_differences_before_push_unload':pre_nonleg,'first_wing_command_difference':wingfirst,'push_first_unload':release,'push_first_recontact':p['first_later_contact_or_missing'],'hold_first_unload':c['first_three_interval_unload'],'push_minus_hold_foot_impulse_Ns':delta,'pre_unload_aero_impulse_Ns':aero,'pre_unload_flap_displacement_rad':flap,'nonfoot_impulse':nonfoot,'toe_speed_threshold_substeps':caps,'endpoint_wing_limits_hit':limits['accepted_state_limit_hit'] or limits['accepted_velocity_limit_hit'],'gated_handoff_screen_pass':passed})
out={'rows':rows,'limitations':'Push data comes from the earlier trial; the held arm extends the actual leg targets to interval 500. Release-gated wings deliberately cause different wing commands between arms after the push; this tests dependence on a control rule, not necessity under identical wing commands. Same development seed, correlated lanes, unaligned force/contact epochs and endpoint-only wing limits. No flight envelope or mission qualification.'};(A/'gated-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
