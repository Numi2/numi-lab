from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-retract-hold-fix-20261114');D={}
for arm in ('push','hold'):
 x=json.load(gzip.open(A/(arm+'-window.json.gz'),'rt'));D[arm]={'state':{(v['lane'],v['step']):v for v in x['state']},'errors':x['errors']}
 for k in ('contact','toe','leg','aero'):D[arm][k]={tuple(v[:3]):v[3:] for v in x[k]}
analysis=json.loads((A/'analysis-corrected.json').read_text());cmd=json.loads((A/'all-command-audit.json').read_text());byarm={(x['arm'],x['lane']):x for x in analysis['rows']};rows=[]
for e in range(8):
 d=D['push'];h=D['hold'];p=byarm['push',e];c=byarm['hold',e];release=p['first_three_interval_unload'];pred=lambda t:(e,t) in d['state'];
 aero_keys=[(e,t,k) for t in range(371,release or 371) for k in range(16)];aero=sum(d['aero'][k][3]*.02/16 for k in aero_keys) if aero_keys and all(k in d['aero'] for k in aero_keys) else None
 flap=max((abs(d['state'][e,t]['q'][j]-d['state'][e,370]['q'][j]) for t in range(371,release or 371) if pred(t) for j in (9,10)),default=None)
 contact_keys=[(e,t,k) for t in range(371,377) for k in range(16)];delta=sum(sum(d['contact'][k][:2])-sum(h['contact'][k][:2]) for k in contact_keys) if all(k in d['contact'] and k in h['contact'] for k in contact_keys) else None
 nonfoot=sum(abs(d['contact'][e,t,k][4]) for t in range(371,501) for k in range(16) if (e,t,k) in d['contact']);caps=sum(max(map(abs,d['toe'][e,t,k]))>=11.99 for t in range(371,501) for k in range(16) if (e,t,k) in d['toe']);par=analysis['parity'][e];cc=cmd['rows'][e];integrity=not par['state_missing'] and not par['state_mismatch'] and not par['wing_mismatch_or_missing'] and cc['only_leg_dofs_differ'] and cc['wing_tail_toe_equal'] and not cc['missing_or_bad_target_records'] and not p['missing_or_nonfinite_substep_count'] and not c['missing_or_nonfinite_substep_count'] and not p['physical_failure'] and not c['physical_failure'] and analysis['shared_nonshader_inputs'] and not any(D[a]['errors'] for a in D)
 transition=integrity and release is not None and p['first_later_contact_or_missing'] is None and aero is not None and abs(aero)<=.005 and flap is not None and flap<=.01 and delta is not None and delta>0 and nonfoot<=1e-7 and caps==0
 held_sustained=c['first_three_interval_unload'] is not None and c['first_later_contact_or_missing'] is None
 rows.append({'lane':e,'leg_extension_duration_seconds':.16,'airborne_retraction':'enabled_after_release' if e%2 else 'held_control','wing_onset_seconds':7.50,'integrity_pass':integrity,'push_first_three_unloaded':release,'push_first_recontact':p['first_later_contact_or_missing'],'hold_first_three_unloaded':c['first_three_interval_unload'],'hold_first_recontact':c['first_later_contact_or_missing'],'modeled_vertical_aero_impulse_pre_unload_Ns':aero,'max_flap_displacement_pre_unload_rad':flap,'push_minus_hold_upward_foot_impulse_371_376_Ns':delta,'push_nonfoot_impulse':nonfoot,'push_toe_speed_threshold_substeps':caps,'continuous_transition_screen_pass':transition,'matched_held_leg_sustained_unload':held_sustained,'necessity_screen_pass':transition and not held_sustained})
out={'rows':rows,'scope':'used-seed onset sweep with matched non-leg targets; substep contact and state epochs remain independently unverified; not biological or mission qualification'};(A/'transition-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
