from pathlib import Path
import collections,gzip,json
A=Path('artifacts/crow-v15-posthop-wing-phase-20261113');O=Path('artifacts/crow-v15-intermediate-winghold-20261113');D={}
for arm in ('push','hold'):
 d={'state':{},'contact':{},'toe':{},'aero':{},'kin':{},'target':{}};D[arm]=d
 with gzip.open(A/(arm+'-wing-20261126-rebound-posthop/run.stderr.gz'),'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);d['state'][x['lane'],x['step']]=x;continue
   z=l.split()
   if not z:continue
   if z[0]=='CROW_DRIVE_TARGET':
    t,e=map(int,z[1:3]);d['target'][e,t]=list(map(float,z[3:]));continue
   if z[0] not in ('CROW_KIN_STATE','CROW_AERO_FORCE','CROW_TWO_TOE_SUBSTEP','CROW_TOE_V_SUBSTEP'):continue
   t,k,e=map(int,z[1:4]);key=e,t,k
   if not 348<=t<=400:continue
   if z[0]=='CROW_KIN_STATE':d['kin'][key]=list(map(float,z[47:]))
   elif z[0]=='CROW_AERO_FORCE':d['aero'][key]=float(z[7])
   elif z[0]=='CROW_TWO_TOE_SUBSTEP':d['contact'][key]=list(map(float,z[6:11]))
   else:d['toe'][key]=max(abs(float(x)) for x in z[4:])
main=json.loads((A/'analysis.json').read_text());mr={(x['arm'],x['lane']):x for x in main['rows']};rows=[]
for e in range(2,8):
 d=D['push']; h=D['hold']; base=d['target'][e,370];first=next((t for t in range(371,401) if any(abs(d['target'][e,t][j]-base[j])>1e-7 for j in range(6,12))),None); r=mr['push',e]['release']; mismatch=[t for t in range(1,373) if any(d['state'][e,t][k]!=h['state'][e,t][k] for k in ('q','v','policy_action','contacts'))]
 keys=[(e,t,k) for t in range(348,first or 401) for k in range(16)];complete=all(q in d[x] for q in keys for x in ('contact','toe','aero','kin')) and all((e,t,k) in h['contact'] for t in range(371,(r or 400)+1) for k in range(16));flap=max((abs(d['kin'][q][j]) for q in keys for j in (8,9)),default=None);toe=sum(d['toe'][q]>=11.99 for q in keys);other=sum(d['contact'][q][4] for q in keys);aero=sum(d['aero'][q] for q in keys)*.00125;delta=sum(sum(d['contact'][e,t,k][2:4])-sum(h['contact'][e,t,k][2:4]) for t in range(371,(r or 370)+1) for k in range(16))
 row={'lane':e,'first_non_neutral_wing_target_tick':first,'release_first_of_three_ticks':r,'com_vz_at_release_m_s':mr['push',e]['com_vz_at_release_m_s'],'prepush_state_mismatches':mismatch,'prewing_observation_complete':complete,'flap_speed_before_wing_target_rad_s':flap,'toe_threshold_substeps_before_wing_target':toe,'nonfoot_impulse_before_wing_target_ns':other,'aero_vertical_impulse_before_wing_target_ns':aero,'matched_push_minus_hold_normal_impulse_to_release_ns':delta}
 row['leg_release_before_wing_pass']=bool(complete and not mismatch and r is not None and first is not None and r+2<first and row['com_vz_at_release_m_s'] is not None and row['com_vz_at_release_m_s']>0 and flap is not None and flap<=1 and toe==0 and other<=1e-7 and abs(aero)<=.02 and delta>0);rows.append(row)
# Check early accepted-state identity against previously archived neutral-wing hop.
parity=[]
for arm in ('push','hold'):
 before={}
 with gzip.open(O/(arm+'-wing-20261126-rebound-intermediate/run.stderr.gz'),'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);
    if x['lane'] in (2,3):before[x['lane'],x['step']]=x
 for e in (2,3):
  end=next(x['first_non_neutral_wing_target_tick'] for x in rows if x['lane']==e)-1
  missing=[t for t in range(1,end+1) if (e,t) not in before or (e,t) not in D[arm]['state']]
  bad=[t for t in range(1,end+1) if (e,t) in before and (e,t) in D[arm]['state'] and any(before[e,t][k]!=D[arm]['state'][e,t][k] for k in ('q','v','policy_action','contacts'))]
  parity.append({'arm':arm,'lane':e,'through_tick':end,'missing':missing,'mismatches':bad})
valid=main['valid_batch'] and all(not x['missing'] and not x['mismatches'] for x in parity);combined={str(e)+'-'+str(e+1):valid and main['paired_push_screen_pass'][str(e)+'-'+str(e+1)] and all(next(x for x in rows if x['lane']==k)['leg_release_before_wing_pass'] for k in (e,e+1)) for e in (2,4,6)};out={'valid':valid,'rows':rows,'reference_hop_parity':parity,'paired_complete_exploratory_screen_pass':combined};(A/'leg-release-check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
