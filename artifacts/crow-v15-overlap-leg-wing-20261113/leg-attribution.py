from pathlib import Path
import gzip,json,subprocess,os,math
R=Path.cwd();A=R/'artifacts/crow-v15-overlap-leg-wing-20261113';D={}
for arm in ('push','hold'):
 d={'state':{},'contact':{},'toe':{},'aero':{},'target':{}};D[arm]=d
 with gzip.open(A/(arm+'-wing-20261126-rebound-overlap/run.stderr.gz'),'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);d['state'][x['lane'],x['step']]=x;continue
   z=l.split()
   if not z:continue
   if z[0]=='CROW_DRIVE_TARGET':
    t,e=map(int,z[1:3]);d['target'][e,t]=list(map(float,z[3:]));continue
   if z[0] not in ('CROW_AERO_FORCE','CROW_TWO_TOE_SUBSTEP','CROW_TOE_V_SUBSTEP'):continue
   t,k,e=map(int,z[1:4]);key=e,t,k
   if not 348<=t<=500:continue
   if z[0]=='CROW_AERO_FORCE':d['aero'][key]=float(z[7])
   elif z[0]=='CROW_TWO_TOE_SUBSTEP':d['contact'][key]=list(map(float,z[6:11]))
   else:d['toe'][key]=max(abs(float(x)) for x in z[4:])
main=json.loads((A/'analysis.json').read_text());mr={(x['arm'],x['lane']):x for x in main['rows']}; ids={};inputs=[]; info={}
for e in range(2,8):
 p,h=D['push'],D['hold'];first=next((t for t in range(348,501) if p['target'][e,t][13:19]!=h['target'][e,t][13:19]),None);release=mr['push',e]['release'];info[e]=(first,release)
 if first is not None and release is not None:
  for arm in ('push','hold'):
   for t in (first-1,release):
    x=D[arm]['state'][e,t];ids[len(inputs)]=(arm,e,t);inputs.append(str(len(inputs))+' 0 '+' '.join(map(str,x['q']+x['v'])))
exe=R/'build/crow-v15-com-momentum-audit-20261113/kinematics';lib=R/'build/crow-v15-presolve-contact-audit-20261113/libmetalrobo.dylib';res=subprocess.run([str(exe)],input='\n'.join(inputs)+'\n',text=True,capture_output=True,check=True,env={**os.environ,'DYLD_LIBRARY_PATH':str(lib.parent)});com={}
for l in res.stdout.splitlines():
 z=l.split();assert z[0]=='C';v=list(map(float,z[3:]));assert len(v)==7 and all(map(math.isfinite,v));com[ids[int(z[1])]]=v
assert len(com)==len(inputs);rows=[]
for e in range(2,8):
 first,release=info[e];p,h=D['push'],D['hold'];mismatch=[] if first is None else [t for t in range(1,first) if any(p['state'][e,t][k]!=h['state'][e,t][k] for k in ('q','v','policy_action','contacts'))];row={'lane':e,'first_leg_target_difference':first,'release_first_of_three_ticks':release,'prepush_state_mismatches':mismatch}
 if first is not None and release is not None and release>=first:
  keys=[(e,t,k) for t in range(first,release+1) for k in range(16)];complete=all(q in D[arm][x] for q in keys for arm in ('push','hold') for x in ('contact','aero','toe'));assert complete
  ground=sum(sum(p['contact'][q][2:4])-sum(h['contact'][q][2:4]) for q in keys);aero=sum(p['aero'][q]-h['aero'][q] for q in keys)*.00125;toe=sum(p['toe'][q]>=11.99 for q in keys);nonfoot=sum(p['contact'][q][4] for q in keys); cp=com['push',e,release];ch=com['hold',e,release];bp=com['push',e,first-1];bh=com['hold',e,first-1];mass=cp[0];assert all(abs(x[0]-mass)<1e-8 for x in (ch,bp,bh));velocitydelta=cp[6]-ch[6];dp=mass*((cp[6]-bp[6])-(ch[6]-bh[6]));row.update({'matched_ground_normal_impulse_difference_ns':ground,'matched_aero_vertical_impulse_difference_ns':aero,'push_com_vz_at_release_m_s':cp[6],'hold_com_vz_at_push_release_m_s':ch[6],'com_vz_difference_at_release_m_s':velocitydelta,'matched_com_vertical_momentum_change_difference_ns':dp,'momentum_residual_after_ground_normal_and_aero_ns':dp-ground-aero,'toe_threshold_substeps_through_release':toe,'nonfoot_normal_impulse_through_release_ns':nonfoot,'observation_complete':complete});row['incremental_leg_screen_pass']=not mismatch and cp[6]>0 and ground>.1 and velocitydelta>=.05 and toe==0 and nonfoot<=1e-7
 else:row['incremental_leg_screen_pass']=False
 rows.append(row)
wing=json.loads((A/'matched-wing-check.json').read_text())['valid'];drive=json.loads((A/'drive-validation.json').read_text())['valid'];effort=all(x['within_3_bound'] for x in json.loads((A/'wing-effort-check.json').read_text())['rows']);valid=main['valid_batch'] and wing and drive and effort;paired={str(e)+'-'+str(e+1):valid and main['paired_push_screen_pass'][str(e)+'-'+str(e+1)] and all(next(x for x in rows if x['lane']==k)['incremental_leg_screen_pass'] for k in (e,e+1)) for e in (2,4,6)};out={'valid':valid,'rows':rows,'paired_incremental_leg_and_flight_pass':paired,'scope':'Comparative ground-normal and aero impulses are not joint muscle work; report residual rather than assuming unobserved forces vanish.'};(A/'leg-attribution.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
