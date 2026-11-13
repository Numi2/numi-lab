from pathlib import Path
import gzip,json,math,subprocess,os,hashlib
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-aero-power-audit-20261113';base=R/'artifacts/crow-v15-late-flight-aero-observer-20261113/push-wing-20261126-rebound-lateaero';new=A/'push-wing-20261126-rebound-aeropower';D={}
for name,p in [('base',base),('new',new)]:
 d={'state':{},'contact':{},'aero':{},'wing':{},'errors':[],'meta':json.loads((p/'run.stdout').read_text())['first_episode_navigation'],'exit':json.loads((p/'run-result.json').read_text())['exit_code']};D[name]=d
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);key=x['lane'],x['step'];assert key not in d['state'];d['state'][key]=x;continue
   z=l.split()
   if not z:continue
   typ=z[0]
   if typ in ('CROW_TWO_TOE_SUBSTEP','CROW_AERO_FORCE','CROW_WING_EFFORT'):
    t,k,e=map(int,z[1:4]);key=e,t,k
    if typ=='CROW_TWO_TOE_SUBSTEP':assert key not in d['contact'];d['contact'][key]=list(map(float,z[6:11]))
    elif typ=='CROW_AERO_FORCE':assert key not in d['aero'];d['aero'][key]=list(map(float,z[5:]));assert len(d['aero'][key])==12 and all(map(math.isfinite,d['aero'][key]))
    else:assert key not in d['wing'];d['wing'][key]=list(map(float,z[4:]));assert len(d['wing'][key])==14
   elif typ in ('CROW_SUBSTEP_ERROR','CROW_AERO_OBSERVER_ERROR'):d['errors'].append(l.strip())
 for e in range(8):
  if d['meta']['steps_by_environment'][e]<400:d['errors'].append(['early end',e])
  if sum((e,t) in d['state'] for t in range(1,401))!=400:d['errors'].append(['state coverage',e])
  if sum((e,t,k) in d['contact'] for t in range(335,401) for k in range(16))!=66*16:d['errors'].append(['contact coverage',e])
  if name=='new' and sum((e,t,k) in d['aero'] and (e,t,k) in d['wing'] for t in range(348,401) for k in range(16))!=53*16:d['errors'].append(['aero or effort coverage',e])
 if d['exit']!=0:d['errors'].append(['exit',d['exit']])
parity=[]
for e in range(8):
 sb=[t for t in range(1,401) if any(D['base']['state'][e,t][k]!=D['new']['state'][e,t][k] for k in ('q','v','policy_action','contacts'))];cb=[(t,k) for t in range(335,401) for k in range(16) if D['base']['contact'][e,t,k]!=D['new']['contact'][e,t,k]];parity.append({'lane':e,'state_mismatches':len(sb),'contact_substep_mismatches':len(cb)})
valid=not any(d['errors'] for d in D.values()) and not any(x['state_mismatches'] or x['contact_substep_mismatches'] for x in parity)
if not valid:
 (A/'integrity-failed.json').write_text(json.dumps({'parity':parity,'errors':{k:v['errors'] for k,v in D.items()}},indent=2)+'\n');raise SystemExit('observer parity or coverage failed')
exe=R/'build/crow-v15-com-momentum-audit-20261113/kinematics';lib=R/'build/crow-v15-presolve-contact-audit-20261113/libmetalrobo.dylib';inputs=[];ids={}
for e in range(8):
 for t in range(375,401):
  x=D['new']['state'][e,t];i=len(inputs);ids[i]=(e,t);inputs.append(str(i)+' 0 '+' '.join(map(str,x['q']+x['v'])))
q=D['new']['state'][5,375]['q'];z=[0.]*40;cal=z.copy();cal[:3]=[.3,-.2,.1];i0=len(inputs)
for i,v in ((i0,z),(i0+1,cal)):inputs.append(f'{i} 0 '+' '.join(map(str,q+v)))
r=subprocess.run([str(exe)],input='\n'.join(inputs)+'\n',text=True,capture_output=True,check=True,env={**os.environ,'DYLD_LIBRARY_PATH':str(lib.parent)});kout={}
for line in r.stdout.splitlines():
 z=line.split();assert z[0]=='C';i=int(z[1]);v=list(map(float,z[3:]));assert len(v)==7 and all(map(math.isfinite,v));kout[i]=v
assert len(kout)==len(inputs);err=max(abs(kout[i0+1][4+j]-kout[i0][4+j]-cal[j]) for j in range(3));assert err<1e-8;com={ids[i]:v for i,v in kout.items() if i in ids};rows=[];d=D['new']
for e in range(8):
 a,b=com[e,375],com[e,400];mass=a[0];assert abs(mass-.45006)<1e-5 and abs(b[0]-mass)<1e-10
 foot=sum(sum(d['contact'][e,t,k][:2]) for t in range(376,401) for k in range(16));other=sum(d['contact'][e,t,k][4] for t in range(376,401) for k in range(16));aero=sum(d['aero'][e,t,k][2]*.00125 for t in range(376,401) for k in range(16));g=-mass*9.81*.02*25;dp=mass*(b[6]-a[6]);res=dp-foot-g-aero
 windows=[]
 for lo,hi in [(376,380),(381,385),(386,390),(391,395),(396,400)]:
  f=sum(d['aero'][e,t,k][2] for t in range(lo,hi+1) for k in range(16))/((hi-lo+1)*16);v=sum(sum(d['contact'][e,t,k][:2]) for t in range(lo,hi+1) for k in range(16));windows.append({'steps':[lo,hi],'mean_logged_vertical_aero_N':f,'foot_normal_impulse_Ns':v,'start_com_z_m':com[e,lo-1][3],'end_com_z_m':com[e,hi][3],'end_com_vz_m_s':com[e,hi][6]})
 rows.append({'lane':e,'mass_kg':mass,'com_z_375_m':a[3],'com_z_400_m':b[3],'com_vz_375_m_s':a[6],'com_vz_400_m_s':b[6],'delta_vertical_momentum_ns':dp,'foot_normal_impulse_ns':foot,'nonfoot_normal_impulse_ns':other,'modeled_aero_vertical_impulse_ns':aero,'gravity_assuming_9_81_ns':g,'remaining_residual_ns':res,'max_abs_modeled_wing_effort':max(abs(d['wing'][e,t,k][j]) for t in range(376,401) for k in range(16) for j in (0,7)),'windows':windows})
out={'valid':valid,'parity':parity,'source_sha256':{name:hashlib.sha256((p/'run.stderr.gz').read_bytes()).hexdigest() for name,p in [('base',base),('new',new)]},'fk_uniform_translation_error_m_s':err,'dt_substep_s':.00125,'rows':rows,'scope':'Modeled forces in ticks 376–400 with endpoint COM velocity at 375 and 400, assumed gravity; does not by itself prove biological forces or complete energy accounting. Observer parity applies to logged states and contacts, not unobserved internal state.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print('valid',valid,'max FK translation error',err)
for x in rows[2:]:print(x['lane'],'dp',round(x['delta_vertical_momentum_ns'],3),'aero',round(x['modeled_aero_vertical_impulse_ns'],3),'feet',round(x['foot_normal_impulse_ns'],3),'gravity',round(x['gravity_assuming_9_81_ns'],3),'residual',round(x['remaining_residual_ns'],4),'Vz',round(x['com_vz_375_m_s'],3),'->',round(x['com_vz_400_m_s'],3))
