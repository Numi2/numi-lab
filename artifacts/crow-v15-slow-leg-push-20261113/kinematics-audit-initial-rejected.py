from pathlib import Path
import gzip,hashlib,json,math,os,subprocess
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-slow-leg-push-20261113';O=R/'artifacts/crow-v15-leg-work-observer-20261113';arms={'slow':A/'push-nowings-20261127-rebound-slow160','hold':O/'nopush-nowings-20261127-rebound-work','fast':O/'push-nowings-20261127-rebound-work'};exe=R/'build/crow-v15-hip-roll-walk-launch-matched-20261111/kinematics';dylib=R/'build/crow-v15-hip-roll-journey-integration-20261111/libmetalrobo.dylib';env={**os.environ,'DYLD_LIBRARY_PATH':str(dylib.parent)};rows=[];valid=[]
for label,p in arms.items():
 st={}
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step']
    if 350<=t<=380:st[e,t]=x
 assert len(st)==8*31
 inp=''.join(f'{e} {t} '+' '.join(map(str,x['q']+x['v']))+'\n' for (e,t),x in sorted(st.items()));res=subprocess.run([str(exe)],input=inp,text=True,capture_output=True,check=True,env=env);body={};com={}
 for line in res.stdout.splitlines():
  z=line.split();e,t=map(int,z[1:3]);v=list(map(float,z[4:]));assert all(math.isfinite(n) for n in v)
  if z[0]=='com':com[(e,t)]=v
  elif z[0]=='body' and z[3] in ('crow_body','crow_left_foot','crow_right_foot'):body[(e,t,z[3])]=v
 assert len(com)==8*31 and len(body)==8*31*3
 with gzip.open(p/'replay.json.gz','rt') as f:replay=json.load(f)['payload']
 assert replay['environment']==5;frames={f['step']:f for f in replay['frames']};names=replay['body_names'];maxq=0.;maxpos=0.
 for t in (350,355,365,375):
  frame=frames[t-1];maxq=max(maxq,max(abs(a-b) for a,b in zip(frame['q'],st[5,t]['q'])))
  for name in ('crow_body','crow_left_foot','crow_right_foot'):
   i=names.index(name)*13;maxpos=max(maxpos,math.dist(frame['body_states'][i:i+3],body[(5,t,name)][:3]))
 assert maxq<1e-6 and maxpos<1e-5,(label,maxq,maxpos);valid.append({'arm':label,'q_error_max':maxq,'body_position_error_m_max':maxpos})
 for e in range(8):
  c0=com[(e,350)];assert abs(c0[0]-.45006)<1e-5
  row={'arm':label,'lane':e,'modeled_mass_kg':c0[0],'com_max_rise_m':max(com[(e,t)][3]-c0[3] for t in range(351,381)),'com_z350_m':c0[3],'com_z375_m':com[(e,375)][3]}
  for side in ('left','right'):
   name='crow_'+side+'_foot';v0=body[(e,350,name)];row[side+'_foot_max_horizontal_origin_displacement_m']=max(math.dist(v0[:2],body[(e,t,name)][:2]) for t in range(351,381))
  rows.append(row)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();out={'verification':valid,'kinematics_executable_sha256':h(exe),'dylib_sha256':h(dylib),'rows':rows,'scope':'Post-hoc whole-body FK analysis of existing recorded development endpoints. COM uses simulated model mass; selected lane-5 FK body poses and q verified against replay. Foot origin displacement does not measure contact-point slip or ground work, and maxima are not necessarily during loaded intervals. No biological or takeoff qualification.'};(A/'kinematics-audit.json').write_text(json.dumps(out,indent=2)+'\n')
for label in arms:
 vals=[r for r in rows if r['arm']==label and r['lane']>=2];print(label,'COM max-rise range',round(min(r['com_max_rise_m'] for r in vals),5),round(max(r['com_max_rise_m'] for r in vals),5),'foot horizontal origin max range',round(min(min(r['left_foot_max_horizontal_origin_displacement_m'],r['right_foot_max_horizontal_origin_displacement_m']) for r in vals),5),round(max(max(r['left_foot_max_horizontal_origin_displacement_m'],r['right_foot_max_horizontal_origin_displacement_m']) for r in vals),5))
print('Replay FK verification',valid)
