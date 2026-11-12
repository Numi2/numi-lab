from pathlib import Path
import collections,gzip,hashlib,json,math,os,subprocess
R=Path('.');A=R/'artifacts/crow-v15-foot-settling-refined-sensitivity-20261112';G=R/'artifacts/crow-v15-foot-settling-geometry-audit-20261112/analysis.json';prior=R/'artifacts/crow-v15-negative-ankle-matched-control-20261112';kin=R/'build/crow-v15-hip-roll-walk-launch-matched-20261111/kinematics';g=json.loads(G.read_text());out={}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def gap(v,half,ground):
 x,y,z,w=v[3:7];normal=(2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y));return v[2]-ground-sum(abs(a)*b for a,b in zip(normal,half))
for seed in (20261126,20261127):
 data=g[f'{seed}-previous'];events=data['events'];ground=data['ground_z_m'];source=prior/f'candidate-{seed}-rebound-phase/run.stderr.gz';assert sha(source)==data['source_sha256'];keys={(e['lane'],e['first_sample']) for e in events};q={};v={}
 with gzip.open(source,'rt') as f:
  for line in f:
   if not line.startswith(('CROW_Q_SUBSTEP ','CROW_V_SUBSTEP ')):continue
   z=line.split();t,k,e=map(int,z[1:4]);i=(t-201)*16+k
   if (e,i) in keys:(q if z[0]=='CROW_Q_SUBSTEP' else v)[e,i]=list(map(float,z[4:]))
 assert q.keys()==v.keys()==keys
 requests=[];m=[]
 for j,ev in enumerate(events):
  e,i=ev['lane'],ev['first_sample'];ki,ai=(15,16) if ev['side']=='left' else (18,19)
  for dk,da in [(0.,0.)]+[(dk,da) for dk in (-.04,-.03,-.02) for da in (.04,.05,.06)]:
   x=q[e,i].copy();x[ki]+=dk;x[ai]+=da;n=len(m);m.append((j,dk,da));requests.append(str(n)+' 0 '+' '.join(map(str,x+v[e,i])))
 env={**os.environ,'DYLD_LIBRARY_PATH':str((R/'build/crow-v15-hip-roll-journey-integration-20261111').resolve())};p=subprocess.run([str(kin)],input='\n'.join(requests)+'\n',capture_output=True,text=True,check=True,env=env);b=collections.defaultdict(dict)
 for line in p.stdout.splitlines():
  z=line.split()
  if not z or z[0]!='body':continue
  n=int(z[1]);j,_,_=m[n];side=events[j]['side'];names=[f'crow_{side}_foot']+[f'crow_{side}_{toe}_{part}' for toe in ('front_toe_1','front_toe_2','front_toe_3','rear_toe') for part in ('proximal','distal')]
  if z[3] in names:b[n][z[3]]=list(map(float,z[4:]))
 assert len(b)==len(m) and all(len(x)==9 for x in b.values())
 rows=[];zero_error=0.
 for n,(j,dk,da) in enumerate(m):
  ev=events[j];side=ev['side'];vals={name:gap(v,(.010,.01725,.0092) if name.endswith('_foot') else (.009,.0025,.0092),ground) for name,v in b[n].items()};front=min(v for name,v in vals.items() if 'front_toe_2_' in name);rear=min(v for name,v in vals.items() if 'rear_toe_' in name);sole=vals[f'crow_{side}_foot'];reference=ev['samples']['first'];front0=reference['front_group_min_clearance_m'];rear0=reference['rear_group_min_clearance_m'];min_all=min(vals.values());reduction=front0-front
  if dk==da==0.:zero_error=max(zero_error,abs(front-front0),abs(rear-rear0),abs(sole-reference['foot_sole_clearance_m']))
  ok=reduction>=.001 and abs(rear-rear0)<=.0005 and min_all>=-.0005
  rows.append({'event':j,'lane':ev['lane'],'side':side,'sample':ev['first_sample'],'knee_offset_rad':dk,'ankle_offset_rad':da,'front_clearance_m':front,'rear_clearance_m':rear,'sole_clearance_m':sole,'min_all_m':min_all,'front_reduction_m':reduction,'rear_change_m':rear-rear0,'geometry_pass':ok})
 assert zero_error<1e-8,(seed,zero_error)
 counts=collections.Counter((r['knee_offset_rad'],r['ankle_offset_rad']) for r in rows if r['geometry_pass']);passed=len({r['event'] for r in rows if r['geometry_pass']})
 out[str(seed)]={'source_sha256':sha(source),'kinematics_sha256':sha(kin),'zero_offset_max_clearance_error_m':zero_error,'events':len(events),'events_with_candidate':passed,'pass_count_by_offsets':{f'{k[0]:+.2f},{k[1]:+.2f}':v for k,v in counts.items()},'rows':rows}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for seed,x in out.items():print(seed,'events with candidate',x['events_with_candidate'],'/',x['events'],'offset counts',x['pass_count_by_offsets'],'zero offset error',x['zero_offset_max_clearance_error_m'])
