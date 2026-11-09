from pathlib import Path
import gzip,json,math,statistics
R=Path.cwd();A=R/'artifacts/crow-v15-contact-slip-20261109';O=R/'artifacts/crow-v15-ground-contact-20261109'
# Contact observer logs submittedControlSteps+1. Independent task root-velocity
# telemetry identifies its matching replay state as frame step-1.
def evaluate(replay,stderr):
 p=json.load(gzip.open(replay,'rt'))['payload'];frames={f['step']:f for f in p['frames']};names=('crow_left_foot','crow_right_foot');speeds={n:[] for n in names};speeds_all={n:[] for n in names};loads={};nonfoot=set();qerr=0.;verr=0.;matched=0
 for line in gzip.open(stderr,'rt'):
  if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
  sample=json.loads(line.split(' ',1)[1]);step=sample['step']
  if sample['lane']!=1 or not 101<=step<=250:continue
  f=frames[step-1];matched+=1;qerr=max(qerr,max(abs(x-y) for x,y in zip(sample['q'],f['q'])));verr=max(verr,max(abs(x-y) for x,y in zip(sample['v'],f['v'])));stride=p['body_state_stride'];flags=loads.setdefault(step,set())
  for c in sample['contacts']:
   a,b=c['a_name'],c['b_name'];name=next((v for v in (a,b) if v in names),None)
   if 'locomotion_ground' not in (a,b) or c['impulses'][0]<=0:continue
   if not name:nonfoot.add(step);continue
   normal=c['normal'];up=(-1 if a==name else 1)*normal[2]*c['impulses'][0]
   if up<=0:continue
   flags.add(name);idx=p['body_names'].index(name);st=f['body_states'][idx*stride:(idx+1)*stride];v=st[7:10];w=st[10:13];r=[c['point'][i]-st[i] for i in range(3)];cross=[w[1]*r[2]-w[2]*r[1],w[2]*r[0]-w[0]*r[2],w[0]*r[1]-w[1]*r[0]];vc=[v[i]+cross[i] for i in range(3)];dot=sum(vc[i]*normal[i] for i in range(3))/sum(x*x for x in normal);speed=math.sqrt(sum((vc[i]-dot*normal[i])**2 for i in range(3)));speeds_all[name].append(speed)
   if up>.0001:speeds[name].append(speed)
 assert matched and qerr<1e-6 and verr<1e-6,(replay,matched,qerr,verr)
 def distribution(xs):
  xs=sorted(xs);return {'samples':len(xs),'median_m_s':statistics.median(xs) if xs else None,'p95_m_s':xs[math.ceil(.95*len(xs))-1] if xs else None,'max_m_s':max(xs) if xs else None}
 a,b=frames[100]['q'],frames[250]['q'];d=math.hypot(b[0]-a[0],b[1]-a[1]);return {'window_net_displacement_m':d,'left_only_steps':sum(x=={names[0]} for x in loads.values()),'right_only_steps':sum(x=={names[1]} for x in loads.values()),'nonfoot_steps':len(nonfoot),'loaded_contacts_gt_0_0001_ns':{n:distribution(speeds[n]) for n in names},'all_positive_upward_contacts':{n:distribution(speeds_all[n]) for n in names},'matched_endpoint_samples':matched,'max_q_alignment_error':qerr,'max_v_alignment_error':verr}
rows={};rows['baseline']=evaluate(O/'feedback_live.replay.json.gz',A/'rollout.stderr.gz')
for n in ('low-amplitude','slow-cycle','slow-low','stance-gain-05','stance-gain-15','stance-gain-30','stance-activation','velocity-signal'):
 rows[n]=evaluate(A/(n+'.replay.json.gz'),A/(n+'.stderr.gz'))
for n,row in rows.items():
 summary=json.loads((A/(n+'-summary.json')).read_text()) if n!='baseline' else json.loads((A/'tuning-baseline-summary.json').read_text());row['physical_failure_count']=summary['physical_failure_count'];row['failed_environment_steps']=summary['failed_environment_steps'];row['necessary_screen_pass']=bool(row['window_net_displacement_m']>=.03 and row['left_only_steps'] and row['right_only_steps'] and not row['nonfoot_steps'] and not row['physical_failure_count'] and not row['failed_environment_steps'] and all(x['p95_m_s'] is not None and x['p95_m_s']<=.04 for x in row['loaded_contacts_gt_0_0001_ns'].values()))
report={'alignment':'contact trace step N matches replay state N-1; every used q and v checked within 1e-6','prior_unaligned_metrics_invalid':True,'rows':rows,'scope':'endpoint contact velocity; not continuous no-slip or integrated work'};(A/'aligned-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
for n,row in rows.items():print(n,'distance',round(row['window_net_displacement_m'],4),'p95',*[round(x['p95_m_s'],4) if x['p95_m_s'] is not None else None for x in row['loaded_contacts_gt_0_0001_ns'].values()],'pass',row['necessary_screen_pass'])
