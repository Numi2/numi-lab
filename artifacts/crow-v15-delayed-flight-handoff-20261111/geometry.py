from pathlib import Path
import gzip,hashlib,json,math,subprocess
D=Path('artifacts/crow-v15-delayed-flight-handoff-20261111');S=Path('artifacts/crow-v15-forward-leg-jump-20261111')
def longest(x):
 best=cur=0
 for a in x:cur=cur+1 if a else 0;best=max(best,cur)
 return best
results=[]
for name in ('hn50-ap18','hn50-ap30','hn50-ap45','hn70-ap45','hn85-ap45','hn100-ap45'):
 sup={};sub={}
 with gzip.open(S/(name+'.stderr.gz'),'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    d=json.loads(l.split(' ',1)[1]);
    if 248<=d['step']<=270:sup[d['lane'],d['step']]=d
   elif l.startswith('CROW_WALK_SUBSTEP '):
    v=l.split();lane,t,k=int(v[3]),int(v[1]),int(v[2]);
    if 248<=t<=270:sub[lane,t,k]=list(map(float,v[6:]))
 assert len(sup)==8*23 and len(sub)==8*23*16
 text=''.join(str(lane)+' '+str(t)+' '+' '.join(map(str,d['q']+d['v']))+'\n' for (lane,t),d in sorted(sup.items()));p=subprocess.run(['build/crow-v15-walk-all-lanes-geometry-20261111/kinematics'],input=text,text=True,capture_output=True,check=True);pose={}
 for l in p.stdout.splitlines():
  v=l.split();pose[int(v[0]),int(v[1]),v[2]]=list(map(float,v[3:]))
 assert len(pose)==8*23*10
 with gzip.open(S/(name+'.replay.json.gz'),'rt') as f:replay=json.load(f)['payload']
 frames={f['step']:f for f in replay['frames'] if 247<=f['step']<=269};assert replay['environment']==1 and len(frames)==23
 maxq=maxpos=maxrot=maxcontact=0.
 for (lane,t),d in sup.items():
  norms=[0.,0.]
  for c in d['contacts']:
   a,b=c['a_name'],c['b_name'];body=b if a=='locomotion_ground' else a if b=='locomotion_ground' else ''
   for i,side in enumerate(('left','right')):
    if body.startswith('crow_'+side+'_') and ('foot' in body or '_toe' in body):norms[i]+=max(0.,c['impulses'][0])
  maxcontact=max(maxcontact,max(abs(sub[lane,t,15][2+i]-norms[i]) for i in (0,1)))
 for t,f in frames.items():
  d=sup[1,t+1];maxq=max(maxq,max(abs(a-b) for a,b in zip(f['q'],d['q'])))
  for n in replay['body_names']:
   if (1,t+1,n) not in pose:continue
   z=pose[1,t+1,n];i=replay['body_names'].index(n)*13;v=f['body_states'][i:i+13];maxpos=max(maxpos,math.sqrt(sum((a-b)**2 for a,b in zip(z[:3],v[:3]))));maxrot=max(maxrot,min(math.sqrt(sum((a-b)**2 for a,b in zip(z[3:7],v[3:7]))),math.sqrt(sum((a+b)**2 for a,b in zip(z[3:7],v[3:7])))))
 assert maxq<1e-6 and maxpos<1e-5 and maxrot<1e-5 and maxcontact<1e-6,(maxq,maxpos,maxrot,maxcontact)
 rows=[]
 for lane in range(8):
  flags=[];valid=[]
  for t in range(248,271):
   ss=[sub[lane,t,k] for k in range(16)];imp=[sum(x[i] for x in ss) for i in range(7)];gaps=[]
   for side in ('left','right'):
    for n,h in [('crow_'+side+'_foot',(.010,.01725,.0092))]+[('crow_'+side+'_'+x,(.018,.0025,.0092)) for x in ('front_toe_1','front_toe_2','front_toe_3','rear_toe')]:
     z=pose[lane,t,n];x,y,q,w=z[3:7];row=(2*(x*q-y*w),2*(y*q+x*w),1-2*(x*x+y*y));gaps.append(z[2]-sum(abs(a)*b for a,b in zip(row,h)))
   yes=min(gaps)>.001 and imp[2]+imp[3]<=1e-7 and imp[6]<=1e-7 and sup[lane,t]['v'][2]>.05;flags.append(yes)
   if yes:valid.append({'observer_step':t,'min_foot_gap_m':min(gaps),'root_vx_m_s':sup[lane,t]['v'][0],'root_vz_m_s':sup[lane,t]['v'][2]})
  rows.append({'lane':lane,'longest_clear_unloaded_rising_run':longest(flags),'valid_samples':valid})
 results.append({'variant':name,'source_sha256':hashlib.sha256((S/(name+'.stderr.gz')).read_bytes()).hexdigest(),'max_q_error':maxq,'max_position_error_m':maxpos,'max_quaternion_error':maxrot,'max_last_contact_error_Ns':maxcontact,'all_lanes_two_consecutive':all(r['longest_clear_unloaded_rising_run']>=2 for r in rows),'lanes':rows})
out={'results':results,'scope':'development starts, endpoint geometry and full-interval forces; not continuous swept clearance or launch-to-flight qualification'};(D/'geometry-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
