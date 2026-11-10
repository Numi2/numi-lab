from pathlib import Path
import gzip,json,subprocess,math,statistics
R=Path.cwd();A=R/'artifacts/crow-v15-landing-slip-observer-20261110';name='diagnostic-reserved-final-j-slip-observer';prior=R/'artifacts/crow-v15-landing-support-alignment-20261110/diagnostic-reserved-final-j-ahead-original.stderr.gz'
def samples(p):
 d={}
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);d[x['step']]=x
 return d
x=samples(prior);y=samples(A/(name+'.stderr.gz'));end=json.loads((A/(name+'.stdout')).read_text())['first_episode_navigation']['steps_by_environment'][0];common=sorted(t for t in set(x)&set(y) if t<=end);bad=[t for t in common if x[t]['q']!=y[t]['q'] or x[t]['v']!=y[t]['v']];assert len(common)>=end and not bad,(len(common),end,bad[:10]);rows={t:z for t,z in y.items() if 2500<=t<=end};inp=''.join(str(t)+' '+' '.join(map(str,z['q']))+' '+' '.join(map(str,z['v']))+'\n' for t,z in sorted(rows.items()));p=subprocess.run([str(R/'build/crow-v15-landing-slip-observer-20261110/velocity_probe')],input=inp,text=True,capture_output=True,check=True);v={}
for line in p.stdout.splitlines():
 a=line.split();v[(int(a[0]),a[1])]=list(map(float,a[2:]))
measure=[]
for t,z in rows.items():
 for c in z['contacts']:
  if 'crow_course_perch' not in (c['a_name'],c['b_name']):continue
  bird=c['a_name'] if c['b_name']=='crow_course_perch' else c['b_name'];n,tu,tv=c['impulses']
  if (t,bird) not in v or n<=1e-5:continue
  a=v[(t,bird)];r=[c['point'][j]-a[j] for j in range(3)];w=a[6:9];vel=[a[3]+w[1]*r[2]-w[2]*r[1],a[4]+w[2]*r[0]-w[0]*r[2],a[5]+w[0]*r[1]-w[1]*r[0]];normal=c['normal'];dn=sum(vel[j]*normal[j] for j in range(3));tan=math.sqrt(max(0,sum(k*k for k in vel)-dn*dn));ratio=math.hypot(tu,tv)/n
  measure.append({'step':t,'body':bird,'normal_impulse_Ns':n,'tangent_to_normal':ratio,'endpoint_tangent_speed_m_s':tan})
(A/'slip-analysis.json').write_text(json.dumps({'matching_q_v_samples':len(common),'end':end,'mismatches':bad,'samples':measure,'scope':'Point velocities reconstructed at accepted endpoints and compared to final-substep contact impulses; point position may be from the preceding detection boundary, so this is an endpoint estimate rather than continuous slip or dissipated work.'},indent=2)+'\n')
for lo,hi in ((2550,2600),(2600,2650),(2650,2700),(2700,2750)):
 a=[r for r in measure if lo<=r['step']<hi and '_toe' in r['body']];cap=[r for r in a if r['tangent_to_normal']>.59];print(lo,hi,'toe contacts',len(a),'near .6',len(cap),'median tangential speed',statistics.median(r['endpoint_tangent_speed_m_s'] for r in a) if a else None,'near .6 median speed',statistics.median(r['endpoint_tangent_speed_m_s'] for r in cap) if cap else None)
print('q/v parity',len(common),'end',end)
