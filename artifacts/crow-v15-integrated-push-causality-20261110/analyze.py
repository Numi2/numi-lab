from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-integrated-push-causality-20261110')
def load(arm):
 d={}
 with gzip.open(A/('diagnostic-reserved-final-j-'+arm+'.stderr.gz'),'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);d[x['step']]=x
 return d
p,n=load('push'),load('nopush');common=sorted(set(p)&set(n));first=next((t for t in common if p[t]['q']!=n[t]['q'] or p[t]['v']!=n[t]['v']),None)
print('sample counts',len(p),len(n),'first q/v divergence',first)
rows=[]
for t in (640,645,648,650,652,655,660,665,670,675,680,700,730,750):
 for label,d in [('push',p),('nopush',n)]:
  if t not in d:continue
  x=d[t];imp=[0.,0.,0.];contacts=[]
  for c in x['contacts']:
   if 'locomotion_ground' not in (c['a_name'],c['b_name']):continue
   bird=c['a_name'] if c['b_name']=='locomotion_ground' else c['b_name'];sgn=-1 if bird==c['a_name'] else 1
   for j in range(3):imp[j]+=sgn*sum(c[k][j]*c['impulses'][i] for i,k in enumerate(('normal','tangent_u','tangent_v')))
   contacts.append(bird)
  row={'step':t,'arm':label,'x':x['q'][0],'z':x['q'][2],'vx':x['v'][0],'vz':x['v'][2],'leg_q':x['q'][14:20],'sampled_ground_impulse':imp,'contacts':contacts};rows.append(row)
  print(t,label,'x,z',*[round(a,3) for a in (row['x'],row['z'])],'vx,vz',*[round(a,3) for a in (row['vx'],row['vz'])],'sampled impulse',*[round(a,5) for a in imp],'contacts',','.join(contacts))
(A/'causality.json').write_text(json.dumps({'first_q_v_divergence':first,'samples':rows,'scope':'Impulse is from the final physics substep only, not the total interval impulse. Later divergence may include the contact-gated wing response.'},indent=2)+'\n')
