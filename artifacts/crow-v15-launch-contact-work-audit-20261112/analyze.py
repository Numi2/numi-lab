from pathlib import Path
import json,gzip,collections
R=Path('artifacts');A=R/'crow-v15-launch-contact-work-audit-20261112';names=['crow-v15-slow-push-uniform-dev-20261112','crow-v15-slow-push-crouch-control-20261112']
def read(p):
 state={};contact={}
 with gzip.open(p/'candidate-20261127-rebound-flexed/run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);state[x['lane'],x['step']]=x
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);contact[e,t,k]=[float(z[6]),float(z[7]),float(z[8]),float(z[9])]
 return state,contact
(a,ac),(b,bc)=map(lambda n:read(R/n),names)
rows=[]
for e in range(8):
 out={'lane':e}
 for label,s,c in [('extension',a,ac),('crouch_hold',b,bc)]:
  windows={}
  for start,end in ((348,355),(348,360),(348,375),(348,400)):
   z=[c[e,t,k] for t in range(start,end+1) for k in range(16)]
   windows[f'{start}_{end}']={'vertical_contact_impulse_ns':sum(v[0]+v[1] for v in z),'unloaded_substeps':sum(v[2]+v[3]<=1e-7 for v in z)}
  out[label]={'windows':windows,'samples':{str(t):{'root_z_m':s[e,t]['q'][2],'root_vz_m_s':s[e,t]['v'][2]} for t in (347,355,360,375,400)}}
 rows.append(out)
out={'rows':rows,'limits':'Retained exact-prefix matched development pair. Foot vertical total contact impulse includes normal and tangential components; excludes moments and all aerodynamic impulse; windows include gravity, recontacts and feedback interactions. Crouch hold is powered and wings active. Differences do not identify leg muscle work or leg-only takeoff.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:
 e=r['extension'];c=r['crouch_hold'];print(r['lane'],'impulse 348-355 ext/hold',round(e['windows']['348_355']['vertical_contact_impulse_ns'],4),round(c['windows']['348_355']['vertical_contact_impulse_ns'],4),'vz at 355',round(e['samples']['355']['root_vz_m_s'],4),round(c['samples']['355']['root_vz_m_s'],4))
