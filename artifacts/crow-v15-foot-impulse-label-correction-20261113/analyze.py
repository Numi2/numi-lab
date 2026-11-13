from pathlib import Path
import collections,gzip,hashlib,json
R=Path.cwd();A=R/'artifacts/crow-v15-foot-impulse-label-correction-20261113';spec=[('crow-v15-consistent-wingheld-leg-control-20261113','rebound-wingheld',348,357),('crow-v15-wingheld-retraction-screen-20261113','rebound-retraction',348,357),('crow-v15-ramped-leg-launch-20261113','rebound-ramped',371,400)];out=[]
for name,tag,lo,hi in spec:
 d=collections.defaultdict(lambda:[0.,0.,0.,0]);hashes={}
 for arm in ('push','hold'):
  p=R/'artifacts'/name/(arm+'-wing-20261126-'+tag)/'run.stderr.gz';h=hashlib.sha256()
  with p.open('rb') as raw:
   for b in iter(lambda:raw.read(1048576),b''):h.update(b)
  hashes[arm]=h.hexdigest()
  with gzip.open(p,'rt') as f:
   for l in f:
    z=l.split()
    if not z or z[0]!='CROW_TWO_TOE_SUBSTEP':continue
    t,k,e=map(int,z[1:4]);
    if e<2 or not lo<=t<=hi:continue
    v=list(map(float,z[6:11]));q=d[arm,e];vert=sum(v[:2]);normal=sum(v[2:4]);q[0]+=vert;q[1]+=normal;q[2]=max(q[2],abs(vert-normal));q[3]+=1
 for e in range(2,8):
  p,h=d['push',e],d['hold',e];assert p[3]==h[3]==(hi-lo+1)*16
  out.append({'experiment':name,'ticks':[lo,hi],'lane':e,'push_minus_hold_vertical_ns':p[0]-h[0],'push_minus_hold_normal_ns':p[1]-h[1],'max_per_substep_vertical_minus_normal_abs_ns':max(p[2],h[2]),'source_logs_sha256':hashes})
(A/'analysis.json').write_text(json.dumps({'rows':out,'scope':'Recorded component meanings are vertical foot impulse, normal foot impulse, and other contact normal impulse. This verifies the specified flat-ground windows only; the quantities are not interchangeable on arbitrary terrain.'},indent=2)+'\n')
print('rows',len(out),'max absolute paired-delta difference',max(abs(x['push_minus_hold_vertical_ns']-x['push_minus_hold_normal_ns']) for x in out),'max substep difference',max(x['max_per_substep_vertical_minus_normal_abs_ns'] for x in out))
