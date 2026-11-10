from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-supported-balance-identification-20261110');O=Path('artifacts/crow-v15-toe-soft-drive-probe-20261110');old={};trace={};contacts={};outcomes={}
with gzip.open(O/'results.txt.gz','rt') as f:
 for l in f:
  v=l.split()
  if v and v[0]=='trace' and int(v[2])==3:old[(int(v[1]),int(v[3]))]=v[4:]
with gzip.open(A/'results.txt.gz','rt') as f:
 for l in f:
  v=l.split()
  if not v:continue
  if v[0]=='trace':trace[(int(v[1]),int(v[2]),int(v[3]))]=v[4:]
  elif v[0]=='case_result':outcomes[(int(v[1]),int(v[2]))]=' '.join(v[3:])
  elif v[0]=='allcontact':
   i,a,t=int(v[1]),int(v[2]),int(v[3]);an,bn=v[4],v[5];nz=float(v[7]);imp=float(v[8]);body=bn if an=='locomotion_ground' else an;sgn=1. if body==bn else -1.;up=max(sgn*nz*imp,0.);d=contacts.setdefault((i,a,t),{'left':0.,'right':0.,'other':0.});side=next((s for s in ('left','right') if '_'+s+'_' in body),None)
   if side and ('_foot' in body or '_toe' in body):d[side]+=up
   else:d['other']+=imp
keys=set(old)|{(i,t) for i,a,t in trace if a==0};diff=[(i,t) for i,t in sorted(keys) if old.get((i,t))!=trace.get((i,0,t))];rows=[]
for i in range(8):
 for a in range(4):
  rec=[]
  for t in range(200):
   v=trace.get((i,a,t))
   if v is None:continue
   qx,qy=float(v[9]),float(v[10]);vel=[float(v[8]),float(v[11]),float(v[12])];tilt=math.acos(max(-1.,min(1.,1.-2.*(qx*qx+qy*qy))));speed=math.sqrt(sum(x*x for x in vel));c=contacts.get((i,a,t),{'left':0.,'right':0.,'other':0.});support=min(c['left'],c['right'])>.00005 and c['left']+c['right']>=.5*.45004*9.81*.005 and c['other']<=1e-7 and tilt<=.35 and speed<=.12
   rec.append({'step':t,'root_x':float(v[6]),'root_z':float(v[7]),'tilt':tilt,'speed':speed,'other':c['other'],'support':support})
  tail=rec[-25:];completed=len(rec)==200 and outcomes.get((i,a),'').startswith('completed');passes=completed and len(tail)==25 and all(r['support'] for r in tail)
  rows.append({'case':i,'arm':a,'samples':len(rec),'outcome':outcomes.get((i,a),'missing'),'terminal_25_supported':passes,'first_nonfoot_impulse_step':next((r['step'] for r in rec if r['other']>1e-7),None),'first_root_below_perch_top':next((r['step'] for r in rec if r['root_z']<0),None),'max_tilt_rad':max((r['tilt'] for r in rec),default=None),'max_speed_m_s':max((r['speed'] for r in rec),default=None),'root_x_displacement_m':(rec[-1]['root_x']-rec[0]['root_x']) if rec else None})
out={'original_open_baseline_parity_exact':not diff,'first_differences':diff[:20],'rows':rows,'limitations':'Already-supported isolated starts, estimated mechanics, endpoint contacts; no grip, takeoff, flight, landing, substep or biological qualification.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print('Open baseline exact parity',not diff,'differences',len(diff))
for a in range(4):
 z=[r for r in rows if r['arm']==a];print('arm',a,'terminal supported',sum(r['terminal_25_supported'] for r in z),'rollbacks',sum(not r['outcome'].startswith('completed') for r in z),'nonfoot contact',sum(r['first_nonfoot_impulse_step'] is not None for r in z))
