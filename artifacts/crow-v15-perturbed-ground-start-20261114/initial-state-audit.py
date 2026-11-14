from pathlib import Path
import gzip,json,math,sys
p=Path('artifacts/crow-v15-perturbed-ground-start-20261114');new=p/sys.argv[1];ref=Path('artifacts/crow-v15-early-height-turns-20261114/reference-states.json.gz');baseline={(x['lane'],x['step']):x for x in json.load(gzip.open(ref,'rt'))};states={}
with gzip.open(new/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);
   if x['step']==1:states[x['lane']]=x
def angles(q):
 x,y,z,w=q[3:7];return (math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),math.asin(max(-1,min(1,2*(w*y-z*x)))))
rows=[]
for e in range(8):
 a=states[e];b=baseline[e,1];r,pitch=angles(a['q']);r0,p0=angles(b['q']);rows.append({'lane':e,'roll_difference_rad':r-r0,'pitch_difference_rad':pitch-p0,'forward_velocity_difference_m_s':a['v'][0]-b['v'][0],'roll_pitch_changed':abs(r-r0)>.02 and abs(pitch-p0)>.02})
out={'rows':rows,'all_changed':len(rows)==8 and all(r['roll_pitch_changed'] for r in rows),'scope':'first accepted state after initial contact solve; not exact pre-solve reset state'};(p/'initial-state-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
