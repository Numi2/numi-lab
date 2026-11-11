from pathlib import Path
import gzip,json,subprocess
A=Path('artifacts/crow-v15-toe-release-placement-20261111');P=Path('artifacts/crow-v15-toe-release-attitude-narrow-20261111/trace.log.gz');q=v=reference=None
with gzip.open(P,'rt') as f:
 for l in f:
  a=l.split()
  if a[:3]==['step','0','150']:
   qi=a.index('q');vi=a.index('v');q=list(map(float,a[qi+1:vi]));v=list(map(float,a[vi+1:]))
  if a[:4]==['pose','0','150','crow_left_foot']:reference=list(map(float,a[4:7]))
assert q and v and reference
path=Path('/tmp/crow-foot-jacobian.txt')
with path.open('w') as f:
 for e in range(4):
  x=q.copy()
  if e:x[13+e]+=.01
  f.write(f'{e} 1 '+' '.join(map(str,x+v))+'\n')
p=subprocess.run(['build/crow-v15-two-segment-gait-geometry-20261111/probe',str(path)],capture_output=True,text=True,check=True);positions={}
for l in p.stdout.splitlines():
 a=l.split()
 if a[2]=='crow_left_foot':positions[int(a[0])]=list(map(float,a[3:6]))
err=max(abs(a-b) for a,b in zip(positions[0],reference));r={'scope':'local fixed-root kinematic sensitivity at one constructed regrip state, not a dynamic stability or biological claim','reference_foot_position_error_m':err,'hip_foot_dx_m_per_rad':(positions[1][0]-positions[0][0])/.01,'knee_foot_dx_m_per_rad':(positions[2][0]-positions[0][0])/.01,'ankle_foot_dx_m_per_rad':(positions[3][0]-positions[0][0])/.01};(A/'jacobian.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
