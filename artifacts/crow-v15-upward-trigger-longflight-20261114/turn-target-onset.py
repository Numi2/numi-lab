from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-upward-trigger-longflight-20261114'); P=A/'turn-20261126-rebound-longflight'
s={(x['lane'],x['step']):x for x in json.load(gzip.open(A/'turn-reference.json.gz','rt')) if x['lane']==7}; out=[]
def smooth(a,b,x):
 u=max(0.,min(1.,(x-a)/(b-a)));return u*u*(3.-2.*u)
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  if not line.startswith('CROW_DRIVE_TARGET '):continue
  z=line.split(); t,e=map(int,z[1:3]);
  if e!=7 or not 840<=t<=920:continue
  v=list(map(float,z[3:]));sec=(t-2)*.02;phase=2*math.pi*6.4*(sec-7.42);goal=.80-.25*smooth(24.,24.5,sec);height=max(-.020,min(.012,.08*(goal-s[7,t-1]['q'][2])-.003))*smooth(9.,9.5,sec);boost=.08*smooth(7.90,8.10,sec)*(1.-smooth(10.,10.5,sec));amp=.55+.15*smooth(7.50,7.58,sec)+height+boost;expected=.35+amp*math.sin(phase);actual=(v[8]+v[9])/2
  out.append({'step':t,'actual_flap_mean':actual,'expected_full_release_flap':expected,'difference':actual-expected,'pronation_targets':v[10:12]})
o={'scope':'turn lane 7 full-release target reconstruction, not independent contact epoch proof','rows':out,'first_error_gt_0_005':next((x for x in out if abs(x['difference'])>.005),None)}
(A/'turn-target-onset.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o['first_error_gt_0_005'],indent=2))
