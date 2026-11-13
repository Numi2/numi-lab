from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-ankle-clearance-recovery-20261113');p={0:[],1:[]};targets={0:{15:[],16:[]},1:{15:[],16:[]}}
with gzip.open(A/'trace.log.gz','rt') as f:
 for line in f:
  x=line.split()
  if x and x[0]=='reposition':p[int(x[1])].append({'t':int(x[2]),'started':bool(int(x[3])),'fraction':float(x[4]),'support':bool(int(x[5])),'posture':bool(int(x[6])),'open':bool(int(x[7])),'clear':float(x[8])})
  elif x and x[0]=='jointtarget':targets[int(x[1])][int(x[3])].append(float(x[4]))
r={}
for e in (0,1):
 z=p[e];inc=[(a,b) for a,b in zip(z,z[1:]) if b['fraction']>a['fraction']+1e-8];tar=targets[e];r[str(e)]={'samples':len(z),'first_started_zero_based':next((x['t'] for x in z if x['started']),None),'first_motion_zero_based':next((x['t'] for x in z if x['fraction']>0),None),'first_full_zero_based':next((x['t'] for x in z if x['fraction']>=.999999),None),'max_fraction':max((x['fraction'] for x in z),default=None),'max_fraction_slew':max((abs(a['fraction']-b['fraction']) for a,b in zip(z,z[1:])),default=None),'increments_without_support':sum(not b['support'] for a,b in inc),'increments_without_posture':sum(not b['posture'] for a,b in inc),'increments_without_open_toes':sum(not b['open'] for a,b in inc),'increments_without_clearance':sum(b['clear']<.001 for a,b in inc),'knee_target_range_rad':[min(tar[15]),max(tar[15])] if tar[15] else None,'ankle_target_range_rad':[min(tar[16]),max(tar[16])] if tar[16] else None,'target_position_limits_ok':len(tar[15])==550 and len(tar[16])==550 and all(-.100001<=v<=1.550001 for v in tar[15]) and all(-.900001<=v<=.700001 for v in tar[16]),'scope':'commanded targets and accepted support state, not substep effort'}
(A/'reposition-check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
