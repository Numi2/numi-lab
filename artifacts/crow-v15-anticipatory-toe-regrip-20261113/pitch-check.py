from pathlib import Path
import gzip,json,collections
A=Path('artifacts/crow-v15-anticipatory-toe-regrip-20261113');data=collections.defaultdict(list);commands=collections.defaultdict(list);positions=collections.defaultdict(list)
with gzip.open(A/'trace.log.gz','rt') as f:
 for l in f:
  x=l.split()
  if not x:continue
  if x[0]=='pitchcontrol':data[int(x[1])].append({'t':int(x[2]),'valid':bool(int(x[3])),'support':bool(int(x[4])),'pitch':float(x[5]),'rate':float(x[6]),'ref':float(x[7]),'desired':float(x[8]),'correction':float(x[9])})
  elif x[0]=='rightHipCommand':commands[int(x[1])].append(float(x[3]))
  elif x[0]=='step':
   q=x.index('q');v=x.index('v');positions[int(x[1])].append(float(x[q+1+17]))
out={}
for e,rows in data.items():
 out[str(e)]={'sample_count':len(rows),'max_abs_correction_rad':max(abs(r['correction']) for r in rows),'max_correction_slew_rad':max(abs(a['correction']-b['correction']) for a,b in zip(rows,rows[1:])),'unsupported_nonzero_desired_count':sum(not r['support'] and abs(r['desired'])>1e-8 for r in rows),'pre_reference_nonzero_count':sum(not r['valid'] and abs(r['correction'])>1e-8 for r in rows),'hip_command_range_rad':[min(commands[e]),max(commands[e])],'hip_position_range_rad':[min(positions[e]),max(positions[e])],'position_limits_ok':all(-.850001<=x<=.850001 for x in commands[e]+positions[e]),'scope':'accepted control-state kinematics and commands; not substep actuator effort'}
(A/'pitch-check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
