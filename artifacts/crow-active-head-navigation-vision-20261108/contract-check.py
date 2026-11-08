from pathlib import Path
import json, subprocess, time
R=Path.cwd(); A=R/'artifacts/crow-active-head-navigation-vision-20261108'; B=R/'build/crow-active-head-vision-corrected-01'
log=json.loads((A/'corrected-execution.json').read_text())
base=next(x['argv'] for x in log if x['name']=='corrected-actual')
policy=B/'corrected/zero.policypack'
def setarg(cmd,name,value):
 i=cmd.index(name);cmd[i+1]=str(value)
def run(name,config,remove_capture=False):
 cmd=list(base);setarg(cmd,'--visual-observation-config',A/'fixtures'/config);setarg(cmd,'--policy-pack',policy)
 setarg(cmd,'--crow-replay-pack',B/(name+'.json'));setarg(cmd,'--capture-dir',B/(name+'-capture'))
 if remove_capture:
  for flag in ('--capture-dir','--capture-width','--capture-height','--capture-stride'):
   i=cmd.index(flag);del cmd[i:i+2]
  cmd.remove('--capture-policy-camera')
 r=subprocess.run(cmd,cwd=R,text=True,capture_output=True,timeout=180)
 (B/(name+'.stdout')).write_text(r.stdout);(B/(name+'.stderr')).write_text(r.stderr)
 return {'returncode':r.returncode,'stderr':r.stderr[-2500:]}
results={
 'old_misbound_assets':run('old-misbound','no-robot.json'),
 'shifted_sensor_with_old_policy':run('shifted-old-policy','corrected-camera-shift.json'),
 'same_sensor_without_capture':run('without-capture','corrected.json',True)}
results['checks']={
 'old_misbound_assets_rejected':results['old_misbound_assets']['returncode']!=0 and 'wrong physics body' in results['old_misbound_assets']['stderr'],
 'shifted_sensor_rejected':results['shifted_sensor_with_old_policy']['returncode']!=0 and 'incompatible_contract' in results['shifted_sensor_with_old_policy']['stderr'],
 'same_sensor_capture_optional':results['same_sensor_without_capture']['returncode']==0}
(A/'contract-result.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results['checks']))
